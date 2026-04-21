import torch
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Any

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import RayCasterCfg, patterns

# 导入世界模型
from task3_world import Task3WorldConfig, Task3WorldManager, spawn_world_assets, get_lidar_cfg

# ===================================================================
# 1. 配置类：同步最新的物理场景参数
# ===================================================================
class Task3Config:
    # --- 1. 基础仿真参数 ---
    num_envs = 64
    device = "cuda:0"
    decimation = 4         # 25Hz 控制频率
    max_episode_length = 800 
    frame_stack = 3        

    # --- 2. 机器人配置 (Jetbot) ---
    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Robots/Jetbot/jetbot.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=10.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, 
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel_joint"], # 抓取所有包含 wheel_joint 的关节
                effort_limit_sim=400.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=10.0,  
            )
        },
    )

    # --- 3. Sim2Real 与 传感器参数 ---
    action_delay_frames = 2     
    action_deadband = 0.05      
    action_ema_alpha = 0.5      
    lidar_noise_std = 0.02      
    lidar_outlier_prob = 0.01   
    # 修正雷达挂载偏移，确保高于车顶
    lidar_offset = (0.0, 0.0, 0.15) 
    
    # --- 4. 奖励函数系数 (根据新地图坐标校准) ---
    # 地形区间: 柏油起步[-5, -2.5], 冰雪[-2.5, 0], 地毯[0, 2.5], 泊车[2.5, 5]
    milestone_carpet_x = 0.0     # 进入地毯区的界限
    milestone_asphalt_x = 2.5    # 进入泊车区的界限
    
    # --- 4. 奖励函数系数 (精调版) ---
    rew_step = -0.02             # 稍微加重时间惩罚，逼迫智能体尽快行动，拒绝“原地发呆”
    rew_approach = 15.0          # 势能差奖励保持不变，这是驱动前进的核心动力
    
    # 里程奖励加大
    rew_milestone_terrain = 2.0  # (原 1.0 -> 2.0)
    rew_milestone_bump = 1.0     # (原 0.5 -> 1.0) 4个减速带总计 4.0 奖励
    
    rew_jerk = -0.05             # (原 -0.02 -> -0.05) 加大动作切换惩罚，强制让策略学到平滑的电机曲线
    rew_alignment = 0.5          # (原 0.1 -> 0.5) 在泊车区加大朝向对齐的诱惑，帮助解决最后 1 米的精度问题
    
    # 终局事件
    rew_crash = -50.0            # 撞墙直接重扣分 
    rew_success = 100.0          # 泊车成功给予巨额奖励 

# ===================================================================
# 2. 环境类：集成物理同步与里程碑逻辑
# ===================================================================
class JetbotParkingEnv(gym.Env):
    def __init__(self, cfg: Task3Config):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = cfg.device
        
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=self.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        
        # 1. 构建场景
        scene_cfg = InteractiveSceneCfg(num_envs=self.num_envs, env_spacing=15.0)
        self.world_cfg = Task3WorldConfig()
        
        spawn_world_assets(scene_cfg, self.world_cfg)
        scene_cfg.robot = cfg.robot_cfg
        
        # 直接覆盖并抬高 offset
        scene_cfg.lidar = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/chassis",
            update_period=0.0,
            offset=RayCasterCfg.OffsetCfg(pos=cfg.lidar_offset),
            ray_alignment="yaw",
            pattern_cfg=patterns.BpearlPatternCfg(),
            mesh_prim_paths=["/World"],
            max_distance=10.0,
        )
        
        self.scene = InteractiveScene(scene_cfg)
        self.world = Task3WorldManager(self.scene, self.world_cfg, self.num_envs, self.device)
        self.robot: Articulation = self.scene.articulations["robot"]
        self.lidar = self.scene.sensors["lidar"]
        
        self.sim.reset()

        # 2. 状态/动作空间定义 (LiDAR 降采样为 36 扇区)
        self.obs_dim_per_frame = 2 + 1 + 2 + 36 # [轮速(2), 距离(1), 航向sin/cos(2), 雷达(36)]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim_per_frame * cfg.frame_stack,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

        # 3. 缓冲区与状态追踪
        self.action_buffer = [torch.zeros((self.num_envs, 2), device=self.device) for _ in range(cfg.action_delay_frames + 1)]
        self.last_action = torch.zeros((self.num_envs, 2), device=self.device)
        self.obs_stack = torch.zeros((self.num_envs, cfg.frame_stack, self.obs_dim_per_frame), device=self.device)
        self.wheel_joint_ids, _ = self.robot.find_joints(".*wheel_joint")
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device)
        self.last_dist_to_goal = torch.zeros(self.num_envs, device=self.device)
        # 追踪里程碑 [已过地毯, 已过标准区]
        self.milestone_terrain_mask = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.bool)
        self.milestone_bump_mask = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.bool)

    def reset(self, env_ids=None, options=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 重置世界随机布局与机器人位姿
        self.world.reset_world(env_ids)
        
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:2] = self.world.start_pos[env_ids] + self.scene.env_origins[env_ids, 0:2]
        root_state[:, 3:7] = quat_from_euler_xyz(
            torch.zeros_like(env_ids), torch.zeros_like(env_ids), self.world.start_yaw[env_ids]
        )
        self.robot.write_root_state_to_sim(root_state, env_ids)
        
        # 清空缓冲区
        self.episode_length_buf[env_ids] = 0
        self.milestone_terrain_mask[env_ids] = False
        self.milestone_bump_mask[env_ids] = False
        self.last_action[env_ids] = 0
        for i in range(len(self.action_buffer)):
            self.action_buffer[i][env_ids] = 0
            
        self.last_dist_to_goal[env_ids] = torch.norm(self.world.goal_pos[env_ids] - self.world.start_pos[env_ids], dim=-1)
        
        # 同步张量并更新初始观测
        self.scene.update(0.0)
        obs_single = self._compute_single_frame_obs(env_ids)
        for i in range(self.cfg.frame_stack):
            self.obs_stack[env_ids, i] = obs_single

        return self.obs_stack[env_ids].view(len(env_ids), -1), {}

    def step(self, action: torch.Tensor):
        # 1. 执行器退化管道
        current_action = self.cfg.action_ema_alpha * action + (1 - self.cfg.action_ema_alpha) * self.last_action
        self.last_action = current_action.clone()
        
        self.action_buffer.append(current_action)
        delayed_action = self.action_buffer.pop(0)
        
        final_action = delayed_action.clone()
        final_action[torch.abs(final_action) < self.cfg.action_deadband] = 0.0
        wheel_velocities = final_action * 15.0 # 映射到 15 rad/s

        # 2. 物理仿真进帧 (Decimation Loop)
        for _ in range(self.cfg.decimation):
            # 只把速度写给真正的轮子关节
            self.robot.set_joint_velocity_target(wheel_velocities, joint_ids=self.wheel_joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step()
        
        # 物理步进后，按真实的流逝时间 (0.01 * 4 = 0.04s) 更新场景张量
        self.scene.update(0.04) 
        self.episode_length_buf += 1
        
        # 3. 观测采集与堆叠
        obs_single = self._compute_single_frame_obs()
        self.obs_stack = torch.roll(self.obs_stack, shifts=-1, dims=1)
        self.obs_stack[:, -1] = obs_single
        
        # 4. 奖励与终止判定
        rewards, terminated, truncated, info = self._compute_rewards_and_dones(action)
        
        # 5. 自动重置处理
        resets = terminated | truncated
        reset_env_ids = resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            info["last_observation"] = self.obs_stack[reset_env_ids].clone().view(len(reset_env_ids), -1)
            self.reset(reset_env_ids)
        
        return self.obs_stack.view(self.num_envs, -1), rewards, terminated, truncated, info

    def _compute_single_frame_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # 基础运动状态
        wheel_vel = self.robot.data.joint_vel[env_ids]
        curr_pos_local = self.robot.data.root_pos_w[env_ids, 0:2] - self.scene.env_origins[env_ids, 0:2]
        diff = self.world.goal_pos[env_ids] - curr_pos_local
        dist = torch.norm(diff, dim=-1, keepdim=True)
        
        # 航向误差计算
        goal_angle = torch.atan2(diff[:, 1], diff[:, 0])
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w[env_ids])
        rel_angle = wrap_to_pi(goal_angle - yaw)
        
        # LiDAR 处理与噪声注入
        raw_lidar = self.world.process_lidar_data(self.lidar)[env_ids]
        raw_lidar += torch.randn_like(raw_lidar) * self.cfg.lidar_noise_std
        raw_lidar[torch.rand_like(raw_lidar) < self.cfg.lidar_outlier_prob] = 10.0
        
        # 动态池化为 36 扇区
        num_rays = raw_lidar.shape[1]
        bin_size = num_rays // 36
        lidar_pooled = raw_lidar[:, :36 * bin_size].view(len(env_ids), 36, bin_size).min(dim=-1)[0]
        
        return torch.cat([wheel_vel, dist, torch.sin(rel_angle).unsqueeze(-1), torch.cos(rel_angle).unsqueeze(-1), lidar_pooled], dim=-1)

    def _compute_rewards_and_dones(self, action):
        curr_pos_local = self.robot.data.root_pos_w[:, 0:2] - self.scene.env_origins[:, 0:2]
        dist_to_goal = torch.norm(self.world.goal_pos - curr_pos_local, dim=-1)
        
        # 1. 势能奖励 (Approach)
        rew_approach = (self.last_dist_to_goal - dist_to_goal) * self.cfg.rew_approach
        self.last_dist_to_goal = dist_to_goal.clone()
        
        # 2. ✅ 地形里程碑奖励 (基于校准后的 X 坐标)
        rew_milestone = torch.zeros(self.num_envs, device=self.device)
        # 过地毯区间 (X > 0.0)
        carpet_pass = (curr_pos_local[:, 0] > self.cfg.milestone_carpet_x) & (~self.milestone_terrain_mask[:, 0])
        rew_milestone[carpet_pass] += self.cfg.rew_milestone_terrain
        self.milestone_terrain_mask[carpet_pass, 0] = True
        # 进泊车区 (X > 2.5)
        asphalt_pass = (curr_pos_local[:, 0] > self.cfg.milestone_asphalt_x) & (~self.milestone_terrain_mask[:, 1])
        rew_milestone[asphalt_pass] += self.cfg.rew_milestone_terrain
        self.milestone_terrain_mask[asphalt_pass, 1] = True
        
        # 3. 减速带里程碑 (增加 0.15m 容错确保车身后轮完全越过)
        for i in range(4):
            bump_x = self.world.speed_bumps[i].data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]
            pass_mask = (curr_pos_local[:, 0] > (bump_x + 0.15)) & (~self.milestone_bump_mask[:, i])
            rew_milestone[pass_mask] += self.cfg.rew_milestone_bump
            self.milestone_bump_mask[pass_mask, i] = True

        # 4. 泊车对齐与平顺性
        rew_jerk = torch.sum(torch.square(action - self.last_action), dim=-1) * self.cfg.rew_jerk
        rew_align = torch.zeros(self.num_envs, device=self.device)
        in_parking_zone = curr_pos_local[:, 0] > self.cfg.milestone_asphalt_x
        _, _, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        rew_align[in_parking_zone] = torch.cos(yaw[in_parking_zone] - self.world.goal_yaw[in_parking_zone]) * self.cfg.rew_alignment

        # 5. 终止条件判定
        out_of_lane = torch.abs(curr_pos_local[:, 1]) > 1.45
        success = dist_to_goal < 0.2 # 泊车精度要求
        
        # 6. 总奖励合成
        total_reward = self.cfg.rew_step + rew_approach + rew_jerk + rew_align + rew_milestone
        total_reward = torch.clamp(total_reward, -1.0, 1.0)
        total_reward[out_of_lane] += self.cfg.rew_crash
        total_reward[success] += self.cfg.rew_success
        
        terminated = out_of_lane | success
        truncated = self.episode_length_buf >= self.cfg.max_episode_length
        
        info = {
            "telemetry": {
                "dist": dist_to_goal.mean().item(),
                "x_pos": curr_pos_local[:, 0].mean().item(),
                "success": success.float().mean().item(),
                "terrain_progress": self.milestone_terrain_mask.float().sum(dim=-1).mean().item()
            }
        }
        return total_reward, terminated, truncated, info
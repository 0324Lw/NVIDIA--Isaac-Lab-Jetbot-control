import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat

# 引入我们刚才写好的世界模型
from task2_world import Task2WorldConfig, Task2WorldManager, spawn_world_assets, get_lidar_cfg

# ===================================================================
# 1. 核心参数配置类 (Config)
# ===================================================================
class Task2Config:
    # --- 环境与硬件参数 ---
    num_envs = 64
    sim_dt = 0.01             # 物理引擎底层步长 (100Hz)
    decimation = 5            # 控制截断比，RL频率 = 100Hz / 5 = 20Hz
    policy_dt = sim_dt * decimation
    max_episode_length = 3500 # 3500 步 * 0.05s = 175 秒 (足够穿越50m场地)

    # --- 动作平滑与归一化 ---
    action_ema_alpha = 0.5    # EMA低通滤波系数 (越小越平滑，延迟越大)
    max_wheel_speed = 15.0    # 左右轮最大转速 (rad/s)
    frame_stack = 3           # 状态堆叠帧数

    # --- 奖励函数超参数 (微调后) ---
    rew_step = -0.005         # 加大生存惩罚，逼迫小车拒绝原地发呆
    rew_approach_weight = 2.5 # 放大接近目标的拉力
    rew_heading_weight = 0.02 # 略微提升方向感引导
    rew_smooth = -0.05        # 动作变化惩罚
    
    # 柔性斥力场参数 (高斯衰减)
    rew_prox_weight = 0.5     # 提高危险区域的惩罚极值
    prox_sigma = 1.0          # 将雷达斥力圈扩大到约 2-3 米范围，提前发出警告
    
    # 终局大奖与极刑
    rew_collision = -1.0      # 碰撞/越界惩罚
    rew_success = 1.0         # 成功到达终点奖励
    
    # --- 判定阈值 ---
    success_threshold = 0.4   # 到达终点的判定半径 (米)
    collision_threshold = 0.25 # 最小碰撞判定半径 (米)

# ===================================================================
# 2. 交互场景配置类 (Scene)
# ===================================================================
@configclass
class Task2SceneCfg(InteractiveSceneCfg):
    num_envs: int = Task2Config.num_envs
    env_spacing: float = 60.0  # 为 50x50 地图留足空间

    # 1. 挂载 Jetbot 无人车
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Robots/Jetbot/jetbot.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=10.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
        actuators={
            "drive": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel_joint"], 
                effort_limit_sim=400.0, velocity_limit_sim=100.0,
                stiffness=0.0, damping=10.0,
            )
        }
    )

    # 2. 挂载 LiDAR 传感器 (追踪全图)
    lidar = get_lidar_cfg(prim_path="{ENV_REGEX_NS}/Robot/chassis")

# ===================================================================
# 3. 强化学习核心环境类 (Env)
# ===================================================================
class JetbotObstacleEnv:
    def __init__(self, cfg: Task2Config):
        self.cfg = cfg
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.num_envs = cfg.num_envs
        
        # 1. 初始化物理引擎
        sim_cfg = sim_utils.SimulationCfg(dt=self.cfg.sim_dt)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        sim_utils.spawn_ground_plane("/World/GroundPlane", sim_utils.GroundPlaneCfg(color=(0.8, 0.8, 0.8)))
        
        # 2. 初始化世界模型与资产
        scene_cfg = Task2SceneCfg()
        scene_cfg.num_envs = self.num_envs 
        self.world_cfg = Task2WorldConfig()
        spawn_world_assets(scene_cfg, self.world_cfg)
        
        self.scene = InteractiveScene(scene_cfg)
        self.world = Task2WorldManager(self.scene, self.world_cfg, self.num_envs, self.device)
        self.robot: Articulation = self.scene.articulations["robot"]
        self.lidar = self.scene.sensors["lidar"]
        
        self.sim.reset()
        
        # 3. 状态与动作缓存分配
        # 维度计算: 距离(1) + 角度(2) + 动作反馈(2) + 雷达池化(36) = 41维/帧
        self.obs_dim_per_frame = 41
        self.obs_buffer = torch.zeros((self.num_envs, self.cfg.frame_stack, self.obs_dim_per_frame), device=self.device)
        
        self.current_actions = torch.zeros((self.num_envs, 2), device=self.device)
        self.last_actions = torch.zeros((self.num_envs, 2), device=self.device)
        
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_distances = torch.zeros(self.num_envs, device=self.device)
        
        # 4. 全局重置初始化
        self.reset()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return self._get_stacked_obs(), {}

        # 1. 世界模型空间重组与起终点刷新
        self.world.reset_world(env_ids)
        
        # 2. 将无人车传送到对应的随机起点
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:2] = self.world.start_pos[env_ids]
        root_state[:, 2] = 0.05 # 高度
        # 赋予随机朝向
        random_yaw = torch.rand(len(env_ids), device=self.device) * 2 * math.pi
        root_state[:, 3:7] = self._yaw_to_quat(random_yaw)
        self.robot.write_root_state_to_sim(root_state, env_ids)
        
        # 清零关节速度
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        # 用统一接口同时写入
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # 3. 清理缓存池
        self.episode_length_buf[env_ids] = 0
        self.current_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        
        # 4. 强制物理推演一帧以同步雷达和坐标
        self.sim.step()
        self.scene.update(0.0)
        
        # 初始化观测值
        initial_obs, self.last_distances[env_ids] = self._compute_single_frame_obs(env_ids)
        # 用初始帧填满堆叠池
        self.obs_buffer[env_ids] = initial_obs.unsqueeze(1).expand(-1, self.cfg.frame_stack, -1)
        
        return self._get_stacked_obs(), {}

    def step(self, actions: torch.Tensor):
        self.last_actions = self.current_actions.clone()
        
        # 1. 动作平滑 (EMA 滤波)
        self.current_actions = self.cfg.action_ema_alpha * actions + (1 - self.cfg.action_ema_alpha) * self.current_actions
        
        # 映射到真实角速度
        velocity_targets = self.current_actions * self.cfg.max_wheel_speed
        self.robot.set_joint_velocity_target(velocity_targets)

        # 2. 物理与世界引擎多帧推演
        for _ in range(self.cfg.decimation):
            # 推进动态障碍物
            self.world.step_kinematic_obstacles(self.cfg.sim_dt)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.cfg.sim_dt)
            
        self.episode_length_buf += 1

        # 3. 获取新状态与计算奖励
        new_frame, current_dists = self._compute_single_frame_obs()
        self._update_obs_buffer(new_frame)
        
        rewards, dones, info = self._compute_rewards_and_dones(current_dists, new_frame)
        self.last_distances = current_dists.clone()

        timeouts = self.episode_length_buf >= self.cfg.max_episode_length

        # 4. 自动重置
        if dones.any():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            self.reset(reset_ids)

        # 把提取出来的 timeouts 传出去
        return self._get_stacked_obs(), rewards, dones, timeouts, info   

    def _compute_single_frame_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # 1. 获取小车坐标与姿态
        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]
        _, _, yaw = euler_xyz_from_quat(root_quat)
        
        # 2. 目标相对关系计算
        goal_pos = self.world.goal_pos[env_ids]
        vec_to_goal = goal_pos - root_pos[:, :2]
        dist_to_goal = torch.norm(vec_to_goal, dim=-1)
        
        target_angle = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])
        angle_err = target_angle - yaw
        # 归一化到 [-pi, pi]
        angle_err = torch.atan2(torch.sin(angle_err), torch.cos(angle_err))
        
        # 3. 雷达降维 (池化加速算法)
        # 从 360 根提取 36 根 (每10度取视野内的最小值，保留最危险的障碍物信息)
        raw_lidar = self.world.process_lidar_data(self.lidar)[env_ids]
        # view -> [N, 36, 10], 对10根射线求最小值
        pooled_lidar = raw_lidar.view(-1, 36, 10).min(dim=2)[0]
        
        # 4. 拼装单帧 (41 维)
        obs_frame = torch.cat([
            dist_to_goal.unsqueeze(-1) / 50.0, # 粗略归一化
            torch.sin(angle_err).unsqueeze(-1),
            torch.cos(angle_err).unsqueeze(-1),
            self.current_actions[env_ids],
            pooled_lidar
        ], dim=-1)
        
        return obs_frame, dist_to_goal

    def _update_obs_buffer(self, new_frame):
        # 队列向左滚动并填入最新帧
        self.obs_buffer = torch.roll(self.obs_buffer, shifts=-1, dims=1)
        self.obs_buffer[:, -1, :] = new_frame

    def _get_stacked_obs(self):
        # 展平输出 [num_envs, 123]
        return self.obs_buffer.view(self.num_envs, -1)

    def _compute_rewards_and_dones(self, current_dists, obs_frame):
        # === 连续奖励计算 ===
        # 1. 步数惩罚
        rew_step = torch.full((self.num_envs,), self.cfg.rew_step, device=self.device)
        
        # 2. 接近奖励
        rew_approach = (self.last_distances - current_dists) * self.cfg.rew_approach_weight
        
        # 3. 朝向奖励 (仅奖励正向，不惩罚背向)
        heading_cos = obs_frame[:, 2] # 观测中索引 2 为 cos(angle_err)
        rew_heading = torch.clamp(heading_cos, min=0.0) * self.cfg.rew_heading_weight
        
        # 4. 柔性斥力场 (高斯惩罚)
        min_lidar_dist = obs_frame[:, 5:].min(dim=1)[0] * 10.0 # 还原为真实米数
        rew_prox = -self.cfg.rew_prox_weight * torch.exp(- (min_lidar_dist**2) / (2 * self.cfg.prox_sigma**2))
        
        # 5. 平滑惩罚
        rew_smooth = self.cfg.rew_smooth * torch.sum((self.current_actions - self.last_actions)**2, dim=-1)
        
        # 汇总连续组件并截断在 [-0.5, 0.5]，为突发事件保留数值空间
        raw_continuous = rew_step + rew_approach + rew_heading + rew_prox + rew_smooth
        clamped_continuous = torch.clamp(raw_continuous, -0.5, 0.5)

        # === 终局判定与事件极刑 ===
        # 到达终点
        success = current_dists < self.cfg.success_threshold
        
        # 碰撞 (雷达探测小于0.25) 或越界 (超过24.5米)
        collision = min_lidar_dist < self.cfg.collision_threshold
        root_pos = self.robot.data.root_pos_w
        out_of_bounds = (torch.abs(root_pos[:, 0]) > 24.5) | (torch.abs(root_pos[:, 1]) > 24.5)
        crash = collision | out_of_bounds

        # 优先判定成功，其次判定撞毁
        final_reward = torch.where(
            success, torch.full_like(clamped_continuous, self.cfg.rew_success),
            torch.where(crash, torch.full_like(clamped_continuous, self.cfg.rew_collision), clamped_continuous)
        )
        
        # 组合终止条件
        timeout = self.episode_length_buf >= self.cfg.max_episode_length
        dones = success | crash | timeout

        info = {
            "reward_components": {
                "step": rew_step.mean().item(),
                "approach": rew_approach.mean().item(),
                "heading": rew_heading.mean().item(),
                "prox": rew_prox.mean().item(),
                "smooth": rew_smooth.mean().item(),
                "raw_total": raw_continuous.mean().item()
            },
            "success_rate": success.float().mean().item(),
            "crash_rate": crash.float().mean().item()
        }

        return final_reward, dones, info

    def _yaw_to_quat(self, yaw):
        # 将欧拉角 yaw 转换为四元数 [w, x, y, z]
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        quat = torch.zeros((yaw.shape[0], 4), device=self.device)
        quat[:, 0] = cy # w
        quat[:, 3] = sy # z
        return quat

# ===================================================================
# 4. 通用绘图类 
# ===================================================================
class PlotTraining:
    def __init__(self):
        self.rewards_history = []
        
    def update(self, reward):
        # 训练时只做极其轻量的数据追加
        self.rewards_history.append(reward)

    def save_and_show(self, save_dir="runs/task2_sb3"):
        """训练结束后调用，生成高清趋势图并保存"""
        import os
        import matplotlib.pyplot as plt
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "training_trend.png")
        
        plt.figure(figsize=(10, 5), dpi=150)
        plt.plot(self.rewards_history, color='royalblue', linewidth=2, label='Average Reward')
        plt.title("Task 2: PPO Training Reward Trend", fontsize=14)
        plt.xlabel("Update Intervals (x 2048 env-steps)", fontsize=12)
        plt.ylabel("Reward", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 1. 自动保存到本地
        plt.savefig(save_path)
        print(f"\n📊 训练趋势图已自动保存至: {save_path}")
        
        # 2. 尝试弹出显示 (如果服务器环境不支持GUI会自动跳过)
        try:
            plt.show()
        except Exception:
            pass
        finally:
            plt.close()
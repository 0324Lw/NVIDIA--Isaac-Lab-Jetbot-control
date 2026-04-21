import torch
import numpy as np
import gymnasium as gym
import math
from typing import Dict, Tuple, Any

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat, quat_from_euler_xyz
from isaaclab.utils import configclass

# 导入 Task 4 世界模型
from task4_world import Task4WorldConfig, Task4WorldManager, spawn_world_assets, get_lidar_cfg

# ===================================================================
# 0. 全局共享物理资产
# ===================================================================
shared_jetbot_actuator = ImplicitActuatorCfg(
    joint_names_expr=[".*wheel_joint"],
    effort_limit_sim=400.0,
    velocity_limit_sim=100.0,
    stiffness=0.0,
    damping=10.0,
)

shared_jetbot_spawn = sim_utils.UsdFileCfg(
    usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Robots/Jetbot/jetbot.usd",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=10.0),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
    ),
)

# ===================================================================
# 1. RL 与环境配置类 (MAPPO 核心参数)
# ===================================================================
class Task4Config:
    # --- 1. 基础仿真与并行参数 ---
    num_envs = 256
    device = "cuda:0"
    decimation = 2             # 50Hz 控制频率
    max_episode_length = 2000  # 最大步数约 40s 物理时间
    frame_stack = 3            # 3 帧堆叠
    num_agents = 4             # 4 辆异构无人车

    # --- 2. 异构动作与控制参数 ---
    # 4辆车的最大线速度 [N, S, E, W]，保证交互的多样性与时间差
    max_speeds = [1.0, 0.8, 1.2, 0.6] 
    action_ema_alpha = 0.5     # 动作平滑系数
    
    # 物理退化模拟
    lidar_noise_std = 0.02
    lidar_outlier_prob = 0.01

    # --- 3. 奖励函数系数 (Reward Shaping) ---
    # 连续型奖励 (单步将被截断在 [-1, 1] 之间)
    rew_step = -0.01           # 步数惩罚，鼓励快速完赛
    rew_approach = 2.0         # 势能差奖励
    rew_prox_scale = 0.1       # 靠近惩罚的高斯基底系数
    prox_safe_dist = 0.4       # 安全距离阈值 (小于此距离开始呈高斯级数扣分)
    prox_sigma = 0.3           # 高斯分布标准差
    rew_jerk = -0.01           # 动作突变惩罚
    
    # 事件型奖励 (脱离截断，保证强烈的终局信号)
    rew_milestone = 2.0        # 成功穿越路口中心
    rew_crash = -20.0          # 出界或物理碰撞
    rew_reach = 2.0           # 个人到达终点
    rew_team_base = 50.0       # 团队全部到达基础分
    rew_team_time_coef = 0.1   # 团队到达的时间分乘数

# ===================================================================
# 2. 带有 LiDAR 的多智能体场景配置
# ===================================================================
@configclass
class Task4SceneCfg(InteractiveSceneCfg):
    num_envs: int = Task4Config.num_envs
    env_spacing: float = 25.0

    # 4 辆小车实例化
    robot_0: ArticulationCfg = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot_0", spawn=shared_jetbot_spawn, actuators={"wheels": shared_jetbot_actuator})
    robot_1: ArticulationCfg = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot_1", spawn=shared_jetbot_spawn, actuators={"wheels": shared_jetbot_actuator})
    robot_2: ArticulationCfg = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot_2", spawn=shared_jetbot_spawn, actuators={"wheels": shared_jetbot_actuator})
    robot_3: ArticulationCfg = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot_3", spawn=shared_jetbot_spawn, actuators={"wheels": shared_jetbot_actuator})

    # 给每辆车配置独立的 LiDAR
    lidar_0 = get_lidar_cfg("{ENV_REGEX_NS}/Robot_0/chassis")
    lidar_1 = get_lidar_cfg("{ENV_REGEX_NS}/Robot_1/chassis")
    lidar_2 = get_lidar_cfg("{ENV_REGEX_NS}/Robot_2/chassis")
    lidar_3 = get_lidar_cfg("{ENV_REGEX_NS}/Robot_3/chassis")

# ===================================================================
# 3. 多智能体环境类 (遵循 CTDE 架构)
# ===================================================================
class Task4MappoEnv(gym.Env):
    def __init__(self, cfg: Task4Config):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = cfg.device
        
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=self.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        
        # 1. 场景构建
        scene_cfg = Task4SceneCfg()
        scene_cfg.num_envs = self.num_envs
        self.world_cfg = Task4WorldConfig()
        spawn_world_assets(scene_cfg, self.world_cfg)
        
        self.scene = InteractiveScene(scene_cfg)
        self.world = Task4WorldManager(self.scene, self.world_cfg, self.num_envs, self.device)
        
        # 提取 4 辆车的句柄和雷达
        self.robots = [self.scene.articulations[f"robot_{i}"] for i in range(4)]
        self.lidars = [self.scene.sensors[f"lidar_{i}"] for i in range(4)]
        self.max_speeds = torch.tensor(cfg.max_speeds, device=self.device).view(1, 4, 1) # [1, 4, 1]
        
        # 必须先重置/启动仿真引擎，让 PhysX 底层视图完成初始化
        self.sim.reset()

        # 此时物理引擎已就绪，再抓取真实轮子索引就绝对安全了
        self.wheel_ids = []
        for r in self.robots:
            w_id, _ = r.find_joints(".*wheel_joint")
            self.wheel_ids.append(w_id)

        # 2. 状态/动作空间定义
        # Actor 局部感知: 自我速度(2)+距目标(1)+目标方向(2)+队友距离(3)+队友方向(6)+雷达(36) = 50 维
        self.obs_dim_per_frame = 2 + 1 + 2 + 3 + 6 + 36 
        self.actor_obs_dim = self.obs_dim_per_frame * cfg.frame_stack
        
        # Critic 上帝视角: 4辆车*(绝对XY(2)+绝对Yaw(1)+速度(2)+终点XY(2)) = 28 维
        self.critic_state_dim = 4 * (2 + 1 + 2 + 2)
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, self.actor_obs_dim))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4, 2))

        # 3. 状态与追踪缓冲区
        self.obs_stack = torch.zeros((self.num_envs, 4, cfg.frame_stack, self.obs_dim_per_frame), device=self.device)
        self.last_action = torch.zeros((self.num_envs, 4, 2), device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device)
        
        # 里程碑与终点接管判定 [num_envs, 4]
        self.last_dist_to_goal = torch.zeros((self.num_envs, 4), device=self.device)
        self.reached_goal_mask = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)
        self.milestone_mask = torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None, options=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        self.world.reset_world(env_ids)
        
        # 分配 4 辆车的位姿
        for i in range(4):
            state = self.robots[i].data.default_root_state[env_ids].clone()
            state[:, 0:2] = self.world.start_pos[env_ids, i] + self.scene.env_origins[env_ids, 0:2]
            state[:, 2] = 0.05
            state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(env_ids), torch.zeros_like(env_ids), self.world.start_yaw[env_ids, i])
            self.robots[i].write_root_state_to_sim(state, env_ids)
            
            self.last_dist_to_goal[env_ids, i] = torch.norm(self.world.goal_pos[env_ids, i] - self.world.start_pos[env_ids, i], dim=-1)

        # 重置缓冲区
        self.episode_length_buf[env_ids] = 0
        self.last_action[env_ids] = 0
        self.reached_goal_mask[env_ids] = False
        self.milestone_mask[env_ids] = False

        self.scene.update(0.0)
        obs_single, state_global = self._compute_obs_and_state(env_ids)
        for f in range(self.cfg.frame_stack):
            self.obs_stack[env_ids, :, f, :] = obs_single

        # MAPPO 标准接口：obs 返回局部，state 放在 info 中
        return self.obs_stack[env_ids].view(len(env_ids), 4, -1), {"state": state_global}

    def step(self, action: torch.Tensor):
        old_action = self.last_action.clone()
        # 1. 动作平滑与无倒车映射
        # action shape: [num_envs, 4, 2]
        current_action = self.cfg.action_ema_alpha * action + (1 - self.cfg.action_ema_alpha) * self.last_action
        self.last_action = current_action.clone()
        
        # 映射平移指令 v 和转向指令 w
        # v_raw 从 [-1, 1] 映射到 [0, 1]，彻底封死倒车
        v_raw = (current_action[:, :, 0:1] + 1.0) / 2.0 
        w_raw = current_action[:, :, 1:2]
        
        # 异构最大速度乘法
        v_cmd = v_raw * self.max_speeds
        w_cmd = w_raw * 2.0 # 最大转向角速度 2.0 rad/s
        
        # 系统终点接管：已到达终点的车，速度强制归零锁死
        v_cmd[self.reached_goal_mask] = 0.0
        w_cmd[self.reached_goal_mask] = 0.0

        # 在循环外生成当前活跃的环境索引张量
        env_ids_tensor = torch.arange(self.num_envs, device=self.device)

        # 运动学差速逆解算并下发
        for i in range(4):
            v_i, w_i = v_cmd[:, i, 0], w_cmd[:, i, 0]
            v_left = v_i - (w_i * 0.15) / 2.0
            v_right = v_i + (w_i * 0.15) / 2.0
            w_left = v_left / 0.03
            w_right = v_right / 0.03
            joint_vels = torch.cat([w_left.unsqueeze(1), w_right.unsqueeze(1)], dim=1)
            
            # 明确传入 env_ids_tensor，拒绝底层张量发生不受控的广播
            self.robots[i].set_joint_velocity_target(
                joint_vels, 
                joint_ids=self.wheel_ids[i], 
                env_ids=env_ids_tensor
            )

        # 2. 物理仿真进帧
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            
        self.scene.update(0.01 * self.cfg.decimation)
        self.episode_length_buf += 1

        # 3. 观测与奖励
        obs_single, state_global = self._compute_obs_and_state()
        self.obs_stack = torch.roll(self.obs_stack, shifts=-1, dims=2)
        self.obs_stack[:, :, -1, :] = obs_single
        
        rewards, terminated, truncated, info = self._compute_rewards_and_dones(current_action, old_action)
        info["state"] = state_global

        # 4. 自动重置
        resets = terminated | truncated
        reset_env_ids = resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # MAPPO 要求最后一帧的全局 State 和 局部 Obs 留存
            info["terminal_observation"] = self.obs_stack[reset_env_ids].clone().view(len(reset_env_ids), 4, -1)
            info["terminal_state"] = state_global[reset_env_ids].clone()
            self.reset(reset_env_ids)

        return self.obs_stack.view(self.num_envs, 4, -1), rewards, terminated, truncated, info

    def _compute_obs_and_state(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        n_ids = len(env_ids)
        obs_list = []
        state_list = []
        
        # 获取 4 辆车的全局与局部坐标张量
        pos_all = torch.stack([self.robots[i].data.root_pos_w[env_ids, 0:2] - self.scene.env_origins[env_ids, 0:2] for i in range(4)], dim=1)
        quats = torch.stack([self.robots[i].data.root_quat_w[env_ids] for i in range(4)], dim=1)
        yaws_all = torch.zeros((n_ids, 4), device=self.device)
        for i in range(4):
            _, _, y = euler_xyz_from_quat(quats[:, i, :])
            yaws_all[:, i] = y
        vels_all = torch.stack([self.robots[i].data.root_lin_vel_w[env_ids, 0:2] for i in range(4)], dim=1)
        goals_all = self.world.goal_pos[env_ids]

        # 逐车计算 Actor 局部观测
        for i in range(4):
            pos_i = pos_all[:, i, :]
            yaw_i = yaws_all[:, i]
            
            # 1. 自身运动
            wheel_vel = self.robots[i].data.joint_vel[env_ids]
            
            # 2. 相对目标
            diff_goal = goals_all[:, i, :] - pos_i
            dist_goal = torch.norm(diff_goal, dim=-1, keepdim=True)
            angle_goal = torch.atan2(diff_goal[:, 1], diff_goal[:, 0])
            rel_yaw_goal = wrap_to_pi(angle_goal - yaw_i)
            goal_feats = torch.cat([dist_goal, torch.sin(rel_yaw_goal).unsqueeze(-1), torch.cos(rel_yaw_goal).unsqueeze(-1)], dim=-1)
            
            # 3. 队友相对感知
            teammate_dists = []
            teammate_dirs = []
            for j in range(4):
                if i == j: continue
                diff_t = pos_all[:, j, :] - pos_i
                d = torch.norm(diff_t, dim=-1, keepdim=True)
                ang_t = torch.atan2(diff_t[:, 1], diff_t[:, 0])
                rel_ang_t = wrap_to_pi(ang_t - yaw_i)
                teammate_dists.append(d)
                teammate_dirs.append(torch.sin(rel_ang_t).unsqueeze(-1))
                teammate_dirs.append(torch.cos(rel_ang_t).unsqueeze(-1))
                
            tm_feats = torch.cat(teammate_dists + teammate_dirs, dim=-1)
            
            # 4. 雷达池化
            raw_lidar = self.world.process_lidar_data(self.lidars[i], env_ids)
            raw_lidar += torch.randn_like(raw_lidar) * self.cfg.lidar_noise_std
            raw_lidar[torch.rand_like(raw_lidar) < self.cfg.lidar_outlier_prob] = 10.0
            bin_size = raw_lidar.shape[1] // 36
            lidar_pooled = raw_lidar[:, :36 * bin_size].view(n_ids, 36, bin_size).min(dim=-1)[0]
            
            # 拼接 50 维单车特征
            agent_obs = torch.cat([wheel_vel, goal_feats, tm_feats, lidar_pooled], dim=-1)
            obs_list.append(agent_obs)
            
            # 收集上帝视角 State (位置2, 朝向1, 速度2, 目标2)
            state_list.append(pos_i)
            state_list.append(yaw_i.unsqueeze(-1))
            state_list.append(vels_all[:, i, :])
            state_list.append(goals_all[:, i, :])

        obs_tensor = torch.stack(obs_list, dim=1) # [n_ids, 4, 50]
        state_tensor = torch.cat(state_list, dim=-1) # [n_ids, 28]

        return obs_tensor, state_tensor

    def _compute_rewards_and_dones(self, action,old_action):
        # 提取全车坐标
        pos_all = torch.stack([self.robots[i].data.root_pos_w[:, 0:2] - self.scene.env_origins[:, 0:2] for i in range(4)], dim=1)
        
        # 1. 连续型奖励初始化
        rew_continuous = torch.zeros((self.num_envs, 4), device=self.device)
        rew_continuous += self.cfg.rew_step
        
        # Jerk 计算 (使用真实的动作差分)
        jerk = torch.sum(torch.square(action - old_action), dim=-1)
        rew_continuous += jerk * self.cfg.rew_jerk

        # 2. 势能前进奖励与高斯靠近惩罚
        prox_penalties = torch.zeros((self.num_envs, 4), device=self.device)
        dists_to_goal = torch.zeros((self.num_envs, 4), device=self.device)
        
        for i in range(4):
            # 势能差
            dists_to_goal[:, i] = torch.norm(self.world.goal_pos[:, i, :] - pos_all[:, i, :], dim=-1)
            approach = (self.last_dist_to_goal[:, i] - dists_to_goal[:, i]) * self.cfg.rew_approach
            
            # 已经被接管的车不享受前进奖励也不扣步数时间
            active_mask = ~self.reached_goal_mask[:, i]
            rew_continuous[active_mask, i] += approach[active_mask]
            
            # 共用高斯碰撞斥力计算
            for j in range(i + 1, 4):
                dist_ij = torch.norm(pos_all[:, i, :] - pos_all[:, j, :], dim=-1)
                danger_mask = dist_ij < self.cfg.prox_safe_dist
                # 高斯公式：-exp(-d^2 / 2*sigma^2)
                penalty = -torch.exp(-(dist_ij**2) / (2 * self.cfg.prox_sigma**2)) * self.cfg.rew_prox_scale
                
                # 共享扣分
                prox_penalties[danger_mask, i] += penalty[danger_mask] / 2.0
                prox_penalties[danger_mask, j] += penalty[danger_mask] / 2.0
                
        rew_continuous += prox_penalties
        self.last_dist_to_goal = dists_to_goal.clone()
        
        # 截断连续奖励在 [-1, 1] 之间，保留梯度
        rew_total = torch.clamp(rew_continuous, -1.0, 1.0)
        
        # 3. 离散事件奖励 (事件叠加不受截断限制)
        info_metrics = {"crash": 0.0, "reach": 0.0, "team_success": 0.0, "milestone": 0.0}
        
        # A. 里程碑：离开通道，距离目标点 < 6.0m 视为突破十字路口中心
        for i in range(4):
            passed = (dists_to_goal[:, i] < 6.0) & (~self.milestone_mask[:, i])
            rew_total[passed, i] += self.cfg.rew_milestone
            self.milestone_mask[passed, i] = True
            info_metrics["milestone"] += passed.float().sum().item()

        # B. 碰撞判定 (任意一辆车出界即全盘崩溃)
        # 车道边界判定 (如果 Y>5，它必须在 X [-0.4, 0.4] 通道内)
        # 优化后的精准边缘碰撞检测 (物理墙厚度在 0.4，车辆半径约 0.1，0.32 意味着发生擦碰)
        is_crash = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(4):
            abs_x = torch.abs(pos_all[:, i, 0])
            abs_y = torch.abs(pos_all[:, i, 1])
            
            # 一个极其优雅的逻辑：只要 X 和 Y 同时大于 0.32，说明必然挤进了四个直角死区
            # 同时加上通道尽头的死区判定 (> 9.8m)
            out_of_bounds = ((abs_x > 0.32) & (abs_y > 0.32)) | (abs_x > 9.8) | (abs_y > 9.8)
            
            # 队内硬碰撞 (距离 < 0.22)
            for j in range(i + 1, 4):
                d_ij = torch.norm(pos_all[:, i, :] - pos_all[:, j, :], dim=-1)
                out_of_bounds |= (d_ij < 0.22)
            is_crash |= out_of_bounds

        # 一损俱损：只要发生 crash，全队 4 人均扣分
        rew_total[is_crash, :] += self.cfg.rew_crash
        info_metrics["crash"] = is_crash.float().mean().item()

        # C. 个人到达终点
        for i in range(4):
            reach = (dists_to_goal[:, i] < 0.3) & (~self.reached_goal_mask[:, i])
            rew_total[reach, i] += self.cfg.rew_reach
            self.reached_goal_mask[reach, i] = True
            info_metrics["reach"] += reach.float().sum().item()

        # D. 团队全体完赛
        team_success = self.reached_goal_mask.all(dim=-1)
        steps_left = self.cfg.max_episode_length - self.episode_length_buf
        team_bonus = self.cfg.rew_team_base + steps_left * self.cfg.rew_team_time_coef
        # 广播给队伍所有人
        rew_total[team_success, :] += team_bonus[team_success].unsqueeze(1)
        info_metrics["team_success"] = team_success.float().mean().item()

        # 4. 终止条件聚合
        terminated = is_crash | team_success
        truncated = self.episode_length_buf >= self.cfg.max_episode_length

        info = {
            "reward_components": {
                "approach": (approach.mean().item() * self.cfg.rew_approach), # 仅记录均值
                "prox_penalty": prox_penalties.mean().item(),
                "jerk": (jerk.mean().item() * self.cfg.rew_jerk),
            },
            "events": info_metrics,
            "telemetry": {
                "mean_dist": dists_to_goal.mean().item()
            }
        }
        
        return rew_total, terminated, truncated, info
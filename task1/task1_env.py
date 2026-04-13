from isaaclab.utils import configclass
import torch
import math
import matplotlib.pyplot as plt
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# 引入官方标准的场景管家
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# ===================================================================
# 1. Config 参数配置类
# ===================================================================
class Task1Config:
    num_envs = 64                  # 并行训练环境数量
    sim_dt = 1.0 / 144.0           # 物理仿真步长 144Hz
    decimation = 3                 # 决策间隔: 控制频率 = 144/3 = 48Hz
    policy_dt = sim_dt * decimation
    max_episode_length = 500       # 最大控制步数 (约 10.4 秒)
    
    num_waypoints = 5              # 需要连续经过的目标点数量
    waypoint_radius = 3.0          # 目标点生成的最大半径范围
    reach_threshold = 0.25         # 到达判定距离 (米)
    
    action_tau = 0.2               # 动作低通滤波平滑系数
    
# --- 奖励函数系数 ---
    rew_step_penalty = -0.002      # 让小车有时间思考，不至于急着等死
    rew_approach_weight = 1.0      # 只要它朝着目标走，单步奖励能达到 +0.01 左右，远大于惩罚
    rew_smooth_penalty = -0.01     # 早期不要过度惩罚动作突变，否则小车不敢探索
    rew_waypoint = 0.5             # 到达单个目标点的奖励
    rew_finish = 1.0               # 彻底完成大奖

# ===================================================================
# 2. 官方场景配置类
# ===================================================================
@configclass
class JetbotSceneCfg(InteractiveSceneCfg):
    num_envs: int = 64
    env_spacing: float = 4.0
    
    
    # 批量生成 64 辆小车
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

# ===================================================================
# 3. 通用绘图类
# ===================================================================
class PlotTraining:
    def __init__(self):
        self.episode_rewards = []
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.line, = self.ax.plot([], [], label='Episode Total Reward')
        self.ax.set_xlabel('Episodes')
        self.ax.set_ylabel('Reward')
        self.ax.set_title('Task 1: UGV Navigation Training Curve')
        self.ax.legend()
        self.ax.grid(True)

    def update(self, new_reward):
        self.episode_rewards.append(new_reward)
        self.line.set_data(range(len(self.episode_rewards)), self.episode_rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ===================================================================
# 4. 强化学习环境类
# ===================================================================
class JetbotNavigationEnv:
    def __init__(self, cfg: Task1Config):
        self.cfg = cfg
        
        # 1. 物理引擎初始化
        sim_cfg = sim_utils.SimulationCfg(dt=self.cfg.sim_dt)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])

        # 在场景管家接手之前，提前把物理世界的地板铺好
        sim_utils.spawn_ground_plane("/World/GroundPlane", sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)))

        # 2. 场景生成 (让底层 C++ 处理所有克隆与排列)
        scene_cfg = JetbotSceneCfg()
        scene_cfg.num_envs = self.cfg.num_envs
        self.scene = InteractiveScene(scene_cfg)
        
        # 提取环境原点和机器人控制柄
        self.env_origins = self.scene.env_origins
        self.robot = self.scene.articulations["robot"]
        
        self.sim.reset()
        
        # 3. 张量初始化
        self.device = self.sim.device
        self.num_envs = self.cfg.num_envs
        self.step_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.waypoints = torch.zeros((self.num_envs, self.cfg.num_waypoints, 2), device=self.device)
        self.current_wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.actions = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.last_distances = torch.zeros(self.num_envs, device=self.device)
        
        # 三帧堆叠缓冲: dist, sin(yaw), cos(yaw), vL, vR, aL, aR (7维)
        self.frame_dim = 7
        self.obs_buffer = torch.zeros((self.num_envs, 3, self.frame_dim), device=self.device)
        
        self.scene.update(self.cfg.sim_dt)

    def _generate_waypoints(self, env_ids):
        num_reset = len(env_ids)
        rand_offsets = (torch.rand((num_reset, self.cfg.num_waypoints, 2), device=self.device) * 2 - 1) * self.cfg.waypoint_radius
        env_bases = self.env_origins[env_ids, :2].unsqueeze(1)
        self.waypoints[env_ids] = env_bases + rand_offsets
        self.current_wp_idx[env_ids] = 0

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.env_origins[env_ids] 
        self.robot.write_root_state_to_sim(default_root_state, env_ids=env_ids)
        
        default_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        
        self._generate_waypoints(env_ids)
        self.step_counts[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.obs_buffer[env_ids] = 0.0
        
        self.sim.step()
        self.scene.update(self.cfg.sim_dt)
        
        obs = self._compute_obs()
        return obs, {}

    def _compute_obs(self):
        idx = self.current_wp_idx.unsqueeze(-1).expand(-1, 2).unsqueeze(1)
        # 防止越界 (如果已经完成所有点，停留在最后一个点)
        clamped_idx = torch.clamp(idx, max=self.cfg.num_waypoints - 1)
        current_target = torch.gather(self.waypoints, 1, clamped_idx).squeeze(1)
        
        root_pos = self.robot.data.root_pos_w[:, :2] 
        root_quat = self.robot.data.root_quat_w     
        
        yaw = torch.atan2(2.0 * (root_quat[:, 0] * root_quat[:, 3] + root_quat[:, 1] * root_quat[:, 2]),
                          1.0 - 2.0 * (root_quat[:, 2]**2 + root_quat[:, 3]**2))
        
        to_target = current_target - root_pos
        dist = torch.norm(to_target, dim=-1)
        target_angle = torch.atan2(to_target[:, 1], to_target[:, 0])
        heading_diff = target_angle - yaw
        
        joint_vel = self.robot.data.joint_vel
        
        current_frame = torch.stack([
            dist, 
            torch.sin(heading_diff), 
            torch.cos(heading_diff),
            joint_vel[:, 0], joint_vel[:, 1],
            self.actions[:, 0], self.actions[:, 1]
        ], dim=-1)
        
        self.obs_buffer = torch.roll(self.obs_buffer, shifts=-1, dims=1)
        self.obs_buffer[:, -1, :] = current_frame
        self.last_distances = dist.clone()
        
        return self.obs_buffer.view(self.num_envs, -1)

    def step(self, actions_nn: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = self.cfg.action_tau * actions_nn + (1 - self.cfg.action_tau) * self.prev_actions
        
        joint_vel_targets = self.actions * 15.0
        
        # 物理步进
        for _ in range(self.cfg.decimation):
            self.robot.set_joint_velocity_target(joint_vel_targets)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.cfg.sim_dt)
            
        self.step_counts += 1
        
        #  3. 计算观测与奖励
        pre_distances = self.last_distances.clone()
        
        obs = self._compute_obs()
        current_dist = self.obs_buffer[:, -1, 0] 
        
        # ---------------- 奖励计算 ----------------
        # 3.1 步数惩罚
        rew_step = torch.ones(self.num_envs, device=self.device) * self.cfg.rew_step_penalty
        
        # 3.2 接近奖励 (势能差)
        rew_approach = (pre_distances - current_dist) * self.cfg.rew_approach_weight
        
        action_diff = torch.sum((self.actions - self.prev_actions)**2, dim=-1)
        rew_smooth = action_diff * self.cfg.rew_smooth_penalty
        
        # 3.4 目标点达成与更新判定
        reached = current_dist < self.cfg.reach_threshold
        rew_wp = torch.zeros(self.num_envs, device=self.device)
        rew_finish = torch.zeros(self.num_envs, device=self.device)
        
        # 更新目标点索引
        self.current_wp_idx = torch.where(reached, self.current_wp_idx + 1, self.current_wp_idx)
        
        # 立刻修正到达目标点的小车的距离记忆，防止下一帧出现巨大的空间跳跃惩罚
        if reached.any():
            idx = self.current_wp_idx.unsqueeze(-1).expand(-1, 2).unsqueeze(1)
            clamped_idx = torch.clamp(idx, max=self.cfg.num_waypoints - 1)
            new_target = torch.gather(self.waypoints, 1, clamped_idx).squeeze(1)
            root_pos = self.robot.data.root_pos_w[:, :2]
            new_dist = torch.norm(new_target - root_pos, dim=-1)
            # 用到新目标的距离，覆盖掉旧距离记忆
            self.last_distances = torch.where(reached, new_dist, self.last_distances)
            
        # 发放普通点奖励
        is_normal_wp = reached & (self.current_wp_idx < self.cfg.num_waypoints)
        rew_wp[is_normal_wp] = self.cfg.rew_waypoint
        
        is_finished = self.current_wp_idx >= self.cfg.num_waypoints
        rew_finish[is_finished] = self.cfg.rew_finish
        
        total_reward = rew_step + rew_approach + rew_smooth + rew_wp + rew_finish
        clipped_reward = torch.clamp(total_reward, min=-1.0, max=1.0)
        
        terminated = is_finished
        truncated = self.step_counts >= self.cfg.max_episode_length
        dones = terminated | truncated
        
        info = {
            "reward_components": {
                "step": rew_step.mean().item(),
                "approach": rew_approach.mean().item(),
                "smooth": rew_smooth.mean().item(),
                "waypoint": rew_wp.mean().item(),
                "finish": rew_finish.mean().item(),
                "raw_total": total_reward.mean().item()
            },
            "waypoint_progress": self.current_wp_idx.clone()
        }
        
        if dones.any():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            self.reset(reset_ids)
            obs[reset_ids] = self.obs_buffer[reset_ids].view(len(reset_ids), -1)

        return obs, clipped_reward, terminated, truncated, info
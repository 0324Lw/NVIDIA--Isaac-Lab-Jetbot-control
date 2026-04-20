import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.sensors import RayCasterCfg, patterns

# ===================================================================
# 1. 世界模型参数配置类
# ===================================================================
class Task2WorldConfig:
    # --- 场地参数 ---
    arena_size = 50.0               # 场地边长 50x50m
    wall_thickness = 1.0            # 边界墙壁厚度
    wall_height = 2.0               # 边界墙壁高度
    
    # --- 障碍物参数 ---
    num_static_obs = 20             # 静态障碍物数量 (单环境)
    radius_static = [0.5, 2.0]      # 静态障碍物半径范围
    
    num_dyn_obs = 5                 # 动态障碍物数量 (单环境)
    radius_dyn = [0.5, 1.0]         # 动态障碍物半径范围
    dyn_speed = 0.5                 # 动态障碍物游走速度 (m/s)
    
    # --- 空间分布参数 ---
    min_obs_spacing = 1.5           # 障碍物中心之间的最小间距界限 (剔除半径后)
    safe_zone_radius = 2.0          # 起终点绝对安全区半径
    goal_dist_range = [40.0, 50.0]  # 起终点间距要求
    
    layout_reset_interval = 50      # 每 50 个回合重置一次世界布局

# ===================================================================
# 2. LiDAR 传感器配置模板 (供环境类挂载到机器人上)
# ===================================================================
def get_lidar_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot/chassis"):
    """配置一个 360 度，360 根射线，量程 10m 的高性能张量雷达"""
    return RayCasterCfg(
        prim_path=prim_path,
        mesh_prim_paths=["/World"],  
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            horizontal_fov_range=[-180.0, 180.0],
            horizontal_res=1.0,
            vertical_fov_range=[0.0, 0.0],
        ),
        max_distance=10.0,
        debug_vis=False             # 训练时绝对不要开启可视化，会卡死！
    )

# ===================================================================
# 3. 资产生成器 (供 SceneCfg 调用)
# ===================================================================
def spawn_world_assets(scene_cfg: InteractiveSceneCfg, cfg: Task2WorldConfig):
    """在场景管家中批量注册世界组件"""
    
    # 1. 准备实体配置
    wall_cfg = sim_utils.CuboidCfg(
        size=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2))
    )
    
    static_cfg = sim_utils.CylinderCfg(
        radius=1.0, height=2.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.8))
    )
    
    dyn_cfg = sim_utils.CylinderCfg(
        radius=1.0, height=2.0, 
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1))
    )

    # 2. 循环生成，强行塞入 /World/envs/env_0 模板
    for i in range(4):
        wall_cfg.func(f"/World/envs/env_0/Walls/wall_{i}", wall_cfg)
        
    for i in range(cfg.num_static_obs):
        static_cfg.func(f"/World/envs/env_0/Obstacles/Static_{i}", static_cfg)
        
    for i in range(cfg.num_dyn_obs):
        dyn_cfg.func(f"/World/envs/env_0/Obstacles/Dynamic_{i}", dyn_cfg)

    # 3. 将它们作为张量视图注册回场景管家 
    scene_cfg.walls = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Walls/wall_.*")
    scene_cfg.static_obs = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obstacles/Static_.*")
    scene_cfg.dyn_obs = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obstacles/Dynamic_.*")

# ===================================================================
# 4. 世界模型核心逻辑管理器
# ===================================================================
class Task2WorldManager:
    def __init__(self, scene, cfg: Task2WorldConfig, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.scene = scene
        
        # 提取引擎中的对象句柄
        self.static_obs: RigidObject = scene.rigid_objects["static_obs"]
        self.dyn_obs: RigidObject = scene.rigid_objects["dyn_obs"]
        
        # 维护动态障碍物的张量速度池 [num_envs, 5, 2]
        self.dyn_obs_vel = torch.zeros((self.num_envs, self.cfg.num_dyn_obs, 2), device=self.device)
        self.dyn_obs_pos = torch.zeros((self.num_envs, self.cfg.num_dyn_obs, 2), device=self.device)
        # 维护起终点信息
        self.start_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.goal_pos = torch.zeros((self.num_envs, 2), device=self.device)
        
        # 维护环境的布局重置倒计时
        self.layout_reset_counters = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 首次启动，全量初始化障碍物尺寸
        self._initialize_obstacle_sizes()

    def _initialize_obstacle_sizes(self):
        """引擎冷启动时，一次性分配所有障碍物的随机半径"""
        # --- 静态障碍物 ---
        # 形状: [num_envs * 20, 3] (Scale X, Y, Z)
        static_scales = torch.ones((self.num_envs * self.cfg.num_static_obs, 3), device=self.device)
        r_static = torch.empty(self.num_envs * self.cfg.num_static_obs, device=self.device).uniform_(*self.cfg.radius_static)
        static_scales[:, 0] = r_static # X缩放 (半径)
        static_scales[:, 1] = r_static # Y缩放 (半径)


    def reset_world(self, env_ids: torch.Tensor):
        """核心张量算法：重置起终点，并在需要时通过空间跃迁重置障碍物"""
        if len(env_ids) == 0: return

        # 1. 检测哪些环境需要进行深度空间重组 (障碍物瞬移)
        self.layout_reset_counters[env_ids] += 1
        needs_full_reset = env_ids[self.layout_reset_counters[env_ids] >= self.cfg.layout_reset_interval]
        
        # 2. 生成起终点 (极坐标边缘对角生成法)
        # 随机抽取一个大圆上的角度 [0, 2pi]
        theta = torch.rand(len(env_ids), device=self.device) * 2 * math.pi
        # 起点放在圆周附近 (半径 22m，刚好贴着 25m 的墙)
        self.start_pos[env_ids, 0] = 22.0 * torch.cos(theta)
        self.start_pos[env_ids, 1] = 22.0 * torch.sin(theta)
        
        # 终点放在相反方向，加入一定的扇形噪声扰动
        theta_goal = theta + math.pi + (torch.rand_like(theta) * 0.5 - 0.25)
        self.goal_pos[env_ids, 0] = 22.0 * torch.cos(theta_goal)
        self.goal_pos[env_ids, 1] = 22.0 * torch.sin(theta_goal)

        # 3. 执行深度空间重组 (泊松盘张量拒绝采样)
        if len(needs_full_reset) > 0:
            self._teleport_obstacles(needs_full_reset)
            self.layout_reset_counters[needs_full_reset] = 0

    def _teleport_obstacles(self, env_ids):
        """张量拒绝采样算法：极速生成合法坐标并瞬间传送"""
        num_reset = len(env_ids)
        total_obs = self.cfg.num_static_obs + self.cfg.num_dyn_obs
        
        # 存放生成好的坐标 [num_reset, 25, 2]
        new_obs_pos = torch.zeros((num_reset, total_obs, 2), device=self.device)
        
        # 逐个生成障碍物，确保不与前面的障碍物、起终点重叠
        for i in range(total_obs):
            valid_mask = torch.zeros(num_reset, dtype=torch.bool, device=self.device)
            
            # 张量 while 循环：只有未生成合法的环境才会继续采样
            while not valid_mask.all():
                invalid_idx = (~valid_mask).nonzero(as_tuple=True)[0]
                if len(invalid_idx) == 0: break
                
                # 在 [-23, 23] 范围内生成候选点
                candidates = (torch.rand((len(invalid_idx), 2), device=self.device) * 46.0) - 23.0
                
                # 检查与起点的距离 (> 2m)
                dist_start = torch.norm(candidates - self.start_pos[env_ids[invalid_idx]], dim=-1)
                # 检查与终点的距离 (> 2m)
                dist_goal = torch.norm(candidates - self.goal_pos[env_ids[invalid_idx]], dim=-1)
                
                is_valid = (dist_start > self.cfg.safe_zone_radius) & (dist_goal > self.cfg.safe_zone_radius)
                
                # 检查与已生成的 [0...i-1] 障碍物的距离
                if i > 0:
                    # [len(invalid), i, 2] 的距离计算
                    dist_obs = torch.norm(candidates.unsqueeze(1) - new_obs_pos[invalid_idx, :i, :], dim=-1)
                    # 距离必须大于 (最大半径+最大半径+安全间距) = (2.0 + 2.0 + 1.5) = 5.5m 以确保绝对不会卡死
                    safe_clearance = 5.5 
                    is_valid &= (dist_obs > safe_clearance).all(dim=-1)
                
                # 更新合法掩码和结果容器
                valid_this_round = invalid_idx[is_valid]
                new_obs_pos[valid_this_round, i] = candidates[is_valid]
                valid_mask[valid_this_round] = True

        # --- 将坐标推入物理底层 (瞬间传送) ---
        # 1. 静态障碍物
        static_poses = self.static_obs.data.default_root_state.clone() # 获取默认位姿缓冲
        # 重塑并修改坐标
        static_poses_view = static_poses.view(self.num_envs, self.cfg.num_static_obs, 13)
        static_poses_view[env_ids, :, :2] += new_obs_pos[:, :self.cfg.num_static_obs, :]
        self.static_obs.write_root_state_to_sim(static_poses)

        # 2. 动态障碍物
        dyn_poses = self.dyn_obs.data.default_root_state.clone()
        dyn_poses_view = dyn_poses.view(self.num_envs, self.cfg.num_dyn_obs, 13)
        self.dyn_obs_pos[env_ids] = new_obs_pos[:, self.cfg.num_static_obs:, :]
        dyn_poses_view[env_ids, :, :2] += self.dyn_obs_pos[env_ids]
        self.dyn_obs.write_root_state_to_sim(dyn_poses)

        # 为动态障碍物分配初始随机速度
        angles = torch.rand((num_reset, self.cfg.num_dyn_obs), device=self.device) * 2 * math.pi
        self.dyn_obs_vel[env_ids, :, 0] = torch.cos(angles) * self.cfg.dyn_speed
        self.dyn_obs_vel[env_ids, :, 1] = torch.sin(angles) * self.cfg.dyn_speed

    def step_kinematic_obstacles(self, dt: float):
        """幽灵游走引擎：接管物理引擎，通过张量直接推进动态障碍物"""
        self.dyn_obs_pos += self.dyn_obs_vel * dt
        
        # 边界反弹逻辑 (触碰 ±24m 墙壁则对应轴速度取反)
        bounce_x = self.dyn_obs_pos[:, :, 0].abs() > 24.0
        self.dyn_obs_vel[:, :, 0] *= torch.where(bounce_x, -1.0, 1.0)
        
        bounce_y = self.dyn_obs_pos[:, :, 1].abs() > 24.0
        self.dyn_obs_vel[:, :, 1] *= torch.where(bounce_y, -1.0, 1.0)
        
        # 极限钳制
        self.dyn_obs_pos[:, :, 0] = torch.clamp(self.dyn_obs_pos[:, :, 0], -24.0, 24.0)
        self.dyn_obs_pos[:, :, 1] = torch.clamp(self.dyn_obs_pos[:, :, 1], -24.0, 24.0)
        
        # 瞬间覆写回物理世界
        dyn_poses = self.dyn_obs.data.default_root_state.clone()
        dyn_poses_view = dyn_poses.view(self.num_envs, self.cfg.num_dyn_obs, 13)
        dyn_poses_view[:, :, :2] += self.dyn_obs_pos
        self.dyn_obs.write_root_state_to_sim(dyn_poses)

    def process_lidar_data(self, raycaster) -> torch.Tensor:
        """获取并标准化雷达点云，供强化学习观测使用"""
        # 1. 提取光线命中点坐标与传感器原点坐标
        hit_pos_w = raycaster.data.ray_hits_w
        sensor_pos_w = raycaster.data.pos_w
        
        # 自动对齐张量维度 (不同 Isaac Lab 版本可能有细微的 batch 维度差异)
        if sensor_pos_w.dim() == 2:
            sensor_pos_w = sensor_pos_w.unsqueeze(1)  # 补齐到 [num_envs, 1, 3]
        if hit_pos_w.dim() == 4:
            hit_pos_w = hit_pos_w.squeeze(1)          # 压缩到 [num_envs, 360, 3]

        # 2. 计算极坐标欧氏距离 -> 得到 [num_envs, 360]
        distances = torch.norm(hit_pos_w - sensor_pos_w, dim=-1)
        
        # 3. 截断与清洗
        # 若光线未命中任何物体射向天空，底层可能返回无穷大(inf)或空值(nan)
        # 用 nan_to_num 强行把它们转化为最大距离 10.0，防止梯度瞬间爆炸
        distances = torch.nan_to_num(distances, posinf=10.0, neginf=10.0, nan=10.0)
        distances = torch.clamp(distances, max=10.0)
        
        # 4. 归一化到 [0, 1] 之间，喂给神经网络
        normalized_lidar = distances.view(self.num_envs, 360) / 10.0
        return normalized_lidar
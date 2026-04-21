import math
import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.math import quat_from_euler_xyz

# ===================================================================
# 1. 场景配置参数 (多智能体十字路口)
# ===================================================================
class Task4WorldConfig:
    # --- 1. 场地总体尺寸 ---
    track_width = 0.8        # 通道宽度: 0.8m (刚好允许两辆 0.2m 宽的小车惊险错车)
    arm_length = 10.0        # 每个方向的通道长度: 10m (总跨度 20m)
    
    # 十字路口中心区域坐标范围 (X, Y): [-0.4, 0.4]
    center_bound = track_width / 2.0 
    
    # --- 2. 物理材质库 ---
    # 极低的摩擦力用于中心交叉口，标准摩擦力用于直道
    mat_ice = (0.1, 0.05)       # 冰雪路面 (中心区域)
    mat_asphalt = (0.8, 0.75)   # 柏油路面 (四个通道)
    
    # --- 3. 四个端点的基准位姿 (北, 南, 东, 西) ---
    # 假设 +X 为东, +Y 为北
    # 数组索引含义: 0=北(N), 1=南(S), 2=东(E), 3=西(W)
    base_pos = [
        [0.0,  9.5],   # N 端点
        [0.0, -9.5],   # S 端点
        [ 9.5, 0.0],   # E 端点
        [-9.5, 0.0]    # W 端点
    ]
    base_yaw = [
        -math.pi / 2,  # N 朝向南
         math.pi / 2,  # S 朝向北
         math.pi,      # E 朝向西
         0.0           # W 朝向东
    ]

# ===================================================================
# 2. 场景张量管理核心类 (World Manager)
# ===================================================================
class Task4WorldManager:
    def __init__(self, scene: InteractiveScene, cfg: Task4WorldConfig, num_envs: int, device: str):
        self.scene = scene
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # 1. 预分配真值坐标张量 
        # 形状为 [num_envs, 4, dim]，对应每个环境中的 4 辆车
        self.start_pos = torch.zeros((self.num_envs, 4, 2), device=self.device)
        self.start_yaw = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_pos = torch.zeros((self.num_envs, 4, 2), device=self.device)
        
        # 2. 预加载 4 个元素的全部 "无不动点排列"
        # 确保任何车辆的目标索引都不等于其初始索引 (例如北不能去北)
        # [0,1,2,3] 的无不动点排列共有 9 种
        self.valid_derangements = torch.tensor([
            [1, 0, 3, 2], [1, 2, 3, 0], [1, 3, 0, 2],
            [2, 0, 3, 1], [2, 3, 0, 1], [2, 3, 1, 0],
            [3, 0, 1, 2], [3, 2, 0, 1], [3, 2, 1, 0]
        ], dtype=torch.long, device=self.device)

        # 3. 基础坐标张量化，方便批处理
        self.base_pos_tensor = torch.tensor(self.cfg.base_pos, dtype=torch.float32, device=self.device)
        self.base_yaw_tensor = torch.tensor(self.cfg.base_yaw, dtype=torch.float32, device=self.device)

    def reset_world(self, env_ids: torch.Tensor):
        """重置指定环境的起终点分配"""
        num_resets = len(env_ids)
        if num_resets == 0:
            return
            
        # 1. 初始化起点 (固定分配: 车0=北, 车1=南, 车2=东, 车3=西)
        # 增加微小的随机扰动 (±0.05m)，防止物理引擎在完全对称状态下产生奇点
        jitter_pos = (torch.rand((num_resets, 4, 2), device=self.device) - 0.5) * 0.1
        jitter_yaw = (torch.rand((num_resets, 4), device=self.device) - 0.5) * 0.1
        
        self.start_pos[env_ids] = self.base_pos_tensor.unsqueeze(0).expand(num_resets, 4, 2) + jitter_pos
        self.start_yaw[env_ids] = self.base_yaw_tensor.unsqueeze(0).expand(num_resets, 4) + jitter_yaw

        # 2. 目标分配 (随机选取无不动点排列)
        # 为每个发生重置的环境随机抽取一种排列方案
        derangement_indices = torch.randint(0, 9, (num_resets,), device=self.device)
        target_assignments = self.valid_derangements[derangement_indices] # shape: [num_resets, 4]

        # 3. 映射目标坐标
        # 根据 target_assignments 从 base_pos_tensor 中提取对应的目标坐标
        # 使用 batched index select
        flat_targets = target_assignments.view(-1)
        selected_goals = self.base_pos_tensor[flat_targets].view(num_resets, 4, 2)
        
        # 终点位置也加入极小扰动
        goal_jitter = (torch.rand((num_resets, 4, 2), device=self.device) - 0.5) * 0.1
        self.goal_pos[env_ids] = selected_goals + goal_jitter

    def process_lidar_data(self, raycaster, env_ids) -> torch.Tensor:
        """全局激光雷达扫描处理 (自动适配单车张量扩维)"""
        hit_pos_w = raycaster.data.ray_hits_w[env_ids]
        sensor_pos_w = raycaster.data.pos_w[env_ids]

        # 如果 sensor_pos_w 是 [num_envs, 3] (2维)，则扩维成 [num_envs, 1, 3]
        if sensor_pos_w.dim() == 2:
            sensor_pos_w = sensor_pos_w.unsqueeze(1) 
            
        # 兼容处理：如果 hit_pos_w 包含了冗余的传感器维度 [num_envs, 1, num_rays, 3]
        if hit_pos_w.dim() == 4:
            hit_pos_w = hit_pos_w.squeeze(1)
            
        # 此时计算 norm 是安全的: [64, 1152, 3] - [64, 1, 3]
        distances = torch.norm(hit_pos_w - sensor_pos_w, dim=-1)
        
        # 清理由于射线未命中带来的 NaN 或 Inf，并截断至最大探测距离
        distances = torch.nan_to_num(distances, posinf=10.0, neginf=10.0, nan=10.0)
        return torch.clamp(distances, max=10.0)

# ===================================================================
# 3. 底层资产生成器 (构建十字路口物理场景)
# ===================================================================
def spawn_world_assets(scene_cfg: InteractiveSceneCfg, cfg: Task4WorldConfig):
    ground_height = 0.02
    
    def spawn_cuboid(prim_name, size_xyz, center_xyz, mat_params, color, is_wall=False):
        """通用长方体生成函数 (处理地板与空气墙)"""
        props = sim_utils.CollisionPropertiesCfg()
        if is_wall:
            rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
            vis_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=0.1)
        else:
            rigid_props = None # 静态地板不需要 RigidBody
            vis_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=color)

        cuboid_cfg = sim_utils.CuboidCfg(
            size=size_xyz,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=mat_params[0], dynamic_friction=mat_params[1]
            ),
            rigid_props=rigid_props,
            collision_props=props,
            visual_material=vis_mat
        )
        cuboid_cfg.func(f"/World/envs/env_0/{prim_name}", cuboid_cfg, translation=center_xyz)

    # -------------------------------------------------------------
    # A. 生成地板 (5 个拼图模块，表面统一 Z=0)
    # -------------------------------------------------------------
    z_floor = -ground_height / 2.0
    w = cfg.track_width
    cb = cfg.center_bound
    arm_l = cfg.arm_length - cb # 单个臂的实际延伸长度 (9.6m)
    arm_center = cb + arm_l / 2.0 # 臂中心位置 (5.2m)

    # 1. 中心路口 (正方形，冰雪材质)
    spawn_cuboid("Floor/Center_Ice", (w, w, ground_height), (0.0, 0.0, z_floor), cfg.mat_ice, (0.8, 0.9, 1.0))
    
    # 2. 四个通道 (长方形，柏油材质)
    spawn_cuboid("Floor/North_Asphalt", (w, arm_l, ground_height), (0.0, arm_center, z_floor), cfg.mat_asphalt, (0.2, 0.2, 0.2))
    spawn_cuboid("Floor/South_Asphalt", (w, arm_l, ground_height), (0.0, -arm_center, z_floor), cfg.mat_asphalt, (0.2, 0.2, 0.2))
    spawn_cuboid("Floor/East_Asphalt",  (arm_l, w, ground_height), (arm_center, 0.0, z_floor), cfg.mat_asphalt, (0.2, 0.2, 0.2))
    spawn_cuboid("Floor/West_Asphalt",  (arm_l, w, ground_height), (-arm_center, 0.0, z_floor), cfg.mat_asphalt, (0.2, 0.2, 0.2))

    # -------------------------------------------------------------
    # B. 生成空气墙 (防坠落围栏)
    # -------------------------------------------------------------
    wall_h = 1.0
    wall_t = 0.1 # 墙厚 10cm
    z_wall = wall_h / 2.0
    wall_mat = (0.5, 0.5) # 墙壁摩擦力无关紧要
    wall_col = (1.0, 0.0, 0.0)

    # N 通道两侧
    spawn_cuboid("AirWalls/N_Left",  (wall_t, arm_l, wall_h), (-cb, arm_center, z_wall), wall_mat, wall_col, True)
    spawn_cuboid("AirWalls/N_Right", (wall_t, arm_l, wall_h), ( cb, arm_center, z_wall), wall_mat, wall_col, True)
    # S 通道两侧
    spawn_cuboid("AirWalls/S_Left",  (wall_t, arm_l, wall_h), ( cb, -arm_center, z_wall), wall_mat, wall_col, True)
    spawn_cuboid("AirWalls/S_Right", (wall_t, arm_l, wall_h), (-cb, -arm_center, z_wall), wall_mat, wall_col, True)
    # E 通道两侧
    spawn_cuboid("AirWalls/E_Top",   (arm_l, wall_t, wall_h), (arm_center,  cb, z_wall), wall_mat, wall_col, True)
    spawn_cuboid("AirWalls/E_Bot",   (arm_l, wall_t, wall_h), (arm_center, -cb, z_wall), wall_mat, wall_col, True)
    # W 通道两侧
    spawn_cuboid("AirWalls/W_Top",   (arm_l, wall_t, wall_h), (-arm_center,  cb, z_wall), wall_mat, wall_col, True)
    spawn_cuboid("AirWalls/W_Bot",   (arm_l, wall_t, wall_h), (-arm_center, -cb, z_wall), wall_mat, wall_col, True)
    
    # 四个端点封口墙 (防止跑出赛道尽头)
    end_offset = cfg.arm_length + wall_t / 2.0
    spawn_cuboid("AirWalls/N_End", (w, wall_t, wall_h), (0.0,  end_offset, z_wall), wall_mat, wall_col, True)
    spawn_cuboid("AirWalls/S_End", (w, wall_t, wall_h), (0.0, -end_offset, z_wall), wall_mat, wall_col, True)
    spawn_cuboid("AirWalls/E_End", (wall_t, w, wall_h), ( end_offset, 0.0, z_wall), wall_mat, wall_col, True)
    spawn_cuboid("AirWalls/W_End", (wall_t, w, wall_h), (-end_offset, 0.0, z_wall), wall_mat, wall_col, True)

# ===================================================================
# 4. 雷达配置
# ===================================================================
def get_lidar_cfg(prim_path: str):
    return RayCasterCfg(
        prim_path=prim_path,
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)), # 保持 15cm 高度避免扫到车顶
        ray_alignment="yaw",
        pattern_cfg=patterns.BpearlPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World"],
        max_distance=10.0,
    )
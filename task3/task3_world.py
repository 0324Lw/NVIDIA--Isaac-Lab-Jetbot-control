import math
import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.math import quat_from_euler_xyz

# ===================================================================
# 1. 场景配置参数 (集中存放，预留DR领域随机化接口)
# ===================================================================
class Task3WorldConfig:
    # --- 1. 场地总体尺寸 (10m x 3m) ---
    track_length = 10.0
    track_width = 3.0
    
    # --- 2. 四段拼图地形 (X轴方向, 表面统一为 Z=0) ---
    # 柏油(起点) -> 冰雪(干扰) -> 地毯(干扰) -> 柏油(泊车)
    bound_asphalt_start = [-5.0, -2.5]
    bound_ice           = [-2.5,  0.0]
    bound_carpet        = [ 0.0,  2.5]
    bound_asphalt_park  = [ 2.5,  5.0]
    
    # 物理材质库 (静摩擦, 动摩擦)
    mat_ice = (0.1, 0.05)
    mat_carpet = (0.9, 0.85)
    mat_asphalt = (0.8, 0.75) # 提高标准路面摩擦，确保起步稳定
    
    # --- 3. 障碍物与减速带 ---
    num_speed_bumps = 4
    bump_size = (0.2, 3.0, 0.005) # 宽度0.2, 跨越赛道3.0, 高度2cm
    # 为保证不重叠，将4个减速带限制在4个独立的X轴区间内
    bump_zones = [
        [-3.5, -2.5], # 区域 1
        [-1.5, -0.5], # 区域 2
        [ 0.5,  1.5], # 区域 3
        [ 2.5,  3.5]  # 区域 4
    ]
    
    # --- 4. 泊车位参数 (拓宽以降低初期探索难度) ---
    spot_width_inner = 0.5   # 内部宽度 (原0.3 -> 0.5)
    spot_depth_inner = 0.6   # 内部深度 (原0.4 -> 0.6)
    wall_thickness = 0.05    # 墙壁厚度
    wall_height = 0.3        # 墙壁高度

# ===================================================================
# 2. 场景张量管理核心类 (World Manager)
# ===================================================================
class Task3WorldManager:
    def __init__(self, scene: InteractiveScene, cfg: Task3WorldConfig, num_envs: int, device: str):
        self.scene = scene
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # 1. 获取张量资产句柄
        self.speed_bumps = [
            self.scene.rigid_objects[f"speed_bump_{i}"] for i in range(self.cfg.num_speed_bumps)
        ]
        self.park_back: RigidObject = self.scene.rigid_objects["park_back"]
        self.park_left: RigidObject = self.scene.rigid_objects["park_left"]
        self.park_right: RigidObject = self.scene.rigid_objects["park_right"]
        
        # 2. 预分配真值坐标张量 (供强化学习 Reward / Observation 使用)
        self.start_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.start_yaw = torch.zeros((self.num_envs,), device=self.device)
        self.goal_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.goal_yaw = torch.zeros((self.num_envs,), device=self.device)

    def reset_world(self, env_ids: torch.Tensor):
        """重置指定环境的赛道与泊车位"""
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        
        env_origins = self.scene.env_origins[env_ids]

        # -------------------------------------------------------------
        # 1. 刷新无人车起点 (柏油路起点区)
        # -------------------------------------------------------------
        # 限制在 X: [-4.8, -4.0], Y: [-0.8, 0.8]
        self.start_pos[env_ids, 0] = torch.rand(num_resets, device=self.device) * 0.8 - 4.8
        self.start_pos[env_ids, 1] = torch.rand(num_resets, device=self.device) * 1.6 - 0.8
        self.start_yaw[env_ids] = (torch.rand(num_resets, device=self.device) - 0.5) * 0.2

        # -------------------------------------------------------------
        # 2. 刷新 U 型泊车位 (柏油路终点区)
        # -------------------------------------------------------------
        spot_x = torch.rand(num_resets, device=self.device) * 1.0 + 3.5
        spot_y = torch.rand(num_resets, device=self.device) * 1.6 - 0.8
        self.goal_pos[env_ids, 0] = spot_x
        self.goal_pos[env_ids, 1] = spot_y
        
        # 将 math.pi 改为 0.0，让背墙在正X轴方向，开口朝向负X轴（面向起点）
        yaw_offset = (torch.rand(num_resets, device=self.device) * 2 - 1) * (math.pi / 6)
        self.goal_yaw[env_ids] = 0.0 + yaw_offset 
        
        self._teleport_parking_spot(env_ids, spot_x, spot_y, self.goal_yaw[env_ids])

        # -------------------------------------------------------------
        # 3. 区间网格化生成减速带 (绝对防重叠)
        # -------------------------------------------------------------
        for i, bump in enumerate(self.speed_bumps):
            bump_state = bump.data.default_root_state[env_ids].clone()
            
            # 从预先分配好的独立区间中采样 X 坐标
            zone_min, zone_max = self.cfg.bump_zones[i]
            zone_length = zone_max - zone_min
            bump_x = torch.rand(num_resets, device=self.device) * zone_length + zone_min
            
            bump_state[:, 0] = bump_x + env_origins[:, 0]
            bump_state[:, 1] = 0.0 + env_origins[:, 1]
            bump_state[:, 2] = self.cfg.bump_size[2] / 2.0  # 完美贴地 (Z=0.01)
            
            # 随机偏转角 ±15度
            bump_yaw = (torch.rand(num_resets, device=self.device) * 2 - 1) * (math.pi / 12)
            bump_state[:, 3:7] = quat_from_euler_xyz(
                torch.zeros_like(bump_yaw), torch.zeros_like(bump_yaw), bump_yaw
            )
            bump_state[:, 7:] = 0.0 # 运动学刚体速度必须清零
            bump.write_root_state_to_sim(bump_state, env_ids)

    def _teleport_parking_spot(self, env_ids, spot_x, spot_y, spot_yaw):
        """矢量化计算泊车位墙壁位置并传送"""
        env_origins = self.scene.env_origins[env_ids]
        
        back_state = self.park_back.data.default_root_state[env_ids].clone()
        left_state = self.park_left.data.default_root_state[env_ids].clone()
        right_state = self.park_right.data.default_root_state[env_ids].clone()

        cos_y = torch.cos(spot_yaw)
        sin_y = torch.sin(spot_yaw)

        def get_global_pos(dx, dy):
            gx = spot_x + dx * cos_y - dy * sin_y
            gy = spot_y + dx * sin_y + dy * cos_y
            return gx + env_origins[:, 0], gy + env_origins[:, 1]

        wall_z = self.cfg.wall_height / 2.0  # 完美贴地 (Z=0.15)

        # 1. 背墙
        back_dx = self.cfg.spot_depth_inner / 2.0 + self.cfg.wall_thickness / 2.0
        back_state[:, 0], back_state[:, 1] = get_global_pos(back_dx, 0.0)
        back_state[:, 2] = wall_z
        back_state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(spot_yaw), torch.zeros_like(spot_yaw), spot_yaw)
        back_state[:, 7:] = 0.0

        # 2. 左墙
        side_dy = self.cfg.spot_width_inner / 2.0 + self.cfg.wall_thickness / 2.0
        left_state[:, 0], left_state[:, 1] = get_global_pos(0.0, side_dy)
        left_state[:, 2] = wall_z
        left_state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(spot_yaw), torch.zeros_like(spot_yaw), spot_yaw)
        left_state[:, 7:] = 0.0

        # 3. 右墙
        right_state[:, 0], right_state[:, 1] = get_global_pos(0.0, -side_dy)
        right_state[:, 2] = wall_z
        right_state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(spot_yaw), torch.zeros_like(spot_yaw), spot_yaw)
        right_state[:, 7:] = 0.0

        self.park_back.write_root_state_to_sim(back_state, env_ids)
        self.park_left.write_root_state_to_sim(left_state, env_ids)
        self.park_right.write_root_state_to_sim(right_state, env_ids)

    def process_lidar_data(self, raycaster) -> torch.Tensor:
        """全局激光雷达扫描处理"""
        hit_pos_w = raycaster.data.ray_hits_w
        sensor_pos_w = raycaster.data.pos_w
        if sensor_pos_w.dim() == 2:
            sensor_pos_w = sensor_pos_w.unsqueeze(1)
        if hit_pos_w.dim() == 4:
            hit_pos_w = hit_pos_w.squeeze(1)
            
        distances = torch.norm(hit_pos_w - sensor_pos_w, dim=-1)
        distances = torch.nan_to_num(distances, posinf=10.0, neginf=10.0, nan=10.0)
        return torch.clamp(distances, max=10.0)

# ===================================================================
# 3. 底层资产生成器 (供 SceneCfg 调用)
# ===================================================================
def spawn_world_assets(scene_cfg: InteractiveSceneCfg, cfg: Task3WorldConfig):
    ground_height = 0.02
    
    def spawn_floor_tile(prim_name, length, center_x, mat_params, color):
        floor_cfg = sim_utils.CuboidCfg(
            size=(length, cfg.track_width, ground_height),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=mat_params[0], dynamic_friction=mat_params[1]
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        )
        # 将地板中心下移，保证表面正好处于 Z=0，方便所有刚体计算
        floor_cfg.func(f"/World/envs/env_0/Floor/{prim_name}", floor_cfg, translation=(center_x, 0.0, -ground_height/2.0))

    # 生成 4 段拼图地板
    for name, bound, mat, col in [
        ("Start_Asphalt", cfg.bound_asphalt_start, cfg.mat_asphalt, (0.2, 0.2, 0.2)),
        ("Middle_Ice",    cfg.bound_ice,           cfg.mat_ice,     (0.9, 0.9, 1.0)),
        ("Middle_Carpet", cfg.bound_carpet,        cfg.mat_carpet,  (0.4, 0.4, 0.4)),
        ("Park_Asphalt",  cfg.bound_asphalt_park,  cfg.mat_asphalt, (0.2, 0.2, 0.2))
    ]:
        length = bound[1] - bound[0]
        center = bound[0] + length / 2.0
        spawn_floor_tile(name, length, center, mat, col)

    # 空气墙 (Kinematic 确保不可推动)
    air_wall_cfg = sim_utils.CuboidCfg(
        size=(cfg.track_length, 0.1, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1)
    )
    air_wall_cfg.func("/World/envs/env_0/AirWalls/Wall_North", air_wall_cfg, translation=(0.0, cfg.track_width/2.0, 0.5))
    air_wall_cfg.func("/World/envs/env_0/AirWalls/Wall_South", air_wall_cfg, translation=(0.0, -cfg.track_width/2.0, 0.5))

    # 减速带 (修改为 Kinematic 运动学刚体)
    bump_cfg = sim_utils.CuboidCfg(
        size=cfg.bump_size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.1))
    )
    for i in range(cfg.num_speed_bumps):
        bump_cfg.func(f"/World/envs/env_0/SpeedBumps/Bump_{i}", bump_cfg, translation=(0.0, 0.0, cfg.bump_size[2]/2.0))
        setattr(scene_cfg, f"speed_bump_{i}", RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/SpeedBumps/Bump_{i}"))

    # 泊车位 U 型墙 (Kinematic 刚体)
    wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2))
    park_back_cfg = sim_utils.CuboidCfg(
        size=(cfg.wall_thickness, cfg.spot_width_inner + cfg.wall_thickness*2, cfg.wall_height),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=wall_mat
    )
    park_side_cfg = sim_utils.CuboidCfg(
        size=(cfg.spot_depth_inner, cfg.wall_thickness, cfg.wall_height),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=wall_mat
    )
    
    wall_z = cfg.wall_height / 2.0
    park_back_cfg.func("/World/envs/env_0/Parking/Back", park_back_cfg, translation=(0.0, 0.0, wall_z))
    park_side_cfg.func("/World/envs/env_0/Parking/Left", park_side_cfg, translation=(0.0, 0.0, wall_z))
    park_side_cfg.func("/World/envs/env_0/Parking/Right", park_side_cfg, translation=(0.0, 0.0, wall_z))

    scene_cfg.park_back = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Parking/Back")
    scene_cfg.park_left = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Parking/Left")
    scene_cfg.park_right = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Parking/Right")

# ===================================================================
# 4. 雷达配置
# ===================================================================
def get_lidar_cfg(prim_path: str):
    return RayCasterCfg(
        prim_path=prim_path,
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
        ray_alignment="yaw",
        pattern_cfg=patterns.BpearlPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World"],
        max_distance=10.0,
    )
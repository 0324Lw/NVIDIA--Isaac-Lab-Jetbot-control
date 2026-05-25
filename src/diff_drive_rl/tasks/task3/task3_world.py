from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.math import quat_from_euler_xyz


@dataclass
class Task3WorldConfig:
    """Jetbot Task3 Conservative Sim2Real Parking World.

    Coordinate convention:
        start_pos / goal_pos / bump_pos / root_pos_local are all env-local xy.
        Isaac world xy must be env_origin.xy + local_xy.

    Conservative asset policy:
        Floors, lane walls, speed bumps and parking walls are manually spawned
        into /World/envs/env_0, then selected rigid objects are registered by
        RigidObjectCfg(prim_path="{ENV_REGEX_NS}/..."). Geometry is fixed to
        nominal dimensions. Sim2Real randomization is kept in GPU buffers.
    """

    # ------------------------------------------------------------------
    # Track geometry
    # ------------------------------------------------------------------
    track_length: float = 10.0
    track_width: float = 3.0
    ground_height: float = 0.02

    x_min: float = -5.0
    x_max: float = 5.0
    y_min: float = -1.5
    y_max: float = 1.5

    lane_wall_thickness: float = 0.10
    lane_wall_height: float = 0.60

    bound_asphalt_start: Tuple[float, float] = (-5.0, -2.5)
    bound_ice: Tuple[float, float] = (-2.5, 0.0)
    bound_carpet: Tuple[float, float] = (0.0, 2.5)
    bound_asphalt_park: Tuple[float, float] = (2.5, 5.0)

    mat_asphalt_nominal: Tuple[float, float] = (0.80, 0.75)
    mat_ice_nominal: Tuple[float, float] = (0.12, 0.08)
    mat_carpet_nominal: Tuple[float, float] = (0.95, 0.85)

    asphalt_static_friction_range: Tuple[float, float] = (0.55, 1.10)
    asphalt_dynamic_friction_range: Tuple[float, float] = (0.45, 1.00)

    ice_static_friction_range: Tuple[float, float] = (0.03, 0.25)
    ice_dynamic_friction_range: Tuple[float, float] = (0.02, 0.20)

    carpet_static_friction_range: Tuple[float, float] = (0.65, 1.35)
    carpet_dynamic_friction_range: Tuple[float, float] = (0.55, 1.20)

    # ------------------------------------------------------------------
    # Start / parking spot
    # ------------------------------------------------------------------
    start_x_range: Tuple[float, float] = (-4.80, -4.20)
    start_y_range: Tuple[float, float] = (-0.70, 0.70)
    start_yaw_range: Tuple[float, float] = (-0.15, 0.15)

    parking_x_range: Tuple[float, float] = (3.55, 4.35)
    parking_y_range: Tuple[float, float] = (-0.70, 0.70)
    parking_yaw_range: Tuple[float, float] = (-math.pi / 8.0, math.pi / 8.0)

    # Conservative version: real wall geometry is fixed to nominal.
    spot_width_inner_range: Tuple[float, float] = (0.60, 0.60)
    spot_depth_inner_range: Tuple[float, float] = (0.73, 0.73)

    wall_thickness: float = 0.05
    wall_height: float = 0.30

    # ------------------------------------------------------------------
    # Speed bumps
    # ------------------------------------------------------------------
    num_speed_bumps: int = 4

    bump_length_x_range: Tuple[float, float] = (0.45, 0.45)
    bump_width_y_range: Tuple[float, float] = (2.80, 2.80)
    bump_height_range: Tuple[float, float] = (0.006, 0.006)
    bump_yaw_range: Tuple[float, float] = (0.0, 0.0)

    # 3-section low ramp: front low section + top section + rear low section.
    bump_ramp_segments: int = 3
    bump_top_length: float = 0.13
    bump_low_height_ratio: float = 0.35

    bump_static_friction: float = 1.10
    bump_dynamic_friction: float = 0.95

    bump_zones: Tuple[Tuple[float, float], ...] = (
        (-3.60, -2.70),
        (-1.60, -0.55),
        (0.55, 1.60),
        (2.65, 3.45),
    )

    # ------------------------------------------------------------------
    # Vehicle / event geometry
    # ------------------------------------------------------------------
    robot_radius: float = 0.18
    wall_collision_margin: float = 0.02
    lane_collision_margin: float = 0.02

    success_pos_tol: float = 0.22
    success_yaw_tol: float = math.radians(15.0)
    success_lin_vel_tol: float = 0.10
    success_ang_vel_tol: float = 0.18
    success_hold_steps: int = 12

    # ------------------------------------------------------------------
    # Sim2Real domain randomization buffers
    # ------------------------------------------------------------------
    action_delay_frame_range: Tuple[int, int] = (0, 4)
    action_deadband_range: Tuple[float, float] = (0.02, 0.08)
    action_ema_alpha_range: Tuple[float, float] = (0.35, 0.75)

    motor_strength_range: Tuple[float, float] = (0.80, 1.20)
    motor_bias_range: Tuple[float, float] = (-0.05, 0.05)
    wheel_radius_scale_range: Tuple[float, float] = (0.92, 1.08)

    lidar_noise_std_range: Tuple[float, float] = (0.005, 0.050)
    lidar_outlier_prob_range: Tuple[float, float] = (0.000, 0.030)
    lidar_dropout_prob_range: Tuple[float, float] = (0.000, 0.020)
    lidar_yaw_offset_range: Tuple[float, float] = (-math.radians(2.0), math.radians(2.0))
    lidar_z_offset_range: Tuple[float, float] = (0.10, 0.20)

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------
    lidar_max_distance: float = 10.0
    lidar_min_distance: float = 0.02
    lidar_pool_bins: int = 36
    lidar_default_offset: Tuple[float, float, float] = (0.0, 0.0, 0.15)

    front_angle_deg: float = 35.0
    side_angle_deg: float = 110.0
    risk_distance: float = 0.85

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    track_x_norm: float = 5.0
    track_y_norm: float = 1.5
    goal_xy_norm: float = 5.0
    goal_dist_norm: float = 10.0
    yaw_norm: float = math.pi

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    debug_assert: bool = False

    @property
    def half_width(self) -> float:
        return float(self.track_width) * 0.5

    @property
    def terrain_names(self) -> Tuple[str, str, str, str]:
        return ("asphalt_start", "ice", "carpet", "asphalt_park")

    @property
    def spot_width_inner_nominal(self) -> float:
        return 0.5 * (float(self.spot_width_inner_range[0]) + float(self.spot_width_inner_range[1]))

    @property
    def spot_depth_inner_nominal(self) -> float:
        return 0.5 * (float(self.spot_depth_inner_range[0]) + float(self.spot_depth_inner_range[1]))

    @property
    def bump_length_x_nominal(self) -> float:
        return 0.5 * (float(self.bump_length_x_range[0]) + float(self.bump_length_x_range[1]))

    @property
    def bump_width_y_nominal(self) -> float:
        return 0.5 * (float(self.bump_width_y_range[0]) + float(self.bump_width_y_range[1]))

    @property
    def bump_height_nominal(self) -> float:
        return 0.5 * (float(self.bump_height_range[0]) + float(self.bump_height_range[1]))

    @property
    def parking_x_nominal(self) -> float:
        return 0.5 * (float(self.parking_x_range[0]) + float(self.parking_x_range[1]))

    @property
    def parking_y_nominal(self) -> float:
        return 0.0

    @property
    def parking_yaw_nominal(self) -> float:
        return 0.0

    def validate(self) -> None:
        assert self.track_length > 0.0
        assert self.track_width > 0.0
        assert self.ground_height > 0.0
        assert self.x_min < self.x_max
        assert self.y_min < self.y_max
        assert abs(self.x_max - self.x_min - self.track_length) < 1e-5
        assert abs(self.y_max - self.y_min - self.track_width) < 1e-5

        assert self.lane_wall_thickness > 0.0
        assert self.lane_wall_height > 0.0

        bounds = [
            self.bound_asphalt_start,
            self.bound_ice,
            self.bound_carpet,
            self.bound_asphalt_park,
        ]
        assert bounds[0][0] == self.x_min
        assert bounds[-1][1] == self.x_max
        for lo, hi in bounds:
            assert lo < hi
        for i in range(len(bounds) - 1):
            assert abs(float(bounds[i][1]) - float(bounds[i + 1][0])) < 1e-6

        for rng in [
            self.asphalt_static_friction_range,
            self.asphalt_dynamic_friction_range,
            self.ice_static_friction_range,
            self.ice_dynamic_friction_range,
            self.carpet_static_friction_range,
            self.carpet_dynamic_friction_range,
            self.start_x_range,
            self.start_y_range,
            self.start_yaw_range,
            self.parking_x_range,
            self.parking_y_range,
            self.parking_yaw_range,
            self.spot_width_inner_range,
            self.spot_depth_inner_range,
            self.bump_length_x_range,
            self.bump_width_y_range,
            self.bump_height_range,
            self.bump_yaw_range,
            self.action_deadband_range,
            self.action_ema_alpha_range,
            self.motor_strength_range,
            self.motor_bias_range,
            self.wheel_radius_scale_range,
            self.lidar_noise_std_range,
            self.lidar_outlier_prob_range,
            self.lidar_dropout_prob_range,
            self.lidar_yaw_offset_range,
            self.lidar_z_offset_range,
        ]:
            assert float(rng[1]) >= float(rng[0]), f"invalid range: {rng}"

        assert self.num_speed_bumps > 0
        assert len(self.bump_zones) == self.num_speed_bumps
        assert self.bump_ramp_segments == 3
        assert self.bump_top_length > 0.0
        assert 0.0 < self.bump_low_height_ratio <= 1.0

        for lo, hi in self.bump_zones:
            assert self.x_min <= lo < hi <= self.x_max

        assert self.wall_thickness > 0.0
        assert self.wall_height > 0.0
        assert self.robot_radius > 0.0
        assert self.wall_collision_margin >= 0.0
        assert self.lane_collision_margin >= 0.0

        assert self.success_pos_tol > 0.0
        assert self.success_yaw_tol > 0.0
        assert self.success_lin_vel_tol >= 0.0
        assert self.success_ang_vel_tol >= 0.0
        assert self.success_hold_steps >= 1

        assert self.action_delay_frame_range[1] >= self.action_delay_frame_range[0] >= 0

        assert self.lidar_max_distance > self.lidar_min_distance > 0.0
        assert self.lidar_pool_bins >= 8
        assert len(self.lidar_default_offset) == 3

        assert 0.0 < self.front_angle_deg < self.side_angle_deg <= 180.0
        assert self.risk_distance > 0.0

        assert self.track_x_norm > 0.0
        assert self.track_y_norm > 0.0
        assert self.goal_xy_norm > 0.0
        assert self.goal_dist_norm > 0.0
        assert self.yaw_norm > 0.0


class Task3WorldManager:
    """Conservative Sim2Real parking world tensor manager."""

    def __init__(
        self,
        scene: InteractiveScene,
        cfg: Task3WorldConfig,
        num_envs: int,
        device: str,
    ):
        cfg.validate()

        self.scene = scene
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = str(device)

        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")

        # ------------------------------------------------------------------
        # Rigid object handles
        # ------------------------------------------------------------------
        self.speed_bumps = [
            self.scene.rigid_objects[f"speed_bump_{i}"]
            for i in range(int(self.cfg.num_speed_bumps))
        ]

        self.park_back: RigidObject = self.scene.rigid_objects["park_back"]
        self.park_left: RigidObject = self.scene.rigid_objects["park_left"]
        self.park_right: RigidObject = self.scene.rigid_objects["park_right"]

        # ------------------------------------------------------------------
        # Task geometry tensors
        # ------------------------------------------------------------------
        self.start_pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.start_yaw = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        self.goal_pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.goal_yaw = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        self.spot_width_inner = torch.full(
            (self.num_envs,),
            float(self.cfg.spot_width_inner_nominal),
            dtype=torch.float32,
            device=self.device,
        )
        self.spot_depth_inner = torch.full(
            (self.num_envs,),
            float(self.cfg.spot_depth_inner_nominal),
            dtype=torch.float32,
            device=self.device,
        )

        self.bump_pos = torch.zeros(
            (self.num_envs, self.cfg.num_speed_bumps, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.bump_yaw = torch.zeros(
            (self.num_envs, self.cfg.num_speed_bumps),
            dtype=torch.float32,
            device=self.device,
        )
        self.bump_length_x = torch.full(
            (self.num_envs, self.cfg.num_speed_bumps),
            float(self.cfg.bump_length_x_nominal),
            dtype=torch.float32,
            device=self.device,
        )
        self.bump_width_y = torch.full(
            (self.num_envs, self.cfg.num_speed_bumps),
            float(self.cfg.bump_width_y_nominal),
            dtype=torch.float32,
            device=self.device,
        )
        self.bump_height = torch.full(
            (self.num_envs, self.cfg.num_speed_bumps),
            float(self.cfg.bump_height_nominal),
            dtype=torch.float32,
            device=self.device,
        )

        # ------------------------------------------------------------------
        # Sim2Real randomized parameter buffers
        # ------------------------------------------------------------------
        self.terrain_static_friction = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.terrain_dynamic_friction = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

        self.action_delay_frames = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.action_deadband = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.action_ema_alpha = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        self.motor_strength = torch.ones((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.motor_bias = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.wheel_radius_scale = torch.ones((self.num_envs, 2), dtype=torch.float32, device=self.device)

        self.lidar_noise_std = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.lidar_outlier_prob = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.lidar_dropout_prob = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.lidar_yaw_offset = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.lidar_z_offset = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        # ------------------------------------------------------------------
        # LiDAR / risk history
        # ------------------------------------------------------------------
        self.pooled_lidar_angles = torch.linspace(
            -math.pi,
            math.pi,
            int(self.cfg.lidar_pool_bins) + 1,
            dtype=torch.float32,
            device=self.device,
        )[:-1]

        self.prev_lidar = torch.full(
            (self.num_envs, int(self.cfg.lidar_pool_bins)),
            float(self.cfg.lidar_max_distance),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_lidar = self.prev_lidar.clone()
        self.last_lidar_delta = torch.zeros_like(self.prev_lidar)
        self.last_risk_features = torch.zeros(
            (self.num_envs, self.risk_feature_dim()),
            dtype=torch.float32,
            device=self.device,
        )

        self.reset_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    @staticmethod
    def risk_feature_dim() -> int:
        return 10

    @staticmethod
    def privileged_feature_dim() -> int:
        # terrain one-hot 4
        # current static/dynamic friction 2
        # sim2real params 14
        # parking params/errors 8
        # bump x/height 8
        # local x/y 2
        return 38

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_world(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        self.reset_counter[env_ids] += 1

        self._sample_domain_randomization(env_ids)
        self._sample_start_pose(env_ids)
        self._sample_parking_spot(env_ids)
        self._sample_speed_bumps(env_ids)

        self.prev_lidar[env_ids] = float(self.cfg.lidar_max_distance)
        self.last_lidar[env_ids] = float(self.cfg.lidar_max_distance)
        self.last_lidar_delta[env_ids] = 0.0
        self.last_risk_features[env_ids] = 0.0

        if bool(self.cfg.debug_assert):
            self._debug_validate_reset(env_ids)

    def _sample_domain_randomization(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())

        asphalt_static = self._uniform(self.cfg.asphalt_static_friction_range, (n,))
        asphalt_dynamic = self._uniform(self.cfg.asphalt_dynamic_friction_range, (n,))
        ice_static = self._uniform(self.cfg.ice_static_friction_range, (n,))
        ice_dynamic = self._uniform(self.cfg.ice_dynamic_friction_range, (n,))
        carpet_static = self._uniform(self.cfg.carpet_static_friction_range, (n,))
        carpet_dynamic = self._uniform(self.cfg.carpet_dynamic_friction_range, (n,))
        park_static = self._uniform(self.cfg.asphalt_static_friction_range, (n,))
        park_dynamic = self._uniform(self.cfg.asphalt_dynamic_friction_range, (n,))

        self.terrain_static_friction[env_ids, 0] = asphalt_static
        self.terrain_static_friction[env_ids, 1] = ice_static
        self.terrain_static_friction[env_ids, 2] = carpet_static
        self.terrain_static_friction[env_ids, 3] = park_static

        self.terrain_dynamic_friction[env_ids, 0] = asphalt_dynamic
        self.terrain_dynamic_friction[env_ids, 1] = ice_dynamic
        self.terrain_dynamic_friction[env_ids, 2] = carpet_dynamic
        self.terrain_dynamic_friction[env_ids, 3] = park_dynamic

        dmin, dmax = self.cfg.action_delay_frame_range
        self.action_delay_frames[env_ids] = torch.randint(
            low=int(dmin),
            high=int(dmax) + 1,
            size=(n,),
            dtype=torch.long,
            device=self.device,
        )

        self.action_deadband[env_ids] = self._uniform(self.cfg.action_deadband_range, (n,))
        self.action_ema_alpha[env_ids] = self._uniform(self.cfg.action_ema_alpha_range, (n,))

        self.motor_strength[env_ids] = self._uniform(self.cfg.motor_strength_range, (n, 2))
        self.motor_bias[env_ids] = self._uniform(self.cfg.motor_bias_range, (n, 2))
        self.wheel_radius_scale[env_ids] = self._uniform(self.cfg.wheel_radius_scale_range, (n, 2))

        self.lidar_noise_std[env_ids] = self._uniform(self.cfg.lidar_noise_std_range, (n,))
        self.lidar_outlier_prob[env_ids] = self._uniform(self.cfg.lidar_outlier_prob_range, (n,))
        self.lidar_dropout_prob[env_ids] = self._uniform(self.cfg.lidar_dropout_prob_range, (n,))
        self.lidar_yaw_offset[env_ids] = self._uniform(self.cfg.lidar_yaw_offset_range, (n,))
        self.lidar_z_offset[env_ids] = self._uniform(self.cfg.lidar_z_offset_range, (n,))

    def _sample_start_pose(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())
        self.start_pos[env_ids, 0] = self._uniform(self.cfg.start_x_range, (n,))
        self.start_pos[env_ids, 1] = self._uniform(self.cfg.start_y_range, (n,))
        self.start_yaw[env_ids] = self._uniform(self.cfg.start_yaw_range, (n,))

    def _sample_parking_spot(self, env_ids: torch.Tensor) -> None:
        """Conservative fixed real wall geometry.

        Parking-wall rigid objects are fixed in Isaac. Therefore tensor target is
        fixed to nominal position, so reward/event geometry matches colliders.
        """
        self.goal_pos[env_ids, 0] = float(self.cfg.parking_x_nominal)
        self.goal_pos[env_ids, 1] = float(self.cfg.parking_y_nominal)
        self.goal_yaw[env_ids] = float(self.cfg.parking_yaw_nominal)

        self.spot_width_inner[env_ids] = float(self.cfg.spot_width_inner_nominal)
        self.spot_depth_inner[env_ids] = float(self.cfg.spot_depth_inner_nominal)

    def _sample_speed_bumps(self, env_ids: torch.Tensor) -> None:
        nominal_lx = float(self.cfg.bump_length_x_nominal)
        nominal_wy = float(self.cfg.bump_width_y_nominal)
        nominal_h = float(self.cfg.bump_height_nominal)

        for i in range(int(self.cfg.num_speed_bumps)):
            zone_min, zone_max = self.cfg.bump_zones[i]
            fixed_x = 0.5 * (float(zone_min) + float(zone_max))

            self.bump_pos[env_ids, i, 0] = fixed_x
            self.bump_pos[env_ids, i, 1] = 0.0
            self.bump_yaw[env_ids, i] = 0.0
            self.bump_length_x[env_ids, i] = nominal_lx
            self.bump_width_y[env_ids, i] = nominal_wy
            self.bump_height[env_ids, i] = nominal_h

    def _debug_validate_reset(self, env_ids: torch.Tensor) -> None:
        tensors = [
            self.start_pos[env_ids],
            self.start_yaw[env_ids],
            self.goal_pos[env_ids],
            self.goal_yaw[env_ids],
            self.bump_pos[env_ids],
            self.bump_yaw[env_ids],
            self.terrain_static_friction[env_ids],
            self.terrain_dynamic_friction[env_ids],
            self.motor_strength[env_ids],
            self.motor_bias[env_ids],
            self.wheel_radius_scale[env_ids],
        ]
        for x in tensors:
            assert torch.isfinite(x).all().item(), "Task3World reset generated NaN/Inf"

    # ------------------------------------------------------------------
    # Optional parking spot teleport, kept for future non-conservative mode.
    # Conservative mode does not call this in reset_world.
    # ------------------------------------------------------------------
    def _teleport_parking_spot(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        env_origins = self.scene.env_origins[env_ids]

        spot_x = self.goal_pos[env_ids, 0]
        spot_y = self.goal_pos[env_ids, 1]
        spot_yaw = self.goal_yaw[env_ids]

        width = self.spot_width_inner[env_ids]
        depth = self.spot_depth_inner[env_ids]

        cos_y = torch.cos(spot_yaw)
        sin_y = torch.sin(spot_yaw)

        def local_to_world(dx: torch.Tensor, dy: torch.Tensor):
            gx = spot_x + dx * cos_y - dy * sin_y
            gy = spot_y + dx * sin_y + dy * cos_y
            return gx + env_origins[:, 0], gy + env_origins[:, 1]

        wall_z = float(self.cfg.wall_height) * 0.5
        zeros = torch.zeros_like(spot_yaw)

        back_dx = depth * 0.5 + float(self.cfg.wall_thickness) * 0.5
        side_dy = width * 0.5 + float(self.cfg.wall_thickness) * 0.5

        back_state = self.park_back.data.default_root_state[env_ids].clone()
        left_state = self.park_left.data.default_root_state[env_ids].clone()
        right_state = self.park_right.data.default_root_state[env_ids].clone()

        back_state[:, 0], back_state[:, 1] = local_to_world(back_dx, torch.zeros_like(back_dx))
        back_state[:, 2] = env_origins[:, 2] + wall_z
        back_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, spot_yaw)
        back_state[:, 7:13] = 0.0

        left_state[:, 0], left_state[:, 1] = local_to_world(torch.zeros_like(back_dx), side_dy)
        left_state[:, 2] = env_origins[:, 2] + wall_z
        left_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, spot_yaw)
        left_state[:, 7:13] = 0.0

        right_state[:, 0], right_state[:, 1] = local_to_world(torch.zeros_like(back_dx), -side_dy)
        right_state[:, 2] = env_origins[:, 2] + wall_z
        right_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, spot_yaw)
        right_state[:, 7:13] = 0.0

        self.park_back.write_root_state_to_sim(back_state, env_ids=env_ids)
        self.park_left.write_root_state_to_sim(left_state, env_ids=env_ids)
        self.park_right.write_root_state_to_sim(right_state, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Goal / parking geometry
    # ------------------------------------------------------------------
    def compute_goal_terms(self, root_pos_local: torch.Tensor, yaw: torch.Tensor) -> Dict[str, torch.Tensor]:
        vec_w = self.goal_pos - root_pos_local[:, :2]
        dist = torch.norm(vec_w, dim=-1)

        goal_angle = torch.atan2(vec_w[:, 1], vec_w[:, 0])
        heading_error = self.wrap_to_pi(goal_angle - yaw)
        vec_b = self.rotate_world_to_body_2d(vec_w, yaw)

        yaw_error = self.wrap_to_pi(self.goal_yaw - yaw)
        parking_frame = self.world_to_parking_frame(root_pos_local[:, :2])

        return {
            "goal_vec_w": vec_w,
            "goal_vec_b": vec_b,
            "goal_dist": dist,
            "heading_error": heading_error,
            "heading_sin": torch.sin(heading_error),
            "heading_cos": torch.cos(heading_error),
            "goal_yaw_error": yaw_error,
            "goal_yaw_sin": torch.sin(yaw_error),
            "goal_yaw_cos": torch.cos(yaw_error),
            "goal_x_body_norm": torch.clamp(vec_b[:, 0] / float(self.cfg.goal_xy_norm), -5.0, 5.0),
            "goal_y_body_norm": torch.clamp(vec_b[:, 1] / float(self.cfg.goal_xy_norm), -5.0, 5.0),
            "goal_dist_norm": torch.clamp(dist / float(self.cfg.goal_dist_norm), 0.0, 5.0),
            "parking_x": parking_frame[:, 0],
            "parking_y": parking_frame[:, 1],
            "parking_x_norm": torch.clamp(
                parking_frame[:, 0] / torch.clamp(self.spot_depth_inner, min=1e-6),
                -5.0,
                5.0,
            ),
            "parking_y_norm": torch.clamp(
                parking_frame[:, 1] / torch.clamp(self.spot_width_inner, min=1e-6),
                -5.0,
                5.0,
            ),
        }

    def world_to_parking_frame(self, pos_local_xy: torch.Tensor) -> torch.Tensor:
        rel = pos_local_xy[:, :2] - self.goal_pos
        return self.rotate_world_to_body_2d(rel, self.goal_yaw)

    def is_inside_parking_box(self, root_pos_local: torch.Tensor) -> torch.Tensor:
        p = self.world_to_parking_frame(root_pos_local[:, :2])
        inside_x = torch.abs(p[:, 0]) < self.spot_depth_inner * 0.5
        inside_y = torch.abs(p[:, 1]) < self.spot_width_inner * 0.5
        return inside_x & inside_y

    # ------------------------------------------------------------------
    # Events / collision
    # ------------------------------------------------------------------
    def check_events(
        self,
        root_pos_local: torch.Tensor,
        yaw: torch.Tensor,
        body_lin_vel: Optional[torch.Tensor] = None,
        body_ang_vel: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        goal = self.compute_goal_terms(root_pos_local, yaw)

        goal_dist = goal["goal_dist"]
        yaw_error_abs = torch.abs(goal["goal_yaw_error"])
        inside_box = self.is_inside_parking_box(root_pos_local)

        if body_lin_vel is None:
            body_speed = torch.zeros((root_pos_local.shape[0],), dtype=torch.float32, device=self.device)
        else:
            body_speed = torch.norm(body_lin_vel[:, :2], dim=-1)

        if body_ang_vel is None:
            yaw_rate_abs = torch.zeros((root_pos_local.shape[0],), dtype=torch.float32, device=self.device)
        else:
            yaw_rate_abs = torch.abs(body_ang_vel[:, 2])

        success_candidate = (
            (goal_dist < float(self.cfg.success_pos_tol))
            & (yaw_error_abs < float(self.cfg.success_yaw_tol))
            & (body_speed < float(self.cfg.success_lin_vel_tol))
            & (yaw_rate_abs < float(self.cfg.success_ang_vel_tol))
            & inside_box
        )

        out_of_lane = (
            (root_pos_local[:, 0] < float(self.cfg.x_min) + float(self.cfg.robot_radius))
            | (root_pos_local[:, 0] > float(self.cfg.x_max) - float(self.cfg.robot_radius))
            | (root_pos_local[:, 1] < float(self.cfg.y_min) + float(self.cfg.robot_radius) + float(self.cfg.lane_collision_margin))
            | (root_pos_local[:, 1] > float(self.cfg.y_max) - float(self.cfg.robot_radius) - float(self.cfg.lane_collision_margin))
        )

        parking_collision = self.check_parking_wall_collision(root_pos_local)
        bump_overlap = self.check_bump_overlap(root_pos_local)
        crash = out_of_lane | parking_collision

        return {
            "success_candidate": success_candidate,
            "out_of_lane": out_of_lane,
            "parking_wall_collision": parking_collision,
            "bump_overlap": bump_overlap,
            "crash": crash,
            "inside_parking_box": inside_box,
            "goal_dist": goal_dist,
            "goal_yaw_error_abs": yaw_error_abs,
            "body_speed": body_speed,
            "yaw_rate_abs": yaw_rate_abs,
            "parking_x": goal["parking_x"],
            "parking_y": goal["parking_y"],
            "lane_margin": self.lane_margin(root_pos_local),
            "wall_signed_distance": self.parking_wall_signed_distance(root_pos_local),
        }

    def check_parking_wall_collision(self, root_pos_local: torch.Tensor) -> torch.Tensor:
        signed = self.parking_wall_signed_distance(root_pos_local)
        return signed < float(self.cfg.robot_radius) + float(self.cfg.wall_collision_margin)

    def parking_wall_signed_distance(self, root_pos_local: torch.Tensor) -> torch.Tensor:
        p = self.world_to_parking_frame(root_pos_local[:, :2])
        px, py = p[:, 0], p[:, 1]

        width = self.spot_width_inner
        depth = self.spot_depth_inner
        t = float(self.cfg.wall_thickness)

        back_cx = depth * 0.5 + t * 0.5
        back_cy = torch.zeros_like(back_cx)
        back_hx = torch.full_like(back_cx, t * 0.5)
        back_hy = width * 0.5 + t

        left_cx = torch.zeros_like(back_cx)
        left_cy = width * 0.5 + t * 0.5
        left_hx = depth * 0.5
        left_hy = torch.full_like(back_cx, t * 0.5)

        right_cx = torch.zeros_like(back_cx)
        right_cy = -width * 0.5 - t * 0.5
        right_hx = depth * 0.5
        right_hy = torch.full_like(back_cx, t * 0.5)

        d_back = self._point_rect_signed_distance(px, py, back_cx, back_cy, back_hx, back_hy)
        d_left = self._point_rect_signed_distance(px, py, left_cx, left_cy, left_hx, left_hy)
        d_right = self._point_rect_signed_distance(px, py, right_cx, right_cy, right_hx, right_hy)

        return torch.minimum(d_back, torch.minimum(d_left, d_right))

    def check_bump_overlap(self, root_pos_local: torch.Tensor) -> torch.Tensor:
        p = root_pos_local[:, None, :2] - self.bump_pos
        yaw = self.bump_yaw

        c = torch.cos(-yaw)
        s = torch.sin(-yaw)

        bx = c * p[:, :, 0] - s * p[:, :, 1]
        by = s * p[:, :, 0] + c * p[:, :, 1]

        hx = self.bump_length_x * 0.5
        hy = self.bump_width_y * 0.5

        d = self._point_rect_signed_distance(
            bx,
            by,
            torch.zeros_like(bx),
            torch.zeros_like(by),
            hx,
            hy,
        )

        return (d < float(self.cfg.robot_radius)).any(dim=-1)

    def lane_margin(self, root_pos_local: torch.Tensor) -> torch.Tensor:
        margin_x = torch.minimum(
            root_pos_local[:, 0] - float(self.cfg.x_min),
            float(self.cfg.x_max) - root_pos_local[:, 0],
        )
        margin_y = torch.minimum(
            root_pos_local[:, 1] - float(self.cfg.y_min),
            float(self.cfg.y_max) - root_pos_local[:, 1],
        )
        return torch.minimum(margin_x, margin_y)

    # ------------------------------------------------------------------
    # Terrain / milestones
    # ------------------------------------------------------------------
    def terrain_id(self, x_local: torch.Tensor) -> torch.Tensor:
        tid = torch.zeros_like(x_local, dtype=torch.long)

        tid = torch.where(
            (x_local >= float(self.cfg.bound_ice[0])) & (x_local < float(self.cfg.bound_ice[1])),
            torch.ones_like(tid) * 1,
            tid,
        )
        tid = torch.where(
            (x_local >= float(self.cfg.bound_carpet[0])) & (x_local < float(self.cfg.bound_carpet[1])),
            torch.ones_like(tid) * 2,
            tid,
        )
        tid = torch.where(
            x_local >= float(self.cfg.bound_asphalt_park[0]),
            torch.ones_like(tid) * 3,
            tid,
        )

        return tid

    def terrain_one_hot(self, x_local: torch.Tensor) -> torch.Tensor:
        tid = self.terrain_id(x_local)
        out = torch.zeros((x_local.shape[0], 4), dtype=torch.float32, device=self.device)
        out.scatter_(1, tid.unsqueeze(-1), 1.0)
        return out

    def current_friction(self, x_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tid = self.terrain_id(x_local)
        static = torch.gather(self.terrain_static_friction, 1, tid.unsqueeze(-1)).squeeze(-1)
        dynamic = torch.gather(self.terrain_dynamic_friction, 1, tid.unsqueeze(-1)).squeeze(-1)
        return static, dynamic

    def compute_milestones(self, root_pos_local: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = root_pos_local[:, 0]

        pass_carpet = x > float(self.cfg.bound_carpet[0])
        pass_park_asphalt = x > float(self.cfg.bound_asphalt_park[0])
        pass_bumps = x.unsqueeze(-1) > (self.bump_pos[:, :, 0] + 0.15)

        return {
            "pass_carpet": pass_carpet,
            "pass_park_asphalt": pass_park_asphalt,
            "pass_bumps": pass_bumps,
            "terrain_progress_count": pass_carpet.float() + pass_park_asphalt.float(),
            "bump_progress_count": pass_bumps.float().sum(dim=-1),
        }

    # ------------------------------------------------------------------
    # Analytic 2D LiDAR
    # ------------------------------------------------------------------
    def compute_analytic_lidar(
        self,
        root_pos_local: torch.Tensor,
        yaw: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None,
        add_noise: bool = True,
        update_history: bool = True,
        normalize: bool = False,
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(root_pos_local.shape[0], dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        p = root_pos_local[:, :2]
        n = int(p.shape[0])
        r = int(self.cfg.lidar_pool_bins)

        ray_angles_w = (
            yaw[:, None]
            + self.pooled_lidar_angles[None, :]
            + self.lidar_yaw_offset[env_ids].unsqueeze(-1)
        )
        ray_dirs = torch.stack([torch.cos(ray_angles_w), torch.sin(ray_angles_w)], dim=-1)

        lidar = torch.full(
            (n, r),
            float(self.cfg.lidar_max_distance),
            dtype=torch.float32,
            device=self.device,
        )

        zeros = torch.zeros((n,), dtype=torch.float32, device=self.device)
        theta0 = torch.zeros((n,), dtype=torch.float32, device=self.device)

        hx_lane = torch.full((n,), float(self.cfg.track_length) * 0.5, dtype=torch.float32, device=self.device)
        hy_lane = torch.full((n,), float(self.cfg.lane_wall_thickness) * 0.5, dtype=torch.float32, device=self.device)

        north_center = torch.stack([zeros, torch.full_like(zeros, float(self.cfg.y_max))], dim=-1)
        south_center = torch.stack([zeros, torch.full_like(zeros, float(self.cfg.y_min))], dim=-1)

        lidar = torch.minimum(lidar, self._ray_obb_distance(p, ray_dirs, north_center, hx_lane, hy_lane, theta0))
        lidar = torch.minimum(lidar, self._ray_obb_distance(p, ray_dirs, south_center, hx_lane, hy_lane, theta0))

        spot_x = self.goal_pos[env_ids, 0]
        spot_y = self.goal_pos[env_ids, 1]
        spot_yaw = self.goal_yaw[env_ids]

        width = self.spot_width_inner[env_ids]
        depth = self.spot_depth_inner[env_ids]
        t = float(self.cfg.wall_thickness)

        c = torch.cos(spot_yaw)
        s = torch.sin(spot_yaw)

        def parking_local_to_world(dx: torch.Tensor, dy: torch.Tensor):
            x = spot_x + dx * c - dy * s
            y = spot_y + dx * s + dy * c
            return torch.stack([x, y], dim=-1)

        back_center = parking_local_to_world(depth * 0.5 + t * 0.5, torch.zeros_like(depth))
        back_hx = torch.full((n,), t * 0.5, dtype=torch.float32, device=self.device)
        back_hy = width * 0.5 + t

        left_center = parking_local_to_world(torch.zeros_like(depth), width * 0.5 + t * 0.5)
        left_hx = depth * 0.5
        left_hy = torch.full((n,), t * 0.5, dtype=torch.float32, device=self.device)

        right_center = parking_local_to_world(torch.zeros_like(depth), -width * 0.5 - t * 0.5)
        right_hx = depth * 0.5
        right_hy = torch.full((n,), t * 0.5, dtype=torch.float32, device=self.device)

        lidar = torch.minimum(lidar, self._ray_obb_distance(p, ray_dirs, back_center, back_hx, back_hy, spot_yaw))
        lidar = torch.minimum(lidar, self._ray_obb_distance(p, ray_dirs, left_center, left_hx, left_hy, spot_yaw))
        lidar = torch.minimum(lidar, self._ray_obb_distance(p, ray_dirs, right_center, right_hx, right_hy, spot_yaw))

        lidar = torch.clamp(lidar, min=float(self.cfg.lidar_min_distance), max=float(self.cfg.lidar_max_distance))
        lidar = torch.nan_to_num(
            lidar,
            nan=float(self.cfg.lidar_max_distance),
            posinf=float(self.cfg.lidar_max_distance),
            neginf=float(self.cfg.lidar_max_distance),
        )

        if add_noise:
            noise_std = self.lidar_noise_std[env_ids].unsqueeze(-1)
            outlier_prob = self.lidar_outlier_prob[env_ids].unsqueeze(-1)
            dropout_prob = self.lidar_dropout_prob[env_ids].unsqueeze(-1)

            lidar = lidar + torch.randn_like(lidar) * noise_std

            outlier_mask = torch.rand_like(lidar) < outlier_prob
            dropout_mask = torch.rand_like(lidar) < dropout_prob

            lidar = torch.where(
                outlier_mask | dropout_mask,
                torch.full_like(lidar, float(self.cfg.lidar_max_distance)),
                lidar,
            )
            lidar = torch.clamp(lidar, min=float(self.cfg.lidar_min_distance), max=float(self.cfg.lidar_max_distance))

        if update_history:
            prev = self.prev_lidar[env_ids].clone()
            delta = torch.clamp(
                (lidar - prev) / max(float(self.cfg.lidar_max_distance), 1e-6),
                -1.0,
                1.0,
            )
            self.last_lidar_delta[env_ids] = delta
            self.prev_lidar[env_ids] = lidar.detach().clone()
            self.last_lidar[env_ids] = lidar.detach().clone()

        if normalize:
            return torch.clamp(lidar / float(self.cfg.lidar_max_distance), 0.0, 1.0)

        return lidar

    def _ray_obb_distance(
        self,
        ray_origin: torch.Tensor,
        ray_dirs: torch.Tensor,
        rect_center: torch.Tensor,
        rect_hx: torch.Tensor,
        rect_hy: torch.Tensor,
        rect_yaw: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-8
        max_d = float(self.cfg.lidar_max_distance)

        rel = ray_origin[:, None, :] - rect_center[:, None, :]

        c = torch.cos(-rect_yaw)[:, None]
        s = torch.sin(-rect_yaw)[:, None]

        ox = c * rel[:, :, 0] - s * rel[:, :, 1]
        oy = s * rel[:, :, 0] + c * rel[:, :, 1]

        dx = c * ray_dirs[:, :, 0] - s * ray_dirs[:, :, 1]
        dy = s * ray_dirs[:, :, 0] + c * ray_dirs[:, :, 1]

        hx = rect_hx[:, None]
        hy = rect_hy[:, None]

        inv_dx = torch.where(torch.abs(dx) > eps, 1.0 / dx, torch.full_like(dx, 1.0e8))
        inv_dy = torch.where(torch.abs(dy) > eps, 1.0 / dy, torch.full_like(dy, 1.0e8))

        tx1 = (-hx - ox) * inv_dx
        tx2 = (hx - ox) * inv_dx
        ty1 = (-hy - oy) * inv_dy
        ty2 = (hy - oy) * inv_dy

        tmin_x = torch.minimum(tx1, tx2)
        tmax_x = torch.maximum(tx1, tx2)
        tmin_y = torch.minimum(ty1, ty2)
        tmax_y = torch.maximum(ty1, ty2)

        t_near = torch.maximum(tmin_x, tmin_y)
        t_far = torch.minimum(tmax_x, tmax_y)

        hit = t_far >= torch.maximum(t_near, torch.zeros_like(t_near))
        t = torch.where(t_near > 0.0, t_near, t_far)

        dist = torch.where(hit & (t >= 0.0), t, torch.full_like(t, max_d))
        return torch.clamp(dist, min=float(self.cfg.lidar_min_distance), max=max_d)

    # ------------------------------------------------------------------
    # RayCaster LiDAR processing helper
    # ------------------------------------------------------------------
    def process_lidar_data(
        self,
        raycaster,
        env_ids: Optional[torch.Tensor] = None,
        add_noise: bool = True,
        update_history: bool = True,
        normalize: bool = False,
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        hit_pos_w = raycaster.data.ray_hits_w
        sensor_pos_w = raycaster.data.pos_w

        if sensor_pos_w.dim() == 2:
            sensor_pos_w = sensor_pos_w.unsqueeze(1)
        if hit_pos_w.dim() == 4:
            hit_pos_w = hit_pos_w.squeeze(1)

        hit_pos_w = hit_pos_w[env_ids]
        sensor_pos_w = sensor_pos_w[env_ids]

        distances = torch.norm(hit_pos_w - sensor_pos_w, dim=-1)
        distances = torch.nan_to_num(
            distances,
            nan=float(self.cfg.lidar_max_distance),
            posinf=float(self.cfg.lidar_max_distance),
            neginf=float(self.cfg.lidar_max_distance),
        )
        distances = torch.clamp(
            distances,
            min=float(self.cfg.lidar_min_distance),
            max=float(self.cfg.lidar_max_distance),
        )

        pooled = self._pool_lidar(distances)

        if add_noise:
            noise_std = self.lidar_noise_std[env_ids].unsqueeze(-1)
            outlier_prob = self.lidar_outlier_prob[env_ids].unsqueeze(-1)
            dropout_prob = self.lidar_dropout_prob[env_ids].unsqueeze(-1)

            pooled = pooled + torch.randn_like(pooled) * noise_std
            outlier_mask = torch.rand_like(pooled) < outlier_prob
            dropout_mask = torch.rand_like(pooled) < dropout_prob

            pooled = torch.where(
                outlier_mask | dropout_mask,
                torch.full_like(pooled, float(self.cfg.lidar_max_distance)),
                pooled,
            )

        pooled = torch.clamp(pooled, min=float(self.cfg.lidar_min_distance), max=float(self.cfg.lidar_max_distance))
        pooled = torch.nan_to_num(
            pooled,
            nan=float(self.cfg.lidar_max_distance),
            posinf=float(self.cfg.lidar_max_distance),
            neginf=float(self.cfg.lidar_max_distance),
        )

        if update_history:
            prev = self.prev_lidar[env_ids].clone()
            delta = torch.clamp(
                (pooled - prev) / max(float(self.cfg.lidar_max_distance), 1e-6),
                -1.0,
                1.0,
            )
            self.last_lidar_delta[env_ids] = delta
            self.prev_lidar[env_ids] = pooled.detach().clone()
            self.last_lidar[env_ids] = pooled.detach().clone()

        if normalize:
            return torch.clamp(pooled / float(self.cfg.lidar_max_distance), 0.0, 1.0)

        return pooled

    def _pool_lidar(self, raw_distances: torch.Tensor) -> torch.Tensor:
        n, r = raw_distances.shape
        bins = int(self.cfg.lidar_pool_bins)

        if r == bins:
            return raw_distances

        if r > bins:
            bin_size = max(r // bins, 1)
            usable = bins * bin_size
            return raw_distances[:, :usable].reshape(n, bins, bin_size).min(dim=-1)[0]

        repeat = math.ceil(bins / max(r, 1))
        expanded = raw_distances.repeat(1, repeat)
        return expanded[:, :bins]

    # ------------------------------------------------------------------
    # Risk / privileged features
    # ------------------------------------------------------------------
    def compute_risk_features(
        self,
        root_pos_local: torch.Tensor,
        yaw: torch.Tensor,
        lidar_pooled: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = int(root_pos_local.shape[0])

        if lidar_pooled is None:
            lidar_pooled = self.last_lidar[:n]

        angles = self.pooled_lidar_angles
        front_rad = math.radians(float(self.cfg.front_angle_deg))
        side_rad = math.radians(float(self.cfg.side_angle_deg))

        front_mask = torch.abs(angles) <= front_rad
        left_mask = (angles > front_rad) & (angles <= side_rad)
        right_mask = (angles < -front_rad) & (angles >= -side_rad)

        all_min = lidar_pooled.min(dim=-1)[0]
        front_min = lidar_pooled[:, front_mask].min(dim=-1)[0]
        left_min = lidar_pooled[:, left_mask].min(dim=-1)[0]
        right_min = lidar_pooled[:, right_mask].min(dim=-1)[0]

        risk_d = float(self.cfg.risk_distance)

        all_risk = torch.clamp((risk_d - all_min) / risk_d, 0.0, 1.0)
        front_risk = torch.clamp((risk_d - front_min) / risk_d, 0.0, 1.0)
        left_risk = torch.clamp((risk_d - left_min) / risk_d, 0.0, 1.0)
        right_risk = torch.clamp((risk_d - right_min) / risk_d, 0.0, 1.0)

        lane_margin_center = self.lane_margin(root_pos_local)
        lane_body_clearance = (
            lane_margin_center
            - float(self.cfg.robot_radius)
            - float(self.cfg.lane_collision_margin)
        )
        lane_risk = torch.clamp((0.45 - lane_body_clearance) / 0.45, 0.0, 1.0)

        wall_dist = self.parking_wall_signed_distance(root_pos_local)
        parking_wall_risk = torch.clamp((0.50 - wall_dist) / 0.50, 0.0, 1.0)

        bump_overlap = self.check_bump_overlap(root_pos_local)
        bump_dist = self._min_bump_signed_distance(root_pos_local)
        bump_risk = torch.clamp((0.35 - bump_dist) / 0.35, 0.0, 1.0)
        bump_risk = torch.maximum(bump_risk, bump_overlap.float())

        front_clearance = torch.clamp(front_min / float(self.cfg.lidar_max_distance), 0.0, 1.0)
        near_parking_zone = (root_pos_local[:, 0] > float(self.cfg.bound_asphalt_park[0])).float()
        inside_box = self.is_inside_parking_box(root_pos_local).float()

        features = torch.stack(
            [
                all_risk,
                front_risk,
                left_risk,
                right_risk,
                lane_risk,
                parking_wall_risk,
                bump_risk,
                front_clearance,
                near_parking_zone,
                inside_box,
            ],
            dim=-1,
        )

        features = torch.nan_to_num(
            torch.clamp(features, 0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        self.last_risk_features[:n] = features.detach().clone()
        return features

    def _min_bump_signed_distance(self, root_pos_local: torch.Tensor) -> torch.Tensor:
        p = root_pos_local[:, None, :2] - self.bump_pos
        yaw = self.bump_yaw

        c = torch.cos(-yaw)
        s = torch.sin(-yaw)

        bx = c * p[:, :, 0] - s * p[:, :, 1]
        by = s * p[:, :, 0] + c * p[:, :, 1]

        hx = self.bump_length_x * 0.5
        hy = self.bump_width_y * 0.5

        d = self._point_rect_signed_distance(
            bx,
            by,
            torch.zeros_like(bx),
            torch.zeros_like(by),
            hx,
            hy,
        )

        return d.min(dim=-1)[0]

    def compute_privileged_features(self, root_pos_local: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        x = root_pos_local[:, 0]
        y = root_pos_local[:, 1]

        terrain_oh = self.terrain_one_hot(x)
        static_mu, dynamic_mu = self.current_friction(x)

        action_delay_norm = self.action_delay_frames.float() / max(float(self.cfg.action_delay_frame_range[1]), 1.0)
        deadband_norm = self.action_deadband / max(float(self.cfg.action_deadband_range[1]), 1e-6)

        motor_l = self.motor_strength[:, 0]
        motor_r = self.motor_strength[:, 1]
        bias_l = self.motor_bias[:, 0]
        bias_r = self.motor_bias[:, 1]
        wheel_l = self.wheel_radius_scale[:, 0]
        wheel_r = self.wheel_radius_scale[:, 1]

        lidar_noise_norm = self.lidar_noise_std / max(float(self.cfg.lidar_noise_std_range[1]), 1e-6)
        lidar_yaw_norm = self.lidar_yaw_offset / max(abs(float(self.cfg.lidar_yaw_offset_range[1])), 1e-6)
        lidar_z_norm = (self.lidar_z_offset - float(self.cfg.lidar_z_offset_range[0])) / max(
            float(self.cfg.lidar_z_offset_range[1]) - float(self.cfg.lidar_z_offset_range[0]),
            1e-6,
        )

        dr = torch.stack(
            [
                action_delay_norm,
                deadband_norm,
                self.action_ema_alpha,
                motor_l,
                motor_r,
                bias_l,
                bias_r,
                wheel_l,
                wheel_r,
                lidar_noise_norm,
                self.lidar_outlier_prob,
                self.lidar_dropout_prob,
                lidar_yaw_norm,
                lidar_z_norm,
            ],
            dim=-1,
        )

        goal = self.compute_goal_terms(root_pos_local, yaw)

        parking = torch.stack(
            [
                self.spot_width_inner / max(float(self.cfg.spot_width_inner_range[1]), 1e-6),
                self.spot_depth_inner / max(float(self.cfg.spot_depth_inner_range[1]), 1e-6),
                torch.sin(self.goal_yaw),
                torch.cos(self.goal_yaw),
                goal["parking_x_norm"],
                goal["parking_y_norm"],
                goal["goal_yaw_sin"],
                goal["goal_yaw_cos"],
            ],
            dim=-1,
        )

        bump_x_norm = torch.clamp(self.bump_pos[:, :, 0] / float(self.cfg.track_x_norm), -1.0, 1.0)
        bump_h_norm = torch.clamp(self.bump_height / max(float(self.cfg.bump_height_range[1]), 1e-6), 0.0, 2.0)
        bump_info = torch.cat([bump_x_norm, bump_h_norm], dim=-1)

        local_xy = torch.stack(
            [
                torch.clamp(x / float(self.cfg.track_x_norm), -1.5, 1.5),
                torch.clamp(y / float(self.cfg.track_y_norm), -1.5, 1.5),
            ],
            dim=-1,
        )

        priv = torch.cat(
            [
                terrain_oh,
                static_mu.unsqueeze(-1),
                dynamic_mu.unsqueeze(-1),
                dr,
                parking,
                bump_info,
                local_xy,
            ],
            dim=-1,
        )

        if priv.shape[-1] != self.privileged_feature_dim():
            raise RuntimeError(
                f"privileged feature dim mismatch: {priv.shape[-1]} != {self.privileged_feature_dim()}"
            )

        return torch.nan_to_num(
            torch.clamp(priv, -10.0, 10.0),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

    # ------------------------------------------------------------------
    # Debug stats
    # ------------------------------------------------------------------
    def get_debug_stats(
        self,
        root_pos_local: Optional[torch.Tensor] = None,
        yaw: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        stats = {
            "Start_X": self.start_pos[:, 0].mean().item(),
            "Start_Y": self.start_pos[:, 1].mean().item(),
            "Goal_X": self.goal_pos[:, 0].mean().item(),
            "Goal_Y": self.goal_pos[:, 1].mean().item(),
            "Goal_Yaw": self.goal_yaw.mean().item(),
            "Spot_Width": self.spot_width_inner.mean().item(),
            "Spot_Depth": self.spot_depth_inner.mean().item(),
            "Bump_X_Mean": self.bump_pos[:, :, 0].mean().item(),
            "Bump_Height": self.bump_height.mean().item(),
            "Action_Delay": self.action_delay_frames.float().mean().item(),
            "Action_Deadband": self.action_deadband.mean().item(),
            "Action_EMA": self.action_ema_alpha.mean().item(),
            "Motor_Strength": self.motor_strength.mean().item(),
            "Wheel_Radius_Scale": self.wheel_radius_scale.mean().item(),
            "Lidar_Noise_Std": self.lidar_noise_std.mean().item(),
            "Lidar_Outlier_Prob": self.lidar_outlier_prob.mean().item(),
            "Lidar_Dropout_Prob": self.lidar_dropout_prob.mean().item(),
            "Lidar_Min": self.last_lidar.min().item(),
            "Lidar_Mean": self.last_lidar.mean().item(),
            "Risk_All": self.last_risk_features[:, 0].mean().item(),
            "Risk_Front": self.last_risk_features[:, 1].mean().item(),
            "Risk_Wall": self.last_risk_features[:, 5].mean().item(),
            "Risk_Bump": self.last_risk_features[:, 6].mean().item(),
        }

        if root_pos_local is not None and yaw is not None:
            goal = self.compute_goal_terms(root_pos_local, yaw)
            event = self.check_events(root_pos_local, yaw)
            static_mu, dynamic_mu = self.current_friction(root_pos_local[:, 0])

            stats.update(
                {
                    "Goal_Dist": goal["goal_dist"].mean().item(),
                    "Heading_Error": torch.abs(goal["heading_error"]).mean().item(),
                    "Goal_Yaw_Error": torch.abs(goal["goal_yaw_error"]).mean().item(),
                    "Parking_X": goal["parking_x"].mean().item(),
                    "Parking_Y": goal["parking_y"].mean().item(),
                    "Inside_Box": event["inside_parking_box"].float().mean().item(),
                    "Success_Candidate": event["success_candidate"].float().mean().item(),
                    "Out_Of_Lane": event["out_of_lane"].float().mean().item(),
                    "Parking_Wall_Collision": event["parking_wall_collision"].float().mean().item(),
                    "Bump_Overlap": event["bump_overlap"].float().mean().item(),
                    "Crash": event["crash"].float().mean().item(),
                    "Lane_Margin": event["lane_margin"].mean().item(),
                    "Wall_Signed_Distance": event["wall_signed_distance"].mean().item(),
                    "Current_Static_Friction": static_mu.mean().item(),
                    "Current_Dynamic_Friction": dynamic_mu.mean().item(),
                }
            )

        return stats

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _uniform(self, rng: Tuple[float, float], shape) -> torch.Tensor:
        lo, hi = float(rng[0]), float(rng[1])
        return lo + torch.rand(shape, dtype=torch.float32, device=self.device) * (hi - lo)

    @staticmethod
    def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def rotate_world_to_body_2d(self, vec_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        c = torch.cos(-yaw)
        s = torch.sin(-yaw)

        x = vec_w[:, 0]
        y = vec_w[:, 1]

        bx = c * x - s * y
        by = s * x + c * y

        return torch.stack([bx, by], dim=-1)

    def rotate_body_to_world_2d(self, vec_b: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        c = torch.cos(yaw)
        s = torch.sin(yaw)

        x = vec_b[:, 0]
        y = vec_b[:, 1]

        wx = c * x - s * y
        wy = s * x + c * y

        return torch.stack([wx, wy], dim=-1)

    @staticmethod
    def _point_rect_signed_distance(
        px: torch.Tensor,
        py: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
        hx: torch.Tensor,
        hy: torch.Tensor,
    ) -> torch.Tensor:
        dx = torch.abs(px - cx) - hx
        dy = torch.abs(py - cy) - hy

        outside_x = torch.clamp(dx, min=0.0)
        outside_y = torch.clamp(dy, min=0.0)
        outside_dist = torch.sqrt(outside_x * outside_x + outside_y * outside_y)

        inside_dist = torch.minimum(torch.maximum(dx, dy), torch.zeros_like(dx))
        return outside_dist + inside_dist


# ======================================================================
# Asset spawner
# ======================================================================

def spawn_world_assets(scene_cfg: InteractiveSceneCfg, cfg: Task3WorldConfig) -> None:
    """Conservative manual asset spawn + RigidObject registration.

    This follows the stable Task3 policy:
        1. spawn geometry manually into /World/envs/env_0
        2. register selected existing prims with RigidObjectCfg(prim_path=...)
        3. do not use RigidObjectCfg(spawn=...) for bumps / parking walls
    """

    cfg.validate()

    # ------------------------------------------------------------------
    # Floors
    # ------------------------------------------------------------------
    def spawn_floor_tile(
        prim_name: str,
        bound: Tuple[float, float],
        mat_params: Tuple[float, float],
        color: Tuple[float, float, float],
    ) -> None:
        length = float(bound[1] - bound[0])
        center_x = float(bound[0] + 0.5 * length)

        floor_cfg = sim_utils.CuboidCfg(
            size=(length, float(cfg.track_width), float(cfg.ground_height)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=float(mat_params[0]),
                dynamic_friction=float(mat_params[1]),
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )

        floor_cfg.func(
            f"/World/envs/env_0/Floor/{prim_name}",
            floor_cfg,
            translation=(center_x, 0.0, -float(cfg.ground_height) * 0.5),
        )

    spawn_floor_tile("Start_Asphalt", cfg.bound_asphalt_start, cfg.mat_asphalt_nominal, (0.18, 0.18, 0.18))
    spawn_floor_tile("Middle_Ice", cfg.bound_ice, cfg.mat_ice_nominal, (0.82, 0.92, 1.0))
    spawn_floor_tile("Middle_Carpet", cfg.bound_carpet, cfg.mat_carpet_nominal, (0.35, 0.35, 0.35))
    spawn_floor_tile("Park_Asphalt", cfg.bound_asphalt_park, cfg.mat_asphalt_nominal, (0.20, 0.20, 0.20))

    # ------------------------------------------------------------------
    # Lane air walls
    # ------------------------------------------------------------------
    air_wall_cfg = sim_utils.CuboidCfg(
        size=(float(cfg.track_length), float(cfg.lane_wall_thickness), float(cfg.lane_wall_height)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )

    air_wall_cfg.func(
        "/World/envs/env_0/AirWalls/Wall_North",
        air_wall_cfg,
        translation=(0.0, float(cfg.track_width) * 0.5, float(cfg.lane_wall_height) * 0.5),
    )
    air_wall_cfg.func(
        "/World/envs/env_0/AirWalls/Wall_South",
        air_wall_cfg,
        translation=(0.0, -float(cfg.track_width) * 0.5, float(cfg.lane_wall_height) * 0.5),
    )

    # ------------------------------------------------------------------
    # 3-section speed bumps
    # ------------------------------------------------------------------
    def spawn_ramp_bump(i: int) -> None:
        zone_min, zone_max = cfg.bump_zones[i]
        fixed_x = 0.5 * (float(zone_min) + float(zone_max))

        total_lx = float(cfg.bump_length_x_nominal)
        width_y = float(cfg.bump_width_y_nominal)
        height = float(cfg.bump_height_nominal)

        top_lx = min(float(cfg.bump_top_length), total_lx * 0.40)
        side_lx = max((total_lx - top_lx) * 0.5, 1.0e-4)

        low_ratio = float(cfg.bump_low_height_ratio)
        low_h = max(height * low_ratio, 0.001)
        low_h = min(low_h, height)

        bump_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.78, 0.10))

        front_x = fixed_x - 0.5 * top_lx - 0.5 * side_lx
        front_cfg = sim_utils.CuboidCfg(
            size=(side_lx, width_y, low_h),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=bump_visual,
        )
        front_cfg.func(
            f"/World/envs/env_0/SpeedBumps/Bump_{i}_FrontLow",
            front_cfg,
            translation=(front_x, 0.0, low_h * 0.5),
        )

        top_cfg = sim_utils.CuboidCfg(
            size=(top_lx, width_y, height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=bump_visual,
        )
        top_cfg.func(
            f"/World/envs/env_0/SpeedBumps/Bump_{i}",
            top_cfg,
            translation=(fixed_x, 0.0, height * 0.5),
        )

        setattr(
            scene_cfg,
            f"speed_bump_{i}",
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/SpeedBumps/Bump_{i}",
            ),
        )

        rear_x = fixed_x + 0.5 * top_lx + 0.5 * side_lx
        rear_cfg = sim_utils.CuboidCfg(
            size=(side_lx, width_y, low_h),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=bump_visual,
        )
        rear_cfg.func(
            f"/World/envs/env_0/SpeedBumps/Bump_{i}_RearLow",
            rear_cfg,
            translation=(rear_x, 0.0, low_h * 0.5),
        )

    for i in range(int(cfg.num_speed_bumps)):
        spawn_ramp_bump(i)

    # ------------------------------------------------------------------
    # Parking U walls
    # ------------------------------------------------------------------
    wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.80, 0.20))

    nominal_width = float(cfg.spot_width_inner_nominal)
    nominal_depth = float(cfg.spot_depth_inner_nominal)

    park_back_cfg = sim_utils.CuboidCfg(
        size=(float(cfg.wall_thickness), nominal_width + float(cfg.wall_thickness) * 2.0, float(cfg.wall_height)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=wall_mat,
    )

    park_side_cfg = sim_utils.CuboidCfg(
        size=(nominal_depth, float(cfg.wall_thickness), float(cfg.wall_height)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=wall_mat,
    )

    wall_z = float(cfg.wall_height) * 0.5

    goal_x = float(cfg.parking_x_nominal)
    goal_y = float(cfg.parking_y_nominal)

    back_x = goal_x + nominal_depth * 0.5 + float(cfg.wall_thickness) * 0.5
    back_y = goal_y

    left_x = goal_x
    left_y = goal_y + nominal_width * 0.5 + float(cfg.wall_thickness) * 0.5

    right_x = goal_x
    right_y = goal_y - nominal_width * 0.5 - float(cfg.wall_thickness) * 0.5

    park_back_cfg.func(
        "/World/envs/env_0/Parking/Back",
        park_back_cfg,
        translation=(back_x, back_y, wall_z),
    )
    park_side_cfg.func(
        "/World/envs/env_0/Parking/Left",
        park_side_cfg,
        translation=(left_x, left_y, wall_z),
    )
    park_side_cfg.func(
        "/World/envs/env_0/Parking/Right",
        park_side_cfg,
        translation=(right_x, right_y, wall_z),
    )

    scene_cfg.park_back = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Parking/Back")
    scene_cfg.park_left = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Parking/Left")
    scene_cfg.park_right = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Parking/Right")


def get_lidar_cfg(prim_path: str, cfg: Optional[Task3WorldConfig] = None) -> RayCasterCfg:
    if cfg is None:
        cfg = Task3WorldConfig()

    cfg.validate()

    return RayCasterCfg(
        prim_path=prim_path,
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=cfg.lidar_default_offset),
        ray_alignment="yaw",
        pattern_cfg=patterns.BpearlPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World"],
        max_distance=float(cfg.lidar_max_distance),
    )


JetbotTask3WorldConfig = Task3WorldConfig
JetbotTask3WorldManager = Task3WorldManager

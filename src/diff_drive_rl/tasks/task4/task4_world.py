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
class Task4WorldConfig:
    """Jetbot Task4 Multi-UGV Cooperative Formation Escort World.

    Task:
        Four two-wheel differential-drive UGVs escort a virtual formation
        center to a shared team goal. They must keep Diamond / Wedge / Line
        formation, avoid static obstacles, and pass through a narrow gate.

    Coordinate convention:
        start_pos / goal_pos / obstacle_pos / gate_x are all env-local.
        Isaac world xy must be env_origin.xy + local_xy.

    Conservative asset policy:
        Arena floor, boundary walls, static obstacles, and gate walls are
        spawned manually into /World/envs/env_0. Dynamic reset only teleports
        obstacle / gate rigid objects. Collider sizes stay nominal.
    """

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------
    num_agents: int = 4
    robot_radius: float = 0.18

    # ------------------------------------------------------------------
    # Arena geometry
    # ------------------------------------------------------------------
    arena_length: float = 18.0
    arena_width: float = 12.0
    ground_height: float = 0.02

    x_min: float = -9.0
    x_max: float = 9.0
    y_min: float = -6.0
    y_max: float = 6.0

    wall_thickness: float = 0.12
    wall_height: float = 0.65

    mat_floor_nominal: Tuple[float, float] = (0.80, 0.75)
    mat_wall_nominal: Tuple[float, float] = (0.60, 0.55)

    # ------------------------------------------------------------------
    # Start / goal sampling
    # ------------------------------------------------------------------
    start_center_x_range: Tuple[float, float] = (-7.0, -6.0)
    start_center_y_range: Tuple[float, float] = (-1.5, 1.5)
    start_heading_range: Tuple[float, float] = (-0.20, 0.20)

    goal_x_range: Tuple[float, float] = (6.0, 7.5)
    goal_y_range: Tuple[float, float] = (-3.0, 3.0)
    goal_yaw_range: Tuple[float, float] = (-0.25, 0.25)

    goal_center_success_tol: float = 0.45
    goal_formation_success_tol: float = 0.35
    goal_speed_success_tol: float = 0.20
    success_hold_steps: int = 10

    # ------------------------------------------------------------------
    # Formation definitions
    # ------------------------------------------------------------------
    # 0 = Diamond, 1 = Wedge, 2 = Line
    num_formation_types: int = 3
    default_formation_type: int = 0

    formation_scale_range: Tuple[float, float] = (0.85, 1.20)
    formation_scale_gate: float = 0.80

    slot_error_norm: float = 1.5
    team_spread_norm: float = 2.0

    # ------------------------------------------------------------------
    # Static obstacles
    # ------------------------------------------------------------------
    max_static_obstacles: int = 8
    obstacle_size_xy: Tuple[float, float] = (0.42, 0.42)
    obstacle_height: float = 0.55

    obstacle_x_range: Tuple[float, float] = (-3.8, 4.8)
    obstacle_y_range: Tuple[float, float] = (-4.2, 4.2)

    obstacle_clearance_start: float = 1.40
    obstacle_clearance_goal: float = 1.30
    obstacle_clearance_gate: float = 0.75

    obstacle_count_by_stage: Tuple[int, int, int, int, int, int] = (0, 2, 4, 5, 7, 8)

    # ------------------------------------------------------------------
    # Narrow gate
    # ------------------------------------------------------------------
    gate_stage_start: int = 3
    gate_x_range: Tuple[float, float] = (-0.8, 1.8)

    gate_gap_width: float = 1.45
    gate_wall_size_x: float = 0.28
    gate_wall_height: float = 0.65

    gate_region_half_x: float = 1.0
    gate_pass_margin_x: float = 0.45

    # ------------------------------------------------------------------
    # Sim2Real / domain randomization buffers
    # ------------------------------------------------------------------
    max_speed_range: Tuple[float, float] = (0.70, 1.25)
    max_yaw_rate_range: Tuple[float, float] = (1.50, 2.60)

    action_delay_frame_range: Tuple[int, int] = (0, 3)
    action_deadband_range: Tuple[float, float] = (0.00, 0.06)
    action_ema_alpha_range: Tuple[float, float] = (0.35, 0.75)

    motor_strength_range: Tuple[float, float] = (0.82, 1.18)
    motor_bias_range: Tuple[float, float] = (-0.04, 0.04)
    wheel_radius_scale_range: Tuple[float, float] = (0.94, 1.06)

    lidar_noise_std_range: Tuple[float, float] = (0.005, 0.045)
    lidar_outlier_prob_range: Tuple[float, float] = (0.000, 0.025)
    lidar_dropout_prob_range: Tuple[float, float] = (0.000, 0.020)
    lidar_yaw_offset_range: Tuple[float, float] = (-math.radians(2.0), math.radians(2.0))
    lidar_z_offset_range: Tuple[float, float] = (0.10, 0.20)

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------
    lidar_max_distance: float = 8.0
    lidar_min_distance: float = 0.02
    lidar_pool_bins: int = 48
    lidar_default_offset: Tuple[float, float, float] = (0.0, 0.0, 0.15)

    front_angle_deg: float = 35.0
    side_angle_deg: float = 120.0
    risk_distance: float = 0.95
    pair_risk_distance: float = 0.70
    obstacle_risk_distance: float = 0.90
    boundary_risk_distance: float = 0.80
    gate_risk_distance: float = 0.80

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    arena_x_norm: float = 9.0
    arena_y_norm: float = 6.0
    goal_xy_norm: float = 9.0
    goal_dist_norm: float = 18.0
    yaw_norm: float = math.pi
    speed_norm: float = 1.5

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    debug_assert: bool = False

    # World-test safety switch:
    # Isaac/PhysX GPU tensor can be unstable if many kinematic RigidObject
    # views are teleported repeatedly in a pure world-layer test. The analytic
    # world tensors, obstacle geometry, gate geometry, LiDAR, risk and privileged
    # features do not require physical teleport during the world test.
    # Env-level tests/training can keep this True.
    enable_physical_asset_teleport: bool = True

    @property
    def obstacle_half_extents_nominal(self) -> Tuple[float, float]:
        return (0.5 * float(self.obstacle_size_xy[0]), 0.5 * float(self.obstacle_size_xy[1]))

    @property
    def gate_wall_length_y(self) -> float:
        return max(0.5 * (float(self.arena_width) - float(self.gate_gap_width)), 0.10)

    @property
    def gate_top_center_y(self) -> float:
        return 0.5 * float(self.gate_gap_width) + 0.5 * float(self.gate_wall_length_y)

    @property
    def gate_bottom_center_y(self) -> float:
        return -0.5 * float(self.gate_gap_width) - 0.5 * float(self.gate_wall_length_y)

    def validate(self) -> None:
        assert self.num_agents == 4, "Task4 currently expects exactly 4 agents"
        assert self.robot_radius > 0.0

        assert self.arena_length > 0.0
        assert self.arena_width > 0.0
        assert self.ground_height > 0.0
        assert self.x_min < self.x_max
        assert self.y_min < self.y_max
        assert abs(self.x_max - self.x_min - self.arena_length) < 1e-5
        assert abs(self.y_max - self.y_min - self.arena_width) < 1e-5

        assert self.wall_thickness > 0.0
        assert self.wall_height > 0.0

        for rng in [
            self.start_center_x_range,
            self.start_center_y_range,
            self.start_heading_range,
            self.goal_x_range,
            self.goal_y_range,
            self.goal_yaw_range,
            self.formation_scale_range,
            self.obstacle_x_range,
            self.obstacle_y_range,
            self.gate_x_range,
            self.max_speed_range,
            self.max_yaw_rate_range,
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

        assert self.goal_center_success_tol > 0.0
        assert self.goal_formation_success_tol > 0.0
        assert self.goal_speed_success_tol >= 0.0
        assert self.success_hold_steps >= 1

        assert self.num_formation_types == 3
        assert 0 <= self.default_formation_type < self.num_formation_types
        assert self.formation_scale_gate > 0.0
        assert self.slot_error_norm > 0.0
        assert self.team_spread_norm > 0.0

        assert self.max_static_obstacles > 0
        assert len(self.obstacle_count_by_stage) >= 1
        assert max(self.obstacle_count_by_stage) <= self.max_static_obstacles
        assert self.obstacle_size_xy[0] > 0.0 and self.obstacle_size_xy[1] > 0.0
        assert self.obstacle_height > 0.0
        assert self.obstacle_clearance_start > 0.0
        assert self.obstacle_clearance_goal > 0.0
        assert self.obstacle_clearance_gate >= 0.0

        assert self.gate_stage_start >= 0
        assert self.gate_gap_width > 0.0
        assert self.gate_wall_size_x > 0.0
        assert self.gate_wall_height > 0.0
        assert self.gate_region_half_x > 0.0
        assert self.gate_pass_margin_x >= 0.0
        assert self.gate_wall_length_y > 0.0

        assert self.action_delay_frame_range[1] >= self.action_delay_frame_range[0] >= 0

        assert self.lidar_max_distance > self.lidar_min_distance > 0.0
        assert self.lidar_pool_bins >= 16
        assert 0.0 < self.front_angle_deg < self.side_angle_deg <= 180.0
        assert self.risk_distance > 0.0
        assert self.pair_risk_distance > 0.0
        assert self.obstacle_risk_distance > 0.0
        assert self.boundary_risk_distance > 0.0
        assert self.gate_risk_distance > 0.0

        assert self.arena_x_norm > 0.0
        assert self.arena_y_norm > 0.0
        assert self.goal_xy_norm > 0.0
        assert self.goal_dist_norm > 0.0
        assert self.yaw_norm > 0.0
        assert self.speed_norm > 0.0


class Task4WorldManager:
    """Multi-UGV formation escort world tensor manager."""

    def __init__(
        self,
        scene: InteractiveScene,
        cfg: Task4WorldConfig,
        num_envs: int,
        device: str,
    ):
        cfg.validate()

        self.scene = scene
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = str(device)
        self.num_agents = int(cfg.num_agents)

        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if self.num_agents != 4:
            raise ValueError("Task4 currently expects exactly 4 agents")

        self.static_obstacles = [
            self.scene.rigid_objects[f"static_obstacle_{i}"]
            for i in range(int(self.cfg.max_static_obstacles))
        ]
        self.gate_top: RigidObject = self.scene.rigid_objects["gate_top"]
        self.gate_bottom: RigidObject = self.scene.rigid_objects["gate_bottom"]

        self._init_formation_offsets()
        self._init_buffers()

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------
    @staticmethod
    def risk_feature_dim() -> int:
        # [all, front, left, right,
        #  boundary, obstacle, gate, pair,
        #  slot_error, center_goal_dist, heading_align, slot_heading_error,
        #  team_spread, nearest_teammate_clearance, near_gate, gate_active]
        return 16

    @staticmethod
    def privileged_feature_dim(max_static_obstacles: int = 8, num_agents: int = 4) -> int:
        # goal 4
        # formation 4
        # gate 3
        # obstacles K * 4
        # robots A * 5
        # team 9
        # domain A * 6
        return 4 + 4 + 3 + int(max_static_obstacles) * 4 + int(num_agents) * 5 + 9 + int(num_agents) * 6

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def _init_formation_offsets(self) -> None:
        diamond = torch.tensor(
            [
                [0.45, 0.00],
                [0.00, 0.45],
                [0.00, -0.45],
                [-0.45, 0.00],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        wedge = torch.tensor(
            [
                [0.50, 0.00],
                [0.05, 0.42],
                [0.05, -0.42],
                [-0.45, 0.00],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Line nominal spacing is deliberately large enough so Stage3
        # compressed line still avoids false pair-collision at reset.
        line = torch.tensor(
            [
                [0.84, 0.00],
                [0.28, 0.00],
                [-0.28, 0.00],
                [-0.84, 0.00],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        self.formation_offsets = torch.stack([diamond, wedge, line], dim=0)

    def _init_buffers(self) -> None:
        n = self.num_envs
        a = self.num_agents
        k = int(self.cfg.max_static_obstacles)

        self.curriculum_stage = torch.zeros((n,), dtype=torch.long, device=self.device)

        self.start_center = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        self.start_heading = torch.zeros((n,), dtype=torch.float32, device=self.device)

        self.start_pos = torch.zeros((n, a, 2), dtype=torch.float32, device=self.device)
        self.start_yaw = torch.zeros((n, a), dtype=torch.float32, device=self.device)

        self.goal_pos = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        self.goal_yaw = torch.zeros((n,), dtype=torch.float32, device=self.device)

        self.formation_type = torch.full(
            (n,),
            int(self.cfg.default_formation_type),
            dtype=torch.long,
            device=self.device,
        )
        self.formation_scale = torch.ones((n,), dtype=torch.float32, device=self.device)

        self.obstacle_pos = torch.zeros((n, k, 2), dtype=torch.float32, device=self.device)
        self.obstacle_half_extents = torch.zeros((n, k, 2), dtype=torch.float32, device=self.device)
        self.obstacle_active = torch.zeros((n, k), dtype=torch.bool, device=self.device)

        self.gate_active = torch.zeros((n,), dtype=torch.bool, device=self.device)
        self.gate_x = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.gate_gap_width = torch.full(
            (n,),
            float(self.cfg.gate_gap_width),
            dtype=torch.float32,
            device=self.device,
        )

        self.max_speed = torch.ones((n, a), dtype=torch.float32, device=self.device)
        self.max_yaw_rate = torch.ones((n, a), dtype=torch.float32, device=self.device)

        self.action_delay_frames = torch.zeros((n, a), dtype=torch.long, device=self.device)
        self.action_deadband = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.action_ema_alpha = torch.zeros((n, a), dtype=torch.float32, device=self.device)

        self.motor_strength = torch.ones((n, a, 2), dtype=torch.float32, device=self.device)
        self.motor_bias = torch.zeros((n, a, 2), dtype=torch.float32, device=self.device)
        self.wheel_radius_scale = torch.ones((n, a, 2), dtype=torch.float32, device=self.device)

        self.lidar_noise_std = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.lidar_outlier_prob = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.lidar_dropout_prob = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.lidar_yaw_offset = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.lidar_z_offset = torch.zeros((n, a), dtype=torch.float32, device=self.device)

        self.pooled_lidar_angles = torch.linspace(
            -math.pi,
            math.pi,
            int(self.cfg.lidar_pool_bins) + 1,
            dtype=torch.float32,
            device=self.device,
        )[:-1]

        self.prev_lidar = torch.full(
            (n, a, int(self.cfg.lidar_pool_bins)),
            float(self.cfg.lidar_max_distance),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_lidar = self.prev_lidar.clone()
        self.last_lidar_delta = torch.zeros_like(self.prev_lidar)

        self.last_risk_features = torch.zeros(
            (n, a, self.risk_feature_dim()),
            dtype=torch.float32,
            device=self.device,
        )

        self.reset_counter = torch.zeros((n,), dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    def set_curriculum_stage(self, stage: int, env_ids: Optional[torch.Tensor] = None) -> None:
        stage = int(stage)
        if env_ids is None:
            self.curriculum_stage[:] = stage
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            self.curriculum_stage[env_ids] = stage

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_world(self, env_ids: torch.Tensor, stage: Optional[int] = None) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        if stage is not None:
            self.set_curriculum_stage(int(stage), env_ids)

        self.reset_counter[env_ids] += 1

        self._sample_domain_randomization(env_ids)
        self._sample_task_geometry(env_ids)

        # Gate must be sampled before obstacles so obstacle rejection checks the
        # current gate location.
        self._sample_gate(env_ids)
        self._sample_obstacles(env_ids)

        if bool(self.cfg.enable_physical_asset_teleport):
            self._teleport_static_obstacles(env_ids)
            self._teleport_gate_walls(env_ids)

        self.prev_lidar[env_ids] = float(self.cfg.lidar_max_distance)
        self.last_lidar[env_ids] = float(self.cfg.lidar_max_distance)
        self.last_lidar_delta[env_ids] = 0.0
        self.last_risk_features[env_ids] = 0.0

        if bool(self.cfg.debug_assert):
            self._debug_validate_reset(env_ids)

    def _sample_domain_randomization(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())
        a = self.num_agents

        self.max_speed[env_ids] = self._uniform(self.cfg.max_speed_range, (n, a))
        self.max_yaw_rate[env_ids] = self._uniform(self.cfg.max_yaw_rate_range, (n, a))

        dmin, dmax = self.cfg.action_delay_frame_range
        self.action_delay_frames[env_ids] = torch.randint(
            low=int(dmin),
            high=int(dmax) + 1,
            size=(n, a),
            dtype=torch.long,
            device=self.device,
        )

        self.action_deadband[env_ids] = self._uniform(self.cfg.action_deadband_range, (n, a))
        self.action_ema_alpha[env_ids] = self._uniform(self.cfg.action_ema_alpha_range, (n, a))

        self.motor_strength[env_ids] = self._uniform(self.cfg.motor_strength_range, (n, a, 2))
        self.motor_bias[env_ids] = self._uniform(self.cfg.motor_bias_range, (n, a, 2))
        self.wheel_radius_scale[env_ids] = self._uniform(self.cfg.wheel_radius_scale_range, (n, a, 2))

        self.lidar_noise_std[env_ids] = self._uniform(self.cfg.lidar_noise_std_range, (n, a))
        self.lidar_outlier_prob[env_ids] = self._uniform(self.cfg.lidar_outlier_prob_range, (n, a))
        self.lidar_dropout_prob[env_ids] = self._uniform(self.cfg.lidar_dropout_prob_range, (n, a))
        self.lidar_yaw_offset[env_ids] = self._uniform(self.cfg.lidar_yaw_offset_range, (n, a))
        self.lidar_z_offset[env_ids] = self._uniform(self.cfg.lidar_z_offset_range, (n, a))

    def _sample_task_geometry(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())
        stage = self.curriculum_stage[env_ids]

        self.start_center[env_ids, 0] = self._uniform(self.cfg.start_center_x_range, (n,))
        self.start_center[env_ids, 1] = self._uniform(self.cfg.start_center_y_range, (n,))
        self.start_heading[env_ids] = self._uniform(self.cfg.start_heading_range, (n,))

        self.goal_pos[env_ids, 0] = self._uniform(self.cfg.goal_x_range, (n,))
        self.goal_pos[env_ids, 1] = self._uniform(self.cfg.goal_y_range, (n,))
        self.goal_yaw[env_ids] = self._uniform(self.cfg.goal_yaw_range, (n,))

        ftype = torch.full((n,), 0, dtype=torch.long, device=self.device)

        random_dw = torch.randint(0, 2, (n,), dtype=torch.long, device=self.device)
        ftype = torch.where(stage >= 2, random_dw, ftype)

        ftype = torch.where(stage == 3, torch.full_like(ftype, 2), ftype)

        random_all = torch.randint(0, int(self.cfg.num_formation_types), (n,), dtype=torch.long, device=self.device)
        ftype = torch.where(stage >= 4, random_all, ftype)

        self.formation_type[env_ids] = ftype

        scale = self._uniform(self.cfg.formation_scale_range, (n,))
        scale = torch.where(
            stage >= 3,
            torch.minimum(scale, torch.full_like(scale, float(self.cfg.formation_scale_gate))),
            scale,
        )
        self.formation_scale[env_ids] = scale

        start_slots = self.compute_formation_slots(
            center_xy=self.start_center[env_ids],
            heading=self.start_heading[env_ids],
            formation_type=ftype,
            scale=scale,
        )

        self.start_pos[env_ids] = start_slots
        self.start_yaw[env_ids] = self.start_heading[env_ids].unsqueeze(-1).expand(n, self.num_agents)

    def _sample_gate(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())
        stage = self.curriculum_stage[env_ids]

        active = stage >= int(self.cfg.gate_stage_start)
        self.gate_active[env_ids] = active
        self.gate_x[env_ids] = self._uniform(self.cfg.gate_x_range, (n,))
        self.gate_gap_width[env_ids] = float(self.cfg.gate_gap_width)

    def _sample_obstacles(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())
        stage = self.curriculum_stage[env_ids]

        max_k = int(self.cfg.max_static_obstacles)
        half_x, half_y = self.cfg.obstacle_half_extents_nominal

        self.obstacle_pos[env_ids] = 0.0
        self.obstacle_half_extents[env_ids, :, 0] = float(half_x)
        self.obstacle_half_extents[env_ids, :, 1] = float(half_y)
        self.obstacle_active[env_ids] = False

        count_table = torch.tensor(
            self.cfg.obstacle_count_by_stage,
            dtype=torch.long,
            device=self.device,
        )
        clipped_stage = torch.clamp(stage, 0, len(self.cfg.obstacle_count_by_stage) - 1)
        counts = count_table[clipped_stage]

        for k in range(max_k):
            active_k = k < counts
            self.obstacle_active[env_ids, k] = active_k

            pos = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
            pos[:, 0] = self._uniform(self.cfg.obstacle_x_range, (n,))
            pos[:, 1] = self._uniform(self.cfg.obstacle_y_range, (n,))

            for _ in range(16):
                bad = self._obstacle_sample_bad(pos, env_ids)
                if not bad.any():
                    break
                num_bad = int(bad.sum().item())
                pos[bad, 0] = self._uniform(self.cfg.obstacle_x_range, (num_bad,))
                pos[bad, 1] = self._uniform(self.cfg.obstacle_y_range, (num_bad,))

            self.obstacle_pos[env_ids, k] = pos

    def _obstacle_sample_bad(self, pos: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        start_d = torch.norm(pos - self.start_center[env_ids], dim=-1)
        goal_d = torch.norm(pos - self.goal_pos[env_ids], dim=-1)

        bad = (
            (start_d < float(self.cfg.obstacle_clearance_start))
            | (goal_d < float(self.cfg.obstacle_clearance_goal))
        )

        gate_stage = self.curriculum_stage[env_ids] >= int(self.cfg.gate_stage_start)
        gate_x = self.gate_x[env_ids]
        near_gate_x = torch.abs(pos[:, 0] - gate_x) < 0.85
        near_gate_gap = torch.abs(pos[:, 1]) < (
            0.5 * float(self.cfg.gate_gap_width) + float(self.cfg.obstacle_clearance_gate)
        )
        bad = bad | (gate_stage & near_gate_x & near_gate_gap)
        return bad

    def _debug_validate_reset(self, env_ids: torch.Tensor) -> None:
        tensors = [
            self.start_center[env_ids],
            self.start_heading[env_ids],
            self.start_pos[env_ids],
            self.start_yaw[env_ids],
            self.goal_pos[env_ids],
            self.goal_yaw[env_ids],
            self.formation_scale[env_ids],
            self.obstacle_pos[env_ids],
            self.obstacle_half_extents[env_ids],
            self.gate_x[env_ids],
            self.max_speed[env_ids],
            self.max_yaw_rate[env_ids],
            self.motor_strength[env_ids],
            self.motor_bias[env_ids],
            self.wheel_radius_scale[env_ids],
        ]
        for x in tensors:
            assert torch.isfinite(x).all().item(), "Task4World reset generated NaN/Inf"

    # ------------------------------------------------------------------
    # Teleport physical assets
    # ------------------------------------------------------------------
    def _teleport_static_obstacles(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        env_origins = self.scene.env_origins[env_ids]
        n = int(env_ids.numel())
        zeros = torch.zeros((n,), dtype=torch.float32, device=self.device)

        for k, obj in enumerate(self.static_obstacles):
            state = obj.data.default_root_state[env_ids].clone()

            active = self.obstacle_active[env_ids, k]
            pos = self.obstacle_pos[env_ids, k]

            state[:, 0] = env_origins[:, 0] + pos[:, 0]
            state[:, 1] = env_origins[:, 1] + pos[:, 1]
            state[:, 2] = torch.where(
                active,
                env_origins[:, 2] + float(self.cfg.obstacle_height) * 0.5,
                env_origins[:, 2] - 10.0,
            )
            state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, zeros)
            state[:, 7:13] = 0.0

            obj.write_root_state_to_sim(state, env_ids=env_ids)

    def _teleport_gate_walls(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        env_origins = self.scene.env_origins[env_ids]
        n = int(env_ids.numel())
        zeros = torch.zeros((n,), dtype=torch.float32, device=self.device)

        top_state = self.gate_top.data.default_root_state[env_ids].clone()
        bottom_state = self.gate_bottom.data.default_root_state[env_ids].clone()

        active = self.gate_active[env_ids]
        gate_x = self.gate_x[env_ids]

        top_state[:, 0] = env_origins[:, 0] + gate_x
        top_state[:, 1] = env_origins[:, 1] + float(self.cfg.gate_top_center_y)
        top_state[:, 2] = torch.where(
            active,
            env_origins[:, 2] + float(self.cfg.gate_wall_height) * 0.5,
            env_origins[:, 2] - 10.0,
        )
        top_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, zeros)
        top_state[:, 7:13] = 0.0

        bottom_state[:, 0] = env_origins[:, 0] + gate_x
        bottom_state[:, 1] = env_origins[:, 1] + float(self.cfg.gate_bottom_center_y)
        bottom_state[:, 2] = torch.where(
            active,
            env_origins[:, 2] + float(self.cfg.gate_wall_height) * 0.5,
            env_origins[:, 2] - 10.0,
        )
        bottom_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, zeros)
        bottom_state[:, 7:13] = 0.0

        self.gate_top.write_root_state_to_sim(top_state, env_ids=env_ids)
        self.gate_bottom.write_root_state_to_sim(bottom_state, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Formation / team geometry
    # ------------------------------------------------------------------
    def compute_formation_slots(
        self,
        center_xy: torch.Tensor,
        heading: torch.Tensor,
        formation_type: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        formation_type = torch.as_tensor(formation_type, dtype=torch.long, device=self.device).flatten()
        offsets = self.formation_offsets[formation_type]

        if scale is None:
            scale = torch.ones((center_xy.shape[0],), dtype=torch.float32, device=self.device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.device).flatten()

        offsets = offsets * scale[:, None, None]

        c = torch.cos(heading)
        s = torch.sin(heading)

        x = offsets[:, :, 0]
        y = offsets[:, :, 1]

        wx = c[:, None] * x - s[:, None] * y
        wy = s[:, None] * x + c[:, None] * y

        return center_xy[:, None, :] + torch.stack([wx, wy], dim=-1)

    def compute_pairwise_distances(self, root_pos_local: torch.Tensor) -> torch.Tensor:
        pairs = []
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                pairs.append(torch.norm(root_pos_local[:, i] - root_pos_local[:, j], dim=-1))
        return torch.stack(pairs, dim=-1)

    def compute_team_terms(
        self,
        root_pos_local: torch.Tensor,
        yaw: torch.Tensor,
        lin_vel: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(root_pos_local.shape[0], dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        center = root_pos_local.mean(dim=1)
        vec_goal = self.goal_pos[env_ids] - center
        center_goal_dist = torch.norm(vec_goal, dim=-1)

        team_heading = torch.atan2(vec_goal[:, 1], vec_goal[:, 0])
        team_heading = torch.where(center_goal_dist > 1.0e-5, team_heading, self.goal_yaw[env_ids])

        desired_slots = self.compute_formation_slots(
            center_xy=center,
            heading=team_heading,
            formation_type=self.formation_type[env_ids],
            scale=self.formation_scale[env_ids],
        )

        slot_error_vec = root_pos_local - desired_slots
        slot_error = torch.norm(slot_error_vec, dim=-1)
        mean_slot_error = slot_error.mean(dim=-1)
        max_slot_error = slot_error.max(dim=-1)[0]

        pair_dists = self.compute_pairwise_distances(root_pos_local)
        min_pair_dist = pair_dists.min(dim=-1)[0]

        rel_center = root_pos_local - center[:, None, :]
        team_spread = torch.sqrt(torch.mean(torch.sum(rel_center * rel_center, dim=-1), dim=-1))

        if lin_vel is None:
            speed = torch.zeros((root_pos_local.shape[0], self.num_agents), dtype=torch.float32, device=self.device)
            center_speed = torch.zeros((root_pos_local.shape[0],), dtype=torch.float32, device=self.device)
        else:
            speed = torch.norm(lin_vel[:, :, :2], dim=-1)
            center_speed = torch.norm(lin_vel[:, :, :2].mean(dim=1), dim=-1)

        heading_error = self.wrap_to_pi(team_heading[:, None] - yaw)

        return {
            "center": center,
            "vec_goal": vec_goal,
            "center_goal_dist": center_goal_dist,
            "team_heading": team_heading,
            "desired_slots": desired_slots,
            "slot_error_vec": slot_error_vec,
            "slot_error": slot_error,
            "mean_slot_error": mean_slot_error,
            "max_slot_error": max_slot_error,
            "pair_dists": pair_dists,
            "min_pair_dist": min_pair_dist,
            "team_spread": team_spread,
            "speed": speed,
            "center_speed": center_speed,
            "heading_error": heading_error,
        }

    # ------------------------------------------------------------------
    # Events / collisions
    # ------------------------------------------------------------------
    def check_events(
        self,
        root_pos_local: torch.Tensor,
        yaw: torch.Tensor,
        lin_vel: Optional[torch.Tensor] = None,
        ang_vel: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(root_pos_local.shape[0], dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        n = root_pos_local.shape[0]
        radius = float(self.cfg.robot_radius)

        x = root_pos_local[:, :, 0]
        y = root_pos_local[:, :, 1]

        out_of_bounds = (
            (x < float(self.cfg.x_min) + radius)
            | (x > float(self.cfg.x_max) - radius)
            | (y < float(self.cfg.y_min) + radius)
            | (y > float(self.cfg.y_max) - radius)
        )

        obstacle_collision = self.check_obstacle_collision(root_pos_local, env_ids)
        gate_collision = self.check_gate_collision(root_pos_local, env_ids)

        pair_collision_agents = torch.zeros((n, self.num_agents), dtype=torch.bool, device=self.device)
        pair_dists = self.compute_pairwise_distances(root_pos_local)
        pair_collision_any_pairs = pair_dists < (2.0 * radius + 0.03)

        pair_idx = 0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                hit = pair_collision_any_pairs[:, pair_idx]
                pair_collision_agents[:, i] |= hit
                pair_collision_agents[:, j] |= hit
                pair_idx += 1

        team = self.compute_team_terms(root_pos_local, yaw, lin_vel=lin_vel, env_ids=env_ids)

        if lin_vel is None:
            center_speed = torch.zeros((n,), dtype=torch.float32, device=self.device)
        else:
            center_speed = torch.norm(lin_vel[:, :, :2].mean(dim=1), dim=-1)

        success_candidate = (
            (team["center_goal_dist"] < float(self.cfg.goal_center_success_tol))
            & (team["mean_slot_error"] < float(self.cfg.goal_formation_success_tol))
            & (center_speed < float(self.cfg.goal_speed_success_tol))
        )

        agent_crash = out_of_bounds | obstacle_collision | gate_collision | pair_collision_agents
        crash = agent_crash.any(dim=-1)

        return {
            "success_candidate": success_candidate,
            "out_of_bounds": out_of_bounds,
            "obstacle_collision": obstacle_collision,
            "gate_collision": gate_collision,
            "pair_collision_agents": pair_collision_agents,
            "pair_collision_any": pair_collision_any_pairs.any(dim=-1),
            "agent_crash": agent_crash,
            "crash": crash,
            "center_goal_dist": team["center_goal_dist"],
            "mean_slot_error": team["mean_slot_error"],
            "max_slot_error": team["max_slot_error"],
            "min_pair_dist": team["min_pair_dist"],
            "team_spread": team["team_spread"],
            "center_speed": center_speed,
            "lane_margin": self.boundary_margin(root_pos_local),
            "obstacle_signed_distance": self.min_obstacle_signed_distance(root_pos_local, env_ids),
            "gate_signed_distance": self.gate_signed_distance(root_pos_local, env_ids),
        }

    def check_obstacle_collision(self, root_pos_local: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        signed = self.min_obstacle_signed_distance(root_pos_local, env_ids)
        return signed < float(self.cfg.robot_radius)

    def min_obstacle_signed_distance(self, root_pos_local: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        n, a, _ = root_pos_local.shape
        best_d = torch.full((n, a), float(self.cfg.lidar_max_distance), dtype=torch.float32, device=self.device)

        px = root_pos_local[:, :, 0]
        py = root_pos_local[:, :, 1]

        for k in range(int(self.cfg.max_static_obstacles)):
            active = self.obstacle_active[env_ids, k].unsqueeze(-1)
            center = self.obstacle_pos[env_ids, k]
            half = self.obstacle_half_extents[env_ids, k]

            d = self._point_rect_signed_distance(
                px,
                py,
                center[:, 0].unsqueeze(-1),
                center[:, 1].unsqueeze(-1),
                half[:, 0].unsqueeze(-1),
                half[:, 1].unsqueeze(-1),
            )
            d = torch.where(active, d, torch.full_like(d, float(self.cfg.lidar_max_distance)))
            best_d = torch.minimum(best_d, d)

        return best_d

    def check_gate_collision(self, root_pos_local: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        signed = self.gate_signed_distance(root_pos_local, env_ids)
        active = self.gate_active[env_ids].unsqueeze(-1)
        return active & (signed < float(self.cfg.robot_radius))

    def gate_signed_distance(self, root_pos_local: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        n, a, _ = root_pos_local.shape

        px = root_pos_local[:, :, 0]
        py = root_pos_local[:, :, 1]

        gate_x = self.gate_x[env_ids]
        hx = torch.full((n, 1), float(self.cfg.gate_wall_size_x) * 0.5, dtype=torch.float32, device=self.device)
        hy = torch.full((n, 1), float(self.cfg.gate_wall_length_y) * 0.5, dtype=torch.float32, device=self.device)

        top_cy = torch.full((n, 1), float(self.cfg.gate_top_center_y), dtype=torch.float32, device=self.device)
        bot_cy = torch.full((n, 1), float(self.cfg.gate_bottom_center_y), dtype=torch.float32, device=self.device)

        top = self._point_rect_signed_distance(px, py, gate_x.unsqueeze(-1), top_cy, hx, hy)
        bottom = self._point_rect_signed_distance(px, py, gate_x.unsqueeze(-1), bot_cy, hx, hy)

        signed = torch.minimum(top, bottom)
        signed = torch.where(
            self.gate_active[env_ids].unsqueeze(-1),
            signed,
            torch.full_like(signed, float(self.cfg.lidar_max_distance)),
        )
        return signed

    def boundary_margin(self, root_pos_local: torch.Tensor) -> torch.Tensor:
        x = root_pos_local[:, :, 0]
        y = root_pos_local[:, :, 1]

        mx = torch.minimum(x - float(self.cfg.x_min), float(self.cfg.x_max) - x)
        my = torch.minimum(y - float(self.cfg.y_min), float(self.cfg.y_max) - y)
        return torch.minimum(mx, my)

    def gate_progress_terms(self, root_pos_local: torch.Tensor, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(root_pos_local.shape[0], dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        center = root_pos_local.mean(dim=1)
        rel_x = center[:, 0] - self.gate_x[env_ids]

        near_gate = self.gate_active[env_ids] & (torch.abs(rel_x) < float(self.cfg.gate_region_half_x))
        passed_gate = self.gate_active[env_ids] & (rel_x > float(self.cfg.gate_pass_margin_x))
        before_gate = self.gate_active[env_ids] & (rel_x < -float(self.cfg.gate_pass_margin_x))

        return {
            "near_gate": near_gate,
            "passed_gate": passed_gate,
            "before_gate": before_gate,
            "gate_rel_x": rel_x,
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
        include_teammates: bool = True,
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(root_pos_local.shape[0], dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        n, a, _ = root_pos_local.shape
        assert a == self.num_agents, f"expected {self.num_agents} agents, got {a}"

        r = int(self.cfg.lidar_pool_bins)
        m = n * a

        origin = root_pos_local.reshape(m, 2)
        yaw_flat = yaw.reshape(m)
        env_flat = env_ids[:, None].expand(n, a).reshape(m)
        agent_flat = torch.arange(a, dtype=torch.long, device=self.device).unsqueeze(0).expand(n, a).reshape(m)

        ray_angles_w = (
            yaw_flat[:, None]
            + self.pooled_lidar_angles[None, :]
            + self.lidar_yaw_offset[env_flat, agent_flat].unsqueeze(-1)
        )
        ray_dirs = torch.stack([torch.cos(ray_angles_w), torch.sin(ray_angles_w)], dim=-1)

        lidar = torch.full((m, r), float(self.cfg.lidar_max_distance), dtype=torch.float32, device=self.device)

        zero_yaw = torch.zeros((m,), dtype=torch.float32, device=self.device)

        hx_ns = torch.full((m,), float(self.cfg.arena_length) * 0.5, dtype=torch.float32, device=self.device)
        hy_ns = torch.full((m,), float(self.cfg.wall_thickness) * 0.5, dtype=torch.float32, device=self.device)

        north_center = torch.stack(
            [
                torch.zeros((m,), dtype=torch.float32, device=self.device),
                torch.full((m,), float(self.cfg.y_max), dtype=torch.float32, device=self.device),
            ],
            dim=-1,
        )
        south_center = torch.stack(
            [
                torch.zeros((m,), dtype=torch.float32, device=self.device),
                torch.full((m,), float(self.cfg.y_min), dtype=torch.float32, device=self.device),
            ],
            dim=-1,
        )

        lidar = torch.minimum(lidar, self._ray_obb_distance(origin, ray_dirs, north_center, hx_ns, hy_ns, zero_yaw))
        lidar = torch.minimum(lidar, self._ray_obb_distance(origin, ray_dirs, south_center, hx_ns, hy_ns, zero_yaw))

        hx_ew = torch.full((m,), float(self.cfg.wall_thickness) * 0.5, dtype=torch.float32, device=self.device)
        hy_ew = torch.full((m,), float(self.cfg.arena_width) * 0.5, dtype=torch.float32, device=self.device)

        east_center = torch.stack(
            [
                torch.full((m,), float(self.cfg.x_max), dtype=torch.float32, device=self.device),
                torch.zeros((m,), dtype=torch.float32, device=self.device),
            ],
            dim=-1,
        )
        west_center = torch.stack(
            [
                torch.full((m,), float(self.cfg.x_min), dtype=torch.float32, device=self.device),
                torch.zeros((m,), dtype=torch.float32, device=self.device),
            ],
            dim=-1,
        )

        lidar = torch.minimum(lidar, self._ray_obb_distance(origin, ray_dirs, east_center, hx_ew, hy_ew, zero_yaw))
        lidar = torch.minimum(lidar, self._ray_obb_distance(origin, ray_dirs, west_center, hx_ew, hy_ew, zero_yaw))

        obs_yaw = torch.zeros((m,), dtype=torch.float32, device=self.device)
        for k in range(int(self.cfg.max_static_obstacles)):
            active = self.obstacle_active[env_flat, k]
            center = self.obstacle_pos[env_flat, k]
            half = self.obstacle_half_extents[env_flat, k]
            d = self._ray_obb_distance(origin, ray_dirs, center, half[:, 0], half[:, 1], obs_yaw)
            d = torch.where(active.unsqueeze(-1), d, torch.full_like(d, float(self.cfg.lidar_max_distance)))
            lidar = torch.minimum(lidar, d)

        gate_active = self.gate_active[env_flat]

        gate_hx = torch.full((m,), float(self.cfg.gate_wall_size_x) * 0.5, dtype=torch.float32, device=self.device)
        gate_hy = torch.full((m,), float(self.cfg.gate_wall_length_y) * 0.5, dtype=torch.float32, device=self.device)
        gate_yaw = torch.zeros((m,), dtype=torch.float32, device=self.device)

        top_center = torch.stack(
            [
                self.gate_x[env_flat],
                torch.full((m,), float(self.cfg.gate_top_center_y), dtype=torch.float32, device=self.device),
            ],
            dim=-1,
        )
        bottom_center = torch.stack(
            [
                self.gate_x[env_flat],
                torch.full((m,), float(self.cfg.gate_bottom_center_y), dtype=torch.float32, device=self.device),
            ],
            dim=-1,
        )

        d_top = self._ray_obb_distance(origin, ray_dirs, top_center, gate_hx, gate_hy, gate_yaw)
        d_bottom = self._ray_obb_distance(origin, ray_dirs, bottom_center, gate_hx, gate_hy, gate_yaw)

        d_top = torch.where(gate_active.unsqueeze(-1), d_top, torch.full_like(d_top, float(self.cfg.lidar_max_distance)))
        d_bottom = torch.where(gate_active.unsqueeze(-1), d_bottom, torch.full_like(d_bottom, float(self.cfg.lidar_max_distance)))

        lidar = torch.minimum(lidar, d_top)
        lidar = torch.minimum(lidar, d_bottom)

        if include_teammates:
            lidar_view = lidar.view(n, a, r)
            for i in range(self.num_agents):
                agent_origin = root_pos_local[:, i, :]
                agent_yaw = yaw[:, i]
                ray_angles_i = (
                    agent_yaw[:, None]
                    + self.pooled_lidar_angles[None, :]
                    + self.lidar_yaw_offset[env_ids, i].unsqueeze(-1)
                )
                ray_dirs_i = torch.stack([torch.cos(ray_angles_i), torch.sin(ray_angles_i)], dim=-1)

                lidar_i = lidar_view[:, i, :]

                for j in range(self.num_agents):
                    if i == j:
                        continue
                    d = self._ray_circle_distance(
                        ray_origin=agent_origin,
                        ray_dirs=ray_dirs_i,
                        circle_center=root_pos_local[:, j, :],
                        radius=float(self.cfg.robot_radius),
                    )
                    lidar_i = torch.minimum(lidar_i, d)

                lidar_view[:, i, :] = lidar_i

        lidar = torch.clamp(lidar, min=float(self.cfg.lidar_min_distance), max=float(self.cfg.lidar_max_distance))
        lidar = torch.nan_to_num(
            lidar,
            nan=float(self.cfg.lidar_max_distance),
            posinf=float(self.cfg.lidar_max_distance),
            neginf=float(self.cfg.lidar_max_distance),
        ).view(n, a, r)

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
                (lidar - prev) / max(float(self.cfg.lidar_max_distance), 1.0e-6),
                -1.0,
                1.0,
            )

            self.last_lidar_delta[env_ids] = delta
            self.prev_lidar[env_ids] = lidar.detach().clone()
            self.last_lidar[env_ids] = lidar.detach().clone()

        if normalize:
            return torch.clamp(lidar / float(self.cfg.lidar_max_distance), 0.0, 1.0)

        return lidar

    def process_lidar_data(
        self,
        raycaster,
        env_ids: Optional[torch.Tensor] = None,
        agent_id: int = 0,
        add_noise: bool = True,
        update_history: bool = True,
        normalize: bool = False,
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        agent_id = int(agent_id)

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
            noise_std = self.lidar_noise_std[env_ids, agent_id].unsqueeze(-1)
            outlier_prob = self.lidar_outlier_prob[env_ids, agent_id].unsqueeze(-1)
            dropout_prob = self.lidar_dropout_prob[env_ids, agent_id].unsqueeze(-1)

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
            prev = self.prev_lidar[env_ids, agent_id].clone()
            delta = torch.clamp(
                (pooled - prev) / max(float(self.cfg.lidar_max_distance), 1.0e-6),
                -1.0,
                1.0,
            )
            self.last_lidar_delta[env_ids, agent_id] = delta
            self.prev_lidar[env_ids, agent_id] = pooled.detach().clone()
            self.last_lidar[env_ids, agent_id] = pooled.detach().clone()

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
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(root_pos_local.shape[0], dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        if lidar_pooled is None:
            lidar_pooled = self.last_lidar[env_ids]

        n, a, _ = root_pos_local.shape

        angles = self.pooled_lidar_angles
        front_rad = math.radians(float(self.cfg.front_angle_deg))
        side_rad = math.radians(float(self.cfg.side_angle_deg))

        front_mask = torch.abs(angles) <= front_rad
        left_mask = (angles > front_rad) & (angles <= side_rad)
        right_mask = (angles < -front_rad) & (angles >= -side_rad)

        all_min = lidar_pooled.min(dim=-1)[0]
        front_min = lidar_pooled[:, :, front_mask].min(dim=-1)[0]
        left_min = lidar_pooled[:, :, left_mask].min(dim=-1)[0]
        right_min = lidar_pooled[:, :, right_mask].min(dim=-1)[0]

        risk_d = float(self.cfg.risk_distance)
        all_risk = torch.clamp((risk_d - all_min) / risk_d, 0.0, 1.0)
        front_risk = torch.clamp((risk_d - front_min) / risk_d, 0.0, 1.0)
        left_risk = torch.clamp((risk_d - left_min) / risk_d, 0.0, 1.0)
        right_risk = torch.clamp((risk_d - right_min) / risk_d, 0.0, 1.0)

        boundary_clearance = self.boundary_margin(root_pos_local) - float(self.cfg.robot_radius)
        boundary_risk = torch.clamp(
            (float(self.cfg.boundary_risk_distance) - boundary_clearance)
            / max(float(self.cfg.boundary_risk_distance), 1.0e-6),
            0.0,
            1.0,
        )

        obstacle_signed = self.min_obstacle_signed_distance(root_pos_local, env_ids) - float(self.cfg.robot_radius)
        obstacle_risk = torch.clamp(
            (float(self.cfg.obstacle_risk_distance) - obstacle_signed)
            / max(float(self.cfg.obstacle_risk_distance), 1.0e-6),
            0.0,
            1.0,
        )

        gate_signed = self.gate_signed_distance(root_pos_local, env_ids) - float(self.cfg.robot_radius)
        gate_risk = torch.clamp(
            (float(self.cfg.gate_risk_distance) - gate_signed)
            / max(float(self.cfg.gate_risk_distance), 1.0e-6),
            0.0,
            1.0,
        )
        gate_risk = torch.where(self.gate_active[env_ids].unsqueeze(-1), gate_risk, torch.zeros_like(gate_risk))

        pair_d = torch.cdist(root_pos_local, root_pos_local, p=2.0)
        eye = torch.eye(self.num_agents, dtype=torch.bool, device=self.device).unsqueeze(0)
        pair_d = torch.where(eye, torch.full_like(pair_d, 1.0e6), pair_d)
        nearest_teammate = pair_d.min(dim=-1)[0]
        pair_risk = torch.clamp(
            (float(self.cfg.pair_risk_distance) - nearest_teammate)
            / max(float(self.cfg.pair_risk_distance), 1.0e-6),
            0.0,
            1.0,
        )

        team = self.compute_team_terms(root_pos_local, yaw, env_ids=env_ids)

        slot_error_norm = torch.clamp(
            team["slot_error"] / max(float(self.cfg.slot_error_norm), 1.0e-6),
            0.0,
            2.0,
        ) * 0.5

        center_goal_dist_norm = torch.clamp(
            team["center_goal_dist"].unsqueeze(-1) / max(float(self.cfg.goal_dist_norm), 1.0e-6),
            0.0,
            1.0,
        ).expand(n, a)

        heading_align = 0.5 * (torch.cos(team["heading_error"]) + 1.0)
        slot_heading_error = torch.clamp(torch.abs(team["heading_error"]) / math.pi, 0.0, 1.0)

        team_spread = torch.clamp(
            team["team_spread"].unsqueeze(-1) / max(float(self.cfg.team_spread_norm), 1.0e-6),
            0.0,
            1.0,
        ).expand(n, a)

        nearest_clearance = torch.clamp(
            nearest_teammate / max(float(self.cfg.team_spread_norm), 1.0e-6),
            0.0,
            1.0,
        )

        gate_terms = self.gate_progress_terms(root_pos_local, env_ids)
        near_gate = gate_terms["near_gate"].float().unsqueeze(-1).expand(n, a)
        gate_active = self.gate_active[env_ids].float().unsqueeze(-1).expand(n, a)

        features = torch.stack(
            [
                all_risk,
                front_risk,
                left_risk,
                right_risk,
                boundary_risk,
                obstacle_risk,
                gate_risk,
                pair_risk,
                slot_error_norm,
                center_goal_dist_norm,
                heading_align,
                slot_heading_error,
                team_spread,
                nearest_clearance,
                near_gate,
                gate_active,
            ],
            dim=-1,
        )

        features = torch.nan_to_num(
            torch.clamp(features, 0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        self.last_risk_features[env_ids] = features.detach().clone()
        return features

    def compute_privileged_features(
        self,
        root_pos_local: torch.Tensor,
        yaw: torch.Tensor,
        lin_vel: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(root_pos_local.shape[0], dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        n = root_pos_local.shape[0]
        team = self.compute_team_terms(root_pos_local, yaw, lin_vel=lin_vel, env_ids=env_ids)

        goal_feats = torch.stack(
            [
                torch.clamp(self.goal_pos[env_ids, 0] / float(self.cfg.arena_x_norm), -1.5, 1.5),
                torch.clamp(self.goal_pos[env_ids, 1] / float(self.cfg.arena_y_norm), -1.5, 1.5),
                torch.sin(self.goal_yaw[env_ids]),
                torch.cos(self.goal_yaw[env_ids]),
            ],
            dim=-1,
        )

        ftype = self.formation_type[env_ids]
        f_oh = torch.zeros((n, int(self.cfg.num_formation_types)), dtype=torch.float32, device=self.device)
        f_oh.scatter_(1, ftype.unsqueeze(-1), 1.0)
        formation_feats = torch.cat(
            [
                f_oh,
                torch.clamp(self.formation_scale[env_ids].unsqueeze(-1), 0.0, 2.0),
            ],
            dim=-1,
        )

        gate_feats = torch.stack(
            [
                self.gate_active[env_ids].float(),
                torch.clamp(self.gate_x[env_ids] / float(self.cfg.arena_x_norm), -1.0, 1.0),
                torch.clamp(self.gate_gap_width[env_ids] / float(self.cfg.arena_y_norm), 0.0, 1.0),
            ],
            dim=-1,
        )

        obs_active = self.obstacle_active[env_ids].float().unsqueeze(-1)
        obs_x = torch.clamp(self.obstacle_pos[env_ids, :, 0:1] / float(self.cfg.arena_x_norm), -1.0, 1.0)
        obs_y = torch.clamp(self.obstacle_pos[env_ids, :, 1:2] / float(self.cfg.arena_y_norm), -1.0, 1.0)
        obs_size = torch.clamp(
            self.obstacle_half_extents[env_ids].mean(dim=-1, keepdim=True) / 0.5,
            0.0,
            2.0,
        )
        obs_feats = torch.cat([obs_active, obs_x, obs_y, obs_size], dim=-1).reshape(n, -1)

        if lin_vel is None:
            speed = torch.zeros((n, self.num_agents), dtype=torch.float32, device=self.device)
        else:
            speed = torch.norm(lin_vel[:, :, :2], dim=-1)

        robot_feats = torch.stack(
            [
                torch.clamp(root_pos_local[:, :, 0] / float(self.cfg.arena_x_norm), -1.5, 1.5),
                torch.clamp(root_pos_local[:, :, 1] / float(self.cfg.arena_y_norm), -1.5, 1.5),
                torch.sin(yaw),
                torch.cos(yaw),
                torch.clamp(speed / float(self.cfg.speed_norm), 0.0, 2.0),
            ],
            dim=-1,
        ).reshape(n, -1)

        team_feats = torch.stack(
            [
                torch.clamp(team["center"][:, 0] / float(self.cfg.arena_x_norm), -1.5, 1.5),
                torch.clamp(team["center"][:, 1] / float(self.cfg.arena_y_norm), -1.5, 1.5),
                torch.clamp(team["vec_goal"][:, 0] / float(self.cfg.goal_xy_norm), -2.0, 2.0),
                torch.clamp(team["vec_goal"][:, 1] / float(self.cfg.goal_xy_norm), -2.0, 2.0),
                torch.clamp(team["center_goal_dist"] / float(self.cfg.goal_dist_norm), 0.0, 2.0),
                torch.clamp(team["mean_slot_error"] / float(self.cfg.slot_error_norm), 0.0, 2.0),
                torch.clamp(team["max_slot_error"] / float(self.cfg.slot_error_norm), 0.0, 2.0),
                torch.clamp(team["min_pair_dist"] / float(self.cfg.team_spread_norm), 0.0, 2.0),
                torch.clamp(team["team_spread"] / float(self.cfg.team_spread_norm), 0.0, 2.0),
            ],
            dim=-1,
        )

        delay_norm = self.action_delay_frames[env_ids].float() / max(float(self.cfg.action_delay_frame_range[1]), 1.0)
        deadband_norm = self.action_deadband[env_ids] / max(float(self.cfg.action_deadband_range[1]), 1.0e-6)

        domain_feats = torch.stack(
            [
                torch.clamp(self.max_speed[env_ids] / max(float(self.cfg.max_speed_range[1]), 1.0e-6), 0.0, 2.0),
                torch.clamp(delay_norm, 0.0, 1.0),
                torch.clamp(deadband_norm, 0.0, 1.0),
                torch.clamp(self.action_ema_alpha[env_ids], 0.0, 1.0),
                torch.clamp(self.motor_strength[env_ids, :, 0], 0.0, 2.0),
                torch.clamp(self.motor_strength[env_ids, :, 1], 0.0, 2.0),
            ],
            dim=-1,
        ).reshape(n, -1)

        priv = torch.cat(
            [
                goal_feats,
                formation_feats,
                gate_feats,
                obs_feats,
                robot_feats,
                team_feats,
                domain_feats,
            ],
            dim=-1,
        )

        expected = self.privileged_feature_dim(
            max_static_obstacles=int(self.cfg.max_static_obstacles),
            num_agents=int(self.cfg.num_agents),
        )
        if priv.shape[-1] != expected:
            raise RuntimeError(f"privileged feature dim mismatch: {priv.shape[-1]} != {expected}")

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
            "Stage": self.curriculum_stage.float().mean().item(),
            "Start_Center_X": self.start_center[:, 0].mean().item(),
            "Start_Center_Y": self.start_center[:, 1].mean().item(),
            "Goal_X": self.goal_pos[:, 0].mean().item(),
            "Goal_Y": self.goal_pos[:, 1].mean().item(),
            "Formation_Type": self.formation_type.float().mean().item(),
            "Formation_Scale": self.formation_scale.mean().item(),
            "Obstacle_Count": self.obstacle_active.float().sum(dim=-1).mean().item(),
            "Gate_Active": self.gate_active.float().mean().item(),
            "Gate_X": self.gate_x.mean().item(),
            "Max_Speed": self.max_speed.mean().item(),
            "Action_Delay": self.action_delay_frames.float().mean().item(),
            "Action_Deadband": self.action_deadband.mean().item(),
            "Action_EMA": self.action_ema_alpha.mean().item(),
            "Motor_Strength": self.motor_strength.mean().item(),
            "Wheel_Radius_Scale": self.wheel_radius_scale.mean().item(),
            "Lidar_Noise_Std": self.lidar_noise_std.mean().item(),
            "Lidar_Min": self.last_lidar.min().item(),
            "Lidar_Mean": self.last_lidar.mean().item(),
            "Risk_All": self.last_risk_features[:, :, 0].mean().item(),
            "Risk_Front": self.last_risk_features[:, :, 1].mean().item(),
            "Risk_Obstacle": self.last_risk_features[:, :, 5].mean().item(),
            "Risk_Gate": self.last_risk_features[:, :, 6].mean().item(),
            "Risk_Pair": self.last_risk_features[:, :, 7].mean().item(),
        }

        if root_pos_local is not None and yaw is not None:
            team = self.compute_team_terms(root_pos_local, yaw)
            events = self.check_events(root_pos_local, yaw)
            gate = self.gate_progress_terms(root_pos_local)

            stats.update(
                {
                    "Center_Goal_Dist": team["center_goal_dist"].mean().item(),
                    "Mean_Slot_Error": team["mean_slot_error"].mean().item(),
                    "Max_Slot_Error": team["max_slot_error"].mean().item(),
                    "Min_Pair_Dist": team["min_pair_dist"].mean().item(),
                    "Team_Spread": team["team_spread"].mean().item(),
                    "Success_Candidate": events["success_candidate"].float().mean().item(),
                    "Crash": events["crash"].float().mean().item(),
                    "Out_Of_Bounds": events["out_of_bounds"].float().mean().item(),
                    "Obstacle_Collision": events["obstacle_collision"].float().mean().item(),
                    "Gate_Collision": events["gate_collision"].float().mean().item(),
                    "Pair_Collision": events["pair_collision_any"].float().mean().item(),
                    "Near_Gate": gate["near_gate"].float().mean().item(),
                    "Passed_Gate": gate["passed_gate"].float().mean().item(),
                }
            )

        return stats

    # ------------------------------------------------------------------
    # Math utilities
    # ------------------------------------------------------------------
    def _uniform(self, rng: Tuple[float, float], shape) -> torch.Tensor:
        lo, hi = float(rng[0]), float(rng[1])
        return lo + torch.rand(shape, dtype=torch.float32, device=self.device) * (hi - lo)

    @staticmethod
    def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

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

    def _ray_circle_distance(
        self,
        ray_origin: torch.Tensor,
        ray_dirs: torch.Tensor,
        circle_center: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        # ray_origin [N,2], ray_dirs [N,R,2], circle_center [N,2]
        oc = ray_origin[:, None, :] - circle_center[:, None, :]
        b = torch.sum(oc * ray_dirs, dim=-1)
        c = torch.sum(oc * oc, dim=-1) - float(radius) * float(radius)
        disc = b * b - c

        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        t = -b - sqrt_disc

        hit = (disc >= 0.0) & (t >= 0.0)
        dist = torch.where(hit, t, torch.full_like(t, float(self.cfg.lidar_max_distance)))
        return torch.clamp(dist, min=float(self.cfg.lidar_min_distance), max=float(self.cfg.lidar_max_distance))


# ======================================================================
# Asset spawner
# ======================================================================

def spawn_world_assets(scene_cfg: InteractiveSceneCfg, cfg: Task4WorldConfig) -> None:
    """Conservative manual asset spawn + RigidObject registration."""

    cfg.validate()

    floor_cfg = sim_utils.CuboidCfg(
        size=(float(cfg.arena_length), float(cfg.arena_width), float(cfg.ground_height)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=float(cfg.mat_floor_nominal[0]),
            dynamic_friction=float(cfg.mat_floor_nominal[1]),
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
    )
    floor_cfg.func(
        "/World/envs/env_0/Floor/ArenaFloor",
        floor_cfg,
        translation=(0.0, 0.0, -float(cfg.ground_height) * 0.5),
    )

    wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.15, 0.15))

    wall_ns_cfg = sim_utils.CuboidCfg(
        size=(float(cfg.arena_length), float(cfg.wall_thickness), float(cfg.wall_height)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=wall_mat,
    )

    wall_ew_cfg = sim_utils.CuboidCfg(
        size=(float(cfg.wall_thickness), float(cfg.arena_width), float(cfg.wall_height)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=wall_mat,
    )

    z_wall = float(cfg.wall_height) * 0.5
    wall_ns_cfg.func("/World/envs/env_0/ArenaWalls/North", wall_ns_cfg, translation=(0.0, float(cfg.y_max), z_wall))
    wall_ns_cfg.func("/World/envs/env_0/ArenaWalls/South", wall_ns_cfg, translation=(0.0, float(cfg.y_min), z_wall))
    wall_ew_cfg.func("/World/envs/env_0/ArenaWalls/East", wall_ew_cfg, translation=(float(cfg.x_max), 0.0, z_wall))
    wall_ew_cfg.func("/World/envs/env_0/ArenaWalls/West", wall_ew_cfg, translation=(float(cfg.x_min), 0.0, z_wall))

    obstacle_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.55, 0.12))
    obstacle_cfg = sim_utils.CuboidCfg(
        size=(float(cfg.obstacle_size_xy[0]), float(cfg.obstacle_size_xy[1]), float(cfg.obstacle_height)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=obstacle_visual,
    )

    for i in range(int(cfg.max_static_obstacles)):
        obstacle_cfg.func(
            f"/World/envs/env_0/StaticObstacles/Obstacle_{i}",
            obstacle_cfg,
            translation=(0.0, 0.0, -10.0),
        )
        setattr(
            scene_cfg,
            f"static_obstacle_{i}",
            RigidObjectCfg(prim_path=f"{{ENV_REGEX_NS}}/StaticObstacles/Obstacle_{i}"),
        )

    gate_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.80, 0.20))
    gate_cfg = sim_utils.CuboidCfg(
        size=(float(cfg.gate_wall_size_x), float(cfg.gate_wall_length_y), float(cfg.gate_wall_height)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=gate_visual,
    )

    gate_cfg.func(
        "/World/envs/env_0/NarrowGate/Top",
        gate_cfg,
        translation=(0.0, float(cfg.gate_top_center_y), -10.0),
    )
    gate_cfg.func(
        "/World/envs/env_0/NarrowGate/Bottom",
        gate_cfg,
        translation=(0.0, float(cfg.gate_bottom_center_y), -10.0),
    )

    scene_cfg.gate_top = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/NarrowGate/Top")
    scene_cfg.gate_bottom = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/NarrowGate/Bottom")

    # Optional visual pads. These are fixed hints only; reward and reset use
    # world tensors, not these prims.
    start_pad = sim_utils.CuboidCfg(
        size=(1.6, 3.2, 0.01),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.45, 1.00), opacity=0.25),
    )
    goal_pad = sim_utils.CuboidCfg(
        size=(1.6, 4.5, 0.01),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 1.00, 0.35), opacity=0.25),
    )

    start_pad.func(
        "/World/envs/env_0/VisualPads/StartZone",
        start_pad,
        translation=(-6.5, 0.0, 0.006),
    )
    goal_pad.func(
        "/World/envs/env_0/VisualPads/GoalZone",
        goal_pad,
        translation=(6.75, 0.0, 0.006),
    )


def get_lidar_cfg(prim_path: str, cfg: Optional[Task4WorldConfig] = None) -> RayCasterCfg:
    if cfg is None:
        cfg = Task4WorldConfig()

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


JetbotTask4WorldConfig = Task4WorldConfig
JetbotTask4WorldManager = Task4WorldManager
DiffDriveTask4WorldConfig = Task4WorldConfig
DiffDriveTask4WorldManager = Task4WorldManager

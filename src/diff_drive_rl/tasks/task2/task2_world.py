from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class Task2WorldConfig:
    """Jetbot Task2 analytic GPU navigation world.

    Coordinate convention:
        All positions in this file are env-local xy coordinates.

    This world layer is responsible for:
        1. start / goal sampling
        2. static / dynamic obstacle sampling
        3. dynamic obstacle integration and boundary bounce
        4. analytic 2D LiDAR
        5. success / collision / out-of-bounds events
        6. risk and navigation features
    """

    # ------------------------------------------------------------------
    # Arena
    # ------------------------------------------------------------------
    arena_size: float = 50.0
    boundary_margin: float = 0.50

    robot_radius: float = 0.23
    success_radius: float = 0.55
    collision_margin: float = 0.03

    # ------------------------------------------------------------------
    # Obstacle sampling
    # ------------------------------------------------------------------
    max_static_obs: int = 20
    max_dynamic_obs: int = 5

    min_obs_spacing: float = 0.45
    start_goal_safe_radius: float = 1.25
    max_sampling_attempts: int = 80

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    curriculum_total_steps: int = 250_000_000

    stage_thresholds: Tuple[float, ...] = (
        0.0,
        0.08,
        0.20,
        0.38,
        0.60,
        0.80,
    )

    goal_dist_ranges: Tuple[Tuple[float, float], ...] = (
        (3.0, 6.0),
        (5.0, 10.0),
        (8.0, 15.0),
        (10.0, 20.0),
        (15.0, 30.0),
        (20.0, 42.0),
    )

    static_count_ranges: Tuple[Tuple[int, int], ...] = (
        (0, 0),
        (3, 5),
        (6, 10),
        (10, 15),
        (14, 20),
        (18, 20),
    )

    dynamic_count_ranges: Tuple[Tuple[int, int], ...] = (
        (0, 0),
        (0, 0),
        (0, 0),
        (1, 2),
        (2, 4),
        (3, 5),
    )

    target_speed_ranges: Tuple[Tuple[float, float], ...] = (
        (0.35, 0.65),
        (0.40, 0.75),
        (0.45, 0.90),
        (0.50, 1.00),
        (0.55, 1.15),
        (0.60, 1.25),
    )

    static_radius_ranges: Tuple[Tuple[float, float], ...] = (
        (0.35, 0.60),
        (0.35, 0.80),
        (0.40, 1.00),
        (0.45, 1.15),
        (0.50, 1.30),
        (0.50, 1.50),
    )

    dynamic_radius_ranges: Tuple[Tuple[float, float], ...] = (
        (0.30, 0.45),
        (0.30, 0.45),
        (0.30, 0.55),
        (0.35, 0.65),
        (0.40, 0.75),
        (0.45, 0.85),
    )

    dynamic_speed_ranges: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.20, 0.45),
        (0.25, 0.60),
        (0.30, 0.75),
    )

    # ------------------------------------------------------------------
    # LiDAR / risk features
    # ------------------------------------------------------------------
    num_lidar_rays: int = 72
    lidar_max_distance: float = 10.0
    lidar_min_distance: float = 0.02

    front_angle_deg: float = 35.0
    side_angle_deg: float = 110.0

    risk_distance: float = 1.80
    ttc_distance: float = 2.20
    ttc_speed_scale: float = 0.80

    # ------------------------------------------------------------------
    # Observation normalization helper constants
    # ------------------------------------------------------------------
    goal_dist_norm: float = 42.0
    goal_xy_norm: float = 10.0

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    debug_assert: bool = False

    @property
    def half_extent(self) -> float:
        return 0.5 * float(self.arena_size) - float(self.boundary_margin)

    @property
    def num_stages(self) -> int:
        return len(self.stage_thresholds)

    def validate(self) -> None:
        assert self.arena_size > 0.0
        assert self.boundary_margin >= 0.0
        assert self.half_extent > 0.0

        assert self.robot_radius > 0.0
        assert self.success_radius > 0.0
        assert self.collision_margin >= 0.0

        assert self.max_static_obs >= 0
        assert self.max_dynamic_obs >= 0
        assert self.max_static_obs + self.max_dynamic_obs > 0
        assert self.min_obs_spacing >= 0.0
        assert self.start_goal_safe_radius > 0.0
        assert self.max_sampling_attempts > 0

        assert self.curriculum_total_steps > 0
        assert len(self.stage_thresholds) >= 1
        assert abs(float(self.stage_thresholds[0]) - 0.0) < 1e-8
        assert all(self.stage_thresholds[i] <= self.stage_thresholds[i + 1] for i in range(len(self.stage_thresholds) - 1))
        assert all(0.0 <= float(x) <= 1.0 for x in self.stage_thresholds)

        expected = self.num_stages
        for name in [
            "goal_dist_ranges",
            "static_count_ranges",
            "dynamic_count_ranges",
            "target_speed_ranges",
            "static_radius_ranges",
            "dynamic_radius_ranges",
            "dynamic_speed_ranges",
        ]:
            value = getattr(self, name)
            assert len(value) == expected, f"{name} length mismatch: {len(value)} != {expected}"
            for lo, hi in value:
                assert hi >= lo, f"{name} invalid range: {(lo, hi)}"

        for lo, hi in self.goal_dist_ranges:
            assert lo > 0.0 and hi <= self.arena_size

        for lo, hi in self.static_count_ranges:
            assert 0 <= lo <= hi <= self.max_static_obs

        for lo, hi in self.dynamic_count_ranges:
            assert 0 <= lo <= hi <= self.max_dynamic_obs

        for lo, hi in self.target_speed_ranges:
            assert lo >= 0.0 and hi > 0.0

        for lo, hi in self.static_radius_ranges:
            assert lo > 0.0 and hi > 0.0

        for lo, hi in self.dynamic_radius_ranges:
            assert lo > 0.0 and hi > 0.0

        for lo, hi in self.dynamic_speed_ranges:
            assert lo >= 0.0 and hi >= 0.0

        assert self.num_lidar_rays >= 8
        assert self.lidar_max_distance > self.lidar_min_distance > 0.0

        assert self.front_angle_deg > 0.0
        assert self.side_angle_deg > self.front_angle_deg
        assert self.side_angle_deg <= 180.0

        assert self.risk_distance > 0.0
        assert self.ttc_distance > 0.0
        assert self.ttc_speed_scale > 0.0

        assert self.goal_dist_norm > 0.0
        assert self.goal_xy_norm > 0.0


class Task2WorldManager:
    """Pure analytic GPU world manager for Jetbot Task2.

    All tensors live on cfg.device selected by the caller.
    No Isaac prims are created here.
    """

    def __init__(self, cfg: Task2WorldConfig, num_envs: int, device: str):
        cfg.validate()

        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = str(device)

        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")

        self.env_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Navigation target tensors.
        self.start_pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.goal_pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.goal_distance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.env_target_speed = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Static obstacle tensors.
        self.static_pos = torch.zeros(
            (self.num_envs, self.cfg.max_static_obs, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.static_radius = torch.zeros(
            (self.num_envs, self.cfg.max_static_obs),
            dtype=torch.float32,
            device=self.device,
        )
        self.static_mask = torch.zeros(
            (self.num_envs, self.cfg.max_static_obs),
            dtype=torch.bool,
            device=self.device,
        )

        # Dynamic obstacle tensors.
        self.dynamic_pos = torch.zeros(
            (self.num_envs, self.cfg.max_dynamic_obs, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.dynamic_vel = torch.zeros(
            (self.num_envs, self.cfg.max_dynamic_obs, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.dynamic_radius = torch.zeros(
            (self.num_envs, self.cfg.max_dynamic_obs),
            dtype=torch.float32,
            device=self.device,
        )
        self.dynamic_mask = torch.zeros(
            (self.num_envs, self.cfg.max_dynamic_obs),
            dtype=torch.bool,
            device=self.device,
        )

        # LiDAR history.
        self.lidar_angles = torch.linspace(
            -math.pi,
            math.pi,
            self.cfg.num_lidar_rays + 1,
            dtype=torch.float32,
            device=self.device,
        )[:-1]

        self.prev_lidar_dist = torch.full(
            (self.num_envs, self.cfg.num_lidar_rays),
            float(self.cfg.lidar_max_distance),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_lidar_dist = self.prev_lidar_dist.clone()
        self.last_lidar_delta = torch.zeros_like(self.last_lidar_dist)
        self.last_risk_features = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.device)

        # Debug counters.
        self.reset_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    def curriculum_k(self, global_steps: int) -> float:
        return min(1.0, max(0.0, float(global_steps) / max(float(self.cfg.curriculum_total_steps), 1.0)))

    def stage_from_progress(self, k: float) -> int:
        k = min(1.0, max(0.0, float(k)))
        stage = 0

        for i, th in enumerate(self.cfg.stage_thresholds):
            if k >= float(th):
                stage = i

        return int(min(stage, self.cfg.num_stages - 1))

    def stage_from_global_steps(self, global_steps: int) -> int:
        return self.stage_from_progress(self.curriculum_k(global_steps))

    def _stage_float_range(
        self,
        ranges: Tuple[Tuple[float, float], ...],
        stages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lo_table = torch.tensor([r[0] for r in ranges], dtype=torch.float32, device=self.device)
        hi_table = torch.tensor([r[1] for r in ranges], dtype=torch.float32, device=self.device)
        return lo_table[stages], hi_table[stages]

    def _stage_int_range(
        self,
        ranges: Tuple[Tuple[int, int], ...],
        stages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lo_table = torch.tensor([r[0] for r in ranges], dtype=torch.long, device=self.device)
        hi_table = torch.tensor([r[1] for r in ranges], dtype=torch.long, device=self.device)
        return lo_table[stages], hi_table[stages]

    def _sample_stage_float(
        self,
        ranges: Tuple[Tuple[float, float], ...],
        stages: torch.Tensor,
    ) -> torch.Tensor:
        lo, hi = self._stage_float_range(ranges, stages)
        return lo + torch.rand_like(lo) * (hi - lo)

    def _sample_stage_int(
        self,
        ranges: Tuple[Tuple[int, int], ...],
        stages: torch.Tensor,
    ) -> torch.Tensor:
        lo, hi = self._stage_int_range(ranges, stages)
        span = torch.clamp(hi - lo + 1, min=1)
        return lo + torch.floor(torch.rand(lo.shape, dtype=torch.float32, device=self.device) * span.float()).long()

    # ------------------------------------------------------------------
    # Reset / sampling
    # ------------------------------------------------------------------
    def reset(self, env_ids: Optional[torch.Tensor] = None, global_steps: int = 0) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        if env_ids.numel() == 0:
            return

        stage = self.stage_from_global_steps(global_steps)
        stages = torch.full((int(env_ids.numel()),), stage, dtype=torch.long, device=self.device)

        self.env_stage[env_ids] = stages
        self.reset_counter[env_ids] += 1

        self._sample_start_goal(env_ids, stages)
        self._sample_obstacle_layout(env_ids, stages)
        self._sample_dynamic_velocity(env_ids, stages)

        self.prev_lidar_dist[env_ids] = float(self.cfg.lidar_max_distance)
        self.last_lidar_dist[env_ids] = float(self.cfg.lidar_max_distance)
        self.last_lidar_delta[env_ids] = 0.0
        self.last_risk_features[env_ids] = 0.0

        if bool(self.cfg.debug_assert):
            self._debug_validate_reset(env_ids)

    def _sample_start_goal(self, env_ids: torch.Tensor, stages: torch.Tensor) -> None:
        n = int(env_ids.numel())

        d = self._sample_stage_float(self.cfg.goal_dist_ranges, stages)
        theta = torch.rand(n, dtype=torch.float32, device=self.device) * 2.0 * math.pi

        direction = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

        half = float(self.cfg.half_extent)
        safety = float(self.cfg.start_goal_safe_radius + self.cfg.robot_radius + 0.20)

        max_mid_x = half - safety - torch.abs(direction[:, 0]) * d * 0.5
        max_mid_y = half - safety - torch.abs(direction[:, 1]) * d * 0.5

        max_mid_x = torch.clamp(max_mid_x, min=0.0)
        max_mid_y = torch.clamp(max_mid_y, min=0.0)

        midpoint = torch.empty((n, 2), dtype=torch.float32, device=self.device)
        midpoint[:, 0] = (torch.rand(n, dtype=torch.float32, device=self.device) * 2.0 - 1.0) * max_mid_x
        midpoint[:, 1] = (torch.rand(n, dtype=torch.float32, device=self.device) * 2.0 - 1.0) * max_mid_y

        start = midpoint - 0.5 * d.unsqueeze(-1) * direction
        goal = midpoint + 0.5 * d.unsqueeze(-1) * direction

        self.start_pos[env_ids] = start
        self.goal_pos[env_ids] = goal
        self.goal_distance[env_ids] = d
        self.env_target_speed[env_ids] = self._sample_stage_float(self.cfg.target_speed_ranges, stages)

    def _sample_obstacle_layout(self, env_ids: torch.Tensor, stages: torch.Tensor) -> None:
        n = int(env_ids.numel())
        max_static = int(self.cfg.max_static_obs)
        max_dynamic = int(self.cfg.max_dynamic_obs)
        total = max_static + max_dynamic

        static_counts = self._sample_stage_int(self.cfg.static_count_ranges, stages)
        dynamic_counts = self._sample_stage_int(self.cfg.dynamic_count_ranges, stages)

        static_arange = torch.arange(max_static, device=self.device).unsqueeze(0)
        dynamic_arange = torch.arange(max_dynamic, device=self.device).unsqueeze(0)

        static_mask = static_arange < static_counts.unsqueeze(-1)
        dynamic_mask = dynamic_arange < dynamic_counts.unsqueeze(-1)

        static_radius = torch.zeros((n, max_static), dtype=torch.float32, device=self.device)
        dynamic_radius = torch.zeros((n, max_dynamic), dtype=torch.float32, device=self.device)

        if max_static > 0:
            lo, hi = self._stage_float_range(self.cfg.static_radius_ranges, stages)
            radius = lo.unsqueeze(-1) + torch.rand((n, max_static), dtype=torch.float32, device=self.device) * (
                hi - lo
            ).unsqueeze(-1)
            static_radius = torch.where(static_mask, radius, torch.zeros_like(radius))

        if max_dynamic > 0:
            lo, hi = self._stage_float_range(self.cfg.dynamic_radius_ranges, stages)
            radius = lo.unsqueeze(-1) + torch.rand((n, max_dynamic), dtype=torch.float32, device=self.device) * (
                hi - lo
            ).unsqueeze(-1)
            dynamic_radius = torch.where(dynamic_mask, radius, torch.zeros_like(radius))

        combined_mask = torch.cat([static_mask, dynamic_mask], dim=-1)
        combined_radius = torch.cat([static_radius, dynamic_radius], dim=-1)
        combined_pos = torch.zeros((n, total, 2), dtype=torch.float32, device=self.device)

        half = float(self.cfg.half_extent)

        for i in range(total):
            active = combined_mask[:, i]
            if not active.any():
                continue

            valid = ~active
            radius_i = combined_radius[:, i]

            for _ in range(int(self.cfg.max_sampling_attempts)):
                invalid = (~valid).nonzero(as_tuple=False).squeeze(-1)
                if invalid.numel() == 0:
                    break

                r = radius_i[invalid]
                bound = torch.clamp(
                    half - r - float(self.cfg.robot_radius) - 0.10,
                    min=0.50,
                )

                cand = torch.empty((int(invalid.numel()), 2), dtype=torch.float32, device=self.device)
                cand[:, 0] = (torch.rand(int(invalid.numel()), dtype=torch.float32, device=self.device) * 2.0 - 1.0) * bound
                cand[:, 1] = (torch.rand(int(invalid.numel()), dtype=torch.float32, device=self.device) * 2.0 - 1.0) * bound

                start = self.start_pos[env_ids[invalid]]
                goal = self.goal_pos[env_ids[invalid]]

                safe_start = torch.norm(cand - start, dim=-1) > (
                    float(self.cfg.start_goal_safe_radius) + float(self.cfg.robot_radius) + r
                )
                safe_goal = torch.norm(cand - goal, dim=-1) > (
                    float(self.cfg.start_goal_safe_radius) + float(self.cfg.robot_radius) + r
                )

                ok = safe_start & safe_goal

                if i > 0:
                    prev_pos = combined_pos[invalid, :i, :]
                    prev_rad = combined_radius[invalid, :i]
                    prev_mask = combined_mask[invalid, :i]

                    dist_prev = torch.norm(cand.unsqueeze(1) - prev_pos, dim=-1)
                    required = r.unsqueeze(-1) + prev_rad + float(self.cfg.min_obs_spacing) + float(self.cfg.robot_radius)

                    sep_ok = torch.where(
                        prev_mask,
                        dist_prev > required,
                        torch.ones_like(prev_mask, dtype=torch.bool),
                    )
                    ok = ok & sep_ok.all(dim=-1)

                good_invalid = invalid[ok]
                if good_invalid.numel() > 0:
                    combined_pos[good_invalid, i, :] = cand[ok]
                    valid[good_invalid] = True

            remaining = (~valid).nonzero(as_tuple=False).squeeze(-1)
            if remaining.numel() > 0:
                # Fallback should be rare. It keeps tensors finite and inside arena.
                angle = torch.rand(int(remaining.numel()), dtype=torch.float32, device=self.device) * 2.0 * math.pi
                r_fallback = torch.clamp(
                    half * 0.50 + torch.rand(int(remaining.numel()), dtype=torch.float32, device=self.device) * half * 0.35,
                    max=half - 1.0,
                )
                combined_pos[remaining, i, 0] = r_fallback * torch.cos(angle)
                combined_pos[remaining, i, 1] = r_fallback * torch.sin(angle)

        self.static_pos[env_ids] = combined_pos[:, :max_static, :]
        self.dynamic_pos[env_ids] = combined_pos[:, max_static:, :]

        self.static_radius[env_ids] = static_radius
        self.dynamic_radius[env_ids] = dynamic_radius

        self.static_mask[env_ids] = static_mask
        self.dynamic_mask[env_ids] = dynamic_mask

    def _sample_dynamic_velocity(self, env_ids: torch.Tensor, stages: torch.Tensor) -> None:
        n = int(env_ids.numel())
        max_dynamic = int(self.cfg.max_dynamic_obs)

        speed = self._sample_stage_float(self.cfg.dynamic_speed_ranges, stages).unsqueeze(-1)
        angle = torch.rand((n, max_dynamic), dtype=torch.float32, device=self.device) * 2.0 * math.pi

        vel = torch.zeros((n, max_dynamic, 2), dtype=torch.float32, device=self.device)
        vel[:, :, 0] = torch.cos(angle) * speed
        vel[:, :, 1] = torch.sin(angle) * speed

        vel = torch.where(self.dynamic_mask[env_ids].unsqueeze(-1), vel, torch.zeros_like(vel))
        self.dynamic_vel[env_ids] = vel

    def _debug_validate_reset(self, env_ids: torch.Tensor) -> None:
        half = float(self.cfg.half_extent)

        tensors = [
            self.start_pos[env_ids],
            self.goal_pos[env_ids],
            self.static_pos[env_ids],
            self.dynamic_pos[env_ids],
            self.static_radius[env_ids],
            self.dynamic_radius[env_ids],
            self.dynamic_vel[env_ids],
        ]

        for x in tensors:
            assert torch.isfinite(x).all().item(), "Task2World reset generated NaN/Inf"

        assert self.start_pos[env_ids].abs().max().item() <= half + 1e-4
        assert self.goal_pos[env_ids].abs().max().item() <= half + 1e-4

    # ------------------------------------------------------------------
    # Dynamic obstacles
    # ------------------------------------------------------------------
    def step_dynamic_obstacles(self, dt: float) -> None:
        if int(self.cfg.max_dynamic_obs) <= 0:
            return

        active = self.dynamic_mask.unsqueeze(-1)

        self.dynamic_pos = torch.where(
            active,
            self.dynamic_pos + self.dynamic_vel * float(dt),
            self.dynamic_pos,
        )

        half = float(self.cfg.half_extent)

        x_limit = half - self.dynamic_radius
        y_limit = half - self.dynamic_radius

        bounce_x = (self.dynamic_pos[:, :, 0] > x_limit) | (self.dynamic_pos[:, :, 0] < -x_limit)
        bounce_y = (self.dynamic_pos[:, :, 1] > y_limit) | (self.dynamic_pos[:, :, 1] < -y_limit)

        self.dynamic_vel[:, :, 0] = torch.where(
            bounce_x & self.dynamic_mask,
            -self.dynamic_vel[:, :, 0],
            self.dynamic_vel[:, :, 0],
        )
        self.dynamic_vel[:, :, 1] = torch.where(
            bounce_y & self.dynamic_mask,
            -self.dynamic_vel[:, :, 1],
            self.dynamic_vel[:, :, 1],
        )

        self.dynamic_pos[:, :, 0] = torch.minimum(torch.maximum(self.dynamic_pos[:, :, 0], -x_limit), x_limit)
        self.dynamic_pos[:, :, 1] = torch.minimum(torch.maximum(self.dynamic_pos[:, :, 1], -y_limit), y_limit)

    # ------------------------------------------------------------------
    # Goal / navigation features
    # ------------------------------------------------------------------
    def compute_goal_terms(self, root_pos_local: torch.Tensor, yaw: torch.Tensor) -> Dict[str, torch.Tensor]:
        root_xy = root_pos_local[:, :2]

        vec_w = self.goal_pos - root_xy
        dist = torch.norm(vec_w, dim=-1)

        goal_angle = torch.atan2(vec_w[:, 1], vec_w[:, 0])
        heading_error = torch.atan2(
            torch.sin(goal_angle - yaw),
            torch.cos(goal_angle - yaw),
        )

        vec_b = self.rotate_world_to_body_2d(vec_w, yaw)

        return {
            "goal_vec_w": vec_w,
            "goal_vec_b": vec_b,
            "goal_dist": dist,
            "heading_error": heading_error,
            "heading_sin": torch.sin(heading_error),
            "heading_cos": torch.cos(heading_error),
            "goal_x_body_norm": torch.clamp(vec_b[:, 0] / float(self.cfg.goal_xy_norm), -5.0, 5.0),
            "goal_y_body_norm": torch.clamp(vec_b[:, 1] / float(self.cfg.goal_xy_norm), -5.0, 5.0),
            "goal_dist_norm": torch.clamp(dist / float(self.cfg.goal_dist_norm), 0.0, 5.0),
        }

    def compute_navigation_features(
        self,
        root_pos_local: torch.Tensor,
        yaw: torch.Tensor,
        body_vx: Optional[torch.Tensor] = None,
        update_lidar_history: bool = True,
    ) -> Dict[str, torch.Tensor]:
        goal = self.compute_goal_terms(root_pos_local, yaw)

        lidar_dist = self.compute_lidar_distances(
            root_pos_local=root_pos_local,
            yaw=yaw,
            update_history=update_lidar_history,
        )

        lidar_norm = torch.clamp(lidar_dist / float(self.cfg.lidar_max_distance), 0.0, 1.0)
        lidar_delta = self.last_lidar_delta

        risk = self.compute_risk_features(
            root_pos_local=root_pos_local,
            yaw=yaw,
            body_vx=body_vx,
            lidar_dist=lidar_dist,
        )

        return {
            **goal,
            "lidar_dist": lidar_dist,
            "lidar_norm": lidar_norm,
            "lidar_delta": lidar_delta,
            "risk_features": risk,
        }

    # ------------------------------------------------------------------
    # Analytic 2D LiDAR
    # ------------------------------------------------------------------
    def compute_lidar_distances(
        self,
        root_pos_local: torch.Tensor,
        yaw: torch.Tensor,
        update_history: bool = True,
    ) -> torch.Tensor:
        """Analytic 2D LiDAR distances in meters."""

        n = root_pos_local.shape[0]
        r = int(self.cfg.num_lidar_rays)

        if n != self.num_envs:
            raise ValueError(
                f"Task2WorldManager expects root_pos_local with num_envs={self.num_envs}, got {n}"
            )

        root_xy = root_pos_local[:, :2]

        angles_w = yaw.unsqueeze(-1) + self.lidar_angles.unsqueeze(0)
        dirs = torch.stack([torch.cos(angles_w), torch.sin(angles_w)], dim=-1)

        obs_pos = torch.cat([self.static_pos, self.dynamic_pos], dim=1)
        obs_radius = torch.cat([self.static_radius, self.dynamic_radius], dim=1)
        obs_mask = torch.cat([self.static_mask, self.dynamic_mask], dim=1)

        max_dist = float(self.cfg.lidar_max_distance)

        if obs_pos.shape[1] > 0:
            rel = obs_pos[:, None, :, :] - root_xy[:, None, None, :]
            b = torch.sum(rel * dirs[:, :, None, :], dim=-1)
            c2 = torch.sum(rel * rel, dim=-1)

            radius = obs_radius[:, None, :]
            disc = radius * radius - (c2 - b * b)

            valid = obs_mask[:, None, :] & (disc >= 0.0) & (b > 0.0)

            t = b - torch.sqrt(torch.clamp(disc, min=0.0))
            valid = valid & (t > float(self.cfg.lidar_min_distance))

            t = torch.where(valid, t, torch.full_like(t, max_dist))
            obs_hit = t.min(dim=-1)[0]
        else:
            obs_hit = torch.full((n, r), max_dist, dtype=torch.float32, device=self.device)

        boundary_hit = self._ray_square_boundary_distance(root_xy, dirs)
        lidar = torch.minimum(obs_hit, boundary_hit)
        lidar = torch.clamp(lidar, min=float(self.cfg.lidar_min_distance), max=max_dist)
        lidar = torch.nan_to_num(lidar, nan=max_dist, posinf=max_dist, neginf=max_dist)

        if update_history:
            prev = self.prev_lidar_dist.clone()
            self.last_lidar_delta = torch.clamp((lidar - prev) / max_dist, -1.0, 1.0)
            self.prev_lidar_dist = lidar.detach().clone()
            self.last_lidar_dist = lidar.detach().clone()

        return lidar

    def _ray_square_boundary_distance(self, origin: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        half = float(self.cfg.half_extent)

        ox = origin[:, None, 0]
        oy = origin[:, None, 1]

        dx = dirs[:, :, 0]
        dy = dirs[:, :, 1]

        eps = 1e-6
        dx_safe = torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
        dy_safe = torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)

        tx1 = (-half - ox) / dx_safe
        tx2 = (half - ox) / dx_safe
        ty1 = (-half - oy) / dy_safe
        ty2 = (half - oy) / dy_safe

        tx_near = torch.minimum(tx1, tx2)
        tx_far = torch.maximum(tx1, tx2)
        ty_near = torch.minimum(ty1, ty2)
        ty_far = torch.maximum(ty1, ty2)

        t_min = torch.maximum(tx_near, ty_near)
        t_max = torch.minimum(tx_far, ty_far)

        t_boundary = torch.where(
            (t_max > torch.clamp(t_min, min=0.0)),
            t_max,
            torch.full_like(t_max, float(self.cfg.lidar_max_distance)),
        )

        return torch.clamp(
            t_boundary,
            min=float(self.cfg.lidar_min_distance),
            max=float(self.cfg.lidar_max_distance),
        )

    # ------------------------------------------------------------------
    # Risk / collision
    # ------------------------------------------------------------------
    def compute_risk_features(
        self,
        root_pos_local: torch.Tensor,
        yaw: torch.Tensor,
        body_vx: Optional[torch.Tensor] = None,
        lidar_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if lidar_dist is None:
            lidar_dist = self.compute_lidar_distances(root_pos_local, yaw, update_history=False)

        front_rad = math.radians(float(self.cfg.front_angle_deg))
        side_rad = math.radians(float(self.cfg.side_angle_deg))

        angles = self.lidar_angles

        front_mask = torch.abs(angles) <= front_rad
        left_front_mask = (angles > front_rad) & (angles <= side_rad)
        right_front_mask = (angles < -front_rad) & (angles >= -side_rad)

        front_min = lidar_dist[:, front_mask].min(dim=-1)[0]
        left_front_min = lidar_dist[:, left_front_mask].min(dim=-1)[0]
        right_front_min = lidar_dist[:, right_front_mask].min(dim=-1)[0]
        all_min = lidar_dist.min(dim=-1)[0]

        risk_d = float(self.cfg.risk_distance)

        all_risk = torch.clamp((risk_d - all_min) / risk_d, 0.0, 1.0)
        front_risk = torch.clamp((risk_d - front_min) / risk_d, 0.0, 1.0)
        left_risk = torch.clamp((risk_d - left_front_min) / risk_d, 0.0, 1.0)
        right_risk = torch.clamp((risk_d - right_front_min) / risk_d, 0.0, 1.0)

        min_dyn_signed = self.min_dynamic_signed_distance(root_pos_local)
        dynamic_risk = torch.clamp((risk_d - min_dyn_signed) / risk_d, 0.0, 1.0)

        if body_vx is None:
            body_vx = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        speed_gate = torch.clamp(body_vx / max(float(self.cfg.ttc_speed_scale), 1e-6), 0.0, 1.0)
        ttc_proxy = (
            torch.clamp(
                (float(self.cfg.ttc_distance) - front_min) / max(float(self.cfg.ttc_distance), 1e-6),
                0.0,
                1.0,
            )
            * speed_gate
        )

        root_xy = root_pos_local[:, :2]
        half = float(self.cfg.half_extent)
        boundary_margin = half - torch.max(torch.abs(root_xy[:, 0]), torch.abs(root_xy[:, 1]))
        boundary_risk = torch.clamp((1.5 - boundary_margin) / 1.5, 0.0, 1.0)

        front_clearance = torch.clamp(front_min / float(self.cfg.lidar_max_distance), 0.0, 1.0)

        features = torch.stack(
            [
                all_risk,
                front_risk,
                left_risk,
                right_risk,
                dynamic_risk,
                ttc_proxy,
                boundary_risk,
                front_clearance,
            ],
            dim=-1,
        )

        features = torch.nan_to_num(
            torch.clamp(features, 0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        self.last_risk_features = features.detach().clone()

        return features

    def min_static_signed_distance(self, root_pos_local: torch.Tensor) -> torch.Tensor:
        if int(self.cfg.max_static_obs) <= 0:
            return torch.full((self.num_envs,), 1e6, dtype=torch.float32, device=self.device)

        root_xy = root_pos_local[:, :2]
        dist = torch.norm(root_xy[:, None, :] - self.static_pos, dim=-1)
        signed = dist - (self.static_radius + float(self.cfg.robot_radius) + float(self.cfg.collision_margin))
        signed = torch.where(self.static_mask, signed, torch.full_like(signed, 1e6))

        return signed.min(dim=-1)[0]

    def min_dynamic_signed_distance(self, root_pos_local: torch.Tensor) -> torch.Tensor:
        if int(self.cfg.max_dynamic_obs) <= 0:
            return torch.full((self.num_envs,), 1e6, dtype=torch.float32, device=self.device)

        root_xy = root_pos_local[:, :2]
        dist = torch.norm(root_xy[:, None, :] - self.dynamic_pos, dim=-1)
        signed = dist - (self.dynamic_radius + float(self.cfg.robot_radius) + float(self.cfg.collision_margin))
        signed = torch.where(self.dynamic_mask, signed, torch.full_like(signed, 1e6))

        return signed.min(dim=-1)[0]

    def min_obstacle_signed_distance(self, root_pos_local: torch.Tensor) -> torch.Tensor:
        return torch.minimum(
            self.min_static_signed_distance(root_pos_local),
            self.min_dynamic_signed_distance(root_pos_local),
        )

    def check_events(self, root_pos_local: torch.Tensor) -> Dict[str, torch.Tensor]:
        root_xy = root_pos_local[:, :2]
        goal_dist = torch.norm(self.goal_pos - root_xy, dim=-1)

        success = goal_dist < float(self.cfg.success_radius)

        min_static_signed = self.min_static_signed_distance(root_pos_local)
        min_dynamic_signed = self.min_dynamic_signed_distance(root_pos_local)
        min_signed = torch.minimum(min_static_signed, min_dynamic_signed)

        static_collision = min_static_signed < 0.0
        dynamic_collision = min_dynamic_signed < 0.0
        collision = static_collision | dynamic_collision

        half = float(self.cfg.half_extent)
        out_of_bounds = (torch.abs(root_xy[:, 0]) > half) | (torch.abs(root_xy[:, 1]) > half)

        boundary_margin = half - torch.max(torch.abs(root_xy[:, 0]), torch.abs(root_xy[:, 1]))

        return {
            "success": success,
            "collision": collision,
            "static_collision": static_collision,
            "dynamic_collision": dynamic_collision,
            "out_of_bounds": out_of_bounds,
            "goal_dist": goal_dist,
            "min_static_signed_distance": min_static_signed,
            "min_dynamic_signed_distance": min_dynamic_signed,
            "min_obstacle_signed_distance": min_signed,
            "boundary_margin": boundary_margin,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def rotate_world_to_body_2d(vec_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        c = torch.cos(-yaw)
        s = torch.sin(-yaw)

        x = vec_w[:, 0]
        y = vec_w[:, 1]

        bx = c * x - s * y
        by = s * x + c * y

        return torch.stack([bx, by], dim=-1)

    @staticmethod
    def rotate_body_to_world_2d(vec_b: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        c = torch.cos(yaw)
        s = torch.sin(yaw)

        x = vec_b[:, 0]
        y = vec_b[:, 1]

        wx = c * x - s * y
        wy = s * x + c * y

        return torch.stack([wx, wy], dim=-1)

    def get_counts(self) -> Dict[str, torch.Tensor]:
        return {
            "static_count": self.static_mask.float().sum(dim=-1),
            "dynamic_count": self.dynamic_mask.float().sum(dim=-1),
        }

    def get_debug_stats(self, root_pos_local: Optional[torch.Tensor] = None) -> Dict[str, float]:
        counts = self.get_counts()

        stats = {
            "Stage": self.env_stage.float().mean().item(),
            "Goal_Distance_Init_Mean": self.goal_distance.mean().item(),
            "Target_Speed_Mean": self.env_target_speed.mean().item(),
            "Static_Count": counts["static_count"].mean().item(),
            "Dynamic_Count": counts["dynamic_count"].mean().item(),
            "Static_Radius_Mean": self.static_radius[self.static_mask].mean().item() if self.static_mask.any() else 0.0,
            "Dynamic_Radius_Mean": self.dynamic_radius[self.dynamic_mask].mean().item() if self.dynamic_mask.any() else 0.0,
            "Lidar_Min": self.last_lidar_dist.min().item(),
            "Lidar_Mean": self.last_lidar_dist.mean().item(),
            "Risk_Front": self.last_risk_features[:, 1].mean().item(),
            "Risk_Dynamic": self.last_risk_features[:, 4].mean().item(),
            "Risk_Boundary": self.last_risk_features[:, 6].mean().item(),
        }

        if root_pos_local is not None:
            event = self.check_events(root_pos_local)
            stats.update(
                {
                    "Goal_Dist": event["goal_dist"].mean().item(),
                    "Min_Obstacle_Signed_Distance": event["min_obstacle_signed_distance"].mean().item(),
                    "Boundary_Margin": event["boundary_margin"].mean().item(),
                }
            )

        return stats


JetbotTask2WorldConfig = Task2WorldConfig
JetbotTask2WorldManager = Task2WorldManager

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from diff_drive_rl.tasks.task2.task2_world import Task2WorldConfig


def make_default_task2_world_cfg() -> Task2WorldConfig:
    """Create the default analytic navigation world configuration.

    The defaults keep the task suitable for open-source training: early stages
    provide clear navigation and static-obstacle pressure, while later stages
    add denser static layouts and slow dynamic obstacles without creating
    unavoidable dead ends.
    """

    cfg = Task2WorldConfig()
    cfg.curriculum_total_steps = 500_000_000
    # Six curriculum stages are kept as integer stage IDs for simple CLI usage:
    #   Stage0: Task2 observation protocol, no obstacle, navigation recovery.
    #   Stage1: weak obstacle transition, LiDAR/risk varies but no forced corridor.
    #   Stage2: one light corridor obstacle.
    #   Stage3: multiple static obstacles.
    #   Stage4: static obstacles plus a few slow dynamic obstacles.
    #   Stage5: full random obstacle navigation.
    cfg.stage_thresholds = (0.0, 0.12, 0.28, 0.46, 0.68, 0.86)
    cfg.goal_dist_ranges = (
        (4.0, 7.0),
        (4.0, 7.0),
        (4.5, 7.5),
        (5.5, 9.5),
        (7.0, 12.0),
        (10.0, 18.0),
    )
    cfg.static_count_ranges = (
        (0, 0),
        (0, 1),
        (1, 1),
        (2, 4),
        (3, 6),
        (5, 10),
    )
    cfg.dynamic_count_ranges = (
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (1, 2),
        (2, 4),
    )
    cfg.target_speed_ranges = (
        (0.42, 0.58),
        (0.42, 0.60),
        (0.42, 0.65),
        (0.44, 0.72),
        (0.46, 0.80),
        (0.50, 0.90),
    )
    cfg.static_radius_ranges = (
        (0.35, 0.60),
        (0.30, 0.55),
        (0.35, 0.65),
        (0.38, 0.85),
        (0.42, 0.95),
        (0.45, 1.10),
    )
    cfg.dynamic_radius_ranges = (
        (0.30, 0.45),
        (0.30, 0.45),
        (0.30, 0.50),
        (0.30, 0.55),
        (0.35, 0.65),
        (0.40, 0.75),
    )
    cfg.dynamic_speed_ranges = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.16, 0.34),
        (0.18, 0.45),
    )

    # Obstacle layout shaping. Stage1 is intentionally weak; Stage2 introduces
    # a single light corridor obstacle. Stage2 is deliberately kept conservative
    # after logs showed that a tighter corridor caused ~8% static collisions even
    # with good heading/progress. Later stages increase density only after the
    # stage pass checkpoint is selected.
    cfg.static_corridor_ratio_by_stage = (0.0, 0.20, 0.28, 0.55, 0.55, 0.45)
    cfg.dynamic_corridor_ratio_by_stage = (0.0, 0.0, 0.0, 0.0, 0.55, 0.60)
    cfg.corridor_longitudinal_range = (0.25, 0.82)
    cfg.corridor_lateral_offset_ranges_by_stage = (
        (2.2, 3.2),
        (1.8, 2.8),
        (1.75, 3.00),
        (1.1, 2.1),
        (0.9, 1.9),
        (0.8, 1.8),
    )
    cfg.corridor_lateral_offset_range = cfg.corridor_lateral_offset_ranges_by_stage[2]
    cfg.dynamic_crossing_lateral_range = (2.8, 4.6)
    cfg.dynamic_speed_target_ratio = 0.58
    cfg.min_passage_width = 1.60
    cfg.min_corridor_static_by_stage = (0, 0, 1, 1, 1, 2)
    return cfg


@dataclass
class Task2Config:
    """Diff-Drive UGV Task2: analytic obstacle navigation.

    Objective:
        A two-wheel differential-drive robot navigates from a sampled start
        position to a sampled goal while avoiding analytic static obstacles,
        dynamic obstacles, and arena boundaries.

    Structure:
        - Isaac Lab provides the robot articulation and wheel physics.
        - Task2WorldManager provides analytic GPU world, LiDAR, risk features,
          collision checks, success checks, and curriculum sampling.
        - No real obstacle prims are created in Isaac Sim.
        - No RayCaster is used.

    Action:
        2-D normalized navigation command:
            action[:, 0] = forward throttle in [-1, 1], internally mapped to [0, 1]
            action[:, 1] = turn command in [-1, 1]
        The environment converts [forward, turn] to left/right wheel targets.
        The commanded chassis linear component is never negative; the robot learns
        forward navigation plus steering instead of learning a reverse-driving mode.

    Protocols:
        action_protocol = "forward_throttle_turn"
        obs_protocol = "core_navigation+obstacle_perception"
        model_protocol = "modular_actor"

    Single-frame observation layout, total 166:
        Core navigation, 14 dims:
            0      goal_dist_norm
            1:3    goal_x_body_norm, goal_y_body_norm
            3:5    sin_heading_error, cos_heading_error
            5:8    body_vx, body_vy, body_wz
            8      target_speed_norm
            9      last_forward_throttle
            10     last_turn_command
            11     action_delta_forward
            12     action_delta_turn
            13     progress_ema
        Task2 extra perception, 152 dims:
            14:86   lidar_norm, 72 dims
            86:158  lidar_delta, 72 dims
            158:166 risk_features, 8 dims

    Stacked observation:
        frame_stack = 3
        stacked core navigation dim = 14 * 3 = 42
        actor obs = 166 * 3 = 498
        critic state = same as actor obs for Task2 baseline
    """

    # ------------------------------------------------------------------
    # Basic
    # ------------------------------------------------------------------
    num_envs: int = 512
    device: str = "cuda:0"
    seed: int = 42

    # 训练/调试时可固定课程阶段；-1 表示按 global_steps 正常推进。
    force_stage: int = -1

    # ------------------------------------------------------------------
    # Protocol metadata
    # ------------------------------------------------------------------
    action_protocol: str = "forward_throttle_turn"
    obs_protocol: str = "core_navigation+obstacle_perception"
    model_protocol: str = "modular_actor"
    core_single_obs_dim: int = 14
    task_extra_single_obs_dim: int = 152

    sim_dt: float = 1.0 / 144.0
    decimation: int = 3

    # Stage5 has long-distance goal navigation plus obstacle avoidance.
    # 80s is intentionally conservative.
    max_episode_length_s: float = 80.0

    # ------------------------------------------------------------------
    # Analytic world
    # ------------------------------------------------------------------
    world_cfg: Task2WorldConfig = field(default_factory=make_default_task2_world_cfg)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    num_actions: int = 2
    max_wheel_speed: float = 15.0
    action_tau: float = 0.35

    # Action semantics:
    #   action[0] -> forward throttle, mapped from [-1, 1] to [0, 1]
    #   action[1] -> yaw/turn command, mapped to differential wheel component
    # This removes the unnecessary reverse-driving option from the policy while
    # still allowing small-radius turns through differential wheel speeds.
    forward_min_norm: float = 0.0
    forward_max_norm: float = 1.0
    # 按阶段设置最小正向速度比例。Stage0/1 强制更高最小速度，
    # 防止 policy 通过负 throttle 把 speed_factor 压到 0.25 左右形成慢速保守解。
    # 后续高难度阶段逐步放松，给避障和终点收敛留出减速空间。
    forward_min_norm_by_stage: Tuple[float, ...] = (0.32, 0.30, 0.28, 0.26, 0.22, 0.18)
    turn_scale_norm: float = 0.75

    # If converted wheel targets do not match the physical robot direction in
    # env tests, tune these signs. White-box tests showed [positive, positive]
    # wheel targets move forward with the default signs.
    left_wheel_sign: float = 1.0
    right_wheel_sign: float = 1.0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    frame_stack: int = 3
    single_obs_dim: int = 166

    lin_vel_scale: float = 2.0
    ang_vel_scale: float = 4.0
    target_speed_norm: float = 1.50
    progress_norm_scale: float = 0.10
    obs_clip: float = 10.0

    # ------------------------------------------------------------------
    # Reward weights: simplified three-layer reward
    # ------------------------------------------------------------------
    # Task layer (about 60%): define effective navigation.
    # Stage0 is a formal adaptation stage from Task1 core navigation to Task2's
    # 498-D observation/model protocol. Dense shaping is intentionally moderate:
    # it should guide progress but must not let timeout episodes obtain high
    # cumulative return. Positive progress uses a nonlinear speed-ratio curve;
    # negative progress remains directly penalized.
    w_goal_progress_velocity: float = 3.00
    w_heading_improve: float = 0.60
    w_target_speed: float = 0.60
    # Alignment efficiency layer: only active during real positive progress.
    # This is not a static heading reward; it rewards aligned motion but is kept
    # below the terminal objective so that partial progress cannot replace goal
    # completion.
    w_aligned_motion: float = 0.50
    # A small auxiliary penalty discourages pushing high forward throttle while
    # the robot is still strongly misaligned. It encourages low-speed turning
    # first, without reintroducing a spin/stuck penalty family.
    w_misaligned_forward: float = 0.15

    # Heading improve is a change reward rather than a static heading reward.
    # It provides turn guidance while the heading error is being reduced, then
    # naturally vanishes once the robot is aligned with the goal.
    heading_improve_ref: float = 0.08

    # Target-speed Gaussian uses the progress velocity and is only active when
    # the robot is actually moving toward the goal.
    target_speed_sigma_ratio: float = 0.35
    positive_progress_power: float = 2.0
    target_speed_direction_power: float = 2.0
    aligned_motion_progress_power: float = 0.50
    misaligned_forward_heading_cos: float = 0.45

    # Constraint layer (about 20%): unified obstacle / dynamic / boundary safety.
    # Safety is not a positive clearance reward. Outside the safety margin the
    # penalty is zero; inside the margin it grows quadratically.
    w_safety_proximity: float = 1.50
    safety_margin: float = 1.00
    safety_risk_clip: float = 2.00
    dynamic_prediction_horizon_s: float = 0.75
    dynamic_prediction_samples: int = 3

    # Auxiliary layer (about 20%): light control smoothness and time cost only.
    w_action_smooth: float = 0.010
    w_step: float = 0.006

    # 指标窗口 EMA，用于日志判断当前阶段表现，避免累计成功率误导。
    window_metric_alpha: float = 0.02

    # ------------------------------------------------------------------
    # Stage-wise training / checkpoint selection policy
    # ------------------------------------------------------------------
    # Task2 should be trained as a sequence of short curriculum stages.  These
    # values are used by the training script to decide whether a checkpoint is
    # a useful stage hand-off candidate; they do not change PPO itself.
    stage_recommended_train_steps: Tuple[int, ...] = (5_000_000, 3_000_000, 5_000_000, 8_000_000, 10_000_000, 20_000_000)
    stage_pass_success_rate: Tuple[float, ...] = (0.90, 0.85, 0.80, 0.75, 0.70, 0.70)
    stage_pass_timeout_rate: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.25)
    stage_pass_collision_rate: Tuple[float, ...] = (0.00, 0.03, 0.05, 0.06, 0.08, 0.10)
    stage_pass_out_of_bounds_rate: Tuple[float, ...] = (0.00, 0.02, 0.03, 0.04, 0.05, 0.05)
    stage_pass_progress_velocity: Tuple[float, ...] = (0.18, 0.15, 0.11, 0.09, 0.07, 0.07)
    stage_pass_heading_cos: Tuple[float, ...] = (0.75, 0.60, 0.45, 0.35, 0.25, 0.25)
    stage_pass_speed_ratio: Tuple[float, ...] = (0.40, 0.35, 0.30, 0.25, 0.20, 0.20)

    # Balanced checkpoint score weights.  A stage checkpoint is selected by
    # success/safety first, then by efficiency.  This prevents using a final
    # checkpoint that is safe but slow.
    best_score_success_weight: float = 1.00
    best_score_collision_weight: float = 0.50
    best_score_timeout_weight: float = 0.50
    best_score_out_of_bounds_weight: float = 0.30
    best_score_progress_weight: float = 0.20
    best_score_heading_weight: float = 0.20
    best_score_speed_weight: float = 0.10

    # Optional risk-aware speed. Disabled by default so the main training path
    # remains: simple reward + curriculum + checkpoint selection. When enabled,
    # low-risk scenes keep the nominal target speed, while high-risk scenes use
    # a lower tracking target instead of adding more reward terms.
    # Enable regulated target speed from Stage2 onward. Stage0/1 usually have
    # near-zero risk, so the scale remains 1.0. In Stage2+ the speed target is
    # reduced only when analytic risk becomes non-trivial; this mirrors
    # regulated navigation controllers without adding new reward terms.
    enable_risk_aware_target_speed: bool = True
    risk_speed_medium_threshold: float = 0.03
    risk_speed_high_threshold: float = 0.08
    risk_speed_medium_scale: float = 0.85
    risk_speed_high_scale: float = 0.65

    # ------------------------------------------------------------------
    # Event rewards
    # Success/timeout separation is deliberately large. A timeout episode should
    # not remain competitive merely by accumulating small positive shaping reward.
    # ------------------------------------------------------------------
    rew_success: float = 30.0
    rew_collision: float = -15.0
    rew_out_of_bounds: float = -20.0
    rew_timeout: float = -25.0

    # ------------------------------------------------------------------
    # Reward shaping clips
    # ------------------------------------------------------------------
    progress_clip: float = 0.30
    continuous_reward_clip: float = 2.00
    max_episode_return_abs: float = 2000.0

    # ------------------------------------------------------------------
    # Stuck detection
    # ------------------------------------------------------------------
    stuck_progress_threshold: float = 0.002
    stuck_speed_threshold: float = 0.04
    # 如果 heading 已经对准但速度比例过低，也视为低速卡住。
    stuck_speed_ratio_threshold: float = 0.35
    stuck_heading_cos_threshold: float = 0.80
    # 近目标允许慢速收敛，不把 near-goal 慢速误判为 stuck。
    stuck_near_goal_radius_scale: float = 2.0
    stuck_after_steps: int = 40

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    env_spacing: float = 60.0
    spawn_height: float = 0.05
    ground_color: Tuple[float, float, float] = (0.85, 0.85, 0.85)

    robot_usd_path: str = (
        "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/2023.1.1/Isaac/Robots/Jetbot/jetbot.usd"
    )

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    print_debug_info: bool = True

    @property
    def policy_dt(self) -> float:
        return float(self.sim_dt) * int(self.decimation)

    @property
    def max_episode_length(self) -> int:
        return int(float(self.max_episode_length_s) / max(float(self.policy_dt), 1e-6))

    @property
    def num_observations(self) -> int:
        return int(self.frame_stack) * int(self.single_obs_dim)

    @property
    def stacked_obs_dim(self) -> int:
        return self.num_observations

    @property
    def stacked_core_obs_dim(self) -> int:
        return int(self.frame_stack) * int(self.core_single_obs_dim)

    @property
    def stacked_task_extra_obs_dim(self) -> int:
        return int(self.frame_stack) * int(self.task_extra_single_obs_dim)

    def validate(self) -> None:
        assert self.num_envs > 0
        assert -1 <= int(self.force_stage) < self.world_cfg.num_stages
        assert isinstance(self.device, str) and len(self.device) > 0
        assert self.sim_dt > 0.0
        assert self.decimation >= 1
        assert self.policy_dt > 0.0
        assert self.max_episode_length_s > 0.0
        assert self.max_episode_length > 0

        assert isinstance(self.world_cfg, Task2WorldConfig)
        self.world_cfg.validate()

        assert self.num_actions == 2
        assert self.max_wheel_speed > 0.0
        assert 0.0 <= self.action_tau <= 1.0
        assert 0.0 <= self.forward_min_norm <= self.forward_max_norm <= 1.0
        assert len(self.forward_min_norm_by_stage) == self.world_cfg.num_stages
        assert all(0.0 <= float(v) <= self.forward_max_norm for v in self.forward_min_norm_by_stage)
        assert self.turn_scale_norm >= 0.0
        assert self.left_wheel_sign in (-1.0, 1.0)
        assert self.right_wheel_sign in (-1.0, 1.0)

        assert isinstance(self.action_protocol, str) and self.action_protocol == "forward_throttle_turn"
        assert isinstance(self.obs_protocol, str) and self.obs_protocol.startswith("core_navigation")
        assert isinstance(self.model_protocol, str) and self.model_protocol == "modular_actor"
        assert self.core_single_obs_dim == 14
        assert self.task_extra_single_obs_dim == 152

        assert self.frame_stack >= 1
        assert self.world_cfg.num_lidar_rays == 72
        assert self.single_obs_dim == self.core_single_obs_dim + self.task_extra_single_obs_dim
        assert self.single_obs_dim == 166
        assert self.num_observations == self.frame_stack * self.single_obs_dim
        assert self.num_observations == 498

        assert self.lin_vel_scale > 0.0
        assert self.ang_vel_scale > 0.0
        assert self.target_speed_norm > 0.0
        assert self.progress_norm_scale > 0.0
        assert self.obs_clip > 0.0

        assert self.w_goal_progress_velocity >= 0.0
        assert self.w_heading_improve >= 0.0
        assert self.w_target_speed >= 0.0
        assert self.w_aligned_motion >= 0.0
        assert self.w_misaligned_forward >= 0.0
        assert self.heading_improve_ref > 0.0
        assert self.target_speed_sigma_ratio > 0.0
        assert self.positive_progress_power >= 1.0
        assert self.target_speed_direction_power >= 1.0
        assert self.aligned_motion_progress_power > 0.0
        assert -1.0 <= self.misaligned_forward_heading_cos <= 1.0
        assert self.w_safety_proximity >= 0.0
        assert self.safety_margin > 0.0
        assert self.safety_risk_clip > 0.0
        assert self.dynamic_prediction_horizon_s >= 0.0
        assert self.dynamic_prediction_samples >= 1
        assert self.w_action_smooth >= 0.0
        assert self.w_step >= 0.0
        assert 0.0 < self.window_metric_alpha <= 1.0

        stage_len = self.world_cfg.num_stages
        for name in [
            "stage_recommended_train_steps",
            "stage_pass_success_rate",
            "stage_pass_timeout_rate",
            "stage_pass_collision_rate",
            "stage_pass_out_of_bounds_rate",
            "stage_pass_progress_velocity",
            "stage_pass_heading_cos",
            "stage_pass_speed_ratio",
        ]:
            assert len(getattr(self, name)) == stage_len, f"{name} length must equal num_stages"
        assert all(int(v) > 0 for v in self.stage_recommended_train_steps)
        assert all(0.0 <= float(v) <= 1.0 for v in self.stage_pass_success_rate)
        assert all(0.0 <= float(v) <= 1.0 for v in self.stage_pass_timeout_rate)
        assert all(0.0 <= float(v) <= 1.0 for v in self.stage_pass_collision_rate)
        assert all(0.0 <= float(v) <= 1.0 for v in self.stage_pass_out_of_bounds_rate)
        assert all(float(v) >= 0.0 for v in self.stage_pass_progress_velocity)
        assert all(-1.0 <= float(v) <= 1.0 for v in self.stage_pass_heading_cos)
        assert all(float(v) >= 0.0 for v in self.stage_pass_speed_ratio)
        assert self.best_score_success_weight >= 0.0
        assert self.best_score_collision_weight >= 0.0
        assert self.best_score_timeout_weight >= 0.0
        assert self.best_score_out_of_bounds_weight >= 0.0
        assert self.best_score_progress_weight >= 0.0
        assert self.best_score_heading_weight >= 0.0
        assert self.best_score_speed_weight >= 0.0
        assert 0.0 <= self.risk_speed_medium_threshold <= 1.0
        assert 0.0 <= self.risk_speed_high_threshold <= 1.0
        assert self.risk_speed_medium_threshold <= self.risk_speed_high_threshold
        assert 0.0 < self.risk_speed_high_scale <= self.risk_speed_medium_scale <= 1.0

        assert self.progress_clip > 0.0
        assert self.continuous_reward_clip > 0.0
        assert self.max_episode_return_abs > 0.0

        assert self.env_spacing > 0.0
        assert self.env_spacing >= self.world_cfg.arena_size
        assert self.spawn_height >= 0.0
        assert len(self.ground_color) == 3
        assert all(0.0 <= float(c) <= 1.0 for c in self.ground_color)
        assert isinstance(self.robot_usd_path, str) and len(self.robot_usd_path) > 0


DiffDriveTask2Config = Task2Config
DiffDriveTask2Config = Task2Config
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from diff_drive_rl.tasks.task2.task2_world import Task2WorldConfig


@dataclass
class Task2Config:
    """Diff-Drive UGV / Jetbot Task2: analytic obstacle navigation.

    Objective:
        A two-wheel differential-drive Jetbot navigates from a sampled start
        position to a sampled goal while avoiding analytic static obstacles,
        dynamic obstacles, and arena boundaries.

    Structure:
        - Isaac Lab provides the real Jetbot articulation and wheel physics.
        - Task2WorldManager provides analytic GPU world, LiDAR, risk features,
          collision checks, success checks, and curriculum sampling.
        - No real obstacle prims are created in Isaac Sim.
        - No RayCaster is used.

    Action:
        2-D normalized wheel velocity command:
            action[:, 0] = left wheel command
            action[:, 1] = right wheel command

    Single-frame observation layout, total 166:
        0      goal_dist_norm
        1:3    goal_x_body_norm, goal_y_body_norm
        3:5    sin_heading, cos_heading
        5:8    body_vx, body_vy, body_wz
        8      target_speed_norm
        9:11   last_action_left, last_action_right
        11:13  action_delta_left, action_delta_right
        13     progress_ema
        14:86  lidar_norm, 72 dims
        86:158 lidar_delta, 72 dims
        158:166 risk_features, 8 dims

    Stacked observation:
        frame_stack = 3
        actor obs = 166 * 3 = 498
        critic state = same as actor obs for Task2 baseline
    """

    # ------------------------------------------------------------------
    # Basic
    # ------------------------------------------------------------------
    num_envs: int = 512
    device: str = "cuda:0"
    seed: int = 42

    sim_dt: float = 1.0 / 144.0
    decimation: int = 3

    # Stage5 has long-distance goal navigation plus obstacle avoidance.
    # 80s is intentionally conservative.
    max_episode_length_s: float = 80.0

    # ------------------------------------------------------------------
    # Analytic world
    # ------------------------------------------------------------------
    world_cfg: Task2WorldConfig = field(default_factory=Task2WorldConfig)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    num_actions: int = 2
    max_wheel_speed: float = 15.0
    action_tau: float = 0.35

    # If action=[1,1] does not move forward in env test, tune these signs.
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
    # Reward weights: navigation
    # ------------------------------------------------------------------
    w_progress: float = 4.00
    w_goal_speed: float = 0.45
    w_heading: float = 0.10
    w_turn_to_goal: float = 0.06

    # ------------------------------------------------------------------
    # Reward weights: safety
    # ------------------------------------------------------------------
    w_front_clearance: float = 0.12
    w_collision_risk: float = 0.25
    w_ttc: float = 0.20
    w_boundary: float = 0.10

    # ------------------------------------------------------------------
    # Reward weights: regularization / anti-degenerate
    # ------------------------------------------------------------------
    w_spin: float = 0.030
    w_stuck: float = 0.040
    w_action_smooth: float = 0.010
    w_action_mag: float = 0.0005
    w_wheel_speed: float = 0.0008
    w_step: float = 0.002

    # ------------------------------------------------------------------
    # Event rewards
    # ------------------------------------------------------------------
    rew_success: float = 20.0
    rew_collision: float = -15.0
    rew_out_of_bounds: float = -15.0
    rew_timeout: float = -5.0

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
    stuck_after_steps: int = 40

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    env_spacing: float = 60.0
    spawn_height: float = 0.05
    ground_color: Tuple[float, float, float] = (0.85, 0.85, 0.85)

    jetbot_usd_path: str = (
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

    def validate(self) -> None:
        assert self.num_envs > 0
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
        assert self.left_wheel_sign in (-1.0, 1.0)
        assert self.right_wheel_sign in (-1.0, 1.0)

        assert self.frame_stack >= 1
        assert self.world_cfg.num_lidar_rays == 72
        assert self.single_obs_dim == 166
        assert self.num_observations == self.frame_stack * self.single_obs_dim
        assert self.num_observations == 498

        assert self.lin_vel_scale > 0.0
        assert self.ang_vel_scale > 0.0
        assert self.target_speed_norm > 0.0
        assert self.progress_norm_scale > 0.0
        assert self.obs_clip > 0.0

        assert self.progress_clip > 0.0
        assert self.continuous_reward_clip > 0.0
        assert self.max_episode_return_abs > 0.0

        assert self.stuck_progress_threshold >= 0.0
        assert self.stuck_speed_threshold >= 0.0
        assert self.stuck_after_steps >= 0

        assert self.env_spacing > 0.0
        assert self.env_spacing >= self.world_cfg.arena_size
        assert self.spawn_height >= 0.0
        assert len(self.ground_color) == 3
        assert all(0.0 <= float(c) <= 1.0 for c in self.ground_color)
        assert isinstance(self.jetbot_usd_path, str) and len(self.jetbot_usd_path) > 0


DiffDriveTask2Config = Task2Config
JetbotTask2Config = Task2Config

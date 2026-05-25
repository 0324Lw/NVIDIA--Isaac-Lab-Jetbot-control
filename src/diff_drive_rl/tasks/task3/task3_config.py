from __future__ import annotations

from dataclasses import dataclass, field

from diff_drive_rl.tasks.task3.task3_world import Task3WorldConfig, Task3WorldManager


@dataclass
class Task3Config:
    """Diff-Drive UGV / Jetbot Task3: Conservative Sim2Real Parking.

    Objective:
        A two-wheel differential-drive Jetbot starts from the left side of a
        mixed-material track, crosses low conservative speed bumps, enters a
        U-shaped parking spot, and completes stable parking.

    Structure:
        - Isaac Lab provides real Jetbot articulation physics.
        - Task3WorldManager provides conservative world geometry tensors,
          Sim2Real randomization buffers, analytic LiDAR, risk features,
          events, and privileged features.
        - task3_scene.py creates the Isaac scene and calls spawn_world_assets().
        - task3_env.py owns reset / step / observation / reward / done logic.

    Action:
        2-D normalized wheel velocity command:
            action[:, 0] = left wheel command
            action[:, 1] = right wheel command

    Single actor observation layout, total 101:
        0:3    body_vx, body_vy, body_wz
        3:5    wheel_vel_left, wheel_vel_right
        5      goal_dist_norm
        6:8    goal_x_body_norm, goal_y_body_norm
        8:10   heading_sin, heading_cos
        10:12  goal_yaw_sin, goal_yaw_cos
        12:14  parking_x_norm, parking_y_norm
        14:16  applied_action_left, applied_action_right
        16:18  action_delta_left, action_delta_right
        18     progress_ema
        19:55  lidar_norm, 36 dims
        55:91  lidar_delta, 36 dims
        91:101 risk_features, 10 dims

    Stacked actor obs:
        frame_stack = 4
        actor obs = 101 * 4 = 404

    Critic privileged obs:
        actor obs stack + world privileged features
        critic obs = 404 + 38 = 442
    """

    # ------------------------------------------------------------------
    # Basic
    # ------------------------------------------------------------------
    num_envs: int = 512
    device: str = "cuda:0"
    seed: int = 42

    sim_dt: float = 0.01
    decimation: int = 4

    # 25 Hz policy. 40s covers 10m track + parking.
    max_episode_length_s: float = 40.0

    # ------------------------------------------------------------------
    # World
    # ------------------------------------------------------------------
    world_cfg: Task3WorldConfig = field(default_factory=Task3WorldConfig)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    num_actions: int = 2
    max_wheel_speed: float = 14.0

    # If action=[1,1] does not move forward in env test, tune these signs.
    left_wheel_sign: float = 1.0
    right_wheel_sign: float = 1.0

    # Must cover world_cfg.action_delay_frame_range max.
    max_action_delay_frames: int = 4

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    frame_stack: int = 4
    single_actor_obs_dim: int = 101

    lin_vel_scale: float = 2.0
    ang_vel_scale: float = 4.0
    wheel_vel_scale: float = 20.0
    progress_norm_scale: float = 0.10

    obs_clip: float = 10.0
    priv_clip: float = 20.0

    # ------------------------------------------------------------------
    # Reward weights: navigation
    # ------------------------------------------------------------------
    w_progress: float = 4.00
    w_goal_speed: float = 0.35
    w_heading: float = 0.12

    # ------------------------------------------------------------------
    # Reward weights: parking
    # ------------------------------------------------------------------
    w_parking_pos: float = 0.80
    w_parking_yaw: float = 0.55
    w_inside_box: float = 0.25
    w_parking_low_speed: float = 0.25

    # ------------------------------------------------------------------
    # Reward weights: terrain / speed bumps
    # ------------------------------------------------------------------
    w_terrain_progress: float = 0.10

    # Speed bumps are conservative low ramp bumps. Do not over-penalize
    # bump overlap, otherwise early policy may learn to stop before bumps.
    w_bump_progress: float = 0.04
    w_bump_smooth: float = 0.01

    # ------------------------------------------------------------------
    # Reward weights: safety
    # ------------------------------------------------------------------
    w_front_clearance: float = 0.08
    w_lidar_risk: float = 0.20
    w_wall_risk: float = 0.30
    w_lane_risk: float = 0.25
    w_bump_risk: float = 0.00

    # ------------------------------------------------------------------
    # Reward weights: regularization
    # ------------------------------------------------------------------
    w_spin: float = 0.025
    w_action_smooth: float = 0.015
    w_action_mag: float = 0.002
    w_wheel_speed: float = 0.001
    w_stuck: float = 0.050
    w_step: float = 0.003

    # ------------------------------------------------------------------
    # Event rewards
    # ------------------------------------------------------------------
    rew_success: float = 40.0
    rew_crash: float = -25.0
    rew_timeout: float = -8.0

    # ------------------------------------------------------------------
    # Reward shaping clips
    # ------------------------------------------------------------------
    progress_clip: float = 0.30
    continuous_reward_clip: float = 2.50
    max_episode_return_abs: float = 3000.0

    # ------------------------------------------------------------------
    # Stuck detection
    # ------------------------------------------------------------------
    stuck_progress_threshold: float = 0.0015
    stuck_speed_threshold: float = 0.035
    stuck_after_steps: int = 60

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    env_spacing: float = 15.0
    spawn_height: float = 0.05

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
        return int(float(self.max_episode_length_s) / max(float(self.policy_dt), 1.0e-6))

    @property
    def actor_obs_dim(self) -> int:
        return int(self.single_actor_obs_dim) * int(self.frame_stack)

    @property
    def privileged_feature_dim(self) -> int:
        return int(Task3WorldManager.privileged_feature_dim())

    @property
    def critic_obs_dim(self) -> int:
        return int(self.actor_obs_dim + self.privileged_feature_dim)

    @property
    def num_observations(self) -> int:
        return int(self.actor_obs_dim)

    @property
    def num_privileged_obs(self) -> int:
        return int(self.critic_obs_dim)

    def validate(self) -> None:
        assert self.num_envs > 0
        assert isinstance(self.device, str) and len(self.device) > 0
        assert self.seed >= 0

        assert self.sim_dt > 0.0
        assert self.decimation >= 1
        assert self.policy_dt > 0.0
        assert self.max_episode_length_s > 0.0
        assert self.max_episode_length > 0

        assert isinstance(self.world_cfg, Task3WorldConfig)
        self.world_cfg.validate()

        assert self.num_actions == 2
        assert self.max_wheel_speed > 0.0
        assert self.left_wheel_sign in (-1.0, 1.0)
        assert self.right_wheel_sign in (-1.0, 1.0)

        assert self.max_action_delay_frames >= int(self.world_cfg.action_delay_frame_range[1])
        assert self.max_action_delay_frames >= 0

        assert self.frame_stack >= 1
        assert self.world_cfg.lidar_pool_bins == 36
        assert Task3WorldManager.risk_feature_dim() == 10
        assert Task3WorldManager.privileged_feature_dim() == 38

        assert self.single_actor_obs_dim == 101
        assert self.actor_obs_dim == 404
        assert self.privileged_feature_dim == 38
        assert self.critic_obs_dim == 442

        assert self.lin_vel_scale > 0.0
        assert self.ang_vel_scale > 0.0
        assert self.wheel_vel_scale > 0.0
        assert self.progress_norm_scale > 0.0
        assert self.obs_clip > 0.0
        assert self.priv_clip > 0.0

        assert self.progress_clip > 0.0
        assert self.continuous_reward_clip > 0.0
        assert self.max_episode_return_abs > 0.0

        assert self.stuck_progress_threshold >= 0.0
        assert self.stuck_speed_threshold >= 0.0
        assert self.stuck_after_steps >= 0

        assert self.env_spacing > 0.0
        assert self.spawn_height >= 0.0
        assert isinstance(self.jetbot_usd_path, str) and len(self.jetbot_usd_path) > 0


DiffDriveTask3Config = Task3Config
JetbotTask3Config = Task3Config

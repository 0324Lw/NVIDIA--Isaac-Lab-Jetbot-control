from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Task1Config:
    """Diff-drive UGV / Jetbot Task1: multi-waypoint navigation.

    Objective:
        A two-wheel differential-drive robot must sequentially reach several
        randomly sampled waypoints on flat ground.

    Action:
        2-D normalized wheel velocity command:
            action[:, 0] = left wheel command
            action[:, 1] = right wheel command

    Observation:
        3-frame stack. Each frame is 12-D:
            0  dist_norm
            1  sin(heading_error)
            2  cos(heading_error)
            3  body_vx
            4  body_vy
            5  body_wz
            6  left_wheel_vel
            7  right_wheel_vel
            8  last_action_left
            9  last_action_right
            10 progress_ema
            11 waypoint_index_norm

    Project positioning:
        This is an educational pure-RL baseline for two-wheel differential-drive
        navigation in Isaac Lab.
    """

    # ----------------------------- Basic -----------------------------
    num_envs: int = 512
    device: str = "cuda:0"
    seed: int = 42

    sim_dt: float = 1.0 / 144.0
    decimation: int = 3
    max_episode_length_s: float = 18.0

    # ----------------------------- Task -----------------------------
    num_waypoints: int = 3
    waypoint_min_radius: float = 0.50
    waypoint_max_radius: float = 1.40
    waypoint_world_radius: float = 3.00
    reach_threshold: float = 0.25

    # ----------------------------- Control -----------------------------
    num_actions: int = 2
    wheel_speed_scale: float = 16.0
    action_tau: float = 0.35

    # If action=[1, 1] does not move forward in white-box tests,
    # tune these signs instead of changing the reward.
    left_wheel_sign: float = 1.0
    right_wheel_sign: float = 1.0

    # ----------------------------- Observation -----------------------------
    frame_stack: int = 3
    single_obs_dim: int = 12

    dist_norm_scale: float = 3.0
    lin_vel_scale: float = 2.0
    ang_vel_scale: float = 4.0
    wheel_vel_scale: float = 20.0
    progress_norm_scale: float = 0.10

    obs_clip: float = 10.0

    # ----------------------------- Reward weights -----------------------------
    w_progress: float = 5.00
    w_goal_heading: float = 0.10
    w_goal_forward: float = 0.24
    w_spin: float = 0.035
    w_lateral_vel: float = 0.020
    w_action_smooth: float = 0.015
    w_action_mag: float = 0.002
    w_wheel_speed: float = 0.001
    w_stuck: float = 0.030
    w_step: float = 0.0045

    rew_waypoint: float = 6.00
    rew_finish: float = 24.00
    rew_timeout: float = 0.0

    progress_clip: float = 0.20
    continuous_reward_clip: float = 1.00
    max_episode_return_abs: float = 500.0

    stuck_progress_threshold: float = 0.002
    stuck_speed_threshold: float = 0.04
    stuck_after_steps: int = 30

    # ----------------------------- Scene -----------------------------
    env_spacing: float = 7.0
    spawn_height: float = 0.05
    ground_color: Tuple[float, float, float] = (0.85, 0.85, 0.85)

    # NVIDIA Isaac Jetbot USD.
    # Keep this as a URL by default so the repo does not need to commit USD assets.
    jetbot_usd_path: str = (
        "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/2023.1.1/Isaac/Robots/Jetbot/jetbot.usd"
    )

    # ----------------------------- Debug -----------------------------
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

        assert self.num_waypoints >= 1
        assert self.waypoint_min_radius >= 0.0
        assert self.waypoint_max_radius >= self.waypoint_min_radius
        assert self.waypoint_world_radius >= self.waypoint_max_radius
        assert self.reach_threshold > 0.0

        assert self.num_actions == 2
        assert self.wheel_speed_scale > 0.0
        assert 0.0 <= self.action_tau <= 1.0
        assert self.left_wheel_sign in (-1.0, 1.0)
        assert self.right_wheel_sign in (-1.0, 1.0)

        assert self.frame_stack >= 1
        assert self.single_obs_dim == 12
        assert self.num_observations == self.frame_stack * self.single_obs_dim
        assert self.num_observations == 36

        assert self.dist_norm_scale > 0.0
        assert self.lin_vel_scale > 0.0
        assert self.ang_vel_scale > 0.0
        assert self.wheel_vel_scale > 0.0
        assert self.progress_norm_scale > 0.0
        assert self.obs_clip > 0.0

        assert self.progress_clip > 0.0
        assert self.continuous_reward_clip > 0.0
        assert self.max_episode_return_abs > 0.0

        assert self.stuck_progress_threshold >= 0.0
        assert self.stuck_speed_threshold >= 0.0
        assert self.stuck_after_steps >= 0

        assert self.env_spacing > 0.0
        assert self.spawn_height >= 0.0
        assert len(self.ground_color) == 3
        assert all(0.0 <= float(c) <= 1.0 for c in self.ground_color)
        assert isinstance(self.jetbot_usd_path, str) and len(self.jetbot_usd_path) > 0


DiffDriveTask1Config = Task1Config
JetbotTask1Config = Task1Config

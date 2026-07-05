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
        2-D normalized network command:
            action[:, 0] = forward throttle in [-1, 1]
            action[:, 1] = turn command in [-1, 1]

        Environment execution:
            The forward throttle is continuously mapped to a non-negative
            chassis forward command. The turn command is added differentially to
            the left / right wheels, so a wheel may become negative during a
            pivot turn while the commanded chassis forward component is never
            negative. This matches normal forward-only differential-drive
            navigation and keeps Task1 pretraining compatible with Task2.

    Observation:
        CoreNav-v1 single frame, 14-D:
            0      goal_dist_norm
            1      goal_x_body_norm
            2      goal_y_body_norm
            3      sin_heading_error
            4      cos_heading_error
            5      body_vx
            6      body_vy
            7      body_wz
            8      target_speed_norm
            9      last_forward_throttle
            10     last_turn_command
            11     action_delta_forward
            12     action_delta_turn
            13     progress_ema

        3-frame stack -> actor obs = 14 * 3 = 42 dims.
        Task1 intentionally exposes only the active waypoint as the current goal.
        waypoint_index / remaining_waypoints are kept in telemetry and may be
        added to a future privileged critic, but not to the actor CoreNav input.

    Protocols:
        action_protocol = "forward_throttle_turn_v1"
        obs_protocol = "CoreNav-v1"
        model_protocol = "ModularActor-v1"

    Project positioning:
        This is the formal single-vehicle navigation foundation model for
        downstream Task2 / Task3 reuse. The policy should learn reusable local
        goal tracking from CoreNav-v1 instead of task-specific wheel features.
    """

    # ----------------------------- Basic -----------------------------
    num_envs: int = 512
    device: str = "cuda:0"
    seed: int = 42

    action_protocol: str = "forward_throttle_turn_v1"
    obs_protocol: str = "CoreNav-v1"
    model_protocol: str = "ModularActor-v1"
    core_single_obs_dim: int = 14
    task_extra_single_obs_dim: int = 0

    sim_dt: float = 1.0 / 144.0
    decimation: int = 3
    max_episode_length_s: float = 24.0

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

    # PPO still outputs actions in [-1, 1]. action[0] is mapped continuously to
    # a non-negative forward command. action[1] controls differential turning.
    # Individual wheel targets may be negative during pivot turns; the chassis
    # forward command is never negative.
    min_forward_action: float = 0.05
    max_forward_action: float = 1.00
    # Nonlinear forward mapping: raw forward=0 should not become medium speed.
    # This prevents the Stage2 local optimum "go straight fast and ignore turns".
    forward_curve_power: float = 2.00
    turn_scale_norm: float = 0.85

    # If action=[1, 1] does not move forward in white-box tests,
    # tune these signs instead of changing the reward.
    left_wheel_sign: float = 1.0
    right_wheel_sign: float = 1.0

    # ----------------------------- Observation -----------------------------
    frame_stack: int = 3
    single_obs_dim: int = 14

    dist_norm_scale: float = 3.0
    goal_xy_norm: float = 3.0
    lin_vel_scale: float = 2.0
    ang_vel_scale: float = 4.0
    target_speed_norm: float = 1.50
    # Used only for wheel-speed regularization in reward diagnostics; wheel
    # velocities are intentionally not part of CoreNav-v1 actor observation.
    wheel_vel_scale: float = 20.0
    progress_norm_scale: float = 0.10

    obs_clip: float = 10.0

    # ----------------------------- Reward gates -----------------------------
    # The main reward only opens when direction, speed, distance and progress
    # are consistent with forward waypoint tracking.
    heading_gate_min: float = 0.00
    heading_gate_full: float = 0.75
    min_goal_speed: float = 0.04
    target_goal_speed: float = 0.30
    distance_gate_near: float = 0.30
    distance_gate_far: float = 0.90

    # ----------------------------- Reward weights -----------------------------
    w_progress: float = 12.00
    w_goal_heading: float = 0.08
    w_goal_forward: float = 1.00
    w_heading_improve: float = 0.08
    # Immediate turn-recovery reward. It uses the commanded turn direction and
    # the actual yaw-rate direction so the policy receives a positive signal
    # before it has already completed the turn. This is the key fix for Stage2.
    w_turn_to_goal: float = 0.40
    w_negative_progress: float = 2.00
    w_backward: float = 0.20
    w_misaligned_forward: float = 0.12
    w_slow: float = 0.04
    w_no_progress: float = 0.02
    w_spin_in_place: float = 0.00

    w_spin: float = 0.010
    w_bad_turn: float = 0.00
    w_lateral_vel: float = 0.000
    w_action_smooth: float = 0.010
    w_action_mag: float = 0.001
    w_wheel_speed: float = 0.000
    w_stuck: float = 0.000
    w_step: float = 0.0030

    # Turn-recovery shaping. In this robot convention, a negative turn command
    # produces positive yaw; keep this sign explicit for easier wheel-sign fixes.
    turn_command_to_yaw_sign: float = -1.0
    turn_recovery_heading_threshold: float = 0.35
    turn_recovery_full_heading: float = 1.20
    target_turn_command: float = 0.45

    rew_waypoint: float = 6.00
    rew_finish: float = 24.00
    rew_timeout: float = 0.0
    rew_hard_stuck: float = -0.50

    progress_clip: float = 0.20
    continuous_reward_clip: float = 1.00
    max_episode_return_abs: float = 500.0

    stuck_progress_threshold: float = 0.0008
    stuck_speed_threshold: float = 0.035
    stuck_after_steps: int = 80
    hard_stuck_after_steps: int = 900

    # Waypoint sampling. Task1 is a simple local waypoint-following teacher, not
    # a global path-planning task. Each newly activated waypoint is sampled in
    # the robot heading cone so the forward-only policy never receives an
    # artificial target behind the vehicle after a waypoint switch. This keeps
    # the task simple, stable and useful as Task2 / Task3 navigation pretraining.
    forward_cone_waypoint_sampling: bool = True
    # Sample in a heading cone, but avoid almost-straight-only training.
    # Stage0/1 must force some steering, otherwise PPO learns a straight-line
    # policy and collapses at the first two-waypoint curriculum.
    waypoint_front_angle_deg: float = 60.0
    waypoint_min_front_angle_deg: float = 15.0
    waypoint_total_path_length: float = 4.0
    waypoint_path_resample_attempts: int = 8
    # Do not align reset yaw to the first waypoint. Keeping the robot yaw fixed
    # makes the sampled first-waypoint angle visible to the policy.
    reset_align_to_first_waypoint: bool = False

    # Optional late-stage anti-degeneration diagnostics. These remain available,
    # turns that do not reduce heading error or produce progress.
    bad_turn_wz_threshold: float = 1.25
    bad_turn_progress_threshold: float = 0.0006
    bad_turn_heading_improve_threshold: float = 1.0e-4

    # Recent-window statistics. This does not affect environment dynamics.
    recent_window_size: int = 4096

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
        assert 0.0 <= self.min_forward_action <= self.max_forward_action <= 1.0
        assert self.forward_curve_power >= 1.0
        assert self.turn_scale_norm >= 0.0
        assert self.left_wheel_sign in (-1.0, 1.0)
        assert self.right_wheel_sign in (-1.0, 1.0)

        assert self.action_protocol == "forward_throttle_turn_v1"
        assert self.obs_protocol == "CoreNav-v1"
        assert self.model_protocol == "ModularActor-v1"
        assert self.frame_stack >= 1
        assert self.core_single_obs_dim == 14
        assert self.task_extra_single_obs_dim == 0
        assert self.single_obs_dim == self.core_single_obs_dim + self.task_extra_single_obs_dim
        assert self.single_obs_dim == 14
        assert self.num_observations == self.frame_stack * self.single_obs_dim
        assert self.num_observations == 42

        assert self.dist_norm_scale > 0.0
        assert self.goal_xy_norm > 0.0
        assert self.lin_vel_scale > 0.0
        assert self.ang_vel_scale > 0.0
        assert self.target_speed_norm > 0.0
        assert self.wheel_vel_scale > 0.0
        assert self.progress_norm_scale > 0.0
        assert self.obs_clip > 0.0

        assert self.heading_gate_full > self.heading_gate_min
        assert self.min_goal_speed >= 0.0
        assert self.target_goal_speed > self.min_goal_speed
        assert self.distance_gate_far > self.distance_gate_near >= 0.0
        assert self.turn_command_to_yaw_sign in (-1.0, 1.0)
        assert self.turn_recovery_full_heading > self.turn_recovery_heading_threshold >= 0.0
        assert self.target_turn_command > 0.0
        assert 0.0 <= self.waypoint_min_front_angle_deg <= self.waypoint_front_angle_deg <= 180.0
        assert self.waypoint_total_path_length >= self.num_waypoints * self.waypoint_min_radius

        assert self.w_heading_improve >= 0.0
        assert self.w_turn_to_goal >= 0.0
        assert self.progress_clip > 0.0
        assert self.continuous_reward_clip > 0.0
        assert self.max_episode_return_abs > 0.0

        assert self.stuck_progress_threshold >= 0.0
        assert self.stuck_speed_threshold >= 0.0
        assert self.stuck_after_steps >= 0
        assert self.hard_stuck_after_steps >= self.stuck_after_steps
        assert self.w_bad_turn >= 0.0
        assert 0.0 <= self.waypoint_front_angle_deg <= 120.0
        assert self.waypoint_total_path_length >= self.num_waypoints * self.waypoint_min_radius
        assert self.waypoint_path_resample_attempts >= 1
        assert self.bad_turn_wz_threshold >= 0.0
        assert self.bad_turn_progress_threshold >= 0.0
        assert self.bad_turn_heading_improve_threshold >= 0.0
        assert self.recent_window_size >= 1

        assert self.env_spacing > 0.0
        assert self.spawn_height >= 0.0
        assert len(self.ground_color) == 3
        assert all(0.0 <= float(c) <= 1.0 for c in self.ground_color)
        assert isinstance(self.jetbot_usd_path, str) and len(self.jetbot_usd_path) > 0


DiffDriveTask1Config = Task1Config
JetbotTask1Config = Task1Config
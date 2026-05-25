from __future__ import annotations

from dataclasses import dataclass, field

from diff_drive_rl.tasks.task4.task4_world import Task4WorldConfig, Task4WorldManager


@dataclass
class Task4Config:
    """Diff-Drive UGV / Jetbot Task4: Multi-UGV Formation Escort.

    Task:
        Four Jetbot differential-drive UGVs escort a virtual formation center
        to a shared team goal while keeping formation and avoiding static
        obstacles, narrow gate walls, arena boundaries, and teammate collisions.

    Interface:
        action:
            [num_envs, 4, 2]
            action[..., 0] = normalized linear velocity command
            action[..., 1] = normalized yaw rate command

        actor observation:
            [num_envs, 4, actor_obs_dim]
            actor_obs_dim = frame_stack * single_actor_obs_dim = 4 * 156 = 624

        centralized critic state:
            [num_envs, critic_obs_dim]
            critic_obs_dim = Task4WorldManager.privileged_feature_dim(...) = 96

        reward:
            [num_envs, 4]

        terminated / truncated:
            [num_envs]
    """

    # ------------------------------------------------------------------
    # Basic
    # ------------------------------------------------------------------
    num_envs: int = 512
    device: str = "cuda:0"
    seed: int = 42

    sim_dt: float = 0.01
    decimation: int = 4
    max_episode_length_s: float = 35.0

    # ------------------------------------------------------------------
    # World
    # ------------------------------------------------------------------
    world_cfg: Task4WorldConfig = field(default_factory=Task4WorldConfig)
    curriculum_stage: int = 0

    # ------------------------------------------------------------------
    # Multi-agent
    # ------------------------------------------------------------------
    num_agents: int = 4
    num_actions_per_agent: int = 2

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    # Jetbot nominal geometry. These values are deliberately conservative
    # because this repository is an educational Isaac Lab baseline.
    wheel_radius: float = 0.03
    wheel_base: float = 0.15
    max_wheel_speed: float = 32.0

    left_wheel_sign: float = 1.0
    right_wheel_sign: float = 1.0

    # Allow limited reverse motion for recovery in gate / obstacle cases.
    reverse_speed_fraction: float = 0.35

    # Must cover world_cfg.action_delay_frame_range.
    max_action_delay_frames: int = 3

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    frame_stack: int = 4

    # Single-agent single-frame actor observation layout:
    #   self body velocity                      3
    #   wheel velocity                          2
    #   center-goal body vector / heading        5
    #   slot error in body frame                 3
    #   team heading error                       2
    #   formation one-hot + scale                4
    #   agent id one-hot                         4
    #   teammate relative position               9
    #   teammate relative velocity               6
    #   applied action                           2
    #   action delta                             2
    #   progress ema                             1
    #   center speed                             1
    #   lidar norm                              48
    #   lidar delta                             48
    #   risk features                           16
    #   total                                  156
    single_actor_obs_dim: int = 156

    lin_vel_scale: float = 1.8
    ang_vel_scale: float = 4.0
    wheel_vel_scale: float = 28.0

    rel_pos_norm: float = 9.0
    teammate_dist_norm: float = 3.0
    teammate_vel_scale: float = 1.8
    slot_error_norm: float = 1.5
    progress_norm_scale: float = 0.12
    center_speed_scale: float = 1.5

    obs_clip: float = 10.0
    priv_clip: float = 20.0

    # ------------------------------------------------------------------
    # Reward weights: team navigation
    # ------------------------------------------------------------------
    w_team_progress: float = 5.00
    w_center_speed: float = 0.30
    w_team_heading: float = 0.18

    # ------------------------------------------------------------------
    # Reward weights: formation
    # ------------------------------------------------------------------
    w_formation_mean: float = 0.70
    w_formation_agent: float = 0.35
    w_team_spread: float = 0.10
    w_speed_sync: float = 0.10

    # ------------------------------------------------------------------
    # Reward weights: gate / obstacle / safety
    # ------------------------------------------------------------------
    w_gate_pass: float = 1.20
    w_front_clearance: float = 0.06
    w_obstacle_risk: float = 0.30
    w_gate_risk: float = 0.25
    w_boundary_risk: float = 0.20
    w_pair_risk: float = 0.35

    # ------------------------------------------------------------------
    # Reward weights: regularization
    # ------------------------------------------------------------------
    w_spin: float = 0.020
    w_action_smooth: float = 0.018
    w_action_mag: float = 0.002
    w_wheel_speed: float = 0.001
    w_stuck: float = 0.060
    w_step: float = 0.003

    # ------------------------------------------------------------------
    # Event rewards
    # ------------------------------------------------------------------
    rew_success: float = 60.0
    rew_crash_team: float = -30.0
    rew_crash_agent: float = -8.0
    rew_timeout: float = -8.0

    # ------------------------------------------------------------------
    # Reward clips / safety
    # ------------------------------------------------------------------
    progress_clip: float = 0.35
    continuous_reward_clip: float = 3.00
    max_episode_return_abs: float = 5000.0

    # ------------------------------------------------------------------
    # Stuck detection
    # ------------------------------------------------------------------
    stuck_progress_threshold: float = 0.0015
    stuck_center_speed_threshold: float = 0.035
    stuck_after_steps: int = 80

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    env_spacing: float = 22.0
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
        return float(self.sim_dt * self.decimation)

    @property
    def max_episode_length(self) -> int:
        return int(self.max_episode_length_s / max(self.policy_dt, 1.0e-6))

    @property
    def actor_obs_dim(self) -> int:
        return int(self.single_actor_obs_dim * self.frame_stack)

    @property
    def world_privileged_feature_dim(self) -> int:
        return int(
            Task4WorldManager.privileged_feature_dim(
                max_static_obstacles=int(self.world_cfg.max_static_obstacles),
                num_agents=int(self.world_cfg.num_agents),
            )
        )

    @property
    def critic_obs_dim(self) -> int:
        return int(self.world_privileged_feature_dim)

    @property
    def action_dim_total(self) -> int:
        return int(self.num_agents * self.num_actions_per_agent)

    def validate(self) -> None:
        self.world_cfg.validate()

        assert self.num_envs > 0, f"num_envs must be positive, got {self.num_envs}"
        assert self.num_agents == 4, "Task4 currently expects exactly 4 agents"
        assert self.world_cfg.num_agents == self.num_agents, (
            f"world_cfg.num_agents={self.world_cfg.num_agents} != num_agents={self.num_agents}"
        )

        assert self.num_actions_per_agent == 2, "Task4 action per agent must be [v_norm, w_norm]"

        assert self.sim_dt > 0.0
        assert self.decimation >= 1
        assert self.max_episode_length_s > 0.0
        assert self.policy_dt > 0.0
        assert self.max_episode_length >= 1

        assert self.wheel_radius > 0.0
        assert self.wheel_base > 0.0
        assert self.max_wheel_speed > 0.0
        assert self.reverse_speed_fraction >= 0.0

        assert self.max_action_delay_frames >= int(self.world_cfg.action_delay_frame_range[1]), (
            "max_action_delay_frames must cover world_cfg.action_delay_frame_range"
        )

        assert self.frame_stack >= 1
        assert self.single_actor_obs_dim == 156, (
            f"Task4 single_actor_obs_dim must be 156, got {self.single_actor_obs_dim}"
        )
        assert self.actor_obs_dim == 624, f"Task4 actor_obs_dim must be 624, got {self.actor_obs_dim}"
        assert self.critic_obs_dim == 96, f"Task4 critic_obs_dim must be 96, got {self.critic_obs_dim}"

        assert self.world_cfg.lidar_pool_bins == 48, (
            f"Task4 expects 48 lidar pooled bins, got {self.world_cfg.lidar_pool_bins}"
        )
        assert Task4WorldManager.risk_feature_dim() == 16

        for name in [
            "lin_vel_scale",
            "ang_vel_scale",
            "wheel_vel_scale",
            "rel_pos_norm",
            "teammate_dist_norm",
            "teammate_vel_scale",
            "slot_error_norm",
            "progress_norm_scale",
            "center_speed_scale",
            "obs_clip",
            "priv_clip",
        ]:
            assert float(getattr(self, name)) > 0.0, f"{name} must be positive"

        assert self.continuous_reward_clip > 0.0
        assert self.max_episode_return_abs > 0.0
        assert self.stuck_after_steps >= 0
        assert self.stuck_progress_threshold >= 0.0
        assert self.stuck_center_speed_threshold >= 0.0
        assert self.env_spacing > 0.0
        assert self.spawn_height > 0.0


JetbotTask4Config = Task4Config
DiffDriveTask4Config = Task4Config

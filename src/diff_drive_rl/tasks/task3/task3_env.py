from __future__ import annotations

import math
import warnings
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*getTypes called on non-existent path.*")

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene

from diff_drive_rl.tasks.task3.task3_config import Task3Config
from diff_drive_rl.tasks.task3.task3_scene import make_diff_drive_task3_scene_cfg
from diff_drive_rl.tasks.task3.task3_world import Task3WorldManager


class DiffDriveTask3Env(gym.Env):
    """Diff-Drive UGV / Jetbot Task3: Conservative Sim2Real Parking.

    Task:
        The Jetbot starts from the left side of a short mixed-material track,
        crosses conservative low speed bumps, enters a U-shaped parking spot,
        and completes stable parking.

    World:
        Task3WorldManager provides conservative parking geometry, analytic
        LiDAR, risk features, privileged features, Sim2Real DR buffers, and
        event checks.

    Action:
        action[:, 0] = normalized left wheel velocity command
        action[:, 1] = normalized right wheel velocity command

    Actor observation:
        [num_envs, frame_stack * single_actor_obs_dim] = [N, 404]

    Critic privileged observation:
        actor_obs_stack + 38 world privileged features = [N, 442]
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, cfg: Task3Config):
        super().__init__()

        cfg.validate()
        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.device = str(cfg.device)
        self.dt = float(cfg.policy_dt)

        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

        sim_cfg = sim_utils.SimulationCfg(
            dt=float(self.cfg.sim_dt),
            device=self.device,
            physx=sim_utils.PhysxCfg(
                enable_external_forces_every_iteration=True,
                min_position_iteration_count=4,
                max_position_iteration_count=8,
                min_velocity_iteration_count=1,
                max_velocity_iteration_count=2,
            ),
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)

        # Dome light. Conservative world assets are created in task3_scene.py.
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0)
        light_cfg.func("/World/Light", light_cfg)

        SceneCfg = make_diff_drive_task3_scene_cfg(self.cfg)
        self.scene = InteractiveScene(
            SceneCfg(
                num_envs=int(self.cfg.num_envs),
                env_spacing=float(self.cfg.env_spacing),
            )
        )

        self.sim.reset()
        self.scene.update(0.0)

        try:
            self.robot: Articulation = self.scene["robot"]
        except Exception:
            self.robot = self.scene.articulations["robot"]

        try:
            self.lidar = self.scene.sensors["lidar"]
        except Exception:
            self.lidar = None

        self.env_origins = self.scene.env_origins.to(self.device)

        self._resolve_robot_indices()

        self.world = Task3WorldManager(
            scene=self.scene,
            cfg=self.cfg.world_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        self.world_priv_dim = int(self.world.privileged_feature_dim())

        # Spaces
        self.num_actions = int(self.cfg.num_actions)
        self.num_observations = int(self.cfg.actor_obs_dim)
        self.num_privileged_obs = int(self.cfg.critic_obs_dim)

        if self.num_observations != 404:
            raise RuntimeError(f"[Task3] actor obs dim must be 404, got {self.num_observations}")
        if self.num_privileged_obs != 442:
            raise RuntimeError(f"[Task3] critic obs dim must be 442, got {self.num_privileged_obs}")
        if self.world_priv_dim != 38:
            raise RuntimeError(f"[Task3] world privileged dim must be 38, got {self.world_priv_dim}")

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_observations,),
            dtype=np.float32,
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_privileged_obs,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        # Episode buffers
        self.global_steps = 0

        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Action buffers
        self.raw_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros_like(self.raw_actions)
        self.prev_actions = torch.zeros_like(self.raw_actions)

        self.applied_actions = torch.zeros_like(self.raw_actions)
        self.prev_applied_actions = torch.zeros_like(self.raw_actions)

        self.action_delay_buffer = torch.zeros(
            (
                self.num_envs,
                int(self.cfg.max_action_delay_frames) + 1,
                self.num_actions,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # Progress / event buffers
        self.last_goal_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.progress_ema = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.stuck_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.prev_terrain_progress_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_bump_progress_count = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Observation stack
        self.obs_buffer = torch.zeros(
            (
                self.num_envs,
                int(self.cfg.frame_stack),
                int(self.cfg.single_actor_obs_dim),
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # Episode counters
        self.total_done_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_success_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_crash_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)

        self.reset()

        if bool(self.cfg.print_debug_info):
            self._print_debug_info()

    # ------------------------------------------------------------------
    # Robot mapping
    # ------------------------------------------------------------------
    def _resolve_robot_indices(self) -> None:
        self.robot_joint_names = list(getattr(self.robot, "joint_names", []))
        self.robot_body_names = list(getattr(self.robot, "body_names", []))

        wheel_ids = []

        try:
            ids, _ = self.robot.find_joints(".*wheel_joint")
            wheel_ids = [int(i) for i in ids]
        except Exception:
            wheel_ids = []

        if len(wheel_ids) < 2:
            print("[WARN][Task3] 未找到 .*wheel_joint，fallback to first 2 joints.", flush=True)
            wheel_ids = [0, 1]

        self.wheel_joint_ids = wheel_ids[:2]
        self.wheel_joint_ids_t = torch.tensor(self.wheel_joint_ids, dtype=torch.long, device=self.device)

        self.wheel_signs = torch.tensor(
            [float(self.cfg.left_wheel_sign), float(self.cfg.right_wheel_sign)],
            dtype=torch.float32,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset(
        self,
        env_ids: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            full_reset = True
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            full_reset = int(env_ids.numel()) == self.num_envs

        if env_ids.numel() == 0:
            obs = self.compute_obs()
            return obs if full_reset else obs[env_ids], {}

        # 1. Reset world tensors.
        self.world.reset_world(env_ids)

        # 2. Reset robot root pose.
        root_state = self.robot.data.default_root_state[env_ids].clone()

        root_state[:, :3] += self.env_origins[env_ids]
        root_state[:, 0:2] = self.env_origins[env_ids, :2] + self.world.start_pos[env_ids]
        root_state[:, 2] = self.env_origins[env_ids, 2] + float(self.cfg.spawn_height)
        root_state[:, 3:7] = self._yaw_to_quat_wxyz(self.world.start_yaw[env_ids])
        root_state[:, 7:13] = 0.0

        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.reset(env_ids)

        # 3. Reset buffers.
        self.episode_steps[env_ids] = 0
        self.episode_return[env_ids] = 0.0

        self.raw_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        self.applied_actions[env_ids] = 0.0
        self.prev_applied_actions[env_ids] = 0.0
        self.action_delay_buffer[env_ids] = 0.0

        self.progress_ema[env_ids] = 0.0
        self.stuck_counter[env_ids] = 0
        self.success_hold_counter[env_ids] = 0

        self.prev_terrain_progress_count[env_ids] = 0.0
        self.prev_bump_progress_count[env_ids] = 0.0
        self.obs_buffer[env_ids] = 0.0

        # 4. Sync scene and initialize derived buffers.
        self.scene.update(0.0)

        root_local = self._root_pos_local()
        yaw = self._yaw()

        goal_terms = self.world.compute_goal_terms(root_local, yaw)
        self.last_goal_dist[env_ids] = goal_terms["goal_dist"][env_ids]

        milestones = self.world.compute_milestones(root_local)
        self.prev_terrain_progress_count[env_ids] = milestones["terrain_progress_count"][env_ids]
        self.prev_bump_progress_count[env_ids] = milestones["bump_progress_count"][env_ids]

        # Warm observation stack.
        obs_single = self._compute_single_actor_obs(update_lidar_history=True)

        for i in range(int(self.cfg.frame_stack)):
            self.obs_buffer[env_ids, i, :] = obs_single[env_ids]

        obs = self.compute_obs()

        return obs if full_reset else obs[env_ids], {}

    @torch.no_grad()
    def step(self, actions_nn: torch.Tensor):
        actions_nn = torch.as_tensor(actions_nn, dtype=torch.float32, device=self.device)
        actions_nn = torch.nan_to_num(actions_nn, nan=0.0, posinf=1.0, neginf=-1.0)
        actions_nn = torch.clamp(actions_nn, -1.0, 1.0)

        self.prev_actions = self.raw_actions.clone()
        self.raw_actions = actions_nn.clone()

        pre_goal_dist = self.last_goal_dist.clone()

        # Sim2Real action model.
        applied = self._apply_action_model(actions_nn)
        self.prev_applied_actions = self.applied_actions.clone()
        self.applied_actions = applied.clone()
        self.actions = applied.clone()

        # Wheel velocity target.
        wheel_vel_targets = applied * self.wheel_signs.unsqueeze(0) * float(self.cfg.max_wheel_speed)

        full_joint_vel_targets = torch.zeros(
            (self.num_envs, self.robot.num_joints),
            dtype=torch.float32,
            device=self.device,
        )
        full_joint_vel_targets[:, self.wheel_joint_ids_t] = wheel_vel_targets

        self.robot.set_joint_velocity_target(full_joint_vel_targets)

        for _ in range(int(self.cfg.decimation)):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(float(self.cfg.sim_dt))

        self.episode_steps += 1
        self.global_steps += self.num_envs

        reward, terminated, truncated, info = self._compute_rewards_and_dones(pre_goal_dist=pre_goal_dist)
        done = terminated | truncated

        obs_single = self._compute_single_actor_obs(update_lidar_history=True)
        self.obs_buffer = torch.roll(self.obs_buffer, shifts=-1, dims=1)
        self.obs_buffer[:, -1, :] = obs_single
        obs = self.compute_obs()

        self.episode_return += reward

        if done.any():
            reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
            info["terminal_observation"] = obs[reset_ids].clone()
            self.reset(reset_ids)
            obs = self.compute_obs().clone()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Sim2Real action model
    # ------------------------------------------------------------------
    def _apply_action_model(self, actions_nn: torch.Tensor) -> torch.Tensor:
        """Apply Sim2Real action path.

        raw action -> delay -> deadband -> motor bias -> EMA
        -> motor strength -> wheel radius scale
        """

        # Shift delay buffer and insert newest action at index 0.
        self.action_delay_buffer = torch.roll(self.action_delay_buffer, shifts=1, dims=1)
        self.action_delay_buffer[:, 0, :] = actions_nn

        delay = torch.clamp(
            self.world.action_delay_frames,
            min=0,
            max=int(self.cfg.max_action_delay_frames),
        )

        delayed = self.action_delay_buffer[
            torch.arange(self.num_envs, dtype=torch.long, device=self.device),
            delay,
            :,
        ]

        # Deadband in normalized action space.
        deadband = torch.clamp(self.world.action_deadband.unsqueeze(-1), 0.0, 0.95)
        abs_a = torch.abs(delayed)
        sign_a = torch.sign(delayed)

        after_deadband = torch.where(
            abs_a > deadband,
            sign_a * (abs_a - deadband) / torch.clamp(1.0 - deadband, min=1.0e-6),
            torch.zeros_like(delayed),
        )

        # Motor bias in normalized action space.
        biased = torch.clamp(after_deadband + self.world.motor_bias, -1.0, 1.0)

        # EMA smoothing, alpha is per-env.
        alpha = torch.clamp(self.world.action_ema_alpha.unsqueeze(-1), 0.0, 1.0)
        ema = alpha * biased + (1.0 - alpha) * self.applied_actions

        # Motor strength and wheel radius scale.
        scaled = ema * self.world.motor_strength * self.world.wheel_radius_scale

        return torch.nan_to_num(
            torch.clamp(scaled, -1.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def compute_obs(self) -> torch.Tensor:
        obs = self.obs_buffer.reshape(self.num_envs, -1)

        if obs.shape[-1] != self.num_observations:
            raise RuntimeError(
                f"[Task3] actor obs dim mismatch: got {obs.shape[-1]}, expected {self.num_observations}"
            )

        return torch.nan_to_num(
            torch.clamp(obs, -float(self.cfg.obs_clip), float(self.cfg.obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.obs_clip),
            neginf=-float(self.cfg.obs_clip),
        )

    def compute_privileged_obs(self) -> torch.Tensor:
        actor_obs = self.compute_obs()
        root_local = self._root_pos_local()
        yaw = self._yaw()

        priv = self.world.compute_privileged_features(root_local, yaw)
        priv = torch.nan_to_num(
            torch.clamp(priv, -float(self.cfg.priv_clip), float(self.cfg.priv_clip)),
            nan=0.0,
            posinf=float(self.cfg.priv_clip),
            neginf=-float(self.cfg.priv_clip),
        )

        critic_obs = torch.cat([actor_obs, priv], dim=-1)

        if critic_obs.shape[-1] != self.num_privileged_obs:
            raise RuntimeError(
                f"[Task3] critic obs dim mismatch: got {critic_obs.shape[-1]}, expected {self.num_privileged_obs}"
            )

        return critic_obs

    def get_privileged_observations(self) -> torch.Tensor:
        return self.compute_privileged_obs()

    def _compute_states(self) -> torch.Tensor:
        return self.compute_privileged_obs()

    def _compute_single_actor_obs(self, update_lidar_history: bool = True) -> torch.Tensor:
        root_local = self._root_pos_local()
        yaw = self._yaw()

        base_lin_vel_b = self.robot.data.root_lin_vel_b
        base_ang_vel_b = self.robot.data.root_ang_vel_b
        wheel_vel = self.robot.data.joint_vel[:, self.wheel_joint_ids_t]

        goal = self.world.compute_goal_terms(root_local, yaw)

        # Training uses analytic 2D LiDAR for deterministic and stable world
        # interaction. RayCaster is registered in the scene for compatibility.
        lidar = self.world.compute_analytic_lidar(
            root_pos_local=root_local,
            yaw=yaw,
            add_noise=True,
            update_history=update_lidar_history,
            normalize=False,
        )

        lidar_norm = torch.clamp(
            lidar / max(float(self.cfg.world_cfg.lidar_max_distance), 1.0e-6),
            0.0,
            1.0,
        )
        lidar_delta = self.world.last_lidar_delta

        risk = self.world.compute_risk_features(root_local, yaw, lidar_pooled=lidar)

        vel_obs = torch.stack(
            [
                torch.clamp(base_lin_vel_b[:, 0] / float(self.cfg.lin_vel_scale), -5.0, 5.0),
                torch.clamp(base_lin_vel_b[:, 1] / float(self.cfg.lin_vel_scale), -5.0, 5.0),
                torch.clamp(base_ang_vel_b[:, 2] / float(self.cfg.ang_vel_scale), -5.0, 5.0),
            ],
            dim=-1,
        )

        wheel_obs = torch.clamp(
            wheel_vel / max(float(self.cfg.wheel_vel_scale), 1.0e-6),
            -5.0,
            5.0,
        )

        goal_dist_norm = goal["goal_dist_norm"].unsqueeze(-1)

        goal_xy_body = torch.stack(
            [
                goal["goal_x_body_norm"],
                goal["goal_y_body_norm"],
            ],
            dim=-1,
        )

        heading_obs = torch.stack(
            [
                goal["heading_sin"],
                goal["heading_cos"],
            ],
            dim=-1,
        )

        goal_yaw_obs = torch.stack(
            [
                goal["goal_yaw_sin"],
                goal["goal_yaw_cos"],
            ],
            dim=-1,
        )

        parking_obs = torch.stack(
            [
                goal["parking_x_norm"],
                goal["parking_y_norm"],
            ],
            dim=-1,
        )

        action_delta = self.applied_actions - self.prev_applied_actions

        progress_obs = torch.clamp(
            self.progress_ema / max(float(self.cfg.progress_norm_scale), 1.0e-6),
            -5.0,
            5.0,
        ).unsqueeze(-1)

        obs = torch.cat(
            [
                vel_obs,               # 3
                wheel_obs,             # 2
                goal_dist_norm,        # 1
                goal_xy_body,          # 2
                heading_obs,           # 2
                goal_yaw_obs,          # 2
                parking_obs,           # 2
                self.applied_actions,  # 2
                action_delta,          # 2
                progress_obs,          # 1
                lidar_norm,            # 36
                lidar_delta,           # 36
                risk,                  # 10
            ],
            dim=-1,
        )

        if obs.shape[-1] != int(self.cfg.single_actor_obs_dim):
            raise RuntimeError(
                f"[Task3] single actor obs dim mismatch: got {obs.shape[-1]}, expected {self.cfg.single_actor_obs_dim}"
            )

        return torch.nan_to_num(
            torch.clamp(obs, -float(self.cfg.obs_clip), float(self.cfg.obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.obs_clip),
            neginf=-float(self.cfg.obs_clip),
        )

    # ------------------------------------------------------------------
    # Reward / done
    # ------------------------------------------------------------------
    def _compute_rewards_and_dones(self, pre_goal_dist: torch.Tensor):
        root_local = self._root_pos_local()
        yaw = self._yaw()

        base_lin_vel_w = self._root_lin_vel_w()
        base_lin_vel_b = self.robot.data.root_lin_vel_b
        base_ang_vel_b = self.robot.data.root_ang_vel_b

        goal = self.world.compute_goal_terms(root_local, yaw)

        events = self.world.check_events(
            root_pos_local=root_local,
            yaw=yaw,
            body_lin_vel=base_lin_vel_b,
            body_ang_vel=base_ang_vel_b,
        )

        milestones = self.world.compute_milestones(root_local)

        current_goal_dist = events["goal_dist"]
        progress = pre_goal_dist - current_goal_dist
        progress_clamped = torch.clamp(
            progress,
            -float(self.cfg.progress_clip),
            float(self.cfg.progress_clip),
        )

        self.progress_ema = 0.90 * self.progress_ema + 0.10 * progress
        self.last_goal_dist = current_goal_dist.detach().clone()

        goal_vec_w = goal["goal_vec_w"]
        goal_dir_w = goal_vec_w / torch.clamp(current_goal_dist.unsqueeze(-1), min=1.0e-6)
        goal_aligned_speed = torch.sum(base_lin_vel_w[:, :2] * goal_dir_w, dim=-1)

        heading_cos = goal["heading_cos"]
        heading_gate = torch.clamp(heading_cos, 0.0, 1.0)

        # Desired speed: faster far from the spot, slow near parking.
        desired_speed = torch.where(
            current_goal_dist > 1.5,
            torch.full_like(current_goal_dist, 0.55),
            torch.clamp(0.30 * current_goal_dist, min=0.06, max=0.35),
        )

        speed_error = torch.abs(goal_aligned_speed - desired_speed)
        forward_gate = torch.clamp(
            goal_aligned_speed / torch.clamp(desired_speed, min=0.10),
            0.0,
            1.0,
        )

        parking_pos_error = torch.norm(
            torch.stack(
                [
                    events["parking_x"],
                    events["parking_y"],
                ],
                dim=-1,
            ),
            dim=-1,
        )
        parking_yaw_error = events["goal_yaw_error_abs"]

        body_speed = events["body_speed"]
        yaw_rate_abs = events["yaw_rate_abs"]

        risk = self.world.compute_risk_features(root_local, yaw, lidar_pooled=self.world.last_lidar)

        all_risk = risk[:, 0]
        front_risk = risk[:, 1]
        lane_risk = risk[:, 4]
        wall_risk = risk[:, 5]
        bump_risk = risk[:, 6]
        front_clearance = risk[:, 7]
        inside_box = risk[:, 9]

        # Reward terms
        r_progress = progress_clamped
        r_goal_speed = torch.exp(-2.0 * torch.square(speed_error)) * heading_gate * forward_gate
        r_heading = heading_gate

        # Parking shaping becomes dominant near the parking zone.
        near_parking = (root_local[:, 0] > float(self.cfg.world_cfg.bound_asphalt_park[0])).float()

        r_parking_pos = torch.exp(-4.0 * torch.square(parking_pos_error)) * near_parking
        r_parking_yaw = torch.exp(-2.0 * torch.square(parking_yaw_error)) * near_parking
        r_inside_box = inside_box * near_parking

        low_speed_score = (
            torch.exp(-8.0 * torch.square(body_speed))
            * torch.exp(-4.0 * torch.square(yaw_rate_abs))
        )
        r_parking_low_speed = low_speed_score * inside_box

        terrain_progress_delta = torch.clamp(
            milestones["terrain_progress_count"] - self.prev_terrain_progress_count,
            min=0.0,
            max=1.0,
        )
        bump_progress_delta = torch.clamp(
            milestones["bump_progress_count"] - self.prev_bump_progress_count,
            min=0.0,
            max=1.0,
        )

        self.prev_terrain_progress_count = milestones["terrain_progress_count"].detach().clone()
        self.prev_bump_progress_count = milestones["bump_progress_count"].detach().clone()

        r_terrain_progress = terrain_progress_delta
        r_bump_progress = bump_progress_delta

        bump_overlap = events["bump_overlap"].float()
        p_bump_smooth = -bump_overlap * (
            torch.square(base_lin_vel_b[:, 2])
            + 0.25 * torch.square(base_ang_vel_b[:, 2])
        )

        r_front_clearance = front_clearance
        p_lidar_risk = -torch.maximum(all_risk, front_risk)
        p_wall_risk = -wall_risk
        p_lane_risk = -lane_risk
        p_bump_risk = -bump_risk

        p_spin = -torch.square(base_ang_vel_b[:, 2])
        p_action_smooth = -torch.mean(torch.square(self.applied_actions - self.prev_applied_actions), dim=-1)
        p_action_mag = -torch.mean(torch.square(self.applied_actions), dim=-1)

        wheel_vel = self.robot.data.joint_vel[:, self.wheel_joint_ids_t]
        p_wheel_speed = -torch.mean(
            torch.square(wheel_vel / max(float(self.cfg.wheel_vel_scale), 1.0e-6)),
            dim=-1,
        )

        low_progress = torch.abs(progress) < float(self.cfg.stuck_progress_threshold)
        low_speed = torch.abs(goal_aligned_speed) < float(self.cfg.stuck_speed_threshold)
        far_from_goal = current_goal_dist > float(self.cfg.world_cfg.success_pos_tol)

        stuck_now = (
            low_progress
            & low_speed
            & far_from_goal
            & (self.episode_steps > int(self.cfg.stuck_after_steps))
        )

        self.stuck_counter = torch.where(
            stuck_now,
            self.stuck_counter + 1,
            torch.zeros_like(self.stuck_counter),
        )
        p_stuck = -torch.clamp(self.stuck_counter.float() / 80.0, 0.0, 1.0)

        r_step = -torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        continuous_raw = (
            float(self.cfg.w_progress) * r_progress
            + float(self.cfg.w_goal_speed) * r_goal_speed
            + float(self.cfg.w_heading) * r_heading
            + float(self.cfg.w_parking_pos) * r_parking_pos
            + float(self.cfg.w_parking_yaw) * r_parking_yaw
            + float(self.cfg.w_inside_box) * r_inside_box
            + float(self.cfg.w_parking_low_speed) * r_parking_low_speed
            + float(self.cfg.w_terrain_progress) * r_terrain_progress
            + float(self.cfg.w_bump_progress) * r_bump_progress
            + float(self.cfg.w_bump_smooth) * p_bump_smooth
            + float(self.cfg.w_front_clearance) * r_front_clearance
            + float(self.cfg.w_lidar_risk) * p_lidar_risk
            + float(self.cfg.w_wall_risk) * p_wall_risk
            + float(self.cfg.w_lane_risk) * p_lane_risk
            + float(self.cfg.w_bump_risk) * p_bump_risk
            + float(self.cfg.w_spin) * p_spin
            + float(self.cfg.w_action_smooth) * p_action_smooth
            + float(self.cfg.w_action_mag) * p_action_mag
            + float(self.cfg.w_wheel_speed) * p_wheel_speed
            + float(self.cfg.w_stuck) * p_stuck
            + float(self.cfg.w_step) * r_step
        )

        continuous = torch.clamp(
            continuous_raw,
            -float(self.cfg.continuous_reward_clip),
            float(self.cfg.continuous_reward_clip),
        )

        # Events
        success_candidate = events["success_candidate"]
        crash = events["crash"]

        self.success_hold_counter = torch.where(
            success_candidate,
            self.success_hold_counter + 1,
            torch.zeros_like(self.success_hold_counter),
        )

        stable_success = self.success_hold_counter >= int(self.cfg.world_cfg.success_hold_steps)

        timeout = self.episode_steps >= int(self.cfg.max_episode_length)

        terminated = stable_success | crash
        truncated = timeout & (~terminated)

        event_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        event_reward = torch.where(
            stable_success,
            torch.full_like(event_reward, float(self.cfg.rew_success)),
            event_reward,
        )
        event_reward = torch.where(
            crash,
            torch.full_like(event_reward, float(self.cfg.rew_crash)),
            event_reward,
        )
        event_reward = torch.where(
            truncated,
            torch.full_like(event_reward, float(self.cfg.rew_timeout)),
            event_reward,
        )

        reward_raw = continuous + event_reward

        projected_return = self.episode_return + reward_raw
        no_event = event_reward.abs() < 1.0e-6

        reward = torch.where(
            (projected_return > float(self.cfg.max_episode_return_abs)) & no_event,
            float(self.cfg.max_episode_return_abs) - self.episode_return,
            reward_raw,
        )
        reward = torch.where(
            (projected_return < -float(self.cfg.max_episode_return_abs)) & no_event,
            -float(self.cfg.max_episode_return_abs) - self.episode_return,
            reward,
        )
        reward = torch.nan_to_num(reward, nan=0.0, posinf=50.0, neginf=-50.0)

        done = terminated | truncated

        # Episode counters
        done_count = done.float().sum()
        success_count = stable_success.float().sum()
        crash_count = crash.float().sum()
        timeout_count = truncated.float().sum()

        self.total_done_episodes += done_count.detach()
        self.total_success_episodes += success_count.detach()
        self.total_crash_episodes += crash_count.detach()
        self.total_timeout_episodes += timeout_count.detach()

        denom = torch.clamp(self.total_done_episodes, min=1.0)

        episode_success_rate = self.total_success_episodes / denom
        episode_crash_rate = self.total_crash_episodes / denom
        episode_timeout_rate = self.total_timeout_episodes / denom

        world_stats = self.world.get_debug_stats(root_local, yaw)

        info = {
            "reward_components": {
                "R_Progress": (float(self.cfg.w_progress) * r_progress).mean().item(),
                "R_Goal_Speed": (float(self.cfg.w_goal_speed) * r_goal_speed).mean().item(),
                "R_Heading": (float(self.cfg.w_heading) * r_heading).mean().item(),
                "R_Parking_Pos": (float(self.cfg.w_parking_pos) * r_parking_pos).mean().item(),
                "R_Parking_Yaw": (float(self.cfg.w_parking_yaw) * r_parking_yaw).mean().item(),
                "R_Inside_Box": (float(self.cfg.w_inside_box) * r_inside_box).mean().item(),
                "R_Parking_Low_Speed": (float(self.cfg.w_parking_low_speed) * r_parking_low_speed).mean().item(),
                "R_Terrain_Progress": (float(self.cfg.w_terrain_progress) * r_terrain_progress).mean().item(),
                "R_Bump_Progress": (float(self.cfg.w_bump_progress) * r_bump_progress).mean().item(),
                "P_Bump_Smooth": (float(self.cfg.w_bump_smooth) * p_bump_smooth).mean().item(),
                "R_Front_Clearance": (float(self.cfg.w_front_clearance) * r_front_clearance).mean().item(),
                "P_Lidar_Risk": (float(self.cfg.w_lidar_risk) * p_lidar_risk).mean().item(),
                "P_Wall_Risk": (float(self.cfg.w_wall_risk) * p_wall_risk).mean().item(),
                "P_Lane_Risk": (float(self.cfg.w_lane_risk) * p_lane_risk).mean().item(),
                "P_Bump_Risk": (float(self.cfg.w_bump_risk) * p_bump_risk).mean().item(),
                "P_Spin": (float(self.cfg.w_spin) * p_spin).mean().item(),
                "P_Action_Smooth": (float(self.cfg.w_action_smooth) * p_action_smooth).mean().item(),
                "P_Action_Mag": (float(self.cfg.w_action_mag) * p_action_mag).mean().item(),
                "P_Wheel_Speed": (float(self.cfg.w_wheel_speed) * p_wheel_speed).mean().item(),
                "P_Stuck": (float(self.cfg.w_stuck) * p_stuck).mean().item(),
                "Step": (float(self.cfg.w_step) * r_step).mean().item(),
                "Continuous": continuous.mean().item(),
                "Event": event_reward.mean().item(),
                "Total": reward.mean().item(),
            },
            "events": {
                "Success_Rate": stable_success.float().mean().item(),
                "Success_Candidate_Rate": success_candidate.float().mean().item(),
                "Crash_Rate": crash.float().mean().item(),
                "Out_Of_Lane_Rate": events["out_of_lane"].float().mean().item(),
                "Parking_Wall_Collision_Rate": events["parking_wall_collision"].float().mean().item(),
                "Bump_Overlap_Rate": events["bump_overlap"].float().mean().item(),
                "Timeout_Rate": truncated.float().mean().item(),
                "Done_Rate": done.float().mean().item(),
                "Episode_Success_Rate": episode_success_rate.item(),
                "Episode_Crash_Rate": episode_crash_rate.item(),
                "Episode_Timeout_Rate": episode_timeout_rate.item(),
                "Episode_Done_Count": self.total_done_episodes.item(),
            },
            "telemetry": {
                "Goal_Dist": current_goal_dist.mean().item(),
                "Progress": progress.mean().item(),
                "Progress_EMA": self.progress_ema.mean().item(),
                "Goal_Aligned_Speed": goal_aligned_speed.mean().item(),
                "Desired_Speed": desired_speed.mean().item(),
                "Speed_Error": speed_error.mean().item(),
                "Heading_Error": torch.abs(goal["heading_error"]).mean().item(),
                "Heading_Cos": heading_cos.mean().item(),
                "Goal_Yaw_Error": parking_yaw_error.mean().item(),
                "Parking_Pos_Error": parking_pos_error.mean().item(),
                "Parking_X": events["parking_x"].mean().item(),
                "Parking_Y": events["parking_y"].mean().item(),
                "Inside_Box": events["inside_parking_box"].float().mean().item(),
                "Success_Hold": self.success_hold_counter.float().mean().item(),
                "Body_Vx": base_lin_vel_b[:, 0].mean().item(),
                "Body_Vy": base_lin_vel_b[:, 1].mean().item(),
                "Body_Wz": base_ang_vel_b[:, 2].mean().item(),
                "Body_Speed": body_speed.mean().item(),
                "Yaw_Rate_Abs": yaw_rate_abs.mean().item(),
                "Lidar_Min": self.world.last_lidar.min().item(),
                "Lidar_Mean": self.world.last_lidar.mean().item(),
                "Risk_All": all_risk.mean().item(),
                "Risk_Front": front_risk.mean().item(),
                "Risk_Lane": lane_risk.mean().item(),
                "Risk_Wall": wall_risk.mean().item(),
                "Risk_Bump": bump_risk.mean().item(),
                "Front_Clearance": front_clearance.mean().item(),
                "Terrain_Progress_Count": milestones["terrain_progress_count"].mean().item(),
                "Bump_Progress_Count": milestones["bump_progress_count"].mean().item(),
                "Current_Terrain_ID": self.world.terrain_id(root_local[:, 0]).float().mean().item(),
                "Action_Left": self.applied_actions[:, 0].mean().item(),
                "Action_Right": self.applied_actions[:, 1].mean().item(),
                "Raw_Action_Left": self.raw_actions[:, 0].mean().item(),
                "Raw_Action_Right": self.raw_actions[:, 1].mean().item(),
                "Wheel_Vel_Left": wheel_vel[:, 0].mean().item(),
                "Wheel_Vel_Right": wheel_vel[:, 1].mean().item(),
                "Action_Delay": self.world.action_delay_frames.float().mean().item(),
                "Action_Deadband": self.world.action_deadband.mean().item(),
                "Action_EMA": self.world.action_ema_alpha.mean().item(),
                "Motor_Strength": self.world.motor_strength.mean().item(),
                "Wheel_Radius_Scale": self.world.wheel_radius_scale.mean().item(),
                "Stuck_Ratio": stuck_now.float().mean().item(),
                "Episode_Length": self.episode_steps.float().mean().item(),
                "Episode_Return": self.episode_return.mean().item(),
            },
            "world": world_stats,
            "debug": {
                "Actor_Obs_Dim": float(self.num_observations),
                "Single_Actor_Obs_Dim": float(self.cfg.single_actor_obs_dim),
                "Privileged_Feature_Dim": float(self.world_priv_dim),
                "Critic_Obs_Dim": float(self.num_privileged_obs),
                "Action_Dim": float(self.num_actions),
                "Reward_Min": reward.min().item(),
                "Reward_Max": reward.max().item(),
                "Continuous_Min": continuous.min().item(),
                "Continuous_Max": continuous.max().item(),
                "Event_Min": event_reward.min().item(),
                "Event_Max": event_reward.max().item(),
                "Root_X_Local": root_local[:, 0].mean().item(),
                "Root_Y_Local": root_local[:, 1].mean().item(),
                "Base_Height_Mean": self.robot.data.root_pos_w[:, 2].mean().item(),
            },
            "is_success": stable_success.detach().clone(),
        }

        return reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Geometry / state helpers
    # ------------------------------------------------------------------
    def _root_pos_local(self) -> torch.Tensor:
        root_pos_w = self.robot.data.root_pos_w
        local = root_pos_w - self.env_origins
        return local[:, :2]

    def _yaw(self) -> torch.Tensor:
        return self._quat_yaw(self.robot.data.root_quat_w)

    @staticmethod
    def _quat_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
        w = quat_wxyz[:, 0]
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        z = quat_wxyz[:, 3]

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

        return torch.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
        quat = torch.zeros((yaw.shape[0], 4), dtype=torch.float32, device=yaw.device)
        quat[:, 0] = torch.cos(0.5 * yaw)
        quat[:, 3] = torch.sin(0.5 * yaw)
        return quat

    def _root_lin_vel_w(self) -> torch.Tensor:
        if hasattr(self.robot.data, "root_lin_vel_w"):
            return self.robot.data.root_lin_vel_w

        vel_b = self.robot.data.root_lin_vel_b
        yaw = self._yaw()

        c = torch.cos(yaw)
        s = torch.sin(yaw)

        vx_w = c * vel_b[:, 0] - s * vel_b[:, 1]
        vy_w = s * vel_b[:, 0] + c * vel_b[:, 1]

        return torch.stack([vx_w, vy_w, vel_b[:, 2]], dim=-1)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def _print_debug_info(self) -> None:
        print("\n" + "=" * 120)
        print("✅ [Task3] Diff-Drive UGV / Jetbot Sim2Real Parking Env Initialized")
        print(f"  num_envs               : {self.num_envs}")
        print(f"  device                 : {self.device}")
        print(f"  robot.num_joints       : {self.robot.num_joints}")
        print(f"  num_actions            : {self.num_actions}")
        print(f"  single_actor_obs_dim   : {self.cfg.single_actor_obs_dim}")
        print(f"  frame_stack            : {self.cfg.frame_stack}")
        print(f"  actor_obs_dim          : {self.num_observations}")
        print(f"  privileged_feature_dim : {self.world_priv_dim}")
        print(f"  critic_obs_dim         : {self.num_privileged_obs}")
        print(f"  sim_dt                 : {self.cfg.sim_dt}")
        print(f"  policy_dt              : {self.dt}")
        print(f"  decimation             : {self.cfg.decimation}")
        print(f"  max_episode_length_s   : {self.cfg.max_episode_length_s}")
        print(f"  max_episode_length     : {self.cfg.max_episode_length}")
        print(f"  lidar_pool_bins        : {self.cfg.world_cfg.lidar_pool_bins}")
        print(f"  risk_feature_dim       : {self.world.risk_feature_dim()}")
        print(f"  wheel_joint_ids        : {self.wheel_joint_ids}")

        if self.robot_joint_names:
            print("  robot.joint_names:")
            for i, name in enumerate(self.robot_joint_names):
                mark = " <wheel>" if i in self.wheel_joint_ids else ""
                print(f"    {i:02d}: {name}{mark}")

        print("=" * 120 + "\n")


JetbotTask3Env = DiffDriveTask3Env
Task3Env = DiffDriveTask3Env
DiffDriveSim2RealParkingEnv = DiffDriveTask3Env

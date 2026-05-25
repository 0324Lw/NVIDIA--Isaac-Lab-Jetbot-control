from __future__ import annotations

import warnings
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*getTypes called on non-existent path.*")

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from diff_drive_rl.tasks.task4.task4_config import Task4Config
from diff_drive_rl.tasks.task4.task4_scene import (
    make_diff_drive_task4_scene_cfg,
    spawn_task4_world_assets,
)
from diff_drive_rl.tasks.task4.task4_world import Task4WorldManager


class DiffDriveTask4Env(gym.Env):
    """Diff-Drive UGV / Jetbot Task4: Multi-UGV Formation Escort.

    Task:
        Four Jetbot differential-drive UGVs escort a virtual formation center
        to a shared team goal while keeping formation, avoiding obstacles,
        avoiding narrow-gate walls, avoiding arena boundaries, and avoiding
        teammate collisions.

    Interface:
        action:
            [num_envs, 4, 2]
            action[..., 0] = normalized linear velocity command
            action[..., 1] = normalized yaw rate command

        actor observation:
            [num_envs, 4, 624]

        critic state:
            [num_envs, 96]

        reward:
            [num_envs, 4]

        terminated / truncated:
            [num_envs]

    This file is environment only. It does not contain PPO / skrl / SB3 logic.
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, cfg: Task4Config):
        super().__init__()

        cfg.validate()
        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.num_agents = int(cfg.num_agents)
        self.device = str(cfg.device)
        self.dt = float(cfg.policy_dt)
        self.curriculum_stage = int(cfg.curriculum_stage)

        if self.num_agents != 4:
            raise RuntimeError("Task4 currently expects exactly 4 Jetbot agents.")

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

        light_cfg = sim_utils.DomeLightCfg(intensity=2800.0)
        light_cfg.func("/World/Light", light_cfg)

        SceneCfg = make_diff_drive_task4_scene_cfg(self.cfg)
        scene_cfg = SceneCfg(
            num_envs=int(self.num_envs),
            env_spacing=float(self.cfg.env_spacing),
        )

        spawn_task4_world_assets(scene_cfg, self.cfg)

        self.scene = InteractiveScene(scene_cfg)

        self.sim.reset()
        self.scene.update(0.0)

        self.robots = []
        for agent_id in range(self.num_agents):
            name = f"robot_{agent_id}"
            try:
                robot = self.scene[name]
            except Exception:
                robot = self.scene.articulations[name]
            self.robots.append(robot)

        self.lidars = []
        for agent_id in range(self.num_agents):
            name = f"lidar_{agent_id}"
            try:
                self.lidars.append(self.scene.sensors[name])
            except Exception:
                self.lidars.append(None)

        self.env_origins = self.scene.env_origins.to(self.device)

        self._resolve_robot_indices()

        self.world = Task4WorldManager(
            scene=self.scene,
            cfg=self.cfg.world_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )
        self.world.set_curriculum_stage(self.curriculum_stage)

        self.world_priv_dim = int(
            self.world.privileged_feature_dim(
                max_static_obstacles=int(self.cfg.world_cfg.max_static_obstacles),
                num_agents=int(self.cfg.world_cfg.num_agents),
            )
        )

        # Spaces
        self.num_actions = int(self.cfg.num_actions_per_agent)
        self.num_observations = int(self.cfg.actor_obs_dim)
        self.num_privileged_obs = int(self.world_priv_dim)

        if self.num_actions != 2:
            raise RuntimeError(f"[Task4] num_actions_per_agent must be 2, got {self.num_actions}")
        if self.num_observations != 624:
            raise RuntimeError(f"[Task4] actor obs dim must be 624, got {self.num_observations}")
        if self.num_privileged_obs != 96:
            raise RuntimeError(f"[Task4] critic state dim must be 96, got {self.num_privileged_obs}")

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, self.num_observations),
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
            shape=(self.num_agents, self.num_actions),
            dtype=np.float32,
        )

        # Episode buffers
        self.global_steps = 0

        self.episode_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.episode_return = torch.zeros(
            (self.num_envs, self.num_agents),
            dtype=torch.float32,
            device=self.device,
        )

        # Action buffers
        self.raw_actions = torch.zeros(
            (self.num_envs, self.num_agents, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.zeros_like(self.raw_actions)
        self.prev_actions = torch.zeros_like(self.raw_actions)

        self.applied_actions = torch.zeros_like(self.raw_actions)
        self.prev_applied_actions = torch.zeros_like(self.raw_actions)

        self.action_delay_buffer = torch.zeros(
            (
                self.num_envs,
                self.num_agents,
                int(self.cfg.max_action_delay_frames) + 1,
                self.num_actions,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # Progress / events
        self.last_center_goal_dist = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.progress_ema = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.stuck_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.success_hold_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.prev_gate_passed = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        # Observation stack
        self.obs_buffer = torch.zeros(
            (
                self.num_envs,
                self.num_agents,
                int(self.cfg.frame_stack),
                int(self.cfg.single_actor_obs_dim),
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.agent_id_onehot = torch.eye(
            self.num_agents,
            dtype=torch.float32,
            device=self.device,
        ).view(1, self.num_agents, self.num_agents)

        self.wheel_signs = torch.tensor(
            [
                float(self.cfg.left_wheel_sign),
                float(self.cfg.right_wheel_sign),
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 2)

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
        self.robot_joint_names = []
        self.robot_body_names = []
        self.wheel_joint_ids = []
        self.wheel_joint_ids_t = []

        for agent_id, robot in enumerate(self.robots):
            joint_names = list(getattr(robot, "joint_names", []))
            body_names = list(getattr(robot, "body_names", []))

            if agent_id == 0:
                self.robot_joint_names = joint_names
                self.robot_body_names = body_names

            wheel_ids = []

            try:
                ids, _ = robot.find_joints(".*wheel_joint")
                wheel_ids = [int(i) for i in ids]
            except Exception:
                wheel_ids = []

            if len(wheel_ids) < 2:
                print(
                    f"[WARN][Task4] Robot_{agent_id} 未找到 .*wheel_joint，fallback to first 2 joints.",
                    flush=True,
                )
                wheel_ids = [0, 1]

            wheel_ids = wheel_ids[:2]
            self.wheel_joint_ids.append(wheel_ids)
            self.wheel_joint_ids_t.append(torch.tensor(wheel_ids, dtype=torch.long, device=self.device))

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------
    def set_curriculum_stage(self, stage: int, env_ids: Optional[torch.Tensor] = None) -> None:
        self.curriculum_stage = int(stage)
        if hasattr(self, "world"):
            self.world.set_curriculum_stage(int(stage), env_ids=env_ids)

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
            return obs if full_reset else obs[env_ids], {"state": self.compute_privileged_obs()}

        stage = None
        if isinstance(options, dict) and "stage" in options:
            stage = int(options["stage"])
            self.set_curriculum_stage(stage, env_ids=env_ids)

        # 1. Reset analytic world and physical obstacles / gate objects.
        self.world.reset_world(env_ids, stage=stage)

        # 2. Reset each robot root pose and joints.
        for agent_id, robot in enumerate(self.robots):
            root_state = robot.data.default_root_state[env_ids].clone()

            root_state[:, :3] += self.env_origins[env_ids]
            root_state[:, 0:2] = self.env_origins[env_ids, :2] + self.world.start_pos[env_ids, agent_id]
            root_state[:, 2] = self.env_origins[env_ids, 2] + float(self.cfg.spawn_height)
            root_state[:, 3:7] = self._yaw_to_quat_wxyz(self.world.start_yaw[env_ids, agent_id])
            root_state[:, 7:13] = 0.0

            robot.write_root_state_to_sim(root_state, env_ids=env_ids)

            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            robot.reset(env_ids)

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
        self.prev_gate_passed[env_ids] = False
        self.obs_buffer[env_ids] = 0.0

        # 4. Sync scene.
        self.scene.write_data_to_sim()
        self.scene.update(0.0)

        root_local = self._root_pos_local()
        yaw = self._yaw()
        lin_vel_w = self._root_lin_vel_w()

        team = self.world.compute_team_terms(
            root_pos_local=root_local,
            yaw=yaw,
            lin_vel=lin_vel_w[:, :, :2],
        )
        self.last_center_goal_dist[env_ids] = team["center_goal_dist"][env_ids]

        gate_terms = self.world.gate_progress_terms(root_local)
        self.prev_gate_passed[env_ids] = gate_terms["passed_gate"][env_ids]

        # Warm observation stack.
        obs_single = self._compute_single_actor_obs(update_lidar_history=True)
        for frame_id in range(int(self.cfg.frame_stack)):
            self.obs_buffer[env_ids, :, frame_id, :] = obs_single[env_ids]

        obs = self.compute_obs()
        state = self.compute_privileged_obs()
        info = {"state": state}

        if full_reset:
            return obs, info
        return obs[env_ids], {"state": state[env_ids]}

    @torch.no_grad()
    def step(self, actions_nn: torch.Tensor):
        actions_nn = torch.as_tensor(actions_nn, dtype=torch.float32, device=self.device)

        if actions_nn.dim() == 2 and actions_nn.shape[0] == self.num_envs * self.num_agents:
            actions_nn = actions_nn.view(self.num_envs, self.num_agents, self.num_actions)

        if actions_nn.shape != (self.num_envs, self.num_agents, self.num_actions):
            raise RuntimeError(
                f"expected action shape {(self.num_envs, self.num_agents, self.num_actions)}, "
                f"got {tuple(actions_nn.shape)}"
            )

        actions_nn = torch.nan_to_num(actions_nn, nan=0.0, posinf=1.0, neginf=-1.0)
        actions_nn = torch.clamp(actions_nn, -1.0, 1.0)

        self.prev_actions = self.raw_actions.clone()
        self.raw_actions = actions_nn.clone()

        pre_center_goal_dist = self.last_center_goal_dist.clone()

        applied = self._apply_action_model(actions_nn)
        self.prev_applied_actions = self.applied_actions.clone()
        self.applied_actions = applied.clone()
        self.actions = applied.clone()

        wheel_targets = self._actions_to_wheel_targets(applied)

        env_ids_all = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        for agent_id, robot in enumerate(self.robots):
            robot.set_joint_velocity_target(
                wheel_targets[:, agent_id, :],
                joint_ids=self.wheel_joint_ids[agent_id],
                env_ids=env_ids_all,
            )

        for _ in range(int(self.cfg.decimation)):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(float(self.cfg.sim_dt))

        self.episode_steps += 1
        self.global_steps += self.num_envs

        reward, terminated, truncated, info = self._compute_rewards_and_dones(
            pre_center_goal_dist=pre_center_goal_dist,
        )
        done = terminated | truncated

        obs_single = self._compute_single_actor_obs(update_lidar_history=True)
        self.obs_buffer = torch.roll(self.obs_buffer, shifts=-1, dims=2)
        self.obs_buffer[:, :, -1, :] = obs_single
        obs = self.compute_obs()

        self.episode_return += reward

        state_before_reset = self.compute_privileged_obs()
        info["state"] = state_before_reset

        if done.any():
            reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
            info["terminal_observation"] = obs[reset_ids].clone()
            info["terminal_state"] = state_before_reset[reset_ids].clone()

            self.reset(reset_ids)
            obs = self.compute_obs().clone()
            info["state"] = self.compute_privileged_obs()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Action model
    # ------------------------------------------------------------------
    def _apply_action_model(self, actions_nn: torch.Tensor) -> torch.Tensor:
        """raw action -> delay -> deadband -> bias -> EMA -> clamp."""

        self.action_delay_buffer = torch.roll(self.action_delay_buffer, shifts=1, dims=2)
        self.action_delay_buffer[:, :, 0, :] = actions_nn

        delay = torch.clamp(
            self.world.action_delay_frames,
            min=0,
            max=int(self.cfg.max_action_delay_frames),
        )

        env_idx = torch.arange(self.num_envs, dtype=torch.long, device=self.device).view(-1, 1)
        agent_idx = torch.arange(self.num_agents, dtype=torch.long, device=self.device).view(1, -1)

        delayed = self.action_delay_buffer[env_idx, agent_idx, delay, :]

        deadband = torch.clamp(self.world.action_deadband.unsqueeze(-1), 0.0, 0.95)
        abs_a = torch.abs(delayed)
        sign_a = torch.sign(delayed)

        after_deadband = torch.where(
            abs_a > deadband,
            sign_a * (abs_a - deadband) / torch.clamp(1.0 - deadband, min=1.0e-6),
            torch.zeros_like(delayed),
        )

        # motor_bias is used as normalized command bias here.
        biased = torch.clamp(after_deadband + self.world.motor_bias, -1.0, 1.0)

        alpha = torch.clamp(self.world.action_ema_alpha.unsqueeze(-1), 0.0, 1.0)
        ema = alpha * biased + (1.0 - alpha) * self.applied_actions

        return torch.nan_to_num(
            torch.clamp(ema, -1.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

    def _actions_to_wheel_targets(self, applied: torch.Tensor) -> torch.Tensor:
        """normalized [v, w] -> differential wheel angular velocity targets."""

        v_norm = applied[:, :, 0]
        w_norm = applied[:, :, 1]

        v_norm_scaled = torch.where(
            v_norm >= 0.0,
            v_norm,
            v_norm * float(self.cfg.reverse_speed_fraction),
        )

        v_cmd = v_norm_scaled * self.world.max_speed
        w_cmd = w_norm * self.world.max_yaw_rate

        v_left = v_cmd - 0.5 * w_cmd * float(self.cfg.wheel_base)
        v_right = v_cmd + 0.5 * w_cmd * float(self.cfg.wheel_base)

        wheel = torch.stack(
            [
                v_left / max(float(self.cfg.wheel_radius), 1.0e-6),
                v_right / max(float(self.cfg.wheel_radius), 1.0e-6),
            ],
            dim=-1,
        )

        wheel = wheel * self.world.motor_strength * self.world.wheel_radius_scale
        wheel = wheel * self.wheel_signs

        return torch.nan_to_num(
            torch.clamp(
                wheel,
                min=-float(self.cfg.max_wheel_speed),
                max=float(self.cfg.max_wheel_speed),
            ),
            nan=0.0,
            posinf=float(self.cfg.max_wheel_speed),
            neginf=-float(self.cfg.max_wheel_speed),
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def compute_obs(self) -> torch.Tensor:
        obs = self.obs_buffer.reshape(self.num_envs, self.num_agents, -1)

        if obs.shape[-1] != self.num_observations:
            raise RuntimeError(
                f"[Task4] actor obs dim mismatch: got {obs.shape[-1]}, expected {self.num_observations}"
            )

        return torch.nan_to_num(
            torch.clamp(obs, -float(self.cfg.obs_clip), float(self.cfg.obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.obs_clip),
            neginf=-float(self.cfg.obs_clip),
        )

    def compute_privileged_obs(self) -> torch.Tensor:
        root_local = self._root_pos_local()
        yaw = self._yaw()
        lin_vel_w = self._root_lin_vel_w()

        priv = self.world.compute_privileged_features(
            root_pos_local=root_local,
            yaw=yaw,
            lin_vel=lin_vel_w[:, :, :2],
        )
        priv = torch.nan_to_num(
            torch.clamp(priv, -float(self.cfg.priv_clip), float(self.cfg.priv_clip)),
            nan=0.0,
            posinf=float(self.cfg.priv_clip),
            neginf=-float(self.cfg.priv_clip),
        )

        if priv.shape[-1] != self.num_privileged_obs:
            raise RuntimeError(
                f"[Task4] critic state dim mismatch: got {priv.shape[-1]}, expected {self.num_privileged_obs}"
            )

        return priv

    def get_privileged_observations(self) -> torch.Tensor:
        return self.compute_privileged_obs()

    def _compute_states(self) -> torch.Tensor:
        return self.compute_privileged_obs()

    def _compute_single_actor_obs(self, update_lidar_history: bool = True) -> torch.Tensor:
        root_local = self._root_pos_local()
        yaw = self._yaw()
        lin_vel_w = self._root_lin_vel_w()
        lin_vel_b = self._root_lin_vel_b()
        ang_vel_b = self._root_ang_vel_b()
        wheel_vel = self._wheel_vel()

        team = self.world.compute_team_terms(
            root_pos_local=root_local,
            yaw=yaw,
            lin_vel=lin_vel_w[:, :, :2],
        )

        lidar = self.world.compute_analytic_lidar(
            root_pos_local=root_local,
            yaw=yaw,
            add_noise=True,
            update_history=update_lidar_history,
            normalize=False,
            include_teammates=True,
        )
        lidar_norm = torch.clamp(
            lidar / max(float(self.cfg.world_cfg.lidar_max_distance), 1.0e-6),
            0.0,
            1.0,
        )
        lidar_delta = self.world.last_lidar_delta

        risk = self.world.compute_risk_features(root_local, yaw, lidar_pooled=lidar)

        # Self velocity: 3
        vel_obs = torch.stack(
            [
                torch.clamp(lin_vel_b[:, :, 0] / float(self.cfg.lin_vel_scale), -5.0, 5.0),
                torch.clamp(lin_vel_b[:, :, 1] / float(self.cfg.lin_vel_scale), -5.0, 5.0),
                torch.clamp(ang_vel_b[:, :, 2] / float(self.cfg.ang_vel_scale), -5.0, 5.0),
            ],
            dim=-1,
        )

        # Wheel velocity: 2
        wheel_obs = torch.clamp(
            wheel_vel / max(float(self.cfg.wheel_vel_scale), 1.0e-6),
            -5.0,
            5.0,
        )

        # Center-goal relative to each agent body frame: 5
        goal_vec_from_agent_w = self.world.goal_pos[:, None, :] - root_local
        goal_vec_from_agent_b = self._rotate_world_to_body_2d(goal_vec_from_agent_w, yaw)
        goal_dist_agent = torch.norm(goal_vec_from_agent_w, dim=-1)
        goal_angle_agent = torch.atan2(goal_vec_from_agent_w[:, :, 1], goal_vec_from_agent_w[:, :, 0])
        goal_heading_error = self.world.wrap_to_pi(goal_angle_agent - yaw)

        goal_obs = torch.cat(
            [
                torch.clamp(
                    goal_dist_agent.unsqueeze(-1) / max(float(self.cfg.world_cfg.goal_dist_norm), 1.0e-6),
                    0.0,
                    2.0,
                ),
                torch.clamp(
                    goal_vec_from_agent_b / max(float(self.cfg.world_cfg.goal_xy_norm), 1.0e-6),
                    -2.0,
                    2.0,
                ),
                torch.sin(goal_heading_error).unsqueeze(-1),
                torch.cos(goal_heading_error).unsqueeze(-1),
            ],
            dim=-1,
        )

        # Slot error in body frame: 3
        slot_error_b = self._rotate_world_to_body_2d(team["slot_error_vec"], yaw)
        slot_error_dist = team["slot_error"]

        slot_obs = torch.cat(
            [
                torch.clamp(slot_error_b / max(float(self.cfg.slot_error_norm), 1.0e-6), -3.0, 3.0),
                torch.clamp(
                    slot_error_dist.unsqueeze(-1) / max(float(self.cfg.slot_error_norm), 1.0e-6),
                    0.0,
                    3.0,
                ),
            ],
            dim=-1,
        )

        # Team heading error: 2
        team_heading_error = self.world.wrap_to_pi(team["team_heading"].unsqueeze(-1) - yaw)
        team_heading_obs = torch.cat(
            [
                torch.sin(team_heading_error).unsqueeze(-1),
                torch.cos(team_heading_error).unsqueeze(-1),
            ],
            dim=-1,
        )

        # Formation one-hot + scale: 4
        ftype = self.world.formation_type
        formation_oh = torch.zeros(
            (self.num_envs, int(self.cfg.world_cfg.num_formation_types)),
            dtype=torch.float32,
            device=self.device,
        )
        formation_oh.scatter_(1, ftype.unsqueeze(-1), 1.0)

        formation_obs = torch.cat(
            [
                formation_oh[:, None, :].expand(-1, self.num_agents, -1),
                torch.clamp(self.world.formation_scale, 0.0, 2.0)
                .view(self.num_envs, 1, 1)
                .expand(-1, self.num_agents, -1),
            ],
            dim=-1,
        )

        # Agent id one-hot: 4
        agent_id_obs = self.agent_id_onehot.expand(self.num_envs, -1, -1)

        # Teammate relative position and velocity:
        # position: 3 teammates * [dist_norm, sin, cos] = 9
        # velocity: 3 teammates * [vx_b, vy_b] = 6
        teammate_pos_list = []
        teammate_vel_list = []

        for i in range(self.num_agents):
            pos_i = root_local[:, i, :]
            yaw_i = yaw[:, i]
            vel_i = lin_vel_w[:, i, :2]

            per_i_pos = []
            per_i_vel = []

            for j in range(self.num_agents):
                if i == j:
                    continue

                rel_w = root_local[:, j, :] - pos_i
                dist = torch.norm(rel_w, dim=-1)
                ang = torch.atan2(rel_w[:, 1], rel_w[:, 0])
                rel_ang = self.world.wrap_to_pi(ang - yaw_i)

                per_i_pos.append(
                    torch.clamp(
                        dist / max(float(self.cfg.teammate_dist_norm), 1.0e-6),
                        0.0,
                        3.0,
                    ).unsqueeze(-1)
                )
                per_i_pos.append(torch.sin(rel_ang).unsqueeze(-1))
                per_i_pos.append(torch.cos(rel_ang).unsqueeze(-1))

                rel_vel_w = lin_vel_w[:, j, :2] - vel_i
                rel_vel_b = self._rotate_world_to_body_2d(rel_vel_w, yaw_i)
                per_i_vel.append(
                    torch.clamp(
                        rel_vel_b / max(float(self.cfg.teammate_vel_scale), 1.0e-6),
                        -5.0,
                        5.0,
                    )
                )

            teammate_pos_list.append(torch.cat(per_i_pos, dim=-1))
            teammate_vel_list.append(torch.cat(per_i_vel, dim=-1))

        teammate_pos_obs = torch.stack(teammate_pos_list, dim=1)
        teammate_vel_obs = torch.stack(teammate_vel_list, dim=1)

        action_delta = self.applied_actions - self.prev_applied_actions

        progress_obs = torch.clamp(
            self.progress_ema / max(float(self.cfg.progress_norm_scale), 1.0e-6),
            -5.0,
            5.0,
        ).view(self.num_envs, 1, 1).expand(-1, self.num_agents, -1)

        center_speed_obs = torch.clamp(
            team["center_speed"] / max(float(self.cfg.center_speed_scale), 1.0e-6),
            0.0,
            3.0,
        ).view(self.num_envs, 1, 1).expand(-1, self.num_agents, -1)

        obs = torch.cat(
            [
                vel_obs,                  # 3
                wheel_obs,                # 2
                goal_obs,                 # 5
                slot_obs,                 # 3
                team_heading_obs,         # 2
                formation_obs,            # 4
                agent_id_obs,             # 4
                teammate_pos_obs,         # 9
                teammate_vel_obs,         # 6
                self.applied_actions,     # 2
                action_delta,             # 2
                progress_obs,             # 1
                center_speed_obs,         # 1
                lidar_norm,               # 48
                lidar_delta,              # 48
                risk,                     # 16
            ],
            dim=-1,
        )

        if obs.shape[-1] != int(self.cfg.single_actor_obs_dim):
            raise RuntimeError(
                f"[Task4] single actor obs dim mismatch: got {obs.shape[-1]}, "
                f"expected {self.cfg.single_actor_obs_dim}"
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
    def _compute_rewards_and_dones(self, pre_center_goal_dist: torch.Tensor):
        root_local = self._root_pos_local()
        yaw = self._yaw()
        lin_vel_w = self._root_lin_vel_w()
        ang_vel_b = self._root_ang_vel_b()

        team = self.world.compute_team_terms(
            root_pos_local=root_local,
            yaw=yaw,
            lin_vel=lin_vel_w[:, :, :2],
        )
        events = self.world.check_events(
            root_pos_local=root_local,
            yaw=yaw,
            lin_vel=lin_vel_w[:, :, :2],
            ang_vel=ang_vel_b,
        )
        gate_terms = self.world.gate_progress_terms(root_local)

        current_center_goal_dist = events["center_goal_dist"]
        progress = pre_center_goal_dist - current_center_goal_dist
        progress_clamped = torch.clamp(
            progress,
            -float(self.cfg.progress_clip),
            float(self.cfg.progress_clip),
        )

        self.progress_ema = 0.90 * self.progress_ema + 0.10 * progress
        self.last_center_goal_dist = current_center_goal_dist.detach().clone()

        goal_dir = team["vec_goal"] / torch.clamp(current_center_goal_dist.unsqueeze(-1), min=1.0e-6)
        center_vel_w = lin_vel_w[:, :, :2].mean(dim=1)
        goal_aligned_center_speed = torch.sum(center_vel_w * goal_dir, dim=-1)

        heading_cos_agents = torch.cos(team["heading_error"])
        heading_gate_agents = torch.clamp(heading_cos_agents, 0.0, 1.0)
        heading_gate_team = heading_gate_agents.mean(dim=-1)

        desired_center_speed = torch.where(
            current_center_goal_dist > 2.0,
            torch.full_like(current_center_goal_dist, 0.65),
            torch.clamp(0.35 * current_center_goal_dist, min=0.08, max=0.45),
        )
        speed_error = torch.abs(goal_aligned_center_speed - desired_center_speed)
        forward_gate = torch.clamp(
            goal_aligned_center_speed / torch.clamp(desired_center_speed, min=0.10),
            0.0,
            1.0,
        )

        speed = team["speed"]
        speed_sync_error = torch.std(speed, dim=-1)

        risk = self.world.compute_risk_features(root_local, yaw, lidar_pooled=self.world.last_lidar)

        front_risk = risk[:, :, 1]
        boundary_risk = risk[:, :, 4]
        obstacle_risk = risk[:, :, 5]
        gate_risk = risk[:, :, 6]
        pair_risk = risk[:, :, 7]
        front_clearance = 1.0 - front_risk

        gate_pass_new = gate_terms["passed_gate"] & (~self.prev_gate_passed)
        self.prev_gate_passed = self.prev_gate_passed | gate_terms["passed_gate"]

        r_team_progress = progress_clamped.unsqueeze(-1).expand(-1, self.num_agents)
        r_center_speed = (
            torch.exp(-2.0 * torch.square(speed_error)) * heading_gate_team * forward_gate
        ).unsqueeze(-1).expand(-1, self.num_agents)
        r_team_heading = heading_gate_agents

        p_formation_mean = -team["mean_slot_error"].unsqueeze(-1).expand(-1, self.num_agents)
        p_formation_agent = -team["slot_error"]
        p_team_spread = -team["team_spread"].unsqueeze(-1).expand(-1, self.num_agents)
        p_speed_sync = -speed_sync_error.unsqueeze(-1).expand(-1, self.num_agents)

        r_gate_pass = gate_pass_new.float().unsqueeze(-1).expand(-1, self.num_agents)
        r_front_clearance = front_clearance
        p_obstacle_risk = -obstacle_risk
        p_gate_risk = -gate_risk
        p_boundary_risk = -boundary_risk
        p_pair_risk = -pair_risk

        p_spin = -torch.square(ang_vel_b[:, :, 2])
        p_action_smooth = -torch.mean(torch.square(self.applied_actions - self.prev_applied_actions), dim=-1)
        p_action_mag = -torch.mean(torch.square(self.applied_actions), dim=-1)

        wheel_vel = self._wheel_vel()
        p_wheel_speed = -torch.mean(
            torch.square(wheel_vel / max(float(self.cfg.wheel_vel_scale), 1.0e-6)),
            dim=-1,
        )

        low_progress = torch.abs(progress) < float(self.cfg.stuck_progress_threshold)
        low_speed = torch.abs(goal_aligned_center_speed) < float(self.cfg.stuck_center_speed_threshold)
        far_from_goal = current_center_goal_dist > float(self.cfg.world_cfg.goal_center_success_tol)

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
        p_stuck = -torch.clamp(
            self.stuck_counter.float() / 100.0,
            0.0,
            1.0,
        ).unsqueeze(-1).expand(-1, self.num_agents)

        r_step = -torch.ones((self.num_envs, self.num_agents), dtype=torch.float32, device=self.device)

        continuous_raw = (
            float(self.cfg.w_team_progress) * r_team_progress
            + float(self.cfg.w_center_speed) * r_center_speed
            + float(self.cfg.w_team_heading) * r_team_heading
            + float(self.cfg.w_formation_mean) * p_formation_mean
            + float(self.cfg.w_formation_agent) * p_formation_agent
            + float(self.cfg.w_team_spread) * p_team_spread
            + float(self.cfg.w_speed_sync) * p_speed_sync
            + float(self.cfg.w_gate_pass) * r_gate_pass
            + float(self.cfg.w_front_clearance) * r_front_clearance
            + float(self.cfg.w_obstacle_risk) * p_obstacle_risk
            + float(self.cfg.w_gate_risk) * p_gate_risk
            + float(self.cfg.w_boundary_risk) * p_boundary_risk
            + float(self.cfg.w_pair_risk) * p_pair_risk
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

        success_candidate = events["success_candidate"]
        crash = events["crash"]
        agent_crash = events["agent_crash"]

        self.success_hold_counter = torch.where(
            success_candidate,
            self.success_hold_counter + 1,
            torch.zeros_like(self.success_hold_counter),
        )
        stable_success = self.success_hold_counter >= int(self.cfg.world_cfg.success_hold_steps)

        timeout = self.episode_steps >= int(self.cfg.max_episode_length)

        terminated = stable_success | crash
        truncated = timeout & (~terminated)

        event_reward = torch.zeros((self.num_envs, self.num_agents), dtype=torch.float32, device=self.device)

        event_reward = torch.where(
            stable_success.unsqueeze(-1),
            torch.full_like(event_reward, float(self.cfg.rew_success)),
            event_reward,
        )
        event_reward = torch.where(
            crash.unsqueeze(-1),
            event_reward + float(self.cfg.rew_crash_team),
            event_reward,
        )
        event_reward = torch.where(
            agent_crash,
            event_reward + float(self.cfg.rew_crash_agent),
            event_reward,
        )
        event_reward = torch.where(
            truncated.unsqueeze(-1),
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
        reward = torch.nan_to_num(reward, nan=0.0, posinf=100.0, neginf=-100.0)

        done = terminated | truncated

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
                "R_Team_Progress": (float(self.cfg.w_team_progress) * r_team_progress).mean().item(),
                "R_Center_Speed": (float(self.cfg.w_center_speed) * r_center_speed).mean().item(),
                "R_Team_Heading": (float(self.cfg.w_team_heading) * r_team_heading).mean().item(),
                "P_Formation_Mean": (float(self.cfg.w_formation_mean) * p_formation_mean).mean().item(),
                "P_Formation_Agent": (float(self.cfg.w_formation_agent) * p_formation_agent).mean().item(),
                "P_Team_Spread": (float(self.cfg.w_team_spread) * p_team_spread).mean().item(),
                "P_Speed_Sync": (float(self.cfg.w_speed_sync) * p_speed_sync).mean().item(),
                "R_Gate_Pass": (float(self.cfg.w_gate_pass) * r_gate_pass).mean().item(),
                "R_Front_Clearance": (float(self.cfg.w_front_clearance) * r_front_clearance).mean().item(),
                "P_Obstacle_Risk": (float(self.cfg.w_obstacle_risk) * p_obstacle_risk).mean().item(),
                "P_Gate_Risk": (float(self.cfg.w_gate_risk) * p_gate_risk).mean().item(),
                "P_Boundary_Risk": (float(self.cfg.w_boundary_risk) * p_boundary_risk).mean().item(),
                "P_Pair_Risk": (float(self.cfg.w_pair_risk) * p_pair_risk).mean().item(),
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
                "Agent_Crash_Rate": agent_crash.float().mean().item(),
                "Out_Of_Bounds_Rate": events["out_of_bounds"].float().mean().item(),
                "Obstacle_Collision_Rate": events["obstacle_collision"].float().mean().item(),
                "Gate_Collision_Rate": events["gate_collision"].float().mean().item(),
                "Pair_Collision_Rate": events["pair_collision_any"].float().mean().item(),
                "Timeout_Rate": truncated.float().mean().item(),
                "Done_Rate": done.float().mean().item(),
                "Episode_Success_Rate": episode_success_rate.item(),
                "Episode_Crash_Rate": episode_crash_rate.item(),
                "Episode_Timeout_Rate": episode_timeout_rate.item(),
                "Episode_Done_Count": self.total_done_episodes.item(),
            },
            "telemetry": {
                "Center_Goal_Dist": current_center_goal_dist.mean().item(),
                "Progress": progress.mean().item(),
                "Progress_EMA": self.progress_ema.mean().item(),
                "Goal_Aligned_Center_Speed": goal_aligned_center_speed.mean().item(),
                "Desired_Center_Speed": desired_center_speed.mean().item(),
                "Speed_Error": speed_error.mean().item(),
                "Heading_Cos": heading_cos_agents.mean().item(),
                "Mean_Slot_Error": team["mean_slot_error"].mean().item(),
                "Max_Slot_Error": team["max_slot_error"].mean().item(),
                "Min_Pair_Dist": team["min_pair_dist"].mean().item(),
                "Team_Spread": team["team_spread"].mean().item(),
                "Center_Speed": team["center_speed"].mean().item(),
                "Speed_Sync_Error": speed_sync_error.mean().item(),
                "Gate_Active": self.world.gate_active.float().mean().item(),
                "Near_Gate": gate_terms["near_gate"].float().mean().item(),
                "Passed_Gate": gate_terms["passed_gate"].float().mean().item(),
                "Gate_Pass_New": gate_pass_new.float().mean().item(),
                "Lidar_Min": self.world.last_lidar.min().item(),
                "Lidar_Mean": self.world.last_lidar.mean().item(),
                "Risk_All": risk[:, :, 0].mean().item(),
                "Risk_Front": front_risk.mean().item(),
                "Risk_Obstacle": obstacle_risk.mean().item(),
                "Risk_Gate": gate_risk.mean().item(),
                "Risk_Boundary": boundary_risk.mean().item(),
                "Risk_Pair": pair_risk.mean().item(),
                "Front_Clearance": front_clearance.mean().item(),
                "Action_V": self.applied_actions[:, :, 0].mean().item(),
                "Action_W": self.applied_actions[:, :, 1].mean().item(),
                "Raw_Action_V": self.raw_actions[:, :, 0].mean().item(),
                "Raw_Action_W": self.raw_actions[:, :, 1].mean().item(),
                "Wheel_Vel_Left": wheel_vel[:, :, 0].mean().item(),
                "Wheel_Vel_Right": wheel_vel[:, :, 1].mean().item(),
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
                "Num_Agents": float(self.num_agents),
                "Action_Dim_Per_Agent": float(self.num_actions),
                "Reward_Min": reward.min().item(),
                "Reward_Max": reward.max().item(),
                "Continuous_Min": continuous.min().item(),
                "Continuous_Max": continuous.max().item(),
                "Event_Min": event_reward.min().item(),
                "Event_Max": event_reward.max().item(),
                "Base_Height_Mean": self._root_pos_w()[:, :, 2].mean().item(),
            },
            "is_success": stable_success.detach().clone(),
        }

        return reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Geometry / state helpers
    # ------------------------------------------------------------------
    def _root_pos_w(self) -> torch.Tensor:
        return torch.stack([robot.data.root_pos_w for robot in self.robots], dim=1)

    def _root_pos_local(self) -> torch.Tensor:
        root_pos_w = self._root_pos_w()
        return root_pos_w[:, :, :2] - self.env_origins[:, None, :2]

    def _yaw(self) -> torch.Tensor:
        quats = torch.stack([robot.data.root_quat_w for robot in self.robots], dim=1)
        return self._quat_yaw(quats)

    @staticmethod
    def _quat_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
        w = quat_wxyz[..., 0]
        x = quat_wxyz[..., 1]
        y = quat_wxyz[..., 2]
        z = quat_wxyz[..., 3]

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
        vals = []
        for robot in self.robots:
            if hasattr(robot.data, "root_lin_vel_w"):
                vals.append(robot.data.root_lin_vel_w)
            else:
                vals.append(self._body_to_world_vec(robot.data.root_lin_vel_b, self._quat_yaw(robot.data.root_quat_w)))
        return torch.stack(vals, dim=1)

    def _root_lin_vel_b(self) -> torch.Tensor:
        return torch.stack([robot.data.root_lin_vel_b for robot in self.robots], dim=1)

    def _root_ang_vel_b(self) -> torch.Tensor:
        return torch.stack([robot.data.root_ang_vel_b for robot in self.robots], dim=1)

    def _wheel_vel(self) -> torch.Tensor:
        vals = []
        for agent_id, robot in enumerate(self.robots):
            vals.append(robot.data.joint_vel[:, self.wheel_joint_ids_t[agent_id]])
        return torch.stack(vals, dim=1)

    @staticmethod
    def _body_to_world_vec(vec_b: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        vx_w = c * vec_b[:, 0] - s * vec_b[:, 1]
        vy_w = s * vec_b[:, 0] + c * vec_b[:, 1]
        return torch.stack([vx_w, vy_w, vec_b[:, 2]], dim=-1)

    @staticmethod
    def _rotate_world_to_body_2d(vec_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Rotate 2D world-frame vectors into body frame.

        Supports:
            vec_w [N, A, 2], yaw [N, A]
            vec_w [N, 2],    yaw [N]
        """

        c = torch.cos(yaw)
        s = torch.sin(yaw)

        x_w = vec_w[..., 0]
        y_w = vec_w[..., 1]

        x_b = c * x_w + s * y_w
        y_b = -s * x_w + c * y_w

        return torch.stack([x_b, y_b], dim=-1)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def _print_debug_info(self) -> None:
        print("\n" + "=" * 120)
        print("✅ [Task4] Diff-Drive UGV / Jetbot Multi-UGV Formation Escort Env Initialized")
        print(f"  num_envs                 : {self.num_envs}")
        print(f"  device                   : {self.device}")
        print(f"  num_agents               : {self.num_agents}")
        print(f"  curriculum_stage         : {self.curriculum_stage}")
        print(f"  num_actions_per_agent    : {self.num_actions}")
        print(f"  single_actor_obs_dim     : {self.cfg.single_actor_obs_dim}")
        print(f"  frame_stack              : {self.cfg.frame_stack}")
        print(f"  actor_obs_dim            : {self.num_observations}")
        print(f"  privileged_feature_dim   : {self.world_priv_dim}")
        print(f"  critic_obs_dim           : {self.num_privileged_obs}")
        print(f"  sim_dt                   : {self.cfg.sim_dt}")
        print(f"  policy_dt                : {self.dt}")
        print(f"  decimation               : {self.cfg.decimation}")
        print(f"  max_episode_length_s     : {self.cfg.max_episode_length_s}")
        print(f"  max_episode_length       : {self.cfg.max_episode_length}")
        print(f"  lidar_pool_bins          : {self.cfg.world_cfg.lidar_pool_bins}")
        print(f"  risk_feature_dim         : {self.world.risk_feature_dim()}")

        for agent_id in range(self.num_agents):
            print(f"  Robot_{agent_id} wheel_joint_ids : {self.wheel_joint_ids[agent_id]}")

        if self.robot_joint_names:
            print("  robot_0.joint_names:")
            for i, name in enumerate(self.robot_joint_names):
                mark = " <wheel>" if i in self.wheel_joint_ids[0] else ""
                print(f"    {i:02d}: {name}{mark}")

        print("=" * 120 + "\n")


JetbotTask4Env = DiffDriveTask4Env
Task4Env = DiffDriveTask4Env
Task4MappoEnv = DiffDriveTask4Env
DiffDriveMultiUGVFormationEnv = DiffDriveTask4Env

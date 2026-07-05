from __future__ import annotations

import math
import warnings
from collections import deque
from typing import Any, Deque, Dict, Optional

import gymnasium as gym
import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*getTypes called on non-existent path.*")

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene

from diff_drive_rl.tasks.task1.task1_config import Task1Config
from diff_drive_rl.tasks.task1.task1_scene import make_diff_drive_task1_scene_cfg
from diff_drive_rl.core.physics.action_protocol import ForwardTurnProtocol


class DiffDriveTask1Env(gym.Env):
    """Jetbot / two-wheel differential-drive Task1: multi-waypoint navigation.

    Action:
        action[:, 0] = forward throttle in [-1, 1]
        action[:, 1] = turn command in [-1, 1]

        The environment maps the forward throttle to a non-negative chassis
        command and adds the turn command differentially to the wheels. This
        prevents reverse-driving while still allowing pivot turns when a waypoint
        appears behind or to the side of the robot.

    Observation:
        CoreNav-v1, frame_stack = 3, single frame dim = 14. The actor sees
        only the currently active waypoint as the local goal, not the whole
        waypoint list.

    Waypoint sampling:
        Task1 deliberately uses a simple forward-cone waypoint generator. On
        reset and after each non-terminal waypoint, the next target is sampled
        within +/- cfg.waypoint_front_angle_deg around the current robot yaw.
        This avoids artificial rear-waypoint switches and keeps Task1 a stable
        local navigation teacher for Task2 / Task3.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task1Config):
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

        SceneCfg = make_diff_drive_task1_scene_cfg(cfg)
        self.scene = InteractiveScene(
            SceneCfg(
                num_envs=int(cfg.num_envs),
                env_spacing=float(cfg.env_spacing),
            )
        )

        self.sim.reset()

        try:
            self.robot: Articulation = self.scene["robot"]
        except Exception:
            self.robot = self.scene.articulations["robot"]

        self.env_origins = self.scene.env_origins.to(self.device)

        self._resolve_robot_indices()
        self.forward_turn_protocol = ForwardTurnProtocol(
            min_forward_action=float(self.cfg.min_forward_action),
            max_forward_action=float(self.cfg.max_forward_action),
            forward_curve_power=float(getattr(self.cfg, "forward_curve_power", 1.0)),
            turn_scale_norm=float(self.cfg.turn_scale_norm),
        )

        self.num_actions = int(self.cfg.num_actions)
        self.num_observations = int(self.cfg.num_observations)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_observations,),
            dtype=np.float32,
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_observations,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        self.step_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.waypoints_local = torch.zeros(
            (self.num_envs, int(self.cfg.num_waypoints), 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.current_wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.path_length_used = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # raw_actions are policy outputs in [-1, 1]. actions are smoothed
        # [forward_throttle, turn] commands in [-1, 1]. The wheel command tensor
        # stores the converted differential-drive targets for diagnostics.
        self.raw_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.prev_raw_actions = torch.zeros_like(self.raw_actions)
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.speed_factor = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.forward_command_norm = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.turn_command_norm = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.wheel_command_norm = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.wheel_vel_targets_last = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        self.last_distances = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_heading_error_abs = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.progress_ema = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.stuck_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.obs_buffer = torch.zeros(
            (self.num_envs, int(self.cfg.frame_stack), int(self.cfg.single_obs_dim)),
            dtype=torch.float32,
            device=self.device,
        )

        self.total_done_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_finished_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_hard_stuck_episodes = torch.zeros((), dtype=torch.float32, device=self.device)

        window = int(self.cfg.recent_window_size)
        self.recent_finished: Deque[float] = deque(maxlen=window)
        self.recent_timeout: Deque[float] = deque(maxlen=window)
        self.recent_hard_stuck: Deque[float] = deque(maxlen=window)
        self.recent_terminal_return: Deque[float] = deque(maxlen=window)
        self.recent_terminal_length: Deque[float] = deque(maxlen=window)
        self.recent_terminal_waypoints: Deque[float] = deque(maxlen=window)
        self.recent_backward_ratio: Deque[float] = deque(maxlen=window)
        self.recent_slow_ratio: Deque[float] = deque(maxlen=window)
        self.recent_stuck_ratio: Deque[float] = deque(maxlen=window)
        self.recent_no_progress_ratio: Deque[float] = deque(maxlen=window)

        self.scene.update(0.0)
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
            ids, names = self.robot.find_joints(".*wheel_joint")
            wheel_ids = [int(i) for i in ids]
        except Exception:
            wheel_ids = []

        if len(wheel_ids) < 2:
            print("[WARN][Jetbot] 未找到 .*wheel_joint，fallback to first 2 joints.", flush=True)
            wheel_ids = [0, 1]

        wheel_ids = wheel_ids[:2]
        self.wheel_joint_ids = wheel_ids
        self.wheel_joint_ids_t = torch.tensor(wheel_ids, dtype=torch.long, device=self.device)

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
            return_full = True
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            return_full = False

        if env_ids.numel() == 0:
            obs = self.compute_obs()
            return obs if return_full else obs[env_ids], {}

        n = int(env_ids.numel())

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.env_origins[env_ids]
        root_state[:, 2] = self.env_origins[env_ids, 2] + float(self.cfg.spawn_height)
        self._generate_waypoints(env_ids)
        first_waypoint = self.waypoints_local[env_ids, 0, :]
        if bool(getattr(self.cfg, "reset_align_to_first_waypoint", False)):
            reset_yaw = torch.atan2(first_waypoint[:, 1], first_waypoint[:, 0])
        else:
            reset_yaw = torch.zeros(n, dtype=torch.float32, device=self.device)
        root_state[:, 3:7] = self._yaw_to_quat_wxyz(reset_yaw)
        root_state[:, 7:13] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.reset(env_ids)

        self.step_counts[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        self.raw_actions[env_ids] = 0.0
        self.prev_raw_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.speed_factor[env_ids] = 0.0
        # Keep diagnostic command buffers inside the Task1 forward-only action
        # invariant after auto-reset.  These buffers are not used as actor
        # observations, but long-running environment tests inspect them right
        # after ``step``.  Resetting the chassis forward command to zero violates
        # the configured lower bound even though no invalid wheel target was
        # applied.
        default_forward_command = float(self.cfg.min_forward_action)
        self.forward_command_norm[env_ids] = default_forward_command
        self.turn_command_norm[env_ids] = 0.0
        self.wheel_command_norm[env_ids, 0] = default_forward_command
        self.wheel_command_norm[env_ids, 1] = default_forward_command
        self.wheel_vel_targets_last[env_ids] = 0.0
        self.progress_ema[env_ids] = 0.0
        self.stuck_counter[env_ids] = 0
        self.obs_buffer[env_ids] = 0.0

        self.scene.update(0.0)

        dist = self._distance_to_current_waypoint()
        self.last_distances[env_ids] = dist[env_ids]
        heading_error_abs = self._heading_error_abs_to_current_waypoint()
        self.last_heading_error_abs[env_ids] = heading_error_abs[env_ids]

        obs_single = self._compute_single_obs()
        for i in range(int(self.cfg.frame_stack)):
            self.obs_buffer[env_ids, i, :] = obs_single[env_ids]

        obs = self.compute_obs()
        return obs if return_full else obs[env_ids], {}

    @torch.no_grad()
    def step(self, actions_nn: torch.Tensor):
        actions_nn = torch.as_tensor(actions_nn, dtype=torch.float32, device=self.device)
        actions_nn = torch.nan_to_num(actions_nn, nan=0.0, posinf=1.0, neginf=-1.0)
        raw_actions = torch.clamp(actions_nn, -1.0, 1.0)

        self.prev_raw_actions = self.raw_actions.clone()
        self.raw_actions = raw_actions.clone()

        self.prev_actions = self.actions.clone()
        self.actions = (
            float(self.cfg.action_tau) * raw_actions
            + (1.0 - float(self.cfg.action_tau)) * self.prev_actions
        )
        self.actions = torch.clamp(self.actions, -1.0, 1.0)

        pre_distances = self.last_distances.clone()

        wheel_vel_targets = self._actions_to_wheel_velocity_targets(self.actions)
        self.wheel_vel_targets_last = wheel_vel_targets.clone()

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

        self.step_counts += 1

        current_dist = self._distance_to_current_waypoint()
        progress = pre_distances - current_dist
        self.progress_ema = 0.90 * self.progress_ema + 0.10 * progress

        reward, terminated, truncated, info = self._compute_rewards_and_dones(
            pre_distances=pre_distances,
            current_dist=current_dist,
            progress=progress,
        )

        done = terminated | truncated

        self.last_distances = self._distance_to_current_waypoint().clone()

        obs_single = self._compute_single_obs()
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

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------
    def _actions_to_wheel_velocity_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert [forward_throttle, turn] into wheel velocity targets.

        The chassis forward command is non-negative. Individual wheel targets
        are allowed to become negative during tight pivot turns, which is
        necessary for a differential-drive robot to face waypoints that are
        initially behind or after a waypoint switch. This prevents reverse
        driving without removing turning authority.
        """

        wheel_targets, command = self.forward_turn_protocol.map_to_wheel_velocity_targets(
            actions,
            wheel_speed_scale=float(self.cfg.wheel_speed_scale),
            wheel_signs=self.wheel_signs,
        )

        self.speed_factor = command.speed_factor.detach().clone()
        self.forward_command_norm = command.forward_norm.detach().clone()
        self.turn_command_norm = command.turn_norm.detach().clone()
        self.wheel_command_norm = command.wheel_norm.detach().clone()

        return wheel_targets

    def _map_raw_actions_to_exec(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """Pure action-mapping helper for white-box tests.

        This function intentionally does not mutate environment buffers such as
        ``forward_command_norm`` or ``wheel_command_norm``.  Earlier test-only
        code reused ``_actions_to_wheel_velocity_targets`` with a small custom
        action batch, which resized these diagnostic tensors from ``num_envs``
        to the test batch size.  The next env reset then indexed those short
        tensors with full environment ids on CUDA and triggered an asynchronous
        device-side index assertion.

        Return value is the normalized wheel command in [-1, 1].
        """

        raw_actions = torch.as_tensor(raw_actions, dtype=torch.float32, device=self.device)
        command = self.forward_turn_protocol.map_to_normalized_wheels(raw_actions)
        return command.wheel_norm

    # ------------------------------------------------------------------
    # Waypoints
    # ------------------------------------------------------------------
    def _generate_waypoints(self, env_ids: torch.Tensor) -> None:
        """Generate the first active waypoint for reset.

        Task1 intentionally uses a simple forward-cone generator. Future
        waypoints are not preplanned as independent global points. Instead, each
        new waypoint is generated when the previous one is reached, using the
        current robot yaw. This keeps the task consistent with forward-only
        local navigation and prevents sudden rear-target switches.
        """

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        n = int(env_ids.numel())
        if n == 0:
            return

        self.waypoints_local[env_ids] = 0.0
        self.current_wp_idx[env_ids] = 0
        self.path_length_used[env_ids] = 0.0

        origin_xy = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        yaw0 = torch.zeros(n, dtype=torch.float32, device=self.device)
        slot0 = torch.zeros(n, dtype=torch.long, device=self.device)
        self._generate_next_forward_waypoint(env_ids, origin_xy, yaw0, slot0)

    def _generate_next_forward_waypoint(
        self,
        env_ids: torch.Tensor,
        root_xy: torch.Tensor,
        yaw: torch.Tensor,
        slot_idx: torch.Tensor | int,
    ) -> None:
        """Sample one waypoint in the robot's forward heading cone.

        Args:
            env_ids: environment ids to update.
            root_xy: current robot local xy for each env id.
            yaw: current robot yaw for each env id.
            slot_idx: waypoint slot to write for each env id.

        The sampled segment length respects cfg.waypoint_min_radius /
        cfg.waypoint_max_radius and the remaining path-length budget whenever
        possible. If the arena radius would be exceeded, the target is projected
        back into cfg.waypoint_world_radius.
        """

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        n = int(env_ids.numel())
        if n == 0:
            return

        root_xy = torch.as_tensor(root_xy, dtype=torch.float32, device=self.device).reshape(n, 2)
        yaw = torch.as_tensor(yaw, dtype=torch.float32, device=self.device).flatten()

        if isinstance(slot_idx, int):
            slot = torch.full((n,), int(slot_idx), dtype=torch.long, device=self.device)
        else:
            slot = torch.as_tensor(slot_idx, dtype=torch.long, device=self.device).flatten()

        slot = torch.clamp(slot, min=0, max=int(self.cfg.num_waypoints) - 1)

        min_d = float(self.cfg.waypoint_min_radius)
        max_d_cfg = float(self.cfg.waypoint_max_radius)
        total_budget = max(
            float(getattr(self.cfg, "waypoint_total_path_length", self.cfg.num_waypoints * max_d_cfg)),
            float(self.cfg.num_waypoints) * min_d,
        )

        remaining_after = torch.clamp((int(self.cfg.num_waypoints) - 1 - slot).float(), min=0.0)
        max_allowed = total_budget - self.path_length_used[env_ids] - remaining_after * min_d
        max_d = torch.clamp(max_allowed, min=min_d, max=max_d_cfg)
        seg_len = min_d + torch.rand(n, dtype=torch.float32, device=self.device) * (max_d - min_d)

        cone = math.radians(float(getattr(self.cfg, "waypoint_front_angle_deg", 60.0)))
        min_cone = math.radians(float(getattr(self.cfg, "waypoint_min_front_angle_deg", 0.0)))
        min_cone = min(max(min_cone, 0.0), max(cone, 0.0))
        attempts = int(getattr(self.cfg, "waypoint_path_resample_attempts", 8))
        world_radius = float(self.cfg.waypoint_world_radius)

        # Force visible steering samples. The previous reset-aligned version
        # made Stage0/1 almost straight, so PPO learned high-speed straight
        # driving and failed at the first two-waypoint curriculum.
        sign = torch.where(
            torch.rand(n, dtype=torch.float32, device=self.device) < 0.5,
            -torch.ones(n, dtype=torch.float32, device=self.device),
            torch.ones(n, dtype=torch.float32, device=self.device),
        )
        angle_mag = min_cone + torch.rand(n, dtype=torch.float32, device=self.device) * max(cone - min_cone, 0.0)
        rel_angle = sign * angle_mag
        target_angle = yaw + rel_angle
        target = root_xy + torch.stack([seg_len * torch.cos(target_angle), seg_len * torch.sin(target_angle)], dim=-1)

        valid = torch.norm(target, dim=-1) <= world_radius
        for _ in range(attempts):
            if valid.all():
                break
            bad = (~valid).nonzero(as_tuple=False).squeeze(-1)
            m = int(bad.numel())
            if m == 0:
                break
            sign_bad = torch.where(
                torch.rand(m, dtype=torch.float32, device=self.device) < 0.5,
                -torch.ones(m, dtype=torch.float32, device=self.device),
                torch.ones(m, dtype=torch.float32, device=self.device),
            )
            mag_bad = min_cone + torch.rand(m, dtype=torch.float32, device=self.device) * max(cone - min_cone, 0.0)
            rel_bad = sign_bad * mag_bad
            angle_bad = yaw[bad] + rel_bad
            target_bad = root_xy[bad] + torch.stack(
                [seg_len[bad] * torch.cos(angle_bad), seg_len[bad] * torch.sin(angle_bad)],
                dim=-1,
            )
            target[bad] = target_bad
            valid[bad] = torch.norm(target_bad, dim=-1) <= world_radius

        norm = torch.norm(target, dim=-1, keepdim=True)
        target = torch.where(
            norm > world_radius,
            target / torch.clamp(norm, min=1e-6) * (0.98 * world_radius),
            target,
        )

        self.waypoints_local[env_ids, slot, :] = target
        actual_len = torch.norm(target - root_xy, dim=-1)
        self.path_length_used[env_ids] = torch.clamp(
            self.path_length_used[env_ids] + actual_len,
            max=total_budget,
        )

    def _sample_waypoint_candidate(self, n: int) -> torch.Tensor:
        # Legacy helper retained for compatibility with older tests / scripts.
        # The active Task1 reset / switch logic uses _generate_next_forward_waypoint.
        angle = torch.rand(int(n), dtype=torch.float32, device=self.device) * 2.0 * math.pi
        radius = float(self.cfg.waypoint_min_radius) + torch.rand(
            int(n), dtype=torch.float32, device=self.device
        ) * (float(self.cfg.waypoint_max_radius) - float(self.cfg.waypoint_min_radius))

        target = torch.stack(
            [radius * torch.cos(angle), radius * torch.sin(angle)],
            dim=-1,
        )

        norm = torch.norm(target, dim=-1, keepdim=True)
        max_r = float(self.cfg.waypoint_world_radius)
        target = torch.where(
            norm > max_r,
            target / torch.clamp(norm, min=1e-6) * max_r,
            target,
        )

        return target

    def _current_waypoint_local(self) -> torch.Tensor:
        idx = torch.clamp(self.current_wp_idx, max=int(self.cfg.num_waypoints) - 1)
        gather_idx = idx.view(-1, 1, 1).expand(-1, 1, 2)
        return torch.gather(self.waypoints_local, 1, gather_idx).squeeze(1)

    def current_waypoint_world(self) -> torch.Tensor:
        target_local = self._current_waypoint_local()
        target_world = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        target_world[:, :2] = self.env_origins[:, :2] + target_local
        target_world[:, 2] = self.env_origins[:, 2] + 0.10
        return target_world

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def compute_obs(self) -> torch.Tensor:
        obs = self.obs_buffer.reshape(self.num_envs, -1)
        return torch.nan_to_num(
            torch.clamp(obs, -float(self.cfg.obs_clip), float(self.cfg.obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.obs_clip),
            neginf=-float(self.cfg.obs_clip),
        )

    def _compute_single_obs(self) -> torch.Tensor:
        """Compute one CoreNav-v1 observation frame.

        CoreNav-v1 is the shared single-vehicle navigation protocol used by
        Task1 and intended as the fixed prefix for Task2 / Task3. The actor sees
        only the active waypoint as the current local goal; multi-waypoint
        progress remains in telemetry instead of the actor input.
        """

        root_pos_local = self._root_pos_local()
        root_quat = self.robot.data.root_quat_w
        yaw = self._quat_yaw(root_quat)

        target_local = self._current_waypoint_local()
        to_target_w = target_local - root_pos_local[:, :2]

        dist = torch.norm(to_target_w, dim=-1)
        target_angle = torch.atan2(to_target_w[:, 1], to_target_w[:, 0])
        heading_error = torch.atan2(torch.sin(target_angle - yaw), torch.cos(target_angle - yaw))

        c = torch.cos(-yaw)
        s = torch.sin(-yaw)
        goal_x_body = c * to_target_w[:, 0] - s * to_target_w[:, 1]
        goal_y_body = s * to_target_w[:, 0] + c * to_target_w[:, 1]

        base_lin_vel_b = self.robot.data.root_lin_vel_b
        base_ang_vel_b = self.robot.data.root_ang_vel_b

        dist_norm = torch.clamp(dist / float(self.cfg.dist_norm_scale), 0.0, 5.0)
        goal_x_body_norm = torch.clamp(goal_x_body / float(self.cfg.goal_xy_norm), -5.0, 5.0)
        goal_y_body_norm = torch.clamp(goal_y_body / float(self.cfg.goal_xy_norm), -5.0, 5.0)
        body_vx = torch.clamp(base_lin_vel_b[:, 0] / float(self.cfg.lin_vel_scale), -5.0, 5.0)
        body_vy = torch.clamp(base_lin_vel_b[:, 1] / float(self.cfg.lin_vel_scale), -5.0, 5.0)
        body_wz = torch.clamp(base_ang_vel_b[:, 2] / float(self.cfg.ang_vel_scale), -5.0, 5.0)

        target_speed_norm = torch.full_like(
            dist_norm,
            float(self.cfg.target_goal_speed) / max(float(self.cfg.target_speed_norm), 1e-6),
        )
        target_speed_norm = torch.clamp(target_speed_norm, 0.0, 5.0)

        action_delta = self.actions - self.prev_actions
        progress_norm = torch.clamp(
            self.progress_ema / max(float(self.cfg.progress_norm_scale), 1e-6),
            -5.0,
            5.0,
        )

        obs = torch.stack(
            [
                dist_norm,
                goal_x_body_norm,
                goal_y_body_norm,
                torch.sin(heading_error),
                torch.cos(heading_error),
                body_vx,
                body_vy,
                body_wz,
                target_speed_norm,
                self.actions[:, 0],
                self.actions[:, 1],
                action_delta[:, 0],
                action_delta[:, 1],
                progress_norm,
            ],
            dim=-1,
        )

        if obs.shape[-1] != int(self.cfg.single_obs_dim):
            raise RuntimeError(
                f"Task1 CoreNav-v1 obs dim mismatch: got {obs.shape[-1]}, expected {self.cfg.single_obs_dim}"
            )

        return torch.nan_to_num(
            torch.clamp(obs, -float(self.cfg.obs_clip), float(self.cfg.obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.obs_clip),
            neginf=-float(self.cfg.obs_clip),
        )

    def _compute_states(self) -> torch.Tensor:
        return self.compute_obs()

    def get_privileged_observations(self) -> torch.Tensor:
        return self.compute_obs()

    # ------------------------------------------------------------------
    # Reward / Done
    # ------------------------------------------------------------------
    def _compute_rewards_and_dones(
        self,
        pre_distances: torch.Tensor,
        current_dist: torch.Tensor,
        progress: torch.Tensor,
    ):
        root_pos_local = self._root_pos_local()
        root_quat = self.robot.data.root_quat_w
        yaw = self._quat_yaw(root_quat)

        target_local = self._current_waypoint_local()
        to_target = target_local - root_pos_local[:, :2]
        dist_to_goal = torch.norm(to_target, dim=-1)
        dir_to_goal = to_target / torch.clamp(dist_to_goal.unsqueeze(-1), min=1e-6)

        target_angle = torch.atan2(to_target[:, 1], to_target[:, 0])
        heading_error = torch.atan2(torch.sin(target_angle - yaw), torch.cos(target_angle - yaw))

        heading_cos = torch.cos(heading_error)
        heading_gate = torch.clamp(
            (heading_cos - float(self.cfg.heading_gate_min))
            / max(float(self.cfg.heading_gate_full) - float(self.cfg.heading_gate_min), 1e-6),
            0.0,
            1.0,
        )

        distance_gate = torch.clamp(
            (dist_to_goal - float(self.cfg.distance_gate_near))
            / max(float(self.cfg.distance_gate_far) - float(self.cfg.distance_gate_near), 1e-6),
            0.0,
            1.0,
        )

        base_lin_vel_w = self._root_lin_vel_w()
        base_lin_vel_b = self.robot.data.root_lin_vel_b
        base_ang_vel_b = self.robot.data.root_ang_vel_b

        goal_aligned_speed = torch.sum(base_lin_vel_w[:, :2] * dir_to_goal, dim=-1)
        goal_speed_pos = torch.clamp(goal_aligned_speed, min=0.0)

        speed_gate = torch.clamp(
            (goal_speed_pos - float(self.cfg.min_goal_speed))
            / max(float(self.cfg.target_goal_speed) - float(self.cfg.min_goal_speed), 1e-6),
            0.0,
            1.0,
        )

        progress_clamped = torch.clamp(
            progress,
            -float(self.cfg.progress_clip),
            float(self.cfg.progress_clip),
        )
        progress_pos = torch.clamp(progress_clamped, min=0.0)
        progress_neg = torch.clamp(-progress_clamped, min=0.0)

        heading_error_abs = torch.abs(heading_error)
        heading_improve = torch.clamp(
            self.last_heading_error_abs - heading_error_abs,
            min=0.0,
            max=0.25,
        )

        r_progress = progress_pos * heading_gate * speed_gate * distance_gate
        r_heading = heading_gate * speed_gate * distance_gate
        r_heading_improve = heading_improve * distance_gate
        r_forward = torch.clamp(goal_speed_pos / float(self.cfg.target_goal_speed), 0.0, 1.0) * heading_gate * distance_gate

        turn_direction = torch.sign(heading_error)
        turn_rate_to_goal = torch.clamp(
            base_ang_vel_b[:, 2] * turn_direction,
            min=0.0,
            max=1.2,
        ) / 1.2
        commanded_yaw = float(getattr(self.cfg, "turn_command_to_yaw_sign", -1.0)) * self.turn_command_norm
        commanded_turn_to_goal = torch.clamp(
            commanded_yaw * turn_direction / max(float(getattr(self.cfg, "target_turn_command", 0.45)), 1e-6),
            min=0.0,
            max=1.0,
        )
        heading_improve_gate = torch.clamp(
            heading_improve / 0.04,
            min=0.0,
            max=1.0,
        )
        misalignment_gate = torch.clamp(
            (heading_error_abs - float(getattr(self.cfg, "turn_recovery_heading_threshold", 0.35)))
            / max(float(getattr(self.cfg, "turn_recovery_full_heading", 1.20)) - float(getattr(self.cfg, "turn_recovery_heading_threshold", 0.35)), 1e-6),
            min=0.0,
            max=1.0,
        )
        # Commanded turn gives an immediate learning signal; actual yaw-rate and
        # heading-improvement terms prevent rewarding turn commands that do not
        # physically recover heading.
        r_turn_to_goal = (
            0.65 * commanded_turn_to_goal
            + 0.25 * turn_rate_to_goal
            + 0.10 * heading_improve_gate
        ) * misalignment_gate * distance_gate

        p_negative_progress = -progress_neg
        p_backward = -torch.clamp(-goal_aligned_speed / float(self.cfg.target_goal_speed), 0.0, 1.0) * distance_gate

        far_from_goal = current_dist > float(self.cfg.reach_threshold)
        turn_recovery_gate = torch.clamp(
            (heading_error_abs - float(getattr(self.cfg, "turn_recovery_heading_threshold", 0.35)))
            / max(float(getattr(self.cfg, "turn_recovery_full_heading", 1.20)) - float(getattr(self.cfg, "turn_recovery_heading_threshold", 0.35)), 1e-6),
            min=0.0,
            max=1.0,
        )
        p_misaligned_forward = -self.forward_command_norm * turn_recovery_gate * distance_gate * far_from_goal.float()

        heading_ok = heading_gate > 0.50

        # Final tuning note:
        # Do not punish every small positive progress as no-progress. In the
        # previous version, progress_threshold=0.002 made most valid slow
        # approaches look like failure, which suppressed exploration. Here,
        # no-progress is reserved for non-positive progress, while slow is only
        # applied when both speed and progress are insufficient.
        no_progress_now = far_from_goal & (progress <= 0.0)
        slow_now = (
            far_from_goal
            & heading_ok
            & (goal_aligned_speed < float(self.cfg.min_goal_speed))
            & (progress <= float(self.cfg.stuck_progress_threshold))
        )

        planar_speed = torch.norm(base_lin_vel_b[:, :2], dim=-1)
        spin_in_place_now = far_from_goal & (planar_speed < 0.08) & (torch.abs(base_ang_vel_b[:, 2]) > 0.35)

        p_slow = -slow_now.float()
        p_no_progress = -no_progress_now.float()
        p_spin_in_place = -torch.clamp(torch.abs(base_ang_vel_b[:, 2]) / 2.0, 0.0, 1.0) * spin_in_place_now.float()

        abs_wz = torch.abs(base_ang_vel_b[:, 2])
        p_spin = -torch.square(base_ang_vel_b[:, 2])
        bad_turn_now = (
            far_from_goal
            & (abs_wz > float(getattr(self.cfg, "bad_turn_wz_threshold", 1.25)))
            & (heading_improve <= float(getattr(self.cfg, "bad_turn_heading_improve_threshold", 1.0e-4)))
            & ((progress <= float(getattr(self.cfg, "bad_turn_progress_threshold", 0.0006))) | (heading_cos < 0.20))
        )
        p_bad_turn = -torch.clamp(
            (abs_wz - float(getattr(self.cfg, "bad_turn_wz_threshold", 1.25))) / 2.0,
            min=0.0,
            max=1.0,
        ) * bad_turn_now.float()
        p_lateral_vel = -torch.square(base_lin_vel_b[:, 1])

        action_diff = self.actions - self.prev_actions
        p_action_smooth = -torch.mean(torch.square(action_diff), dim=-1)
        p_action_mag = -torch.mean(torch.square(self.actions), dim=-1)

        wheel_vel = self.robot.data.joint_vel[:, self.wheel_joint_ids_t]
        p_wheel_speed = -torch.mean(torch.square(wheel_vel / float(self.cfg.wheel_vel_scale)), dim=-1)

        low_progress = progress <= float(self.cfg.stuck_progress_threshold)
        low_goal_speed = goal_aligned_speed < float(self.cfg.stuck_speed_threshold)
        low_planar_speed = planar_speed < 0.06
        stuck_now = (
            low_progress
            & low_goal_speed
            & low_planar_speed
            & far_from_goal
            & (self.step_counts > int(self.cfg.stuck_after_steps))
        )

        self.stuck_counter = torch.where(
            stuck_now,
            self.stuck_counter + 1,
            torch.zeros_like(self.stuck_counter),
        )

        p_stuck = -torch.clamp(self.stuck_counter.float() / 40.0, 0.0, 1.0)
        hard_stuck = self.stuck_counter >= int(self.cfg.hard_stuck_after_steps)
        r_step = -torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        continuous_raw = (
            float(self.cfg.w_progress) * r_progress
            + float(self.cfg.w_goal_heading) * r_heading
            + float(self.cfg.w_heading_improve) * r_heading_improve
            + float(self.cfg.w_turn_to_goal) * r_turn_to_goal
            + float(self.cfg.w_goal_forward) * r_forward
            + float(self.cfg.w_negative_progress) * p_negative_progress
            + float(self.cfg.w_backward) * p_backward
            + float(getattr(self.cfg, "w_misaligned_forward", 0.0)) * p_misaligned_forward
            + float(self.cfg.w_slow) * p_slow
            + float(self.cfg.w_no_progress) * p_no_progress
            + float(self.cfg.w_spin_in_place) * p_spin_in_place
            + float(self.cfg.w_spin) * p_spin
            + float(getattr(self.cfg, "w_bad_turn", 0.0)) * p_bad_turn
            + float(self.cfg.w_lateral_vel) * p_lateral_vel
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

        reached = current_dist < float(self.cfg.reach_threshold)

        old_wp_idx = self.current_wp_idx.clone()
        new_wp_idx = torch.where(reached, self.current_wp_idx + 1, self.current_wp_idx)
        self.current_wp_idx = torch.clamp(new_wp_idx, max=int(self.cfg.num_waypoints))

        is_finished = reached & (old_wp_idx >= int(self.cfg.num_waypoints) - 1)
        is_normal_wp = reached & (~is_finished)

        if is_normal_wp.any():
            switch_ids = is_normal_wp.nonzero(as_tuple=False).squeeze(-1)
            next_slots = torch.clamp(old_wp_idx[switch_ids] + 1, max=int(self.cfg.num_waypoints) - 1)
            self._generate_next_forward_waypoint(
                env_ids=switch_ids,
                root_xy=root_pos_local[switch_ids, :2],
                yaw=yaw[switch_ids],
                slot_idx=next_slots,
            )

        event_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        event_reward = torch.where(
            is_normal_wp,
            event_reward + float(self.cfg.rew_waypoint),
            event_reward,
        )
        event_reward = torch.where(
            is_finished,
            event_reward + float(self.cfg.rew_finish),
            event_reward,
        )

        terminated = is_finished
        timeout = self.step_counts >= int(self.cfg.max_episode_length)
        truncated = timeout | hard_stuck

        event_reward = torch.where(
            timeout & (~terminated),
            event_reward + float(self.cfg.rew_timeout),
            event_reward,
        )
        event_reward = torch.where(
            hard_stuck & (~terminated),
            event_reward + float(self.cfg.rew_hard_stuck),
            event_reward,
        )

        reward_raw = continuous + event_reward

        projected_return = self.episode_return + reward_raw
        no_event = event_reward.abs() < 1e-6

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

        if reached.any():
            new_target = self._current_waypoint_local()
            new_dist = torch.norm(new_target - root_pos_local[:, :2], dim=-1)
            self.last_distances = torch.where(reached, new_dist, self.last_distances)

            new_target_angle = torch.atan2(new_target[:, 1] - root_pos_local[:, 1], new_target[:, 0] - root_pos_local[:, 0])
            new_heading_error = torch.atan2(torch.sin(new_target_angle - yaw), torch.cos(new_target_angle - yaw))
            self.last_heading_error_abs = torch.where(reached, torch.abs(new_heading_error), heading_error_abs)
        else:
            self.last_heading_error_abs = heading_error_abs

        done = terminated | truncated

        done_count = done.float().sum()
        finished_count = terminated.float().sum()
        timeout_count = (timeout & (~terminated)).float().sum()
        hard_stuck_count = (hard_stuck & (~terminated)).float().sum()

        self.total_done_episodes += done_count.detach()
        self.total_finished_episodes += finished_count.detach()
        self.total_timeout_episodes += timeout_count.detach()
        self.total_hard_stuck_episodes += hard_stuck_count.detach()

        terminal_return = self.episode_return + reward
        self._update_recent_statistics(
            done=done,
            terminated=terminated,
            timeout=timeout & (~terminated),
            hard_stuck=hard_stuck & (~terminated),
            terminal_return=terminal_return,
            terminal_length=self.step_counts.float(),
            terminal_waypoints=old_wp_idx.float(),
            backward_now=goal_aligned_speed < 0.0,
            slow_now=slow_now,
            stuck_now=stuck_now,
            no_progress_now=no_progress_now,
        )

        episode_finish_rate = self.total_finished_episodes / torch.clamp(self.total_done_episodes, min=1.0)
        episode_timeout_rate = self.total_timeout_episodes / torch.clamp(self.total_done_episodes, min=1.0)
        episode_hard_stuck_rate = self.total_hard_stuck_episodes / torch.clamp(self.total_done_episodes, min=1.0)

        recent = self._recent_stats()

        info = {
            "reward_components": {
                "R_Progress": (float(self.cfg.w_progress) * r_progress).mean().item(),
                "R_Heading": (float(self.cfg.w_goal_heading) * r_heading).mean().item(),
                "R_Heading_Improve": (float(self.cfg.w_heading_improve) * r_heading_improve).mean().item(),
                "R_Forward": (float(self.cfg.w_goal_forward) * r_forward).mean().item(),
                "R_Turn_To_Goal": (float(self.cfg.w_turn_to_goal) * r_turn_to_goal).mean().item(),
                "P_Negative_Progress": (float(self.cfg.w_negative_progress) * p_negative_progress).mean().item(),
                "P_Backward": (float(self.cfg.w_backward) * p_backward).mean().item(),
                "P_Misaligned_Forward": (float(getattr(self.cfg, "w_misaligned_forward", 0.0)) * p_misaligned_forward).mean().item(),
                "P_Slow": (float(self.cfg.w_slow) * p_slow).mean().item(),
                "P_No_Progress": (float(self.cfg.w_no_progress) * p_no_progress).mean().item(),
                "P_Spin_In_Place": (float(self.cfg.w_spin_in_place) * p_spin_in_place).mean().item(),
                "P_Spin": (float(self.cfg.w_spin) * p_spin).mean().item(),
                "P_Bad_Turn": (float(getattr(self.cfg, "w_bad_turn", 0.0)) * p_bad_turn).mean().item(),
                "P_Lateral_Vel": (float(self.cfg.w_lateral_vel) * p_lateral_vel).mean().item(),
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
                "Waypoint_Rate": is_normal_wp.float().mean().item(),
                "Finish_Rate": terminated.float().mean().item(),
                "Timeout_Rate": (timeout & (~terminated)).float().mean().item(),
                "Hard_Stuck_Rate": (hard_stuck & (~terminated)).float().mean().item(),
                "Done_Rate": done.float().mean().item(),
                "Episode_Finish_Rate": episode_finish_rate.item(),
                "Episode_Timeout_Rate": episode_timeout_rate.item(),
                "Episode_Hard_Stuck_Rate": episode_hard_stuck_rate.item(),
                "Episode_Done_Count": self.total_done_episodes.item(),
                "Recent_Finish_Rate": recent["finish_rate"],
                "Recent_Timeout_Rate": recent["timeout_rate"],
                "Recent_Hard_Stuck_Rate": recent["hard_stuck_rate"],
            },
            "telemetry": {
                "Distance_To_Waypoint": current_dist.mean().item(),
                "Distance_P10": torch.quantile(current_dist.detach(), 0.10).item(),
                "Progress": progress.mean().item(),
                "Progress_Pos": progress_pos.mean().item(),
                "Progress_Neg": progress_neg.mean().item(),
                "Progress_EMA": self.progress_ema.mean().item(),
                "Heading_Error": heading_error_abs.mean().item(),
                "Heading_Improve": heading_improve.mean().item(),
                "Heading_Cos": heading_cos.mean().item(),
                "Heading_Gate": heading_gate.mean().item(),
                "Speed_Gate": speed_gate.mean().item(),
                "Distance_Gate": distance_gate.mean().item(),
                "Goal_Aligned_Speed": goal_aligned_speed.mean().item(),
                "Body_Vx": base_lin_vel_b[:, 0].mean().item(),
                "Body_Vy": base_lin_vel_b[:, 1].mean().item(),
                "Body_Wz": base_ang_vel_b[:, 2].mean().item(),
                "Planar_Speed": planar_speed.mean().item(),
                "Forward_Command_Norm": self.forward_command_norm.mean().item(),
                "Turn_Command_Norm": self.turn_command_norm.mean().item(),
                "Left_Wheel_Target_Norm": self.wheel_command_norm[:, 0].mean().item(),
                "Right_Wheel_Target_Norm": self.wheel_command_norm[:, 1].mean().item(),
                "Positive_Linear_Command_Rate": (self.forward_command_norm > 1e-5).float().mean().item(),
                "Wheel_Target_Left": self.wheel_vel_targets_last[:, 0].mean().item(),
                "Wheel_Target_Right": self.wheel_vel_targets_last[:, 1].mean().item(),
                "Wheel_Vel_Left": wheel_vel[:, 0].mean().item(),
                "Wheel_Vel_Right": wheel_vel[:, 1].mean().item(),
                "Raw_Action_Left": self.raw_actions[:, 0].mean().item(),
                "Raw_Action_Right": self.raw_actions[:, 1].mean().item(),
                "Action_Forward_Throttle": self.actions[:, 0].mean().item(),
                "Action_Turn": self.actions[:, 1].mean().item(),
                "Exec_Action_Left": self.wheel_command_norm[:, 0].mean().item(),
                "Exec_Action_Right": self.wheel_command_norm[:, 1].mean().item(),
                "Action_Left": self.wheel_command_norm[:, 0].mean().item(),
                "Action_Right": self.wheel_command_norm[:, 1].mean().item(),
                "Waypoint_Index": self.current_wp_idx.float().mean().item(),
                "Backward_Ratio": (goal_aligned_speed < 0.0).float().mean().item(),
                "Slow_Ratio": slow_now.float().mean().item(),
                "No_Progress_Ratio": no_progress_now.float().mean().item(),
                "Spin_In_Place_Ratio": spin_in_place_now.float().mean().item(),
                "Bad_Turn_Ratio": bad_turn_now.float().mean().item(),
                "Stuck_Ratio": stuck_now.float().mean().item(),
                "Recent_Backward_Ratio": recent["backward_ratio"],
                "Recent_Slow_Ratio": recent["slow_ratio"],
                "Recent_No_Progress_Ratio": recent["no_progress_ratio"],
                "Recent_Stuck_Ratio": recent["stuck_ratio"],
                "Episode_Length": self.step_counts.float().mean().item(),
                "Episode_Return": self.episode_return.mean().item(),
                "Recent_Terminal_Return": recent["terminal_return"],
                "Recent_Terminal_Length": recent["terminal_length"],
                "Recent_Terminal_Waypoints": recent["terminal_waypoints"],
            },
            "debug": {
                "Reward_Min": reward.min().item(),
                "Reward_Max": reward.max().item(),
                "Continuous_Min": continuous.min().item(),
                "Continuous_Max": continuous.max().item(),
                "Event_Min": event_reward.min().item(),
                "Event_Max": event_reward.max().item(),
                "Raw_Action_Min": self.raw_actions.min().item(),
                "Raw_Action_Max": self.raw_actions.max().item(),
                "Smoothed_Action_Min": self.actions.min().item(),
                "Smoothed_Action_Max": self.actions.max().item(),
                "Forward_Command_Min": self.forward_command_norm.min().item(),
                "Forward_Command_Max": self.forward_command_norm.max().item(),
                # Backward-compatible aliases for older tests/log readers.
                # In the forward-throttle + turn action semantics, these refer to
                # the non-negative chassis forward command, not the wheel targets.
                "Exec_Action_Min": self.forward_command_norm.min().item(),
                "Exec_Action_Max": self.forward_command_norm.max().item(),
                "Wheel_Command_Min": self.wheel_command_norm.min().item(),
                "Wheel_Command_Max": self.wheel_command_norm.max().item(),
                "Wheel_Target_Min": self.wheel_vel_targets_last.min().item(),
                "Wheel_Target_Max": self.wheel_vel_targets_last.max().item(),
                "Obs_Dim": float(self.num_observations),
                "Action_Dim": float(self.num_actions),
                "Root_XY_Local_Max": torch.norm(root_pos_local[:, :2], dim=-1).max().item(),
            },
            "waypoint_progress": self.current_wp_idx.clone(),
            "is_success": terminated.detach().clone(),
        }

        return reward, terminated, truncated, info

    def _update_recent_statistics(
        self,
        done: torch.Tensor,
        terminated: torch.Tensor,
        timeout: torch.Tensor,
        hard_stuck: torch.Tensor,
        terminal_return: torch.Tensor,
        terminal_length: torch.Tensor,
        terminal_waypoints: torch.Tensor,
        backward_now: torch.Tensor,
        slow_now: torch.Tensor,
        stuck_now: torch.Tensor,
        no_progress_now: torch.Tensor,
    ) -> None:
        self.recent_backward_ratio.append(float(backward_now.float().mean().detach().cpu().item()))
        self.recent_slow_ratio.append(float(slow_now.float().mean().detach().cpu().item()))
        self.recent_stuck_ratio.append(float(stuck_now.float().mean().detach().cpu().item()))
        self.recent_no_progress_ratio.append(float(no_progress_now.float().mean().detach().cpu().item()))

        if not done.any():
            return

        ids = done.nonzero(as_tuple=False).squeeze(-1)
        finished = terminated[ids].float().detach().cpu().tolist()
        timeouts = timeout[ids].float().detach().cpu().tolist()
        hard_stucks = hard_stuck[ids].float().detach().cpu().tolist()
        returns = terminal_return[ids].float().detach().cpu().tolist()
        lengths = terminal_length[ids].float().detach().cpu().tolist()
        waypoints = terminal_waypoints[ids].float().detach().cpu().tolist()

        for value in finished:
            self.recent_finished.append(float(value))
        for value in timeouts:
            self.recent_timeout.append(float(value))
        for value in hard_stucks:
            self.recent_hard_stuck.append(float(value))
        for value in returns:
            self.recent_terminal_return.append(float(value))
        for value in lengths:
            self.recent_terminal_length.append(float(value))
        for value in waypoints:
            self.recent_terminal_waypoints.append(float(value))

    @staticmethod
    def _mean_deque(values: Deque[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / max(len(values), 1))

    def _recent_stats(self) -> Dict[str, float]:
        return {
            "finish_rate": self._mean_deque(self.recent_finished),
            "timeout_rate": self._mean_deque(self.recent_timeout),
            "hard_stuck_rate": self._mean_deque(self.recent_hard_stuck),
            "terminal_return": self._mean_deque(self.recent_terminal_return),
            "terminal_length": self._mean_deque(self.recent_terminal_length),
            "terminal_waypoints": self._mean_deque(self.recent_terminal_waypoints),
            "backward_ratio": self._mean_deque(self.recent_backward_ratio),
            "slow_ratio": self._mean_deque(self.recent_slow_ratio),
            "stuck_ratio": self._mean_deque(self.recent_stuck_ratio),
            "no_progress_ratio": self._mean_deque(self.recent_no_progress_ratio),
        }

    # ------------------------------------------------------------------
    # Geometry / state helpers
    # ------------------------------------------------------------------
    def _root_pos_local(self) -> torch.Tensor:
        root_pos_w = self.robot.data.root_pos_w
        return root_pos_w - self.env_origins

    def _distance_to_current_waypoint(self) -> torch.Tensor:
        root_pos_local = self._root_pos_local()
        target_local = self._current_waypoint_local()
        return torch.norm(target_local - root_pos_local[:, :2], dim=-1)

    def _heading_error_abs_to_current_waypoint(self) -> torch.Tensor:
        root_pos_local = self._root_pos_local()
        yaw = self._quat_yaw(self.robot.data.root_quat_w)
        target_local = self._current_waypoint_local()
        to_target = target_local - root_pos_local[:, :2]
        target_angle = torch.atan2(to_target[:, 1], to_target[:, 0])
        heading_error = torch.atan2(torch.sin(target_angle - yaw), torch.cos(target_angle - yaw))
        return torch.abs(heading_error)

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
        yaw = self._quat_yaw(self.robot.data.root_quat_w)

        c = torch.cos(yaw)
        s = torch.sin(yaw)

        vx_w = c * vel_b[:, 0] - s * vel_b[:, 1]
        vy_w = s * vel_b[:, 0] + c * vel_b[:, 1]
        return torch.stack([vx_w, vy_w, vel_b[:, 2]], dim=-1)

    def _print_debug_info(self) -> None:
        print("\n" + "=" * 120)
        print("✅ [Task1] Diff-Drive UGV / Jetbot Multi-waypoint Navigation Env Initialized")
        print(f"  num_envs             : {self.num_envs}")
        print(f"  device               : {self.device}")
        print(f"  robot.num_joints     : {self.robot.num_joints}")
        print(f"  num_actions          : {self.num_actions}")
        print(f"  action_protocol      : {self.cfg.action_protocol}")
        print(f"  obs_protocol         : {self.cfg.obs_protocol}")
        print(f"  model_protocol       : {self.cfg.model_protocol}")
        print(f"  single_obs_dim       : {self.cfg.single_obs_dim}")
        print(f"  frame_stack          : {self.cfg.frame_stack}")
        print(f"  num_observations     : {self.num_observations}")
        print(f"  sim_dt               : {self.cfg.sim_dt}")
        print(f"  policy_dt            : {self.dt}")
        print(f"  decimation           : {self.cfg.decimation}")
        print(f"  max_episode_length   : {self.cfg.max_episode_length}")
        print(f"  min_forward_action   : {self.cfg.min_forward_action}")
        print(f"  max_forward_action   : {self.cfg.max_forward_action}")
        print(f"  turn_scale_norm      : {self.cfg.turn_scale_norm}")
        print(f"  wheel_joint_ids      : {self.wheel_joint_ids}")

        if self.robot_joint_names:
            print("  robot.joint_names:")
            for i, name in enumerate(self.robot_joint_names):
                mark = " <wheel>" if i in self.wheel_joint_ids else ""
                print(f"    {i:02d}: {name}{mark}")

        print("=" * 120 + "\n")


JetbotNavigationEnv = DiffDriveTask1Env
Task1Env = DiffDriveTask1Env
DiffDriveNavigationEnv = DiffDriveTask1Env
from __future__ import annotations

import math
import warnings
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*getTypes called on non-existent path.*")

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene

from diff_drive_rl.tasks.task1.task1_config import Task1Config
from diff_drive_rl.tasks.task1.task1_scene import make_diff_drive_task1_scene_cfg


class DiffDriveTask1Env(gym.Env):
    """Jetbot / two-wheel differential-drive Task1: multi-waypoint navigation.

    Action:
        action[:, 0] = left wheel normalized velocity command
        action[:, 1] = right wheel normalized velocity command

    Observation:
        frame_stack = 3, single frame dim = 12:
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

        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        self.last_distances = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
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
        root_state[:, 3:7] = self._yaw_to_quat_wxyz(torch.zeros(n, dtype=torch.float32, device=self.device))
        root_state[:, 7:13] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.reset(env_ids)

        self._generate_waypoints(env_ids)

        self.step_counts[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.progress_ema[env_ids] = 0.0
        self.stuck_counter[env_ids] = 0
        self.obs_buffer[env_ids] = 0.0

        self.scene.update(0.0)

        dist = self._distance_to_current_waypoint()
        self.last_distances[env_ids] = dist[env_ids]

        obs_single = self._compute_single_obs()
        for i in range(int(self.cfg.frame_stack)):
            self.obs_buffer[env_ids, i, :] = obs_single[env_ids]

        obs = self.compute_obs()
        return obs if return_full else obs[env_ids], {}

    @torch.no_grad()
    def step(self, actions_nn: torch.Tensor):
        actions_nn = torch.as_tensor(actions_nn, dtype=torch.float32, device=self.device)
        actions_nn = torch.nan_to_num(actions_nn, nan=0.0, posinf=1.0, neginf=-1.0)
        actions_nn = torch.clamp(actions_nn, -1.0, 1.0)

        self.prev_actions = self.actions.clone()
        self.actions = (
            float(self.cfg.action_tau) * actions_nn
            + (1.0 - float(self.cfg.action_tau)) * self.prev_actions
        )
        self.actions = torch.clamp(self.actions, -1.0, 1.0)

        pre_distances = self.last_distances.clone()

        wheel_vel_targets = self.actions * self.wheel_signs.unsqueeze(0) * float(self.cfg.wheel_speed_scale)

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
    # Waypoints
    # ------------------------------------------------------------------
    def _generate_waypoints(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        n = int(env_ids.numel())
        if n == 0:
            return

        prev = torch.zeros((n, 2), dtype=torch.float32, device=self.device)

        for i in range(int(self.cfg.num_waypoints)):
            target = self._sample_waypoint_candidate(n)

            for _ in range(8):
                dist_to_prev = torch.norm(target - prev, dim=-1)
                bad = dist_to_prev < float(self.cfg.waypoint_min_radius)

                if not bad.any():
                    break

                resample = self._sample_waypoint_candidate(int(bad.sum().item()))
                target[bad] = resample

            self.waypoints_local[env_ids, i, :] = target
            prev = target

        self.current_wp_idx[env_ids] = 0

    def _sample_waypoint_candidate(self, n: int) -> torch.Tensor:
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
        root_pos_local = self._root_pos_local()
        root_quat = self.robot.data.root_quat_w
        yaw = self._quat_yaw(root_quat)

        target_local = self._current_waypoint_local()
        to_target = target_local - root_pos_local[:, :2]

        dist = torch.norm(to_target, dim=-1)
        target_angle = torch.atan2(to_target[:, 1], to_target[:, 0])
        heading_error = torch.atan2(torch.sin(target_angle - yaw), torch.cos(target_angle - yaw))

        base_lin_vel_b = self.robot.data.root_lin_vel_b
        base_ang_vel_b = self.robot.data.root_ang_vel_b
        joint_vel = self.robot.data.joint_vel[:, self.wheel_joint_ids_t]

        dist_norm = torch.clamp(dist / float(self.cfg.dist_norm_scale), 0.0, 5.0)
        body_vx = torch.clamp(base_lin_vel_b[:, 0] / float(self.cfg.lin_vel_scale), -5.0, 5.0)
        body_vy = torch.clamp(base_lin_vel_b[:, 1] / float(self.cfg.lin_vel_scale), -5.0, 5.0)
        body_wz = torch.clamp(base_ang_vel_b[:, 2] / float(self.cfg.ang_vel_scale), -5.0, 5.0)

        wheel_vel = torch.clamp(joint_vel / float(self.cfg.wheel_vel_scale), -5.0, 5.0)

        progress_norm = torch.clamp(
            self.progress_ema / max(float(self.cfg.progress_norm_scale), 1e-6),
            -5.0,
            5.0,
        )

        wp_idx_norm = self.current_wp_idx.float() / max(float(self.cfg.num_waypoints), 1.0)

        obs = torch.stack(
            [
                dist_norm,
                torch.sin(heading_error),
                torch.cos(heading_error),
                body_vx,
                body_vy,
                body_wz,
                wheel_vel[:, 0],
                wheel_vel[:, 1],
                self.actions[:, 0],
                self.actions[:, 1],
                progress_norm,
                wp_idx_norm,
            ],
            dim=-1,
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
        heading_gate = torch.clamp(heading_cos, 0.0, 1.0)

        base_lin_vel_w = self._root_lin_vel_w()
        base_lin_vel_b = self.robot.data.root_lin_vel_b
        base_ang_vel_b = self.robot.data.root_ang_vel_b

        goal_aligned_speed = torch.sum(base_lin_vel_w[:, :2] * dir_to_goal, dim=-1)

        progress_clamped = torch.clamp(
            progress,
            -float(self.cfg.progress_clip),
            float(self.cfg.progress_clip),
        )

        r_progress = progress_clamped
        r_heading = heading_gate
        r_forward = torch.clamp(goal_aligned_speed / 0.8, 0.0, 1.0) * heading_gate

        p_spin = -torch.square(base_ang_vel_b[:, 2])
        p_lateral_vel = -torch.square(base_lin_vel_b[:, 1])

        action_diff = self.actions - self.prev_actions
        p_action_smooth = -torch.mean(torch.square(action_diff), dim=-1)
        p_action_mag = -torch.mean(torch.square(self.actions), dim=-1)

        wheel_vel = self.robot.data.joint_vel[:, self.wheel_joint_ids_t]
        p_wheel_speed = -torch.mean(torch.square(wheel_vel / float(self.cfg.wheel_vel_scale)), dim=-1)

        low_progress = torch.abs(progress) < float(self.cfg.stuck_progress_threshold)
        low_speed = torch.abs(goal_aligned_speed) < float(self.cfg.stuck_speed_threshold)
        far_from_goal = current_dist > float(self.cfg.reach_threshold)
        stuck_now = low_progress & low_speed & far_from_goal & (self.step_counts > int(self.cfg.stuck_after_steps))

        self.stuck_counter = torch.where(
            stuck_now,
            self.stuck_counter + 1,
            torch.zeros_like(self.stuck_counter),
        )

        p_stuck = -torch.clamp(self.stuck_counter.float() / 50.0, 0.0, 1.0)
        r_step = -torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        continuous_raw = (
            float(self.cfg.w_progress) * r_progress
            + float(self.cfg.w_goal_heading) * r_heading
            + float(self.cfg.w_goal_forward) * r_forward
            + float(self.cfg.w_spin) * p_spin
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
        truncated = self.step_counts >= int(self.cfg.max_episode_length)

        event_reward = torch.where(
            truncated & (~terminated),
            event_reward + float(self.cfg.rew_timeout),
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

        done = terminated | truncated

        done_count = done.float().sum()
        finished_count = terminated.float().sum()
        timeout_count = (truncated & (~terminated)).float().sum()

        self.total_done_episodes += done_count.detach()
        self.total_finished_episodes += finished_count.detach()
        self.total_timeout_episodes += timeout_count.detach()

        episode_finish_rate = self.total_finished_episodes / torch.clamp(self.total_done_episodes, min=1.0)
        episode_timeout_rate = self.total_timeout_episodes / torch.clamp(self.total_done_episodes, min=1.0)

        info = {
            "reward_components": {
                "R_Progress": (float(self.cfg.w_progress) * r_progress).mean().item(),
                "R_Heading": (float(self.cfg.w_goal_heading) * r_heading).mean().item(),
                "R_Forward": (float(self.cfg.w_goal_forward) * r_forward).mean().item(),
                "P_Spin": (float(self.cfg.w_spin) * p_spin).mean().item(),
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
                "Timeout_Rate": (truncated & (~terminated)).float().mean().item(),
                "Done_Rate": done.float().mean().item(),
                "Episode_Finish_Rate": episode_finish_rate.item(),
                "Episode_Timeout_Rate": episode_timeout_rate.item(),
                "Episode_Done_Count": self.total_done_episodes.item(),
            },
            "telemetry": {
                "Distance_To_Waypoint": current_dist.mean().item(),
                "Distance_P10": torch.quantile(current_dist.detach(), 0.10).item(),
                "Progress": progress.mean().item(),
                "Progress_EMA": self.progress_ema.mean().item(),
                "Heading_Error": torch.abs(heading_error).mean().item(),
                "Heading_Cos": heading_cos.mean().item(),
                "Goal_Aligned_Speed": goal_aligned_speed.mean().item(),
                "Body_Vx": base_lin_vel_b[:, 0].mean().item(),
                "Body_Vy": base_lin_vel_b[:, 1].mean().item(),
                "Body_Wz": base_ang_vel_b[:, 2].mean().item(),
                "Wheel_Vel_Left": wheel_vel[:, 0].mean().item(),
                "Wheel_Vel_Right": wheel_vel[:, 1].mean().item(),
                "Action_Left": self.actions[:, 0].mean().item(),
                "Action_Right": self.actions[:, 1].mean().item(),
                "Waypoint_Index": self.current_wp_idx.float().mean().item(),
                "Stuck_Ratio": stuck_now.float().mean().item(),
                "Episode_Length": self.step_counts.float().mean().item(),
                "Episode_Return": self.episode_return.mean().item(),
            },
            "debug": {
                "Reward_Min": reward.min().item(),
                "Reward_Max": reward.max().item(),
                "Continuous_Min": continuous.min().item(),
                "Continuous_Max": continuous.max().item(),
                "Event_Min": event_reward.min().item(),
                "Event_Max": event_reward.max().item(),
                "Obs_Dim": float(self.num_observations),
                "Action_Dim": float(self.num_actions),
                "Root_XY_Local_Max": torch.norm(root_pos_local[:, :2], dim=-1).max().item(),
            },
            "waypoint_progress": self.current_wp_idx.clone(),
            "is_success": terminated.detach().clone(),
        }

        return reward, terminated, truncated, info

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
        print(f"  single_obs_dim       : {self.cfg.single_obs_dim}")
        print(f"  frame_stack          : {self.cfg.frame_stack}")
        print(f"  num_observations     : {self.num_observations}")
        print(f"  sim_dt               : {self.cfg.sim_dt}")
        print(f"  policy_dt            : {self.dt}")
        print(f"  decimation           : {self.cfg.decimation}")
        print(f"  max_episode_length   : {self.cfg.max_episode_length}")
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

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

from diff_drive_rl.tasks.task2.task2_config import Task2Config
from diff_drive_rl.tasks.task2.task2_scene import make_diff_drive_task2_scene_cfg
from diff_drive_rl.tasks.task2.task2_world import Task2WorldManager


class DiffDriveTask2Env(gym.Env):
    """Jetbot Task2: analytic obstacle navigation with real Jetbot physics.

    The robot is simulated by Isaac Lab as a real articulation. The large
    obstacle world, LiDAR, risk features, collisions and success checks are
    computed analytically by Task2WorldManager on GPU tensors.

    Action:
        action[:, 0] = left wheel normalized velocity command
        action[:, 1] = right wheel normalized velocity command

    Observation:
        3-frame stack. Single frame dim = 166. Stacked obs dim = 498.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task2Config):
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

        SceneCfg = make_diff_drive_task2_scene_cfg(cfg)
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

        self.world = Task2WorldManager(
            cfg=self.cfg.world_cfg,
            num_envs=self.num_envs,
            device=self.device,
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

        self.global_steps = 0

        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        self.last_goal_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.progress_ema = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.stuck_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.obs_buffer = torch.zeros(
            (self.num_envs, int(self.cfg.frame_stack), int(self.cfg.single_obs_dim)),
            dtype=torch.float32,
            device=self.device,
        )

        self.total_done_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_success_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_collision_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_oob_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
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
            ids, _ = self.robot.find_joints(".*wheel_joint")
            wheel_ids = [int(i) for i in ids]
        except Exception:
            wheel_ids = []

        if len(wheel_ids) < 2:
            print("[WARN][Task2] 未找到 .*wheel_joint，fallback to first 2 joints.", flush=True)
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

        self.world.reset(env_ids=env_ids, global_steps=int(self.global_steps))

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.env_origins[env_ids]
        root_state[:, 0:2] = self.env_origins[env_ids, :2] + self.world.start_pos[env_ids]
        root_state[:, 2] = self.env_origins[env_ids, 2] + float(self.cfg.spawn_height)

        vec_to_goal = self.world.goal_pos[env_ids] - self.world.start_pos[env_ids]
        yaw_to_goal = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])
        yaw_noise = (torch.rand(int(env_ids.numel()), dtype=torch.float32, device=self.device) * 2.0 - 1.0) * 0.35
        reset_yaw = yaw_to_goal + yaw_noise

        root_state[:, 3:7] = self._yaw_to_quat_wxyz(reset_yaw)
        root_state[:, 7:13] = 0.0
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.reset(env_ids)

        self.episode_steps[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.progress_ema[env_ids] = 0.0
        self.stuck_counter[env_ids] = 0
        self.obs_buffer[env_ids] = 0.0

        self.scene.update(0.0)

        root_local = self._root_pos_local()
        yaw = self._yaw()
        nav = self.world.compute_navigation_features(
            root_pos_local=root_local,
            yaw=yaw,
            body_vx=self.robot.data.root_lin_vel_b[:, 0],
            update_lidar_history=True,
        )
        self.last_goal_dist[env_ids] = nav["goal_dist"][env_ids]

        obs_single = self._compute_single_obs(update_lidar_history=False)
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

        pre_goal_dist = self.last_goal_dist.clone()

        wheel_vel_targets = self.actions * self.wheel_signs.unsqueeze(0) * float(self.cfg.max_wheel_speed)

        full_joint_vel_targets = torch.zeros(
            (self.num_envs, self.robot.num_joints),
            dtype=torch.float32,
            device=self.device,
        )
        full_joint_vel_targets[:, self.wheel_joint_ids_t] = wheel_vel_targets
        self.robot.set_joint_velocity_target(full_joint_vel_targets)

        for _ in range(int(self.cfg.decimation)):
            self.world.step_dynamic_obstacles(dt=float(self.cfg.sim_dt))
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(float(self.cfg.sim_dt))

        self.episode_steps += 1
        self.global_steps += self.num_envs

        reward, terminated, truncated, info = self._compute_rewards_and_dones(pre_goal_dist=pre_goal_dist)
        done = terminated | truncated

        obs_single = self._compute_single_obs(update_lidar_history=True)
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

    def _compute_single_obs(self, update_lidar_history: bool = True) -> torch.Tensor:
        root_local = self._root_pos_local()
        yaw = self._yaw()

        base_lin_vel_b = self.robot.data.root_lin_vel_b
        base_ang_vel_b = self.robot.data.root_ang_vel_b

        nav = self.world.compute_navigation_features(
            root_pos_local=root_local,
            yaw=yaw,
            body_vx=base_lin_vel_b[:, 0],
            update_lidar_history=update_lidar_history,
        )

        goal_dist_norm = nav["goal_dist_norm"].unsqueeze(-1)
        goal_xy_body = torch.stack([nav["goal_x_body_norm"], nav["goal_y_body_norm"]], dim=-1)
        heading = torch.stack([nav["heading_sin"], nav["heading_cos"]], dim=-1)

        vel_obs = torch.stack(
            [
                torch.clamp(base_lin_vel_b[:, 0] / float(self.cfg.lin_vel_scale), -5.0, 5.0),
                torch.clamp(base_lin_vel_b[:, 1] / float(self.cfg.lin_vel_scale), -5.0, 5.0),
                torch.clamp(base_ang_vel_b[:, 2] / float(self.cfg.ang_vel_scale), -5.0, 5.0),
            ],
            dim=-1,
        )

        target_speed = torch.clamp(
            self.world.env_target_speed / max(float(self.cfg.target_speed_norm), 1e-6),
            0.0,
            5.0,
        ).unsqueeze(-1)

        action_delta = self.actions - self.prev_actions

        progress_obs = torch.clamp(
            self.progress_ema / max(float(self.cfg.progress_norm_scale), 1e-6),
            -5.0,
            5.0,
        ).unsqueeze(-1)

        obs = torch.cat(
            [
                goal_dist_norm,
                goal_xy_body,
                heading,
                vel_obs,
                target_speed,
                self.actions,
                action_delta,
                progress_obs,
                nav["lidar_norm"],
                nav["lidar_delta"],
                nav["risk_features"],
            ],
            dim=-1,
        )

        if obs.shape[-1] != int(self.cfg.single_obs_dim):
            raise RuntimeError(
                f"Task2 single obs dim mismatch: got {obs.shape[-1]}, expected {self.cfg.single_obs_dim}"
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
    def _compute_rewards_and_dones(self, pre_goal_dist: torch.Tensor):
        root_local = self._root_pos_local()
        yaw = self._yaw()

        base_lin_vel_w = self._root_lin_vel_w()
        base_lin_vel_b = self.robot.data.root_lin_vel_b
        base_ang_vel_b = self.robot.data.root_ang_vel_b

        nav = self.world.compute_navigation_features(
            root_pos_local=root_local,
            yaw=yaw,
            body_vx=base_lin_vel_b[:, 0],
            update_lidar_history=False,
        )
        events = self.world.check_events(root_local)

        current_goal_dist = events["goal_dist"]
        progress = pre_goal_dist - current_goal_dist
        progress_clamped = torch.clamp(progress, -float(self.cfg.progress_clip), float(self.cfg.progress_clip))

        self.progress_ema = 0.90 * self.progress_ema + 0.10 * progress
        self.last_goal_dist = current_goal_dist.detach().clone()

        goal_vec_w = nav["goal_vec_w"]
        goal_dir_w = goal_vec_w / torch.clamp(current_goal_dist.unsqueeze(-1), min=1e-6)
        goal_aligned_speed = torch.sum(base_lin_vel_w[:, :2] * goal_dir_w, dim=-1)

        heading_error = nav["heading_error"]
        heading_cos = nav["heading_cos"]
        heading_gate = torch.clamp(heading_cos, 0.0, 1.0)

        target_speed = self.world.env_target_speed
        target_speed_error = torch.abs(goal_aligned_speed - target_speed)

        r_progress = progress_clamped
        r_goal_speed = torch.exp(-2.0 * torch.square(target_speed_error)) * heading_gate
        r_heading = heading_gate

        turn_direction = torch.sign(heading_error)
        r_turn_to_goal = torch.clamp(
            base_ang_vel_b[:, 2] * turn_direction,
            min=0.0,
            max=1.0,
        ) * torch.clamp(torch.abs(heading_error) / math.pi, 0.0, 1.0)

        risk = nav["risk_features"]
        all_risk = risk[:, 0]
        front_risk = risk[:, 1]
        ttc_proxy = risk[:, 5]
        boundary_risk = risk[:, 6]
        front_clearance = risk[:, 7]

        r_front_clearance = front_clearance
        p_collision_risk = -torch.maximum(all_risk, front_risk)
        p_ttc = -ttc_proxy
        p_boundary = -boundary_risk

        p_spin = -torch.square(base_ang_vel_b[:, 2])
        p_action_smooth = -torch.mean(torch.square(self.actions - self.prev_actions), dim=-1)
        p_action_mag = -torch.mean(torch.square(self.actions), dim=-1)

        wheel_vel = self.robot.data.joint_vel[:, self.wheel_joint_ids_t]
        p_wheel_speed = -torch.mean(
            torch.square(wheel_vel / max(float(self.cfg.max_wheel_speed), 1e-6)),
            dim=-1,
        )

        low_progress = torch.abs(progress) < float(self.cfg.stuck_progress_threshold)
        low_speed = torch.abs(goal_aligned_speed) < float(self.cfg.stuck_speed_threshold)
        far_from_goal = current_goal_dist > float(self.cfg.world_cfg.success_radius)
        stuck_now = low_progress & low_speed & far_from_goal & (self.episode_steps > int(self.cfg.stuck_after_steps))

        self.stuck_counter = torch.where(
            stuck_now,
            self.stuck_counter + 1,
            torch.zeros_like(self.stuck_counter),
        )
        p_stuck = -torch.clamp(self.stuck_counter.float() / 60.0, 0.0, 1.0)

        r_step = -torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        continuous_raw = (
            float(self.cfg.w_progress) * r_progress
            + float(self.cfg.w_goal_speed) * r_goal_speed
            + float(self.cfg.w_heading) * r_heading
            + float(self.cfg.w_turn_to_goal) * r_turn_to_goal
            + float(self.cfg.w_front_clearance) * r_front_clearance
            + float(self.cfg.w_collision_risk) * p_collision_risk
            + float(self.cfg.w_ttc) * p_ttc
            + float(self.cfg.w_boundary) * p_boundary
            + float(self.cfg.w_spin) * p_spin
            + float(self.cfg.w_stuck) * p_stuck
            + float(self.cfg.w_action_smooth) * p_action_smooth
            + float(self.cfg.w_action_mag) * p_action_mag
            + float(self.cfg.w_wheel_speed) * p_wheel_speed
            + float(self.cfg.w_step) * r_step
        )

        continuous = torch.clamp(
            continuous_raw,
            -float(self.cfg.continuous_reward_clip),
            float(self.cfg.continuous_reward_clip),
        )

        success = events["success"]
        collision = events["collision"]
        static_collision = events["static_collision"]
        dynamic_collision = events["dynamic_collision"]
        out_of_bounds = events["out_of_bounds"]
        timeout = self.episode_steps >= int(self.cfg.max_episode_length)

        terminated = success | collision | out_of_bounds
        truncated = timeout & (~terminated)

        event_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        event_reward = torch.where(success, torch.full_like(event_reward, float(self.cfg.rew_success)), event_reward)
        event_reward = torch.where(collision, torch.full_like(event_reward, float(self.cfg.rew_collision)), event_reward)
        event_reward = torch.where(out_of_bounds, torch.full_like(event_reward, float(self.cfg.rew_out_of_bounds)), event_reward)
        event_reward = torch.where(truncated, torch.full_like(event_reward, float(self.cfg.rew_timeout)), event_reward)

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

        done = terminated | truncated

        done_count = done.float().sum()
        success_count = success.float().sum()
        collision_count = collision.float().sum()
        oob_count = out_of_bounds.float().sum()
        timeout_count = truncated.float().sum()

        self.total_done_episodes += done_count.detach()
        self.total_success_episodes += success_count.detach()
        self.total_collision_episodes += collision_count.detach()
        self.total_oob_episodes += oob_count.detach()
        self.total_timeout_episodes += timeout_count.detach()

        denom = torch.clamp(self.total_done_episodes, min=1.0)
        episode_success_rate = self.total_success_episodes / denom
        episode_collision_rate = self.total_collision_episodes / denom
        episode_oob_rate = self.total_oob_episodes / denom
        episode_timeout_rate = self.total_timeout_episodes / denom

        counts = self.world.get_counts()
        world_stats = self.world.get_debug_stats(root_local)

        info = {
            "reward_components": {
                "R_Progress": (float(self.cfg.w_progress) * r_progress).mean().item(),
                "R_Goal_Speed": (float(self.cfg.w_goal_speed) * r_goal_speed).mean().item(),
                "R_Heading": (float(self.cfg.w_heading) * r_heading).mean().item(),
                "R_Turn_To_Goal": (float(self.cfg.w_turn_to_goal) * r_turn_to_goal).mean().item(),
                "R_Front_Clearance": (float(self.cfg.w_front_clearance) * r_front_clearance).mean().item(),
                "P_Collision_Risk": (float(self.cfg.w_collision_risk) * p_collision_risk).mean().item(),
                "P_TTC": (float(self.cfg.w_ttc) * p_ttc).mean().item(),
                "P_Boundary": (float(self.cfg.w_boundary) * p_boundary).mean().item(),
                "P_Spin": (float(self.cfg.w_spin) * p_spin).mean().item(),
                "P_Stuck": (float(self.cfg.w_stuck) * p_stuck).mean().item(),
                "P_Action_Smooth": (float(self.cfg.w_action_smooth) * p_action_smooth).mean().item(),
                "P_Action_Mag": (float(self.cfg.w_action_mag) * p_action_mag).mean().item(),
                "P_Wheel_Speed": (float(self.cfg.w_wheel_speed) * p_wheel_speed).mean().item(),
                "Step": (float(self.cfg.w_step) * r_step).mean().item(),
                "Continuous": continuous.mean().item(),
                "Event": event_reward.mean().item(),
                "Total": reward.mean().item(),
            },
            "events": {
                "Success_Rate": success.float().mean().item(),
                "Collision_Rate": collision.float().mean().item(),
                "Static_Collision_Rate": static_collision.float().mean().item(),
                "Dynamic_Collision_Rate": dynamic_collision.float().mean().item(),
                "Out_Of_Bounds_Rate": out_of_bounds.float().mean().item(),
                "Timeout_Rate": truncated.float().mean().item(),
                "Done_Rate": done.float().mean().item(),
                "Episode_Success_Rate": episode_success_rate.item(),
                "Episode_Collision_Rate": episode_collision_rate.item(),
                "Episode_Out_Of_Bounds_Rate": episode_oob_rate.item(),
                "Episode_Timeout_Rate": episode_timeout_rate.item(),
                "Episode_Done_Count": self.total_done_episodes.item(),
            },
            "telemetry": {
                "Curriculum_K": self.world.curriculum_k(int(self.global_steps)),
                "Stage": self.world.env_stage.float().mean().item(),
                "Target_Speed": target_speed.mean().item(),
                "Goal_Dist": current_goal_dist.mean().item(),
                "Progress": progress.mean().item(),
                "Progress_EMA": self.progress_ema.mean().item(),
                "Goal_Aligned_Speed": goal_aligned_speed.mean().item(),
                "Target_Speed_Error": target_speed_error.mean().item(),
                "Heading_Error": torch.abs(heading_error).mean().item(),
                "Heading_Cos": heading_cos.mean().item(),
                "Body_Vx": base_lin_vel_b[:, 0].mean().item(),
                "Body_Vy": base_lin_vel_b[:, 1].mean().item(),
                "Body_Wz": base_ang_vel_b[:, 2].mean().item(),
                "Lidar_Min": nav["lidar_dist"].min().item(),
                "Lidar_Mean": nav["lidar_dist"].mean().item(),
                "Risk_All": all_risk.mean().item(),
                "Risk_Front": front_risk.mean().item(),
                "Risk_TTC": ttc_proxy.mean().item(),
                "Risk_Boundary": boundary_risk.mean().item(),
                "Front_Clearance": front_clearance.mean().item(),
                "Static_Count": counts["static_count"].mean().item(),
                "Dynamic_Count": counts["dynamic_count"].mean().item(),
                "Min_Static_Signed_Dist": events["min_static_signed_distance"].mean().item(),
                "Min_Dynamic_Signed_Dist": events["min_dynamic_signed_distance"].mean().item(),
                "Min_Obstacle_Signed_Dist": events["min_obstacle_signed_distance"].mean().item(),
                "Boundary_Margin": events["boundary_margin"].mean().item(),
                "Action_Left": self.actions[:, 0].mean().item(),
                "Action_Right": self.actions[:, 1].mean().item(),
                "Wheel_Vel_Left": wheel_vel[:, 0].mean().item(),
                "Wheel_Vel_Right": wheel_vel[:, 1].mean().item(),
                "Stuck_Ratio": stuck_now.float().mean().item(),
                "Episode_Length": self.episode_steps.float().mean().item(),
                "Episode_Return": self.episode_return.mean().item(),
            },
            "world": world_stats,
            "debug": {
                "Obs_Dim": float(self.num_observations),
                "Single_Obs_Dim": float(self.cfg.single_obs_dim),
                "Action_Dim": float(self.num_actions),
                "Reward_Min": reward.min().item(),
                "Reward_Max": reward.max().item(),
                "Continuous_Min": continuous.min().item(),
                "Continuous_Max": continuous.max().item(),
                "Event_Min": event_reward.min().item(),
                "Event_Max": event_reward.max().item(),
                "Root_XY_Local_Max": torch.norm(root_local[:, :2], dim=-1).max().item(),
                "Base_Height_Mean": self.robot.data.root_pos_w[:, 2].mean().item(),
            },
            "is_success": success.detach().clone(),
        }

        return reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Geometry / state helpers
    # ------------------------------------------------------------------
    def _root_pos_local(self) -> torch.Tensor:
        root_pos_w = self.robot.data.root_pos_w
        return (root_pos_w - self.env_origins)[:, :2]

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

    def _print_debug_info(self) -> None:
        print("\n" + "=" * 120)
        print("✅ [Task2] Diff-Drive UGV / Jetbot Analytic Obstacle Navigation Env Initialized")
        print(f"  num_envs               : {self.num_envs}")
        print(f"  device                 : {self.device}")
        print(f"  robot.num_joints       : {self.robot.num_joints}")
        print(f"  num_actions            : {self.num_actions}")
        print(f"  single_obs_dim         : {self.cfg.single_obs_dim}")
        print(f"  frame_stack            : {self.cfg.frame_stack}")
        print(f"  num_observations       : {self.num_observations}")
        print(f"  sim_dt                 : {self.cfg.sim_dt}")
        print(f"  policy_dt              : {self.dt}")
        print(f"  decimation             : {self.cfg.decimation}")
        print(f"  max_episode_length_s   : {self.cfg.max_episode_length_s}")
        print(f"  max_episode_length     : {self.cfg.max_episode_length}")
        print(f"  world.num_lidar_rays   : {self.cfg.world_cfg.num_lidar_rays}")
        print(f"  world.max_static_obs   : {self.cfg.world_cfg.max_static_obs}")
        print(f"  world.max_dynamic_obs  : {self.cfg.world_cfg.max_dynamic_obs}")
        print(f"  wheel_joint_ids        : {self.wheel_joint_ids}")

        if self.robot_joint_names:
            print("  robot.joint_names:")
            for i, name in enumerate(self.robot_joint_names):
                mark = " <wheel>" if i in self.wheel_joint_ids else ""
                print(f"    {i:02d}: {name}{mark}")

        print("=" * 120 + "\n")


JetbotTask2Env = DiffDriveTask2Env
Task2Env = DiffDriveTask2Env
DiffDriveObstacleNavigationEnv = DiffDriveTask2Env

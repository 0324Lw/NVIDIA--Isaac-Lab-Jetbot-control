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
    """Analytic obstacle navigation environment for a differential-drive robot.

    The robot is simulated by Isaac Lab as a real articulation. The large
    obstacle world, LiDAR, risk features, collisions and success checks are
    computed analytically by Task2WorldManager on GPU tensors.

    Action:
        action[:, 0] = forward throttle in [-1, 1], mapped to non-negative forward command
        action[:, 1] = turn command in [-1, 1]

        The environment converts [forward, turn] to left/right wheel velocity targets.
        Chassis-level commanded linear velocity is never negative, so the policy learns
        forward navigation plus steering instead of reverse driving.

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

        # Policy actions are [forward_throttle, turn]. These tensors store the
        # converted wheel commands for logging, reward diagnostics and smoothness checks.
        self.speed_factor = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.forward_command_norm = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.forward_min_stage_norm = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.required_speed_ratio_stage = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.turn_command_norm = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.wheel_command_norm = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        self.last_goal_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.progress_ema = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_heading_error_abs = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.stuck_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.obs_buffer = torch.zeros(
            (self.num_envs, int(self.cfg.frame_stack), int(self.cfg.single_obs_dim)),
            dtype=torch.float32,
            device=self.device,
        )

        self.total_done_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_success_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_collision_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_static_collision_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_dynamic_collision_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_oob_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)

        # Episode-level conditional counters.  Step-level collision masks are often
        # zero at a printed summary instant even if previous episodes ended by
        # collision.  These counters keep terminal statistics consistent with
        # Episode_Collision_Rate and avoid misleading diagnostics in obstacle stages.
        self.total_static_present_done_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_static_present_success_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_static_present_collision_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_static_present_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)

        # 当前窗口 EMA 指标用于训练诊断。累计 Episode_* 会被早期阶段和手动测试污染，
        # 因此训练判断优先看这些 Current_Window_* 字段。
        self.window_success_rate = torch.zeros((), dtype=torch.float32, device=self.device)
        self.window_collision_rate = torch.zeros((), dtype=torch.float32, device=self.device)
        self.window_oob_rate = torch.zeros((), dtype=torch.float32, device=self.device)
        self.window_timeout_rate = torch.zeros((), dtype=torch.float32, device=self.device)
        self.window_goal_aligned_speed = torch.zeros((), dtype=torch.float32, device=self.device)
        self.window_speed_ratio = torch.zeros((), dtype=torch.float32, device=self.device)
        self.window_progress = torch.zeros((), dtype=torch.float32, device=self.device)
        self.window_heading_cos = torch.zeros((), dtype=torch.float32, device=self.device)

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

    def _effective_reset_global_steps(self) -> int:
        """Return the curriculum step used when sampling new episodes.

        When ``force_stage`` is non-negative, all resets must sample that
        exact curriculum stage regardless of the training/global step. The
        world manager accepts a global-step value, so this helper maps the
        requested stage to a representative point inside that stage's threshold
        interval. When ``force_stage < 0``, the environment follows the normal
        global-step curriculum.
        """

        force_stage = int(getattr(self.cfg, "force_stage", -1))
        if force_stage < 0:
            return int(self.global_steps)

        thresholds = tuple(float(v) for v in self.cfg.world_cfg.stage_thresholds)
        if not thresholds:
            return int(self.global_steps)

        stage_idx = max(0, min(force_stage, len(thresholds) - 1))
        k = thresholds[stage_idx]

        # Move slightly inside the requested stage instead of sitting exactly
        # on a boundary. Stage 0 starts at 0 and should remain exactly 0.
        if stage_idx > 0:
            next_k = thresholds[stage_idx + 1] if stage_idx + 1 < len(thresholds) else 1.0
            eps = min(1.0e-6, max(0.0, next_k - k) * 0.01)
            k = min(k + eps, 1.0)

        return int(k * int(self.cfg.world_cfg.curriculum_total_steps))

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

        self.world.reset(env_ids=env_ids, global_steps=self._effective_reset_global_steps())

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
        self.speed_factor[env_ids] = 0.0
        self.forward_command_norm[env_ids] = 0.0
        self.forward_min_stage_norm[env_ids] = 0.0
        self.required_speed_ratio_stage[env_ids] = 0.0
        self.turn_command_norm[env_ids] = 0.0
        self.wheel_command_norm[env_ids] = 0.0
        self.progress_ema[env_ids] = 0.0
        self.last_heading_error_abs[env_ids] = 0.0
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
        self.last_heading_error_abs[env_ids] = torch.abs(nav["heading_error"][env_ids])

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

        wheel_vel_targets = self._actions_to_wheel_velocity_targets(self.actions)

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


    def _actions_to_wheel_velocity_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """把 policy 的 [forward_throttle, turn] 转换成左右轮速度目标。

        action[:, 0] 只控制非负线速度比例；action[:, 1] 控制差速转向。
        最终左右轮可以因转向而一正一负，但左右轮平均命令保持非负，
        从动作空间层面消除“整体倒车”这个坏选择。
        """

        forward_raw = torch.clamp(actions[:, 0], -1.0, 1.0)
        turn_raw = torch.clamp(actions[:, 1], -1.0, 1.0)

        speed_factor = 0.5 * (forward_raw + 1.0)

        stage_ids = torch.clamp(
            getattr(self.world, "env_stage", torch.zeros_like(forward_raw, dtype=torch.long)).long(),
            min=0,
            max=int(self.cfg.world_cfg.num_stages) - 1,
        )
        min_table = torch.tensor(
            tuple(float(v) for v in self.cfg.forward_min_norm_by_stage),
            dtype=torch.float32,
            device=self.device,
        )
        forward_min = min_table[stage_ids]
        forward_min = torch.maximum(forward_min, torch.full_like(forward_min, float(self.cfg.forward_min_norm)))

        forward_norm = forward_min + (float(self.cfg.forward_max_norm) - forward_min) * speed_factor
        forward_norm = torch.clamp(forward_norm, 0.0, 1.0)

        turn_norm = torch.clamp(turn_raw * float(self.cfg.turn_scale_norm), -1.0, 1.0)

        left_norm = torch.clamp(forward_norm - turn_norm, -1.0, 1.0)
        right_norm = torch.clamp(forward_norm + turn_norm, -1.0, 1.0)
        wheel_norm = torch.stack([left_norm, right_norm], dim=-1)

        self.speed_factor = speed_factor.detach().clone()
        self.forward_command_norm = forward_norm.detach().clone()
        self.forward_min_stage_norm = forward_min.detach().clone()
        self.turn_command_norm = turn_norm.detach().clone()
        self.wheel_command_norm = wheel_norm.detach().clone()

        return wheel_norm * self.wheel_signs.unsqueeze(0) * float(self.cfg.max_wheel_speed)

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
    def _predict_dynamic_signed_distance(
        self,
        root_pos_local: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return a short-horizon predicted signed distance to dynamic obstacles.

        Static obstacles and arena boundaries are handled separately. This helper
        keeps dynamic-obstacle anticipation inside the unified safety-proximity
        reward instead of creating another reward component.
        """

        if int(self.cfg.world_cfg.max_dynamic_obs) <= 0:
            return torch.full((self.num_envs,), 1.0e6, dtype=torch.float32, device=self.device)

        if not hasattr(self.world, "dynamic_pos") or not hasattr(self.world, "dynamic_vel"):
            return torch.full((self.num_envs,), 1.0e6, dtype=torch.float32, device=self.device)

        samples = max(int(self.cfg.dynamic_prediction_samples), 1)
        horizon = max(float(self.cfg.dynamic_prediction_horizon_s), 0.0)
        times = torch.linspace(0.0, horizon, samples, dtype=torch.float32, device=self.device)

        robot_xy = root_pos_local[:, None, None, :] + root_lin_vel_w[:, None, None, :2] * times[None, :, None, None]
        dyn_xy = self.world.dynamic_pos[:, None, :, :] + self.world.dynamic_vel[:, None, :, :] * times[None, :, None, None]

        dist = torch.linalg.norm(robot_xy - dyn_xy, dim=-1)
        dyn_radius = self.world.dynamic_radius[:, None, :]
        signed = dist - dyn_radius - float(self.cfg.world_cfg.robot_radius) - float(self.cfg.world_cfg.collision_margin)

        mask = self.world.dynamic_mask[:, None, :]
        signed = torch.where(mask, signed, torch.full_like(signed, 1.0e6))
        return signed.amin(dim=(1, 2))

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
        progress_velocity = progress / max(float(self.dt), 1e-6)
        self.progress_ema = 0.90 * self.progress_ema + 0.10 * progress
        progress_velocity_ema = self.progress_ema / max(float(self.dt), 1e-6)
        self.last_goal_dist = current_goal_dist.detach().clone()

        goal_vec_w = nav["goal_vec_w"]
        goal_dir_w = goal_vec_w / torch.clamp(current_goal_dist.unsqueeze(-1), min=1e-6)
        goal_aligned_speed = torch.sum(base_lin_vel_w[:, :2] * goal_dir_w, dim=-1)

        heading_error = nav["heading_error"]
        heading_error_abs = torch.abs(heading_error)
        heading_cos = nav["heading_cos"]
        heading_improve = self.last_heading_error_abs - heading_error_abs
        self.last_heading_error_abs = heading_error_abs.detach().clone()

        target_speed = torch.clamp(self.world.env_target_speed, min=0.05)

        # Use the EMA progress velocity for reward shaping. Raw progress can be
        # noisy at small dt, while the EMA keeps the reward direction stable.
        # Positive progress is intentionally nonlinear: slow positive movement
        # receives only a small reward, so long low-speed episodes no longer
        # accumulate high return. Negative progress is kept linear and clipped so
        # moving away from the goal is still punished immediately.
        progress_speed_ratio_raw = progress_velocity / target_speed
        progress_speed_ratio = progress_velocity_ema / target_speed
        progress_speed_ratio_clipped = torch.clamp(progress_speed_ratio, -1.0, 1.0)
        positive_progress_ratio = torch.clamp(progress_speed_ratio, 0.0, 1.0)
        negative_progress_ratio = torch.clamp(progress_speed_ratio, -1.0, 0.0)
        r_goal_progress_velocity = torch.where(
            progress_speed_ratio >= 0.0,
            torch.pow(positive_progress_ratio, float(self.cfg.positive_progress_power)),
            negative_progress_ratio,
        )

        static_signed = events["min_static_signed_distance"]
        dynamic_current_signed = events["min_dynamic_signed_distance"]
        dynamic_predicted_signed = self._predict_dynamic_signed_distance(root_local, base_lin_vel_w)
        dynamic_signed = torch.minimum(dynamic_current_signed, dynamic_predicted_signed)
        boundary_signed = events["boundary_margin"]
        path_corridor_signed = self.world.static_path_corridor_signed_distance()
        safety_signed_distance = torch.minimum(torch.minimum(static_signed, dynamic_signed), boundary_signed)

        safety_risk = torch.relu((float(self.cfg.safety_margin) - safety_signed_distance) / max(float(self.cfg.safety_margin), 1e-6))
        safety_risk = torch.clamp(safety_risk, 0.0, float(self.cfg.safety_risk_clip))
        p_safety_proximity = -torch.square(safety_risk)
        safe_navigation_gate = 1.0 - torch.clamp(safety_risk, 0.0, 1.0)

        # Optional risk-aware target speed. Disabled by default.  When enabled,
        # reward tracking speed is reduced only in risky scenes; low-risk scenes
        # keep the nominal target speed, so the policy is not encouraged to learn
        # a globally slow compromise.
        risk_speed_scale = torch.ones_like(target_speed)
        if bool(getattr(self.cfg, "enable_risk_aware_target_speed", False)):
            medium = float(getattr(self.cfg, "risk_speed_medium_threshold", 0.25))
            high = float(getattr(self.cfg, "risk_speed_high_threshold", 0.55))
            medium_scale = float(getattr(self.cfg, "risk_speed_medium_scale", 0.80))
            high_scale = float(getattr(self.cfg, "risk_speed_high_scale", 0.55))
            risk_speed_scale = torch.where(safety_risk >= high, torch.full_like(risk_speed_scale, high_scale), risk_speed_scale)
            risk_speed_scale = torch.where((safety_risk >= medium) & (safety_risk < high), torch.full_like(risk_speed_scale, medium_scale), risk_speed_scale)
            target_speed = torch.clamp(target_speed * risk_speed_scale, min=0.05)

            progress_speed_ratio_raw = progress_velocity / target_speed
            progress_speed_ratio = progress_velocity_ema / target_speed
            progress_speed_ratio_clipped = torch.clamp(progress_speed_ratio, -1.0, 1.0)
            positive_progress_ratio = torch.clamp(progress_speed_ratio, 0.0, 1.0)
            negative_progress_ratio = torch.clamp(progress_speed_ratio, -1.0, 0.0)
            r_goal_progress_velocity = torch.where(
                progress_speed_ratio >= 0.0,
                torch.pow(positive_progress_ratio, float(self.cfg.positive_progress_power)),
                negative_progress_ratio,
            )

        r_heading_improve = torch.clamp(
            heading_improve / max(float(self.cfg.heading_improve_ref), 1e-6),
            -1.0,
            1.0,
        ) * safe_navigation_gate

        # Alignment efficiency: this is not a static heading reward. It is only
        # active when the robot is actually reducing goal distance, so it cannot
        # be exploited by standing still and facing the goal. It encourages the
        # policy to convert forward motion into goal-directed motion instead of
        # driving large low-efficiency arcs.
        positive_motion_gate = torch.pow(
            torch.clamp(progress_speed_ratio, 0.0, 1.0),
            float(self.cfg.aligned_motion_progress_power),
        )
        heading_alignment = torch.clamp(heading_cos, 0.0, 1.0)
        r_aligned_motion = heading_alignment * positive_motion_gate * safe_navigation_gate

        # Focused auxiliary pressure: when heading is poor, do not keep pushing
        # high forward throttle. This lets the differential-drive robot slow down
        # and turn first while preserving the forward-only action protocol.
        misalignment = torch.relu(float(self.cfg.misaligned_forward_heading_cos) - heading_cos) / max(
            float(self.cfg.misaligned_forward_heading_cos) + 1.0,
            1e-6,
        )
        forward_pressure = torch.clamp(self.forward_command_norm - self.forward_min_stage_norm, 0.0, 1.0)
        p_misaligned_forward = -torch.square(misalignment) * forward_pressure * safe_navigation_gate

        sigma_v = torch.clamp(float(self.cfg.target_speed_sigma_ratio) * target_speed, min=0.05)
        speed_gaussian = torch.exp(-torch.square(progress_velocity_ema - target_speed) / (2.0 * torch.square(sigma_v)))
        speed_direction_gate = torch.pow(
            torch.clamp(progress_speed_ratio, 0.0, 1.0),
            float(self.cfg.target_speed_direction_power),
        )
        r_target_speed = speed_gaussian * speed_direction_gate * safe_navigation_gate

        p_action_smooth = -torch.mean(torch.square(self.actions - self.prev_actions), dim=-1)
        r_step = -torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        continuous_raw = (
            float(self.cfg.w_goal_progress_velocity) * r_goal_progress_velocity
            + float(self.cfg.w_heading_improve) * r_heading_improve
            + float(self.cfg.w_target_speed) * r_target_speed
            + float(self.cfg.w_aligned_motion) * r_aligned_motion
            + float(self.cfg.w_misaligned_forward) * p_misaligned_forward
            + float(self.cfg.w_safety_proximity) * p_safety_proximity
            + float(self.cfg.w_action_smooth) * p_action_smooth
            + float(self.cfg.w_step) * r_step
        )

        continuous = torch.clamp(
            continuous_raw,
            -float(self.cfg.continuous_reward_clip),
            float(self.cfg.continuous_reward_clip),
        )

        raw_success = events["success"]
        raw_collision = events["collision"]
        raw_static_collision = events["static_collision"]
        raw_dynamic_collision = events["dynamic_collision"]
        raw_out_of_bounds = events["out_of_bounds"]

        # 事件互斥优先级：out_of_bounds > collision > success > timeout。
        # 目标圈附近撞障碍或出界不能同时计入成功。
        out_of_bounds = raw_out_of_bounds
        collision = raw_collision & (~out_of_bounds)
        static_collision = raw_static_collision & collision
        dynamic_collision = raw_dynamic_collision & collision
        success = raw_success & (~collision) & (~out_of_bounds)
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

        speed_ratio = progress_speed_ratio_raw
        reward_speed_ratio = progress_speed_ratio
        speed_ratio_clamped = torch.clamp(speed_ratio, -2.0, 2.0)

        # Turn diagnostics only: positive means the executed turn command tends
        # to reduce the signed heading error under the action convention
        # action[1] > 0 -> larger right-wheel command -> counter-clockwise turn.
        desired_turn_sign = torch.sign(heading_error)
        turn_to_goal_alignment = desired_turn_sign * self.turn_command_norm
        valid_turn_need = heading_error_abs > 0.10
        correct_turn_ratio = torch.where(
            valid_turn_need,
            (turn_to_goal_alignment > 0.0).float(),
            torch.ones_like(turn_to_goal_alignment),
        )

        low_progress_stuck = (torch.abs(progress_velocity_ema) < 0.08) & (current_goal_dist > 2.0 * float(self.cfg.world_cfg.success_radius))

        alpha = float(self.cfg.window_metric_alpha)
        self.window_success_rate = (1.0 - alpha) * self.window_success_rate + alpha * success.float().mean()
        self.window_collision_rate = (1.0 - alpha) * self.window_collision_rate + alpha * collision.float().mean()
        self.window_oob_rate = (1.0 - alpha) * self.window_oob_rate + alpha * out_of_bounds.float().mean()
        self.window_timeout_rate = (1.0 - alpha) * self.window_timeout_rate + alpha * truncated.float().mean()
        self.window_goal_aligned_speed = (1.0 - alpha) * self.window_goal_aligned_speed + alpha * goal_aligned_speed.mean()
        self.window_speed_ratio = (1.0 - alpha) * self.window_speed_ratio + alpha * torch.clamp(speed_ratio, 0.0, 2.0).mean()
        self.window_progress = (1.0 - alpha) * self.window_progress + alpha * progress.mean()
        self.window_heading_cos = (1.0 - alpha) * self.window_heading_cos + alpha * heading_cos.mean()

        counts = self.world.get_counts()
        static_present = counts["static_count"] > 0.5
        dynamic_present = counts["dynamic_count"] > 0.5
        obstacle_present = static_present | dynamic_present

        done_count = done.float().sum()
        success_count = success.float().sum()
        collision_count = collision.float().sum()
        static_collision_count = static_collision.float().sum()
        dynamic_collision_count = dynamic_collision.float().sum()
        oob_count = out_of_bounds.float().sum()
        timeout_count = truncated.float().sum()

        static_present_done_count = (done & static_present).float().sum()
        static_present_success_count = (success & static_present).float().sum()
        static_present_collision_count = (collision & static_present).float().sum()
        static_present_timeout_count = (truncated & static_present).float().sum()

        self.total_done_episodes += done_count.detach()
        self.total_success_episodes += success_count.detach()
        self.total_collision_episodes += collision_count.detach()
        self.total_static_collision_episodes += static_collision_count.detach()
        self.total_dynamic_collision_episodes += dynamic_collision_count.detach()
        self.total_oob_episodes += oob_count.detach()
        self.total_timeout_episodes += timeout_count.detach()

        self.total_static_present_done_episodes += static_present_done_count.detach()
        self.total_static_present_success_episodes += static_present_success_count.detach()
        self.total_static_present_collision_episodes += static_present_collision_count.detach()
        self.total_static_present_timeout_episodes += static_present_timeout_count.detach()

        denom = torch.clamp(self.total_done_episodes, min=1.0)
        episode_success_rate = self.total_success_episodes / denom
        episode_collision_rate = self.total_collision_episodes / denom
        episode_static_collision_rate = self.total_static_collision_episodes / denom
        episode_dynamic_collision_rate = self.total_dynamic_collision_episodes / denom
        episode_oob_rate = self.total_oob_episodes / denom
        episode_timeout_rate = self.total_timeout_episodes / denom

        static_present_denom = torch.clamp(self.total_static_present_done_episodes, min=1.0)
        episode_success_rate_when_static_present = self.total_static_present_success_episodes / static_present_denom
        episode_collision_rate_when_static_present = self.total_static_present_collision_episodes / static_present_denom
        episode_timeout_rate_when_static_present = self.total_static_present_timeout_episodes / static_present_denom

        world_stats = self.world.get_debug_stats(root_local)
        risk = nav["risk_features"]
        all_risk = risk[:, 0]
        front_risk = risk[:, 1]
        dynamic_risk = risk[:, 4]
        ttc_proxy = risk[:, 5]
        boundary_risk = risk[:, 6]
        front_clearance = risk[:, 7]
        wheel_vel = self.robot.data.joint_vel[:, self.wheel_joint_ids_t]

        def masked_mean(values: torch.Tensor, mask: torch.Tensor, default: float = 0.0) -> torch.Tensor:
            mask = mask.bool()
            if mask.float().sum().item() <= 0:
                return torch.tensor(float(default), dtype=torch.float32, device=self.device)
            return values.float()[mask].mean()

        info = {
            "reward_components": {
                "R_Goal_Progress_Velocity": (float(self.cfg.w_goal_progress_velocity) * r_goal_progress_velocity).mean().item(),
                "R_Heading_Improve": (float(self.cfg.w_heading_improve) * r_heading_improve).mean().item(),
                "R_Target_Speed": (float(self.cfg.w_target_speed) * r_target_speed).mean().item(),
                "R_Aligned_Motion": (float(self.cfg.w_aligned_motion) * r_aligned_motion).mean().item(),
                "P_Misaligned_Forward": (float(self.cfg.w_misaligned_forward) * p_misaligned_forward).mean().item(),
                "P_Safety_Proximity": (float(self.cfg.w_safety_proximity) * p_safety_proximity).mean().item(),
                "P_Action_Smooth": (float(self.cfg.w_action_smooth) * p_action_smooth).mean().item(),
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
                "Episode_Static_Collision_Rate": episode_static_collision_rate.item(),
                "Episode_Dynamic_Collision_Rate": episode_dynamic_collision_rate.item(),
                "Episode_Boundary_Collision_Rate": episode_oob_rate.item(),
                "Episode_Out_Of_Bounds_Rate": episode_oob_rate.item(),
                "Episode_Timeout_Rate": episode_timeout_rate.item(),
                "Episode_Success_Rate_When_Static_Present": episode_success_rate_when_static_present.item(),
                "Episode_Collision_Rate_When_Static_Present": episode_collision_rate_when_static_present.item(),
                "Episode_Timeout_Rate_When_Static_Present": episode_timeout_rate_when_static_present.item(),
                "Episode_Done_Count": self.total_done_episodes.item(),
                "Current_Window_Success_Rate": self.window_success_rate.item(),
                "Current_Window_Collision_Rate": self.window_collision_rate.item(),
                "Current_Window_Out_Of_Bounds_Rate": self.window_oob_rate.item(),
                "Current_Window_Timeout_Rate": self.window_timeout_rate.item(),
            },
            "telemetry": {
                "Curriculum_K": self.world.curriculum_k(int(self.global_steps)),
                "Stage": self.world.env_stage.float().mean().item(),
                "Target_Speed": target_speed.mean().item(),
                "Risk_Speed_Scale": risk_speed_scale.mean().item(),
                "Goal_Dist": current_goal_dist.mean().item(),
                "Progress": progress.mean().item(),
                "Progress_EMA": self.progress_ema.mean().item(),
                "Progress_Velocity": progress_velocity.mean().item(),
                "Progress_Velocity_EMA": progress_velocity_ema.mean().item(),
                "Progress_Speed_Ratio": torch.clamp(progress_speed_ratio_raw, -2.0, 2.0).mean().item(),
                "Reward_Progress_Speed_Ratio": torch.clamp(reward_speed_ratio, -2.0, 2.0).mean().item(),
                "Positive_Progress_Ratio": positive_progress_ratio.mean().item(),
                "Goal_Aligned_Speed": goal_aligned_speed.mean().item(),
                "Speed_Ratio": torch.clamp(speed_ratio, 0.0, 2.0).mean().item(),
                "Signed_Speed_Ratio": speed_ratio_clamped.mean().item(),
                "Target_Speed_Error": torch.abs(progress_velocity_ema - target_speed).mean().item(),
                "Target_Speed_Direction_Gate": speed_direction_gate.mean().item(),
                "Heading_Alignment_Gate": heading_alignment.mean().item(),
                "Positive_Motion_Gate": positive_motion_gate.mean().item(),
                "Misaligned_Forward_Pressure": (-p_misaligned_forward).mean().item(),
                "Heading_Error": heading_error_abs.mean().item(),
                "Heading_Cos": heading_cos.mean().item(),
                "Heading_Improve": heading_improve.mean().item(),
                "Current_Window_Goal_Aligned_Speed": self.window_goal_aligned_speed.item(),
                "Current_Window_Speed_Ratio": self.window_speed_ratio.item(),
                "Current_Window_Progress": self.window_progress.item(),
                "Current_Window_Heading_Cos": self.window_heading_cos.item(),
                "Safety_Risk": safety_risk.mean().item(),
                "Safety_Signed_Distance": safety_signed_distance.mean().item(),
                "Static_Signed_Distance": static_signed.mean().item(),
                "Path_Corridor_Signed_Distance": path_corridor_signed.mean().item(),
                "Dynamic_Signed_Distance": dynamic_current_signed.mean().item(),
                "Dynamic_Predicted_Signed_Distance": dynamic_predicted_signed.mean().item(),
                "Boundary_Margin": boundary_signed.mean().item(),
                "Risk_All": all_risk.mean().item(),
                "Risk_Front": front_risk.mean().item(),
                "Risk_Dynamic": dynamic_risk.mean().item(),
                "Risk_TTC": ttc_proxy.mean().item(),
                "Risk_Boundary": boundary_risk.mean().item(),
                "Front_Clearance": front_clearance.mean().item(),
                "Lidar_Min": nav["lidar_dist"].min().item(),
                "Lidar_Mean": nav["lidar_dist"].mean().item(),
                "Static_Count": counts["static_count"].mean().item(),
                "Dynamic_Count": counts["dynamic_count"].mean().item(),
                "Static_Present_Rate": static_present.float().mean().item(),
                "Dynamic_Present_Rate": dynamic_present.float().mean().item(),
                "Obstacle_Present_Rate": obstacle_present.float().mean().item(),
                # Step-level conditional metrics are useful for instantaneous
                # debugging; the Episode_* versions below are the authoritative
                # terminal statistics used for stage decisions.
                "Success_Rate_When_Static_Present": masked_mean(success.float(), static_present, 0.0).item(),
                "Collision_Rate_When_Static_Present": masked_mean(collision.float(), static_present, 0.0).item(),
                "Timeout_Rate_When_Static_Present": masked_mean(truncated.float(), static_present, 0.0).item(),
                "Episode_Success_Rate_When_Static_Present": episode_success_rate_when_static_present.item(),
                "Episode_Collision_Rate_When_Static_Present": episode_collision_rate_when_static_present.item(),
                "Episode_Timeout_Rate_When_Static_Present": episode_timeout_rate_when_static_present.item(),
                "Min_Static_Signed_Dist_When_Present": masked_mean(static_signed, static_present, 1e6).item(),
                "Path_Corridor_Signed_Distance_When_Present": masked_mean(path_corridor_signed, static_present, 1e6).item(),
                "Risk_Front_When_Static_Present": masked_mean(front_risk, static_present, 0.0).item(),
                "Min_Static_Signed_Dist": static_signed.mean().item(),
                "Min_Dynamic_Signed_Dist": dynamic_current_signed.mean().item(),
                "Min_Obstacle_Signed_Dist": events["min_obstacle_signed_distance"].mean().item(),
                "Action_Forward_Throttle": self.actions[:, 0].mean().item(),
                "Action_Turn": self.actions[:, 1].mean().item(),
                "Speed_Factor": self.speed_factor.mean().item(),
                "Forward_Min_Stage_Norm": self.forward_min_stage_norm.mean().item(),
                "Forward_Command_Norm": self.forward_command_norm.mean().item(),
                "Turn_Command_Norm": self.turn_command_norm.mean().item(),
                "Turn_To_Goal_Alignment": turn_to_goal_alignment.mean().item(),
                "Correct_Turn_Ratio": correct_turn_ratio.mean().item(),
                "Left_Wheel_Target_Norm": self.wheel_command_norm[:, 0].mean().item(),
                "Right_Wheel_Target_Norm": self.wheel_command_norm[:, 1].mean().item(),
                "Positive_Linear_Command_Rate": (self.forward_command_norm > 1e-4).float().mean().item(),
                "Wheel_Vel_Left": wheel_vel[:, 0].mean().item(),
                "Wheel_Vel_Right": wheel_vel[:, 1].mean().item(),
                "Action_Smooth": torch.mean(torch.square(self.actions - self.prev_actions), dim=-1).mean().item(),
                "Low_Progress_Ratio": low_progress_stuck.float().mean().item(),
                "Stuck_Ratio": low_progress_stuck.float().mean().item(),
                "Near_Goal_Rate": (current_goal_dist <= (2.0 * float(self.cfg.world_cfg.success_radius))).float().mean().item(),
                "Episode_Length": self.episode_steps.float().mean().item(),
                "Episode_Return": self.episode_return.mean().item(),
            },
            "world": world_stats,
            "debug": {
                "Obs_Dim": float(self.num_observations),
                "Single_Obs_Dim": float(self.cfg.single_obs_dim),
                "Action_Dim": float(self.num_actions),
                "Continuous_Min": continuous.min().item(),
                "Continuous_Max": continuous.max().item(),
                "Event_Min": event_reward.min().item(),
                "Event_Max": event_reward.max().item(),
                "Reward_Min": reward.min().item(),
                "Reward_Max": reward.max().item(),
                "Root_XY_Local_Max": root_local.abs().max().item(),
                "Base_Height_Mean": self.robot.data.root_pos_w[:, 2].mean().item(),
            },
            "is_success": success.float(),
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
        print("✅ [Task2] Diff-Drive UGV Analytic Obstacle Navigation Env Initialized")
        print(f"  num_envs               : {self.num_envs}")
        print(f"  device                 : {self.device}")
        print(f"  robot.num_joints       : {self.robot.num_joints}")
        print(f"  num_actions            : {self.num_actions} ([forward_throttle, turn])")
        print(f"  action_protocol        : {self.cfg.action_protocol}")
        print(f"  obs_protocol           : {self.cfg.obs_protocol}")
        print(f"  model_protocol         : {self.cfg.model_protocol}")
        print(f"  core_single_obs_dim    : {self.cfg.core_single_obs_dim}")
        print(f"  single_obs_dim         : {self.cfg.single_obs_dim}")
        print(f"  frame_stack            : {self.cfg.frame_stack}")
        print(f"  num_observations       : {self.num_observations}")
        print(f"  sim_dt                 : {self.cfg.sim_dt}")
        print(f"  policy_dt              : {self.dt}")
        print(f"  decimation             : {self.cfg.decimation}")
        print(f"  max_episode_length_s   : {self.cfg.max_episode_length_s}")
        print(f"  max_episode_length     : {self.cfg.max_episode_length}")
        print(f"  force_stage            : {self.cfg.force_stage}")
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


Task2Env = DiffDriveTask2Env
DiffDriveObstacleNavigationEnv = DiffDriveTask2Env
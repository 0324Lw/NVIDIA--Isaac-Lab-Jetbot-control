from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Diff-Drive UGV / Jetbot Task4 TRUE skrl PPO model")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=0.0)
parser.add_argument("--stage", type=int, default=-1)
parser.add_argument("--print-interval", type=int, default=20)
parser.add_argument("--max-episode-length-s", type=float, default=35.0)
parser.add_argument("--max-wheel-speed", type=float, default=32.0)
parser.add_argument("--reverse-speed-fraction", type=float, default=0.35)
parser.add_argument("--left-wheel-sign", type=float, default=1.0)
parser.add_argument("--right-wheel-sign", type=float, default=1.0)
parser.add_argument("--slow-action-scale", type=float, default=1.0)
parser.add_argument("--disable-physical-asset-teleport", action="store_true")
parser.add_argument("--visualize", action="store_true")
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.headless = not bool(args_cli.visualize)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from diff_drive_rl.tasks.task4.task4_config import Task4Config
from diff_drive_rl.tasks.task4.task4_env import DiffDriveTask4Env
from skrl.models.torch import GaussianMixin, Model


# ======================================================================
# Evaluation curriculum
# ======================================================================

class Task4EvalCurriculum:
    @staticmethod
    def stage_from_progress(k: float) -> int:
        k = max(0.0, min(1.0, float(k)))
        if k < 0.12:
            return 0
        if k < 0.28:
            return 1
        if k < 0.45:
            return 2
        if k < 0.65:
            return 3
        if k < 0.82:
            return 4
        return 5

    @classmethod
    def apply(cls, cfg: Task4Config, start_k: float, fixed_stage: int = -1) -> Dict[str, float]:
        wcfg = cfg.world_cfg

        if int(fixed_stage) >= 0:
            stage = int(max(0, min(5, fixed_stage)))
            k = float(start_k)
        else:
            k = max(0.0, min(1.0, float(start_k)))
            stage = cls.stage_from_progress(k)

        wcfg.gate_gap_width = 1.45
        wcfg.formation_scale_gate = 0.80
        wcfg.max_static_obstacles = 8
        wcfg.obstacle_count_by_stage = (0, 2, 4, 5, 7, 8)

        if stage == 0:
            wcfg.start_center_y_range = (-0.60, 0.60)
            wcfg.start_heading_range = (-0.06, 0.06)
            wcfg.goal_y_range = (-0.90, 0.90)
            wcfg.goal_yaw_range = (-0.08, 0.08)
            wcfg.formation_scale_range = (1.00, 1.10)
            wcfg.max_speed_range = (0.72, 0.90)
            wcfg.max_yaw_rate_range = (1.50, 1.90)
            wcfg.action_delay_frame_range = (0, 0)
            wcfg.action_deadband_range = (0.00, 0.010)
            wcfg.action_ema_alpha_range = (0.90, 1.00)
            wcfg.motor_strength_range = (0.99, 1.01)
            wcfg.motor_bias_range = (-0.004, 0.004)
            wcfg.wheel_radius_scale_range = (0.995, 1.005)
            wcfg.lidar_noise_std_range = (0.000, 0.006)
            wcfg.lidar_outlier_prob_range = (0.000, 0.001)
            wcfg.lidar_dropout_prob_range = (0.000, 0.001)
            wcfg.lidar_yaw_offset_range = (-math.radians(0.3), math.radians(0.3))
            cfg.w_team_progress = 5.20
            cfg.w_center_speed = 0.28
            cfg.w_formation_mean = 0.99
            cfg.w_formation_agent = 0.45
            cfg.w_gate_pass = 0.0
            cfg.w_obstacle_risk = 0.04
            cfg.w_gate_risk = 0.00
            cfg.w_pair_risk = 0.38

        elif stage == 1:
            wcfg.start_center_y_range = (-0.90, 0.90)
            wcfg.start_heading_range = (-0.09, 0.09)
            wcfg.goal_y_range = (-1.50, 1.50)
            wcfg.goal_yaw_range = (-0.12, 0.12)
            wcfg.formation_scale_range = (0.95, 1.15)
            wcfg.max_speed_range = (0.70, 1.00)
            wcfg.max_yaw_rate_range = (1.50, 2.10)
            wcfg.action_delay_frame_range = (0, 1)
            wcfg.action_deadband_range = (0.00, 0.020)
            wcfg.action_ema_alpha_range = (0.80, 1.00)
            wcfg.motor_strength_range = (0.96, 1.04)
            wcfg.motor_bias_range = (-0.012, 0.012)
            wcfg.wheel_radius_scale_range = (0.985, 1.015)
            wcfg.lidar_noise_std_range = (0.000, 0.012)
            wcfg.lidar_outlier_prob_range = (0.000, 0.004)
            wcfg.lidar_dropout_prob_range = (0.000, 0.004)
            wcfg.lidar_yaw_offset_range = (-math.radians(0.7), math.radians(0.7))
            cfg.w_team_progress = 5.60
            cfg.w_center_speed = 0.38
            cfg.w_formation_mean = 0.55
            cfg.w_formation_agent = 0.28
            cfg.w_gate_pass = 0.0
            cfg.w_obstacle_risk = 0.12
            cfg.w_gate_risk = 0.00
            cfg.w_pair_risk = 0.28

        elif stage == 2:
            wcfg.start_center_y_range = (-1.10, 1.10)
            wcfg.start_heading_range = (-0.12, 0.12)
            wcfg.goal_y_range = (-2.20, 2.20)
            wcfg.goal_yaw_range = (-0.18, 0.18)
            wcfg.formation_scale_range = (0.90, 1.18)
            wcfg.max_speed_range = (0.70, 1.10)
            wcfg.max_yaw_rate_range = (1.50, 2.30)
            wcfg.action_delay_frame_range = (0, 1)
            wcfg.action_deadband_range = (0.00, 0.035)
            wcfg.action_ema_alpha_range = (0.70, 0.98)
            wcfg.motor_strength_range = (0.93, 1.07)
            wcfg.motor_bias_range = (-0.020, 0.020)
            wcfg.wheel_radius_scale_range = (0.975, 1.025)
            wcfg.lidar_noise_std_range = (0.003, 0.022)
            wcfg.lidar_outlier_prob_range = (0.000, 0.008)
            wcfg.lidar_dropout_prob_range = (0.000, 0.008)
            wcfg.lidar_yaw_offset_range = (-math.radians(1.1), math.radians(1.1))
            cfg.w_team_progress = 5.15
            cfg.w_center_speed = 0.34
            cfg.w_formation_mean = 0.65
            cfg.w_formation_agent = 0.32
            cfg.w_gate_pass = 0.0
            cfg.w_obstacle_risk = 0.22
            cfg.w_gate_risk = 0.00
            cfg.w_pair_risk = 0.32

        elif stage == 3:
            wcfg.start_center_y_range = (-1.20, 1.20)
            wcfg.start_heading_range = (-0.13, 0.13)
            wcfg.goal_y_range = (-2.40, 2.40)
            wcfg.goal_yaw_range = (-0.18, 0.18)
            wcfg.formation_scale_range = (0.92, 1.10)
            wcfg.max_speed_range = (0.68, 1.05)
            wcfg.max_yaw_rate_range = (1.60, 2.35)
            wcfg.action_delay_frame_range = (0, 2)
            wcfg.action_deadband_range = (0.005, 0.045)
            wcfg.action_ema_alpha_range = (0.62, 0.94)
            wcfg.motor_strength_range = (0.90, 1.10)
            wcfg.motor_bias_range = (-0.030, 0.030)
            wcfg.wheel_radius_scale_range = (0.960, 1.040)
            wcfg.lidar_noise_std_range = (0.004, 0.030)
            wcfg.lidar_outlier_prob_range = (0.000, 0.013)
            wcfg.lidar_dropout_prob_range = (0.000, 0.010)
            wcfg.lidar_yaw_offset_range = (-math.radians(1.5), math.radians(1.5))
            cfg.w_team_progress = 4.80
            cfg.w_center_speed = 0.30
            cfg.w_formation_mean = 0.72
            cfg.w_formation_agent = 0.36
            cfg.w_gate_pass = 1.20
            cfg.w_obstacle_risk = 0.26
            cfg.w_gate_risk = 0.22
            cfg.w_pair_risk = 0.35

        elif stage == 4:
            wcfg.start_center_y_range = (-1.35, 1.35)
            wcfg.start_heading_range = (-0.16, 0.16)
            wcfg.goal_y_range = (-2.80, 2.80)
            wcfg.goal_yaw_range = (-0.22, 0.22)
            wcfg.formation_scale_range = (0.88, 1.18)
            wcfg.max_speed_range = (0.65, 1.18)
            wcfg.max_yaw_rate_range = (1.50, 2.50)
            wcfg.action_delay_frame_range = (0, 3)
            wcfg.action_deadband_range = (0.010, 0.055)
            wcfg.action_ema_alpha_range = (0.50, 0.88)
            wcfg.motor_strength_range = (0.86, 1.14)
            wcfg.motor_bias_range = (-0.036, 0.036)
            wcfg.wheel_radius_scale_range = (0.950, 1.050)
            wcfg.lidar_noise_std_range = (0.005, 0.038)
            wcfg.lidar_outlier_prob_range = (0.000, 0.020)
            wcfg.lidar_dropout_prob_range = (0.000, 0.015)
            wcfg.lidar_yaw_offset_range = (-math.radians(1.8), math.radians(1.8))
            cfg.w_team_progress = 4.45
            cfg.w_center_speed = 0.28
            cfg.w_formation_mean = 0.76
            cfg.w_formation_agent = 0.38
            cfg.w_gate_pass = 1.10
            cfg.w_obstacle_risk = 0.30
            cfg.w_gate_risk = 0.25
            cfg.w_pair_risk = 0.36

        else:
            wcfg.start_center_y_range = (-1.50, 1.50)
            wcfg.start_heading_range = (-0.20, 0.20)
            wcfg.goal_y_range = (-3.00, 3.00)
            wcfg.goal_yaw_range = (-0.25, 0.25)
            wcfg.formation_scale_range = (0.85, 1.20)
            wcfg.max_speed_range = (0.70, 1.25)
            wcfg.max_yaw_rate_range = (1.50, 2.60)
            wcfg.action_delay_frame_range = (0, 3)
            wcfg.action_deadband_range = (0.00, 0.060)
            wcfg.action_ema_alpha_range = (0.35, 0.75)
            wcfg.motor_strength_range = (0.82, 1.18)
            wcfg.motor_bias_range = (-0.040, 0.040)
            wcfg.wheel_radius_scale_range = (0.94, 1.06)
            wcfg.lidar_noise_std_range = (0.005, 0.045)
            wcfg.lidar_outlier_prob_range = (0.000, 0.025)
            wcfg.lidar_dropout_prob_range = (0.000, 0.020)
            wcfg.lidar_yaw_offset_range = (-math.radians(2.0), math.radians(2.0))
            cfg.w_team_progress = 4.20
            cfg.w_center_speed = 0.26
            cfg.w_formation_mean = 0.78
            cfg.w_formation_agent = 0.40
            cfg.w_gate_pass = 1.00
            cfg.w_obstacle_risk = 0.32
            cfg.w_gate_risk = 0.28
            cfg.w_pair_risk = 0.38

        cfg.curriculum_stage = stage

        return {
            "Curriculum_K": float(k),
            "Curriculum_Stage": float(stage),
            "Fixed_Stage": float(fixed_stage),
            "Obstacle_Count_Expected": float(wcfg.obstacle_count_by_stage[stage]),
            "Gate_Expected": float(stage >= wcfg.gate_stage_start),
            "Formation_Scale_Low": float(wcfg.formation_scale_range[0]),
            "Formation_Scale_High": float(wcfg.formation_scale_range[1]),
            "Max_Speed_High": float(wcfg.max_speed_range[1]),
            "Delay_Max": float(wcfg.action_delay_frame_range[1]),
            "Deadband_Max": float(wcfg.action_deadband_range[1]),
            "Lidar_Noise_Max": float(wcfg.lidar_noise_std_range[1]),
            "Reward_w_team_progress": float(cfg.w_team_progress),
            "Reward_w_formation_mean": float(cfg.w_formation_mean),
            "Reward_w_obstacle_risk": float(cfg.w_obstacle_risk),
            "Reward_w_gate_risk": float(cfg.w_gate_risk),
            "Reward_w_pair_risk": float(cfg.w_pair_risk),
        }


# ======================================================================
# Flattened eval wrapper
# ======================================================================

class DiffDriveTask4EvalWrapper(gym.Env):
    def __init__(self, env: DiffDriveTask4Env):
        super().__init__()

        self.env = env
        self.physical_num_envs = int(env.num_envs)
        self.num_agents = int(env.num_agents)
        self.num_envs = self.physical_num_envs * self.num_agents
        self.device = env.device

        self.policy_obs_dim = int(env.num_observations)
        self.base_state_dim = int(env.num_privileged_obs)
        self.critic_obs_dim = self.base_state_dim + self.num_agents
        self.action_dim = int(env.num_actions)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.policy_obs_dim,),
            dtype=np.float32,
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.critic_obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.agent_id_onehot_flat = torch.eye(
            self.num_agents,
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.physical_num_envs, 1)

    def _flatten_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.reshape(self.num_envs, self.policy_obs_dim)

    def _flatten_state(self, state: torch.Tensor) -> torch.Tensor:
        repeated_state = state.repeat_interleave(self.num_agents, dim=0)
        return torch.cat([repeated_state, self.agent_id_onehot_flat], dim=-1)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        state = info.get("state", self.env.compute_privileged_obs())
        return {
            "policy": self._flatten_obs(obs).clone(),
            "critic": self._flatten_state(state).clone(),
        }, info or {}

    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = torch.clamp(
            torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0),
            -1.0,
            1.0,
        )

        action_env = actions.view(self.physical_num_envs, self.num_agents, self.action_dim)
        obs, reward, terminated, truncated, info = self.env.step(action_env)
        state = info.get("state", self.env.compute_privileged_obs())

        return {
            "policy": self._flatten_obs(obs).clone(),
            "critic": self._flatten_state(state).clone(),
        }, reward.reshape(self.num_envs), terminated.repeat_interleave(self.num_agents), truncated.repeat_interleave(self.num_agents), info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# ======================================================================
# Actor
# ======================================================================

class DiffDriveTask4Actor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        init_log_std: float = -0.85,
        min_log_std: float = -3.0,
        max_log_std: float = 0.5,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=float(min_log_std),
            max_log_std=float(max_log_std),
            reduction="sum",
        )

        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.init_log_std = float(init_log_std)

        obs_dim = int(observation_space.shape[0])
        act_dim = int(action_space.shape[0])

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
        )
        self.log_std_parameter = nn.Parameter(
            torch.full((act_dim,), float(init_log_std), dtype=torch.float32)
        )

    def compute(self, inputs, role):
        states = inputs.get("observations", inputs.get("states"))
        states = torch.nan_to_num(states, nan=0.0, posinf=10.0, neginf=-10.0)
        states = torch.clamp(states, -10.0, 10.0)

        mean = self.net(states)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        mean = torch.clamp(mean, -5.0, 5.0)

        log_std = torch.clamp(self.log_std_parameter, self.min_log_std, self.max_log_std)
        log_std = torch.nan_to_num(
            log_std,
            nan=self.init_log_std,
            posinf=self.max_log_std,
            neginf=self.min_log_std,
        )

        return mean, {"log_std": log_std}

    @torch.no_grad()
    def act_deterministic_direct(self, states: torch.Tensor) -> torch.Tensor:
        actions, _ = self.compute({"states": states}, role="policy")
        return torch.clamp(actions, -1.0, 1.0)


# ======================================================================
# Checkpoint helpers
# ======================================================================

def torch_load_checkpoint(path: Path, device: str):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def resolve_checkpoint(path: str) -> Path:
    p = Path(path).expanduser().resolve()

    if p.is_file():
        return p

    if p.is_dir():
        candidates = [
            p / "diff_drive_task4_model.pt",
            p / "final_checkpoint" / "diff_drive_task4_model.pt",
            p / "checkpoint_5000" / "diff_drive_task4_model.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        pt_files = sorted(p.glob("*.pt"))
        for pt in pt_files:
            if pt.name.endswith("_preprocessor.pt"):
                continue
            if pt.name in {"train_metadata.pt", "jetbot_task4_skrl_model.pt"}:
                continue
            return pt

    return p


def _find_norm_tensors(norm_state: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not isinstance(norm_state, dict):
        return None, None

    mean = None
    var = None

    for key in ["running_mean", "_running_mean", "mean", "obs_mean"]:
        if key in norm_state:
            mean = norm_state[key]
            break

    for key in ["running_variance", "_running_variance", "variance", "var", "obs_var"]:
        if key in norm_state:
            var = norm_state[key]
            break

    return mean, var


def normalize_with_saved_obs_norm(obs: torch.Tensor, obs_norm: Optional[Dict[str, Any]]) -> torch.Tensor:
    if not obs_norm:
        return obs

    mean, var = _find_norm_tensors(obs_norm)
    if mean is None or var is None:
        return obs

    mean = torch.as_tensor(mean, device=obs.device, dtype=torch.float32).view(-1)
    var = torch.as_tensor(var, device=obs.device, dtype=torch.float32).view(-1)

    if mean.numel() != obs.shape[-1] or var.numel() != obs.shape[-1]:
        return obs

    return torch.clamp((obs - mean) / torch.sqrt(var + 1.0e-8), -10.0, 10.0)


def load_policy_checkpoint(ckpt_path: Path, env: DiffDriveTask4EvalWrapper):
    ckpt = torch_load_checkpoint(ckpt_path, env.device)

    if not isinstance(ckpt, dict) or "policy" not in ckpt:
        raise RuntimeError(
            f"当前测试脚本需要 task4_train.py 保存的 eval checkpoint: diff_drive_task4_model.pt\n"
            f"当前文件不是 eval checkpoint: {ckpt_path}"
        )

    metadata = ckpt.get("metadata", {})
    args = ckpt.get("args", {})

    if not bool(metadata.get("uses_skrl", False)):
        raise RuntimeError("checkpoint metadata 缺少 uses_skrl=True，请使用 TRUE skrl 版本重新训练。")

    if not bool(metadata.get("ctde_mappo_style", False)):
        raise RuntimeError("checkpoint metadata 缺少 ctde_mappo_style=True。")

    if not bool(metadata.get("shared_actor", False)):
        raise RuntimeError("checkpoint metadata 缺少 shared_actor=True。")

    if not bool(metadata.get("centralized_critic", False)):
        raise RuntimeError("checkpoint metadata 缺少 centralized_critic=True。")

    expected_num_agents = int(metadata.get("num_agents", env.num_agents))
    expected_actor_dim = int(metadata.get("actor_obs_dim", env.observation_space.shape[0]))
    expected_action_dim = int(metadata.get("action_dim", env.action_space.shape[0]))
    expected_world_priv_dim = int(metadata.get("world_priv_dim", 96))
    expected_critic_dim = int(metadata.get("critic_obs_dim", env.state_space.shape[0]))

    if expected_num_agents != env.num_agents:
        raise RuntimeError(f"num_agents mismatch: checkpoint={expected_num_agents}, env={env.num_agents}")

    if expected_actor_dim != env.observation_space.shape[0]:
        raise RuntimeError(f"actor obs dim mismatch: checkpoint={expected_actor_dim}, env={env.observation_space.shape[0]}")

    if expected_action_dim != env.action_space.shape[0]:
        raise RuntimeError(f"action dim mismatch: checkpoint={expected_action_dim}, env={env.action_space.shape[0]}")

    if expected_world_priv_dim != 96:
        raise RuntimeError(f"world priv dim mismatch: checkpoint={expected_world_priv_dim}, expected=96")

    if expected_critic_dim != env.state_space.shape[0]:
        raise RuntimeError(f"critic obs dim mismatch: checkpoint={expected_critic_dim}, env={env.state_space.shape[0]}")

    if int(metadata.get("single_actor_obs_dim", 156)) != 156:
        raise RuntimeError(f"single_actor_obs_dim mismatch: {metadata.get('single_actor_obs_dim')}")

    if int(metadata.get("frame_stack", 4)) != 4:
        raise RuntimeError(f"frame_stack mismatch: {metadata.get('frame_stack')}")

    if int(metadata.get("lidar_pool_bins", 48)) != 48:
        raise RuntimeError(f"lidar_pool_bins mismatch: {metadata.get('lidar_pool_bins')}")

    if int(metadata.get("risk_feature_dim", 16)) != 16:
        raise RuntimeError(f"risk_feature_dim mismatch: {metadata.get('risk_feature_dim')}")

    policy = DiffDriveTask4Actor(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        init_log_std=float(args.get("init_log_std", -0.85)),
        min_log_std=float(args.get("min_log_std", -3.0)),
        max_log_std=float(args.get("max_log_std", 0.5)),
    ).to(env.device)

    policy.load_state_dict(ckpt["policy"], strict=True)
    policy.eval()

    actor_obs_norm = ckpt.get("actor_obs_norm", None)
    trained_agent_steps = int(ckpt.get("agent_steps", 0))
    curriculum_info = ckpt.get("curriculum_info", {})

    return policy, actor_obs_norm, trained_agent_steps, metadata, curriculum_info


# ======================================================================
# Reporting helpers
# ======================================================================

def to_float(x: Any):
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
    except Exception:
        return None
    return None


def flat_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}

    for key, value in (data or {}).items():
        name = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flat_dict(value, name))
        else:
            val = to_float(value)
            if val is not None and np.isfinite(val):
                out[name] = val

    return out


def summarize(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not records:
        return {}

    keys = sorted({key for row in records for key in row.keys()})
    out: Dict[str, Dict[str, float]] = {}

    for key in keys:
        vals = np.asarray([row[key] for row in records if key in row], dtype=np.float64)
        if vals.size == 0:
            continue

        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "max": float(np.max(vals)),
        }

    return out


def print_summary_table(summary: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 188)
    print("Diff-Drive UGV / Jetbot Task4 TRUE skrl PPO Model Test Summary")
    print("=" * 188)
    print(
        f"{'metric':<86} | {'mean':>12} | {'std':>12} | {'min':>12} | "
        f"{'p25':>12} | {'p50':>12} | {'p75':>12} | {'max':>12}"
    )
    print("-" * 188)

    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<86} | "
            f"{row['mean']:>12.6f} | "
            f"{row['std']:>12.6f} | "
            f"{row['min']:>12.6f} | "
            f"{row['p25']:>12.6f} | "
            f"{row['p50']:>12.6f} | "
            f"{row['p75']:>12.6f} | "
            f"{row['max']:>12.6f}"
        )

    print("=" * 188 + "\n")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    cfg = Task4Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.device)
    cfg.seed = int(args_cli.seed)
    cfg.max_episode_length_s = float(args_cli.max_episode_length_s)
    cfg.max_wheel_speed = float(args_cli.max_wheel_speed)
    cfg.reverse_speed_fraction = float(args_cli.reverse_speed_fraction)
    cfg.left_wheel_sign = float(args_cli.left_wheel_sign)
    cfg.right_wheel_sign = float(args_cli.right_wheel_sign)
    cfg.print_debug_info = False

    if bool(args_cli.disable_physical_asset_teleport):
        cfg.world_cfg.enable_physical_asset_teleport = False

    eval_curriculum_info = Task4EvalCurriculum.apply(
        cfg,
        start_k=float(args_cli.start_k),
        fixed_stage=int(args_cli.stage),
    )

    cfg.validate()

    base_env = DiffDriveTask4Env(cfg)
    env = DiffDriveTask4EvalWrapper(base_env)

    obs_dict, _ = env.reset(seed=int(args_cli.seed), options={"stage": int(eval_curriculum_info["Curriculum_Stage"])})

    ckpt_path = resolve_checkpoint(args_cli.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    policy, actor_obs_norm, trained_agent_steps, metadata, ckpt_curriculum_info = load_policy_checkpoint(ckpt_path, env)

    print("\n" + "=" * 150)
    print("Diff-Drive UGV / Jetbot Task4 TRUE skrl PPO model test started")
    print("=" * 150)
    print(f"checkpoint          : {ckpt_path}")
    print(f"trained_agent_steps : {trained_agent_steps:,}")
    print(f"physical_num_envs   : {base_env.num_envs}")
    print(f"skrl_agent_envs     : {env.num_envs}")
    print(f"num_agents          : {base_env.num_agents}")
    print(f"steps               : {args_cli.steps}")
    print(f"start_k             : {args_cli.start_k}")
    print(f"eval_stage          : {eval_curriculum_info.get('Curriculum_Stage', -1)}")
    print(f"actor_obs_dim       : {env.observation_space.shape[0]}")
    print(f"critic_obs_dim      : {env.state_space.shape[0]}  # 96 world state + 4 agent id")
    print(f"world_priv_dim      : {base_env.world_priv_dim}")
    print(f"single_obs_dim      : {cfg.single_actor_obs_dim}")
    print(f"frame_stack         : {cfg.frame_stack}")
    print(f"lidar_pool_bins     : {cfg.world_cfg.lidar_pool_bins}")
    print(f"risk_feature_dim    : {base_env.world.risk_feature_dim()}")
    print(f"action_dim          : {env.action_space.shape[0]}")
    print(f"max_wheel_speed     : {cfg.max_wheel_speed}")
    print(f"slow_action_scale   : {args_cli.slow_action_scale}")
    print(f"physical teleport   : {bool(cfg.world_cfg.enable_physical_asset_teleport)}")
    print(f"device              : {base_env.device}")
    print(f"visualize           : {bool(args_cli.visualize)}")
    print("algorithm           : skrl PPO")
    print("checkpoint metadata : uses_skrl=True, ctde_mappo_style=True, shared_actor=True, centralized_critic=True")
    print("eval curriculum     :", eval_curriculum_info)
    print("ckpt curriculum     :", ckpt_curriculum_info)
    print("=" * 150 + "\n")

    records: List[Dict[str, float]] = []
    total_terminated = 0
    total_truncated = 0

    start_time = time.time()

    try:
        with tqdm(
            total=int(args_cli.steps),
            desc="Diff-Drive Task4 skrl Model Test",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as pbar:
            for step in range(int(args_cli.steps)):
                with torch.no_grad():
                    actor_obs = obs_dict["policy"]
                    actor_obs_n = normalize_with_saved_obs_norm(actor_obs, actor_obs_norm)
                    actions = policy.act_deterministic_direct(actor_obs_n)
                    actions = torch.clamp(actions * float(args_cli.slow_action_scale), -1.0, 1.0)

                if step < 3:
                    print(
                        f"[DEBUG][eval step {step}] action_mean={actions.mean().item():+.6f}, "
                        f"action_abs_max={actions.abs().max().item():.6f}",
                        flush=True,
                    )

                obs_dict, rewards, terminated, truncated, info = env.step(actions)

                total_terminated += int(terminated.sum().item())
                total_truncated += int(truncated.sum().item())

                if bool(args_cli.visualize):
                    try:
                        time.sleep(float(cfg.policy_dt))
                    except Exception:
                        pass

                if step % max(int(args_cli.print_interval), 1) == 0 or step == int(args_cli.steps) - 1:
                    flat = flat_dict(info)
                    row = {
                        "test/reward_mean": float(rewards.detach().float().mean().cpu().item()),
                        "test/reward_min": float(rewards.detach().float().min().cpu().item()),
                        "test/reward_max": float(rewards.detach().float().max().cpu().item()),
                        "test/terminated_rate": float(terminated.float().mean().cpu().item()),
                        "test/truncated_rate": float(truncated.float().mean().cpu().item()),
                    }
                    row.update(flat)
                    records.append(row)

                    pbar.set_postfix(
                        {
                            "rew": f"{row['test/reward_mean']:+.3f}",
                            "dist": f"{flat.get('telemetry/Center_Goal_Dist', 0.0):.2f}",
                            "prog": f"{flat.get('telemetry/Progress', 0.0):+.4f}",
                            "slot": f"{flat.get('telemetry/Mean_Slot_Error', 0.0):.2f}",
                            "pair": f"{flat.get('telemetry/Min_Pair_Dist', 0.0):.2f}",
                            "gate": f"{flat.get('telemetry/Gate_Active', 0.0):.0f}",
                            "succ": f"{flat.get('events/Success_Rate', 0.0):.3f}",
                            "crash": f"{flat.get('events/Crash_Rate', 0.0):.3f}",
                        }
                    )

                    if bool(args_cli.visualize):
                        sys.stdout.write(
                            f"\r🚗🚗🚗🚗 "
                            f"Dist={flat.get('telemetry/Center_Goal_Dist', 0.0):.3f} | "
                            f"Prog={flat.get('telemetry/Progress', 0.0):+.4f} | "
                            f"GoalV={flat.get('telemetry/Goal_Aligned_Center_Speed', 0.0):+.3f} | "
                            f"Slot={flat.get('telemetry/Mean_Slot_Error', 0.0):.3f} | "
                            f"MinPair={flat.get('telemetry/Min_Pair_Dist', 0.0):.3f} | "
                            f"Gate={flat.get('telemetry/Gate_Active', 0.0):.0f} | "
                            f"ObsRisk={flat.get('telemetry/Risk_Obstacle', 0.0):.3f} | "
                            f"GateRisk={flat.get('telemetry/Risk_Gate', 0.0):.3f} | "
                            f"PairRisk={flat.get('telemetry/Risk_Pair', 0.0):.3f} | "
                            f"R={row['test/reward_mean']:+.3f} | "
                            f"Succ={flat.get('events/Success_Rate', 0.0):.3f} | "
                            f"Crash={flat.get('events/Crash_Rate', 0.0):.3f} | "
                            f"Timeout={flat.get('events/Timeout_Rate', 0.0):.3f}"
                        )
                        sys.stdout.flush()

                pbar.update(1)

                if bool(args_cli.visualize) and not simulation_app.is_running():
                    print("\n[INFO] Isaac Sim window closed.")
                    break

        elapsed = time.time() - start_time
        physical_env_steps = int(args_cli.steps) * int(base_env.num_envs)
        agent_steps = physical_env_steps * int(base_env.num_agents)
        fps = agent_steps / max(elapsed, 1.0e-6)

        print("\n✅ Diff-Drive Task4 TRUE skrl PPO model test rollout finished")
        print(f"  physical env steps : {physical_env_steps:,}")
        print(f"  agent steps        : {agent_steps:,}")
        print(f"  fps agent steps    : {fps:,.2f}")
        print(f"  total terminated   : {total_terminated:,}")
        print(f"  total truncated    : {total_truncated:,}")

        print_summary_table(summarize(records))

        print("Diff-Drive Task4 model test checklist:")
        print("1. checkpoint metadata 必须标记 uses_skrl=True。")
        print("2. checkpoint metadata 必须标记 ctde_mappo_style=True。")
        print("3. actor obs 必须为 624 维。")
        print("4. critic obs 必须为 100 维，也就是 96 world state + 4 agent id。")
        print("5. action 必须为 2 维，最终还原到 [num_envs, 4, 2]。")
        print("6. smoke checkpoint 效果差是正常的，先看加载、rollout 和无 NaN/Inf。")
        print("7. 正式效果重点看 Center_Goal_Dist、Progress、Mean_Slot_Error、Min_Pair_Dist、Success_Rate、Crash_Rate。")

    finally:
        try:
            env.close()
        except Exception:
            pass

        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

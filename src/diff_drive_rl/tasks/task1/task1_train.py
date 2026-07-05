from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logging.getLogger("isaaclab.assets.articulation").setLevel(logging.ERROR)
logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Diff-Drive UGV / Jetbot Task1 with TRUE skrl PPO")

# Runtime
parser.add_argument("--total-env-steps", type=int, default=200_000_000)
parser.add_argument("--save-freq-env-steps", type=int, default=5_000_000)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default="", help="Optional skrl checkpoint or final_checkpoint directory")
parser.add_argument(
    "--start-env-steps",
    type=int,
    default=0,
    help="Logical env-step offset used for curriculum/progress when resuming from a selected checkpoint",
)

# Env convenience
parser.add_argument("--max-episode-length-s", type=float, default=24.0)
parser.add_argument("--num-waypoints", type=int, default=3)
parser.add_argument("--disable-curriculum", action="store_true")
parser.add_argument("--curriculum-interval", type=int, default=262_144)

# PPO
parser.add_argument("--rollouts", type=int, default=64)
parser.add_argument("--learning-epochs", type=int, default=4)
parser.add_argument("--mini-batches", type=int, default=4)

parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--min-lr", type=float, default=5e-5)
parser.add_argument("--max-lr", type=float, default=6e-4)
# Late-stage refinement learning rates. Continuing PPO updates with the same
# learning rate after a good Stage2/2.5 policy can destroy it, so the trainer
# lowers LR once multi-waypoint stages are reached.
parser.add_argument("--stage2-lr", type=float, default=2.0e-5)
parser.add_argument("--stage25-lr", type=float, default=1.0e-5)
parser.add_argument("--stage3-lr", type=float, default=7.5e-6)

parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-range", type=float, default=0.20)
parser.add_argument("--value-clip", type=float, default=0.20)
parser.add_argument("--entropy-coef", type=float, default=0.004)
parser.add_argument("--value-coef", type=float, default=1.0)
parser.add_argument("--grad-clip", type=float, default=0.5)

# Policy distribution
parser.add_argument("--init-log-std", type=float, default=-0.60)
parser.add_argument("--min-log-std", type=float, default=-4.0)
parser.add_argument("--max-log-std", type=float, default=0.30)

# KL
parser.add_argument("--target-kl", type=float, default=0.020)
parser.add_argument("--hard-kl-stop", type=float, default=0.120)

# Logging
parser.add_argument("--log-root", type=str, default=str(PROJECT_ROOT / "logs" / "task1"))
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--summary-interval", type=int, default=1)
parser.add_argument("--skrl-write-interval", type=int, default=1_000_000)
parser.add_argument("--skrl-checkpoint-interval", type=int, default=0)
# Best-checkpoint and anti-regression guard. The final checkpoint of a long PPO
# run is not necessarily the best checkpoint. By default the guard only warns
# and keeps training; pass --stop-on-collapse to stop automatically.
parser.add_argument("--best-min-finish-rate", type=float, default=0.60)
parser.add_argument("--collapse-finish-rate", type=float, default=0.20)
parser.add_argument("--collapse-timeout-rate", type=float, default=0.80)
parser.add_argument("--collapse-patience", type=int, default=3)
parser.add_argument("--stop-on-collapse", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from diff_drive_rl.tasks.task1.task1_config import Task1Config
from diff_drive_rl.tasks.task1.task1_env import DiffDriveTask1Env
from diff_drive_rl.export.policy_io import build_policy_io, write_policy_io

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import StepTrainer
from skrl.utils import set_seed

try:
    from skrl.agents.torch.ppo import PPO, PPO_CFG
except ImportError:
    from skrl.agents.torch.ppo import PPO
    from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

try:
    from skrl.resources.schedulers.torch import KLAdaptiveLR
except Exception:
    KLAdaptiveLR = None


# ======================================================================
# Utility
# ======================================================================

def to_float(x: Any):
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        if isinstance(x, (list, tuple)) and len(x) > 0:
            return float(np.mean(x))
    except Exception:
        return None
    return None


def flat_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}

    for key, value in (d or {}).items():
        name = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flat_dict(value, name))
        else:
            val = to_float(value)
            if val is not None and math.isfinite(val):
                out[name] = val

    return out


def write_scalars(writer, data: Dict[str, Any], step: int, prefix: str) -> None:
    if writer is None:
        return

    for key, value in (data or {}).items():
        val = to_float(value)
        if val is None:
            continue
        try:
            writer.add_scalar(f"{prefix}/{key}".replace("//", "/"), val, step)
        except Exception:
            pass


def make_table(title: str, data: Dict[str, Any], width: int = 118) -> str:
    lines = ["-" * width, f"| {title:<{width - 4}} |", "-" * width]

    if not data:
        lines += [f"| {'<empty>':<{width - 4}} |", "-" * width]
        return "\n".join(lines)

    for key in sorted(data.keys()):
        value = data[key]
        key_str = (key[:74] + "...") if len(key) > 77 else key

        if isinstance(value, float):
            value_str = f"{value:.6e}" if abs(value) > 1e4 or 0 < abs(value) < 1e-3 else f"{value:.6f}"
        else:
            value_str = str(value)

        value_str = (value_str[:36] + "...") if len(value_str) > 39 else value_str
        lines.append(f"| {key_str:<77} | {value_str:>{width - 84}} |")

    lines.append("-" * width)
    return "\n".join(lines)


def tracking_mean(agent) -> Dict[str, float]:
    out: Dict[str, float] = {}

    for key, value in getattr(agent, "tracking_data", {}).items():
        if value is None:
            continue
        try:
            if len(value) == 0:
                continue
        except Exception:
            pass

        try:
            arr = np.asarray(value, dtype=np.float64)
            if key.endswith("(min)"):
                out[key] = float(np.min(arr))
            elif key.endswith("(max)"):
                out[key] = float(np.max(arr))
            else:
                out[key] = float(np.mean(arr))
        except Exception:
            val = to_float(value)
            if val is not None:
                out[key] = val

    return out


def current_lr(agent) -> float:
    for obj in [getattr(agent, "optimizer", None), getattr(getattr(agent, "scheduler", None), "optimizer", None)]:
        try:
            if obj is not None:
                return float(obj.param_groups[0]["lr"])
        except Exception:
            pass
    return float("nan")


def set_agent_lr(agent, lr: float) -> None:
    """Set PPO optimizer LR defensively, independent of skrl version."""

    lr = float(lr)
    seen = set()
    for obj in [getattr(agent, "optimizer", None), getattr(getattr(agent, "scheduler", None), "optimizer", None)]:
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        try:
            for group in obj.param_groups:
                group["lr"] = lr
        except Exception:
            pass


def lr_for_stage(stage: float) -> float:
    if stage >= 3.0:
        return float(args_cli.stage3_lr)
    if stage >= 2.5:
        return float(args_cli.stage25_lr)
    if stage >= 2.0:
        return float(args_cli.stage2_lr)
    return float(args_cli.lr)


def make_run_name() -> str:
    run_name = args_cli.run_name.strip()
    if run_name:
        return run_name
    return f"diff_drive_task1_skrl_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def apply_task1_curriculum(env: DiffDriveTask1Env, env_steps: int) -> float:
    """Apply the Task1 navigation curriculum in-place.

    Final open-source curriculum for the forward-only differential-drive action
    interface. The policy action is [forward_throttle, turn], so the robot can
    pivot toward side / rear waypoints without learning reverse driving. The
    schedule is intentionally gradual: single-waypoint tracking is kept long
    enough to stabilize heading control, then two-waypoint tracking is introduced
    with generous episode budgets before the full three-waypoint task.
    """
    total = max(int(args_cli.total_env_steps), 1)
    ratio = float(env_steps) / float(total)

    def set_common_speed_and_stuck(
        *,
        min_goal_speed: float,
        target_goal_speed: float,
        stuck_progress_threshold: float,
        stuck_speed_threshold: float,
        stuck_after_steps: int,
        hard_stuck_after_steps: int,
    ) -> None:
        env.cfg.min_goal_speed = float(min_goal_speed)
        env.cfg.target_goal_speed = float(target_goal_speed)
        env.cfg.stuck_progress_threshold = float(stuck_progress_threshold)
        env.cfg.stuck_speed_threshold = float(stuck_speed_threshold)
        env.cfg.stuck_after_steps = int(stuck_after_steps)
        env.cfg.hard_stuck_after_steps = int(hard_stuck_after_steps)

    def set_forward_cone_waypoints(*, front_angle_deg: float, total_path_length: float) -> None:
        env.cfg.forward_cone_waypoint_sampling = True
        env.cfg.waypoint_front_angle_deg = float(front_angle_deg)
        env.cfg.waypoint_total_path_length = float(total_path_length)

    if bool(args_cli.disable_curriculum):
        stage = 99.0
        env.cfg.num_waypoints = int(args_cli.num_waypoints)
        env.cfg.waypoint_min_radius = 0.60
        env.cfg.waypoint_max_radius = 1.40
        set_forward_cone_waypoints(front_angle_deg=60.0, total_path_length=max(3.8, int(args_cli.num_waypoints) * 1.25))
        env.cfg.reach_threshold = 0.25
        env.cfg.max_episode_length_s = max(float(args_cli.max_episode_length_s), 36.0)
        set_common_speed_and_stuck(
            min_goal_speed=0.035,
            target_goal_speed=0.30,
            stuck_progress_threshold=0.0006,
            stuck_speed_threshold=0.030,
            stuck_after_steps=80,
            hard_stuck_after_steps=900,
        )
        env.cfg.validate()
        return stage

    if ratio < 0.15:
        stage = 0.0
        env.cfg.num_waypoints = 1
        env.cfg.waypoint_min_radius = 0.40
        env.cfg.waypoint_max_radius = 0.90
        set_forward_cone_waypoints(front_angle_deg=30.0, total_path_length=0.90)
        env.cfg.reach_threshold = 0.32
        env.cfg.max_episode_length_s = 14.0
        set_common_speed_and_stuck(
            min_goal_speed=0.030,
            target_goal_speed=0.25,
            stuck_progress_threshold=0.0004,
            stuck_speed_threshold=0.025,
            stuck_after_steps=70,
            hard_stuck_after_steps=500,
        )
    elif ratio < 0.40:
        stage = 1.0
        env.cfg.num_waypoints = 1
        env.cfg.waypoint_min_radius = 0.50
        env.cfg.waypoint_max_radius = 1.20
        set_forward_cone_waypoints(front_angle_deg=45.0, total_path_length=1.20)
        env.cfg.reach_threshold = 0.30
        env.cfg.max_episode_length_s = 18.0
        set_common_speed_and_stuck(
            min_goal_speed=0.035,
            target_goal_speed=0.30,
            stuck_progress_threshold=0.0005,
            stuck_speed_threshold=0.028,
            stuck_after_steps=75,
            hard_stuck_after_steps=650,
        )
    elif ratio < 0.65:
        stage = 2.0
        env.cfg.num_waypoints = min(2, int(args_cli.num_waypoints))
        env.cfg.waypoint_min_radius = 0.50
        env.cfg.waypoint_max_radius = 1.25
        set_forward_cone_waypoints(front_angle_deg=60.0, total_path_length=2.20)
        env.cfg.reach_threshold = 0.28
        env.cfg.max_episode_length_s = 28.0
        set_common_speed_and_stuck(
            min_goal_speed=0.035,
            target_goal_speed=0.30,
            stuck_progress_threshold=0.0006,
            stuck_speed_threshold=0.030,
            stuck_after_steps=80,
            hard_stuck_after_steps=800,
        )
    elif ratio < 0.82:
        stage = 2.5
        env.cfg.num_waypoints = min(2, int(args_cli.num_waypoints))
        env.cfg.waypoint_min_radius = 0.60
        env.cfg.waypoint_max_radius = 1.40
        set_forward_cone_waypoints(front_angle_deg=60.0, total_path_length=2.60)
        env.cfg.reach_threshold = 0.26
        env.cfg.max_episode_length_s = 32.0
        set_common_speed_and_stuck(
            min_goal_speed=0.035,
            target_goal_speed=0.30,
            stuck_progress_threshold=0.0006,
            stuck_speed_threshold=0.030,
            stuck_after_steps=80,
            hard_stuck_after_steps=900,
        )
    else:
        stage = 3.0
        env.cfg.num_waypoints = int(args_cli.num_waypoints)
        env.cfg.waypoint_min_radius = 0.60
        env.cfg.waypoint_max_radius = 1.40
        set_forward_cone_waypoints(front_angle_deg=60.0, total_path_length=max(3.8, int(args_cli.num_waypoints) * 1.25))
        env.cfg.reach_threshold = 0.25
        env.cfg.max_episode_length_s = max(float(args_cli.max_episode_length_s), 40.0)
        set_common_speed_and_stuck(
            min_goal_speed=0.035,
            target_goal_speed=0.30,
            stuck_progress_threshold=0.0007,
            stuck_speed_threshold=0.030,
            stuck_after_steps=90,
            hard_stuck_after_steps=1000,
        )

    env.cfg.validate()
    return stage


# ======================================================================
# skrl wrapper
# ======================================================================

class DiffDriveTask1SkrlWrapper(gym.Env):
    """Task1 skrl wrapper.

    The underlying environment already outputs the 3-frame stacked CoreNav-v1
    observation. For skrl asymmetric-compatible format, policy and critic both
    receive the same 42-D tensor in Task1.
    """

    def __init__(self, env: DiffDriveTask1Env):
        super().__init__()

        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = env.device

        self.obs_dim = int(env.num_observations)
        self.action_dim = int(env.num_actions)

        self.policy_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.critic_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": self.policy_space,
                "critic": self.critic_space,
            }
        )

        self.observation_space = self.policy_space
        self.state_space = self.critic_space
        self.action_space = env.action_space
        self.single_action_space = env.action_space

        self.last_obs = torch.zeros((self.num_envs, self.obs_dim), dtype=torch.float32, device=self.device)
        self.last_info: Dict[str, Any] = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0
        self.global_env_steps = 0
        self.curriculum_stage = 0.0

    @property
    def unwrapped(self):
        return self

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs.clone()
        self.last_info = info or {}
        return {"policy": obs.clone(), "critic": obs.clone()}, self.last_info

    def step(self, actions: torch.Tensor):
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        self.last_obs = obs.clone()
        self.last_info = info or {}
        self.last_info.setdefault("curriculum", {})
        self.last_info["curriculum"].update(
            {
                "Stage": float(self.curriculum_stage),
                "Num_Waypoints": float(self.env.cfg.num_waypoints),
                "Waypoint_Min_Radius": float(self.env.cfg.waypoint_min_radius),
                "Waypoint_Max_Radius": float(self.env.cfg.waypoint_max_radius),
                "Waypoint_Front_Angle_Deg": float(getattr(self.env.cfg, "waypoint_front_angle_deg", 60.0)),
                "Waypoint_Total_Path_Length": float(getattr(self.env.cfg, "waypoint_total_path_length", 0.0)),
                "Reach_Threshold": float(self.env.cfg.reach_threshold),
                "Max_Episode_Length_S": float(self.env.cfg.max_episode_length_s),
                "Min_Goal_Speed": float(self.env.cfg.min_goal_speed),
                "Target_Goal_Speed": float(self.env.cfg.target_goal_speed),
                "Stuck_Progress_Threshold": float(self.env.cfg.stuck_progress_threshold),
                "Stuck_Speed_Threshold": float(self.env.cfg.stuck_speed_threshold),
                "Hard_Stuck_After_Steps": float(self.env.cfg.hard_stuck_after_steps),
            }
        )
        self.last_reward_mean = to_float(rewards) or 0.0
        self.last_done_count = int((terminated | truncated).sum().detach().cpu().item())
        self.global_env_steps += self.num_envs

        return {"policy": obs.clone(), "critic": obs.clone()}, rewards, terminated, truncated, self.last_info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# ======================================================================
# Models
# ======================================================================

class CoreNavEncoder(nn.Module):
    """Shared CoreNav-v1 encoder.

    Task1 uses only the 42-D stacked CoreNav input. Downstream Task2 / Task3 can
    load this module directly and attach task-specific extra encoders.
    """

    def __init__(self, input_dim: int = 42, latent_dim: int = 128):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ELU(),
            nn.Linear(128, self.latent_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IdentityExtraEncoder(nn.Module):
    """Task-extra encoder placeholder for ModularActor-v1.

    Task1 has no task-specific actor extras, so this encoder returns an empty
    latent tensor while keeping the same code path as future tasks.
    """

    def __init__(self, output_dim: int = 0):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros((x.shape[0], self.output_dim))


class DiffDriveActor(GaussianMixin, Model):
    """Task1 ModularActor-v1 policy.

    Input layout:
        observations[:, :42] = CoreNav-v1 stacked input
        observations[:, 42:] = task extra input, empty for Task1

    The saved checkpoint exposes ``core_encoder`` as a stable module name so
    Task2 / Task3 can load the reusable local-navigation representation.
    """

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        init_log_std: float = -0.60,
        min_log_std: float = -4.0,
        max_log_std: float = 0.30,
        core_obs_dim: int = 42,
        core_latent_dim: int = 128,
        extra_latent_dim: int = 0,
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

        self.obs_dim = int(observation_space.shape[0])
        self.core_obs_dim = int(core_obs_dim)
        self.extra_obs_dim = max(0, self.obs_dim - self.core_obs_dim)
        self.core_latent_dim = int(core_latent_dim)
        self.extra_latent_dim = int(extra_latent_dim if self.extra_obs_dim > 0 else 0)

        if self.obs_dim < self.core_obs_dim:
            raise ValueError(f"obs_dim={self.obs_dim} is smaller than core_obs_dim={self.core_obs_dim}")

        self.core_encoder = CoreNavEncoder(input_dim=self.core_obs_dim, latent_dim=self.core_latent_dim)
        if self.extra_obs_dim > 0:
            self.extra_encoder = nn.Sequential(
                nn.Linear(self.extra_obs_dim, 128),
                nn.ELU(),
                nn.Linear(128, self.extra_latent_dim),
                nn.ELU(),
            )
        else:
            self.extra_encoder = IdentityExtraEncoder(output_dim=0)

        fusion_dim = self.core_latent_dim + self.extra_latent_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.policy_head = nn.Linear(64, action_space.shape[0])

        self.log_std_parameter = nn.Parameter(torch.full((action_space.shape[0],), float(init_log_std)))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)

    def _encode(self, states: torch.Tensor) -> torch.Tensor:
        core_obs = states[:, : self.core_obs_dim]
        core_latent = self.core_encoder(core_obs)

        if self.extra_obs_dim > 0:
            extra_obs = states[:, self.core_obs_dim :]
            extra_latent = self.extra_encoder(extra_obs)
            latent = torch.cat([core_latent, extra_latent], dim=-1)
        else:
            latent = core_latent
        return self.fusion_head(latent)

    def compute(self, inputs, role):
        states = inputs.get("observations", inputs.get("states"))
        actions = self.policy_head(self._encode(states))
        return actions, {"log_std": self.log_std_parameter}


class DiffDriveCritic(DeterministicMixin, Model):
    """Task1 ModularActor-v1 value model.

    The critic mirrors the actor protocol for Task1. A future privileged critic
    can keep the same CoreNav prefix and append critic-only extras.
    """

    def __init__(self, observation_space, state_space, action_space, device, core_obs_dim: int = 42):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)

        self.state_dim = int(state_space.shape[0])
        self.core_obs_dim = int(core_obs_dim)
        self.extra_state_dim = max(0, self.state_dim - self.core_obs_dim)
        if self.state_dim < self.core_obs_dim:
            raise ValueError(f"state_dim={self.state_dim} is smaller than core_obs_dim={self.core_obs_dim}")

        self.core_encoder = CoreNavEncoder(input_dim=self.core_obs_dim, latent_dim=128)
        if self.extra_state_dim > 0:
            self.extra_encoder = nn.Sequential(
                nn.Linear(self.extra_state_dim, 128),
                nn.ELU(),
                nn.Linear(128, 32),
                nn.ELU(),
            )
            fusion_dim = 160
        else:
            self.extra_encoder = IdentityExtraEncoder(output_dim=0)
            fusion_dim = 128

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.value_head = nn.Linear(64, 1)
        self.apply(DiffDriveActor._init_weights)

    def _encode(self, states: torch.Tensor) -> torch.Tensor:
        core_state = states[:, : self.core_obs_dim]
        core_latent = self.core_encoder(core_state)
        if self.extra_state_dim > 0:
            extra_state = states[:, self.core_obs_dim :]
            extra_latent = self.extra_encoder(extra_state)
            latent = torch.cat([core_latent, extra_latent], dim=-1)
        else:
            latent = core_latent
        return self.fusion_head(latent)

    def compute(self, inputs, role):
        states = inputs.get("states", None)
        if states is None:
            states = inputs.get("observations", None)
        if states is None:
            raise RuntimeError("Critic received no states / observations.")
        return self.value_head(self._encode(states)), {}


# ======================================================================
# skrl config / checkpoint helpers
# ======================================================================

def _base_ppo_cfg_dict():
    cfg = PPO_CFG()
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return cfg.copy()


def _set_if_supported(cfg: dict, requested: dict) -> None:
    skipped = []

    for key, value in requested.items():
        if key in cfg:
            cfg[key] = value
        else:
            skipped.append(key)

    if skipped:
        print(f"[WARN] 当前 skrl.PPO_CFG 不支持这些字段，已跳过: {skipped}")


def build_skrl_cfg(env, log_dir: str, run_name: str):
    cfg = _base_ppo_cfg_dict()

    requested = {
        "rollouts": int(args_cli.rollouts),
        "learning_epochs": int(args_cli.learning_epochs),
        "mini_batches": int(args_cli.mini_batches),
        "discount_factor": float(args_cli.gamma),
        "gae_lambda": float(args_cli.gae_lambda),
        "learning_rate": float(args_cli.lr),
        "grad_norm_clip": float(args_cli.grad_clip),
        "ratio_clip": float(args_cli.clip_range),
        "value_clip": float(args_cli.value_clip),
        "entropy_loss_scale": float(args_cli.entropy_coef),
        "value_loss_scale": float(args_cli.value_coef),
        "kl_threshold": float(args_cli.hard_kl_stop),
        "observation_preprocessor": RunningStandardScaler,
        "observation_preprocessor_kwargs": {
            "size": env.observation_space,
            "device": env.device,
        },
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {
            "size": env.state_space,
            "device": env.device,
        },
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {
            "size": 1,
            "device": env.device,
        },
    }

    if KLAdaptiveLR is not None:
        requested["learning_rate_scheduler"] = KLAdaptiveLR
        requested["learning_rate_scheduler_kwargs"] = {
            "kl_threshold": float(args_cli.target_kl),
            "min_lr": float(args_cli.min_lr),
            "max_lr": float(args_cli.max_lr),
        }

    _set_if_supported(cfg, requested)

    cfg.setdefault("experiment", {})
    cfg["experiment"].update(
        {
            "directory": log_dir,
            "experiment_name": run_name,
            "write_interval": int(getattr(args_cli, "skrl_write_interval", 1_000_000)),
            "checkpoint_interval": int(getattr(args_cli, "skrl_checkpoint_interval", 0)),
            "store_separately": True,
            "wandb": False,
        }
    )

    return cfg


def resolve_resume_checkpoint(path: str) -> str:
    if not path:
        return ""

    p = Path(path).expanduser().resolve()

    if p.is_file():
        return str(p)

    if p.is_dir():
        candidates = [
            p / "diff_drive_task1_skrl_agent.pt",
            p / "agent.pt",
            p / "final_checkpoint" / "diff_drive_task1_skrl_agent.pt",
            p / "final_checkpoint" / "agent.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    return str(p)


def _find_norm_tensors_from_state_dict(state_dict: Dict[str, Any], obs_dim: int):
    mean = None
    var = None

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if value.numel() != obs_dim:
            continue

        lower = key.lower()
        if "mean" in lower:
            mean = value.detach().cpu()
        if "var" in lower or "variance" in lower:
            var = value.detach().cpu()

    if mean is not None and var is not None:
        return {"mean": mean, "var": var, "clip": 10.0}

    return None


def extract_norm(agent, attr_names, dim: int):
    for attr_name in attr_names:
        obj = getattr(agent, attr_name, None)
        if obj is None:
            continue

        try:
            state = obj.state_dict()
        except Exception:
            continue

        out = _find_norm_tensors_from_state_dict(state, dim)
        if out is not None:
            out["source_attr"] = attr_name
            return out

    return None


def save_project_checkpoint(
    directory: str,
    agent: PPO,
    models: Dict[str, Model],
    env_cfg: Task1Config,
    env: DiffDriveTask1SkrlWrapper,
    env_steps: int,
    args,
) -> None:
    os.makedirs(directory, exist_ok=True)

    skrl_agent_path = os.path.join(directory, "diff_drive_task1_skrl_agent.pt")
    eval_model_path = os.path.join(directory, "diff_drive_task1_model.pt")

    try:
        agent.save(skrl_agent_path)
        agent.save(os.path.join(directory, "agent.pt"))
    except Exception as exc:
        print(f"[WARN] agent.save failed: {type(exc).__name__}: {exc}")

    actor_norm = extract_norm(
        agent,
        ["_observation_preprocessor", "_state_preprocessor", "observation_preprocessor", "state_preprocessor"],
        env.obs_dim,
    )
    critic_norm = extract_norm(
        agent,
        ["_state_preprocessor", "state_preprocessor"],
        env.obs_dim,
    )

    torch.save(
        {
            "policy": models["policy"].state_dict(),
            "value": models["value"].state_dict(),
            "policy_core_encoder": models["policy"].core_encoder.state_dict(),
            "value_core_encoder": models["value"].core_encoder.state_dict(),
            "actor_obs_norm": actor_norm,
            "critic_obs_norm": critic_norm,
            "env_steps": int(env_steps),
            "args": vars(args),
            "metadata": {
                "robot": "Jetbot / two-wheel differential-drive UGV",
                "task": "task1_multi_waypoint_navigation",
                "algorithm": "skrl_PPO",
                "uses_skrl": True,
                "asymmetric_actor_critic": False,
                "policy_input": "CoreNav-v1_3_frame_stack",
                "critic_input": "same_as_policy",
                "obs_protocol": str(env_cfg.obs_protocol),
                "action_protocol": str(env_cfg.action_protocol),
                "model_protocol": str(env_cfg.model_protocol),
                "single_obs_dim": int(env_cfg.single_obs_dim),
                "core_single_obs_dim": int(env_cfg.core_single_obs_dim),
                "task_extra_single_obs_dim": int(env_cfg.task_extra_single_obs_dim),
                "frame_stack": int(env_cfg.frame_stack),
                "stacked_core_dim": int(env_cfg.frame_stack * env_cfg.core_single_obs_dim),
                "actor_obs_dim": int(env_cfg.num_observations),
                "critic_obs_dim": int(env_cfg.num_observations),
                "action_dim": int(env_cfg.num_actions),
                "num_waypoints": int(env_cfg.num_waypoints),
                "curriculum_stage": float(getattr(env, "curriculum_stage", 0.0)),
                "control": "forward_throttle_plus_turn_to_left_right_wheel_velocity",
                "transfer": "Load policy.core_encoder into downstream ModularActor-v1 tasks; task_extra_encoder and heads may be reinitialized.",
                "min_forward_action": float(env_cfg.min_forward_action),
                "max_forward_action": float(env_cfg.max_forward_action),
                "note": "TRUE skrl PPO checkpoint. Evaluation uses deterministic policy forward, not agent.act.",
            },
        },
        eval_model_path,
    )

    torch.save(
        {
            "stage": "task1_skrl_ppo",
            "algorithm": "skrl_PPO",
            "uses_skrl": True,
            "env_steps": int(env_steps),
            "num_envs": int(env_cfg.num_envs),
            "curriculum_stage": float(getattr(env, "curriculum_stage", 0.0)),
            "env_cfg": {
                "obs_protocol": str(env_cfg.obs_protocol),
                "action_protocol": str(env_cfg.action_protocol),
                "model_protocol": str(env_cfg.model_protocol),
                "num_observations": int(env_cfg.num_observations),
                "single_obs_dim": int(env_cfg.single_obs_dim),
                "core_single_obs_dim": int(env_cfg.core_single_obs_dim),
                "task_extra_single_obs_dim": int(env_cfg.task_extra_single_obs_dim),
                "stacked_core_dim": int(env_cfg.frame_stack * env_cfg.core_single_obs_dim),
                "frame_stack": int(env_cfg.frame_stack),
                "num_actions": int(env_cfg.num_actions),
                "num_waypoints": int(env_cfg.num_waypoints),
                "max_episode_length": int(env_cfg.max_episode_length),
                "min_forward_action": float(env_cfg.min_forward_action),
                "max_forward_action": float(env_cfg.max_forward_action),
            },
        },
        os.path.join(directory, "task1_train_metadata.pt"),
    )

    policy_io = build_policy_io(
        task_name="task1_multi_waypoint_navigation",
        actor_obs_dim=int(env_cfg.num_observations),
        critic_obs_dim=int(env_cfg.num_observations),
        action_dim=int(env_cfg.num_actions),
        action_protocol=str(env_cfg.action_protocol),
        observation_protocol=str(env_cfg.obs_protocol),
        model_protocol=str(env_cfg.model_protocol),
        control_dt=float(env_cfg.policy_dt),
        frame_stack=int(env_cfg.frame_stack),
        normalizer_source="actor_obs_norm",
        onnx_export_target="actor_only",
        extra={
            "env_steps": int(env_steps),
            "single_obs_dim": int(env_cfg.single_obs_dim),
            "core_single_obs_dim": int(env_cfg.core_single_obs_dim),
            "task_extra_single_obs_dim": int(env_cfg.task_extra_single_obs_dim),
            "wheel_speed_scale": float(env_cfg.wheel_speed_scale),
            "min_forward_action": float(env_cfg.min_forward_action),
            "max_forward_action": float(env_cfg.max_forward_action),
            "forward_curve_power": float(getattr(env_cfg, "forward_curve_power", 1.0)),
            "turn_scale_norm": float(env_cfg.turn_scale_norm),
            "asymmetric_actor_critic": False,
        },
    )
    write_policy_io(os.path.join(directory, "policy_io.json"), policy_io)

    print(f"💾 [Diff-Drive Task1 skrl checkpoint] saved to: {directory}", flush=True)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    set_seed(int(args_cli.seed))

    run_name = make_run_name()
    log_dir = os.path.abspath(args_cli.log_root)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 118)
    print("🚀 Diff-Drive UGV / Jetbot Task1 Multi-waypoint Navigation - TRUE skrl PPO Training")
    print("=" * 118)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_root     = {log_dir}")
    print(f"[INFO] run_name     = {run_name}")
    print("[INFO] This version uses skrl PPO, not torch-native PPO.")
    print("[INFO] Raw policy actions remain [-1, 1]; action[0] is non-negative forward throttle and action[1] is turn.")

    env_cfg = Task1Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.device)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.max_episode_length_s = float(args_cli.max_episode_length_s)
    env_cfg.num_waypoints = int(max(args_cli.num_waypoints, 3 if not args_cli.disable_curriculum else args_cli.num_waypoints))
    env_cfg.print_debug_info = False
    env_cfg.validate()

    base_env = DiffDriveTask1Env(env_cfg)
    local_env = DiffDriveTask1SkrlWrapper(base_env)

    local_env.curriculum_stage = apply_task1_curriculum(base_env, 0)

    env = wrap_env(local_env, wrapper="isaaclab")
    num_envs = getattr(env, "num_envs", local_env.num_envs)

    if env.state_space is None:
        raise RuntimeError("env.state_space is None. Task1 requires critic state space, even if same as actor obs.")

    print("\n[DEBUG] Diff-Drive Task1 Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  obs_protocol          = {env_cfg.obs_protocol}")
    print(f"  action_protocol       = {env_cfg.action_protocol}")
    print(f"  model_protocol        = {env_cfg.model_protocol}")
    print(f"  single_obs_dim        = {env_cfg.single_obs_dim}")
    print(f"  frame_stack           = {env_cfg.frame_stack}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")

    models = {
        "policy": DiffDriveActor(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            init_log_std=float(args_cli.init_log_std),
            min_log_std=float(args_cli.min_log_std),
            max_log_std=float(args_cli.max_log_std),
            core_obs_dim=int(env_cfg.frame_stack * env_cfg.core_single_obs_dim),
        ),
        "value": DiffDriveCritic(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            core_obs_dim=int(env_cfg.frame_stack * env_cfg.core_single_obs_dim),
        ),
    }

    cfg = build_skrl_cfg(env, log_dir=log_dir, run_name=run_name)

    memory = RandomMemory(
        memory_size=int(cfg["rollouts"]),
        num_envs=num_envs,
        device=env.device,
    )

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
    )

    resume_ckpt = resolve_resume_checkpoint(args_cli.resume)
    if resume_ckpt:
        if os.path.exists(resume_ckpt):
            print(f"[INFO] Loading skrl agent checkpoint: {resume_ckpt}")
            agent.load(resume_ckpt)
        else:
            print(f"[WARN] resume checkpoint not found: {resume_ckpt}")

    total_env_steps = int(args_cli.total_env_steps)
    initial_env_steps = max(0, min(int(args_cli.start_env_steps), total_env_steps))
    remaining_env_steps = max(total_env_steps - initial_env_steps, 0)
    total_vector_steps = math.ceil(remaining_env_steps / int(num_envs))
    save_freq_env_steps = int(args_cli.save_freq_env_steps)
    update_env_steps = int(cfg["rollouts"]) * int(num_envs)

    trainer = StepTrainer(
        cfg={
            "timesteps": int(total_vector_steps),
            "headless": True,
            "disable_progressbar": True,
        },
        env=env,
        agents=agent,
    )

    print("\n[INFO] skrl PPO configuration")
    print(f"  - num_envs            : {num_envs:,}")
    print(f"  - total_env_steps     : {total_env_steps:,}")
    print(f"  - start_env_steps     : {initial_env_steps:,}")
    print(f"  - remaining_env_steps : {remaining_env_steps:,}")
    print(f"  - total_vector_steps  : {total_vector_steps:,}")
    print(f"  - update_env_steps    : {update_env_steps:,}")
    print(f"  - save_freq_env_steps : {save_freq_env_steps:,}")
    print(f"  - curriculum          : {not args_cli.disable_curriculum}")
    print(f"  - obs_protocol        : {env_cfg.obs_protocol}")
    print(f"  - action_protocol     : {env_cfg.action_protocol}")
    print(f"  - model_protocol      : {env_cfg.model_protocol}")
    print(f"  - single_obs_dim      : {env_cfg.single_obs_dim}")
    print(f"  - core_single_obs_dim : {env_cfg.core_single_obs_dim}")
    print(f"  - frame_stack         : {env_cfg.frame_stack}")
    print(f"  - actor_obs_dim       : {env.observation_space.shape[0]}")
    print(f"  - critic_obs_dim      : {env.state_space.shape[0]}")
    print(f"  - action_dim          : {env.action_space.shape[0]}")
    print(f"  - num_waypoints       : {env_cfg.num_waypoints}")
    print(f"  - max_episode_length  : {env_cfg.max_episode_length}")
    print(f"  - min_forward_action  : {env_cfg.min_forward_action}")
    print(f"  - max_forward_action  : {env_cfg.max_forward_action}")
    print(f"  - rollouts            : {cfg['rollouts']}")
    print(f"  - learning_epochs     : {cfg.get('learning_epochs')}")
    print(f"  - mini_batches        : {cfg.get('mini_batches')}")
    print(f"  - lr                  : {cfg.get('learning_rate')}")
    print(f"  - tensorboard         : tensorboard --logdir={log_dir}")

    print("\n🔥 [Diff-Drive Task1 TRUE skrl PPO 已点火]")
    print("👉 任务目标：两轮差速无人车连续通过多个随机 waypoint")
    print("👉 Actor/Critic 输入：CoreNav-v1，3 帧堆叠观测，42 维")
    print("👉 模型协议：ModularActor-v1 = CoreNavEncoder + TaskExtraEncoder + Fusion/PolicyHead")
    print("👉 网络动作：2 维 [forward_throttle, turn]；车体前进命令非负，左右轮可差速反转用于掉头")
    print("👉 日志重点：Progress / Goal_Aligned_Speed / Exec_Action / Wheel_Target / Recent_Finish_Rate / Backward_Ratio\n")

    last_save = initial_env_steps
    update_id = 0
    start_time = time.time()
    env_steps = initial_env_steps
    last_curriculum_update = -1
    best_score = -float("inf")
    best_finish_rate = 0.0
    best_checkpoint_dir = ""
    collapse_count = 0

    try:
        trainer.reset()

        with tqdm(
            total=total_env_steps,
            initial=initial_env_steps,
            desc="Diff-Drive Task1 skrl PPO",
            unit="steps",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.05,
        ) as pbar:
            for t in range(total_vector_steps):
                previous_env_steps = min(initial_env_steps + t * int(num_envs), total_env_steps)
                env_steps = min(initial_env_steps + (t + 1) * int(num_envs), total_env_steps)

                if (
                    previous_env_steps == 0
                    or previous_env_steps - last_curriculum_update >= int(args_cli.curriculum_interval)
                ):
                    local_env.curriculum_stage = apply_task1_curriculum(base_env, previous_env_steps)
                    set_agent_lr(agent, lr_for_stage(float(local_env.curriculum_stage)))
                    last_curriculum_update = previous_env_steps

                trainer.train(timestep=t, timesteps=total_vector_steps)
                set_agent_lr(agent, lr_for_stage(float(local_env.curriculum_stage)))

                pbar.update(env_steps - previous_env_steps)

                flat = flat_dict(local_env.last_info)
                elapsed = time.time() - start_time
                fps = max(env_steps - initial_env_steps, 0) / max(elapsed, 1e-6)

                pbar.set_postfix(
                    {
                        "steps": f"{env_steps:,}",
                        "fps": f"{fps:,.0f}",
                        "rew": f"{local_env.last_reward_mean:+.3f}",
                        "done": local_env.last_done_count,
                        "stage": f"{float(local_env.curriculum_stage):.1f}",
                        "dist": f"{flat.get('telemetry/Distance_To_Waypoint', 0.0):.2f}",
                        "prog": f"{flat.get('telemetry/Progress', 0.0):+.3f}",
                        "goal_v": f"{flat.get('telemetry/Goal_Aligned_Speed', 0.0):+.2f}",
                        "exec": f"{flat.get('telemetry/Exec_Action_Left', 0.0):.2f}/{flat.get('telemetry/Exec_Action_Right', 0.0):.2f}",
                        "back": f"{flat.get('telemetry/Backward_Ratio', 0.0):.2f}",
                        "slow": f"{flat.get('telemetry/Slow_Ratio', 0.0):.2f}",
                        "wp": f"{flat.get('telemetry/Waypoint_Index', 0.0):.2f}",
                        "recent_finish": f"{flat.get('events/Recent_Finish_Rate', 0.0):.3f}",
                        "recent_timeout": f"{flat.get('events/Recent_Timeout_Rate', 0.0):.3f}",
                    }
                )

                writer = getattr(agent, "writer", None)
                write_scalars(writer, local_env.last_info.get("reward_components", {}), env_steps, "rewards")
                write_scalars(writer, local_env.last_info.get("events", {}), env_steps, "events")
                write_scalars(writer, local_env.last_info.get("telemetry", {}), env_steps, "telemetry")
                write_scalars(writer, local_env.last_info.get("debug", {}), env_steps, "debug")
                write_scalars(writer, local_env.last_info.get("curriculum", {}), env_steps, "curriculum")

                try:
                    if writer is not None:
                        writer.add_scalar("rollout/reward_mean_raw", local_env.last_reward_mean, env_steps)
                        writer.add_scalar("rollout/done_count", local_env.last_done_count, env_steps)
                except Exception:
                    pass

                if (t + 1) % int(cfg["rollouts"]) == 0:
                    update_id += 1

                    ppo_info = tracking_mean(agent)
                    lr = current_lr(agent)
                    ppo_info["learning_rate"] = lr

                    write_scalars(writer, ppo_info, env_steps, "ppo")
                    write_scalars(writer, flat, env_steps, "env_info")

                    recent_finish = float(flat.get("events/Recent_Finish_Rate", 0.0))
                    recent_timeout = float(flat.get("events/Recent_Timeout_Rate", 0.0))
                    recent_waypoints = float(flat.get("telemetry/Recent_Terminal_Waypoints", 0.0))
                    recent_return = float(flat.get("telemetry/Recent_Terminal_Return", 0.0))
                    current_stage = float(local_env.curriculum_stage)
                    stage_score = (
                        10.0 * recent_finish
                        - 4.0 * recent_timeout
                        + 0.75 * recent_waypoints
                        + 0.01 * recent_return
                    )

                    if current_stage >= 1.0 and recent_finish >= float(args_cli.best_min_finish_rate) and stage_score > best_score + 0.05:
                        best_score = stage_score
                        best_finish_rate = recent_finish
                        best_checkpoint_dir = os.path.join(log_dir, run_name, "best_checkpoint")
                        try:
                            save_project_checkpoint(
                                best_checkpoint_dir,
                                agent=agent,
                                models=models,
                                env_cfg=env_cfg,
                                env=local_env,
                                env_steps=env_steps,
                                args=args_cli,
                            )
                            pbar.write(
                                f"\n🏆 [Task1 best checkpoint] steps={env_steps:,} | "
                                f"stage={current_stage:.1f} | finish={recent_finish:.3f} | "
                                f"timeout={recent_timeout:.3f} | score={best_score:.3f} | {best_checkpoint_dir}\n"
                            )
                        except Exception as exc:
                            pbar.write(f"\n[WARN] best checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

                    if (
                        current_stage >= 2.5
                        and best_finish_rate >= float(args_cli.best_min_finish_rate)
                        and recent_finish <= float(args_cli.collapse_finish_rate)
                        and recent_timeout >= float(args_cli.collapse_timeout_rate)
                    ):
                        collapse_count += 1
                    else:
                        collapse_count = 0

                    if collapse_count >= int(args_cli.collapse_patience):
                        pbar.write(
                            "\n[WARN] Task1 anti-regression guard triggered: "
                            f"recent_finish={recent_finish:.3f}, recent_timeout={recent_timeout:.3f}. "
                            f"Use best checkpoint: {best_checkpoint_dir}\n"
                        )
                        if bool(args_cli.stop_on_collapse):
                            break
                        collapse_count = 0

                    if update_id % max(int(args_cli.summary_interval), 1) == 0:
                        stat = {
                            "update": float(update_id),
                            "env_steps": float(env_steps),
                            "total_env_steps": float(total_env_steps),
                            "progress_percent": 100.0 * env_steps / max(total_env_steps, 1),
                            "num_envs": float(num_envs),
                            "rollouts_per_update": float(cfg["rollouts"]),
                            "fps": float(fps),
                            "learning_rate": float(lr),
                        }

                        pbar.write(
                            "\n".join(
                                [
                                    "\n" + "=" * 118,
                                    f"📊 [Diff-Drive Task1 skrl PPO 更新 {update_id}] "
                                    f"总步数: {env_steps:,} / {total_env_steps:,} | "
                                    f"FPS: {fps:,.0f} | LR: {lr:.3e} | Curriculum Stage: {local_env.curriculum_stage}",
                                    "=" * 118,
                                    make_table("time / progress", stat),
                                    make_table("env info: rewards + events + telemetry + debug", flat),
                                    make_table("ppo update info", ppo_info),
                                    "=" * 118 + "\n",
                                ]
                            )
                        )

                    try:
                        agent.tracking_data.clear()
                    except Exception:
                        pass

                if env_steps - last_save >= save_freq_env_steps:
                    last_save = env_steps
                    save_dir = os.path.join(log_dir, run_name, f"checkpoint_{env_steps}")

                    try:
                        save_project_checkpoint(
                            save_dir,
                            agent=agent,
                            models=models,
                            env_cfg=env_cfg,
                            env=local_env,
                            env_steps=env_steps,
                            args=args_cli,
                        )
                        pbar.write(f"\n💾 [Diff-Drive Task1 skrl 备份] 总步数: {env_steps:,} | 已保存至: {save_dir}\n")
                    except Exception as exc:
                        pbar.write(f"\n[WARN] checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

    except KeyboardInterrupt:
        print("\n[WARN] 接收到 Ctrl+C，正在保存当前 Diff-Drive Task1 skrl 模型...")
    except Exception:
        print("\n[ERROR] Diff-Drive Task1 skrl PPO 训练过程中发生真实异常：")
        traceback.print_exc()
        raise
    finally:
        final_dir = os.path.join(log_dir, run_name, "final_checkpoint")

        try:
            save_project_checkpoint(
                final_dir,
                agent=agent,
                models=models,
                env_cfg=env_cfg,
                env=local_env,
                env_steps=int(env_steps),
                args=args_cli,
            )
            print(f"✅ Diff-Drive Task1 skrl 模型已保存至 {final_dir}")
            if best_checkpoint_dir:
                print(f"🏆 Task1 best checkpoint: {best_checkpoint_dir}")
        except Exception as exc:
            print(f"[WARN] 保存最终 skrl 模型失败: {type(exc).__name__}: {exc}")
            if best_checkpoint_dir:
                print(f"🏆 Task1 best checkpoint: {best_checkpoint_dir}")

        try:
            env.close()
        except Exception:
            try:
                local_env.close()
            except Exception:
                pass

        try:
            simulation_app.close()
        except Exception:
            pass

        print("✅ Diff-Drive Task1 TRUE skrl PPO training pipeline safely exited")


if __name__ == "__main__":
    main()
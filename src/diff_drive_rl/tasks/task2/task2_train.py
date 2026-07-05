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

parser = argparse.ArgumentParser(description="Train Diff-Drive UGV Task2 with skrl PPO")

# Runtime
parser.add_argument("--total-env-steps", type=int, default=50_000_000)
parser.add_argument("--save-freq-env-steps", type=int, default=10_000_000)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default="", help="Optional Task2 skrl checkpoint or final_checkpoint directory")
parser.add_argument("--task1-core-checkpoint", type=str, default="", help="Optional Task1 checkpoint/model path used to initialize Task2 policy.core_encoder")

# Curriculum / env
parser.add_argument("--start-k", type=float, default=0.0)
parser.add_argument("--force-stage", type=int, default=0, help="Fix reset sampling to a curriculum stage. -1 means normal global-step curriculum")
parser.add_argument("--max-episode-length-s", type=float, default=80.0)

# PPO
parser.add_argument("--rollouts", type=int, default=64)
parser.add_argument("--learning-epochs", type=int, default=4)
parser.add_argument("--mini-batches", type=int, default=8)

parser.add_argument("--lr", type=float, default=1.0e-4)
parser.add_argument("--min-lr", type=float, default=5e-5)
parser.add_argument("--max-lr", type=float, default=2.5e-4)

parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-range", type=float, default=0.20)
parser.add_argument("--value-clip", type=float, default=0.20)
parser.add_argument("--entropy-coef", type=float, default=0.004)
parser.add_argument("--value-coef", type=float, default=1.0)
parser.add_argument("--grad-clip", type=float, default=0.5)

# Policy distribution
parser.add_argument("--init-log-std", type=float, default=-0.70)
parser.add_argument("--min-log-std", type=float, default=-4.0)
parser.add_argument("--max-log-std", type=float, default=0.30)

# KL
parser.add_argument("--target-kl", type=float, default=0.020)
parser.add_argument("--hard-kl-stop", type=float, default=0.120)

# Logging
parser.add_argument("--log-root", type=str, default=str(PROJECT_ROOT / "logs" / "task2"))
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--summary-interval", type=int, default=10)
parser.add_argument("--skrl-write-interval", type=int, default=1_000_000)
parser.add_argument("--skrl-checkpoint-interval", type=int, default=0)

# Stage-wise checkpoint selection.  Final checkpoints are still saved, but Task2
# hand-off should use best_balanced / best_efficiency / stage_pass checkpoints.
parser.add_argument("--disable-best-checkpoints", action="store_true", help="Disable metric-based best checkpoint saving")
parser.add_argument("--best-checkpoint-min-interval-env-steps", type=int, default=500_000)
parser.add_argument("--stage-pass-patience", type=int, default=3)
parser.add_argument("--stage-pass-early-stop", action="store_true", help="Stop training once stage pass criteria are met for patience summaries")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from diff_drive_rl.tasks.task2.task2_config import Task2Config
from diff_drive_rl.tasks.task2.task2_env import DiffDriveTask2Env

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


def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def current_stage_index(env_cfg: Task2Config, flat: Dict[str, float]) -> int:
    raw_stage = flat.get("telemetry/Stage", float(env_cfg.force_stage if env_cfg.force_stage >= 0 else 0))
    idx = int(round(float(raw_stage)))
    return max(0, min(idx, int(env_cfg.world_cfg.num_stages) - 1))


def stage_threshold(env_cfg: Task2Config, name: str, stage: int) -> float:
    values = getattr(env_cfg, name)
    stage = max(0, min(int(stage), len(values) - 1))
    return float(values[stage])


def compute_balanced_score(env_cfg: Task2Config, flat: Dict[str, float]) -> Dict[str, float]:
    stage = current_stage_index(env_cfg, flat)
    success = clamp01(flat.get("events/Episode_Success_Rate", flat.get("events/Current_Window_Success_Rate", 0.0)))
    collision = clamp01(flat.get("events/Episode_Collision_Rate", flat.get("events/Current_Window_Collision_Rate", 0.0)))
    timeout = clamp01(flat.get("events/Episode_Timeout_Rate", flat.get("events/Current_Window_Timeout_Rate", 0.0)))
    oob = clamp01(flat.get("events/Episode_Out_Of_Bounds_Rate", flat.get("events/Current_Window_Out_Of_Bounds_Rate", 0.0)))
    progress_velocity = max(0.0, float(flat.get("telemetry/Progress_Velocity", flat.get("telemetry/Current_Window_Goal_Aligned_Speed", 0.0))))
    target_speed = max(0.05, float(flat.get("telemetry/Target_Speed", 0.5)))
    progress_norm = clamp01(progress_velocity / target_speed)
    heading_cos = clamp01((float(flat.get("telemetry/Heading_Cos", 0.0)) + 1.0) * 0.5)
    speed_ratio = clamp01(float(flat.get("telemetry/Speed_Ratio", 0.0)))

    score = (
        float(env_cfg.best_score_success_weight) * success
        - float(env_cfg.best_score_collision_weight) * collision
        - float(env_cfg.best_score_timeout_weight) * timeout
        - float(env_cfg.best_score_out_of_bounds_weight) * oob
        + float(env_cfg.best_score_progress_weight) * progress_norm
        + float(env_cfg.best_score_heading_weight) * heading_cos
        + float(env_cfg.best_score_speed_weight) * speed_ratio
    )

    efficiency_score = (
        success
        - 0.30 * collision
        - 0.30 * timeout
        - 0.20 * oob
        + 0.35 * progress_norm
        + 0.35 * heading_cos
        + 0.15 * speed_ratio
    )

    success_score = success - 0.50 * collision - 0.50 * timeout - 0.30 * oob

    return {
        "stage": float(stage),
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "out_of_bounds": oob,
        "progress_velocity": progress_velocity,
        "progress_norm": progress_norm,
        "heading_cos": float(flat.get("telemetry/Heading_Cos", 0.0)),
        "heading_score": heading_cos,
        "speed_ratio": speed_ratio,
        "balanced_score": float(score),
        "efficiency_score": float(efficiency_score),
        "success_score": float(success_score),
    }


def stage_passed(env_cfg: Task2Config, flat: Dict[str, float]) -> bool:
    stage = current_stage_index(env_cfg, flat)
    success = float(flat.get("events/Episode_Success_Rate", 0.0))
    timeout = float(flat.get("events/Episode_Timeout_Rate", 1.0))
    collision = float(flat.get("events/Episode_Collision_Rate", 1.0))
    oob = float(flat.get("events/Episode_Out_Of_Bounds_Rate", 1.0))
    progress_velocity = float(flat.get("telemetry/Progress_Velocity", 0.0))
    heading_cos = float(flat.get("telemetry/Heading_Cos", -1.0))
    speed_ratio = float(flat.get("telemetry/Speed_Ratio", 0.0))

    return (
        success >= stage_threshold(env_cfg, "stage_pass_success_rate", stage)
        and timeout <= stage_threshold(env_cfg, "stage_pass_timeout_rate", stage)
        and collision <= stage_threshold(env_cfg, "stage_pass_collision_rate", stage)
        and oob <= stage_threshold(env_cfg, "stage_pass_out_of_bounds_rate", stage)
        and progress_velocity >= stage_threshold(env_cfg, "stage_pass_progress_velocity", stage)
        and heading_cos >= stage_threshold(env_cfg, "stage_pass_heading_cos", stage)
        and speed_ratio >= stage_threshold(env_cfg, "stage_pass_speed_ratio", stage)
    )


def model_param_vector(model: nn.Module) -> torch.Tensor:
    """返回 CPU 上的参数向量，用于核查 PPO 是否真实更新。"""

    with torch.no_grad():
        return torch.cat([p.detach().float().view(-1).cpu() for p in model.parameters()])


def model_param_norm(model: nn.Module) -> float:
    vec = model_param_vector(model)
    return float(torch.norm(vec).item())


def resolve_resume_metadata(path: str) -> str:
    if not path:
        return ""
    p = Path(path).expanduser().resolve()
    candidates = []
    if p.is_file():
        candidates += [p.parent / "task2_train_metadata.pt", p.parent.parent / "task2_train_metadata.pt"]
    elif p.is_dir():
        candidates += [p / "task2_train_metadata.pt", p / "final_checkpoint" / "task2_train_metadata.pt"]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def make_run_name() -> str:
    run_name = args_cli.run_name.strip()
    if run_name:
        return run_name
    return f"diff_drive_task2_skrl_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ======================================================================
# skrl wrapper
# ======================================================================

class DiffDriveTask2SkrlWrapper(gym.Env):
    """Task2 skrl wrapper.

    The underlying environment already outputs the 3-frame stacked 498-D
    observation. For Task2 baseline, actor and critic both receive the same
    observation. Future asymmetric critic can be added without changing the
    external training pipeline.
    """

    def __init__(self, env: DiffDriveTask2Env):
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

class DiffDriveTask2Actor(GaussianMixin, Model):
    """Modular actor for Task2.

    Observation layout is 3 stacked frames. In each frame, the first 14 dims are
    core navigation and the remaining 152 dims are Task2 obstacle perception. The
    policy explicitly separates these two parts so Task1 can initialize only the
    shared core_encoder.
    """

    core_single_dim = 14
    frame_stack = 3

    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        init_log_std: float = -0.70,
        min_log_std: float = -4.0,
        max_log_std: float = 0.30,
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

        obs_dim = int(observation_space.shape[0])
        if obs_dim % self.frame_stack != 0:
            raise RuntimeError(f"Task2 obs_dim={obs_dim} is not divisible by frame_stack={self.frame_stack}")

        self.single_obs_dim = obs_dim // self.frame_stack
        self.extra_single_dim = self.single_obs_dim - self.core_single_dim
        if self.extra_single_dim < 0:
            raise RuntimeError(f"Task2 single_obs_dim={self.single_obs_dim} < CoreNav dim={self.core_single_dim}")

        self.core_obs_dim = self.core_single_dim * self.frame_stack
        self.extra_obs_dim = self.extra_single_dim * self.frame_stack

        self.core_encoder = nn.Sequential(
            nn.Linear(self.core_obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
        )
        self.extra_encoder = nn.Sequential(
            nn.Linear(self.extra_obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.mean_head = nn.Linear(128, action_space.shape[0])

        self.log_std_parameter = nn.Parameter(torch.full((action_space.shape[0],), float(init_log_std)))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)

    def _split_obs(self, states: torch.Tensor):
        x = states.reshape(states.shape[0], self.frame_stack, self.single_obs_dim)
        core = x[:, :, : self.core_single_dim].reshape(states.shape[0], self.core_obs_dim)
        extra = x[:, :, self.core_single_dim :].reshape(states.shape[0], self.extra_obs_dim)
        return core, extra

    def compute(self, inputs, role):
        states = inputs.get("observations", inputs.get("states"))
        core_obs, extra_obs = self._split_obs(states)
        core_latent = self.core_encoder(core_obs)
        extra_latent = self.extra_encoder(extra_obs)
        fused = self.fusion_head(torch.cat([core_latent, extra_latent], dim=-1))
        actions = self.mean_head(fused)
        return actions, {"log_std": self.log_std_parameter}


class DiffDriveTask2Critic(DeterministicMixin, Model):
    """Modular critic using the same CoreNav / extra split as the actor."""

    core_single_dim = 14
    frame_stack = 3

    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)

        state_dim = int(state_space.shape[0])
        if state_dim % self.frame_stack != 0:
            raise RuntimeError(f"Task2 state_dim={state_dim} is not divisible by frame_stack={self.frame_stack}")

        self.single_obs_dim = state_dim // self.frame_stack
        self.extra_single_dim = self.single_obs_dim - self.core_single_dim
        self.core_obs_dim = self.core_single_dim * self.frame_stack
        self.extra_obs_dim = self.extra_single_dim * self.frame_stack

        self.core_encoder = nn.Sequential(
            nn.Linear(self.core_obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
        )
        self.extra_encoder = nn.Sequential(
            nn.Linear(self.extra_obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.value_head = nn.Linear(128, 1)
        self.apply(DiffDriveTask2Actor._init_weights)

    def _split_obs(self, states: torch.Tensor):
        x = states.reshape(states.shape[0], self.frame_stack, self.single_obs_dim)
        core = x[:, :, : self.core_single_dim].reshape(states.shape[0], self.core_obs_dim)
        extra = x[:, :, self.core_single_dim :].reshape(states.shape[0], self.extra_obs_dim)
        return core, extra

    def compute(self, inputs, role):
        states = inputs.get("states", None)
        if states is None:
            states = inputs.get("observations", None)
        if states is None:
            raise RuntimeError("Critic received no states / observations.")
        core_obs, extra_obs = self._split_obs(states)
        core_latent = self.core_encoder(core_obs)
        extra_latent = self.extra_encoder(extra_obs)
        fused = self.fusion_head(torch.cat([core_latent, extra_latent], dim=-1))
        return self.value_head(fused), {}


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
            p / "diff_drive_task2_skrl_agent.pt",
            p / "agent.pt",
            p / "final_checkpoint" / "diff_drive_task2_skrl_agent.pt",
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
    env_cfg: Task2Config,
    env: DiffDriveTask2SkrlWrapper,
    env_steps: int,
    args,
) -> None:
    os.makedirs(directory, exist_ok=True)

    skrl_agent_path = os.path.join(directory, "diff_drive_task2_skrl_agent.pt")
    eval_model_path = os.path.join(directory, "diff_drive_task2_model.pt")

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
            "actor_obs_norm": actor_norm,
            "critic_obs_norm": critic_norm,
            "env_steps": int(env_steps),
            "args": vars(args),
            "metadata": {
                "robot": "two-wheel differential-drive UGV",
                "task": "task2_obstacle_navigation",
                "algorithm": "skrl_PPO",
                "uses_skrl": True,
                "asymmetric_actor_critic": False,
                "policy_input": "obs_3_frame_stack",
                "critic_input": "same_as_policy",
                "action_protocol": str(env_cfg.action_protocol),
                "obs_protocol": str(env_cfg.obs_protocol),
                "model_protocol": str(env_cfg.model_protocol),
                "core_single_obs_dim": int(env_cfg.core_single_obs_dim),
                "stacked_core_obs_dim": int(env_cfg.stacked_core_obs_dim),
                "task_extra_single_obs_dim": int(env_cfg.task_extra_single_obs_dim),
                "stacked_task_extra_obs_dim": int(env_cfg.stacked_task_extra_obs_dim),
                "single_obs_dim": int(env_cfg.single_obs_dim),
                "frame_stack": int(env_cfg.frame_stack),
                "actor_obs_dim": int(env_cfg.num_observations),
                "critic_obs_dim": int(env_cfg.num_observations),
                "action_dim": int(env_cfg.num_actions),
                "num_lidar_rays": int(env_cfg.world_cfg.num_lidar_rays),
                "max_static_obs": int(env_cfg.world_cfg.max_static_obs),
                "max_dynamic_obs": int(env_cfg.world_cfg.max_dynamic_obs),
                "world": "analytic_gpu_world",
                "control": "forward_throttle_plus_turn_to_left_right_wheel_velocity",
                "action_semantics": "action[0]=forward_throttle_nonnegative_with_stage_min_speed, action[1]=turn",
                "transfer_note": "Task1 -> Task2 migration loads policy.core_encoder only; Task2 extra_encoder/fusion/head are task-specific.",
                "note": "skrl PPO checkpoint. Evaluation uses deterministic policy forward, not agent.act.",
            },
        },
        eval_model_path,
    )

    torch.save(
        {
            "stage": "task2_skrl_ppo",
            "algorithm": "skrl_PPO",
            "uses_skrl": True,
            "env_steps": int(env_steps),
            "num_envs": int(env_cfg.num_envs),
            "global_steps": int(getattr(env.env, "global_steps", env_steps)),
            "force_stage": int(env_cfg.force_stage),
            "env_cfg": {
                "num_observations": int(env_cfg.num_observations),
                "single_obs_dim": int(env_cfg.single_obs_dim),
                "frame_stack": int(env_cfg.frame_stack),
                "num_actions": int(env_cfg.num_actions),
                "max_episode_length": int(env_cfg.max_episode_length),
                "action_protocol": str(env_cfg.action_protocol),
                "obs_protocol": str(env_cfg.obs_protocol),
                "model_protocol": str(env_cfg.model_protocol),
                "core_single_obs_dim": int(env_cfg.core_single_obs_dim),
                "stacked_core_obs_dim": int(env_cfg.stacked_core_obs_dim),
                "task_extra_single_obs_dim": int(env_cfg.task_extra_single_obs_dim),
                "world": {
                    "num_lidar_rays": int(env_cfg.world_cfg.num_lidar_rays),
                    "max_static_obs": int(env_cfg.world_cfg.max_static_obs),
                    "max_dynamic_obs": int(env_cfg.world_cfg.max_dynamic_obs),
                    "curriculum_total_steps": int(env_cfg.world_cfg.curriculum_total_steps),
                    "force_stage": int(env_cfg.force_stage),
                },
            },
        },
        os.path.join(directory, "task2_train_metadata.pt"),
    )

    print(f"💾 [Diff-Drive Task2 skrl checkpoint] saved to: {directory}", flush=True)



def _candidate_state_dicts(payload: Any):
    """Yield tensor dictionaries from common checkpoint formats."""
    if isinstance(payload, dict):
        # Direct state_dict.
        if any(torch.is_tensor(v) for v in payload.values()):
            yield payload

        for key in ["policy", "actor", "model", "state_dict"]:
            value = payload.get(key, None)
            if isinstance(value, dict):
                yield from _candidate_state_dicts(value)

        models = payload.get("models", None)
        if isinstance(models, dict):
            for value in models.values():
                if isinstance(value, dict):
                    yield from _candidate_state_dicts(value)


def load_task1_core_encoder(policy: DiffDriveTask2Actor, checkpoint_path: str) -> bool:
    """Load Task1 CoreNav encoder weights into Task2 policy.core_encoder.

    This intentionally loads only the shared CoreNav encoder. Task2's obstacle
    extra_encoder, fusion_head and action head remain Task2-specific.
    """
    if not checkpoint_path:
        return False

    path = Path(checkpoint_path).expanduser().resolve()
    if path.is_dir():
        for name in ["diff_drive_task1_model.pt", "diff_drive_task1_skrl_agent.pt", "agent.pt"]:
            candidate = path / name
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        print(f"[WARN] Task1 core checkpoint not found: {path}")
        return False

    payload = torch.load(str(path), map_location="cpu")
    target_state = policy.core_encoder.state_dict()

    for state in _candidate_state_dicts(payload):
        core_state: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            normalized = str(key)
            if "core_encoder." not in normalized:
                continue
            suffix = normalized.split("core_encoder.", 1)[1]
            if suffix in target_state and tuple(target_state[suffix].shape) == tuple(value.shape):
                core_state[suffix] = value.detach().cpu()

        if core_state:
            missing, unexpected = policy.core_encoder.load_state_dict(core_state, strict=False)
            loaded = sorted(core_state.keys())
            print(f"[INFO] Loaded Task1 CoreNav encoder from: {path}")
            print(f"[INFO] Loaded core_encoder tensors: {loaded}")
            if missing:
                print(f"[WARN] Missing core_encoder tensors after partial load: {list(missing)}")
            if unexpected:
                print(f"[WARN] Unexpected core_encoder tensors after partial load: {list(unexpected)}")
            return True

    print(f"[WARN] No compatible core_encoder weights found in Task1 checkpoint: {path}")
    return False


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    set_seed(int(args_cli.seed))

    run_name = make_run_name()
    log_dir = os.path.abspath(args_cli.log_root)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 118)
    print("🚀 Diff-Drive UGV Task2 Core Navigation + Obstacle Navigation - skrl PPO Training")
    print("=" * 118)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_root     = {log_dir}")
    print(f"[INFO] run_name     = {run_name}")
    print("[INFO] This version uses skrl PPO, not torch-native PPO or stable-baselines3.")

    env_cfg = Task2Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.device)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.force_stage = int(args_cli.force_stage)
    env_cfg.max_episode_length_s = float(args_cli.max_episode_length_s)
    env_cfg.print_debug_info = False
    env_cfg.validate()

    if float(args_cli.start_k) > 0.0:
        initial_global_steps = int(float(args_cli.start_k) * env_cfg.world_cfg.curriculum_total_steps)
    else:
        initial_global_steps = 0

    base_env = DiffDriveTask2Env(env_cfg)
    base_env.global_steps = int(initial_global_steps)
    # __init__ 内已经做过一次 reset；start_k / force_stage 设置后需要重置一次，
    # 确保首批 episode 就来自正确课程阶段。
    base_env.reset()

    if initial_global_steps > 0:
        print(
            f"[INFO] start_k={args_cli.start_k:.4f}, "
            f"initial_global_steps={initial_global_steps:,}, "
            f"stage={base_env.world.stage_from_global_steps(initial_global_steps)}"
        )

    if int(args_cli.force_stage) >= 0:
        print(f"[INFO] force_stage={int(args_cli.force_stage)} enabled: all resets sample this stage")

    local_env = DiffDriveTask2SkrlWrapper(base_env)
    env = wrap_env(local_env, wrapper="isaaclab")
    num_envs = getattr(env, "num_envs", local_env.num_envs)

    if env.state_space is None:
        raise RuntimeError("env.state_space is None. Task2 requires critic state space, even if same as actor obs.")

    print("\n[DEBUG] Diff-Drive Task2 Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  action_protocol       = {env_cfg.action_protocol}")
    print(f"  obs_protocol          = {env_cfg.obs_protocol}")
    print(f"  model_protocol        = {env_cfg.model_protocol}")
    print(f"  single_obs_dim        = {env_cfg.single_obs_dim}")
    print(f"  core_single_obs_dim   = {env_cfg.core_single_obs_dim}")
    print(f"  frame_stack           = {env_cfg.frame_stack}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")
    print(f"  lidar rays            = {env_cfg.world_cfg.num_lidar_rays}")
    print(f"  max static obs        = {env_cfg.world_cfg.max_static_obs}")
    print(f"  max dynamic obs       = {env_cfg.world_cfg.max_dynamic_obs}")

    models = {
        "policy": DiffDriveTask2Actor(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            init_log_std=float(args_cli.init_log_std),
            min_log_std=float(args_cli.min_log_std),
            max_log_std=float(args_cli.max_log_std),
        ),
        "value": DiffDriveTask2Critic(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
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

    if str(args_cli.task1_core_checkpoint).strip() and not str(args_cli.resume).strip():
        load_task1_core_encoder(models["policy"], str(args_cli.task1_core_checkpoint).strip())
    elif str(args_cli.task1_core_checkpoint).strip() and str(args_cli.resume).strip():
        print("[WARN] --resume is set, so full Task2 checkpoint loading takes precedence over --task1-core-checkpoint")

    resume_ckpt = resolve_resume_checkpoint(args_cli.resume)
    if resume_ckpt:
        if os.path.exists(resume_ckpt):
            print(f"[INFO] Loading skrl agent checkpoint: {resume_ckpt}")
            agent.load(resume_ckpt)

            metadata_path = resolve_resume_metadata(args_cli.resume)
            if metadata_path and float(args_cli.start_k) <= 0.0:
                try:
                    metadata = torch.load(metadata_path, map_location="cpu")
                    restored_steps = int(metadata.get("global_steps", metadata.get("env_steps", 0)))
                    if restored_steps > 0 and int(args_cli.force_stage) < 0:
                        initial_global_steps = restored_steps
                        base_env.global_steps = restored_steps
                        base_env.reset()
                        print(f"[INFO] Restored Task2 curriculum/global steps from metadata: {restored_steps:,}")
                except Exception as exc:
                    print(f"[WARN] failed to restore task2_train_metadata.pt: {type(exc).__name__}: {exc}")
            elif resume_ckpt and not metadata_path:
                print("[WARN] resume metadata not found; curriculum uses --start-k / current default global_steps")
        else:
            print(f"[WARN] resume checkpoint not found: {resume_ckpt}")

    train_env_steps_total = int(args_cli.total_env_steps)
    display_global_steps_total = int(initial_global_steps) + int(train_env_steps_total)
    total_vector_steps = math.ceil(train_env_steps_total / int(num_envs))
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
    print(f"  - num_envs              : {num_envs:,}")
    print(f"  - train_env_steps       : {train_env_steps_total:,}  # 本次新增训练步数")
    print(f"  - initial_global_steps  : {initial_global_steps:,}")
    print(f"  - final_global_steps    : {display_global_steps_total:,}")
    print(f"  - total_vector_steps    : {total_vector_steps:,}")
    print(f"  - update_env_steps      : {update_env_steps:,}")
    print(f"  - save_freq_env_steps   : {save_freq_env_steps:,}")
    print(f"  - start_k               : {args_cli.start_k:.4f}")
    print(f"  - force_stage           : {args_cli.force_stage}")
    print(f"  - curriculum_total_steps: {env_cfg.world_cfg.curriculum_total_steps:,}")
    print(f"  - single_obs_dim        : {env_cfg.single_obs_dim}")
    print(f"  - frame_stack           : {env_cfg.frame_stack}")
    print(f"  - actor_obs_dim         : {env.observation_space.shape[0]}")
    print(f"  - critic_obs_dim        : {env.state_space.shape[0]}")
    print(f"  - action_dim            : {env.action_space.shape[0]}")
    print(f"  - action_protocol       : {env_cfg.action_protocol}")
    print(f"  - obs_protocol          : {env_cfg.obs_protocol}")
    print(f"  - model_protocol        : {env_cfg.model_protocol}")
    print(f"  - task1_core_checkpoint : {args_cli.task1_core_checkpoint or '<none>'}")
    print(f"  - lidar_rays            : {env_cfg.world_cfg.num_lidar_rays}")
    print(f"  - max_static_obs        : {env_cfg.world_cfg.max_static_obs}")
    print(f"  - max_dynamic_obs       : {env_cfg.world_cfg.max_dynamic_obs}")
    print(f"  - max_episode_length    : {env_cfg.max_episode_length}")
    print(f"  - rollouts              : {cfg['rollouts']}")
    print(f"  - learning_epochs       : {cfg.get('learning_epochs')}")
    print(f"  - mini_batches          : {cfg.get('mini_batches')}")
    print(f"  - lr                    : {cfg.get('learning_rate')}")
    print(f"  - tensorboard           : tensorboard --logdir={log_dir}")

    print("\n🔥 [Diff-Drive Task2 skrl PPO 已点火]")
    print("👉 任务目标：两轮差速无人车在解析障碍世界中导航到目标点")
    print("👉 Actor/Critic 输入：3 帧堆叠观测，498 维，其中每帧前 14 维为 core navigation")
    print("👉 动作：2 维 [forward_throttle, turn]，Stage0/1 带最小正向速度约束，环境内部转换为左右轮速度，线速度命令非负")
    print("👉 世界层：analytic GPU world + LiDAR + risk features")
    print("👉 日志重点：Progress / Goal_Aligned_Speed / Goal_Dist / Success_Rate / Collision_Rate / Stuck_Ratio\n")

    last_save = 0
    update_id = 0
    start_time = time.time()
    best_checkpoints_enabled = not bool(getattr(args_cli, "disable_best_checkpoints", False))
    best_min_interval = max(int(getattr(args_cli, "best_checkpoint_min_interval_env_steps", 500_000)), int(num_envs))
    best_scores = {"balanced": -1e9, "efficiency": -1e9, "success": -1e9}
    best_last_save_steps = {"balanced": -10**18, "efficiency": -10**18, "success": -10**18}
    stage_pass_counter = 0
    stage_pass_saved = False
    train_env_steps_done = 0
    display_global_steps = int(initial_global_steps)
    env_steps = display_global_steps
    prev_policy_vec = model_param_vector(models["policy"])
    prev_value_vec = model_param_vector(models["value"])
    prev_summary_policy_vec = prev_policy_vec.clone()
    prev_summary_value_vec = prev_value_vec.clone()
    ppo_summary_buffer = []

    try:
        trainer.reset()

        with tqdm(
            total=train_env_steps_total,
            initial=0,
            desc="Diff-Drive Task2 skrl PPO",
            unit="steps",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.05,
        ) as pbar:
            for t in range(total_vector_steps):
                trainer.train(timestep=t, timesteps=total_vector_steps)

                train_env_steps_done = min((t + 1) * int(num_envs), train_env_steps_total)
                previous_train_env_steps = min(t * int(num_envs), train_env_steps_total)
                display_global_steps = int(initial_global_steps) + int(train_env_steps_done)
                env_steps = display_global_steps
                pbar.update(max(train_env_steps_done - previous_train_env_steps, 0))

                flat = flat_dict(local_env.last_info)
                elapsed = time.time() - start_time
                fps = max(env_steps - initial_global_steps, 0) / max(elapsed, 1e-6)

                pbar.set_postfix(
                    {
                        "global": f"{env_steps:,}",
                        "train": f"{train_env_steps_done:,}",
                        "fps": f"{fps:,.0f}",
                        "rew": f"{local_env.last_reward_mean:+.3f}",
                        "done": local_env.last_done_count,
                        "stage": f"{flat.get('telemetry/Stage', 0.0):.1f}",
                        "dist": f"{flat.get('telemetry/Goal_Dist', 0.0):.2f}",
                        "prog": f"{flat.get('telemetry/Progress', 0.0):+.3f}",
                        "goal_v": f"{flat.get('telemetry/Goal_Aligned_Speed', 0.0):+.2f}",
                        "spd": f"{flat.get('telemetry/Speed_Ratio', 0.0):+.2f}",
                        "stuck": f"{flat.get('telemetry/Stuck_Ratio', 0.0):.2f}",
                        "risk": f"{flat.get('telemetry/Risk_Front', 0.0):.2f}",
                        "succ": f"{flat.get('events/Current_Window_Success_Rate', 0.0):.3f}",
                        "oob": f"{flat.get('events/Current_Window_Out_Of_Bounds_Rate', 0.0):.3f}",
                    }
                )

                writer = getattr(agent, "writer", None)
                write_scalars(writer, local_env.last_info.get("reward_components", {}), env_steps, "rewards")
                write_scalars(writer, local_env.last_info.get("events", {}), env_steps, "events")
                write_scalars(writer, local_env.last_info.get("telemetry", {}), env_steps, "telemetry")
                write_scalars(writer, local_env.last_info.get("world", {}), env_steps, "world")
                write_scalars(writer, local_env.last_info.get("debug", {}), env_steps, "debug")

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

                    policy_vec = model_param_vector(models["policy"])
                    value_vec = model_param_vector(models["value"])
                    ppo_info["Policy / Param norm"] = float(torch.norm(policy_vec).item())
                    ppo_info["Policy / Param delta update"] = float(torch.norm(policy_vec - prev_policy_vec).item())
                    ppo_info["Policy / Param delta since summary"] = float(torch.norm(policy_vec - prev_summary_policy_vec).item())
                    ppo_info["Value / Param norm"] = float(torch.norm(value_vec).item())
                    ppo_info["Value / Param delta update"] = float(torch.norm(value_vec - prev_value_vec).item())
                    ppo_info["Value / Param delta since summary"] = float(torch.norm(value_vec - prev_summary_value_vec).item())
                    ppo_info["PPO / Tracking data keys"] = float(len(getattr(agent, "tracking_data", {}) or {}))

                    try:
                        with torch.no_grad():
                            raw_action, _ = models["policy"].compute({"observations": local_env.last_obs}, role="policy")
                            raw_action = torch.clamp(raw_action, -1.0, 1.0)
                            speed_factor = 0.5 * (raw_action[:, 0] + 1.0)
                            ppo_info["PolicyMeanDebug_NotExecuted / MeanForward"] = float(raw_action[:, 0].mean().item())
                            ppo_info["PolicyMeanDebug_NotExecuted / MeanTurn"] = float(raw_action[:, 1].mean().item())
                            ppo_info["PolicyMeanDebug_NotExecuted / MeanSpeedFactor"] = float(speed_factor.mean().item())
                            ppo_info["PolicyMeanDebug_NotExecuted / ForwardPositiveRatio"] = float((speed_factor > 1e-4).float().mean().item())

                            # 真正执行到环境里的动作以 telemetry 为准，避免 raw actor debug 误导判断。
                            ppo_info["ExecutedAction / ForwardThrottleMean"] = float(flat.get("telemetry/Action_Forward_Throttle", 0.0))
                            ppo_info["ExecutedAction / SpeedFactorMean"] = float(flat.get("telemetry/Speed_Factor", 0.0))
                            ppo_info["ExecutedAction / ForwardCommandNormMean"] = float(flat.get("telemetry/Forward_Command_Norm", 0.0))
                            ppo_info["ExecutedAction / TurnMean"] = float(flat.get("telemetry/Action_Turn", 0.0))
                            ppo_info["ExecutedAction / TurnCommandNormMean"] = float(flat.get("telemetry/Turn_Command_Norm", 0.0))
                            ppo_info["ExecutedAction / TurnToGoalAlignmentMean"] = float(flat.get("telemetry/Turn_To_Goal_Alignment", 0.0))
                            ppo_info["ExecutedAction / CorrectTurnRatio"] = float(flat.get("telemetry/Correct_Turn_Ratio", 0.0))
                            ppo_info["ExecutedAction / LeftWheelTargetNormMean"] = float(flat.get("telemetry/Left_Wheel_Target_Norm", 0.0))
                            ppo_info["ExecutedAction / RightWheelTargetNormMean"] = float(flat.get("telemetry/Right_Wheel_Target_Norm", 0.0))
                    except Exception:
                        pass

                    prev_policy_vec = policy_vec
                    prev_value_vec = value_vec

                    score_info = compute_balanced_score(env_cfg, flat)
                    ppo_info["CheckpointSelection / BalancedScore"] = score_info["balanced_score"]
                    ppo_info["CheckpointSelection / EfficiencyScore"] = score_info["efficiency_score"]
                    ppo_info["CheckpointSelection / SuccessScore"] = score_info["success_score"]
                    ppo_info["CheckpointSelection / Stage"] = score_info["stage"]
                    ppo_info["CheckpointSelection / StagePassed"] = 1.0 if stage_passed(env_cfg, flat) else 0.0

                    ppo_summary_buffer.append(dict(ppo_info))
                    write_scalars(writer, ppo_info, env_steps, "ppo")
                    write_scalars(writer, flat, env_steps, "env_info")

                    if best_checkpoints_enabled:
                        best_targets = [
                            ("balanced", score_info["balanced_score"], "best_balanced_checkpoint"),
                            ("efficiency", score_info["efficiency_score"], "best_efficiency_checkpoint"),
                            ("success", score_info["success_score"], "best_success_checkpoint"),
                        ]
                        for metric_name, score_value, dirname in best_targets:
                            if (
                                score_value > best_scores[metric_name]
                                and env_steps - best_last_save_steps[metric_name] >= best_min_interval
                            ):
                                best_scores[metric_name] = float(score_value)
                                best_last_save_steps[metric_name] = int(env_steps)
                                best_dir = os.path.join(log_dir, run_name, dirname)
                                try:
                                    save_project_checkpoint(
                                        best_dir,
                                        agent=agent,
                                        models=models,
                                        env_cfg=env_cfg,
                                        env=local_env,
                                        env_steps=env_steps,
                                        args=args_cli,
                                    )
                                    torch.save(
                                        {
                                            "metric_name": metric_name,
                                            "score": float(score_value),
                                            "score_info": score_info,
                                            "flat_metrics": flat,
                                            "env_steps": int(env_steps),
                                        },
                                        os.path.join(best_dir, "checkpoint_selection.pt"),
                                    )
                                    pbar.write(
                                        f"\n🏆 [Task2 {metric_name} checkpoint] score={score_value:.6f} | "
                                        f"stage={score_info['stage']:.0f} | saved to: {best_dir}\n"
                                    )
                                except Exception as exc:
                                    pbar.write(f"\n[WARN] best checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

                    if stage_passed(env_cfg, flat):
                        stage_pass_counter += 1
                    else:
                        stage_pass_counter = 0

                    if (
                        best_checkpoints_enabled
                        and (not stage_pass_saved)
                        and stage_pass_counter >= max(int(getattr(args_cli, "stage_pass_patience", 3)), 1)
                    ):
                        pass_dir = os.path.join(log_dir, run_name, "stage_pass_checkpoint")
                        try:
                            save_project_checkpoint(
                                pass_dir,
                                agent=agent,
                                models=models,
                                env_cfg=env_cfg,
                                env=local_env,
                                env_steps=env_steps,
                                args=args_cli,
                            )
                            torch.save(
                                {
                                    "score_info": score_info,
                                    "flat_metrics": flat,
                                    "env_steps": int(env_steps),
                                    "pass_patience": int(getattr(args_cli, "stage_pass_patience", 3)),
                                },
                                os.path.join(pass_dir, "stage_pass_metrics.pt"),
                            )
                            stage_pass_saved = True
                            pbar.write(
                                f"\n✅ [Task2 stage pass] stage={score_info['stage']:.0f} | "
                                f"patience={stage_pass_counter} | checkpoint: {pass_dir}\n"
                            )
                            if bool(getattr(args_cli, "stage_pass_early_stop", False)):
                                pbar.write("\n🛑 stage_pass_early_stop enabled: stopping current stage training.\n")
                                break
                        except Exception as exc:
                            pbar.write(f"\n[WARN] stage pass checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

                    if update_id % max(int(args_cli.summary_interval), 1) == 0:
                        if ppo_summary_buffer:
                            keys = sorted(set().union(*[d.keys() for d in ppo_summary_buffer]))
                            summary_ppo_info = {}
                            for key in keys:
                                values = [d[key] for d in ppo_summary_buffer if key in d and math.isfinite(float(d[key]))]
                                if values:
                                    summary_ppo_info[key] = float(np.mean(values))
                            # These two are more useful as latest values rather than averages.
                            for key in [
                                "Policy / Param norm",
                                "Value / Param norm",
                                "Policy / Param delta since summary",
                                "Value / Param delta since summary",
                                "learning_rate",
                            ]:
                                if key in ppo_info:
                                    summary_ppo_info[key] = ppo_info[key]
                            ppo_info_to_print = summary_ppo_info
                        else:
                            ppo_info_to_print = ppo_info
                        stat = {
                            "update": float(update_id),
                            "global_env_steps": float(env_steps),
                            "initial_global_steps": float(initial_global_steps),
                            "train_env_steps": float(train_env_steps_done),
                            "train_env_steps_total": float(train_env_steps_total),
                            "display_global_steps_total": float(display_global_steps_total),
                            "progress_percent": 100.0 * train_env_steps_done / max(train_env_steps_total, 1),
                            "num_envs": float(num_envs),
                            "rollouts_per_update": float(cfg["rollouts"]),
                            "fps": float(fps),
                            "learning_rate": float(lr),
                        }

                        pbar.write(
                            "\n".join(
                                [
                                    "\n" + "=" * 118,
                                    f"📊 [Diff-Drive Task2 skrl PPO 更新 {update_id}] "
                                    f"本次训练: {train_env_steps_done:,} / {train_env_steps_total:,} | "
                                    f"全局步数: {env_steps:,} / {display_global_steps_total:,} | "
                                    f"FPS: {fps:,.0f} | LR: {lr:.3e}",
                                    "=" * 118,
                                    make_table("time / progress", stat),
                                    make_table("env info: rewards + events + telemetry + world + debug", flat),
                                    make_table("ppo update info", ppo_info_to_print),
                                    "=" * 118 + "\n",
                                ]
                            )
                        )
                        prev_summary_policy_vec = policy_vec.clone()
                        prev_summary_value_vec = value_vec.clone()
                        ppo_summary_buffer.clear()

                    try:
                        agent.tracking_data.clear()
                    except Exception:
                        pass

                if train_env_steps_done - last_save >= save_freq_env_steps:
                    last_save = train_env_steps_done
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
                        pbar.write(f"\n💾 [Diff-Drive Task2 skrl 备份] 总步数: {env_steps:,} | 已保存至: {save_dir}\n")
                    except Exception as exc:
                        pbar.write(f"\n[WARN] checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

    except KeyboardInterrupt:
        print("\n[WARN] 接收到 Ctrl+C，正在保存当前 Diff-Drive Task2 skrl 模型...")
    except Exception:
        print("\n[ERROR] Diff-Drive Task2 skrl PPO 训练过程中发生真实异常：")
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
            print(f"✅ Diff-Drive Task2 skrl 模型已保存至 {final_dir}")
        except Exception as exc:
            print(f"[WARN] 保存最终 skrl 模型失败: {type(exc).__name__}: {exc}")

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

        print("✅ Diff-Drive Task2 skrl PPO training pipeline safely exited")


if __name__ == "__main__":
    main()
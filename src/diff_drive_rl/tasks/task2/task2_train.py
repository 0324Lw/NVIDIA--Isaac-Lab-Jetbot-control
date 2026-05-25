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

parser = argparse.ArgumentParser(description="Train Diff-Drive UGV / Jetbot Task2 with TRUE skrl PPO")

# Runtime
parser.add_argument("--total-env-steps", type=int, default=250_000_000)
parser.add_argument("--save-freq-env-steps", type=int, default=10_000_000)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default="", help="Optional skrl checkpoint or final_checkpoint directory")

# Curriculum / env
parser.add_argument("--start-k", type=float, default=0.0)
parser.add_argument("--max-episode-length-s", type=float, default=80.0)

# PPO
parser.add_argument("--rollouts", type=int, default=64)
parser.add_argument("--learning-epochs", type=int, default=4)
parser.add_argument("--mini-batches", type=int, default=8)

parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--min-lr", type=float, default=5e-5)
parser.add_argument("--max-lr", type=float, default=5e-4)

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
parser.add_argument("--summary-interval", type=int, default=1)
parser.add_argument("--skrl-write-interval", type=int, default=1_000_000)
parser.add_argument("--skrl-checkpoint-interval", type=int, default=0)

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

        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_space.shape[0]),
        )

        self.log_std_parameter = nn.Parameter(torch.full((action_space.shape[0],), float(init_log_std)))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)

    def compute(self, inputs, role):
        states = inputs.get("observations", inputs.get("states"))
        actions = self.net(states)
        return actions, {"log_std": self.log_std_parameter}


class DiffDriveTask2Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = nn.Sequential(
            nn.Linear(state_space.shape[0], 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.apply(DiffDriveTask2Actor._init_weights)

    def compute(self, inputs, role):
        states = inputs.get("states", None)
        if states is None:
            states = inputs.get("observations", None)
        if states is None:
            raise RuntimeError("Critic received no states / observations.")
        return self.net(states), {}


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
                "robot": "Jetbot / two-wheel differential-drive UGV",
                "task": "task2_analytic_obstacle_navigation",
                "algorithm": "skrl_PPO",
                "uses_skrl": True,
                "asymmetric_actor_critic": False,
                "policy_input": "obs_3_frame_stack",
                "critic_input": "same_as_policy",
                "single_obs_dim": int(env_cfg.single_obs_dim),
                "frame_stack": int(env_cfg.frame_stack),
                "actor_obs_dim": int(env_cfg.num_observations),
                "critic_obs_dim": int(env_cfg.num_observations),
                "action_dim": int(env_cfg.num_actions),
                "num_lidar_rays": int(env_cfg.world_cfg.num_lidar_rays),
                "max_static_obs": int(env_cfg.world_cfg.max_static_obs),
                "max_dynamic_obs": int(env_cfg.world_cfg.max_dynamic_obs),
                "world": "analytic_gpu_world",
                "control": "left_right_wheel_velocity",
                "note": "TRUE skrl PPO checkpoint. Evaluation uses deterministic policy forward, not agent.act.",
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
            "env_cfg": {
                "num_observations": int(env_cfg.num_observations),
                "single_obs_dim": int(env_cfg.single_obs_dim),
                "frame_stack": int(env_cfg.frame_stack),
                "num_actions": int(env_cfg.num_actions),
                "max_episode_length": int(env_cfg.max_episode_length),
                "world": {
                    "num_lidar_rays": int(env_cfg.world_cfg.num_lidar_rays),
                    "max_static_obs": int(env_cfg.world_cfg.max_static_obs),
                    "max_dynamic_obs": int(env_cfg.world_cfg.max_dynamic_obs),
                    "curriculum_total_steps": int(env_cfg.world_cfg.curriculum_total_steps),
                },
            },
        },
        os.path.join(directory, "task2_train_metadata.pt"),
    )

    print(f"💾 [Diff-Drive Task2 skrl checkpoint] saved to: {directory}", flush=True)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    set_seed(int(args_cli.seed))

    run_name = make_run_name()
    log_dir = os.path.abspath(args_cli.log_root)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 118)
    print("🚀 Diff-Drive UGV / Jetbot Task2 Analytic Obstacle Navigation - TRUE skrl PPO Training")
    print("=" * 118)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_root     = {log_dir}")
    print(f"[INFO] run_name     = {run_name}")
    print("[INFO] This version uses skrl PPO, not torch-native PPO or stable-baselines3.")

    env_cfg = Task2Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.device)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.max_episode_length_s = float(args_cli.max_episode_length_s)
    env_cfg.print_debug_info = False
    env_cfg.validate()

    if float(args_cli.start_k) > 0.0:
        initial_global_steps = int(float(args_cli.start_k) * env_cfg.world_cfg.curriculum_total_steps)
    else:
        initial_global_steps = 0

    base_env = DiffDriveTask2Env(env_cfg)
    base_env.global_steps = int(initial_global_steps)

    if initial_global_steps > 0:
        print(
            f"[INFO] start_k={args_cli.start_k:.4f}, "
            f"initial_global_steps={initial_global_steps:,}, "
            f"stage={base_env.world.stage_from_global_steps(initial_global_steps)}"
        )

    local_env = DiffDriveTask2SkrlWrapper(base_env)
    env = wrap_env(local_env, wrapper="isaaclab")
    num_envs = getattr(env, "num_envs", local_env.num_envs)

    if env.state_space is None:
        raise RuntimeError("env.state_space is None. Task2 requires critic state space, even if same as actor obs.")

    print("\n[DEBUG] Diff-Drive Task2 Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  single_obs_dim        = {env_cfg.single_obs_dim}")
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

    resume_ckpt = resolve_resume_checkpoint(args_cli.resume)
    if resume_ckpt:
        if os.path.exists(resume_ckpt):
            print(f"[INFO] Loading skrl agent checkpoint: {resume_ckpt}")
            agent.load(resume_ckpt)
        else:
            print(f"[WARN] resume checkpoint not found: {resume_ckpt}")

    total_env_steps = int(args_cli.total_env_steps)
    total_vector_steps = math.ceil(total_env_steps / int(num_envs))
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
    print(f"  - total_env_steps       : {total_env_steps:,}")
    print(f"  - total_vector_steps    : {total_vector_steps:,}")
    print(f"  - update_env_steps      : {update_env_steps:,}")
    print(f"  - save_freq_env_steps   : {save_freq_env_steps:,}")
    print(f"  - start_k               : {args_cli.start_k:.4f}")
    print(f"  - curriculum_total_steps: {env_cfg.world_cfg.curriculum_total_steps:,}")
    print(f"  - single_obs_dim        : {env_cfg.single_obs_dim}")
    print(f"  - frame_stack           : {env_cfg.frame_stack}")
    print(f"  - actor_obs_dim         : {env.observation_space.shape[0]}")
    print(f"  - critic_obs_dim        : {env.state_space.shape[0]}")
    print(f"  - action_dim            : {env.action_space.shape[0]}")
    print(f"  - lidar_rays            : {env_cfg.world_cfg.num_lidar_rays}")
    print(f"  - max_static_obs        : {env_cfg.world_cfg.max_static_obs}")
    print(f"  - max_dynamic_obs       : {env_cfg.world_cfg.max_dynamic_obs}")
    print(f"  - max_episode_length    : {env_cfg.max_episode_length}")
    print(f"  - rollouts              : {cfg['rollouts']}")
    print(f"  - learning_epochs       : {cfg.get('learning_epochs')}")
    print(f"  - mini_batches          : {cfg.get('mini_batches')}")
    print(f"  - lr                    : {cfg.get('learning_rate')}")
    print(f"  - tensorboard           : tensorboard --logdir={log_dir}")

    print("\n🔥 [Diff-Drive Task2 TRUE skrl PPO 已点火]")
    print("👉 任务目标：两轮差速无人车在解析障碍世界中导航到目标点")
    print("👉 Actor/Critic 输入：3 帧堆叠观测，498 维")
    print("👉 动作：2 维左右轮速度控制")
    print("👉 世界层：analytic GPU world + LiDAR + risk features")
    print("👉 日志重点：Progress / Goal_Aligned_Speed / Goal_Dist / Success_Rate / Collision_Rate / Stuck_Ratio\n")

    last_save = 0
    update_id = 0
    start_time = time.time()
    env_steps = int(initial_global_steps)

    try:
        trainer.reset()

        with tqdm(
            total=total_env_steps,
            initial=min(env_steps, total_env_steps),
            desc="Diff-Drive Task2 skrl PPO",
            unit="steps",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.05,
        ) as pbar:
            for t in range(total_vector_steps):
                trainer.train(timestep=t, timesteps=total_vector_steps)

                env_steps = min(initial_global_steps + (t + 1) * int(num_envs), total_env_steps)
                previous_env_steps = min(initial_global_steps + t * int(num_envs), total_env_steps)
                pbar.update(max(env_steps - previous_env_steps, 0))

                flat = flat_dict(local_env.last_info)
                elapsed = time.time() - start_time
                fps = max(env_steps - initial_global_steps, 0) / max(elapsed, 1e-6)

                pbar.set_postfix(
                    {
                        "steps": f"{env_steps:,}",
                        "fps": f"{fps:,.0f}",
                        "rew": f"{local_env.last_reward_mean:+.3f}",
                        "done": local_env.last_done_count,
                        "stage": f"{flat.get('telemetry/Stage', 0.0):.1f}",
                        "dist": f"{flat.get('telemetry/Goal_Dist', 0.0):.2f}",
                        "prog": f"{flat.get('telemetry/Progress', 0.0):+.3f}",
                        "goal_v": f"{flat.get('telemetry/Goal_Aligned_Speed', 0.0):+.2f}",
                        "risk": f"{flat.get('telemetry/Risk_Front', 0.0):.2f}",
                        "succ": f"{flat.get('events/Episode_Success_Rate', 0.0):.3f}",
                        "coll": f"{flat.get('events/Episode_Collision_Rate', 0.0):.3f}",
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

                    write_scalars(writer, ppo_info, env_steps, "ppo")
                    write_scalars(writer, flat, env_steps, "env_info")

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
                                    f"📊 [Diff-Drive Task2 skrl PPO 更新 {update_id}] "
                                    f"总步数: {env_steps:,} / {total_env_steps:,} | "
                                    f"FPS: {fps:,.0f} | LR: {lr:.3e}",
                                    "=" * 118,
                                    make_table("time / progress", stat),
                                    make_table("env info: rewards + events + telemetry + world + debug", flat),
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

        print("✅ Diff-Drive Task2 TRUE skrl PPO training pipeline safely exited")


if __name__ == "__main__":
    main()

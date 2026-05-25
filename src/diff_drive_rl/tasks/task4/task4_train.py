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
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.getLogger("isaaclab.assets.articulation").setLevel(logging.ERROR)
logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Train Diff-Drive UGV / Jetbot Task4 Multi-UGV Formation Escort with TRUE skrl PPO"
)

# Env / run
parser.add_argument("--num-envs", type=int, default=512, help="physical parallel environments. skrl agent envs = num_envs * 4")
parser.add_argument("--total-agent-steps", type=int, default=300_000_000, help="total skrl agent transitions = physical_envs * 4 * vector_steps")
parser.add_argument("--save-freq-agent-steps", type=int, default=10_000_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")

# checkpoint
parser.add_argument("--resume", type=str, default="", help="skrl checkpoint path to resume training")
parser.add_argument("--pretrained", type=str, default="", help="optional skrl checkpoint path for initialization")
parser.add_argument("--start-agent-steps", type=int, default=0, help="absolute agent steps when resuming")

# PPO
parser.add_argument("--rollouts", type=int, default=64)
parser.add_argument("--learning-epochs", type=int, default=5)
parser.add_argument("--mini-batches", type=int, default=8)
parser.add_argument("--lr", type=float, default=8.0e-5)
parser.add_argument("--min-lr", type=float, default=2.0e-5)
parser.add_argument("--max-lr", type=float, default=1.2e-4)
parser.add_argument("--discount-factor", type=float, default=0.995)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--kl-threshold", type=float, default=0.012)
parser.add_argument("--ratio-clip", type=float, default=0.18)
parser.add_argument("--value-clip", type=float, default=0.20)
parser.add_argument("--entropy-loss-scale", type=float, default=0.004)
parser.add_argument("--value-loss-scale", type=float, default=1.0)
parser.add_argument("--grad-norm-clip", type=float, default=0.5)

# Policy distribution
parser.add_argument("--init-log-std", type=float, default=-0.85)
parser.add_argument("--min-log-std", type=float, default=-3.0)
parser.add_argument("--max-log-std", type=float, default=0.5)

# Env overrides
parser.add_argument("--max-episode-length-s", type=float, default=35.0)
parser.add_argument("--max-wheel-speed", type=float, default=32.0)
parser.add_argument("--reverse-speed-fraction", type=float, default=0.35)
parser.add_argument("--left-wheel-sign", type=float, default=1.0)
parser.add_argument("--right-wheel-sign", type=float, default=1.0)

# Curriculum
parser.add_argument("--disable-curriculum", action="store_true")
parser.add_argument("--fixed-stage", type=int, default=-1, help="0~5 fixed stage; -1 = automatic curriculum")
parser.add_argument("--curriculum-debug", action="store_true")

# Debug / safety
parser.add_argument("--print-debug-info", action="store_true")
parser.add_argument("--disable-physical-asset-teleport", action="store_true")
parser.add_argument("--log-interval-updates", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
except ImportError:
    try:
        from skrl.resources.schedulers.torch import KLAdaptiveRL as KLAdaptiveLR
    except ImportError:
        KLAdaptiveLR = None

from diff_drive_rl.tasks.task4.task4_config import Task4Config
from diff_drive_rl.tasks.task4.task4_env import DiffDriveTask4Env


# ======================================================================
# Generic utilities
# ======================================================================

def to_float(x: Any):
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
        if isinstance(x, (list, tuple)):
            return float(np.mean(x)) if len(x) else None
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
    except Exception:
        return None
    return None


def is_bad_number(x) -> bool:
    v = to_float(x)
    if v is None:
        return False
    return not math.isfinite(v)


def flat_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        name = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flat_dict(v, name))
        else:
            val = to_float(v)
            if val is not None:
                out[name] = val
    return out


def tracking_mean(agent) -> Dict[str, float]:
    out: Dict[str, float] = {}

    for k, v in getattr(agent, "tracking_data", {}).items():
        if v is None:
            continue

        try:
            if len(v) == 0:
                continue
        except Exception:
            pass

        try:
            arr = np.asarray(v, dtype=np.float64)
            if arr.size == 0:
                continue
            if k.endswith("(min)"):
                out[k] = float(np.min(arr))
            elif k.endswith("(max)"):
                out[k] = float(np.max(arr))
            else:
                out[k] = float(np.mean(arr))
        except Exception:
            val = to_float(v)
            if val is not None:
                out[k] = val

    return out


def current_lr(agent) -> float:
    for obj in [
        getattr(agent, "optimizer", None),
        getattr(getattr(agent, "scheduler", None), "optimizer", None),
    ]:
        try:
            if obj is not None:
                return float(obj.param_groups[0]["lr"])
        except Exception:
            pass
    return float("nan")


def write_scalars(writer, data, step, prefix: str):
    if writer is None:
        return

    for k, v in (data or {}).items():
        val = to_float(v)
        if val is not None and math.isfinite(val):
            try:
                writer.add_scalar(f"{prefix}/{k}".replace("//", "/"), val, step)
            except Exception:
                pass


def make_table(title: str, data: Dict[str, Any], width: int = 124) -> str:
    lines = [
        "-" * width,
        f"| {title:<{width - 4}} |",
        "-" * width,
    ]

    if not data:
        lines += [f"| {'<empty>':<{width - 4}} |", "-" * width]
        return "\n".join(lines)

    for k in sorted(data.keys()):
        v = data[k]
        ks = (k[:80] + "...") if len(k) > 83 else k

        if isinstance(v, float):
            if math.isnan(v):
                vs = "nan"
            elif math.isinf(v):
                vs = "inf"
            else:
                vs = f"{v:.6e}" if abs(v) > 1e4 or 0 < abs(v) < 1e-3 else f"{v:.6f}"
        else:
            vs = str(v)

        vs = (vs[:34] + "...") if len(vs) > 37 else vs
        lines.append(f"| {ks:<83} | {vs:>{width - 90}} |")

    lines.append("-" * width)
    return "\n".join(lines)


def resolve_checkpoint_path(path: str) -> str:
    if not path:
        return ""

    if os.path.isdir(path):
        for name in [
            "jetbot_task4_skrl_model.pt",
            "diff_drive_task4_model.pt",
            "agent.pt",
            "model.pt",
            "checkpoint.pt",
        ]:
            p = os.path.join(path, name)
            if os.path.exists(p):
                return p

    return path


def try_load_agent(agent, path: str, label: str) -> bool:
    path = resolve_checkpoint_path(path)
    if not path:
        return False
    if not os.path.exists(path):
        print(f"[WARN] {label} checkpoint 不存在: {path}")
        return False

    print("\n" + "=" * 108)
    print(f"🔁 尝试加载 {label}: {path}")
    print("=" * 108)

    try:
        agent.load(path)
        print(f"✅ 已通过 agent.load() 成功加载 {label}")
        return True
    except Exception as exc:
        print(f"[WARN] agent.load() 加载 {label} 失败: {type(exc).__name__}: {exc}")
        return False


def sanitize_tensor_inplace(x: torch.Tensor, nan=0.0, posinf=1.0, neginf=-1.0, clamp_abs=None) -> None:
    if x is None or not torch.is_tensor(x):
        return

    with torch.no_grad():
        x.data = torch.nan_to_num(x.data, nan=nan, posinf=posinf, neginf=neginf)
        if clamp_abs is not None:
            x.data.clamp_(-float(clamp_abs), float(clamp_abs))


def sanitize_agent_numerics(agent, models: Dict[str, nn.Module], min_log_std=-3.0, max_log_std=0.5) -> None:
    for _, model in models.items():
        for p in model.parameters():
            sanitize_tensor_inplace(p, nan=0.0, posinf=1.0, neginf=-1.0, clamp_abs=20.0)

        if hasattr(model, "log_std_parameter"):
            with torch.no_grad():
                model.log_std_parameter.data = torch.nan_to_num(
                    model.log_std_parameter.data,
                    nan=float(args_cli.init_log_std),
                    posinf=float(max_log_std),
                    neginf=float(min_log_std),
                )
                model.log_std_parameter.data.clamp_(float(min_log_std), float(max_log_std))

    opt = getattr(agent, "optimizer", None)
    if opt is not None:
        for state in opt.state.values():
            for _, v in state.items():
                if torch.is_tensor(v):
                    with torch.no_grad():
                        v.data = torch.nan_to_num(v.data, nan=0.0, posinf=1.0, neginf=-1.0)
                        v.data.clamp_(-100.0, 100.0)


def ppo_info_has_nan(ppo_info: Dict[str, Any]) -> Tuple[bool, str]:
    keys_to_check = [
        "Loss / Entropy loss",
        "Loss / Policy loss",
        "Loss / Value loss",
        "Policy / Standard deviation",
        "Learning / Learning rate",
        "learning_rate",
    ]

    for k in keys_to_check:
        if k in ppo_info and is_bad_number(ppo_info[k]):
            return True, k

    for k, v in ppo_info.items():
        if "Loss" in k or "Standard deviation" in k or "Learning" in k:
            if is_bad_number(v):
                return True, k

    return False, ""


def save_normalizers(agent, save_dir: str) -> None:
    names = [
        "observation_preprocessor",
        "state_preprocessor",
        "value_preprocessor",
        "_observation_preprocessor",
        "_state_preprocessor",
        "_value_preprocessor",
    ]

    for name in names:
        obj = getattr(agent, name, None)
        if obj is not None:
            try:
                torch.save(obj.state_dict(), os.path.join(save_dir, f"{name}.pt"))
            except Exception:
                pass


def extract_preprocessor_state(agent, names):
    for name in names:
        obj = getattr(agent, name, None)
        if obj is not None:
            try:
                return obj.state_dict()
            except Exception:
                pass
    return None


def save_training_metadata(
    path: str,
    env_cfg: Task4Config,
    base_env: DiffDriveTask4Env,
    env,
    args,
    agent_steps: int,
    curriculum_info: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    try:
        cfg_dict = dataclasses.asdict(env_cfg)
    except Exception:
        cfg_dict = {}

    metadata = {
        "stage": "diff_drive_task4_multi_ugv_formation_escort_skrl_curriculum",
        "uses_skrl": True,
        "algorithm": "skrl PPO",
        "ctde_mappo_style": True,
        "global_agent_steps": int(agent_steps),
        "physical_num_envs": int(base_env.num_envs),
        "skrl_num_envs": int(env.num_envs),
        "num_agents": int(base_env.num_agents),
        "actor_obs_dim": int(env.observation_space.shape[0]),
        "critic_obs_dim": int(env.state_space.shape[0]),
        "base_world_priv_dim": int(base_env.world_priv_dim),
        "single_actor_obs_dim": int(base_env.cfg.single_actor_obs_dim),
        "frame_stack": int(base_env.cfg.frame_stack),
        "action_dim": int(env.action_space.shape[0]),
        "max_episode_length_s": float(env_cfg.max_episode_length_s),
        "max_episode_length": int(env_cfg.max_episode_length),
        "policy_dt": float(env_cfg.policy_dt),
        "max_wheel_speed": float(env_cfg.max_wheel_speed),
        "curriculum_info": curriculum_info or {},
        "args": vars(args),
        "env_cfg": cfg_dict,
        "extra": extra or {},
    }

    torch.save(metadata, os.path.join(path, "train_metadata.pt"))


def save_eval_checkpoint(
    path: str,
    agent,
    models: Dict[str, nn.Module],
    env_cfg: Task4Config,
    base_env: DiffDriveTask4Env,
    env,
    args,
    agent_steps: int,
    curriculum_info: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """Save a lightweight eval checkpoint for task4_model_test.py."""

    obs_norm = extract_preprocessor_state(
        agent,
        ["observation_preprocessor", "_observation_preprocessor"],
    )
    state_norm = extract_preprocessor_state(
        agent,
        ["state_preprocessor", "_state_preprocessor"],
    )
    value_norm = extract_preprocessor_state(
        agent,
        ["value_preprocessor", "_value_preprocessor"],
    )

    ckpt = {
        "policy": models["policy"].state_dict(),
        "value": models["value"].state_dict(),
        "actor_obs_norm": obs_norm,
        "critic_obs_norm": state_norm,
        "value_norm": value_norm,
        "agent_steps": int(agent_steps),
        "curriculum_info": curriculum_info or {},
        "args": vars(args),
        "metadata": {
            "uses_skrl": True,
            "algorithm": "skrl PPO",
            "ctde_mappo_style": True,
            "shared_actor": True,
            "centralized_critic": True,
            "physical_num_envs": int(base_env.num_envs),
            "skrl_num_envs": int(env.num_envs),
            "num_agents": int(base_env.num_agents),
            "actor_obs_dim": int(env.observation_space.shape[0]),
            "critic_obs_dim": int(env.state_space.shape[0]),
            "base_world_priv_dim": int(base_env.world_priv_dim),
            "single_actor_obs_dim": int(base_env.cfg.single_actor_obs_dim),
            "frame_stack": int(base_env.cfg.frame_stack),
            "action_dim": int(env.action_space.shape[0]),
            "lidar_pool_bins": int(base_env.cfg.world_cfg.lidar_pool_bins),
            "risk_feature_dim": int(base_env.world.risk_feature_dim()),
            "world_priv_dim": int(base_env.world_priv_dim),
            "max_episode_length_s": float(env_cfg.max_episode_length_s),
            "max_episode_length": int(env_cfg.max_episode_length),
            "policy_dt": float(env_cfg.policy_dt),
            "extra": extra or {},
        },
    }

    torch.save(ckpt, os.path.join(path, "diff_drive_task4_model.pt"))


def save_all_checkpoints(
    save_dir: str,
    agent,
    models: Dict[str, nn.Module],
    env_cfg: Task4Config,
    base_env: DiffDriveTask4Env,
    env,
    args,
    agent_steps: int,
    curriculum_info: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    sanitize_agent_numerics(agent, models, args.min_log_std, args.max_log_std)

    agent.save(os.path.join(save_dir, "jetbot_task4_skrl_model.pt"))
    save_normalizers(agent, save_dir)
    save_training_metadata(
        path=save_dir,
        env_cfg=env_cfg,
        base_env=base_env,
        env=env,
        args=args,
        agent_steps=agent_steps,
        curriculum_info=curriculum_info,
        extra=extra,
    )
    save_eval_checkpoint(
        path=save_dir,
        agent=agent,
        models=models,
        env_cfg=env_cfg,
        base_env=base_env,
        env=env,
        args=args,
        agent_steps=agent_steps,
        curriculum_info=curriculum_info,
        extra=extra,
    )


# ======================================================================
# Curriculum
# ======================================================================

class Task4FormationCurriculum:
    """Task4 multi-UGV formation escort curriculum.

    Stage 0:
        Clean formation navigation. Diamond, no obstacle, no gate.

    Stage 1:
        Mild obstacles. 2 obstacles, no gate.

    Stage 2:
        More obstacles + Diamond/Wedge. 4 obstacles.

    Stage 3:
        Narrow gate + compressed Line formation. 5 obstacles.

    Stage 4:
        Random formations + gate + 7 obstacles.

    Stage 5:
        Full domain randomization + 8 obstacles.
    """

    def __init__(self, total_agent_steps: int, disabled: bool = False, fixed_stage: int = -1):
        self.total_agent_steps = max(int(total_agent_steps), 1)
        self.disabled = bool(disabled)
        self.fixed_stage = int(fixed_stage)
        self.last_stage = -1

    def stage_from_progress(self, k: float) -> int:
        if self.fixed_stage >= 0:
            return int(max(0, min(5, self.fixed_stage)))
        if self.disabled:
            return 5
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

    def apply(self, env, agent_steps: int) -> Dict[str, float]:
        cfg = env.cfg
        wcfg = cfg.world_cfg

        k = float(agent_steps) / float(self.total_agent_steps)
        k = max(0.0, min(1.0, k))
        stage = self.stage_from_progress(k)

        # Physical geometry stays fixed. Curriculum changes sampling ranges,
        # randomization ranges and reward weights.
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
            cfg.w_team_heading = 0.17
            cfg.w_formation_mean = 0.99
            cfg.w_formation_agent = 0.45
            cfg.w_team_spread = 0.14
            cfg.w_speed_sync = 0.09
            cfg.w_gate_pass = 0.0
            cfg.w_front_clearance = 0.035
            cfg.w_obstacle_risk = 0.04
            cfg.w_gate_risk = 0.00
            cfg.w_boundary_risk = 0.07
            cfg.w_pair_risk = 0.38
            cfg.w_stuck = 0.08
            cfg.w_step = 0.004
            cfg.w_action_smooth = 0.012
            cfg.w_action_mag = 0.0015
            cfg.w_spin = 0.018

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
            cfg.w_team_heading = 0.17
            cfg.w_formation_mean = 0.55
            cfg.w_formation_agent = 0.28
            cfg.w_team_spread = 0.08
            cfg.w_speed_sync = 0.08
            cfg.w_gate_pass = 0.0
            cfg.w_front_clearance = 0.045
            cfg.w_obstacle_risk = 0.12
            cfg.w_gate_risk = 0.00
            cfg.w_boundary_risk = 0.10
            cfg.w_pair_risk = 0.28
            cfg.w_stuck = 0.08
            cfg.w_step = 0.0035
            cfg.w_action_smooth = 0.012
            cfg.w_spin = 0.020

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
            cfg.w_team_heading = 0.18
            cfg.w_formation_mean = 0.65
            cfg.w_formation_agent = 0.32
            cfg.w_team_spread = 0.09
            cfg.w_speed_sync = 0.09
            cfg.w_gate_pass = 0.0
            cfg.w_front_clearance = 0.055
            cfg.w_obstacle_risk = 0.22
            cfg.w_gate_risk = 0.00
            cfg.w_boundary_risk = 0.14
            cfg.w_pair_risk = 0.32
            cfg.w_stuck = 0.07
            cfg.w_step = 0.0032
            cfg.w_action_smooth = 0.014
            cfg.w_spin = 0.020

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
            cfg.w_team_heading = 0.18
            cfg.w_formation_mean = 0.72
            cfg.w_formation_agent = 0.36
            cfg.w_team_spread = 0.10
            cfg.w_speed_sync = 0.10
            cfg.w_gate_pass = 1.20
            cfg.w_front_clearance = 0.060
            cfg.w_obstacle_risk = 0.26
            cfg.w_gate_risk = 0.22
            cfg.w_boundary_risk = 0.18
            cfg.w_pair_risk = 0.35
            cfg.w_stuck = 0.065
            cfg.w_step = 0.0030
            cfg.w_action_smooth = 0.016
            cfg.w_spin = 0.021

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
            cfg.w_team_heading = 0.18
            cfg.w_formation_mean = 0.76
            cfg.w_formation_agent = 0.38
            cfg.w_team_spread = 0.11
            cfg.w_speed_sync = 0.10
            cfg.w_gate_pass = 1.10
            cfg.w_front_clearance = 0.065
            cfg.w_obstacle_risk = 0.30
            cfg.w_gate_risk = 0.25
            cfg.w_boundary_risk = 0.20
            cfg.w_pair_risk = 0.36
            cfg.w_stuck = 0.060
            cfg.w_step = 0.0030
            cfg.w_action_smooth = 0.018
            cfg.w_spin = 0.022

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
            cfg.w_team_heading = 0.18
            cfg.w_formation_mean = 0.78
            cfg.w_formation_agent = 0.40
            cfg.w_team_spread = 0.12
            cfg.w_speed_sync = 0.11
            cfg.w_gate_pass = 1.00
            cfg.w_front_clearance = 0.070
            cfg.w_obstacle_risk = 0.32
            cfg.w_gate_risk = 0.28
            cfg.w_boundary_risk = 0.22
            cfg.w_pair_risk = 0.38
            cfg.w_stuck = 0.055
            cfg.w_step = 0.0030
            cfg.w_action_smooth = 0.020
            cfg.w_spin = 0.024

        if hasattr(env, "set_curriculum_stage"):
            env.set_curriculum_stage(stage)
        elif hasattr(env, "world"):
            env.world.set_curriculum_stage(stage)
            env.curriculum_stage = stage
            env.cfg.curriculum_stage = stage

        changed = stage != self.last_stage
        self.last_stage = stage

        info = {
            "Curriculum_K": float(k),
            "Curriculum_Stage": float(stage),
            "Curriculum_Changed": float(changed),
            "Fixed_Stage": float(self.fixed_stage),
            "Physical_Stage": float(stage),
            "Start_Y_Range": float(wcfg.start_center_y_range[1]),
            "Start_Heading_Range": float(wcfg.start_heading_range[1]),
            "Goal_Y_Range": float(wcfg.goal_y_range[1]),
            "Formation_Scale_Low": float(wcfg.formation_scale_range[0]),
            "Formation_Scale_High": float(wcfg.formation_scale_range[1]),
            "Obstacle_Count_Expected": float(wcfg.obstacle_count_by_stage[stage]),
            "Gate_Expected": float(stage >= wcfg.gate_stage_start),
            "Max_Speed_Low": float(wcfg.max_speed_range[0]),
            "Max_Speed_High": float(wcfg.max_speed_range[1]),
            "Max_Yaw_Rate_High": float(wcfg.max_yaw_rate_range[1]),
            "Delay_Max": float(wcfg.action_delay_frame_range[1]),
            "Deadband_Max": float(wcfg.action_deadband_range[1]),
            "EMA_Min": float(wcfg.action_ema_alpha_range[0]),
            "Motor_Strength_Low": float(wcfg.motor_strength_range[0]),
            "Motor_Strength_High": float(wcfg.motor_strength_range[1]),
            "Wheel_Scale_Low": float(wcfg.wheel_radius_scale_range[0]),
            "Wheel_Scale_High": float(wcfg.wheel_radius_scale_range[1]),
            "Lidar_Noise_Max": float(wcfg.lidar_noise_std_range[1]),
            "Lidar_Outlier_Max": float(wcfg.lidar_outlier_prob_range[1]),
            "Lidar_Dropout_Max": float(wcfg.lidar_dropout_prob_range[1]),
            "Reward_w_team_progress": float(cfg.w_team_progress),
            "Reward_w_center_speed": float(cfg.w_center_speed),
            "Reward_w_formation_mean": float(cfg.w_formation_mean),
            "Reward_w_formation_agent": float(cfg.w_formation_agent),
            "Reward_w_gate_pass": float(cfg.w_gate_pass),
            "Reward_w_obstacle_risk": float(cfg.w_obstacle_risk),
            "Reward_w_gate_risk": float(cfg.w_gate_risk),
            "Reward_w_pair_risk": float(cfg.w_pair_risk),
            "Reward_w_stuck": float(cfg.w_stuck),
            "Reward_w_action_smooth": float(cfg.w_action_smooth),
        }

        if changed:
            print("\n" + "=" * 120)
            print(f"🎓 [Task4 Curriculum] 切换到 Stage {stage} | K={k:.4f} | agent_steps={agent_steps:,}")
            for key in [
                "Obstacle_Count_Expected",
                "Gate_Expected",
                "Start_Y_Range",
                "Goal_Y_Range",
                "Formation_Scale_Low",
                "Formation_Scale_High",
                "Max_Speed_High",
                "Delay_Max",
                "Deadband_Max",
                "EMA_Min",
                "Lidar_Noise_Max",
                "Reward_w_team_progress",
                "Reward_w_formation_mean",
                "Reward_w_obstacle_risk",
                "Reward_w_gate_risk",
                "Reward_w_pair_risk",
            ]:
                print(f"  - {key:<30s}: {info[key]}")
            print("=" * 120 + "\n")

        return info


# ======================================================================
# skrl wrapper
# ======================================================================

class DiffDriveTask4SkrlWrapper(gym.Env):
    """Flatten Task4 multi-agent env into a skrl vector env.

    physical env:
        obs   [N, 4, 624]
        state [N, 96]
        act   [N, 4, 2]
        rew   [N, 4]
        done  [N]

    skrl env:
        policy obs [N * 4, 624]
        critic obs [N * 4, 100] = world state 96 + agent id 4
        action     [N * 4, 2]
        reward     [N * 4]
        done       [N * 4]
    """

    metadata = {"render_modes": []}

    def __init__(self, env: DiffDriveTask4Env, log_dir: str):
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

        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": self.observation_space,
                "critic": self.state_space,
            }
        )
        self.single_action_space = self.action_space

        self.agent_id_onehot_flat = torch.eye(
            self.num_agents,
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.physical_num_envs, 1)

        self.writer = SummaryWriter(log_dir)
        self.global_agent_steps = 0
        self.global_physical_steps = 0
        self.last_info: Dict[str, Any] = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0
        self.nan_action_count = 0

    @property
    def unwrapped(self):
        return self

    def _flatten_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.reshape(self.num_envs, self.policy_obs_dim)

    def _flatten_state(self, state: torch.Tensor) -> torch.Tensor:
        repeated_state = state.repeat_interleave(self.num_agents, dim=0)
        return torch.cat([repeated_state, self.agent_id_onehot_flat], dim=-1)

    def reset(self, seed=None, options=None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options)
        state = info.get("state", self.env.compute_privileged_obs())

        obs_flat = self._flatten_obs(obs)
        state_flat = self._flatten_state(state)

        obs_flat = torch.nan_to_num(
            obs_flat,
            nan=0.0,
            posinf=self.env.cfg.obs_clip,
            neginf=-self.env.cfg.obs_clip,
        )
        state_flat = torch.nan_to_num(
            state_flat,
            nan=0.0,
            posinf=self.env.cfg.priv_clip,
            neginf=-self.env.cfg.priv_clip,
        )

        self.last_info = info or {}
        return {"policy": obs_flat.clone(), "critic": state_flat.clone()}, info

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)

        if action.dim() == 3 and action.shape[:2] == (self.physical_num_envs, self.num_agents):
            action_env = action
        else:
            action_env = action.view(self.physical_num_envs, self.num_agents, self.action_dim)

        if not torch.isfinite(action_env).all():
            bad_ratio = (~torch.isfinite(action_env)).float().mean().item()
            self.nan_action_count += 1
            print(
                f"[WARN][Task4Train] NaN/Inf action detected, "
                f"bad_ratio={bad_ratio:.6f}, replace with safe action",
                flush=True,
            )
            action_env = torch.nan_to_num(action_env, nan=0.0, posinf=1.0, neginf=-1.0)

        action_env = torch.clamp(action_env, -1.0, 1.0)

        obs, reward, terminated, truncated, info = self.env.step(action_env)
        state = info.get("state", self.env.compute_privileged_obs())

        obs_flat = self._flatten_obs(obs)
        state_flat = self._flatten_state(state)
        reward_flat = reward.reshape(self.num_envs)
        terminated_flat = terminated.repeat_interleave(self.num_agents, dim=0)
        truncated_flat = truncated.repeat_interleave(self.num_agents, dim=0)

        obs_flat = torch.nan_to_num(
            obs_flat,
            nan=0.0,
            posinf=self.env.cfg.obs_clip,
            neginf=-self.env.cfg.obs_clip,
        )
        state_flat = torch.nan_to_num(
            state_flat,
            nan=0.0,
            posinf=self.env.cfg.priv_clip,
            neginf=-self.env.cfg.priv_clip,
        )
        reward_flat = torch.nan_to_num(reward_flat, nan=0.0, posinf=10.0, neginf=-10.0)
        reward_flat = torch.clamp(reward_flat, -100.0, 100.0)

        done_flat = terminated_flat | truncated_flat

        self.global_physical_steps += self.physical_num_envs
        self.global_agent_steps += self.num_envs
        self.last_info = info or {}
        self.last_reward_mean = to_float(reward_flat) or 0.0
        self.last_done_count = int(done_flat.sum().detach().cpu().item())

        write_scalars(self.writer, self.last_info.get("reward_components", {}), self.global_agent_steps, "rewards")
        write_scalars(self.writer, self.last_info.get("events", {}), self.global_agent_steps, "events")
        write_scalars(self.writer, self.last_info.get("telemetry", {}), self.global_agent_steps, "telemetry")
        write_scalars(self.writer, self.last_info.get("world", {}), self.global_agent_steps, "world")
        write_scalars(self.writer, self.last_info.get("debug", {}), self.global_agent_steps, "debug")
        self.writer.add_scalar("rollout/reward_mean_raw", self.last_reward_mean, self.global_agent_steps)
        self.writer.add_scalar("rollout/done_count_agent", self.last_done_count, self.global_agent_steps)
        self.writer.add_scalar("rollout/done_count_physical", int((terminated | truncated).sum().item()), self.global_agent_steps)
        self.writer.add_scalar("safety/nan_action_count", self.nan_action_count, self.global_agent_steps)

        return {"policy": obs_flat.clone(), "critic": state_flat.clone()}, reward_flat, terminated_flat, truncated_flat, info

    def close(self):
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass

        try:
            self.env.close()
        except Exception:
            pass


# ======================================================================
# Actor-Critic
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

        self.apply(self._orthogonal_init)

        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.mul_(0.03)
                last.bias.zero_()

    @staticmethod
    def _orthogonal_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def compute(self, inputs, role):
        x = inputs.get("observations", inputs.get("states"))
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, -10.0, 10.0)

        mean = self.net(x)
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


class DiffDriveTask4Critic(DeterministicMixin, Model):
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

        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.apply(DiffDriveTask4Actor._orthogonal_init)

    def compute(self, inputs, role):
        x = inputs.get("states")
        x = torch.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)
        x = torch.clamp(x, -20.0, 20.0)

        value = self.net(x)
        value = torch.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0)
        value = torch.clamp(value, -500.0, 500.0)

        return value, {}


# ======================================================================
# PPO config / print
# ======================================================================

def build_skrl_cfg(env, log_dir: str) -> Dict[str, Any]:
    default_cfg = PPO_CFG()

    if dataclasses.is_dataclass(default_cfg):
        cfg = dataclasses.asdict(default_cfg)
    elif isinstance(default_cfg, dict):
        cfg = default_cfg.copy()
    else:
        cfg = dict(default_cfg.__dict__)

    cfg.update(
        {
            "rollouts": int(args_cli.rollouts),
            "learning_epochs": int(args_cli.learning_epochs),
            "mini_batches": int(args_cli.mini_batches),
            "discount_factor": float(args_cli.discount_factor),
            "gae_lambda": float(args_cli.gae_lambda),
            "learning_rate": float(args_cli.lr),
            "grad_norm_clip": float(args_cli.grad_norm_clip),
            "ratio_clip": float(args_cli.ratio_clip),
            "value_clip": float(args_cli.value_clip),
            "entropy_loss_scale": float(args_cli.entropy_loss_scale),
            "value_loss_scale": float(args_cli.value_loss_scale),
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
    )

    if KLAdaptiveLR is not None:
        cfg["learning_rate_scheduler"] = KLAdaptiveLR
        cfg["learning_rate_scheduler_kwargs"] = {
            "kl_threshold": float(args_cli.kl_threshold),
            "min_lr": float(args_cli.min_lr),
            "max_lr": float(args_cli.max_lr),
        }

    cfg.setdefault("experiment", {})
    cfg["experiment"].update(
        {
            "directory": log_dir,
            "experiment_name": "diff_drive_task4_multi_ugv_formation_escort_curriculum",
            "write_interval": 1_000_000,
            "checkpoint_interval": 0,
        }
    )

    return cfg


def print_update(
    pbar,
    update_id: int,
    agent_steps: int,
    total_steps: int,
    elapsed: float,
    physical_num_envs: int,
    skrl_num_envs: int,
    rollouts: int,
    info: Dict[str, Any],
    ppo: Dict[str, Any],
    lr: float,
    curriculum_info: Dict[str, Any],
):
    physical_steps = agent_steps / 4.0

    stat = {
        "update": float(update_id),
        "agent_env_steps": float(agent_steps),
        "physical_env_steps_equiv": float(physical_steps),
        "target_agent_steps": float(total_steps),
        "progress_percent": 100.0 * agent_steps / max(total_steps, 1),
        "physical_num_envs": float(physical_num_envs),
        "skrl_num_envs": float(skrl_num_envs),
        "rollouts_per_update": float(rollouts),
        "fps_agent_steps": agent_steps / max(elapsed, 1.0e-6),
        "learning_rate": lr,
    }

    tel = info.get("telemetry", {}) if isinstance(info, dict) else {}
    ev = info.get("events", {}) if isinstance(info, dict) else {}

    pbar.write(
        "\n".join(
            [
                "\n" + "=" * 124,
                f"📊 [Diff-Drive Task4 PPO 更新 {update_id}] "
                f"Agent步数: {agent_steps:,} / {total_steps:,} | "
                f"FPS(agent): {stat['fps_agent_steps']:,.0f} | LR: {lr:.3e} | "
                f"Stage: {int(curriculum_info.get('Curriculum_Stage', -1))} | "
                f"Dist: {tel.get('Center_Goal_Dist', 0):.2f} | "
                f"Slot: {tel.get('Mean_Slot_Error', 0):.2f} | "
                f"Succ: {ev.get('Success_Rate', 0):.3f} | Crash: {ev.get('Crash_Rate', 0):.3f}",
                "=" * 124,
                make_table("time / progress", stat),
                make_table("curriculum", curriculum_info),
                make_table("env info: rewards + events + telemetry + world + debug", flat_dict(info)),
                make_table("ppo update info", ppo),
                "=" * 124 + "\n",
            ]
        )
    )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    set_seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    log_root = PROJECT_ROOT / "logs" / "task4"
    run_name = f"diff_drive_task4_skrl_ppo_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = str(log_root / run_name)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 124)
    print("🚀 Diff-Drive UGV / Jetbot Task4: Multi-UGV Formation Escort TRUE skrl PPO Training")
    print("=" * 124)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_root     = {log_root}")
    print(f"[INFO] run_name     = {run_name}")
    print("[INFO] This version uses skrl PPO, not legacy_framework and not a custom PPO loop.")

    env_cfg = Task4Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.test_device)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.max_episode_length_s = float(args_cli.max_episode_length_s)
    env_cfg.max_wheel_speed = float(args_cli.max_wheel_speed)
    env_cfg.reverse_speed_fraction = float(args_cli.reverse_speed_fraction)
    env_cfg.left_wheel_sign = float(args_cli.left_wheel_sign)
    env_cfg.right_wheel_sign = float(args_cli.right_wheel_sign)
    env_cfg.print_debug_info = bool(args_cli.print_debug_info)

    if bool(args_cli.disable_physical_asset_teleport):
        env_cfg.world_cfg.enable_physical_asset_teleport = False

    curriculum = Task4FormationCurriculum(
        total_agent_steps=int(args_cli.total_agent_steps),
        disabled=bool(args_cli.disable_curriculum),
        fixed_stage=int(args_cli.fixed_stage),
    )

    dummy_for_cfg = type("DummyEnv", (), {})()
    dummy_for_cfg.cfg = env_cfg
    curriculum.apply(dummy_for_cfg, int(args_cli.start_agent_steps))

    env_cfg.validate()

    base_env = DiffDriveTask4Env(env_cfg)
    skrl_base_env = DiffDriveTask4SkrlWrapper(base_env, log_dir=log_dir)
    env = wrap_env(skrl_base_env, wrapper="isaaclab")

    skrl_num_envs = int(getattr(env, "num_envs", skrl_base_env.num_envs))
    physical_num_envs = int(base_env.num_envs)

    print("\n[DEBUG] Diff-Drive Task4 skrl spaces")
    print(f"  physical num_envs     = {physical_num_envs}")
    print(f"  skrl num_envs         = {skrl_num_envs}")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")

    assert int(env.observation_space.shape[0]) == int(env_cfg.actor_obs_dim)
    assert int(env.state_space.shape[0]) == int(env_cfg.critic_obs_dim + env_cfg.num_agents)
    assert int(env.action_space.shape[0]) == int(env_cfg.num_actions_per_agent)
    assert skrl_num_envs == physical_num_envs * env_cfg.num_agents

    models = {
        "policy": DiffDriveTask4Actor(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            init_log_std=args_cli.init_log_std,
            min_log_std=args_cli.min_log_std,
            max_log_std=args_cli.max_log_std,
        ),
        "value": DiffDriveTask4Critic(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
        ),
    }

    cfg = build_skrl_cfg(env, log_dir=log_dir)

    total_agent_steps = int(args_cli.total_agent_steps)
    start_agent_steps = int(args_cli.start_agent_steps)
    remaining_agent_steps = max(total_agent_steps - start_agent_steps, 1)
    total_vector_steps = int(math.ceil(remaining_agent_steps / max(skrl_num_envs, 1)))
    save_freq_agent_steps = int(args_cli.save_freq_agent_steps)
    update_agent_steps = int(cfg["rollouts"] * skrl_num_envs)

    # Keep skrl internal writers silent; our wrapper/TensorBoard handles logging.
    cfg.setdefault("experiment", {})
    cfg["experiment"]["write_interval"] = max(total_vector_steps + 1, 1_000_000)
    cfg["experiment"]["checkpoint_interval"] = 0

    print("\n[INFO] Diff-Drive Task4 skrl PPO 训练配置")
    print(f"  - physical_num_envs      : {physical_num_envs:,}")
    print(f"  - skrl_num_envs          : {skrl_num_envs:,}")
    print(f"  - total_agent_steps      : {total_agent_steps:,}")
    print(f"  - start_agent_steps      : {start_agent_steps:,}")
    print(f"  - remaining_agent_steps  : {remaining_agent_steps:,}")
    print(f"  - total_vector_steps     : {total_vector_steps:,}")
    print(f"  - rollouts               : {cfg['rollouts']}")
    print(f"  - update_agent_steps     : {update_agent_steps:,}")
    print(f"  - save_freq_agent_steps  : {save_freq_agent_steps:,}")
    print(f"  - actor_obs_dim          : {env.observation_space.shape[0]}")
    print(f"  - critic_obs_dim         : {env.state_space.shape[0]}  (96 world + 4 agent id)")
    print(f"  - action_dim             : {env.action_space.shape[0]}")
    print(f"  - max_episode_length_s   : {env_cfg.max_episode_length_s}")
    print(f"  - max_episode_length     : {env_cfg.max_episode_length}")
    print(f"  - max_wheel_speed        : {env_cfg.max_wheel_speed}")
    print(f"  - lr/min/max             : {args_cli.lr} / {args_cli.min_lr} / {args_cli.max_lr}")
    print(f"  - gamma                  : {args_cli.discount_factor}")
    print(f"  - gae_lambda             : {args_cli.gae_lambda}")
    print(f"  - entropy_loss_scale     : {args_cli.entropy_loss_scale}")
    print(f"  - init_log_std           : {args_cli.init_log_std}")
    print(f"  - log_std clamp          : [{args_cli.min_log_std}, {args_cli.max_log_std}]")
    print(f"  - curriculum             : {'disabled' if args_cli.disable_curriculum else 'enabled'}")
    print(f"  - fixed_stage            : {args_cli.fixed_stage}")
    print(f"  - resume                 : {args_cli.resume if args_cli.resume else '<none>'}")
    print(f"  - pretrained             : {args_cli.pretrained if args_cli.pretrained else '<none>'}")
    print(f"  - tensorboard            : tensorboard --logdir={PROJECT_ROOT / 'logs'}")

    memory = RandomMemory(
        memory_size=int(cfg["rollouts"]),
        num_envs=skrl_num_envs,
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

    pretrained_loaded = False
    resumed = False

    if args_cli.resume:
        resumed = try_load_agent(agent, args_cli.resume, "resume checkpoint")
        sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)
    elif args_cli.pretrained:
        pretrained_loaded = try_load_agent(agent, args_cli.pretrained, "pretrained checkpoint")
        sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)

    trainer = StepTrainer(
        cfg={
            "timesteps": total_vector_steps,
            "headless": True,
            "disable_progressbar": True,
        },
        env=env,
        agents=agent,
    )

    print("\n🔥 [Diff-Drive Task4 TRUE skrl PPO + Curriculum 已启动]")
    print("👉 skrl 看到的是 physical_envs × 4 个 agent env；Actor 共享参数。")
    print("👉 Critic 输入为 96 维全局 state + 4 维 agent_id。")
    print("👉 课程从 clean formation navigation 逐步过渡到 obstacles / narrow gate / full DR。")
    print("👉 重点观察：Center_Goal_Dist / Progress / Mean_Slot_Error / Min_Pair_Dist / Success_Rate / Crash_Rate。")
    print(f"👉 TensorBoard: tensorboard --logdir={PROJECT_ROOT / 'logs'}\n")

    last_save = start_agent_steps
    update_id = 0
    start_time = time.time()
    curriculum_info = curriculum.apply(base_env, start_agent_steps)

    try:
        trainer.reset()

        with tqdm(
            total=total_agent_steps,
            initial=start_agent_steps,
            desc="Diff-Drive Task4 skrl PPO",
            unit="agent_steps",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as pbar:
            for t in range(total_vector_steps):
                absolute_agent_steps = min(start_agent_steps + (t + 1) * skrl_num_envs, total_agent_steps)

                curriculum_info = curriculum.apply(base_env, absolute_agent_steps)
                write_scalars(skrl_base_env.writer, curriculum_info, absolute_agent_steps, "curriculum")

                trainer.train(timestep=t, timesteps=total_vector_steps)

                prev_steps = min(start_agent_steps + t * skrl_num_envs, total_agent_steps)
                pbar.update(absolute_agent_steps - prev_steps)

                tel = skrl_base_env.last_info.get("telemetry", {})
                ev = skrl_base_env.last_info.get("events", {})
                rew = skrl_base_env.last_info.get("reward_components", {})

                pbar.set_postfix(
                    {
                        "steps": f"{absolute_agent_steps:,}",
                        "stage": int(curriculum_info.get("Curriculum_Stage", -1)),
                        "fps": f"{(absolute_agent_steps - start_agent_steps) / max(time.time() - start_time, 1e-6):,.0f}",
                        "rew": f"{skrl_base_env.last_reward_mean:+.3f}",
                        "dist": f"{tel.get('Center_Goal_Dist', 0):.2f}",
                        "prog": f"{tel.get('Progress', 0):+.4f}",
                        "slot": f"{tel.get('Mean_Slot_Error', 0):.2f}",
                        "pair": f"{tel.get('Min_Pair_Dist', 0):.2f}",
                        "gate": f"{tel.get('Gate_Active', 0):.0f}",
                        "succ": f"{ev.get('Success_Rate', 0):.3f}",
                        "crash": f"{ev.get('Crash_Rate', 0):.3f}",
                        "r_prog": f"{rew.get('R_Team_Progress', 0):+.3f}",
                        "p_form": f"{rew.get('P_Formation_Mean', 0):+.3f}",
                    }
                )

                if (t + 1) % 32 == 0:
                    sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)

                if (t + 1) % int(cfg["rollouts"]) == 0:
                    update_id += 1
                    sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)

                    ppo_info = tracking_mean(agent)
                    ppo_info["learning_rate"] = current_lr(agent)

                    writer = getattr(agent, "writer", None)
                    write_scalars(writer, ppo_info, absolute_agent_steps, "ppo")
                    write_scalars(writer, flat_dict(skrl_base_env.last_info), absolute_agent_steps, "env_info")
                    write_scalars(writer, curriculum_info, absolute_agent_steps, "curriculum")

                    bad, bad_key = ppo_info_has_nan(ppo_info)

                    if update_id % max(int(args_cli.log_interval_updates), 1) == 0:
                        print_update(
                            pbar=pbar,
                            update_id=update_id,
                            agent_steps=absolute_agent_steps,
                            total_steps=total_agent_steps,
                            elapsed=time.time() - start_time,
                            physical_num_envs=physical_num_envs,
                            skrl_num_envs=skrl_num_envs,
                            rollouts=int(cfg["rollouts"]),
                            info=skrl_base_env.last_info,
                            ppo=ppo_info,
                            lr=ppo_info["learning_rate"],
                            curriculum_info=curriculum_info,
                        )

                    try:
                        agent.tracking_data.clear()
                    except Exception:
                        pass

                    if bad:
                        emergency_dir = os.path.join(log_dir, f"emergency_nan_checkpoint_{absolute_agent_steps}")
                        save_all_checkpoints(
                            save_dir=emergency_dir,
                            agent=agent,
                            models=models,
                            env_cfg=env_cfg,
                            base_env=base_env,
                            env=env,
                            args=args_cli,
                            agent_steps=absolute_agent_steps,
                            curriculum_info=curriculum_info,
                            extra={
                                "reason": f"ppo_nan_detected: {bad_key}",
                                "last_info": skrl_base_env.last_info,
                                "ppo_info": ppo_info,
                            },
                        )
                        raise RuntimeError(
                            f"PPO 数值异常：{bad_key}=NaN/Inf。已保存 emergency checkpoint: {emergency_dir}"
                        )

                if absolute_agent_steps - last_save >= save_freq_agent_steps:
                    last_save = absolute_agent_steps
                    save_dir = os.path.join(log_dir, f"checkpoint_{absolute_agent_steps}")
                    try:
                        save_all_checkpoints(
                            save_dir=save_dir,
                            agent=agent,
                            models=models,
                            env_cfg=env_cfg,
                            base_env=base_env,
                            env=env,
                            args=args_cli,
                            agent_steps=absolute_agent_steps,
                            curriculum_info=curriculum_info,
                            extra={
                                "pretrained_loaded": pretrained_loaded,
                                "resumed": resumed,
                                "last_info": skrl_base_env.last_info,
                            },
                        )
                        pbar.write(
                            f"\n💾 [Diff-Drive Task4 skrl 备份] "
                            f"Agent步数: {absolute_agent_steps:,} | 已保存至: {save_dir}\n"
                        )
                    except Exception as exc:
                        pbar.write(f"\n[WARN] checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

    except KeyboardInterrupt:
        print("\n[WARN] 接收到手动中断信号，正在安全保存...")
    except Exception:
        print("\n[ERROR] Diff-Drive Task4 skrl PPO 训练过程中发生真实异常：")
        traceback.print_exc()
    finally:
        final_agent_steps = min(
            total_agent_steps,
            start_agent_steps + int(getattr(skrl_base_env, "global_agent_steps", 0)),
        )
        final_dir = os.path.join(log_dir, "final_checkpoint")
        os.makedirs(final_dir, exist_ok=True)

        try:
            save_all_checkpoints(
                save_dir=final_dir,
                agent=agent,
                models=models,
                env_cfg=env_cfg,
                base_env=base_env,
                env=env,
                args=args_cli,
                agent_steps=final_agent_steps,
                curriculum_info=curriculum_info,
                extra={
                    "final": True,
                    "pretrained_loaded": pretrained_loaded,
                    "resumed": resumed,
                    "last_info": skrl_base_env.last_info,
                },
            )
            print(f"✅ Diff-Drive Task4 skrl 模型与归一化统计已保存至 {final_dir}")
        except Exception as exc:
            print(f"[WARN] 保存最终模型失败: {type(exc).__name__}: {exc}")

        try:
            env.close()
        except Exception:
            pass

        try:
            simulation_app.close()
        except Exception:
            pass

        print("✅ Diff-Drive Task4 skrl PPO + Curriculum 训练管线安全退出")


if __name__ == "__main__":
    main()

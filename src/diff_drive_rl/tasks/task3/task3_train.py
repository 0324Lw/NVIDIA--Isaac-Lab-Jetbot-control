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
from typing import Any, Dict, Optional, Tuple

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

parser = argparse.ArgumentParser(description="Train Diff-Drive UGV / Jetbot Task3 with TRUE skrl PPO")

# Runtime
parser.add_argument("--total-env-steps", type=int, default=300_000_000)
parser.add_argument("--save-freq-env-steps", type=int, default=10_000_000)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", type=str, default="", help="Optional skrl checkpoint or final_checkpoint directory")
parser.add_argument("--pretrained", type=str, default="", help="Optional skrl checkpoint for initialization")
parser.add_argument("--start-env-steps", type=int, default=0)

# Env overrides
parser.add_argument("--max-episode-length-s", type=float, default=40.0)
parser.add_argument("--max-wheel-speed", type=float, default=14.0)
parser.add_argument("--left-wheel-sign", type=float, default=1.0)
parser.add_argument("--right-wheel-sign", type=float, default=1.0)

# Curriculum
parser.add_argument("--disable-curriculum", action="store_true")
parser.add_argument("--curriculum-debug", action="store_true")

# PPO
parser.add_argument("--rollouts", type=int, default=64)
parser.add_argument("--learning-epochs", type=int, default=5)
parser.add_argument("--mini-batches", type=int, default=8)

parser.add_argument("--lr", type=float, default=1.0e-4)
parser.add_argument("--min-lr", type=float, default=2.0e-5)
parser.add_argument("--max-lr", type=float, default=1.5e-4)

parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-range", type=float, default=0.18)
parser.add_argument("--value-clip", type=float, default=0.20)
parser.add_argument("--entropy-coef", type=float, default=0.002)
parser.add_argument("--value-coef", type=float, default=1.0)
parser.add_argument("--grad-clip", type=float, default=0.5)

# Policy distribution
parser.add_argument("--init-log-std", type=float, default=-1.0)
parser.add_argument("--min-log-std", type=float, default=-3.0)
parser.add_argument("--max-log-std", type=float, default=0.5)

# KL
parser.add_argument("--target-kl", type=float, default=0.010)
parser.add_argument("--hard-kl-stop", type=float, default=0.120)

# Logging
parser.add_argument("--log-root", type=str, default=str(PROJECT_ROOT / "logs" / "task3"))
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--summary-interval", type=int, default=1)
parser.add_argument("--skrl-write-interval", type=int, default=1_000_000)
parser.add_argument("--skrl-checkpoint-interval", type=int, default=0)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from diff_drive_rl.tasks.task3.task3_config import Task3Config
from diff_drive_rl.tasks.task3.task3_env import DiffDriveTask3Env

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


def is_bad_number(x: Any) -> bool:
    value = to_float(x)
    if value is None:
        return False
    return not math.isfinite(value)


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
        if val is None or not math.isfinite(val):
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
            if math.isnan(value):
                value_str = "nan"
            elif math.isinf(value):
                value_str = "inf"
            else:
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
            if arr.size == 0:
                continue
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


def sanitize_tensor_inplace(
    x: Optional[torch.Tensor],
    nan: float = 0.0,
    posinf: float = 1.0,
    neginf: float = -1.0,
    clamp_abs: Optional[float] = None,
) -> None:
    if x is None or not torch.is_tensor(x):
        return
    with torch.no_grad():
        x.data = torch.nan_to_num(x.data, nan=nan, posinf=posinf, neginf=neginf)
        if clamp_abs is not None:
            x.data.clamp_(-float(clamp_abs), float(clamp_abs))


def sanitize_agent_numerics(agent, models: Dict[str, Model], min_log_std: float, max_log_std: float) -> None:
    for model in models.values():
        for param in model.parameters():
            sanitize_tensor_inplace(param, nan=0.0, posinf=1.0, neginf=-1.0, clamp_abs=20.0)

        if hasattr(model, "log_std_parameter"):
            with torch.no_grad():
                model.log_std_parameter.data = torch.nan_to_num(
                    model.log_std_parameter.data,
                    nan=float(args_cli.init_log_std),
                    posinf=float(max_log_std),
                    neginf=float(min_log_std),
                )
                model.log_std_parameter.data.clamp_(float(min_log_std), float(max_log_std))

    optimizer = getattr(agent, "optimizer", None)
    if optimizer is not None:
        for state in optimizer.state.values():
            for _, value in state.items():
                if torch.is_tensor(value):
                    with torch.no_grad():
                        value.data = torch.nan_to_num(value.data, nan=0.0, posinf=1.0, neginf=-1.0)
                        value.data.clamp_(-100.0, 100.0)


def ppo_info_has_nan(ppo_info: Dict[str, Any]) -> Tuple[bool, str]:
    priority = [
        "Loss / Entropy loss",
        "Loss / Policy loss",
        "Loss / Value loss",
        "Policy / Standard deviation",
        "Learning / Learning rate",
        "learning_rate",
    ]

    for key in priority:
        if key in ppo_info and is_bad_number(ppo_info[key]):
            return True, key

    for key, value in ppo_info.items():
        if "Loss" in key or "Standard deviation" in key:
            if is_bad_number(value):
                return True, key

    return False, ""


def make_run_name() -> str:
    run_name = args_cli.run_name.strip()
    if run_name:
        return run_name
    return f"diff_drive_task3_skrl_ppo_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ======================================================================
# Curriculum
# ======================================================================

class Task3Sim2RealCurriculum:
    """Task3 conservative Sim2Real parking curriculum.

    Stage0:
        Clean navigation, weak randomization, stronger progress reward.
    Stage1:
        Mild action / sensor / motor randomization.
    Stage2:
        Moderate Sim2Real and stronger parking shaping.
    Stage3:
        Strong Sim2Real and stronger risk shaping.
    Stage4:
        Full Sim2Real parking objective.

    Real scene geometry is fixed. Curriculum only changes sampling ranges,
    action model randomization ranges, sensor noise ranges, and reward weights.
    """

    def __init__(self, total_env_steps: int, disabled: bool = False):
        self.total_env_steps = max(int(total_env_steps), 1)
        self.disabled = bool(disabled)
        self.last_stage = -1

    def stage_from_progress(self, k: float) -> int:
        if self.disabled:
            return 4
        if k < 0.16:
            return 0
        if k < 0.32:
            return 1
        if k < 0.55:
            return 2
        if k < 0.78:
            return 3
        return 4

    @staticmethod
    def lock_conservative_scene_geometry(wcfg) -> None:
        # This must remain consistent with task3_world.py conservative geometry.
        wcfg.spot_width_inner_range = (0.60, 0.60)
        wcfg.spot_depth_inner_range = (0.73, 0.73)

        wcfg.bump_length_x_range = (0.45, 0.45)
        wcfg.bump_width_y_range = (2.80, 2.80)
        wcfg.bump_height_range = (0.006, 0.006)
        wcfg.bump_yaw_range = (0.0, 0.0)
        wcfg.bump_ramp_segments = 3
        wcfg.bump_top_length = 0.13
        wcfg.bump_low_height_ratio = 0.35
        wcfg.bump_static_friction = 1.10
        wcfg.bump_dynamic_friction = 0.95

    def apply(self, env_or_dummy, env_steps: int) -> Dict[str, float]:
        cfg = env_or_dummy.cfg
        wcfg = cfg.world_cfg
        self.lock_conservative_scene_geometry(wcfg)

        k = max(0.0, min(1.0, float(env_steps) / float(self.total_env_steps)))
        stage = self.stage_from_progress(k)

        if stage == 0:
            wcfg.start_y_range = (-0.35, 0.35)
            wcfg.start_yaw_range = (-0.06, 0.06)

            wcfg.action_delay_frame_range = (0, 0)
            wcfg.action_deadband_range = (0.00, 0.010)
            wcfg.action_ema_alpha_range = (0.90, 1.00)

            wcfg.motor_strength_range = (0.99, 1.01)
            wcfg.motor_bias_range = (-0.005, 0.005)
            wcfg.wheel_radius_scale_range = (0.995, 1.005)

            wcfg.lidar_noise_std_range = (0.000, 0.008)
            wcfg.lidar_outlier_prob_range = (0.000, 0.001)
            wcfg.lidar_dropout_prob_range = (0.000, 0.001)
            wcfg.lidar_yaw_offset_range = (-math.radians(0.3), math.radians(0.3))
            wcfg.lidar_z_offset_range = (0.145, 0.155)

            cfg.w_progress = 6.00
            cfg.w_goal_speed = 0.65
            cfg.w_heading = 0.12
            cfg.w_stuck = 0.11
            cfg.w_step = 0.006

            cfg.w_terrain_progress = 0.16
            cfg.w_bump_progress = 0.06
            cfg.w_bump_smooth = 0.00
            cfg.w_bump_risk = 0.00

            cfg.w_parking_pos = 0.12
            cfg.w_parking_yaw = 0.08
            cfg.w_inside_box = 0.04
            cfg.w_parking_low_speed = 0.05

            cfg.w_front_clearance = 0.035
            cfg.w_lidar_risk = 0.04
            cfg.w_wall_risk = 0.05
            cfg.w_lane_risk = 0.08

            cfg.w_action_smooth = 0.008
            cfg.w_action_mag = 0.0015
            cfg.w_spin = 0.020
            cfg.w_wheel_speed = 0.001

        elif stage == 1:
            wcfg.start_y_range = (-0.50, 0.50)
            wcfg.start_yaw_range = (-0.09, 0.09)

            wcfg.action_delay_frame_range = (0, 1)
            wcfg.action_deadband_range = (0.00, 0.025)
            wcfg.action_ema_alpha_range = (0.80, 1.00)

            wcfg.motor_strength_range = (0.96, 1.04)
            wcfg.motor_bias_range = (-0.015, 0.015)
            wcfg.wheel_radius_scale_range = (0.985, 1.015)

            wcfg.lidar_noise_std_range = (0.000, 0.015)
            wcfg.lidar_outlier_prob_range = (0.000, 0.004)
            wcfg.lidar_dropout_prob_range = (0.000, 0.004)
            wcfg.lidar_yaw_offset_range = (-math.radians(0.8), math.radians(0.8))
            wcfg.lidar_z_offset_range = (0.135, 0.165)

            cfg.w_progress = 5.60
            cfg.w_goal_speed = 0.58
            cfg.w_heading = 0.12
            cfg.w_stuck = 0.10
            cfg.w_step = 0.005

            cfg.w_terrain_progress = 0.14
            cfg.w_bump_progress = 0.055
            cfg.w_bump_smooth = 0.004
            cfg.w_bump_risk = 0.00

            cfg.w_parking_pos = 0.32
            cfg.w_parking_yaw = 0.20
            cfg.w_inside_box = 0.10
            cfg.w_parking_low_speed = 0.12

            cfg.w_front_clearance = 0.05
            cfg.w_lidar_risk = 0.08
            cfg.w_wall_risk = 0.10
            cfg.w_lane_risk = 0.12

            cfg.w_action_smooth = 0.010
            cfg.w_action_mag = 0.0018
            cfg.w_spin = 0.022
            cfg.w_wheel_speed = 0.001

        elif stage == 2:
            wcfg.start_y_range = (-0.60, 0.60)
            wcfg.start_yaw_range = (-0.12, 0.12)

            wcfg.action_delay_frame_range = (0, 2)
            wcfg.action_deadband_range = (0.005, 0.045)
            wcfg.action_ema_alpha_range = (0.65, 0.95)

            wcfg.motor_strength_range = (0.92, 1.08)
            wcfg.motor_bias_range = (-0.025, 0.025)
            wcfg.wheel_radius_scale_range = (0.970, 1.030)

            wcfg.lidar_noise_std_range = (0.003, 0.026)
            wcfg.lidar_outlier_prob_range = (0.000, 0.010)
            wcfg.lidar_dropout_prob_range = (0.000, 0.008)
            wcfg.lidar_yaw_offset_range = (-math.radians(1.2), math.radians(1.2))
            wcfg.lidar_z_offset_range = (0.125, 0.175)

            cfg.w_progress = 4.70
            cfg.w_goal_speed = 0.43
            cfg.w_heading = 0.13
            cfg.w_stuck = 0.07
            cfg.w_step = 0.0035

            cfg.w_parking_pos = 0.58
            cfg.w_parking_yaw = 0.38
            cfg.w_inside_box = 0.18
            cfg.w_parking_low_speed = 0.20

            cfg.w_front_clearance = 0.06
            cfg.w_lidar_risk = 0.13
            cfg.w_wall_risk = 0.16
            cfg.w_lane_risk = 0.17

            cfg.w_bump_progress = 0.045
            cfg.w_bump_smooth = 0.006
            cfg.w_bump_risk = 0.005

            cfg.w_action_smooth = 0.012
            cfg.w_action_mag = 0.002
            cfg.w_spin = 0.024
            cfg.w_wheel_speed = 0.001

        elif stage == 3:
            wcfg.start_y_range = (-0.68, 0.68)
            wcfg.start_yaw_range = (-0.15, 0.15)

            wcfg.action_delay_frame_range = (0, 3)
            wcfg.action_deadband_range = (0.010, 0.065)
            wcfg.action_ema_alpha_range = (0.50, 0.90)

            wcfg.motor_strength_range = (0.87, 1.13)
            wcfg.motor_bias_range = (-0.040, 0.040)
            wcfg.wheel_radius_scale_range = (0.945, 1.055)

            wcfg.lidar_noise_std_range = (0.005, 0.040)
            wcfg.lidar_outlier_prob_range = (0.000, 0.020)
            wcfg.lidar_dropout_prob_range = (0.000, 0.015)
            wcfg.lidar_yaw_offset_range = (-math.radians(1.8), math.radians(1.8))
            wcfg.lidar_z_offset_range = (0.115, 0.185)

            cfg.w_progress = 4.35
            cfg.w_goal_speed = 0.39
            cfg.w_heading = 0.12
            cfg.w_stuck = 0.06
            cfg.w_step = 0.003

            cfg.w_parking_pos = 0.72
            cfg.w_parking_yaw = 0.48
            cfg.w_inside_box = 0.23
            cfg.w_parking_low_speed = 0.25

            cfg.w_front_clearance = 0.07
            cfg.w_lidar_risk = 0.17
            cfg.w_wall_risk = 0.24
            cfg.w_lane_risk = 0.22

            cfg.w_bump_progress = 0.05
            cfg.w_bump_smooth = 0.008
            cfg.w_bump_risk = 0.010

            cfg.w_action_smooth = 0.015
            cfg.w_action_mag = 0.002
            cfg.w_spin = 0.025
            cfg.w_wheel_speed = 0.001

        else:
            wcfg.start_y_range = (-0.70, 0.70)
            wcfg.start_yaw_range = (-0.15, 0.15)

            wcfg.action_delay_frame_range = (0, 4)
            wcfg.action_deadband_range = (0.020, 0.080)
            wcfg.action_ema_alpha_range = (0.35, 0.75)

            wcfg.motor_strength_range = (0.80, 1.20)
            wcfg.motor_bias_range = (-0.050, 0.050)
            wcfg.wheel_radius_scale_range = (0.920, 1.080)

            wcfg.lidar_noise_std_range = (0.005, 0.050)
            wcfg.lidar_outlier_prob_range = (0.000, 0.030)
            wcfg.lidar_dropout_prob_range = (0.000, 0.020)
            wcfg.lidar_yaw_offset_range = (-math.radians(2.0), math.radians(2.0))
            wcfg.lidar_z_offset_range = (0.100, 0.200)

            cfg.w_progress = 4.00
            cfg.w_goal_speed = 0.35
            cfg.w_heading = 0.12
            cfg.w_stuck = 0.05
            cfg.w_step = 0.003

            cfg.w_parking_pos = 0.80
            cfg.w_parking_yaw = 0.55
            cfg.w_inside_box = 0.25
            cfg.w_parking_low_speed = 0.25

            cfg.w_front_clearance = 0.08
            cfg.w_lidar_risk = 0.20
            cfg.w_wall_risk = 0.30
            cfg.w_lane_risk = 0.25

            cfg.w_bump_progress = 0.05
            cfg.w_bump_smooth = 0.010
            cfg.w_bump_risk = 0.015

            cfg.w_action_smooth = 0.015
            cfg.w_action_mag = 0.002
            cfg.w_spin = 0.025
            cfg.w_wheel_speed = 0.001

        changed = stage != self.last_stage
        self.last_stage = stage

        info = {
            "Curriculum_K": float(k),
            "Curriculum_Stage": float(stage),
            "Curriculum_Changed": float(changed),

            "Bump_Height": float(wcfg.bump_height_range[0]),
            "Bump_Length": float(wcfg.bump_length_x_range[0]),
            "Bump_Width": float(wcfg.bump_width_y_range[0]),
            "Bump_Ramp_Segments": float(wcfg.bump_ramp_segments),
            "Bump_Top_Length": float(wcfg.bump_top_length),
            "Bump_Low_Height_Ratio": float(wcfg.bump_low_height_ratio),

            "Start_Y_Range": float(wcfg.start_y_range[1]),
            "Start_Yaw_Range": float(wcfg.start_yaw_range[1]),

            "Delay_Min": float(wcfg.action_delay_frame_range[0]),
            "Delay_Max": float(wcfg.action_delay_frame_range[1]),
            "Deadband_Min": float(wcfg.action_deadband_range[0]),
            "Deadband_Max": float(wcfg.action_deadband_range[1]),
            "EMA_Min": float(wcfg.action_ema_alpha_range[0]),
            "EMA_Max": float(wcfg.action_ema_alpha_range[1]),

            "Motor_Strength_Low": float(wcfg.motor_strength_range[0]),
            "Motor_Strength_High": float(wcfg.motor_strength_range[1]),
            "Motor_Bias_Low": float(wcfg.motor_bias_range[0]),
            "Motor_Bias_High": float(wcfg.motor_bias_range[1]),
            "Wheel_Scale_Low": float(wcfg.wheel_radius_scale_range[0]),
            "Wheel_Scale_High": float(wcfg.wheel_radius_scale_range[1]),

            "Lidar_Noise_Min": float(wcfg.lidar_noise_std_range[0]),
            "Lidar_Noise_Max": float(wcfg.lidar_noise_std_range[1]),
            "Lidar_Outlier_Max": float(wcfg.lidar_outlier_prob_range[1]),
            "Lidar_Dropout_Max": float(wcfg.lidar_dropout_prob_range[1]),
            "Lidar_Yaw_Offset_Max_Deg": float(math.degrees(wcfg.lidar_yaw_offset_range[1])),

            "Reward_w_progress": float(cfg.w_progress),
            "Reward_w_goal_speed": float(cfg.w_goal_speed),
            "Reward_w_heading": float(cfg.w_heading),
            "Reward_w_parking_pos": float(cfg.w_parking_pos),
            "Reward_w_parking_yaw": float(cfg.w_parking_yaw),
            "Reward_w_inside_box": float(cfg.w_inside_box),
            "Reward_w_parking_low_speed": float(cfg.w_parking_low_speed),
            "Reward_w_lidar_risk": float(cfg.w_lidar_risk),
            "Reward_w_wall_risk": float(cfg.w_wall_risk),
            "Reward_w_lane_risk": float(cfg.w_lane_risk),
            "Reward_w_bump_progress": float(cfg.w_bump_progress),
            "Reward_w_bump_smooth": float(cfg.w_bump_smooth),
            "Reward_w_bump_risk": float(cfg.w_bump_risk),
            "Reward_w_stuck": float(cfg.w_stuck),
            "Reward_w_step": float(cfg.w_step),
            "Reward_w_terrain_progress": float(cfg.w_terrain_progress),
            "Reward_w_action_smooth": float(cfg.w_action_smooth),
            "Reward_w_action_mag": float(cfg.w_action_mag),
            "Reward_w_spin": float(cfg.w_spin),
            "Reward_w_wheel_speed": float(cfg.w_wheel_speed),
        }

        if changed:
            print("\n" + "=" * 120)
            print(f"🎓 [Diff-Drive Task3 Curriculum] Stage {stage} | K={k:.4f} | env_steps={env_steps:,}")
            for key in [
                "Bump_Height",
                "Bump_Length",
                "Bump_Ramp_Segments",
                "Start_Y_Range",
                "Start_Yaw_Range",
                "Delay_Max",
                "Deadband_Max",
                "EMA_Min",
                "Motor_Strength_Low",
                "Motor_Strength_High",
                "Wheel_Scale_Low",
                "Wheel_Scale_High",
                "Lidar_Noise_Max",
                "Lidar_Outlier_Max",
                "Lidar_Dropout_Max",
                "Reward_w_progress",
                "Reward_w_goal_speed",
                "Reward_w_parking_pos",
                "Reward_w_parking_yaw",
                "Reward_w_wall_risk",
                "Reward_w_bump_risk",
                "Reward_w_stuck",
                "Reward_w_action_smooth",
            ]:
                print(f"  - {key:<28s}: {info[key]}")
            print("=" * 120 + "\n")

        return info


# ======================================================================
# skrl wrapper
# ======================================================================

class DiffDriveTask3SkrlWrapper(gym.Env):
    """Task3 skrl wrapper.

    policy obs:
        actor obs [N, 404]
    critic state:
        privileged obs [N, 442]
    """

    def __init__(self, env: DiffDriveTask3Env):
        super().__init__()

        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = env.device

        self.observation_space = env.observation_space
        self.state_space = env.state_space
        self.action_space = env.action_space

        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": self.observation_space,
                "critic": self.state_space,
            }
        )
        self.single_action_space = env.action_space

        self.global_env_steps = 0
        self.last_info: Dict[str, Any] = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0
        self.nan_action_count = 0

    @property
    def unwrapped(self):
        return self

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options)
        priv = self.env.compute_privileged_obs()

        obs = torch.nan_to_num(
            obs,
            nan=0.0,
            posinf=float(self.env.cfg.obs_clip),
            neginf=-float(self.env.cfg.obs_clip),
        )
        priv = torch.nan_to_num(
            priv,
            nan=0.0,
            posinf=float(self.env.cfg.priv_clip),
            neginf=-float(self.env.cfg.priv_clip),
        )

        self.last_info = info or {}
        return {"policy": obs.clone(), "critic": priv.clone()}, self.last_info

    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

        if not torch.isfinite(actions).all():
            bad_ratio = (~torch.isfinite(actions)).float().mean().item()
            self.nan_action_count += 1
            print(f"[WARN][Task3Train] NaN/Inf action detected, bad_ratio={bad_ratio:.6f}; replaced with safe action")
            actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)

        actions = torch.clamp(actions, -1.0, 1.0)

        obs, reward, terminated, truncated, info = self.env.step(actions)
        priv = self.env.compute_privileged_obs()

        obs = torch.nan_to_num(
            obs,
            nan=0.0,
            posinf=float(self.env.cfg.obs_clip),
            neginf=-float(self.env.cfg.obs_clip),
        )
        priv = torch.nan_to_num(
            priv,
            nan=0.0,
            posinf=float(self.env.cfg.priv_clip),
            neginf=-float(self.env.cfg.priv_clip),
        )
        reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)
        reward = torch.clamp(reward, -100.0, 100.0)

        done = terminated | truncated

        self.global_env_steps += self.num_envs
        self.last_info = info or {}
        self.last_reward_mean = to_float(reward) or 0.0
        self.last_done_count = int(done.sum().detach().cpu().item())

        return {"policy": obs.clone(), "critic": priv.clone()}, reward, terminated, truncated, self.last_info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# ======================================================================
# Models
# ======================================================================

class DiffDriveTask3Actor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        init_log_std: float = -1.0,
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
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
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
                last.weight.mul_(0.05)
                last.bias.zero_()

    @staticmethod
    def _orthogonal_init(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)

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


class DiffDriveTask3Critic(DeterministicMixin, Model):
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
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        self.apply(DiffDriveTask3Actor._orthogonal_init)

    def compute(self, inputs, role):
        states = inputs.get("states", None)
        if states is None:
            states = inputs.get("observations", None)
        if states is None:
            raise RuntimeError("Critic received no states / observations.")

        states = torch.nan_to_num(states, nan=0.0, posinf=20.0, neginf=-20.0)
        states = torch.clamp(states, -20.0, 20.0)

        value = self.net(states)
        value = torch.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0)
        value = torch.clamp(value, -500.0, 500.0)

        return value, {}


# ======================================================================
# skrl config / checkpoint
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


def resolve_checkpoint(path: str) -> str:
    if not path:
        return ""

    p = Path(path).expanduser().resolve()

    if p.is_file():
        return str(p)

    if p.is_dir():
        candidates = [
            p / "diff_drive_task3_skrl_agent.pt",
            p / "agent.pt",
            p / "final_checkpoint" / "diff_drive_task3_skrl_agent.pt",
            p / "final_checkpoint" / "agent.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    return str(p)


def try_load_agent(agent, path: str, label: str) -> bool:
    ckpt = resolve_checkpoint(path)
    if not ckpt:
        return False

    if not os.path.exists(ckpt):
        print(f"[WARN] {label} checkpoint not found: {ckpt}")
        return False

    print("\n" + "=" * 108)
    print(f"🔁 Loading {label}: {ckpt}")
    print("=" * 108)

    try:
        agent.load(ckpt)
        print(f"✅ Loaded {label} with agent.load()")
        return True
    except Exception as exc:
        print(f"[WARN] agent.load failed for {label}: {type(exc).__name__}: {exc}")
        return False


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
    env_cfg: Task3Config,
    env: DiffDriveTask3SkrlWrapper,
    env_steps: int,
    curriculum_info: Optional[Dict[str, Any]],
    args,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(directory, exist_ok=True)

    skrl_agent_path = os.path.join(directory, "diff_drive_task3_skrl_agent.pt")
    eval_model_path = os.path.join(directory, "diff_drive_task3_model.pt")

    try:
        agent.save(skrl_agent_path)
        agent.save(os.path.join(directory, "agent.pt"))
    except Exception as exc:
        print(f"[WARN] agent.save failed: {type(exc).__name__}: {exc}")

    actor_norm = extract_norm(
        agent,
        ["_observation_preprocessor", "observation_preprocessor"],
        int(env.observation_space.shape[0]),
    )
    critic_norm = extract_norm(
        agent,
        ["_state_preprocessor", "state_preprocessor"],
        int(env.state_space.shape[0]),
    )

    torch.save(
        {
            "policy": models["policy"].state_dict(),
            "value": models["value"].state_dict(),
            "actor_obs_norm": actor_norm,
            "critic_obs_norm": critic_norm,
            "env_steps": int(env_steps),
            "curriculum_info": curriculum_info or {},
            "args": vars(args),
            "metadata": {
                "robot": "Jetbot / two-wheel differential-drive UGV",
                "task": "task3_conservative_sim2real_parking",
                "algorithm": "skrl_PPO",
                "uses_skrl": True,
                "asymmetric_actor_critic": True,
                "policy_input": "actor_obs_4_frame_stack",
                "critic_input": "actor_obs_stack_plus_world_privileged_features",
                "single_actor_obs_dim": int(env_cfg.single_actor_obs_dim),
                "frame_stack": int(env_cfg.frame_stack),
                "actor_obs_dim": int(env_cfg.actor_obs_dim),
                "world_privileged_dim": int(env_cfg.privileged_feature_dim),
                "critic_obs_dim": int(env_cfg.critic_obs_dim),
                "action_dim": int(env_cfg.num_actions),
                "lidar_pool_bins": int(env_cfg.world_cfg.lidar_pool_bins),
                "risk_feature_dim": 10,
                "world_priv_dim": 38,
                "world": "conservative_sim2real_parking_world",
                "control": "left_right_wheel_velocity",
                "note": "TRUE skrl PPO eval checkpoint. Evaluation uses deterministic direct policy forward, not agent.act.",
            },
            "extra": extra or {},
        },
        eval_model_path,
    )

    try:
        torch.save(
            {
                "stage": "task3_skrl_ppo_curriculum",
                "algorithm": "skrl_PPO",
                "uses_skrl": True,
                "env_steps": int(env_steps),
                "num_envs": int(env_cfg.num_envs),
                "actor_obs_dim": int(env_cfg.actor_obs_dim),
                "critic_obs_dim": int(env_cfg.critic_obs_dim),
                "curriculum_info": curriculum_info or {},
                "args": vars(args),
                "extra": extra or {},
            },
            os.path.join(directory, "task3_train_metadata.pt"),
        )
    except Exception as exc:
        print(f"[WARN] metadata save failed: {type(exc).__name__}: {exc}")

    print(f"💾 [Diff-Drive Task3 skrl checkpoint] saved to: {directory}", flush=True)


def print_update(
    pbar,
    update_id: int,
    env_steps: int,
    total_steps: int,
    elapsed: float,
    num_envs: int,
    rollouts: int,
    info: Dict[str, Any],
    ppo: Dict[str, Any],
    lr: float,
    curriculum_info: Dict[str, Any],
) -> None:
    stat = {
        "update": float(update_id),
        "env_steps": float(env_steps),
        "total_env_steps": float(total_steps),
        "progress_percent": 100.0 * env_steps / max(total_steps, 1),
        "num_envs": float(num_envs),
        "rollouts_per_update": float(rollouts),
        "fps_env_steps": max(env_steps - int(args_cli.start_env_steps), 0) / max(elapsed, 1.0e-6),
        "learning_rate": float(lr),
    }

    pbar.write(
        "\n".join(
            [
                "\n" + "=" * 118,
                f"📊 [Diff-Drive Task3 skrl PPO 更新 {update_id}] "
                f"总步数: {env_steps:,} / {total_steps:,} | "
                f"FPS: {stat['fps_env_steps']:,.0f} | LR: {lr:.3e} | "
                f"Stage: {int(curriculum_info.get('Curriculum_Stage', -1))}",
                "=" * 118,
                make_table("time / progress", stat),
                make_table("curriculum", curriculum_info),
                make_table("env info: rewards + events + telemetry + world + debug", flat_dict(info)),
                make_table("ppo update info", ppo),
                "=" * 118 + "\n",
            ]
        )
    )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    set_seed(int(args_cli.seed))
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    run_name = make_run_name()
    log_root = os.path.abspath(args_cli.log_root)
    os.makedirs(log_root, exist_ok=True)

    print("\n" + "=" * 118)
    print("🚀 Diff-Drive UGV / Jetbot Task3 Conservative Sim2Real Parking - TRUE skrl PPO Training")
    print("=" * 118)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_root     = {log_root}")
    print(f"[INFO] run_name     = {run_name}")
    print("[INFO] This version uses skrl PPO, not torch-native PPO or stable-baselines3.")

    env_cfg = Task3Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.device)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.max_episode_length_s = float(args_cli.max_episode_length_s)
    env_cfg.max_wheel_speed = float(args_cli.max_wheel_speed)
    env_cfg.left_wheel_sign = float(args_cli.left_wheel_sign)
    env_cfg.right_wheel_sign = float(args_cli.right_wheel_sign)
    env_cfg.print_debug_info = True

    curriculum = Task3Sim2RealCurriculum(
        total_env_steps=int(args_cli.total_env_steps),
        disabled=bool(args_cli.disable_curriculum),
    )

    dummy = type("DummyEnv", (), {})()
    dummy.cfg = env_cfg
    curriculum_info = curriculum.apply(dummy, int(args_cli.start_env_steps))

    env_cfg.validate()

    base_env = DiffDriveTask3Env(env_cfg)

    local_env = DiffDriveTask3SkrlWrapper(base_env)
    env = wrap_env(local_env, wrapper="isaaclab")
    num_envs = getattr(env, "num_envs", local_env.num_envs)

    if env.state_space is None:
        raise RuntimeError("env.state_space is None. Task3 requires privileged critic state space.")

    print("\n[DEBUG] Diff-Drive Task3 Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")

    assert env.observation_space.shape[0] == 404
    assert env.state_space.shape[0] == 442
    assert env.action_space.shape[0] == 2

    models = {
        "policy": DiffDriveTask3Actor(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            init_log_std=float(args_cli.init_log_std),
            min_log_std=float(args_cli.min_log_std),
            max_log_std=float(args_cli.max_log_std),
        ),
        "value": DiffDriveTask3Critic(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
        ),
    }

    cfg = build_skrl_cfg(env, log_dir=log_root, run_name=run_name)

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

    pretrained_loaded = False
    resumed = False

    if args_cli.resume:
        resumed = try_load_agent(agent, args_cli.resume, "resume checkpoint")
        sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)
    elif args_cli.pretrained:
        pretrained_loaded = try_load_agent(agent, args_cli.pretrained, "pretrained checkpoint")
        sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)

    total_env_steps = int(args_cli.total_env_steps)
    start_env_steps = int(args_cli.start_env_steps)
    remaining_env_steps = max(total_env_steps - start_env_steps, 1)
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
    print(f"  - num_envs              : {num_envs:,}")
    print(f"  - total_env_steps       : {total_env_steps:,}")
    print(f"  - start_env_steps       : {start_env_steps:,}")
    print(f"  - remaining_env_steps   : {remaining_env_steps:,}")
    print(f"  - total_vector_steps    : {total_vector_steps:,}")
    print(f"  - update_env_steps      : {update_env_steps:,}")
    print(f"  - save_freq_env_steps   : {save_freq_env_steps:,}")
    print(f"  - actor_obs_dim         : {env.observation_space.shape[0]}")
    print(f"  - critic_obs_dim        : {env.state_space.shape[0]}")
    print(f"  - action_dim            : {env.action_space.shape[0]}")
    print(f"  - rollouts              : {cfg['rollouts']}")
    print(f"  - learning_epochs       : {cfg.get('learning_epochs')}")
    print(f"  - mini_batches          : {cfg.get('mini_batches')}")
    print(f"  - lr                    : {cfg.get('learning_rate')}")
    print(f"  - max_episode_length    : {env_cfg.max_episode_length}")
    print(f"  - max_wheel_speed       : {env_cfg.max_wheel_speed}")
    print(f"  - curriculum            : {'disabled' if args_cli.disable_curriculum else 'enabled'}")
    print(f"  - tensorboard           : tensorboard --logdir={log_root}")

    print("\n🔥 [Diff-Drive Task3 TRUE skrl PPO 已点火]")
    print("👉 Stage0 弱随机化，先学会通过赛道并接近泊车位。")
    print("👉 后续逐步打开 delay / deadband / EMA / motor / wheel / LiDAR noise。")
    print("👉 Actor：404 维部署观测；Critic：442 维 privileged critic。")
    print("👉 Actor / action / optimizer / PPO tracking 均加入 NaN 防护。")
    print("👉 一旦 loss/std 出现 NaN，会保存 emergency checkpoint 并停止，避免污染模型。")
    print("👉 重点观察：Goal_Dist / Goal_Aligned_Speed / Progress / Parking_Pos_Error / Goal_Yaw_Error / Success_Rate / Crash_Rate。\n")

    last_save = start_env_steps
    update_id = 0
    start_time = time.time()
    env_steps = start_env_steps

    try:
        trainer.reset()

        with tqdm(
            total=total_env_steps,
            initial=min(start_env_steps, total_env_steps),
            desc="Diff-Drive Task3 skrl PPO",
            unit="steps",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.05,
        ) as pbar:
            for t in range(total_vector_steps):
                absolute_env_steps = min(start_env_steps + (t + 1) * int(num_envs), total_env_steps)

                curriculum_info = curriculum.apply(base_env, absolute_env_steps)

                trainer.train(timestep=t, timesteps=total_vector_steps)

                previous_env_steps = min(start_env_steps + t * int(num_envs), total_env_steps)
                pbar.update(max(absolute_env_steps - previous_env_steps, 0))
                env_steps = absolute_env_steps

                tel = local_env.last_info.get("telemetry", {})
                ev = local_env.last_info.get("events", {})
                rew = local_env.last_info.get("reward_components", {})

                elapsed = time.time() - start_time
                fps = max(env_steps - start_env_steps, 0) / max(elapsed, 1e-6)

                pbar.set_postfix(
                    {
                        "steps": f"{env_steps:,}",
                        "stage": int(curriculum_info.get("Curriculum_Stage", -1)),
                        "fps": f"{fps:,.0f}",
                        "rew": f"{local_env.last_reward_mean:+.3f}",
                        "dist": f"{tel.get('Goal_Dist', 0.0):.2f}",
                        "prog": f"{tel.get('Progress', 0.0):+.4f}",
                        "goal_v": f"{tel.get('Goal_Aligned_Speed', 0.0):+.3f}",
                        "park": f"{tel.get('Parking_Pos_Error', 0.0):.2f}",
                        "yaw": f"{tel.get('Goal_Yaw_Error', 0.0):.2f}",
                        "stuck": f"{tel.get('Stuck_Ratio', 0.0):.2f}",
                        "succ": f"{ev.get('Success_Rate', 0.0):.3f}",
                        "crash": f"{ev.get('Crash_Rate', 0.0):.3f}",
                        "timeout": f"{ev.get('Timeout_Rate', 0.0):.3f}",
                        "r_prog": f"{rew.get('R_Progress', 0.0):+.3f}",
                    }
                )

                writer = getattr(agent, "writer", None)
                write_scalars(writer, local_env.last_info.get("reward_components", {}), env_steps, "rewards")
                write_scalars(writer, local_env.last_info.get("events", {}), env_steps, "events")
                write_scalars(writer, local_env.last_info.get("telemetry", {}), env_steps, "telemetry")
                write_scalars(writer, local_env.last_info.get("world", {}), env_steps, "world")
                write_scalars(writer, local_env.last_info.get("debug", {}), env_steps, "debug")
                write_scalars(writer, curriculum_info, env_steps, "curriculum")

                try:
                    if writer is not None:
                        writer.add_scalar("rollout/reward_mean_raw", local_env.last_reward_mean, env_steps)
                        writer.add_scalar("rollout/done_count", local_env.last_done_count, env_steps)
                        writer.add_scalar("safety/nan_action_count", local_env.nan_action_count, env_steps)
                except Exception:
                    pass

                if (t + 1) % 32 == 0:
                    sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)

                if (t + 1) % int(cfg["rollouts"]) == 0:
                    update_id += 1
                    sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)

                    ppo_info = tracking_mean(agent)
                    ppo_info["learning_rate"] = current_lr(agent)

                    write_scalars(writer, ppo_info, env_steps, "ppo")
                    write_scalars(writer, flat_dict(local_env.last_info), env_steps, "env_info")
                    write_scalars(writer, curriculum_info, env_steps, "curriculum")

                    if update_id % max(int(args_cli.summary_interval), 1) == 0:
                        print_update(
                            pbar=pbar,
                            update_id=update_id,
                            env_steps=env_steps,
                            total_steps=total_env_steps,
                            elapsed=elapsed,
                            num_envs=int(num_envs),
                            rollouts=int(cfg["rollouts"]),
                            info=local_env.last_info,
                            ppo=ppo_info,
                            lr=ppo_info["learning_rate"],
                            curriculum_info=curriculum_info,
                        )

                    try:
                        agent.tracking_data.clear()
                    except Exception:
                        pass

                    bad, bad_key = ppo_info_has_nan(ppo_info)
                    if bad:
                        emergency_dir = os.path.join(log_root, run_name, f"emergency_nan_checkpoint_{env_steps}")
                        save_project_checkpoint(
                            emergency_dir,
                            agent=agent,
                            models=models,
                            env_cfg=env_cfg,
                            env=local_env,
                            env_steps=env_steps,
                            curriculum_info=curriculum_info,
                            args=args_cli,
                            extra={
                                "reason": f"ppo_nan_detected: {bad_key}",
                                "last_info": local_env.last_info,
                                "ppo_info": ppo_info,
                            },
                        )
                        raise RuntimeError(
                            f"PPO 数值异常：{bad_key}=NaN/Inf。已保存 emergency checkpoint: {emergency_dir}"
                        )

                if env_steps - last_save >= save_freq_env_steps:
                    last_save = env_steps
                    save_dir = os.path.join(log_root, run_name, f"checkpoint_{env_steps}")

                    try:
                        sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)
                        save_project_checkpoint(
                            save_dir,
                            agent=agent,
                            models=models,
                            env_cfg=env_cfg,
                            env=local_env,
                            env_steps=env_steps,
                            curriculum_info=curriculum_info,
                            args=args_cli,
                            extra={
                                "pretrained_loaded": pretrained_loaded,
                                "resumed": resumed,
                                "last_info": local_env.last_info,
                            },
                        )
                        pbar.write(f"\n💾 [Diff-Drive Task3 skrl 备份] 总步数: {env_steps:,} | 已保存至: {save_dir}\n")
                    except Exception as exc:
                        pbar.write(f"\n[WARN] checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

    except KeyboardInterrupt:
        print("\n[WARN] 接收到 Ctrl+C，正在安全保存当前 Diff-Drive Task3 skrl 模型...")
    except Exception:
        print("\n[ERROR] Diff-Drive Task3 skrl PPO 训练过程中发生真实异常：")
        traceback.print_exc()
        raise
    finally:
        final_dir = os.path.join(log_root, run_name, "final_checkpoint")
        final_env_steps = int(env_steps)

        try:
            sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)
            save_project_checkpoint(
                final_dir,
                agent=agent,
                models=models,
                env_cfg=env_cfg,
                env=local_env,
                env_steps=final_env_steps,
                curriculum_info=curriculum_info,
                args=args_cli,
                extra={
                    "final": True,
                    "pretrained_loaded": pretrained_loaded,
                    "resumed": resumed,
                    "last_info": local_env.last_info,
                },
            )
            print(f"✅ Diff-Drive Task3 skrl 模型已保存至 {final_dir}")
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

        print("✅ Diff-Drive Task3 TRUE skrl PPO training pipeline safely exited")


if __name__ == "__main__":
    main()

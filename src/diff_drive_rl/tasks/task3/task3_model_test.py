from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

parser = argparse.ArgumentParser(description="Evaluate Diff-Drive UGV / Jetbot Task3 TRUE skrl PPO model")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=1.0)
parser.add_argument("--print-interval", type=int, default=20)
parser.add_argument("--max-episode-length-s", type=float, default=40.0)
parser.add_argument("--max-wheel-speed", type=float, default=14.0)
parser.add_argument("--slow-action-scale", type=float, default=1.0)
parser.add_argument("--visualize", action="store_true")
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.headless = not bool(args_cli.visualize)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from diff_drive_rl.tasks.task3.task3_config import Task3Config
from diff_drive_rl.tasks.task3.task3_env import DiffDriveTask3Env
from skrl.models.torch import GaussianMixin, Model

try:
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
except Exception:
    sim_utils = None
    VisualizationMarkers = None
    VisualizationMarkersCfg = None


class Task3EvalCurriculum:
    """Evaluation-only copy of the Task3 curriculum.

    Do not import task3_train.py here because task3_train.py owns an AppLauncher
    entrypoint. This tiny copy only sets evaluation randomization ranges before
    env creation.
    """

    @staticmethod
    def stage_from_k(k: float) -> int:
        k = max(0.0, min(1.0, float(k)))
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
        wcfg.spot_width_inner_range = (0.60, 0.60)
        wcfg.spot_depth_inner_range = (0.73, 0.73)
        wcfg.bump_length_x_range = (0.45, 0.45)
        wcfg.bump_width_y_range = (2.80, 2.80)
        wcfg.bump_height_range = (0.006, 0.006)
        wcfg.bump_yaw_range = (0.0, 0.0)
        wcfg.bump_ramp_segments = 3
        wcfg.bump_top_length = 0.13
        wcfg.bump_low_height_ratio = 0.35

    @classmethod
    def apply(cls, cfg: Task3Config, start_k: float) -> Dict[str, float]:
        wcfg = cfg.world_cfg
        cls.lock_conservative_scene_geometry(wcfg)

        k = max(0.0, min(1.0, float(start_k)))
        stage = cls.stage_from_k(k)

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
            cfg.w_parking_pos = 0.12
            cfg.w_parking_yaw = 0.08
            cfg.w_lidar_risk = 0.04
            cfg.w_wall_risk = 0.05
            cfg.w_lane_risk = 0.08

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
            cfg.w_parking_pos = 0.32
            cfg.w_parking_yaw = 0.20
            cfg.w_lidar_risk = 0.08
            cfg.w_wall_risk = 0.10
            cfg.w_lane_risk = 0.12

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
            cfg.w_parking_pos = 0.58
            cfg.w_parking_yaw = 0.38
            cfg.w_lidar_risk = 0.13
            cfg.w_wall_risk = 0.16
            cfg.w_lane_risk = 0.17

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
            cfg.w_parking_pos = 0.72
            cfg.w_parking_yaw = 0.48
            cfg.w_lidar_risk = 0.17
            cfg.w_wall_risk = 0.24
            cfg.w_lane_risk = 0.22

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
            cfg.w_parking_pos = 0.80
            cfg.w_parking_yaw = 0.55
            cfg.w_lidar_risk = 0.20
            cfg.w_wall_risk = 0.30
            cfg.w_lane_risk = 0.25

        return {
            "Curriculum_K": float(k),
            "Curriculum_Stage": float(stage),
            "Delay_Max": float(wcfg.action_delay_frame_range[1]),
            "Deadband_Max": float(wcfg.action_deadband_range[1]),
            "Motor_Strength_Low": float(wcfg.motor_strength_range[0]),
            "Motor_Strength_High": float(wcfg.motor_strength_range[1]),
            "Lidar_Noise_Max": float(wcfg.lidar_noise_std_range[1]),
            "Reward_w_progress": float(cfg.w_progress),
            "Reward_w_goal_speed": float(cfg.w_goal_speed),
            "Reward_w_parking_pos": float(cfg.w_parking_pos),
            "Reward_w_parking_yaw": float(cfg.w_parking_yaw),
        }


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


class DiffDriveTask3EvalWrapper(gym.Env):
    def __init__(self, env: DiffDriveTask3Env):
        super().__init__()
        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = env.device
        self.observation_space = env.observation_space
        self.state_space = env.state_space
        self.action_space = env.action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return {"policy": obs.clone(), "critic": self.env.compute_privileged_obs().clone()}, info or {}

    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        return {"policy": obs.clone(), "critic": self.env.compute_privileged_obs().clone()}, rewards, terminated, truncated, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


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
            p / "diff_drive_task3_model.pt",
            p / "final_checkpoint" / "diff_drive_task3_model.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        pt_files = sorted(p.glob("*.pt"))
        for pt in pt_files:
            if pt.name in {"agent.pt", "diff_drive_task3_skrl_agent.pt"}:
                continue
            if pt.name.endswith("_preprocessor.pt"):
                continue
            return pt

    return p


def normalize_with_saved_obs_norm(obs: torch.Tensor, obs_norm: Optional[Dict[str, Any]]) -> torch.Tensor:
    if not obs_norm:
        return obs

    mean = obs_norm.get("mean", None)
    var = obs_norm.get("var", None)
    clip = float(obs_norm.get("clip", 10.0))

    if mean is None or var is None:
        return obs

    mean = mean.to(device=obs.device, dtype=torch.float32)
    var = var.to(device=obs.device, dtype=torch.float32)

    if mean.numel() != obs.shape[-1] or var.numel() != obs.shape[-1]:
        return obs

    return torch.clamp((obs - mean) / torch.sqrt(var + 1e-8), -clip, clip)


def load_policy_checkpoint(ckpt_path: Path, env: DiffDriveTask3EvalWrapper):
    ckpt = torch_load_checkpoint(ckpt_path, env.device)

    if not isinstance(ckpt, dict) or "policy" not in ckpt:
        raise RuntimeError(
            f"当前测试脚本需要 task3_train.py 保存的 eval checkpoint: diff_drive_task3_model.pt\n"
            f"当前文件不是 eval checkpoint: {ckpt_path}"
        )

    metadata = ckpt.get("metadata", {})
    args = ckpt.get("args", {})

    if not bool(metadata.get("uses_skrl", False)):
        raise RuntimeError("checkpoint metadata 缺少 uses_skrl=True，请使用当前 TRUE skrl 版本重新训练。")

    if not bool(metadata.get("asymmetric_actor_critic", False)):
        raise RuntimeError("checkpoint metadata 缺少 asymmetric_actor_critic=True。")

    expected_actor_dim = int(metadata.get("actor_obs_dim", env.observation_space.shape[0]))
    expected_critic_dim = int(metadata.get("critic_obs_dim", env.state_space.shape[0]))
    expected_action_dim = int(metadata.get("action_dim", env.action_space.shape[0]))

    if expected_actor_dim != env.observation_space.shape[0]:
        raise RuntimeError(f"actor obs dim mismatch: checkpoint={expected_actor_dim}, env={env.observation_space.shape[0]}")
    if expected_critic_dim != env.state_space.shape[0]:
        raise RuntimeError(f"critic obs dim mismatch: checkpoint={expected_critic_dim}, env={env.state_space.shape[0]}")
    if expected_action_dim != env.action_space.shape[0]:
        raise RuntimeError(f"action dim mismatch: checkpoint={expected_action_dim}, env={env.action_space.shape[0]}")

    if int(metadata.get("single_actor_obs_dim", 101)) != 101:
        raise RuntimeError(f"single_actor_obs_dim mismatch: {metadata.get('single_actor_obs_dim')}")
    if int(metadata.get("frame_stack", 4)) != 4:
        raise RuntimeError(f"frame_stack mismatch: {metadata.get('frame_stack')}")
    if int(metadata.get("lidar_pool_bins", 36)) != 36:
        raise RuntimeError(f"lidar_pool_bins mismatch: {metadata.get('lidar_pool_bins')}")

    policy = DiffDriveTask3Actor(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        init_log_std=float(args.get("init_log_std", -1.0)),
        min_log_std=float(args.get("min_log_std", -3.0)),
        max_log_std=float(args.get("max_log_std", 0.5)),
    ).to(env.device)

    policy.load_state_dict(ckpt["policy"], strict=True)
    policy.eval()

    actor_obs_norm = ckpt.get("actor_obs_norm", None)
    trained_env_steps = int(ckpt.get("env_steps", 0))
    curriculum_info = ckpt.get("curriculum_info", {})

    return policy, actor_obs_norm, trained_env_steps, metadata, curriculum_info


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
    print("Diff-Drive UGV / Jetbot Task3 TRUE skrl PPO Model Test Summary")
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


def create_goal_visualizer():
    if VisualizationMarkers is None or VisualizationMarkersCfg is None or sim_utils is None:
        print("[WARN] VisualizationMarkers unavailable. Goal marker disabled.")
        return None

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/DiffDriveTask3Goal",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.18,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                    emissive_color=(0.0, 0.8, 0.0),
                ),
            )
        },
    )
    return VisualizationMarkers(marker_cfg)


def update_goal_visualizer(visualizer, base_env: DiffDriveTask3Env):
    if visualizer is None:
        return

    try:
        goal_3d = torch.zeros((base_env.num_envs, 3), dtype=torch.float32, device=base_env.device)
        goal_3d[:, :2] = base_env.env_origins[:, :2] + base_env.world.goal_pos
        goal_3d[:, 2] = base_env.env_origins[:, 2] + 0.12
        visualizer.visualize(translations=goal_3d)
    except Exception as exc:
        print(f"[WARN] goal visualization update failed: {type(exc).__name__}: {exc}")


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    cfg = Task3Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.device)
    cfg.seed = int(args_cli.seed)
    cfg.max_episode_length_s = float(args_cli.max_episode_length_s)
    cfg.max_wheel_speed = float(args_cli.max_wheel_speed)
    cfg.print_debug_info = False

    eval_curriculum_info = Task3EvalCurriculum.apply(cfg, float(args_cli.start_k))
    cfg.validate()

    base_env = DiffDriveTask3Env(cfg)
    env = DiffDriveTask3EvalWrapper(base_env)

    obs_dict, _ = env.reset(seed=int(args_cli.seed))

    ckpt_path = resolve_checkpoint(args_cli.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    policy, actor_obs_norm, trained_env_steps, metadata, ckpt_curriculum_info = load_policy_checkpoint(ckpt_path, env)

    visualizer = create_goal_visualizer() if bool(args_cli.visualize) else None

    print("\n" + "=" * 150)
    print("Diff-Drive UGV / Jetbot Task3 TRUE skrl PPO model test started")
    print("=" * 150)
    print(f"checkpoint          : {ckpt_path}")
    print(f"trained_env_steps   : {trained_env_steps:,}")
    print(f"num_envs            : {base_env.num_envs}")
    print(f"steps               : {args_cli.steps}")
    print(f"start_k             : {args_cli.start_k}")
    print(f"eval_stage          : {eval_curriculum_info.get('Curriculum_Stage', -1)}")
    print(f"actor_obs_dim       : {base_env.num_observations}")
    print(f"critic_obs_dim      : {base_env.num_privileged_obs}")
    print(f"single_obs_dim      : {cfg.single_actor_obs_dim}")
    print(f"frame_stack         : {cfg.frame_stack}")
    print(f"world_priv_dim      : {cfg.privileged_feature_dim}")
    print(f"lidar_pool_bins     : {cfg.world_cfg.lidar_pool_bins}")
    print(f"action_dim          : {base_env.num_actions}")
    print(f"max_wheel_speed     : {cfg.max_wheel_speed}")
    print(f"slow_action_scale   : {args_cli.slow_action_scale}")
    print(f"device              : {base_env.device}")
    print(f"visualize           : {bool(args_cli.visualize)}")
    print("algorithm           : skrl PPO")
    print("policy forward      : deterministic direct policy forward; no agent.act")
    print("checkpoint metadata : uses_skrl=True, asymmetric_actor_critic=True")
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
            desc="Diff-Drive Task3 skrl Model Test",
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
                    print(f"[DEBUG][eval step {step}] action mean={actions.mean().item():+.6f}", flush=True)

                obs_dict, rewards, terminated, truncated, info = env.step(actions)

                total_terminated += int(terminated.sum().item())
                total_truncated += int(truncated.sum().item())

                if bool(args_cli.visualize):
                    update_goal_visualizer(visualizer, base_env)
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
                            "dist": f"{flat.get('telemetry/Goal_Dist', 0.0):.2f}",
                            "prog": f"{flat.get('telemetry/Progress', 0.0):+.4f}",
                            "goal_v": f"{flat.get('telemetry/Goal_Aligned_Speed', 0.0):+.3f}",
                            "park": f"{flat.get('telemetry/Parking_Pos_Error', 0.0):.2f}",
                            "yaw": f"{flat.get('telemetry/Goal_Yaw_Error', 0.0):.2f}",
                            "risk": f"{flat.get('telemetry/Risk_Front', 0.0):.2f}",
                            "succ": f"{flat.get('events/Success_Rate', 0.0):.3f}",
                            "crash": f"{flat.get('events/Crash_Rate', 0.0):.3f}",
                        }
                    )

                    if bool(args_cli.visualize):
                        sys.stdout.write(
                            f"\r🚗 "
                            f"Dist={flat.get('telemetry/Goal_Dist', 0.0):.3f} | "
                            f"Prog={flat.get('telemetry/Progress', 0.0):+.4f} | "
                            f"GoalV={flat.get('telemetry/Goal_Aligned_Speed', 0.0):+.3f} | "
                            f"ParkErr={flat.get('telemetry/Parking_Pos_Error', 0.0):.3f} | "
                            f"YawErr={flat.get('telemetry/Goal_Yaw_Error', 0.0):.3f} | "
                            f"LidarMin={flat.get('telemetry/Lidar_Min', 0.0):.3f} | "
                            f"RiskF={flat.get('telemetry/Risk_Front', 0.0):.3f} | "
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
        env_steps = int(args_cli.steps) * int(base_env.num_envs)
        fps = env_steps / max(elapsed, 1e-6)

        print("\n✅ Diff-Drive Task3 TRUE skrl PPO model test rollout finished")
        print(f"  env steps        : {env_steps:,}")
        print(f"  fps              : {fps:,.2f}")
        print(f"  total terminated : {total_terminated:,}")
        print(f"  total truncated  : {total_truncated:,}")

        print_summary_table(summarize(records))

        print("Diff-Drive Task3 model test checklist:")
        print("1. checkpoint metadata 必须标记 uses_skrl=True。")
        print("2. checkpoint metadata 必须标记 asymmetric_actor_critic=True。")
        print("3. actor obs 必须为 404 维，critic obs 必须为 442 维。")
        print("4. 测试脚本不调用 agent.act，不使用 non-skrl legacy evaluation framework。")
        print("5. smoke checkpoint 效果差是正常的，先看推理稳定性和无 NaN/Inf。")
        print("6. 正式效果重点看 Goal_Dist、Progress、Goal_Aligned_Speed、Parking_Pos_Error、Goal_Yaw_Error、Success_Rate、Crash_Rate。")

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

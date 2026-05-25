from __future__ import annotations

import argparse
import os
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

parser = argparse.ArgumentParser(description="Evaluate Diff-Drive UGV / Jetbot Task1 TRUE skrl PPO model")

parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--print-interval", type=int, default=20)
parser.add_argument("--max-episode-length-s", type=float, default=12.0)
parser.add_argument("--num-waypoints", type=int, default=3)
parser.add_argument("--visualize", action="store_true")

# Kept for unified CLI style with other robots. Task1 has no curriculum.
parser.add_argument("--start-k", type=float, default=1.0)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = not bool(args_cli.visualize)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from diff_drive_rl.tasks.task1.task1_config import Task1Config
from diff_drive_rl.tasks.task1.task1_env import DiffDriveTask1Env

from skrl.models.torch import GaussianMixin, Model

try:
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
except Exception:
    sim_utils = None
    VisualizationMarkers = None
    VisualizationMarkersCfg = None


class DiffDriveActor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        init_log_std: float = -0.60,
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
            nn.Linear(observation_space.shape[0], 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_space.shape[0]),
        )

        self.log_std_parameter = nn.Parameter(torch.full((action_space.shape[0],), float(init_log_std)))

    def compute(self, inputs, role):
        states = inputs.get("observations", inputs.get("states"))
        actions = self.net(states)
        return actions, {"log_std": self.log_std_parameter}

    @torch.no_grad()
    def act_deterministic_direct(self, states: torch.Tensor) -> torch.Tensor:
        actions, _ = self.compute({"states": states}, role="policy")
        return torch.clamp(actions, -1.0, 1.0)


class DiffDriveTask1EvalWrapper(gym.Env):
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

        self.observation_space = self.policy_space
        self.state_space = self.critic_space
        self.action_space = env.action_space

        self.last_obs = torch.zeros((self.num_envs, self.obs_dim), dtype=torch.float32, device=self.device)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs.clone()
        return {"policy": obs.clone(), "critic": obs.clone()}, info or {}

    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        self.last_obs = obs.clone()
        return {"policy": obs.clone(), "critic": obs.clone()}, rewards, terminated, truncated, info

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
            p / "diff_drive_task1_model.pt",
            p / "final_checkpoint" / "diff_drive_task1_model.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        pt_files = sorted(p.glob("*.pt"))
        for pt in pt_files:
            if pt.name in {"agent.pt", "diff_drive_task1_skrl_agent.pt"}:
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


def load_policy_checkpoint(ckpt_path: Path, env: DiffDriveTask1EvalWrapper):
    ckpt = torch_load_checkpoint(ckpt_path, env.device)

    if not isinstance(ckpt, dict) or "policy" not in ckpt:
        raise RuntimeError(
            f"当前测试脚本需要 task1_train.py 保存的 eval checkpoint: diff_drive_task1_model.pt\n"
            f"收到的文件不是 eval checkpoint: {ckpt_path}\n"
            "请传入 final_checkpoint 目录，或传入 final_checkpoint/diff_drive_task1_model.pt。"
        )

    metadata = ckpt.get("metadata", {})
    args = ckpt.get("args", {})

    if not bool(metadata.get("uses_skrl", False)):
        raise RuntimeError(
            f"Checkpoint is not marked as skrl PPO: {ckpt_path}\n"
            "请使用当前 TRUE skrl 版本 task1_train.py 重新训练。"
        )

    expected_actor_dim = int(metadata.get("actor_obs_dim", env.observation_space.shape[0]))
    expected_critic_dim = int(metadata.get("critic_obs_dim", env.state_space.shape[0]))
    expected_action_dim = int(metadata.get("action_dim", env.action_space.shape[0]))

    if expected_actor_dim != env.observation_space.shape[0]:
        raise RuntimeError(
            f"actor obs dim mismatch: checkpoint={expected_actor_dim}, env={env.observation_space.shape[0]}"
        )

    if expected_critic_dim != env.state_space.shape[0]:
        raise RuntimeError(
            f"critic obs dim mismatch: checkpoint={expected_critic_dim}, env={env.state_space.shape[0]}"
        )

    if expected_action_dim != env.action_space.shape[0]:
        raise RuntimeError(
            f"action dim mismatch: checkpoint={expected_action_dim}, env={env.action_space.shape[0]}"
        )

    policy = DiffDriveActor(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        init_log_std=float(args.get("init_log_std", -0.60)),
        min_log_std=float(args.get("min_log_std", -4.0)),
        max_log_std=float(args.get("max_log_std", 0.30)),
    ).to(env.device)

    policy.load_state_dict(ckpt["policy"], strict=True)
    policy.eval()

    actor_obs_norm = ckpt.get("actor_obs_norm", None)
    trained_env_steps = int(ckpt.get("env_steps", 0))

    return policy, actor_obs_norm, trained_env_steps, metadata


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
    print("Diff-Drive UGV / Jetbot Task1 TRUE skrl PPO Model Test Summary")
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


def create_waypoint_visualizer():
    if VisualizationMarkers is None or VisualizationMarkersCfg is None or sim_utils is None:
        print("[WARN] VisualizationMarkers unavailable. Waypoint markers disabled.")
        return None

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/DiffDriveTask1Waypoints",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                    emissive_color=(0.0, 0.8, 0.0),
                ),
            )
        },
    )
    return VisualizationMarkers(marker_cfg)


def update_waypoint_visualizer(visualizer, base_env: DiffDriveTask1Env):
    if visualizer is None:
        return

    try:
        target_3d = base_env.current_waypoint_world()
        visualizer.visualize(translations=target_3d)
    except Exception as exc:
        print(f"[WARN] waypoint visualization update failed: {type(exc).__name__}: {exc}")


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    cfg = Task1Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.device)
    cfg.seed = int(args_cli.seed)
    cfg.max_episode_length_s = float(args_cli.max_episode_length_s)
    cfg.num_waypoints = int(args_cli.num_waypoints)
    cfg.print_debug_info = False
    cfg.validate()

    base_env = DiffDriveTask1Env(cfg)
    env = DiffDriveTask1EvalWrapper(base_env)

    obs_dict, _ = env.reset(seed=int(args_cli.seed))

    ckpt_path = resolve_checkpoint(args_cli.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    policy, actor_obs_norm, trained_env_steps, metadata = load_policy_checkpoint(ckpt_path, env)

    visualizer = create_waypoint_visualizer() if bool(args_cli.visualize) else None

    print("\n" + "=" * 150)
    print("Diff-Drive UGV / Jetbot Task1 TRUE skrl PPO model test started")
    print("=" * 150)
    print(f"checkpoint         : {ckpt_path}")
    print(f"trained_env_steps  : {trained_env_steps:,}")
    print(f"num_envs           : {base_env.num_envs}")
    print(f"steps              : {args_cli.steps}")
    print(f"start_k            : {args_cli.start_k}  (accepted for unified CLI; Task1 has no curriculum)")
    print(f"num_waypoints      : {cfg.num_waypoints}")
    print(f"max_episode_length : {cfg.max_episode_length}")
    print(f"obs_dim            : {base_env.num_observations}")
    print(f"action_dim         : {base_env.num_actions}")
    print(f"device             : {base_env.device}")
    print(f"visualize          : {bool(args_cli.visualize)}")
    print("algorithm          : skrl PPO")
    print("control            : left/right wheel velocity")
    print("note               : deterministic direct policy forward; no agent.act")
    print("=" * 150 + "\n")

    records: List[Dict[str, float]] = []
    total_terminated = 0
    total_truncated = 0

    start_time = time.time()

    try:
        with tqdm(
            total=int(args_cli.steps),
            desc="Diff-Drive Task1 skrl Model Test",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as pbar:
            for step in range(int(args_cli.steps)):
                if step < 3:
                    print(f"[DEBUG][eval step {step}] before policy forward", flush=True)

                with torch.no_grad():
                    actor_obs = obs_dict["policy"]
                    actor_obs_n = normalize_with_saved_obs_norm(actor_obs, actor_obs_norm)
                    actions = policy.act_deterministic_direct(actor_obs_n)

                if step < 3:
                    print(f"[DEBUG][eval step {step}] after policy forward", flush=True)
                    print(f"[DEBUG][eval step {step}] before env.step", flush=True)

                obs_dict, rewards, terminated, truncated, info = env.step(actions)

                if step < 3:
                    print(f"[DEBUG][eval step {step}] after env.step", flush=True)

                total_terminated += int(terminated.sum().item())
                total_truncated += int(truncated.sum().item())

                if bool(args_cli.visualize):
                    update_waypoint_visualizer(visualizer, base_env)
                    elapsed_step = 0.0
                    try:
                        # Slow down GUI playback roughly to policy rate.
                        elapsed_step = 0.0
                        sleep_time = float(cfg.policy_dt) - elapsed_step
                        if sleep_time > 0:
                            time.sleep(sleep_time)
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
                            "dist": f"{flat.get('telemetry/Distance_To_Waypoint', 0.0):.2f}",
                            "prog": f"{flat.get('telemetry/Progress', 0.0):+.3f}",
                            "goal_v": f"{flat.get('telemetry/Goal_Aligned_Speed', 0.0):+.2f}",
                            "wp": f"{flat.get('telemetry/Waypoint_Index', 0.0):.2f}",
                            "finish": f"{flat.get('events/Episode_Finish_Rate', 0.0):.3f}",
                            "timeout": f"{flat.get('events/Episode_Timeout_Rate', 0.0):.3f}",
                        }
                    )

                    if bool(args_cli.visualize):
                        sys.stdout.write(
                            f"\r🚗 Dist={flat.get('telemetry/Distance_To_Waypoint', 0.0):.3f} | "
                            f"Prog={flat.get('telemetry/Progress', 0.0):+.4f} | "
                            f"GoalV={flat.get('telemetry/Goal_Aligned_Speed', 0.0):+.3f} | "
                            f"HeadErr={flat.get('telemetry/Heading_Error', 0.0):.3f} | "
                            f"Vx={flat.get('telemetry/Body_Vx', 0.0):+.3f} | "
                            f"Wz={flat.get('telemetry/Body_Wz', 0.0):+.3f} | "
                            f"WPIdx={flat.get('telemetry/Waypoint_Index', 0.0):.2f} | "
                            f"R={row['test/reward_mean']:+.3f} | "
                            f"Finish={flat.get('events/Finish_Rate', 0.0):.3f} | "
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

        print("\n✅ Diff-Drive Task1 TRUE skrl PPO model test rollout finished")
        print(f"  env steps        : {env_steps:,}")
        print(f"  fps              : {fps:,.2f}")
        print(f"  total terminated : {total_terminated:,}")
        print(f"  total truncated  : {total_truncated:,}")

        print_summary_table(summarize(records))

        print("Diff-Drive Task1 model test checklist:")
        print("1. checkpoint metadata 必须标记 uses_skrl=True。")
        print("2. 当前 obs_dim 必须是 36，不再使用旧测试里的 21 维 observation_space。")
        print("3. 测试脚本不调用 agent.act，避免模型测试卡在 0%。")
        print("4. smoke checkpoint 效果差是正常的，先看推理稳定性和无 NaN/Inf。")
        print("5. 正式效果重点看 Progress、Goal_Aligned_Speed、Waypoint_Index、Finish_Rate、Timeout_Rate、Stuck_Ratio。")
        print("6. GUI 可视化时绿色球表示当前 waypoint。")

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

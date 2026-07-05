from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diff-Drive UGV Task2 Analytic Obstacle Navigation Env Test")
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--rollout-k", type=float, default=0.20)
parser.add_argument("--collect-interval", type=int, default=250)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-names", action="store_true")
parser.add_argument("--strict-action-test", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

simulation_app = AppLauncher(args_cli).app

from diff_drive_rl.tasks.task2.task2_config import Task2Config
from diff_drive_rl.tasks.task2.task2_env import DiffDriveTask2Env


def heading(title: str) -> None:
    print("\n" + "=" * 140)
    print(title)
    print("=" * 140, flush=True)


def print_ok(msg: str) -> None:
    print(f" ✅ {msg}", flush=True)


def print_warn(msg: str) -> None:
    print(f" ⚠️ {msg}", flush=True)


def assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    assert torch.is_tensor(x), f"{name} 必须是 torch.Tensor，当前为 {type(x)}"
    assert torch.isfinite(x).all().item(), f"{name} 出现 NaN 或 Inf"


def check_shape(name: str, x: torch.Tensor, expected) -> None:
    assert tuple(x.shape) == tuple(expected), f"{name} shape 错误: {tuple(x.shape)} != {tuple(expected)}"


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float().flatten()
    return {
        "mean": x.mean().item(),
        "min": x.min().item(),
        "p10": torch.quantile(x, 0.10).item(),
        "p50": torch.quantile(x, 0.50).item(),
        "p90": torch.quantile(x, 0.90).item(),
        "max": x.max().item(),
    }


def print_stats(name: str, x: torch.Tensor) -> None:
    s = tensor_stats(x)
    print(
        f"{name:<40s} "
        f"mean={s['mean']:+.6f} | min={s['min']:+.6f} | p10={s['p10']:+.6f} | "
        f"p50={s['p50']:+.6f} | p90={s['p90']:+.6f} | max={s['max']:+.6f}",
        flush=True,
    )


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


def flatten_info(info: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in (info or {}).items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_info(value, name))
        else:
            val = to_float(value)
            if val is not None and math.isfinite(val):
                out[name] = val
    return out


def summarize_records(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not records:
        return {}

    keys = sorted({k for row in records for k in row.keys()})
    out: Dict[str, Dict[str, float]] = {}

    for key in keys:
        vals = np.asarray([row[key] for row in records if key in row], dtype=np.float64)
        if vals.size == 0:
            continue

        out[key] = {
            "mean": float(np.mean(vals)),
            "var": float(np.var(vals)),
            "min": float(np.min(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "max": float(np.max(vals)),
        }

    return out


def print_summary_table(summary: Dict[str, Dict[str, float]]) -> None:
    if not summary:
        print_warn("没有收集到有效统计字段")
        return

    print("\n" + "=" * 188)
    print(" " * 48 + "Diff-Drive UGV Task2 Env 统计报告")
    print("=" * 188)
    print(
        f"{'metric':<82} | {'mean':>11} | {'var':>11} | {'min':>11} | "
        f"{'p25':>11} | {'p50':>11} | {'p75':>11} | {'max':>11}"
    )
    print("-" * 188)

    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<82} | "
            f"{row['mean']:>11.5f} | "
            f"{row['var']:>11.5f} | "
            f"{row['min']:>11.5f} | "
            f"{row['p25']:>11.5f} | "
            f"{row['p50']:>11.5f} | "
            f"{row['p75']:>11.5f} | "
            f"{row['max']:>11.5f}"
        )

    print("=" * 188 + "\n")


def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros((yaw.shape[0], 4), dtype=torch.float32, device=yaw.device)
    quat[:, 0] = torch.cos(yaw * 0.5)
    quat[:, 3] = torch.sin(yaw * 0.5)
    return quat


def force_root_local_pose(
    env: DiffDriveTask2Env,
    env_ids: torch.Tensor,
    local_xy: torch.Tensor | None = None,
    height: float | None = None,
    yaw: torch.Tensor | None = None,
    zero_vel: bool = True,
):
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
    n = int(env_ids.numel())

    root_state = torch.zeros((n, 13), dtype=torch.float32, device=env.device)
    root_state[:, 0:3] = env.robot.data.root_pos_w[env_ids]
    root_state[:, 3:7] = env.robot.data.root_quat_w[env_ids]

    if hasattr(env.robot.data, "root_lin_vel_w"):
        root_state[:, 7:10] = env.robot.data.root_lin_vel_w[env_ids]
    else:
        root_state[:, 7:10] = env.robot.data.root_lin_vel_b[env_ids]

    if hasattr(env.robot.data, "root_ang_vel_w"):
        root_state[:, 10:13] = env.robot.data.root_ang_vel_w[env_ids]
    else:
        root_state[:, 10:13] = env.robot.data.root_ang_vel_b[env_ids]

    if local_xy is not None:
        local_xy = torch.as_tensor(local_xy, dtype=torch.float32, device=env.device)
        if local_xy.ndim == 1:
            local_xy = local_xy.unsqueeze(0).repeat(n, 1)
        root_state[:, 0:2] = env.env_origins[env_ids, :2] + local_xy

    if height is not None:
        root_state[:, 2] = env.env_origins[env_ids, 2] + float(height)

    if yaw is not None:
        yaw = torch.as_tensor(yaw, dtype=torch.float32, device=env.device)
        if yaw.ndim == 0:
            yaw = yaw.repeat(n)
        root_state[:, 3:7] = yaw_to_quat_wxyz(yaw)

    if zero_vel:
        root_state[:, 7:13] = 0.0

    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    env.scene.update(dt=0.0)


def disable_all_obstacles(env: DiffDriveTask2Env) -> None:
    env.world.static_mask[:] = False
    env.world.dynamic_mask[:] = False
    env.world.static_pos[:] = 0.0
    env.world.dynamic_pos[:] = 0.0
    env.world.static_radius[:] = 0.0
    env.world.dynamic_radius[:] = 0.0
    env.world.dynamic_vel[:] = 0.0


def check_obs(env: DiffDriveTask2Env, obs: torch.Tensor) -> None:
    check_shape("obs", obs, (env.num_envs, env.num_observations))
    assert_finite_tensor("obs", obs)
    assert obs.abs().max().item() <= float(env.cfg.obs_clip) + 1e-5, (
        f"obs 超出 clamp 范围 [-{env.cfg.obs_clip}, {env.cfg.obs_clip}]"
    )



def get_single_obs_slices(cfg: Task2Config):
    """返回 Task2 单帧观测切片。

    前 14 维必须严格遵守 core navigation：
        0      goal_dist_norm
        1:3    goal_x_body_norm, goal_y_body_norm
        3:5    sin_heading_error, cos_heading_error
        5:8    body_vx, body_vy, body_wz
        8      target_speed_norm
        9      last_forward_throttle
        10     last_turn_command
        11     action_delta_forward
        12     action_delta_turn
        13     progress_ema
    """
    s = {}
    idx = 0

    s["goal_dist_norm"] = slice(idx, idx + 1); idx += 1
    s["goal_xy_body"] = slice(idx, idx + 2); idx += 2
    s["heading"] = slice(idx, idx + 2); idx += 2
    s["vel_obs"] = slice(idx, idx + 3); idx += 3
    s["target_speed"] = slice(idx, idx + 1); idx += 1
    s["last_forward_turn"] = slice(idx, idx + 2); idx += 2
    s["action_delta_forward_turn"] = slice(idx, idx + 2); idx += 2
    s["progress_ema"] = slice(idx, idx + 1); idx += 1

    # Backward-compatible aliases for tests that only care about dimensions.
    s["actions"] = s["last_forward_turn"]
    s["action_delta"] = s["action_delta_forward_turn"]

    s["core_nav"] = slice(0, idx)
    s["lidar"] = slice(idx, idx + cfg.world_cfg.num_lidar_rays); idx += cfg.world_cfg.num_lidar_rays
    s["lidar_delta"] = slice(idx, idx + cfg.world_cfg.num_lidar_rays); idx += cfg.world_cfg.num_lidar_rays
    s["risk"] = slice(idx, idx + 8); idx += 8

    assert s["core_nav"].stop == int(getattr(cfg, "core_single_obs_dim", 14)), (
        f"core navigation 维度错误: {s['core_nav'].stop} != {getattr(cfg, 'core_single_obs_dim', 14)}"
    )
    assert idx == int(cfg.single_obs_dim), f"单帧 obs slice 总维度错误: {idx} != {cfg.single_obs_dim}"
    return s


def set_env_to_stage(env: DiffDriveTask2Env, stage_idx: int, eps: float = 1e-6) -> None:
    """按 cfg.world_cfg.stage_thresholds 动态设置 env.global_steps。

    不再在测试中硬编码 0.08/0.20/0.38 等旧课程阈值，避免课程配置
    更新后白盒测试失效。
    """
    thresholds = list(env.cfg.world_cfg.stage_thresholds)
    assert 0 <= stage_idx < len(thresholds), f"stage_idx 越界: {stage_idx}"
    k = float(thresholds[stage_idx]) + (eps if stage_idx > 0 else 0.0)
    k = min(max(k, 0.0), 1.0)
    env.global_steps = int(k * env.cfg.world_cfg.curriculum_total_steps)


def stage_k(env: DiffDriveTask2Env, stage_idx: int, eps: float = 1e-6) -> float:
    thresholds = list(env.cfg.world_cfg.stage_thresholds)
    assert 0 <= stage_idx < len(thresholds), f"stage_idx 越界: {stage_idx}"
    k = float(thresholds[stage_idx]) + (eps if stage_idx > 0 else 0.0)
    return min(max(k, 0.0), 1.0)


def run_fixed_action(env: DiffDriveTask2Env, action_tensor: torch.Tensor, steps: int = 80):
    env.reset()
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    disable_all_obstacles(env)
    env.world.start_pos[:] = 0.0
    env.world.goal_pos[:] = torch.tensor([8.0, 0.0], dtype=torch.float32, device=env.device)
    env.world.env_target_speed[:] = 0.6

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device),
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
    )

    root0 = env._root_pos_local().clone()
    yaw0 = env._yaw().clone()
    env.last_goal_dist[:] = torch.norm(env.world.goal_pos - env._root_pos_local(), dim=-1)

    action = action_tensor.to(env.device).view(1, 2).repeat(env.num_envs, 1)

    info = {}
    reward = None
    terminated = None
    truncated = None
    obs = None

    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(action)

    root1 = env._root_pos_local().clone()
    yaw1 = env._yaw().clone()

    delta_pos = root1 - root0
    delta_yaw = torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0))

    return delta_pos, delta_yaw, obs, reward, terminated, truncated, info

def check_project_files() -> None:
    heading("[测试 0] Task2 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task2_obstacle_navigation.yaml",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task2" / "task2_world.py",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task2" / "task2_config.py",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task2" / "task2_scene.py",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task2" / "task2_env.py",
    ]

    missing = [str(p) for p in required if not p.exists()]
    assert not missing, "缺少 Task2 必要文件:\n" + "\n".join(missing)

    for path in required:
        print_ok(str(path.relative_to(PROJECT_ROOT)))



def check_config() -> None:
    heading("[测试 1] Task2Config 基础配置与协议检测")

    cfg = Task2Config()
    cfg.validate()

    assert cfg.num_actions == 2
    assert cfg.single_obs_dim == 166
    assert cfg.frame_stack == 3
    assert cfg.num_observations == 498
    assert cfg.world_cfg.num_lidar_rays == 72

    assert getattr(cfg, "action_protocol", "forward_throttle_turn") == "forward_throttle_turn"
    assert int(getattr(cfg, "core_single_obs_dim", 14)) == 14
    assert int(getattr(cfg, "stacked_core_obs_dim", 42)) == 42
    assert int(getattr(cfg, "extra_single_obs_dim", 152)) == 152
    assert int(getattr(cfg, "stacked_extra_obs_dim", 456)) == 456

    print_ok(f"num_actions = {cfg.num_actions}")
    print_ok(f"single_obs_dim = {cfg.single_obs_dim}")
    print_ok(f"frame_stack = {cfg.frame_stack}")
    print_ok(f"num_observations = {cfg.num_observations}")
    print_ok(f"core navigation single dim = {getattr(cfg, 'core_single_obs_dim', 14)}")
    print_ok(f"core navigation stacked dim = {getattr(cfg, 'stacked_core_obs_dim', 42)}")
    print_ok(f"Task2 extra stacked dim = {getattr(cfg, 'stacked_extra_obs_dim', 456)}")
    print_ok(f"action_protocol = {getattr(cfg, 'action_protocol', 'forward_throttle_turn')}")
    print_ok(f"world.num_lidar_rays = {cfg.world_cfg.num_lidar_rays}")
    print_ok(f"max_episode_length = {cfg.max_episode_length}")
    print_ok("Task2Config 基础配置正常")

def check_env_init(cfg: Task2Config) -> DiffDriveTask2Env:
    heading("[测试 2] DiffDriveTask2Env 初始化 / 名称映射 / 空间维度检测")

    env = DiffDriveTask2Env(cfg)

    print_ok(f"device = {cfg.device}")
    print_ok(f"num_envs = {cfg.num_envs}")
    print_ok(f"robot.num_joints = {env.robot.num_joints}")
    print_ok(f"num_actions = {env.num_actions}")
    print_ok(f"single_obs_dim = {env.cfg.single_obs_dim}")
    print_ok(f"frame_stack = {env.cfg.frame_stack}")
    print_ok(f"num_observations = {env.num_observations}")
    print_ok(f"policy_dt = {env.dt}")
    print_ok(f"max_episode_length = {env.cfg.max_episode_length}")
    print_ok(f"wheel_joint_ids = {env.wheel_joint_ids}")

    assert env.robot.num_joints >= 2, f"关节数量异常: {env.robot.num_joints}"
    assert env.num_actions == 2
    assert env.cfg.single_obs_dim == 166
    assert env.num_observations == env.cfg.frame_stack * env.cfg.single_obs_dim
    assert env.num_observations == 498
    assert len(env.wheel_joint_ids) == 2

    assert env.observation_space.shape == (env.num_observations,)
    assert env.state_space.shape == (env.num_observations,)
    assert env.action_space.shape == (2,)

    if args_cli.print_names:
        print("\nrobot.joint_names:")
        for i, name in enumerate(env.robot_joint_names):
            mark = " <wheel>" if i in env.wheel_joint_ids else ""
            print(f"  {i:02d}: {name}{mark}")

        print("\nrobot.body_names:")
        for i, name in enumerate(env.robot_body_names):
            print(f"  {i:02d}: {name}")

    return env


def test_reset_step_basic(env: DiffDriveTask2Env):
    heading("[测试 3] reset / obs / step 基础检测")

    obs, _ = env.reset()
    check_obs(env, obs)

    print_ok(f"reset obs shape = {tuple(obs.shape)}")
    print_ok(f"obs range = {obs.min().item():+.6f} ~ {obs.max().item():+.6f}")

    action = torch.rand((env.num_envs, env.num_actions), device=env.device) * 0.4 - 0.2
    obs, reward, terminated, truncated, info = env.step(action)

    check_obs(env, obs)
    check_shape("reward", reward, (env.num_envs,))
    check_shape("terminated", terminated, (env.num_envs,))
    check_shape("truncated", truncated, (env.num_envs,))
    assert_finite_tensor("reward", reward)

    print_ok(f"reward mean = {reward.mean().item():+.6f}")
    print_ok(f"terminated count = {terminated.sum().item()}")
    print_ok(f"truncated count = {truncated.sum().item()}")


def test_reset_alignment(env: DiffDriveTask2Env):
    heading("[测试 4] reset 后 robot local xy 与 world.start_pos 对齐检测")

    old_steps = int(env.global_steps)
    env.global_steps = 0

    obs, _ = env.reset()
    check_obs(env, obs)

    root_local = env._root_pos_local()
    xy_err = torch.norm(root_local - env.world.start_pos, dim=-1)

    assert xy_err.max().item() < 5e-4, (
        f"reset 后 root local xy 未与 world.start_pos 对齐，max_err={xy_err.max().item()}"
    )

    goal_dist = torch.norm(env.world.goal_pos - env.world.start_pos, dim=-1)
    gmin, gmax = env.cfg.world_cfg.goal_dist_ranges[0]
    assert goal_dist.min().item() >= gmin - 1e-4
    assert goal_dist.max().item() <= gmax + 1e-4

    last_err = torch.abs(goal_dist - env.last_goal_dist)
    assert last_err.max().item() < 1e-4, "reset 后 last_goal_dist 未正确同步"

    env.global_steps = old_steps

    print_stats("reset xy error", xy_err)
    print_stats("stage0 goal distance", goal_dist)
    print_ok("reset 对齐正常")


def test_stage_sampling(env: DiffDriveTask2Env):
    heading("[测试 5] 不同课程阶段 reset 采样检测")

    cfg = env.cfg
    old_steps = int(env.global_steps)
    rows = []
    stage_ks = [float(k) + (1e-6 if i > 0 else 0.0) for i, k in enumerate(cfg.world_cfg.stage_thresholds)]

    for expected_stage, k in enumerate(stage_ks):
        env.global_steps = int(k * cfg.world_cfg.curriculum_total_steps)
        obs, _ = env.reset()
        check_obs(env, obs)

        root_local = env._root_pos_local()
        xy_err = torch.norm(root_local - env.world.start_pos, dim=-1)
        assert xy_err.max().item() < 5e-4

        goal_dist = torch.norm(env.world.goal_pos - env.world.start_pos, dim=-1)
        static_count = env.world.static_mask.float().sum(dim=-1)
        dynamic_count = env.world.dynamic_mask.float().sum(dim=-1)

        gmin, gmax = cfg.world_cfg.goal_dist_ranges[expected_stage]
        smin, smax = cfg.world_cfg.static_count_ranges[expected_stage]
        dmin, dmax = cfg.world_cfg.dynamic_count_ranges[expected_stage]

        assert int(round(env.world.env_stage.float().mean().item())) == expected_stage
        assert goal_dist.min().item() >= gmin - 1e-4
        assert goal_dist.max().item() <= gmax + 1e-4
        assert static_count.min().item() >= smin
        assert static_count.max().item() <= smax
        assert dynamic_count.min().item() >= dmin
        assert dynamic_count.max().item() <= dmax

        rows.append(
            {
                "K": k,
                "Stage": float(expected_stage),
                "GoalDistMean": goal_dist.mean().item(),
                "StaticMean": static_count.mean().item(),
                "DynamicMean": dynamic_count.mean().item(),
                "TargetSpeedMean": env.world.env_target_speed.mean().item(),
                "XYErrMax": xy_err.max().item(),
            }
        )

    env.global_steps = old_steps

    print(f"{'K':>6} | {'Stage':>5} | {'GoalMean':>10} | {'Static':>8} | {'Dynamic':>8} | {'Speed':>8} | {'XYErr':>8}")
    print("-" * 78)
    for r in rows:
        print(
            f"{r['K']:>6.2f} | {int(r['Stage']):>5d} | {r['GoalDistMean']:>10.3f} | "
            f"{r['StaticMean']:>8.3f} | {r['DynamicMean']:>8.3f} | "
            f"{r['TargetSpeedMean']:>8.3f} | {r['XYErrMax']:>8.6f}"
        )

    print_ok("不同课程阶段 reset 采样正常")



def test_obs_slices(env: DiffDriveTask2Env):
    heading("[测试 6] core navigation / obstacle observation 切片与数值范围检测")

    set_env_to_stage(env, stage_idx=0)
    obs, _ = env.reset()
    check_obs(env, obs)

    single_dim = env.cfg.single_obs_dim
    last_frame = obs[:, -single_dim:]
    s = get_single_obs_slices(env.cfg)

    parts = {
        "core_nav": last_frame[:, s["core_nav"]],
        "goal_dist_norm": last_frame[:, s["goal_dist_norm"]],
        "goal_xy_body": last_frame[:, s["goal_xy_body"]],
        "heading": last_frame[:, s["heading"]],
        "vel_obs": last_frame[:, s["vel_obs"]],
        "target_speed": last_frame[:, s["target_speed"]],
        "last_forward_turn": last_frame[:, s["last_forward_turn"]],
        "action_delta_forward_turn": last_frame[:, s["action_delta_forward_turn"]],
        "progress_ema": last_frame[:, s["progress_ema"]],
        "lidar": last_frame[:, s["lidar"]],
        "lidar_delta": last_frame[:, s["lidar_delta"]],
        "risk": last_frame[:, s["risk"]],
    }

    for name, x in parts.items():
        assert_finite_tensor(name, x)

    check_shape("core navigation", parts["core_nav"], (env.num_envs, 14))
    check_shape("lidar", parts["lidar"], (env.num_envs, env.cfg.world_cfg.num_lidar_rays))
    check_shape("lidar_delta", parts["lidar_delta"], (env.num_envs, env.cfg.world_cfg.num_lidar_rays))
    check_shape("risk", parts["risk"], (env.num_envs, 8))

    assert parts["heading"].abs().max().item() <= 1.0 + 1e-5
    assert parts["last_forward_turn"].abs().max().item() <= 1.0 + 1e-5
    assert parts["action_delta_forward_turn"].abs().max().item() <= 2.0 + 1e-5
    assert parts["lidar"].min().item() >= 0.0
    assert parts["lidar"].max().item() <= 1.0 + 1e-5
    assert parts["lidar_delta"].min().item() >= -1.0 - 1e-5
    assert parts["lidar_delta"].max().item() <= 1.0 + 1e-5
    assert parts["risk"].min().item() >= 0.0
    assert parts["risk"].max().item() <= 1.0 + 1e-5

    print_ok(f"obs shape = {tuple(obs.shape)}")
    print_ok(f"last_frame range = {last_frame.min().item():+.6f} ~ {last_frame.max().item():+.6f}")
    print_ok(f"core navigation range = {parts['core_nav'].min().item():+.6f} ~ {parts['core_nav'].max().item():+.6f}")
    print_ok(f"lidar range = {parts['lidar'].min().item():.6f} ~ {parts['lidar'].max().item():.6f}")
    print_ok(f"risk range = {parts['risk'].min().item():.6f} ~ {parts['risk'].max().item():.6f}")


def test_step_return_structure(env: DiffDriveTask2Env):
    heading("[测试 7] 向量化 step 返回结构与 info 字典检测")

    set_env_to_stage(env, stage_idx=0)
    obs, _ = env.reset()

    action = torch.rand((env.num_envs, env.num_actions), device=env.device) * 0.4 - 0.2
    obs, reward, terminated, truncated, info = env.step(action)

    check_obs(env, obs)
    check_shape("reward", reward, (env.num_envs,))
    check_shape("terminated", terminated, (env.num_envs,))
    check_shape("truncated", truncated, (env.num_envs,))
    assert_finite_tensor("reward", reward)

    for group in ["reward_components", "events", "telemetry", "world", "debug"]:
        assert group in info, f"info 缺少分组: {group}"

    required_reward_keys = [
        "R_Goal_Progress_Velocity",
        "R_Heading_Improve",
        "R_Target_Speed",
        "R_Aligned_Motion",
        "P_Misaligned_Forward",
        "P_Safety_Proximity",
        "P_Action_Smooth",
        "Step",
        "Continuous",
        "Event",
        "Total",
    ]
    for key in required_reward_keys:
        assert key in info["reward_components"], f"reward_components 缺少 {key}"

    removed_reward_keys = [
        "P_Bad_Turn",
        "P_Lateral_Vel",
        "P_Spin_In_Place",
        "R_Progress",
        "R_Goal_Speed",
        "R_Goal_Speed_Forward",
        "R_Goal_Speed_Gaussian",
        "R_Heading",
        "R_Turn_To_Goal",
        "R_Front_Clearance",
        "R_Speed_Factor",
        "P_Under_Speed",
        "P_Low_Speed_Aligned",
        "P_Backtrack",
        "P_Collision_Risk",
        "P_TTC",
        "P_Boundary",
        "P_Spin",
        "P_Stuck",
        "P_Action_Mag",
        "P_Wheel_Speed",
    ]
    for key in removed_reward_keys:
        assert key not in info["reward_components"], f"reward_components 不应再输出旧字段 {key}"

    required_event_keys = [
        "Success_Rate",
        "Collision_Rate",
        "Static_Collision_Rate",
        "Dynamic_Collision_Rate",
        "Out_Of_Bounds_Rate",
        "Timeout_Rate",
        "Done_Rate",
        "Episode_Success_Rate",
        "Episode_Collision_Rate",
        "Episode_Out_Of_Bounds_Rate",
        "Episode_Timeout_Rate",
        "Episode_Done_Count",
        "Current_Window_Success_Rate",
        "Current_Window_Collision_Rate",
        "Current_Window_Out_Of_Bounds_Rate",
        "Current_Window_Timeout_Rate",
    ]
    for key in required_event_keys:
        assert key in info["events"], f"events 缺少 {key}"

    required_tel_keys = [
        "Curriculum_K",
        "Stage",
        "Target_Speed",
        "Goal_Dist",
        "Progress",
        "Progress_EMA",
        "Progress_Velocity",
        "Progress_Velocity_EMA",
        "Goal_Aligned_Speed",
        "Speed_Ratio",
        "Signed_Speed_Ratio",
        "Heading_Error",
        "Heading_Cos",
        "Heading_Improve",
        "Safety_Risk",
        "Safety_Signed_Distance",
        "Static_Signed_Distance",
        "Path_Corridor_Signed_Distance",
        "Dynamic_Predicted_Signed_Distance",
        "Lidar_Min",
        "Risk_Front",
        "Static_Count",
        "Dynamic_Count",
        "Action_Forward_Throttle",
        "Action_Turn",
        "Forward_Command_Norm",
        "Turn_Command_Norm",
        "Left_Wheel_Target_Norm",
        "Right_Wheel_Target_Norm",
        "Positive_Linear_Command_Rate",
        "Episode_Length",
    ]
    for key in required_tel_keys:
        assert key in info["telemetry"], f"telemetry 缺少 {key}"

    removed_tel_keys = ["Action_Left", "Action_Right", "Spin_In_Place_Ratio", "Bad_Turn_Ratio"]
    for key in removed_tel_keys:
        assert key not in info["telemetry"], f"telemetry 不应再输出旧/误导字段 {key}"

    print_ok(f"reward mean = {reward.mean().item():+.6f}")
    print_ok(f"reward min/max = {reward.min().item():+.6f} / {reward.max().item():+.6f}")
    print_ok(f"terminated count = {terminated.sum().item()}")
    print_ok(f"truncated count = {truncated.sum().item()}")
    print_ok("step 返回结构正常")


def test_action_direction(env: DiffDriveTask2Env):
    heading("[测试 8] Action protocol 动作方向白盒测试：前进 / 最小前进 / 差速转向")

    steps = 80
    set_env_to_stage(env, stage_idx=0)

    # Action protocol:
    #   action[0] = forward_throttle
    #   action[1] = turn_command
    # 因此 [1, 0] 是前进，[−1, 0] 是最小正向速度，不是倒车；
    # [−1, ±1] 用于低速/原地附近差速转向。
    delta_forward, yaw_forward, _, _, _, _, info_forward = run_fixed_action(env, torch.tensor([1.0, 0.0]), steps=steps)
    forward_x = delta_forward[:, 0]

    delta_min_forward, yaw_min_forward, _, _, _, _, info_min_forward = run_fixed_action(env, torch.tensor([-1.0, 0.0]), steps=steps)
    min_forward_x = delta_min_forward[:, 0]

    delta_turn_left, yaw_turn_left, _, _, _, _, info_turn_left = run_fixed_action(env, torch.tensor([-1.0, 1.0]), steps=steps)
    delta_turn_right, yaw_turn_right, _, _, _, _, info_turn_right = run_fixed_action(env, torch.tensor([-1.0, -1.0]), steps=steps)
    turn_xy = 0.5 * (torch.norm(delta_turn_left, dim=-1) + torch.norm(delta_turn_right, dim=-1))

    print_stats("forward [1,0] delta x", forward_x)
    print_stats("min-forward [-1,0] delta x", min_forward_x)
    print_stats("turn-left [-1,1] delta yaw", yaw_turn_left)
    print_stats("turn-right [-1,-1] delta yaw", yaw_turn_right)
    print_stats("turn xy movement", turn_xy)

    tel_fwd = info_forward.get("telemetry", {})
    tel_min = info_min_forward.get("telemetry", {})
    tel_left = info_turn_left.get("telemetry", {})
    tel_right = info_turn_right.get("telemetry", {})

    assert tel_fwd.get("Forward_Command_Norm", 0.0) > 0.50, "[1,0] 应产生明显非负 forward command"
    assert abs(tel_fwd.get("Turn_Command_Norm", 999.0)) < 0.05, "[1,0] 不应产生明显 turn command"
    assert tel_min.get("Forward_Command_Norm", -1.0) >= -1e-6, "[-1,0] 的 forward command 不应为负"
    assert tel_min.get("Positive_Linear_Command_Rate", 0.0) > 0.99, "线速度命令应保持非负"
    assert abs(tel_left.get("Turn_Command_Norm", 0.0)) > 0.10, "[-1,1] 应产生明显转向命令"
    assert abs(tel_right.get("Turn_Command_Norm", 0.0)) > 0.10, "[-1,-1] 应产生明显转向命令"
    assert tel_left.get("Turn_Command_Norm", 0.0) * tel_right.get("Turn_Command_Norm", 0.0) < 0.0, (
        "左右转测试的 turn command 应方向相反"
    )

    forward_ok = forward_x.mean().item() > 0.05
    min_forward_no_reverse = min_forward_x.mean().item() > -0.05
    turn_ok = yaw_turn_left.abs().mean().item() > 0.10 and yaw_turn_right.abs().mean().item() > 0.10
    turn_opposite = yaw_turn_left.mean().item() * yaw_turn_right.mean().item() < 0.0

    if args_cli.strict_action_test:
        assert forward_ok, "action=[1,0] 没有让车沿 +x 明显前进。请检查 wheel signs。"
        assert min_forward_no_reverse, "action=[-1,0] 出现明显整体倒车。请检查 ActionProtocol 映射。"
        assert turn_ok, "action=[-1,±1] 没有让车明显转向。请检查 wheel signs 或 joint 顺序。"
        assert turn_opposite, "左右转 yaw 方向没有相反。请检查 turn sign。"
    else:
        if not forward_ok:
            print_warn("action=[1,0] 没有沿 +x 明显前进。若训练变慢，请调整 left_wheel_sign/right_wheel_sign。")
        if not min_forward_no_reverse:
            print_warn("action=[-1,0] 出现明显整体倒车；这不符合 Task1/Task2 的 forward-only 语义。")
        if not turn_ok:
            print_warn("action=[-1,±1] 没有明显转向。若训练变慢，请检查 wheel signs 或 joint 顺序。")
        if not turn_opposite:
            print_warn("左右转 yaw 方向没有相反。若训练转向混乱，请检查 turn sign。")

    print_ok("Action protocol 动作方向测试完成")

def test_success_event(env: DiffDriveTask2Env):
    heading("[测试 9] 手动触发 success 事件检测")

    env.global_steps = 0
    env.reset()

    env_ids = torch.arange(min(64, env.num_envs), dtype=torch.long, device=env.device)

    target_xy = env.world.goal_pos[env_ids]
    delta = target_xy - env.world.start_pos[env_ids]
    yaw = torch.atan2(delta[:, 1], delta[:, 0])

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=target_xy,
        height=env.cfg.spawn_height,
        yaw=yaw,
    )

    reward, terminated, truncated, info = env._compute_rewards_and_dones(pre_goal_dist=env.last_goal_dist.clone())

    success_count = int(terminated[env_ids].sum().item())
    assert success_count == len(env_ids), "手动放到 goal 后没有全部触发 success terminated"
    assert truncated[env_ids].float().mean().item() == 0.0, "success 不应同时 truncated"

    print_ok(f"success triggered = {success_count}/{len(env_ids)}")
    print_ok(f"Success_Rate = {info['events']['Success_Rate']:.6f}")
    print_ok(f"Event reward mean = {info['reward_components']['Event']:.6f}")


def test_static_collision_event(env: DiffDriveTask2Env):
    heading("[测试 10] 手动触发 static collision 事件检测")

    cfg = env.cfg
    set_env_to_stage(env, stage_idx=2)
    env.reset()

    env_ids = torch.arange(min(64, env.num_envs), dtype=torch.long, device=env.device)
    assert env.world.static_mask[env_ids, 0].all().item(), "Stage2 第一个静态障碍未激活"

    obstacle_xy = env.world.static_pos[env_ids, 0, :]

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=obstacle_xy,
        height=env.cfg.spawn_height,
        yaw=torch.zeros(len(env_ids), device=env.device),
    )

    reward, terminated, truncated, info = env._compute_rewards_and_dones(pre_goal_dist=env.last_goal_dist.clone())

    collision_count = int(terminated[env_ids].sum().item())
    assert collision_count == len(env_ids), "手动放到静态障碍物上后没有全部触发 terminated"
    assert info["events"]["Static_Collision_Rate"] > 0.0
    assert info["events"]["Collision_Rate"] > 0.0

    print_ok(f"static collision triggered = {collision_count}/{len(env_ids)}")
    print_ok(f"Collision_Rate = {info['events']['Collision_Rate']:.6f}")
    print_ok(f"Static_Collision_Rate = {info['events']['Static_Collision_Rate']:.6f}")


def test_dynamic_collision_event(env: DiffDriveTask2Env):
    heading("[测试 11] 手动触发 dynamic collision 事件检测")

    cfg = env.cfg
    set_env_to_stage(env, stage_idx=4)
    env.reset()

    env_ids = torch.arange(min(64, env.num_envs), dtype=torch.long, device=env.device)
    assert env.world.dynamic_mask[env_ids, 0].all().item(), "Stage4 第一个动态障碍未激活"

    obstacle_xy = env.world.dynamic_pos[env_ids, 0, :]

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=obstacle_xy,
        height=env.cfg.spawn_height,
        yaw=torch.zeros(len(env_ids), device=env.device),
    )

    reward, terminated, truncated, info = env._compute_rewards_and_dones(pre_goal_dist=env.last_goal_dist.clone())

    collision_count = int(terminated[env_ids].sum().item())
    assert collision_count == len(env_ids), "手动放到动态障碍物上后没有全部触发 terminated"
    assert info["events"]["Dynamic_Collision_Rate"] > 0.0
    assert info["events"]["Collision_Rate"] > 0.0

    print_ok(f"dynamic collision triggered = {collision_count}/{len(env_ids)}")
    print_ok(f"Collision_Rate = {info['events']['Collision_Rate']:.6f}")
    print_ok(f"Dynamic_Collision_Rate = {info['events']['Dynamic_Collision_Rate']:.6f}")


def test_out_of_bounds_event(env: DiffDriveTask2Env):
    heading("[测试 12] 手动触发 out_of_bounds 事件检测")

    env.global_steps = 0
    env.reset()

    env_ids = torch.arange(min(64, env.num_envs), dtype=torch.long, device=env.device)

    local_xy = torch.zeros((len(env_ids), 2), dtype=torch.float32, device=env.device)
    local_xy[:, 0] = env.cfg.world_cfg.half_extent + 1.0

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=local_xy,
        height=env.cfg.spawn_height,
        yaw=torch.zeros(len(env_ids), device=env.device),
    )

    reward, terminated, truncated, info = env._compute_rewards_and_dones(pre_goal_dist=env.last_goal_dist.clone())

    oob_count = int(terminated[env_ids].sum().item())
    assert oob_count == len(env_ids), "手动放出边界后没有全部触发 terminated"
    assert info["events"]["Out_Of_Bounds_Rate"] > 0.0

    print_ok(f"out_of_bounds triggered = {oob_count}/{len(env_ids)}")
    print_ok(f"Out_Of_Bounds_Rate = {info['events']['Out_Of_Bounds_Rate']:.6f}")


def test_timeout_event(env: DiffDriveTask2Env):
    heading("[测试 13] timeout truncated 事件检测")

    env.global_steps = 0
    env.reset()
    env.episode_steps[:] = int(env.cfg.max_episode_length)

    reward, terminated, truncated, info = env._compute_rewards_and_dones(pre_goal_dist=env.last_goal_dist.clone())

    timeout_count = int(truncated.sum().item())
    assert timeout_count > 0, "episode_steps 到达最大长度后没有触发 truncated"

    print_ok(f"timeout truncated count = {timeout_count}")
    print_ok(f"Timeout_Rate = {info['events']['Timeout_Rate']:.6f}")
    print_ok(f"terminated mean = {terminated.float().mean().item():.6f}")
    print_ok(f"truncated mean = {truncated.float().mean().item():.6f}")


def test_dynamic_obstacle_integration(env: DiffDriveTask2Env):
    heading("[测试 14] 动态障碍物与 env.step 集成检测")

    # 当前 default 配置下，Stage 4 才开始生成动态障碍物。
    # 不再写死 0.38，直接从 cfg.world_cfg.stage_thresholds[3] 进入 Stage 4。
    set_env_to_stage(env, stage_idx=4)
    env.reset()

    stage_mean = int(round(env.world.env_stage.float().mean().item()))
    assert stage_mean == 4, f"未进入 Stage4，当前 stage={stage_mean}"

    dyn_count = env.world.dynamic_mask.float().sum(dim=-1)
    assert dyn_count.mean().item() > 0.0, "Stage4 应生成动态障碍物"

    pos0 = env.world.dynamic_pos.clone()
    zero_action = torch.zeros((env.num_envs, env.num_actions), device=env.device)

    info = {}
    obs = None
    reward = None
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(zero_action)

    pos1 = env.world.dynamic_pos.clone()
    moved = torch.norm(pos1 - pos0, dim=-1)
    moved_valid = moved[env.world.dynamic_mask]

    assert moved_valid.numel() > 0, "Stage4 没有有效动态障碍物样本"
    assert moved_valid.mean().item() > 0.01, "env.step 后动态障碍物没有移动"

    check_obs(env, obs)
    assert_finite_tensor("reward", reward)

    print_stats("dynamic obstacle movement", moved_valid)
    print_ok(f"Stage = {stage_mean}")
    print_ok(f"Dynamic_Count = {info['telemetry']['Dynamic_Count']:.6f}")
    print_ok("动态障碍物 env.step 集成正常")



def test_reward_direction(env: DiffDriveTask2Env):
    heading("[测试 15] simplified reward 方向与防刷分检测")

    set_env_to_stage(env, stage_idx=0)
    env.reset()

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    disable_all_obstacles(env)
    env.world.start_pos[:] = 0.0
    env.world.goal_pos[:] = torch.tensor([5.0, 0.0], dtype=torch.float32, device=env.device)
    env.world.env_target_speed[:] = 0.5

    # 1) 对准目标但不移动：目标速度、期望速度和朝向改善都不应长期刷分。
    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=torch.zeros((env.num_envs, 2), device=env.device),
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, device=env.device),
        zero_vel=True,
    )
    env.last_goal_dist[:] = torch.norm(env.world.goal_pos - env._root_pos_local(), dim=-1)
    env.last_heading_error_abs[:] = 0.0
    reward_static, _, _, info_static = env._compute_rewards_and_dones(
        pre_goal_dist=torch.full((env.num_envs,), 5.0, device=env.device)
    )

    assert abs(info_static["telemetry"]["Progress"]) < 1e-6, "静止时 Progress 应接近 0"
    assert abs(info_static["reward_components"]["R_Goal_Progress_Velocity"]) < 1e-5, (
        "静止时 R_Goal_Progress_Velocity 不应给正收益"
    )
    assert info_static["reward_components"]["R_Target_Speed"] < 1e-5, (
        "静止时 R_Target_Speed 不应给正收益"
    )
    assert abs(info_static["reward_components"]["R_Heading_Improve"]) < 1e-5, (
        "静止且朝向未改善时 R_Heading_Improve 不应刷分"
    )
    assert abs(info_static["reward_components"]["R_Aligned_Motion"]) < 1e-5, (
        "静止时 R_Aligned_Motion 不应给正收益"
    )

    # 2) 朝 goal 前进，Progress 与主任务奖励应为正，且优于远离 goal。
    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=torch.tensor([0.30, 0.0], device=env.device),
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, device=env.device),
    )
    env.last_heading_error_abs[:] = 0.0
    reward_fwd, _, _, info_fwd = env._compute_rewards_and_dones(
        pre_goal_dist=torch.full((env.num_envs,), 5.0, device=env.device)
    )

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=torch.tensor([-0.30, 0.0], device=env.device),
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, device=env.device),
    )
    env.last_heading_error_abs[:] = 0.0
    reward_back, _, _, info_back = env._compute_rewards_and_dones(
        pre_goal_dist=torch.full((env.num_envs,), 5.0, device=env.device)
    )

    assert info_fwd["telemetry"]["Progress"] > 0.0, "朝 goal 前进时 Progress 应为正"
    assert info_back["telemetry"]["Progress"] < 0.0, "远离 goal 时 Progress 应为负"
    assert info_fwd["reward_components"]["R_Goal_Progress_Velocity"] > 0.0, (
        "朝 goal 前进时 R_Goal_Progress_Velocity 应为正"
    )
    assert info_back["reward_components"]["R_Goal_Progress_Velocity"] < 0.0, (
        "远离 goal 时 R_Goal_Progress_Velocity 应为负"
    )
    assert reward_fwd.mean().item() > reward_back.mean().item(), "朝 goal 前进的奖励应大于远离 goal"

    print_ok(
        f"static progress = {info_static['telemetry']['Progress']:+.6f}, "
        f"R_GPV = {info_static['reward_components']['R_Goal_Progress_Velocity']:+.6f}, "
        f"R_Target = {info_static['reward_components']['R_Target_Speed']:+.6f}, "
        f"R_HeadImprove = {info_static['reward_components']['R_Heading_Improve']:+.6f}, "
        f"R_Aligned = {info_static['reward_components']['R_Aligned_Motion']:+.6f}, "
        f"reward = {reward_static.mean().item():+.6f}"
    )
    print_ok(f"forward progress = {info_fwd['telemetry']['Progress']:+.6f}, reward = {reward_fwd.mean().item():+.6f}")
    print_ok(f"backward progress = {info_back['telemetry']['Progress']:+.6f}, reward = {reward_back.mean().item():+.6f}")
    print_ok("simplified reward 方向与防刷分检测通过")

def random_rollout(env: DiffDriveTask2Env):
    heading(f"[测试 16] 随机策略运行 {args_cli.steps} 步，收集奖励组件 / 事件 / 遥测")

    # rollout_k 仍然允许命令行指定，但这里增加边界保护和 stage 打印。
    rollout_k = float(args_cli.rollout_k)
    rollout_k = min(max(rollout_k, 0.0), 1.0)
    env.global_steps = int(rollout_k * env.cfg.world_cfg.curriculum_total_steps)

    obs, _ = env.reset()

    records = []
    total_success = 0
    total_collision = 0
    total_oob = 0
    total_timeout = 0

    start_time = time.time()

    rollout_stage = int(round(env.world.env_stage.float().mean().item()))
    print_ok(f"rollout_k = {rollout_k:.6f}, rollout stage = {rollout_stage}")

    for step in range(int(args_cli.steps)):
        actions = torch.rand((env.num_envs, env.num_actions), device=env.device) * 2.0 - 1.0
        obs, reward, terminated, truncated, info = env.step(actions)

        flat = flatten_info(info)
        flat["Reward_Mean_Step"] = reward.mean().item()
        flat["Reward_Min_Step"] = reward.min().item()
        flat["Reward_Max_Step"] = reward.max().item()
        flat["Terminated_Count"] = terminated.sum().item()
        flat["Truncated_Count"] = truncated.sum().item()
        records.append(flat)

        total_success += int(info["events"]["Success_Rate"] * env.num_envs)
        total_collision += int(info["events"]["Collision_Rate"] * env.num_envs)
        total_oob += int(info["events"]["Out_Of_Bounds_Rate"] * env.num_envs)
        total_timeout += int(info["events"]["Timeout_Rate"] * env.num_envs)

        if (step + 1) % max(int(args_cli.collect_interval), 1) == 0 or (step + 1) == int(args_cli.steps):
            tel = info.get("telemetry", {})
            ev = info.get("events", {})
            rew = info.get("reward_components", {})

            print(
                f" -> Step {step + 1:05d} | "
                f"Reward={reward.mean().item():+.4f} | "
                f"Stage={tel.get('Stage', 0.0):.2f} | "
                f"GoalDist={tel.get('Goal_Dist', 0.0):.2f} | "
                f"Progress={tel.get('Progress', 0.0):+.4f} | "
                f"GoalV={tel.get('Goal_Aligned_Speed', 0.0):+.3f} | "
                f"HeadErr={tel.get('Heading_Error', 0.0):.3f} | "
                f"LidarMin={tel.get('Lidar_Min', 0.0):.3f} | "
                f"RiskF={tel.get('Risk_Front', 0.0):.3f} | "
                f"Static={tel.get('Static_Count', 0.0):.1f} | "
                f"Dyn={tel.get('Dynamic_Count', 0.0):.1f} | "
                f"Succ={ev.get('Success_Rate', 0.0):.4f} | "
                f"Coll={ev.get('Collision_Rate', 0.0):.4f} | "
                f"OOB={ev.get('Out_Of_Bounds_Rate', 0.0):.4f} | "
                f"Timeout={ev.get('Timeout_Rate', 0.0):.4f} | "
                f"R_GPV={rew.get('R_Goal_Progress_Velocity', 0.0):+.3f} | "
                f"R_Target={rew.get('R_Target_Speed', 0.0):+.3f} | "
                f"P_Safety={rew.get('P_Safety_Proximity', 0.0):+.3f}",
                flush=True,
            )

        if (step + 1) % 500 == 0 or (step + 1) == int(args_cli.steps):
            check_obs(env, obs)
            assert_finite_tensor("reward rollout", reward)
            assert torch.isfinite(env.actions).all().item(), "actions 出现 NaN/Inf"
            assert torch.isfinite(env.prev_actions).all().item(), "prev_actions 出现 NaN/Inf"
            assert torch.isfinite(env.last_goal_dist).all().item(), "last_goal_dist 出现 NaN/Inf"
            assert torch.isfinite(env.progress_ema).all().item(), "progress_ema 出现 NaN/Inf"
            assert torch.isfinite(env.world.start_pos).all().item(), "world.start_pos 出现 NaN/Inf"
            assert torch.isfinite(env.world.goal_pos).all().item(), "world.goal_pos 出现 NaN/Inf"
            assert torch.isfinite(env.world.static_pos).all().item(), "world.static_pos 出现 NaN/Inf"
            assert torch.isfinite(env.world.dynamic_pos).all().item(), "world.dynamic_pos 出现 NaN/Inf"

    elapsed = time.time() - start_time
    fps = int(args_cli.steps) * env.num_envs / max(elapsed, 1e-6)

    print_ok(f"随机策略长跑完成: {args_cli.steps} steps, {int(args_cli.steps) * env.num_envs:,} transitions")
    print_ok(f"吞吐约: {fps:,.2f} env steps/s")
    print_ok(f"累计 success approx: {total_success:,}")
    print_ok(f"累计 collision approx: {total_collision:,}")
    print_ok(f"累计 out_of_bounds approx: {total_oob:,}")
    print_ok(f"累计 timeout approx: {total_timeout:,}")

    heading("[测试 17] 奖励组件 / 事件 / 遥测统计报告")
    print_summary_table(summarize_records(records))

    print("Diff-Drive UGV Task2 training pre-check guide:")
    print("1. obs 应为 498 维，即 3 帧 × 166 维。")
    print("2. Stage / Static_Count / Dynamic_Count 应随 rollout_k 对应课程变化。")
    print("3. Lidar_Min / Risk_Front / Front_Clearance 应能正常进入 telemetry。")
    print("4. success / collision / oob / timeout 必须能被手动触发。")
    print("5. 随机策略下 collision 较高是正常的，但不能出现 NaN/Inf。")
    print("6. 训练时重点看 Progress、Goal_Aligned_Speed、Goal_Dist、Risk_Front、Success_Rate、Collision_Rate。")


def run_tests() -> None:
    heading("🚀 Diff-Drive UGV Task2 Analytic Obstacle Navigation Env 全量测试启动")

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    if args_cli.test_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print_warn("CUDA 不可用，自动切换到 CPU")
    else:
        device = args_cli.test_device

    if bool(args_cli.quick):
        args_cli.num_envs = min(int(args_cli.num_envs), 64)
        args_cli.steps = min(int(args_cli.steps), 500)
        args_cli.collect_interval = min(int(args_cli.collect_interval), 100)

    check_project_files()
    check_config()

    cfg = Task2Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.seed = int(args_cli.seed)
    cfg.print_debug_info = bool(args_cli.print_names)
    cfg.validate()

    env: DiffDriveTask2Env | None = None

    try:
        env = check_env_init(cfg)
        test_reset_step_basic(env)
        test_reset_alignment(env)
        test_stage_sampling(env)
        test_obs_slices(env)
        test_step_return_structure(env)
        test_action_direction(env)
        test_success_event(env)
        test_static_collision_event(env)
        test_dynamic_collision_event(env)
        test_out_of_bounds_event(env)
        test_timeout_event(env)
        test_dynamic_obstacle_integration(env)
        test_reward_direction(env)
        random_rollout(env)

        heading("Diff-Drive UGV Task2 环境测试全部通过")

    except Exception as exc:
        print("\n❌ Diff-Drive UGV Task2 环境测试失败：")
        print(type(exc).__name__, ":", exc)
        raise

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass



if __name__ == "__main__":
    try:
        run_tests()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
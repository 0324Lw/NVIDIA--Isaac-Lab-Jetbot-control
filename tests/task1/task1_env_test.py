from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diff-Drive UGV / Jetbot Task1 Multi-waypoint Navigation Env Test")
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--collect-interval", type=int, default=500)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-names", action="store_true")
parser.add_argument("--strict-action-test", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

simulation_app = AppLauncher(args_cli).app

from diff_drive_rl.tasks.task1.task1_config import Task1Config
from diff_drive_rl.tasks.task1.task1_env import DiffDriveTask1Env


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
        f"{name:<32s} "
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

    keys = sorted({key for row in records for key in row.keys()})
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

    print("\n" + "=" * 184)
    print(" " * 54 + "Diff-Drive UGV / Jetbot Task1 环境统计报告")
    print("=" * 184)
    print(
        f"{'metric':<78} | {'mean':>11} | {'var':>11} | {'min':>11} | "
        f"{'p25':>11} | {'p50':>11} | {'p75':>11} | {'max':>11}"
    )
    print("-" * 184)

    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<78} | "
            f"{row['mean']:>11.5f} | "
            f"{row['var']:>11.5f} | "
            f"{row['min']:>11.5f} | "
            f"{row['p25']:>11.5f} | "
            f"{row['p50']:>11.5f} | "
            f"{row['p75']:>11.5f} | "
            f"{row['max']:>11.5f}"
        )

    print("=" * 184 + "\n")


def yaw_to_quat_wxyz(yaw: torch.Tensor):
    quat = torch.zeros((yaw.shape[0], 4), dtype=torch.float32, device=yaw.device)
    quat[:, 0] = torch.cos(yaw * 0.5)
    quat[:, 3] = torch.sin(yaw * 0.5)
    return quat


def force_root_local_pose(
    env: DiffDriveTask1Env,
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


def check_obs(env: DiffDriveTask1Env, obs: torch.Tensor) -> None:
    check_shape("obs", obs, (env.num_envs, env.num_observations))
    assert_finite_tensor("obs", obs)
    assert obs.abs().max().item() <= float(env.cfg.obs_clip) + 1e-5, "obs 超出 clamp 范围"


def get_single_obs_slices():
    slices = {}

    idx = 0
    slices["dist_norm"] = slice(idx, idx + 1); idx += 1
    slices["sin_heading"] = slice(idx, idx + 1); idx += 1
    slices["cos_heading"] = slice(idx, idx + 1); idx += 1
    slices["body_vx"] = slice(idx, idx + 1); idx += 1
    slices["body_vy"] = slice(idx, idx + 1); idx += 1
    slices["body_wz"] = slice(idx, idx + 1); idx += 1
    slices["wheel_vel"] = slice(idx, idx + 2); idx += 2
    slices["last_action"] = slice(idx, idx + 2); idx += 2
    slices["progress_ema"] = slice(idx, idx + 1); idx += 1
    slices["waypoint_index"] = slice(idx, idx + 1); idx += 1

    assert idx == 12, f"单帧 obs slice 总维度错误: {idx} != 12"
    return slices


def run_fixed_action(env: DiffDriveTask1Env, action_tensor: torch.Tensor, steps: int = 60):
    obs, _ = env.reset()

    pos0 = env._root_pos_local()[:, :2].clone()
    yaw0 = env._quat_yaw(env.robot.data.root_quat_w).clone()

    action = action_tensor.to(env.device).view(1, 2).repeat(env.num_envs, 1)

    info = {}
    reward = None
    terminated = None
    truncated = None

    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(action)

    pos1 = env._root_pos_local()[:, :2].clone()
    yaw1 = env._quat_yaw(env.robot.data.root_quat_w).clone()

    delta_pos = pos1 - pos0
    delta_yaw = torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0))

    return delta_pos, delta_yaw, obs, reward, terminated, truncated, info


def check_project_files() -> None:
    heading("[测试 0] Diff-Drive UGV / Jetbot Task1 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task1_multi_waypoint_navigation.yaml",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task1" / "task1_config.py",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task1" / "task1_scene.py",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task1" / "task1_env.py",
    ]

    missing = [str(path) for path in required if not path.exists()]
    assert not missing, "缺少 Task1 必要文件:\n" + "\n".join(missing)

    for path in required:
        print_ok(str(path.relative_to(PROJECT_ROOT)))

    print_ok("Task1 工程文件结构正常")


def check_config() -> None:
    heading("[测试 1] Task1Config 基础配置检测")

    cfg = Task1Config()
    cfg.validate()

    assert cfg.num_actions == 2
    assert cfg.single_obs_dim == 12
    assert cfg.frame_stack == 3
    assert cfg.num_observations == 36
    assert cfg.num_waypoints >= 1

    print_ok(f"num_actions = {cfg.num_actions}")
    print_ok(f"single_obs_dim = {cfg.single_obs_dim}")
    print_ok(f"frame_stack = {cfg.frame_stack}")
    print_ok(f"num_observations = {cfg.num_observations}")
    print_ok(f"num_waypoints = {cfg.num_waypoints}")
    print_ok(f"max_episode_length = {cfg.max_episode_length}")
    print_ok("Task1Config 基础配置正常")


def check_env_init(cfg: Task1Config) -> DiffDriveTask1Env:
    heading("[测试 2] JetbotNavigationEnv 初始化 / 名称映射 / 空间维度检测")

    env = DiffDriveTask1Env(cfg)

    print_ok(f"device = {cfg.device}")
    print_ok(f"num_envs = {cfg.num_envs}")
    print_ok(f"robot.num_joints = {env.robot.num_joints}")
    print_ok(f"num_actions = {env.num_actions}")
    print_ok(f"single_obs_dim = {env.cfg.single_obs_dim}")
    print_ok(f"frame_stack = {env.cfg.frame_stack}")
    print_ok(f"num_observations = {env.num_observations}")
    print_ok(f"wheel_joint_ids = {env.wheel_joint_ids}")

    assert env.robot.num_joints >= 2, f"Jetbot 关节数量异常: {env.robot.num_joints}"
    assert env.num_actions == 2
    assert env.cfg.single_obs_dim == 12
    assert env.num_observations == env.cfg.frame_stack * env.cfg.single_obs_dim
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


def test_reset_obs_step(env: DiffDriveTask1Env):
    heading("[测试 3] reset / obs / step 返回结构与 info 字典检测")

    obs, info = env.reset()
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

    for group in ["reward_components", "events", "telemetry", "debug"]:
        assert group in info, f"info 缺少分组: {group}"

    required_reward_keys = [
        "R_Progress",
        "R_Heading",
        "R_Forward",
        "P_Spin",
        "P_Lateral_Vel",
        "P_Action_Smooth",
        "P_Action_Mag",
        "P_Wheel_Speed",
        "P_Stuck",
        "Step",
        "Continuous",
        "Event",
        "Total",
    ]
    for key in required_reward_keys:
        assert key in info["reward_components"], f"reward_components 缺少 {key}"

    required_event_keys = [
        "Waypoint_Rate",
        "Finish_Rate",
        "Timeout_Rate",
        "Done_Rate",
        "Episode_Finish_Rate",
        "Episode_Timeout_Rate",
        "Episode_Done_Count",
    ]
    for key in required_event_keys:
        assert key in info["events"], f"events 缺少 {key}"

    required_tel_keys = [
        "Distance_To_Waypoint",
        "Progress",
        "Progress_EMA",
        "Heading_Error",
        "Goal_Aligned_Speed",
        "Body_Vx",
        "Body_Wz",
        "Waypoint_Index",
        "Stuck_Ratio",
    ]
    for key in required_tel_keys:
        assert key in info["telemetry"], f"telemetry 缺少 {key}"

    print_ok(f"reward mean = {reward.mean().item():+.6f}")
    print_ok(f"reward min/max = {reward.min().item():+.6f} / {reward.max().item():+.6f}")
    print_ok(f"terminated count = {terminated.sum().item()}")
    print_ok(f"truncated count = {truncated.sum().item()}")
    print_ok("reset / step 返回结构正常")


def test_reset_alignment_and_waypoints(env: DiffDriveTask1Env):
    heading("[测试 4] reset 后 root local 对齐 / waypoint 采样检测")

    obs, _ = env.reset()
    check_obs(env, obs)

    root_local = env._root_pos_local()
    xy_error = torch.norm(root_local[:, :2], dim=-1)

    assert xy_error.max().item() < 5e-4, f"reset 后 root local xy 不接近 0，max={xy_error.max().item()}"

    wp = env.waypoints_local
    wp_norm = torch.norm(wp, dim=-1)

    assert_finite_tensor("waypoints_local", wp)
    assert wp_norm.min().item() >= 0.0
    assert wp_norm.max().item() <= float(env.cfg.waypoint_world_radius) + 1e-4

    first_dist = torch.norm(env.waypoints_local[:, 0, :], dim=-1)
    assert first_dist.min().item() >= float(env.cfg.waypoint_min_radius) - 1e-4
    assert first_dist.max().item() <= float(env.cfg.waypoint_max_radius) + 1e-4

    dist_now = env._distance_to_current_waypoint()
    last_err = torch.abs(dist_now - env.last_distances)
    assert last_err.max().item() < 1e-5, "reset 后 last_distances 未同步"

    print_stats("root local xy error", xy_error)
    print_stats("all waypoint radius", wp_norm)
    print_stats("first waypoint dist", first_dist)
    print_ok("reset 对齐与 waypoint 采样正常")


def test_obs_slices(env: DiffDriveTask1Env):
    heading("[测试 5] observation 切片与数值范围检测")

    obs, _ = env.reset()
    check_obs(env, obs)

    single_dim = env.cfg.single_obs_dim
    last_frame = obs[:, -single_dim:]
    s = get_single_obs_slices()

    dist_norm = last_frame[:, s["dist_norm"]]
    sin_heading = last_frame[:, s["sin_heading"]]
    cos_heading = last_frame[:, s["cos_heading"]]
    body_vx = last_frame[:, s["body_vx"]]
    body_vy = last_frame[:, s["body_vy"]]
    body_wz = last_frame[:, s["body_wz"]]
    wheel_vel = last_frame[:, s["wheel_vel"]]
    last_action = last_frame[:, s["last_action"]]
    progress_ema = last_frame[:, s["progress_ema"]]
    waypoint_index = last_frame[:, s["waypoint_index"]]

    for name, x in [
        ("dist_norm", dist_norm),
        ("sin_heading", sin_heading),
        ("cos_heading", cos_heading),
        ("body_vx", body_vx),
        ("body_vy", body_vy),
        ("body_wz", body_wz),
        ("wheel_vel", wheel_vel),
        ("last_action", last_action),
        ("progress_ema", progress_ema),
        ("waypoint_index", waypoint_index),
    ]:
        assert_finite_tensor(name, x)

    assert sin_heading.abs().max().item() <= 1.0 + 1e-5
    assert cos_heading.abs().max().item() <= 1.0 + 1e-5
    assert last_action.abs().max().item() <= 1.0 + 1e-5
    assert waypoint_index.min().item() >= 0.0
    assert waypoint_index.max().item() <= 1.0 + 1e-5

    print_ok(f"obs shape = {tuple(obs.shape)}")
    print_ok(f"last_frame range = {last_frame.min().item():+.6f} ~ {last_frame.max().item():+.6f}")
    print_ok(f"dist_norm mean = {dist_norm.mean().item():.6f}")
    print_ok(f"waypoint_index mean = {waypoint_index.mean().item():.6f}")


def test_compute_obs_purity(env: DiffDriveTask1Env):
    heading("[测试 6] compute_obs / _compute_single_obs 不污染 last_distances 检测")

    env.reset()
    before = env.last_distances.clone()

    for _ in range(10):
        obs = env.compute_obs()
        single = env._compute_single_obs()
        assert_finite_tensor("compute_obs", obs)
        assert_finite_tensor("_compute_single_obs", single)

    after = env.last_distances.clone()
    err = torch.abs(after - before)

    assert err.max().item() == 0.0, "compute_obs 或 _compute_single_obs 修改了 last_distances"
    print_ok("观测函数不污染 reward distance buffer")


def test_action_direction(env: DiffDriveTask1Env):
    heading("[测试 7] 动作方向白盒测试：前进 / 后退 / 原地旋转")

    steps = 80

    delta_forward, yaw_forward, obs, reward, terminated, truncated, info = run_fixed_action(
        env,
        torch.tensor([1.0, 1.0]),
        steps=steps,
    )
    forward_x = delta_forward[:, 0]

    delta_backward, yaw_backward, *_ = run_fixed_action(
        env,
        torch.tensor([-1.0, -1.0]),
        steps=steps,
    )
    backward_x = delta_backward[:, 0]

    delta_turn, yaw_turn, *_ = run_fixed_action(
        env,
        torch.tensor([1.0, -1.0]),
        steps=steps,
    )
    turn_xy = torch.norm(delta_turn, dim=-1)

    print_stats("forward delta x", forward_x)
    print_stats("backward delta x", backward_x)
    print_stats("turn delta yaw", yaw_turn)
    print_stats("turn xy movement", turn_xy)

    forward_ok = forward_x.mean().item() > 0.05
    backward_ok = backward_x.mean().item() < -0.05
    turn_ok = yaw_turn.abs().mean().item() > 0.10

    if args_cli.strict_action_test:
        assert forward_ok, "action=[1,1] 没有让车明显前进。请检查 wheel signs。"
        assert backward_ok, "action=[-1,-1] 没有让车明显后退。请检查 wheel signs。"
        assert turn_ok, "action=[1,-1] 没有让车明显原地转向。请检查 wheel sign 或 joint 顺序。"
    else:
        if not forward_ok:
            print_warn("action=[1,1] 没有明显前进。若训练变慢，请尝试修改 left_wheel_sign/right_wheel_sign。")
        if not backward_ok:
            print_warn("action=[-1,-1] 没有明显后退。若训练变慢，请尝试修改 wheel signs。")
        if not turn_ok:
            print_warn("action=[1,-1] 没有明显转向。若训练变慢，请检查 wheel joint 顺序或 wheel signs。")

    print_ok("动作方向测试完成")


def test_progress_reward_sign(env: DiffDriveTask1Env):
    heading("[测试 8] progress reward 正负方向检测")

    env.reset()

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    env.waypoints_local[:, :, :] = 0.0
    env.waypoints_local[:, 0, 0] = 1.5
    env.current_wp_idx[:] = 0

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=torch.zeros((env.num_envs, 2), device=env.device),
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, device=env.device),
    )
    env.last_distances[:] = env._distance_to_current_waypoint()

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=torch.tensor([0.20, 0.0], device=env.device),
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, device=env.device),
    )

    current_dist = env._distance_to_current_waypoint()
    progress_forward = env.last_distances - current_dist
    assert progress_forward.mean().item() > 0.0, "朝 waypoint 前进时 progress 应为正"

    env.last_distances[:] = current_dist

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=torch.tensor([-0.10, 0.0], device=env.device),
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, device=env.device),
    )

    current_dist2 = env._distance_to_current_waypoint()
    progress_backward = env.last_distances - current_dist2
    assert progress_backward.mean().item() < 0.0, "远离 waypoint 时 progress 应为负"

    print_stats("progress forward", progress_forward)
    print_stats("progress backward", progress_backward)
    print_ok("progress reward 方向正确")


def test_waypoint_event(env: DiffDriveTask1Env):
    heading("[测试 9] 手动触发普通 waypoint 事件检测")

    env.reset()
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    target = env._current_waypoint_local()
    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=target,
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, device=env.device),
    )

    zero_action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, reward, terminated, truncated, info = env.step(zero_action)

    waypoint_rate = info["events"]["Waypoint_Rate"]
    finish_rate = info["events"]["Finish_Rate"]

    assert waypoint_rate > 0.99, "手动放到第一个 waypoint 后没有 100% 触发普通 waypoint"
    assert finish_rate < 1e-6, "第一个 waypoint 不应触发 finish"
    assert env.current_wp_idx.float().mean().item() >= 1.0, "current_wp_idx 没有递增"

    check_obs(env, obs)
    assert_finite_tensor("reward", reward)

    print_ok(f"Waypoint_Rate = {waypoint_rate:.6f}")
    print_ok(f"Finish_Rate = {finish_rate:.6f}")
    print_ok(f"Event reward mean = {info['reward_components']['Event']:.6f}")
    print_ok("普通 waypoint 事件正常")


def test_finish_event(env: DiffDriveTask1Env):
    heading("[测试 10] 手动触发 finish 事件检测")

    env.reset()
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    env.current_wp_idx[:] = int(env.cfg.num_waypoints) - 1
    target = env._current_waypoint_local()

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=target,
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, device=env.device),
    )

    zero_action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, reward, terminated, truncated, info = env.step(zero_action)

    finish_rate = info["events"]["Finish_Rate"]
    assert finish_rate > 0.99, "手动放到最后一个 waypoint 后没有 100% finish"
    assert terminated.float().mean().item() > 0.99, "finish 后 terminated 没有 100% 触发"

    check_obs(env, obs)
    assert_finite_tensor("reward", reward)

    print_ok(f"Finish_Rate = {finish_rate:.6f}")
    print_ok(f"terminated mean = {terminated.float().mean().item():.6f}")
    print_ok(f"Event reward mean = {info['reward_components']['Event']:.6f}")
    print_ok("finish 事件正常")


def test_timeout_event(env: DiffDriveTask1Env):
    heading("[测试 11] timeout truncated 事件检测")

    env.reset()
    env.step_counts[:] = int(env.cfg.max_episode_length)

    zero_action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, reward, terminated, truncated, info = env.step(zero_action)

    timeout_rate = info["events"]["Timeout_Rate"]
    assert timeout_rate > 0.99, "step_counts 到达 max_episode_length 后没有 100% timeout"
    assert truncated.float().mean().item() > 0.99, "timeout 后 truncated 没有 100% 触发"

    check_obs(env, obs)
    assert_finite_tensor("reward", reward)

    print_ok(f"Timeout_Rate = {timeout_rate:.6f}")
    print_ok(f"truncated mean = {truncated.float().mean().item():.6f}")


def test_stuck_penalty(env: DiffDriveTask1Env):
    heading("[测试 12] stuck penalty 检测")

    env.reset()

    env.step_counts[:] = int(env.cfg.stuck_after_steps) + 5
    env.last_distances[:] = env._distance_to_current_waypoint()
    env.progress_ema[:] = 0.0
    env.stuck_counter[:] = 0
    env.actions[:] = 0.0
    env.prev_actions[:] = 0.0

    pre = env.last_distances.clone()
    current = env._distance_to_current_waypoint()
    progress = pre - current

    reward, terminated, truncated, info = env._compute_rewards_and_dones(
        pre_distances=pre,
        current_dist=current,
        progress=progress,
    )

    assert info["telemetry"]["Stuck_Ratio"] > 0.5, "静止且未到目标时 stuck ratio 应较高"
    assert info["reward_components"]["P_Stuck"] <= 0.0, "P_Stuck 应为非正"
    assert_finite_tensor("reward stuck", reward)

    print_ok(f"Stuck_Ratio = {info['telemetry']['Stuck_Ratio']:.6f}")
    print_ok(f"P_Stuck = {info['reward_components']['P_Stuck']:.6f}")


def random_rollout(env: DiffDriveTask1Env):
    heading(f"[测试 13] 随机策略运行 {args_cli.steps} 步，收集奖励组件 / 事件 / 遥测")

    obs, _ = env.reset()

    records: List[Dict[str, float]] = []
    total_finished = 0
    total_timeout = 0
    total_waypoints = 0

    start_time = time.time()

    for step in range(int(args_cli.steps)):
        actions = torch.rand((env.num_envs, env.num_actions), device=env.device) * 2.0 - 1.0
        obs, reward, terminated, truncated, info = env.step(actions)

        flat = flatten_info(info)
        flat["test/Reward_Mean_Step"] = reward.mean().item()
        flat["test/Reward_Min_Step"] = reward.min().item()
        flat["test/Reward_Max_Step"] = reward.max().item()
        flat["test/Terminated_Count"] = terminated.sum().item()
        flat["test/Truncated_Count"] = truncated.sum().item()
        records.append(flat)

        total_finished += int(info["events"]["Finish_Rate"] * env.num_envs)
        total_timeout += int(info["events"]["Timeout_Rate"] * env.num_envs)
        total_waypoints += int(info["events"]["Waypoint_Rate"] * env.num_envs)

        if (step + 1) % max(int(args_cli.collect_interval), 1) == 0 or (step + 1) == int(args_cli.steps):
            tel = info.get("telemetry", {})
            ev = info.get("events", {})
            rew = info.get("reward_components", {})

            print(
                f" -> Step {step + 1:05d} | "
                f"Reward={reward.mean().item():+.4f} | "
                f"Dist={tel.get('Distance_To_Waypoint', 0.0):.3f} | "
                f"Prog={tel.get('Progress', 0.0):+.4f} | "
                f"HeadErr={tel.get('Heading_Error', 0.0):.3f} | "
                f"GoalV={tel.get('Goal_Aligned_Speed', 0.0):+.3f} | "
                f"Vx={tel.get('Body_Vx', 0.0):+.3f} | "
                f"Wz={tel.get('Body_Wz', 0.0):+.3f} | "
                f"WPIdx={tel.get('Waypoint_Index', 0.0):.2f} | "
                f"Stuck={tel.get('Stuck_Ratio', 0.0):.3f} | "
                f"WP={ev.get('Waypoint_Rate', 0.0):.4f} | "
                f"Finish={ev.get('Finish_Rate', 0.0):.4f} | "
                f"Timeout={ev.get('Timeout_Rate', 0.0):.4f} | "
                f"R_Prog={rew.get('R_Progress', 0.0):+.3f} | "
                f"R_Fwd={rew.get('R_Forward', 0.0):+.3f} | "
                f"P_Spin={rew.get('P_Spin', 0.0):+.3f}",
                flush=True,
            )

        if (step + 1) % 500 == 0 or (step + 1) == int(args_cli.steps):
            check_obs(env, obs)
            assert_finite_tensor("reward rollout", reward)
            assert torch.isfinite(env.actions).all().item(), "actions 出现 NaN/Inf"
            assert torch.isfinite(env.prev_actions).all().item(), "prev_actions 出现 NaN/Inf"
            assert torch.isfinite(env.last_distances).all().item(), "last_distances 出现 NaN/Inf"
            assert torch.isfinite(env.progress_ema).all().item(), "progress_ema 出现 NaN/Inf"
            assert torch.isfinite(env.waypoints_local).all().item(), "waypoints_local 出现 NaN/Inf"

    elapsed = time.time() - start_time
    fps = int(args_cli.steps) * env.num_envs / max(elapsed, 1e-6)

    print_ok(f"随机策略长跑完成: {args_cli.steps} steps, {int(args_cli.steps) * env.num_envs:,} transitions")
    print_ok(f"吞吐约: {fps:,.2f} env steps/s")
    print_ok(f"累计 waypoint approx: {total_waypoints:,}")
    print_ok(f"累计 finish approx: {total_finished:,}")
    print_ok(f"累计 timeout approx: {total_timeout:,}")

    heading("[测试 14] 奖励组件 / 事件 / 遥测统计报告")
    print_summary_table(summarize_records(records))

    print("Diff-Drive UGV / Jetbot Task1 training pre-check guide:")
    print("1. obs 应为 36 维，即 3 帧 × 12 维。")
    print("2. action=[1,1] 理想情况下应让车前进；如果不是，请调整 left_wheel_sign/right_wheel_sign。")
    print("3. 普通 waypoint 事件和 finish 事件必须能被手动触发。")
    print("4. compute_obs / _compute_single_obs 不应修改 last_distances。")
    print("5. 随机策略下 finish 很低是正常的，但不能出现 NaN/Inf。")
    print("6. 训练时重点看 Progress、Goal_Aligned_Speed、Waypoint_Index、Finish_Rate、Timeout_Rate、Stuck_Ratio。")


def run_tests() -> None:
    heading("Diff-Drive UGV / Jetbot Task1 Multi-waypoint Navigation Env 全量测试启动")

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    if args_cli.test_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print_warn("CUDA 不可用，自动切换到 CPU")
    else:
        device = args_cli.test_device

    if bool(args_cli.quick):
        args_cli.num_envs = min(int(args_cli.num_envs), 64)
        args_cli.steps = min(int(args_cli.steps), 200)
        args_cli.collect_interval = min(int(args_cli.collect_interval), 50)

    check_project_files()
    check_config()

    cfg = Task1Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.seed = int(args_cli.seed)
    cfg.print_debug_info = bool(args_cli.print_names)
    cfg.validate()

    env: DiffDriveTask1Env | None = None

    try:
        env = check_env_init(cfg)
        test_reset_obs_step(env)
        test_reset_alignment_and_waypoints(env)
        test_obs_slices(env)
        test_compute_obs_purity(env)
        test_action_direction(env)
        test_progress_reward_sign(env)
        test_waypoint_event(env)
        test_finish_event(env)
        test_timeout_event(env)
        test_stuck_penalty(env)
        random_rollout(env)

        heading("Diff-Drive UGV / Jetbot Task1 环境测试全部通过")

    except Exception as exc:
        print("\n❌ Diff-Drive UGV / Jetbot Task1 环境测试失败：")
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

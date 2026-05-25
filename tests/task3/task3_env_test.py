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

parser = argparse.ArgumentParser(description="Diff-Drive UGV / Jetbot Task3 Sim2Real Parking Env Test")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--collect-interval", type=int, default=20)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-names", action="store_true")
parser.add_argument("--strict-action-test", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from diff_drive_rl.tasks.task3.task3_config import Task3Config
from diff_drive_rl.tasks.task3.task3_env import DiffDriveTask3Env


# ======================================================================
# Utilities
# ======================================================================

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
    if x.numel() == 0:
        return {"mean": 0.0, "min": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}

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
        f"{name:<44s} "
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
    print(" " * 48 + "Diff-Drive UGV / Jetbot Task3 Env 统计报告")
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
    quat[:, 0] = torch.cos(0.5 * yaw)
    quat[:, 3] = torch.sin(0.5 * yaw)
    return quat


def force_root_local_pose(
    env: DiffDriveTask3Env,
    env_ids: torch.Tensor,
    local_xy: torch.Tensor | None = None,
    height: float | None = None,
    yaw: torch.Tensor | None = None,
    zero_vel: bool = True,
) -> None:
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
    n = int(env_ids.numel())

    root_state = env.robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.env_origins[env_ids]

    if local_xy is not None:
        local_xy = torch.as_tensor(local_xy, dtype=torch.float32, device=env.device)
        if local_xy.ndim == 1:
            local_xy = local_xy.unsqueeze(0).repeat(n, 1)
        root_state[:, 0:2] = env.env_origins[env_ids, :2] + local_xy[:, :2]

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
    env.scene.update(0.0)


def parking_local_to_world(env: DiffDriveTask3Env, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    world = env.world
    yaw = world.goal_yaw
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    x = world.goal_pos[:, 0] + px * c - py * s
    y = world.goal_pos[:, 1] + px * s + py * c
    return torch.stack([x, y], dim=-1)


def reset_action_dr_to_nominal(env: DiffDriveTask3Env) -> None:
    env.world.action_delay_frames[:] = 0
    env.world.action_deadband[:] = 0.0
    env.world.action_ema_alpha[:] = 1.0
    env.world.motor_strength[:] = 1.0
    env.world.motor_bias[:] = 0.0
    env.world.wheel_radius_scale[:] = 1.0

    env.action_delay_buffer[:] = 0.0
    env.raw_actions[:] = 0.0
    env.actions[:] = 0.0
    env.prev_actions[:] = 0.0
    env.applied_actions[:] = 0.0
    env.prev_applied_actions[:] = 0.0


def check_obs(env: DiffDriveTask3Env, obs: torch.Tensor) -> None:
    check_shape("actor obs", obs, (env.num_envs, env.num_observations))
    assert_finite_tensor("actor obs", obs)
    assert obs.abs().max().item() <= float(env.cfg.obs_clip) + 1e-5, (
        f"actor obs 超出 clamp 范围 [-{env.cfg.obs_clip}, {env.cfg.obs_clip}]"
    )


def check_priv_obs(env: DiffDriveTask3Env, priv: torch.Tensor) -> None:
    check_shape("critic privileged obs", priv, (env.num_envs, env.num_privileged_obs))
    assert_finite_tensor("critic privileged obs", priv)
    assert priv.abs().max().item() <= float(env.cfg.priv_clip) + 1e-5, (
        f"critic obs 超出 clamp 范围 [-{env.cfg.priv_clip}, {env.cfg.priv_clip}]"
    )


def get_single_obs_slices(cfg: Task3Config):
    s = {}
    idx = 0

    s["vel_obs"] = slice(idx, idx + 3); idx += 3
    s["wheel_obs"] = slice(idx, idx + 2); idx += 2
    s["goal_dist_norm"] = slice(idx, idx + 1); idx += 1
    s["goal_xy_body"] = slice(idx, idx + 2); idx += 2
    s["heading"] = slice(idx, idx + 2); idx += 2
    s["goal_yaw"] = slice(idx, idx + 2); idx += 2
    s["parking"] = slice(idx, idx + 2); idx += 2
    s["applied_action"] = slice(idx, idx + 2); idx += 2
    s["action_delta"] = slice(idx, idx + 2); idx += 2
    s["progress_ema"] = slice(idx, idx + 1); idx += 1
    s["lidar"] = slice(idx, idx + cfg.world_cfg.lidar_pool_bins); idx += cfg.world_cfg.lidar_pool_bins
    s["lidar_delta"] = slice(idx, idx + cfg.world_cfg.lidar_pool_bins); idx += cfg.world_cfg.lidar_pool_bins
    s["risk"] = slice(idx, idx + 10); idx += 10

    assert idx == int(cfg.single_actor_obs_dim), f"single actor obs slice dim mismatch: {idx} != {cfg.single_actor_obs_dim}"
    return s


def run_fixed_action(env: DiffDriveTask3Env, action_tensor: torch.Tensor, steps: int = 50):
    env.reset()
    reset_action_dr_to_nominal(env)

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    env.world.goal_pos[:] = torch.tensor([4.0, 0.0], dtype=torch.float32, device=env.device)
    env.world.goal_yaw[:] = 0.0

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=torch.tensor([-4.3, 0.0], dtype=torch.float32, device=env.device),
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
    )

    root0 = env._root_pos_local().clone()
    yaw0 = env._yaw().clone()

    env.last_goal_dist[:] = torch.norm(env.world.goal_pos - env._root_pos_local(), dim=-1)

    action = action_tensor.to(env.device).view(1, 2).repeat(env.num_envs, 1)

    obs = None
    reward = None
    terminated = None
    truncated = None
    info = {}

    for _ in range(int(steps)):
        obs, reward, terminated, truncated, info = env.step(action)

    root1 = env._root_pos_local().clone()
    yaw1 = env._yaw().clone()

    delta_pos = root1 - root0
    delta_yaw = torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0))

    return delta_pos, delta_yaw, obs, reward, terminated, truncated, info


# ======================================================================
# Tests
# ======================================================================

def check_project_files() -> None:
    heading("[测试 0] Task3 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task3_sim2real_parking.yaml",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task3" / "task3_world.py",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task3" / "task3_config.py",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task3" / "task3_scene.py",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task3" / "task3_env.py",
        PROJECT_ROOT / "tests" / "task3" / "task3_env_test.py",
    ]

    missing = [str(p) for p in required if not p.exists()]
    assert not missing, "缺少 Task3 必要文件:\n" + "\n".join(missing)

    for path in required:
        print_ok(str(path.relative_to(PROJECT_ROOT)))


def check_config() -> None:
    heading("[测试 1] Task3Config 基础配置检测")

    cfg = Task3Config()
    cfg.validate()

    assert cfg.num_actions == 2
    assert cfg.frame_stack == 4
    assert cfg.single_actor_obs_dim == 101
    assert cfg.actor_obs_dim == 404
    assert cfg.privileged_feature_dim == 38
    assert cfg.critic_obs_dim == 442
    assert cfg.world_cfg.lidar_pool_bins == 36

    print_ok(f"num_actions = {cfg.num_actions}")
    print_ok(f"single_actor_obs_dim = {cfg.single_actor_obs_dim}")
    print_ok(f"frame_stack = {cfg.frame_stack}")
    print_ok(f"actor_obs_dim = {cfg.actor_obs_dim}")
    print_ok(f"privileged_feature_dim = {cfg.privileged_feature_dim}")
    print_ok(f"critic_obs_dim = {cfg.critic_obs_dim}")
    print_ok(f"max_episode_length = {cfg.max_episode_length}")
    print_ok("Task3Config 基础配置正常")


def check_env_init(cfg: Task3Config) -> DiffDriveTask3Env:
    heading("[测试 2] DiffDriveTask3Env 初始化 / 名称映射 / 空间维度检测")

    env = DiffDriveTask3Env(cfg)

    print_ok(f"device = {env.device}")
    print_ok(f"num_envs = {env.num_envs}")
    print_ok(f"robot.num_joints = {env.robot.num_joints}")
    print_ok(f"num_actions = {env.num_actions}")
    print_ok(f"single_actor_obs_dim = {env.cfg.single_actor_obs_dim}")
    print_ok(f"frame_stack = {env.cfg.frame_stack}")
    print_ok(f"actor_obs_dim = {env.num_observations}")
    print_ok(f"world_priv_dim = {env.world_priv_dim}")
    print_ok(f"critic_obs_dim = {env.num_privileged_obs}")
    print_ok(f"policy_dt = {env.dt}")
    print_ok(f"max_episode_length = {env.cfg.max_episode_length}")
    print_ok(f"wheel_joint_ids = {env.wheel_joint_ids}")

    assert env.robot.num_joints >= 2, f"Jetbot 关节数量异常: {env.robot.num_joints}"
    assert env.num_actions == 2
    assert env.cfg.single_actor_obs_dim == 101
    assert env.num_observations == 404
    assert env.world_priv_dim == 38
    assert env.num_privileged_obs == 442
    assert len(env.wheel_joint_ids) == 2

    assert env.observation_space.shape == (404,)
    assert env.state_space.shape == (442,)
    assert env.action_space.shape == (2,)

    assert "park_back" in env.scene.rigid_objects
    assert "park_left" in env.scene.rigid_objects
    assert "park_right" in env.scene.rigid_objects
    for i in range(env.cfg.world_cfg.num_speed_bumps):
        assert f"speed_bump_{i}" in env.scene.rigid_objects

    if args_cli.print_names:
        print("\nrobot.joint_names:")
        for i, name in enumerate(env.robot_joint_names):
            mark = " <wheel>" if i in env.wheel_joint_ids else ""
            print(f"  {i:02d}: {name}{mark}")

        print("\nscene.rigid_objects:")
        for name in sorted(env.scene.rigid_objects.keys()):
            print(f"  - {name}")

        print("\nscene.sensors:")
        for name in sorted(env.scene.sensors.keys()):
            print(f"  - {name}")

    return env


def test_reset_step_basic(env: DiffDriveTask3Env) -> None:
    heading("[测试 3] reset / obs / privileged obs / step 基础检测")

    obs, _ = env.reset()
    check_obs(env, obs)

    priv = env.compute_privileged_obs()
    check_priv_obs(env, priv)

    action = torch.rand((env.num_envs, env.num_actions), device=env.device) * 0.4 - 0.2
    obs, reward, terminated, truncated, info = env.step(action)

    check_obs(env, obs)
    check_priv_obs(env, env.compute_privileged_obs())
    check_shape("reward", reward, (env.num_envs,))
    check_shape("terminated", terminated, (env.num_envs,))
    check_shape("truncated", truncated, (env.num_envs,))
    assert_finite_tensor("reward", reward)

    print_ok(f"reset obs shape = {tuple(obs.shape)}")
    print_ok(f"priv obs shape = {tuple(priv.shape)}")
    print_ok(f"reward mean = {reward.mean().item():+.6f}")
    print_ok(f"terminated count = {terminated.sum().item()}")
    print_ok(f"truncated count = {truncated.sum().item()}")


def test_reset_alignment(env: DiffDriveTask3Env) -> None:
    heading("[测试 4] reset 后 robot local xy / yaw 与 world.start_pos / start_yaw 对齐检测")

    obs, _ = env.reset()
    check_obs(env, obs)

    root_local = env._root_pos_local()
    yaw = env._yaw()

    xy_err = torch.norm(root_local - env.world.start_pos, dim=-1)
    yaw_err = torch.atan2(
        torch.sin(yaw - env.world.start_yaw),
        torch.cos(yaw - env.world.start_yaw),
    ).abs()

    assert xy_err.max().item() < 5e-4, f"reset xy 未对齐，max_err={xy_err.max().item():.8f}"
    assert yaw_err.max().item() < 5e-4, f"reset yaw 未对齐，max_err={yaw_err.max().item():.8f}"

    goal_dist = torch.norm(env.world.goal_pos - env.world.start_pos, dim=-1)
    last_err = torch.abs(goal_dist - env.last_goal_dist)

    assert last_err.max().item() < 1e-4, "reset 后 last_goal_dist 未同步"

    print_stats("reset xy error", xy_err)
    print_stats("reset yaw error", yaw_err)
    print_stats("start-to-goal distance", goal_dist)
    print_ok("reset 对齐正常")


def test_domain_randomization_buffers(env: DiffDriveTask3Env) -> None:
    heading("[测试 5] Sim2Real DR buffers / privileged obs 检测")

    env.reset()

    cfg = env.cfg.world_cfg
    checks = [
        ("action_deadband", env.world.action_deadband, cfg.action_deadband_range),
        ("action_ema_alpha", env.world.action_ema_alpha, cfg.action_ema_alpha_range),
        ("motor_strength", env.world.motor_strength, cfg.motor_strength_range),
        ("motor_bias", env.world.motor_bias, cfg.motor_bias_range),
        ("wheel_radius_scale", env.world.wheel_radius_scale, cfg.wheel_radius_scale_range),
        ("lidar_noise_std", env.world.lidar_noise_std, cfg.lidar_noise_std_range),
        ("lidar_outlier_prob", env.world.lidar_outlier_prob, cfg.lidar_outlier_prob_range),
        ("lidar_dropout_prob", env.world.lidar_dropout_prob, cfg.lidar_dropout_prob_range),
        ("lidar_yaw_offset", env.world.lidar_yaw_offset, cfg.lidar_yaw_offset_range),
        ("lidar_z_offset", env.world.lidar_z_offset, cfg.lidar_z_offset_range),
    ]

    for name, tensor, rng in checks:
        assert_finite_tensor(name, tensor)
        assert tensor.min().item() >= float(rng[0]) - 1e-6, f"{name} below range"
        assert tensor.max().item() <= float(rng[1]) + 1e-6, f"{name} above range"
        print_ok(f"{name:<22s} range = {tensor.min().item():+.6f} ~ {tensor.max().item():+.6f}")

    assert env.world.action_delay_frames.min().item() >= cfg.action_delay_frame_range[0]
    assert env.world.action_delay_frames.max().item() <= cfg.action_delay_frame_range[1]

    priv = env.compute_privileged_obs()
    check_priv_obs(env, priv)

    print_ok(f"action_delay range = {env.world.action_delay_frames.min().item()} ~ {env.world.action_delay_frames.max().item()}")
    print_ok("DR buffers / privileged obs 正常")


def test_action_model(env: DiffDriveTask3Env) -> None:
    heading("[测试 6] Sim2Real action delay / deadband / EMA / motor scaling 白盒检测")

    env.reset()

    a = torch.tensor([[1.0, -1.0]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)

    # Delay test: delay=2 means first two outputs are zero, third output receives the action.
    reset_action_dr_to_nominal(env)
    env.world.action_delay_frames[:] = 2
    env.world.action_deadband[:] = 0.0
    env.world.action_ema_alpha[:] = 1.0

    out1 = env._apply_action_model(a)
    out2 = env._apply_action_model(a)
    out3 = env._apply_action_model(a)

    assert out1.abs().max().item() < 1e-6, f"delay out1 应为 0，实际 {out1.abs().max().item()}"
    assert out2.abs().max().item() < 1e-6, f"delay out2 应为 0，实际 {out2.abs().max().item()}"
    assert torch.allclose(out3, a, atol=1e-5), "delay 第三帧应输出原始 action"

    # Deadband test.
    reset_action_dr_to_nominal(env)
    env.world.action_deadband[:] = 0.2
    env.world.action_ema_alpha[:] = 1.0

    small = torch.tensor([[0.10, -0.10]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    medium = torch.tensor([[0.60, -0.60]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)

    small_out = env._apply_action_model(small)
    medium_out = env._apply_action_model(medium)

    expected_medium = torch.tensor([[0.50, -0.50]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)

    assert small_out.abs().max().item() < 1e-6, "deadband 内 action 应为 0"
    assert torch.allclose(medium_out, expected_medium, atol=1e-5), "deadband 输出不符合线性重标定"

    # EMA + motor scaling test.
    reset_action_dr_to_nominal(env)
    env.world.action_ema_alpha[:] = 0.5
    env.world.motor_strength[:] = 0.8
    env.world.wheel_radius_scale[:] = 1.0
    env.world.motor_bias[:] = 0.0

    ema_out = env._apply_action_model(a)
    expected_ema = a * 0.5 * 0.8

    assert torch.allclose(ema_out, expected_ema, atol=1e-5), "EMA / motor_strength 输出异常"

    print_ok("action delay 生效")
    print_ok("action deadband 生效")
    print_ok("action EMA / motor strength 生效")


def test_obs_slices(env: DiffDriveTask3Env) -> None:
    heading("[测试 7] actor observation 切片 / lidar / risk 范围检测")

    obs, _ = env.reset()
    check_obs(env, obs)

    single_dim = env.cfg.single_actor_obs_dim
    last_frame = obs[:, -single_dim:]
    s = get_single_obs_slices(env.cfg)

    parts = {
        "vel_obs": last_frame[:, s["vel_obs"]],
        "wheel_obs": last_frame[:, s["wheel_obs"]],
        "goal_dist_norm": last_frame[:, s["goal_dist_norm"]],
        "goal_xy_body": last_frame[:, s["goal_xy_body"]],
        "heading": last_frame[:, s["heading"]],
        "goal_yaw": last_frame[:, s["goal_yaw"]],
        "parking": last_frame[:, s["parking"]],
        "applied_action": last_frame[:, s["applied_action"]],
        "action_delta": last_frame[:, s["action_delta"]],
        "progress_ema": last_frame[:, s["progress_ema"]],
        "lidar": last_frame[:, s["lidar"]],
        "lidar_delta": last_frame[:, s["lidar_delta"]],
        "risk": last_frame[:, s["risk"]],
    }

    for name, x in parts.items():
        assert_finite_tensor(name, x)

    check_shape("lidar", parts["lidar"], (env.num_envs, env.cfg.world_cfg.lidar_pool_bins))
    check_shape("lidar_delta", parts["lidar_delta"], (env.num_envs, env.cfg.world_cfg.lidar_pool_bins))
    check_shape("risk", parts["risk"], (env.num_envs, 10))

    assert parts["heading"].abs().max().item() <= 1.0 + 1e-5
    assert parts["goal_yaw"].abs().max().item() <= 1.0 + 1e-5
    assert parts["applied_action"].abs().max().item() <= 1.0 + 1e-5
    assert parts["lidar"].min().item() >= 0.0
    assert parts["lidar"].max().item() <= 1.0 + 1e-5
    assert parts["lidar_delta"].min().item() >= -1.0 - 1e-5
    assert parts["lidar_delta"].max().item() <= 1.0 + 1e-5
    assert parts["risk"].min().item() >= 0.0
    assert parts["risk"].max().item() <= 1.0 + 1e-5

    print_ok(f"actor obs shape = {tuple(obs.shape)}")
    print_ok(f"last frame range = {last_frame.min().item():+.6f} ~ {last_frame.max().item():+.6f}")
    print_ok(f"lidar range = {parts['lidar'].min().item():.6f} ~ {parts['lidar'].max().item():.6f}")
    print_ok(f"risk range = {parts['risk'].min().item():.6f} ~ {parts['risk'].max().item():.6f}")


def test_step_return_structure(env: DiffDriveTask3Env) -> None:
    heading("[测试 8] 向量化 step 返回结构与 info 字典检测")

    obs, _ = env.reset()
    check_obs(env, obs)

    actions = torch.rand((env.num_envs, env.num_actions), device=env.device) * 0.4 - 0.2
    obs, reward, terminated, truncated, info = env.step(actions)

    check_obs(env, obs)
    check_priv_obs(env, env.compute_privileged_obs())
    check_shape("reward", reward, (env.num_envs,))
    check_shape("terminated", terminated, (env.num_envs,))
    check_shape("truncated", truncated, (env.num_envs,))
    assert_finite_tensor("reward", reward)

    for group in ["reward_components", "events", "telemetry", "world", "debug"]:
        assert group in info, f"info 缺少分组: {group}"

    required_reward_keys = [
        "R_Progress",
        "R_Goal_Speed",
        "R_Heading",
        "R_Parking_Pos",
        "R_Parking_Yaw",
        "R_Inside_Box",
        "R_Parking_Low_Speed",
        "R_Terrain_Progress",
        "R_Bump_Progress",
        "P_Bump_Smooth",
        "R_Front_Clearance",
        "P_Lidar_Risk",
        "P_Wall_Risk",
        "P_Lane_Risk",
        "P_Bump_Risk",
        "P_Spin",
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
        "Success_Rate",
        "Success_Candidate_Rate",
        "Crash_Rate",
        "Out_Of_Lane_Rate",
        "Parking_Wall_Collision_Rate",
        "Bump_Overlap_Rate",
        "Timeout_Rate",
        "Done_Rate",
        "Episode_Success_Rate",
        "Episode_Crash_Rate",
        "Episode_Timeout_Rate",
        "Episode_Done_Count",
    ]

    for key in required_event_keys:
        assert key in info["events"], f"events 缺少 {key}"

    required_tel_keys = [
        "Goal_Dist",
        "Progress",
        "Goal_Aligned_Speed",
        "Heading_Error",
        "Goal_Yaw_Error",
        "Parking_Pos_Error",
        "Inside_Box",
        "Success_Hold",
        "Lidar_Min",
        "Risk_Front",
        "Risk_Wall",
        "Risk_Lane",
        "Action_Delay",
        "Action_Deadband",
        "Motor_Strength",
        "Wheel_Radius_Scale",
        "Episode_Length",
    ]

    for key in required_tel_keys:
        assert key in info["telemetry"], f"telemetry 缺少 {key}"

    print_ok(f"reward mean = {reward.mean().item():+.6f}")
    print_ok(f"reward min/max = {reward.min().item():+.6f} / {reward.max().item():+.6f}")
    print_ok(f"terminated count = {terminated.sum().item()}")
    print_ok(f"truncated count = {truncated.sum().item()}")
    print_ok("step 返回结构正常")


def test_action_direction(env: DiffDriveTask3Env) -> None:
    heading("[测试 9] 动作方向白盒测试：前进 / 后退 / 原地旋转")

    steps = 50

    delta_forward, yaw_forward, *_ = run_fixed_action(env, torch.tensor([1.0, 1.0]), steps=steps)
    forward_x = delta_forward[:, 0]

    delta_backward, yaw_backward, *_ = run_fixed_action(env, torch.tensor([-1.0, -1.0]), steps=steps)
    backward_x = delta_backward[:, 0]

    delta_turn, yaw_turn, *_ = run_fixed_action(env, torch.tensor([1.0, -1.0]), steps=steps)
    turn_xy = torch.norm(delta_turn, dim=-1)

    print_stats("forward delta x", forward_x)
    print_stats("backward delta x", backward_x)
    print_stats("turn delta yaw", yaw_turn)
    print_stats("turn xy movement", turn_xy)

    forward_ok = forward_x.mean().item() > 0.03
    backward_ok = backward_x.mean().item() < -0.03
    turn_ok = yaw_turn.abs().mean().item() > 0.08

    if args_cli.strict_action_test:
        assert forward_ok, "action=[1,1] 没有让车明显前进。请检查 wheel signs。"
        assert backward_ok, "action=[-1,-1] 没有让车明显后退。请检查 wheel signs。"
        assert turn_ok, "action=[1,-1] 没有让车明显原地转向。请检查 wheel signs 或 joint 顺序。"
    else:
        if not forward_ok:
            print_warn("action=[1,1] 没有明显前进。若训练变慢，请调整 left_wheel_sign/right_wheel_sign。")
        if not backward_ok:
            print_warn("action=[-1,-1] 没有明显后退。若训练变慢，请调整 wheel signs。")
        if not turn_ok:
            print_warn("action=[1,-1] 没有明显转向。若训练变慢，请检查 wheel signs 或 joint 顺序。")

    print_ok("动作方向测试完成")


def test_success_event(env: DiffDriveTask3Env) -> None:
    heading("[测试 10] 手动触发 stable success 事件检测")

    env.reset()
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=env.world.goal_pos.clone(),
        height=env.cfg.spawn_height,
        yaw=env.world.goal_yaw.clone(),
        zero_vel=True,
    )

    env.last_goal_dist[:] = torch.norm(env.world.goal_pos - env._root_pos_local(), dim=-1)
    env.success_hold_counter[:] = int(env.cfg.world_cfg.success_hold_steps) - 1

    reward, terminated, truncated, info = env._compute_rewards_and_dones(pre_goal_dist=env.last_goal_dist.clone())

    assert terminated.float().mean().item() > 0.99, "到达泊车位且 hold 满足后没有触发 terminated"
    assert info["events"]["Success_Rate"] > 0.99, "Success_Rate 未触发"
    assert info["events"]["Crash_Rate"] < 1e-6, "success 不应 crash"
    assert truncated.float().mean().item() < 1e-6, "success 不应 truncated"

    print_ok(f"Success_Rate = {info['events']['Success_Rate']:.6f}")
    print_ok(f"Success_Candidate_Rate = {info['events']['Success_Candidate_Rate']:.6f}")
    print_ok(f"Event reward mean = {info['reward_components']['Event']:.6f}")


def test_out_of_lane_event(env: DiffDriveTask3Env) -> None:
    heading("[测试 11] 手动触发 out_of_lane / crash 事件检测")

    env.reset()
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    local_xy = torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
    local_xy[:, 0] = 0.0
    local_xy[:, 1] = float(env.cfg.world_cfg.y_max) + 0.30

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=local_xy,
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, device=env.device),
        zero_vel=True,
    )

    reward, terminated, truncated, info = env._compute_rewards_and_dones(pre_goal_dist=env.last_goal_dist.clone())

    assert terminated.float().mean().item() > 0.99, "越界后没有触发 terminated"
    assert info["events"]["Out_Of_Lane_Rate"] > 0.99
    assert info["events"]["Crash_Rate"] > 0.99

    print_ok(f"Out_Of_Lane_Rate = {info['events']['Out_Of_Lane_Rate']:.6f}")
    print_ok(f"Crash_Rate = {info['events']['Crash_Rate']:.6f}")


def test_parking_wall_collision_event(env: DiffDriveTask3Env) -> None:
    heading("[测试 12] 手动触发 parking wall collision / crash 事件检测")

    env.reset()
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    px = env.world.spot_depth_inner * 0.5 + float(env.cfg.world_cfg.wall_thickness) * 0.5
    py = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    root_wall = parking_local_to_world(env, px, py)

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=root_wall,
        height=env.cfg.spawn_height,
        yaw=env.world.goal_yaw.clone(),
        zero_vel=True,
    )

    reward, terminated, truncated, info = env._compute_rewards_and_dones(pre_goal_dist=env.last_goal_dist.clone())

    assert terminated.float().mean().item() > 0.99, "压到泊车墙后没有触发 terminated"
    assert info["events"]["Parking_Wall_Collision_Rate"] > 0.99
    assert info["events"]["Crash_Rate"] > 0.99

    print_ok(f"Parking_Wall_Collision_Rate = {info['events']['Parking_Wall_Collision_Rate']:.6f}")
    print_ok(f"Crash_Rate = {info['events']['Crash_Rate']:.6f}")


def test_bump_overlap_event(env: DiffDriveTask3Env) -> None:
    heading("[测试 13] 手动触发 bump overlap 检测")

    env.reset()
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    root_bump = env.world.bump_pos[:, 0, :].clone()

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=root_bump,
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
        zero_vel=True,
    )

    events = env.world.check_events(
        env._root_pos_local(),
        env._yaw(),
        body_lin_vel=env.robot.data.root_lin_vel_b,
        body_ang_vel=env.robot.data.root_ang_vel_b,
    )

    assert events["bump_overlap"].float().mean().item() > 0.99, "放到减速带上没有触发 bump_overlap"

    reward, terminated, truncated, info = env._compute_rewards_and_dones(pre_goal_dist=env.last_goal_dist.clone())
    assert info["events"]["Bump_Overlap_Rate"] > 0.99

    print_ok(f"Bump_Overlap_Rate = {info['events']['Bump_Overlap_Rate']:.6f}")
    print_ok("注意：bump_overlap 本身不一定终止 episode，这是保守低减速带设计。")


def test_timeout_event(env: DiffDriveTask3Env) -> None:
    heading("[测试 14] timeout truncated 事件检测")

    env.reset()
    env.episode_steps[:] = int(env.cfg.max_episode_length)

    reward, terminated, truncated, info = env._compute_rewards_and_dones(pre_goal_dist=env.last_goal_dist.clone())

    assert truncated.float().mean().item() > 0.99, "episode_steps 到达最大长度后没有触发 truncated"
    assert terminated.float().mean().item() < 1e-6, "timeout 不应 terminated"

    print_ok(f"Timeout_Rate = {info['events']['Timeout_Rate']:.6f}")
    print_ok(f"terminated mean = {terminated.float().mean().item():.6f}")
    print_ok(f"truncated mean = {truncated.float().mean().item():.6f}")


def test_reward_direction(env: DiffDriveTask3Env) -> None:
    heading("[测试 15] progress reward 正负方向检测")

    env.reset()
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    env.world.goal_pos[:] = torch.tensor([4.0, 0.0], dtype=torch.float32, device=env.device)
    env.world.goal_yaw[:] = 0.0

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=torch.tensor([0.0, 0.0], dtype=torch.float32, device=env.device),
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, device=env.device),
        zero_vel=True,
    )

    pre_dist = torch.full((env.num_envs,), 4.0, dtype=torch.float32, device=env.device)

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=torch.tensor([0.30, 0.0], dtype=torch.float32, device=env.device),
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, device=env.device),
        zero_vel=True,
    )
    reward_fwd, _, _, info_fwd = env._compute_rewards_and_dones(pre_goal_dist=pre_dist)

    env.reset()
    env.world.goal_pos[:] = torch.tensor([4.0, 0.0], dtype=torch.float32, device=env.device)
    env.world.goal_yaw[:] = 0.0

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=torch.tensor([-0.30, 0.0], dtype=torch.float32, device=env.device),
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, device=env.device),
        zero_vel=True,
    )
    reward_back, _, _, info_back = env._compute_rewards_and_dones(pre_goal_dist=pre_dist)

    assert info_fwd["telemetry"]["Progress"] > 0.0, "朝 goal 前进时 Progress 应为正"
    assert info_back["telemetry"]["Progress"] < 0.0, "远离 goal 时 Progress 应为负"
    assert reward_fwd.mean().item() > reward_back.mean().item(), "朝 goal 前进奖励应大于远离 goal"

    print_ok(f"forward progress = {info_fwd['telemetry']['Progress']:+.6f}, reward = {reward_fwd.mean().item():+.6f}")
    print_ok(f"backward progress = {info_back['telemetry']['Progress']:+.6f}, reward = {reward_back.mean().item():+.6f}")
    print_ok("progress reward 方向正确")


def random_rollout(env: DiffDriveTask3Env) -> None:
    heading(f"[测试 16] 随机策略运行 {args_cli.steps} 步，收集奖励组件 / 事件 / 遥测")

    obs, _ = env.reset()
    check_obs(env, obs)

    records: List[Dict[str, float]] = []

    total_success = 0
    total_crash = 0
    total_timeout = 0

    start_time = time.time()

    for step in range(int(args_cli.steps)):
        actions = torch.rand((env.num_envs, env.num_actions), device=env.device) * 2.0 - 1.0
        obs, reward, terminated, truncated, info = env.step(actions)

        check_obs(env, obs)
        assert_finite_tensor("reward rollout", reward)

        if (step + 1) % max(int(args_cli.collect_interval), 1) == 0 or (step + 1) == int(args_cli.steps):
            priv = env.compute_privileged_obs()
            check_priv_obs(env, priv)

            flat = flatten_info(info)
            flat["Reward_Mean_Step"] = reward.mean().item()
            flat["Reward_Min_Step"] = reward.min().item()
            flat["Reward_Max_Step"] = reward.max().item()
            flat["Terminated_Count"] = terminated.sum().item()
            flat["Truncated_Count"] = truncated.sum().item()
            records.append(flat)

            tel = info.get("telemetry", {})
            ev = info.get("events", {})
            rew = info.get("reward_components", {})

            print(
                f" -> Step {step + 1:05d} | "
                f"Reward={reward.mean().item():+.4f} | "
                f"GoalDist={tel.get('Goal_Dist', 0.0):.2f} | "
                f"Progress={tel.get('Progress', 0.0):+.4f} | "
                f"GoalV={tel.get('Goal_Aligned_Speed', 0.0):+.3f} | "
                f"HeadErr={tel.get('Heading_Error', 0.0):.3f} | "
                f"YawErr={tel.get('Goal_Yaw_Error', 0.0):.3f} | "
                f"ParkErr={tel.get('Parking_Pos_Error', 0.0):.3f} | "
                f"LidarMin={tel.get('Lidar_Min', 0.0):.3f} | "
                f"RiskF={tel.get('Risk_Front', 0.0):.3f} | "
                f"RiskWall={tel.get('Risk_Wall', 0.0):.3f} | "
                f"Succ={ev.get('Success_Rate', 0.0):.4f} | "
                f"Crash={ev.get('Crash_Rate', 0.0):.4f} | "
                f"Timeout={ev.get('Timeout_Rate', 0.0):.4f} | "
                f"R_Prog={rew.get('R_Progress', 0.0):+.3f} | "
                f"R_Park={rew.get('R_Parking_Pos', 0.0):+.3f} | "
                f"P_Wall={rew.get('P_Wall_Risk', 0.0):+.3f}",
                flush=True,
            )

        total_success += int(info["events"]["Success_Rate"] * env.num_envs)
        total_crash += int(info["events"]["Crash_Rate"] * env.num_envs)
        total_timeout += int(info["events"]["Timeout_Rate"] * env.num_envs)

        assert torch.isfinite(env.raw_actions).all().item(), "raw_actions 出现 NaN/Inf"
        assert torch.isfinite(env.applied_actions).all().item(), "applied_actions 出现 NaN/Inf"
        assert torch.isfinite(env.last_goal_dist).all().item(), "last_goal_dist 出现 NaN/Inf"
        assert torch.isfinite(env.progress_ema).all().item(), "progress_ema 出现 NaN/Inf"
        assert torch.isfinite(env.world.start_pos).all().item(), "world.start_pos 出现 NaN/Inf"
        assert torch.isfinite(env.world.goal_pos).all().item(), "world.goal_pos 出现 NaN/Inf"
        assert torch.isfinite(env.world.last_lidar).all().item(), "world.last_lidar 出现 NaN/Inf"
        assert torch.isfinite(env.world.last_risk_features).all().item(), "world.last_risk_features 出现 NaN/Inf"

    elapsed = time.time() - start_time
    fps = int(args_cli.steps) * env.num_envs / max(elapsed, 1e-6)

    print_ok(f"随机策略长跑完成: {args_cli.steps} steps, {int(args_cli.steps) * env.num_envs:,} transitions")
    print_ok(f"吞吐约: {fps:,.2f} env steps/s")
    print_ok(f"累计 success approx: {total_success:,}")
    print_ok(f"累计 crash approx: {total_crash:,}")
    print_ok(f"累计 timeout approx: {total_timeout:,}")

    heading("[测试 17] 奖励组件 / 事件 / 遥测统计报告")
    print_summary_table(summarize_records(records))

    print("Diff-Drive UGV / Jetbot Task3 training pre-check guide:")
    print("1. actor obs 必须为 404 维，即 4 帧 × 101 维。")
    print("2. critic obs 必须为 442 维，即 actor obs 404 + world privileged 38。")
    print("3. reset 后 root local xy / yaw 必须对齐 start_pos / start_yaw。")
    print("4. Sim2Real action model 的 delay / deadband / EMA / motor strength 必须通过白盒测试。")
    print("5. success / crash / timeout 必须能被手动触发。")
    print("6. 随机策略下 crash 非零是正常的，但不能出现 NaN/Inf。")
    print("7. 训练重点看 Goal_Dist、Progress、Parking_Pos_Error、Goal_Yaw_Error、Success_Rate、Crash_Rate。")


# ======================================================================
# Main
# ======================================================================

def run_tests() -> None:
    heading("🚀 Diff-Drive UGV / Jetbot Task3 Sim2Real Parking Env 全量测试启动")

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    if args_cli.test_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print_warn("CUDA 不可用，自动切换到 CPU")
    else:
        device = args_cli.test_device

    if bool(args_cli.quick):
        args_cli.num_envs = min(int(args_cli.num_envs), 2)
        args_cli.steps = min(int(args_cli.steps), 50)
        args_cli.collect_interval = min(int(args_cli.collect_interval), 10)

    check_project_files()
    check_config()

    cfg = Task3Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.seed = int(args_cli.seed)
    cfg.print_debug_info = bool(args_cli.print_names)
    cfg.validate()

    env: DiffDriveTask3Env | None = None

    try:
        env = check_env_init(cfg)

        test_reset_step_basic(env)
        test_reset_alignment(env)
        test_domain_randomization_buffers(env)
        test_action_model(env)
        test_obs_slices(env)
        test_step_return_structure(env)
        test_action_direction(env)
        test_success_event(env)
        test_out_of_lane_event(env)
        test_parking_wall_collision_event(env)
        test_bump_overlap_event(env)
        test_timeout_event(env)
        test_reward_direction(env)
        random_rollout(env)

        heading("✅ Diff-Drive UGV / Jetbot Task3 环境测试全部通过")

    except Exception as exc:
        print("\n❌ Diff-Drive UGV / Jetbot Task3 环境测试失败：")
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

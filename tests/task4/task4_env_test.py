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

parser = argparse.ArgumentParser(description="Diff-Drive UGV / Jetbot Task4 Multi-UGV Formation Escort Env Test")
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

from diff_drive_rl.tasks.task4.task4_config import Task4Config
from diff_drive_rl.tasks.task4.task4_env import DiffDriveTask4Env


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
        f"{name:<46s} "
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
    print(" " * 48 + "Diff-Drive UGV / Jetbot Task4 Env 统计报告")
    print("=" * 188)
    print(
        f"{'metric':<84} | {'mean':>11} | {'var':>11} | {'min':>11} | "
        f"{'p25':>11} | {'p50':>11} | {'p75':>11} | {'max':>11}"
    )
    print("-" * 188)

    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<84} | "
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


def force_robot_root_poses(
    env: DiffDriveTask4Env,
    root_local_xy: torch.Tensor,
    yaw: torch.Tensor,
    height: float | None = None,
    zero_vel: bool = True,
) -> None:
    """Force all 4 robots to desired env-local xy / yaw.

    root_local_xy: [N, 4, 2]
    yaw:           [N, 4]
    """

    root_local_xy = torch.as_tensor(root_local_xy, dtype=torch.float32, device=env.device)
    yaw = torch.as_tensor(yaw, dtype=torch.float32, device=env.device)

    assert root_local_xy.shape == (env.num_envs, env.num_agents, 2)
    assert yaw.shape == (env.num_envs, env.num_agents)

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    z = float(env.cfg.spawn_height if height is None else height)

    for agent_id, robot in enumerate(env.robots):
        root_state = robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += env.env_origins[env_ids]
        root_state[:, 0:2] = env.env_origins[env_ids, :2] + root_local_xy[:, agent_id, :]
        root_state[:, 2] = env.env_origins[env_ids, 2] + z
        root_state[:, 3:7] = yaw_to_quat_wxyz(yaw[:, agent_id])

        if zero_vel:
            root_state[:, 7:13] = 0.0

        robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    env.scene.write_data_to_sim()
    env.scene.update(0.0)


def reset_action_dr_to_nominal(env: DiffDriveTask4Env) -> None:
    env.world.action_delay_frames[:] = 0
    env.world.action_deadband[:] = 0.0
    env.world.action_ema_alpha[:] = 1.0
    env.world.motor_strength[:] = 1.0
    env.world.motor_bias[:] = 0.0
    env.world.wheel_radius_scale[:] = 1.0
    env.world.max_speed[:] = 1.0
    env.world.max_yaw_rate[:] = 1.0

    env.action_delay_buffer[:] = 0.0
    env.raw_actions[:] = 0.0
    env.actions[:] = 0.0
    env.prev_actions[:] = 0.0
    env.applied_actions[:] = 0.0
    env.prev_applied_actions[:] = 0.0


def check_obs(env: DiffDriveTask4Env, obs: torch.Tensor) -> None:
    check_shape("actor obs", obs, (env.num_envs, env.num_agents, env.num_observations))
    assert_finite_tensor("actor obs", obs)
    assert obs.abs().max().item() <= float(env.cfg.obs_clip) + 1e-5


def check_state(env: DiffDriveTask4Env, state: torch.Tensor) -> None:
    check_shape("critic state", state, (env.num_envs, env.num_privileged_obs))
    assert_finite_tensor("critic state", state)
    assert state.abs().max().item() <= float(env.cfg.priv_clip) + 1e-5


def get_single_obs_slices(cfg: Task4Config):
    s = {}
    idx = 0

    s["vel_obs"] = slice(idx, idx + 3); idx += 3
    s["wheel_obs"] = slice(idx, idx + 2); idx += 2
    s["goal_obs"] = slice(idx, idx + 5); idx += 5
    s["slot_obs"] = slice(idx, idx + 3); idx += 3
    s["team_heading"] = slice(idx, idx + 2); idx += 2
    s["formation"] = slice(idx, idx + 4); idx += 4
    s["agent_id"] = slice(idx, idx + 4); idx += 4
    s["teammate_pos"] = slice(idx, idx + 9); idx += 9
    s["teammate_vel"] = slice(idx, idx + 6); idx += 6
    s["applied_action"] = slice(idx, idx + 2); idx += 2
    s["action_delta"] = slice(idx, idx + 2); idx += 2
    s["progress_ema"] = slice(idx, idx + 1); idx += 1
    s["center_speed"] = slice(idx, idx + 1); idx += 1
    s["lidar"] = slice(idx, idx + cfg.world_cfg.lidar_pool_bins); idx += cfg.world_cfg.lidar_pool_bins
    s["lidar_delta"] = slice(idx, idx + cfg.world_cfg.lidar_pool_bins); idx += cfg.world_cfg.lidar_pool_bins
    s["risk"] = slice(idx, idx + 16); idx += 16

    assert idx == int(cfg.single_actor_obs_dim), f"single actor obs slice dim mismatch: {idx} != {cfg.single_actor_obs_dim}"
    return s


def run_fixed_action(env: DiffDriveTask4Env, action_per_agent: torch.Tensor, steps: int = 40):
    env.reset(options={"stage": 0})
    reset_action_dr_to_nominal(env)

    start_center = torch.tensor([[-6.0, 0.0]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    goal = torch.tensor([[4.0, 0.0]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)

    env.world.goal_pos[:] = goal
    env.world.goal_yaw[:] = 0.0
    env.world.formation_type[:] = 0
    env.world.formation_scale[:] = 1.0

    slots = env.world.compute_formation_slots(
        center_xy=start_center,
        heading=torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
        formation_type=env.world.formation_type,
        scale=env.world.formation_scale,
    )
    yaw = torch.zeros((env.num_envs, env.num_agents), dtype=torch.float32, device=env.device)

    force_robot_root_poses(env, slots, yaw, zero_vel=True)

    root0 = env._root_pos_local().clone()
    yaw0 = env._yaw().clone()

    env.last_center_goal_dist[:] = torch.norm(env.world.goal_pos - root0.mean(dim=1), dim=-1)

    action = action_per_agent.to(env.device).view(1, 1, 2).repeat(env.num_envs, env.num_agents, 1)

    obs = None
    reward = None
    terminated = None
    truncated = None
    info = {}

    for _ in range(int(steps)):
        obs, reward, terminated, truncated, info = env.step(action)

    root1 = env._root_pos_local().clone()
    yaw1 = env._yaw().clone()

    center_delta = root1.mean(dim=1) - root0.mean(dim=1)
    yaw_delta = torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0))

    return center_delta, yaw_delta, obs, reward, terminated, truncated, info


# ======================================================================
# Tests
# ======================================================================

def check_project_files() -> None:
    heading("[测试 0] Task4 环境工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task4_multi_ugv_formation_escort.yaml",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task4" / "task4_world.py",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task4" / "task4_config.py",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task4" / "task4_scene.py",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task4" / "task4_env.py",
        PROJECT_ROOT / "tests" / "task4" / "task4_env_test.py",
    ]

    missing = [str(p) for p in required if not p.exists()]
    assert not missing, "缺少 Task4 必要文件:\n" + "\n".join(missing)

    for path in required:
        print_ok(str(path.relative_to(PROJECT_ROOT)))


def check_config() -> None:
    heading("[测试 1] Task4Config 基础配置检测")

    cfg = Task4Config()
    cfg.validate()

    assert cfg.num_agents == 4
    assert cfg.num_actions_per_agent == 2
    assert cfg.single_actor_obs_dim == 156
    assert cfg.frame_stack == 4
    assert cfg.actor_obs_dim == 624
    assert cfg.critic_obs_dim == 96
    assert cfg.world_cfg.lidar_pool_bins == 48

    print_ok(f"num_agents = {cfg.num_agents}")
    print_ok(f"action_per_agent = {cfg.num_actions_per_agent}")
    print_ok(f"single_actor_obs_dim = {cfg.single_actor_obs_dim}")
    print_ok(f"frame_stack = {cfg.frame_stack}")
    print_ok(f"actor_obs_dim = {cfg.actor_obs_dim}")
    print_ok(f"critic_obs_dim = {cfg.critic_obs_dim}")
    print_ok(f"max_episode_length = {cfg.max_episode_length}")
    print_ok("Task4Config 基础配置正常")


def check_env_init(cfg: Task4Config) -> DiffDriveTask4Env:
    heading("[测试 2] DiffDriveTask4Env 初始化 / 名称映射 / 空间维度检测")

    env = DiffDriveTask4Env(cfg)

    print_ok(f"device = {env.device}")
    print_ok(f"num_envs = {env.num_envs}")
    print_ok(f"num_agents = {env.num_agents}")
    print_ok(f"num_actions_per_agent = {env.num_actions}")
    print_ok(f"single_actor_obs_dim = {env.cfg.single_actor_obs_dim}")
    print_ok(f"frame_stack = {env.cfg.frame_stack}")
    print_ok(f"actor_obs_dim = {env.num_observations}")
    print_ok(f"critic_state_dim = {env.num_privileged_obs}")
    print_ok(f"policy_dt = {env.dt}")
    print_ok(f"max_episode_length = {env.cfg.max_episode_length}")

    assert len(env.robots) == 4
    assert env.num_agents == 4
    assert env.num_actions == 2
    assert env.num_observations == 624
    assert env.num_privileged_obs == 96

    assert env.observation_space.shape == (4, 624)
    assert env.state_space.shape == (96,)
    assert env.action_space.shape == (4, 2)

    for agent_id in range(4):
        assert len(env.wheel_joint_ids[agent_id]) == 2

    if args_cli.print_names:
        for agent_id, robot in enumerate(env.robots):
            print(f"\nrobot_{agent_id}.joint_names:")
            for i, name in enumerate(list(getattr(robot, "joint_names", []))):
                mark = " <wheel>" if i in env.wheel_joint_ids[agent_id] else ""
                print(f"  {i:02d}: {name}{mark}")

        print("\nscene.rigid_objects:")
        for name in sorted(env.scene.rigid_objects.keys()):
            print(f"  - {name}")

        print("\nscene.sensors:")
        for name in sorted(env.scene.sensors.keys()):
            print(f"  - {name}")

    return env


def test_reset_step_basic(env: DiffDriveTask4Env) -> None:
    heading("[测试 3] reset / obs / state / action / step 基础检测")

    obs, info = env.reset(options={"stage": 0})
    check_obs(env, obs)
    assert "state" in info
    check_state(env, info["state"])

    state = env.compute_privileged_obs()
    check_state(env, state)

    action = torch.rand((env.num_envs, env.num_agents, env.num_actions), device=env.device) * 0.4 - 0.2
    obs, reward, terminated, truncated, info = env.step(action)

    check_obs(env, obs)
    check_state(env, info["state"])
    check_shape("reward", reward, (env.num_envs, env.num_agents))
    check_shape("terminated", terminated, (env.num_envs,))
    check_shape("truncated", truncated, (env.num_envs,))
    assert_finite_tensor("reward", reward)

    print_ok(f"obs shape = {tuple(obs.shape)}")
    print_ok(f"state shape = {tuple(info['state'].shape)}")
    print_ok(f"reward shape = {tuple(reward.shape)}")
    print_ok(f"reward mean = {reward.mean().item():+.6f}")
    print_ok(f"terminated count = {terminated.sum().item()}")
    print_ok(f"truncated count = {truncated.sum().item()}")


def test_reset_alignment(env: DiffDriveTask4Env) -> None:
    heading("[测试 4] reset 后 4 车 root local xy / yaw 与 world.start_pos / start_yaw 对齐检测")

    obs, info = env.reset(options={"stage": 0})
    check_obs(env, obs)
    check_state(env, info["state"])

    root_local = env._root_pos_local()
    yaw = env._yaw()

    xy_err = torch.norm(root_local - env.world.start_pos, dim=-1)
    yaw_err = torch.atan2(
        torch.sin(yaw - env.world.start_yaw),
        torch.cos(yaw - env.world.start_yaw),
    ).abs()

    assert xy_err.max().item() < 5e-4, f"reset xy 未对齐，max_err={xy_err.max().item():.8f}"
    assert yaw_err.max().item() < 5e-4, f"reset yaw 未对齐，max_err={yaw_err.max().item():.8f}"

    team = env.world.compute_team_terms(root_local, yaw)
    dist_err = torch.abs(team["center_goal_dist"] - env.last_center_goal_dist)

    assert dist_err.max().item() < 1e-4, "reset 后 last_center_goal_dist 未同步"

    print_stats("reset xy error", xy_err)
    print_stats("reset yaw error", yaw_err)
    print_stats("center-to-goal distance", team["center_goal_dist"])
    print_ok("reset 对齐正常")


def test_action_model(env: DiffDriveTask4Env) -> None:
    heading("[测试 5] Sim2Real action delay / deadband / EMA / wheel target 白盒检测")

    env.reset(options={"stage": 0})

    a = torch.zeros((env.num_envs, env.num_agents, 2), dtype=torch.float32, device=env.device)
    a[..., 0] = 1.0
    a[..., 1] = -1.0

    reset_action_dr_to_nominal(env)

    env.world.action_delay_frames[:] = 2
    env.world.action_deadband[:] = 0.0
    env.world.action_ema_alpha[:] = 1.0

    out1 = env._apply_action_model(a)
    out2 = env._apply_action_model(a)
    out3 = env._apply_action_model(a)

    assert out1.abs().max().item() < 1e-6
    assert out2.abs().max().item() < 1e-6
    assert torch.allclose(out3, a, atol=1e-5)

    reset_action_dr_to_nominal(env)
    env.world.action_deadband[:] = 0.2
    env.world.action_ema_alpha[:] = 1.0

    small = torch.full_like(a, 0.10)
    medium = torch.full_like(a, 0.60)

    small_out = env._apply_action_model(small)
    medium_out = env._apply_action_model(medium)

    assert small_out.abs().max().item() < 1e-6
    assert torch.allclose(medium_out, torch.full_like(medium_out, 0.50), atol=1e-5)

    reset_action_dr_to_nominal(env)
    env.world.action_ema_alpha[:] = 0.5
    ema_out = env._apply_action_model(a)
    assert torch.allclose(ema_out, a * 0.5, atol=1e-5)

    reset_action_dr_to_nominal(env)
    forward = torch.zeros_like(a)
    forward[..., 0] = 1.0
    forward_wheel = env._actions_to_wheel_targets(forward)

    reverse = torch.zeros_like(a)
    reverse[..., 0] = -1.0
    reverse_wheel = env._actions_to_wheel_targets(reverse)

    turn = torch.zeros_like(a)
    turn[..., 1] = 1.0
    turn_wheel = env._actions_to_wheel_targets(turn)

    assert torch.isfinite(forward_wheel).all().item()
    assert torch.isfinite(reverse_wheel).all().item()
    assert torch.isfinite(turn_wheel).all().item()
    assert forward_wheel.mean().item() > 0.0, "forward wheel target 应为正"
    assert reverse_wheel.mean().item() < 0.0, "reverse wheel target 应为负"
    assert (turn_wheel[..., 1] - turn_wheel[..., 0]).mean().item() > 0.0, "positive yaw 应右轮大于左轮"

    print_ok("action delay 生效")
    print_ok("action deadband 生效")
    print_ok("action EMA 生效")
    print_ok("wheel target 转换正常")


def test_obs_slices(env: DiffDriveTask4Env) -> None:
    heading("[测试 6] actor observation 切片 / lidar / risk / one-hot 范围检测")

    obs, info = env.reset(options={"stage": 4})
    check_obs(env, obs)
    check_state(env, info["state"])

    single_dim = env.cfg.single_actor_obs_dim
    last_frame = obs[:, :, -single_dim:]
    s = get_single_obs_slices(env.cfg)

    parts = {
        "vel_obs": last_frame[:, :, s["vel_obs"]],
        "wheel_obs": last_frame[:, :, s["wheel_obs"]],
        "goal_obs": last_frame[:, :, s["goal_obs"]],
        "slot_obs": last_frame[:, :, s["slot_obs"]],
        "team_heading": last_frame[:, :, s["team_heading"]],
        "formation": last_frame[:, :, s["formation"]],
        "agent_id": last_frame[:, :, s["agent_id"]],
        "teammate_pos": last_frame[:, :, s["teammate_pos"]],
        "teammate_vel": last_frame[:, :, s["teammate_vel"]],
        "applied_action": last_frame[:, :, s["applied_action"]],
        "action_delta": last_frame[:, :, s["action_delta"]],
        "progress_ema": last_frame[:, :, s["progress_ema"]],
        "center_speed": last_frame[:, :, s["center_speed"]],
        "lidar": last_frame[:, :, s["lidar"]],
        "lidar_delta": last_frame[:, :, s["lidar_delta"]],
        "risk": last_frame[:, :, s["risk"]],
    }

    for name, x in parts.items():
        assert_finite_tensor(name, x)

    check_shape("lidar", parts["lidar"], (env.num_envs, env.num_agents, env.cfg.world_cfg.lidar_pool_bins))
    check_shape("lidar_delta", parts["lidar_delta"], (env.num_envs, env.num_agents, env.cfg.world_cfg.lidar_pool_bins))
    check_shape("risk", parts["risk"], (env.num_envs, env.num_agents, 16))

    assert parts["lidar"].min().item() >= 0.0
    assert parts["lidar"].max().item() <= 1.0 + 1e-5
    assert parts["lidar_delta"].min().item() >= -1.0 - 1e-5
    assert parts["lidar_delta"].max().item() <= 1.0 + 1e-5
    assert parts["risk"].min().item() >= 0.0
    assert parts["risk"].max().item() <= 1.0 + 1e-5
    assert parts["applied_action"].abs().max().item() <= 1.0 + 1e-5

    agent_id_sum = parts["agent_id"].sum(dim=-1)
    formation_sum = parts["formation"][:, :, :3].sum(dim=-1)

    assert torch.allclose(agent_id_sum, torch.ones_like(agent_id_sum), atol=1e-5)
    assert torch.allclose(formation_sum, torch.ones_like(formation_sum), atol=1e-5)

    print_ok(f"actor obs shape = {tuple(obs.shape)}")
    print_ok(f"last frame shape = {tuple(last_frame.shape)}")
    print_ok(f"lidar range = {parts['lidar'].min().item():.6f} ~ {parts['lidar'].max().item():.6f}")
    print_ok(f"risk range = {parts['risk'].min().item():.6f} ~ {parts['risk'].max().item():.6f}")
    print_ok("observation 切片正常")


def test_step_return_structure(env: DiffDriveTask4Env) -> None:
    heading("[测试 7] 向量化 step 返回结构与 info 字典检测")

    obs, info = env.reset(options={"stage": 4})
    check_obs(env, obs)
    check_state(env, info["state"])

    actions = torch.rand((env.num_envs, env.num_agents, env.num_actions), device=env.device) * 0.4 - 0.2
    obs, reward, terminated, truncated, info = env.step(actions)

    check_obs(env, obs)
    check_state(env, info["state"])
    check_shape("reward", reward, (env.num_envs, env.num_agents))
    check_shape("terminated", terminated, (env.num_envs,))
    check_shape("truncated", truncated, (env.num_envs,))
    assert_finite_tensor("reward", reward)

    for group in ["reward_components", "events", "telemetry", "world", "debug"]:
        assert group in info, f"info 缺少分组: {group}"

    required_reward_keys = [
        "R_Team_Progress",
        "R_Center_Speed",
        "R_Team_Heading",
        "P_Formation_Mean",
        "P_Formation_Agent",
        "P_Team_Spread",
        "P_Speed_Sync",
        "R_Gate_Pass",
        "R_Front_Clearance",
        "P_Obstacle_Risk",
        "P_Gate_Risk",
        "P_Boundary_Risk",
        "P_Pair_Risk",
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
        "Agent_Crash_Rate",
        "Out_Of_Bounds_Rate",
        "Obstacle_Collision_Rate",
        "Gate_Collision_Rate",
        "Pair_Collision_Rate",
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
        "Center_Goal_Dist",
        "Progress",
        "Goal_Aligned_Center_Speed",
        "Mean_Slot_Error",
        "Min_Pair_Dist",
        "Team_Spread",
        "Gate_Active",
        "Near_Gate",
        "Passed_Gate",
        "Lidar_Min",
        "Risk_Front",
        "Risk_Obstacle",
        "Risk_Gate",
        "Risk_Pair",
        "Action_Delay",
        "Action_Deadband",
        "Motor_Strength",
        "Wheel_Radius_Scale",
    ]
    for key in required_tel_keys:
        assert key in info["telemetry"], f"telemetry 缺少 {key}"

    print_ok(f"reward mean = {reward.mean().item():+.6f}")
    print_ok(f"reward min/max = {reward.min().item():+.6f} / {reward.max().item():+.6f}")
    print_ok(f"terminated count = {terminated.sum().item()}")
    print_ok(f"truncated count = {truncated.sum().item()}")
    print_ok("step 返回结构正常")


def test_action_direction(env: DiffDriveTask4Env) -> None:
    heading("[测试 8] 动作方向测试：全队前进 / 后退 / 原地转向")

    steps = 40

    delta_forward, yaw_forward, *_ = run_fixed_action(env, torch.tensor([1.0, 0.0]), steps=steps)
    forward_x = delta_forward[:, 0]

    delta_backward, yaw_backward, *_ = run_fixed_action(env, torch.tensor([-1.0, 0.0]), steps=steps)
    backward_x = delta_backward[:, 0]

    delta_turn, yaw_turn, *_ = run_fixed_action(env, torch.tensor([0.0, 1.0]), steps=steps)
    turn_yaw_abs = yaw_turn.abs().mean(dim=1)

    print_stats("forward center delta x", forward_x)
    print_stats("backward center delta x", backward_x)
    print_stats("turn yaw abs mean", turn_yaw_abs)

    forward_ok = forward_x.mean().item() > 0.03
    backward_ok = backward_x.mean().item() < -0.01
    turn_ok = turn_yaw_abs.mean().item() > 0.03

    if args_cli.strict_action_test:
        assert forward_ok, "action=[1,0] 没有让全队明显前进。请检查 wheel signs。"
        assert backward_ok, "action=[-1,0] 没有让全队明显后退。请检查 reverse_speed_fraction / wheel signs。"
        assert turn_ok, "action=[0,1] 没有让机器人明显转向。请检查 wheel target 转换。"
    else:
        if not forward_ok:
            print_warn("action=[1,0] 没有明显前进。后续若训练异常，请检查 wheel signs。")
        if not backward_ok:
            print_warn("action=[-1,0] 没有明显后退。后退幅度小可能来自 reverse_speed_fraction。")
        if not turn_ok:
            print_warn("action=[0,1] 没有明显转向。后续若训练异常，请检查 wheel target。")

    print_ok("动作方向测试完成")


def test_success_event(env: DiffDriveTask4Env) -> None:
    heading("[测试 9] 手动触发 stable success 事件检测")

    env.reset(options={"stage": 0})

    slots = env.world.compute_formation_slots(
        center_xy=env.world.goal_pos,
        heading=env.world.goal_yaw,
        formation_type=env.world.formation_type,
        scale=env.world.formation_scale,
    )
    yaw = env.world.goal_yaw.unsqueeze(-1).expand(env.num_envs, env.num_agents)

    force_robot_root_poses(env, slots, yaw, zero_vel=True)

    center = env._root_pos_local().mean(dim=1)
    env.last_center_goal_dist[:] = torch.norm(env.world.goal_pos - center, dim=-1)
    env.success_hold_counter[:] = int(env.cfg.world_cfg.success_hold_steps) - 1

    reward, terminated, truncated, info = env._compute_rewards_and_dones(
        pre_center_goal_dist=env.last_center_goal_dist.clone()
    )

    assert terminated.float().mean().item() > 0.99
    assert info["events"]["Success_Rate"] > 0.99
    assert info["events"]["Crash_Rate"] < 1e-6
    assert truncated.float().mean().item() < 1e-6

    print_ok(f"Success_Rate = {info['events']['Success_Rate']:.6f}")
    print_ok(f"Success_Candidate_Rate = {info['events']['Success_Candidate_Rate']:.6f}")
    print_ok(f"Event reward mean = {info['reward_components']['Event']:.6f}")


def test_crash_events(env: DiffDriveTask4Env) -> None:
    heading("[测试 10] out_of_bounds / obstacle / gate / pair collision 事件检测")

    zero_yaw = torch.zeros((env.num_envs, env.num_agents), dtype=torch.float32, device=env.device)

    # out of bounds
    env.reset(options={"stage": 0})
    root = env.world.start_pos.clone()
    root[:, :, 0] = float(env.cfg.world_cfg.x_max) + 0.5
    root[:, :, 1] = 0.0
    force_robot_root_poses(env, root, zero_yaw, zero_vel=True)
    reward, terminated, truncated, info_oob = env._compute_rewards_and_dones(env.last_center_goal_dist.clone())
    assert terminated.float().mean().item() > 0.99
    assert info_oob["events"]["Out_Of_Bounds_Rate"] > 0.99
    assert info_oob["events"]["Crash_Rate"] > 0.99

    # obstacle collision
    env.reset(options={"stage": 2})
    root = env.world.start_pos.clone()
    root[:, 0, :] = env.world.obstacle_pos[:, 0, :]
    force_robot_root_poses(env, root, zero_yaw, zero_vel=True)
    reward, terminated, truncated, info_obs = env._compute_rewards_and_dones(env.last_center_goal_dist.clone())
    assert terminated.float().mean().item() > 0.99
    assert info_obs["events"]["Obstacle_Collision_Rate"] > 0.20
    assert info_obs["events"]["Crash_Rate"] > 0.99

    # gate collision
    env.reset(options={"stage": 3})
    root = env.world.start_pos.clone()
    root[:, 0, 0] = env.world.gate_x
    root[:, 0, 1] = float(env.cfg.world_cfg.gate_top_center_y)
    force_robot_root_poses(env, root, zero_yaw, zero_vel=True)
    reward, terminated, truncated, info_gate = env._compute_rewards_and_dones(env.last_center_goal_dist.clone())
    assert terminated.float().mean().item() > 0.99
    assert info_gate["events"]["Gate_Collision_Rate"] > 0.20
    assert info_gate["events"]["Crash_Rate"] > 0.99

    # pair collision
    env.reset(options={"stage": 0})
    root = env.world.start_pos.clone()
    root[:, 1, :] = root[:, 0, :]
    force_robot_root_poses(env, root, zero_yaw, zero_vel=True)
    reward, terminated, truncated, info_pair = env._compute_rewards_and_dones(env.last_center_goal_dist.clone())
    assert terminated.float().mean().item() > 0.99
    assert info_pair["events"]["Pair_Collision_Rate"] > 0.99
    assert info_pair["events"]["Crash_Rate"] > 0.99

    print_ok(f"Out_Of_Bounds_Rate = {info_oob['events']['Out_Of_Bounds_Rate']:.6f}")
    print_ok(f"Obstacle_Collision_Rate = {info_obs['events']['Obstacle_Collision_Rate']:.6f}")
    print_ok(f"Gate_Collision_Rate = {info_gate['events']['Gate_Collision_Rate']:.6f}")
    print_ok(f"Pair_Collision_Rate = {info_pair['events']['Pair_Collision_Rate']:.6f}")
    print_ok("crash 事件检测正常")


def test_timeout_event(env: DiffDriveTask4Env) -> None:
    heading("[测试 11] timeout truncated 事件检测")

    env.reset(options={"stage": 0})
    env.episode_steps[:] = int(env.cfg.max_episode_length)

    reward, terminated, truncated, info = env._compute_rewards_and_dones(
        pre_center_goal_dist=env.last_center_goal_dist.clone()
    )

    assert truncated.float().mean().item() > 0.99
    assert terminated.float().mean().item() < 1e-6

    print_ok(f"Timeout_Rate = {info['events']['Timeout_Rate']:.6f}")
    print_ok(f"terminated mean = {terminated.float().mean().item():.6f}")
    print_ok(f"truncated mean = {truncated.float().mean().item():.6f}")


def test_reward_direction(env: DiffDriveTask4Env) -> None:
    heading("[测试 12] progress reward 正负方向检测")

    env.reset(options={"stage": 0})

    env.world.goal_pos[:] = torch.tensor([4.0, 0.0], dtype=torch.float32, device=env.device)
    env.world.goal_yaw[:] = 0.0
    env.world.formation_type[:] = 0
    env.world.formation_scale[:] = 1.0

    center_fwd = torch.tensor([[0.30, 0.0]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    center_back = torch.tensor([[-0.30, 0.0]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)

    yaw0 = torch.zeros((env.num_envs, env.num_agents), dtype=torch.float32, device=env.device)
    pre_dist = torch.full((env.num_envs,), 4.0, dtype=torch.float32, device=env.device)

    slots_fwd = env.world.compute_formation_slots(
        center_xy=center_fwd,
        heading=torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
        formation_type=env.world.formation_type,
        scale=env.world.formation_scale,
    )
    force_robot_root_poses(env, slots_fwd, yaw0, zero_vel=True)
    reward_fwd, _, _, info_fwd = env._compute_rewards_and_dones(pre_center_goal_dist=pre_dist)

    env.reset(options={"stage": 0})
    env.world.goal_pos[:] = torch.tensor([4.0, 0.0], dtype=torch.float32, device=env.device)
    env.world.goal_yaw[:] = 0.0
    env.world.formation_type[:] = 0
    env.world.formation_scale[:] = 1.0

    slots_back = env.world.compute_formation_slots(
        center_xy=center_back,
        heading=torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
        formation_type=env.world.formation_type,
        scale=env.world.formation_scale,
    )
    force_robot_root_poses(env, slots_back, yaw0, zero_vel=True)
    reward_back, _, _, info_back = env._compute_rewards_and_dones(pre_center_goal_dist=pre_dist)

    assert info_fwd["telemetry"]["Progress"] > 0.0
    assert info_back["telemetry"]["Progress"] < 0.0
    assert reward_fwd.mean().item() > reward_back.mean().item()

    print_ok(f"forward progress = {info_fwd['telemetry']['Progress']:+.6f}, reward = {reward_fwd.mean().item():+.6f}")
    print_ok(f"backward progress = {info_back['telemetry']['Progress']:+.6f}, reward = {reward_back.mean().item():+.6f}")
    print_ok("progress reward 方向正确")


def random_rollout(env: DiffDriveTask4Env) -> None:
    heading(f"[测试 13] 随机策略运行 {args_cli.steps} 步，收集奖励组件 / 事件 / 遥测")

    obs, info = env.reset(options={"stage": 4})
    check_obs(env, obs)
    check_state(env, info["state"])

    records: List[Dict[str, float]] = []

    total_success = 0
    total_crash = 0
    total_timeout = 0

    start_time = time.time()

    for step in range(int(args_cli.steps)):
        actions = torch.rand((env.num_envs, env.num_agents, env.num_actions), device=env.device) * 2.0 - 1.0
        obs, reward, terminated, truncated, info = env.step(actions)

        check_obs(env, obs)
        check_state(env, info["state"])
        assert_finite_tensor("reward rollout", reward)

        if (step + 1) % max(int(args_cli.collect_interval), 1) == 0 or (step + 1) == int(args_cli.steps):
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
                f"GoalDist={tel.get('Center_Goal_Dist', 0.0):.2f} | "
                f"Progress={tel.get('Progress', 0.0):+.4f} | "
                f"GoalV={tel.get('Goal_Aligned_Center_Speed', 0.0):+.3f} | "
                f"SlotErr={tel.get('Mean_Slot_Error', 0.0):.3f} | "
                f"MinPair={tel.get('Min_Pair_Dist', 0.0):.3f} | "
                f"LidarMin={tel.get('Lidar_Min', 0.0):.3f} | "
                f"RiskObs={tel.get('Risk_Obstacle', 0.0):.3f} | "
                f"RiskGate={tel.get('Risk_Gate', 0.0):.3f} | "
                f"RiskPair={tel.get('Risk_Pair', 0.0):.3f} | "
                f"Succ={ev.get('Success_Rate', 0.0):.4f} | "
                f"Crash={ev.get('Crash_Rate', 0.0):.4f} | "
                f"Timeout={ev.get('Timeout_Rate', 0.0):.4f} | "
                f"R_Prog={rew.get('R_Team_Progress', 0.0):+.3f} | "
                f"P_Form={rew.get('P_Formation_Mean', 0.0):+.3f}",
                flush=True,
            )

        total_success += int(info["events"]["Success_Rate"] * env.num_envs)
        total_crash += int(info["events"]["Crash_Rate"] * env.num_envs)
        total_timeout += int(info["events"]["Timeout_Rate"] * env.num_envs)

        assert torch.isfinite(env.raw_actions).all().item()
        assert torch.isfinite(env.applied_actions).all().item()
        assert torch.isfinite(env.last_center_goal_dist).all().item()
        assert torch.isfinite(env.progress_ema).all().item()
        assert torch.isfinite(env.world.start_pos).all().item()
        assert torch.isfinite(env.world.goal_pos).all().item()
        assert torch.isfinite(env.world.last_lidar).all().item()
        assert torch.isfinite(env.world.last_risk_features).all().item()

    elapsed = time.time() - start_time
    fps = int(args_cli.steps) * env.num_envs / max(elapsed, 1e-6)

    print_ok(f"随机策略长跑完成: {args_cli.steps} steps, {int(args_cli.steps) * env.num_envs:,} env transitions")
    print_ok(f"吞吐约: {fps:,.2f} env steps/s")
    print_ok(f"累计 success approx: {total_success:,}")
    print_ok(f"累计 crash approx: {total_crash:,}")
    print_ok(f"累计 timeout approx: {total_timeout:,}")

    heading("[测试 14] 奖励组件 / 事件 / 遥测统计报告")
    print_summary_table(summarize_records(records))

    print("Diff-Drive UGV / Jetbot Task4 training pre-check guide:")
    print("1. actor obs 必须为 [N, 4, 624]。")
    print("2. critic state 必须为 [N, 96]。")
    print("3. action 必须为 [N, 4, 2]。")
    print("4. reset 后 4 车 root local xy / yaw 必须对齐 world.start_pos / start_yaw。")
    print("5. success / crash / timeout 必须能被手动触发。")
    print("6. 随机策略下 crash 非零是正常的，但不能出现 NaN/Inf。")
    print("7. 训练重点看 Center_Goal_Dist、Progress、Mean_Slot_Error、Min_Pair_Dist、Success_Rate、Crash_Rate。")


# ======================================================================
# Main
# ======================================================================

def run_tests() -> None:
    heading("🚀 Diff-Drive UGV / Jetbot Task4 Multi-UGV Formation Escort Env 全量测试启动")

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

    cfg = Task4Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.seed = int(args_cli.seed)
    cfg.curriculum_stage = 0
    cfg.print_debug_info = bool(args_cli.print_names)

    # Env-level test intentionally keeps physical asset teleport enabled.
    # If PhysX GPU illegal-memory-access appears, rerun with fewer envs first.
    cfg.world_cfg.enable_physical_asset_teleport = True

    cfg.validate()

    env: DiffDriveTask4Env | None = None

    try:
        env = check_env_init(cfg)

        test_reset_step_basic(env)
        test_reset_alignment(env)
        test_action_model(env)
        test_obs_slices(env)
        test_step_return_structure(env)
        test_action_direction(env)
        test_success_event(env)
        test_crash_events(env)
        test_timeout_event(env)
        test_reward_direction(env)
        random_rollout(env)

        heading("✅ Diff-Drive UGV / Jetbot Task4 环境测试全部通过")

    except Exception as exc:
        print("\n❌ Diff-Drive UGV / Jetbot Task4 环境测试失败：")
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

from __future__ import annotations

import argparse
import math
import os
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
    """Return CoreNav-v1 single-frame slices.

    CoreNav-v1 fixed layout:
        0      goal_dist_norm
        1      goal_x_body_norm
        2      goal_y_body_norm
        3      sin_heading_error
        4      cos_heading_error
        5      body_vx
        6      body_vy
        7      body_wz
        8      target_speed_norm
        9      last_forward_throttle
        10     last_turn_command
        11     action_delta_forward
        12     action_delta_turn
        13     progress_ema
    """

    slices = {}

    idx = 0
    slices["goal_dist_norm"] = slice(idx, idx + 1); idx += 1
    slices["goal_xy_body"] = slice(idx, idx + 2); idx += 2
    slices["heading"] = slice(idx, idx + 2); idx += 2
    slices["body_vel"] = slice(idx, idx + 3); idx += 3
    slices["target_speed"] = slice(idx, idx + 1); idx += 1
    slices["last_action"] = slice(idx, idx + 2); idx += 2
    slices["action_delta"] = slice(idx, idx + 2); idx += 2
    slices["progress_ema"] = slice(idx, idx + 1); idx += 1

    assert idx == 14, f"CoreNav-v1 单帧 obs slice 总维度错误: {idx} != 14"
    return slices


def run_fixed_action(env: DiffDriveTask1Env, action_tensor: torch.Tensor, steps: int = 60):
    """Run a deterministic fixed-action motion probe.

    This helper is used only by the white-box action-direction test. It must not
    depend on randomly sampled waypoints because a fast fixed action can reach a
    random near waypoint, trigger partial reset, and make the measured displacement
    meaningless. A partial reset during this low-level probe can also surface CUDA
    assertions asynchronously, which hides the real test intent.

    Therefore the probe installs a far, straight-ahead waypoint and keeps all
    waypoint buffers consistent before stepping the simulator.
    """

    obs, _ = env.reset()

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    # Use far straight-ahead waypoints to prevent success/reset during the probe.
    env.waypoints_local[:, :, :] = 0.0
    env.waypoints_local[:, :, 0] = 8.0
    env.current_wp_idx[:] = 0

    force_root_local_pose(
        env,
        env_ids=env_ids,
        local_xy=torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device),
        height=env.cfg.spawn_height,
        yaw=torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
    )

    env.step_counts[:] = 0
    env.episode_return[:] = 0.0
    env.last_distances[:] = env._distance_to_current_waypoint()
    env.last_heading_error_abs[:] = env._heading_error_abs_to_current_waypoint()

    pos0 = env._root_pos_local()[:, :2].clone()
    yaw0 = env._quat_yaw(env.robot.data.root_quat_w).clone()

    action = action_tensor.to(env.device).view(1, 2).repeat(env.num_envs, 1)

    info = {}
    reward = None
    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    truncated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(action)
        if bool((terminated | truncated).any().item()):
            raise AssertionError("fixed-action probe unexpectedly triggered done/reset")

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
    heading("[测试 1] Task1Config 基础配置与 forward-only 转向动作参数检测")

    cfg = Task1Config()
    cfg.validate()

    assert cfg.action_protocol == "forward_throttle_turn_v1"
    assert cfg.obs_protocol == "CoreNav-v1"
    assert cfg.model_protocol == "ModularActor-v1"
    assert cfg.num_actions == 2
    assert cfg.core_single_obs_dim == 14
    assert cfg.task_extra_single_obs_dim == 0
    assert cfg.single_obs_dim == 14
    assert cfg.frame_stack == 3
    assert cfg.num_observations == 42
    assert cfg.num_waypoints >= 1
    assert 0.0 <= cfg.min_forward_action <= cfg.max_forward_action <= 1.0
    assert cfg.heading_gate_full > cfg.heading_gate_min
    assert cfg.target_goal_speed > cfg.min_goal_speed
    assert cfg.hard_stuck_after_steps >= cfg.stuck_after_steps

    print_ok(f"action_protocol = {cfg.action_protocol}")
    print_ok(f"obs_protocol = {cfg.obs_protocol}")
    print_ok(f"model_protocol = {cfg.model_protocol}")
    print_ok(f"num_actions = {cfg.num_actions}")
    print_ok(f"single_obs_dim = {cfg.single_obs_dim}")
    print_ok(f"frame_stack = {cfg.frame_stack}")
    print_ok(f"num_observations = {cfg.num_observations}")
    print_ok(f"num_waypoints = {cfg.num_waypoints}")
    print_ok(f"min_forward_action = {cfg.min_forward_action}")
    print_ok(f"max_forward_action = {cfg.max_forward_action}")
    print_ok(f"turn_scale_norm = {cfg.turn_scale_norm}")
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
    assert env.cfg.action_protocol == "forward_throttle_turn_v1"
    assert env.cfg.obs_protocol == "CoreNav-v1"
    assert env.cfg.model_protocol == "ModularActor-v1"
    assert env.cfg.single_obs_dim == 14
    assert env.num_observations == env.cfg.frame_stack * env.cfg.single_obs_dim
    assert env.num_observations == 42
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
        "P_Negative_Progress",
        "P_Backward",
        "P_Slow",
        "P_No_Progress",
        "P_Spin_In_Place",
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
        "Hard_Stuck_Rate",
        "Done_Rate",
        "Episode_Finish_Rate",
        "Episode_Timeout_Rate",
        "Episode_Hard_Stuck_Rate",
        "Episode_Done_Count",
        "Recent_Finish_Rate",
        "Recent_Timeout_Rate",
    ]
    for key in required_event_keys:
        assert key in info["events"], f"events 缺少 {key}"

    required_tel_keys = [
        "Distance_To_Waypoint",
        "Progress",
        "Progress_EMA",
        "Heading_Error",
        "Heading_Gate",
        "Speed_Gate",
        "Distance_Gate",
        "Goal_Aligned_Speed",
        "Body_Vx",
        "Body_Wz",
        "Raw_Action_Left",
        "Raw_Action_Right",
        "Exec_Action_Left",
        "Exec_Action_Right",
        "Wheel_Target_Left",
        "Wheel_Target_Right",
        "Backward_Ratio",
        "Slow_Ratio",
        "No_Progress_Ratio",
        "Waypoint_Index",
        "Stuck_Ratio",
    ]
    for key in required_tel_keys:
        assert key in info["telemetry"], f"telemetry 缺少 {key}"

    # In the final Task1 action semantics, the policy outputs [forward_throttle, turn].
    # The chassis forward command must be non-negative; individual wheel commands
    # may be negative during pivot turns.
    assert info["debug"]["Forward_Command_Min"] >= float(env.cfg.min_forward_action) - 1e-5
    assert info["debug"]["Forward_Command_Max"] <= float(env.cfg.max_forward_action) + 1e-5

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

    wp = env.waypoints_local[:, : int(env.cfg.num_waypoints), :]
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
    print_stats("active waypoint radius", wp_norm)
    print_stats("first waypoint dist", first_dist)
    print_ok("reset 对齐与 waypoint 采样正常")


def test_obs_slices(env: DiffDriveTask1Env):
    heading("[测试 5] CoreNav-v1 observation 切片与数值范围检测")

    obs, _ = env.reset()
    check_obs(env, obs)

    single_dim = env.cfg.single_obs_dim
    last_frame = obs[:, -single_dim:]
    s = get_single_obs_slices()

    goal_dist_norm = last_frame[:, s["goal_dist_norm"]]
    goal_xy_body = last_frame[:, s["goal_xy_body"]]
    heading_pair = last_frame[:, s["heading"]]
    body_vel = last_frame[:, s["body_vel"]]
    target_speed = last_frame[:, s["target_speed"]]
    last_action = last_frame[:, s["last_action"]]
    action_delta = last_frame[:, s["action_delta"]]
    progress_ema = last_frame[:, s["progress_ema"]]

    for name, x in [
        ("goal_dist_norm", goal_dist_norm),
        ("goal_xy_body", goal_xy_body),
        ("heading", heading_pair),
        ("body_vel", body_vel),
        ("target_speed", target_speed),
        ("last_action", last_action),
        ("action_delta", action_delta),
        ("progress_ema", progress_ema),
    ]:
        assert_finite_tensor(name, x)

    assert heading_pair.abs().max().item() <= 1.0 + 1e-5
    heading_norm = torch.norm(heading_pair, dim=-1)
    assert torch.max(torch.abs(heading_norm - 1.0)).item() < 1e-4, "sin/cos heading 不满足单位圆"

    assert target_speed.min().item() >= 0.0
    assert last_action.min().item() >= -1.0 - 1e-5
    assert last_action.max().item() <= 1.0 + 1e-5
    assert action_delta.min().item() >= -2.0 - 1e-5
    assert action_delta.max().item() <= 2.0 + 1e-5

    # Actor obs must not contain old left/right wheel velocity or waypoint-index fields.
    assert last_frame.shape[-1] == 14

    print_ok(f"obs shape = {tuple(obs.shape)}")
    print_ok(f"last_frame range = {last_frame.min().item():+.6f} ~ {last_frame.max().item():+.6f}")
    print_ok(f"goal_dist_norm mean = {goal_dist_norm.mean().item():.6f}")
    print_ok(f"target_speed_norm mean = {target_speed.mean().item():.6f}")
    print_ok("CoreNav-v1 字段顺序与数值范围正常")


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


def test_nonnegative_action_mapping(env: DiffDriveTask1Env):
    heading("[测试 7] forward-only + turn 动作映射白盒测试")

    raw = torch.tensor(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=torch.float32,
        device=env.device,
    )

    wheel_norm = env._map_raw_actions_to_exec(raw)

    # Compute expected forward / turn command locally.  Do not read the env's
    # diagnostic buffers here: this test uses a small custom action batch while
    # the actual environment buffers are num_envs-sized.
    speed_factor_linear = 0.5 * (torch.clamp(raw[:, 0], -1.0, 1.0) + 1.0)
    speed_factor = torch.pow(
        torch.clamp(speed_factor_linear, 0.0, 1.0),
        float(getattr(env.cfg, "forward_curve_power", 1.0)),
    )
    forward_norm = float(env.cfg.min_forward_action) + (
        float(env.cfg.max_forward_action) - float(env.cfg.min_forward_action)
    ) * speed_factor
    forward_norm = torch.clamp(forward_norm, 0.0, 1.0)
    turn_norm = torch.clamp(raw[:, 1] * float(env.cfg.turn_scale_norm), -1.0, 1.0)

    assert_finite_tensor("wheel_norm", wheel_norm)
    assert_finite_tensor("forward_norm", forward_norm)
    assert_finite_tensor("turn_norm", turn_norm)
    assert tuple(env.forward_command_norm.shape) == (env.num_envs,), "forward_command_norm buffer shape 被测试污染"
    assert tuple(env.wheel_command_norm.shape) == (env.num_envs, 2), "wheel_command_norm buffer shape 被测试污染"

    assert forward_norm.min().item() >= float(env.cfg.min_forward_action) - 1e-6, (
        "forward command 必须保持非负最小前进量"
    )
    assert forward_norm.max().item() <= float(env.cfg.max_forward_action) + 1e-6, (
        "forward command 不能超过最大前进量"
    )
    assert wheel_norm.min().item() >= -1.0 - 1e-6
    assert wheel_norm.max().item() <= 1.0 + 1e-6

    # turn = 0 时，左右轮命令均等，轮速均值必须等于非负车体前进命令。
    no_turn_rows = torch.tensor([0, 1, 2], dtype=torch.long, device=env.device)
    no_turn_mean = 0.5 * (wheel_norm[no_turn_rows, 0] + wheel_norm[no_turn_rows, 1])
    assert torch.allclose(no_turn_mean, forward_norm[no_turn_rows], atol=1e-5), (
        "无转向动作时，左右轮均值应等于 forward command"
    )
    assert torch.allclose(wheel_norm[no_turn_rows, 0], wheel_norm[no_turn_rows, 1], atol=1e-5), (
        "无转向动作时，左右轮命令应相等"
    )

    # 最小前进 + 强转向时允许一正一负轮速，用于原地/小半径掉头。
    assert wheel_norm[3, 0].item() < 0.0 and wheel_norm[3, 1].item() > 0.0, (
        "action=[-1,+1] 应产生左轮负、右轮正的左转/掉头能力"
    )
    assert wheel_norm[4, 0].item() > 0.0 and wheel_norm[4, 1].item() < 0.0, (
        "action=[-1,-1] 应产生左轮正、右轮负的右转/掉头能力"
    )

    # 高前进 + 强转向会发生单侧轮速饱和，此时轮速均值不再严格等于 forward command。
    # 这是物理限幅的正常结果，测试只检查转向方向和数值范围。
    assert wheel_norm[5, 0].item() < wheel_norm[5, 1].item(), "action=[+1,+1] 转向方向应正确"
    assert wheel_norm[6, 0].item() > wheel_norm[6, 1].item(), "action=[+1,-1] 转向方向应正确"

    print_stats("forward command norm", forward_norm)
    print_stats("turn command norm", turn_norm)
    print_stats("wheel command norm", wheel_norm)
    print_ok("原始动作已映射为非负车体前进命令 + 差速转向命令")


def test_action_direction(env: DiffDriveTask1Env):
    heading("[测试 8] 动作方向白盒测试：前进 / 最小前进不倒车 / 原地转向")

    steps = 80

    delta_forward, yaw_forward, obs, reward, terminated, truncated, info_forward = run_fixed_action(
        env,
        torch.tensor([1.0, 0.0]),
        steps=steps,
    )
    forward_x = delta_forward[:, 0]

    delta_slow, yaw_slow, *_ = run_fixed_action(
        env,
        torch.tensor([-1.0, 0.0]),
        steps=steps,
    )
    slow_x = delta_slow[:, 0]

    delta_turn_left, yaw_turn_left, *_ = run_fixed_action(
        env,
        torch.tensor([-1.0, 1.0]),
        steps=steps,
    )
    delta_turn_right, yaw_turn_right, *_ = run_fixed_action(
        env,
        torch.tensor([-1.0, -1.0]),
        steps=steps,
    )

    print_stats("forward delta x", forward_x)
    print_stats("min-forward delta x", slow_x)
    print_stats("turn-left delta yaw", yaw_turn_left)
    print_stats("turn-right delta yaw", yaw_turn_right)

    forward_ok = forward_x.mean().item() > 0.05
    slow_not_backward_ok = slow_x.mean().item() > -0.02
    turn_ok = yaw_turn_left.abs().mean().item() > 0.10 and yaw_turn_right.abs().mean().item() > 0.10
    turn_opposite_ok = (yaw_turn_left.mean() * yaw_turn_right.mean()).item() < 0.0

    assert info_forward["debug"]["Forward_Command_Min"] >= float(env.cfg.min_forward_action) - 1e-5

    if args_cli.strict_action_test:
        assert forward_ok, "action=[1,0] 没有让车明显前进。请检查 wheel signs。"
        assert slow_not_backward_ok, "action=[-1,0] 不应让车明显倒退。"
        assert turn_ok, "action=[-1,±1] 没有让车明显原地/小半径转向。"
        assert turn_opposite_ok, "左右转动作没有产生相反 yaw 方向。"
    else:
        if not forward_ok:
            print_warn("action=[1,0] 没有明显前进。若训练变慢，请检查 wheel signs。")
        if not slow_not_backward_ok:
            print_warn("action=[-1,0] 出现明显倒退，请检查 forward-only 映射。")
        if not turn_ok or not turn_opposite_ok:
            print_warn("action=[-1,±1] 转向异常，请检查 turn_scale_norm 或 wheel signs。")

    print_ok("动作方向测试完成")


def test_progress_reward_sign(env: DiffDriveTask1Env):
    heading("[测试 9] progress reward 正负方向检测")

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


def test_no_static_heading_reward_farming(env: DiffDriveTask1Env):
    heading("[测试 10] 原地 / 慢速 / 朝向正确时不能刷正奖励")

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
        zero_vel=True,
    )

    env.step_counts[:] = int(env.cfg.stuck_after_steps) + 5
    env.last_distances[:] = env._distance_to_current_waypoint()
    env.progress_ema[:] = 0.0
    env.stuck_counter[:] = 0
    env.actions[:] = float(env.cfg.min_forward_action)
    env.prev_actions[:] = float(env.cfg.min_forward_action)

    pre = env.last_distances.clone()
    current = env._distance_to_current_waypoint()
    progress = pre - current

    reward, terminated, truncated, info = env._compute_rewards_and_dones(
        pre_distances=pre,
        current_dist=current,
        progress=progress,
    )

    assert info["reward_components"]["R_Progress"] <= 1e-6, "无 progress 时 R_Progress 不应刷正奖励"
    assert info["reward_components"]["P_Slow"] < 0.0, "朝向正确但速度过低时应触发 P_Slow"
    assert info["reward_components"]["Continuous"] <= 0.0, "原地/慢速/未进展时 continuous reward 不应为正"
    assert_finite_tensor("reward static heading", reward)

    print_ok(f"R_Heading = {info['reward_components']['R_Heading']:.6f}")
    print_ok(f"P_Slow = {info['reward_components']['P_Slow']:.6f}")
    print_ok(f"Continuous = {info['reward_components']['Continuous']:.6f}")
    print_ok("原地刷分约束正常")


def test_waypoint_event(env: DiffDriveTask1Env):
    heading("[测试 11] 手动触发普通 waypoint 事件检测")

    env.reset()
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    if int(env.cfg.num_waypoints) < 2:
        print_warn("num_waypoints < 2，跳过普通 waypoint 事件测试")
        return

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
    heading("[测试 12] 手动触发 finish 事件检测")

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
    heading("[测试 13] timeout truncated 事件检测")

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
    heading("[测试 14] stuck penalty / hard stuck 检测")

    env.reset()

    env.step_counts[:] = int(env.cfg.stuck_after_steps) + 5
    env.last_distances[:] = env._distance_to_current_waypoint()
    env.progress_ema[:] = 0.0
    env.stuck_counter[:] = 0
    env.actions[:] = float(env.cfg.min_forward_action)
    env.prev_actions[:] = float(env.cfg.min_forward_action)

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
    heading(f"[测试 15] 随机策略运行 {args_cli.steps} 步，收集奖励组件 / 事件 / 遥测")

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
                f"GoalV={tel.get('Goal_Aligned_Speed', 0.0):+.3f} | "
                f"Fwd={tel.get('Forward_Command_Norm', 0.0):.3f} | "
                f"Turn={tel.get('Turn_Command_Norm', 0.0):+.3f} | "
                f"WheelT={tel.get('Wheel_Target_Left', 0.0):+.2f}/{tel.get('Wheel_Target_Right', 0.0):+.2f} | "
                f"Back={tel.get('Backward_Ratio', 0.0):.3f} | "
                f"Slow={tel.get('Slow_Ratio', 0.0):.3f} | "
                f"Stuck={tel.get('Stuck_Ratio', 0.0):.3f} | "
                f"WPIdx={tel.get('Waypoint_Index', 0.0):.2f} | "
                f"WP={ev.get('Waypoint_Rate', 0.0):.4f} | "
                f"Finish={ev.get('Finish_Rate', 0.0):.4f} | "
                f"Timeout={ev.get('Timeout_Rate', 0.0):.4f} | "
                f"R_Prog={rew.get('R_Progress', 0.0):+.3f} | "
                f"R_Fwd={rew.get('R_Forward', 0.0):+.3f} | "
                f"P_Back={rew.get('P_Backward', 0.0):+.3f} | "
                f"P_Slow={rew.get('P_Slow', 0.0):+.3f}",
                flush=True,
            )

        if (step + 1) % 500 == 0 or (step + 1) == int(args_cli.steps):
            check_obs(env, obs)
            assert_finite_tensor("reward rollout", reward)
            assert torch.isfinite(env.actions).all().item(), "smoothed network actions 出现 NaN/Inf"
            assert torch.isfinite(env.raw_actions).all().item(), "raw_actions 出现 NaN/Inf"
            assert torch.isfinite(env.prev_actions).all().item(), "prev_actions 出现 NaN/Inf"
            assert torch.isfinite(env.forward_command_norm).all().item(), "forward_command_norm 出现 NaN/Inf"
            assert torch.isfinite(env.turn_command_norm).all().item(), "turn_command_norm 出现 NaN/Inf"
            assert torch.isfinite(env.wheel_command_norm).all().item(), "wheel_command_norm 出现 NaN/Inf"
            assert torch.isfinite(env.last_distances).all().item(), "last_distances 出现 NaN/Inf"
            assert torch.isfinite(env.progress_ema).all().item(), "progress_ema 出现 NaN/Inf"
            assert torch.isfinite(env.waypoints_local).all().item(), "waypoints_local 出现 NaN/Inf"

            # 新动作语义：env.actions 是网络动作 [forward_throttle, turn]，允许为负；
            # forward-only 约束只应检查 forward_command_norm，轮速命令允许为负用于差速转向。
            assert env.raw_actions.min().item() >= -1.0 - 1e-5, "raw_actions 低于 -1"
            assert env.raw_actions.max().item() <= 1.0 + 1e-5, "raw_actions 高于 +1"
            assert env.actions.min().item() >= -1.0 - 1e-5, "smoothed network actions 低于 -1"
            assert env.actions.max().item() <= 1.0 + 1e-5, "smoothed network actions 高于 +1"
            assert env.forward_command_norm.min().item() >= float(env.cfg.min_forward_action) - 1e-5, "前进命令低于 min_forward_action"
            assert env.forward_command_norm.max().item() <= float(env.cfg.max_forward_action) + 1e-5, "前进命令超过 max_forward_action"
            assert env.turn_command_norm.min().item() >= -float(env.cfg.turn_scale_norm) - 1e-5, "转向命令低于 -turn_scale_norm"
            assert env.turn_command_norm.max().item() <= float(env.cfg.turn_scale_norm) + 1e-5, "转向命令超过 +turn_scale_norm"
            assert env.wheel_command_norm.min().item() >= -1.0 - 1e-5, "轮速归一化命令低于 -1"
            assert env.wheel_command_norm.max().item() <= 1.0 + 1e-5, "轮速归一化命令超过 +1"

    elapsed = time.time() - start_time
    fps = int(args_cli.steps) * env.num_envs / max(elapsed, 1e-6)

    print_ok(f"随机策略长跑完成: {args_cli.steps} steps, {int(args_cli.steps) * env.num_envs:,} transitions")
    print_ok(f"吞吐约: {fps:,.2f} env steps/s")
    print_ok(f"累计 waypoint approx: {total_waypoints:,}")
    print_ok(f"累计 finish approx: {total_finished:,}")
    print_ok(f"累计 timeout approx: {total_timeout:,}")

    heading("[测试 16] 奖励组件 / 事件 / 遥测统计报告")
    print_summary_table(summarize_records(records))

    print("Diff-Drive UGV / Jetbot Task1 training pre-check guide:")
    print("1. obs 应为 42 维，即 CoreNav-v1 14 维 × 3 帧。")
    print("2. 网络动作仍为 [-1,1]；forward_command_norm 必须在 [min_forward_action, 1]，turn / wheel command 可为负用于差速转向。")
    print("3. Actor obs 不再包含左右轮速度或 waypoint_index，便于 Task1 core_encoder 迁移到 Task2 / Task3。")
    print("3. action=[-1,-1] 不应再导致明显倒车；action=[1,-1] 应能差速转向。")
    print("4. 原地、慢速、倒退、无进展不能靠 R_Heading 刷正奖励。")
    print("5. 普通 waypoint 事件和 finish 事件必须能被手动触发。")
    print("6. 训练时重点看 Progress、Goal_Aligned_Speed、Forward_Command_Norm、Turn_Command_Norm、Wheel_Target、Recent_Finish_Rate、Backward_Ratio。")


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
        test_nonnegative_action_mapping(env)
        test_action_direction(env)
        test_progress_reward_sign(env)
        test_no_static_heading_reward_farming(env)
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
from __future__ import annotations

import argparse
import math
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

from diff_drive_rl.tasks.task2.task2_world import Task2WorldConfig, Task2WorldManager


# ======================================================================
# Args
# ======================================================================

parser = argparse.ArgumentParser(description="Diff-Drive UGV / Jetbot Task2 Analytic World Test")
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--strict-spacing", action="store_true")

args = parser.parse_args()


# ======================================================================
# Utils
# ======================================================================

def heading(title: str) -> None:
    print("\n" + "=" * 140)
    print(title)
    print("=" * 140, flush=True)


def print_ok(msg: str) -> None:
    print(f" ✅ {msg}", flush=True)


def print_warn(msg: str) -> None:
    print(f" ⚠️ {msg}", flush=True)


def assert_finite(name: str, x: torch.Tensor) -> None:
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
        f"{name:<40s} "
        f"mean={s['mean']:+.6f} | min={s['min']:+.6f} | p10={s['p10']:+.6f} | "
        f"p50={s['p50']:+.6f} | p90={s['p90']:+.6f} | max={s['max']:+.6f}",
        flush=True,
    )


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
    print(" " * 52 + "Diff-Drive UGV / Jetbot Task2 Analytic World 统计报告")
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


def random_root_positions(cfg: Task2WorldConfig, num_envs: int, device: str) -> torch.Tensor:
    half = cfg.half_extent - 1.0
    root = torch.empty((num_envs, 2), dtype=torch.float32, device=device)
    root[:, 0] = (torch.rand(num_envs, dtype=torch.float32, device=device) * 2.0 - 1.0) * half
    root[:, 1] = (torch.rand(num_envs, dtype=torch.float32, device=device) * 2.0 - 1.0) * half
    return root


def disable_all_obstacles(world: Task2WorldManager) -> None:
    world.static_mask[:] = False
    world.dynamic_mask[:] = False
    world.static_pos[:] = 0.0
    world.dynamic_pos[:] = 0.0
    world.static_radius[:] = 0.0
    world.dynamic_radius[:] = 0.0
    world.dynamic_vel[:] = 0.0


def combined_obstacles(world: Task2WorldManager) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = torch.cat([world.static_pos, world.dynamic_pos], dim=1)
    radius = torch.cat([world.static_radius, world.dynamic_radius], dim=1)
    mask = torch.cat([world.static_mask, world.dynamic_mask], dim=1)
    return pos, radius, mask


# ======================================================================
# Tests
# ======================================================================

def test_config(world: Task2WorldManager) -> None:
    heading("[测试 0] Task2WorldConfig 基础配置检测")

    cfg = world.cfg
    cfg.validate()

    assert cfg.num_stages == 6
    assert cfg.max_static_obs == 20
    assert cfg.max_dynamic_obs == 5
    assert cfg.num_lidar_rays == 72
    assert cfg.lidar_max_distance > cfg.lidar_min_distance
    assert cfg.half_extent > 0.0

    print_ok(f"num_envs = {world.num_envs}")
    print_ok(f"device = {world.device}")
    print_ok(f"arena_size = {cfg.arena_size}")
    print_ok(f"half_extent = {cfg.half_extent}")
    print_ok(f"max_static_obs = {cfg.max_static_obs}")
    print_ok(f"max_dynamic_obs = {cfg.max_dynamic_obs}")
    print_ok(f"num_lidar_rays = {cfg.num_lidar_rays}")
    print_ok(f"lidar_max_distance = {cfg.lidar_max_distance}")
    print_ok("Task2WorldConfig 基础配置正常")


def test_curriculum_mapping(world: Task2WorldManager) -> None:
    heading("[测试 1] 课程阶段映射检测")

    cfg = world.cfg
    rows = []

    ks = [0.0, 0.04, 0.08, 0.19, 0.20, 0.37, 0.38, 0.59, 0.60, 0.79, 0.80, 1.0]

    for k in ks:
        stage = world.stage_from_progress(k)
        steps = int(k * cfg.curriculum_total_steps)
        stage_from_steps = world.stage_from_global_steps(steps)

        assert stage == stage_from_steps, f"k 和 global_steps 映射不一致: k={k}, stage={stage}, steps_stage={stage_from_steps}"

        rows.append(
            {
                "K": k,
                "Steps": steps,
                "Stage": stage,
                "GoalDist": str(cfg.goal_dist_ranges[stage]),
                "StaticCount": str(cfg.static_count_ranges[stage]),
                "DynamicCount": str(cfg.dynamic_count_ranges[stage]),
                "TargetSpeed": str(cfg.target_speed_ranges[stage]),
            }
        )

    print(f"{'K':>7} | {'Steps':>12} | {'Stage':>5} | {'GoalDist':>12} | {'Static':>10} | {'Dynamic':>10} | {'TargetSpeed':>14}")
    print("-" * 92)
    for row in rows:
        print(
            f"{row['K']:>7.3f} | {row['Steps']:>12,d} | {row['Stage']:>5d} | "
            f"{row['GoalDist']:>12} | {row['StaticCount']:>10} | {row['DynamicCount']:>10} | {row['TargetSpeed']:>14}"
        )

    assert world.stage_from_progress(0.0) == 0
    assert world.stage_from_progress(1.0) == cfg.num_stages - 1

    print_ok("课程阶段映射正常")


def test_stage_reset_sampling(world: Task2WorldManager) -> None:
    heading("[测试 2] 各课程阶段 reset 采样检测")

    cfg = world.cfg
    E = world.num_envs
    env_ids = torch.arange(E, dtype=torch.long, device=world.device)

    stage_ks = [0.0, 0.08, 0.20, 0.38, 0.60, 0.80]
    rows: List[Dict[str, float]] = []

    for expected_stage, k in enumerate(stage_ks):
        global_steps = int(k * cfg.curriculum_total_steps)
        world.reset(env_ids, global_steps=global_steps)

        stage_mean = int(round(world.env_stage.float().mean().item()))
        assert stage_mean == expected_stage, f"stage mean 错误: {stage_mean} != {expected_stage}"

        goal_dist = torch.norm(world.goal_pos - world.start_pos, dim=-1)
        static_count = world.static_mask.float().sum(dim=-1)
        dynamic_count = world.dynamic_mask.float().sum(dim=-1)

        gmin, gmax = cfg.goal_dist_ranges[expected_stage]
        smin, smax = cfg.static_count_ranges[expected_stage]
        dmin, dmax = cfg.dynamic_count_ranges[expected_stage]
        vmin, vmax = cfg.target_speed_ranges[expected_stage]

        assert goal_dist.min().item() >= gmin - 1e-4
        assert goal_dist.max().item() <= gmax + 1e-4

        assert static_count.min().item() >= smin
        assert static_count.max().item() <= smax

        assert dynamic_count.min().item() >= dmin
        assert dynamic_count.max().item() <= dmax

        assert world.env_target_speed.min().item() >= vmin - 1e-4
        assert world.env_target_speed.max().item() <= vmax + 1e-4

        half = cfg.half_extent
        assert world.start_pos.abs().max().item() <= half + 1e-4
        assert world.goal_pos.abs().max().item() <= half + 1e-4

        for name, tensor in [
            ("start_pos", world.start_pos),
            ("goal_pos", world.goal_pos),
            ("static_pos", world.static_pos),
            ("dynamic_pos", world.dynamic_pos),
            ("static_radius", world.static_radius),
            ("dynamic_radius", world.dynamic_radius),
            ("dynamic_vel", world.dynamic_vel),
        ]:
            assert_finite(name, tensor)

        rows.append(
            {
                "K": k,
                "Stage": float(expected_stage),
                "GoalDistMean": goal_dist.mean().item(),
                "GoalDistMin": goal_dist.min().item(),
                "GoalDistMax": goal_dist.max().item(),
                "StaticMean": static_count.mean().item(),
                "StaticMin": static_count.min().item(),
                "StaticMax": static_count.max().item(),
                "DynamicMean": dynamic_count.mean().item(),
                "DynamicMin": dynamic_count.min().item(),
                "DynamicMax": dynamic_count.max().item(),
                "TargetSpeedMean": world.env_target_speed.mean().item(),
            }
        )

    print(f"{'K':>6} | {'Stage':>5} | {'GoalMean':>9} | {'GoalMin':>8} | {'GoalMax':>8} | {'StaticMean':>10} | {'DynamicMean':>11} | {'SpeedMean':>10}")
    print("-" * 92)
    for row in rows:
        print(
            f"{row['K']:>6.2f} | {int(row['Stage']):>5d} | {row['GoalDistMean']:>9.3f} | "
            f"{row['GoalDistMin']:>8.3f} | {row['GoalDistMax']:>8.3f} | "
            f"{row['StaticMean']:>10.3f} | {row['DynamicMean']:>11.3f} | {row['TargetSpeedMean']:>10.3f}"
        )

    print_ok("各课程阶段 reset 采样正常")


def test_obstacle_safety(world: Task2WorldManager) -> None:
    heading("[测试 3] 障碍物安全区 / 边界 / 间距检测")

    cfg = world.cfg
    E = world.num_envs
    env_ids = torch.arange(E, dtype=torch.long, device=world.device)

    world.reset(env_ids, global_steps=int(0.80 * cfg.curriculum_total_steps))

    half = cfg.half_extent

    pos, radius, mask = combined_obstacles(world)

    if mask.any():
        active_pos = pos[mask]
        active_radius = radius[mask]

        assert_finite("active obstacle pos", active_pos)
        assert_finite("active obstacle radius", active_radius)

        assert active_radius.min().item() > 0.0
        assert active_pos.abs().max().item() <= half + 1e-4

        dist_start = torch.norm(pos - world.start_pos[:, None, :], dim=-1)
        dist_goal = torch.norm(pos - world.goal_pos[:, None, :], dim=-1)

        required = cfg.start_goal_safe_radius + cfg.robot_radius + radius

        start_safe = torch.where(mask, dist_start > required - 1e-4, torch.ones_like(mask))
        goal_safe = torch.where(mask, dist_goal > required - 1e-4, torch.ones_like(mask))

        assert start_safe.all().item(), "障碍物压到起点安全区"
        assert goal_safe.all().item(), "障碍物压到终点安全区"

        total = pos.shape[1]
        if total >= 2:
            dist_pair = torch.norm(pos[:, :, None, :] - pos[:, None, :, :], dim=-1)
            rad_pair = radius[:, :, None] + radius[:, None, :]
            active_pair = mask[:, :, None] & mask[:, None, :]

            upper = torch.triu(
                torch.ones((total, total), dtype=torch.bool, device=world.device),
                diagonal=1,
            )
            active_pair = active_pair & upper.unsqueeze(0)

            if active_pair.any():
                min_required = rad_pair + cfg.min_obs_spacing + cfg.robot_radius
                sep_margin = dist_pair - min_required
                active_sep_margin = sep_margin[active_pair]

                min_margin = active_sep_margin.min().item()
                print_stats("obstacle pair sep margin", active_sep_margin)

                if args.strict_spacing:
                    assert min_margin > -1e-4, f"障碍物间距不足: min_margin={min_margin:.6f}"
                elif min_margin < -1e-4:
                    print_warn(
                        f"发现少量障碍物间距不足 min_margin={min_margin:.6f}。"
                        "这通常来自极端 fallback 采样；训练不一定受影响，但后续可加强采样。"
                    )

        print_stats("active obstacle radius", active_radius)

    goal_dist = torch.norm(world.goal_pos - world.start_pos, dim=-1)
    print_stats("goal distance", goal_dist)
    print_ok("障碍物边界和起终点安全区正常")


def test_goal_terms(world: Task2WorldManager) -> None:
    heading("[测试 4] goal terms / body-frame 目标向量检测")

    cfg = world.cfg
    E = world.num_envs
    env_ids = torch.arange(E, dtype=torch.long, device=world.device)

    world.reset(env_ids, global_steps=0)

    world.start_pos[:] = 0.0
    world.goal_pos[:] = torch.tensor([5.0, 0.0], dtype=torch.float32, device=world.device)

    root = torch.zeros((E, 2), dtype=torch.float32, device=world.device)
    yaw0 = torch.zeros(E, dtype=torch.float32, device=world.device)

    terms = world.compute_goal_terms(root, yaw0)

    assert torch.allclose(terms["goal_dist"], torch.full((E,), 5.0, device=world.device), atol=1e-5)
    assert torch.allclose(terms["heading_error"], torch.zeros(E, device=world.device), atol=1e-5)
    assert torch.allclose(terms["heading_cos"], torch.ones(E, device=world.device), atol=1e-5)
    assert torch.allclose(terms["goal_vec_b"][:, 0], torch.full((E,), 5.0, device=world.device), atol=1e-5)
    assert torch.allclose(terms["goal_vec_b"][:, 1], torch.zeros(E, device=world.device), atol=1e-5)

    yaw90 = torch.full((E,), math.pi / 2.0, dtype=torch.float32, device=world.device)
    terms90 = world.compute_goal_terms(root, yaw90)

    assert torch.allclose(terms90["heading_error"], torch.full((E,), -math.pi / 2.0, device=world.device), atol=1e-5)
    assert torch.allclose(terms90["goal_vec_b"][:, 0], torch.zeros(E, device=world.device), atol=1e-5)
    assert torch.allclose(terms90["goal_vec_b"][:, 1], torch.full((E,), -5.0, device=world.device), atol=1e-5)

    print_ok("goal terms 和 body-frame 坐标变换正常")


def test_lidar_manual_front_obstacle(world: Task2WorldManager) -> None:
    heading("[测试 5] 解析 LiDAR：正前方障碍物命中检测")

    cfg = world.cfg
    E = world.num_envs
    env_ids = torch.arange(E, dtype=torch.long, device=world.device)

    world.reset(env_ids, global_steps=0)
    disable_all_obstacles(world)

    root = torch.zeros((E, 2), dtype=torch.float32, device=world.device)
    yaw = torch.zeros(E, dtype=torch.float32, device=world.device)

    world.static_mask[:, 0] = True
    world.static_pos[:, 0, 0] = 3.0
    world.static_pos[:, 0, 1] = 0.0
    world.static_radius[:, 0] = 0.5

    lidar = world.compute_lidar_distances(root, yaw, update_history=True)

    check_shape("lidar", lidar, (E, cfg.num_lidar_rays))
    assert_finite("lidar", lidar)

    front_idx = torch.argmin(torch.abs(world.lidar_angles))
    front_dist = lidar[:, front_idx]

    assert abs(front_dist.mean().item() - 2.5) < 0.10, (
        f"正前方 LiDAR 命中距离异常: mean={front_dist.mean().item():.4f}, expected≈2.5"
    )

    assert lidar.min().item() >= cfg.lidar_min_distance - 1e-6
    assert lidar.max().item() <= cfg.lidar_max_distance + 1e-6

    print_stats("front lidar distance", front_dist)
    print_ok("正前方障碍物 LiDAR 命中正常")


def test_lidar_boundary(world: Task2WorldManager) -> None:
    heading("[测试 6] 解析 LiDAR：边界距离检测")

    cfg = world.cfg
    E = world.num_envs
    env_ids = torch.arange(E, dtype=torch.long, device=world.device)

    world.reset(env_ids, global_steps=0)
    disable_all_obstacles(world)

    half = cfg.half_extent

    root = torch.zeros((E, 2), dtype=torch.float32, device=world.device)
    root[:, 0] = half - 1.0
    root[:, 1] = 0.0

    yaw = torch.zeros(E, dtype=torch.float32, device=world.device)

    lidar = world.compute_lidar_distances(root, yaw, update_history=False)
    front_idx = torch.argmin(torch.abs(world.lidar_angles))
    front_dist = lidar[:, front_idx]

    assert abs(front_dist.mean().item() - 1.0) < 0.05, (
        f"边界 LiDAR 距离异常: mean={front_dist.mean().item():.4f}, expected≈1.0"
    )

    print_stats("boundary front distance", front_dist)
    print_ok("边界 LiDAR 检测正常")


def test_risk_features(world: Task2WorldManager) -> None:
    heading("[测试 7] risk features 范围与前方风险检测")

    cfg = world.cfg
    E = world.num_envs
    env_ids = torch.arange(E, dtype=torch.long, device=world.device)

    world.reset(env_ids, global_steps=0)
    disable_all_obstacles(world)

    root = torch.zeros((E, 2), dtype=torch.float32, device=world.device)
    yaw = torch.zeros(E, dtype=torch.float32, device=world.device)
    body_vx = torch.ones(E, dtype=torch.float32, device=world.device) * 0.6

    world.static_mask[:, 0] = True
    world.static_pos[:, 0, 0] = 1.2
    world.static_pos[:, 0, 1] = 0.0
    world.static_radius[:, 0] = 0.4

    features = world.compute_risk_features(root, yaw, body_vx=body_vx)

    check_shape("risk features", features, (E, 8))
    assert_finite("risk features", features)

    assert features.min().item() >= -1e-6
    assert features.max().item() <= 1.0 + 1e-6

    front_risk = features[:, 1]
    ttc = features[:, 5]
    front_clearance = features[:, 7]

    assert front_risk.mean().item() > 0.3, "前方障碍物距离很近，但 front_risk 不明显"
    assert ttc.mean().item() > 0.1, "前方有速度接近障碍物，但 ttc_proxy 不明显"
    assert front_clearance.mean().item() < 0.2, "前方障碍物很近，但 front_clearance 过大"

    print_stats("front risk", front_risk)
    print_stats("ttc proxy", ttc)
    print_stats("front clearance", front_clearance)
    print_ok("risk features 正常")


def test_event_detection(world: Task2WorldManager) -> None:
    heading("[测试 8] success / collision / out_of_bounds 事件检测")

    cfg = world.cfg
    E = world.num_envs
    env_ids = torch.arange(E, dtype=torch.long, device=world.device)

    world.reset(env_ids, global_steps=int(0.60 * cfg.curriculum_total_steps))
    disable_all_obstacles(world)

    root_goal = world.goal_pos.clone()
    event_goal = world.check_events(root_goal)

    assert event_goal["success"].float().mean().item() > 0.99
    assert event_goal["collision"].float().mean().item() < 1e-6
    assert event_goal["out_of_bounds"].float().mean().item() < 1e-6

    root_col = torch.zeros((E, 2), dtype=torch.float32, device=world.device)

    world.static_mask[:, 0] = True
    world.static_pos[:, 0, :] = 0.0
    world.static_radius[:, 0] = 0.7

    event_col = world.check_events(root_col)
    assert event_col["collision"].float().mean().item() > 0.99
    assert event_col["static_collision"].float().mean().item() > 0.99

    world.static_mask[:] = False
    world.dynamic_mask[:, 0] = True
    world.dynamic_pos[:, 0, :] = 0.0
    world.dynamic_radius[:, 0] = 0.5

    event_dyn = world.check_events(root_col)
    assert event_dyn["dynamic_collision"].float().mean().item() > 0.99

    root_oob = torch.zeros((E, 2), dtype=torch.float32, device=world.device)
    root_oob[:, 0] = cfg.half_extent + 1.0

    event_oob = world.check_events(root_oob)
    assert event_oob["out_of_bounds"].float().mean().item() > 0.99

    print_ok(f"success mean = {event_goal['success'].float().mean().item():.6f}")
    print_ok(f"static collision mean = {event_col['static_collision'].float().mean().item():.6f}")
    print_ok(f"dynamic collision mean = {event_dyn['dynamic_collision'].float().mean().item():.6f}")
    print_ok(f"out_of_bounds mean = {event_oob['out_of_bounds'].float().mean().item():.6f}")
    print_ok("事件检测正常")


def test_dynamic_obstacle_motion(world: Task2WorldManager) -> None:
    heading("[测试 9] 动态障碍物运动 / 边界反弹检测")

    cfg = world.cfg
    E = world.num_envs
    env_ids = torch.arange(E, dtype=torch.long, device=world.device)

    world.reset(env_ids, global_steps=int(0.60 * cfg.curriculum_total_steps))

    dyn_count = world.dynamic_mask.float().sum(dim=-1)
    assert dyn_count.mean().item() > 0.0, "Stage4 应有动态障碍物"

    pos0 = world.dynamic_pos.clone()

    for _ in range(50):
        world.step_dynamic_obstacles(dt=0.05)

    pos1 = world.dynamic_pos.clone()
    moved = torch.norm(pos1 - pos0, dim=-1)
    moved_valid = moved[world.dynamic_mask]

    assert moved_valid.mean().item() > 0.01, "动态障碍物没有明显移动"

    world.dynamic_mask[:, 0] = True
    world.dynamic_radius[:, 0] = 0.5
    world.dynamic_pos[:, 0, 0] = cfg.half_extent - 0.1
    world.dynamic_pos[:, 0, 1] = 0.0
    world.dynamic_vel[:, 0, 0] = 1.0
    world.dynamic_vel[:, 0, 1] = 0.0

    world.step_dynamic_obstacles(dt=0.5)

    assert (world.dynamic_vel[:, 0, 0] < 0.0).float().mean().item() > 0.99, (
        "动态障碍物碰边界后没有 x 方向反弹"
    )

    print_stats("dynamic movement valid", moved_valid)
    print_ok("动态障碍物运动与边界反弹正常")


def test_navigation_features(world: Task2WorldManager) -> None:
    heading("[测试 10] compute_navigation_features 综合检测")

    cfg = world.cfg
    E = world.num_envs
    env_ids = torch.arange(E, dtype=torch.long, device=world.device)

    world.reset(env_ids, global_steps=int(0.38 * cfg.curriculum_total_steps))

    root = world.start_pos.clone()
    yaw = torch.zeros(E, dtype=torch.float32, device=world.device)
    body_vx = torch.ones(E, dtype=torch.float32, device=world.device) * 0.4

    features = world.compute_navigation_features(
        root_pos_local=root,
        yaw=yaw,
        body_vx=body_vx,
        update_lidar_history=True,
    )

    required = [
        "goal_vec_w",
        "goal_vec_b",
        "goal_dist",
        "heading_error",
        "heading_sin",
        "heading_cos",
        "goal_x_body_norm",
        "goal_y_body_norm",
        "goal_dist_norm",
        "lidar_dist",
        "lidar_norm",
        "lidar_delta",
        "risk_features",
    ]

    for key in required:
        assert key in features, f"compute_navigation_features 缺少 {key}"
        assert_finite(key, features[key])

    check_shape("goal_vec_w", features["goal_vec_w"], (E, 2))
    check_shape("goal_vec_b", features["goal_vec_b"], (E, 2))
    check_shape("lidar_dist", features["lidar_dist"], (E, cfg.num_lidar_rays))
    check_shape("lidar_norm", features["lidar_norm"], (E, cfg.num_lidar_rays))
    check_shape("lidar_delta", features["lidar_delta"], (E, cfg.num_lidar_rays))
    check_shape("risk_features", features["risk_features"], (E, 8))

    assert features["lidar_norm"].min().item() >= 0.0
    assert features["lidar_norm"].max().item() <= 1.0 + 1e-6
    assert features["risk_features"].min().item() >= 0.0
    assert features["risk_features"].max().item() <= 1.0 + 1e-6

    print_stats("goal dist", features["goal_dist"])
    print_stats("lidar norm", features["lidar_norm"])
    print_stats("risk features", features["risk_features"])
    print_ok("综合导航特征正常")


def test_rotate_inverse(world: Task2WorldManager) -> None:
    heading("[测试 11] rotate_world_to_body_2d / rotate_body_to_world_2d 互逆检测")

    E = world.num_envs
    vec_w = torch.randn((E, 2), dtype=torch.float32, device=world.device)
    yaw = torch.rand(E, dtype=torch.float32, device=world.device) * 2.0 * math.pi - math.pi

    vec_b = world.rotate_world_to_body_2d(vec_w, yaw)
    vec_w2 = world.rotate_body_to_world_2d(vec_b, yaw)

    err = torch.norm(vec_w - vec_w2, dim=-1)

    assert err.max().item() < 1e-5, f"2D 旋转互逆误差过大: {err.max().item():.8f}"
    print_stats("rotation inverse error", err)
    print_ok("2D 坐标旋转互逆正常")


def test_random_world_rollout(world: Task2WorldManager) -> None:
    heading(f"[测试 12] 随机世界 rollout {args.steps} 步稳定性检测")

    cfg = world.cfg
    E = world.num_envs
    env_ids = torch.arange(E, dtype=torch.long, device=world.device)

    world.reset(env_ids, global_steps=int(0.80 * cfg.curriculum_total_steps))

    records: List[Dict[str, float]] = []
    start_time = time.time()

    for step in range(int(args.steps)):
        world.step_dynamic_obstacles(dt=0.05)

        root = random_root_positions(cfg, E, world.device)
        yaw = torch.rand(E, dtype=torch.float32, device=world.device) * 2.0 * math.pi - math.pi
        body_vx = torch.rand(E, dtype=torch.float32, device=world.device) * 1.0

        nav = world.compute_navigation_features(
            root_pos_local=root,
            yaw=yaw,
            body_vx=body_vx,
            update_lidar_history=True,
        )
        event = world.check_events(root)

        for key, value in nav.items():
            assert_finite(f"nav/{key}", value)
        for key, value in event.items():
            assert_finite(f"event/{key}", value)

        if (step + 1) % max(100, int(args.steps) // 10) == 0 or (step + 1) == int(args.steps):
            stats = world.get_debug_stats(root)
            rec = {
                "step": float(step + 1),
                "GoalDist": nav["goal_dist"].mean().item(),
                "LidarMin": nav["lidar_dist"].min().item(),
                "LidarMean": nav["lidar_dist"].mean().item(),
                "RiskAll": nav["risk_features"][:, 0].mean().item(),
                "RiskFront": nav["risk_features"][:, 1].mean().item(),
                "RiskDynamic": nav["risk_features"][:, 4].mean().item(),
                "RiskBoundary": nav["risk_features"][:, 6].mean().item(),
                "FrontClearance": nav["risk_features"][:, 7].mean().item(),
                "SuccessRate": event["success"].float().mean().item(),
                "CollisionRate": event["collision"].float().mean().item(),
                "StaticCollisionRate": event["static_collision"].float().mean().item(),
                "DynamicCollisionRate": event["dynamic_collision"].float().mean().item(),
                "OOBRate": event["out_of_bounds"].float().mean().item(),
                "MinObstacleSignedDistance": event["min_obstacle_signed_distance"].mean().item(),
                "BoundaryMargin": event["boundary_margin"].mean().item(),
                **stats,
            }
            records.append(rec)

            print(
                f" -> Step {step + 1:05d} | "
                f"GoalDist={rec['GoalDist']:.2f} | "
                f"LidarMin={rec['LidarMin']:.2f} | "
                f"LidarMean={rec['LidarMean']:.2f} | "
                f"RiskFront={rec['RiskFront']:.3f} | "
                f"RiskDyn={rec['RiskDynamic']:.3f} | "
                f"Succ={rec['SuccessRate']:.4f} | "
                f"Coll={rec['CollisionRate']:.4f} | "
                f"OOB={rec['OOBRate']:.4f}",
                flush=True,
            )

    elapsed = time.time() - start_time
    fps = int(args.steps) * E / max(elapsed, 1e-6)

    print_ok(f"随机世界 rollout 完成: {args.steps} steps, {args.steps * E:,} transitions")
    print_ok(f"纯世界层吞吐约: {fps:,.2f} world steps/s")

    heading("[测试 13] 随机 rollout 统计报告")
    print_summary_table(summarize_records(records))

    print_ok("随机世界 rollout 无 NaN/Inf")


# ======================================================================
# Main
# ======================================================================

def run_tests() -> None:
    heading("Diff-Drive UGV / Jetbot Task2 Analytic World 全量测试启动")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print_warn("CUDA 不可用，自动切换到 CPU")
    else:
        device = args.device

    if bool(args.quick):
        args.num_envs = min(int(args.num_envs), 512)
        args.steps = min(int(args.steps), 300)

    cfg = Task2WorldConfig()
    cfg.validate()

    world = Task2WorldManager(
        cfg=cfg,
        num_envs=int(args.num_envs),
        device=str(device),
    )

    test_config(world)
    test_curriculum_mapping(world)
    test_stage_reset_sampling(world)
    test_obstacle_safety(world)
    test_goal_terms(world)
    test_lidar_manual_front_obstacle(world)
    test_lidar_boundary(world)
    test_risk_features(world)
    test_event_detection(world)
    test_dynamic_obstacle_motion(world)
    test_navigation_features(world)
    test_rotate_inverse(world)
    test_random_world_rollout(world)

    print("\n💡 【Task2 World 测试解读】")
    print("1. goal distance、static/dynamic count 必须随课程阶段递增。")
    print("2. start_pos / goal_pos / obstacle_pos 全部是 env local xy 坐标。")
    print("3. 正前方障碍物 LiDAR 命中距离应接近 obstacle_center_dist - radius。")
    print("4. success / static_collision / dynamic_collision / out_of_bounds 都应能手动触发。")
    print("5. 随机 rollout 中 collision / OOB 可以非零，因为 root 是随机采样的。")
    print("6. 世界层通过后，再进入 task2_env.py 编写。")

    heading("Diff-Drive UGV / Jetbot Task2 Analytic World 测试全部通过")


if __name__ == "__main__":
    run_tests()

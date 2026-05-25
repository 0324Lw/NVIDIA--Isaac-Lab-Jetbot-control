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

parser = argparse.ArgumentParser(description="Diff-Drive UGV / Jetbot Task4 Multi-UGV Formation Escort World Test")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-assets", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from diff_drive_rl.tasks.task4.task4_world import (
    Task4WorldConfig,
    Task4WorldManager,
    get_lidar_cfg,
    spawn_world_assets,
)


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
    print(" " * 50 + "Diff-Drive UGV / Jetbot Task4 World 统计报告")
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


def random_root_positions(cfg: Task4WorldConfig, num_envs: int, device: str) -> torch.Tensor:
    root = torch.empty((num_envs, cfg.num_agents, 2), dtype=torch.float32, device=device)
    root[:, :, 0] = float(cfg.x_min) + torch.rand((num_envs, cfg.num_agents), dtype=torch.float32, device=device) * (
        float(cfg.x_max) - float(cfg.x_min)
    )
    root[:, :, 1] = float(cfg.y_min) + torch.rand((num_envs, cfg.num_agents), dtype=torch.float32, device=device) * (
        float(cfg.y_max) - float(cfg.y_min)
    )
    return root


def make_scene_and_world(device: str):
    cfg = Task4WorldConfig()
    # World-layer test focuses on analytic world tensors. Avoid repeated
    # PhysX GPU RigidObject teleport in Stage 0~5 reset loops.
    cfg.enable_physical_asset_teleport = False
    cfg.validate()

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0)
    light_cfg.func("/World/Light", light_cfg)

    scene_cfg = InteractiveSceneCfg(
        num_envs=int(args_cli.num_envs),
        env_spacing=22.0,
    )

    spawn_world_assets(scene_cfg, cfg)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.update(0.0)

    world = Task4WorldManager(
        scene=scene,
        cfg=cfg,
        num_envs=int(args_cli.num_envs),
        device=device,
    )

    return cfg, sim, scene, world


# ======================================================================
# Tests
# ======================================================================

def test_project_files() -> None:
    heading("[测试 0] Task4 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task4_multi_ugv_formation_escort.yaml",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task4" / "task4_world.py",
        PROJECT_ROOT / "tests" / "task4" / "task4_world_test.py",
    ]

    missing = [str(p) for p in required if not p.exists()]
    assert not missing, "缺少 Task4 必要文件:\n" + "\n".join(missing)

    for path in required:
        print_ok(str(path.relative_to(PROJECT_ROOT)))


def test_config(cfg: Task4WorldConfig) -> None:
    heading("[测试 1] Task4WorldConfig 基础配置检测")

    cfg.validate()

    assert cfg.num_agents == 4
    assert cfg.max_static_obstacles == 8
    assert cfg.num_formation_types == 3
    assert cfg.lidar_pool_bins == 48
    assert Task4WorldManager.risk_feature_dim() == 16
    assert Task4WorldManager.privileged_feature_dim(cfg.max_static_obstacles, cfg.num_agents) == 96

    assert cfg.gate_stage_start == 3
    assert cfg.gate_gap_width > 1.0
    assert cfg.formation_scale_gate > 0.0

    print_ok(f"num_agents = {cfg.num_agents}")
    print_ok(f"arena = {cfg.arena_length} x {cfg.arena_width}")
    print_ok(f"max_static_obstacles = {cfg.max_static_obstacles}")
    print_ok(f"formation_types = {cfg.num_formation_types}  # 0 Diamond, 1 Wedge, 2 Line")
    print_ok(f"gate_stage_start = {cfg.gate_stage_start}")
    print_ok(f"lidar_pool_bins = {cfg.lidar_pool_bins}")
    print_ok(f"risk_feature_dim = {Task4WorldManager.risk_feature_dim()}")
    print_ok(f"privileged_feature_dim = {Task4WorldManager.privileged_feature_dim(cfg.max_static_obstacles, cfg.num_agents)}")
    print_ok("Task4WorldConfig 正常")


def test_init_assets(scene: InteractiveScene, world: Task4WorldManager) -> None:
    heading("[测试 2] Isaac 世界资产 / RigidObject 注册检测")

    rigid_names = sorted(list(scene.rigid_objects.keys()))

    if args_cli.print_assets:
        print("scene.rigid_objects:")
        for name in rigid_names:
            print(f"  - {name}")

    for i in range(world.cfg.max_static_obstacles):
        name = f"static_obstacle_{i}"
        assert name in scene.rigid_objects, f"scene.rigid_objects 缺少 {name}"
        assert world.static_obstacles[i] is scene.rigid_objects[name]

    for name in ["gate_top", "gate_bottom"]:
        assert name in scene.rigid_objects, f"scene.rigid_objects 缺少 {name}"

    assert world.gate_top is scene.rigid_objects["gate_top"]
    assert world.gate_bottom is scene.rigid_objects["gate_bottom"]

    print_ok(f"rigid object count = {len(rigid_names)}")
    print_ok("static_obstacle_0~7 / gate_top / gate_bottom 注册正常")


def test_curriculum_sampling(world: Task4WorldManager, scene: InteractiveScene) -> None:
    heading("[测试 3] Stage 0~5 课程采样：障碍数量 / gate / formation 检测")

    cfg = world.cfg
    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)

    for stage in range(6):
        world.reset_world(env_ids, stage=stage)
        scene.update(0.0)

        assert (world.curriculum_stage[env_ids] == stage).all().item()

        expected_obs = int(cfg.obstacle_count_by_stage[min(stage, len(cfg.obstacle_count_by_stage) - 1)])
        obstacle_counts = world.obstacle_active[env_ids].float().sum(dim=-1)

        assert torch.all(obstacle_counts == expected_obs).item(), (
            f"stage {stage} obstacle count mismatch: got {obstacle_counts.unique().detach().cpu().tolist()}, "
            f"expected {expected_obs}"
        )

        expected_gate = stage >= int(cfg.gate_stage_start)
        assert (world.gate_active[env_ids] == expected_gate).all().item(), f"stage {stage} gate_active mismatch"

        if stage == 0:
            assert (world.formation_type[env_ids] == 0).all().item(), "Stage0 应固定 Diamond"
        if stage == 3:
            assert (world.formation_type[env_ids] == 2).all().item(), "Stage3 应使用 Line 队形通过窄门"
            assert (world.formation_scale[env_ids] <= cfg.formation_scale_gate + 1e-6).all().item(), (
                "Stage3 Line 队形应压缩 formation_scale"
            )

        print_ok(
            f"stage={stage} | obs={expected_obs} | gate={int(expected_gate)} | "
            f"formation_mean={world.formation_type.float().mean().item():.3f} | "
            f"scale_mean={world.formation_scale.mean().item():.3f}"
        )

    print_ok("Stage 0~5 课程采样正常")


def test_reset_sampling(world: Task4WorldManager, scene: InteractiveScene) -> None:
    heading("[测试 4] reset_world 起点 / 目标 / 编队槽位对齐检测")

    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)

    world.reset_world(env_ids, stage=4)
    scene.update(0.0)

    assert_finite_tensor("start_center", world.start_center)
    assert_finite_tensor("start_heading", world.start_heading)
    assert_finite_tensor("start_pos", world.start_pos)
    assert_finite_tensor("start_yaw", world.start_yaw)
    assert_finite_tensor("goal_pos", world.goal_pos)
    assert_finite_tensor("goal_yaw", world.goal_yaw)
    assert_finite_tensor("formation_scale", world.formation_scale)
    assert_finite_tensor("obstacle_pos", world.obstacle_pos)
    assert_finite_tensor("gate_x", world.gate_x)

    slots = world.compute_formation_slots(
        center_xy=world.start_center,
        heading=world.start_heading,
        formation_type=world.formation_type,
        scale=world.formation_scale,
    )

    slot_err = torch.norm(world.start_pos - slots, dim=-1)
    assert slot_err.max().item() < 1e-5, f"start_pos 与 formation slots 不一致: {slot_err.max().item()}"

    pair_dists = world.compute_pairwise_distances(world.start_pos)
    assert pair_dists.min().item() > 2.0 * world.cfg.robot_radius + 0.02, (
        f"reset 后队内距离过近: min_pair={pair_dists.min().item():.6f}"
    )

    assert world.start_center[:, 0].min().item() >= world.cfg.start_center_x_range[0] - 1e-5
    assert world.start_center[:, 0].max().item() <= world.cfg.start_center_x_range[1] + 1e-5
    assert world.goal_pos[:, 0].min().item() >= world.cfg.goal_x_range[0] - 1e-5
    assert world.goal_pos[:, 0].max().item() <= world.cfg.goal_x_range[1] + 1e-5

    print_stats("start_center_x", world.start_center[:, 0])
    print_stats("start_center_y", world.start_center[:, 1])
    print_stats("goal_x", world.goal_pos[:, 0])
    print_stats("goal_y", world.goal_pos[:, 1])
    print_stats("slot_err", slot_err)
    print_stats("pair_dists", pair_dists)
    print_ok("reset 起点 / 目标 / 编队槽位对齐正常")


def test_domain_randomization_ranges(world: Task4WorldManager) -> None:
    heading("[测试 5] Sim2Real domain randomization buffer 范围检测")

    cfg = world.cfg
    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)

    world.reset_world(env_ids, stage=5)

    checks = [
        ("max_speed", world.max_speed, cfg.max_speed_range),
        ("max_yaw_rate", world.max_yaw_rate, cfg.max_yaw_rate_range),
        ("action_deadband", world.action_deadband, cfg.action_deadband_range),
        ("action_ema_alpha", world.action_ema_alpha, cfg.action_ema_alpha_range),
        ("motor_strength", world.motor_strength, cfg.motor_strength_range),
        ("motor_bias", world.motor_bias, cfg.motor_bias_range),
        ("wheel_radius_scale", world.wheel_radius_scale, cfg.wheel_radius_scale_range),
        ("lidar_noise_std", world.lidar_noise_std, cfg.lidar_noise_std_range),
        ("lidar_outlier_prob", world.lidar_outlier_prob, cfg.lidar_outlier_prob_range),
        ("lidar_dropout_prob", world.lidar_dropout_prob, cfg.lidar_dropout_prob_range),
        ("lidar_yaw_offset", world.lidar_yaw_offset, cfg.lidar_yaw_offset_range),
        ("lidar_z_offset", world.lidar_z_offset, cfg.lidar_z_offset_range),
    ]

    for name, tensor, rng in checks:
        assert_finite_tensor(name, tensor)
        assert tensor.min().item() >= float(rng[0]) - 1e-6, f"{name} below range"
        assert tensor.max().item() <= float(rng[1]) + 1e-6, f"{name} above range"
        print_ok(f"{name:<22s} range = {tensor.min().item():+.6f} ~ {tensor.max().item():+.6f}")

    dmin, dmax = cfg.action_delay_frame_range
    assert world.action_delay_frames.min().item() >= int(dmin)
    assert world.action_delay_frames.max().item() <= int(dmax)
    print_ok(f"action_delay_frames range = {world.action_delay_frames.min().item()} ~ {world.action_delay_frames.max().item()}")
    print_ok("Sim2Real DR buffer 范围正常")


def test_rigid_object_positions(world: Task4WorldManager, scene: InteractiveScene) -> None:
    heading("[测试 6] RigidObject 注册 / world-test safety mode 检测")

    # In pure world-layer tests we deliberately disable repeated physical
    # teleport to avoid PhysX GPU tensor illegal-memory-access from repeated
    # kinematic RigidObject writes. This test still verifies that assets are
    # registered and can be accessed by InteractiveScene. Full physical reset
    # will be checked later in Task4 env test with very small num_envs.
    assert bool(world.cfg.enable_physical_asset_teleport) is False

    for i, obj in enumerate(world.static_obstacles):
        pos = obj.data.root_pos_w
        quat = obj.data.root_quat_w
        assert_finite_tensor(f"static_obstacle_{i}.root_pos_w", pos)
        assert_finite_tensor(f"static_obstacle_{i}.root_quat_w", quat)
        print_ok(f"static_obstacle_{i} registered, root_pos_shape={tuple(pos.shape)}")

    for name, obj in [
        ("gate_top", world.gate_top),
        ("gate_bottom", world.gate_bottom),
    ]:
        pos = obj.data.root_pos_w
        quat = obj.data.root_quat_w
        assert_finite_tensor(f"{name}.root_pos_w", pos)
        assert_finite_tensor(f"{name}.root_quat_w", quat)
        print_ok(f"{name} registered, root_pos_shape={tuple(pos.shape)}")

    print_ok("RigidObject 注册正常；world test safety mode 已启用")


def test_formation_geometry(world: Task4WorldManager) -> None:
    heading("[测试 7] Diamond / Wedge / Line 队形几何检测")

    n = world.num_envs
    center = torch.zeros((n, 2), dtype=torch.float32, device=world.device)
    heading0 = torch.zeros((n,), dtype=torch.float32, device=world.device)
    scale = torch.ones((n,), dtype=torch.float32, device=world.device)

    for ftype in range(3):
        f = torch.full((n,), ftype, dtype=torch.long, device=world.device)
        slots = world.compute_formation_slots(center, heading0, f, scale)
        pair_dists = world.compute_pairwise_distances(slots)

        assert_finite_tensor(f"formation slots {ftype}", slots)
        assert slots.shape == (n, world.num_agents, 2)
        assert pair_dists.min().item() > 2.0 * world.cfg.robot_radius + 0.02

        team = world.compute_team_terms(
            root_pos_local=slots,
            yaw=torch.zeros((n, world.num_agents), dtype=torch.float32, device=world.device),
            env_ids=torch.arange(n, dtype=torch.long, device=world.device),
        )

        assert_finite_tensor(f"team mean slot error {ftype}", team["mean_slot_error"])

        print_ok(
            f"formation={ftype} | min_pair={pair_dists.min().item():.4f} | "
            f"spread={team['team_spread'].mean().item():.4f}"
        )

    # Stage3 compressed Line must still avoid pair collision.
    line = torch.full((n,), 2, dtype=torch.long, device=world.device)
    compressed = torch.full((n,), float(world.cfg.formation_scale_gate), dtype=torch.float32, device=world.device)
    line_slots = world.compute_formation_slots(center, heading0, line, compressed)
    line_pair = world.compute_pairwise_distances(line_slots)

    assert line_pair.min().item() > 2.0 * world.cfg.robot_radius + 0.02, (
        f"compressed line too close: {line_pair.min().item()}"
    )

    print_ok(f"compressed Line min_pair = {line_pair.min().item():.4f}")
    print_ok("队形几何正常")


def test_events(world: Task4WorldManager) -> None:
    heading("[测试 8] success / out_of_bounds / obstacle / gate / pair collision 事件检测")

    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)
    zero_yaw = torch.zeros((world.num_envs, world.num_agents), dtype=torch.float32, device=world.device)
    zero_lin = torch.zeros((world.num_envs, world.num_agents, 2), dtype=torch.float32, device=world.device)

    # success: center at goal, slots around goal_yaw, zero speed.
    world.reset_world(env_ids, stage=0)
    success_root = world.compute_formation_slots(
        center_xy=world.goal_pos,
        heading=world.goal_yaw,
        formation_type=world.formation_type,
        scale=world.formation_scale,
    )
    success_yaw = world.goal_yaw.unsqueeze(-1).expand(world.num_envs, world.num_agents)
    ev_success = world.check_events(success_root, success_yaw, lin_vel=zero_lin)
    assert ev_success["success_candidate"].float().mean().item() > 0.99
    assert ev_success["crash"].float().mean().item() < 1e-6

    # out of bounds.
    world.reset_world(env_ids, stage=0)
    oob_root = world.start_pos.clone()
    oob_root[:, :, 0] = float(world.cfg.x_max) + 0.5
    ev_oob = world.check_events(oob_root, zero_yaw, lin_vel=zero_lin)
    assert ev_oob["out_of_bounds"].float().mean().item() > 0.99
    assert ev_oob["crash"].float().mean().item() > 0.99

    # obstacle collision.
    world.reset_world(env_ids, stage=2)
    obs_root = world.start_pos.clone()
    obs_root[:, 0, :] = world.obstacle_pos[:, 0, :]
    ev_obs = world.check_events(obs_root, zero_yaw, lin_vel=zero_lin)
    assert ev_obs["obstacle_collision"][:, 0].float().mean().item() > 0.99
    assert ev_obs["crash"].float().mean().item() > 0.99

    # gate collision.
    world.reset_world(env_ids, stage=3)
    gate_root = world.start_pos.clone()
    gate_root[:, 0, 0] = world.gate_x
    gate_root[:, 0, 1] = float(world.cfg.gate_top_center_y)
    ev_gate = world.check_events(gate_root, zero_yaw, lin_vel=zero_lin)
    assert ev_gate["gate_collision"][:, 0].float().mean().item() > 0.99
    assert ev_gate["crash"].float().mean().item() > 0.99

    # pair collision.
    world.reset_world(env_ids, stage=0)
    pair_root = world.start_pos.clone()
    pair_root[:, 1, :] = pair_root[:, 0, :]
    ev_pair = world.check_events(pair_root, zero_yaw, lin_vel=zero_lin)
    assert ev_pair["pair_collision_any"].float().mean().item() > 0.99
    assert ev_pair["crash"].float().mean().item() > 0.99

    print_ok(f"success_candidate = {ev_success['success_candidate'].float().mean().item():.6f}")
    print_ok(f"out_of_bounds = {ev_oob['out_of_bounds'].float().mean().item():.6f}")
    print_ok(f"obstacle_collision agent0 = {ev_obs['obstacle_collision'][:, 0].float().mean().item():.6f}")
    print_ok(f"gate_collision agent0 = {ev_gate['gate_collision'][:, 0].float().mean().item():.6f}")
    print_ok(f"pair_collision_any = {ev_pair['pair_collision_any'].float().mean().item():.6f}")
    print_ok("事件检测正常")


def test_gate_progress(world: Task4WorldManager) -> None:
    heading("[测试 9] gate near / before / passed 进度项检测")

    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)

    world.reset_world(env_ids, stage=3)

    root_before = world.start_pos.clone()
    root_before[:, :, 0] = world.gate_x.unsqueeze(-1) - float(world.cfg.gate_pass_margin_x) - 0.20
    root_before[:, :, 1] = 0.0
    gate_before = world.gate_progress_terms(root_before)

    root_near = world.start_pos.clone()
    root_near[:, :, 0] = world.gate_x.unsqueeze(-1)
    root_near[:, :, 1] = 0.0
    gate_near = world.gate_progress_terms(root_near)

    root_passed = world.start_pos.clone()
    root_passed[:, :, 0] = world.gate_x.unsqueeze(-1) + float(world.cfg.gate_pass_margin_x) + 0.20
    root_passed[:, :, 1] = 0.0
    gate_passed = world.gate_progress_terms(root_passed)

    assert gate_before["before_gate"].float().mean().item() > 0.99
    assert gate_near["near_gate"].float().mean().item() > 0.99
    assert gate_passed["passed_gate"].float().mean().item() > 0.99

    print_ok(f"before_gate = {gate_before['before_gate'].float().mean().item():.6f}")
    print_ok(f"near_gate = {gate_near['near_gate'].float().mean().item():.6f}")
    print_ok(f"passed_gate = {gate_passed['passed_gate'].float().mean().item():.6f}")


def test_lidar_processing(world: Task4WorldManager) -> None:
    heading("[测试 10] analytic LiDAR / pooling / delta 检测")

    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)

    world.reset_world(env_ids, stage=5)

    root = world.start_pos.clone()
    yaw = world.start_yaw.clone()

    lidar = world.compute_analytic_lidar(
        root_pos_local=root,
        yaw=yaw,
        env_ids=env_ids,
        add_noise=False,
        update_history=True,
        normalize=False,
    )

    check_shape("lidar", lidar, (world.num_envs, world.num_agents, world.cfg.lidar_pool_bins))
    assert_finite_tensor("lidar", lidar)
    assert lidar.min().item() >= world.cfg.lidar_min_distance - 1e-6
    assert lidar.max().item() <= world.cfg.lidar_max_distance + 1e-6

    lidar_norm = world.compute_analytic_lidar(
        root_pos_local=root,
        yaw=yaw,
        env_ids=env_ids,
        add_noise=False,
        update_history=True,
        normalize=True,
    )

    check_shape("lidar_norm", lidar_norm, (world.num_envs, world.num_agents, world.cfg.lidar_pool_bins))
    assert lidar_norm.min().item() >= 0.0
    assert lidar_norm.max().item() <= 1.0 + 1e-6

    assert world.last_lidar_delta.min().item() >= -1.0 - 1e-6
    assert world.last_lidar_delta.max().item() <= 1.0 + 1e-6

    raw = torch.arange(world.num_envs * 96, dtype=torch.float32, device=world.device).reshape(world.num_envs, 96)
    pooled = world._pool_lidar(raw)
    check_shape("pooled raw lidar", pooled, (world.num_envs, world.cfg.lidar_pool_bins))

    raw_short = torch.arange(world.num_envs * 12, dtype=torch.float32, device=world.device).reshape(world.num_envs, 12)
    pooled_short = world._pool_lidar(raw_short)
    check_shape("pooled short lidar", pooled_short, (world.num_envs, world.cfg.lidar_pool_bins))

    print_stats("lidar", lidar)
    print_stats("lidar_norm", lidar_norm)
    print_stats("lidar_delta", world.last_lidar_delta)
    print_ok("analytic LiDAR / pooling / delta 正常")


def test_risk_features(world: Task4WorldManager) -> None:
    heading("[测试 11] risk features 维度 / 范围 / 风险响应检测")

    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)

    world.reset_world(env_ids, stage=5)

    root = world.start_pos.clone()
    yaw = world.start_yaw.clone()

    fake_lidar = torch.full(
        (world.num_envs, world.num_agents, world.cfg.lidar_pool_bins),
        float(world.cfg.lidar_max_distance),
        dtype=torch.float32,
        device=world.device,
    )

    angles = world.pooled_lidar_angles
    front_mask = torch.abs(angles) <= math.radians(world.cfg.front_angle_deg)
    fake_lidar[:, :, front_mask] = 0.20

    risk = world.compute_risk_features(root, yaw, lidar_pooled=fake_lidar)

    check_shape("risk features", risk, (world.num_envs, world.num_agents, world.risk_feature_dim()))
    assert_finite_tensor("risk features", risk)
    assert risk.min().item() >= -1e-6
    assert risk.max().item() <= 1.0 + 1e-6

    assert risk[:, :, 1].mean().item() > 0.50, "front lidar 很近时 front_risk 应升高"

    root_edge = root.clone()
    root_edge[:, :, 1] = float(world.cfg.y_max) - 0.05
    risk_edge = world.compute_risk_features(root_edge, yaw, lidar_pooled=fake_lidar)
    assert risk_edge[:, :, 4].mean().item() > 0.50, "靠近边界时 boundary_risk 应升高"

    root_pair = root.clone()
    root_pair[:, 1, :] = root_pair[:, 0, :]
    risk_pair = world.compute_risk_features(root_pair, yaw, lidar_pooled=fake_lidar)
    assert risk_pair[:, :, 7].mean().item() > 0.20, "队友过近时 pair_risk 应升高"

    print_stats("front risk", risk[:, :, 1])
    print_stats("boundary risk edge", risk_edge[:, :, 4])
    print_stats("pair risk", risk_pair[:, :, 7])
    print_ok("risk features 正常")


def test_privileged_features(world: Task4WorldManager) -> None:
    heading("[测试 12] privileged features 维度与数值检测")

    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)

    world.reset_world(env_ids, stage=5)

    root = world.start_pos.clone()
    yaw = world.start_yaw.clone()

    priv = world.compute_privileged_features(root, yaw)

    expected = world.privileged_feature_dim(world.cfg.max_static_obstacles, world.cfg.num_agents)

    check_shape("privileged features", priv, (world.num_envs, expected))
    assert_finite_tensor("privileged features", priv)
    assert priv.abs().max().item() <= 10.0 + 1e-5

    assert expected == 96

    print_ok(f"privileged feature dim = {expected}")
    print_ok(f"priv range = {priv.min().item():+.6f} ~ {priv.max().item():+.6f}")
    print_ok("privileged features 正常")


def test_lidar_cfg() -> None:
    heading("[测试 13] get_lidar_cfg 配置检测")

    cfg = Task4WorldConfig()
    cfg.validate()

    lidar_cfg = get_lidar_cfg("{ENV_REGEX_NS}/Robot_0/chassis", cfg)

    assert lidar_cfg.prim_path == "{ENV_REGEX_NS}/Robot_0/chassis"
    assert abs(float(lidar_cfg.max_distance) - float(cfg.lidar_max_distance)) < 1e-6

    print_ok(f"lidar prim_path = {lidar_cfg.prim_path}")
    print_ok(f"lidar max_distance = {lidar_cfg.max_distance}")
    print_ok("get_lidar_cfg 正常")


def test_random_world_rollout(world: Task4WorldManager, scene: InteractiveScene) -> None:
    heading(f"[测试 14] 随机世界 rollout {args_cli.steps} 步稳定性检测")

    cfg = world.cfg
    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)

    world.reset_world(env_ids, stage=5)
    scene.update(0.0)

    records: List[Dict[str, float]] = []
    start_time = time.time()
    collect_interval = max(100, int(args_cli.steps) // 10)

    for step in range(int(args_cli.steps)):
        root = random_root_positions(cfg, world.num_envs, world.device)
        yaw = torch.rand((world.num_envs, world.num_agents), dtype=torch.float32, device=world.device) * 2.0 * math.pi - math.pi
        lin_vel = torch.randn((world.num_envs, world.num_agents, 2), dtype=torch.float32, device=world.device) * 0.25

        event = world.check_events(root, yaw, lin_vel=lin_vel)
        lidar = world.compute_analytic_lidar(
            root_pos_local=root,
            yaw=yaw,
            env_ids=env_ids,
            add_noise=True,
            update_history=True,
            normalize=False,
        )
        risk = world.compute_risk_features(root, yaw, lidar_pooled=lidar)
        priv = world.compute_privileged_features(root, yaw, lin_vel=lin_vel)
        stats = world.get_debug_stats(root, yaw)

        for name, value in event.items():
            assert_finite_tensor(f"event/{name}", value)
        assert_finite_tensor("lidar rollout", lidar)
        assert_finite_tensor("risk rollout", risk)
        assert_finite_tensor("priv rollout", priv)

        if (step + 1) % collect_interval == 0 or (step + 1) == int(args_cli.steps):
            rec = {
                "step": float(step + 1),
                "CenterGoalDist": event["center_goal_dist"].mean().item(),
                "MeanSlotError": event["mean_slot_error"].mean().item(),
                "MaxSlotError": event["max_slot_error"].mean().item(),
                "MinPairDist": event["min_pair_dist"].mean().item(),
                "TeamSpread": event["team_spread"].mean().item(),
                "SuccessCandidate": event["success_candidate"].float().mean().item(),
                "Crash": event["crash"].float().mean().item(),
                "OutOfBounds": event["out_of_bounds"].float().mean().item(),
                "ObstacleCollision": event["obstacle_collision"].float().mean().item(),
                "GateCollision": event["gate_collision"].float().mean().item(),
                "PairCollision": event["pair_collision_any"].float().mean().item(),
                "LidarMin": lidar.min().item(),
                "LidarMean": lidar.mean().item(),
                "RiskFront": risk[:, :, 1].mean().item(),
                "RiskObstacle": risk[:, :, 5].mean().item(),
                "RiskGate": risk[:, :, 6].mean().item(),
                "RiskPair": risk[:, :, 7].mean().item(),
                "PrivAbsMax": priv.abs().max().item(),
                **stats,
            }
            records.append(rec)

            print(
                f" -> Step {step + 1:05d} | "
                f"GoalDist={rec['CenterGoalDist']:.2f} | "
                f"SlotErr={rec['MeanSlotError']:.2f} | "
                f"MinPair={rec['MinPairDist']:.2f} | "
                f"LidarMin={rec['LidarMin']:.2f} | "
                f"RiskF={rec['RiskFront']:.3f} | "
                f"RiskObs={rec['RiskObstacle']:.3f} | "
                f"RiskGate={rec['RiskGate']:.3f} | "
                f"RiskPair={rec['RiskPair']:.3f} | "
                f"SuccCand={rec['SuccessCandidate']:.4f} | "
                f"Crash={rec['Crash']:.4f}",
                flush=True,
            )

    elapsed = time.time() - start_time
    fps = int(args_cli.steps) * world.num_envs / max(elapsed, 1e-6)

    print_ok(f"随机世界 rollout 完成: {args_cli.steps} steps, {args_cli.steps * world.num_envs:,} transitions")
    print_ok(f"世界层吞吐约: {fps:,.2f} env steps/s")

    heading("[测试 15] 随机 rollout 统计报告")
    print_summary_table(summarize_records(records))

    print_ok("随机世界 rollout 无 NaN/Inf")


# ======================================================================
# Main
# ======================================================================

def run_tests() -> None:
    heading("🚀 Diff-Drive UGV / Jetbot Task4 Multi-UGV Formation Escort World 全量测试启动")

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    if args_cli.test_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print_warn("CUDA 不可用，自动切换到 CPU")
    else:
        device = args_cli.test_device

    if bool(args_cli.quick):
        args_cli.num_envs = min(int(args_cli.num_envs), 16)
        args_cli.steps = min(int(args_cli.steps), 400)

    test_project_files()

    cfg, sim, scene, world = make_scene_and_world(device)

    print_ok(f"device = {device}")
    print_ok(f"num_envs = {world.num_envs}")
    print_ok(f"num_agents = {world.num_agents}")
    print_ok(f"max_static_obstacles = {world.cfg.max_static_obstacles}")
    print_ok(f"risk_feature_dim = {world.risk_feature_dim()}")
    print_ok(f"privileged_feature_dim = {world.privileged_feature_dim(world.cfg.max_static_obstacles, world.cfg.num_agents)}")

    test_config(cfg)
    test_init_assets(scene, world)
    test_curriculum_sampling(world, scene)
    test_reset_sampling(world, scene)
    test_domain_randomization_ranges(world)
    test_rigid_object_positions(world, scene)
    test_formation_geometry(world)
    test_events(world)
    test_gate_progress(world)
    test_lidar_processing(world)
    test_risk_features(world)
    test_privileged_features(world)
    test_lidar_cfg()
    test_random_world_rollout(world, scene)

    print("\n💡 【Task4 World 测试解读】")
    print("1. Stage0 应无障碍、无窄门、固定 Diamond，方便早期学习团队目标导航。")
    print("2. Stage2 开始加入静态障碍；Stage3 加入窄门并使用压缩 Line 队形。")
    print("3. Stage4/5 队形随机，障碍与 Sim2Real 参数更复杂。")
    print("4. reset 后 4 车起点必须满足 formation slots，且不能队内初始碰撞。")
    print("5. success_candidate 只表示几何成功候选，环境层后续还需要 stable hold。")
    print("6. risk features 为 [N, 4, 16]，privileged features 为 [N, 96]。")
    print("7. 世界层通过后，再进入 task4_config.py / task4_scene.py / task4_env.py。")

    heading("✅ Diff-Drive UGV / Jetbot Task4 Multi-UGV Formation Escort World 测试全部通过")


if __name__ == "__main__":
    try:
        run_tests()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass

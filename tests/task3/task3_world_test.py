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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diff-Drive UGV / Jetbot Task3 Conservative Sim2Real Parking World Test")
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

from diff_drive_rl.tasks.task3.task3_world import (
    Task3WorldConfig,
    Task3WorldManager,
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
        f"{name:<42s} "
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
    print(" " * 48 + "Diff-Drive UGV / Jetbot Task3 World 统计报告")
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


def random_root_positions(cfg: Task3WorldConfig, num_envs: int, device: str) -> torch.Tensor:
    root = torch.empty((num_envs, 2), dtype=torch.float32, device=device)
    root[:, 0] = float(cfg.x_min) + torch.rand(num_envs, dtype=torch.float32, device=device) * (
        float(cfg.x_max) - float(cfg.x_min)
    )
    root[:, 1] = float(cfg.y_min) + torch.rand(num_envs, dtype=torch.float32, device=device) * (
        float(cfg.y_max) - float(cfg.y_min)
    )
    return root


def parking_local_to_world(world: Task3WorldManager, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    yaw = world.goal_yaw
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    x = world.goal_pos[:, 0] + px * c - py * s
    y = world.goal_pos[:, 1] + px * s + py * c

    return torch.stack([x, y], dim=-1)


def make_scene_and_world(device: str):
    cfg = Task3WorldConfig()
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
        env_spacing=15.0,
    )

    spawn_world_assets(scene_cfg, cfg)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.update(0.0)

    world = Task3WorldManager(
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
    heading("[测试 0] Task3 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task3_sim2real_parking.yaml",
        PROJECT_ROOT / "src" / "diff_drive_rl" / "tasks" / "task3" / "task3_world.py",
        PROJECT_ROOT / "tests" / "task3" / "task3_world_test.py",
    ]

    missing = [str(p) for p in required if not p.exists()]
    assert not missing, "缺少 Task3 必要文件:\n" + "\n".join(missing)

    for path in required:
        print_ok(str(path.relative_to(PROJECT_ROOT)))


def test_config(cfg: Task3WorldConfig) -> None:
    heading("[测试 1] Task3WorldConfig 基础配置检测")

    cfg.validate()

    assert cfg.track_length == 10.0
    assert cfg.track_width == 3.0
    assert cfg.num_speed_bumps == 4
    assert cfg.lidar_pool_bins == 36
    assert cfg.bump_ramp_segments == 3

    assert cfg.x_min < cfg.x_max
    assert cfg.y_min < cfg.y_max
    assert cfg.spot_width_inner_nominal > 0.0
    assert cfg.spot_depth_inner_nominal > 0.0
    assert cfg.bump_height_nominal > 0.0

    print_ok(f"track_length = {cfg.track_length}")
    print_ok(f"track_width = {cfg.track_width}")
    print_ok(f"terrain_names = {cfg.terrain_names}")
    print_ok(f"num_speed_bumps = {cfg.num_speed_bumps}")
    print_ok(f"lidar_pool_bins = {cfg.lidar_pool_bins}")
    print_ok(f"risk_distance = {cfg.risk_distance}")
    print_ok("Task3WorldConfig 基础配置正常")


def test_init_assets(scene: InteractiveScene, world: Task3WorldManager) -> None:
    heading("[测试 2] Isaac 世界资产 / RigidObject 注册检测")

    rigid_names = sorted(list(scene.rigid_objects.keys()))

    if args_cli.print_assets:
        print("scene.rigid_objects:")
        for name in rigid_names:
            print(f"  - {name}")

    for i in range(world.cfg.num_speed_bumps):
        name = f"speed_bump_{i}"
        assert name in scene.rigid_objects, f"scene.rigid_objects 缺少 {name}"
        assert world.speed_bumps[i] is scene.rigid_objects[name]

    for name in ["park_back", "park_left", "park_right"]:
        assert name in scene.rigid_objects, f"scene.rigid_objects 缺少 {name}"

    assert world.park_back is scene.rigid_objects["park_back"]
    assert world.park_left is scene.rigid_objects["park_left"]
    assert world.park_right is scene.rigid_objects["park_right"]

    print_ok(f"rigid object count = {len(rigid_names)}")
    print_ok("speed_bump_0~3 / park_back / park_left / park_right 注册正常")


def test_reset_sampling(world: Task3WorldManager, scene: InteractiveScene) -> None:
    heading("[测试 3] reset_world 起点 / 泊车位 / 减速带采样检测")

    cfg = world.cfg
    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)

    world.reset_world(env_ids)
    scene.update(0.0)

    assert_finite_tensor("start_pos", world.start_pos)
    assert_finite_tensor("start_yaw", world.start_yaw)
    assert_finite_tensor("goal_pos", world.goal_pos)
    assert_finite_tensor("goal_yaw", world.goal_yaw)
    assert_finite_tensor("bump_pos", world.bump_pos)
    assert_finite_tensor("bump_height", world.bump_height)

    assert world.start_pos[:, 0].min().item() >= cfg.start_x_range[0] - 1e-5
    assert world.start_pos[:, 0].max().item() <= cfg.start_x_range[1] + 1e-5
    assert world.start_pos[:, 1].min().item() >= cfg.start_y_range[0] - 1e-5
    assert world.start_pos[:, 1].max().item() <= cfg.start_y_range[1] + 1e-5
    assert world.start_yaw.min().item() >= cfg.start_yaw_range[0] - 1e-5
    assert world.start_yaw.max().item() <= cfg.start_yaw_range[1] + 1e-5

    assert torch.allclose(
        world.goal_pos[:, 0],
        torch.full((world.num_envs,), cfg.parking_x_nominal, device=world.device),
        atol=1e-6,
    )
    assert torch.allclose(
        world.goal_pos[:, 1],
        torch.full((world.num_envs,), cfg.parking_y_nominal, device=world.device),
        atol=1e-6,
    )
    assert torch.allclose(world.goal_yaw, torch.zeros_like(world.goal_yaw), atol=1e-6)

    for i, zone in enumerate(cfg.bump_zones):
        expected_x = 0.5 * (float(zone[0]) + float(zone[1]))
        assert torch.allclose(
            world.bump_pos[:, i, 0],
            torch.full((world.num_envs,), expected_x, device=world.device),
            atol=1e-6,
        )
        assert torch.allclose(world.bump_pos[:, i, 1], torch.zeros(world.num_envs, device=world.device), atol=1e-6)

    print_stats("start_x", world.start_pos[:, 0])
    print_stats("start_y", world.start_pos[:, 1])
    print_stats("start_yaw", world.start_yaw)
    print_stats("goal_x", world.goal_pos[:, 0])
    print_stats("bump_x", world.bump_pos[:, :, 0])
    print_ok("reset_world 采样正常")


def test_domain_randomization_ranges(world: Task3WorldManager) -> None:
    heading("[测试 4] Sim2Real domain randomization buffer 范围检测")

    cfg = world.cfg
    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)

    world.reset_world(env_ids)

    checks = [
        ("asphalt_static", world.terrain_static_friction[:, 0], cfg.asphalt_static_friction_range),
        ("ice_static", world.terrain_static_friction[:, 1], cfg.ice_static_friction_range),
        ("carpet_static", world.terrain_static_friction[:, 2], cfg.carpet_static_friction_range),
        ("park_static", world.terrain_static_friction[:, 3], cfg.asphalt_static_friction_range),
        ("asphalt_dynamic", world.terrain_dynamic_friction[:, 0], cfg.asphalt_dynamic_friction_range),
        ("ice_dynamic", world.terrain_dynamic_friction[:, 1], cfg.ice_dynamic_friction_range),
        ("carpet_dynamic", world.terrain_dynamic_friction[:, 2], cfg.carpet_dynamic_friction_range),
        ("park_dynamic", world.terrain_dynamic_friction[:, 3], cfg.asphalt_dynamic_friction_range),
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
        assert tensor.min().item() >= float(rng[0]) - 1e-6, f"{name} below range: {tensor.min().item()} < {rng[0]}"
        assert tensor.max().item() <= float(rng[1]) + 1e-6, f"{name} above range: {tensor.max().item()} > {rng[1]}"
        print_ok(f"{name:<22s} range = {tensor.min().item():+.6f} ~ {tensor.max().item():+.6f}")

    dmin, dmax = cfg.action_delay_frame_range
    assert world.action_delay_frames.min().item() >= int(dmin)
    assert world.action_delay_frames.max().item() <= int(dmax)
    print_ok(f"action_delay_frames range = {world.action_delay_frames.min().item()} ~ {world.action_delay_frames.max().item()}")
    print_ok("Sim2Real DR buffer 范围正常")


def test_rigid_object_positions(world: Task3WorldManager, scene: InteractiveScene) -> None:
    heading("[测试 5] RigidObject root pose 有限性检测")

    for i, obj in enumerate(world.speed_bumps):
        pos = obj.data.root_pos_w
        quat = obj.data.root_quat_w
        assert_finite_tensor(f"speed_bump_{i}.root_pos_w", pos)
        assert_finite_tensor(f"speed_bump_{i}.root_quat_w", quat)
        print_ok(f"speed_bump_{i} root_pos_w mean = {pos.mean(dim=0).detach().cpu().numpy()}")

    for name, obj in [
        ("park_back", world.park_back),
        ("park_left", world.park_left),
        ("park_right", world.park_right),
    ]:
        pos = obj.data.root_pos_w
        quat = obj.data.root_quat_w
        assert_finite_tensor(f"{name}.root_pos_w", pos)
        assert_finite_tensor(f"{name}.root_quat_w", quat)
        print_ok(f"{name} root_pos_w mean = {pos.mean(dim=0).detach().cpu().numpy()}")

    scene.update(0.0)
    print_ok("RigidObject root pose 有限性正常")


def test_parking_geometry(world: Task3WorldManager) -> None:
    heading("[测试 6] 泊车位几何 / 坐标变换检测")

    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)
    world.reset_world(env_ids)

    root_goal = world.goal_pos.clone()
    yaw_goal = world.goal_yaw.clone()

    goal_terms = world.compute_goal_terms(root_goal, yaw_goal)
    parking_frame = world.world_to_parking_frame(root_goal)

    assert torch.norm(parking_frame, dim=-1).max().item() < 1e-5
    assert goal_terms["goal_dist"].max().item() < 1e-5
    assert torch.abs(goal_terms["goal_yaw_error"]).max().item() < 1e-5

    inside = world.is_inside_parking_box(root_goal)
    assert inside.float().mean().item() > 0.99, "goal_pos 应该在 parking box 内"

    vec_w = torch.randn((world.num_envs, 2), dtype=torch.float32, device=world.device)
    yaw = torch.rand(world.num_envs, dtype=torch.float32, device=world.device) * 2.0 * math.pi - math.pi
    vec_b = world.rotate_world_to_body_2d(vec_w, yaw)
    vec_w2 = world.rotate_body_to_world_2d(vec_b, yaw)
    inv_err = torch.norm(vec_w - vec_w2, dim=-1)

    assert inv_err.max().item() < 1e-5

    print_stats("parking frame at goal", torch.norm(parking_frame, dim=-1))
    print_stats("rotation inverse error", inv_err)
    print_ok("泊车位几何与坐标变换正常")


def test_events(world: Task3WorldManager) -> None:
    heading("[测试 7] success_candidate / out_of_lane / parking_wall_collision / bump_overlap 事件检测")

    cfg = world.cfg
    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)
    world.reset_world(env_ids)

    zero_lin = torch.zeros((world.num_envs, 3), dtype=torch.float32, device=world.device)
    zero_ang = torch.zeros((world.num_envs, 3), dtype=torch.float32, device=world.device)

    # Success: at parking goal, aligned yaw, low velocity.
    root_success = world.goal_pos.clone()
    yaw_success = world.goal_yaw.clone()
    event_success = world.check_events(root_success, yaw_success, body_lin_vel=zero_lin, body_ang_vel=zero_ang)

    assert event_success["success_candidate"].float().mean().item() > 0.99
    assert event_success["inside_parking_box"].float().mean().item() > 0.99
    assert event_success["crash"].float().mean().item() < 1e-6

    # Out of lane.
    root_oob = root_success.clone()
    root_oob[:, 1] = float(cfg.y_max) + 0.20
    event_oob = world.check_events(root_oob, yaw_success, body_lin_vel=zero_lin, body_ang_vel=zero_ang)
    assert event_oob["out_of_lane"].float().mean().item() > 0.99
    assert event_oob["crash"].float().mean().item() > 0.99

    # Parking back wall collision: put center inside the back wall rectangle.
    back_px = world.spot_depth_inner * 0.5 + float(cfg.wall_thickness) * 0.5
    back_py = torch.zeros_like(back_px)
    root_wall = parking_local_to_world(world, back_px, back_py)
    event_wall = world.check_events(root_wall, yaw_success, body_lin_vel=zero_lin, body_ang_vel=zero_ang)
    assert event_wall["parking_wall_collision"].float().mean().item() > 0.99
    assert event_wall["crash"].float().mean().item() > 0.99

    # Bump overlap.
    root_bump = world.bump_pos[:, 0, :].clone()
    yaw_bump = torch.zeros(world.num_envs, dtype=torch.float32, device=world.device)
    event_bump = world.check_events(root_bump, yaw_bump, body_lin_vel=zero_lin, body_ang_vel=zero_ang)
    assert event_bump["bump_overlap"].float().mean().item() > 0.99

    print_ok(f"success_candidate = {event_success['success_candidate'].float().mean().item():.6f}")
    print_ok(f"out_of_lane = {event_oob['out_of_lane'].float().mean().item():.6f}")
    print_ok(f"parking_wall_collision = {event_wall['parking_wall_collision'].float().mean().item():.6f}")
    print_ok(f"bump_overlap = {event_bump['bump_overlap'].float().mean().item():.6f}")
    print_ok("事件检测正常")


def test_terrain_and_milestones(world: Task3WorldManager) -> None:
    heading("[测试 8] terrain id / friction / milestones 检测")

    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)
    world.reset_world(env_ids)

    x_probe = torch.tensor([-4.0, -1.0, 1.0, 4.0], dtype=torch.float32, device=world.device)
    terrain = world.terrain_id(x_probe)
    expected = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=world.device)
    assert torch.equal(terrain, expected), f"terrain_id 错误: {terrain} != {expected}"

    one_hot = world.terrain_one_hot(x_probe)
    check_shape("terrain_one_hot", one_hot, (4, 4))
    assert torch.allclose(one_hot.sum(dim=-1), torch.ones(4, device=world.device))

    root = world.start_pos.clone()
    static_mu, dynamic_mu = world.current_friction(root[:, 0])
    assert_finite_tensor("static_mu", static_mu)
    assert_finite_tensor("dynamic_mu", dynamic_mu)

    milestone_start = world.compute_milestones(root)
    root_far = root.clone()
    root_far[:, 0] = float(world.cfg.x_max)
    milestone_far = world.compute_milestones(root_far)

    assert milestone_far["terrain_progress_count"].mean().item() >= milestone_start["terrain_progress_count"].mean().item()
    assert milestone_far["bump_progress_count"].mean().item() >= world.cfg.num_speed_bumps - 1e-5

    print_ok(f"terrain probe = {terrain.detach().cpu().tolist()}")
    print_stats("current static friction", static_mu)
    print_stats("current dynamic friction", dynamic_mu)
    print_ok(f"bump_progress_count far = {milestone_far['bump_progress_count'].mean().item():.6f}")
    print_ok("terrain / friction / milestones 正常")


def test_goal_terms(world: Task3WorldManager) -> None:
    heading("[测试 9] goal terms 维度与数值检测")

    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)
    world.reset_world(env_ids)

    root = world.start_pos.clone()
    yaw = world.start_yaw.clone()

    goal = world.compute_goal_terms(root, yaw)

    required = [
        "goal_vec_w",
        "goal_vec_b",
        "goal_dist",
        "heading_error",
        "heading_sin",
        "heading_cos",
        "goal_yaw_error",
        "goal_yaw_sin",
        "goal_yaw_cos",
        "goal_x_body_norm",
        "goal_y_body_norm",
        "goal_dist_norm",
        "parking_x",
        "parking_y",
        "parking_x_norm",
        "parking_y_norm",
    ]

    for key in required:
        assert key in goal, f"compute_goal_terms 缺少 {key}"
        assert_finite_tensor(key, goal[key])

    check_shape("goal_vec_w", goal["goal_vec_w"], (world.num_envs, 2))
    check_shape("goal_vec_b", goal["goal_vec_b"], (world.num_envs, 2))

    assert goal["goal_dist"].mean().item() > 7.0, "起点到泊车目标距离应该较远"

    print_stats("goal_dist", goal["goal_dist"])
    print_stats("heading_error abs", torch.abs(goal["heading_error"]))
    print_stats("goal_yaw_error abs", torch.abs(goal["goal_yaw_error"]))
    print_ok("goal terms 正常")


def test_lidar_processing(world: Task3WorldManager, scene: InteractiveScene) -> None:
    heading("[测试 10] analytic LiDAR / pooling / delta / noise 检测")

    cfg = world.cfg
    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)
    world.reset_world(env_ids)

    # At parking goal, facing +x, front ray should see the back wall.
    root = world.goal_pos.clone()
    yaw = world.goal_yaw.clone()

    lidar_no_noise = world.compute_analytic_lidar(
        root_pos_local=root,
        yaw=yaw,
        env_ids=env_ids,
        add_noise=False,
        update_history=True,
        normalize=False,
    )

    check_shape("lidar_no_noise", lidar_no_noise, (world.num_envs, cfg.lidar_pool_bins))
    assert_finite_tensor("lidar_no_noise", lidar_no_noise)
    assert lidar_no_noise.min().item() >= cfg.lidar_min_distance - 1e-6
    assert lidar_no_noise.max().item() <= cfg.lidar_max_distance + 1e-6

    front_idx = torch.argmin(torch.abs(world.pooled_lidar_angles))
    front_dist = lidar_no_noise[:, front_idx]

    assert front_dist.mean().item() < 1.0, (
        f"泊车位正前方应能看到背墙，但 front_dist={front_dist.mean().item():.4f}"
    )

    lidar_norm = world.compute_analytic_lidar(
        root_pos_local=root,
        yaw=yaw,
        env_ids=env_ids,
        add_noise=False,
        update_history=True,
        normalize=True,
    )
    check_shape("lidar_norm", lidar_norm, (world.num_envs, cfg.lidar_pool_bins))
    assert lidar_norm.min().item() >= 0.0
    assert lidar_norm.max().item() <= 1.0 + 1e-6

    assert world.last_lidar_delta.min().item() >= -1.0 - 1e-6
    assert world.last_lidar_delta.max().item() <= 1.0 + 1e-6

    raw = torch.arange(world.num_envs * 72, dtype=torch.float32, device=world.device).reshape(world.num_envs, 72)
    pooled = world._pool_lidar(raw)
    check_shape("pooled raw lidar", pooled, (world.num_envs, cfg.lidar_pool_bins))

    raw_short = torch.arange(world.num_envs * 12, dtype=torch.float32, device=world.device).reshape(world.num_envs, 12)
    pooled_short = world._pool_lidar(raw_short)
    check_shape("pooled short raw lidar", pooled_short, (world.num_envs, cfg.lidar_pool_bins))

    print_stats("front lidar distance", front_dist)
    print_stats("lidar_norm", lidar_norm)
    print_stats("lidar_delta", world.last_lidar_delta)
    print_ok("analytic LiDAR / pooling / delta 正常")


def test_risk_features(world: Task3WorldManager) -> None:
    heading("[测试 11] risk features 维度 / 范围 / 风险响应检测")

    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)
    world.reset_world(env_ids)

    root = torch.zeros((world.num_envs, 2), dtype=torch.float32, device=world.device)
    yaw = torch.zeros(world.num_envs, dtype=torch.float32, device=world.device)

    fake_lidar = torch.full(
        (world.num_envs, world.cfg.lidar_pool_bins),
        float(world.cfg.lidar_max_distance),
        dtype=torch.float32,
        device=world.device,
    )

    angles = world.pooled_lidar_angles
    front_mask = torch.abs(angles) <= math.radians(world.cfg.front_angle_deg)
    fake_lidar[:, front_mask] = 0.25

    risk = world.compute_risk_features(root, yaw, lidar_pooled=fake_lidar)

    check_shape("risk features", risk, (world.num_envs, world.risk_feature_dim()))
    assert_finite_tensor("risk features", risk)
    assert risk.min().item() >= -1e-6
    assert risk.max().item() <= 1.0 + 1e-6

    front_risk = risk[:, 1]
    front_clearance = risk[:, 7]

    assert front_risk.mean().item() > 0.50, "front lidar 很近时 front_risk 应较高"
    assert front_clearance.mean().item() < 0.10, "front lidar 很近时 front_clearance 应较小"

    root_edge = root.clone()
    root_edge[:, 1] = float(world.cfg.y_max) - 0.05
    risk_edge = world.compute_risk_features(root_edge, yaw, lidar_pooled=fake_lidar)
    assert risk_edge[:, 4].mean().item() > 0.50, "靠近车道边界时 lane_risk 应升高"

    root_wall = parking_local_to_world(
        world,
        world.spot_depth_inner * 0.5 + float(world.cfg.wall_thickness) * 0.5,
        torch.zeros(world.num_envs, dtype=torch.float32, device=world.device),
    )
    risk_wall = world.compute_risk_features(root_wall, yaw, lidar_pooled=fake_lidar)
    assert risk_wall[:, 5].mean().item() > 0.50, "靠近泊车墙时 parking_wall_risk 应升高"

    root_bump = world.bump_pos[:, 0, :].clone()
    risk_bump = world.compute_risk_features(root_bump, yaw, lidar_pooled=fake_lidar)
    assert risk_bump[:, 6].mean().item() > 0.50, "压到减速带时 bump_risk 应升高"

    print_stats("front risk", front_risk)
    print_stats("front clearance", front_clearance)
    print_stats("lane risk edge", risk_edge[:, 4])
    print_stats("wall risk", risk_wall[:, 5])
    print_stats("bump risk", risk_bump[:, 6])
    print_ok("risk features 正常")


def test_privileged_features(world: Task3WorldManager) -> None:
    heading("[测试 12] privileged features 维度与数值检测")

    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)
    world.reset_world(env_ids)

    root = world.start_pos.clone()
    yaw = world.start_yaw.clone()

    priv = world.compute_privileged_features(root, yaw)

    check_shape("privileged features", priv, (world.num_envs, world.privileged_feature_dim()))
    assert_finite_tensor("privileged features", priv)
    assert priv.abs().max().item() <= 10.0 + 1e-5

    assert world.risk_feature_dim() == 10
    assert world.privileged_feature_dim() == 38

    print_ok(f"risk feature dim = {world.risk_feature_dim()}")
    print_ok(f"privileged feature dim = {world.privileged_feature_dim()}")
    print_ok(f"priv range = {priv.min().item():+.6f} ~ {priv.max().item():+.6f}")
    print_ok("privileged features 正常")


def test_lidar_cfg() -> None:
    heading("[测试 13] get_lidar_cfg 配置检测")

    cfg = Task3WorldConfig()
    cfg.validate()

    lidar_cfg = get_lidar_cfg("{ENV_REGEX_NS}/Robot/chassis", cfg)

    assert lidar_cfg.prim_path == "{ENV_REGEX_NS}/Robot/chassis"
    assert abs(float(lidar_cfg.max_distance) - float(cfg.lidar_max_distance)) < 1e-6

    print_ok(f"lidar prim_path = {lidar_cfg.prim_path}")
    print_ok(f"lidar max_distance = {lidar_cfg.max_distance}")
    print_ok("get_lidar_cfg 正常")


def test_random_world_rollout(world: Task3WorldManager, scene: InteractiveScene) -> None:
    heading(f"[测试 14] 随机世界 rollout {args_cli.steps} 步稳定性检测")

    cfg = world.cfg
    env_ids = torch.arange(world.num_envs, dtype=torch.long, device=world.device)

    world.reset_world(env_ids)
    scene.update(0.0)

    records: List[Dict[str, float]] = []
    start_time = time.time()

    collect_interval = max(100, int(args_cli.steps) // 10)

    for step in range(int(args_cli.steps)):
        root = random_root_positions(cfg, world.num_envs, world.device)
        yaw = torch.rand(world.num_envs, dtype=torch.float32, device=world.device) * 2.0 * math.pi - math.pi

        body_lin_vel = torch.randn((world.num_envs, 3), dtype=torch.float32, device=world.device) * 0.2
        body_ang_vel = torch.randn((world.num_envs, 3), dtype=torch.float32, device=world.device) * 0.2

        event = world.check_events(
            root_pos_local=root,
            yaw=yaw,
            body_lin_vel=body_lin_vel,
            body_ang_vel=body_ang_vel,
        )
        lidar = world.compute_analytic_lidar(
            root_pos_local=root,
            yaw=yaw,
            env_ids=env_ids,
            add_noise=True,
            update_history=True,
            normalize=False,
        )
        risk = world.compute_risk_features(root, yaw, lidar_pooled=lidar)
        priv = world.compute_privileged_features(root, yaw)
        goal = world.compute_goal_terms(root, yaw)
        stats = world.get_debug_stats(root, yaw)

        for name, value in event.items():
            assert_finite_tensor(f"event/{name}", value)
        for name, value in goal.items():
            assert_finite_tensor(f"goal/{name}", value)

        assert_finite_tensor("lidar rollout", lidar)
        assert_finite_tensor("risk rollout", risk)
        assert_finite_tensor("priv rollout", priv)

        if (step + 1) % collect_interval == 0 or (step + 1) == int(args_cli.steps):
            rec = {
                "step": float(step + 1),
                "GoalDist": event["goal_dist"].mean().item(),
                "YawErr": event["goal_yaw_error_abs"].mean().item(),
                "InsideBox": event["inside_parking_box"].float().mean().item(),
                "SuccessCandidate": event["success_candidate"].float().mean().item(),
                "Crash": event["crash"].float().mean().item(),
                "OutOfLane": event["out_of_lane"].float().mean().item(),
                "WallCollision": event["parking_wall_collision"].float().mean().item(),
                "BumpOverlap": event["bump_overlap"].float().mean().item(),
                "LidarMin": lidar.min().item(),
                "LidarMean": lidar.mean().item(),
                "RiskFront": risk[:, 1].mean().item(),
                "RiskLane": risk[:, 4].mean().item(),
                "RiskWall": risk[:, 5].mean().item(),
                "RiskBump": risk[:, 6].mean().item(),
                "FrontClearance": risk[:, 7].mean().item(),
                "PrivAbsMax": priv.abs().max().item(),
                **stats,
            }
            records.append(rec)

            print(
                f" -> Step {step + 1:05d} | "
                f"GoalDist={rec['GoalDist']:.2f} | "
                f"YawErr={rec['YawErr']:.2f} | "
                f"LidarMin={rec['LidarMin']:.2f} | "
                f"RiskF={rec['RiskFront']:.3f} | "
                f"RiskLane={rec['RiskLane']:.3f} | "
                f"RiskWall={rec['RiskWall']:.3f} | "
                f"RiskBump={rec['RiskBump']:.3f} | "
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
    heading("🚀 Diff-Drive UGV / Jetbot Task3 Conservative Sim2Real Parking World 全量测试启动")

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
    print_ok(f"track_length = {cfg.track_length}")
    print_ok(f"track_width = {cfg.track_width}")
    print_ok(f"num_speed_bumps = {cfg.num_speed_bumps}")
    print_ok(f"risk_feature_dim = {world.risk_feature_dim()}")
    print_ok(f"privileged_feature_dim = {world.privileged_feature_dim()}")

    test_config(cfg)
    test_init_assets(scene, world)
    test_reset_sampling(world, scene)
    test_domain_randomization_ranges(world)
    test_rigid_object_positions(world, scene)
    test_parking_geometry(world)
    test_events(world)
    test_terrain_and_milestones(world)
    test_goal_terms(world)
    test_lidar_processing(world, scene)
    test_risk_features(world)
    test_privileged_features(world)
    test_lidar_cfg()
    test_random_world_rollout(world, scene)

    print("\n💡 【Task3 World 测试解读】")
    print("1. 起点应在 x≈-4.8~-4.2，泊车位目标应固定在 nominal U 型车位中心。")
    print("2. U 型泊车位几何必须满足：goal_pos 在 parking frame 中接近 (0, 0)。")
    print("3. success_candidate 必须同时要求位置、yaw、低速、inside parking box。")
    print("4. 车道越界、泊车墙碰撞、减速带 overlap 都应能手动触发。")
    print("5. risk features 应为 10 维，privileged features 应为 38 维。")
    print("6. analytic LiDAR 必须输出 36 维 pooled lidar，并能正确更新 lidar_delta。")
    print("7. 世界层通过后，再进入 task3_config.py / task3_scene.py / task3_env.py。")

    heading("✅ Diff-Drive UGV / Jetbot Task3 Conservative Sim2Real Parking World 测试全部通过")


if __name__ == "__main__":
    try:
        run_tests()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass

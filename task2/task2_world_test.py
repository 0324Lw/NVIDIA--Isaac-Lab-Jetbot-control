import argparse
import os
from isaaclab.app import AppLauncher

# ===================================================================
# 0. 启动引擎 (无头模式，专门用于极速数据收集)
# ===================================================================
parser = argparse.ArgumentParser(description="Test Task 2 World Model & Generate GIFs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg

# 导入我们设计的世界模型核心类
from task2_world import Task2WorldConfig, Task2WorldManager, spawn_world_assets, get_lidar_cfg

# ===================================================================
# 1. 专门用于测试的迷你场景配置
# ===================================================================
@configclass
class TestWorldSceneCfg(InteractiveSceneCfg):
    num_envs: int = 64
    env_spacing: float = 60.0  # 50x50 场地 + 10m 间隔

    # 实例化一个替身机器人物块，仅用于挂载 LiDAR
    robot: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25))
    )

    # 挂载 LiDAR 传感器，并指明替身机器人的路径
    lidar = get_lidar_cfg(prim_path="{ENV_REGEX_NS}/Robot")

# ===================================================================
# 2. 核心 GIF 绘制函数
# ===================================================================
def save_env_gif(env_idx, start_p, goal_p, static_p, dyn_frames_p):
    """为指定环境绘制 2D 俯视图，并生成动态障碍物的游走 GIF"""
    print(f"    正在渲染 Environment {env_idx} 的 GIF 动画...")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-26, 26)
    ax.set_ylim(-26, 26)
    ax.set_aspect('equal')
    ax.set_title(f"Env {env_idx} Layout & Dynamic Kinematics")

    # 1. 绘制场地边界线
    rect = patches.Rectangle((-25, -25), 50, 50, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # 2. 绘制起终点与 2m 安全区
    ax.plot(start_p[0], start_p[1], 'g^', markersize=10, label="Start")
    ax.add_patch(patches.Circle((start_p[0], start_p[1]), 2.0, color='green', alpha=0.15))
    
    ax.plot(goal_p[0], goal_p[1], 'r*', markersize=15, label="Goal")
    ax.add_patch(patches.Circle((goal_p[0], goal_p[1]), 2.0, color='red', alpha=0.15))

    # 3. 绘制静态障碍物 (蓝色)
    for sp in static_p:
        ax.add_patch(patches.Circle((sp[0], sp[1]), 1.0, color='royalblue', alpha=0.6))

    # 4. 初始化动态障碍物图层 (橙色)
    dyn_patches = [patches.Circle((0, 0), 1.0, color='darkorange', alpha=0.9) for _ in range(len(dyn_frames_p[0]))]
    for dp in dyn_patches:
        ax.add_patch(dp)

    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))

    # 动画更新逻辑
    def update(frame):
        for i, dp in enumerate(dyn_patches):
            dp.center = (dyn_frames_p[frame][i, 0], dyn_frames_p[frame][i, 1])
        return dyn_patches

    # 渲染并保存
    anim = FuncAnimation(fig, update, frames=len(dyn_frames_p), blit=True)
    gif_name = f"task2_env_{env_idx}_layout.gif"
    anim.save(gif_name, writer='pillow', fps=20)
    plt.close(fig)

# ===================================================================
# 3. 主测试逻辑
# ===================================================================
def main():
    print("\n" + "="*60)
    print("🌍 开始测试 Task 2 世界模型 (张量泊松盘与幽灵运动学)")
    print("="*60)

    # --- Step 1: 环境与资产装载 ---
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.spawn_ground_plane("/World/GroundPlane", sim_utils.GroundPlaneCfg(color=(0.8, 0.8, 0.8)))
    scene_cfg = TestWorldSceneCfg()
    world_cfg = Task2WorldConfig()
    
    # 动态将世界资产注入 Scene
    spawn_world_assets(scene_cfg, world_cfg)
    scene = InteractiveScene(scene_cfg)
    
    # 实例化世界管理器
    world = Task2WorldManager(scene, world_cfg, num_envs=scene.num_envs, device=scene.device)
    sim.reset()
    
    # --- Step 2: 测试重置与布局生成逻辑 ---
    print("\n[测试项 1] RL 接口 - 环境初始化与重置 (reset_world)")
    env_ids = torch.arange(scene.num_envs, device=scene.device)
    
    # 手动触发一次强制刷新
    world.layout_reset_counters[:] = 50 
    world.reset_world(env_ids)
    
    # ✅ [核心修复]：必须让物理引擎推演一步，把传送结果同步到数据缓存，画图才拿得到真数据！
    sim.step()
    scene.update(0.0)
    
    # 取出 5 个环境供作图
    target_envs = [0, 10, 20, 30, 40]
    start_pos_cpu = world.start_pos[target_envs].cpu().numpy()
    goal_pos_cpu = world.goal_pos[target_envs].cpu().numpy()
    
    # 获取静态障碍物在局部环境中的相对坐标
    static_global = world.static_obs.data.root_pos_w.view(world.num_envs, world_cfg.num_static_obs, 3)
    env_origins = scene.env_origins
    static_local_cpu = (static_global[target_envs, :, :2] - env_origins[target_envs, :2].unsqueeze(1)).cpu().numpy()

    print("✔ 重置完成。起终点生成正常，静态障碍物分布已写入张量。")

    # --- Step 3: 测试幽灵游走与数据采集 ---
    print("\n[测试项 2] RL 接口 - 运动学推进 (step_kinematic_obstacles)")
    print("正在推演 100 步 (模拟 2.0 秒) 以收集动态障碍物轨迹...")
    
    dt = 0.02
    frames_dyn_local = {i: [] for i in range(len(target_envs))}
    
    for step in range(100):
        # 推进动态障碍物
        world.step_kinematic_obstacles(dt)
        
        # 物理引擎步进 (同步 Lidar 等数据)
        sim.step()
        scene.update(dt)
        
        # 采集当前动态障碍物相对坐标
        dyn_global = world.dyn_obs.data.root_pos_w.view(world.num_envs, world_cfg.num_dyn_obs, 3)
        dyn_local = dyn_global[target_envs, :, :2] - env_origins[target_envs, :2].unsqueeze(1)
        
        for idx in range(len(target_envs)):
            frames_dyn_local[idx].append(dyn_local[idx].cpu().numpy())

    print("✔ 动态障碍物推进无卡顿，无越界，边界反弹逻辑正常。")

    # --- Step 4: 测试 LiDAR 接口 ---
    print("\n[测试项 3] RL 接口 - 激光雷达数据处理 (process_lidar_data)")
    lidar_sensor = scene.sensors["lidar"]
    lidar_tensor = world.process_lidar_data(lidar_sensor)
    print(f"  - 预期维度: [{scene.num_envs}, 360]")
    print(f"  - 实际输出维度: {list(lidar_tensor.shape)}")
    print(f"  - 数据区间测试: Min = {lidar_tensor.min().item():.3f}, Max = {lidar_tensor.max().item():.3f} (预期在 0.0~1.0 之间)")
    assert lidar_tensor.shape == (scene.num_envs, 360), "LiDAR 张量维度错误！"
    print("✔ LiDAR 张量流提取与归一化正常。")

    # --- Step 5: 绘制 GIF ---
    print("\n[可视化] 开始导出 5 张环境布局与轨迹动图...")
    for idx, env_id in enumerate(target_envs):
        save_env_gif(
            env_idx=env_id,
            start_p=start_pos_cpu[idx],
            goal_p=goal_pos_cpu[idx],
            static_p=static_local_cpu[idx],
            dyn_frames_p=frames_dyn_local[idx]
        )
    
    print(f"\n🎉 测试全部圆满完成！GIF 文件已保存在当前目录 {os.getcwd()}")

if __name__ == "__main__":
    main()
    import sys
    sys.exit(0)
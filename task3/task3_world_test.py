import argparse
import torch
import math
import numpy as np
import sys  # ⬅️ 引入 sys 用于安全退出
from isaaclab.app import AppLauncher

# ===================================================================
# 0. 启动底层引擎
# ===================================================================
parser = argparse.ArgumentParser(description="Task 3 UGV Closed-Loop Test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# 测试阶段建议开启 GUI，方便直观观察物理碰撞和减速带跨越
args_cli.headless = False 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz  # ⬅️ 新增，用于初始化朝向
from isaaclab.sensors import RayCasterCfg, patterns

from task3_world import Task3WorldConfig, Task3WorldManager, spawn_world_assets, get_lidar_cfg
# ===================================================================
# 1. 场景配置 (集成世界模型 + 无人车 + 雷达)
# ===================================================================
@configclass
class Task3TestSceneCfg(InteractiveSceneCfg):
    num_envs: int = 4
    env_spacing: float = 15.0

    # 1. 挂载无人车资产 (强制使用你 Task 1 中验证过可用的确切云端路径)
    ugv_robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            # ✅ 换回你之前成功跑通的绝对网络路径 (2023.1.1 版本)
            usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Robots/Jetbot/jetbot.usd", 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, 
                max_depenetration_velocity=10.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, 
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05), # 初始高度离地 5cm
        ),
        # 定义底层关节控制器 
        # (注意：这里必须保留 left/right_wheel_joint 独立列表，以便于我们在闭环测试中下发差速运动学指令，而不是像任务1那样只用 .* 模糊匹配)
        actuators={
            "velocity_actuator": ImplicitActuatorCfg(
                joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
                effort_limit_sim=400.0,     # ✅ 采用你 Task 1 的参数，消除弃用警告
                velocity_limit_sim=100.0,   # ✅ 采用你 Task 1 的参数
                stiffness=0.0, 
                damping=1e5,               # ✅ 采用你 Task 1 的阻尼值
            )
        }
    )

    # 2. 挂载激光雷达 (绑定到无人车底盘)
   # 2. 挂载激光雷达 (强制抬高至 15cm 确保超越 Jetbot 车顶边界)
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis", # 直接挂在底盘
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)), # ✅ 抬高 15cm
        ray_alignment="yaw",
        pattern_cfg=patterns.BpearlPatternCfg(),
        debug_vis=False, # ✅ 临时开启雷达可视化，在 GUI 中看红线是否打在墙上
        mesh_prim_paths=["/World"],
        max_distance=10.0,
    )

def calculate_differential_kinematics(v: torch.Tensor, omega: torch.Tensor, wheel_radius: float = 0.03, wheel_base: float = 0.15) -> torch.Tensor:
    """
    Sim2Real 核心：将线速度和角速度 (cmd_vel) 转化为左右轮的关节角速度
    v: 线速度 [num_envs, 1]
    omega: 角速度 [num_envs, 1]
    """
    # 计算左右轮线速度
    v_left = v - (omega * wheel_base) / 2.0
    v_right = v + (omega * wheel_base) / 2.0
    
    # 转化为关节角速度 (rad/s)
    w_left = v_left / wheel_radius
    w_right = v_right / wheel_radius
    
    return torch.cat([w_left.unsqueeze(1), w_right.unsqueeze(1)], dim=1)

def main():
    print("\n" + "="*80)
    print("🚀 开始 Task 3 闭环驱动与物理引擎测试")
    print("="*80)

    # 1. 初始化引擎与场景
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = Task3TestSceneCfg()
    world_cfg = Task3WorldConfig()
    
    spawn_world_assets(scene_cfg, world_cfg)
    scene = InteractiveScene(scene_cfg)
    world = Task3WorldManager(scene, world_cfg, scene.num_envs, "cuda:0")
    
    # 获取机器人句柄
    robot: Articulation = scene.articulations["ugv_robot"]
    
    sim.reset()
    env_ids = torch.arange(scene.num_envs, device="cuda:0")
    
    # 2. 初始化世界模型坐标并传送机器人
    world.reset_world(env_ids)
    
    robot_state = robot.data.default_root_state[env_ids].clone()
    
    # ✅ [核心修复 1] 必须加上 scene.env_origins，将局部坐标转为绝对全局坐标
    robot_state[:, 0:2] = world.start_pos + scene.env_origins[:, 0:2]
    robot_state[:, 2] = 0.05 # 离地高度
    
    # ✅ [核心修复 2] 正确初始化小车朝向 (面向终点)
    robot_state[:, 3:7] = quat_from_euler_xyz(
        torch.zeros_like(world.start_yaw), 
        torch.zeros_like(world.start_yaw), 
        world.start_yaw
    )
    
    robot.write_root_state_to_sim(robot_state, env_ids)
    
    sim.step()
    scene.update(0.0)
    print("[✔] 物理引擎就绪，障碍物网格化生成无干涉。")

    # 3. 闭环控制测试循环 (模拟 RL 的 step)
    num_steps = 1000
    control_dt = 0.02 # 50Hz 控制频率
    step_count = 0

    print("⏳ 开始驱动模拟...")
    while simulation_app.is_running() and step_count < num_steps:
        # --- A. 传感器读取与状态检测 ---
        current_pos = robot.data.root_pos_w[:, 0:2] - scene.env_origins[:, 0:2]
        
        # 测试雷达数据提取
        lidar_distances = world.process_lidar_data(scene.sensors["lidar"])
        min_lidar_dist = torch.min(lidar_distances[0]).item()
        
        # --- B. 简易状态机控制策略 ---
        # 我们对 Env 0 施加一个简单的导航逻辑，测试物理反馈
        v_cmd = torch.zeros(scene.num_envs, device="cuda:0")
        omega_cmd = torch.zeros(scene.num_envs, device="cuda:0")
        
        for i in range(scene.num_envs):
            x, y = current_pos[i, 0], current_pos[i, 1]
            goal_x, goal_y = world.goal_pos[i, 0], world.goal_pos[i, 1]
            
            # 阶段 1：直线穿越减速带区
            if x < 2.0:
                v_cmd[i] = 0.8  # 0.8 m/s 直线加速
                # 简单的 P 控制保持在 Y=0 中轴线
                omega_cmd[i] = -0.5 * y 
            # 阶段 2：进入标准区，准备转向泊车
            else:
                dist_to_goal = math.hypot(goal_x - x, goal_y - y)
                if dist_to_goal > 0.2:
                    v_cmd[i] = 0.3 # 减速靠近
                    # 简单的朝向 P 控制 (此处简化，假定已知相对角度)
                    omega_cmd[i] = 0.5 
                else:
                    v_cmd[i] = 0.0 # 停车
                    omega_cmd[i] = 0.0

        # --- C. 运动学逆解与物理驱动 ---
        # 将线速度/角速度转换为底层电机的驱动指令
        joint_vel_targets = calculate_differential_kinematics(v_cmd, omega_cmd)
        
        # 写入物理引擎的控制器
        robot.set_joint_velocity_target(joint_vel_targets)
        scene.write_data_to_sim()
        # --- D. 引擎步进 ---
        sim.step()
        scene.update(control_dt)
        step_count += 1
        
        # ✅ [增加详尽打印] 显示 XY 坐标、最近障碍物距离、下发的控制量
        if step_count % 50 == 0:
            print(f"Step {step_count:04d} | Env 0 位置 -> X: {current_pos[0, 0].item():.2f}m, Y: {current_pos[0, 1].item():.2f}m | 雷达最近障碍: {min_lidar_dist:.2f}m | 指令: v={v_cmd[0]:.2f}, w={omega_cmd[0]:.2f}")

    print("🎉 闭环测试结束。")

if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0)  # ✅ [防卡死修复] 强制杀掉进程，避免 GUI 锁死
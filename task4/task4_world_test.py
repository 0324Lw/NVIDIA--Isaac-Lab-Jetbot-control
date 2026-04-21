import argparse
import sys
import torch
import math

# ===================================================================
# 0. 启动底层引擎 (强制无头极速模式)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 4 MAPPO World Model Headless Test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # ✅ 强制无头模式，关闭 GUI 渲染，极速推演
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi

# 导入任务 4 世界模型
from task4_world import Task4WorldConfig, Task4WorldManager, spawn_world_assets

# ===================================================================
# 1. 专门用于多智能体测试的场景配置
# ===================================================================
# 提取共享配置，避免引擎误认
shared_jetbot_actuator = ImplicitActuatorCfg(
    joint_names_expr=[".*wheel_joint"],
    effort_limit_sim=400.0,
    velocity_limit_sim=100.0,
    stiffness=0.0,
    damping=10.0,
)

shared_jetbot_spawn = sim_utils.UsdFileCfg(
    usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Robots/Jetbot/jetbot.usd",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=10.0),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
    ),
)

@configclass
class Task4TestSceneCfg(InteractiveSceneCfg):
    num_envs: int = 16
    env_spacing: float = 25.0

    robot_0: ArticulationCfg = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot_0", spawn=shared_jetbot_spawn, actuators={"wheels": shared_jetbot_actuator})
    robot_1: ArticulationCfg = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot_1", spawn=shared_jetbot_spawn, actuators={"wheels": shared_jetbot_actuator})
    robot_2: ArticulationCfg = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot_2", spawn=shared_jetbot_spawn, actuators={"wheels": shared_jetbot_actuator})
    robot_3: ArticulationCfg = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot_3", spawn=shared_jetbot_spawn, actuators={"wheels": shared_jetbot_actuator})

def calculate_differential_kinematics(v: torch.Tensor, omega: torch.Tensor, wheel_radius: float = 0.03, wheel_base: float = 0.15) -> torch.Tensor:
    v_left = v - (omega * wheel_base) / 2.0
    v_right = v + (omega * wheel_base) / 2.0
    w_left = v_left / wheel_radius
    w_right = v_right / wheel_radius
    return torch.cat([w_left.unsqueeze(1), w_right.unsqueeze(1)], dim=1)

# ===================================================================
# 2. 主测试逻辑
# ===================================================================
def main():
    print("\n" + "="*80)
    print("🌍 开始 Task 4 十字路口多智能体世界模型测试 (无头高频侦测版)")
    print("="*80)

    # --- Step 1: 环境初始化 ---
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = Task4TestSceneCfg()
    world_cfg = Task4WorldConfig()
    
    spawn_world_assets(scene_cfg, world_cfg)
    scene = InteractiveScene(scene_cfg)
    world = Task4WorldManager(scene, world_cfg, num_envs=scene.num_envs, device="cuda:0")
    
    robots = [scene.articulations[f"robot_{i}"] for i in range(4)]
    
    # --- Step 2: 随机重置与分配 ---
    sim.reset()
    env_ids = torch.arange(scene.num_envs, device="cuda:0")
    world.reset_world(env_ids)
    
    for i in range(4):
        state = robots[i].data.default_root_state[env_ids].clone()
        state[:, 0:2] = world.start_pos[:, i, :] + scene.env_origins[:, 0:2]
        state[:, 2] = 0.05
        state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(env_ids), torch.zeros_like(env_ids), world.start_yaw[:, i])
        robots[i].write_root_state_to_sim(state, env_ids)
        
    scene.write_data_to_sim()
    sim.step()
    scene.update(0.0)
    print("\n[✔] 世界模型生成与重置完成。开始高频物理推演...\n")

    # --- Step 3: 异构控制器与高密度日志输出 ---
    # 异构速度设定：N车:1.0, S车:0.8, E车:1.2, W车:0.6
    max_speeds = torch.tensor([1.0, 0.8, 1.2, 0.6], device="cuda:0") 
    
    wheel_ids = []
    for r in robots:
        w_id, _ = r.find_joints(".*wheel_joint")
        wheel_ids.append(w_id)

    control_dt = 0.02
    step_count = 0
    total_steps = 1500 # 约 30 秒物理时间
    
    # 我们只重点观察 Environment 0，避免日志刷屏
    target_env = 0
    car_labels = ["Robot 0 (N)", "Robot 1 (S)", "Robot 2 (E)", "Robot 3 (W)"]
    
    while simulation_app.is_running() and step_count < total_steps:
        for i in range(4):
            curr_pos = robots[i].data.root_pos_w[:, 0:2] - scene.env_origins[:, 0:2]
            goal_pos = world.goal_pos[:, i, :]
            
            diff = goal_pos - curr_pos
            dist = torch.norm(diff, dim=-1)
            goal_yaw = torch.atan2(diff[:, 1], diff[:, 0])
            
            _, _, curr_yaw = euler_xyz_from_quat(robots[i].data.root_quat_w)
            yaw_error = wrap_to_pi(goal_yaw - curr_yaw)
            
            v_cmd = torch.where(dist > 0.3, max_speeds[i], 0.0)
            w_cmd = 2.0 * yaw_error
            v_cmd = torch.where(torch.abs(yaw_error) > 0.5, v_cmd * 0.3, v_cmd)
            
            joint_vels = calculate_differential_kinematics(v_cmd, w_cmd)
            robots[i].set_joint_velocity_target(joint_vels, joint_ids=wheel_ids[i])
            
        scene.write_data_to_sim()
        sim.step()
        scene.update(control_dt)
        step_count += 1
        
        # ✅ 每 100 步 (物理时间约 2 秒)，输出一次 Env 0 的极其详细的运行状态
        if step_count % 100 == 0:
            print(f"\n[{step_count:04d} / {total_steps}] 物理流逝时间: {step_count * control_dt:.2f}s")
            print("-" * 75)
            for i in range(4):
                # 提取 Env 0 的瞬时数据
                c_pos = robots[i].data.root_pos_w[target_env, 0:2] - scene.env_origins[target_env, 0:2]
                s_pos = world.start_pos[target_env, i, :]
                g_pos = world.goal_pos[target_env, i, :]
                
                # 计算各种遥测距离
                travel_dist = torch.norm(c_pos - s_pos, dim=-1).item()
                dist_to_goal = torch.norm(g_pos - c_pos, dim=-1).item()
                
                # 提取物理底盘的真实线速度矢量并求模长
                lin_vel_w = robots[i].data.root_lin_vel_w[target_env, 0:2]
                real_speed = torch.norm(lin_vel_w, dim=-1).item()
                
                # 智能状态判定
                if dist_to_goal <= 0.3:
                    status = "✅ 抵达终点"
                elif real_speed < 0.05 and step_count > 150: # 给起步一点时间
                    status = "💥 碰撞/死锁"
                else:
                    status = "🚀 移动中  "
                
                print(f"{car_labels[i]:<12} | 状态: {status} | 已行进: {travel_dist:5.2f}m | 距终点: {dist_to_goal:5.2f}m | 真实速度: {real_speed:4.2f}m/s")
            print("-" * 75)

    print("\n🎉 无头模式极限测试圆满完成！")

if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0)
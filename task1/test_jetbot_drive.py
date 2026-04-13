import argparse
from isaaclab.app import AppLauncher

# 1. 启动引擎
parser = argparse.ArgumentParser(description="Test Jetbot Drive and Tensor API.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

def main():
    # 设置仿真步长 (0.01s = 100Hz，RL 常用频率)
    sim_dt = 0.01
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.0])

    # 铺设地板
    cfg_ground = sim_utils.GroundPlaneCfg()
    sim_utils.spawn_ground_plane("/World/GroundPlane", cfg_ground)

    # Jetbot 模型配置
    jetbot_usd_path = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Robots/Jetbot/jetbot.usd"
    
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=jetbot_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05), # 降低出生高度，直接贴地
        ),
        actuators={
            "drive": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel_joint"], 
                effort_limit_sim=400.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,                  
                damping=10.0,                   
            )
        }
    )

    # 2. 实例化机器人并重置环境
    robot = Articulation(cfg=robot_cfg)
    sim.reset()
    print("\n[INFO] 🚀 测试环境初始化完成！")

    # 3. 极其重要：在开始控制前，必须先更新一次机器人的内部张量缓存
    robot.update(sim_dt)

    # 获取轮子关节的索引 (确认张量输出顺序)
    print(f"[INFO] 检测到的可控关节: {robot.data.joint_names}")
    
    step_count = 0
    while simulation_app.is_running():
        # --- 阶段测试逻辑 (每 200 步 / 2秒 切换一次动作) ---
        phase = (step_count // 200) % 4
        
        # 初始化动作张量 [num_envs, num_joints] -> [1, 2]
        # Jetbot 默认顺序: 索引0是左轮(left_wheel)，索引1是右轮(right_wheel)
        joint_vel_targets = torch.zeros((robot.num_instances, robot.num_joints), device=sim.device)

        if phase == 0:
            # 阶段 1：全速前进 (左右轮同速正转)
            joint_vel_targets[:, 0] = 15.0  # 左轮
            joint_vel_targets[:, 1] = 15.0  # 右轮
            action_name = "全速前进 ⬆️"
        elif phase == 1:
            # 阶段 2：原地右转 (左轮正转，右轮反转)
            joint_vel_targets[:, 0] = 10.0
            joint_vel_targets[:, 1] = -10.0
            action_name = "原地右转 🔄"
        elif phase == 2:
            # 阶段 3：全速后退 (左右轮同速反转)
            joint_vel_targets[:, 0] = -15.0
            joint_vel_targets[:, 1] = -15.0
            action_name = "全速后退 ⬇️"
        elif phase == 3:
            # 阶段 4：原地左转 (左轮反转，右轮正转)
            joint_vel_targets[:, 0] = -10.0
            joint_vel_targets[:, 1] = 10.0
            action_name = "原地左转 🔄"

        # --- 执行控制指令 ---
        # 1. 将目标速度写入机器人的控制缓存
        robot.set_joint_velocity_target(joint_vel_targets)
        # 2. 将缓存数据同步到物理引擎
        robot.write_data_to_sim()
        
        # --- 推进物理仿真 ---
        sim.step()
        
        # --- 读取状态反馈 ---
        # 3. 从物理引擎拉取最新状态到张量缓存
        robot.update(sim_dt)

        if step_count % 50 == 0:
            # 读取底盘全局坐标 [x, y, z]
            root_pos = robot.data.root_pos_w[0]
            print(f"步数: {step_count:04d} | 动作: {action_name} | 小车坐标: X={root_pos[0]:.2f}, Y={root_pos[1]:.2f}")

        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()
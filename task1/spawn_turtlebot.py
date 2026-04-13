import argparse
from isaaclab.app import AppLauncher

# 1. 启动引擎
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])

    # 铺地板
    cfg_ground = sim_utils.GroundPlaneCfg()
    sim_utils.spawn_ground_plane("/World/GroundPlane", cfg_ground)

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
            pos=(0.0, 0.0, 0.2), 
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

    print("\n" + "="*60)
    print("⏳ [系统提示] 准备实例化机器人...")
    print("="*60 + "\n")

    # 这一步是假死的元凶！
    robot = Articulation(cfg=robot_cfg)
    sim.reset()
    
    print("\n[INFO] 下载完成")
    
    step_count = 0
    while simulation_app.is_running():
        sim.step()
        # 每跑 100 步打印一次心跳，证明循环没有卡死
        if step_count % 100 == 0:
            print(f"✅ 物理仿真稳定运行中... 当前步数: {step_count}")
        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()

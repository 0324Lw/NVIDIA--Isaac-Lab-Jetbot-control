import argparse
import time
import torch
import numpy as np
from isaaclab.app import AppLauncher

# ===================================================================
# 0. 启动引擎 (GUI 模式)
# ===================================================================
parser = argparse.ArgumentParser(description="Task 2 UGV Navigation Model Test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# 确保打开 GUI 界面
args_cli.headless = False 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入 (必须在引擎启动后)
# ===================================================================
from stable_baselines3 import PPO
from task2_env import JetbotObstacleEnv, Task2Config

def main():
    print("\n" + "="*80)
    print("🚀 启动 Task 2 模型演示：动态避障与自主导航")
    print("="*80)

    # 1. 配置加载 (设置为单环境演示)
    cfg = Task2Config()
    cfg.num_envs = 1  
    
    # 2. 实例化环境
    # 注意：环境内部会自动创建 WorldManager 和障碍物布局
    env = JetbotObstacleEnv(cfg)
    
    # 3. 加载训练好的 SB3 模型
    # 请根据你实际保存的文件名修改路径，例如：ppo_jetbot_final.zip
    model_path = "runs/task2_sb3/checkpoints/ppo_jetbot_step_300000"
    
    try:
        # 使用 CPU 加载模型权重（演示不需要高频训练）
        model = PPO.load(model_path)
        print(f"✅ 成功加载模型: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败，请检查路径。错误: {e}")
        simulation_app.close()
        return

    # 4. 开始演示循环
    print("\n🎬 正在生成随机环境布局，演示即将开始...")
    obs, _ = env.reset()
    
    # 将视角锁定在小车上方或斜后方 (可选)
    # env.sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])

    step_count = 0
    try:
        while simulation_app.is_running():
            # 1. 神经网络推理 (去掉堆叠维度的 Numpy 转换)
            obs_cpu = obs.cpu().numpy()
            action, _ = model.predict(obs_cpu, deterministic=True)
            # 2. 环境步进
            # actions 需要是张量格式传给 env.step
            action_tensor = torch.tensor(action, device=env.device)
            obs, rewards, dones, timeouts, info = env.step(action_tensor)
            
            # 3. 打印实时状态
            if step_count % 20 == 0:
                dist = info["reward_components"]["approach"] # 示意距离变化
                print(f"Step: {step_count:04d} | 目标距离: {env.last_distances[0]:.2f}m")

            # 4. 慢动作控制
            # 通过增加微小的 sleep 来让演示更加易于观察
            # 默认控制频率是 20Hz (0.05s)，增加 0.02s 延迟可实现“慢动作”
            time.sleep(0.02)
            
            step_count += 1
            
            # 5. 到达终点或发生碰撞后停留片刻再重置
            if dones[0]:
                if info["success_rate"] > 0:
                    print("🎉 任务成功：顺利到达终点！")
                elif info["crash_rate"] > 0:
                    print("💥 任务失败：发生碰撞或越界。")
                else:
                    print("⏰ 任务超时。")
                
                print("等待 2 秒后自动刷新环境布局...")
                time.sleep(2.0)
                obs, _ = env.reset()
                step_count = 0

    except KeyboardInterrupt:
        print("\n🛑 演示被用户手动停止。")

if __name__ == "__main__":
    main()
    simulation_app.close()
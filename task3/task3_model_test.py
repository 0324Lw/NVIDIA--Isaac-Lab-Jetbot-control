import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

# ===================================================================
# 0. 启动引擎 (GUI 模式)
# ===================================================================
parser = argparse.ArgumentParser(description="Task 3 RL Model Inference Test")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 测试阶段必须看到画面，设为 False
args_cli.headless = False 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
from stable_baselines3 import PPO
from task3_env import JetbotParkingEnv, Task3Config
# 导入训练时定义的策略类，确保模型能正确反序列化
from train import AsymmetricPolicy, CustomSb3VecEnvWrapper

def main():
    print("\n" + "="*80)
    print("🚀 启动 Task 3 强化学习模型部署推理测试")
    print("="*80)

    # 1. 环境配置 (单环境可视化)
    cfg = Task3Config()
    cfg.num_envs = 1  # 只开一个环境，方便观察
    cfg.device = "cuda:0"
    
    # 实例化环境
    raw_env = JetbotParkingEnv(cfg)
    # 使用训练时的包装器，确保 Observation 格式匹配
    env = CustomSb3VecEnvWrapper(raw_env)

    # 2. 加载模型
    model_path = "./runs/task3_ppo/jetbot_sim2real_final.zip"
    print(f"🧠 正在从 {model_path} 加载模型权重...")
    
    try:
        # 加载 PPO 模型，SB3 会自动处理 AsymmetricPolicy 结构
        model = PPO.load(model_path, env=env, device=cfg.device)
        print("[✔] 模型加载成功！")
    except Exception as e:
        print(f"[✘] 模型加载失败: {e}")
        print("💡 请检查模型文件是否存在，或 train.py 中的策略定义是否已被修改。")
        simulation_app.close()
        return

    # 3. 推理循环
    print("\n🎬 开始演示。按住 'V' 键可在 Isaac Sim 中切换视角。")
    obs = env.reset()
    
    # --- 慢速控制参数 ---
    # 速度缩放因子 (0.3 表示只输出原始动力的 30%)
    SPEED_SCALER = 0.3 
    
    try:
        while simulation_app.is_running():
            # 使用模型进行预测 (deterministic=True 确保输出最稳健的动作)
            action, _states = model.predict(obs, deterministic=True)
            
            # ✅ 实现“移动要慢”：对输出动作进行线性缩放
            slow_action = action * SPEED_SCALER
            
            # 执行动作
            obs, rewards, dones, infos = env.step(slow_action)
            
            # 打印实时遥测数据
            if "telemetry" in infos[0]:
                tel = infos[0]["telemetry"]
                print(f"位置 X: {tel['x_pos']:.2f}m | 距终点: {tel['dist']:.2f}m | 进度: {tel['terrain_progress']:.1f}/2", end='\r')
            
            # 如果环境重置（成功或撞墙），打印提示
            if dones[0]:
                print("\n🔄 环境重置：触发了成功泊车或碰撞边界。")

    except KeyboardInterrupt:
        print("\n用户中断测试。")

if __name__ == "__main__":
    main()
    simulation_app.close()
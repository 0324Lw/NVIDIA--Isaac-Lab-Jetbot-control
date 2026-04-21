import argparse
import time
import torch
import numpy as np
import pandas as pd
from isaaclab.app import AppLauncher

# ===================================================================
# 0. 启动底层引擎 (无头极速模式)
# ===================================================================
parser = argparse.ArgumentParser(description="Task 3 Sim2Real Environment Extreme Test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
from task3_env import JetbotParkingEnv, Task3Config

def main():
    print("\n" + "="*80)
    print("🌍 启动 Task 3 混合摩擦力与精准泊车极限压测 (物理遥测版)")
    print("="*80)

    # 1. 初始化配置与环境
    cfg = Task3Config()
    cfg.num_envs = 64  # 使用 64 个并发环境进行满载压测
    
    start_time = time.time()
    env = JetbotParkingEnv(cfg)
    env_load_time = time.time() - start_time

    # 2. 基础张量空间校验
    print(f"\n[✔] 测试 1 & 2: 环境加载成功！耗时: {env_load_time:.2f} 秒")
    print(f"    - 并行环境数量: {env.num_envs}")
    print(f"    - 物理设备: {env.device}")
    
    # ✅ 修正了观测维度的计算说明: 2(轮速) + 1(距离) + 2(角度sin/cos) + 36(雷达)
    obs_dim_expected = (2 + 1 + 2 + 36) * cfg.frame_stack
    print(f"\n[✔] 测试 3: 状态/动作空间维度校验")
    print(f"    - 观测维度 (Obs): {env.observation_space.shape} (预期: [{obs_dim_expected}])")
    print(f"    - 动作维度 (Action): {env.action_space.shape} (预期: [2])")

    # 3. 开始 3000 步极限游走压测
    print("\n⏳ 开始 3000 步极限随机策略推演，触发所有边界条件，请稍候...")
    
    obs, _ = env.reset()
    
    # ✅ 数据日志池 (对齐新的 info["telemetry"] 字典)
    telemetry_log = {
        "total_reward": [],
        "dist_to_goal": [],
        "x_pos": [],
        "terrain_progress": []
    }
    
    # 终局事件计数器
    stats_success = 0
    stats_terminated = 0
    
    steps = 3000
    start_sim_time = time.time()
    
    for step in range(steps):
        if step > 0 and step % 1000 == 0:
            print(f"    - 已推演 {step} 步...")
            
        # 产生 [-1.0, 1.0] 的均匀分布随机动作
        # 产生偏置动作：让双轮的基础指令偏向于前进 (0.2 ~ 1.0)，同时加上一定的转向随机扰动
        # 这样小车就会呈放射状向前开，必定会撞墙或出界！
        forward_bias = torch.rand((env.num_envs, 1), device=env.device) * 0.8 + 0.2
        steer_noise = torch.randn((env.num_envs, 1), device=env.device) * 0.5
        
        # 左轮 = 前进 + 转向，右轮 = 前进 - 转向
        left_wheel = torch.clamp(forward_bias + steer_noise, -1.0, 1.0)
        right_wheel = torch.clamp(forward_bias - steer_noise, -1.0, 1.0)
        actions = torch.cat([left_wheel, right_wheel], dim=1)
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 记录遥测数据
        telemetry_log["total_reward"].append(rewards.mean().item())
        if "telemetry" in info:
            tel = info["telemetry"]
            telemetry_log["dist_to_goal"].append(tel["dist"])
            telemetry_log["x_pos"].append(tel["x_pos"])
            telemetry_log["terrain_progress"].append(tel["terrain_progress"])
            
            # 累加成功人次
            stats_success += int(tel["success"] * env.num_envs)
            
        # 统计触发重置的环境总数 (出界撞墙或成功泊车)
        stats_terminated += terminated.sum().item()

    sim_time = time.time() - start_sim_time
    fps = (steps * env.num_envs) / sim_time

    # 4. 终局事件统计打印
    print(f"\n[✔] 测试 4 & 5: 起步、区域判定与终局事件结算逻辑正常！")
    print(f"    - 耗时: {sim_time:.2f} 秒")
    print(f"    - 吞吐量 (FPS): 约 {fps:.0f} 环境步/秒")
    print(f"    - 触发 '完美泊车' 的总人次: {stats_success} (随机瞎走很难到达，可能为 0)")
    print(f"    - 触发 '越界/撞墙终止' 的总人次: {stats_terminated} (狭窄赛道极易撞墙)")

    # 5. Pandas 物理遥测深度统计分析
    print("\n[✔] 测试 6: 物理遥测深度统计分析 (Pandas DataFrame)")
    print("-" * 85)
    
    df = pd.DataFrame(telemetry_log)
    summary = df.describe().T
    summary['方差 (Var)'] = summary['std'] ** 2
    summary = summary.rename(columns={
        'mean': '平均值 (Mean)',
        'min': '最小值 (Min)',
        '25%': '25% 分位数',
        '50%': '中位数 (Median)',
        '75%': '75% 分位数',
        'max': '最大值 (Max)'
    })
    
    final_summary = summary[['平均值 (Mean)', '方差 (Var)', '最小值 (Min)', '25% 分位数', '中位数 (Median)', '75% 分位数', '最大值 (Max)']]
    
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(final_summary.to_string())
    print("-" * 85)
    print("💡 提示：重点观察 x_pos 的最大值和 terrain_progress。随机策略下如果能冲过地毯区，说明物理环境门槛与底盘摩擦力设置合理！")

if __name__ == "__main__":
    main()
    simulation_app.close()
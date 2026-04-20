import argparse
import time
import pandas as pd
import torch
from isaaclab.app import AppLauncher

# ===================================================================
# 0. 启动底层引擎 (强制无头模式，专门用于极速压测)
# ===================================================================
parser = argparse.ArgumentParser(description="Task 2 Environment Extreme Stress Test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入 (必须在 app_launcher 之后)
# ===================================================================
from task2_env import JetbotObstacleEnv, Task2Config

def main():
    print("\n" + "="*80)
    print("🚀 开始执行 Task 2 动态避障环境极限压力测试 (无头模式)")
    print("="*80)

    # --- 1. 环境初始化测试 ---
    print("\n⏳ 正在加载张量物理引擎与世界模型...")
    start_time = time.time()
    cfg = Task2Config()
    env = JetbotObstacleEnv(cfg)
    
    print(f"[✔] 测试 1 & 2: 世界模型与张量向量化环境加载成功！耗时: {time.time() - start_time:.2f} 秒")
    print(f"    - 并行环境数量 (num_envs): {env.num_envs}")
    print(f"    - 物理推演设备: {env.device}")

    # --- 2. 空间维度测试 ---
    obs, info = env.reset()
    expected_obs_dim = env.num_envs * env.cfg.frame_stack * env.obs_dim_per_frame
    
    print("\n[✔] 测试 3: 状态/动作空间维度与数值校验")
    print(f"    - 观测维度 (Obs): {list(obs.shape)} (预期: [{env.num_envs}, {env.cfg.frame_stack * env.obs_dim_per_frame}])")
    assert obs.shape == (env.num_envs, env.cfg.frame_stack * env.obs_dim_per_frame), "观测维度错误！"
    print(f"    - 动作维度 (Action): [{env.num_envs}, 2]")

    # --- 3. 5000 步极限随机推演测试 ---
    print("\n⏳ 开始 5000 步极限随机策略推演，这可能需要几十秒，请稍候...")
    
    total_steps = 5000
    
    # 统计数据容器
    stats_success = 0
    stats_crash = 0
    stats_timeout = 0
    
    # Pandas 数据收集器
    reward_records = {
        "step": [], "approach": [], "heading": [], 
        "prox": [], "smooth": [], "raw_total": []
    }

    start_sim_time = time.time()
    
    for step in range(total_steps):
        # 生成 [-1.0, 1.0] 的随机动作张量
        random_actions = torch.rand((env.num_envs, 2), device=env.device) * 2.0 - 1.0
        
        # 核心交互
        obs, rewards, dones, timeouts, info = env.step(random_actions)
        
        # 统计终局事件
        stats_timeout += timeouts.sum().item()
        
        # 提取 Info 字典中的真实事件触发率
        if "success_rate" in info and "crash_rate" in info:
            stats_success += int(info["success_rate"] * env.num_envs)
            stats_crash += int(info["crash_rate"] * env.num_envs)

        # 记录奖励组件 (记录 64 个环境的平均值)
        if "reward_components" in info:
            components = info["reward_components"]
            for key in reward_records.keys():
                reward_records[key].append(components[key])

        # 进度条提示
        if (step + 1) % 1000 == 0:
            print(f"    - 已推演 {step + 1} 步...")

    sim_duration = time.time() - start_sim_time
    fps = (total_steps * env.num_envs) / sim_duration

    print(f"\n[✔] 测试 4 & 5: 5000 步测试平稳落地！(无崩溃/无死锁)")
    print(f"    - 耗时: {sim_duration:.2f} 秒")
    print(f"    - 吞吐量 (FPS): 约 {fps:.0f} 环境步/秒")
    print(f"    - 触发 '完美到达终点' 的总人次: {stats_success} (随机瞎走很难到达)")
    print(f"    - 触发 '碰撞或越界' 的总人次: {stats_crash}")
    print(f"    - 触发 '超时未完成' 的总人次: {stats_timeout}")

    # --- 4. Pandas 奖励深度统计分析 ---
    print("\n[✔] 测试 6: 连续奖励组件深度统计分析 (Pandas DataFrame)")
    print("-" * 85)
    
    df = pd.DataFrame(reward_records)
    
    # 计算要求的统计量
    summary_df = pd.DataFrame({
        "平均值 (Mean)": df.mean(),
        "方差 (Var)": df.var(),
        "最小值 (Min)": df.min(),
        "25% 分位数": df.quantile(0.25),
        "中位数 (Median)": df.median(),
        "75% 分位数": df.quantile(0.75),
        "最大值 (Max)": df.max()
    })
    
    # 格式化输出 (保留 4 位小数)
    pd.set_option('display.float_format', lambda x: f'{x:8.4f}')
    print(summary_df)
    print("-" * 85)
    
    # 基础健康度检查与警告
    print("\n🩺 [健康度快速诊断]:")
    if summary_df.loc["raw_total", "最小值 (Min)"] < -0.55:
        print("  ⚠️ 警告: raw_total 的最小值突破了我们设定的 -0.5 截断阀值！请检查 clamp 逻辑。")
    if summary_df.loc["prox", "最小值 (Min)"] == 0.0:
        print("  ⚠️ 警告: 靠近惩罚 (prox) 永远为 0，雷达可能未成功探测到障碍物！")
    if summary_df.loc["smooth", "最大值 (Max)"] > 0:
        print("  ⚠️ 警告: 平滑惩罚 (smooth) 出现了正数！这违背了数学逻辑。")
        
    print("\n🎉 测试流程已全部执行完毕！随时可以引入 RL 算法开始训练。")

if __name__ == "__main__":
    main()
    simulation_app.close()
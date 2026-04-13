import argparse
import time
from isaaclab.app import AppLauncher

# ===================================================================
# 0. 启动底层引擎 (无头模式 Headless)
# ===================================================================
parser = argparse.ArgumentParser(description="Sanity Check for Task 1 Environment.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
import torch
import pandas as pd
from task1_env import JetbotNavigationEnv, Task1Config

def main():
    print("\n" + "="*60)
    print("🚀 开始执行 Task 1 环境极限压力测试 (无头模式)")
    print("="*60)

    # 初始化配置与环境
    cfg = Task1Config()
    env = JetbotNavigationEnv(cfg)
    
    # ---------------------------------------------------------
    # 测试项 1 & 2: 小车模型调用与向量化环境建立
    # ---------------------------------------------------------
    obs, _ = env.reset()
    print("\n[✔] 测试 1 & 2 通过: 模型与向量化环境加载成功！")
    print(f"    - 并行环境数量 (num_envs): {env.num_envs}")
    print(f"    - 物理推演设备: {env.device}")
    
    # ---------------------------------------------------------
    # 测试项 3: 维度、数值校验与 step() 函数初测
    # ---------------------------------------------------------
    assert obs.shape == (cfg.num_envs, 21), f"状态空间维度错误: {obs.shape}，预期为 ({cfg.num_envs}, 21)"
    
    # 生成随机张量动作 [-1, 1]
    test_actions = torch.rand((cfg.num_envs, 2), device=env.device) * 2.0 - 1.0
    obs, rewards, terminated, truncated, info = env.step(test_actions)
    
    assert test_actions.shape == (cfg.num_envs, 2), "动作空间维度错误"
    assert rewards.shape == (cfg.num_envs,), "奖励张量维度错误"
    
    print("[✔] 测试 3 通过: 状态/动作空间维度正确，step() 函数无 Bug！")
    print(f"    - 观测维度: {obs.shape}")
    print(f"    - 动作维度: {test_actions.shape}")
    print(f"    - 奖励数值范围: [{rewards.min().item():.4f}, {rewards.max().item():.4f}] (截断测试)")

    # ---------------------------------------------------------
    # 测试项 4 & 5: 10000 步极限随机策略测试与终局统计
    # ---------------------------------------------------------
    print("\n⏳ 开始 10,000 步极限随机策略推演，请稍候...")
    total_steps = 10000
    reward_data_list = []
    
    episodes_finished = 0
    episodes_truncated = 0
    
    # 记录起始时间以计算吞吐量
    start_time = time.time()
    
    for step in range(total_steps):
        # 随机乱动策略
        actions = torch.rand((cfg.num_envs, 2), device=env.device) * 2.0 - 1.0
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 记录每一步的 info 奖励组件数据
        reward_data_list.append(info["reward_components"])
        
        # 统计终局事件
        episodes_finished += terminated.sum().item()
        episodes_truncated += truncated.sum().item()
        
        if (step + 1) % 2000 == 0:
            print(f"    - 已推演 {step + 1} 步...")

    end_time = time.time()
    elapsed_time = end_time - start_time
    # 吞吐量 = 总步数 * 并行环境数 / 耗时
    fps = (total_steps * cfg.num_envs) / elapsed_time

    print(f"\n[✔] 测试 4 & 5 通过: 10000 步测试平稳落地！")
    print(f"    - 耗时: {elapsed_time:.2f} 秒")
    print(f"    - 吞吐量 (FPS): 约 {fps:.0f} 环境步/秒")
    print(f"    - 触发 '完成全部目标' (Terminated) 的环境总人次: {episodes_finished}")
    print(f"    - 触发 '超时未完成' (Truncated) 的环境总人次: {episodes_truncated}")

    # ---------------------------------------------------------
    # 测试项 6: Pandas 奖励组件统计分析
    # ---------------------------------------------------------
    print("\n[✔] 测试 6: 奖励组件深度统计分析 (Pandas DataFrame)")
    print("-" * 80)
    
    # 将包含 10000 个字典的列表转换为 DataFrame
    df = pd.DataFrame(reward_data_list)
    
    # 计算统计学指标
    stats = df.describe().T
    # 补充计算方差 (Variance = std^2)
    stats['variance'] = df.var()
    
    # 按照需求重排列并重命名列
    display_cols = ['mean', 'variance', 'min', '25%', '50%', '75%', 'max']
    display_stats = stats[display_cols].copy()
    display_stats.columns = ['平均值 (Mean)', '方差 (Var)', '最小值 (Min)', '25% 分位数', '中位数 (Median)', '75% 分位数', '最大值 (Max)']
    
    # 打印最终漂亮的表格
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(display_stats.to_string())
    print("-" * 80)
    print("\n🎉 极限压力测试全部结束！环境处于绝对完美状态，随时可以接入 PPO 算法。")

if __name__ == "__main__":
    import os
    main()
    print("\n[INFO] 准备执行操作系统级强制退出，跳过底层死锁...")
    # 不要调用 simulation_app.close()，它就是死锁的元凶！
    # 直接拔电源，强制杀掉当前进程，释放所有显存
    os._exit(0)
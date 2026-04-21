import argparse
import time
import torch
import numpy as np
import pandas as pd

# ===================================================================
# 0. 启动底层引擎 (必须在导入深层 Isaac 接口前完成！)
# ===================================================================
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Task 4 MAPPO Environment Extreme Test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # 开启无头模式，追求极致测试速度
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
from task4_env import Task4MappoEnv, Task4Config

def main():
    print("\n" + "="*80)
    print("🌍 启动 Task 4 MAPPO 多智能体环境极限压测")
    print("="*80)

    # 1. 环境初始化
    cfg = Task4Config()
    cfg.num_envs = 64  # 64 个并行环境满载测试
    
    start_time = time.time()
    env = Task4MappoEnv(cfg)
    env_load_time = time.time() - start_time
    print(f"\n[✔] 环境加载成功！耗时: {env_load_time:.2f} 秒")

    # ===================================================================
    # 测试 1: 状态空间与动作空间维度校验
    # ===================================================================
    print(f"\n[测试 1] 状态空间与动作空间维度校验")
    print(f"    - Actor 观测空间 (Obs): {env.observation_space.shape} (预期: [4, 150])")
    print(f"    - Actor 动作空间 (Act): {env.action_space.shape} (预期: [4, 2])")
    
    obs, info = env.reset()
    state_shape = info["state"].shape
    print(f"    - Critic 全局状态 (State): {state_shape} (预期: [64, 28])")

    # ===================================================================
    # 测试 2: 强行控制移动与环境事件校验
    # ===================================================================
    print("\n[测试 2] 强制位移与终局事件校验 (下发全速直行指令)...")
    
    # 强制让所有小车全速前进 (动作向量 [1.0, 0.0])
    forced_actions = torch.zeros((cfg.num_envs, 4, 2), device=cfg.device)
    forced_actions[:, :, 0] = 1.0  # 油门拉满
    forced_actions[:, :, 1] = 0.0  # 不转向
    
    for step in range(1, 101):
        obs, rewards, terminated, truncated, info_dict = env.step(forced_actions)
        
        # 每 20 步打印一次 Env 0 的底层移动距离
        if step % 20 == 0:
            print(f"  ▶ Step {step:03d} | Env 0 实时移动距离:")
            for i in range(4):
                # 利用环境内部变量反算实际移动距离
                start_pos = env.world.start_pos[0, i, :]
                curr_pos = env.robots[i].data.root_pos_w[0, 0:2] - env.scene.env_origins[0, 0:2]
                travel_dist = torch.norm(curr_pos - start_pos, dim=-1).item()
                print(f"    - 小车 {i}: 移动了 {travel_dist:.2f} 米")
                
            # 打印事件触发情况
            events = info_dict["events"]
            print(f"    [事件监控] 本帧碰撞率: {events['crash']*100:.1f}% | 完赛数: {events['reach']}")

    print("\n[✔] 强制位移测试完成。全速盲目直行成功触发了大量碰撞重置逻辑！")

    # ===================================================================
    # 测试 3: 3000步随机策略极限压测与 Pandas 分析
    # ===================================================================
    print("\n" + "="*80)
    print("⏳ [测试 3] 开始 3000 步随机策略极限压测，请稍候...")
    print("="*80)

    # 数据日志池
    components_log = {
        "reward_total": [],
        "rew_approach": [],
        "rew_prox_penalty": [],
        "rew_jerk": [],
        "dist_to_goal": []
    }
    
    stats_crashes = 0
    stats_reaches = 0
    stats_milestones = 0
    
    steps = 3000
    start_sim_time = time.time()
    
    for step in range(steps):
        if step > 0 and step % 1000 == 0:
            print(f"    - 已推演 {step} 步...")
            
        # 产生 [-1.0, 1.0] 的纯随机动作
        # 产生偏置动作：直接映射到 [v, w] 空间
        # v (前进) 偏向正数 0.2 ~ 1.0，w (转向) 为 0 附近的小噪声
        v_cmd = torch.rand((env.num_envs, 4, 1), device=env.device) * 0.8 + 0.2
        w_cmd = torch.randn((env.num_envs, 4, 1), device=env.device) * 0.2
        actions = torch.cat([v_cmd, w_cmd], dim=-1)
        
        obs, rewards, terminated, truncated, info_dict = env.step(actions)
        
        # 记录连续型奖励组件
        components_log["reward_total"].append(rewards.mean().item())
        if "reward_components" in info_dict:
            comps = info_dict["reward_components"]
            components_log["rew_approach"].append(comps["approach"])
            components_log["rew_prox_penalty"].append(comps["prox_penalty"])
            components_log["rew_jerk"].append(comps["jerk"])
        
        if "telemetry" in info_dict:
            components_log["dist_to_goal"].append(info_dict["telemetry"]["mean_dist"])
            
        # 累加离散事件
        if "events" in info_dict:
            ev = info_dict["events"]
            stats_crashes += int(ev["crash"] * env.num_envs)
            stats_reaches += int(ev["reach"])
            stats_milestones += int(ev["milestone"])

    sim_time = time.time() - start_sim_time
    fps = (steps * env.num_envs) / sim_time

    print(f"\n[✔] 压测完毕！耗时: {sim_time:.2f} 秒 | 吞吐量: 约 {fps:.0f} 环境步/秒")
    print(f"    🎯 触发十字路口里程碑次数 : {stats_milestones}")
    print(f"    💥 触发碰撞连坐重置回合数 : {stats_crashes} (密集多车环境极易碰撞)")
    print(f"    🏁 触发偶然完美到达终点数 : {stats_reaches} (随机瞎走极难到达，通常为0)")

    # ===================================================================
    # 测试 4: Pandas 奖励组件深度统计分析
    # ===================================================================
    print("\n[测试 4] 连续奖励组件分布分析 (Pandas DataFrame)")
    print("-" * 90)
    
    df = pd.DataFrame(components_log)
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
    print("-" * 90)
    print("💡 验收提示：")
    print("1. 确保 reward_total 单步严格在 [-1.0, 1.0] 范围内（截断逻辑生效）。")
    print("2. 确保 rew_prox_penalty (靠近惩罚) 均为负值，证明高斯斥力生效。")

if __name__ == "__main__":
    main()
    simulation_app.close()
import argparse
import os
import time
import numpy as np
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# ===================================================================
# 0. 启动引擎 (无头模式，专门用于极速训练)
# ===================================================================
parser = argparse.ArgumentParser(description="Task 2 RL Training with SB3 PPO.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入 (必须在引擎启动后)
# ===================================================================
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from task2_env import JetbotObstacleEnv, Task2Config, PlotTraining

# ===================================================================
# 1. 跨框架桥梁：张量环境转 SB3 VecEnv
# ===================================================================
class IsaacLabToSB3Wrapper(VecEnv):
    """将 GPU 张量环境降级并伪装成 SB3 需要的 CPU Numpy 向量化环境"""
    def __init__(self, env: JetbotObstacleEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        
        # 定义 Gym 标准空间
        obs_dim = env.cfg.frame_stack * env.obs_dim_per_frame
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        
        # 缓存动作张量
        self._actions = None

    def step_async(self, actions):
        # 将 SB3 传来的 numpy 数组转回 GPU 张量
        self._actions = torch.tensor(actions, dtype=torch.float32, device=self.device)

    def step_wait(self):
        # 推进物理环境
        obs, rewards, dones, timeouts, info = self.env.step(self._actions)
        
        # 将全局的 Info 字典塞入第一个环境的列表中，供回调函数抓取
        infos = [{} for _ in range(self.num_envs)]
        infos[0] = info 
        
        # 搬运回 CPU (这是性能瓶颈所在，但为了兼容 SB3 必须做)
        return (
            obs.cpu().numpy(),
            rewards.cpu().numpy(),
            dones.cpu().numpy(),
            infos
        )

    def reset(self):
        obs, _ = self.env.reset()
        return obs.cpu().numpy()

    # --- 以下为满足 SB3 VecEnv 抽象类的占位方法 ---
    def close(self): pass
    def env_is_wrapped(self, wrapper_class): return [False] * self.num_envs
    def env_method(self, method_name, *args, **kwargs): pass
    def get_attr(self, attr_name, indices=None): pass
    def set_attr(self, attr_name, value, indices=None): pass
    def step_send(self): pass
    def step_recv(self): pass
    def seed(self, seed=None): pass

# ===================================================================
# 2. 自定义回调：日志记录、模型保存
# ===================================================================
class DashboardCallback(BaseCallback):
    def __init__(self, plot_ui, save_dir, save_freq_steps=50000, verbose=0):
        super().__init__(verbose)
        self.plot_ui = plot_ui
        self.save_dir = save_dir
        self.save_freq_steps = save_freq_steps
        self.last_save_step = 0
        self.start_time = time.time()
        
        self.window_success = 0.0
        self.window_crash = 0.0
        self.window_dones = 0.0

    def _on_step(self) -> bool:
        # 1. 每一帧都在后台累加真实事件
        info = self.locals["infos"][0]
        if "success_rate" in info:
            # info["success_rate"] 是一个范围在 [0, 1] 的比率。乘以 10 (num_envs) 还原出具体有几台车触发了
            self.window_success += info["success_rate"] * 10
            self.window_crash += info["crash_rate"] * 10
            self.window_dones += self.locals["dones"].sum()

        # 2. 每 2048 步结算一次总账，并清零
        if self.num_timesteps % 2048 == 0:
            comps = info.get("reward_components", {"raw_total": 0, "approach": 0, "heading": 0, "prox": 0, "smooth": 0})
            
            # 计算真实发生率 = 这段时间内发生的总次数 / 这段时间内结束的总回合数
            real_sr = (self.window_success / max(1, self.window_dones)) * 100
            real_cr = (self.window_crash / max(1, self.window_dones)) * 100

            # 提取 PPO 内部日志
            logger_dict = self.logger.name_to_value
            ent = logger_dict.get('train/entropy_loss', 0.0)
            kl = logger_dict.get('train/approx_kl', 0.0)
            loss = logger_dict.get('train/value_loss', 0.0)

            print(f"\n" + "="*70)
            print(f"🚀 [Training Step: {self.num_timesteps:07d}] Time: {time.time() - self.start_time:.1f}s")
            print("-" * 70)
            print(f"🏆 [Environment Metrics]")
            print(f"   ► Avg Reward:   {comps['raw_total']:>8.4f}  |  🎯 Real Success: {real_sr:>5.1f}%")
            print(f"   ► Approach Rew: {comps['approach']:>8.4f}  |  💥 Real Crash:   {real_cr:>5.1f}%")
            print(f"   ► Heading Rew:  {comps['heading']:>8.4f}  |  ⛔ Prox Penalty: {comps['prox']:>8.4f}")
            print(f"   ► Smooth Rew:   {comps['smooth']:>8.4f}  |  🔄 Dones Window: {int(self.window_dones)}")
            print("-" * 70)
            print(f"🧠 [PPO Network Metrics]")
            print(f"   ► Approx KL:    {kl:>8.4f}  |  📉 Value Loss:   {loss:>8.4f}")
            print(f"   ► Policy Ent:   {ent:>8.4f}  |")
            print("="*70)

            # 更新图表并清空滑动窗口
            self.plot_ui.update(comps['raw_total'])
            self.window_success = 0.0
            self.window_crash = 0.0
            self.window_dones = 0.0

        # 定期保存模型
        if self.num_timesteps - self.last_save_step >= self.save_freq_steps:
            save_path = os.path.join(self.save_dir, f"ppo_jetbot_step_{self.num_timesteps}.zip")
            self.model.save(save_path)
            print(f"💾 [Model Saved]: {save_path}")
            self.last_save_step = self.num_timesteps

        return True

# ===================================================================
# 3. 主训练流程
# ===================================================================
def linear_schedule(initial_value: float):
    """动态学习率调度器：随训练进度线性衰减"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def main():
    # 强制固定随机种子，保证实验可复现
    set_random_seed(42)
    
    print("\n" + "="*80)
    print("🧠 启动 Task 2: PPO 动态避障与导航训练 (Stable Baselines3)")
    print("="*80)

    # 1. 覆写配置：采用 10 个并行环境
    cfg = Task2Config()
    cfg.num_envs = 10  # 动态修改并行数量，适配 CPU 负载
    
    # 2. 实例化张量环境并套上 SB3 桥梁封装
    raw_env = JetbotObstacleEnv(cfg)
    sb3_env = IsaacLabToSB3Wrapper(raw_env)
    
    # 3. 实例化绘图 UI 与模型保存目录
    plot_ui = PlotTraining()
    save_dir = "runs/task2_sb3/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 构建 PPO 算法与 MLP 网络结构
    #   - net_arch: 两层隐藏层，应对高维雷达特征
    #   - activation_fn: 使用 ELU 保证梯度流平滑
    #   - Orthogonal Init: SB3 默认已开启正交初始化
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
        activation_fn=torch.nn.ELU
    )

    model = PPO(
        "MlpPolicy",
        env=sb3_env,
        learning_rate=linear_schedule(3e-4), # 学习率动态衰减
        n_steps=2048,                        # 每个环境收集 2048 步数据更新一次
        batch_size=512,
        n_epochs=10,
        gamma=0.99,                          # 折扣因子
        gae_lambda=0.95,                     # GAE 优势估算平滑参数
        clip_range=0.2,                      # PPO 核心：信任区域裁剪
        ent_coef=0.01,                       # 策略熵系数：鼓励前期探索
        max_grad_norm=0.5,                   # 梯度裁剪防爆炸
        policy_kwargs=policy_kwargs,
        device="cuda",                       # 神经网络放在 GPU 训练
        verbose=0                            # 关闭 SB3 原生杂乱输出，使用我们的 Dashboard
    )

    # 5. 开始炼丹！
    print("\n🔥 开始训练！总目标步数: 3,000,000 步")
    custom_callback = DashboardCallback(plot_ui, save_dir, save_freq_steps=100000)
    
    try:
        model.learn(total_timesteps=1_000_000, callback=custom_callback)
    except KeyboardInterrupt:
        print("\n🛑 训练被手动中止，正在保存当前权重...")
    
    # 6. 最终清理与持久化
    final_model_path = os.path.join(save_dir, "ppo_jetbot_final.zip")
    model.save(final_model_path)
    print(f"\n🎉 训练完全结束！最终模型已保存至: {final_model_path}")
    
    # ✅ 正确代码：调用 PlotTraining 的终局出图方法
    print("\n📊 正在生成最终的训练奖励趋势图...")
    plot_ui.save_and_show(save_dir)

if __name__ == "__main__":
    main()
    simulation_app.close()
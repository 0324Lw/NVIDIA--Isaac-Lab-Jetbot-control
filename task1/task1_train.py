import argparse
import os
from isaaclab.app import AppLauncher

# ===================================================================
# 0. 启动底层引擎 (强制无头模式, CUDA 训练)
# ===================================================================
parser = argparse.ArgumentParser(description="PPO Training for Jetbot Navigation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Box

# 导入 SKRL PPO 核心组件
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL

# 导入我们的环境
from task1_env import JetbotNavigationEnv, Task1Config, PlotTraining

# ===================================================================
# 1. 桥接器：将我们的自定义环境包装为 SKRL 标准接口
# ===================================================================
class SkrlEnvWrapper:
    def __init__(self, env: JetbotNavigationEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        
        # 显式告诉 SKRL：每个并行环境里只有 1 个智能体
        self.num_agents = 1 
        
        # SKRL 需要明确的 Gym spaces
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(21,))
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        self.state_space = self.observation_space
        
        # 统计信息槽
        self.total_steps = 0
        self.current_episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self.current_episode_lengths = torch.zeros(self.num_envs, device=self.device)
        self.completed_episodes = 0
        
        # 自定义指标记录
        self.hist_rewards = []
        self.hist_lengths = []
        self.hist_waypoints = []
        self.hist_finish = []
        
        self.plotter = PlotTraining()

    def reset(self):
        obs, info = self.env.reset()
        return obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self.total_steps += 1
        
        self.current_episode_rewards += reward
        self.current_episode_lengths += 1
        
        dones = terminated | truncated
        
        # 统计与控制台输出逻辑
        if dones.any():
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            num_dones = len(done_indices)
            self.completed_episodes += num_dones
            
            waypoints_reached = info["waypoint_progress"][done_indices].float().mean().item()
            
            finished_rate = terminated[done_indices].float().mean().item() * 100.0
            
            avg_reward = self.current_episode_rewards[done_indices].mean().item()
            avg_length = self.current_episode_lengths[done_indices].mean().item()
            
            self.hist_rewards.append(avg_reward)
            self.hist_lengths.append(avg_length)
            self.hist_waypoints.append(waypoints_reached)
            self.hist_finish.append(finished_rate)
            
            self.current_episode_rewards[done_indices] = 0.0
            self.current_episode_lengths[done_indices] = 0.0
            
            if self.total_steps % 100 == 0:
                print(f"[{self.total_steps:06d} 步] "
                      f"回合: {self.completed_episodes:04d} | "
                      f"平均奖励: {avg_reward:+.2f} | "
                      f"存活步数: {avg_length:03.0f} | "
                      f"吃点进度: {waypoints_reached:.1f}/5 | "
                      f"任务完成率: {finished_rate:05.1f}%")
                # self.plotter.update(avg_reward)
        
        
        # 将一维张量转换为 SKRL 强制要求的二维张量格式 [num_envs, 1]
        return obs, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info

    def close(self):
        pass

# ===================================================================
# 2. 网络架构定义 (MLP, 正交初始化)
# ===================================================================
def init_weights(m):
    if isinstance(m, nn.Linear):
        # 正交初始化：防止梯度消失/爆炸，稳定连续控制
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)

class PolicyNetwork(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False)
        
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, self.num_actions)
        )
        self.net.apply(init_weights) # 应用正交初始化
        
        # 动作标准差作为可训练参数 (初始化为 0.0，即 exp(0)=1.0 的标准差)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class ValueNetwork(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1)
        )
        self.net.apply(init_weights)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# ===================================================================
# 3. 训练主流程
# ===================================================================
def main():
    print("\n" + "="*60)
    print("🧠 启动 PPO 强化学习训练框架...")
    print("="*60)

    # 1. 初始化物理环境与包装器
    cfg = Task1Config()
    env = JetbotNavigationEnv(cfg)
    wrapped_env = SkrlEnvWrapper(env)

    # 2. 实例化神经网络
    models = {
        "policy": PolicyNetwork(wrapped_env.observation_space, wrapped_env.action_space, wrapped_env.device),
        "value": ValueNetwork(wrapped_env.observation_space, wrapped_env.action_space, wrapped_env.device)
    }

    # 3. 配置 PPO 超参数
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg["rollouts"] = 100                     # 每次采样的步数
    ppo_cfg["learning_epochs"] = 5                # 每批数据的复用次数 (PPO 核心)
    ppo_cfg["mini_batches"] = 4                   # 切分 mini-batch 降低显存压力
    ppo_cfg["discount_factor"] = 0.99             # 折扣因子
    ppo_cfg["lambda"] = 0.95                      # GAE 优势估计系数
    ppo_cfg["learning_rate"] = 3e-4               # 初始学习率
    ppo_cfg["grad_norm_clip"] = 1.0               # [重要] 梯度裁剪，防止权重崩溃
    ppo_cfg["ratio_clip"] = 0.2                   # PPO 截断范围
    
    # 启用状态和奖励的滑动平均归一化
    ppo_cfg["state_preprocessor"] = RunningStandardScaler
    ppo_cfg["state_preprocessor_kwargs"] = {"size": wrapped_env.observation_space.shape, "device": wrapped_env.device}
    ppo_cfg["value_preprocessor"] = RunningStandardScaler
    ppo_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": wrapped_env.device}
    
    # 启用基于 KL 散度的自适应学习率调度器
    ppo_cfg["learning_rate_scheduler"] = KLAdaptiveRL
    ppo_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    
    # 自动保存机制
    ppo_cfg["experiment"]["directory"] = "runs/task1_ugv"
    ppo_cfg["experiment"]["experiment_name"] = "ppo_mlp"
    ppo_cfg["experiment"]["write_interval"] = 500  # 每 500 次 rollout 保存一次

    # 4. 实例化 Memory 和 Agent (显式分配经验池)
    from skrl.memories.torch import RandomMemory
    
    # 明确告诉 SKRL 分配多大的经验回放池: [rollout步数, 环境数]
    memory = RandomMemory(memory_size=ppo_cfg["rollouts"], 
                          num_envs=wrapped_env.num_envs, 
                          device=wrapped_env.device)

    agent = PPO(models=models,
                memory=memory,  # 传入显式分配的 Memory
                cfg=ppo_cfg,
                observation_space=wrapped_env.observation_space,
                action_space=wrapped_env.action_space,
                device=wrapped_env.device)

    # 5. 配置并启动 Trainer
    trainer_cfg = {"timesteps": 40000, "headless": True, "disable_progressbar": True}
    trainer = SequentialTrainer(cfg=trainer_cfg, env=wrapped_env, agents=agent)

    print(f"\n[INFO] 训练开始！保存路径: {ppo_cfg['experiment']['directory']}/{ppo_cfg['experiment']['experiment_name']}")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[INFO] 接收到中断信号，提前终止训练。")

    print("\n🎉 训练结束！正在生成历史训练趋势图...")
    
    # 训练结束后，阻塞显示最后的绘图结果
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
    import os
    os._exit(0) # 防止物理引擎死锁
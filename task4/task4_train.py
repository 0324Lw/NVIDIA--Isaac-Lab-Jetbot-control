import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# ===================================================================
# 0. 启动引擎 (强制无头模式)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 4 Custom MAPPO Training")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
from task4_env import Task4MappoEnv, Task4Config

# ===================================================================
# 1. 网络架构定义 (正交初始化纯 MLP)
# ===================================================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化，提升早期探索稳定性"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MAPPOActor(nn.Module):
    def __init__(self, obs_dim=150, act_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 512)), nn.ELU(),
            layer_init(nn.Linear(512, 256)), nn.ELU(),
            layer_init(nn.Linear(256, 128)), nn.ELU(),
            layer_init(nn.Linear(128, act_dim), std=0.01) # 输出动作均值，增益极小防止爆炸
        )
        # 独立的可学习标准差参数
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs):
        action_mean = self.net(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        return probs

class MAPPOCritic(nn.Module):
    def __init__(self, state_dim=28, num_agents=4):
        super().__init__()
        # Critic 接收上帝视角，输出 num_agents 个独立的 Value
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 512)), nn.ELU(),
            layer_init(nn.Linear(512, 512)), nn.ELU(),
            layer_init(nn.Linear(512, 256)), nn.ELU(),
            layer_init(nn.Linear(256, num_agents), std=1.0)
        )

    def forward(self, state):
        return self.net(state)

# ===================================================================
# 2. 经验回放池 (GPU 张量化 Rollout Buffer)
# ===================================================================
class RolloutBuffer:
    def __init__(self, num_envs, num_steps, num_agents, obs_dim, state_dim, act_dim, device):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs, num_agents, obs_dim), device=device)
        self.states = torch.zeros((num_steps, num_envs, state_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, num_agents, act_dim), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs, num_agents), device=device)
        self.rewards = torch.zeros((num_steps, num_envs, num_agents), device=device)
        self.values = torch.zeros((num_steps, num_envs, num_agents), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.step = 0

    def insert(self, obs, state, action, logprob, reward, value, done):
        self.obs[self.step] = obs
        self.states[self.step] = state
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done
        self.step = (self.step + 1) % self.num_steps

    def compute_returns_and_advantages(self, next_value, next_done, gamma=0.99, gae_lambda=0.95):
        advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1].float()
                nextvalues = self.values[t + 1]
            
            # 因为 dones 是 [num_envs]，需要扩展到 [num_envs, num_agents] 以匹配 values 和 rewards
            nextnonterminal = nextnonterminal.unsqueeze(-1).expand_as(nextvalues)
            
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + self.values
        return returns, advantages

# ===================================================================
# 3. 主训练流程
# ===================================================================
def main():
    print("🌍 初始化 Task 4 自定义 MAPPO 并行环境...")
    env_cfg = Task4Config()
    env_cfg.num_envs = 256 # 256 个环境并行
    env = Task4MappoEnv(env_cfg)
    device = env.device

    # 算法超参数
    num_steps = 128        # PPO 每次 Rollout 步数
    num_epochs = 4          # PPO 更新轮数
    num_minibatches = 8     # Mini-batch 划分
    batch_size = env_cfg.num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    
    lr = 3e-4               # 初始学习率
    target_kl = 0.01        # KL 散度阈值 (用于自适应调节学习率)
    clip_coef = 0.2         # PPO 截断系数
    ent_coef = 0.01         # 策略熵系数 (鼓励探索)
    vf_coef = 0.5           # Value 损失权重
    max_grad_norm = 0.5     # 梯度裁剪

    # 实例化网络与优化器
    actor = MAPPOActor(obs_dim=150, act_dim=2).to(device)
    critic = MAPPOCritic(state_dim=28, num_agents=4).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr, eps=1e-5)
    
    # 实例化经验池
    buffer = RolloutBuffer(env_cfg.num_envs, num_steps, 4, 150, 28, 2, device)

    # 日志与保存路径
    run_dir = "./runs/task4_custom_mappo"
    os.makedirs(run_dir, exist_ok=True)
    
    global_step = 0
    total_timesteps = 300_000_000
    num_updates = total_timesteps // batch_size

    # 初始环境重置
    obs, info = env.reset()
    state = info["state"]
    next_done = torch.zeros(env_cfg.num_envs, device=device)

    print(f"\n🚀 开始MAPPO 训练! 总更新次数: {num_updates}")
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # 遥测累加器
        log_dists, log_crashes, log_success = [], 0, 0
        
        # ---------------------------------------------------------
        # A. 数据收集阶段 (Rollout)
        # ---------------------------------------------------------
        actor.eval()
        critic.eval()
        for step in range(num_steps):
            global_step += env_cfg.num_envs
            
            with torch.no_grad():
                # 动作推断 (将 [num_envs, 4, 150] 压平后传入)
                flat_obs = obs.view(-1, 150)
                probs = actor(flat_obs)
                action = probs.sample()
                logprob = probs.log_prob(action).sum(1)
                
                # Value 预测 (传入上帝视角 [num_envs, 28])
                value = critic(state)
            
            # 恢复维度
            action = action.view(env_cfg.num_envs, 4, 2)
            logprob = logprob.view(env_cfg.num_envs, 4)
            
            # 环境交互
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = info["state"]
            done = terminated | truncated
            
            # 记录遥测
            if "telemetry" in info:
                log_dists.append(info["telemetry"].get("mean_dist", 0))
            if "events" in info:
                log_crashes += int(info["events"].get("crash", 0) * env_cfg.num_envs)
                log_success += int(info["events"].get("team_success", 0) * env_cfg.num_envs)

            # 写入 Buffer (确保截断发生在存入之后)
            buffer.insert(obs, state, action, logprob, reward, value, next_done)
            
            # Bootstrap 处理：如果环境重置，下一帧的 Value 必须基于真实的 terminal_state 计算
            obs = next_obs
            state = next_state
            next_done = done
            
            if done.any():
                reset_idx = done.nonzero(as_tuple=False).squeeze(-1)
                if "terminal_state" in info:
                    state[reset_idx] = info["terminal_state"]

        # ---------------------------------------------------------
        # B. 优势函数计算 (GAE)
        # ---------------------------------------------------------
        with torch.no_grad():
            next_value = critic(state)
            returns, advantages = buffer.compute_returns_and_advantages(next_value, next_done)

        # 压平张量准备网络更新
        b_obs = buffer.obs.view(-1, 150)
        b_states = buffer.states.view(-1, 28)
        b_actions = buffer.actions.view(-1, 2)
        b_logprobs = buffer.logprobs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = buffer.values.view(-1)

        # ---------------------------------------------------------
        # C. 网络更新阶段 (PPO Epochs)
        # ---------------------------------------------------------
        actor.train()
        critic.train()
        
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        for epoch in range(num_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # Actor 损失计算 (由于包含 4 个智能体，Actor minibatch 是 4 倍大小)
                # b_obs 的索引实际上是对齐了 (env*step*4) 的，这里做一个简单扩展获取对应的智能体索引
                # 注意：b_inds 是针对 [step*env] 的索引 (大小 256,000)，每个包含 4 个智能体
                mb_agent_inds = (torch.tensor(mb_inds).view(-1, 1) * 4 + torch.arange(4)).view(-1).to(device)
                
                mb_obs = b_obs[mb_agent_inds]
                mb_actions = b_actions[mb_agent_inds]
                mb_advantages = b_advantages[mb_agent_inds]
                
                # 优势归一化 (极大地提升训练稳定性)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                probs = actor(mb_obs)
                newlogprob = probs.log_prob(mb_actions).sum(1)
                entropy = probs.entropy().sum(1)
                logratio = newlogprob - b_logprobs[mb_agent_inds]
                ratio = logratio.exp()

                # KL 散度估计 (用于 Adaptive LR)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()
                
                actor_loss = pg_loss - ent_coef * entropy_loss

                # Critic 损失计算 (Critic 接收上帝视角 mb_states)
                mb_states = b_states[mb_inds]
                newvalue = critic(mb_states) # 输出形状 [minibatch_size, 4]
                newvalue = newvalue.view(-1) # 压平成 [minibatch_size * 4] 与 b_returns 对齐
                
                mb_returns = b_returns[mb_agent_inds]
                v_loss = F.mse_loss(newvalue, mb_returns) * vf_coef

                # 整体优化
                loss = actor_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                optimizer.step()

        # ---------------------------------------------------------
        # D. KL 自适应学习率衰减
        # ---------------------------------------------------------
        if approx_kl > target_kl * 1.5:
            lr = max(1e-5, lr / 1.5)
        elif approx_kl < target_kl / 1.5:
            lr = min(1e-3, lr * 1.5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---------------------------------------------------------
        # E. 日志打印与模型保存
        # ---------------------------------------------------------
        if update % 1 == 0:
            uptime = time.time() - start_time
            mean_dist = np.mean(log_dists) if log_dists else 0
            
            print(f"\n[{update:04d}/{num_updates}] 步数: {global_step} | 耗时: {uptime:.0f}s | FPS: {int(global_step / uptime)}")
            print("-" * 80)
            print(f"🌍 环境监控:")
            print(f"   距终点均距: {mean_dist:.2f} m | 累计车祸: {log_crashes} | 团队完赛: {log_success}")
            print(f"🧠 PPO 监控:")
            print(f"   Actor Loss: {pg_loss.item():.4f} | Critic Loss: {v_loss.item():.4f} | Entropy: {entropy_loss.item():.4f}")
            print(f"   Approx KL : {approx_kl.item():.5f} | Clip Frac  : {np.mean(clipfracs):.4f}")
            print(f"   自适应 LR : {lr:.6f} | 平均价值 : {buffer.values.mean().item():.3f}")
            print("-" * 80)

        # 自动模型保存
        if update % 100 == 0:
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{run_dir}/mappo_checkpoint_{global_step}.pt")
            print(f"💾 模型已自动保存 -> {run_dir}/mappo_checkpoint_{global_step}.pt")

    print("\n🎉 MAPPO 训练圆满结束！")
    torch.save(actor.state_dict(), f"{run_dir}/mappo_actor_final.pt")

if __name__ == "__main__":
    main()
    simulation_app.close()
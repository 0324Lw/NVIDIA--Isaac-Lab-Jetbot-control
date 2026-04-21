import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from isaaclab.app import AppLauncher

# ===================================================================
# 0. 启动引擎
# ===================================================================
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from task3_env import JetbotParkingEnv, Task3Config

# ===================================================================
# 1. 非对称 Actor-Critic 核心架构
# ===================================================================
class AsymmetricMlpExtractor(nn.Module):
    """自定义特征提取器：动态分离观测空间"""
    def __init__(self, actor_in_dim: int, critic_in_dim: int, net_arch: dict):
        super().__init__()
        self.actor_in_dim = actor_in_dim
        self.latent_dim_pi = net_arch['pi'][-1]
        self.latent_dim_vf = net_arch['vf'][-1]

        # 1. 凡人视角 Actor (目前输入 123 维传感器与运动学数据)
        pi_layers = []
        last_dim = actor_in_dim
        for dim in net_arch['pi']:
            pi_layers.append(nn.Linear(last_dim, dim))
            pi_layers.append(nn.ELU())
            last_dim = dim
        self.policy_net = nn.Sequential(*pi_layers)

        # 2. 上帝视角 Critic
        vf_layers = []
        last_dim = critic_in_dim
        for dim in net_arch['vf']:
            vf_layers.append(nn.Linear(last_dim, dim))
            vf_layers.append(nn.ELU())
            last_dim = dim
        self.value_net = nn.Sequential(*vf_layers)

    def forward(self, features: torch.Tensor):
        actor_features = features[:, :self.actor_in_dim]
        return self.policy_net(actor_features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor):
        return self.policy_net(features[:, :self.actor_in_dim])

    def forward_critic(self, features: torch.Tensor):
        return self.value_net(features)

class AsymmetricPolicy(ActorCriticPolicy):
    def __init__(self, *args, actor_obs_dim=123, **kwargs):
        self.actor_obs_dim = actor_obs_dim
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricMlpExtractor(
            actor_in_dim=self.actor_obs_dim,
            critic_in_dim=self.features_dim,
            net_arch=self.net_arch
        )

# ===================================================================
# 1.5 自定义 SB3 向量化包装器
# ===================================================================
class CustomSb3VecEnvWrapper(VecEnv):
    """桥接原生 Gym 环境与 SB3 张量流的适配器"""
    def __init__(self, env):
        self.env = env
        super().__init__(env.num_envs, env.observation_space, env.action_space)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = torch.tensor(actions, dtype=torch.float32, device=self.env.device)

    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.env.step(self._actions)
        dones = terminated | truncated

        obs_np = obs.cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        dones_np = dones.cpu().numpy()

        infos = []
        reset_idx = 0 # 用于跟踪稀疏张量 info["last_observation"] 的索引
        
        for i in range(self.num_envs):
            env_info = {}
            # 将全局平均遥测数据挂载到第一个环境，避免重复记录
            if i == 0 and "telemetry" in info:
                env_info["telemetry"] = info["telemetry"]
            
            # SB3 核心机制：精准映射终端观测值 (处理稀疏张量)
            if dones_np[i] and "last_observation" in info:
                env_info["terminal_observation"] = info["last_observation"][reset_idx].cpu().numpy()
                reset_idx += 1
                
            infos.append(env_info)

        return obs_np, rewards_np, dones_np, infos

    def reset(self):
        obs, _ = self.env.reset()
        return obs.cpu().numpy()

    def close(self): pass
    def get_attr(self, attr_name, indices=None): return [getattr(self.env, attr_name)] * self.num_envs
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): pass
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs

# ===================================================================
# 2. 自定义日志回调 (轻量化解析版)
# ===================================================================
class InfoLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.dist_buf = []
        self.x_pos_buf = []
        self.success_buf = []
        self.terrain_buf = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        # ✅ 直接从 info[0] 读取平均遥测数据，极大地提升效率
        if len(infos) > 0 and "telemetry" in infos[0]:
            tel = infos[0]["telemetry"]
            self.dist_buf.append(tel["dist"])
            self.x_pos_buf.append(tel["x_pos"])
            self.success_buf.append(tel["success"])
            self.terrain_buf.append(tel["terrain_progress"])
        return True

    def _on_rollout_end(self) -> None:
        mean_dist = np.mean(self.dist_buf) if self.dist_buf else 0
        mean_x = np.mean(self.x_pos_buf) if self.x_pos_buf else 0
        mean_succ = np.mean(self.success_buf) if self.success_buf else 0
        mean_terr = np.mean(self.terrain_buf) if self.terrain_buf else 0

        self.logger.record("custom/dist_to_goal", mean_dist)
        self.logger.record("custom/x_position", mean_x)
        self.logger.record("custom/success_rate", mean_succ)
        self.logger.record("custom/terrain_progress", mean_terr)
        
        print("\n" + "="*50)
        print(f"🚀 PPO Rollout Update | Step: {self.num_timesteps}")
        print("-" * 50)
        print(f"  🏁 泊车成功率   : {mean_succ*100:.2f}%")
        print(f"  🌟 地形里程碑进度: {mean_terr:.2f} / 2.0 (地毯+泊车区)")
        print(f"  📍 平均前沿位置 : X = {mean_x:.2f} m")
        print(f"  🎯 距终点距离   : {mean_dist:.2f} m")
        print("="*50)

        self.dist_buf, self.x_pos_buf, self.success_buf, self.terrain_buf = [], [], [], []

# ===================================================================
# 3. 学习率调度器
# ===================================================================
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# ===================================================================
# 4. 主训练流程
# ===================================================================
def main():
    print("🌍 初始化 Task 3 并行环境 (Num: 64)...")
    env_cfg = Task3Config()
    env_cfg.num_envs = 64
    
    raw_env = JetbotParkingEnv(env_cfg)
    env = CustomSb3VecEnvWrapper(raw_env)

    run_dir = "./runs/task3_ppo"
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(save_freq=2048, save_path=run_dir, name_prefix="jetbot_sim2real")
    logger_callback = InfoLoggerCallback()

    print("🧠 构建 Asymmetric PPO 网络架构...")
    model = PPO(
        policy=AsymmetricPolicy,
        env=env,
        policy_kwargs=dict(
            actor_obs_dim=123, 
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
            activation_fn=nn.ELU, 
            ortho_init=True       
        ),
        learning_rate=linear_schedule(3e-4),
        n_steps=1024,             
        batch_size=4096,
        n_epochs=5,               
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,           
        ent_coef=0.01,            # 适度的熵系数，配合重时间惩罚，能有效逼迫探索
        max_grad_norm=0.5,        
        tensorboard_log=run_dir,
        device=env_cfg.device,
        verbose=1                 
    )

    print("\n🚀 开始强化学习训练...")
    # 可以通过另起一个终端运行 `tensorboard --logdir ./runs/task3_ppo` 来监控曲线
    total_timesteps = 50_000_000 
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, logger_callback],
        tb_log_name="PPO_Asymmetric"
    )
    
    print("\n🎉 训练完成！正在保存最终模型...")
    model.save(f"{run_dir}/jetbot_sim2real_final")

if __name__ == "__main__":
    main()
    simulation_app.close()
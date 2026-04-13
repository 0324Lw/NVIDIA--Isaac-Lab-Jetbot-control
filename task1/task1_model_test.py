import argparse
import os
import glob
import time
from isaaclab.app import AppLauncher

# ===================================================================
# 0. 启动引擎 (开启图形界面 UI)
# ===================================================================
parser = argparse.ArgumentParser(description="Evaluate Trained Jetbot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心库导入
# ===================================================================
import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Box

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from task1_env import JetbotNavigationEnv, Task1Config

# ===================================================================
# 1. 网络架构 (为了让 SKRL 能把权重填进去)
# ===================================================================
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
    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# ===================================================================
# 2. 评估环境专属 Config
# ===================================================================
class Task1EvalConfig(Task1Config):
    # 减少数量，方便观赏。4 台刚好能清晰看到每台的独立轨迹
    num_envs = 4 

# ===================================================================
# 3. 自动寻找最新模型权重的函数
# ===================================================================
def get_latest_checkpoint(base_dir="runs/task1_ugv"):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"找不到训练记录目录: {base_dir}")
    
    # 使用 glob 进行递归搜索 (** 代表所有子目录)
    all_models = glob.glob(os.path.join(base_dir, "**", "*.pt"), recursive=True)
    
    if not all_models:
        raise FileNotFoundError(f"在 {base_dir} 及其所有子目录下找不到任何 .pt 模型文件！")
    
    # 优先寻找名为 best_agent.pt 的最优模型
    best_models = [m for m in all_models if "best_agent.pt" in os.path.basename(m)]
    if best_models:
        # 如果有多个，按照文件修改时间取最新生成的那个
        latest_best = max(best_models, key=os.path.getmtime)
        return latest_best
        
    # 如果没有 best_agent.pt，则直接返回最新修改的任意 .pt 文件
    latest_model = max(all_models, key=os.path.getmtime)
    return latest_model

# ===================================================================
# 4. 视觉展示主程序
# ===================================================================
def main():
    print("\n" + "="*60)
    print("🎬 启动 Jetbot 导航任务视觉验收演示...")
    print("="*60)

    # 1. 实例化小规模观赏环境
    cfg = Task1EvalConfig()
    env = JetbotNavigationEnv(cfg)
    
    # 2. 挂载视觉追踪信标 (生成发光的绿球)
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Waypoint",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),   # 绿色
                    emissive_color=(0.0, 0.8, 0.0)   # 荧光发光效果
                )
            )
        }
    )
    waypoints_visualizer = VisualizationMarkers(marker_cfg)

    # 3. 建立 SKRL 推理环境
    observation_space = Box(low=-np.inf, high=np.inf, shape=(21,))
    action_space = Box(low=-1.0, high=1.0, shape=(2,))
    
    models = {
        "policy": PolicyNetwork(observation_space, action_space, env.device),
        "value": ValueNetwork(observation_space, action_space, env.device)
    }

    # 读取训练好的模型
    checkpoint_path = get_latest_checkpoint()
    print(f"[INFO] 成功定位并加载最强大脑: {checkpoint_path}")
    
    # 配置并加载 Agent (重点：开启评估模式)
    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg["state_preprocessor"] = RunningStandardScaler
    ppo_cfg["state_preprocessor_kwargs"] = {"size": observation_space.shape, "device": env.device}
    
    agent = PPO(models=models, memory=None, cfg=ppo_cfg, 
                observation_space=observation_space, action_space=action_space, device=env.device)
    
    agent.load(checkpoint_path)
    agent.set_mode("eval") # 切换为推理模式，不加噪音，不计算梯度！

    obs, _ = env.reset()
    
    print("\n🚀 开始自动驾驶演示！(按 Ctrl+C 退出)")
    
    try:
        while simulation_app.is_running():
            start_time = time.time()
            
            # --- 视觉信标同步 ---
            # 提取 4 台车当前各自要去的 2D 目标坐标
            idx = env.current_wp_idx.unsqueeze(-1).expand(-1, 2).unsqueeze(1)
            clamped_idx = torch.clamp(idx, max=cfg.num_waypoints - 1)
            current_target_2d = torch.gather(env.waypoints, 1, clamped_idx).squeeze(1)
            
            # 拼装成 3D 坐标 (Z 轴高度设为 0.1 米，让绿球半悬浮在地面)
            target_3d = torch.cat([current_target_2d, torch.ones_like(current_target_2d[:, :1]) * 0.1], dim=-1)
            
            # 通知画笔将绿球移动到目标位置
            waypoints_visualizer.visualize(translations=target_3d)

            # --- 大脑决策 ---
            # 直接调用 agent.act 获取确定性输出，不需要 wrapper
            actions, _, _ = agent.act(obs, timestep=0, timesteps=0)
            
            # --- 物理执行 ---
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # --- 现实时间同步 (强行降速到 1x 现实时间) ---
            elapsed = time.time() - start_time
            sleep_time = cfg.policy_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] 演示结束。")

if __name__ == "__main__":
    main()
    simulation_app.close()
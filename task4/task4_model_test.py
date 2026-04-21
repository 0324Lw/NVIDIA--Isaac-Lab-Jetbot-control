import argparse
import torch
import torch.nn as nn
import numpy as np

# ===================================================================
# 0. 啟動底層引擎 (GUI 模式)
# ===================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 4 MAPPO Model Inference Test")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 測試階段必須看到畫面，強制 headless 為 False
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===================================================================
# 核心庫導入
# ===================================================================
from task4_env import Task4MappoEnv, Task4Config


# 定義模型結構 (必須與 train.py 中的架構完全一致)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
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
            layer_init(nn.Linear(128, act_dim), std=0.01)
        )

    def forward(self, obs):
        return self.net(obs)


# ===================================================================
# 3. 主推理測試程序
# ===================================================================
def main():
    print("\n" + "=" * 80)
    print("🚀 啟動 Task 4 多智能體協同 (MAPPO) 模型部署測試")
    print("=" * 80)

    # 1. 環境與模型配置
    cfg = Task4Config()
    cfg.num_envs = 1  # 測試演示僅需開啟 1 個環境
    cfg.device = "cuda:0"

    # 實例化環境
    env = Task4MappoEnv(cfg)

    # 2. 載入訓練好的 Actor 權重
    model_path = "./runs/task4_custom_mappo/mappo_actor_final.pt"
    print(f"🧠 正在載入模型權重: {model_path}")

    actor = MAPPOActor(obs_dim=150, act_dim=2).to(cfg.device)

    try:
        # 載入權重字典
        actor.load_state_dict(torch.load(model_path, map_location=cfg.device))
        actor.eval()  # 切換至評估模式
        print("[✔] 模型載入成功！")
    except Exception as e:
        print(f"[✘] 模型載入失敗: {e}")
        print("💡 請檢查權重檔案路徑是否正確，或網絡層維度是否與訓練時匹配。")
        simulation_app.close()
        return

    # 3. 推理循環
    print("\n🎬 開始演示。觀察四輛無人車如何在無信控十字路口進行博弈與協作。")
    obs, info = env.reset()

    # --- 控制微調參數 ---
    # 速度縮放因子：0.8 表示輸出平移速度為原始指令的 80%，方便觀察細節
    SPEED_SCALER = 0.8

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                # 處理多智能體觀測張量
                # obs 形状: [num_envs, 4, 150] -> 展開為 [4, 150] 進行批處理推理
                flat_obs = obs.view(-1, 150)
                action_mean = actor(flat_obs)

                # 恢復為環境所需的動作形狀 [num_envs, 4, 2]
                action = action_mean.view(cfg.num_envs, 4, 2)

                # 執行動作縮放（僅縮放平移速度 v，保持轉向 w 靈敏度）
                action[:, :, 0] *= SPEED_SCALER

            # 執行環境步進
            obs, rewards, terminated, truncated, info = env.step(action)

            # 實時遙測打印
            if "telemetry" in info:
                print(f"📊 隊伍狀態 | 平均距終點: {info['telemetry']['mean_dist']:.2f}m", end='\r')

            # 終結事件處理（全隊完賽或發生碰撞）
            if (terminated | truncated).any():
                print("\n🔄 觸發重置：團隊成功完賽或發生物理碰撞。")

    except KeyboardInterrupt:
        print("\n用戶中斷測試。")


if __name__ == "__main__":
    main()
    simulation_app.close()
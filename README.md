# 🚗 基于 NVIDIA Isaac Lab 的两轮差速无人车 / Jetbot 强化学习控制项目
 
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.x-brightgreen)
![skrl](https://img.shields.io/badge/RL-skrl%20PPO-purple)
![OS](https://img.shields.io/badge/OS-Ubuntu%20%7C%20Windows-green)
 
本项目是一个基于 NVIDIA Isaac Lab 的两轮差速无人车 / Jetbot 强化学习训练项目。项目包含 4 个递进任务：多路点导航、障碍物导航、Sim2Real 泊车、四车协同编队护送。
 
这个仓库最开始是我在学习强化学习时做的个人 demo。后来我对代码进行了重新整理和工程化重构：统一了项目目录结构，统一采用 `skrl` 的 PPO 训练流程，增加了环境测试、世界模型测试、模型测试、Ubuntu / Windows 脚本、训练进度条、日志与 checkpoint 管理。希望这个项目能为同样在学习 Isaac Lab、无人车控制、移动机器人导航和强化学习的同学提供一个可参考、可复现、可继续修改的基础工程。
项目重点不是追求完美训练，而是把每个任务从环境、测试、训练到评估尽量拆清楚。代码中仍然会有可以继续改进的地方，欢迎大家根据自己的 Isaac Lab 版本、显卡配置和研究目标继续修改。
 
---
 
## 🎬 训练效果展示
 
| Scene | Preview |
|---|---|
| 多路点 / 障碍物导航 | ![Diff-Drive navigation demo](assets/gifs/diff_drive_navigation_demo.gif) |
| 泊车 / 多车编队护送 | ![Diff-Drive formation demo](assets/gifs/diff_drive_formation_demo.gif) |
 
---
 
## ✨ 项目特点
 
- 基于 NVIDIA Isaac Lab 和 Jetbot / differential-drive UGV 机器人资产。
- 包含 4 个递进任务，从基础多路点导航到障碍物导航、Sim2Real 泊车和多车协同编队护送。
- 所有任务统一使用 `skrl` PPO 训练框架。
- 每个任务提供独立的环境测试、训练脚本和模型测试脚本。
- Task2 / Task3 / Task4 将 world 逻辑与 IsaacLab 物理环境分离，方便单独测试障碍物、泊车场景、雷达、队形、窄门和课程逻辑。
- 支持 Ubuntu / Windows 本地开发、测试和训练。
- 训练采用 `tqdm` 进度条，方便查看实时进度和日志信息。
 
---
 
## 📁 项目结构
 
```text
diff_drive_ugv_isaaclab_rl/
├── configs/
│   ├── task1_multi_waypoint_navigation.yaml
│   ├── task2_obstacle_navigation.yaml
│   ├── task3_sim2real_parking.yaml
│   └── task4_multi_ugv_formation_escort.yaml
├── src/
│   └── diff_drive_rl/
│       ├── common/
│       │   ├── diff_drive_skrl_models.py
│       │   ├── diff_drive_skrl_wrappers.py
│       │   ├── info_utils.py
│       │   └── paths.py
│       └── tasks/
│           ├── task1/
│           │   ├── task1_config.py
│           │   ├── task1_scene.py
│           │   ├── task1_env.py
│           │   ├── task1_train.py
│           │   └── task1_model_test.py
│           ├── task2/
│           │   ├── task2_config.py
│           │   ├── task2_scene.py
│           │   ├── task2_world.py
│           │   ├── task2_env.py
│           │   ├── task2_train.py
│           │   └── task2_model_test.py
│           ├── task3/
│           │   ├── task3_config.py
│           │   ├── task3_scene.py
│           │   ├── task3_world.py
│           │   ├── task3_env.py
│           │   ├── task3_train.py
│           │   └── task3_model_test.py
│           └── task4/
│               ├── task4_config.py
│               ├── task4_scene.py
│               ├── task4_world.py
│               ├── task4_env.py
│               ├── task4_train.py
│               └── task4_model_test.py
├── tests/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
├── scripts/
│   ├── ubuntu/
│   └── windows/
├── logs/
├── assets/
│   ├── gifs/
│   └── usd/
├── LICENSE
└── README.md
```
 
| 目录 | 说明 |
|---|---|
| `configs/` | 每个任务的配置文件，便于统一管理任务参数。 |
| `src/diff_drive_rl/common/` | 通用网络模型、日志工具、路径工具、进度条与评估工具等。 |
| `src/diff_drive_rl/tasks/taskX/` | 每个任务的场景、世界模型、环境、训练脚本和模型测试脚本。 |
| `tests/` | 环境测试和世界模型测试脚本。 |
| `scripts/ubuntu/` | Ubuntu 下的测试、训练、评估脚本。 |
| `scripts/windows/` | Windows 下的准备检查、训练、评估脚本。 |
| `logs/` | 默认训练日志和 checkpoint 输出目录。 |
| `assets/` | README 图片、GIF、USD 占位文件和其他展示素材。 |
 
---
 
## 🛠️ 建议硬件与系统配置
 
### 最低测试配置
 
用于环境测试、world 测试、smoke training 和低并发调试：
 
- Ubuntu 22.04 / 24.04
- NVIDIA GPU，显存 16GB 以上
- Python 3.11
- PyTorch 2.x
- Isaac Sim / Isaac Lab
- `skrl`, `tensorboard`, `tqdm`
 
在 16GB 显存设备上，建议从小并发开始：
 
```bash
--num-envs 16
--num-envs 32
--num-envs 64
--num-envs 128
```
 
### 推荐训练配置
 
用于较大规模训练和长时间实验：
 
- NVIDIA RTX 3090 / 4090 或同级别 GPU
- 显存 24GB 或更高
- Windows 或 Ubuntu 均可，但需要保证 Isaac Lab 环境可正常运行
 
较大显存设备可以尝试：
 
```bash
--num-envs 512
--num-envs 1024
--num-envs 2048
```
 
具体并发数需要根据任务复杂度、显存占用和 Isaac Lab 版本调整。不要一开始直接使用最大并发，建议先运行 smoke training。
 
---
 
## 🚀 基础准备
 
### 1. 安装 Isaac Lab 环境
 
请先按照 NVIDIA Isaac Lab 官方文档安装 Isaac Sim / Isaac Lab，并确认 Isaac Lab 的 Python 环境可以正常导入：
 
```bash
python -c "import isaaclab; print('isaaclab ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
 
### 2. 克隆项目
 
```bash
git clone <your-repo-url> diff_drive_ugv_isaaclab_rl
cd diff_drive_ugv_isaaclab_rl
```
 
如果你保留旧仓库名，也可以直接进入对应目录，只需要保证项目结构与 README 中的结构一致。
 
### 3. 设置 PYTHONPATH
 
```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```
 
也可以直接使用 `scripts/ubuntu/` 下的脚本，这些脚本会自动设置项目路径。
 
### 4. 安装 Python 依赖
 
在 Isaac Lab 对应的 Python 环境中安装必要依赖：
 
```bash
pip install skrl tensorboard tqdm numpy
```
 
如果你的 Isaac Lab 安装方式已经包含部分依赖，可以按需跳过。
 
---
 
## ⚡ 快速开始
 
### 1. 环境测试
 
建议先从 Task1 开始测试，再进入后续任务。
 
```bash
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/test_task2_world.sh
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/test_task3_world.sh
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/test_task4_world.sh
bash scripts/ubuntu/test_task4_env.sh
```
 
如果显存不足，可以打开对应脚本，降低 `--num-envs`。
 
### 2. Smoke 训练
 
Smoke training 用于确认训练管线可以启动、日志可以写入、checkpoint 可以保存，不用于评估最终效果。
 
```bash
bash scripts/ubuntu/train_task1_skrl_smoke.sh
bash scripts/ubuntu/train_task2_skrl_smoke.sh
bash scripts/ubuntu/train_task3_skrl_smoke.sh
bash scripts/ubuntu/train_task4_skrl_smoke.sh
```
 
### 3. 模型测试
 
训练完成后，可以使用 eval 脚本加载 checkpoint 做推理测试。
 
```bash
bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint/diff_drive_task1_model.pt
bash scripts/ubuntu/eval_task2_skrl.sh logs/task2/<run_name>/final_checkpoint/diff_drive_task2_model.pt
bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint/diff_drive_task3_model.pt 0.30
bash scripts/ubuntu/eval_task4_skrl.sh logs/task4/<run_name>/final_checkpoint/diff_drive_task4_model.pt 0.50
```
 
---
 
## 🧩 任务设计总览
 
| Task | 目标 | 环境特点 | 训练重点 | 主要脚本 |
|---|---|---|---|---|
| Task1 | 多路点导航 | 平坦场地、随机路点、差速底盘速度控制 | 基础前进、转向、航向跟踪、路点到达 | `task1_env.py`, `task1_train.py`, `task1_model_test.py` |
| Task2 | 障碍物导航 | 解析障碍物世界、LiDAR、碰撞检测 | 避障、到达目标、保持运动连续性 | `task2_world.py`, `task2_env.py`, `task2_train.py` |
| Task3 | Sim2Real 泊车 | 泊车目标、车位几何、域随机化、低速精确控制 | 倒车/转向/入库、动作平滑、抗参数扰动 | `task3_world.py`, `task3_env.py`, `task3_train.py` |
| Task4 | 多车协同编队护送 | 4 车协同、队形槽位、障碍物、窄门、CTDE | 编队保持、团队目标导航、队内避碰 | `task4_world.py`, `task4_env.py`, `task4_train.py` |
 
---
 
## ➡️ Task 1：多路点导航
 
Task1 是最基础的差速无人车导航任务，用于训练 Jetbot 在平坦场地上根据路点指令稳定前进、转向并到达目标点。
 
### 任务目标
 
- 无人车在平坦地面上保持稳定运动。
- 根据随机生成的目标路点进行导航。
- 学习基础前进、减速、转向、航向对齐和目标到达能力。
- 为 Task2 / Task3 / Task4 提供可参考的底层移动控制 baseline。
 
### 环境设计
 
- 使用 Isaac Lab 中的 Jetbot / two-wheel differential-drive 机器人资产。
- 控制方式为低频 RL 策略输出归一化线速度和角速度指令。
- 环境将 `[v, w]` 动作转换为左右轮速度目标。
- 观测包含车体速度、角速度、目标相对位置、目标距离、航向误差、历史动作和进度信息。
- 训练代码统一采用 `skrl` PPO。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/train_task1_skrl_smoke.sh
bash scripts/ubuntu/train_task1_skrl_laptop.sh
bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint/diff_drive_task1_model.pt
```
 
### 训练时重点观察
 
- `Distance_To_Goal` 是否逐步下降
- `Progress` 是否持续为正
- `Heading_Error` 是否逐步减小
- `Success_Rate` 是否开始上升
- `Timeout_Rate` 是否过高
- PPO 的 `approx_kl`、`clip_fraction` 是否稳定
 
---
 
## ➡️ Task 2：障碍物导航
 
Task2 在 Task1 的基础上加入解析障碍物世界，用于训练差速无人车在存在障碍物的场景中导航到目标点。
 
### 任务目标
 
- 在随机障碍物场景中保持稳定运动。
- 使用 LiDAR / risk features 感知前方障碍物。
- 避免与障碍物、边界发生碰撞。
- 在避障的同时尽量向目标点前进。
- 使用课程学习逐步提升障碍物数量和导航难度。
 
### 环境设计
 
Task2 将障碍物世界和 Isaac Lab 环境拆开：
 
- `task2_world.py`：负责目标点、障碍物采样、LiDAR、碰撞检测、risk features 和课程逻辑。
- `task2_scene.py`：负责 Jetbot 和基础场景资产配置。
- `task2_env.py`：负责真实 Jetbot 物理控制、观测、奖励、终止条件和与 IsaacLab 的交互。
- `task2_world_test.py`：不启动完整训练，用于检查障碍物世界逻辑。
- `task2_env_test.py`：用于检查 IsaacLab 环境和观测、奖励、reset、动作控制等逻辑。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task2_world.sh
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/train_task2_skrl_smoke.sh
bash scripts/ubuntu/train_task2_skrl_laptop.sh
bash scripts/ubuntu/eval_task2_skrl.sh logs/task2/<run_name>/final_checkpoint/diff_drive_task2_model.pt
```
 
### 训练时重点观察
 
- `Distance_To_Goal`
- `Progress`
- `Success_Rate`
- `Collision_Rate`
- `Front_Clearance`
- `Collision_Risk`
- `Action_Smoothness`
- 不同障碍物密度下的导航稳定性
 
---
 
## ➡️ Task 3：Sim2Real 泊车
 
Task3 面向低速精确控制和 Sim2Real 场景，训练两轮差速无人车在带有目标车位几何约束的环境中完成泊车任务。
 
### 任务目标
 
- 根据泊车目标位姿完成低速精确导航。
- 同时约束位置误差、航向误差和最终停止状态。
- 在动作延迟、电机强度、轮半径比例、传感器噪声等随机化下保持鲁棒性。
- 为后续真实小车或嵌入式差速底盘部署提供训练 baseline。
 
### 环境设计
 
Task3 使用“真实 Jetbot 物理 + 解析泊车世界”的结构：
 
- `task3_world.py`：负责泊车目标、泊车框几何、障碍物、课程逻辑、LiDAR、碰撞检测和 privileged features。
- `task3_scene.py`：负责 Jetbot、场地和传感器相关 scene 配置。
- `task3_env.py`：接入 IsaacLab 的 Jetbot 物理环境，将 RL 输出转换为左右轮速度目标。
- `task3_world_test.py`：检查泊车世界、车位几何、碰撞与课程逻辑。
- `task3_env_test.py`：检查 reset、动作控制、观测切片、reward 和终止逻辑。
 
### 观测结构
 
- actor observation 使用差速底盘状态、目标相对位姿、泊车误差、LiDAR、risk features 和历史动作。
- critic observation 使用 actor observation 和 privileged world features。
- 训练脚本统一采用 `skrl` PPO，并保留进度条、checkpoint 和 TensorBoard 日志。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task3_world.sh
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/train_task3_skrl_smoke.sh
bash scripts/ubuntu/train_task3_skrl_laptop.sh
bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint/diff_drive_task3_model.pt 0.30
```
 
### 训练时重点观察
 
- `Distance_To_Goal`
- `Pose_Error`
- `Heading_Error`
- `Progress`
- `Success_Rate`
- `Collision_Rate`
- `Parking_Alignment`
- `Action_Smoothness`
 
Task3 比 Task1 / Task2 更重视低速精确控制和终态姿态，训练时不要只看距离下降，还要同时观察航向误差和泊车成功率。
 
---
 
## ➡️ Task 4：多车协同编队护送
 
Task4 面向多智能体协同控制。4 辆差速无人车需要围绕虚拟队形中心保持队形，并将团队中心护送到共享目标点，同时避开障碍物、窄门、边界和队友碰撞。
 
### 任务目标
 
- 4 辆 Jetbot 根据共享团队目标进行协同导航。
- 保持 Diamond / Wedge / Line 等队形槽位。
- 在障碍物和窄门场景中保持队形压缩与队内避碰。
- 使用 centralized critic / decentralized actor 形式，为 MAPPO / CTDE 类方法提供基础工程。
 
### 环境设计
 
Task4 采用多智能体 CTDE 训练结构：
 
```text
actor_obs = per-agent observation
critic_state = world privileged state + agent id
```
 
其中：
 
- `task4_world.py`：负责 4 车队形槽位、团队目标、障碍物、窄门、LiDAR、risk features、privileged features 和事件检测。
- `task4_scene.py`：负责 4 个 Jetbot articulation 和静态世界资产配置。
- `task4_env.py`：负责 4 车物理控制、动作展平、奖励、终止条件和与 IsaacLab 的交互。
- `task4_train.py`：将 `[num_envs, 4, obs_dim]` 展平成 `num_envs * 4` 个 skrl agent env，共享 actor，centralized critic 使用全局 state。
- `task4_model_test.py`：加载 TRUE skrl PPO checkpoint，做 deterministic policy rollout。
 
### 常用命令
 
```bash
bash scripts/ubuntu/test_task4_world.sh
bash scripts/ubuntu/test_task4_env.sh
bash scripts/ubuntu/train_task4_skrl_smoke.sh
bash scripts/ubuntu/train_task4_skrl_laptop.sh
bash scripts/ubuntu/eval_task4_skrl.sh logs/task4/<run_name>/final_checkpoint/diff_drive_task4_model.pt 0.50
```
 
### 训练时重点观察
 
- `Center_Goal_Dist`
- `Progress`
- `Mean_Slot_Error`
- `Max_Slot_Error`
- `Min_Pair_Dist`
- `Success_Rate`
- `Crash_Rate`
- `Obstacle_Collision_Rate`
- `Gate_Collision_Rate`
- `Pair_Collision_Rate`
- `Risk_Obstacle`
- `Risk_Gate`
- `Risk_Pair`
 
Task4 是当前项目中最复杂的任务，包含多车、多目标、多事件和 CTDE 训练结构。建议先用小并发 smoke training 确认训练链路，再逐步增加并发和训练步数。
 
---
 
## 📊 日志与模型保存
 
训练日志默认保存在：
 
```text
logs/task1/
logs/task2/
logs/task3/
logs/task4/
```
 
每个训练 run 通常包含：
 
```text
checkpoint_<env_steps>/
final_checkpoint/
train_metadata.pt
```
 
可以使用 TensorBoard 查看训练过程：
 
```bash
tensorboard --logdir logs
```
 
训练过程中会记录以下类型的信息：
 
- `reward_components`：各奖励项。
- `events`：成功、碰撞、越界、超时等事件。
- `telemetry`：距离、进度、航向误差、速度、课程阶段等训练指标。
- `world`：目标点、障碍物、队形、LiDAR、risk features 等世界层统计。
- `debug`：观测维度、reward 范围、异常值检查等。
- `ppo`：PPO 更新信息，例如 KL、loss、学习率等。
 
---
 
## 💻 Ubuntu / Windows 使用说明
 
### Ubuntu
 
Ubuntu 用于：
 
- 代码开发
- 环境测试
- world 测试
- smoke training
- 训练验证
 
常用脚本在：
 
```text
scripts/ubuntu/
```
 
### Windows 
 
Windows 脚本在：
 
```text
scripts/windows/
```
 
建议先运行 readiness check：
 
```powershell
.\scripts\windows\check_task1_windows_ready.ps1
.\scripts\windows\check_task2_windows_ready.ps1
.\scripts\windows\check_task3_windows_ready.ps1
.\scripts\windows\check_task4_windows_ready.ps1
```
 
Windows 训练脚本通常带有审批环境变量或手动参数，避免误启动长时间训练。例如：
 
```powershell
.\scripts\windows\train_task3_skrl_smoke_3090.ps1
.\scripts\windows\train_task4_skrl_smoke_3090.ps1
```
 
正式训练前建议先运行 smoke 版本，确认路径、IsaacLab Python、显卡和日志输出都正常。
 
---
 
## 🧭 推荐训练顺序
 
推荐顺序：
 
1. 先训练 Task1，获得基础差速底盘导航 checkpoint。
2. Task2 在 Task1 的基础上训练障碍物导航和避障能力。
3. Task3 重点训练低速精确泊车、终态姿态对齐和 Sim2Real 域随机化。
4. Task4 使用多车协同训练结构，从简单队形导航逐步过渡到障碍物、窄门和完整编队护送。
 
也可以每个任务从零开始训练，但训练时间会更长，早期调参也会更困难。
 
---
 
## 📌 当前状态与限制
 
- 本项目主要用于学习、复现实验和开源交流。
- 当前代码完成了四个任务的 IsaacLab 环境、测试、`skrl` PPO 训练和模型测试脚本。
- Task2 / Task3 / Task4 使用解析 world 层，便于单独测试目标、障碍物、LiDAR、泊车几何和编队逻辑。
- Task4 当前是多车协同控制的基础工程版本，不等同于完整工业级 MAPPO / 通信协同 / 真实多车部署方案。
- 不同 Isaac Lab / Isaac Sim 版本之间可能存在 API 差异，需要根据本地环境做少量适配。
- 训练效果会受到 GPU、并发数、随机种子、训练步数和超参数影响。
- Windows 脚本中的默认路径可能需要根据自己的机器修改。
- 本项目不是官方 Jetbot、NVIDIA 或 Isaac Lab 项目，只是个人学习和开源整理。
 
---
 
## ❓ 常见问题
 
### 1. `ModuleNotFoundError: No module named torch`
 
通常是没有进入 Isaac Lab 对应的 Python / conda 环境。请先确认：
 
```bash
which python
python -c "import torch; print(torch.__version__)"
```
 
### 2. IsaacLab / `pxr` 导入报错
 
涉及 IsaacLab、USD、`pxr` 的文件需要在 Isaac Sim / Isaac Lab 环境中运行。测试脚本中如果需要 AppLauncher，应保证先启动 AppLauncher，再导入依赖 IsaacLab 的环境文件。
 
### 3. 训练启动后显存不足怎么办?
 
先降低并发数：
 
```bash
--num-envs 16
--num-envs 32
--num-envs 64
--num-envs 128
```
 
确认能跑通后再逐步增加。
 
### 4. Smoke training 的效果不好正常吗?
 
正常。Smoke training 只用于检查训练流程是否能启动和保存模型，不代表最终策略效果。
 
### 5. Task2 / Task3 / Task4 为什么要单独做 world test?
 
这些任务包含解析目标、障碍物、车位、LiDAR、risk features、队形槽位和事件检测。很多训练问题不是 PPO 本身造成的，而是 world 采样、坐标系、碰撞检测或奖励项有问题。先跑 world test 可以减少后续训练调参的时间。
 
### 6. Windows 路径需要怎么改?
 
打开 `scripts/windows/` 下的 `.ps1` 文件，修改：
 
```powershell
$ProjectRoot
$Python
$Device
```
 
确保它们对应你本机的项目路径、IsaacLab Python 路径和显卡设备。
 
### 7. 为什么要先跑环境测试?
 
无人车训练中的很多问题不是 PPO 本身造成的，而是 reset、观测维度、坐标系、动作映射、轮速符号、奖励项或终止条件有问题。先跑测试可以减少后续训练调参的时间。
 
---
 
## 📄 License
 
This project is released under the MIT License.
 
See the `LICENSE` file for details.
 
---
 
## 🙏 Acknowledgements
 
感谢以下开源项目和工具：
 
- NVIDIA Isaac Sim / Isaac Lab
- Jetbot / differential-drive robot assets in Isaac Lab
- PyTorch
- skrl reinforcement learning library
- TensorBoard
- tqdm
- 移动机器人导航、强化学习和 Isaac Lab 开源社区
 
如果这个项目对你有帮助，欢迎参考、修改和继续完善。也欢迎指出代码或文档中的问题。
联系邮箱：2559906288@qq.com 小红书账号：574661219

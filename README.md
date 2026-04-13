# 🤖 基于 NVIDIA Isaac Lab 的 Jetbot 差速无人车 RL 导航与控制

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![OS](https://img.shields.io/badge/os-Ubuntu_24.04-green)
![Isaac](https://img.shields.io/badge/Isaac%20Lab-0.54-brightgreen)

本项目致力于使用深度强化学习（Deep Reinforcement Learning）实现双轮差速无人车（UGV）的自主导航控制，并面向真实的物理样机进行 Sim2Real（仿真到现实）部署准备。

本项目基于 NVIDIA 官方开源的下一代物理仿真平台 Isaac Lab，利用 GPU 张量并行技术实现数千个环境的同步推演。相比传统仿真，本项目直接在物理引擎的张量 API 上进行端到端控制，涵盖了差速动力学、地面摩擦力建模以及传感器延迟模拟。

---

## 🛠️ 硬件与系统要求 (Hardware & OS)

* **操作系统**：Ubuntu 24.04 LTS
* **GPU 算力**：NVIDIA RTX 5060 Laptop (8GB VRAM) 或同等支持 Ada Lovelace 架构的显卡
* **底层驱动**：CUDA 12.x，显卡驱动版本 $\ge$ 580.126
* **仿真平台**：Isaac Sim 4.x + Isaac Lab 0.54
* **环境管理**：Miniconda / Anaconda

---

## 🚀 基础准备：物理仿真环境配置

本项目依赖于 NVIDIA Isaac 系列套件，请确保已按照官方文档安装好基础环境。

### Step 1. 创建虚拟环境与安装 Isaac Lab

建议使用 Isaac Lab 自带的整合安装脚本，确保所有 C++ 扩展正确编译：

```bash
# 进入 Isaac Lab 根目录
cd IsaacLab
# 创建并安装环境
./isaaclab.sh --install
```

### Step 2. 安装强化学习扩展库
本项目采用模块化的 skrl 库作为算法后端，需手动安装：
```bash
pip install skrl pandas matplotlib
```

### Step 3. 物理引擎与底层驱动测试
在进入强化学习之前，运行底层驱动测试脚本，验证 Jetbot 的差速运动学模型及张量 API 是否正常工作：
```bash
# 运行我们将左右轮控制映射到 Tensor 的测试脚本
python test_jetbot_drive.py
```
(预期结果：弹出 3D 界面，小车自动执行前进、原地旋转、后退等动作，终端实时输出全局坐标)

## ➡️ Task 1: 序列目标点自主导航
本任务要求无人车在不依赖经典路径规划算法的情况下，仅通过感知数据实现连续目标点的寻迹。

### 1. 任务描述
* **初始状态**：无人车随机出生在网格地图原点。
* **任务目标**：按照发送顺序，依次到达 5 个随机生成的连续目标点（目标点范围 $\pm 3.0\text{m}$）。
* **判定标准**：小车质心与目标点距离 $< 0.25\text{m}$ 视为到达当前点，并自动切换至下一目标。
* **挑战**：由于差速模型存在非完整约束（Non-holonomic），模型必须学会先原地或边行进边调整航向，严禁输出高频震荡动作。

### 2. 代码拉取与运行指南

本任务的所有核心代码均位于 `task1` 文件夹下。你可以通过以下指令获取代码：

```bash
# 如果尚未克隆仓库，请先克隆整个项目
git clone [https://github.com/0324Lw/NVIDIA--Isaac-Lab-Jetbot-control.git](https://github.com/0324Lw/NVIDIA--Isaac-Lab-Jetbot-control.git)
cd NVIDIA--Isaac-Lab-Jetbot-control/task1
```
| 文件名 | 功能说明 |
|--------|----------|
| `task1_env.py` | 环境核心类。包含 InteractiveScene 场景配置、48Hz 决策频率控制及观测/奖励逻辑。 |
| `test_task1_env.py` | 极限压力测试脚本。在无头模式（Headless）下运行 10000 步，利用 Pandas 分析各奖励组件的数值分布。 |
| `task1_train.py` | 基于 skrl 的 PPO 训练程序。包含状态归一化、正交初始化及 KL 散度自适应学习率。 |
| `task1_model_test.py` | 视觉验收脚本。在 GUI 界面中实时标注绿色发光目标点，同步现实时间展示控制效果。 |

运行训练代码：
```bash
python task1_train.py
```

### 3. 强化学习建模 (RL Modeling)
针对物理部署的需求，本项目对 RL 三要素进行了针对性设计：

#### A. 状态空间设计 (Observation Space)
为了实现平移不变性与感知动态特性，采用 **3 帧堆叠 (Frame Stacking)** 方案：
- **单帧特征 (7 维)**：包含相对目标点的距离 $d$、目标点相对航向的 $\sin(\alpha)/\cos(\alpha)$、左右轮实时转速、以及上一时刻动作反馈。
- **最终输入 (21 维)**：将过去 3 帧拼接，使 MLP 网络能隐式感知加速度与传感器滞后。
- **在线归一化**：通过 RunningStandardScaler 实现输入特征的实时标准化。

#### B. 动作空间与平滑处理 (Action Space)
- **输出定义**：2 维连续值，映射至左右轮的目标角速度。
- **动作平滑滤波**：引入低通滤波机制 $Action_{real} = \tau \cdot a_t + (1-\tau) \cdot a_{t-1}$，其中 $\tau=0.2$。此设计可滤除高频抖动，防止真机部署时损坏减速齿轮箱。

#### C. 奖励函数设计 (Reward Shaping)
所有奖励组件单步截断在 $[-1, 1]$，确保梯度稳定性。
1. **步数惩罚**：-0.002，鼓励最短路径。
2. **接近奖励 (Approach Reward)**：基于势能差设计。$R = 1.0 \cdot (d_{t-1} - d_t)$，只有缩短目标距离时才给正奖。
3. **动作平滑惩罚**：惩罚动作变化的平方和，引导网络生成平滑曲线。
4. **目标达成奖励**：每到达一个点给予 $+0.5$ 分；彻底完成 5 个点给予 $+1.0$ 分大奖。
5. **目标切换修正**：在切换目标点的瞬间刷新距离记忆，消除跨目标点带来的巨大瞬时负奖励。

### 4. 算法与训练细节
#### 核心算法
- **PPO (Proximal Policy Optimization)**：利用 SKRL 实现。采用逻辑与物理分离架构，通过 64 个并行环境在 GPU 上直接收集 Transition 数据。

#### 网络架构
- Actor/Critic：独立的 MLP 结构 `[256, 128, 64]`。
- 激活函数：ELU，提供比 ReLU 更平滑的梯度流。
- 初始化：采用 Orthogonal 正交初始化，gain 设为 $\sqrt{2}$。

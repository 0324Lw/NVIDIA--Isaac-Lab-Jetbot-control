# 🚗 基于 NVIDIA Isaac Lab 的差速无人车多场景强化学习控制框架

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-supported-orange)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-supported-brightgreen)
![RL](https://img.shields.io/badge/RL-skrl%20PPO-purple)
![Platform](https://img.shields.io/badge/platform-Ubuntu%20%7C%20Windows-green)

## 项目摘要

本项目是一个面向差速无人车（Differential-Drive UGV / Jetbot 类平台）的多场景强化学习训练与验证框架。框架基于 NVIDIA Isaac Lab 构建仿真环境，使用 `skrl` PPO 训练策略，围绕地面移动机器人常见任务组织了四个递进式控制场景：多航点导航、障碍物导航、Sim2Real 泊车和多车编队护航。项目重点在于构建一套结构清晰、接口稳定、测试可复现、训练可追踪、结果可评估的移动机器人强化学习工程，而不是只提供单一任务的训练脚本。

框架从任务建模、环境交互、动作协议、观测构造、奖励分解、终止条件、训练日志、checkpoint 管理、模型评估和策略元信息记录等方面进行组织。每个任务保留独立的 `config / scene / env / train / model_test` 文件，便于单独维护任务逻辑；公共工具层则负责差速运动学、动作安全处理、数值检查、checkpoint 元信息、policy IO 文件和基础测试工具。这样的结构既能保证任务之间互不干扰，又能避免训练、评估、动作映射和序列化逻辑在多个任务中重复维护。

四个任务具有明确的递进关系。Task1 关注基础多航点导航，主要验证差速车动作接口、目标点切换、路径推进、速度控制和基础终止事件；Task2 在导航任务中加入障碍物、LiDAR 和风险特征，用于研究目标推进与避障安全之间的权衡；Task3 面向泊车任务，强调位姿误差、低速精细控制、动作延迟、动作噪声和轮速扰动等 Sim2Real 相关因素；Task4 面向多车编队护航，包含多智能体观测、共享 actor、集中 critic、队形误差、安全间距和协同任务完成等机制。

框架强调“低维移动机器人纯强化学习基线”的工程价值。差速无人车的动作空间较低，任务目标可以通过几何距离、航向误差、碰撞事件、轨迹长度、泊车位姿误差、队形偏差等指标清晰描述，因此保留纯 RL 路线具有合理性。项目通过多任务递进、自动测试、训练日志、评估指标和 policy metadata 来展示移动机器人 RL 的完整训练链路，为后续 ONNX 导出、policy output range 检查、跨仿真回放和部署前一致性验证提供基础。

本项目适合作为 Isaac Lab 移动机器人环境开发、差速车强化学习控制、导航避障任务设计、泊车控制、编队协同、多任务训练工程和策略导出验证流程的参考工程。项目不包含真实车底层驱动接口，不提供真实设备部署安全保证。若要扩展到真实平台，需要结合实际底盘、轮速控制器、通信接口、传感器标定、动力学辨识、安全限幅和紧急停止机制进行额外验证。

---

## 框架定位

本仓库在多机器人强化学习作品集中承担“地面移动机器人纯 RL 多场景工程基线”的角色，重点体现：

- 差速无人车动作协议与轮速控制接口；
- 多航点导航、避障、泊车和多车协同任务设计；
- 解析 world 与 Isaac Lab 物理环境的协同使用；
- LiDAR、风险特征、目标几何特征和队形特征的观测组织；
- reward components、events、telemetry 的日志化；
- checkpoint、normalizer 和 policy metadata 的保存；
- 纯单元测试与 IsaacLab standalone 环境测试的分层；
- 后续 ONNX 导出和 sim2sim 验证的标准接口。

---

## 任务总览

| 任务 | 名称 | 目标 | 关键机制 | 训练关注点 |
|---|---|---|---|---|
| Task1 | 多航点导航 | 差速车连续通过多个目标点 | CoreNav-v1 observation、forward-only + turn 动作映射、waypoint 事件 | 目标推进、路径效率、动作平滑、超时率 |
| Task2 | 障碍物导航 | 在障碍物场景中到达目标 | 解析 world、LiDAR、risk features、课程阶段 | 避障安全、目标距离、碰撞率、路径绕行 |
| Task3 | Sim2Real 泊车 | 在泊车区域完成目标位姿对齐 | 泊车 world、非对称 actor-critic、动作延迟/噪声、轮速扰动 | 位姿误差、低速控制、动作稳定性 |
| Task4 | 多车编队护航 | 多车协同保持队形并完成护航任务 | 多智能体观测、共享 actor、集中 critic、队形约束 | 编队误差、安全间距、协同完成率 |

---

## 项目结构

```text
diff_drive_ugv_isaaclab_rl/
├── assets/
│   ├── gifs/
│   ├── motions/
│   └── usd/
├── configs/
│   ├── local_paths.example.yaml
│   ├── task1_multi_waypoint_navigation.yaml
│   ├── task2_obstacle_navigation.yaml
│   ├── task3_sim2real_parking.yaml
│   └── task4_multi_ugv_formation_escort.yaml
├── docs/
│   ├── project_overview.md
│   ├── results_and_checkpoints.md
│   ├── task1_design.md
│   ├── task2_design.md
│   ├── task3_design.md
│   ├── task4_design.md
│   ├── troubleshooting.md
│   ├── ubuntu_training.md
│   └── windows_training.md
├── scripts/
│   ├── ubuntu/
│   │   ├── checkProjectStructure.sh
│   │   ├── testTask1Environment.sh
│   │   ├── testTask2Environment.sh
│   │   ├── testTask2World.sh
│   │   ├── testTask3Environment.sh
│   │   ├── testTask3World.sh
│   │   ├── testTask4Environment.sh
│   │   ├── testTask4World.sh
│   │   ├── trainTask1Waypoint.sh
│   │   ├── trainTask1WaypointSmoke.sh
│   │   ├── trainTask2Obstacle.sh
│   │   ├── trainTask2ObstacleSmoke.sh
│   │   ├── trainTask3Parking.sh
│   │   ├── trainTask3ParkingSmoke.sh
│   │   ├── trainTask4Formation.sh
│   │   ├── trainTask4FormationSmoke.sh
│   │   ├── evaluateTask1Waypoint.sh
│   │   ├── evaluateTask2Obstacle.sh
│   │   ├── evaluateTask3Parking.sh
│   │   ├── evaluateTask4Formation.sh
│   │   └── visual/
│   └── windows/
├── src/
│   └── diff_drive_rl/
│       ├── common/
│       ├── core/
│       │   ├── math/
│       │   └── physics/
│       ├── data/
│       ├── export/
│       ├── tasks/
│       │   ├── task1/
│       │   ├── task2/
│       │   ├── task3/
│       │   └── task4/
│       └── training/
├── tests/
│   ├── core/
│   ├── export/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## 目录说明

| 目录 | 说明 |
|---|---|
| `assets/` | 存放展示图片、GIF、USD 资源说明和占位素材。 |
| `configs/` | 存放任务配置和本地路径配置模板，任务参数与代码逻辑分离。 |
| `docs/` | 存放任务设计、训练说明、排错说明和结果说明文档。 |
| `scripts/ubuntu/` | Ubuntu 下的测试、训练、评估和可视化脚本。 |
| `scripts/windows/` | Windows 下的测试、训练、评估和可视化脚本。 |
| `src/diff_drive_rl/common/` | 通用模型、wrapper、日志辅助、路径工具和评估辅助函数。 |
| `src/diff_drive_rl/core/` | 差速车公共核心层，包括角度工具、差速运动学、动作协议等。 |
| `src/diff_drive_rl/tasks/` | 四个任务的 MDP 实现，每个任务保留独立环境和训练入口。 |
| `src/diff_drive_rl/training/` | 训练相关公共工具，包括数值安全、checkpoint 工具和元信息保存。 |
| `src/diff_drive_rl/export/` | 策略元信息工具，包括 `policy_io.json` 的生成和检查。 |
| `tests/core/` | 不依赖 Isaac Lab 的纯 Python 单元测试。 |
| `tests/export/` | policy IO 和 JSON 安全序列化相关测试。 |
| `tests/taskX/` | 依赖 IsaacLab 的任务环境测试和 world 测试。 |

---

## 框架分层

### 公共工具层

公共工具层主要负责与任务无关的基础逻辑。差速车任务中最容易重复的内容包括角度归一化、航向误差、差速轮运动学、动作清洗、轮速限幅、数值安全检查、JSON 序列化和策略元信息保存。将这些内容集中管理可以降低多个任务之间的维护成本，也能避免某个任务修复了动作或 checkpoint 问题，而其他任务仍然保留旧逻辑。

当前公共工具重点包括：

- `angle_math.py`：角度归一化、角度差、航向误差等函数；
- `diff_drive_kinematics.py`：线速度/角速度与左右轮速度之间的转换；
- `action_protocol.py`：差速动作协议和 Task1 forward-turn 映射；
- `numeric_safety.py`：Tensor 清洗、NaN/Inf 检查和安全统计；
- `checkpoint_utils.py`：JSON 安全转换和元信息写入；
- `policy_io.py`：策略输入输出协议记录。

### 任务 MDP 层

任务 MDP 层负责具体环境逻辑。每个任务独立维护：

- 任务配置；
- 场景构建；
- 环境 step；
- observation；
- reward components；
- termination / truncation；
- info 字典；
- 训练入口；
- 模型测试入口。

这种结构适合四个任务差异较大的场景。Task2、Task3 和 Task4 都具有较重的 world 逻辑，如果强行合并成一个环境，会增加 shape 错误、状态污染和调试复杂度。因此每个任务保持独立，公共层只抽取可复用工具。

### 训练与评估层

训练层围绕 `skrl` PPO 组织，包含训练配置、进度打印、checkpoint 保存、normalizer 管理和训练元信息记录。评估层主要通过各任务的 model test 与脚本入口完成，关注成功率、碰撞率、距离误差、路径长度、动作幅值和任务事件统计。

对于移动机器人任务，不能只看 reward 曲线。reward 上升不一定代表任务完成能力提升，也可能是策略学会了某个局部行为。评估阶段应同时查看：

- 是否到达目标；
- 是否碰撞；
- 是否超时；
- 是否卡住；
- 路径是否过长；
- 动作是否饱和；
- 是否出现反复震荡；
- 多车任务是否破坏队形。

---

## 动作协议

差速无人车的动作协议是本项目的关键。不同任务的动作含义并不完全相同，因此不能将所有任务强行套入同一个动作公式。

当前任务动作语义如下：

| 任务 | 动作形式 | 说明 |
|---|---|---|
| Task1 | `forward_throttle + turn` | 前进优先的导航动作，输出左右轮目标。 |
| Task2 | `forward_throttle + turn` | 与 Task1 类似，但包含课程阶段相关的前进下界。 |
| Task3 | `left_wheel + right_wheel` | 更接近底层左右轮控制，并包含延迟、噪声和扰动。 |
| Task4 | `linear_velocity + angular_velocity` | 多车场景中每辆车的线速度和角速度命令。 |

Task1 的 forward-turn 动作链路包含：

```text
raw action
→ sanitize
→ clip
→ forward curve
→ min/max forward range
→ turn scale
→ left/right wheel command
→ wheel target
```

其中，forward 通道保持前进优先，turn 通道控制左右轮差分。无转向时左右轮命令应保持一致；存在转向时，左右轮命令围绕 forward command 对称变化。该映射通过白盒测试验证，确保动作协议调整不会改变 Task1 原训练行为。

---

## 观测设计

四个任务的观测都围绕移动机器人导航控制展开，但复杂度逐步提升。

### Task1 观测

Task1 使用 CoreNav-v1 风格观测，主要包含：

- 目标相对位置；
- 目标距离；
- 当前速度；
- 航向误差；
- 前进方向对齐程度；
- 历史动作；
- waypoint 状态；
- episode 进度特征。

该任务中的观测重点是让策略理解“目标在哪里、车朝哪里、当前速度如何、应该继续前进还是转向”。

### Task2 观测

Task2 在导航观测基础上加入障碍物相关信息，例如：

- LiDAR 距离；
- 最近障碍物距离；
- 风险特征；
- 目标方向；
- 障碍物与目标方向关系；
- curriculum stage。

该任务需要策略同时考虑目标推进和障碍物安全，因此观测中既包含目标特征，也包含局部安全感知。

### Task3 观测

Task3 关注泊车控制，因此观测更强调：

- 目标泊车位姿；
- 位置误差；
- yaw 误差；
- 左右轮动作历史；
- 低速运动状态；
- 随机化相关状态；
- actor / critic 非对称信息。

泊车任务中的动作通常更精细，目标不是快速移动，而是稳定、准确地收敛到目标位姿。

### Task4 观测

Task4 是多车协同任务，观测包含：

- 自车状态；
- 目标护航对象状态；
- 队友相对位置；
- 编队误差；
- 安全间距；
- 碰撞风险；
- agent id / role 相关信息；
- 集中 critic 的全局状态。

Task4 的重点是保持每辆车的局部动作可控，同时让整体队形满足协同任务要求。

---

## 奖励与事件

奖励函数用于引导策略学习，但事件和遥测指标同样重要。本项目的任务通常会记录：

- `reward_components`：各奖励项和惩罚项；
- `events`：成功、碰撞、超时、卡住、越界、完成 waypoint 等事件；
- `telemetry`：距离、速度、动作、轮速、航向误差、队形误差等指标。

常见奖励/惩罚来源包括：

- 目标距离减少；
- 朝目标方向前进；
- 到达 waypoint；
- 泊车位姿误差减少；
- 编队误差减小；
- 避免障碍物；
- 避免碰撞；
- 避免倒退或停滞；
- 控制动作平滑；
- 限制轮速饱和。

训练分析时建议同时观察 reward、success rate、collision rate、timeout rate、path length 和 action statistics。单独依赖 reward 可能掩盖失败类型。

---

## 环境要求

项目需要在可运行 Isaac Lab 的 Python 环境中使用。建议先完成 NVIDIA Isaac Sim / Isaac Lab 官方安装，并确认以下命令正常：

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import isaaclab; print('Isaac Lab import ok')"
```

常用依赖：

```bash
pip install skrl tensorboard tqdm numpy matplotlib
```

不同 Isaac Sim、Isaac Lab、PyTorch、CUDA 和系统环境之间存在兼容关系。具体版本应以官方文档和本地运行条件为准。仓库主文档不绑定具体工作站、显卡型号或个人路径。

---

## 快速开始

### 克隆仓库

```bash
git clone https://github.com/0324Lw/NVIDIA--Isaac-Lab-Jetbot-control diff_drive_ugv_isaaclab_rl
cd diff_drive_ugv_isaaclab_rl
```

### 设置 Python 路径

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

脚本会自动识别项目根目录并设置 `PYTHONPATH`，通常可以直接通过 `scripts/ubuntu/` 或 `scripts/windows/` 运行。

### 检查项目结构

```bash
bash scripts/ubuntu/checkProjectStructure.sh
```

---

## 测试方法

### 纯单元测试

纯单元测试不需要启动 IsaacLab 仿真应用，主要检查差速运动学、动作协议和 policy IO。

```bash
cd /path/to/diff_drive_ugv_isaaclab_rl

PYTHONPATH="$PWD/src" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/core/test_diff_drive_kinematics.py \
  tests/core/test_task1_action_protocol_equivalence.py \
  tests/export/test_policy_io.py
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于避免系统中其他 pytest 插件影响当前测试环境。

### IsaacLab 环境测试

任务环境测试需要启动 IsaacLab 仿真上下文，应通过脚本运行：

```bash
bash scripts/ubuntu/testTask1Environment.sh
bash scripts/ubuntu/testTask2Environment.sh
bash scripts/ubuntu/testTask2World.sh
bash scripts/ubuntu/testTask3Environment.sh
bash scripts/ubuntu/testTask3World.sh
bash scripts/ubuntu/testTask4Environment.sh
bash scripts/ubuntu/testTask4World.sh
```

不要直接用 pytest 收集 `tests/taskX/taskX_env_test.py`。这些文件属于 standalone IsaacLab 测试程序，脚本会按正确方式启动环境。

---

## 训练方法

### Smoke 训练

Smoke 训练用于检查训练入口、环境 step、日志写入和 checkpoint 保存是否正常，不用于判断最终策略效果。

```bash
bash scripts/ubuntu/trainTask1WaypointSmoke.sh
bash scripts/ubuntu/trainTask2ObstacleSmoke.sh
bash scripts/ubuntu/trainTask3ParkingSmoke.sh
bash scripts/ubuntu/trainTask4FormationSmoke.sh
```

可以通过环境变量调整训练规模：

```bash
NUM_ENVS=512 TOTAL_ENV_STEPS=20000 bash scripts/ubuntu/trainTask1WaypointSmoke.sh
```

### 正式训练

```bash
bash scripts/ubuntu/trainTask1Waypoint.sh
bash scripts/ubuntu/trainTask2Obstacle.sh
bash scripts/ubuntu/trainTask3Parking.sh
bash scripts/ubuntu/trainTask4Formation.sh
```

推荐训练顺序：

```text
Task1 多航点导航
→ Task2 障碍物导航
→ Task3 Sim2Real 泊车
→ Task4 多车编队护航
```

Task3 和 Task4 难度更高，建议在基础任务和 world 测试通过后再进行较长训练。

---

## 评估方法

评估脚本用于加载 checkpoint 并运行策略推理：

```bash
bash scripts/ubuntu/evaluateTask1Waypoint.sh
bash scripts/ubuntu/evaluateTask2Obstacle.sh
bash scripts/ubuntu/evaluateTask3Parking.sh
bash scripts/ubuntu/evaluateTask4Formation.sh
```

可以通过 `CHECKPOINT` 指定模型目录：

```bash
CHECKPOINT=logs/task1/<run_name>/final_checkpoint bash scripts/ubuntu/evaluateTask1Waypoint.sh
```

评估时建议记录：

| 指标 | 说明 |
|---|---|
| `success_rate` | 任务完成率。 |
| `collision_rate` | 碰撞比例。 |
| `timeout_rate` | 超时比例。 |
| `stuck_rate` | 卡住或长时间无进展比例。 |
| `path_length` | 路径长度。 |
| `time_to_goal` | 到达目标所需时间。 |
| `final_goal_distance` | episode 结束时目标距离。 |
| `action_abs_mean` | 动作幅值。 |
| `action_rate_mean` | 动作变化率。 |
| `wheel_saturation_ratio` | 轮速饱和比例。 |
| `formation_error` | 多车任务队形误差。 |
| `parking_pose_error` | 泊车任务最终位姿误差。 |

---

## 可视化方法

```bash
CHECKPOINT=logs/task1/<run_name>/final_checkpoint bash scripts/ubuntu/visual/visualizeTask1Waypoint.sh
CHECKPOINT=logs/task2/<run_name>/final_checkpoint bash scripts/ubuntu/visual/visualizeTask2Obstacle.sh
CHECKPOINT=logs/task3/<run_name>/final_checkpoint bash scripts/ubuntu/visual/visualizeTask3Parking.sh
CHECKPOINT=logs/task4/<run_name>/final_checkpoint bash scripts/ubuntu/visual/visualizeTask4Formation.sh
```

可视化运行通常需要图形环境和 Isaac Sim 渲染支持。无显示环境下建议优先运行 headless 测试、训练和评估脚本。

---

## Task1：多航点导航

### 任务目标

Task1 要求差速无人车在平面环境中连续通过多个目标点。该任务用于验证基础导航能力、forward-turn 动作映射、目标切换逻辑、距离推进奖励和基础终止事件。

### 关键机制

- CoreNav-v1 observation；
- 多 waypoint 目标序列；
- forward-only + turn 动作映射；
- 目标距离归一化；
- 航向误差和目标方向对齐；
- waypoint 到达事件；
- 超时与完成事件；
- Task1 action protocol 等价测试；
- `policy_io.json` 元信息保存。

### 训练关注点

Task1 训练时建议重点观察：

- `Goal_Dist` 是否下降；
- `Progress` 是否为正；
- `Waypoint` 是否逐步触发；
- `Finish` 是否出现；
- `Back` 是否过高；
- `Slow / Stuck` 是否异常；
- 左右轮目标是否稳定；
- `policy_io.json` 是否在 final checkpoint 中生成。

### 常用命令

```bash
bash scripts/ubuntu/testTask1Environment.sh
bash scripts/ubuntu/trainTask1WaypointSmoke.sh
bash scripts/ubuntu/trainTask1Waypoint.sh
bash scripts/ubuntu/evaluateTask1Waypoint.sh
```

---

## Task2：障碍物导航

### 任务目标

Task2 要求无人车在包含障碍物的场景中到达目标点。该任务在 Task1 的导航基础上加入障碍物感知、LiDAR、风险特征和课程阶段，使策略学习如何在安全约束下推进目标。

### 关键机制

- 解析障碍物 world；
- LiDAR rays；
- risk features；
- curriculum stage；
- 障碍物碰撞检测；
- 最近障碍物距离；
- 目标推进奖励；
- 避障安全惩罚。

### 训练关注点

Task2 训练时建议同时观察目标推进和避障安全：

- `Goal_Dist`；
- `Progress`；
- `Min_Lidar`；
- `Obstacle_Collision_Rate`；
- `Near_Obstacle_Rate`；
- `Risk_Mean`；
- `Success_Rate`；
- `Timeout_Rate`。

如果目标距离下降但碰撞率上升，说明策略可能过于激进；如果碰撞率很低但目标推进很慢，说明安全惩罚可能过强或目标奖励不足。

### 常用命令

```bash
bash scripts/ubuntu/testTask2World.sh
bash scripts/ubuntu/testTask2Environment.sh
bash scripts/ubuntu/trainTask2ObstacleSmoke.sh
bash scripts/ubuntu/trainTask2Obstacle.sh
bash scripts/ubuntu/evaluateTask2Obstacle.sh
```

---

## Task3：Sim2Real 泊车

### 任务目标

Task3 面向低速泊车控制，要求无人车收敛到目标泊车位姿。该任务更关注精细动作、位姿误差、低速控制稳定性和 Sim2Real 相关扰动。

### 关键机制

- 泊车 world；
- 目标位姿误差；
- 非对称 actor-critic；
- 左右轮控制；
- action delay；
- deadband；
- motor bias；
- EMA；
- motor strength；
- wheel radius scale；
- wheel velocity target。

### 训练关注点

Task3 训练时建议重点观察：

- 位置误差；
- yaw 误差；
- 最终位姿误差；
- 泊车成功率；
- 倒车比例；
- 动作延迟后的稳定性；
- 左右轮动作是否饱和；
- critic state 维度是否正确；
- actor observation 与 critic observation 是否混用。

### 常用命令

```bash
bash scripts/ubuntu/testTask3World.sh
bash scripts/ubuntu/testTask3Environment.sh
bash scripts/ubuntu/trainTask3ParkingSmoke.sh
bash scripts/ubuntu/trainTask3Parking.sh
bash scripts/ubuntu/evaluateTask3Parking.sh
```

---

## Task4：多车编队护航

### 任务目标

Task4 面向多车协同任务。多个差速无人车需要在护航对象周围保持队形，同时避免碰撞、保持安全间距并完成协同推进。

### 关键机制

- 多智能体环境；
- shared actor；
- centralized critic；
- per-agent observation；
- formation features；
- escort target features；
- inter-vehicle distance；
- collision / risk features；
- flatten / unflatten wrapper；
- reward repeat alignment。

### 训练关注点

Task4 最容易出现 shape 错误，因此训练前应重点检查：

- env obs shape；
- wrapper obs shape；
- env action shape；
- wrapper action shape；
- critic state shape；
- reward repeat 是否与 agent 数一致；
- agent collision rate；
- formation error；
- escort progress；
- multi-agent success rate。

### 常用命令

```bash
bash scripts/ubuntu/testTask4World.sh
bash scripts/ubuntu/testTask4Environment.sh
bash scripts/ubuntu/trainTask4FormationSmoke.sh
bash scripts/ubuntu/trainTask4Formation.sh
bash scripts/ubuntu/evaluateTask4Formation.sh
```

---

## Checkpoint 与 Policy IO

训练结果默认保存在：

```text
logs/task1/
logs/task2/
logs/task3/
logs/task4/
```

典型 checkpoint 目录包含：

```text
final_checkpoint/
├── diff_drive_taskX_model.pt
├── diff_drive_taskX_skrl_model.pt
├── _observation_preprocessor.pt
├── _state_preprocessor.pt
├── _value_preprocessor.pt
├── train_metadata.pt
└── policy_io.json
```

`policy_io.json` 用于记录策略输入输出协议，例如：

```json
{
  "task_name": "task1_multi_waypoint",
  "actor_obs_dim": 42,
  "critic_obs_dim": 42,
  "action_dim": 2,
  "action_protocol": "forward_turn",
  "onnx_export_target": "actor_only",
  "normalizer_source": "actor_obs_norm"
}
```

该文件用于后续检查：

- actor observation 维度；
- critic observation 维度；
- action 维度；
- 动作协议；
- normalizer 来源；
- ONNX 导出目标；
- checkpoint 与任务是否匹配。

---

## ONNX 与 Sim2Sim 扩展方向

项目中的 policy IO 结构为后续 ONNX 和 sim2sim 提供接口基础。推荐流程如下：

1. 训练策略并保存 checkpoint；
2. 检查 `policy_io.json`；
3. 导出 actor-only ONNX；
4. 使用相同 observation sample 检查 Torch / ONNX 输出差异；
5. 在简化差速运动学模型或 MuJoCo 中进行闭环回放；
6. 对比轨迹、动作范围、轮速饱和和任务事件。

Sim2Sim 验证重点不是直接证明真实部署可行，而是检查：

- observation 是否一致；
- normalizer 是否一致；
- action scale 是否一致；
- 差速运动学是否一致；
- 控制周期是否一致；
- wheel limit 是否一致；
- 策略输出是否长期饱和；
- 轨迹是否明显发散。

---

## 资源调度建议

- 先运行结构检查和环境测试；
- 先进行 smoke 训练，再进行长时间训练；
- Task4 多车任务计算压力更高，建议先降低并发数；
- 可视化运行时不要使用过高并发；
- 训练脚本中限制常见数值库线程数，避免 CPU 线程过度占用；
- 若出现显存或内存压力，优先降低 `NUM_ENVS`、关闭可视化、缩短 smoke 步数；
- 不建议将日志、checkpoint、缓存和临时 tree 文件提交到仓库。

示例：

```bash
NUM_ENVS=256 TOTAL_ENV_STEPS=10000 bash scripts/ubuntu/trainTask4FormationSmoke.sh
```

---

## 常见问题

### `ModuleNotFoundError: No module named diff_drive_rl`

确认位于项目根目录，并设置：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

或使用 `scripts/ubuntu/` 下的脚本运行。

### pytest 报外部插件相关错误

在混合 ROS、IsaacLab 或 conda 环境中，pytest 可能自动加载外部插件。运行纯单元测试时使用：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest ...
```

### pytest 报 `fixture 'env' not found`

不要直接用 pytest 运行 `tests/taskX/taskX_env_test.py`。这些是 IsaacLab standalone 测试，应通过脚本运行：

```bash
bash scripts/ubuntu/testTask1Environment.sh
```

### Task1 报 forward command 相关断言

Task1 使用 forward curve 和 min forward action。白盒测试会检查无转向时左右轮均值是否等于 forward command，以及 forward command 是否不低于下界。若修改动作协议，应同步运行：

```bash
PYTHONPATH="$PWD/src" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/core/test_task1_action_protocol_equivalence.py
```

### Smoke 训练效果不稳定

Smoke 训练只用于检查工程链路，不用于评估最终策略质量。策略质量需要较长训练、稳定评估和多随机种子验证。

---

## 开发与维护建议

建议按照以下顺序修改项目：

1. 修改动作协议前，先运行纯单元测试；
2. 修改 Task1 映射后，运行 Task1 action equivalence test；
3. 修改 world 文件后，先运行对应 world test；
4. 修改 env 后，先运行对应 environment test；
5. 修改 checkpoint 或 policy metadata 后，检查 `policy_io.json`；
6. 修改 Task3 时，重点保护 actor / critic 维度和 action model；
7. 修改 Task4 时，重点保护 multi-agent shape、flatten/unflatten 和 critic state；
8. 修改 reward 后，同时观察 reward components、events 和 telemetry；
9. 长时间训练前先跑 smoke training；
10. 上传仓库前清理缓存、日志、checkpoint 和临时文件。

---

## 当前限制

- 项目主要用于仿真训练和工程验证，不提供真实底盘部署安全保证；
- Task2、Task3、Task4 的 world 主要为解析逻辑，和真实传感器/场景存在差异；
- Task3 的 Sim2Real 相关机制需要结合真实底盘参数进一步标定；
- Task4 多车任务对 shape、wrapper 和 critic state 要求较高，扩展时需要额外测试；
- 训练结果受随机种子、并发数、reward 权重、课程难度和训练步数影响；
- 真实部署需要额外加入通信接口、轮速控制器、安全保护、传感器标定和紧急停止机制。

---

## 许可证

This project is released under the MIT License.

See the `LICENSE` file for details.

---

## 致谢

本项目基于以下开源工具和社区生态构建：

- NVIDIA Isaac Sim / Isaac Lab
- Jetbot / Differential-Drive UGV simulation assets
- PyTorch
- skrl reinforcement learning library
- TensorBoard
- tqdm
- Open-source robotics and reinforcement learning communities
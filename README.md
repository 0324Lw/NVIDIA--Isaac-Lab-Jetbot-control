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
python tutorials/01_ugv_basics/test_jetbot_drive.py
```
(预期结果：弹出 3D 界面，小车自动执行前进、原地旋转、后退等动作，终端实时输出全局坐标)

**将非平衡态统计物理的概念（有效温度、耗散率、FDT）应用于理解和可视化深度学习训练过程**

---

# ThermoDL: Unveiling Deep Learning Dynamics with Statistical Physics

ThermoDL is not just another training platform. It offers a unique lens into the deep learning optimization process by applying principles from non-equilibrium statistical physics. Built with PyTorch and Streamlit, it empowers users to train models, visualize standard metrics, and crucially, explore the thermodynamic-like quantities that govern training dynamics.

## 🌟 Core Highlights

*   🧠 **Physics-Informed Training Analysis:** Go beyond loss and accuracy. Monitor:
    *   **Effective Temperature (T_eff):** `η * Var(g) / 2` – Quantifies the "hotness" or stochasticity of parameter updates.
    *   **Dissipation Rate:** `η * ||g||²` – Measures the rate of "energy" loss as the optimizer navigates the loss landscape.
    *   **Parameter Diffusion:** Tracks how parameters explore the space.
    *   **FDT-Related Insights:** Observe correlations inspired by the Fluctuation-Dissipation Theorem, helping to understand phenomena like sharp loss drops and noise-induced escapes from local minima.
*   📊 **Rich Visualization Suite:** Interactive charts for all standard and physics-based metrics.
*   🚀 **ResNet50 & CIFAR-10 Out-of-the-Box:** Start experimenting immediately with a well-known architecture and dataset.
*   📂 **Customizable & Extensible:** Easily upload your own datasets and adapt the framework for other models.
*   🌐 **Intuitive Web Interface:** Powered by Streamlit for a seamless user experience.
*   📤 **Export Capabilities:** Save your trained models, logs, and analysis reports.

## 🔧 Installation Guide

### Windows

1.  Clone or download this project.
2.  In the project root directory, execute the environment setup script:
    ```powershell
    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
    ```
3.  Activate the virtual environment:
    ```powershell
    . .\.venv\Scripts\Activate.ps1
    ```

### Linux/macOS

1.  Clone or download this project.
2.  In the project root directory, execute the environment setup script:
    ```bash
    bash setup_env.sh
    ```
3.  Activate the virtual environment:
    ```bash
    source .venv/bin/activate
    ```

## 🚀 How to Use

### Command-Line Training (Core Engine)

After activating the virtual environment, run:

```bash
python mit.py
```
or for Chinese comments:
```bash
python mit_cn.py
```
This will train a ResNet50 model on the CIFAR-10 dataset and display plots related to training dynamics and thermodynamics.

### Web Interface (Recommended)

After activating the virtual environment, launch the Streamlit application:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser. The interface allows you to:

1.  Configure training parameters in the sidebar.
2.  Upload custom datasets or use the default CIFAR-10.
3.  Click "Start Training" to initiate the process.
4.  Explore training progress, visualizations, and dynamic analyses across different tabs.
5.  Download the trained model and reports.

## 🔬 Focus on Training Dynamics Analysis

This project's unique strength lies in its application of statistical physics concepts to deep learning:

*   **Effective Temperature (T_eff):** Calculated as `η * Var(g) / 2` (where `η` is learning rate, `Var(g)` is gradient variance). It reflects the intensity of stochastic fluctuations in parameter updates, analogous to thermal energy in physical systems. High `T_eff` can help escape shallow local minima.
*   **Dissipation Rate:** Calculated as `η * ||g||²`. This represents the work done by the optimization algorithm or the rate at which "energy" is removed from the system as it descends the loss landscape, akin to energy dissipation in physical processes.
*   **Parameter Diffusion Coefficient:** Measures how quickly parameters spread out in the parameter space due to stochastic updates.
*   **FDT-Inspired Analysis:** While a full FDT verification is complex, the platform allows for observing relationships between fluctuation measures (like `T_eff` or parameter diffusion) and response/dissipation measures. This can provide insights into how efficiently the "noise" (from mini-batch sampling) is utilized for exploration versus how much "energy" is spent. Understanding these relationships can shed light on phenomena like sharp loss drops or the effectiveness of learning rate schedules.

These metrics provide a deeper, more mechanistic understanding of the optimization trajectory beyond simple performance curves.

## 📁 File Structure

*   `mit.py`: Core training and dynamics analysis script (English comments).
*   `mit_cn.py`: Core training and dynamics analysis script (Chinese comments).
*   `app.py`: Streamlit web application interface.
*   `setup_env.ps1`: Windows environment setup script.
*   `setup_env.sh`: Linux/macOS environment setup script.
*   `requirements.txt`: List of Python dependencies.

## 🎯 Use Cases

*   **Educational Tool:** Excellent for teaching and demonstrating the inner workings of SGD and advanced optimization concepts.
*   **Research Platform:** For researchers investigating the statistical physics of deep learning, gradient dynamics, and the role of noise in optimization.
*   **Rapid Prototyping:** Quickly set up a training and visualization pipeline for new models or datasets with advanced dynamic monitoring.
*   **Debugging & Tuning:** Gain new insights into why a model might be stuck or how hyperparameters affect the "thermodynamics" of training.

---
<br>

# ThermoDL: 用统计物理揭示深度学习动力学

ThermoDL 不仅仅是一个普通的训练平台。它通过应用非平衡态统计物理的原理，为深入理解深度学习优化过程提供了一个独特的视角。该平台基于 PyTorch 和 Streamlit 构建，用户不仅可以训练模型、可视化标准指标，更关键的是，能够探索控制训练动态的类热力学量。

## 🌟核心亮点

*   🧠 **物理启发的训练分析:** 超越损失和准确率，监测：
    *   **有效温度 (T_eff):** `η * Var(g) / 2` – 量化参数更新的“热度”或随机性。
    *   **耗散率:** `η * ||g||²` – 衡量优化器在损失格局中导航时“能量”损失的速率。
    *   **参数扩散:** 追踪参数如何在参数空间中探索。
    *   **FDT相关洞察:** 观察受涨落耗散定理启发的关联，有助于理解损失急剧下降、噪声诱导逃离局部极小值等现象。
*   📊 **丰富的可视化套件:** 为所有标准指标和基于物理的指标提供交互式图表。
*   🚀 **内置ResNet50与CIFAR-10:** 使用知名架构和数据集快速开始实验。
*   📂 **可定制与可扩展:** 轻松上传您自己的数据集，并为其他模型调整框架。
*   🌐 **直观的Web界面:** 由Streamlit驱动，提供流畅的用户体验。
*   📤 **导出功能:** 保存您训练好的模型、日志和分析报告。

## 🔧 安装指南

### Windows

1.  克隆或下载本项目。
2.  在项目根目录中，执行环境设置脚本：
    ```powershell
    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
    ```
3.  激活虚拟环境：
    ```powershell
    . .\.venv\Scripts\Activate.ps1
    ```

### Linux/macOS

1.  克隆或下载本项目。
2.  在项目根目录中，执行环境设置脚本：
    ```bash
    bash setup_env.sh
    ```
3.  激活虚拟环境：
    ```bash
    source .venv/bin/activate
    ```

## 🚀 使用方法

### 命令行训练 (核心引擎)

激活虚拟环境后，运行：
```bash
python mit_cn.py
```
或英文注释版本：
```bash
python mit.py
```
这将使用CIFAR-10数据集训练ResNet50模型，并显示与训练动力学和类热力学相关的图表。

### Web界面 (推荐)

激活虚拟环境后，启动Streamlit应用程序：
```bash
streamlit run app.py
```
在您的网络浏览器中访问 `http://localhost:8501`。该界面允许您：

1.  在侧边栏配置训练参数。
2.  上传自定义数据集或使用默认的CIFAR-10。
3.  点击“开始训练”以启动过程。
4.  在不同选项卡中探索训练进度、可视化结果和动力学分析。
5.  下载训练好的模型和报告。

## 🔬 聚焦训练动力学分析

本项目独特之处在于其将统计物理概念应用于深度学习：

*   **有效温度 (T_eff):** 计算公式为 `η * Var(g) / 2` (其中 `η` 是学习率, `Var(g)` 是梯度方差)。它反映了参数更新中随机波动的强度，类似于物理系统中的热能。高 `T_eff` 有助于逃离浅的局部极小值。
*   **耗散率:** 计算公式为 `η * ||g||²`。这代表了优化算法所做的功，或者当系统在损失格局中下降时从系统中移除“能量”的速率，类似于物理过程中的能量耗散。
*   **参数扩散系数:** 衡量由于随机更新，参数在参数空间中扩散的速度。
*   **FDT启发式分析:** 虽然完全的FDT验证很复杂，但该平台允许观察涨落度量（如`T_eff`或参数扩散）与响应/耗散度量之间的关系。这可以为了解“噪声”（来自小批量采样）如何被有效用于探索，以及多少“能量”被消耗提供洞察。理解这些关系可以揭示诸如损失急剧下降或学习率调度有效性等现象。

这些指标不仅仅是简单的性能曲线，它们为优化轨迹提供了更深层次、更具机理性的理解。

## 📁 文件结构

*   `mit.py`: 核心训练和动力学分析脚本 (英文注释)。
*   `mit_cn.py`: 核心训练和动力学分析脚本 (中文注释)。
*   `app.py`: Streamlit Web应用程序界面。
*   `setup_env.ps1`: Windows环境安装脚本。
*   `setup_env.sh`: Linux/macOS环境安装脚本。
*   `requirements.txt`: Python依赖包列表。

## 🎯 适用场景

*   **教学工具:** 非常适合用于教学和演示SGD的内部工作原理及高级优化概念。
*   **研究平台:** 供研究深度学习统计物理、梯度动力学以及噪声在优化中作用的研究人员使用。
*   **快速原型开发:** 通过高级动态监控，为新模型或数据集快速搭建训练和可视化流程。
*   **调试与调优:** 为理解模型为何卡住或超参数如何影响训练的“热力学”特性提供新见解。

---

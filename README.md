# 深度学习训练可视化平台

这个项目提供了一个基于深度学习的训练和可视化平台，使用PyTorch和Streamlit构建，支持模型训练、数据上传/下载以及结果可视化。

## 特性

- 🧠 基于ResNet50的深度学习模型训练
- 📊 丰富的训练过程可视化工具
- 🌡️ 训练动力学分析（有效温度、耗散率等）
- 📤 模型和训练日志导出功能
- 📂 支持自定义数据集上传和处理
- 🌐 直观的Web用户界面

## 安装指南

### Windows

1. 克隆或下载本项目
2. 在项目根目录执行环境安装脚本：
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
   ```
3. 激活虚拟环境：
   ```powershell
   . .\.venv\Scripts\Activate.ps1
   ```

### Linux/macOS

1. 克隆或下载本项目
2. 在项目根目录执行环境安装脚本：
   ```bash
   bash setup_env.sh
   ```
3. 激活虚拟环境：
   ```bash
   source .venv/bin/activate
   ```

## 使用方法

### 命令行训练（原始模式）

激活虚拟环境后，直接运行：

```
python mit_cn.py
```

这将使用CIFAR-10数据集训练ResNet50模型，并显示热力学相关的可视化结果。

### Web界面（推荐）

激活虚拟环境后，启动Streamlit应用：

```
streamlit run app.py
```

这将启动Web界面，您可以在浏览器中访问`http://localhost:8501`使用图形界面：

1. 在左侧边栏设置训练参数
2. 上传自定义数据集或使用默认CIFAR-10
3. 点击"开始训练"按钮启动训练
4. 在不同选项卡查看训练进度、可视化结果和数据分析
5. 下载训练好的模型和报告

## 文件说明

- `mit_cn.py`: 核心训练代码（中文注释版本）
- `mit.py`: 核心训练代码（英文注释版本）
- `app.py`: Streamlit Web应用程序
- `setup_env.ps1`: Windows环境安装脚本
- `setup_env.sh`: Linux/macOS环境安装脚本
- `requirements.txt`: 依赖包列表

## 训练动力学分析

本项目不仅提供常规的损失和准确率监控，还基于统计物理学原理，计算和可视化：

- 有效温度 (T_eff)：η * Var(g)/2
- 耗散率：η * ||g||²
- 参数更新的扩散系数
- FDT（涨落耗散定理）相关性分析

这些指标有助于理解深度学习优化过程的动力学特性。

## 适用场景

- 深度学习教学和演示
- 研究梯度下降的物理性质
- 快速部署模型训练和可视化平台
- 自定义数据集的模型训练和评估 
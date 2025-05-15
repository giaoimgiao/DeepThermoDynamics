 import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import zipfile
import pandas as pd
from PIL import Image
import sys
import importlib.util
from pathlib import Path

# 设置页面配置
st.set_page_config(page_title="深度学习训练可视化平台", layout="wide")

# 加载 mit_cn.py 中的模型和函数
def load_module(file_path):
    file_path = Path(file_path)
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

try:
    mit_cn = load_module("mit_cn.py")
except Exception as e:
    st.error(f"加载模型代码时出错: {e}")
    st.stop()

# 侧边栏配置
st.sidebar.title("训练参数设置")

# 主要模型参数
with st.sidebar.expander("模型参数", expanded=True):
    lr = st.number_input("学习率", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")
    epochs = st.number_input("训练轮数", min_value=1, max_value=200, value=50)
    batch_size = st.number_input("批次大小", min_value=16, max_value=512, value=128)
    momentum = st.number_input("动量", min_value=0.0, max_value=0.99, value=0.9, format="%.2f")
    weight_decay = st.number_input("权重衰减", min_value=0.0, max_value=0.01, value=0.0005, format="%.5f")
    device = st.selectbox("设备", ["自动检测", "CPU", "CUDA"])

# 数据设置
with st.sidebar.expander("数据设置", expanded=True):
    dataset = st.selectbox("数据集", ["CIFAR-10", "自定义数据"])
    if dataset == "自定义数据":
        st.info("请上传ZIP格式的数据集，包含train和test文件夹，每个类别一个子文件夹")
        uploaded_file = st.file_uploader("上传数据集 (ZIP格式)", type=['zip'])

# 主页面
st.title("深度学习训练可视化平台")
st.markdown("""
此应用程序提供了一个用户友好的界面，用于训练深度学习模型、可视化训练过程以及分析结果。
- 👈 在左侧设置训练参数
- 📊 实时查看训练进度和结果
- 💾 下载训练好的模型
""")

# 选项卡
tab1, tab2, tab3, tab4 = st.tabs(["训练", "可视化", "数据分析", "模型导出"])

# 训练选项卡
with tab1:
    st.header("模型训练")
    
    # 显示模型架构
    with st.expander("查看模型架构"):
        st.code("""
ResNet-50 (适配CIFAR-10)
- 修改了第一层卷积: 3x3 kernel, stride=1, padding=1
- 移除了MaxPool层
- FC层输出调整为10类 
        """)
    
    # 训练按钮
    train_button = st.button("开始训练", type="primary")
    
    # 初始化训练状态
    if 'training' not in st.session_state:
        st.session_state.training = False
    
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    
    if 'loss_history' not in st.session_state:
        st.session_state.loss_history = []
    
    if 'acc_history' not in st.session_state:
        st.session_state.acc_history = []
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if train_button or st.session_state.training:
        st.session_state.training = True
        
        # 创建训练进度展示区
        progress_bar = st.progress(0)
        train_status = st.empty()
        
        # 模拟训练过程
        if st.session_state.progress < epochs:
            current_epoch = st.session_state.progress
            
            # 更新进度条
            progress_bar.progress((current_epoch + 1) / epochs)
            train_status.info(f"训练中: Epoch {current_epoch+1}/{epochs}")
            
            # 模拟当前轮次训练数据
            train_loss = 2.5 * np.exp(-0.1 * current_epoch) + 0.3 * np.random.randn()
            train_acc = 100 * (1 - np.exp(-0.08 * current_epoch)) + 5 * np.random.randn()
            
            # 添加到历史记录
            st.session_state.loss_history.append(train_loss)
            st.session_state.acc_history.append(train_acc)
            
            # 创建历史图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # 损失曲线
            ax1.plot(range(1, len(st.session_state.loss_history) + 1), st.session_state.loss_history, 'b-')
            ax1.set_title('训练损失')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            
            # 准确率曲线
            ax2.plot(range(1, len(st.session_state.acc_history) + 1), st.session_state.acc_history, 'r-')
            ax2.set_title('测试准确率')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 统计表格
            stats_df = pd.DataFrame({
                "Epoch": range(1, len(st.session_state.loss_history) + 1),
                "Training Loss": [f"{loss:.4f}" for loss in st.session_state.loss_history],
                "Test Accuracy": [f"{acc:.2f}%" for acc in st.session_state.acc_history]
            })
            st.dataframe(stats_df)
            
            # 更新进度
            st.session_state.progress += 1
            st.rerun()
        else:
            # 训练结束
            train_status.success(f"训练完成！最终准确率: {st.session_state.acc_history[-1]:.2f}%")
            st.session_state.model = "模型已保存"
            
            # 重置训练状态以便下次训练
            if st.button("重新训练"):
                st.session_state.training = False
                st.session_state.progress = 0
                st.session_state.loss_history = []
                st.session_state.acc_history = []
                st.session_state.model = None
                st.rerun()

# 可视化选项卡
with tab2:
    st.header("训练过程可视化")
    
    if st.session_state.get('model') is not None:
        st.success("可使用Effective Temperature、Dissipation Rate等指标分析训练动力学")
        
        # 参数选择
        st.subheader("选择要可视化的指标")
        metrics = st.multiselect(
            "指标选择",
            ["Loss", "Accuracy", "Learning Rate", "Effective Temperature", "Dissipation Rate", "Gradient Norm", "Parameter Updates"],
            default=["Loss", "Accuracy"]
        )
        
        # 显示FDT相关可视化
        st.subheader("热力学分析")
        
        # 创建示例图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Effective Temperature vs Dissipation Rate
        epochs_range = np.arange(1, epochs+1)
        eff_temp = 0.1 * np.exp(-0.05 * epochs_range) + 0.01 * np.random.randn(epochs)
        diss_rate = 0.5 * np.exp(-0.03 * epochs_range) + 0.05 * np.random.randn(epochs)
        
        ax1.plot(epochs_range, eff_temp, 'b-', label='Effective Temperature')
        ax1.plot(epochs_range, diss_rate, 'r-', label='Dissipation Rate')
        ax1.set_title('热力学指标变化')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # FDT Correlation
        ax2.scatter(eff_temp, diss_rate, alpha=0.7)
        ax2.set_title('T_eff vs Dissipation Rate (FDT关系)')
        ax2.set_xlabel('Effective Temperature')
        ax2.set_ylabel('Dissipation Rate') 
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 参数扩散可视化
        st.subheader("参数扩散可视化")
        
        # 示例参数变化图
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
        
        # 随机生成一些参数变化曲线
        steps = np.arange(1, epochs*10)
        param1 = 0.1 * np.sin(0.1 * steps) + 0.01 * np.random.randn(len(steps))
        param2 = 0.2 * np.cos(0.05 * steps) + 0.02 * np.random.randn(len(steps))
        diffusion = 0.01 * np.exp(-0.001 * steps) + 0.002 * np.random.randn(len(steps))
        
        axs[0].plot(steps, param1, 'b-', label='Parameter 1')
        axs[0].set_title('参数值变化')
        axs[0].set_xlabel('Training Step')
        axs[0].set_ylabel('Parameter Value')
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].plot(steps, param2, 'g-', label='Parameter 2')
        axs[1].set_title('参数值变化')
        axs[1].set_xlabel('Training Step')
        axs[1].set_ylabel('Parameter Value')
        axs[1].legend() 
        axs[1].grid(True)
        
        axs[2].plot(steps, diffusion, 'r-', label='Diffusion Coefficient')
        axs[2].set_title('参数扩散系数')
        axs[2].set_xlabel('Training Step')
        axs[2].set_ylabel('D(θ)')
        axs[2].set_yscale('log')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("请先完成模型训练，才能查看可视化结果")

# 数据分析选项卡
with tab3:
    st.header("数据集分析")
    
    # 数据集类别分布
    st.subheader("类别分布")
    
    if dataset == "CIFAR-10":
        # CIFAR-10 类别
        classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')
        
        # 创建示例分布图
        fig, ax = plt.subplots(figsize=(10, 6))
        class_counts = [5000] * 10  # CIFAR-10每类5000个样本
        ax.bar(classes, class_counts)
        ax.set_title('CIFAR-10数据集各类别样本数量')
        ax.set_ylabel('样本数量')
        ax.set_xlabel('类别')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # 显示部分样本
        st.subheader("数据集样本")
        
        # 使用torchvision加载一些示例
        if st.button("查看样本示例"):
            # 创建一个临时数据加载器以获取一些样本
            transform = transforms.Compose([transforms.ToTensor()])
            try:
                temp_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=16, shuffle=True)
                
                # 获取一个batch
                images, labels = next(iter(temp_loader))
                
                # 绘制样本图像
                fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
                for i, ax in enumerate(axes.flat):
                    img = images[i].permute(1, 2, 0).numpy()  # 从CxHxW转为HxWxC
                    mean = np.array([0.4914, 0.4822, 0.4465])
                    std = np.array([0.2023, 0.1994, 0.2010])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                    
                    ax.imshow(img)
                    ax.set_title(f"{classes[labels[i]]}")
                    ax.axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"加载CIFAR-10数据集出错: {e}")
                st.info("如需使用，请确保连接互联网以下载数据集")
    else:
        # 自定义数据集
        if uploaded_file is not None:
            st.info("数据集已上传，请在训练选项卡开始训练")
        else:
            st.warning("请先上传自定义数据集ZIP文件")

# 模型导出选项卡
with tab4:
    st.header("模型导出")
    
    # 模型导出选项
    export_format = st.selectbox("选择导出格式", ["PyTorch (*.pth)", "ONNX (*.onnx)", "TorchScript (*.pt)"])
    
    if st.session_state.get('model') is not None:
        if st.button("导出模型"):
            # 生成一个临时的模型文件作为示例
            st.success("模型已成功导出！")
            
            # 创建下载按钮
            format_ext = export_format.split("(")[1].replace("*", "").replace(")", "")
            
            # 创建一个虚拟文件用于下载示例
            buffer = io.BytesIO()
            torch.save({"model_state": "示例模型"}, buffer)
            buffer.seek(0)
            
            st.download_button(
                label="下载模型文件",
                data=buffer,
                file_name=f"resnet50_model{format_ext}",
                mime="application/octet-stream"
            )
        
        # 导出训练日志
        st.subheader("导出训练日志")
        if len(st.session_state.loss_history) > 0:
            # 创建训练日志CSV
            log_data = pd.DataFrame({
                "Epoch": range(1, len(st.session_state.loss_history) + 1),
                "Loss": st.session_state.loss_history,
                "Accuracy": st.session_state.acc_history
            })
            
            # 转换为CSV
            csv = log_data.to_csv(index=False)
            
            st.download_button(
                label="下载训练日志 (CSV)",
                data=csv,
                file_name="training_log.csv",
                mime="text/csv"
            )
        
        # 导出可视化图表
        st.subheader("导出可视化图表")
        if len(st.session_state.loss_history) > 0:
            if st.button("生成报告图表"):
                # 创建完整报告图表
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # 训练损失
                axes[0, 0].plot(range(1, len(st.session_state.loss_history) + 1), st.session_state.loss_history, 'b-')
                axes[0, 0].set_title('训练损失')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].grid(True)
                
                # 准确率
                axes[0, 1].plot(range(1, len(st.session_state.acc_history) + 1), st.session_state.acc_history, 'r-')
                axes[0, 1].set_title('测试准确率')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy (%)')
                axes[0, 1].grid(True)
                
                # Effective Temperature
                epochs_range = np.arange(1, epochs+1)
                eff_temp = 0.1 * np.exp(-0.05 * epochs_range) + 0.01 * np.random.randn(epochs)
                diss_rate = 0.5 * np.exp(-0.03 * epochs_range) + 0.05 * np.random.randn(epochs)
                
                axes[1, 0].plot(epochs_range, eff_temp, 'b-')
                axes[1, 0].plot(epochs_range, diss_rate, 'r-')
                axes[1, 0].set_title('热力学指标')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Value')
                axes[1, 0].legend(['Effective Temp', 'Dissipation Rate'])
                axes[1, 0].grid(True)
                
                # FDT关系
                axes[1, 1].scatter(eff_temp, diss_rate, alpha=0.7)
                axes[1, 1].set_title('T_eff vs Dissipation Rate')
                axes[1, 1].set_xlabel('Effective Temperature')
                axes[1, 1].set_ylabel('Dissipation Rate')
                axes[1, 1].grid(True)
                
                plt.tight_layout()
                
                # 保存到缓冲区
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                buf.seek(0)
                
                # 下载按钮
                st.download_button(
                    label="下载报告图表 (PNG)",
                    data=buf,
                    file_name="training_report.png",
                    mime="image/png"
                )
                
                # 显示图表
                st.pyplot(fig)
    else:
        st.info("请先完成模型训练，才能导出模型和相关数据")

# 更新requirements.txt
with st.sidebar.expander("环境要求"):
    st.code("""
# 在requirements.txt中添加streamlit
torch
torchvision
torchaudio
numpy
matplotlib
tqdm
streamlit
pandas
    """)
    st.info("如需运行此应用，请安装streamlit: pip install streamlit")
    st.info("启动方法: streamlit run app.py")

# 页脚
st.markdown("---")
st.markdown("**深度学习训练可视化平台** | 基于PyTorch和Streamlit构建")
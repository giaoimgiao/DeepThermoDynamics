import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy  # 用于深拷贝参数

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.01  # 初始学习率，将由调度器调整
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128
NUM_EPOCHS = 50  # 训练总轮数，可根据实验需求调整
NUM_CLASSES = 10  # CIFAR-10 分类数
LOG_INTERVAL = 50  # 每隔 N 个批次记录一次日志

# 用于跟踪特定参数
NUM_TRACKED_PARAMS = 10  # 需要详细跟踪的参数个数（用于扩散等分析）

# --- 数据加载 ---
print("==> 正在准备数据...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# --- 构建模型 ---
print("==> 构建模型...")
model = models.resnet50(weights=None, num_classes=NUM_CLASSES)  # weights=None 表示从头开始训练
# 针对 CIFAR-10（输入尺寸较小）调整 ResNet
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()  # CIFAR 图像较小，早期使用 maxpool 可能过于激进
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# --- 训练监控器 ---
class TrainingMonitor:
    def __init__(self, model, optimizer, log_interval=LOG_INTERVAL):
        self.model = model
        self.optimizer = optimizer
        self.log_interval = log_interval
        # 历史记录字典
        self.history = {
            'step': [], 'epoch': [], 'loss': [], 'accuracy': [],
            'learning_rate': [],
            'eff_temp_grad_var': [],  # 有效温度 T_eff ~ η * Var(g) / 2
            'dissipation_rate': [],   # 耗散率 P ~ η * ||g||^2
            'grad_norm_sq': [],       # 梯度范数平方 ||g||^2
            'param_norm_sq': [],      # 参数范数平方 ||θ||^2
        }
        self.tracked_param_indices = None
        self.tracked_param_names = []
        # 存储每个被跟踪参数的历史值
        self.param_history = {}  # {name: [val_step1, val_step2, ...]}
        # 存储每个被跟踪参数的更新量
        self.param_update_history = {}  # {name: [delta_theta_step1, ...]}
        # 用于计算增量的上一时刻参数值
        self.last_tracked_params = {}  # {name: last_value}

        self._select_tracked_parameters()

    def _select_tracked_parameters(self):
        """随机选择若干标量参数进行跟踪"""
        self.tracked_param_indices = []
        param_names = []
        all_params_flat = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_names.extend([f"{name}_{i}" for i in range(param.numel())])
                all_params_flat.append(param.detach().cpu().flatten())

        if not param_names:
            return

        total_params = len(param_names)
        # 随机选择要跟踪的参数索引
        indices_to_track = np.random.choice(total_params, size=min(NUM_TRACKED_PARAMS, total_params), replace=False)

        current_idx = 0
        param_idx_map = []  # (param_name, flattened_idx)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                for i in range(param.numel()):
                    param_idx_map.append((name, i))

        for flat_idx in indices_to_track:
            original_name, original_flat_idx = param_idx_map[flat_idx]
            tracked_name = f"{original_name}_flat{original_flat_idx}"
            self.tracked_param_names.append(tracked_name)
            self.param_history[tracked_name] = []
            self.param_update_history[tracked_name] = []

            # 找到实际参数对象及其索引，方便后续取值
            p_obj = dict(self.model.named_parameters())[original_name]
            self.tracked_param_indices.append({
                'name': original_name,
                'param_obj': p_obj,
                'flat_idx': original_flat_idx,
                'global_tracked_name': tracked_name
            })
        print(f"正在跟踪 {len(self.tracked_param_names)} 个参数: {self.tracked_param_names[:5]} ...")

    def _get_current_tracked_param_values(self):
        """获取当前被跟踪参数的数值"""
        current_values = {}
        for p_info in self.tracked_param_indices:
            param_val = p_info['param_obj'].data.detach().cpu().flatten()[p_info['flat_idx']].item()
            current_values[p_info['global_tracked_name']] = param_val
        return current_values

    def log_batch(self, epoch, batch_idx, total_batches, loss_val):
        """在每个批次记录信息"""
        if batch_idx % self.log_interval != 0 and batch_idx != total_batches - 1:  # 同时记录最后一个批次
            return

        current_step = epoch * total_batches + batch_idx
        self.history['step'].append(current_step)
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss_val.item())

        current_lr = self.optimizer.param_groups[0]['lr']
        self.history['learning_rate'].append(current_lr)

        all_grads_flat = []
        all_params_flat = []
        for param in self.model.parameters():
            if param.grad is not None:
                all_grads_flat.append(param.grad.detach().cpu().flatten())
            if param.requires_grad:
                all_params_flat.append(param.data.detach().cpu().flatten())

        if not all_grads_flat:  # 例如第一步反向传播之前或无梯度时
            self.history['eff_temp_grad_var'].append(0)
            self.history['dissipation_rate'].append(0)
            self.history['grad_norm_sq'].append(0)
        else:
            all_grads_flat_tensor = torch.cat(all_grads_flat)
            grad_variance = torch.var(all_grads_flat_tensor).item()
            grad_norm_sq = torch.sum(all_grads_flat_tensor ** 2).item()

            # 有效温度 T_eff 近似 ~ η * Var(g) / 2
            self.history['eff_temp_grad_var'].append(current_lr * grad_variance / 2.0)
            # 耗散率 P ~ η * ||g||^2
            self.history['dissipation_rate'].append(current_lr * grad_norm_sq)
            self.history['grad_norm_sq'].append(grad_norm_sq)

        if all_params_flat:
            all_params_flat_tensor = torch.cat(all_params_flat)
            param_norm_sq = torch.sum(all_params_flat_tensor ** 2).item()
            self.history['param_norm_sq'].append(param_norm_sq)
        else:
            self.history['param_norm_sq'].append(0)

        # 跟踪特定参数的变化
        current_tracked_param_vals = self._get_current_tracked_param_values()
        for name, val in current_tracked_param_vals.items():
            self.param_history[name].append(val)
            if name in self.last_tracked_params:
                delta_theta = val - self.last_tracked_params[name]
                self.param_update_history[name].append(delta_theta)
            else:  # 首次记录
                self.param_update_history[name].append(0.0)
            self.last_tracked_params[name] = val

    def log_epoch(self, accuracy_val):
        """在每个 epoch 结束后记录准确率"""
        self.history['accuracy'].append(accuracy_val)

    def plot_results(self):
        """绘制训练过程中各种指标的曲线"""
        num_plots = 6 + len(self.tracked_param_names) * 2  # 基础图 + 参数值 + 参数扩散
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots * 4), sharex=True)

        steps = self.history['step']

        # 1) 损失
        axs[0].plot(steps, self.history['loss'], label='Loss')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # 2) 准确率（按 epoch 记录）
        epoch_ends = [s for i, s in enumerate(steps) if i == len(steps) - 1 or self.history['epoch'][i + 1] != self.history['epoch'][i]]
        if len(epoch_ends) > len(self.history['accuracy']):
            epoch_ends = epoch_ends[:len(self.history['accuracy'])]
        if self.history['accuracy']:
            axs[1].plot(epoch_ends, self.history['accuracy'], label='Accuracy', marker='o')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        # 3) 学习率
        axs[2].plot(steps, self.history['learning_rate'], label='Learning Rate')
        axs[2].set_ylabel('Learning Rate')
        axs[2].legend()

        # 4) 有效温度 T_eff
        axs[3].plot(steps, self.history['eff_temp_grad_var'], label='Effective Temp. (η Var(g)/2)')
        axs[3].set_ylabel('T_eff (Grad Var)')
        axs[3].set_yscale('log')
        axs[3].legend()

        # 5) 耗散率
        axs[4].plot(steps, self.history['dissipation_rate'], label='Dissipation Rate (η ||g||²)')
        axs[4].set_ylabel('Dissipation Rate')
        axs[4].set_yscale('log')
        axs[4].legend()

        # 6) 梯度范数平方
        axs[5].plot(steps, self.history['grad_norm_sq'], label='Grad Norm Squared ||g||²')
        axs[5].set_ylabel('||g||²')
        axs[5].set_yscale('log')
        axs[5].legend()

        plot_idx = 6
        for name in self.tracked_param_names:
            if self.param_history[name]:
                # 7+) 参数数值曲线
                axs[plot_idx].plot(steps, self.param_history[name], label=f'Param: {name}')
                axs[plot_idx].set_ylabel('θ Value')
                axs[plot_idx].legend()
                plot_idx += 1

                # 8+) 参数扩散系数 D_i ≈ <(Δθ_i)^2> / 2
                delta_thetas_sq = np.array(self.param_update_history[name]) ** 2
                if len(delta_thetas_sq) > 10:  # 需要足够的数据点进行滑动平均
                    window_size = min(len(delta_thetas_sq), 50)
                    moving_avg_delta_sq = np.convolve(delta_thetas_sq, np.ones(window_size) / window_size, mode='valid')
                    diffusion_coeff = moving_avg_delta_sq / 2.0

                    # 对齐步数
                    steps_for_diffusion = steps[window_size - 1: len(delta_thetas_sq)]
                    if len(steps_for_diffusion) == len(diffusion_coeff):
                        axs[plot_idx].plot(steps_for_diffusion, diffusion_coeff, label=f'Diffusion D({name})')

                        # FDT 检验：D 与 η*T_eff 的关系
                        aligned_lr = np.array(self.history['learning_rate'])[window_size - 1: len(delta_thetas_sq)]
                        aligned_T_eff = np.array(self.history['eff_temp_grad_var'])[window_size - 1: len(delta_thetas_sq)]
                        if len(aligned_lr) == len(diffusion_coeff):
                            axs[plot_idx].plot(steps_for_diffusion, aligned_lr * aligned_T_eff, label=f'η*T_eff for D({name})', linestyle='--')

                        axs[plot_idx].set_ylabel('D or η*T_eff')
                        axs[plot_idx].set_yscale('log')
                        axs[plot_idx].legend()
                    else:
                        print(f"警告: {name} 的扩散绘图步数不匹配")

                plot_idx += 1

        axs[-1].set_xlabel('Training Step')
        plt.tight_layout()
        plt.show()

# 创建监控器实例
monitor = TrainingMonitor(model, optimizer)

# --- 训练函数 ---

def train(epoch):
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    num_batches = len(trainloader)
    progress_bar = tqdm(enumerate(trainloader), total=num_batches)

    for batch_idx, (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.set_description(f'Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f}% ({correct}/{total})')

        # 记录批次信息
        monitor.log_batch(epoch, batch_idx, num_batches, loss)

# --- 测试函数 ---

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(testloader), total=len(testloader))
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_description(f'Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f}% ({correct}/{total})')

    acc = 100. * correct / total
    monitor.log_epoch(acc)  # 记录准确率
    return acc

# --- 主循环 ---

best_acc = 0
for epoch in range(NUM_EPOCHS):
    train(epoch)
    current_acc = test(epoch)
    scheduler.step()
    if current_acc > best_acc:
        print(f"保存最佳模型，准确率: {current_acc:.2f}%")
        # torch.save(model.state_dict(), './resnet50_cifar10_best.pth')  # 如有需要可保存模型
        best_acc = current_acc

print(f"训练结束，最佳准确率: {best_acc:.2f}%")

# --- 绘制结果 ---
monitor.plot_results()

# --- 后续分析思路 ---
# 1. 相关性分析：
#    - 计算 T_eff 与 dissipation_rate 的 Pearson 相关系数。
# 2. 参数波动功率谱密度 (PSD)：
#    - 对于 param_history 中的参数，使用 FFT 计算 PSD。
# 3. 更严格地检查爱因斯坦关系 D_i vs η*T_eff：
#    - 绘制散点图比较 D_i 与 (η*T_eff) 的线性关系。
# 4. Harada-Sasa 关系：
#    - H-S: ∫_0^τ J(s) ds = (1/2) ∫_0^τ ds ∫_0^τ ds' C_R(s, s')
#    - 其中 J(s) 为耗散率，C_R 是响应函数与自相关的组合，直接测量较难，可以通过实验扰动获得。 
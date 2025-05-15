import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy # For deep copying parameters

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.01 # Initial LR, will be scheduled
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128
NUM_EPOCHS = 50 # Adjust as needed for your experiment
NUM_CLASSES = 10 # CIFAR-10
LOG_INTERVAL = 50 # Log data every N batches

# For tracking specific parameters
NUM_TRACKED_PARAMS = 10 # Number of parameters to track in detail for diffusion etc.

# --- Data Loading ---
print("==> Preparing data...")
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

# --- Model ---
print("==> Building model...")
model = models.resnet50(weights=None, num_classes=NUM_CLASSES) # weights=None for training from scratch
# Modify ResNet for CIFAR-10 (smaller input size)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity() # CIFAR images are small, maxpool might be too aggressive early
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# --- Measurement Monitor ---
class TrainingMonitor:
    def __init__(self, model, optimizer, log_interval=LOG_INTERVAL):
        self.model = model
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.history = {
            'step': [], 'epoch': [], 'loss': [], 'accuracy': [],
            'learning_rate': [],
            'eff_temp_grad_var': [], # T_eff ~ η * Var(g)
            'dissipation_rate': [],  # P ~ η * ||g||^2
            'grad_norm_sq': [],      # ||g||^2
            'param_norm_sq': [],     # ||θ||^2
        }
        self.tracked_param_indices = None
        self.tracked_param_names = []
        self.param_history = {} # Stores history for specific parameters: {name: [val_step1, val_step2,...]}
        self.param_update_history = {} # Stores history of updates: {name: [delta_theta_step1, ...]}
        self.last_tracked_params = {} # Stores {name: last_value} for calculating delta

        self._select_tracked_parameters()

    def _select_tracked_parameters(self):
        self.tracked_param_indices = []
        param_names = []
        all_params_flat = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_names.extend([f"{name}_{i}" for i in range(param.numel())])
                all_params_flat.append(param.detach().cpu().flatten())
        
        if not param_names: return

        total_params = len(param_names)
        indices_to_track = np.random.choice(total_params, size=min(NUM_TRACKED_PARAMS, total_params), replace=False)
        
        current_idx = 0
        param_idx_map = [] # (original_param_name, original_param_flattened_idx)
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
            
            # Find the actual parameter and its index to retrieve value later
            # This is a bit indirect but ensures we track individual scalar components
            p_obj = dict(self.model.named_parameters())[original_name]
            self.tracked_param_indices.append({
                'name': original_name, # Name of the nn.Parameter object
                'param_obj': p_obj,
                'flat_idx': original_flat_idx, # Index within the flattened version of that parameter
                'global_tracked_name': tracked_name
            })
        print(f"Tracking {len(self.tracked_param_names)} parameters: {self.tracked_param_names[:5]}...")


    def _get_current_tracked_param_values(self):
        current_values = {}
        for p_info in self.tracked_param_indices:
            param_val = p_info['param_obj'].data.detach().cpu().flatten()[p_info['flat_idx']].item()
            current_values[p_info['global_tracked_name']] = param_val
        return current_values

    def log_batch(self, epoch, batch_idx, total_batches, loss_val):
        if batch_idx % self.log_interval != 0 and batch_idx != total_batches -1 : # also log last batch
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
        
        if not all_grads_flat: # E.g. first step before backward or if no params have grads
            self.history['eff_temp_grad_var'].append(0)
            self.history['dissipation_rate'].append(0)
            self.history['grad_norm_sq'].append(0)
        else:
            all_grads_flat_tensor = torch.cat(all_grads_flat)
            grad_variance = torch.var(all_grads_flat_tensor).item()
            grad_norm_sq = torch.sum(all_grads_flat_tensor**2).item()

            self.history['eff_temp_grad_var'].append(current_lr * grad_variance / 2.0) # T_eff ~ η * Var(g) / 2
            self.history['dissipation_rate'].append(current_lr * grad_norm_sq)      # P ~ η * ||g||^2
            self.history['grad_norm_sq'].append(grad_norm_sq)

        if all_params_flat:
            all_params_flat_tensor = torch.cat(all_params_flat)
            param_norm_sq = torch.sum(all_params_flat_tensor**2).item()
            self.history['param_norm_sq'].append(param_norm_sq)
        else:
            self.history['param_norm_sq'].append(0)

        # Track specific parameters
        current_tracked_param_vals = self._get_current_tracked_param_values()
        for name, val in current_tracked_param_vals.items():
            self.param_history[name].append(val)
            if name in self.last_tracked_params:
                delta_theta = val - self.last_tracked_params[name]
                self.param_update_history[name].append(delta_theta)
            else: # First time seeing this param
                self.param_update_history[name].append(0.0) # or np.nan
            self.last_tracked_params[name] = val
            
    def log_epoch(self, accuracy_val):
        self.history['accuracy'].append(accuracy_val) # Accuracy logged per epoch

    def plot_results(self):
        num_plots = 6 + len(self.tracked_param_names) * 2 # Basic + params + deltas
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots * 4), sharex=True)
        
        steps = self.history['step']

        axs[0].plot(steps, self.history['loss'], label='Loss')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # Accuracy is per epoch, need to align its x-axis or plot separately
        epoch_steps = [s for i, s in enumerate(steps) if (i == 0 or self.history['epoch'][i] != self.history['epoch'][i-1]) and len(self.history['accuracy']) > self.history['epoch'][i]]
        epoch_acc_values = [self.history['accuracy'][self.history['epoch'][i]] for i,s in enumerate(steps) if (i == 0 or self.history['epoch'][i] != self.history['epoch'][i-1]) and len(self.history['accuracy']) > self.history['epoch'][i]]
        # A bit complex due to logging accuracy at end of epoch, steps are mid-epoch. Simpler:
        epoch_ends = [s for i,s in enumerate(steps) if i == len(steps)-1 or self.history['epoch'][i+1] != self.history['epoch'][i]]
        if len(epoch_ends) > len(self.history['accuracy']): # If training stopped mid-epoch
           epoch_ends = epoch_ends[:len(self.history['accuracy'])]

        if self.history['accuracy']:
             axs[1].plot(epoch_ends, self.history['accuracy'], label='Accuracy', marker='o')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        axs[2].plot(steps, self.history['learning_rate'], label='Learning Rate')
        axs[2].set_ylabel('Learning Rate')
        axs[2].legend()

        axs[3].plot(steps, self.history['eff_temp_grad_var'], label='Effective Temp. (η Var(g)/2)')
        axs[3].set_ylabel('T_eff (Grad Var)')
        axs[3].set_yscale('log') # Often spans orders of magnitude
        axs[3].legend()

        axs[4].plot(steps, self.history['dissipation_rate'], label='Dissipation Rate (η ||g||²)')
        axs[4].set_ylabel('Dissipation Rate')
        axs[4].set_yscale('log')
        axs[4].legend()
        
        axs[5].plot(steps, self.history['grad_norm_sq'], label='Grad Norm Squared ||g||²')
        axs[5].set_ylabel('||g||²')
        axs[5].set_yscale('log')
        axs[5].legend()

        plot_idx = 6
        for name in self.tracked_param_names:
            if self.param_history[name]:
                axs[plot_idx].plot(steps, self.param_history[name], label=f'Param: {name}')
                axs[plot_idx].set_ylabel(f'θ Val')
                axs[plot_idx].legend()
                plot_idx += 1
                
                # Calculate Diffusion D_i ≈ <(Δθ_i)^2> / (2*Δt_eff), Δt_eff = 1 step for simplicity
                # We use moving average for <(Δθ_i)^2>
                delta_thetas_sq = np.array(self.param_update_history[name])**2
                if len(delta_thetas_sq) > 10: # Need some points for moving average
                    window_size = min(len(delta_thetas_sq), 50) # Moving average window
                    moving_avg_delta_sq = np.convolve(delta_thetas_sq, np.ones(window_size)/window_size, mode='valid')
                    diffusion_coeff = moving_avg_delta_sq / 2.0
                    
                    # Align steps for plotting convolved data
                    steps_for_diffusion = steps[window_size -1 : len(delta_thetas_sq)]
                    if len(steps_for_diffusion) == len(diffusion_coeff):
                        axs[plot_idx].plot(steps_for_diffusion, diffusion_coeff, label=f'Diffusion D({name})')
                        
                        # FDT-like check: D vs η * T_eff
                        # Ensure T_eff is aligned with diffusion steps
                        aligned_lr = np.array(self.history['learning_rate'])[window_size -1 : len(delta_thetas_sq)]
                        aligned_T_eff = np.array(self.history['eff_temp_grad_var'])[window_size -1 : len(delta_thetas_sq)]
                        if len(aligned_lr) == len(diffusion_coeff):
                            axs[plot_idx].plot(steps_for_diffusion, aligned_lr * aligned_T_eff, label=f'η*T_eff for D({name})', linestyle='--')
                        
                        axs[plot_idx].set_ylabel(f'D or η*T_eff')
                        axs[plot_idx].set_yscale('log')
                        axs[plot_idx].legend()
                    else:
                        print(f"Warning: Mismatch in length for diffusion plot of {name}")

                plot_idx += 1
        
        axs[-1].set_xlabel('Training Step')
        plt.tight_layout()
        plt.show()

monitor = TrainingMonitor(model, optimizer)

# --- Training Function ---
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

        progress_bar.set_description(f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')
        
        # Log batch data
        monitor.log_batch(epoch, batch_idx, num_batches, loss)

# --- Test Function ---
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
            
            progress_bar.set_description(f'Loss: {test_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')
    
    acc = 100.*correct/total
    monitor.log_epoch(acc) # Log accuracy after epoch test
    return acc

# --- Main Loop ---
best_acc = 0
for epoch in range(NUM_EPOCHS):
    train(epoch)
    current_acc = test(epoch)
    scheduler.step()
    if current_acc > best_acc:
        print(f"Saving best model with acc: {current_acc:.2f}%")
        # torch.save(model.state_dict(), './resnet50_cifar10_best.pth') # Optional: save model
        best_acc = current_acc

print(f"Finished Training. Best Accuracy: {best_acc:.2f}%")

# --- Plot Results ---
monitor.plot_results()

# --- Further Analysis Ideas (after running the above) ---
# 1. Correlation Analysis:
#    - Calculate Pearson correlation between T_eff and dissipation_rate.
#    - Correlate T_eff with periods of sharp loss decrease (escape from local minima).
# 2. Power Spectral Density (PSD) of parameter fluctuations:
#    - For tracked parameters (self.param_history[name]), calculate PSD using FFT.
#    - This relates to the 'C' part of FDT.
# 3. Check Einstein Relation D_i vs η*T_eff more rigorously:
#    - In the plot, we did a visual check. One could do a scatter plot of D_i against (η*T_eff)
#      for all time steps where D_i is valid, and look for linear relationship.
# 4. Harada-Sasa specific:
#    - H-S: ∫_0^τ J(s) ds = (1/2) ∫_0^τ ds ∫_0^τ ds' C_R(s, s')
#    - J(s) is our 'dissipation_rate'. Its integral is total work.
#    - C_R is the hard part: C_R(s, s') = (1/kT_eff) [ ∂<X(s)>/∂h(s')|_{h=0} - θ(s-s') d/ds' <X(s)X(s')>_0 ]
#      where X is an observable (e.g., a parameter θ_i), h is a perturbation field.
#      Directly measuring the response ∂<X(s)>/∂h(s') requires perturbation experiments.
#    - A simpler check might be to see if phases of high dissipation correlate with
#      phases where the simplified FDT (D_i ~ η*T_eff) is more strongly violated.
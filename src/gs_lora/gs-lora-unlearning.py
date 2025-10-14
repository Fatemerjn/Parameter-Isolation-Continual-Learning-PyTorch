import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

class SplitCIFAR100(Dataset):
    def __init__(self, cifar100_dataset, class_indices, task_id):
        self.cifar100_dataset = cifar100_dataset
        self.class_map = {old_idx: new_idx for new_idx, old_idx in enumerate(class_indices)}
        self.task_id = task_id
        self.indices = []
        for i, (_, label) in enumerate(cifar100_dataset):
            if label in self.class_map:
                self.indices.append(i)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, label = self.cifar100_dataset[original_idx]
        new_label = self.class_map[label]
        return image, new_label, self.task_id

def get_cifar100_dataloaders(num_tasks=5, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    classes_per_task = 100 // num_tasks
    task_dataloaders = []
    for task_id in range(num_tasks):
        start_class, end_class = task_id * classes_per_task, (task_id + 1) * classes_per_task
        class_indices = list(range(100))[start_class:end_class]
        train_task_dataset = SplitCIFAR100(train_dataset, class_indices, task_id)
        test_task_dataset = SplitCIFAR100(test_dataset, class_indices, task_id)
        train_loader = DataLoader(train_task_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_task_dataset, batch_size=batch_size, shuffle=False)
        task_dataloaders.append((train_loader, test_loader))
    return task_dataloaders

class ExperienceReplay:
    def __init__(self, buffer_size_per_task=200):
        self.buffer_size_per_task = buffer_size_per_task
        self.buffer = []
    def get_full_buffer(self): return self.buffer
    def on_task_end(self, task_id, train_loader):
        num_samples_to_add = self.buffer_size_per_task
        samples_added = 0
        for images, labels, _ in train_loader:
            for i in range(len(images)):
                if samples_added < num_samples_to_add:
                    self.buffer.append((images[i], labels[i].item(), task_id))
                    samples_added += 1
                else: break
            if samples_added >= num_samples_to_add: break

class MultiHeadResNet(nn.Module):
    def __init__(self, num_tasks, num_classes_per_task):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleList([nn.Linear(in_features, num_classes_per_task) for _ in range(num_tasks)])
    def forward(self, x, task_id):
        features = self.backbone(x)
        return self.heads[task_id](features)

def evaluate(model, test_loaders, device):
    model.eval()
    accuracies = []
    for task_id, loader in enumerate(test_loaders):
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task_id=task_id)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracies.append(100 * correct / total)
    return accuracies

def unlearn_task(model, cl_strategy, task_to_forget, device, unlearn_epochs=5, unlearn_lr=1e-4):
    print(f"\n--- Starting Unlearning Process for Task {task_to_forget + 1} ---")
    
    model.eval()
    
    full_buffer = cl_strategy.get_full_buffer()
    forget_data = [d for d in full_buffer if d[2] == task_to_forget]
    retain_data = [d for d in full_buffer if d[2] != task_to_forget]

    if not forget_data or not retain_data:
        print("Not enough data in buffer to perform unlearning.")
        return

    optimizer = optim.Adam(model.parameters(), lr=unlearn_lr)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    
    num_classes_per_task = model.heads[0].out_features

    for epoch in range(unlearn_epochs):
        random.shuffle(retain_data)
        random.shuffle(forget_data)
        
        num_batches = min(len(retain_data), len(forget_data))
        
        for i in range(num_batches):
            optimizer.zero_grad()
            
            # 1. Loss for Retain Data
            retain_sample = retain_data[i]
            r_img, r_lbl, r_tid = retain_sample
            r_img, r_lbl = r_img.unsqueeze(0).to(device), torch.tensor([r_lbl]).to(device)
            
            retain_outputs = model(r_img, task_id=r_tid)
            loss_retain = criterion_ce(retain_outputs, r_lbl)

            # 2. Loss for Forget Data
            forget_sample = forget_data[i]
            f_img, _, f_tid = forget_sample
            f_img = f_img.unsqueeze(0).to(device)

            forget_outputs = model(f_img, task_id=f_tid)
            
            uniform_dist = torch.full_like(forget_outputs, 1.0 / num_classes_per_task)
            loss_forget = criterion_kl(torch.log_softmax(forget_outputs, dim=1), uniform_dist)
            
            total_loss = loss_retain + loss_forget
            total_loss.backward()
            optimizer.step()

        print(f"Unlearning Epoch {epoch+1}/{unlearn_epochs}, Last Total Loss: {total_loss.item():.4f}")

def main():
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    NUM_TASKS = 5
    
    task_dataloaders = get_cifar100_dataloaders(num_tasks=NUM_TASKS)
    model = MultiHeadResNet(num_tasks=NUM_TASKS, num_classes_per_task=(100 // NUM_TASKS)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    cl_strategy = ExperienceReplay()

    for task_id in range(NUM_TASKS):
        print(f"\n--- Training on Task {task_id + 1}/{NUM_TASKS} ---")
        train_loader, _ = task_dataloaders[task_id]
        for epoch in range(5):
            model.train()
            for images, labels, _ in tqdm(train_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images, task_id=task_id)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        cl_strategy.on_task_end(task_id, train_loader)

    print("\n--- Accuracies BEFORE Unlearning ---")
    test_loaders = [loader for _, loader in task_dataloaders]
    before_accuracies = evaluate(model, test_loaders, DEVICE)
    for i, acc in enumerate(before_accuracies):
        print(f"Task {i+1} Accuracy: {acc:.2f}%")
    print(f"Average Accuracy: {np.mean(before_accuracies):.2f}%")
    
    task_to_forget = 2 
    unlearn_task(model, cl_strategy, task_to_forget, DEVICE)

    print("\n--- Accuracies AFTER Unlearning Task 3 ---")
    after_accuracies = evaluate(model, test_loaders, DEVICE)
    for i, acc in enumerate(after_accuracies):
        status = " (FORGOTTEN)" if i == task_to_forget else " (RETAINED)"
        change = after_accuracies[i] - before_accuracies[i]
        print(f"Task {i+1} Accuracy: {acc:.2f}% {status} | Change: {change:+.2f}%")
    print(f"Average Accuracy: {np.mean(after_accuracies):.2f}%")

if __name__ == "__main__":
    main()
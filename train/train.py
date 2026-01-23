import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

# ============ 超参数（Day4 会优化这里）============
BATCH_SIZE = 64# 批次大小
LEARNING_RATE = 0.001# 学习率
EPOCHS = 3# 训练轮数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")# 设备
SAVE_PATH = "mnist_cnn.pth"

# ============ 数据加载 ============
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))# 数据归一化
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============ 模型、损失、优化器 ============
model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============ 训练循环 ============
print(f"使用设备: {DEVICE}")
print(f"训练开始...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # 👈 loss 在这里算
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 200 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} 完成 | 平均 Loss: {avg_loss:.4f}")

# ============ 保存模型 ============
torch.save(model.state_dict(), SAVE_PATH)
print(f"模型已保存到: {SAVE_PATH}")

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

# ============ 配置 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mnist_cnn.pth"

# ============ 加载测试数据（1万张新图，AI没见过的）============
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ============ 加载训练好的AI ============
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()  # 切换到推理模式（考试模式）

# ============ 测试 ============
correct = 0
total = 0

with torch.no_grad():  # 考试时不需要学习
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f"测试集准确率: {accuracy:.2f}%")
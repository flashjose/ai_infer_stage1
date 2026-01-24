import torch
import os
import sys

# 把 train 文件夹加入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(current_dir, "..", "train")
sys.path.insert(0, train_dir)

from model import SimpleCNN

# ============ 配置 ============
MODEL_PATH = os.path.join(train_dir, "mnist_cnn.pth")
ONNX_PATH = "mnist_cnn.onnx"

# ============ 加载训练好的模型 ============
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ============ 创建假输入 ============
dummy_input = torch.randn(1, 1, 28, 28)

# ============ 导出 ============
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    opset_version=11,
    input_names=["input"],
    output_names=["output"]
)

print(f"✅ ONNX 模型已保存到: {ONNX_PATH}")
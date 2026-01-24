
import onnxruntime as ort
import numpy as np
import time

# ============ 配置 ============
ONNX_PATH = "../export/mnist_cnn.onnx"

# ============ 加载 ONNX 模型 ============
session = ort.InferenceSession(ONNX_PATH)

# 获取输入输出名字
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"输入名: {input_name}")
print(f"输出名: {output_name}")

# ============ 创建测试数据 ============
# 模拟一张 28x28 的图片（随机数据）
dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)

# ============ 推理 ============
# start_time = time.time()
# output = session.run([output_name], {input_name: dummy_input})
# end_time = time.time()
times = []
for i in range(100):
    start_time = time.time()
    output = session.run([output_name], {input_name: dummy_input})
    end_time = time.time()
    times.append(end_time - start_time)

avg_time = sum(times) / len(times) * 1000

# ============ 结果 ============
result = output[0]
predicted = np.argmax(result)  # 取最大值的位置

print(f"输出 shape: {result.shape}")
print(f"预测数字: {predicted}")
print(f"推理时间: {(end_time - start_time) * 1000:.2f} ms")
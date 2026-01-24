# ai_infer_stage1
# AI Inference Engineering - Stage 1

## Project Overview
This project focuses on creating an end-to-end AI inference pipeline, from training a model to exporting it in ONNX format and performing inference using ONNX Runtime.
用 PyTorch 训练 MNIST 手写数字识别模型，导出 ONNX，用 ONNX Runtime 推理。
## Environment
- Python 3.8+
- PyTorch
- ONNX Runtime
- Ubuntu 22.04
- GPU (可选，有 CUDA 更快)

## 安装依赖
```bash
pip install -r requirements.txt
```
## Training
- Dataset: MNIST / CIFAR-10
- Model: Simple Neural Network (MLP)
- Training Script: `train/train.py`
```bash
cd train
python train.py
```
## 准确率测试
```bash
cd train
python test.py
```

## Export ONNX
- Export Model: `export/export_onnx.py`
```bash
cd export
python export_onnx.py
```
## Python Inference
- Run Inference: `infer_py/infer.py`
```bash
cd infer_py
python infer.py
```
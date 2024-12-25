# imageclassify

## 导入必要库

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import *
```

- torch: PyTorch 的核心库。
- torch.nn: 包含构建神经网络的模块。
- torch.nn.functional: 提供了不包含参数的函数（如激活函数、损失函数）。
- torch.optim: 提供优化器实现。
- torchvision: 提供图像处理工具和数据集。
- torchvision.transforms: 用于对图像进行变换和数据增强。
- from model import *: 假设引入其他模型代码。

## 定义残差块

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        ...
```

- Residual Block 的功能:
    - 引入跳跃连接（Skip Connection），缓解梯度消失问题。
    - 在网络中加入残差连接，保证信息流动更加顺畅。
- 核心模块: 
    1. 卷积层 + BatchNorm + ReLU:
        - 两次卷积操作 (conv1 和 conv2) 用于提取特征。
        - BatchNorm 用于加速训练收敛和减轻过拟合。
        - ReLU 激活函数引入非线性。
    2. 跳跃连接 (identity):
        - 当需要调整通道或分辨率（通过 stride）时，使用 downsample。
        - 将输入直接加到输出上形成残差。

## 定义 ResNet 网络

```python
class ResNet(nn.Module):
    def __init__(self):
        ...
```

- 核心组件:
    1. 初始卷积层:
        - self.conv1 是网络的起始卷积层。
        - 接收输入图像（3通道）并生成64通道特征图。
    2. 残差层生成函数 (_make_layer):
        - 构建由多个 ResidualBlock 组成的网络层。
        - 每一层包括：
            - 第一个块（可能有 downsample）。
            - 后续块（保持通道和分辨率）。
    3. 全局平均池化和全连接层:
        - AdaptiveAvgPool2d 将特征图大小固定为 1x1。
        - fc 为最终的分类器，将特征向量映射到 10 个类别。

- forward 方法:
    - 将输入依次通过卷积、残差层、池化和全连接层。

## 检查设备

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

检查是否有可用 GPU。
如果有，使用 CUDA 加速，否则使用 CPU。

## 数据预处理

```python
transform_train = transforms.Compose([...])
transform_test = transforms.Compose([...])
``` 

- 使用 transforms.Compose 链接多个数据增强操作。
- 训练集增强:
    - 随机水平翻转 (RandomHorizontalFlip) 和随机裁剪 (RandomCrop) 增加数据多样性。
- 归一化:
    - 将图像像素从 [0, 1] 映射到 [-1, 1]，提高训练稳定性。

## 加载 CIFAR-10 数据集

```python
trainset = torchvision.datasets.CIFAR10(...)
trainloader = torch.utils.data.DataLoader(...)
testset = torchvision.datasets.CIFAR10(...)
testloader = torch.utils.data.DataLoader(...)
```

- CIFAR-10 数据集:
    - 10 个类别，每张图片为 32x32 大小。
- DataLoader:
    - 提供批量数据加载和自动打乱数据的功能。

## 模型初始化

```python
net = ResNet().to(device)
```

创建 ResNet 实例，并将模型加载到指定设备（GPU/CPU）。

## 定义损失函数和优化器

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
```

- 损失函数:
    - 使用 CrossEntropyLoss，适合多分类任务。
- 优化器:
    - 使用 Adam 优化器，具有自适应学习率的优势。

## 训练模型

```python
for epoch in range(20):
    ...
```

- 步骤:
    - 模型切换到训练模式 (net.train())。
    - 获取输入 inputs 和标签 labels。
    - 清空梯度 (optimizer.zero_grad())，计算损失，反向传播 (loss.backward())，更新参数 (optimizer.step())。
    - 打印每轮训练的平均损失。

## 测试模型

```python
net.eval()
correct = 0
total = 0
with torch.no_grad():
    ...
```

- 测试模式:
    - 切换到测试模式 (net.eval())。
    - 禁用梯度计算，提升推理速度。
- 计算准确率:
    - 使用 torch.max 获取预测类别。
    - 比较预测结果和真实标签，统计正确数量。

## 打印测试集准确率

```python
print(f"Accuracy on 10000 test images: {100 * correct / total}%")
```

打印模型在测试集上的准确率，用于评估性能。






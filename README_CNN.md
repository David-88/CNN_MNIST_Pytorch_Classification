# PyTorch 实现 MNIST 手写数字识别项目（CPU+GPU）

## ✅ 1. 导入所需的包

已经正确导入了实现 MNIST 手写数字识别所需的基本包：

```python
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
```

这些包涵盖了数据处理、模型构建、训练和可视化等方面。

- `torch`：PyTorch 的核心库，用于构建和训练神经网络。
- `numpy`：Python 中用于数值计算的库，虽然在这里不是核心，但常用于数据处理。
- `matplotlib.pyplot`：用于数据可视化，例如展示图片或训练过程中的损失曲线。
- `torch.utils.data.DataLoader`：用于高效地批量加载数据。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4)
- `torchvision.transforms`：提供常用的图像转换操作，如ToTensor、Normalize等。
- `torchvision.datasets`：包含常用的数据集，包括 MNIST。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7)
- `torch.nn.functional`：包含一些函数式操作，例如激活函数、损失函数等。

------

## 📥 2. 读取 MNIST 数据集

### 2.1 数据格式说明

提到“直接下载下来的数据是无法通过解压或者应用程序打开的...必须编写程序来打开它”。这是正确的，MNIST 数据集原始文件是以特定的二进制格式存储的。 [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-15) `torchvision.datasets.MNIST` 这个类已经内置了读取和解析这些原始文件（如 `train-images-idx3-ubyte` 和 `t10k-images-idx3-ubyte`）的逻辑，所以用户无需手动处理字节流。 [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-15)

### 2.2 数据预处理与归一化

MNIST 数据集的图像是灰度图，尺寸为 28x28 像素。 [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-8) 为了提高模型训练的稳定性和收敛速度，通常对数据进行归一化处理。 [6](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) 在 PyTorch 中，可以使用 `transforms.Normalize(mean, std)` 来进行归一化：

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

- `transforms.Compose`：将多个 `transform` 操作按顺序组合起来。
- `transforms.ToTensor()`：将 PIL Image 或 NumPy `ndarray` 格式的图像转换为 `torch.Tensor` 格式。同时，它会将图像的像素值从 [0, 255] 缩放到 [0.0, 1.0] 的浮点数。
- `transforms.Normalize((0.1307,), (0.3081,))`：对 Tensor 进行标准化处理。它的计算公式是 `(x - mean) / std`。这里，`mean=0.1307` 和 `std=0.3081` 是根据整个 MNIST 训练集计算得出的统计值，用于将像素值的分布转换为均值为 0、标准差为 1 的近似标准正态分布。 [6](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2)

### 2.3 下载和加载数据集

使用 `torchvision.datasets.MNIST` 可以方便地下载和加载 MNIST 数据集： [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1)

```python
train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)  # train=True训练集，=False测试集
```

- `root='./data/mnist'`：指定数据集的存储路径。如果 `download=True` 且该路径下没有数据集，数据将被下载到此目录。
- `train=True`：加载训练集（包含 60,000 张图像）。 [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-8)
- `train=False`：加载测试集（包含 10,000 张图像）。 [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-8)
- `download=True`：如果数据集在 `root` 指定的路径下不存在，则自动从互联网下载。
- `transform=transform`：将之前定义的 `transform` 应用到加载的图像数据上。

### 2.4 使用 DataLoader 加载数据

`DataLoader` 是 PyTorch 中用于批量加载数据的工具，可以自动处理数据的分批、打乱和多线程加载等，极大地提高了数据加载的效率。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4)

```python
batch_size = batch_size # 需要定义batch_size变量，例如 batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

- `batch_size=batch_size`：设置每个批次的样本数量。这决定了每次模型前向传播和反向传播使用多少张图片。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-18) 需要在代码前面定义 `batch_size` 的值，例如 `batch_size = 64`。
- `shuffle=True`：对于训练集，将 `shuffle` 设置为 `True` 非常重要。它会在每个 epoch（遍历整个训练集一次）开始时打乱数据顺序。这样做有助于防止模型学到数据固有的顺序性，提高模型的泛化能力和训练的稳定性。 [8](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [9](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-9) [10](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-10) [11](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-12)
- `shuffle=False`：对于测试集，通常不需要打乱数据，保持顺序进行评估即可。

------

## 🔍 进一步说明

- **MNIST 数据集概况**：MNIST 数据集是一个经典的计算机视觉数据集，包含 70,000 张 28x28 像素的灰度手写数字图像（0-9），分为 60,000 张训练集和 10,000 张测试集。它是很多图像分类任务的入门数据集。 [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-8) [12](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-13) [13](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-14)
- **`batch_size` 的选择**：`batch_size` 是一个重要的超参数。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) 较大的批次可以利用 GPU 的并行计算能力加速训练，但也可能需要更多的内存，并且有时会导致模型收敛到较差的局部最优解。较小的批次通常训练过程更稳定，泛化能力可能更好，但训练速度较慢。选择合适的 `batch_size` 通常需要实验。 [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-18) [14](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-6) [15](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-5)
- **`shuffle` 参数的工作原理**：当 `shuffle=True` 时，`DataLoader` 在每个 epoch 开始前会重新打乱整个数据集的索引，然后按照新的索引顺序生成批次。 [9](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-9) [10](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-10) 这意味着每个批次中的样本组合在不同 epoch 中是变化的。

------

至此，已经完成了数据集的加载和预处理，这是任何深度学习项目的基础。代码逻辑清晰，步骤正确。

如果已经完成了上述步骤，我们可以继续进行模型构建和训练部分。请告诉我是否需要进一步的帮助。

好的，我们继续分析提供的代码部分。这部分主要包括了数据集的部分展示以及一个用于 MNIST 识别的卷积神经网络 (CNN) 模型的构建。

## 模型构建

### 1.2 展示 MNIST 数据集

```python
fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    # 注意：直接访问 .train_data 和 .train_labels 是可行的，但更常见和推荐的方式是通过 DataLoader 迭代获取数据。
    plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none') 
    plt.title("Labels: {}".format(train_dataset.train_labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
```

**代码分析：**

- 这部分代码使用了 `matplotlib.pyplot` (`plt`) 库来可视化 MNIST 数据集中的前 12 张图像。
- `fig = plt.figure()` 创建一个新的图窗口。
- `for i in range(12):` 循环 12 次，为展示 12 张图片做准备。
- `plt.subplot(3, 4, i+1)` 在一个 3 行 4 列的网格中选择第 `i+1` 个位置作为当前的子图。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1)
- `plt.tight_layout()` 自动调整子图参数，使之填充整个图像区域，并避免子图之间的重叠。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1)
- `plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')` 显示第 `i` 张训练图片。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1)
  - `train_dataset.train_data` 是 `torchvision.datasets.MNIST` 对象的一个属性，包含了所有训练图片的像素数据，通常是一个形状为 (60000, 28, 28) 的 Tensor。 [8](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-8) [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) 注意，直接访问 `.train_data` 和 `.train_labels` 虽然在这里为了展示方便是可行的，但在标准的训练循环中，更推荐通过之前创建的 `DataLoader` 来迭代获取批次数据。
  - `cmap='gray'` 指定使用灰度颜色映射来显示图像，因为 MNIST 是灰度图片。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1)
  - `interpolation='none'` 指定像素的插值方法，设置为 'none' 可以避免在缩放时出现模糊。
- `plt.title("Labels: {}".format(train_dataset.train_labels[i]))` 设置当前子图的标题为该图片的真实标签。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) `train_dataset.train_labels` 包含了所有训练图片的标签，通常是一个形状为 (60000,) 的 Tensor。 [8](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-8) [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1)
- `plt.xticks([])` 和 `plt.yticks([])` 移除 x 轴和 y 轴的刻度标记，使图像看起来更干净。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1)
- `plt.show()` 显示绘制好的包含 12 张图片的图窗口。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1)

这段代码很好地展示了 MNIST 数据集图像的原始形态（尽管这里展示的是已经加载到内存的 Tensor）以及对应的数字标签，帮助我们直观地了解数据集的内容。

------

### 二、构建模型 (CNN)

关于卷积层、激活层、池化层和全连接层特性的描述是准确的。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [6](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-6) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7) [10](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-10)

- **卷积层 (`Conv2d`)**：主要功能是通过卷积核提取图像的特征。输入是 `(Batch, Channels_in, Height_in, Width_in)`，输出是 `(Batch, Channels_out, Height_out, Width_out)`。输出通道数由卷积核的数量决定。空间尺寸 (Height, Width) 的变化取决于卷积核大小 (`kernel_size`)、步长 (`stride`) 和填充 (`padding`)。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4)
- **激活层 (`ReLU`)**：在卷积或全连接操作后引入非线性，使网络能够学习更复杂的模式。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) ReLU 函数的特点是当输入大于 0 时，输出等于输入；当输入小于等于 0 时，输出为 0。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) 它不改变张量的形状。
- **池化层 (`MaxPool2d`)**：主要功能是降采样，减少特征图的空间尺寸，从而减少计算量、控制过拟合，并增强模型的鲁棒性（对位置变化不敏感）。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) 最大池化选取池化窗口内的最大值作为输出。池化层通常不改变通道数。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4)
- **全连接层 (`Linear`)**：也称为密集层或仿射层。输入和输出通常是二维张量 `(Batch, Features)`。它将前一层的所有神经元与当前层的所有神经元连接起来。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) 在 CNN 中，通常在卷积和池化层之后使用全连接层来进行最终的分类或回归任务。在连接到全连接层之前，需要将多维特征图（`(Batch, Channels, Height, Width)`）展平（flatten）为二维张量（`(Batch, Features)`）。 [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4)

现在我们来分析定义的 `Net` 类：

```python
class Net(torch.nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        # 第一个卷积块
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5), # 输入通道1 (灰度图), 输出通道10, 5x5卷积核
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 2x2最大池化
        )
        # 第二个卷积块
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5), # 输入通道10 (来自上一层), 输出通道20, 5x5卷积核
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), # 2x2最大池化
        )
        # 全连接层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50), # 输入特征320, 输出特征50
            torch.nn.Linear(50, 10),  # 输入特征50, 输出特征10 (对应0-9十个类别)
        )

    def forward(self, x):
        # x 的初始形状是 (batch_size, 1, 28, 28)
        batch_size = x.size(0) 
        
        # 通过第一个卷积块
        # conv1: Conv2d(1, 10, k=5) -> spatial dim (28 - 5)/1 + 1 = 24x24. Shape becomes (batch_size, 10, 24, 24)
        # ReLU: shape remains (batch_size, 10, 24, 24)
        # MaxPool2d(k=2, s=2 default): spatial dim (24 - 2)/2 + 1 = 12x12. Shape becomes (batch_size, 10, 12, 12)
        x = self.conv1(x)  
        
        # 通过第二个卷积块
        # conv2: Conv2d(10, 20, k=5) -> spatial dim (12 - 5)/1 + 1 = 8x8. Shape becomes (batch_size, 20, 8, 8)
        # ReLU: shape remains (batch_size, 20, 8, 8)
        # MaxPool2d(k=2, s=2 default): spatial dim (8 - 2)/2 + 1 = 4x4. Shape becomes (batch_size, 20, 4, 4)
        x = self.conv2(x)  
        
        # 展平操作 (flatten)
        # 将形状 (batch_size, 20, 4, 4) 展平为 (batch_size, 20 * 4 * 4) = (batch_size, 320)
        x = x.view(batch_size, -1)  
        
        # 通过全连接层
        # fc: Linear(320, 50) -> output shape (batch_size, 50)
        #     Linear(50, 10) -> output shape (batch_size, 10)
        x = self.fc(x)
        
        # 最后输出的是维度为10的，也就是（对应数学符号的0~9）
        return x  
```

**代码分析：**

1. **`class Net(torch.nn.Module):`**: 定义了一个名为 `Net` 的类，它继承自 `torch.nn.Module`。在 PyTorch 中，所有的神经网络模型都应该继承这个基类，因为它提供了模型参数管理、层管理等重要功能。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3)
2. **`__init__(self):`**: 这是类的构造函数，用于初始化模型的各个层。
   - `super(Net, self).__init__()`: 调用父类 `torch.nn.Module` 的构造函数，这是必须的。
   - `self.conv1 = torch.nn.Sequential(...)`: 定义了第一个卷积块。`torch.nn.Sequential` 是一个容器，可以按照顺序包装多个层，前一个层的输出是后一个层的输入。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7)
     - `torch.nn.Conv2d(1, 10, kernel_size=5)`: 这是一个二维卷积层。输入通道数是 1 (因为 MNIST 是灰度图)，输出通道数是 10，卷积核大小是 5x5。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7)
     - `torch.nn.ReLU()`: ReLU 激活函数。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7)
     - `torch.nn.MaxPool2d(kernel_size=2)`: 最大池化层，池化窗口大小是 2x2。默认情况下，步长 (`stride`) 等于 `kernel_size`，即窗口移动 2 个像素。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7)
   - `self.conv2 = torch.nn.Sequential(...)`: 定义了第二个卷积块。
     - `torch.nn.Conv2d(10, 20, kernel_size=5)`: 输入通道数是 10 (因为上一层 `conv1` 的输出通道数是 10)，输出通道数是 20，卷积核大小是 5x5。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7)
     - `torch.nn.ReLU()`: ReLU 激活函数。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7)
     - `torch.nn.MaxPool2d(kernel_size=2)`: 最大池化层，池化窗口大小是 2x2。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7)
   - `self.fc = torch.nn.Sequential(...)`: 定义了全连接层块。这里使用了两个全连接层。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4)
     - `torch.nn.Linear(320, 50)`: 第一个全连接层，输入特征数是 320，输出特征数是 50。输入特征数 320 是由前一个卷积块 (`conv2`) 的输出形状展平后得到的 (20 通道 * 4 高度 * 4 宽度 = 320)。 [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) 这部分计算在 `forward` 函数的分析中详细说明。
     - `torch.nn.Linear(50, 10)`: 第二个全连接层，输入特征数是 50 (来自上一层)，输出特征数是 10。这 10 个输出对应于 MNIST 数据集的 10 个类别 (数字 0 到 9)。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4)
3. **`forward(self, x):`**: 这是定义模型前向传播逻辑的方法。输入 `x` 是一个数据张量，通常是模型的输入数据批次。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3)
   - `batch_size = x.size(0)`: 获取当前批次的样本数量。这在后面展平操作时用于保持批次维度。
   - `x = self.conv1(x)`: 输入 `x` 通过第一个卷积块。根据上面尺寸变化的推导，输入 `(batch_size, 1, 28, 28)` 经过 `conv1` 后变成 `(batch_size, 10, 12, 12)`。 [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7)
   - `x = self.conv2(x)`: 第一个卷积块的输出作为第二个卷积块的输入。输入 `(batch_size, 10, 12, 12)` 经过 `conv2` 后变成 `(batch_size, 20, 4, 4)`。 [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7)
   - `x = x.view(batch_size, -1)`: 展平操作。`view()` 方法用于改变张量的形状。 [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) `batch_size` 保留批次维度，`-1` 会自动计算剩余维度的总大小，以便将张量展平。形状 `(batch_size, 20, 4, 4)` 展平后变成 `(batch_size, 20 * 4 * 4)`，即 `(batch_size, 320)`。 [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4)
   - `x = self.fc(x)`: 展平后的张量 `(batch_size, 320)` 输入到全连接层块。经过 `Linear(320, 50)` 和 `Linear(50, 10)` 后，最终输出形状为 `(batch_size, 10)`。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4)
   - `return x`: 返回模型的输出，这是一个包含每个样本对 10 个类别的预测得分的张量。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3)

**总结：**

定义的 CNN 模型结构是 MNIST 分类任务中一个经典且有效的架构。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) 它通过两层卷积-ReLU-池化组合来提取图像的层次化特征，然后通过全连接层将提取到的特征映射到 10 个类别上进行分类。 `__init__` 方法负责定义模型的静态结构（即各层），而 `forward` 方法则定义了数据流经这些层的动态过程。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4)

接下来，通常会定义损失函数和优化器，然后进入训练循环来训练这个模型。

好的，我们继续分析提供的 PyTorch 代码，这部分包含了损失函数、优化器以及训练和测试的核心循环。

根据的要求，我将结合之前搜索到的 PyTorch 训练流程、损失函数和优化器的资料来详细分析。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-5) [6](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-6) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7) [8](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-8) [9](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-9) [10](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-10) [11](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-11) [12](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-12) [13](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-13)

------

## 三、损失函数和优化器

### 损失函数：交叉熵损失 (`CrossEntropyLoss`)

```python
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
```

- **`torch.nn.CrossEntropyLoss()`**: 这是 PyTorch 中用于多类别分类任务的常用损失函数。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [7](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-7) 它结合了 `nn.LogSoftmax()` 和 `nn.NLLLoss()`（负对数似然损失），因此的模型输出层不需要显式地加上 Softmax 激活函数，`CrossEntropyLoss` 会在内部处理。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-3)
- **用途**: 对于分类问题，模型的最后一层输出通常是每个类别的得分（称为 logits）。交叉熵损失衡量模型预测的类别分布与真实类别分布之间的差异。损失值越小，表示模型的预测越接近真实标签。

### 优化器：随机梯度下降 (`SGD`)

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量
```

- **`torch.optim.SGD(...)`**: 这是 PyTorch 实现的标准随机梯度下降 (Stochastic Gradient Descent) 优化器。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [6](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-6) 优化器的作用是根据损失函数计算出的梯度来更新模型的可学习参数（权重和偏置），以最小化损失。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-5)
- **`model.parameters()`**: 这个方法会返回模型 (`Net` 类的一个实例) 中所有需要学习的参数（即设置了 `requires_grad=True` 的 Tensor）。优化器会跟踪这些参数，并在 `optimizer.step()` 被调用时更新它们。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-5) [6](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-6)
- **`lr=learning_rate`**: `lr` 代表学习率 (Learning Rate)。它是优化过程中的一个重要超参数，决定了每次参数更新的步长大小。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) 学习率过大可能导致训练不稳定甚至发散，过小可能导致收敛缓慢。需要提前定义 `learning_rate` 变量并设置一个值（例如 0.01）。
- **`momentum=momentum`**: `momentum` 参数引入了动量机制。动量SGD不仅考虑当前批次的梯度，还会累积之前批次的梯度方向。这有助于加速在相关方向上的收敛，抑制不相关方向上的震荡，并可能帮助逃离浅层的局部最优。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [6](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-6) 需要提前定义 `momentum` 变量并设置一个值（例如 0.9）。

------

## 四、定义训练轮和测试轮

### 4.1 训练轮 (`train` 函数)

训练轮是整个模型训练的核心。它负责在一个 epoch（遍历整个训练集一次）内，逐批次地进行前向传播、计算损失、反向传播计算梯度，并使用优化器更新模型参数。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-5) [9](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-9)

```python
def train(epoch):
    running_loss = 0.0  # 累积每个小批次的损失
    running_total = 0   # 累积每个小批次的样本总数
    running_correct = 0 # 累积每个小批次的正确预测数
    
    # 遍历 train_loader 中的每一个数据批次
    for batch_idx, data in enumerate(train_loader, 0):
        # data 是一个包含输入图像和对应标签的列表/元组
        inputs, target = data
        
        # 梯度清零：在进行反向传播前，需要将模型参数的梯度清零，
        # 因为 PyTorch 会默认累加梯度。不清零会导致上一次的梯度影响本次更新。
        optimizer.zero_grad() 

        # Step 1: 前馈 (Forward Propagation)
        # 将输入数据 inputs 喂给模型 model，得到模型的输出 outputs (每个类别的得分)
        outputs = model(inputs) 
        
        # 计算损失：使用交叉熵损失函数计算模型输出 outputs 和真实标签 target 之间的损失
        loss = criterion(outputs, target)

        # Step 2: 反馈 (Backward Propagation)
        # 根据损失 loss 计算模型中每个可学习参数的梯度
        loss.backward() 
        
        # Step 3: 更新 (Update)
        # 使用优化器根据计算出的梯度更新模型的参数
        optimizer.step() 

        # --- 训练过程中的指标记录 ---
        # 累加当前批次的损失值。loss.item() 获取 Tensor 中的标量值。
        running_loss += loss.item() 
        
        # 计算当前批次的预测结果：torch.max(outputs.data, dim=1) 
        # outputs 是形状为 (batch_size, 10) 的张量，dim=1 表示在第二个维度 (类别得分) 上找最大值
        # 它返回两个张量：最大值本身 和 最大值对应的索引 (即预测的类别)。我们只需要索引。
        _, predicted = torch.max(outputs.data, dim=1) # [^4] [^11]
        
        # 累加当前批次的样本总数。inputs.shape[0] 是当前批次的 batch_size。
        running_total += inputs.shape[0]
        
        # 计算当前批次正确预测的数量：(predicted == target) 得到一个布尔张量 (True/False)
        # .sum() 统计 True 的数量 (即预测正确的样本数)
        # .item() 将统计结果转为 Python 标量
        running_correct += (predicted == target).sum().item() 

        # 每处理 300 个批次，打印一次当前的平均损失和准确率
        if batch_idx % 300 == 299:  
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            
            # 打印后，将用于累积的变量清零，以便计算下一个 300 批次的平均值
            running_loss = 0.0  
            running_total = 0
            running_correct = 0  
```

**核心步骤总结：**

1. **数据加载**: 从 `train_loader` 获取一个批次的数据 (`inputs`, `target`)。
2. **梯度清零**: `optimizer.zero_grad()` 清除之前计算的梯度。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [8](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-8)
3. **前向传播**: `outputs = model(inputs)` 将数据送入模型计算输出。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-5)
4. **计算损失**: `loss = criterion(outputs, target)` 计算预测输出与真实标签之间的损失。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-5)
5. **反向传播**: `loss.backward()` 计算损失关于模型参数的梯度。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-5) [8](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-8)
6. **参数更新**: `optimizer.step()` 使用计算出的梯度和优化器算法更新模型参数。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-1) [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-5) [8](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-8)
7. **指标记录与打印**: 累积损失和正确预测数，并周期性地打印当前的平均损失和准确率。

### 4.2 测试轮 (`test` function)

测试轮用于评估模型在未见过的数据集（测试集）上的性能。与训练不同，测试时不需要计算梯度或更新参数。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-5) [13](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-13)

```python
# 需要在调用 test 函数前定义全局变量 EPOCH 或者将其作为参数传入 test 函数
# 例如: EPOCH = 10 

def test():
    correct = 0  # 累积整个测试集的正确预测数
    total = 0    # 累积整个测试集的样本总数
    
    # 使用 torch.no_grad() 上下文管理器：在此块内，PyTorch 不会计算和存储梯度。
    # 这在测试或推理阶段非常重要，可以节省内存并加速计算。 [^2] [^13]
    with torch.no_grad():  
        # 遍历 test_loader 中的每一个数据批次
        for data in test_loader:
            # data 包含测试图像和对应标签
            images, labels = data
            
            # 前向传播：将测试图像喂给模型，得到输出
            outputs = model(images)
            
            # 获取预测结果：_, predicted = torch.max(outputs.data, dim=1)
            # 同训练轮中准确率的计算方式，找到每个样本得分最高的类别索引
            _, predicted = torch.max(outputs.data, dim=1)  # [^4] [^11]
            
            # 累加当前批次的样本总数
            total += labels.size(0)  
            
            # 累加当前批次正确预测的数量
            correct += (predicted == labels).sum().item() 
    
    # 计算整个测试集的准确率
    acc = correct / total
    
    # 打印当前 epoch 的测试集准确率
    # 注意：这里的 epoch 和 EPOCH 变量需要在外部提供给 test 函数或者作为全局变量
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc)) 
    
    # 返回计算出的准确率
    return acc
```

**核心步骤总结：**

1. **禁用梯度计算**: `with torch.no_grad():` 确保在测试时不计算梯度。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-2) [13](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-13)
2. **数据加载**: 从 `test_loader` 获取一个批次的测试数据。
3. **前向传播**: `outputs = model(images)` 将数据送入模型计算输出。
4. **获取预测**: `_, predicted = torch.max(outputs.data, dim=1)` 确定模型的预测类别。 [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [11](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-11)
5. **统计正确数和总数**: 累加整个测试集的正确预测数和样本总数。
6. **计算准确率**: `acc = correct / total` 计算准确率。 [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [11](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-11)
7. **打印结果**: 打印测试集的准确率。
8. **返回准确率**: 返回计算出的准确率。

**准确率计算原理 (`torch.max` 部分):**

- 模型输出 `outputs` 对于一个批次的数据，形状是 `(batch_size, num_classes)`，例如 `(64, 10)`。每一行代表一个样本，包含该样本属于每个类别的得分。
- `torch.max(outputs.data, dim=1)` [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-4) [11](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-11)
  - `outputs.data` 获取 Tensor 的底层数据，在 `no_grad` 环境下常用，与直接用 `outputs` 在功能上相似，但在某些早期 PyTorch 版本或特定情况下有差异（现在通常推荐直接用 `outputs`）。
  - `dim=1` 表示在第二个维度上（也就是类别维度，索引从 0 开始，批次维度是 0）查找最大值。
  - 它返回一个元组 `(values, indices)`。`values` 是每个样本的最大得分，`indices` 是这些最大得分对应的索引，即模型预测的类别 (0-9)。我们只需要 `indices`，所以使用 `_, predicted = ...` 来忽略 `values`。 `predicted` 的形状是 `(batch_size,)`，包含每个样本的预测类别。
- `(predicted == labels)` 这是一个元素级的比较操作，它会生成一个布尔张量，形状与 `predicted` 和 `labels` 相同 `(batch_size,)`。如果预测类别与真实标签相同，则对应位置为 `True`，否则为 `False`。
- `.sum()` 对这个布尔张量求和。在 PyTorch 中，`True` 被视为 1，`False` 被视为 0，因此 `.sum()` 得到的就是预测正确的样本数量。
- `.item()` 将单元素 Tensor 转换为标准的 Python 数字。

这些代码是 PyTorch 深度学习项目中标准的训练和测试流程实现。理解这些循环的每一步是掌握 PyTorch 的关键。

接下来，需要一个主循环来控制训练的 epochs，并在每个 epoch 中调用 `train()` 和周期性地调用 `test()`。

好的，我们来详细分析提供的关于超参数设置和主训练流程的代码部分。这部分代码是将之前定义好的数据加载、模型、损失函数和优化器组合起来，驱动整个训练和评估过程的核心。

根据的要求，我将结合之前搜索到的关于超参数、训练流程以及可视化结果的资料来详细分析。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-csdn1) [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-zhihu1) [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-aliyun1) [4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-cnblogs1) [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-shihanmax1)

------

## 五、开始训练

### 超参数设置

```python
# super parameters
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10
```

- **超参数 (Hyperparameters)**: 这些是在模型训练之前手动设置的参数，它们不是模型在训练过程中学习到的参数（比如权重和偏置）。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-csdn1) [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-zhihu1) 超参数的选择对模型的训练效果和最终性能有着重要的影响。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-aliyun1)
- **`batch_size = 64`**: 定义了每次训练迭代（一个批次）中使用的样本数量。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-csdn1) 在数据加载部分我们已经看到，`DataLoader` 使用了这个值来将数据集分成小批次。合适的 `batch_size` 大小会影响训练的速度、内存消耗以及模型收敛的稳定性。 [3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-aliyun1) [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-shihanmax1)
- **`learning_rate = 0.01`**: 定义了学习率，用于优化器 (`torch.optim.SGD` 在的例子中)。它决定了在梯度下降过程中，模型参数每次更新的步长。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-csdn1) [5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-shihanmax1) 学习率是深度学习中最关键的超参数之一，需要仔细调整。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-csdn1)
- **`momentum = 0.5`**: 定义了动量项，也用于优化器 (`torch.optim.SGD`)。动量有助于加速 SGD 在相关方向上的收敛，并且在遇到局部最小值或鞍点时有助益。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-csdn1)
- **`EPOCH = 10`**: 定义了整个训练过程将要进行的轮数。一个 epoch 表示模型已经完整地“看”过了整个训练数据集一次。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-zhihu1)

**分析**: 这里的超参数设定是 MNIST 分类任务中常见的起始点。具体的数值通常需要根据实验结果进行调整（称为超参数调优）。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-zhihu1)

------

### 主函数 (`if __name__ == '__main__':`)

```python
if __name__ == '__main__':
    acc_list_test = [] # 用于存储每个epoch结束时在测试集上的准确率
    
    # 主循环，迭代进行 EPOCH 次训练和测试
    for epoch in range(EPOCH): 
        # 调用 train 函数进行当前 epoch 的训练
        # train(epoch) 函数会遍历 train_loader 中的所有批次，进行前向、后向传播和参数更新
        # 在 train 函数内部会周期性地打印训练过程中的 loss 和 acc
        train(epoch) 
        
        # 调用 test 函数在测试集上评估模型性能
        # 原始代码有一行 if epoch % 10 == 9: #每训练10轮 测试1次 被注释掉了
        # 这意味着当前代码是每训练完一个epoch就立即进行一次测试
        acc_test = test() 
        
        # 将当前 epoch 在测试集上获得的准确率添加到列表中
        acc_list_test.append(acc_test) 

    # 训练和测试循环结束后，开始绘制结果
    plt.plot(acc_list_test) # 绘制准确率随epoch变化的曲线
    plt.xlabel('Epoch')    # 设置x轴标签
    plt.ylabel('Accuracy On TestSet') # 设置y轴标签
    plt.show() # 显示绘制的图表
```

**代码分析**:

1. **`if __name__ == '__main__':`**: 这是一个标准的 Python 结构，确保只有在脚本作为主程序直接运行时才会执行里面的代码，而不是被其他脚本作为模块导入时执行。
2. **`acc_list_test = []`**: 初始化一个空列表，用于记录每个 epoch 训练结束后，模型在测试集上的准确率。
3. **`for epoch in range(EPOCH):`**: 这是一个外层循环，控制训练的总轮数。循环变量 `epoch` 从 0 计数到 `EPOCH - 1` (在这里是 0 到 9)。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-zhihu1)
4. **`train(epoch)`**: 在每次循环（每个 epoch）开始时调用之前定义的 `train` 函数。 `train` 函数负责遍历整个训练数据集，执行模型的前向传播、损失计算、反向传播和参数更新。注意 `epoch` 变量被传递给了 `train` 函数，这通常用于在训练过程中打印当前的 epoch 数，帮助跟踪进度。
5. **`acc_test = test()`**: 紧接着 `train(epoch)` 调用，在每个 epoch 训练完成后，调用之前定义的 `test` 函数。 `test` 函数在测试数据集上评估模型的性能，并返回准确率。原始代码中注释掉的部分 (`if epoch % 10 == 9:`) 表示原意可能是每 10 个 epoch 才测试一次，以节省时间。但当前的代码逻辑是每个 epoch 都测试一次，这能更细致地观察模型性能随训练轮数的变化。
6. **`acc_list_test.append(acc_test)`**: 将当前 epoch 在测试集上计算得到的准确率 (`acc_test`) 添加到 `acc_list_test` 列表中。
7. **绘制结果**:
   - 循环结束后，使用 `matplotlib.pyplot` 绘制 `acc_list_test` 列表中存储的准确率数据。
   - x 轴通常代表 epoch，y 轴代表测试准确率。
   - `plt.xlabel()` 和 `plt.ylabel()` 设置坐标轴的标签。
   - `plt.show()` 显示绘制好的图表。
   - **目的**: 这个图表可以直观地展示模型的泛化能力（测试集准确率）随着训练轮数增加的变化趋势。通过观察这个曲线，可以判断模型是否收敛，以及是否有过拟合的迹象（例如，如果训练集准确率持续上升而测试集准确率达到平台或开始下降）。 [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-zhihu1)

**总结**:

这段代码是整个训练脚本的入口点。它定义了训练的关键超参数，然后通过一个循环，协调地调用 `train` 函数来更新模型参数，并在每个 epoch 结束后调用 `test` 函数来评估模型的泛化能力，最后将测试准确率的变化可视化。这是一个标准的深度学习模型训练流程。 [1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-csdn1) [2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-zhihu1)

至此，已经完成了整个 PyTorch MNIST 项目的代码结构和核心逻辑。接下来运行这个脚本，模型就会开始训练并在完成后展示测试准确率随 epoch 变化的曲线。

# 完整CPU代码：

```
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

"""
卷积运算 使用mnist数据集，和10-4，11类似的，只是这里：1.输出训练轮的acc 2.模型上使用torch.nn.Sequential
"""
# Super parameter ------------------------------------------------------------------------------------
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

# Prepare dataset ------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# softmax归一化指数函数(https://blog.csdn.net/lz_peter/article/details/84574716),其中0.1307是mean均值和0.3081是std标准差

train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform)  # 本地没有就加上download=True
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform)  # train=True训练集，=False测试集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
    plt.title("Labels: {}".format(train_dataset.train_labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()


# 训练集乱序，测试集有序
# Design model using class ------------------------------------------------------------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


model = Net()


# Construct loss and optimizer ------------------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量


# Train and Test CLASS --------------------------------------------------------------------------------------
# 把单独的一轮一环封装在函数类里
def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零

        # torch.save(model.state_dict(), './model_Mnist.pth')
        # torch.save(optimizer.state_dict(), './optimizer_Mnist.pth')


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc


# Start train and Test --------------------------------------------------------------------------------------
if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test()
        acc_list_test.append(acc_test)

    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()

```



# 完整GPU代码：

现在我们将之前提供的 CPU 版本代码进行修改，使其能够在 GPU (RTX A5000) 上运行。

主要修改的地方包括：

1. 检测可用的计算设备 (CPU 或 GPU)。
2. 将的 `Net` 模型实例移动到检测到的设备上。
3. 在训练和测试循环中，将每个批次的输入数据和对应的标签也移动到相同的设备上。
4. (可选但推荐) 在训练和测试函数中设置模型的模式 (`model.train()` 和 `model.eval()`)。

以下是修改后的完整代码（包含之前导入包、数据集读取、模型定义、损失函数和优化器定义的部分，并加入了 GPU 相关的修改）：

```python
# 1. 首先导入需要的包
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

# super parameters (请确保这些参数在使用它们之前已经定义)
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10 # 定义总的训练轮次

# --- 新增代码：检测并指定设备 ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检查 CUDA 是否可用
# 或者如果有多块GPU，可以指定使用哪一块，例如使用第一块GPU：
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(f"Using device: {device}")
# ---------------------------------


# 2、数据集(MNIST)读取
# Step1：归一化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Step2：下载/获取数据集
# download=True 会在指定路径不存在时自动下载
train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)

# Step3：载入数据集，用Dataloader 包起来
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 1.2 展示MNIST数据集 (这部分保持原样，因为它只是展示数据，不参与训练计算)
# fig = plt.figure()
# # 注意：这里直接访问了 train_dataset.train_data 和 train_dataset.train_labels
# # 这在确定数据已经完全加载并且内存足够时是可行的，但通常通过 DataLoader 迭代更普遍
# # 另外，新版本的 torchvision.datasets.MNIST 可能推荐通过索引访问而不是直接访问属性
# # 例如 img, label = train_dataset[i]
# # 为了兼容旧代码，我们保留原样，但如果报错，可以尝试新的访问方式
# for i in range(12):
#     plt.subplot(3, 4, i+1)
#     plt.tight_layout()
#     # 检查 train_data/train_labels 是否存在，或者使用 try-except
#     try:
#         plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
#         plt.title("Labels: {}".format(train_dataset.train_labels[i]))
#     except AttributeError:
#          print("Warning: Could not access .train_data/.train_labels directly. Dataset might be stored differently.")
#          # 可以尝试通过索引访问并转换回可显示格式 (如果需要展示的话)
#          # img, label = train_dataset[i]
#          # plt.imshow(img.squeeze().numpy(), cmap='gray', interpolation='none') # Assuming ToTensor was used
#          # plt.title("Labels: {}".format(label))

#     plt.xticks([])
#     plt.yticks([])
# # plt.show() # 默认不显示，如果想看可以取消注释


# 二、构建模型(CNN)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        # 计算全连接层输入维度：
        # 原始图片: 28x28
        # Conv1 (kernel 5, no padding, stride 1): (28 - 5) + 1 = 24x24
        # MaxPool1 (kernel 2, stride 2): 24 / 2 = 12x12
        # Conv2 (kernel 5, no padding, stride 1): (12 - 5) + 1 = 8x8
        # MaxPool2 (kernel 2, stride 2): 8 / 2 = 4x4
        # 经过 conv2_block 后，张量形状是 (batch_size, 20, 4, 4)
        # 展平后特征数: 20 * 4 * 4 = 320
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10), # 输出 10 个类别 (0-9)
        )

    def forward(self, x):
        # x 的初始形状是 (batch_size, 1, 28, 28)
        batch_size = x.size(0)
        x = self.conv1(x)  # 经过 conv1_block, 形状变为 (batch_size, 10, 12, 12)
        x = self.conv2(x)  # 经过 conv2_block, 形状变为 (batch_size, 20, 4, 4)
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch_size, 320)
        x = self.fc(x) # 经过全连接层, 形状变为 (batch_size, 10)
        return x


# 三、损失函数和优化器
# model 实例化应该放在这里，在定义优化器之前，并且在将模型移动到设备之前
# model = Net() # 实例化模型

criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) # 优化器定义


# 四、定义训练轮和测试轮

def train(epoch):
    # --- 新增代码：设置模型为训练模式 ---
    model.train() # [^digitalocean.com_train]
    # -----------------------------------
    
    running_loss = 0.0  # 累积每个小批次的损失
    running_total = 0   # 累积每个小批次的样本总数
    running_correct = 0 # 累积每个小批次的正确预测数

    # 遍历 train_loader 中的每一个数据批次
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        
        # --- 修改代码：将数据移动到指定设备 ---
        inputs, target = inputs.to(device), target.to(device) # [^pytorch.org_move][^medium.com_to_device][^wandb.ai]
        # ---------------------------------------
        
        optimizer.zero_grad() # 梯度清零

        # forward + backward + update
        outputs = model(inputs) # 模型在前向传播中处理设备上的数据
        loss = criterion(outputs, target) # 损失函数计算设备上的输出和目标
        loss.backward() # 反向传播计算设备上的梯度
        optimizer.step() # 优化器更新设备上的模型参数

        # --- 训练过程中的指标记录 ---
        running_loss += loss.item() # loss.item() 将设备上的损失值转为 Python 标量
        
        # 计算当前批次的预测结果
        # torch.max 在设备上执行，但 .data 和 item() 将结果转回 Python/CPU
        _, predicted = torch.max(outputs.data, dim=1) # [^pytorch.org_max][^geeksforgeeks.org_max]
        
        # 累加当前批次的样本总数
        running_total += inputs.shape[0]
        
        # 计算当前批次正确预测的数量 (比较和求和都在设备上执行，item() 转回 Python)
        running_correct += (predicted == target).sum().item() 

        # 每处理 300 个批次，打印一次当前的平均损失和准确率
        if batch_idx % 300 == 299:  
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            # 打印后，将用于累积的变量清零
            running_loss = 0.0  
            running_total = 0
            running_correct = 0


# test 函数 (需要在调用 test 函数前定义全局变量 EPOCH 或者将其作为参数传入 test 函数)
def test(epoch): # 将 epoch 作为参数传入，使函数更独立
    # --- 新增代码：设置模型为评估模式 ---
    model.eval() # [^digitalocean.com_train]
    # -----------------------------------
    
    correct = 0  # 累积整个测试集的正确预测数
    total = 0    # 累积整个测试集的样本总数

    # 使用 torch.no_grad() 上下文管理器：在测试/推理阶段不计算梯度，节省内存和加速。 [^pytorch.org_no_grad][^digitalocean.com_train]
    with torch.no_grad():  
        for data in test_loader:
            images, labels = data
            
            # --- 修改代码：将数据移动到指定设备 ---
            images, labels = images.to(device), labels.to(device) # [^pytorch.org_move][^medium.com_to_device][^wandb.ai]
            # -------------------------------------
            
            outputs = model(images) # 模型在前向传播中处理设备上的数据
            
            # 获取预测结果
            _, predicted = torch.max(outputs.data, dim=1) # [^pytorch.org_max][^geeksforgeeks.org_max]
            
            # 累加当前批次的样本总数
            total += labels.size(0)  
            
            # 累加当前批次正确预测的数量 (比较和求和都在设备上执行，item() 转回 Python)
            correct += (predicted == labels).sum().item() 
    
    # 计算整个测试集的准确率
    acc = correct / total
    
    # 打印当前 epoch 的测试集准确率
    # epoch 和 EPOCH 通过函数参数和全局变量获取
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc)) 
    
    # 返回计算出的准确率
    return acc


# 五、开始训练

if __name__ == '__main__':
    # --- 修改代码：实例化模型后，立即将其移动到指定设备 ---
    model = Net().to(device) # [^pytorch.org_move_model][^medium.com_to_device][^wandb.ai]
    # -------------------------------------------------------
    
    # 优化器定义 (在模型已经移动到设备后定义，这样优化器跟踪的就是设备上的参数)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) 

    acc_list_test = [] # 用于存储每个epoch结束时在测试集上的准确率
    
    # 主循环，迭代进行 EPOCH 次训练和测试
    for epoch in range(EPOCH): 
        # 调用 train 函数进行当前 epoch 的训练
        train(epoch) 
        
        # 调用 test 函数在测试集上评估模型性能，传入当前 epoch
        acc_test = test(epoch) # 修改：传入 epoch 参数
        
        # 将当前 epoch 在测试集上获得的准确率添加到列表中
        acc_list_test.append(acc_test) 

    # 训练和测试循环结束后，开始绘制结果 (使用 matplotlib 在 CPU 上绘制，因为 matplotlib 不直接支持 GPU 张量，需要先转回 CPU)
    # 注意：这里的 acc_list_test 存储的是 Python float，可以直接绘制
    plt.plot(acc_list_test) 
    plt.xlabel('Epoch')    
    plt.ylabel('Accuracy On TestSet') 
    plt.title('Test Accuracy over Epochs') # 添加图表标题
    plt.show()
```

------

**主要修改点回顾：**

1. 增加了设备检测的代码 (`device = torch.device(...)`)。
2. 在 `if __name__ == '__main__':` 块中实例化模型后，调用了 `model.to(device)` 将模型参数移动到 GPU。[1](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-pytorch.org_move_model)[2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-medium.com_to_device)[3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-wandb.ai)
3. 在 `train` 函数的数据循环内部，将 `inputs` 和 `target` 张量通过 `.to(device)` 移动到 GPU。[4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-pytorch.org_move)[2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-medium.com_to_device)[3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-wandb.ai)
4. 在 `test` 函数的数据循环内部，将 `images` 和 `labels` 张量通过 `.to(device)` 移动到 GPU。[4](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-pytorch.org_move)[2](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-medium.com_to_device)[3](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-wandb.ai)
5. 在 `train` 函数开始时添加了 `model.train()`，在 `test` 函数开始时添加了 `model.eval()`。[5](https://test.zhangsan.cool/c/b86050f7-0b05-40fd-816f-1aa88927c45e#user-content-fn-digitalocean.com_train)
6. 修改了 `test` 函数的定义 `def test(epoch):` 并更新了在主函数中的调用 `test(epoch)`，使其能够正确打印当前的 epoch 数。

确保的 PyTorch 环境是支持 CUDA 的版本，并且的驱动和 CUDA Toolkit 与 RTX A5000 兼容。运行这段修改后的代码，训练和测试过程就会利用的 GPU 进行加速。



# 参考地址：

CSDN:[用PyTorch实现MNIST手写数字识别（最新，非常详细）_mnist pytorch-CSDN博客](https://blog.csdn.net/qq_45588019/article/details/120935828#:~:text=本文基于 PyTorch 框架，采用 CNN卷积神经网络 实现 MNIST 手写数字识别，仅在 CPU,已分别实现使用Linear纯线性层、 CNN 卷积 神经网络 、Inception网络、和Residual残差网络四种结构对 MNIST数据集 进行手写数字识别，并对其识别准确率进行比较分析。 （另外三种还未发布）)
B站:https://www.bilibili.com/video/BV1Y7411d7Ys?p=10
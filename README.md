# 神经网络训练演示（Neural Network Training Demo）

## 1.项目介绍（Introduction to the project）

本项目是mountain_lake项目的演示程序，演示了如何使用mountain_lake项目提供的神经网络模型进行训练。

This project is a demo program for the mountain_lake project that demonstrates how to train with the neural network model provided by the mountain_lake project.

## 2.项目结构（Struct of project）

### 2.1 data 文件夹（data folder）

此目录存放训练与测试数据。

This directory holds the training and test data.

### 2.2 lib 文件夹（lib folder）

此目录存放来自我编写的其他项目的库文件。

This directory holds library files from other projects I've written.

### 2.3 src 文件夹（src folder）

此目录存放本项目自己的源代码文件。

This directory holds the project's own source code files.

#### 2.3.1 mnist 子文件夹（mnist subfolder）

此目录存放载入与初始化MNIST数据的代码。

This directory holds the code for loading and initializing MNIST data.

### 2.4 tests 文件夹（tests folder）

此目录存放测试代码。

This directory holds the test code.

## 3. 验证环境（Environment）

- Ubuntu 22.04
- g++ 11.4.0
- cmake 3.22.1
- Intel i5-6500 CPU @ 3.20GHz × 4

## 4. 依赖库（Dependencies）

### 4.1 线性代数库 Eigen（Linear Algebra Library: Eigen）

Eigen 3.4.0

Ubuntu下的安装命令（Installation commands under Ubuntu）：

```bash
sudo apt install libeigen3-dev
```

### 4.2 测试工具 GoogleTest（Testing Tool: GoogleTest）

GoogleTest 1.11.0

Ubuntu下的安装命令（Installation commands under Ubuntu）：

```bash
sudo apt install libgtest-dev
```

## 5. 使用示例（Examples of how to use）

我在开发时，使用的编辑器是Visual Studio Code，所以也推荐大家使用。当然，你也可以使用其他编辑器或IDE。因为我没有使用Windows系统做开发，所以并不清楚相关的内容，请见谅。

The editor I use for development is Visual Studio Code, so I recommend it as well. Of course, you can also use other editors or IDEs. since I don't use Windows to do development, I don't really know what's involved, so please forgive me.

### 5.1 场景一（Scene 1）

首先介绍最简单的使用方法。当神经网络中只存在全连接层时，可以按照以下步骤进行操作。

The simplest way to use it is described first. When only fully connected layers are present in a neural network, the following steps can be followed.

#### 5.1.1 配置表（configuration table）：config.toml

```toml
[MNIST]
train_img = "data/train-images.idx3-ubyte"
train_label = "data/train-labels.idx1-ubyte"
test_img = "data/t10k-images.idx3-ubyte"
test_label = "data/t10k-labels.idx1-ubyte"
train_size = 60000
test_size = 10000

[hyper_parameters]
iterations = 10000
batch_size = 100
learning_rate = 0.04

```

#### 5.1.1 神经网络配置表（Neural Network Configuration Table）：neural_network.toml

```toml
[neural_network]
struct = ["Affine:50", "Sigmoid", "Affine:10", "SoftmaxWithLoss"]

```

以上两个配置表的内容从相关名称中即可看出其作用，就不再赘述。有关神经网络配置表的详细说明请看mountain_lake项目中的介绍。

The contents of the above two configuration tables are obvious from their related names and will not be repeated. For a detailed description of the neural network configuration tables see the introduction in the mountain_lake project.

### 5.2 场景二（Scene 2）

当需要使用卷积神经网络时，神经网络配置文件的内容可以参考下面的配置。

When a convolutional neural network is required, the contents of the neural network configuration file can be found in the configuration below.

#### 5.2.1 神经网络配置表（Neural Network Configuration Table）：neural_network.toml

```toml
[neural_network]
struct = [
  "Convolution-1",
  "ReLU",
  "Pooling-1",
  "Affine:100",
  "ReLU",
  "Affine:10",
  "SoftmaxWithLoss",
]

# 卷积层
[Convolution-1]
pad = 0
stride = 1
channel_num = 1
filter_num = 30
filter_height = 5
filter_width = 5

# 池化层
[Pooling-1]
pool_height = 2
pool_width = 2
stride = 2
filter_num = 30
type = "Max"

```

#### 5.2.2 说明

在配置卷积层与池化层时，要注意选择合适的数值，当前程序中没有对配置合理性的检测，所以需要使用者自行保证配置的合理性，以免造成程序无法正常运行。下面的图表，是对MNIST数据集进行训练的结果，使用的是卷积神经网络。

When configuring the convolutional and pooling layers, care should be taken to choose the appropriate values. The current program does not have a test for the reasonableness of the configurations, so the user needs to make sure that the configurations are reasonable on their own so that the program does not run properly. The following chart shows the results of training on the MNIST dataset, using a convolutional neural network.

![](csv/acc2024-2-27_17:22:7.png)

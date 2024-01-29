// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <chrono>
#include <iostream>

#include "mnist/mnist.h"
#include "neural_network/neural_network.h"

using std::cout;
using std::endl;
using std::string;

// 运行程序时，后面跟上MNIST数据所在路径，不需要输入文件名。
int main(int argc, char* argv[]) {
  Mnist mnist;
  string err = mnist.LoadMnist(argc, argv[1]);
  if (!err.empty()) {
    cout << err << endl;
    return -1;
  }

  // 超参数
  const float kLearningRate = 0.01f;
  const int kIterations = 10000;
  const int kBatchSize = 100;
  const int kIterPerEpoch = 60000 / kBatchSize;

  // 神经网络相关数据
  const int kInputSize = 784;
  const int kHiddenSize = 50;
  const int kOutputSize = 10;
  NeuralNetwork nn;
  nn.Init(&mnist, kInputSize, kHiddenSize, kOutputSize, kLearningRate);

  cout << "开始训练" << endl;
  auto start = std::chrono::high_resolution_clock::now();
  int samples[kBatchSize];  // 存放随机抽样的图片索引
  string csv = ", train acc, test acc\n";
  for (int i = 0; i < kIterations; ++i) {
    // 随机抽取训练数据
    BetterRand(samples, kBatchSize, 0, mnist.train_size_ - 1);
    for (int j = 0; j < kBatchSize; ++j) {
      nn.Gradient(samples[j]);  // 计算梯度
      nn.Update();              // 更新参数
    }

    // 每个epoch计算一次准确率
    if ((i + 1) % kIterPerEpoch == 0) {
      int k = (i + 1) / kIterPerEpoch;
      csv = csv + std::to_string(k) + ", ";
      cout << "Epoch " << k;
      nn.Accuracy(csv);
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  cout << "训练时间：" << duration.count() / 1000000.0 << " 秒" << endl << endl;

  // 写入CSV文件
  if (!csv.empty()) {
    string path = "acc.csv";
    std::ofstream file(path);
    file << csv;
    file.close();
  }

  return 0;
}

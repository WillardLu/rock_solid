// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <mountain_lake/neural_network/neural_network.h>
#include <mountain_town/string/toml.h>

#include <chrono>

#include "mnist/mnist.h"

using std::to_string;
using std::chrono::duration_cast;
using std::chrono::microseconds;

int main() {
  // 载入配置表（Load Configuration Table）
  unordered_map<string, string> conf;
  string err = ReadSTOML("config.toml", conf);
  if (err.empty() == false) {
    cout << err << endl;
    return -1;
  }
  // 载入MNIST数据集（Load the MNIST dataset）
  Mnist mnist(conf);
  err = mnist.LoadMnist();
  if (!err.empty()) {
    cout << err << endl;
    return -1;
  }
  RawData raw_data;
  raw_data.train_data = mnist.train_img_;
  raw_data.train_labels = mnist.train_label_;
  raw_data.test_data = mnist.test_img_;
  raw_data.test_labels = mnist.test_label_;
  raw_data.row = mnist.img_height_;
  raw_data.col = mnist.img_width_;
  raw_data.size = mnist.img_size_;
  raw_data.train_number = mnist.train_size_;
  raw_data.test_number = mnist.test_size_;
  // 创建并初始化神经网络（Create and initialize a neural network）
  NeuralNetwork nn;
  nn.Init("neural_network.toml", raw_data);
  nn.SetLearningRate(stof(conf["hyper_parameters.learning_rate"]));
  const int kTrainSize = stoi(conf["MNIST.train_size"]);
  const int kIterations = stoi(conf["hyper_parameters.iterations"]);
  const int kBatchSize = stoi(conf["hyper_parameters.batch_size"]);
  int iter_per_epoch = kTrainSize / kBatchSize;
  auto start = std::chrono::high_resolution_clock::now();
  int samples[kBatchSize];
  auto e_start = start;
  auto e_end = start;
  string conf_str =
      "Number of iterations: " + conf["hyper_parameters.iterations"] +
      ", quantity per batch:" + conf["hyper_parameters.batch_size"] +
      ", learning rate: " + conf["hyper_parameters.learning_rate"];
  string csv = ", train acc, test acc\n";

  cout << conf_str << endl;
  cout << "Start training" << endl;
  for (int i = 0; i < kIterations; ++i) {
    // 随机抽取训练数据（Random sampling of training data）
    BetterRand(samples, kBatchSize, 0, kTrainSize - 1);
    for (int j = 0; j < kBatchSize; ++j) {
      nn.Gradient(samples[j]);
      nn.Update();
    }

    // 每个epoch计算一次准确率
    // Accuracy is calculated once per epoch.
    if ((i + 1) % iter_per_epoch == 0) {
      e_end = std::chrono::high_resolution_clock::now();
      int k = (i + 1) / iter_per_epoch;
      csv = csv + std::to_string(k) + ", ";
      cout << "Epoch " << k << " finished, Time-consuming: "
           << duration_cast<microseconds>(e_end - e_start).count() / 1000000.0
           << " second" << endl;
      e_start = std::chrono::high_resolution_clock::now();
      nn.Accuracy(csv);
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "Time to train: " << duration.count() / 1000000.0 << " second" << endl
       << endl;

  // 写入CSV文件（Write to CSV file）
  if (!csv.empty()) {
    time_t now = time(0);
    tm *ltm = localtime(&now);
    string time1 = to_string(1900 + ltm->tm_year) + "-" +
                   to_string(1 + ltm->tm_mon) + "-" + to_string(ltm->tm_mday) +
                   "_" + to_string(ltm->tm_hour) + ":" +
                   to_string(ltm->tm_min) + ":" + to_string(ltm->tm_sec);
    string path = "csv/acc" + time1 + ".csv";
    cout << "Generate csv file: " << path << endl;
    std::ofstream file(path);
    file << csv;
    file.close();
  }

  return 0;
}

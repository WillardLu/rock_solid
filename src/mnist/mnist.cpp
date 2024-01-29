// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "mnist.h"

/// @brief 分配空间给MNIST数据集结构
/// @param mnist MNIST数据集结构
Mnist::Mnist() {}

/// @brief 释放MNIST数据集结构
/// @param mnist MNIST数据集结构
Mnist::~Mnist() {}

/// @brief 载入MNIST数据集
/// @param argc 来自main()函数的参数argc
/// @param dir 数据文件所在目录
/// @param mnist MNIST数据集结构
/// @return 错误信息
string Mnist::LoadMnist(int argc, char *dir) {
  string path = argc != 2 ? "data/" : dir;
  string file[4] = {
      path + "train-images.idx3-ubyte", path + "train-labels.idx1-ubyte",
      path + "t10k-images.idx3-ubyte", path + "t10k-labels.idx1-ubyte"};
  int data_size[4] = {60000 * 784, 60000, 10000 * 784, 10000};
  int offset[4] = {16, 8, 16, 8};
  string f;
  // 逐一读取与格式化
  for (int i = 0; i < 4; i++) {
    f = file[i];
    ifstream fp(f, ios::in | ios::binary | ios::ate);
    if (fp.is_open() == false) {
      return f + " 文件打开失败。";
    }
    // 检查文件大小和预期的大小是否一致
    if (data_size[i] != (int)fp.tellg() - offset[i]) {
      fp.close();
      return f + " 文件大小不正确。";
    }
    // 为临时中转区分配空间
    uint8_t *buffer = new uint8_t[data_size[i]];
    if (buffer == nullptr) {
      fp.close();
      return "内存分配失败。";
    }
    // 把文件位置进行指定的偏移
    fp.seekg(offset[i]);
    // 读入数据
    if (!fp.read((char *)buffer, data_size[i])) {
      delete[] buffer;
      fp.close();
      return "读取数据失败。";
    }
    for (int j = 0; j < data_size[i]; ++j) {
      switch (i) {
        case 0:
          this->train_img_(j / 784, j % 784) =
              static_cast<float>(buffer[j]) / 255.0f;
          break;
        case 1:
          this->train_label_(j, 0) = buffer[j];
          this->train_one_hot_(j, this->train_label_(j, 0)) = 1;
          break;
        case 2:
          this->test_img_(j / 784, j % 784) =
              static_cast<float>(buffer[j]) / 255.0f;
          break;
        case 3:
          this->test_label_(j, 0) = buffer[j];
          this->test_one_hot_(j, this->test_label_(j, 0)) = 1;
          break;
        default:
          break;
      }
    }
    delete[] buffer;
    fp.close();
  }
  return "";
}

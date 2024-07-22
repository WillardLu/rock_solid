// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "mnist.h"

/// @brief 初始化（initialization）
/// @param config 配置内容（Configuration contents）
Mnist::Mnist(unordered_map<std::string, std::string> &config) {
  this->train_size_ = stoi(config["MNIST.train_size"]);
  this->test_size_ = stoi(config["MNIST.test_size"]);
  this->train_img_ = MatrixXf::Zero(this->train_size_, this->img_size_);
  this->train_label_ = MatrixXb::Zero(this->train_size_, 1);
  this->train_one_hot_ = MatrixXb::Zero(this->train_size_, 10);
  this->test_img_ = MatrixXf::Zero(this->test_size_, this->img_size_);
  this->test_label_ = MatrixXb::Zero(this->test_size_, 1);
  this->test_one_hot_ = MatrixXb::Zero(this->test_size_, 10);
  this->file_[0] = config["MNIST.train_img"];
  this->file_[1] = config["MNIST.train_label"];
  this->file_[2] = config["MNIST.test_img"];
  this->file_[3] = config["MNIST.test_label"];
}

Mnist::~Mnist() {}

/// @brief 载入MNIST数据集（Load the MNIST dataset）
/// @return 错误信息（error information）
string Mnist::LoadMnist() {
  int data_size[4] = {this->train_size_ * this->img_size_, this->train_size_,
                      this->test_size_ * this->img_size_, this->test_size_};
  int offset[4] = {16, 8, 16, 8};
  string f;
  // 逐一读取与格式化（Step-by-step reading and formatting）
  for (int i = 0; i < 4; i++) {
    f = this->file_[i];
    ifstream fp(f, ios::in | ios::binary | ios::ate);
    if (fp.is_open() == false) {
      return f + " file failed to open.";
    }
    // 检查文件大小和预期的大小是否一致
    // Check that the file size is the same as the expected size.
    if (data_size[i] != (int)fp.tellg() - offset[i]) {
      fp.close();
      return f + " file size is incorrect.";
    }
    // 为临时中转区分配空间。
    // Allocate space for a temporary staging area.
    uint8_t *buffer = new uint8_t[data_size[i]];
    if (buffer == nullptr) {
      fp.close();
      return "Memory allocation failed.";
    }
    // 把文件位置进行指定的偏移。
    // Offset the file location by the specified amount.
    fp.seekg(offset[i]);
    // 读入数据（Read data）
    if (!fp.read((char *)buffer, data_size[i])) {
      delete[] buffer;
      fp.close();
      return "Failed to read data.";
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

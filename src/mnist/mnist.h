// @copyright Copyright 2023 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef MNIST_MNIST_H_
#define MNIST_MNIST_H_

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <unordered_map>

using std::ifstream;
using std::ios;
using std::string;
using std::unordered_map;

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::RowMajor;

typedef Matrix<uint8_t, Dynamic, Dynamic> MatrixXb;

class Mnist {
 public:
  Mnist(unordered_map<std::string, std::string> &config);
  ~Mnist();
  string LoadMnist();

  int train_size_;
  int test_size_;
  int img_size_ = 784;
  int img_height_ = 28;
  int img_width_ = 28;
  string file_[4];
  MatrixXf train_img_;
  MatrixXb train_label_;
  MatrixXb train_one_hot_;
  MatrixXf test_img_;
  MatrixXb test_label_;
  MatrixXb test_one_hot_;
};

#endif  // MNIST_MNIST_H_
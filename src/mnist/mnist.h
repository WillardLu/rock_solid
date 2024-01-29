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

using std::ifstream;
using std::ios;
using std::string;

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::RowMajor;

typedef Matrix<uint8_t, Dynamic, Dynamic> MatrixXb;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXf_rm;

class Mnist {
 public:
  Mnist();
  ~Mnist();
  string LoadMnist(int argc, char *dir);

  int train_size_ = 60000;
  int test_size_ = 10000;
  int img_size_ = 784;
  MatrixXf_rm train_img_ = MatrixXf_rm::Zero(60000, 784);
  MatrixXb train_label_ = MatrixXb::Zero(60000, 1);
  MatrixXb train_one_hot_ = MatrixXb::Zero(60000, 10);
  MatrixXf_rm test_img_ = MatrixXf_rm::Zero(10000, 784);
  MatrixXb test_label_ = MatrixXb::Zero(10000, 1);
  MatrixXb test_one_hot_ = MatrixXb::Zero(10000, 10);
};

#endif  // MNIST_MNIST_H_
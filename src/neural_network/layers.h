// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef NEURAL_NETWORK_LAYERS_H_
#define NEURAL_NETWORK_LAYERS_H_

#include <lib/general.h>

#include <eigen3/Eigen/Dense>

using Eigen::MatrixXf;

typedef Eigen::Map<Eigen::Matrix<float, 1, 784>> MapImg;

/// @brief 第一个仿射变换层类
/// @remark 因为第一个仿射变换层在处理上和后续的仿射变换层上有区别，所以单独定义。
///         第一个仿射变换层输入的信号是原始信号，在数据传递上与后续的不一样；
///         第一个仿射变换层反向传播时并不需要处理X的层数，因为没有意义。
class FirstAffineLayer {
 public:
  FirstAffineLayer(){};
  ~FirstAffineLayer(){};
  void Init(int input_size, int output_size);
  void Forward(MapImg &X, MatrixXf &W, MatrixXf &B, MatrixXf &A);
  void Backward(MapImg &X, MatrixXf &dA);

  MatrixXf dW_;
  MatrixXf dB_;
  MatrixXf dX_;
};

/// @brief 仿射变换层类
class AffineLayer {
 public:
  AffineLayer(){};
  ~AffineLayer(){};
  void Init(int input_size, int output_size);
  void Forward(MatrixXf X, MatrixXf &W, MatrixXf &B, MatrixXf &A);
  void Backward(MatrixXf X, MatrixXf &W, MatrixXf &dA);

  MatrixXf dW_;
  MatrixXf dB_;
  MatrixXf dX_;
};

// Softmax与Loss合并层类
class SoftmaxWithLossLayer {
 public:
  explicit SoftmaxWithLossLayer(){};
  ~SoftmaxWithLossLayer(){};
  void Init(int output_size);
  float Forward(int label, MatrixXf &A, MatrixXf &Y);
  void Backward(MatrixXf &Y, uint8_t label);

  MatrixXf dA_;
};

class SigmoidLayer {
 public:
  explicit SigmoidLayer(){};
  ~SigmoidLayer(){};
  void Init(int hidden_size);
  void Forward(MatrixXf &A, MatrixXf &Z);
  void Backward(MatrixXf &dZ, MatrixXf &Y);

  MatrixXf dA_;
};

class ReLULayer {
 public:
  ReLULayer(){};
  ~ReLULayer(){};
  void Init(int hidden_size);
  void Forward(MatrixXf &A, MatrixXf &Z);
  void Backward(MatrixXf &dZ, MatrixXf &Y);

  MatrixXf dA_;
};

#endif  // NEURAL_NETWORK_LAYERS_H_
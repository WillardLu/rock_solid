// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef NEURAL_NETWORK_NEURAL_NETWORK_H_
#define NEURAL_NETWORK_NEURAL_NETWORK_H_

#include <mnist/mnist.h>

#include <eigen3/Eigen/Dense>
#include <iostream>

#include "layers.h"

using Eigen::MatrixXf;

typedef Eigen::Map<Eigen::Matrix<float, 1, 784, RowMajor>> MapImg;

// 两层神经网络类
class NeuralNetwork {
 public:
  NeuralNetwork();
  ~NeuralNetwork();
  void Init(Mnist *mnist, int input_size, int hidden_size, int output_size,
            float learning_rate);
  void Gradient(int index);
  void SigmoidPredict(MapImg X_p);
  void SigmoidForward(MapImg X_p);
  void SigmoidBackward(MapImg X_p);
  void ReLUPredict(MapImg X_p);
  void ReLUForward(MapImg X_p);
  void ReLUBackward(MapImg X_p);
  void Update();
  void Accuracy(string &csv);

 private:
  Mnist *mnist_;  // MNIST数据集
  uint8_t label_;
  float learning_rate_;
  MatrixXf W1_;
  MatrixXf B1_;
  MatrixXf A1_;
  MatrixXf Z_;
  MatrixXf W2_;
  MatrixXf B2_;
  MatrixXf A2_;
  MatrixXf Y_;
  float loss_;
  // 层
  FirstAffineLayer affine1_;
  SigmoidLayer sigmoid_;
  ReLULayer relu_;
  AffineLayer affine2_;
  SoftmaxWithLossLayer softmax_loss_;
};

#endif  // NEURAL_NETWORK_NEURAL_NETWORK_H_
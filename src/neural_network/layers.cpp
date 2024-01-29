// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "layers.h"

/// @brief 初始化仿射变换层
/// @param input_size 输入尺寸
/// @param output_size 输出尺寸
void FirstAffineLayer::Init(int input_size, int output_size) {
  dW_ = MatrixXf::Zero(input_size, output_size);
  dB_ = MatrixXf::Zero(1, output_size);
  this->dX_ = MatrixXf::Zero(1, output_size);
}

/// @brief 仿射变换层正向传播
/// @param X 输入信号
/// @param W 权重
/// @param B 偏置
/// @param A 输出信号
void FirstAffineLayer::Forward(MapImg &X, MatrixXf &W, MatrixXf &B,
                               MatrixXf &A) {
  A = X * W + B;
}

/// @brief 仿射变换层反向传播
/// @param X 输入信号
/// @param W 权重
/// @param dA 输出信号的导数
void FirstAffineLayer::Backward(MapImg &X, MatrixXf &dA) {
  this->dB_ = dA;
  this->dW_.noalias() = X.transpose() * dA;
}

/// @brief 初始化仿射变换层
/// @param input_size 输入尺寸
/// @param output_size 输出尺寸
void AffineLayer::Init(int input_size, int output_size) {
  dW_ = MatrixXf::Zero(input_size, output_size);
  dB_ = MatrixXf::Zero(1, output_size);
  this->dX_ = MatrixXf::Zero(1, output_size);
}

/// @brief 仿射变换层正向传播
/// @param X 输入信号
/// @param W 权重
/// @param B 偏置
/// @param A 输出信号
void AffineLayer::Forward(MatrixXf X, MatrixXf &W, MatrixXf &B, MatrixXf &A) {
  A = X * W + B;
}

/// @brief 仿射变换层反向传播
/// @param X 输入信号
/// @param W 权重
/// @param dA 输出信号的导数
void AffineLayer::Backward(MatrixXf X, MatrixXf &W, MatrixXf &dA) {
  this->dB_.noalias() = dA;
  this->dW_.noalias() = X.transpose() * dA;
  this->dX_.noalias() = dA * W.transpose();
}

/// @brief SigmoideLayer初始化
/// @param hidden_size 隐藏层尺寸
void SigmoidLayer::Init(int hidden_size) {
  this->dA_ = MatrixXf::Zero(1, hidden_size);
}

/// @brief sigmoid层正向传播
/// @param A 输入信号
/// @param Z 输出信号
void SigmoidLayer::Forward(MatrixXf &A, MatrixXf &Z) {
  Z = 1 / (1 + (-A).array().exp());
}

/// @brief sigamoid层反向传播
/// @param dZ 输出信号的导数
/// @param Z 输出信号
void SigmoidLayer::Backward(MatrixXf &dZ, MatrixXf &Z) {
  this->dA_ = dZ.array() * Z.array() * (1 - Z.array());
}

/// @brief SoftmaxWithLossLayer初始化
/// @param output_size 输出尺寸
void SoftmaxWithLossLayer::Init(int output_size) {
  this->dA_ = MatrixXf::Zero(1, output_size);
}

/// @brief softmax与交叉熵误差合并层的正向传播
/// @param labels 监督标签
/// @param A 输入信号
/// @param A2 输出信号
/// @return 误差
float SoftmaxWithLossLayer::Forward(int label, MatrixXf &A, MatrixXf &Y) {
  Softmax(A, Y);
  return CrossEntropy(Y, label);
}

/// @brief softmax与交叉熵误差合并层的反向传播
/// @param Y 经过softmax函数处理过的信号
/// @param labels 监督标签
void SoftmaxWithLossLayer::Backward(MatrixXf &Y, uint8_t label) {
  Y(0, label) -= 1;
  this->dA_ = Y;
}

/// @brief SigmoideLayer初始化
/// @param hidden_size 隐藏层尺寸
void ReLULayer::Init(int hidden_size) {
  this->dA_ = MatrixXf::Zero(1, hidden_size);
}

/// @brief 线性整流层正向传播
/// @param A 输入信号
/// @param Z 输出信号
void ReLULayer::Forward(MatrixXf &A, MatrixXf &Z) {
  // 如果A的值大于0，Z就等于A，否则Z等于0
  Z = (A.array() > 0).select(A, 0);
}

/// @brief 线性整流层反向传播
/// @param dZ 输出信号的导数
/// @param Z 输出信号
void ReLULayer::Backward(MatrixXf &dZ, MatrixXf &Z) {
  this->dA_ = (Z.array() > 0).select(dZ, 0);
}
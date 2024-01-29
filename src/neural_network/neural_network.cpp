// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "neural_network.h"

using std::cout;
using std::endl;

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::~NeuralNetwork() {}

/// @brief 初始化神经网络
/// @param mnist MNIST数据集
void NeuralNetwork::Init(Mnist *mnist, int input_size, int hidden_size,
                         int output_size, float learning_rate) {
  this->mnist_ = mnist;
  this->learning_rate_ = learning_rate;
  this->W1_ = MatrixXf::Zero(input_size, hidden_size);
  this->B1_ = MatrixXf::Zero(1, hidden_size);
  this->A1_ = MatrixXf::Zero(1, hidden_size);
  this->Z_ = MatrixXf::Zero(1, hidden_size);
  this->W2_ = MatrixXf::Zero(hidden_size, output_size);
  this->B2_ = MatrixXf::Zero(1, output_size);
  this->A2_ = MatrixXf::Zero(1, output_size);
  this->Y_ = MatrixXf::Zero(1, output_size);
  this->affine1_.Init(input_size, hidden_size);
  this->affine2_.Init(hidden_size, output_size);
  this->sigmoid_.Init(hidden_size);
  this->relu_.Init(hidden_size);
  this->softmax_loss_.Init(output_size);
  // 初始化权重
  float tmp_w1[hidden_size * input_size] = {0};
  NormalDistr(tmp_w1, hidden_size * input_size);
  memcpy(this->W1_.data(), tmp_w1, hidden_size * input_size);
  this->W1_.noalias() = this->W1_;

  float tmp_w2[output_size * hidden_size] = {0};
  NormalDistr(tmp_w2, output_size * hidden_size);
  memcpy(this->W2_.data(), tmp_w2, output_size * hidden_size);
  this->W2_.noalias() = this->W2_;
}

/// @brief 计算梯度
/// @param index 训练数据索引
void NeuralNetwork::Gradient(int index) {
  MapImg X_p(this->mnist_->train_img_.row(index).data());
  this->label_ = this->mnist_->train_label_(index);
  // 正向传播
  this->SigmoidForward(X_p);
  // this->ReLUForward(X_p);
  //   反向传播计算梯度
  this->SigmoidBackward(X_p);
  // this->ReLUBackward(X_p);
}

/// @brief 正向传播
/// @param X_p 训练图像数据的映射
void NeuralNetwork::SigmoidForward(MapImg X_p) {
  this->SigmoidPredict(X_p);
  this->loss_ = this->softmax_loss_.Forward(this->label_, this->A2_, this->Y_);
};

/// @brief 正向传播
/// @param X_p 训练图像数据的映射
void NeuralNetwork::ReLUForward(MapImg X_p) {
  this->ReLUPredict(X_p);
  this->loss_ = this->softmax_loss_.Forward(this->label_, this->A2_, this->Y_);
};

/// @brief 预测
void NeuralNetwork::SigmoidPredict(MapImg X_p) {
  this->affine1_.Forward(X_p, this->W1_, this->B1_, this->A1_);
  this->sigmoid_.Forward(this->A1_, this->Z_);
  this->affine2_.Forward(this->Z_, this->W2_, this->B2_, this->A2_);
};

/// @brief 预测
void NeuralNetwork::ReLUPredict(MapImg X_p) {
  this->affine1_.Forward(X_p, this->W1_, this->B1_, this->A1_);
  this->relu_.Forward(this->A1_, this->Z_);
  this->affine2_.Forward(this->Z_, this->W2_, this->B2_, this->A2_);
};

/// @brief 反向传播
/// @param X_p 训练图像数据的映射
void NeuralNetwork::SigmoidBackward(MapImg X_p) {
  this->softmax_loss_.Backward(this->Y_, this->label_);
  this->affine2_.Backward(this->Z_, this->W2_, this->softmax_loss_.dA_);
  this->sigmoid_.Backward(this->affine2_.dX_, this->Z_);
  this->affine1_.Backward(X_p, this->sigmoid_.dA_);
}

/// @brief 反向传播
/// @param X_p 训练图像数据的映射
void NeuralNetwork::ReLUBackward(MapImg X_p) {
  this->softmax_loss_.Backward(this->Y_, this->label_);
  this->affine2_.Backward(this->Z_, this->W2_, this->softmax_loss_.dA_);
  this->relu_.Backward(this->affine2_.dX_, this->Z_);
  this->affine1_.Backward(X_p, this->relu_.dA_);
}

/// @brief 更新参数
void NeuralNetwork::Update() {
  this->W1_.noalias() -= this->learning_rate_ * this->affine1_.dW_;
  this->B1_.noalias() -= this->learning_rate_ * this->affine1_.dB_;
  this->W2_.noalias() -= this->learning_rate_ * this->affine2_.dW_;
  this->B2_.noalias() -= this->learning_rate_ * this->affine2_.dB_;
}

/// @brief 计算准确率
void NeuralNetwork::Accuracy(string &csv) {
  // 计算训练数据的准确率
  int correct = 0;
  int index = 0;
  int temp = 0;
  for (int i = 0; i < this->mnist_->train_size_; ++i) {
    MapImg X_p1(this->mnist_->train_img_.row(i).data());
    this->SigmoidPredict(X_p1);
    // this->ReLUPredict(X_p1);
    this->A2_.maxCoeff(&temp, &index);
    if (this->mnist_->train_label_(i) == index) {
      ++correct;
    }
  }
  float acc1 = (float)correct / this->mnist_->train_size_;
  cout.precision(4);
  cout << "  训练数据精度：" << acc1 * 100 << "%，";

  correct = 0.0;
  for (int i = 0; i < this->mnist_->test_size_; ++i) {
    MapImg X_p2(this->mnist_->test_img_.row(i).data());
    this->SigmoidPredict(X_p2);
    // this->ReLUPredict(X_p2);
    this->A2_.maxCoeff(&temp, &index);
    if (this->mnist_->test_label_(i) == index) {
      ++correct;
    }
  }
  float acc2 = (float)correct / this->mnist_->test_size_;
  cout << "测试数据精度：" << acc2 * 100 << "%" << endl;
  csv += std::to_string(acc1) + "," + std::to_string(acc2) + "\n";
}
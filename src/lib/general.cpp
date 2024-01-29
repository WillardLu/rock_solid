// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#include "general.h"

/// @brief 更好的随机数生成器
/// @param rn 指向存放随机数数组的指针
/// @param size 数组尺寸
/// @param min 最小值
/// @param max 最大值
/// @note 最小值与最大值都包含在生成的随机数范围内。
void BetterRand(int *rn, int size, int min, int max) {
  // 均匀分布的整数随机数生成器，可生成非确定性随机数（如果可用）。
  std::random_device crypto_random_generator;
  // 产生在封闭区间 [a, b]
  // 上均匀分布的随机整数值，即按照离散概率函数分布的随机整数值。
  std::uniform_int_distribution<int> int_distribution(min, max);

  for (int i = 0; i < size; i++) {
    rn[i] = int_distribution(crypto_random_generator);
  }
}

/// @brief 生成正态分布随机数
/// @param rn 指向存放随机数数组的指针
/// @param size 数组尺寸
void NormalDistr(float *rn, int size) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> dis{0.0, 1.0};
  for (int i = 0; i < size; i++) {
    rn[i] = dis(gen);
  }
}

/// @brief 计算softmax函数
/// @param A 输入矩阵
/// @param Y 输出矩阵
void Softmax(Eigen::MatrixXf &A, Eigen::MatrixXf &Y) {
  Eigen::MatrixXf X_exp = (A.array() - A.maxCoeff()).array().exp();
  Y = X_exp / X_exp.sum();
}

/// @brief 交叉熵函数
/// @param Y 输入矩阵
/// @param t 标签
/// @return 交叉熵误差
float CrossEntropy(Eigen::MatrixXf &Y, int t) {
  return -std::log(Y(0, t) + 1e-7);
}
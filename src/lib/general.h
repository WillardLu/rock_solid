// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.
#ifndef LIB_GENERAL_H_
#define LIB_GENERAL_H_

#include <eigen3/Eigen/Dense>
#include <random>

void BetterRand(int *rn, int size, int min, int max);
void NormalDistr(float *rn, int size);
void Softmax(Eigen::MatrixXf &A, Eigen::MatrixXf &Y);
float CrossEntropy(Eigen::MatrixXf &A, int t);

#endif  // LIB_GENERAL_H_
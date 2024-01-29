// @copyright Copyright 2024 Willard Lu
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <cmath>
#include <mnist/mnist.h>
#include <gtest/gtest.h>


TEST(MnistTests, LoadMnist) {
  // 准备参数
  Mnist mnist;
  char *dir = {};
  std::string err = mnist.LoadMnist(0, dir);
  // 1. 正常文件载入测试
  ASSERT_EQ(err, "");
  // 2. 测试载入的训练用图像数据
  // 2.1 普通测试
  ASSERT_EQ(round(mnist.train_img_(0, 572) * 10000) / 10000, 0.8667);
  // 2.2 边界测试（第一张图片第一个非0值，最后一张图片最后一个非0值）
  ASSERT_EQ(round(mnist.train_img_(0, 152) * 10000) / 10000, 0.0118);
  ASSERT_EQ(round(mnist.train_img_(59999, 682) * 10000) / 10000, 0.5255);
  // 3. 测试载入的训练用标签数据
  // 3.1 普通测试
  ASSERT_EQ(mnist.train_label_(404, 0), 8);
  ASSERT_EQ(mnist.train_one_hot_(404, 8), 1);
  // 3.2 边界测试（第一张图片的标签，最后一张图片的标签）
  ASSERT_EQ(mnist.train_label_(0, 0), 5);
  ASSERT_EQ(mnist.train_one_hot_(0, 5), 1);
  ASSERT_EQ(mnist.train_label_(59999, 0), 8);
  ASSERT_EQ(mnist.train_one_hot_(59999, 8), 1);
  // 4. 测试载入的测试图像数据
  // 4.1 普奶测试
  ASSERT_EQ(round(mnist.test_img_(0, 684) * 10000) / 10000, 0.9961);
  // 4.2 边界测试（第一张图片第一个非0值，最后一张图片最后一个非0值）
  ASSERT_EQ(round(mnist.test_img_(0, 202) * 10000) / 10000, 0.3294);
  ASSERT_EQ(round(mnist.test_img_(9999, 607) * 10000) / 10000, 0.0157);
  // 5. 测试载入的测试标签数据
  // 5.1 普通测试
  ASSERT_EQ(mnist.test_label_(292, 0), 9);
  ASSERT_EQ(mnist.test_one_hot_(292, 9), 1);
  // 5.2 边界测试（第一张图片的标签，最后一张图片的标签）
  ASSERT_EQ(mnist.test_label_(0, 0), 7);
  ASSERT_EQ(mnist.test_one_hot_(0, 7), 1);
  ASSERT_EQ(mnist.test_label_(9999, 0), 6);
  ASSERT_EQ(mnist.test_one_hot_(9999, 6), 1);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();  // 运行所有测试
}
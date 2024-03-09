#include <gtest/gtest.h>
#include <mnist/mnist.h>
#include <mountain_town/string/toml.h>

#include <cmath>
#include <unordered_map>

TEST(MnistTests, LoadMnist) {
  std::unordered_map<std::string, std::string> conf;
  std::string err = ReadSTOML("tests/testdata/config.toml", conf);
  if (err.empty() == false) {
    cout << err << endl;
    return;
  }
  // 准备数据（Prepare data）
  Mnist mnist(conf);
  err = mnist.LoadMnist();
  // 1. 正常文件载入测试
  // 1. Normal file loading test
  ASSERT_EQ(err, "");
  // 2. 测试用于训练的图像数据
  // 2. Testing the image data used for training
  // 2.1 General Test
  ASSERT_EQ(round(mnist.train_img_(0, 572) * 10000) / 10000, 0.8667);
  // 2.2 边界测试（第一张图片第一个非0值，最后一张图片最后一个非0值）
  // 2.2 Boundary test (first non-zero value in the first image, last non-zero
  // value in the last image)
  ASSERT_EQ(round(mnist.train_img_(0, 152) * 10000) / 10000, 0.0118);
  ASSERT_EQ(round(mnist.train_img_(59999, 682) * 10000) / 10000, 0.5255);
  // 3. 测试用于训练的标签数据
  // 3. Testing the labeled data used for training
  // 3.1 General Test
  ASSERT_EQ(mnist.train_label_(404, 0), 8);
  ASSERT_EQ(mnist.train_one_hot_(404, 8), 1);
  // 3.2 边界测试（第一张图片的标签，最后一张图片的标签）
  // 3.2 Boundary tests (labeling of the first image, labeling of the last
  // image)
  ASSERT_EQ(mnist.train_label_(0, 0), 5);
  ASSERT_EQ(mnist.train_one_hot_(0, 5), 1);
  ASSERT_EQ(mnist.train_label_(59999, 0), 8);
  ASSERT_EQ(mnist.train_one_hot_(59999, 8), 1);
  // 4. 测试用于测试的图像数据
  // 4. Testing the image data used for testing
  // 4.1 General Test
  ASSERT_EQ(round(mnist.test_img_(0, 684) * 10000) / 10000, 0.9961);
  // 4.2 边界测试（第一张图片第一个非0值，最后一张图片最后一个非0值）
  // 4.2 Boundary test (first non-zero value in the first image, last non-zero
  // value in the last image)
  ASSERT_EQ(round(mnist.test_img_(0, 202) * 10000) / 10000, 0.3294);
  ASSERT_EQ(round(mnist.test_img_(9999, 607) * 10000) / 10000, 0.0157);
  // 5. 测试用于测试的标签数据
  // 5. Test the labeling data used for testing
  // 5.1 General Test
  ASSERT_EQ(mnist.test_label_(292, 0), 9);
  ASSERT_EQ(mnist.test_one_hot_(292, 9), 1);
  // 5.2 边界测试（第一张图片的标签，最后一张图片的标签）
  // 5.2 Boundary tests (labeling of the first image, labeling of the last
  // image)
  ASSERT_EQ(mnist.test_label_(0, 0), 7);
  ASSERT_EQ(mnist.test_one_hot_(0, 7), 1);
  ASSERT_EQ(mnist.test_label_(9999, 0), 6);
  ASSERT_EQ(mnist.test_one_hot_(9999, 6), 1);
}
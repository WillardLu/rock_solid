set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(rock_solid
  main.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_lake/layers/affine.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_lake/layers/convolution.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_lake/layers/pooling.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_lake/layers/relu.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_lake/layers/sigmoid.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_lake/layers/softmaxwithloss.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_lake/neural_network/neural_network.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_town/math/random.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_town/string/basic.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_town/string/toml.cpp
  ${PROJECT_SOURCE_DIR}/src/mnist/mnist.cpp)

# 通过下面的设置，就可以不用在 main.cpp 中包含 mnist.cpp 的相对路径了，
# 因为 CMake 会自动把 mnist.cpp 添加到 rock_solid 的链接库中。
target_include_directories(rock_solid
 PUBLIC
  ${PROJECT_SOURCE_DIR}/src/
  ${PROJECT_SOURCE_DIR}/lib/)
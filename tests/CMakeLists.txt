find_package(GTest REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(
  rock_solid_test
  main_test.cpp
  mnist/mnist_test.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_town/string/basic.cpp
  ${PROJECT_SOURCE_DIR}/lib/mountain_town/string/toml.cpp
  ${PROJECT_SOURCE_DIR}/src/mnist/mnist.cpp)

target_link_libraries(rock_solid_test
 PRIVATE
  GTest::GTest)

# 通过下面的设置，就可以不用在 main_test.cpp 中包含 mnist.cpp 的相对路径了，
# 因为 CMake 会自动把 mnist.cpp 添加到 rock_solid_test 的链接库中。
target_include_directories(rock_solid_test
 PUBLIC
  ${PROJECT_SOURCE_DIR}/src/
  ${PROJECT_SOURCE_DIR}/lib/)

# add_test(NAME LoadMnist COMMAND rock_solid_test 1)

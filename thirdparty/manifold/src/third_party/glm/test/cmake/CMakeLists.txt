cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(test_find_glm)

find_package(glm REQUIRED)

add_executable(test_find_glm test_find_glm.cpp)
target_link_libraries(test_find_glm glm::glm)


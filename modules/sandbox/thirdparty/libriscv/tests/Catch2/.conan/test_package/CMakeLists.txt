cmake_minimum_required(VERSION 3.16)
project(PackageTest LANGUAGES CXX)

find_package(Catch2 CONFIG REQUIRED)

add_executable(test_package test_package.cpp)
target_link_libraries(test_package Catch2::Catch2WithMain)

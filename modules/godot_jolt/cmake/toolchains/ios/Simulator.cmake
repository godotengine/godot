include(${CMAKE_CURRENT_LIST_DIR}/Common.cmake)

set(CMAKE_OSX_SYSROOT iphonesimulator)
set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
set(CMAKE_C_COMPILER_TARGET arm64-apple-ios-simulator)
set(CMAKE_CXX_COMPILER_TARGET arm64-apple-ios-simulator)

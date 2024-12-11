include(${CMAKE_CURRENT_LIST_DIR}/Common.cmake)

set(CMAKE_OSX_SYSROOT iphoneos)
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_C_COMPILER_TARGET arm64-apple-ios)
set(CMAKE_CXX_COMPILER_TARGET arm64-apple-ios)

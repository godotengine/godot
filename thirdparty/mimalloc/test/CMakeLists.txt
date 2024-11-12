cmake_minimum_required(VERSION 3.0)
project(mimalloc-test C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
  if ("${CMAKE_BINARY_DIR}" MATCHES ".*(D|d)ebug$")
    message(STATUS "No build type selected, default to *** Debug ***")
    set(CMAKE_BUILD_TYPE "Debug")
  else()
    message(STATUS "No build type selected, default to *** Release ***")
    set(CMAKE_BUILD_TYPE "Release")
  endif()
endif()

# Import mimalloc (if installed)
find_package(mimalloc 2.0 REQUIRED NO_SYSTEM_ENVIRONMENT_PATH)
message(STATUS "Found mimalloc installed at: ${MIMALLOC_LIBRARY_DIR} (${MIMALLOC_VERSION_DIR})")

# overriding with a dynamic library
add_executable(dynamic-override  main-override.c)
target_link_libraries(dynamic-override PUBLIC mimalloc)

add_executable(dynamic-override-cxx  main-override.cpp)
target_link_libraries(dynamic-override-cxx PUBLIC mimalloc)


# overriding with a static object file works reliable as the symbols in the
# object file have priority over those in library files
add_executable(static-override-obj main-override.c ${MIMALLOC_OBJECT_DIR}/mimalloc.o)
target_include_directories(static-override-obj PUBLIC ${MIMALLOC_INCLUDE_DIR})
target_link_libraries(static-override-obj PUBLIC pthread)


# overriding with a static library works too if using the `mimalloc-override.h`
# header to redefine malloc/free. (the library already overrides new/delete)
add_executable(static-override-static main-override-static.c)
target_link_libraries(static-override-static PUBLIC mimalloc-static)


# overriding with a static library: this may not work if the library is linked too late
# on the command line after the C runtime library; but we cannot control that well in CMake
add_executable(static-override main-override.c)
target_link_libraries(static-override PUBLIC mimalloc-static)

add_executable(static-override-cxx  main-override.cpp)
target_link_libraries(static-override-cxx PUBLIC mimalloc-static)


## test memory errors
add_executable(test-wrong  test-wrong.c)
target_link_libraries(test-wrong PUBLIC mimalloc)

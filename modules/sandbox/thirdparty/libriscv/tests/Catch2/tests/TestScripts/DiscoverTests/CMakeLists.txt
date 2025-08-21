cmake_minimum_required(VERSION 3.16)
project(discover-tests-test LANGUAGES CXX)

add_executable(tests register-tests.cpp)
add_subdirectory(${CATCH2_PATH} catch2-build)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

enable_testing()
include(Catch)
set(extra_args)
if(CMAKE_VERSION GREATER_EQUAL 3.27)
  list(APPEND extra_args
    DL_PATHS "${CMAKE_CURRENT_LIST_DIR};${CMAKE_CURRENT_LIST_DIR}/.."
  )
endif()
catch_discover_tests(
  tests
  ADD_TAGS_AS_LABELS
  DISCOVERY_MODE PRE_TEST
  ${extra_args}
)

# DISCOVERY_MODE <POST_BUILD|PRE_TEST>

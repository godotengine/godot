add_library(effcee
            check.cc
            match.cc)
effcee_default_compile_options(effcee)
# We need to expose RE2's StringPiece.
target_include_directories(effcee
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/.. ${EFFCEE_RE2_DIR})
target_link_libraries(effcee PUBLIC re2 ${CMAKE_THREADS_LIB_INIT})

# TODO(dneto): Avoid installing gtest and gtest_main. ?!
install(
  FILES
    effcee.h
  DESTINATION
    include/effcee)
install(TARGETS effcee
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

if(EFFCEE_BUILD_TESTING)
  add_executable(effcee-test
                 check_test.cc
                 cursor_test.cc
                 diagnostic_test.cc
                 match_test.cc
                 options_test.cc
                 result_test.cc)
  effcee_default_compile_options(effcee-test)
  target_include_directories(effcee-test PRIVATE
                             ${gmock_SOURCE_DIR}/include
                             ${gtest_SOURCE_DIR}/include)
  target_link_libraries(effcee-test PRIVATE effcee gmock gtest_main)
  add_test(NAME effcee-test COMMAND effcee-test)
endif(EFFCEE_BUILD_TESTING)

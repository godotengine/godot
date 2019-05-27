add_executable(effcee-example main.cc)

target_link_libraries(effcee-example effcee)
if(UNIX AND NOT MINGW)
  set_target_properties(effcee-example PROPERTIES LINK_FLAGS -pthread)
endif()
if (WIN32 AND NOT MSVC)
  # For MinGW cross-compile, statically link to the C++ runtime
  set_target_properties(effcee-example PROPERTIES
	  LINK_FLAGS "-static -static-libgcc -static-libstdc++")
endif(WIN32 AND NOT MSVC)


if(EFFCEE_BUILD_TESTING)
  add_test(NAME effcee-example
          COMMAND ${PYTHON_EXE}
                  effcee-example-driver.py
                  $<TARGET_FILE:effcee-example>
                  example_data.txt
                  "CHECK: Hello"
                  "CHECK-SAME: world"
                  "CHECK-NEXT: Bees"
                  "CHECK-NOT: Sting"
                  "CHECK: Honey"
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif(EFFCEE_BUILD_TESTING)

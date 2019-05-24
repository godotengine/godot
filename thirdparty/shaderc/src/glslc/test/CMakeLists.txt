shaderc_add_nosetests(expect)
shaderc_add_nosetests(glslc_test_framework)

if(${SHADERC_ENABLE_TESTS})
  add_test(NAME glslc_tests
    COMMAND ${PYTHON_EXE}
    ${CMAKE_CURRENT_SOURCE_DIR}/glslc_test_framework.py
    $<TARGET_FILE:glslc_exe> $<TARGET_FILE:spirv-dis>
    --test-dir ${CMAKE_CURRENT_SOURCE_DIR})
endif()

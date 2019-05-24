# Find nosetests; see spirv_add_nosetests() for opting in to nosetests in a
# specific directory.
find_program(NOSETESTS_EXE NAMES nosetests PATHS $ENV{PYTHON_PACKAGE_PATH})
if (NOT NOSETESTS_EXE)
    message(STATUS "SPIRV-Tools: nosetests was not found - python support code will not be tested")
else()
    message(STATUS "SPIRV-Tools: nosetests found - python support code will be tested")
endif()

# Run nosetests on file ${PREFIX}_nosetest.py. Nosetests will look for classes
# and functions whose names start with "nosetest". The test name will be
# ${PREFIX}_nosetests.
function(spirv_add_nosetests PREFIX)
  if(NOT "${SPIRV_SKIP_TESTS}" AND NOSETESTS_EXE)
    add_test(
      NAME ${PREFIX}_nosetests
      COMMAND ${NOSETESTS_EXE} -m "^[Nn]ose[Tt]est" -v
        ${CMAKE_CURRENT_SOURCE_DIR}/${PREFIX}_nosetest.py)
  endif()
endfunction()

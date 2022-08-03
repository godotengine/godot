# Inputs:
#
# Variable             | Type     | Doc
# ---------------------|----------|--------------------------------------
# EXAMPLE_EXECUTABLE   | FilePath | Path to example executable
# FILECHECK_ENABLED    | Boolean  | Run FileCheck comparison test
# FILECHECK_EXECUTABLE | FilePath | Path to the LLVM FileCheck utility
# REFERENCE_FILE       | FilePath | Path to the FileCheck reference file

if (FILECHECK_ENABLED)
  if (NOT EXISTS "${REFERENCE_FILE}")
    message(FATAL_ERROR
      "FileCheck requested for '${EXAMPLE_EXECUTABLE}', but reference file "
      "does not exist at '${REFERENCE_FILE}`."
    )
  endif()

  # If the reference file is empty, validate that the example doesn't
  # produce any output.
  file(SIZE "${REFERENCE_FILE}" file_size)
  message("${REFERENCE_FILE}: ${file_size} bytes")

  if (file_size EQUAL 0)
    set(check_empty_output TRUE)
    set(filecheck_command)
  else()
    set(check_empty_output FALSE)
    set(filecheck_command COMMAND "${FILECHECK_EXECUTABLE}" "${REFERENCE_FILE}")
  endif()
endif()

execute_process(
  COMMAND "${EXAMPLE_EXECUTABLE}"
  ${filecheck_command}
  RESULT_VARIABLE exit_code
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if (NOT 0 EQUAL exit_code)
  message(FATAL_ERROR "${EXAMPLE_EXECUTABLE} failed (${exit_code}):\n${stderr}")
endif()

if (check_empty_output)
  string(LENGTH "${stdout}" stdout_size)
  if (NOT stdout_size EQUAL 0)
    message(FATAL_ERROR "${EXAMPLE_EXECUTABLE}: output received, but not expected:\n${stdout}")
  endif()
endif()

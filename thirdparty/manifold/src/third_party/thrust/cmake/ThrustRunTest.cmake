execute_process(
  COMMAND "${THRUST_BINARY}"
  RESULT_VARIABLE EXIT_CODE
)

if (NOT "0" STREQUAL "${EXIT_CODE}")
    message(FATAL_ERROR "${THRUST_BINARY} failed (${EXIT_CODE})")
endif ()

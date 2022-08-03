## This CMake script parses the output of ctest and prints a formatted list
## of individual test runtimes, sorted longest first.
##
## ctest > ctest_log
## cmake -DLOGFILE=ctest_log \
##       -P PrintCTestRunTimes.cmake
##
################################################################################

cmake_minimum_required(VERSION 3.15)

# Prepend the string with "0" until the string length equals the specified width
function(pad_string_with_zeros string_var width)
  set(local_string "${${string_var}}")
  string(LENGTH "${local_string}" size)
  while(size LESS width)
    string(PREPEND local_string "0")
    string(LENGTH "${local_string}" size)
  endwhile()
  set(${string_var} "${local_string}" PARENT_SCOPE)
endfunction()

################################################################################

if (NOT LOGFILE)
  message(FATAL_ERROR "Missing -DLOGFILE=<ctest output> argument.")
endif()

# Check if logfile exists
if (NOT EXISTS "${LOGFILE}")
  message(FATAL_ERROR "LOGFILE does not exist ('${LOGFILE}').")
endif()

string(JOIN "" regex
  "^[ ]*[0-9]+/[0-9]+[ ]+Test[ ]+#"
  "([0-9]+)"                          # Test ID
  ":[ ]+"
  "(.+)"                              # Test Name
  "[ ]+\\.+[ ]+"
  "(.+[^ ])"                              # Result
  "[ ]+"
  "([0-9]+)"                          # Seconds
  "\\.[0-9]+[ ]+sec[ ]*$"
)

message(DEBUG "Regex: ${regex}")

# Read the logfile and generate a map / keylist
set(keys)
file(STRINGS "${LOGFILE}" lines)
foreach(line ${lines})

  # Parse each build time
  string(REGEX MATCH "${regex}" _DUMMY "${line}")

  if (CMAKE_MATCH_COUNT EQUAL 4)
    set(test_id      "${CMAKE_MATCH_1}")
    set(test_name    "${CMAKE_MATCH_2}")
    set(test_result  "${CMAKE_MATCH_3}")
    set(tmp          "${CMAKE_MATCH_4}") # floor(runtime_seconds)

    # Compute human readable time
    math(EXPR days         "${tmp} / (60 * 60 * 24)")
    math(EXPR tmp          "${tmp} - (${days} * 60 * 60 * 24)")
    math(EXPR hours        "${tmp} / (60 * 60)")
    math(EXPR tmp          "${tmp} - (${hours} * 60 * 60)")
    math(EXPR minutes      "${tmp} / (60)")
    math(EXPR tmp          "${tmp} - (${minutes} * 60)")
    math(EXPR seconds      "${tmp}")

    # Format time components
    pad_string_with_zeros(days 3)
    pad_string_with_zeros(hours 2)
    pad_string_with_zeros(minutes 2)
    pad_string_with_zeros(seconds 2)

    # Construct table entry
    # Later values in the file for the same command overwrite earlier entries
    string(MAKE_C_IDENTIFIER "${test_id}" key)
    string(JOIN " | " ENTRY_${key}
      "${days}d ${hours}h ${minutes}m ${seconds}s"
      "${test_result}"
      "${test_id}: ${test_name}"
    )

    # Record the key:
    list(APPEND keys "${key}")
  endif()
endforeach()

list(REMOVE_DUPLICATES keys)

# Build the entry list:
set(entries)
foreach(key ${keys})
  list(APPEND entries "${ENTRY_${key}}")
endforeach()

if (NOT entries)
  message(FATAL_ERROR "LOGFILE contained no test times ('${LOGFILE}').")
endif()

# Sort in descending order:
list(SORT entries ORDER DESCENDING)

# Dump table:
foreach(entry ${entries})
  message(STATUS ${entry})
endforeach()

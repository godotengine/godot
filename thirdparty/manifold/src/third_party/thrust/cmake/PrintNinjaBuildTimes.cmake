## This CMake script parses a .ninja_log file (LOGFILE) and prints a list of
## build/link times, sorted longest first.
##
## cmake -DLOGFILE=<.ninja_log file> \
##       -P PrintNinjaBuildTimes.cmake
##
## If LOGFILE is omitted, the current directory's .ninja_log file is used.
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
  set(LOGFILE ".ninja_log")
endif()

# Check if logfile exists
if (NOT EXISTS "${LOGFILE}")
  message(FATAL_ERROR "LOGFILE does not exist ('${LOGFILE}').")
endif()

# Read the logfile and generate a map / keylist
set(keys)
file(STRINGS "${LOGFILE}" lines)
foreach(line ${lines})

  # Parse each build time
  string(REGEX MATCH
    "^([0-9]+)\t([0-9]+)\t[0-9]+\t([^\t]+)+\t[0-9a-fA-F]+$" _DUMMY "${line}")

  if (CMAKE_MATCH_COUNT EQUAL 3)
    set(start_ms ${CMAKE_MATCH_1})
    set(end_ms ${CMAKE_MATCH_2})
    set(command "${CMAKE_MATCH_3}")
    math(EXPR runtime_ms "${end_ms} - ${start_ms}")

    # Compute human readable time
    math(EXPR days         "${runtime_ms} / (1000 * 60 * 60 * 24)")
    math(EXPR runtime_ms   "${runtime_ms} - (${days} * 1000 * 60 * 60 * 24)")
    math(EXPR hours        "${runtime_ms} / (1000 * 60 * 60)")
    math(EXPR runtime_ms   "${runtime_ms} - (${hours} * 1000 * 60 * 60)")
    math(EXPR minutes      "${runtime_ms} / (1000 * 60)")
    math(EXPR runtime_ms   "${runtime_ms} - (${minutes} * 1000 * 60)")
    math(EXPR seconds      "${runtime_ms} / 1000")
    math(EXPR milliseconds "${runtime_ms} - (${seconds} * 1000)")

    # Format time components
    pad_string_with_zeros(days 3)
    pad_string_with_zeros(hours 2)
    pad_string_with_zeros(minutes 2)
    pad_string_with_zeros(seconds 2)
    pad_string_with_zeros(milliseconds 3)

    # Construct table entry
    # Later values in the file for the same command overwrite earlier entries
    string(MAKE_C_IDENTIFIER "${command}" key)
    set(ENTRY_${key}
      "${days}d ${hours}h ${minutes}m ${seconds}s ${milliseconds}ms | ${command}"
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
  message(FATAL_ERROR "LOGFILE contained no build entries ('${LOGFILE}').")
endif()

# Sort in descending order:
list(SORT entries)
list(REVERSE entries)

# Dump table:
message(STATUS "-----------------------+----------------------------")
message(STATUS "Time                   | Command                    ")
message(STATUS "-----------------------+----------------------------")

foreach(entry ${entries})
  message(STATUS ${entry})
endforeach()

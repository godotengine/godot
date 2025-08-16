
#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)

# SPDX-License-Identifier: BSL-1.0

##
# This file contains options that are materialized into the Catch2
# compiled library. All of them default to OFF, as even the positive
# forms correspond to the user _forcing_ them to ON, while being OFF
# means that Catch2 can use its own autodetection.
#
# For detailed docs look into docs/configuration.md

macro(AddOverridableConfigOption OptionBaseName)
  option(CATCH_CONFIG_${OptionBaseName} "Read docs/configuration.md for details" OFF)
  option(CATCH_CONFIG_NO_${OptionBaseName} "Read docs/configuration.md for details" OFF)
  mark_as_advanced(CATCH_CONFIG_${OptionBaseName} CATCH_CONFIG_NO_${OptionBaseName})
endmacro()

macro(AddConfigOption OptionBaseName)
  option(CATCH_CONFIG_${OptionBaseName} "Read docs/configuration.md for details" OFF)
  mark_as_advanced(CATCH_CONFIG_${OptionBaseName})
endmacro()

set(_OverridableOptions
  "ANDROID_LOGWRITE"
  "BAZEL_SUPPORT"
  "COLOUR_WIN32"
  "COUNTER"
  "CPP11_TO_STRING"
  "CPP17_BYTE"
  "CPP17_OPTIONAL"
  "CPP17_STRING_VIEW"
  "CPP17_UNCAUGHT_EXCEPTIONS"
  "CPP17_VARIANT"
  "GLOBAL_NEXTAFTER"
  "POSIX_SIGNALS"
  "USE_ASYNC"
  "WCHAR"
  "WINDOWS_SEH"
  "GETENV"
  "EXPERIMENTAL_STATIC_ANALYSIS_SUPPORT"
  "USE_BUILTIN_CONSTANT_P"
  "DEPRECATION_ANNOTATIONS"
  "EXPERIMENTAL_THREAD_SAFE_ASSERTIONS"
)

foreach(OptionName ${_OverridableOptions})
  AddOverridableConfigOption(${OptionName})
endforeach()

set(_OtherConfigOptions
  "DISABLE_EXCEPTIONS"
  "DISABLE_EXCEPTIONS_CUSTOM_HANDLER"
  "DISABLE"
  "DISABLE_STRINGIFICATION"
  "ENABLE_ALL_STRINGMAKERS"
  "ENABLE_OPTIONAL_STRINGMAKER"
  "ENABLE_PAIR_STRINGMAKER"
  "ENABLE_TUPLE_STRINGMAKER"
  "ENABLE_VARIANT_STRINGMAKER"
  "EXPERIMENTAL_REDIRECT"
  "FAST_COMPILE"
  "NOSTDOUT"
  "PREFIX_ALL"
  "PREFIX_MESSAGES"
  "WINDOWS_CRTDBG"
)


foreach(OptionName ${_OtherConfigOptions})
  AddConfigOption(${OptionName})
endforeach()
if(DEFINED BUILD_SHARED_LIBS)
  set(CATCH_CONFIG_SHARED_LIBRARY ${BUILD_SHARED_LIBS})
else()
  set(CATCH_CONFIG_SHARED_LIBRARY "")
endif()

set(CATCH_CONFIG_DEFAULT_REPORTER "console" CACHE STRING "Read docs/configuration.md for details. The name of the reporter should be without quotes.")
set(CATCH_CONFIG_CONSOLE_WIDTH "80" CACHE STRING "Read docs/configuration.md for details. Must form a valid integer literal.")

mark_as_advanced(CATCH_CONFIG_SHARED_LIBRARY CATCH_CONFIG_DEFAULT_REPORTER CATCH_CONFIG_CONSOLE_WIDTH)

# There is no good way to both turn this into a CMake cache variable,
# and keep reasonable default semantics inside the project. Thus we do
# not define it and users have to provide it as an outside variable.
#set(CATCH_CONFIG_FALLBACK_STRINGIFIER "" CACHE STRING "Read docs/configuration.md for details.")

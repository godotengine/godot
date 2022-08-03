# Set up compiler paths and apply temporary hacks to support NVC++.
# This file must be included before enabling any languages.

# Temporary hacks to make NVC++ work; this requires you to define
# `CMAKE_CUDA_COMPILER_ID=NVCXX and `CMAKE_CUDA_COMPILER_FORCED=ON`.
if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  # If using NVC++, don't set CXX compiler
  if (NOT "${CMAKE_CXX_COMPILER}" STREQUAL "")
    unset(CMAKE_CXX_COMPILER CACHE)
    message(FATAL_ERROR "You are using NVC++ as your CUDA C++ compiler, but have"
      " specified a different ISO C++ compiler; NVC++ acts as both, so please"
      " unset the CMAKE_CXX_COMPILER variable."
    )
  endif()

  # We don't set CMAKE_CUDA_HOST_COMPILER for NVC++; if we do, CMake tries to
  # pass `-ccbin ${CMAKE_CUDA_HOST_COMPILER}` to NVC++, which it doesn't
  # understand.
  if (NOT "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "")
    unset(CMAKE_CUDA_HOST_COMPILER CACHE)
    message(FATAL_ERROR "You are using NVC++ as your CUDA C++ compiler, but have"
      " specified a different host ISO C++ compiler; NVC++ acts as both, so"
      " please unset the CMAKE_CUDA_HOST_COMPILER variable."
    )
  endif()

  set(CMAKE_CXX_COMPILER "${CMAKE_CUDA_COMPILER}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -stdpar")
  set(CMAKE_CUDA_HOST_LINK_LAUNCHER "${CMAKE_CUDA_COMPILER}")
  set(CMAKE_CUDA_LINK_EXECUTABLE
    "<CMAKE_CUDA_HOST_LINK_LAUNCHER> <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

  # Setup CMAKE_CXX_LIBRARY_ARCHITECTURE on Debian/Ubuntu so that find_package
  # works properly.
  if (EXISTS /etc/debian_version)
    if (NOT CMAKE_CXX_LIBRARY_ARCHITECTURE)
      file(GLOB files_in_lib RELATIVE /lib /lib/*-linux-gnu* )
      foreach (file ${files_in_lib})
        if ("${file}" MATCHES "${CMAKE_LIBRARY_ARCHITECTURE_REGEX}")
          set(CMAKE_CXX_LIBRARY_ARCHITECTURE ${file})
          break()
        endif()
      endforeach()
    endif()
    if (NOT CMAKE_LIBRARY_ARCHITECTURE)
      set(CMAKE_LIBRARY_ARCHITECTURE ${CMAKE_CXX_LIBRARY_ARCHITECTURE})
    endif()
  endif()
endif()

# We don't set CMAKE_CUDA_HOST_COMPILER for NVC++; if we do, CMake tries to
# pass `-ccbin ${CMAKE_CUDA_HOST_COMPILER}` to NVC++, which it doesn't
# understand.
if ((NOT "NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}"))
  if (NOT ("${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "" OR
    "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "${CMAKE_CXX_COMPILER}"))
    set(tmp "${CMAKE_CUDA_HOST_COMPILER}")
    unset(CMAKE_CUDA_HOST_COMPILER CACHE)
    message(FATAL_ERROR
      "For convenience, Thrust's test harness uses CMAKE_CXX_COMPILER for the "
      "CUDA host compiler. Refusing to overwrite specified "
      "CMAKE_CUDA_HOST_COMPILER -- please reconfigure without setting this "
      "variable. Currently:\n"
      "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}\n"
      "CMAKE_CUDA_HOST_COMPILER=${tmp}"
    )
  endif ()
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif ()

# Temporary hacks to make NVC++ work; this requires you to define
# `CMAKE_CUDA_COMPILER_ID=NVCXX and `CMAKE_CUDA_COMPILER_FORCED=ON`.
if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  # Need 3.17 for the properties used below.
  cmake_minimum_required(VERSION 3.17)

  set(CMAKE_CUDA_STANDARD_DEFAULT 03)

  set(CMAKE_CUDA03_STANDARD_COMPILE_OPTION "-std=c++03")
  set(CMAKE_CUDA03_EXTENSION_COMPILE_OPTION "-std=c++03")
  set(CMAKE_CUDA03_STANDARD__HAS_FULL_SUPPORT TRUE)
  set_property(GLOBAL PROPERTY CMAKE_CUDA03_KNOWN_FEATURES)

  set(CMAKE_CUDA11_STANDARD_COMPILE_OPTION "-std=c++11")
  set(CMAKE_CUDA11_EXTENSION_COMPILE_OPTION "-std=c++11")
  set(CMAKE_CUDA11_STANDARD__HAS_FULL_SUPPORT TRUE)
  set_property(GLOBAL PROPERTY CMAKE_CUDA11_KNOWN_FEATURES)

  set(CMAKE_CUDA14_STANDARD_COMPILE_OPTION "-std=c++14")
  set(CMAKE_CUDA14_EXTENSION_COMPILE_OPTION "-std=c++14")
  set(CMAKE_CUDA14_STANDARD__HAS_FULL_SUPPORT TRUE)
  set_property(GLOBAL PROPERTY CMAKE_CUDA14_KNOWN_FEATURES)

  set(CMAKE_CUDA17_STANDARD_COMPILE_OPTION "-std=c++17")
  set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "-std=c++17")
  set(CMAKE_CUDA17_STANDARD__HAS_FULL_SUPPORT TRUE)
  set_property(GLOBAL PROPERTY CMAKE_CUDA17_KNOWN_FEATURES)

  include(Internal/FeatureTesting)
  include(Compiler/CMakeCommonCompilerMacros)
  cmake_record_cuda_compile_features()

  set(CMAKE_CUDA_COMPILE_FEATURES
    ${CMAKE_CUDA03_COMPILE_FEATURES}
    ${CMAKE_CUDA11_COMPILE_FEATURES}
    ${CMAKE_CUDA14_COMPILE_FEATURES}
    ${CMAKE_CUDA17_COMPILE_FEATURES}
    ${CMAKE_CUDA20_COMPILE_FEATURES}
  )
endif()

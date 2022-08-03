# This file defines thrust_configure_multiconfig(), which sets up and handles
# the MultiConfig options that allow multiple host/device/dialect configurations
# to be generated from a single thrust build.

function(thrust_configure_multiconfig)
  option(THRUST_ENABLE_MULTICONFIG "Enable multiconfig options for coverage testing." OFF)

  # Dialects:
  set(THRUST_CPP_DIALECT_OPTIONS
    11 14 17
    CACHE INTERNAL "C++ dialects supported by Thrust." FORCE
  )

  if (THRUST_ENABLE_MULTICONFIG)
    # Handle dialect options:
    foreach (dialect IN LISTS THRUST_CPP_DIALECT_OPTIONS)
      set(default_value OFF)
      if (dialect EQUAL 14) # Default to just 14 on:
        set(default_value ON)
      endif()
      option(THRUST_MULTICONFIG_ENABLE_DIALECT_CPP${dialect}
        "Generate C++${dialect} build configurations."
        ${default_value}
      )
    endforeach()

    # Option to enable all standards supported by the CUDA and CXX compilers:
    option(THRUST_MULTICONFIG_ENABLE_DIALECT_ALL
      "Generate build configurations for all C++ standards supported by the configured compilers."
      OFF
    )

    # Option to enable only the most recent supported dialect:
    option(THRUST_MULTICONFIG_ENABLE_DIALECT_LATEST
      "Generate a single build configuration for the most recent C++ standard supported by the configured compilers."
      OFF
    )

    # Systems:
    option(THRUST_MULTICONFIG_ENABLE_SYSTEM_CPP "Generate build configurations that use CPP." ON)
    option(THRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA "Generate build configurations that use CUDA." ON)
    option(THRUST_MULTICONFIG_ENABLE_SYSTEM_OMP "Generate build configurations that use OpenMP." OFF)
    option(THRUST_MULTICONFIG_ENABLE_SYSTEM_TBB "Generate build configurations that use TBB." OFF)

    # CMake fixed C++17 support for NVCC + MSVC targets in 3.18.3:
    if (THRUST_MULTICONFIG_ENABLE_DIALECT_CPP17 AND
        THRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA)
      cmake_minimum_required(VERSION 3.18.3)
    endif()

    # Workload:
    # - `SMALL`: [3 configs] Minimal coverage and validation of each device system against the `CPP` host.
    # - `MEDIUM`: [6 configs] Cheap extended coverage.
    # - `LARGE`: [8 configs] Expensive extended coverage. Include all useful build configurations.
    # - `FULL`: [12 configs] The complete cross product of all possible build configurations.
    #
    # Config   | Workloads | Value      | Expense   | Note
    # ---------|-----------|------------|-----------|-----------------------------
    # CPP/CUDA | F L M S   | Essential  | Expensive | Validates CUDA against CPP
    # CPP/OMP  | F L M S   | Essential  | Cheap     | Validates OMP against CPP
    # CPP/TBB  | F L M S   | Essential  | Cheap     | Validates TBB against CPP
    # CPP/CPP  | F L M     | Important  | Cheap     | Tests CPP as device
    # OMP/OMP  | F L M     | Important  | Cheap     | Tests OMP as host
    # TBB/TBB  | F L M     | Important  | Cheap     | Tests TBB as host
    # TBB/CUDA | F L       | Important  | Expensive | Validates TBB/CUDA interop
    # OMP/CUDA | F L       | Important  | Expensive | Validates OMP/CUDA interop
    # TBB/OMP  | F         | Not useful | Cheap     | Mixes CPU-parallel systems
    # OMP/TBB  | F         | Not useful | Cheap     | Mixes CPU-parallel systems
    # TBB/CPP  | F         | Not Useful | Cheap     | Parallel host, serial device
    # OMP/CPP  | F         | Not Useful | Cheap     | Parallel host, serial device

    set(THRUST_MULTICONFIG_WORKLOAD SMALL CACHE STRING
      "Limit host/device configs: SMALL (up to 3 h/d combos per dialect), MEDIUM(6), LARGE(8), FULL(12)"
    )
    set_property(CACHE THRUST_MULTICONFIG_WORKLOAD PROPERTY STRINGS
      SMALL MEDIUM LARGE FULL
    )
    set(THRUST_MULTICONFIG_WORKLOAD_SMALL_CONFIGS
      CPP_OMP CPP_TBB CPP_CUDA
      CACHE INTERNAL "Host/device combos enabled for SMALL workloads." FORCE
    )
    set(THRUST_MULTICONFIG_WORKLOAD_MEDIUM_CONFIGS
      ${THRUST_MULTICONFIG_WORKLOAD_SMALL_CONFIGS}
      CPP_CPP TBB_TBB OMP_OMP
      CACHE INTERNAL "Host/device combos enabled for MEDIUM workloads." FORCE
    )
    set(THRUST_MULTICONFIG_WORKLOAD_LARGE_CONFIGS
      ${THRUST_MULTICONFIG_WORKLOAD_MEDIUM_CONFIGS}
      OMP_CUDA TBB_CUDA
      CACHE INTERNAL "Host/device combos enabled for LARGE workloads." FORCE
    )
    set(THRUST_MULTICONFIG_WORKLOAD_FULL_CONFIGS
      ${THRUST_MULTICONFIG_WORKLOAD_LARGE_CONFIGS}
      OMP_CPP TBB_CPP OMP_TBB TBB_OMP
      CACHE INTERNAL "Host/device combos enabled for FULL workloads." FORCE
    )

    # Hide the single config options if they exist from a previous run:
    if (DEFINED THRUST_HOST_SYSTEM)
      set_property(CACHE THRUST_HOST_SYSTEM PROPERTY TYPE INTERNAL)
      set_property(CACHE THRUST_DEVICE_SYSTEM PROPERTY TYPE INTERNAL)
    endif()
    if (DEFINED THRUST_CPP_DIALECT)
      set_property(CACHE THRUST_CPP_DIALECT PROPERTY TYPE INTERNAL)
    endif()

  else() # Single config:
    # Restore system option visibility if these cache options already exist
    # from a previous run.
    if (DEFINED THRUST_HOST_SYSTEM)
      set_property(CACHE THRUST_HOST_SYSTEM PROPERTY TYPE STRING)
      set_property(CACHE THRUST_DEVICE_SYSTEM PROPERTY TYPE STRING)
    endif()

    set(THRUST_CPP_DIALECT 14
      CACHE STRING "The C++ standard to target: ${THRUST_CPP_DIALECT_OPTIONS}"
    )
    set_property(CACHE THRUST_CPP_DIALECT
      PROPERTY STRINGS
      ${THRUST_CPP_DIALECT_OPTIONS}
    )

    # CMake fixed C++17 support for NVCC + MSVC targets in 3.18.3:
    if (THRUST_CPP_DIALECT EQUAL 17 AND
        THRUST_DEVICE_SYSTEM STREQUAL "CUDA")
      cmake_minimum_required(VERSION 3.18.3)
    endif()
  endif()
endfunction()

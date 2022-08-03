# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Meta target for all configs' header builds:
add_custom_target(thrust.all.headers)

foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)
  set(config_systems ${config_host} ${config_device})

  string(TOLOWER "${config_host}" host_lower)
  string(TOLOWER "${config_device}" device_lower)

  # GLOB ALL THE THINGS
  set(headers_globs thrust/*.h)
  set(headers_exclude_systems_globs thrust/system/*/*)
  set(headers_systems_globs
    thrust/system/${host_lower}/*
    thrust/system/${device_lower}/*
  )
  set(headers_exclude_details_globs
    thrust/detail/*
    thrust/*/detail/*
    thrust/*/*/detail/*
  )

  # Get all .h files...
  file(GLOB_RECURSE headers
    RELATIVE "${Thrust_SOURCE_DIR}/thrust"
    CONFIGURE_DEPENDS
    ${headers_globs}
  )

  # ...then remove all system specific headers...
  file(GLOB_RECURSE headers_exclude_systems
    RELATIVE "${Thrust_SOURCE_DIR}/thrust"
    CONFIGURE_DEPENDS
    ${headers_exclude_systems_globs}
  )
  list(REMOVE_ITEM headers ${headers_exclude_systems})

  # ...then add all headers specific to the selected host and device systems back again...
  file(GLOB_RECURSE headers_systems
    RELATIVE ${Thrust_SOURCE_DIR}/thrust
    CONFIGURE_DEPENDS
    ${headers_systems_globs}
  )
  list(APPEND headers ${headers_systems})

  # ...and remove all the detail headers (also removing the detail headers from the selected systems).
  file(GLOB_RECURSE headers_exclude_details
    RELATIVE "${Thrust_SOURCE_DIR}/thrust"
    CONFIGURE_DEPENDS
    ${headers_exclude_details_globs}
  )
  list(REMOVE_ITEM headers ${headers_exclude_details})

  # List of headers that aren't implemented for all backends, but are implemented for CUDA.
  set(partially_implemented_CUDA
    async/copy.h
    async/for_each.h
    async/reduce.h
    async/scan.h
    async/sort.h
    async/transform.h
    event.h
    future.h
  )

  # List of headers that aren't implemented for all backends, but are implemented for CPP.
  set(partially_implemented_CPP
  )

  # List of headers that aren't implemented for all backends, but are implemented for TBB.
  set(partially_implemented_TBB
  )

  # List of headers that aren't implemented for all backends, but are implemented for OMP.
  set(partially_implemented_OMP
  )

  # List of all partially implemented headers.
  set(partially_implemented
    ${partially_implemented_CUDA}
    ${partially_implemented_CPP}
    ${partially_implemented_TBB}
    ${partially_implemented_OMP}
  )
  list(REMOVE_DUPLICATES partially_implemented)

  set(headertest_srcs)

  foreach (header IN LISTS headers)
    if ("${header}" IN_LIST partially_implemented)
      # This header is partially implemented on _some_ backends...
      if (NOT "${header}" IN_LIST partially_implemented_${config_device})
        # ...but not on the selected one.
        continue()
      endif()
    endif()

    set(headertest_src_ext .cpp)
    if ("CUDA" STREQUAL "${config_device}")
      set(headertest_src_ext .cu)
    endif()

    set(headertest_src "headers/${config_prefix}/${header}${headertest_src_ext}")
    configure_file("${Thrust_SOURCE_DIR}/cmake/header_test.in" "${headertest_src}")

    list(APPEND headertest_srcs "${headertest_src}")
  endforeach()

  set(headertest_target ${config_prefix}.headers)
  add_library(${headertest_target} OBJECT ${headertest_srcs})
  target_link_libraries(${headertest_target} PUBLIC ${thrust_target})
  # Wrap Thrust/CUB in a custom namespace to check proper use of ns macros:
  target_compile_definitions(${headertest_target} PRIVATE
    "THRUST_WRAPPED_NAMESPACE=wrapped_thrust"
    "CUB_WRAPPED_NAMESPACE=wrapped_cub"
  )
  thrust_clone_target_properties(${headertest_target} ${thrust_target})

  # Disable macro checks on TBB; the TBB atomic implementation uses `I` and
  # our checks will issue false errors.
  if ("TBB" IN_LIST config_systems)
    target_compile_definitions(${headertest_target}
      PRIVATE THRUST_IGNORE_MACRO_CHECKS
    )
  endif()

  add_dependencies(thrust.all.headers ${headertest_target})
  add_dependencies(${config_prefix}.all ${headertest_target})
endforeach()

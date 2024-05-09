set(files
  __bit_reference
  __bsd_locale_defaults.h
  __bsd_locale_fallbacks.h
  __errc
  __debug
  __functional_03
  __functional_base
  __functional_base_03
  __hash_table
  __libcpp_version
  __locale
  __mutex_base
  __node_handle
  __nullptr
  __split_buffer
  __sso_allocator
  __std_stream
  __string
  __threading_support
  __tree
  __tuple
  __undef_macros
  algorithm
  any
  array
  atomic
  barrier
  bit
  bitset
  cassert
  ccomplex
  cctype
  cerrno
  cfenv
  cfloat
  charconv
  chrono
  cinttypes
  ciso646
  climits
  clocale
  cmath
  codecvt
  compare
  complex
  complex.h
  condition_variable
  csetjmp
  csignal
  cstdarg
  cstdbool
  cstddef
  cstdint
  cstdio
  cstdlib
  cstring
  ctgmath
  ctime
  ctype.h
  cwchar
  cwctype
  deque
  errno.h
  exception
  execution
  experimental/__config
  experimental/__memory
  experimental/algorithm
  experimental/coroutine
  experimental/deque
  experimental/filesystem
  experimental/forward_list
  experimental/functional
  experimental/iterator
  experimental/list
  experimental/map
  experimental/memory_resource
  experimental/propagate_const
  experimental/regex
  experimental/set
  experimental/simd
  experimental/string
  experimental/type_traits
  experimental/unordered_map
  experimental/unordered_set
  experimental/utility
  experimental/vector
  ext/__hash
  ext/hash_map
  ext/hash_set
  fenv.h
  filesystem
  float.h
  forward_list
  fstream
  functional
  future
  initializer_list
  inttypes.h
  iomanip
  ios
  iosfwd
  iostream
  istream
  iterator
  latch
  limits
  limits.h
  list
  locale
  locale.h
  map
  math.h
  memory
  module.modulemap
  mutex
  new
  numeric
  optional
  ostream
  queue
  random
  ratio
  regex
  scoped_allocator
  semaphore
  set
  setjmp.h
  shared_mutex
  span
  sstream
  stack
  stdbool.h
  stddef.h
  stdexcept
  stdint.h
  stdio.h
  stdlib.h
  streambuf
  string
  string.h
  string_view
  strstream
  system_error
  tgmath.h
  thread
  tuple
  type_traits
  typeindex
  typeinfo
  unordered_map
  unordered_set
  utility
  valarray
  variant
  vector
  version
  wchar.h
  wctype.h
  )

if(LIBCXX_INSTALL_SUPPORT_HEADERS)
  set(files
    ${files}
    support/android/locale_bionic.h
    support/fuchsia/xlocale.h
    support/ibm/limits.h
    support/ibm/locale_mgmt_aix.h
    support/ibm/support.h
    support/ibm/xlocale.h
    support/musl/xlocale.h
    support/newlib/xlocale.h
    support/solaris/floatingpoint.h
    support/solaris/wchar.h
    support/solaris/xlocale.h
    support/win32/limits_msvc_win32.h
    support/win32/locale_win32.h
    support/xlocale/__nop_locale_mgmt.h
    support/xlocale/__posix_l_fallback.h
    support/xlocale/__strtonum_fallback.h
    )
endif()

if (LIBCXX_NEEDS_SITE_CONFIG)
  # Generate a custom __config header. The new header is created
  # by prepending __config_site to the current __config header.
  add_custom_command(OUTPUT ${LIBCXX_BINARY_DIR}/__generated_config
    COMMAND ${PYTHON_EXECUTABLE} ${LIBCXX_SOURCE_DIR}/utils/cat_files.py
      ${LIBCXX_BINARY_DIR}/__config_site
      ${LIBCXX_SOURCE_DIR}/include/__config
      -o ${LIBCXX_BINARY_DIR}/__generated_config
    DEPENDS ${LIBCXX_SOURCE_DIR}/include/__config
            ${LIBCXX_BINARY_DIR}/__config_site
  )
  # Add a target that executes the generation commands.
  add_custom_target(cxx-generated-config ALL
    DEPENDS ${LIBCXX_BINARY_DIR}/__generated_config)
  set(generated_config_deps cxx-generated-config)
else()
  set(files
    ${files}
    __config
    )
endif()

# In some build configuraitons (like boostrapping clang), we need to be able to
# install the libcxx headers before CMake configuraiton for libcxx runs. Making
# the name of this target configurable allows LLVM/runtimes/CMakeLists.txt to
# add this subdirectory to the LLVM build to put libcxx's headers in place
# before libcxx's build configuration is run.
if (NOT CXX_HEADER_TARGET)
  set(CXX_HEADER_TARGET cxx-headers)
endif()
if(NOT LIBCXX_USING_INSTALLED_LLVM AND LIBCXX_HEADER_DIR)
  set(output_dir ${LIBCXX_HEADER_DIR}/include/c++/v1)

  set(out_files)
  foreach(f ${files})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    set(dst ${output_dir}/${f})
    add_custom_command(OUTPUT ${dst}
      DEPENDS ${src}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
      COMMENT "Copying CXX header ${f}")
    list(APPEND out_files ${dst})
  endforeach()

  if (LIBCXX_NEEDS_SITE_CONFIG)
    # Copy the generated header as __config into build directory.
    set(src ${LIBCXX_BINARY_DIR}/__generated_config)
    set(dst ${output_dir}/__config)
    add_custom_command(OUTPUT ${dst}
        DEPENDS ${src} ${generated_config_deps}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
        COMMENT "Copying CXX __config")
    list(APPEND out_files ${dst})
  endif()

  add_custom_target(${CXX_HEADER_TARGET} ALL DEPENDS ${out_files} ${LIBCXX_CXX_ABI_HEADER_TARGET})
else()
  add_custom_target(${CXX_HEADER_TARGET})
endif()
set_target_properties(${CXX_HEADER_TARGET} PROPERTIES FOLDER "Misc")

if (LIBCXX_INSTALL_HEADERS)
  foreach(file ${files})
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file}
      DESTINATION ${LIBCXX_INSTALL_HEADER_PREFIX}include/c++/v1/${dir}
      COMPONENT ${CXX_HEADER_TARGET}
      PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    )
  endforeach()

  if (LIBCXX_NEEDS_SITE_CONFIG)
    # Install the generated header as __config.
    install(FILES ${LIBCXX_BINARY_DIR}/__generated_config
      DESTINATION ${LIBCXX_INSTALL_HEADER_PREFIX}include/c++/v1
      PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
      RENAME __config
      COMPONENT ${CXX_HEADER_TARGET})
  endif()

  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_custom_target(install-${CXX_HEADER_TARGET}
                      DEPENDS ${CXX_HEADER_TARGET} ${generated_config_deps}
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=${CXX_HEADER_TARGET}
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    # Stripping is a no-op for headers
    add_custom_target(install-${CXX_HEADER_TARGET}-stripped DEPENDS install-${CXX_HEADER_TARGET})

    add_custom_target(install-libcxx-headers DEPENDS install-${CXX_HEADER_TARGET})
    add_custom_target(install-libcxx-headers-stripped DEPENDS install-${CXX_HEADER_TARGET}-stripped)
  endif()
endif()

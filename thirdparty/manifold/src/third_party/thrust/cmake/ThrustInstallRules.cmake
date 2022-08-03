# Bring in CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# Thrust is a header library; no need to build anything before installing:
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

install(DIRECTORY "${Thrust_SOURCE_DIR}/thrust"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.inl"
)

install(DIRECTORY "${Thrust_SOURCE_DIR}/thrust/cmake/"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/thrust"
  PATTERN thrust-header-search EXCLUDE
)
# Need to configure a file to store the infix specified in
# CMAKE_INSTALL_INCLUDEDIR since it can be defined by the user
set(install_location "${CMAKE_INSTALL_LIBDIR}/cmake/thrust")
configure_file("${Thrust_SOURCE_DIR}/thrust/cmake/thrust-header-search.cmake.in"
  "${Thrust_BINARY_DIR}/thrust/cmake/thrust-header-search.cmake"
  @ONLY)
install(FILES "${Thrust_BINARY_DIR}/thrust/cmake/thrust-header-search.cmake"
  DESTINATION "${install_location}")

# Depending on how Thrust is configured, CUB's CMake scripts may or may not be
# included, so maintain a set of CUB install rules in both projects. By default
# CUB headers are installed alongside Thrust -- this may be disabled by turning
# off THRUST_INSTALL_CUB_HEADERS.
option(THRUST_INSTALL_CUB_HEADERS "Include cub headers when installing." ON)
if (THRUST_INSTALL_CUB_HEADERS)
  install(DIRECTORY "${Thrust_SOURCE_DIR}/dependencies/cub/cub"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
      PATTERN "*.cuh"
  )

  # Need to configure a file to store THRUST_INSTALL_HEADER_INFIX
  install(DIRECTORY "${Thrust_SOURCE_DIR}/dependencies/cub/cub/cmake/"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cub"
    PATTERN cub-header-search EXCLUDE
  )
  set(install_location "${CMAKE_INSTALL_LIBDIR}/cmake/cub")
  configure_file("${Thrust_SOURCE_DIR}/dependencies/cub/cub/cmake/cub-header-search.cmake.in"
    "${Thrust_BINARY_DIR}/dependencies/cub/cub/cmake/cub-header-search.cmake"
    @ONLY)
  install(FILES "${Thrust_BINARY_DIR}/dependencies/cub/cub/cmake/cub-header-search.cmake"
    DESTINATION "${install_location}")
endif()

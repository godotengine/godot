include(${CMAKE_CURRENT_LIST_DIR}/mimalloc.cmake)
get_filename_component(MIMALLOC_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" PATH)  # one up from the cmake dir, e.g. /usr/local/lib/cmake/mimalloc-2.0
get_filename_component(MIMALLOC_VERSION_DIR "${CMAKE_CURRENT_LIST_DIR}" NAME)
string(REPLACE "/lib/cmake" "/lib" MIMALLOC_LIBRARY_DIR "${MIMALLOC_CMAKE_DIR}")
if("${MIMALLOC_VERSION_DIR}" EQUAL "mimalloc")
  # top level install
  string(REPLACE "/lib/cmake" "/include" MIMALLOC_INCLUDE_DIR "${MIMALLOC_CMAKE_DIR}")
  set(MIMALLOC_OBJECT_DIR "${MIMALLOC_LIBRARY_DIR}")
else()
  # versioned
  string(REPLACE "/lib/cmake/" "/include/" MIMALLOC_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}")
  string(REPLACE "/lib/cmake/" "/lib/" MIMALLOC_OBJECT_DIR "${CMAKE_CURRENT_LIST_DIR}")
endif()
set(MIMALLOC_TARGET_DIR "${MIMALLOC_LIBRARY_DIR}") # legacy

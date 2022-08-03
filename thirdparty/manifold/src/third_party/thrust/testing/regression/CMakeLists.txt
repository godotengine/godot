#
# Disabled as these test names are too long for CMAKE_OBJECT_PATH_MAX.
# We should integrate these with the other unit tests.
# See issue #1205.
#
return()

file(GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}}"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)

foreach(thrust_target IN LISTS THRUST_TARGETS)
  foreach(test_src IN LISTS test_srcs)
    get_filename_component(test_name "${test_src}" NAME_WLE)
    string(PREPEND test_name "regression.")
    thrust_add_test(test_target ${test_name} "${test_src}" ${thrust_target})
  endforeach()
endforeach()

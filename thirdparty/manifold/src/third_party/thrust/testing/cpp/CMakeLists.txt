file(GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}}"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)

foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  if (NOT config_device STREQUAL "CPP")
    continue()
  endif()

  foreach(test_src IN LISTS test_srcs)
    get_filename_component(test_name "${test_src}" NAME_WLE)
    string(PREPEND test_name "cpp.")
    thrust_add_test(test_target ${test_name} "${test_src}" ${thrust_target})
  endforeach()
endforeach()

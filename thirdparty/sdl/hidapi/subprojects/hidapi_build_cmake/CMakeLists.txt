cmake_minimum_required(VERSION 3.1.3...3.25 FATAL_ERROR)
project(hidapi LANGUAGES C)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/root")

foreach(ROOT_ELEMENT CMakeLists.txt hidapi src windows linux mac libusb pc VERSION)
  file(COPY "${CMAKE_CURRENT_LIST_DIR}/../../${ROOT_ELEMENT}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/root/")
endforeach()

add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/root" hidapi_root)

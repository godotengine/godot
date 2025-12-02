add_executable(ufbxw_unit)

file(GLOB SRC_UNIT *.c *.cpp *.h)

target_link_libraries(ufbxw_unit PRIVATE ufbx_write ufbx ufbxw_extra)
target_sources(ufbxw_unit PRIVATE ${SRC_UNIT})

set_property(TARGET ufbxw_unit PROPERTY CXX_STANDARD 17)

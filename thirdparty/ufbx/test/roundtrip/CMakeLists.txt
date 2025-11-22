add_executable(roundtrip)

file(GLOB SRC_ROUNDTRIP *.c *.h)

target_link_libraries(roundtrip PRIVATE ufbx_write ufbx ufbxw_extra ufbxwt_util compare_fbx)
target_sources(roundtrip PRIVATE ${SRC_ROUNDTRIP})

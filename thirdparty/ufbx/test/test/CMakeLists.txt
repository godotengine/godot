add_executable(ufbxw_test)

file(GLOB SRC_TEST *.c *.h)

target_link_libraries(ufbxw_test PRIVATE ufbx_write ufbx ufbxw_extra ufbxwt_util)

target_sources(ufbxw_test PRIVATE ${SRC_TEST})

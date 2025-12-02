add_library(ufbxwt_util)

file(GLOB SRC_UTIL *.c *.h)

target_link_libraries(ufbxwt_util PRIVATE ufbx ufbxw_extra)
target_sources(ufbxwt_util PRIVATE ${SRC_UTIL})

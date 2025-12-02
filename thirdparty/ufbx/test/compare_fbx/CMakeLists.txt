add_library(compare_fbx)

file(GLOB SRC_COMPARE_FBX *.c *.cpp *.h)

target_link_libraries(compare_fbx PRIVATE ufbx)
target_sources(compare_fbx PRIVATE ${SRC_COMPARE_FBX})


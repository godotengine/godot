
add_executable(c_example c_example.c elfio_c_wrapper.cpp elfio_c_wrapper.h)
target_link_libraries(c_example PRIVATE elfio::elfio)


add_executable(elfio-ldd elfio_ldd.cpp)
target_link_libraries(elfio-ldd PRIVATE elfio::elfio)

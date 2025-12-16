
add_executable(writer writer.cpp)
target_link_libraries(writer PRIVATE elfio::elfio)


add_executable(proc_mem proc_mem.cpp)
target_link_libraries(proc_mem PRIVATE elfio::elfio)


add_executable(elfdump elfdump.cpp)
target_link_libraries(elfdump PRIVATE elfio::elfio)

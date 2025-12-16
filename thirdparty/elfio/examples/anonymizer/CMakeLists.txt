
add_executable(anonymizer anonymizer.cpp)
target_link_libraries(anonymizer PRIVATE elfio::elfio)

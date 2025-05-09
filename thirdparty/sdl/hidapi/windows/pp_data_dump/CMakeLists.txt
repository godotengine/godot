project(pp_data_dump C)

add_executable(pp_data_dump pp_data_dump.c)    
set_target_properties(pp_data_dump
    PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED TRUE
)
target_link_libraries(pp_data_dump
    PRIVATE hidapi_winapi
)

install(TARGETS pp_data_dump
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

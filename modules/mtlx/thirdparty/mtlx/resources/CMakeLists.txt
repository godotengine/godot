if(NOT SKBUILD)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
            DESTINATION "resources" MESSAGE_NEVER)
endif()

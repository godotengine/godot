file(GLOB SOURCES Source/*.cpp)

include_directories(Include)

if(RECASTNAVIGATION_STATIC)
    add_library(Detour STATIC ${SOURCES})
else()
    add_library(Detour SHARED ${SOURCES})
endif()

set_target_properties(Detour PROPERTIES
        SOVERSION ${SOVERSION}
        VERSION ${VERSION}
        )

install(TARGETS Detour
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        COMPONENT library
        )

file(GLOB INCLUDES Include/*.h)
install(FILES ${INCLUDES} DESTINATION include)

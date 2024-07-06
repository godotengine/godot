cmake_minimum_required(VERSION 3.0)

set(CMAKE_INSTALL_PREFIX "install" CACHE STRING "prefix" FORCE)

project(buildSpvHeaders)

set(SOURCES
    main.cpp
    jsonToSpirv.cpp
    header.cpp
    jsoncpp/dist/jsoncpp.cpp)

set(HEADERS
    jsonToSpirv.h
    header.h
    jsoncpp/dist/json/json.h)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_definitions(-std=c++11)
endif()

add_executable(buildSpvHeaders ${SOURCES} ${HEADERS})

install(TARGETS buildSpvHeaders RUNTIME DESTINATION bin)

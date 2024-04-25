file(GLOB_RECURSE source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE objc_source "")
if(APPLE)
    file(GLOB_RECURSE objc_source "${CMAKE_CURRENT_SOURCE_DIR}/*.mm")
endif()

target_sources(MaterialXTest PUBLIC ${source} ${headers} ${objc_source})

add_tests("${source}")

assign_source_group("Source Files" ${source})
assign_source_group("Header Files" ${headers})
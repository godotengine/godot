set(CMAKE_CXX_STANDARD 17)

set(DEAR_IMGUI_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/External/ImGui" CACHE STRING "Path to Dear ImGui")

if (NOT IS_DIRECTORY "${DEAR_IMGUI_PREFIX}/backends")
    message(FATAL_ERROR "Building the MaterialX graph editor requires the ImGui submodule "
        "to be present. Update your repository by calling the following:\n"
        "git submodule update --init --recursive")
endif()

if(MSVC)
    add_compile_options(-wd4100 -wd4152 -wd4201 -wd4244 -wd4456)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-unused -Wno-deprecated -Wno-comment -Wno-unused-variable)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wno-format-truncation -Wno-use-after-free -Wno-comment -Wno-unused-but-set-variable)
endif()

file(GLOB materialx_source "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB materialx_headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h*")

if (APPLE)
    list(APPEND materialx_source ${CMAKE_CURRENT_SOURCE_DIR}/FileDialog.mm)
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/FileDialog.mm" PROPERTIES LANGUAGE CXX)
endif ()

file(GLOB imgui_source "${DEAR_IMGUI_PREFIX}/*.cpp")
file(GLOB imgui_headers "${DEAR_IMGUI_PREFIX}/*.h*")

set(imgui_source ${imgui_source}
    "${DEAR_IMGUI_PREFIX}/backends/imgui_impl_glfw.cpp"
    "${DEAR_IMGUI_PREFIX}/backends/imgui_impl_opengl3.cpp"
    "${DEAR_IMGUI_PREFIX}/misc/cpp/imgui_stdlib.cpp")
set(imgui_headers ${imgui_headers}
    "${DEAR_IMGUI_PREFIX}/backends/imgui_impl_glfw.h"
    "${DEAR_IMGUI_PREFIX}/backends/imgui_impl_opengl3.h"
    "${DEAR_IMGUI_PREFIX}/misc/cpp/imgui_stdlib.h")

file(GLOB imguinode_source "${CMAKE_CURRENT_SOURCE_DIR}/External/ImGuiNodeEditor/*.cpp")
file(GLOB imguinode_headers "${CMAKE_CURRENT_SOURCE_DIR}/External/ImGuiNodeEditor/*.h*")

set(imguinode_source ${imguinode_source}
    "${CMAKE_CURRENT_SOURCE_DIR}/External/ImGuiNodeEditor/examples/blueprints-example/utilities/drawing.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/External/ImGuiNodeEditor/examples/blueprints-example/utilities/widgets.cpp")
set(imguinode_headers ${imguinode_headers}
    "${CMAKE_CURRENT_SOURCE_DIR}/External/ImGuiNodeEditor/examples/blueprints-example/utilities/drawing.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/External/ImGuiNodeEditor/examples/blueprints-example/utilities/widgets.h")

assign_source_group("Source Files" ${materialx_source} ${imgui_source} ${imguinode_source})
assign_source_group("Header Files" ${materialx_headers} ${imgui_headers} ${imguinode_headers})

include_directories("${DEAR_IMGUI_PREFIX}")
include_directories("${DEAR_IMGUI_PREFIX}/backends")
include_directories("${DEAR_IMGUI_PREFIX}/misc/cpp")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/External/ImGuiNodeEditor")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/External/ImGuiNodeEditor/examples/blueprints-example/utilities")

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)

# Explicitly build GLFW with static libraries, independent of the broader build settings.
set(BUILD_SHARED_LIBS OFF)

add_subdirectory(External/Glfw)

if(MSVC)
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup")
endif()

add_executable(MaterialXGraphEditor
    ${materialx_source}
    ${materialx_headers}
    ${imgui_source}
    ${imgui_headers}
    ${imguinode_source}
    ${imguinode_headers})

set(MATERIALX_LIBRARIES
    MaterialXFormat
    MaterialXGenGlsl
    MaterialXRenderGlsl)

if (APPLE)
    find_library(CORE_FOUNDATION Foundation REQUIRED)
    list(APPEND MATERIALX_LIBRARIES ${CORE_FOUNDATION})
endif ()

target_link_libraries(
    MaterialXGraphEditor
    PRIVATE
    ${MATERIALX_LIBRARIES}
 	glfw_minimal)

install(TARGETS MaterialXGraphEditor
    EXPORT MaterialX
    RUNTIME DESTINATION bin)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/MaterialXGraphEditor.pdb"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/" OPTIONAL)

set(CMAKE_CXX_STANDARD 17)

option(MATERIALX_NANOGUI_EXTERNAL "Build aginst an external install of NanoGUI (NANOGUI_ROOT may also need to be set)" OFF)

set(MATERIALXVIEW_RENDER_BACKEND_DEFINITIONS "")

set(NANOGUI_PREFERRED_BACKEND OpenGL)
if(APPLE)
    option(USE_OPENGL_BACKEND_ON_APPLE_PLATFORM "Use OpenGL Backend on Apple Platform" OFF)
    
    # Disables MaterialXView Metal Rendering on MacOS version below 10.14
    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_HOST_SYSTEM_VERSION VERSION_LESS 19)
      set(USE_OPENGL_BACKEND_ON_APPLE_PLATFORM ON)
    endif()
    
    if(USE_OPENGL_BACKEND_ON_APPLE_PLATFORM)
        set(NANOGUI_PREFERRED_BACKEND OpenGL)
	set(MATERIALXVIEW_RENDER_BACKEND_DEFINITIONS "-DMATERIALXVIEW_OPENGL_BACKEND=1")
    else()
        set(NANOGUI_PREFERRED_BACKEND Metal)
	set(MATERIALXVIEW_RENDER_BACKEND_DEFINITIONS "-DMATERIALXVIEW_METAL_BACKEND=1")
    endif()
endif()

if("${MATERIALXVIEW_RENDER_BACKEND_DEFINITIONS}" STREQUAL "")
    set(MATERIALXVIEW_RENDER_BACKEND_DEFINITIONS "-DMATERIALXVIEW_OPENGL_BACKEND=1")
endif()

if (MATERIALX_NANOGUI_EXTERNAL)
    find_path(NANOGUI_INCLUDE_DIRS
        NAMES
            nanogui/nanogui.h
        HINTS
            "${NANOGUI_ROOT}/include"
            "$ENV{NANOGUI_ROOT}/include")
    find_library(NANOGUI_LIBRARIES
        NAMES
            nanogui
        HINTS
            "${NANOGUI_ROOT}/lib"
            "$ENV{NANOGUI_ROOT}/lib"
            "${NANOGUI_ROOT}/lib64"
            "$ENV{NANOGUI_ROOT}/lib64")

    if (NOT NANOGUI_INCLUDE_DIRS OR NOT NANOGUI_LIBRARIES)
        message(FATAL_ERROR "Could not find external NanoGUI installation, is NANOGUI_ROOT set?")
    endif()
else()
    if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/NanoGUI/ext/glfw/src")
        message(FATAL_ERROR "Building the MaterialX viewer requires the NanoGUI submodule "
            "to be present. Update your repository by calling the following:\n"
            "git submodule update --init --recursive")
    endif()

    set(NANOGUI_BACKEND ${NANOGUI_PREFERRED_BACKEND} CACHE STRING " " FORCE)
    set(NANOGUI_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
    set(NANOGUI_BUILD_SHARED OFF CACHE BOOL " " FORCE)
    set(NANOGUI_BUILD_PYTHON OFF CACHE BOOL " " FORCE)
    set(NANOGUI_INSTALL OFF CACHE BOOL " " FORCE)
    
    if(APPLE)
        set(NANOGUI_NATIVE_FLAGS CACHE STRING "" FORCE)
    endif()

    # Locally disable additional warnings for NanoGUI and its dependencies
    set(PREV_CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    set(PREV_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    if(MSVC)
        add_compile_options(-wd4389)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-Wno-deprecated)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-Wno-format-truncation -Wno-use-after-free)
    endif()

    # Disable NanoGUI compiler modifications for Clang
    set(PREV_COMPILER_ID CMAKE_CXX_COMPILER_ID)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE)
        set(CMAKE_CXX_COMPILER_ID "None")
    endif()

    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/NanoGUI external/NanoGUI)
    set_property(TARGET glfw glfw_objects nanogui PROPERTY FOLDER "External")

    set(NANOGUI_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../;${CMAKE_CURRENT_SOURCE_DIR}/NanoGUI/include")
    set(NANOGUI_LIBRARIES "nanogui")

    # Restore warnings for MaterialXView
    set(CMAKE_C_FLAGS ${PREV_CMAKE_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${PREV_CMAKE_CXX_FLAGS})

    # Restore compiler ID for MaterialXView
    set(CMAKE_CXX_COMPILER_ID PREV_COMPILER_ID)
endif()

file(GLOB materialx_source       "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB materialx_headers      "${CMAKE_CURRENT_SOURCE_DIR}/*.h*")

add_definitions(${NANOGUI_EXTRA_DEFS})
add_definitions(${MATERIALXVIEW_RENDER_BACKEND_DEFINITIONS})

set(MATERIALX_LIBRARIES
    MaterialXFormat
    MaterialXGenGlsl
    MaterialXRender
    MaterialXRenderGlsl)
    
if("${NANOGUI_PREFERRED_BACKEND}" STREQUAL "Metal")
    set(MATERIALX_LIBRARIES
        MaterialXFormat
        MaterialXRender
        MaterialXGenMsl
        MaterialXRenderMsl)

    file(GLOB materialx_source 
        "${CMAKE_CURRENT_SOURCE_DIR}/Viewer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ViewerMSL.mm"
        "${CMAKE_CURRENT_SOURCE_DIR}/RenderPipeline.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RenderPipelineMetal.mm"
        "${CMAKE_CURRENT_SOURCE_DIR}/Editor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MetalState.mm"
        "${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp")    
endif()

if (MATERIALX_BUILD_GEN_OSL)
    LIST(APPEND MATERIALX_LIBRARIES MaterialXGenOsl)
endif()
if (MATERIALX_BUILD_GEN_MDL)
    LIST(APPEND MATERIALX_LIBRARIES MaterialXGenMdl)
endif()

add_executable(MaterialXView ${materialx_source} ${materialx_headers})

target_link_libraries(
    MaterialXView
    ${MATERIALX_LIBRARIES}
    ${NANOGUI_LIBRARIES}
    ${NANOGUI_EXTRA_LIBS})

target_include_directories(
    MaterialXView
    PRIVATE
    ${NANOGUI_INCLUDE_DIRS}
    ${NANOGUI_EXTRA_INCS})

if(MATERIALX_BUILD_OIIO AND OPENIMAGEIO_ROOT_DIR)
    add_custom_command(TARGET MaterialXView POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${OPENIMAGEIO_ROOT_DIR}/bin ${CMAKE_BINARY_DIR}/bin)
endif()

set_target_properties(
    MaterialXView PROPERTIES
    INSTALL_RPATH "${MATERIALX_UP_ONE_RPATH}")

install(TARGETS MaterialXView
    EXPORT MaterialX
    RUNTIME DESTINATION bin)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/MaterialXView.pdb"
    DESTINATION "bin" OPTIONAL)

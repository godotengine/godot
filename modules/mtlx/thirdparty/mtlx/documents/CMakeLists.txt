set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(DOXYGEN_HTML_OUTPUT_DIR ${DOXYGEN_OUTPUT_DIR}/html)
set(DOXYGEN_INPUT_LIST ${CMAKE_SOURCE_DIR}/documents/DeveloperGuide/MainPage.md
                       ${CMAKE_SOURCE_DIR}/source/MaterialXCore
                       ${CMAKE_SOURCE_DIR}/source/MaterialXFormat
                       ${CMAKE_SOURCE_DIR}/source/MaterialXGenShader
                       ${CMAKE_SOURCE_DIR}/source/MaterialXGenShader/Nodes
                       ${CMAKE_SOURCE_DIR}/source/MaterialXGenGlsl
                       ${CMAKE_SOURCE_DIR}/source/MaterialXGenGlsl/Nodes
                       ${CMAKE_SOURCE_DIR}/source/MaterialXGenOsl
                       ${CMAKE_SOURCE_DIR}/source/MaterialXGenMdl
                       ${CMAKE_SOURCE_DIR}/source/MaterialXRender
                       ${CMAKE_SOURCE_DIR}/source/MaterialXRenderHw
                       ${CMAKE_SOURCE_DIR}/source/MaterialXRenderGlsl
                       ${CMAKE_SOURCE_DIR}/source/MaterialXRenderOsl
                       )
string (REPLACE ";" " " DOXYGEN_INPUT_STR "${DOXYGEN_INPUT_LIST}")

find_package(Doxygen)

if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    add_custom_target(MaterialXDocs ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      COMMENT "Generating HTML documentation: ${DOXYGEN_HTML_OUTPUT_DIR}/index.html")
    add_custom_command(TARGET MaterialXDocs PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/documents/DoxygenAwesome ${CMAKE_CURRENT_BINARY_DIR})
    add_custom_command(TARGET MaterialXDocs PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/documents/Images ${CMAKE_CURRENT_BINARY_DIR})
    install(DIRECTORY ${DOXYGEN_HTML_OUTPUT_DIR}
            DESTINATION "documents" MESSAGE_NEVER)
endif(DOXYGEN_FOUND)

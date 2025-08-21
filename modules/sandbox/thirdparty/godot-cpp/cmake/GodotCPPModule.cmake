#[=======================================================================[.rst:
GodotCPPModule.cmake
---------------------

This file contains functions and tests which may be needed by consumers.

* Generate Trimmed API
* Generate File List
* Generate Bindings

If you want to use these functions in your project extend the CMAKE_MODULE_PATH
by adding these two lines into your CMakeLists.txt after the inclusion
godot-cpp

.. highlight:: cmake

    list(APPEND CMAKE_MODULE_PATH "${godot-cpp_SOURCE_DIR}/cmake")
    include( GodotCPPModule )

]=======================================================================]
find_package(Python3 3.4 REQUIRED) # pathlib should be present

#[[ Generate Trimmed API

The build_profile.py has a __main__ and is used as a tool
Its usage is listed as:
    $ python build_profile.py BUILD_PROFILE INPUT_JSON [OUTPUT_JSON]
]]
function(build_profile_generate_trimmed_api BUILD_PROFILE INPUT_JSON OUTPUT_JSON)
    execute_process(
        COMMAND
            "${Python3_EXECUTABLE}" "${godot-cpp_SOURCE_DIR}/build_profile.py" "${BUILD_PROFILE}" "${INPUT_JSON}"
            "${OUTPUT_JSON}"
        WORKING_DIRECTORY ${godot-cpp_SOURCE_DIR}
    )
endfunction()

#[[ Generate File List

Use the binding_generator.py Python script to determine the list of files that
will be passed to the code generator using extension_api.json.
NOTE: This happens for every configure.]]
function(binding_generator_get_file_list OUT_VAR_NAME API_FILEPATH OUTPUT_DIR)
    # This code snippet will be squashed into a single line
    # The two strings make this a list, in CMake lists are semicolon delimited strings.
    set(PYTHON_SCRIPT
        "from binding_generator import print_file_list"
        "print_file_list( api_filepath='${API_FILEPATH}',
    output_dir='${OUTPUT_DIR}',
    headers=True,
    sources=True)"
    )
    message(DEBUG "Python:\n${PYTHON_SCRIPT}")

    # Strip newlines and whitespace to make it a one-liner.
    string(REGEX REPLACE "\n *" " " PYTHON_SCRIPT "${PYTHON_SCRIPT}")

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "-c" "${PYTHON_SCRIPT}"
        WORKING_DIRECTORY "${godot-cpp_SOURCE_DIR}"
        OUTPUT_VARIABLE GENERATED_FILES_LIST
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Debug output
    message(DEBUG "FileList-Begin")
    foreach(PATH ${GENERATED_FILES_LIST})
        message(DEBUG ${PATH})
    endforeach()

    # Error out if the file list generator returned no files.
    list(LENGTH GENERATED_FILES_LIST LIST_LENGTH)
    if(NOT LIST_LENGTH GREATER 0)
        message(FATAL_ERROR "File List Generation Failed")
    endif()
    message(STATUS "There are ${LIST_LENGTH} Files to generate")

    set(${OUT_VAR_NAME} ${GENERATED_FILES_LIST} PARENT_SCOPE)
endfunction()

#[[ Generate Bindings

Using the generated file list, use the binding_generator.py to generate the
godot-cpp bindings. This will run at build time only if there are files
missing. ]]
function(
    binding_generator_generate_bindings
    API_FILE
    USE_TEMPLATE_GET_NODE
    BITS
    PRECISION
    OUTPUT_DIR
)
    # This code snippet will be squashed into a single line
    set(PYTHON_SCRIPT
        "from binding_generator import generate_bindings"
        "generate_bindings(
    api_filepath='${API_FILE}',
    use_template_get_node='${USE_TEMPLATE_GET_NODE}',
    bits='${BITS}',
    precision='${PRECISION}',
    output_dir='${OUTPUT_DIR}')"
    )

    message(DEBUG "Python:\n${PYTHON_SCRIPT}")

    # Strip newlines and whitespace to make it a one-liner.
    string(REGEX REPLACE "\n *" " " PYTHON_SCRIPT "${PYTHON_SCRIPT}")

    add_custom_command(
        OUTPUT ${GENERATED_FILES_LIST}
        COMMAND "${Python3_EXECUTABLE}" "-c" "${PYTHON_SCRIPT}"
        VERBATIM
        WORKING_DIRECTORY ${godot-cpp_SOURCE_DIR}
        MAIN_DEPENDENCY ${GODOTCPP_GDEXTENSION_API_FILE}
        DEPENDS ${godot-cpp_SOURCE_DIR}/binding_generator.py
        COMMENT "Generating bindings"
    )
    add_custom_target(generate_bindings DEPENDS ${GENERATED_FILES_LIST})
    set_target_properties(generate_bindings PROPERTIES FOLDER "godot-cpp")
endfunction()

#[[ Generate doc_data.cpp
The documentation displayed in the Godot editor is compiled into the extension.
It takes a list of XML source files, and transforms them into a cpp file that
is added to the sources list.]]
function(generate_doc_source OUTPUT_PATH SOURCES)
    # Transform SOURCES CMake LIST
    # quote each path with ''
    # join with , to transform into a python list minus the surrounding []
    set(PYTHON_LIST "${SOURCES}")
    list(TRANSFORM PYTHON_LIST REPLACE "(.*\.xml)" "'\\1'")
    list(JOIN PYTHON_LIST "," PYTHON_LIST)

    get_filename_component(OUTPUT_DIR "${OUTPUT_PATH}" DIRECTORY)
    file(MAKE_DIRECTORY ${OUTPUT_DIR})

    # Python one-liner to run our command
    # lists in CMake are just strings delimited by ';', so this works.
    set(PYTHON_SCRIPT
        "from doc_source_generator import generate_doc_source"
        "generate_doc_source( '${OUTPUT_PATH}', [${PYTHON_LIST}] )"
    )

    add_custom_command(
        OUTPUT "${OUTPUT_PATH}"
        COMMAND "${Python3_EXECUTABLE}" "-c" "${PYTHON_SCRIPT}"
        VERBATIM
        WORKING_DIRECTORY "${godot-cpp_SOURCE_DIR}"
        DEPENDS #
            "${godot-cpp_SOURCE_DIR}/doc_source_generator.py"
            "${SOURCES}"
        COMMENT "Generating: ${OUTPUT_PATH}"
    )
    add_custom_target(generate_doc_source DEPENDS "${OUTPUT_PATH}")
    set_target_properties(generate_doc_source PROPERTIES FOLDER "godot-cpp")
endfunction()

#[[ target_doc_sources
A simpler interface to add xml files as doc source to a output target.
TARGET: The gdexension library target
SOURCES: a list of xml files to use for source generation and inclusion.]]
function(target_doc_sources TARGET SOURCES)
    # set the generated file name
    set(DOC_SOURCE_FILE "${CMAKE_CURRENT_BINARY_DIR}/gen/doc_source.cpp")

    # Create the file generation target, this won't be triggered unless a target
    # that depends on DOC_SOURCE_FILE is built
    generate_doc_source( "${DOC_SOURCE_FILE}" ${SOURCES} )

    # Add DOC_SOURCE_FILE as a dependency to TARGET
    target_sources(${TARGET} PRIVATE "${DOC_SOURCE_FILE}")

    # Without adding this dependency to the doc_source_generator, XCode will complain.
    add_dependencies(${TARGET} generate_doc_source)
endfunction()

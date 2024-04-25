if (MATERIALX_OSL_LEGACY_CLOSURES)
    set(PBRLIB_SUFFIX "legacy")
else()
    set(PBRLIB_SUFFIX "mtlx")
endif()

if(NOT SKBUILD)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
            DESTINATION "${MATERIALX_INSTALL_STDLIB_PATH}"
            PATTERN "CMakeLists.txt" EXCLUDE
            PATTERN "pbrlib_genosl_impl.*" EXCLUDE)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/pbrlib/genosl/pbrlib_genosl_impl.${PBRLIB_SUFFIX}"
            DESTINATION "${MATERIALX_INSTALL_STDLIB_PATH}/pbrlib/genosl/" RENAME pbrlib_genosl_impl.mtlx)
endif()

set(MATERIALX_PYTHON_LIBRARIES_PATH "${MATERIALX_PYTHON_FOLDER_NAME}/${MATERIALX_INSTALL_STDLIB_PATH}")
if(SKBUILD)
    set(MATERIALX_PYTHON_LIBRARIES_PATH "${SKBUILD_PLATLIB_DIR}/MaterialX/libraries")
endif()

if(MATERIALX_BUILD_PYTHON)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
            DESTINATION "${MATERIALX_PYTHON_LIBRARIES_PATH}"
            PATTERN "CMakeLists.txt" EXCLUDE
            PATTERN "pbrlib_genosl_impl.*" EXCLUDE)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/pbrlib/genosl/pbrlib_genosl_impl.${PBRLIB_SUFFIX}"
            DESTINATION "${MATERIALX_PYTHON_LIBRARIES_PATH}/pbrlib/genosl/" RENAME pbrlib_genosl_impl.mtlx)
endif()

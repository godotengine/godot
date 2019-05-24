# Suppress all warnings from external projects.
set_property(DIRECTORY APPEND PROPERTY COMPILE_OPTIONS -w)

if(BUILD_TESTING)
    if(TARGET gmock)
        message(STATUS "Google Mock already configured - use it")
    elseif(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/googletest)
        # We need to make sure Google Test does not mess up with the
        # global CRT settings on Windows.
        if(WIN32)
            set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        endif(WIN32)
        # EXCLUDE_FROM_ALL keeps the install target from installing GTEST files.
        add_subdirectory(googletest EXCLUDE_FROM_ALL)
        set(GTEST_TARGETS
            gtest
            gtest_main
            gmock
            gmock_main)
        foreach(target ${GTEST_TARGETS})
            set_property(TARGET ${target} PROPERTY FOLDER gtest)
        endforeach()
        mark_as_advanced(gmock_build_tests
            BUILD_GMOCK
            BUILD_GTEST
            BUILD_SHARED_LIBS
            gtest_build_samples
            gtest_build_tests
            gtest_disable_pthreads
            gtest_force_shared_crt
            gtest_hide_internal_symbols)
    else()
        message(STATUS
            "Google Mock was not found - tests based on that will not build")
    endif()
endif()

if(ENABLE_OPT AND NOT TARGET SPIRV-Tools-opt)
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/spirv-tools)
        set(SPIRV_SKIP_TESTS ON CACHE BOOL "Skip building SPIRV-Tools tests")
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/spirv-tools spirv-tools)
    endif()
endif()


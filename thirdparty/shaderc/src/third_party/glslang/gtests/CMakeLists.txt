if(BUILD_TESTING)
    if(TARGET gmock)
        message(STATUS "Google Mock found - building tests")

        set(TEST_SOURCES
            # Framework related source files
            ${CMAKE_CURRENT_SOURCE_DIR}/Initializer.h
            ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/Settings.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/Settings.h
            ${CMAKE_CURRENT_SOURCE_DIR}/TestFixture.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/TestFixture.h

            # Test related source files
            ${CMAKE_CURRENT_SOURCE_DIR}/AST.FromFile.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/BuiltInResource.FromFile.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/Config.FromFile.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/HexFloat.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/Hlsl.FromFile.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/Link.FromFile.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/Link.FromFile.Vk.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/Pp.FromFile.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/Spv.FromFile.cpp

            # -- Remapper tests
            ${CMAKE_CURRENT_SOURCE_DIR}/Remap.FromFile.cpp)

        glslang_pch(TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/pch.cpp)

        add_executable(glslangtests ${TEST_SOURCES})
        set_property(TARGET glslangtests PROPERTY FOLDER tests)
        glslang_set_link_args(glslangtests)
        if(ENABLE_GLSLANG_INSTALL)
            install(TARGETS glslangtests
                    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
        endif(ENABLE_GLSLANG_INSTALL)

        set(GLSLANG_TEST_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../Test")
        # Supply a default test root directory, so that manual testing
        # doesn't have to specify the --test-root option in the normal
        # case that you want to use the tests from the same source tree.
        target_compile_definitions(glslangtests
                                   PRIVATE GLSLANG_TEST_DIRECTORY="${GLSLANG_TEST_DIRECTORY}")
        target_include_directories(glslangtests PRIVATE
                                   ${CMAKE_CURRENT_SOURCE_DIR}
                                   ${PROJECT_SOURCE_DIR}
                                   ${gmock_SOURCE_DIR}/include
                                   ${gtest_SOURCE_DIR}/include)

        set(LIBRARIES
            SPVRemapper glslang OSDependent OGLCompiler glslang
            SPIRV glslang-default-resource-limits)
        if(ENABLE_HLSL)
            set(LIBRARIES ${LIBRARIES} HLSL)
        endif(ENABLE_HLSL)
        target_link_libraries(glslangtests PRIVATE ${LIBRARIES} gmock)

        add_test(NAME glslang-gtests
                 COMMAND glslangtests --test-root "${GLSLANG_TEST_DIRECTORY}")
    endif()
endif()

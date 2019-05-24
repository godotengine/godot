add_library(OSDependent STATIC ossource.cpp ../osinclude.h)
set_property(TARGET OSDependent PROPERTY FOLDER glslang)
set_property(TARGET OSDependent PROPERTY POSITION_INDEPENDENT_CODE ON)

# Link pthread
set(CMAKE_THREAD_PREFER_PTHREAD ON)
if(${CMAKE_VERSION} VERSION_LESS "3.1.0" OR CMAKE_CROSSCOMPILING)
    # Needed as long as we support CMake 2.8 for Ubuntu 14.04,
    # which does not support the recommended Threads::Threads target.
    # https://cmake.org/cmake/help/v2.8.12/cmake.html#module:FindThreads
    # Also needed when cross-compiling to work around
    # https://gitlab.kitware.com/cmake/cmake/issues/16920
    find_package(Threads)
    target_link_libraries(OSDependent ${CMAKE_THREAD_LIBS_INIT})
else()
    # This is the recommended way, so we use it for 3.1+.
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads)
    target_link_libraries(OSDependent Threads::Threads)
endif()

if(ENABLE_GLSLANG_INSTALL)
    install(TARGETS OSDependent
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(ENABLE_GLSLANG_INSTALL)

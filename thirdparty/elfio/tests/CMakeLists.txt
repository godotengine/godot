include(FetchContent)

if(${CMAKE_VERSION} VERSION_LESS "3.24.0")
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/530d5c8c84.zip
    )
else()
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
        FIND_PACKAGE_ARGS NAMES GTest
        DOWNLOAD_EXTRACT_TIMESTAMP = TRUE
    )
endif()

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

# Find all the binary files used for testing and copy them into the build
# directory. This allows the test to be run from the build directory

# First, create an elf_examples folder under the current build directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/elf_examples)

# Second, glob all files under elf_examples
file(GLOB elf_examples
    LIST_DIRECTORIES
    false
    CONFIGURE_DEPENDS
    elf_examples/*)

# Third, copy each file globbed to the elf_examples folder under the current
# build directory
foreach(example ${elf_examples})
    configure_file(${example} ${CMAKE_CURRENT_BINARY_DIR}/elf_examples COPYONLY)
endforeach()

# First, create an ario folder under the current build directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ario)

# Second, glob all files under ario
file(GLOB ario
    LIST_DIRECTORIES
    false
    CONFIGURE_DEPENDS
    ario/*)

# Third, copy each file globbed to the ario folder under the current
# build directory
foreach(example ${ario})
    configure_file(${example} ${CMAKE_CURRENT_BINARY_DIR}/ario COPYONLY)
endforeach()


# Lastly, copy the script to run the tests
configure_file(runELFtests ${CMAKE_CURRENT_BINARY_DIR}/runELFtests COPYONLY)

add_executable(
    ELFIOTest
    ELFIOTest.cpp
    ELFIOTest1.cpp
    ELFIOTest2.cpp
    ARIOTest.cpp)

target_link_libraries(
    ELFIOTest
    PRIVATE
    elfio::elfio
    ario::ario
    gtest_main
    GTest::gtest_main)

add_test(
    NAME
    ELFIOTest
    COMMAND
    ${CMAKE_CURRENT_BINARY_DIR}/ELFIOTest
    WORKING_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR})

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_executable(
        elfio_fuzzer
        elfio_fuzzer.cpp)

    target_link_libraries(
        elfio_fuzzer
        PRIVATE
        elfio::elfio)

    target_compile_options(elfio_fuzzer
        PRIVATE $<$<C_COMPILER_ID:Clang>:-g -O1 -fsanitize=fuzzer,address>
    )

    target_link_libraries(elfio_fuzzer
        PRIVATE $<$<C_COMPILER_ID:Clang>:-fsanitize=fuzzer,address>
    )
endif()

add_dependencies(check ELFIOTest)

include(GoogleTest)
gtest_discover_tests(ELFIOTest)

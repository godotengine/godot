include(FetchContent)

add_library(ufbxw_extra)

option(USE_LIBDEFLATE "Enable libdeflate" ON)
option(USE_ZLIB "Enable Zlib" ON)
option(USE_FMTLIB "Enable fmtlib" ON)
option(USE_TO_CHARS "Enable to_chars" ON)
option(USE_CPP_THREADS "Enable C++ threads" ON)

set(SRC_EXTRA "")

if(USE_LIBDEFLATE)
    FetchContent_Declare(
        libdeflate
        GIT_REPOSITORY "https://github.com/ebiggers/libdeflate.git"
        EXCLUDE_FROM_ALL
    )

    set(LIBDEFLATE_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
    set(LIBDEFLATE_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
    set(LIBDEFLATE_BUILD_GZIP OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(libdeflate)

    target_link_libraries(ufbxw_extra PRIVATE libdeflate_static)
    target_compile_definitions(ufbxw_extra PUBLIC UFBXWT_HAS_LIBDEFLATE=1)

    list(APPEND SRC_EXTRA "../../extra/ufbxw_libdeflate.h" "ufbxw_libdeflate.c")
endif()

if(USE_ZLIB)
    FetchContent_Declare(
        zlib
        GIT_REPOSITORY "https://github.com/madler/zlib.git"
        EXCLUDE_FROM_ALL
    )

    set(ZLIB_BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(ZLIB_BUILD_MINIZIP OFF CACHE BOOL "" FORCE)
    set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(zlib)

    target_link_libraries(ufbxw_extra PRIVATE zlibstatic)
    target_compile_definitions(ufbxw_extra PUBLIC UFBXWT_HAS_ZLIB=1)

    list(APPEND SRC_EXTRA "../../extra/ufbxw_zlib.h" "ufbxw_zlib.c")
endif()

if(USE_FMTLIB)
    FetchContent_Declare(
        fmtlib
        GIT_REPOSITORY "https://github.com/fmtlib/fmt"
        EXCLUDE_FROM_ALL
    )

    FetchContent_MakeAvailable(fmtlib)

    target_link_libraries(ufbxw_extra PRIVATE fmt)
    target_compile_definitions(ufbxw_extra PUBLIC UFBXWT_HAS_FMTLIB=1)

    list(APPEND SRC_EXTRA "../../extra/ufbxw_fmtlib.h" "ufbxw_fmtlib.cpp")
endif()

if(USE_TO_CHARS)
    target_compile_definitions(ufbxw_extra PUBLIC UFBXWT_HAS_TO_CHARS=1)

    list(APPEND SRC_EXTRA "../../extra/ufbxw_to_chars.h" "ufbxw_to_chars.cpp")
endif()

if(USE_CPP_THREADS)
    target_compile_definitions(ufbxw_extra PUBLIC UFBXWT_HAS_CPP_THREADS=1)

    list(APPEND SRC_EXTRA "../../extra/ufbxw_cpp_threads.h" "ufbxw_cpp_threads.cpp")
endif()

target_sources(ufbxw_extra PRIVATE ${SRC_EXTRA})

target_link_libraries(ufbxw_extra PRIVATE ufbx_write)

set_property(TARGET ufbxw_extra PROPERTY CXX_STANDARD 17)

option(GLM_QUIET "No CMake Message" OFF)
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(BUILD_STATIC_LIBS "Build static library" ON)
option(GLM_TEST_ENABLE_CXX_98 "Enable C++ 98" OFF)
option(GLM_TEST_ENABLE_CXX_11 "Enable C++ 11" OFF)
option(GLM_TEST_ENABLE_CXX_14 "Enable C++ 14" OFF)
option(GLM_TEST_ENABLE_CXX_17 "Enable C++ 17" OFF)
option(GLM_TEST_ENABLE_CXX_20 "Enable C++ 20" OFF)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(GLM_TEST_ENABLE_CXX_20)
	set(CMAKE_CXX_STANDARD 20)
	add_definitions(-DGLM_FORCE_CXX2A)
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with C++20 features")
	endif()

elseif(GLM_TEST_ENABLE_CXX_17)
	set(CMAKE_CXX_STANDARD 17)
	add_definitions(-DGLM_FORCE_CXX17)
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with C++17 features")
	endif()

elseif(GLM_TEST_ENABLE_CXX_14)
	set(CMAKE_CXX_STANDARD 14)
	add_definitions(-DGLM_FORCE_CXX14)
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with C++14 features")
	endif()

elseif(GLM_TEST_ENABLE_CXX_11)
	set(CMAKE_CXX_STANDARD 11)
	add_definitions(-DGLM_FORCE_CXX11)
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with C++11 features")
	endif()

elseif(GLM_TEST_ENABLE_CXX_98)
	set(CMAKE_CXX_STANDARD 98)
	add_definitions(-DGLM_FORCE_CXX98)
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with C++98 features")
	endif()
endif()

option(GLM_TEST_ENABLE_LANG_EXTENSIONS "Enable language extensions" OFF)

option(GLM_DISABLE_AUTO_DETECTION "Enable language extensions" OFF)

if(GLM_DISABLE_AUTO_DETECTION)
	add_definitions(-DGLM_FORCE_PLATFORM_UNKNOWN -DGLM_FORCE_COMPILER_UNKNOWN -DGLM_FORCE_ARCH_UNKNOWN -DGLM_FORCE_CXX_UNKNOWN)
endif()

if(GLM_TEST_ENABLE_LANG_EXTENSIONS)
	set(CMAKE_CXX_EXTENSIONS ON)
	if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
		add_compile_options(-fms-extensions)
	endif()
	message(STATUS "GLM: Build with C++ language extensions")
else()
	set(CMAKE_CXX_EXTENSIONS OFF)
	if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/Za)
		if(MSVC15)
			add_compile_options(/permissive-)
		endif()
	endif()
endif()

option(GLM_TEST_ENABLE_FAST_MATH "Enable fast math optimizations" OFF)
if(GLM_TEST_ENABLE_FAST_MATH)
	if(NOT GLM_QUIET)
		message(STATUS "GLM: Build with fast math optimizations")
	endif()

	if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
		add_compile_options(-ffast-math)

	elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/fp:fast)
	endif()
else()
	if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/fp:precise)
	endif()
endif()

option(GLM_TEST_ENABLE "Build unit tests" ON)
option(GLM_TEST_ENABLE_SIMD_SSE2 "Enable SSE2 optimizations" OFF)
option(GLM_TEST_ENABLE_SIMD_SSE3 "Enable SSE3 optimizations" OFF)
option(GLM_TEST_ENABLE_SIMD_SSSE3 "Enable SSSE3 optimizations" OFF)
option(GLM_TEST_ENABLE_SIMD_SSE4_1 "Enable SSE 4.1 optimizations" OFF)
option(GLM_TEST_ENABLE_SIMD_SSE4_2 "Enable SSE 4.2 optimizations" OFF)
option(GLM_TEST_ENABLE_SIMD_AVX "Enable AVX optimizations" OFF)
option(GLM_TEST_ENABLE_SIMD_AVX2 "Enable AVX2 optimizations" OFF)
option(GLM_TEST_FORCE_PURE "Force 'pure' instructions" OFF)

if(GLM_TEST_FORCE_PURE)
	add_definitions(-DGLM_FORCE_PURE)

	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
		add_compile_options(-mfpmath=387)
	endif()
	message(STATUS "GLM: No SIMD instruction set")

elseif(GLM_TEST_ENABLE_SIMD_AVX2)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-mavx2)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxAVX2)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/arch:AVX2)
	endif()
	message(STATUS "GLM: AVX2 instruction set")

elseif(GLM_TEST_ENABLE_SIMD_AVX)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-mavx)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxAVX)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/arch:AVX)
	endif()
	message(STATUS "GLM: AVX instruction set")

elseif(GLM_TEST_ENABLE_SIMD_SSE4_2)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-msse4.2)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxSSE4.2)
	elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
		add_compile_options(/arch:SSE2) # VC doesn't support SSE4.2
	endif()
	message(STATUS "GLM: SSE4.2 instruction set")

elseif(GLM_TEST_ENABLE_SIMD_SSE4_1)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-msse4.1)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxSSE4.1)
	elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
		add_compile_options(/arch:SSE2) # VC doesn't support SSE4.1
	endif()
	message(STATUS "GLM: SSE4.1 instruction set")

elseif(GLM_TEST_ENABLE_SIMD_SSSE3)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-mssse3)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxSSSE3)
	elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
		add_compile_options(/arch:SSE2) # VC doesn't support SSSE3
	endif()
	message(STATUS "GLM: SSSE3 instruction set")

elseif(GLM_TEST_ENABLE_SIMD_SSE3)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-msse3)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxSSE3)
	elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
		add_compile_options(/arch:SSE2) # VC doesn't support SSE3
	endif()
	message(STATUS "GLM: SSE3 instruction set")

elseif(GLM_TEST_ENABLE_SIMD_SSE2)
	add_definitions(-DGLM_FORCE_INTRINSICS)

	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-msse2)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxSSE2)
	elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
		add_compile_options(/arch:SSE2)
	endif()
	message(STATUS "GLM: SSE2 instruction set")
endif()

# Compiler and default options

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	if(NOT GLM_QUIET)
		message("GLM: Clang - ${CMAKE_CXX_COMPILER_ID} compiler")
	endif()

	add_compile_options(-Werror -Weverything)
	add_compile_options(-Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-c++11-long-long -Wno-padded -Wno-gnu-anonymous-struct -Wno-nested-anon-types)
	add_compile_options(-Wno-undefined-reinterpret-cast -Wno-sign-conversion -Wno-unused-variable -Wno-missing-prototypes -Wno-unreachable-code -Wno-missing-variable-declarations -Wno-sign-compare -Wno-global-constructors -Wno-unused-macros -Wno-format-nonliteral)

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	if(NOT GLM_QUIET)
		message("GLM: GCC - ${CMAKE_CXX_COMPILER_ID} compiler")
	endif()

	add_compile_options(-O2)
	add_compile_options(-Wno-long-long)

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
	if(NOT GLM_QUIET)
		message("GLM: Intel - ${CMAKE_CXX_COMPILER_ID} compiler")
	endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	if(NOT GLM_QUIET)
		message("GLM: Visual C++ - ${CMAKE_CXX_COMPILER_ID} compiler")
	endif()

	add_compile_options(/W4 /WX)
	add_compile_options(/wd4309 /wd4324 /wd4389 /wd4127 /wd4267 /wd4146 /wd4201 /wd4464 /wd4514 /wd4701 /wd4820 /wd4365)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

function(glmCreateTestGTC NAME)
	set(SAMPLE_NAME test-${NAME})
	add_executable(${SAMPLE_NAME} ${NAME}.cpp)

	add_test(
		NAME ${SAMPLE_NAME}
		COMMAND $<TARGET_FILE:${SAMPLE_NAME}> )
	target_link_libraries(${SAMPLE_NAME} PRIVATE glm::glm)
endfunction()

if(GLM_TEST_ENABLE)
	add_subdirectory(bug)
	add_subdirectory(core)
	add_subdirectory(ext)
	add_subdirectory(gtc)
	add_subdirectory(gtx)
	add_subdirectory(perf)
endif()



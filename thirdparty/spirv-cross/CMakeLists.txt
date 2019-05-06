# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
project(SPIRV-Cross LANGUAGES CXX C)
enable_testing()

option(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS "Instead of throwing exceptions assert" OFF)
option(SPIRV_CROSS_SHARED "Build the C API as a single shared library." OFF)
option(SPIRV_CROSS_STATIC "Build the C and C++ API as static libraries." ON)
option(SPIRV_CROSS_CLI "Build the CLI binary. Requires SPIRV_CROSS_STATIC." ON)
option(SPIRV_CROSS_ENABLE_TESTS "Enable SPIRV-Cross tests." ON)

option(SPIRV_CROSS_ENABLE_GLSL "Enable GLSL support." ON)
option(SPIRV_CROSS_ENABLE_HLSL "Enable HLSL target support." ON)
option(SPIRV_CROSS_ENABLE_MSL "Enable MSL target support." ON)
option(SPIRV_CROSS_ENABLE_CPP "Enable C++ target support." ON)
option(SPIRV_CROSS_ENABLE_REFLECT "Enable JSON reflection target support." ON)
option(SPIRV_CROSS_ENABLE_C_API "Enable C API wrapper support in static library." ON)
option(SPIRV_CROSS_ENABLE_UTIL "Enable util module support." ON)

option(SPIRV_CROSS_SANITIZE_ADDRESS "Sanitize address" OFF)
option(SPIRV_CROSS_SANITIZE_MEMORY "Sanitize memory" OFF)
option(SPIRV_CROSS_SANITIZE_THREADS "Sanitize threads" OFF)
option(SPIRV_CROSS_SANITIZE_UNDEFINED "Sanitize undefined" OFF)

option(SPIRV_CROSS_NAMESPACE_OVERRIDE "" "Override the namespace used in the C++ API.")
option(SPIRV_CROSS_FORCE_STL_TYPES "Force use of STL types instead of STL replacements in certain places. Might reduce performance." OFF)

if(${CMAKE_GENERATOR} MATCHES "Makefile")
	if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
		message(FATAL_ERROR "Build out of tree to avoid overwriting Makefile")
	endif()
endif()

set(spirv-compiler-options "")
set(spirv-compiler-defines "")
set(spirv-cross-link-flags "")

if(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS)
	set(spirv-compiler-defines ${spirv-compiler-defines} SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS)
endif()

if(SPIRV_CROSS_FORCE_STL_TYPES)
	set(spirv-compiler-defines ${spirv-compiler-defines} SPIRV_CROSS_FORCE_STL_TYPES)
endif()

if (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
	set(spirv-compiler-options ${spirv-compiler-options} -Wall -Wextra -Werror -Wshadow)

	if (SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS)
		set(spirv-compiler-options ${spirv-compiler-options} -fno-exceptions)
	endif()

	if (SPIRV_CROSS_SANITIZE_ADDRESS)
		set(spirv-compiler-options ${spirv-compiler-options} -fsanitize=address)
		set(spirv-cross-link-flags "${spirv-cross-link-flags} -fsanitize=address")
	endif()

	if (SPIRV_CROSS_SANITIZE_UNDEFINED)
		set(spirv-compiler-options ${spirv-compiler-options} -fsanitize=undefined)
		set(spirv-cross-link-flags "${spirv-cross-link-flags} -fsanitize=undefined")
	endif()

	if (SPIRV_CROSS_SANITIZE_MEMORY)
		set(spirv-compiler-options ${spirv-compiler-options} -fsanitize=memory)
		set(spirv-cross-link-flags "${spirv-cross-link-flags} -fsanitize=memory")
	endif()

	if (SPIRV_CROSS_SANITIZE_THREADS)
		set(spirv-compiler-options ${spirv-compiler-options} -fsanitize=thread)
		set(spirv-cross-link-flags "${spirv-cross-link-flags} -fsanitize=thread")
	endif()
elseif (MSVC)
	set(spirv-compiler-options ${spirv-compiler-options} /wd4267)
endif()

macro(extract_headers out_abs file_list)
	set(${out_abs}) # absolute paths
	foreach(_a ${file_list})
		# get_filename_component only returns the longest extension, so use a regex
		string(REGEX REPLACE ".*\\.(h|hpp)" "\\1" ext ${_a})

		# For shared library, we are only interested in the C header.
		if (SPIRV_CROSS_STATIC)
			if(("${ext}" STREQUAL "h") OR ("${ext}" STREQUAL "hpp"))
				list(APPEND ${out_abs} "${_a}")
			endif()
		else()
			if("${ext}" STREQUAL "h")
				list(APPEND ${out_abs} "${_a}")
			endif()
		endif()
	endforeach()
endmacro()

macro(spirv_cross_add_library name config_name library_type)
	add_library(${name} ${library_type} ${ARGN})
	extract_headers(hdrs "${ARGN}")
	target_include_directories(${name} PUBLIC
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
			$<INSTALL_INTERFACE:include/spirv_cross>)
	set_target_properties(${name} PROPERTIES
			PUBLIC_HEADERS "${hdrs}")
	target_compile_options(${name} PRIVATE ${spirv-compiler-options})
	target_compile_definitions(${name} PRIVATE ${spirv-compiler-defines})
	if (SPIRV_CROSS_NAMESPACE_OVERRIDE)
		if (${library_type} MATCHES "STATIC")
			target_compile_definitions(${name} PUBLIC SPIRV_CROSS_NAMESPACE_OVERRIDE=${SPIRV_CROSS_NAMESPACE_OVERRIDE})
		else()
			target_compile_definitions(${name} PRIVATE SPIRV_CROSS_NAMESPACE_OVERRIDE=${SPIRV_CROSS_NAMESPACE_OVERRIDE})
		endif()
	endif()
	install(TARGETS ${name}
			EXPORT ${config_name}Config
			RUNTIME DESTINATION bin
			LIBRARY DESTINATION lib
			ARCHIVE DESTINATION lib
			PUBLIC_HEADER DESTINATION include/spirv_cross)
	install(FILES ${hdrs} DESTINATION include/spirv_cross)
	install(EXPORT ${config_name}Config DESTINATION share/${config_name}/cmake)
	export(TARGETS ${name} FILE ${config_name}Config.cmake)
endmacro()

set(spirv-cross-core-sources
		${CMAKE_CURRENT_SOURCE_DIR}/GLSL.std.450.h
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_common.hpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cross_containers.hpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cross_error_handling.hpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv.hpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cross.hpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cross.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_parser.hpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_parser.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cross_parsed_ir.hpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cross_parsed_ir.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cfg.hpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cfg.cpp)

set(spirv-cross-c-sources
		spirv.h
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cross_c.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cross_c.h)

set(spirv-cross-glsl-sources
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_glsl.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_glsl.hpp)

set(spirv-cross-cpp-sources
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cpp.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cpp.hpp)

set(spirv-cross-msl-sources
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_msl.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_msl.hpp)

set(spirv-cross-hlsl-sources
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_hlsl.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_hlsl.hpp)

set(spirv-cross-reflect-sources
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_reflect.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_reflect.hpp)

set(spirv-cross-util-sources
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cross_util.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/spirv_cross_util.hpp)

if (SPIRV_CROSS_STATIC)
	spirv_cross_add_library(spirv-cross-core spirv_cross_core STATIC
			${spirv-cross-core-sources})

	if (SPIRV_CROSS_ENABLE_GLSL)
		spirv_cross_add_library(spirv-cross-glsl spirv_cross_glsl STATIC
				${spirv-cross-glsl-sources})
		target_link_libraries(spirv-cross-glsl PRIVATE spirv-cross-core)
	endif()

	if (SPIRV_CROSS_ENABLE_CPP)
		spirv_cross_add_library(spirv-cross-cpp spirv_cross_cpp STATIC
				${spirv-cross-cpp-sources})

		if (SPIRV_CROSS_ENABLE_GLSL)
			target_link_libraries(spirv-cross-cpp PRIVATE spirv-cross-glsl)
		else()
			message(FATAL_ERROR "Must enable GLSL support to enable C++ support.")
		endif()
	endif()

	if (SPIRV_CROSS_ENABLE_REFLECT)
		if (SPIRV_CROSS_ENABLE_GLSL)
			spirv_cross_add_library(spirv-cross-reflect spirv_cross_reflect STATIC
					${spirv-cross-reflect-sources})
		else()
			message(FATAL_ERROR "Must enable GLSL support to enable JSON reflection support.")
		endif()
	endif()

	if (SPIRV_CROSS_ENABLE_MSL)
		spirv_cross_add_library(spirv-cross-msl spirv_cross_msl STATIC
				${spirv-cross-msl-sources})
		if (SPIRV_CROSS_ENABLE_GLSL)
			target_link_libraries(spirv-cross-msl PRIVATE spirv-cross-glsl)
		else()
			message(FATAL_ERROR "Must enable GLSL support to enable MSL support.")
		endif()
	endif()

	if (SPIRV_CROSS_ENABLE_HLSL)
		spirv_cross_add_library(spirv-cross-hlsl spirv_cross_hlsl STATIC
				${spirv-cross-hlsl-sources})
		if (SPIRV_CROSS_ENABLE_GLSL)
			target_link_libraries(spirv-cross-hlsl PRIVATE spirv-cross-glsl)
		else()
			message(FATAL_ERROR "Must enable GLSL support to enable HLSL support.")
		endif()
	endif()

	if (SPIRV_CROSS_ENABLE_UTIL)
		spirv_cross_add_library(spirv-cross-util spirv_cross_util STATIC
				${spirv-cross-util-sources})
		target_link_libraries(spirv-cross-util PRIVATE spirv-cross-core)
	endif()

	if (SPIRV_CROSS_ENABLE_C_API)
		spirv_cross_add_library(spirv-cross-c spirv_cross_c STATIC
				${spirv-cross-c-sources})

		if (SPIRV_CROSS_ENABLE_GLSL)
			target_link_libraries(spirv-cross-c PRIVATE spirv-cross-glsl)
			target_compile_definitions(spirv-cross-c PRIVATE SPIRV_CROSS_C_API_GLSL=1)
		endif()

		if (SPIRV_CROSS_ENABLE_HLSL)
			target_link_libraries(spirv-cross-c PRIVATE spirv-cross-hlsl)
			target_compile_definitions(spirv-cross-c PRIVATE SPIRV_CROSS_C_API_HLSL=1)
		endif()

		if (SPIRV_CROSS_ENABLE_MSL)
			target_link_libraries(spirv-cross-c PRIVATE spirv-cross-msl)
			target_compile_definitions(spirv-cross-c PRIVATE SPIRV_CROSS_C_API_MSL=1)
		endif()

		if (SPIRV_CROSS_ENABLE_CPP)
			target_link_libraries(spirv-cross-c PRIVATE spirv-cross-cpp)
			target_compile_definitions(spirv-cross-c PRIVATE SPIRV_CROSS_C_API_CPP=1)
		endif()

		if (SPIRV_CROSS_ENABLE_REFLECT)
			target_link_libraries(spirv-cross-c PRIVATE spirv-cross-reflect)
			target_compile_definitions(spirv-cross-c PRIVATE SPIRV_CROSS_C_API_REFLECT=1)
		endif()
	endif()
endif()

set(spirv-cross-abi-major 0)
set(spirv-cross-abi-minor 9)
set(spirv-cross-abi-patch 0)

if (SPIRV_CROSS_SHARED)
	set(SPIRV_CROSS_VERSION ${spirv-cross-abi-major}.${spirv-cross-abi-minor}.${spirv-cross-abi-patch})
	set(SPIRV_CROSS_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
	set(SPIRV_CROSS_INSTALL_INC_DIR ${CMAKE_INSTALL_PREFIX}/include/spirv_cross)
	configure_file(
			${CMAKE_CURRENT_SOURCE_DIR}/pkg-config/spirv-cross-c-shared.pc.in
			${CMAKE_CURRENT_BINARY_DIR}/spirv-cross-c-shared.pc @ONLY)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/spirv-cross-c-shared.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)

	spirv_cross_add_library(spirv-cross-c-shared spirv_cross_c_shared SHARED
			${spirv-cross-core-sources}
			${spirv-cross-c-sources})

	if (SPIRV_CROSS_ENABLE_GLSL)
		target_sources(spirv-cross-c-shared PRIVATE ${spirv-cross-glsl-sources})
		target_compile_definitions(spirv-cross-c-shared PRIVATE SPIRV_CROSS_C_API_GLSL=1)
	endif()

	if (SPIRV_CROSS_ENABLE_HLSL)
		if (SPIRV_CROSS_ENABLE_GLSL)
			target_sources(spirv-cross-c-shared PRIVATE ${spirv-cross-hlsl-sources})
		else()
			message(FATAL_ERROR "Must enable GLSL support to enable HLSL support.")
		endif()
		target_compile_definitions(spirv-cross-c-shared PRIVATE SPIRV_CROSS_C_API_HLSL=1)
	endif()

	if (SPIRV_CROSS_ENABLE_MSL)
		if (SPIRV_CROSS_ENABLE_GLSL)
			target_sources(spirv-cross-c-shared PRIVATE ${spirv-cross-msl-sources})
		else()
			message(FATAL_ERROR "Must enable GLSL support to enable MSL support.")
		endif()
		target_compile_definitions(spirv-cross-c-shared PRIVATE SPIRV_CROSS_C_API_MSL=1)
	endif()

	if (SPIRV_CROSS_ENABLE_CPP)
		if (SPIRV_CROSS_ENABLE_GLSL)
			target_sources(spirv-cross-c-shared PRIVATE ${spirv-cross-cpp-sources})
		else()
			message(FATAL_ERROR "Must enable GLSL support to enable C++ support.")
		endif()
		target_compile_definitions(spirv-cross-c-shared PRIVATE SPIRV_CROSS_C_API_CPP=1)
	endif()

	if (SPIRV_CROSS_ENABLE_REFLECT)
		if (SPIRV_CROSS_ENABLE_GLSL)
			target_sources(spirv-cross-c-shared PRIVATE ${spirv-cross-reflect-sources})
		else()
			message(FATAL_ERROR "Must enable GLSL support to enable JSON reflection support.")
		endif()
		target_compile_definitions(spirv-cross-c-shared PRIVATE SPIRV_CROSS_C_API_REFLECT=1)
	endif()

	if (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
		# Only export the C API.
		target_compile_options(spirv-cross-c-shared PRIVATE -fvisibility=hidden)
		if (NOT APPLE)
			set_target_properties(spirv-cross-c-shared PROPERTIES LINK_FLAGS "${spirv-cross-link-flags}")
		endif()
	endif()

	target_compile_definitions(spirv-cross-c-shared PRIVATE SPVC_EXPORT_SYMBOLS)

	set_target_properties(spirv-cross-c-shared PROPERTIES
			VERSION ${SPIRV_CROSS_VERSION}
			SOVERSION ${spirv-cross-abi-major})
endif()

if (SPIRV_CROSS_CLI)
	if (NOT SPIRV_CROSS_ENABLE_GLSL)
		message(FATAL_ERROR "Must enable GLSL if building CLI.")
	endif()

	if (NOT SPIRV_CROSS_ENABLE_HLSL)
		message(FATAL_ERROR "Must enable HLSL if building CLI.")
	endif()

	if (NOT SPIRV_CROSS_ENABLE_MSL)
		message(FATAL_ERROR "Must enable MSL if building CLI.")
	endif()

	if (NOT SPIRV_CROSS_ENABLE_CPP)
		message(FATAL_ERROR "Must enable C++ if building CLI.")
	endif()

	if (NOT SPIRV_CROSS_ENABLE_REFLECT)
		message(FATAL_ERROR "Must enable reflection if building CLI.")
	endif()

	if (NOT SPIRV_CROSS_ENABLE_UTIL)
		message(FATAL_ERROR "Must enable utils if building CLI.")
	endif()

	if (NOT SPIRV_CROSS_STATIC)
		message(FATAL_ERROR "Must build static libraries if building CLI.")
	endif()
	add_executable(spirv-cross main.cpp)
	target_compile_options(spirv-cross PRIVATE ${spirv-compiler-options})
	target_compile_definitions(spirv-cross PRIVATE ${spirv-compiler-defines})
	set_target_properties(spirv-cross PROPERTIES LINK_FLAGS "${spirv-cross-link-flags}")
	install(TARGETS spirv-cross RUNTIME DESTINATION bin)
	target_link_libraries(spirv-cross PRIVATE
			spirv-cross-glsl
			spirv-cross-hlsl
			spirv-cross-cpp
			spirv-cross-reflect
			spirv-cross-msl
			spirv-cross-util
			spirv-cross-core)

	if (SPIRV_CROSS_ENABLE_TESTS)
		# Set up tests, using only the simplest modes of the test_shaders
		# script.  You have to invoke the script manually to:
		#  - Update the reference files
		#  - Get cycle counts from malisc
		#  - Keep failing outputs
		find_package(PythonInterp)
		find_program(spirv-cross-glslang NAMES glslangValidator
				PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/glslang-build/output/bin
				NO_DEFAULT_PATH)
		find_program(spirv-cross-spirv-as NAMES spirv-as
				PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/spirv-tools-build/output/bin
				NO_DEFAULT_PATH)
		find_program(spirv-cross-spirv-val NAMES spirv-val
				PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/spirv-tools-build/output/bin
				NO_DEFAULT_PATH)
		find_program(spirv-cross-spirv-opt NAMES spirv-opt
				PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/spirv-tools-build/output/bin
				NO_DEFAULT_PATH)

		if ((${spirv-cross-glslang} MATCHES "NOTFOUND") OR (${spirv-cross-spirv-as} MATCHES "NOTFOUND") OR (${spirv-cross-spirv-val} MATCHES "NOTFOUND") OR (${spirv-cross-spirv-opt} MATCHES "NOTFOUND"))
			set(SPIRV_CROSS_ENABLE_TESTS OFF)
			message("Could not find glslang or SPIRV-Tools build under external/. Run ./checkout_glslang_spirv_tools.sh and ./build_glslang_spirv_tools.sh. Testing will be disabled.")
		else()
			set(SPIRV_CROSS_ENABLE_TESTS ON)
			message("Found glslang and SPIRV-Tools. Enabling test suite.")
			message("Found glslangValidator in: ${spirv-cross-glslang}.")
			message("Found spirv-as in: ${spirv-cross-spirv-as}.")
			message("Found spirv-val in: ${spirv-cross-spirv-val}.")
			message("Found spirv-opt in: ${spirv-cross-spirv-opt}.")
		endif()

		set(spirv-cross-externals
				--glslang "${spirv-cross-glslang}"
				--spirv-as "${spirv-cross-spirv-as}"
				--spirv-opt "${spirv-cross-spirv-opt}"
				--spirv-val "${spirv-cross-spirv-val}")

		if (${PYTHONINTERP_FOUND} AND SPIRV_CROSS_ENABLE_TESTS)
			if (${PYTHON_VERSION_MAJOR} GREATER 2)
				add_executable(spirv-cross-c-api-test tests-other/c_api_test.c)
				target_link_libraries(spirv-cross-c-api-test spirv-cross-c)
				set_target_properties(spirv-cross-c-api-test PROPERTIES LINK_FLAGS "${spirv-cross-link-flags}")

				add_executable(spirv-cross-small-vector-test tests-other/small_vector.cpp)
				target_link_libraries(spirv-cross-small-vector-test spirv-cross-core)
				set_target_properties(spirv-cross-small-vector-test PROPERTIES LINK_FLAGS "${spirv-cross-link-flags}")

				if (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
					target_compile_options(spirv-cross-c-api-test PRIVATE -std=c89 -Wall -Wextra)
				endif()
				add_test(NAME spirv-cross-c-api-test
						COMMAND $<TARGET_FILE:spirv-cross-c-api-test> ${CMAKE_CURRENT_SOURCE_DIR}/tests-other/c_api_test.spv
						${spirv-cross-abi-major}
						${spirv-cross-abi-minor}
						${spirv-cross-abi-patch})
				add_test(NAME spirv-cross-small-vector-test
						COMMAND $<TARGET_FILE:spirv-cross-small-vector-test>)
				add_test(NAME spirv-cross-test
						COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_shaders.py --parallel
						${spirv-cross-externals}
						${CMAKE_CURRENT_SOURCE_DIR}/shaders
						WORKING_DIRECTORY $<TARGET_FILE_DIR:spirv-cross>)
				add_test(NAME spirv-cross-test-no-opt
						COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_shaders.py --parallel
						${spirv-cross-externals}
						${CMAKE_CURRENT_SOURCE_DIR}/shaders-no-opt
						WORKING_DIRECTORY $<TARGET_FILE_DIR:spirv-cross>)
				add_test(NAME spirv-cross-test-metal
						COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_shaders.py --metal --parallel
						${spirv-cross-externals}
						${CMAKE_CURRENT_SOURCE_DIR}/shaders-msl
						WORKING_DIRECTORY $<TARGET_FILE_DIR:spirv-cross>)
				add_test(NAME spirv-cross-test-metal-no-opt
						COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_shaders.py --metal --parallel
						${spirv-cross-externals}
						${CMAKE_CURRENT_SOURCE_DIR}/shaders-msl-no-opt
						WORKING_DIRECTORY $<TARGET_FILE_DIR:spirv-cross>)
				add_test(NAME spirv-cross-test-hlsl
						COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_shaders.py --hlsl --parallel
						${spirv-cross-externals}
						${CMAKE_CURRENT_SOURCE_DIR}/shaders-hlsl
						WORKING_DIRECTORY $<TARGET_FILE_DIR:spirv-cross>)
				add_test(NAME spirv-cross-test-hlsl-no-opt
						COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_shaders.py --hlsl --parallel
						${spirv-cross-externals}
						${CMAKE_CURRENT_SOURCE_DIR}/shaders-hlsl-no-opt
						WORKING_DIRECTORY $<TARGET_FILE_DIR:spirv-cross>)
				add_test(NAME spirv-cross-test-opt
						COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_shaders.py --opt --parallel
						${spirv-cross-externals}
						${CMAKE_CURRENT_SOURCE_DIR}/shaders
						WORKING_DIRECTORY $<TARGET_FILE_DIR:spirv-cross>)
				add_test(NAME spirv-cross-test-metal-opt
						COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_shaders.py --metal --opt --parallel
						${spirv-cross-externals}
						${CMAKE_CURRENT_SOURCE_DIR}/shaders-msl
						WORKING_DIRECTORY $<TARGET_FILE_DIR:spirv-cross>)
				add_test(NAME spirv-cross-test-hlsl-opt
						COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_shaders.py --hlsl --opt --parallel
						${spirv-cross-externals}
						${CMAKE_CURRENT_SOURCE_DIR}/shaders-hlsl
						WORKING_DIRECTORY $<TARGET_FILE_DIR:spirv-cross>)
				add_test(NAME spirv-cross-test-reflection
						COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_shaders.py --reflect --parallel
						${spirv-cross-externals}
						${CMAKE_CURRENT_SOURCE_DIR}/shaders-reflection
						WORKING_DIRECTORY $<TARGET_FILE_DIR:spirv-cross>)
			endif()
		elseif(NOT ${PYTHONINTERP_FOUND})
			message(WARNING "Testing disabled. Could not find python3. If you have python3 installed try running "
					"cmake with -DPYTHON_EXECUTABLE:FILEPATH=/path/to/python3 to help it find the executable")
		endif()
	endif()
endif()

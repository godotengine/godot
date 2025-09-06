/**************************************************************************/
/*  resource_saver_cpp.cpp                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "resource_saver_cpp.h"
#include "../elf/script_elf.h"
#include "../elf/script_language_elf.h"
#include "../register_types.h"
#include "../sandbox.h"
#include "../sandbox_project_settings.h"
#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "core/object/script_language.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"
#include "script_cpp.h"
#include <libriscv/util/threadpool.h>

static Ref<ResourceFormatSaverCPP> cpp_saver;
static std::unique_ptr<riscv::ThreadPool> thread_pool;

static const char cmake_toolchain_bytes[] = R"(
if (NOT DEFINED ZIG_PATH)
	set(ZIG_PATH "zig")
endif()
set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR "riscv64")
set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_C_COMPILER "${ZIG_PATH}" cc -target riscv64-linux-musl)
set(CMAKE_CXX_COMPILER "${ZIG_PATH}" c++ -target riscv64-linux-musl)

if (CMAKE_HOST_WIN32)
	# Windows: Disable .d files
	set(CMAKE_C_LINKER_DEPFILE_SUPPORTED FALSE)
	set(CMAKE_CXX_LINKER_DEPFILE_SUPPORTED FALSE)
	# Windows: Work-around for zig ar and zig ranlib
	set(CMAKE_AR "${CMAKE_CURRENT_LIST_DIR}/zig-ar.cmd")
	set(CMAKE_RANLIB "${CMAKE_CURRENT_LIST_DIR}/zig-ranlib.cmd")
endif()
if (CMAKE_HOST_APPLE)
	set(CMAKE_AR "${CMAKE_CURRENT_LIST_DIR}/zig-ar.cmd")
	set(CMAKE_RANLIB "${CMAKE_CURRENT_LIST_DIR}/zig-ranlib.cmd")
endif()
)";

#if defined(_WIN32) || defined(__APPLE__)
static const char cmake_zig_ar_bytes[] = R"(
@echo off
zig ar %*
)";
static const char cmake_zig_ranlib_bytes[] = R"(
@echo off
zig ranlib %*
)";
#endif

static const char cmake_cmakelists_bytes[] = R"(
cmake_minimum_required(VERSION 3.10)
project(example LANGUAGES CXX)

# Fetch godot-sandbox C++ API
include(FetchContent)
FetchContent_Declare(
	godot-sandbox
	GIT_REPOSITORY https://github.com/libriscv/godot-sandbox.git
	GIT_TAG        main
	GIT_SHALLOW    TRUE
	GIT_SUBMODULES ""
	SOURCE_SUBDIR  "program/cpp/cmake"
)
FetchContent_MakeAvailable(godot-sandbox)

# Put an example.cpp in a src folder and CMake
# will create example.elf in the same folder with this:
add_sandbox_program_at(example.elf ../src
	../src/example.cpp
)
)";

void ResourceFormatSaverCPP::init() {
	thread_pool = std::make_unique<riscv::ThreadPool>(1); // Maximum 1 compiler job at a time
	cpp_saver.instantiate();
	// Register the CPPScript resource saver
	ResourceSaver::add_resource_format_saver(cpp_saver);
}

void ResourceFormatSaverCPP::deinit() {
	// Stop the thread pool
	thread_pool.reset();
	// Unregister the CPPScript resource saver
	ResourceSaver::remove_resource_format_saver(cpp_saver);
	cpp_saver.unref();
}

static void auto_generate_cpp_api(const String &path) {
	static bool api_written_to_project_root = false;
	if (!api_written_to_project_root) {
		// Check if the run-time API should be generated
		if (!SandboxProjectSettings::generate_runtime_api()) {
			api_written_to_project_root = true;
			return;
		}
		// Write the API to the project root
		Ref<FileAccess> api_handle = FileAccess::open(path, FileAccess::ModeFlags::WRITE);
		if (api_handle.is_valid()) {
			const bool use_argument_names = SandboxProjectSettings::generate_method_arguments();
			api_handle->store_string(Sandbox::generate_api("cpp", "", use_argument_names));
			api_handle->close();
		}
		api_written_to_project_root = true;
	}
}

static bool configure_cmake(const String &path) {
	OS *os = OS::get_singleton();
	// Configure cmake to generate the build files
	// Example: cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRIPPED=ON -DCMAKE_TOOLCHAIN_FILE=%HERE%\toolchain.cmake
	Ref<DirAccess> dir_access = DirAccess::open(path);
	if (!dir_access.is_valid()) {
		ERR_PRINT("Failed to open directory: " + path);
		return false;
	}
	// Create the .build directory if it does not exist
	if (!dir_access->dir_exists(".build")) {
		Error err = dir_access->make_dir(".build");
		if (err != Error::OK) {
			ERR_PRINT("Failed to create .build directory: " + path);
			return false;
		}
	}

	const String runtime_api_path = path + String("/.build/generated_api.hpp");
	if (!FileAccess::exists(runtime_api_path)) {
		// Generate the C++ run-time API in the .build directory
		// This will be used by C++ programs to access the wider Godot API
		auto_generate_cpp_api(runtime_api_path);
	}

	// Create the CMakeLists.txt file if it does not exist
	const String cmakelists_path = path + String("/CMakeLists.txt");
	if (!FileAccess::exists(cmakelists_path)) {
		Ref<FileAccess> cmakelists_file = FileAccess::open(cmakelists_path, FileAccess::ModeFlags::WRITE);
		if (cmakelists_file.is_valid()) {
			cmakelists_file->store_string(cmake_cmakelists_bytes);
			cmakelists_file->close();
		} else {
			ERR_PRINT("Failed to create CMakeLists.txt file: " + cmakelists_path);
			return false;
		}
	}

	// Execute cmake to configure the project
	// We assume that zig, cmake and git are available in the PATH
	// TODO: Verify that zig, cmake and git are available in the PATH?
	const String toolchain_path = path + String("/toolchain.cmake");
	if (!FileAccess::exists(toolchain_path)) {
		// Create the toolchain file
		Ref<FileAccess> toolchain_file = FileAccess::open(toolchain_path, FileAccess::ModeFlags::WRITE);
		if (toolchain_file.is_valid()) {
			toolchain_file->store_string(cmake_toolchain_bytes);
			toolchain_file->close();
		} else {
			ERR_PRINT("Failed to create toolchain file: " + toolchain_path);
			return false;
		}
	}
#if defined(_WIN32) || defined(__APPLE__)
	// Create the zig-ar.cmd file, if it does not exist
	const String zig_ar_path = path + String("/zig-ar.cmd");
	if (!FileAccess::exists(zig_ar_path)) {
		Ref<FileAccess> zig_ar_file = FileAccess::open(zig_ar_path, FileAccess::ModeFlags::WRITE);
		if (zig_ar_file.is_valid()) {
			zig_ar_file->store_string(cmake_zig_ar_bytes);
			zig_ar_file->close();
		} else {
			ERR_PRINT("Failed to create zig-ar.cmd file: " + zig_ar_path);
			return false;
		}
	}
	// Create the zig-ranlib.cmd file, if it does not exist
	const String zig_ranlib_path = path + String("/zig-ranlib.cmd");
	if (!FileAccess::exists(zig_ranlib_path)) {
		Ref<FileAccess> zig_ranlib_file = FileAccess::open(zig_ranlib_path, FileAccess::ModeFlags::WRITE);
		if (zig_ranlib_file.is_valid()) {
			zig_ranlib_file->store_string(cmake_zig_ranlib_bytes);
			zig_ranlib_file->close();
		} else {
			ERR_PRINT("Failed to create zig-ranlib.cmd file: " + zig_ranlib_path);
			return false;
		}
	}
#endif

	// Create toolchain absolute path
	const String toolchain_path_absolute = ProjectSettings::get_singleton()->globalize_path("res://") + toolchain_path;

	PackedStringArray arguments;
	arguments.push_back(SandboxProjectSettings::get_cmake_path()); // CMake executable
	arguments.push_back(path); // CMake directory
	arguments.push_back("-B");
	arguments.push_back(path + String("/.build")); // Build directory
#ifdef _WIN32
	arguments.push_back("-GNinja"); // Use Ninja as the build system
#endif
	arguments.push_back("-DCMAKE_BUILD_TYPE=Release");
	arguments.push_back("-DSTRIPPED=OFF");
	arguments.push_back("-DCMAKE_TOOLCHAIN_FILE=" + toolchain_path_absolute);
	// Zig path
	const String zig_path = SandboxProjectSettings::get_zig_path();
	if (!zig_path.is_empty()) {
		arguments.push_back("-DZIG_PATH=" + zig_path + "");
	}
	//arguments.push_back("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON");

	if (true) {
		String args_str = "CMake arguments: ";
		for (int i = 0; i < arguments.size(); i++) {
			args_str += arguments[i];
			if (i < arguments.size() - 1)
				args_str += " ";
		}
		print_line(args_str);
	}

	// Convert to List<String> for OS::execute
	List<String> args_list;
	for (int i = 0; i < arguments.size(); i++) {
		args_list.push_back(arguments[i]);
	}

	String output;
	int exit_code;
	Error error = os->execute(SandboxProjectSettings::get_cmake_path(), args_list, &output, &exit_code);
	if (error != OK || exit_code != 0) {
		if (!output.is_empty()) {
			Vector<String> lines = output.split("\n");
			for (int i = 0; i < lines.size(); i++) {
				print_error(lines[i]);
			}
		}
		ERR_PRINT("Failed to configure cmake: " + itos(exit_code));
		return false;
	}
	print_line("CMake configured successfully in: " + path);
	return true;
}

static Array invoke_cmake(const String &path) {
	Ref<DirAccess> dir_access = DirAccess::open(path);
	if (!dir_access.is_valid()) {
		ERR_PRINT("Failed to open directory: " + path);
		return Array();
	}
	// Check if path/.build exists, if not, configure CMake
	if (!dir_access->dir_exists(".build") ||
			(!dir_access->file_exists(".build/build.ninja") && !dir_access->file_exists(".build/Makefile"))) {
		// Configure cmake to generate the build files
		if (!configure_cmake(path)) {
			ERR_PRINT("Failed to configure cmake in: " + path);
			return Array();
		}
	}

	// Invoke cmake to build the project
	PackedStringArray arguments;
	arguments.push_back("--build");
	arguments.push_back(String(path) + "/.build"); // Build directory
	arguments.push_back("-j");
	arguments.push_back(itos(OS::get_singleton()->get_processor_count()));

	OS *os = OS::get_singleton();

	String args_str = "Invoking cmake: ";
	for (int i = 0; i < arguments.size(); i++) {
		args_str += arguments[i];
		if (i < arguments.size() - 1)
			args_str += " ";
	}
	print_line(args_str);

	// Convert to List<String> for OS::execute
	List<String> args_list;
	for (int i = 0; i < arguments.size(); i++) {
		args_list.push_back(arguments[i]);
	}

	String output_str;
	int exit_code;
	Error error = os->execute(SandboxProjectSettings::get_cmake_path(), args_list, &output_str, &exit_code);

	Array output;
	if (!output_str.is_empty()) {
		output.push_back(output_str);
	}

	if (error != OK || exit_code != 0) {
		if (!output_str.is_empty()) {
			Vector<String> lines = output_str.split("\n");
			for (int i = 0; i < lines.size(); i++) {
				print_error(lines[i]);
			}
		}
		ERR_PRINT("Failed to invoke cmake: " + itos(exit_code));
	}
	return output;
}

static bool detect_and_build_cmake_project_instead() {
	// If the project root contains a CMakeLists.txt file, or a cmake/CMakeLists.txt,
	// build the project using CMake
	// Get the project root using res://
	String project_root = "res://";

	// Check for CMakeLists.txt in the project root
	const bool cmake_root = FileAccess::exists(project_root + "CMakeLists.txt");
	if (cmake_root) {
		(void)invoke_cmake(".");
		// Always return true, as this indicates that the project is built using CMake
		return true;
	}
	const bool cmake_dir = FileAccess::exists(project_root + "cmake/CMakeLists.txt");
	if (cmake_dir) {
		(void)invoke_cmake("./cmake");
		// Always return true, as this indicates that the project is built using CMake
		return true;
	}
	return false;
}

static Array invoke_scons(const String &path) {
	// Invoke scons to build the project
	PackedStringArray arguments;
	// TODO get arguments from project settings

	OS *os = OS::get_singleton();

	String args_str = "Invoking scons: ";
	for (int i = 0; i < arguments.size(); i++) {
		args_str += arguments[i];
		if (i < arguments.size() - 1)
			args_str += " ";
	}
	print_line(args_str);

	// Convert to List<String> for OS::execute
	List<String> args_list;
	for (int i = 0; i < arguments.size(); i++) {
		args_list.push_back(arguments[i]);
	}

	String output_str;
	int exit_code;
	Error error = os->execute(SandboxProjectSettings::get_scons_path(), args_list, &output_str, &exit_code);

	Array output;
	if (!output_str.is_empty()) {
		output.push_back(output_str);
	}

	if (error != OK || exit_code != 0) {
		if (!output_str.is_empty()) {
			Vector<String> lines = output_str.split("\n");
			for (int i = 0; i < lines.size(); i++) {
				print_error(lines[i]);
			}
		}
		ERR_PRINT("Failed to invoke scons: " + itos(exit_code));
	}
	return output;
}

static bool detect_and_build_scons_project_instead() {
	// If the project root contains a SConstruct file,
	// build the project using SConstruct
	// Get the project root using res://
	String project_root = "res://";

	// Check for SConstruct in the project root
	const bool scons_root = FileAccess::exists(project_root + "SConstruct");
	if (scons_root) {
		(void)invoke_scons(".");
		// Always return true, as this indicates that the project is built using SConstruct
		return true;
	}
	return false;
}

Error ResourceFormatSaverCPP::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	CPPScript *cpp_script = Object::cast_to<CPPScript>(p_resource.ptr());
	if (cpp_script != nullptr) {
		Ref<FileAccess> handle = FileAccess::open(p_path, FileAccess::ModeFlags::WRITE);
		if (handle.is_valid()) {
			handle->store_string(cpp_script->get_source_code());
			handle->close();

			if (CPPScript::DetectCMakeOrSConsProject()) {
				// Check if the project is a CMake project
				if (detect_and_build_cmake_project_instead()) {
					return Error::OK;
				}
				// Check if the project is a SCons project
				if (detect_and_build_scons_project_instead()) {
					return Error::OK;
				}
			}

			// Generate the C++ run-time API in the project root
			auto_generate_cpp_api("res://generated_api.hpp");

			// Docker compilation support has been removed
			// Compilation should now be handled by CMake or SCons projects
			print_line("C++ script saved - compilation now handled by CMake/SCons projects only");
			return Error::OK;
		} else {
			return Error::ERR_FILE_CANT_OPEN;
		}
	}
	return Error::ERR_SCRIPT_FAILED;
}
Error ResourceFormatSaverCPP::set_uid(const String &p_path, ResourceUID::ID p_uid) {
	return Error::OK;
}
bool ResourceFormatSaverCPP::recognize(const Ref<Resource> &p_resource) const {
	return Object::cast_to<CPPScript>(p_resource.ptr()) != nullptr;
}
void ResourceFormatSaverCPP::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	if (Object::cast_to<CPPScript>(p_resource.ptr()) == nullptr)
		return;
	p_extensions->push_back("cpp");
	p_extensions->push_back("cc");
	p_extensions->push_back("hh");
	p_extensions->push_back("h");
	p_extensions->push_back("hpp");
}
bool ResourceFormatSaverCPP::recognize_path(const Ref<Resource> &p_resource, const String &p_path) const {
	return Object::cast_to<CPPScript>(p_resource.ptr()) != nullptr;
}

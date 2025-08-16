#include "resource_saver_cpp.h"
#include "../elf/script_elf.h"
#include "../elf/script_language_elf.h"
#include "../register_types.h"
#include "../sandbox.h"
#include "../sandbox_project_settings.h"
#include "script_cpp.h"
#include <libriscv/util/threadpool.h>
#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/editor_file_system.hpp>
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/editor_settings.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/script.hpp>
#include <godot_cpp/classes/script_editor.hpp>
#include <godot_cpp/classes/script_editor_base.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

static Ref<ResourceFormatSaverCPP> cpp_saver;
static std::unique_ptr<riscv::ThreadPool> thread_pool;
static constexpr bool VERBOSE_CMD = false;

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
static const char cmake_zig_ar_bytes[] = R"(
@echo off
zig ar %*
)";
static const char cmake_zig_ranlib_bytes[] = R"(
@echo off
zig ranlib %*
)";
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
	ResourceSaver::get_singleton()->add_resource_format_saver(cpp_saver);
}

void ResourceFormatSaverCPP::deinit() {
	// Stop the thread pool
	thread_pool.reset();
	// Unregister the CPPScript resource saver
	ResourceSaver::get_singleton()->remove_resource_format_saver(cpp_saver);
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
	if (!FileAccess::file_exists(runtime_api_path)) {
		// Generate the C++ run-time API in the .build directory
		// This will be used by C++ programs to access the wider Godot API
		auto_generate_cpp_api(runtime_api_path);
	}

	// Create the CMakeLists.txt file if it does not exist
	const String cmakelists_path = path + String("/CMakeLists.txt");
	if (!FileAccess::file_exists(cmakelists_path)) {
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
	if (!FileAccess::file_exists(toolchain_path)) {
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
	if (!FileAccess::file_exists(zig_ar_path)) {
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
	if (!FileAccess::file_exists(zig_ranlib_path)) {
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
		UtilityFunctions::print("CMake arguments: ", arguments);
	}
	Array output;
	int32_t result = os->execute(SandboxProjectSettings::get_cmake_path(), arguments, output, true);
	if (result != 0) {
		if (!output.is_empty()) {
			output = output[0].operator String().split("\n");
			for (int i = 0; i < output.size(); i++) {
				String line = output[i].operator String();
				UtilityFunctions::printerr(line);
			}
		}
		ERR_PRINT("Failed to configure cmake: " + itos(result));
		return false;
	}
	UtilityFunctions::print("CMake configured successfully in: ", path);
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
		(!dir_access->file_exists(".build/build.ninja")
		&& !dir_access->file_exists(".build/Makefile"))) {
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
	UtilityFunctions::print("Invoking cmake: ", arguments);
	Array output;
	int32_t result = os->execute(SandboxProjectSettings::get_cmake_path(), arguments, output, true);

	if (result != 0) {
		if (!output.is_empty()) {
			output = output[0].operator String().split("\n");
			for (int i = 0; i < output.size(); i++) {
				String line = output[i].operator String();
				UtilityFunctions::printerr(line);
			}
		}
		ERR_PRINT("Failed to invoke cmake: " + itos(result));
	}
	return output;
}

static bool detect_and_build_cmake_project_instead() {
	// If the project root contains a CMakeLists.txt file, or a cmake/CMakeLists.txt,
	// build the project using CMake
	// Get the project root using res://
	String project_root = "res://";

	// Check for CMakeLists.txt in the project root
	const bool cmake_root = FileAccess::file_exists(project_root + "CMakeLists.txt");
	if (cmake_root) {
		(void)invoke_cmake(".");
		// Always return true, as this indicates that the project is built using CMake
		return true;
	}
	const bool cmake_dir = FileAccess::file_exists(project_root + "cmake/CMakeLists.txt");
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
	UtilityFunctions::print("Invoking scons: ", arguments);
	Array output;
	int32_t result = os->execute(SandboxProjectSettings::get_scons_path(), arguments, output, true);

	if (result != 0) {
		if (!output.is_empty()) {
			output = output[0].operator String().split("\n");
			for (int i = 0; i < output.size(); i++) {
				String line = output[i].operator String();
				UtilityFunctions::printerr(line);
			}
		}
		ERR_PRINT("Failed to invoke scons: " + itos(result));
	}
	return output;
}

static bool detect_and_build_scons_project_instead() {
	// If the project root contains a SConstruct file,
	// build the project using SConstruct
	// Get the project root using res://
	String project_root = "res://";

	// Check for SConstruct in the project root
	const bool scons_root = FileAccess::file_exists(project_root + "SConstruct");
	if (scons_root) {
		(void)invoke_scons(".");
		// Always return true, as this indicates that the project is built using SConstruct
		return true;
	}
	return false;
}

Error ResourceFormatSaverCPP::_save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	CPPScript *script = Object::cast_to<CPPScript>(p_resource.ptr());
	if (script != nullptr) {
		Ref<FileAccess> handle = FileAccess::open(p_path, FileAccess::ModeFlags::WRITE);
		if (handle.is_valid()) {
			handle->store_string(script->_get_source_code());
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

			// Get the absolute path without the file name
			String path = handle->get_path().get_base_dir().replace("res://", "") + "/";
			String inpname = path + "*.cpp";
			String foldername = Docker::GetFolderName(handle->get_path().get_base_dir());
			String outname = path + foldername + String(".elf");

			auto builder = [inpname = std::move(inpname), outname = std::move(outname)] {
				// Invoke docker to compile the file
				Array output;
				PackedStringArray arguments;
				arguments.push_back("/usr/api/build.sh");
				if (SandboxProjectSettings::debug_info())
					arguments.push_back("--debug");
				Array global_defines = SandboxProjectSettings::get_global_defines();
				for (int i = 0; i < global_defines.size(); i++) {
					arguments.push_back("-D");
					arguments.push_back(global_defines[i]);
				}
				arguments.push_back("-o");
				arguments.push_back(outname);
				arguments.push_back(inpname);
				// CPPScript::DockerContainerExecute({ "/usr/api/build.sh", "-o", outname, inpname }, output);
				CPPScript::DockerContainerExecute(arguments, output);
				if (!output.is_empty() && !output[0].operator String().is_empty()) {
					for (int i = 0; i < output.size(); i++) {
						String line = output[i].operator String();
						if constexpr (VERBOSE_CMD)
							ERR_PRINT(line);
						// Remove (most) console color codes
						line = line.replace("\033[0;31m", "");
						line = line.replace("\033[0;32m", "");
						line = line.replace("\033[0;33m", "");
						line = line.replace("\033[0;34m", "");
						line = line.replace("\033[0;35m", "");
						line = line.replace("\033[0;36m", "");
						line = line.replace("\033[0;37m", "");
						line = line.replace("\033[01;31m", "");
						line = line.replace("\033[01;32m", "");
						line = line.replace("\033[01;33m", "");
						line = line.replace("\033[01;34m", "");
						line = line.replace("\033[01;35m", "");
						line = line.replace("\033[01;36m", "");
						line = line.replace("\033[01;37m", "");
						line = line.replace("\033[m", "");
						line = line.replace("\033[0m", "");
						line = line.replace("\033[01m", "");
						line = line.replace("\033[32m", "");
						line = line.replace("[K", "");
						WARN_PRINT(line);
					}
				}
			};

			// If async compilation is enabled, enqueue the builder to the thread pool
			if (SandboxProjectSettings::async_compilation())
				thread_pool->enqueue(builder);
			else {
				builder();
			}
			return Error::OK;
		} else {
			return Error::ERR_FILE_CANT_OPEN;
		}
	}
	return Error::ERR_SCRIPT_FAILED;
}
Error ResourceFormatSaverCPP::_set_uid(const String &p_path, int64_t p_uid) {
	return Error::OK;
}
bool ResourceFormatSaverCPP::_recognize(const Ref<Resource> &p_resource) const {
	return Object::cast_to<CPPScript>(p_resource.ptr()) != nullptr;
}
PackedStringArray ResourceFormatSaverCPP::_get_recognized_extensions(const Ref<Resource> &p_resource) const {
	PackedStringArray array;
	if (Object::cast_to<CPPScript>(p_resource.ptr()) == nullptr)
		return array;
	array.push_back("cpp");
	array.push_back("cc");
	array.push_back("hh");
	array.push_back("h");
	array.push_back("hpp");
	return array;
}
bool ResourceFormatSaverCPP::_recognize_path(const Ref<Resource> &p_resource, const String &p_path) const {
	return Object::cast_to<CPPScript>(p_resource.ptr()) != nullptr;
}

/**************************************************************************/
/*  sandbox_bintr.cpp                                                     */
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

#include "sandbox.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/variant/variant_utility.h"

#if defined(__linux__)
#include <dlfcn.h>
#elif defined(__MINGW32__) || defined(__MINGW64__) || defined(_MSC_VER)
#define YEP_IS_WINDOWS 1
#include <libriscv/win32/dlfcn.h>
#ifdef _MSC_VER
#define access _access
#define unlink _unlink
extern "C" int access(const char *path, int mode);
extern "C" int unlink(const char *path);
#define R_OK 4 /* Test for read permission.  */
#else // _MSC_VER
#include <unistd.h>
#endif
#elif defined(__APPLE__) && defined(__MACH__) // macOS OSX
#include <TargetConditionals.h>
#if TARGET_OS_MAC
#include <dlfcn.h>
#define YEP_IS_OSX 1
#endif
#endif
extern "C" void libriscv_register_translation8(...);

String Sandbox::emit_binary_translation(bool ignore_instruction_limit, bool automatic_nbit_as) const {
	const std::string_view &binary = machine().memory.binary();
	if (binary.empty()) {
		ERR_PRINT("Sandbox: No binary loaded.");
		return String();
	}
#ifdef RISCV_BINARY_TRANSLATION
	if (machine().is_binary_translation_enabled() && !is_jit()) {
		WARN_PRINT("Sandbox: Binary translation is already enabled.");
		return String();
	}
	std::string code_output;
	// 1. Re-create the same options
	auto options = std::make_shared<riscv::MachineOptions<RISCV_ARCH>>(machine().options());
	options->use_shared_execute_segments = false;
	options->translate_enabled = false;
	options->translate_enable_embedded = true;
	options->translate_invoke_compiler = false;
	options->translate_ignore_instruction_limit = ignore_instruction_limit;
	options->translate_automatic_nbit_address_space = automatic_nbit_as;
	options->translate_use_register_caching = false;
	if constexpr (riscv::libtcc_enabled) {
		// Avoid any shenanigans with background compilation
		options->translate_background_callback = nullptr;
	}
	// TODO: Make this configurable
	options->translate_instr_max = 75'000u;

	// 2. Enable binary translation output to a string
	options->cross_compile.push_back(riscv::MachineTranslationEmbeddableCodeOptions{
			.result_c99 = &code_output,
	});

	// 3. Emit the binary translation by constructing a new machine
	machine_t m{ binary, *options };

	// 4. Wait for any potential background compilation to finish
	if constexpr (riscv::libtcc_enabled) {
		m.cpu.current_execute_segment().wait_for_compilation_complete();
	}

	// 4. Verify that the translation was successful
	if (code_output.empty()) {
		ERR_PRINT("Sandbox: Binary translation failed.");
		return String();
	}
	// 5. Return the translated code
	return String::utf8(code_output.c_str(), code_output.size());
#else
	ERR_PRINT("Sandbox: Binary translation is not enabled.");
	return String();
#endif
}

bool Sandbox::load_binary_translation(const String &shared_library_path, bool allow_insecure) {
	if (m_global_instances_seen > 0 && !allow_insecure) {
		ERR_PRINT("Sandbox: Loading shared libraries after Sandbox instances have been created is a security risk."
				  "Please load shared libraries before creating any Sandbox instances.");
		return false;
	}
#ifdef RISCV_BINARY_TRANSLATION
	// Load the shared library on platforms that support it
#if defined(__linux__) || defined(YEP_IS_WINDOWS) || defined(YEP_IS_OSX)
	Ref<FileAccess> fa = FileAccess::open(shared_library_path, FileAccess::ModeFlags::READ);
	if (!fa.is_valid() || !fa->is_open()) {
		//ERR_PRINT("Sandbox: Failed to open shared library: " + shared_library_path);
		return false;
	}
	String path = fa->get_path_absolute();
	fa->close();
	void *handle = dlopen(path.utf8().ptr(), RTLD_LAZY);
	if (handle == nullptr) {
		ERR_PRINT("Sandbox: Failed to load shared library: " + shared_library_path);
		return false;
	}
	// If the shared library has a callback-based registration function, call it
	void *register_translation = dlsym(handle, "libriscv_init_with_callback8");
	if (register_translation != nullptr) {
		using CallbackFunction = void (*)(void (*)(...));
		((CallbackFunction)register_translation)(libriscv_register_translation8);
	}
#else
	WARN_PRINT_ONCE("Sandbox: Loading shared libraries has not been implemented on this platform.");
#endif
	// We don't need to do anything with the handle, as the shared library should self-register its functions
	return true;
#else
	WARN_PRINT_ONCE("Sandbox: Binary translation is not enabled.");
#endif
	return false;
}

bool Sandbox::try_compile_binary_translation(String shared_library_path, const String &cc, const String &extra_cflags, bool ignore_instruction_limit, bool automatic_nbit_as) {
	if (this->is_binary_translated() && !this->is_jit()) {
		return true;
	}
	if (this->is_in_vmcall()) {
		ERR_PRINT("Sandbox: Cannot produce binary translation while in a VM call. This is a security risk.");
		return false;
	}
	if (this->get_restrictions()) {
		ERR_PRINT("Sandbox: Cannot produce binary translation while restrictions are enabled.");
		return false;
	}
	if (shared_library_path.is_empty()) {
		ERR_PRINT("Sandbox: No shared library path specified.");
		return false;
	}
	if (!shared_library_path.begins_with("res://")) {
		ERR_PRINT("Sandbox: Shared library path must begin with 'res://'.");
		return false;
	}
	// Android, WebAssembly, Nintendo Switch, and iOS do not support direct
	// compilation of binary translations into shared libraries (on that platform).
#if defined(__ANDROID__) || defined(__wasm__) || defined(__SWITCH__) || defined(__EMSCRIPTEN__)
	ERR_PRINT("Sandbox: Directly compiling binary translation is not supported on this platform.");
	return false;
#elif defined(__APPLE__) && !defined(__MACH__) // iOS?
	// TODO: Check for iOS?
	ERR_PRINT("Sandbox: Directly compiling binary translation is not supported on this platform.");
	return false;
#endif

#ifdef __linux__
	shared_library_path += ".so";
#elif defined(YEP_IS_WINDOWS)
	shared_library_path += ".dll";
#elif defined(YEP_IS_OSX)
	shared_library_path += ".dylib";
#else
	WARN_PRINT_ONCE("Sandbox: Compiling binary translations has not been implemented on this platform.");
	return false;
#endif
	const String code = this->emit_binary_translation(ignore_instruction_limit, automatic_nbit_as);
	if (code.is_empty()) {
		ERR_PRINT("Sandbox: Failed to emit binary translation.");
		return false;
	}
	static const String c99_path = "user://temp_sandbox_generated.c";
	Ref<FileAccess> fa = FileAccess::open(c99_path, FileAccess::ModeFlags::WRITE);
	if (!fa->is_open()) {
		ERR_PRINT("Sandbox: Failed to open file for writing: " + c99_path);
		return false;
	}
	fa->store_string(code);
	fa->close();
	// Compile the generated code
	Array args;
	if (cc.ends_with("zig")) {
		// Zig cc - C compiler (faster than C++)
		args.push_back("cc");
	}
#if defined(__linux__) || defined(YEP_IS_OSX)
	args.push_back("-shared");
	args.push_back("-fPIC");
	args.push_back("-fvisibility=hidden");
	args.push_back("-O2");
	args.push_back("-w");
	args.push_back("-DCALLBACK_INIT");
	args.push_back("-o");
#elif defined(YEP_IS_WINDOWS)
	if (cc.ends_with("zig")) {
		// Zig cc - C compiler
		args.push_back("-shared");
		args.push_back("-fPIC");
		args.push_back("-fvisibility=hidden");
		args.push_back("-O2");
		args.push_back("-w");
		args.push_back("-DCALLBACK_INIT");
		args.push_back("-o");
	} else {
		args.push_back("/LD");
		args.push_back("/O2");
		args.push_back("/w");
		args.push_back("/DCALLBACK_INIT");
		args.push_back("/Fe");
	}
#endif
	args.push_back(shared_library_path.replace("res://", ""));
	if (!extra_cflags.is_empty()) {
		Vector<String> extra_flags = extra_cflags.split(" ");
		for (int i = 0; i < extra_flags.size(); i++) {
			args.push_back(extra_flags[i]);
		}
	}
	args.push_back(ProjectSettings::get_singleton()->globalize_path(c99_path));

	// Convert Array to Vector<String> for OS::execute
	Vector<String> args_vector;
	for (int i = 0; i < args.size(); i++) {
		args_vector.push_back(args[i]);
	}

	String args_str = "";
	for (int i = 0; i < args_vector.size(); i++) {
		args_str += args_vector[i];
		if (i < args_vector.size() - 1)
			args_str += " ";
	}
	print_line("Compiling: " + cc + " with args: " + args_str);

	List<String> args_list;
	for (int i = 0; i < args_vector.size(); i++) {
		args_list.push_back(args_vector[i]);
	}

	String output;
	int exit_code;
	Error error = OS::get_singleton()->execute(cc, args_list, &output, &exit_code, true);
	int ret = (error == OK) ? exit_code : 1;
	// Remove the generated C99 file
	Ref<DirAccess> dir = DirAccess::open("user://");
	dir->remove(c99_path);
	if (ret != 0) {
		ERR_PRINT("Sandbox: Failed to compile generated code: " + shared_library_path);
		if (!output.is_empty()) {
			print_line("Compilation output: " + output);
		}
		return false;
	}
	return true;
}

bool Sandbox::is_binary_translated() const {
	// Get main execute segment
	auto &main_seg = this->m_machine->memory.exec_segment_for(this->m_machine->memory.start_address());
	return main_seg->is_binary_translated();
}

bool Sandbox::is_jit() const {
#ifdef RISCV_BINARY_TRANSLATION
	auto &main_seg = this->m_machine->memory.exec_segment_for(this->m_machine->memory.start_address());
	return main_seg->is_libtcc() || main_seg->is_background_compiling();
#else
	return false;
#endif
}

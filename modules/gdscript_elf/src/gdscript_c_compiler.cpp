/**************************************************************************/
/*  gdscript_c_compiler.cpp                                               */
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

#include "gdscript_c_compiler.h"

#include "core/error/error_macros.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/templates/list.h"
#include "core/os/os.h"
#include "core/string/string_builder.h"
#include "core/variant/array.h"

GDScriptCCompiler::GDScriptCCompiler() {
	last_error.clear();
	compiler_path = detect_cross_compiler();
}

GDScriptCCompiler::~GDScriptCCompiler() {
	last_error.clear();
	compiler_path.clear();
}

PackedByteArray GDScriptCCompiler::compile_to_elf(const String &p_c_code) {
	last_error.clear();

	if (p_c_code.is_empty()) {
		return PackedByteArray();
	}
	if (compiler_path.is_empty()) {
		compiler_path = detect_cross_compiler();
		if (compiler_path.is_empty()) {
			return PackedByteArray();
		}
	}

	String c_file = write_temp_c_file(p_c_code);
	if (c_file.is_empty()) {
		return PackedByteArray();
	}
	String elf_file = c_file.get_basename() + ".elf";
	if (invoke_compiler(c_file, elf_file) != OK) {
		cleanup_temp_files(c_file, elf_file);
		return PackedByteArray();
	}
	PackedByteArray elf_binary = read_elf_file(elf_file);
	cleanup_temp_files(c_file, elf_file);
	return elf_binary;
}

String GDScriptCCompiler::detect_cross_compiler() {
	// Try common RISC-V cross-compiler names
	const char *compiler_names[] = {
		"riscv64-unknown-elf-gcc",
		"riscv64-linux-gnu-gcc",
		"riscv64-elf-gcc",
		nullptr
	};

	for (int i = 0; compiler_names[i] != nullptr; i++) {
		String compiler = find_compiler_in_path(compiler_names[i]);
		if (!compiler.is_empty()) {
			return compiler;
		}
	}

	return String();
}

bool GDScriptCCompiler::is_compiler_available() {
	return !detect_cross_compiler().is_empty();
}

String GDScriptCCompiler::find_compiler_in_path(const String &p_compiler_name) {
	// Try to find compiler using 'which' command (Unix) or 'where' (Windows)
	String command;
#ifdef _WIN32
	command = "where " + p_compiler_name;
#else
	command = "which " + p_compiler_name;
#endif

	// Execute command and check if compiler exists
	// For now, just return the name if it's in a common location
	// In a real implementation, we'd execute the command and parse output
	// For simplicity, we'll assume the compiler name is the full path if it contains '/'
	if (p_compiler_name.contains("/")) {
		// Already a path, check if file exists
		Ref<FileAccess> fa = FileAccess::open(p_compiler_name, FileAccess::READ);
		if (fa.is_valid()) {
			return p_compiler_name;
		}
	}

	// Try common installation paths
	Vector<String> common_paths;
#ifdef _WIN32
	common_paths.push_back("C:/Program Files/RISC-V/bin/" + p_compiler_name + ".exe");
#else
	common_paths.push_back("/usr/bin/" + p_compiler_name);
	common_paths.push_back("/usr/local/bin/" + p_compiler_name);
	common_paths.push_back("/opt/riscv/bin/" + p_compiler_name);
#endif

	for (const String &path : common_paths) {
		Ref<FileAccess> fa = FileAccess::open(path, FileAccess::READ);
		if (fa.is_valid()) {
			return path;
		}
	}

	// Return just the name - let the shell find it
	return p_compiler_name;
}

String GDScriptCCompiler::write_temp_c_file(const String &p_c_code) {
	// Create temporary file
	String temp_dir = OS::get_singleton()->get_cache_path();
	String temp_file = temp_dir.path_join("gdscript_elf_" + itos(OS::get_singleton()->get_ticks_msec()) + ".c");

	Ref<FileAccess> fa = FileAccess::open(temp_file, FileAccess::WRITE);
	if (fa.is_null()) {
		return String();
	}

	fa->store_string(p_c_code);
	fa->close();

	return temp_file;
}

Error GDScriptCCompiler::invoke_compiler(const String &p_c_file, const String &p_elf_file) {
	// Build compiler arguments
	// riscv64-unknown-elf-gcc -o output.elf -nostdlib -static -O0 input.c
	List<String> arguments;
	arguments.push_back("-o");
	arguments.push_back(p_elf_file);
	arguments.push_back("-nostdlib");
	arguments.push_back("-static");
	arguments.push_back("-O0");
	arguments.push_back(p_c_file);

	String output;
	int exit_code = 0;
	Error err = OS::get_singleton()->execute(compiler_path, arguments, &output, &exit_code, true, nullptr, false);

	if (exit_code != 0 || err != OK) {
		last_error = output;
		return ERR_COMPILATION_FAILED;
	}
	if (!FileAccess::file_exists(p_elf_file)) {
		return ERR_FILE_NOT_FOUND;
	}

	return OK;
}

PackedByteArray GDScriptCCompiler::read_elf_file(const String &p_elf_file) {
	Ref<FileAccess> fa = FileAccess::open(p_elf_file, FileAccess::READ);
	if (fa.is_null()) {
		return PackedByteArray();
	}

	// Read entire file
	int64_t file_size = fa->get_length();
	PackedByteArray data;
	data.resize(file_size);
	fa->get_buffer(data.ptrw(), file_size);
	fa->close();

	return data;
}

void GDScriptCCompiler::cleanup_temp_files(const String &p_c_file, const String &p_elf_file) {
	// Delete temporary files
	if (!p_c_file.is_empty()) {
		Ref<DirAccess> da = DirAccess::open(p_c_file.get_base_dir());
		if (da.is_valid()) {
			da->remove(p_c_file);
		}
	}

	if (!p_elf_file.is_empty()) {
		Ref<DirAccess> da = DirAccess::open(p_elf_file.get_base_dir());
		if (da.is_valid()) {
			da->remove(p_elf_file);
		}
	}
}

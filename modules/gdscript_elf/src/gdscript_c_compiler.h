/**************************************************************************/
/*  gdscript_c_compiler.h                                                 */
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

#pragma once

#include "core/error/error_list.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"

// Invokes RISC-V cross-compiler to compile C++ code to ELF
class GDScriptCCompiler {
public:
	GDScriptCCompiler();
	~GDScriptCCompiler();

	// Compile C++ source code to ELF binary
	// p_c_code: C++ source code to compile
	// Returns ELF binary as PackedByteArray, empty on error
	PackedByteArray compile_to_elf(const String &p_c_code);

	// Auto-detect RISC-V cross-compiler from PATH
	// Returns compiler path if found, empty String if not found
	static String detect_cross_compiler();

	// Check if cross-compiler is available
	static bool is_compiler_available();

	// Get last compilation error (if any)
	String get_last_error() const { return last_error; }

private:
	String last_error;
	String compiler_path;

	// Try to find compiler in PATH
	static String find_compiler_in_path(const String &p_compiler_name);

	// Write C code to temporary file
	String write_temp_c_file(const String &p_c_code);

	// Invoke compiler via shell
	Error invoke_compiler(const String &p_c_file, const String &p_elf_file);

	// Read ELF binary from file
	PackedByteArray read_elf_file(const String &p_elf_file);

	// Clean up temporary files
	void cleanup_temp_files(const String &p_c_file, const String &p_elf_file);
};

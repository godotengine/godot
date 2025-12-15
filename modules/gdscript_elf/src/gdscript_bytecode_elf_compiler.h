/**************************************************************************/
/*  gdscript_bytecode_elf_compiler.h                                      */
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

#include "core/string/string_name.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

// Forward declarations
class GDScriptFunction;

// Orchestrates bytecode-to-ELF compilation via C code generation
class GDScriptBytecodeELFCompiler {
public:
	// Compile a GDScriptFunction's bytecode to RISC-V ELF
	// Returns the ELF binary as a PackedByteArray
	// Returns empty PackedByteArray on error
	static PackedByteArray compile_function_to_elf(GDScriptFunction *p_function);

	// Check if a function can be compiled to ELF
	// Returns true if cross-compiler is available and function has bytecode
	static bool can_compile_function(GDScriptFunction *p_function);

	// Get list of unsupported opcodes for a function
	// Returns list of opcode names that would require fallback
	static Vector<String> get_unsupported_opcodes(GDScriptFunction *p_function);

	// Check if cross-compiler is available
	static bool is_compiler_available();

	// Get last compilation error (if any)
	static String get_last_error();

private:
	// Internal compilation state
	struct CompilationState {
		GDScriptFunction *function = nullptr;
		PackedByteArray elf_output;
		Vector<String> errors;
		Vector<String> warnings;
		bool has_unsupported_opcodes = false;
	};

	// Main compilation logic
	static Error compile_internal(CompilationState &p_state);

	// Validate function can be compiled
	static bool validate_function(GDScriptFunction *p_function, CompilationState &p_state);
};

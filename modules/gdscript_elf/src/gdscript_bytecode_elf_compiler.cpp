/**************************************************************************/
/*  gdscript_bytecode_elf_compiler.cpp                                    */
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

#include "gdscript_bytecode_elf_compiler.h"

#include "core/error/error_macros.h"
#include "gdscript_bytecode_c_codegen.h"
#include "gdscript_c_compiler.h"
#include "modules/gdscript/gdscript_function.h"

static String last_compilation_error;

PackedByteArray GDScriptBytecodeELFCompiler::compile_function_to_elf(GDScriptFunction *p_function) {
	ERR_FAIL_NULL_V(p_function, PackedByteArray());

	last_compilation_error.clear();

	CompilationState state;
	state.function = p_function;

	Error err = compile_internal(state);
	if (err != OK) {
		// Log errors but return empty array
		for (const String &error : state.errors) {
			ERR_PRINT("GDScriptBytecodeELFCompiler: " + error);
		}
		if (!state.errors.is_empty()) {
			last_compilation_error = state.errors[0];
		}
		return PackedByteArray();
	}

	return state.elf_output;
}

bool GDScriptBytecodeELFCompiler::can_compile_function(GDScriptFunction *p_function) {
	ERR_FAIL_NULL_V(p_function, false);

	// Check if compiler is available
	if (!is_compiler_available()) {
		return false;
	}

	// Check if function has bytecode
	if (p_function->code.is_empty()) {
		return false;
	}

	return true;
}

Vector<String> GDScriptBytecodeELFCompiler::get_unsupported_opcodes(GDScriptFunction *p_function) {
	ERR_FAIL_NULL_V(p_function, Vector<String>());

	CompilationState state;
	state.function = p_function;

	validate_function(p_function, state);
	return state.warnings; // Warnings contain unsupported opcode names
}

bool GDScriptBytecodeELFCompiler::is_compiler_available() {
	return GDScriptCCompiler::is_compiler_available();
}

String GDScriptBytecodeELFCompiler::get_last_error() {
	return last_compilation_error;
}

Error GDScriptBytecodeELFCompiler::compile_internal(CompilationState &p_state) {
	ERR_FAIL_NULL_V(p_state.function, ERR_INVALID_PARAMETER);

	// Validate function
	if (!validate_function(p_state.function, p_state)) {
		return ERR_INVALID_DATA;
	}

	// Check if compiler is available
	if (!is_compiler_available()) {
		p_state.errors.push_back("RISC-V cross-compiler not available");
		return ERR_UNAVAILABLE;
	}

	// Generate C code from bytecode
	GDScriptBytecodeCCodeGenerator codegen;
	String c_code = codegen.generate_c_code(p_state.function);
	if (c_code.is_empty()) {
		p_state.errors.push_back("Failed to generate C code");
		return ERR_COMPILATION_FAILED;
	}

	// Note: Generated C code expects constants and operator_funcs as parameters
	// These need to be passed when calling the generated function
	// For now, the generated code will compile but runtime linking is Phase 3

	// Compile C code to ELF
	GDScriptCCompiler compiler;
	p_state.elf_output = compiler.compile_to_elf(c_code);
	if (p_state.elf_output.is_empty()) {
		p_state.errors.push_back("Failed to compile C code to ELF: " + compiler.get_last_error());
		last_compilation_error = compiler.get_last_error();
		return ERR_COMPILATION_FAILED;
	}

	return OK;
}

bool GDScriptBytecodeELFCompiler::validate_function(GDScriptFunction *p_function, CompilationState &p_state) {
	ERR_FAIL_NULL_V(p_function, false);

	// Check if function has bytecode
	if (p_function->code.is_empty()) {
		p_state.errors.push_back("Function has no bytecode");
		return false;
	}

	// Check for unsupported opcodes (for now, all opcodes are unsupported in Phase 1)
	// This will be gradually filled in as we implement opcode support
	// For now, we'll allow compilation but mark it as having unsupported opcodes
	p_state.has_unsupported_opcodes = true;
	p_state.warnings.push_back("ELF compilation is in early phase - all opcodes will use fallback");

	return true;
}

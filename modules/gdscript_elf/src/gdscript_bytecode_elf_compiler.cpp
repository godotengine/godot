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
	if (!is_compiler_available() || p_function->code.is_empty()) {
		return PackedByteArray();
	}
	GDScriptBytecodeCCodeGenerator codegen;
	String c_code = codegen.generate_c_code(p_function);
	if (c_code.is_empty()) {
		last_compilation_error = "Failed to generate C code";
		return PackedByteArray();
	}
	GDScriptCCompiler compiler;
	PackedByteArray elf = compiler.compile_to_elf(c_code);
	if (elf.is_empty()) {
		last_compilation_error = compiler.get_last_error();
	}
	return elf;
}

bool GDScriptBytecodeELFCompiler::can_compile_function(GDScriptFunction *p_function) {
	return p_function && is_compiler_available() && !p_function->code.is_empty();
}

Vector<String> GDScriptBytecodeELFCompiler::get_unsupported_opcodes(GDScriptFunction *p_function) {
	return Vector<String>(); // Simplified - no opcode tracking
}

bool GDScriptBytecodeELFCompiler::is_compiler_available() {
	return GDScriptCCompiler::is_compiler_available();
}

String GDScriptBytecodeELFCompiler::get_last_error() {
	return last_compilation_error;
}

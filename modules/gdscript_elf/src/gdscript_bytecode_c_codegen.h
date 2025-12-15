/**************************************************************************/
/*  gdscript_bytecode_c_codegen.h                                         */
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

#include "core/string/string_builder.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"

// Forward declarations
class GDScriptFunction;

// Generates C++ code from GDScript bytecode
// The generated code will be compiled by RISC-V cross-compiler to ELF
class GDScriptBytecodeCCodeGenerator {
public:
	GDScriptBytecodeCCodeGenerator();
	~GDScriptBytecodeCCodeGenerator();

	// Generate C++ code from GDScriptFunction bytecode
	// Returns the generated C++ source code as a String
	// Returns empty String on error
	String generate_c_code(GDScriptFunction *p_function);

	// Get the generated C++ code
	String get_generated_code() const { return generated_code; }

	// Check if generation was successful
	bool is_valid() const { return !generated_code.is_empty(); }

private:
	String generated_code;

	// Generate function signature
	void generate_function_signature(GDScriptFunction *p_function, StringBuilder &r_code);

	// Generate function body
	void generate_function_body(GDScriptFunction *p_function, StringBuilder &r_code);

	// Generate stack variable declarations
	void generate_stack_variables(int p_stack_size, StringBuilder &r_code);

	// Generate parameter extraction from args array
	// Extracts: result, instance, constants, operator_funcs, actual_args, actual_argcount
	// Also handles default arguments and initializes stack
	void generate_parameter_extraction(GDScriptFunction *p_function, StringBuilder &r_code);

	// Generate code for a single bytecode opcode
	void generate_opcode(GDScriptFunction *p_function, int p_opcode, const int *p_code_ptr, int &p_ip, StringBuilder &r_code);

	// Generate fallback call for unsupported opcodes
	void generate_fallback_call(int p_opcode, const int *p_args, int p_arg_count, StringBuilder &r_code);

	// Generate syscall using inline assembly
	void generate_syscall(int p_syscall_number, StringBuilder &r_code);

	// Helper: Get address type from encoded address
	int get_address_type(int p_address);

	// Helper: Get address value from encoded address
	int get_address_value(int p_address);

	// Helper: Generate C variable name for stack slot
	String get_stack_var_name(int p_slot);

	// Helper: Generate C variable name for constant
	String get_constant_var_name(int p_index);

	// Helper: Resolve address to C variable name
	String resolve_address(int p_address);
};

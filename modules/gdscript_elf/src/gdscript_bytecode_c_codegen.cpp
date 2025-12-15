/**************************************************************************/
/*  gdscript_bytecode_c_codegen.cpp                                       */
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

#include "gdscript_bytecode_c_codegen.h"

#include "core/error/error_macros.h"
#include "core/string/string_builder.h"
#include "core/templates/hash_map.h"
#include "gdscript_elf_fallback.h"
#include "modules/gdscript/gdscript_function.h"
#include "modules/sandbox/src/syscalls.h"

GDScriptBytecodeCCodeGenerator::GDScriptBytecodeCCodeGenerator() {
	generated_code.clear();
}

GDScriptBytecodeCCodeGenerator::~GDScriptBytecodeCCodeGenerator() {
	generated_code.clear();
}

String GDScriptBytecodeCCodeGenerator::generate_c_code(GDScriptFunction *p_function) {
	ERR_FAIL_NULL_V(p_function, String());

	generated_code.clear();

	// Check if function has bytecode
	if (p_function->code.is_empty()) {
		ERR_PRINT("GDScriptBytecodeCCodeGenerator: Function has no bytecode");
		return String();
	}

	StringBuilder code;

	// Generate includes
	code.append("#include <stdint.h>\n");
	code.append("// Note: Variant and other Godot types need to be available\n");
	code.append("// This will be linked with Godot's runtime\n\n");

	// Generate extern declaration for fallback function
	code.append("extern void gdscript_vm_fallback(int opcode, void* instance, void* stack, int ip);\n");

	// Generate syscall number definitions (from modules/sandbox/src/syscalls.h)
	code.append("#define GAME_API_BASE 500\n");
	code.append("#define ECALL_OBJ_PROP_GET (GAME_API_BASE + 45)\n");
	code.append("#define ECALL_OBJ_PROP_SET (GAME_API_BASE + 46)\n");
	code.append("#define ECALL_VCALL (GAME_API_BASE + 1)\n");

	// Generate helper function declarations for global_names access
	code.append("extern const StringName* get_global_name(int index);\n");
	code.append("extern const char* get_global_name_cstr(int index, size_t* out_len);\n\n");

	// Constants and operator functions are passed as parameters
	// No need for extern declarations - they're function parameters

	// Generate function
	generate_function_signature(p_function, code);
	code.append(" {\n");

	// Generate stack variables
	generate_stack_variables(p_function->get_max_stack_size(), code);

	// Generate parameter extraction from args array
	// Extended args structure: [result_ptr, arg0, arg1, ..., argN, instance_ptr, constants_addr, operator_funcs_addr]
	generate_parameter_extraction(p_function, code);

	// Generate function body
	generate_function_body(p_function, code);

	code.append("}\n");

	generated_code = code.as_string();
	return generated_code;
}

void GDScriptBytecodeCCodeGenerator::generate_function_signature(GDScriptFunction *p_function, StringBuilder &r_code) {
	// Generate function signature matching vmcall_address API
	// Extended args structure: [result_ptr, arg0, arg1, ..., argN, instance_ptr, constants_addr, operator_funcs_addr]
	// void gdscript_function_name(Variant** args, int argcount)

	String func_name = p_function->get_name().operator String();
	// Sanitize function name for C (replace invalid characters)
	func_name = func_name.replace(".", "_").replace(" ", "_");

	r_code += "void gdscript_" + func_name + "(Variant** args, int argcount)";
}

void GDScriptBytecodeCCodeGenerator::generate_stack_variables(int p_stack_size, StringBuilder &r_code) {
	// Generate stack array: Variant stack[STACK_SIZE];
	r_code += "    Variant stack[" + itos(p_stack_size) + "];\n";
	r_code += "    int ip = 0; // Instruction pointer\n";
	r_code += "\n";
}

void GDScriptBytecodeCCodeGenerator::generate_parameter_extraction(GDScriptFunction *p_function, StringBuilder &r_code) {
	// Extract parameters from extended args array
	// Extended args structure: [result_ptr, arg0, arg1, ..., argN, instance_ptr, constants_addr, operator_funcs_addr]
	// Total: 1 (result) + actual_argcount + 3 (instance, constants, operator_funcs) = argcount

	int argument_count = p_function->get_argument_count();
	int default_arg_count = p_function->_default_arg_count;

	r_code += "    // Extract parameters from extended args array\n";
	r_code += "    // args[0] = result pointer\n";
	r_code += "    // args[1..argcount-3] = actual function arguments\n";
	r_code += "    // args[argcount-3] = instance pointer\n";
	r_code += "    // args[argcount-2] = constants address (gaddr_t as int64_t)\n";
	r_code += "    // args[argcount-1] = operator_funcs address (gaddr_t as int64_t)\n";
	r_code += "    \n";
	r_code += "    if (argcount < 4) {\n";
	r_code += "        // Invalid argcount, set defaults and return\n";
	r_code += "        if (argcount > 0 && args[0] != NULL) *args[0] = Variant();\n";
	r_code += "        return;\n";
	r_code += "    }\n";
	r_code += "    \n";
	r_code += "    Variant* result = args[0];\n";
	r_code += "    \n";
	r_code += "    // Calculate actual argument count (excluding result, instance, constants, operator_funcs)\n";
	r_code += "    int actual_argcount = argcount - 4;\n";
	r_code += "    if (actual_argcount < 0) actual_argcount = 0;\n";
	r_code += "    \n";
	r_code += "    // Calculate defarg for default argument handling\n";
	r_code += "    // defarg = argument_count - actual_argcount (when fewer args provided)\n";
	r_code += "    int defarg = 0;\n";
	if (default_arg_count > 0) {
		r_code += "    if (actual_argcount < " + itos(argument_count) + ") {\n";
		r_code += "        int min_args = " + itos(argument_count) + " - " + itos(default_arg_count) + ";\n";
		r_code += "        if (actual_argcount < min_args) {\n";
		r_code += "            // Too few arguments - error case, return default\n";
		r_code += "            if (result != NULL) *result = Variant();\n";
		r_code += "            return;\n";
		r_code += "        }\n";
		r_code += "        defarg = " + itos(argument_count) + " - actual_argcount;\n";
		r_code += "    }\n";
	}
	r_code += "    \n";
	r_code += "    // Extract instance pointer\n";
	r_code += "    void* instance = (void*)(uintptr_t)(int64_t)(*args[argcount - 3]);\n";
	r_code += "    \n";
	r_code += "    // Extract constants array pointer from shared memory address\n";
	r_code += "    typedef uint64_t gaddr_t; // gaddr_t is uint64_t\n";
	r_code += "    gaddr_t constants_addr = (gaddr_t)(int64_t)(*args[argcount - 2]);\n";
	r_code += "    Variant* constants = (constants_addr != 0) ? (Variant*)constants_addr : NULL;\n";
	r_code += "    \n";
	r_code += "    // Extract operator_funcs array pointer from shared memory address\n";
	r_code += "    gaddr_t operator_funcs_addr = (gaddr_t)(int64_t)(*args[argcount - 1]);\n";
	r_code += "    Variant::ValidatedOperatorEvaluator* operator_funcs = (operator_funcs_addr != 0) ? (Variant::ValidatedOperatorEvaluator*)operator_funcs_addr : NULL;\n";
	r_code += "    \n";
	r_code += "    // Initialize fixed addresses (self, class, nil)\n";
	r_code += "    // stack[0] = self (instance)\n";
	r_code += "    // stack[1] = class (script)\n";
	r_code += "    // stack[2] = nil\n";
	r_code += "    if (instance != NULL) {\n";
	r_code += "        stack[0] = Variant((Object*)instance);\n";
	r_code += "    } else {\n";
	r_code += "        stack[0] = Variant();\n";
	r_code += "    }\n";
	r_code += "    stack[1] = Variant(); // class - would need script reference\n";
	r_code += "    stack[2] = Variant(); // nil\n";
	r_code += "    \n";
	r_code += "    // Initialize function arguments on stack\n";
	r_code += "    // Arguments are placed at stack[FIXED_ADDRESSES_MAX + i] for i in [0, argument_count-1]\n";
	r_code += "    const int FIXED_ADDRESSES_MAX = 3;\n";
	r_code += "    for (int i = 0; i < " + itos(argument_count) + "; i++) {\n";
	r_code += "        if (i < actual_argcount) {\n";
	r_code += "            // Copy provided argument\n";
	r_code += "            stack[FIXED_ADDRESSES_MAX + i] = *args[i + 1];\n";
	r_code += "        } else {\n";
	r_code += "            // Initialize with default (will be set by OPCODE_JUMP_TO_DEF_ARGUMENT)\n";
	r_code += "            stack[FIXED_ADDRESSES_MAX + i] = Variant();\n";
	r_code += "        }\n";
	r_code += "    }\n";
	r_code += "    \n";
	r_code += "    // Initialize remaining stack slots\n";
	r_code += "    // Note: This is simplified - full initialization would handle typed arguments, varargs, etc.\n";
	r_code += "\n";
}

void GDScriptBytecodeCCodeGenerator::generate_function_body(GDScriptFunction *p_function, StringBuilder &r_code) {
	// Generate code for each bytecode instruction
	const int *code_ptr = p_function->code.ptr();
	int code_size = p_function->code.size();
	int ip = 0;

	// Generate label for each instruction (for jumps)
	HashMap<int, String> labels;

	// Pre-generate all labels (needed for forward jumps)
	while (ip < code_size) {
		String label = "label_" + itos(ip) + ":";
		labels[ip] = label;
		ip++;
	}

	// Reset IP and generate code
	ip = 0;
	while (ip < code_size) {
		// Generate label for this instruction
		r_code += "    " + labels[ip] + "\n";

		int opcode = code_ptr[ip];
		generate_opcode(p_function, opcode, code_ptr, ip, r_code);
		// ip is updated by generate_opcode to point to next instruction
	}
}

void GDScriptBytecodeCCodeGenerator::generate_opcode(GDScriptFunction *p_function, int p_opcode, const int *p_code_ptr, int &p_ip, StringBuilder &r_code) {
	// Phase 2: Implement direct C code generation for supported opcodes
	// Check if opcode is supported (Phase 2: simple assignments and control flow)
	if (GDScriptELFFallback::is_opcode_supported(p_opcode)) {
		// Generate direct C code for supported opcodes
		switch (p_opcode) {
			case GDScriptFunction::OPCODE_ASSIGN: {
				// OPCODE_ASSIGN: code[ip+1] = dst_addr, code[ip+2] = src_addr
				int dst_addr = p_code_ptr[p_ip + 1];
				int src_addr = p_code_ptr[p_ip + 2];
				String dst_var = resolve_address(dst_addr);
				String src_var = resolve_address(src_addr);
				r_code += "    " + dst_var + " = " + src_var + ";\n";
				p_ip += 3;
				return;
			}
			case GDScriptFunction::OPCODE_ASSIGN_NULL: {
				// OPCODE_ASSIGN_NULL: code[ip+1] = dst_addr
				int dst_addr = p_code_ptr[p_ip + 1];
				String dst_var = resolve_address(dst_addr);
				r_code += "    " + dst_var + " = Variant();\n";
				p_ip += 2;
				return;
			}
			case GDScriptFunction::OPCODE_ASSIGN_TRUE: {
				// OPCODE_ASSIGN_TRUE: code[ip+1] = dst_addr
				int dst_addr = p_code_ptr[p_ip + 1];
				String dst_var = resolve_address(dst_addr);
				r_code += "    " + dst_var + " = true;\n";
				p_ip += 2;
				return;
			}
			case GDScriptFunction::OPCODE_ASSIGN_FALSE: {
				// OPCODE_ASSIGN_FALSE: code[ip+1] = dst_addr
				int dst_addr = p_code_ptr[p_ip + 1];
				String dst_var = resolve_address(dst_addr);
				r_code += "    " + dst_var + " = false;\n";
				p_ip += 2;
				return;
			}
			case GDScriptFunction::OPCODE_JUMP: {
				// OPCODE_JUMP: code[ip+1] = target_ip
				int target_ip = p_code_ptr[p_ip + 1];
				r_code += "    goto label_" + itos(target_ip) + ";\n";
				p_ip += 2;
				return;
			}
			case GDScriptFunction::OPCODE_JUMP_IF: {
				// OPCODE_JUMP_IF: code[ip+1] = test_addr, code[ip+2] = target_ip
				int test_addr = p_code_ptr[p_ip + 1];
				int target_ip = p_code_ptr[p_ip + 2];
				String test_var = resolve_address(test_addr);
				r_code += "    if (" + test_var + ".booleanize()) goto label_" + itos(target_ip) + ";\n";
				p_ip += 3;
				return;
			}
			case GDScriptFunction::OPCODE_JUMP_IF_NOT: {
				// OPCODE_JUMP_IF_NOT: code[ip+1] = test_addr, code[ip+2] = target_ip
				int test_addr = p_code_ptr[p_ip + 1];
				int target_ip = p_code_ptr[p_ip + 2];
				String test_var = resolve_address(test_addr);
				r_code += "    if (!" + test_var + ".booleanize()) goto label_" + itos(target_ip) + ";\n";
				p_ip += 3;
				return;
			}
			case GDScriptFunction::OPCODE_OPERATOR_VALIDATED: {
				// OPCODE_OPERATOR_VALIDATED: code[ip+1] = left_addr, code[ip+2] = right_addr, code[ip+3] = dst_addr, code[ip+4] = operator_idx
				int left_addr = p_code_ptr[p_ip + 1];
				int right_addr = p_code_ptr[p_ip + 2];
				int dst_addr = p_code_ptr[p_ip + 3];
				int operator_idx = p_code_ptr[p_ip + 4];

				String left_var = resolve_address(left_addr);
				String right_var = resolve_address(right_addr);
				String dst_var = resolve_address(dst_addr);

				// Use validated operator evaluator from operator_funcs array (passed as parameter)
				// Phase 2: Use validated operator evaluator
				r_code += "    {\n";
				r_code += "        if (operator_funcs != NULL) {\n";
				r_code += "            Variant::ValidatedOperatorEvaluator op_func = operator_funcs[" + itos(operator_idx) + "];\n";
				r_code += "            op_func(&" + left_var + ", &" + right_var + ", &" + dst_var + ");\n";
				r_code += "        } else {\n";
				r_code += "            // Fallback: operator_funcs not available\n";
				r_code += "            gdscript_vm_fallback(" + itos(p_opcode) + ", instance, stack, " + itos(p_ip) + ");\n";
				r_code += "        }\n";
				r_code += "    }\n";
				p_ip += 5;
				return;
			}
			case GDScriptFunction::OPCODE_RETURN: {
				// OPCODE_RETURN: code[ip+1] = return_value_addr
				int return_addr = p_code_ptr[p_ip + 1];
				String return_var = resolve_address(return_addr);
				r_code += "    *result = " + return_var + ";\n";
				r_code += "    return;\n";
				p_ip += 2;
				return;
			}
			case GDScriptFunction::OPCODE_LINE: {
				// OPCODE_LINE: code[ip+1] = line_number
				// Metadata opcode for debugging - just skip it
				p_ip += 2;
				return;
			}
			case GDScriptFunction::OPCODE_JUMP_TO_DEF_ARGUMENT: {
				// OPCODE_JUMP_TO_DEF_ARGUMENT: code[ip+1] is not used, defarg variable determines jump target
				// The default_arguments array contains IP addresses for each default argument
				// We need to generate a switch statement based on defarg
				// Note: default_arguments is a Vector<int> in p_function, but we can't access it directly
				// For now, use fallback - proper implementation would require passing default_arguments array
				// or generating a switch statement with all default argument IPs
				if (p_function->default_arguments.size() > 0) {
					r_code += "    // Jump to default argument initialization based on defarg\n";
					r_code += "    switch (defarg) {\n";
					for (int i = 0; i < p_function->default_arguments.size(); i++) {
						int target_ip = p_function->default_arguments[i];
						r_code += "        case " + itos(i + 1) + ": goto label_" + itos(target_ip) + "; break;\n";
					}
					r_code += "        default: break; // No default argument needed\n";
					r_code += "    }\n";
				}
				p_ip += 2;
				return;
			}
			case GDScriptFunction::OPCODE_END: {
				// OPCODE_END: End of function
				r_code += "    return;\n";
				p_ip += 1;
				return;
			}
			case GDScriptFunction::OPCODE_GET_MEMBER: {
				// OPCODE_GET_MEMBER: code[ip+1] = dst_addr, code[ip+2] = name_idx (global_names index)
				int dst_addr = p_code_ptr[p_ip + 1];
				int name_idx = p_code_ptr[p_ip + 2];
				String dst_var = resolve_address(dst_addr);

				// Generate syscall using inline assembly
				// Parameters: a0 = instance (uint64_t), a1 = property name (const char*), a2 = name length (size_t), a3 = result (Variant*), a7 = syscall number
				r_code += "    {\n";
				r_code += "        size_t name_len;\n";
				r_code += "        const char* name_cstr = get_global_name_cstr(" + itos(name_idx) + ", &name_len);\n";
				r_code += "        register uint64_t object asm(\"a0\") = (uint64_t)instance;\n";
				r_code += "        register const char* property asm(\"a1\") = name_cstr;\n";
				r_code += "        register size_t property_size asm(\"a2\") = name_len;\n";
				r_code += "        register Variant* var_ptr asm(\"a3\") = &" + dst_var + ";\n";
				r_code += "        register int syscall_number asm(\"a7\") = ECALL_OBJ_PROP_GET;\n";
				r_code += "        __asm__ volatile(\"ecall\" : \"=m\"(*var_ptr) : \"r\"(object), \"r\"(property), \"m\"(*property), \"r\"(property_size), \"r\"(var_ptr), \"r\"(syscall_number));\n";
				r_code += "    }\n";
				p_ip += 3;
				return;
			}
			case GDScriptFunction::OPCODE_SET_MEMBER: {
				// OPCODE_SET_MEMBER: code[ip+1] = src_addr, code[ip+2] = name_idx (global_names index)
				int src_addr = p_code_ptr[p_ip + 1];
				int name_idx = p_code_ptr[p_ip + 2];
				String src_var = resolve_address(src_addr);

				// Generate syscall using inline assembly
				// Parameters: a0 = instance (uint64_t), a1 = property name (const char*), a2 = name length (size_t), a3 = value (const Variant*), a7 = syscall number
				r_code += "    {\n";
				r_code += "        size_t name_len;\n";
				r_code += "        const char* name_cstr = get_global_name_cstr(" + itos(name_idx) + ", &name_len);\n";
				r_code += "        register uint64_t object asm(\"a0\") = (uint64_t)instance;\n";
				r_code += "        register const char* property asm(\"a1\") = name_cstr;\n";
				r_code += "        register size_t property_size asm(\"a2\") = name_len;\n";
				r_code += "        register const Variant* value_ptr asm(\"a3\") = &" + src_var + ";\n";
				r_code += "        register int syscall_number asm(\"a7\") = ECALL_OBJ_PROP_SET;\n";
				r_code += "        __asm__ volatile(\"ecall\" : : \"r\"(object), \"r\"(property), \"m\"(*property), \"r\"(property_size), \"r\"(value_ptr), \"m\"(*value_ptr), \"r\"(syscall_number));\n";
				r_code += "    }\n";
				p_ip += 3;
				return;
			}
			case GDScriptFunction::OPCODE_CALL:
			case GDScriptFunction::OPCODE_CALL_RETURN: {
				// OPCODE_CALL/OPCODE_CALL_RETURN: instruction_args (base, args...), then argc, then methodname_idx, then (if RETURN) ret_addr
				// Format: code[ip] = opcode, code[ip+1] = instr_arg_count, code[ip+2..ip+1+instr_arg_count] = instruction args,
				//         After loading instruction args, ip is advanced by instr_arg_count, then:
				//         code[ip+1] = argc, code[ip+2] = methodname_idx, [code[ip+3] = ret_addr if RETURN]
				//         instruction_args[0..argc-1] = argument addresses, instruction_args[argc] = base object address
				bool call_ret = (p_opcode == GDScriptFunction::OPCODE_CALL_RETURN);

				int instr_arg_count = p_code_ptr[p_ip + 1];
				// After loading instruction args, ip would be at ip + 1 + instr_arg_count
				int argc = p_code_ptr[p_ip + 2 + instr_arg_count];
				int methodname_idx = p_code_ptr[p_ip + 3 + instr_arg_count];

				// Get base object (at instruction_args[argc], which is code[ip+2+argc])
				int base_addr = p_code_ptr[p_ip + 2 + argc];
				String base_var = resolve_address(base_addr);

				// Get method name
				r_code += "    {\n";
				r_code += "        size_t method_len;\n";
				r_code += "        const char* method_cstr = get_global_name_cstr(" + itos(methodname_idx) + ", &method_len);\n";

				// Collect arguments into array (instruction_args[0..argc-1] are at code[ip+2..ip+1+argc])
				r_code += "        Variant call_args[" + itos(argc) + "];\n";
				for (int i = 0; i < argc; i++) {
					int arg_addr = p_code_ptr[p_ip + 2 + i];
					String arg_var = resolve_address(arg_addr);
					r_code += "        call_args[" + itos(i) + "] = " + arg_var + ";\n";
				}

				// Generate syscall using inline assembly
				// Parameters: a0 = object (Variant*), a1 = method name (const char*), a2 = method length (size_t),
				//             a3 = args array (const Variant*), a4 = argcount (size_t), a5 = result (Variant*), a7 = syscall number
				if (call_ret) {
					int ret_addr = p_code_ptr[p_ip + 4 + instr_arg_count];
					String ret_var = resolve_address(ret_addr);
					r_code += "        Variant call_result;\n";
					r_code += "        register const Variant* object asm(\"a0\") = &" + base_var + ";\n";
					r_code += "        register const char* method_ptr asm(\"a1\") = method_cstr;\n";
					r_code += "        register size_t method_size asm(\"a2\") = method_len;\n";
					r_code += "        register const Variant* args_ptr asm(\"a3\") = call_args;\n";
					r_code += "        register size_t argcount_reg asm(\"a4\") = " + itos(argc) + ";\n";
					r_code += "        register Variant* ret_ptr asm(\"a5\") = &call_result;\n";
					r_code += "        register int syscall_number asm(\"a7\") = ECALL_VCALL;\n";
					r_code += "        __asm__ volatile(\"ecall\" : \"=m\"(*ret_ptr) : \"r\"(object), \"m\"(*object), \"r\"(method_ptr), \"r\"(method_size), \"m\"(*method_ptr), \"r\"(args_ptr), \"r\"(argcount_reg), \"r\"(ret_ptr), \"m\"(*args_ptr), \"r\"(syscall_number));\n";
					r_code += "        " + ret_var + " = call_result;\n";
				} else {
					r_code += "        register const Variant* object asm(\"a0\") = &" + base_var + ";\n";
					r_code += "        register const char* method_ptr asm(\"a1\") = method_cstr;\n";
					r_code += "        register size_t method_size asm(\"a2\") = method_len;\n";
					r_code += "        register const Variant* args_ptr asm(\"a3\") = call_args;\n";
					r_code += "        register size_t argcount_reg asm(\"a4\") = " + itos(argc) + ";\n";
					r_code += "        register Variant* ret_ptr asm(\"a5\") = 0;\n";
					r_code += "        register int syscall_number asm(\"a7\") = ECALL_VCALL;\n";
					r_code += "        __asm__ volatile(\"ecall\" : : \"r\"(object), \"m\"(*object), \"r\"(method_ptr), \"r\"(method_size), \"m\"(*method_ptr), \"r\"(args_ptr), \"r\"(argcount_reg), \"r\"(ret_ptr), \"m\"(*args_ptr), \"r\"(syscall_number));\n";
				}
				r_code += "    }\n";

				// Advance IP: opcode(1) + instr_arg_count(1) + instruction_args(instr_arg_count) + argc(1) + methodname_idx(1) + [ret_addr(1) if RETURN]
				p_ip += 1 + 1 + instr_arg_count + 1 + 1 + (call_ret ? 1 : 0);
				return;
			}
			case GDScriptFunction::OPCODE_CALL_METHOD_BIND:
			case GDScriptFunction::OPCODE_CALL_METHOD_BIND_RET: {
				// OPCODE_CALL_METHOD_BIND/OPCODE_CALL_METHOD_BIND_RET: Similar to CALL but uses method_bind_idx instead of methodname_idx
				// For Phase 2: Use fallback (requires MethodBind* access which is complex)
				// TODO: Implement method bind calls in Phase 2+ using ECALL_OBJ_CALLP or similar
				break;
			}
			default:
				// Unsupported opcode - use fallback
				break;
		}
	}

	// Phase 1: All unsupported opcodes generate fallback calls
	generate_fallback_call(p_opcode, p_code_ptr, 0, r_code);

	// Advance instruction pointer
	// In real implementation, we'd decode the opcode to know how many arguments it has
	// For now, just advance by 1 (opcode only) - fallback will handle the rest
	p_ip += 1;
}

void GDScriptBytecodeCCodeGenerator::generate_fallback_call(int p_opcode, const int *p_args, int p_arg_count, StringBuilder &r_code) {
	// Generate call to fallback function: gdscript_vm_fallback(opcode, instance, stack, ip)
	r_code += "    gdscript_vm_fallback(" + itos(p_opcode) + ", instance, stack, ip);\n";
}

void GDScriptBytecodeCCodeGenerator::generate_syscall(int p_syscall_number, StringBuilder &r_code) {
	// Generate inline assembly syscall
	// __asm__ volatile ("li a7, %0\n ecall\n" : : "i" (ECALL_NUMBER) : "a7");
	r_code += "    __asm__ volatile (\"li a7, %0\\n ecall\\n\" : : \"i\" (" + itos(p_syscall_number) + ") : \"a7\");\n";
}

int GDScriptBytecodeCCodeGenerator::get_address_type(int p_address) {
	// Extract address type from encoded address
	// Address format: (type << ADDR_BITS) | value
	return (p_address >> GDScriptFunction::ADDR_BITS) & 0x3;
}

int GDScriptBytecodeCCodeGenerator::get_address_value(int p_address) {
	// Extract address value from encoded address
	return p_address & GDScriptFunction::ADDR_MASK;
}

String GDScriptBytecodeCCodeGenerator::get_stack_var_name(int p_slot) {
	return "stack[" + itos(p_slot) + "]";
}

String GDScriptBytecodeCCodeGenerator::get_constant_var_name(int p_index) {
	return "constant_" + itos(p_index);
}

String GDScriptBytecodeCCodeGenerator::resolve_address(int p_address) {
	// Check for fixed addresses first
	if (p_address == GDScriptFunction::ADDR_SELF) {
		return "stack[0]";
	} else if (p_address == GDScriptFunction::ADDR_CLASS) {
		return "stack[1]";
	} else if (p_address == GDScriptFunction::ADDR_NIL) {
		return "stack[2]";
	}

	int addr_type = get_address_type(p_address);
	int addr_value = get_address_value(p_address);

	switch (addr_type) {
		case GDScriptFunction::ADDR_TYPE_STACK: {
			// Stack address - direct stack access
			return get_stack_var_name(addr_value);
		}
		case GDScriptFunction::ADDR_TYPE_CONSTANT: {
			// Constant address - use constants array passed as parameter
			// Handle NULL pointer case (constants not shared)
			return "(constants != NULL ? constants[" + itos(addr_value) + "] : Variant())";
		}
		case GDScriptFunction::ADDR_TYPE_MEMBER: {
			// Member address - use member variables via instance
			// For Phase 2, members require instance access which is complex
			// Use fallback for member access for now
			// TODO: Implement member access in Phase 2+
			return "stack[0]"; // Fallback - will trigger fallback if used
		}
		default:
			return "stack[0]"; // Fallback
	}
}

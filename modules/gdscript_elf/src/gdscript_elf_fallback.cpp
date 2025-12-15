/**************************************************************************/
/*  gdscript_elf_fallback.cpp                                             */
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

#include "gdscript_elf_fallback.h"

#include "core/error/error_macros.h"
#include "modules/gdscript/gdscript_function.h"

HashMap<int, uint64_t> GDScriptELFFallback::fallback_counts;

// C interface for ELF-compiled code to call back to VM
extern "C" void gdscript_vm_fallback(int p_opcode, void *p_instance, void *p_stack, int p_ip) {
	// This is called from ELF-compiled C code
	// For now, just record the fallback - actual VM call will be implemented
	// when we integrate ELF execution with sandbox
	GDScriptELFFallback::record_fallback_opcode(p_opcode);

	// TODO: Convert stack pointer to Variant array and call original function
	// This requires integration with sandbox execution context
}

Variant GDScriptELFFallback::call_original_function(
		GDScriptFunction *p_function,
		GDScriptInstance *p_instance,
		const Variant **p_args,
		int p_argcount,
		Callable::CallError &r_err,
		GDScriptFunction::CallState *p_state) {
	ERR_FAIL_NULL_V(p_function, Variant());

	// Simply delegate to original GDScriptFunction::call()
	// This maintains the same execution model - stack/registers work same
	return p_function->call(p_instance, p_args, p_argcount, r_err, p_state);
}

void GDScriptELFFallback::record_fallback_opcode(int p_opcode) {
	// Get current count (0 if doesn't exist) and increment
	uint64_t current_count = fallback_counts.has(p_opcode) ? fallback_counts[p_opcode] : 0;
	fallback_counts[p_opcode] = current_count + 1;
}

HashMap<int, uint64_t> GDScriptELFFallback::get_fallback_statistics() {
	return fallback_counts;
}

void GDScriptELFFallback::reset_statistics() {
	fallback_counts.clear();
}

bool GDScriptELFFallback::is_opcode_supported(int p_opcode) {
	// Phase 2: Support simple opcodes
	switch (p_opcode) {
		case GDScriptFunction::OPCODE_ASSIGN:
		case GDScriptFunction::OPCODE_ASSIGN_NULL:
		case GDScriptFunction::OPCODE_ASSIGN_TRUE:
		case GDScriptFunction::OPCODE_ASSIGN_FALSE:
		case GDScriptFunction::OPCODE_JUMP:
		case GDScriptFunction::OPCODE_JUMP_IF:
		case GDScriptFunction::OPCODE_JUMP_IF_NOT:
		case GDScriptFunction::OPCODE_OPERATOR_VALIDATED:
		case GDScriptFunction::OPCODE_RETURN:
		case GDScriptFunction::OPCODE_LINE: // Metadata opcode, just skip
		case GDScriptFunction::OPCODE_END: // End of function
		case GDScriptFunction::OPCODE_GET_MEMBER: // Property access via syscall
		case GDScriptFunction::OPCODE_SET_MEMBER: // Property access via syscall
		case GDScriptFunction::OPCODE_CALL: // Method call via syscall
		case GDScriptFunction::OPCODE_CALL_RETURN: // Method call with return via syscall
			return true;
		default:
			return false;
	}
}

/**************************************************************************/
/*  gdscript_function_wrapper.cpp                                         */
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

#include "gdscript_function_wrapper.h"

#include "gdscript_elf_fallback.h"
#include "modules/gdscript/gdscript.h"
#include "modules/gdscript/gdscript_function.h"
#include "modules/sandbox/src/sandbox.h"

// Static member definition
HashMap<GDScriptInstance *, Sandbox *> GDScriptFunctionWrapper::instance_sandboxes;

GDScriptFunctionWrapper::GDScriptFunctionWrapper() {
	original_function = nullptr;
	elf_binary.clear();
}

GDScriptFunctionWrapper::~GDScriptFunctionWrapper() {
	// Don't delete original_function - it's managed elsewhere
	original_function = nullptr;
	elf_binary.clear();
}

void GDScriptFunctionWrapper::set_original_function(GDScriptFunction *p_function) {
	original_function = p_function;
}

// Main execution method - intercepts GDScriptFunction::call()
// Phase 3: ELF execution integration with sandbox
Variant GDScriptFunctionWrapper::call(GDScriptInstance *p_instance, const Variant **p_args, int p_argcount, Callable::CallError &r_err, GDScriptFunction::CallState *p_state) {
	ERR_FAIL_NULL_V(original_function, Variant());

	// Phase 3: Try to execute ELF code if available
	if (has_elf_code() && !elf_binary.is_empty()) {
		// Get or create sandbox for this instance
		Sandbox *sandbox = get_or_create_sandbox(p_instance);
		if (sandbox == nullptr) {
			// Fallback to original VM if sandbox creation fails
			ERR_PRINT("GDScriptFunctionWrapper: Failed to get sandbox, falling back to VM");
			return original_function->call(p_instance, p_args, p_argcount, r_err, p_state);
		}

		// Load ELF binary into sandbox if not already loaded
		if (!sandbox->has_program_loaded()) {
			sandbox->load_buffer(elf_binary);
			if (!sandbox->has_program_loaded()) {
				// Fallback to original VM if loading fails
				ERR_PRINT("GDScriptFunctionWrapper: Failed to load ELF binary, falling back to VM");
				return original_function->call(p_instance, p_args, p_argcount, r_err, p_state);
			}
		}

		// Resolve function address if not cached
		if (cached_function_address == 0) {
			// Generate function name matching C code generation
			String func_name = original_function->get_name().operator String();
			func_name = func_name.replace(".", "_").replace(" ", "_");
			String symbol_name = "gdscript_" + func_name;

			cached_function_address = sandbox->address_of(symbol_name);
			if (cached_function_address == 0) {
				// Fallback to original VM if function not found
				ERR_PRINT("GDScriptFunctionWrapper: Function symbol not found: " + symbol_name + ", falling back to VM");
				return original_function->call(p_instance, p_args, p_argcount, r_err, p_state);
			}
		}

		// Prepare arguments for sandbox call
		// Option B: Store constants/operator_funcs in sandbox memory and pass addresses
		// Extended args array: [result_ptr, arg0, arg1, ..., argN, instance_ptr, constants_addr, operator_funcs_addr]
		// Note: result_ptr is passed as first arg (A0) for return value storage

		bool needs_constants = !original_function->constants.is_empty();
		bool needs_operator_funcs = !original_function->operator_funcs.is_empty();

		// Share constants array in sandbox memory if needed
		// Note: We use share_byte_array with PackedByteArray wrapper since share_array_internal is private
		if (needs_constants && cached_constants_address == 0) {
			if (!original_function->constants.is_empty()) {
				// Create PackedByteArray wrapper for constants array
				// We share the raw bytes, and the C code will cast back to Variant*
				PackedByteArray constants_bytes;
				constants_bytes.resize(original_function->constants.size() * sizeof(Variant));
				memcpy(constants_bytes.ptrw(), original_function->constants.ptr(), constants_bytes.size());

				gaddr_t constants_addr = sandbox->share_byte_array(false, constants_bytes); // Read-only
				if (constants_addr == 0) {
					ERR_PRINT("GDScriptFunctionWrapper: Failed to share constants array, falling back to VM");
					return original_function->call(p_instance, p_args, p_argcount, r_err, p_state);
				}
				cached_constants_address = constants_addr;
			}
		}

		// Share operator_funcs array in sandbox memory if needed
		if (needs_operator_funcs && cached_operator_funcs_address == 0) {
			if (!original_function->operator_funcs.is_empty()) {
				// Create PackedByteArray wrapper for operator_funcs array
				PackedByteArray operator_funcs_bytes;
				operator_funcs_bytes.resize(original_function->operator_funcs.size() * sizeof(Variant::ValidatedOperatorEvaluator));
				memcpy(operator_funcs_bytes.ptrw(), original_function->operator_funcs.ptr(), operator_funcs_bytes.size());

				gaddr_t operator_funcs_addr = sandbox->share_byte_array(false, operator_funcs_bytes); // Read-only
				if (operator_funcs_addr == 0) {
					ERR_PRINT("GDScriptFunctionWrapper: Failed to share operator_funcs array, falling back to VM");
					return original_function->call(p_instance, p_args, p_argcount, r_err, p_state);
				}
				cached_operator_funcs_address = operator_funcs_addr;
			}
		}

		// Create extended args array
		// Structure: [result_ptr, arg0, arg1, ..., argN, instance_ptr, constants_addr, operator_funcs_addr]
		// Total: 1 (result) + p_argcount + 3 (instance, constants, operator_funcs) = p_argcount + 4
		int extended_argcount = p_argcount + 4;
		Vector<Variant> extended_args_storage;
		extended_args_storage.resize(extended_argcount);
		const Variant **extended_args = (const Variant **)alloca(sizeof(Variant *) * extended_argcount);

		// First arg: result pointer (will be filled by ELF function)
		Variant result_variant;
		extended_args[0] = &result_variant;

		// Copy original arguments
		for (int i = 0; i < p_argcount; i++) {
			extended_args[i + 1] = p_args[i];
		}

		// Add instance pointer as Variant integer
		extended_args_storage.write[p_argcount + 1] = Variant((int64_t)(uintptr_t)p_instance);
		extended_args[p_argcount + 1] = &extended_args_storage[p_argcount + 1];

		// Add constants address as Variant integer
		if (needs_constants) {
			extended_args_storage.write[p_argcount + 2] = Variant((int64_t)cached_constants_address);
			extended_args[p_argcount + 2] = &extended_args_storage[p_argcount + 2];
		} else {
			extended_args_storage.write[p_argcount + 2] = Variant((int64_t)0);
			extended_args[p_argcount + 2] = &extended_args_storage[p_argcount + 2];
		}

		// Add operator_funcs address as Variant integer
		if (needs_operator_funcs) {
			extended_args_storage.write[p_argcount + 3] = Variant((int64_t)cached_operator_funcs_address);
			extended_args[p_argcount + 3] = &extended_args_storage[p_argcount + 3];
		} else {
			extended_args_storage.write[p_argcount + 3] = Variant((int64_t)0);
			extended_args[p_argcount + 3] = &extended_args_storage[p_argcount + 3];
		}

		// Call function via sandbox with extended args
		Callable::CallError call_error;
		Variant result = sandbox->vmcall_address(cached_function_address, extended_args, extended_argcount, call_error);

		if (call_error.error != Callable::CallError::CALL_OK) {
			// Fallback to original VM on error
			ERR_PRINT("GDScriptFunctionWrapper: vmcall_address failed with error " + itos(call_error.error) + ", falling back to VM");
			return original_function->call(p_instance, p_args, p_argcount, r_err, p_state);
		}

		// Extract result from result_variant (stored in first arg)
		// The ELF function should have written the result there
		// Note: result_variant is passed as args[0], so *result in ELF code updates it
		result = result_variant;

		// Update r_err with call_error
		r_err = call_error;
		return result;
	}

	// Fallback to original VM if no ELF code
	return original_function->call(p_instance, p_args, p_argcount, r_err, p_state);
}

Sandbox *GDScriptFunctionWrapper::get_or_create_sandbox(GDScriptInstance *p_instance) {
	ERR_FAIL_NULL_V(p_instance, nullptr);

	// Check if sandbox already exists for this instance
	if (instance_sandboxes.has(p_instance)) {
		Sandbox *existing = instance_sandboxes[p_instance];
		if (existing != nullptr) {
			return existing;
		}
		// If invalid, remove from map and create new one
		instance_sandboxes.erase(p_instance);
	}

	// Create new sandbox instance
	Sandbox *sandbox = memnew(Sandbox);
	if (sandbox == nullptr) {
		ERR_PRINT("GDScriptFunctionWrapper: Failed to create Sandbox instance");
		return nullptr;
	}

	// Store sandbox for this instance
	instance_sandboxes[p_instance] = sandbox;

	return sandbox;
}

void GDScriptFunctionWrapper::cleanup_sandbox(GDScriptInstance *p_instance) {
	if (p_instance && instance_sandboxes.has(p_instance)) {
		instance_sandboxes.erase(p_instance);
	}
}

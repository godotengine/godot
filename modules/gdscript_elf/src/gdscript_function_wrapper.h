/**************************************************************************/
/*  gdscript_function_wrapper.h                                           */
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

#include "core/templates/hash_map.h"
#include "core/variant/callable.h"
#include "core/variant/variant.h"
#include "modules/gdscript/gdscript_function.h"

// Forward declarations
class GDScriptInstance;
class Sandbox;

// Wrapper for GDScriptFunction that implements strangler vine pattern
// Phase 0: 100% pass-through to original GDScriptFunction::call()
// Future phases: Intercept call() to use ELF-compiled code if available, fallback to original
class GDScriptFunctionWrapper {
public:
	// Pointer to the original GDScriptFunction instance
	// We keep it alive and delegate all calls to it
	GDScriptFunction *original_function = nullptr;

	// ELF-compiled code (stores ELF binary)
	PackedByteArray elf_binary;

	// Cached function address (resolved from ELF symbol table)
	// Using uint64_t to match gaddr_t type from sandbox
	// gaddr_t is typedef'd as riscv::address_type which is uint64_t
	uint64_t cached_function_address = 0;

	// Cached shared memory addresses for constants and operator_funcs
	// These are shared in sandbox memory and cached to avoid re-sharing on every call
	uint64_t cached_constants_address = 0;
	uint64_t cached_operator_funcs_address = 0;

	GDScriptFunctionWrapper();
	~GDScriptFunctionWrapper();

	// Initialize the wrapper with the original GDScriptFunction instance
	void set_original_function(GDScriptFunction *p_function);

	// Get the original function (for direct access if needed)
	GDScriptFunction *get_original_function() const { return original_function; }

	// Set compiled ELF binary
	void set_elf_binary(const PackedByteArray &p_elf) { elf_binary = p_elf; }

	// Get compiled ELF binary
	PackedByteArray get_elf_binary() const { return elf_binary; }

	// Main execution method - intercepts GDScriptFunction::call()
	// Phase 0: Just delegates to original
	// Phase 1+: Uses ELF if available, falls back to original for unsupported opcodes
	Variant call(GDScriptInstance *p_instance, const Variant **p_args, int p_argcount, Callable::CallError &r_err, GDScriptFunction::CallState *p_state = nullptr);

	// Check if ELF compilation is available for this function
	bool has_elf_code() const { return !elf_binary.is_empty(); }

	// Get or create sandbox instance for a GDScriptInstance
	// Returns null if sandbox creation fails
	static Sandbox *get_or_create_sandbox(GDScriptInstance *p_instance);

	// Clean up sandbox for a GDScriptInstance (called when instance is destroyed)
	static void cleanup_sandbox(GDScriptInstance *p_instance);

private:
	// Static map to associate GDScriptInstance with Sandbox
	// Key: GDScriptInstance pointer, Value: Sandbox pointer (Node, managed by scene tree)
	static HashMap<GDScriptInstance *, Sandbox *> instance_sandboxes;
};

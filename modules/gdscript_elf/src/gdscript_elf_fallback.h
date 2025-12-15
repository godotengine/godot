/**************************************************************************/
/*  gdscript_elf_fallback.h                                               */
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
#include "core/templates/vector.h"
#include "core/variant/callable.h"
#include "core/variant/variant.h"
#include "modules/gdscript/gdscript_function.h"

// Forward declarations
class GDScriptInstance;

// Fallback mechanism to original GDScriptFunction for unsupported opcodes
// Maintains same stack/register model (x/y arrays work same)
// Bridges between ELF-compiled code and original GDScript VM
class GDScriptELFFallback {
public:
	// Call original GDScriptFunction for unsupported opcode
	// This is called from ELF-compiled code when encountering unsupported opcodes
	// C interface: extern "C" void gdscript_vm_fallback(int opcode, void* instance, void* stack, int ip);
	static void gdscript_vm_fallback(int p_opcode, void *p_instance, void *p_stack, int p_ip);

	// C++ interface for calling original function
	static Variant call_original_function(
			GDScriptFunction *p_function,
			GDScriptInstance *p_instance,
			const Variant **p_args,
			int p_argcount,
			Callable::CallError &r_err,
			GDScriptFunction::CallState *p_state = nullptr);

	// Track migration progress - which opcodes use fallback
	static void record_fallback_opcode(int p_opcode);
	static HashMap<int, uint64_t> get_fallback_statistics();
	static void reset_statistics();

	// Check if an opcode is supported (doesn't need fallback)
	static bool is_opcode_supported(int p_opcode);

private:
	// Statistics: opcode -> fallback count
	static HashMap<int, uint64_t> fallback_counts;
};

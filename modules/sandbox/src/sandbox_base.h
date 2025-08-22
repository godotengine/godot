/**************************************************************************/
/*  sandbox_base.h                                                        */
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
#include "scene/main/node.h"

// Forward declarations
class ELFScript;

// Base Sandbox implementation providing default/dummy implementations
// This allows the Sandbox class to work without libriscv dependencies for testing
class SandboxBase : public Node {
	GDCLASS(SandboxBase, Node);

protected:
	// Internal state for base implementation
	uint32_t max_refs = 100;
	uint32_t memory_max = 16;
	uint32_t instructions_max = 8000;
	int64_t allocations_max = 4000;
	bool unboxed_arguments = false;
	bool precise_simulation = false;
	bool profiling = false;
	bool restrictions = false;
	bool program_loaded = false;

	// Statistics
	uint64_t calls_made = 0;
	uint64_t exceptions = 0;
	uint64_t timeouts = 0;
	int64_t heap_usage = 1152;
	int64_t heap_chunk_count = 5;
	int64_t heap_allocation_counter = 10;
	int64_t heap_deallocation_counter = 8;

	// Static statistics
	static uint64_t global_calls_made;
	static uint64_t global_exceptions;
	static uint64_t global_timeouts;
	static uint64_t global_instance_count;
	static double accumulated_startup_time;

	// Function simulation for base implementation
	HashMap<String, bool> available_functions;
	HashMap<String, uint64_t> function_addresses;

	// Program data
	PackedByteArray program_bytes;
	Node *tree_base;

public:
	// Constants matching real Sandbox
	static const uint32_t MAX_REFS = 100;
	static const uint32_t MAX_VMEM = 16;
	static const uint32_t MAX_INSTRUCTIONS = 8000;
	static const int64_t MAX_HEAP_ALLOCS = 4000;

	SandboxBase();
	virtual ~SandboxBase();

	// Basic property management (virtual for override)
	virtual void set_max_refs(uint32_t p_max) { max_refs = p_max; }
	virtual uint32_t get_max_refs() const { return max_refs; }

	virtual void set_memory_max(uint32_t p_max) { memory_max = p_max; }
	virtual uint32_t get_memory_max() const { return memory_max; }

	virtual void set_instructions_max(int64_t p_max) { instructions_max = p_max; }
	virtual int64_t get_instructions_max() const { return instructions_max; }

	virtual void set_allocations_max(int64_t p_max) { allocations_max = p_max; }
	virtual int64_t get_allocations_max() const { return allocations_max; }

	virtual void set_unboxed_arguments(bool p_enable) { unboxed_arguments = p_enable; }
	virtual bool get_unboxed_arguments() const { return unboxed_arguments; }

	virtual void set_precise_simulation(bool p_enable) { precise_simulation = p_enable; }
	virtual bool get_precise_simulation() const { return precise_simulation; }

	virtual void set_profiling(bool p_enable) { profiling = p_enable; }
	virtual bool get_profiling() const { return profiling; }

	virtual void set_restrictions(bool p_enable) { restrictions = p_enable; }
	virtual bool get_restrictions() const { return restrictions; }

	// Program management (virtual for override)
	virtual void load_buffer(const PackedByteArray &p_buffer);
	virtual bool has_program_loaded() const { return program_loaded; }
	virtual void set_program(Ref<ELFScript> program);
	virtual Ref<ELFScript> get_program();

	virtual void reset(bool p_unload = false);

	// Function management (virtual for override)
	virtual bool has_function(const StringName &p_function) const;
	virtual uint64_t address_of(const String &p_symbol) const;
	virtual String lookup_address(uint64_t p_address) const;

	// VM call simulation (virtual for override)
	virtual Variant vmcall_fn(const StringName &p_function, const Variant **p_args, int p_arg_count, Callable::CallError &r_error);

	// Statistics (virtual for override)
	virtual unsigned get_calls_made() const { return calls_made; }
	virtual unsigned get_exceptions() const { return exceptions; }
	virtual unsigned get_timeouts() const { return timeouts; }
	virtual int64_t get_heap_usage() const { return heap_usage; }
	virtual int64_t get_heap_chunk_count() const { return heap_chunk_count; }
	virtual int64_t get_heap_allocation_counter() const { return heap_allocation_counter; }
	virtual int64_t get_heap_deallocation_counter() const { return heap_deallocation_counter; }

	// Global statistics (static)
	static uint64_t get_global_calls_made() { return global_calls_made; }
	static uint64_t get_global_exceptions() { return global_exceptions; }
	static uint64_t get_global_timeouts() { return global_timeouts; }
	static uint64_t get_global_instance_count() { return global_instance_count; }
	static double get_accumulated_startup_time() { return accumulated_startup_time; }

	// Binary translation (virtual for override)
	virtual bool is_binary_translated() const { return false; }
	virtual bool is_jit() const { return false; }
	static bool is_jit_enabled() { return false; }
	static bool has_feature_jit() { return false; }

	// Tree base management (virtual for override)
	virtual void set_tree_base(Node *p_base) { tree_base = p_base; }
	virtual Node *get_tree_base() const { return tree_base; }

	// Object management (virtual for override)
	virtual bool is_allowed_object(Object *p_object) const { return !restrictions; }
	virtual void add_allowed_object(Object *p_object) { /* base implementation */ }
	virtual void remove_allowed_object(Object *p_object) { /* base implementation */ }
	virtual void clear_allowed_objects() { /* base implementation */ }

	// Profiling (virtual for override)
	virtual void enable_profiling(bool p_enable, int p_interval = 1000) {
		profiling = p_enable;
	}

protected:
	static void _bind_methods();
	void setup_base_functions();
};

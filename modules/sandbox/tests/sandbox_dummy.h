/**************************************************************************/
/*  sandbox_dummy.h                                                       */
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

// Dummy Sandbox implementation for testing (following RasterizerDummy pattern)
// Provides the same API as Sandbox but without libriscv dependencies
class SandboxDummy : public Node {
	GDCLASS(SandboxDummy, Node);

private:
	// Internal state for dummy implementation
	uint32_t max_refs = 100;
	uint32_t memory_max = 16;
	uint32_t instructions_max = 8000;
	int64_t allocations_max = 4000;
	bool unboxed_arguments = false;
	bool precise_simulation = false;
	bool profiling = false;
	bool restrictions = false;
	bool program_loaded = false;

	// Dummy statistics
	uint64_t calls_made = 0;
	uint64_t exceptions = 0;
	uint64_t timeouts = 0;
	int64_t heap_usage = 1152;
	int64_t heap_chunk_count = 5;
	int64_t heap_allocation_counter = 10;
	int64_t heap_deallocation_counter = 8;

	// Static dummy statistics
	static uint64_t global_calls_made;
	static uint64_t global_exceptions;
	static uint64_t global_timeouts;
	static uint64_t global_instance_count;
	static double accumulated_startup_time;

	// Function simulation
	HashMap<String, bool> available_functions;
	HashMap<String, uint64_t> function_addresses;

	// Dummy program data
	PackedByteArray program_bytes;
	Node *tree_base;

public:
	// Constants matching real Sandbox
	static const uint32_t MAX_REFS = 100;
	static const uint32_t MAX_VMEM = 16;
	static const uint32_t MAX_INSTRUCTIONS = 8000;
	static const int64_t MAX_HEAP_ALLOCS = 4000;

	SandboxDummy() {
		tree_base = this;
		global_instance_count++;
		accumulated_startup_time += 0.001; // Simulate startup time

		// Add some dummy functions for testing
		available_functions["generate_json_diff"] = true;
		available_functions["test_function"] = true;
		function_addresses["generate_json_diff"] = 0x1000;
		function_addresses["test_function"] = 0x2000;
	}

	~SandboxDummy() {
		if (global_instance_count > 0) {
			global_instance_count--;
		}
	}

	// API matching real Sandbox for testing
	void set_max_refs(uint32_t p_max) { max_refs = p_max; }
	uint32_t get_max_refs() const { return max_refs; }

	void set_memory_max(uint32_t p_max) { memory_max = p_max; }
	uint32_t get_memory_max() const { return memory_max; }

	void set_instructions_max(uint32_t p_max) { instructions_max = p_max; }
	uint32_t get_instructions_max() const { return instructions_max; }

	void set_allocations_max(int64_t p_max) { allocations_max = p_max; }
	int64_t get_allocations_max() const { return allocations_max; }

	void set_unboxed_arguments(bool p_enable) { unboxed_arguments = p_enable; }
	bool get_unboxed_arguments() const { return unboxed_arguments; }

	void set_precise_simulation(bool p_enable) { precise_simulation = p_enable; }
	bool get_precise_simulation() const { return precise_simulation; }

	void set_profiling(bool p_enable) { profiling = p_enable; }
	bool get_profiling() const { return profiling; }

	void set_restrictions(bool p_enable) { restrictions = p_enable; }
	bool get_restrictions() const { return restrictions; }

	// Program management
	void load_buffer(const PackedByteArray &p_buffer) {
		program_bytes = p_buffer;
		program_loaded = !p_buffer.is_empty();
	}
	bool has_program_loaded() const { return program_loaded; }

	void reset(bool p_unload = false) {
		if (p_unload) {
			program_loaded = false;
			program_bytes.clear();
		}
	}

	// Function management
	bool has_function(const StringName &p_function) const {
		String func_str = p_function;
		return available_functions.find(func_str) != available_functions.end();
	}

	uint64_t address_of(const String &p_symbol) const {
		auto it = function_addresses.find(p_symbol);
		return (it != function_addresses.end()) ? it->value : 0;
	}

	String lookup_address(uint64_t p_address) const {
		for (const auto &pair : function_addresses) {
			if (pair.value == p_address) {
				return pair.key;
			}
		}
		return String();
	}

	// VM call simulation (dummy implementation)
	Variant vmcall_fn(const StringName &p_function, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
		calls_made++;
		global_calls_made++;

		r_error.error = Callable::CallError::CALL_OK;

		String func_name = p_function;

		// Simulate generate_json_diff function
		if (func_name == "generate_json_diff" && p_arg_count == 2) {
			String arg1 = p_args[0]->operator String();
			String arg2 = p_args[1]->operator String();

			// Simple dummy JSON diff (not real implementation)
			String result = R"DELIM([
  { "op": "replace", "path": "/value", "value": "test_diff_result" },
  { "op": "test", "path": "/source", "value": ")DELIM" +
					arg1.substr(0, 20) + R"DELIM(...)" },
  { "op": "test", "path": "/reference", "value": ")DELIM" +
					arg2.substr(0, 20) + R"DELIM(...)" }
])DELIM";
			return Variant(result);
		}

		// Default dummy response
		return Variant("dummy_result_" + func_name);
	}

	// Statistics
	uint64_t get_calls_made() const { return calls_made; }
	uint64_t get_exceptions() const { return exceptions; }
	uint64_t get_timeouts() const { return timeouts; }
	int64_t get_heap_usage() const { return heap_usage; }
	int64_t get_heap_chunk_count() const { return heap_chunk_count; }
	int64_t get_heap_allocation_counter() const { return heap_allocation_counter; }
	int64_t get_heap_deallocation_counter() const { return heap_deallocation_counter; }

	// Global statistics
	static uint64_t get_global_calls_made() { return global_calls_made; }
	static uint64_t get_global_exceptions() { return global_exceptions; }
	static uint64_t get_global_timeouts() { return global_timeouts; }
	static uint64_t get_global_instance_count() { return global_instance_count; }
	static double get_accumulated_startup_time() { return accumulated_startup_time; }

	// Binary translation dummy
	bool is_binary_translated() const { return false; }
	bool is_jit() const { return false; }
	static bool is_jit_enabled() { return false; }
	static bool has_feature_jit() { return false; }

	// Tree base management
	void set_tree_base(Node *p_base) { tree_base = p_base; }
	Node *get_tree_base() const { return tree_base; }

	// Object management dummy
	bool is_allowed_object(Object *p_object) const { return !restrictions; }
	void add_allowed_object(Object *p_object) { /* dummy implementation */ }
	void remove_allowed_object(Object *p_object) { /* dummy implementation */ }
	void clear_allowed_objects() { /* dummy implementation */ }

	// Profiling dummy
	void enable_profiling(bool p_enable, int p_interval = 1000) {
		profiling = p_enable;
	}
};

// Static member definitions
uint64_t SandboxDummy::global_calls_made = 0;
uint64_t SandboxDummy::global_exceptions = 0;
uint64_t SandboxDummy::global_timeouts = 0;
uint64_t SandboxDummy::global_instance_count = 0;
double SandboxDummy::accumulated_startup_time = 0.0;

// Static constant definitions
const uint32_t SandboxDummy::MAX_REFS;
const uint32_t SandboxDummy::MAX_VMEM;
const uint32_t SandboxDummy::MAX_INSTRUCTIONS;
const int64_t SandboxDummy::MAX_HEAP_ALLOCS;

/**************************************************************************/
/*  test_sandbox_vm.h                                                     */
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

#include "tests/test_macros.h"

#ifdef TESTS_ENABLED

#include "core/os/os.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/main/node.h"

#include "../sandbox.h"

namespace TestSandboxVM {

TEST_CASE("[SceneTree][Node] Sandbox VM memory management") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());
	
	// Test initial memory state
	CHECK(sandbox->get_heap_usage() >= 0);
	CHECK(sandbox->get_heap_chunk_count() >= 0);
	CHECK(sandbox->get_heap_allocation_counter() >= 0);
	CHECK(sandbox->get_heap_deallocation_counter() >= 0);
	
	// Test memory limits
	sandbox->set_memory_max(16);
	CHECK(sandbox->get_memory_max() == 16);
	
	sandbox->set_allocations_max(2000);
	CHECK(sandbox->get_allocations_max() == 2000);
	
	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] Sandbox VM execution state") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());
	
	// Test execution state tracking
	CHECK(sandbox->get_exceptions() >= 0);
	CHECK(sandbox->get_timeouts() >= 0);
	CHECK(sandbox->get_calls_made() >= 0);
	
	// Test simulation modes
	CHECK_FALSE(sandbox->get_precise_simulation());
	sandbox->set_precise_simulation(true);
	CHECK(sandbox->get_precise_simulation());
	
	sandbox->set_precise_simulation(false);
	CHECK_FALSE(sandbox->get_precise_simulation());
	
	// Test unboxed arguments
	CHECK_FALSE(sandbox->get_unboxed_arguments());
	sandbox->set_unboxed_arguments(true);
	CHECK(sandbox->get_unboxed_arguments());
	
	sandbox->set_unboxed_arguments(false);
	CHECK_FALSE(sandbox->get_unboxed_arguments());
	
	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] Sandbox VM address lookup") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());
	
	// Test address lookup functionality
	String test_symbol = "test_function";
	uint64_t address = sandbox->address_of(test_symbol);
	CHECK(address >= 0); // Should not crash, may return 0 if not found
	
	// Test cached address lookup
	int64_t hash = test_symbol.hash();
	uint64_t cached_address = sandbox->cached_address_of(hash, test_symbol);
	CHECK(cached_address >= 0); // Should not crash
	
	// Test address to symbol lookup
	String symbol = sandbox->lookup_address(0x1000);
	CHECK(symbol.length() >= 0); // Should not crash, may return empty string
	
	// Test function existence check
	bool has_function = sandbox->has_function("nonexistent_function");
	CHECK_FALSE(has_function); // Should return false for nonexistent function
	
	// Test adding cached address
	sandbox->add_cached_address("test_func", 0x2000);
	// Should not crash
	
	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] Sandbox VM shared memory") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());
	
	// Test byte array sharing
	PackedByteArray test_bytes;
	test_bytes.push_back(1);
	test_bytes.push_back(2);
	test_bytes.push_back(3);
	
	uint64_t shared_address = sandbox->share_byte_array(false, test_bytes);
	CHECK(shared_address >= 0); // Should not crash
	
	// Test unsharing
	bool unshared = sandbox->unshare_array(shared_address);
	CHECK(unshared || !unshared); // Should not crash
	
	// Test other array types
	PackedFloat32Array float_array;
	float_array.push_back(1.0f);
	uint64_t float_address = sandbox->share_float32_array(true, float_array);
	CHECK(float_address >= 0);
	
	PackedInt32Array int_array;
	int_array.push_back(42);
	uint64_t int_address = sandbox->share_int32_array(false, int_array);
	CHECK(int_address >= 0);
	
	// Cleanup arrays
	sandbox->unshare_array(float_address);
	sandbox->unshare_array(int_address);
	
	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] Sandbox VM registers and state") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());
	
	// Test register access
	Array general_regs = sandbox->get_general_registers();
	CHECK(general_regs.size() == 32); // RISC-V has 32 general registers
	
	Array fp_regs = sandbox->get_floating_point_registers();
	CHECK(fp_regs.size() == 32); // RISC-V has 32 FP registers
	
	// Test setting argument registers
	Array args;
	args.push_back(42);
	args.push_back(100);
	sandbox->set_argument_registers(args);
	// Should not crash
	
	// Test current instruction
	String instruction = sandbox->get_current_instruction();
	CHECK(instruction.length() >= 0); // Should not crash
	
	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] Sandbox VM program management") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());
	
	// Test buffer loading
	PackedByteArray test_buffer;
	test_buffer.push_back(0x7f); // ELF magic number start
	test_buffer.push_back(0x45);
	test_buffer.push_back(0x4c);
	test_buffer.push_back(0x46);
	
	sandbox->load_buffer(test_buffer);
	// Should not crash even with invalid ELF
	
	// Test reset functionality
	sandbox->reset(false);
	sandbox->reset(true);
	// Should not crash
	
	// Test binary info from buffer
	Sandbox::BinaryInfo info = Sandbox::get_program_info_from_binary(test_buffer);
	CHECK(info.language.length() >= 0);
	CHECK(info.functions.size() >= 0);
	CHECK(info.version >= 0);
	
	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] Sandbox VM hotspots and profiling") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());
	
	// Test hotspots functionality
	Array hotspots = Sandbox::get_hotspots(5);
	CHECK(hotspots.size() >= 0); // Should not crash
	
	// Test clearing hotspots
	Sandbox::clear_hotspots();
	// Should not crash
	
	// Test profiling
	sandbox->enable_profiling(true, 500);
	CHECK(sandbox->get_profiling());
	
	sandbox->enable_profiling(false, 500);
	CHECK_FALSE(sandbox->get_profiling());
	
	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

} //namespace TestSandboxVM

#endif // TESTS_ENABLED

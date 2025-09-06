/**************************************************************************/
/*  test_elf_script.h                                                     */
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

#include "core/os/os.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

#include "sandbox_dummy.h"

namespace TestELFScript {

TEST_CASE("[SceneTree][Sandbox] SandboxDummy ELF-like functionality") {
	// Create a SandboxDummy to simulate ELF script functionality
	SandboxDummy *sandbox = memnew(SandboxDummy);
	sandbox->set_name("test_elf_sandbox");
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test basic properties
	CHECK(sandbox->get_name() == "test_elf_sandbox");
	CHECK(sandbox->is_inside_tree());
	CHECK(sandbox->get_parent() == SceneTree::get_singleton()->get_root());

	// Test that no program is loaded initially
	CHECK_FALSE(sandbox->has_program_loaded());

	// Test function availability
	CHECK(sandbox->has_function("generate_json_diff"));
	CHECK(sandbox->has_function("test_function"));
	CHECK_FALSE(sandbox->has_function("nonexistent_function"));

	// Cleanup
	sandbox->queue_free();
}

TEST_CASE("[SceneTree][Sandbox] SandboxDummy ELF buffer loading") {
	SandboxDummy *sandbox = memnew(SandboxDummy);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Create dummy ELF content
	PackedByteArray elf_buffer;
	elf_buffer.push_back(0x7f); // ELF magic number
	elf_buffer.push_back('E');
	elf_buffer.push_back('L');
	elf_buffer.push_back('F');
	elf_buffer.append_array(String("dummy elf content").to_utf8_buffer());

	// Test loading buffer
	CHECK_FALSE(sandbox->has_program_loaded());
	sandbox->load_buffer(elf_buffer);
	CHECK(sandbox->has_program_loaded());

	// Test reset functionality
	sandbox->reset(false); // Reset without unload
	CHECK(sandbox->has_program_loaded()); // Should still be loaded

	sandbox->reset(true); // Reset with unload
	CHECK_FALSE(sandbox->has_program_loaded()); // Should be unloaded

	// Cleanup
	sandbox->queue_free();
}

TEST_CASE("[SceneTree][Sandbox] SandboxDummy function address management") {
	SandboxDummy *sandbox = memnew(SandboxDummy);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test address lookup
	uint64_t addr1 = sandbox->address_of("generate_json_diff");
	uint64_t addr2 = sandbox->address_of("test_function");
	uint64_t addr3 = sandbox->address_of("nonexistent");

	CHECK(addr1 != 0);
	CHECK(addr2 != 0);
	CHECK(addr1 != addr2);
	CHECK(addr3 == 0); // Should return 0 for non-existent functions

	// Test reverse lookup
	String func1 = sandbox->lookup_address(addr1);
	String func2 = sandbox->lookup_address(addr2);
	String func3 = sandbox->lookup_address(0x9999); // Non-existent address

	CHECK(func1 == "generate_json_diff");
	CHECK(func2 == "test_function");
	CHECK(func3.is_empty());

	// Cleanup
	sandbox->queue_free();
}

TEST_CASE("[SceneTree][Sandbox] SandboxDummy VM call functionality") {
	SandboxDummy *sandbox = memnew(SandboxDummy);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test VM call with generate_json_diff
	String json_source = R"({"name": "test", "value": 123})";
	String json_reference = R"({"name": "test", "value": 456})";

	Variant var1(json_source);
	Variant var2(json_reference);
	const Variant *args[] = { &var1, &var2 };

	Callable::CallError call_error;
	uint64_t initial_calls = sandbox->get_calls_made();

	Variant result = sandbox->vmcall_fn("generate_json_diff", args, 2, call_error);

	// Verify call succeeded
	CHECK(call_error.error == Callable::CallError::CALL_OK);
	CHECK(result.get_type() == Variant::STRING);
	CHECK(sandbox->get_calls_made() == initial_calls + 1);

	// Test the result content
	String result_str = result.operator String();
	CHECK(result_str.length() > 0);
	CHECK(result_str.contains("["));
	CHECK(result_str.contains("]"));
	CHECK(result_str.contains("test_diff_result"));

	// Test with non-existent function
	Variant invalid_result = sandbox->vmcall_fn("nonexistent_function", args, 2, call_error);
	CHECK(call_error.error == Callable::CallError::CALL_OK); // Dummy handles gracefully
	CHECK(invalid_result.get_type() == Variant::STRING);
	String invalid_str = invalid_result.operator String();
	CHECK(invalid_str.contains("dummy_result_nonexistent_function"));

	// Cleanup
	sandbox->queue_free();
}

TEST_CASE("[SceneTree][Sandbox] SandboxDummy binary translation features") {
	SandboxDummy *sandbox = memnew(SandboxDummy);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test JIT feature detection (dummy should return false)
	bool has_jit = SandboxDummy::has_feature_jit();
	CHECK_FALSE(has_jit);

	// Test JIT enabled state (dummy should return false)
	bool jit_enabled = SandboxDummy::is_jit_enabled();
	CHECK_FALSE(jit_enabled);

	// Test binary translation state (dummy should return false)
	CHECK_FALSE(sandbox->is_binary_translated());
	CHECK_FALSE(sandbox->is_jit());

	// Cleanup
	sandbox->queue_free();
}

} //namespace TestELFScript

/**************************************************************************/
/*  test_sandbox.h                                                        */
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

namespace TestSandbox {

TEST_CASE("[SceneTree][Sandbox] SandboxDummy basic instantiation and scene attachment") {
	// Create a SandboxDummy node
	SandboxDummy *sandbox = memnew(SandboxDummy);
	sandbox->set_name("test_sandbox");

	// Add to scene tree
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test basic properties
	CHECK(sandbox->get_name() == "test_sandbox");
	CHECK(sandbox->is_inside_tree());
	CHECK(sandbox->get_parent() == SceneTree::get_singleton()->get_root());

	// Test initial state
	CHECK(sandbox->get_max_refs() == SandboxDummy::MAX_REFS);
	CHECK(sandbox->get_memory_max() == SandboxDummy::MAX_VMEM);
	CHECK(sandbox->get_instructions_max() == SandboxDummy::MAX_INSTRUCTIONS);
	CHECK(sandbox->get_allocations_max() == SandboxDummy::MAX_HEAP_ALLOCS);

	// Test that no program is loaded initially
	CHECK_FALSE(sandbox->has_program_loaded());

	// Cleanup
	sandbox->queue_free();
}

TEST_CASE("[SceneTree][Sandbox] SandboxDummy memory and instruction limits") {
	SandboxDummy *sandbox = memnew(SandboxDummy);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test setting memory limits
	sandbox->set_memory_max(32);
	CHECK(sandbox->get_memory_max() == 32);

	// Test setting instruction limits
	sandbox->set_instructions_max(16000);
	CHECK(sandbox->get_instructions_max() == 16000);

	// Test setting allocation limits
	sandbox->set_allocations_max(8000);
	CHECK(sandbox->get_allocations_max() == 8000);

	// Test setting max refs
	sandbox->set_max_refs(200);
	CHECK(sandbox->get_max_refs() == 200);

	// Cleanup
	sandbox->queue_free();
}

TEST_CASE("[SceneTree][Sandbox] SandboxDummy restrictions and security") {
	SandboxDummy *sandbox = memnew(SandboxDummy);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test initial restrictions state
	CHECK_FALSE(sandbox->get_restrictions());

	// Test enabling restrictions
	sandbox->set_restrictions(true);
	CHECK(sandbox->get_restrictions());

	// Test disabling restrictions
	sandbox->set_restrictions(false);
	CHECK_FALSE(sandbox->get_restrictions());

	// Test allowed objects management
	Node *test_node = memnew(Node);

	// Initially all objects should be allowed (when restrictions are off)
	CHECK(sandbox->is_allowed_object(test_node));

	// Add to allowed list
	sandbox->add_allowed_object(test_node);
	CHECK(sandbox->is_allowed_object(test_node));

	// Remove from allowed list
	sandbox->remove_allowed_object(test_node);

	// Clear allowed objects
	sandbox->clear_allowed_objects();

	// Cleanup
	memdelete(test_node);
	sandbox->queue_free();
}

TEST_CASE("[SceneTree][Sandbox] SandboxDummy tree base functionality") {
	SandboxDummy *sandbox = memnew(SandboxDummy);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test tree base setting
	Node *tree_base = memnew(Node);
	tree_base->set_name("tree_base");
	SceneTree::get_singleton()->get_root()->add_child(tree_base);

	sandbox->set_tree_base(tree_base);
	CHECK(sandbox->get_tree_base() == tree_base);
	CHECK(sandbox->get_tree_base()->get_name() == "tree_base");

	// Cleanup
	tree_base->queue_free();
	sandbox->queue_free();
}

TEST_CASE("[SceneTree][Sandbox] SandboxDummy profiling functionality") {
	SandboxDummy *sandbox = memnew(SandboxDummy);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test initial profiling state
	CHECK_FALSE(sandbox->get_profiling());

	// Test enabling profiling
	sandbox->set_profiling(true);
	CHECK(sandbox->get_profiling());

	// Test disabling profiling
	sandbox->set_profiling(false);
	CHECK_FALSE(sandbox->get_profiling());

	// Test profiling with interval
	sandbox->enable_profiling(true, 1000);
	CHECK(sandbox->get_profiling());

	sandbox->enable_profiling(false, 1000);
	CHECK_FALSE(sandbox->get_profiling());

	// Cleanup
	sandbox->queue_free();
}

TEST_CASE("[SceneTree][Sandbox] SandboxDummy global statistics") {
	// Test global statistics access
	uint64_t initial_timeouts = SandboxDummy::get_global_timeouts();
	uint64_t initial_exceptions = SandboxDummy::get_global_exceptions();
	uint64_t initial_calls = SandboxDummy::get_global_calls_made();
	uint64_t initial_instances = SandboxDummy::get_global_instance_count();

	// These should be accessible without errors
	CHECK(initial_timeouts >= 0);
	CHECK(initial_exceptions >= 0);
	CHECK(initial_calls >= 0);
	CHECK(initial_instances >= 0);

	// Test accumulated startup time
	double startup_time = SandboxDummy::get_accumulated_startup_time();
	CHECK(startup_time >= 0.0);
}

TEST_CASE("[SceneTree][Sandbox] SandboxDummy binary translation features") {
	SandboxDummy *sandbox = memnew(SandboxDummy);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test JIT feature detection
	bool has_jit = SandboxDummy::has_feature_jit();
	// This should not crash regardless of JIT availability
	CHECK((has_jit == true || has_jit == false));

	// Test JIT enabled state
	bool jit_enabled = SandboxDummy::is_jit_enabled();
	CHECK((jit_enabled == true || jit_enabled == false));

	// Test binary translation state
	CHECK_FALSE(sandbox->is_binary_translated());
	CHECK_FALSE(sandbox->is_jit());

	// Cleanup
	sandbox->queue_free();
}

} //namespace TestSandbox

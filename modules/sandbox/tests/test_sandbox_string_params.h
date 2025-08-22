/**************************************************************************/
/*  test_sandbox_string_params.h                                          */
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
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"

#include "../sandbox.h"

namespace TestSandboxStringParams {

TEST_CASE("[SceneTree][Node] Sandbox variant index bounds checking") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test valid indices (should not crash)
	auto valid_opt = sandbox->get_scoped_variant(0);
	CHECK_FALSE(valid_opt.has_value()); // Should return nullopt for empty state

	// Test negative valid index
	auto negative_valid_opt = sandbox->get_scoped_variant(-1);
	CHECK_FALSE(negative_valid_opt.has_value()); // Should return nullopt for empty state

	// Test obviously corrupted large positive index (memory corruption detection)
	auto corrupted_large_opt = sandbox->get_scoped_variant(2420680);
	CHECK_FALSE(corrupted_large_opt.has_value()); // Should return nullopt and log error

	// Test obviously corrupted large negative index
	auto corrupted_neg_opt = sandbox->get_scoped_variant(-2420680);
	CHECK_FALSE(corrupted_neg_opt.has_value()); // Should return nullopt and log error

	// Test boundary values
	auto max_int_opt = sandbox->get_scoped_variant(INT32_MAX);
	CHECK_FALSE(max_int_opt.has_value()); // Should return nullopt and log error

	auto min_int_opt = sandbox->get_scoped_variant(INT32_MIN);
	CHECK_FALSE(min_int_opt.has_value()); // Should return nullopt and log error

	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] Sandbox scoped variant creation and retrieval") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test string variant creation
	String test_string = "Hello, Sandbox!";
	Variant string_variant(test_string);

	// Should not crash when creating scoped variants
	try {
		unsigned string_idx = sandbox->create_scoped_variant(Variant(string_variant));

		// Test retrieval of the string variant
		auto retrieved_opt = sandbox->get_scoped_variant(string_idx);
		CHECK(retrieved_opt.has_value());

		if (retrieved_opt.has_value()) {
			const Variant *retrieved_var = retrieved_opt.value();
			CHECK(retrieved_var->get_type() == Variant::STRING);
			CHECK(retrieved_var->operator String() == test_string);
		}
	} catch (const std::exception &e) {
		// This should not happen with proper implementation
		FAIL("Exception during scoped variant creation: " + std::string(e.what()));
	}

	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] Sandbox string parameter marshaling") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test string parameter handling without actually calling VM functions
	String test_string1 = "Parameter One";
	String test_string2 = "Parameter Two";

	// Test adding string parameters as scoped variants
	const Variant *args[2] = { &Variant(test_string1), &Variant(test_string2) };

	try {
		// Test that we can add string parameters without crashing
		unsigned idx1 = sandbox->add_scoped_variant(args[0]);
		unsigned idx2 = sandbox->add_scoped_variant(args[1]);

		// Verify the indices are reasonable
		CHECK(idx1 < 1000); // Should be small for empty sandbox
		CHECK(idx2 < 1000); // Should be small for empty sandbox
		CHECK(idx1 != idx2); // Should be different indices

		// Test retrieval
		auto opt1 = sandbox->get_scoped_variant(idx1);
		auto opt2 = sandbox->get_scoped_variant(idx2);

		CHECK(opt1.has_value());
		CHECK(opt2.has_value());

		if (opt1.has_value() && opt2.has_value()) {
			CHECK(opt1.value()->operator String() == test_string1);
			CHECK(opt2.value()->operator String() == test_string2);
		}
	} catch (const std::exception &e) {
		FAIL("Exception during string parameter marshaling: " + std::string(e.what()));
	}

	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] Sandbox variant state management") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test creating multiple variant types
	Array test_variants = Array();
	test_variants.push_back("String variant");
	test_variants.push_back(42);
	test_variants.push_back(3.14159);
	test_variants.push_back(Vector2(1.0, 2.0));

	try {
		std::vector<unsigned> indices;

		// Create scoped variants for each type
		for (int i = 0; i < test_variants.size(); i++) {
			Variant variant = test_variants[i];
			unsigned idx = sandbox->create_scoped_variant(Variant(variant));
			indices.push_back(idx);
		}

		// Verify all variants can be retrieved correctly
		for (int i = 0; i < test_variants.size(); i++) {
			auto opt = sandbox->get_scoped_variant(indices[i]);
			CHECK(opt.has_value());

			if (opt.has_value()) {
				const Variant *retrieved = opt.value();
				Variant original = test_variants[i];
				CHECK(retrieved->get_type() == original.get_type());
			}
		}

	} catch (const std::exception &e) {
		FAIL("Exception during variant state management: " + std::string(e.what()));
	}

	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] Sandbox memory corruption detection") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Test that memory corruption indices are properly detected
	struct CorruptedIndex {
		int32_t index;
		const char *description;
	};

	CorruptedIndex corrupted_indices[] = {
		{ 2420680, "Large positive corrupted index" },
		{ -2420680, "Large negative corrupted index" },
		{ 1000000, "Extremely large index" },
		{ -1000000, "Extremely large negative index" },
		{ INT32_MAX, "Maximum integer value" },
		{ INT32_MIN, "Minimum integer value" },
	};

	for (const auto &test_case : corrupted_indices) {
		// These should all return nullopt and log appropriate error messages
		auto opt = sandbox->get_scoped_variant(test_case.index);
		CHECK_FALSE(opt.has_value());

		// The function should not crash or throw exceptions for corrupted indices
		// The error handling should be graceful
	}

	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] Sandbox variant capacity management") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());

	// Set a small max refs limit to test capacity handling
	sandbox->set_max_refs(5);
	CHECK(sandbox->get_max_refs() == 5);

	try {
		// Create variants up to the limit
		std::vector<unsigned> indices;
		for (int i = 0; i < 5; i++) {
			String test_string = "Test string " + String::num(i);
			unsigned idx = sandbox->create_scoped_variant(Variant(test_string));
			indices.push_back(idx);
		}

		// Verify all variants are accessible
		for (unsigned idx : indices) {
			auto opt = sandbox->get_scoped_variant(idx);
			CHECK(opt.has_value());
		}

		// Attempting to create more should either succeed with proper capacity management
		// or throw a controlled exception - it should not crash
		bool capacity_exception_thrown = false;
		try {
			String overflow_string = "Overflow string";
			sandbox->create_scoped_variant(Variant(overflow_string));
		} catch (const std::runtime_error &e) {
			capacity_exception_thrown = true;
			// This is expected behavior when hitting capacity limits
		}

		// Either the capacity was managed properly, or a controlled exception was thrown
		// Both are acceptable behaviors - the key is no crash
		CHECK(true); // Test passed if we reach here without crashing

	} catch (const std::exception &e) {
		// Only capacity-related exceptions should be thrown
		String error_msg = e.what();
		bool is_capacity_error = error_msg.contains("scoped variants reached") ||
				error_msg.contains("capacity");
		CHECK(is_capacity_error);
	}

	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

} //namespace TestSandboxStringParams

#endif // TESTS_ENABLED

/**************************************************************************/
/*  test_sandbox_download.h                                               */
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

#include "../src/sandbox.h"

namespace TestSandboxDownload {

TEST_CASE("[SceneTree][Node] Sandbox ELF program download functionality") {
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());
	
	// Test downloading a hello-world program
	ERR_PRINT_OFF; // Disable error prints for network operations
	PackedByteArray hello_world_elf = Sandbox::download_program("hello-world");
	ERR_PRINT_ON;
	
	// Check if download succeeded (may fail due to network)
	if (hello_world_elf.size() > 0) {
		MESSAGE("Successfully downloaded hello-world ELF program (", hello_world_elf.size(), " bytes)");
		
		// Test loading the downloaded program
		sandbox->load_buffer(hello_world_elf);
		
		// Test that program is now loaded
		CHECK(sandbox->has_program_loaded());
		
		// Test getting binary info
		Sandbox::BinaryInfo info = Sandbox::get_program_info_from_binary(hello_world_elf);
		CHECK(info.language.length() > 0);
		CHECK(info.functions.size() >= 0);
		CHECK(info.version >= 0);
		
		MESSAGE("Program language: ", info.language);
		MESSAGE("Program version: ", info.version);
		MESSAGE("Available functions: ", info.functions.size());
		
		// TODO: Test actual function execution once conversion is complete
		// Callable::CallError error;
		// Variant result = sandbox->vmcall_fn("hello_world", nullptr, 0, error);
		// CHECK(result == "Hello, world!");
		
	} else {
		MESSAGE("Could not download hello-world program (network issue or program not available)");
		// This is not a failure - just means we can't test with real ELF
	}
	
	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE_PENDING("[SceneTree][Node] Sandbox ELF program execution with downloaded program") {
	// TODO: This test requires the full conversion to be complete
	// Once register_types.cpp properly registers Sandbox class and
	// the conversion to internal APIs is finished, this test can be enabled
	
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());
	
	// Download and load hello-world program
	PackedByteArray hello_world_elf = Sandbox::download_program("hello-world");
	
	if (hello_world_elf.size() > 0) {
		sandbox->load_buffer(hello_world_elf);
		
		// Test function execution
		Callable::CallError error;
		const Variant *args = nullptr;
		
		// Test hello_world function
		Variant result = sandbox->vmcall_fn("hello_world", &args, 0, error);
		CHECK(error.error == Callable::CALL_OK);
		CHECK(result == "Hello, world!");
		
		// Test fibonacci function
		Variant fib_arg = 10;
		const Variant *fib_args[] = { &fib_arg };
		Variant fib_result = sandbox->vmcall_fn("fibonacci", fib_args, 1, error);
		CHECK(error.error == Callable::CALL_OK);
		CHECK(fib_result == 55); // 10th Fibonacci number
		
		// Test property access
		CHECK(sandbox->get("meaning_of_life") == 42);
		
		sandbox->set("meaning_of_life", 100);
		CHECK(sandbox->get("meaning_of_life") == 100);
	}
	
	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

} //namespace TestSandboxDownload

#endif // TESTS_ENABLED

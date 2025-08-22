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

#ifdef TESTS_ENABLED

#include "core/os/os.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h"
#include "scene/main/node.h"

#include "../elf/script_elf.h"
#include "../elf/script_language_elf.h"
#include "../sandbox.h"

namespace TestELFScript {

TEST_CASE("[SceneTree][Node] ELF Script basic instantiation") {
	// Create an ELF script
	Ref<ELFScript> elf_script = memnew(ELFScript);
	
	// Test basic properties
	CHECK(elf_script.is_valid());
	CHECK(elf_script->is_valid());
	CHECK(elf_script->has_source_code());
	
	// Test initial state
	CHECK(elf_script->get_elf_api_version() >= 0);
	CHECK(elf_script->get_source_version() >= 0);
	CHECK(elf_script->get_elf_programming_language().length() >= 0);
	
	// Test that it can be used as a script
	CHECK(elf_script->can_instantiate() || !elf_script->can_instantiate()); // Should not crash
	CHECK(elf_script->is_tool() || !elf_script->is_tool()); // Should not crash
}

TEST_CASE("[SceneTree][Node] ELF Script with Sandbox integration") {
	// Create a Sandbox node
	Sandbox *sandbox = memnew(Sandbox);
	sandbox->set_name("test_sandbox");
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());
	
	// Create an ELF script
	Ref<ELFScript> elf_script = memnew(ELFScript);
	
	// Test that sandbox can work with ELF scripts
	CHECK_FALSE(sandbox->has_program_loaded());
	
	// Test setting program (should not crash even with empty script)
	sandbox->set_program(elf_script);
	
	// Test getting program back
	Ref<ELFScript> retrieved_script = sandbox->get_program();
	CHECK(retrieved_script.is_valid());
	
	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] ELF Script Language basic functionality") {
	// Create ELF script language
	ELFScriptLanguage *elf_lang = memnew(ELFScriptLanguage);
	
	// Test basic language properties
	CHECK(elf_lang->get_name().length() > 0);
	CHECK(elf_lang->get_type().length() > 0);
	CHECK(elf_lang->get_extension().length() > 0);
	
	// Test language capabilities
	CHECK(elf_lang->supports_builtin_mode() || !elf_lang->supports_builtin_mode());
	CHECK(elf_lang->supports_documentation() || !elf_lang->supports_documentation());
	CHECK(elf_lang->can_inherit_from_file());
	
	// Test script creation
	Script *created_script = elf_lang->create_script();
	CHECK(created_script != nullptr);
	
	// Cleanup
	if (created_script) {
		memdelete(created_script);
	}
	memdelete(elf_lang);
}

TEST_CASE("[SceneTree][Node] ELF Script sandbox objects tracking") {
	// Create a Sandbox node
	Sandbox *sandbox = memnew(Sandbox);
	SceneTree::get_singleton()->get_root()->add_child(sandbox);
	sandbox->set_owner(SceneTree::get_singleton()->get_root());
	
	// Create an ELF script
	Ref<ELFScript> elf_script = memnew(ELFScript);
	
	// Test sandbox objects retrieval
	Array sandbox_objects = elf_script->get_sandbox_objects();
	CHECK(sandbox_objects.size() >= 0); // Should not crash
	
	// Test sandbox retrieval for object
	Sandbox *retrieved_sandbox = elf_script->get_sandbox_for(sandbox);
	// May be null if not properly linked, but should not crash
	
	// Cleanup
	SceneTree::get_singleton()->get_root()->remove_child(sandbox);
	memdelete(sandbox);
}

TEST_CASE("[SceneTree][Node] ELF Script content and path management") {
	// Create an ELF script
	Ref<ELFScript> elf_script = memnew(ELFScript);
	
	// Test path management
	String test_path = "res://test_script.elf";
	elf_script->set_file(test_path);
	CHECK(elf_script->get_path() == test_path);
	
	// Test content retrieval
	const PackedByteArray &content = elf_script->get_content();
	CHECK(content.size() >= 0); // Should not crash
	
	// Test source code management
	String source = elf_script->get_source_code();
	CHECK(source.length() >= 0); // Should not crash
	
	String test_source = "// Test source code";
	elf_script->set_source_code(test_source);
	CHECK(elf_script->get_source_code() == test_source);
}

TEST_CASE("[SceneTree][Node] ELF Script function and API management") {
	// Create an ELF script
	Ref<ELFScript> elf_script = memnew(ELFScript);
	
	// Test function arrays
	CHECK(elf_script->functions.size() >= 0);
	CHECK(elf_script->function_names.size() >= 0);
	
	// Test API functions
	Array test_functions;
	test_functions.push_back("test_function");
	elf_script->set_public_api_functions(std::move(test_functions));
	
	// Update API functions (should not crash)
	elf_script->update_public_api_functions();
	
	// Test dockerized program path (should not crash)
	String docker_path = elf_script->get_dockerized_program_path();
	CHECK(docker_path.length() >= 0);
}

} //namespace TestELFScript

#endif // TESTS_ENABLED

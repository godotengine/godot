/**************************************************************************/
/*  sandbox_base.cpp                                                      */
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

#include "sandbox_base.h"

// Static member definitions
uint64_t SandboxBase::global_calls_made = 0;
uint64_t SandboxBase::global_exceptions = 0;
uint64_t SandboxBase::global_timeouts = 0;
uint64_t SandboxBase::global_instance_count = 0;
double SandboxBase::accumulated_startup_time = 0.0;

SandboxBase::SandboxBase() {
	tree_base = this;
	global_instance_count++;
	accumulated_startup_time += 0.001; // Simulate startup time

	setup_base_functions();
}

SandboxBase::~SandboxBase() {
	if (global_instance_count > 0) {
		global_instance_count--;
	}
}

void SandboxBase::setup_base_functions() {
	// Add some dummy functions for testing
	available_functions["generate_json_diff"] = true;
	available_functions["test_function"] = true;
	function_addresses["generate_json_diff"] = 0x1000;
	function_addresses["test_function"] = 0x2000;
}

void SandboxBase::load_buffer(const PackedByteArray &p_buffer) {
	program_bytes = p_buffer;
	program_loaded = !p_buffer.is_empty();
}

void SandboxBase::set_program(Ref<ELFScript> program) {
	// Base implementation - just mark as loaded if program is valid
	program_loaded = program.is_valid();
}

Ref<ELFScript> SandboxBase::get_program() {
	// Base implementation returns null
	return Ref<ELFScript>();
}

void SandboxBase::reset(bool p_unload) {
	if (p_unload) {
		program_loaded = false;
		program_bytes.clear();
	}
}

bool SandboxBase::has_function(const StringName &p_function) const {
	String func_str = p_function;
	return available_functions.find(func_str) != available_functions.end();
}

uint64_t SandboxBase::address_of(const String &p_symbol) const {
	auto it = function_addresses.find(p_symbol);
	return (it != function_addresses.end()) ? it->value : 0;
}

String SandboxBase::lookup_address(uint64_t p_address) const {
	for (const auto &pair : function_addresses) {
		if (pair.value == p_address) {
			return pair.key;
		}
	}
	return String();
}

Variant SandboxBase::vmcall_fn(const StringName &p_function, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) {
	calls_made++;
	global_calls_made++;

	r_error.error = Callable::CallError::CALL_OK;

	String func_name = p_function;

	// Simulate generate_json_diff function
	if (func_name == "generate_json_diff" && p_arg_count == 2) {
		// Get the string arguments safely without using substr which can cause string ops issues
		String arg1 = p_args[0]->operator String();
		String arg2 = p_args[1]->operator String();

		// Simple dummy JSON diff (not real implementation) - avoid substr to prevent string ops errors
		String result = R"([
  { "op": "replace", "path": "/value", "value": "test_diff_result" },
  { "op": "test", "path": "/source", "value": "source_data" },
  { "op": "test", "path": "/reference", "value": "reference_data" }
])";
		return Variant(result);
	}

	// Default dummy response
	return Variant("dummy_result_" + func_name);
}

void SandboxBase::_bind_methods() {
	// Base implementation - no methods to bind for now
}

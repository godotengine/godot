/**************************************************************************/
/*  gdscript_test_runner_suite.h                                          */
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

#include "../gdscript_cache.h"
#include "gdscript_test_runner.h"

#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

#ifdef TOOLS_ENABLED
#include "core/os/os.h"
#endif

namespace GDScriptTests {

class TestGDScriptCacheAccessor {
public:
	static bool has_shallow(String p_path) {
		return GDScriptCache::singleton->shallow_gdscript_cache.has(p_path);
	}

	static bool has_full(String p_path) {
		return GDScriptCache::singleton->full_gdscript_cache.has(p_path);
	}
};

// TODO: Handle some cases failing on release builds. See: https://github.com/godotengine/godot/pull/88452
#ifdef TOOLS_ENABLED
TEST_SUITE("[Modules][GDScript]") {
	TEST_CASE("Script compilation and runtime") {
		bool print_filenames = OS::get_singleton()->get_cmdline_args().find("--print-filenames") != nullptr;
		bool use_binary_tokens = OS::get_singleton()->get_cmdline_args().find("--use-binary-tokens") != nullptr;
		GDScriptTestRunner runner("modules/gdscript/tests/scripts", true, print_filenames, use_binary_tokens);
		int fail_count = runner.run_tests();
		INFO("Make sure `*.out` files have expected results.");
		REQUIRE_MESSAGE(fail_count == 0, "All GDScript tests should pass.");
	}
}
#endif // TOOLS_ENABLED

TEST_CASE("[Modules][GDScript] Load source code dynamically and run it") {
	GDScriptLanguage::get_singleton()->init();
	Ref<GDScript> gdscript = memnew(GDScript);
	gdscript->set_source_code(R"(
extends RefCounted

func _init():
	set_meta("result", 42)
)");
	// A spurious `Condition "err" is true` message is printed (despite parsing being successful and returning `OK`).
	// Silence it.
	ERR_PRINT_OFF;
	const Error error = gdscript->reload();
	ERR_PRINT_ON;
	CHECK_MESSAGE(error == OK, "The script should parse successfully.");

	// Run the script by assigning it to a reference-counted object.
	Ref<RefCounted> ref_counted = memnew(RefCounted);
	ref_counted->set_script(gdscript);
	CHECK_MESSAGE(int(ref_counted->get_meta("result")) == 42, "The script should assign object metadata successfully.");
}

#ifdef DEBUG_ENABLED
TEST_CASE("[Modules][GDScript] Hot reload initializes newly added member defaults") {
	GDScriptLanguage::get_singleton()->init();
	Ref<GDScript> gdscript = memnew(GDScript);
	gdscript->set_source_code(R"(
extends RefCounted

func describe() -> String:
	return "v1"
)");

	ERR_PRINT_OFF;
	Error error = gdscript->reload();
	ERR_PRINT_ON;
	REQUIRE_MESSAGE(error == OK, "The v1 script should parse successfully.");

	Callable::CallError call_error;
	Variant instance_a = gdscript->_new(nullptr, 0, call_error);
	REQUIRE(call_error.error == Callable::CallError::CALL_OK);
	Ref<RefCounted> ref_a = instance_a;
	REQUIRE(ref_a.is_valid());

	call_error = Callable::CallError();
	Variant instance_b = gdscript->_new(nullptr, 0, call_error);
	REQUIRE(call_error.error == Callable::CallError::CALL_OK);
	Ref<RefCounted> ref_b = instance_b;
	REQUIRE(ref_b.is_valid());

	gdscript->set_source_code(R"(
extends RefCounted

var injected_dict: Dictionary = {}
var injected_array: Array = []

func describe() -> String:
	return "dict=%d array=%d" % [injected_dict.keys().size(), injected_array.size()]
)");

	ERR_PRINT_OFF;
	error = gdscript->reload(true);
	ERR_PRINT_ON;
	REQUIRE_MESSAGE(error == OK, "The v2 script should hot-reload successfully.");

	Object *object_a = ref_a.ptr();
	List<PropertyInfo> property_list;
	object_a->get_property_list(&property_list);

	bool has_injected_dict = false;
	bool has_injected_array = false;
	for (const PropertyInfo &property : property_list) {
		if (property.name == StringName("injected_dict")) {
			has_injected_dict = true;
		} else if (property.name == StringName("injected_array")) {
			has_injected_array = true;
		}
	}
	CHECK(has_injected_dict);
	CHECK(has_injected_array);

	bool valid = false;
	Variant dict_value = object_a->get(StringName("injected_dict"), &valid);
	CHECK(valid);
	REQUIRE(dict_value.get_type() == Variant::DICTIONARY);
	Dictionary dict_value_as_dictionary = dict_value;
	CHECK_EQ(dict_value_as_dictionary.size(), 0);

	valid = false;
	Variant array_value = object_a->get(StringName("injected_array"), &valid);
	CHECK(valid);
	REQUIRE(array_value.get_type() == Variant::ARRAY);
	Array array_value_as_array = array_value;
	CHECK_EQ(array_value_as_array.size(), 0);

	call_error = Callable::CallError();
	Variant result = object_a->callp(StringName("describe"), nullptr, 0, call_error);
	REQUIRE(call_error.error == Callable::CallError::CALL_OK);
	String result_string = result;
	CHECK_EQ(result_string, "dict=0 array=0");

	Dictionary dict_a = object_a->get(StringName("injected_dict"));
	dict_a[String("key")] = 1;
	Array array_a = object_a->get(StringName("injected_array"));
	array_a.push_back(1);

	Object *object_b = ref_b.ptr();
	Dictionary dict_b = object_b->get(StringName("injected_dict"));
	Array array_b = object_b->get(StringName("injected_array"));
	CHECK_EQ(dict_b.size(), 0);
	CHECK_EQ(array_b.size(), 0);
}
#endif // DEBUG_ENABLED

TEST_CASE("[Modules][GDScript] Loading keeps ResourceCache and GDScriptCache in sync") {
	const String path = TestUtils::get_temp_path("gdscript_load_test.gd");

	{
		Ref<FileAccess> fa = FileAccess::open(path, FileAccess::ModeFlags::WRITE);
		fa->store_string("extends Node\n");
		fa->close();
	}

	CHECK(!ResourceCache::has(path));
	CHECK(!TestGDScriptCacheAccessor::has_shallow(path));
	CHECK(!TestGDScriptCacheAccessor::has_full(path));

	Ref<GDScript> loaded = ResourceLoader::load(path);

	CHECK(ResourceCache::has(path));
	CHECK(!TestGDScriptCacheAccessor::has_shallow(path));
	CHECK(TestGDScriptCacheAccessor::has_full(path));
}

TEST_CASE("[Modules][GDScript] Validate built-in API") {
	GDScriptLanguage *lang = GDScriptLanguage::get_singleton();

	// Validate methods.
	List<MethodInfo> builtin_methods;
	lang->get_public_functions(&builtin_methods);

	SUBCASE("[Modules][GDScript] Validate built-in methods") {
		for (const MethodInfo &mi : builtin_methods) {
			for (int64_t i = 0; i < mi.arguments.size(); ++i) {
				TEST_COND((mi.arguments[i].name.is_empty() || mi.arguments[i].name.begins_with("_unnamed_arg")),
						vformat("Unnamed argument in position %d of built-in method '%s'.", i, mi.name));
			}
		}
	}

	// Validate annotations.
	List<MethodInfo> builtin_annotations;
	lang->get_public_annotations(&builtin_annotations);

	SUBCASE("[Modules][GDScript] Validate built-in annotations") {
		for (const MethodInfo &ai : builtin_annotations) {
			for (int64_t i = 0; i < ai.arguments.size(); ++i) {
				TEST_COND((ai.arguments[i].name.is_empty() || ai.arguments[i].name.begins_with("_unnamed_arg")),
						vformat("Unnamed argument in position %d of built-in annotation '%s'.", i, ai.name));
			}
		}
	}
}

} // namespace GDScriptTests

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
#include "../gdscript_parser.h"
#include "gdscript_test_runner.h"

#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

#ifdef TOOLS_ENABLED
#include "core/object/script_language.h"
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

#ifdef TOOLS_ENABLED
static void write_test_script(const String &p_path, const String &p_source) {
	Ref<FileAccess> fa = FileAccess::open(p_path, FileAccess::ModeFlags::WRITE);
	REQUIRE(fa.is_valid());
	fa->store_string(p_source);
	fa->close();
}

static PropertyInfo find_script_property(const Ref<GDScript> &p_script, const StringName &p_property) {
	List<PropertyInfo> properties;
	p_script->get_script_property_list(&properties);

	for (const PropertyInfo &property : properties) {
		if (property.name == p_property) {
			return property;
		}
	}

	return PropertyInfo();
}

// TODO: Handle some cases failing on release builds. See: https://github.com/godotengine/godot/pull/88452
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

TEST_CASE("[Modules][GDScript] Loading keeps ResourceCache and GDScriptCache in sync") {
	GDScriptLanguage::get_singleton()->init();
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

#ifdef TOOLS_ENABLED
TEST_CASE("[Modules][GDScript] Reloading a script refreshes dependent enum exports") {
	GDScriptLanguage::get_singleton()->init();

	const String provider_path = TestUtils::get_temp_path("gdscript_external_enum_provider.gd");
	const String consumer_path = TestUtils::get_temp_path("gdscript_external_enum_consumer.gd");
	const StringName provider_class_name = SNAME("ExternalTestEnum");
	ScriptServer::remove_global_class(provider_class_name);

	write_test_script(provider_path, R"(
class_name ExternalTestEnum
extends Resource

enum Kind { First, Second }
)");

	String base_type;
	bool is_abstract = false;
	bool is_tool = false;
	String class_name = GDScriptLanguage::get_singleton()->get_global_class_name(provider_path, &base_type, nullptr, &is_abstract, &is_tool);
	REQUIRE(class_name == provider_class_name);
	ScriptServer::add_global_class(provider_class_name, base_type, GDScriptLanguage::get_singleton()->get_name(), provider_path, is_abstract, is_tool);

	write_test_script(consumer_path, R"(
extends Resource

@export var external_enum: ExternalTestEnum.Kind
)");

	Ref<GDScript> provider = ResourceLoader::load(provider_path);
	Ref<GDScript> consumer = ResourceLoader::load(consumer_path);
	REQUIRE(provider.is_valid());
	REQUIRE(consumer.is_valid());

	const PropertyInfo original_property = find_script_property(consumer, SNAME("external_enum"));
	CHECK(original_property.hint == PROPERTY_HINT_ENUM);
	CHECK(original_property.hint_string == "First:0,Second:1");

	write_test_script(provider_path, R"(
class_name ExternalTestEnum
extends Resource

enum Kind { First, Second, Third }
)");

	GDScriptLanguage::get_singleton()->reload_tool_script(provider, true);

	const PropertyInfo updated_property = find_script_property(consumer, SNAME("external_enum"));
	CHECK(updated_property.hint == PROPERTY_HINT_ENUM);
	CHECK(updated_property.hint_string == "First:0,Second:1,Third:2");

	ScriptServer::remove_global_class(provider_class_name);
	GDScriptCache::remove_script(consumer_path);
	GDScriptCache::remove_script(provider_path);
}
#endif // TOOLS_ENABLED

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

static void check_single_multiline_string_error(const String &p_string, int p_start_line, int p_start_column, int p_end_line, int p_end_column, const String &p_message) {
	const String source = String("var s = \"\"\"\n") + p_string + "\n\"\"\"\n";

	GDScriptParser parser;
	parser.parse(source, "", false);
	const List<GDScriptParser::ParserError> &errors = parser.get_errors();

	REQUIRE_MESSAGE(errors.size() == 1, vformat("Expected exactly one error, got %d.", errors.size()));

	const GDScriptParser::ParserError &error = errors.front()->get();

	CHECK(error.start_line == p_start_line);
	CHECK(error.start_column == p_start_column);
	CHECK(error.end_line == p_end_line);
	CHECK(error.end_column == p_end_column);
	CHECK(error.message == p_message);
}

// These can't be part of the regular parser tests because they cannot test the position information of errors.
TEST_CASE("[Modules][GDScript] Validate multi-line string error positions") {
	check_single_multiline_string_error(String::chr(0x200E), 2, 1, 2, 2, "Invisible text direction control character present in the string, escape it (\"\\u200e\") to avoid confusion.");
	check_single_multiline_string_error("\\Uzzzzzz", 2, 3, 2, 4, "Invalid hexadecimal digit in unicode escape sequence.");
	check_single_multiline_string_error("\\p", 2, 1, 2, 3, "Invalid escape in string.");
	check_single_multiline_string_error("abc\ndef\nghi\\p", 4, 4, 4, 6, "Invalid escape in string.");
	check_single_multiline_string_error("\\uD800\\uD800", 2, 11, 2, 13, "Invalid UTF-16 sequence in string, unpaired lead surrogate.");
	check_single_multiline_string_error("\\uDC00", 2, 5, 2, 7, "Invalid UTF-16 sequence in string, unpaired trail surrogate.");
	check_single_multiline_string_error("\\uD800\\u0041", 2, 5, 2, 7, "Invalid UTF-16 sequence in string, unpaired lead surrogate.");
	check_single_multiline_string_error("\\uD800\"abc", 2, 5, 2, 7, "Invalid UTF-16 sequence in string, unpaired lead surrogate");
	check_single_multiline_string_error("\\uD800x", 2, 5, 2, 7, "Invalid UTF-16 sequence in string, unpaired lead surrogate");
	check_single_multiline_string_error("\\uD800\\\n\\u0041", 2, 5, 2, 7, "Invalid UTF-16 sequence in string, unpaired lead surrogate.");
	check_single_multiline_string_error("\\uD800\\", 2, 5, 2, 7, "Invalid UTF-16 sequence in string, unpaired lead surrogate");
	check_single_multiline_string_error("\\uD800\\\nx", 2, 5, 2, 7, "Invalid UTF-16 sequence in string, unpaired lead surrogate");
}

} // namespace GDScriptTests

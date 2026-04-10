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

#include "../gdscript_bytecode_serializer.h"
#include "../gdscript_cache.h"
#include "gdscript_test_runner.h"

#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/object/script_language.h"
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

static void write_text_file(const String &p_path, const String &p_text) {
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
	REQUIRE_MESSAGE(file.is_valid(), vformat("Failed to open '%s' for writing.", p_path));
	file->store_string(p_text);
	file->close();
}

static void write_binary_file(const String &p_path, const Vector<uint8_t> &p_bytes) {
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE);
	REQUIRE_MESSAGE(file.is_valid(), vformat("Failed to open '%s' for writing.", p_path));
	file->store_buffer(p_bytes.ptr(), p_bytes.size());
	file->close();
}

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

TEST_CASE("[Modules][GDScript] Bytecode remap preserves self refs and static typed arrays") {
	GDScriptLanguage::get_singleton()->init();

	const String script_path = TestUtils::get_temp_path("gdscript_bytecode_regression.gd");
	const String bytecode_path = script_path.get_basename() + ".gdb";
	const String remap_path = script_path + ".remap";

	String source = R"(
extends RefCounted
class_name BytecodeRegression
const SELF = preload("__SELF_PATH__")

static var registry := []
static var typed_registry: Array[String] = []
static var scripted_registry: Array[BytecodeRegression] = []

static func registry_size() -> int:
	return typed_registry.size()

static func builtin_typed_array_roundtrip() -> bool:
	var actions: Array[String] = []
	actions.append("jump")
	var copy: Array[String] = actions
	return copy.size() == 1 and copy[0] == "jump"

static func builtin_object_typed_array_roundtrip() -> bool:
	var events: Array[InputEvent] = []
	events.append(InputEventKey.new())
	var copy: Array[InputEvent] = events
	return copy.size() == 1 and copy[0] is InputEventKey

static func self_class_name_roundtrip() -> bool:
	var instance = BytecodeRegression.new()
	return instance.get_script() == SELF

static func script_typed_array_roundtrip() -> bool:
	var entries: Array[BytecodeRegression] = []
	entries.append(BytecodeRegression.new())
	var copy: Array[BytecodeRegression] = entries
	return copy.size() == 1 and copy[0].get_script() == SELF

static func typed_array_roundtrip() -> bool:
	var instance = SELF.new()
	return instance.get_script() == SELF

static func add_self() -> bool:
	registry.append(SELF.new())
	scripted_registry.append(BytecodeRegression.new())
	typed_registry.append("ok")
	return registry.size() == 1 and registry[0].get_script() == SELF and scripted_registry.size() == 1 and scripted_registry[0].get_script() == SELF and typed_registry.size() == 1
)";
	source = source.replace("__SELF_PATH__", script_path.c_escape());

	write_text_file(script_path, source);
	ScriptServer::remove_global_class_by_path(script_path);

	struct GlobalClassCleanup {
		String path;

		~GlobalClassCleanup() {
			if (!path.is_empty()) {
				ScriptServer::remove_global_class_by_path(path);
			}
		}
	} global_class_cleanup { script_path };

	String base_type;
	bool is_abstract = false;
	bool is_tool = false;
	String class_name = GDScriptLanguage::get_singleton()->get_global_class_name(script_path, &base_type, nullptr, &is_abstract, &is_tool);
	REQUIRE_MESSAGE(!class_name.is_empty(), "Temp bytecode regression script should expose a global class name.");
	ScriptServer::add_global_class(class_name, base_type, GDScriptLanguage::get_singleton()->get_name(), script_path, is_abstract, is_tool);

	GDScriptCache::remove_script(script_path);

	Error err = OK;
	Ref<GDScript> source_script = ResourceLoader::load(script_path, "GDScript", ResourceFormatLoader::CACHE_MODE_IGNORE, &err);
	REQUIRE_MESSAGE(err == OK, "Source script should load before bytecode export.");
	REQUIRE(source_script.is_valid());
	REQUIRE(source_script->is_valid());

	const Vector<uint8_t> bytecode = GDScriptBytecodeSerializer::serialize_script(source_script.ptr());
	REQUIRE_MESSAGE(!bytecode.is_empty(), "Bytecode export should produce data.");

	write_binary_file(bytecode_path, bytecode);
	write_text_file(remap_path, vformat("[remap]\npath=\"%s\"\n", bytecode_path));

	GDScriptCache::remove_script(script_path);
	if (Ref<Resource> cached = ResourceCache::get_ref(script_path); cached.is_valid()) {
		cached->set_path(String());
	}

	Ref<GDScript> bytecode_script = ResourceLoader::load(script_path, "GDScript", ResourceFormatLoader::CACHE_MODE_REPLACE, &err);
	REQUIRE_MESSAGE(err == OK, "Bytecode-remapped script should load successfully.");
	REQUIRE(bytecode_script.is_valid());
	REQUIRE(bytecode_script->is_valid());

	Array no_args;
	Variant result = bytecode_script->callv("registry_size", no_args);
	REQUIRE(result.get_type() == Variant::INT);
	CHECK(int(result) == 0);

	result = bytecode_script->callv("builtin_typed_array_roundtrip", no_args);
	REQUIRE(result.get_type() == Variant::BOOL);
	CHECK(bool(result));

	result = bytecode_script->callv("builtin_object_typed_array_roundtrip", no_args);
	REQUIRE(result.get_type() == Variant::BOOL);
	CHECK(bool(result));

	result = bytecode_script->callv("self_class_name_roundtrip", no_args);
	REQUIRE(result.get_type() == Variant::BOOL);
	CHECK(bool(result));

	result = bytecode_script->callv("script_typed_array_roundtrip", no_args);
	REQUIRE(result.get_type() == Variant::BOOL);
	CHECK(bool(result));

	result = bytecode_script->callv("typed_array_roundtrip", no_args);
	REQUIRE(result.get_type() == Variant::BOOL);
	CHECK(bool(result));

	result = bytecode_script->callv("add_self", no_args);
	REQUIRE(result.get_type() == Variant::BOOL);
	CHECK(bool(result));

	result = bytecode_script->callv("registry_size", no_args);
	REQUIRE(result.get_type() == Variant::INT);
	CHECK(int(result) == 1);

	GDScriptCache::remove_script(script_path);
	if (bytecode_script.is_valid()) {
		bytecode_script->set_path(String());
	}
	if (source_script.is_valid()) {
		source_script->set_path(String());
	}
	if (Ref<Resource> cached = ResourceCache::get_ref(script_path); cached.is_valid()) {
		cached->set_path(String());
	}
}

TEST_CASE("[Modules][GDScript] Bytecode serializer roundtrip preserves runtime and dump") {
	GDScriptLanguage::get_singleton()->init();

	const String script_path = TestUtils::get_temp_path("gdscript_bytecode_roundtrip.gd");
	const String roundtrip_script_path = TestUtils::get_temp_path("gdscript_bytecode_roundtrip_copy.gd");

	const String source = R"(
extends RefCounted

func helper(value: int) -> int:
	return value * 2

func exercise(seed: int = 2) -> Dictionary:
	var total := seed
	var history: Array[String] = []
	for i in range(3):
		total += helper(i)

	if total > 3:
		history.append(str(total))
	else:
		total -= 1

	var adder := func(extra: int) -> int:
		return total + extra

	return {
		"total": total,
		"lambda": adder.call(1),
		"history": history.duplicate()
	}
)";

	write_text_file(script_path, source);
	GDScriptCache::remove_script(script_path);

	Error err = OK;
	Ref<GDScript> source_script = ResourceLoader::load(script_path, "GDScript", ResourceFormatLoader::CACHE_MODE_IGNORE, &err);
	REQUIRE_MESSAGE(err == OK, "Source script should load before bytecode roundtrip.");
	REQUIRE(source_script.is_valid());
	REQUIRE(source_script->is_valid());

	const String source_dump = GDScriptBytecodeSerializer::dump_script_text(source_script.ptr());
	const Vector<uint8_t> bytecode = GDScriptBytecodeSerializer::serialize_script(source_script.ptr());
	REQUIRE_MESSAGE(!bytecode.is_empty(), "Bytecode roundtrip should produce serialized data.");

	Ref<GDScript> roundtrip_script = memnew(GDScript);
	roundtrip_script->set_path(roundtrip_script_path);
	const Error deserialize_error = GDScriptBytecodeSerializer::deserialize_script(bytecode, roundtrip_script.ptr());
	REQUIRE(deserialize_error == OK);
	REQUIRE(roundtrip_script->is_valid());

	const String roundtrip_dump = GDScriptBytecodeSerializer::dump_script_text(roundtrip_script.ptr()).replace(roundtrip_script_path, script_path);
	CHECK(source_dump == roundtrip_dump);

	Ref<RefCounted> source_instance = memnew(RefCounted);
	source_instance->set_script(source_script);

	Ref<RefCounted> roundtrip_instance = memnew(RefCounted);
	roundtrip_instance->set_script(roundtrip_script);

	Array args;
	args.push_back(5);

	const Variant source_result = source_instance->callv("exercise", args);
	const Variant roundtrip_result = roundtrip_instance->callv("exercise", args);
	REQUIRE(source_result.get_type() == Variant::DICTIONARY);
	REQUIRE(roundtrip_result.get_type() == Variant::DICTIONARY);
	CHECK(Dictionary(source_result) == Dictionary(roundtrip_result));

	source_script->set_path(String());
	roundtrip_script->set_path(String());
}

} // namespace GDScriptTests

/**************************************************************************/
/*  test_refactor.h                                                       */
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

#ifdef TOOLS_ENABLED

#include "tests/test_macros.h"

#include "../gdscript.h"
#include "gdscript_test_runner.h"

#include "core/config/project_settings.h"
#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "core/io/resource_loader.h"
#include "core/object/script_language.h"
#include "core/string/string_builder.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/builtin_fonts.gen.h"
#include "scene/resources/packed_scene.h"
#include "scene/theme/theme_db.h"
#include "servers/text/text_server.h"

#define FAIL_COND_V_MSG(cond, return_value, msg) \
	if (cond) {                                  \
		FAIL(msg);                               \
		return (return_value);                   \
	}                                            \
	(void)0
#define FAIL_COND_MSG(cond, msg) \
	FAIL_COND_V_MSG(cond, (void)0, msg)

namespace GDScriptTests {
namespace TestRefactor {

const int INDENT_SIZE = 4;

static void setup_global_classes(const String &p_path);

static String process_code(const String &p_code, const Point2i &p_position) {
	String code_with_sentinel;
	String sentinel_character = String::chr(0xFFFF);

	Ref<TextServer> text_server = TextServerManager::get_singleton()->get_primary_interface();
	int refactor_line = p_position.y;
	int refactor_column = p_position.x;

	RID code_shaped_text = text_server->create_shaped_text();
	FAIL_COND_V_MSG(code_shaped_text == RID(), p_code, "Creating text buffer failed.");
	RID font = text_server->create_font();
	FAIL_COND_V_MSG(font == RID(), p_code, "Creating font failed.");
	text_server->font_set_data_ptr(font, _font_Inter_Regular, _font_Inter_Regular_size);
	text_server->font_set_allow_system_fallback(font, true);
	Array fonts = { font };
	text_server->shaped_text_add_string(code_shaped_text, p_code, fonts, 12);

	PackedInt32Array code_lines = text_server->shaped_text_get_line_breaks(code_shaped_text, 0);
	int code_lines_index = 0;
	int current_line = 1;
	int current_column = 1;
	int current_index = 0;
	String current_char;

	for (code_lines_index = 0; code_lines_index < code_lines.size(); code_lines_index += 2) {
		current_index = code_lines[code_lines_index];
		ERR_FAIL_COND_V(current_line > refactor_line, p_code);
		if (current_line < refactor_line) {
			current_index = code_lines[code_lines_index + 1];
			current_line += 1;
			continue;
		}

		for (; current_index <= code_lines[code_lines_index + 1]; current_index++) {
			ERR_FAIL_COND_V(current_column > refactor_column, p_code);
			if (current_column == refactor_column) {
				code_with_sentinel = p_code.insert(current_index, sentinel_character);
				goto after_inserting_sentinel_character;
			}

			current_char = p_code.substr(current_index, 1);
			current_column += current_char == "\t"
					? INDENT_SIZE
					: 1;
		}
	}

after_inserting_sentinel_character:

	text_server->free_rid(code_shaped_text);
	text_server->free_rid(font);

	return code_with_sentinel;
}

static void run_test_cfg(const String &p_config_path) {
#define CHECK_REFACTOR_DICT_ENTRY(dict_element, key_name, variant_type)                  \
	FAIL_COND_MSG(!dict_element.has(#key_name), "refactor." #key_name " doesn't exist"); \
	FAIL_COND_MSG(dict_element[#key_name].get_type() != Variant::Type::variant_type, "refactor." #key_name " is not of variant type " #variant_type)
#define CHECK_REFACTOR_DICT_ENTRY_POSITION(dict_element, key_name) \
	CHECK_REFACTOR_DICT_ENTRY(dict_element, key_name, INT);        \
	FAIL_COND_MSG(((int)dict_element[#key_name]) < 1, "refactor." #key_name " is invalid");
#define CHECK_REFACTOR_DICT_ENTRY_STRING_NOT_EMPTY(dict_element, key_name) \
	CHECK_REFACTOR_DICT_ENTRY(dict_element, key_name, STRING);             \
	FAIL_COND_MSG(((String)dict_element[#key_name]).is_empty(), "refactor." #key_name " is invalid")

	typedef ScriptLanguage::RefactorRenameSymbolResult::Match Match;
	typedef LocalVector<Match> MatchList;
	typedef HashMap<String, MatchList> MatchMap;

	String config_path = ProjectSettings::get_singleton()->localize_path(p_config_path);

	String file_message = vformat("[File] %s", config_path);
	SUBCASE(file_message.utf8().get_data()) {
		ConfigFile conf;
		FAIL_COND_MSG(conf.load(config_path) != OK, vformat("No config file found at \"%s\".", config_path));

		EditorSettings::get_singleton()->set_setting("text_editor/completion/use_single_quotes", conf.get_value("input", "use_single_quotes", false));
		EditorSettings::get_singleton()->set_setting("text_editor/completion/add_node_path_literals", conf.get_value("input", "add_node_path_literals", false));
		EditorSettings::get_singleton()->set_setting("text_editor/completion/add_string_name_literals", conf.get_value("input", "add_string_name_literals", false));

		// [input] refactor
		FAIL_COND_MSG(conf.get_value("input", "refactor", Dictionary()).get_type() != Variant::Type::DICTIONARY, "input.load is not of type Dictionary");
		Dictionary refactor = conf.get_value("input", "refactor", Dictionary());

		CHECK_REFACTOR_DICT_ENTRY_STRING_NOT_EMPTY(refactor, file);
		CHECK_REFACTOR_DICT_ENTRY_POSITION(refactor, line);
		CHECK_REFACTOR_DICT_ENTRY_POSITION(refactor, column);
		// CHECK_REFACTOR_DICT_ENTRY_STRING_NOT_EMPTY(refactor, symbol);

		String refactor_file = refactor["file"];
		int refactor_line = refactor["line"];
		int refactor_column = refactor["column"];
		String refactor_symbol = refactor["symbol"];
		String refactor_to;

		if (refactor.has("to")) {
			CHECK_REFACTOR_DICT_ENTRY_STRING_NOT_EMPTY(refactor, to);
			refactor_to = refactor["to"];
		}

		// [output] matches
		MatchMap expected_result_matches;
		Dictionary expected_result_matches_dict = conf.get_value("output", "matches", Dictionary());
		for (Variant matches_array_key_variant : expected_result_matches_dict.keys()) {
			FAIL_COND_MSG(matches_array_key_variant.get_type() != Variant::Type::STRING, "match key is not a String");
			FAIL_COND_MSG(expected_result_matches_dict[matches_array_key_variant].get_type() != Variant::Type::ARRAY, "match value is not an Array");
			String matches_array_key = matches_array_key_variant;
			Array matches_array = expected_result_matches_dict[matches_array_key_variant];
			for (Variant match_variant : matches_array) {
				FAIL_COND_MSG(match_variant.get_type() != Variant::Type::DICTIONARY, "match entry is not a Dictionary");
				Dictionary match_dict = match_variant;
				CHECK_REFACTOR_DICT_ENTRY_POSITION(match_dict, start_line);
				CHECK_REFACTOR_DICT_ENTRY_POSITION(match_dict, start_column);
				CHECK_REFACTOR_DICT_ENTRY_POSITION(match_dict, end_line);
				CHECK_REFACTOR_DICT_ENTRY_POSITION(match_dict, end_column);

				Match match = { match_dict["start_line"],
					match_dict["start_column"],
					match_dict["end_line"],
					match_dict["end_column"] };
				expected_result_matches[matches_array_key_variant]
						.push_back(match);
			}
		}

		// [output] outside_refactor
		FAIL_COND_MSG(conf.get_value("output", "outside_refactor", false).get_type() != Variant::Type::BOOL, "output.outside_refactor is not of type bool");
		bool outside_refactor = conf.get_value("output", "outside_refactor", false);
		// [output] refactor_has_failed
		FAIL_COND_MSG(conf.get_value("output", "refactor_failed", false).get_type() != Variant::Type::BOOL, "output.refactor_failed is not of type bool");
		bool refactor_has_failed = conf.get_value("output", "refactor_failed", false);
		// [output] refactor_result_type
		FAIL_COND_MSG(conf.get_value("output", "refactor_result_type", "").get_type() != Variant::Type::STRING, "output.refactor_result_type is not of type String");
		String refactor_result_type = conf.get_value("output", "refactor_result_type", "");

		Ref<GDScript> gdscript = ResourceLoader::load(refactor_file);
		FAIL_COND_MSG(gdscript.is_null(), vformat("couldn't load script \"%s\"", refactor_file));

		String code = gdscript->get_source_code();
		String code_with_sentinel = process_code(code, Point2i(refactor_column, refactor_line));
		FAIL_COND_MSG(code == code_with_sentinel, "couldn't add sentinel character, code is identical");

		ScriptLanguage::RefactorRenameSymbolResult refactor_result;
		const HashMap<String, String> unsaved_scripts_code;
		Node *owner = nullptr;

		Error refactor_error_result = GDScriptLanguage::get_singleton()->refactor_rename_symbol_code(code_with_sentinel, refactor_file, owner, unsaved_scripts_code, refactor_result);
		FAIL_COND_MSG(refactor_error_result != OK, vformat("could not refactor rename symbol code: %s", error_names[refactor_error_result]));

		FAIL_COND_MSG(refactor_result.symbol != refactor_symbol, vformat("symbol found (%s) doesn't match with the symbol specified in the configuration file (%s)", refactor_result.symbol, refactor_symbol));

		CHECK(outside_refactor == refactor_result.outside_refactor);
		if (!outside_refactor) {
			CHECK(refactor_has_failed == refactor_result.has_failed());
		}

		if (!refactor_to.is_empty()) {
			CHECK(refactor_to == refactor_result.new_symbol);
		}

		if (!refactor_result_type.is_empty()) {
			String refactor_result_type_as_string;
			switch (refactor_result.type) {
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_NONE: {
					refactor_result_type_as_string = "NONE";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_CONTROL_FLOW: {
					refactor_result_type_as_string = "CONTROL_FLOW";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_LITERAL: {
					refactor_result_type_as_string = "LITERAL";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_KEYWORD: {
					refactor_result_type_as_string = "KEYWORD";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_NATIVE: {
					refactor_result_type_as_string = "NATIVE";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_NOT_EXPOSED: {
					refactor_result_type_as_string = "NOT_EXPOSED";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_SCRIPT: {
					refactor_result_type_as_string = "SCRIPT";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_GLOBAL_CLASS_NAME: {
					refactor_result_type_as_string = "GLOBAL_CLASS_NAME";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_CLASS_NAME: {
					refactor_result_type_as_string = "CLASS_NAME";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_CLASS_CONSTANT: {
					refactor_result_type_as_string = "CLASS_CONSTANT";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_CLASS_PROPERTY: {
					refactor_result_type_as_string = "CLASS_PROPERTY";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_CLASS_METHOD: {
					refactor_result_type_as_string = "CLASS_METHOD";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_CLASS_SIGNAL: {
					refactor_result_type_as_string = "CLASS_SIGNAL";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_CLASS_ENUM: {
					refactor_result_type_as_string = "CLASS_ENUM";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_CLASS_ENUM_VALUE: {
					refactor_result_type_as_string = "CLASS_ENUM_VALUE";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_CLASS_ANNOTATION: {
					refactor_result_type_as_string = "CLASS_ANNOTATION";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_LOCAL_CONSTANT: {
					refactor_result_type_as_string = "LOCAL_CONSTANT";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_LOCAL_VARIABLE: {
					refactor_result_type_as_string = "LOCAL_VARIABLE";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_LOCAL_FOR_VARIABLE: {
					refactor_result_type_as_string = "LOCAL_FOR_VARIABLE";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_LOCAL_PATTERN_BIND: {
					refactor_result_type_as_string = "LOCAL_PATTERN_BIND";
				} break;
				case ScriptLanguage::REFACTOR_RENAME_SYMBOL_RESULT_MAX: {
					refactor_result_type_as_string = "MAX";
				} break;
			}
			CHECK(refactor_result_type == refactor_result_type_as_string);
		}

		auto indent_string = [](const String &p_text, uint8_t p_size) -> String {
			PackedStringArray indent_result;
			for (const String &line : p_text.split("\n")) {
				indent_result.append(String(" ").repeat(p_size) + line);
			}
			return String("\n").join(indent_result);
		};

		auto matchlist_to_string = [&indent_string](const MatchList &p_match_list) -> String {
			StringBuilder matchlist_string;
			matchlist_string.append("[\n");

			bool first_match = true;
			for (const Match &match : p_match_list) {
				if (first_match) {
					first_match = false;
				} else {
					matchlist_string.append(",\n");
				}
				matchlist_string.append(indent_string(match.to_string(), 4));
			}
			matchlist_string.append("\n]");
			return matchlist_string.as_string();
		};

		auto matchmap_to_string = [&matchlist_to_string, &indent_string](const MatchMap &p_match_map) -> String {
			StringBuilder match_string;
			match_string.append("{\n");
			bool first_match_entry = true;
			for (KeyValue<String, LocalVector<Match>> KV : p_match_map) {
				if (first_match_entry) {
					first_match_entry = false;
				} else {
					match_string.append(",\n");
				}

				StringBuilder match_file_entry;
				match_file_entry.append(vformat("\"%s\": %s", KV.key, matchlist_to_string(KV.value)));
				match_string.append(indent_string(match_file_entry.as_string(), 4));
			}

			match_string.append("\n}\n");
			return match_string.as_string();
		};

		expected_result_matches.sort();
		refactor_result.matches.sort();

		// Comparing results.
		for (KeyValue<String, LocalVector<Match>> KV : refactor_result.matches) {
			String refactor_result_key = KV.key;
			MatchList refactor_result_key_matches = KV.value;

			if (expected_result_matches.size() > 0) {
				INFO(vformat("\nGot: %s", matchlist_to_string(refactor_result_key_matches)));
				CHECK_MESSAGE(expected_result_matches.has(refactor_result_key), vformat("unexpected match for \"%s\" not defined in the cfg", refactor_result_key));
			}

			for (Match &match : KV.value) {
				bool found_match = false;
				if (expected_result_matches.has(KV.key)) {
					for (Match &expected_result_match : expected_result_matches[KV.key]) {
						if (match != expected_result_match) {
							continue;
						}
						found_match = true;
						break;
					}
				} else {
					MESSAGE(vformat("[DEBUG] %s: %s", KV.key, match.to_string()));
					found_match = true;
				}

				CHECK_MESSAGE(found_match, vformat("got a %s for \"%s\", but it wasn't expected", match.to_string(), KV.key));
			}

			if (expected_result_matches.has(refactor_result_key)) {
				INFO(vformat("\nFor \"%s\":\nExpected: %s\nGot: %s", refactor_result_key, matchlist_to_string(expected_result_matches[refactor_result_key]), matchlist_to_string(refactor_result_key_matches)));
				CHECK_MESSAGE(expected_result_matches[refactor_result_key].size() == refactor_result_key_matches.size(), vformat("unexpected number of matches for \"%s\", expected %s but got %s", refactor_result_key, expected_result_matches[refactor_result_key].size(), refactor_result_key_matches.size()));
			}
		}

		INFO(vformat("\nExpected: %s\nGot: %s", matchmap_to_string(expected_result_matches), matchmap_to_string(refactor_result.matches)));
		CHECK(expected_result_matches.size() == refactor_result.matches.size());
	} // SUBCASE

#undef CHECK_REFACTOR_ENTRY_SYMBOL
#undef CHECK_REFACTOR_ENTRY_POSITION
#undef CHECK_REFACTOR_ENTRY
}

static void test_directory(const String &p_dir) {
	Error err = OK;
	Ref<DirAccess> dir = DirAccess::open(p_dir, &err);

	FAIL_COND_MSG(err != OK, "Invalid test directory.");

	String path = dir->get_current_dir();

	dir->list_dir_begin();
	String next = dir->get_next();

	while (!next.is_empty()) {
		if (dir->current_is_dir()) {
			if (next == "." || next == "..") {
				goto next;
			}
			if (next == ".godot") {
				goto next;
			}
			test_directory(path.path_join(next));
		} else if (next.ends_with(".cfg")) {
			String element_path = ProjectSettings::get_singleton()->localize_path(path.path_join(next));
			run_test_cfg(element_path);
		}
	next:
		next = dir->get_next();
	}
}

static void setup_global_classes(const String &p_dir) {
	Error err = OK;
	Ref<DirAccess> dir(DirAccess::open(p_dir, &err));

	FAIL_COND_MSG(err != OK, vformat("failed to open directory %s", p_dir));

	String current_dir = dir->get_current_dir();

	dir->list_dir_begin();
	String next = dir->get_next();

	StringName gdscript_singleton_name = GDScriptLanguage::get_singleton()->get_name();
	while (!next.is_empty()) {
		if (dir->current_is_dir()) {
			if (next == "." || next == "..") {
				goto next;
			}
			if (next == ".godot") {
				goto next;
			}
			setup_global_classes(current_dir.path_join(next));
		} else {
			if (!next.ends_with(".gd")) {
				goto next;
			}

			String element_path = ProjectSettings::get_singleton()->localize_path(current_dir.path_join(next));
			String base_type;
			bool is_abstract = false;
			bool is_tool = false;
			String class_name = GDScriptLanguage::get_singleton()->get_global_class_name(element_path, &base_type, nullptr, &is_abstract, &is_tool);
			if (class_name.is_empty()) {
				goto next;
			}
			FAIL_COND_MSG(ScriptServer::is_global_class(class_name),
					vformat("Class name '%s' from %s is already used in %s", class_name, element_path, ScriptServer::get_global_class_path(class_name)));

			ScriptServer::add_global_class(class_name, base_type, gdscript_singleton_name, element_path, is_abstract, is_tool);
			INFO(vformat("Added global class \"%s\" (from \"%s\")", class_name, element_path));
		}
	next:
		next = dir->get_next();
	}

	dir->list_dir_end();
}

static void load_scenes(const String &p_dir) {
	Error err = OK;
	Ref<DirAccess> dir(DirAccess::open(p_dir, &err));

	FAIL_COND_MSG(err != OK, vformat("failed to open directory %s", p_dir));

	String current_dir = dir->get_current_dir();

	dir->list_dir_begin();
	String next = dir->get_next();

	StringName gdscript_singleton_name = GDScriptLanguage::get_singleton()->get_name();
	while (!next.is_empty()) {
		if (dir->current_is_dir()) {
			if (next == "." || next == "..") {
				goto next;
			}
			if (next == ".godot") {
				goto next;
			}
			load_scenes(current_dir.path_join(next));
		} else {
			if (!next.ends_with(".tscn")) {
				goto next;
			}

			String element_path = ProjectSettings::get_singleton()->localize_path(current_dir.path_join(next));

			Ref<PackedScene> scene = ResourceLoader::load(element_path);
			FAIL_COND_MSG(scene.is_null(), vformat("couldn't load PackedScene \"%s\"", element_path));

			// No need to do anything further, as loading a packed scene also loads its scripts.
		}
	next:
		next = dir->get_next();
	}

	dir->list_dir_end();
}

TEST_SUITE("[Modules][GDScript][Refactor tools]") {
	const String gdscript_tests_scripts_refactor_path = "modules/gdscript/tests/scripts/refactor";

	TEST_CASE("[Editor] Rename symbol") {
		EditorSettings::get_singleton()->set_setting("text_editor/completion/use_single_quotes", false);
		EditorSettings::get_singleton()->set_setting("text_editor/behavior/indent/size", INDENT_SIZE);

		const String gdscript_tests_scripts_refactor_rename_path = gdscript_tests_scripts_refactor_path.path_join("rename");

		init_project_dir(gdscript_tests_scripts_refactor_rename_path, "refactor_rename");
		init_language();

		setup_global_classes("res://");
		load_scenes("res://");
		test_directory("res://");

		finish_language();
	}
}

} // namespace TestRefactor
} // namespace GDScriptTests

#undef FAIL_COND_V_MSG
#undef FAIL_COND_MSG

#endif // TOOLS_ENABLED

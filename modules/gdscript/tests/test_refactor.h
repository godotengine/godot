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

#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/io/resource_loader.h"
#include "servers/text_server.h"
#ifdef TOOLS_ENABLED

#include "tests/test_macros.h"

#include "../gdscript.h"
#include "gdscript_test_runner.h"

#include "core/config/project_settings.h"
#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/object/script_language.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/builtin_fonts.gen.h"
#include "scene/resources/packed_scene.h"
#include "scene/theme/theme_db.h"

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

static void setup_global_classes(const String &p_path);

struct ProcessCodeData {
	String code;
	String symbol;
};

static ProcessCodeData process_code(const String &p_code, const Point2i &p_pos) {
	Ref<TextServer> text_server = TextServerManager::get_singleton()->get_primary_interface();
	int refactor_line = p_pos.y;
	int refactor_column = p_pos.x;

	RID code_shaped_text = text_server->create_shaped_text();
	FAIL_COND_V_MSG(code_shaped_text == RID(), ProcessCodeData(), "Creating text buffer failed.");
	RID font = text_server->create_font();
	FAIL_COND_V_MSG(font == RID(), ProcessCodeData(), "Creating font failed.");
	text_server->font_set_data_ptr(font, _font_NotoSans_Regular, _font_NotoSans_Regular_size);
	text_server->font_set_allow_system_fallback(font, true);
	Array fonts = { font };
	text_server->shaped_text_add_string(code_shaped_text, p_code, fonts, 12);
	PackedInt32Array code_lines = text_server->shaped_text_get_line_breaks(code_shaped_text, 0);

	String code_with_sentinel = p_code;
	String found_word;
	int current_line = 1;
	int current_index = 0;

	for (int i = 0; i < code_lines.size(); i += 2) {
		current_index = code_lines[i];

		if (current_line == refactor_line) {
			String line = p_code.substr(code_lines[i], code_lines[i + 1] - code_lines[i]);
			RID line_shaped_text = text_server->create_shaped_text();
			text_server->shaped_text_add_string(line_shaped_text, line, fonts, 12);

			PackedInt32Array words = text_server->shaped_text_get_word_breaks(line_shaped_text);
			for (int j = 0; j < words.size(); j += 2) {
				int refactor_column_index = (refactor_column - 1);
				if (refactor_column_index < words[j] || refactor_column_index > words[j + 1]) {
					continue;
				}

				found_word = line.substr(words[j], words[j + 1] - words[j]);
				current_index += refactor_column_index;

				code_with_sentinel = p_code.insert(current_index, String::chr(0xFFFF));
				break;
			}

			text_server->free_rid(line_shaped_text);
		} else {
			current_index = code_lines[i + 1];
		}

		if (!found_word.is_empty()) {
			break;
		}

		current_line += 1;
	}

	text_server->free_rid(code_shaped_text);
	text_server->free_rid(font);

	ProcessCodeData return_value;
	return_value.code = code_with_sentinel;
	return_value.symbol = found_word;

	return return_value;
}

static void run_test_cfg(const String &p_config_path) {
#define CHECK_REFACTOR_DICT_ENTRY(dict_element, key_name, variant_type)                  \
	FAIL_COND_MSG(!dict_element.has(#key_name), "refactor." #key_name " doesn't exist"); \
	FAIL_COND_MSG(dict_element[#key_name].get_type() != Variant::Type::variant_type, "refactor." #key_name " is not of variant type" #variant_type)
#define CHECK_REFACTOR_DICT_ENTRY_POSITION(dict_element, key_name) \
	CHECK_REFACTOR_DICT_ENTRY(dict_element, key_name, INT);        \
	FAIL_COND_MSG(((int)dict_element[#key_name]) < 1, "refactor." #key_name " is invalid");
#define CHECK_REFACTOR_DICT_ENTRY_STRING_NOT_EMPTY(dict_element, key_name) \
	CHECK_REFACTOR_DICT_ENTRY(dict_element, key_name, STRING);             \
	FAIL_COND_MSG(((String)dict_element[#key_name]).is_empty(), "refactor." #key_name " is invalid")

	String config_path = ProjectSettings::get_singleton()->localize_path(p_config_path);

	String file_message = vformat("[File] %s", config_path);
	// SUBCASE(file_message.utf8().get_data()) {
	INFO(vformat("=> %s", file_message));
	if (true) {
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
		CHECK_REFACTOR_DICT_ENTRY_STRING_NOT_EMPTY(refactor, symbol);

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
		Dictionary expected_result_matches = conf.get_value("output", "matches", Dictionary());
		for (Variant matches_array_key : expected_result_matches.keys()) {
			FAIL_COND_MSG(matches_array_key.get_type() != Variant::Type::STRING, "match key is not a String");
			FAIL_COND_MSG(expected_result_matches[matches_array_key].get_type() != Variant::Type::ARRAY, "match value is not an Array");
			Array matches_array = expected_result_matches[matches_array_key];
			for (Variant match : matches_array) {
				FAIL_COND_MSG(match.get_type() != Variant::Type::DICTIONARY, "match entry is not a Dictionary");
				Dictionary match_dict = match;

				CHECK_REFACTOR_DICT_ENTRY_POSITION(match_dict, start_line);
				CHECK_REFACTOR_DICT_ENTRY_POSITION(match_dict, start_column);
				CHECK_REFACTOR_DICT_ENTRY_POSITION(match_dict, end_line);
				CHECK_REFACTOR_DICT_ENTRY_POSITION(match_dict, end_column);
			}
		}

		Ref<GDScript> gdscript = ResourceLoader::load(refactor_file);
		FAIL_COND_MSG(gdscript.is_null(), vformat("couldn't load script \"%s\"", refactor_file));

		ProcessCodeData data = process_code(gdscript->get_source_code(), Point2i(refactor_column, refactor_line));
		FAIL_COND_MSG(data.symbol != refactor_symbol, vformat("symbol found (%s) doesn't match with the symbol specified in the configuration file (%s)", data.symbol, refactor_symbol));

		ScriptLanguage::RefactorRenameSymbolResult refactor_result;
		const HashMap<String, String> unsaved_scripts_code;
		Node *owner = nullptr;

		Error refactor_error_result = GDScriptLanguage::get_singleton()->refactor_rename_symbol_code(data.code, data.symbol, refactor_file, owner, unsaved_scripts_code, refactor_result);
		FAIL_COND_MSG(refactor_error_result != OK, vformat("could not refactor rename symbol code: %s", error_names[refactor_error_result]));

		// Comparing results.
		if (expected_result_matches.size() == 0) {
			FAIL(vformat("no matches set in \"%s\"", config_path));
		}

		for (KeyValue<String, LocalVector<ScriptLanguage::RefactorRenameSymbolResult::Match>> KV : refactor_result.matches) {
			Array expected_result_matches_array;
			if (expected_result_matches.size() > 0) {
				CHECK_MESSAGE(expected_result_matches.has(KV.key), vformat("unexpected match for \"%s\" not defined in the cfg", KV.key));
				CHECK_MESSAGE(expected_result_matches[KV.key].get_type() == Variant::Type::ARRAY, vformat("number of matches don't match the expected result for \"%s\"", KV.key));
				CHECK_MESSAGE((uint32_t)((Array)expected_result_matches[KV.key]).size() == KV.value.size(), vformat("number of matches don't match the expected result for \"%s\"", KV.key));
				expected_result_matches_array = expected_result_matches[KV.key];
			}

			for (ScriptLanguage::RefactorRenameSymbolResult::Match match : KV.value) {
				bool found_result = false;

				if (expected_result_matches.size() > 0) {
					for (Dictionary expected_result_match : expected_result_matches_array) {
						ScriptLanguage::RefactorRenameSymbolResult::Match refactor_match = {
							expected_result_match["start_line"],
							expected_result_match["start_column"],
							expected_result_match["end_line"],
							expected_result_match["end_column"]
						};
						if (refactor_match == match) {
							found_result = true;
							break;
						}
					}
				} else {
					INFO(vformat("[DEBUG] %s: %s", KV.key, match.to_string()));
					found_result = true;
				}

				CHECK_MESSAGE(found_result, vformat("got a %s for \"%s\", but it wasn't expected", match.to_string(), KV.key));
			}
		}
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
		EditorSettings::get_singleton()->set_setting("text_editor/behavior/indent/size", 4);

		const String gdscript_tests_scripts_refactor_rename_path = gdscript_tests_scripts_refactor_path.path_join("rename");

		init_language(gdscript_tests_scripts_refactor_rename_path, "refactor_rename", "res://");

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

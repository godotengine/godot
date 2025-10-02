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

static void test_script(const String &p_code, const String &p_path, const String &p_config_path, const String &p_friendly_name = "") {
#define CHECK_REFACTOR_DICT_ENTRY(dict_element, key_name, variant_type)                  \
	FAIL_COND_MSG(!dict_element.has(#key_name), "refactor." #key_name " doesn't exist"); \
	FAIL_COND_MSG(dict_element[#key_name].get_type() != Variant::Type::variant_type, "refactor." #key_name " is not of variant type" #variant_type)
#define CHECK_REFACTOR_DICT_ENTRY_POSITION(dict_element, key_name) \
	CHECK_REFACTOR_DICT_ENTRY(dict_element, key_name, INT);        \
	FAIL_COND_MSG(((int)dict_element[#key_name]) < 1, "refactor." #key_name " is invalid");
#define CHECK_REFACTOR_DICT_ENTRY_SYMBOL(dict_element, key_name) \
	CHECK_REFACTOR_DICT_ENTRY(dict_element, key_name, STRING);   \
	FAIL_COND_MSG(((String)dict_element[#key_name]).is_empty(), "refactor." #key_name " is invalid")

	String path = ProjectSettings::get_singleton()->localize_path(p_path);
	String config_path = ProjectSettings::get_singleton()->localize_path(p_config_path);
	String subcase_path = path;
	if (!p_friendly_name.is_empty()) {
		subcase_path += vformat(" (%s)", p_friendly_name);
	}

	String file_message = vformat("[File] %s", subcase_path);
	// SUBCASE(file_message.utf8().get_data()) {
	MESSAGE(vformat("=> %s", file_message));
	if (true) {
		ConfigFile conf;
		FAIL_COND_MSG(conf.load(config_path) != OK, vformat("No config file found at \"%s\".", config_path));

		EditorSettings::get_singleton()->set_setting("text_editor/completion/use_single_quotes", conf.get_value("input", "use_single_quotes", false));
		EditorSettings::get_singleton()->set_setting("text_editor/completion/add_node_path_literals", conf.get_value("input", "add_node_path_literals", false));
		EditorSettings::get_singleton()->set_setting("text_editor/completion/add_string_name_literals", conf.get_value("input", "add_string_name_literals", false));

		// [input] load
		Variant resources_to_load_variant;
		if (conf.has_section_key("input", "resources_to_load")) {
			resources_to_load_variant = conf.get_value("input", "resources_to_load");
		}
		FAIL_COND_MSG(resources_to_load_variant.get_type() != Variant::Type::ARRAY && resources_to_load_variant.get_type() != Variant::Type::NIL, "input.resources_to_load is not of type Array");
		Array resources_to_load = resources_to_load_variant;
		LocalVector<Ref<Resource>> loaded_resources;
		for (Variant resource_to_load_variant : resources_to_load) {
			FAIL_COND_MSG(resource_to_load_variant.get_type() != Variant::Type::STRING, "input.load[] is not of type String");
			String resource_to_load = resource_to_load_variant;
			Ref<Resource> loaded_resource = ResourceLoader::load(resource_to_load);
			FAIL_COND_MSG(loaded_resource.is_null(), vformat("Couldn't load resource %s", resource_to_load));
			if (loaded_resource.is_valid()) {
				MESSAGE(vformat("Loaded resource %s", resource_to_load));
				loaded_resources.push_back(loaded_resource);
			}
		}

		// [input] refactor
		FAIL_COND_MSG(conf.get_value("input", "refactor", Dictionary()).get_type() != Variant::Type::DICTIONARY, "input.load is not of type Dictionary");
		Dictionary refactor = conf.get_value("input", "refactor", Dictionary());

		CHECK_REFACTOR_DICT_ENTRY_POSITION(refactor, line);
		CHECK_REFACTOR_DICT_ENTRY_POSITION(refactor, column);
		CHECK_REFACTOR_DICT_ENTRY_SYMBOL(refactor, symbol);

		int refactor_line = refactor["line"];
		int refactor_column = refactor["column"];
		String refactor_symbol = refactor["symbol"];
		String refactor_to;

		if (refactor.has("to")) {
			CHECK_REFACTOR_DICT_ENTRY_SYMBOL(refactor, to);
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

		ProcessCodeData data = process_code(p_code, Point2i(refactor_column, refactor_line));
		FAIL_COND_MSG(data.symbol != refactor_symbol, vformat("symbol found (%s) doesn't match with the symbol specified in the configuration file (%s)", data.symbol, refactor_symbol));

		ScriptLanguage::RefactorRenameSymbolResult refactor_result;
		const HashMap<String, String> unsaved_scripts_code;
		Node *owner = nullptr;

		Error refactor_error_result = GDScriptLanguage::get_singleton()->refactor_rename_symbol_code(data.code, data.symbol, p_path, owner, unsaved_scripts_code, refactor_result);
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
					MESSAGE(vformat("[DEBUG] %s: %s", KV.key, match.to_string()));
					found_result = true;
				}

				CHECK_MESSAGE(found_result, vformat("got a %s for \"%s\", but it wasn't expected", match.to_string(), KV.key));
			}
		}

		loaded_resources.clear();
	} // SUBCASE

#undef CHECK_REFACTOR_ENTRY_SYMBOL
#undef CHECK_REFACTOR_ENTRY_POSITION
#undef CHECK_REFACTOR_ENTRY
}

static void test_scene_file(const String &p_path) {
	FAIL_COND_MSG(!p_path.ends_with(".tscn"), vformat("\"%s\" is not a scene file", p_path));
	Ref<PackedScene> scene = ResourceLoader::load(p_path);
	FAIL_COND_MSG(scene.is_null(), vformat("couldn't load PackedScene \"%s\"", p_path));
	FAIL_COND_MSG(!scene->can_instantiate(), vformat("cannot instantiate PackedScene \"%s\"", p_path));

	// ERR_PRINT_OFF;
	Node *scene_instance = scene->instantiate();
	// ERR_PRINT_ON;

	auto process_scene_node = [&p_path](Node *p_node) -> void {
		Ref<GDScript> node_gdscript = p_node->get_script();
		if (node_gdscript.is_null()) {
			return;
		}
		String node_gdscript_name = node_gdscript->get_name();
		String cfg_path = vformat("%s__%s.cfg", p_path.get_basename(), node_gdscript_name);

		if (!FileAccess::exists(cfg_path)) {
			return;
		}

		test_script(node_gdscript->get_source_code(), node_gdscript->get_script_path(), cfg_path, node_gdscript_name);
	};

	process_scene_node(scene_instance);
	for (Node *child : scene_instance->iterate_children()) {
		process_scene_node(child);
	}

	scene_instance->queue_free();
}

static void test_script_file(const String &p_path) {
	FAIL_COND_MSG(!p_path.ends_with(".gd"), vformat("\"%s\" is not a script file", p_path));
	Error err = OK;
	Ref<FileAccess> acc = FileAccess::open(p_path, FileAccess::READ, &err);
	FAIL_COND_MSG(err != OK, vformat("couldn't open %s", p_path));

	test_script(acc->get_as_utf8_string(), p_path, p_path.get_basename() + ".cfg");
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
				next = dir->get_next();
				continue;
			}
			test_directory(path.path_join(next));
		} else if ((next.ends_with(".gd") &&
						   (!next.ends_with(".notest.gd") && !next.ends_with(".global.gd"))) ||
				next.ends_with(".tscn")) {
			String next_path = ProjectSettings::get_singleton()->localize_path(path.path_join(next));

			if (next_path.ends_with(".tscn")) {
				test_scene_file(next_path);
			} else {
				test_script_file(next_path);
			}
		}
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
			if (next == "." || next == ".." || next == "completion" || next == "lsp" || next == "refactor") {
				goto next;
			}
			setup_global_classes(current_dir.path_join(next));
		} else {
			if (!next.ends_with(".global.gd") /* && !next.ends_with(".tscn") */) {
				goto next;
			}

			String element_path = ProjectSettings::get_singleton()->localize_path(current_dir.path_join(next));

			auto add_global_class = [&gdscript_singleton_name](const String &p_path, const String &p_code = "") -> void {
				String base_type;
				bool is_abstract = false;
				bool is_tool = false;
				String class_name = p_code.is_empty()
						? GDScriptLanguage::get_singleton()->get_global_class_name(p_path, &base_type, nullptr, &is_abstract, &is_tool)
						: "";
				if (class_name.is_empty()) {
					return;
				}
				FAIL_COND_MSG(ScriptServer::is_global_class(class_name),
						vformat("Class name '%s' from %s is already used in %s", class_name, p_path, ScriptServer::get_global_class_path(class_name)));

				ScriptServer::add_global_class(class_name, base_type, gdscript_singleton_name, p_path, is_abstract, is_tool);
				MESSAGE(vformat("Added global class \"%s\" (from \"%s\")", class_name, p_path));
			};

			if (next.ends_with(".tscn")) {
				MESSAGE(vformat("before loading %s", element_path));

				Ref<PackedScene> scene = ResourceLoader::load(element_path);
				FAIL_COND_MSG(scene.is_null(), vformat("couldn't load PackedScene \"%s\"", element_path));
				FAIL_COND_MSG(!scene->can_instantiate(), vformat("cannot instantiate PackedScene \"%s\"", element_path));

				// ERR_PRINT_OFF;
				Node *scene_instance = scene->instantiate();
				// ERR_PRINT_ON;
				MESSAGE(vformat("instantiated %s", element_path));

				auto process_scene_node = [&add_global_class](Node *p_node) -> void {
					Ref<GDScript> node_gdscript = p_node->get_script();
					if (node_gdscript.is_null()) {
						return;
					}
					add_global_class(node_gdscript->get_script_path(), node_gdscript->get_source_code());
				};

				process_scene_node(scene_instance);
				for (Node *child : scene_instance->iterate_children()) {
					process_scene_node(child);
				}

				scene_instance->queue_free();
			} else {
				add_global_class(element_path);
			}
		}

	next:
		next = dir->get_next();
	}

	dir->list_dir_end();
}

TEST_SUITE("[Modules][GDScript][Refactor tools]") {
	TEST_CASE("[Editor] Rename symbol") {
		EditorSettings::get_singleton()->set_setting("text_editor/completion/use_single_quotes", false);
		init_language("modules/gdscript/tests/scripts");

		setup_global_classes("modules/gdscript/tests/scripts/refactor/rename/");
		test_directory("modules/gdscript/tests/scripts/refactor/rename/");

		finish_language();
	}
}

} // namespace TestRefactor
} // namespace GDScriptTests

#undef FAIL_COND_V_MSG
#undef FAIL_COND_MSG

#endif // TOOLS_ENABLED

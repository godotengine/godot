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

#define FAIL_COND_MSG(cond, msg) \
	if (cond) {                  \
		FAIL(msg);               \
		return;                  \
	}                            \
	(void)0

namespace GDScriptTests {
namespace TestRefactor {

// static bool match_option(const Dictionary p_expected, const ScriptLanguage::RefactorRenameSymbolResult p_got) {
// 	return true;
// }

struct ProcessCodeData {
	String code;
	String symbol;
};

static ProcessCodeData process_code(Ref<TextServer> text_server, const String &p_code, const Point2i &p_pos) {
	int refactor_line = p_pos.y;
	int refactor_column = p_pos.x;

	RID code_shaped_text = text_server->create_shaped_text();
	CHECK_FALSE_MESSAGE(code_shaped_text == RID(), "Creating text buffer failed.");
	RID font = text_server->create_font();
	CHECK_FALSE_MESSAGE(font == RID(), "Creating font failed.");
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

static void test_directory(const String &p_dir) {
	Error err = OK;
	Ref<DirAccess> dir = DirAccess::open(p_dir, &err);

	FAIL_COND_MSG(err != OK, "Invalid test directory.");

	Ref<TextServer> ts = TextServerManager::get_singleton()->get_primary_interface();
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
		} else if (next.ends_with(".gd") && !next.ends_with(".notest.gd") && !next.ends_with(".global.gd")) {
			Ref<FileAccess> acc = FileAccess::open(path.path_join(next), FileAccess::READ, &err);

			if (err != OK) {
				next = dir->get_next();
				continue;
			}

			String code = acc->get_as_utf8_string();

			String res_path = ProjectSettings::get_singleton()->localize_path(path.path_join(next));
			String cfg_path = ProjectSettings::get_singleton()->localize_path(path.path_join(next.get_basename() + ".cfg"));

			MESSAGE(vformat("Parsing \"%s\".", cfg_path));
			ConfigFile conf;
			FAIL_COND_MSG(conf.load(cfg_path) != OK, "No config file found.");

			EditorSettings::get_singleton()->set_setting("text_editor/completion/use_single_quotes", conf.get_value("input", "use_single_quotes", false));
			EditorSettings::get_singleton()->set_setting("text_editor/completion/add_node_path_literals", conf.get_value("input", "add_node_path_literals", false));
			EditorSettings::get_singleton()->set_setting("text_editor/completion/add_string_name_literals", conf.get_value("input", "add_string_name_literals", false));

			// [test] refactor
			Dictionary refactor = conf.get_value("test", "refactor", Dictionary());

			FAIL_COND_MSG(!refactor.has("line"), "refactor.line doesn't exist");
			FAIL_COND_MSG(!refactor.has("column"), "refactor.line doesn't exist");
			FAIL_COND_MSG(refactor["line"].get_type() != Variant::Type::INT, "refactor.line is not an int");
			FAIL_COND_MSG(refactor["column"].get_type() != Variant::Type::INT, "refactor.column is not an int");
			int refactor_line = refactor["line"];
			FAIL_COND_MSG(refactor_line < 1, "refactor_line is invalid");
			int refactor_column = refactor["column"];
			FAIL_COND_MSG(refactor_column < 1, "refactor_column is invalid");

			String refactor_to;
			if (refactor.has("to")) {
				FAIL_COND_MSG(refactor["line"].get_type() != Variant::Type::STRING, "refactor.to is not a String");
				refactor_to = refactor["to"];
			}

			// [expected_result] matches
			Dictionary expected_result_matches = conf.get_value("expected_result", "matches", Dictionary());
			for (Variant matches_array_key : expected_result_matches.keys()) {
				FAIL_COND_MSG(matches_array_key.get_type() != Variant::Type::STRING, "match key is not a String");
				FAIL_COND_MSG(expected_result_matches[matches_array_key].get_type() != Variant::Type::ARRAY, "match value is not an Array");
				Array matches_array = expected_result_matches[matches_array_key];
				for (Variant match : matches_array) {
					FAIL_COND_MSG(match.get_type() != Variant::Type::DICTIONARY, "match entry is not a Dictionary");
					Dictionary match_dict = match;

#define CHECK_MATCH_ENTRY(key_name)                                                                                      \
	FAIL_COND_MSG(!match_dict.has(#key_name), "match entry doesn't have \"" #key_name "\" key");                         \
	FAIL_COND_MSG(match_dict[#key_name].get_type() != Variant::Type::INT, "match entry \"" #key_name "\" isn't an int"); \
	FAIL_COND_MSG((int)(match_dict[#key_name]) < 1, "match entry \"" #key_name "\" is invalid");                         \
	(void)0

					CHECK_MATCH_ENTRY(start_line);
					CHECK_MATCH_ENTRY(start_column);
					CHECK_MATCH_ENTRY(end_line);
					CHECK_MATCH_ENTRY(end_column);

#undef CHECK_MATCH_ENTRY
				}
			}

			MESSAGE(vformat("Testing \"%s\".", res_path));

			ProcessCodeData data = process_code(ts, code, Point2i(refactor_column, refactor_line));

			ScriptLanguage::RefactorRenameSymbolResult refactor_result;
			const HashMap<String, String> unsaved_scripts_code;
			Node *owner = nullptr;

			Error refactor_error_result = GDScriptLanguage::get_singleton()->refactor_rename_symbol_code(data.code, data.symbol, res_path, owner, unsaved_scripts_code, refactor_result);
			FAIL_COND_MSG(refactor_error_result != OK, vformat("could not refactor rename symbol code: %s", error_names[refactor_error_result]));

			// Comparing results.
			if (expected_result_matches.size() > 0) {
				for (KeyValue<String, LocalVector<ScriptLanguage::RefactorRenameSymbolResult::Match>> KV : refactor_result.matches) {
					FAIL_COND_MSG(!expected_result_matches.has(KV.key), vformat("unexpected match for \"%s\" not defined in the cfg", KV.key));
					Array expected_result_matches_array = expected_result_matches[KV.key];
					FAIL_COND_MSG((uint32_t)expected_result_matches_array.size() != KV.value.size(), vformat("number of matches don't match the expected result for \"%s\"", KV.key));

					for (ScriptLanguage::RefactorRenameSymbolResult::Match match : KV.value) {
						bool found_result = false;

						for (Dictionary expected_result_match : expected_result_matches_array) {
							int start_line = expected_result_match["start_line"];
							int start_column = expected_result_match["start_column"];
							int end_line = expected_result_match["end_line"];
							int end_column = expected_result_match["end_column"];

							if (start_line == match.start_line &&
									start_column == match.start_column &&
									end_line == match.end_line &&
									end_column == match.end_column) {
								found_result = true;
								break;
							}
						}

						FAIL_COND_MSG(!found_result, vformat("got a %s, but it wasn't expected", match.to_string()));
					}
				}
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

	StringName gdscript_name = GDScriptLanguage::get_singleton()->get_name();
	while (!next.is_empty()) {
		if (dir->current_is_dir()) {
			if (next == "." || next == ".." || next == "completion" || next == "lsp" || next == "refactor") {
				next = dir->get_next();
				continue;
			}
			setup_global_classes(current_dir.path_join(next));
		} else {
			if (!next.ends_with(".global.gd")) {
				next = dir->get_next();
				continue;
			}

			String base_type;
			String source_file = ProjectSettings::get_singleton()->localize_path(current_dir.path_join(next));
			bool is_abstract = false;
			bool is_tool = false;
			String class_name = GDScriptLanguage::get_singleton()->get_global_class_name(source_file, &base_type, nullptr, &is_abstract, &is_tool);
			if (class_name.is_empty()) {
				next = dir->get_next();
				continue;
			}
			FAIL_COND_MSG(ScriptServer::is_global_class(class_name),
					vformat("Class name '%s' from %s is already used in %s", class_name, source_file, ScriptServer::get_global_class_path(class_name)));

			ScriptServer::add_global_class(class_name, base_type, gdscript_name, source_file, is_abstract, is_tool);
		}

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

#endif // TOOLS_ENABLED

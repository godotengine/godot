/**************************************************************************/
/*  test_code_actions.h                                                   */
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

#include "../editor/gdscript_editor_language.h"
#include "../gdscript.h"
#include "gdscript_test_runner.h"
// #include "test_completion.h" // For setup_global_classes(). (Probably best to move it to a shared file later.)

#include "core/config/project_settings.h"
#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/object/script_language.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"
#include "editor/gui/code_editor.h"
#include "editor/settings/editor_settings.h"
#include "scene/resources/packed_scene.h"
#include "scene/theme/theme_db.h"
#include "tests/test_macros.h"

#include "modules/modules_enabled.gen.h" // IWYU pragma: keep. For mono.

// Largely based off of HolonProduction's test_completion.h implementation.

namespace GDScriptTests {

static void setup_global_classes_for_code_actions(const String &p_dir) {
	Error err = OK;
	Ref<DirAccess> dir = DirAccess::open(p_dir, &err);

	if (err != OK) {
		FAIL("Invalid test directory.");
		return;
	}

	String path = dir->get_current_dir();

	dir->list_dir_begin();
	String next = dir->get_next();

	while (!next.is_empty()) {
		if (dir->current_is_dir() && next != "." && next != "..") {
			setup_global_classes_for_code_actions(path.path_join(next));
		} else if (next.ends_with(".gd")) {
			String base_type;
			bool is_abstract;
			bool is_tool;
			String source_file = path.path_join(next);
			String class_name = GDScriptLanguage::get_singleton()->get_global_class_name(source_file, &base_type, nullptr, &is_abstract, &is_tool);
			if (class_name.is_empty()) {
				next = dir->get_next();
				continue;
			}
			ERR_FAIL_COND_MSG(ScriptServer::is_global_class(class_name),
					"Class name \"" + class_name + "\" from \"" + source_file + "\" is already used in \"" + ScriptServer::get_global_class_path(class_name) + "\".");

			ScriptServer::add_global_class(class_name, base_type, GDScriptLanguage::get_singleton()->get_name(), source_file, is_abstract, is_tool);
		}
		next = dir->get_next();
	}
}

static const String modify_code_with_doc_edit(const String &p_code, const EditorLanguage::DocumentEditOperation &p_doc_edit) {
	CodeEdit *ce = memnew(CodeEdit);
	ce->set_text(p_code);
	ce->begin_complex_operation();
	for (const ScriptLanguage::TextEdit &text_edit : p_doc_edit.edits) {
		int end_line = text_edit.end_line;
		int end_col = text_edit.end_column;
		if (text_edit.end_line >= ce->get_line_count()) {
			// Case where the edit ends at the end of the file (such as deleting a variable declaration
			// that is on the last line).
			end_line = ce->get_line_count() - 1;
			end_col = ce->get_line(end_line).length();
		}
		ce->remove_text(text_edit.start_line, text_edit.start_column, end_line, end_col);
		ce->insert_text(text_edit.new_text, text_edit.start_line, text_edit.start_column);
	}
	ce->end_complex_operation();
	String text = ce->get_text();
	memdelete(ce);
	return text;
}

static void test_directory_for_code_actions(const String &p_dir) {
	Error err = OK;
	Ref<DirAccess> dir = DirAccess::open(p_dir, &err);

	if (err != OK) {
		FAIL("Invalid test directory.");
		return;
	}

	String path = dir->get_current_dir();

	dir->list_dir_begin();
	String next = dir->get_next();

	while (!next.is_empty()) {
		if (dir->current_is_dir()) {
			if (next == "." || next == "..") {
				next = dir->get_next();
				continue;
			}
			test_directory_for_code_actions(path.path_join(next));
		} else if (next.ends_with(".gd") && !next.ends_with(".notest.gd") && !next.ends_with(".out.gd")) {
			Ref<FileAccess> acc = FileAccess::open(path.path_join(next), FileAccess::READ, &err);

			if (err != OK) {
				next = dir->get_next();
				continue;
			}

			String code = acc->get_as_utf8_string();

			String res_path = ProjectSettings::get_singleton()->localize_path(path.path_join(next));

			ConfigFile conf;
			if (conf.load(path.path_join(next.get_basename() + ".cfg")) != OK) {
				FAIL(vformat("Test at \"%s\": No config file found.", res_path));
				next = dir->get_next();
				continue;
			}

			Array include_only_warnings = conf.get_value("warnings", "include_only", Array());
			for (int i = 0; i < GDScriptWarning::Code::WARNING_MAX; i++) {
				GDScriptWarning::Code warning_code = (GDScriptWarning::Code)i;
				String warning_name = GDScriptWarning::get_name_from_code(warning_code);
				String setting_path = GDScriptWarning::get_setting_path_from_code((GDScriptWarning::Code)i);
				ProjectSettings::get_singleton()->set_setting(setting_path, include_only_warnings.has(warning_name) ? GDScriptWarning::WARN : GDScriptWarning::IGNORE);
			}
			GDScriptParser::update_project_settings();

			Vector<EditorLanguage::CodeActionGroupWithDiagnostics> code_actions;
			ERR_PRINT_OFF;
			GDScriptEditorLanguage::get_singleton()->get_code_actions(code, res_path, &code_actions);
			ERR_PRINT_ON;

			// Display available code action options.
			// print_line(vformat("File: %s", res_path));
			// for (const EditorLanguage::CodeActionGroupWithDiagnostics &group : code_actions) {
			// 	print_line(vformat("Title: %s", group.title));
			// 	for (const EditorLanguage::CodeActionAndDiagnostics &ca : group.actions) {
			// 		print_line(ca.code_action.to_dict());
			// 	}
			// }

			if (!conf.has_section_key("apply", "group_idx")) {
				FAIL(vformat("Test at \"%s\": Config file does not have group_idx key for applying a code action.", res_path));
				next = dir->get_next();
				continue;
			}
			if (!conf.has_section_key("apply", "action_idx")) {
				FAIL(vformat("Test at \"%s\": Config file does not have action_idx key for applying a code action.", res_path));
				next = dir->get_next();
				continue;
			}

			int group_to_use = conf.get_value("apply", "group_idx");
			int action_to_use = conf.get_value("apply", "action_idx");

			if (group_to_use < 0 || group_to_use >= code_actions.size()) {
				FAIL(vformat("Test at \"%s\": group_idx is out of range (requesting group at index %s but there are %s groups)", res_path, group_to_use, code_actions.size()));
				next = dir->get_next();
				continue;
			}

			if (action_to_use < 0 || action_to_use >= code_actions[group_to_use].actions.size()) {
				FAIL(vformat("Test at \"%s\": action_idx is out of range (requesting code action at index %s but there are %s code actions)", res_path, action_to_use, code_actions[group_to_use].actions.size()));
				next = dir->get_next();
				continue;
			}

			Array doc_edits = code_actions[group_to_use].actions[action_to_use].code_action.to_dict().get("document_edits", Array());
			if (doc_edits.size() != 1) {
				FAIL(vformat("Test at \"%s\": %s document edits are requested, but unit testing currently supports at most 1.", res_path, doc_edits.size()));
				next = dir->get_next();
				continue;
			}

			// Apply edits.
			// NOTE: Currently only one document edit is supported, and it has to
			// be from the same file as the code action is being performed on.
			// Once multi-document code actions start being implemented, the test runner will need
			// to be updated to support testing that.
			EditorLanguage::DocumentEditOperation doc_edit = EditorLanguage::DocumentEditOperation::from_dict(doc_edits[0]);

			if (doc_edit.file_path != res_path) {
				FAIL(vformat("Test at \"%s\": Document edit requested to \"%s\"; unit testing currently requires that the test file be the only file edited.", res_path, doc_edit.file_path));
				next = dir->get_next();
				continue;
			}

			Ref<FileAccess> doc_to_edit = FileAccess::open(doc_edit.file_path, FileAccess::READ, &err);
			if (err != OK) {
				FAIL(vformat("Test at \"%s\": Attempting to open document errored with code %s", res_path, err));
				next = dir->get_next();
				continue;
			}

			String code_before_code_action = doc_to_edit->get_as_utf8_string();

			String produced_code = modify_code_with_doc_edit(code_before_code_action, doc_edit);

			// Compare the code with the code action applied to what we expect.
			String expected_file_path = path.path_join(next.get_basename() + ".out.gd");
			Ref<FileAccess> expected_file = FileAccess::open(expected_file_path, FileAccess::READ, &err);
			if (err != OK) {
				FAIL(vformat("Test at \"%s\": No expected output file \"%s\" found.", res_path, expected_file_path));
				next = dir->get_next();
				continue;
			}

			String expected_code = expected_file->get_as_utf8_string();
			if (produced_code != expected_code) {
				FAIL(vformat("Test at \"%s\": Produced output differs from expected output.\nExpected:\n%s\n----------\nProduced:\n%s\n----------", res_path, expected_code, produced_code));
				next = dir->get_next();
				continue;
			}
		}
		next = dir->get_next();
	}
}

TEST_SUITE("[Modules][GDScript][Code Actions]") {
	TEST_CASE("[Editor] Check code actions") {
		EditorSettings::get_singleton()->set_setting("text_editor/completion/use_single_quotes", false);

		init_language("modules/gdscript/tests/scripts");

		setup_global_classes_for_code_actions("modules/gdscript/tests/scripts/code_actions");
		test_directory_for_code_actions("modules/gdscript/tests/scripts/code_actions");

		finish_language();
	}
}
} // namespace GDScriptTests

#endif // TOOLS_ENABLED

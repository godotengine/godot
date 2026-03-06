/**************************************************************************/
/*  shader_text_editor.cpp                                                */
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

#include "shader_text_editor.h"

#include "core/config/project_settings.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/version_generated.gen.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/script/syntax_highlighters.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_theme_manager.h"
#include "scene/gui/rich_text_label.h"
#include "scene/resources/visual_shader.h"
#include "servers/rendering/rendering_server.h"
#include "servers/rendering/shader_preprocessor.h"
#include "servers/rendering/shader_types.h"

static bool saved_warnings_enabled = false;
static bool saved_treat_warning_as_errors = false;
static HashMap<ShaderWarning::Code, bool> saved_warnings;
static uint32_t saved_warning_flags = 0U;

void ShaderTextEditor::_bind_methods() {
	ClassDB::bind_method("_show_warnings_panel", &ShaderTextEditor::_show_warnings_panel);
	ClassDB::bind_method("_warning_clicked", &ShaderTextEditor::_warning_clicked);

	ADD_SIGNAL(MethodInfo("validation_changed"));
}

void ShaderTextEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			if (is_visible_in_tree()) {
				_load_theme_settings();
				if (warnings.size() > 0 && last_compile_result == OK) {
					warnings_panel->clear();
					_update_warning_panel();
				}
			}
		} break;
	}
}

void ShaderTextEditor::_shader_changed() {
	// This function is used for dependencies (include changing changes main shader and forces it to revalidate)
	if (block_shader_changed) {
		return;
	}
	dependencies_version++;
	_validate_script();
}

void ShaderTextEditor::set_edited_code(const String &p_code) {
	_load_theme_settings();

	code_editor->get_text_editor()->set_text(p_code);
	code_editor->get_text_editor()->clear_undo_history();
	callable_mp((TextEdit *)code_editor->get_text_editor(), &TextEdit::set_h_scroll).call_deferred(0);
	callable_mp((TextEdit *)code_editor->get_text_editor(), &TextEdit::set_v_scroll).call_deferred(0);
	code_editor->get_text_editor()->tag_saved_version();

	_validate_script();
	code_editor->update_line_and_column();
}

void ShaderTextEditor::reload_text() {
	ERR_FAIL_COND(edited_res.is_null());

	Ref<ShaderInclude> shader_inc = edited_res;
	Ref<Shader> shader = edited_res;

	String code;
	if (shader_inc.is_valid()) {
		code = shader_inc->get_code();
	} else {
		code = shader->get_code();
	}

	CodeEdit *te = code_editor->get_text_editor();
	int column = te->get_caret_column();
	int row = te->get_caret_line();
	int h = te->get_h_scroll();
	int v = te->get_v_scroll();

	te->set_text(code);
	te->set_caret_line(row);
	te->set_caret_column(column);
	te->set_h_scroll(h);
	te->set_v_scroll(v);

	te->tag_saved_version();

	code_editor->update_line_and_column();
}

void ShaderTextEditor::_load_theme_settings() {
	CodeEdit *te = code_editor->get_text_editor();
	Color updated_marked_line_color = EDITOR_GET("text_editor/theme/highlighting/mark_color");
	if (updated_marked_line_color != marked_line_color) {
		for (int i = 0; i < te->get_line_count(); i++) {
			if (te->get_line_background_color(i) == marked_line_color) {
				te->set_line_background_color(i, updated_marked_line_color);
			}
		}
		marked_line_color = updated_marked_line_color;
	}

	te->clear_comment_delimiters();
	te->add_comment_delimiter("/*", "*/", false);
	te->add_comment_delimiter("//", "", true);

	if (!te->has_auto_brace_completion_open_key("/*")) {
		te->add_auto_brace_completion_pair("/*", "*/");
	}
	code_editor->get_text_editor()->get_syntax_highlighter()->update_cache();
}

static ShaderLanguage::DataType _get_global_shader_uniform_type(const StringName &p_variable) {
	RSE::GlobalShaderParameterType gvt = RS::get_singleton()->global_shader_parameter_get_type(p_variable);
	return (ShaderLanguage::DataType)RS::global_shader_uniform_type_get_shader_datatype(gvt);
}

static String complete_from_path;

static void _complete_include_paths_search(EditorFileSystemDirectory *p_efsd, List<ScriptLanguage::CodeCompletionOption> *r_options) {
	if (!p_efsd) {
		return;
	}
	for (int i = 0; i < p_efsd->get_file_count(); i++) {
		if (p_efsd->get_file_type(i) == SNAME("ShaderInclude")) {
			String path = p_efsd->get_file_path(i);
			if (path.begins_with(complete_from_path)) {
				path = path.replace_first(complete_from_path, "");
			}
			r_options->push_back(ScriptLanguage::CodeCompletionOption(path, ScriptLanguage::CODE_COMPLETION_KIND_FILE_PATH));
		}
	}
	for (int j = 0; j < p_efsd->get_subdir_count(); j++) {
		_complete_include_paths_search(p_efsd->get_subdir(j), r_options);
	}
}

static void _complete_include_paths(List<ScriptLanguage::CodeCompletionOption> *r_options) {
	_complete_include_paths_search(EditorFileSystem::get_singleton()->get_filesystem(), r_options);
}

void ShaderTextEditor::_code_complete_script(const String &p_code, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_force) {
	List<ScriptLanguage::CodeCompletionOption> pp_options;
	List<ScriptLanguage::CodeCompletionOption> pp_defines;
	ShaderPreprocessor preprocessor;
	String code;
	String resource_path = edited_res->get_path();
	complete_from_path = resource_path.get_base_dir();
	if (!complete_from_path.ends_with("/")) {
		complete_from_path += "/";
	}
	preprocessor.preprocess(p_code, resource_path, code, nullptr, nullptr, nullptr, nullptr, &pp_options, &pp_defines, _complete_include_paths);
	complete_from_path = String();
	if (pp_options.size()) {
		for (const ScriptLanguage::CodeCompletionOption &E : pp_options) {
			r_options->push_back(E);
		}
		return;
	}

	ShaderLanguage sl;
	String calltip;
	ShaderLanguage::ShaderCompileInfo comp_info;
	comp_info.global_shader_uniform_type_func = _get_global_shader_uniform_type;

	Ref<Shader> shader = edited_res;
	if (shader.is_valid()) {
		_check_shader_mode();
		comp_info.functions = ShaderTypes::get_singleton()->get_functions(RSE::ShaderMode(shader->get_mode()));
		comp_info.render_modes = ShaderTypes::get_singleton()->get_modes(RSE::ShaderMode(shader->get_mode()));
		comp_info.stencil_modes = ShaderTypes::get_singleton()->get_stencil_modes(RSE::ShaderMode(shader->get_mode()));
		comp_info.shader_types = ShaderTypes::get_singleton()->get_types();
	} else {
		comp_info.is_include = true;
	}

	sl.complete(code, comp_info, r_options, calltip);
	if (sl.get_completion_type() == ShaderLanguage::COMPLETION_IDENTIFIER) {
		for (const ScriptLanguage::CodeCompletionOption &E : pp_defines) {
			r_options->push_back(E);
		}
	}
	code_editor->get_text_editor()->set_code_hint(calltip);
}

void ShaderTextEditor::_update_warning_panel() {
	int warning_count = 0;

	warnings_panel->push_table(2);
	for (const ShaderWarning &w : warnings) {
		if (warning_count == 0) {
			if (saved_treat_warning_as_errors) {
				const String message = (w.get_message() + " " + TTR("Warnings should be fixed to prevent errors.")).replace("[", "[lb]");
				const String error_text = vformat(TTR("Error at line %d:"), w.get_line()) + " " + message;

				code_editor->set_error(error_text);
				code_editor->set_error_pos(w.get_line() - 1, 0);

				code_editor->get_text_editor()->set_line_background_color(w.get_line() - 1, marked_line_color);
			}
		}

		warning_count++;
		int line = w.get_line();

		// First cell.
		warnings_panel->push_cell();
		warnings_panel->push_color(warnings_panel->get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));
		if (line != -1) {
			warnings_panel->push_meta(line - 1);
			warnings_panel->add_text(vformat(TTR("Line %d (%s):"), line, w.get_name()));
			warnings_panel->pop(); // Meta goto.
		} else {
			warnings_panel->add_text(w.get_name() + ":");
		}
		warnings_panel->pop(); // Color.
		warnings_panel->pop(); // Cell.

		// Second cell.
		warnings_panel->push_cell();
		warnings_panel->add_text(w.get_message());
		warnings_panel->pop(); // Cell.
	}
	warnings_panel->pop(); // Table.

	code_editor->set_warning_count(warning_count);
}

bool ShaderTextEditor::_edit_option(int p_option) {
	CodeEdit *tx = code_editor->get_text_editor();
	tx->apply_ime();

	switch (p_option) {
		case EDIT_TOGGLE_COMMENT: {
			if (edited_res.is_null()) {
				return true;
			}
			code_editor->toggle_inline_comment("//");
		} break;
		case EDIT_COMPLETE: {
			tx->request_code_completion();
		} break;
		case HELP_DOCS: {
			OS::get_singleton()->shell_open(vformat("%s/tutorials/shaders/shader_reference/index.html", GODOT_VERSION_DOCS_URL));
		} break;
		default:
			if (CodeEditorBase::_edit_option(p_option)) {
				return true;
			}
	}
	if (p_option != SEARCH_FIND && p_option != SEARCH_REPLACE && p_option != SEARCH_GOTO_LINE) {
		callable_mp((Control *)tx, &Control::grab_focus).call_deferred(false);
	}
	return true;
}

void ShaderTextEditor::_check_shader_mode() {
	String type = ShaderLanguage::get_shader_type(code_editor->get_text_editor()->get_text());

	Shader::Mode mode;
	Ref<Shader> shader = edited_res;
	if (shader.is_null()) {
		return;
	}

	if (type == "canvas_item") {
		mode = Shader::MODE_CANVAS_ITEM;
	} else if (type == "particles") {
		mode = Shader::MODE_PARTICLES;
	} else if (type == "sky") {
		mode = Shader::MODE_SKY;
	} else if (type == "fog") {
		mode = Shader::MODE_FOG;
	} else {
		mode = Shader::MODE_SPATIAL;
	}

	if (shader->get_mode() != mode) {
		block_shader_changed = true;
		shader->set_code(code_editor->get_text_editor()->get_text());
		block_shader_changed = false;
		_load_theme_settings();
	}
}

void ShaderTextEditor::_validate_script() {
	TextEditorBase::_validate_script();

	CodeEdit *te = code_editor->get_text_editor();
	String code = te->get_text();

	ShaderPreprocessor preprocessor;
	String code_pp;
	String error_pp;
	List<ShaderPreprocessor::FilePosition> err_positions;
	List<ShaderPreprocessor::Region> regions;
	String filename = edited_res->get_path();
	last_compile_result = preprocessor.preprocess(code, filename, code_pp, &error_pp, &err_positions, &regions);

	for (int i = 0; i < code_editor->get_text_editor()->get_line_count(); i++) {
		code_editor->get_text_editor()->set_line_background_color(i, Color(0, 0, 0, 0));
	}

	Ref<GDShaderSyntaxHighlighter> sh = code_editor->get_text_editor()->get_syntax_highlighter();
	sh.instantiate();
	sh->clear_disabled_branch_regions();
	for (const ShaderPreprocessor::Region &region : regions) {
		if (!region.enabled && filename != region.file) {
			sh->add_disabled_branch_region(Point2i(region.from_line, region.to_line));
		}
	}

	code_editor->set_error("");
	code_editor->set_error_count(0);

	if (last_compile_result != OK) {
		// Preprocessor error.
		ERR_FAIL_COND(err_positions.is_empty());

		String err_text;
		const int err_line = err_positions.front()->get().line;
		if (err_positions.size() == 1) {
			// Error in the main file.
			const String message = error_pp.replace("[", "[lb]");

			err_text = vformat(TTR("Error at line %d:"), err_line) + " " + message;
		} else {
			// Error in an included file.
			const String inc_file = err_positions.back()->get().file.get_file();
			const int inc_line = err_positions.back()->get().line;
			const String message = error_pp.replace("[", "[lb]");

			err_text = vformat(TTR("Error at line %d in include %s:%d:"), err_line, inc_file, inc_line) + " " + message;
			code_editor->set_error_count(err_positions.size() - 1);
		}

		code_editor->set_error(err_text);
		code_editor->set_error_pos(err_line - 1, 0);

		for (int i = 0; i < code_editor->get_text_editor()->get_line_count(); i++) {
			code_editor->get_text_editor()->set_line_background_color(i, Color(0, 0, 0, 0));
		}
		code_editor->get_text_editor()->set_line_background_color(err_line - 1, marked_line_color);

		code_editor->set_warning_count(0);
	} else {
		ShaderLanguage sl;

		sl.enable_warning_checking(saved_warnings_enabled);
		uint32_t flags = saved_warning_flags;
		Ref<Shader> shader = edited_res;
		if (shader.is_null()) {
			if (flags & ShaderWarning::UNUSED_CONSTANT) {
				flags &= ~(ShaderWarning::UNUSED_CONSTANT);
			}
			if (flags & ShaderWarning::UNUSED_FUNCTION) {
				flags &= ~(ShaderWarning::UNUSED_FUNCTION);
			}
			if (flags & ShaderWarning::UNUSED_STRUCT) {
				flags &= ~(ShaderWarning::UNUSED_STRUCT);
			}
			if (flags & ShaderWarning::UNUSED_UNIFORM) {
				flags &= ~(ShaderWarning::UNUSED_UNIFORM);
			}
			if (flags & ShaderWarning::UNUSED_VARYING) {
				flags &= ~(ShaderWarning::UNUSED_VARYING);
			}
		}
		sl.set_warning_flags(flags);

		ShaderLanguage::ShaderCompileInfo comp_info;
		comp_info.global_shader_uniform_type_func = _get_global_shader_uniform_type;

		if (shader.is_null()) {
			comp_info.is_include = true;
		} else {
			Shader::Mode mode = shader->get_mode();
			comp_info.functions = ShaderTypes::get_singleton()->get_functions(RSE::ShaderMode(mode));
			comp_info.render_modes = ShaderTypes::get_singleton()->get_modes(RSE::ShaderMode(mode));
			comp_info.stencil_modes = ShaderTypes::get_singleton()->get_stencil_modes(RSE::ShaderMode(mode));
			comp_info.shader_types = ShaderTypes::get_singleton()->get_types();
		}

		code = code_pp;
		//compiler error
		last_compile_result = sl.compile(code, comp_info);

		if (last_compile_result != OK) {
			Vector<ShaderLanguage::FilePosition> include_positions = sl.get_include_positions();

			String err_text;
			int err_line;
			if (include_positions.size() > 1) {
				// Error in an included file.
				err_line = include_positions[0].line;

				const String inc_file = include_positions[include_positions.size() - 1].file;
				const int inc_line = include_positions[include_positions.size() - 1].line;
				const String message = sl.get_error_text().replace("[", "[lb]");

				err_text = vformat(TTR("Error at line %d in include %s:%d:"), err_line, inc_file, inc_line) + " " + message;
				code_editor->set_error_count(include_positions.size() - 1);
			} else {
				// Error in the main file.
				err_line = sl.get_error_line();

				const String message = sl.get_error_text().replace("[", "[lb]");

				err_text = vformat(TTR("Error at line %d:"), err_line) + " " + message;
				code_editor->set_error_count(0);
			}

			code_editor->set_error(err_text);
			code_editor->set_error_pos(err_line - 1, 0);

			code_editor->get_text_editor()->set_line_background_color(err_line - 1, marked_line_color);
		} else {
			code_editor->set_error("");
		}

		if (warnings.size() > 0 || last_compile_result != OK) {
			warnings_panel->clear();
		}
		warnings.clear();
		for (List<ShaderWarning>::Element *E = sl.get_warnings_ptr(); E; E = E->next()) {
			warnings.push_back(E->get());
		}
		if (warnings.size() > 0 && last_compile_result == OK) {
			warnings.sort_custom<WarningsComparator>();
			_update_warning_panel();
		} else {
			code_editor->set_warning_count(0);
		}
	}

	_script_validated(last_compile_result == OK); // Notify that validation finished, to update the list of scripts
}

void ShaderTextEditor::_editor_settings_changed() {
	if (!EditorThemeManager::is_generated_theme_outdated() &&
			!EditorSettings::get_singleton()->check_changed_settings_in_group("interface/editor") &&
			!EditorSettings::get_singleton()->check_changed_settings_in_group("text_editor")) {
		return;
	}

	code_editor->update_editor_settings();
}

void ShaderTextEditor::_show_warnings_panel(bool p_show) {
	warnings_panel->set_visible(p_show);
}

void ShaderTextEditor::_project_settings_changed() {
	_update_warnings(true);
}

void ShaderTextEditor::_update_warnings(bool p_validate) {
	bool changed = false;

	bool warnings_enabled = GLOBAL_GET("debug/shader_language/warnings/enable").booleanize();
	if (warnings_enabled != saved_warnings_enabled) {
		saved_warnings_enabled = warnings_enabled;
		changed = true;
	}

	bool treat_warning_as_errors = GLOBAL_GET("debug/shader_language/warnings/treat_warnings_as_errors").booleanize();
	if (treat_warning_as_errors != saved_treat_warning_as_errors) {
		saved_treat_warning_as_errors = treat_warning_as_errors;
		changed = true;
	}

	bool update_flags = false;

	for (int i = 0; i < ShaderWarning::WARNING_MAX; i++) {
		ShaderWarning::Code code = (ShaderWarning::Code)i;
		bool value = GLOBAL_GET("debug/shader_language/warnings/" + ShaderWarning::get_name_from_code(code).to_lower());

		if (saved_warnings[code] != value) {
			saved_warnings[code] = value;
			update_flags = true;
			changed = true;
		}
	}

	if (update_flags) {
		saved_warning_flags = (uint32_t)ShaderWarning::get_flags_from_codemap(saved_warnings);
	}

	Ref<Shader> shader = edited_res;
	if (p_validate && changed && code_editor && shader.is_valid()) {
		code_editor->validate_script();
	}
}

void ShaderTextEditor::set_edited_resource(const Ref<Resource> &p_res) {
	Ref<Shader> shader = p_res;
	Ref<ShaderInclude> shader_inc = p_res;
	if (shader.is_valid()) {
		set_edited_resource(p_res, shader->get_code());
	} else if (shader_inc.is_valid()) {
		set_edited_resource(p_res, shader_inc->get_code());
	}
}

void ShaderTextEditor::set_edited_resource(const Ref<Resource> &p_res, const String &p_code) {
	if (p_res.is_null() || edited_res == p_res) {
		return;
	}

	Ref<Shader> shader = p_res;
	Ref<ShaderInclude> shader_inc = p_res;
	if (shader.is_null() && shader_inc.is_null()) {
		return;
	}
	p_res->disconnect_changed(callable_mp(this, &ShaderTextEditor::_shader_changed));

	edited_res = p_res;
	set_edited_code(p_code);

	p_res->connect_changed(callable_mp(this, &ShaderTextEditor::_shader_changed));
}

bool ShaderTextEditor::is_unsaved() {
	return code_editor->get_text_editor()->get_saved_version() != code_editor->get_text_editor()->get_version();
}

void ShaderTextEditor::apply_code() {
	String editor_code = code_editor->get_text_editor()->get_text();
	Ref<Shader> shader = edited_res;
	if (shader.is_valid()) {
		String shader_code = shader->get_code();
		if (shader_code != editor_code || dependencies_version != get_dependencies_version()) {
			block_shader_changed = true;
			shader->set_code(editor_code);
			block_shader_changed = false;
			shader->set_edited(true);
		}
	}
	Ref<ShaderInclude> shader_inc = edited_res;
	if (shader_inc.is_valid()) {
		String shader_inc_code = shader_inc->get_code();
		if (shader_inc_code != editor_code || dependencies_version != get_dependencies_version()) {
			block_shader_changed = true;
			shader_inc->set_code(editor_code);
			block_shader_changed = false;
			shader_inc->set_edited(true);
		}
	}

	dependencies_version = get_dependencies_version();
	code_editor->get_text_editor()->get_syntax_highlighter()->update_cache();
}

ScriptEditorBase *ShaderTextEditor::create_editor(const Ref<Resource> &p_resource) {
	if (Object::cast_to<VisualShader>(*p_resource)) {
		return nullptr;
	}
	if (Object::cast_to<Shader>(*p_resource) || Object::cast_to<ShaderInclude>(*p_resource)) {
		return memnew(ShaderTextEditor);
	}
	return nullptr;
}

void ShaderTextEditor::register_editor() {
	ScriptEditor::register_create_script_editor_function(create_editor);
}

ShaderTextEditor::ShaderTextEditor() {
	_update_warnings(false);

	code_editor->connect("show_warnings_panel", callable_mp(this, &ShaderTextEditor::_show_warnings_panel));
	code_editor->connect(CoreStringName(script_changed), callable_mp(this, &ShaderTextEditor::apply_code));

	EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &ShaderTextEditor::_editor_settings_changed));
	ProjectSettings::get_singleton()->connect("settings_changed", callable_mp(this, &ShaderTextEditor::_project_settings_changed));

	code_editor->get_text_editor()->set_draw_breakpoints_gutter(false);
	code_editor->get_text_editor()->set_draw_executing_lines_gutter(false);

	code_editor->update_editor_settings();

	_editor_settings_changed();
}

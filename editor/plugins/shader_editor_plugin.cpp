/*************************************************************************/
/*  shader_editor_plugin.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "shader_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/version_generated.gen.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/filesystem_dock.h"
#include "editor/plugins/visual_shader_editor_plugin.h"
#include "editor/project_settings_editor.h"
#include "editor/shader_create_dialog.h"
#include "scene/gui/split_container.h"
#include "servers/display_server.h"
#include "servers/rendering/shader_preprocessor.h"
#include "servers/rendering/shader_types.h"

/*** SHADER SCRIPT EDITOR ****/

static bool saved_warnings_enabled = false;
static bool saved_treat_warning_as_errors = false;
static HashMap<ShaderWarning::Code, bool> saved_warnings;
static uint32_t saved_warning_flags = 0U;

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

Ref<Shader> ShaderTextEditor::get_edited_shader() const {
	return shader;
}

Ref<ShaderInclude> ShaderTextEditor::get_edited_shader_include() const {
	return shader_inc;
}

void ShaderTextEditor::set_edited_shader(const Ref<Shader> &p_shader) {
	set_edited_shader(p_shader, p_shader->get_code());
}

void ShaderTextEditor::set_edited_shader(const Ref<Shader> &p_shader, const String &p_code) {
	if (shader == p_shader) {
		return;
	}
	if (shader.is_valid()) {
		shader->disconnect(SNAME("changed"), callable_mp(this, &ShaderTextEditor::_shader_changed));
	}
	shader = p_shader;
	shader_inc = Ref<ShaderInclude>();

	set_edited_code(p_code);

	if (shader.is_valid()) {
		shader->connect(SNAME("changed"), callable_mp(this, &ShaderTextEditor::_shader_changed));
	}
}

void ShaderTextEditor::set_edited_shader_include(const Ref<ShaderInclude> &p_shader_inc) {
	set_edited_shader_include(p_shader_inc, p_shader_inc->get_code());
}

void ShaderTextEditor::_shader_changed() {
	// This function is used for dependencies (include changing changes main shader and forces it to revalidate)
	if (block_shader_changed) {
		return;
	}
	dependencies_version++;
	_validate_script();
}

void ShaderTextEditor::set_edited_shader_include(const Ref<ShaderInclude> &p_shader_inc, const String &p_code) {
	if (shader_inc == p_shader_inc) {
		return;
	}
	if (shader_inc.is_valid()) {
		shader_inc->disconnect(SNAME("changed"), callable_mp(this, &ShaderTextEditor::_shader_changed));
	}
	shader_inc = p_shader_inc;
	shader = Ref<Shader>();

	set_edited_code(p_code);

	if (shader_inc.is_valid()) {
		shader_inc->connect(SNAME("changed"), callable_mp(this, &ShaderTextEditor::_shader_changed));
	}
}

void ShaderTextEditor::set_edited_code(const String &p_code) {
	_load_theme_settings();

	get_text_editor()->set_text(p_code);
	get_text_editor()->clear_undo_history();
	get_text_editor()->call_deferred(SNAME("set_h_scroll"), 0);
	get_text_editor()->call_deferred(SNAME("set_v_scroll"), 0);
	get_text_editor()->tag_saved_version();

	_validate_script();
	_line_col_changed();
}

void ShaderTextEditor::reload_text() {
	ERR_FAIL_COND(shader.is_null());

	CodeEdit *te = get_text_editor();
	int column = te->get_caret_column();
	int row = te->get_caret_line();
	int h = te->get_h_scroll();
	int v = te->get_v_scroll();

	te->set_text(shader->get_code());
	te->set_caret_line(row);
	te->set_caret_column(column);
	te->set_h_scroll(h);
	te->set_v_scroll(v);

	te->tag_saved_version();

	update_line_and_column();
}

void ShaderTextEditor::set_warnings_panel(RichTextLabel *p_warnings_panel) {
	warnings_panel = p_warnings_panel;
}

void ShaderTextEditor::_load_theme_settings() {
	CodeEdit *text_editor = get_text_editor();
	Color updated_marked_line_color = EDITOR_GET("text_editor/theme/highlighting/mark_color");
	if (updated_marked_line_color != marked_line_color) {
		for (int i = 0; i < text_editor->get_line_count(); i++) {
			if (text_editor->get_line_background_color(i) == marked_line_color) {
				text_editor->set_line_background_color(i, updated_marked_line_color);
			}
		}
		marked_line_color = updated_marked_line_color;
	}

	syntax_highlighter->set_number_color(EDITOR_GET("text_editor/theme/highlighting/number_color"));
	syntax_highlighter->set_symbol_color(EDITOR_GET("text_editor/theme/highlighting/symbol_color"));
	syntax_highlighter->set_function_color(EDITOR_GET("text_editor/theme/highlighting/function_color"));
	syntax_highlighter->set_member_variable_color(EDITOR_GET("text_editor/theme/highlighting/member_variable_color"));

	syntax_highlighter->clear_keyword_colors();

	const Color keyword_color = EDITOR_GET("text_editor/theme/highlighting/keyword_color");
	const Color control_flow_keyword_color = EDITOR_GET("text_editor/theme/highlighting/control_flow_keyword_color");

	List<String> keywords;
	ShaderLanguage::get_keyword_list(&keywords);

	for (const String &E : keywords) {
		if (ShaderLanguage::is_control_flow_keyword(E)) {
			syntax_highlighter->add_keyword_color(E, control_flow_keyword_color);
		} else {
			syntax_highlighter->add_keyword_color(E, keyword_color);
		}
	}

	List<String> pp_keywords;
	ShaderPreprocessor::get_keyword_list(&pp_keywords, false);

	for (const String &E : pp_keywords) {
		syntax_highlighter->add_keyword_color(E, keyword_color);
	}

	// Colorize built-ins like `COLOR` differently to make them easier
	// to distinguish from keywords at a quick glance.

	List<String> built_ins;

	if (shader_inc.is_valid()) {
		for (int i = 0; i < RenderingServer::SHADER_MAX; i++) {
			for (const KeyValue<StringName, ShaderLanguage::FunctionInfo> &E : ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(i))) {
				for (const KeyValue<StringName, ShaderLanguage::BuiltInInfo> &F : E.value.built_ins) {
					built_ins.push_back(F.key);
				}
			}

			const Vector<ShaderLanguage::ModeInfo> &modes = ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(i));

			for (int j = 0; j < modes.size(); j++) {
				const ShaderLanguage::ModeInfo &info = modes[j];

				if (!info.options.is_empty()) {
					for (int k = 0; k < info.options.size(); k++) {
						built_ins.push_back(String(info.name) + "_" + String(info.options[k]));
					}
				} else {
					built_ins.push_back(String(info.name));
				}
			}
		}
	} else if (shader.is_valid()) {
		for (const KeyValue<StringName, ShaderLanguage::FunctionInfo> &E : ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(shader->get_mode()))) {
			for (const KeyValue<StringName, ShaderLanguage::BuiltInInfo> &F : E.value.built_ins) {
				built_ins.push_back(F.key);
			}
		}

		const Vector<ShaderLanguage::ModeInfo> &modes = ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(shader->get_mode()));

		for (int i = 0; i < modes.size(); i++) {
			const ShaderLanguage::ModeInfo &info = modes[i];

			if (!info.options.is_empty()) {
				for (int j = 0; j < info.options.size(); j++) {
					built_ins.push_back(String(info.name) + "_" + String(info.options[j]));
				}
			} else {
				built_ins.push_back(String(info.name));
			}
		}
	}

	const Color user_type_color = EDITOR_GET("text_editor/theme/highlighting/user_type_color");

	for (const String &E : built_ins) {
		syntax_highlighter->add_keyword_color(E, user_type_color);
	}

	// Colorize comments.
	const Color comment_color = EDITOR_GET("text_editor/theme/highlighting/comment_color");
	syntax_highlighter->clear_color_regions();
	syntax_highlighter->add_color_region("/*", "*/", comment_color, false);
	syntax_highlighter->add_color_region("//", "", comment_color, true);

	text_editor->clear_comment_delimiters();
	text_editor->add_comment_delimiter("/*", "*/", false);
	text_editor->add_comment_delimiter("//", "", true);

	if (!text_editor->has_auto_brace_completion_open_key("/*")) {
		text_editor->add_auto_brace_completion_pair("/*", "*/");
	}

	// Colorize preprocessor include strings.
	const Color string_color = EDITOR_GET("text_editor/theme/highlighting/string_color");
	syntax_highlighter->add_color_region("\"", "\"", string_color, false);

	if (warnings_panel) {
		// Warnings panel.
		warnings_panel->add_theme_font_override("normal_font", EditorNode::get_singleton()->get_gui_base()->get_theme_font(SNAME("main"), SNAME("EditorFonts")));
		warnings_panel->add_theme_font_size_override("normal_font_size", EditorNode::get_singleton()->get_gui_base()->get_theme_font_size(SNAME("main_size"), SNAME("EditorFonts")));
	}
}

void ShaderTextEditor::_check_shader_mode() {
	String type = ShaderLanguage::get_shader_type(get_text_editor()->get_text());

	Shader::Mode mode;

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
		set_block_shader_changed(true);
		shader->set_code(get_text_editor()->get_text());
		set_block_shader_changed(false);
		_load_theme_settings();
	}
}

static ShaderLanguage::DataType _get_global_shader_uniform_type(const StringName &p_variable) {
	RS::GlobalShaderUniformType gvt = RS::get_singleton()->global_shader_uniform_get_type(p_variable);
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

void ShaderTextEditor::_code_complete_script(const String &p_code, List<ScriptLanguage::CodeCompletionOption> *r_options) {
	List<ScriptLanguage::CodeCompletionOption> pp_options;
	ShaderPreprocessor preprocessor;
	String code;
	complete_from_path = (shader.is_valid() ? shader->get_path() : shader_inc->get_path()).get_base_dir();
	if (!complete_from_path.ends_with("/")) {
		complete_from_path += "/";
	}
	preprocessor.preprocess(p_code, code, nullptr, nullptr, nullptr, &pp_options, _complete_include_paths);
	complete_from_path = String();
	if (pp_options.size()) {
		for (const ScriptLanguage::CodeCompletionOption &E : pp_options) {
			r_options->push_back(E);
		}
		return;
	}

	ShaderLanguage sl;
	String calltip;
	ShaderLanguage::ShaderCompileInfo info;
	info.global_shader_uniform_type_func = _get_global_shader_uniform_type;

	if (shader.is_null()) {
		info.is_include = true;

		sl.complete(code, info, r_options, calltip);
		get_text_editor()->set_code_hint(calltip);
		return;
	}
	_check_shader_mode();
	info.functions = ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(shader->get_mode()));
	info.render_modes = ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(shader->get_mode()));
	info.shader_types = ShaderTypes::get_singleton()->get_types();

	sl.complete(code, info, r_options, calltip);
	get_text_editor()->set_code_hint(calltip);
}

void ShaderTextEditor::_validate_script() {
	emit_signal(SNAME("script_changed")); // Ensure to notify that it changed, so it is applied

	String code;

	if (shader.is_valid()) {
		_check_shader_mode();
		code = shader->get_code();
	} else {
		code = shader_inc->get_code();
	}

	ShaderPreprocessor preprocessor;
	String code_pp;
	String error_pp;
	List<ShaderPreprocessor::FilePosition> err_positions;
	last_compile_result = preprocessor.preprocess(code, code_pp, &error_pp, &err_positions);

	for (int i = 0; i < get_text_editor()->get_line_count(); i++) {
		get_text_editor()->set_line_background_color(i, Color(0, 0, 0, 0));
	}
	set_error("");
	set_error_count(0);

	if (last_compile_result != OK) {
		//preprocessor error
		ERR_FAIL_COND(err_positions.size() == 0);

		String error_text = error_pp;
		int error_line = err_positions.front()->get().line;
		if (err_positions.size() == 1) {
			// Error in main file
			error_text = "error(" + itos(error_line) + "): " + error_text;
		} else {
			error_text = "error(" + itos(error_line) + ") in include " + err_positions.back()->get().file.get_file() + ":" + itos(err_positions.back()->get().line) + ": " + error_text;
			set_error_count(err_positions.size() - 1);
		}

		set_error(error_text);
		set_error_pos(error_line - 1, 0);
		for (int i = 0; i < get_text_editor()->get_line_count(); i++) {
			get_text_editor()->set_line_background_color(i, Color(0, 0, 0, 0));
		}
		get_text_editor()->set_line_background_color(error_line - 1, marked_line_color);

		set_warning_count(0);

	} else {
		ShaderLanguage sl;

		sl.enable_warning_checking(saved_warnings_enabled);
		uint32_t flags = saved_warning_flags;
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

		ShaderLanguage::ShaderCompileInfo info;
		info.global_shader_uniform_type_func = _get_global_shader_uniform_type;

		if (shader.is_null()) {
			info.is_include = true;
		} else {
			Shader::Mode mode = shader->get_mode();
			info.functions = ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(mode));
			info.render_modes = ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(mode));
			info.shader_types = ShaderTypes::get_singleton()->get_types();
		}

		code = code_pp;
		//compiler error
		last_compile_result = sl.compile(code, info);

		if (last_compile_result != OK) {
			String error_text;
			int error_line;
			Vector<ShaderLanguage::FilePosition> include_positions = sl.get_include_positions();
			if (include_positions.size() > 1) {
				//error is in an include
				error_line = include_positions[0].line;
				error_text = "error(" + itos(error_line) + ") in include " + include_positions[include_positions.size() - 1].file + ":" + itos(include_positions[include_positions.size() - 1].line) + ": " + sl.get_error_text();
				set_error_count(include_positions.size() - 1);
			} else {
				error_line = sl.get_error_line();
				error_text = "error(" + itos(error_line) + "): " + sl.get_error_text();
				set_error_count(0);
			}
			set_error(error_text);
			set_error_pos(error_line - 1, 0);
			get_text_editor()->set_line_background_color(error_line - 1, marked_line_color);
		} else {
			set_error("");
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
			set_warning_count(0);
		}
	}

	emit_signal(SNAME("script_validated"), last_compile_result == OK); // Notify that validation finished, to update the list of scripts
}

void ShaderTextEditor::_update_warning_panel() {
	int warning_count = 0;

	warnings_panel->push_table(2);
	for (int i = 0; i < warnings.size(); i++) {
		ShaderWarning &w = warnings[i];

		if (warning_count == 0) {
			if (saved_treat_warning_as_errors) {
				String error_text = "error(" + itos(w.get_line()) + "): " + w.get_message() + " " + TTR("Warnings should be fixed to prevent errors.");
				set_error_pos(w.get_line() - 1, 0);
				set_error(error_text);
				get_text_editor()->set_line_background_color(w.get_line() - 1, marked_line_color);
			}
		}

		warning_count++;
		int line = w.get_line();

		// First cell.
		warnings_panel->push_cell();
		warnings_panel->push_color(warnings_panel->get_theme_color(SNAME("warning_color"), SNAME("Editor")));
		if (line != -1) {
			warnings_panel->push_meta(line - 1);
			warnings_panel->add_text(TTR("Line") + " " + itos(line));
			warnings_panel->add_text(" (" + w.get_name() + "):");
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

	set_warning_count(warning_count);
}

void ShaderTextEditor::_bind_methods() {
	ADD_SIGNAL(MethodInfo("script_validated", PropertyInfo(Variant::BOOL, "valid")));
}

ShaderTextEditor::ShaderTextEditor() {
	syntax_highlighter.instantiate();
	get_text_editor()->set_syntax_highlighter(syntax_highlighter);
}

/*** SCRIPT EDITOR ******/

void ShaderEditor::_menu_option(int p_option) {
	switch (p_option) {
		case EDIT_UNDO: {
			shader_editor->get_text_editor()->undo();
		} break;
		case EDIT_REDO: {
			shader_editor->get_text_editor()->redo();
		} break;
		case EDIT_CUT: {
			shader_editor->get_text_editor()->cut();
		} break;
		case EDIT_COPY: {
			shader_editor->get_text_editor()->copy();
		} break;
		case EDIT_PASTE: {
			shader_editor->get_text_editor()->paste();
		} break;
		case EDIT_SELECT_ALL: {
			shader_editor->get_text_editor()->select_all();
		} break;
		case EDIT_MOVE_LINE_UP: {
			shader_editor->move_lines_up();
		} break;
		case EDIT_MOVE_LINE_DOWN: {
			shader_editor->move_lines_down();
		} break;
		case EDIT_INDENT_LEFT: {
			if (shader.is_null()) {
				return;
			}
			shader_editor->get_text_editor()->unindent_lines();
		} break;
		case EDIT_INDENT_RIGHT: {
			if (shader.is_null()) {
				return;
			}
			shader_editor->get_text_editor()->indent_lines();
		} break;
		case EDIT_DELETE_LINE: {
			shader_editor->delete_lines();
		} break;
		case EDIT_DUPLICATE_SELECTION: {
			shader_editor->duplicate_selection();
		} break;
		case EDIT_TOGGLE_COMMENT: {
			if (shader.is_null()) {
				return;
			}

			shader_editor->toggle_inline_comment("//");

		} break;
		case EDIT_COMPLETE: {
			shader_editor->get_text_editor()->request_code_completion();
		} break;
		case SEARCH_FIND: {
			shader_editor->get_find_replace_bar()->popup_search();
		} break;
		case SEARCH_FIND_NEXT: {
			shader_editor->get_find_replace_bar()->search_next();
		} break;
		case SEARCH_FIND_PREV: {
			shader_editor->get_find_replace_bar()->search_prev();
		} break;
		case SEARCH_REPLACE: {
			shader_editor->get_find_replace_bar()->popup_replace();
		} break;
		case SEARCH_GOTO_LINE: {
			goto_line_dialog->popup_find_line(shader_editor->get_text_editor());
		} break;
		case BOOKMARK_TOGGLE: {
			shader_editor->toggle_bookmark();
		} break;
		case BOOKMARK_GOTO_NEXT: {
			shader_editor->goto_next_bookmark();
		} break;
		case BOOKMARK_GOTO_PREV: {
			shader_editor->goto_prev_bookmark();
		} break;
		case BOOKMARK_REMOVE_ALL: {
			shader_editor->remove_all_bookmarks();
		} break;
		case HELP_DOCS: {
			OS::get_singleton()->shell_open(vformat("%s/tutorials/shaders/shader_reference/index.html", VERSION_DOCS_URL));
		} break;
	}
	if (p_option != SEARCH_FIND && p_option != SEARCH_REPLACE && p_option != SEARCH_GOTO_LINE) {
		shader_editor->get_text_editor()->call_deferred(SNAME("grab_focus"));
	}
}

void ShaderEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			PopupMenu *popup = help_menu->get_popup();
			popup->set_item_icon(popup->get_item_index(HELP_DOCS), get_theme_icon(SNAME("ExternalLink"), SNAME("EditorIcons")));
		} break;

		case NOTIFICATION_WM_WINDOW_FOCUS_IN: {
			_check_for_external_edit();
		} break;
	}
}

void ShaderEditor::_editor_settings_changed() {
	shader_editor->update_editor_settings();

	shader_editor->get_text_editor()->add_theme_constant_override("line_spacing", EditorSettings::get_singleton()->get("text_editor/appearance/whitespace/line_spacing"));
	shader_editor->get_text_editor()->set_draw_breakpoints_gutter(false);
	shader_editor->get_text_editor()->set_draw_executing_lines_gutter(false);
}

void ShaderEditor::_show_warnings_panel(bool p_show) {
	warnings_panel->set_visible(p_show);
}

void ShaderEditor::_warning_clicked(Variant p_line) {
	if (p_line.get_type() == Variant::INT) {
		shader_editor->get_text_editor()->set_caret_line(p_line.operator int64_t());
	}
}

void ShaderEditor::_bind_methods() {
	ClassDB::bind_method("_show_warnings_panel", &ShaderEditor::_show_warnings_panel);
	ClassDB::bind_method("_warning_clicked", &ShaderEditor::_warning_clicked);

	ADD_SIGNAL(MethodInfo("validation_changed"));
}

void ShaderEditor::ensure_select_current() {
}

void ShaderEditor::goto_line_selection(int p_line, int p_begin, int p_end) {
	shader_editor->goto_line_selection(p_line, p_begin, p_end);
}

void ShaderEditor::_project_settings_changed() {
	_update_warnings(true);
}

void ShaderEditor::_update_warnings(bool p_validate) {
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

	if (p_validate && changed && shader_editor && shader_editor->get_edited_shader().is_valid()) {
		shader_editor->validate_script();
	}
}

void ShaderEditor::_check_for_external_edit() {
	bool use_autoreload = bool(EDITOR_GET("text_editor/behavior/files/auto_reload_scripts_on_external_change"));

	if (shader_inc.is_valid()) {
		if (shader_inc->get_last_modified_time() != FileAccess::get_modified_time(shader_inc->get_path())) {
			if (use_autoreload) {
				_reload_shader_include_from_disk();
			} else {
				disk_changed->call_deferred(SNAME("popup_centered"));
			}
		}
		return;
	}

	if (shader.is_null() || shader->is_built_in()) {
		return;
	}

	if (shader->get_last_modified_time() != FileAccess::get_modified_time(shader->get_path())) {
		if (use_autoreload) {
			_reload_shader_from_disk();
		} else {
			disk_changed->call_deferred(SNAME("popup_centered"));
		}
	}
}

void ShaderEditor::_reload_shader_from_disk() {
	Ref<Shader> rel_shader = ResourceLoader::load(shader->get_path(), shader->get_class(), ResourceFormatLoader::CACHE_MODE_IGNORE);
	ERR_FAIL_COND(!rel_shader.is_valid());

	shader_editor->set_block_shader_changed(true);
	shader->set_code(rel_shader->get_code());
	shader_editor->set_block_shader_changed(false);
	shader->set_last_modified_time(rel_shader->get_last_modified_time());
	shader_editor->reload_text();
}

void ShaderEditor::_reload_shader_include_from_disk() {
	Ref<ShaderInclude> rel_shader_include = ResourceLoader::load(shader_inc->get_path(), shader_inc->get_class(), ResourceFormatLoader::CACHE_MODE_IGNORE);
	ERR_FAIL_COND(!rel_shader_include.is_valid());

	shader_editor->set_block_shader_changed(true);
	shader_inc->set_code(rel_shader_include->get_code());
	shader_editor->set_block_shader_changed(false);
	shader_inc->set_last_modified_time(rel_shader_include->get_last_modified_time());
	shader_editor->reload_text();
}

void ShaderEditor::_reload() {
	if (shader.is_valid()) {
		_reload_shader_from_disk();
	} else if (shader_inc.is_valid()) {
		_reload_shader_include_from_disk();
	}
}

void ShaderEditor::edit(const Ref<Shader> &p_shader) {
	if (p_shader.is_null() || !p_shader->is_text_shader()) {
		return;
	}

	if (shader == p_shader) {
		return;
	}

	shader = p_shader;
	shader_inc = Ref<ShaderInclude>();

	shader_editor->set_edited_shader(shader);
}

void ShaderEditor::edit(const Ref<ShaderInclude> &p_shader_inc) {
	if (p_shader_inc.is_null()) {
		return;
	}

	if (shader_inc == p_shader_inc) {
		return;
	}

	shader_inc = p_shader_inc;
	shader = Ref<Shader>();

	shader_editor->set_edited_shader_include(p_shader_inc);
}

void ShaderEditor::save_external_data(const String &p_str) {
	if (shader.is_null() && shader_inc.is_null()) {
		disk_changed->hide();
		return;
	}

	apply_shaders();

	Ref<Shader> edited_shader = shader_editor->get_edited_shader();
	if (edited_shader.is_valid()) {
		ResourceSaver::save(edited_shader);
	}
	if (shader.is_valid() && shader != edited_shader) {
		ResourceSaver::save(shader);
	}

	Ref<ShaderInclude> edited_shader_inc = shader_editor->get_edited_shader_include();
	if (edited_shader_inc.is_valid()) {
		ResourceSaver::save(edited_shader_inc);
	}
	if (shader_inc.is_valid() && shader_inc != edited_shader_inc) {
		ResourceSaver::save(shader_inc);
	}
	shader_editor->get_text_editor()->tag_saved_version();

	disk_changed->hide();
}

void ShaderEditor::validate_script() {
	shader_editor->_validate_script();
}

bool ShaderEditor::is_unsaved() const {
	return shader_editor->get_text_editor()->get_saved_version() != shader_editor->get_text_editor()->get_version();
}

void ShaderEditor::apply_shaders() {
	String editor_code = shader_editor->get_text_editor()->get_text();
	if (shader.is_valid()) {
		String shader_code = shader->get_code();
		if (shader_code != editor_code || dependencies_version != shader_editor->get_dependencies_version()) {
			shader_editor->set_block_shader_changed(true);
			shader->set_code(editor_code);
			shader_editor->set_block_shader_changed(false);
			shader->set_edited(true);
		}
	}
	if (shader_inc.is_valid()) {
		String shader_inc_code = shader_inc->get_code();
		if (shader_inc_code != editor_code || dependencies_version != shader_editor->get_dependencies_version()) {
			shader_editor->set_block_shader_changed(true);
			shader_inc->set_code(editor_code);
			shader_editor->set_block_shader_changed(false);
			shader_inc->set_edited(true);
		}
	}

	dependencies_version = shader_editor->get_dependencies_version();
}

void ShaderEditor::_text_edit_gui_input(const Ref<InputEvent> &ev) {
	Ref<InputEventMouseButton> mb = ev;

	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::RIGHT && mb->is_pressed()) {
			CodeEdit *tx = shader_editor->get_text_editor();

			Point2i pos = tx->get_line_column_at_pos(mb->get_global_position() - tx->get_global_position());
			int row = pos.y;
			int col = pos.x;
			tx->set_move_caret_on_right_click_enabled(EditorSettings::get_singleton()->get("text_editor/behavior/navigation/move_caret_on_right_click"));

			if (tx->is_move_caret_on_right_click_enabled()) {
				if (tx->has_selection()) {
					int from_line = tx->get_selection_from_line();
					int to_line = tx->get_selection_to_line();
					int from_column = tx->get_selection_from_column();
					int to_column = tx->get_selection_to_column();

					if (row < from_line || row > to_line || (row == from_line && col < from_column) || (row == to_line && col > to_column)) {
						// Right click is outside the selected text
						tx->deselect();
					}
				}
				if (!tx->has_selection()) {
					tx->set_caret_line(row, true, false);
					tx->set_caret_column(col);
				}
			}
			_make_context_menu(tx->has_selection(), get_local_mouse_position());
		}
	}

	Ref<InputEventKey> k = ev;
	if (k.is_valid() && k->is_pressed() && k->is_action("ui_menu", true)) {
		CodeEdit *tx = shader_editor->get_text_editor();
		tx->adjust_viewport_to_caret();
		_make_context_menu(tx->has_selection(), (get_global_transform().inverse() * tx->get_global_transform()).xform(tx->get_caret_draw_pos()));
		context_menu->grab_focus();
	}
}

void ShaderEditor::_update_bookmark_list() {
	bookmarks_menu->clear();

	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_bookmark"), BOOKMARK_TOGGLE);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/remove_all_bookmarks"), BOOKMARK_REMOVE_ALL);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_next_bookmark"), BOOKMARK_GOTO_NEXT);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_previous_bookmark"), BOOKMARK_GOTO_PREV);

	Array bookmark_list = shader_editor->get_text_editor()->get_bookmarked_lines();
	if (bookmark_list.size() == 0) {
		return;
	}

	bookmarks_menu->add_separator();

	for (int i = 0; i < bookmark_list.size(); i++) {
		String line = shader_editor->get_text_editor()->get_line(bookmark_list[i]).strip_edges();
		// Limit the size of the line if too big.
		if (line.length() > 50) {
			line = line.substr(0, 50);
		}

		bookmarks_menu->add_item(String::num((int)bookmark_list[i] + 1) + " - \"" + line + "\"");
		bookmarks_menu->set_item_metadata(-1, bookmark_list[i]);
	}
}

void ShaderEditor::_bookmark_item_pressed(int p_idx) {
	if (p_idx < 4) { // Any item before the separator.
		_menu_option(bookmarks_menu->get_item_id(p_idx));
	} else {
		shader_editor->goto_line(bookmarks_menu->get_item_metadata(p_idx));
	}
}

void ShaderEditor::_make_context_menu(bool p_selection, Vector2 p_position) {
	context_menu->clear();
	if (p_selection) {
		context_menu->add_shortcut(ED_GET_SHORTCUT("ui_cut"), EDIT_CUT);
		context_menu->add_shortcut(ED_GET_SHORTCUT("ui_copy"), EDIT_COPY);
	}

	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_paste"), EDIT_PASTE);
	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_text_select_all"), EDIT_SELECT_ALL);
	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_undo"), EDIT_UNDO);
	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_redo"), EDIT_REDO);

	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_left"), EDIT_INDENT_LEFT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_right"), EDIT_INDENT_RIGHT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_bookmark"), BOOKMARK_TOGGLE);

	context_menu->set_position(get_screen_position() + p_position);
	context_menu->reset_size();
	context_menu->popup();
}

ShaderEditor::ShaderEditor() {
	GLOBAL_DEF("debug/shader_language/warnings/enable", true);
	GLOBAL_DEF("debug/shader_language/warnings/treat_warnings_as_errors", false);
	for (int i = 0; i < (int)ShaderWarning::WARNING_MAX; i++) {
		GLOBAL_DEF("debug/shader_language/warnings/" + ShaderWarning::get_name_from_code((ShaderWarning::Code)i).to_lower(), true);
	}
	_update_warnings(false);

	shader_editor = memnew(ShaderTextEditor);

	shader_editor->connect("script_validated", callable_mp(this, &ShaderEditor::_script_validated));

	shader_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	shader_editor->add_theme_constant_override("separation", 0);
	shader_editor->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);

	shader_editor->connect("show_warnings_panel", callable_mp(this, &ShaderEditor::_show_warnings_panel));
	shader_editor->connect("script_changed", callable_mp(this, &ShaderEditor::apply_shaders));
	EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &ShaderEditor::_editor_settings_changed));
	ProjectSettingsEditor::get_singleton()->connect("confirmed", callable_mp(this, &ShaderEditor::_project_settings_changed));

	shader_editor->get_text_editor()->set_code_hint_draw_below(EditorSettings::get_singleton()->get("text_editor/completion/put_callhint_tooltip_below_current_line"));

	shader_editor->get_text_editor()->set_symbol_lookup_on_click_enabled(true);
	shader_editor->get_text_editor()->set_context_menu_enabled(false);
	shader_editor->get_text_editor()->connect("gui_input", callable_mp(this, &ShaderEditor::_text_edit_gui_input));

	shader_editor->update_editor_settings();

	context_menu = memnew(PopupMenu);
	add_child(context_menu);
	context_menu->connect("id_pressed", callable_mp(this, &ShaderEditor::_menu_option));

	VBoxContainer *main_container = memnew(VBoxContainer);
	HBoxContainer *hbc = memnew(HBoxContainer);

	edit_menu = memnew(MenuButton);
	edit_menu->set_shortcut_context(this);
	edit_menu->set_text(TTR("Edit"));
	edit_menu->set_switch_on_hover(true);

	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_undo"), EDIT_UNDO);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_redo"), EDIT_REDO);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_cut"), EDIT_CUT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_copy"), EDIT_COPY);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_paste"), EDIT_PASTE);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_text_select_all"), EDIT_SELECT_ALL);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_up"), EDIT_MOVE_LINE_UP);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_down"), EDIT_MOVE_LINE_DOWN);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_left"), EDIT_INDENT_LEFT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_right"), EDIT_INDENT_RIGHT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/delete_line"), EDIT_DELETE_LINE);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/duplicate_selection"), EDIT_DUPLICATE_SELECTION);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_text_completion_query"), EDIT_COMPLETE);
	edit_menu->get_popup()->connect("id_pressed", callable_mp(this, &ShaderEditor::_menu_option));

	search_menu = memnew(MenuButton);
	search_menu->set_shortcut_context(this);
	search_menu->set_text(TTR("Search"));
	search_menu->set_switch_on_hover(true);

	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find"), SEARCH_FIND);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_next"), SEARCH_FIND_NEXT);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_previous"), SEARCH_FIND_PREV);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/replace"), SEARCH_REPLACE);
	search_menu->get_popup()->connect("id_pressed", callable_mp(this, &ShaderEditor::_menu_option));

	MenuButton *goto_menu = memnew(MenuButton);
	goto_menu->set_shortcut_context(this);
	goto_menu->set_text(TTR("Go To"));
	goto_menu->set_switch_on_hover(true);
	goto_menu->get_popup()->connect("id_pressed", callable_mp(this, &ShaderEditor::_menu_option));

	goto_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_line"), SEARCH_GOTO_LINE);
	goto_menu->get_popup()->add_separator();

	bookmarks_menu = memnew(PopupMenu);
	bookmarks_menu->set_name("Bookmarks");
	goto_menu->get_popup()->add_child(bookmarks_menu);
	goto_menu->get_popup()->add_submenu_item(TTR("Bookmarks"), "Bookmarks");
	_update_bookmark_list();
	bookmarks_menu->connect("about_to_popup", callable_mp(this, &ShaderEditor::_update_bookmark_list));
	bookmarks_menu->connect("index_pressed", callable_mp(this, &ShaderEditor::_bookmark_item_pressed));

	help_menu = memnew(MenuButton);
	help_menu->set_text(TTR("Help"));
	help_menu->set_switch_on_hover(true);
	help_menu->get_popup()->add_item(TTR("Online Docs"), HELP_DOCS);
	help_menu->get_popup()->connect("id_pressed", callable_mp(this, &ShaderEditor::_menu_option));

	add_child(main_container);
	main_container->add_child(hbc);
	hbc->add_child(search_menu);
	hbc->add_child(edit_menu);
	hbc->add_child(goto_menu);
	hbc->add_child(help_menu);
	hbc->add_theme_style_override("panel", EditorNode::get_singleton()->get_gui_base()->get_theme_stylebox(SNAME("ScriptEditorPanel"), SNAME("EditorStyles")));

	VSplitContainer *editor_box = memnew(VSplitContainer);
	main_container->add_child(editor_box);
	editor_box->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	editor_box->set_v_size_flags(SIZE_EXPAND_FILL);
	editor_box->add_child(shader_editor);

	FindReplaceBar *bar = memnew(FindReplaceBar);
	main_container->add_child(bar);
	bar->hide();
	shader_editor->set_find_replace_bar(bar);

	warnings_panel = memnew(RichTextLabel);
	warnings_panel->set_custom_minimum_size(Size2(0, 100 * EDSCALE));
	warnings_panel->set_h_size_flags(SIZE_EXPAND_FILL);
	warnings_panel->set_meta_underline(true);
	warnings_panel->set_selection_enabled(true);
	warnings_panel->set_focus_mode(FOCUS_CLICK);
	warnings_panel->hide();
	warnings_panel->connect("meta_clicked", callable_mp(this, &ShaderEditor::_warning_clicked));
	editor_box->add_child(warnings_panel);
	shader_editor->set_warnings_panel(warnings_panel);

	goto_line_dialog = memnew(GotoLineDialog);
	add_child(goto_line_dialog);

	disk_changed = memnew(ConfirmationDialog);

	VBoxContainer *vbc = memnew(VBoxContainer);
	disk_changed->add_child(vbc);

	Label *dl = memnew(Label);
	dl->set_text(TTR("This shader has been modified on disk.\nWhat action should be taken?"));
	vbc->add_child(dl);

	disk_changed->connect("confirmed", callable_mp(this, &ShaderEditor::_reload));
	disk_changed->set_ok_button_text(TTR("Reload"));

	disk_changed->add_button(TTR("Resave"), !DisplayServer::get_singleton()->get_swap_cancel_ok(), "resave");
	disk_changed->connect("custom_action", callable_mp(this, &ShaderEditor::save_external_data));

	add_child(disk_changed);

	_editor_settings_changed();
}

void ShaderEditorPlugin::_update_shader_list() {
	shader_list->clear();
	for (uint32_t i = 0; i < edited_shaders.size(); i++) {
		Ref<Resource> shader = edited_shaders[i].shader;
		if (shader.is_null()) {
			shader = edited_shaders[i].shader_inc;
		}

		String path = shader->get_path();
		String text = path.get_file();
		if (text.is_empty()) {
			// This appears for newly created built-in shaders before saving the scene.
			text = TTR("[unsaved]");
		} else if (shader->is_built_in()) {
			const String &shader_name = shader->get_name();
			if (!shader_name.is_empty()) {
				text = vformat("%s (%s)", shader_name, text.get_slice("::", 0));
			}
		}

		bool unsaved = false;
		if (edited_shaders[i].shader_editor) {
			unsaved = edited_shaders[i].shader_editor->is_unsaved();
		}
		// TODO: Handle visual shaders too.

		if (unsaved) {
			text += "(*)";
		}

		String _class = shader->get_class();
		if (!shader_list->has_theme_icon(_class, SNAME("EditorIcons"))) {
			_class = "TextFile";
		}
		Ref<Texture2D> icon = shader_list->get_theme_icon(_class, SNAME("EditorIcons"));

		shader_list->add_item(text, icon);
		shader_list->set_item_tooltip(shader_list->get_item_count() - 1, path);
	}

	if (shader_tabs->get_tab_count()) {
		shader_list->select(shader_tabs->get_current_tab());
	}

	for (int i = 1; i < FILE_MAX; i++) {
		file_menu->get_popup()->set_item_disabled(file_menu->get_popup()->get_item_index(i), edited_shaders.size() == 0);
	}

	_update_shader_list_status();
}

void ShaderEditorPlugin::_update_shader_list_status() {
	for (int i = 0; i < shader_list->get_item_count(); i++) {
		ShaderEditor *se = Object::cast_to<ShaderEditor>(shader_tabs->get_tab_control(i));
		if (se) {
			if (se->was_compilation_successful()) {
				shader_list->set_item_tag_icon(i, Ref<Texture2D>());
			} else {
				shader_list->set_item_tag_icon(i, shader_list->get_theme_icon(SNAME("Error"), SNAME("EditorIcons")));
			}
		}
	}
}

void ShaderEditorPlugin::edit(Object *p_object) {
	EditedShader es;

	ShaderInclude *si = Object::cast_to<ShaderInclude>(p_object);
	if (si != nullptr) {
		for (uint32_t i = 0; i < edited_shaders.size(); i++) {
			if (edited_shaders[i].shader_inc.ptr() == si) {
				shader_tabs->set_current_tab(i);
				shader_list->select(i);
				return;
			}
		}
		es.shader_inc = Ref<ShaderInclude>(si);
		es.shader_editor = memnew(ShaderEditor);
		es.shader_editor->edit(si);
		shader_tabs->add_child(es.shader_editor);
		es.shader_editor->connect("validation_changed", callable_mp(this, &ShaderEditorPlugin::_update_shader_list));
	} else {
		Shader *s = Object::cast_to<Shader>(p_object);
		for (uint32_t i = 0; i < edited_shaders.size(); i++) {
			if (edited_shaders[i].shader.ptr() == s) {
				shader_tabs->set_current_tab(i);
				shader_list->select(i);
				return;
			}
		}
		es.shader = Ref<Shader>(s);
		Ref<VisualShader> vs = es.shader;
		if (vs.is_valid()) {
			es.visual_shader_editor = memnew(VisualShaderEditor);
			shader_tabs->add_child(es.visual_shader_editor);
			es.visual_shader_editor->edit(vs.ptr());
		} else {
			es.shader_editor = memnew(ShaderEditor);
			shader_tabs->add_child(es.shader_editor);
			es.shader_editor->edit(s);
			es.shader_editor->connect("validation_changed", callable_mp(this, &ShaderEditorPlugin::_update_shader_list));
		}
	}

	shader_tabs->set_current_tab(shader_tabs->get_tab_count() - 1);
	edited_shaders.push_back(es);
	_update_shader_list();
}

bool ShaderEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<Shader>(p_object) != nullptr || Object::cast_to<ShaderInclude>(p_object) != nullptr;
}

void ShaderEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		EditorNode::get_singleton()->make_bottom_panel_item_visible(main_split);
	}
}

void ShaderEditorPlugin::selected_notify() {
}

ShaderEditor *ShaderEditorPlugin::get_shader_editor(const Ref<Shader> &p_for_shader) {
	for (uint32_t i = 0; i < edited_shaders.size(); i++) {
		if (edited_shaders[i].shader == p_for_shader) {
			return edited_shaders[i].shader_editor;
		}
	}
	return nullptr;
}

VisualShaderEditor *ShaderEditorPlugin::get_visual_shader_editor(const Ref<Shader> &p_for_shader) {
	for (uint32_t i = 0; i < edited_shaders.size(); i++) {
		if (edited_shaders[i].shader == p_for_shader) {
			return edited_shaders[i].visual_shader_editor;
		}
	}
	return nullptr;
}

void ShaderEditorPlugin::save_external_data() {
	for (uint32_t i = 0; i < edited_shaders.size(); i++) {
		if (edited_shaders[i].shader_editor) {
			edited_shaders[i].shader_editor->save_external_data();
		}
	}
	_update_shader_list();
}

void ShaderEditorPlugin::apply_changes() {
	for (uint32_t i = 0; i < edited_shaders.size(); i++) {
		if (edited_shaders[i].shader_editor) {
			edited_shaders[i].shader_editor->apply_shaders();
		}
	}
}

void ShaderEditorPlugin::_shader_selected(int p_index) {
	if (edited_shaders[p_index].shader_editor) {
		edited_shaders[p_index].shader_editor->validate_script();
	}
	shader_tabs->set_current_tab(p_index);
}

void ShaderEditorPlugin::_shader_list_clicked(int p_item, Vector2 p_local_mouse_pos, MouseButton p_mouse_button_index) {
	if (p_mouse_button_index == MouseButton::MIDDLE) {
		_close_shader(p_item);
	}
}

void ShaderEditorPlugin::_close_shader(int p_index) {
	int index = shader_tabs->get_current_tab();
	ERR_FAIL_INDEX(index, shader_tabs->get_tab_count());
	Control *c = shader_tabs->get_tab_control(index);
	memdelete(c);
	edited_shaders.remove_at(index);
	_update_shader_list();
	EditorNode::get_singleton()->get_undo_redo()->clear_history(); // To prevent undo on deleted graphs.
}

void ShaderEditorPlugin::_resource_saved(Object *obj) {
	// May have been renamed on save.
	for (uint32_t i = 0; i < edited_shaders.size(); i++) {
		if (edited_shaders[i].shader.ptr() == obj) {
			_update_shader_list();
			return;
		}
	}
}

void ShaderEditorPlugin::_menu_item_pressed(int p_index) {
	switch (p_index) {
		case FILE_NEW: {
			String base_path = FileSystemDock::get_singleton()->get_current_path().get_base_dir();
			shader_create_dialog->config(base_path.plus_file("new_shader"), false, false, 0);
			shader_create_dialog->popup_centered();
		} break;
		case FILE_NEW_INCLUDE: {
			String base_path = FileSystemDock::get_singleton()->get_current_path().get_base_dir();
			shader_create_dialog->config(base_path.plus_file("new_shader"), false, false, 2);
			shader_create_dialog->popup_centered();
		} break;
		case FILE_OPEN: {
			InspectorDock::get_singleton()->open_resource("Shader");
		} break;
		case FILE_OPEN_INCLUDE: {
			InspectorDock::get_singleton()->open_resource("ShaderInclude");
		} break;
		case FILE_SAVE: {
			int index = shader_tabs->get_current_tab();
			ERR_FAIL_INDEX(index, shader_tabs->get_tab_count());
			if (edited_shaders[index].shader.is_valid()) {
				EditorNode::get_singleton()->save_resource(edited_shaders[index].shader);
			} else {
				EditorNode::get_singleton()->save_resource(edited_shaders[index].shader_inc);
			}
		} break;
		case FILE_SAVE_AS: {
			int index = shader_tabs->get_current_tab();
			ERR_FAIL_INDEX(index, shader_tabs->get_tab_count());
			String path;
			if (edited_shaders[index].shader.is_valid()) {
				path = edited_shaders[index].shader->get_path();
				if (!path.is_resource_file()) {
					path = "";
				}
				EditorNode::get_singleton()->save_resource_as(edited_shaders[index].shader, path);
			} else {
				path = edited_shaders[index].shader_inc->get_path();
				if (!path.is_resource_file()) {
					path = "";
				}
				EditorNode::get_singleton()->save_resource_as(edited_shaders[index].shader_inc, path);
			}
		} break;
		case FILE_INSPECT: {
			int index = shader_tabs->get_current_tab();
			ERR_FAIL_INDEX(index, shader_tabs->get_tab_count());
			if (edited_shaders[index].shader.is_valid()) {
				EditorNode::get_singleton()->push_item(edited_shaders[index].shader.ptr());
			} else {
				EditorNode::get_singleton()->push_item(edited_shaders[index].shader_inc.ptr());
			}
		} break;
		case FILE_CLOSE: {
			_close_shader(shader_tabs->get_current_tab());
		} break;
	}
}

void ShaderEditorPlugin::_shader_created(Ref<Shader> p_shader) {
	EditorNode::get_singleton()->push_item(p_shader.ptr());
}

void ShaderEditorPlugin::_shader_include_created(Ref<ShaderInclude> p_shader_inc) {
	EditorNode::get_singleton()->push_item(p_shader_inc.ptr());
}

ShaderEditorPlugin::ShaderEditorPlugin() {
	main_split = memnew(HSplitContainer);

	VBoxContainer *vb = memnew(VBoxContainer);

	HBoxContainer *file_hb = memnew(HBoxContainer);
	vb->add_child(file_hb);
	file_menu = memnew(MenuButton);
	file_menu->set_text(TTR("File"));
	file_menu->get_popup()->add_item(TTR("New Shader"), FILE_NEW);
	file_menu->get_popup()->add_item(TTR("New Shader Include"), FILE_NEW_INCLUDE);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_item(TTR("Load Shader File"), FILE_OPEN);
	file_menu->get_popup()->add_item(TTR("Load Shader Include File"), FILE_OPEN_INCLUDE);
	file_menu->get_popup()->add_item(TTR("Save File"), FILE_SAVE);
	file_menu->get_popup()->add_item(TTR("Save File As"), FILE_SAVE_AS);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_item(TTR("Open File in Inspector"), FILE_INSPECT);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_item(TTR("Close File"), FILE_CLOSE);
	file_menu->get_popup()->connect("id_pressed", callable_mp(this, &ShaderEditorPlugin::_menu_item_pressed));
	file_hb->add_child(file_menu);

	for (int i = 2; i < FILE_MAX; i++) {
		file_menu->get_popup()->set_item_disabled(file_menu->get_popup()->get_item_index(i), true);
	}

	shader_list = memnew(ItemList);
	shader_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb->add_child(shader_list);
	shader_list->connect("item_selected", callable_mp(this, &ShaderEditorPlugin::_shader_selected));
	shader_list->connect("item_clicked", callable_mp(this, &ShaderEditorPlugin::_shader_list_clicked));

	main_split->add_child(vb);
	vb->set_custom_minimum_size(Size2(200, 300) * EDSCALE);

	shader_tabs = memnew(TabContainer);
	shader_tabs->set_tabs_visible(false);
	shader_tabs->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_split->add_child(shader_tabs);
	Ref<StyleBoxEmpty> empty;
	empty.instantiate();
	shader_tabs->add_theme_style_override("panel", empty);

	button = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Shader Editor"), main_split);

	// Defer connect because Editor class is not in the binding system yet.
	EditorNode::get_singleton()->call_deferred("connect", "resource_saved", callable_mp(this, &ShaderEditorPlugin::_resource_saved), CONNECT_DEFERRED);

	shader_create_dialog = memnew(ShaderCreateDialog);
	vb->add_child(shader_create_dialog);
	shader_create_dialog->connect("shader_created", callable_mp(this, &ShaderEditorPlugin::_shader_created));
	shader_create_dialog->connect("shader_include_created", callable_mp(this, &ShaderEditorPlugin::_shader_include_created));
}

ShaderEditorPlugin::~ShaderEditorPlugin() {
}

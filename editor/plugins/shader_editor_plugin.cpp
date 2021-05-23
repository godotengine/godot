/*************************************************************************/
/*  shader_editor_plugin.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/project_settings_editor.h"
#include "editor/property_editor.h"
#include "servers/display_server.h"
#include "servers/rendering/shader_types.h"

/*** SHADER SCRIPT EDITOR ****/

static bool saved_warnings_enabled = false;
static bool saved_treat_warning_as_errors = false;
static Map<ShaderWarning::Code, bool> saved_warnings;
static uint32_t saved_warning_flags = 0U;

Ref<Shader> ShaderTextEditor::get_edited_shader() const {
	return shader;
}

void ShaderTextEditor::set_edited_shader(const Ref<Shader> &p_shader) {
	if (shader == p_shader) {
		return;
	}
	shader = p_shader;

	_load_theme_settings();

	get_text_editor()->set_text(p_shader->get_code());
	get_text_editor()->clear_undo_history();
	get_text_editor()->call_deferred("set_h_scroll", 0);
	get_text_editor()->call_deferred("set_v_scroll", 0);

	_validate_script();
	_line_col_changed();
}

void ShaderTextEditor::reload_text() {
	ERR_FAIL_COND(shader.is_null());

	CodeEdit *te = get_text_editor();
	int column = te->cursor_get_column();
	int row = te->cursor_get_line();
	int h = te->get_h_scroll();
	int v = te->get_v_scroll();

	te->set_text(shader->get_code());
	te->cursor_set_line(row);
	te->cursor_set_column(column);
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
	Color updated_marked_line_color = EDITOR_GET("text_editor/highlighting/mark_color");
	if (updated_marked_line_color != marked_line_color) {
		for (int i = 0; i < text_editor->get_line_count(); i++) {
			if (text_editor->get_line_background_color(i) == marked_line_color) {
				text_editor->set_line_background_color(i, updated_marked_line_color);
			}
		}
		marked_line_color = updated_marked_line_color;
	}

	syntax_highlighter->set_number_color(EDITOR_GET("text_editor/highlighting/number_color"));
	syntax_highlighter->set_symbol_color(EDITOR_GET("text_editor/highlighting/symbol_color"));
	syntax_highlighter->set_function_color(EDITOR_GET("text_editor/highlighting/function_color"));
	syntax_highlighter->set_member_variable_color(EDITOR_GET("text_editor/highlighting/member_variable_color"));

	syntax_highlighter->clear_keyword_colors();

	List<String> keywords;
	ShaderLanguage::get_keyword_list(&keywords);
	const Color keyword_color = EDITOR_GET("text_editor/highlighting/keyword_color");
	const Color control_flow_keyword_color = EDITOR_GET("text_editor/highlighting/control_flow_keyword_color");

	for (List<String>::Element *E = keywords.front(); E; E = E->next()) {
		if (ShaderLanguage::is_control_flow_keyword(E->get())) {
			syntax_highlighter->add_keyword_color(E->get(), control_flow_keyword_color);
		} else {
			syntax_highlighter->add_keyword_color(E->get(), keyword_color);
		}
	}

	// Colorize built-ins like `COLOR` differently to make them easier
	// to distinguish from keywords at a quick glance.

	List<String> built_ins;
	if (shader.is_valid()) {
		for (const Map<StringName, ShaderLanguage::FunctionInfo>::Element *E = ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(shader->get_mode())).front(); E; E = E->next()) {
			for (const Map<StringName, ShaderLanguage::BuiltInInfo>::Element *F = E->get().built_ins.front(); F; F = F->next()) {
				built_ins.push_back(F->key());
			}
		}

		for (int i = 0; i < ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(shader->get_mode())).size(); i++) {
			built_ins.push_back(ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(shader->get_mode()))[i]);
		}
	}

	const Color member_variable_color = EDITOR_GET("text_editor/highlighting/member_variable_color");

	for (List<String>::Element *E = built_ins.front(); E; E = E->next()) {
		syntax_highlighter->add_keyword_color(E->get(), member_variable_color);
	}

	// Colorize comments.
	const Color comment_color = EDITOR_GET("text_editor/highlighting/comment_color");
	syntax_highlighter->clear_color_regions();
	syntax_highlighter->add_color_region("/*", "*/", comment_color, false);
	syntax_highlighter->add_color_region("//", "", comment_color, true);

	if (warnings_panel) {
		// Warnings panel
		warnings_panel->add_theme_font_override("normal_font", EditorNode::get_singleton()->get_gui_base()->get_theme_font("main", "EditorFonts"));
		warnings_panel->add_theme_font_size_override("normal_font_size", EditorNode::get_singleton()->get_gui_base()->get_theme_font_size("main_size", "EditorFonts"));
	}
}

void ShaderTextEditor::_check_shader_mode() {
	String type = ShaderLanguage::get_shader_type(get_text_editor()->get_text());

	Shader::Mode mode;

	if (type == "canvas_item") {
		mode = Shader::MODE_CANVAS_ITEM;
	} else if (type == "particles") {
		mode = Shader::MODE_PARTICLES;
	} else {
		mode = Shader::MODE_SPATIAL;
	}

	if (shader->get_mode() != mode) {
		shader->set_code(get_text_editor()->get_text());
		_load_theme_settings();
	}
}

static ShaderLanguage::DataType _get_global_variable_type(const StringName &p_variable) {
	RS::GlobalVariableType gvt = RS::get_singleton()->global_variable_get_type(p_variable);
	return RS::global_variable_type_get_shader_datatype(gvt);
}

void ShaderTextEditor::_code_complete_script(const String &p_code, List<ScriptCodeCompletionOption> *r_options) {
	_check_shader_mode();

	ShaderLanguage sl;
	String calltip;

	sl.complete(p_code, ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(shader->get_mode())), ShaderLanguage::VaryingFunctionNames(), ShaderTypes::get_singleton()->get_types(), _get_global_variable_type, r_options, calltip);

	get_text_editor()->set_code_hint(calltip);
}

void ShaderTextEditor::_validate_script() {
	_check_shader_mode();

	String code = get_text_editor()->get_text();
	//List<StringName> params;
	//shader->get_param_list(&params);

	ShaderLanguage sl;

	sl.enable_warning_checking(saved_warnings_enabled);
	sl.set_warning_flags(saved_warning_flags);

	Error err = sl.compile(code, ShaderTypes::get_singleton()->get_functions(RenderingServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_modes(RenderingServer::ShaderMode(shader->get_mode())), ShaderLanguage::VaryingFunctionNames(), ShaderTypes::get_singleton()->get_types(), _get_global_variable_type);

	if (err != OK) {
		String error_text = "error(" + itos(sl.get_error_line()) + "): " + sl.get_error_text();
		set_error(error_text);
		set_error_pos(sl.get_error_line() - 1, 0);
		for (int i = 0; i < get_text_editor()->get_line_count(); i++) {
			get_text_editor()->set_line_background_color(i, Color(0, 0, 0, 0));
		}
		get_text_editor()->set_line_background_color(sl.get_error_line() - 1, marked_line_color);
	} else {
		for (int i = 0; i < get_text_editor()->get_line_count(); i++) {
			get_text_editor()->set_line_background_color(i, Color(0, 0, 0, 0));
		}
		set_error("");
	}

	if (warnings.size() > 0 || err != OK) {
		warnings_panel->clear();
	}
	warnings.clear();
	for (List<ShaderWarning>::Element *E = sl.get_warnings_ptr(); E; E = E->next()) {
		warnings.push_back(E->get());
	}
	if (warnings.size() > 0 && err == OK) {
		warnings.sort_custom<WarningsComparator>();
		_update_warning_panel();
	} else {
		set_warning_nb(0);
	}
	emit_signal("script_changed");
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

		// First cell.
		warnings_panel->push_cell();
		warnings_panel->push_meta(w.get_line() - 1);
		warnings_panel->push_color(warnings_panel->get_theme_color("warning_color", "Editor"));
		warnings_panel->add_text(TTR("Line") + " " + itos(w.get_line()));
		warnings_panel->add_text(" (" + w.get_name() + "):");
		warnings_panel->pop(); // Color.
		warnings_panel->pop(); // Meta goto.
		warnings_panel->pop(); // Cell.

		// Second cell.
		warnings_panel->push_cell();
		warnings_panel->add_text(w.get_message());
		warnings_panel->pop(); // Cell.
	}
	warnings_panel->pop(); // Table.

	set_warning_nb(warning_count);
}

void ShaderTextEditor::_bind_methods() {
}

ShaderTextEditor::ShaderTextEditor() {
	syntax_highlighter.instance();
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

			CodeEdit *tx = shader_editor->get_text_editor();
			tx->indent_selected_lines_left();

		} break;
		case EDIT_INDENT_RIGHT: {
			if (shader.is_null()) {
				return;
			}

			CodeEdit *tx = shader_editor->get_text_editor();
			tx->indent_selected_lines_right();

		} break;
		case EDIT_DELETE_LINE: {
			shader_editor->delete_lines();
		} break;
		case EDIT_CLONE_DOWN: {
			shader_editor->clone_lines_down();
		} break;
		case EDIT_TOGGLE_COMMENT: {
			if (shader.is_null()) {
				return;
			}

			shader_editor->toggle_inline_comment("//");

		} break;
		case EDIT_COMPLETE: {
			shader_editor->get_text_editor()->query_code_comple();
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
			OS::get_singleton()->shell_open("https://docs.godotengine.org/en/latest/tutorials/shaders/shader_reference/index.html");
		} break;
	}
	if (p_option != SEARCH_FIND && p_option != SEARCH_REPLACE && p_option != SEARCH_GOTO_LINE) {
		shader_editor->get_text_editor()->call_deferred("grab_focus");
	}
}

void ShaderEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_WM_WINDOW_FOCUS_IN) {
		_check_for_external_edit();
	}
}

void ShaderEditor::_editor_settings_changed() {
	shader_editor->update_editor_settings();

	shader_editor->get_text_editor()->add_theme_constant_override("line_spacing", EditorSettings::get_singleton()->get("text_editor/theme/line_spacing"));
	shader_editor->get_text_editor()->set_draw_breakpoints_gutter(false);
	shader_editor->get_text_editor()->set_draw_executing_lines_gutter(false);
}

void ShaderEditor::_show_warnings_panel(bool p_show) {
	warnings_panel->set_visible(p_show);
}

void ShaderEditor::_warning_clicked(Variant p_line) {
	if (p_line.get_type() == Variant::INT) {
		shader_editor->get_text_editor()->cursor_set_line(p_line.operator int64_t());
	}
}

void ShaderEditor::_bind_methods() {
	ClassDB::bind_method("_show_warnings_panel", &ShaderEditor::_show_warnings_panel);
	ClassDB::bind_method("_warning_clicked", &ShaderEditor::_warning_clicked);
}

void ShaderEditor::ensure_select_current() {
	/*
	if (tab_container->get_child_count() && tab_container->get_current_tab()>=0) {
		ShaderTextEditor *ste = Object::cast_to<ShaderTextEditor>(tab_container->get_child(tab_container->get_current_tab()));
		if (!ste)
			return;
		Ref<Shader> shader = ste->get_edited_shader();
		get_scene()->get_root_node()->call("_resource_selected",shader);
	}*/
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
	if (shader.is_null() || !shader.is_valid()) {
		return;
	}

	// internal shader.
	if (shader->get_path() == "" || shader->get_path().find("local://") != -1 || shader->get_path().find("::") != -1) {
		return;
	}

	bool use_autoreload = bool(EDITOR_DEF("text_editor/files/auto_reload_scripts_on_external_change", false));
	if (shader->get_last_modified_time() != FileAccess::get_modified_time(shader->get_path())) {
		if (use_autoreload) {
			_reload_shader_from_disk();
		} else {
			disk_changed->call_deferred("popup_centered");
		}
	}
}

void ShaderEditor::_reload_shader_from_disk() {
	Ref<Shader> rel_shader = ResourceLoader::load(shader->get_path(), shader->get_class(), ResourceFormatLoader::CACHE_MODE_IGNORE);
	ERR_FAIL_COND(!rel_shader.is_valid());

	shader->set_code(rel_shader->get_code());
	shader->set_last_modified_time(rel_shader->get_last_modified_time());
	shader_editor->reload_text();
}

void ShaderEditor::edit(const Ref<Shader> &p_shader) {
	if (p_shader.is_null() || !p_shader->is_text_shader()) {
		return;
	}

	if (shader == p_shader) {
		return;
	}

	shader = p_shader;

	shader_editor->set_edited_shader(p_shader);

	//vertex_editor->set_edited_shader(shader,ShaderLanguage::SHADER_MATERIAL_VERTEX);
	// see if already has it
}

void ShaderEditor::save_external_data(const String &p_str) {
	if (shader.is_null()) {
		disk_changed->hide();
		return;
	}

	apply_shaders();
	if (shader->get_path() != "" && shader->get_path().find("local://") == -1 && shader->get_path().find("::") == -1) {
		//external shader, save it
		ResourceSaver::save(shader->get_path(), shader);
	}

	disk_changed->hide();
}

void ShaderEditor::apply_shaders() {
	if (shader.is_valid()) {
		String shader_code = shader->get_code();
		String editor_code = shader_editor->get_text_editor()->get_text();
		if (shader_code != editor_code) {
			shader->set_code(editor_code);
			shader->set_edited(true);
		}
	}
}

void ShaderEditor::_text_edit_gui_input(const Ref<InputEvent> &ev) {
	Ref<InputEventMouseButton> mb = ev;

	if (mb.is_valid()) {
		if (mb->get_button_index() == MOUSE_BUTTON_RIGHT && mb->is_pressed()) {
			int col, row;
			CodeEdit *tx = shader_editor->get_text_editor();
			tx->_get_mouse_pos(mb->get_global_position() - tx->get_global_position(), row, col);
			tx->set_right_click_moves_caret(EditorSettings::get_singleton()->get("text_editor/cursor/right_click_moves_caret"));

			if (tx->is_right_click_moving_caret()) {
				if (tx->is_selection_active()) {
					int from_line = tx->get_selection_from_line();
					int to_line = tx->get_selection_to_line();
					int from_column = tx->get_selection_from_column();
					int to_column = tx->get_selection_to_column();

					if (row < from_line || row > to_line || (row == from_line && col < from_column) || (row == to_line && col > to_column)) {
						// Right click is outside the selected text
						tx->deselect();
					}
				}
				if (!tx->is_selection_active()) {
					tx->cursor_set_line(row, true, false);
					tx->cursor_set_column(col);
				}
			}
			_make_context_menu(tx->is_selection_active(), get_local_mouse_position());
		}
	}

	Ref<InputEventKey> k = ev;
	if (k.is_valid() && k->is_pressed() && k->get_keycode() == KEY_MENU) {
		CodeEdit *tx = shader_editor->get_text_editor();
		_make_context_menu(tx->is_selection_active(), (get_global_transform().inverse() * tx->get_global_transform()).xform(tx->_get_cursor_pixel_pos()));
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
		bookmarks_menu->set_item_metadata(bookmarks_menu->get_item_count() - 1, bookmark_list[i]);
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

	context_menu->set_position(get_global_transform().xform(p_position));
	context_menu->set_size(Vector2(1, 1));
	context_menu->popup();
}

ShaderEditor::ShaderEditor(EditorNode *p_node) {
	GLOBAL_DEF("debug/shader_language/warnings/enable", true);
	GLOBAL_DEF("debug/shader_language/warnings/treat_warnings_as_errors", false);
	for (int i = 0; i < (int)ShaderWarning::WARNING_MAX; i++) {
		GLOBAL_DEF("debug/shader_language/warnings/" + ShaderWarning::get_name_from_code((ShaderWarning::Code)i).to_lower(), true);
	}
	_update_warnings(false);

	shader_editor = memnew(ShaderTextEditor);
	shader_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	shader_editor->add_theme_constant_override("separation", 0);
	shader_editor->set_anchors_and_offsets_preset(Control::PRESET_WIDE);

	shader_editor->connect("show_warnings_panel", callable_mp(this, &ShaderEditor::_show_warnings_panel));
	shader_editor->connect("script_changed", callable_mp(this, &ShaderEditor::apply_shaders));
	EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &ShaderEditor::_editor_settings_changed));
	ProjectSettingsEditor::get_singleton()->connect("confirmed", callable_mp(this, &ShaderEditor::_project_settings_changed));

	shader_editor->get_text_editor()->set_callhint_settings(
			EditorSettings::get_singleton()->get("text_editor/completion/put_callhint_tooltip_below_current_line"),
			EditorSettings::get_singleton()->get("text_editor/completion/callhint_tooltip_offset"));

	shader_editor->get_text_editor()->set_select_identifiers_on_hover(true);
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
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/clone_down"), EDIT_CLONE_DOWN);
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
	help_menu->get_popup()->add_icon_item(p_node->get_gui_base()->get_theme_icon("Instance", "EditorIcons"), TTR("Online Docs"), HELP_DOCS);
	help_menu->get_popup()->connect("id_pressed", callable_mp(this, &ShaderEditor::_menu_option));

	add_child(main_container);
	main_container->add_child(hbc);
	hbc->add_child(search_menu);
	hbc->add_child(edit_menu);
	hbc->add_child(goto_menu);
	hbc->add_child(help_menu);
	hbc->add_theme_style_override("panel", p_node->get_gui_base()->get_theme_stylebox("ScriptEditorPanel", "EditorStyles"));

	VSplitContainer *editor_box = memnew(VSplitContainer);
	main_container->add_child(editor_box);
	editor_box->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	editor_box->set_v_size_flags(SIZE_EXPAND_FILL);
	editor_box->add_child(shader_editor);

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
	dl->set_text(TTR("This shader has been modified on on disk.\nWhat action should be taken?"));
	vbc->add_child(dl);

	disk_changed->connect("confirmed", callable_mp(this, &ShaderEditor::_reload_shader_from_disk));
	disk_changed->get_ok_button()->set_text(TTR("Reload"));

	disk_changed->add_button(TTR("Resave"), !DisplayServer::get_singleton()->get_swap_cancel_ok(), "resave");
	disk_changed->connect("custom_action", callable_mp(this, &ShaderEditor::save_external_data));

	add_child(disk_changed);

	_editor_settings_changed();
}

void ShaderEditorPlugin::edit(Object *p_object) {
	Shader *s = Object::cast_to<Shader>(p_object);
	shader_editor->edit(s);
}

bool ShaderEditorPlugin::handles(Object *p_object) const {
	Shader *shader = Object::cast_to<Shader>(p_object);
	return shader != nullptr && shader->is_text_shader();
}

void ShaderEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		button->show();
		editor->make_bottom_panel_item_visible(shader_editor);

	} else {
		button->hide();
		if (shader_editor->is_visible_in_tree()) {
			editor->hide_bottom_panel();
		}
		shader_editor->apply_shaders();
	}
}

void ShaderEditorPlugin::selected_notify() {
	shader_editor->ensure_select_current();
}

void ShaderEditorPlugin::save_external_data() {
	shader_editor->save_external_data();
}

void ShaderEditorPlugin::apply_changes() {
	shader_editor->apply_shaders();
}

ShaderEditorPlugin::ShaderEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	shader_editor = memnew(ShaderEditor(p_node));

	shader_editor->set_custom_minimum_size(Size2(0, 300) * EDSCALE);
	button = editor->add_bottom_panel_item(TTR("Shader"), shader_editor);
	button->hide();

	_2d = false;
}

ShaderEditorPlugin::~ShaderEditorPlugin() {
}

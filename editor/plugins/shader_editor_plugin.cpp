/*************************************************************************/
/*  shader_editor_plugin.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor/editor_settings.h"
#include "editor/property_editor.h"
#include "scene/resources/shader_graph.h"
#include "servers/visual/shader_types.h"

/*** SHADER SCRIPT EDITOR ****/

Ref<Shader> ShaderTextEditor::get_edited_shader() const {

	return shader;
}
void ShaderTextEditor::set_edited_shader(const Ref<Shader> &p_shader) {

	shader = p_shader;

	_load_theme_settings();

	get_text_edit()->set_text(p_shader->get_code());

	_line_col_changed();
}

void ShaderTextEditor::_load_theme_settings() {

	get_text_edit()->clear_colors();

	Color background_color = EDITOR_DEF("text_editor/highlighting/background_color", Color(0, 0, 0, 0));
	Color completion_background_color = EDITOR_DEF("text_editor/highlighting/completion_background_color", Color(0, 0, 0, 0));
	Color completion_selected_color = EDITOR_DEF("text_editor/highlighting/completion_selected_color", Color::html("434244"));
	Color completion_existing_color = EDITOR_DEF("text_editor/highlighting/completion_existing_color", Color::html("21dfdfdf"));
	Color completion_scroll_color = EDITOR_DEF("text_editor/highlighting/completion_scroll_color", Color::html("ffffff"));
	Color completion_font_color = EDITOR_DEF("text_editor/highlighting/completion_font_color", Color::html("aaaaaa"));
	Color text_color = EDITOR_DEF("text_editor/highlighting/text_color", Color(0, 0, 0));
	Color line_number_color = EDITOR_DEF("text_editor/highlighting/line_number_color", Color(0, 0, 0));
	Color caret_color = EDITOR_DEF("text_editor/highlighting/caret_color", Color(0, 0, 0));
	Color caret_background_color = EDITOR_DEF("text_editor/highlighting/caret_background_color", Color(0, 0, 0));
	Color text_selected_color = EDITOR_DEF("text_editor/highlighting/text_selected_color", Color(1, 1, 1));
	Color selection_color = EDITOR_DEF("text_editor/highlighting/selection_color", Color(0.2, 0.2, 1));
	Color brace_mismatch_color = EDITOR_DEF("text_editor/highlighting/brace_mismatch_color", Color(1, 0.2, 0.2));
	Color current_line_color = EDITOR_DEF("text_editor/highlighting/current_line_color", Color(0.3, 0.5, 0.8, 0.15));
	Color line_length_guideline_color = EDITOR_DEF("text_editor/highlighting/line_length_guideline_color", Color(0, 0, 0));
	Color word_highlighted_color = EDITOR_DEF("text_editor/highlighting/word_highlighted_color", Color(0.8, 0.9, 0.9, 0.15));
	Color number_color = EDITOR_DEF("text_editor/highlighting/number_color", Color(0.9, 0.6, 0.0, 2));
	Color function_color = EDITOR_DEF("text_editor/highlighting/function_color", Color(0.4, 0.6, 0.8));
	Color member_variable_color = EDITOR_DEF("text_editor/highlighting/member_variable_color", Color(0.9, 0.3, 0.3));
	Color mark_color = EDITOR_DEF("text_editor/highlighting/mark_color", Color(1.0, 0.4, 0.4, 0.4));
	Color breakpoint_color = EDITOR_DEF("text_editor/highlighting/breakpoint_color", Color(0.8, 0.8, 0.4, 0.2));
	Color code_folding_color = EDITOR_DEF("text_editor/highlighting/code_folding_color", Color(0.8, 0.8, 0.8, 0.8));
	Color search_result_color = EDITOR_DEF("text_editor/highlighting/search_result_color", Color(0.05, 0.25, 0.05, 1));
	Color search_result_border_color = EDITOR_DEF("text_editor/highlighting/search_result_border_color", Color(0.1, 0.45, 0.1, 1));
	Color symbol_color = EDITOR_DEF("text_editor/highlighting/symbol_color", Color::hex(0x005291ff));

	Color keyword_color = EDITOR_DEF("text_editor/highlighting/keyword_color", Color(0.5, 0.0, 0.2));
	Color basetype_color = EDITOR_DEF("text_editor/highlighting/base_type_color", Color(0.3, 0.3, 0.0));
	Color type_color = EDITOR_DEF("text_editor/highlighting/engine_type_color", Color(0.0, 0.2, 0.4));
	Color comment_color = EDITOR_DEF("text_editor/highlighting/comment_color", Color::hex(0x797e7eff));
	Color string_color = EDITOR_DEF("text_editor/highlighting/string_color", Color::hex(0x6b6f00ff));

	// Adapt
	if (EditorSettings::get_singleton()->get("text_editor/theme/color_theme") == "Adaptive") {
		Ref<Theme> tm = EditorNode::get_singleton()->get_theme_base()->get_theme();

		symbol_color = tm->get_color("text_editor/theme/symbol_color", "Editor");
		keyword_color = tm->get_color("text_editor/theme/keyword_color", "Editor");
		basetype_color = tm->get_color("text_editor/theme/basetype_color", "Editor");
		type_color = tm->get_color("text_editor/theme/type_color", "Editor");
		comment_color = tm->get_color("text_editor/theme/comment_color", "Editor");
		string_color = tm->get_color("text_editor/theme/string_color", "Editor");
		background_color = tm->get_color("text_editor/theme/background_color", "Editor");
		completion_background_color = tm->get_color("text_editor/theme/completion_background_color", "Editor");
		completion_selected_color = tm->get_color("text_editor/theme/completion_selected_color", "Editor");
		completion_existing_color = tm->get_color("text_editor/theme/completion_existing_color", "Editor");
		completion_scroll_color = tm->get_color("text_editor/theme/completion_scroll_color", "Editor");
		completion_font_color = tm->get_color("text_editor/theme/completion_font_color", "Editor");
		text_color = tm->get_color("text_editor/theme/text_color", "Editor");
		line_number_color = tm->get_color("text_editor/theme/line_number_color", "Editor");
		caret_color = tm->get_color("text_editor/theme/caret_color", "Editor");
		caret_background_color = tm->get_color("text_editor/theme/caret_background_color", "Editor");
		text_selected_color = tm->get_color("text_editor/theme/text_selected_color", "Editor");
		selection_color = tm->get_color("text_editor/theme/selection_color", "Editor");
		brace_mismatch_color = tm->get_color("text_editor/theme/brace_mismatch_color", "Editor");
		current_line_color = tm->get_color("text_editor/theme/current_line_color", "Editor");
		line_length_guideline_color = tm->get_color("text_editor/theme/line_length_guideline_color", "Editor");
		word_highlighted_color = tm->get_color("text_editor/theme/word_highlighted_color", "Editor");
		number_color = tm->get_color("text_editor/theme/number_color", "Editor");
		function_color = tm->get_color("text_editor/theme/function_color", "Editor");
		member_variable_color = tm->get_color("text_editor/theme/member_variable_color", "Editor");
		mark_color = tm->get_color("text_editor/theme/mark_color", "Editor");
		breakpoint_color = tm->get_color("text_editor/theme/breakpoint_color", "Editor");
		code_folding_color = tm->get_color("text_editor/theme/code_folding_color", "Editor");
		search_result_color = tm->get_color("text_editor/theme/search_result_color", "Editor");
		search_result_border_color = tm->get_color("text_editor/theme/search_result_border_color", "Editor");
	}

	get_text_edit()->add_color_override("background_color", background_color);
	get_text_edit()->add_color_override("completion_background_color", completion_background_color);
	get_text_edit()->add_color_override("completion_selected_color", completion_selected_color);
	get_text_edit()->add_color_override("completion_existing_color", completion_existing_color);
	get_text_edit()->add_color_override("completion_scroll_color", completion_scroll_color);
	get_text_edit()->add_color_override("completion_font_color", completion_font_color);
	get_text_edit()->add_color_override("font_color", text_color);
	get_text_edit()->add_color_override("line_number_color", line_number_color);
	get_text_edit()->add_color_override("caret_color", caret_color);
	get_text_edit()->add_color_override("caret_background_color", caret_background_color);
	get_text_edit()->add_color_override("font_selected_color", text_selected_color);
	get_text_edit()->add_color_override("selection_color", selection_color);
	get_text_edit()->add_color_override("brace_mismatch_color", brace_mismatch_color);
	get_text_edit()->add_color_override("current_line_color", current_line_color);
	get_text_edit()->add_color_override("line_length_guideline_color", line_length_guideline_color);
	get_text_edit()->add_color_override("word_highlighted_color", word_highlighted_color);
	get_text_edit()->add_color_override("number_color", number_color);
	get_text_edit()->add_color_override("function_color", function_color);
	get_text_edit()->add_color_override("member_variable_color", member_variable_color);
	get_text_edit()->add_color_override("mark_color", mark_color);
	get_text_edit()->add_color_override("breakpoint_color", breakpoint_color);
	get_text_edit()->add_color_override("code_folding_color", code_folding_color);
	get_text_edit()->add_color_override("search_result_color", search_result_color);
	get_text_edit()->add_color_override("search_result_border_color", search_result_border_color);
	get_text_edit()->add_color_override("symbol_color", symbol_color);

	List<String> keywords;
	ShaderLanguage::get_keyword_list(&keywords);

	if (shader.is_valid()) {

		for (const Map<StringName, ShaderLanguage::FunctionInfo>::Element *E = ShaderTypes::get_singleton()->get_functions(VisualServer::ShaderMode(shader->get_mode())).front(); E; E = E->next()) {

			for (const Map<StringName, ShaderLanguage::BuiltInInfo>::Element *F = E->get().built_ins.front(); F; F = F->next()) {
				keywords.push_back(F->key());
			}
		}

		for (const Set<String>::Element *E = ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader->get_mode())).front(); E; E = E->next()) {

			keywords.push_back(E->get());
		}
	}

	for (List<String>::Element *E = keywords.front(); E; E = E->next()) {

		get_text_edit()->add_keyword_color(E->get(), keyword_color);
	}

	//colorize core types
	//Color basetype_color= EDITOR_DEF("text_editor/base_type_color",Color(0.3,0.3,0.0));

	//colorize comments
	get_text_edit()->add_color_region("/*", "*/", comment_color, false);
	get_text_edit()->add_color_region("//", "", comment_color, false);

	/*//colorize strings
	Color string_color = EDITOR_DEF("text_editor/string_color",Color::hex(0x6b6f00ff));

	List<String> strings;
	shader->get_shader_mode()->get_string_delimiters(&strings);

	for (List<String>::Element *E=strings.front();E;E=E->next()) {

		String string = E->get();
		String beg = string.get_slice(" ",0);
		String end = string.get_slice_count(" ")>1?string.get_slice(" ",1):String();
		get_text_edit()->add_color_region(beg,end,string_color,end=="");
	}*/
}

void ShaderTextEditor::_check_shader_mode() {

	String type = ShaderLanguage::get_shader_type(get_text_edit()->get_text());

	print_line("type is: " + type);
	Shader::Mode mode;

	if (type == "canvas_item") {
		mode = Shader::MODE_CANVAS_ITEM;
	} else if (type == "particles") {
		mode = Shader::MODE_PARTICLES;
	} else {
		mode = Shader::MODE_SPATIAL;
	}

	if (shader->get_mode() != mode) {
		shader->set_code(get_text_edit()->get_text());
		_load_theme_settings();
	}
}

void ShaderTextEditor::_code_complete_script(const String &p_code, List<String> *r_options) {

	_check_shader_mode();

	ShaderLanguage sl;
	String calltip;

	Error err = sl.complete(p_code, ShaderTypes::get_singleton()->get_functions(VisualServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_types(), r_options, calltip);
	if (err != OK)
		ERR_PRINT("Shaderlang complete failed");

	if (calltip != "") {
		get_text_edit()->set_code_hint(calltip);
	}
}

void ShaderTextEditor::_validate_script() {

	_check_shader_mode();

	String code = get_text_edit()->get_text();
	//List<StringName> params;
	//shader->get_param_list(&params);

	ShaderLanguage sl;

	Error err = sl.compile(code, ShaderTypes::get_singleton()->get_functions(VisualServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_types());

	if (err != OK) {
		String error_text = "error(" + itos(sl.get_error_line()) + "): " + sl.get_error_text();
		set_error(error_text);
		for (int i = 0; i < get_text_edit()->get_line_count(); i++)
			get_text_edit()->set_line_as_marked(i, false);
		get_text_edit()->set_line_as_marked(sl.get_error_line() - 1, true);

	} else {
		for (int i = 0; i < get_text_edit()->get_line_count(); i++)
			get_text_edit()->set_line_as_marked(i, false);
		set_error("");
	}

	emit_signal("script_changed");
}

void ShaderTextEditor::_bind_methods() {
}

ShaderTextEditor::ShaderTextEditor() {
}

/*** SCRIPT EDITOR ******/

void ShaderEditor::_menu_option(int p_option) {

	switch (p_option) {
		case EDIT_UNDO: {
			shader_editor->get_text_edit()->undo();
		} break;
		case EDIT_REDO: {
			shader_editor->get_text_edit()->redo();
		} break;
		case EDIT_CUT: {
			shader_editor->get_text_edit()->cut();
		} break;
		case EDIT_COPY: {
			shader_editor->get_text_edit()->copy();
		} break;
		case EDIT_PASTE: {
			shader_editor->get_text_edit()->paste();
		} break;
		case EDIT_SELECT_ALL: {
			shader_editor->get_text_edit()->select_all();
		} break;
		case EDIT_MOVE_LINE_UP: {

			TextEdit *tx = shader_editor->get_text_edit();
			if (shader.is_null())
				return;

			tx->begin_complex_operation();
			if (tx->is_selection_active()) {
				int from_line = tx->get_selection_from_line();
				int from_col = tx->get_selection_from_column();
				int to_line = tx->get_selection_to_line();
				int to_column = tx->get_selection_to_column();

				for (int i = from_line; i <= to_line; i++) {
					int line_id = i;
					int next_id = i - 1;

					if (line_id == 0 || next_id < 0)
						return;

					tx->swap_lines(line_id, next_id);
					tx->cursor_set_line(next_id);
				}
				int from_line_up = from_line > 0 ? from_line - 1 : from_line;
				int to_line_up = to_line > 0 ? to_line - 1 : to_line;
				tx->select(from_line_up, from_col, to_line_up, to_column);
			} else {
				int line_id = tx->cursor_get_line();
				int next_id = line_id - 1;

				if (line_id == 0 || next_id < 0)
					return;

				tx->swap_lines(line_id, next_id);
				tx->cursor_set_line(next_id);
			}
			tx->end_complex_operation();
			tx->update();

		} break;
		case EDIT_MOVE_LINE_DOWN: {

			TextEdit *tx = shader_editor->get_text_edit();
			if (shader.is_null())
				return;

			tx->begin_complex_operation();
			if (tx->is_selection_active()) {
				int from_line = tx->get_selection_from_line();
				int from_col = tx->get_selection_from_column();
				int to_line = tx->get_selection_to_line();
				int to_column = tx->get_selection_to_column();

				for (int i = to_line; i >= from_line; i--) {
					int line_id = i;
					int next_id = i + 1;

					if (line_id == tx->get_line_count() - 1 || next_id > tx->get_line_count())
						return;

					tx->swap_lines(line_id, next_id);
					tx->cursor_set_line(next_id);
				}
				int from_line_down = from_line < tx->get_line_count() ? from_line + 1 : from_line;
				int to_line_down = to_line < tx->get_line_count() ? to_line + 1 : to_line;
				tx->select(from_line_down, from_col, to_line_down, to_column);
			} else {
				int line_id = tx->cursor_get_line();
				int next_id = line_id + 1;

				if (line_id == tx->get_line_count() - 1 || next_id > tx->get_line_count())
					return;

				tx->swap_lines(line_id, next_id);
				tx->cursor_set_line(next_id);
			}
			tx->end_complex_operation();
			tx->update();

		} break;
		case EDIT_INDENT_LEFT: {

			TextEdit *tx = shader_editor->get_text_edit();
			if (shader.is_null())
				return;

			tx->indent_left();

		} break;
		case EDIT_INDENT_RIGHT: {

			TextEdit *tx = shader_editor->get_text_edit();
			if (shader.is_null())
				return;

			tx->indent_right();

		} break;
		case EDIT_DELETE_LINE: {

			TextEdit *tx = shader_editor->get_text_edit();
			if (shader.is_null())
				return;

			tx->begin_complex_operation();
			int line = tx->cursor_get_line();
			tx->set_line(tx->cursor_get_line(), "");
			tx->backspace_at_cursor();
			tx->cursor_set_line(line);
			tx->end_complex_operation();

		} break;
		case EDIT_CLONE_DOWN: {

			TextEdit *tx = shader_editor->get_text_edit();
			if (shader.is_null())
				return;

			int from_line = tx->cursor_get_line();
			int to_line = tx->cursor_get_line();
			int column = tx->cursor_get_column();

			if (tx->is_selection_active()) {
				from_line = tx->get_selection_from_line();
				to_line = tx->get_selection_to_line();
				column = tx->cursor_get_column();
			}
			int next_line = to_line + 1;

			tx->begin_complex_operation();
			for (int i = from_line; i <= to_line; i++) {

				if (i >= tx->get_line_count() - 1) {
					tx->set_line(i, tx->get_line(i) + "\n");
				}
				String line_clone = tx->get_line(i);
				tx->insert_at(line_clone, next_line);
				next_line++;
			}

			tx->cursor_set_column(column);
			if (tx->is_selection_active()) {
				tx->select(to_line + 1, tx->get_selection_from_column(), next_line - 1, tx->get_selection_to_column());
			}

			tx->end_complex_operation();
			tx->update();

		} break;
		case EDIT_TOGGLE_COMMENT: {

			TextEdit *tx = shader_editor->get_text_edit();
			if (shader.is_null())
				return;

			tx->begin_complex_operation();
			if (tx->is_selection_active()) {
				int begin = tx->get_selection_from_line();
				int end = tx->get_selection_to_line();

				// End of selection ends on the first column of the last line, ignore it.
				if (tx->get_selection_to_column() == 0)
					end -= 1;

				// Check if all lines in the selected block are commented
				bool is_commented = true;
				for (int i = begin; i <= end; i++) {
					if (!tx->get_line(i).begins_with("//")) {
						is_commented = false;
						break;
					}
				}
				for (int i = begin; i <= end; i++) {
					String line_text = tx->get_line(i);

					if (line_text.strip_edges().empty()) {
						line_text = "//";
					} else {
						if (is_commented) {
							line_text = line_text.substr(2, line_text.length());
						} else {
							line_text = "//" + line_text;
						}
					}
					tx->set_line(i, line_text);
				}
			} else {
				int begin = tx->cursor_get_line();
				String line_text = tx->get_line(begin);

				if (line_text.begins_with("//"))
					line_text = line_text.substr(2, line_text.length());
				else
					line_text = "//" + line_text;
				tx->set_line(begin, line_text);
			}
			tx->end_complex_operation();
			tx->update();
			//tx->deselect();

		} break;
		case EDIT_COMPLETE: {

			shader_editor->get_text_edit()->query_code_comple();
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

			goto_line_dialog->popup_find_line(shader_editor->get_text_edit());
		} break;
	}
	if (p_option != SEARCH_FIND && p_option != SEARCH_REPLACE && p_option != SEARCH_GOTO_LINE) {
		shader_editor->get_text_edit()->call_deferred("grab_focus");
	}
}

void ShaderEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {
	}
	if (p_what == NOTIFICATION_DRAW) {
	}
}

void ShaderEditor::_params_changed() {

	shader_editor->_validate_script();
}

void ShaderEditor::_editor_settings_changed() {

	shader_editor->get_text_edit()->set_auto_brace_completion(EditorSettings::get_singleton()->get("text_editor/completion/auto_brace_complete"));
	shader_editor->get_text_edit()->set_scroll_pass_end_of_file(EditorSettings::get_singleton()->get("text_editor/cursor/scroll_past_end_of_file"));
	shader_editor->get_text_edit()->set_indent_size(EditorSettings::get_singleton()->get("text_editor/indent/size"));
	shader_editor->get_text_edit()->set_indent_using_spaces(EditorSettings::get_singleton()->get("text_editor/indent/type"));
	shader_editor->get_text_edit()->set_auto_indent(EditorSettings::get_singleton()->get("text_editor/indent/auto_indent"));
	shader_editor->get_text_edit()->set_draw_tabs(EditorSettings::get_singleton()->get("text_editor/indent/draw_tabs"));
	shader_editor->get_text_edit()->set_show_line_numbers(EditorSettings::get_singleton()->get("text_editor/line_numbers/show_line_numbers"));
	shader_editor->get_text_edit()->set_syntax_coloring(EditorSettings::get_singleton()->get("text_editor/highlighting/syntax_highlighting"));
	shader_editor->get_text_edit()->set_highlight_all_occurrences(EditorSettings::get_singleton()->get("text_editor/highlighting/highlight_all_occurrences"));
	shader_editor->get_text_edit()->set_highlight_current_line(EditorSettings::get_singleton()->get("text_editor/highlighting/highlight_current_line"));
	shader_editor->get_text_edit()->cursor_set_blink_enabled(EditorSettings::get_singleton()->get("text_editor/cursor/caret_blink"));
	shader_editor->get_text_edit()->cursor_set_blink_speed(EditorSettings::get_singleton()->get("text_editor/cursor/caret_blink_speed"));
	shader_editor->get_text_edit()->add_constant_override("line_spacing", EditorSettings::get_singleton()->get("text_editor/theme/line_spacing"));
	shader_editor->get_text_edit()->cursor_set_block_mode(EditorSettings::get_singleton()->get("text_editor/cursor/block_caret"));
	shader_editor->get_text_edit()->set_smooth_scroll_enabled(EditorSettings::get_singleton()->get("text_editor/open_scripts/smooth_scrolling"));
	shader_editor->get_text_edit()->set_v_scroll_speed(EditorSettings::get_singleton()->get("text_editor/open_scripts/v_scroll_speed"));
}

void ShaderEditor::_bind_methods() {

	ClassDB::bind_method("_editor_settings_changed", &ShaderEditor::_editor_settings_changed);
	ClassDB::bind_method("_text_edit_gui_input", &ShaderEditor::_text_edit_gui_input);

	ClassDB::bind_method("_menu_option", &ShaderEditor::_menu_option);
	ClassDB::bind_method("_params_changed", &ShaderEditor::_params_changed);
	ClassDB::bind_method("apply_shaders", &ShaderEditor::apply_shaders);
	//ClassDB::bind_method("_close_current_tab",&ShaderEditor::_close_current_tab);
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

void ShaderEditor::edit(const Ref<Shader> &p_shader) {

	if (p_shader.is_null())
		return;

	shader = p_shader;

	shader_editor->set_edited_shader(p_shader);

	//vertex_editor->set_edited_shader(shader,ShaderLanguage::SHADER_MATERIAL_VERTEX);
	// see if already has it
}

void ShaderEditor::save_external_data() {

	if (shader.is_null())
		return;
	apply_shaders();

	if (shader->get_path() != "" && shader->get_path().find("local://") == -1 && shader->get_path().find("::") == -1) {
		//external shader, save it
		ResourceSaver::save(shader->get_path(), shader);
	}
}

void ShaderEditor::apply_shaders() {

	if (shader.is_valid()) {
		shader->set_code(shader_editor->get_text_edit()->get_text());
		shader->set_edited(true);
	}
}

void ShaderEditor::_text_edit_gui_input(const Ref<InputEvent> &ev) {

	Ref<InputEventMouseButton> mb = ev;

	if (mb.is_valid()) {

		if (mb->get_button_index() == BUTTON_RIGHT && !mb->is_pressed()) {

			int col, row;
			TextEdit *tx = shader_editor->get_text_edit();
			tx->_get_mouse_pos(mb->get_global_position() - tx->get_global_position(), row, col);
			Vector2 mpos = mb->get_global_position() - tx->get_global_position();
			bool have_selection = (tx->get_selection_text().length() > 0);
			_make_context_menu(have_selection);
		}
	}
}

void ShaderEditor::_make_context_menu(bool p_selection) {

	context_menu->clear();
	if (p_selection) {
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/cut"), EDIT_CUT);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/copy"), EDIT_COPY);
	}

	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/paste"), EDIT_PASTE);
	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/select_all"), EDIT_SELECT_ALL);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/undo"), EDIT_UNDO);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/redo"), EDIT_REDO);

	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_left"), EDIT_INDENT_LEFT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_right"), EDIT_INDENT_RIGHT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);

	context_menu->set_position(get_global_transform().xform(get_local_mouse_position()));
	context_menu->set_size(Vector2(1, 1));
	context_menu->popup();
}

ShaderEditor::ShaderEditor(EditorNode *p_node) {

	shader_editor = memnew(ShaderTextEditor);
	shader_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	shader_editor->add_constant_override("separation", 0);
	shader_editor->set_anchors_and_margins_preset(Control::PRESET_WIDE);

	shader_editor->connect("script_changed", this, "apply_shaders");
	EditorSettings::get_singleton()->connect("settings_changed", this, "_editor_settings_changed");

	shader_editor->get_text_edit()->set_callhint_settings(
			EditorSettings::get_singleton()->get("text_editor/completion/put_callhint_tooltip_below_current_line"),
			EditorSettings::get_singleton()->get("text_editor/completion/callhint_tooltip_offset"));

	shader_editor->get_text_edit()->set_select_identifiers_on_hover(true);
	shader_editor->get_text_edit()->set_context_menu_enabled(false);
	shader_editor->get_text_edit()->connect("gui_input", this, "_text_edit_gui_input");

	shader_editor->update_editor_settings();

	context_menu = memnew(PopupMenu);
	add_child(context_menu);
	context_menu->connect("id_pressed", this, "_menu_option");

	VBoxContainer *main_container = memnew(VBoxContainer);
	HBoxContainer *hbc = memnew(HBoxContainer);

	edit_menu = memnew(MenuButton);
	//edit_menu->set_position(Point2(5, -1));
	edit_menu->set_text(TTR("Edit"));

	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/undo"), EDIT_UNDO);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/redo"), EDIT_REDO);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/cut"), EDIT_CUT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/copy"), EDIT_COPY);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/paste"), EDIT_PASTE);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/select_all"), EDIT_SELECT_ALL);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_up"), EDIT_MOVE_LINE_UP);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_down"), EDIT_MOVE_LINE_DOWN);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_left"), EDIT_INDENT_LEFT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_right"), EDIT_INDENT_RIGHT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/delete_line"), EDIT_DELETE_LINE);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/clone_down"), EDIT_CLONE_DOWN);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/complete_symbol"), EDIT_COMPLETE);

	edit_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	search_menu = memnew(MenuButton);
	//search_menu->set_position(Point2(38, -1));
	search_menu->set_text(TTR("Search"));

	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find"), SEARCH_FIND);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_next"), SEARCH_FIND_NEXT);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_previous"), SEARCH_FIND_PREV);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/replace"), SEARCH_REPLACE);
	search_menu->get_popup()->add_separator();
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_line"), SEARCH_GOTO_LINE);
	search_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	add_child(main_container);
	main_container->add_child(hbc);
	hbc->add_child(search_menu);
	hbc->add_child(edit_menu);
	hbc->add_style_override("panel", p_node->get_gui_base()->get_stylebox("ScriptEditorPanel", "EditorStyles"));
	main_container->add_child(shader_editor);

	goto_line_dialog = memnew(GotoLineDialog);
	add_child(goto_line_dialog);

	_editor_settings_changed();
}

void ShaderEditorPlugin::edit(Object *p_object) {

	Shader *s = Object::cast_to<Shader>(p_object);
	shader_editor->edit(s);
}

bool ShaderEditorPlugin::handles(Object *p_object) const {

	Shader *shader = Object::cast_to<Shader>(p_object);
	return shader != NULL;
}

void ShaderEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		button->show();
		editor->make_bottom_panel_item_visible(shader_editor);

	} else {

		button->hide();
		if (shader_editor->is_visible_in_tree())
			editor->hide_bottom_panel();
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

	shader_editor->set_custom_minimum_size(Size2(0, 300));
	button = editor->add_bottom_panel_item(TTR("Shader"), shader_editor);
	button->hide();
}

ShaderEditorPlugin::~ShaderEditorPlugin() {
}

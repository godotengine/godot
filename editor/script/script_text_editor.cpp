/**************************************************************************/
/*  script_text_editor.cpp                                                */
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

#include "script_text_editor.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/json.h"
#include "core/math/expression.h"
#include "core/os/keyboard.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/doc/editor_help.h"
#include "editor/docks/filesystem_dock.h"
#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_toaster.h"
#include "editor/inspector/editor_context_menu_plugin.h"
#include "editor/inspector/editor_inspector.h"
#include "editor/inspector/multi_node_edit.h"
#include "editor/settings/editor_command_palette.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/split_container.h"

void ConnectionInfoDialog::ok_pressed() {
}

void ConnectionInfoDialog::popup_connections(const String &p_method, const Vector<Node *> &p_nodes) {
	method->set_text(p_method);

	tree->clear();
	TreeItem *root = tree->create_item();

	for (int i = 0; i < p_nodes.size(); i++) {
		List<Connection> all_connections;
		p_nodes[i]->get_signals_connected_to_this(&all_connections);

		for (const Connection &connection : all_connections) {
			if (connection.callable.get_method() != p_method) {
				continue;
			}

			TreeItem *node_item = tree->create_item(root);

			node_item->set_text(0, Object::cast_to<Node>(connection.signal.get_object())->get_name());
			node_item->set_icon(0, EditorNode::get_singleton()->get_object_icon(connection.signal.get_object()));
			node_item->set_selectable(0, false);
			node_item->set_editable(0, false);

			node_item->set_text(1, connection.signal.get_name());
			Control *p = Object::cast_to<Control>(get_parent());
			node_item->set_icon(1, p->get_editor_theme_icon(SNAME("Slot")));
			node_item->set_selectable(1, false);
			node_item->set_editable(1, false);

			node_item->set_text(2, Object::cast_to<Node>(connection.callable.get_object())->get_name());
			node_item->set_icon(2, EditorNode::get_singleton()->get_object_icon(connection.callable.get_object()));
			node_item->set_selectable(2, false);
			node_item->set_editable(2, false);
		}
	}

	popup_centered(Size2(600, 300) * EDSCALE);
}

ConnectionInfoDialog::ConnectionInfoDialog() {
	set_title(TTRC("Connections to method:"));

	VBoxContainer *vbc = memnew(VBoxContainer);
	vbc->set_anchor_and_offset(SIDE_LEFT, Control::ANCHOR_BEGIN, 8 * EDSCALE);
	vbc->set_anchor_and_offset(SIDE_TOP, Control::ANCHOR_BEGIN, 8 * EDSCALE);
	vbc->set_anchor_and_offset(SIDE_RIGHT, Control::ANCHOR_END, -8 * EDSCALE);
	vbc->set_anchor_and_offset(SIDE_BOTTOM, Control::ANCHOR_END, -8 * EDSCALE);
	add_child(vbc);

	method = memnew(Label);
	method->set_focus_mode(Control::FOCUS_ACCESSIBILITY);
	method->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	method->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	vbc->add_child(method);

	tree = memnew(Tree);
	tree->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	tree->set_columns(3);
	tree->set_hide_root(true);
	tree->set_column_titles_visible(true);
	tree->set_column_title(0, TTRC("Source"));
	tree->set_column_title(1, TTRC("Signal"));
	tree->set_column_title(2, TTRC("Target"));
	vbc->add_child(tree);
	tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tree->set_allow_rmb_select(true);
}

////////////////////////////////////////////////////////////////////////////////

Vector<String> ScriptTextEditor::get_functions() {
	CodeEdit *te = code_editor->get_text_editor();
	String text = te->get_text();
	List<String> fnc;

	if (script->get_language()->validate(text, script->get_path(), &fnc)) {
		//if valid rewrite functions to latest
		functions.clear();
		for (const String &E : fnc) {
			functions.push_back(E);
		}
	}

	return functions;
}

void ScriptTextEditor::apply_code() {
	if (script.is_null()) {
		return;
	}
	script->set_source_code(code_editor->get_text_editor()->get_text());
	script->update_exports();
	if (!pending_dragged_exports.is_empty()) {
		_assign_dragged_export_variables();
	}

	code_editor->get_text_editor()->get_syntax_highlighter()->update_cache();
}

Ref<Resource> ScriptTextEditor::get_edited_resource() const {
	return script;
}

void ScriptTextEditor::set_edited_resource(const Ref<Resource> &p_res) {
	ERR_FAIL_COND(script.is_valid());
	ERR_FAIL_COND(p_res.is_null());

	script = p_res;

	code_editor->get_text_editor()->set_text(script->get_source_code());
	code_editor->get_text_editor()->clear_undo_history();
	code_editor->get_text_editor()->tag_saved_version();

	emit_signal(SNAME("name_changed"));
	code_editor->update_line_and_column();
}

void ScriptTextEditor::enable_editor(Control *p_shortcut_context) {
	if (editor_enabled) {
		return;
	}

	editor_enabled = true;

	_enable_code_editor();

	_validate_script();

	if (pending_state != Variant()) {
		code_editor->set_edit_state(pending_state);
		pending_state = Variant();
	}

	if (p_shortcut_context) {
		for (int i = 0; i < edit_hb->get_child_count(); ++i) {
			Control *c = cast_to<Control>(edit_hb->get_child(i));
			if (c) {
				c->set_shortcut_context(p_shortcut_context);
			}
		}
	}
}

void ScriptTextEditor::_load_theme_settings() {
	CodeEdit *text_edit = code_editor->get_text_editor();

	Color updated_warning_line_color = EDITOR_GET("text_editor/theme/highlighting/warning_color");
	Color updated_marked_line_color = EDITOR_GET("text_editor/theme/highlighting/mark_color");
	Color updated_safe_line_number_color = EDITOR_GET("text_editor/theme/highlighting/safe_line_number_color");
	Color updated_folded_code_region_color = EDITOR_GET("text_editor/theme/highlighting/folded_code_region_color");

	bool warning_line_color_updated = updated_warning_line_color != warning_line_color;
	bool marked_line_color_updated = updated_marked_line_color != marked_line_color;
	bool safe_line_number_color_updated = updated_safe_line_number_color != safe_line_number_color;
	bool folded_code_region_color_updated = updated_folded_code_region_color != folded_code_region_color;
	if (safe_line_number_color_updated || warning_line_color_updated || marked_line_color_updated || folded_code_region_color_updated) {
		safe_line_number_color = updated_safe_line_number_color;
		for (int i = 0; i < text_edit->get_line_count(); i++) {
			if (warning_line_color_updated && text_edit->get_line_background_color(i) == warning_line_color) {
				text_edit->set_line_background_color(i, updated_warning_line_color);
			}

			if (marked_line_color_updated && text_edit->get_line_background_color(i) == marked_line_color) {
				text_edit->set_line_background_color(i, updated_marked_line_color);
			}

			if (safe_line_number_color_updated && text_edit->get_line_gutter_item_color(i, line_number_gutter) != default_line_number_color) {
				text_edit->set_line_gutter_item_color(i, line_number_gutter, safe_line_number_color);
			}

			if (folded_code_region_color_updated && text_edit->get_line_background_color(i) == folded_code_region_color) {
				text_edit->set_line_background_color(i, updated_folded_code_region_color);
			}
		}
		warning_line_color = updated_warning_line_color;
		marked_line_color = updated_marked_line_color;
		folded_code_region_color = updated_folded_code_region_color;
	}

	theme_loaded = true;
	if (script.is_valid()) {
		_set_theme_for_script();
	}
}

void ScriptTextEditor::_set_theme_for_script() {
	if (!theme_loaded) {
		return;
	}

	CodeEdit *text_edit = code_editor->get_text_editor();
	text_edit->get_syntax_highlighter()->update_cache();

	Vector<String> strings = script->get_language()->get_string_delimiters();
	text_edit->clear_string_delimiters();
	for (const String &string : strings) {
		String beg = string.get_slicec(' ', 0);
		String end = string.get_slice_count(" ") > 1 ? string.get_slicec(' ', 1) : String();
		if (!text_edit->has_string_delimiter(beg)) {
			text_edit->add_string_delimiter(beg, end, end.is_empty());
		}

		if (!end.is_empty() && !text_edit->has_auto_brace_completion_open_key(beg)) {
			text_edit->add_auto_brace_completion_pair(beg, end);
		}
	}

	text_edit->clear_comment_delimiters();

	for (const String &comment : script->get_language()->get_comment_delimiters()) {
		String beg = comment.get_slicec(' ', 0);
		String end = comment.get_slice_count(" ") > 1 ? comment.get_slicec(' ', 1) : String();
		text_edit->add_comment_delimiter(beg, end, end.is_empty());

		if (!end.is_empty() && !text_edit->has_auto_brace_completion_open_key(beg)) {
			text_edit->add_auto_brace_completion_pair(beg, end);
		}
	}

	for (const String &doc_comment : script->get_language()->get_doc_comment_delimiters()) {
		String beg = doc_comment.get_slicec(' ', 0);
		String end = doc_comment.get_slice_count(" ") > 1 ? doc_comment.get_slicec(' ', 1) : String();
		text_edit->add_comment_delimiter(beg, end, end.is_empty());

		if (!end.is_empty() && !text_edit->has_auto_brace_completion_open_key(beg)) {
			text_edit->add_auto_brace_completion_pair(beg, end);
		}
	}
}

void ScriptTextEditor::_show_errors_panel(bool p_show) {
	errors_panel->set_visible(p_show);
}

void ScriptTextEditor::_show_warnings_panel(bool p_show) {
	warnings_panel->set_visible(p_show);
}

void ScriptTextEditor::_warning_clicked(const Variant &p_line) {
	if (p_line.get_type() == Variant::INT) {
		goto_line_centered(p_line.operator int64_t());
	} else if (p_line.get_type() == Variant::DICTIONARY) {
		Dictionary meta = p_line.operator Dictionary();
		const int line = meta["line"].operator int64_t() - 1;
		const String code = meta["code"].operator String();
		const String quote_style = EDITOR_GET("text_editor/completion/use_single_quotes") ? "'" : "\"";

		CodeEdit *text_editor = code_editor->get_text_editor();
		String prev_line = line > 0 ? text_editor->get_line(line - 1) : "";
		if (prev_line.contains("@warning_ignore")) {
			const int closing_bracket_idx = prev_line.find_char(')');
			const String text_to_insert = ", " + code.quote(quote_style);
			text_editor->insert_text(text_to_insert, line - 1, closing_bracket_idx);
		} else {
			const int indent = text_editor->get_indent_level(line) / text_editor->get_indent_size();
			String annotation_indent;
			if (!text_editor->is_indent_using_spaces()) {
				annotation_indent = String("\t").repeat(indent);
			} else {
				annotation_indent = String(" ").repeat(text_editor->get_indent_size() * indent);
			}
			text_editor->insert_line_at(line, annotation_indent + "@warning_ignore(" + code.quote(quote_style) + ")");
		}

		_validate_script();
	}
}

void ScriptTextEditor::_error_clicked(const Variant &p_line) {
	if (p_line.get_type() == Variant::INT) {
		goto_line_centered(p_line.operator int64_t());
	} else if (p_line.get_type() == Variant::DICTIONARY) {
		Dictionary meta = p_line.operator Dictionary();
		const String path = meta["path"].operator String();
		const int line = meta["line"].operator int64_t();
		const int column = meta["column"].operator int64_t();
		if (path.is_empty()) {
			goto_line_centered(line, column);
		} else {
			Ref<Resource> scr = ResourceLoader::load(path);
			if (scr.is_null()) {
				EditorNode::get_singleton()->show_warning(TTR("Could not load file at:") + "\n\n" + path, TTR("Error!"));
			} else {
				int corrected_column = column;

				const String line_text = code_editor->get_text_editor()->get_line(line);
				const int indent_size = code_editor->get_text_editor()->get_indent_size();
				if (indent_size > 1) {
					const int tab_count = line_text.length() - line_text.lstrip("\t").length();
					corrected_column -= tab_count * (indent_size - 1);
				}

				ScriptEditor::get_singleton()->edit(scr, line, corrected_column);
			}
		}
	}
}

void ScriptTextEditor::reload_text() {
	ERR_FAIL_COND(script.is_null());

	CodeEdit *te = code_editor->get_text_editor();
	int column = te->get_caret_column();
	int row = te->get_caret_line();
	int h = te->get_h_scroll();
	int v = te->get_v_scroll();

	te->set_text(script->get_source_code());
	te->set_caret_line(row);
	te->set_caret_column(column);
	te->set_h_scroll(h);
	te->set_v_scroll(v);

	te->tag_saved_version();

	code_editor->update_line_and_column();
	if (editor_enabled) {
		_validate_script();
	}
}

void ScriptTextEditor::add_callback(const String &p_function, const PackedStringArray &p_args) {
	ScriptLanguage *language = script->get_language();
	if (!language->can_make_function()) {
		return;
	}
	code_editor->get_text_editor()->begin_complex_operation();
	code_editor->get_text_editor()->remove_secondary_carets();
	code_editor->get_text_editor()->deselect();
	String code = code_editor->get_text_editor()->get_text();
	int pos = language->find_function(p_function, code);
	if (pos == -1) {
		// Function does not exist, create it at the end of the file.
		int last_line = code_editor->get_text_editor()->get_line_count() - 1;
		String func = language->make_function("", p_function, p_args);
		code_editor->get_text_editor()->insert_text("\n\n" + func, last_line, code_editor->get_text_editor()->get_line(last_line).length());
		pos = last_line + 3;
	}
	// Put caret on the line after the function, after the indent.
	int indent_column = 1;
	if (EDITOR_GET("text_editor/behavior/indent/type")) {
		indent_column = EDITOR_GET("text_editor/behavior/indent/size");
	}
	code_editor->get_text_editor()->set_caret_line(pos, true, true, -1);
	code_editor->get_text_editor()->set_caret_column(indent_column);
	code_editor->get_text_editor()->end_complex_operation();
}

bool ScriptTextEditor::show_members_overview() {
	return true;
}

bool ScriptTextEditor::_is_valid_color_info(const Dictionary &p_info) {
	if (p_info.get_valid("color").get_type() != Variant::COLOR) {
		return false;
	}
	if (!p_info.get_valid("color_end").is_num() || !p_info.get_valid("color_mode").is_num()) {
		return false;
	}
	return true;
}

Array ScriptTextEditor::_inline_object_parse(const String &p_text) {
	Array result;
	int i_end_previous = 0;
	int i_start = p_text.find("Color");

	while (i_start != -1) {
		// Ignore words that just have "Color" in them.
		if (i_start != 0 && ('_' + p_text.substr(i_start - 1, 1)).is_valid_ascii_identifier()) {
			i_end_previous = MAX(i_end_previous, i_start);
			i_start = p_text.find("Color", i_start + 1);
			continue;
		}

		const int i_par_start = p_text.find_char('(', i_start + 5);
		const int i_par_end = p_text.find_char(')', i_start + 5);
		if (i_par_start == -1 || i_par_end == -1) {
			i_end_previous = MAX(i_end_previous, i_start);
			i_start = p_text.find("Color", i_start + 1);
			continue;
		}

		Dictionary color_info;
		color_info["column"] = i_start;
		color_info["width_ratio"] = 1.0;
		color_info["color_end"] = i_par_end;

		const String fn_name = p_text.substr(i_start + 5, i_par_start - i_start - 5);
		const String s_params = p_text.substr(i_par_start + 1, i_par_end - i_par_start - 1);
		bool has_added_color = false;

		if (fn_name.is_empty()) {
			String stripped = s_params.strip_edges(true, true);
			if (stripped.length() > 1 && (stripped[0] == '"' || stripped[0] == '\'')) {
				// String constructor.
				const char32_t string_delimiter = stripped[0];
				if (stripped[stripped.length() - 1] == string_delimiter) {
					const String color_string = stripped.substr(1, stripped.length() - 2);
					if (!color_string.contains_char(string_delimiter)) {
						color_info["color"] = Color::from_string(color_string, Color());
						color_info["color_mode"] = MODE_STRING;
						has_added_color = true;
					}
				}
			} else if (stripped.length() == 10 && stripped.begins_with("0x")) {
				// Hex constructor.
				const String color_string = stripped.substr(2);
				if (color_string.is_valid_hex_number(false)) {
					color_info["color"] = Color::from_string(color_string, Color());
					color_info["color_mode"] = MODE_HEX;
					has_added_color = true;
				}
			} else if (stripped.is_empty()) {
				// Empty Color() constructor.
				color_info["color"] = Color();
				color_info["color_mode"] = MODE_RGB;
				has_added_color = true;
			}
		}
		// Float & int parameters.
		if (!has_added_color && s_params.size() > 0) {
			const PackedStringArray s_params_split = s_params.split(",", false, 4);
			PackedFloat64Array params;
			bool valid_floats = true;
			for (const String &s_param : s_params_split) {
				// Only allow float literals, expressions won't be evaluated and could get replaced.
				if (!s_param.strip_edges().is_valid_float()) {
					valid_floats = false;
					break;
				}
				params.push_back(s_param.to_float());
			}
			if (valid_floats && params.size() == 3) {
				if (fn_name == ".from_rgba8") {
					params.push_back(255);
				} else {
					params.push_back(1.0);
				}
			}
			if (valid_floats && params.size() == 4) {
				has_added_color = true;
				if (fn_name == ".from_ok_hsl") {
					color_info["color"] = Color::from_ok_hsl(params[0], params[1], params[2], params[3]);
					color_info["color_mode"] = MODE_OKHSL;
				} else if (fn_name == ".from_hsv") {
					color_info["color"] = Color::from_hsv(params[0], params[1], params[2], params[3]);
					color_info["color_mode"] = MODE_HSV;
				} else if (fn_name == ".from_rgba8") {
					color_info["color"] = Color::from_rgba8(int(params[0]), int(params[1]), int(params[2]), int(params[3]));
					color_info["color_mode"] = MODE_RGB8;
				} else if (fn_name.is_empty()) {
					color_info["color"] = Color(params[0], params[1], params[2], params[3]);
					color_info["color_mode"] = MODE_RGB;
				} else {
					has_added_color = false;
				}
			}
		}

		if (has_added_color) {
			result.push_back(color_info);
			i_end_previous = i_par_end + 1;
		}
		i_end_previous = MAX(i_end_previous, i_start);
		i_start = p_text.find("Color", i_start + 1);
	}
	return result;
}

void ScriptTextEditor::_inline_object_draw(const Dictionary &p_info, const Rect2 &p_rect) {
	if (_is_valid_color_info(p_info)) {
		Rect2 col_rect = p_rect.grow(-4);
		if (color_alpha_texture.is_null()) {
			color_alpha_texture = inline_color_picker->get_theme_icon("sample_bg", "ColorPicker");
		}
		RID text_ci = code_editor->get_text_editor()->get_text_canvas_item();
		RS::get_singleton()->canvas_item_add_rect(text_ci, p_rect.grow(-3), Color(1, 1, 1));
		color_alpha_texture->draw_rect(text_ci, col_rect);
		RS::get_singleton()->canvas_item_add_rect(text_ci, col_rect, Color(p_info["color"]));
	}
}

void ScriptTextEditor::_inline_object_handle_click(const Dictionary &p_info, const Rect2 &p_rect) {
	if (_is_valid_color_info(p_info)) {
		inline_color_picker->set_pick_color(p_info["color"]);
		inline_color_line = p_info["line"];
		inline_color_start = p_info["column"];
		inline_color_end = p_info["color_end"];

		// Reset tooltip hover timer.
		code_editor->get_text_editor()->set_symbol_tooltip_on_hover_enabled(false);
		code_editor->get_text_editor()->set_symbol_tooltip_on_hover_enabled(true);

		_update_color_constructor_options();
		inline_color_options->select(p_info["color_mode"]);
		EditorNode::get_singleton()->setup_color_picker(inline_color_picker);

		// Move popup above the line if it's too low.
		float_t view_h = get_viewport_rect().size.y;
		float_t pop_h = inline_color_popup->get_contents_minimum_size().y;
		float_t pop_y = p_rect.get_end().y;
		float_t pop_x = p_rect.position.x;
		if (pop_y + pop_h > view_h) {
			pop_y = p_rect.position.y - pop_h;
		}
		// Move popup to the right if it's too high.
		if (pop_y < 0) {
			pop_x = p_rect.get_end().x;
		}

		inline_color_popup->popup(Rect2(pop_x, pop_y, 0, 0));
	}
}

String ScriptTextEditor::_picker_color_stringify(const Color &p_color, COLOR_MODE p_mode) {
	String result;
	String fname;
	Vector<String> str_params;
	switch (p_mode) {
		case ScriptTextEditor::MODE_STRING: {
			str_params.push_back("\"" + p_color.to_html() + "\"");
		} break;
		case ScriptTextEditor::MODE_HEX: {
			str_params.push_back("0x" + p_color.to_html());
		} break;
		case ScriptTextEditor::MODE_RGB: {
			str_params = {
				String::num(p_color.r, 3),
				String::num(p_color.g, 3),
				String::num(p_color.b, 3),
				String::num(p_color.a, 3)
			};
		} break;
		case ScriptTextEditor::MODE_HSV: {
			str_params = {
				String::num(p_color.get_h(), 3),
				String::num(p_color.get_s(), 3),
				String::num(p_color.get_v(), 3),
				String::num(p_color.a, 3)
			};
			fname = ".from_hsv";
		} break;
		case ScriptTextEditor::MODE_OKHSL: {
			str_params = {
				String::num(p_color.get_ok_hsl_h(), 3),
				String::num(p_color.get_ok_hsl_s(), 3),
				String::num(p_color.get_ok_hsl_l(), 3),
				String::num(p_color.a, 3)
			};
			fname = ".from_ok_hsl";
		} break;
		case ScriptTextEditor::MODE_RGB8: {
			str_params = {
				itos(p_color.get_r8()),
				itos(p_color.get_g8()),
				itos(p_color.get_b8()),
				itos(p_color.get_a8())
			};
			fname = ".from_rgba8";
		} break;
		default: {
		} break;
	}
	result = "Color" + fname + "(" + String(", ").join(str_params) + ")";
	return result;
}

void ScriptTextEditor::_picker_color_changed(const Color &p_color) {
	_update_color_constructor_options();
	_update_color_text();
}

void ScriptTextEditor::_update_color_constructor_options() {
	int item_count = inline_color_options->get_item_count();
	// Update or add each constructor as an option.
	for (int i = 0; i < MODE_MAX; i++) {
		String option_text = _picker_color_stringify(inline_color_picker->get_pick_color(), (COLOR_MODE)i);
		if (i >= item_count) {
			inline_color_options->add_item(option_text);
		} else {
			inline_color_options->set_item_text(i, option_text);
		}
	}
}

void ScriptTextEditor::_update_background_color() {
	// Clear background lines.
	CodeEdit *te = code_editor->get_text_editor();
	for (int i = 0; i < te->get_line_count(); i++) {
		bool is_folded_code_region = te->is_line_code_region_start(i) && te->is_line_folded(i);
		te->set_line_background_color(i, is_folded_code_region ? folded_code_region_color : Color(0, 0, 0, 0));
	}

	// Set the warning background.
	if (warning_line_color.a != 0.0) {
		for (const ScriptLanguage::Warning &warning : warnings) {
			int warning_start_line = CLAMP(warning.start_line - 1, 0, te->get_line_count() - 1);
			int warning_end_line = CLAMP(warning.end_line - 1, 0, te->get_line_count() - 1);
			int folded_line_header = te->get_folded_line_header(warning_start_line);

			// If the warning highlight is too long, only highlight the start line.
			const int warning_max_lines = 20;

			te->set_line_background_color(folded_line_header, warning_line_color);
			if (warning_end_line - warning_start_line < warning_max_lines) {
				for (int i = warning_start_line + 1; i <= warning_end_line; i++) {
					te->set_line_background_color(i, warning_line_color);
				}
			}
		}
	}

	// Set the error background.
	if (marked_line_color.a != 0.0) {
		for (const ScriptLanguage::ScriptError &error : errors) {
			int error_line = CLAMP(error.line - 1, 0, te->get_line_count() - 1);
			int folded_line_header = te->get_folded_line_header(error_line);

			te->set_line_background_color(folded_line_header, marked_line_color);
		}
	}
}

void ScriptTextEditor::_update_color_text() {
	if (inline_color_line < 0) {
		return;
	}
	String result = inline_color_options->get_item_text(inline_color_options->get_selected_id());
	code_editor->get_text_editor()->begin_complex_operation();
	code_editor->get_text_editor()->remove_text(inline_color_line, inline_color_start, inline_color_line, inline_color_end + 1);
	inline_color_end = inline_color_start + result.size() - 2;
	code_editor->get_text_editor()->insert_text(result, inline_color_line, inline_color_start);
	code_editor->get_text_editor()->end_complex_operation();
}

void ScriptTextEditor::update_settings() {
	code_editor->get_text_editor()->set_gutter_draw(connection_gutter, EDITOR_GET("text_editor/appearance/gutters/show_info_gutter"));
	if (EDITOR_GET("text_editor/appearance/enable_inline_color_picker")) {
		code_editor->get_text_editor()->set_inline_object_handlers(
				callable_mp(this, &ScriptTextEditor::_inline_object_parse),
				callable_mp(this, &ScriptTextEditor::_inline_object_draw),
				callable_mp(this, &ScriptTextEditor::_inline_object_handle_click));
	} else {
		code_editor->get_text_editor()->set_inline_object_handlers(Callable(), Callable(), Callable());
	}
	code_editor->update_editor_settings();
}

bool ScriptTextEditor::is_unsaved() {
	const bool unsaved =
			code_editor->get_text_editor()->get_version() != code_editor->get_text_editor()->get_saved_version() ||
			script->get_path().is_empty(); // In memory.
	return unsaved;
}

Variant ScriptTextEditor::get_edit_state() {
	if (pending_state != Variant()) {
		return pending_state;
	}
	return code_editor->get_edit_state();
}

void ScriptTextEditor::set_edit_state(const Variant &p_state) {
	if (editor_enabled) {
		code_editor->set_edit_state(p_state);
	} else {
		// The editor is not fully initialized, so the state can't be loaded properly.
		pending_state = p_state;
	}

	Dictionary state = p_state;
	if (state.has("syntax_highlighter")) {
		int idx = highlighter_menu->get_item_idx_from_text(state["syntax_highlighter"]);
		if (idx >= 0) {
			_change_syntax_highlighter(idx);
		}
	}

	if (editor_enabled) {
#ifndef ANDROID_ENABLED
		ensure_focus();
#endif
	}
}

Variant ScriptTextEditor::get_navigation_state() {
	return code_editor->get_navigation_state();
}

Variant ScriptTextEditor::get_previous_state() {
	return code_editor->get_previous_state();
}

void ScriptTextEditor::store_previous_state() {
	return code_editor->store_previous_state();
}

void ScriptTextEditor::_convert_case(CodeTextEditor::CaseStyle p_case) {
	code_editor->convert_case(p_case);
}

void ScriptTextEditor::trim_trailing_whitespace() {
	code_editor->trim_trailing_whitespace();
}

void ScriptTextEditor::trim_final_newlines() {
	code_editor->trim_final_newlines();
}

void ScriptTextEditor::insert_final_newline() {
	code_editor->insert_final_newline();
}

void ScriptTextEditor::convert_indent() {
	code_editor->get_text_editor()->convert_indent();
}

void ScriptTextEditor::tag_saved_version() {
	code_editor->get_text_editor()->tag_saved_version();
	edited_file_data.last_modified_time = FileAccess::get_modified_time(edited_file_data.path);
}

void ScriptTextEditor::goto_line(int p_line, int p_column) {
	code_editor->goto_line(p_line, p_column);
}

void ScriptTextEditor::goto_line_selection(int p_line, int p_begin, int p_end) {
	code_editor->goto_line_selection(p_line, p_begin, p_end);
}

void ScriptTextEditor::goto_line_centered(int p_line, int p_column) {
	code_editor->goto_line_centered(p_line, p_column);
}

void ScriptTextEditor::set_executing_line(int p_line) {
	code_editor->set_executing_line(p_line);
}

void ScriptTextEditor::clear_executing_line() {
	code_editor->clear_executing_line();
}

void ScriptTextEditor::ensure_focus() {
	code_editor->get_text_editor()->grab_focus();
}

String ScriptTextEditor::get_name() {
	String name;

	name = script->get_path().get_file();
	if (name.is_empty()) {
		// This appears for newly created built-in scripts before saving the scene.
		name = TTR("[unsaved]");
	} else if (script->is_built_in()) {
		const String &script_name = script->get_name();
		if (!script_name.is_empty()) {
			// If the built-in script has a custom resource name defined,
			// display the built-in script name as follows: `ResourceName (scene_file.tscn)`
			name = vformat("%s (%s)", script_name, name.get_slice("::", 0));
		}
	}

	if (is_unsaved()) {
		name += "(*)";
	}

	return name;
}

Ref<Texture2D> ScriptTextEditor::get_theme_icon() {
	if (get_parent_control()) {
		String icon_name = script->get_class();
		if (script->is_built_in()) {
			icon_name += "Internal";
		}

		if (get_parent_control()->has_theme_icon(icon_name, EditorStringName(EditorIcons))) {
			return get_parent_control()->get_editor_theme_icon(icon_name);
		} else if (get_parent_control()->has_theme_icon(script->get_class(), EditorStringName(EditorIcons))) {
			return get_parent_control()->get_editor_theme_icon(script->get_class());
		}
	}

	Ref<Texture2D> extension_language_icon = EditorNode::get_editor_data().extension_class_get_icon(script->get_class());
	Ref<Texture2D> extension_language_alt_icon;
	if (script->is_built_in()) {
		extension_language_alt_icon = EditorNode::get_editor_data().extension_class_get_icon(script->get_class() + "Internal");
	}

	if (extension_language_alt_icon.is_valid()) {
		return extension_language_alt_icon;
	} else if (extension_language_icon.is_valid()) {
		return extension_language_icon;
	}

	return Ref<Texture2D>();
}

void ScriptTextEditor::_validate_script() {
	CodeEdit *te = code_editor->get_text_editor();

	String text = te->get_text();
	List<String> fnc;

	warnings.clear();
	errors.clear();
	depended_errors.clear();
	safe_lines.clear();

	if (!script->get_language()->validate(text, script->get_path(), &fnc, &errors, &warnings, &safe_lines)) {
		List<ScriptLanguage::ScriptError>::Element *E = errors.front();
		while (E) {
			List<ScriptLanguage::ScriptError>::Element *next_E = E->next();
			if ((E->get().path.is_empty() && !script->get_path().is_empty()) || E->get().path != script->get_path()) {
				depended_errors[E->get().path].push_back(E->get());
				E->erase();
			}
			E = next_E;
		}

		if (errors.size() > 0) {
			const int line = errors.front()->get().line;
			const int column = errors.front()->get().column;
			const String message = errors.front()->get().message.replace("[", "[lb]");
			const String error_text = vformat(TTR("Error at ([hint=Line %d, column %d]%d, %d[/hint]):"), line, column, line, column) + " " + message;
			code_editor->set_error(error_text);
			code_editor->set_error_pos(line - 1, column - 1);
		}
		script_is_valid = false;
	} else {
		code_editor->set_error("");
		if (!script->is_tool()) {
			script->set_source_code(text);
			script->update_exports();
			te->get_syntax_highlighter()->update_cache();
		}

		functions.clear();
		for (const String &E : fnc) {
			functions.push_back(E);
		}
		script_is_valid = true;
	}
	_update_connected_methods();
	_update_warnings();
	_update_errors();
	_update_background_color();

	if (!pending_dragged_exports.is_empty()) {
		_assign_dragged_export_variables();
	}

	emit_signal(SNAME("name_changed"));
	emit_signal(SNAME("edited_script_changed"));
}

void ScriptTextEditor::_update_warnings() {
	int warning_nb = warnings.size();
	warnings_panel->clear();

	bool has_connections_table = false;
	// Add missing connections.
	if (GLOBAL_GET("debug/gdscript/warnings/enable")) {
		Node *base = get_tree()->get_edited_scene_root();
		if (base && missing_connections.size() > 0) {
			has_connections_table = true;
			warnings_panel->push_table(1);
			for (const Connection &connection : missing_connections) {
				String base_path = base->get_name();
				String source_path = base == connection.signal.get_object() ? base_path : base_path + "/" + String(base->get_path_to(Object::cast_to<Node>(connection.signal.get_object())));
				String target_path = base == connection.callable.get_object() ? base_path : base_path + "/" + String(base->get_path_to(Object::cast_to<Node>(connection.callable.get_object())));

				warnings_panel->push_cell();
				warnings_panel->push_color(warnings_panel->get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));
				warnings_panel->add_text(vformat(TTR("Missing connected method '%s' for signal '%s' from node '%s' to node '%s'."), connection.callable.get_method(), connection.signal.get_name(), source_path, target_path));
				warnings_panel->pop(); // Color.
				warnings_panel->pop(); // Cell.
			}
			warnings_panel->pop(); // Table.

			warning_nb += missing_connections.size();
		}
	}

	code_editor->set_warning_count(warning_nb);

	if (has_connections_table) {
		warnings_panel->add_newline();
	}

	// Add script warnings.
	warnings_panel->push_table(3);
	for (const ScriptLanguage::Warning &w : warnings) {
		Dictionary ignore_meta;
		ignore_meta["line"] = w.start_line;
		ignore_meta["code"] = w.string_code.to_lower();
		warnings_panel->push_cell();
		warnings_panel->push_meta(ignore_meta);
		warnings_panel->push_color(
				warnings_panel->get_theme_color(SNAME("accent_color"), EditorStringName(Editor)).lerp(warnings_panel->get_theme_color(SNAME("mono_color"), EditorStringName(Editor)), 0.5f));
		warnings_panel->add_text(TTR("[Ignore]"));
		warnings_panel->pop(); // Color.
		warnings_panel->pop(); // Meta ignore.
		warnings_panel->pop(); // Cell.

		warnings_panel->push_cell();
		warnings_panel->push_meta(w.start_line - 1);
		warnings_panel->push_color(warnings_panel->get_theme_color(SNAME("warning_color"), EditorStringName(Editor)));
		warnings_panel->add_text(vformat(TTR("Line %d (%s):"), w.start_line, w.string_code));
		warnings_panel->pop(); // Color.
		warnings_panel->pop(); // Meta goto.
		warnings_panel->pop(); // Cell.

		warnings_panel->push_cell();
		warnings_panel->add_text(w.message);
		warnings_panel->add_newline();
		warnings_panel->pop(); // Cell.
	}
	warnings_panel->pop(); // Table.
}

void ScriptTextEditor::_update_errors() {
	code_editor->set_error_count(errors.size());

	errors_panel->clear();
	errors_panel->push_table(2);
	for (const ScriptLanguage::ScriptError &err : errors) {
		Dictionary click_meta;
		click_meta["line"] = err.line;
		click_meta["column"] = err.column;

		errors_panel->push_cell();
		errors_panel->push_meta(err.line - 1);
		errors_panel->push_color(warnings_panel->get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
		errors_panel->add_text(vformat(TTR("Line %d:"), err.line));
		errors_panel->pop(); // Color.
		errors_panel->pop(); // Meta goto.
		errors_panel->pop(); // Cell.

		errors_panel->push_cell();
		errors_panel->add_text(err.message);
		errors_panel->add_newline();
		errors_panel->pop(); // Cell.
	}
	errors_panel->pop(); // Table

	for (const KeyValue<String, List<ScriptLanguage::ScriptError>> &KV : depended_errors) {
		Dictionary click_meta;
		click_meta["path"] = KV.key;
		click_meta["line"] = 1;

		errors_panel->add_newline();
		errors_panel->add_newline();
		errors_panel->push_meta(click_meta);
		errors_panel->add_text(vformat(R"(%s:)", KV.key));
		errors_panel->pop(); // Meta goto.
		errors_panel->add_newline();

		errors_panel->push_indent(1);
		errors_panel->push_table(2);
		String filename = KV.key.get_file();
		for (const ScriptLanguage::ScriptError &err : KV.value) {
			click_meta["line"] = err.line;
			click_meta["column"] = err.column;

			errors_panel->push_cell();
			errors_panel->push_meta(click_meta);
			errors_panel->push_color(errors_panel->get_theme_color(SNAME("error_color"), EditorStringName(Editor)));
			errors_panel->add_text(vformat(TTR("Line %d:"), err.line));
			errors_panel->pop(); // Color.
			errors_panel->pop(); // Meta goto.
			errors_panel->pop(); // Cell.

			errors_panel->push_cell();
			errors_panel->add_text(err.message);
			errors_panel->pop(); // Cell.
		}
		errors_panel->pop(); // Table
		errors_panel->pop(); // Indent.
	}

	bool highlight_safe = EDITOR_GET("text_editor/appearance/gutters/highlight_type_safe_lines");
	bool last_is_safe = false;
	CodeEdit *te = code_editor->get_text_editor();

	for (int i = 0; i < te->get_line_count(); i++) {
		if (highlight_safe) {
			if (safe_lines.has(i + 1)) {
				te->set_line_gutter_item_color(i, line_number_gutter, safe_line_number_color);
				last_is_safe = true;
			} else if (last_is_safe && (te->is_in_comment(i) != -1 || te->get_line(i).strip_edges().is_empty())) {
				te->set_line_gutter_item_color(i, line_number_gutter, safe_line_number_color);
			} else {
				te->set_line_gutter_item_color(i, line_number_gutter, default_line_number_color);
				last_is_safe = false;
			}
		} else {
			te->set_line_gutter_item_color(i, 1, default_line_number_color);
		}
	}
}

void ScriptTextEditor::_update_bookmark_list() {
	bookmarks_menu->clear();
	bookmarks_menu->reset_size();

	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_bookmark"), BOOKMARK_TOGGLE);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/remove_all_bookmarks"), BOOKMARK_REMOVE_ALL);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_next_bookmark"), BOOKMARK_GOTO_NEXT);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_previous_bookmark"), BOOKMARK_GOTO_PREV);

	PackedInt32Array bookmark_list = code_editor->get_text_editor()->get_bookmarked_lines();
	if (bookmark_list.is_empty()) {
		return;
	}

	bookmarks_menu->add_separator();

	for (int i = 0; i < bookmark_list.size(); i++) {
		// Strip edges to remove spaces or tabs.
		// Also replace any tabs by spaces, since we can't print tabs in the menu.
		String line = code_editor->get_text_editor()->get_line(bookmark_list[i]).replace("\t", "  ").strip_edges();

		// Limit the size of the line if too big.
		if (line.length() > 50) {
			line = line.substr(0, 50);
		}

		bookmarks_menu->add_item(String::num_int64(bookmark_list[i] + 1) + " - `" + line + "`");
		bookmarks_menu->set_item_metadata(-1, bookmark_list[i]);
	}
}

void ScriptTextEditor::_bookmark_item_pressed(int p_idx) {
	if (p_idx < 4) { // Any item before the separator.
		_edit_option(bookmarks_menu->get_item_id(p_idx));
	} else {
		code_editor->goto_line_centered(bookmarks_menu->get_item_metadata(p_idx));
	}
}

static Vector<Node *> _find_all_node_for_script(Node *p_base, Node *p_current, const Ref<Script> &p_script) {
	Vector<Node *> nodes;

	if (p_current->get_owner() != p_base && p_base != p_current) {
		return nodes;
	}

	Ref<Script> c = p_current->get_script();
	if (c == p_script) {
		nodes.push_back(p_current);
	}

	for (int i = 0; i < p_current->get_child_count(); i++) {
		Vector<Node *> found = _find_all_node_for_script(p_base, p_current->get_child(i), p_script);
		nodes.append_array(found);
	}

	return nodes;
}

static Node *_find_node_for_script(Node *p_base, Node *p_current, const Ref<Script> &p_script) {
	if (p_current->get_owner() != p_base && p_base != p_current) {
		return nullptr;
	}
	Ref<Script> c = p_current->get_script();
	if (c == p_script) {
		return p_current;
	}
	for (int i = 0; i < p_current->get_child_count(); i++) {
		Node *found = _find_node_for_script(p_base, p_current->get_child(i), p_script);
		if (found) {
			return found;
		}
	}

	return nullptr;
}

static void _find_changed_scripts_for_external_editor(Node *p_base, Node *p_current, HashSet<Ref<Script>> &r_scripts) {
	if (p_current->get_owner() != p_base && p_base != p_current) {
		return;
	}
	Ref<Script> c = p_current->get_script();

	if (c.is_valid()) {
		r_scripts.insert(c);
	}

	for (int i = 0; i < p_current->get_child_count(); i++) {
		_find_changed_scripts_for_external_editor(p_base, p_current->get_child(i), r_scripts);
	}
}

void ScriptEditor::_update_modified_scripts_for_external_editor(Ref<Script> p_for_script) {
	bool use_external_editor = bool(EDITOR_GET("text_editor/external/use_external_editor"));

	ERR_FAIL_NULL(get_tree());

	HashSet<Ref<Script>> scripts;

	Node *base = get_tree()->get_edited_scene_root();
	if (base) {
		_find_changed_scripts_for_external_editor(base, base, scripts);
	}

	for (const Ref<Script> &E : scripts) {
		Ref<Script> scr = E;

		if (!use_external_editor && !scr->get_language()->overrides_external_editor()) {
			continue; // We're not using an external editor for this script.
		}

		if (p_for_script.is_valid() && p_for_script != scr) {
			continue;
		}

		if (scr->is_built_in()) {
			continue; //internal script, who cares, though weird
		}

		uint64_t last_date = scr->get_last_modified_time();
		uint64_t date = FileAccess::get_modified_time(scr->get_path());

		if (last_date != date) {
			Ref<Script> rel_scr = ResourceLoader::load(scr->get_path(), scr->get_class(), ResourceFormatLoader::CACHE_MODE_IGNORE);
			ERR_CONTINUE(rel_scr.is_null());
			scr->set_source_code(rel_scr->get_source_code());
			scr->set_last_modified_time(rel_scr->get_last_modified_time());
			scr->update_exports();

			trigger_live_script_reload(scr->get_path());
		}
	}
}

void ScriptTextEditor::_code_complete_scripts(void *p_ud, const String &p_code, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_force) {
	ScriptTextEditor *ste = (ScriptTextEditor *)p_ud;
	ste->_code_complete_script(p_code, r_options, r_force);
}

void ScriptTextEditor::_code_complete_script(const String &p_code, List<ScriptLanguage::CodeCompletionOption> *r_options, bool &r_force) {
	if (color_panel->is_visible()) {
		return;
	}
	Node *base = get_tree()->get_edited_scene_root();
	if (base) {
		base = _find_node_for_script(base, base, script);
	}
	String hint;
	Error err = script->get_language()->complete_code(p_code, script->get_path(), base, r_options, r_force, hint);

	if (err == OK) {
		code_editor->get_text_editor()->set_code_hint(hint);
	}
}

void ScriptTextEditor::_update_breakpoint_list() {
	breakpoints_menu->clear();
	breakpoints_menu->reset_size();

	breakpoints_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_breakpoint"), DEBUG_TOGGLE_BREAKPOINT);
	breakpoints_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/remove_all_breakpoints"), DEBUG_REMOVE_ALL_BREAKPOINTS);
	breakpoints_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_next_breakpoint"), DEBUG_GOTO_NEXT_BREAKPOINT);
	breakpoints_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_previous_breakpoint"), DEBUG_GOTO_PREV_BREAKPOINT);

	PackedInt32Array breakpoint_list = code_editor->get_text_editor()->get_breakpointed_lines();
	if (breakpoint_list.is_empty()) {
		return;
	}

	breakpoints_menu->add_separator();

	for (int i = 0; i < breakpoint_list.size(); i++) {
		// Strip edges to remove spaces or tabs.
		// Also replace any tabs by spaces, since we can't print tabs in the menu.
		String line = code_editor->get_text_editor()->get_line(breakpoint_list[i]).replace("\t", "  ").strip_edges();

		// Limit the size of the line if too big.
		if (line.length() > 50) {
			line = line.substr(0, 50);
		}

		breakpoints_menu->add_item(String::num_int64(breakpoint_list[i] + 1) + " - `" + line + "`");
		breakpoints_menu->set_item_metadata(-1, breakpoint_list[i]);
	}
}

void ScriptTextEditor::_breakpoint_item_pressed(int p_idx) {
	if (p_idx < 4) { // Any item before the separator.
		_edit_option(breakpoints_menu->get_item_id(p_idx));
	} else {
		code_editor->goto_line_centered(breakpoints_menu->get_item_metadata(p_idx));
	}
}

void ScriptTextEditor::_breakpoint_toggled(int p_row) {
	const CodeEdit *ce = code_editor->get_text_editor();
	bool enabled = p_row < ce->get_line_count() && ce->is_line_breakpointed(p_row);
	EditorDebuggerNode::get_singleton()->set_breakpoint(script->get_path(), p_row + 1, enabled);
}

void ScriptTextEditor::_on_caret_moved() {
	if (code_editor->is_previewing_navigation_change()) {
		return;
	}
	int current_line = code_editor->get_text_editor()->get_caret_line();
	if (Math::abs(current_line - previous_line) >= 10) {
		Dictionary nav_state = get_navigation_state();
		nav_state["row"] = previous_line;
		nav_state["scroll_position"] = -1;
		emit_signal(SNAME("request_save_previous_state"), nav_state);
		store_previous_state();
	}
	previous_line = current_line;
}

void ScriptTextEditor::_lookup_symbol(const String &p_symbol, int p_row, int p_column) {
	Node *base = get_tree()->get_edited_scene_root();
	if (base) {
		base = _find_node_for_script(base, base, script);
	}

	ScriptLanguage::LookupResult result;
	String code_text = code_editor->get_text_editor()->get_text_with_cursor_char(p_row, p_column);
	Error lc_error = script->get_language()->lookup_code(code_text, p_symbol, script->get_path(), base, result);
	if (ScriptServer::is_global_class(p_symbol)) {
		EditorNode::get_singleton()->load_resource(ScriptServer::get_global_class_path(p_symbol));
	} else if (p_symbol.is_resource_file() || p_symbol.begins_with("uid://")) {
		if (DirAccess::dir_exists_absolute(p_symbol)) {
			FileSystemDock::get_singleton()->navigate_to_path(p_symbol);
		} else {
			EditorNode::get_singleton()->load_scene_or_resource(p_symbol);
		}
	} else if (lc_error == OK) {
		_goto_line(p_row);

		if (!result.class_name.is_empty() && EditorHelp::get_doc_data()->class_list.has(result.class_name) && !EditorHelp::get_doc_data()->class_list[result.class_name].is_script_doc) {
			switch (result.type) {
				case ScriptLanguage::LOOKUP_RESULT_CLASS: {
					emit_signal(SNAME("go_to_help"), "class_name:" + result.class_name);
				} break;
				case ScriptLanguage::LOOKUP_RESULT_CLASS_CONSTANT: {
					StringName cname = result.class_name;
					while (ClassDB::class_exists(cname)) {
						if (ClassDB::has_integer_constant(cname, result.class_member, true)) {
							result.class_name = cname;
							break;
						}
						cname = ClassDB::get_parent_class(cname);
					}
					emit_signal(SNAME("go_to_help"), "class_constant:" + result.class_name + ":" + result.class_member);
				} break;
				case ScriptLanguage::LOOKUP_RESULT_CLASS_PROPERTY: {
					StringName cname = result.class_name;
					while (ClassDB::class_exists(cname)) {
						if (ClassDB::has_property(cname, result.class_member, true)) {
							result.class_name = cname;
							break;
						}
						cname = ClassDB::get_parent_class(cname);
					}
					emit_signal(SNAME("go_to_help"), "class_property:" + result.class_name + ":" + result.class_member);
				} break;
				case ScriptLanguage::LOOKUP_RESULT_CLASS_METHOD: {
					StringName cname = result.class_name;
					while (ClassDB::class_exists(cname)) {
						if (ClassDB::has_method(cname, result.class_member, true)) {
							result.class_name = cname;
							break;
						}
						cname = ClassDB::get_parent_class(cname);
					}
					emit_signal(SNAME("go_to_help"), "class_method:" + result.class_name + ":" + result.class_member);
				} break;
				case ScriptLanguage::LOOKUP_RESULT_CLASS_SIGNAL: {
					StringName cname = result.class_name;
					while (ClassDB::class_exists(cname)) {
						if (ClassDB::has_signal(cname, result.class_member, true)) {
							result.class_name = cname;
							break;
						}
						cname = ClassDB::get_parent_class(cname);
					}
					emit_signal(SNAME("go_to_help"), "class_signal:" + result.class_name + ":" + result.class_member);
				} break;
				case ScriptLanguage::LOOKUP_RESULT_CLASS_ENUM: {
					StringName cname = result.class_name;
					while (ClassDB::class_exists(cname)) {
						if (ClassDB::has_enum(cname, result.class_member, true)) {
							result.class_name = cname;
							break;
						}
						cname = ClassDB::get_parent_class(cname);
					}
					emit_signal(SNAME("go_to_help"), "class_enum:" + result.class_name + ":" + result.class_member);
				} break;
				case ScriptLanguage::LOOKUP_RESULT_CLASS_ANNOTATION: {
					emit_signal(SNAME("go_to_help"), "class_annotation:" + result.class_name + ":" + result.class_member);
				} break;
				case ScriptLanguage::LOOKUP_RESULT_CLASS_TBD_GLOBALSCOPE: { // Deprecated.
					emit_signal(SNAME("go_to_help"), "class_global:" + result.class_name + ":" + result.class_member);
				} break;
				case ScriptLanguage::LOOKUP_RESULT_SCRIPT_LOCATION:
				case ScriptLanguage::LOOKUP_RESULT_LOCAL_CONSTANT:
				case ScriptLanguage::LOOKUP_RESULT_LOCAL_VARIABLE:
				case ScriptLanguage::LOOKUP_RESULT_MAX: {
					// Nothing to do.
				} break;
			}
		} else if (result.location >= 0) {
			if (result.script.is_valid()) {
				emit_signal(SNAME("request_open_script_at_line"), result.script, result.location - 1);
			} else {
				emit_signal(SNAME("request_save_history"));
				goto_line_centered(result.location - 1);
			}
		}
	} else if (ProjectSettings::get_singleton()->has_autoload(p_symbol)) {
		// Check for Autoload scenes.
		const ProjectSettings::AutoloadInfo &info = ProjectSettings::get_singleton()->get_autoload(p_symbol);
		if (info.is_singleton) {
			EditorNode::get_singleton()->load_scene(info.path);
		}
	} else if (p_symbol.is_relative_path()) {
		// Every symbol other than absolute path is relative path so keep this condition at last.
		String path = _get_absolute_path(p_symbol);
		if (FileAccess::exists(path)) {
			EditorNode::get_singleton()->load_scene_or_resource(path);
		}
	}
}

void ScriptTextEditor::_validate_symbol(const String &p_symbol) {
	CodeEdit *text_edit = code_editor->get_text_editor();

	Node *base = get_tree()->get_edited_scene_root();
	if (base) {
		base = _find_node_for_script(base, base, script);
	}

	ScriptLanguage::LookupResult result;
	String lc_text = code_editor->get_text_editor()->get_text_for_symbol_lookup();
	Error lc_error = script->get_language()->lookup_code(lc_text, p_symbol, script->get_path(), base, result);
	bool is_singleton = ProjectSettings::get_singleton()->has_autoload(p_symbol) && ProjectSettings::get_singleton()->get_autoload(p_symbol).is_singleton;
	if (lc_error == OK || is_singleton || ScriptServer::is_global_class(p_symbol) || p_symbol.is_resource_file() || p_symbol.begins_with("uid://")) {
		text_edit->set_symbol_lookup_word_as_valid(true);
	} else if (p_symbol.is_relative_path()) {
		String path = _get_absolute_path(p_symbol);
		if (FileAccess::exists(path)) {
			text_edit->set_symbol_lookup_word_as_valid(true);
		} else {
			text_edit->set_symbol_lookup_word_as_valid(false);
		}
	} else {
		text_edit->set_symbol_lookup_word_as_valid(false);
	}
}

void ScriptTextEditor::_show_symbol_tooltip(const String &p_symbol, int p_row, int p_column) {
	if (!EDITOR_GET("text_editor/behavior/documentation/enable_tooltips").booleanize()) {
		return;
	}

	if (p_symbol.begins_with("res://") || p_symbol.begins_with("uid://")) {
		Control *tmp = EditorHelpBitTooltip::make_tooltip(code_editor->get_text_editor(), "resource||" + p_symbol);
		memdelete(tmp);
		return;
	}

	Node *base = get_tree()->get_edited_scene_root();
	if (base) {
		base = _find_node_for_script(base, base, script);
	}

	ScriptLanguage::LookupResult result;
	String doc_symbol;
	const String code_text = code_editor->get_text_editor()->get_text_with_cursor_char(p_row, p_column);
	const Error lc_error = script->get_language()->lookup_code(code_text, p_symbol, script->get_path(), base, result);
	if (lc_error == OK) {
		switch (result.type) {
			case ScriptLanguage::LOOKUP_RESULT_CLASS: {
				doc_symbol = "class|" + result.class_name + "|";
			} break;
			case ScriptLanguage::LOOKUP_RESULT_CLASS_CONSTANT: {
				StringName cname = result.class_name;
				while (ClassDB::class_exists(cname)) {
					if (ClassDB::has_integer_constant(cname, result.class_member, true)) {
						result.class_name = cname;
						break;
					}
					cname = ClassDB::get_parent_class(cname);
				}
				doc_symbol = "constant|" + result.class_name + "|" + result.class_member;
			} break;
			case ScriptLanguage::LOOKUP_RESULT_CLASS_PROPERTY: {
				StringName cname = result.class_name;
				while (ClassDB::class_exists(cname)) {
					if (ClassDB::has_property(cname, result.class_member, true)) {
						result.class_name = cname;
						break;
					}
					cname = ClassDB::get_parent_class(cname);
				}
				doc_symbol = "property|" + result.class_name + "|" + result.class_member;
			} break;
			case ScriptLanguage::LOOKUP_RESULT_CLASS_METHOD: {
				StringName cname = result.class_name;
				while (ClassDB::class_exists(cname)) {
					if (ClassDB::has_method(cname, result.class_member, true)) {
						result.class_name = cname;
						break;
					}
					cname = ClassDB::get_parent_class(cname);
				}
				doc_symbol = "method|" + result.class_name + "|" + result.class_member;
			} break;
			case ScriptLanguage::LOOKUP_RESULT_CLASS_SIGNAL: {
				StringName cname = result.class_name;
				while (ClassDB::class_exists(cname)) {
					if (ClassDB::has_signal(cname, result.class_member, true)) {
						result.class_name = cname;
						break;
					}
					cname = ClassDB::get_parent_class(cname);
				}
				doc_symbol = "signal|" + result.class_name + "|" + result.class_member;
			} break;
			case ScriptLanguage::LOOKUP_RESULT_CLASS_ENUM: {
				StringName cname = result.class_name;
				while (ClassDB::class_exists(cname)) {
					if (ClassDB::has_enum(cname, result.class_member, true)) {
						result.class_name = cname;
						break;
					}
					cname = ClassDB::get_parent_class(cname);
				}
				doc_symbol = "enum|" + result.class_name + "|" + result.class_member;
			} break;
			case ScriptLanguage::LOOKUP_RESULT_CLASS_ANNOTATION: {
				doc_symbol = "annotation|" + result.class_name + "|" + result.class_member;
			} break;
			case ScriptLanguage::LOOKUP_RESULT_LOCAL_CONSTANT:
			case ScriptLanguage::LOOKUP_RESULT_LOCAL_VARIABLE: {
				const String item_type = (result.type == ScriptLanguage::LOOKUP_RESULT_LOCAL_CONSTANT) ? "local_constant" : "local_variable";
				Dictionary item_data;
				item_data["description"] = result.description;
				item_data["is_deprecated"] = result.is_deprecated;
				item_data["deprecated_message"] = result.deprecated_message;
				item_data["is_experimental"] = result.is_experimental;
				item_data["experimental_message"] = result.experimental_message;
				item_data["doc_type"] = result.doc_type;
				item_data["enumeration"] = result.enumeration;
				item_data["is_bitfield"] = result.is_bitfield;
				item_data["value"] = result.value;
				doc_symbol = item_type + "||" + p_symbol + "|" + JSON::stringify(item_data);
			} break;
			case ScriptLanguage::LOOKUP_RESULT_SCRIPT_LOCATION:
			case ScriptLanguage::LOOKUP_RESULT_CLASS_TBD_GLOBALSCOPE: // Deprecated.
			case ScriptLanguage::LOOKUP_RESULT_MAX: {
				// Nothing to do.
			} break;
		}
	}

	// NOTE: See also `ScriptEditor::_get_debug_tooltip()` for documentation tooltips disabled.
	String debug_value = EditorDebuggerNode::get_singleton()->get_var_value(p_symbol);
	if (!debug_value.is_empty()) {
		constexpr int DISPLAY_LIMIT = 1024;
		if (debug_value.size() > DISPLAY_LIMIT) {
			debug_value = debug_value.left(DISPLAY_LIMIT) + "... " + TTR("(truncated)");
		}
		debug_value = TTR("Current value: ") + debug_value.replace("[", "[lb]");
	}

	if (!doc_symbol.is_empty() || !debug_value.is_empty()) {
		Control *tmp = EditorHelpBitTooltip::make_tooltip(code_editor->get_text_editor(), doc_symbol, debug_value, true);
		memdelete(tmp);
	}
}

String ScriptTextEditor::_get_absolute_path(const String &rel_path) {
	String base_path = script->get_path().get_base_dir();
	String path = base_path.path_join(rel_path);
	return path.replace("///", "//").simplify_path();
}

void ScriptTextEditor::update_toggle_files_button() {
	code_editor->update_toggle_files_button();
}

void ScriptTextEditor::_update_connected_methods() {
	CodeEdit *text_edit = code_editor->get_text_editor();
	text_edit->set_gutter_width(connection_gutter, text_edit->get_line_height());
	for (int i = 0; i < text_edit->get_line_count(); i++) {
		text_edit->set_line_gutter_metadata(i, connection_gutter, Dictionary());
		text_edit->set_line_gutter_icon(i, connection_gutter, nullptr);
		text_edit->set_line_gutter_clickable(i, connection_gutter, false);
	}
	missing_connections.clear();

	if (!script_is_valid) {
		return;
	}

	Node *base = get_tree()->get_edited_scene_root();
	if (!base) {
		return;
	}

	// Add connection icons to methods.
	Vector<Node *> nodes = _find_all_node_for_script(base, base, script);
	HashSet<StringName> methods_found;
	for (int i = 0; i < nodes.size(); i++) {
		List<Connection> signal_connections;
		nodes[i]->get_signals_connected_to_this(&signal_connections);

		for (const Connection &connection : signal_connections) {
			if (!(connection.flags & CONNECT_PERSIST)) {
				continue;
			}

			// As deleted nodes are still accessible via the undo/redo system, check if they're still on the tree.
			Node *source = Object::cast_to<Node>(connection.signal.get_object());
			if (source && !source->is_inside_tree()) {
				continue;
			}

			const StringName method = connection.callable.get_method();
			if (methods_found.has(method)) {
				continue;
			}

			if (!ClassDB::has_method(script->get_instance_base_type(), method)) {
				int line = -1;

				for (int j = 0; j < functions.size(); j++) {
					String name = functions[j].get_slicec(':', 0);
					if (name == method) {
						Dictionary line_meta;
						line_meta["type"] = "connection";
						line_meta["method"] = method;
						line = functions[j].get_slicec(':', 1).to_int() - 1;
						text_edit->set_line_gutter_metadata(line, connection_gutter, line_meta);
						text_edit->set_line_gutter_icon(line, connection_gutter, get_parent_control()->get_editor_theme_icon(SNAME("Slot")));
						text_edit->set_line_gutter_clickable(line, connection_gutter, true);
						methods_found.insert(method);
						break;
					}
				}

				if (line >= 0) {
					continue;
				}

				// There is a chance that the method is inherited from another script.
				bool found_inherited_function = false;
				Ref<Script> inherited_script = script->get_base_script();
				while (inherited_script.is_valid()) {
					if (inherited_script->has_method(method)) {
						found_inherited_function = true;
						break;
					}

					inherited_script = inherited_script->get_base_script();
				}

				if (!found_inherited_function) {
					missing_connections.push_back(connection);
				}
			}
		}
	}

	// Add override icons to methods.
	methods_found.clear();
	for (int i = 0; i < functions.size(); i++) {
		String raw_name = functions[i].get_slicec(':', 0);
		StringName name = StringName(raw_name);
		if (methods_found.has(name)) {
			continue;
		}

		// Account for inner classes by stripping the class names from the method,
		// starting from the right since our inner class might be inside of another inner class.
		int pos = raw_name.rfind_char('.');
		if (pos != -1) {
			name = raw_name.substr(pos + 1);
		}

		String found_base_class;
		StringName base_class = script->get_instance_base_type();
		Ref<Script> inherited_script = script->get_base_script();
		while (inherited_script.is_valid()) {
			if (inherited_script->has_method(name)) {
				found_base_class = "script:" + inherited_script->get_path();
				break;
			}

			base_class = inherited_script->get_instance_base_type();
			inherited_script = inherited_script->get_base_script();
		}

		if (found_base_class.is_empty()) {
			while (base_class) {
				List<MethodInfo> methods;
				ClassDB::get_method_list(base_class, &methods, true);
				for (const MethodInfo &mi : methods) {
					if (mi.name == name) {
						found_base_class = "builtin:" + base_class;
						break;
					}
				}

				ClassDB::ClassInfo *base_class_ptr = ClassDB::classes.getptr(base_class)->inherits_ptr;
				if (base_class_ptr == nullptr) {
					break;
				}
				base_class = base_class_ptr->name;
			}
		}

		if (!found_base_class.is_empty()) {
			int line = functions[i].get_slicec(':', 1).to_int() - 1;

			Dictionary line_meta = text_edit->get_line_gutter_metadata(line, connection_gutter);
			if (line_meta.is_empty()) {
				// Add override icon to gutter.
				line_meta["type"] = "inherits";
				line_meta["method"] = name;
				line_meta["base_class"] = found_base_class;
				text_edit->set_line_gutter_icon(line, connection_gutter, get_parent_control()->get_editor_theme_icon(SNAME("MethodOverride")));
				text_edit->set_line_gutter_clickable(line, connection_gutter, true);
			} else {
				// If method is also connected to signal, then merge icons and keep the click behavior of the slot.
				text_edit->set_line_gutter_icon(line, connection_gutter, get_parent_control()->get_editor_theme_icon(SNAME("MethodOverrideAndSlot")));
			}

			methods_found.insert(StringName(raw_name));
		}
	}
}

void ScriptTextEditor::_update_gutter_indexes() {
	for (int i = 0; i < code_editor->get_text_editor()->get_gutter_count(); i++) {
		if (code_editor->get_text_editor()->get_gutter_name(i) == "connection_gutter") {
			connection_gutter = i;
			continue;
		}

		if (code_editor->get_text_editor()->get_gutter_name(i) == "line_numbers") {
			line_number_gutter = i;
			continue;
		}
	}
}

void ScriptTextEditor::_gutter_clicked(int p_line, int p_gutter) {
	if (p_gutter != connection_gutter) {
		return;
	}

	Dictionary meta = code_editor->get_text_editor()->get_line_gutter_metadata(p_line, p_gutter);
	String type = meta.get("type", "");
	if (type.is_empty()) {
		return;
	}

	// All types currently need a method name.
	String method = meta.get("method", "");
	if (method.is_empty()) {
		return;
	}

	if (type == "connection") {
		Node *base = get_tree()->get_edited_scene_root();
		if (!base) {
			return;
		}

		Vector<Node *> nodes = _find_all_node_for_script(base, base, script);
		connection_info_dialog->popup_connections(method, nodes);
	} else if (type == "inherits") {
		String base_class_raw = meta["base_class"];
		PackedStringArray base_class_split = base_class_raw.split(":", true, 1);

		if (base_class_split[0] == "script") {
			// Go to function declaration.
			Ref<Script> base_script = ResourceLoader::load(base_class_split[1]);
			ERR_FAIL_COND(base_script.is_null());
			emit_signal(SNAME("go_to_method"), base_script, method);
		} else if (base_class_split[0] == "builtin") {
			// Open method documentation.
			emit_signal(SNAME("go_to_help"), "class_method:" + base_class_split[1] + ":" + method);
		}
	}
}

void ScriptTextEditor::_edit_option(int p_op) {
	CodeEdit *tx = code_editor->get_text_editor();
	tx->apply_ime();

	switch (p_op) {
		case EDIT_UNDO: {
			tx->undo();
			callable_mp((Control *)tx, &Control::grab_focus).call_deferred(false);
		} break;
		case EDIT_REDO: {
			tx->redo();
			callable_mp((Control *)tx, &Control::grab_focus).call_deferred(false);
		} break;
		case EDIT_CUT: {
			tx->cut();
			callable_mp((Control *)tx, &Control::grab_focus).call_deferred(false);
		} break;
		case EDIT_COPY: {
			tx->copy();
			callable_mp((Control *)tx, &Control::grab_focus).call_deferred(false);
		} break;
		case EDIT_PASTE: {
			tx->paste();
			callable_mp((Control *)tx, &Control::grab_focus).call_deferred(false);
		} break;
		case EDIT_SELECT_ALL: {
			tx->select_all();
			callable_mp((Control *)tx, &Control::grab_focus).call_deferred(false);
		} break;
		case EDIT_MOVE_LINE_UP: {
			code_editor->get_text_editor()->move_lines_up();
		} break;
		case EDIT_MOVE_LINE_DOWN: {
			code_editor->get_text_editor()->move_lines_down();
		} break;
		case EDIT_INDENT: {
			Ref<Script> scr = script;
			if (scr.is_null()) {
				return;
			}
			tx->indent_lines();
		} break;
		case EDIT_UNINDENT: {
			Ref<Script> scr = script;
			if (scr.is_null()) {
				return;
			}
			tx->unindent_lines();
		} break;
		case EDIT_DELETE_LINE: {
			code_editor->get_text_editor()->delete_lines();
		} break;
		case EDIT_DUPLICATE_SELECTION: {
			code_editor->get_text_editor()->duplicate_selection();
		} break;
		case EDIT_DUPLICATE_LINES: {
			code_editor->get_text_editor()->duplicate_lines();
		} break;
		case EDIT_TOGGLE_FOLD_LINE: {
			tx->toggle_foldable_lines_at_carets();
		} break;
		case EDIT_FOLD_ALL_LINES: {
			tx->fold_all_lines();
		} break;
		case EDIT_UNFOLD_ALL_LINES: {
			tx->unfold_all_lines();
		} break;
		case EDIT_CREATE_CODE_REGION: {
			tx->create_code_region();
		} break;
		case EDIT_TOGGLE_COMMENT: {
			_edit_option_toggle_inline_comment();
		} break;
		case EDIT_COMPLETE: {
			tx->request_code_completion(true);
		} break;
		case EDIT_AUTO_INDENT: {
			String text = tx->get_text();
			Ref<Script> scr = script;
			if (scr.is_null()) {
				return;
			}

			tx->begin_complex_operation();
			tx->begin_multicaret_edit();
			int begin = tx->get_line_count() - 1, end = 0;
			if (tx->has_selection()) {
				// Auto indent all lines that have a caret or selection on it.
				Vector<Point2i> line_ranges = tx->get_line_ranges_from_carets();
				for (Point2i line_range : line_ranges) {
					scr->get_language()->auto_indent_code(text, line_range.x, line_range.y);
					if (line_range.x < begin) {
						begin = line_range.x;
					}
					if (line_range.y > end) {
						end = line_range.y;
					}
				}
			} else {
				// Auto indent entire text.
				begin = 0;
				end = tx->get_line_count() - 1;
				scr->get_language()->auto_indent_code(text, begin, end);
			}

			// Apply auto indented code.
			Vector<String> lines = text.split("\n");
			for (int i = begin; i <= end; ++i) {
				tx->set_line(i, lines[i]);
			}

			tx->end_multicaret_edit();
			tx->end_complex_operation();
		} break;
		case EDIT_TRIM_TRAILING_WHITESAPCE: {
			trim_trailing_whitespace();
		} break;
		case EDIT_TRIM_FINAL_NEWLINES: {
			trim_final_newlines();
		} break;
		case EDIT_CONVERT_INDENT_TO_SPACES: {
			code_editor->set_indent_using_spaces(true);
			convert_indent();
		} break;
		case EDIT_CONVERT_INDENT_TO_TABS: {
			code_editor->set_indent_using_spaces(false);
			convert_indent();
		} break;
		case EDIT_PICK_COLOR: {
			color_panel->popup();
		} break;
		case EDIT_TO_UPPERCASE: {
			_convert_case(CodeTextEditor::UPPER);
		} break;
		case EDIT_TO_LOWERCASE: {
			_convert_case(CodeTextEditor::LOWER);
		} break;
		case EDIT_CAPITALIZE: {
			_convert_case(CodeTextEditor::CAPITALIZE);
		} break;
		case EDIT_EVALUATE: {
			Expression expression;
			tx->begin_complex_operation();
			for (int caret_idx = 0; caret_idx < tx->get_caret_count(); caret_idx++) {
				Vector<String> lines = tx->get_selected_text(caret_idx).split("\n");
				PackedStringArray results;

				for (int i = 0; i < lines.size(); i++) {
					const String &line = lines[i];
					String whitespace = line.substr(0, line.size() - line.strip_edges(true, false).size()); // Extract the whitespace at the beginning.
					if (expression.parse(line) == OK) {
						Variant result = expression.execute(Array(), Variant(), false, true);
						if (expression.get_error_text().is_empty()) {
							results.push_back(whitespace + result.get_construct_string());
						} else {
							results.push_back(line);
						}
					} else {
						results.push_back(line);
					}
				}
				tx->insert_text_at_caret(String("\n").join(results), caret_idx);
			}
			tx->end_complex_operation();
		} break;
		case EDIT_TOGGLE_WORD_WRAP: {
			TextEdit::LineWrappingMode wrap = code_editor->get_text_editor()->get_line_wrapping_mode();
			code_editor->get_text_editor()->set_line_wrapping_mode(wrap == TextEdit::LINE_WRAPPING_BOUNDARY ? TextEdit::LINE_WRAPPING_NONE : TextEdit::LINE_WRAPPING_BOUNDARY);
		} break;
		case SEARCH_FIND: {
			code_editor->get_find_replace_bar()->popup_search();
		} break;
		case SEARCH_FIND_NEXT: {
			code_editor->get_find_replace_bar()->search_next();
		} break;
		case SEARCH_FIND_PREV: {
			code_editor->get_find_replace_bar()->search_prev();
		} break;
		case SEARCH_REPLACE: {
			code_editor->get_find_replace_bar()->popup_replace();
		} break;
		case SEARCH_IN_FILES: {
			String selected_text = tx->get_selected_text();

			// Yep, because it doesn't make sense to instance this dialog for every single script open...
			// So this will be delegated to the ScriptEditor.
			emit_signal(SNAME("search_in_files_requested"), selected_text);
		} break;
		case REPLACE_IN_FILES: {
			String selected_text = tx->get_selected_text();

			emit_signal(SNAME("replace_in_files_requested"), selected_text);
		} break;
		case SEARCH_LOCATE_FUNCTION: {
			quick_open->popup_dialog(get_functions());
		} break;
		case SEARCH_GOTO_LINE: {
			goto_line_popup->popup_find_line(code_editor);
		} break;
		case BOOKMARK_TOGGLE: {
			code_editor->toggle_bookmark();
		} break;
		case BOOKMARK_GOTO_NEXT: {
			code_editor->goto_next_bookmark();
		} break;
		case BOOKMARK_GOTO_PREV: {
			code_editor->goto_prev_bookmark();
		} break;
		case BOOKMARK_REMOVE_ALL: {
			code_editor->remove_all_bookmarks();
		} break;
		case DEBUG_TOGGLE_BREAKPOINT: {
			Vector<int> sorted_carets = tx->get_sorted_carets();
			int last_line = -1;
			for (const int &c : sorted_carets) {
				int from = tx->get_selection_from_line(c);
				from += from == last_line ? 1 : 0;
				int to = tx->get_selection_to_line(c);
				if (to < from) {
					continue;
				}
				// Check first if there's any lines with breakpoints in the selection.
				bool selection_has_breakpoints = false;
				for (int line = from; line <= to; line++) {
					if (tx->is_line_breakpointed(line)) {
						selection_has_breakpoints = true;
						break;
					}
				}

				// Set breakpoint on caret or remove all bookmarks from the selection.
				if (!selection_has_breakpoints) {
					if (tx->get_caret_line(c) != last_line) {
						tx->set_line_as_breakpoint(tx->get_caret_line(c), true);
					}
				} else {
					for (int line = from; line <= to; line++) {
						tx->set_line_as_breakpoint(line, false);
					}
				}
				last_line = to;
			}
		} break;
		case DEBUG_REMOVE_ALL_BREAKPOINTS: {
			PackedInt32Array bpoints = tx->get_breakpointed_lines();

			for (int i = 0; i < bpoints.size(); i++) {
				int line = bpoints[i];
				bool dobreak = !tx->is_line_breakpointed(line);
				tx->set_line_as_breakpoint(line, dobreak);
				EditorDebuggerNode::get_singleton()->set_breakpoint(script->get_path(), line + 1, dobreak);
			}
		} break;
		case DEBUG_GOTO_NEXT_BREAKPOINT: {
			PackedInt32Array bpoints = tx->get_breakpointed_lines();
			if (bpoints.is_empty()) {
				return;
			}

			int current_line = tx->get_caret_line();
			int bpoint_idx = 0;
			if (current_line < (int)bpoints[bpoints.size() - 1]) {
				while (bpoint_idx < bpoints.size() && bpoints[bpoint_idx] <= current_line) {
					bpoint_idx++;
				}
			}
			code_editor->goto_line_centered(bpoints[bpoint_idx]);
		} break;
		case DEBUG_GOTO_PREV_BREAKPOINT: {
			PackedInt32Array bpoints = tx->get_breakpointed_lines();
			if (bpoints.is_empty()) {
				return;
			}

			int current_line = tx->get_caret_line();
			int bpoint_idx = bpoints.size() - 1;
			if (current_line > (int)bpoints[0]) {
				while (bpoint_idx >= 0 && bpoints[bpoint_idx] >= current_line) {
					bpoint_idx--;
				}
			}
			code_editor->goto_line_centered(bpoints[bpoint_idx]);
		} break;
		case HELP_CONTEXTUAL: {
			String text = tx->get_selected_text(0);
			if (text.is_empty()) {
				text = tx->get_word_under_caret(0);
			}
			if (!text.is_empty()) {
				emit_signal(SNAME("request_help"), text);
			}
		} break;
		case LOOKUP_SYMBOL: {
			String text = tx->get_word_under_caret(0);
			if (text.is_empty()) {
				text = tx->get_selected_text(0);
			}
			if (!text.is_empty()) {
				_lookup_symbol(text, tx->get_caret_line(0), tx->get_caret_column(0));
			}
		} break;
		case EDIT_EMOJI_AND_SYMBOL: {
			code_editor->get_text_editor()->show_emoji_and_symbol_picker();
		} break;
		case EDIT_JOIN_LINES: {
			code_editor->get_text_editor()->join_lines();
		} break;
		default: {
			if (p_op >= EditorContextMenuPlugin::BASE_ID) {
				EditorContextMenuPluginManager::get_singleton()->activate_custom_option(EditorContextMenuPlugin::CONTEXT_SLOT_SCRIPT_EDITOR_CODE, p_op, tx);
			}
		}
	}
}

void ScriptTextEditor::_edit_option_toggle_inline_comment() {
	if (script.is_null()) {
		return;
	}

	String delimiter = "#";

	for (const String &script_delimiter : script->get_language()->get_comment_delimiters()) {
		if (!script_delimiter.contains_char(' ')) {
			delimiter = script_delimiter;
			break;
		}
	}

	code_editor->toggle_inline_comment(delimiter);
}

void ScriptTextEditor::add_syntax_highlighter(Ref<EditorSyntaxHighlighter> p_highlighter) {
	ERR_FAIL_COND(p_highlighter.is_null());

	highlighters[p_highlighter->_get_name()] = p_highlighter;
	highlighter_menu->add_radio_check_item(p_highlighter->_get_name());
}

void ScriptTextEditor::set_syntax_highlighter(Ref<EditorSyntaxHighlighter> p_highlighter) {
	ERR_FAIL_COND(p_highlighter.is_null());

	HashMap<String, Ref<EditorSyntaxHighlighter>>::Iterator el = highlighters.begin();
	while (el) {
		int highlighter_index = highlighter_menu->get_item_idx_from_text(el->key);
		highlighter_menu->set_item_checked(highlighter_index, el->value == p_highlighter);
		++el;
	}

	CodeEdit *te = code_editor->get_text_editor();
	p_highlighter->_set_edited_resource(script);
	te->set_syntax_highlighter(p_highlighter);
}

void ScriptTextEditor::_change_syntax_highlighter(int p_idx) {
	set_syntax_highlighter(highlighters[highlighter_menu->get_item_text(p_idx)]);
}

void ScriptTextEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			if (is_ready() && is_visible_in_tree()) {
				_update_errors();
				_update_warnings();
			}
		} break;

		case NOTIFICATION_THEME_CHANGED:
			if (!editor_enabled) {
				break;
			}
			if (is_visible_in_tree()) {
				_update_warnings();
				_update_errors();
				_update_background_color();
			}
			[[fallthrough]];
		case NOTIFICATION_ENTER_TREE: {
			code_editor->get_text_editor()->set_gutter_width(connection_gutter, code_editor->get_text_editor()->get_line_height());
			Ref<Font> code_font = get_theme_font("font", "CodeEdit");
			inline_color_options->add_theme_font_override("font", code_font);
			inline_color_options->get_popup()->add_theme_font_override("font", code_font);
		} break;
	}
}

Control *ScriptTextEditor::get_edit_menu() {
	return edit_hb;
}

void ScriptTextEditor::clear_edit_menu() {
	if (editor_enabled) {
		memdelete(edit_hb);
	}
}

void ScriptTextEditor::set_find_replace_bar(FindReplaceBar *p_bar) {
	code_editor->set_find_replace_bar(p_bar);
}

void ScriptTextEditor::reload(bool p_soft) {
	CodeEdit *te = code_editor->get_text_editor();
	Ref<Script> scr = script;
	if (scr.is_null()) {
		return;
	}
	scr->set_source_code(te->get_text());
	bool soft = p_soft || ClassDB::is_parent_class(scr->get_instance_base_type(), "EditorPlugin"); // Always soft-reload editor plugins.

	scr->get_language()->reload_tool_script(scr, soft);
}

PackedInt32Array ScriptTextEditor::get_breakpoints() {
	return code_editor->get_text_editor()->get_breakpointed_lines();
}

void ScriptTextEditor::set_breakpoint(int p_line, bool p_enabled) {
	code_editor->get_text_editor()->set_line_as_breakpoint(p_line, p_enabled);
}

void ScriptTextEditor::clear_breakpoints() {
	code_editor->get_text_editor()->clear_breakpointed_lines();
}

void ScriptTextEditor::set_tooltip_request_func(const Callable &p_toolip_callback) {
	Variant args[1] = { this };
	const Variant *argp[] = { &args[0] };
	code_editor->get_text_editor()->set_tooltip_request_func(p_toolip_callback.bindp(argp, 1));
}

void ScriptTextEditor::set_debugger_active(bool p_active) {
}

Control *ScriptTextEditor::get_base_editor() const {
	return code_editor->get_text_editor();
}

CodeTextEditor *ScriptTextEditor::get_code_editor() const {
	return code_editor;
}

Variant ScriptTextEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	return Variant();
}

bool ScriptTextEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	Dictionary d = p_data;
	if (d.has("type") &&
			(String(d["type"]) == "resource" ||
					String(d["type"]) == "files" ||
					String(d["type"]) == "nodes" ||
					String(d["type"]) == "obj_property" ||
					String(d["type"]) == "files_and_dirs")) {
		return true;
	}

	return false;
}

static Node *_find_script_node(Node *p_current_node, const Ref<Script> &script) {
	if (p_current_node->get_script() == script) {
		return p_current_node;
	}

	for (int i = 0; i < p_current_node->get_child_count(); i++) {
		Node *n = _find_script_node(p_current_node->get_child(i), script);
		if (n) {
			return n;
		}
	}

	return nullptr;
}

static String _quote_drop_data(const String &str) {
	// This function prepares a string for being "dropped" into the script editor.
	// The string can be a resource path, node path or property name.

	const bool using_single_quotes = EDITOR_GET("text_editor/completion/use_single_quotes");

	String escaped = str.c_escape();

	// If string is double quoted, there is no need to escape single quotes.
	// We can revert the extra escaping added in c_escape().
	if (!using_single_quotes) {
		escaped = escaped.replace("\\'", "\'");
	}

	return escaped.quote(using_single_quotes ? "'" : "\"");
}

static String _get_dropped_resource_as_member(const Ref<Resource> &p_resource, bool p_create_field, bool p_allow_uid) {
	String path = p_resource->get_path();
	if (p_allow_uid) {
		ResourceUID::ID id = ResourceLoader::get_resource_uid(path);
		if (id != ResourceUID::INVALID_ID) {
			path = ResourceUID::get_singleton()->id_to_text(id);
		}
	}
	const bool is_script = ClassDB::is_parent_class(p_resource->get_class(), "Script");

	if (!p_create_field) {
		return vformat("preload(%s)", _quote_drop_data(path));
	}

	String variable_name = p_resource->get_name();
	if (variable_name.is_empty()) {
		variable_name = p_resource->get_path().get_file().get_basename();
	}

	if (is_script) {
		variable_name = variable_name.to_pascal_case().validate_unicode_identifier();
	} else {
		variable_name = variable_name.to_snake_case().to_upper().validate_unicode_identifier();
	}
	return vformat("const %s = preload(%s)", variable_name, _quote_drop_data(path));
}

String ScriptTextEditor::_get_dropped_resource_as_exported_member(const Ref<Resource> &p_resource, const Vector<ObjectID> &p_script_instance_obj_ids) {
	String variable_name = p_resource->get_name();
	if (variable_name.is_empty()) {
		variable_name = p_resource->get_path().get_file().get_basename();
	}

	variable_name = variable_name.to_snake_case().validate_unicode_identifier();

	StringName class_name = p_resource->get_class();
	Ref<Script> resource_script = p_resource->get_script();

	if (resource_script.is_valid()) {
		StringName global_resource_script_name = resource_script->get_global_name();
		if (!global_resource_script_name.is_empty()) {
			class_name = global_resource_script_name;
		}
	}

	for (ObjectID obj_id : p_script_instance_obj_ids) {
		pending_dragged_exports.push_back(DraggedExport{ obj_id, variable_name, p_resource, class_name });
	}

	return vformat("@export var %s: %s", variable_name, class_name);
}

void ScriptTextEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	Dictionary d = p_data;

	CodeEdit *te = code_editor->get_text_editor();
	Point2i pos = (p_point == Vector2(Math::INF, Math::INF)) ? Point2i(te->get_caret_line(0), te->get_caret_column(0)) : te->get_line_column_at_pos(p_point);
	int drop_at_line = pos.y;
	int drop_at_column = pos.x;
	int selection_index = te->get_selection_at_line_column(drop_at_line, drop_at_column);

	bool is_empty_line = false;
	if (selection_index >= 0) {
		// Dropped on a selection, it will be replaced.
		drop_at_line = te->get_selection_from_line(selection_index);
		drop_at_column = te->get_selection_from_column(selection_index);
		is_empty_line = drop_at_column <= te->get_first_non_whitespace_column(drop_at_line) && te->get_selection_to_column(selection_index) == te->get_line(te->get_selection_to_line(selection_index)).length();
	}

	Node *scene_root = get_tree()->get_edited_scene_root();

	const bool member_drop_modifier_pressed = Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL);
	const bool export_drop_modifier_pressed = Input::get_singleton()->is_key_pressed(Key::ALT);

	const bool allow_uid = Input::get_singleton()->is_key_pressed(Key::SHIFT) != bool(EDITOR_GET("text_editor/behavior/files/drop_preload_resources_as_uid"));
	const String &line = te->get_line(drop_at_line);

	if (selection_index < 0) {
		is_empty_line = line.is_empty() || te->get_first_non_whitespace_column(drop_at_line) == line.length();
	}

	String text_to_drop;
	bool add_new_line = false;

	const String type = d.get("type", "");
	if (type == "resource") {
		Ref<Resource> resource = d["resource"];
		if (resource.is_null()) {
			return;
		}

		const String &path = resource->get_path();
		if (path.is_empty() || path.ends_with("::")) {
			String warning = TTR("The resource does not have a valid path because it has not been saved.\nPlease save the scene or resource that contains this resource and try again.");
			EditorToaster::get_singleton()->popup_str(warning, EditorToaster::SEVERITY_ERROR);
			return;
		}

		if (member_drop_modifier_pressed) {
			if (resource->is_built_in()) {
				String warning = TTR("Preloading internal resources is not supported.");
				EditorToaster::get_singleton()->popup_str(warning, EditorToaster::SEVERITY_ERROR);
			} else {
				text_to_drop = _get_dropped_resource_as_member(resource, is_empty_line, allow_uid);
			}
		} else if (export_drop_modifier_pressed) {
			Vector<ObjectID> obj_ids = _get_objects_for_export_assignment();
			text_to_drop = _get_dropped_resource_as_exported_member(resource, obj_ids);

		} else {
			text_to_drop = _quote_drop_data(path);
		}

		if (is_empty_line) {
			text_to_drop += "\n";
		}
	}

	if (type == "files" || type == "files_and_dirs") {
		const PackedStringArray files = d["files"];
		PackedStringArray parts;

		for (const String &path : files) {
			if ((member_drop_modifier_pressed || export_drop_modifier_pressed) && ResourceLoader::exists(path)) {
				Ref<Resource> resource = ResourceLoader::load(path);
				if (resource.is_null()) {
					// Resource exists, but failed to load. We need only path and name, so we can use a dummy Resource instead.
					resource.instantiate();
					resource->set_path_cache(path);
				}

				if (member_drop_modifier_pressed) {
					parts.append(_get_dropped_resource_as_member(resource, is_empty_line, allow_uid));
				} else if (export_drop_modifier_pressed) {
					Vector<ObjectID> obj_ids = _get_objects_for_export_assignment();
					parts.append(_get_dropped_resource_as_exported_member(resource, obj_ids));
				}
			} else {
				parts.append(_quote_drop_data(path));
			}
		}
		String join_string;
		if (is_empty_line) {
			int indent_level = te->get_indent_level(drop_at_line);
			if (te->is_indent_using_spaces()) {
				join_string = "\n" + String(" ").repeat(indent_level);
			} else {
				join_string = "\n" + String("\t").repeat(indent_level / te->get_tab_size());
			}
		} else {
			join_string = ", ";
		}
		text_to_drop = join_string.join(parts);
		if (is_empty_line) {
			text_to_drop += join_string;
		}
	}

	if (type == "nodes") {
		if (!scene_root) {
			EditorNode::get_singleton()->show_warning(TTR("Can't drop nodes without an open scene."));
			return;
		}

		if (!ClassDB::is_parent_class(script->get_instance_base_type(), "Node")) {
			EditorToaster::get_singleton()->popup_str(vformat(TTR("Can't drop nodes because script '%s' does not inherit Node."), get_name()), EditorToaster::SEVERITY_WARNING);
			return;
		}

		Node *sn = _find_script_node(scene_root, script);
		if (!sn) {
			sn = scene_root;
		}

		Array nodes = d["nodes"];

		if (member_drop_modifier_pressed) {
			const bool use_type = EDITOR_GET("text_editor/completion/add_type_hints");
			add_new_line = !is_empty_line && drop_at_column != 0;

			for (int i = 0; i < nodes.size(); i++) {
				NodePath np = nodes[i];
				Node *node = get_node(np);
				if (!node) {
					continue;
				}

				bool is_unique = node->is_unique_name_in_owner() && (node->get_owner() == sn || node->get_owner() == sn->get_owner());
				String path = is_unique ? String(node->get_name()) : String(sn->get_path_to(node));
				for (const String &segment : path.split("/")) {
					if (!segment.is_valid_unicode_identifier()) {
						path = _quote_drop_data(path);
						break;
					}
				}

				String variable_name = String(node->get_name()).to_snake_case().validate_unicode_identifier();
				if (use_type) {
					StringName class_name = node->get_class_name();
					Ref<Script> node_script = node->get_script();
					if (node_script.is_valid()) {
						StringName global_node_script_name = node_script->get_global_name();
						if (!global_node_script_name.is_empty()) {
							class_name = global_node_script_name;
						}
					}
					text_to_drop += vformat("@onready var %s: %s = %c%s", variable_name, class_name, is_unique ? '%' : '$', path);
				} else {
					text_to_drop += vformat("@onready var %s = %c%s", variable_name, is_unique ? '%' : '$', path);
				}
				if (i < nodes.size() - 1) {
					text_to_drop += "\n";
				}
			}

			if (is_empty_line || drop_at_column == 0) {
				text_to_drop += "\n";
			}
		} else if (export_drop_modifier_pressed) {
			Vector<ObjectID> obj_ids = _get_objects_for_export_assignment();

			for (int i = 0; i < nodes.size(); i++) {
				NodePath np = nodes[i];
				Node *node = get_node(np);
				if (!node) {
					continue;
				}

				String variable_name = String(node->get_name()).to_snake_case().validate_unicode_identifier();
				StringName class_name = node->get_class_name();
				Ref<Script> node_script = node->get_script();
				if (node_script.is_valid()) {
					StringName global_node_script_name = node_script->get_global_name();
					if (!global_node_script_name.is_empty()) {
						class_name = global_node_script_name;
					}
				}

				text_to_drop += vformat("@export var %s: %s\n", variable_name, class_name);
				for (ObjectID obj_id : obj_ids) {
					pending_dragged_exports.push_back(DraggedExport{ obj_id, variable_name, node, class_name });
				}
			}
		} else {
			for (int i = 0; i < nodes.size(); i++) {
				if (i > 0) {
					text_to_drop += ", ";
				}

				NodePath np = nodes[i];
				Node *node = get_node(np);
				if (!node) {
					continue;
				}

				bool is_unique = node->is_unique_name_in_owner() && (node->get_owner() == sn || node->get_owner() == sn->get_owner());
				String path = is_unique ? String(node->get_name()) : String(sn->get_path_to(node));
				for (const String &segment : path.split("/")) {
					if (!segment.is_valid_ascii_identifier()) {
						path = _quote_drop_data(path);
						break;
					}
				}
				text_to_drop += (is_unique ? "%" : "$") + path;
			}
		}
	}

	if (type == "obj_property") {
		bool add_literal = EDITOR_GET("text_editor/completion/add_node_path_literals");
		text_to_drop = add_literal ? "^" : "";
		// It is unclear whether properties may contain single or double quotes.
		// Assume here that double-quotes may not exist. We are escaping single-quotes if necessary.
		text_to_drop += _quote_drop_data(String(d["property"]));
	}

	if (text_to_drop.is_empty()) {
		return;
	}

	// Remove drag caret before any actions so it is not included in undo.
	te->remove_drag_caret();
	te->begin_complex_operation();
	if (selection_index >= 0) {
		te->delete_selection(selection_index);
	}
	te->remove_secondary_carets();
	te->deselect();
	te->set_caret_line(drop_at_line);
	if (add_new_line) {
		te->set_caret_column(te->get_line(drop_at_line).length());
		text_to_drop = "\n" + text_to_drop;
	} else {
		te->set_caret_column(drop_at_column);
	}
	te->insert_text_at_caret(text_to_drop);
	te->end_complex_operation();
	te->grab_focus();
}

Vector<ObjectID> ScriptTextEditor::_get_objects_for_export_assignment() const {
	Vector<ObjectID> objects;
	Node *scene_root = get_tree()->get_edited_scene_root();
	bool assign_export_variables = scene_root && ClassDB::is_parent_class(script->get_instance_base_type(), "Node");

	if (!assign_export_variables) {
		return objects;
	}

	EditorInspector *inspector = EditorInterface::get_singleton()->get_inspector();
	if (inspector) {
		Object *edited_object = inspector->get_edited_object();
		Node *node_edit = Object::cast_to<Node>(edited_object);
		MultiNodeEdit *multi_node_edit = Object::cast_to<MultiNodeEdit>(edited_object);

		if (node_edit != nullptr) {
			if (node_edit->get_script() == script) {
				objects.push_back(node_edit->get_instance_id());
			}
		} else if (multi_node_edit != nullptr) {
			Node *es = EditorNode::get_singleton()->get_edited_scene();
			for (int i = 0; i < multi_node_edit->get_node_count(); i++) {
				NodePath np = multi_node_edit->get_node(i);
				Node *node = es->get_node(np);
				if (node->get_script() == script) {
					objects.push_back(node->get_instance_id());
				}
			}
		}
	}

	// In case there is no current editor selection/editor selection does not contain this script,
	// it often still makes sense to try to assign the export variable,
	// so we default to the first node with the script we find in the scene.
	if (objects.is_empty()) {
		Node *sn = _find_script_node(scene_root, script);
		if (sn) {
			objects.push_back(sn->get_instance_id());
		}
	}

	return objects;
}

void ScriptTextEditor::_assign_dragged_export_variables() {
	ERR_FAIL_COND(pending_dragged_exports.is_empty());

	bool export_variable_set = false;

	for (int i = pending_dragged_exports.size() - 1; i >= 0; i--) {
		const DraggedExport &dragged_export = pending_dragged_exports[i];
		Object *obj = ObjectDB::get_instance(dragged_export.obj_id);
		if (!obj) {
			WARN_PRINT("Object not found, can't assign export variable.");
			pending_dragged_exports.remove_at(i);
			continue;
		}

		ScriptInstance *si = obj->get_script_instance();
		if (!si) {
			WARN_PRINT("Script on " + obj->to_string() + " does not exist anymore, can't assign export variable.");
			pending_dragged_exports.remove_at(i);
			continue;
		}

		bool script_has_errors = false;
		String scr_path = si->get_script()->get_path();

		for (const ScriptLanguage::ScriptError &error : errors) {
			if (error.path == scr_path) {
				script_has_errors = true;
				break;
			}
		}

		if (!script_has_errors) {
			bool success = false;
			List<PropertyInfo> properties;
			si->get_property_list(&properties);
			for (const PropertyInfo &pi : properties) {
				if (pi.name == dragged_export.variable_name && pi.hint_string == dragged_export.class_name) {
					success = si->set(dragged_export.variable_name, dragged_export.value);
					break;
				}
			}

			if (success) {
				export_variable_set = true;
			}
			pending_dragged_exports.remove_at(i);
		}
	}

	if (export_variable_set) {
		EditorInterface::get_singleton()->mark_scene_as_unsaved();
	}
}

void ScriptTextEditor::_text_edit_gui_input(const Ref<InputEvent> &ev) {
	Ref<InputEventMouseButton> mb = ev;
	Ref<InputEventKey> k = ev;
	Point2 local_pos;
	bool create_menu = false;

	CodeEdit *tx = code_editor->get_text_editor();
	if (mb.is_valid() && mb->get_button_index() == MouseButton::RIGHT && mb->is_pressed()) {
		local_pos = mb->get_global_position() - tx->get_global_position();
		create_menu = true;
	} else if (k.is_valid() && k->is_action("ui_menu", true)) {
		tx->adjust_viewport_to_caret(0);
		local_pos = tx->get_caret_draw_pos(0);
		create_menu = true;
	}

	if (create_menu) {
		tx->apply_ime();

		Point2i pos = tx->get_line_column_at_pos(local_pos);
		int mouse_line = pos.y;
		int mouse_column = pos.x;

		tx->set_move_caret_on_right_click_enabled(EDITOR_GET("text_editor/behavior/navigation/move_caret_on_right_click"));
		int selection_clicked = -1;
		if (tx->is_move_caret_on_right_click_enabled()) {
			selection_clicked = tx->get_selection_at_line_column(mouse_line, mouse_column, true);
			if (selection_clicked < 0) {
				tx->deselect();
				tx->remove_secondary_carets();
				selection_clicked = 0;
				tx->set_caret_line(mouse_line, false, false, -1);
				tx->set_caret_column(mouse_column);
			}
		}

		String word_at_pos = tx->get_lookup_word(mouse_line, mouse_column);
		if (word_at_pos.is_empty()) {
			word_at_pos = tx->get_word_under_caret(selection_clicked);
		}
		if (word_at_pos.is_empty()) {
			word_at_pos = tx->get_selected_text(selection_clicked);
		}

		bool has_color = (word_at_pos == "Color");
		bool foldable = tx->can_fold_line(mouse_line) || tx->is_line_folded(mouse_line);
		bool open_docs = false;
		bool goto_definition = false;

		if (ScriptServer::is_global_class(word_at_pos) || word_at_pos.is_resource_file()) {
			open_docs = true;
		} else {
			Node *base = get_tree()->get_edited_scene_root();
			if (base) {
				base = _find_node_for_script(base, base, script);
			}
			ScriptLanguage::LookupResult result;
			if (script->get_language()->lookup_code(tx->get_text_for_symbol_lookup(), word_at_pos, script->get_path(), base, result) == OK) {
				open_docs = true;
			}
		}

		if (has_color) {
			String line = tx->get_line(mouse_line);
			color_position.x = mouse_line;

			int begin = -1;
			int end = -1;
			enum EXPRESSION_PATTERNS {
				NOT_PARSED,
				RGBA_PARAMETER, // Color(float,float,float) or Color(float,float,float,float)
				COLOR_NAME, // Color.COLOR_NAME
			} expression_pattern = NOT_PARSED;

			for (int i = mouse_column; i < line.length(); i++) {
				if (line[i] == '(') {
					if (expression_pattern == NOT_PARSED) {
						begin = i;
						expression_pattern = RGBA_PARAMETER;
					} else {
						// Method call or '(' appearing twice.
						expression_pattern = NOT_PARSED;

						break;
					}
				} else if (expression_pattern == RGBA_PARAMETER && line[i] == ')' && end < 0) {
					end = i + 1;

					break;
				} else if (expression_pattern == NOT_PARSED && line[i] == '.') {
					begin = i;
					expression_pattern = COLOR_NAME;
				} else if (expression_pattern == COLOR_NAME && end < 0 && (line[i] == ' ' || line[i] == '\t')) {
					// Including '.' and spaces.
					continue;
				} else if (expression_pattern == COLOR_NAME && !(line[i] == '_' || ('A' <= line[i] && line[i] <= 'Z'))) {
					end = i;

					break;
				}
			}

			switch (expression_pattern) {
				case RGBA_PARAMETER: {
					color_args = line.substr(begin, end - begin);
					String stripped = color_args.remove_chars(" \t()");
					PackedFloat64Array color = stripped.split_floats(",");
					if (color.size() > 2) {
						float alpha = color.size() > 3 ? color[3] : 1.0f;
						color_picker->set_pick_color(Color(color[0], color[1], color[2], alpha));
					}
				} break;
				case COLOR_NAME: {
					if (end < 0) {
						end = line.length();
					}
					color_args = line.substr(begin, end - begin);
					const String color_name = color_args.remove_chars(" \t.");
					const int color_index = Color::find_named_color(color_name);
					if (0 <= color_index) {
						const Color color_constant = Color::get_named_color(color_index);
						color_picker->set_pick_color(color_constant);
					} else {
						has_color = false;
					}
				} break;
				default:
					has_color = false;
					break;
			}
			if (has_color) {
				color_panel->set_position(get_screen_position() + local_pos);
				color_position.y = begin;
				color_position.z = end;
			}
		}
		_make_context_menu(tx->has_selection(), has_color, foldable, open_docs, goto_definition, local_pos);
	}
}

void ScriptTextEditor::_color_changed(const Color &p_color) {
	String new_args;
	const int decimals = 3;
	if (p_color.a == 1.0f) {
		new_args = String("(" + String::num(p_color.r, decimals) + ", " + String::num(p_color.g, decimals) + ", " + String::num(p_color.b, decimals) + ")");
	} else {
		new_args = String("(" + String::num(p_color.r, decimals) + ", " + String::num(p_color.g, decimals) + ", " + String::num(p_color.b, decimals) + ", " + String::num(p_color.a, decimals) + ")");
	}

	String line = code_editor->get_text_editor()->get_line(color_position.x);
	String line_with_replaced_args = line.substr(0, color_position.y) + line.substr(color_position.y, color_position.z - color_position.y).replace(color_args, new_args) + line.substr(color_position.z);

	color_args = new_args;
	code_editor->get_text_editor()->begin_complex_operation();
	code_editor->get_text_editor()->set_line(color_position.x, line_with_replaced_args);
	code_editor->get_text_editor()->end_complex_operation();
}

void ScriptTextEditor::_prepare_edit_menu() {
	const CodeEdit *tx = code_editor->get_text_editor();
	PopupMenu *popup = edit_menu->get_popup();
	popup->set_item_disabled(popup->get_item_index(EDIT_UNDO), !tx->has_undo());
	popup->set_item_disabled(popup->get_item_index(EDIT_REDO), !tx->has_redo());
}

void ScriptTextEditor::_make_context_menu(bool p_selection, bool p_color, bool p_foldable, bool p_open_docs, bool p_goto_definition, Vector2 p_pos) {
	context_menu->clear();
	if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_EMOJI_AND_SYMBOL_PICKER)) {
		context_menu->add_item(TTRC("Emoji & Symbols"), EDIT_EMOJI_AND_SYMBOL);
		context_menu->add_separator();
	}
	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_undo"), EDIT_UNDO);
	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_redo"), EDIT_REDO);

	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_cut"), EDIT_CUT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_copy"), EDIT_COPY);
	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_paste"), EDIT_PASTE);

	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("ui_text_select_all"), EDIT_SELECT_ALL);

	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent"), EDIT_INDENT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/unindent"), EDIT_UNINDENT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_bookmark"), BOOKMARK_TOGGLE);

	if (p_selection) {
		context_menu->add_separator();
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/convert_to_uppercase"), EDIT_TO_UPPERCASE);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/convert_to_lowercase"), EDIT_TO_LOWERCASE);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/evaluate_selection"), EDIT_EVALUATE);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/create_code_region"), EDIT_CREATE_CODE_REGION);
	}
	if (p_foldable) {
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_fold_line"), EDIT_TOGGLE_FOLD_LINE);
	}

	if (p_color || p_open_docs || p_goto_definition) {
		context_menu->add_separator();
		if (p_open_docs) {
			context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_symbol"), LOOKUP_SYMBOL);
		}
		if (p_color) {
			context_menu->add_item(TTRC("Pick Color"), EDIT_PICK_COLOR);
		}
	}

	const PackedStringArray paths = { String(code_editor->get_text_editor()->get_path()) };
	EditorContextMenuPluginManager::get_singleton()->add_options_from_plugins(context_menu, EditorContextMenuPlugin::CONTEXT_SLOT_SCRIPT_EDITOR_CODE, paths);

	const CodeEdit *tx = code_editor->get_text_editor();
	context_menu->set_item_disabled(context_menu->get_item_index(EDIT_UNDO), !tx->has_undo());
	context_menu->set_item_disabled(context_menu->get_item_index(EDIT_REDO), !tx->has_redo());

	context_menu->set_position(get_screen_position() + p_pos);
	context_menu->reset_size();
	context_menu->popup();
}

void ScriptTextEditor::_enable_code_editor() {
	ERR_FAIL_COND(code_editor->get_parent());

	VSplitContainer *editor_box = memnew(VSplitContainer);
	add_child(editor_box);
	editor_box->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	editor_box->set_v_size_flags(SIZE_EXPAND_FILL);

	editor_box->add_child(code_editor);
	code_editor->connect("show_errors_panel", callable_mp(this, &ScriptTextEditor::_show_errors_panel));
	code_editor->connect("show_warnings_panel", callable_mp(this, &ScriptTextEditor::_show_warnings_panel));
	code_editor->connect("validate_script", callable_mp(this, &ScriptTextEditor::_validate_script));
	code_editor->connect("load_theme_settings", callable_mp(this, &ScriptTextEditor::_load_theme_settings));
	code_editor->get_text_editor()->connect("symbol_lookup", callable_mp(this, &ScriptTextEditor::_lookup_symbol));
	code_editor->get_text_editor()->connect("symbol_hovered", callable_mp(this, &ScriptTextEditor::_show_symbol_tooltip));
	code_editor->get_text_editor()->connect("symbol_validate", callable_mp(this, &ScriptTextEditor::_validate_symbol));
	code_editor->get_text_editor()->connect("gutter_added", callable_mp(this, &ScriptTextEditor::_update_gutter_indexes));
	code_editor->get_text_editor()->connect("gutter_removed", callable_mp(this, &ScriptTextEditor::_update_gutter_indexes));
	code_editor->get_text_editor()->connect("gutter_clicked", callable_mp(this, &ScriptTextEditor::_gutter_clicked));
	code_editor->get_text_editor()->connect("_fold_line_updated", callable_mp(this, &ScriptTextEditor::_update_background_color));
	code_editor->get_text_editor()->connect(SceneStringName(gui_input), callable_mp(this, &ScriptTextEditor::_text_edit_gui_input));
	code_editor->show_toggle_files_button();
	_update_gutter_indexes();

	editor_box->add_child(warnings_panel);
	warnings_panel->connect("meta_clicked", callable_mp(this, &ScriptTextEditor::_warning_clicked));

	editor_box->add_child(errors_panel);
	errors_panel->connect("meta_clicked", callable_mp(this, &ScriptTextEditor::_error_clicked));

	add_child(context_menu);
	context_menu->connect(SceneStringName(id_pressed), callable_mp(this, &ScriptTextEditor::_edit_option));

	add_child(color_panel);

	color_picker = memnew(ColorPicker);
	color_picker->set_deferred_mode(true);
	color_picker->connect("color_changed", callable_mp(this, &ScriptTextEditor::_color_changed));
	color_panel->connect("about_to_popup", callable_mp(EditorNode::get_singleton(), &EditorNode::setup_color_picker).bind(color_picker));

	color_panel->add_child(color_picker);

	quick_open = memnew(ScriptEditorQuickOpen);
	quick_open->set_title(TTRC("Go to Function"));
	quick_open->connect("goto_line", callable_mp(this, &ScriptTextEditor::_goto_line));
	add_child(quick_open);

	goto_line_popup = memnew(GotoLinePopup);
	add_child(goto_line_popup);

	add_child(connection_info_dialog);

	edit_hb->add_child(edit_menu);
	edit_menu->connect("about_to_popup", callable_mp(this, &ScriptTextEditor::_prepare_edit_menu));
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_undo"), EDIT_UNDO);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_redo"), EDIT_REDO);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_cut"), EDIT_CUT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_copy"), EDIT_COPY);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_paste"), EDIT_PASTE);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_text_select_all"), EDIT_SELECT_ALL);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/duplicate_selection"), EDIT_DUPLICATE_SELECTION);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/duplicate_lines"), EDIT_DUPLICATE_LINES);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/evaluate_selection"), EDIT_EVALUATE);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_word_wrap"), EDIT_TOGGLE_WORD_WRAP);
	edit_menu->get_popup()->add_separator();
	{
		PopupMenu *sub_menu = memnew(PopupMenu);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_up"), EDIT_MOVE_LINE_UP);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_down"), EDIT_MOVE_LINE_DOWN);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent"), EDIT_INDENT);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/unindent"), EDIT_UNINDENT);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/delete_line"), EDIT_DELETE_LINE);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/join_lines"), EDIT_JOIN_LINES);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);
		sub_menu->connect(SceneStringName(id_pressed), callable_mp(this, &ScriptTextEditor::_edit_option));
		edit_menu->get_popup()->add_submenu_node_item(TTRC("Line"), sub_menu);
	}
	{
		PopupMenu *sub_menu = memnew(PopupMenu);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_fold_line"), EDIT_TOGGLE_FOLD_LINE);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/fold_all_lines"), EDIT_FOLD_ALL_LINES);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/unfold_all_lines"), EDIT_UNFOLD_ALL_LINES);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/create_code_region"), EDIT_CREATE_CODE_REGION);
		sub_menu->connect(SceneStringName(id_pressed), callable_mp(this, &ScriptTextEditor::_edit_option));
		edit_menu->get_popup()->add_submenu_node_item(TTRC("Folding"), sub_menu);
	}
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_text_completion_query"), EDIT_COMPLETE);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/trim_trailing_whitespace"), EDIT_TRIM_TRAILING_WHITESAPCE);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/trim_final_newlines"), EDIT_TRIM_FINAL_NEWLINES);
	{
		PopupMenu *sub_menu = memnew(PopupMenu);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/convert_indent_to_spaces"), EDIT_CONVERT_INDENT_TO_SPACES);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/convert_indent_to_tabs"), EDIT_CONVERT_INDENT_TO_TABS);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/auto_indent"), EDIT_AUTO_INDENT);
		sub_menu->connect(SceneStringName(id_pressed), callable_mp(this, &ScriptTextEditor::_edit_option));
		edit_menu->get_popup()->add_submenu_node_item(TTRC("Indentation"), sub_menu);
	}
	edit_menu->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &ScriptTextEditor::_edit_option));
	edit_menu->get_popup()->add_separator();
	{
		PopupMenu *sub_menu = memnew(PopupMenu);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/convert_to_uppercase"), EDIT_TO_UPPERCASE);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/convert_to_lowercase"), EDIT_TO_LOWERCASE);
		sub_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/capitalize"), EDIT_CAPITALIZE);
		sub_menu->connect(SceneStringName(id_pressed), callable_mp(this, &ScriptTextEditor::_edit_option));
		edit_menu->get_popup()->add_submenu_node_item(TTRC("Convert Case"), sub_menu);
	}
	edit_menu->get_popup()->add_submenu_node_item(TTRC("Syntax Highlighter"), highlighter_menu);
	highlighter_menu->connect(SceneStringName(id_pressed), callable_mp(this, &ScriptTextEditor::_change_syntax_highlighter));

	edit_hb->add_child(search_menu);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find"), SEARCH_FIND);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_next"), SEARCH_FIND_NEXT);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_previous"), SEARCH_FIND_PREV);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/replace"), SEARCH_REPLACE);
	search_menu->get_popup()->add_separator();
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("editor/find_in_files"), SEARCH_IN_FILES);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/replace_in_files"), REPLACE_IN_FILES);
	search_menu->get_popup()->add_separator();
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/contextual_help"), HELP_CONTEXTUAL);
	search_menu->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &ScriptTextEditor::_edit_option));

	_load_theme_settings();

	edit_hb->add_child(goto_menu);
	goto_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_function"), SEARCH_LOCATE_FUNCTION);
	goto_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_line"), SEARCH_GOTO_LINE);
	goto_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_symbol"), LOOKUP_SYMBOL);
	goto_menu->get_popup()->add_separator();

	goto_menu->get_popup()->add_submenu_node_item(TTRC("Bookmarks"), bookmarks_menu);
	_update_bookmark_list();
	bookmarks_menu->connect("about_to_popup", callable_mp(this, &ScriptTextEditor::_update_bookmark_list));
	bookmarks_menu->connect("index_pressed", callable_mp(this, &ScriptTextEditor::_bookmark_item_pressed));

	goto_menu->get_popup()->add_submenu_node_item(TTRC("Breakpoints"), breakpoints_menu);
	_update_breakpoint_list();
	breakpoints_menu->connect("about_to_popup", callable_mp(this, &ScriptTextEditor::_update_breakpoint_list));
	breakpoints_menu->connect("index_pressed", callable_mp(this, &ScriptTextEditor::_breakpoint_item_pressed));

	goto_menu->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &ScriptTextEditor::_edit_option));
}

ScriptTextEditor::ScriptTextEditor() {
	code_editor = memnew(CodeTextEditor);
	code_editor->set_toggle_list_control(ScriptEditor::get_singleton()->get_left_list_split());
	code_editor->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	code_editor->set_code_complete_func(_code_complete_scripts, this);
	code_editor->set_v_size_flags(SIZE_EXPAND_FILL);

	code_editor->get_text_editor()->set_draw_breakpoints_gutter(true);
	code_editor->get_text_editor()->set_draw_executing_lines_gutter(true);
	code_editor->get_text_editor()->connect("breakpoint_toggled", callable_mp(this, &ScriptTextEditor::_breakpoint_toggled));
	code_editor->get_text_editor()->connect("caret_changed", callable_mp(this, &ScriptTextEditor::_on_caret_moved));
	code_editor->connect("navigation_preview_ended", callable_mp(this, &ScriptTextEditor::_on_caret_moved));

	connection_gutter = 1;
	code_editor->get_text_editor()->add_gutter(connection_gutter);
	code_editor->get_text_editor()->set_gutter_name(connection_gutter, "connection_gutter");
	code_editor->get_text_editor()->set_gutter_draw(connection_gutter, false);
	code_editor->get_text_editor()->set_gutter_overwritable(connection_gutter, true);
	code_editor->get_text_editor()->set_gutter_type(connection_gutter, TextEdit::GUTTER_TYPE_ICON);

	warnings_panel = memnew(RichTextLabel);
	warnings_panel->set_custom_minimum_size(Size2(0, 100 * EDSCALE));
	warnings_panel->set_h_size_flags(SIZE_EXPAND_FILL);
	warnings_panel->set_meta_underline(true);
	warnings_panel->set_selection_enabled(true);
	warnings_panel->set_context_menu_enabled(true);
	warnings_panel->set_focus_mode(FOCUS_CLICK);
	warnings_panel->hide();

	errors_panel = memnew(RichTextLabel);
	errors_panel->set_custom_minimum_size(Size2(0, 100 * EDSCALE));
	errors_panel->set_h_size_flags(SIZE_EXPAND_FILL);
	errors_panel->set_meta_underline(true);
	errors_panel->set_selection_enabled(true);
	errors_panel->set_context_menu_enabled(true);
	errors_panel->set_focus_mode(FOCUS_CLICK);
	errors_panel->hide();

	update_settings();

	code_editor->get_text_editor()->set_symbol_lookup_on_click_enabled(true);
	code_editor->get_text_editor()->set_symbol_tooltip_on_hover_enabled(true);
	code_editor->get_text_editor()->set_context_menu_enabled(false);

	context_menu = memnew(PopupMenu);

	color_panel = memnew(PopupPanel);

	edit_hb = memnew(HBoxContainer);

	edit_menu = memnew(MenuButton);
	edit_menu->set_flat(false);
	edit_menu->set_theme_type_variation("FlatMenuButton");
	edit_menu->set_text(TTRC("Edit"));
	edit_menu->set_switch_on_hover(true);
	edit_menu->set_shortcut_context(this);

	highlighter_menu = memnew(PopupMenu);

	Ref<EditorPlainTextSyntaxHighlighter> plain_highlighter;
	plain_highlighter.instantiate();
	add_syntax_highlighter(plain_highlighter);

	Ref<EditorStandardSyntaxHighlighter> highlighter;
	highlighter.instantiate();
	add_syntax_highlighter(highlighter);
	set_syntax_highlighter(highlighter);

	search_menu = memnew(MenuButton);
	search_menu->set_flat(false);
	search_menu->set_theme_type_variation("FlatMenuButton");
	search_menu->set_text(TTRC("Search"));
	search_menu->set_switch_on_hover(true);
	search_menu->set_shortcut_context(this);

	goto_menu = memnew(MenuButton);
	goto_menu->set_flat(false);
	goto_menu->set_theme_type_variation("FlatMenuButton");
	goto_menu->set_text(TTRC("Go To"));
	goto_menu->set_switch_on_hover(true);
	goto_menu->set_shortcut_context(this);

	bookmarks_menu = memnew(PopupMenu);
	breakpoints_menu = memnew(PopupMenu);

	inline_color_popup = memnew(PopupPanel);
	add_child(inline_color_popup);

	inline_color_picker = memnew(ColorPicker);
	inline_color_picker->set_mouse_filter(MOUSE_FILTER_STOP);
	inline_color_picker->set_deferred_mode(true);
	inline_color_picker->set_hex_visible(false);
	inline_color_picker->connect("color_changed", callable_mp(this, &ScriptTextEditor::_picker_color_changed));
	inline_color_popup->add_child(inline_color_picker);

	inline_color_options = memnew(OptionButton);
	inline_color_options->set_h_size_flags(SIZE_FILL);
	inline_color_options->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	inline_color_options->set_fit_to_longest_item(false);
	inline_color_options->connect("item_selected", callable_mp(this, &ScriptTextEditor::_update_color_text).unbind(1));
	inline_color_picker->get_slider_container()->add_sibling(inline_color_options);

	connection_info_dialog = memnew(ConnectionInfoDialog);

	SET_DRAG_FORWARDING_GCD(code_editor->get_text_editor(), ScriptTextEditor);
}

ScriptTextEditor::~ScriptTextEditor() {
	highlighters.clear();

	if (!editor_enabled) {
		memdelete(code_editor);
		memdelete(warnings_panel);
		memdelete(errors_panel);
		memdelete(context_menu);
		memdelete(color_panel);
		memdelete(edit_hb);
		memdelete(edit_menu);
		memdelete(highlighter_menu);
		memdelete(search_menu);
		memdelete(goto_menu);
		memdelete(bookmarks_menu);
		memdelete(breakpoints_menu);
		memdelete(connection_info_dialog);
	}
}

static ScriptEditorBase *create_editor(const Ref<Resource> &p_resource) {
	if (Object::cast_to<Script>(*p_resource)) {
		return memnew(ScriptTextEditor);
	}
	return nullptr;
}

void ScriptTextEditor::register_editor() {
	ED_SHORTCUT("script_text_editor/move_up", TTRC("Move Up"), KeyModifierMask::ALT | Key::UP);
	ED_SHORTCUT("script_text_editor/move_down", TTRC("Move Down"), KeyModifierMask::ALT | Key::DOWN);
	ED_SHORTCUT("script_text_editor/delete_line", TTRC("Delete Line"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::K);
	ED_SHORTCUT("script_text_editor/join_lines", TTRC("Join Lines"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::J);

	// Leave these at zero, same can be accomplished with tab/shift-tab, including selection.
	// The next/previous in history shortcut in this case makes a lot more sense.

	ED_SHORTCUT("script_text_editor/indent", TTRC("Indent"), Key::NONE);
	ED_SHORTCUT("script_text_editor/unindent", TTRC("Unindent"), KeyModifierMask::SHIFT | Key::TAB);
	ED_SHORTCUT_ARRAY("script_text_editor/toggle_comment", TTRC("Toggle Comment"), { int32_t(KeyModifierMask::CMD_OR_CTRL | Key::K), int32_t(KeyModifierMask::CMD_OR_CTRL | Key::SLASH), int32_t(KeyModifierMask::CMD_OR_CTRL | Key::KP_DIVIDE), int32_t(KeyModifierMask::CMD_OR_CTRL | Key::NUMBERSIGN) });
	ED_SHORTCUT("script_text_editor/toggle_fold_line", TTRC("Fold/Unfold Line"), KeyModifierMask::ALT | Key::F);
	ED_SHORTCUT_OVERRIDE("script_text_editor/toggle_fold_line", "macos", KeyModifierMask::CTRL | KeyModifierMask::META | Key::F);
	ED_SHORTCUT("script_text_editor/fold_all_lines", TTRC("Fold All Lines"), Key::NONE);
	ED_SHORTCUT("script_text_editor/create_code_region", TTRC("Create Code Region"), KeyModifierMask::ALT | Key::R);
	ED_SHORTCUT("script_text_editor/unfold_all_lines", TTRC("Unfold All Lines"), Key::NONE);
	ED_SHORTCUT("script_text_editor/duplicate_selection", TTRC("Duplicate Selection"), KeyModifierMask::SHIFT | KeyModifierMask::CTRL | Key::D);
	ED_SHORTCUT_OVERRIDE("script_text_editor/duplicate_selection", "macos", KeyModifierMask::SHIFT | KeyModifierMask::META | Key::C);
	ED_SHORTCUT("script_text_editor/duplicate_lines", TTRC("Duplicate Lines"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::ALT | Key::DOWN);
	ED_SHORTCUT_OVERRIDE("script_text_editor/duplicate_lines", "macos", KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::DOWN);
	ED_SHORTCUT("script_text_editor/evaluate_selection", TTRC("Evaluate Selection"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::E);
	ED_SHORTCUT("script_text_editor/toggle_word_wrap", TTRC("Toggle Word Wrap"), KeyModifierMask::ALT | Key::Z);
	ED_SHORTCUT("script_text_editor/trim_trailing_whitespace", TTRC("Trim Trailing Whitespace"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::ALT | Key::T);
	ED_SHORTCUT("script_text_editor/trim_final_newlines", TTRC("Trim Final Newlines"), Key::NONE);
	ED_SHORTCUT("script_text_editor/convert_indent_to_spaces", TTRC("Convert Indent to Spaces"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::Y);
	ED_SHORTCUT("script_text_editor/convert_indent_to_tabs", TTRC("Convert Indent to Tabs"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::I);
	ED_SHORTCUT("script_text_editor/auto_indent", TTRC("Auto Indent"), KeyModifierMask::CMD_OR_CTRL | Key::I);

	ED_SHORTCUT_AND_COMMAND("script_text_editor/find", TTRC("Find..."), KeyModifierMask::CMD_OR_CTRL | Key::F);

	ED_SHORTCUT("script_text_editor/find_next", TTRC("Find Next"), Key::F3);
	ED_SHORTCUT_OVERRIDE("script_text_editor/find_next", "macos", KeyModifierMask::META | Key::G);

	ED_SHORTCUT("script_text_editor/find_previous", TTRC("Find Previous"), KeyModifierMask::SHIFT | Key::F3);
	ED_SHORTCUT_OVERRIDE("script_text_editor/find_previous", "macos", KeyModifierMask::META | KeyModifierMask::SHIFT | Key::G);

	ED_SHORTCUT_AND_COMMAND("script_text_editor/replace", TTRC("Replace..."), KeyModifierMask::CTRL | Key::R);
	ED_SHORTCUT_OVERRIDE("script_text_editor/replace", "macos", KeyModifierMask::ALT | KeyModifierMask::META | Key::F);

	ED_SHORTCUT("script_text_editor/replace_in_files", TTRC("Replace in Files..."), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::R);

	ED_SHORTCUT("script_text_editor/contextual_help", TTRC("Contextual Help"), KeyModifierMask::ALT | Key::F1);
	ED_SHORTCUT_OVERRIDE("script_text_editor/contextual_help", "macos", KeyModifierMask::ALT | KeyModifierMask::SHIFT | Key::SPACE);

	ED_SHORTCUT("script_text_editor/toggle_bookmark", TTRC("Toggle Bookmark"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::ALT | Key::B);

	ED_SHORTCUT("script_text_editor/goto_next_bookmark", TTRC("Go to Next Bookmark"), KeyModifierMask::CMD_OR_CTRL | Key::B);
	ED_SHORTCUT_OVERRIDE("script_text_editor/goto_next_bookmark", "macos", KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | KeyModifierMask::ALT | Key::B);

	ED_SHORTCUT("script_text_editor/goto_previous_bookmark", TTRC("Go to Previous Bookmark"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::B);
	ED_SHORTCUT("script_text_editor/remove_all_bookmarks", TTRC("Remove All Bookmarks"), Key::NONE);

	ED_SHORTCUT("script_text_editor/goto_function", TTRC("Go to Function..."), KeyModifierMask::ALT | KeyModifierMask::CTRL | Key::F);
	ED_SHORTCUT_OVERRIDE("script_text_editor/goto_function", "macos", KeyModifierMask::CTRL | KeyModifierMask::META | Key::J);

	ED_SHORTCUT("script_text_editor/goto_line", TTRC("Go to Line..."), KeyModifierMask::CMD_OR_CTRL | Key::G);
	ED_SHORTCUT_OVERRIDE("script_text_editor/goto_line", "macos", KeyModifierMask::CMD_OR_CTRL | Key::L);
	ED_SHORTCUT("script_text_editor/goto_symbol", TTRC("Lookup Symbol"));

	ED_SHORTCUT("script_text_editor/toggle_breakpoint", TTRC("Toggle Breakpoint"), Key::F9);
	ED_SHORTCUT_OVERRIDE("script_text_editor/toggle_breakpoint", "macos", KeyModifierMask::META | KeyModifierMask::SHIFT | Key::B);

	ED_SHORTCUT("script_text_editor/remove_all_breakpoints", TTRC("Remove All Breakpoints"), KeyModifierMask::CMD_OR_CTRL | KeyModifierMask::SHIFT | Key::F9);
	// Using Control for these shortcuts even on macOS because Command+Comma is taken for opening Editor Settings.
	ED_SHORTCUT("script_text_editor/goto_next_breakpoint", TTRC("Go to Next Breakpoint"), KeyModifierMask::CTRL | Key::PERIOD);
	ED_SHORTCUT("script_text_editor/goto_previous_breakpoint", TTRC("Go to Previous Breakpoint"), KeyModifierMask::CTRL | Key::COMMA);

	ScriptEditor::register_create_script_editor_function(create_editor);
}

void ScriptTextEditor::validate() {
	code_editor->validate_script();
}

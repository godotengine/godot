/**************************************************************************/
/*  script_editor_base.cpp                                                */
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

#include "script_editor_base.h"

#include "core/io/json.h"
#include "editor/editor_node.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/script/syntax_highlighters.h"
#include "editor/settings/editor_settings.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/split_container.h"

void ScriptEditorBase::_bind_methods() {
	ADD_SIGNAL(MethodInfo("name_changed"));

	// First use in TextEditorBase.
	ADD_SIGNAL(MethodInfo("edited_script_changed"));
	ADD_SIGNAL(MethodInfo("search_in_files_requested", PropertyInfo(Variant::STRING, "text")));
	ClassDB::bind_method(D_METHOD("add_syntax_highlighter", "highlighter"), &ScriptEditorBase::add_syntax_highlighter);
	ClassDB::bind_method(D_METHOD("get_base_editor"), &ScriptEditorBase::get_base_editor);

	// First use in ScriptTextEditor.
	ADD_SIGNAL(MethodInfo("request_save_history"));
	ADD_SIGNAL(MethodInfo("request_help", PropertyInfo(Variant::STRING, "topic")));
	ADD_SIGNAL(MethodInfo("request_open_script_at_line", PropertyInfo(Variant::OBJECT, "script"), PropertyInfo(Variant::INT, "line")));
	ADD_SIGNAL(MethodInfo("go_to_help", PropertyInfo(Variant::STRING, "what")));
	ADD_SIGNAL(MethodInfo("request_save_previous_state", PropertyInfo(Variant::DICTIONARY, "state")));
	ADD_SIGNAL(MethodInfo("replace_in_files_requested", PropertyInfo(Variant::STRING, "text")));
	ADD_SIGNAL(MethodInfo("go_to_method", PropertyInfo(Variant::OBJECT, "script"), PropertyInfo(Variant::STRING, "method")));
}

String ScriptEditorBase::get_name() {
	String name;

	name = edited_res->get_path().get_file();
	if (name.is_empty()) {
		// This appears for newly created built-in text_files before saving the scene.
		name = TTR("[unsaved]");
	} else if (edited_res->is_built_in()) {
		const String &text_file_name = edited_res->get_name();
		if (!text_file_name.is_empty()) {
			// If the built-in text_file has a custom resource name defined,
			// display the built-in text_file name as follows: `ResourceName (scene_file.tscn)`
			name = vformat("%s (%s)", text_file_name, name.get_slice("::", 0));
		}
	}

	if (is_unsaved()) {
		name += "(*)";
	}

	return name;
}

Ref<Texture2D> ScriptEditorBase::get_theme_icon() {
	return EditorNode::get_singleton()->get_object_icon(edited_res.ptr());
}

void ScriptEditorBase::tag_saved_version() {
	edited_file_data.last_modified_time = FileAccess::get_modified_time(edited_file_data.path);
}

//// TextEditorBase

TextEditorBase *TextEditorBase::EditMenus::_get_active_editor() {
	return Object::cast_to<TextEditorBase>(ScriptEditor::get_singleton()->get_current_editor());
}

void TextEditorBase::EditMenus::_edit_option(int p_op) {
	TextEditorBase *script_text_editor = _get_active_editor();
	ERR_FAIL_NULL(script_text_editor);
	script_text_editor->_edit_option(p_op);
}

void TextEditorBase::EditMenus::_prepare_edit_menu() {
	TextEditorBase *script_text_editor = _get_active_editor();
	ERR_FAIL_NULL(script_text_editor);
	const CodeEdit *tx = script_text_editor->code_editor->get_text_editor();
	PopupMenu *popup = edit_menu->get_popup();
	popup->set_item_disabled(popup->get_item_index(EDIT_UNDO), !tx->has_undo());
	popup->set_item_disabled(popup->get_item_index(EDIT_REDO), !tx->has_redo());
}

void TextEditorBase::EditMenus::_update_highlighter_menu() {
	TextEditorBase *script_text_editor = _get_active_editor();
	ERR_FAIL_NULL(script_text_editor);

	Ref<EditorSyntaxHighlighter> current_highlighter = script_text_editor->get_code_editor()->get_text_editor()->get_syntax_highlighter();
	highlighter_menu->clear();
	for (const Ref<EditorSyntaxHighlighter> &highlighter : script_text_editor->highlighters) {
		highlighter_menu->add_radio_check_item(highlighter->_get_name());
		highlighter_menu->set_item_checked(-1, highlighter == current_highlighter);
	}
}

void TextEditorBase::EditMenus::_change_syntax_highlighter(int p_idx) {
	TextEditorBase *script_text_editor = _get_active_editor();
	ERR_FAIL_NULL(script_text_editor);
	ERR_FAIL_INDEX(p_idx, (int)script_text_editor->highlighters.size());
	script_text_editor->set_syntax_highlighter(script_text_editor->highlighters[p_idx]);
}

void TextEditorBase::EditMenus::_update_bookmark_list() {
	TextEditorBase *script_text_editor = _get_active_editor();
	ERR_FAIL_NULL(script_text_editor);
	bookmarks_menu->clear();
	bookmarks_menu->reset_size();

	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_bookmark"), BOOKMARK_TOGGLE);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/remove_all_bookmarks"), BOOKMARK_REMOVE_ALL);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_next_bookmark"), BOOKMARK_GOTO_NEXT);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_previous_bookmark"), BOOKMARK_GOTO_PREV);

	PackedInt32Array bookmark_list = script_text_editor->code_editor->get_text_editor()->get_bookmarked_lines();
	if (bookmark_list.is_empty()) {
		return;
	}

	bookmarks_menu->add_separator();

	for (int32_t bookmark : bookmark_list) {
		// Strip edges to remove spaces or tabs.
		// Also replace any tabs by spaces, since we can't print tabs in the menu.
		String line = script_text_editor->code_editor->get_text_editor()->get_line(bookmark).replace("\t", "  ").strip_edges();

		// Limit the size of the line if too big.
		if (line.length() > 50) {
			line = line.substr(0, 50);
		}

		bookmarks_menu->add_item(String::num_int64(bookmark + 1) + " - `" + line + "`");
		bookmarks_menu->set_item_metadata(-1, bookmark);
	}
}

void TextEditorBase::EditMenus::_bookmark_item_pressed(int p_idx) {
	TextEditorBase *script_text_editor = _get_active_editor();
	ERR_FAIL_NULL(script_text_editor);
	if (p_idx < 4) { // Any item before the separator.
		script_text_editor->_edit_option(bookmarks_menu->get_item_id(p_idx));
	} else {
		script_text_editor->code_editor->goto_line_centered(bookmarks_menu->get_item_metadata(p_idx));
	}
}

TextEditorBase::EditMenus::EditMenus() {
	edit_menu = memnew(MenuButton);
	edit_menu->set_flat(false);
	edit_menu->set_theme_type_variation("FlatMenuButton");
	edit_menu->set_text(TTRC("Edit"));
	edit_menu->set_switch_on_hover(true);

	add_child(edit_menu);
	edit_menu->connect("about_to_popup", callable_mp(this, &EditMenus::_prepare_edit_menu));
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
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_word_wrap"), EDIT_TOGGLE_WORD_WRAP);
	edit_menu->get_popup()->add_separator();
	{
		edit_menu_line = memnew(PopupMenu);
		edit_menu_line->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_up"), EDIT_MOVE_LINE_UP);
		edit_menu_line->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_down"), EDIT_MOVE_LINE_DOWN);
		edit_menu_line->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent"), EDIT_INDENT);
		edit_menu_line->add_shortcut(ED_GET_SHORTCUT("script_text_editor/unindent"), EDIT_UNINDENT);
		edit_menu_line->add_shortcut(ED_GET_SHORTCUT("script_text_editor/delete_line"), EDIT_DELETE_LINE);
		edit_menu_line->connect(SceneStringName(id_pressed), callable_mp(this, &EditMenus::_edit_option));
		edit_menu->get_popup()->add_submenu_node_item(TTRC("Line"), edit_menu_line);
	}
	{
		edit_menu_fold = memnew(PopupMenu);
		edit_menu_fold->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_fold_line"), EDIT_TOGGLE_FOLD_LINE);
		edit_menu_fold->add_shortcut(ED_GET_SHORTCUT("script_text_editor/fold_all_lines"), EDIT_FOLD_ALL_LINES);
		edit_menu_fold->add_shortcut(ED_GET_SHORTCUT("script_text_editor/unfold_all_lines"), EDIT_UNFOLD_ALL_LINES);
		edit_menu_fold->connect(SceneStringName(id_pressed), callable_mp(this, &EditMenus::_edit_option));
		edit_menu->get_popup()->add_submenu_node_item(TTRC("Folding"), edit_menu_fold);
	}
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/trim_trailing_whitespace"), EDIT_TRIM_TRAILING_WHITESAPCE);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/trim_final_newlines"), EDIT_TRIM_FINAL_NEWLINES);
	{
		edit_menu_convert_indent = memnew(PopupMenu);
		edit_menu_convert_indent->add_shortcut(ED_GET_SHORTCUT("script_text_editor/convert_indent_to_spaces"), EDIT_CONVERT_INDENT_TO_SPACES);
		edit_menu_convert_indent->add_shortcut(ED_GET_SHORTCUT("script_text_editor/convert_indent_to_tabs"), EDIT_CONVERT_INDENT_TO_TABS);
		edit_menu_convert_indent->connect(SceneStringName(id_pressed), callable_mp(this, &EditMenus::_edit_option));
		edit_menu->get_popup()->add_submenu_node_item(TTRC("Indentation"), edit_menu_convert_indent);
	}
	edit_menu->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &EditMenus::_edit_option));
	edit_menu->get_popup()->add_separator();
	{
		edit_menu_convert = memnew(PopupMenu);
		edit_menu_convert->add_shortcut(ED_GET_SHORTCUT("script_text_editor/convert_to_uppercase"), EDIT_TO_UPPERCASE);
		edit_menu_convert->add_shortcut(ED_GET_SHORTCUT("script_text_editor/convert_to_lowercase"), EDIT_TO_LOWERCASE);
		edit_menu_convert->add_shortcut(ED_GET_SHORTCUT("script_text_editor/capitalize"), EDIT_CAPITALIZE);
		edit_menu_convert->connect(SceneStringName(id_pressed), callable_mp(this, &EditMenus::_edit_option));
		edit_menu->get_popup()->add_submenu_node_item(TTRC("Convert Case"), edit_menu_convert);
	}
	highlighter_menu = memnew(PopupMenu);
	edit_menu->get_popup()->add_submenu_node_item(TTRC("Syntax Highlighter"), highlighter_menu);

	highlighter_menu->connect("about_to_popup", callable_mp(this, &EditMenus::_update_highlighter_menu));
	highlighter_menu->connect(SceneStringName(id_pressed), callable_mp(this, &EditMenus::_change_syntax_highlighter));

	search_menu = memnew(MenuButton);
	search_menu->set_flat(false);
	search_menu->set_theme_type_variation("FlatMenuButton");
	search_menu->set_text(TTRC("Search"));
	search_menu->set_switch_on_hover(true);

	add_child(search_menu);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find"), SEARCH_FIND);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_next"), SEARCH_FIND_NEXT);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_previous"), SEARCH_FIND_PREV);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/replace"), SEARCH_REPLACE);
	search_menu->get_popup()->add_separator();
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("editor/find_in_files"), SEARCH_IN_FILES);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/replace_in_files"), REPLACE_IN_FILES);
	search_menu->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &EditMenus::_edit_option));

	goto_menu = memnew(MenuButton);
	goto_menu->set_flat(false);
	goto_menu->set_theme_type_variation("FlatMenuButton");
	goto_menu->set_text(TTRC("Go To"));
	goto_menu->set_switch_on_hover(true);
	add_child(goto_menu);
	goto_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_line"), SEARCH_GOTO_LINE);
	goto_menu->get_popup()->add_separator();

	bookmarks_menu = memnew(PopupMenu);
	goto_menu->get_popup()->add_submenu_node_item(TTRC("Bookmarks"), bookmarks_menu);
	bookmarks_menu->connect("about_to_popup", callable_mp(this, &EditMenus::_update_bookmark_list));
	bookmarks_menu->connect("index_pressed", callable_mp(this, &EditMenus::_bookmark_item_pressed));

	goto_menu->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &EditMenus::_edit_option));
}

void TextEditorBase::_make_context_menu(bool p_selection, bool p_foldable, const Vector2 &p_position, bool p_show) {
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
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_bookmark"), BOOKMARK_TOGGLE);

	if (p_selection) {
		context_menu->add_separator();
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/convert_to_uppercase"), EDIT_TO_UPPERCASE);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/convert_to_lowercase"), EDIT_TO_LOWERCASE);
	}

	if (p_foldable) {
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_fold_line"), EDIT_TOGGLE_FOLD_LINE);
	}

	if (p_show) {
		_show_context_menu(p_position);
	}
}

void TextEditorBase::_show_context_menu(const Vector2 &p_position) {
	const CodeEdit *tx = code_editor->get_text_editor();
	context_menu->set_item_disabled(context_menu->get_item_index(EDIT_UNDO), !tx->has_undo());
	context_menu->set_item_disabled(context_menu->get_item_index(EDIT_REDO), !tx->has_redo());

	context_menu->set_position(get_screen_position() + p_position);
	context_menu->reset_size();
	context_menu->popup();
}

void TextEditorBase::_text_edit_gui_input(const Ref<InputEvent> &p_ev) {
	Ref<InputEventMouseButton> mb = p_ev;

	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::RIGHT) {
			CodeEdit *tx = code_editor->get_text_editor();

			tx->apply_ime();

			Point2i pos = tx->get_line_column_at_pos(mb->get_global_position() - tx->get_global_position());
			int row = pos.y;
			int col = pos.x;

			tx->set_move_caret_on_right_click_enabled(EDITOR_GET("text_editor/behavior/navigation/move_caret_on_right_click"));

			if (tx->is_move_caret_on_right_click_enabled()) {
				tx->remove_secondary_carets();
				if (tx->has_selection()) {
					int from_line = tx->get_selection_from_line();
					int to_line = tx->get_selection_to_line();
					int from_column = tx->get_selection_from_column();
					int to_column = tx->get_selection_to_column();

					if (row < from_line || row > to_line || (row == from_line && col < from_column) || (row == to_line && col > to_column)) {
						// Right click is outside the selected text.
						tx->deselect();
					}
				}
				if (!tx->has_selection()) {
					tx->set_caret_line(row, true, false, -1);
					tx->set_caret_column(col);
				}
			}

			if (!mb->is_pressed()) {
				bool can_fold = tx->can_fold_line(row);
				bool is_folded = tx->is_line_folded(row);
				_make_context_menu(tx->has_selection(), can_fold || is_folded, get_local_mouse_position());
			}
		}
	}

	Ref<InputEventKey> k = p_ev;
	if (k.is_valid() && k->is_pressed() && k->is_action("ui_menu", true)) {
		CodeEdit *tx = code_editor->get_text_editor();
		int line = tx->get_caret_line();
		tx->adjust_viewport_to_caret();
		bool can_fold = tx->can_fold_line(line);
		bool is_folded = tx->is_line_folded(line);
		_make_context_menu(tx->has_selection(0), can_fold || is_folded, (get_global_transform().inverse() * tx->get_global_transform()).xform(tx->get_caret_draw_pos(0)));
		context_menu->grab_focus();
	}
}

bool TextEditorBase::_edit_option(int p_op) {
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
			tx->move_lines_up();
		} break;
		case EDIT_MOVE_LINE_DOWN: {
			tx->move_lines_down();
		} break;
		case EDIT_INDENT: {
			tx->indent_lines();
		} break;
		case EDIT_UNINDENT: {
			tx->unindent_lines();
		} break;
		case EDIT_DELETE_LINE: {
			tx->delete_lines();
		} break;
		case EDIT_DUPLICATE_SELECTION: {
			tx->duplicate_selection();
		} break;
		case EDIT_DUPLICATE_LINES: {
			tx->duplicate_lines();
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
		case EDIT_TO_UPPERCASE: {
			_convert_case(CodeTextEditor::UPPER);
		} break;
		case EDIT_TO_LOWERCASE: {
			_convert_case(CodeTextEditor::LOWER);
		} break;
		case EDIT_CAPITALIZE: {
			_convert_case(CodeTextEditor::CAPITALIZE);
		} break;
		case EDIT_TOGGLE_WORD_WRAP: {
			TextEdit::LineWrappingMode wrap = tx->get_line_wrapping_mode();
			tx->set_line_wrapping_mode(wrap == TextEdit::LINE_WRAPPING_BOUNDARY ? TextEdit::LINE_WRAPPING_NONE : TextEdit::LINE_WRAPPING_BOUNDARY);
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
		case EDIT_EMOJI_AND_SYMBOL: {
			tx->show_emoji_and_symbol_picker();
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
		default: {
			return false;
		}
	}
	return true;
}

void TextEditorBase::_load_theme_settings() {
	code_editor->get_text_editor()->get_syntax_highlighter()->update_cache();
}

void TextEditorBase::_validate_script() {
	emit_signal(SNAME("name_changed"));
	emit_signal(SNAME("edited_script_changed"));
}

void TextEditorBase::add_syntax_highlighter(Ref<EditorSyntaxHighlighter> p_highlighter) {
	ERR_FAIL_COND(p_highlighter.is_null());

	highlighters.push_back(p_highlighter);
}

void TextEditorBase::set_syntax_highlighter(Ref<EditorSyntaxHighlighter> p_highlighter) {
	ERR_FAIL_COND(p_highlighter.is_null());

	CodeEdit *te = code_editor->get_text_editor();
	p_highlighter->_set_edited_resource(edited_res);
	te->set_syntax_highlighter(p_highlighter);
}

void TextEditorBase::set_edited_resource(const Ref<Resource> &p_res) {
	ERR_FAIL_COND(edited_res.is_valid());
	ERR_FAIL_COND(p_res.is_null());

	edited_res = p_res;

	Ref<TextFile> text_file = edited_res;
	String text;
	if (text_file.is_valid()) {
		text = text_file->get_text();
	}

	Ref<JSON> json_file = edited_res;
	if (json_file.is_valid()) {
		text = json_file->get_parsed_text();
	}

	code_editor->get_text_editor()->set_text(text);
	code_editor->get_text_editor()->clear_undo_history();
	code_editor->get_text_editor()->tag_saved_version();

	emit_signal(SNAME("name_changed"));
	code_editor->update_line_and_column();
}

bool TextEditorBase::is_unsaved() {
	return code_editor->get_text_editor()->get_version() != code_editor->get_text_editor()->get_saved_version() || edited_res->get_path().is_empty(); // In memory.
}

void TextEditorBase::tag_saved_version() {
	code_editor->get_text_editor()->tag_saved_version();
	ScriptEditorBase::tag_saved_version();
}

void TextEditorBase::reload_text() {
	ERR_FAIL_COND(edited_res.is_null());

	CodeEdit *te = code_editor->get_text_editor();
	int column = te->get_caret_column();
	int row = te->get_caret_line();
	int h = te->get_h_scroll();
	int v = te->get_v_scroll();

	Ref<TextFile> text_file = edited_res;
	if (text_file.is_valid()) {
		te->set_text(text_file->get_text());
	}

	Ref<JSON> json_file = edited_res;
	if (json_file.is_valid()) {
		te->set_text(json_file->get_parsed_text());
	}

	Ref<Script> script = edited_res;
	if (script.is_valid()) {
		te->set_text(script->get_source_code());
	}

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

void TextEditorBase::enable_editor() {
	if (editor_enabled) {
		return;
	}

	editor_enabled = true;

	_load_theme_settings();

	_validate_script();
}

void TextEditorBase::set_tooltip_request_func(const Callable &p_toolip_callback) {
	Variant args[1] = { this };
	const Variant *argp[] = { &args[0] };
	code_editor->get_text_editor()->set_tooltip_request_func(p_toolip_callback.bindp(argp, 1));
}

void TextEditorBase::set_edit_state(const Variant &p_state) {
	code_editor->set_edit_state(p_state);

	Dictionary state = p_state;
	if (state.has("syntax_highlighter")) {
		for (const Ref<EditorSyntaxHighlighter> &highlighter : highlighters) {
			if (highlighter->_get_name() == String(state["syntax_highlighter"])) {
				set_syntax_highlighter(highlighter);
				break;
			}
		}
	}

	ensure_focus();
}

TextEditorBase::TextEditorBase() {
	code_editor = memnew(CodeTextEditor);
	code_editor->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	code_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	code_editor->show_toggle_files_button();
	code_editor->get_text_editor()->set_context_menu_enabled(false);
	code_editor->get_text_editor()->connect(SceneStringName(gui_input), callable_mp(this, &TextEditorBase::_text_edit_gui_input));
	code_editor->connect("validate_script", callable_mp(this, &TextEditorBase::_validate_script));
	code_editor->connect("load_theme_settings", callable_mp(this, &TextEditorBase::_load_theme_settings));

	context_menu = memnew(PopupMenu);
	context_menu->connect(SceneStringName(id_pressed), callable_mp(this, &TextEditorBase::_edit_option));
	add_child(context_menu);

	edit_hb = memnew(HBoxContainer);

	goto_line_popup = memnew(GotoLinePopup);
	add_child(goto_line_popup);

	Ref<EditorPlainTextSyntaxHighlighter> plain_highlighter;
	plain_highlighter.instantiate();
	add_syntax_highlighter(plain_highlighter);

	Ref<EditorStandardSyntaxHighlighter> highlighter;
	highlighter.instantiate();
	add_syntax_highlighter(highlighter);
	set_syntax_highlighter(plain_highlighter);

	update_settings();
}

TextEditorBase::~TextEditorBase() {
	highlighters.clear();
}

//// CodeEditorBase

bool CodeEditorBase::_warning_clicked(const Variant &p_line) {
	if (p_line.get_type() == Variant::INT) {
		goto_line_centered(p_line.operator int64_t());
		return true;
	}
	return false;
}

CodeEditorBase::EditMenusCEB::EditMenusCEB() {
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("ui_text_completion_query"), EDIT_COMPLETE);
	_popup_move_item(EDIT_TRIM_TRAILING_WHITESAPCE, edit_menu->get_popup(), false);
	edit_menu_line->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);
}

CodeEditorBase::CodeEditorBase() {
	warnings_panel = memnew(RichTextLabel);
	warnings_panel->set_custom_minimum_size(Size2(0, 100 * EDSCALE));
	warnings_panel->set_h_size_flags(SIZE_EXPAND_FILL);
	warnings_panel->set_meta_underline(true);
	warnings_panel->set_selection_enabled(true);
	warnings_panel->set_context_menu_enabled(true);
	warnings_panel->set_focus_mode(FOCUS_CLICK);
	warnings_panel->hide();
	warnings_panel->connect("meta_clicked", callable_mp(this, &CodeEditorBase::_warning_clicked));

	editor_box = memnew(VSplitContainer);
	add_child(editor_box);
	editor_box->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	editor_box->set_v_size_flags(SIZE_EXPAND_FILL);
	editor_box->add_child(code_editor);
	editor_box->add_child(warnings_panel);
}

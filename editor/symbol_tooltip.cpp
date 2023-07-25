/**************************************************************************/
/*  symbol_tooltip.cpp                                                    */
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

#include "symbol_tooltip.h"
#include "editor/plugins/script_text_editor.h"
#include "editor_help.h"
#include "editor_node.h"

SymbolTooltip::SymbolTooltip(CodeTextEditor* code_editor) : code_editor(code_editor) { //(CodeTextEditor* code_editor) : code_editor(code_editor) {
	// Initialize the tooltip components

	// Set the tooltip's theme (PanelContainer's theme)
	//set_theme(EditorNode::get_singleton()->get_gui_base()->get_theme());
	hide();
	set_theme(_create_panel_theme());

	// Create VBoxContainer to hold the tooltip's header and body
	layout_container = memnew(VBoxContainer);
	add_child(layout_container);

	// Create RichTextLabel for the tooltip's header
	header_label = memnew(RichTextLabel);
	header_label->set_use_bbcode(true);
	header_label->set_selection_enabled(true);
	header_label->set_custom_minimum_size(Size2(0, 50));
	header_label->set_focus_mode(FOCUS_ALL);
	header_label->set_theme(_create_header_label_theme());
	layout_container->add_child(header_label);

	// Create RichTextLabel for the tooltip's body
	body_label = memnew(RichTextLabel);
	body_label->set_use_bbcode(true);
	body_label->set_selection_enabled(true);
	body_label->set_focus_mode(FOCUS_ALL);
	body_label->set_v_size_flags(SIZE_EXPAND_FILL);
	body_label->set_theme(_create_body_label_theme());
	layout_container->add_child(body_label);


	// Connect the tooltip's update function to the mouse motion signal
	// connect("mouse_motion", callable_mp(this, &SymbolTooltip::_update_symbol_tooltip));
}

void SymbolTooltip::update_symbol_tooltip(const Vector2 &mouse_position) {
	// Get the word under the mouse cursor
	CodeEdit *text_editor = code_editor->get_text_editor();
	String _symbol_word = text_editor->get_word_at_pos(mouse_position);
	if (!_symbol_word.is_empty()) {
		symbol_word = _symbol_word;
		header_content = symbol_word;
	}

	// Get the symbol type of the word under the mouse cursor
	//String symbol_type = text_editor->get_symbol_type_at_pos(mouse_position);
	//String symbol_info = text_editor->get_symbol_info_at_pos(mouse_position);
	// symbol_info is a string containing the symbol's type, name, and documentation

	// Retrieve the EditorInterface singleton
	//EditorInterface *editor_interface = EditorInterface::get_singleton();

	// Retrieve the currently active ScriptTextEditor
	//ScriptTextEditor *script_editor = editor_interface->get_editor_viewport()->get_script_text_editor();
	//ScriptLanguage *language;

	//ScriptTextEditor *script_editor = text_editor.script //EditorNode::get_script_text_editor();

	/*Node *base = get_tree()->get_edited_scene_root();
	Ref<Script> script = base->get_script();*/
	/*if (base) {
		base = _find_node_for_script(base, base, script);
	}*/

	// https://github.com/godotengine/godot/pull/63908/files#diff-fa2cafb4b2d8b59e5baeba0b861f7f2f6bf6c213d6387254fee5a7682478e588R1564-R1573
	//ScriptLanguage::LookupResult result;
	/*if (ScriptServer::is_global_class(symbol_word)) {
		header_content = "class " + symbol_word; //ScriptLanguage::SymbolHint::SYMBOL_CLASS;
		body_content = "In " + ScriptServer::get_global_class_path(symbol_word);
		// https://github.com/godotengine/godot/pull/63908/files#diff-fa2cafb4b2d8b59e5baeba0b861f7f2f6bf6c213d6387254fee5a7682478e588R1564-R1573
	}*/

	/*if (script != NULL && script->get_language()->lookup_code(text_editor->get_text_for_symbol_lookup(), symbol_word, script->get_path(), base, result) != OK) {
		//_hide_symbol_hint();
		print_line(vformat("Result for %s:\n%s", symbol_word, ""));
	}*/

	// Get the documentation of the word under the mouse cursor
	// TODO: Account for the context of the word (e.g. class, method, etc.)
	//		and the context of the current line and surrounding lines.
	String documentation = _get_doc_of_word(symbol_word);
	body_content = documentation;

	// Calculate and set the tooltip's position
	Vector2 line_col = text_editor->get_line_column_at_pos(mouse_position);
	int row = line_col.y;
	int col = line_col.x;
	int num_lines = text_editor->get_line_count();
	if (row >= 0 && row < num_lines) {
		String line = text_editor->get_line(row);
		int symbol_col = _get_column_pos_of_word(symbol_word, line, text_editor->SEARCH_MATCH_CASE | text_editor->SEARCH_WHOLE_WORDS, 0);
		if(symbol_col >= 0) {
			// Calculate the symbol end column
			int symbol_end = symbol_col + symbol_word.length();
			if (col > symbol_end) {
				// Mouse is to the right of the last symbol character
				hide();
				return;
			}
			Vector2 _symbol_position = text_editor->get_pos_at_line_column(row, symbol_col);
			if(symbol_position != _symbol_position) {
				symbol_position = _symbol_position;
				Vector2 tooltip_position = Vector2(mouse_position.x, symbol_position.y);
				set_position(tooltip_position);
			}
		}
	}

	// Calculate and set the tooltip's size
	Vector2 size = Vector2(600, 300);
	set_size(size);

	// Update the tooltip's header and body
	_update_header_label(header_content); //symbol_word);
	_update_body_label(body_content); //documentation);

	// Bring the tooltip to the front
	move_to_front();

	Rect2 tooltip_rect = Rect2(get_position(), get_size());
	bool mouse_over_tooltip = tooltip_rect.has_point(mouse_position);
	if (!symbol_word.is_empty() || mouse_over_tooltip) {
		show();
	} else {
		hide();
	}
}

void SymbolTooltip::_update_header_label(const String &header_content) {
	// Set the tooltip's header text
	header_label->set_text(header_content);
}

void SymbolTooltip::_update_body_label(const String &body_content) {
	// Set the tooltip's body text
	body_label->clear();
	_add_text_to_rt(body_content, body_label, this);
	//body_label->set_text(body_content);
}

String SymbolTooltip::_get_doc_of_word(const String &symbol_word) {
	String documentation;

	const HashMap<String, DocData::ClassDoc> &class_list = EditorHelp::get_doc_data()->class_list;
	for (const KeyValue<String, DocData::ClassDoc> &E : class_list) {
		const DocData::ClassDoc &class_doc = E.value;

		if (class_doc.name == symbol_word) {
			documentation = class_doc.brief_description.strip_edges(); //class_doc.brief_description + "\n\n" + class_doc.description;
			break;
		}

		for (int i = 0; i < class_doc.methods.size(); ++i) {
			const DocData::MethodDoc &method_doc = class_doc.methods[i];

			if (method_doc.name == symbol_word) {
				documentation = method_doc.description.strip_edges();
				break;
			}
		}

		if (!documentation.is_empty()) {
			break;
		}
	}

	/*if (!documentation.is_empty()) {
		print_line(vformat("Documentation for %s:\n%s", symbol_word, documentation));
	}*/
	return documentation;
}

Ref<Theme> SymbolTooltip::_create_panel_theme() {
	Ref<Theme> theme = memnew(Theme); // TODO: Get the global theme instead (e.g. dark mode, light mode)

	Ref<StyleBoxFlat> style_box = memnew(StyleBoxFlat);
	style_box->set_bg_color(Color().html("#363d4a")); // Set the background color (RGBA)
	style_box->set_border_color(Color(0.8, 0.81, 0.82, 0.47)); // Set the border color (RGBA)
	style_box->set_border_width_all(1); // Set the border width
	style_box->set_corner_radius_all(4); // Set the border radius for curved corners
	//style_box->set_content_margin_all(20);
	theme->set_stylebox("panel", "PanelContainer", style_box);

	return theme;
}

Ref<Theme> SymbolTooltip::_create_header_label_theme() {
	Ref<Theme> theme = memnew(Theme); // TODO: Get the global theme instead (e.g. dark mode, light mode)

	Ref<StyleBoxFlat> style_box = memnew(StyleBoxFlat);
	style_box->set_draw_center(false);
	style_box->set_border_color(Color(0.8, 0.81, 0.82, 0.27)); // Set the border color (RGBA)
	style_box->set_border_width(SIDE_BOTTOM, 1);
	style_box->set_content_margin_individual(15, 10, 15, 10);
	theme->set_stylebox("normal", "RichTextLabel", style_box);

	return theme;
}

Ref<Theme> SymbolTooltip::_create_body_label_theme() {
	Ref<Theme> theme = memnew(Theme); // TODO: Get the global theme instead (e.g. dark mode, light mode)

	Ref<StyleBoxFlat> style_box = memnew(StyleBoxFlat);
	style_box->set_draw_center(false);
	style_box->set_content_margin_individual(15, 10, 15, 10);
	theme->set_stylebox("normal", "RichTextLabel", style_box);

	return theme;
}

// Copied from text_edit.cpp
int SymbolTooltip::_get_column_pos_of_word(const String &p_key, const String &p_search, uint32_t p_search_flags, int p_from_column) const {
	int col = -1;

	if (p_key.length() > 0 && p_search.length() > 0) {
		if (p_from_column < 0 || p_from_column > p_search.length()) {
			p_from_column = 0;
		}

		while (col == -1 && p_from_column <= p_search.length()) {
			if (p_search_flags & CodeEdit::SEARCH_MATCH_CASE) {
				col = p_search.find(p_key, p_from_column);
			} else {
				col = p_search.findn(p_key, p_from_column);
			}

			// Whole words only.
			if (col != -1 && p_search_flags & CodeEdit::SEARCH_WHOLE_WORDS) {
				p_from_column = col;

				if (col > 0 && !is_symbol(p_search[col - 1])) {
					col = -1;
				} else if ((col + p_key.length()) < p_search.length() && !is_symbol(p_search[col + p_key.length()])) {
					col = -1;
				}
			}

			p_from_column += 1;
		}
	}
	return col;
}

// Copied from script_text_editor.cpp
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

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

	tooltip_delay = memnew(Timer);
	tooltip_delay->set_one_shot(true);
	tooltip_delay->set_wait_time(0.5);
	add_child(tooltip_delay);

	tooltip_delay->connect("timeout", callable_mp(this, &SymbolTooltip::_on_tooltip_delay_timeout));

	// Connect the tooltip's update function to the mouse motion signal
	// connect("mouse_motion", callable_mp(this, &SymbolTooltip::_update_symbol_tooltip));
}

SymbolTooltip::~SymbolTooltip() {
	memdelete(tooltip_delay);
}

void SymbolTooltip::_on_tooltip_delay_timeout() {
	move_to_front(); // Bring the tooltip to the front
	show();
}

void SymbolTooltip::update_symbol_tooltip(const Vector2 &mouse_position) {
	CodeEdit *text_editor = code_editor->get_text_editor();
	String symbol_word = _get_symbol_word(text_editor, mouse_position);
	if (symbol_word.is_empty()) {
		tooltip_delay->stop();
		hide();
		return;
	}

	_update_tooltip_size();

	// Get the documentation of the word under the mouse cursor
	String documentation = _get_doc_of_word(symbol_word);
	_update_tooltip_content(symbol_word, documentation);

	Rect2 tooltip_rect = Rect2(get_position(), get_size());
	bool mouse_over_tooltip = tooltip_rect.has_point(mouse_position);
	if (!mouse_over_tooltip) {
		Vector2 tooltip_position = _calculate_tooltip_position(symbol_word, mouse_position);
		if (tooltip_position == Vector2(-1, -1)) { // If invalid position
			tooltip_delay->stop();
			hide();
			return;
		} else {
			set_position(tooltip_position);
		}
	}

	// Start the timer to show the tooltip after a delay
	//tooltip_delay->start();
	_on_tooltip_delay_timeout();
}

String SymbolTooltip::_get_symbol_word(CodeEdit *text_editor, const Vector2 &mouse_position) {
	// Get the word under the mouse cursor
	return text_editor->get_word_at_pos(mouse_position);
}

Vector2 SymbolTooltip::_calculate_tooltip_position(const String &symbol_word, const Vector2 &mouse_position) {
	CodeEdit *text_editor = code_editor->get_text_editor();
	Vector2 line_col = text_editor->get_line_column_at_pos(mouse_position);
	int row = line_col.y;
	int col = line_col.x;
	int num_lines = text_editor->get_line_count();
	if (row >= 0 && row < num_lines) {
		String line = text_editor->get_line(row);
		uint32_t search_flags = text_editor->SEARCH_MATCH_CASE | text_editor->SEARCH_WHOLE_WORDS;
		int symbol_col = _get_column_pos_of_word(symbol_word, line, search_flags, 0) + 1;
		if (symbol_col >= 0 && col >= symbol_col && col <= symbol_col + symbol_word.length()) {
			Vector2 symbol_position = text_editor->get_pos_at_line_column(row, symbol_col);
			return Vector2(symbol_position.x, symbol_position.y + 5); // Adjust the position to be below the symbol
		}
	}
	return Vector2(-1,-1); // indicates an invalid position
}

void SymbolTooltip::_update_tooltip_size() {
	// Calculate and set the tooltip's size
	set_size(Vector2(600, 300));
}

void SymbolTooltip::_update_tooltip_content(const String &header_content, const String &body_content) {
	// Update the tooltip's header and body
	_update_header_label(header_content);
	_update_body_label(body_content);
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

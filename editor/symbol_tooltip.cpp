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
#include "core/config/project_settings.h"
#include "editor/plugins/script_text_editor.h"
#include "editor_help.h"
#include "modules/gdscript/editor/gdscript_highlighter.h"
#include <queue>

SymbolTooltip::SymbolTooltip(CodeTextEditor *code_editor) :
		code_editor(code_editor) {
	// Initialize the tooltip components.

	// Set the tooltip's theme (PanelContainer's theme)
	//set_theme(EditorNode::get_singleton()->get_gui_base()->get_theme());

	set_transient(true);
	set_flag(Window::FLAG_NO_FOCUS, true);
	set_flag(Window::FLAG_POPUP, false);
	set_flag(Window::FLAG_MOUSE_PASSTHROUGH, false);
	set_theme(_create_popup_panel_theme());
	panel_container = memnew(PanelContainer);
	panel_container->set_theme(_create_panel_theme());
	add_child(panel_container);

	// Create VBoxContainer to hold the tooltip's header and body.
	layout_container = memnew(VBoxContainer);
	panel_container->add_child(layout_container);

	// Create RichTextLabel for the tooltip's header.
	header_label = memnew(TextEdit);
	//header_label->set_readonly(true);
	header_label->set_context_menu_enabled(false);
	header_label->set_h_scroll(false);
	header_label->set_v_scroll(false);
	Ref<GDScriptSyntaxHighlighter> highlighter;
	highlighter.instantiate();
	header_label->set_syntax_highlighter(highlighter);
	//header_label->set_selection_enabled(true);
	header_label->set_custom_minimum_size(Size2(0, 50));
	header_label->set_focus_mode(Control::FOCUS_ALL);
	header_label->set_theme(_create_header_label_theme());
	layout_container->add_child(header_label);

	// Create RichTextLabel for the tooltip's body.
	body_label = memnew(RichTextLabel);
	body_label->set_use_bbcode(true);
	body_label->set_selection_enabled(true);
	body_label->set_focus_mode(Control::FOCUS_ALL);
	body_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	body_label->set_theme(_create_body_label_theme());
	layout_container->add_child(body_label);

	float tooltip_delay_time = ProjectSettings::get_singleton()->get("gui/timers/tooltip_delay_sec");
	tooltip_delay = memnew(Timer);
	tooltip_delay->set_one_shot(true);
	tooltip_delay->set_wait_time(tooltip_delay_time);
	add_child(tooltip_delay);

	tooltip_delay->connect("timeout", callable_mp(this, &SymbolTooltip::_on_tooltip_delay_timeout));

	// Connect the tooltip's update function to the mouse motion signal.
	// connect("mouse_motion", callable_mp(this, &SymbolTooltip::_update_symbol_tooltip));
}

SymbolTooltip::~SymbolTooltip() {
	memdelete(tooltip_delay);
}

void SymbolTooltip::_on_tooltip_delay_timeout() {
	show();
}

void SymbolTooltip::update_symbol_tooltip(const Vector2 &mouse_position, Ref<Script> script) {
	CodeEdit *text_editor = code_editor->get_text_editor();
	String symbol_word = _get_symbol_word(text_editor, mouse_position);
	if (symbol_word.is_empty()) {
		tooltip_delay->stop();
		last_symbol_word = "";
		hide();
		return;
	}

	if (symbol_word == last_symbol_word && is_visible()) {
		return;
	} else {
		// Symbol has changed, reset the timer.
		tooltip_delay->stop();
		last_symbol_word = symbol_word;
	}

	ExtendGDScriptParser *parser = get_script_parser(script);
	HashMap<String, const lsp::DocumentSymbol *> members = parser->get_members();
	const lsp::DocumentSymbol *member_symbol = get_member_symbol(members, symbol_word);

	if (member_symbol == nullptr) {
		tooltip_delay->stop();
		hide();
		return;
	}

	_update_tooltip_size();

	// Get the documentation of the word under the mouse cursor.
	String official_documentation = _get_doc_of_word(symbol_word);
	String comment_documentation = member_symbol->documentation;

	// TODO: Improve header content. Add the ability to see documentation comments or official documentation.
	String header_content = member_symbol->reduced_detail.is_empty() ? symbol_word : member_symbol->reduced_detail;
	String body_content = comment_documentation.replace("\n ", " ");
	_update_tooltip_content(header_content, body_content);

	Rect2 tooltip_rect = Rect2(get_position(), get_size());
	bool mouse_over_tooltip = tooltip_rect.has_point(mouse_position);
	if (!mouse_over_tooltip) {
		Vector2 tooltip_position = _calculate_tooltip_position(symbol_word, mouse_position);
		if (tooltip_position == Vector2(-1, -1)) { // If invalid position
			tooltip_delay->stop();
			hide();
			return;
		} else {
			//Vector2 symbol_position = tooltip_position - text_editor->get_screen_position();
			//Vector2 temp_line_col = text_editor->get_line_column_at_pos(symbol_position);
			set_position(tooltip_position);
		}
	}

	// Start the timer to show the tooltip after a delay.
	tooltip_delay->start();
}

String SymbolTooltip::_get_symbol_word(CodeEdit *text_editor, const Vector2 &mouse_position) {
	// Get the word under the mouse cursor.
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
		int symbol_col = _get_word_pos_under_mouse(symbol_word, line, col);
		if (symbol_col >= 0) {
			Vector2 symbol_position = text_editor->get_pos_at_line_column(row, symbol_col);
			return text_editor->get_screen_position() + symbol_position;
		}
	}
	return Vector2(-1, -1); // Indicates an invalid position.
}

void SymbolTooltip::_update_tooltip_size() {
	// Calculate and set the tooltip's size.
	set_size(Vector2(600, 300));
}

void SymbolTooltip::_update_tooltip_content(const String &header_content, const String &body_content) {
	// Update the tooltip's header and body.
	_update_header_label(header_content);
	_update_body_label(body_content);
}

void SymbolTooltip::_update_header_label(const String &header_content) {
	// Set the tooltip's header text.
	//Ref<SyntaxHighlighter> highlighter = code_editor->get_text_editor()->get_syntax_highlighter();
	header_label->set_text(header_content);
}

void SymbolTooltip::_update_body_label(const String &body_content) {
	// Set the tooltip's body text.
	body_label->clear();
	_add_text_to_rt(body_content, body_label, layout_container);
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

Ref<Theme> SymbolTooltip::_create_popup_panel_theme() {
	Ref<Theme> theme = memnew(Theme);

	Ref<StyleBoxFlat> style_box = memnew(StyleBoxFlat);
	style_box->set_bg_color(Color(0, 0, 0, 0)); // Set the background color (RGBA).
	theme->set_stylebox("panel", "PopupPanel", style_box);

	return theme;
}

Ref<Theme> SymbolTooltip::_create_panel_theme() {
	Ref<Theme> theme = memnew(Theme); // TODO: Get the global theme instead (e.g. dark mode, light mode).

	Ref<StyleBoxFlat> style_box = memnew(StyleBoxFlat);
	style_box->set_bg_color(Color().html("#363d4a")); // Set the background color (RGBA).
	style_box->set_border_color(Color(0.8, 0.81, 0.82, 0.47)); // Set the border color (RGBA).
	style_box->set_border_width_all(1); // Set the border width.
	style_box->set_corner_radius_all(4); // Set the border radius for curved corners.
	//style_box->set_content_margin_all(20);
	theme->set_stylebox("panel", "PanelContainer", style_box);

	return theme;
}

Ref<Theme> SymbolTooltip::_create_header_label_theme() {
	Ref<Theme> theme = memnew(Theme); // TODO: Get the global theme instead (e.g. dark mode, light mode).

	Ref<StyleBoxFlat> style_box = memnew(StyleBoxFlat);
	style_box->set_draw_center(false);
	style_box->set_border_color(Color(0.8, 0.81, 0.82, 0.27)); // Set the border color (RGBA).
	style_box->set_border_width(SIDE_BOTTOM, 1);
	style_box->set_content_margin_individual(15, 10, 15, 10);

	// Set the style boxes for the TextEdit
	theme->set_stylebox("normal", "TextEdit", style_box);
	theme->set_stylebox("focus", "TextEdit", style_box);
	theme->set_stylebox("hover", "TextEdit", style_box);

	// Set the font color.
	theme->set_color("font_color", "TextEdit", Color(1, 1, 1));

	return theme;
}

Ref<Theme> SymbolTooltip::_create_body_label_theme() {
	Ref<Theme> theme = memnew(Theme); // TODO: Get the global theme instead (e.g. dark mode, light mode).

	Ref<StyleBoxFlat> style_box = memnew(StyleBoxFlat);
	style_box->set_draw_center(false);
	style_box->set_content_margin_individual(15, 10, 15, 10);
	theme->set_stylebox("normal", "RichTextLabel", style_box);

	return theme;
}

int SymbolTooltip::_get_word_pos_under_mouse(const String &symbol_word, const String &p_search, int mouse_x) const {
	// Created this because _get_column_pos_of_word() only gets the column position of the first occurrence of the word in the line.

	// Early exit if the symbol word is empty, the search string is empty, or the mouse is outside the string.
	if (symbol_word.is_empty() || p_search.is_empty() || mouse_x < 0 || mouse_x >= p_search.length()) {
		return -1;
	}

	int start = mouse_x;
	int end = mouse_x;

	// Extend the start and end until they reach the beginning or end of the word.
	while (start > 0 && is_ascii_identifier_char(p_search[start - 1])) {
		start--;
	}
	while (end < p_search.length() && is_ascii_identifier_char(p_search[end])) {
		end++;
	}

	String word_under_mouse = p_search.substr(start, end - start);

	// If the word under the mouse matches the symbol word, return the start position.
	if (word_under_mouse == symbol_word) {
		return start + 1; // Note: +1 is added to account for zero-based indexing.
	}

	return -1; // Return -1 if no match is found.
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

const GDScriptParser::ClassNode::Member *find_symbol(const GDScriptParser::ClassNode *node, const String &symbol_word) {
	for (int i = 0; i < node->members.size(); ++i) {
		const GDScriptParser::ClassNode::Member &member = node->members[i];

		if (member.get_name() == symbol_word) {
			// Found the symbol.
			return &member;
		} else if (member.type == GDScriptParser::ClassNode::Member::CLASS) {
			const GDScriptParser::ClassNode::Member *found_symbol = find_symbol(member.m_class, symbol_word);
			if (found_symbol) {
				return found_symbol;
			}
		}
	}

	return nullptr;
}

// Gets the head of the GDScriptParser AST tree.
/*static const GDScriptParser::ClassNode *get_ast_tree(const Ref<Script> &p_script) {
	// Create and initialize the parser.
	GDScriptParser *parser = memnew(GDScriptParser);
	Error err = parser->parse(p_script->get_source_code(), p_script->get_path(), false);

	if (err != OK) {
		ERR_PRINT("Failed to parse GDScript with GDScriptParser.");
		return nullptr;
	}

	// Get the AST tree.
	const GDScriptParser::ClassNode *ast_tree = parser->get_tree();
	return ast_tree;
}*/

static ExtendGDScriptParser *get_script_parser(const Ref<Script> &p_script) {
	// Create and initialize the parser.
	ExtendGDScriptParser *parser = memnew(ExtendGDScriptParser);
	Error err = parser->parse(p_script->get_source_code(), p_script->get_path());

	if (err != OK) {
		ERR_PRINT("Failed to parse GDScript with GDScriptParser.");
		return nullptr;
	}

	return parser;
}

// TODO: Need to find the correct symbol instance instead of just the first match.
const lsp::DocumentSymbol *get_member_symbol(
		HashMap<String, const lsp::DocumentSymbol *> &members,
		const String &symbol_word) {//,
		//const Vector2 &symbol_position) {
	// Use a queue to implement breadth-first search.
	std::queue<const lsp::DocumentSymbol*> queue;

	// Add all members to the queue.
	for (const KeyValue<String, const lsp::DocumentSymbol *> &E : members) {
		queue.push(E.value);
	}

	// While there are still elements in the queue.
	while (!queue.empty()) {
		// Get the next symbol.
		const lsp::DocumentSymbol *symbol = queue.front();
		queue.pop();

		// If the name matches, return the symbol.
		if (symbol->name == symbol_word && !symbol->detail.is_empty()) { // && symbol->range.is_point_inside(symbol_position)) {
			return symbol;
		}

		// Add the children to the queue for later processing.
		for (int i = 0; i < symbol->children.size(); ++i) {
			queue.push(&symbol->children[i]);
		}
	}

	return nullptr;  // If the symbol is not found, return nullptr.
}

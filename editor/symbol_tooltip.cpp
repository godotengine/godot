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

SymbolTooltip::SymbolTooltip(CodeEdit *p_text_editor) :
		text_editor(p_text_editor) {
	Ref<GDScriptSyntaxHighlighter> highlighter;
	highlighter.instantiate();

	// Set the tooltip's theme (PanelContainer's theme)
	//set_theme(EditorNode::get_singleton()->get_gui_base()->get_theme());

	set_as_top_level(true); // Prevents the tooltip from affecting the editor's layout.
	hide();
	/*set_v_size_flags(0);
	set_h_size_flags(0);*/
	set_z_index(1000);
	set_theme(_create_panel_theme());
	set_v_size_flags(Control::SIZE_SHRINK_BEGIN);
	//set_size(Size2(400, 400));
	//set_position(Size2(800, 800));
	set_process(true);
	//set_clip_contents(true);

	// Create VBoxContainer to hold the tooltip's header and body.
	layout_container = memnew(VBoxContainer);
	layout_container->add_theme_constant_override("separation", 0);
	layout_container->set_v_size_flags(Control::SIZE_SHRINK_BEGIN);
	add_child(layout_container);

	// Create RichTextLabel for the tooltip's header.
	header_label = memnew(TextEdit);
	header_label->set_focus_mode(Control::FOCUS_ALL);
	header_label->set_context_menu_enabled(false);
	header_label->set_h_scroll_visibility(false);
	header_label->set_v_scroll_visibility(false);
	header_label->set_syntax_highlighter(highlighter);
	header_label->set_custom_minimum_size(Size2(50, 45));

	header_label->set_v_size_flags(Control::SIZE_SHRINK_BEGIN);
	header_label->set_line_wrapping_mode(TextEdit::LINE_WRAPPING_BOUNDARY);
	header_label->set_fit_content_height_enabled(true);

	//header_label->set_editable(false); // WARNING!! - Enabling this will mess with the theme.
	//header_label->set_selection_enabled(true);
	header_label->set_theme(_create_header_label_theme());
	layout_container->add_child(header_label);

	// Create RichTextLabel for the tooltip's body.
	body_label = memnew(RichTextLabel);
	body_label->set_use_bbcode(true);
	body_label->set_selection_enabled(true);
	body_label->set_custom_minimum_size(Size2(400, 45));
	body_label->set_focus_mode(Control::FOCUS_ALL);
	//body_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	body_label->set_theme(_create_body_label_theme());
	//body_label->set_fit_content(true); // WARNING!! - Enabling this will cause issues in _update_tooltip_size().
	body_label->hide();
	layout_container->add_child(body_label);

	float tooltip_delay_time = ProjectSettings::get_singleton()->get("gui/timers/tooltip_delay_sec");
	tooltip_delay = memnew(Timer);
	tooltip_delay->set_one_shot(true);
	tooltip_delay->set_wait_time(tooltip_delay_time);
	add_child(tooltip_delay);

	tooltip_delay->connect("timeout", callable_mp(this, &SymbolTooltip::_on_tooltip_delay_timeout));

	mouse_inside = false;

	// Connect the tooltip's update function to the mouse motion signal.
	// connect("mouse_motion", callable_mp(this, &SymbolTooltip::_update_symbol_tooltip));
}

SymbolTooltip::~SymbolTooltip() {
	memdelete(tooltip_delay);
}

void SymbolTooltip::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PROCESS: {
			// Note: Child components prevent NOTIFICATION_MOUSE_ENTER and NOTIFICATION_MOUSE_EXIT from working properly.
			// Get the local mouse position
			Vector2 local_mouse_position = get_global_mouse_position() - get_global_position();

			// Check if it's within the rect of the PanelContainer
			Rect2 tooltip_rect = Rect2(Vector2(0, 0), get_size());
			if (tooltip_rect.has_point(local_mouse_position)) {
				if (is_visible() && !mouse_inside) {
					// Mouse just entered
					mouse_inside = true;
				}
			} else {
				if (mouse_inside) {
					// Mouse just exited
					mouse_inside = false;
					if (is_visible()) {
						_close_tooltip();
					}
				}
			}
		} break;
	}
}

void SymbolTooltip::_on_tooltip_delay_timeout() {
	print_line("size: " + get_size() + ", vbox_size: " + layout_container->get_size() + ", header_size: " + header_label->get_size() + ", body_size: " + body_label->get_size());

	show();
}

void SymbolTooltip::_close_tooltip() {
	tooltip_delay->stop();
	hide();
}

void SymbolTooltip::update_symbol_tooltip(const Vector2 &p_mouse_position, const Ref<Script> &p_script) {
	String symbol_word = _get_symbol_word(p_mouse_position);
	if (symbol_word.is_empty()) {
		last_symbol_word = "";
		_close_tooltip();
		return;
	}

	if (symbol_word == last_symbol_word && is_visible()) { // Symbol has not changed.
		return;
	} else {
		// Symbol has changed, reset the timer.
		_close_tooltip();
		last_symbol_word = symbol_word;
	}

	if(!_update_tooltip_content(p_script, symbol_word)) {
		_close_tooltip();
		return;
	}
	_update_tooltip_size();

	Rect2 tooltip_rect = Rect2(get_position(), get_size());
	bool mouse_over_tooltip = tooltip_rect.has_point(p_mouse_position) && is_visible();
	if (!mouse_over_tooltip) {
		Vector2 tooltip_position = _calculate_tooltip_position(symbol_word, p_mouse_position);
		if (tooltip_position == Vector2(-1, -1)) { // If position is invalid.
			_close_tooltip();
			return;
		} else {
			set_position(tooltip_position);
		}
	}

	// Start the timer to show the tooltip after a delay.
	if (!is_visible()) {
		tooltip_delay->start();
	}
}

String SymbolTooltip::_get_symbol_word(const Vector2 &p_mouse_position) {
	// Get the word under the mouse cursor.
	return text_editor->get_word_at_pos(p_mouse_position);
}

bool SymbolTooltip::_update_tooltip_content(const Ref<Script> &p_script, const String &p_symbol_word) {
	// Update the tooltip's header and body.

	bool is_built_in = false;
	String built_in_header = "";
	String built_in_body = "";
	Vector<const DocData::ClassDoc*> class_docs = _get_built_in_class_docs();
	for (const DocData::ClassDoc *class_doc : class_docs) {
		const DocData::ConstantDoc *constant_doc = _get_class_constant_doc(class_doc, p_symbol_word);
		if (constant_doc) {
			built_in_header = constant_doc->name + " = " + constant_doc->value;
			is_built_in = true;
			break;
		} else {
			const DocData::MethodDoc *method_doc = _get_class_method_doc(class_doc, p_symbol_word);
			if (method_doc) {
				built_in_header = _get_class_method_doc_definition(method_doc);
				built_in_body = method_doc->description.strip_edges();
				is_built_in = true;
				break;
			}
		}
	}

	// String official_documentation = _get_doc_of_word(p_symbol_word);

	ExtendGDScriptParser *parser = _get_script_parser(p_script);
	HashMap<String, const lsp::DocumentSymbol *> members = parser->get_members();
	const lsp::DocumentSymbol *member_symbol = _get_member_symbol(members, p_symbol_word);

	if (member_symbol == nullptr && !is_built_in) { // Symbol is not a member of the script.
		return false;
	}

	// TODO: Improve header content. Add the ability to see documentation comments or official documentation.
	// Add constant value to the header content. e.g. "PI" becomes "PI = 3.141592653589793" (similar to how we do with ENUMs)
	String header_content = _get_header_content(p_symbol_word, member_symbol, built_in_header);
	String body_content = _get_body_content(member_symbol, built_in_body);

	_update_header_label(header_content);
	_update_body_label(body_content);
	return true;
}

void SymbolTooltip::_update_tooltip_size() {
	// Constants and References
	const int MAX_WIDTH = 800;
	Ref<Theme> header_theme = header_label->get_theme();
	Ref<StyleBox> header_style_box = header_theme->get_stylebox("normal", "TextEdit");
	Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Editor"));
	int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Editor"));
	String header_text = header_label->get_text();
	String body_text = body_label->get_parsed_text();
	Ref<Theme> body_theme = body_label->get_theme();
	Ref<StyleBox> body_style_box = body_theme->get_stylebox("normal", "RichTextLabel");

	// Calculate content size and style box paddings
	Size2 header_content_size = font->get_string_size(header_text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size);
	Size2 body_content_size = font->get_string_size(body_text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size);
	real_t header_h_padding = header_style_box->get_content_margin(SIDE_LEFT) + header_style_box->get_content_margin(SIDE_RIGHT);
	//real_t header_v_padding = header_style_box->get_content_margin(SIDE_TOP) + header_style_box->get_content_margin(SIDE_BOTTOM);
	real_t body_h_padding = body_style_box->get_content_margin(SIDE_LEFT) + body_style_box->get_content_margin(SIDE_RIGHT);
	real_t body_v_padding = body_style_box->get_content_margin(SIDE_TOP) + body_style_box->get_content_margin(SIDE_BOTTOM);

	// Determine tooltip width based on max width, header width, and body visibility
	real_t header_width = header_content_size.width + header_h_padding;
	real_t body_width = body_content_size.width + body_h_padding;
	real_t tooltip_width = MIN(MAX_WIDTH, MAX(header_width, body_label->is_visible() ? body_width : 0) + 10); // TODO: Should be +2, but +10 is needed and I'm not sure why.

	// Set sizes
	header_label->set_custom_minimum_size(Size2(tooltip_width, -1)); // TODO: Calculate this accurately instead of using -1.
	body_label->set_custom_minimum_size(Size2(tooltip_width, MIN(body_label->get_content_height() + body_v_padding, 400)));
	set_size(Vector2(tooltip_width, -1));
}

Vector2 SymbolTooltip::_calculate_tooltip_position(const String &p_symbol_word, const Vector2 &p_mouse_position) {
	Vector2 line_col = text_editor->get_line_column_at_pos(p_mouse_position);
	int row = line_col.y;
	int col = line_col.x;
	int num_lines = text_editor->get_line_count();
	if (row >= 0 && row < num_lines) {
		String line = text_editor->get_line(row);
		int symbol_col = _get_word_pos_under_mouse(p_symbol_word, line, col);
		if (symbol_col >= 0) {
			Vector2 symbol_position = text_editor->get_pos_at_line_column(row, symbol_col);
			return text_editor->get_global_position() + symbol_position;
		}
	}
	return Vector2(-1, -1); // Indicates an invalid position.
}

Vector<const DocData::ClassDoc*> SymbolTooltip::_get_built_in_class_docs() {
	const HashMap<String, DocData::ClassDoc> &class_list = EditorHelp::get_doc_data()->class_list;
	Vector<const DocData::ClassDoc*> class_docs;
	class_docs.append(&class_list["@GDScript"]);
	class_docs.append(&class_list["@GlobalScope"]);
	return class_docs;
}

ExtendGDScriptParser *SymbolTooltip::_get_script_parser(const Ref<Script> &p_script) {
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
const lsp::DocumentSymbol *SymbolTooltip::_get_member_symbol(
		HashMap<String, const lsp::DocumentSymbol *> &p_members,
		const String &p_symbol_word) { // , const Vector2 &symbol_position) {
	// Use a queue to implement breadth-first search.
	std::queue<const lsp::DocumentSymbol *> queue;

	// Add all members to the queue.
	for (const KeyValue<String, const lsp::DocumentSymbol *> &E : p_members) {
		queue.push(E.value);
	}

	// While there are still elements in the queue.
	while (!queue.empty()) {
		// Get the next symbol.
		const lsp::DocumentSymbol *symbol = queue.front();
		queue.pop();

		// If the name matches, return the symbol.
		if (symbol->name == p_symbol_word && !symbol->detail.is_empty()) { // && symbol->range.is_point_inside(symbol_position)) {
			return symbol;
		}

		// Add the children to the queue for later processing.
		for (int i = 0; i < symbol->children.size(); ++i) {
			queue.push(&symbol->children[i]);
		}
	}

	return nullptr; // If the symbol is not found, return nullptr.
}

const DocData::MethodDoc *SymbolTooltip::_get_class_method_doc(const DocData::ClassDoc *p_class_doc, const String &p_symbol_word) {
	for (int i = 0; i < p_class_doc->methods.size(); ++i) {
		const DocData::MethodDoc &method_doc = p_class_doc->methods[i];

		if (method_doc.name == p_symbol_word) {
			return &method_doc;
		}
	}
	return nullptr;
}

const DocData::ConstantDoc *SymbolTooltip::_get_class_constant_doc(const DocData::ClassDoc *p_class_doc, const String &p_symbol_word) {
	for (int i = 0; i < p_class_doc->constants.size(); ++i) {
		const DocData::ConstantDoc &constant_doc = p_class_doc->constants[i];

		if (constant_doc.name == p_symbol_word) {
			if (constant_doc.is_value_valid) {
				return &constant_doc;
			}
		}
	}
	return nullptr;
}

String SymbolTooltip::_get_class_method_doc_definition(const DocData::MethodDoc *p_method_doc) {
	String definition = "func " + p_method_doc->name + "(";
	String parameters;
	for (int i = 0; i < p_method_doc->arguments.size(); i++) {
		const DocData::ArgumentDoc &parameter = p_method_doc->arguments[i];
		if (i > 0) {
			parameters += ", ";
		}
		parameters += parameter.name;
		if (!parameter.type.is_empty()) {
			parameters += ": " + parameter.type;
		}
	}
	if (p_method_doc->qualifiers.contains("vararg")) {
		parameters += parameters.is_empty() ? "..." : ", ...";
	}

	String return_type = p_method_doc->return_type;
	if (return_type.is_empty()) {
		return_type = "void";
	}
	definition += parameters + ") -> " + return_type;
	return definition;
}

String SymbolTooltip::_get_header_content(const String &p_symbol_word, const lsp::DocumentSymbol *p_member_symbol, String p_built_in_header) {
	if (p_member_symbol && !p_member_symbol->reduced_detail.is_empty()) {
		return p_member_symbol->reduced_detail;
	}
	if (!p_built_in_header.is_empty()) {
		return p_built_in_header;
	}
	return p_symbol_word;
}

String SymbolTooltip::_get_body_content(const lsp::DocumentSymbol *p_member_symbol, String p_built_in_body) {
	String body_content = "";
	if (p_member_symbol) {
		// Append relevant docstrings.
		body_content += p_member_symbol->documentation.replace("\n ", " ");
	}
	if (!p_built_in_body.is_empty()) {
		// Append official documentation.
		body_content += p_built_in_body;
	}
	return body_content;
}

void SymbolTooltip::_update_header_label(const String &p_header_content) {
	// Set the tooltip's header text.
	header_label->set_text(p_header_content);
}

void SymbolTooltip::_update_body_label(const String &p_body_content) {
	// Set the tooltip's body text.
	if (p_body_content.is_empty()) {
		if (body_label->is_visible()) {
			body_label->hide();
		}
		return;
	}
	body_label->clear();
	_add_text_to_rt(p_body_content, body_label, layout_container);
	if (!body_label->is_visible()) {
		body_label->show();
	}
}

int SymbolTooltip::_get_word_pos_under_mouse(const String &p_symbol_word, const String &p_search, int p_mouse_x) const {
	// Created this because _get_column_pos_of_word() only gets the column position of the first occurrence of the word in the line.

	// Early exit if the symbol word is empty, the search string is empty, or the mouse is outside the string.
	if (p_symbol_word.is_empty() || p_search.is_empty() || p_mouse_x < 0 || p_mouse_x >= p_search.length()) {
		return -1;
	}

	int start = p_mouse_x;
	int end = p_mouse_x;

	// Extend the start and end until they reach the beginning or end of the word.
	while (start > 0 && is_ascii_identifier_char(p_search[start - 1])) {
		start--;
	}
	while (end < p_search.length() && is_ascii_identifier_char(p_search[end])) {
		end++;
	}

	String word_under_mouse = p_search.substr(start, end - start);

	// If the word under the mouse matches the symbol word, return the start position.
	if (word_under_mouse == p_symbol_word) {
		return start + 1; // Note: +1 is added to account for zero-based indexing.
	}

	return -1; // Return -1 if no match is found.
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
	style_box->set_content_margin_individual(15, 10, 15, 10);

	// Set the style boxes for the TextEdit
	theme->set_stylebox("normal", "TextEdit", style_box);
	theme->set_stylebox("focus", "TextEdit", style_box);
	theme->set_stylebox("readonly", "TextEdit", style_box);

	// Set the font color.
	theme->set_color("font_color", "TextEdit", Color(1, 1, 1));

	return theme;
}

Ref<Theme> SymbolTooltip::_create_body_label_theme() {
	Ref<Theme> theme = memnew(Theme); // TODO: Get the global theme instead (e.g. dark mode, light mode).

	Ref<StyleBoxFlat> style_box = memnew(StyleBoxFlat);
	style_box->set_draw_center(false);
	style_box->set_border_width(SIDE_TOP, 1);
	style_box->set_border_color(Color(0.8, 0.81, 0.82, 0.27)); // Set the border color (RGBA).
	style_box->set_content_margin_individual(15, 10, 15, 10);
	theme->set_stylebox("normal", "RichTextLabel", style_box);

	return theme;
}

/*const GDScriptParser::ClassNode::Member *find_symbol(const GDScriptParser::ClassNode *node, const String &p_symbol_word) {
	for (int i = 0; i < node->members.size(); ++i) {
		const GDScriptParser::ClassNode::Member &member = node->members[i];

		if (member.get_name() == p_symbol_word) {
			// Found the symbol.
			return &member;
		} else if (member.type == GDScriptParser::ClassNode::Member::CLASS) {
			const GDScriptParser::ClassNode::Member *found_symbol = find_symbol(member.m_class, p_symbol_word);
			if (found_symbol) {
				return found_symbol;
			}
		}
	}

	return nullptr;
}*/

/*String SymbolTooltip::_get_doc_of_word(const String &p_symbol_word) {
	String documentation;

const HashMap<String, DocData::ClassDoc> &class_list = EditorHelp::get_doc_data()->class_list;
for (const KeyValue<String, DocData::ClassDoc> &E : class_list) {
	const DocData::ClassDoc &class_doc = E.value;

	if (class_doc.name == p_symbol_word) {
		documentation = class_doc.brief_description.strip_edges(); //class_doc.brief_description + "\n\n" + class_doc.description;
		if (documentation.is_empty()) {
			documentation = class_doc.description.strip_edges();
		}
		break;
	}

	for (int i = 0; i < class_doc.methods.size(); ++i) {
		const DocData::MethodDoc &method_doc = class_doc.methods[i];

		if (method_doc.name == p_symbol_word) {
			documentation = method_doc.description.strip_edges();
			break;
		}
	}

	for (int i = 0; i < class_doc.constants.size(); ++i) {
		const DocData::ConstantDoc &constant_doc = class_doc.constants[i];

		if (constant_doc.name == p_symbol_word) {
			if (constant_doc.is_value_valid) {
				documentation = constant_doc.value.strip_edges();
			}
			break;
		}
	}

	if (!documentation.is_empty()) {
		break;
	}
}

*//*if (!documentation.is_empty()) {
		print_line(vformat("Documentation for %s:\n%s", p_symbol_word, documentation));
}*//*
		return documentation;
}*/

// Copied from script_text_editor.cpp
/*static Node *_find_node_for_script(Node *p_base, Node *p_current, const Ref<Script> &p_script) {
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
}*/

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


/**************************************************************************/
/*  symbol_tooltip.h                                                      */
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

#ifndef SYMBOL_TOOLTIP_H
#define SYMBOL_TOOLTIP_H

#include "code_editor.h"
#include "modules/gdscript/gdscript_parser.h"
#include "modules/gdscript/language_server/gdscript_extend_parser.h"
#include "scene/gui/box_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/rich_text_label.h"

class SymbolTooltip : public PanelContainer {
	GDCLASS(SymbolTooltip, PanelContainer);

	CodeEdit *text_editor = nullptr;
	PanelContainer *panel_container = nullptr;
	VBoxContainer *layout_container = nullptr;
	TextEdit *header_label = nullptr;
	RichTextLabel *body_label = nullptr;
	Timer *tooltip_delay = nullptr;
	String last_symbol_word;
	bool mouse_inside;

	// String _get_doc_of_word(const String &p_symbol_word);
	String _get_class_method_documentation(const DocData::ClassDoc &p_class_doc, const String &p_symbol_word);
	String _get_class_constant_documentation(const DocData::ClassDoc &p_class_doc, const String &p_symbol_word);
	String _get_built_in_documentation(const String &p_symbol_word);
	String _get_header_content(const String &p_symbol_word, const lsp::DocumentSymbol *p_member_symbol, bool p_is_builtin);
	String _get_body_content(const lsp::DocumentSymbol *p_member_symbol, String p_builtin_doc);
	void _update_header_label(const String &p_header_content);
	void _update_body_label(const String &p_body_content);
	Ref<Theme> _create_panel_theme();
	Ref<Theme> _create_header_label_theme();
	Ref<Theme> _create_body_label_theme();
	int _get_word_pos_under_mouse(const String &p_symbol_word, const String &p_search, int p_mouse_x) const;
	void _on_tooltip_delay_timeout();
	void _close_tooltip();
	String _get_symbol_word(const Vector2 &p_mouse_position);
	Vector2 _calculate_tooltip_position(const String &p_symbol_word, const Vector2 &p_mouse_position);
	void _update_tooltip_size();
	void _update_tooltip_content(const Ref<Script> &p_script, const String &p_symbol_word);
	ExtendGDScriptParser *_get_script_parser(const Ref<Script> &p_script);
	const lsp::DocumentSymbol *_get_member_symbol(HashMap<String, const lsp::DocumentSymbol *> &p_members, const String &p_symbol_word);

protected:
	void _notification(int p_what);

public:
	void update_symbol_tooltip(const Vector2 &p_mouse_position, const Ref<Script> &p_script);
	SymbolTooltip(CodeEdit *p_text_editor);
	~SymbolTooltip();
};

// static Node *_find_node_for_script(Node *p_base, Node *p_current, const Ref<Script> &p_script);
// static const GDScriptParser::ClassNode *get_ast_tree(const Ref<Script> &p_script);
// static ExtendGDScriptParser *get_script_parser(const Ref<Script> &p_script);
// const lsp::DocumentSymbol *get_member_symbol(HashMap<String, const lsp::DocumentSymbol *> &p_members, const String &p_symbol_word);

#endif // SYMBOL_TOOLTIP_H

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

//#include "core/script_language.h"
#include "code_editor.h"
#include "scene/gui/box_container.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/rich_text_label.h"

class SymbolTooltip : public PanelContainer {
	GDCLASS(SymbolTooltip, PanelContainer);

	//Ref<Script> script;
	CodeTextEditor *code_editor = nullptr;
	VBoxContainer *layout_container = nullptr;
	RichTextLabel *header_label = nullptr;
	RichTextLabel *body_label = nullptr;

	String _get_doc_of_word(const String &symbol_word);
	void _update_header_label(const String &symbol_word);
	void _update_body_label(const String &documentation);
	Ref<Theme> _create_panel_theme();
	Ref<Theme> _create_header_label_theme();
	Ref<Theme> _create_body_label_theme();
	int _get_column_pos_of_word(const String &p_key, const String &p_search, uint32_t p_search_flags, int p_from_column) const;

	String symbol_word;
	String header_content;
	String body_content;
	Vector2 symbol_position;
	//Vector2 tooltip_position;
public:
	void update_symbol_tooltip(const Vector2 &mouse_position);
	void show_tooltip();
	void hide_tooltip();
	SymbolTooltip(CodeTextEditor* code_editor);
};

//static Node *_find_node_for_script(Node *p_base, Node *p_current, const Ref<Script> &p_script);

#endif //GODOT_SYMBOL_TOOLTIP_H

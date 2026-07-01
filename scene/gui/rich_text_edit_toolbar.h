/**************************************************************************/
/*  rich_text_edit_toolbar.h                                              */
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

#pragma once

#include "scene/gui/box_container.h"

class Button;
class ColorPickerButton;
class Label;
class LineEdit;
class PopupPanel;
class RichTextEdit;
class SpinBox;

class RichTextEditToolbar : public HBoxContainer {
	GDCLASS(RichTextEditToolbar, HBoxContainer);

	NodePath rich_text_edit_path;
	ObjectID rich_text_edit_id;

	Button *bold_button = nullptr;
	Button *italic_button = nullptr;
	Button *underline_button = nullptr;
	Button *strikethrough_button = nullptr;
	Button *quote_button = nullptr;
	Button *outline_button = nullptr;
	Button *link_button = nullptr;
	Button *unordered_list_button = nullptr;
	Button *ordered_list_button = nullptr;
	Button *indent_decrease_button = nullptr;
	Button *indent_increase_button = nullptr;
	Button *alignment_button = nullptr;
	Button *align_left_button = nullptr;
	Button *align_center_button = nullptr;
	Button *align_right_button = nullptr;
	Label *outline_color_label = nullptr;
	Label *outline_size_label = nullptr;
	ColorPickerButton *color_button = nullptr;
	ColorPickerButton *bg_color_button = nullptr;
	ColorPickerButton *outline_color_button = nullptr;
	SpinBox *font_size_spin = nullptr;
	SpinBox *outline_size_spin = nullptr;
	PopupPanel *alignment_popup = nullptr;
	PopupPanel *outline_popup = nullptr;
	PopupPanel *link_popup = nullptr;
	LineEdit *link_line_edit = nullptr;
	bool updating_controls = false;
	bool color_picker_open = false;
	bool bg_color_picker_open = false;
	Color color_before_picker_open;
	Color bg_color_before_picker_open;
	Color pending_picker_color;
	Color pending_bg_picker_color;

	RichTextEdit *_get_rich_text_edit() const;
	void _resolve_rich_text_edit();
	void _update_controls_from_target();
	void _target_caret_changed();
	void _apply_toolbar_icons();
	void _apply_toolbar_style();
	void _apply_toolbar_button_style(Button *p_button);
	void _apply_toolbar_color_button_style(Button *p_button);
	void _apply_outline_button_color();
	void _apply_dropdown_font_color();
	void _set_dropdown_button_arrow(Button *p_button, const String &p_fallback_text);
	void _set_button_icon_or_text(Button *p_button, const StringName &p_icon_name, const String &p_fallback_text);

	void _pressed_bold();
	void _pressed_italic();
	void _pressed_underline();
	void _pressed_strikethrough();
	void _pressed_quote();
	void _pressed_outline();
	void _pressed_alignment();
	void _pressed_link();
	void _link_apply_pressed();
	void _link_clear_pressed();
	void _alignment_selected(int p_id);
	void _pressed_unordered_list();
	void _pressed_ordered_list();
	void _color_picker_created();
	void _color_popup_about_to_popup();
	void _color_popup_closed();
	void _color_changed(const Color &p_color);
	void _bg_color_picker_created();
	void _bg_color_popup_about_to_popup();
	void _bg_color_popup_closed();
	void _bg_color_changed(const Color &p_color);
	void _outline_color_changed(const Color &p_color);
	void _font_size_changed(double p_value);
	void _outline_size_changed(double p_value);
	void _pressed_decrease_indent();
	void _pressed_increase_indent();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_rich_text_edit_path(const NodePath &p_path);
	NodePath get_rich_text_edit_path() const;

	RichTextEdit *get_rich_text_edit() const;

	RichTextEditToolbar();
};

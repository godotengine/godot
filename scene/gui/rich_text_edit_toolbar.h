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
#include "scene/gui/button.h"
#include "scene/gui/rich_text_edit.h"

class ColorPicker;
class FileDialog;
class Label;
class LineEdit;
class Panel;
class PanelContainer;
class PopupPanel;
class HSeparator;
class SpinBox;
class TextureRect;
class VSeparator;

// Builds the tooltip described by the toolbar's own `tooltip_*` theme items.
// Tooltips are owned by the control they are attached to, so the toolbar
// buttons need their own type to opt out of the default tooltip look.
class RichTextEditToolbarButton : public Button {
	GDCLASS(RichTextEditToolbarButton, Button);

public:
	static Control *make_toolbar_tooltip(const Control *p_owner, const String &p_text);

	virtual Control *make_custom_tooltip(const String &p_text) const override;
};

class RichTextEditToolbar : public HBoxContainer {
	GDCLASS(RichTextEditToolbar, HBoxContainer);

	NodePath rich_text_edit_path;
	ObjectID rich_text_edit_id;

	Button *bold_button = nullptr;
	Button *italic_button = nullptr;
	Button *underline_button = nullptr;
	Button *strikethrough_button = nullptr;
	Button *quote_button = nullptr;
	Button *horizontal_rule_button = nullptr;
	Button *outline_button = nullptr;
	Button *link_button = nullptr;
	Button *insert_image_button = nullptr;
	Button *line_height_button = nullptr;
	Button *unordered_list_button = nullptr;
	Button *ordered_list_button = nullptr;
	Button *indent_decrease_button = nullptr;
	Button *indent_increase_button = nullptr;
	Button *alignment_button = nullptr;
	Button *align_left_button = nullptr;
	Button *align_center_button = nullptr;
	Button *align_right_button = nullptr;
	Button *font_color_button = nullptr;
	Button *bg_color_button = nullptr;
	Button *bg_color_none_button = nullptr;
	Button *link_apply_button = nullptr;
	Button *link_cancel_button = nullptr;
	Label *link_url_label = nullptr;
	Label *line_height_presets_label = nullptr;
	Label *line_height_custom_label = nullptr;
	SpinBox *font_size_spin = nullptr;
	SpinBox *outline_size_spin = nullptr;
	PanelContainer *font_size_field = nullptr;
	PanelContainer *outline_size_field = nullptr;
	PopupPanel *alignment_popup = nullptr;
	PopupPanel *link_popup = nullptr;
	PopupPanel *line_height_popup = nullptr;
	PopupPanel *font_color_popup = nullptr;
	PopupPanel *bg_color_popup = nullptr;
	PopupPanel *outline_color_popup = nullptr;
	ColorPicker *font_color_picker = nullptr;
	ColorPicker *bg_color_picker = nullptr;
	ColorPicker *outline_color_picker = nullptr;
	LineEdit *link_line_edit = nullptr;
	LineEdit *line_height_line_edit = nullptr;
	Button *line_height_cancel_button = nullptr;
	Button *line_height_apply_button = nullptr;
	Vector<Button *> line_height_preset_buttons;
	FileDialog *image_file_dialog = nullptr;

	// Color previews drawn underneath the icon of the three color buttons.
	Panel *font_color_bar = nullptr;
	Panel *outline_color_bar = nullptr;
	Panel *bg_color_swatch = nullptr;
	TextureRect *bg_color_checker = nullptr;
	// "No Color" row at the top of the background color dropdown.
	Panel *bg_color_none_swatch = nullptr;
	TextureRect *bg_color_none_checker = nullptr;
	HSeparator *bg_color_none_separator = nullptr;

	Vector<VSeparator *> separators;
	Vector<TextureRect *> dropdown_carets;
	Vector<Control *> paddings;

	bool updating_controls = false;
	bool color_picker_open = false;
	bool bg_color_picker_open = false;
	// Mirrors ColorPickerButton: a click that closes a popup can also reach the
	// button that opened it, which would immediately reopen the popup.
	bool alignment_popup_was_open = false;
	bool link_popup_was_open = false;
	bool line_height_popup_was_open = false;
	bool font_color_popup_was_open = false;
	bool bg_color_popup_was_open = false;
	bool outline_color_popup_was_open = false;
	Color font_color_value = Color(0, 0, 0);
	Color outline_color_value = Color(0, 0, 0);
	Color bg_color_value = Color(1, 1, 1);
	bool bg_color_assigned = false;
	Color color_before_picker_open;
	Color bg_color_before_picker_open;
	bool bg_color_assigned_before_picker_open = false;
	Color pending_picker_color;
	Color pending_bg_picker_color;

	RichTextEdit *_get_rich_text_edit() const;
	void _resolve_rich_text_edit();
	bool _get_current_style(RichTextEdit::TextStyle &r_style) const;
	void _update_controls_from_target();
	void _target_caret_changed();

	void _apply_toolbar_icons();
	void _apply_toolbar_style();
	void _style_tool_button(Button *p_button);
	void _style_dropdown_button(Button *p_button);
	void _style_menu_item_button(Button *p_button);
	void _style_none_button();
	void _style_action_button(Button *p_button, const StringName &p_normal, const StringName &p_hover, const Color &p_font_color, const Color &p_hover_font_color);
	void _style_number_field(PanelContainer *p_field, SpinBox *p_spin);
	void _style_line_edit(LineEdit *p_line_edit, const StringName &p_normal);
	void _style_popup(PopupPanel *p_popup, const StringName &p_panel);
	void _style_dropdown_label(Label *p_label);
	void _apply_color_button_previews();
	void _set_button_icon(Button *p_button, const StringName &p_icon_name);
	void _ensure_font_color_picker();
	void _ensure_bg_color_picker();
	void _ensure_outline_color_picker();
	Button *_get_dropdown_button(int p_dropdown) const;
	PopupPanel *_get_dropdown_popup(int p_dropdown) const;
	void _dropdown_button_down(int p_dropdown);
	void _dropdown_popup_hidden(int p_dropdown);
	void _popup_below(PopupPanel *p_popup, Button *p_button, int p_min_width);
	void _popup_below_right(PopupPanel *p_popup, Button *p_button, int p_min_width);

	void _pressed_bold();
	void _pressed_italic();
	void _pressed_underline();
	void _pressed_strikethrough();
	void _pressed_quote();
	void _pressed_horizontal_rule();
	void _pressed_outline_color();
	void _pressed_alignment();
	void _pressed_link();
	void _pressed_insert_image();
	void _image_file_selected(const String &p_path);
	void _pressed_line_height();
	void _line_height_preset_selected(const String &p_value);
	void _line_height_apply_pressed();
	void _line_height_cancel_pressed();
	void _pressed_font_color();
	void _pressed_bg_color();
	void _link_apply_pressed();
	void _link_cancel_pressed();
	void _alignment_selected(int p_id);
	void _pressed_unordered_list();
	void _pressed_ordered_list();
	void _color_popup_about_to_popup();
	void _color_popup_closed();
	void _color_changed(const Color &p_color);
	void _bg_color_popup_about_to_popup();
	void _bg_color_popup_closed();
	void _bg_color_changed(const Color &p_color);
	void _bg_color_cleared();
	void _outline_color_changed(const Color &p_color);
	void _outline_color_popup_closed();
	void _font_size_changed(double p_value);
	void _outline_size_changed(double p_value);
	void _pressed_decrease_indent();
	void _pressed_increase_indent();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual String get_tooltip(const Point2 &p_pos) const override;
	virtual Control *make_custom_tooltip(const String &p_text) const override;

	void set_rich_text_edit_path(const NodePath &p_path);
	NodePath get_rich_text_edit_path() const;

	RichTextEdit *get_rich_text_edit() const;

	RichTextEditToolbar();
};

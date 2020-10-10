/*************************************************************************/
/*  color_picker.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef COLOR_PICKER_H
#define COLOR_PICKER_H

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/check_button.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/popup.h"
#include "scene/gui/separator.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/texture_rect.h"

class ColorPicker : public BoxContainer {
	GDCLASS(ColorPicker, BoxContainer);

private:
	Control *screen;
	Control *uv_edit;
	Control *w_edit;
	TextureRect *sample;
	TextureRect *preset;
	HBoxContainer *preset_container;
	HBoxContainer *preset_container2;
	HSeparator *preset_separator;
	Button *bt_add_preset;
	List<Color> presets;
	Button *btn_pick;
	CheckButton *btn_hsv;
	CheckButton *btn_raw;
	HSlider *scroll[4];
	SpinBox *values[4];
	Label *labels[4];
	Button *text_type;
	LineEdit *c_text;
	bool edit_alpha;
	Size2i ms;
	bool text_is_constructor;
	int presets_per_row;

	Color color;
	bool raw_mode_enabled;
	bool hsv_mode_enabled;
	bool deferred_mode_enabled;
	bool updating;
	bool changing_color;
	bool presets_enabled;
	bool presets_visible;
	float h, s, v;
	Color last_hsv;

	void _html_entered(const String &p_html);
	void _value_changed(double);
	void _update_controls();
	void _update_color(bool p_update_sliders = true);
	void _update_presets();
	void _update_text_value();
	void _text_type_toggled();
	void _sample_draw();
	void _hsv_draw(int p_which, Control *c);

	void _uv_input(const Ref<InputEvent> &p_event);
	void _w_input(const Ref<InputEvent> &p_event);
	void _preset_input(const Ref<InputEvent> &p_event);
	void _screen_input(const Ref<InputEvent> &p_event);
	void _add_preset_pressed();
	void _screen_pick_pressed();
	void _focus_enter();
	void _focus_exit();
	void _html_focus_exit();

protected:
	void _notification(int);
	static void _bind_methods();

public:
	void set_edit_alpha(bool p_show);
	bool is_editing_alpha() const;

	void _set_pick_color(const Color &p_color, bool p_update_sliders);
	void set_pick_color(const Color &p_color);
	Color get_pick_color() const;

	void add_preset(const Color &p_color);
	void erase_preset(const Color &p_color);
	PackedColorArray get_presets() const;

	void set_hsv_mode(bool p_enabled);
	bool is_hsv_mode() const;

	void set_raw_mode(bool p_enabled);
	bool is_raw_mode() const;

	void set_deferred_mode(bool p_enabled);
	bool is_deferred_mode() const;

	void set_presets_enabled(bool p_enabled);
	bool are_presets_enabled() const;

	void set_presets_visible(bool p_visible);
	bool are_presets_visible() const;

	void set_focus_on_line_edit();

	ColorPicker();
};

class ColorPickerButton : public Button {
	GDCLASS(ColorPickerButton, Button);

	PopupPanel *popup;
	ColorPicker *picker;
	Color color;
	bool edit_alpha;

	void _color_changed(const Color &p_color);
	void _modal_closed();

	virtual void pressed() override;

	void _update_picker();

protected:
	void _notification(int);
	static void _bind_methods();

public:
	void set_pick_color(const Color &p_color);
	Color get_pick_color() const;

	void set_edit_alpha(bool p_show);
	bool is_editing_alpha() const;

	ColorPicker *get_picker();
	PopupPanel *get_popup();

	ColorPickerButton();
};

#endif // COLOR_PICKER_H

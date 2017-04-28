/*************************************************************************/
/*  color_picker.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tool_button.h"

class ColorPicker : public BoxContainer {

	GDCLASS(ColorPicker, BoxContainer);

private:
	Control *screen;
	Image last_capture;
	Control *uv_edit;
	Control *w_edit;
	TextureRect *sample;
	TextureRect *preset;
	Button *bt_add_preset;
	List<Color> presets;
	ToolButton *btn_pick;
	CheckButton *btn_mode;
	HSlider *scroll[4];
	SpinBox *values[4];
	Label *labels[4];
	Button *text_type;
	LineEdit *c_text;
	bool edit_alpha;
	Size2i ms;
	bool text_is_constructor;

	Color color;
	bool raw_mode_enabled;
	bool updating;
	bool changing_color;
	float h, s, v;
	Color last_hsv;

	void _html_entered(const String &p_html);
	void _value_changed(double);
	void _update_controls();
	void _update_color();
	void _update_presets();
	void _update_text_value();
	void _text_type_toggled();
	void _sample_draw();
	void _hsv_draw(int p_wich, Control *c);

	void _uv_input(const InputEvent &p_input);
	void _w_input(const InputEvent &p_input);
	void _preset_input(const InputEvent &p_input);
	void _screen_input(const InputEvent &p_input);
	void _add_preset_pressed();
	void _screen_pick_pressed();

protected:
	void _notification(int);
	static void _bind_methods();

public:
	void set_edit_alpha(bool p_show);
	bool is_editing_alpha() const;

	void set_pick_color(const Color &p_color);
	Color get_pick_color() const;

	void add_preset(const Color &p_color);
	void set_raw_mode(bool p_enabled);
	bool is_raw_mode() const;

	void set_focus_on_line_edit();

	ColorPicker();
};

class ColorPickerButton : public Button {

	GDCLASS(ColorPickerButton, Button);

	PopupPanel *popup;
	ColorPicker *picker;

	void _color_changed(const Color &p_color);
	virtual void pressed();

protected:
	void _notification(int);
	static void _bind_methods();

public:
	void set_pick_color(const Color &p_color);
	Color get_pick_color() const;

	void set_edit_alpha(bool p_show);
	bool is_editing_alpha() const;

	ColorPicker *get_picker();

	ColorPickerButton();
};

#endif // COLOR_PICKER_H

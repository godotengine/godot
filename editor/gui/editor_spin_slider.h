/**************************************************************************/
/*  editor_spin_slider.h                                                  */
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

#include "scene/gui/range.h"

class TextureRect;
class LineEdit;

class EditorSpinSlider : public Range {
	GDCLASS(EditorSpinSlider, Range);

	enum ControlState {
		CONTROL_STATE_USING_SLIDER,
		CONTROL_STATE_USING_ARROWS,
		CONTROL_STATE_HIDDEN,
	};
	ControlState state = CONTROL_STATE_USING_SLIDER;

	String label;
	String suffix;
	String prev_value;

	RID text_rid;
	bool text_dirty = true;
	int suffix_start = 0;

	Point2 buttons_offset = Point2(-1, -1);
	MouseButton held_button_index = MouseButton::NONE;
	bool up_button_held = false;
	bool down_button_held = false;
	bool up_button_hovered = false;
	bool down_button_hovered = false;

	TextureRect *grabber = nullptr;
	int grabber_range = 1;

	bool mouse_over_spin = false;
	bool mouse_over_grabber = false;

	bool warping_mouse = false;
	bool warping_mouse_queue_disable = false;
	Size2 warping_mouse_offset = Size2();

	bool grabbing_grabber = false;
	int grabbing_from = 0;
	float grabbing_ratio = 0.0f;

	bool grabbing_spinner_attempt = false;
	bool grabbing_spinner = false;

	bool read_only = false;
	float grabbing_spinner_dist_cache = 0.0f;
	float grabbing_spinner_speed = 0.0f;
	Vector2 grabbing_spinner_mouse_pos;
	double pre_grab_value = 0.0;

	LineEdit *value_input = nullptr;
	uint64_t value_input_closed_frame = 0;
	bool value_input_dirty = false;

private:
	bool flat = false;
	bool editing_integer = false;
	bool hide_control = false;
	bool float_prefer_arrows = false;
	bool integer_prefer_slider = false;

	void _grab_start();
	void _grab_end();

	void _grabber_gui_input(const Ref<InputEvent> &p_event);
	void _value_input_closed();
	void _value_input_submitted(const String &);
	void _value_focus_exited();
	void _value_input_gui_input(const Ref<InputEvent> &p_event);

	void _evaluate_input_text();

	void _update_grabbing_speed();
	void _update_value_input_stylebox();
	void _ensure_value_input();
	void _draw_spin_slider();
	void _shape();

	bool _update_buttons(const Point2 &p_pos, MouseButton p_button = MouseButton::NONE, bool p_pressed = false);

	void _update_control_state();
	ControlState _get_control_state() const;

	struct ThemeCache {
		Ref<Texture2D> grabber_icon;
		Ref<Texture2D> grabber_highlight_icon;
		Ref<Texture2D> up_icon;
		Ref<Texture2D> down_icon;

		Ref<Font> font;
		int font_size = 0;

		int updown_v_separation = 0;

		Color font_color;
		Color font_uneditable_color;
		Color label_color;
		Color read_only_label_color;

		Ref<StyleBox> normal_style;
		Ref<StyleBox> read_only_style;
		Ref<StyleBox> focus_style;
		Ref<StyleBox> label_bg_style;
		Ref<StyleBox> updown_hovered_style;
		Ref<StyleBox> updown_pressed_style;
	} theme_cache;

protected:
	void _notification(int p_what);
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	static void _bind_methods();
	void _grabber_mouse_entered();
	void _grabber_mouse_exited();
	void _focus_entered(bool p_hide_focus = false);

public:
	String get_tooltip(const Point2 &p_pos) const override;

	String get_text_value() const;
	void set_label(const String &p_label);
	String get_label() const;

	void set_suffix(const String &p_suffix);
	String get_suffix() const;

#ifndef DISABLE_DEPRECATED
	void set_hide_slider(bool p_hide);
	bool is_hiding_slider() const;
#endif

	void set_hide_control(bool p_visible);
	bool is_hiding_control() const;

	void set_editing_integer(bool p_editing_integer);
	bool is_editing_integer() const;

	void set_integer_prefer_slider(bool p_enable);
	bool is_integer_preferring_slider() const;

	void set_float_prefer_arrows(bool p_enable);
	bool is_float_preferring_arrows() const;

	void set_read_only(bool p_enable);
	bool is_read_only() const;

	void set_flat(bool p_enable);
	bool is_flat() const;

	bool is_grabbing() const;

	void setup_and_show() { _focus_entered(); }
	LineEdit *get_line_edit();

	virtual Size2 get_minimum_size() const override;
	EditorSpinSlider();
	~EditorSpinSlider();
};

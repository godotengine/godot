/*************************************************************************/
/*  scroll_container.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SCROLL_CONTAINER_H
#define SCROLL_CONTAINER_H

#include "container.h"

#include "scroll_bar.h"

class ScrollContainer : public Container {
	GDCLASS(ScrollContainer, Container);

	HScrollBar *h_scroll;
	VScrollBar *v_scroll;

	Size2 child_max_size;
	Size2 scroll;

	void update_scrollbars();

	Vector2 drag_speed = Vector2();
	Vector2 drag_accum = Vector2();
	Vector2 drag_from = Vector2();
	float last_drag_time = 0.0;
	bool drag_touching = false;
	bool animating = false;
	bool click_handled = false;
	bool beyond_deadzone = false;

	bool scroll_h = true;
	bool scroll_v = true;

	int deadzone = 0;
	bool follow_focus = false;

	bool always_smoothed;
	float scroll_step;

	float smooth_scroll_duration_button;
	float inertial_scroll_duration_touch;
	float inertial_scroll_duration_current;

	float inertial_time_left;
	Vector2 expected_scroll_value;
	Vector2 inertial_start;
	Vector2 inertial_target;

	void _cancel_drag();
	void _button_scroll(bool horizontal, float amount);
	void _start_inertial_scroll();

	void _check_expected_scroll();
	void _update_expected_scroll();

protected:
	Size2 get_minimum_size() const override;

	void _gui_input(const Ref<InputEvent> &p_gui_input);
	void _update_dimensions();
	void _notification(int p_what);

	void _scroll_moved(float);
	static void _bind_methods();

	bool _updating_scrollbars = false;
	void _update_scrollbar_position();
	void _ensure_focused_visible(Control *p_node);

public:
	int get_v_scroll() const;
	void set_v_scroll(int p_pos);

	int get_h_scroll() const;
	void set_h_scroll(int p_pos);

	void set_enable_h_scroll(bool p_enable);
	bool is_h_scroll_enabled() const;

	void set_enable_v_scroll(bool p_enable);
	bool is_v_scroll_enabled() const;

	int get_deadzone() const;
	void set_deadzone(int p_deadzone);

	float get_scroll_step() const;
	void set_scroll_step(float p_step);

	bool is_following_focus() const;
	void set_follow_focus(bool p_follow);

	bool is_always_smoothed() const;
	void set_always_smoothed(bool p_smoothed);

	HScrollBar *get_h_scrollbar();
	VScrollBar *get_v_scrollbar();

	virtual bool clips_input() const override;

	TypedArray<String> get_configuration_warnings() const override;

	ScrollContainer();
};

#endif

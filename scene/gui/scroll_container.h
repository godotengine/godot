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

	void update_scrollbars();

	Vector2 drag_speed;
	Vector2 drag_accum;
	Vector2 drag_from;
	Vector2 last_drag_accum;
	float time_since_motion = 0.0f;
	bool drag_touching = false;
	bool drag_touching_deaccel = false;
	bool beyond_deadzone = false;

	bool scroll_h = true;
	bool scroll_v = true;

	bool h_scroll_visible = true;
	bool v_scroll_visible = true;

	int deadzone = 0;
	bool follow_focus = false;

	void _cancel_drag();

protected:
	Size2 get_minimum_size() const override;

	void _gui_focus_changed(Control *p_control);
	void _update_dimensions();
	void _notification(int p_what);

	void _scroll_moved(float);
	static void _bind_methods();

	bool _updating_scrollbars = false;
	void _update_scrollbar_position();

public:
	virtual void gui_input(const Ref<InputEvent> &p_gui_input) override;

	void set_h_scroll(int p_pos);
	int get_h_scroll() const;

	void set_v_scroll(int p_pos);
	int get_v_scroll() const;

	void set_enable_h_scroll(bool p_enable);
	bool is_h_scroll_enabled() const;

	void set_enable_v_scroll(bool p_enable);
	bool is_v_scroll_enabled() const;

	void set_h_scroll_visible(bool p_visible);
	bool is_h_scroll_visible() const;

	void set_v_scroll_visible(bool p_visible);
	bool is_v_scroll_visible() const;

	int get_deadzone() const;
	void set_deadzone(int p_deadzone);

	bool is_following_focus() const;
	void set_follow_focus(bool p_follow);

	HScrollBar *get_h_scrollbar();
	VScrollBar *get_v_scrollbar();
	void ensure_control_visible(Control *p_control);

	TypedArray<String> get_configuration_warnings() const override;

	ScrollContainer();
};

#endif

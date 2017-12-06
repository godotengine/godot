/*************************************************************************/
/*  scroll_bar.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef SCROLL_BAR_H
#define SCROLL_BAR_H

#include "scene/gui/range.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class ScrollBar : public Range {

	GDCLASS(ScrollBar, Range);

	enum HighlightStatus {
		HIGHLIGHT_NONE,
		HIGHLIGHT_DECR,
		HIGHLIGHT_RANGE,
		HIGHLIGHT_INCR,
	};

	static bool focus_by_default;

	Orientation orientation;
	Size2 size;
	float custom_step;

	HighlightStatus highlight;

	struct Drag {

		bool active;
		float pos_at_click;
		float value_at_click;
	} drag;

	double get_grabber_size() const;
	double get_grabber_min_size() const;
	double get_area_size() const;
	double get_area_offset() const;
	double get_click_pos(const Point2 &p_pos) const;
	double get_grabber_offset() const;

	static void set_can_focus_by_default(bool p_can_focus);

	Node *drag_slave;
	NodePath drag_slave_path;

	Vector2 drag_slave_speed;
	Vector2 drag_slave_accum;
	Vector2 drag_slave_from;
	Vector2 last_drag_slave_accum;
	float last_drag_slave_time;
	float time_since_motion;
	bool drag_slave_touching;
	bool drag_slave_touching_deaccel;
	bool click_handled;

	bool scrolling;
	double target_scroll;
	bool smooth_scroll_enabled;

	void _drag_slave_exit();
	void _drag_slave_input(const Ref<InputEvent> &p_input);

	void _gui_input(Ref<InputEvent> p_event);

protected:
	void _notification(int p_what);

	static void _bind_methods();

public:
	void set_custom_step(float p_custom_step);
	float get_custom_step() const;

	void set_drag_slave(const NodePath &p_path);
	NodePath get_drag_slave() const;

	void set_smooth_scroll_enabled(bool p_enable);
	bool is_smooth_scroll_enabled() const;

	virtual Size2 get_minimum_size() const;
	ScrollBar(Orientation p_orientation = VERTICAL);
	~ScrollBar();
};

class HScrollBar : public ScrollBar {

	GDCLASS(HScrollBar, ScrollBar);

public:
	HScrollBar() :
			ScrollBar(HORIZONTAL) { set_v_size_flags(0); }
};

class VScrollBar : public ScrollBar {

	GDCLASS(VScrollBar, ScrollBar);

public:
	VScrollBar() :
			ScrollBar(VERTICAL) { set_h_size_flags(0); }
};

#endif

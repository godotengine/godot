/**************************************************************************/
/*  resizable_scroll_bar.h                                                */
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

#include "scroll_bar.h"

class ResizableScrollBar : public ScrollBar {
	GDCLASS(ResizableScrollBar, ScrollBar);

	enum HighlightStatus {
		HIGHLIGHT_NONE,
		HIGHLIGHT_DECR,
		HIGHLIGHT_RANGE,
		HIGHLIGHT_INCR,
		HIGHLIGHT_MIN_HANDLE,
		HIGHLIGHT_MAX_HANDLE,
	};

	Orientation orientation;

	struct Drag {
		bool grabber_being_dragged = false;
		bool min_handle_being_dragged = false;
		bool max_handle_being_dragged = false;
		float pos_at_click = 0.0;
		float value_at_click = 0.0;
		float end_page_at_click = 0.0;
		float start_page_at_click = 0.0;
		float start_page_at_drag = 0.0;
		float end_page_at_drag = 0.0;
		float handle_offset = 0.0;
	} drag;

	HighlightStatus highlight = HIGHLIGHT_NONE;

	double ratio_to_value(double p_value);
	double get_handle_size();

protected:
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	double get_end_page();
	double get_start_page();
	double get_end_page_at_drag();
	double get_start_page_at_drag();

	double get_handle_min_size();

	bool is_min_handle_being_dragged();
	bool is_max_handle_being_dragged();

	ResizableScrollBar(Orientation p_orientation = HORIZONTAL);
	~ResizableScrollBar();
};

class HResizableScrollBar : public ResizableScrollBar {
	GDCLASS(HResizableScrollBar, ResizableScrollBar);

public:
	HResizableScrollBar() :
			ResizableScrollBar(HORIZONTAL) { set_v_size_flags(0); }
};

// TODO: Implement vertical Resizable ScrollBar

// class VResizableScrollBar : public ResizableScrollBar {
// 	GDCLASS(VResizableScrollBar, ResizableScrollBar);

// public:
// 	VResizableScrollBar() :
// 			ResizableScrollBar(VERTICAL) { set_h_size_flags(0); }
// };

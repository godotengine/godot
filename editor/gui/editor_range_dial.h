/**************************************************************************/
/*  editor_range_dial.h                                                   */
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

#include "scene/gui/popup.h"
#include "scene/gui/range.h"

class EditorRangeDial : public Range {
	GDCLASS(EditorRangeDial, Range);

	double zoom;
	double value_no_snap;

	double zoom_speed;
	bool inverted_zoom_y;

	bool highlighting_range = false;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_zoom(double p_zoom);
	double get_zoom() const;

	void add_zoom_from_mouse_dist(double p_relative);
	double scale_diff(double p_diff);

	void set_highlighting_range(bool p_enable);
	bool is_highlighting_range() const;

	void set_value_no_step(double p_value, bool p_rounded = false);
	void set_zoom_from_value(double p_value);

	EditorRangeDial();
};

class EditorRangeDialPopup : public PopupPanel {
	GDCLASS(EditorRangeDialPopup, PopupPanel);

	EditorRangeDial *dial;

protected:
	virtual void _pre_popup() override;

	static void _bind_methods();

public:
	EditorRangeDial *get_dial() const;

	EditorRangeDialPopup();
};

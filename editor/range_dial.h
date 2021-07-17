/*************************************************************************/
/*  range_dial.h                                                         */
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

#ifndef RANGE_DIAL_H
#define RANGE_DIAL_H

#include "scene/gui/popup.h"
#include "scene/gui/range.h"

class RangeDial : public Range {
	GDCLASS(RangeDial, Range);

	double zoom;
	double cached_value;

	void _draw_rulers(double p_level);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_zoom(double p_zoom);
	double get_zoom() const;

	virtual void set_value(double p_val);

	void update_by_relative(double p_relative);
	double get_zoom_relative(double p_relative);
	RangeDial();
};

class RangeDialPopup : public PopupPanel {
	GDCLASS(RangeDialPopup, PopupPanel);

	RangeDial *dial;

protected:
	static void _bind_methods();

public:
	RangeDial *get_dial() const;

	RangeDialPopup();
};

#endif // RANGE_DIAL_H

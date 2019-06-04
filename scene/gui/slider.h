/*************************************************************************/
/*  slider.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef SLIDER_H
#define SLIDER_H

#include "scene/gui/range.h"

class Slider : public Range {

	OBJ_TYPE(Slider, Range);

	struct Grab {
		int pos;
		float uvalue;
		bool active;
	} grab;

	int ticks;
	bool mouse_inside;
	Orientation orientation;
	float custom_step;

protected:
	void _input_event(InputEvent p_event);
	void _notification(int p_what);
	static void _bind_methods();
	bool ticks_on_borders;

public:
	virtual Size2 get_minimum_size() const;

	void set_custom_step(float p_custom_step);
	float get_custom_step() const;

	void set_ticks(int p_count);
	int get_ticks() const;

	void set_ticks_on_borders(bool);
	bool get_ticks_on_borders() const;

	Slider(Orientation p_orientation = VERTICAL);
};

class HSlider : public Slider {

	OBJ_TYPE(HSlider, Slider);

public:
	HSlider() :
			Slider(HORIZONTAL) { set_v_size_flags(0); }
};

class VSlider : public Slider {

	OBJ_TYPE(VSlider, Slider);

public:
	VSlider() :
			Slider(VERTICAL) { set_h_size_flags(0); }
};

#endif // SLIDER_H

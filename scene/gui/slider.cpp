/*************************************************************************/
/*  slider.cpp                                                           */
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
#include "slider.h"
#include "os/keyboard.h"

Size2 Slider::get_minimum_size() const {

	Ref<StyleBox> style = get_stylebox("slider");
	Size2i ms = style->get_minimum_size() + style->get_center_size();
	return ms;
}

void Slider::_gui_input(InputEvent p_event) {

	if (p_event.type == InputEvent::MOUSE_BUTTON) {

		InputEventMouseButton &mb = p_event.mouse_button;
		if (mb.button_index == BUTTON_LEFT) {

			if (mb.pressed) {
				Ref<Texture> grabber = get_icon(mouse_inside || has_focus() ? "grabber_hilite" : "grabber");
				grab.pos = orientation == VERTICAL ? mb.y : mb.x;
				double grab_width = (double)grabber->get_size().width;
				double grab_height = (double)grabber->get_size().height;
				double max = orientation == VERTICAL ? get_size().height - grab_height : get_size().width - grab_width;
				if (orientation == VERTICAL)
					set_as_ratio(1 - (((double)grab.pos - (grab_height / 2.0)) / max));
				else
					set_as_ratio(((double)grab.pos - (grab_width / 2.0)) / max);
				grab.active = true;
				grab.uvalue = get_as_ratio();
			} else {
				grab.active = false;
			}
		} else if (mb.pressed && mb.button_index == BUTTON_WHEEL_UP) {

			set_value(get_value() + get_step());
		} else if (mb.pressed && mb.button_index == BUTTON_WHEEL_DOWN) {
			set_value(get_value() - get_step());
		}

	} else if (p_event.type == InputEvent::MOUSE_MOTION) {

		if (grab.active) {

			Size2i size = get_size();
			Ref<Texture> grabber = get_icon("grabber");
			float motion = (orientation == VERTICAL ? p_event.mouse_motion.y : p_event.mouse_motion.x) - grab.pos;
			if (orientation == VERTICAL)
				motion = -motion;
			float areasize = orientation == VERTICAL ? size.height - grabber->get_size().height : size.width - grabber->get_size().width;
			if (areasize <= 0)
				return;
			float umotion = motion / float(areasize);
			set_as_ratio(grab.uvalue + umotion);
		}
	} else {

		if (p_event.is_action("ui_left") && p_event.is_pressed()) {

			if (orientation != HORIZONTAL)
				return;
			set_value(get_value() - (custom_step >= 0 ? custom_step : get_step()));
			accept_event();
		} else if (p_event.is_action("ui_right") && p_event.is_pressed()) {

			if (orientation != HORIZONTAL)
				return;
			set_value(get_value() + (custom_step >= 0 ? custom_step : get_step()));
			accept_event();
		} else if (p_event.is_action("ui_up") && p_event.is_pressed()) {

			if (orientation != VERTICAL)
				return;

			set_value(get_value() + (custom_step >= 0 ? custom_step : get_step()));
			accept_event();
		} else if (p_event.is_action("ui_down") && p_event.is_pressed()) {

			if (orientation != VERTICAL)
				return;
			set_value(get_value() - (custom_step >= 0 ? custom_step : get_step()));
			accept_event();

		} else if (p_event.type == InputEvent::KEY) {

			const InputEventKey &k = p_event.key;

			if (!k.pressed)
				return;

			switch (k.scancode) {

				case KEY_HOME: {

					set_value(get_min());
					accept_event();
				} break;
				case KEY_END: {

					set_value(get_max());
					accept_event();

				} break;
			};
		}
	}
}

void Slider::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_MOUSE_ENTER: {

			mouse_inside = true;
			update();
		} break;
		case NOTIFICATION_MOUSE_EXIT: {

			mouse_inside = false;
			update();
		} break;
		case NOTIFICATION_DRAW: {
			RID ci = get_canvas_item();
			Size2i size = get_size();
			Ref<StyleBox> style = get_stylebox("slider");
			Ref<StyleBox> focus = get_stylebox("focus");
			Ref<Texture> grabber = get_icon(mouse_inside || has_focus() ? "grabber_hilite" : "grabber");
			Ref<Texture> tick = get_icon("tick");

			if (orientation == VERTICAL) {

				style->draw(ci, Rect2i(Point2i(), Size2i(style->get_minimum_size().width + style->get_center_size().width, size.height)));
				/*
				if (mouse_inside||has_focus())
					focus->draw(ci,Rect2i(Point2i(),Size2i(style->get_minimum_size().width+style->get_center_size().width,size.height)));
				*/
				float areasize = size.height - grabber->get_size().height;
				if (ticks > 1) {
					int tickarea = size.height - tick->get_height();
					for (int i = 0; i < ticks; i++) {
						if (!ticks_on_borders && (i == 0 || i + 1 == ticks)) continue;
						int ofs = i * tickarea / (ticks - 1);
						tick->draw(ci, Point2(0, ofs));
					}
				}
				grabber->draw(ci, Point2i(size.width / 2 - grabber->get_size().width / 2, size.height - get_as_ratio() * areasize - grabber->get_size().height));
			} else {
				style->draw(ci, Rect2i(Point2i(), Size2i(size.width, style->get_minimum_size().height + style->get_center_size().height)));
				/*
				if (mouse_inside||has_focus())
					focus->draw(ci,Rect2i(Point2i(),Size2i(size.width,style->get_minimum_size().height+style->get_center_size().height)));
				*/

				float areasize = size.width - grabber->get_size().width;
				if (ticks > 1) {
					int tickarea = size.width - tick->get_width();
					for (int i = 0; i < ticks; i++) {
						if ((!ticks_on_borders) && ((i == 0) || ((i + 1) == ticks))) continue;
						int ofs = i * tickarea / (ticks - 1);
						tick->draw(ci, Point2(ofs, 0));
					}
				}
				grabber->draw(ci, Point2i(get_as_ratio() * areasize, size.height / 2 - grabber->get_size().height / 2));
			}

		} break;
	}
}

void Slider::set_custom_step(float p_custom_step) {

	custom_step = p_custom_step;
}

float Slider::get_custom_step() const {

	return custom_step;
}

void Slider::set_ticks(int p_count) {

	ticks = p_count;
	update();
}

int Slider::get_ticks() const {

	return ticks;
}

bool Slider::get_ticks_on_borders() const {
	return ticks_on_borders;
}

void Slider::set_ticks_on_borders(bool _tob) {
	ticks_on_borders = _tob;
	update();
}

void Slider::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &Slider::_gui_input);
	ClassDB::bind_method(D_METHOD("set_ticks", "count"), &Slider::set_ticks);
	ClassDB::bind_method(D_METHOD("get_ticks"), &Slider::get_ticks);

	ClassDB::bind_method(D_METHOD("get_ticks_on_borders"), &Slider::get_ticks_on_borders);
	ClassDB::bind_method(D_METHOD("set_ticks_on_borders", "ticks_on_border"), &Slider::set_ticks_on_borders);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "tick_count", PROPERTY_HINT_RANGE, "0,4096,1"), "set_ticks", "get_ticks");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ticks_on_borders"), "set_ticks_on_borders", "get_ticks_on_borders");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "focus_mode", PROPERTY_HINT_ENUM, "None,Click,All"), "set_focus_mode", "get_focus_mode");
}

Slider::Slider(Orientation p_orientation) {
	orientation = p_orientation;
	mouse_inside = false;
	grab.active = false;
	ticks = 0;
	custom_step = -1;
	set_focus_mode(FOCUS_ALL);
}

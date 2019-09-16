/*************************************************************************/
/*  aspect_ratio_container.cpp                                                 */
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

#include "aspect_ratio_container.h"

Size2 AspectRatioContainer::get_minimum_size() const {

	Size2 ms;
	for (int i = 0; i < get_child_count(); i++) {

		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c)
			continue;
		if (c->is_set_as_toplevel())
			continue;
		if (!c->is_visible())
			continue;
		Size2 minsize = c->get_combined_minimum_size();
		ms.width = MAX(ms.width, minsize.width);
		ms.height = MAX(ms.height, minsize.height);
	}

	return ms;
}

void AspectRatioContainer::set_ratio(float p_ratio) {

	ratio = p_ratio;

	queue_sort();
}

float AspectRatioContainer::get_ratio() const {

	return ratio;
}

void AspectRatioContainer::set_stretch_mode(StretchMode p_mode) {

	stretch_mode = p_mode;

	queue_sort();
}

AspectRatioContainer::StretchMode AspectRatioContainer::get_stretch_mode() const {

	return stretch_mode;
}

void AspectRatioContainer::set_alignment_x(float p_alignment_x) {

	alignment_x = CLAMP(p_alignment_x, 0.f, 1.f);

	queue_sort();
}

float AspectRatioContainer::get_alignment_x() const {

	return alignment_x;
}

void AspectRatioContainer::set_alignment_y(float p_alignment_y) {

	alignment_y = CLAMP(p_alignment_y, 0.f, 1.f);

	queue_sort();
}

float AspectRatioContainer::get_alignment_y() const {

	return alignment_y;
}

void AspectRatioContainer::_notification(int p_what) {

	if (p_what == NOTIFICATION_SORT_CHILDREN) {

		Size2 size = get_size();
		for (int i = 0; i < get_child_count(); i++) {

			Control *c = Object::cast_to<Control>(get_child(i));
			if (!c)
				continue;
			if (c->is_set_as_toplevel())
				continue;

			Vector2 offset = Vector2();
			Size2 child_minsize = c->get_combined_minimum_size();
			Size2 child_size = Size2(ratio, 1.0);
			float scale_factor = 1.0;

			switch (stretch_mode) {
				case WIDTH_CONTROLS_HEIGHT: {
					scale_factor = size.x / child_size.x;
				} break;

				case HEIGHT_CONTROLS_WIDTH: {
					scale_factor = size.y / child_size.y;
				} break;

				case FIT: {
					scale_factor = MIN(size.x / child_size.x, size.y / child_size.y);
				} break;

				case COVER: {
					scale_factor = MAX(size.x / child_size.x, size.y / child_size.y);
				} break;
			}

			child_size *= scale_factor;
			child_size.x = MAX(child_size.x, child_minsize.x);
			child_size.y = MAX(child_size.y, child_minsize.y);

			offset = (size - child_size) * Vector2(alignment_x, alignment_y);

			fit_child_in_rect(c, Rect2(offset, child_size));
		}
	}
}

void AspectRatioContainer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_ratio", "ratio"), &AspectRatioContainer::set_ratio);
	ClassDB::bind_method(D_METHOD("get_ratio"), &AspectRatioContainer::get_ratio);
	ClassDB::bind_method(D_METHOD("set_stretch_mode", "stretch_mode"), &AspectRatioContainer::set_stretch_mode);
	ClassDB::bind_method(D_METHOD("get_stretch_mode"), &AspectRatioContainer::get_stretch_mode);
	ClassDB::bind_method(D_METHOD("set_alignment_x", "alignment_x"), &AspectRatioContainer::set_alignment_x);
	ClassDB::bind_method(D_METHOD("get_alignment_x"), &AspectRatioContainer::get_alignment_x);
	ClassDB::bind_method(D_METHOD("set_alignment_y", "alignment_y"), &AspectRatioContainer::set_alignment_y);
	ClassDB::bind_method(D_METHOD("get_alignment_y"), &AspectRatioContainer::get_alignment_y);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ratio"), "set_ratio", "get_ratio");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "stretch_mode", PROPERTY_HINT_ENUM, "Width control height,Height controls width,Fit,Fill"), "set_stretch_mode", "get_stretch_mode");

	ADD_GROUP("Alignment", "");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alignment_x", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_alignment_x", "get_alignment_x");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alignment_y", PROPERTY_HINT_RANGE, "0,1, 0.001"), "set_alignment_y", "get_alignment_y");

	BIND_ENUM_CONSTANT(WIDTH_CONTROLS_HEIGHT);
	BIND_ENUM_CONSTANT(HEIGHT_CONTROLS_WIDTH);
	BIND_ENUM_CONSTANT(FIT);
	BIND_ENUM_CONSTANT(COVER);
}

AspectRatioContainer::AspectRatioContainer() {

	ratio = 1.0;
	stretch_mode = FIT;
	alignment_x = 0.5f;
	alignment_y = 0.5f;
}

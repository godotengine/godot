/**************************************************************************/
/*  transform_container.cpp                                               */
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

#include "transform_container.h"

void TransformContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_SORT_CHILDREN: {
			const Rect2 rect(Vector2(), get_size());

			for (int i = 0; i < get_child_count(); i++) {
				Control *c = as_sortable_control(get_child(i));
				if (!c) {
					continue;
				}
				fit_child_in_rect(c, rect);

				Vector2 absolute_transform_offset = transform_offset;
				Vector2 absolute_transform_pivot = transform_pivot;

				if (transform_offset_relative_to_size) {
					absolute_transform_offset *= c->get_size();
				}

				if (transform_pivot_relative_to_size) {
					absolute_transform_pivot *= c->get_size();
				}

				c->set_position(c->get_position() + absolute_transform_offset);
				c->set_pivot_offset(absolute_transform_pivot);
				c->set_rotation(transform_rotation);
				c->set_scale(transform_scale);
			}
		}
	}
}

void TransformContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_transform_offset", "offset"), &TransformContainer::set_transform_offset);
	ClassDB::bind_method(D_METHOD("get_transform_offset"), &TransformContainer::get_transform_offset);

	ClassDB::bind_method(D_METHOD("set_transform_offset_relative_to_size", "relative"), &TransformContainer::set_transform_offset_relative_to_size);
	ClassDB::bind_method(D_METHOD("is_transform_offset_relative_to_size"), &TransformContainer::is_transform_offset_relative_to_size);

	ClassDB::bind_method(D_METHOD("set_transform_scale", "scale"), &TransformContainer::set_transform_scale);
	ClassDB::bind_method(D_METHOD("get_transform_scale"), &TransformContainer::get_transform_scale);

	ClassDB::bind_method(D_METHOD("set_transform_rotation", "rotation"), &TransformContainer::set_transform_rotation);
	ClassDB::bind_method(D_METHOD("get_transform_rotation"), &TransformContainer::get_transform_rotation);

	ClassDB::bind_method(D_METHOD("set_transform_rotation_degrees", "degrees"), &TransformContainer::set_transform_rotation_degrees);
	ClassDB::bind_method(D_METHOD("get_transform_rotation_degrees"), &TransformContainer::get_transform_rotation_degrees);

	ClassDB::bind_method(D_METHOD("set_transform_pivot", "pivot"), &TransformContainer::set_transform_pivot);
	ClassDB::bind_method(D_METHOD("get_transform_pivot"), &TransformContainer::get_transform_pivot);

	ClassDB::bind_method(D_METHOD("set_transform_pivot_relative_to_size", "relative"), &TransformContainer::set_transform_pivot_relative_to_size);
	ClassDB::bind_method(D_METHOD("is_transform_pivot_relative_to_size"), &TransformContainer::is_transform_pivot_relative_to_size);

	ADD_GROUP("Transform", "transform_");

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "transform_offset"), "set_transform_offset", "get_transform_offset");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "transform_offset_relative_to_size"), "set_transform_offset_relative_to_size", "is_transform_offset_relative_to_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "transform_scale", PROPERTY_HINT_LINK), "set_transform_scale", "get_transform_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "transform_rotation", PROPERTY_HINT_RANGE, "-360,360,0.1,or_less,or_greater,radians_as_degrees"), "set_transform_rotation", "get_transform_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "transform_rotation_degrees", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_transform_rotation_degrees", "get_transform_rotation_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "transform_pivot"), "set_transform_pivot", "get_transform_pivot");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "transform_pivot_relative_to_size"), "set_transform_pivot_relative_to_size", "is_transform_pivot_relative_to_size");
}

Size2 TransformContainer::get_minimum_size() const {
	Size2 ms;
	for (int i = 0; i < get_child_count(); i++) {
		Control *c = as_sortable_control(get_child(i), SortableVisibilityMode::VISIBLE);
		if (c) {
			ms = ms.max(c->get_combined_minimum_size());
		}
	}
	return ms;
}

Vector<int> TransformContainer::get_allowed_size_flags_horizontal() const {
	return {
		SIZE_FILL,
		SIZE_SHRINK_BEGIN,
		SIZE_SHRINK_CENTER,
		SIZE_SHRINK_END,
	};
}

Vector<int> TransformContainer::get_allowed_size_flags_vertical() const {
	return {
		SIZE_FILL,
		SIZE_SHRINK_BEGIN,
		SIZE_SHRINK_CENTER,
		SIZE_SHRINK_END,
	};
}

void TransformContainer::set_transform_offset(const Vector2 &p_offset) {
	if (transform_offset == p_offset) {
		return;
	}
	transform_offset = p_offset;
	queue_sort();
}

void TransformContainer::set_transform_offset_relative_to_size(bool p_relative) {
	if (transform_offset_relative_to_size == p_relative) {
		return;
	}
	transform_offset_relative_to_size = p_relative;
	queue_sort();
}

void TransformContainer::set_transform_scale(const Vector2 &p_scale) {
	if (transform_scale == p_scale) {
		return;
	}
	transform_scale = p_scale;
	queue_sort();
}

void TransformContainer::set_transform_rotation(real_t p_rotation) {
	if (transform_rotation == p_rotation) {
		return;
	}
	transform_rotation = p_rotation;
	queue_sort();
}

void TransformContainer::set_transform_pivot(const Vector2 &p_pivot) {
	if (transform_pivot == p_pivot) {
		return;
	}
	transform_pivot = p_pivot;
	queue_sort();
}

void TransformContainer::set_transform_pivot_relative_to_size(bool p_relative) {
	if (transform_pivot_relative_to_size == p_relative) {
		return;
	}
	transform_pivot_relative_to_size = p_relative;
	queue_sort();
}

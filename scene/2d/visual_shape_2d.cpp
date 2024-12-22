/**************************************************************************/
/*  visual_shape_2d.cpp                                                   */
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

#include "visual_shape_2d.h"

#include "core/math/geometry_2d.h"

#ifdef TOOLS_ENABLED
Dictionary VisualShape2D::_edit_get_state() const {
	Dictionary state = Node2D::_edit_get_state();
	state["size"] = size;
	state["offset"] = offset;
	return state;
}

void VisualShape2D::_edit_set_state(const Dictionary &p_state) {
	Node2D::_edit_set_state(p_state);
	set_size(p_state["size"]);
	set_offset(p_state["offset"]);
}

void VisualShape2D::_edit_set_pivot(const Point2 &p_pivot) {
	set_position(get_transform().xform(p_pivot));
	set_offset(get_offset() - p_pivot);
}

Point2 VisualShape2D::_edit_get_pivot() const {
	return Vector2();
}

bool VisualShape2D::_edit_use_pivot() const {
	return true;
}

void VisualShape2D::_edit_set_rect(const Rect2 &p_edit_rect) {
	Rect2 r = _edit_get_rect();

	Vector2 size_ratio;

	if (r.size.x != 0 && r.size.y != 0) {
		size_ratio = p_edit_rect.size / r.size;
	}

	Point2 new_pos = p_edit_rect.position - r.position * size_ratio;

	Transform2D postxf;
	postxf.set_rotation_scale_and_skew(get_rotation(), get_scale(), get_skew());
	new_pos = postxf.xform(new_pos);
	set_position(get_position() + new_pos);

	Size2 new_size = p_edit_rect.size;
	set_size(new_size.maxf(0.0001));

	Point2 new_offset = offset * size_ratio;
	set_offset(new_offset);
}
#endif // TOOLS_ENABLED

#ifdef DEBUG_ENABLED
bool VisualShape2D::_edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const {
	switch (shape_type) {
		case SHAPE_RECTANGLE: {
			return get_rect().has_point(p_point);
		} break;
		case SHAPE_CIRCLE: {
			Point2 rel_point = (p_point - offset) / (size / 2);
			return rel_point.length_squared() <= 1.0;
		} break;
		case SHAPE_EQUILATERAL_TRIANGLE:
		case SHAPE_RIGHT_TRIANGLE:
		case SHAPE_CAPSULE: {
			return Geometry2D::is_point_in_polygon(p_point, get_points());
		} break;
	}
	return false;
}

Rect2 VisualShape2D::_edit_get_rect() const {
	return get_rect();
}

bool VisualShape2D::_edit_use_rect() const {
	return true;
}
#endif // DEBUG_ENABLED

void VisualShape2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			bool filled = outline_width <= 0;
			switch (shape_type) {
				case SHAPE_RECTANGLE: {
					draw_rect(get_rect(), color, filled, filled ? -1 : outline_width, antialiased);
				} break;
				case SHAPE_CIRCLE: // Don't use draw_circle for UVs.
				case SHAPE_EQUILATERAL_TRIANGLE:
				case SHAPE_RIGHT_TRIANGLE:
				case SHAPE_CAPSULE: {
					if (!filled || antialiased) {
						PackedVector2Array points = get_points();
						points.push_back(points[0]);
						draw_polyline(points, color, filled ? 1 : outline_width, antialiased);
					}
					if (filled) {
						draw_colored_polygon(get_points(), color, get_uvs());
					}
				} break;
			}
		} break;
	}
}

PackedVector2Array VisualShape2D::_get_shape_points() const {
	// In range [-1,1].
	PackedVector2Array points;
	switch (shape_type) {
		case SHAPE_RECTANGLE: {
			points = {
				Vector2(-1, 1),
				Vector2(-1, -1),
				Vector2(1, -1),
				Vector2(1, 1)
			};
		} break;
		case SHAPE_CIRCLE: {
			float angle_delta = Math_TAU / resolution;
			points.resize(resolution);
			Vector2 *points_write = points.ptrw();
			for (int i = 0; i < resolution; i++) {
				// Start at top to orient shapes with odd resolution.
				points_write[i] = Vector2::from_angle(i * angle_delta - Math_PI / 2.0);
			}
		} break;
		case SHAPE_EQUILATERAL_TRIANGLE: {
			points = {
				Vector2(-1, 1),
				Vector2(0, -1),
				Vector2(1, 1)
			};
		} break;
		case SHAPE_RIGHT_TRIANGLE: {
			points = {
				Vector2(-1, 1),
				Vector2(-1, -1),
				Vector2(1, 1)
			};
		} break;
		case SHAPE_CAPSULE: {
			// Also see CapsuleShape2D::_get_points().
			int capsule_res = resolution;
			// Must be even.
			if (resolution % 2 == 1) {
				capsule_res += 1;
			}
			points.resize(capsule_res + 2);
			Vector2 *points_write = points.ptrw();

			int first_half = capsule_res / 2;
			real_t angle_delta = Math_TAU / capsule_res;
			real_t radius = MIN(size.x, size.y);
			real_t height = MAX(size.x, size.y);
			if (radius == height) {
				height += 0.0001;
			}

			Vector2 capsule_offset;
			if (size.y >= size.x) {
				capsule_offset = Vector2(0, 1.0 - radius / height);
			} else {
				capsule_offset = Vector2(1.0 - radius / height, 0);
			}

			int index = 0;
			for (int i = 0; i < capsule_res; i++) {
				Vector2 circle_point;
				if (size.y >= size.x) {
					// Start at right for vertical capsules.
					circle_point = Vector2::from_angle(i * angle_delta) * radius / size;
				} else {
					// Start at top for horizontal capsules.
					circle_point = Vector2::from_angle(i * angle_delta - Math_PI / 2.0) * radius / size;
				}

				if (i == 0) {
					points_write[index++] = circle_point - capsule_offset;
				}
				points_write[index++] = circle_point + capsule_offset;
				if (i == first_half) {
					points_write[index++] = circle_point - capsule_offset;
					capsule_offset *= -1;
				}
			}
		}
	}
	return points;
}

PackedVector2Array VisualShape2D::get_points() const {
	return Transform2D(0, size / 2, 0, offset).xform(_get_shape_points());
}

PackedVector2Array VisualShape2D::get_uvs() const {
	return Transform2D(0, Size2(0.5, 0.5), 0, Point2(0.5, 0.5)).xform(_get_shape_points());
}

Rect2 VisualShape2D::get_rect() const {
	return Rect2(offset - size / 2, size);
}

void VisualShape2D::set_shape_type(ShapeType p_shape_type) {
	if (shape_type == p_shape_type) {
		return;
	}
	if (shape_type == SHAPE_CAPSULE) {
		size.x *= 2;
	}

	shape_type = p_shape_type;
	if (shape_type == SHAPE_CAPSULE) {
		// Capsule default size should not have the same width and height.
		size.x /= 2;
	}
	queue_redraw();
	notify_property_list_changed();
}

VisualShape2D::ShapeType VisualShape2D::get_shape_type() const {
	return shape_type;
}

void VisualShape2D::set_color(const Color &p_color) {
	if (color == p_color) {
		return;
	}
	color = p_color;
	queue_redraw();
}

Color VisualShape2D::get_color() const {
	return color;
}

void VisualShape2D::set_size(const Size2 &p_size) {
	ERR_FAIL_COND_MSG(p_size.x <= 0 || p_size.y <= 0, "Size must be greater than 0.");
	if (size == p_size) {
		return;
	}
	size = p_size;
	queue_redraw();
}

Size2 VisualShape2D::get_size() const {
	return size;
}

void VisualShape2D::set_offset(const Point2 &p_offset) {
	if (offset == p_offset) {
		return;
	}
	offset = p_offset;
	queue_redraw();
}

Point2 VisualShape2D::get_offset() const {
	return offset;
}

void VisualShape2D::set_antialiased(bool p_antialiased) {
	if (antialiased == p_antialiased) {
		return;
	}
	antialiased = p_antialiased;
	queue_redraw();
}

bool VisualShape2D::is_antialiased() const {
	return antialiased;
}

void VisualShape2D::set_outline_width(float p_outline_width) {
	if (outline_width == p_outline_width) {
		return;
	}
	outline_width = p_outline_width;
	queue_redraw();
}

float VisualShape2D::get_outline_width() const {
	return outline_width;
}

void VisualShape2D::set_resolution(int p_resolution) {
	ERR_FAIL_COND_MSG(p_resolution < 3, "Resolution must be at least 3.");
	if (resolution == p_resolution) {
		return;
	}
	resolution = p_resolution;
	queue_redraw();
}

int VisualShape2D::get_resolution() const {
	return resolution;
}

void VisualShape2D::_validate_property(PropertyInfo &p_property) const {
	if (shape_type != SHAPE_CIRCLE && shape_type != SHAPE_CAPSULE && p_property.name == "resolution") {
		// Resolution is only used by circles and capsules.
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

void VisualShape2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shape_type", "shape_type"), &VisualShape2D::set_shape_type);
	ClassDB::bind_method(D_METHOD("get_shape_type"), &VisualShape2D::get_shape_type);
	ClassDB::bind_method(D_METHOD("set_color", "color"), &VisualShape2D::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &VisualShape2D::get_color);
	ClassDB::bind_method(D_METHOD("set_size", "size"), &VisualShape2D::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &VisualShape2D::get_size);
	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &VisualShape2D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &VisualShape2D::get_offset);

	ClassDB::bind_method(D_METHOD("set_resolution", "resolution"), &VisualShape2D::set_resolution);
	ClassDB::bind_method(D_METHOD("get_resolution"), &VisualShape2D::get_resolution);
	ClassDB::bind_method(D_METHOD("set_antialiased", "antialiased"), &VisualShape2D::set_antialiased);
	ClassDB::bind_method(D_METHOD("is_antialiased"), &VisualShape2D::is_antialiased);
	ClassDB::bind_method(D_METHOD("set_outline_width", "outline_width"), &VisualShape2D::set_outline_width);
	ClassDB::bind_method(D_METHOD("get_outline_width"), &VisualShape2D::get_outline_width);

	ClassDB::bind_method(D_METHOD("get_points"), &VisualShape2D::get_points);
	ClassDB::bind_method(D_METHOD("get_uvs"), &VisualShape2D::get_uvs);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "shape_type", PROPERTY_HINT_ENUM, "Rectangle,Circle,Equilateral Triangle,Right Triangle,Capsule"), "set_shape_type", "get_shape_type");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "color"), "set_color", "get_color");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "size", PROPERTY_HINT_RANGE, "0.0001,99999,0.001,or_greater,hide_slider,suffix:px"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "offset", PROPERTY_HINT_NONE, "suffix:px"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "resolution", PROPERTY_HINT_RANGE, "3,1024,1,or_greater"), "set_resolution", "get_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "antialiased"), "set_antialiased", "is_antialiased");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "outline_width", PROPERTY_HINT_RANGE, "0,16,0.01,or_greater"), "set_outline_width", "get_outline_width");

	BIND_ENUM_CONSTANT(SHAPE_RECTANGLE);
	BIND_ENUM_CONSTANT(SHAPE_CIRCLE);
	BIND_ENUM_CONSTANT(SHAPE_EQUILATERAL_TRIANGLE);
	BIND_ENUM_CONSTANT(SHAPE_RIGHT_TRIANGLE);
	BIND_ENUM_CONSTANT(SHAPE_CAPSULE);
}

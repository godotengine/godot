/*************************************************************************/
/*  node_2d.cpp                                                          */
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

#include "node_2d.h"

#include "core/object/message_queue.h"
#include "scene/gui/control.h"
#include "scene/main/window.h"
#include "servers/rendering_server.h"

#ifdef TOOLS_ENABLED
Dictionary Node2D::_edit_get_state() const {
	Dictionary state;
	state["position"] = get_position();
	state["rotation"] = get_rotation();
	state["scale"] = get_scale();
	state["skew"] = get_skew();

	return state;
}

void Node2D::_edit_set_state(const Dictionary &p_state) {
	pos = p_state["position"];
	angle = p_state["rotation"];
	_scale = p_state["scale"];
	skew = p_state["skew"];

	_update_transform();
}

void Node2D::_edit_set_position(const Point2 &p_position) {
	set_position(p_position);
}

Point2 Node2D::_edit_get_position() const {
	return pos;
}

void Node2D::_edit_set_scale(const Size2 &p_scale) {
	set_scale(p_scale);
}

Size2 Node2D::_edit_get_scale() const {
	return _scale;
}

void Node2D::_edit_set_rotation(real_t p_rotation) {
	angle = p_rotation;
	_update_transform();
}

real_t Node2D::_edit_get_rotation() const {
	return angle;
}

bool Node2D::_edit_use_rotation() const {
	return true;
}

void Node2D::_edit_set_rect(const Rect2 &p_edit_rect) {
	ERR_FAIL_COND(!_edit_use_rect());

	Rect2 r = _edit_get_rect();

	Vector2 zero_offset;
	if (r.size.x != 0) {
		zero_offset.x = -r.position.x / r.size.x;
	}
	if (r.size.y != 0) {
		zero_offset.y = -r.position.y / r.size.y;
	}

	Size2 new_scale(1, 1);

	if (r.size.x != 0) {
		new_scale.x = p_edit_rect.size.x / r.size.x;
	}
	if (r.size.y != 0) {
		new_scale.y = p_edit_rect.size.y / r.size.y;
	}

	Point2 new_pos = p_edit_rect.position + p_edit_rect.size * zero_offset;

	Transform2D postxf;
	postxf.set_rotation_scale_and_skew(angle, _scale, skew);
	new_pos = postxf.xform(new_pos);

	pos += new_pos;
	_scale *= new_scale;

	_update_transform();
}
#endif

void Node2D::_update_xform_values() {
	pos = _mat.elements[2];
	angle = _mat.get_rotation();
	_scale = _mat.get_scale();
	skew = _mat.get_skew();
	_xform_dirty = false;
}

void Node2D::_update_transform() {
	_mat.set_rotation_scale_and_skew(angle, _scale, skew);
	_mat.elements[2] = pos;

	RenderingServer::get_singleton()->canvas_item_set_transform(get_canvas_item(), _mat);

	if (!is_inside_tree()) {
		return;
	}

	_notify_transform();
}

void Node2D::set_position(const Point2 &p_pos) {
	if (_xform_dirty) {
		((Node2D *)this)->_update_xform_values();
	}
	pos = p_pos;
	_update_transform();
}

void Node2D::set_rotation(real_t p_radians) {
	if (_xform_dirty) {
		((Node2D *)this)->_update_xform_values();
	}
	angle = p_radians;
	_update_transform();
}

void Node2D::set_skew(real_t p_radians) {
	if (_xform_dirty) {
		((Node2D *)this)->_update_xform_values();
	}
	skew = p_radians;
	_update_transform();
}

void Node2D::set_rotation_degrees(real_t p_degrees) {
	set_rotation(Math::deg2rad(p_degrees));
}

void Node2D::set_skew_degrees(real_t p_degrees) {
	set_skew(Math::deg2rad(p_degrees));
}

void Node2D::set_scale(const Size2 &p_scale) {
	if (_xform_dirty) {
		((Node2D *)this)->_update_xform_values();
	}
	_scale = p_scale;
	// Avoid having 0 scale values, can lead to errors in physics and rendering.
	if (_scale.x == 0) {
		_scale.x = CMP_EPSILON;
	}
	if (_scale.y == 0) {
		_scale.y = CMP_EPSILON;
	}
	_update_transform();
}

Point2 Node2D::get_position() const {
	if (_xform_dirty) {
		((Node2D *)this)->_update_xform_values();
	}
	return pos;
}

real_t Node2D::get_rotation() const {
	if (_xform_dirty) {
		((Node2D *)this)->_update_xform_values();
	}

	return angle;
}

real_t Node2D::get_skew() const {
	if (_xform_dirty) {
		((Node2D *)this)->_update_xform_values();
	}

	return skew;
}

real_t Node2D::get_rotation_degrees() const {
	return Math::rad2deg(get_rotation());
}

real_t Node2D::get_skew_degrees() const {
	return Math::rad2deg(get_skew());
}

Size2 Node2D::get_scale() const {
	if (_xform_dirty) {
		((Node2D *)this)->_update_xform_values();
	}

	return _scale;
}

Transform2D Node2D::get_transform() const {
	return _mat;
}

void Node2D::rotate(real_t p_radians) {
	set_rotation(get_rotation() + p_radians);
}

void Node2D::translate(const Vector2 &p_amount) {
	set_position(get_position() + p_amount);
}

void Node2D::global_translate(const Vector2 &p_amount) {
	set_global_position(get_global_position() + p_amount);
}

void Node2D::apply_scale(const Size2 &p_amount) {
	set_scale(get_scale() * p_amount);
}

void Node2D::move_x(real_t p_delta, bool p_scaled) {
	Transform2D t = get_transform();
	Vector2 m = t[0];
	if (!p_scaled) {
		m.normalize();
	}
	set_position(t[2] + m * p_delta);
}

void Node2D::move_y(real_t p_delta, bool p_scaled) {
	Transform2D t = get_transform();
	Vector2 m = t[1];
	if (!p_scaled) {
		m.normalize();
	}
	set_position(t[2] + m * p_delta);
}

Point2 Node2D::get_global_position() const {
	return get_global_transform().get_origin();
}

void Node2D::set_global_position(const Point2 &p_pos) {
	Transform2D inv;
	CanvasItem *pi = get_parent_item();
	if (pi) {
		inv = pi->get_global_transform().affine_inverse();
		set_position(inv.xform(p_pos));
	} else {
		set_position(p_pos);
	}
}

real_t Node2D::get_global_rotation() const {
	return get_global_transform().get_rotation();
}

void Node2D::set_global_rotation(real_t p_radians) {
	CanvasItem *pi = get_parent_item();
	if (pi) {
		const real_t parent_global_rot = pi->get_global_transform().get_rotation();
		set_rotation(p_radians - parent_global_rot);
	} else {
		set_rotation(p_radians);
	}
}

real_t Node2D::get_global_rotation_degrees() const {
	return Math::rad2deg(get_global_rotation());
}

void Node2D::set_global_rotation_degrees(real_t p_degrees) {
	set_global_rotation(Math::deg2rad(p_degrees));
}

Size2 Node2D::get_global_scale() const {
	return get_global_transform().get_scale();
}

void Node2D::set_global_scale(const Size2 &p_scale) {
	CanvasItem *pi = get_parent_item();
	if (pi) {
		const Size2 parent_global_scale = pi->get_global_transform().get_scale();
		set_scale(p_scale / parent_global_scale);
	} else {
		set_scale(p_scale);
	}
}

void Node2D::set_transform(const Transform2D &p_transform) {
	_mat = p_transform;
	_xform_dirty = true;

	RenderingServer::get_singleton()->canvas_item_set_transform(get_canvas_item(), _mat);

	if (!is_inside_tree()) {
		return;
	}

	_notify_transform();
}

void Node2D::set_global_transform(const Transform2D &p_transform) {
	CanvasItem *pi = get_parent_item();
	if (pi) {
		set_transform(pi->get_global_transform().affine_inverse() * p_transform);
	} else {
		set_transform(p_transform);
	}
}

void Node2D::set_z_index(int p_z) {
	ERR_FAIL_COND(p_z < RS::CANVAS_ITEM_Z_MIN);
	ERR_FAIL_COND(p_z > RS::CANVAS_ITEM_Z_MAX);
	z_index = p_z;
	RS::get_singleton()->canvas_item_set_z_index(get_canvas_item(), z_index);
}

void Node2D::set_z_as_relative(bool p_enabled) {
	if (z_relative == p_enabled) {
		return;
	}
	z_relative = p_enabled;
	RS::get_singleton()->canvas_item_set_z_as_relative_to_parent(get_canvas_item(), p_enabled);
}

bool Node2D::is_z_relative() const {
	return z_relative;
}

int Node2D::get_z_index() const {
	return z_index;
}

Transform2D Node2D::get_relative_transform_to_parent(const Node *p_parent) const {
	if (p_parent == this) {
		return Transform2D();
	}

	Node2D *parent_2d = Object::cast_to<Node2D>(get_parent());

	ERR_FAIL_COND_V(!parent_2d, Transform2D());
	if (p_parent == parent_2d) {
		return get_transform();
	} else {
		return parent_2d->get_relative_transform_to_parent(p_parent) * get_transform();
	}
}

void Node2D::look_at(const Vector2 &p_pos) {
	rotate(get_angle_to(p_pos));
}

real_t Node2D::get_angle_to(const Vector2 &p_pos) const {
	return (to_local(p_pos) * get_scale()).angle();
}

Point2 Node2D::to_local(Point2 p_global) const {
	return get_global_transform().affine_inverse().xform(p_global);
}

Point2 Node2D::to_global(Point2 p_local) const {
	return get_global_transform().xform(p_local);
}

void Node2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_position", "position"), &Node2D::set_position);
	ClassDB::bind_method(D_METHOD("set_rotation", "radians"), &Node2D::set_rotation);
	ClassDB::bind_method(D_METHOD("set_rotation_degrees", "degrees"), &Node2D::set_rotation_degrees);
	ClassDB::bind_method(D_METHOD("set_skew", "radians"), &Node2D::set_skew);
	ClassDB::bind_method(D_METHOD("set_skew_degrees", "degrees"), &Node2D::set_skew_degrees);
	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &Node2D::set_scale);

	ClassDB::bind_method(D_METHOD("get_position"), &Node2D::get_position);
	ClassDB::bind_method(D_METHOD("get_rotation"), &Node2D::get_rotation);
	ClassDB::bind_method(D_METHOD("get_rotation_degrees"), &Node2D::get_rotation_degrees);
	ClassDB::bind_method(D_METHOD("get_skew"), &Node2D::get_skew);
	ClassDB::bind_method(D_METHOD("get_skew_degrees"), &Node2D::get_skew_degrees);
	ClassDB::bind_method(D_METHOD("get_scale"), &Node2D::get_scale);

	ClassDB::bind_method(D_METHOD("rotate", "radians"), &Node2D::rotate);
	ClassDB::bind_method(D_METHOD("move_local_x", "delta", "scaled"), &Node2D::move_x, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("move_local_y", "delta", "scaled"), &Node2D::move_y, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("translate", "offset"), &Node2D::translate);
	ClassDB::bind_method(D_METHOD("global_translate", "offset"), &Node2D::global_translate);
	ClassDB::bind_method(D_METHOD("apply_scale", "ratio"), &Node2D::apply_scale);

	ClassDB::bind_method(D_METHOD("set_global_position", "position"), &Node2D::set_global_position);
	ClassDB::bind_method(D_METHOD("get_global_position"), &Node2D::get_global_position);
	ClassDB::bind_method(D_METHOD("set_global_rotation", "radians"), &Node2D::set_global_rotation);
	ClassDB::bind_method(D_METHOD("get_global_rotation"), &Node2D::get_global_rotation);
	ClassDB::bind_method(D_METHOD("set_global_rotation_degrees", "degrees"), &Node2D::set_global_rotation_degrees);
	ClassDB::bind_method(D_METHOD("get_global_rotation_degrees"), &Node2D::get_global_rotation_degrees);
	ClassDB::bind_method(D_METHOD("set_global_scale", "scale"), &Node2D::set_global_scale);
	ClassDB::bind_method(D_METHOD("get_global_scale"), &Node2D::get_global_scale);

	ClassDB::bind_method(D_METHOD("set_transform", "xform"), &Node2D::set_transform);
	ClassDB::bind_method(D_METHOD("set_global_transform", "xform"), &Node2D::set_global_transform);

	ClassDB::bind_method(D_METHOD("look_at", "point"), &Node2D::look_at);
	ClassDB::bind_method(D_METHOD("get_angle_to", "point"), &Node2D::get_angle_to);

	ClassDB::bind_method(D_METHOD("to_local", "global_point"), &Node2D::to_local);
	ClassDB::bind_method(D_METHOD("to_global", "local_point"), &Node2D::to_global);

	ClassDB::bind_method(D_METHOD("set_z_index", "z_index"), &Node2D::set_z_index);
	ClassDB::bind_method(D_METHOD("get_z_index"), &Node2D::get_z_index);

	ClassDB::bind_method(D_METHOD("set_z_as_relative", "enable"), &Node2D::set_z_as_relative);
	ClassDB::bind_method(D_METHOD("is_z_relative"), &Node2D::is_z_relative);

	ClassDB::bind_method(D_METHOD("get_relative_transform_to_parent", "parent"), &Node2D::get_relative_transform_to_parent);

	ADD_GROUP("Transform", "");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "position"), "set_position", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rotation", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_rotation", "get_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rotation_degrees", PROPERTY_HINT_RANGE, "-360,360,0.1,or_lesser,or_greater", PROPERTY_USAGE_EDITOR), "set_rotation_degrees", "get_rotation_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "scale"), "set_scale", "get_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "skew", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_skew", "get_skew");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "skew_degrees", PROPERTY_HINT_RANGE, "-89.9,89.9,0.1", PROPERTY_USAGE_EDITOR), "set_skew_degrees", "get_skew_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "transform", PROPERTY_HINT_NONE, "", 0), "set_transform", "get_transform");

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "global_position", PROPERTY_HINT_NONE, "", 0), "set_global_position", "get_global_position");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "global_rotation", PROPERTY_HINT_NONE, "", 0), "set_global_rotation", "get_global_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "global_rotation_degrees", PROPERTY_HINT_NONE, "", 0), "set_global_rotation_degrees", "get_global_rotation_degrees");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "global_scale", PROPERTY_HINT_NONE, "", 0), "set_global_scale", "get_global_scale");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "global_transform", PROPERTY_HINT_NONE, "", 0), "set_global_transform", "get_global_transform");

	ADD_GROUP("Z Index", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "z_index", PROPERTY_HINT_RANGE, itos(RS::CANVAS_ITEM_Z_MIN) + "," + itos(RS::CANVAS_ITEM_Z_MAX) + ",1"), "set_z_index", "get_z_index");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "z_as_relative"), "set_z_as_relative", "is_z_relative");
}

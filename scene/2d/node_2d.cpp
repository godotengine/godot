/*************************************************************************/
/*  node_2d.cpp                                                          */
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
#include "node_2d.h"

#include "message_queue.h"
#include "scene/gui/control.h"
#include "scene/main/viewport.h"
#include "servers/visual_server.h"

void Node2D::edit_set_pivot(const Point2 &p_pivot) {
}

Point2 Node2D::edit_get_pivot() const {

	return Point2();
}
bool Node2D::edit_has_pivot() const {

	return false;
}

Variant Node2D::edit_get_state() const {

	Array state;
	state.push_back(get_pos());
	state.push_back(get_rot());
	state.push_back(get_scale());

	return state;
}
void Node2D::edit_set_state(const Variant &p_state) {

	Array state = p_state;
	ERR_FAIL_COND(state.size() != 3);

	pos = state[0];
	angle = state[1];
	_scale = state[2];
	_update_transform();
	_change_notify("transform/rot");
	_change_notify("transform/scale");
	_change_notify("transform/pos");
}

void Node2D::edit_set_rect(const Rect2 &p_edit_rect) {

	Rect2 r = get_item_rect();

	Vector2 zero_offset;
	if (r.size.x != 0)
		zero_offset.x = -r.pos.x / r.size.x;
	if (r.size.y != 0)
		zero_offset.y = -r.pos.y / r.size.y;

	Size2 new_scale(1, 1);

	if (r.size.x != 0)
		new_scale.x = p_edit_rect.size.x / r.size.x;
	if (r.size.y != 0)
		new_scale.y = p_edit_rect.size.y / r.size.y;

	Point2 new_pos = p_edit_rect.pos + p_edit_rect.size * zero_offset; //p_edit_rect.pos - r.pos;

	Matrix32 postxf;
	postxf.set_rotation_and_scale(angle, _scale);
	new_pos = postxf.xform(new_pos);

	pos += new_pos;
	_scale *= new_scale;

	_update_transform();
	_change_notify("transform/scale");
	_change_notify("transform/pos");
}

void Node2D::edit_rotate(float p_rot) {

	angle += p_rot;
	_update_transform();
	_change_notify("transform/rot");
}

void Node2D::_update_xform_values() {

	pos = _mat.elements[2];
	angle = _mat.get_rotation();
	_scale = _mat.get_scale();
	_xform_dirty = false;
}

void Node2D::_update_transform() {

	Matrix32 mat(angle, pos);
	_mat.set_rotation_and_scale(angle, _scale);
	_mat.elements[2] = pos;

	VisualServer::get_singleton()->canvas_item_set_transform(get_canvas_item(), _mat);

	if (!is_inside_tree())
		return;

	_notify_transform();
}

void Node2D::set_pos(const Point2 &p_pos) {

	if (_xform_dirty)
		((Node2D *)this)->_update_xform_values();
	pos = p_pos;
	_update_transform();
	_change_notify("transform/pos");
}

void Node2D::set_rot(float p_radians) {

	if (_xform_dirty)
		((Node2D *)this)->_update_xform_values();
	angle = p_radians;
	_update_transform();
	_change_notify("transform/rot");
}

void Node2D::set_rotd(float p_degrees) {

	set_rot(Math::deg2rad(p_degrees));
}

// Kept for compatibility after rename to set_rotd.
// Could be removed after a couple releases.
void Node2D::_set_rotd(float p_degrees) {

	WARN_PRINT("Deprecated method Node2D._set_rotd(): This method was renamed to set_rotd. Please adapt your code accordingly, as the old method will be obsoleted.");
	set_rotd(p_degrees);
}

void Node2D::set_scale(const Size2 &p_scale) {

	if (_xform_dirty)
		((Node2D *)this)->_update_xform_values();
	_scale = p_scale;
	if (_scale.x == 0)
		_scale.x = CMP_EPSILON;
	if (_scale.y == 0)
		_scale.y = CMP_EPSILON;
	_update_transform();
	_change_notify("transform/scale");
}

Point2 Node2D::get_pos() const {

	if (_xform_dirty)
		((Node2D *)this)->_update_xform_values();
	return pos;
}
float Node2D::get_rot() const {
	if (_xform_dirty)
		((Node2D *)this)->_update_xform_values();

	return angle;
}
float Node2D::get_rotd() const {

	return Math::rad2deg(get_rot());
}
// Kept for compatibility after rename to get_rotd.
// Could be removed after a couple releases.
float Node2D::_get_rotd() const {

	WARN_PRINT("Deprecated method Node2D._get_rotd(): This method was renamed to get_rotd. Please adapt your code accordingly, as the old method will be obsoleted.");
	return get_rotd();
}
Size2 Node2D::get_scale() const {
	if (_xform_dirty)
		((Node2D *)this)->_update_xform_values();

	return _scale;
}

void Node2D::_notification(int p_what) {

	switch (p_what) {
	}
}

Matrix32 Node2D::get_transform() const {

	return _mat;
}

Rect2 Node2D::get_item_rect() const {

	if (get_script_instance()) {
		Variant::CallError err;
		Rect2 r = get_script_instance()->call("_get_item_rect", NULL, 0, err);
		if (err.error == Variant::CallError::CALL_OK)
			return r;
	}
	return Rect2(Point2(-32, -32), Size2(64, 64));
}

void Node2D::rotate(float p_radians) {

	set_rot(get_rot() + p_radians);
}

void Node2D::translate(const Vector2 &p_amount) {

	set_pos(get_pos() + p_amount);
}

void Node2D::global_translate(const Vector2 &p_amount) {

	set_global_pos(get_global_pos() + p_amount);
}

void Node2D::scale(const Size2 &p_amount) {

	set_scale(get_scale() * p_amount);
}

void Node2D::move_x(float p_delta, bool p_scaled) {

	Matrix32 t = get_transform();
	Vector2 m = t[0];
	if (!p_scaled)
		m.normalize();
	set_pos(t[2] + m * p_delta);
}

void Node2D::move_y(float p_delta, bool p_scaled) {

	Matrix32 t = get_transform();
	Vector2 m = t[1];
	if (!p_scaled)
		m.normalize();
	set_pos(t[2] + m * p_delta);
}

Point2 Node2D::get_global_pos() const {

	return get_global_transform().get_origin();
}

void Node2D::set_global_pos(const Point2 &p_pos) {

	Matrix32 inv;
	CanvasItem *pi = get_parent_item();
	if (pi) {
		inv = pi->get_global_transform().affine_inverse();
		set_pos(inv.xform(p_pos));
	} else {
		set_pos(p_pos);
	}
}

float Node2D::get_global_rot() const {

	return get_global_transform().get_rotation();
}

void Node2D::set_global_rot(float p_radians) {

	CanvasItem *pi = get_parent_item();
	if (pi) {
		const float parent_global_rot = pi->get_global_transform().get_rotation();
		set_rot(p_radians - parent_global_rot);
	} else {
		set_rot(p_radians);
	}
}

float Node2D::get_global_rotd() const {

	return Math::rad2deg(get_global_rot());
}

void Node2D::set_global_rotd(float p_degrees) {

	set_global_rot(Math::deg2rad(p_degrees));
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

void Node2D::set_transform(const Matrix32 &p_transform) {

	_mat = p_transform;
	_xform_dirty = true;

	VisualServer::get_singleton()->canvas_item_set_transform(get_canvas_item(), _mat);

	if (!is_inside_tree())
		return;

	_notify_transform();
}

void Node2D::set_global_transform(const Matrix32 &p_transform) {

	CanvasItem *pi = get_parent_item();
	if (pi)
		set_transform(pi->get_global_transform().affine_inverse() * p_transform);
	else
		set_transform(p_transform);
}

void Node2D::set_z(int p_z) {

	ERR_FAIL_COND(p_z < VS::CANVAS_ITEM_Z_MIN);
	ERR_FAIL_COND(p_z > VS::CANVAS_ITEM_Z_MAX);
	z = p_z;
	VS::get_singleton()->canvas_item_set_z(get_canvas_item(), z);
}

void Node2D::set_z_as_relative(bool p_enabled) {

	if (z_relative == p_enabled)
		return;
	z_relative = p_enabled;
	VS::get_singleton()->canvas_item_set_z_as_relative_to_parent(get_canvas_item(), p_enabled);
}

bool Node2D::is_z_relative() const {

	return z_relative;
}

int Node2D::get_z() const {

	return z;
}

Matrix32 Node2D::get_relative_transform_to_parent(const Node *p_parent) const {

	if (p_parent == this)
		return Matrix32();

	Node2D *parent_2d = get_parent()->cast_to<Node2D>();

	ERR_FAIL_COND_V(!parent_2d, Matrix32());
	if (p_parent == parent_2d)
		return get_transform();
	else
		return parent_2d->get_relative_transform_to_parent(p_parent) * get_transform();
}

void Node2D::look_at(const Vector2 &p_pos) {

	rotate(get_angle_to(p_pos));
}

float Node2D::get_angle_to(const Vector2 &p_pos) const {

	return (get_global_transform().affine_inverse().xform(p_pos)).angle();
}

void Node2D::_bind_methods() {

	// TODO: Obsolete those two methods (old name) properly (GH-4397)
	ObjectTypeDB::bind_method(_MD("_get_rotd"), &Node2D::_get_rotd);
	ObjectTypeDB::bind_method(_MD("_set_rotd", "degrees"), &Node2D::_set_rotd);

	ObjectTypeDB::bind_method(_MD("set_pos", "pos"), &Node2D::set_pos);
	ObjectTypeDB::bind_method(_MD("set_rot", "radians"), &Node2D::set_rot);
	ObjectTypeDB::bind_method(_MD("set_rotd", "degrees"), &Node2D::set_rotd);
	ObjectTypeDB::bind_method(_MD("set_scale", "scale"), &Node2D::set_scale);

	ObjectTypeDB::bind_method(_MD("get_pos"), &Node2D::get_pos);
	ObjectTypeDB::bind_method(_MD("get_rot"), &Node2D::get_rot);
	ObjectTypeDB::bind_method(_MD("get_rotd"), &Node2D::get_rotd);
	ObjectTypeDB::bind_method(_MD("get_scale"), &Node2D::get_scale);

	ObjectTypeDB::bind_method(_MD("rotate", "radians"), &Node2D::rotate);
	ObjectTypeDB::bind_method(_MD("move_local_x", "delta", "scaled"), &Node2D::move_x, DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("move_local_y", "delta", "scaled"), &Node2D::move_y, DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("translate", "offset"), &Node2D::translate);
	ObjectTypeDB::bind_method(_MD("global_translate", "offset"), &Node2D::global_translate);
	ObjectTypeDB::bind_method(_MD("scale", "ratio"), &Node2D::scale);

	ObjectTypeDB::bind_method(_MD("set_global_pos", "pos"), &Node2D::set_global_pos);
	ObjectTypeDB::bind_method(_MD("get_global_pos"), &Node2D::get_global_pos);
	ObjectTypeDB::bind_method(_MD("set_global_rot", "radians"), &Node2D::set_global_rot);
	ObjectTypeDB::bind_method(_MD("get_global_rot"), &Node2D::get_global_rot);
	ObjectTypeDB::bind_method(_MD("set_global_rotd", "degrees"), &Node2D::set_global_rotd);
	ObjectTypeDB::bind_method(_MD("get_global_rotd"), &Node2D::get_global_rotd);
	ObjectTypeDB::bind_method(_MD("set_global_scale", "scale"), &Node2D::set_global_scale);
	ObjectTypeDB::bind_method(_MD("get_global_scale"), &Node2D::get_global_scale);

	ObjectTypeDB::bind_method(_MD("set_transform", "xform"), &Node2D::set_transform);
	ObjectTypeDB::bind_method(_MD("set_global_transform", "xform"), &Node2D::set_global_transform);

	ObjectTypeDB::bind_method(_MD("look_at", "point"), &Node2D::look_at);
	ObjectTypeDB::bind_method(_MD("get_angle_to", "point"), &Node2D::get_angle_to);

	ObjectTypeDB::bind_method(_MD("set_z", "z"), &Node2D::set_z);
	ObjectTypeDB::bind_method(_MD("get_z"), &Node2D::get_z);

	ObjectTypeDB::bind_method(_MD("set_z_as_relative", "enable"), &Node2D::set_z_as_relative);
	ObjectTypeDB::bind_method(_MD("is_z_relative"), &Node2D::is_z_relative);

	ObjectTypeDB::bind_method(_MD("edit_set_pivot", "pivot"), &Node2D::edit_set_pivot);

	ObjectTypeDB::bind_method(_MD("get_relative_transform_to_parent", "parent"), &Node2D::get_relative_transform_to_parent);

	ADD_PROPERTYNZ(PropertyInfo(Variant::VECTOR2, "transform/pos"), _SCS("set_pos"), _SCS("get_pos"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::REAL, "transform/rot", PROPERTY_HINT_RANGE, "-1440,1440,0.1"), _SCS("set_rotd"), _SCS("get_rotd"));
	ADD_PROPERTYNO(PropertyInfo(Variant::VECTOR2, "transform/scale"), _SCS("set_scale"), _SCS("get_scale"));
	ADD_PROPERTYNZ(PropertyInfo(Variant::INT, "z/z", PROPERTY_HINT_RANGE, itos(VS::CANVAS_ITEM_Z_MIN) + "," + itos(VS::CANVAS_ITEM_Z_MAX) + ",1"), _SCS("set_z"), _SCS("get_z"));
	ADD_PROPERTYNO(PropertyInfo(Variant::BOOL, "z/relative"), _SCS("set_z_as_relative"), _SCS("is_z_relative"));
}

Node2D::Node2D() {

	angle = 0;
	_scale = Vector2(1, 1);
	_xform_dirty = false;
	z = 0;
	z_relative = true;
}

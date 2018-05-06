/*************************************************************************/
/*  path.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "path.h"

#include "engine.h"
#include "scene/scene_string_names.h"

void Path::_notification(int p_what) {
}

void Path::_curve_changed() {

	if (is_inside_tree() && Engine::get_singleton()->is_editor_hint())
		update_gizmo();
	if (is_inside_tree()) {
		emit_signal("curve_changed");
	}
}

void Path::set_curve(const Ref<Curve3D> &p_curve) {

	if (curve.is_valid()) {
		curve->disconnect("changed", this, "_curve_changed");
	}

	curve = p_curve;

	if (curve.is_valid()) {
		curve->connect("changed", this, "_curve_changed");
	}
	_curve_changed();
}

Ref<Curve3D> Path::get_curve() const {

	return curve;
}

void Path::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_curve", "curve"), &Path::set_curve);
	ClassDB::bind_method(D_METHOD("get_curve"), &Path::get_curve);
	ClassDB::bind_method(D_METHOD("_curve_changed"), &Path::_curve_changed);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve3D"), "set_curve", "get_curve");

	ADD_SIGNAL(MethodInfo("curve_changed"));
}

Path::Path() {

	set_curve(Ref<Curve3D>(memnew(Curve3D))); //create one by default
}

//////////////

static _FORCE_INLINE_ Basis xy_only(const Basis &basis) {

	Vector3 sideways = basis.get_axis(0);
	Vector3 up = basis.get_axis(1);
	Vector3 forward = basis.get_axis(2);

	Vector3 proj = sideways;
	proj.y = 0.0f;

	if (proj.length_squared() < CMP_EPSILON2) {
		float dir = Vector3(0, 1, 0).dot(sideways) < 0.0f ? -1 : 1;

		sideways = Vector3(1, 0, 0) * dir;
		up = Vector3(0, 1, 0) * dir;
	} else {
		sideways = proj;
		sideways.normalize();

		up = forward.cross(sideways).normalized();
	}

	Basis basis1;
	basis1.set(sideways, up, forward);

	return basis1;
}

static _FORCE_INLINE_ Basis y_only(const Basis &basis) {

	Vector3 sideways = basis.get_axis(0);
	Vector3 up = basis.get_axis(1);
	Vector3 forward = basis.get_axis(2);
	float y_dot = Vector3(0, 1, 0).dot(up);

	// Are we pointing up or down?
	if (fabs(y_dot) < CMP_EPSILON) {
		sideways = Vector3(1, 0, 0);
		forward = Vector3(0, 0, 1);
	} else {
		forward.y = 0.0f;
		forward.normalize();

		sideways = Vector3(0, 1, 0).cross(forward).normalized();
	}

	Basis basis1;
	basis1.set_axis(0, sideways);
	basis1.set_axis(2, forward);

	return basis1;
}

void PathFollow::_update_transform() {

	if (!path)
		return;

	Ref<Curve3D> c = path->get_curve();
	if (!c.is_valid())
		return;

	int count = c->get_point_count();
	if (count < 2)
		return;

	if (delta_offset == 0) {
		return;
	}

	float bl = c->get_baked_length();
	float bi = c->get_bake_interval();
	float o = offset;
	float o_next = offset + bi;

	Vector3 forward;

	if (loop) {
		o = Math::fposmod(o, bl);
		o_next = Math::fposmod(o_next, bl);
	} else if (o_next > bl) {
		o = bl - bi;
		o_next = bl;
	}

	Vector3 pos = c->interpolate_baked(o, cubic);
	forward = c->interpolate_baked(o_next, cubic) - pos;

	if (forward.length_squared() < CMP_EPSILON2) {
		forward = Vector3(0, 0, 1);
	} else {
		forward.normalize();
	}

	Transform t = get_transform();

	Vector3 sideways = c->interpolate_baked_up_vector(o, rotation_mode == ROTATION_XYZ).cross(forward).normalized();
	Vector3 up = forward.cross(sideways).normalized();

	if (rotation_mode != ROTATION_NONE) {

		Basis basis;
		basis.set(sideways, up, forward);

		// process constraints, if any
		if (rotation_mode == ROTATION_Y)
			basis = y_only(basis);
		else if (rotation_mode == ROTATION_XY)
			basis = xy_only(basis);

		sideways = basis.get_axis(0);
		up = basis.get_axis(1);

		Vector3 scale = t.basis.get_scale();
		basis.set(basis.get_axis(0) * scale.x, basis.get_axis(1) * scale.y, basis.get_axis(2) * scale.z);

		t.basis = basis;
	}

	t.origin = pos + sideways * h_offset + up * v_offset;

	set_transform(t);
}

void PathFollow::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {

			Node *parent = get_parent();
			if (parent) {
				path = Object::cast_to<Path>(parent);
				if (path) {
					_update_transform();
				}
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {

			path = NULL;
		} break;
	}
}

void PathFollow::set_cubic_interpolation(bool p_enable) {

	cubic = p_enable;
}

bool PathFollow::get_cubic_interpolation() const {

	return cubic;
}

void PathFollow::_validate_property(PropertyInfo &property) const {

	if (property.name == "offset") {

		float max = 10000;
		if (path && path->get_curve().is_valid())
			max = path->get_curve()->get_baked_length();

		property.hint_string = "0," + rtos(max) + ",0.01";
	}
}

void PathFollow::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &PathFollow::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &PathFollow::get_offset);

	ClassDB::bind_method(D_METHOD("set_h_offset", "h_offset"), &PathFollow::set_h_offset);
	ClassDB::bind_method(D_METHOD("get_h_offset"), &PathFollow::get_h_offset);

	ClassDB::bind_method(D_METHOD("set_v_offset", "v_offset"), &PathFollow::set_v_offset);
	ClassDB::bind_method(D_METHOD("get_v_offset"), &PathFollow::get_v_offset);

	ClassDB::bind_method(D_METHOD("set_unit_offset", "unit_offset"), &PathFollow::set_unit_offset);
	ClassDB::bind_method(D_METHOD("get_unit_offset"), &PathFollow::get_unit_offset);

	ClassDB::bind_method(D_METHOD("set_rotation_mode", "rotation_mode"), &PathFollow::set_rotation_mode);
	ClassDB::bind_method(D_METHOD("get_rotation_mode"), &PathFollow::get_rotation_mode);

	ClassDB::bind_method(D_METHOD("set_cubic_interpolation", "enable"), &PathFollow::set_cubic_interpolation);
	ClassDB::bind_method(D_METHOD("get_cubic_interpolation"), &PathFollow::get_cubic_interpolation);

	ClassDB::bind_method(D_METHOD("set_loop", "loop"), &PathFollow::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &PathFollow::has_loop);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "offset", PROPERTY_HINT_RANGE, "0,10000,0.01"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "unit_offset", PROPERTY_HINT_RANGE, "0,1,0.0001", PROPERTY_USAGE_EDITOR), "set_unit_offset", "get_unit_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "h_offset"), "set_h_offset", "get_h_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "v_offset"), "set_v_offset", "get_v_offset");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rotation_mode", PROPERTY_HINT_ENUM, "None,Y,XY,XYZ"), "set_rotation_mode", "get_rotation_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cubic_interp"), "set_cubic_interpolation", "get_cubic_interpolation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");

	BIND_ENUM_CONSTANT(ROTATION_NONE);
	BIND_ENUM_CONSTANT(ROTATION_Y);
	BIND_ENUM_CONSTANT(ROTATION_XY);
	BIND_ENUM_CONSTANT(ROTATION_XYZ);
}

void PathFollow::set_offset(float p_offset) {
	delta_offset = p_offset - offset;
	offset = p_offset;

	if (path)
		_update_transform();
	_change_notify("offset");
	_change_notify("unit_offset");
}

void PathFollow::set_h_offset(float p_h_offset) {

	h_offset = p_h_offset;
	if (path)
		_update_transform();
}

float PathFollow::get_h_offset() const {

	return h_offset;
}

void PathFollow::set_v_offset(float p_v_offset) {

	v_offset = p_v_offset;
	if (path)
		_update_transform();
}

float PathFollow::get_v_offset() const {

	return v_offset;
}

float PathFollow::get_offset() const {

	return offset;
}

void PathFollow::set_unit_offset(float p_unit_offset) {

	if (path && path->get_curve().is_valid() && path->get_curve()->get_baked_length())
		set_offset(p_unit_offset * path->get_curve()->get_baked_length());
}

float PathFollow::get_unit_offset() const {

	if (path && path->get_curve().is_valid() && path->get_curve()->get_baked_length())
		return get_offset() / path->get_curve()->get_baked_length();
	else
		return 0;
}

void PathFollow::set_rotation_mode(RotationMode p_rotation_mode) {

	rotation_mode = p_rotation_mode;
	_update_transform();
}

PathFollow::RotationMode PathFollow::get_rotation_mode() const {

	return rotation_mode;
}

void PathFollow::set_loop(bool p_loop) {

	loop = p_loop;
}

bool PathFollow::has_loop() const {

	return loop;
}

PathFollow::PathFollow() {

	offset = 0;
	delta_offset = 0;
	h_offset = 0;
	v_offset = 0;
	path = NULL;
	rotation_mode = ROTATION_XYZ;
	cubic = true;
	loop = true;
}

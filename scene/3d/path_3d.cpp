/*************************************************************************/
/*  path_3d.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "path_3d.h"

void Path3D::_notification(int p_what) {
}

void Path3D::_curve_changed() {
	if (is_inside_tree() && Engine::get_singleton()->is_editor_hint()) {
		update_gizmos();
	}
	if (is_inside_tree()) {
		emit_signal(SNAME("curve_changed"));
	}

	// update the configuration warnings of all children of type PathFollow
	// previously used for PathFollowOriented (now enforced orientation is done in PathFollow)
	if (is_inside_tree()) {
		for (int i = 0; i < get_child_count(); i++) {
			PathFollow3D *child = Object::cast_to<PathFollow3D>(get_child(i));
			if (child) {
				child->update_configuration_warnings();
			}
		}
	}
}

void Path3D::set_curve(const Ref<Curve3D> &p_curve) {
	if (curve.is_valid()) {
		curve->disconnect("changed", callable_mp(this, &Path3D::_curve_changed));
	}

	curve = p_curve;

	if (curve.is_valid()) {
		curve->connect("changed", callable_mp(this, &Path3D::_curve_changed));
	}
	_curve_changed();
}

Ref<Curve3D> Path3D::get_curve() const {
	return curve;
}

void Path3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_curve", "curve"), &Path3D::set_curve);
	ClassDB::bind_method(D_METHOD("get_curve"), &Path3D::get_curve);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve3D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT), "set_curve", "get_curve");

	ADD_SIGNAL(MethodInfo("curve_changed"));
}

//////////////

void PathFollow3D::_update_transform(bool p_update_xyz_rot) {
	if (!path) {
		return;
	}

	Ref<Curve3D> c = path->get_curve();
	if (!c.is_valid()) {
		return;
	}

	real_t bl = c->get_baked_length();
	if (bl == 0.0) {
		return;
	}
	real_t bi = c->get_bake_interval();
	real_t o_next = offset + bi;
	real_t o_prev = offset - bi;

	if (loop) {
		o_next = Math::fposmod(o_next, bl);
		o_prev = Math::fposmod(o_prev, bl);
	} else if (rotation_mode == ROTATION_ORIENTED) {
		if (o_next >= bl) {
			o_next = bl;
		}
		if (o_prev <= 0) {
			o_prev = 0;
		}
	}

	Vector3 pos = c->interpolate_baked(offset, cubic);
	Transform3D t = get_transform();
	// Vector3 pos_offset = Vector3(h_offset, v_offset, 0); not used in all cases
	// will be replaced by "Vector3(h_offset, v_offset, 0)" where it was formerly used

	if (rotation_mode == ROTATION_ORIENTED) {
		Vector3 forward = c->interpolate_baked(o_next, cubic) - pos;

		// Try with the previous position
		if (forward.length_squared() < CMP_EPSILON2) {
			forward = pos - c->interpolate_baked(o_prev, cubic);
		}

		if (forward.length_squared() < CMP_EPSILON2) {
			forward = Vector3(0, 0, 1);
		} else {
			forward.normalize();
		}

		Vector3 up = c->interpolate_baked_up_vector(offset, true);

		if (o_next < offset) {
			Vector3 up1 = c->interpolate_baked_up_vector(o_next, true);
			Vector3 axis = up.cross(up1);

			if (axis.length_squared() < CMP_EPSILON2) {
				axis = forward;
			} else {
				axis.normalize();
			}

			up.rotate(axis, up.angle_to(up1) * 0.5f);
		}

		Vector3 scale = t.basis.get_scale();
		Vector3 sideways = up.cross(forward).normalized();
		up = forward.cross(sideways).normalized();

		t.basis.set(sideways, up, forward);
		t.basis.scale_local(scale);

		t.origin = pos + sideways * h_offset + up * v_offset;
	} else if (rotation_mode != ROTATION_NONE) {
		// perform parallel transport
		//
		// see C. Dougan, The Parallel Transport Frame, Game Programming Gems 2 for example
		// for a discussion about why not Frenet frame.

		t.origin = pos;

		if (p_update_xyz_rot && delta_offset != 0) { // Only update rotation if some parameter has changed - i.e. not on addition to scene tree.
			Vector3 t_prev = (pos - c->interpolate_baked(offset - delta_offset, cubic)).normalized();
			Vector3 t_cur = (c->interpolate_baked(offset + delta_offset, cubic) - pos).normalized();

			Vector3 axis = t_prev.cross(t_cur);
			real_t dot = t_prev.dot(t_cur);
			real_t angle = Math::acos(CLAMP(dot, -1, 1));

			if (likely(!Math::is_zero_approx(angle))) {
				if (rotation_mode == ROTATION_Y) {
					// assuming we're referring to global Y-axis. is this correct?
					axis.x = 0;
					axis.z = 0;
				} else if (rotation_mode == ROTATION_XY) {
					axis.z = 0;
				} else if (rotation_mode == ROTATION_XYZ) {
					// all components are allowed
				}

				if (likely(!Math::is_zero_approx(axis.length()))) {
					t.rotate_basis(axis.normalized(), angle);
				}
			}

			// do the additional tilting
			real_t tilt_angle = c->interpolate_baked_tilt(offset);
			Vector3 tilt_axis = t_cur; // not sure what tilt is supposed to do, is this correct??

			if (likely(!Math::is_zero_approx(Math::abs(tilt_angle)))) {
				if (rotation_mode == ROTATION_Y) {
					tilt_axis.x = 0;
					tilt_axis.z = 0;
				} else if (rotation_mode == ROTATION_XY) {
					tilt_axis.z = 0;
				} else if (rotation_mode == ROTATION_XYZ) {
					// all components are allowed
				}

				if (likely(!Math::is_zero_approx(tilt_axis.length()))) {
					t.rotate_basis(tilt_axis.normalized(), tilt_angle);
				}
			}
		}

		t.translate(Vector3(h_offset, v_offset, 0));
	} else {
		t.origin = pos + Vector3(h_offset, v_offset, 0);
	}

	set_transform(t);
}

void PathFollow3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			Node *parent = get_parent();
			if (parent) {
				path = Object::cast_to<Path3D>(parent);
				if (path) {
					_update_transform(false);
				}
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			path = nullptr;
		} break;
	}
}

void PathFollow3D::set_cubic_interpolation(bool p_enable) {
	cubic = p_enable;
}

bool PathFollow3D::get_cubic_interpolation() const {
	return cubic;
}

void PathFollow3D::_validate_property(PropertyInfo &property) const {
	if (property.name == "offset") {
		real_t max = 10000;
		if (path && path->get_curve().is_valid()) {
			max = path->get_curve()->get_baked_length();
		}

		property.hint_string = "0," + rtos(max) + ",0.01,or_lesser,or_greater";
	}
	Node3D::_validate_property(property);
}

TypedArray<String> PathFollow3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (is_visible_in_tree() && is_inside_tree()) {
		if (!Object::cast_to<Path3D>(get_parent())) {
			warnings.push_back(TTR("PathFollow3D only works when set as a child of a Path3D node."));
		} else {
			Path3D *path = Object::cast_to<Path3D>(get_parent());
			if (path->get_curve().is_valid() && !path->get_curve()->is_up_vector_enabled() && rotation_mode == ROTATION_ORIENTED) {
				warnings.push_back(TTR("PathFollow3D's ROTATION_ORIENTED requires \"Up Vector\" to be enabled in its parent Path3D's Curve resource."));
			}
		}
	}

	return warnings;
}

void PathFollow3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_offset", "offset"), &PathFollow3D::set_offset);
	ClassDB::bind_method(D_METHOD("get_offset"), &PathFollow3D::get_offset);

	ClassDB::bind_method(D_METHOD("set_h_offset", "h_offset"), &PathFollow3D::set_h_offset);
	ClassDB::bind_method(D_METHOD("get_h_offset"), &PathFollow3D::get_h_offset);

	ClassDB::bind_method(D_METHOD("set_v_offset", "v_offset"), &PathFollow3D::set_v_offset);
	ClassDB::bind_method(D_METHOD("get_v_offset"), &PathFollow3D::get_v_offset);

	ClassDB::bind_method(D_METHOD("set_unit_offset", "unit_offset"), &PathFollow3D::set_unit_offset);
	ClassDB::bind_method(D_METHOD("get_unit_offset"), &PathFollow3D::get_unit_offset);

	ClassDB::bind_method(D_METHOD("set_rotation_mode", "rotation_mode"), &PathFollow3D::set_rotation_mode);
	ClassDB::bind_method(D_METHOD("get_rotation_mode"), &PathFollow3D::get_rotation_mode);

	ClassDB::bind_method(D_METHOD("set_cubic_interpolation", "enable"), &PathFollow3D::set_cubic_interpolation);
	ClassDB::bind_method(D_METHOD("get_cubic_interpolation"), &PathFollow3D::get_cubic_interpolation);

	ClassDB::bind_method(D_METHOD("set_loop", "loop"), &PathFollow3D::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &PathFollow3D::has_loop);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "offset", PROPERTY_HINT_RANGE, "0,10000,0.01,or_lesser,or_greater"), "set_offset", "get_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "unit_offset", PROPERTY_HINT_RANGE, "0,1,0.0001,or_lesser,or_greater", PROPERTY_USAGE_EDITOR), "set_unit_offset", "get_unit_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "h_offset"), "set_h_offset", "get_h_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "v_offset"), "set_v_offset", "get_v_offset");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rotation_mode", PROPERTY_HINT_ENUM, "None,Y,XY,XYZ,Oriented"), "set_rotation_mode", "get_rotation_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "cubic_interp"), "set_cubic_interpolation", "get_cubic_interpolation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");

	BIND_ENUM_CONSTANT(ROTATION_NONE);
	BIND_ENUM_CONSTANT(ROTATION_Y);
	BIND_ENUM_CONSTANT(ROTATION_XY);
	BIND_ENUM_CONSTANT(ROTATION_XYZ);
	BIND_ENUM_CONSTANT(ROTATION_ORIENTED);
}

void PathFollow3D::set_offset(real_t p_offset) {
	delta_offset = p_offset - offset;
	offset = p_offset;

	if (path) {
		if (path->get_curve().is_valid()) {
			real_t path_length = path->get_curve()->get_baked_length();

			if (loop) {
				offset = Math::fposmod(offset, path_length);
				if (!Math::is_zero_approx(p_offset) && Math::is_zero_approx(offset)) {
					offset = path_length;
				}
			} else {
				offset = CLAMP(offset, 0, path_length);
			}
		}

		_update_transform();
	}
}

void PathFollow3D::set_h_offset(real_t p_h_offset) {
	h_offset = p_h_offset;
	if (path) {
		_update_transform();
	}
}

real_t PathFollow3D::get_h_offset() const {
	return h_offset;
}

void PathFollow3D::set_v_offset(real_t p_v_offset) {
	v_offset = p_v_offset;
	if (path) {
		_update_transform();
	}
}

real_t PathFollow3D::get_v_offset() const {
	return v_offset;
}

real_t PathFollow3D::get_offset() const {
	return offset;
}

void PathFollow3D::set_unit_offset(real_t p_unit_offset) {
	if (path && path->get_curve().is_valid() && path->get_curve()->get_baked_length()) {
		set_offset(p_unit_offset * path->get_curve()->get_baked_length());
	}
}

real_t PathFollow3D::get_unit_offset() const {
	if (path && path->get_curve().is_valid() && path->get_curve()->get_baked_length()) {
		return get_offset() / path->get_curve()->get_baked_length();
	} else {
		return 0;
	}
}

void PathFollow3D::set_rotation_mode(RotationMode p_rotation_mode) {
	rotation_mode = p_rotation_mode;

	update_configuration_warnings();
	_update_transform();
}

PathFollow3D::RotationMode PathFollow3D::get_rotation_mode() const {
	return rotation_mode;
}

void PathFollow3D::set_loop(bool p_loop) {
	loop = p_loop;
}

bool PathFollow3D::has_loop() const {
	return loop;
}

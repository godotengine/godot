/*************************************************************************/
/*  godot_area_2d.cpp                                                    */
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

#include "godot_area_2d.h"
#include "godot_body_2d.h"
#include "godot_space_2d.h"

GodotArea2D::BodyKey::BodyKey(GodotBody2D *p_body, uint32_t p_body_shape, uint32_t p_area_shape) {
	rid = p_body->get_self();
	instance_id = p_body->get_instance_id();
	body_shape = p_body_shape;
	area_shape = p_area_shape;
}

GodotArea2D::BodyKey::BodyKey(GodotArea2D *p_body, uint32_t p_body_shape, uint32_t p_area_shape) {
	rid = p_body->get_self();
	instance_id = p_body->get_instance_id();
	body_shape = p_body_shape;
	area_shape = p_area_shape;
}

void GodotArea2D::_shapes_changed() {
	if (!moved_list.in_list() && get_space()) {
		get_space()->area_add_to_moved_list(&moved_list);
	}
}

void GodotArea2D::set_transform(const Transform2D &p_transform) {
	if (!moved_list.in_list() && get_space()) {
		get_space()->area_add_to_moved_list(&moved_list);
	}

	_set_transform(p_transform);
	_set_inv_transform(p_transform.affine_inverse());
}

void GodotArea2D::set_space(GodotSpace2D *p_space) {
	if (get_space()) {
		if (monitor_query_list.in_list()) {
			get_space()->area_remove_from_monitor_query_list(&monitor_query_list);
		}
		if (moved_list.in_list()) {
			get_space()->area_remove_from_moved_list(&moved_list);
		}
	}

	monitored_bodies.clear();
	monitored_areas.clear();

	_set_space(p_space);
}

void GodotArea2D::set_monitor_callback(ObjectID p_id, const StringName &p_method) {
	if (p_id == monitor_callback_id) {
		monitor_callback_method = p_method;
		return;
	}

	_unregister_shapes();

	monitor_callback_id = p_id;
	monitor_callback_method = p_method;

	monitored_bodies.clear();
	monitored_areas.clear();

	_shape_changed();

	if (!moved_list.in_list() && get_space()) {
		get_space()->area_add_to_moved_list(&moved_list);
	}
}

void GodotArea2D::set_area_monitor_callback(ObjectID p_id, const StringName &p_method) {
	if (p_id == area_monitor_callback_id) {
		area_monitor_callback_method = p_method;
		return;
	}

	_unregister_shapes();

	area_monitor_callback_id = p_id;
	area_monitor_callback_method = p_method;

	monitored_bodies.clear();
	monitored_areas.clear();

	_shape_changed();

	if (!moved_list.in_list() && get_space()) {
		get_space()->area_add_to_moved_list(&moved_list);
	}
}

void GodotArea2D::set_space_override_mode(PhysicsServer2D::AreaSpaceOverrideMode p_mode) {
	bool do_override = p_mode != PhysicsServer2D::AREA_SPACE_OVERRIDE_DISABLED;
	if (do_override == (space_override_mode != PhysicsServer2D::AREA_SPACE_OVERRIDE_DISABLED)) {
		return;
	}
	_unregister_shapes();
	space_override_mode = p_mode;
	_shape_changed();
}

void GodotArea2D::set_param(PhysicsServer2D::AreaParameter p_param, const Variant &p_value) {
	switch (p_param) {
		case PhysicsServer2D::AREA_PARAM_GRAVITY:
			gravity = p_value;
			break;
		case PhysicsServer2D::AREA_PARAM_GRAVITY_VECTOR:
			gravity_vector = p_value;
			break;
		case PhysicsServer2D::AREA_PARAM_GRAVITY_IS_POINT:
			gravity_is_point = p_value;
			break;
		case PhysicsServer2D::AREA_PARAM_GRAVITY_DISTANCE_SCALE:
			gravity_distance_scale = p_value;
			break;
		case PhysicsServer2D::AREA_PARAM_GRAVITY_POINT_ATTENUATION:
			point_attenuation = p_value;
			break;
		case PhysicsServer2D::AREA_PARAM_LINEAR_DAMP:
			linear_damp = p_value;
			break;
		case PhysicsServer2D::AREA_PARAM_ANGULAR_DAMP:
			angular_damp = p_value;
			break;
		case PhysicsServer2D::AREA_PARAM_PRIORITY:
			priority = p_value;
			break;
	}
}

Variant GodotArea2D::get_param(PhysicsServer2D::AreaParameter p_param) const {
	switch (p_param) {
		case PhysicsServer2D::AREA_PARAM_GRAVITY:
			return gravity;
		case PhysicsServer2D::AREA_PARAM_GRAVITY_VECTOR:
			return gravity_vector;
		case PhysicsServer2D::AREA_PARAM_GRAVITY_IS_POINT:
			return gravity_is_point;
		case PhysicsServer2D::AREA_PARAM_GRAVITY_DISTANCE_SCALE:
			return gravity_distance_scale;
		case PhysicsServer2D::AREA_PARAM_GRAVITY_POINT_ATTENUATION:
			return point_attenuation;
		case PhysicsServer2D::AREA_PARAM_LINEAR_DAMP:
			return linear_damp;
		case PhysicsServer2D::AREA_PARAM_ANGULAR_DAMP:
			return angular_damp;
		case PhysicsServer2D::AREA_PARAM_PRIORITY:
			return priority;
	}

	return Variant();
}

void GodotArea2D::_queue_monitor_update() {
	ERR_FAIL_COND(!get_space());

	if (!monitor_query_list.in_list()) {
		get_space()->area_add_to_monitor_query_list(&monitor_query_list);
	}
}

void GodotArea2D::set_monitorable(bool p_monitorable) {
	if (monitorable == p_monitorable) {
		return;
	}

	monitorable = p_monitorable;
	_set_static(!monitorable);
}

void GodotArea2D::call_queries() {
	if (monitor_callback_id.is_valid() && !monitored_bodies.is_empty()) {
		Variant res[5];
		Variant *resptr[5];
		for (int i = 0; i < 5; i++) {
			resptr[i] = &res[i];
		}

		Object *obj = ObjectDB::get_instance(monitor_callback_id);
		if (!obj) {
			monitored_bodies.clear();
			monitor_callback_id = ObjectID();
			return;
		}

		for (Map<BodyKey, BodyState>::Element *E = monitored_bodies.front(); E;) {
			if (E->get().state == 0) { // Nothing happened
				Map<BodyKey, BodyState>::Element *next = E->next();
				monitored_bodies.erase(E);
				E = next;
				continue;
			}

			res[0] = E->get().state > 0 ? PhysicsServer2D::AREA_BODY_ADDED : PhysicsServer2D::AREA_BODY_REMOVED;
			res[1] = E->key().rid;
			res[2] = E->key().instance_id;
			res[3] = E->key().body_shape;
			res[4] = E->key().area_shape;

			Map<BodyKey, BodyState>::Element *next = E->next();
			monitored_bodies.erase(E);
			E = next;

			Callable::CallError ce;
			obj->call(monitor_callback_method, (const Variant **)resptr, 5, ce);
		}
	}

	if (area_monitor_callback_id.is_valid() && !monitored_areas.is_empty()) {
		Variant res[5];
		Variant *resptr[5];
		for (int i = 0; i < 5; i++) {
			resptr[i] = &res[i];
		}

		Object *obj = ObjectDB::get_instance(area_monitor_callback_id);
		if (!obj) {
			monitored_areas.clear();
			area_monitor_callback_id = ObjectID();
			return;
		}

		for (Map<BodyKey, BodyState>::Element *E = monitored_areas.front(); E;) {
			if (E->get().state == 0) { // Nothing happened
				Map<BodyKey, BodyState>::Element *next = E->next();
				monitored_areas.erase(E);
				E = next;
				continue;
			}

			res[0] = E->get().state > 0 ? PhysicsServer2D::AREA_BODY_ADDED : PhysicsServer2D::AREA_BODY_REMOVED;
			res[1] = E->key().rid;
			res[2] = E->key().instance_id;
			res[3] = E->key().body_shape;
			res[4] = E->key().area_shape;

			Map<BodyKey, BodyState>::Element *next = E->next();
			monitored_areas.erase(E);
			E = next;

			Callable::CallError ce;
			obj->call(area_monitor_callback_method, (const Variant **)resptr, 5, ce);
		}
	}
}

void GodotArea2D::compute_gravity(const Vector2 &p_position, Vector2 &r_gravity) const {
	if (is_gravity_point()) {
		const real_t gravity_distance_scale = get_gravity_distance_scale();
		Vector2 v = get_transform().xform(get_gravity_vector()) - p_position;
		if (gravity_distance_scale > 0) {
			const real_t v_length = v.length();
			if (v_length > 0) {
				const real_t v_scaled = v_length * gravity_distance_scale;
				r_gravity = (v.normalized() * (get_gravity() / (v_scaled * v_scaled)));
			} else {
				r_gravity = Vector2();
			}
		} else {
			r_gravity = v.normalized() * get_gravity();
		}
	} else {
		r_gravity = get_gravity_vector() * get_gravity();
	}
}

GodotArea2D::GodotArea2D() :
		GodotCollisionObject2D(TYPE_AREA),
		monitor_query_list(this),
		moved_list(this) {
	_set_static(true); //areas are not active by default
}

GodotArea2D::~GodotArea2D() {
}

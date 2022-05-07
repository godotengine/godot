/*************************************************************************/
/*  area_2d_sw.cpp                                                       */
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

#include "area_2d_sw.h"
#include "body_2d_sw.h"
#include "space_2d_sw.h"

Area2DSW::BodyKey::BodyKey(Body2DSW *p_body, uint32_t p_body_shape, uint32_t p_area_shape) {
	rid = p_body->get_self();
	instance_id = p_body->get_instance_id();
	body_shape = p_body_shape;
	area_shape = p_area_shape;
}
Area2DSW::BodyKey::BodyKey(Area2DSW *p_body, uint32_t p_body_shape, uint32_t p_area_shape) {
	rid = p_body->get_self();
	instance_id = p_body->get_instance_id();
	body_shape = p_body_shape;
	area_shape = p_area_shape;
}

void Area2DSW::_shapes_changed() {
	if (!moved_list.in_list() && get_space()) {
		get_space()->area_add_to_moved_list(&moved_list);
	}
}

void Area2DSW::set_transform(const Transform2D &p_transform) {
	if (!moved_list.in_list() && get_space()) {
		get_space()->area_add_to_moved_list(&moved_list);
	}

	_set_transform(p_transform);
	_set_inv_transform(p_transform.affine_inverse());
}

void Area2DSW::set_space(Space2DSW *p_space) {
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

void Area2DSW::set_monitor_callback(ObjectID p_id, const StringName &p_method) {
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

void Area2DSW::set_area_monitor_callback(ObjectID p_id, const StringName &p_method) {
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

void Area2DSW::set_space_override_mode(Physics2DServer::AreaSpaceOverrideMode p_mode) {
	bool do_override = p_mode != Physics2DServer::AREA_SPACE_OVERRIDE_DISABLED;
	if (do_override == (space_override_mode != Physics2DServer::AREA_SPACE_OVERRIDE_DISABLED)) {
		return;
	}
	_unregister_shapes();
	space_override_mode = p_mode;
	_shape_changed();
}

void Area2DSW::set_param(Physics2DServer::AreaParameter p_param, const Variant &p_value) {
	switch (p_param) {
		case Physics2DServer::AREA_PARAM_GRAVITY:
			gravity = p_value;
			break;
		case Physics2DServer::AREA_PARAM_GRAVITY_VECTOR:
			gravity_vector = p_value;
			break;
		case Physics2DServer::AREA_PARAM_GRAVITY_IS_POINT:
			gravity_is_point = p_value;
			break;
		case Physics2DServer::AREA_PARAM_GRAVITY_DISTANCE_SCALE:
			gravity_distance_scale = p_value;
			break;
		case Physics2DServer::AREA_PARAM_GRAVITY_POINT_ATTENUATION:
			point_attenuation = p_value;
			break;
		case Physics2DServer::AREA_PARAM_LINEAR_DAMP:
			linear_damp = p_value;
			break;
		case Physics2DServer::AREA_PARAM_ANGULAR_DAMP:
			angular_damp = p_value;
			break;
		case Physics2DServer::AREA_PARAM_PRIORITY:
			priority = p_value;
			break;
	}
}

Variant Area2DSW::get_param(Physics2DServer::AreaParameter p_param) const {
	switch (p_param) {
		case Physics2DServer::AREA_PARAM_GRAVITY:
			return gravity;
		case Physics2DServer::AREA_PARAM_GRAVITY_VECTOR:
			return gravity_vector;
		case Physics2DServer::AREA_PARAM_GRAVITY_IS_POINT:
			return gravity_is_point;
		case Physics2DServer::AREA_PARAM_GRAVITY_DISTANCE_SCALE:
			return gravity_distance_scale;
		case Physics2DServer::AREA_PARAM_GRAVITY_POINT_ATTENUATION:
			return point_attenuation;
		case Physics2DServer::AREA_PARAM_LINEAR_DAMP:
			return linear_damp;
		case Physics2DServer::AREA_PARAM_ANGULAR_DAMP:
			return angular_damp;
		case Physics2DServer::AREA_PARAM_PRIORITY:
			return priority;
	}

	return Variant();
}

void Area2DSW::_queue_monitor_update() {
	ERR_FAIL_COND(!get_space());

	if (!monitor_query_list.in_list()) {
		get_space()->area_add_to_monitor_query_list(&monitor_query_list);
	}
}

void Area2DSW::set_monitorable(bool p_monitorable) {
	if (monitorable == p_monitorable) {
		return;
	}

	monitorable = p_monitorable;
	_set_static(!monitorable);
	_shapes_changed();
}

void Area2DSW::call_queries() {
	if (monitor_callback_id && !monitored_bodies.empty()) {
		Object *obj = ObjectDB::get_instance(monitor_callback_id);
		if (obj) {
			Variant res[5];
			Variant *resptr[5];
			for (int i = 0; i < 5; i++) {
				resptr[i] = &res[i];
			}

			for (Map<BodyKey, BodyState>::Element *E = monitored_bodies.front(); E;) {
				if (E->get().state == 0) { // Nothing happened
					Map<BodyKey, BodyState>::Element *next = E->next();
					monitored_bodies.erase(E);
					E = next;
					continue;
				}

				res[0] = E->get().state > 0 ? Physics2DServer::AREA_BODY_ADDED : Physics2DServer::AREA_BODY_REMOVED;
				res[1] = E->key().rid;
				res[2] = E->key().instance_id;
				res[3] = E->key().body_shape;
				res[4] = E->key().area_shape;

				Map<BodyKey, BodyState>::Element *next = E->next();
				monitored_bodies.erase(E);
				E = next;

				Variant::CallError ce;
				obj->call(monitor_callback_method, (const Variant **)resptr, 5, ce);
			}
		} else {
			monitored_bodies.clear();
			monitor_callback_id = 0;
		}
	}

	if (area_monitor_callback_id && !monitored_areas.empty()) {
		Object *obj = ObjectDB::get_instance(area_monitor_callback_id);
		if (obj) {
			Variant res[5];
			Variant *resptr[5];
			for (int i = 0; i < 5; i++) {
				resptr[i] = &res[i];
			}

			for (Map<BodyKey, BodyState>::Element *E = monitored_areas.front(); E;) {
				if (E->get().state == 0) { // Nothing happened
					Map<BodyKey, BodyState>::Element *next = E->next();
					monitored_areas.erase(E);
					E = next;
					continue;
				}

				res[0] = E->get().state > 0 ? Physics2DServer::AREA_BODY_ADDED : Physics2DServer::AREA_BODY_REMOVED;
				res[1] = E->key().rid;
				res[2] = E->key().instance_id;
				res[3] = E->key().body_shape;
				res[4] = E->key().area_shape;

				Map<BodyKey, BodyState>::Element *next = E->next();
				monitored_areas.erase(E);
				E = next;

				Variant::CallError ce;
				obj->call(area_monitor_callback_method, (const Variant **)resptr, 5, ce);
			}
		} else {
			monitored_areas.clear();
			area_monitor_callback_id = 0;
		}
	}
}

Area2DSW::Area2DSW() :
		CollisionObject2DSW(TYPE_AREA),
		monitor_query_list(this),
		moved_list(this) {
	_set_static(true); //areas are not active by default
	space_override_mode = Physics2DServer::AREA_SPACE_OVERRIDE_DISABLED;
	gravity = 9.80665;
	gravity_vector = Vector2(0, -1);
	gravity_is_point = false;
	gravity_distance_scale = 0;
	point_attenuation = 1;

	angular_damp = 1.0;
	linear_damp = 0.1;
	priority = 0;
	monitor_callback_id = 0;
	area_monitor_callback_id = 0;
	monitorable = false;
}

Area2DSW::~Area2DSW() {
}

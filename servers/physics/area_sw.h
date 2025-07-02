/**************************************************************************/
/*  area_sw.h                                                             */
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

#ifndef AREA_SW_H
#define AREA_SW_H

#include "collision_object_sw.h"
#include "core/self_list.h"
#include "servers/physics_server.h"
//#include "servers/physics/query_sw.h"

class SpaceSW;
class BodySW;
class ConstraintSW;

class AreaSW : public CollisionObjectSW {
	PhysicsServer::AreaSpaceOverrideMode space_override_mode;
	real_t gravity;
	Vector3 gravity_vector;
	bool gravity_is_point;
	real_t gravity_distance_scale;
	real_t point_attenuation;
	real_t linear_damp;
	real_t angular_damp;
	int priority;
	bool monitorable;

	ObjectID monitor_callback_id;
	StringName monitor_callback_method;

	ObjectID area_monitor_callback_id;
	StringName area_monitor_callback_method;

	SelfList<AreaSW> monitor_query_list;
	SelfList<AreaSW> moved_list;

	struct BodyKey {
		RID rid;
		ObjectID instance_id;
		uint32_t body_shape;
		uint32_t area_shape;

		_FORCE_INLINE_ bool operator<(const BodyKey &p_key) const {
			if (rid == p_key.rid) {
				if (body_shape == p_key.body_shape) {
					return area_shape < p_key.area_shape;
				} else {
					return body_shape < p_key.body_shape;
				}
			} else {
				return rid < p_key.rid;
			}
		}

		_FORCE_INLINE_ BodyKey() {}
		BodyKey(BodySW *p_body, uint32_t p_body_shape, uint32_t p_area_shape);
		BodyKey(AreaSW *p_body, uint32_t p_body_shape, uint32_t p_area_shape);
	};

	struct BodyState {
		int state;
		_FORCE_INLINE_ void inc() { state++; }
		_FORCE_INLINE_ void dec() { state--; }
		_FORCE_INLINE_ BodyState() { state = 0; }
	};

	Map<BodyKey, BodyState> monitored_bodies;
	Map<BodyKey, BodyState> monitored_areas;

	//virtual void shape_changed_notify(ShapeSW *p_shape);
	//virtual void shape_deleted_notify(ShapeSW *p_shape);

	Set<ConstraintSW *> constraints;

	virtual void _shapes_changed();
	void _queue_monitor_update();

public:
	//_FORCE_INLINE_ const Transform& get_inverse_transform() const { return inverse_transform; }
	//_FORCE_INLINE_ SpaceSW* get_owner() { return owner; }

	void set_monitor_callback(ObjectID p_id, const StringName &p_method);
	_FORCE_INLINE_ bool has_monitor_callback() const { return monitor_callback_id.is_valid(); }

	void set_area_monitor_callback(ObjectID p_id, const StringName &p_method);
	_FORCE_INLINE_ bool has_area_monitor_callback() const { return area_monitor_callback_id.is_valid(); }

	_FORCE_INLINE_ void add_body_to_query(BodySW *p_body, uint32_t p_body_shape, uint32_t p_area_shape);
	_FORCE_INLINE_ void remove_body_from_query(BodySW *p_body, uint32_t p_body_shape, uint32_t p_area_shape);

	_FORCE_INLINE_ void add_area_to_query(AreaSW *p_area, uint32_t p_area_shape, uint32_t p_self_shape);
	_FORCE_INLINE_ void remove_area_from_query(AreaSW *p_area, uint32_t p_area_shape, uint32_t p_self_shape);

	void set_param(PhysicsServer::AreaParameter p_param, const Variant &p_value);
	Variant get_param(PhysicsServer::AreaParameter p_param) const;

	void set_space_override_mode(PhysicsServer::AreaSpaceOverrideMode p_mode);
	PhysicsServer::AreaSpaceOverrideMode get_space_override_mode() const { return space_override_mode; }

	_FORCE_INLINE_ void set_gravity(real_t p_gravity) { gravity = p_gravity; }
	_FORCE_INLINE_ real_t get_gravity() const { return gravity; }

	_FORCE_INLINE_ void set_gravity_vector(const Vector3 &p_gravity) { gravity_vector = p_gravity; }
	_FORCE_INLINE_ Vector3 get_gravity_vector() const { return gravity_vector; }

	_FORCE_INLINE_ void set_gravity_as_point(bool p_enable) { gravity_is_point = p_enable; }
	_FORCE_INLINE_ bool is_gravity_point() const { return gravity_is_point; }

	_FORCE_INLINE_ void set_gravity_distance_scale(real_t scale) { gravity_distance_scale = scale; }
	_FORCE_INLINE_ real_t get_gravity_distance_scale() const { return gravity_distance_scale; }

	_FORCE_INLINE_ void set_point_attenuation(real_t p_point_attenuation) { point_attenuation = p_point_attenuation; }
	_FORCE_INLINE_ real_t get_point_attenuation() const { return point_attenuation; }

	_FORCE_INLINE_ void set_linear_damp(real_t p_linear_damp) { linear_damp = p_linear_damp; }
	_FORCE_INLINE_ real_t get_linear_damp() const { return linear_damp; }

	_FORCE_INLINE_ void set_angular_damp(real_t p_angular_damp) { angular_damp = p_angular_damp; }
	_FORCE_INLINE_ real_t get_angular_damp() const { return angular_damp; }

	_FORCE_INLINE_ void set_priority(int p_priority) { priority = p_priority; }
	_FORCE_INLINE_ int get_priority() const { return priority; }

	_FORCE_INLINE_ void add_constraint(ConstraintSW *p_constraint) { constraints.insert(p_constraint); }
	_FORCE_INLINE_ void remove_constraint(ConstraintSW *p_constraint) { constraints.erase(p_constraint); }
	_FORCE_INLINE_ const Set<ConstraintSW *> &get_constraints() const { return constraints; }
	_FORCE_INLINE_ void clear_constraints() { constraints.clear(); }

	void set_monitorable(bool p_monitorable);
	_FORCE_INLINE_ bool is_monitorable() const { return monitorable; }

	void set_transform(const Transform &p_transform);

	void set_space(SpaceSW *p_space);

	void call_queries();

	AreaSW();
	~AreaSW();
};

void AreaSW::add_body_to_query(BodySW *p_body, uint32_t p_body_shape, uint32_t p_area_shape) {
	BodyKey bk(p_body, p_body_shape, p_area_shape);
	monitored_bodies[bk].inc();
	if (!monitor_query_list.in_list()) {
		_queue_monitor_update();
	}
}
void AreaSW::remove_body_from_query(BodySW *p_body, uint32_t p_body_shape, uint32_t p_area_shape) {
	BodyKey bk(p_body, p_body_shape, p_area_shape);
	monitored_bodies[bk].dec();
	if (!monitor_query_list.in_list()) {
		_queue_monitor_update();
	}
}

void AreaSW::add_area_to_query(AreaSW *p_area, uint32_t p_area_shape, uint32_t p_self_shape) {
	BodyKey bk(p_area, p_area_shape, p_self_shape);
	monitored_areas[bk].inc();
	if (!monitor_query_list.in_list()) {
		_queue_monitor_update();
	}
}
void AreaSW::remove_area_from_query(AreaSW *p_area, uint32_t p_area_shape, uint32_t p_self_shape) {
	BodyKey bk(p_area, p_area_shape, p_self_shape);
	monitored_areas[bk].dec();
	if (!monitor_query_list.in_list()) {
		_queue_monitor_update();
	}
}

#endif // AREA_SW_H

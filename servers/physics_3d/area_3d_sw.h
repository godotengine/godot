/*************************************************************************/
/*  area_3d_sw.h                                                         */
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

#ifndef AREA_SW_H
#define AREA_SW_H

#include "collision_object_3d_sw.h"
#include "core/templates/self_list.h"
#include "servers/physics_server_3d.h"

class Space3DSW;
class Body3DSW;
class SoftBody3DSW;
class Constraint3DSW;

class Area3DSW : public CollisionObject3DSW {
	PhysicsServer3D::AreaSpaceOverrideMode space_override_mode = PhysicsServer3D::AREA_SPACE_OVERRIDE_DISABLED;
	real_t gravity = 9.80665;
	Vector3 gravity_vector = Vector3(0, -1, 0);
	bool gravity_is_point = false;
	real_t gravity_distance_scale = 0.0;
	real_t point_attenuation = 1.0;
	real_t linear_damp = 0.1;
	real_t angular_damp = 0.1;
	real_t wind_force_magnitude = 0.0;
	real_t wind_attenuation_factor = 0.0;
	Vector3 wind_source;
	Vector3 wind_direction;
	int priority = 0;
	bool monitorable = false;

	ObjectID monitor_callback_id;
	StringName monitor_callback_method;

	ObjectID area_monitor_callback_id;
	StringName area_monitor_callback_method;

	SelfList<Area3DSW> monitor_query_list;
	SelfList<Area3DSW> moved_list;

	struct BodyKey {
		RID rid;
		ObjectID instance_id;
		uint32_t body_shape = 0;
		uint32_t area_shape = 0;

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
		BodyKey(SoftBody3DSW *p_body, uint32_t p_body_shape, uint32_t p_area_shape);
		BodyKey(Body3DSW *p_body, uint32_t p_body_shape, uint32_t p_area_shape);
		BodyKey(Area3DSW *p_body, uint32_t p_body_shape, uint32_t p_area_shape);
	};

	struct BodyState {
		int state = 0;
		_FORCE_INLINE_ void inc() { state++; }
		_FORCE_INLINE_ void dec() { state--; }
	};

	Map<BodyKey, BodyState> monitored_soft_bodies;
	Map<BodyKey, BodyState> monitored_bodies;
	Map<BodyKey, BodyState> monitored_areas;

	Set<Constraint3DSW *> constraints;

	virtual void _shapes_changed();
	void _queue_monitor_update();

public:
	void set_monitor_callback(ObjectID p_id, const StringName &p_method);
	_FORCE_INLINE_ bool has_monitor_callback() const { return monitor_callback_id.is_valid(); }

	void set_area_monitor_callback(ObjectID p_id, const StringName &p_method);
	_FORCE_INLINE_ bool has_area_monitor_callback() const { return area_monitor_callback_id.is_valid(); }

	_FORCE_INLINE_ void add_body_to_query(Body3DSW *p_body, uint32_t p_body_shape, uint32_t p_area_shape);
	_FORCE_INLINE_ void remove_body_from_query(Body3DSW *p_body, uint32_t p_body_shape, uint32_t p_area_shape);

	_FORCE_INLINE_ void add_soft_body_to_query(SoftBody3DSW *p_soft_body, uint32_t p_soft_body_shape, uint32_t p_area_shape);
	_FORCE_INLINE_ void remove_soft_body_from_query(SoftBody3DSW *p_soft_body, uint32_t p_soft_body_shape, uint32_t p_area_shape);

	_FORCE_INLINE_ void add_area_to_query(Area3DSW *p_area, uint32_t p_area_shape, uint32_t p_self_shape);
	_FORCE_INLINE_ void remove_area_from_query(Area3DSW *p_area, uint32_t p_area_shape, uint32_t p_self_shape);

	void set_param(PhysicsServer3D::AreaParameter p_param, const Variant &p_value);
	Variant get_param(PhysicsServer3D::AreaParameter p_param) const;

	void set_space_override_mode(PhysicsServer3D::AreaSpaceOverrideMode p_mode);
	PhysicsServer3D::AreaSpaceOverrideMode get_space_override_mode() const { return space_override_mode; }

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

	_FORCE_INLINE_ void set_wind_force_magnitude(real_t p_wind_force_magnitude) { wind_force_magnitude = p_wind_force_magnitude; }
	_FORCE_INLINE_ real_t get_wind_force_magnitude() const { return wind_force_magnitude; }

	_FORCE_INLINE_ void set_wind_attenuation_factor(real_t p_wind_attenuation_factor) { wind_attenuation_factor = p_wind_attenuation_factor; }
	_FORCE_INLINE_ real_t get_wind_attenuation_factor() const { return wind_attenuation_factor; }

	_FORCE_INLINE_ void set_wind_source(const Vector3 &p_wind_source) { wind_source = p_wind_source; }
	_FORCE_INLINE_ const Vector3 &get_wind_source() const { return wind_source; }

	_FORCE_INLINE_ void set_wind_direction(const Vector3 &p_wind_direction) { wind_direction = p_wind_direction; }
	_FORCE_INLINE_ const Vector3 &get_wind_direction() const { return wind_direction; }

	_FORCE_INLINE_ void add_constraint(Constraint3DSW *p_constraint) { constraints.insert(p_constraint); }
	_FORCE_INLINE_ void remove_constraint(Constraint3DSW *p_constraint) { constraints.erase(p_constraint); }
	_FORCE_INLINE_ const Set<Constraint3DSW *> &get_constraints() const { return constraints; }
	_FORCE_INLINE_ void clear_constraints() { constraints.clear(); }

	void set_monitorable(bool p_monitorable);
	_FORCE_INLINE_ bool is_monitorable() const { return monitorable; }

	void set_transform(const Transform3D &p_transform);

	void set_space(Space3DSW *p_space);

	void call_queries();

	void compute_gravity(const Vector3 &p_position, Vector3 &r_gravity) const;

	Area3DSW();
	~Area3DSW();
};

void Area3DSW::add_soft_body_to_query(SoftBody3DSW *p_soft_body, uint32_t p_soft_body_shape, uint32_t p_area_shape) {
	BodyKey bk(p_soft_body, p_soft_body_shape, p_area_shape);
	monitored_soft_bodies[bk].inc();
	if (!monitor_query_list.in_list()) {
		_queue_monitor_update();
	}
}

void Area3DSW::remove_soft_body_from_query(SoftBody3DSW *p_soft_body, uint32_t p_soft_body_shape, uint32_t p_area_shape) {
	BodyKey bk(p_soft_body, p_soft_body_shape, p_area_shape);
	monitored_soft_bodies[bk].dec();
	if (!monitor_query_list.in_list()) {
		_queue_monitor_update();
	}
}

void Area3DSW::add_body_to_query(Body3DSW *p_body, uint32_t p_body_shape, uint32_t p_area_shape) {
	BodyKey bk(p_body, p_body_shape, p_area_shape);
	monitored_bodies[bk].inc();
	if (!monitor_query_list.in_list()) {
		_queue_monitor_update();
	}
}

void Area3DSW::remove_body_from_query(Body3DSW *p_body, uint32_t p_body_shape, uint32_t p_area_shape) {
	BodyKey bk(p_body, p_body_shape, p_area_shape);
	monitored_bodies[bk].dec();
	if (!monitor_query_list.in_list()) {
		_queue_monitor_update();
	}
}

void Area3DSW::add_area_to_query(Area3DSW *p_area, uint32_t p_area_shape, uint32_t p_self_shape) {
	BodyKey bk(p_area, p_area_shape, p_self_shape);
	monitored_areas[bk].inc();
	if (!monitor_query_list.in_list()) {
		_queue_monitor_update();
	}
}

void Area3DSW::remove_area_from_query(Area3DSW *p_area, uint32_t p_area_shape, uint32_t p_self_shape) {
	BodyKey bk(p_area, p_area_shape, p_self_shape);
	monitored_areas[bk].dec();
	if (!monitor_query_list.in_list()) {
		_queue_monitor_update();
	}
}

struct AreaCMP {
	Area3DSW *area = nullptr;
	int refCount = 0;
	_FORCE_INLINE_ bool operator==(const AreaCMP &p_cmp) const { return area->get_self() == p_cmp.area->get_self(); }
	_FORCE_INLINE_ bool operator<(const AreaCMP &p_cmp) const { return area->get_priority() < p_cmp.area->get_priority(); }
	_FORCE_INLINE_ AreaCMP() {}
	_FORCE_INLINE_ AreaCMP(Area3DSW *p_area) {
		area = p_area;
		refCount = 1;
	}
};

#endif // AREA__SW_H

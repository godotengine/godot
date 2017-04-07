/*************************************************************************/
/*  physics_body.h                                                       */
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
#ifndef PHYSICS_BODY__H
#define PHYSICS_BODY__H

#include "scene/3d/collision_object.h"
#include "servers/physics_server.h"
#include "vset.h"

class PhysicsBody : public CollisionObject {

	GDCLASS(PhysicsBody, CollisionObject);

	uint32_t layer_mask;
	uint32_t collision_mask;

	void _set_layers(uint32_t p_mask);
	uint32_t _get_layers() const;

protected:
	static void _bind_methods();
	void _notification(int p_what);
	PhysicsBody(PhysicsServer::BodyMode p_mode);

public:
	virtual Vector3 get_linear_velocity() const;
	virtual Vector3 get_angular_velocity() const;
	virtual float get_inverse_mass() const;

	void set_collision_layer(uint32_t p_mask);
	uint32_t get_collision_layer() const;

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_collision_layer_bit(int p_bit, bool p_value);
	bool get_collision_layer_bit(int p_bit) const;

	void set_collision_mask_bit(int p_bit, bool p_value);
	bool get_collision_mask_bit(int p_bit) const;

	void add_collision_exception_with(Node *p_node); //must be physicsbody
	void remove_collision_exception_with(Node *p_node);

	PhysicsBody();
};

class StaticBody : public PhysicsBody {

	GDCLASS(StaticBody, PhysicsBody);

	Vector3 constant_linear_velocity;
	Vector3 constant_angular_velocity;

	real_t bounce;
	real_t friction;

protected:
	static void _bind_methods();

public:
	void set_friction(real_t p_friction);
	real_t get_friction() const;

	void set_bounce(real_t p_bounce);
	real_t get_bounce() const;

	void set_constant_linear_velocity(const Vector3 &p_vel);
	void set_constant_angular_velocity(const Vector3 &p_vel);

	Vector3 get_constant_linear_velocity() const;
	Vector3 get_constant_angular_velocity() const;

	StaticBody();
	~StaticBody();
};

class RigidBody : public PhysicsBody {

	GDCLASS(RigidBody, PhysicsBody);

public:
	enum Mode {
		MODE_RIGID,
		MODE_STATIC,
		MODE_CHARACTER,
		MODE_KINEMATIC,
	};

	enum AxisLock {
		AXIS_LOCK_DISABLED,
		AXIS_LOCK_X,
		AXIS_LOCK_Y,
		AXIS_LOCK_Z,
	};

private:
	bool can_sleep;
	PhysicsDirectBodyState *state;
	Mode mode;

	real_t bounce;
	real_t mass;
	real_t friction;

	Vector3 linear_velocity;
	Vector3 angular_velocity;
	real_t gravity_scale;
	real_t linear_damp;
	real_t angular_damp;

	bool sleeping;
	bool ccd;

	AxisLock axis_lock;

	int max_contacts_reported;

	bool custom_integrator;

	struct ShapePair {

		int body_shape;
		int local_shape;
		bool tagged;
		bool operator<(const ShapePair &p_sp) const {
			if (body_shape == p_sp.body_shape)
				return local_shape < p_sp.local_shape;
			else
				return body_shape < p_sp.body_shape;
		}

		ShapePair() {}
		ShapePair(int p_bs, int p_ls) {
			body_shape = p_bs;
			local_shape = p_ls;
		}
	};
	struct RigidBody_RemoveAction {

		ObjectID body_id;
		ShapePair pair;
	};
	struct BodyState {

		//int rc;
		bool in_tree;
		VSet<ShapePair> shapes;
	};

	struct ContactMonitor {

		bool locked;
		Map<ObjectID, BodyState> body_map;
	};

	ContactMonitor *contact_monitor;
	void _body_enter_tree(ObjectID p_id);
	void _body_exit_tree(ObjectID p_id);

	void _body_inout(int p_status, ObjectID p_instance, int p_body_shape, int p_local_shape);
	void _direct_state_changed(Object *p_state);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_mode(Mode p_mode);
	Mode get_mode() const;

	void set_mass(real_t p_mass);
	real_t get_mass() const;

	virtual float get_inverse_mass() const { return 1.0 / mass; }

	void set_weight(real_t p_weight);
	real_t get_weight() const;

	void set_friction(real_t p_friction);
	real_t get_friction() const;

	void set_bounce(real_t p_bounce);
	real_t get_bounce() const;

	void set_linear_velocity(const Vector3 &p_velocity);
	Vector3 get_linear_velocity() const;

	void set_axis_velocity(const Vector3 &p_axis);

	void set_angular_velocity(const Vector3 &p_velocity);
	Vector3 get_angular_velocity() const;

	void set_gravity_scale(real_t p_gravity_scale);
	real_t get_gravity_scale() const;

	void set_linear_damp(real_t p_linear_damp);
	real_t get_linear_damp() const;

	void set_angular_damp(real_t p_angular_damp);
	real_t get_angular_damp() const;

	void set_use_custom_integrator(bool p_enable);
	bool is_using_custom_integrator();

	void set_sleeping(bool p_sleeping);
	bool is_sleeping() const;

	void set_can_sleep(bool p_active);
	bool is_able_to_sleep() const;

	void set_contact_monitor(bool p_enabled);
	bool is_contact_monitor_enabled() const;

	void set_max_contacts_reported(int p_amount);
	int get_max_contacts_reported() const;

	void set_use_continuous_collision_detection(bool p_enable);
	bool is_using_continuous_collision_detection() const;

	void set_axis_lock(AxisLock p_lock);
	AxisLock get_axis_lock() const;

	Array get_colliding_bodies() const;

	void apply_impulse(const Vector3 &p_pos, const Vector3 &p_impulse);

	RigidBody();
	~RigidBody();
};

VARIANT_ENUM_CAST(RigidBody::Mode);
VARIANT_ENUM_CAST(RigidBody::AxisLock);

class KinematicBody : public PhysicsBody {

	GDCLASS(KinematicBody, PhysicsBody);

	float margin;
	bool collide_static;
	bool collide_rigid;
	bool collide_kinematic;
	bool collide_character;

	bool colliding;
	Vector3 collision;
	Vector3 normal;
	Vector3 collider_vel;
	ObjectID collider;
	int collider_shape;

	Variant _get_collider() const;

	_FORCE_INLINE_ bool _ignores_mode(PhysicsServer::BodyMode) const;

protected:
	static void _bind_methods();

public:
	enum {
		SLIDE_FLAG_FLOOR,
		SLIDE_FLAG_WALL,
		SLIDE_FLAG_ROOF
	};

	Vector3 move(const Vector3 &p_motion);
	Vector3 move_to(const Vector3 &p_position);

	bool can_teleport_to(const Vector3 &p_position);
	bool is_colliding() const;
	Vector3 get_collision_pos() const;
	Vector3 get_collision_normal() const;
	Vector3 get_collider_velocity() const;
	ObjectID get_collider() const;
	int get_collider_shape() const;

	void set_collide_with_static_bodies(bool p_enable);
	bool can_collide_with_static_bodies() const;

	void set_collide_with_rigid_bodies(bool p_enable);
	bool can_collide_with_rigid_bodies() const;

	void set_collide_with_kinematic_bodies(bool p_enable);
	bool can_collide_with_kinematic_bodies() const;

	void set_collide_with_character_bodies(bool p_enable);
	bool can_collide_with_character_bodies() const;

	void set_collision_margin(float p_margin);
	float get_collision_margin() const;

	KinematicBody();
	~KinematicBody();
};

#endif // PHYSICS_BODY__H

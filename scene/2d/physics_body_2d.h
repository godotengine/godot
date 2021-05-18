/*************************************************************************/
/*  physics_body_2d.h                                                    */
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

#ifndef PHYSICS_BODY_2D_H
#define PHYSICS_BODY_2D_H

#include "core/templates/vset.h"
#include "scene/2d/collision_object_2d.h"
#include "scene/resources/physics_material.h"
#include "servers/physics_server_2d.h"

class KinematicCollision2D;

class PhysicsBody2D : public CollisionObject2D {
	GDCLASS(PhysicsBody2D, CollisionObject2D);

protected:
	void _notification(int p_what);
	PhysicsBody2D(PhysicsServer2D::BodyMode p_mode);

	static void _bind_methods();

public:
	TypedArray<PhysicsBody2D> get_collision_exceptions();
	void add_collision_exception_with(Node *p_node); //must be physicsbody
	void remove_collision_exception_with(Node *p_node);

	PhysicsBody2D();
};

class StaticBody2D : public PhysicsBody2D {
	GDCLASS(StaticBody2D, PhysicsBody2D);

	Vector2 constant_linear_velocity;
	real_t constant_angular_velocity = 0.0;

	Ref<PhysicsMaterial> physics_material_override;

protected:
	static void _bind_methods();

public:
	void set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override);
	Ref<PhysicsMaterial> get_physics_material_override() const;

	void set_constant_linear_velocity(const Vector2 &p_vel);
	void set_constant_angular_velocity(real_t p_vel);

	Vector2 get_constant_linear_velocity() const;
	real_t get_constant_angular_velocity() const;

	StaticBody2D();
	~StaticBody2D();

private:
	void _reload_physics_characteristics();
};

class RigidBody2D : public PhysicsBody2D {
	GDCLASS(RigidBody2D, PhysicsBody2D);

public:
	enum Mode {
		MODE_RIGID,
		MODE_STATIC,
		MODE_CHARACTER,
		MODE_KINEMATIC,
	};

	enum CCDMode {
		CCD_MODE_DISABLED,
		CCD_MODE_CAST_RAY,
		CCD_MODE_CAST_SHAPE,
	};

private:
	bool can_sleep = true;
	PhysicsDirectBodyState2D *state = nullptr;
	Mode mode = MODE_RIGID;

	real_t mass = 1.0;
	Ref<PhysicsMaterial> physics_material_override;
	real_t gravity_scale = 1.0;
	real_t linear_damp = -1.0;
	real_t angular_damp = -1.0;

	Vector2 linear_velocity;
	real_t angular_velocity = 0.0;
	bool sleeping = false;

	int max_contacts_reported = 0;

	bool custom_integrator = false;

	CCDMode ccd_mode = CCD_MODE_DISABLED;

	struct ShapePair {
		int body_shape = 0;
		int local_shape = 0;
		bool tagged = false;
		bool operator<(const ShapePair &p_sp) const {
			if (body_shape == p_sp.body_shape) {
				return local_shape < p_sp.local_shape;
			}

			return body_shape < p_sp.body_shape;
		}

		ShapePair() {}
		ShapePair(int p_bs, int p_ls) {
			body_shape = p_bs;
			local_shape = p_ls;
		}
	};
	struct RigidBody2D_RemoveAction {
		RID rid;
		ObjectID body_id;
		ShapePair pair;
	};
	struct BodyState {
		RID rid;
		//int rc;
		bool in_scene = false;
		VSet<ShapePair> shapes;
	};

	struct ContactMonitor {
		bool locked = false;
		Map<ObjectID, BodyState> body_map;
	};

	ContactMonitor *contact_monitor = nullptr;
	void _body_enter_tree(ObjectID p_id);
	void _body_exit_tree(ObjectID p_id);

	void _body_inout(int p_status, const RID &p_body, ObjectID p_instance, int p_body_shape, int p_local_shape);
	void _direct_state_changed(Object *p_state);

	bool _test_motion(const Vector2 &p_motion, bool p_infinite_inertia = true, real_t p_margin = 0.08, const Ref<PhysicsTestMotionResult2D> &p_result = Ref<PhysicsTestMotionResult2D>());

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_mode(Mode p_mode);
	Mode get_mode() const;

	void set_mass(real_t p_mass);
	real_t get_mass() const;

	void set_inertia(real_t p_inertia);
	real_t get_inertia() const;

	void set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override);
	Ref<PhysicsMaterial> get_physics_material_override() const;

	void set_gravity_scale(real_t p_gravity_scale);
	real_t get_gravity_scale() const;

	void set_linear_damp(real_t p_linear_damp);
	real_t get_linear_damp() const;

	void set_angular_damp(real_t p_angular_damp);
	real_t get_angular_damp() const;

	void set_linear_velocity(const Vector2 &p_velocity);
	Vector2 get_linear_velocity() const;

	void set_axis_velocity(const Vector2 &p_axis);

	void set_angular_velocity(real_t p_velocity);
	real_t get_angular_velocity() const;

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

	void set_continuous_collision_detection_mode(CCDMode p_mode);
	CCDMode get_continuous_collision_detection_mode() const;

	void apply_central_impulse(const Vector2 &p_impulse);
	void apply_impulse(const Vector2 &p_impulse, const Vector2 &p_position = Vector2());
	void apply_torque_impulse(real_t p_torque);

	void set_applied_force(const Vector2 &p_force);
	Vector2 get_applied_force() const;

	void set_applied_torque(const real_t p_torque);
	real_t get_applied_torque() const;

	void add_central_force(const Vector2 &p_force);
	void add_force(const Vector2 &p_force, const Vector2 &p_position = Vector2());
	void add_torque(real_t p_torque);

	TypedArray<Node2D> get_colliding_bodies() const; //function for script

	TypedArray<String> get_configuration_warnings() const override;

	RigidBody2D();
	~RigidBody2D();

private:
	void _reload_physics_characteristics();
};

VARIANT_ENUM_CAST(RigidBody2D::Mode);
VARIANT_ENUM_CAST(RigidBody2D::CCDMode);

class KinematicBody2D : public PhysicsBody2D {
	GDCLASS(KinematicBody2D, PhysicsBody2D);

public:
	struct Collision {
		Vector2 collision;
		Vector2 normal;
		Vector2 collider_vel;
		ObjectID collider;
		RID collider_rid;
		int collider_shape = 0;
		Variant collider_metadata;
		Vector2 remainder;
		Vector2 travel;
		int local_shape = 0;
	};

private:
	real_t margin;

	Vector2 floor_normal;
	Vector2 floor_velocity;
	RID on_floor_body;
	bool on_floor;
	bool on_ceiling;
	bool on_wall;
	bool sync_to_physics;

	Vector<Collision> colliders;
	Vector<Ref<KinematicCollision2D>> slide_colliders;
	Ref<KinematicCollision2D> motion_cache;

	_FORCE_INLINE_ bool _ignores_mode(PhysicsServer2D::BodyMode) const;

	Ref<KinematicCollision2D> _move(const Vector2 &p_motion, bool p_infinite_inertia = true, bool p_exclude_raycast_shapes = true, bool p_test_only = false);
	Ref<KinematicCollision2D> _get_slide_collision(int p_bounce);

	Transform2D last_valid_transform;
	void _direct_state_changed(Object *p_state);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	bool move_and_collide(const Vector2 &p_motion, bool p_infinite_inertia, Collision &r_collision, bool p_exclude_raycast_shapes = true, bool p_test_only = false);

	bool test_move(const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia = true);

	bool separate_raycast_shapes(bool p_infinite_inertia, Collision &r_collision);

	void set_safe_margin(real_t p_margin);
	real_t get_safe_margin() const;

	Vector2 move_and_slide(const Vector2 &p_linear_velocity, const Vector2 &p_up_direction = Vector2(0, 0), bool p_stop_on_slope = false, int p_max_slides = 4, real_t p_floor_max_angle = Math::deg2rad((real_t)45.0), bool p_infinite_inertia = true);
	Vector2 move_and_slide_with_snap(const Vector2 &p_linear_velocity, const Vector2 &p_snap, const Vector2 &p_up_direction = Vector2(0, 0), bool p_stop_on_slope = false, int p_max_slides = 4, real_t p_floor_max_angle = Math::deg2rad((real_t)45.0), bool p_infinite_inertia = true);
	bool is_on_floor() const;
	bool is_on_wall() const;
	bool is_on_ceiling() const;
	Vector2 get_floor_normal() const;
	Vector2 get_floor_velocity() const;

	int get_slide_count() const;
	Collision get_slide_collision(int p_bounce) const;

	void set_sync_to_physics(bool p_enable);
	bool is_sync_to_physics_enabled() const;

	KinematicBody2D();
	~KinematicBody2D();
};

class KinematicCollision2D : public Reference {
	GDCLASS(KinematicCollision2D, Reference);

	KinematicBody2D *owner;
	friend class KinematicBody2D;
	KinematicBody2D::Collision collision;

protected:
	static void _bind_methods();

public:
	Vector2 get_position() const;
	Vector2 get_normal() const;
	Vector2 get_travel() const;
	Vector2 get_remainder() const;
	Object *get_local_shape() const;
	Object *get_collider() const;
	ObjectID get_collider_id() const;
	Object *get_collider_shape() const;
	int get_collider_shape_index() const;
	Vector2 get_collider_velocity() const;
	Variant get_collider_metadata() const;

	KinematicCollision2D();
};

#endif // PHYSICS_BODY_2D_H

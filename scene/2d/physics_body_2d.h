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
	static void _bind_methods();
	PhysicsBody2D(PhysicsServer2D::BodyMode p_mode);

	Ref<KinematicCollision2D> motion_cache;

	Ref<KinematicCollision2D> _move(const Vector2 &p_motion, bool p_infinite_inertia = true, bool p_exclude_raycast_shapes = true, bool p_test_only = false, real_t p_margin = 0.08);

public:
	bool move_and_collide(const Vector2 &p_motion, bool p_infinite_inertia, PhysicsServer2D::MotionResult &r_result, real_t p_margin, bool p_exclude_raycast_shapes = true, bool p_test_only = false, bool p_cancel_sliding = true, const Set<RID> &p_exclude = Set<RID>());
	bool test_move(const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia = true, bool p_exclude_raycast_shapes = true, const Ref<KinematicCollision2D> &r_collision = Ref<KinematicCollision2D>(), real_t p_margin = 0.08);

	TypedArray<PhysicsBody2D> get_collision_exceptions();
	void add_collision_exception_with(Node *p_node); //must be physicsbody
	void remove_collision_exception_with(Node *p_node);

	virtual ~PhysicsBody2D();
};

class StaticBody2D : public PhysicsBody2D {
	GDCLASS(StaticBody2D, PhysicsBody2D);

	Vector2 constant_linear_velocity;
	real_t constant_angular_velocity = 0.0;

	Ref<PhysicsMaterial> physics_material_override;

	bool kinematic_motion = false;
	bool sync_to_physics = false;

	Transform2D last_valid_transform;

	void _direct_state_changed(Object *p_state);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override);
	Ref<PhysicsMaterial> get_physics_material_override() const;

	void set_constant_linear_velocity(const Vector2 &p_vel);
	void set_constant_angular_velocity(real_t p_vel);

	Vector2 get_constant_linear_velocity() const;
	real_t get_constant_angular_velocity() const;

	virtual TypedArray<String> get_configuration_warnings() const override;

	StaticBody2D();

private:
	void _reload_physics_characteristics();

	void _update_kinematic_motion();

	void set_kinematic_motion_enabled(bool p_enabled);
	bool is_kinematic_motion_enabled() const;

	void set_sync_to_physics(bool p_enable);
	bool is_sync_to_physics_enabled() const;
};

class RigidBody2D : public PhysicsBody2D {
	GDCLASS(RigidBody2D, PhysicsBody2D);

public:
	enum Mode {
		MODE_DYNAMIC,
		MODE_STATIC,
		MODE_DYNAMIC_LOCKED,
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
	Mode mode = MODE_DYNAMIC;

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

	virtual TypedArray<String> get_configuration_warnings() const override;

	RigidBody2D();
	~RigidBody2D();

private:
	void _reload_physics_characteristics();
};

VARIANT_ENUM_CAST(RigidBody2D::Mode);
VARIANT_ENUM_CAST(RigidBody2D::CCDMode);

class CharacterBody2D : public PhysicsBody2D {
	GDCLASS(CharacterBody2D, PhysicsBody2D);

private:
	real_t margin = 0.08;

	bool stop_on_slope = false;
	bool infinite_inertia = true;
	int max_slides = 4;
	real_t floor_max_angle = Math::deg2rad((real_t)45.0);
	Vector2 snap;
	Vector2 up_direction = Vector2(0.0, -1.0);

	Vector2 linear_velocity;

	Vector2 floor_normal;
	Vector2 floor_velocity;
	RID on_floor_body;
	bool on_floor = false;
	bool on_ceiling = false;
	bool on_wall = false;

	Vector<PhysicsServer2D::MotionResult> motion_results;
	Vector<Ref<KinematicCollision2D>> slide_colliders;

	Ref<KinematicCollision2D> _get_slide_collision(int p_bounce);

	bool separate_raycast_shapes(PhysicsServer2D::MotionResult &r_result);

	void set_safe_margin(real_t p_margin);
	real_t get_safe_margin() const;

	bool is_stop_on_slope_enabled() const;
	void set_stop_on_slope_enabled(bool p_enabled);

	bool is_infinite_inertia_enabled() const;
	void set_infinite_inertia_enabled(bool p_enabled);

	int get_max_slides() const;
	void set_max_slides(int p_max_slides);

	real_t get_floor_max_angle() const;
	void set_floor_max_angle(real_t p_radians);

	const Vector2 &get_snap() const;
	void set_snap(const Vector2 &p_snap);

	const Vector2 &get_up_direction() const;
	void set_up_direction(const Vector2 &p_up_direction);
	void _set_collision_direction(const PhysicsServer2D::MotionResult &p_result);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void move_and_slide();

	const Vector2 &get_linear_velocity() const;
	void set_linear_velocity(const Vector2 &p_velocity);

	bool is_on_floor() const;
	bool is_on_wall() const;
	bool is_on_ceiling() const;
	Vector2 get_floor_normal() const;
	Vector2 get_floor_velocity() const;

	int get_slide_count() const;
	PhysicsServer2D::MotionResult get_slide_collision(int p_bounce) const;

	CharacterBody2D();
	~CharacterBody2D();
};

class KinematicCollision2D : public RefCounted {
	GDCLASS(KinematicCollision2D, RefCounted);

	PhysicsBody2D *owner = nullptr;
	friend class PhysicsBody2D;
	friend class CharacterBody2D;
	PhysicsServer2D::MotionResult result;

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
	RID get_collider_rid() const;
	Object *get_collider_shape() const;
	int get_collider_shape_index() const;
	Vector2 get_collider_velocity() const;
	Variant get_collider_metadata() const;
};

#endif // PHYSICS_BODY_2D_H

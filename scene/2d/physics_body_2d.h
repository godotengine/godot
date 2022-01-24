/*************************************************************************/
/*  physics_body_2d.h                                                    */
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

	Ref<KinematicCollision2D> _move(const Vector2 &p_linear_velocity, bool p_test_only = false, real_t p_margin = 0.08);

public:
	bool move_and_collide(const PhysicsServer2D::MotionParameters &p_parameters, PhysicsServer2D::MotionResult &r_result, bool p_test_only = false, bool p_cancel_sliding = true);
	bool test_move(const Transform2D &p_from, const Vector2 &p_linear_velocity, const Ref<KinematicCollision2D> &r_collision = Ref<KinematicCollision2D>(), real_t p_margin = 0.08);

	TypedArray<PhysicsBody2D> get_collision_exceptions();
	void add_collision_exception_with(Node *p_node); //must be physicsbody
	void remove_collision_exception_with(Node *p_node);

	virtual ~PhysicsBody2D();
};

class StaticBody2D : public PhysicsBody2D {
	GDCLASS(StaticBody2D, PhysicsBody2D);

private:
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

	StaticBody2D(PhysicsServer2D::BodyMode p_mode = PhysicsServer2D::BODY_MODE_STATIC);

private:
	void _reload_physics_characteristics();
};

class AnimatableBody2D : public StaticBody2D {
	GDCLASS(AnimatableBody2D, StaticBody2D);

private:
	bool sync_to_physics = true;

	Transform2D last_valid_transform;

	static void _body_state_changed_callback(void *p_instance, PhysicsDirectBodyState2D *p_state);
	void _body_state_changed(PhysicsDirectBodyState2D *p_state);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	AnimatableBody2D();

private:
	void _update_kinematic_motion();

	void set_sync_to_physics(bool p_enable);
	bool is_sync_to_physics_enabled() const;
};

class RigidDynamicBody2D : public PhysicsBody2D {
	GDCLASS(RigidDynamicBody2D, PhysicsBody2D);

public:
	enum FreezeMode {
		FREEZE_MODE_STATIC,
		FREEZE_MODE_KINEMATIC,
	};

	enum CenterOfMassMode {
		CENTER_OF_MASS_MODE_AUTO,
		CENTER_OF_MASS_MODE_CUSTOM,
	};

	enum DampMode {
		DAMP_MODE_COMBINE,
		DAMP_MODE_REPLACE,
	};

	enum CCDMode {
		CCD_MODE_DISABLED,
		CCD_MODE_CAST_RAY,
		CCD_MODE_CAST_SHAPE,
	};

private:
	bool can_sleep = true;
	bool lock_rotation = false;
	bool freeze = false;
	FreezeMode freeze_mode = FREEZE_MODE_STATIC;

	real_t mass = 1.0;
	real_t inertia = 0.0;
	CenterOfMassMode center_of_mass_mode = CENTER_OF_MASS_MODE_AUTO;
	Vector2 center_of_mass;

	Ref<PhysicsMaterial> physics_material_override;
	real_t gravity_scale = 1.0;

	DampMode linear_damp_mode = DAMP_MODE_COMBINE;
	DampMode angular_damp_mode = DAMP_MODE_COMBINE;

	real_t linear_damp = 0.0;
	real_t angular_damp = 0.0;

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
	struct RigidDynamicBody2D_RemoveAction {
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

	static void _body_state_changed_callback(void *p_instance, PhysicsDirectBodyState2D *p_state);
	void _body_state_changed(PhysicsDirectBodyState2D *p_state);

protected:
	void _notification(int p_what);
	static void _bind_methods();

	virtual void _validate_property(PropertyInfo &property) const override;

	GDVIRTUAL1(_integrate_forces, PhysicsDirectBodyState2D *)

	void _apply_body_mode();

public:
	void set_lock_rotation_enabled(bool p_lock_rotation);
	bool is_lock_rotation_enabled() const;

	void set_freeze_enabled(bool p_freeze);
	bool is_freeze_enabled() const;

	void set_freeze_mode(FreezeMode p_freeze_mode);
	FreezeMode get_freeze_mode() const;

	void set_mass(real_t p_mass);
	real_t get_mass() const;

	void set_inertia(real_t p_inertia);
	real_t get_inertia() const;

	void set_center_of_mass_mode(CenterOfMassMode p_mode);
	CenterOfMassMode get_center_of_mass_mode() const;

	void set_center_of_mass(const Vector2 &p_center_of_mass);
	const Vector2 &get_center_of_mass() const;

	void set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override);
	Ref<PhysicsMaterial> get_physics_material_override() const;

	void set_gravity_scale(real_t p_gravity_scale);
	real_t get_gravity_scale() const;

	void set_linear_damp_mode(DampMode p_mode);
	DampMode get_linear_damp_mode() const;

	void set_angular_damp_mode(DampMode p_mode);
	DampMode get_angular_damp_mode() const;

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

	void apply_central_force(const Vector2 &p_force);
	void apply_force(const Vector2 &p_force, const Vector2 &p_position = Vector2());
	void apply_torque(real_t p_torque);

	void add_constant_central_force(const Vector2 &p_force);
	void add_constant_force(const Vector2 &p_force, const Vector2 &p_position = Vector2());
	void add_constant_torque(real_t p_torque);

	void set_constant_force(const Vector2 &p_force);
	Vector2 get_constant_force() const;

	void set_constant_torque(real_t p_torque);
	real_t get_constant_torque() const;

	TypedArray<Node2D> get_colliding_bodies() const; //function for script

	virtual TypedArray<String> get_configuration_warnings() const override;

	RigidDynamicBody2D();
	~RigidDynamicBody2D();

private:
	void _reload_physics_characteristics();
};

VARIANT_ENUM_CAST(RigidDynamicBody2D::FreezeMode);
VARIANT_ENUM_CAST(RigidDynamicBody2D::CenterOfMassMode);
VARIANT_ENUM_CAST(RigidDynamicBody2D::DampMode);
VARIANT_ENUM_CAST(RigidDynamicBody2D::CCDMode);

class CharacterBody2D : public PhysicsBody2D {
	GDCLASS(CharacterBody2D, PhysicsBody2D);

public:
	enum MotionMode {
		MOTION_MODE_GROUNDED,
		MOTION_MODE_FLOATING,
	};
	enum MovingPlatformApplyVelocityOnLeave {
		PLATFORM_VEL_ON_LEAVE_ALWAYS,
		PLATFORM_VEL_ON_LEAVE_UPWARD_ONLY,
		PLATFORM_VEL_ON_LEAVE_NEVER,
	};
	bool move_and_slide();

	const Vector2 &get_motion_velocity() const;
	void set_motion_velocity(const Vector2 &p_velocity);

	bool is_on_floor() const;
	bool is_on_floor_only() const;
	bool is_on_wall() const;
	bool is_on_wall_only() const;
	bool is_on_ceiling() const;
	bool is_on_ceiling_only() const;
	const Vector2 &get_last_motion() const;
	Vector2 get_position_delta() const;
	const Vector2 &get_floor_normal() const;
	const Vector2 &get_wall_normal() const;
	const Vector2 &get_real_velocity() const;

	real_t get_floor_angle(const Vector2 &p_up_direction = Vector2(0.0, -1.0)) const;
	const Vector2 &get_platform_velocity() const;

	int get_slide_collision_count() const;
	PhysicsServer2D::MotionResult get_slide_collision(int p_bounce) const;

	CharacterBody2D();
	~CharacterBody2D();

private:
	real_t margin = 0.08;
	MotionMode motion_mode = MOTION_MODE_GROUNDED;
	MovingPlatformApplyVelocityOnLeave moving_platform_apply_velocity_on_leave = PLATFORM_VEL_ON_LEAVE_ALWAYS;

	bool floor_constant_speed = false;
	bool floor_stop_on_slope = true;
	bool floor_block_on_wall = true;
	bool slide_on_ceiling = true;
	int max_slides = 4;
	int platform_layer = 0;
	real_t floor_max_angle = Math::deg2rad((real_t)45.0);
	real_t floor_snap_length = 1;
	real_t wall_min_slide_angle = Math::deg2rad((real_t)15.0);
	Vector2 up_direction = Vector2(0.0, -1.0);
	uint32_t moving_platform_floor_layers = UINT32_MAX;
	uint32_t moving_platform_wall_layers = 0;
	Vector2 motion_velocity;

	Vector2 floor_normal;
	Vector2 platform_velocity;
	Vector2 wall_normal;
	Vector2 last_motion;
	Vector2 previous_position;
	Vector2 real_velocity;

	RID platform_rid;
	ObjectID platform_object_id;
	bool on_floor = false;
	bool on_ceiling = false;
	bool on_wall = false;

	Vector<PhysicsServer2D::MotionResult> motion_results;
	Vector<Ref<KinematicCollision2D>> slide_colliders;

	void set_safe_margin(real_t p_margin);
	real_t get_safe_margin() const;

	bool is_floor_stop_on_slope_enabled() const;
	void set_floor_stop_on_slope_enabled(bool p_enabled);

	bool is_floor_constant_speed_enabled() const;
	void set_floor_constant_speed_enabled(bool p_enabled);

	bool is_floor_block_on_wall_enabled() const;
	void set_floor_block_on_wall_enabled(bool p_enabled);

	bool is_slide_on_ceiling_enabled() const;
	void set_slide_on_ceiling_enabled(bool p_enabled);

	int get_max_slides() const;
	void set_max_slides(int p_max_slides);

	real_t get_floor_max_angle() const;
	void set_floor_max_angle(real_t p_radians);

	real_t get_floor_snap_length();
	void set_floor_snap_length(real_t p_floor_snap_length);

	real_t get_wall_min_slide_angle() const;
	void set_wall_min_slide_angle(real_t p_radians);

	uint32_t get_moving_platform_floor_layers() const;
	void set_moving_platform_floor_layers(const uint32_t p_exclude_layer);

	uint32_t get_moving_platform_wall_layers() const;
	void set_moving_platform_wall_layers(const uint32_t p_exclude_layer);

	void set_motion_mode(MotionMode p_mode);
	MotionMode get_motion_mode() const;

	void set_moving_platform_apply_velocity_on_leave(MovingPlatformApplyVelocityOnLeave p_on_leave_velocity);
	MovingPlatformApplyVelocityOnLeave get_moving_platform_apply_velocity_on_leave() const;

	void _move_and_slide_floating(double p_delta);
	void _move_and_slide_grounded(double p_delta, bool p_was_on_floor);

	Ref<KinematicCollision2D> _get_slide_collision(int p_bounce);
	Ref<KinematicCollision2D> _get_last_slide_collision();
	const Vector2 &get_up_direction() const;
	bool _on_floor_if_snapped(bool was_on_floor, bool vel_dir_facing_up);
	void set_up_direction(const Vector2 &p_up_direction);
	void _set_collision_direction(const PhysicsServer2D::MotionResult &p_result);
	void _set_platform_data(const PhysicsServer2D::MotionResult &p_result);
	void _snap_on_floor(bool was_on_floor, bool vel_dir_facing_up);

protected:
	void _notification(int p_what);
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const override;
};

VARIANT_ENUM_CAST(CharacterBody2D::MotionMode);
VARIANT_ENUM_CAST(CharacterBody2D::MovingPlatformApplyVelocityOnLeave);

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
	real_t get_angle(const Vector2 &p_up_direction = Vector2(0.0, -1.0)) const;
	Object *get_local_shape() const;
	Object *get_collider() const;
	ObjectID get_collider_id() const;
	RID get_collider_rid() const;
	Object *get_collider_shape() const;
	int get_collider_shape_index() const;
	Vector2 get_collider_velocity() const;
};

#endif // PHYSICS_BODY_2D_H

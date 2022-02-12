/*************************************************************************/
/*  physics_body_3d.h                                                    */
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

#ifndef PHYSICS_BODY_3D_H
#define PHYSICS_BODY_3D_H

#include "core/templates/vset.h"
#include "scene/3d/collision_object_3d.h"
#include "scene/resources/physics_material.h"
#include "servers/physics_server_3d.h"
#include "skeleton_3d.h"

class KinematicCollision3D;

class PhysicsBody3D : public CollisionObject3D {
	GDCLASS(PhysicsBody3D, CollisionObject3D);

protected:
	static void _bind_methods();
	PhysicsBody3D(PhysicsServer3D::BodyMode p_mode);

	Ref<KinematicCollision3D> motion_cache;

	uint16_t locked_axis = 0;

	Ref<KinematicCollision3D> _move(const Vector3 &p_distance, bool p_test_only = false, real_t p_margin = 0.001, int p_max_collisions = 1);

public:
	bool move_and_collide(const PhysicsServer3D::MotionParameters &p_parameters, PhysicsServer3D::MotionResult &r_result, bool p_test_only = false, bool p_cancel_sliding = true);
	bool test_move(const Transform3D &p_from, const Vector3 &p_distance, const Ref<KinematicCollision3D> &r_collision = Ref<KinematicCollision3D>(), real_t p_margin = 0.001, int p_max_collisions = 1);

	void set_axis_lock(PhysicsServer3D::BodyAxis p_axis, bool p_lock);
	bool get_axis_lock(PhysicsServer3D::BodyAxis p_axis) const;

	virtual Vector3 get_linear_velocity() const;
	virtual Vector3 get_angular_velocity() const;
	virtual real_t get_inverse_mass() const;

	TypedArray<PhysicsBody3D> get_collision_exceptions();
	void add_collision_exception_with(Node *p_node); //must be physicsbody
	void remove_collision_exception_with(Node *p_node);

	virtual ~PhysicsBody3D();
};

class StaticBody3D : public PhysicsBody3D {
	GDCLASS(StaticBody3D, PhysicsBody3D);

private:
	Vector3 constant_linear_velocity;
	Vector3 constant_angular_velocity;

	Ref<PhysicsMaterial> physics_material_override;

protected:
	static void _bind_methods();

public:
	void set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override);
	Ref<PhysicsMaterial> get_physics_material_override() const;

	void set_constant_linear_velocity(const Vector3 &p_vel);
	void set_constant_angular_velocity(const Vector3 &p_vel);

	Vector3 get_constant_linear_velocity() const;
	Vector3 get_constant_angular_velocity() const;

	StaticBody3D(PhysicsServer3D::BodyMode p_mode = PhysicsServer3D::BODY_MODE_STATIC);

private:
	void _reload_physics_characteristics();
};

class AnimatableBody3D : public StaticBody3D {
	GDCLASS(AnimatableBody3D, StaticBody3D);

private:
	Vector3 linear_velocity;
	Vector3 angular_velocity;

	bool sync_to_physics = true;

	Transform3D last_valid_transform;

	static void _body_state_changed_callback(void *p_instance, PhysicsDirectBodyState3D *p_state);
	void _body_state_changed(PhysicsDirectBodyState3D *p_state);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual Vector3 get_linear_velocity() const override;
	virtual Vector3 get_angular_velocity() const override;

	AnimatableBody3D();

private:
	void _update_kinematic_motion();

	void set_sync_to_physics(bool p_enable);
	bool is_sync_to_physics_enabled() const;
};

class RigidDynamicBody3D : public PhysicsBody3D {
	GDCLASS(RigidDynamicBody3D, PhysicsBody3D);

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

private:
	bool can_sleep = true;
	bool lock_rotation = false;
	bool freeze = false;
	FreezeMode freeze_mode = FREEZE_MODE_STATIC;

	real_t mass = 1.0;
	Vector3 inertia;
	CenterOfMassMode center_of_mass_mode = CENTER_OF_MASS_MODE_AUTO;
	Vector3 center_of_mass;

	Ref<PhysicsMaterial> physics_material_override;

	Vector3 linear_velocity;
	Vector3 angular_velocity;
	Basis inverse_inertia_tensor;
	real_t gravity_scale = 1.0;

	DampMode linear_damp_mode = DAMP_MODE_COMBINE;
	DampMode angular_damp_mode = DAMP_MODE_COMBINE;

	real_t linear_damp = 0.0;
	real_t angular_damp = 0.0;

	bool sleeping = false;
	bool ccd = false;

	int max_contacts_reported = 0;

	bool custom_integrator = false;

	struct ShapePair {
		int body_shape = 0;
		int local_shape = 0;
		bool tagged = false;
		bool operator<(const ShapePair &p_sp) const {
			if (body_shape == p_sp.body_shape) {
				return local_shape < p_sp.local_shape;
			} else {
				return body_shape < p_sp.body_shape;
			}
		}

		ShapePair() {}
		ShapePair(int p_bs, int p_ls) {
			body_shape = p_bs;
			local_shape = p_ls;
			tagged = false;
		}
	};
	struct RigidDynamicBody3D_RemoveAction {
		RID rid;
		ObjectID body_id;
		ShapePair pair;
	};
	struct BodyState {
		RID rid;
		//int rc;
		bool in_tree = false;
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
	static void _body_state_changed_callback(void *p_instance, PhysicsDirectBodyState3D *p_state);

protected:
	void _notification(int p_what);
	static void _bind_methods();

	virtual void _validate_property(PropertyInfo &property) const override;

	GDVIRTUAL1(_integrate_forces, PhysicsDirectBodyState3D *)

	virtual void _body_state_changed(PhysicsDirectBodyState3D *p_state);

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

	virtual real_t get_inverse_mass() const override { return 1.0 / mass; }

	void set_inertia(const Vector3 &p_inertia);
	const Vector3 &get_inertia() const;

	void set_center_of_mass_mode(CenterOfMassMode p_mode);
	CenterOfMassMode get_center_of_mass_mode() const;

	void set_center_of_mass(const Vector3 &p_center_of_mass);
	const Vector3 &get_center_of_mass() const;

	void set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override);
	Ref<PhysicsMaterial> get_physics_material_override() const;

	void set_linear_velocity(const Vector3 &p_velocity);
	Vector3 get_linear_velocity() const override;

	void set_axis_velocity(const Vector3 &p_axis);

	void set_angular_velocity(const Vector3 &p_velocity);
	Vector3 get_angular_velocity() const override;

	Basis get_inverse_inertia_tensor() const;

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

	Array get_colliding_bodies() const;

	void apply_central_impulse(const Vector3 &p_impulse);
	void apply_impulse(const Vector3 &p_impulse, const Vector3 &p_position = Vector3());
	void apply_torque_impulse(const Vector3 &p_impulse);

	void apply_central_force(const Vector3 &p_force);
	void apply_force(const Vector3 &p_force, const Vector3 &p_position = Vector3());
	void apply_torque(const Vector3 &p_torque);

	void add_constant_central_force(const Vector3 &p_force);
	void add_constant_force(const Vector3 &p_force, const Vector3 &p_position = Vector3());
	void add_constant_torque(const Vector3 &p_torque);

	void set_constant_force(const Vector3 &p_force);
	Vector3 get_constant_force() const;

	void set_constant_torque(const Vector3 &p_torque);
	Vector3 get_constant_torque() const;

	virtual TypedArray<String> get_configuration_warnings() const override;

	RigidDynamicBody3D();
	~RigidDynamicBody3D();

private:
	void _reload_physics_characteristics();
};

VARIANT_ENUM_CAST(RigidDynamicBody3D::FreezeMode);
VARIANT_ENUM_CAST(RigidDynamicBody3D::CenterOfMassMode);
VARIANT_ENUM_CAST(RigidDynamicBody3D::DampMode);

class KinematicCollision3D;

class CharacterBody3D : public PhysicsBody3D {
	GDCLASS(CharacterBody3D, PhysicsBody3D);

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

	const Vector3 &get_motion_velocity() const;
	void set_motion_velocity(const Vector3 &p_velocity);

	bool is_on_floor() const;
	bool is_on_floor_only() const;
	bool is_on_wall() const;
	bool is_on_wall_only() const;
	bool is_on_ceiling() const;
	bool is_on_ceiling_only() const;
	const Vector3 &get_last_motion() const;
	Vector3 get_position_delta() const;
	const Vector3 &get_floor_normal() const;
	const Vector3 &get_wall_normal() const;
	const Vector3 &get_real_velocity() const;
	real_t get_floor_angle(const Vector3 &p_up_direction = Vector3(0.0, 1.0, 0.0)) const;
	const Vector3 &get_platform_velocity() const;

	virtual Vector3 get_linear_velocity() const override;

	int get_slide_collision_count() const;
	PhysicsServer3D::MotionResult get_slide_collision(int p_bounce) const;

	CharacterBody3D();
	~CharacterBody3D();

private:
	real_t margin = 0.001;
	MotionMode motion_mode = MOTION_MODE_GROUNDED;
	MovingPlatformApplyVelocityOnLeave moving_platform_apply_velocity_on_leave = PLATFORM_VEL_ON_LEAVE_ALWAYS;
	union CollisionState {
		uint32_t state = 0;
		struct {
			bool floor;
			bool wall;
			bool ceiling;
		};

		CollisionState() {
		}

		CollisionState(bool p_floor, bool p_wall, bool p_ceiling) {
			floor = p_floor;
			wall = p_wall;
			ceiling = p_ceiling;
		}
	};

	CollisionState collision_state;
	bool floor_constant_speed = false;
	bool floor_stop_on_slope = true;
	bool floor_block_on_wall = true;
	bool slide_on_ceiling = true;
	int max_slides = 6;
	int platform_layer = 0;
	RID platform_rid;
	ObjectID platform_object_id;
	uint32_t moving_platform_floor_layers = UINT32_MAX;
	uint32_t moving_platform_wall_layers = 0;
	real_t floor_snap_length = 0.1;
	real_t floor_max_angle = Math::deg2rad((real_t)45.0);
	real_t wall_min_slide_angle = Math::deg2rad((real_t)15.0);
	Vector3 up_direction = Vector3(0.0, 1.0, 0.0);
	Vector3 motion_velocity;
	Vector3 floor_normal;
	Vector3 wall_normal;
	Vector3 ceiling_normal;
	Vector3 last_motion;
	Vector3 platform_velocity;
	Vector3 platform_ceiling_velocity;
	Vector3 previous_position;
	Vector3 real_velocity;

	Vector<PhysicsServer3D::MotionResult> motion_results;
	Vector<Ref<KinematicCollision3D>> slide_colliders;

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

	Ref<KinematicCollision3D> _get_slide_collision(int p_bounce);
	Ref<KinematicCollision3D> _get_last_slide_collision();
	const Vector3 &get_up_direction() const;
	bool _on_floor_if_snapped(bool was_on_floor, bool vel_dir_facing_up);
	void set_up_direction(const Vector3 &p_up_direction);
	void _set_collision_direction(const PhysicsServer3D::MotionResult &p_result, CollisionState &r_state, CollisionState p_apply_state = CollisionState(true, true, true));
	void _set_platform_data(const PhysicsServer3D::MotionCollision &p_collision);
	void _snap_on_floor(bool was_on_floor, bool vel_dir_facing_up);

protected:
	void _notification(int p_what);
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const override;
};

VARIANT_ENUM_CAST(CharacterBody3D::MotionMode);
VARIANT_ENUM_CAST(CharacterBody3D::MovingPlatformApplyVelocityOnLeave);

class KinematicCollision3D : public RefCounted {
	GDCLASS(KinematicCollision3D, RefCounted);

	PhysicsBody3D *owner = nullptr;
	friend class PhysicsBody3D;
	friend class CharacterBody3D;
	PhysicsServer3D::MotionResult result;

protected:
	static void _bind_methods();

public:
	Vector3 get_travel() const;
	Vector3 get_remainder() const;
	int get_collision_count() const;
	Vector3 get_position(int p_collision_index = 0) const;
	Vector3 get_normal(int p_collision_index = 0) const;
	real_t get_angle(int p_collision_index = 0, const Vector3 &p_up_direction = Vector3(0.0, 1.0, 0.0)) const;
	Object *get_local_shape(int p_collision_index = 0) const;
	Object *get_collider(int p_collision_index = 0) const;
	ObjectID get_collider_id(int p_collision_index = 0) const;
	RID get_collider_rid(int p_collision_index = 0) const;
	Object *get_collider_shape(int p_collision_index = 0) const;
	int get_collider_shape_index(int p_collision_index = 0) const;
	Vector3 get_collider_velocity(int p_collision_index = 0) const;
};

class PhysicalBone3D : public PhysicsBody3D {
	GDCLASS(PhysicalBone3D, PhysicsBody3D);

public:
	enum DampMode {
		DAMP_MODE_COMBINE,
		DAMP_MODE_REPLACE,
	};

	enum JointType {
		JOINT_TYPE_NONE,
		JOINT_TYPE_PIN,
		JOINT_TYPE_CONE,
		JOINT_TYPE_HINGE,
		JOINT_TYPE_SLIDER,
		JOINT_TYPE_6DOF
	};

	struct JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_NONE; }

		/// "j" is used to set the parameter inside the PhysicsServer3D
		virtual bool _set(const StringName &p_name, const Variant &p_value, RID j);
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		virtual ~JointData() {}
	};

	struct PinJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_PIN; }

		virtual bool _set(const StringName &p_name, const Variant &p_value, RID j);
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		real_t bias = 0.3;
		real_t damping = 1.0;
		real_t impulse_clamp = 0.0;
	};

	struct ConeJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_CONE; }

		virtual bool _set(const StringName &p_name, const Variant &p_value, RID j);
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		real_t swing_span = Math_PI * 0.25;
		real_t twist_span = Math_PI;
		real_t bias = 0.3;
		real_t softness = 0.8;
		real_t relaxation = 1.;
	};

	struct HingeJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_HINGE; }

		virtual bool _set(const StringName &p_name, const Variant &p_value, RID j);
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		bool angular_limit_enabled = false;
		real_t angular_limit_upper = Math_PI * 0.5;
		real_t angular_limit_lower = -Math_PI * 0.5;
		real_t angular_limit_bias = 0.3;
		real_t angular_limit_softness = 0.9;
		real_t angular_limit_relaxation = 1.;
	};

	struct SliderJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_SLIDER; }

		virtual bool _set(const StringName &p_name, const Variant &p_value, RID j);
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		real_t linear_limit_upper = 1.0;
		real_t linear_limit_lower = -1.0;
		real_t linear_limit_softness = 1.0;
		real_t linear_limit_restitution = 0.7;
		real_t linear_limit_damping = 1.0;
		real_t angular_limit_upper = 0.0;
		real_t angular_limit_lower = 0.0;
		real_t angular_limit_softness = 1.0;
		real_t angular_limit_restitution = 0.7;
		real_t angular_limit_damping = 1.0;
	};

	struct SixDOFJointData : public JointData {
		struct SixDOFAxisData {
			bool linear_limit_enabled = true;
			real_t linear_limit_upper = 0.0;
			real_t linear_limit_lower = 0.0;
			real_t linear_limit_softness = 0.7;
			real_t linear_restitution = 0.5;
			real_t linear_damping = 1.0;
			bool linear_spring_enabled = false;
			real_t linear_spring_stiffness = 0.0;
			real_t linear_spring_damping = 0.0;
			real_t linear_equilibrium_point = 0.0;
			bool angular_limit_enabled = true;
			real_t angular_limit_upper = 0.0;
			real_t angular_limit_lower = 0.0;
			real_t angular_limit_softness = 0.5;
			real_t angular_restitution = 0.0;
			real_t angular_damping = 1.0;
			real_t erp = 0.5;
			bool angular_spring_enabled = false;
			real_t angular_spring_stiffness = 0.0;
			real_t angular_spring_damping = 0.0;
			real_t angular_equilibrium_point = 0.0;
		};

		virtual JointType get_joint_type() { return JOINT_TYPE_6DOF; }

		virtual bool _set(const StringName &p_name, const Variant &p_value, RID j);
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		SixDOFAxisData axis_data[3];

		SixDOFJointData() {}
	};

private:
#ifdef TOOLS_ENABLED
	// if false gizmo move body
	bool gizmo_move_joint = false;
#endif

	JointData *joint_data = nullptr;
	Transform3D joint_offset;
	RID joint;

	Skeleton3D *parent_skeleton = nullptr;
	Transform3D body_offset;
	Transform3D body_offset_inverse;
	bool simulate_physics = false;
	bool _internal_simulate_physics = false;
	int bone_id = -1;

	String bone_name;
	real_t bounce = 0.0;
	real_t mass = 1.0;
	real_t friction = 1.0;
	real_t gravity_scale = 1.0;
	bool can_sleep = true;

	DampMode linear_damp_mode = DAMP_MODE_COMBINE;
	DampMode angular_damp_mode = DAMP_MODE_COMBINE;

	real_t linear_damp = 0.0;
	real_t angular_damp = 0.0;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);
	static void _body_state_changed_callback(void *p_instance, PhysicsDirectBodyState3D *p_state);
	void _body_state_changed(PhysicsDirectBodyState3D *p_state);

	static void _bind_methods();

private:
	static Skeleton3D *find_skeleton_parent(Node *p_parent);

	void _update_joint_offset();
	void _fix_joint_offset();
	void _reload_joint();

public:
	void _on_bone_parent_changed();

#ifdef TOOLS_ENABLED
	void _set_gizmo_move_joint(bool p_move_joint);
	virtual Transform3D get_global_gizmo_transform() const override;
	virtual Transform3D get_local_gizmo_transform() const override;
#endif

	const JointData *get_joint_data() const;
	Skeleton3D *find_skeleton_parent();

	int get_bone_id() const { return bone_id; }

	void set_joint_type(JointType p_joint_type);
	JointType get_joint_type() const;

	void set_joint_offset(const Transform3D &p_offset);
	const Transform3D &get_joint_offset() const;

	void set_joint_rotation(const Vector3 &p_euler_rad);
	Vector3 get_joint_rotation() const;

	void set_body_offset(const Transform3D &p_offset);
	const Transform3D &get_body_offset() const;

	void set_simulate_physics(bool p_simulate);
	bool get_simulate_physics();
	bool is_simulating_physics();

	void set_bone_name(const String &p_name);
	const String &get_bone_name() const;

	void set_mass(real_t p_mass);
	real_t get_mass() const;

	void set_friction(real_t p_friction);
	real_t get_friction() const;

	void set_bounce(real_t p_bounce);
	real_t get_bounce() const;

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

	void set_can_sleep(bool p_active);
	bool is_able_to_sleep() const;

	void apply_central_impulse(const Vector3 &p_impulse);
	void apply_impulse(const Vector3 &p_impulse, const Vector3 &p_position = Vector3());

	void reset_physics_simulation_state();
	void reset_to_rest_position();

	PhysicalBone3D();
	~PhysicalBone3D();

private:
	void update_bone_id();
	void update_offset();

	void _start_physics_simulation();
	void _stop_physics_simulation();
};

VARIANT_ENUM_CAST(PhysicalBone3D::JointType);
VARIANT_ENUM_CAST(PhysicalBone3D::DampMode);

#endif // PHYSICS_BODY__H

/*************************************************************************/
/*  physics_body_3d.h                                                    */
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

#ifndef PHYSICS_BODY_3D_H
#define PHYSICS_BODY_3D_H

#include "core/templates/vset.h"
#include "scene/3d/collision_object_3d.h"
#include "scene/resources/physics_material.h"
#include "servers/physics_server_3d.h"
#include "skeleton_3d.h"

class PhysicsBody3D : public CollisionObject3D {
	GDCLASS(PhysicsBody3D, CollisionObject3D);

protected:
	static void _bind_methods();
	PhysicsBody3D(PhysicsServer3D::BodyMode p_mode);

public:
	virtual Vector3 get_linear_velocity() const;
	virtual Vector3 get_angular_velocity() const;
	virtual real_t get_inverse_mass() const;

	TypedArray<PhysicsBody3D> get_collision_exceptions();
	void add_collision_exception_with(Node *p_node); //must be physicsbody
	void remove_collision_exception_with(Node *p_node);

	PhysicsBody3D();
};

class StaticBody3D : public PhysicsBody3D {
	GDCLASS(StaticBody3D, PhysicsBody3D);

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

	StaticBody3D();
	~StaticBody3D();

private:
	void _reload_physics_characteristics();
};

class RigidBody3D : public PhysicsBody3D {
	GDCLASS(RigidBody3D, PhysicsBody3D);

public:
	enum Mode {
		MODE_RIGID,
		MODE_STATIC,
		MODE_CHARACTER,
		MODE_KINEMATIC,
	};

protected:
	bool can_sleep = true;
	PhysicsDirectBodyState3D *state = nullptr;
	Mode mode = MODE_RIGID;

	real_t mass = 1.0;
	Ref<PhysicsMaterial> physics_material_override;

	Vector3 linear_velocity;
	Vector3 angular_velocity;
	Basis inverse_inertia_tensor;
	real_t gravity_scale = 1.0;
	real_t linear_damp = -1.0;
	real_t angular_damp = -1.0;

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
	struct RigidBody3D_RemoveAction {
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
	virtual void _direct_state_changed(Object *p_state);

	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_mode(Mode p_mode);
	Mode get_mode() const;

	void set_mass(real_t p_mass);
	real_t get_mass() const;

	virtual real_t get_inverse_mass() const override { return 1.0 / mass; }

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

	void set_axis_lock(PhysicsServer3D::BodyAxis p_axis, bool p_lock);
	bool get_axis_lock(PhysicsServer3D::BodyAxis p_axis) const;

	Array get_colliding_bodies() const;

	void add_central_force(const Vector3 &p_force);
	void add_force(const Vector3 &p_force, const Vector3 &p_position = Vector3());
	void add_torque(const Vector3 &p_torque);

	void apply_central_impulse(const Vector3 &p_impulse);
	void apply_impulse(const Vector3 &p_impulse, const Vector3 &p_position = Vector3());
	void apply_torque_impulse(const Vector3 &p_impulse);

	TypedArray<String> get_configuration_warnings() const override;

	RigidBody3D();
	~RigidBody3D();

private:
	void _reload_physics_characteristics();
};

VARIANT_ENUM_CAST(RigidBody3D::Mode);

class KinematicCollision3D;

class KinematicBody3D : public PhysicsBody3D {
	GDCLASS(KinematicBody3D, PhysicsBody3D);

public:
	struct Collision {
		Vector3 collision;
		Vector3 normal;
		Vector3 collider_vel;
		ObjectID collider;
		RID collider_rid;
		int collider_shape = 0;
		Variant collider_metadata;
		Vector3 remainder;
		Vector3 travel;
		int local_shape = 0;
	};

private:
	Vector3 linear_velocity;
	Vector3 angular_velocity;

	uint16_t locked_axis = 0;

	real_t margin;

	Vector3 floor_normal;
	Vector3 floor_velocity;
	RID on_floor_body;
	bool on_floor = false;
	bool on_ceiling = false;
	bool on_wall = false;
	Vector<Collision> colliders;
	Vector<Ref<KinematicCollision3D>> slide_colliders;
	Ref<KinematicCollision3D> motion_cache;

	_FORCE_INLINE_ bool _ignores_mode(PhysicsServer3D::BodyMode) const;

	Ref<KinematicCollision3D> _move(const Vector3 &p_motion, bool p_infinite_inertia = true, bool p_exclude_raycast_shapes = true, bool p_test_only = false);
	Ref<KinematicCollision3D> _get_slide_collision(int p_bounce);

protected:
	void _notification(int p_what);
	static void _bind_methods();

	virtual void _direct_state_changed(Object *p_state);

public:
	virtual Vector3 get_linear_velocity() const override;
	virtual Vector3 get_angular_velocity() const override;

	bool move_and_collide(const Vector3 &p_motion, bool p_infinite_inertia, Collision &r_collision, bool p_exclude_raycast_shapes = true, bool p_test_only = false);
	bool test_move(const Transform &p_from, const Vector3 &p_motion, bool p_infinite_inertia);

	bool separate_raycast_shapes(bool p_infinite_inertia, Collision &r_collision);

	void set_axis_lock(PhysicsServer3D::BodyAxis p_axis, bool p_lock);
	bool get_axis_lock(PhysicsServer3D::BodyAxis p_axis) const;

	void set_safe_margin(real_t p_margin);
	real_t get_safe_margin() const;

	Vector3 move_and_slide(const Vector3 &p_linear_velocity, const Vector3 &p_up_direction = Vector3(0, 0, 0), bool p_stop_on_slope = false, int p_max_slides = 4, real_t p_floor_max_angle = Math::deg2rad((real_t)45.0), bool p_infinite_inertia = true);
	Vector3 move_and_slide_with_snap(const Vector3 &p_linear_velocity, const Vector3 &p_snap, const Vector3 &p_up_direction = Vector3(0, 0, 0), bool p_stop_on_slope = false, int p_max_slides = 4, real_t p_floor_max_angle = Math::deg2rad((real_t)45.0), bool p_infinite_inertia = true);
	bool is_on_floor() const;
	bool is_on_wall() const;
	bool is_on_ceiling() const;
	Vector3 get_floor_normal() const;
	Vector3 get_floor_velocity() const;

	int get_slide_count() const;
	Collision get_slide_collision(int p_bounce) const;

	KinematicBody3D();
	~KinematicBody3D();
};

class KinematicCollision3D : public Reference {
	GDCLASS(KinematicCollision3D, Reference);

	KinematicBody3D *owner;
	friend class KinematicBody3D;
	KinematicBody3D::Collision collision;

protected:
	static void _bind_methods();

public:
	Vector3 get_position() const;
	Vector3 get_normal() const;
	Vector3 get_travel() const;
	Vector3 get_remainder() const;
	Object *get_local_shape() const;
	Object *get_collider() const;
	ObjectID get_collider_id() const;
	Object *get_collider_shape() const;
	int get_collider_shape_index() const;
	Vector3 get_collider_velocity() const;
	Variant get_collider_metadata() const;

	KinematicCollision3D();
};

class PhysicalBone3D : public PhysicsBody3D {
	GDCLASS(PhysicalBone3D, PhysicsBody3D);

public:
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
	Transform joint_offset;
	RID joint;

	Skeleton3D *parent_skeleton = nullptr;
	Transform body_offset;
	Transform body_offset_inverse;
	bool simulate_physics = false;
	bool _internal_simulate_physics = false;
	int bone_id = -1;

	String bone_name;
	real_t bounce = 0.0;
	real_t mass = 1.0;
	real_t friction = 1.0;
	real_t gravity_scale = 1.0;
	real_t linear_damp = -1.0;
	real_t angular_damp = -1.0;
	bool can_sleep = true;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);
	void _direct_state_changed(Object *p_state);

	static void _bind_methods();

private:
	static Skeleton3D *find_skeleton_parent(Node *p_parent);

	void _update_joint_offset();
	void _fix_joint_offset();
	void _reload_joint();

public:
	void _on_bone_parent_changed();
	void _set_gizmo_move_joint(bool p_move_joint);

public:
#ifdef TOOLS_ENABLED
	virtual Transform get_global_gizmo_transform() const override;
	virtual Transform get_local_gizmo_transform() const override;
#endif

	const JointData *get_joint_data() const;
	Skeleton3D *find_skeleton_parent();

	int get_bone_id() const { return bone_id; }

	void set_joint_type(JointType p_joint_type);
	JointType get_joint_type() const;

	void set_joint_offset(const Transform &p_offset);
	const Transform &get_joint_offset() const;

	void set_joint_rotation(const Vector3 &p_euler_rad);
	Vector3 get_joint_rotation() const;

	void set_joint_rotation_degrees(const Vector3 &p_euler_deg);
	Vector3 get_joint_rotation_degrees() const;

	void set_body_offset(const Transform &p_offset);
	const Transform &get_body_offset() const;

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

	void set_linear_damp(real_t p_linear_damp);
	real_t get_linear_damp() const;

	void set_angular_damp(real_t p_angular_damp);
	real_t get_angular_damp() const;

	void set_can_sleep(bool p_active);
	bool is_able_to_sleep() const;

	void set_axis_lock(PhysicsServer3D::BodyAxis p_axis, bool p_lock);
	bool get_axis_lock(PhysicsServer3D::BodyAxis p_axis) const;

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

#endif // PHYSICS_BODY__H

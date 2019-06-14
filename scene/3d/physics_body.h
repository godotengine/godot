/*************************************************************************/
/*  physics_body.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/vset.h"
#include "scene/3d/collision_object.h"
#include "scene/resources/physics_material.h"
#include "servers/physics_server.h"
#include "skeleton.h"

class PhysicsBody : public CollisionObject {

	GDCLASS(PhysicsBody, CollisionObject);

protected:
	static void _bind_methods();
	void _notification(int p_what);
	PhysicsBody(PhysicsServer::BodyMode p_mode);

public:
	virtual Vector3 get_linear_velocity() const;
	virtual Vector3 get_angular_velocity() const;
	virtual float get_inverse_mass() const;

	PhysicsBody();
};

class StaticBody : public PhysicsBody {

	GDCLASS(StaticBody, PhysicsBody);

	Vector3 constant_linear_velocity;
	Vector3 constant_angular_velocity;

	Ref<PhysicsMaterial> physics_material_override;

protected:
	static void _bind_methods();

public:
#ifndef DISABLE_DEPRECATED
	void set_friction(real_t p_friction);
	real_t get_friction() const;

	void set_bounce(real_t p_bounce);
	real_t get_bounce() const;
#endif

	void set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override);
	Ref<PhysicsMaterial> get_physics_material_override() const;

	void set_constant_linear_velocity(const Vector3 &p_vel);
	void set_constant_angular_velocity(const Vector3 &p_vel);

	Vector3 get_constant_linear_velocity() const;
	Vector3 get_constant_angular_velocity() const;

	StaticBody();
	~StaticBody();

private:
	void _reload_physics_characteristics();
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

protected:
	bool can_sleep;
	PhysicsDirectBodyState *state;
	Mode mode;

	real_t mass;
	Ref<PhysicsMaterial> physics_material_override;

	Vector3 linear_velocity;
	Vector3 angular_velocity;
	real_t gravity_scale;
	real_t linear_damp;
	real_t angular_damp;

	bool sleeping;
	bool ccd;

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
	virtual void _direct_state_changed(Object *p_state);

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

#ifndef DISABLE_DEPRECATED
	void set_friction(real_t p_friction);
	real_t get_friction() const;

	void set_bounce(real_t p_bounce);
	real_t get_bounce() const;
#endif

	void set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override);
	Ref<PhysicsMaterial> get_physics_material_override() const;

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

	void set_axis_lock(PhysicsServer::BodyAxis p_axis, bool p_lock);
	bool get_axis_lock(PhysicsServer::BodyAxis p_axis) const;

	Array get_colliding_bodies() const;

	void add_central_force(const Vector3 &p_force);
	void add_force(const Vector3 &p_force, const Vector3 &p_pos);
	void add_torque(const Vector3 &p_torque);

	void apply_central_impulse(const Vector3 &p_impulse);
	void apply_impulse(const Vector3 &p_pos, const Vector3 &p_impulse);
	void apply_torque_impulse(const Vector3 &p_impulse);

	virtual String get_configuration_warning() const;

	RigidBody();
	~RigidBody();

private:
	void _reload_physics_characteristics();
};

VARIANT_ENUM_CAST(RigidBody::Mode);

class KinematicCollision;

class KinematicBody : public PhysicsBody {

	GDCLASS(KinematicBody, PhysicsBody);

public:
	struct Collision {
		Vector3 collision;
		Vector3 normal;
		Vector3 collider_vel;
		ObjectID collider;
		RID collider_rid;
		int collider_shape;
		Variant collider_metadata;
		Vector3 remainder;
		Vector3 travel;
		int local_shape;
	};

private:
	uint16_t locked_axis;

	float margin;

	Vector3 floor_velocity;
	RID on_floor_body;
	bool on_floor;
	bool on_ceiling;
	bool on_wall;
	Vector<Collision> colliders;
	Vector<Ref<KinematicCollision> > slide_colliders;
	Ref<KinematicCollision> motion_cache;

	_FORCE_INLINE_ bool _ignores_mode(PhysicsServer::BodyMode) const;

	Ref<KinematicCollision> _move(const Vector3 &p_motion, bool p_infinite_inertia = true, bool p_exclude_raycast_shapes = true, bool p_test_only = false);
	Ref<KinematicCollision> _get_slide_collision(int p_bounce);

protected:
	static void _bind_methods();

public:
	bool move_and_collide(const Vector3 &p_motion, bool p_infinite_inertia, Collision &r_collision, bool p_exclude_raycast_shapes = true, bool p_test_only = false);
	bool test_move(const Transform &p_from, const Vector3 &p_motion, bool p_infinite_inertia);

	bool separate_raycast_shapes(bool p_infinite_inertia, Collision &r_collision);

	void set_axis_lock(PhysicsServer::BodyAxis p_axis, bool p_lock);
	bool get_axis_lock(PhysicsServer::BodyAxis p_axis) const;

	void set_safe_margin(float p_margin);
	float get_safe_margin() const;

	Vector3 move_and_slide(const Vector3 &p_linear_velocity, const Vector3 &p_floor_direction = Vector3(0, 0, 0), bool p_stop_on_slope = false, int p_max_slides = 4, float p_floor_max_angle = Math::deg2rad((float)45), bool p_infinite_inertia = true);
	Vector3 move_and_slide_with_snap(const Vector3 &p_linear_velocity, const Vector3 &p_snap, const Vector3 &p_floor_direction = Vector3(0, 0, 0), bool p_stop_on_slope = false, int p_max_slides = 4, float p_floor_max_angle = Math::deg2rad((float)45), bool p_infinite_inertia = true);
	bool is_on_floor() const;
	bool is_on_wall() const;
	bool is_on_ceiling() const;
	Vector3 get_floor_velocity() const;

	int get_slide_count() const;
	Collision get_slide_collision(int p_bounce) const;

	KinematicBody();
	~KinematicBody();
};

class KinematicCollision : public Reference {

	GDCLASS(KinematicCollision, Reference);

	KinematicBody *owner;
	friend class KinematicBody;
	KinematicBody::Collision collision;

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

	KinematicCollision();
};

class PhysicalBone : public CollisionObject {

	GDCLASS(PhysicalBone, CollisionObject);

	friend class Skeleton;

public:
	enum JointType {
		JOINT_TYPE_NONE,
		JOINT_TYPE_FIXED,
		JOINT_TYPE_SLIDER,
		JOINT_TYPE_HINGE,
		JOINT_TYPE_SPHERICAL,
		JOINT_TYPE_PLANAR
	};

	struct JointData {

		virtual JointType get_joint_type() { return JOINT_TYPE_NONE; }

		/// "j" is used to set the parameter inside the PhysicsServer
		virtual bool _set(const StringName &p_name, const Variant &p_value);
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		JointData() {}

		virtual ~JointData() {}
	};

	struct FixedJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_FIXED; }

		virtual bool _set(const StringName &p_name, const Variant &p_value);
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		FixedJointData() :
				JointData() {}
	};

	struct SliderJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_SLIDER; }

		virtual bool _set(const StringName &p_name, const Variant &p_value);
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		// Limits
		bool limit_active;
		real_t lower_limit;
		real_t upper_limit;

		// Motors
		bool motor_is_enabled;
		real_t motor_velocity_target;
		real_t motor_position_target;
		real_t motor_max_impulse;
		real_t motor_error_reduction_parameter;
		real_t motor_spring_constant;
		real_t motor_damping_constant;
		real_t motor_maximum_error;

		SliderJointData() :
				JointData(),
				limit_active(true),
				lower_limit(-1),
				upper_limit(1),
				motor_is_enabled(false),
				motor_velocity_target(0),
				motor_position_target(0),
				motor_max_impulse(1),
				motor_error_reduction_parameter(1),
				motor_spring_constant(0.1),
				motor_damping_constant(1),
				motor_maximum_error(99999.) {}
	};

	struct HingeJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_HINGE; }

		virtual bool _set(const StringName &p_name, const Variant &p_value);
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		// Limits
		bool limit_active;
		real_t lower_limit;
		real_t upper_limit;

		// Motors
		bool motor_is_enabled;
		real_t motor_velocity_target;
		real_t motor_position_target;
		real_t motor_max_impulse;
		real_t motor_error_reduction_parameter;
		real_t motor_spring_constant;
		real_t motor_damping_constant;
		real_t motor_maximum_error;

		HingeJointData() :
				JointData(),
				limit_active(true),
				lower_limit(-90),
				upper_limit(90),
				motor_is_enabled(false),
				motor_velocity_target(0),
				motor_position_target(0),
				motor_max_impulse(1),
				motor_error_reduction_parameter(1),
				motor_spring_constant(0.1),
				motor_damping_constant(1),
				motor_maximum_error(99999.) {}
	};

	struct SphericalJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_SPHERICAL; }

		virtual bool _set(const StringName &p_name, const Variant &p_value);
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		// Motors
		bool motor_is_enabled;
		Vector3 motor_velocity_target;
		Vector3 motor_rotation_target;
		real_t motor_max_impulse;
		real_t motor_error_reduction_parameter;
		real_t motor_spring_constant;
		real_t motor_damping_constant;
		real_t motor_maximum_error;

		SphericalJointData() :
				JointData(),
				motor_is_enabled(false),
				motor_velocity_target(0, 0, 0),
				motor_rotation_target(0, 0, 0),
				motor_max_impulse(1),
				motor_error_reduction_parameter(1),
				motor_spring_constant(0.1),
				motor_damping_constant(1),
				motor_maximum_error(99999.) {}
	};

	struct PlanarJointData : public JointData {
		virtual JointType get_joint_type() { return JOINT_TYPE_PLANAR; }

		virtual bool _set(const StringName &p_name, const Variant &p_value);
		virtual bool _get(const StringName &p_name, Variant &r_ret) const;
		virtual void _get_property_list(List<PropertyInfo> *p_list) const;

		PlanarJointData() :
				JointData() {}
	};

	// Contact monitoring stuff
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

private:
#ifdef TOOLS_ENABLED
	// if false gizmo move body
	bool gizmo_move_joint;
#endif

	ContactMonitor *contact_monitor;
	int max_contacts_reported;

	JointData *joint_data;
	Transform joint_offset;
	bool disable_parent_collision;
	RID joint;

	Skeleton *parent_skeleton;
	Transform body_offset;
	Transform body_offset_inverse;
	bool _internal_simulate_physics;
	int bone_id;
	int physical_bone_id;
	int parent_physical_bone_id;

	String bone_name;
	real_t bounce;
	real_t link_mass;
	real_t friction;
	real_t gravity_scale;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);
	void _direct_state_changed(Object *p_state);

	void _body_enter_tree(ObjectID p_id);
	void _body_exit_tree(ObjectID p_id);

	void _body_inout(int p_status, ObjectID p_instance, int p_body_shape, int p_local_shape);

	static void _bind_methods();

private:
	static Skeleton *find_skeleton_parent(Node *p_parent);

	void _reset_joint_offset_origin();

public:
	void _setup_physical_bone();
	void _reload_joint();
	void _set_gizmo_move_joint(bool p_move_joint);

public:
#ifdef TOOLS_ENABLED
	virtual Transform get_global_gizmo_transform() const;
	virtual Transform get_local_gizmo_transform() const;
#endif

	const JointData *get_joint_data() const;
	Skeleton *find_skeleton_parent();

	void set_disable_parent_collision(bool p_col);
	bool get_disable_parent_collision() const;

	void set_joint_type(JointType p_joint_type);
	JointType get_joint_type() const;

	void set_joint_offset(const Transform &p_offset);
	const Transform &get_joint_offset() const;

	void set_body_offset(const Transform &p_offset);
	const Transform &get_body_offset() const;

	void set_static_body(bool p_static);
	bool is_static_body();

	void reset_physics_simulation_state();
	bool is_simulating_physics();

	void set_bone_id(int p_bode_id);
	int get_bone_id() const { return bone_id; }

	void set_bone_name(const String &p_name);
	const String &get_bone_name() const;

	void set_is_static(bool p_static);
	bool get_is_static() const;

	void set_link_mass(real_t p_mass);
	real_t get_link_mass() const;

	void set_weight(real_t p_weight);
	real_t get_weight() const;

	void set_friction(real_t p_friction);
	real_t get_friction() const;

	void set_bounce(real_t p_bounce);
	real_t get_bounce() const;

	void set_contact_monitor(bool p_enabled);
	bool is_contact_monitor_enabled() const;

	void set_max_contacts_reported(int p_amount);
	int get_max_contacts_reported() const;

	Array get_colliding_bodies() const;

	void motor_set_active(bool p_active);
	void motor_set_position_target(real_t p_position);
	void motor_set_rotation_target(Vector3 p_rotation);
	void motor_set_rotation_target_basis(Basis p_rotation);
	void motor_set_velocity(Vector3 p_velocity);
	void motor_set_max_impulse(real_t p_impulse);
	void motor_set_error_reduction_parameter(real_t p_erp);
	void motor_set_spring_constant(real_t p_sk);
	void motor_set_damping_constant(real_t p_dk);
	void motor_set_maximum_error(real_t p_me);

	Vector3 get_joint_force();
	Vector3 get_joint_torque();

	PhysicalBone();
	virtual ~PhysicalBone();

private:
	void update_bone_id();
	void update_offset();
	void reset_to_rest_position();

	void _update_link_mass();

	void _start_physics_simulation();
	void _stop_physics_simulation();
};

VARIANT_ENUM_CAST(PhysicalBone::JointType);

#endif // PHYSICS_BODY__H

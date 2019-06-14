/*************************************************************************/
/*  bullet_physics_server.h                                              */
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

#ifndef ARMATURE_BULLET_H
#define ARMATURE_BULLET_H

/**
	@author AndreaCatania
*/

#include "collision_object_bullet.h"
#include "space_bullet.h"
#include <BulletDynamics/Featherstone/btMultiBodyLink.h>

class btMultiBody;
class btMultiBodyLinkCollider;
class btMultiBodyConstraint;
class JointBullet;

class ArmatureBullet : public RIDBullet {

	struct ForceIntegrationCallback {
		ObjectID id;
		StringName method;
		Variant udata;
	};

	SpaceBullet *space;
	btMultiBody *bt_body;
	bool active;
	btTransform transform;

	ForceIntegrationCallback *force_integration_callback;

	/**
	 * @brief The list of joints between two multibody, or between a rigid body
	 * So this is not the list of internal joints
	 */
	Vector<JointBullet *> ext_joints;

public:
	ArmatureBullet();
	~ArmatureBullet();

	btMultiBody *get_bt_body() const;

	void set_bone_count(int p_count);
	int get_bone_count() const;

	void set_space(SpaceBullet *p_space);
	SpaceBullet *get_space() const;

	void set_active(bool p_active);
	bool is_active() const { return active; }

	void set_transform(const Transform &p_global_transform);
	Transform get_transform() const;
	void set_transform__bullet(const btTransform &p_global_transform);
	const btTransform &get_transform__bullet() const;

	void set_force_integration_callback(
			ObjectID p_id,
			const StringName &p_method,
			const Variant &p_udata = Variant());

	void set_bone(BoneBullet *p_bone);
	BoneBullet *get_bone(int p_link_id) const;
	void remove_bone(int p_link_id);

	void update_base_inertia();
	void update_link_mass_and_inertia(int p_link_id);

	void set_param(PhysicsServer::ArmatureParameter p_param, real_t p_value);
	real_t get_param(PhysicsServer::ArmatureParameter p_param) const;

	void register_ext_joint(JointBullet *p_joint);
	void erase_ext_joint(JointBullet *p_joint);

private:
	void clear_links();

	void update_activation();
	void update_ext_joints();
};

class BoneBullet;

class BoneBulletPhysicsDirectBodyState : public PhysicsDirectBodyState {
	GDCLASS(BoneBulletPhysicsDirectBodyState, PhysicsDirectBodyState)

	static BoneBulletPhysicsDirectBodyState *singleton;

public:
	/// This class avoid the creation of more object of this class
	static void initSingleton() {
		if (!singleton) {
			singleton = memnew(BoneBulletPhysicsDirectBodyState);
		}
	}

	static void destroySingleton() {
		memdelete(singleton);
		singleton = NULL;
	}

	static void singleton_setDeltaTime(real_t p_deltaTime) {
		singleton->deltaTime = p_deltaTime;
	}

	static BoneBulletPhysicsDirectBodyState *get_singleton(BoneBullet *p_body) {
		singleton->bone = p_body;
		return singleton;
	}

public:
	BoneBullet *bone;
	real_t deltaTime;

private:
	BoneBulletPhysicsDirectBodyState() {}

public:
	virtual Vector3 get_total_gravity() const;
	virtual float get_total_angular_damp() const;
	virtual float get_total_linear_damp() const;

	virtual Vector3 get_center_of_mass() const;
	virtual Basis get_principal_inertia_axes() const;
	// get the mass
	virtual float get_inverse_mass() const;
	// get density of this body space
	virtual Vector3 get_inverse_inertia() const;
	// get density of this body space
	virtual Basis get_inverse_inertia_tensor() const;

	virtual void set_linear_velocity(const Vector3 &p_velocity);
	virtual Vector3 get_linear_velocity() const;

	virtual void set_angular_velocity(const Vector3 &p_velocity);
	virtual Vector3 get_angular_velocity() const;

	virtual void set_transform(const Transform &p_transform);
	virtual Transform get_transform() const;

	virtual void add_central_force(const Vector3 &p_force);
	virtual void add_force(const Vector3 &p_force, const Vector3 &p_pos);
	virtual void add_torque(const Vector3 &p_torque);
	virtual void apply_central_impulse(const Vector3 &p_impulse);
	virtual void apply_impulse(const Vector3 &p_pos, const Vector3 &p_j);
	virtual void apply_torque_impulse(const Vector3 &p_j);

	virtual void set_sleep_state(bool p_enable);
	virtual bool is_sleeping() const;

	virtual int get_contact_count() const;

	virtual Vector3 get_contact_local_position(int p_contact_idx) const;
	virtual Vector3 get_contact_local_normal(int p_contact_idx) const;
	virtual float get_contact_impulse(int p_contact_idx) const;
	virtual int get_contact_local_shape(int p_contact_idx) const;

	virtual RID get_contact_collider(int p_contact_idx) const;
	virtual Vector3 get_contact_collider_position(int p_contact_idx) const;
	virtual ObjectID get_contact_collider_id(int p_contact_idx) const;
	virtual int get_contact_collider_shape(int p_contact_idx) const;
	virtual Vector3 get_contact_collider_velocity_at_position(int p_contact_idx) const;

	virtual real_t get_step() const { return deltaTime; }
	virtual void integrate_forces() {
		// Skip the execution of this function
	}

	virtual PhysicsDirectSpaceState *get_space_state();
};

class BoneBullet : public RigidCollisionObjectBullet {

	friend class ArmatureBullet;
	friend class BoneBulletPhysicsDirectBodyState;

	struct ForceIntegrationCallback {
		ObjectID id;
		StringName method;
		Variant udata;
	};

	btMultiBodyLinkCollider *bt_body;
	btMultiBodyConstraint *bt_joint_limiter;
	btMultiBodyConstraint *bt_joint_motor;
	ArmatureBullet *armature;
	ForceIntegrationCallback *force_integration_callback;
	real_t link_mass;
	real_t link_half_size;
	int parent_link_id; // Set -1 if no parent
	btTransform joint_offset;
	bool disable_parent_collision;

	bool is_limit_active;
	real_t lower_limit; // In m if slider, in deg if hinge
	real_t upper_limit; // In m if slider, in deg if hinge

	// Check this http://www.ode.org/ode-latest-userguide.html#sec_3_7_0
	// For information about these parameters
	bool is_motor_enabled;
	btVector3 velocity_target; // Used for 1D and 3D motor
	real_t position_target; // Used for 1D
	btQuaternion rotation_target; // Used for 3D motor
	real_t max_motor_impulse;
	real_t error_reduction_parameter; // From 0 to 1
	real_t spring_constant;
	real_t damping_constant;
	real_t maximum_error;

	bool is_root;

public:
	BoneBullet();
	virtual ~BoneBullet();

	virtual void set_space(SpaceBullet *p_space);
	virtual void on_collision_filters_change();
	virtual void on_collision_checker_end();
	virtual void dispatch_callbacks();
	virtual void on_enter_area(AreaBullet *p_area);

	virtual void main_shape_changed();
	virtual void reload_body();

	virtual void set_transform__bullet(const btTransform &p_global_transform);
	virtual const btTransform &get_transform__bullet() const;

	btMultiBodyLinkCollider *get_bt_body() const;
	btMultiBodyConstraint *get_bt_joint_limiter() const;
	btMultiBodyConstraint *get_bt_joint_motor() const;

	void set_force_integration_callback(
			ObjectID p_id,
			const StringName &p_method,
			const Variant &p_udata = Variant());

	void set_armature(ArmatureBullet *p_armature);
	ArmatureBullet *get_armature() const;

	void set_bone_id(int p_link_id);
	int get_bone_id() const;
	int get_link_id() const;

	void set_parent_bone_id(int p_bone_id);
	int get_parent_bone_id() const;
	int get_parent_link_id() const;

	void set_joint_offset(const Transform &p_transform);
	Transform get_joint_offset() const;
	const btTransform &get_joint_offset__bullet() const;
	btTransform get_joint_offset_scaled__bullet() const;

	void set_link_mass(real_t p_link_mass);
	real_t get_link_mass() const;

	void set_param(PhysicsServer::BodyParameter p_param, real_t p_value);
	real_t get_param(PhysicsServer::BodyParameter p_param) const;

	void set_disable_parent_collision(bool p_disable);
	bool get_disable_parent_collision() const;

	void set_limit_active(bool p_limit_active);
	bool get_is_limit_active() const;

	void set_lower_limit(real_t p_lower_limit);
	real_t get_lower_limit() const;

	void set_upper_limit(real_t p_upper_limit);
	real_t get_upper_limit() const;

	void set_motor_enabled(bool p_v);
	bool get_motor_enabled() const;

	void set_velocity_target(const Vector3 &p_v);
	Vector3 get_velocity_target() const;

	void set_position_target(real_t p_v);
	real_t get_position_target() const;

	void set_rotation_target(const Basis &p_v);
	Basis get_rotation_target() const;

	void set_max_motor_impulse(real_t p_v);
	real_t get_max_motor_impulse() const;

	void set_error_reduction_parameter(real_t p_v);
	real_t get_error_reduction_parameter() const;

	void set_spring_constant(real_t p_v);
	real_t get_spring_constant() const;

	void set_damping_constant(real_t p_v);
	real_t get_damping_constant() const;

	void set_maximum_error(real_t p_v);
	real_t get_maximum_error() const;

	Vector3 get_joint_force() const;
	Vector3 get_joint_torque() const;

	void setup_joint_fixed();
	void setup_joint_prismatic();
	void setup_joint_revolute();
	void setup_joint_spherical();
	void setup_joint_planar();

	void update_joint_limits();
	void update_joint_motor();
	void update_joint_motor_params();

	void calc_link_inertia(
			btMultibodyLink::eFeatherstoneJointType p_joint_type,
			real_t p_link_mass,
			real_t p_link_half_length,
			btVector3 &r_inertia);
};

#endif // ARMATURE_BULLET_H

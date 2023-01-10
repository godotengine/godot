/**************************************************************************/
/*  rigid_body_bullet.h                                                   */
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

#ifndef RIGID_BODY_BULLET_H
#define RIGID_BODY_BULLET_H

#include "collision_object_bullet.h"
#include "space_bullet.h"

#include <BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h>
#include <LinearMath/btTransform.h>

/**
	@author AndreaCatania
*/

class AreaBullet;
class SpaceBullet;
class btRigidBody;
class GodotMotionState;

class BulletPhysicsDirectBodyState : public PhysicsDirectBodyState {
	GDCLASS(BulletPhysicsDirectBodyState, PhysicsDirectBodyState);

public:
	RigidBodyBullet *body = nullptr;

	BulletPhysicsDirectBodyState() {}

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

	virtual Vector3 get_velocity_at_local_position(const Vector3 &p_position) const;

	virtual void add_central_force(const Vector3 &p_force);
	virtual void add_force(const Vector3 &p_force, const Vector3 &p_pos);
	virtual void add_torque(const Vector3 &p_torque);
	virtual void apply_central_impulse(const Vector3 &p_impulse);
	virtual void apply_impulse(const Vector3 &p_pos, const Vector3 &p_impulse);
	virtual void apply_torque_impulse(const Vector3 &p_impulse);

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

	virtual real_t get_step() const;
	virtual void integrate_forces() {
		// Skip the execution of this function
	}

	virtual PhysicsDirectSpaceState *get_space_state();
};

class RigidBodyBullet : public RigidCollisionObjectBullet {
public:
	struct CollisionData {
		RigidBodyBullet *otherObject;
		int other_object_shape;
		int local_shape;
		Vector3 hitLocalLocation;
		Vector3 hitWorldLocation;
		Vector3 hitNormal;
		float appliedImpulse;
	};

	struct ForceIntegrationCallback {
		ObjectID id;
		StringName method;
		Variant udata;
	};

	/// Used to hold shapes
	struct KinematicShape {
		class btConvexShape *shape;
		btTransform transform;

		KinematicShape() :
				shape(nullptr) {}
		bool is_active() const { return shape; }
	};

	struct KinematicUtilities {
		RigidBodyBullet *owner;
		btScalar safe_margin;
		Vector<KinematicShape> shapes;

		KinematicUtilities(RigidBodyBullet *p_owner);
		~KinematicUtilities();

		void setSafeMargin(btScalar p_margin);
		/// Used to set the default shape to ghost
		void copyAllOwnerShapes();

	private:
		void just_delete_shapes(int new_size);
	};

private:
	BulletPhysicsDirectBodyState *direct_access = nullptr;
	friend class BulletPhysicsDirectBodyState;

	// This is required only for Kinematic movement
	KinematicUtilities *kinematic_utilities;

	PhysicsServer::BodyMode mode;
	GodotMotionState *godotMotionState;
	btRigidBody *btBody;
	uint16_t locked_axis;
	real_t mass;
	real_t gravity_scale;
	real_t linearDamp;
	real_t angularDamp;
	Vector3 total_gravity;
	real_t total_linear_damp;
	real_t total_angular_damp;
	bool can_sleep;
	bool omit_forces_integration;
	bool can_integrate_forces;

	Vector<CollisionData> collisions;
	Vector<RigidBodyBullet *> collision_traces_1;
	Vector<RigidBodyBullet *> collision_traces_2;
	Vector<RigidBodyBullet *> *prev_collision_traces;
	Vector<RigidBodyBullet *> *curr_collision_traces;

	// these parameters are used to avoid vector resize
	int maxCollisionsDetection;
	int collisionsCount;
	int prev_collision_count;

	Vector<AreaBullet *> areasWhereIam;
	// these parameters are used to avoid vector resize
	int maxAreasWhereIam;
	int areaWhereIamCount;
	// Used to know if the area is used as gravity point
	int countGravityPointSpaces;
	bool isScratchedSpaceOverrideModificator;

	bool previousActiveState; // Last check state

	ForceIntegrationCallback *force_integration_callback;

public:
	RigidBodyBullet();
	~RigidBodyBullet();

	BulletPhysicsDirectBodyState *get_direct_state() const { return direct_access; }

	void init_kinematic_utilities();
	void destroy_kinematic_utilities();
	_FORCE_INLINE_ KinematicUtilities *get_kinematic_utilities() const { return kinematic_utilities; }

	_FORCE_INLINE_ btRigidBody *get_bt_rigid_body() { return btBody; }

	virtual void main_shape_changed();
	virtual void reload_body();
	virtual void set_space(SpaceBullet *p_space);

	virtual void dispatch_callbacks();
	void set_force_integration_callback(ObjectID p_id, const StringName &p_method, const Variant &p_udata = Variant());
	void scratch_space_override_modificator();

	virtual void on_collision_filters_change();
	virtual void on_collision_checker_start();
	virtual void on_collision_checker_end();

	void set_max_collisions_detection(int p_maxCollisionsDetection) {
		ERR_FAIL_COND(0 > p_maxCollisionsDetection);

		maxCollisionsDetection = p_maxCollisionsDetection;

		collisions.resize(p_maxCollisionsDetection);
		collision_traces_1.resize(p_maxCollisionsDetection);
		collision_traces_2.resize(p_maxCollisionsDetection);

		collisionsCount = 0;
		prev_collision_count = MIN(prev_collision_count, p_maxCollisionsDetection);
	}
	int get_max_collisions_detection() {
		return maxCollisionsDetection;
	}

	bool can_add_collision() { return collisionsCount < maxCollisionsDetection; }
	bool add_collision_object(RigidBodyBullet *p_otherObject, const Vector3 &p_hitWorldLocation, const Vector3 &p_hitLocalLocation, const Vector3 &p_hitNormal, const float &p_appliedImpulse, int p_other_shape_index, int p_local_shape_index);
	bool was_colliding(RigidBodyBullet *p_other_object);

	void set_activation_state(bool p_active);
	bool is_active() const;

	void set_omit_forces_integration(bool p_omit);
	_FORCE_INLINE_ bool get_omit_forces_integration() const { return omit_forces_integration; }

	void set_param(PhysicsServer::BodyParameter p_param, real_t);
	real_t get_param(PhysicsServer::BodyParameter p_param) const;

	void set_mode(PhysicsServer::BodyMode p_mode);
	PhysicsServer::BodyMode get_mode() const;

	void set_state(PhysicsServer::BodyState p_state, const Variant &p_variant);
	Variant get_state(PhysicsServer::BodyState p_state) const;

	void apply_impulse(const Vector3 &p_pos, const Vector3 &p_impulse);
	void apply_central_impulse(const Vector3 &p_impulse);
	void apply_torque_impulse(const Vector3 &p_impulse);

	void apply_force(const Vector3 &p_force, const Vector3 &p_pos);
	void apply_central_force(const Vector3 &p_force);
	void apply_torque(const Vector3 &p_torque);

	void set_applied_force(const Vector3 &p_force);
	Vector3 get_applied_force() const;
	void set_applied_torque(const Vector3 &p_torque);
	Vector3 get_applied_torque() const;

	void set_axis_lock(PhysicsServer::BodyAxis p_axis, bool lock);
	bool is_axis_locked(PhysicsServer::BodyAxis p_axis) const;
	void reload_axis_lock();

	/// Doc:
	/// https://web.archive.org/web/20180404091446/http://www.bulletphysics.org/mediawiki-1.5.8/index.php/Anti_tunneling_by_Motion_Clamping
	void set_continuous_collision_detection(bool p_enable);
	bool is_continuous_collision_detection_enabled() const;

	void set_linear_velocity(const Vector3 &p_velocity);
	Vector3 get_linear_velocity() const;

	void set_angular_velocity(const Vector3 &p_velocity);
	Vector3 get_angular_velocity() const;

	virtual void set_transform__bullet(const btTransform &p_global_transform);
	virtual const btTransform &get_transform__bullet() const;

	virtual void reload_shapes();

	virtual void on_enter_area(AreaBullet *p_area);
	virtual void on_exit_area(AreaBullet *p_area);
	void reload_space_override_modificator();

	/// Kinematic
	void reload_kinematic_shapes();

	virtual void notify_transform_changed();

private:
	void _internal_set_mass(real_t p_mass);
};

#endif // RIGID_BODY_BULLET_H

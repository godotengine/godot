/*************************************************************************/
/*  body_bullet.h                                                        */
/*  Author: AndreaCatania                                                */
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

#ifndef BODYBULLET_H
#define BODYBULLET_H

#include "BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h"
#include "LinearMath/btTransform.h"
#include "collision_object_bullet.h"
#include "space_bullet.h"

class AreaBullet;
class SpaceBullet;
class btRigidBody;
class GodotMotionState;
class BulletPhysicsDirectBodyState;

/// This class could be used in multi thread with few changes but currently
/// is setted to be only in one single thread.
///
/// In the system there is only one object at a time that manage all bodies and is
/// created by BulletPhysicsServer and is held by the "singleton" variable of this class
/// Each time something require it, the body must be setted again.
class BulletPhysicsDirectBodyState : public PhysicsDirectBodyState {
	GDCLASS(BulletPhysicsDirectBodyState, PhysicsDirectBodyState)

	static BulletPhysicsDirectBodyState *singleton;

public:
	/// This class avoid the creation of more object of this class
	static void initSingleton() {
		if (!singleton) {
			singleton = memnew(BulletPhysicsDirectBodyState);
		}
	}

	static void destroySingleton() {
		memdelete(singleton);
		singleton = NULL;
	}

	static void singleton_setDeltaTime(real_t p_deltaTime) {
		singleton->deltaTime = p_deltaTime;
	}

	static BulletPhysicsDirectBodyState *get_singleton(RigidBodyBullet *p_body) {
		singleton->body = p_body;
		return singleton;
	}

public:
	RigidBodyBullet *body;
	real_t deltaTime;

private:
	BulletPhysicsDirectBodyState() {}

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

	virtual void add_force(const Vector3 &p_force, const Vector3 &p_pos);
	virtual void apply_impulse(const Vector3 &p_pos, const Vector3 &p_j);
	virtual void apply_torque_impulse(const Vector3 &p_j);

	virtual void set_sleep_state(bool p_enable);
	virtual bool is_sleeping() const;

	virtual int get_contact_count() const;

	virtual Vector3 get_contact_local_position(int p_contact_idx) const;
	virtual Vector3 get_contact_local_normal(int p_contact_idx) const;
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

class RigidBodyBullet : public RigidCollisionObjectBullet {

public:
	struct CollisionData {
		RigidBodyBullet *otherObject;
		int other_object_shape;
		int local_shape;
		Vector3 hitLocalLocation;
		Vector3 hitWorldLocation;
		Vector3 hitNormal;
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
				shape(NULL) {}
		const bool is_active() const { return shape; }
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
	bool can_sleep;

	Vector<CollisionData> collisions;
	// these parameters are used to avoid vector resize
	int maxCollisionsDetection;
	int collisionsCount;

	Vector<AreaBullet *> areasWhereIam;
	// these parameters are used to avoid vector resize
	int maxAreasWhereIam;
	int areaWhereIamCount;
	// Used to know if the area is used as gravity point
	int countGravityPointSpaces;
	bool isScratchedSpaceOverrideModificator;

	bool isTransformChanged;
	bool previousActiveState; // Last check state

	ForceIntegrationCallback *force_integration_callback;

public:
	RigidBodyBullet();
	~RigidBodyBullet();

	void init_kinematic_utilities();
	void destroy_kinematic_utilities();
	_FORCE_INLINE_ class KinematicUtilities *get_kinematic_utilities() const { return kinematic_utilities; }

	_FORCE_INLINE_ btRigidBody *get_bt_rigid_body() { return btBody; }

	virtual void reload_body();
	virtual void set_space(SpaceBullet *p_space);

	virtual void dispatch_callbacks();
	void set_force_integration_callback(ObjectID p_id, const StringName &p_method, const Variant &p_udata = Variant());
	void scratch();
	void scratch_space_override_modificator();

	virtual void on_collision_filters_change();
	virtual void on_collision_checker_start();
	void set_max_collisions_detection(int p_maxCollisionsDetection) {
		maxCollisionsDetection = p_maxCollisionsDetection;
		collisions.resize(p_maxCollisionsDetection);
		collisionsCount = 0;
	}
	int get_max_collisions_detection() {
		return maxCollisionsDetection;
	}

	bool can_add_collision() { return collisionsCount < maxCollisionsDetection; }
	bool add_collision_object(RigidBodyBullet *p_otherObject, const Vector3 &p_hitWorldLocation, const Vector3 &p_hitLocalLocation, const Vector3 &p_hitNormal, int p_other_shape_index, int p_local_shape_index);

	void assert_no_constraints();

	void set_activation_state(bool p_active);
	bool is_active() const;

	void set_param(PhysicsServer::BodyParameter p_param, real_t);
	real_t get_param(PhysicsServer::BodyParameter p_param) const;

	void set_mode(PhysicsServer::BodyMode p_mode);
	PhysicsServer::BodyMode get_mode() const;

	void set_state(PhysicsServer::BodyState p_state, const Variant &p_variant);
	Variant get_state(PhysicsServer::BodyState p_state) const;

	void apply_impulse(const Vector3 &p_pos, const Vector3 &p_impulse);
	void apply_central_impulse(const Vector3 &p_force);
	void apply_torque_impulse(const Vector3 &p_impulse);

	void apply_force(const Vector3 &p_force, const Vector3 &p_pos);
	void apply_central_force(const Vector3 &p_force);
	void apply_torque(const Vector3 &p_force);

	void set_applied_force(const Vector3 &p_force);
	Vector3 get_applied_force() const;
	void set_applied_torque(const Vector3 &p_torque);
	Vector3 get_applied_torque() const;

	void set_axis_lock(PhysicsServer::BodyAxis p_axis, bool lock);
	bool is_axis_locked(PhysicsServer::BodyAxis p_axis) const;
	void reload_axis_lock();

	/// Doc:
	/// http://www.bulletphysics.org/mediawiki-1.5.8/index.php?title=Anti_tunneling_by_Motion_Clamping
	void set_continuous_collision_detection(bool p_enable);
	bool is_continuous_collision_detection_enabled() const;

	void set_linear_velocity(const Vector3 &p_velocity);
	Vector3 get_linear_velocity() const;

	void set_angular_velocity(const Vector3 &p_velocity);
	Vector3 get_angular_velocity() const;

	virtual void set_transform__bullet(const btTransform &p_global_transform);
	virtual const btTransform &get_transform__bullet() const;

	virtual void on_shapes_changed();

	virtual void on_enter_area(AreaBullet *p_area);
	virtual void on_exit_area(AreaBullet *p_area);
	void reload_space_override_modificator();

	/// Kinematic
	void reload_kinematic_shapes();

private:
	void _internal_set_mass(real_t p_mass);
};

#endif

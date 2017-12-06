/*************************************************************************/
/*  space_bullet.h                                                       */
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

#ifndef SPACE_BULLET_H
#define SPACE_BULLET_H

#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"
#include "BulletCollision/BroadphaseCollision/btOverlappingPairCache.h"
#include "LinearMath/btScalar.h"
#include "LinearMath/btTransform.h"
#include "LinearMath/btVector3.h"
#include "core/variant.h"
#include "core/vector.h"
#include "godot_result_callbacks.h"
#include "rid_bullet.h"
#include "servers/physics_server.h"

class AreaBullet;
class btBroadphaseInterface;
class btCollisionDispatcher;
class btConstraintSolver;
class btDefaultCollisionConfiguration;
class btDynamicsWorld;
class btDiscreteDynamicsWorld;
class btEmptyShape;
class btGhostPairCallback;
class btSoftRigidDynamicsWorld;
class btSoftBodyWorldInfo;
class ConstraintBullet;
class CollisionObjectBullet;
class RigidBodyBullet;
class SpaceBullet;
class SoftBodyBullet;
class btGjkEpaPenetrationDepthSolver;

class BulletPhysicsDirectSpaceState : public PhysicsDirectSpaceState {
	GDCLASS(BulletPhysicsDirectSpaceState, PhysicsDirectSpaceState)
private:
	SpaceBullet *space;

public:
	BulletPhysicsDirectSpaceState(SpaceBullet *p_space);

	virtual int intersect_point(const Vector3 &p_point, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF);
	virtual bool intersect_ray(const Vector3 &p_from, const Vector3 &p_to, RayResult &r_result, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF, bool p_pick_ray = false);
	virtual int intersect_shape(const RID &p_shape, const Transform &p_xform, float p_margin, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF);
	virtual bool cast_motion(const RID &p_shape, const Transform &p_xform, const Vector3 &p_motion, float p_margin, float &p_closest_safe, float &p_closest_unsafe, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF, ShapeRestInfo *r_info = NULL);
	/// Returns the list of contacts pairs in this order: Local contact, other body contact
	virtual bool collide_shape(RID p_shape, const Transform &p_shape_xform, float p_margin, Vector3 *r_results, int p_result_max, int &r_result_count, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF);
	virtual bool rest_info(RID p_shape, const Transform &p_shape_xform, float p_margin, ShapeRestInfo *r_info, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_mask = 0xFFFFFFFF);
	virtual Vector3 get_closest_point_to_object_volume(RID p_object, const Vector3 p_point) const;
};

class SpaceBullet : public RIDBullet {
private:
	friend class AreaBullet;
	friend void onBulletTickCallback(btDynamicsWorld *world, btScalar timeStep);
	friend class BulletPhysicsDirectSpaceState;

	btBroadphaseInterface *broadphase;
	btDefaultCollisionConfiguration *collisionConfiguration;
	btCollisionDispatcher *dispatcher;
	btConstraintSolver *solver;
	btDiscreteDynamicsWorld *dynamicsWorld;
	btGhostPairCallback *ghostPairCallback;
	GodotFilterCallback *godotFilterCallback;
	btSoftBodyWorldInfo *soft_body_world_info;

	btGjkEpaPenetrationDepthSolver *gjk_epa_pen_solver;
	btVoronoiSimplexSolver *gjk_simplex_solver;

	BulletPhysicsDirectSpaceState *direct_access;
	Vector3 gravityDirection;
	real_t gravityMagnitude;

	Vector<AreaBullet *> areas;

	Vector<Vector3> contactDebug;
	int contactDebugCount;

public:
	SpaceBullet(bool p_create_soft_world);
	virtual ~SpaceBullet();

	void flush_queries();
	void step(real_t p_delta_time);

	_FORCE_INLINE_ btBroadphaseInterface *get_broadphase() { return broadphase; }
	_FORCE_INLINE_ btCollisionDispatcher *get_dispatcher() { return dispatcher; }
	_FORCE_INLINE_ btSoftBodyWorldInfo *get_soft_body_world_info() { return soft_body_world_info; }
	_FORCE_INLINE_ bool is_using_soft_world() { return soft_body_world_info; }

	/// Used to set some parameters to Bullet world
	/// @param p_param:
	///     AREA_PARAM_GRAVITY          to set the gravity magnitude of entire world
	///     AREA_PARAM_GRAVITY_VECTOR   to set the gravity direction of entire world
	void set_param(PhysicsServer::AreaParameter p_param, const Variant &p_value);
	/// Used to get some parameters to Bullet world
	/// @param p_param:
	///     AREA_PARAM_GRAVITY          to get the gravity magnitude of entire world
	///     AREA_PARAM_GRAVITY_VECTOR   to get the gravity direction of entire world
	Variant get_param(PhysicsServer::AreaParameter p_param);

	void set_param(PhysicsServer::SpaceParameter p_param, real_t p_value);
	real_t get_param(PhysicsServer::SpaceParameter p_param);

	void add_area(AreaBullet *p_area);
	void remove_area(AreaBullet *p_area);
	void reload_collision_filters(AreaBullet *p_area);

	void add_rigid_body(RigidBodyBullet *p_body);
	void remove_rigid_body(RigidBodyBullet *p_body);
	void reload_collision_filters(RigidBodyBullet *p_body);

	void add_soft_body(SoftBodyBullet *p_body);
	void remove_soft_body(SoftBodyBullet *p_body);
	void reload_collision_filters(SoftBodyBullet *p_body);

	void add_constraint(ConstraintBullet *p_constraint, bool disableCollisionsBetweenLinkedBodies = false);
	void remove_constraint(ConstraintBullet *p_constraint);

	int get_num_collision_objects() const;
	void remove_all_collision_objects();

	BulletPhysicsDirectSpaceState *get_direct_state();

	void set_debug_contacts(int p_amount) { contactDebug.resize(p_amount); }
	_FORCE_INLINE_ bool is_debugging_contacts() const { return !contactDebug.empty(); }
	_FORCE_INLINE_ void reset_debug_contact_count() {
		contactDebugCount = 0;
	}
	_FORCE_INLINE_ void add_debug_contact(const Vector3 &p_contact) {
		if (contactDebugCount < contactDebug.size()) contactDebug[contactDebugCount++] = p_contact;
	}
	_FORCE_INLINE_ Vector<Vector3> get_debug_contacts() { return contactDebug; }
	_FORCE_INLINE_ int get_debug_contact_count() { return contactDebugCount; }

	const Vector3 &get_gravity_direction() const { return gravityDirection; }
	real_t get_gravity_magnitude() const { return gravityMagnitude; }

	void update_gravity();

	bool test_body_motion(RigidBodyBullet *p_body, const Transform &p_from, const Vector3 &p_motion, PhysicsServer::MotionResult *r_result);

private:
	void create_empty_world(bool p_create_soft_world);
	void destroy_world();
	void check_ghost_overlaps();
	void check_body_collision();

	struct RecoverResult {
		bool hasPenetration;
		btVector3 pointNormalWorld;
		btVector3 pointWorld;
		btScalar penetration_distance; // Negative is penetration
		int other_compound_shape_index;
		const btCollisionObject *other_collision_object;
		int local_shape_most_recovered;

		RecoverResult() :
				hasPenetration(false) {}
	};

	bool recover_from_penetration(RigidBodyBullet *p_body, const btTransform &p_from, btVector3 &r_recover_position, RecoverResult *r_recover_result = NULL);
	/// This is an API that recover a kinematic object from penetration
	/// This allow only Convex Convex test and it always use GJK algorithm, With this API we don't benefit of Bullet special accelerated functions
	bool RFP_convex_convex_test(const btConvexShape *p_shapeA, const btConvexShape *p_shapeB, btCollisionObject *p_objectB, int p_shapeId_B, const btTransform &p_transformA, const btTransform &p_transformB, btVector3 &r_recover_position, RecoverResult *r_recover_result);
	/// This is an API that recover a kinematic object from penetration
	/// Using this we leave Bullet to select the best algorithm, For example GJK in case we have Convex Convex, or a Bullet accelerated algorithm
	bool RFP_convex_world_test(const btConvexShape *p_shapeA, const btCollisionShape *p_shapeB, btCollisionObject *p_objectA, btCollisionObject *p_objectB, int p_shapeId_A, int p_shapeId_B, const btTransform &p_transformA, const btTransform &p_transformB, btVector3 &r_recover_position, RecoverResult *r_recover_result);
};
#endif

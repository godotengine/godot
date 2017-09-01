/*************************************************************************/
/*  body_bullet.cpp                                                      */
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

#include "body_bullet.h"
#include "BulletCollision/CollisionDispatch/btGhostObject.h"
#include "BulletCollision/CollisionShapes/btConvexPointCloudShape.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"
#include "btBulletCollisionCommon.h"
#include "btRayShape.h"
#include "bullet_physics_server.h"
#include "bullet_types_converter.h"
#include "bullet_utilities.h"
#include "godot_motion_state.h"
#include "joint_bullet.h"
#include <assert.h>

BulletPhysicsDirectBodyState *BulletPhysicsDirectBodyState::singleton = NULL;

Vector3 BulletPhysicsDirectBodyState::get_total_gravity() const {
	Vector3 gVec;
	B_TO_G(body->btBody->getGravity(), gVec);
	return gVec;
}

float BulletPhysicsDirectBodyState::get_total_angular_damp() const {
	return body->btBody->getAngularDamping();
}

float BulletPhysicsDirectBodyState::get_total_linear_damp() const {
	return body->btBody->getLinearDamping();
}

Vector3 BulletPhysicsDirectBodyState::get_center_of_mass() const {
	Vector3 gVec;
	B_TO_G(body->btBody->getCenterOfMassPosition(), gVec);
	return gVec;
}

Basis BulletPhysicsDirectBodyState::get_principal_inertia_axes() const {
	return Basis();
}

float BulletPhysicsDirectBodyState::get_inverse_mass() const {
	return body->btBody->getInvMass();
}

Vector3 BulletPhysicsDirectBodyState::get_inverse_inertia() const {
	Vector3 gVec;
	B_TO_G(body->btBody->getInvInertiaDiagLocal(), gVec);
	return gVec;
}

Basis BulletPhysicsDirectBodyState::get_inverse_inertia_tensor() const {
	Basis gInertia;
	B_TO_G(body->btBody->getInvInertiaTensorWorld(), gInertia);
	return gInertia;
}

void BulletPhysicsDirectBodyState::set_linear_velocity(const Vector3 &p_velocity) {
	body->set_linear_velocity(p_velocity);
}

Vector3 BulletPhysicsDirectBodyState::get_linear_velocity() const {
	return body->get_linear_velocity();
}

void BulletPhysicsDirectBodyState::set_angular_velocity(const Vector3 &p_velocity) {
	body->set_angular_velocity(p_velocity);
}

Vector3 BulletPhysicsDirectBodyState::get_angular_velocity() const {
	return body->get_angular_velocity();
}

void BulletPhysicsDirectBodyState::set_transform(const Transform &p_transform) {
	body->set_transform(p_transform);
}

Transform BulletPhysicsDirectBodyState::get_transform() const {
	return body->get_transform();
}

void BulletPhysicsDirectBodyState::add_force(const Vector3 &p_force, const Vector3 &p_pos) {
	body->apply_force(p_force, p_pos);
}

void BulletPhysicsDirectBodyState::apply_impulse(const Vector3 &p_pos, const Vector3 &p_j) {
	body->apply_impulse(p_pos, p_j);
}

void BulletPhysicsDirectBodyState::apply_torque_impulse(const Vector3 &p_j) {
	body->apply_torque_impulse(p_j);
}

void BulletPhysicsDirectBodyState::set_sleep_state(bool p_enable) {
	body->set_active(p_enable);
}

bool BulletPhysicsDirectBodyState::is_sleeping() const {
	return !body->is_active();
}

int BulletPhysicsDirectBodyState::get_contact_count() const {
	return body->collisionsCount;
}

Vector3 BulletPhysicsDirectBodyState::get_contact_local_pos(int p_contact_idx) const {
	return body->collisions[p_contact_idx].hitLocalLocation;
}

Vector3 BulletPhysicsDirectBodyState::get_contact_local_normal(int p_contact_idx) const {
	return body->collisions[p_contact_idx].hitNormal;
}

int BulletPhysicsDirectBodyState::get_contact_local_shape(int p_contact_idx) const {
	return body->collisions[p_contact_idx].local_shape;
}

RID BulletPhysicsDirectBodyState::get_contact_collider(int p_contact_idx) const {
	return body->collisions[p_contact_idx].otherObject->get_self();
}

Vector3 BulletPhysicsDirectBodyState::get_contact_collider_pos(int p_contact_idx) const {
	return body->collisions[p_contact_idx].hitWorldLocation;
}

ObjectID BulletPhysicsDirectBodyState::get_contact_collider_id(int p_contact_idx) const {
	return body->collisions[p_contact_idx].otherObject->get_instance_id();
}

int BulletPhysicsDirectBodyState::get_contact_collider_shape(int p_contact_idx) const {
	return body->collisions[p_contact_idx].other_object_shape;
}

Vector3 BulletPhysicsDirectBodyState::get_contact_collider_velocity_at_pos(int p_contact_idx) const {
	BodyBullet::CollisionData &colDat = body->collisions[p_contact_idx];
	btVector3 hitLocation;
	G_TO_B(colDat.hitLocalLocation, hitLocation);
	Vector3 velocityAtPoint;
	B_TO_G(body->collisions[p_contact_idx].otherObject->get_bt_body()->getVelocityInLocalPoint(hitLocation), velocityAtPoint);
	return velocityAtPoint;
}

PhysicsDirectSpaceState *BulletPhysicsDirectBodyState::get_space_state() {
	return body->get_space()->get_direct_state();
}

BodyBullet::KinematicUtilities::KinematicUtilities(BodyBullet *p_owner)
	: m_owner(p_owner), m_margin(0.01) // Godot default margin 0.001
{
	m_ghostObject = bulletnew(btPairCachingGhostObject);

	int clearedCurrentFlags = m_ghostObject->getCollisionFlags();
	clearedCurrentFlags &= ~(btCollisionObject::CF_KINEMATIC_OBJECT | btCollisionObject::CF_STATIC_OBJECT);

	m_ghostObject->setCollisionFlags(clearedCurrentFlags | btCollisionObject::CF_KINEMATIC_OBJECT);
	m_ghostObject->setUserPointer(p_owner);
	m_ghostObject->setUserIndex(static_cast<int>(SpaceBullet::QUERY_TYPE_SKIP));

	resetDefShape();
}

BodyBullet::KinematicUtilities::~KinematicUtilities() {
	just_delete_shapes(m_shapes.size()); // don't need to resize
	bulletdelete(m_ghostObject);
}

void BodyBullet::KinematicUtilities::resetDefShape() {
	m_ghostObject->setCollisionShape(BulletPhysicsServer::get_empty_shape());
}

void BodyBullet::KinematicUtilities::copyAllOwnerShapes() {
	const Vector<CollisionObjectBullet::ShapeWrapper> &shapes_wrappers(m_owner->get_shapes_wrappers());
	const int shapes_count = shapes_wrappers.size();

	just_delete_shapes(shapes_count);

	const CollisionObjectBullet::ShapeWrapper *shape_wrapper;

	for (int i = shapes_count - 1; 0 <= i; --i) {
		shape_wrapper = &shapes_wrappers[i];
		if (!shape_wrapper->active) {
			continue;
		}
		m_shapes[i].transform = shape_wrapper->transform;

		btConvexShape *&kin_shape_ref = m_shapes[i].shape;

		switch (shape_wrapper->shape->get_type()) {
			case PhysicsServer::SHAPE_SPHERE: {
				SphereShapeBullet *sphere = static_cast<SphereShapeBullet *>(shape_wrapper->shape);
				kin_shape_ref = ShapeBullet::create_shape_sphere(sphere->get_radius() * m_owner->body_scale + m_margin);
				break;
			}
			case PhysicsServer::SHAPE_BOX: {
				BoxShapeBullet *box = static_cast<BoxShapeBullet *>(shape_wrapper->shape);
				kin_shape_ref = ShapeBullet::create_shape_box((box->get_half_extents() * m_owner->body_scale) + btVector3(m_margin, m_margin, m_margin));
				break;
			}
			case PhysicsServer::SHAPE_CAPSULE: {
				CapsuleShapeBullet *capsule = static_cast<CapsuleShapeBullet *>(shape_wrapper->shape);
				kin_shape_ref = ShapeBullet::create_shape_capsule(capsule->get_radius() * m_owner->body_scale + m_margin, capsule->get_height() * m_owner->body_scale + m_margin);
				break;
			}
			case PhysicsServer::SHAPE_CONVEX_POLYGON: {
				ConvexPolygonShapeBullet *godot_convex = static_cast<ConvexPolygonShapeBullet *>(shape_wrapper->shape);
				kin_shape_ref = ShapeBullet::create_shape_convex(godot_convex->vertices);
				kin_shape_ref->setLocalScaling(btVector3(m_owner->body_scale + m_margin, m_owner->body_scale + m_margin, m_owner->body_scale + m_margin));
				break;
			}
			case PhysicsServer::SHAPE_RAY: {
				RayShapeBullet *godot_ray = static_cast<RayShapeBullet *>(shape_wrapper->shape);
				kin_shape_ref = ShapeBullet::create_shape_ray(godot_ray->length * m_owner->body_scale + m_margin);
				break;
			}
			case PhysicsServer::SHAPE_HEIGHTMAP:
			case PhysicsServer::SHAPE_CONCAVE_POLYGON:
			case PhysicsServer::SHAPE_PLANE:
				WARN_PRINT("This shape is not supported to be kinematic!");
				kin_shape_ref = NULL;
		}
	}
}

void BodyBullet::KinematicUtilities::just_delete_shapes(int new_size) {
	for (int i = m_shapes.size() - 1; 0 <= i; --i) {
		if (m_shapes[i].shape) {
			bulletdelete(m_shapes[i].shape);
		}
	}
	m_shapes.resize(new_size);
}

BodyBullet::BodyBullet()
	: CollisionObjectBullet(CollisionObjectBullet::TYPE_BODY), kinematic_utilities(NULL), space(NULL), gravity_scale(1), linearDamp(0), angularDamp(0), onStateChange_callback(NULL), isScratched(false), maxCollisionsDetection(0), collisionsCount(0), maxAreasWhereIam(10), areaWhereIamCount(0), countGravityPointSpaces(0), isScratchedSpaceOverrideModificator(false) {
	godotMotionState = bulletnew(GodotMotionState(this));

	// Initial properties
	const btScalar mass = 0;
	const btVector3 localInertia(0, 0, 0);
	btRigidBody::btRigidBodyConstructionInfo cInfo(mass, godotMotionState, compoundShape, localInertia);

	btBody = bulletnew(btRigidBody(cInfo));
	btBody->setUserIndex2(CollisionObjectBullet::TYPE_BODY);
	setupCollisionObject(btBody);

	areasWhereIam.resize(maxAreasWhereIam);
	for (int i = areasWhereIam.size() - 1; 0 <= i; --i) {
		areasWhereIam[i] = NULL;
	}
}

BodyBullet::~BodyBullet() {
	bulletdelete(godotMotionState);

	if (onStateChange_callback)
		memdelete(onStateChange_callback);

	destroy_kinematic_utilities();
}

void BodyBullet::init_kinematic_utilities() {
	kinematic_utilities = memnew(KinematicUtilities(this));
	if (get_space()) {
		get_space()->add_ghost(this);
	}
}

void BodyBullet::destroy_kinematic_utilities() {
	if (kinematic_utilities) {
		if (get_space()) {
			get_space()->remove_ghost(this);
		}
		memdelete(kinematic_utilities);
		kinematic_utilities = NULL;
	}
}

void BodyBullet::reload_body() {
	if (space) {
		space->remove_body(this);
		space->add_body(this);
	}
}

void BodyBullet::set_space(SpaceBullet *p_space) {
	// Clear the old space if there is one
	if (space) {
		isScratched = false;

		// Remove all eventual constraints
		assert_no_constraints();

		// Remove this object form the physics world
		space->remove_body(this);
	}

	space = p_space;

	if (space) {
		space->add_body(this);
	}
}

void BodyBullet::dispatch_callbacks() {
	if (isScratched) {
		isScratched = false;

		if (onStateChange_callback) {
			BulletPhysicsDirectBodyState *bodyDirect = BulletPhysicsDirectBodyState::getSingleton(this);

			Variant variantBodyDirect = bodyDirect;

			Object *obj = ObjectDB::get_instance(onStateChange_callback->id);
			if (!obj) {
				// Remove integration callback
				set_on_state_change(0, StringName());
			} else {
				const Variant *vp[2] = { &variantBodyDirect, &onStateChange_callback->udata };

				Variant::CallError responseCallError;
				int argc = (onStateChange_callback->udata.get_type() == Variant::NIL) ? 1 : 2;
				obj->call(onStateChange_callback->method, vp, argc, responseCallError);
			}
		}
	}

	if (isScratchedSpaceOverrideModificator || 0 < countGravityPointSpaces) {
		isScratchedSpaceOverrideModificator = false;
		reload_space_override_modificator();
	}
}

void BodyBullet::set_on_state_change(ObjectID p_id, const StringName &p_method, const Variant &p_udata) {

	if (onStateChange_callback) {
		memdelete(onStateChange_callback);
		onStateChange_callback = NULL;
	}

	if (p_id != 0) {
		onStateChange_callback = memnew(StateChangeCallback);
		onStateChange_callback->id = p_id;
		onStateChange_callback->method = p_method;
		onStateChange_callback->udata = p_udata;
	}
}

void BodyBullet::scratch() {
	isScratched = true;
}

void BodyBullet::scratch_space_override_modificator() {
	isScratchedSpaceOverrideModificator = true;
}

void BodyBullet::on_collision_filters_change() {
	if (space) {
		space->reload_collision_filters(this);
	}
}

void BodyBullet::on_collision_checker_start() {
	collisionsCount = 0;
}

bool BodyBullet::add_collision_object(BodyBullet *p_otherObject, const Vector3 &p_hitWorldLocation, const Vector3 &p_hitLocalLocation, const Vector3 &p_hitNormal, int p_other_shape_index, int p_local_shape_index) {

	if (collisionsCount >= maxCollisionsDetection) {
		return false;
	}

	CollisionData &cd = collisions[collisionsCount];
	cd.hitLocalLocation = p_hitLocalLocation;
	cd.otherObject = p_otherObject;
	cd.hitWorldLocation = p_hitWorldLocation;
	cd.hitNormal = p_hitNormal;
	cd.other_object_shape = p_other_shape_index;
	cd.local_shape = p_local_shape_index;

	++collisionsCount;
	return true;
}

void BodyBullet::assert_no_constraints() {
	if (btBody->getNumConstraintRefs()) {
		WARN_PRINT("A body with a joints is destroyed. Please check the implementation in order to destroy the joint before the body.");
	}
	/*for(int i = btBody->getNumConstraintRefs()-1; 0<=i; --i){
        btTypedConstraint* btConst = btBody->getConstraintRef(i);
        JointBullet* joint = static_cast<JointBullet*>( btConst->getUserConstraintPtr() );
        space->removeConstraint(joint);
    }*/
}

void BodyBullet::set_active(bool p_active) {
	btBody->activate(p_active);
}

bool BodyBullet::is_active() const {
	return btBody->isActive();
}

void BodyBullet::set_param(PhysicsServer::BodyParameter p_param, real_t p_value) {
	switch (p_param) {
		case PhysicsServer::BODY_PARAM_BOUNCE:
			btBody->setRestitution(p_value);
			break;
		case PhysicsServer::BODY_PARAM_FRICTION:
			btBody->setFriction(p_value);
			break;
		case PhysicsServer::BODY_PARAM_MASS: {
			ERR_FAIL_COND(p_value < 0);

			if (space)
				space->remove_body(this);

			// Rigidbody is dynamic if and only if mass is non Zero, otherwise static
			const bool isDynamic = p_value != 0.f;
			m_isStatic = !isDynamic;

			btVector3 localInertia(0, 0, 0);
			if (isDynamic)
				compoundShape->calculateLocalInertia(p_value, localInertia);

			btBody->setMassProps(p_value, localInertia);
			btBody->updateInertiaTensor();

			int clearedCurrentFlags = btBody->getCollisionFlags();
			clearedCurrentFlags &= ~(btCollisionObject::CF_KINEMATIC_OBJECT | btCollisionObject::CF_STATIC_OBJECT);

			if (isDynamic) {
				mode = PhysicsServer::BODY_MODE_RIGID;
				btBody->setCollisionFlags(clearedCurrentFlags); // Just set the flags without Kin and Static
				btBody->forceActivationState(ACTIVE_TAG); // ACTIVE_TAG 1
			} else {
				if (PhysicsServer::BODY_MODE_KINEMATIC == mode) {
					btBody->setCollisionFlags(clearedCurrentFlags | btCollisionObject::CF_KINEMATIC_OBJECT);
				} else {
					mode = PhysicsServer::BODY_MODE_STATIC;
					btBody->setCollisionFlags(clearedCurrentFlags | btCollisionObject::CF_STATIC_OBJECT);
				}
				btBody->forceActivationState(DISABLE_SIMULATION); // DISABLE_SIMULATION 5
			}

			if (space)
				space->add_body(this);

			break;
		}
		case PhysicsServer::BODY_PARAM_LINEAR_DAMP:
			linearDamp = p_value;
			btBody->setDamping(linearDamp, angularDamp);
			break;
		case PhysicsServer::BODY_PARAM_ANGULAR_DAMP:
			angularDamp = p_value;
			btBody->setDamping(linearDamp, angularDamp);
			break;
		case PhysicsServer::BODY_PARAM_GRAVITY_SCALE:
			gravity_scale = p_value;
			/// The Bullet gravity will be is set by reload_space_override_modificator
			scratch_space_override_modificator();
			break;
		default:
			WARN_PRINTS("Parameter " + itos(p_param) + " not supported by bullet. Value: " + itos(p_value));
	}
}

real_t BodyBullet::get_param(PhysicsServer::BodyParameter p_param) const {
	switch (p_param) {
		case PhysicsServer::BODY_PARAM_BOUNCE:
			return btBody->getRestitution();
		case PhysicsServer::BODY_PARAM_FRICTION:
			return btBody->getFriction();
		case PhysicsServer::BODY_PARAM_MASS: {
			const btScalar invMass = btBody->getInvMass();
			return 0 == invMass ? 0 : 1 / invMass;
		}
		case PhysicsServer::BODY_PARAM_LINEAR_DAMP:
			return linearDamp;
		case PhysicsServer::BODY_PARAM_ANGULAR_DAMP:
			return angularDamp;
		case PhysicsServer::BODY_PARAM_GRAVITY_SCALE:
			return gravity_scale;
		default:
			WARN_PRINTS("Parameter " + itos(p_param) + " not supported by bullet");
			return 0;
	}
}

void BodyBullet::set_mode(PhysicsServer::BodyMode p_mode) {
	destroy_kinematic_utilities();
	// The mode change is relevant to its mass
	switch (p_mode) {
		case PhysicsServer::BODY_MODE_KINEMATIC:
			// This allow me to set KINEMATIC
			mode = PhysicsServer::BODY_MODE_KINEMATIC;
			set_param(PhysicsServer::BODY_PARAM_MASS, 0);
			init_kinematic_utilities();
			break;
		case PhysicsServer::BODY_MODE_STATIC:
			mode = PhysicsServer::BODY_MODE_STATIC;
			set_param(PhysicsServer::BODY_PARAM_MASS, 0);
			break;
		case PhysicsServer::BODY_MODE_RIGID: {
			mode = PhysicsServer::BODY_MODE_RIGID;
			const btScalar invMass = btBody->getInvMass();
			set_param(PhysicsServer::BODY_PARAM_MASS, 0 == invMass ? 1 : 1 / invMass);
			btBody->setAngularFactor(1);
			break;
		}
		case PhysicsServer::BODY_MODE_CHARACTER: {
			mode = PhysicsServer::BODY_MODE_CHARACTER;
			const btScalar invMass = btBody->getInvMass();
			set_param(PhysicsServer::BODY_PARAM_MASS, 0 == invMass ? 1 : 1 / invMass);
			btBody->setAngularFactor(0);
			break;
		}
	}
}
PhysicsServer::BodyMode BodyBullet::get_mode() const {
	return mode;
}

void BodyBullet::set_state(PhysicsServer::BodyState p_state, const Variant &p_variant) {

	switch (p_state) {
		case PhysicsServer::BODY_STATE_TRANSFORM:
			set_transform(p_variant);
			break;
		case PhysicsServer::BODY_STATE_LINEAR_VELOCITY:
			set_linear_velocity(p_variant);
			break;
		case PhysicsServer::BODY_STATE_ANGULAR_VELOCITY:
			set_angular_velocity(p_variant);
			break;
		case PhysicsServer::BODY_STATE_SLEEPING:
			set_active(p_variant);
			break;
		case PhysicsServer::BODY_STATE_CAN_SLEEP:
			if (bool(p_variant)) {
				// Can sleep
				btBody->forceActivationState(ACTIVE_TAG);
			} else {
				// Can't sleep
				btBody->forceActivationState(DISABLE_DEACTIVATION);
			}
			break;
	}
}

Variant BodyBullet::get_state(PhysicsServer::BodyState p_state) const {
	switch (p_state) {
		case PhysicsServer::BODY_STATE_TRANSFORM:
			return get_transform();
		case PhysicsServer::BODY_STATE_LINEAR_VELOCITY:
			return get_linear_velocity();
		case PhysicsServer::BODY_STATE_ANGULAR_VELOCITY:
			return get_angular_velocity();
		case PhysicsServer::BODY_STATE_SLEEPING:
			return !is_active();
		case PhysicsServer::BODY_STATE_CAN_SLEEP:
			return btBody->getActivationState();
		default:
			WARN_PRINTS("This state " + itos(p_state) + " is not supported by Bullet");
			return Variant();
	}
}

void BodyBullet::apply_central_impulse(const Vector3 &p_impulse) {
	btVector3 btImpu;
	G_TO_B(p_impulse, btImpu);
	btBody->activate();
	btBody->applyCentralImpulse(btImpu);
}

void BodyBullet::apply_impulse(const Vector3 &p_pos, const Vector3 &p_impulse) {
	btVector3 btImpu;
	btVector3 btPos;
	G_TO_B(p_impulse, btImpu);
	G_TO_B(p_pos, btPos);
	btBody->activate();
	btBody->applyImpulse(btImpu, btPos);
}

void BodyBullet::apply_torque_impulse(const Vector3 &p_impulse) {
	btVector3 btImp;
	G_TO_B(p_impulse, btImp);
	btBody->activate();
	btBody->applyTorqueImpulse(btImp);
}

void BodyBullet::apply_force(const Vector3 &p_force, const Vector3 &p_pos) {
	btVector3 btForce;
	btVector3 btPos;
	G_TO_B(p_force, btForce);
	G_TO_B(p_pos, btPos);
	btBody->activate();
	btBody->applyForce(btForce, btPos);
}

void BodyBullet::apply_central_force(const Vector3 &p_force) {
	btVector3 btForce;
	G_TO_B(p_force, btForce);
	btBody->activate();
	btBody->applyCentralForce(btForce);
}

void BodyBullet::apply_torque(const Vector3 &p_torque) {
	btVector3 btTorq;
	G_TO_B(p_torque, btTorq);
	btBody->activate();
	btBody->applyTorque(btTorq);
}

void BodyBullet::set_applied_force(const Vector3 &p_force) {
	btVector3 btVec = btBody->getTotalTorque();

	btBody->activate();

	btBody->clearForces();
	btBody->applyTorque(btVec);

	G_TO_B(p_force, btVec);
	btBody->applyCentralForce(btVec);
}

Vector3 BodyBullet::get_applied_force() const {
	Vector3 gTotForc;
	B_TO_G(btBody->getTotalForce(), gTotForc);
	return gTotForc;
}

void BodyBullet::set_applied_torque(const Vector3 &p_torque) {
	btVector3 btVec = btBody->getTotalForce();

	btBody->activate();

	btBody->clearForces();
	btBody->applyCentralForce(btVec);

	G_TO_B(p_torque, btVec);
	btBody->applyTorque(btVec);
}

Vector3 BodyBullet::get_applied_torque() const {
	Vector3 gTotTorq;
	B_TO_G(btBody->getTotalTorque(), gTotTorq);
	return gTotTorq;
}

void BodyBullet::set_axis_lock(PhysicsServer::BodyAxisLock p_lock) {
	if (PhysicsServer::BODY_AXIS_LOCK_DISABLED == p_lock) {
		btBody->setLinearFactor(btVector3(1., 1., 1.));
	} else if (PhysicsServer::BODY_AXIS_LOCK_X == p_lock) {
		btBody->setLinearFactor(btVector3(0., 1., 1.));
	} else if (PhysicsServer::BODY_AXIS_LOCK_Y == p_lock) {
		btBody->setLinearFactor(btVector3(1., 0., 1.));
	} else if (PhysicsServer::BODY_AXIS_LOCK_Z == p_lock) {
		btBody->setLinearFactor(btVector3(1., 1., 0.));
	}
}

PhysicsServer::BodyAxisLock BodyBullet::get_axis_lock() const {
	btVector3 vec = btBody->getLinearFactor();
	if (0. == vec.x()) {
		return PhysicsServer::BODY_AXIS_LOCK_X;
	} else if (0. == vec.y()) {
		return PhysicsServer::BODY_AXIS_LOCK_Y;
	} else if (0. == vec.z()) {
		return PhysicsServer::BODY_AXIS_LOCK_Z;
	} else {
		return PhysicsServer::BODY_AXIS_LOCK_DISABLED;
	}
}

void BodyBullet::set_continuous_collision_detection(bool p_enable) {
	if (p_enable) {
		// This threshold enable CCD if the object moves more than
		// 1 meter in one simulation frame
		btBody->setCcdMotionThreshold(1);

		/// Calculate using the rule writte below the CCD swept sphere radius
		///     CCD works on an embedded sphere of radius, make sure this radius
		///     is embedded inside the convex objects, preferably smaller:
		///     for an object of dimentions 1 meter, try 0.2
		btVector3 center;
		btScalar radius;
		btBody->getCollisionShape()->getBoundingSphere(center, radius);
		btBody->setCcdSweptSphereRadius(radius * 0.2);
	} else {
		btBody->setCcdMotionThreshold(0.);
		btBody->setCcdSweptSphereRadius(0.);
	}
}

bool BodyBullet::is_continuous_collision_detection_enabled() const {
	return 0. != btBody->getCcdMotionThreshold();
}

void BodyBullet::set_linear_velocity(const Vector3 &p_velocity) {
	btVector3 btVec;
	G_TO_B(p_velocity, btVec);
	btBody->activate();
	btBody->setLinearVelocity(btVec);
}

Vector3 BodyBullet::get_linear_velocity() const {
	Vector3 gVec;
	B_TO_G(btBody->getLinearVelocity(), gVec);
	return gVec;
}

void BodyBullet::set_angular_velocity(const Vector3 &p_velocity) {
	btVector3 btVec;
	G_TO_B(p_velocity, btVec);
	btBody->activate();
	btBody->setAngularVelocity(btVec);
}

Vector3 BodyBullet::get_angular_velocity() const {
	Vector3 gVec;
	B_TO_G(btBody->getAngularVelocity(), gVec);
	return gVec;
}

void BodyBullet::set_transform(const Transform &p_global_transform) {
	btTransform btTrans;
	/// Ortonormalize returns a transform that is different from the real transform if the original basis is scaled or sheared
	/// In this case the editor will alert the developer.
	G_TO_B(p_global_transform.orthonormalized(), btTrans);
	set_body_scale(static_cast<const CollisionObjectBullet::EnhancedBasis &>(p_global_transform.get_basis()).get_uniform_scale());
	if (mode == PhysicsServer::BODY_MODE_KINEMATIC) {
		// The kinematic use MotionState class
		godotMotionState->moveBody(btTrans);
	}
	btBody->setWorldTransform(btTrans);
}

Transform BodyBullet::get_transform() const {
	btTransform btTrans;
	Transform gTrans;
	godotMotionState->getCurrentWorldTransform(btTrans);
	B_TO_G(btTrans, gTrans);
	return gTrans;
}

void BodyBullet::on_shapes_changed() {
	CollisionObjectBullet::on_shapes_changed();

	if (space)
		space->remove_body(this);

	const btScalar invMass = btBody->getInvMass();
	const btScalar mass = invMass == 0 ? 0 : 1 / invMass;

	btVector3 inertia;
	btBody->getCollisionShape()->calculateLocalInertia(mass, inertia);
	btBody->setMassProps(mass, inertia);
	btBody->updateInertiaTensor();

	reload_kinematic_shapes();

	if (space)
		space->add_body(this);
}

void BodyBullet::on_enter_area(AreaBullet *p_area) {
	/// Add this area to the array in an ordered way
	++areaWhereIamCount;
	if (areaWhereIamCount >= maxAreasWhereIam) {
		--areaWhereIamCount;
		return;
	}
	for (int i = 0; i < areaWhereIamCount; ++i) {

		if (NULL == areasWhereIam[i]) {
			// This area has the highest priority
			areasWhereIam[i] = p_area;
			break;
		} else {
			if (areasWhereIam[i]->get_spOv_priority() > p_area->get_spOv_priority()) {
				// The position was found, just shift all elements
				for (int j = i; j < areaWhereIamCount; ++j) {
					areasWhereIam[j + 1] = areasWhereIam[j];
				}
				areasWhereIam[i] = p_area;
				break;
			}
		}
	}
	if (PhysicsServer::AREA_SPACE_OVERRIDE_DISABLED != p_area->get_spOv_mode()) {
		scratch_space_override_modificator();
	}

	if (p_area->is_spOv_gravityPoint()) {
		++countGravityPointSpaces;
		assert(0 < countGravityPointSpaces);
	}
}

void BodyBullet::on_exit_area(AreaBullet *p_area) {
	CollisionObjectBullet::on_exit_area(p_area);
	/// Remove this area and keep the order
	/// N.B. Since I don't want resize the array I can't use the "erase" function
	bool wasTheAreaFound = false;
	for (int i = 0; i < areaWhereIamCount; ++i) {
		if (p_area == areasWhereIam[i]) {
			// The area was fount, just shift down all elements
			for (int j = i; j < areaWhereIamCount; ++j) {
				areasWhereIam[j] = areasWhereIam[j + 1];
			}
			wasTheAreaFound = true;
			break;
		}
	}
	if (wasTheAreaFound) {
		if (p_area->is_spOv_gravityPoint()) {
			--countGravityPointSpaces;
			assert(0 <= countGravityPointSpaces);
		}

		--areaWhereIamCount;
		areasWhereIam[areaWhereIamCount] = NULL; // Even if this is not required, I clear the last element to be safe
		if (PhysicsServer::AREA_SPACE_OVERRIDE_DISABLED != p_area->get_spOv_mode()) {
			scratch_space_override_modificator();
		}
	}
}

void BodyBullet::reload_space_override_modificator() {

	Vector3 newGravity(space->get_gravity_direction() * space->get_gravity_magnitude());
	real_t newLinearDamp(linearDamp);
	real_t newAngularDamp(angularDamp);

	AreaBullet *currentArea;
	// Variable used to calculate new gravity for gravity point areas, it is pointed by currentGravity pointer
	Vector3 support_gravity(0, 0, 0);

	int countCombined(0);
	for (int i = areaWhereIamCount - 1; 0 <= i; --i) {

		currentArea = areasWhereIam[i];

		if (PhysicsServer::AREA_SPACE_OVERRIDE_DISABLED == currentArea->get_spOv_mode()) {
			continue;
		}

		/// Here is calculated the gravity
		if (currentArea->is_spOv_gravityPoint()) {

			/// It calculates the direction of new gravity
			support_gravity = currentArea->get_transform().xform(currentArea->get_spOv_gravityVec()) - get_transform().get_origin();
			real_t distanceMag = support_gravity.length();
			// Normalized in this way to avoid the double call of function "length()"
			if (distanceMag == 0) {
				support_gravity.x = 0;
				support_gravity.y = 0;
				support_gravity.z = 0;
			} else {
				support_gravity.x /= distanceMag;
				support_gravity.y /= distanceMag;
				support_gravity.z /= distanceMag;
			}

			/// Here is calculated the final gravity
			if (currentArea->get_spOv_gravityPointDistanceScale() > 0) {
				// Scaled gravity by distance
				support_gravity *= currentArea->get_spOv_gravityMag() / Math::pow(distanceMag * currentArea->get_spOv_gravityPointDistanceScale() + 1, 2);
			} else {
				// Unscaled gravity
				support_gravity *= currentArea->get_spOv_gravityMag();
			}
		} else {
			support_gravity = currentArea->get_spOv_gravityVec() * currentArea->get_spOv_gravityMag();
		}

		switch (currentArea->get_spOv_mode()) {
			///case PhysicsServer::AREA_SPACE_OVERRIDE_DISABLED:
			/// This area does not affect gravity/damp. These are generally areas
			/// that exist only to detect collisions, and objects entering or exiting them.
			///    break;
			case PhysicsServer::AREA_SPACE_OVERRIDE_COMBINE:
				/// This area adds its gravity/damp values to whatever has been
				/// calculated so far. This way, many overlapping areas can combine
				/// their physics to make interesting
				newGravity += support_gravity;
				newLinearDamp += currentArea->get_spOv_linearDamp();
				newAngularDamp += currentArea->get_spOv_angularDamp();
				++countCombined;
				break;
			case PhysicsServer::AREA_SPACE_OVERRIDE_COMBINE_REPLACE:
				/// This area adds its gravity/damp values to whatever has been calculated
				/// so far. Then stops taking into account the rest of the areas, even the
				/// default one.
				newGravity += support_gravity;
				newLinearDamp += currentArea->get_spOv_linearDamp();
				newAngularDamp += currentArea->get_spOv_angularDamp();
				++countCombined;
				goto endAreasCycle;
			case PhysicsServer::AREA_SPACE_OVERRIDE_REPLACE:
				/// This area replaces any gravity/damp, even the default one, and
				/// stops taking into account the rest of the areas.
				newGravity = support_gravity;
				newLinearDamp = currentArea->get_spOv_linearDamp();
				newAngularDamp = currentArea->get_spOv_angularDamp();
				countCombined = 1;
				goto endAreasCycle;
			case PhysicsServer::AREA_SPACE_OVERRIDE_REPLACE_COMBINE:
				/// This area replaces any gravity/damp calculated so far, but keeps
				/// calculating the rest of the areas, down to the default one.
				newGravity = support_gravity;
				newLinearDamp = currentArea->get_spOv_linearDamp();
				newAngularDamp = currentArea->get_spOv_angularDamp();
				countCombined = 1;
				break;
		}
	}
endAreasCycle:

	if (1 < countCombined) {
		newGravity /= countCombined;
		newLinearDamp /= countCombined;
		newAngularDamp /= countCombined;
	}

	btVector3 newBtGravity;
	G_TO_B(newGravity * gravity_scale, newBtGravity);

	btBody->setGravity(newBtGravity);
	btBody->setDamping(newLinearDamp, newAngularDamp);
}

void BodyBullet::reload_kinematic_shapes() {
	if (!kinematic_utilities) {
		return;
	}
	kinematic_utilities->copyAllOwnerShapes();
}

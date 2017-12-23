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

#include "rigid_body_bullet.h"
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
	body->set_activation_state(p_enable);
}

bool BulletPhysicsDirectBodyState::is_sleeping() const {
	return !body->is_active();
}

int BulletPhysicsDirectBodyState::get_contact_count() const {
	return body->collisionsCount;
}

Vector3 BulletPhysicsDirectBodyState::get_contact_local_position(int p_contact_idx) const {
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

Vector3 BulletPhysicsDirectBodyState::get_contact_collider_position(int p_contact_idx) const {
	return body->collisions[p_contact_idx].hitWorldLocation;
}

ObjectID BulletPhysicsDirectBodyState::get_contact_collider_id(int p_contact_idx) const {
	return body->collisions[p_contact_idx].otherObject->get_instance_id();
}

int BulletPhysicsDirectBodyState::get_contact_collider_shape(int p_contact_idx) const {
	return body->collisions[p_contact_idx].other_object_shape;
}

Vector3 BulletPhysicsDirectBodyState::get_contact_collider_velocity_at_position(int p_contact_idx) const {
	RigidBodyBullet::CollisionData &colDat = body->collisions[p_contact_idx];

	btVector3 hitLocation;
	G_TO_B(colDat.hitLocalLocation, hitLocation);

	Vector3 velocityAtPoint;
	B_TO_G(colDat.otherObject->get_bt_rigid_body()->getVelocityInLocalPoint(hitLocation), velocityAtPoint);

	return velocityAtPoint;
}

PhysicsDirectSpaceState *BulletPhysicsDirectBodyState::get_space_state() {
	return body->get_space()->get_direct_state();
}

RigidBodyBullet::KinematicUtilities::KinematicUtilities(RigidBodyBullet *p_owner) :
		owner(p_owner),
		safe_margin(0.001) {
}

RigidBodyBullet::KinematicUtilities::~KinematicUtilities() {
	just_delete_shapes(shapes.size()); // don't need to resize
}

void RigidBodyBullet::KinematicUtilities::setSafeMargin(btScalar p_margin) {
	safe_margin = p_margin;
	copyAllOwnerShapes();
}

void RigidBodyBullet::KinematicUtilities::copyAllOwnerShapes() {
	const Vector<CollisionObjectBullet::ShapeWrapper> &shapes_wrappers(owner->get_shapes_wrappers());
	const int shapes_count = shapes_wrappers.size();

	just_delete_shapes(shapes_count);

	const CollisionObjectBullet::ShapeWrapper *shape_wrapper;

	btVector3 owner_body_scale(owner->get_bt_body_scale());

	for (int i = shapes_count - 1; 0 <= i; --i) {
		shape_wrapper = &shapes_wrappers[i];
		if (!shape_wrapper->active) {
			continue;
		}
		shapes[i].transform = shape_wrapper->transform;

		btConvexShape *&kin_shape_ref = shapes[i].shape;

		switch (shape_wrapper->shape->get_type()) {
			case PhysicsServer::SHAPE_SPHERE: {
				SphereShapeBullet *sphere = static_cast<SphereShapeBullet *>(shape_wrapper->shape);
				kin_shape_ref = ShapeBullet::create_shape_sphere(sphere->get_radius() * owner_body_scale[0] + safe_margin);
				break;
			}
			case PhysicsServer::SHAPE_BOX: {
				BoxShapeBullet *box = static_cast<BoxShapeBullet *>(shape_wrapper->shape);
				kin_shape_ref = ShapeBullet::create_shape_box((box->get_half_extents() * owner_body_scale) + btVector3(safe_margin, safe_margin, safe_margin));
				break;
			}
			case PhysicsServer::SHAPE_CAPSULE: {
				CapsuleShapeBullet *capsule = static_cast<CapsuleShapeBullet *>(shape_wrapper->shape);

				kin_shape_ref = ShapeBullet::create_shape_capsule(capsule->get_radius() * owner_body_scale[0] + safe_margin, capsule->get_height() * owner_body_scale[1] + safe_margin);
				break;
			}
			case PhysicsServer::SHAPE_CONVEX_POLYGON: {
				ConvexPolygonShapeBullet *godot_convex = static_cast<ConvexPolygonShapeBullet *>(shape_wrapper->shape);
				kin_shape_ref = ShapeBullet::create_shape_convex(godot_convex->vertices);
				kin_shape_ref->setLocalScaling(owner_body_scale + btVector3(safe_margin, safe_margin, safe_margin));
				break;
			}
			case PhysicsServer::SHAPE_RAY: {
				RayShapeBullet *godot_ray = static_cast<RayShapeBullet *>(shape_wrapper->shape);
				kin_shape_ref = ShapeBullet::create_shape_ray(godot_ray->length * owner_body_scale[1] + safe_margin);
				break;
			}
			default:
				WARN_PRINT("This shape is not supported to be kinematic!");
				kin_shape_ref = NULL;
		}
	}
}

void RigidBodyBullet::KinematicUtilities::just_delete_shapes(int new_size) {
	for (int i = shapes.size() - 1; 0 <= i; --i) {
		if (shapes[i].shape) {
			bulletdelete(shapes[i].shape);
		}
	}
	shapes.resize(new_size);
}

RigidBodyBullet::RigidBodyBullet() :
		RigidCollisionObjectBullet(CollisionObjectBullet::TYPE_RIGID_BODY),
		kinematic_utilities(NULL),
		locked_axis(0),
		gravity_scale(1),
		mass(1),
		linearDamp(0),
		angularDamp(0),
		can_sleep(true),
		force_integration_callback(NULL),
		isTransformChanged(false),
		previousActiveState(true),
		maxCollisionsDetection(0),
		collisionsCount(0),
		maxAreasWhereIam(10),
		areaWhereIamCount(0),
		countGravityPointSpaces(0),
		isScratchedSpaceOverrideModificator(false) {

	godotMotionState = bulletnew(GodotMotionState(this));

	// Initial properties
	const btVector3 localInertia(0, 0, 0);
	btRigidBody::btRigidBodyConstructionInfo cInfo(mass, godotMotionState, compoundShape, localInertia);

	btBody = bulletnew(btRigidBody(cInfo));
	setupBulletCollisionObject(btBody);

	set_mode(PhysicsServer::BODY_MODE_RIGID);
	reload_axis_lock();

	areasWhereIam.resize(maxAreasWhereIam);
	for (int i = areasWhereIam.size() - 1; 0 <= i; --i) {
		areasWhereIam[i] = NULL;
	}
	btBody->setSleepingThresholds(0.2, 0.2);
}

RigidBodyBullet::~RigidBodyBullet() {
	bulletdelete(godotMotionState);

	if (force_integration_callback)
		memdelete(force_integration_callback);

	destroy_kinematic_utilities();
}

void RigidBodyBullet::init_kinematic_utilities() {
	kinematic_utilities = memnew(KinematicUtilities(this));
}

void RigidBodyBullet::destroy_kinematic_utilities() {
	if (kinematic_utilities) {
		memdelete(kinematic_utilities);
		kinematic_utilities = NULL;
	}
}

void RigidBodyBullet::reload_body() {
	if (space) {
		space->remove_rigid_body(this);
		space->add_rigid_body(this);
	}
}

void RigidBodyBullet::set_space(SpaceBullet *p_space) {
	// Clear the old space if there is one
	if (space) {
		isTransformChanged = false;

		// Remove all eventual constraints
		assert_no_constraints();

		// Remove this object form the physics world
		space->remove_rigid_body(this);
	}

	space = p_space;

	if (space) {
		space->add_rigid_body(this);
	}
}

void RigidBodyBullet::dispatch_callbacks() {
	/// The check isTransformChanged is necessary in order to call integrated forces only when the first transform is sent
	if ((btBody->isActive() || previousActiveState != btBody->isActive()) && force_integration_callback && isTransformChanged) {

		BulletPhysicsDirectBodyState *bodyDirect = BulletPhysicsDirectBodyState::get_singleton(this);

		Variant variantBodyDirect = bodyDirect;

		Object *obj = ObjectDB::get_instance(force_integration_callback->id);
		if (!obj) {
			// Remove integration callback
			set_force_integration_callback(0, StringName());
		} else {
			const Variant *vp[2] = { &variantBodyDirect, &force_integration_callback->udata };

			Variant::CallError responseCallError;
			int argc = (force_integration_callback->udata.get_type() == Variant::NIL) ? 1 : 2;
			obj->call(force_integration_callback->method, vp, argc, responseCallError);
		}
	}

	if (isScratchedSpaceOverrideModificator || 0 < countGravityPointSpaces) {
		isScratchedSpaceOverrideModificator = false;
		reload_space_override_modificator();
	}

	/// Lock axis
	btBody->setLinearVelocity(btBody->getLinearVelocity() * btBody->getLinearFactor());
	btBody->setAngularVelocity(btBody->getAngularVelocity() * btBody->getAngularFactor());

	previousActiveState = btBody->isActive();
}

void RigidBodyBullet::set_force_integration_callback(ObjectID p_id, const StringName &p_method, const Variant &p_udata) {

	if (force_integration_callback) {
		memdelete(force_integration_callback);
		force_integration_callback = NULL;
	}

	if (p_id != 0) {
		force_integration_callback = memnew(ForceIntegrationCallback);
		force_integration_callback->id = p_id;
		force_integration_callback->method = p_method;
		force_integration_callback->udata = p_udata;
	}
}

void RigidBodyBullet::scratch() {
	isTransformChanged = true;
}

void RigidBodyBullet::scratch_space_override_modificator() {
	isScratchedSpaceOverrideModificator = true;
}

void RigidBodyBullet::on_collision_filters_change() {
	if (space) {
		space->reload_collision_filters(this);
	}
}

void RigidBodyBullet::on_collision_checker_start() {
	collisionsCount = 0;
}

bool RigidBodyBullet::add_collision_object(RigidBodyBullet *p_otherObject, const Vector3 &p_hitWorldLocation, const Vector3 &p_hitLocalLocation, const Vector3 &p_hitNormal, int p_other_shape_index, int p_local_shape_index) {

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

void RigidBodyBullet::assert_no_constraints() {
	if (btBody->getNumConstraintRefs()) {
		WARN_PRINT("A body with a joints is destroyed. Please check the implementation in order to destroy the joint before the body.");
	}
	/*for(int i = btBody->getNumConstraintRefs()-1; 0<=i; --i){
        btTypedConstraint* btConst = btBody->getConstraintRef(i);
        JointBullet* joint = static_cast<JointBullet*>( btConst->getUserConstraintPtr() );
        space->removeConstraint(joint);
    }*/
}

void RigidBodyBullet::set_activation_state(bool p_active) {
	if (p_active) {
		btBody->setActivationState(ACTIVE_TAG);
	} else {
		btBody->setActivationState(WANTS_DEACTIVATION);
	}
}

bool RigidBodyBullet::is_active() const {
	return btBody->isActive();
}

void RigidBodyBullet::set_param(PhysicsServer::BodyParameter p_param, real_t p_value) {
	switch (p_param) {
		case PhysicsServer::BODY_PARAM_BOUNCE:
			btBody->setRestitution(p_value);
			break;
		case PhysicsServer::BODY_PARAM_FRICTION:
			btBody->setFriction(p_value);
			break;
		case PhysicsServer::BODY_PARAM_MASS: {
			ERR_FAIL_COND(p_value < 0);
			mass = p_value;
			_internal_set_mass(p_value);
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

real_t RigidBodyBullet::get_param(PhysicsServer::BodyParameter p_param) const {
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

void RigidBodyBullet::set_mode(PhysicsServer::BodyMode p_mode) {
	// This is necessary to block force_integration untile next move
	isTransformChanged = false;
	destroy_kinematic_utilities();
	// The mode change is relevant to its mass
	switch (p_mode) {
		case PhysicsServer::BODY_MODE_KINEMATIC:
			mode = PhysicsServer::BODY_MODE_KINEMATIC;
			reload_axis_lock();
			_internal_set_mass(0);
			init_kinematic_utilities();
			break;
		case PhysicsServer::BODY_MODE_STATIC:
			mode = PhysicsServer::BODY_MODE_STATIC;
			reload_axis_lock();
			_internal_set_mass(0);
			break;
		case PhysicsServer::BODY_MODE_RIGID: {
			mode = PhysicsServer::BODY_MODE_RIGID;
			reload_axis_lock();
			_internal_set_mass(0 == mass ? 1 : mass);
			scratch_space_override_modificator();
			break;
		}
		case PhysicsServer::BODY_MODE_CHARACTER: {
			mode = PhysicsServer::BODY_MODE_CHARACTER;
			reload_axis_lock();
			_internal_set_mass(0 == mass ? 1 : mass);
			scratch_space_override_modificator();
			break;
		}
	}

	btBody->setAngularVelocity(btVector3(0, 0, 0));
	btBody->setLinearVelocity(btVector3(0, 0, 0));
}
PhysicsServer::BodyMode RigidBodyBullet::get_mode() const {
	return mode;
}

void RigidBodyBullet::set_state(PhysicsServer::BodyState p_state, const Variant &p_variant) {

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
			set_activation_state(!bool(p_variant));
			break;
		case PhysicsServer::BODY_STATE_CAN_SLEEP:
			can_sleep = bool(p_variant);
			if (!can_sleep) {
				// Can't sleep
				btBody->forceActivationState(DISABLE_DEACTIVATION);
			}
			break;
	}
}

Variant RigidBodyBullet::get_state(PhysicsServer::BodyState p_state) const {
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
			return can_sleep;
		default:
			WARN_PRINTS("This state " + itos(p_state) + " is not supported by Bullet");
			return Variant();
	}
}

void RigidBodyBullet::apply_central_impulse(const Vector3 &p_impulse) {
	btVector3 btImpu;
	G_TO_B(p_impulse, btImpu);
	if (Vector3() != p_impulse)
		btBody->activate();
	btBody->applyCentralImpulse(btImpu);
}

void RigidBodyBullet::apply_impulse(const Vector3 &p_pos, const Vector3 &p_impulse) {
	btVector3 btImpu;
	btVector3 btPos;
	G_TO_B(p_impulse, btImpu);
	G_TO_B(p_pos, btPos);
	if (Vector3() != p_impulse)
		btBody->activate();
	btBody->applyImpulse(btImpu, btPos);
}

void RigidBodyBullet::apply_torque_impulse(const Vector3 &p_impulse) {
	btVector3 btImp;
	G_TO_B(p_impulse, btImp);
	if (Vector3() != p_impulse)
		btBody->activate();
	btBody->applyTorqueImpulse(btImp);
}

void RigidBodyBullet::apply_force(const Vector3 &p_force, const Vector3 &p_pos) {
	btVector3 btForce;
	btVector3 btPos;
	G_TO_B(p_force, btForce);
	G_TO_B(p_pos, btPos);
	if (Vector3() != p_force)
		btBody->activate();
	btBody->applyForce(btForce, btPos);
}

void RigidBodyBullet::apply_central_force(const Vector3 &p_force) {
	btVector3 btForce;
	G_TO_B(p_force, btForce);
	if (Vector3() != p_force)
		btBody->activate();
	btBody->applyCentralForce(btForce);
}

void RigidBodyBullet::apply_torque(const Vector3 &p_torque) {
	btVector3 btTorq;
	G_TO_B(p_torque, btTorq);
	if (Vector3() != p_torque)
		btBody->activate();
	btBody->applyTorque(btTorq);
}

void RigidBodyBullet::set_applied_force(const Vector3 &p_force) {
	btVector3 btVec = btBody->getTotalTorque();

	if (Vector3() != p_force)
		btBody->activate();

	btBody->clearForces();
	btBody->applyTorque(btVec);

	G_TO_B(p_force, btVec);
	btBody->applyCentralForce(btVec);
}

Vector3 RigidBodyBullet::get_applied_force() const {
	Vector3 gTotForc;
	B_TO_G(btBody->getTotalForce(), gTotForc);
	return gTotForc;
}

void RigidBodyBullet::set_applied_torque(const Vector3 &p_torque) {
	btVector3 btVec = btBody->getTotalForce();

	if (Vector3() != p_torque)
		btBody->activate();

	btBody->clearForces();
	btBody->applyCentralForce(btVec);

	G_TO_B(p_torque, btVec);
	btBody->applyTorque(btVec);
}

Vector3 RigidBodyBullet::get_applied_torque() const {
	Vector3 gTotTorq;
	B_TO_G(btBody->getTotalTorque(), gTotTorq);
	return gTotTorq;
}

void RigidBodyBullet::set_axis_lock(PhysicsServer::BodyAxis p_axis, bool lock) {
	if (lock) {
		locked_axis |= p_axis;
	} else {
		locked_axis &= ~p_axis;
	}

	reload_axis_lock();
}

bool RigidBodyBullet::is_axis_locked(PhysicsServer::BodyAxis p_axis) const {
	return locked_axis & p_axis;
}

void RigidBodyBullet::reload_axis_lock() {

	btBody->setLinearFactor(btVector3(!is_axis_locked(PhysicsServer::BODY_AXIS_LINEAR_X), !is_axis_locked(PhysicsServer::BODY_AXIS_LINEAR_Y), !is_axis_locked(PhysicsServer::BODY_AXIS_LINEAR_Z)));
	if (PhysicsServer::BODY_MODE_CHARACTER == mode) {
		/// When character angular is always locked
		btBody->setAngularFactor(btVector3(0., 0., 0.));
	} else {
		btBody->setAngularFactor(btVector3(!is_axis_locked(PhysicsServer::BODY_AXIS_ANGULAR_X), !is_axis_locked(PhysicsServer::BODY_AXIS_ANGULAR_Y), !is_axis_locked(PhysicsServer::BODY_AXIS_ANGULAR_Z)));
	}
}

void RigidBodyBullet::set_continuous_collision_detection(bool p_enable) {
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

bool RigidBodyBullet::is_continuous_collision_detection_enabled() const {
	return 0. != btBody->getCcdMotionThreshold();
}

void RigidBodyBullet::set_linear_velocity(const Vector3 &p_velocity) {
	btVector3 btVec;
	G_TO_B(p_velocity, btVec);
	if (Vector3() != p_velocity)
		btBody->activate();
	btBody->setLinearVelocity(btVec);
}

Vector3 RigidBodyBullet::get_linear_velocity() const {
	Vector3 gVec;
	B_TO_G(btBody->getLinearVelocity(), gVec);
	return gVec;
}

void RigidBodyBullet::set_angular_velocity(const Vector3 &p_velocity) {
	btVector3 btVec;
	G_TO_B(p_velocity, btVec);
	if (Vector3() != p_velocity)
		btBody->activate();
	btBody->setAngularVelocity(btVec);
}

Vector3 RigidBodyBullet::get_angular_velocity() const {
	Vector3 gVec;
	B_TO_G(btBody->getAngularVelocity(), gVec);
	return gVec;
}

void RigidBodyBullet::set_transform__bullet(const btTransform &p_global_transform) {
	if (mode == PhysicsServer::BODY_MODE_KINEMATIC) {
		// The kinematic use MotionState class
		godotMotionState->moveBody(p_global_transform);
	}
	btBody->setWorldTransform(p_global_transform);
}

const btTransform &RigidBodyBullet::get_transform__bullet() const {
	if (is_static()) {

		return RigidCollisionObjectBullet::get_transform__bullet();
	} else {

		return godotMotionState->getCurrentWorldTransform();
	}
}

void RigidBodyBullet::on_shapes_changed() {
	RigidCollisionObjectBullet::on_shapes_changed();

	const btScalar invMass = btBody->getInvMass();
	const btScalar mass = invMass == 0 ? 0 : 1 / invMass;

	btVector3 inertia;
	btBody->getCollisionShape()->calculateLocalInertia(mass, inertia);
	btBody->setMassProps(mass, inertia);
	btBody->updateInertiaTensor();

	reload_kinematic_shapes();

	reload_body();
}

void RigidBodyBullet::on_enter_area(AreaBullet *p_area) {
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

void RigidBodyBullet::on_exit_area(AreaBullet *p_area) {
	RigidCollisionObjectBullet::on_exit_area(p_area);
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

void RigidBodyBullet::reload_space_override_modificator() {

	if (!is_active())
		return;

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

void RigidBodyBullet::reload_kinematic_shapes() {
	if (!kinematic_utilities) {
		return;
	}
	kinematic_utilities->copyAllOwnerShapes();
}

void RigidBodyBullet::_internal_set_mass(real_t p_mass) {

	btVector3 localInertia(0, 0, 0);

	int clearedCurrentFlags = btBody->getCollisionFlags();
	clearedCurrentFlags &= ~(btCollisionObject::CF_KINEMATIC_OBJECT | btCollisionObject::CF_STATIC_OBJECT | btCollisionObject::CF_CHARACTER_OBJECT);

	// Rigidbody is dynamic if and only if mass is non Zero, otherwise static
	const bool isDynamic = p_mass != 0.f;
	if (isDynamic) {

		ERR_FAIL_COND(PhysicsServer::BODY_MODE_RIGID != mode && PhysicsServer::BODY_MODE_CHARACTER != mode);

		m_isStatic = false;
		compoundShape->calculateLocalInertia(p_mass, localInertia);

		if (PhysicsServer::BODY_MODE_RIGID == mode) {

			btBody->setCollisionFlags(clearedCurrentFlags); // Just set the flags without Kin and Static
		} else {

			btBody->setCollisionFlags(clearedCurrentFlags | btCollisionObject::CF_CHARACTER_OBJECT);
		}

		if (can_sleep) {
			btBody->forceActivationState(ACTIVE_TAG); // ACTIVE_TAG 1
		} else {
			btBody->forceActivationState(DISABLE_DEACTIVATION); // DISABLE_DEACTIVATION 4
		}
	} else {

		ERR_FAIL_COND(PhysicsServer::BODY_MODE_STATIC != mode && PhysicsServer::BODY_MODE_KINEMATIC != mode);

		m_isStatic = true;
		if (PhysicsServer::BODY_MODE_STATIC == mode) {

			btBody->setCollisionFlags(clearedCurrentFlags | btCollisionObject::CF_STATIC_OBJECT);
		} else {

			btBody->setCollisionFlags(clearedCurrentFlags | btCollisionObject::CF_KINEMATIC_OBJECT);
			set_transform__bullet(btBody->getWorldTransform()); // Set current Transform using kinematic method
		}
		btBody->forceActivationState(DISABLE_SIMULATION); // DISABLE_SIMULATION 5
	}

	btBody->setMassProps(p_mass, localInertia);
	btBody->updateInertiaTensor();

	reload_body();
}

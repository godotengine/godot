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

#include "armature_bullet.h"

#include "bullet_physics_server.h"
#include "bullet_types_converter.h"
#include "bullet_utilities.h"
#include "joint_bullet.h"

#include <BulletDynamics/Featherstone/btMultiBody.h>
#include <BulletDynamics/Featherstone/btMultiBodyJointLimitConstraint.h>
#include <BulletDynamics/Featherstone/btMultiBodyJointMotor.h>
#include <BulletDynamics/Featherstone/btMultiBodyLinkCollider.h>
#include <BulletDynamics/Featherstone/btMultiBodySphericalJointMotor.h>

/**
	@author AndreaCatania
*/

ArmatureBullet::ArmatureBullet() :
		space(NULL),
		active(false),
		force_integration_callback(NULL) {

	bt_body = bulletnew(btMultiBody(
			0, // N of links
			1, // mass
			btVector3(1, 1, 1), // inertia
			false, // fixed base
			true // Can sleep
			));

	bt_body->finalizeMultiDof();
}

ArmatureBullet::~ArmatureBullet() {
	if (force_integration_callback) {
		memdelete(force_integration_callback);
		force_integration_callback = NULL;
	}

	clear_links();
}

btMultiBody *ArmatureBullet::get_bt_body() const {
	return bt_body;
}

void ArmatureBullet::set_bone_count(int p_count) {
	if (p_count > 0) {

		bt_body->setNumLinks(p_count - 1);

	} else {
		bt_body->setNumLinks(0);
	}
	bt_body->finalizeMultiDof();
}

int ArmatureBullet::get_bone_count() const {
	return bt_body->getNumLinks();
}

void ArmatureBullet::set_space(SpaceBullet *p_space) {
	if (space == p_space)
		return; //pointles

	if (space) {
		space->remove_armature(this);
	}

	space = p_space;

	update_activation();
}

SpaceBullet *ArmatureBullet::get_space() const {
	return space;
}

void ArmatureBullet::set_active(bool p_active) {
	active = p_active;
	update_activation();
}

void ArmatureBullet::set_transform(const Transform &p_global_transform) {
	//set_body_scale(p_global_transform.basis.get_scale_abs());

	btTransform bt_transform;
	G_TO_B(p_global_transform, bt_transform);
	UNSCALE_BT_BASIS(bt_transform);

	set_transform__bullet(bt_transform);
}

Transform ArmatureBullet::get_transform() const {
	Transform t;
	B_TO_G(get_transform__bullet(), t);
	//t.basis.scale(body_scale);
	return t;
}

void ArmatureBullet::set_transform__bullet(const btTransform &p_global_transform) {
	transform = p_global_transform;
	bt_body->setBaseWorldTransform(p_global_transform);
}

const btTransform &ArmatureBullet::get_transform__bullet() const {
	return transform;
}

void ArmatureBullet::set_force_integration_callback(ObjectID p_id, const StringName &p_method, const Variant &p_udata) {

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

void ArmatureBullet::set_bone(BoneBullet *p_bone) {
	if (p_bone->get_link_id() == -1) {

		bt_body->setBaseCollider(p_bone->get_bt_body());
		set_transform__bullet(p_bone->get_transform__bullet());
		update_base_inertia();

	} else {

		bt_body->getLink(p_bone->get_link_id()).m_collider = p_bone->get_bt_body();
	}
	update_ext_joints();
}

BoneBullet *ArmatureBullet::get_bone(int p_link_id) const {
	if (p_link_id == -1) {
		if (bt_body->getBaseCollider())
			return static_cast<BoneBullet *>(bt_body->getBaseCollider()->getUserPointer());
		return NULL;
	} else {
		ERR_FAIL_INDEX_V(p_link_id, bt_body->getNumLinks(), NULL);
		return static_cast<BoneBullet *>(bt_body->getLink(p_link_id).m_collider->getUserPointer());
	}
}

void ArmatureBullet::remove_bone(int p_link_id) {
	BoneBullet *bone = get_bone(p_link_id);

	if (bone) {
		for (int i = 0; i < ext_joints.size(); ++i) {
			if (ext_joints[i]->get_body_a() == bone || ext_joints[i]->get_body_b() == bone) {

				ext_joints[i]->clear_internal_joint();
			}
		}
	}

	if (p_link_id == -1) {

		bt_body->setBaseCollider(NULL);

	} else {
		bt_body->getLink(p_link_id).m_collider = NULL;
	}
}

void ArmatureBullet::update_base_inertia() {

	btVector3 inertia(0, 0, 0);

	if (bt_body->getBaseCollider()) {
		if (bt_body->getBaseCollider()->getCollisionShape()) {
			bt_body->getBaseCollider()->getCollisionShape()->calculateLocalInertia(
					bt_body->getBaseMass(),
					inertia);
		}
	}

	bt_body->setBaseInertia(inertia);
}

void ArmatureBullet::update_link_mass_and_inertia(int p_link_id) {

	ERR_FAIL_COND(p_link_id == -1);

	BoneBullet *bone = get_bone(p_link_id);
	ERR_FAIL_COND(bone == NULL);

	btVector3 inertia(0, 0, 0);

	bone->calc_link_inertia(
			bt_body->getLink(p_link_id).m_jointType,
			bone->link_mass,
			bone->link_half_size,
			inertia);

	bt_body->getLink(p_link_id).m_mass = bone->link_mass;
	bt_body->getLink(p_link_id).m_inertiaLocal = inertia;
}

void ArmatureBullet::set_param(PhysicsServer::ArmatureParameter p_param, real_t p_value) {
	switch (p_param) {
		case PhysicsServer::ARMATURE_PARAM_MASS: {

			if (p_value > CMP_EPSILON) {
				bt_body->setBaseMass(p_value);
			} else {
				ERR_FAIL_COND(p_value < 0);
				bt_body->setBaseMass(0);
				// bt_body->setBaseFixed(true) TODO add this
			}
			update_base_inertia();

		} break;
		case PhysicsServer::ARMATURE_PARAM_MAX:
		default:
			WARN_PRINTS("Parameter " + itos(p_param) + " not supported by bullet. Value: " + itos(p_value));
	}
}

real_t ArmatureBullet::get_param(PhysicsServer::ArmatureParameter p_param) const {
	switch (p_param) {
		case PhysicsServer::ARMATURE_PARAM_MASS:
			return bt_body->getBaseMass();
		case PhysicsServer::ARMATURE_PARAM_MAX:
		default:
			WARN_PRINTS("Parameter " + itos(p_param) + " not supported by bullet.");
			return 0;
	}
}

void ArmatureBullet::register_ext_joint(JointBullet *p_joint) {
	if (ext_joints.find(p_joint) == -1)
		ext_joints.push_back(p_joint);
}

void ArmatureBullet::erase_ext_joint(JointBullet *p_joint) {
	ext_joints.erase(p_joint);
}

void ArmatureBullet::clear_links() {
	const int link_count = bt_body->getNumLinks();
	for (int i = 0; i < link_count; ++i) {
		if (bt_body->getLink(i).m_collider) {
			BoneBullet *bb = get_bone(i);
			bb->set_armature(NULL);
			bt_body->getLink(i).m_collider = NULL;
		}
	}
}

void ArmatureBullet::update_activation() {

	if (space) {
		space->remove_armature(this);
		if (active) {
			space->add_armature(this);
			get_bt_body()->wakeUp();
		} else {
			get_bt_body()->goToSleep();
		}
	}

	// -1 to take care of the base
	for (int i = -1; i < get_bone_count(); ++i) {

		BoneBullet *bp = get_bone(i);
		if (!bp)
			continue;

		bp->reload_body();
		if (active) {
			bp->get_bt_body()->setActivationState(ACTIVE_TAG);
		}
	}
}

void ArmatureBullet::update_ext_joints() {
	for (int i = 0; i < ext_joints.size(); ++i) {
		ext_joints[i]->reload_internal();
	}
}

BoneBulletPhysicsDirectBodyState *BoneBulletPhysicsDirectBodyState::singleton = NULL;

Vector3 BoneBulletPhysicsDirectBodyState::get_total_gravity() const {
	return bone->get_space()->get_gravity_direction() * bone->get_space()->get_gravity_magnitude();
}

float BoneBulletPhysicsDirectBodyState::get_total_angular_damp() const {
	return 0;
}

float BoneBulletPhysicsDirectBodyState::get_total_linear_damp() const {
	return 0;
}

Vector3 BoneBulletPhysicsDirectBodyState::get_center_of_mass() const {
	return bone->get_transform().get_origin();
}

Basis BoneBulletPhysicsDirectBodyState::get_principal_inertia_axes() const {
	return Basis();
}

float BoneBulletPhysicsDirectBodyState::get_inverse_mass() const {
	return 0;
}

Vector3 BoneBulletPhysicsDirectBodyState::get_inverse_inertia() const {
	return Vector3();
}

Basis BoneBulletPhysicsDirectBodyState::get_inverse_inertia_tensor() const {
	return Basis();
}

void BoneBulletPhysicsDirectBodyState::set_linear_velocity(const Vector3 &p_velocity) {
}

Vector3 BoneBulletPhysicsDirectBodyState::get_linear_velocity() const {
	return Vector3();
}

void BoneBulletPhysicsDirectBodyState::set_angular_velocity(const Vector3 &p_velocity) {
}

Vector3 BoneBulletPhysicsDirectBodyState::get_angular_velocity() const {
	return Vector3();
}

void BoneBulletPhysicsDirectBodyState::set_transform(const Transform &p_transform) {
	bone->set_transform(p_transform);
}

Transform BoneBulletPhysicsDirectBodyState::get_transform() const {
	return bone->get_transform();
}

void BoneBulletPhysicsDirectBodyState::add_central_force(const Vector3 &p_force) {
}

void BoneBulletPhysicsDirectBodyState::add_force(const Vector3 &p_force, const Vector3 &p_pos) {
}

void BoneBulletPhysicsDirectBodyState::add_torque(const Vector3 &p_torque) {
}

void BoneBulletPhysicsDirectBodyState::apply_central_impulse(const Vector3 &p_j) {
}

void BoneBulletPhysicsDirectBodyState::apply_impulse(const Vector3 &p_pos, const Vector3 &p_j) {
}

void BoneBulletPhysicsDirectBodyState::apply_torque_impulse(const Vector3 &p_j) {
}

void BoneBulletPhysicsDirectBodyState::set_sleep_state(bool p_enable) {
}

bool BoneBulletPhysicsDirectBodyState::is_sleeping() const {
	return !bone->get_bt_body()->isActive();
}

int BoneBulletPhysicsDirectBodyState::get_contact_count() const {
	return bone->collisionsCount;
}

Vector3 BoneBulletPhysicsDirectBodyState::get_contact_local_position(int p_contact_idx) const {
	return bone->collisions[p_contact_idx].hitLocalLocation;
}

Vector3 BoneBulletPhysicsDirectBodyState::get_contact_local_normal(int p_contact_idx) const {
	return bone->collisions[p_contact_idx].hitNormal;
}

float BoneBulletPhysicsDirectBodyState::get_contact_impulse(int p_contact_idx) const {
	return bone->collisions[p_contact_idx].appliedImpulse;
}

int BoneBulletPhysicsDirectBodyState::get_contact_local_shape(int p_contact_idx) const {
	return bone->collisions[p_contact_idx].local_shape;
}

RID BoneBulletPhysicsDirectBodyState::get_contact_collider(int p_contact_idx) const {
	return bone->collisions[p_contact_idx].otherObject->get_self();
}

Vector3 BoneBulletPhysicsDirectBodyState::get_contact_collider_position(int p_contact_idx) const {
	return bone->collisions[p_contact_idx].hitWorldLocation;
}

ObjectID BoneBulletPhysicsDirectBodyState::get_contact_collider_id(int p_contact_idx) const {
	return bone->collisions[p_contact_idx].otherObject->get_instance_id();
}

int BoneBulletPhysicsDirectBodyState::get_contact_collider_shape(int p_contact_idx) const {
	return bone->collisions[p_contact_idx].other_object_shape;
}

Vector3 BoneBulletPhysicsDirectBodyState::get_contact_collider_velocity_at_position(int p_contact_idx) const {
	BoneBullet::CollisionData &colDat = bone->collisions.write[p_contact_idx];

	btVector3 hitLocation;
	G_TO_B(colDat.hitLocalLocation, hitLocation);

	btCollisionObject *co = colDat.otherObject->get_bt_collision_object();
	btRigidBody *rb = btRigidBody::upcast(co);

	Vector3 velocityAtPoint;
	if (rb)
		B_TO_G(rb->getVelocityInLocalPoint(hitLocation), velocityAtPoint);

	return velocityAtPoint;
}

PhysicsDirectSpaceState *BoneBulletPhysicsDirectBodyState::get_space_state() {
	return bone->get_space()->get_direct_state();
}

BoneBullet::BoneBullet() :
		RigidCollisionObjectBullet(TYPE_BONE_BODY),
		bt_body(NULL),
		bt_joint_limiter(NULL),
		bt_joint_motor(NULL),
		armature(NULL),
		force_integration_callback(NULL),
		link_mass(1),
		link_half_size(0),
		parent_link_id(-1),
		disable_parent_collision(true),
		lower_limit(-45),
		upper_limit(45),
		is_motor_enabled(false),
		velocity_target(0, 0, 0),
		position_target(0),
		rotation_target(0, 0, 0),
		max_motor_impulse(1),
		error_reduction_parameter(1.),
		spring_constant(0.1),
		damping_constant(1.),
		maximum_error(FLT_MAX),
		is_root(false) {

	bt_body = bulletnew(
			btMultiBodyLinkCollider(
					NULL,
					-1));

	setupBulletCollisionObject(bt_body);
	reload_shapes();
}

BoneBullet::~BoneBullet() {
	if (force_integration_callback) {
		memdelete(force_integration_callback);
		force_integration_callback = NULL;
	}
	if (armature) {
		set_armature(NULL);
	}
}

void BoneBullet::reload_body() {
	if (space) {
		space->remove_bone(this);
		space->add_bone(this);
	}
}

void BoneBullet::set_space(SpaceBullet *p_space) {
	if (p_space == space)
		return;

	if (space) {
		space->remove_bone(this);

		if (bt_joint_limiter)
			space->remove_bone_joint_limit(this);

		if (bt_joint_motor)
			space->remove_bone_joint_motor(this);
	}

	space = p_space;

	if (space) {
		space->add_bone(this);

		if (bt_joint_limiter)
			space->add_bone_joint_limit(this);

		if (bt_joint_motor)
			space->add_bone_joint_motor(this);
	}
}

void BoneBullet::on_collision_filters_change() {
	if (space) {
		space->reload_collision_filters(this);
	}
}

void BoneBullet::on_collision_checker_end() {
	isTransformChanged = true;
}

void BoneBullet::dispatch_callbacks() {

	if (!force_integration_callback)
		return;

	BoneBulletPhysicsDirectBodyState *bodyDirect = BoneBulletPhysicsDirectBodyState::get_singleton(this);

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

void BoneBullet::on_enter_area(AreaBullet *p_area) {
}

void BoneBullet::main_shape_changed() {
	CRASH_COND(!get_main_shape())
	bt_body->setCollisionShape(get_main_shape());
	if (armature) {
		if (get_link_id() == -1)
			armature->update_base_inertia();
	}
}

void BoneBullet::set_transform__bullet(const btTransform &p_global_transform) {
	bt_body->setWorldTransform(p_global_transform);
	CollisionObjectBullet::set_transform__bullet(p_global_transform);

	if (is_root) {
		if (armature) {
			armature->set_transform__bullet(p_global_transform);
		}
	}
}

const btTransform &BoneBullet::get_transform__bullet() const {
	return bt_body->getWorldTransform();
}

btMultiBodyLinkCollider *BoneBullet::get_bt_body() const {
	return bt_body;
}

btMultiBodyConstraint *BoneBullet::get_bt_joint_limiter() const {
	return bt_joint_limiter;
}

btMultiBodyConstraint *BoneBullet::get_bt_joint_motor() const {
	return bt_joint_motor;
}

void BoneBullet::set_force_integration_callback(ObjectID p_id, const StringName &p_method, const Variant &p_udata) {

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

void BoneBullet::set_armature(ArmatureBullet *p_armature) {

	if (armature == p_armature)
		return;

	if (armature) {
		armature->remove_bone(get_link_id());
	}

	armature = p_armature;
	bt_body->m_link = -1;
	is_root = false;

	if (armature) {

		bt_body->m_multiBody = armature->get_bt_body();
		set_space(armature->get_space());

	} else {

		set_space(NULL);
		bt_body->m_multiBody = NULL;
	}

	update_joint_limits();
	update_joint_motor();
}

ArmatureBullet *BoneBullet::get_armature() const {
	return armature;
}

void BoneBullet::set_bone_id(int p_bone_id) {
	ERR_FAIL_COND(!armature);
	ERR_FAIL_COND(p_bone_id < 0);
	is_root = false;
	bt_body->m_link = p_bone_id - 1; // This is correct even if p_link_id is 0
	armature->set_bone(this);
	if (get_link_id() == -1)
		is_root = true;

	update_joint_limits();
	update_joint_motor();
}

int BoneBullet::get_bone_id() const {
	return get_link_id() + 1;
}

int BoneBullet::get_link_id() const {
	return bt_body->m_link;
}

void BoneBullet::set_parent_bone_id(int p_bone_id) {
	ERR_FAIL_COND(p_bone_id < 0);
	parent_link_id = p_bone_id - 1;
}

int BoneBullet::get_parent_bone_id() const {
	return get_parent_link_id() + 1;
}

int BoneBullet::get_parent_link_id() const {
	return parent_link_id;
}

void BoneBullet::set_joint_offset(const Transform &p_transform) {
	G_TO_B(p_transform, joint_offset);
	UNSCALE_BT_BASIS(joint_offset);
}

Transform BoneBullet::get_joint_offset() const {
	Transform t;
	B_TO_G(joint_offset, t);
	return t;
}

const btTransform &BoneBullet::get_joint_offset__bullet() const {
	return joint_offset;
}

btTransform BoneBullet::get_joint_offset_scaled__bullet() const {
	btTransform joint_offset_scaled;
	G_TO_B(get_joint_offset().scaled(get_body_scale()), joint_offset_scaled);
	return joint_offset_scaled;
}

void BoneBullet::set_link_mass(real_t p_link_mass) {
	ERR_FAIL_COND(p_link_mass < 0);
	link_mass = p_link_mass;
	if (armature)
		if (get_link_id() != -1)
			armature->update_link_mass_and_inertia(link_mass);
}

real_t BoneBullet::get_link_mass() const {
	return link_mass;
}

void BoneBullet::set_param(PhysicsServer::BodyParameter p_param, real_t p_value) {
	switch (p_param) {
		case PhysicsServer::BODY_PARAM_BOUNCE:
			bt_body->setRestitution(p_value);
			break;
		case PhysicsServer::BODY_PARAM_FRICTION:
			bt_body->setFriction(p_value);
			break;
		case PhysicsServer::BODY_PARAM_MASS:
		case PhysicsServer::BODY_PARAM_LINEAR_DAMP:
		case PhysicsServer::BODY_PARAM_ANGULAR_DAMP:
		case PhysicsServer::BODY_PARAM_GRAVITY_SCALE:
		default:
			WARN_PRINTS("Parameter " + itos(p_param) + " not supported by bullet. Value: " + itos(p_value));
	}
}

real_t BoneBullet::get_param(PhysicsServer::BodyParameter p_param) const {
	switch (p_param) {
		case PhysicsServer::BODY_PARAM_BOUNCE:
			return bt_body->getRestitution();
		case PhysicsServer::BODY_PARAM_FRICTION:
			return bt_body->getFriction();
		case PhysicsServer::BODY_PARAM_MASS:
		case PhysicsServer::BODY_PARAM_LINEAR_DAMP:
		case PhysicsServer::BODY_PARAM_ANGULAR_DAMP:
		case PhysicsServer::BODY_PARAM_GRAVITY_SCALE:
		default:
			WARN_PRINTS("Parameter " + itos(p_param) + " not supported by bullet");
			return 0;
	}
}

void BoneBullet::set_disable_parent_collision(bool p_disable) {
	disable_parent_collision = p_disable;

	if (armature) {
		if (get_link_id() >= 0) {
			if (disable_parent_collision) {
				armature->get_bt_body()->getLink(get_link_id()).m_flags |= BT_MULTIBODYLINKFLAGS_DISABLE_PARENT_COLLISION;
			} else {
				armature->get_bt_body()->getLink(get_link_id()).m_flags &= ~BT_MULTIBODYLINKFLAGS_DISABLE_PARENT_COLLISION;
			}
		}
	}
}

bool BoneBullet::get_disable_parent_collision() const {
	return disable_parent_collision;
}

void BoneBullet::set_limit_active(bool p_limit_active) {
	is_limit_active = p_limit_active;
	update_joint_limits();
}

bool BoneBullet::get_is_limit_active() const {
	return is_limit_active;
}

void BoneBullet::set_lower_limit(real_t p_lower_limit) {
	lower_limit = p_lower_limit;
	update_joint_limits();
}

real_t BoneBullet::get_lower_limit() const {
	return lower_limit;
}

void BoneBullet::set_upper_limit(real_t p_upper_limit) {
	upper_limit = p_upper_limit;
	update_joint_limits();
}

real_t BoneBullet::get_upper_limit() const {
	return upper_limit;
}

void BoneBullet::set_motor_enabled(bool p_v) {
	is_motor_enabled = p_v;
	update_joint_motor();
}
bool BoneBullet::get_motor_enabled() const {
	return is_motor_enabled;
}

void BoneBullet::set_velocity_target(const Vector3 &p_v) {
	G_TO_B(p_v, velocity_target);
	update_joint_motor_params();
}
Vector3 BoneBullet::get_velocity_target() const {
	Vector3 r;
	B_TO_G(velocity_target, r);
	return r;
}

void BoneBullet::set_position_target(real_t p_v) {
	position_target = p_v;
	update_joint_motor_params();
}
real_t BoneBullet::get_position_target() const {
	return position_target;
}

void BoneBullet::set_rotation_target(const Basis &p_v) {
	btMatrix3x3 m;
	G_TO_B(p_v, m);
	m.getRotation(rotation_target);
	update_joint_motor_params();
}
Basis BoneBullet::get_rotation_target() const {
	Basis r;
	B_TO_G(btMatrix3x3(rotation_target), r);
	return r;
}

void BoneBullet::set_max_motor_impulse(real_t p_v) {
	max_motor_impulse = p_v;
	update_joint_motor();
}
real_t BoneBullet::get_max_motor_impulse() const {
	return max_motor_impulse;
}

void BoneBullet::set_error_reduction_parameter(real_t p_v) {
	error_reduction_parameter = p_v;
	update_joint_motor_params();
}
real_t BoneBullet::get_error_reduction_parameter() const {
	return error_reduction_parameter;
}

void BoneBullet::set_spring_constant(real_t p_v) {
	spring_constant = p_v;
	update_joint_motor_params();
}
real_t BoneBullet::get_spring_constant() const {
	return spring_constant;
}

void BoneBullet::set_damping_constant(real_t p_v) {
	damping_constant = p_v;
	update_joint_motor_params();
}
real_t BoneBullet::get_damping_constant() const {
	return damping_constant;
}

void BoneBullet::set_maximum_error(real_t p_v) {
	maximum_error = p_v;
	update_joint_motor_params();
}
real_t BoneBullet::get_maximum_error() const {
	return maximum_error;
}

Vector3 BoneBullet::get_joint_force() const {

	if (get_link_id() == -1)
		return Vector3();

	Vector3 ret;
	B_TO_G(armature->get_bt_body()->getLink(get_link_id()).m_appliedConstraintForce, ret);
	return ret;
}

Vector3 BoneBullet::get_joint_torque() const {

	if (get_link_id() == -1)
		return Vector3();

	Vector3 ret;
	B_TO_G(armature->get_bt_body()->getLink(get_link_id()).m_appliedConstraintTorque, ret);
	return ret;
}

void BoneBullet::setup_joint_fixed() {

	if (get_link_id() == -1)
		return;

	ERR_FAIL_COND(armature->get_bone(get_parent_link_id()) == NULL);

	const btTransform joint_offset_scaled = get_joint_offset_scaled__bullet();
	const btTransform parent_transform = armature->get_bone(get_parent_link_id())->get_transform__bullet();
	const btTransform bone_transform = get_transform__bullet();
	const btTransform parent_to_this = parent_transform.inverse() * bone_transform;
	const btTransform parent_to_joint = parent_to_this * joint_offset_scaled;

	link_half_size = 0;

	// 0 Inertia for the fixed joints
	btVector3 inertia(0, 0, 0);

	btQuaternion q;
	(parent_to_this.getBasis() * joint_offset_scaled.getBasis()).getRotation(q);

	armature->get_bt_body()->setupFixed(
			get_link_id(),
			link_mass,
			inertia,
			get_parent_link_id(),

			parent_to_this.getRotation(),
			parent_to_joint.getOrigin(),
			joint_offset_scaled.getOrigin() * -1);
	armature->get_bt_body()->finalizeMultiDof();

	update_joint_limits();
	update_joint_motor();
}

void BoneBullet::setup_joint_prismatic() {

	if (get_link_id() == -1)
		return;

	ERR_FAIL_COND(armature->get_bone(get_parent_link_id()) == NULL);

	const btTransform joint_offset_scaled = get_joint_offset_scaled__bullet();
	const btTransform parent_transform = armature->get_bone(get_parent_link_id())->get_transform__bullet();
	const btTransform bone_transform = get_transform__bullet();
	const btTransform parent_to_this = parent_transform.inverse() * bone_transform;
	const btTransform parent_to_joint = parent_to_this * joint_offset_scaled;

	link_half_size = parent_to_this.getOrigin().length() * 0.5;

	// Calculates inertia along X
	btVector3 inertia(0, 0, 0);
	calc_link_inertia(btMultibodyLink::ePrismatic, link_mass, link_half_size, inertia);

	armature->get_bt_body()->setupPrismatic(
			get_link_id(),
			link_mass,
			inertia,
			get_parent_link_id(),

			parent_to_this.getRotation(),
			parent_to_joint.getBasis() * btVector3(1, 0, 0), // Slide along X
			parent_to_joint.getOrigin(),
			joint_offset_scaled.getOrigin() * -1,
			disable_parent_collision);
	armature->get_bt_body()->finalizeMultiDof();

	update_joint_limits();
	update_joint_motor();
}

void BoneBullet::setup_joint_revolute() {

	if (get_link_id() == -1)
		return;

	ERR_FAIL_COND(armature->get_bone(get_parent_link_id()) == NULL);

	const btTransform joint_offset_scaled = get_joint_offset_scaled__bullet();
	const btTransform parent_transform = armature->get_bone(get_parent_link_id())->get_transform__bullet();
	const btTransform bone_transform = get_transform__bullet();
	const btTransform parent_to_this = parent_transform.inverse() * bone_transform;
	const btTransform parent_to_joint = parent_to_this * joint_offset_scaled;

	link_half_size = parent_to_this.getOrigin().length() * 0.5;

	// Calculates inertia along YZ
	btVector3 inertia(0, 0, 0);
	calc_link_inertia(btMultibodyLink::eRevolute, link_mass, link_half_size, inertia);

	armature->get_bt_body()->setupRevolute(
			get_link_id(),
			link_mass,
			inertia,
			get_parent_link_id(),

			parent_to_this.getRotation(),
			parent_to_joint.getBasis() * btVector3(0, 0, 1), // Rotate along Z
			parent_to_joint.getOrigin(),
			joint_offset_scaled.getOrigin() * -1,
			disable_parent_collision);

	armature->get_bt_body()->finalizeMultiDof();

	update_joint_limits();
	update_joint_motor();
}

void BoneBullet::setup_joint_spherical() {

	if (get_link_id() == -1)
		return;

	ERR_FAIL_COND(armature->get_bone(get_parent_link_id()) == NULL);

	const btTransform joint_offset_scaled = get_joint_offset_scaled__bullet();
	const btTransform parent_transform = armature->get_bone(get_parent_link_id())->get_transform__bullet();
	const btTransform bone_transform = get_transform__bullet();
	const btTransform parent_to_this = parent_transform.inverse() * bone_transform;
	const btTransform parent_to_joint = parent_to_this * joint_offset_scaled;

	link_half_size = parent_to_this.getOrigin().length() * 0.5;

	// Calculates spherical inertia
	btVector3 inertia(0, 0, 0);
	calc_link_inertia(btMultibodyLink::eSpherical, link_mass, link_half_size, inertia);

	armature->get_bt_body()->setupSpherical(
			get_link_id(),
			link_mass,
			inertia,
			get_parent_link_id(),

			parent_to_this.getRotation(),
			parent_to_joint.getOrigin(),
			joint_offset_scaled.getOrigin() * -1,
			disable_parent_collision);
	armature->get_bt_body()->finalizeMultiDof();

	update_joint_limits();
	update_joint_motor();
}

void BoneBullet::setup_joint_planar() {

	if (get_link_id() == -1)
		return;

	ERR_FAIL_COND(armature->get_bone(get_parent_link_id()) == NULL);

	const btTransform joint_offset_scaled = get_joint_offset_scaled__bullet();
	const btTransform parent_transform = armature->get_bone(get_parent_link_id())->get_transform__bullet();
	const btTransform bone_transform = get_transform__bullet();
	const btTransform parent_to_this = parent_transform.inverse() * bone_transform;
	const btTransform parent_to_joint = parent_to_this * joint_offset_scaled;

	link_half_size = parent_to_this.getOrigin().length() * 0.5;

	// Calculates inertia along XZ
	btVector3 inertia(0, 0, 0);
	calc_link_inertia(btMultibodyLink::ePlanar, link_mass, link_half_size, inertia);

	armature->get_bt_body()->setupPlanar(
			get_link_id(),
			link_mass,
			inertia,
			get_parent_link_id(),

			parent_to_this.getRotation(),
			parent_to_joint.getBasis() * btVector3(0, 1, 0), // Allow rotation along Y so the plain is XZ
			parent_to_this.getOrigin(),
			disable_parent_collision);

	armature->get_bt_body()->finalizeMultiDof();

	update_joint_limits();
	update_joint_motor();
}

void BoneBullet::update_joint_limits() {

	if (space)
		if (bt_joint_limiter)
			space->remove_bone_joint_limit(this);

	bulletdelete(bt_joint_limiter);
	bt_joint_limiter = NULL;

	if (!is_limit_active)
		return;

	if (get_link_id() == -1)
		return;

	if (!space)
		return;

	real_t ll = lower_limit;
	real_t ul = upper_limit;

	switch (armature->get_bt_body()->getLink(get_link_id()).m_jointType) {
		case btMultibodyLink::eRevolute:
			ll = lower_limit * Math_PI / 180.f;
			ul = upper_limit * Math_PI / 180.f;
			break;
		case btMultibodyLink::ePrismatic:
		case btMultibodyLink::eSpherical:
		case btMultibodyLink::ePlanar:
		case btMultibodyLink::eFixed:
		case btMultibodyLink::eInvalid:
			break;
	}

	bt_joint_limiter = bulletnew(btMultiBodyJointLimitConstraint(
			armature->get_bt_body(),
			get_link_id(),
			ll,
			ul));

	space->add_bone_joint_limit(this);
}

void BoneBullet::update_joint_motor() {

	if (space)
		if (bt_joint_motor)
			space->remove_bone_joint_motor(this);

	bulletdelete(bt_joint_motor);
	bt_joint_motor = NULL;

	if (!is_motor_enabled)
		return;

	if (get_link_id() == -1)
		return;

	if (!space)
		return;

	switch (armature->get_bt_body()->getLink(get_link_id()).m_jointType) {
		case btMultibodyLink::eRevolute:
		case btMultibodyLink::ePrismatic:
			bt_joint_motor = bulletnew(btMultiBodyJointMotor(
					armature->get_bt_body(),
					get_link_id(),
					velocity_target.getX(),
					max_motor_impulse));

			break;
		case btMultibodyLink::eSpherical:

			bt_joint_motor = bulletnew(btMultiBodySphericalJointMotor(
					armature->get_bt_body(),
					get_link_id(),
					max_motor_impulse));

			break;
		case btMultibodyLink::ePlanar:
		case btMultibodyLink::eFixed:
		case btMultibodyLink::eInvalid:
			return; // Stop here
	}

	update_joint_motor_params();
	space->add_bone_joint_motor(this);
}

void BoneBullet::update_joint_motor_params() {

	if (!bt_joint_motor)
		return;

	if (!is_motor_enabled)
		return;

	if (get_link_id() == -1)
		return;

	if (!space)
		return;

	switch (armature->get_bt_body()->getLink(get_link_id()).m_jointType) {
		case btMultibodyLink::eRevolute: {

			btMultiBodyJointMotor *j = static_cast<btMultiBodyJointMotor *>(bt_joint_motor);
			j->setVelocityTarget(velocity_target.getX() * Math_PI / 180.f, damping_constant);
			j->setPositionTarget(position_target * Math_PI / 180.f, spring_constant);
			j->setErp(error_reduction_parameter);
			j->setRhsClamp(maximum_error);

		} break;
		case btMultibodyLink::ePrismatic: {

			btMultiBodyJointMotor *j = static_cast<btMultiBodyJointMotor *>(bt_joint_motor);
			j->setVelocityTarget(velocity_target.getX(), damping_constant);
			j->setPositionTarget(position_target, spring_constant);
			j->setErp(error_reduction_parameter);
			j->setRhsClamp(maximum_error);

		} break;
		case btMultibodyLink::eSpherical: {

			btMultiBodySphericalJointMotor *j = static_cast<btMultiBodySphericalJointMotor *>(bt_joint_motor);
			j->setVelocityTarget(velocity_target * Math_PI / 180.f, damping_constant);
			j->setPositionTarget(rotation_target, spring_constant);
			j->setErp(error_reduction_parameter);
			j->setRhsClamp(maximum_error);

		} break;
		case btMultibodyLink::ePlanar:
		case btMultibodyLink::eFixed:
		case btMultibodyLink::eInvalid:
			return; // Stop here
	}
}

void BoneBullet::calc_link_inertia(
		btMultibodyLink::eFeatherstoneJointType p_joint_type,
		real_t p_link_half_size,
		real_t p_link_mass,
		btVector3 &r_inertia) {

	r_inertia = btVector3(0, 0, 0);

	switch (p_joint_type) {
		case btMultibodyLink::eRevolute:

			btBoxShape(
					btVector3(0.1, p_link_half_size, p_link_half_size))
					.calculateLocalInertia(p_link_mass, r_inertia);

			break;
		case btMultibodyLink::ePrismatic:

			btBoxShape(
					btVector3(p_link_half_size, 0.1, 0.1))
					.calculateLocalInertia(p_link_mass, r_inertia);

			break;
		case btMultibodyLink::eSpherical:

			btSphereShape(p_link_half_size)
					.calculateLocalInertia(p_link_mass, r_inertia);

			break;
		case btMultibodyLink::ePlanar:

			btBoxShape(
					btVector3(p_link_half_size, 0.1, p_link_half_size))
					.calculateLocalInertia(p_link_mass, r_inertia);

			break;
		case btMultibodyLink::eFixed:
		case btMultibodyLink::eInvalid:
			break;
	}
}

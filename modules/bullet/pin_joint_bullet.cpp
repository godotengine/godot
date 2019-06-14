/*************************************************************************/
/*  pin_joint_bullet.cpp                                                 */
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

#include "pin_joint_bullet.h"

#include "armature_bullet.h"
#include "bullet_types_converter.h"
#include "rigid_body_bullet.h"

#include <BulletDynamics/ConstraintSolver/btPoint2PointConstraint.h>
#include <BulletDynamics/Featherstone/btMultiBodyPoint2Point.h>

/**
	@author AndreaCatania
*/

class GodotMultiBodyPoint2Point : public btMultiBodyPoint2Point {

public:
	GodotMultiBodyPoint2Point(
			btMultiBody *body,
			int link,
			btRigidBody *bodyB,
			const btVector3 &pivotInA,
			const btVector3 &pivotInB) :

			btMultiBodyPoint2Point(
					body,
					link,
					bodyB,
					pivotInA,
					pivotInB) {}

	void set_linkA(int p_link_id) {
		m_linkA = p_link_id;
	}

	btRigidBody *get_rigidBodyA() {
		return m_rigidBodyA;
	}

	btRigidBody *get_rigidBodyB() {
		return m_rigidBodyB;
	}

	void set_pivotInA(btVector3 &p_pivot) {
		m_pivotInA = p_pivot;
	}

	btVector3 &get_pivotInA() {
		return m_pivotInA;
	}

	btVector3 &get_pivotInB() {
		return m_pivotInB;
	}
};

PinJointBullet::PinJointBullet(BoneBullet *p_body_a, const Vector3 &p_pos_a, RigidBodyBullet *p_body_b, const Vector3 &p_pos_b) :
		JointBullet() {

	ERR_FAIL_COND(!p_body_a);
	ERR_FAIL_COND(!p_body_b);

	btVector3 btPivotA;
	btVector3 btPivotB;
	G_TO_B(p_pos_a * p_body_a->get_body_scale(), btPivotA);
	G_TO_B(p_pos_b * p_body_b->get_body_scale(), btPivotB);
	mb_p2pConstraint = bulletnew(GodotMultiBodyPoint2Point(
			p_body_a->get_armature()->get_bt_body(),
			p_body_a->get_link_id(),
			p_body_b->get_bt_rigid_body(),
			btPivotA,
			btPivotB));

	p2pConstraint = NULL;
	setup(
			mb_p2pConstraint,
			p_body_a,
			NULL);
}

PinJointBullet::PinJointBullet(RigidBodyBullet *p_body_a, const Vector3 &p_pos_a, RigidBodyBullet *p_body_b, const Vector3 &p_pos_b) :
		JointBullet() {
	if (p_body_b) {

		btVector3 btPivotA;
		btVector3 btPivotB;
		G_TO_B(p_pos_a * p_body_a->get_body_scale(), btPivotA);
		G_TO_B(p_pos_b * p_body_b->get_body_scale(), btPivotB);
		p2pConstraint = bulletnew(btPoint2PointConstraint(
				*p_body_a->get_bt_rigid_body(),
				*p_body_b->get_bt_rigid_body(),
				btPivotA,
				btPivotB));
	} else {
		btVector3 btPivotA;
		G_TO_B(p_pos_a, btPivotA);
		p2pConstraint = bulletnew(btPoint2PointConstraint(*p_body_a->get_bt_rigid_body(), btPivotA));
	}

	mb_p2pConstraint = NULL;
	setup(p2pConstraint);
}

PinJointBullet::~PinJointBullet() {}

void PinJointBullet::reload_internal() {
	if (mb_p2pConstraint) {
		mb_p2pConstraint->set_linkA(body_a->get_link_id());
	}
}

void PinJointBullet::set_param(PhysicsServer::PinJointParam p_param, real_t p_value) {
	if (p2pConstraint)
		switch (p_param) {
			case PhysicsServer::PIN_JOINT_BIAS:
				p2pConstraint->m_setting.m_tau = p_value;
				break;
			case PhysicsServer::PIN_JOINT_DAMPING:
				p2pConstraint->m_setting.m_damping = p_value;
				break;
			case PhysicsServer::PIN_JOINT_IMPULSE_CLAMP:
				p2pConstraint->m_setting.m_impulseClamp = p_value;
				break;
		}

	if (mb_p2pConstraint)
		switch (p_param) {
			case PhysicsServer::PIN_JOINT_BIAS:
				break;
			case PhysicsServer::PIN_JOINT_DAMPING:
				break;
			case PhysicsServer::PIN_JOINT_IMPULSE_CLAMP:
				break;
		}
}

real_t PinJointBullet::get_param(PhysicsServer::PinJointParam p_param) const {

	if (p2pConstraint)
		switch (p_param) {
			case PhysicsServer::PIN_JOINT_BIAS:
				return p2pConstraint->m_setting.m_tau;
			case PhysicsServer::PIN_JOINT_DAMPING:
				return p2pConstraint->m_setting.m_damping;
			case PhysicsServer::PIN_JOINT_IMPULSE_CLAMP:
				return p2pConstraint->m_setting.m_impulseClamp;
			default:
				ERR_EXPLAIN("This parameter " + itos(p_param) + " is deprecated");
				WARN_DEPRECATED
				return 0;
		}

	if (mb_p2pConstraint)
		return 0;

	return 0;
}

void PinJointBullet::setPivotInA(const Vector3 &p_pos) {
	btVector3 btVec;
	G_TO_B(p_pos, btVec);

	if (p2pConstraint)
		p2pConstraint->setPivotA(btVec);
	else
		mb_p2pConstraint->set_pivotInA(btVec);
}

void PinJointBullet::setPivotInB(const Vector3 &p_pos) {
	btVector3 btVec;
	G_TO_B(p_pos, btVec);

	if (p2pConstraint)
		p2pConstraint->setPivotB(btVec);
	else
		mb_p2pConstraint->setPivotInB(btVec);
}

Vector3 PinJointBullet::getPivotInA() {
	btVector3 vec;

	if (p2pConstraint)
		vec = p2pConstraint->getPivotInA();
	else
		vec = mb_p2pConstraint->get_pivotInA();

	Vector3 gVec;
	B_TO_G(vec, gVec);
	return gVec;
}

Vector3 PinJointBullet::getPivotInB() {
	btVector3 vec;

	if (p2pConstraint)
		vec = p2pConstraint->getPivotInB();
	else
		vec = mb_p2pConstraint->getPivotInB();

	Vector3 gVec;
	B_TO_G(vec, gVec);
	return gVec;
}

void PinJointBullet::clear_internal_joint() {
	JointBullet::clear_internal_joint();
	p2pConstraint = NULL;
	mb_p2pConstraint = NULL;
}

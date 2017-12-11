/*************************************************************************/
/*  generic_6dof_joint_bullet.cpp                                        */
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

#include "generic_6dof_joint_bullet.h"
#include "BulletDynamics/ConstraintSolver/btGeneric6DofConstraint.h"
#include "bullet_types_converter.h"
#include "bullet_utilities.h"
#include "rigid_body_bullet.h"

Generic6DOFJointBullet::Generic6DOFJointBullet(RigidBodyBullet *rbA, RigidBodyBullet *rbB, const Transform &frameInA, const Transform &frameInB, bool useLinearReferenceFrameA) :
		JointBullet() {

	Transform scaled_AFrame(frameInA.scaled(rbA->get_body_scale()));

	scaled_AFrame.basis.rotref_posscale_decomposition(scaled_AFrame.basis);

	btTransform btFrameA;
	G_TO_B(scaled_AFrame, btFrameA);

	if (rbB) {
		Transform scaled_BFrame(frameInB.scaled(rbB->get_body_scale()));

		scaled_BFrame.basis.rotref_posscale_decomposition(scaled_BFrame.basis);

		btTransform btFrameB;
		G_TO_B(scaled_BFrame, btFrameB);

		sixDOFConstraint = bulletnew(btGeneric6DofConstraint(*rbA->get_bt_rigid_body(), *rbB->get_bt_rigid_body(), btFrameA, btFrameB, useLinearReferenceFrameA));
	} else {
		sixDOFConstraint = bulletnew(btGeneric6DofConstraint(*rbA->get_bt_rigid_body(), btFrameA, useLinearReferenceFrameA));
	}

	setup(sixDOFConstraint);
}

Transform Generic6DOFJointBullet::getFrameOffsetA() const {
	btTransform btTrs = sixDOFConstraint->getFrameOffsetA();
	Transform gTrs;
	B_TO_G(btTrs, gTrs);
	return gTrs;
}

Transform Generic6DOFJointBullet::getFrameOffsetB() const {
	btTransform btTrs = sixDOFConstraint->getFrameOffsetB();
	Transform gTrs;
	B_TO_G(btTrs, gTrs);
	return gTrs;
}

Transform Generic6DOFJointBullet::getFrameOffsetA() {
	btTransform btTrs = sixDOFConstraint->getFrameOffsetA();
	Transform gTrs;
	B_TO_G(btTrs, gTrs);
	return gTrs;
}

Transform Generic6DOFJointBullet::getFrameOffsetB() {
	btTransform btTrs = sixDOFConstraint->getFrameOffsetB();
	Transform gTrs;
	B_TO_G(btTrs, gTrs);
	return gTrs;
}

void Generic6DOFJointBullet::set_linear_lower_limit(const Vector3 &linearLower) {
	btVector3 btVec;
	G_TO_B(linearLower, btVec);
	sixDOFConstraint->setLinearLowerLimit(btVec);
}

void Generic6DOFJointBullet::set_linear_upper_limit(const Vector3 &linearUpper) {
	btVector3 btVec;
	G_TO_B(linearUpper, btVec);
	sixDOFConstraint->setLinearUpperLimit(btVec);
}

void Generic6DOFJointBullet::set_angular_lower_limit(const Vector3 &angularLower) {
	btVector3 btVec;
	G_TO_B(angularLower, btVec);
	sixDOFConstraint->setAngularLowerLimit(btVec);
}

void Generic6DOFJointBullet::set_angular_upper_limit(const Vector3 &angularUpper) {
	btVector3 btVec;
	G_TO_B(angularUpper, btVec);
	sixDOFConstraint->setAngularUpperLimit(btVec);
}

void Generic6DOFJointBullet::set_param(Vector3::Axis p_axis, PhysicsServer::G6DOFJointAxisParam p_param, real_t p_value) {
	ERR_FAIL_INDEX(p_axis, 3);
	switch (p_param) {
		case PhysicsServer::G6DOF_JOINT_LINEAR_LOWER_LIMIT:
			limits_lower[0][p_axis] = p_value;
			set_flag(p_axis, PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT, flags[p_axis][p_param]); // Reload bullet parameter
			break;
		case PhysicsServer::G6DOF_JOINT_LINEAR_UPPER_LIMIT:
			limits_upper[0][p_axis] = p_value;
			set_flag(p_axis, PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT, flags[p_axis][p_param]); // Reload bullet parameter
			break;
		case PhysicsServer::G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS:
			sixDOFConstraint->getTranslationalLimitMotor()->m_limitSoftness = p_value;
			break;
		case PhysicsServer::G6DOF_JOINT_LINEAR_RESTITUTION:
			sixDOFConstraint->getTranslationalLimitMotor()->m_restitution = p_value;
			break;
		case PhysicsServer::G6DOF_JOINT_LINEAR_DAMPING:
			sixDOFConstraint->getTranslationalLimitMotor()->m_damping = p_value;
			break;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_LOWER_LIMIT:
			limits_lower[1][p_axis] = p_value;
			set_flag(p_axis, PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT, flags[p_axis][p_param]); // Reload bullet parameter
			break;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_UPPER_LIMIT:
			limits_upper[1][p_axis] = p_value;
			set_flag(p_axis, PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT, flags[p_axis][p_param]); // Reload bullet parameter
			break;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_LIMIT_SOFTNESS:
			sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_limitSoftness = p_value;
			break;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_DAMPING:
			sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_damping = p_value;
			break;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_RESTITUTION:
			sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_bounce = p_value;
			break;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_FORCE_LIMIT:
			sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_maxLimitForce = p_value;
			break;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_ERP:
			sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_stopERP = p_value;
			break;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_MOTOR_TARGET_VELOCITY:
			sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_targetVelocity = p_value;
			break;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_MOTOR_FORCE_LIMIT:
			sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_maxLimitForce = p_value;
			break;
		default:
			WARN_PRINT("This parameter is not supported");
	}
}

real_t Generic6DOFJointBullet::get_param(Vector3::Axis p_axis, PhysicsServer::G6DOFJointAxisParam p_param) const {
	ERR_FAIL_INDEX_V(p_axis, 3, 0.);
	switch (p_param) {
		case PhysicsServer::G6DOF_JOINT_LINEAR_LOWER_LIMIT:
			return limits_lower[0][p_axis];
		case PhysicsServer::G6DOF_JOINT_LINEAR_UPPER_LIMIT:
			return limits_upper[0][p_axis];
		case PhysicsServer::G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS:
			return sixDOFConstraint->getTranslationalLimitMotor()->m_limitSoftness;
		case PhysicsServer::G6DOF_JOINT_LINEAR_RESTITUTION:
			return sixDOFConstraint->getTranslationalLimitMotor()->m_restitution;
		case PhysicsServer::G6DOF_JOINT_LINEAR_DAMPING:
			return sixDOFConstraint->getTranslationalLimitMotor()->m_damping;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_LOWER_LIMIT:
			return limits_lower[1][p_axis];
		case PhysicsServer::G6DOF_JOINT_ANGULAR_UPPER_LIMIT:
			return limits_upper[1][p_axis];
		case PhysicsServer::G6DOF_JOINT_ANGULAR_LIMIT_SOFTNESS:
			return sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_limitSoftness;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_DAMPING:
			return sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_damping;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_RESTITUTION:
			return sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_bounce;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_FORCE_LIMIT:
			return sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_maxLimitForce;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_ERP:
			return sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_stopERP;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_MOTOR_TARGET_VELOCITY:
			return sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_targetVelocity;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_MOTOR_FORCE_LIMIT:
			return sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_maxLimitForce;
		default:
			WARN_PRINT("This parameter is not supported");
			return 0.;
	}
}

void Generic6DOFJointBullet::set_flag(Vector3::Axis p_axis, PhysicsServer::G6DOFJointAxisFlag p_flag, bool p_value) {
	ERR_FAIL_INDEX(p_axis, 3);

	flags[p_axis][p_flag] = p_value;

	switch (p_flag) {
		case PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT:
			if (flags[p_axis][p_flag]) {
				sixDOFConstraint->setLimit(p_axis, limits_lower[0][p_axis], limits_upper[0][p_axis]);
			} else {
				sixDOFConstraint->setLimit(p_axis, 0, -1); // Free
			}
			break;
		case PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT:
			if (flags[p_axis][p_flag]) {
				sixDOFConstraint->setLimit(p_axis + 3, limits_lower[1][p_axis], limits_upper[1][p_axis]);
			} else {
				sixDOFConstraint->setLimit(p_axis + 3, 0, -1); // Free
			}
			break;
		case PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_MOTOR:
			sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_enableMotor = flags[p_axis][p_flag];
			break;
		default:
			WARN_PRINT("This flag is not supported by Bullet engine");
			return;
	}
}

bool Generic6DOFJointBullet::get_flag(Vector3::Axis p_axis, PhysicsServer::G6DOFJointAxisFlag p_flag) const {
	ERR_FAIL_INDEX_V(p_axis, 3, false);

	return flags[p_axis][p_flag];
}

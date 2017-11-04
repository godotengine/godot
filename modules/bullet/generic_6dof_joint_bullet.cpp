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

Generic6DOFJointBullet::Generic6DOFJointBullet(RigidBodyBullet *rbA, RigidBodyBullet *rbB, const Transform &frameInA, const Transform &frameInB, bool useLinearReferenceFrameA)
	: JointBullet() {

	btTransform btFrameA;
	G_TO_B(frameInA, btFrameA);

	if (rbB) {
		btTransform btFrameB;
		G_TO_B(frameInB, btFrameB);

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
			sixDOFConstraint->getTranslationalLimitMotor()->m_lowerLimit[p_axis] = p_value;
			break;
		case PhysicsServer::G6DOF_JOINT_LINEAR_UPPER_LIMIT:
			sixDOFConstraint->getTranslationalLimitMotor()->m_upperLimit[p_axis] = p_value;
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
			sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_loLimit = p_value;
			break;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_UPPER_LIMIT:
			sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_hiLimit = p_value;
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
			return sixDOFConstraint->getTranslationalLimitMotor()->m_lowerLimit[p_axis];
		case PhysicsServer::G6DOF_JOINT_LINEAR_UPPER_LIMIT:
			return sixDOFConstraint->getTranslationalLimitMotor()->m_upperLimit[p_axis];
		case PhysicsServer::G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS:
			return sixDOFConstraint->getTranslationalLimitMotor()->m_limitSoftness;
		case PhysicsServer::G6DOF_JOINT_LINEAR_RESTITUTION:
			return sixDOFConstraint->getTranslationalLimitMotor()->m_restitution;
		case PhysicsServer::G6DOF_JOINT_LINEAR_DAMPING:
			return sixDOFConstraint->getTranslationalLimitMotor()->m_damping;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_LOWER_LIMIT:
			return sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_loLimit;
		case PhysicsServer::G6DOF_JOINT_ANGULAR_UPPER_LIMIT:
			return sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_hiLimit;
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
	switch (p_flag) {
		case PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT:
			if (p_value) {
				if (!get_flag(p_axis, p_flag)) // avoid overwrite, if limited
					sixDOFConstraint->setLimit(p_axis, 0, 0); // Limited
			} else {
				if (get_flag(p_axis, p_flag)) // avoid overwrite, if free
					sixDOFConstraint->setLimit(p_axis, 0, -1); // Free
			}
			break;
		case PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT: {
			int angularAxis = 3 + p_axis;
			if (p_value) {
				if (!get_flag(p_axis, p_flag)) // avoid overwrite, if Limited
					sixDOFConstraint->setLimit(angularAxis, 0, 0); // Limited
			} else {
				if (get_flag(p_axis, p_flag)) // avoid overwrite, if free
					sixDOFConstraint->setLimit(angularAxis, 0, -1); // Free
			}
			break;
		}
		case PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_MOTOR:
			//sixDOFConstraint->getTranslationalLimitMotor()->m_enableMotor[p_axis] = p_value;
			sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_enableMotor = p_value;
			break;
		default:
			WARN_PRINT("This flag is not supported by Bullet engine");
	}
}

bool Generic6DOFJointBullet::get_flag(Vector3::Axis p_axis, PhysicsServer::G6DOFJointAxisFlag p_flag) const {
	ERR_FAIL_INDEX_V(p_axis, 3, false);
	switch (p_flag) {
		case PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT:
			return sixDOFConstraint->getTranslationalLimitMotor()->isLimited(p_axis);
		case PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT:
			return sixDOFConstraint->getRotationalLimitMotor(p_axis)->isLimited();
		case PhysicsServer::G6DOF_JOINT_FLAG_ENABLE_MOTOR:
			return //sixDOFConstraint->getTranslationalLimitMotor()->m_enableMotor[p_axis] &&
					sixDOFConstraint->getRotationalLimitMotor(p_axis)->m_enableMotor;
		default:
			WARN_PRINT("This flag is not supported by Bullet engine");
			return false;
	}
}

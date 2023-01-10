/**************************************************************************/
/*  slider_joint_sw.cpp                                                   */
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

/*
Adapted to Godot from the Bullet library.
*/

/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

/*
Added by Roman Ponomarev (rponom@gmail.com)
April 04, 2008

*/

#include "slider_joint_sw.h"

//-----------------------------------------------------------------------------

static _FORCE_INLINE_ real_t atan2fast(real_t y, real_t x) {
	real_t coeff_1 = Math_PI / 4.0f;
	real_t coeff_2 = 3.0f * coeff_1;
	real_t abs_y = Math::abs(y);
	real_t angle;
	if (x >= 0.0f) {
		real_t r = (x - abs_y) / (x + abs_y);
		angle = coeff_1 - coeff_1 * r;
	} else {
		real_t r = (x + abs_y) / (abs_y - x);
		angle = coeff_2 - coeff_1 * r;
	}
	return (y < 0.0f) ? -angle : angle;
}

void SliderJointSW::initParams() {
	m_lowerLinLimit = real_t(1.0);
	m_upperLinLimit = real_t(-1.0);
	m_lowerAngLimit = real_t(0.);
	m_upperAngLimit = real_t(0.);
	m_softnessDirLin = SLIDER_CONSTRAINT_DEF_SOFTNESS;
	m_restitutionDirLin = SLIDER_CONSTRAINT_DEF_RESTITUTION;
	m_dampingDirLin = real_t(0.);
	m_softnessDirAng = SLIDER_CONSTRAINT_DEF_SOFTNESS;
	m_restitutionDirAng = SLIDER_CONSTRAINT_DEF_RESTITUTION;
	m_dampingDirAng = real_t(0.);
	m_softnessOrthoLin = SLIDER_CONSTRAINT_DEF_SOFTNESS;
	m_restitutionOrthoLin = SLIDER_CONSTRAINT_DEF_RESTITUTION;
	m_dampingOrthoLin = SLIDER_CONSTRAINT_DEF_DAMPING;
	m_softnessOrthoAng = SLIDER_CONSTRAINT_DEF_SOFTNESS;
	m_restitutionOrthoAng = SLIDER_CONSTRAINT_DEF_RESTITUTION;
	m_dampingOrthoAng = SLIDER_CONSTRAINT_DEF_DAMPING;
	m_softnessLimLin = SLIDER_CONSTRAINT_DEF_SOFTNESS;
	m_restitutionLimLin = SLIDER_CONSTRAINT_DEF_RESTITUTION;
	m_dampingLimLin = SLIDER_CONSTRAINT_DEF_DAMPING;
	m_softnessLimAng = SLIDER_CONSTRAINT_DEF_SOFTNESS;
	m_restitutionLimAng = SLIDER_CONSTRAINT_DEF_RESTITUTION;
	m_dampingLimAng = SLIDER_CONSTRAINT_DEF_DAMPING;

	m_poweredLinMotor = false;
	m_targetLinMotorVelocity = real_t(0.);
	m_maxLinMotorForce = real_t(0.);
	m_accumulatedLinMotorImpulse = real_t(0.0);

	m_poweredAngMotor = false;
	m_targetAngMotorVelocity = real_t(0.);
	m_maxAngMotorForce = real_t(0.);
	m_accumulatedAngMotorImpulse = real_t(0.0);

} // SliderJointSW::initParams()

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

SliderJointSW::SliderJointSW(BodySW *rbA, BodySW *rbB, const Transform &frameInA, const Transform &frameInB) :
		JointSW(_arr, 2),
		m_frameInA(frameInA),
		m_frameInB(frameInB) {
	A = rbA;
	B = rbB;

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);

	initParams();
} // SliderJointSW::SliderJointSW()

//-----------------------------------------------------------------------------

bool SliderJointSW::setup(real_t p_step) {
	if ((A->get_mode() <= PhysicsServer::BODY_MODE_KINEMATIC) && (B->get_mode() <= PhysicsServer::BODY_MODE_KINEMATIC)) {
		return false;
	}

	//calculate transforms
	m_calculatedTransformA = A->get_transform() * m_frameInA;
	m_calculatedTransformB = B->get_transform() * m_frameInB;
	m_realPivotAInW = m_calculatedTransformA.origin;
	m_realPivotBInW = m_calculatedTransformB.origin;
	m_sliderAxis = m_calculatedTransformA.basis.get_axis(0); // along X
	m_delta = m_realPivotBInW - m_realPivotAInW;
	m_projPivotInW = m_realPivotAInW + m_sliderAxis.dot(m_delta) * m_sliderAxis;
	m_relPosA = m_projPivotInW - A->get_transform().origin;
	m_relPosB = m_realPivotBInW - B->get_transform().origin;
	Vector3 normalWorld;
	int i;
	//linear part
	for (i = 0; i < 3; i++) {
		normalWorld = m_calculatedTransformA.basis.get_axis(i);
		memnew_placement(
				&m_jacLin[i],
				JacobianEntrySW(
						A->get_principal_inertia_axes().transposed(),
						B->get_principal_inertia_axes().transposed(),
						m_relPosA - A->get_center_of_mass(), m_relPosB - B->get_center_of_mass(),
						normalWorld,
						A->get_inv_inertia(),
						A->get_inv_mass(),
						B->get_inv_inertia(),
						B->get_inv_mass()));
		m_jacLinDiagABInv[i] = real_t(1.) / m_jacLin[i].getDiagonal();
		m_depth[i] = m_delta.dot(normalWorld);
	}
	testLinLimits();
	// angular part
	for (i = 0; i < 3; i++) {
		normalWorld = m_calculatedTransformA.basis.get_axis(i);
		memnew_placement(
				&m_jacAng[i],
				JacobianEntrySW(
						normalWorld,
						A->get_principal_inertia_axes().transposed(),
						B->get_principal_inertia_axes().transposed(),
						A->get_inv_inertia(),
						B->get_inv_inertia()));
	}
	testAngLimits();
	Vector3 axisA = m_calculatedTransformA.basis.get_axis(0);
	m_kAngle = real_t(1.0) / (A->compute_angular_impulse_denominator(axisA) + B->compute_angular_impulse_denominator(axisA));
	// clear accumulator for motors
	m_accumulatedLinMotorImpulse = real_t(0.0);
	m_accumulatedAngMotorImpulse = real_t(0.0);

	return true;
} // SliderJointSW::buildJacobianInt()

//-----------------------------------------------------------------------------

void SliderJointSW::solve(real_t p_step) {
	int i;
	// linear
	Vector3 velA = A->get_velocity_in_local_point(m_relPosA);
	Vector3 velB = B->get_velocity_in_local_point(m_relPosB);
	Vector3 vel = velA - velB;
	for (i = 0; i < 3; i++) {
		const Vector3 &normal = m_jacLin[i].m_linearJointAxis;
		real_t rel_vel = normal.dot(vel);
		// calculate positional error
		real_t depth = m_depth[i];
		// get parameters
		real_t softness = (i) ? m_softnessOrthoLin : (m_solveLinLim ? m_softnessLimLin : m_softnessDirLin);
		real_t restitution = (i) ? m_restitutionOrthoLin : (m_solveLinLim ? m_restitutionLimLin : m_restitutionDirLin);
		real_t damping = (i) ? m_dampingOrthoLin : (m_solveLinLim ? m_dampingLimLin : m_dampingDirLin);
		// calculate and apply impulse
		real_t normalImpulse = softness * (restitution * depth / p_step - damping * rel_vel) * m_jacLinDiagABInv[i];
		Vector3 impulse_vector = normal * normalImpulse;
		A->apply_impulse(m_relPosA, impulse_vector);
		B->apply_impulse(m_relPosB, -impulse_vector);
		if (m_poweredLinMotor && (!i)) { // apply linear motor
			if (m_accumulatedLinMotorImpulse < m_maxLinMotorForce) {
				real_t desiredMotorVel = m_targetLinMotorVelocity;
				real_t motor_relvel = desiredMotorVel + rel_vel;
				normalImpulse = -motor_relvel * m_jacLinDiagABInv[i];
				// clamp accumulated impulse
				real_t new_acc = m_accumulatedLinMotorImpulse + Math::abs(normalImpulse);
				if (new_acc > m_maxLinMotorForce) {
					new_acc = m_maxLinMotorForce;
				}
				real_t del = new_acc - m_accumulatedLinMotorImpulse;
				if (normalImpulse < real_t(0.0)) {
					normalImpulse = -del;
				} else {
					normalImpulse = del;
				}
				m_accumulatedLinMotorImpulse = new_acc;
				// apply clamped impulse
				impulse_vector = normal * normalImpulse;
				A->apply_impulse(m_relPosA, impulse_vector);
				B->apply_impulse(m_relPosB, -impulse_vector);
			}
		}
	}
	// angular
	// get axes in world space
	Vector3 axisA = m_calculatedTransformA.basis.get_axis(0);
	Vector3 axisB = m_calculatedTransformB.basis.get_axis(0);

	const Vector3 &angVelA = A->get_angular_velocity();
	const Vector3 &angVelB = B->get_angular_velocity();

	Vector3 angVelAroundAxisA = axisA * axisA.dot(angVelA);
	Vector3 angVelAroundAxisB = axisB * axisB.dot(angVelB);

	Vector3 angAorthog = angVelA - angVelAroundAxisA;
	Vector3 angBorthog = angVelB - angVelAroundAxisB;
	Vector3 velrelOrthog = angAorthog - angBorthog;
	//solve orthogonal angular velocity correction
	real_t len = velrelOrthog.length();
	if (len > real_t(0.00001)) {
		Vector3 normal = velrelOrthog.normalized();
		real_t denom = A->compute_angular_impulse_denominator(normal) + B->compute_angular_impulse_denominator(normal);
		velrelOrthog *= (real_t(1.) / denom) * m_dampingOrthoAng * m_softnessOrthoAng;
	}
	//solve angular positional correction
	Vector3 angularError = axisA.cross(axisB) * (real_t(1.) / p_step);
	real_t len2 = angularError.length();
	if (len2 > real_t(0.00001)) {
		Vector3 normal2 = angularError.normalized();
		real_t denom2 = A->compute_angular_impulse_denominator(normal2) + B->compute_angular_impulse_denominator(normal2);
		angularError *= (real_t(1.) / denom2) * m_restitutionOrthoAng * m_softnessOrthoAng;
	}
	// apply impulse
	A->apply_torque_impulse(-velrelOrthog + angularError);
	B->apply_torque_impulse(velrelOrthog - angularError);
	real_t impulseMag;
	//solve angular limits
	if (m_solveAngLim) {
		impulseMag = (angVelB - angVelA).dot(axisA) * m_dampingLimAng + m_angDepth * m_restitutionLimAng / p_step;
		impulseMag *= m_kAngle * m_softnessLimAng;
	} else {
		impulseMag = (angVelB - angVelA).dot(axisA) * m_dampingDirAng + m_angDepth * m_restitutionDirAng / p_step;
		impulseMag *= m_kAngle * m_softnessDirAng;
	}
	Vector3 impulse = axisA * impulseMag;
	A->apply_torque_impulse(impulse);
	B->apply_torque_impulse(-impulse);
	//apply angular motor
	if (m_poweredAngMotor) {
		if (m_accumulatedAngMotorImpulse < m_maxAngMotorForce) {
			Vector3 velrel = angVelAroundAxisA - angVelAroundAxisB;
			real_t projRelVel = velrel.dot(axisA);

			real_t desiredMotorVel = m_targetAngMotorVelocity;
			real_t motor_relvel = desiredMotorVel - projRelVel;

			real_t angImpulse = m_kAngle * motor_relvel;
			// clamp accumulated impulse
			real_t new_acc = m_accumulatedAngMotorImpulse + Math::abs(angImpulse);
			if (new_acc > m_maxAngMotorForce) {
				new_acc = m_maxAngMotorForce;
			}
			real_t del = new_acc - m_accumulatedAngMotorImpulse;
			if (angImpulse < real_t(0.0)) {
				angImpulse = -del;
			} else {
				angImpulse = del;
			}
			m_accumulatedAngMotorImpulse = new_acc;
			// apply clamped impulse
			Vector3 motorImp = angImpulse * axisA;
			A->apply_torque_impulse(motorImp);
			B->apply_torque_impulse(-motorImp);
		}
	}
} // SliderJointSW::solveConstraint()

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

void SliderJointSW::calculateTransforms() {
	m_calculatedTransformA = A->get_transform() * m_frameInA;
	m_calculatedTransformB = B->get_transform() * m_frameInB;
	m_realPivotAInW = m_calculatedTransformA.origin;
	m_realPivotBInW = m_calculatedTransformB.origin;
	m_sliderAxis = m_calculatedTransformA.basis.get_axis(0); // along X
	m_delta = m_realPivotBInW - m_realPivotAInW;
	m_projPivotInW = m_realPivotAInW + m_sliderAxis.dot(m_delta) * m_sliderAxis;
	Vector3 normalWorld;
	int i;
	//linear part
	for (i = 0; i < 3; i++) {
		normalWorld = m_calculatedTransformA.basis.get_axis(i);
		m_depth[i] = m_delta.dot(normalWorld);
	}
} // SliderJointSW::calculateTransforms()

//-----------------------------------------------------------------------------

void SliderJointSW::testLinLimits() {
	m_solveLinLim = false;
	m_linPos = m_depth[0];
	if (m_lowerLinLimit <= m_upperLinLimit) {
		if (m_depth[0] > m_upperLinLimit) {
			m_depth[0] -= m_upperLinLimit;
			m_solveLinLim = true;
		} else if (m_depth[0] < m_lowerLinLimit) {
			m_depth[0] -= m_lowerLinLimit;
			m_solveLinLim = true;
		} else {
			m_depth[0] = real_t(0.);
		}
	} else {
		m_depth[0] = real_t(0.);
	}
} // SliderJointSW::testLinLimits()

//-----------------------------------------------------------------------------

void SliderJointSW::testAngLimits() {
	m_angDepth = real_t(0.);
	m_solveAngLim = false;
	if (m_lowerAngLimit <= m_upperAngLimit) {
		const Vector3 axisA0 = m_calculatedTransformA.basis.get_axis(1);
		const Vector3 axisA1 = m_calculatedTransformA.basis.get_axis(2);
		const Vector3 axisB0 = m_calculatedTransformB.basis.get_axis(1);
		real_t rot = atan2fast(axisB0.dot(axisA1), axisB0.dot(axisA0));
		if (rot < m_lowerAngLimit) {
			m_angDepth = rot - m_lowerAngLimit;
			m_solveAngLim = true;
		} else if (rot > m_upperAngLimit) {
			m_angDepth = rot - m_upperAngLimit;
			m_solveAngLim = true;
		}
	}
} // SliderJointSW::testAngLimits()

//-----------------------------------------------------------------------------

Vector3 SliderJointSW::getAncorInA() {
	Vector3 ancorInA;
	ancorInA = m_realPivotAInW + (m_lowerLinLimit + m_upperLinLimit) * real_t(0.5) * m_sliderAxis;
	ancorInA = A->get_transform().inverse().xform(ancorInA);
	return ancorInA;
} // SliderJointSW::getAncorInA()

//-----------------------------------------------------------------------------

Vector3 SliderJointSW::getAncorInB() {
	Vector3 ancorInB;
	ancorInB = m_frameInB.origin;
	return ancorInB;
} // SliderJointSW::getAncorInB();

void SliderJointSW::set_param(PhysicsServer::SliderJointParam p_param, real_t p_value) {
	switch (p_param) {
		case PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_UPPER:
			m_upperLinLimit = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_LOWER:
			m_lowerLinLimit = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS:
			m_softnessLimLin = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION:
			m_restitutionLimLin = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_DAMPING:
			m_dampingLimLin = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_LINEAR_MOTION_SOFTNESS:
			m_softnessDirLin = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_LINEAR_MOTION_RESTITUTION:
			m_restitutionDirLin = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_LINEAR_MOTION_DAMPING:
			m_dampingDirLin = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_LINEAR_ORTHOGONAL_SOFTNESS:
			m_softnessOrthoLin = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_LINEAR_ORTHOGONAL_RESTITUTION:
			m_restitutionOrthoLin = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_LINEAR_ORTHOGONAL_DAMPING:
			m_dampingOrthoLin = p_value;
			break;

		case PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_UPPER:
			m_upperAngLimit = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_LOWER:
			m_lowerAngLimit = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS:
			m_softnessLimAng = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_RESTITUTION:
			m_restitutionLimAng = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_DAMPING:
			m_dampingLimAng = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_MOTION_SOFTNESS:
			m_softnessDirAng = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_MOTION_RESTITUTION:
			m_restitutionDirAng = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_MOTION_DAMPING:
			m_dampingDirAng = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_ORTHOGONAL_SOFTNESS:
			m_softnessOrthoAng = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_ORTHOGONAL_RESTITUTION:
			m_restitutionOrthoAng = p_value;
			break;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_ORTHOGONAL_DAMPING:
			m_dampingOrthoAng = p_value;
			break;

		case PhysicsServer::SLIDER_JOINT_MAX:
			break; // Can't happen, but silences warning
	}
}

real_t SliderJointSW::get_param(PhysicsServer::SliderJointParam p_param) const {
	switch (p_param) {
		case PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_UPPER:
			return m_upperLinLimit;
		case PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_LOWER:
			return m_lowerLinLimit;
		case PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS:
			return m_softnessLimLin;
		case PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION:
			return m_restitutionLimLin;
		case PhysicsServer::SLIDER_JOINT_LINEAR_LIMIT_DAMPING:
			return m_dampingLimLin;
		case PhysicsServer::SLIDER_JOINT_LINEAR_MOTION_SOFTNESS:
			return m_softnessDirLin;
		case PhysicsServer::SLIDER_JOINT_LINEAR_MOTION_RESTITUTION:
			return m_restitutionDirLin;
		case PhysicsServer::SLIDER_JOINT_LINEAR_MOTION_DAMPING:
			return m_dampingDirLin;
		case PhysicsServer::SLIDER_JOINT_LINEAR_ORTHOGONAL_SOFTNESS:
			return m_softnessOrthoLin;
		case PhysicsServer::SLIDER_JOINT_LINEAR_ORTHOGONAL_RESTITUTION:
			return m_restitutionOrthoLin;
		case PhysicsServer::SLIDER_JOINT_LINEAR_ORTHOGONAL_DAMPING:
			return m_dampingOrthoLin;

		case PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_UPPER:
			return m_upperAngLimit;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_LOWER:
			return m_lowerAngLimit;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS:
			return m_softnessLimAng;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_RESTITUTION:
			return m_restitutionLimAng;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_LIMIT_DAMPING:
			return m_dampingLimAng;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_MOTION_SOFTNESS:
			return m_softnessDirAng;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_MOTION_RESTITUTION:
			return m_restitutionDirAng;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_MOTION_DAMPING:
			return m_dampingDirAng;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_ORTHOGONAL_SOFTNESS:
			return m_softnessOrthoAng;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_ORTHOGONAL_RESTITUTION:
			return m_restitutionOrthoAng;
		case PhysicsServer::SLIDER_JOINT_ANGULAR_ORTHOGONAL_DAMPING:
			return m_dampingOrthoAng;

		case PhysicsServer::SLIDER_JOINT_MAX:
			break; // Can't happen, but silences warning
	}

	return 0;
}

/*************************************************************************/
/*  joints_3d_sw.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

/*
Adapted to Godot from the Bullet library.
*/

/*
Bullet Continuous Collision Detection and Physics Library

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "joints_3d_sw.h"

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

static void plane_space(const Vector3 &n, Vector3 &p, Vector3 &q) {
	if (Math::abs(n.z) > Math_SQRT12) {
		// choose p in y-z plane
		real_t a = n[1] * n[1] + n[2] * n[2];
		real_t k = 1.0 / Math::sqrt(a);
		p = Vector3(0, -n[2] * k, n[1] * k);
		// set q = n x p
		q = Vector3(a * k, -n[0] * p[2], n[0] * p[1]);
	} else {
		// choose p in x-y plane
		real_t a = n.x * n.x + n.y * n.y;
		real_t k = 1.0 / Math::sqrt(a);
		p = Vector3(-n.y * k, n.x * k, 0);
		// set q = n x p
		q = Vector3(-n.z * p.y, n.z * p.x, a * k);
	}
}

/// JacobianEntry3DSW

// Constraint between two different RigidBodies
JacobianEntry3DSW::JacobianEntry3DSW(
		const Basis &world2A,
		const Basis &world2B,
		const Vector3 &rel_pos1, const Vector3 &rel_pos2,
		const Vector3 &jointAxis,
		const Vector3 &inertiaInvA,
		const real_t massInvA,
		const Vector3 &inertiaInvB,
		const real_t massInvB) :
		m_linearJointAxis(jointAxis) {
	m_aJ = world2A.xform(rel_pos1.cross(m_linearJointAxis));
	m_bJ = world2B.xform(rel_pos2.cross(-m_linearJointAxis));
	m_0MinvJt = inertiaInvA * m_aJ;
	m_1MinvJt = inertiaInvB * m_bJ;
	m_Adiag = massInvA + m_0MinvJt.dot(m_aJ) + massInvB + m_1MinvJt.dot(m_bJ);

	ERR_FAIL_COND(m_Adiag <= real_t(0.0));
}

// Angular constraint between two different RigidBodies
JacobianEntry3DSW::JacobianEntry3DSW(const Vector3 &jointAxis,
		const Basis &world2A,
		const Basis &world2B,
		const Vector3 &inertiaInvA,
		const Vector3 &inertiaInvB) :
		m_linearJointAxis(Vector3(real_t(0.), real_t(0.), real_t(0.))) {
	m_aJ = world2A.xform(jointAxis);
	m_bJ = world2B.xform(-jointAxis);
	m_0MinvJt = inertiaInvA * m_aJ;
	m_1MinvJt = inertiaInvB * m_bJ;
	m_Adiag = m_0MinvJt.dot(m_aJ) + m_1MinvJt.dot(m_bJ);

	ERR_FAIL_COND(m_Adiag <= real_t(0.0));
}

// Angular constraint between two different RigidBodies
JacobianEntry3DSW::JacobianEntry3DSW(const Vector3 &axisInA,
		const Vector3 &axisInB,
		const Vector3 &inertiaInvA,
		const Vector3 &inertiaInvB) :
		m_linearJointAxis(Vector3(real_t(0.), real_t(0.), real_t(0.))),
		m_aJ(axisInA),
		m_bJ(-axisInB) {
	m_0MinvJt = inertiaInvA * m_aJ;
	m_1MinvJt = inertiaInvB * m_bJ;
	m_Adiag = m_0MinvJt.dot(m_aJ) + m_1MinvJt.dot(m_bJ);

	ERR_FAIL_COND(m_Adiag <= real_t(0.0));
}

// Constraint on one RigidBody
JacobianEntry3DSW::JacobianEntry3DSW(
		const Basis &world2A,
		const Vector3 &rel_pos1, const Vector3 &rel_pos2,
		const Vector3 &jointAxis,
		const Vector3 &inertiaInvA,
		const real_t massInvA) :
		m_linearJointAxis(jointAxis) {
	m_aJ = world2A.xform(rel_pos1.cross(jointAxis));
	m_bJ = world2A.xform(rel_pos2.cross(-jointAxis));
	m_0MinvJt = inertiaInvA * m_aJ;
	m_1MinvJt = Vector3(real_t(0.), real_t(0.), real_t(0.));
	m_Adiag = massInvA + m_0MinvJt.dot(m_aJ);

	ERR_FAIL_COND(m_Adiag <= real_t(0.0));
}

real_t JacobianEntry3DSW::getDiagonal() const {
	return m_Adiag;
}

// For two constraints on the same RigidBody (for example vehicle friction)
real_t JacobianEntry3DSW::getNonDiagonal(const JacobianEntry3DSW &jacB, const real_t massInvA) const {
	const JacobianEntry3DSW &jacA = *this;
	real_t lin = massInvA * jacA.m_linearJointAxis.dot(jacB.m_linearJointAxis);
	real_t ang = jacA.m_0MinvJt.dot(jacB.m_aJ);
	return lin + ang;
}

// For two constraints on sharing two same RigidBodies (for example two contact points between two RigidBodies)
real_t JacobianEntry3DSW::getNonDiagonal(const JacobianEntry3DSW &jacB, const real_t massInvA, const real_t massInvB) const {
	const JacobianEntry3DSW &jacA = *this;
	Vector3 lin = jacA.m_linearJointAxis * jacB.m_linearJointAxis;
	Vector3 ang0 = jacA.m_0MinvJt * jacB.m_aJ;
	Vector3 ang1 = jacA.m_1MinvJt * jacB.m_bJ;
	Vector3 lin0 = massInvA * lin;
	Vector3 lin1 = massInvB * lin;
	Vector3 sum = ang0 + ang1 + lin0 + lin1;
	return sum[0] + sum[1] + sum[2];
}

real_t JacobianEntry3DSW::getRelativeVelocity(const Vector3 &linvelA, const Vector3 &angvelA, const Vector3 &linvelB, const Vector3 &angvelB) {
	Vector3 linrel = linvelA - linvelB;
	Vector3 angvela = angvelA * m_aJ;
	Vector3 angvelb = angvelB * m_bJ;
	linrel *= m_linearJointAxis;
	angvela += angvelb;
	angvela += linrel;
	real_t rel_vel2 = angvela[0] + angvela[1] + angvela[2];
	return rel_vel2 + CMP_EPSILON;
}

/// ConeTwistJointSW

/*
ConeTwistJointSW is Copyright (c) 2007 Starbreeze Studios

Written by: Marcus Hennix
*/

ConeTwistJoint3DSW::ConeTwistJoint3DSW(Body3DSW *rbA, Body3DSW *rbB, const Transform &rbAFrame, const Transform &rbBFrame) :
		Joint3DSW(_arr, 2) {
	A = rbA;
	B = rbB;

	m_rbAFrame = rbAFrame;
	m_rbBFrame = rbBFrame;

	m_swingSpan1 = Math_TAU / 8.0;
	m_swingSpan2 = Math_TAU / 8.0;
	m_twistSpan = Math_TAU;
	m_biasFactor = 0.3f;
	m_relaxationFactor = 1.0f;

	m_angularOnly = false;
	m_solveTwistLimit = false;
	m_solveSwingLimit = false;

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);

	m_appliedImpulse = 0;
}

bool ConeTwistJoint3DSW::setup(real_t p_timestep) {
	if ((A->get_mode() <= PhysicsServer3D::BODY_MODE_KINEMATIC) && (B->get_mode() <= PhysicsServer3D::BODY_MODE_KINEMATIC)) {
		return false;
	}

	m_appliedImpulse = real_t(0.);

	// Set bias, sign, clear accumulator
	m_swingCorrection = real_t(0.);
	m_twistLimitSign = real_t(0.);
	m_solveTwistLimit = false;
	m_solveSwingLimit = false;
	m_accTwistLimitImpulse = real_t(0.);
	m_accSwingLimitImpulse = real_t(0.);

	if (!m_angularOnly) {
		Vector3 pivotAInW = A->get_transform().xform(m_rbAFrame.origin);
		Vector3 pivotBInW = B->get_transform().xform(m_rbBFrame.origin);
		Vector3 relPos = pivotBInW - pivotAInW;

		Vector3 normal[3];
		if (Math::is_zero_approx(relPos.length_squared())) {
			normal[0] = Vector3(real_t(1.0), 0, 0);
		} else {
			normal[0] = relPos.normalized();
		}

		plane_space(normal[0], normal[1], normal[2]);

		for (int i = 0; i < 3; i++) {
			memnew_placement(&m_jac[i], JacobianEntry3DSW(
												A->get_principal_inertia_axes().transposed(),
												B->get_principal_inertia_axes().transposed(),
												pivotAInW - A->get_transform().origin - A->get_center_of_mass(),
												pivotBInW - B->get_transform().origin - B->get_center_of_mass(),
												normal[i],
												A->get_inv_inertia(),
												A->get_inv_mass(),
												B->get_inv_inertia(),
												B->get_inv_mass()));
		}
	}

	Vector3 b1Axis1, b1Axis2, b1Axis3;
	Vector3 b2Axis1, b2Axis2;

	b1Axis1 = A->get_transform().basis.xform(this->m_rbAFrame.basis.get_axis(0));
	b2Axis1 = B->get_transform().basis.xform(this->m_rbBFrame.basis.get_axis(0));

	real_t swing1 = real_t(0.), swing2 = real_t(0.);

	real_t swx = real_t(0.), swy = real_t(0.);
	real_t thresh = real_t(10.);
	real_t fact;

	// Get Frame into world space
	if (m_swingSpan1 >= real_t(0.05f)) {
		b1Axis2 = A->get_transform().basis.xform(this->m_rbAFrame.basis.get_axis(1));
		swx = b2Axis1.dot(b1Axis1);
		swy = b2Axis1.dot(b1Axis2);
		swing1 = atan2fast(swy, swx);
		fact = (swy * swy + swx * swx) * thresh * thresh;
		fact = fact / (fact + real_t(1.0));
		swing1 *= fact;
	}

	if (m_swingSpan2 >= real_t(0.05f)) {
		b1Axis3 = A->get_transform().basis.xform(this->m_rbAFrame.basis.get_axis(2));
		swx = b2Axis1.dot(b1Axis1);
		swy = b2Axis1.dot(b1Axis3);
		swing2 = atan2fast(swy, swx);
		fact = (swy * swy + swx * swx) * thresh * thresh;
		fact = fact / (fact + real_t(1.0));
		swing2 *= fact;
	}

	real_t RMaxAngle1Sq = 1.0f / (m_swingSpan1 * m_swingSpan1);
	real_t RMaxAngle2Sq = 1.0f / (m_swingSpan2 * m_swingSpan2);
	real_t EllipseAngle = Math::abs(swing1 * swing1) * RMaxAngle1Sq + Math::abs(swing2 * swing2) * RMaxAngle2Sq;

	if (EllipseAngle > 1.0f) {
		m_swingCorrection = EllipseAngle - 1.0f;
		m_solveSwingLimit = true;

		// Calculate necessary axis & factors
		m_swingAxis = b2Axis1.cross(b1Axis2 * b2Axis1.dot(b1Axis2) + b1Axis3 * b2Axis1.dot(b1Axis3));
		m_swingAxis.normalize();

		real_t swingAxisSign = (b2Axis1.dot(b1Axis1) >= 0.0f) ? 1.0f : -1.0f;
		m_swingAxis *= swingAxisSign;

		m_kSwing = real_t(1.) / (A->compute_angular_impulse_denominator(m_swingAxis) +
										B->compute_angular_impulse_denominator(m_swingAxis));
	}

	// Twist limits
	if (m_twistSpan >= real_t(0.)) {
		Vector3 b2Axis22 = B->get_transform().basis.xform(this->m_rbBFrame.basis.get_axis(1));
		Quat rotationArc = Quat(b2Axis1, b1Axis1);
		Vector3 TwistRef = rotationArc.xform(b2Axis22);
		real_t twist = atan2fast(TwistRef.dot(b1Axis3), TwistRef.dot(b1Axis2));

		real_t lockedFreeFactor = (m_twistSpan > real_t(0.05f)) ? m_limitSoftness : real_t(0.);
		if (twist <= -m_twistSpan * lockedFreeFactor) {
			m_twistCorrection = -(twist + m_twistSpan);
			m_solveTwistLimit = true;

			m_twistAxis = (b2Axis1 + b1Axis1) * 0.5f;
			m_twistAxis.normalize();
			m_twistAxis *= -1.0f;

			m_kTwist = real_t(1.) / (A->compute_angular_impulse_denominator(m_twistAxis) +
											B->compute_angular_impulse_denominator(m_twistAxis));

		} else if (twist > m_twistSpan * lockedFreeFactor) {
			m_twistCorrection = (twist - m_twistSpan);
			m_solveTwistLimit = true;

			m_twistAxis = (b2Axis1 + b1Axis1) * 0.5f;
			m_twistAxis.normalize();

			m_kTwist = real_t(1.) / (A->compute_angular_impulse_denominator(m_twistAxis) +
											B->compute_angular_impulse_denominator(m_twistAxis));
		}
	}

	return true;
}

void ConeTwistJoint3DSW::solve(real_t p_timestep) {
	Vector3 pivotAInW = A->get_transform().xform(m_rbAFrame.origin);
	Vector3 pivotBInW = B->get_transform().xform(m_rbBFrame.origin);

	real_t tau = real_t(0.3);

	// Linear part
	if (!m_angularOnly) {
		Vector3 rel_pos1 = pivotAInW - A->get_transform().origin;
		Vector3 rel_pos2 = pivotBInW - B->get_transform().origin;

		Vector3 vel1 = A->get_velocity_in_local_point(rel_pos1);
		Vector3 vel2 = B->get_velocity_in_local_point(rel_pos2);
		Vector3 vel = vel1 - vel2;

		for (int i = 0; i < 3; i++) {
			const Vector3 &normal = m_jac[i].m_linearJointAxis;
			real_t jacDiagABInv = real_t(1.) / m_jac[i].getDiagonal();

			real_t rel_vel;
			rel_vel = normal.dot(vel);
			//positional error (zeroth order error)
			real_t depth = -(pivotAInW - pivotBInW).dot(normal); //this is the error projected on the normal
			real_t impulse = depth * tau / p_timestep * jacDiagABInv - rel_vel * jacDiagABInv;
			m_appliedImpulse += impulse;
			Vector3 impulse_vector = normal * impulse;
			A->apply_impulse(impulse_vector, pivotAInW - A->get_transform().origin);
			B->apply_impulse(-impulse_vector, pivotBInW - B->get_transform().origin);
		}
	}

	// Solve angular part
	const Vector3 &angVelA = A->get_angular_velocity();
	const Vector3 &angVelB = B->get_angular_velocity();

	// Solve swing limit
	if (m_solveSwingLimit) {
		real_t amplitude = ((angVelB - angVelA).dot(m_swingAxis) * m_relaxationFactor * m_relaxationFactor + m_swingCorrection * (real_t(1.) / p_timestep) * m_biasFactor);
		real_t impulseMag = amplitude * m_kSwing;

		// Clamp the accumulated impulse
		real_t temp = m_accSwingLimitImpulse;
		m_accSwingLimitImpulse = MAX(m_accSwingLimitImpulse + impulseMag, real_t(0.0));
		impulseMag = m_accSwingLimitImpulse - temp;

		Vector3 impulse = m_swingAxis * impulseMag;

		A->apply_torque_impulse(impulse);
		B->apply_torque_impulse(-impulse);
	}

	// Solve twist limit
	if (m_solveTwistLimit) {
		real_t amplitude = ((angVelB - angVelA).dot(m_twistAxis) * m_relaxationFactor * m_relaxationFactor + m_twistCorrection * (real_t(1.) / p_timestep) * m_biasFactor);
		real_t impulseMag = amplitude * m_kTwist;

		// Clamp the accumulated impulse
		real_t temp = m_accTwistLimitImpulse;
		m_accTwistLimitImpulse = MAX(m_accTwistLimitImpulse + impulseMag, real_t(0.0));
		impulseMag = m_accTwistLimitImpulse - temp;

		Vector3 impulse = m_twistAxis * impulseMag;

		A->apply_torque_impulse(impulse);
		B->apply_torque_impulse(-impulse);
	}
}

void ConeTwistJoint3DSW::set_param(PhysicsServer3D::ConeTwistJointParam p_param, real_t p_value) {
	switch (p_param) {
		case PhysicsServer3D::CONE_TWIST_JOINT_SWING_SPAN: {
			m_swingSpan1 = p_value;
			m_swingSpan2 = p_value;
		} break;
		case PhysicsServer3D::CONE_TWIST_JOINT_TWIST_SPAN: {
			m_twistSpan = p_value;
		} break;
		case PhysicsServer3D::CONE_TWIST_JOINT_BIAS: {
			m_biasFactor = p_value;
		} break;
		case PhysicsServer3D::CONE_TWIST_JOINT_SOFTNESS: {
			m_limitSoftness = p_value;
		} break;
		case PhysicsServer3D::CONE_TWIST_JOINT_RELAXATION: {
			m_relaxationFactor = p_value;
		} break;
		case PhysicsServer3D::CONE_TWIST_MAX:
			break; // Can't happen, but silences warning
	}
}

real_t ConeTwistJoint3DSW::get_param(PhysicsServer3D::ConeTwistJointParam p_param) const {
	switch (p_param) {
		case PhysicsServer3D::CONE_TWIST_JOINT_SWING_SPAN: {
			return m_swingSpan1;
		} break;
		case PhysicsServer3D::CONE_TWIST_JOINT_TWIST_SPAN: {
			return m_twistSpan;
		} break;
		case PhysicsServer3D::CONE_TWIST_JOINT_BIAS: {
			return m_biasFactor;
		} break;
		case PhysicsServer3D::CONE_TWIST_JOINT_SOFTNESS: {
			return m_limitSoftness;
		} break;
		case PhysicsServer3D::CONE_TWIST_JOINT_RELAXATION: {
			return m_relaxationFactor;
		} break;
		case PhysicsServer3D::CONE_TWIST_MAX:
			break; // Can't happen, but silences warning
	}
	return 0;
}

/// Generic6DOFJoint3DSW

/*
2007-09-09
Generic6DOFJointSW Refactored by Francisco Le?n
email: projectileman@yahoo.com
http://gimpact.sf.net
*/

#define GENERIC_D6_DISABLE_WARMSTARTING 1

// G6DOFRotationalLimitMotor3DSW

int G6DOFRotationalLimitMotor3DSW::testLimitValue(real_t test_value) {
	if (m_loLimit > m_hiLimit) {
		m_currentLimit = 0; // Free from violation
		return 0;
	}

	if (test_value < m_loLimit) {
		m_currentLimit = 1; // Low limit violation
		m_currentLimitError = test_value - m_loLimit;
		return 1;
	} else if (test_value > m_hiLimit) {
		m_currentLimit = 2; // High limit violation
		m_currentLimitError = test_value - m_hiLimit;
		return 2;
	};

	m_currentLimit = 0; // Free from violation
	return 0;
}

real_t G6DOFRotationalLimitMotor3DSW::solveAngularLimits(
		real_t timeStep, Vector3 &axis, real_t jacDiagABInv,
		Body3DSW *body0, Body3DSW *body1) {
	if (!needApplyTorques()) {
		return 0.0f;
	}

	real_t target_velocity = m_targetVelocity;
	real_t maxMotorForce = m_maxMotorForce;

	// Current error correction
	if (m_currentLimit != 0) {
		target_velocity = -m_ERP * m_currentLimitError / (timeStep);
		maxMotorForce = m_maxLimitForce;
	}

	maxMotorForce *= timeStep;

	// Current velocity difference
	Vector3 vel_diff = body0->get_angular_velocity();
	if (body1) {
		vel_diff -= body1->get_angular_velocity();
	}

	real_t rel_vel = axis.dot(vel_diff);

	// Correction velocity
	real_t motor_relvel = m_limitSoftness * (target_velocity - m_damping * rel_vel);

	if (Math::is_zero_approx(motor_relvel)) {
		return 0.0f; // No need for applying force
	}

	// Correction impulse
	real_t unclippedMotorImpulse = (1 + m_bounce) * motor_relvel * jacDiagABInv;

	// Clip correction impulse
	real_t clippedMotorImpulse;

	// TODO: Should clip against accumulated impulse
	if (unclippedMotorImpulse > 0.0f) {
		clippedMotorImpulse = unclippedMotorImpulse > maxMotorForce ? maxMotorForce : unclippedMotorImpulse;
	} else {
		clippedMotorImpulse = unclippedMotorImpulse < -maxMotorForce ? -maxMotorForce : unclippedMotorImpulse;
	}

	// Sort with accumulated impulses
	real_t lo = real_t(-1e30);
	real_t hi = real_t(1e30);

	real_t oldaccumImpulse = m_accumulatedImpulse;
	real_t sum = oldaccumImpulse + clippedMotorImpulse;
	m_accumulatedImpulse = sum > hi ? real_t(0.) : (sum < lo ? real_t(0.) : sum);

	clippedMotorImpulse = m_accumulatedImpulse - oldaccumImpulse;

	Vector3 motorImp = clippedMotorImpulse * axis;

	body0->apply_torque_impulse(motorImp);
	if (body1) {
		body1->apply_torque_impulse(-motorImp);
	}

	return clippedMotorImpulse;
}

// G6DOFTranslationalLimitMotor3DSW

real_t G6DOFTranslationalLimitMotor3DSW::solveLinearAxis(
		real_t timeStep,
		real_t jacDiagABInv,
		Body3DSW *body1, const Vector3 &pointInA,
		Body3DSW *body2, const Vector3 &pointInB,
		int limit_index,
		const Vector3 &axis_normal_on_a,
		const Vector3 &anchorPos) {
	// Find relative velocity
	Vector3 rel_pos1 = anchorPos - body1->get_transform().origin;
	Vector3 rel_pos2 = anchorPos - body2->get_transform().origin;

	Vector3 vel1 = body1->get_velocity_in_local_point(rel_pos1);
	Vector3 vel2 = body2->get_velocity_in_local_point(rel_pos2);
	Vector3 vel = vel1 - vel2;

	real_t rel_vel = axis_normal_on_a.dot(vel);

	// Apply displacement correction

	// Positional error (zeroth order error)
	real_t depth = -(pointInA - pointInB).dot(axis_normal_on_a);
	real_t lo = real_t(-1e30);
	real_t hi = real_t(1e30);

	real_t minLimit = m_lowerLimit[limit_index];
	real_t maxLimit = m_upperLimit[limit_index];

	// Handle the limits
	if (minLimit < maxLimit) {
		{
			if (depth > maxLimit) {
				depth -= maxLimit;
				lo = real_t(0.);
			} else {
				if (depth < minLimit) {
					depth -= minLimit;
					hi = real_t(0.);
				} else {
					return 0.0f;
				}
			}
		}
	}

	real_t normalImpulse = m_limitSoftness[limit_index] * (m_restitution[limit_index] * depth / timeStep - m_damping[limit_index] * rel_vel) * jacDiagABInv;
	real_t oldNormalImpulse = m_accumulatedImpulse[limit_index];
	real_t sum = oldNormalImpulse + normalImpulse;
	m_accumulatedImpulse[limit_index] = sum > hi ? real_t(0.) : (sum < lo ? real_t(0.) : sum);
	normalImpulse = m_accumulatedImpulse[limit_index] - oldNormalImpulse;

	Vector3 impulse_vector = axis_normal_on_a * normalImpulse;
	body1->apply_impulse(impulse_vector, rel_pos1);
	body2->apply_impulse(-impulse_vector, rel_pos2);
	return normalImpulse;
}

// Generic6DOFJoint3DSW

Generic6DOFJoint3DSW::Generic6DOFJoint3DSW(Body3DSW *rbA, Body3DSW *rbB, const Transform &frameInA, const Transform &frameInB, bool useLinearReferenceFrameA) :
		Joint3DSW(_arr, 2),
		m_frameInA(frameInA),
		m_frameInB(frameInB),
		m_useLinearReferenceFrameA(useLinearReferenceFrameA) {
	A = rbA;
	B = rbB;
	A->add_constraint(this, 0);
	B->add_constraint(this, 1);
}

void Generic6DOFJoint3DSW::calculateAngleInfo() {
	Basis relative_frame = m_calculatedTransformB.basis.inverse() * m_calculatedTransformA.basis;

	m_calculatedAxisAngleDiff = relative_frame.get_euler_xyz();

	// In euler angle mode we do not actually constrain the angular velocity
	// along the axes axis[0] and axis[2] (although we do use axis[1]):
	//
	//    to get			constrain w2-w1 along		...not
	//    ------			---------------------		------
	//    d(angle[0])/dt = 0	ax[1] x ax[2]			ax[0]
	//    d(angle[1])/dt = 0	ax[1]
	//    d(angle[2])/dt = 0	ax[0] x ax[1]			ax[2]
	//
	// Constraining w2-w1 along an axis 'a' means that a'*(w2-w1)=0.
	// to prove the result for angle[0], write the expression for angle[0] from
	// GetInfo1 then take the derivative. to prove this for angle[2] it is
	// easier to take the euler rate expression for d(angle[2])/dt with respect
	// to the components of w and set that to 0.

	Vector3 axis0 = m_calculatedTransformB.basis.get_axis(0);
	Vector3 axis2 = m_calculatedTransformA.basis.get_axis(2);

	m_calculatedAxis[1] = axis2.cross(axis0);
	m_calculatedAxis[0] = m_calculatedAxis[1].cross(axis2);
	m_calculatedAxis[2] = axis0.cross(m_calculatedAxis[1]);
}

void Generic6DOFJoint3DSW::calculateTransforms() {
	m_calculatedTransformA = A->get_transform() * m_frameInA;
	m_calculatedTransformB = B->get_transform() * m_frameInB;

	calculateAngleInfo();
}

void Generic6DOFJoint3DSW::buildLinearJacobian(
		JacobianEntry3DSW &jacLinear, const Vector3 &normalWorld,
		const Vector3 &pivotAInW, const Vector3 &pivotBInW) {
	memnew_placement(&jacLinear, JacobianEntry3DSW(
										 A->get_principal_inertia_axes().transposed(),
										 B->get_principal_inertia_axes().transposed(),
										 pivotAInW - A->get_transform().origin - A->get_center_of_mass(),
										 pivotBInW - B->get_transform().origin - B->get_center_of_mass(),
										 normalWorld,
										 A->get_inv_inertia(),
										 A->get_inv_mass(),
										 B->get_inv_inertia(),
										 B->get_inv_mass()));
}

void Generic6DOFJoint3DSW::buildAngularJacobian(
		JacobianEntry3DSW &jacAngular, const Vector3 &jointAxisW) {
	memnew_placement(&jacAngular, JacobianEntry3DSW(jointAxisW,
										  A->get_principal_inertia_axes().transposed(),
										  B->get_principal_inertia_axes().transposed(),
										  A->get_inv_inertia(),
										  B->get_inv_inertia()));
}

bool Generic6DOFJoint3DSW::testAngularLimitMotor(int axis_index) {
	real_t angle = m_calculatedAxisAngleDiff[axis_index];

	// Test limits
	m_angularLimits[axis_index].testLimitValue(angle);
	return m_angularLimits[axis_index].needApplyTorques();
}

bool Generic6DOFJoint3DSW::setup(real_t p_timestep) {
	if ((A->get_mode() <= PhysicsServer3D::BODY_MODE_KINEMATIC) && (B->get_mode() <= PhysicsServer3D::BODY_MODE_KINEMATIC)) {
		return false;
	}

	// Clear accumulated impulses for the next simulation step
	m_linearLimits.m_accumulatedImpulse = Vector3(real_t(0.), real_t(0.), real_t(0.));
	int i;
	for (i = 0; i < 3; i++) {
		m_angularLimits[i].m_accumulatedImpulse = real_t(0.);
	}

	// Calculate transform
	calculateTransforms();

	calcAnchorPos();
	Vector3 pivotAInW = m_AnchorPos;
	Vector3 pivotBInW = m_AnchorPos;

	Vector3 normalWorld;

	// Linear part
	for (i = 0; i < 3; i++) {
		if (m_linearLimits.enable_limit[i] && m_linearLimits.isLimited(i)) {
			if (m_useLinearReferenceFrameA) {
				normalWorld = m_calculatedTransformA.basis.get_axis(i);
			} else {
				normalWorld = m_calculatedTransformB.basis.get_axis(i);
			}

			buildLinearJacobian(
					m_jacLinear[i], normalWorld,
					pivotAInW, pivotBInW);
		}
	}

	// Angular part
	for (i = 0; i < 3; i++) {
		// Calculate error angle
		if (m_angularLimits[i].m_enableLimit && testAngularLimitMotor(i)) {
			normalWorld = this->getAxis(i);
			// Create angular atom
			buildAngularJacobian(m_jacAng[i], normalWorld);
		}
	}

	return true;
}

void Generic6DOFJoint3DSW::solve(real_t p_timestep) {
	m_timeStep = p_timestep;

	int i;

	// Linear part
	Vector3 pointInA = m_calculatedTransformA.origin;
	Vector3 pointInB = m_calculatedTransformB.origin;

	real_t jacDiagABInv;
	Vector3 linear_axis;
	for (i = 0; i < 3; i++) {
		if (m_linearLimits.enable_limit[i] && m_linearLimits.isLimited(i)) {
			jacDiagABInv = real_t(1.) / m_jacLinear[i].getDiagonal();

			if (m_useLinearReferenceFrameA) {
				linear_axis = m_calculatedTransformA.basis.get_axis(i);
			} else {
				linear_axis = m_calculatedTransformB.basis.get_axis(i);
			}

			m_linearLimits.solveLinearAxis(
					m_timeStep,
					jacDiagABInv,
					A, pointInA,
					B, pointInB,
					i, linear_axis, m_AnchorPos);
		}
	}

	// Angular part
	Vector3 angular_axis;
	real_t angularJacDiagABInv;
	for (i = 0; i < 3; i++) {
		if (m_angularLimits[i].m_enableLimit && m_angularLimits[i].needApplyTorques()) {
			// Get axis
			angular_axis = getAxis(i);
			angularJacDiagABInv = real_t(1.) / m_jacAng[i].getDiagonal();
			m_angularLimits[i].solveAngularLimits(m_timeStep, angular_axis, angularJacDiagABInv, A, B);
		}
	}
}

void Generic6DOFJoint3DSW::updateRHS(real_t timeStep) {
	(void)timeStep;
}

Vector3 Generic6DOFJoint3DSW::getAxis(int axis_index) const {
	return m_calculatedAxis[axis_index];
}

real_t Generic6DOFJoint3DSW::getAngle(int axis_index) const {
	return m_calculatedAxisAngleDiff[axis_index];
}

void Generic6DOFJoint3DSW::calcAnchorPos() {
	real_t imA = A->get_inv_mass();
	real_t imB = B->get_inv_mass();
	real_t weight;
	if (imB == real_t(0.0)) {
		weight = real_t(1.0);
	} else {
		weight = imA / (imA + imB);
	}
	const Vector3 &pA = m_calculatedTransformA.origin;
	const Vector3 &pB = m_calculatedTransformB.origin;
	m_AnchorPos = pA * weight + pB * (real_t(1.0) - weight);
}

void Generic6DOFJoint3DSW::set_param(Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisParam p_param, real_t p_value) {
	ERR_FAIL_INDEX(p_axis, 3);
	switch (p_param) {
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_LOWER_LIMIT: {
			m_linearLimits.m_lowerLimit[p_axis] = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_UPPER_LIMIT: {
			m_linearLimits.m_upperLimit[p_axis] = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS: {
			m_linearLimits.m_limitSoftness[p_axis] = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_RESTITUTION: {
			m_linearLimits.m_restitution[p_axis] = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_DAMPING: {
			m_linearLimits.m_damping[p_axis] = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_LOWER_LIMIT: {
			m_angularLimits[p_axis].m_loLimit = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_UPPER_LIMIT: {
			m_angularLimits[p_axis].m_hiLimit = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_LIMIT_SOFTNESS: {
			m_angularLimits[p_axis].m_limitSoftness = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_DAMPING: {
			m_angularLimits[p_axis].m_damping = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_RESTITUTION: {
			m_angularLimits[p_axis].m_bounce = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_FORCE_LIMIT: {
			m_angularLimits[p_axis].m_maxLimitForce = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_ERP: {
			m_angularLimits[p_axis].m_ERP = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_MOTOR_TARGET_VELOCITY: {
			m_angularLimits[p_axis].m_targetVelocity = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_MOTOR_FORCE_LIMIT: {
			m_angularLimits[p_axis].m_maxLimitForce = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_MOTOR_TARGET_VELOCITY: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_MOTOR_FORCE_LIMIT: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_STIFFNESS: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_DAMPING: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_EQUILIBRIUM_POINT: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_STIFFNESS: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_DAMPING: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_EQUILIBRIUM_POINT: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_MAX:
			break; // Can't happen, but silences warning
	}
}

real_t Generic6DOFJoint3DSW::get_param(Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisParam p_param) const {
	ERR_FAIL_INDEX_V(p_axis, 3, 0);
	switch (p_param) {
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_LOWER_LIMIT: {
			return m_linearLimits.m_lowerLimit[p_axis];
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_UPPER_LIMIT: {
			return m_linearLimits.m_upperLimit[p_axis];
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS: {
			return m_linearLimits.m_limitSoftness[p_axis];
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_RESTITUTION: {
			return m_linearLimits.m_restitution[p_axis];
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_DAMPING: {
			return m_linearLimits.m_damping[p_axis];
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_LOWER_LIMIT: {
			return m_angularLimits[p_axis].m_loLimit;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_UPPER_LIMIT: {
			return m_angularLimits[p_axis].m_hiLimit;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_LIMIT_SOFTNESS: {
			return m_angularLimits[p_axis].m_limitSoftness;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_DAMPING: {
			return m_angularLimits[p_axis].m_damping;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_RESTITUTION: {
			return m_angularLimits[p_axis].m_bounce;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_FORCE_LIMIT: {
			return m_angularLimits[p_axis].m_maxLimitForce;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_ERP: {
			return m_angularLimits[p_axis].m_ERP;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_MOTOR_TARGET_VELOCITY: {
			return m_angularLimits[p_axis].m_targetVelocity;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_MOTOR_FORCE_LIMIT: {
			return m_angularLimits[p_axis].m_maxMotorForce;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_MOTOR_TARGET_VELOCITY: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_MOTOR_FORCE_LIMIT: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_STIFFNESS: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_DAMPING: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_EQUILIBRIUM_POINT: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_STIFFNESS: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_DAMPING: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_EQUILIBRIUM_POINT: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_MAX:
			break; // Can't happen, but silences warning
	}
	return 0;
}

void Generic6DOFJoint3DSW::set_flag(Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisFlag p_flag, bool p_value) {
	ERR_FAIL_INDEX(p_axis, 3);

	switch (p_flag) {
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT: {
			m_linearLimits.enable_limit[p_axis] = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT: {
			m_angularLimits[p_axis].m_enableLimit = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_MOTOR: {
			m_angularLimits[p_axis].m_enableMotor = p_value;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_MOTOR: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_SPRING: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_SPRING: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_MAX:
			break; // Can't happen, but silences warning
	}
}

bool Generic6DOFJoint3DSW::get_flag(Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisFlag p_flag) const {
	ERR_FAIL_INDEX_V(p_axis, 3, 0);
	switch (p_flag) {
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT: {
			return m_linearLimits.enable_limit[p_axis];
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT: {
			return m_angularLimits[p_axis].m_enableLimit;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_MOTOR: {
			return m_angularLimits[p_axis].m_enableMotor;
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_MOTOR: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_SPRING: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_SPRING: {
			// Not implemented in GodotPhysics3D backend
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_MAX:
			break; // Can't happen, but silences warning
	}
	return false;
}

/// HingeJoint3DSW

HingeJoint3DSW::HingeJoint3DSW(Body3DSW *rbA, Body3DSW *rbB, const Transform &frameA, const Transform &frameB) :
		Joint3DSW(_arr, 2) {
	A = rbA;
	B = rbB;

	m_rbAFrame = frameA;
	m_rbBFrame = frameB;

	// Flip axis
	m_rbBFrame.basis[0][2] *= real_t(-1.);
	m_rbBFrame.basis[1][2] *= real_t(-1.);
	m_rbBFrame.basis[2][2] *= real_t(-1.);

	// Start with free
	m_lowerLimit = Math_PI;
	m_upperLimit = -Math_PI;

	m_useLimit = false;
	m_biasFactor = 0.3f;
	m_relaxationFactor = 1.0f;
	m_limitSoftness = 0.9f;
	m_solveLimit = false;

	tau = 0.3;

	m_angularOnly = false;
	m_enableAngularMotor = false;

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);
}

HingeJoint3DSW::HingeJoint3DSW(Body3DSW *rbA, Body3DSW *rbB, const Vector3 &pivotInA, const Vector3 &pivotInB,
		const Vector3 &axisInA, const Vector3 &axisInB) :
		Joint3DSW(_arr, 2) {
	A = rbA;
	B = rbB;

	m_rbAFrame.origin = pivotInA;

	// Since no frame is given, assume this to be zero angle and just pick rb transform axis
	Vector3 rbAxisA1 = rbA->get_transform().basis.get_axis(0);

	Vector3 rbAxisA2;
	real_t projection = axisInA.dot(rbAxisA1);
	if (projection >= 1.0f - CMP_EPSILON) {
		rbAxisA1 = -rbA->get_transform().basis.get_axis(2);
		rbAxisA2 = rbA->get_transform().basis.get_axis(1);
	} else if (projection <= -1.0f + CMP_EPSILON) {
		rbAxisA1 = rbA->get_transform().basis.get_axis(2);
		rbAxisA2 = rbA->get_transform().basis.get_axis(1);
	} else {
		rbAxisA2 = axisInA.cross(rbAxisA1);
		rbAxisA1 = rbAxisA2.cross(axisInA);
	}

	m_rbAFrame.basis = Basis(rbAxisA1.x, rbAxisA2.x, axisInA.x,
			rbAxisA1.y, rbAxisA2.y, axisInA.y,
			rbAxisA1.z, rbAxisA2.z, axisInA.z);

	Quat rotationArc = Quat(axisInA, axisInB);
	Vector3 rbAxisB1 = rotationArc.xform(rbAxisA1);
	Vector3 rbAxisB2 = axisInB.cross(rbAxisB1);

	m_rbBFrame.origin = pivotInB;
	m_rbBFrame.basis = Basis(rbAxisB1.x, rbAxisB2.x, -axisInB.x,
			rbAxisB1.y, rbAxisB2.y, -axisInB.y,
			rbAxisB1.z, rbAxisB2.z, -axisInB.z);

	// Start with free
	m_lowerLimit = Math_PI;
	m_upperLimit = -Math_PI;

	m_useLimit = false;
	m_biasFactor = 0.3f;
	m_relaxationFactor = 1.0f;
	m_limitSoftness = 0.9f;
	m_solveLimit = false;

	tau = 0.3;

	m_angularOnly = false;
	m_enableAngularMotor = false;

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);
}

bool HingeJoint3DSW::setup(real_t p_step) {
	if ((A->get_mode() <= PhysicsServer3D::BODY_MODE_KINEMATIC) && (B->get_mode() <= PhysicsServer3D::BODY_MODE_KINEMATIC)) {
		return false;
	}

	m_appliedImpulse = real_t(0.);

	if (!m_angularOnly) {
		Vector3 pivotAInW = A->get_transform().xform(m_rbAFrame.origin);
		Vector3 pivotBInW = B->get_transform().xform(m_rbBFrame.origin);
		Vector3 relPos = pivotBInW - pivotAInW;

		Vector3 normal[3];
		if (Math::is_zero_approx(relPos.length_squared())) {
			normal[0] = Vector3(real_t(1.0), 0, 0);
		} else {
			normal[0] = relPos.normalized();
		}

		plane_space(normal[0], normal[1], normal[2]);

		for (int i = 0; i < 3; i++) {
			memnew_placement(&m_jac[i], JacobianEntry3DSW(
												A->get_principal_inertia_axes().transposed(),
												B->get_principal_inertia_axes().transposed(),
												pivotAInW - A->get_transform().origin - A->get_center_of_mass(),
												pivotBInW - B->get_transform().origin - B->get_center_of_mass(),
												normal[i],
												A->get_inv_inertia(),
												A->get_inv_mass(),
												B->get_inv_inertia(),
												B->get_inv_mass()));
		}
	}

	// Calculate two perpendicular jointAxis, orthogonal to hingeAxis
	// these two jointAxis require equal angular velocities for both bodies

	// TODO: This is unused for now
	Vector3 jointAxis0local;
	Vector3 jointAxis1local;

	plane_space(m_rbAFrame.basis.get_axis(2), jointAxis0local, jointAxis1local);

	Vector3 jointAxis0 = A->get_transform().basis.xform(jointAxis0local);
	Vector3 jointAxis1 = A->get_transform().basis.xform(jointAxis1local);
	Vector3 hingeAxisWorld = A->get_transform().basis.xform(m_rbAFrame.basis.get_axis(2));

	memnew_placement(&m_jacAng[0], JacobianEntry3DSW(jointAxis0,
										   A->get_principal_inertia_axes().transposed(),
										   B->get_principal_inertia_axes().transposed(),
										   A->get_inv_inertia(),
										   B->get_inv_inertia()));

	memnew_placement(&m_jacAng[1], JacobianEntry3DSW(jointAxis1,
										   A->get_principal_inertia_axes().transposed(),
										   B->get_principal_inertia_axes().transposed(),
										   A->get_inv_inertia(),
										   B->get_inv_inertia()));

	memnew_placement(&m_jacAng[2], JacobianEntry3DSW(hingeAxisWorld,
										   A->get_principal_inertia_axes().transposed(),
										   B->get_principal_inertia_axes().transposed(),
										   A->get_inv_inertia(),
										   B->get_inv_inertia()));

	// Compute limit information
	real_t hingeAngle = get_hinge_angle();

	// Set bias, sign, clear accumulator
	m_correction = real_t(0.);
	m_limitSign = real_t(0.);
	m_solveLimit = false;
	m_accLimitImpulse = real_t(0.);

	if (m_useLimit && m_lowerLimit <= m_upperLimit) {
		if (hingeAngle <= m_lowerLimit) {
			m_correction = (m_lowerLimit - hingeAngle);
			m_limitSign = 1.0f;
			m_solveLimit = true;
		} else if (hingeAngle >= m_upperLimit) {
			m_correction = m_upperLimit - hingeAngle;
			m_limitSign = -1.0f;
			m_solveLimit = true;
		}
	}

	// Compute K = J*W*J' for hinge axis
	Vector3 axisA = A->get_transform().basis.xform(m_rbAFrame.basis.get_axis(2));
	m_kHinge = 1.0f / (A->compute_angular_impulse_denominator(axisA) +
							  B->compute_angular_impulse_denominator(axisA));

	return true;
}

void HingeJoint3DSW::solve(real_t p_step) {
	Vector3 pivotAInW = A->get_transform().xform(m_rbAFrame.origin);
	Vector3 pivotBInW = B->get_transform().xform(m_rbBFrame.origin);

	// Linear part
	if (!m_angularOnly) {
		Vector3 rel_pos1 = pivotAInW - A->get_transform().origin;
		Vector3 rel_pos2 = pivotBInW - B->get_transform().origin;

		Vector3 vel1 = A->get_velocity_in_local_point(rel_pos1);
		Vector3 vel2 = B->get_velocity_in_local_point(rel_pos2);
		Vector3 vel = vel1 - vel2;

		for (int i = 0; i < 3; i++) {
			const Vector3 &normal = m_jac[i].m_linearJointAxis;
			real_t jacDiagABInv = real_t(1.) / m_jac[i].getDiagonal();

			real_t rel_vel;
			rel_vel = normal.dot(vel);
			// Positional error (zeroth order error)
			real_t depth = -(pivotAInW - pivotBInW).dot(normal); // This is the error projected on the normal
			real_t impulse = depth * tau / p_step * jacDiagABInv - rel_vel * jacDiagABInv;
			m_appliedImpulse += impulse;
			Vector3 impulse_vector = normal * impulse;
			A->apply_impulse(impulse_vector, pivotAInW - A->get_transform().origin);
			B->apply_impulse(-impulse_vector, pivotBInW - B->get_transform().origin);
		}
	}

	// Angular part

	// Get axes in world space
	Vector3 axisA = A->get_transform().basis.xform(m_rbAFrame.basis.get_axis(2));
	Vector3 axisB = B->get_transform().basis.xform(m_rbBFrame.basis.get_axis(2));

	const Vector3 &angVelA = A->get_angular_velocity();
	const Vector3 &angVelB = B->get_angular_velocity();

	Vector3 angVelAroundHingeAxisA = axisA * axisA.dot(angVelA);
	Vector3 angVelAroundHingeAxisB = axisB * axisB.dot(angVelB);

	Vector3 angAorthog = angVelA - angVelAroundHingeAxisA;
	Vector3 angBorthog = angVelB - angVelAroundHingeAxisB;
	Vector3 velrelOrthog = angAorthog - angBorthog;
	{
		// Solve orthogonal angular velocity correction
		real_t relaxation = real_t(1.);
		real_t len = velrelOrthog.length();
		if (len > real_t(0.00001)) {
			Vector3 normal = velrelOrthog.normalized();
			real_t denom = A->compute_angular_impulse_denominator(normal) +
						   B->compute_angular_impulse_denominator(normal);
			// Scale for mass and relaxation
			velrelOrthog *= (real_t(1.) / denom) * m_relaxationFactor;
		}

		// Solve angular positional correction
		Vector3 angularError = -axisA.cross(axisB) * (real_t(1.) / p_step);
		real_t len2 = angularError.length();
		if (len2 > real_t(0.00001)) {
			Vector3 normal2 = angularError.normalized();
			real_t denom2 = A->compute_angular_impulse_denominator(normal2) +
							B->compute_angular_impulse_denominator(normal2);
			angularError *= (real_t(1.) / denom2) * relaxation;
		}

		A->apply_torque_impulse(-velrelOrthog + angularError);
		B->apply_torque_impulse(velrelOrthog - angularError);

		// Solve limit
		if (m_solveLimit) {
			real_t amplitude = ((angVelB - angVelA).dot(axisA) * m_relaxationFactor + m_correction * (real_t(1.) / p_step) * m_biasFactor) * m_limitSign;

			real_t impulseMag = amplitude * m_kHinge;

			// Clamp the accumulated impulse
			real_t temp = m_accLimitImpulse;
			m_accLimitImpulse = MAX(m_accLimitImpulse + impulseMag, real_t(0));
			impulseMag = m_accLimitImpulse - temp;

			Vector3 impulse = axisA * impulseMag * m_limitSign;
			A->apply_torque_impulse(impulse);
			B->apply_torque_impulse(-impulse);
		}
	}

	// Apply motor
	if (m_enableAngularMotor) {
		// TODO: Add limits too
		Vector3 angularLimit(0, 0, 0);

		Vector3 velrel = angVelAroundHingeAxisA - angVelAroundHingeAxisB;
		real_t projRelVel = velrel.dot(axisA);

		real_t desiredMotorVel = m_motorTargetVelocity;
		real_t motor_relvel = desiredMotorVel - projRelVel;

		real_t unclippedMotorImpulse = m_kHinge * motor_relvel;
		// TODO: Should clip against accumulated impulse
		real_t clippedMotorImpulse = unclippedMotorImpulse > m_maxMotorImpulse ? m_maxMotorImpulse : unclippedMotorImpulse;
		clippedMotorImpulse = clippedMotorImpulse < -m_maxMotorImpulse ? -m_maxMotorImpulse : clippedMotorImpulse;
		Vector3 motorImp = clippedMotorImpulse * axisA;

		A->apply_torque_impulse(motorImp + angularLimit);
		B->apply_torque_impulse(-motorImp - angularLimit);
	}
}

real_t HingeJoint3DSW::get_hinge_angle() {
	const Vector3 refAxis0 = A->get_transform().basis.xform(m_rbAFrame.basis.get_axis(0));
	const Vector3 refAxis1 = A->get_transform().basis.xform(m_rbAFrame.basis.get_axis(1));
	const Vector3 swingAxis = B->get_transform().basis.xform(m_rbBFrame.basis.get_axis(1));

	return atan2fast(swingAxis.dot(refAxis0), swingAxis.dot(refAxis1));
}

void HingeJoint3DSW::set_param(PhysicsServer3D::HingeJointParam p_param, real_t p_value) {
	switch (p_param) {
		case PhysicsServer3D::HINGE_JOINT_BIAS:
			tau = p_value;
			break;
		case PhysicsServer3D::HINGE_JOINT_LIMIT_UPPER:
			m_upperLimit = p_value;
			break;
		case PhysicsServer3D::HINGE_JOINT_LIMIT_LOWER:
			m_lowerLimit = p_value;
			break;
		case PhysicsServer3D::HINGE_JOINT_LIMIT_BIAS:
			m_biasFactor = p_value;
			break;
		case PhysicsServer3D::HINGE_JOINT_LIMIT_SOFTNESS:
			m_limitSoftness = p_value;
			break;
		case PhysicsServer3D::HINGE_JOINT_LIMIT_RELAXATION:
			m_relaxationFactor = p_value;
			break;
		case PhysicsServer3D::HINGE_JOINT_MOTOR_TARGET_VELOCITY:
			m_motorTargetVelocity = p_value;
			break;
		case PhysicsServer3D::HINGE_JOINT_MOTOR_MAX_IMPULSE:
			m_maxMotorImpulse = p_value;
			break;
		case PhysicsServer3D::HINGE_JOINT_MAX:
			break; // Can't happen, but silences warning
	}
}

real_t HingeJoint3DSW::get_param(PhysicsServer3D::HingeJointParam p_param) const {
	switch (p_param) {
		case PhysicsServer3D::HINGE_JOINT_BIAS:
			return tau;
		case PhysicsServer3D::HINGE_JOINT_LIMIT_UPPER:
			return m_upperLimit;
		case PhysicsServer3D::HINGE_JOINT_LIMIT_LOWER:
			return m_lowerLimit;
		case PhysicsServer3D::HINGE_JOINT_LIMIT_BIAS:
			return m_biasFactor;
		case PhysicsServer3D::HINGE_JOINT_LIMIT_SOFTNESS:
			return m_limitSoftness;
		case PhysicsServer3D::HINGE_JOINT_LIMIT_RELAXATION:
			return m_relaxationFactor;
		case PhysicsServer3D::HINGE_JOINT_MOTOR_TARGET_VELOCITY:
			return m_motorTargetVelocity;
		case PhysicsServer3D::HINGE_JOINT_MOTOR_MAX_IMPULSE:
			return m_maxMotorImpulse;
		case PhysicsServer3D::HINGE_JOINT_MAX:
			break; // Can't happen, but silences warning
	}
	return 0;
}

void HingeJoint3DSW::set_flag(PhysicsServer3D::HingeJointFlag p_flag, bool p_value) {
	switch (p_flag) {
		case PhysicsServer3D::HINGE_JOINT_FLAG_USE_LIMIT:
			m_useLimit = p_value;
			break;
		case PhysicsServer3D::HINGE_JOINT_FLAG_ENABLE_MOTOR:
			m_enableAngularMotor = p_value;
			break;
		case PhysicsServer3D::HINGE_JOINT_FLAG_MAX:
			break; // Can't happen, but silences warning
	}
}

bool HingeJoint3DSW::get_flag(PhysicsServer3D::HingeJointFlag p_flag) const {
	switch (p_flag) {
		case PhysicsServer3D::HINGE_JOINT_FLAG_USE_LIMIT:
			return m_useLimit;
		case PhysicsServer3D::HINGE_JOINT_FLAG_ENABLE_MOTOR:
			return m_enableAngularMotor;
		case PhysicsServer3D::HINGE_JOINT_FLAG_MAX:
			break; // Can't happen, but silences warning
	}
	return false;
}

/// PinJoint3DSW

bool PinJoint3DSW::setup(real_t p_step) {
	if ((A->get_mode() <= PhysicsServer3D::BODY_MODE_KINEMATIC) && (B->get_mode() <= PhysicsServer3D::BODY_MODE_KINEMATIC)) {
		return false;
	}

	m_appliedImpulse = real_t(0.);

	Vector3 normal(0, 0, 0);

	for (int i = 0; i < 3; i++) {
		normal[i] = 1;
		memnew_placement(&m_jac[i], JacobianEntry3DSW(
											A->get_principal_inertia_axes().transposed(),
											B->get_principal_inertia_axes().transposed(),
											A->get_transform().xform(m_pivotInA) - A->get_transform().origin - A->get_center_of_mass(),
											B->get_transform().xform(m_pivotInB) - B->get_transform().origin - B->get_center_of_mass(),
											normal,
											A->get_inv_inertia(),
											A->get_inv_mass(),
											B->get_inv_inertia(),
											B->get_inv_mass()));
		normal[i] = 0;
	}

	return true;
}

void PinJoint3DSW::solve(real_t p_step) {
	Vector3 pivotAInW = A->get_transform().xform(m_pivotInA);
	Vector3 pivotBInW = B->get_transform().xform(m_pivotInB);

	Vector3 normal(0, 0, 0);

	for (int i = 0; i < 3; i++) {
		normal[i] = 1;
		// This jacobian entry could be re-used for all iterations
		real_t jacDiagABInv = real_t(1.) / m_jac[i].getDiagonal();

		Vector3 rel_pos1 = pivotAInW - A->get_transform().origin;
		Vector3 rel_pos2 = pivotBInW - B->get_transform().origin;

		Vector3 vel1 = A->get_velocity_in_local_point(rel_pos1);
		Vector3 vel2 = B->get_velocity_in_local_point(rel_pos2);
		Vector3 vel = vel1 - vel2;

		real_t rel_vel;
		rel_vel = normal.dot(vel);

		// Positional error (zeroth order error)
		real_t depth = -(pivotAInW - pivotBInW).dot(normal); // This is the error projected on the normal
		real_t impulse = depth * m_tau / p_step * jacDiagABInv - m_damping * rel_vel * jacDiagABInv;
		real_t impulseClamp = m_impulseClamp;
		if (impulseClamp > 0) {
			if (impulse < -impulseClamp) {
				impulse = -impulseClamp;
			}
			if (impulse > impulseClamp) {
				impulse = impulseClamp;
			}
		}

		m_appliedImpulse += impulse;
		Vector3 impulse_vector = normal * impulse;
		A->apply_impulse(impulse_vector, pivotAInW - A->get_transform().origin);
		B->apply_impulse(-impulse_vector, pivotBInW - B->get_transform().origin);

		normal[i] = 0;
	}
}

void PinJoint3DSW::set_param(PhysicsServer3D::PinJointParam p_param, real_t p_value) {
	switch (p_param) {
		case PhysicsServer3D::PIN_JOINT_BIAS:
			m_tau = p_value;
			break;
		case PhysicsServer3D::PIN_JOINT_DAMPING:
			m_damping = p_value;
			break;
		case PhysicsServer3D::PIN_JOINT_IMPULSE_CLAMP:
			m_impulseClamp = p_value;
			break;
	}
}

real_t PinJoint3DSW::get_param(PhysicsServer3D::PinJointParam p_param) const {
	switch (p_param) {
		case PhysicsServer3D::PIN_JOINT_BIAS:
			return m_tau;
		case PhysicsServer3D::PIN_JOINT_DAMPING:
			return m_damping;
		case PhysicsServer3D::PIN_JOINT_IMPULSE_CLAMP:
			return m_impulseClamp;
	}
	return 0;
}

PinJoint3DSW::PinJoint3DSW(Body3DSW *p_body_a, const Vector3 &p_pos_a, Body3DSW *p_body_b, const Vector3 &p_pos_b) :
		Joint3DSW(_arr, 2) {
	A = p_body_a;
	B = p_body_b;
	m_pivotInA = p_pos_a;
	m_pivotInB = p_pos_b;

	m_tau = 0.3;
	m_damping = 1;
	m_impulseClamp = 0;
	m_appliedImpulse = 0;

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);
}

PinJoint3DSW::~PinJoint3DSW() {
}

/// SliderJoint3DSW

/*
Added by Roman Ponomarev (rponom@gmail.com)
April 04, 2008
*/

void SliderJoint3DSW::initParams() {
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
}

SliderJoint3DSW::SliderJoint3DSW(Body3DSW *rbA, Body3DSW *rbB, const Transform &frameInA, const Transform &frameInB) :
		Joint3DSW(_arr, 2),
		m_frameInA(frameInA),
		m_frameInB(frameInB) {
	A = rbA;
	B = rbB;

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);

	initParams();
}

bool SliderJoint3DSW::setup(real_t p_step) {
	if ((A->get_mode() <= PhysicsServer3D::BODY_MODE_KINEMATIC) && (B->get_mode() <= PhysicsServer3D::BODY_MODE_KINEMATIC)) {
		return false;
	}

	// Calculate transforms
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

	// Linear part
	for (i = 0; i < 3; i++) {
		normalWorld = m_calculatedTransformA.basis.get_axis(i);
		memnew_placement(&m_jacLin[i], JacobianEntry3DSW(
											   A->get_principal_inertia_axes().transposed(),
											   B->get_principal_inertia_axes().transposed(),
											   m_relPosA - A->get_center_of_mass(),
											   m_relPosB - B->get_center_of_mass(),
											   normalWorld,
											   A->get_inv_inertia(),
											   A->get_inv_mass(),
											   B->get_inv_inertia(),
											   B->get_inv_mass()));
		m_jacLinDiagABInv[i] = real_t(1.) / m_jacLin[i].getDiagonal();
		m_depth[i] = m_delta.dot(normalWorld);
	}
	testLinLimits();

	// Angular part
	for (i = 0; i < 3; i++) {
		normalWorld = m_calculatedTransformA.basis.get_axis(i);
		memnew_placement(&m_jacAng[i], JacobianEntry3DSW(
											   normalWorld,
											   A->get_principal_inertia_axes().transposed(),
											   B->get_principal_inertia_axes().transposed(),
											   A->get_inv_inertia(),
											   B->get_inv_inertia()));
	}

	testAngLimits();
	Vector3 axisA = m_calculatedTransformA.basis.get_axis(0);
	m_kAngle = real_t(1.0) / (A->compute_angular_impulse_denominator(axisA) + B->compute_angular_impulse_denominator(axisA));
	// Clear accumulator for motors
	m_accumulatedLinMotorImpulse = real_t(0.0);
	m_accumulatedAngMotorImpulse = real_t(0.0);

	return true;
}

void SliderJoint3DSW::solve(real_t p_step) {
	int i;

	// Linear part
	Vector3 velA = A->get_velocity_in_local_point(m_relPosA);
	Vector3 velB = B->get_velocity_in_local_point(m_relPosB);
	Vector3 vel = velA - velB;
	for (i = 0; i < 3; i++) {
		const Vector3 &normal = m_jacLin[i].m_linearJointAxis;
		real_t rel_vel = normal.dot(vel);
		// Calculate positional error
		real_t depth = m_depth[i];
		// Get parameters
		real_t softness = (i) ? m_softnessOrthoLin : (m_solveLinLim ? m_softnessLimLin : m_softnessDirLin);
		real_t restitution = (i) ? m_restitutionOrthoLin : (m_solveLinLim ? m_restitutionLimLin : m_restitutionDirLin);
		real_t damping = (i) ? m_dampingOrthoLin : (m_solveLinLim ? m_dampingLimLin : m_dampingDirLin);
		// Calcutate and apply impulse
		real_t normalImpulse = softness * (restitution * depth / p_step - damping * rel_vel) * m_jacLinDiagABInv[i];
		Vector3 impulse_vector = normal * normalImpulse;
		A->apply_impulse(impulse_vector, m_relPosA);
		B->apply_impulse(-impulse_vector, m_relPosB);
		if (m_poweredLinMotor && (!i)) { // Apply linear motor
			if (m_accumulatedLinMotorImpulse < m_maxLinMotorForce) {
				real_t desiredMotorVel = m_targetLinMotorVelocity;
				real_t motor_relvel = desiredMotorVel + rel_vel;
				normalImpulse = -motor_relvel * m_jacLinDiagABInv[i];
				// Clamp accumulated impulse
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
				// Apply clamped impulse
				impulse_vector = normal * normalImpulse;
				A->apply_impulse(impulse_vector, m_relPosA);
				B->apply_impulse(-impulse_vector, m_relPosB);
			}
		}
	}

	// Angular part
	// Get axes in world space
	Vector3 axisA = m_calculatedTransformA.basis.get_axis(0);
	Vector3 axisB = m_calculatedTransformB.basis.get_axis(0);

	const Vector3 &angVelA = A->get_angular_velocity();
	const Vector3 &angVelB = B->get_angular_velocity();

	Vector3 angVelAroundAxisA = axisA * axisA.dot(angVelA);
	Vector3 angVelAroundAxisB = axisB * axisB.dot(angVelB);

	Vector3 angAorthog = angVelA - angVelAroundAxisA;
	Vector3 angBorthog = angVelB - angVelAroundAxisB;
	Vector3 velrelOrthog = angAorthog - angBorthog;

	// Solve orthogonal angular velocity correction
	real_t len = velrelOrthog.length();
	if (len > real_t(0.00001)) {
		Vector3 normal = velrelOrthog.normalized();
		real_t denom = A->compute_angular_impulse_denominator(normal) + B->compute_angular_impulse_denominator(normal);
		velrelOrthog *= (real_t(1.) / denom) * m_dampingOrthoAng * m_softnessOrthoAng;
	}

	// Solve angular positional correction
	Vector3 angularError = axisA.cross(axisB) * (real_t(1.) / p_step);
	real_t len2 = angularError.length();
	if (len2 > real_t(0.00001)) {
		Vector3 normal2 = angularError.normalized();
		real_t denom2 = A->compute_angular_impulse_denominator(normal2) + B->compute_angular_impulse_denominator(normal2);
		angularError *= (real_t(1.) / denom2) * m_restitutionOrthoAng * m_softnessOrthoAng;
	}

	// Apply impulse
	A->apply_torque_impulse(-velrelOrthog + angularError);
	B->apply_torque_impulse(velrelOrthog - angularError);
	real_t impulseMag;

	// Solve angular limits
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

	// Apply angular motor
	if (m_poweredAngMotor) {
		if (m_accumulatedAngMotorImpulse < m_maxAngMotorForce) {
			Vector3 velrel = angVelAroundAxisA - angVelAroundAxisB;
			real_t projRelVel = velrel.dot(axisA);
			real_t desiredMotorVel = m_targetAngMotorVelocity;
			real_t motor_relvel = desiredMotorVel - projRelVel;
			real_t angImpulse = m_kAngle * motor_relvel;
			// Clamp accumulated impulse
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
			// Apply clamped impulse
			Vector3 motorImp = angImpulse * axisA;
			A->apply_torque_impulse(motorImp);
			B->apply_torque_impulse(-motorImp);
		}
	}
}

void SliderJoint3DSW::calculateTransforms() {
	m_calculatedTransformA = A->get_transform() * m_frameInA;
	m_calculatedTransformB = B->get_transform() * m_frameInB;
	m_realPivotAInW = m_calculatedTransformA.origin;
	m_realPivotBInW = m_calculatedTransformB.origin;
	m_sliderAxis = m_calculatedTransformA.basis.get_axis(0); // Along X
	m_delta = m_realPivotBInW - m_realPivotAInW;
	m_projPivotInW = m_realPivotAInW + m_sliderAxis.dot(m_delta) * m_sliderAxis;
	Vector3 normalWorld;
	int i;

	// Linear part
	for (i = 0; i < 3; i++) {
		normalWorld = m_calculatedTransformA.basis.get_axis(i);
		m_depth[i] = m_delta.dot(normalWorld);
	}
}

void SliderJoint3DSW::testLinLimits() {
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
}

void SliderJoint3DSW::testAngLimits() {
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
}

Vector3 SliderJoint3DSW::getAncorInA() {
	Vector3 ancorInA;
	ancorInA = m_realPivotAInW + (m_lowerLinLimit + m_upperLinLimit) * real_t(0.5) * m_sliderAxis;
	ancorInA = A->get_transform().inverse().xform(ancorInA);
	return ancorInA;
}

Vector3 SliderJoint3DSW::getAncorInB() {
	Vector3 ancorInB;
	ancorInB = m_frameInB.origin;
	return ancorInB;
}

void SliderJoint3DSW::set_param(PhysicsServer3D::SliderJointParam p_param, real_t p_value) {
	switch (p_param) {
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_UPPER:
			m_upperLinLimit = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_LOWER:
			m_lowerLinLimit = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS:
			m_softnessLimLin = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION:
			m_restitutionLimLin = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_DAMPING:
			m_dampingLimLin = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_SOFTNESS:
			m_softnessDirLin = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_RESTITUTION:
			m_restitutionDirLin = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_DAMPING:
			m_dampingDirLin = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_SOFTNESS:
			m_softnessOrthoLin = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_RESTITUTION:
			m_restitutionOrthoLin = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_DAMPING:
			m_dampingOrthoLin = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_UPPER:
			m_upperAngLimit = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_LOWER:
			m_lowerAngLimit = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS:
			m_softnessLimAng = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_RESTITUTION:
			m_restitutionLimAng = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_DAMPING:
			m_dampingLimAng = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_SOFTNESS:
			m_softnessDirAng = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_RESTITUTION:
			m_restitutionDirAng = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_DAMPING:
			m_dampingDirAng = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_SOFTNESS:
			m_softnessOrthoAng = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_RESTITUTION:
			m_restitutionOrthoAng = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_DAMPING:
			m_dampingOrthoAng = p_value;
			break;
		case PhysicsServer3D::SLIDER_JOINT_MAX:
			break; // Can't happen, but silences warning
	}
}

real_t SliderJoint3DSW::get_param(PhysicsServer3D::SliderJointParam p_param) const {
	switch (p_param) {
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_UPPER:
			return m_upperLinLimit;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_LOWER:
			return m_lowerLinLimit;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS:
			return m_softnessLimLin;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION:
			return m_restitutionLimLin;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_DAMPING:
			return m_dampingLimLin;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_SOFTNESS:
			return m_softnessDirLin;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_RESTITUTION:
			return m_restitutionDirLin;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_DAMPING:
			return m_dampingDirLin;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_SOFTNESS:
			return m_softnessOrthoLin;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_RESTITUTION:
			return m_restitutionOrthoLin;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_DAMPING:
			return m_dampingOrthoLin;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_UPPER:
			return m_upperAngLimit;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_LOWER:
			return m_lowerAngLimit;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS:
			return m_softnessLimAng;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_RESTITUTION:
			return m_restitutionLimAng;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_DAMPING:
			return m_dampingLimAng;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_SOFTNESS:
			return m_softnessDirAng;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_RESTITUTION:
			return m_restitutionDirAng;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_DAMPING:
			return m_dampingDirAng;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_SOFTNESS:
			return m_softnessOrthoAng;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_RESTITUTION:
			return m_restitutionOrthoAng;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_DAMPING:
			return m_dampingOrthoAng;
		case PhysicsServer3D::SLIDER_JOINT_MAX:
			break; // Can't happen, but silences warning
	}
	return 0;
}

/*************************************************************************/
/*  cone_twist_joint_sw.cpp                                              */
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

/*
Adapted to Godot from the Bullet library.
See corresponding header file for licensing info.
*/

#include "cone_twist_joint_sw.h"

static void plane_space(const Vector3 &n, Vector3 &p, Vector3 &q) {

	if (Math::abs(n.z) > 0.707106781186547524400844362) {
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

ConeTwistJointSW::ConeTwistJointSW(BodySW *rbA, BodySW *rbB, const Transform &rbAFrame, const Transform &rbBFrame)
	: JointSW(_arr, 2) {

	A = rbA;
	B = rbB;

	m_rbAFrame = rbAFrame;
	m_rbBFrame = rbBFrame;

	m_swingSpan1 = Math_PI / 4.0;
	m_swingSpan2 = Math_PI / 4.0;
	m_twistSpan = Math_PI * 2;
	m_biasFactor = 0.3f;
	m_relaxationFactor = 1.0f;

	m_solveTwistLimit = false;
	m_solveSwingLimit = false;

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);

	m_appliedImpulse = 0;
}

bool ConeTwistJointSW::setup(real_t p_step) {
	m_appliedImpulse = real_t(0.);

	//set bias, sign, clear accumulator
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
		if (relPos.length_squared() > CMP_EPSILON) {
			normal[0] = relPos.normalized();
		} else {
			normal[0] = Vector3(real_t(1.0), 0, 0);
		}

		plane_space(normal[0], normal[1], normal[2]);

		for (int i = 0; i < 3; i++) {
			memnew_placement(&m_jac[i], JacobianEntrySW(
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
		//swing1  = btAtan2Fast( b2Axis1.dot(b1Axis2),b2Axis1.dot(b1Axis1) );
		swx = b2Axis1.dot(b1Axis1);
		swy = b2Axis1.dot(b1Axis2);
		swing1 = atan2fast(swy, swx);
		fact = (swy * swy + swx * swx) * thresh * thresh;
		fact = fact / (fact + real_t(1.0));
		swing1 *= fact;
	}

	if (m_swingSpan2 >= real_t(0.05f)) {
		b1Axis3 = A->get_transform().basis.xform(this->m_rbAFrame.basis.get_axis(2));
		//swing2 = btAtan2Fast( b2Axis1.dot(b1Axis3),b2Axis1.dot(b1Axis1) );
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
		Vector3 b2Axis2 = B->get_transform().basis.xform(this->m_rbBFrame.basis.get_axis(1));
		Quat rotationArc = Quat(b2Axis1, b1Axis1);
		Vector3 TwistRef = rotationArc.xform(b2Axis2);
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

void ConeTwistJointSW::solve(real_t timeStep) {

	Vector3 pivotAInW = A->get_transform().xform(m_rbAFrame.origin);
	Vector3 pivotBInW = B->get_transform().xform(m_rbBFrame.origin);

	real_t tau = real_t(0.3);

	//linear part
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
			real_t impulse = depth * tau / timeStep * jacDiagABInv - rel_vel * jacDiagABInv;
			m_appliedImpulse += impulse;
			Vector3 impulse_vector = normal * impulse;
			A->apply_impulse(pivotAInW - A->get_transform().origin, impulse_vector);
			B->apply_impulse(pivotBInW - B->get_transform().origin, -impulse_vector);
		}
	}

	{
		///solve angular part
		const Vector3 &angVelA = A->get_angular_velocity();
		const Vector3 &angVelB = B->get_angular_velocity();

		// solve swing limit
		if (m_solveSwingLimit) {
			real_t amplitude = ((angVelB - angVelA).dot(m_swingAxis) * m_relaxationFactor * m_relaxationFactor + m_swingCorrection * (real_t(1.) / timeStep) * m_biasFactor);
			real_t impulseMag = amplitude * m_kSwing;

			// Clamp the accumulated impulse
			real_t temp = m_accSwingLimitImpulse;
			m_accSwingLimitImpulse = MAX(m_accSwingLimitImpulse + impulseMag, real_t(0.0));
			impulseMag = m_accSwingLimitImpulse - temp;

			Vector3 impulse = m_swingAxis * impulseMag;

			A->apply_torque_impulse(impulse);
			B->apply_torque_impulse(-impulse);
		}

		// solve twist limit
		if (m_solveTwistLimit) {
			real_t amplitude = ((angVelB - angVelA).dot(m_twistAxis) * m_relaxationFactor * m_relaxationFactor + m_twistCorrection * (real_t(1.) / timeStep) * m_biasFactor);
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
}

void ConeTwistJointSW::set_param(PhysicsServer::ConeTwistJointParam p_param, real_t p_value) {

	switch (p_param) {
		case PhysicsServer::CONE_TWIST_JOINT_SWING_SPAN: {

			m_swingSpan1 = p_value;
			m_swingSpan2 = p_value;
		} break;
		case PhysicsServer::CONE_TWIST_JOINT_TWIST_SPAN: {

			m_twistSpan = p_value;
		} break;
		case PhysicsServer::CONE_TWIST_JOINT_BIAS: {

			m_biasFactor = p_value;
		} break;
		case PhysicsServer::CONE_TWIST_JOINT_SOFTNESS: {

			m_limitSoftness = p_value;
		} break;
		case PhysicsServer::CONE_TWIST_JOINT_RELAXATION: {

			m_relaxationFactor = p_value;
		} break;
	}
}

real_t ConeTwistJointSW::get_param(PhysicsServer::ConeTwistJointParam p_param) const {

	switch (p_param) {
		case PhysicsServer::CONE_TWIST_JOINT_SWING_SPAN: {

			return m_swingSpan1;
		} break;
		case PhysicsServer::CONE_TWIST_JOINT_TWIST_SPAN: {

			return m_twistSpan;
		} break;
		case PhysicsServer::CONE_TWIST_JOINT_BIAS: {

			return m_biasFactor;
		} break;
		case PhysicsServer::CONE_TWIST_JOINT_SOFTNESS: {

			return m_limitSoftness;
		} break;
		case PhysicsServer::CONE_TWIST_JOINT_RELAXATION: {

			return m_relaxationFactor;
		} break;
	}

	return 0;
}

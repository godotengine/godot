/*************************************************************************/
/*  godot_hinge_joint_3d.cpp                                             */
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

#include "godot_hinge_joint_3d.h"

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

GodotHingeJoint3D::GodotHingeJoint3D(GodotBody3D *rbA, GodotBody3D *rbB, const Transform3D &frameA, const Transform3D &frameB) :
		GodotJoint3D(_arr, 2) {
	A = rbA;
	B = rbB;

	m_rbAFrame = frameA;
	m_rbBFrame = frameB;
	// flip axis
	m_rbBFrame.basis[0][2] *= real_t(-1.);
	m_rbBFrame.basis[1][2] *= real_t(-1.);
	m_rbBFrame.basis[2][2] *= real_t(-1.);

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);
}

GodotHingeJoint3D::GodotHingeJoint3D(GodotBody3D *rbA, GodotBody3D *rbB, const Vector3 &pivotInA, const Vector3 &pivotInB,
		const Vector3 &axisInA, const Vector3 &axisInB) :
		GodotJoint3D(_arr, 2) {
	A = rbA;
	B = rbB;

	m_rbAFrame.origin = pivotInA;

	// since no frame is given, assume this to be zero angle and just pick rb transform axis
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

	Quaternion rotationArc = Quaternion(axisInA, axisInB);
	Vector3 rbAxisB1 = rotationArc.xform(rbAxisA1);
	Vector3 rbAxisB2 = axisInB.cross(rbAxisB1);

	m_rbBFrame.origin = pivotInB;
	m_rbBFrame.basis = Basis(rbAxisB1.x, rbAxisB2.x, -axisInB.x,
			rbAxisB1.y, rbAxisB2.y, -axisInB.y,
			rbAxisB1.z, rbAxisB2.z, -axisInB.z);

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);
}

bool GodotHingeJoint3D::setup(real_t p_step) {
	dynamic_A = (A->get_mode() > PhysicsServer3D::BODY_MODE_KINEMATIC);
	dynamic_B = (B->get_mode() > PhysicsServer3D::BODY_MODE_KINEMATIC);

	if (!dynamic_A && !dynamic_B) {
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
			memnew_placement(&m_jac[i], GodotJacobianEntry3D(
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

	//calculate two perpendicular jointAxis, orthogonal to hingeAxis
	//these two jointAxis require equal angular velocities for both bodies

	//this is unused for now, it's a todo
	Vector3 jointAxis0local;
	Vector3 jointAxis1local;

	plane_space(m_rbAFrame.basis.get_axis(2), jointAxis0local, jointAxis1local);

	Vector3 jointAxis0 = A->get_transform().basis.xform(jointAxis0local);
	Vector3 jointAxis1 = A->get_transform().basis.xform(jointAxis1local);
	Vector3 hingeAxisWorld = A->get_transform().basis.xform(m_rbAFrame.basis.get_axis(2));

	memnew_placement(&m_jacAng[0], GodotJacobianEntry3D(jointAxis0,
										   A->get_principal_inertia_axes().transposed(),
										   B->get_principal_inertia_axes().transposed(),
										   A->get_inv_inertia(),
										   B->get_inv_inertia()));

	memnew_placement(&m_jacAng[1], GodotJacobianEntry3D(jointAxis1,
										   A->get_principal_inertia_axes().transposed(),
										   B->get_principal_inertia_axes().transposed(),
										   A->get_inv_inertia(),
										   B->get_inv_inertia()));

	memnew_placement(&m_jacAng[2], GodotJacobianEntry3D(hingeAxisWorld,
										   A->get_principal_inertia_axes().transposed(),
										   B->get_principal_inertia_axes().transposed(),
										   A->get_inv_inertia(),
										   B->get_inv_inertia()));

	// Compute limit information
	real_t hingeAngle = get_hinge_angle();

	//set bias, sign, clear accumulator
	m_correction = real_t(0.);
	m_limitSign = real_t(0.);
	m_solveLimit = false;
	m_accLimitImpulse = real_t(0.);

	//if (m_lowerLimit < m_upperLimit)
	if (m_useLimit && m_lowerLimit <= m_upperLimit) {
		//if (hingeAngle <= m_lowerLimit*m_limitSoftness)
		if (hingeAngle <= m_lowerLimit) {
			m_correction = (m_lowerLimit - hingeAngle);
			m_limitSign = 1.0f;
			m_solveLimit = true;
		}
		//else if (hingeAngle >= m_upperLimit*m_limitSoftness)
		else if (hingeAngle >= m_upperLimit) {
			m_correction = m_upperLimit - hingeAngle;
			m_limitSign = -1.0f;
			m_solveLimit = true;
		}
	}

	//Compute K = J*W*J' for hinge axis
	Vector3 axisA = A->get_transform().basis.xform(m_rbAFrame.basis.get_axis(2));
	m_kHinge = 1.0f / (A->compute_angular_impulse_denominator(axisA) +
							  B->compute_angular_impulse_denominator(axisA));

	return true;
}

void GodotHingeJoint3D::solve(real_t p_step) {
	Vector3 pivotAInW = A->get_transform().xform(m_rbAFrame.origin);
	Vector3 pivotBInW = B->get_transform().xform(m_rbBFrame.origin);

	//real_t tau = real_t(0.3);

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
			real_t impulse = depth * tau / p_step * jacDiagABInv - rel_vel * jacDiagABInv;
			m_appliedImpulse += impulse;
			Vector3 impulse_vector = normal * impulse;
			if (dynamic_A) {
				A->apply_impulse(impulse_vector, pivotAInW - A->get_transform().origin);
			}
			if (dynamic_B) {
				B->apply_impulse(-impulse_vector, pivotBInW - B->get_transform().origin);
			}
		}
	}

	{
		///solve angular part

		// get axes in world space
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
			//solve orthogonal angular velocity correction
			real_t relaxation = real_t(1.);
			real_t len = velrelOrthog.length();
			if (len > real_t(0.00001)) {
				Vector3 normal = velrelOrthog.normalized();
				real_t denom = A->compute_angular_impulse_denominator(normal) +
							   B->compute_angular_impulse_denominator(normal);
				// scale for mass and relaxation
				velrelOrthog *= (real_t(1.) / denom) * m_relaxationFactor;
			}

			//solve angular positional correction
			Vector3 angularError = -axisA.cross(axisB) * (real_t(1.) / p_step);
			real_t len2 = angularError.length();
			if (len2 > real_t(0.00001)) {
				Vector3 normal2 = angularError.normalized();
				real_t denom2 = A->compute_angular_impulse_denominator(normal2) +
								B->compute_angular_impulse_denominator(normal2);
				angularError *= (real_t(1.) / denom2) * relaxation;
			}

			if (dynamic_A) {
				A->apply_torque_impulse(-velrelOrthog + angularError);
			}
			if (dynamic_B) {
				B->apply_torque_impulse(velrelOrthog - angularError);
			}

			// solve limit
			if (m_solveLimit) {
				real_t amplitude = ((angVelB - angVelA).dot(axisA) * m_relaxationFactor + m_correction * (real_t(1.) / p_step) * m_biasFactor) * m_limitSign;

				real_t impulseMag = amplitude * m_kHinge;

				// Clamp the accumulated impulse
				real_t temp = m_accLimitImpulse;
				m_accLimitImpulse = MAX(m_accLimitImpulse + impulseMag, real_t(0));
				impulseMag = m_accLimitImpulse - temp;

				Vector3 impulse = axisA * impulseMag * m_limitSign;
				if (dynamic_A) {
					A->apply_torque_impulse(impulse);
				}
				if (dynamic_B) {
					B->apply_torque_impulse(-impulse);
				}
			}
		}

		//apply motor
		if (m_enableAngularMotor) {
			//todo: add limits too
			Vector3 angularLimit(0, 0, 0);

			Vector3 velrel = angVelAroundHingeAxisA - angVelAroundHingeAxisB;
			real_t projRelVel = velrel.dot(axisA);

			real_t desiredMotorVel = m_motorTargetVelocity;
			real_t motor_relvel = desiredMotorVel - projRelVel;

			real_t unclippedMotorImpulse = m_kHinge * motor_relvel;
			//todo: should clip against accumulated impulse
			real_t clippedMotorImpulse = unclippedMotorImpulse > m_maxMotorImpulse ? m_maxMotorImpulse : unclippedMotorImpulse;
			clippedMotorImpulse = clippedMotorImpulse < -m_maxMotorImpulse ? -m_maxMotorImpulse : clippedMotorImpulse;
			Vector3 motorImp = clippedMotorImpulse * axisA;

			if (dynamic_A) {
				A->apply_torque_impulse(motorImp + angularLimit);
			}
			if (dynamic_B) {
				B->apply_torque_impulse(-motorImp - angularLimit);
			}
		}
	}
}

/*
void	HingeJointSW::updateRHS(real_t	timeStep)
{
	(void)timeStep;
}

*/

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

real_t GodotHingeJoint3D::get_hinge_angle() {
	const Vector3 refAxis0 = A->get_transform().basis.xform(m_rbAFrame.basis.get_axis(0));
	const Vector3 refAxis1 = A->get_transform().basis.xform(m_rbAFrame.basis.get_axis(1));
	const Vector3 swingAxis = B->get_transform().basis.xform(m_rbBFrame.basis.get_axis(1));

	return atan2fast(swingAxis.dot(refAxis0), swingAxis.dot(refAxis1));
}

void GodotHingeJoint3D::set_param(PhysicsServer3D::HingeJointParam p_param, real_t p_value) {
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

real_t GodotHingeJoint3D::get_param(PhysicsServer3D::HingeJointParam p_param) const {
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

void GodotHingeJoint3D::set_flag(PhysicsServer3D::HingeJointFlag p_flag, bool p_value) {
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

bool GodotHingeJoint3D::get_flag(PhysicsServer3D::HingeJointFlag p_flag) const {
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

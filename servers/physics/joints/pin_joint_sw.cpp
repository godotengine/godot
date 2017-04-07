/*************************************************************************/
/*  pin_joint_sw.cpp                                                     */
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

#include "pin_joint_sw.h"

bool PinJointSW::setup(real_t p_step) {

	m_appliedImpulse = real_t(0.);

	Vector3 normal(0, 0, 0);

	for (int i = 0; i < 3; i++) {
		normal[i] = 1;
		memnew_placement(&m_jac[i], JacobianEntrySW(
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

void PinJointSW::solve(real_t p_step) {

	Vector3 pivotAInW = A->get_transform().xform(m_pivotInA);
	Vector3 pivotBInW = B->get_transform().xform(m_pivotInB);

	Vector3 normal(0, 0, 0);

	//Vector3 angvelA = A->get_transform().origin.getBasis().transpose() * A->getAngularVelocity();
	//Vector3 angvelB = B->get_transform().origin.getBasis().transpose() * B->getAngularVelocity();

	for (int i = 0; i < 3; i++) {
		normal[i] = 1;
		real_t jacDiagABInv = real_t(1.) / m_jac[i].getDiagonal();

		Vector3 rel_pos1 = pivotAInW - A->get_transform().origin;
		Vector3 rel_pos2 = pivotBInW - B->get_transform().origin;
		//this jacobian entry could be re-used for all iterations

		Vector3 vel1 = A->get_velocity_in_local_point(rel_pos1);
		Vector3 vel2 = B->get_velocity_in_local_point(rel_pos2);
		Vector3 vel = vel1 - vel2;

		real_t rel_vel;
		rel_vel = normal.dot(vel);

		/*
		//velocity error (first order error)
		real_t rel_vel = m_jac[i].getRelativeVelocity(A->getLinearVelocity(),angvelA,
														B->getLinearVelocity(),angvelB);
	*/

		//positional error (zeroth order error)
		real_t depth = -(pivotAInW - pivotBInW).dot(normal); //this is the error projected on the normal

		real_t impulse = depth * m_tau / p_step * jacDiagABInv - m_damping * rel_vel * jacDiagABInv;

		real_t impulseClamp = m_impulseClamp;
		if (impulseClamp > 0) {
			if (impulse < -impulseClamp)
				impulse = -impulseClamp;
			if (impulse > impulseClamp)
				impulse = impulseClamp;
		}

		m_appliedImpulse += impulse;
		Vector3 impulse_vector = normal * impulse;
		A->apply_impulse(pivotAInW - A->get_transform().origin, impulse_vector);
		B->apply_impulse(pivotBInW - B->get_transform().origin, -impulse_vector);

		normal[i] = 0;
	}
}

void PinJointSW::set_param(PhysicsServer::PinJointParam p_param, real_t p_value) {

	switch (p_param) {
		case PhysicsServer::PIN_JOINT_BIAS: m_tau = p_value; break;
		case PhysicsServer::PIN_JOINT_DAMPING: m_damping = p_value; break;
		case PhysicsServer::PIN_JOINT_IMPULSE_CLAMP: m_impulseClamp = p_value; break;
	}
}

real_t PinJointSW::get_param(PhysicsServer::PinJointParam p_param) const {

	switch (p_param) {
		case PhysicsServer::PIN_JOINT_BIAS: return m_tau;
		case PhysicsServer::PIN_JOINT_DAMPING: return m_damping;
		case PhysicsServer::PIN_JOINT_IMPULSE_CLAMP: return m_impulseClamp;
	}

	return 0;
}

PinJointSW::PinJointSW(BodySW *p_body_a, const Vector3 &p_pos_a, BodySW *p_body_b, const Vector3 &p_pos_b)
	: JointSW(_arr, 2) {

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

PinJointSW::~PinJointSW() {
}

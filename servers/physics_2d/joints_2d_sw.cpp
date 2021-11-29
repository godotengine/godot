/*************************************************************************/
/*  joints_2d_sw.cpp                                                     */
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

#include "joints_2d_sw.h"

#include "space_2d_sw.h"

//based on chipmunk joint constraints

/* Copyright (c) 2007 Scott Lembcke
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

static inline real_t k_scalar(Body2DSW *a, Body2DSW *b, const Vector2 &rA, const Vector2 &rB, const Vector2 &n) {
	real_t value = 0;

	{
		value += a->get_inv_mass();
		real_t rcn = rA.cross(n);
		value += a->get_inv_inertia() * rcn * rcn;
	}

	if (b) {
		value += b->get_inv_mass();
		real_t rcn = rB.cross(n);
		value += b->get_inv_inertia() * rcn * rcn;
	}

	return value;
}

static inline Vector2
relative_velocity(Body2DSW *a, Body2DSW *b, Vector2 rA, Vector2 rB) {
	Vector2 sum = a->get_linear_velocity() - rA.tangent() * a->get_angular_velocity();
	if (b) {
		return (b->get_linear_velocity() - rB.tangent() * b->get_angular_velocity()) - sum;
	} else {
		return -sum;
	}
}

static inline real_t
normal_relative_velocity(Body2DSW *a, Body2DSW *b, Vector2 rA, Vector2 rB, Vector2 n) {
	return relative_velocity(a, b, rA, rB).dot(n);
}

bool PinJoint2DSW::setup(real_t p_step) {
	if ((A->get_mode() <= Physics2DServer::BODY_MODE_KINEMATIC) && (B->get_mode() <= Physics2DServer::BODY_MODE_KINEMATIC)) {
		return false;
	}

	Space2DSW *space = A->get_space();
	ERR_FAIL_COND_V(!space, false);

	rA = A->get_transform().basis_xform(anchor_A);
	rB = B ? B->get_transform().basis_xform(anchor_B) : anchor_B;

	real_t B_inv_mass = B ? B->get_inv_mass() : 0.0;

	Transform2D K1;
	K1[0].x = A->get_inv_mass() + B_inv_mass;
	K1[1].x = 0.0f;
	K1[0].y = 0.0f;
	K1[1].y = A->get_inv_mass() + B_inv_mass;

	Transform2D K2;
	K2[0].x = A->get_inv_inertia() * rA.y * rA.y;
	K2[1].x = -A->get_inv_inertia() * rA.x * rA.y;
	K2[0].y = -A->get_inv_inertia() * rA.x * rA.y;
	K2[1].y = A->get_inv_inertia() * rA.x * rA.x;

	Transform2D K;
	K[0] = K1[0] + K2[0];
	K[1] = K1[1] + K2[1];

	if (B) {
		Transform2D K3;
		K3[0].x = B->get_inv_inertia() * rB.y * rB.y;
		K3[1].x = -B->get_inv_inertia() * rB.x * rB.y;
		K3[0].y = -B->get_inv_inertia() * rB.x * rB.y;
		K3[1].y = B->get_inv_inertia() * rB.x * rB.x;

		K[0] += K3[0];
		K[1] += K3[1];
	}

	K[0].x += softness;
	K[1].y += softness;

	M = K.affine_inverse();

	Vector2 gA = rA + A->get_transform().get_origin();
	Vector2 gB = B ? rB + B->get_transform().get_origin() : rB;

	Vector2 delta = gB - gA;

	bias = delta * -(get_bias() == 0 ? space->get_constraint_bias() : get_bias()) * (1.0 / p_step);

	// apply accumulated impulse
	A->apply_impulse(rA, -P);
	if (B) {
		B->apply_impulse(rB, P);
	}

	return true;
}

inline Vector2 custom_cross(const Vector2 &p_vec, real_t p_other) {
	return Vector2(p_other * p_vec.y, -p_other * p_vec.x);
}

void PinJoint2DSW::solve(real_t p_step) {
	// compute relative velocity
	Vector2 vA = A->get_linear_velocity() - custom_cross(rA, A->get_angular_velocity());

	Vector2 rel_vel;
	if (B) {
		rel_vel = B->get_linear_velocity() - custom_cross(rB, B->get_angular_velocity()) - vA;
	} else {
		rel_vel = -vA;
	}

	Vector2 impulse = M.basis_xform(bias - rel_vel - Vector2(softness, softness) * P);

	A->apply_impulse(rA, -impulse);
	if (B) {
		B->apply_impulse(rB, impulse);
	}

	P += impulse;
}

void PinJoint2DSW::set_param(Physics2DServer::PinJointParam p_param, real_t p_value) {
	if (p_param == Physics2DServer::PIN_JOINT_SOFTNESS) {
		softness = p_value;
	}
}

real_t PinJoint2DSW::get_param(Physics2DServer::PinJointParam p_param) const {
	if (p_param == Physics2DServer::PIN_JOINT_SOFTNESS) {
		return softness;
	}
	ERR_FAIL_V(0);
}

PinJoint2DSW::PinJoint2DSW(const Vector2 &p_pos, Body2DSW *p_body_a, Body2DSW *p_body_b) :
		Joint2DSW(_arr, p_body_b ? 2 : 1) {
	A = p_body_a;
	B = p_body_b;
	anchor_A = p_body_a->get_inv_transform().xform(p_pos);
	anchor_B = p_body_b ? p_body_b->get_inv_transform().xform(p_pos) : p_pos;

	softness = 0;

	p_body_a->add_constraint(this, 0);
	if (p_body_b) {
		p_body_b->add_constraint(this, 1);
	}
}

PinJoint2DSW::~PinJoint2DSW() {
	if (A) {
		A->remove_constraint(this);
	}
	if (B) {
		B->remove_constraint(this);
	}
}

//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////

static inline void
k_tensor(Body2DSW *a, Body2DSW *b, Vector2 r1, Vector2 r2, Vector2 *k1, Vector2 *k2) {
	// calculate mass matrix
	// If I wasn't lazy and wrote a proper matrix class, this wouldn't be so gross...
	real_t k11, k12, k21, k22;
	real_t m_sum = a->get_inv_mass() + b->get_inv_mass();

	// start with I*m_sum
	k11 = m_sum;
	k12 = 0.0f;
	k21 = 0.0f;
	k22 = m_sum;

	// add the influence from r1
	real_t a_i_inv = a->get_inv_inertia();
	real_t r1xsq = r1.x * r1.x * a_i_inv;
	real_t r1ysq = r1.y * r1.y * a_i_inv;
	real_t r1nxy = -r1.x * r1.y * a_i_inv;
	k11 += r1ysq;
	k12 += r1nxy;
	k21 += r1nxy;
	k22 += r1xsq;

	// add the influnce from r2
	real_t b_i_inv = b->get_inv_inertia();
	real_t r2xsq = r2.x * r2.x * b_i_inv;
	real_t r2ysq = r2.y * r2.y * b_i_inv;
	real_t r2nxy = -r2.x * r2.y * b_i_inv;
	k11 += r2ysq;
	k12 += r2nxy;
	k21 += r2nxy;
	k22 += r2xsq;

	// invert
	real_t determinant = k11 * k22 - k12 * k21;
	ERR_FAIL_COND(determinant == 0.0);

	real_t det_inv = 1.0f / determinant;
	*k1 = Vector2(k22 * det_inv, -k12 * det_inv);
	*k2 = Vector2(-k21 * det_inv, k11 * det_inv);
}

static _FORCE_INLINE_ Vector2
mult_k(const Vector2 &vr, const Vector2 &k1, const Vector2 &k2) {
	return Vector2(vr.dot(k1), vr.dot(k2));
}

bool GrooveJoint2DSW::setup(real_t p_step) {
	if ((A->get_mode() <= Physics2DServer::BODY_MODE_KINEMATIC) && (B->get_mode() <= Physics2DServer::BODY_MODE_KINEMATIC)) {
		return false;
	}

	// calculate endpoints in worldspace
	Vector2 ta = A->get_transform().xform(A_groove_1);
	Vector2 tb = A->get_transform().xform(A_groove_2);
	Space2DSW *space = A->get_space();

	// calculate axis
	Vector2 n = -(tb - ta).tangent().normalized();
	real_t d = ta.dot(n);

	xf_normal = n;
	rB = B->get_transform().basis_xform(B_anchor);

	// calculate tangential distance along the axis of rB
	real_t td = (B->get_transform().get_origin() + rB).cross(n);
	// calculate clamping factor and rB
	if (td <= ta.cross(n)) {
		clamp = 1.0f;
		rA = ta - A->get_transform().get_origin();
	} else if (td >= tb.cross(n)) {
		clamp = -1.0f;
		rA = tb - A->get_transform().get_origin();
	} else {
		clamp = 0.0f;
		//joint->r1 = cpvsub(cpvadd(cpvmult(cpvperp(n), -td), cpvmult(n, d)), a->p);
		rA = ((-n.tangent() * -td) + n * d) - A->get_transform().get_origin();
	}

	// Calculate mass tensor
	k_tensor(A, B, rA, rB, &k1, &k2);

	// compute max impulse
	jn_max = get_max_force() * p_step;

	// calculate bias velocity
	//cpVect delta = cpvsub(cpvadd(b->p, joint->r2), cpvadd(a->p, joint->r1));
	//joint->bias = cpvclamp(cpvmult(delta, -joint->constraint.biasCoef*dt_inv), joint->constraint.maxBias);

	Vector2 delta = (B->get_transform().get_origin() + rB) - (A->get_transform().get_origin() + rA);

	real_t _b = get_bias();
	gbias = (delta * -(_b == 0 ? space->get_constraint_bias() : _b) * (1.0 / p_step)).limit_length(get_max_bias());

	// apply accumulated impulse
	A->apply_impulse(rA, -jn_acc);
	B->apply_impulse(rB, jn_acc);

	correct = true;
	return true;
}

void GrooveJoint2DSW::solve(real_t p_step) {
	// compute impulse
	Vector2 vr = relative_velocity(A, B, rA, rB);

	Vector2 j = mult_k(gbias - vr, k1, k2);
	Vector2 jOld = jn_acc;
	j += jOld;

	jn_acc = (((clamp * j.cross(xf_normal)) > 0) ? j : j.project(xf_normal)).limit_length(jn_max);

	j = jn_acc - jOld;

	A->apply_impulse(rA, -j);
	B->apply_impulse(rB, j);
}

GrooveJoint2DSW::GrooveJoint2DSW(const Vector2 &p_a_groove1, const Vector2 &p_a_groove2, const Vector2 &p_b_anchor, Body2DSW *p_body_a, Body2DSW *p_body_b) :
		Joint2DSW(_arr, 2) {
	A = p_body_a;
	B = p_body_b;

	A_groove_1 = A->get_inv_transform().xform(p_a_groove1);
	A_groove_2 = A->get_inv_transform().xform(p_a_groove2);
	B_anchor = B->get_inv_transform().xform(p_b_anchor);
	A_groove_normal = -(A_groove_2 - A_groove_1).normalized().tangent();

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);
}

GrooveJoint2DSW::~GrooveJoint2DSW() {
	A->remove_constraint(this);
	B->remove_constraint(this);
}

//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////

bool DampedSpringJoint2DSW::setup(real_t p_step) {
	if ((A->get_mode() <= Physics2DServer::BODY_MODE_KINEMATIC) && (B->get_mode() <= Physics2DServer::BODY_MODE_KINEMATIC)) {
		return false;
	}

	rA = A->get_transform().basis_xform(anchor_A);
	rB = B->get_transform().basis_xform(anchor_B);

	Vector2 delta = (B->get_transform().get_origin() + rB) - (A->get_transform().get_origin() + rA);
	real_t dist = delta.length();

	if (dist) {
		n = delta / dist;
	} else {
		n = Vector2();
	}

	real_t k = k_scalar(A, B, rA, rB, n);
	n_mass = 1.0f / k;

	target_vrn = 0.0f;
	v_coef = 1.0f - Math::exp(-damping * (p_step)*k);

	// apply spring force
	real_t f_spring = (rest_length - dist) * stiffness;
	Vector2 j = n * f_spring * (p_step);

	A->apply_impulse(rA, -j);
	B->apply_impulse(rB, j);

	return true;
}

void DampedSpringJoint2DSW::solve(real_t p_step) {
	// compute relative velocity
	real_t vrn = normal_relative_velocity(A, B, rA, rB, n) - target_vrn;

	// compute velocity loss from drag
	// not 100% certain this is derived correctly, though it makes sense
	real_t v_damp = -vrn * v_coef;
	target_vrn = vrn + v_damp;
	Vector2 j = n * v_damp * n_mass;

	A->apply_impulse(rA, -j);
	B->apply_impulse(rB, j);
}

void DampedSpringJoint2DSW::set_param(Physics2DServer::DampedStringParam p_param, real_t p_value) {
	switch (p_param) {
		case Physics2DServer::DAMPED_STRING_REST_LENGTH: {
			rest_length = p_value;
		} break;
		case Physics2DServer::DAMPED_STRING_DAMPING: {
			damping = p_value;
		} break;
		case Physics2DServer::DAMPED_STRING_STIFFNESS: {
			stiffness = p_value;
		} break;
	}
}

real_t DampedSpringJoint2DSW::get_param(Physics2DServer::DampedStringParam p_param) const {
	switch (p_param) {
		case Physics2DServer::DAMPED_STRING_REST_LENGTH: {
			return rest_length;
		} break;
		case Physics2DServer::DAMPED_STRING_DAMPING: {
			return damping;
		} break;
		case Physics2DServer::DAMPED_STRING_STIFFNESS: {
			return stiffness;
		} break;
	}

	ERR_FAIL_V(0);
}

DampedSpringJoint2DSW::DampedSpringJoint2DSW(const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, Body2DSW *p_body_a, Body2DSW *p_body_b) :
		Joint2DSW(_arr, 2) {
	A = p_body_a;
	B = p_body_b;
	anchor_A = A->get_inv_transform().xform(p_anchor_a);
	anchor_B = B->get_inv_transform().xform(p_anchor_b);

	rest_length = p_anchor_a.distance_to(p_anchor_b);
	stiffness = 20;
	damping = 1.5;

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);
}

DampedSpringJoint2DSW::~DampedSpringJoint2DSW() {
	A->remove_constraint(this);
	B->remove_constraint(this);
}

/*************************************************************************/
/*  joints_2d_sw.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef JOINTS_2D_SW_H
#define JOINTS_2D_SW_H

#include "body_2d_sw.h"
#include "constraint_2d_sw.h"

class Joint2DSW : public Constraint2DSW {
	real_t max_force;
	real_t bias;
	real_t max_bias;

public:
	_FORCE_INLINE_ void set_max_force(real_t p_force) { max_force = p_force; }
	_FORCE_INLINE_ real_t get_max_force() const { return max_force; }

	_FORCE_INLINE_ void set_bias(real_t p_bias) { bias = p_bias; }
	_FORCE_INLINE_ real_t get_bias() const { return bias; }

	_FORCE_INLINE_ void set_max_bias(real_t p_bias) { max_bias = p_bias; }
	_FORCE_INLINE_ real_t get_max_bias() const { return max_bias; }

	virtual PhysicsServer2D::JointType get_type() const = 0;
	Joint2DSW(Body2DSW **p_body_ptr = nullptr, int p_body_count = 0) :
			Constraint2DSW(p_body_ptr, p_body_count) {
		bias = 0;
		max_force = max_bias = 3.40282e+38;
	};
};

class PinJoint2DSW : public Joint2DSW {
	union {
		struct {
			Body2DSW *A;
			Body2DSW *B;
		};

		Body2DSW *_arr[2];
	};

	Transform2D M;
	Vector2 rA, rB;
	Vector2 anchor_A;
	Vector2 anchor_B;
	Vector2 bias;
	Vector2 P;
	real_t softness;

public:
	virtual PhysicsServer2D::JointType get_type() const { return PhysicsServer2D::JOINT_PIN; }

	virtual bool setup(real_t p_step);
	virtual void solve(real_t p_step);

	void set_param(PhysicsServer2D::PinJointParam p_param, real_t p_value);
	real_t get_param(PhysicsServer2D::PinJointParam p_param) const;

	PinJoint2DSW(const Vector2 &p_pos, Body2DSW *p_body_a, Body2DSW *p_body_b = nullptr);
	~PinJoint2DSW();
};

class GrooveJoint2DSW : public Joint2DSW {
	union {
		struct {
			Body2DSW *A;
			Body2DSW *B;
		};

		Body2DSW *_arr[2];
	};

	Vector2 A_groove_1;
	Vector2 A_groove_2;
	Vector2 A_groove_normal;
	Vector2 B_anchor;
	Vector2 jn_acc;
	Vector2 gbias;
	real_t jn_max;
	real_t clamp;
	Vector2 xf_normal;
	Vector2 rA, rB;
	Vector2 k1, k2;

	bool correct;

public:
	virtual PhysicsServer2D::JointType get_type() const { return PhysicsServer2D::JOINT_GROOVE; }

	virtual bool setup(real_t p_step);
	virtual void solve(real_t p_step);

	GrooveJoint2DSW(const Vector2 &p_a_groove1, const Vector2 &p_a_groove2, const Vector2 &p_b_anchor, Body2DSW *p_body_a, Body2DSW *p_body_b);
	~GrooveJoint2DSW();
};

class DampedSpringJoint2DSW : public Joint2DSW {
	union {
		struct {
			Body2DSW *A;
			Body2DSW *B;
		};

		Body2DSW *_arr[2];
	};

	Vector2 anchor_A;
	Vector2 anchor_B;

	real_t rest_length;
	real_t damping;
	real_t stiffness;

	Vector2 rA, rB;
	Vector2 n;
	real_t n_mass;
	real_t target_vrn;
	real_t v_coef;

public:
	virtual PhysicsServer2D::JointType get_type() const { return PhysicsServer2D::JOINT_DAMPED_SPRING; }

	virtual bool setup(real_t p_step);
	virtual void solve(real_t p_step);

	void set_param(PhysicsServer2D::DampedSpringParam p_param, real_t p_value);
	real_t get_param(PhysicsServer2D::DampedSpringParam p_param) const;

	DampedSpringJoint2DSW(const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, Body2DSW *p_body_a, Body2DSW *p_body_b);
	~DampedSpringJoint2DSW();
};

#endif // JOINTS_2D_SW_H

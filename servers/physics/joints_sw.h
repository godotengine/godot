/*************************************************************************/
/*  joints_sw.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef JOINTS_SW_H
#define JOINTS_SW_H

#include "constraint_sw.h"
#include "body_sw.h"



class JointSW : public ConstraintSW {


public:

	virtual PhysicsServer::JointType get_type() const=0;
	_FORCE_INLINE_ JointSW(BodySW **p_body_ptr=NULL,int p_body_count=0) : ConstraintSW(p_body_ptr,p_body_count) {
	}

};

#if 0
class PinJointSW : public JointSW {

	union {
		struct {
			BodySW *A;
			BodySW *B;
		};

		BodySW *_arr[2];
	};

	Vector2 anchor_A;
	Vector2 anchor_B;
	real_t dist;
	real_t jn_acc;
	real_t jn_max;
	real_t max_distance;
	real_t mass_normal;
	real_t bias;

	Vector2 rA,rB;
	Vector2 n; //normal
	bool correct;


public:

	virtual PhysicsServer::JointType get_type() const { return PhysicsServer::JOINT_PIN; }

	virtual bool setup(float p_step);
	virtual void solve(float p_step);


	PinJointSW(const Vector2& p_pos,BodySW* p_body_a,BodySW* p_body_b=NULL);
	~PinJointSW();
};


class GrooveJointSW : public JointSW {

	union {
		struct {
			BodySW *A;
			BodySW *B;
		};

		BodySW *_arr[2];
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
	Vector2 rA,rB;
	Vector2 k1,k2;


	bool correct;

public:

	virtual PhysicsServer::JointType get_type() const { return PhysicsServer::JOINT_GROOVE; }

	virtual bool setup(float p_step);
	virtual void solve(float p_step);


	GrooveJointSW(const Vector2& p_a_groove1,const Vector2& p_a_groove2, const Vector2& p_b_anchor, BodySW* p_body_a,BodySW* p_body_b);
	~GrooveJointSW();
};


class DampedSpringJointSW : public JointSW {

	union {
		struct {
			BodySW *A;
			BodySW *B;
		};

		BodySW *_arr[2];
	};


	Vector2 anchor_A;
	Vector2 anchor_B;

	real_t rest_length;
	real_t damping;
	real_t stiffness;

	Vector2 rA,rB;
	Vector2 n;
	real_t n_mass;
	real_t target_vrn;
	real_t v_coef;

public:

	virtual PhysicsServer::JointType get_type() const { return PhysicsServer::JOINT_DAMPED_SPRING; }

	virtual bool setup(float p_step);
	virtual void solve(float p_step);

	void set_param(PhysicsServer::DampedStringParam p_param, real_t p_value);
	real_t get_param(PhysicsServer::DampedStringParam p_param) const;

	DampedSpringJointSW(const Vector2& p_anchor_a,const Vector2& p_anchor_b, BodySW* p_body_a,BodySW* p_body_b);
	~DampedSpringJointSW();
};
#endif

#endif // JOINTS__SW_H

/*************************************************************************/
/*  hinge_joint_3d_sw.h                                                  */
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

/*
Adapted to Godot from the Bullet library.
*/

#ifndef HINGE_JOINT_SW_H
#define HINGE_JOINT_SW_H

#include "servers/physics_3d/joints/jacobian_entry_3d_sw.h"
#include "servers/physics_3d/joints_3d_sw.h"

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

class HingeJoint3DSW : public Joint3DSW {
	union {
		struct {
			Body3DSW *A;
			Body3DSW *B;
		};

		Body3DSW *_arr[2];
	};

	JacobianEntry3DSW m_jac[3]; //3 orthogonal linear constraints
	JacobianEntry3DSW m_jacAng[3]; //2 orthogonal angular constraints+ 1 for limit/motor

	Transform m_rbAFrame; // constraint axii. Assumes z is hinge axis.
	Transform m_rbBFrame;

	real_t m_motorTargetVelocity;
	real_t m_maxMotorImpulse;

	real_t m_limitSoftness;
	real_t m_biasFactor;
	real_t m_relaxationFactor;

	real_t m_lowerLimit;
	real_t m_upperLimit;

	real_t m_kHinge;

	real_t m_limitSign;
	real_t m_correction;

	real_t m_accLimitImpulse;

	real_t tau;

	bool m_useLimit;
	bool m_angularOnly;
	bool m_enableAngularMotor;
	bool m_solveLimit;

	real_t m_appliedImpulse;

public:
	virtual PhysicsServer3D::JointType get_type() const { return PhysicsServer3D::JOINT_HINGE; }

	virtual bool setup(real_t p_step);
	virtual void solve(real_t p_step);

	real_t get_hinge_angle();

	void set_param(PhysicsServer3D::HingeJointParam p_param, real_t p_value);
	real_t get_param(PhysicsServer3D::HingeJointParam p_param) const;

	void set_flag(PhysicsServer3D::HingeJointFlag p_flag, bool p_value);
	bool get_flag(PhysicsServer3D::HingeJointFlag p_flag) const;

	HingeJoint3DSW(Body3DSW *rbA, Body3DSW *rbB, const Transform &frameA, const Transform &frameB);
	HingeJoint3DSW(Body3DSW *rbA, Body3DSW *rbB, const Vector3 &pivotInA, const Vector3 &pivotInB, const Vector3 &axisInA, const Vector3 &axisInB);
};

#endif // HINGE_JOINT_SW_H

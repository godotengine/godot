/**************************************************************************/
/*  pin_joint_sw.h                                                        */
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

#ifndef PIN_JOINT_SW_H
#define PIN_JOINT_SW_H

#include "servers/physics/joints/jacobian_entry_sw.h"
#include "servers/physics/joints_sw.h"

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

class PinJointSW : public JointSW {
	union {
		struct {
			BodySW *A;
			BodySW *B;
		};

		BodySW *_arr[2];
	};

	real_t m_tau; //bias
	real_t m_damping;
	real_t m_impulseClamp;
	real_t m_appliedImpulse;

	JacobianEntrySW m_jac[3]; //3 orthogonal linear constraints

	Vector3 m_pivotInA;
	Vector3 m_pivotInB;

public:
	virtual PhysicsServer::JointType get_type() const { return PhysicsServer::JOINT_PIN; }

	virtual bool setup(real_t p_step);
	virtual void solve(real_t p_step);

	void set_param(PhysicsServer::PinJointParam p_param, real_t p_value);
	real_t get_param(PhysicsServer::PinJointParam p_param) const;

	void set_pos_a(const Vector3 &p_pos) { m_pivotInA = p_pos; }
	void set_pos_b(const Vector3 &p_pos) { m_pivotInB = p_pos; }

	Vector3 get_position_a() { return m_pivotInA; }
	Vector3 get_position_b() { return m_pivotInB; }

	PinJointSW(BodySW *p_body_a, const Vector3 &p_pos_a, BodySW *p_body_b, const Vector3 &p_pos_b);
	~PinJointSW();
};

#endif // PIN_JOINT_SW_H

/**************************************************************************/
/*  pin_joint_bullet.h                                                    */
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

#ifndef PIN_JOINT_BULLET_H
#define PIN_JOINT_BULLET_H

#include "joint_bullet.h"

/**
	@author AndreaCatania
*/

class RigidBodyBullet;

class PinJointBullet : public JointBullet {
	class btPoint2PointConstraint *p2pConstraint;

public:
	PinJointBullet(RigidBodyBullet *p_body_a, const Vector3 &p_pos_a, RigidBodyBullet *p_body_b, const Vector3 &p_pos_b);
	~PinJointBullet();

	virtual PhysicsServer::JointType get_type() const { return PhysicsServer::JOINT_PIN; }

	void set_param(PhysicsServer::PinJointParam p_param, real_t p_value);
	real_t get_param(PhysicsServer::PinJointParam p_param) const;

	void setPivotInA(const Vector3 &p_pos);
	void setPivotInB(const Vector3 &p_pos);

	Vector3 getPivotInA();
	Vector3 getPivotInB();
};

#endif // PIN_JOINT_BULLET_H

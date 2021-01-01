/*************************************************************************/
/*  hinge_joint_bullet.h                                                 */
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

#ifndef HINGE_JOINT_BULLET_H
#define HINGE_JOINT_BULLET_H

#include "joint_bullet.h"

/**
	@author AndreaCatania
*/

class HingeJointBullet : public JointBullet {
	class btHingeConstraint *hingeConstraint;

public:
	HingeJointBullet(RigidBodyBullet *rbA, RigidBodyBullet *rbB, const Transform &frameA, const Transform &frameB);
	HingeJointBullet(RigidBodyBullet *rbA, RigidBodyBullet *rbB, const Vector3 &pivotInA, const Vector3 &pivotInB, const Vector3 &axisInA, const Vector3 &axisInB);

	virtual PhysicsServer3D::JointType get_type() const { return PhysicsServer3D::JOINT_HINGE; }

	real_t get_hinge_angle();

	void set_param(PhysicsServer3D::HingeJointParam p_param, real_t p_value);
	real_t get_param(PhysicsServer3D::HingeJointParam p_param) const;

	void set_flag(PhysicsServer3D::HingeJointFlag p_flag, bool p_value);
	bool get_flag(PhysicsServer3D::HingeJointFlag p_flag) const;
};
#endif

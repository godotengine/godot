/*************************************************************************/
/*  cone_twist_joint_bullet.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef CONE_TWIST_JOINT_BULLET_H
#define CONE_TWIST_JOINT_BULLET_H

#include "joint_bullet.h"

/**
	@author AndreaCatania
*/

class RigidBodyBullet;

class ConeTwistJointBullet : public JointBullet {
	class btConeTwistConstraint *coneConstraint;

public:
	ConeTwistJointBullet(RigidBodyBullet *rbA, RigidBodyBullet *rbB, const Transform &rbAFrame, const Transform &rbBFrame);

	virtual PhysicsServer::JointType get_type() const { return PhysicsServer::JOINT_CONE_TWIST; }

	void set_param(PhysicsServer::ConeTwistJointParam p_param, real_t p_value);
	real_t get_param(PhysicsServer::ConeTwistJointParam p_param) const;
};

#endif // CONE_TWIST_JOINT_BULLET_H

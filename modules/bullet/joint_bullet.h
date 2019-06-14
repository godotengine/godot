/*************************************************************************/
/*  joint_bullet.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef JOINT_BULLET_H
#define JOINT_BULLET_H

#include "bullet_utilities.h"
#include "rid_bullet.h"
#include "servers/physics_server.h"
#include <BulletDynamics/ConstraintSolver/btTypedConstraint.h>
#include <BulletDynamics/Featherstone/btMultiBodyConstraint.h>

/**
	@author AndreaCatania
*/

class RigidBodyBullet;
class btTypedConstraint;
class RigidBodyBullet;
class SpaceBullet;
class BoneBullet;

class JointBullet : public RIDBullet {

protected:
	SpaceBullet *space;
	BoneBullet *body_a;
	BoneBullet *body_b;
	btTypedConstraint *constraint;
	btMultiBodyConstraint *multibody_constraint;
	bool disabled_collisions_between_bodies;

public:
	JointBullet();
	virtual ~JointBullet();

	BoneBullet *get_body_a() const {
		return body_a;
	}

	BoneBullet *get_body_b() const {
		return body_b;
	}

	virtual void reload_internal() {}
	virtual PhysicsServer::JointType get_type() const = 0;

	virtual void setup(btTypedConstraint *p_constraint);
	virtual void setup(btMultiBodyConstraint *p_constraint, BoneBullet *p_body_a, BoneBullet *p_body_b);
	virtual void set_space(SpaceBullet *p_space);

	void disable_collisions_between_bodies(const bool p_disabled);
	_FORCE_INLINE_ bool is_disabled_collisions_between_bodies() const { return disabled_collisions_between_bodies; }

	_FORCE_INLINE_ bool is_multi_joint() { return constraint == NULL; }
	_FORCE_INLINE_ btTypedConstraint *get_bt_constraint() { return constraint; }
	_FORCE_INLINE_ btMultiBodyConstraint *get_bt_mb_constraint() { return multibody_constraint; }

	virtual void clear_internal_joint();
};
#endif

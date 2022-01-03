/*************************************************************************/
/*  constraint_bullet.h                                                  */
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

#ifndef CONSTRAINT_BULLET_H
#define CONSTRAINT_BULLET_H

#include "bullet_utilities.h"
#include "rid_bullet.h"

#include <BulletDynamics/ConstraintSolver/btTypedConstraint.h>

/**
	@author AndreaCatania
*/

class RigidBodyBullet;
class SpaceBullet;
class btTypedConstraint;

class ConstraintBullet : public RIDBullet {
protected:
	SpaceBullet *space = nullptr;
	btTypedConstraint *constraint = nullptr;
	bool disabled_collisions_between_bodies = true;

public:
	ConstraintBullet();

	virtual void setup(btTypedConstraint *p_constraint);
	virtual void set_space(SpaceBullet *p_space);
	virtual void destroy_internal_constraint();

	void disable_collisions_between_bodies(const bool p_disabled);
	_FORCE_INLINE_ bool is_disabled_collisions_between_bodies() const { return disabled_collisions_between_bodies; }

public:
	virtual ~ConstraintBullet() {
		bulletdelete(constraint);
		constraint = nullptr;
	}

	_FORCE_INLINE_ btTypedConstraint *get_bt_constraint() { return constraint; }
};
#endif

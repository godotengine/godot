/*************************************************************************/
/*  constraint_bullet.cpp                                                */
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

#include "constraint_bullet.h"

#include "collision_object_bullet.h"
#include "space_bullet.h"

ConstraintBullet::ConstraintBullet() {}

void ConstraintBullet::setup(btTypedConstraint *p_constraint) {
	constraint = p_constraint;
	constraint->setUserConstraintPtr(this);
}

void ConstraintBullet::set_space(SpaceBullet *p_space) {
	space = p_space;
}

void ConstraintBullet::destroy_internal_constraint() {
	space->remove_constraint(this);
}

void ConstraintBullet::disable_collisions_between_bodies(const bool p_disabled) {
	disabled_collisions_between_bodies = p_disabled;

	if (space) {
		space->remove_constraint(this);
		space->add_constraint(this, disabled_collisions_between_bodies);
	}
}

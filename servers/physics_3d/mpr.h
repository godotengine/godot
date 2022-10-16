/*************************************************************************/
/*  mpr.h                                                                */
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

#ifndef MPR_H
#define MPR_H

#include "godot_collision_solver_3d.h"
#include "servers/physics_3d/godot_shape_3d.h"

/*
 * The following routine implements Minkowski Portal Refinement as described in
 *
 * G. Snethen, Xenocollide: Complex collision made simple, Game Programming Gems 7, 2008
 *
 * It is adapted from an implementation in Simbody (https://github.com/simbody/simbody),
 * which is released under the following license.
 */

/* -------------------------------------------------------------------------- *
 *                        SimTK Simbody: SimTKmath                            *
 * -------------------------------------------------------------------------- *
 * This is part of the SimTK biosimulation toolkit originating from           *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

template <bool withMargin>
static _FORCE_INLINE_ Vector3 _compute_support(const GodotShape3D *obj1, const GodotShape3D *obj2, const Transform3D &transform1, const Transform3D &transform2, const Vector3 &direction, real_t total_margin) {
	Vector3 p1 = transform1.xform(obj1->get_support(transform1.basis.xform_inv(direction).normalized()));
	Vector3 p2 = transform2.xform(obj2->get_support(transform2.basis.xform_inv(-direction).normalized()));
	Vector3 s = p1 - p2;
	if (withMargin) {
		s += total_margin * direction;
	}
	return s;
}

template <bool withMargin>
static _FORCE_INLINE_ Vector3 _collision_mpr(const GodotShape3D *obj1, const Transform3D &p_transform_a, const GodotShape3D *obj2, const Transform3D &p_transform_b, real_t p_margin_a, real_t p_margin_b) {
	real_t total_margin = (withMargin ? p_margin_a + p_margin_b : 0);

	// Compute a point that is known to be inside the Minkowski difference, and
	// a ray directed from that point to the origin.

	Vector3 v0 = _compute_support<withMargin>(obj1, obj2, p_transform_a, p_transform_b, Vector3(1, 0, 0), total_margin) + _compute_support<withMargin>(obj1, obj2, p_transform_a, p_transform_b, Vector3(-1, 0, 0), total_margin);
	if (v0 == Vector3(0, 0, 0)) {
		// This is a pathological case: the two objects are directly on top of
		// each other with their centers at exactly the same place. Just
		// return *some* vaguely plausible contact.

		return Vector3(0, 1, 0);
	}

	// Select three points that define the initial portal.

	Vector3 dir1 = -v0.normalized();
	Vector3 v1 = _compute_support<withMargin>(obj1, obj2, p_transform_a, p_transform_b, dir1, total_margin);
	if (v1.dot(dir1) <= 0.0) {
		return Vector3(0, 0, 0);
	}
	if (v1.cross(v0) == Vector3(0, 0, 0)) {
		return dir1;
	}
	Vector3 dir2 = v1.cross(v0).normalized();
	Vector3 v2 = _compute_support<withMargin>(obj1, obj2, p_transform_a, p_transform_b, dir2, total_margin);
	if (v2.dot(dir2) <= 0.0) {
		return Vector3(0, 0, 0);
	}
	Vector3 dir3 = (v1 - v0).cross(v2 - v0).normalized();
	if (dir3.dot(v0) > 0) {
		Vector3 swap1 = dir1;
		Vector3 swap2 = v1;
		dir1 = dir2;
		v1 = v2;
		dir2 = swap1;
		v2 = swap2;
		dir3 = -dir3;
	}
	Vector3 v3 = _compute_support<withMargin>(obj1, obj2, p_transform_a, p_transform_b, dir3, total_margin);
	if (v3.dot(dir3) <= 0.0) {
		return Vector3(0, 0, 0);
	}
	while (true) {
		if (v0.dot(v1.cross(v3)) < -1e-14) {
			dir2 = dir3;
			v2 = v3;
		} else if (v0.dot(v3.cross(v2)) < -1e-14) {
			dir1 = dir3;
			v1 = v3;
		} else
			break;
		dir3 = (v1 - v0).cross(v2 - v0).normalized();
		v3 = _compute_support<withMargin>(obj1, obj2, p_transform_a, p_transform_b, dir3, total_margin);
	}

	// We have a portal that the origin ray passes through. Now we need to
	// refine it.

	int extra_iterations = 0;
	while (true) {
		Vector3 portal_dir = (v2 - v1).cross(v3 - v1).normalized();
		if (portal_dir.dot(v0) > 0) {
			portal_dir = -portal_dir;
		}
		real_t dist1 = portal_dir.dot(v1);
		if (dist1 >= 0.0) {
			// The origin is inside the portal, so we have an intersection.
			// If the portal is sufficiently small to give a precise collision
			// axis, record the contact.  Otherwise, try iterating a little
			// longer to define it more precisely.

			if (extra_iterations == 10 || fmin(fmin(dir1.dot(dir2), dir2.dot(dir3)), dir3.dot(dir1)) > 0.9999) {
				return -portal_dir;
			}
			extra_iterations++;
		}

		Vector3 v4 = _compute_support<withMargin>(obj1, obj2, p_transform_a, p_transform_b, portal_dir, total_margin);
		real_t dist4 = portal_dir.dot(v4);
		if (dist4 <= CMP_EPSILON) {
			return Vector3(0, 0, 0);
		}

		Vector3 cross = v4.cross(v0);
		if (v1.dot(cross) > 0.0) {
			if (v2.dot(cross) > 0.0) {
				dir1 = portal_dir;
				v1 = v4;
			} else {
				dir3 = portal_dir;
				v3 = v4;
			}
		} else {
			if (v3.dot(cross) > 0.0) {
				dir2 = portal_dir;
				v2 = v4;
			} else {
				dir1 = portal_dir;
				v1 = v4;
			}
		}
	}
}

template <bool withMargin>
static Vector3 mpr_calculate_penetration(const GodotShape3D *p_shape_A, const Transform3D &p_transform_A, const GodotShape3D *p_shape_B, const Transform3D &p_transform_B, real_t p_margin_A, real_t p_margin_B) {
	return _collision_mpr<withMargin>(p_shape_A, p_transform_A, p_shape_B, p_transform_B, p_margin_A, p_margin_B);
}

#endif // MPR_H

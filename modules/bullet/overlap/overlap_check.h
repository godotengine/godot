/*************************************************************************/
/*  overlap_check.h                                                      */
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

#ifndef OVERLAP_CHECK_H
#define OVERLAP_CHECK_H

#include <BulletCollision/BroadphaseCollision/btBroadphaseProxy.h>
#include <LinearMath/btTransform.h>

class btCollisionShape;

typedef bool (*OverlappingFunc)(
		btCollisionShape *p_shape_1,
		const btTransform &p_shape_1_transform,
		btCollisionShape *p_shape_2,
		const btTransform &p_shape_2_transform);

/// Check if two shapes are overlapping each other. The algorithm used are a mix
/// of SAT and some accelerated one.
/// The accelerated checks are implemented for:
/// - Sphere <--> Sphere
/// - Sphere <--> Box
/// - Sphere <--> Capsule
/// - Capsule <--> Capsule
struct OverlapCheck {
	static OverlappingFunc overlapping_funcs[MAX_BROADPHASE_COLLISION_TYPES][MAX_BROADPHASE_COLLISION_TYPES];

	static void init();
	static OverlappingFunc find_algorithm(int body_1, int body_2);
};

#endif

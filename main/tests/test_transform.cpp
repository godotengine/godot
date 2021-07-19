/*************************************************************************/
/*  test_transform.cpp                                                   */
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

#include "test_transform.h"

#include "core/math/random_number_generator.h"
#include "core/math/transform.h"
#include "core/math/vector3.h"
#include "core/os/os.h"
#include "core/ustring.h"

namespace TestTransform {

bool test_plane() {
	bool pass = true;

	// test non-uniform scaling, forward and inverse
	Transform tr;
	tr.scale(Vector3(1, 2, 3));

	Plane p(Vector3(1, 1, 1), Vector3(1, 1, 1).normalized());

	Plane p2 = tr.xform(p);
	Plane p3 = tr.xform_inv(p2);

	if (!p3.normal.is_equal_approx(p.normal)) {
		OS::get_singleton()->print("Fail due to Transform::xform(Plane)\n");
		pass = false;
	}

	return pass;
}

bool test_aabb() {
	bool pass = true;

	Transform tr;
	tr.scale(Vector3(1, 2, 3));

	AABB bb(Vector3(1, 1, 1), Vector3(2, 3, 4));

	AABB bb2 = tr.xform(bb);
	bb2 = tr.xform_inv(bb2);

	if (!bb2.position.is_equal_approx(bb.position)) {
		OS::get_singleton()->print("Fail due to Transform::xform(AABB) position\n");
		pass = false;
	}

	if (!bb2.size.is_equal_approx(bb.size)) {
		OS::get_singleton()->print("Fail due to Transform::xform(AABB) size\n");
		pass = false;
	}

	return pass;
}

bool test_vector3() {
	bool pass = true;

	Transform tr;
	tr.scale(Vector3(1, 2, 3));

	Vector3 pt(1, 1, 1);
	Vector3 res = tr.xform(pt);

	if (!res.is_equal_approx(Vector3(1, 2, 3))) {
		OS::get_singleton()->print("Fail due to Transform::xform(Vector3)\n");
		pass = false;
	}

	pt = Vector3(1, 2, 3);
	res = tr.xform_inv(pt);
	if (!res.is_equal_approx(Vector3(1, 1, 1))) {
		OS::get_singleton()->print("Fail due to Transform::xform_inv(Vector3)\n");
		pass = false;
	}

	return pass;
}

MainLoop *test() {
	OS::get_singleton()->print("Start Transform checks.\n");

	bool success = true;

	if (!test_vector3()) {
		success = false;
	}

	if (!test_plane()) {
		success = false;
	}

	if (!test_aabb()) {
		success = false;
	}

	if (success) {
		OS::get_singleton()->print("Transform checks passed.\n");
	} else {
		OS::get_singleton()->print("Transform checks FAILED.\n");
	}

	return nullptr;
}

} // namespace TestTransform

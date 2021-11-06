/*************************************************************************/
/*  test_path_follow_3d.h                                                */
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

#ifndef TEST_PATH_FOLLOW_3D_H
#define TEST_PATH_FOLLOW_3D_H

#include "scene/3d/path_3d.h"

#include "tests/test_macros.h"

namespace TestPathFollow3D {

TEST_CASE("[PathFollow3D] Sampling with unit offset") {
	const Ref<Curve3D> &curve = memnew(Curve3D());
	curve->add_point(Vector3(0, 0, 0));
	curve->add_point(Vector3(100, 0, 0));
	curve->add_point(Vector3(100, 100, 0));
	curve->add_point(Vector3(100, 100, 100));
	curve->add_point(Vector3(100, 0, 100));
	const Path3D *path = memnew(Path3D);
	path->set_curve(curve);
	const PathFollow3D *path_follow_3d = memnew(PathFollow3D);
	path->add_child(path_follow_3d);

	path_follow_3d->set_unit_offset(0);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(0, 0, 0));

	path_follow_3d->set_unit_offset(0.125);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(50, 0, 0));

	path_follow_3d->set_unit_offset(0.25);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 0, 0);

	path_follow_3d->set_unit_offset(0.375);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 50, 0)));

	path_follow_3d->set_unit_offset(0.5);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 100, 0)));

	path_follow_3d->set_unit_offset(0.625);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 100, 50)));

	path_follow_3d->set_unit_offset(0.75);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 100, 100)));

	path_follow_3d->set_unit_offset(0.875);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 50, 100)));

	path_follow_3d->set_unit_offset(1);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 0, 100)));

	memdelete(path);
}

TEST_CASE("[PathFollow3D] Sampling with offset") {
	const Ref<Curve3D> &curve = memnew(Curve3D());
	curve->add_point(Vector3(0, 0, 0));
	curve->add_point(Vector3(100, 0, 0));
	curve->add_point(Vector3(100, 100, 0));
	curve->add_point(Vector3(100, 100, 100));
	curve->add_point(Vector3(100, 0, 100));
	const Path3D *path = memnew(Path3D);
	path->set_curve(curve);
	const PathFollow3D *path_follow_3d = memnew(PathFollow3D);
	path->add_child(path_follow_3d);

	path_follow_3d->set_offset(0);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(0, 0, 0));

	path_follow_3d->set_offset(50);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(50, 0, 0));

	path_follow_3d->set_offset(100);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 0, 0);

	path_follow_3d->set_offset(150);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 50, 0)));

	path_follow_3d->set_offset(200);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 100, 0)));

	path_follow_3d->set_offset(250);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 100, 50)));

	path_follow_3d->set_offset(300);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 100, 100)));

	path_follow_3d->set_offset(350);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 50, 100)));

	path_follow_3d->set_offset(400);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector3(100, 0, 100)));

	memdelete(path);
}

TEST_CASE("[PathFollow3D] Removal of a point in curve") {
	const Ref<Curve3D> &curve = memnew(Curve3D());
	curve->add_point(Vector3(0, 0, 0));
	curve->add_point(Vector3(100, 0, 0));
	curve->add_point(Vector3(100, 100, 0));
	const Path3D *path = memnew(Path3D);
	path->set_curve(curve);
	const PathFollow3D *path_follow_3d = memnew(PathFollow3D);
	path->add_child(path_follow_3d);

	path_follow_3d->set_unit_offset(0.5);
	CHECK(path_follow_3d->get_transform().get_origin().is_equal_approx(Vector2(100, 0, 0)));

	curve->remove_point(1);

	CHECK_MESSAGE(
			path_follow_3d->get_transform().get_origin().is_equal_approx(Vector2(50, 50, 0)),
			"Path follow's position should be updated after removing a point from the curve");

	memdelete(path);
}

TEST_CASE("[PathFollow3D] Unit offset out of range") {
	const Ref<Curve3D> &curve = memnew(Curve3D());
	curve->add_point(Vector3(0, 0, 0));
	curve->add_point(Vector3(100, 0, 0));
	const Path3D *path = memnew(Path3D);
	path->set_curve(curve);
	const PathFollow3D *path_follow_3d = memnew(PathFollow3D);
	path->add_child(path_follow_3d);

	path_follow_3d->set_loop(true);

	path_follow_3d->set_unit_offset(-0.3);
	CHECK_MESSAGE(
			path_follow_3d->get_unit_offset() == 0.7,
			"Unit Offset should loop back from the end in the opposite direction");

	path_follow_3d->set_unit_offset(1.3);
	CHECK_MESSAGE(
			path_follow_3d->get_unit_offset() == 0.3,
			"Unit Offset should loop back from the end in the opposite direction");

	path_follow_3d->set_loop(false);

	path_follow_3d->set_unit_offset(-0.3);
	CHECK_MESSAGE(
			path_follow_3d->get_unit_offset() == 0,
			"Unit Offset should be clamped at 0");

	path_follow_3d->set_unit_offset(1.3);
	CHECK_MESSAGE(
			path_follow_3d->get_unit_offset() == 1,
			"Unit Offset should be clamped at 1");

	memdelete(path);
}

TEST_CASE("[PathFollow3D] Offset out of range") {
	const Ref<Curve3D> &curve = memnew(Curve3D());
	curve->add_point(Vector3(0, 0, 0));
	curve->add_point(Vector3(100, 0, 0));
	const Path3D *path = memnew(Path3D);
	path->set_curve(curve);
	const PathFollow3D *path_follow_3d = memnew(PathFollow3D);
	path->add_child(path_follow_3d);

	path_follow_3d->set_loop(true);

	path_follow_3d->set_offset(-50);
	CHECK_MESSAGE(
			path_follow_3d->get_offset() == 50,
			"Offset should loop back from the end in the opposite direction");

	path_follow_3d->set_offset(150);
	CHECK_MESSAGE(
			path_follow_3d->get_offset() == 50,
			"Offset should loop back from the end in the opposite direction");

	path_follow_3d->set_loop(false);

	path_follow_3d->set_offset(-50);
	CHECK_MESSAGE(
			path_follow_3d->get_offset() == 0,
			"Offset should be clamped at 0");

	path_follow_3d->set_offset(150);
	CHECK_MESSAGE(
			path_follow_3d->get_offset() == 100,
			"Offset should be clamped at max value of curve");

	memdelete(path);
}
} // namespace TestPathFollow3D

#endif // TEST_PATH_FOLLOW_3D_H

/**************************************************************************/
/*  test_shape_cast_2d.cpp                                                */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_shape_cast_2d)

#include "scene/2d/physics/shape_cast_2d.h"
#include "scene/resources/2d/convex_polygon_shape_2d.h"
#include "scene/resources/2d/rectangle_shape_2d.h"

namespace TestShapeCast2D {

TEST_CASE("[SceneTree][ShapeCast2D] _get_draw_steps returns 2 for zero-size shape") {
	// Regression: empty ConvexPolygonShape2D and ConcavePolygonShape2D have a zero-size bounding rect.
	// The step count must fall back to 2 rather than dividing by zero.
	CHECK_EQ(ShapeCast2D::_get_draw_steps(50.0, 0.0), 2);
}

TEST_CASE("[SceneTree][ShapeCast2D] _get_draw_steps returns 2 for zero target length") {
	CHECK_EQ(ShapeCast2D::_get_draw_steps(0.0, 10.0), 2);
}

TEST_CASE("[SceneTree][ShapeCast2D] _get_draw_steps returns at least 2 for normal inputs") {
	CHECK_GE(ShapeCast2D::_get_draw_steps(50.0, 10.0), 2);
	CHECK_GE(ShapeCast2D::_get_draw_steps(1.0, 100.0), 2);
}

TEST_CASE("[SceneTree][ShapeCast2D] _get_draw_steps scales with target length") {
	int short_cast = ShapeCast2D::_get_draw_steps(50.0, 10.0);
	int long_cast = ShapeCast2D::_get_draw_steps(500.0, 10.0);
	CHECK_LT(short_cast, long_cast);
}

TEST_CASE("[SceneTree][ShapeCast2D] Default shape is null") {
	ShapeCast2D *cast = memnew(ShapeCast2D);

	CHECK(cast->get_shape().is_null());

	memdelete(cast);
}

TEST_CASE("[SceneTree][ShapeCast2D] Default target position is non-zero") {
	ShapeCast2D *cast = memnew(ShapeCast2D);

	// Default target_position is (0, 50); its length must be > 0 for the
	// division in the draw path to matter at all.
	CHECK_GT(cast->get_target_position().length(), 0.0f);

	memdelete(cast);
}

TEST_CASE("[SceneTree][ShapeCast2D] Assigning a null shape clears the shape") {
	ShapeCast2D *cast = memnew(ShapeCast2D);
	Ref<RectangleShape2D> rect_shape;
	rect_shape.instantiate();

	cast->set_shape(rect_shape);
	CHECK(cast->get_shape() == rect_shape);

	cast->set_shape(Ref<Shape2D>());
	CHECK(cast->get_shape().is_null());

	memdelete(cast);
}

TEST_CASE("[SceneTree][ShapeCast2D] Assigning an empty ConvexPolygonShape2D does not crash") {
	ShapeCast2D *cast = memnew(ShapeCast2D);
	Ref<ConvexPolygonShape2D> shape;
	shape.instantiate();

	cast->set_shape(shape);
	CHECK(cast->get_shape() == shape);

	memdelete(cast);
}

TEST_CASE("[SceneTree][ShapeCast2D] Assigning a ConvexPolygonShape2D with points does not crash") {
	ShapeCast2D *cast = memnew(ShapeCast2D);
	Ref<ConvexPolygonShape2D> shape;
	shape.instantiate();

	Vector<Vector2> pts;
	pts.push_back(Vector2(-10, -10));
	pts.push_back(Vector2(10, -10));
	pts.push_back(Vector2(0, 10));
	shape->set_points(pts);

	cast->set_shape(shape);
	CHECK(cast->get_shape() == shape);

	memdelete(cast);
}

TEST_CASE("[SceneTree][ShapeCast2D] Configuration warning is emitted when shape is null") {
	ShapeCast2D *cast = memnew(ShapeCast2D);

	PackedStringArray warnings = cast->get_configuration_warnings();
	CHECK_GT(warnings.size(), 0);

	memdelete(cast);
}

TEST_CASE("[SceneTree][ShapeCast2D] No configuration warning when shape is assigned") {
	ShapeCast2D *cast = memnew(ShapeCast2D);
	Ref<RectangleShape2D> rect_shape;
	rect_shape.instantiate();
	cast->set_shape(rect_shape);

	PackedStringArray warnings = cast->get_configuration_warnings();
	CHECK_EQ(warnings.size(), 0);

	memdelete(cast);
}

} // namespace TestShapeCast2D

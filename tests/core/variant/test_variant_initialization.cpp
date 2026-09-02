/**************************************************************************/
/*  test_variant_initialization.cpp                                       */
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

TEST_FORCE_LINK(test_variant_initialization)

#include "core/variant/variant_internal.h"

namespace TestVariantInitialization {

TEST_CASE("[VariantInitialization] change_and_reset") {
	// Garbage data
	Variant *p = new Variant();
	std::memset((void *)p, -1, sizeof(Variant));
	VariantInternal::set_type(*p, Variant::Type::NIL);

	SUBCASE("bool") {
		VariantTypeChanger<bool>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::BOOL);
		CHECK(*p == Variant(false));

		*p = Variant(true);
		VariantTypeChanger<bool>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::BOOL);
		CHECK(*p == Variant(false));

		*p = Variant(1);
		VariantTypeChanger<bool>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::BOOL);
		CHECK(*p == Variant(false));
	}

	SUBCASE("int") {
		VariantTypeChanger<int64_t>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::INT);
		CHECK(*p == Variant(0));

		*p = Variant(1);
		VariantTypeChanger<int64_t>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::INT);
		CHECK(*p == Variant(0));

		*p = Variant(true);
		VariantTypeChanger<int64_t>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::INT);
		CHECK(*p == Variant(0));
	}

	SUBCASE("float") {
		VariantTypeChanger<double>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::FLOAT);
		CHECK(*p == Variant(0.));

		*p = Variant(1.0);
		VariantTypeChanger<double>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::FLOAT);
		CHECK(*p == Variant(0.));

		*p = Variant(true);
		VariantTypeChanger<double>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::FLOAT);
		CHECK(*p == Variant(0.));
	}

	SUBCASE("String") {
		VariantTypeChanger<String>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::STRING);
		CHECK(*p == Variant(""));

		*p = Variant(1.0);
		VariantTypeChanger<String>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::STRING);
		CHECK(*p == Variant(""));

		*p = Variant(true);
		VariantTypeChanger<String>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::STRING);
		CHECK(*p == Variant(""));
	}

	SUBCASE("Vector2") {
		VariantTypeChanger<Vector2>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR2);
		CHECK(*p == Variant(Vector2()));

		*p = Variant(Vector2(1., 1.));
		VariantTypeChanger<Vector2>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR2);
		CHECK(*p == Variant(Vector2()));

		*p = Variant(true);
		VariantTypeChanger<Vector2>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR2);
		CHECK(*p == Variant(Vector2()));
	}

	SUBCASE("Vector2i") {
		VariantTypeChanger<Vector2i>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR2I);
		CHECK(*p == Variant(Vector2i()));

		*p = Variant(Vector2i(1, 1));
		VariantTypeChanger<Vector2i>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR2I);
		CHECK(*p == Variant(Vector2i()));

		*p = Variant(true);
		VariantTypeChanger<Vector2i>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR2I);
		CHECK(*p == Variant(Vector2i()));
	}

	SUBCASE("Vector3") {
		VariantTypeChanger<Vector3>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR3);
		CHECK(*p == Variant(Vector3()));

		*p = Variant(Vector3(1., 1., 1.));
		VariantTypeChanger<Vector3>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR3);
		CHECK(*p == Variant(Vector3()));

		*p = Variant(true);
		VariantTypeChanger<Vector3>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR3);
		CHECK(*p == Variant(Vector3()));
	}

	SUBCASE("Vector3i") {
		VariantTypeChanger<Vector3i>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR3I);
		CHECK(*p == Variant(Vector3i()));

		*p = Variant(Vector3i(1, 1, 1));
		VariantTypeChanger<Vector3i>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR3I);
		CHECK(*p == Variant(Vector3i()));

		*p = Variant(true);
		VariantTypeChanger<Vector3i>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR3I);
		CHECK(*p == Variant(Vector3i()));
	}

	SUBCASE("Vector4") {
		VariantTypeChanger<Vector4>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR4);
		CHECK(*p == Variant(Vector4()));

		*p = Variant(Vector4(1., 1., 1., 1.));
		VariantTypeChanger<Vector4>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR4);
		CHECK(*p == Variant(Vector4()));

		*p = Variant(true);
		VariantTypeChanger<Vector4>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR4);
		CHECK(*p == Variant(Vector4()));
	}

	SUBCASE("Vector4i") {
		VariantTypeChanger<Vector4i>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR4I);
		CHECK(*p == Variant(Vector4i()));

		*p = Variant(Vector4i(1, 1, 1, 1));
		VariantTypeChanger<Vector4i>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR4I);
		CHECK(*p == Variant(Vector4i()));

		*p = Variant(true);
		VariantTypeChanger<Vector4i>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::VECTOR4I);
		CHECK(*p == Variant(Vector4i()));
	}

	SUBCASE("Rect2") {
		VariantTypeChanger<Rect2>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::RECT2);
		CHECK(*p == Variant(Rect2()));

		*p = Variant(Rect2(1., 1., 1., 1.));
		VariantTypeChanger<Rect2>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::RECT2);
		CHECK(*p == Variant(Rect2()));

		*p = Variant(true);
		VariantTypeChanger<Rect2>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::RECT2);
		CHECK(*p == Variant(Rect2()));
	}

	SUBCASE("Rect2i") {
		VariantTypeChanger<Rect2i>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::RECT2I);
		CHECK(*p == Variant(Rect2i()));

		*p = Variant(Rect2i(1, 1, 1, 1));
		VariantTypeChanger<Rect2i>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::RECT2I);
		CHECK(*p == Variant(Rect2i()));

		*p = Variant(true);
		VariantTypeChanger<Rect2i>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::RECT2I);
		CHECK(*p == Variant(Rect2i()));
	}

	SUBCASE("Transform3D") {
		VariantTypeChanger<Transform3D>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::TRANSFORM3D);
		CHECK(*p == Variant(Transform3D()));

		Transform3D t;
		t.origin = Vector3(1, 1, 1);
		*p = Variant(t);
		VariantTypeChanger<Transform3D>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::TRANSFORM3D);
		CHECK(*p == Variant(Transform3D()));

		*p = Variant(true);
		VariantTypeChanger<Transform3D>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::TRANSFORM3D);
		CHECK(*p == Variant(Transform3D()));
	}

	SUBCASE("Projection") {
		VariantTypeChanger<Projection>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PROJECTION);
		CHECK(*p == Variant(Projection()));

		Projection pj;
		pj.columns[0].x = 1;
		*p = Variant(pj);
		VariantTypeChanger<Projection>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PROJECTION);
		CHECK(*p == Variant(Projection()));

		*p = Variant(true);
		VariantTypeChanger<Projection>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PROJECTION);
		CHECK(*p == Variant(Projection()));
	}

	SUBCASE("Transform2D") {
		VariantTypeChanger<Transform2D>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::TRANSFORM2D);
		CHECK(*p == Variant(Transform2D()));

		Transform2D t;
		t *= 2;
		*p = Variant(t);
		VariantTypeChanger<Transform2D>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::TRANSFORM2D);
		CHECK(*p == Variant(Transform2D()));

		*p = Variant(true);
		VariantTypeChanger<Transform2D>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::TRANSFORM2D);
		CHECK(*p == Variant(Transform2D()));
	}

	SUBCASE("Quaternion") {
		VariantTypeChanger<Quaternion>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::QUATERNION);
		CHECK(*p == Variant(Quaternion()));

		*p = Variant(Quaternion(1., 2., 3., 4.));
		VariantTypeChanger<Quaternion>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::QUATERNION);
		CHECK(*p == Variant(Quaternion()));

		*p = Variant(true);
		VariantTypeChanger<Quaternion>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::QUATERNION);
		CHECK(*p == Variant(Quaternion()));
	}

	SUBCASE("Plane") {
		VariantTypeChanger<Plane>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PLANE);
		CHECK(*p == Variant(Plane()));

		*p = Variant(Plane(Vector3(), 1.));
		VariantTypeChanger<Plane>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PLANE);
		CHECK(*p == Variant(Plane()));

		*p = Variant(true);
		VariantTypeChanger<Plane>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PLANE);
		CHECK(*p == Variant(Plane()));
	}

	SUBCASE("Basis") {
		VariantTypeChanger<Basis>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::BASIS);
		CHECK(*p == Variant(Basis()));

		*p = Variant(Basis::from_scale(Vector3::UP));
		VariantTypeChanger<Basis>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::BASIS);
		CHECK(*p == Variant(Basis()));

		*p = Variant(true);
		VariantTypeChanger<Basis>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::BASIS);
		CHECK(*p == Variant(Basis()));
	}

	SUBCASE("AABB") {
		VariantTypeChanger<AABB>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::AABB);
		CHECK(*p == Variant(AABB()));

		*p = Variant(AABB(Vector3::UP, Vector3::RIGHT));
		VariantTypeChanger<AABB>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::AABB);
		CHECK(*p == Variant(AABB()));

		*p = Variant(true);
		VariantTypeChanger<AABB>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::AABB);
		CHECK(*p == Variant(AABB()));
	}

	SUBCASE("Color") {
		VariantTypeChanger<Color>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::COLOR);
		CHECK(*p == Variant(Color()));

		*p = Variant(Color::from_hsv(0.5, 0.5, 0.5));
		VariantTypeChanger<Color>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::COLOR);
		CHECK(*p == Variant(Color()));

		*p = Variant(true);
		VariantTypeChanger<Color>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::COLOR);
		CHECK(*p == Variant(Color()));
	}

	SUBCASE("StringName") {
		VariantTypeChanger<StringName>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::STRING_NAME);
		CHECK(*p == Variant(StringName()));

		*p = Variant(StringName("abc"));
		VariantTypeChanger<StringName>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::STRING_NAME);
		CHECK(*p == Variant(StringName()));

		*p = Variant(true);
		VariantTypeChanger<StringName>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::STRING_NAME);
		CHECK(*p == Variant(StringName()));
	}

	SUBCASE("NodePath") {
		VariantTypeChanger<NodePath>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::NODE_PATH);
		CHECK(*p == Variant(NodePath()));

		*p = Variant(NodePath("abc"));
		VariantTypeChanger<NodePath>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::NODE_PATH);
		CHECK(*p == Variant(NodePath()));

		*p = Variant(true);
		VariantTypeChanger<NodePath>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::NODE_PATH);
		CHECK(*p == Variant(NodePath()));
	}

	SUBCASE("RID") {
		VariantTypeChanger<RID>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::RID);
		CHECK(*p == Variant(RID()));

		*p = Variant(RID::from_uint64(123ull));
		VariantTypeChanger<RID>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::RID);
		CHECK(*p == Variant(RID()));

		*p = Variant(true);
		VariantTypeChanger<RID>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::RID);
		CHECK(*p == Variant(RID()));
	}

	SUBCASE("Callable") {
		VariantTypeChanger<Callable>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::CALLABLE);
		CHECK(*p == Variant(Callable()));

		Object o;
		*p = Variant(Callable(&o, "abc"));
		VariantTypeChanger<Callable>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::CALLABLE);
		CHECK(*p == Variant(Callable()));

		*p = Variant(true);
		VariantTypeChanger<Callable>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::CALLABLE);
		CHECK(*p == Variant(Callable()));
	}

	SUBCASE("Signal") {
		VariantTypeChanger<Signal>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::SIGNAL);
		CHECK(*p == Variant(Signal()));

		Object o;
		*p = Variant(Signal(&o, "abc"));
		VariantTypeChanger<Signal>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::SIGNAL);
		CHECK(*p == Variant(Signal()));

		*p = Variant(true);
		VariantTypeChanger<Signal>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::SIGNAL);
		CHECK(*p == Variant(Signal()));
	}

	SUBCASE("Dictionary") {
		VariantTypeChanger<Dictionary>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::DICTIONARY);
		CHECK(*p == Variant(Dictionary()));

		*p = Variant(Dictionary{ KeyValue{ Variant(1), Variant(2) } });
		VariantTypeChanger<Dictionary>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::DICTIONARY);
		CHECK(*p == Variant(Dictionary()));

		*p = Variant(true);
		VariantTypeChanger<Dictionary>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::DICTIONARY);
		CHECK(*p == Variant(Dictionary()));
	}

	SUBCASE("Array") {
		VariantTypeChanger<Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::ARRAY);
		CHECK(*p == Variant(Array()));

		*p = Variant(Array{ Variant(1) });
		VariantTypeChanger<Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::ARRAY);
		CHECK(*p == Variant(Array()));

		*p = Variant(true);
		VariantTypeChanger<Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::ARRAY);
		CHECK(*p == Variant(Array()));
	}

	SUBCASE("PackedByteArray") {
		VariantTypeChanger<PackedByteArray>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_BYTE_ARRAY);
		CHECK(*p == Variant(PackedByteArray()));

		*p = Variant(PackedByteArray{ 1, 2 });
		VariantTypeChanger<PackedByteArray>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_BYTE_ARRAY);
		CHECK(*p == Variant(PackedByteArray()));

		*p = Variant(true);
		VariantTypeChanger<PackedByteArray>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_BYTE_ARRAY);
		CHECK(*p == Variant(PackedByteArray()));
	}

	SUBCASE("PackedInt32Array") {
		VariantTypeChanger<PackedInt32Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_INT32_ARRAY);
		CHECK(*p == Variant(PackedInt32Array()));

		*p = Variant(PackedInt32Array{ 1, 2 });
		VariantTypeChanger<PackedInt32Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_INT32_ARRAY);
		CHECK(*p == Variant(PackedInt32Array()));

		*p = Variant(true);
		VariantTypeChanger<PackedInt32Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_INT32_ARRAY);
		CHECK(*p == Variant(PackedInt32Array()));
	}

	SUBCASE("PackedInt64Array") {
		VariantTypeChanger<PackedInt64Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_INT64_ARRAY);
		CHECK(*p == Variant(PackedInt64Array()));

		*p = Variant(PackedInt64Array{ 1, 2 });
		VariantTypeChanger<PackedInt64Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_INT64_ARRAY);
		CHECK(*p == Variant(PackedInt64Array()));

		*p = Variant(true);
		VariantTypeChanger<PackedInt64Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_INT64_ARRAY);
		CHECK(*p == Variant(PackedInt64Array()));
	}

	SUBCASE("PackedFloat32Array") {
		VariantTypeChanger<PackedFloat32Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_FLOAT32_ARRAY);
		CHECK(*p == Variant(PackedFloat32Array()));

		*p = Variant(PackedFloat32Array{ 1, 2 });
		VariantTypeChanger<PackedFloat32Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_FLOAT32_ARRAY);
		CHECK(*p == Variant(PackedFloat32Array()));

		*p = Variant(true);
		VariantTypeChanger<PackedFloat32Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_FLOAT32_ARRAY);
		CHECK(*p == Variant(PackedFloat32Array()));
	}

	SUBCASE("PackedFloat64Array") {
		VariantTypeChanger<PackedFloat64Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_FLOAT64_ARRAY);
		CHECK(*p == Variant(PackedFloat64Array()));

		*p = Variant(PackedFloat64Array{ 1, 2 });
		VariantTypeChanger<PackedFloat64Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_FLOAT64_ARRAY);
		CHECK(*p == Variant(PackedFloat64Array()));

		*p = Variant(true);
		VariantTypeChanger<PackedFloat64Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_FLOAT64_ARRAY);
		CHECK(*p == Variant(PackedFloat64Array()));
	}

	SUBCASE("PackedStringArray") {
		VariantTypeChanger<PackedStringArray>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_STRING_ARRAY);
		CHECK(*p == Variant(PackedStringArray()));

		*p = Variant(PackedStringArray{ "1", "2" });
		VariantTypeChanger<PackedStringArray>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_STRING_ARRAY);
		CHECK(*p == Variant(PackedStringArray()));

		*p = Variant(true);
		VariantTypeChanger<PackedStringArray>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_STRING_ARRAY);
		CHECK(*p == Variant(PackedStringArray()));
	}

	SUBCASE("PackedVector2Array") {
		VariantTypeChanger<PackedVector2Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_VECTOR2_ARRAY);
		CHECK(*p == Variant(PackedVector2Array()));

		*p = Variant(PackedVector2Array{ Vector2::UP, Vector2::RIGHT });
		VariantTypeChanger<PackedVector2Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_VECTOR2_ARRAY);
		CHECK(*p == Variant(PackedVector2Array()));

		*p = Variant(true);
		VariantTypeChanger<PackedVector2Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_VECTOR2_ARRAY);
		CHECK(*p == Variant(PackedVector2Array()));
	}

	SUBCASE("PackedVector3Array") {
		VariantTypeChanger<PackedVector3Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_VECTOR3_ARRAY);
		CHECK(*p == Variant(PackedVector3Array()));

		*p = Variant(PackedVector3Array{ Vector3::UP, Vector3::RIGHT });
		VariantTypeChanger<PackedVector3Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_VECTOR3_ARRAY);
		CHECK(*p == Variant(PackedVector3Array()));

		*p = Variant(true);
		VariantTypeChanger<PackedVector3Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_VECTOR3_ARRAY);
		CHECK(*p == Variant(PackedVector3Array()));
	}

	SUBCASE("PackedColorArray") {
		VariantTypeChanger<PackedColorArray>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_COLOR_ARRAY);
		CHECK(*p == Variant(PackedColorArray()));

		*p = Variant(PackedColorArray{ Color::from_hsv(0., .1, .2) });
		VariantTypeChanger<PackedColorArray>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_COLOR_ARRAY);
		CHECK(*p == Variant(PackedColorArray()));

		*p = Variant(true);
		VariantTypeChanger<PackedColorArray>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_COLOR_ARRAY);
		CHECK(*p == Variant(PackedColorArray()));
	}

	SUBCASE("PackedVector4Array") {
		VariantTypeChanger<PackedVector4Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_VECTOR4_ARRAY);
		CHECK(*p == Variant(PackedVector4Array()));

		*p = Variant(PackedVector4Array{ Vector4(1, 2, 3, 4) });
		VariantTypeChanger<PackedVector4Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_VECTOR4_ARRAY);
		CHECK(*p == Variant(PackedVector4Array()));

		*p = Variant(true);
		VariantTypeChanger<PackedVector4Array>::change_and_reset(p);
		CHECK(p->get_type() == Variant::Type::PACKED_VECTOR4_ARRAY);
		CHECK(*p == Variant(PackedVector4Array()));
	}

	delete p;
}

} // namespace TestVariantInitialization

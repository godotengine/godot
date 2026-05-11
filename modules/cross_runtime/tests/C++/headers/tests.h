/**************************************************************************/
/*  tests.h                                                               */
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

#pragma once
#include "bridge_helpers.h"

#include "core/variant/array.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

#include <cstdio>

namespace VariantBridgeTests {

// Offsets – must match the .NET side exactly
static const int OFF_INT32 = 0;
static const int OFF_INT64 = 4;
static const int OFF_FLOAT = 12;
static const int OFF_DOUBLE = 16;
static const int OFF_BOOL = 24;
static const int OFF_STRING = 28;
static const int OFF_STRINGNAME = 48;
static const int OFF_NODEPATH = 60;
static const int OFF_RID = 76;
static const int OFF_VECTOR2 = 84;
static const int OFF_VECTOR2I = 92;
static const int OFF_RECT2 = 100;
static const int OFF_RECT2I = 116;
static const int OFF_VECTOR3 = 132;
static const int OFF_VECTOR3I = 144;
static const int OFF_VECTOR4 = 156;
static const int OFF_VECTOR4I = 172;
static const int OFF_PLANE = 188;
static const int OFF_QUATERNION = 204;
static const int OFF_AABB = 220;
static const int OFF_BASIS = 244;
static const int OFF_TRANSFORM3D = 280;
static const int OFF_PROJECTION = 328;
static const int OFF_COLOR = 392;
static const int OFF_PACKED_BYTE = 408;
static const int OFF_PACKED_INT32 = 415;
static const int OFF_PACKED_INT64 = 431;
static const int OFF_PACKED_FLOAT32 = 451;
static const int OFF_PACKED_FLOAT64 = 463;
static const int OFF_PACKED_STRING = 483;
static const int OFF_PACKED_VECTOR2 = 501;
static const int OFF_PACKED_VECTOR3 = 521;
static const int OFF_PACKED_COLOR = 537;
static const int OFF_DICTIONARY = 600;
static const int OFF_ARRAY = 720;

static bool run_all_tests(const volatile uint8_t *memory, String &error) {
	//  Scalars
	int val_int32 = read_int32(memory, OFF_INT32);
	if (val_int32 != 42) {
		error = "int32 mismatch";
		return false;
	}
	printf("[C++] read_int32 passed\n");

	int64_t val_int64 = read_int64(memory, OFF_INT64);
	if (val_int64 != 12345678901234LL) {
		error = "int64 mismatch";
		return false;
	}
	printf("[C++] read_int64 passed\n");

	float val_float = read_float(memory, OFF_FLOAT);
	if (val_float != 3.14f) {
		error = "float mismatch";
		return false;
	}
	printf("[C++] read_float passed\n");

	double val_double = read_double(memory, OFF_DOUBLE);
	if (val_double != 2.718281828) {
		error = "double mismatch";
		return false;
	}

	printf("[C++] read_double passed\n");

	bool val_bool = read_int32(memory, OFF_BOOL) != 0;
	if (!val_bool) {
		error = "bool mismatch";
		return false;
	}

	printf("[C++] bool passed\n");

	//  Strings
	String str = read_string_from_data(memory + OFF_STRING);
	if (str != "Hello, WASM!") {
		error = "string mismatch";
		return false;
	}

	printf("[C++] Read String passed\n");

	StringName sn = StringName(read_string_from_data(memory + OFF_STRINGNAME));
	if (sn != StringName("TestName")) {
		error = "StringName mismatch";
		return false;
	}

	printf("[C++] Read stringname passed\n");

	NodePath np = NodePath(read_string_from_data(memory + OFF_NODEPATH));
	if (np != NodePath("/root/Player")) {
		error = "NodePath mismatch";
		return false;
	}
	printf("[C++] Read Node Path passed\n");

	//  RID
	RID rid = read_rid(memory, OFF_RID);
	if (rid.get_id() != 0xDEADBEEFULL) {
		error = "RID mismatch";
		return false;
	}
	printf("[C++] Read RID passed\n");

	//  Vectors & Math
	Vector2 v2 = read_vector2(memory, OFF_VECTOR2);
	if (v2 != Vector2(1.5, 2.5)) {
		error = "Vector2 mismatch";
		return false;
	}
	printf("[C++] Read vector 2 passed\n");

	Vector2i v2i = read_vector2i(memory, OFF_VECTOR2I);
	if (v2i != Vector2i(10, -20)) {
		error = "Vector2i mismatch";
		return false;
	}
	printf("[C++] Read vector2i passed\n");

	Rect2 r2 = read_rect2(memory, OFF_RECT2);
	if (r2 != Rect2(Vector2(1, 2), Vector2(3, 4))) {
		error = "Rect2 mismatch";
		return false;
	}
	printf("[C++] Read Rect2 passed\n");

	Rect2i r2i = read_rect2i(memory, OFF_RECT2I);
	if (r2i != Rect2i(Vector2i(5, 6), Vector2i(7, 8))) {
		error = "Rect2i mismatch";
		return false;
	}
	printf("[C++] Read rect 2i passed\n");

	Vector3 v3 = read_vector3(memory, OFF_VECTOR3);
	if (v3 != Vector3(1, 2, 3)) {
		error = "Vector3 mismatch";
		return false;
	}
	printf("[C++] Read Vector3 passed\n");

	Vector3i v3i = read_vector3i(memory, OFF_VECTOR3I);
	if (v3i != Vector3i(4, 5, 6)) {
		error = "Vector3i mismatch";
		return false;
	}
	printf("[C++] Read Vector 3i passed\n");

	Vector4 v4 = read_vector4(memory, OFF_VECTOR4);
	if (v4 != Vector4(1, 2, 3, 4)) {
		error = "Vector4 mismatch";
		return false;
	}
	printf("[C++] Read vector 4 passed\n");

	Vector4i v4i = read_vector4i(memory, OFF_VECTOR4I);
	if (v4i != Vector4i(5, 6, 7, 8)) {
		error = "Vector4i mismatch";
		return false;
	}
	printf("[C++] Read vector 4i passed\n");

	Plane plane = read_plane(memory, OFF_PLANE);
	if (plane != Plane(Vector3(0, 1, 0), 2.5)) {
		error = "Plane mismatch";
		return false;
	}
	printf("[C++] Read plane passed\n");

	Quaternion q = read_quaternion(memory, OFF_QUATERNION);
	// read_quaternion returns w,x,y,z; Godot's Quaternion constructor expects x,y,z,w
	if (q != Quaternion(1, 0, 0, 0)) { // x=1, y=0, z=0, w=0
		error = "Quaternion mismatch";
		return false;
	}
	printf("[C++] Read Quaternion passed\n");

	AABB aabb = read_aabb(memory, OFF_AABB);
	if (aabb != AABB(Vector3(1, 2, 3), Vector3(4, 5, 6))) {
		error = "AABB mismatch";
		return false;
	}
	printf("[C++] Read AABB passed\n");

	Basis basis = read_basis(memory, OFF_BASIS);
	if (basis != Basis(Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1))) {
		error = "Basis mismatch";
		return false;
	}
	printf("[C++] Read Basis passed\n");

	Transform3D t3d = read_transform3d(memory, OFF_TRANSFORM3D);
	if (t3d != Transform3D(Basis(Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)), Vector3(7, 8, 9))) {
		error = "Transform3D mismatch";
		return false;
	}
	printf("[C++] Read Transform3D passed\n");

	Projection proj = read_projection(memory, OFF_PROJECTION);
	if (proj != Projection(Vector4(1, 0, 0, 0), Vector4(0, 1, 0, 0), Vector4(0, 0, 1, 0), Vector4(0, 0, 0, 1))) {
		error = "Projection mismatch";
		return false;
	}
	printf("[C++] Read Projection passed\n");

	Color color = read_color(memory, OFF_COLOR);
	if (color != Color(0.1, 0.2, 0.3, 0.4)) {
		error = "Color mismatch";
		return false;
	}
	printf("[C++] Read Color passed\n");

	//  Packed Arrays
	PackedByteArray pba = read_packed_byte_array(memory, OFF_PACKED_BYTE);
	if (pba.size() != 3 || pba[0] != 1 || pba[1] != 2 || pba[2] != 3) {
		error = "PackedByteArray mismatch";
		return false;
	}
	printf("[C++] Read PackedByteArray passed\n");

	PackedInt32Array p32 = read_packed_int32_array(memory, OFF_PACKED_INT32);
	if (p32.size() != 3 || p32[0] != 100 || p32[1] != 200 || p32[2] != 300) {
		error = "PackedInt32Array mismatch";
		return false;
	}
	printf("[C++] Read PackedInt32Array passed \n");

	PackedInt64Array p64 = read_packed_int64_array(memory, OFF_PACKED_INT64);
	if (p64.size() != 2 || p64[0] != 1000 || p64[1] != 2000) {
		error = "PackedInt64Array mismatch";
		return false;
	}
	printf("[C++] Read PackedInt64Array passed\n");

	PackedFloat32Array pf32 = read_packed_float32_array(memory, OFF_PACKED_FLOAT32);
	if (pf32.size() != 2 || pf32[0] != 1.1f || pf32[1] != 2.2f) {
		error = "PackedFloat32Array mismatch";
		return false;
	}
	printf("[C++] Read PackedFloat32Array passed\n");

	PackedFloat64Array pf64 = read_packed_float64_array(memory, OFF_PACKED_FLOAT64);
	if (pf64.size() != 2 || pf64[0] != 3.3 || pf64[1] != 4.4) {
		error = "PackedFloat64Array mismatch";
		return false;
	}
	printf("[C++] Read PackedFloat64Array passed \n");

	PackedStringArray ps = read_packed_string_array(memory, OFF_PACKED_STRING);
	if (ps.size() != 2 || ps[0] != "one" || ps[1] != "two") {
		error = "PackedStringArray mismatch";
		return false;
	}
	printf("[C++] Read PackedStringArray passed \n");

	PackedVector2Array pv2 = read_packed_vector2_array(memory, OFF_PACKED_VECTOR2);
	if (pv2.size() != 2 || pv2[0] != Vector2(1, 2) || pv2[1] != Vector2(3, 4)) {
		error = "PackedVector2Array mismatch";
		return false;
	}
	printf("[C++] Read PackedVector2Array passed\n");

	PackedVector3Array pv3 = read_packed_vector3_array(memory, OFF_PACKED_VECTOR3);
	if (pv3.size() != 1 || pv3[0] != Vector3(5, 6, 7)) {
		error = "PackedVector3Array mismatch";
		return false;
	}
	printf("[C++] Read PackedVector3Array passed \n");

	PackedColorArray pc = read_packed_color_array(memory, OFF_PACKED_COLOR);
	if (pc.size() != 2 || pc[0] != Color(1, 0, 0, 1) || pc[1] != Color(0, 1, 0, 1)) {
		error = "PackedColorArray mismatch";
		return false;
	}
	printf("[C++] Read PackedColorArray passed\n");

	//  Dictionary & Array (via length‑prefixed official variant)
	Dictionary dict = read_dictionary(memory, OFF_DICTIONARY);
	if (dict.size() != 3) {
		error = "Dictionary size mismatch";
		return false;
	}
	if (dict[String("id")] != Variant(7) ||
			dict[String("name")] != Variant("alpha") ||
			dict[String("items")] != Array{ 8, 9 }) {
		error = "Dictionary content mismatch";
		return false;
	}
	printf("[C++] Read Dictionary passed\n");

	Array arr = read_array(memory, OFF_ARRAY);
	if (arr.size() != 3 || arr[0] != Variant(1) ||
			arr[1] != Variant("two") ||
			arr[2] != Variant(Vector3(3, 4, 5))) {
		error = "Array content mismatch";
		return false;
	}
	printf("[C++] Read Array passed\n");

	return true;
}

} // namespace VariantBridgeTests

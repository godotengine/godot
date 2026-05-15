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

#ifdef WEB_ENABLED

#include "bridge_helpers.h"

#include "core/variant/array.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

#include <cstdio>

// The offsets are not part of the namespace
static volatile uint32_t *const CMD_OFFSET = reinterpret_cast<volatile uint32_t *>(0x5000u);
static volatile uint8_t *const STATUS_OFFSET = reinterpret_cast<volatile uint8_t *>(0x5004u);
static volatile uint8_t *const RESULT_OFFSET = reinterpret_cast<volatile uint8_t *>(0x5008u);

static const uint8_t *const CMD_DATA = reinterpret_cast<const uint8_t *>(0x5016u);

static const std::uint32_t CMD_NONE = 0;
static const std::uint32_t CMD_RUN_VARIANT_TESTS = 1;
static const std::uint32_t STATUS_PENDING = 0;
static const std::uint32_t STATUS_DONE = 1;

static const uint8_t *const OFF_INT32 = CMD_DATA + 0;
static const uint8_t *const OFF_INT64 = CMD_DATA + 4;
static const uint8_t *const OFF_FLOAT = CMD_DATA + 12;
static const uint8_t *const OFF_DOUBLE = CMD_DATA + 16;
static const uint8_t *const OFF_BOOL = CMD_DATA + 24;
static const uint8_t *const OFF_STRING = CMD_DATA + 28;
static const uint8_t *const OFF_STRINGNAME = CMD_DATA + 48;
static const uint8_t *const OFF_NODEPATH = CMD_DATA + 60;
static const uint8_t *const OFF_RID = CMD_DATA + 76;
static const uint8_t *const OFF_VECTOR2 = CMD_DATA + 84;
static const uint8_t *const OFF_VECTOR2I = CMD_DATA + 92;
static const uint8_t *const OFF_RECT2 = CMD_DATA + 100;
static const uint8_t *const OFF_RECT2I = CMD_DATA + 116;
static const uint8_t *const OFF_VECTOR3 = CMD_DATA + 132;
static const uint8_t *const OFF_VECTOR3I = CMD_DATA + 144;
static const uint8_t *const OFF_VECTOR4 = CMD_DATA + 156;
static const uint8_t *const OFF_VECTOR4I = CMD_DATA + 172;
static const uint8_t *const OFF_PLANE = CMD_DATA + 188;
static const uint8_t *const OFF_QUATERNION = CMD_DATA + 204;
static const uint8_t *const OFF_AABB = CMD_DATA + 220;
static const uint8_t *const OFF_BASIS = CMD_DATA + 244;
static const uint8_t *const OFF_TRANSFORM3D = CMD_DATA + 280;
static const uint8_t *const OFF_PROJECTION = CMD_DATA + 328;
static const uint8_t *const OFF_COLOR = CMD_DATA + 392;
static const uint8_t *const OFF_PACKED_BYTE = CMD_DATA + 408;
static const uint8_t *const OFF_PACKED_INT32 = CMD_DATA + 415;
static const uint8_t *const OFF_PACKED_INT64 = CMD_DATA + 431;
static const uint8_t *const OFF_PACKED_FLOAT32 = CMD_DATA + 451;
static const uint8_t *const OFF_PACKED_FLOAT64 = CMD_DATA + 463;
static const uint8_t *const OFF_PACKED_STRING = CMD_DATA + 483;
static const uint8_t *const OFF_PACKED_VECTOR2 = CMD_DATA + 501;
static const uint8_t *const OFF_PACKED_VECTOR3 = CMD_DATA + 521;
static const uint8_t *const OFF_PACKED_COLOR = CMD_DATA + 537;
static const uint8_t *const OFF_DICTIONARY = CMD_DATA + 600;
static const uint8_t *const OFF_ARRAY = CMD_DATA + 720;
static const uint8_t *const OFF_SIGNAL = CMD_DATA + 900;

namespace VariantBridgeTests {

// Original read‑only test (unchanged)
bool run_read_tests(String &error) {
	//  Scalars
	int val_int32 = read_int32(OFF_INT32);
	if (val_int32 != 42) {
		error = "int32 mismatch";
		return false;
	}
	printf("[C++] read_int32 passed\n");

	int64_t val_int64 = read_int64(OFF_INT64);
	if (val_int64 != 12345678901234LL) {
		error = "int64 mismatch";
		return false;
	}
	printf("[C++] read_int64 passed\n");

	float val_float = read_float(OFF_FLOAT);
	if (val_float != 3.14f) {
		error = "float mismatch";
		return false;
	}
	printf("[C++] read_float passed\n");

	double val_double = read_double(OFF_DOUBLE);
	if (val_double != 2.718281828) {
		error = "double mismatch";
		return false;
	}
	printf("[C++] read_double passed\n");

	bool val_bool = read_int32(OFF_BOOL) != 0;
	if (!val_bool) {
		error = "bool mismatch";
		return false;
	}
	printf("[C++] bool passed\n");

	//  Strings
	String str = read_string_from_data(OFF_STRING);
	if (str != "Hello, WASM!") {
		error = "string mismatch";
		return false;
	}
	printf("[C++] Read String passed\n");

	StringName sn = StringName(read_string_from_data(OFF_STRINGNAME));
	if (sn != StringName("TestName")) {
		error = "StringName mismatch";
		return false;
	}
	printf("[C++] Read stringname passed\n");

	NodePath np = NodePath(read_string_from_data(OFF_NODEPATH));
	if (np != NodePath("/root/Player")) {
		error = "NodePath mismatch";
		return false;
	}
	printf("[C++] Read Node Path passed\n");

	//  RID
	RID rid = read_rid(OFF_RID);
	if (rid.get_id() != 0xDEADBEEFULL) {
		error = "RID mismatch";
		return false;
	}
	printf("[C++] Read RID passed\n");

	//  Vectors & Math
	Vector2 v2 = read_vector2(OFF_VECTOR2);
	if (v2 != Vector2(1.5, 2.5)) {
		error = "Vector2 mismatch";
		return false;
	}
	printf("[C++] Read vector 2 passed\n");

	Vector2i v2i = read_vector2i(OFF_VECTOR2I);
	if (v2i != Vector2i(10, -20)) {
		error = "Vector2i mismatch";
		return false;
	}
	printf("[C++] Read vector2i passed\n");

	Rect2 r2 = read_rect2(OFF_RECT2);
	if (r2 != Rect2(Vector2(1, 2), Vector2(3, 4))) {
		error = "Rect2 mismatch";
		return false;
	}
	printf("[C++] Read Rect2 passed\n");

	Rect2i r2i = read_rect2i(OFF_RECT2I);
	if (r2i != Rect2i(Vector2i(5, 6), Vector2i(7, 8))) {
		error = "Rect2i mismatch";
		return false;
	}
	printf("[C++] Read rect 2i passed\n");

	Vector3 v3 = read_vector3(OFF_VECTOR3);
	if (v3 != Vector3(1, 2, 3)) {
		error = "Vector3 mismatch";
		return false;
	}
	printf("[C++] Read Vector3 passed\n");

	Vector3i v3i = read_vector3i(OFF_VECTOR3I);
	if (v3i != Vector3i(4, 5, 6)) {
		error = "Vector3i mismatch";
		return false;
	}
	printf("[C++] Read Vector 3i passed\n");

	Vector4 v4 = read_vector4(OFF_VECTOR4);
	if (v4 != Vector4(1, 2, 3, 4)) {
		error = "Vector4 mismatch";
		return false;
	}
	printf("[C++] Read vector 4 passed\n");

	Vector4i v4i = read_vector4i(OFF_VECTOR4I);
	if (v4i != Vector4i(5, 6, 7, 8)) {
		error = "Vector4i mismatch";
		return false;
	}
	printf("[C++] Read vector 4i passed\n");

	Plane plane = read_plane(OFF_PLANE);
	if (plane != Plane(Vector3(0, 1, 0), 2.5)) {
		error = "Plane mismatch";
		return false;
	}
	printf("[C++] Read plane passed\n");

	Quaternion q = read_quaternion(OFF_QUATERNION);
	if (q != Quaternion(1, 0, 0, 0)) {
		error = "Quaternion mismatch";
		return false;
	}
	printf("[C++] Read Quaternion passed\n");

	AABB aabb = read_aabb(OFF_AABB);
	if (aabb != AABB(Vector3(1, 2, 3), Vector3(4, 5, 6))) {
		error = "AABB mismatch";
		return false;
	}
	printf("[C++] Read AABB passed\n");

	Basis basis = read_basis(OFF_BASIS);
	if (basis != Basis(Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1))) {
		error = "Basis mismatch";
		return false;
	}
	printf("[C++] Read Basis passed\n");

	Transform3D t3d = read_transform3d(OFF_TRANSFORM3D);
	if (t3d != Transform3D(Basis(Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)), Vector3(7, 8, 9))) {
		error = "Transform3D mismatch";
		return false;
	}
	printf("[C++] Read Transform3D passed\n");

	Projection proj = read_projection(OFF_PROJECTION);
	if (proj != Projection(Vector4(1, 0, 0, 0), Vector4(0, 1, 0, 0), Vector4(0, 0, 1, 0), Vector4(0, 0, 0, 1))) {
		error = "Projection mismatch";
		return false;
	}
	printf("[C++] Read Projection passed\n");

	Color color = read_color(OFF_COLOR);
	if (color != Color(0.1, 0.2, 0.3, 0.4)) {
		error = "Color mismatch";
		return false;
	}
	printf("[C++] Read Color passed\n");

	//  Packed Arrays
	PackedByteArray pba = read_packed_byte_array(OFF_PACKED_BYTE);
	if (pba.size() != 3 || pba[0] != 1 || pba[1] != 2 || pba[2] != 3) {
		error = "PackedByteArray mismatch";
		return false;
	}
	printf("[C++] Read PackedByteArray passed\n");

	PackedInt32Array p32 = read_packed_int32_array(OFF_PACKED_INT32);
	if (p32.size() != 3 || p32[0] != 100 || p32[1] != 200 || p32[2] != 300) {
		error = "PackedInt32Array mismatch";
		return false;
	}
	printf("[C++] Read PackedInt32Array passed \n");

	PackedInt64Array p64 = read_packed_int64_array(OFF_PACKED_INT64);
	if (p64.size() != 2 || p64[0] != 1000 || p64[1] != 2000) {
		error = "PackedInt64Array mismatch";
		return false;
	}
	printf("[C++] Read PackedInt64Array passed\n");

	PackedFloat32Array pf32 = read_packed_float32_array(OFF_PACKED_FLOAT32);
	if (pf32.size() != 2 || pf32[0] != 1.1f || pf32[1] != 2.2f) {
		error = "PackedFloat32Array mismatch";
		return false;
	}
	printf("[C++] Read PackedFloat32Array passed\n");

	PackedFloat64Array pf64 = read_packed_float64_array(OFF_PACKED_FLOAT64);
	if (pf64.size() != 2 || pf64[0] != 3.3 || pf64[1] != 4.4) {
		error = "PackedFloat64Array mismatch";
		return false;
	}
	printf("[C++] Read PackedFloat64Array passed \n");

	PackedStringArray ps = read_packed_string_array(OFF_PACKED_STRING);
	if (ps.size() != 2 || ps[0] != "one" || ps[1] != "two") {
		error = "PackedStringArray mismatch";
		return false;
	}
	printf("[C++] Read PackedStringArray passed \n");

	PackedVector2Array pv2 = read_packed_vector2_array(OFF_PACKED_VECTOR2);
	if (pv2.size() != 2 || pv2[0] != Vector2(1, 2) || pv2[1] != Vector2(3, 4)) {
		error = "PackedVector2Array mismatch";
		return false;
	}
	printf("[C++] Read PackedVector2Array passed\n");

	PackedVector3Array pv3 = read_packed_vector3_array(OFF_PACKED_VECTOR3);
	if (pv3.size() != 1 || pv3[0] != Vector3(5, 6, 7)) {
		error = "PackedVector3Array mismatch";
		return false;
	}
	printf("[C++] Read PackedVector3Array passed \n");

	PackedColorArray pc = read_packed_color_array(OFF_PACKED_COLOR);
	if (pc.size() != 2 || pc[0] != Color(1, 0, 0, 1) || pc[1] != Color(0, 1, 0, 1)) {
		error = "PackedColorArray mismatch";
		return false;
	}
	printf("[C++] Read PackedColorArray passed\n");

	//  Dictionary & Array (via length‑prefixed official variant)
	Dictionary dict = read_dictionary(OFF_DICTIONARY);
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

	Array arr = read_array(OFF_ARRAY);
	if (arr.size() != 3 || arr[0] != Variant(1) ||
			arr[1] != Variant("two") ||
			arr[2] != Variant(Vector3(3, 4, 5))) {
		error = "Array content mismatch";
		return false;
	}
	printf("[C++] Read Array passed\n");

	Signal signal = read_signal(OFF_SIGNAL);
	if (signal.get_object_id() != ObjectID(67890ULL) ||
			signal.get_name() != StringName("test_signal")) {
		error = "Signal mismatch";
		return false;
	}
	printf("[C++] Read Signal passed\n");

	return true;
}

// They write new  values to the offsets which will be read by C#. 
// This runs after the read tests
void run_write_tests() {

	write_int32(const_cast<uint8_t*>(OFF_INT32), -99); 
	printf("[C++] write_int32 passed\n");

	write_int64(const_cast<uint8_t*>(OFF_INT64), -98765432109876LL);
	printf("[C++] write_int64 passed\n");

	write_float(const_cast<uint8_t*>(OFF_FLOAT), -2.71f);
	printf("[C++] write_float passed\n");

	write_double(const_cast<uint8_t*>(OFF_DOUBLE), -1.41421356237);
	printf("[C++] write_double passed\n");

	write_bool(const_cast<uint8_t*>(OFF_BOOL), false); 
	printf("[C++] bool write passed\n");

	write_string_to_data(const_cast<uint8_t*>(OFF_STRING), "WriteOK!");
	printf("[C++] String write passed\n");

	write_string_name(const_cast<uint8_t*>(OFF_STRINGNAME), StringName("WriteSN"));
	printf("[C++] StringName write passed\n");

	write_node_path(const_cast<uint8_t*>(OFF_NODEPATH), NodePath("/write/path"));
	printf("[C++] NodePath write passed\n");

	write_rid(const_cast<uint8_t*>(OFF_RID), RID::from_uint64(0xBEEFCAFEULL));
	printf("[C++] RID write passed\n");

	write_vector2(const_cast<uint8_t*>(OFF_VECTOR2), Vector2(10.5, -5.5));
	printf("[C++] Vector2 write passed\n");

	write_vector2i(const_cast<uint8_t*>(OFF_VECTOR2I), Vector2i(-100, 200));
	printf("[C++] Vector2i write passed\n");

	write_rect2(const_cast<uint8_t*>(OFF_RECT2), Rect2(Vector2(9, 8), Vector2(7, 6)));
	printf("[C++] Rect2 write passed\n");

	write_rect2i(const_cast<uint8_t*>(OFF_RECT2I), Rect2i(Vector2i(15, 16), Vector2i(17, 18)));
	printf("[C++] Rect2i write passed\n");

	write_vector3(const_cast<uint8_t*>(OFF_VECTOR3), Vector3(10, 20, 30));
	printf("[C++] Vector3 write passed\n");

	write_vector3i(const_cast<uint8_t*>(OFF_VECTOR3I), Vector3i(40, 50, 60));
	printf("[C++] Vector3i write passed\n");

	write_vector4(const_cast<uint8_t*>(OFF_VECTOR4), Vector4(11, 22, 33, 44));
	printf("[C++] Vector4 write passed\n");

	write_vector4i(const_cast<uint8_t*>(OFF_VECTOR4I), Vector4i(55, 66, 77, 88));
	printf("[C++] Vector4i write passed\n");

	write_plane(const_cast<uint8_t*>(OFF_PLANE), Plane(Vector3(1, 0, 0), 5.0));
	printf("[C++] Plane write passed\n");

	write_quaternion(const_cast<uint8_t*>(OFF_QUATERNION), Quaternion(0, 0.707, 0, 0.707));
	printf("[C++] Quaternion write passed\n");

	write_aabb(const_cast<uint8_t*>(OFF_AABB), AABB(Vector3(0, 0, 0), Vector3(10, 10, 10)));
	printf("[C++] AABB write passed\n");

	write_basis(const_cast<uint8_t*>(OFF_BASIS), Basis(Vector3(0, 0, -1), Vector3(0, 1, 0), Vector3(1, 0, 0))); 
	printf("[C++] Basis write passed\n");

	write_transform3d(const_cast<uint8_t*>(OFF_TRANSFORM3D),Transform3D(Basis(Vector3(-1, 0, 0), Vector3(0, -1, 0), Vector3(0, 0, -1)), Vector3(-7, -8, -9)));
	printf("[C++] Transform3D write passed\n");

	write_projection(const_cast<uint8_t*>(OFF_PROJECTION), Projection(Vector4(2, 0, 0, 0), Vector4(0, 2, 0, 0), Vector4(0, 0, 2, 0), Vector4(0, 0, 0, 2)));
	printf("[C++] Projection write passed\n");

	write_color(const_cast<uint8_t*>(OFF_COLOR), Color(0.9, 0.8, 0.7, 0.6));
	printf("[C++] Color write passed\n");

	
	PackedByteArray new_pba;
	new_pba.push_back(10);
	new_pba.push_back(20);
	new_pba.push_back(30);
	write_packed_byte_array(const_cast<uint8_t*>(OFF_PACKED_BYTE), new_pba);
	printf("[C++] PackedByteArray write passed\n");

	
	PackedInt32Array new_p32;
	new_p32.push_back(400);
	new_p32.push_back(500);
	new_p32.push_back(600);
	write_packed_int32_array(const_cast<uint8_t*>(OFF_PACKED_INT32), new_p32);	
	printf("[C++] PackedInt32Array write passed\n");


	PackedInt64Array new_p64;
	new_p64.push_back(-1000);
	new_p64.push_back(-2000);
	write_packed_int64_array(const_cast<uint8_t*>(OFF_PACKED_INT64), new_p64);	
	printf("[C++] PackedInt64Array write passed\n");


	PackedFloat32Array new_pf32;
	new_pf32.push_back(-1.1f);
	new_pf32.push_back(-2.2f);
	write_packed_float32_array(const_cast<uint8_t*>(OFF_PACKED_FLOAT32), new_pf32);	
	printf("[C++] PackedFloat32Array write passed\n");


	PackedFloat64Array new_pf64;
	new_pf64.push_back(-3.3);
	new_pf64.push_back(-4.4);
	write_packed_float64_array(const_cast<uint8_t*>(OFF_PACKED_FLOAT64), new_pf64);	
	printf("[C++] PackedFloat64Array write passed\n");


	PackedStringArray new_ps;
	new_ps.push_back("abc");
	new_ps.push_back("xyz");
	write_packed_string_array(const_cast<uint8_t*>(OFF_PACKED_STRING), new_ps);	
	printf("[C++] PackedStringArray write passed\n");


	PackedVector2Array new_pv2;
	new_pv2.push_back(Vector2(10, 20));
	new_pv2.push_back(Vector2(30, 40));
	write_packed_vector2_array(const_cast<uint8_t*>(OFF_PACKED_VECTOR2), new_pv2);	
	printf("[C++] PackedVector2Array write passed\n");


	PackedVector3Array new_pv3;
	new_pv3.push_back(Vector3(50, 60, 70));
	write_packed_vector3_array(const_cast<uint8_t*>(OFF_PACKED_VECTOR3), new_pv3);	
	printf("[C++] PackedVector3Array write passed\n");


	PackedColorArray new_pc;
	new_pc.push_back(Color(0, 0, 1, 1));
	new_pc.push_back(Color(1, 1, 0, 1));
	write_packed_color_array(const_cast<uint8_t*>(OFF_PACKED_COLOR), new_pc);	
	printf("[C++] PackedColorArray write passed\n");

	Dictionary new_dict;
	new_dict[String("x")] = 10;
	new_dict[String("y")] = 20;
	new_dict[String("z")] = Vector3(1, 2, 3);
	write_dictionary(const_cast<uint8_t*>(OFF_DICTIONARY), new_dict);
	printf("[C++] Dictionary write passed\n");


	Array new_arr;
	new_arr.push_back(Vector3(30, 40, 50));
	new_arr.push_back("write");
	new_arr.push_back(42);
	write_array(const_cast<uint8_t*>(OFF_ARRAY), new_arr);	
	printf("[C++] Array write passed\n");

	Signal new_signal(ObjectID(99999ULL), StringName("write_signal"));
	write_signal(const_cast<uint8_t*>(OFF_SIGNAL), new_signal);
	printf("[C++] Signal write passed\n");

}

} // namespace VariantBridgeTests

#else

namespace VariantBridgeTests {
inline bool run_read_tests(String &error) {
	error = String();
	return false;
}
inline void run_write_tests() {
}
} //namespace VariantBridgeTests

#endif
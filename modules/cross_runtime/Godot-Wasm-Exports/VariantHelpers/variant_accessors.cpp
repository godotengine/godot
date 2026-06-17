/**************************************************************************/
/*  variant_accessors.cpp                                                 */
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

// variant_accessors.cpp
// Read-only accessors that extract typed data from a heap-allocated Variant*.
// Every accessor takes a const Variant* returned by a generated export and
// either returns a scalar directly or writes into a caller-supplied output
// buffer — no ownership is transferred for out-buffer paths.
//
// String and packed-array accessors allocate via memalloc and return a raw
// pointer; the JS side must call variant_free_packed() after copying the data.
//
// Packed array layout:
//   Scalar types (byte/int32/int64/float32/float64) — plain memcpy, element count
//   from the matching variant_packed_*_size() call.
//   Vector/Color types — interleaved floats (x,y), (x,y,z), (r,g,b,a), (x,y,z,w).
//   PackedStringArray — [int32 count][int32 offsets[count]][utf8 data...], all in
//   one allocation; offsets index into the data region from data_start.

#include "../header/need.h"

#include <emscripten.h>

#include <cstdlib>
#include <cstring>

extern "C" {

// Callable helpers to extract callable id and method from js.
EMSCRIPTEN_KEEPALIVE
uint64_t callable_get_target_id(uintptr_t ptr) {
	Callable *c = reinterpret_cast<Callable *>(ptr);
	if (!c || !c->is_valid()) {
		return 0;
	}
	return c->get_object_id();
}

EMSCRIPTEN_KEEPALIVE
const char *callable_get_method(uintptr_t ptr) {
	Callable *c = reinterpret_cast<Callable *>(ptr);
	if (!c || !c->is_valid()) {
		return nullptr;
	}
	String method = c->get_method();
	if (method.is_empty()) {
		return nullptr;
	}
	// Return heap-allocated UTF-8 copy; JS will free it
	return strdup(method.utf8().get_data());
}

EMSCRIPTEN_KEEPALIVE
int variant_get_type(const Variant *v) {
	return static_cast<int>(v->get_type());
}

EMSCRIPTEN_KEEPALIVE
void variant_destroy(Variant *v) {
	memdelete(v);
}

/* Free any buffer allocated with memalloc below (strings, packed arrays, etc.) */
EMSCRIPTEN_KEEPALIVE
void variant_free_packed(void *ptr) {
	memfree(ptr);
}

// type 1 — BOOL
EMSCRIPTEN_KEEPALIVE
bool variant_as_bool(const Variant *v) {
	return v->operator bool();
}

// type 2 — INT as uint64 (for object handles) - a uint to handle negative overflows whenever need be
EMSCRIPTEN_KEEPALIVE
uint64_t variant_as_uint64(const Variant *v) {
	return v->operator uint64_t();
}

EMSCRIPTEN_KEEPALIVE
int64_t variant_as_int64(const Variant *v) {
	return v->operator int64_t();
}

// type 3 — FLOAT
EMSCRIPTEN_KEEPALIVE
double variant_as_double(const Variant *v) {
	return v->operator double();
}

// type 4 — STRING
EMSCRIPTEN_KEEPALIVE
const char *variant_as_string(const Variant *v) {
	CharString cs = v->operator String().utf8();
	int len = cs.length() + 1;
	char *buf = (char *)memalloc(len);
	memcpy(buf, cs.get_data(), len);
	return buf;
}

// type 5 — VECTOR2  → out_xy[0]=x, out_xy[1]=y  (doubles)
EMSCRIPTEN_KEEPALIVE
void variant_as_vector2(const Variant *v, double *out_xy) {
	Vector2 p = v->operator Vector2();
	out_xy[0] = (double)p.x;
	out_xy[1] = (double)p.y;
}

// type 6 — VECTOR2I  → out_xy[0]=x, out_xy[1]=y  (int32 written as double)
EMSCRIPTEN_KEEPALIVE
void variant_as_vector2i(const Variant *v, double *out_xy) {
	Vector2i p = v->operator Vector2i();
	out_xy[0] = (double)p.x;
	out_xy[1] = (double)p.y;
}

// type 7 — RECT2  → out[0]=x, out[1]=y, out[2]=w, out[3]=h  (floats)
EMSCRIPTEN_KEEPALIVE
void variant_as_rect2(const Variant *v, float *out_xywh) {
	Rect2 r = v->operator Rect2();
	out_xywh[0] = r.position.x;
	out_xywh[1] = r.position.y;
	out_xywh[2] = r.size.x;
	out_xywh[3] = r.size.y;
}

// type 8 — RECT2I  → out[0]=x, out[1]=y, out[2]=w, out[3]=h  (int32 as float)
EMSCRIPTEN_KEEPALIVE
void variant_as_rect2i(const Variant *v, float *out_xywh) {
	Rect2i r = v->operator Rect2i();
	out_xywh[0] = (float)r.position.x;
	out_xywh[1] = (float)r.position.y;
	out_xywh[2] = (float)r.size.x;
	out_xywh[3] = (float)r.size.y;
}

// type 9 — VECTOR3  → out_xyz[0]=x, out_xyz[1]=y, out_xyz[2]=z  (doubles)
EMSCRIPTEN_KEEPALIVE
void variant_as_vector3(const Variant *v, double *out_xyz) {
	Vector3 p = v->operator Vector3();
	out_xyz[0] = (double)p.x;
	out_xyz[1] = (double)p.y;
	out_xyz[2] = (double)p.z;
}

// type 10 — VECTOR3I
EMSCRIPTEN_KEEPALIVE
void variant_as_vector3i(const Variant *v, double *out_xyz) {
	Vector3i p = v->operator Vector3i();
	out_xyz[0] = (double)p.x;
	out_xyz[1] = (double)p.y;
	out_xyz[2] = (double)p.z;
}

// type 11 — TRANSFORM2D  → 6 floats: [xx, xy, yx, yy, ox, oy]
EMSCRIPTEN_KEEPALIVE
void variant_as_transform2d(const Variant *v, float *out6) {
	Transform2D t = v->operator Transform2D();
	out6[0] = t.columns[0].x;
	out6[1] = t.columns[0].y;
	out6[2] = t.columns[1].x;
	out6[3] = t.columns[1].y;
	out6[4] = t.columns[2].x;
	out6[5] = t.columns[2].y;
}

// type 12 — VECTOR4  → out[0..3]  (doubles)
EMSCRIPTEN_KEEPALIVE
void variant_as_vector4(const Variant *v, double *out_xyzw) {
	Vector4 p = v->operator Vector4();
	out_xyzw[0] = (double)p.x;
	out_xyzw[1] = (double)p.y;
	out_xyzw[2] = (double)p.z;
	out_xyzw[3] = (double)p.w;
}

// type 13 — VECTOR4I
EMSCRIPTEN_KEEPALIVE
void variant_as_vector4i(const Variant *v, double *out_xyzw) {
	Vector4i p = v->operator Vector4i();
	out_xyzw[0] = (double)p.x;
	out_xyzw[1] = (double)p.y;
	out_xyzw[2] = (double)p.z;
	out_xyzw[3] = (double)p.w;
}

// type 14 — PLANE  → out[0]=nx, out[1]=ny, out[2]=nz, out[3]=d  (floats)
EMSCRIPTEN_KEEPALIVE
void variant_as_plane(const Variant *v, float *out4) {
	Plane p = v->operator Plane();
	out4[0] = p.normal.x;
	out4[1] = p.normal.y;
	out4[2] = p.normal.z;
	out4[3] = p.d;
}

// type 15 — QUATERNION  → out[0]=x, out[1]=y, out[2]=z, out[3]=w  (floats)
EMSCRIPTEN_KEEPALIVE
void variant_as_quaternion(const Variant *v, float *out4) {
	Quaternion q = v->operator Quaternion();
	out4[0] = q.x;
	out4[1] = q.y;
	out4[2] = q.z;
	out4[3] = q.w;
}

// type 16 — AABB  → out[0..2]=pos xyz, out[3..5]=size xyz  (floats)
EMSCRIPTEN_KEEPALIVE
void variant_as_aabb(const Variant *v, float *out6) {
	AABB a = v->operator AABB();
	out6[0] = a.position.x;
	out6[1] = a.position.y;
	out6[2] = a.position.z;
	out6[3] = a.size.x;
	out6[4] = a.size.y;
	out6[5] = a.size.z;
}

// type 17 — BASIS  → 9 floats row-major [row0x,row0y,row0z, row1x,...]
EMSCRIPTEN_KEEPALIVE
void variant_as_basis(const Variant *v, float *out9) {
	Basis b = v->operator Basis();
	out9[0] = b.rows[0].x;
	out9[1] = b.rows[0].y;
	out9[2] = b.rows[0].z;
	out9[3] = b.rows[1].x;
	out9[4] = b.rows[1].y;
	out9[5] = b.rows[1].z;
	out9[6] = b.rows[2].x;
	out9[7] = b.rows[2].y;
	out9[8] = b.rows[2].z;
}

// type 18 — TRANSFORM3D  → 12 floats: basis (9) + origin (3)
EMSCRIPTEN_KEEPALIVE
void variant_as_transform3d(const Variant *v, float *out12) {
	Transform3D t = v->operator Transform3D();
	out12[0] = t.basis.rows[0].x;
	out12[1] = t.basis.rows[0].y;
	out12[2] = t.basis.rows[0].z;
	out12[3] = t.basis.rows[1].x;
	out12[4] = t.basis.rows[1].y;
	out12[5] = t.basis.rows[1].z;
	out12[6] = t.basis.rows[2].x;
	out12[7] = t.basis.rows[2].y;
	out12[8] = t.basis.rows[2].z;
	out12[9] = t.origin.x;
	out12[10] = t.origin.y;
	out12[11] = t.origin.z;
}

// type 19 — PROJECTION  → 16 floats column-major
EMSCRIPTEN_KEEPALIVE
void variant_as_projection(const Variant *v, float *out16) {
	Projection p = v->operator Projection();
	for (int col = 0; col < 4; col++) {
		for (int row = 0; row < 4; row++) {
			out16[col * 4 + row] = p.columns[col][row];
		}
	}
}

// type 20 — COLOR  → out[0]=r, out[1]=g, out[2]=b, out[3]=a  (floats)
EMSCRIPTEN_KEEPALIVE
void variant_as_color(const Variant *v, float *out4) {
	Color c = v->operator Color();
	out4[0] = c.r;
	out4[1] = c.g;
	out4[2] = c.b;
	out4[3] = c.a;
}

// type 21 — STRING_NAME
EMSCRIPTEN_KEEPALIVE
const char *variant_as_string_name(const Variant *v) {
	CharString cs = v->operator StringName().operator String().utf8();
	int len = cs.length() + 1;
	char *buf = (char *)memalloc(len);
	memcpy(buf, cs.get_data(), len);
	return buf;
}

// type 22 — NODE_PATH
EMSCRIPTEN_KEEPALIVE
const char *variant_as_node_path(const Variant *v) {
	CharString cs = v->operator NodePath().operator String().utf8();
	int len = cs.length() + 1;
	char *buf = (char *)memalloc(len);
	memcpy(buf, cs.get_data(), len);
	return buf;
}

// type 23 — RID  → returns the internal id as uint64
EMSCRIPTEN_KEEPALIVE
uint64_t variant_as_rid(const Variant *v) {
	RID r = v->operator RID();
	return (uint64_t)r.get_id();
}

// type 24 — OBJECT  → object instance id as uint64
EMSCRIPTEN_KEEPALIVE
uint64_t variant_as_object(const Variant *v) {
	Object *obj = v->operator Object *();
	if (!obj) {
		return 0;
	}
	return (uint64_t)obj->get_instance_id();
}

EMSCRIPTEN_KEEPALIVE Variant *variant_as_callable(const Variant *v) {
	return memnew(Variant(*v));
}
// type 26 – Signal
EMSCRIPTEN_KEEPALIVE Variant *variant_as_signal(const Variant *v) {
	return memnew(Variant(*v));
}

EMSCRIPTEN_KEEPALIVE
uint64_t variant_callable_target_id(const Variant *v) {
	Callable c = v->operator Callable();
	return (uint64_t)c.get_object_id();
}

EMSCRIPTEN_KEEPALIVE
const char *variant_callable_method(const Variant *v) {
	Callable c = v->operator Callable();
	CharString cs = c.get_method().operator String().utf8();
	int len = cs.length() + 1;
	char *buf = (char *)memalloc(len);
	memcpy(buf, cs.get_data(), len);
	return buf;
}

EMSCRIPTEN_KEEPALIVE
int64_t variant_signal_target_id(const Variant *v) {
	Signal s = v->operator Signal();
	return (int64_t)(uint64_t)s.get_object_id();
}

EMSCRIPTEN_KEEPALIVE
const char *variant_signal_name(const Variant *v) {
	Signal s = v->operator Signal();
	CharString cs = s.get_name().operator String().utf8();
	int len = cs.length() + 1;
	char *buf = (char *)memalloc(len);
	memcpy(buf, cs.get_data(), len);
	return buf;
}
// type 27 – Dictionary
EMSCRIPTEN_KEEPALIVE Variant *variant_as_dictionary(const Variant *v) {
	return memnew(Variant(*v));
}

EMSCRIPTEN_KEEPALIVE
int variant_dictionary_size(const Variant *v) {
	Dictionary d = v->operator Dictionary();
	return (int)d.size();
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_dictionary_get_key(const Variant *v, int index) {
	Dictionary d = v->operator Dictionary();
	Array keys = d.keys();
	return memnew(Variant(keys[index]));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_dictionary_get_value(const Variant *v, int index) {
	Dictionary d = v->operator Dictionary();
	Array keys = d.keys();
	return memnew(Variant(d[keys[index]]));
}
// type 28 – Array
EMSCRIPTEN_KEEPALIVE Variant *variant_as_array(const Variant *v) {
	return memnew(Variant(*v));
}

// type 28 helpers — Array element access
EMSCRIPTEN_KEEPALIVE
int variant_array_size(const Variant *v) {
	Array arr = v->operator Array();
	return (int)arr.size();
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_array_get(const Variant *v, int index) {
	Array arr = v->operator Array();
	return memnew(Variant(arr[index]));
}

// type 29 — PackedByteArray
EMSCRIPTEN_KEEPALIVE
const uint8_t *variant_as_packed_byte_array(const Variant *v) {
	PackedByteArray arr = v->operator PackedByteArray();
	int64_t sz = arr.size();
	if (sz == 0) {
		return nullptr;
	}
	uint8_t *buf = (uint8_t *)memalloc(sz);
	memcpy(buf, arr.ptr(), sz);
	return buf;
}
EMSCRIPTEN_KEEPALIVE
int variant_packed_byte_size(const Variant *v) {
	return (int)(v->operator PackedByteArray()).size();
}

// type 30 — PackedInt32Array
EMSCRIPTEN_KEEPALIVE
const int32_t *variant_as_packed_int32_array(const Variant *v) {
	PackedInt32Array arr = v->operator PackedInt32Array();
	int64_t sz = arr.size();
	if (sz == 0) {
		return nullptr;
	}
	int32_t *buf = (int32_t *)memalloc(sz * sizeof(int32_t));
	memcpy(buf, arr.ptr(), sz * sizeof(int32_t));
	return buf;
}
EMSCRIPTEN_KEEPALIVE
int variant_packed_int32_size(const Variant *v) {
	return (int)(v->operator PackedInt32Array()).size();
}

// type 31 — PackedInt64Array
EMSCRIPTEN_KEEPALIVE
const int64_t *variant_as_packed_int64_array(const Variant *v) {
	PackedInt64Array arr = v->operator PackedInt64Array();
	int64_t sz = arr.size();
	if (sz == 0) {
		return nullptr;
	}
	int64_t *buf = (int64_t *)memalloc(sz * sizeof(int64_t));
	memcpy(buf, arr.ptr(), sz * sizeof(int64_t));
	return buf;
}
EMSCRIPTEN_KEEPALIVE
int variant_packed_int64_size(const Variant *v) {
	return (int)(v->operator PackedInt64Array()).size();
}

// type 32 — PackedFloat32Array
EMSCRIPTEN_KEEPALIVE
const float *variant_as_packed_float32_array(const Variant *v) {
	PackedFloat32Array arr = v->operator PackedFloat32Array();
	int64_t sz = arr.size();
	if (sz == 0) {
		return nullptr;
	}
	float *buf = (float *)memalloc(sz * sizeof(float));
	memcpy(buf, arr.ptr(), sz * sizeof(float));
	return buf;
}
EMSCRIPTEN_KEEPALIVE
int variant_packed_float32_size(const Variant *v) {
	return (int)(v->operator PackedFloat32Array()).size();
}

// type 33 — PackedFloat64Array
EMSCRIPTEN_KEEPALIVE
const double *variant_as_packed_float64_array(const Variant *v) {
	PackedFloat64Array arr = v->operator PackedFloat64Array();
	int64_t sz = arr.size();
	if (sz == 0) {
		return nullptr;
	}
	double *buf = (double *)memalloc(sz * sizeof(double));
	memcpy(buf, arr.ptr(), sz * sizeof(double));
	return buf;
}
EMSCRIPTEN_KEEPALIVE
int variant_packed_float64_size(const Variant *v) {
	return (int)(v->operator PackedFloat64Array()).size();
}

// type 34 — PackedStringArray
// Layout: [int32 count][int32 offsets[count]][utf8 data...]
EMSCRIPTEN_KEEPALIVE
const uint8_t *variant_as_packed_string_array(const Variant *v) {
	PackedStringArray arr = v->operator PackedStringArray();
	int64_t count = arr.size();
	if (count == 0) {
		return nullptr;
	}

	Vector<CharString> encoded;
	encoded.resize(count);
	int64_t data_bytes = 0;
	for (int64_t i = 0; i < count; i++) {
		encoded.write[i] = arr[i].utf8();
		data_bytes += encoded[i].length() + 1;
	}

	int64_t header_bytes = sizeof(int32_t) * (1 + count);
	int64_t total = header_bytes + data_bytes;
	uint8_t *buf = (uint8_t *)memalloc(total);

	int32_t *header = (int32_t *)buf;
	header[0] = (int32_t)count;
	uint8_t *data_start = buf + header_bytes;
	int32_t offset = 0;
	for (int64_t i = 0; i < count; i++) {
		header[1 + i] = offset;
		int32_t len = encoded[i].length() + 1;
		memcpy(data_start + offset, encoded[i].get_data(), len);
		offset += len;
	}
	return buf;
}
EMSCRIPTEN_KEEPALIVE
int variant_packed_string_count(const Variant *v) {
	return (int)(v->operator PackedStringArray()).size();
}

// type 35 — PackedVector2Array  → interleaved [x0,y0, x1,y1, ...]  (floats)
EMSCRIPTEN_KEEPALIVE
const float *variant_as_packed_vector2_array(const Variant *v) {
	PackedVector2Array arr = v->operator PackedVector2Array();
	int64_t sz = arr.size();
	if (sz == 0) {
		return nullptr;
	}
	float *buf = (float *)memalloc(sz * 2 * sizeof(float));
	for (int64_t i = 0; i < sz; i++) {
		buf[i * 2 + 0] = arr[i].x;
		buf[i * 2 + 1] = arr[i].y;
	}
	return buf;
}
EMSCRIPTEN_KEEPALIVE
int variant_packed_vector2_size(const Variant *v) {
	return (int)(v->operator PackedVector2Array()).size();
}

// type 36 — PackedVector3Array  → interleaved [x0,y0,z0, ...]  (floats)
EMSCRIPTEN_KEEPALIVE
const float *variant_as_packed_vector3_array(const Variant *v) {
	PackedVector3Array arr = v->operator PackedVector3Array();
	int64_t sz = arr.size();
	if (sz == 0) {
		return nullptr;
	}
	float *buf = (float *)memalloc(sz * 3 * sizeof(float));
	for (int64_t i = 0; i < sz; i++) {
		buf[i * 3 + 0] = arr[i].x;
		buf[i * 3 + 1] = arr[i].y;
		buf[i * 3 + 2] = arr[i].z;
	}
	return buf;
}
EMSCRIPTEN_KEEPALIVE
int variant_packed_vector3_size(const Variant *v) {
	return (int)(v->operator PackedVector3Array()).size();
}

// type 37 — PackedColorArray  → interleaved [r,g,b,a, ...]  (floats)
EMSCRIPTEN_KEEPALIVE
const float *variant_as_packed_color_array(const Variant *v) {
	PackedColorArray arr = v->operator PackedColorArray();
	int64_t sz = arr.size();
	if (sz == 0) {
		return nullptr;
	}
	float *buf = (float *)memalloc(sz * 4 * sizeof(float));
	for (int64_t i = 0; i < sz; i++) {
		buf[i * 4 + 0] = arr[i].r;
		buf[i * 4 + 1] = arr[i].g;
		buf[i * 4 + 2] = arr[i].b;
		buf[i * 4 + 3] = arr[i].a;
	}
	return buf;
}
EMSCRIPTEN_KEEPALIVE
int variant_packed_color_size(const Variant *v) {
	return (int)(v->operator PackedColorArray()).size();
}

// type 38 — PackedVector4Array  → interleaved [x,y,z,w, ...]  (floats)
EMSCRIPTEN_KEEPALIVE
const float *variant_as_packed_vector4_array(const Variant *v) {
	PackedVector4Array arr = v->operator PackedVector4Array();
	int64_t sz = arr.size();
	if (sz == 0) {
		return nullptr;
	}
	float *buf = (float *)memalloc(sz * 4 * sizeof(float));
	for (int64_t i = 0; i < sz; i++) {
		buf[i * 4 + 0] = arr[i].x;
		buf[i * 4 + 1] = arr[i].y;
		buf[i * 4 + 2] = arr[i].z;
		buf[i * 4 + 3] = arr[i].w;
	}
	return buf;
}
EMSCRIPTEN_KEEPALIVE
int variant_packed_vector4_size(const Variant *v) {
	return (int)(v->operator PackedVector4Array()).size();
}
}

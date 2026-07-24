/**************************************************************************/
/*  variant_constructor.cpp                                               */
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

// variant_constructors.cpp
// Constructors that build a heap-allocated Variant* from raw scalars.
// Every function returns a memnew'd Variant* that the caller must free
// via variant_destroy() (memdelete) after the C++ call consuming it returns.
//
// Struct types accept pre-flattened scalar fields matching the layout in
// FLATTENED_STRUCTS (js_generator.py / variant_construct.js).
// Object handles are recovered from ObjectDB by uint64 instance ID.
// Callable and Signal constructors take a raw pointer to an existing
// Callable/Signal in Godot memory — the caller owns that lifetime.

#include "../header/need.h"

#include <emscripten.h>

extern "C" {

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_nil() {
	return memnew(Variant());
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_bool(bool v) {
	return memnew(Variant(v));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_int64(int64_t v) {
	return memnew(Variant(v));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_double(double v) {
	return memnew(Variant(v));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_string(const char *utf8) {
	return memnew(Variant(String::utf8(utf8)));
}

// –– Value types ––
EMSCRIPTEN_KEEPALIVE
Variant *variant_new_vector2(double x, double y) {
	return memnew(Variant(Vector2((float)x, (float)y)));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_vector2i(int x, int y) {
	return memnew(Variant(Vector2i((int)x, (int)y)));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_vector3(double x, double y, double z) {
	return memnew(Variant(Vector3((float)x, (float)y, (float)z)));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_vector3i(int x, int y, int z) {
	return memnew(Variant(Vector3i((int)x, (int)y, (int)z)));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_vector4(double x, double y, double z, double w) {
	return memnew(Variant(Vector4((float)x, (float)y, (float)z, (float)w)));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_vector4i(int x, int y, int z, int w) {
	return memnew(Variant(Vector4i((int)x, (int)y, (int)z, (int)w)));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_rect2(float x, float y, float w, float h) {
	return memnew(Variant(Rect2(x, y, w, h)));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_rect2i(int x, int y, int w, int h) {
	return memnew(Variant(Rect2i(x, y, w, h)));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_color(float r, float g, float b, float a) {
	return memnew(Variant(Color(r, g, b, a)));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_plane(float nx, float ny, float nz, float d) {
	return memnew(Variant(Plane(Vector3(nx, ny, nz), d)));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_quaternion(float x, float y, float z, float w) {
	return memnew(Variant(Quaternion(x, y, z, w)));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_aabb(float px, float py, float pz, float sx, float sy, float sz) {
	return memnew(Variant(AABB(Vector3(px, py, pz), Vector3(sx, sy, sz))));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_basis(float xx, float xy, float xz, float yx, float yy, float yz, float zx, float zy, float zz) {
	return memnew(Variant(Basis(Vector3(xx, xy, xz), Vector3(yx, yy, yz), Vector3(zx, zy, zz))));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_transform2d(float xx, float xy, float yx, float yy, float ox, float oy) {
	return memnew(Variant(Transform2D(Vector2(xx, xy), Vector2(yx, yy), Vector2(ox, oy))));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_transform3d(float bx_x, float bx_y, float bx_z,
		float by_x, float by_y, float by_z,
		float bz_x, float bz_y, float bz_z,
		float origin_x, float origin_y, float origin_z) {
	Basis basis(Vector3(bx_x, bx_y, bx_z), Vector3(by_x, by_y, by_z), Vector3(bz_x, bz_y, bz_z));
	Transform3D t(basis, Vector3(origin_x, origin_y, origin_z));
	return memnew(Variant(t));
}

// –– Godot‑specific handles ––
EMSCRIPTEN_KEEPALIVE
Variant *variant_new_rid(uint64_t id) {
	return memnew(Variant(RID::from_uint64(id)));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_object_id(uint64_t id) {
	return memnew(Variant(ObjectDB::get_instance(ObjectID(id))));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_node_path(const char *utf8) {
	return memnew(Variant(NodePath(String::utf8(utf8))));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_string_name(const char *utf8) {
	return memnew(Variant(StringName(String::utf8(utf8))));
}

// –– Callable / Signal / Dictionary / Array ––
EMSCRIPTEN_KEEPALIVE
Variant *variant_new_callable(void *p_callable) {
	// p_callable is a pointer to a Callable object; caller must ensure lifetime
	return memnew(Variant(*(Callable *)p_callable));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_signal(void *p_signal) {
	return memnew(Variant(*(Signal *)p_signal));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_dictionary() {
	return memnew(Variant(Dictionary()));
}

EMSCRIPTEN_KEEPALIVE
Variant *variant_new_array() {
	return memnew(Variant(Array()));
}

} // extern "C"

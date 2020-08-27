/*************************************************************************/
/*  multimesh.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "multimesh.h"

#include "servers/rendering_server.h"

#ifndef DISABLE_DEPRECATED
// Kept for compatibility from 3.x to 4.0.

void MultiMesh::_set_transform_array(const Vector<Vector3> &p_array) {
	if (transform_format != TRANSFORM_3D) {
		return;
	}

	const Vector<Vector3> &xforms = p_array;
	int len = xforms.size();
	ERR_FAIL_COND((len / 4) != instance_count);
	if (len == 0) {
		return;
	}

	const Vector3 *r = xforms.ptr();

	for (int i = 0; i < len / 4; i++) {
		Transform t;
		t.basis[0] = r[i * 4 + 0];
		t.basis[1] = r[i * 4 + 1];
		t.basis[2] = r[i * 4 + 2];
		t.origin = r[i * 4 + 3];

		set_instance_transform(i, t);
	}
}

Vector<Vector3> MultiMesh::_get_transform_array() const {
	if (transform_format != TRANSFORM_3D) {
		return Vector<Vector3>();
	}

	if (instance_count == 0) {
		return Vector<Vector3>();
	}

	Vector<Vector3> xforms;
	xforms.resize(instance_count * 4);

	Vector3 *w = xforms.ptrw();

	for (int i = 0; i < instance_count; i++) {
		Transform t = get_instance_transform(i);
		w[i * 4 + 0] = t.basis[0];
		w[i * 4 + 1] = t.basis[1];
		w[i * 4 + 2] = t.basis[2];
		w[i * 4 + 3] = t.origin;
	}

	return xforms;
}

void MultiMesh::_set_transform_2d_array(const Vector<Vector2> &p_array) {
	if (transform_format != TRANSFORM_2D) {
		return;
	}

	const Vector<Vector2> &xforms = p_array;
	int len = xforms.size();
	ERR_FAIL_COND((len / 3) != instance_count);
	if (len == 0) {
		return;
	}

	const Vector2 *r = xforms.ptr();

	for (int i = 0; i < len / 3; i++) {
		Transform2D t;
		t.elements[0] = r[i * 3 + 0];
		t.elements[1] = r[i * 3 + 1];
		t.elements[2] = r[i * 3 + 2];

		set_instance_transform_2d(i, t);
	}
}

Vector<Vector2> MultiMesh::_get_transform_2d_array() const {
	if (transform_format != TRANSFORM_2D) {
		return Vector<Vector2>();
	}

	if (instance_count == 0) {
		return Vector<Vector2>();
	}

	Vector<Vector2> xforms;
	xforms.resize(instance_count * 3);

	Vector2 *w = xforms.ptrw();

	for (int i = 0; i < instance_count; i++) {
		Transform2D t = get_instance_transform_2d(i);
		w[i * 3 + 0] = t.elements[0];
		w[i * 3 + 1] = t.elements[1];
		w[i * 3 + 2] = t.elements[2];
	}

	return xforms;
}

void MultiMesh::_set_color_array(const Vector<Color> &p_array) {
	const Vector<Color> &colors = p_array;
	int len = colors.size();
	if (len == 0) {
		return;
	}
	ERR_FAIL_COND(len != instance_count);

	const Color *r = colors.ptr();

	for (int i = 0; i < len; i++) {
		set_instance_color(i, r[i]);
	}
}

Vector<Color> MultiMesh::_get_color_array() const {
	if (instance_count == 0 || !use_colors) {
		return Vector<Color>();
	}

	Vector<Color> colors;
	colors.resize(instance_count);

	for (int i = 0; i < instance_count; i++) {
		colors.set(i, get_instance_color(i));
	}

	return colors;
}

void MultiMesh::_set_custom_data_array(const Vector<Color> &p_array) {
	const Vector<Color> &custom_datas = p_array;
	int len = custom_datas.size();
	if (len == 0) {
		return;
	}
	ERR_FAIL_COND(len != instance_count);

	const Color *r = custom_datas.ptr();

	for (int i = 0; i < len; i++) {
		set_instance_custom_data(i, r[i]);
	}
}

Vector<Color> MultiMesh::_get_custom_data_array() const {
	if (instance_count == 0 || !use_custom_data) {
		return Vector<Color>();
	}

	Vector<Color> custom_datas;
	custom_datas.resize(instance_count);

	for (int i = 0; i < instance_count; i++) {
		custom_datas.set(i, get_instance_custom_data(i));
	}

	return custom_datas;
}
#endif // DISABLE_DEPRECATED

void MultiMesh::set_buffer(const Vector<float> &p_buffer) {
	RS::get_singleton()->multimesh_set_buffer(multimesh, p_buffer);
}

Vector<float> MultiMesh::get_buffer() const {
	return RS::get_singleton()->multimesh_get_buffer(multimesh);
}

void MultiMesh::set_mesh(const Ref<Mesh> &p_mesh) {
	mesh = p_mesh;
	if (!mesh.is_null()) {
		RenderingServer::get_singleton()->multimesh_set_mesh(multimesh, mesh->get_rid());
	} else {
		RenderingServer::get_singleton()->multimesh_set_mesh(multimesh, RID());
	}
}

Ref<Mesh> MultiMesh::get_mesh() const {
	return mesh;
}

void MultiMesh::set_instance_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);
	RenderingServer::get_singleton()->multimesh_allocate(multimesh, p_count, RS::MultimeshTransformFormat(transform_format), use_colors, use_custom_data);
	instance_count = p_count;
}

int MultiMesh::get_instance_count() const {
	return instance_count;
}

void MultiMesh::set_visible_instance_count(int p_count) {
	ERR_FAIL_COND(p_count < -1);
	ERR_FAIL_COND(p_count > instance_count);
	RenderingServer::get_singleton()->multimesh_set_visible_instances(multimesh, p_count);
	visible_instance_count = p_count;
}

int MultiMesh::get_visible_instance_count() const {
	return visible_instance_count;
}

void MultiMesh::set_instance_transform(int p_instance, const Transform &p_transform) {
	RenderingServer::get_singleton()->multimesh_instance_set_transform(multimesh, p_instance, p_transform);
}

void MultiMesh::set_instance_transform_2d(int p_instance, const Transform2D &p_transform) {
	RenderingServer::get_singleton()->multimesh_instance_set_transform_2d(multimesh, p_instance, p_transform);
}

Transform MultiMesh::get_instance_transform(int p_instance) const {
	return RenderingServer::get_singleton()->multimesh_instance_get_transform(multimesh, p_instance);
}

Transform2D MultiMesh::get_instance_transform_2d(int p_instance) const {
	return RenderingServer::get_singleton()->multimesh_instance_get_transform_2d(multimesh, p_instance);
}

void MultiMesh::set_instance_color(int p_instance, const Color &p_color) {
	RenderingServer::get_singleton()->multimesh_instance_set_color(multimesh, p_instance, p_color);
}

Color MultiMesh::get_instance_color(int p_instance) const {
	return RenderingServer::get_singleton()->multimesh_instance_get_color(multimesh, p_instance);
}

void MultiMesh::set_instance_custom_data(int p_instance, const Color &p_custom_data) {
	RenderingServer::get_singleton()->multimesh_instance_set_custom_data(multimesh, p_instance, p_custom_data);
}

Color MultiMesh::get_instance_custom_data(int p_instance) const {
	return RenderingServer::get_singleton()->multimesh_instance_get_custom_data(multimesh, p_instance);
}

AABB MultiMesh::get_aabb() const {
	return RenderingServer::get_singleton()->multimesh_get_aabb(multimesh);
}

RID MultiMesh::get_rid() const {
	return multimesh;
}

void MultiMesh::set_use_colors(bool p_enable) {
	ERR_FAIL_COND(instance_count > 0);
	use_colors = p_enable;
}

bool MultiMesh::is_using_colors() const {
	return use_colors;
}

void MultiMesh::set_use_custom_data(bool p_enable) {
	ERR_FAIL_COND(instance_count > 0);
	use_custom_data = p_enable;
}

bool MultiMesh::is_using_custom_data() const {
	return use_custom_data;
}

void MultiMesh::set_transform_format(TransformFormat p_transform_format) {
	ERR_FAIL_COND(instance_count > 0);
	transform_format = p_transform_format;
}

MultiMesh::TransformFormat MultiMesh::get_transform_format() const {
	return transform_format;
}

void MultiMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &MultiMesh::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &MultiMesh::get_mesh);
	ClassDB::bind_method(D_METHOD("set_use_colors", "enable"), &MultiMesh::set_use_colors);
	ClassDB::bind_method(D_METHOD("is_using_colors"), &MultiMesh::is_using_colors);
	ClassDB::bind_method(D_METHOD("set_use_custom_data", "enable"), &MultiMesh::set_use_custom_data);
	ClassDB::bind_method(D_METHOD("is_using_custom_data"), &MultiMesh::is_using_custom_data);
	ClassDB::bind_method(D_METHOD("set_transform_format", "format"), &MultiMesh::set_transform_format);
	ClassDB::bind_method(D_METHOD("get_transform_format"), &MultiMesh::get_transform_format);

	ClassDB::bind_method(D_METHOD("set_instance_count", "count"), &MultiMesh::set_instance_count);
	ClassDB::bind_method(D_METHOD("get_instance_count"), &MultiMesh::get_instance_count);
	ClassDB::bind_method(D_METHOD("set_visible_instance_count", "count"), &MultiMesh::set_visible_instance_count);
	ClassDB::bind_method(D_METHOD("get_visible_instance_count"), &MultiMesh::get_visible_instance_count);
	ClassDB::bind_method(D_METHOD("set_instance_transform", "instance", "transform"), &MultiMesh::set_instance_transform);
	ClassDB::bind_method(D_METHOD("set_instance_transform_2d", "instance", "transform"), &MultiMesh::set_instance_transform_2d);
	ClassDB::bind_method(D_METHOD("get_instance_transform", "instance"), &MultiMesh::get_instance_transform);
	ClassDB::bind_method(D_METHOD("get_instance_transform_2d", "instance"), &MultiMesh::get_instance_transform_2d);
	ClassDB::bind_method(D_METHOD("set_instance_color", "instance", "color"), &MultiMesh::set_instance_color);
	ClassDB::bind_method(D_METHOD("get_instance_color", "instance"), &MultiMesh::get_instance_color);
	ClassDB::bind_method(D_METHOD("set_instance_custom_data", "instance", "custom_data"), &MultiMesh::set_instance_custom_data);
	ClassDB::bind_method(D_METHOD("get_instance_custom_data", "instance"), &MultiMesh::get_instance_custom_data);
	ClassDB::bind_method(D_METHOD("get_aabb"), &MultiMesh::get_aabb);

	ClassDB::bind_method(D_METHOD("get_buffer"), &MultiMesh::get_buffer);
	ClassDB::bind_method(D_METHOD("set_buffer", "buffer"), &MultiMesh::set_buffer);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "transform_format", PROPERTY_HINT_ENUM, "2D,3D"), "set_transform_format", "get_transform_format");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_colors"), "set_use_colors", "is_using_colors");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_custom_data"), "set_use_custom_data", "is_using_custom_data");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "instance_count", PROPERTY_HINT_RANGE, "0,16384,1,or_greater"), "set_instance_count", "get_instance_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "visible_instance_count", PROPERTY_HINT_RANGE, "-1,16384,1,or_greater"), "set_visible_instance_count", "get_visible_instance_count");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "buffer", PROPERTY_HINT_NONE), "set_buffer", "get_buffer");

#ifndef DISABLE_DEPRECATED
	// Kept for compatibility from 3.x to 4.0.
	ClassDB::bind_method(D_METHOD("_set_transform_array"), &MultiMesh::_set_transform_array);
	ClassDB::bind_method(D_METHOD("_get_transform_array"), &MultiMesh::_get_transform_array);
	ClassDB::bind_method(D_METHOD("_set_transform_2d_array"), &MultiMesh::_set_transform_2d_array);
	ClassDB::bind_method(D_METHOD("_get_transform_2d_array"), &MultiMesh::_get_transform_2d_array);
	ClassDB::bind_method(D_METHOD("_set_color_array"), &MultiMesh::_set_color_array);
	ClassDB::bind_method(D_METHOD("_get_color_array"), &MultiMesh::_get_color_array);
	ClassDB::bind_method(D_METHOD("_set_custom_data_array"), &MultiMesh::_set_custom_data_array);
	ClassDB::bind_method(D_METHOD("_get_custom_data_array"), &MultiMesh::_get_custom_data_array);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR3_ARRAY, "transform_array", PROPERTY_HINT_NONE, "", 0), "_set_transform_array", "_get_transform_array");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, "transform_2d_array", PROPERTY_HINT_NONE, "", 0), "_set_transform_2d_array", "_get_transform_2d_array");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_COLOR_ARRAY, "color_array", PROPERTY_HINT_NONE, "", 0), "_set_color_array", "_get_color_array");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_COLOR_ARRAY, "custom_data_array", PROPERTY_HINT_NONE, "", 0), "_set_custom_data_array", "_get_custom_data_array");
#endif

	BIND_ENUM_CONSTANT(TRANSFORM_2D);
	BIND_ENUM_CONSTANT(TRANSFORM_3D);
}

MultiMesh::MultiMesh() {
	multimesh = RenderingServer::get_singleton()->multimesh_create();
	use_colors = false;
	use_custom_data = false;
	transform_format = TRANSFORM_2D;
	visible_instance_count = -1;
	instance_count = 0;
}

MultiMesh::~MultiMesh() {
	RenderingServer::get_singleton()->free(multimesh);
}

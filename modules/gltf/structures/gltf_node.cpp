/**************************************************************************/
/*  gltf_node.cpp                                                         */
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

#include "gltf_node.h"

void GLTFNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_original_name"), &GLTFNode::get_original_name);
	ClassDB::bind_method(D_METHOD("set_original_name", "original_name"), &GLTFNode::set_original_name);
	ClassDB::bind_method(D_METHOD("get_parent"), &GLTFNode::get_parent);
	ClassDB::bind_method(D_METHOD("set_parent", "parent"), &GLTFNode::set_parent);
	ClassDB::bind_method(D_METHOD("get_height"), &GLTFNode::get_height);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &GLTFNode::set_height);
	ClassDB::bind_method(D_METHOD("get_xform"), &GLTFNode::get_xform);
	ClassDB::bind_method(D_METHOD("set_xform", "xform"), &GLTFNode::set_xform);
	ClassDB::bind_method(D_METHOD("get_mesh"), &GLTFNode::get_mesh);
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &GLTFNode::set_mesh);
	ClassDB::bind_method(D_METHOD("get_camera"), &GLTFNode::get_camera);
	ClassDB::bind_method(D_METHOD("set_camera", "camera"), &GLTFNode::set_camera);
	ClassDB::bind_method(D_METHOD("get_skin"), &GLTFNode::get_skin);
	ClassDB::bind_method(D_METHOD("set_skin", "skin"), &GLTFNode::set_skin);
	ClassDB::bind_method(D_METHOD("get_skeleton"), &GLTFNode::get_skeleton);
	ClassDB::bind_method(D_METHOD("set_skeleton", "skeleton"), &GLTFNode::set_skeleton);
	ClassDB::bind_method(D_METHOD("get_position"), &GLTFNode::get_position);
	ClassDB::bind_method(D_METHOD("set_position", "position"), &GLTFNode::set_position);
	ClassDB::bind_method(D_METHOD("get_rotation"), &GLTFNode::get_rotation);
	ClassDB::bind_method(D_METHOD("set_rotation", "rotation"), &GLTFNode::set_rotation);
	ClassDB::bind_method(D_METHOD("get_scale"), &GLTFNode::get_scale);
	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &GLTFNode::set_scale);
	ClassDB::bind_method(D_METHOD("get_children"), &GLTFNode::get_children);
	ClassDB::bind_method(D_METHOD("set_children", "children"), &GLTFNode::set_children);
	ClassDB::bind_method(D_METHOD("get_light"), &GLTFNode::get_light);
	ClassDB::bind_method(D_METHOD("set_light", "light"), &GLTFNode::set_light);
	ClassDB::bind_method(D_METHOD("get_additional_data", "extension_name"), &GLTFNode::get_additional_data);
	ClassDB::bind_method(D_METHOD("set_additional_data", "extension_name", "additional_data"), &GLTFNode::set_additional_data);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "original_name"), "set_original_name", "get_original_name"); // String
	ADD_PROPERTY(PropertyInfo(Variant::INT, "parent"), "set_parent", "get_parent"); // GLTFNodeIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "height"), "set_height", "get_height"); // int
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "xform"), "set_xform", "get_xform"); // Transform3D
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mesh"), "set_mesh", "get_mesh"); // GLTFMeshIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "camera"), "set_camera", "get_camera"); // GLTFCameraIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "skin"), "set_skin", "get_skin"); // GLTFSkinIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "skeleton"), "set_skeleton", "get_skeleton"); // GLTFSkeletonIndex
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "position"), "set_position", "get_position"); // Vector3
	ADD_PROPERTY(PropertyInfo(Variant::QUATERNION, "rotation"), "set_rotation", "get_rotation"); // Quaternion
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "scale"), "set_scale", "get_scale"); // Vector3
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "children"), "set_children", "get_children"); // Vector<int>
	ADD_PROPERTY(PropertyInfo(Variant::INT, "light"), "set_light", "get_light"); // GLTFLightIndex
}

String GLTFNode::get_original_name() {
	return original_name;
}
void GLTFNode::set_original_name(String p_name) {
	original_name = p_name;
}

GLTFNodeIndex GLTFNode::get_parent() {
	return parent;
}

void GLTFNode::set_parent(GLTFNodeIndex p_parent) {
	parent = p_parent;
}

int GLTFNode::get_height() {
	return height;
}

void GLTFNode::set_height(int p_height) {
	height = p_height;
}

Transform3D GLTFNode::get_xform() {
	return transform;
}

void GLTFNode::set_xform(Transform3D p_xform) {
	transform = p_xform;
}

GLTFMeshIndex GLTFNode::get_mesh() {
	return mesh;
}

void GLTFNode::set_mesh(GLTFMeshIndex p_mesh) {
	mesh = p_mesh;
}

GLTFCameraIndex GLTFNode::get_camera() {
	return camera;
}

void GLTFNode::set_camera(GLTFCameraIndex p_camera) {
	camera = p_camera;
}

GLTFSkinIndex GLTFNode::get_skin() {
	return skin;
}

void GLTFNode::set_skin(GLTFSkinIndex p_skin) {
	skin = p_skin;
}

GLTFSkeletonIndex GLTFNode::get_skeleton() {
	return skeleton;
}

void GLTFNode::set_skeleton(GLTFSkeletonIndex p_skeleton) {
	skeleton = p_skeleton;
}

Vector3 GLTFNode::get_position() {
	return transform.origin;
}

void GLTFNode::set_position(Vector3 p_position) {
	transform.origin = p_position;
}

Quaternion GLTFNode::get_rotation() {
	return transform.basis.get_rotation_quaternion();
}

void GLTFNode::set_rotation(Quaternion p_rotation) {
	transform.basis.set_quaternion_scale(p_rotation, transform.basis.get_scale());
}

Vector3 GLTFNode::get_scale() {
	return transform.basis.get_scale();
}

void GLTFNode::set_scale(Vector3 p_scale) {
	transform.basis = transform.basis.orthonormalized() * Basis::from_scale(p_scale);
}

Vector<int> GLTFNode::get_children() {
	return children;
}

void GLTFNode::set_children(Vector<int> p_children) {
	children = p_children;
}

GLTFLightIndex GLTFNode::get_light() {
	return light;
}

void GLTFNode::set_light(GLTFLightIndex p_light) {
	light = p_light;
}

Variant GLTFNode::get_additional_data(const StringName &p_extension_name) {
	return additional_data[p_extension_name];
}

void GLTFNode::set_additional_data(const StringName &p_extension_name, Variant p_additional_data) {
	additional_data[p_extension_name] = p_additional_data;
}

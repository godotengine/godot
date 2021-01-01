/*************************************************************************/
/*  gltf_node.cpp                                                        */
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

#include "gltf_node.h"

void GLTFNode::_bind_methods() {
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
	ClassDB::bind_method(D_METHOD("get_joint"), &GLTFNode::get_joint);
	ClassDB::bind_method(D_METHOD("set_joint", "joint"), &GLTFNode::set_joint);
	ClassDB::bind_method(D_METHOD("get_translation"), &GLTFNode::get_translation);
	ClassDB::bind_method(D_METHOD("set_translation", "translation"), &GLTFNode::set_translation);
	ClassDB::bind_method(D_METHOD("get_rotation"), &GLTFNode::get_rotation);
	ClassDB::bind_method(D_METHOD("set_rotation", "rotation"), &GLTFNode::set_rotation);
	ClassDB::bind_method(D_METHOD("get_scale"), &GLTFNode::get_scale);
	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &GLTFNode::set_scale);
	ClassDB::bind_method(D_METHOD("get_children"), &GLTFNode::get_children);
	ClassDB::bind_method(D_METHOD("set_children", "children"), &GLTFNode::set_children);
	ClassDB::bind_method(D_METHOD("get_fake_joint_parent"), &GLTFNode::get_fake_joint_parent);
	ClassDB::bind_method(D_METHOD("set_fake_joint_parent", "fake_joint_parent"), &GLTFNode::set_fake_joint_parent);
	ClassDB::bind_method(D_METHOD("get_light"), &GLTFNode::get_light);
	ClassDB::bind_method(D_METHOD("set_light", "light"), &GLTFNode::set_light);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "parent"), "set_parent", "get_parent"); // GLTFNodeIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "height"), "set_height", "get_height"); // int
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "xform"), "set_xform", "get_xform"); // Transform
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mesh"), "set_mesh", "get_mesh"); // GLTFMeshIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "camera"), "set_camera", "get_camera"); // GLTFCameraIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "skin"), "set_skin", "get_skin"); // GLTFSkinIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "skeleton"), "set_skeleton", "get_skeleton"); // GLTFSkeletonIndex
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "joint"), "set_joint", "get_joint"); // bool
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "translation"), "set_translation", "get_translation"); // Vector3
	ADD_PROPERTY(PropertyInfo(Variant::QUAT, "rotation"), "set_rotation", "get_rotation"); // Quat
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "scale"), "set_scale", "get_scale"); // Vector3
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "children"), "set_children", "get_children"); // Vector<int>
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fake_joint_parent"), "set_fake_joint_parent", "get_fake_joint_parent"); // GLTFNodeIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "light"), "set_light", "get_light"); // GLTFLightIndex
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

Transform GLTFNode::get_xform() {
	return xform;
}

void GLTFNode::set_xform(Transform p_xform) {
	xform = p_xform;
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

bool GLTFNode::get_joint() {
	return joint;
}

void GLTFNode::set_joint(bool p_joint) {
	joint = p_joint;
}

Vector3 GLTFNode::get_translation() {
	return translation;
}

void GLTFNode::set_translation(Vector3 p_translation) {
	translation = p_translation;
}

Quat GLTFNode::get_rotation() {
	return rotation;
}

void GLTFNode::set_rotation(Quat p_rotation) {
	rotation = p_rotation;
}

Vector3 GLTFNode::get_scale() {
	return scale;
}

void GLTFNode::set_scale(Vector3 p_scale) {
	scale = p_scale;
}

Vector<int> GLTFNode::get_children() {
	return children;
}

void GLTFNode::set_children(Vector<int> p_children) {
	children = p_children;
}

GLTFNodeIndex GLTFNode::get_fake_joint_parent() {
	return fake_joint_parent;
}

void GLTFNode::set_fake_joint_parent(GLTFNodeIndex p_fake_joint_parent) {
	fake_joint_parent = p_fake_joint_parent;
}

GLTFLightIndex GLTFNode::get_light() {
	return light;
}

void GLTFNode::set_light(GLTFLightIndex p_light) {
	light = p_light;
}

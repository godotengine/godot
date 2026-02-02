/**************************************************************************/
/*  usd_node.cpp                                                          */
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

#include "usd_node.h"

void USDNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_original_name"), &USDNode::get_original_name);
	ClassDB::bind_method(D_METHOD("set_original_name", "original_name"), &USDNode::set_original_name);
	ClassDB::bind_method(D_METHOD("get_parent"), &USDNode::get_parent);
	ClassDB::bind_method(D_METHOD("set_parent", "parent"), &USDNode::set_parent);
	ClassDB::bind_method(D_METHOD("get_children"), &USDNode::get_children);
	ClassDB::bind_method(D_METHOD("set_children", "children"), &USDNode::set_children);
	ClassDB::bind_method(D_METHOD("append_child_index", "child_index"), &USDNode::append_child_index);
	ClassDB::bind_method(D_METHOD("get_xform"), &USDNode::get_xform);
	ClassDB::bind_method(D_METHOD("set_xform", "xform"), &USDNode::set_xform);
	ClassDB::bind_method(D_METHOD("get_mesh"), &USDNode::get_mesh);
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &USDNode::set_mesh);
	ClassDB::bind_method(D_METHOD("get_camera"), &USDNode::get_camera);
	ClassDB::bind_method(D_METHOD("set_camera", "camera"), &USDNode::set_camera);
	ClassDB::bind_method(D_METHOD("get_light"), &USDNode::get_light);
	ClassDB::bind_method(D_METHOD("set_light", "light"), &USDNode::set_light);
	ClassDB::bind_method(D_METHOD("get_skeleton"), &USDNode::get_skeleton);
	ClassDB::bind_method(D_METHOD("set_skeleton", "skeleton"), &USDNode::set_skeleton);
	ClassDB::bind_method(D_METHOD("get_skin"), &USDNode::get_skin);
	ClassDB::bind_method(D_METHOD("set_skin", "skin"), &USDNode::set_skin);
	ClassDB::bind_method(D_METHOD("get_visible"), &USDNode::get_visible);
	ClassDB::bind_method(D_METHOD("set_visible", "visible"), &USDNode::set_visible);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "original_name"), "set_original_name", "get_original_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "parent"), "set_parent", "get_parent");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "children"), "set_children", "get_children");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "xform"), "set_xform", "get_xform");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "camera"), "set_camera", "get_camera");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "light"), "set_light", "get_light");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "skeleton"), "set_skeleton", "get_skeleton");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "skin"), "set_skin", "get_skin");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "visible"), "set_visible", "get_visible");
}

String USDNode::get_original_name() const {
	return original_name;
}

void USDNode::set_original_name(const String &p_name) {
	original_name = p_name;
}

int USDNode::get_parent() const {
	return parent;
}

void USDNode::set_parent(int p_parent) {
	parent = p_parent;
}

Vector<int> USDNode::get_children() const {
	return children;
}

void USDNode::set_children(const Vector<int> &p_children) {
	children = p_children;
}

void USDNode::append_child_index(int p_child_index) {
	children.push_back(p_child_index);
}

Transform3D USDNode::get_xform() const {
	return xform;
}

void USDNode::set_xform(const Transform3D &p_xform) {
	xform = p_xform;
}

int USDNode::get_mesh() const {
	return mesh;
}

void USDNode::set_mesh(int p_mesh) {
	mesh = p_mesh;
}

int USDNode::get_camera() const {
	return camera;
}

void USDNode::set_camera(int p_camera) {
	camera = p_camera;
}

int USDNode::get_light() const {
	return light;
}

void USDNode::set_light(int p_light) {
	light = p_light;
}

int USDNode::get_skeleton() const {
	return skeleton;
}

void USDNode::set_skeleton(int p_skeleton) {
	skeleton = p_skeleton;
}

int USDNode::get_skin() const {
	return skin;
}

void USDNode::set_skin(int p_skin) {
	skin = p_skin;
}

bool USDNode::get_visible() const {
	return visible;
}

void USDNode::set_visible(bool p_visible) {
	visible = p_visible;
}

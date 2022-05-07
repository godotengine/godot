/*************************************************************************/
/*  gltf_skin.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gltf_skin.h"

void GLTFSkin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_skin_root"), &GLTFSkin::get_skin_root);
	ClassDB::bind_method(D_METHOD("set_skin_root", "skin_root"), &GLTFSkin::set_skin_root);
	ClassDB::bind_method(D_METHOD("get_joints_original"), &GLTFSkin::get_joints_original);
	ClassDB::bind_method(D_METHOD("set_joints_original", "joints_original"), &GLTFSkin::set_joints_original);
	ClassDB::bind_method(D_METHOD("get_inverse_binds"), &GLTFSkin::get_inverse_binds);
	ClassDB::bind_method(D_METHOD("set_inverse_binds", "inverse_binds"), &GLTFSkin::set_inverse_binds);
	ClassDB::bind_method(D_METHOD("get_joints"), &GLTFSkin::get_joints);
	ClassDB::bind_method(D_METHOD("set_joints", "joints"), &GLTFSkin::set_joints);
	ClassDB::bind_method(D_METHOD("get_non_joints"), &GLTFSkin::get_non_joints);
	ClassDB::bind_method(D_METHOD("set_non_joints", "non_joints"), &GLTFSkin::set_non_joints);
	ClassDB::bind_method(D_METHOD("get_roots"), &GLTFSkin::get_roots);
	ClassDB::bind_method(D_METHOD("set_roots", "roots"), &GLTFSkin::set_roots);
	ClassDB::bind_method(D_METHOD("get_skeleton"), &GLTFSkin::get_skeleton);
	ClassDB::bind_method(D_METHOD("set_skeleton", "skeleton"), &GLTFSkin::set_skeleton);
	ClassDB::bind_method(D_METHOD("get_joint_i_to_bone_i"), &GLTFSkin::get_joint_i_to_bone_i);
	ClassDB::bind_method(D_METHOD("set_joint_i_to_bone_i", "joint_i_to_bone_i"), &GLTFSkin::set_joint_i_to_bone_i);
	ClassDB::bind_method(D_METHOD("get_joint_i_to_name"), &GLTFSkin::get_joint_i_to_name);
	ClassDB::bind_method(D_METHOD("set_joint_i_to_name", "joint_i_to_name"), &GLTFSkin::set_joint_i_to_name);
	ClassDB::bind_method(D_METHOD("get_godot_skin"), &GLTFSkin::get_godot_skin);
	ClassDB::bind_method(D_METHOD("set_godot_skin", "godot_skin"), &GLTFSkin::set_godot_skin);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "skin_root"), "set_skin_root", "get_skin_root"); // GLTFNodeIndex
	ADD_PROPERTY(PropertyInfo(Variant::POOL_INT_ARRAY, "joints_original"), "set_joints_original", "get_joints_original"); // Vector<GLTFNodeIndex>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "inverse_binds", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL), "set_inverse_binds", "get_inverse_binds"); // Vector<Transform>
	ADD_PROPERTY(PropertyInfo(Variant::POOL_INT_ARRAY, "joints"), "set_joints", "get_joints"); // Vector<GLTFNodeIndex>
	ADD_PROPERTY(PropertyInfo(Variant::POOL_INT_ARRAY, "non_joints"), "set_non_joints", "get_non_joints"); // Vector<GLTFNodeIndex>
	ADD_PROPERTY(PropertyInfo(Variant::POOL_INT_ARRAY, "roots"), "set_roots", "get_roots"); // Vector<GLTFNodeIndex>
	ADD_PROPERTY(PropertyInfo(Variant::INT, "skeleton"), "set_skeleton", "get_skeleton"); // int
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "joint_i_to_bone_i", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL), "set_joint_i_to_bone_i", "get_joint_i_to_bone_i"); // Map<int,
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "joint_i_to_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL), "set_joint_i_to_name", "get_joint_i_to_name"); // Map<int,
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "godot_skin"), "set_godot_skin", "get_godot_skin"); // Ref<Skin>
}

GLTFNodeIndex GLTFSkin::get_skin_root() {
	return skin_root;
}

void GLTFSkin::set_skin_root(GLTFNodeIndex p_skin_root) {
	skin_root = p_skin_root;
}

Vector<GLTFNodeIndex> GLTFSkin::get_joints_original() {
	return joints_original;
}

void GLTFSkin::set_joints_original(Vector<GLTFNodeIndex> p_joints_original) {
	joints_original = p_joints_original;
}

Array GLTFSkin::get_inverse_binds() {
	return GLTFDocument::to_array(inverse_binds);
}

void GLTFSkin::set_inverse_binds(Array p_inverse_binds) {
	GLTFDocument::set_from_array(inverse_binds, p_inverse_binds);
}

Vector<GLTFNodeIndex> GLTFSkin::get_joints() {
	return joints;
}

void GLTFSkin::set_joints(Vector<GLTFNodeIndex> p_joints) {
	joints = p_joints;
}

Vector<GLTFNodeIndex> GLTFSkin::get_non_joints() {
	return non_joints;
}

void GLTFSkin::set_non_joints(Vector<GLTFNodeIndex> p_non_joints) {
	non_joints = p_non_joints;
}

Vector<GLTFNodeIndex> GLTFSkin::get_roots() {
	return roots;
}

void GLTFSkin::set_roots(Vector<GLTFNodeIndex> p_roots) {
	roots = p_roots;
}

int GLTFSkin::get_skeleton() {
	return skeleton;
}

void GLTFSkin::set_skeleton(int p_skeleton) {
	skeleton = p_skeleton;
}

Dictionary GLTFSkin::get_joint_i_to_bone_i() {
	return GLTFDocument::to_dict(joint_i_to_bone_i);
}

void GLTFSkin::set_joint_i_to_bone_i(Dictionary p_joint_i_to_bone_i) {
	GLTFDocument::set_from_dict(joint_i_to_bone_i, p_joint_i_to_bone_i);
}

Dictionary GLTFSkin::get_joint_i_to_name() {
	Dictionary ret;
	Map<int, StringName>::Element *elem = joint_i_to_name.front();
	while (elem) {
		ret[elem->key()] = String(elem->value());
		elem = elem->next();
	}
	return ret;
}

void GLTFSkin::set_joint_i_to_name(Dictionary p_joint_i_to_name) {
	joint_i_to_name = Map<int, StringName>();
	Array keys = p_joint_i_to_name.keys();
	for (int i = 0; i < keys.size(); i++) {
		joint_i_to_name[keys[i]] = p_joint_i_to_name[keys[i]];
	}
}

Ref<Skin> GLTFSkin::get_godot_skin() {
	return godot_skin;
}

void GLTFSkin::set_godot_skin(Ref<Skin> p_godot_skin) {
	godot_skin = p_godot_skin;
}

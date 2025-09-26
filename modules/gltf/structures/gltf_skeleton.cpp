/**************************************************************************/
/*  gltf_skeleton.cpp                                                     */
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

#include "gltf_skeleton.h"

#include "../gltf_template_convert.h"

#include "scene/3d/bone_attachment_3d.h"

void GLTFSkeleton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_joints"), &GLTFSkeleton::get_joints);
	ClassDB::bind_method(D_METHOD("set_joints", "joints"), &GLTFSkeleton::set_joints);
	ClassDB::bind_method(D_METHOD("get_roots"), &GLTFSkeleton::get_roots);
	ClassDB::bind_method(D_METHOD("set_roots", "roots"), &GLTFSkeleton::set_roots);
	ClassDB::bind_method(D_METHOD("get_godot_skeleton"), &GLTFSkeleton::get_godot_skeleton);
	ClassDB::bind_method(D_METHOD("get_unique_names"), &GLTFSkeleton::get_unique_names);
	ClassDB::bind_method(D_METHOD("set_unique_names", "unique_names"), &GLTFSkeleton::set_unique_names);
	ClassDB::bind_method(D_METHOD("get_godot_bone_node"), &GLTFSkeleton::get_godot_bone_node);
	ClassDB::bind_method(D_METHOD("set_godot_bone_node", "godot_bone_node"), &GLTFSkeleton::set_godot_bone_node);
	ClassDB::bind_method(D_METHOD("get_bone_attachment_count"), &GLTFSkeleton::get_bone_attachment_count);
	ClassDB::bind_method(D_METHOD("get_bone_attachment", "idx"), &GLTFSkeleton::get_bone_attachment);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "joints"), "set_joints", "get_joints"); // Vector<GLTFNodeIndex>
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "roots"), "set_roots", "get_roots"); // Vector<GLTFNodeIndex>
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "unique_names", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_unique_names", "get_unique_names"); // Set<String>
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "godot_bone_node", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_EDITOR), "set_godot_bone_node", "get_godot_bone_node"); // RBMap<int32_t,
}

Vector<GLTFNodeIndex> GLTFSkeleton::get_joints() {
	return Vector<GLTFNodeIndex>(joints);
}

void GLTFSkeleton::set_joints(const Vector<GLTFNodeIndex> &p_joints) {
	joints = Vector<GLTFNodeIndex>(p_joints);
}

Vector<GLTFNodeIndex> GLTFSkeleton::get_roots() {
	return Vector<GLTFNodeIndex>(roots);
}

void GLTFSkeleton::set_roots(const Vector<GLTFNodeIndex> &p_roots) {
	roots = Vector<GLTFNodeIndex>(p_roots);
}

Skeleton3D *GLTFSkeleton::get_godot_skeleton() {
	return godot_skeleton;
}

TypedArray<String> GLTFSkeleton::get_unique_names() {
	return GLTFTemplateConvert::to_array(unique_names);
}

void GLTFSkeleton::set_unique_names(const TypedArray<String> &p_unique_names) {
	GLTFTemplateConvert::set_from_array(unique_names, p_unique_names);
}

Dictionary GLTFSkeleton::get_godot_bone_node() {
	return GLTFTemplateConvert::to_dictionary(godot_bone_node);
}

void GLTFSkeleton::set_godot_bone_node(const Dictionary &p_indict) {
	GLTFTemplateConvert::set_from_dictionary(godot_bone_node, p_indict);
}

BoneAttachment3D *GLTFSkeleton::get_bone_attachment(int idx) {
	ERR_FAIL_INDEX_V(idx, bone_attachments.size(), nullptr);
	return bone_attachments[idx];
}

int32_t GLTFSkeleton::get_bone_attachment_count() {
	return bone_attachments.size();
}

/**************************************************************************/
/*  gltf_document_extension_mesh_gpu_instancing.cpp                       */
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

#include "gltf_document_extension_mesh_gpu_instancing.h"

#include "../gltf_document.h"
#include "scene/3d/multimesh_instance_3d.h"

Ref<MultiMesh> GLTFDocumentExtensionMeshGPUInstancing::_generate_multimesh(Ref<GLTFState> p_state, Dictionary &p_extensions) {
	if (!p_extensions.has("EXT_mesh_gpu_instancing")) {
		return nullptr;
	}

	Dictionary mesh_gpu_instancing = p_extensions["EXT_mesh_gpu_instancing"];
	if (!mesh_gpu_instancing.has("attributes")) {
		return nullptr;
	}

	Dictionary instancing_attributes = mesh_gpu_instancing["attributes"];

	int instances_count = 0;
	PackedVector3Array translations;
	Vector<Quaternion> rotations;
	PackedVector3Array scales;

	TypedArray<Ref<GLTFAccessor>> state_accessors = p_state->get_accessors();
	TypedArray<Ref<GLTFBufferView>> state_buffer_views = p_state->get_buffer_views();
	TypedArray<PackedByteArray> state_buffers = p_state->get_buffers();

	if (instancing_attributes.has("TRANSLATION")) {
		int multimesh_translation = instancing_attributes["TRANSLATION"];
		Ref<GLTFAccessor> accessor = state_accessors[multimesh_translation];
		if (instances_count && instances_count != accessor->get_count()) {
			ERR_PRINT(vformat("glTF import: translation instances_count: %i", accessor->get_count()));
			return nullptr;
		}

		instances_count = accessor->get_count();
		translations = accessor->decode_as_vector3s(p_state);
	}

	if (instancing_attributes.has("ROTATION")) {
		int multimesh_rotation = instancing_attributes["ROTATION"];
		Ref<GLTFAccessor> accessor = state_accessors[multimesh_rotation];
		if (instances_count && instances_count != accessor->get_count()) {
			ERR_PRINT(vformat("glTF import: rotation instances_count: %i", accessor->get_count()));
			return nullptr;
		}

		instances_count = accessor->get_count();
		rotations = accessor->decode_as_quaternions(p_state);
	}

	if (instancing_attributes.has("SCALE")) {
		int multimesh_scale = instancing_attributes["SCALE"];
		Ref<GLTFAccessor> accessor = state_accessors[multimesh_scale];
		if (instances_count && instances_count != accessor->get_count()) {
			ERR_PRINT(vformat("glTF import: scale instances_count: %i", accessor->get_count()));
			return nullptr;
		}
		instances_count = accessor->get_count();

		scales = accessor->decode_as_vector3s(p_state);
	}

	Ref<MultiMesh> multimesh = memnew(MultiMesh);
	multimesh->set_instance_count(0);
	multimesh->set_transform_format(MultiMesh::TRANSFORM_3D);
	multimesh->set_instance_count(instances_count);

	for (int i = 0; i < instances_count; i++) {
		Transform3D tr;

		if (translations.size()) {
			tr = tr.translated(translations[i]);
		}

		if (rotations.size()) {
			tr.set_basis(Basis(rotations[i]));
		}

		if (scales.size()) {
			tr.scale_basis(scales[i]);
		}

		multimesh->set_instance_transform(i, tr);
	}

	return multimesh;
}

Error GLTFDocumentExtensionMeshGPUInstancing::import_preflight(Ref<GLTFState> p_state, const Vector<String> &p_extensions) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);
	Error err = OK;

	if (p_extensions.has("EXT_mesh_gpu_instancing")) {
		// TODO ?
	}

	return err;
}

Node3D *GLTFDocumentExtensionMeshGPUInstancing::generate_scene_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Node *p_scene_parent) {
	ERR_FAIL_COND_V(p_state.is_null(), nullptr);
	ERR_FAIL_COND_V(p_gltf_node.is_null(), nullptr);
	Node3D *ret_node = nullptr;

	// TODO actual creation?

	return ret_node;
}

Error GLTFDocumentExtensionMeshGPUInstancing::import_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Dictionary &r_dict, Node *p_node) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_gltf_node.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_NULL_V(p_node, ERR_INVALID_PARAMETER);

	if (!r_dict.has("extensions")) {
		return OK;
	}

	Dictionary extensions = r_dict["extensions"];
	if (!extensions.has("EXT_mesh_gpu_instancing")) {
		return OK;
	}

	ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(p_node);
	if (!mi) {
		return OK;
	}

	Ref<MultiMesh> multimesh = _generate_multimesh(p_state, extensions);
	if (multimesh.is_valid()) {
		if (mi->get_mesh().is_valid()) {
			multimesh->set_mesh(mi->get_mesh());
		}

		mi->set_multimesh(multimesh);
	}

	Error err = OK;
	return err;
}

Error GLTFDocumentExtensionMeshGPUInstancing::export_preflight(Ref<GLTFState> p_state, Node *p_root) {
	ERR_FAIL_NULL_V(p_root, ERR_INVALID_PARAMETER);

	p_state->set_ignore_multimesh_instances(true);

	return OK;
}

void GLTFDocumentExtensionMeshGPUInstancing::convert_scene_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Node *p_scene_node) {
	ERR_FAIL_COND(p_state.is_null());
	ERR_FAIL_COND(p_gltf_node.is_null());
	ERR_FAIL_NULL(p_scene_node);

	MultiMeshInstance3D *multi_mesh_instance = Object::cast_to<MultiMeshInstance3D>(p_scene_node);
	if (!multi_mesh_instance) {
		return;
	}

	GLTFMeshIndex mesh_index = GLTFDocument::_convert_multimesh_to_gltf(p_state, multi_mesh_instance);
	p_gltf_node->set_mesh(mesh_index);

	Ref<MultiMesh> multi_mesh = multi_mesh_instance->get_multimesh();
	if (!multi_mesh.is_valid()) {
		return;
	}

	// TODO 2D?
	if (multi_mesh->get_transform_format() != MultiMesh::TRANSFORM_3D) {
		return;
	}

	///////////////////////////////////////////////////////////////////////////

	Vector<Vector3> translation;
	Vector<Quaternion> rotation;
	Vector<Vector3> scale;

	for (int32_t instance_i = 0; instance_i < multi_mesh->get_instance_count(); instance_i++) {
		Transform3D transform = multi_mesh->get_instance_transform(instance_i);
		translation.push_back(transform.get_origin());
		rotation.push_back(transform.get_basis().get_rotation_quaternion());
		scale.push_back(transform.get_basis().get_scale());
	}

	Dictionary instancing_attributes;
	instancing_attributes["TRANSLATION"] = GLTFAccessor::encode_new_accessor_from_vector3s(p_state, translation, GLTFBufferView::TARGET_ARRAY_BUFFER, false);
	instancing_attributes["ROTATION"] = GLTFAccessor::encode_new_accessor_from_quaternions(p_state, rotation, GLTFBufferView::TARGET_ARRAY_BUFFER, false);
	instancing_attributes["SCALE"] = GLTFAccessor::encode_new_accessor_from_vector3s(p_state, scale, GLTFBufferView::TARGET_ARRAY_BUFFER, false);

	Dictionary mesh_gpu_instancing;
	mesh_gpu_instancing["attributes"] = instancing_attributes;

	///////////////////////////////////////////////////////////////////////////

	p_gltf_node->set_additional_data("EXT_mesh_gpu_instancing", mesh_gpu_instancing);
}

Error GLTFDocumentExtensionMeshGPUInstancing::export_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Dictionary &r_json, Node *p_node) {
	ERR_FAIL_COND_V(p_state.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_gltf_node.is_null(), ERR_INVALID_PARAMETER);

	MultiMeshInstance3D *multi_mesh_instance = Object::cast_to<MultiMeshInstance3D>(p_node);

	if (!multi_mesh_instance) {
		return OK;
	}

	if (p_gltf_node->has_additional_data("EXT_mesh_gpu_instancing")) {
		Dictionary mesh_gpu_instancing = p_gltf_node->get_additional_data("EXT_mesh_gpu_instancing");

		if (!r_json.has("extensions")) {
			r_json["extensions"] = Dictionary();
		}
		Dictionary extensions = r_json["extensions"];
		extensions["EXT_mesh_gpu_instancing"] = mesh_gpu_instancing;
	}

	Error err = OK;
	return err;
}

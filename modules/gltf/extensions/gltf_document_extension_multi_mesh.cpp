/**************************************************************************/
/*  gltf_document_extension_multi_mesh.cpp                                */
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

#include "gltf_document_extension_multi_mesh.h"

// Import process.
Error GLTFDocumentExtensionMultiMesh::import_preflight(Ref<GLTFState> p_gltf_state, const Vector<String> &p_extensions) {
	if (!p_extensions.has("EXT_mesh_gpu_instancing")) {
		return ERR_SKIP;
	}
	return OK;
}

Vector<String> GLTFDocumentExtensionMultiMesh::get_supported_extensions() {
	Vector<String> ret;
	ret.push_back("EXT_mesh_gpu_instancing");
	return ret;
}

Error GLTFDocumentExtensionMultiMesh::parse_node_extensions(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, const Dictionary &p_extensions) {
	if (!p_extensions.has("EXT_mesh_gpu_instancing")) {
		return OK;
	}
	Dictionary ext_mesh_gpu = p_extensions["EXT_mesh_gpu_instancing"];
	if (!ext_mesh_gpu.has("attributes")) {
		if (p_gltf_node->get_children().is_empty()) {
			// This is a multimesh with zero instances and no children, so it is a fallback mesh node.
			// Since Godot supports EXT_mesh_gpu_instancing, we don't need the fallback nodes, and can skip them.
			p_gltf_node->set_additional_data(StringName("SkipNodeGeneration"), true);
		}
		return OK;
	}
	const Dictionary attributes = ext_mesh_gpu["attributes"];
	if (attributes.is_empty()) {
		if (p_gltf_node->get_children().is_empty()) {
			p_gltf_node->set_additional_data(StringName("SkipNodeGeneration"), true);
		}
		return OK;
	}
	const Vector<Ref<GLTFAccessor>> &accessors = p_state->get_accessors();
	PackedVector3Array positions;
	Vector<Quaternion> rotations;
	PackedVector3Array scales;
	PackedColorArray colors;
	PackedColorArray custom_data;
	const int64_t accessor_count = accessors.size();
	if (attributes.has("TRANSLATION")) {
		GLTFAccessorIndex position_accessor_index = attributes["TRANSLATION"];
		ERR_FAIL_INDEX_V(position_accessor_index, accessor_count, ERR_INVALID_DATA);
		Ref<GLTFAccessor> position_accessor = accessors[position_accessor_index];
		positions = position_accessor->decode_as_vector3s(p_state);
	}
	if (attributes.has("ROTATION")) {
		GLTFAccessorIndex rotation_accessor_index = attributes["ROTATION"];
		ERR_FAIL_INDEX_V(rotation_accessor_index, accessor_count, ERR_INVALID_DATA);
		Ref<GLTFAccessor> rotation_accessor = accessors[rotation_accessor_index];
		rotations = rotation_accessor->decode_as_quaternions(p_state);
	}
	if (attributes.has("SCALE")) {
		GLTFAccessorIndex scale_accessor_index = attributes["SCALE"];
		ERR_FAIL_INDEX_V(scale_accessor_index, accessor_count, ERR_INVALID_DATA);
		Ref<GLTFAccessor> scale_accessor = accessors[scale_accessor_index];
		scales = scale_accessor->decode_as_vector3s(p_state);
	}
	if (attributes.has("_COLOR")) {
		GLTFAccessorIndex color_accessor_index = attributes["_COLOR"];
		ERR_FAIL_INDEX_V(color_accessor_index, accessor_count, ERR_INVALID_DATA);
		Ref<GLTFAccessor> color_accessor = accessors[color_accessor_index];
		colors = color_accessor->decode_as_colors(p_state);
	}
	if (attributes.has("_CUSTOM")) {
		GLTFAccessorIndex custom_accessor_index = attributes["_CUSTOM"];
		ERR_FAIL_INDEX_V(custom_accessor_index, accessor_count, ERR_INVALID_DATA);
		Ref<GLTFAccessor> custom_accessor = accessors[custom_accessor_index];
		custom_data = custom_accessor->decode_as_colors(p_state);
	}
	const int64_t instance_count = MAX(MAX(MAX(positions.size(), rotations.size()), scales.size()), MAX(colors.size(), custom_data.size()));
	TypedArray<Transform3D> multi_mesh_transforms;
	multi_mesh_transforms.resize(instance_count);
	for (int64_t i = 0; i < instance_count; i++) {
		Transform3D transform;
		transform.basis.set_quaternion_scale(
				i < rotations.size() ? rotations[i] : Quaternion(),
				i < scales.size() ? scales[i] : Vector3(1, 1, 1));
		transform.origin = i < positions.size() ? positions[i] : Vector3();
		multi_mesh_transforms[i] = transform;
	}
	p_gltf_node->set_additional_data(StringName("MultiMeshTransforms"), multi_mesh_transforms);
	if (!colors.is_empty()) {
		p_gltf_node->set_additional_data(StringName("MultiMeshColors"), colors);
	}
	if (!custom_data.is_empty()) {
		p_gltf_node->set_additional_data(StringName("MultiMeshCustomData"), custom_data);
	}
	return OK;
}

Ref<Mesh> GLTFDocumentExtensionMultiMesh::_generate_mesh(Ref<GLTFState> p_gltf_state, GLTFMeshIndex p_mesh_index) {
	Ref<Mesh> ret;
	const Vector<Ref<GLTFMesh>> &gltf_meshes = p_gltf_state->get_meshes();
	ERR_FAIL_INDEX_V(p_mesh_index, gltf_meshes.size(), ret);
	Ref<GLTFMesh> gltf_mesh = gltf_meshes[p_mesh_index];
	ERR_FAIL_COND_V(gltf_mesh.is_null(), ret);
	Ref<ImporterMesh> importer_mesh = gltf_mesh->get_mesh();
	ERR_FAIL_COND_V(importer_mesh.is_null(), ret);
	ret = importer_mesh->get_mesh();
	return ret;
}

Node3D *GLTFDocumentExtensionMultiMesh::generate_scene_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Node *p_scene_parent) {
	if (!p_gltf_node->has_additional_data(StringName("MultiMeshTransforms"))) {
		return nullptr;
	}
	const TypedArray<Transform3D> multi_mesh_transforms = TypedArray<Transform3D>(p_gltf_node->get_additional_data(StringName("MultiMeshTransforms")));
	const PackedColorArray multi_mesh_colors = p_gltf_node->has_additional_data(StringName("MultiMeshColors")) ? PackedColorArray(p_gltf_node->get_additional_data(StringName("MultiMeshColors"))) : PackedColorArray();
	const PackedColorArray multi_mesh_custom_data = p_gltf_node->has_additional_data(StringName("MultiMeshCustomData")) ? PackedColorArray(p_gltf_node->get_additional_data(StringName("MultiMeshCustomData"))) : PackedColorArray();
	Ref<MultiMesh> multi_mesh;
	multi_mesh.instantiate();
	multi_mesh->set_use_colors(!multi_mesh_colors.is_empty());
	multi_mesh->set_use_custom_data(!multi_mesh_custom_data.is_empty());
	multi_mesh->set_transform_format(MultiMesh::TRANSFORM_3D);
	const GLTFMeshIndex mesh_index = p_gltf_node->get_mesh();
	if (mesh_index >= 0) {
		const Ref<Mesh> mesh = _generate_mesh(p_state, p_gltf_node->get_mesh());
		if (mesh.is_valid()) {
			multi_mesh->set_mesh(mesh);
		}
	}
	const int64_t instance_count = MAX(multi_mesh_transforms.size(), MAX(multi_mesh_colors.size(), multi_mesh_custom_data.size()));
	multi_mesh->set_instance_count(instance_count);
	for (int64_t i = 0; i < multi_mesh_transforms.size(); i++) {
		multi_mesh->set_instance_transform(i, multi_mesh_transforms[i]);
	}
	for (int64_t i = 0; i < multi_mesh_colors.size(); i++) {
		multi_mesh->set_instance_color(i, multi_mesh_colors[i]);
	}
	for (int64_t i = 0; i < multi_mesh_custom_data.size(); i++) {
		multi_mesh->set_instance_custom_data(i, multi_mesh_custom_data[i]);
	}
	MultiMeshInstance3D *multi_mesh_node = memnew(MultiMeshInstance3D);
	multi_mesh_node->set_multimesh(multi_mesh);
	return multi_mesh_node;
}

// Export process.
bool GLTFDocumentExtensionMultiMesh::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == StringName("handling")) {
		_handling = MultiMeshHandling(int(p_value));
		return true;
	}
	return false;
}

bool GLTFDocumentExtensionMultiMesh::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == StringName("handling")) {
		r_ret = int(_handling);
		return true;
	}
	return false;
}

void GLTFDocumentExtensionMultiMesh::_get_property_list(List<PropertyInfo> *p_list) const {
	if (_show_handling_property) {
		p_list->push_back(PropertyInfo(Variant::INT, "handling", PROPERTY_HINT_ENUM, "Optional EXT_mesh_gpu_instancing,Required EXT_mesh_gpu_instancing,Multiple Nodes Only,Multiple Nodes Fallback", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_SCRIPT_VARIABLE));
	}
}

Error GLTFDocumentExtensionMultiMesh::export_configure_for_scene(Node *p_root_node) {
	if (p_root_node == nullptr) {
		_show_handling_property = true;
	} else {
		_show_handling_property = _any_multi_mesh_instance_exists_recursive(p_root_node);
	}
	return OK;
}

bool GLTFDocumentExtensionMultiMesh::_any_multi_mesh_instance_exists_recursive(Node *p_node) {
	if (Object::cast_to<MultiMeshInstance3D>(p_node) != nullptr) {
		return true;
	}
	for (int32_t i = 0; i < p_node->get_child_count(); i++) {
		Node *child = p_node->get_child(i);
		if (_any_multi_mesh_instance_exists_recursive(child)) {
			return true;
		}
	}
	return false;
}

GLTFMeshIndex GLTFDocumentExtensionMultiMesh::_convert_mesh_resource_into_state(Ref<GLTFState> p_gltf_state, const Ref<MultiMesh> &p_multi_mesh) {
	const Ref<Mesh> mesh = p_multi_mesh->get_mesh();
	if (mesh.is_null()) {
		return -1;
	}
	Ref<GLTFMesh> gltf_mesh;
	gltf_mesh.instantiate();
	gltf_mesh->set_mesh(ImporterMesh::from_mesh(mesh));
	const String multi_mesh_name = p_multi_mesh->get_name();
	if (!multi_mesh_name.is_empty()) {
		gltf_mesh->set_original_name(multi_mesh_name);
		gltf_mesh->set_name(p_gltf_state->generate_unique_name(multi_mesh_name));
	}
	Vector<Ref<GLTFMesh>> gltf_meshes = p_gltf_state->get_meshes();
	GLTFMeshIndex mesh_index = gltf_meshes.size();
	gltf_meshes.append(gltf_mesh);
	p_gltf_state->set_meshes(gltf_meshes);
	return mesh_index;
}

void GLTFDocumentExtensionMultiMesh::_convert_multi_mesh_to_extension(Ref<GLTFState> p_gltf_state, Ref<GLTFNode> p_gltf_node, const Ref<MultiMesh> &p_multi_mesh) {
	TypedArray<Transform3D> multi_mesh_transforms;
	const int64_t instance_count = p_multi_mesh->get_instance_count();
	multi_mesh_transforms.resize(instance_count);
	for (int64_t i = 0; i < instance_count; i++) {
		multi_mesh_transforms[i] = p_multi_mesh->get_instance_transform(i);
	}
	p_gltf_node->set_additional_data(StringName("MultiMeshTransforms"), multi_mesh_transforms);
	if (p_multi_mesh->is_using_colors()) {
		PackedColorArray multi_mesh_colors;
		multi_mesh_colors.resize(instance_count);
		for (int64_t i = 0; i < instance_count; i++) {
			multi_mesh_colors.set(i, p_multi_mesh->get_instance_color(i));
		}
		p_gltf_node->set_additional_data(StringName("MultiMeshColors"), multi_mesh_colors);
	}
	if (p_multi_mesh->is_using_custom_data()) {
		PackedColorArray multi_mesh_custom_data;
		multi_mesh_custom_data.resize(instance_count);
		for (int64_t i = 0; i < instance_count; i++) {
			multi_mesh_custom_data.set(i, p_multi_mesh->get_instance_custom_data(i));
		}
		p_gltf_node->set_additional_data(StringName("MultiMeshCustomData"), multi_mesh_custom_data);
	}
}

void GLTFDocumentExtensionMultiMesh::_convert_multi_mesh_to_gltf_nodes(Ref<GLTFState> p_gltf_state, Ref<GLTFNode> p_gltf_node, MultiMeshInstance3D *p_multi_mesh_node, GLTFMeshIndex p_mesh_index, bool p_for_fallback) {
	// Ensure the base node is appended to the state before we append children after it.
	const Vector<Ref<GLTFNode>> gltf_nodes = p_gltf_state->get_nodes();
	GLTFNodeIndex self_index = gltf_nodes.find(p_gltf_node);
	if (self_index == -1) {
		Node *parent_godot_node = p_multi_mesh_node->get_parent();
		const GLTFNodeIndex parent_index = p_gltf_state->get_node_index(parent_godot_node);
		self_index = p_gltf_state->append_gltf_node(p_gltf_node, p_multi_mesh_node, parent_index);
	}
	// These are guaranteed by the only caller (`convert_scene_node`) to not be null.
	const Ref<MultiMesh> &multi_mesh = p_multi_mesh_node->get_multimesh();
	PackedInt32Array children = p_gltf_node->get_children();
	for (int32_t instance_i = 0; instance_i < multi_mesh->get_instance_count(); instance_i++) {
		const Transform3D instance_transform = multi_mesh->get_instance_transform(instance_i);
		if (p_for_fallback && instance_transform.is_equal_approx(Transform3D())) {
			// If this is a glTF node intended as a fallback, and this instance has an identity transform,
			// then the fallback is already represented by the base node, so skip generating a duplicate.
			continue;
		}
		Ref<GLTFNode> new_gltf_node;
		new_gltf_node.instantiate();
		new_gltf_node->set_mesh(p_mesh_index);
		new_gltf_node->set_xform(instance_transform);
		new_gltf_node->set_original_name(p_multi_mesh_node->get_name());
		new_gltf_node->set_name(p_gltf_state->generate_unique_name(p_multi_mesh_node->get_name()));
		if (p_for_fallback) {
			// Mark this fallback node as a multimesh node with zero instances, effectively
			// hiding it for implementations that support EXT_mesh_gpu_instancing.
			TypedArray<Transform3D> transforms;
			new_gltf_node->set_additional_data(StringName("MultiMeshTransforms"), transforms);
		}
		const GLTFNodeIndex new_index = p_gltf_state->append_gltf_node(new_gltf_node, p_multi_mesh_node, self_index);
		children.append(new_index);
	}
	p_gltf_node->set_children(children);
}

void GLTFDocumentExtensionMultiMesh::convert_scene_node(Ref<GLTFState> p_gltf_state, Ref<GLTFNode> p_gltf_node, Node *p_scene_node) {
	MultiMeshInstance3D *multi_mesh_node = Object::cast_to<MultiMeshInstance3D>(p_scene_node);
	if (multi_mesh_node == nullptr) {
		return;
	}
	const Ref<MultiMesh> multi_mesh = multi_mesh_node->get_multimesh();
	if (multi_mesh.is_null()) {
		return;
	}
	const GLTFMeshIndex mesh_index = _convert_mesh_resource_into_state(p_gltf_state, multi_mesh);
	switch (_handling) {
		case MULTI_MESH_HANDLING_OPTIONAL_EXT_MESH_GPU_INSTANCING:
		case MULTI_MESH_HANDLING_REQUIRED_EXT_MESH_GPU_INSTANCING: {
			p_gltf_node->set_mesh(mesh_index);
			_convert_multi_mesh_to_extension(p_gltf_state, p_gltf_node, multi_mesh);
		} break;
		case MULTI_MESH_HANDLING_MULTIPLE_NODES_ONLY: {
			_convert_multi_mesh_to_gltf_nodes(p_gltf_state, p_gltf_node, multi_mesh_node, mesh_index, false);
		} break;
		case MULTI_MESH_HANDLING_MULTIPLE_NODES_FALLBACK: {
			p_gltf_node->set_mesh(mesh_index);
			_convert_multi_mesh_to_extension(p_gltf_state, p_gltf_node, multi_mesh);
			_convert_multi_mesh_to_gltf_nodes(p_gltf_state, p_gltf_node, multi_mesh_node, mesh_index, true);
		} break;
	}
}

Error GLTFDocumentExtensionMultiMesh::export_node(Ref<GLTFState> p_gltf_state, Ref<GLTFNode> p_gltf_node, Dictionary &r_node_json, Node *p_scene_node) {
	if (!p_gltf_node->has_additional_data(StringName("MultiMeshTransforms"))) {
		return OK;
	}
	const TypedArray<Transform3D> multi_mesh_transforms = TypedArray<Transform3D>(p_gltf_node->get_additional_data(StringName("MultiMeshTransforms")));
	const PackedColorArray multi_mesh_colors = p_gltf_node->has_additional_data(StringName("MultiMeshColors")) ? PackedColorArray(p_gltf_node->get_additional_data(StringName("MultiMeshColors"))) : PackedColorArray();
	const PackedColorArray multi_mesh_custom_data = p_gltf_node->has_additional_data(StringName("MultiMeshCustomData")) ? PackedColorArray(p_gltf_node->get_additional_data(StringName("MultiMeshCustomData"))) : PackedColorArray();
	PackedVector3Array positions;
	Vector<Quaternion> rotations;
	PackedVector3Array scales;
	const int64_t transform_count = multi_mesh_transforms.size();
	for (int64_t i = 0; i < multi_mesh_transforms.size(); i++) {
		const Transform3D transform = multi_mesh_transforms[i];
		if (!transform.origin.is_zero_approx()) {
			if (positions.size() < transform_count) {
				positions.resize(transform_count);
			}
			positions.set(i, transform.origin);
		}
		const Quaternion rotation = transform.basis.get_rotation_quaternion();
		if (!rotation.is_equal_approx(Quaternion())) {
			if (rotations.size() < transform_count) {
				rotations.resize(transform_count);
			}
			rotations.set(i, rotation);
		}
		const Vector3 scale = transform.basis.get_scale();
		if (!scale.is_equal_approx(Vector3(1, 1, 1))) {
			const int64_t old_scale_count = scales.size();
			if (old_scale_count < transform_count) {
				scales.resize(transform_count);
				for (int64_t scale_i = old_scale_count; scale_i < transform_count; scale_i++) {
					scales.set(scale_i, Vector3(1, 1, 1));
				}
			}
			scales.set(i, scale);
		}
	}
	Dictionary multi_mesh_attributes;
	if (!positions.is_empty()) {
		const GLTFAccessorIndex position_index = GLTFAccessor::encode_new_accessor_from_vector3s(p_gltf_state, positions);
		multi_mesh_attributes["TRANSLATION"] = position_index;
	}
	if (!rotations.is_empty()) {
		const GLTFAccessorIndex rotation_index = GLTFAccessor::encode_new_accessor_from_quaternions(p_gltf_state, rotations);
		multi_mesh_attributes["ROTATION"] = rotation_index;
	}
	if (!scales.is_empty()) {
		const GLTFAccessorIndex scale_index = GLTFAccessor::encode_new_accessor_from_vector3s(p_gltf_state, scales);
		multi_mesh_attributes["SCALE"] = scale_index;
	}
	if (!multi_mesh_colors.is_empty()) {
		const GLTFAccessorIndex color_index = GLTFAccessor::encode_new_accessor_from_colors(p_gltf_state, multi_mesh_colors);
		multi_mesh_attributes["_COLOR"] = color_index;
	}
	if (!multi_mesh_custom_data.is_empty()) {
		const GLTFAccessorIndex custom_index = GLTFAccessor::encode_new_accessor_from_colors(p_gltf_state, multi_mesh_custom_data);
		multi_mesh_attributes["_CUSTOM"] = custom_index;
	}
	Dictionary ext_mesh_gpu;
	if (!multi_mesh_attributes.is_empty()) {
		ext_mesh_gpu["attributes"] = multi_mesh_attributes;
	}
	if (unlikely(!r_node_json.has("extensions"))) {
		r_node_json["extensions"] = Dictionary();
	}
	Dictionary node_extensions = r_node_json["extensions"];
	node_extensions["EXT_mesh_gpu_instancing"] = ext_mesh_gpu;
	const bool is_required = _handling == MULTI_MESH_HANDLING_REQUIRED_EXT_MESH_GPU_INSTANCING;
	p_gltf_state->add_used_extension("EXT_mesh_gpu_instancing", is_required);
	return OK;
}

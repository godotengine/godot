/**************************************************************************/
/*  gltf_document_extension_multi_mesh.h                                  */
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

#pragma once

#include "gltf_document_extension.h"

#include "scene/3d/multimesh_instance_3d.h"

class GLTFDocumentExtensionMultiMesh : public GLTFDocumentExtension {
	GDCLASS(GLTFDocumentExtensionMultiMesh, GLTFDocumentExtension);

	enum MultiMeshHandling {
		MULTI_MESH_HANDLING_OPTIONAL_EXT_MESH_GPU_INSTANCING,
		MULTI_MESH_HANDLING_REQUIRED_EXT_MESH_GPU_INSTANCING,
		MULTI_MESH_HANDLING_MULTIPLE_NODES_ONLY,
		MULTI_MESH_HANDLING_MULTIPLE_NODES_FALLBACK,
	};
	MultiMeshHandling _handling = MULTI_MESH_HANDLING_OPTIONAL_EXT_MESH_GPU_INSTANCING;
	bool _show_handling_property = true;

	static bool _any_multi_mesh_instance_exists_recursive(Node *p_node);
	static GLTFMeshIndex _convert_mesh_resource_into_state(Ref<GLTFState> p_gltf_state, const Ref<MultiMesh> &p_multi_mesh);
	static void _convert_multi_mesh_to_extension(Ref<GLTFState> p_gltf_state, Ref<GLTFNode> p_gltf_node, const Ref<MultiMesh> &p_multi_mesh);
	static void _convert_multi_mesh_to_gltf_nodes(Ref<GLTFState> p_gltf_state, Ref<GLTFNode> p_gltf_node, MultiMeshInstance3D *p_multi_mesh_node, GLTFMeshIndex p_mesh_index, bool p_for_fallback);
	static Ref<Mesh> _generate_mesh(Ref<GLTFState> p_gltf_state, GLTFMeshIndex p_mesh_index);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	// Import process.
	virtual Error import_preflight(Ref<GLTFState> p_gltf_state, const Vector<String> &p_extensions) override;
	virtual Vector<String> get_supported_extensions() override;
	virtual Error parse_node_extensions(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, const Dictionary &p_extensions) override;
	virtual Node3D *generate_scene_node(Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node, Node *p_scene_parent) override;
	// Export process.
	virtual Error export_configure_for_scene(Node *p_root_node) override;
	virtual void convert_scene_node(Ref<GLTFState> p_gltf_state, Ref<GLTFNode> p_gltf_node, Node *p_scene_node) override;
	virtual Error export_node(Ref<GLTFState> p_gltf_state, Ref<GLTFNode> p_gltf_node, Dictionary &r_node_json, Node *p_scene_node) override;
};

VARIANT_ENUM_CAST(GLTFDocumentExtensionMultiMesh::MultiMeshHandling);

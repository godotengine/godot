/**************************************************************************/
/*  resource_importer_gltf_single_mesh.cpp                                */
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

#include "resource_importer_gltf_single_mesh.h"

#include "../gltf_document.h"

#include "editor/settings/editor_settings.h"

void ResourceImporterGLTFSingleMesh::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("glb");
	p_extensions->push_back("gltf");
}

void ResourceImporterGLTFSingleMesh::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "deduplicate_surfaces"), true));
}

Error ResourceImporterGLTFSingleMesh::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	Ref<GLTFDocument> gltf_document;
	gltf_document.instantiate();
	Ref<GLTFState> gltf_state;
	gltf_state.instantiate();
	Error err = gltf_document->append_from_file(p_source_file, gltf_state);
	if (err != OK) {
		return err;
	}
	const Vector<Ref<GLTFMesh>> &gltf_meshes = gltf_state->get_meshes();
	const int gltf_mesh_count = gltf_meshes.size();
	ERR_FAIL_COND_V_MSG(gltf_mesh_count == 0, ERR_INVALID_DATA, "Cannot import GLTF file " + p_source_file + " as a single mesh, because it contains no meshes.");
	const String save_file_path = p_save_path + ".res";
	int flags = 0;
	if (EditorSettings::get_singleton() && EDITOR_GET("filesystem/on_save/compress_binary_resources")) {
		flags |= ResourceSaver::FLAG_COMPRESS;
	}
	// If there is just one mesh, we can save it directly, preserving that one mesh exactly as it is.
	if (gltf_mesh_count == 1) {
		Ref<GLTFMesh> gltf_mesh = gltf_meshes[0];
		ERR_FAIL_COND_V_MSG(gltf_mesh.is_null(), ERR_INVALID_DATA, "GLTF mesh at index 0 was null when importing " + p_source_file);
		Ref<ImporterMesh> importer_mesh = gltf_mesh->get_mesh();
		ERR_FAIL_COND_V_MSG(importer_mesh.is_null(), ERR_INVALID_DATA, "Importer mesh at index 0 was null when importing " + p_source_file);
		Ref<ArrayMesh> array_mesh = importer_mesh->get_mesh();
		ERR_FAIL_COND_V_MSG(array_mesh.is_null(), ERR_INVALID_DATA, "Array mesh at index 0 was null when importing " + p_source_file);
		array_mesh->set_path(save_file_path, true);
		return ResourceSaver::save(array_mesh, save_file_path, flags);
	}
	// If the file contains multiple meshes, we have to merge them
	// into a single mesh, based on their positions in the scene.
	const Vector<Ref<GLTFNode>> &gltf_nodes = gltf_state->get_nodes();
	const int gltf_node_count = gltf_nodes.size();
	TypedArray<ArrayMesh> array_mesh_instances;
	TypedArray<Transform3D> array_mesh_transforms;
	for (int i = 0; i < gltf_node_count; i++) {
		Ref<GLTFNode> gltf_node = gltf_nodes[i];
		ERR_FAIL_COND_V_MSG(gltf_node.is_null(), ERR_INVALID_DATA, "GLTF node at index " + itos(i) + " was null when importing " + p_source_file);
		const GLTFMeshIndex mesh_index = gltf_node->get_mesh();
		if (mesh_index == -1) {
			continue; // No mesh for this node, skip it.
		}
		ERR_FAIL_INDEX_V_MSG(mesh_index, gltf_mesh_count, ERR_INVALID_DATA, "GLTF node at index " + itos(i) + " references mesh at index " + itos(mesh_index) + ", which is out of bounds of the mesh array size " + itos(gltf_mesh_count) + " when importing " + p_source_file);
		Ref<GLTFMesh> gltf_mesh = gltf_meshes[mesh_index];
		ERR_FAIL_COND_V_MSG(gltf_mesh.is_null(), ERR_INVALID_DATA, "GLTF mesh at index " + itos(mesh_index) + " was null when importing " + p_source_file);
		Ref<ImporterMesh> importer_mesh = gltf_mesh->get_mesh();
		ERR_FAIL_COND_V_MSG(importer_mesh.is_null(), ERR_INVALID_DATA, "Importer mesh at index " + itos(mesh_index) + " was null when importing " + p_source_file);
		Ref<ArrayMesh> array_mesh = importer_mesh->get_mesh();
		ERR_FAIL_COND_V_MSG(array_mesh.is_null(), ERR_INVALID_DATA, "Array mesh at index " + itos(mesh_index) + " was null when importing " + p_source_file);
		const Transform3D global_transform = gltf_node->get_global_transform(gltf_state);
		array_mesh_instances.append(array_mesh);
		array_mesh_transforms.append(global_transform);
	}
	const bool deduplicate_surfaces = p_options.has("deduplicate_surfaces") && p_options["deduplicate_surfaces"];
	Ref<ArrayMesh> merged_mesh = ArrayMesh::merge_array_meshes(array_mesh_instances, array_mesh_transforms, deduplicate_surfaces);
	return ResourceSaver::save(merged_mesh, save_file_path, flags);
}

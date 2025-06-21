/**************************************************************************/
/*  resource_importer_gltf_mesh_library.cpp                               */
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

#include "resource_importer_gltf_mesh_library.h"

#include "../gltf_document.h"

#include "editor/settings/editor_settings.h"
#include "scene/resources/3d/mesh_library.h"

void ResourceImporterGLTFMeshLibrary::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("glb");
	p_extensions->push_back("gltf");
}

void ResourceImporterGLTFMeshLibrary::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "save_meshes_to_files"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "use_node_names_as_mesh_names"), false));
}

Error ResourceImporterGLTFMeshLibrary::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
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
	const bool use_node_names_as_mesh_names = p_options.has("use_node_names_as_mesh_names") && p_options["use_node_names_as_mesh_names"];
	if (use_node_names_as_mesh_names) {
		const Vector<Ref<GLTFNode>> &gltf_nodes = gltf_state->get_nodes();
		for (int i = 0; i < gltf_nodes.size(); i++) {
			Ref<GLTFNode> gltf_node = gltf_nodes[i];
			ERR_CONTINUE_MSG(gltf_node.is_null(), "GLTF node at index " + itos(i) + " was null when importing " + p_source_file);
			const GLTFMeshIndex mesh_index = gltf_node->get_mesh();
			if (mesh_index != -1) {
				ERR_FAIL_INDEX_V_MSG(mesh_index, gltf_mesh_count, ERR_INVALID_DATA, "GLTF node at index " + itos(i) + " references mesh at index " + itos(mesh_index) + ", which is out of bounds of the mesh array size " + itos(gltf_mesh_count) + " when importing " + p_source_file);
				const String node_name = gltf_node->get_name();
				Ref<GLTFMesh> gltf_mesh = gltf_meshes[mesh_index];
				ERR_CONTINUE_MSG(gltf_mesh.is_null(), "GLTF mesh at index " + itos(mesh_index) + " was null when importing " + p_source_file);
				gltf_mesh->set_name(node_name);
				Ref<ImporterMesh> importer_mesh = gltf_mesh->get_mesh();
				ERR_CONTINUE_MSG(importer_mesh.is_null(), "Importer mesh at index " + itos(mesh_index) + " was null when importing " + p_source_file);
				importer_mesh->set_name(node_name);
				Ref<ArrayMesh> array_mesh = importer_mesh->get_mesh();
				ERR_CONTINUE_MSG(array_mesh.is_null(), "Array mesh at index " + itos(mesh_index) + " was null when importing " + p_source_file);
				array_mesh->set_name(node_name);
			}
		}
	}
	int flags = 0;
	if (EditorSettings::get_singleton() && EDITOR_GET("filesystem/on_save/compress_binary_resources")) {
		flags |= ResourceSaver::FLAG_COMPRESS;
	}
	const bool save_meshes_to_files = p_options.has("save_meshes_to_files") && p_options["save_meshes_to_files"];
	const String source_base_dir = p_source_file.get_base_dir();
	Ref<MeshLibrary> mesh_library;
	mesh_library.instantiate();
	for (int i = 0; i < gltf_mesh_count; i++) {
		Ref<GLTFMesh> gltf_mesh = gltf_meshes[i];
		ERR_CONTINUE_MSG(gltf_mesh.is_null(), "GLTF mesh at index " + itos(i) + " was null when importing " + p_source_file);
		Ref<ImporterMesh> importer_mesh = gltf_mesh->get_mesh();
		ERR_CONTINUE_MSG(importer_mesh.is_null(), "Importer mesh at index " + itos(i) + " was null when importing " + p_source_file);
		Ref<ArrayMesh> array_mesh = importer_mesh->get_mesh();
		ERR_CONTINUE_MSG(array_mesh.is_null(), "Array mesh at index " + itos(i) + " was null when importing " + p_source_file);
		// The glTF importer guarantees mesh names to be unique and
		// non-empty, so we can use it safely without fallback.
		const String mesh_name = array_mesh->get_name();
		if (save_meshes_to_files) {
			const String mesh_next_to_source_file_path = source_base_dir.path_join(mesh_name + ".res");
			array_mesh->set_path(mesh_next_to_source_file_path, true);
			ResourceSaver::save(array_mesh, mesh_next_to_source_file_path, flags);
		}
		const int id = mesh_library->get_last_unused_item_id();
		mesh_library->create_item(id);
		mesh_library->set_item_name(id, mesh_name);
		mesh_library->set_item_mesh(id, array_mesh);
	}
	return ResourceSaver::save(mesh_library, p_save_path + ".res", flags);
}

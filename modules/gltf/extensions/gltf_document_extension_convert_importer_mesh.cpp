/**************************************************************************/
/*  gltf_document_extension_convert_importer_mesh.cpp                     */
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

#include "gltf_document_extension_convert_importer_mesh.h"

#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/3d/importer_mesh.h"

void GLTFDocumentExtensionConvertImporterMesh::_copy_meta(Object *p_src_object, Object *p_dst_object) {
	List<StringName> meta_list;
	p_src_object->get_meta_list(&meta_list);
	for (const StringName &meta_key : meta_list) {
		Variant meta_value = p_src_object->get_meta(meta_key);
		p_dst_object->set_meta(meta_key, meta_value);
	}
}

Error GLTFDocumentExtensionConvertImporterMesh::import_post(Ref<GLTFState> p_state, Node *p_root) {
	ERR_FAIL_NULL_V(p_root, ERR_INVALID_PARAMETER);
	ERR_FAIL_NULL_V(p_state, ERR_INVALID_PARAMETER);
	List<Node *> queue;
	queue.push_back(p_root);
	List<Node *> delete_queue;
	while (!queue.is_empty()) {
		List<Node *>::Element *E = queue.front();
		Node *node = E->get();
		ImporterMeshInstance3D *importer_mesh_3d = Object::cast_to<ImporterMeshInstance3D>(node);
		if (importer_mesh_3d) {
			Ref<ImporterMesh> mesh = importer_mesh_3d->get_mesh();
			if (mesh.is_valid()) {
				MeshInstance3D *mesh_instance_node_3d = memnew(MeshInstance3D);
				Ref<ArrayMesh> array_mesh = mesh->get_mesh();
				mesh_instance_node_3d->set_name(node->get_name());
				mesh_instance_node_3d->set_transform(importer_mesh_3d->get_transform());
				mesh_instance_node_3d->set_mesh(array_mesh);
				mesh_instance_node_3d->set_skin(importer_mesh_3d->get_skin());
				mesh_instance_node_3d->set_skeleton_path(importer_mesh_3d->get_skeleton_path());
				node->replace_by(mesh_instance_node_3d);
				_copy_meta(importer_mesh_3d, mesh_instance_node_3d);
				_copy_meta(mesh.ptr(), array_mesh.ptr());
				delete_queue.push_back(node);
				node = mesh_instance_node_3d;
			} else {
				WARN_PRINT("glTF: ImporterMeshInstance3D does not have a valid mesh. This should not happen. Continuing anyway.");
			}
		}
		int child_count = node->get_child_count();
		for (int i = 0; i < child_count; i++) {
			queue.push_back(node->get_child(i));
		}
		queue.pop_front();
	}
	while (!delete_queue.is_empty()) {
		List<Node *>::Element *E = delete_queue.front();
		Node *node = E->get();
		memdelete(node);
		delete_queue.pop_front();
	}
	return OK;
}

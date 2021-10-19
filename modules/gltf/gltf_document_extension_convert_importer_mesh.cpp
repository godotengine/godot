/*************************************************************************/
/*  gltf_document_extension_convert_importer_mesh.cpp                    */
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

#include "gltf_document_extension_convert_importer_mesh.h"
#include "core/error/error_macros.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/importer_mesh.h"

#include <cstddef>

void GLTFDocumentExtensionConvertImporterMesh::_bind_methods() {
}

Error GLTFDocumentExtensionConvertImporterMesh::import_post(Ref<GLTFDocument> p_document, Node *p_node) {
	ERR_FAIL_NULL_V(p_document, ERR_INVALID_PARAMETER);
	ERR_FAIL_NULL_V(p_node, ERR_INVALID_PARAMETER);
	List<Node *> queue;
	queue.push_back(p_node);
	List<Node *> delete_queue;
	while (!queue.is_empty()) {
		List<Node *>::Element *E = queue.front();
		Node *node = E->get();
		{
			ImporterMeshInstance3D *mesh_3d = cast_to<ImporterMeshInstance3D>(node);
			if (mesh_3d) {
				MeshInstance3D *mesh_instance_node_3d = memnew(MeshInstance3D);
				Ref<ImporterMesh> mesh = mesh_3d->get_mesh();
				if (mesh.is_valid()) {
					Ref<ArrayMesh> array_mesh = mesh->get_mesh();
					mesh_instance_node_3d->set_name(node->get_name());
					mesh_instance_node_3d->set_transform(mesh_3d->get_transform());
					mesh_instance_node_3d->set_mesh(array_mesh);
					mesh_instance_node_3d->set_skin(mesh_3d->get_skin());
					mesh_instance_node_3d->set_skeleton_path(mesh_3d->get_skeleton_path());
					node->replace_by(mesh_instance_node_3d);
					delete_queue.push_back(node);
				} else {
					memdelete(mesh_instance_node_3d);
				}
			}
		}
		int child_count = node->get_child_count();
		for (int i = 0; i < child_count; i++) {
			queue.push_back(node->get_child(i));
		}
		queue.pop_front();
	}
	while (!queue.is_empty()) {
		List<Node *>::Element *E = delete_queue.front();
		Node *node = E->get();
		memdelete(node);
		delete_queue.pop_front();
	}
	return OK;
}

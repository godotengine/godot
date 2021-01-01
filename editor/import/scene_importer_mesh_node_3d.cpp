/*************************************************************************/
/*  scene_importer_mesh_node_3d.cpp                                      */
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

#include "scene_importer_mesh_node_3d.h"

void EditorSceneImporterMeshNode3D::set_mesh(const Ref<EditorSceneImporterMesh> &p_mesh) {
	mesh = p_mesh;
}
Ref<EditorSceneImporterMesh> EditorSceneImporterMeshNode3D::get_mesh() const {
	return mesh;
}

void EditorSceneImporterMeshNode3D::set_skin(const Ref<Skin> &p_skin) {
	skin = p_skin;
}
Ref<Skin> EditorSceneImporterMeshNode3D::get_skin() const {
	return skin;
}

void EditorSceneImporterMeshNode3D::set_surface_material(int p_idx, const Ref<Material> &p_material) {
	ERR_FAIL_COND(p_idx < 0);
	if (p_idx >= surface_materials.size()) {
		surface_materials.resize(p_idx + 1);
	}

	surface_materials.write[p_idx] = p_material;
}
Ref<Material> EditorSceneImporterMeshNode3D::get_surface_material(int p_idx) const {
	ERR_FAIL_COND_V(p_idx < 0, Ref<Material>());
	if (p_idx >= surface_materials.size()) {
		return Ref<Material>();
	}
	return surface_materials[p_idx];
}

void EditorSceneImporterMeshNode3D::set_skeleton_path(const NodePath &p_path) {
	skeleton_path = p_path;
}
NodePath EditorSceneImporterMeshNode3D::get_skeleton_path() const {
	return skeleton_path;
}

void EditorSceneImporterMeshNode3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &EditorSceneImporterMeshNode3D::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &EditorSceneImporterMeshNode3D::get_mesh);

	ClassDB::bind_method(D_METHOD("set_skin", "skin"), &EditorSceneImporterMeshNode3D::set_skin);
	ClassDB::bind_method(D_METHOD("get_skin"), &EditorSceneImporterMeshNode3D::get_skin);

	ClassDB::bind_method(D_METHOD("set_skeleton_path", "skeleton_path"), &EditorSceneImporterMeshNode3D::set_skeleton_path);
	ClassDB::bind_method(D_METHOD("get_skeleton_path"), &EditorSceneImporterMeshNode3D::get_skeleton_path);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "EditorSceneImporterMesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "skin", PROPERTY_HINT_RESOURCE_TYPE, "Skin"), "set_skin", "get_skin");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "skeleton_path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton"), "set_skeleton_path", "get_skeleton_path");
}

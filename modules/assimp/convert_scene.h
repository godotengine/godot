/*************************************************************************/
/*  convert_scene_gltf.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#pragma once

#ifndef CONVERT_SCENE_GLTF_H
#define CONVERT_SCENE_GLTF_H

#ifdef TOOLS_ENABLED
#include "core/bind/core_bind.h"
#include "core/reference.h"
#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "modules/csg/csg_shape.h"
#include "modules/gridmap/grid_map.h"
#include "scene/3d/mesh_instance.h"
#include "scene/main/node.h"
#include "thirdparty/assimp/include/assimp/matrix4x4.h"

struct aiScene;
struct aiMaterial;
struct aiMesh;
struct aiNode;
class ConvertScene : public Reference {
private:
	GDCLASS(ConvertScene, Reference);
	struct MeshInfo {
		Transform transform;
		Ref<Mesh> mesh;
		String name;
		Vector<Ref<Material> > materials;
		Node *original_node = NULL;
	};
	void _find_all_mesh_instances(Vector<MeshInstance *> &r_items, Node *p_current_node, const Node *p_owner);
	void _find_all_gridmaps(Vector<GridMap *> &r_items, Node *p_current_node, const Node *p_owner);
	void _find_all_csg_roots(Vector<CSGShape *> &r_items, Node *p_current_node, const Node *p_owner);
	void _set_assimp_materials(Ref<SpatialMaterial> &mat, aiMaterial *assimp_mat);
	void _generate_assimp_scene(Node *p_root_node, aiScene &r_scene);
	void _generate_node(Node *p_node, size_t &num_meshes, aiNode *&p_assimp_current_node, aiNode *&p_assimp_root, Vector<aiMesh *> &assimp_meshes, Vector<aiMaterial *> &assimp_materials);
	aiMatrix4x4 _convert_assimp_transform(Transform xform);

public:
	void export_gltf2(const String p_file, Node *p_root_node);
};

class ConvertScenePlugin : public EditorPlugin {

	GDCLASS(ConvertScenePlugin, EditorPlugin);

	Ref<ConvertScene> convert_gltf2;
	EditorNode *editor;
	CheckBox *file_export_lib_merge;
	EditorFileDialog *file_export_lib;

protected:
	static void _bind_methods();

public:
	void _gltf_dialog_action(String p_file);
	void convert_scene_to_gltf(Variant p_user_data);
	virtual String get_name() const;
	virtual void _notification(int notification);
	bool has_main_screen() const;

	ConvertScenePlugin(class EditorNode *p_node);
	void _gltf2_dialog_action(String p_file);
	void convert_scene_to_gltf2(Variant p_user_data);
};

#endif
#endif

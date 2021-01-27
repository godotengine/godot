/*************************************************************************/
/*  import_state.h                                                       */
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

#ifndef IMPORT_STATE_H
#define IMPORT_STATE_H

#include "fbx_mesh_data.h"
#include "tools/import_utils.h"
#include "tools/validation_tools.h"

#include "pivot_transform.h"

#include "core/core_bind.h"
#include "core/io/resource_importer.h"
#include "core/templates/vector.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/project_settings_editor.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/surface_tool.h"

#include "modules/fbx/fbx_parser/FBXDocument.h"
#include "modules/fbx/fbx_parser/FBXImportSettings.h"
#include "modules/fbx/fbx_parser/FBXMeshGeometry.h"
#include "modules/fbx/fbx_parser/FBXParser.h"
#include "modules/fbx/fbx_parser/FBXTokenizer.h"
#include "modules/fbx/fbx_parser/FBXUtil.h"

struct FBXBone;
struct FBXMeshData;
struct FBXNode;
struct FBXSkeleton;

struct ImportState {
	bool enable_material_import = true;
	bool enable_animation_import = true;
	bool is_blender_fbx = false;

	Map<StringName, Ref<Texture>> cached_image_searches;
	Map<uint64_t, Ref<Material>> cached_materials;

	String path = String();
	Node3D *root_owner = nullptr;
	Node3D *root = nullptr;
	real_t scale = 0.01;
	Ref<FBXNode> fbx_root_node = Ref<FBXNode>();
	// skeleton map - merged automatically when they are on the same x node in the tree so we can merge them automatically.
	Map<uint64_t, Ref<FBXSkeleton>> skeleton_map = Map<uint64_t, Ref<FBXSkeleton>>();

	// nodes on the same level get merged automatically.
	//Map<uint64_t, Skeleton3D *> armature_map;
	AnimationPlayer *animation_player = nullptr;

	// Generation 4 - Raw document accessing for bone/skin/joint/kLocators
	// joints are not necessarily bones but must be merged into the skeleton
	// (bone id), bone
	Map<uint64_t, Ref<FBXBone>> fbx_bone_map = Map<uint64_t, Ref<FBXBone>>(); // this is the bone name and setup information required for joints
	// this will never contain joints only bones attached to a mesh.

	// Generation 4 - Raw document for creating the nodes transforms in the scene
	// this is a list of the nodes in the scene
	// (id, node)
	List<Ref<FBXNode>> fbx_node_list = List<Ref<FBXNode>>();

	// All nodes which have been created in the scene
	// this will not contain the root node of the scene
	Map<uint64_t, Ref<FBXNode>> fbx_target_map = Map<uint64_t, Ref<FBXNode>>();

	// mesh nodes which are created in node / mesh step - used for populating skin poses in MeshSkins
	Map<uint64_t, Ref<FBXNode>> MeshNodes = Map<uint64_t, Ref<FBXNode>>();
	// mesh skin map
	Map<uint64_t, Ref<Skin>> MeshSkins = Map<uint64_t, Ref<Skin>>();

	// this is the container for the mesh weight information and eventually
	// any mesh data
	// but not the skin, just stuff important for rendering
	// skin is applied to mesh instance so not really required to be in here yet.
	// maybe later
	// fbx mesh id, FBXMeshData
	Map<uint64_t, Ref<FBXMeshData>> renderer_mesh_data = Map<uint64_t, Ref<FBXMeshData>>();
};

#endif // IMPORT_STATE_H

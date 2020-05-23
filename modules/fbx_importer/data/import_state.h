/*************************************************************************/
/*  import_state.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef EDITOR_SCENE_IMPORT_STATE_H
#define EDITOR_SCENE_IMPORT_STATE_H

#include "core/bind/core_bind.h"
#include "core/io/resource_importer.h"
#include "core/vector.h"
#include "data/fbx_mesh_data.h"
#include "data/pivot_transform.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/project_settings_editor.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/surface_tool.h"
#include "tools/import_utils.h"

#include <assimp/matrix4x4.h>
#include <assimp/types.h>
#include <code/FBX/FBXDocument.h>
#include <code/FBX/FBXImportSettings.h>
#include <code/FBX/FBXMeshGeometry.h>
#include <code/FBX/FBXParser.h>
#include <code/FBX/FBXTokenizer.h>
#include <code/FBX/FBXUtil.h>

struct FBXBone;
struct FBXMeshVertexData;
struct FBXNode;
struct FBXSkeleton;

struct ImportState {
	String path;
	Spatial *root;
	Ref<FBXNode> fbx_root_node;
	// skeleton map - merged automatically when they are on the same x node in the tree so we can merge them automatically.
	Map<uint64_t, Ref<FBXSkeleton> > skeleton_map;

	// nodes on the same level get merged automatically.
	//Map<uint64_t, Skeleton *> armature_map;
	AnimationPlayer *animation_player;

	// Generation 4 - Raw document accessing for bone/skin/joint/kLocators
	// joints are not necessarily bones but must be merged into the skeleton
	// (bone id), bone
	Map<uint64_t, Ref<FBXBone> > fbx_bone_map; // this is the bone name and setup information required for joints
	// this will never contain joints only bones attached to a mesh.

	// Generation 4 - Raw document for creating the nodes transforms in the scene
	// this is a list of the nodes in the scene
	// (id, node)
	List<Ref<FBXNode> > fbx_node_list;

	// All nodes which have been created in the scene
	// this will not contain the root node of the scene
	Map<uint64_t, Ref<FBXNode> > fbx_target_map;

	// mesh nodes which are created in node / mesh step - used for populating skin poses in MeshSkins
	Map<uint64_t, Ref<FBXNode> > MeshNodes;
	// mesh skin map
	Map<uint64_t, Ref<Skin> > MeshSkins;

	// this is the container for the mesh weight information and eventually
	// any mesh data
	// but not the skin, just stuff important for rendering
	// skin is applied to mesh instance so not really required to be in here yet.
	// maybe later
	// fbx mesh id, FBXMeshData
	Map<uint64_t, Ref<FBXMeshVertexData> > renderer_mesh_data;
};

#endif // EDITOR_SCENE_IMPORT_STATE_H
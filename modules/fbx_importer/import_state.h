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
#include "editor/import/resource_importer_scene.h"
#include "editor/project_settings_editor.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/surface_tool.h"

#include <assimp/matrix4x4.h>
#include <assimp/types.h>
#include <code/FBX/FBXDocument.h>
#include <code/FBX/FBXImportSettings.h>
#include <code/FBX/FBXParser.h>
#include <code/FBX/FBXTokenizer.h>
#include <code/FBX/FBXUtil.h>

namespace AssimpImporter {
/** Import state is for global scene import data
 * This makes the code simpler and contains useful lookups.
 */

struct FBXNode : Reference {
	uint64_t current_node_id;
	String node_name;
	Transform transform;
	Transform geometric_transform;
	Spatial *godot_node = nullptr;
	Ref<FBXNode> fbx_parent; // parent node

	// mutable lets us write to the const pointer
	void set_model(const Assimp::FBX::Model *p_model) {
		fbx_model = p_model;
	}
	mutable const Assimp::FBX::Model *fbx_model = nullptr;
};

// struct FBXMesh : Reference {
// 	Ref<Mesh> godot_mesh;
// 	Vector<Vector3> verticies; // Verticies
// 	Vector<uint64_t> polygon_indicies; // PolygonVertexIndex
// 	Vector<uint64_t> edge_indicies;
// 	Vector<Vector3> normals;
// 	Vector<Vector2> uv_0, uv_1;
// 	Vector<uint64_t> uv_index_0, uv_index_1;
// 	Vector<Vector3> color_0, color_1;
// 	Vector<uint64_t> color_index_0, color_index_1;
// 	Vector<uint64_t> material_index_0; // can we possibly use something else here, we theoretically can have 'n' mappings in some cases, would be nice to handle this properly.
// };

struct FBXBone : Reference {
	uint64_t parent_bone_id;
	bool valid_parent = false; // if the parent bone id is set up.
	String bone_name; // bone name
	Transform rest_pose; // inverse bind matrix
	Transform skin_pose; // skin bind matrix
	bool valid_skin_pose = false; // skin pose valid (not applicable to joints)
	uint64_t target_node_id; // the node target id for the skeleton element
	bool valid_target = false; // only applies to bones with a mesh / in the skin.

	// Godot specific data
	int godot_bone_id = -2; // godot internal bone id assigned after import

	// if a bone / armature is the root then FBX skeleton will contain the bone not any other skeleton.
	// this is to support joints by themselves in scenes
	bool valid_armature_id = false;
	uint64_t armature_id = 0;

	// // Vertex Weight information
	// Map<unsigned int, float> VertexWeightInfo;
};

struct FBXSplitBySurfaceVertexMapping {
	Vector<size_t> vertex_id;
};

struct VertexMapping : Reference {
	Vector<float> weights;
	Vector<Ref<FBXBone> > bones;

	/*** Will only add a vertex weight if it has been validated that it exists in godot **/
	void GetValidatedBoneWeightInfo(Vector<int> &out_bones, Vector<float> &out_weights) {
		ERR_FAIL_COND_MSG(bones.size() != weights.size(), "[doc] error unable to handle incorrect bone weight info");
		ERR_FAIL_COND_MSG(out_bones.size() > 0 && out_weights.size() > 0, "[doc] error input must be empty before using this function, accidental re-use?");
		for (int idx = 0; idx < weights.size(); idx++) {
			Ref<FBXBone> bone = bones[idx];
			float weight = weights[idx];
			if (bone.is_valid()) {
				out_bones.push_back(bone->godot_bone_id);
				out_weights.push_back(weight);
				print_verbose("[" + itos(idx) + "] valid bone weight: " + itos(bone->godot_bone_id) + " weight: " + rtos(weight));
			} else {
				out_bones.push_back(0);
				out_weights.push_back(0);
				if (bone.is_valid()) {
					ERR_PRINT("skeleton misconfigured");
				} else {
					print_verbose("[" + itos(idx) + "] fake weight: 0");
				}
			}
		}
	}
};

struct FBXMeshVertexData : Reference {
	// vertex id, Weight Info
	// later: perf we can use array here
	Map<size_t, Ref<VertexMapping> > vertex_weights;

	// verticies could go here
	// uvs could go here
	// normals could go here

	/* mesh maximum weight count */
	bool valid_weight_count = false;
	int max_weight_count = 0;
	uint64_t mesh_id; // fbx mesh id
};

struct ImportState {
	String path;
	Spatial *root;
	Ref<FBXNode> fbx_root_node;

	Skeleton *skeleton = NULL;

	AnimationPlayer *animation_player;

	// Generation 4 - Raw document accessing for bone/skin/joint/kLocators
	// joints are not neccisarily bones but must be merged into the skeleton
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

	// this is the container for the mesh skin data
	// (mesh id), list( skin poses )
	// yes insertion is slightly harder
	// but this is a trivial cost
	Map<uint64_t, List<Ref<FBXBone> > > skin_bone_map; // this is the true inverse bind matrix container
	Map<uint64_t, Ref<Skin> > MeshSkins;

	// this is the container for the mesh weight information and eventually
	// any mesh data
	// but not the skin, just stuff important for rendering
	// skin is applied to mesh instance so not really required to be in here yet.
	// maybe later
	// fbx mesh id, FBXMeshData
	Map<uint64_t, Ref<FBXMeshVertexData> > renderer_mesh_data;
};

struct AssimpImageData {
	Ref<Image> raw_image;
	Ref<ImageTexture> texture;
	//aiTextureMapMode map_mode[2];
};

} // namespace AssimpImporter

#endif // EDITOR_SCENE_IMPORT_STATE_H
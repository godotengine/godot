/*************************************************************************/
/*  fbx_bone.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef FBX_BONE_H
#define FBX_BONE_H

#include "fbx_node.h"
#include "import_state.h"

#include "fbx_parser/FBXDocument.h"

struct PivotTransform;

struct FBXBone : public Reference {
	uint64_t parent_bone_id = 0;
	uint64_t bone_id = 0;

	bool valid_parent = false; // if the parent bone id is set up.
	String bone_name = String(); // bone name

	bool is_root_bone() const {
		return !valid_parent;
	}

	uint64_t target_node_id; // the node target id for the skeleton element
	bool valid_target = false; // only applies to bones with a mesh / in the skin.

	// Godot specific data
	int godot_bone_id = -2; // godot internal bone id assigned after import

	// if a bone / armature is the root then FBX skeleton will contain the bone not any other skeleton.
	// this is to support joints by themselves in scenes
	bool valid_armature_id = false;
	uint64_t armature_id = 0;

	// Vertex Weight information
	Transform transform_link; // todo remove
	Transform transform_matrix; // todo remove

	/* get associate model - the model can be invalid sometimes */
	Ref<FBXBone> get_associate_model() const {
		return parent_bone;
	}

	/* link node is the parent bone */
	Ref<FBXNode> get_link(const ImportState &state) const;
	Transform get_vertex_skin_xform(const ImportState &state, Transform mesh_global_position, bool &valid);
	Transform vertex_transform_matrix;
	Transform local_cluster_matrix; // set_bone_pose

	mutable const FBXDocParser::Deformer *skin = nullptr;
	mutable const FBXDocParser::Cluster *cluster = nullptr;
	mutable const FBXDocParser::Geometry *geometry = nullptr;
	mutable const FBXDocParser::ModelLimbNode *limb_node = nullptr;

	void set_pivot_xform(Ref<PivotTransform> p_pivot_xform) {
		pivot_xform = p_pivot_xform;
	}

	// pose node / if assigned
	Transform pose_node = Transform();
	bool assigned_pose_node = false;
	Ref<FBXBone> parent_bone = Ref<FBXBone>();
	Ref<PivotTransform> pivot_xform = Ref<PivotTransform>();
	Ref<FBXSkeleton> fbx_skeleton = Ref<FBXSkeleton>();
};

#endif // FBX_BONE_H

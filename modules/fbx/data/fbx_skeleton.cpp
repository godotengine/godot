/*************************************************************************/
/*  fbx_skeleton.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "fbx_skeleton.h"

#include "import_state.h"

#include "tools/import_utils.h"

void FBXSkeleton::init_skeleton(const ImportState &state) {
	int skeleton_bone_count = skeleton_bones.size();

	if (skeleton == nullptr && skeleton_bone_count > 0) {
		skeleton = memnew(Skeleton3D);

		if (fbx_node.is_valid()) {
			// cache skeleton attachment for later during node creation
			// can't be done until after node hierarchy is built
			if (fbx_node->godot_node != state.root) {
				fbx_node->skeleton_node = Ref<FBXSkeleton>(this);
				print_verbose("cached armature skeleton attachment for node " + fbx_node->node_name);
			} else {
				// root node must never be a skeleton to prevent cyclic skeletons from being allowed (skeleton in a skeleton)
				fbx_node->godot_node->add_child(skeleton);
				skeleton->set_owner(state.root_owner);
				skeleton->set_name("Skeleton3D");
				print_verbose("created armature skeleton for root");
			}
		} else {
			memfree(skeleton);
			skeleton = nullptr;
			print_error("[doc] skeleton has no valid node to parent nodes to - erasing");
			skeleton_bones.clear();
			return;
		}
	}

	// Make the bone name uniques.
	for (int x = 0; x < skeleton_bone_count; x++) {
		Ref<FBXBone> bone = skeleton_bones[x];
		if (bone.is_valid()) {
			// Make sure the bone name is unique.
			const String bone_name = bone->bone_name;
			int same_name_count = 0;
			for (int y = x + 1; y < skeleton_bone_count; y++) {
				Ref<FBXBone> other_bone = skeleton_bones[y];
				if (other_bone.is_valid()) {
					if (other_bone->bone_name == bone_name) {
						same_name_count += 1;
						other_bone->bone_name += "_" + itos(same_name_count);
					}
				}
			}
		}
	}

	Map<int, Ref<FBXBone>> bone_map;
	// implement fbx cluster skin logic here this is where it goes
	int bone_count = 0;
	for (int x = 0; x < skeleton_bone_count; x++) {
		Ref<FBXBone> bone = skeleton_bones[x];
		if (bone.is_valid()) {
			skeleton->add_bone(bone->bone_name);
			bone->godot_bone_id = bone_count;
			bone->fbx_skeleton = Ref<FBXSkeleton>(this);
			bone_map.insert(bone_count, bone);
			print_verbose("added bone " + itos(bone->bone_id) + " " + bone->bone_name);
			bone_count++;
		}
	}

	ERR_FAIL_COND_MSG(skeleton->get_bone_count() != bone_count, "Not all bones got added, is the file corrupted?");

	for (const KeyValue<int, Ref<FBXBone>> &bone_element : bone_map) {
		const Ref<FBXBone> bone = bone_element.value;
		int bone_index = bone_element.key;
		print_verbose("working on bone: " + itos(bone_index) + " bone name:" + bone->bone_name);

		skeleton->set_bone_rest(bone->godot_bone_id, get_unscaled_transform(bone->node->pivot_transform->LocalTransform, state.scale));
		{
			Transform3D base_xform = bone->node->pivot_transform->LocalTransform;

			skeleton->set_bone_pose_position(bone_index, base_xform.origin);
			skeleton->set_bone_pose_rotation(bone_index, base_xform.basis.get_rotation_quaternion());
			skeleton->set_bone_pose_scale(bone_index, base_xform.basis.get_scale());
		}

		// lookup parent ID
		if (bone->valid_parent && state.fbx_bone_map.has(bone->parent_bone_id)) {
			Ref<FBXBone> parent_bone = state.fbx_bone_map[bone->parent_bone_id];
			int bone_id = skeleton->find_bone(parent_bone->bone_name);
			if (bone_id != -1) {
				skeleton->set_bone_parent(bone_index, bone_id);
			} else {
				print_error("invalid bone parent: " + parent_bone->bone_name);
			}
		} else {
			if (bone->godot_bone_id != -1) {
				skeleton->set_bone_parent(bone_index, -1); // no parent for this bone
			}
		}
	}
}

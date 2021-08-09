/*************************************************************************/
/*  editor_importer_bake_reset.cpp                                       */
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

#include "editor/import/editor_importer_bake_reset.h"

#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/math/transform_3d.h"
#include "editor/import/scene_importer_mesh_node_3d.h"
#include "resource_importer_scene.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_player.h"

// Given that an engineering team has made a reference character, one wants ten animators to create animations.
// Currently, a tech artist needs to combine the ten files into one exported gltf2 to import into Godot Engine.
// We bake the RESET animation and then set it to identity,
// so that rigs with corresponding RESET animation can have their animations transferred with ease.
//
// The original algorithm for the code was used to change skeleton bone rolls to be parent to child.
//
// Reference https://github.com/godotengine/godot-proposals/issues/2961
void BakeReset::_bake_animation_pose(Node *scene, const String &p_bake_anim) {
	Map<StringName, BakeResetRestBone> r_rest_bones;
	Vector<Node3D *> r_meshes;
	List<Node *> queue;
	queue.push_back(scene);
	while (!queue.is_empty()) {
		List<Node *>::Element *E = queue.front();
		Node *node = E->get();
		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(node);
		// Step 1: import scene with animations into the rest bones data structure.
		_fetch_reset_animation(ap, r_rest_bones, p_bake_anim);

		int child_count = node->get_child_count();
		for (int i = 0; i < child_count; i++) {
			queue.push_back(node->get_child(i));
		}
		queue.pop_front();
	}

	queue.push_back(scene);
	while (!queue.is_empty()) {
		List<Node *>::Element *E = queue.front();
		Node *node = E->get();
		EditorSceneImporterMeshNode3D *editor_mesh_3d = scene->cast_to<EditorSceneImporterMeshNode3D>(node);
		MeshInstance3D *mesh_3d = scene->cast_to<MeshInstance3D>(node);
		if (scene->cast_to<Skeleton3D>(node)) {
			Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(node);

			// Step 2: Bake the RESET animation from the RestBone to the skeleton.
			_fix_skeleton(skeleton, r_rest_bones);
		}
		if (editor_mesh_3d) {
			NodePath path = editor_mesh_3d->get_skeleton_path();
			if (!path.is_empty() && editor_mesh_3d->get_node_or_null(path) && Object::cast_to<Skeleton3D>(editor_mesh_3d->get_node_or_null(path))) {
				r_meshes.push_back(editor_mesh_3d);
			}
		} else if (mesh_3d) {
			NodePath path = mesh_3d->get_skeleton_path();
			if (!path.is_empty() && mesh_3d->get_node_or_null(path) && Object::cast_to<Skeleton3D>(mesh_3d->get_node_or_null(path))) {
				r_meshes.push_back(mesh_3d);
			}
		}
		int child_count = node->get_child_count();
		for (int i = 0; i < child_count; i++) {
			queue.push_back(node->get_child(i));
		}
		queue.pop_front();
	}

	queue.push_back(scene);
	while (!queue.is_empty()) {
		List<Node *>::Element *E = queue.front();
		Node *node = E->get();
		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(node);
		if (ap) {
			// Step 3: Key all RESET animation frames to identity.
			_align_animations(ap, r_rest_bones);
		}

		int child_count = node->get_child_count();
		for (int i = 0; i < child_count; i++) {
			queue.push_back(node->get_child(i));
		}
		queue.pop_front();
	}
}

void BakeReset::_align_animations(AnimationPlayer *p_ap, const Map<StringName, BakeResetRestBone> &r_rest_bones) {
	ERR_FAIL_NULL(p_ap);
	List<StringName> anim_names;
	p_ap->get_animation_list(&anim_names);
	for (List<StringName>::Element *anim_i = anim_names.front(); anim_i; anim_i = anim_i->next()) {
		Ref<Animation> a = p_ap->get_animation(anim_i->get());
		ERR_CONTINUE(a.is_null());
		for (const KeyValue<StringName, BakeResetRestBone> &rest_bone_i : r_rest_bones) {
			int track = a->find_track(NodePath(rest_bone_i.key));
			if (track == -1) {
				continue;
			}
			int new_track = a->add_track(Animation::TYPE_TRANSFORM3D);
			NodePath new_path = NodePath(rest_bone_i.key);
			const BakeResetRestBone rest_bone = rest_bone_i.value;
			a->track_set_path(new_track, new_path);
			for (int key_i = 0; key_i < a->track_get_key_count(track); key_i++) {
				Vector3 loc;
				Quaternion rot;
				Vector3 scale;
				Error err = a->transform_track_get_key(track, key_i, &loc, &rot, &scale);
				ERR_CONTINUE(err);
				real_t time = a->track_get_key_time(track, key_i);
				rot.normalize();
				loc = loc - rest_bone.loc;
				rot = rest_bone.rest_delta.get_rotation_quaternion().inverse() * rot;
				rot.normalize();
				scale = Vector3(1, 1, 1) - (rest_bone.rest_delta.get_scale() - scale);
				// Apply the reverse of the rest changes to make the key be close to identity transform.
				a->transform_track_insert_key(new_track, time, loc, rot, scale);
			}
			a->remove_track(track);
		}
	}
}

void BakeReset::_fetch_reset_animation(AnimationPlayer *p_ap, Map<StringName, BakeResetRestBone> &r_rest_bones, const String &p_bake_anim) {
	if (!p_ap) {
		return;
	}
	List<StringName> anim_names;
	p_ap->get_animation_list(&anim_names);
	Node *root = p_ap->get_owner();
	ERR_FAIL_NULL(root);
	if (!p_ap->has_animation(p_bake_anim)) {
		return;
	}
	Ref<Animation> a = p_ap->get_animation(p_bake_anim);
	if (a.is_null()) {
		return;
	}
	for (int32_t track = 0; track < a->get_track_count(); track++) {
		NodePath path = a->track_get_path(track);
		String string_path = path;
		Skeleton3D *skeleton = root->cast_to<Skeleton3D>(root->get_node(string_path.get_slice(":", 0)));
		if (!skeleton) {
			continue;
		}
		String bone_name = string_path.get_slice(":", 1);
		for (int key_i = 0; key_i < a->track_get_key_count(track); key_i++) {
			Vector3 loc;
			Quaternion rot;
			Vector3 scale;
			Error err = a->transform_track_get_key(track, key_i, &loc, &rot, &scale);
			if (err != OK) {
				ERR_PRINT_ONCE("Reset animation baker can't get key.");
				continue;
			}
			rot.normalize();
			Basis rot_basis = Basis(rot, scale);
			BakeResetRestBone rest_bone;
			rest_bone.rest_delta = rot_basis;
			rest_bone.loc = loc;
			// Store the animation into the RestBone.
			r_rest_bones[StringName(String(skeleton->get_owner()->get_path_to(skeleton)) + ":" + bone_name)] = rest_bone;
			break;
		}
	}
}

void BakeReset::_fix_skeleton(Skeleton3D *p_skeleton, Map<StringName, BakeReset::BakeResetRestBone> &r_rest_bones) {
	int bone_count = p_skeleton->get_bone_count();

	// First iterate through all the bones and update the RestBone.
	for (int j = 0; j < bone_count; j++) {
		StringName final_path = String(p_skeleton->get_owner()->get_path_to(p_skeleton)) + String(":") + p_skeleton->get_bone_name(j);
		BakeResetRestBone &rest_bone = r_rest_bones[final_path];
		rest_bone.rest_local = p_skeleton->get_bone_rest(j);
	}
	for (int i = 0; i < bone_count; i++) {
		int parent_bone = p_skeleton->get_bone_parent(i);
		String path = p_skeleton->get_owner()->get_path_to(p_skeleton);
		StringName final_path = String(path) + String(":") + p_skeleton->get_bone_name(parent_bone);
		if (parent_bone >= 0) {
			r_rest_bones[path].children.push_back(i);
		}
	}

	// When we apply transform to a bone, we also have to move all of its children in the opposite direction.
	for (int i = 0; i < bone_count; i++) {
		StringName final_path = String(p_skeleton->get_owner()->get_path_to(p_skeleton)) + String(":") + p_skeleton->get_bone_name(i);
		r_rest_bones[final_path].rest_local = r_rest_bones[final_path].rest_local * Transform3D(r_rest_bones[final_path].rest_delta, r_rest_bones[final_path].loc);
		// Iterate through the children and move in the opposite direction.
		for (int j = 0; j < r_rest_bones[final_path].children.size(); j++) {
			int child_index = r_rest_bones[final_path].children[j];
			StringName children_path = String(p_skeleton->get_name()) + String(":") + p_skeleton->get_bone_name(child_index);
			r_rest_bones[children_path].rest_local = Transform3D(r_rest_bones[final_path].rest_delta, r_rest_bones[final_path].loc).affine_inverse() * r_rest_bones[children_path].rest_local;
		}
	}

	for (int i = 0; i < bone_count; i++) {
		StringName final_path = String(p_skeleton->get_owner()->get_path_to(p_skeleton)) + String(":") + p_skeleton->get_bone_name(i);
		ERR_CONTINUE(!r_rest_bones.has(final_path));
		Transform3D rest_transform = r_rest_bones[final_path].rest_local;
		p_skeleton->set_bone_rest(i, rest_transform);
	}
}

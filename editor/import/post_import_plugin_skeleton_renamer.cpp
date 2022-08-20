/*************************************************************************/
/*  post_import_plugin_skeleton_renamer.cpp                              */
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

#include "post_import_plugin_skeleton_renamer.h"

#include "editor/import/scene_import_settings.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/bone_map.h"

void PostImportPluginSkeletonRenamer::get_internal_import_options(InternalImportCategory p_category, List<ResourceImporter::ImportOption> *r_options) {
	if (p_category == INTERNAL_IMPORT_CATEGORY_SKELETON_3D_NODE) {
		r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "retarget/bone_renamer/rename_bones"), true));
		r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "retarget/bone_renamer/unique_node/make_unique"), true));
		r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::STRING, "retarget/bone_renamer/unique_node/skeleton_name"), "GeneralSkeleton"));
	}
}

void PostImportPluginSkeletonRenamer::internal_process(InternalImportCategory p_category, Node *p_base_scene, Node *p_node, Ref<Resource> p_resource, const Dictionary &p_options) {
	if (p_category == INTERNAL_IMPORT_CATEGORY_SKELETON_3D_NODE) {
		// Prepare objects.
		Object *map = p_options["retarget/bone_map"].get_validated_object();
		if (!map || !bool(p_options["retarget/bone_renamer/rename_bones"])) {
			return;
		}
		BoneMap *bone_map = Object::cast_to<BoneMap>(map);
		Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(p_node);

		// Rename bones in Skeleton3D.
		{
			int len = skeleton->get_bone_count();
			for (int i = 0; i < len; i++) {
				StringName bn = bone_map->find_profile_bone_name(skeleton->get_bone_name(i));
				if (bn) {
					skeleton->set_bone_name(i, bn);
				}
			}
		}

		// Rename bones in Skin.
		{
			TypedArray<Node> nodes = p_base_scene->find_children("*", "ImporterMeshInstance3D");
			while (nodes.size()) {
				ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(nodes.pop_back());
				Ref<Skin> skin = mi->get_skin();
				if (skin.is_valid()) {
					Node *node = mi->get_node(mi->get_skeleton_path());
					if (node) {
						Skeleton3D *mesh_skeleton = Object::cast_to<Skeleton3D>(node);
						if (mesh_skeleton && node == skeleton) {
							int len = skin->get_bind_count();
							for (int i = 0; i < len; i++) {
								StringName bn = bone_map->find_profile_bone_name(skin->get_bind_name(i));
								if (bn) {
									skin->set_bind_name(i, bn);
								}
							}
						}
					}
				}
			}
		}

		// Rename bones in AnimationPlayer.
		{
			TypedArray<Node> nodes = p_base_scene->find_children("*", "AnimationPlayer");
			while (nodes.size()) {
				AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(nodes.pop_back());
				List<StringName> anims;
				ap->get_animation_list(&anims);
				for (const StringName &name : anims) {
					Ref<Animation> anim = ap->get_animation(name);
					int len = anim->get_track_count();
					for (int i = 0; i < len; i++) {
						if (anim->track_get_path(i).get_subname_count() != 1 || !(anim->track_get_type(i) == Animation::TYPE_POSITION_3D || anim->track_get_type(i) == Animation::TYPE_ROTATION_3D || anim->track_get_type(i) == Animation::TYPE_SCALE_3D)) {
							continue;
						}
						String track_path = String(anim->track_get_path(i).get_concatenated_names());
						Node *node = (ap->get_node(ap->get_root()))->get_node(NodePath(track_path));
						if (node) {
							Skeleton3D *track_skeleton = Object::cast_to<Skeleton3D>(node);
							if (track_skeleton && track_skeleton == skeleton) {
								StringName bn = bone_map->find_profile_bone_name(anim->track_get_path(i).get_subname(0));
								if (bn) {
									anim->track_set_path(i, track_path + ":" + bn);
								}
							}
						}
					}
				}
			}
		}

		// Rename bones in all Nodes by calling method.
		{
			Vector<Variant> vargs;
			vargs.push_back(p_base_scene);
			vargs.push_back(skeleton);
			vargs.push_back(bone_map);
			const Variant **argptrs = (const Variant **)alloca(sizeof(const Variant **) * vargs.size());
			const Variant *args = vargs.ptr();
			uint32_t argcount = vargs.size();
			for (uint32_t i = 0; i < argcount; i++) {
				argptrs[i] = &args[i];
			}

			TypedArray<Node> nodes = p_base_scene->find_children("*");
			while (nodes.size()) {
				Node *nd = Object::cast_to<Node>(nodes.pop_back());
				Callable::CallError ce;
				nd->callp("_notify_skeleton_bones_renamed", argptrs, argcount, ce);
			}
		}

		// Make unique skeleton.
		if (bool(p_options["retarget/bone_renamer/unique_node/make_unique"])) {
			String unique_name = String(p_options["retarget/bone_renamer/unique_node/skeleton_name"]);
			ERR_FAIL_COND_MSG(unique_name == String(), "Skeleton unique name cannot be empty.");

			TypedArray<Node> nodes = p_base_scene->find_children("*", "AnimationPlayer");
			while (nodes.size()) {
				AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(nodes.pop_back());
				List<StringName> anims;
				ap->get_animation_list(&anims);
				for (const StringName &name : anims) {
					Ref<Animation> anim = ap->get_animation(name);
					int track_len = anim->get_track_count();
					for (int i = 0; i < track_len; i++) {
						String track_path = String(anim->track_get_path(i).get_concatenated_names());
						Node *orig_node = (ap->get_node(ap->get_root()))->get_node(NodePath(track_path));
						Node *node = (ap->get_node(ap->get_root()))->get_node(NodePath(track_path));
						while (node) {
							Skeleton3D *track_skeleton = Object::cast_to<Skeleton3D>(node);
							if (track_skeleton && track_skeleton == skeleton) {
								if (node == orig_node) {
									if (anim->track_get_path(i).get_subname_count() > 0) {
										anim->track_set_path(i, UNIQUE_NODE_PREFIX + unique_name + String(":") + anim->track_get_path(i).get_concatenated_subnames());
									} else {
										anim->track_set_path(i, UNIQUE_NODE_PREFIX + unique_name);
									}
								} else {
									if (anim->track_get_path(i).get_subname_count() > 0) {
										anim->track_set_path(i, UNIQUE_NODE_PREFIX + unique_name + "/" + node->get_path_to(orig_node) + String(":") + anim->track_get_path(i).get_concatenated_subnames());
									} else {
										anim->track_set_path(i, UNIQUE_NODE_PREFIX + unique_name + "/" + node->get_path_to(orig_node));
									}
								}
								break;
							}
							node = node->get_parent();
						}
					}
				}
			}
			skeleton->set_name(unique_name);
			skeleton->set_unique_name_in_owner(true);
		}
	}
}

PostImportPluginSkeletonRenamer::PostImportPluginSkeletonRenamer() {
}

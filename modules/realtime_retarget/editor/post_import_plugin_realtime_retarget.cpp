/*************************************************************************/
/*  post_import_plugin_realtime_retarget.cpp                             */
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

#include "post_import_plugin_realtime_retarget.h"

#include "editor/import/3d/scene_import_settings.h"
#include "../src/retarget_animation_player.h"
#include "../src/retarget_profile.h"
#include "../src/retarget_utility.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/bone_map.h"

void PostImportPluginRealtimeRetarget::get_internal_import_options(InternalImportCategory p_category, List<ResourceImporter::ImportOption> *r_options) {
	if (p_category == INTERNAL_IMPORT_CATEGORY_SKELETON_3D_NODE) {
		r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::OBJECT, "retarget/realtime_retarget/profile", PROPERTY_HINT_RESOURCE_TYPE, "RetargetProfile"), Variant()));
	}
}

void PostImportPluginRealtimeRetarget::internal_process(InternalImportCategory p_category, Node *p_base_scene, Node *p_node, Ref<Resource> p_resource, const Dictionary &p_options) {
	if (p_category == INTERNAL_IMPORT_CATEGORY_SKELETON_3D_NODE) {
		// Prepare objects.
		Object *map = p_options["retarget/bone_map"].get_validated_object();
		if (!map) {
			return;
		}
		BoneMap *bone_map = Object::cast_to<BoneMap>(map);
		Ref<SkeletonProfile> profile = bone_map->get_profile();
		if (!profile.is_valid()) {
			return;
		}
		Skeleton3D *src_skeleton = Object::cast_to<Skeleton3D>(p_node);
		if (!src_skeleton) {
			return;
		}
		Object *prof_obj = p_options["retarget/realtime_retarget/profile"].get_validated_object();
		if (!prof_obj) {
			return;
		}
		RetargetProfile *retarget_profile = Object::cast_to<RetargetProfile>(prof_obj);

		bool is_renamed = bool(p_options["retarget/bone_renamer/rename_bones"]);

		// Fix animation.
		Vector<int> bones_to_process = src_skeleton->get_parentless_bones();
		{
			TypedArray<Node> nodes = p_base_scene->find_children("*", "AnimationPlayer");
			while (nodes.size()) {
				AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(nodes.pop_back());

				// Duplicate AnimationPlayer as RetargetAnimationPlayer.
				Node *parent = ap->get_parent();
				RetargetAnimationPlayer *rap = memnew(RetargetAnimationPlayer);
				parent->add_child(rap);
				rap->set_owner(p_base_scene);
				rap->set_name("RetargetAnimationPlayer"); // TODO: Implements _gen_unique_name() like same as gltf_document.
				Ref<AnimationLibrary> lib = memnew(AnimationLibrary);

				List<StringName> anims;
				ap->get_animation_list(&anims);
				for (const StringName &name : anims) {
					Dictionary meta_dict = Dictionary();
					Ref<Animation> anim = ap->get_animation(name);
					int track_len = anim->get_track_count();
					for (int i = 0; i < track_len; i++) {
						if (anim->track_get_path(i).get_subname_count() != 1 || !(anim->track_get_type(i) == Animation::TYPE_POSITION_3D || anim->track_get_type(i) == Animation::TYPE_ROTATION_3D || anim->track_get_type(i) == Animation::TYPE_SCALE_3D)) {
							continue;
						}

						if (anim->track_is_compressed(i)) {
							continue; // Shouldn't occur in internal_process().
						}

						String track_path = String(anim->track_get_path(i).get_concatenated_names());
						Node *node = (ap->get_node(ap->get_root_node()))->get_node(NodePath(track_path));
						ERR_CONTINUE(!node);

						Skeleton3D *track_skeleton = Object::cast_to<Skeleton3D>(node);
						if (!track_skeleton || track_skeleton != src_skeleton) {
							continue;
						}

						StringName bn = anim->track_get_path(i).get_subname(0);
						if (!bn) {
							continue;
						}

						int bone_idx = src_skeleton->find_bone(bn);
						int key_len = anim->track_get_key_count(i);
						StringName search_name = is_renamed ? bn : bone_map->find_profile_bone_name(bn);

						if (retarget_profile->has_global_transform_target(search_name)) {
							meta_dict[String(anim->track_get_path(i))] = RetargetUtility::TYPE_GLOBAL;
							if (anim->track_get_type(i) == Animation::TYPE_POSITION_3D) {
								for (int j = 0; j < key_len; j++) {
									Vector3 ps = static_cast<Vector3>(anim->track_get_key_value(i, j));
									ps = ps * src_skeleton->get_motion_scale();
									ps = RetargetUtility::extract_global_transform_position(src_skeleton, bone_idx, ps);
									ps = ps / src_skeleton->get_motion_scale();
									anim->track_set_key_value(i, j, ps);
								}
							} else if (anim->track_get_type(i) == Animation::TYPE_ROTATION_3D) {
								for (int j = 0; j < key_len; j++) {
									Quaternion qt = static_cast<Quaternion>(anim->track_get_key_value(i, j));
									qt = RetargetUtility::extract_global_transform_rotation(src_skeleton, bone_idx, qt);
									anim->track_set_key_value(i, j, qt);
								}
							} else if (anim->track_get_type(i) == Animation::TYPE_SCALE_3D) {
								for (int j = 0; j < key_len; j++) {
									Vector3 sc = static_cast<Vector3>(anim->track_get_key_value(i, j));
									sc = RetargetUtility::extract_global_transform_scale(src_skeleton, bone_idx, sc);
									anim->track_set_key_value(i, j, sc);
								}
							}
						} else if (retarget_profile->has_local_transform_target(search_name)) {
							meta_dict[String(anim->track_get_path(i))] = RetargetUtility::TYPE_LOCAL;
							if (anim->track_get_type(i) == Animation::TYPE_POSITION_3D) {
								for (int j = 0; j < key_len; j++) {
									Vector3 ps = static_cast<Vector3>(anim->track_get_key_value(i, j));
									ps = ps * src_skeleton->get_motion_scale();
									ps = RetargetUtility::extract_local_transform_position(src_skeleton, bone_idx, ps);
									ps = ps / src_skeleton->get_motion_scale();
									anim->track_set_key_value(i, j, ps);
								}
							} else if (anim->track_get_type(i) == Animation::TYPE_ROTATION_3D) {
								for (int j = 0; j < key_len; j++) {
									Quaternion qt = static_cast<Quaternion>(anim->track_get_key_value(i, j));
									qt = RetargetUtility::extract_local_transform_rotation(src_skeleton, bone_idx, qt);
									anim->track_set_key_value(i, j, qt);
								}
							} else if (anim->track_get_type(i) == Animation::TYPE_SCALE_3D) {
								for (int j = 0; j < key_len; j++) {
									Vector3 sc = static_cast<Vector3>(anim->track_get_key_value(i, j));
									sc = RetargetUtility::extract_local_transform_scale(src_skeleton, bone_idx, sc);
									anim->track_set_key_value(i, j, sc);
								}
							}
						} else {
							meta_dict[String(anim->track_get_path(i))] = RetargetUtility::TYPE_ABSOLUTE;
						}
					}
					anim->set_meta(REALTIME_RETARGET_META, meta_dict);
					rename_map.insert(anim->get_name(), "(" + retarget_profile->get_label_for_animation_name() + ")");
					lib->add_animation(anim->get_name(), anim);
				}

				rap->add_animation_library("", lib);
				parent->remove_child(ap);
			}
		}
	}
}

void PostImportPluginRealtimeRetarget::pre_process(Node *p_scene, const HashMap<StringName, Variant> &p_options) {
	rename_map.clear();
}

void PostImportPluginRealtimeRetarget::post_process(Node *p_scene, const HashMap<StringName, Variant> &p_options) {
	TypedArray<Node> nodes = p_scene->find_children("*", "RetargetAnimationPlayer");
	while (nodes.size()) {
		RetargetAnimationPlayer *rap = Object::cast_to<RetargetAnimationPlayer>(nodes.pop_back());
		Ref<AnimationLibrary> lib = rap->get_animation_library("");
		if (!lib.is_valid()) {
			continue;
		}
		List<StringName> anims;
		lib->get_animation_list(&anims);
		for (const StringName &name : anims) {
			if (rename_map.has(name)) {
				lib->rename_animation(name, String(name) + rename_map.get(name));
			}
		}
	}

	rename_map.clear();
}

PostImportPluginRealtimeRetarget::PostImportPluginRealtimeRetarget() {
	//
}

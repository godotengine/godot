/**************************************************************************/
/*  post_import_plugin_skeleton_track_organizer.cpp                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "post_import_plugin_skeleton_track_organizer.h"

#include "editor/import/scene_import_settings.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/bone_map.h"

void PostImportPluginSkeletonTrackOrganizer::get_internal_import_options(InternalImportCategory p_category, List<ResourceImporter::ImportOption> *r_options) {
	if (p_category == INTERNAL_IMPORT_CATEGORY_SKELETON_3D_NODE) {
		r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "retarget/remove_tracks/except_bone_transform"), false));
		r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "retarget/remove_tracks/unimportant_positions"), true));
		r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "retarget/remove_tracks/unmapped_bones"), false));
	}
}

void PostImportPluginSkeletonTrackOrganizer::internal_process(InternalImportCategory p_category, Node *p_base_scene, Node *p_node, Ref<Resource> p_resource, const Dictionary &p_options) {
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
		bool remove_except_bone = bool(p_options["retarget/remove_tracks/except_bone_transform"]);
		bool remove_positions = bool(p_options["retarget/remove_tracks/unimportant_positions"]);
		bool remove_unmapped_bones = bool(p_options["retarget/remove_tracks/unmapped_bones"]);

		if (!remove_positions && !remove_unmapped_bones) {
			return;
		}

		TypedArray<Node> nodes = p_base_scene->find_children("*", "AnimationPlayer");
		while (nodes.size()) {
			AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(nodes.pop_back());
			List<StringName> anims;
			ap->get_animation_list(&anims);
			for (const StringName &name : anims) {
				Ref<Animation> anim = ap->get_animation(name);
				int track_len = anim->get_track_count();
				Vector<int> remove_indices;
				for (int i = 0; i < track_len; i++) {
					String track_path = String(anim->track_get_path(i).get_concatenated_names());
					Node *node = (ap->get_node(ap->get_root_node()))->get_node(NodePath(track_path));
					if (!node) {
						if (remove_except_bone) {
							remove_indices.push_back(i);
						}
						continue;
					}
					Skeleton3D *track_skeleton = Object::cast_to<Skeleton3D>(node);
					if (track_skeleton && track_skeleton == src_skeleton) {
						if (anim->track_get_path(i).get_subname_count() != 1 || !(anim->track_get_type(i) == Animation::TYPE_POSITION_3D || anim->track_get_type(i) == Animation::TYPE_ROTATION_3D || anim->track_get_type(i) == Animation::TYPE_SCALE_3D)) {
							if (remove_except_bone) {
								remove_indices.push_back(i);
							}
							continue;
						}
						StringName bn = anim->track_get_path(i).get_subname(0);
						if (bn) {
							int prof_idx = profile->find_bone(bone_map->find_profile_bone_name(bn));
							if (remove_unmapped_bones && prof_idx < 0) {
								remove_indices.push_back(i);
								continue;
							}
							if (remove_positions && anim->track_get_type(i) == Animation::TYPE_POSITION_3D && prof_idx >= 0) {
								StringName prof_bn = profile->get_bone_name(prof_idx);
								if (prof_bn == profile->get_root_bone() || prof_bn == profile->get_scale_base_bone()) {
									continue;
								}
								remove_indices.push_back(i);
							}
						}
					}
					if (remove_except_bone) {
						remove_indices.push_back(i);
					}
				}

				remove_indices.reverse();
				for (int i = 0; i < remove_indices.size(); i++) {
					anim->remove_track(remove_indices[i]);
				}
			}
		}
	}
}

PostImportPluginSkeletonTrackOrganizer::PostImportPluginSkeletonTrackOrganizer() {
}

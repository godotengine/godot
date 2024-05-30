/*************************************************************************/
/*  retarget_animation_tree.cpp                                          */
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

#include "retarget_animation_tree.h"

#include "retarget_utility.h"

Variant RetargetAnimationTree::_post_process_key_value(const Ref<Animation> &p_anim, int p_track, Variant p_value, ObjectID p_object, int p_object_idx) {
	Animation::TrackType type = p_anim->track_get_type(p_track);
	Variant tmp_value = p_value;
	switch (type) {
#ifndef _3D_DISABLED
		case Animation::TYPE_POSITION_3D:
		case Animation::TYPE_ROTATION_3D:
		case Animation::TYPE_SCALE_3D: {
			const Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(p_object) );
			if (!skeleton || p_object_idx < 0) {
				break; // If it is not skeleton, do nothing.
			}
			// Apply motion scale.
			if (type == Animation::TYPE_POSITION_3D) {
				tmp_value = Vector3(tmp_value) * skeleton->get_motion_scale();
			}
			// Get transform type from meta.
			if (!p_anim->has_meta(REALTIME_RETARGET_META)) {
				break;
			}
			String key = String(p_anim->track_get_path(p_track));
			Dictionary dict = p_anim->get_meta(REALTIME_RETARGET_META, Dictionary());
			if (!dict.has(key)) {
				break;
			}
			RetargetUtility::TransformType ttype = (RetargetUtility::TransformType)(int)dict.get(key, RetargetUtility::TYPE_ABSOLUTE);
			if (ttype == RetargetUtility::TYPE_GLOBAL) {
				if (type == Animation::TYPE_ROTATION_3D) {
					tmp_value = RetargetUtility::global_transform_rotation_to_bone_pose(skeleton, p_object_idx, Quaternion(tmp_value));
				} else if (type == Animation::TYPE_POSITION_3D) {
					tmp_value = RetargetUtility::global_transform_position_to_bone_pose(skeleton, p_object_idx, Vector3(tmp_value));
				} else {
					tmp_value = RetargetUtility::global_transform_scale_to_bone_pose(skeleton, p_object_idx, Vector3(tmp_value));
				}
			} else if (ttype == RetargetUtility::TYPE_LOCAL) {
				if (type == Animation::TYPE_ROTATION_3D) {
					tmp_value = RetargetUtility::local_transform_rotation_to_bone_pose(skeleton, p_object_idx, Quaternion(tmp_value));
				} else if (type == Animation::TYPE_POSITION_3D) {
					tmp_value = RetargetUtility::local_transform_position_to_bone_pose(skeleton, p_object_idx, Vector3(tmp_value));
				} else {
					tmp_value = RetargetUtility::local_transform_scale_to_bone_pose(skeleton, p_object_idx, Vector3(tmp_value));
				}
			}
		} break;
#endif // _3D_DISABLED
		default: {
		} break;
	}
	return tmp_value;
}

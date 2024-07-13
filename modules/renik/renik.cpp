/**************************************************************************/
/*  renik.cpp                                                             */
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

#include "renik.h"

#include "core/math/quaternion.h"
#include "math/qcp.h"
#include "scene/3d/marker_3d.h"

#ifndef _3D_DISABLED

#define RENIK_PROPERTY_STRING_SKELETON_PATH "armature_skeleton_path"

#define RENIK_PROPERTY_STRING_HEAD_BONE "armature_head"

#define RENIK_PROPERTY_STRING_HAND_LEFT_BONE "armature_left_hand"
#define RENIK_PROPERTY_STRING_LEFT_LOWER_ARM_BONE "armature_left_lower_arm"
#define RENIK_PROPERTY_STRING_LEFT_UPPER_ARM_BONE "armature_left_upper_arm"

#define RENIK_PROPERTY_STRING_HAND_RIGHT_BONE "armature_right_hand"
#define RENIK_PROPERTY_STRING_RIGHT_LOWER_ARM_BONE "armature_right_lower_arm"
#define RENIK_PROPERTY_STRING_RIGHT_UPPER_ARM_BONE "armature_right_upper_arm"

#define RENIK_PROPERTY_STRING_HIP_BONE "armature_hip"

#define RENIK_PROPERTY_STRING_FOOT_LEFT_BONE "armature_left_foot"
#define RENIK_PROPERTY_STRING_LEFT_LOWER_LEG_BONE "armature_left_lower_leg"
#define RENIK_PROPERTY_STRING_LEFT_UPPER_LEG_BONE "armature_left_upper_leg"

#define RENIK_PROPERTY_STRING_FOOT_RIGHT_BONE "armature_right_foot"
#define RENIK_PROPERTY_STRING_RIGHT_LOWER_LEG_BONE "armature_right_lower_leg"
#define RENIK_PROPERTY_STRING_RIGHT_UPPER_LEG_BONE "armature_right_upper_leg"

#define RENIK_PROPERTY_STRING_HEAD_TARGET_PATH "armature_head_target"
#define RENIK_PROPERTY_STRING_HAND_LEFT_TARGET_PATH "armature_left_hand_target"
#define RENIK_PROPERTY_STRING_HAND_RIGHT_TARGET_PATH "armature_right_hand_target"

#define RENIK_PROPERTY_STRING_HIP_TARGET_PATH "armature_hip_target"
#define RENIK_PROPERTY_STRING_FOOT_LEFT_TARGET_PATH "armature_left_foot_target"
#define RENIK_PROPERTY_STRING_FOOT_RIGHT_TARGET_PATH "armature_right_foot_target"

RenIK::RenIK(){};

void RenIK::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_live_preview", "p_enable"),
			&RenIK::set_live_preview);
	ClassDB::bind_method(D_METHOD("get_live_preview"), &RenIK::get_live_preview);

	ClassDB::bind_method(D_METHOD("enable_solve_ik_every_frame", "p_enable"),
			&RenIK::enable_solve_ik_every_frame);
	ClassDB::bind_method(D_METHOD("enable_hip_placement", "p_enable"),
			&RenIK::enable_hip_placement);
	ClassDB::bind_method(D_METHOD("enable_foot_placement", "p_enable"),
			&RenIK::enable_foot_placement);

	// ClassDB::bind_method(D_METHOD("set_skeleton_path", "p_path"),
	// 		&RenIK::set_skeleton_path);
	// ClassDB::bind_method(D_METHOD("set_skeleton_", "p_node"),
	// 		&RenIK::set_skeleton);
	// ClassDB::bind_method(D_METHOD("get_skeleton_path"),
	// 		&RenIK::get_skeleton_path);

	ClassDB::bind_method(D_METHOD("set_head_bone_by_name", "p_bone"),
			&RenIK::set_head_bone_by_name);

	ClassDB::bind_method(D_METHOD("get_head_bone_name"),
			&RenIK::get_head_bone_name);

	ClassDB::bind_method(D_METHOD("set_hand_left_bone_by_name", "p_bone"),
			&RenIK::set_hand_left_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_hand_left_bone_name"),
			&RenIK::get_hand_left_bone_name);
	ClassDB::bind_method(D_METHOD("set_upper_arm_left_bone_by_name", "p_bone"),
			&RenIK::set_upper_arm_left_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_upper_arm_left_bone_name"),
			&RenIK::get_upper_arm_left_bone_name);
	ClassDB::bind_method(D_METHOD("set_lower_arm_left_bone_by_name", "p_bone"),
			&RenIK::set_lower_arm_left_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_lower_arm_left_bone_name"),
			&RenIK::get_lower_arm_left_bone_name);

	ClassDB::bind_method(D_METHOD("set_hand_right_bone_by_name", "p_bone"),
			&RenIK::set_hand_right_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_hand_right_bone_name"),
			&RenIK::get_hand_right_bone_name);
	ClassDB::bind_method(D_METHOD("set_upper_arm_right_bone_by_name", "p_bone"),
			&RenIK::set_upper_arm_right_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_upper_arm_right_bone_name"),
			&RenIK::get_upper_arm_right_bone_name);
	ClassDB::bind_method(D_METHOD("set_lower_arm_right_bone_by_name", "p_bone"),
			&RenIK::set_lower_arm_right_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_lower_arm_right_bone_name"),
			&RenIK::get_lower_arm_right_bone_name);

	ClassDB::bind_method(D_METHOD("set_hip_bone_by_name", "p_bone"),
			&RenIK::set_hip_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_hip_bone_name"),
			&RenIK::get_hip_bone_name);

	ClassDB::bind_method(D_METHOD("set_foot_left_bone_by_name", "p_bone"),
			&RenIK::set_foot_left_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_foot_left_bone_name"),
			&RenIK::get_foot_left_bone_name);
	ClassDB::bind_method(D_METHOD("set_upper_leg_left_bone_by_name", "p_bone"),
			&RenIK::set_upper_leg_left_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_upper_leg_left_bone_name"),
			&RenIK::get_upper_leg_left_bone_name);
	ClassDB::bind_method(D_METHOD("set_lower_leg_left_bone_by_name", "p_bone"),
			&RenIK::set_lower_leg_left_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_lower_leg_left_bone_name"),
			&RenIK::get_lower_leg_left_bone_name);

	ClassDB::bind_method(D_METHOD("set_foot_right_bone_by_name", "p_bone"),
			&RenIK::set_foot_right_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_foot_right_bone_name"),
			&RenIK::get_foot_right_bone_name);
	ClassDB::bind_method(D_METHOD("set_upper_leg_right_bone_by_name", "p_bone"),
			&RenIK::set_upper_leg_right_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_upper_leg_right_bone_name"),
			&RenIK::get_upper_leg_right_bone_name);
	ClassDB::bind_method(D_METHOD("set_lower_leg_right_bone_by_name", "p_bone"),
			&RenIK::set_lower_leg_right_bone_by_name);
	ClassDB::bind_method(D_METHOD("get_lower_leg_right_bone_name"),
			&RenIK::get_lower_leg_right_bone_name);

	ClassDB::bind_method(D_METHOD("set_head_bone", "p_bone"),
			&RenIK::set_head_bone);
	ClassDB::bind_method(D_METHOD("get_head_bone"), &RenIK::get_head_bone);

	ClassDB::bind_method(D_METHOD("set_hand_left_bone", "p_bone"),
			&RenIK::set_hand_left_bone);
	ClassDB::bind_method(D_METHOD("get_hand_left_bone"),
			&RenIK::get_hand_left_bone);
	ClassDB::bind_method(D_METHOD("set_upper_arm_left_bone", "p_bone"),
			&RenIK::set_upper_arm_left_bone);
	ClassDB::bind_method(D_METHOD("get_upper_arm_left_bone"),
			&RenIK::get_upper_arm_left_bone);
	ClassDB::bind_method(D_METHOD("set_lower_arm_left_bone", "p_bone"),
			&RenIK::set_lower_arm_left_bone);
	ClassDB::bind_method(D_METHOD("get_lower_arm_left_bone"),
			&RenIK::get_lower_arm_left_bone);

	ClassDB::bind_method(D_METHOD("set_hand_right_bone", "p_bone"),
			&RenIK::set_hand_right_bone);
	ClassDB::bind_method(D_METHOD("get_hand_right_bone"),
			&RenIK::get_hand_right_bone);
	ClassDB::bind_method(D_METHOD("set_upper_arm_right_bone", "p_bone"),
			&RenIK::set_upper_arm_right_bone);
	ClassDB::bind_method(D_METHOD("get_upper_arm_right_bone"),
			&RenIK::get_upper_arm_right_bone);
	ClassDB::bind_method(D_METHOD("set_lower_arm_right_bone", "p_bone"),
			&RenIK::set_lower_arm_right_bone);
	ClassDB::bind_method(D_METHOD("get_lower_arm_right_bone"),
			&RenIK::get_lower_arm_right_bone);

	ClassDB::bind_method(D_METHOD("set_hip_bone", "p_bone"),
			&RenIK::set_hip_bone);
	ClassDB::bind_method(D_METHOD("get_hip_bone"), &RenIK::get_hip_bone);

	ClassDB::bind_method(D_METHOD("set_foot_left_bone", "p_bone"),
			&RenIK::set_foot_left_bone);
	ClassDB::bind_method(D_METHOD("get_foot_left_bone"),
			&RenIK::get_foot_left_bone);
	ClassDB::bind_method(D_METHOD("set_upper_leg_left_bone", "p_bone"),
			&RenIK::set_upper_leg_left_bone);
	ClassDB::bind_method(D_METHOD("get_upper_leg_left_bone"),
			&RenIK::get_upper_leg_left_bone);
	ClassDB::bind_method(D_METHOD("set_lower_leg_left_bone", "p_bone"),
			&RenIK::set_lower_leg_left_bone);
	ClassDB::bind_method(D_METHOD("get_lower_leg_left_bone"),
			&RenIK::get_lower_leg_left_bone);

	ClassDB::bind_method(D_METHOD("set_foot_right_bone", "p_bone"),
			&RenIK::set_foot_right_bone);
	ClassDB::bind_method(D_METHOD("get_foot_right_bone"),
			&RenIK::get_foot_right_bone);
	ClassDB::bind_method(D_METHOD("set_upper_leg_right_bone", "p_bone"),
			&RenIK::set_upper_leg_right_bone);
	ClassDB::bind_method(D_METHOD("get_upper_leg_right_bone"),
			&RenIK::get_upper_leg_right_bone);
	ClassDB::bind_method(D_METHOD("set_lower_leg_right_bone", "p_bone"),
			&RenIK::set_lower_leg_right_bone);
	ClassDB::bind_method(D_METHOD("get_lower_leg_right_bone"),
			&RenIK::get_lower_leg_right_bone);

	ClassDB::bind_method(D_METHOD("set_head_target_path", "p_path"),
			&RenIK::set_head_target_path);
	ClassDB::bind_method(D_METHOD("get_head_target_path"),
			&RenIK::get_head_target_path);
	ClassDB::bind_method(D_METHOD("set_hand_left_target_path", "p_path"),
			&RenIK::set_hand_left_target_path);
	ClassDB::bind_method(D_METHOD("get_hand_left_target_path"),
			&RenIK::get_hand_left_target_path);
	ClassDB::bind_method(D_METHOD("set_hand_right_target_path", "p_path"),
			&RenIK::set_hand_right_target_path);
	ClassDB::bind_method(D_METHOD("get_hand_right_target_path"),
			&RenIK::get_hand_right_target_path);
	ClassDB::bind_method(D_METHOD("set_hip_target_path", "p_path"),
			&RenIK::set_hip_target_path);
	ClassDB::bind_method(D_METHOD("get_hip_target_path"),
			&RenIK::get_hip_target_path);
	ClassDB::bind_method(D_METHOD("set_foot_left_target_path", "p_path"),
			&RenIK::set_foot_left_target_path);
	ClassDB::bind_method(D_METHOD("get_foot_left_target_path"),
			&RenIK::get_foot_left_target_path);
	ClassDB::bind_method(D_METHOD("set_foot_right_target_path", "p_path"),
			&RenIK::set_foot_right_target_path);
	ClassDB::bind_method(D_METHOD("get_foot_right_target_path"),
			&RenIK::get_foot_right_target_path);

	ClassDB::bind_method(D_METHOD("get_arm_upper_twist_offset"),
			&RenIK::get_arm_upper_twist_offset);
	ClassDB::bind_method(D_METHOD("set_arm_upper_twist_offset", "degrees"),
			&RenIK::set_arm_upper_twist_offset);
	ClassDB::bind_method(D_METHOD("get_arm_lower_twist_offset"),
			&RenIK::get_arm_lower_twist_offset);
	ClassDB::bind_method(D_METHOD("set_arm_lower_twist_offset", "degrees"),
			&RenIK::set_arm_lower_twist_offset);
	ClassDB::bind_method(D_METHOD("get_arm_roll_offset"),
			&RenIK::get_arm_roll_offset);
	ClassDB::bind_method(D_METHOD("set_arm_roll_offset", "degrees"),
			&RenIK::set_arm_roll_offset);
	ClassDB::bind_method(D_METHOD("get_arm_upper_limb_twist"),
			&RenIK::get_arm_upper_limb_twist);
	ClassDB::bind_method(D_METHOD("set_arm_upper_limb_twist", "ratio"),
			&RenIK::set_arm_upper_limb_twist);
	ClassDB::bind_method(D_METHOD("get_arm_lower_limb_twist"),
			&RenIK::get_arm_lower_limb_twist);
	ClassDB::bind_method(D_METHOD("set_arm_lower_limb_twist", "ratio"),
			&RenIK::set_arm_lower_limb_twist);
	ClassDB::bind_method(D_METHOD("get_arm_twist_inflection_point_offset"),
			&RenIK::get_arm_twist_inflection_point_offset);
	ClassDB::bind_method(
			D_METHOD("set_arm_twist_inflection_point_offset", "degrees"),
			&RenIK::set_arm_twist_inflection_point_offset);
	ClassDB::bind_method(D_METHOD("get_arm_twist_overflow"),
			&RenIK::get_arm_twist_overflow);
	ClassDB::bind_method(D_METHOD("set_arm_twist_overflow", "degrees"),
			&RenIK::set_arm_twist_overflow);

	ClassDB::bind_method(D_METHOD("get_arm_pole_offset"),
			&RenIK::get_arm_pole_offset);
	ClassDB::bind_method(D_METHOD("set_arm_pole_offset", "euler"),
			&RenIK::set_arm_pole_offset);
	ClassDB::bind_method(D_METHOD("get_arm_target_position_influence"),
			&RenIK::get_arm_target_position_influence);
	ClassDB::bind_method(D_METHOD("set_arm_target_position_influence", "xyz"),
			&RenIK::set_arm_target_position_influence);
	ClassDB::bind_method(D_METHOD("get_arm_target_rotation_influence"),
			&RenIK::get_arm_target_rotation_influence);
	ClassDB::bind_method(
			D_METHOD("set_arm_target_rotation_influence", "influence"),
			&RenIK::set_arm_target_rotation_influence);

	ClassDB::bind_method(D_METHOD("get_leg_upper_twist_offset"),
			&RenIK::get_leg_upper_twist_offset);
	ClassDB::bind_method(D_METHOD("set_leg_upper_twist_offset", "degrees"),
			&RenIK::set_leg_upper_twist_offset);
	ClassDB::bind_method(D_METHOD("get_leg_lower_twist_offset"),
			&RenIK::get_leg_lower_twist_offset);
	ClassDB::bind_method(D_METHOD("set_leg_lower_twist_offset", "degrees"),
			&RenIK::set_leg_lower_twist_offset);
	ClassDB::bind_method(D_METHOD("get_leg_roll_offset"),
			&RenIK::get_leg_roll_offset);
	ClassDB::bind_method(D_METHOD("set_leg_roll_offset", "degrees"),
			&RenIK::set_leg_roll_offset);
	ClassDB::bind_method(D_METHOD("get_leg_upper_limb_twist"),
			&RenIK::get_leg_upper_limb_twist);
	ClassDB::bind_method(D_METHOD("set_leg_upper_limb_twist", "ratio"),
			&RenIK::set_leg_upper_limb_twist);
	ClassDB::bind_method(D_METHOD("get_leg_lower_limb_twist"),
			&RenIK::get_leg_lower_limb_twist);
	ClassDB::bind_method(D_METHOD("set_leg_lower_limb_twist", "ratio"),
			&RenIK::set_leg_lower_limb_twist);
	ClassDB::bind_method(D_METHOD("get_leg_twist_inflection_point_offset"),
			&RenIK::get_leg_twist_inflection_point_offset);
	ClassDB::bind_method(
			D_METHOD("set_leg_twist_inflection_point_offset", "degrees"),
			&RenIK::set_leg_twist_inflection_point_offset);
	ClassDB::bind_method(D_METHOD("get_leg_twist_overflow"),
			&RenIK::get_leg_twist_overflow);
	ClassDB::bind_method(D_METHOD("set_leg_twist_overflow", "degrees"),
			&RenIK::set_leg_twist_overflow);

	ClassDB::bind_method(D_METHOD("get_leg_pole_offset"),
			&RenIK::get_leg_pole_offset);
	ClassDB::bind_method(D_METHOD("set_leg_pole_offset", "euler"),
			&RenIK::set_leg_pole_offset);
	ClassDB::bind_method(D_METHOD("get_leg_target_position_influence"),
			&RenIK::get_leg_target_position_influence);
	ClassDB::bind_method(D_METHOD("set_leg_target_position_influence", "xyz"),
			&RenIK::set_leg_target_position_influence);
	ClassDB::bind_method(D_METHOD("get_leg_target_rotation_influence"),
			&RenIK::get_leg_target_rotation_influence);
	ClassDB::bind_method(
			D_METHOD("set_leg_target_rotation_influence", "influence"),
			&RenIK::set_leg_target_rotation_influence);

	ClassDB::bind_method(D_METHOD("get_spine_curve"), &RenIK::get_spine_curve);
	ClassDB::bind_method(D_METHOD("set_spine_curve", "direction"),
			&RenIK::set_spine_curve);
	ClassDB::bind_method(D_METHOD("get_upper_spine_stiffness"),
			&RenIK::get_upper_spine_stiffness);
	ClassDB::bind_method(D_METHOD("set_upper_spine_stiffness", "influence"),
			&RenIK::set_upper_spine_stiffness);
	ClassDB::bind_method(D_METHOD("get_lower_spine_stiffness"),
			&RenIK::get_lower_spine_stiffness);
	ClassDB::bind_method(D_METHOD("set_lower_spine_stiffness", "influence"),
			&RenIK::set_lower_spine_stiffness);
	ClassDB::bind_method(D_METHOD("get_spine_twist"), &RenIK::get_spine_twist);
	ClassDB::bind_method(D_METHOD("set_spine_twist", "influence"),
			&RenIK::set_spine_twist);
	ClassDB::bind_method(D_METHOD("get_spine_twist_start"),
			&RenIK::get_spine_twist_start);
	ClassDB::bind_method(D_METHOD("set_spine_twist_start", "influence"),
			&RenIK::set_spine_twist_start);
	ClassDB::bind_method(D_METHOD("get_shoulder_influence"),
			&RenIK::get_shoulder_influence);
	ClassDB::bind_method(D_METHOD("set_shoulder_influence", "influence"),
			&RenIK::set_shoulder_influence);
	ClassDB::bind_method(D_METHOD("get_shoulder_offset"),
			&RenIK::get_shoulder_offset);
	ClassDB::bind_method(D_METHOD("set_shoulder_offset", "euler"),
			&RenIK::set_shoulder_offset);
	ClassDB::bind_method(D_METHOD("get_shoulder_pole_offset"),
			&RenIK::get_shoulder_pole_offset);
	ClassDB::bind_method(D_METHOD("set_shoulder_pole_offset", "euler"),
			&RenIK::set_shoulder_pole_offset);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"),
			&RenIK::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"),
			&RenIK::is_collide_with_areas_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"),
			&RenIK::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"),
			&RenIK::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"),
			&RenIK::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"),
			&RenIK::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_bit", "bit", "value"),
			&RenIK::set_collision_mask_bit);
	ClassDB::bind_method(D_METHOD("get_collision_mask_bit", "bit"),
			&RenIK::get_collision_mask_bit);

	ClassDB::bind_method(D_METHOD("get_forward_speed_scalar_min"),
			&RenIK::get_forward_speed_scalar_min);
	ClassDB::bind_method(
			D_METHOD("set_forward_speed_scalar_min", "speed_scalar_min"),
			&RenIK::set_forward_speed_scalar_min);
	ClassDB::bind_method(D_METHOD("get_forward_speed_scalar_max"),
			&RenIK::get_forward_speed_scalar_max);
	ClassDB::bind_method(
			D_METHOD("set_forward_speed_scalar_max", "speed_scalar_max"),
			&RenIK::set_forward_speed_scalar_max);
	ClassDB::bind_method(D_METHOD("get_forward_ground_time"),
			&RenIK::get_forward_ground_time);
	ClassDB::bind_method(D_METHOD("set_forward_ground_time", "ground_time"),
			&RenIK::set_forward_ground_time);
	ClassDB::bind_method(D_METHOD("get_forward_lift_time_base"),
			&RenIK::get_forward_lift_time_base);
	ClassDB::bind_method(D_METHOD("set_forward_lift_time_base", "lift_time_base"),
			&RenIK::set_forward_lift_time_base);
	ClassDB::bind_method(D_METHOD("get_forward_lift_time_scalar"),
			&RenIK::get_forward_lift_time_scalar);
	ClassDB::bind_method(
			D_METHOD("set_forward_lift_time_scalar", "lift_time_scalar"),
			&RenIK::set_forward_lift_time_scalar);
	ClassDB::bind_method(D_METHOD("get_forward_apex_in_time_base"),
			&RenIK::get_forward_apex_in_time_base);
	ClassDB::bind_method(
			D_METHOD("set_forward_apex_in_time_base", "apex_in_time_base"),
			&RenIK::set_forward_apex_in_time_base);
	ClassDB::bind_method(D_METHOD("get_forward_apex_in_time_scalar"),
			&RenIK::get_forward_apex_in_time_scalar);
	ClassDB::bind_method(
			D_METHOD("set_forward_apex_in_time_scalar", "apex_in_time_scalar"),
			&RenIK::set_forward_apex_in_time_scalar);
	ClassDB::bind_method(D_METHOD("get_forward_apex_out_time_base"),
			&RenIK::get_forward_apex_out_time_base);
	ClassDB::bind_method(
			D_METHOD("set_forward_apex_out_time_base", "apex_out_time_base"),
			&RenIK::set_forward_apex_out_time_base);
	ClassDB::bind_method(D_METHOD("get_forward_apex_out_time_scalar"),
			&RenIK::get_forward_apex_out_time_scalar);
	ClassDB::bind_method(
			D_METHOD("set_forward_apex_out_time_scalar", "apex_out_time_scalar"),
			&RenIK::set_forward_apex_out_time_scalar);
	ClassDB::bind_method(D_METHOD("get_forward_drop_time_base"),
			&RenIK::get_forward_drop_time_base);
	ClassDB::bind_method(D_METHOD("set_forward_drop_time_base", "drop_time_base"),
			&RenIK::set_forward_drop_time_base);
	ClassDB::bind_method(D_METHOD("get_forward_drop_time_scalar"),
			&RenIK::get_forward_drop_time_scalar);
	ClassDB::bind_method(
			D_METHOD("set_forward_drop_time_scalar", "drop_time_scalar"),
			&RenIK::set_forward_drop_time_scalar);
	ClassDB::bind_method(D_METHOD("get_forward_tip_toe_distance_scalar"),
			&RenIK::get_forward_tip_toe_distance_scalar);
	ClassDB::bind_method(D_METHOD("set_forward_tip_toe_distance_scalar",
								 "tip_toe_distance_scalar"),
			&RenIK::set_forward_tip_toe_distance_scalar);
	ClassDB::bind_method(D_METHOD("get_forward_tip_toe_speed_scalar"),
			&RenIK::get_forward_tip_toe_speed_scalar);
	ClassDB::bind_method(
			D_METHOD("set_forward_tip_toe_speed_scalar", "tip_toe_speed_scalar"),
			&RenIK::set_forward_tip_toe_speed_scalar);
	ClassDB::bind_method(D_METHOD("get_forward_tip_toe_angle_max"),
			&RenIK::get_forward_tip_toe_angle_max);
	ClassDB::bind_method(
			D_METHOD("set_forward_tip_toe_angle_max", "tip_toe_angle_max"),
			&RenIK::set_forward_tip_toe_angle_max);
	ClassDB::bind_method(D_METHOD("get_forward_lift_vertical"),
			&RenIK::get_forward_lift_vertical);
	ClassDB::bind_method(D_METHOD("set_forward_lift_vertical", "lift_vertical"),
			&RenIK::set_forward_lift_vertical);
	ClassDB::bind_method(D_METHOD("get_forward_lift_vertical_scalar"),
			&RenIK::get_forward_lift_vertical_scalar);
	ClassDB::bind_method(
			D_METHOD("set_forward_lift_vertical_scalar", "lift_vertical_scalar"),
			&RenIK::set_forward_lift_vertical_scalar);
	ClassDB::bind_method(D_METHOD("get_forward_lift_horizontal_scalar"),
			&RenIK::get_forward_lift_horizontal_scalar);
	ClassDB::bind_method(
			D_METHOD("set_forward_lift_horizontal_scalar", "lift_horizontal_scalar"),
			&RenIK::set_forward_lift_horizontal_scalar);
	ClassDB::bind_method(D_METHOD("get_forward_lift_angle"),
			&RenIK::get_forward_lift_angle);
	ClassDB::bind_method(D_METHOD("set_forward_lift_angle", "lift_angle"),
			&RenIK::set_forward_lift_angle);
	ClassDB::bind_method(D_METHOD("get_forward_apex_vertical"),
			&RenIK::get_forward_apex_vertical);
	ClassDB::bind_method(D_METHOD("set_forward_apex_vertical", "apex_vertical"),
			&RenIK::set_forward_apex_vertical);
	ClassDB::bind_method(D_METHOD("get_forward_apex_vertical_scalar"),
			&RenIK::get_forward_apex_vertical_scalar);
	ClassDB::bind_method(
			D_METHOD("set_forward_apex_vertical_scalar", "apex_vertical_scalar"),
			&RenIK::set_forward_apex_vertical_scalar);
	ClassDB::bind_method(D_METHOD("get_forward_apex_angle"),
			&RenIK::get_forward_apex_angle);
	ClassDB::bind_method(D_METHOD("set_forward_apex_angle", "apex_angle"),
			&RenIK::set_forward_apex_angle);
	ClassDB::bind_method(D_METHOD("get_forward_drop_vertical"),
			&RenIK::get_forward_drop_vertical);
	ClassDB::bind_method(D_METHOD("set_forward_drop_vertical", "drop_vertical"),
			&RenIK::set_forward_drop_vertical);
	ClassDB::bind_method(D_METHOD("get_forward_drop_vertical_scalar"),
			&RenIK::get_forward_drop_vertical_scalar);
	ClassDB::bind_method(
			D_METHOD("set_forward_drop_vertical_scalar", "drop_vertical_scalar"),
			&RenIK::set_forward_drop_vertical_scalar);
	ClassDB::bind_method(D_METHOD("get_forward_drop_horizontal_scalar"),
			&RenIK::get_forward_drop_horizontal_scalar);
	ClassDB::bind_method(
			D_METHOD("set_forward_drop_horizontal_scalar", "drop_horizontal_scalar"),
			&RenIK::set_forward_drop_horizontal_scalar);
	ClassDB::bind_method(D_METHOD("get_forward_drop_angle"),
			&RenIK::get_forward_drop_angle);
	ClassDB::bind_method(D_METHOD("set_forward_drop_angle", "drop_angle"),
			&RenIK::set_forward_drop_angle);
	ClassDB::bind_method(D_METHOD("get_forward_contact_point_ease"),
			&RenIK::get_forward_contact_point_ease);
	ClassDB::bind_method(
			D_METHOD("set_forward_contact_point_ease", "contact_point_ease"),
			&RenIK::set_forward_contact_point_ease);
	ClassDB::bind_method(D_METHOD("get_forward_contact_point_ease_scalar"),
			&RenIK::get_forward_contact_point_ease_scalar);
	ClassDB::bind_method(D_METHOD("set_forward_contact_point_ease_scalar",
								 "contact_point_ease_scalar"),
			&RenIK::set_forward_contact_point_ease_scalar);
	ClassDB::bind_method(D_METHOD("get_forward_scaling_ease"),
			&RenIK::get_forward_scaling_ease);
	ClassDB::bind_method(D_METHOD("set_forward_scaling_ease", "scaling_ease"),
			&RenIK::set_forward_scaling_ease);

	ClassDB::bind_method(D_METHOD("get_backward_speed_scalar_min"),
			&RenIK::get_backward_speed_scalar_min);
	ClassDB::bind_method(
			D_METHOD("set_backward_speed_scalar_min", "speed_scalar_min"),
			&RenIK::set_backward_speed_scalar_min);
	ClassDB::bind_method(D_METHOD("get_backward_speed_scalar_max"),
			&RenIK::get_backward_speed_scalar_max);
	ClassDB::bind_method(
			D_METHOD("set_backward_speed_scalar_max", "speed_scalar_max"),
			&RenIK::set_backward_speed_scalar_max);
	ClassDB::bind_method(D_METHOD("get_backward_ground_time"),
			&RenIK::get_backward_ground_time);
	ClassDB::bind_method(D_METHOD("set_backward_ground_time", "ground_time"),
			&RenIK::set_backward_ground_time);
	ClassDB::bind_method(D_METHOD("get_backward_lift_time_base"),
			&RenIK::get_backward_lift_time_base);
	ClassDB::bind_method(
			D_METHOD("set_backward_lift_time_base", "lift_time_base"),
			&RenIK::set_backward_lift_time_base);
	ClassDB::bind_method(D_METHOD("get_backward_lift_time_scalar"),
			&RenIK::get_backward_lift_time_scalar);
	ClassDB::bind_method(
			D_METHOD("set_backward_lift_time_scalar", "lift_time_scalar"),
			&RenIK::set_backward_lift_time_scalar);
	ClassDB::bind_method(D_METHOD("get_backward_apex_in_time_base"),
			&RenIK::get_backward_apex_in_time_base);
	ClassDB::bind_method(
			D_METHOD("set_backward_apex_in_time_base", "apex_in_time_base"),
			&RenIK::set_backward_apex_in_time_base);
	ClassDB::bind_method(D_METHOD("get_backward_apex_in_time_scalar"),
			&RenIK::get_backward_apex_in_time_scalar);
	ClassDB::bind_method(
			D_METHOD("set_backward_apex_in_time_scalar", "apex_in_time_scalar"),
			&RenIK::set_backward_apex_in_time_scalar);
	ClassDB::bind_method(D_METHOD("get_backward_apex_out_time_base"),
			&RenIK::get_backward_apex_out_time_base);
	ClassDB::bind_method(
			D_METHOD("set_backward_apex_out_time_base", "apex_out_time_base"),
			&RenIK::set_backward_apex_out_time_base);
	ClassDB::bind_method(D_METHOD("get_backward_apex_out_time_scalar"),
			&RenIK::get_backward_apex_out_time_scalar);
	ClassDB::bind_method(
			D_METHOD("set_backward_apex_out_time_scalar", "apex_out_time_scalar"),
			&RenIK::set_backward_apex_out_time_scalar);
	ClassDB::bind_method(D_METHOD("get_backward_drop_time_base"),
			&RenIK::get_backward_drop_time_base);
	ClassDB::bind_method(
			D_METHOD("set_backward_drop_time_base", "drop_time_base"),
			&RenIK::set_backward_drop_time_base);
	ClassDB::bind_method(D_METHOD("get_backward_drop_time_scalar"),
			&RenIK::get_backward_drop_time_scalar);
	ClassDB::bind_method(
			D_METHOD("set_backward_drop_time_scalar", "drop_time_scalar"),
			&RenIK::set_backward_drop_time_scalar);
	ClassDB::bind_method(D_METHOD("get_backward_tip_toe_distance_scalar"),
			&RenIK::get_backward_tip_toe_distance_scalar);
	ClassDB::bind_method(D_METHOD("set_backward_tip_toe_distance_scalar",
								 "tip_toe_distance_scalar"),
			&RenIK::set_backward_tip_toe_distance_scalar);
	ClassDB::bind_method(D_METHOD("get_backward_tip_toe_speed_scalar"),
			&RenIK::get_backward_tip_toe_speed_scalar);
	ClassDB::bind_method(
			D_METHOD("set_backward_tip_toe_speed_scalar", "tip_toe_speed_scalar"),
			&RenIK::set_backward_tip_toe_speed_scalar);
	ClassDB::bind_method(D_METHOD("get_backward_tip_toe_angle_max"),
			&RenIK::get_backward_tip_toe_angle_max);
	ClassDB::bind_method(
			D_METHOD("set_backward_tip_toe_angle_max", "tip_toe_angle_max"),
			&RenIK::set_backward_tip_toe_angle_max);
	ClassDB::bind_method(D_METHOD("get_backward_lift_vertical"),
			&RenIK::get_backward_lift_vertical);
	ClassDB::bind_method(D_METHOD("set_backward_lift_vertical", "lift_vertical"),
			&RenIK::set_backward_lift_vertical);
	ClassDB::bind_method(D_METHOD("get_backward_lift_vertical_scalar"),
			&RenIK::get_backward_lift_vertical_scalar);
	ClassDB::bind_method(
			D_METHOD("set_backward_lift_vertical_scalar", "lift_vertical_scalar"),
			&RenIK::set_backward_lift_vertical_scalar);
	ClassDB::bind_method(D_METHOD("get_backward_lift_horizontal_scalar"),
			&RenIK::get_backward_lift_horizontal_scalar);
	ClassDB::bind_method(
			D_METHOD("set_backward_lift_horizontal_scalar", "lift_horizontal_scalar"),
			&RenIK::set_backward_lift_horizontal_scalar);
	ClassDB::bind_method(D_METHOD("get_backward_lift_angle"),
			&RenIK::get_backward_lift_angle);
	ClassDB::bind_method(D_METHOD("set_backward_lift_angle", "lift_angle"),
			&RenIK::set_backward_lift_angle);
	ClassDB::bind_method(D_METHOD("get_backward_apex_vertical"),
			&RenIK::get_backward_apex_vertical);
	ClassDB::bind_method(D_METHOD("set_backward_apex_vertical", "apex_vertical"),
			&RenIK::set_backward_apex_vertical);
	ClassDB::bind_method(D_METHOD("get_backward_apex_vertical_scalar"),
			&RenIK::get_backward_apex_vertical_scalar);
	ClassDB::bind_method(
			D_METHOD("set_backward_apex_vertical_scalar", "apex_vertical_scalar"),
			&RenIK::set_backward_apex_vertical_scalar);
	ClassDB::bind_method(D_METHOD("get_backward_apex_angle"),
			&RenIK::get_backward_apex_angle);
	ClassDB::bind_method(D_METHOD("set_backward_apex_angle", "apex_angle"),
			&RenIK::set_backward_apex_angle);
	ClassDB::bind_method(D_METHOD("get_backward_drop_vertical"),
			&RenIK::get_backward_drop_vertical);
	ClassDB::bind_method(D_METHOD("set_backward_drop_vertical", "drop_vertical"),
			&RenIK::set_backward_drop_vertical);
	ClassDB::bind_method(D_METHOD("get_backward_drop_vertical_scalar"),
			&RenIK::get_backward_drop_vertical_scalar);
	ClassDB::bind_method(
			D_METHOD("set_backward_drop_vertical_scalar", "drop_vertical_scalar"),
			&RenIK::set_backward_drop_vertical_scalar);
	ClassDB::bind_method(D_METHOD("get_backward_drop_horizontal_scalar"),
			&RenIK::get_backward_drop_horizontal_scalar);
	ClassDB::bind_method(
			D_METHOD("set_backward_drop_horizontal_scalar", "drop_horizontal_scalar"),
			&RenIK::set_backward_drop_horizontal_scalar);
	ClassDB::bind_method(D_METHOD("get_backward_drop_angle"),
			&RenIK::get_backward_drop_angle);
	ClassDB::bind_method(D_METHOD("set_backward_drop_angle", "drop_angle"),
			&RenIK::set_backward_drop_angle);
	ClassDB::bind_method(D_METHOD("get_backward_contact_point_ease"),
			&RenIK::get_backward_contact_point_ease);
	ClassDB::bind_method(
			D_METHOD("set_backward_contact_point_ease", "contact_point_ease"),
			&RenIK::set_backward_contact_point_ease);
	ClassDB::bind_method(D_METHOD("get_backward_contact_point_ease_scalar"),
			&RenIK::get_backward_contact_point_ease_scalar);
	ClassDB::bind_method(D_METHOD("set_backward_contact_point_ease_scalar",
								 "contact_point_ease_scalar"),
			&RenIK::set_backward_contact_point_ease_scalar);
	ClassDB::bind_method(D_METHOD("get_backward_scaling_ease"),
			&RenIK::get_backward_scaling_ease);
	ClassDB::bind_method(D_METHOD("set_backward_scaling_ease", "scaling_ease"),
			&RenIK::set_backward_scaling_ease);

	ClassDB::bind_method(D_METHOD("get_sideways_speed_scalar_min"),
			&RenIK::get_sideways_speed_scalar_min);
	ClassDB::bind_method(
			D_METHOD("set_sideways_speed_scalar_min", "speed_scalar_min"),
			&RenIK::set_sideways_speed_scalar_min);
	ClassDB::bind_method(D_METHOD("get_sideways_speed_scalar_max"),
			&RenIK::get_sideways_speed_scalar_max);
	ClassDB::bind_method(
			D_METHOD("set_sideways_speed_scalar_max", "speed_scalar_max"),
			&RenIK::set_sideways_speed_scalar_max);
	ClassDB::bind_method(D_METHOD("get_sideways_ground_time"),
			&RenIK::get_sideways_ground_time);
	ClassDB::bind_method(D_METHOD("set_sideways_ground_time", "ground_time"),
			&RenIK::set_sideways_ground_time);
	ClassDB::bind_method(D_METHOD("get_sideways_lift_time_base"),
			&RenIK::get_sideways_lift_time_base);
	ClassDB::bind_method(
			D_METHOD("set_sideways_lift_time_base", "lift_time_base"),
			&RenIK::set_sideways_lift_time_base);
	ClassDB::bind_method(D_METHOD("get_sideways_lift_time_scalar"),
			&RenIK::get_sideways_lift_time_scalar);
	ClassDB::bind_method(
			D_METHOD("set_sideways_lift_time_scalar", "lift_time_scalar"),
			&RenIK::set_sideways_lift_time_scalar);
	ClassDB::bind_method(D_METHOD("get_sideways_apex_in_time_base"),
			&RenIK::get_sideways_apex_in_time_base);
	ClassDB::bind_method(
			D_METHOD("set_sideways_apex_in_time_base", "apex_in_time_base"),
			&RenIK::set_sideways_apex_in_time_base);
	ClassDB::bind_method(D_METHOD("get_sideways_apex_in_time_scalar"),
			&RenIK::get_sideways_apex_in_time_scalar);
	ClassDB::bind_method(
			D_METHOD("set_sideways_apex_in_time_scalar", "apex_in_time_scalar"),
			&RenIK::set_sideways_apex_in_time_scalar);
	ClassDB::bind_method(D_METHOD("get_sideways_apex_out_time_base"),
			&RenIK::get_sideways_apex_out_time_base);
	ClassDB::bind_method(
			D_METHOD("set_sideways_apex_out_time_base", "apex_out_time_base"),
			&RenIK::set_sideways_apex_out_time_base);
	ClassDB::bind_method(D_METHOD("get_sideways_apex_out_time_scalar"),
			&RenIK::get_sideways_apex_out_time_scalar);
	ClassDB::bind_method(
			D_METHOD("set_sideways_apex_out_time_scalar", "apex_out_time_scalar"),
			&RenIK::set_sideways_apex_out_time_scalar);
	ClassDB::bind_method(D_METHOD("get_sideways_drop_time_base"),
			&RenIK::get_sideways_drop_time_base);
	ClassDB::bind_method(
			D_METHOD("set_sideways_drop_time_base", "drop_time_base"),
			&RenIK::set_sideways_drop_time_base);
	ClassDB::bind_method(D_METHOD("get_sideways_drop_time_scalar"),
			&RenIK::get_sideways_drop_time_scalar);
	ClassDB::bind_method(
			D_METHOD("set_sideways_drop_time_scalar", "drop_time_scalar"),
			&RenIK::set_sideways_drop_time_scalar);
	ClassDB::bind_method(D_METHOD("get_sideways_tip_toe_distance_scalar"),
			&RenIK::get_sideways_tip_toe_distance_scalar);
	ClassDB::bind_method(D_METHOD("set_sideways_tip_toe_distance_scalar",
								 "tip_toe_distance_scalar"),
			&RenIK::set_sideways_tip_toe_distance_scalar);
	ClassDB::bind_method(D_METHOD("get_sideways_tip_toe_speed_scalar"),
			&RenIK::get_sideways_tip_toe_speed_scalar);
	ClassDB::bind_method(
			D_METHOD("set_sideways_tip_toe_speed_scalar", "tip_toe_speed_scalar"),
			&RenIK::set_sideways_tip_toe_speed_scalar);
	ClassDB::bind_method(D_METHOD("get_sideways_tip_toe_angle_max"),
			&RenIK::get_sideways_tip_toe_angle_max);
	ClassDB::bind_method(
			D_METHOD("set_sideways_tip_toe_angle_max", "tip_toe_angle_max"),
			&RenIK::set_sideways_tip_toe_angle_max);
	ClassDB::bind_method(D_METHOD("get_sideways_lift_vertical"),
			&RenIK::get_sideways_lift_vertical);
	ClassDB::bind_method(D_METHOD("set_sideways_lift_vertical", "lift_vertical"),
			&RenIK::set_sideways_lift_vertical);
	ClassDB::bind_method(D_METHOD("get_sideways_lift_vertical_scalar"),
			&RenIK::get_sideways_lift_vertical_scalar);
	ClassDB::bind_method(
			D_METHOD("set_sideways_lift_vertical_scalar", "lift_vertical_scalar"),
			&RenIK::set_sideways_lift_vertical_scalar);
	ClassDB::bind_method(D_METHOD("get_sideways_lift_horizontal_scalar"),
			&RenIK::get_sideways_lift_horizontal_scalar);
	ClassDB::bind_method(
			D_METHOD("set_sideways_lift_horizontal_scalar", "lift_horizontal_scalar"),
			&RenIK::set_sideways_lift_horizontal_scalar);
	ClassDB::bind_method(D_METHOD("get_sideways_lift_angle"),
			&RenIK::get_sideways_lift_angle);
	ClassDB::bind_method(D_METHOD("set_sideways_lift_angle", "lift_angle"),
			&RenIK::set_sideways_lift_angle);
	ClassDB::bind_method(D_METHOD("get_sideways_apex_vertical"),
			&RenIK::get_sideways_apex_vertical);
	ClassDB::bind_method(D_METHOD("set_sideways_apex_vertical", "apex_vertical"),
			&RenIK::set_sideways_apex_vertical);
	ClassDB::bind_method(D_METHOD("get_sideways_apex_vertical_scalar"),
			&RenIK::get_sideways_apex_vertical_scalar);
	ClassDB::bind_method(
			D_METHOD("set_sideways_apex_vertical_scalar", "apex_vertical_scalar"),
			&RenIK::set_sideways_apex_vertical_scalar);
	ClassDB::bind_method(D_METHOD("get_sideways_apex_angle"),
			&RenIK::get_sideways_apex_angle);
	ClassDB::bind_method(D_METHOD("set_sideways_apex_angle", "apex_angle"),
			&RenIK::set_sideways_apex_angle);
	ClassDB::bind_method(D_METHOD("get_sideways_drop_vertical"),
			&RenIK::get_sideways_drop_vertical);
	ClassDB::bind_method(D_METHOD("set_sideways_drop_vertical", "drop_vertical"),
			&RenIK::set_sideways_drop_vertical);
	ClassDB::bind_method(D_METHOD("get_sideways_drop_vertical_scalar"),
			&RenIK::get_sideways_drop_vertical_scalar);
	ClassDB::bind_method(
			D_METHOD("set_sideways_drop_vertical_scalar", "drop_vertical_scalar"),
			&RenIK::set_sideways_drop_vertical_scalar);
	ClassDB::bind_method(D_METHOD("get_sideways_drop_horizontal_scalar"),
			&RenIK::get_sideways_drop_horizontal_scalar);
	ClassDB::bind_method(
			D_METHOD("set_sideways_drop_horizontal_scalar", "drop_horizontal_scalar"),
			&RenIK::set_sideways_drop_horizontal_scalar);
	ClassDB::bind_method(D_METHOD("get_sideways_drop_angle"),
			&RenIK::get_sideways_drop_angle);
	ClassDB::bind_method(D_METHOD("set_sideways_drop_angle", "drop_angle"),
			&RenIK::set_sideways_drop_angle);
	ClassDB::bind_method(D_METHOD("get_sideways_contact_point_ease"),
			&RenIK::get_sideways_contact_point_ease);
	ClassDB::bind_method(
			D_METHOD("set_sideways_contact_point_ease", "contact_point_ease"),
			&RenIK::set_sideways_contact_point_ease);
	ClassDB::bind_method(D_METHOD("get_sideways_contact_point_ease_scalar"),
			&RenIK::get_sideways_contact_point_ease_scalar);
	ClassDB::bind_method(D_METHOD("set_sideways_contact_point_ease_scalar",
								 "contact_point_ease_scalar"),
			&RenIK::set_sideways_contact_point_ease_scalar);
	ClassDB::bind_method(D_METHOD("get_sideways_scaling_ease"),
			&RenIK::get_sideways_scaling_ease);
	ClassDB::bind_method(D_METHOD("set_sideways_scaling_ease", "scaling_ease"),
			&RenIK::set_sideways_scaling_ease);
	ClassDB::bind_method(D_METHOD("setup_humanoid_bones"),
			&RenIK::setup_humanoid_bones);

	ClassDB::bind_method(D_METHOD("set_setup_humanoid_bones", "set_targets"), &RenIK::set_setup_humanoid_bones);
	ClassDB::bind_method(D_METHOD("get_setup_humanoid_bones"), &RenIK::get_setup_humanoid_bones);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "setup_humanoid_bones"), "set_setup_humanoid_bones", "get_setup_humanoid_bones");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "live_preview"), "set_live_preview",
			"get_live_preview");

	ADD_GROUP("Armature", "armature_");
	// ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,
	// 					 RENIK_PROPERTY_STRING_SKELETON_PATH,
	// 					 PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D",
	// 					 PROPERTY_USAGE_DEFAULT),
	// 		"set_skeleton_path", "get_skeleton_path");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_HEAD_BONE),
			"set_head_bone_by_name", "get_head_bone_name");
	ADD_PROPERTY(
			PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_HAND_LEFT_BONE),
			"set_hand_left_bone_by_name", "get_hand_left_bone_name");
	ADD_PROPERTY(
			PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_LEFT_LOWER_ARM_BONE),
			"set_lower_arm_left_bone_by_name", "get_lower_arm_left_bone_name");
	ADD_PROPERTY(
			PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_LEFT_UPPER_ARM_BONE),
			"set_upper_arm_left_bone_by_name", "get_upper_arm_left_bone_name");

	ADD_PROPERTY(
			PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_HAND_RIGHT_BONE),
			"set_hand_right_bone_by_name", "get_hand_right_bone_name");
	ADD_PROPERTY(
			PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_RIGHT_LOWER_ARM_BONE),
			"set_lower_arm_right_bone_by_name", "get_lower_arm_right_bone_name");
	ADD_PROPERTY(
			PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_RIGHT_UPPER_ARM_BONE),
			"set_upper_arm_right_bone_by_name", "get_upper_arm_right_bone_name");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_HIP_BONE),
			"set_hip_bone_by_name", "get_hip_bone_name");

	ADD_PROPERTY(
			PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_FOOT_LEFT_BONE),
			"set_foot_left_bone_by_name", "get_foot_left_bone_name");
	ADD_PROPERTY(
			PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_LEFT_LOWER_LEG_BONE),
			"set_lower_leg_left_bone_by_name", "get_lower_leg_left_bone_name");
	ADD_PROPERTY(
			PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_LEFT_UPPER_LEG_BONE),
			"set_upper_leg_left_bone_by_name", "get_upper_leg_left_bone_name");

	ADD_PROPERTY(
			PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_FOOT_RIGHT_BONE),
			"set_foot_right_bone_by_name", "get_foot_right_bone_name");
	ADD_PROPERTY(
			PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_RIGHT_LOWER_LEG_BONE),
			"set_lower_leg_right_bone_by_name", "get_lower_leg_right_bone_name");
	ADD_PROPERTY(
			PropertyInfo(Variant::STRING, RENIK_PROPERTY_STRING_RIGHT_UPPER_LEG_BONE),
			"set_upper_leg_right_bone_by_name", "get_upper_leg_right_bone_name");

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,
						 RENIK_PROPERTY_STRING_HEAD_TARGET_PATH,
						 PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"),
			"set_head_target_path", "get_head_target_path");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,
						 RENIK_PROPERTY_STRING_HAND_LEFT_TARGET_PATH,
						 PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"),
			"set_hand_left_target_path", "get_hand_left_target_path");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,
						 RENIK_PROPERTY_STRING_HAND_RIGHT_TARGET_PATH,
						 PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"),
			"set_hand_right_target_path", "get_hand_right_target_path");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,
						 RENIK_PROPERTY_STRING_HIP_TARGET_PATH,
						 PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"),
			"set_hip_target_path", "get_hip_target_path");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,
						 RENIK_PROPERTY_STRING_FOOT_LEFT_TARGET_PATH,
						 PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"),
			"set_foot_left_target_path", "get_foot_left_target_path");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,
						 RENIK_PROPERTY_STRING_FOOT_RIGHT_TARGET_PATH,
						 PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D"),
			"set_foot_right_target_path", "get_foot_right_target_path");

	ADD_GROUP("Arm IK Settings", "arm_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arm_elbow_direction_offset",
						 PROPERTY_HINT_RANGE, "-360,360,0.1"),
			"set_arm_roll_offset", "get_arm_roll_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arm_upper_arm_twisting",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_arm_upper_limb_twist", "get_arm_upper_limb_twist");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arm_upper_arm_twist_offset",
						 PROPERTY_HINT_RANGE, "-360,360,0.1"),
			"set_arm_upper_twist_offset", "get_arm_upper_twist_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arm_forearm_twisting",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_arm_lower_limb_twist", "get_arm_lower_limb_twist");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arm_forearm_twist_offset",
						 PROPERTY_HINT_RANGE, "-360,360,0.1"),
			"set_arm_lower_twist_offset", "get_arm_lower_twist_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arm_twist_inflection_point",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_arm_twist_inflection_point_offset",
			"get_arm_twist_inflection_point_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arm_twist_overflow",
						 PROPERTY_HINT_RANGE, "0,180,0.1"),
			"set_arm_twist_overflow", "get_arm_twist_overflow");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arm_shoulder_influence",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_shoulder_influence", "get_shoulder_influence");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "arm_shoulder_offset"),
			"set_shoulder_offset", "get_shoulder_offset");

	ADD_GROUP("Arm IK Settings (Advanced)", "arm_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "arm_pole_offset"),
			"set_arm_pole_offset", "get_arm_pole_offset");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "arm_target_position_influence"),
			"set_arm_target_position_influence",
			"get_arm_target_position_influence");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arm_target_rotation_influence",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_arm_target_rotation_influence",
			"get_arm_target_rotation_influence");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "arm_shoulder_pole_offset"),
			"set_shoulder_pole_offset", "get_shoulder_pole_offset");

	ADD_GROUP("Leg IK Settings", "leg_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "leg_knee_direction_offset",
						 PROPERTY_HINT_RANGE, "-360,360,0.1"),
			"set_leg_roll_offset", "get_leg_roll_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "leg_thigh_twisting",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_leg_upper_limb_twist", "get_leg_upper_limb_twist");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "leg_thigh_twist_offset",
						 PROPERTY_HINT_RANGE, "-360,360,0.1"),
			"set_leg_upper_twist_offset", "get_leg_upper_twist_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "leg_lower_leg_twisting",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_leg_lower_limb_twist", "get_leg_lower_limb_twist");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "leg_lower_leg_twist_offset",
						 PROPERTY_HINT_RANGE, "-360,360,0.1"),
			"set_leg_lower_twist_offset", "get_leg_lower_twist_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "leg_twist_inflection_point",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_leg_twist_inflection_point_offset",
			"get_leg_twist_inflection_point_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "leg_twist_overflow",
						 PROPERTY_HINT_RANGE, "0,-180,0.1"),
			"set_leg_twist_overflow", "get_leg_twist_overflow");

	ADD_GROUP("Leg IK Settings (Advanced)", "leg_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "leg_pole_offset"),
			"set_leg_pole_offset", "get_leg_pole_offset");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "leg_target_position_influence"),
			"set_leg_target_position_influence",
			"get_leg_target_position_influence");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "leg_target_rotation_influence",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_leg_target_rotation_influence",
			"get_leg_target_rotation_influence");

	ADD_GROUP("Torso IK Settings", "torso_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "torso_spine_curve"),
			"set_spine_curve", "get_spine_curve");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "torso_upper_spine_stiffness",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_upper_spine_stiffness", "get_upper_spine_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "torso_lower_spine_stiffness",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_lower_spine_stiffness", "get_lower_spine_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "torso_spine_twist",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_spine_twist", "get_spine_twist");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "torso_spine_twist_start",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_spine_twist_start", "get_spine_twist_start");

	ADD_GROUP("Walk Collisions (Advanced)", "walk_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "walk_collision_mask",
						 PROPERTY_HINT_LAYERS_3D_PHYSICS),
			"set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "walk_collide_with_areas",
						 PROPERTY_HINT_LAYERS_3D_PHYSICS),
			"set_collide_with_areas", "is_collide_with_areas_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "walk_collide_with_bodies",
						 PROPERTY_HINT_LAYERS_3D_PHYSICS),
			"set_collide_with_bodies", "is_collide_with_bodies_enabled");

	ADD_GROUP("Forward Gait (Advanced)", "forward_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_speed_scalar_min",
						 PROPERTY_HINT_RANGE, "0,200,0.1"),
			"set_forward_speed_scalar_min", "get_forward_speed_scalar_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_speed_scalar_max",
						 PROPERTY_HINT_RANGE, "0,200,0.1"),
			"set_forward_speed_scalar_max", "get_forward_speed_scalar_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_ground_time",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_ground_time", "get_forward_ground_time");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_lift_time_base",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_lift_time_base", "get_forward_lift_time_base");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_lift_time_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_lift_time_scalar", "get_forward_lift_time_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_apex_in_time_base",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_apex_in_time_base",
			"get_forward_apex_in_time_base");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_apex_in_time_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_apex_in_time_scalar",
			"get_forward_apex_in_time_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_apex_out_time_base",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_apex_out_time_base",
			"get_forward_apex_out_time_base");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_apex_out_time_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_apex_out_time_scalar",
			"get_forward_apex_out_time_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_drop_time_base",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_drop_time_base", "get_forward_drop_time_base");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_drop_time_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_drop_time_scalar", "get_forward_drop_time_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_tip_toe_distance_scalar",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_forward_tip_toe_distance_scalar",
			"get_forward_tip_toe_distance_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_tip_toe_speed_scalar",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_forward_tip_toe_speed_scalar",
			"get_forward_tip_toe_speed_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_tip_toe_angle_max",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_forward_tip_toe_angle_max",
			"get_forward_tip_toe_angle_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_lift_vertical",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_lift_vertical", "get_forward_lift_vertical");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_lift_vertical_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_lift_vertical_scalar",
			"get_forward_lift_vertical_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_lift_horizontal_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_lift_horizontal_scalar",
			"get_forward_lift_horizontal_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_lift_angle",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_forward_lift_angle", "get_forward_lift_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_apex_vertical",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_apex_vertical", "get_forward_apex_vertical");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_apex_vertical_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_apex_vertical_scalar",
			"get_forward_apex_vertical_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_apex_angle",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_forward_apex_angle", "get_forward_apex_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_drop_vertical",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_drop_vertical", "get_forward_drop_vertical");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_drop_vertical_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_drop_vertical_scalar",
			"get_forward_drop_vertical_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_drop_horizontal_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_drop_horizontal_scalar",
			"get_forward_drop_horizontal_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_drop_angle",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_forward_drop_angle", "get_forward_drop_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_contact_point_ease",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_contact_point_ease",
			"get_forward_contact_point_ease");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_contact_point_ease_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_contact_point_ease_scalar",
			"get_forward_contact_point_ease_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "forward_scaling_ease",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_forward_scaling_ease", "get_forward_scaling_ease");

	ADD_GROUP("Backward Gait (Advanced)", "backward_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_speed_scalar_min",
						 PROPERTY_HINT_RANGE, "0,200,0.1"),
			"set_backward_speed_scalar_min",
			"get_backward_speed_scalar_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_speed_scalar_max",
						 PROPERTY_HINT_RANGE, "0,200,0.1"),
			"set_backward_speed_scalar_max",
			"get_backward_speed_scalar_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_ground_time",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_ground_time", "get_backward_ground_time");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_lift_time_base",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_lift_time_base", "get_backward_lift_time_base");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_lift_time_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_lift_time_scalar",
			"get_backward_lift_time_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_apex_in_time_base",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_apex_in_time_base",
			"get_backward_apex_in_time_base");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_apex_in_time_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_apex_in_time_scalar",
			"get_backward_apex_in_time_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_apex_out_time_base",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_apex_out_time_base",
			"get_backward_apex_out_time_base");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_apex_out_time_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_apex_out_time_scalar",
			"get_backward_apex_out_time_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_drop_time_base",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_drop_time_base", "get_backward_drop_time_base");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_drop_time_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_drop_time_scalar",
			"get_backward_drop_time_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_tip_toe_distance_scalar",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_backward_tip_toe_distance_scalar",
			"get_backward_tip_toe_distance_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_tip_toe_speed_scalar",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_backward_tip_toe_speed_scalar",
			"get_backward_tip_toe_speed_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_tip_toe_angle_max",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_backward_tip_toe_angle_max",
			"get_backward_tip_toe_angle_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_lift_vertical",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_lift_vertical", "get_backward_lift_vertical");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_lift_vertical_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_lift_vertical_scalar",
			"get_backward_lift_vertical_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_lift_horizontal_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_lift_horizontal_scalar",
			"get_backward_lift_horizontal_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_lift_angle",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_backward_lift_angle", "get_backward_lift_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_apex_vertical",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_apex_vertical", "get_backward_apex_vertical");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_apex_vertical_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_apex_vertical_scalar",
			"get_backward_apex_vertical_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_apex_angle",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_backward_apex_angle", "get_backward_apex_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_drop_vertical",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_drop_vertical", "get_backward_drop_vertical");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_drop_vertical_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_drop_vertical_scalar",
			"get_backward_drop_vertical_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_drop_horizontal_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_drop_horizontal_scalar",
			"get_backward_drop_horizontal_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_drop_angle",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_backward_drop_angle", "get_backward_drop_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_contact_point_ease",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_contact_point_ease",
			"get_backward_contact_point_ease");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT,
						 "backward_contact_point_ease_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_contact_point_ease_scalar",
			"get_backward_contact_point_ease_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "backward_scaling_ease",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_backward_scaling_ease", "get_backward_scaling_ease");

	ADD_GROUP("Sideways Gait (Advanced)", "sideways_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_speed_scalar_min",
						 PROPERTY_HINT_RANGE, "0,200,0.1"),
			"set_sideways_speed_scalar_min",
			"get_sideways_speed_scalar_min");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_speed_scalar_max",
						 PROPERTY_HINT_RANGE, "0,200,0.1"),
			"set_sideways_speed_scalar_max",
			"get_sideways_speed_scalar_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_ground_time",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_ground_time", "get_sideways_ground_time");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_lift_time_base",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_lift_time_base", "get_sideways_lift_time_base");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_lift_time_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_lift_time_scalar",
			"get_sideways_lift_time_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_apex_in_time_base",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_apex_in_time_base",
			"get_sideways_apex_in_time_base");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_apex_in_time_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_apex_in_time_scalar",
			"get_sideways_apex_in_time_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_apex_out_time_base",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_apex_out_time_base",
			"get_sideways_apex_out_time_base");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_apex_out_time_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_apex_out_time_scalar",
			"get_sideways_apex_out_time_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_drop_time_base",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_drop_time_base", "get_sideways_drop_time_base");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_drop_time_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_drop_time_scalar",
			"get_sideways_drop_time_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_tip_toe_distance_scalar",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_sideways_tip_toe_distance_scalar",
			"get_sideways_tip_toe_distance_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_tip_toe_speed_scalar",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_sideways_tip_toe_speed_scalar",
			"get_sideways_tip_toe_speed_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_tip_toe_angle_max",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_sideways_tip_toe_angle_max",
			"get_sideways_tip_toe_angle_max");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_lift_vertical",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_lift_vertical", "get_sideways_lift_vertical");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_lift_vertical_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_lift_vertical_scalar",
			"get_sideways_lift_vertical_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_lift_horizontal_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_lift_horizontal_scalar",
			"get_sideways_lift_horizontal_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_lift_angle",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_sideways_lift_angle", "get_sideways_lift_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_apex_vertical",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_apex_vertical", "get_sideways_apex_vertical");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_apex_vertical_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_apex_vertical_scalar",
			"get_sideways_apex_vertical_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_apex_angle",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_sideways_apex_angle", "get_sideways_apex_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_drop_vertical",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_drop_vertical", "get_sideways_drop_vertical");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_drop_vertical_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_drop_vertical_scalar",
			"get_sideways_drop_vertical_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_drop_horizontal_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_drop_horizontal_scalar",
			"get_sideways_drop_horizontal_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_drop_angle",
						 PROPERTY_HINT_RANGE, "-180,180,0.1"),
			"set_sideways_drop_angle", "get_sideways_drop_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_contact_point_ease",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_contact_point_ease",
			"get_sideways_contact_point_ease");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT,
						 "sideways_contact_point_ease_scalar",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_contact_point_ease_scalar",
			"get_sideways_contact_point_ease_scalar");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sideways_scaling_ease",
						 PROPERTY_HINT_RANGE, "0,100,0.1"),
			"set_sideways_scaling_ease", "get_sideways_scaling_ease");

	ClassDB::bind_method(D_METHOD("update_ik"), &RenIK::update_ik);
	ClassDB::bind_method(D_METHOD("update_placement", "delta"), &RenIK::update_placement);
}

void RenIK::_validate_property(PropertyInfo &property) const {
	if (property.name == RENIK_PROPERTY_STRING_HEAD_BONE ||
			property.name == RENIK_PROPERTY_STRING_HIP_BONE ||
			property.name == RENIK_PROPERTY_STRING_HAND_LEFT_BONE ||
			property.name == RENIK_PROPERTY_STRING_LEFT_LOWER_ARM_BONE ||
			property.name == RENIK_PROPERTY_STRING_LEFT_UPPER_ARM_BONE ||
			property.name == RENIK_PROPERTY_STRING_HAND_RIGHT_BONE ||
			property.name == RENIK_PROPERTY_STRING_RIGHT_LOWER_ARM_BONE ||
			property.name == RENIK_PROPERTY_STRING_RIGHT_UPPER_ARM_BONE ||
			property.name == RENIK_PROPERTY_STRING_FOOT_LEFT_BONE ||
			property.name == RENIK_PROPERTY_STRING_LEFT_LOWER_LEG_BONE ||
			property.name == RENIK_PROPERTY_STRING_LEFT_UPPER_LEG_BONE ||
			property.name == RENIK_PROPERTY_STRING_FOOT_RIGHT_BONE ||
			property.name == RENIK_PROPERTY_STRING_RIGHT_LOWER_LEG_BONE ||
			property.name == RENIK_PROPERTY_STRING_RIGHT_UPPER_LEG_BONE) {
		if (skeleton) {
			String names(",");
			for (int i = 0; i < skeleton->get_bone_count(); i++) {
				if (i > 0) {
					names += ",";
				}
				names += skeleton->get_bone_name(i);
			}
			property.hint = PROPERTY_HINT_ENUM;
			property.hint_string = names;
		} else {
			property.hint = PROPERTY_HINT_NONE;
			property.hint_string = "";
		}
	}
}

void RenIK::_notification(int p_what) {
	// switch (p_what) {
	// 	case NOTIFICATION_POSTINITIALIZE: {
	// 		left_shoulder_offset = Vector3(Math::deg_to_rad(0.0), Math::deg_to_rad(0.0),
	// 				Math::deg_to_rad(0.0));
	// 		right_shoulder_offset = Vector3(Math::deg_to_rad(0.0), Math::deg_to_rad(0.0),
	// 				Math::deg_to_rad(0.0));
	// 		left_shoulder_pole_offset = Vector3(Math::deg_to_rad(0.0), Math::deg_to_rad(0.0),
	// 				Math::deg_to_rad(78.0));
	// 		right_shoulder_pole_offset = Vector3(Math::deg_to_rad(0.0), Math::deg_to_rad(0.0),
	// 				Math::deg_to_rad(-78.0));
	// 		spine_chain.instantiate();
	// 		spine_chain->init(Vector3(0, 15, -15), 0.5, 0.5, 1, 0);
	// 		limb_arm_left.instantiate();
	// 		limb_arm_left->init(0, 0, Math::deg_to_rad(80.0), 0.5, 0,
	// 				Math::deg_to_rad(-180.0), Math::deg_to_rad(45.0), 0.33,
	// 				Vector3(Math::deg_to_rad(15.0), 0, Math::deg_to_rad(60.0)),
	// 				Vector3(2.0, -1.5, -1.0));
	// 		limb_arm_right.instantiate();
	// 		limb_arm_right->init(0, 0, Math::deg_to_rad(80.0), 0.5, 0,
	// 				Math::deg_to_rad(-180.0), Math::deg_to_rad(45.0), 0.33,
	// 				Vector3(Math::deg_to_rad(15.0), 0, Math::deg_to_rad(-60.0)),
	// 				Vector3(2.0, 1.5, 1.0));
	// 		limb_leg_left.instantiate();
	// 		limb_leg_left->init(0, Math::deg_to_rad(-180.0), 0, 0.25, 0.25, 0, Math::deg_to_rad(45.0), 0.5,
	// 				Vector3(0, 0, Math_PI), Vector3());
	// 		limb_leg_right.instantiate();
	// 		limb_leg_right->init(0, Math::deg_to_rad(-180.0), 0, 0.25, 0.25, 0, Math::deg_to_rad(45.0), 0.5,
	// 				Vector3(0, 0, -Math_PI), Vector3());
	// 		set_leg_pole_offset(Vector3(0, 0, 180));
	// 		set_arm_pole_offset(Vector3(15, 0, 60));
	// 	} break;
	// 	case NOTIFICATION_READY: {
	// 		_initialize();
	// 	} break;
	// 	case NOTIFICATION_INTERNAL_PROCESS: {
	// 		if (!Engine::get_singleton()->is_editor_hint() || live_preview) {
	// 			update_ik();
	// 		}
	// 		break;
	// 	}
	// 	case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
	// 		if (!Engine::get_singleton()->is_editor_hint() || live_preview) {
	// 			update_placement(get_physics_process_delta_time());
	// 		}
	// 	} break;
	// }
}
void RenIK::on_post_initialize()
{
	left_shoulder_offset = Vector3(Math::deg_to_rad(0.0), Math::deg_to_rad(0.0),
			Math::deg_to_rad(0.0));
	right_shoulder_offset = Vector3(Math::deg_to_rad(0.0), Math::deg_to_rad(0.0),
			Math::deg_to_rad(0.0));
	left_shoulder_pole_offset = Vector3(Math::deg_to_rad(0.0), Math::deg_to_rad(0.0),
			Math::deg_to_rad(78.0));
	right_shoulder_pole_offset = Vector3(Math::deg_to_rad(0.0), Math::deg_to_rad(0.0),
			Math::deg_to_rad(-78.0));
	spine_chain.instantiate();
	spine_chain->init(Vector3(0, 15, -15), 0.5, 0.5, 1, 0);
	limb_arm_left.instantiate();
	limb_arm_left->init(0, 0, Math::deg_to_rad(80.0), 0.5, 0,
			Math::deg_to_rad(-180.0), Math::deg_to_rad(45.0), 0.33,
			Vector3(Math::deg_to_rad(15.0), 0, Math::deg_to_rad(60.0)),
			Vector3(2.0, -1.5, -1.0));
	limb_arm_right.instantiate();
	limb_arm_right->init(0, 0, Math::deg_to_rad(80.0), 0.5, 0,
			Math::deg_to_rad(-180.0), Math::deg_to_rad(45.0), 0.33,
			Vector3(Math::deg_to_rad(15.0), 0, Math::deg_to_rad(-60.0)),
			Vector3(2.0, 1.5, 1.0));
	limb_leg_left.instantiate();
	limb_leg_left->init(0, Math::deg_to_rad(-180.0), 0, 0.25, 0.25, 0, Math::deg_to_rad(45.0), 0.5,
			Vector3(0, 0, Math_PI), Vector3());
	limb_leg_right.instantiate();
	limb_leg_right->init(0, Math::deg_to_rad(-180.0), 0, 0.25, 0.25, 0, Math::deg_to_rad(45.0), 0.5,
			Vector3(0, 0, -Math_PI), Vector3());
	set_leg_pole_offset(Vector3(0, 0, 180));
	set_arm_pole_offset(Vector3(15, 0, 60));

}

void RenIK::_initialize(Skeleton3D* p_skeleton) {
	// set the skeleton to the parent if we can
	set_skeleton(p_skeleton);
	set_head_target_path(get_head_target_path());
	set_hip_target_path(get_hip_target_path());
	set_hand_left_target_path(get_hand_left_target_path());
	set_hand_right_target_path(get_hand_right_target_path());
	set_foot_left_target_path(get_foot_left_target_path());
	set_foot_right_target_path(get_foot_right_target_path());

	// if (Engine::get_singleton()->is_editor_hint()) {
	// 	set_process_internal(true);
	// 	set_physics_process_internal(true);
	// }
	// #ifdef TOOLS_ENABLED
	// set_process_priority(1); //makes sure that ik is done last after all
	// physics and movement have taken place enable_solve_ik_every_frame(true);
	// enable_hip_placement(true);
	// enable_foot_placement(true);
	// set_physics_process_internal(true);
	// #endif
}

void RenIK::enable_solve_ik_every_frame(bool automatically_update_ik) {
	//set_process_internal(automatically_update_ik);
}

void RenIK::enable_hip_placement(bool enabled) { hip_placement = enabled; }

void RenIK::enable_foot_placement(bool enabled) { foot_placement = enabled; }

void RenIK::update_ik() {
	// Saracen: since the foot placement is updated in the physics frame,
	// interpolate the results to avoid jitter
	placement.interpolate_transforms(
			Engine::get_singleton()->get_physics_interpolation_fraction(),
			!hip_target_spatial, foot_placement);

	if (skeleton) {
		Transform3D skel_inverse = skeleton->get_global_transform().affine_inverse();
		SpineTransforms spine_global_transforms = perform_torso_ik();
		if (hand_left_target_spatial) {
			perform_hand_left_ik(spine_global_transforms.left_arm_parent_transform, skel_inverse * hand_left_target_spatial->get_global_transform());
		}
		if (hand_right_target_spatial) {
			perform_hand_right_ik(spine_global_transforms.right_arm_parent_transform, skel_inverse * hand_right_target_spatial->get_global_transform());
		}
		if (foot_left_target_spatial) {
			perform_foot_left_ik(spine_global_transforms.hip_transform, skel_inverse * foot_left_target_spatial->get_global_transform());
		} else if (foot_placement) {
			perform_foot_left_ik(spine_global_transforms.hip_transform, skel_inverse * placement.interpolated_left_foot);
		}
		if (foot_right_target_spatial) {
			perform_foot_right_ik(spine_global_transforms.hip_transform, skel_inverse * foot_right_target_spatial->get_global_transform());
		} else if (foot_placement) {
			perform_foot_right_ik(spine_global_transforms.hip_transform, skel_inverse * placement.interpolated_right_foot);
		}
	}
}

void RenIK::update_placement(float delta) {
	// Saracen: save the transforms from the last update for use with
	// interpolation
	placement.save_previous_transforms();

	// Based on head position and delta time, we calc our speed and distance from
	// the ground and place the feet accordingly
	if (foot_placement && head_target_spatial &&
			head_target_spatial->is_inside_world()) {
		placement.foot_place(delta, head_target_spatial->get_global_transform(),
				head_target_spatial->get_world_3d(), false);
	}
	if (hip_placement && head_target_spatial) {
		// calc twist from hands here
		float twist = 0;
		if (foot_placement) {
			placement.hip_place(delta, head_target_spatial->get_global_transform(),
					placement.target_left_foot,
					placement.target_right_foot, twist, false);
		} else {
			if (foot_left_target_spatial != nullptr &&
					foot_right_target_spatial != nullptr) {
				placement.hip_place(delta, head_target_spatial->get_global_transform(),
						foot_left_target_spatial->get_global_transform(),
						foot_right_target_spatial->get_global_transform(),
						twist, false);
			} else {
				// I have no idea when you would hit this code, but if you do, it'll lag
				// everything because of the call to get_bone_global_pose
				placement.hip_place(
						delta, head_target_spatial->get_global_transform(),
						skeleton->get_global_transform() *
								skeleton->get_bone_global_pose(get_foot_left_bone()),
						skeleton->get_global_transform() *
								skeleton->get_bone_global_pose(get_foot_right_bone()),
						twist, false);
			}
		}
	}
}

void RenIK::apply_ik_map(HashMap<BoneId, Quaternion> ik_map,
		Transform3D global_parent,
		Vector<BoneId> apply_order) {
	if (skeleton) {
		for (int i = 0; i < apply_order.size(); i++) {
			Quaternion local_quat = ik_map[apply_order[i]];
			skeleton->set_bone_pose_rotation(apply_order[i], local_quat);
		}
	}
}

void RenIK::apply_ik_map(HashMap<BoneId, Basis> ik_map, Transform3D global_parent,
		Vector<BoneId> apply_order) {
	if (skeleton) {
		for (int i = 0; i < apply_order.size(); i++) {
			Basis local_basis = ik_map[apply_order[i]];
			skeleton->set_bone_pose_rotation(apply_order[i], local_basis.get_rotation_quaternion());
		}
	}
}

Transform3D RenIK::get_global_parent_pose(BoneId child,
		HashMap<BoneId, Quaternion> ik_map,
		Transform3D map_global_parent) {
	Transform3D full_transform;
	BoneId parent_id = skeleton->get_bone_parent(child);
	while (parent_id >= 0) {
		if (ik_map.has(parent_id)) {
			BoneId super_parent = parent_id;
			full_transform = skeleton->get_bone_rest(super_parent) *
					Transform3D(ik_map[super_parent]) * full_transform;
			while (skeleton->get_bone_parent(super_parent) >= 0) {
				super_parent = skeleton->get_bone_parent(super_parent);
				if (ik_map.has(super_parent)) {
					full_transform = skeleton->get_bone_rest(super_parent) *
							Transform3D(ik_map[super_parent]) * full_transform;
				} else {
					full_transform = map_global_parent * full_transform;
					break;
				}
			}
			return full_transform;
		}
		parent_id = skeleton->get_bone_parent(parent_id);
	}
	return Transform3D();
}

RenIK::SpineTransforms RenIK::perform_torso_ik() {
	if (head_target_spatial && skeleton && spine_chain->is_valid()) {
		Transform3D skel_inverse = skeleton->get_global_transform().affine_inverse();
		Transform3D headGlobalTransform =
				skel_inverse * head_target_spatial->get_global_transform();
		Transform3D hipGlobalTransform =
				skel_inverse * (hip_target_spatial ? hip_target_spatial->get_global_transform() : placement.interpolated_hip) *
				skeleton->get_bone_rest(hip).basis;
		Vector3 delta = hipGlobalTransform.origin +
				hipGlobalTransform.basis.xform(
						spine_chain->get_joints()[0].relative_prev) -
				headGlobalTransform.origin;
		float fullLength = spine_chain->get_total_length();
		if (delta.length() > fullLength) {
			hipGlobalTransform.set_origin(
					headGlobalTransform.origin + (delta.normalized() * fullLength) -
					hipGlobalTransform.basis.xform(
							spine_chain->get_joints()[0].relative_prev));
		}

		HashMap<BoneId, Quaternion> ik_map = solve_ik_qcp(
				spine_chain,
				hipGlobalTransform * skeleton->get_bone_rest(hip).basis.inverse(),
				headGlobalTransform);
		//skeleton->set_bone_global_pose_override(
		//    hip, hipGlobalTransform, 1.0f, true);
		skeleton->set_bone_pose_rotation(hip, hipGlobalTransform.get_basis().get_rotation_quaternion());
		skeleton->set_bone_pose_position(hip, hipGlobalTransform.get_origin());

		apply_ik_map(ik_map, hipGlobalTransform,
				bone_id_order(spine_chain));

		// Keep Hip and Head as global poses tand then apply them as global pose
		// override
		Quaternion neckQuaternion = Quaternion();
		int parent_bone = skeleton->get_bone_parent(head);
		while (parent_bone != -1) {
			neckQuaternion = skeleton->get_bone_pose_rotation(parent_bone) * neckQuaternion;
			parent_bone = skeleton->get_bone_parent(parent_bone);
		}
		//skeleton->set_bone_global_pose_override(
		//    head, headGlobalTransform, 1.0f, true);
		skeleton->set_bone_pose_rotation(head, neckQuaternion.inverse() * headGlobalTransform.get_basis().get_rotation_quaternion());

		// Calculate and return the parent bone position for the arms
		Transform3D left_global_parent_pose = Transform3D();
		Transform3D right_global_parent_pose = Transform3D();
		if (limb_arm_left != nullptr) {
			left_global_parent_pose = get_global_parent_pose(
					limb_arm_left->upper_id, ik_map, hipGlobalTransform);
		}
		if (limb_arm_right != nullptr) {
			right_global_parent_pose = get_global_parent_pose(
					limb_arm_right->upper_id, ik_map, hipGlobalTransform);
		}
		return SpineTransforms(hipGlobalTransform, left_global_parent_pose,
				right_global_parent_pose, headGlobalTransform);
	}
	return SpineTransforms();
}

void RenIK::perform_hand_left_ik(Transform3D global_parent, Transform3D target) {
	if (hand_left_target_spatial && skeleton &&
			limb_arm_left->is_valid_in_skeleton(skeleton)) {
		Transform3D root = global_parent; //  skeleton->get_global_transform() * global_parent
		BoneId rootBone =
				skeleton->get_bone_parent(limb_arm_left->get_upper_bone());
		if (rootBone >= 0) {
			if (left_shoulder_enabled) {
				root = root * skeleton->get_bone_rest(rootBone);
				Vector3 targetVector = root.affine_inverse().xform(target.origin);
				Quaternion offsetQuat = Quaternion::from_euler(left_shoulder_offset);
				Quaternion poleOffset = Quaternion::from_euler(left_shoulder_pole_offset);
				Quaternion poleOffsetScaled =
						poleOffset.slerp(Quaternion(), 1 - shoulder_influence);
				Quaternion quatAlignToTarget =
						poleOffsetScaled *
						Quaternion(
								Vector3(0, 1, 0), poleOffset.inverse().xform(offsetQuat.inverse().xform(targetVector)).normalized())
								.slerp(Quaternion(), 1 - shoulder_influence);
				Transform3D customPose =
						Transform3D(offsetQuat * quatAlignToTarget, Vector3());
				skeleton->set_bone_pose_rotation(rootBone, skeleton->get_bone_rest(rootBone).get_basis().get_rotation_quaternion() * offsetQuat * quatAlignToTarget);
				root = root * customPose;
			}
		}
		apply_ik_map(
				solve_trig_ik_redux(limb_arm_left, root, target),
				root, bone_id_order(limb_arm_left));
	}
}

void RenIK::perform_hand_right_ik(Transform3D global_parent, Transform3D target) {
	if (hand_right_target_spatial && skeleton &&
			limb_arm_right->is_valid_in_skeleton(skeleton)) {
		Transform3D root = global_parent;
		BoneId rootBone =
				skeleton->get_bone_parent(limb_arm_right->get_upper_bone());
		if (rootBone >= 0) {
			if (right_shoulder_enabled) {
				// BoneId shoulderParent = skeleton->get_bone_parent(rootBone);
				// if (shoulderParent >= 0) {
				// 	root = root * skeleton->get_bone_global_pose(shoulderParent);
				// }
				root = root * skeleton->get_bone_rest(rootBone);
				Vector3 targetVector = root.affine_inverse().xform(target.origin);
				Quaternion offsetQuat = Quaternion::from_euler(right_shoulder_offset);
				Quaternion poleOffset = Quaternion::from_euler(right_shoulder_pole_offset);
				Quaternion poleOffsetScaled =
						poleOffset.normalized().slerp(Quaternion(), 1 - shoulder_influence);
				Quaternion quatAlignToTarget =
						poleOffsetScaled *
						Quaternion(
								Vector3(0, 1, 0), poleOffset.inverse().xform(offsetQuat.inverse().xform(targetVector)))
								.normalized()
								.slerp(Quaternion(), 1 - shoulder_influence);
				Transform3D customPose =
						Transform3D(offsetQuat * quatAlignToTarget, Vector3());
				skeleton->set_bone_pose_rotation(rootBone, skeleton->get_bone_rest(rootBone).get_basis().get_rotation_quaternion() * offsetQuat * quatAlignToTarget);
				root = root * customPose;
			}
		}
		apply_ik_map(
				solve_trig_ik_redux(limb_arm_right, root, target),
				root, bone_id_order(limb_arm_right));
	}
}

void RenIK::perform_foot_left_ik(Transform3D global_parent, Transform3D target) {
	if (skeleton && limb_leg_left->is_valid_in_skeleton(skeleton)) {
		Transform3D root = global_parent;
		apply_ik_map(
				solve_trig_ik_redux(limb_leg_left, root, target),
				global_parent, bone_id_order(limb_leg_left));
	}
}

void RenIK::perform_foot_right_ik(Transform3D global_parent, Transform3D target) {
	if (skeleton && limb_leg_right->is_valid_in_skeleton(skeleton)) {
		Transform3D root = global_parent;
		// Transform3D root = skeleton->get_global_transform();
		// BoneId rootBone =
		// skeleton->get_bone_parent(limb_leg_right->get_upper_bone()); if
		// (rootBone >= 0) { 	root = root *
		// skeleton->get_bone_global_pose(rootBone);
		// }
		apply_ik_map(solve_trig_ik_redux(
							 limb_leg_right, root, target),
				global_parent, bone_id_order(limb_leg_right));
	}
}

void RenIK::reset_chain(Ref<RenIKChain> chain) {
}

void RenIK::reset_limb(Ref<RenIKLimb> limb) {
}

Vector<BoneId> RenIK::bone_id_order(Ref<RenIKChain> chain) {
	Vector<BoneId> ret;
	Vector<RenIKChain::Joint> spine_joints = spine_chain->get_joints();
	for (int i = 0; i < spine_joints.size();
			i++) { // the last one's rotation is defined by the leaf position not a
				   // joint so we skip it
		ret.push_back(spine_joints[i].id);
	}
	return ret;
}

Vector<BoneId> RenIK::bone_id_order(Ref<RenIKLimb> limb) {
	Vector<BoneId> ret;
	ret.push_back(limb->get_upper_bone());
	ret.append_array(limb->upper_extra_bone_ids);
	ret.push_back(limb->get_lower_bone());
	ret.append_array(limb->lower_extra_bone_ids);
	ret.push_back(limb->get_leaf_bone());
	return ret;
}

HashMap<BoneId, Quaternion> RenIK::solve_trig_ik(Ref<RenIKLimb> limb,
		Transform3D root,
		Transform3D target) {
	HashMap<BoneId, Quaternion> map;

	if (limb->is_valid()) { // There's no way to find a valid upperId if any of
							// the other Id's are invalid, so we only check
							// upperId
		Vector3 upperVector = limb->get_lower().get_origin();
		Vector3 lowerVector = limb->get_leaf().get_origin();
		Quaternion upperRest =
				limb->get_upper().get_basis().get_rotation_quaternion();
		Quaternion lowerRest =
				limb->get_lower().get_basis().get_rotation_quaternion();
		Quaternion upper = upperRest.inverse();
		Quaternion lower = lowerRest.inverse();
		// The true root of the limb is the point where the upper bone starts
		Transform3D trueRoot = root.translated_local(limb->get_upper().get_origin());
		Transform3D localTarget = trueRoot.affine_inverse() * target;

		// First we offset the pole
		upper = upper * Quaternion::from_euler(limb->pole_offset).normalized(); // pole_offset is a euler because
																				// that's more human readable
		upper.normalize();
		lower.normalize();
		// Then we line up the limb with our target
		Vector3 targetVector = Quaternion::from_euler(limb->pole_offset).inverse().xform(localTarget.get_origin());
		upper = upper * Quaternion(upperVector, targetVector);
		// Then we calculate how much we need to bend so we don't extend past the
		// target Law of Cosines
		float upperLength = upperVector.length();
		float lowerLength = lowerVector.length();
		float upperLength2 = upperVector.length_squared();
		float lowerLength2 = lowerVector.length_squared();
		float targetDistance = targetVector.length();
		float targetDistance2 = targetVector.length_squared();
		float upperAngle =
				RenIKHelper::safe_acos((upperLength2 + targetDistance2 - lowerLength2) /
						(2 * upperLength * targetDistance));
		float lowerAngle =
				RenIKHelper::safe_acos((upperLength2 + lowerLength2 - targetDistance2) /
						(2 * upperLength * lowerLength)) -
				Math_PI;
		Vector3 bendAxis = RenIKHelper::get_perpendicular_vector(
				upperVector); // TODO figure out how to set this automatically to the
							  // right axis
		Quaternion upperBend = Quaternion(bendAxis, upperAngle);
		Quaternion lowerBend = Quaternion(bendAxis, lowerAngle);
		upper = upper * upperBend;
		lower = lower * lowerBend;
		// Then we roll the limb based on the target position
		Vector3 targetRestPosition =
				upperVector.normalized() * (upperLength + lowerLength);
		Vector3 rollVector = upperBend.inverse().xform(upperVector).normalized();
		float positionalRollAmount =
				limb->target_position_influence.dot(targetRestPosition - targetVector);
		Quaternion positionRoll = Quaternion(rollVector, positionalRollAmount);
		upper = upper.normalized() * positionRoll;
		// And the target rotation

		Quaternion leafRest =
				limb->get_leaf().get_basis().get_rotation_quaternion();
		Quaternion armCombined =
				(upperRest * upper * lowerRest * lower).normalized();
		Quaternion targetQuat =
				localTarget.get_basis().get_rotation_quaternion() * leafRest;
		Quaternion leaf =
				((armCombined * leafRest).inverse() * targetQuat).normalized();
		// if we had a plane along the roll vector we can project the leaf and lower
		// limb on it to see which direction we need to roll to reduce the angle
		// between the two
		Vector3 restVector = (armCombined).xform(lowerVector).normalized();
		Vector3 leafVector = leaf.xform(restVector).normalized();
		Vector3 restRejection =
				RenIKHelper::vector_rejection(restVector.normalized(), rollVector);
		Vector3 leafRejection =
				RenIKHelper::vector_rejection(leafVector.normalized(), rollVector);
		float directionalRollAmount =
				RenIKHelper::safe_acos(
						restRejection.normalized().dot(leafRejection.normalized())) *
				limb->target_rotation_influence;
		Vector3 directionality =
				restRejection.normalized().cross(leafRejection.normalized());
		float check = directionality.dot(targetVector.normalized());
		if (check > 0) {
			directionalRollAmount *= -1;
		}
		Quaternion directionalRoll = Quaternion(rollVector, directionalRollAmount);
		upper = upper * directionalRoll;

		armCombined = (upperRest * upper * lowerRest * lower).normalized();
		leaf = ((armCombined * leafRest).inverse() * targetQuat).normalized();
		Vector3 twist = (leafRest * leaf).get_euler();
		Quaternion lowerTwist =
				Quaternion::from_euler((leafRest * leaf).get_euler() * lowerVector.normalized() *
						(limb->lower_limb_twist));
		lower = lower * lowerTwist;
		leaf = (lowerTwist * leafRest).inverse() * leafRest * leaf;

		Quaternion upperTwist =
				Quaternion::from_euler(twist * upperVector.normalized() *
						(limb->upper_limb_twist * limb->lower_limb_twist));
		upper = upper * upperTwist;
		lower = (upperTwist * lowerRest).inverse() * lowerRest * lower;

		// save data and return
		map.insert(limb->get_upper_bone(), upper);
		map.insert(limb->get_lower_bone(), lower);
		map.insert(limb->get_leaf_bone(), leaf);
	}
	return map;
}

std::pair<float, float> RenIK::trig_angles(Vector3 const &side1,
		Vector3 const &side2,
		Vector3 const &side3) {
	// Law of Cosines
	float length1Squared = side1.length_squared();
	float length2Squared = side2.length_squared();
	float length3Squared = side3.length_squared();
	float length1 = sqrt(length1Squared) * 2;
	float length2 = sqrt(length2Squared);
	float length3 = sqrt(length3Squared); // multiply by 2 here to save on having
										  // to multiply by 2 twice later
	float angle1 = RenIKHelper::safe_acos(
			(length1Squared + length3Squared - length2Squared) / (length1 * length3));
	float angle2 =
			Math_PI - RenIKHelper::safe_acos((length1Squared + length2Squared - length3Squared) / (length1 * length2));
	return std::make_pair(angle1, angle2);
}

HashMap<BoneId, Basis> RenIK::solve_trig_ik_redux(Ref<RenIKLimb> limb,
		Transform3D root,
		Transform3D target) {
	HashMap<BoneId, Basis> map;
	if (limb->is_valid()) {
		// The true root of the limb is the point where the upper bone starts
		Transform3D trueRoot = root.translated_local(limb->get_upper().get_origin());
		Transform3D localTarget = trueRoot.affine_inverse() * target;

		Transform3D full_upper = limb->get_upper();
		Transform3D full_lower = limb->get_lower();

		// The Triangle
		Vector3 upperVector =
				(limb->upper_extra_bones * limb->get_lower()).get_origin();
		Vector3 lowerVector =
				(limb->lower_extra_bones * limb->get_leaf()).get_origin();
		Vector3 targetVector = localTarget.get_origin();
		Vector3 normalizedTargetVector = targetVector.normalized();
		float limbLength = upperVector.length() + lowerVector.length();
		if (targetVector.length() > upperVector.length() + lowerVector.length()) {
			targetVector = normalizedTargetVector * limbLength;
		}
		std::pair<float, float> angles =
				trig_angles(upperVector, lowerVector, targetVector);

		// The local x-axis of the upper limb is axis along which the limb will bend
		// We take into account how the pole offset and alignment with the target
		// vector will affect this axis
		Vector3 startingPole = Quaternion::from_euler(limb->pole_offset).xform(Vector3(0, 1, 0)); // the opposite of this vector is where the pole is
		Vector3 jointAxis = RenIKHelper::align_vectors(startingPole, targetVector)
									.xform(Quaternion::from_euler(limb->pole_offset).xform(Vector3(1, 0, 0)));

		// //We then find how far away from the rest position the leaf is and use
		// that to change the rotational axis more.
		Vector3 leafRestVector = full_upper.get_basis().xform(
				full_lower.xform(limb->get_leaf().get_origin()));
		float positionalOffset =
				limb->target_position_influence.dot(targetVector - leafRestVector);
		jointAxis.rotate(normalizedTargetVector,
				positionalOffset + limb->roll_offset);

		// Leaf Rotations... here we go...
		// Let's always try to avoid having the leaf intersect the lowerlimb
		// First we find the a vector that corresponds with the direction the leaf
		// and lower limbs are pointing local to the true root
		Vector3 localLeafVector =
				localTarget.get_basis().xform(Vector3(0, 1, 0)); // y axis of the target
		Vector3 localLowerVector =
				normalizedTargetVector.rotated(jointAxis, angles.first - angles.second)
						.normalized();
		// We then take the vector rejections of the leaf and lower limb against the
		// target vector A rejection is the opposite of a projection. We use the
		// target vector because that's our axis of rotation for the whole limb-> We
		// then turn the whole arm along the target vector based on how close the
		// rejections are We scale the amount we rotate with the rotation influence
		// setting and the angle between the leaf and lower vector so if the arm is
		// mostly straight, we rotate less
		Vector3 leafRejection =
				RenIKHelper::vector_rejection(localLeafVector, normalizedTargetVector);
		Vector3 lowerRejection =
				RenIKHelper::vector_rejection(localLowerVector, normalizedTargetVector);
		float jointRollAmount = (leafRejection.angle_to(lowerRejection)) *
				limb->target_rotation_influence;
		jointRollAmount *= abs(
				localLeafVector.cross(localLowerVector).dot(normalizedTargetVector));
		if (leafRejection.cross(lowerRejection).dot(normalizedTargetVector) > 0) {
			jointRollAmount *= -1;
		}
		jointAxis.rotate(normalizedTargetVector, jointRollAmount);
		float totalRoll = jointRollAmount + positionalOffset + limb->roll_offset;

		// Add a little twist
		// We align the leaf's y axis with the lower limb's y-axis and see how far
		// off the x-axis is from the joint axis to calculate the twist.
		Vector3 leafX =
				Quaternion(
						localLeafVector.rotated(normalizedTargetVector, jointRollAmount),
						localLowerVector.rotated(normalizedTargetVector, jointRollAmount))
						.xform(localTarget.get_basis().xform(Vector3(1, 0, 0)));
		Vector3 rolledJointAxis = jointAxis.rotated(localLowerVector, -totalRoll);
		Vector3 lowerZ = rolledJointAxis.cross(localLowerVector);
		float twistAngle = leafX.angle_to(rolledJointAxis);
		if (leafX.dot(lowerZ) > 0) {
			twistAngle *= -1;
		}

		float inflectionPoint =
				twistAngle > 0 ? Math_PI - limb->twist_inflection_point_offset
							   : -Math_PI - limb->twist_inflection_point_offset;
		float overflowArea = limb->overflow_state * limb->twist_overflow;
		float inflectionDistance = twistAngle - inflectionPoint;

		if (Math::abs(inflectionDistance) < limb->twist_overflow) {
			if (limb->overflow_state == 0) {
				limb->overflow_state = inflectionDistance < 0 ? 1 : -1;
			}
		} else {
			limb->overflow_state = 0;
		}

		inflectionPoint += overflowArea;
		if (twistAngle > 0 && twistAngle > inflectionPoint) {
			twistAngle -= Math_TAU; // Change to complement angle
		} else if (twistAngle < 0 && twistAngle < inflectionPoint) {
			twistAngle += Math_TAU; // Change to complement angle
		}

		float lowerTwist = twistAngle * limb->lower_limb_twist;
		float upperTwist = lowerTwist * limb->upper_limb_twist +
				limb->upper_twist_offset - totalRoll;
		lowerTwist += limb->lower_twist_offset - 2 * limb->roll_offset -
				positionalOffset - jointRollAmount;

		jointAxis.rotate(normalizedTargetVector,
				twistAngle * limb->target_rotation_influence);

		// Rebuild the rotations
		Vector3 upperJointVector =
				normalizedTargetVector.rotated(jointAxis, angles.first);
		Vector3 rolledLowerJointAxis =
				Vector3(1, 0, 0).rotated(Vector3(0, 1, 0), -limb->roll_offset);
		Vector3 lowerJointVector =
				Vector3(0, 1, 0).rotated(rolledLowerJointAxis, angles.second);
		Vector3 twistedJointAxis = jointAxis.rotated(upperJointVector, upperTwist);
		Basis upperBasis = Basis(twistedJointAxis, upperJointVector,
				twistedJointAxis.cross(upperJointVector));
		Basis lowerBasis = Basis(rolledLowerJointAxis, lowerJointVector,
				rolledLowerJointAxis.cross(lowerJointVector));
		lowerBasis.transpose();
		lowerBasis.rotate_local(Vector3(0, 1, 0), lowerTwist);
		lowerBasis.rotate(Vector3(0, 1, 0), -upperTwist);

		Basis upperTransform =
				((full_upper.get_basis().inverse() * upperBasis).orthonormalized());
		Basis lowerTransform =
				((full_lower.get_basis().inverse() * lowerBasis).orthonormalized());
		Basis leafTransform =
				(limb->get_leaf().get_basis().inverse() *
						(upperBasis * lowerBasis).inverse() * localTarget.get_basis() *
						limb->get_leaf().get_basis());
		map[limb->get_upper_bone()] = upperTransform;
		for (int i = 0; i < limb->upper_extra_bone_ids.size(); i++) {
			map[limb->upper_extra_bone_ids[i]] = Basis();
		}

		map[limb->get_lower_bone()] =
				lowerTransform;
		for (int i = 0; i < limb->lower_extra_bone_ids.size(); i++) {
			map[limb->lower_extra_bone_ids[i]] = Basis();
		}

		map[limb->get_leaf_bone()] = leafTransform;
	}
	return map;
}

Vector<BoneId> RenIK::calculate_bone_chain(BoneId root, BoneId leaf) {
	Vector<BoneId> chain;
	BoneId b = leaf;
	chain.push_back(b);
	if (skeleton) {
		while (b >= 0 && b != root) {
			b = skeleton->get_bone_parent(b);
			chain.push_back(b);
		}
		if (b < 0) {
			chain.clear();
			chain.push_back(leaf);
		} else {
			chain.reverse();
		}
	}
	return chain;
}

bool RenIK::get_live_preview() { return live_preview; }
void RenIK::set_live_preview(bool p_enable) {
	live_preview = p_enable;
}

void RenIK::reset_all_bones() {
}

// NodePath RenIK::get_skeleton_path() {
// 	if (skeleton) {
// 		return skeleton->get_path();
// 	}
// 	return skeleton_path;
// }
// void RenIK::set_skeleton_path(NodePath p_path) {
// 	skeleton_path = p_path;
// 	if (is_inside_tree()) {
// 		Skeleton3D *new_node =
// 				Object::cast_to<Skeleton3D>(get_node_or_null(p_path));
// 		set_skeleton(new_node);
// 	}
// }
void RenIK::set_skeleton(Node *p_node) {
	Skeleton3D *new_node = Object::cast_to<Skeleton3D>(p_node);
	if (new_node != nullptr) {
		//skeleton_path = get_path_to(new_node);

		if (skeleton != nullptr) {
			reset_all_bones();
		}

		skeleton = new_node;
		set_head_bone_by_name(get_head_bone_name());
		set_hip_bone_by_name(get_hip_bone_name());

		set_upper_arm_left_bone_by_name(get_upper_arm_left_bone_name());
		set_lower_arm_left_bone_by_name(get_lower_arm_left_bone_name());
		set_hand_left_bone_by_name(get_hand_left_bone_name());

		set_upper_arm_right_bone_by_name(get_upper_arm_right_bone_name());
		set_lower_arm_right_bone_by_name(get_lower_arm_right_bone_name());
		set_hand_right_bone_by_name(get_hand_right_bone_name());

		set_upper_leg_left_bone_by_name(get_upper_leg_left_bone_name());
		set_lower_leg_left_bone_by_name(get_lower_leg_left_bone_name());
		set_foot_left_bone_by_name(get_foot_left_bone_name());

		set_upper_leg_right_bone_by_name(get_upper_leg_right_bone_name());
		set_lower_leg_right_bone_by_name(get_lower_leg_right_bone_name());
		set_foot_right_bone_by_name(get_foot_right_bone_name());
	}
}

void RenIK::set_head_bone_by_name(String p_bone) {
	head_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);

		set_head_bone(id);
	}
}
void RenIK::set_hand_left_bone_by_name(String p_bone) {
	hand_left_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);

		set_hand_left_bone(id);
	}
}
void RenIK::set_lower_arm_left_bone_by_name(String p_bone) {
	lower_left_arm_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);
		set_lower_arm_left_bone(id);
	}
}
void RenIK::set_upper_arm_left_bone_by_name(String p_bone) {
	upper_left_arm_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);
		set_upper_arm_left_bone(id);
	}
}
void RenIK::set_hand_right_bone_by_name(String p_bone) {
	hand_right_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);

		set_hand_right_bone(id);
	}
}
void RenIK::set_lower_arm_right_bone_by_name(String p_bone) {
	lower_right_arm_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);
		set_lower_arm_right_bone(id);
	}
}
void RenIK::set_upper_arm_right_bone_by_name(String p_bone) {
	upper_right_arm_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);
		set_upper_arm_right_bone(id);
	}
}
void RenIK::set_hip_bone_by_name(String p_bone) {
	hip_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);

		set_hip_bone(id);
	}
}
void RenIK::set_foot_left_bone_by_name(String p_bone) {
	foot_left_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);

		set_foot_left_bone(id);
	}
}
void RenIK::set_lower_leg_left_bone_by_name(String p_bone) {
	lower_left_leg_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);
		set_lower_leg_left_bone(id);
	}
}
void RenIK::set_upper_leg_left_bone_by_name(String p_bone) {
	upper_left_leg_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);
		set_upper_leg_left_bone(id);
	}
}
void RenIK::set_foot_right_bone_by_name(String p_bone) {
	foot_right_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);

		set_foot_right_bone(id);
	}
}
void RenIK::set_lower_leg_right_bone_by_name(String p_bone) {
	lower_right_leg_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);
		set_lower_leg_right_bone(id);
	}
}
void RenIK::set_upper_leg_right_bone_by_name(String p_bone) {
	upper_right_leg_bone_name = p_bone;
	if (skeleton) {
		BoneId id = skeleton->find_bone(p_bone);
		set_upper_leg_right_bone(id);
	}
}

void RenIK::calculate_hip_offset() {
	placement.spine_length = spine_chain->get_total_length();
	// calc rest offset of hips
	if (skeleton && head >= 0 && head < skeleton->get_bone_count() && hip >= 0 &&
			hip < skeleton->get_bone_count()) {
		Transform3D delta = skeleton->get_bone_rest(head);
		BoneId bone_parent = skeleton->get_bone_parent(head);
		while (bone_parent != hip && bone_parent >= 0) {
			delta = skeleton->get_bone_rest(bone_parent) * delta;
			bone_parent = skeleton->get_bone_parent(bone_parent);
		}
		while (bone_parent >= 0) {
			delta = Transform3D(skeleton->get_bone_rest(bone_parent).basis) * delta;
			bone_parent = skeleton->get_bone_parent(bone_parent);
		}
		placement.hip_offset = -delta.origin;
	}
}

void RenIK::set_head_bone(BoneId p_bone) {
	head = p_bone;
	spine_chain->set_leaf_bone(skeleton, p_bone);
	calculate_hip_offset();
}

void RenIK::set_hand_left_bone(BoneId p_bone) {
	limb_arm_left->set_leaf(skeleton, p_bone);
	if (lower_left_arm_bone_name.is_empty() && skeleton) {
		if (p_bone >= 0) {
			set_lower_arm_left_bone(skeleton->get_bone_parent(p_bone));
		}
	}
	left_shoulder_enabled =
			skeleton && limb_arm_left->is_valid_in_skeleton(skeleton) &&
			!spine_chain->contains_bone(
					skeleton, skeleton->get_bone_parent(limb_arm_left->get_lower_bone()));
}

void RenIK::set_lower_arm_left_bone(BoneId p_bone) {
	limb_arm_left->set_lower(skeleton, p_bone);
	if (lower_left_arm_bone_name.is_empty() && skeleton) {
		if (p_bone >= 0) {
			set_upper_arm_left_bone(skeleton->get_bone_parent(p_bone));
		}
	}
	left_shoulder_enabled =
			skeleton && limb_arm_left->is_valid_in_skeleton(skeleton) &&
			!spine_chain->contains_bone(
					skeleton, skeleton->get_bone_parent(limb_arm_left->get_lower_bone()));
}

void RenIK::set_upper_arm_left_bone(BoneId p_bone) {
	limb_arm_left->set_upper(skeleton, p_bone);
	left_shoulder_enabled =
			skeleton && limb_arm_left->is_valid_in_skeleton(skeleton) &&
			!spine_chain->contains_bone(
					skeleton, skeleton->get_bone_parent(limb_arm_left->get_lower_bone()));
}

void RenIK::set_hand_right_bone(BoneId p_bone) {
	limb_arm_right->set_leaf(skeleton, p_bone);
	if (lower_right_arm_bone_name.is_empty() && skeleton) {
		if (p_bone >= 0) {
			set_lower_arm_right_bone(skeleton->get_bone_parent(p_bone));
		}
	}
	right_shoulder_enabled =
			skeleton && limb_arm_right->is_valid_in_skeleton(skeleton) &&
			!spine_chain->contains_bone(
					skeleton,
					skeleton->get_bone_parent(limb_arm_right->get_lower_bone()));
}

void RenIK::set_lower_arm_right_bone(BoneId p_bone) {
	limb_arm_right->set_lower(skeleton, p_bone);
	if (upper_right_arm_bone_name.is_empty() && skeleton) {
		if (p_bone >= 0) {
			set_upper_arm_right_bone(skeleton->get_bone_parent(p_bone));
		}
	}
	right_shoulder_enabled =
			skeleton && limb_arm_right->is_valid_in_skeleton(skeleton) &&
			!spine_chain->contains_bone(
					skeleton,
					skeleton->get_bone_parent(limb_arm_right->get_lower_bone()));
}

void RenIK::set_upper_arm_right_bone(BoneId p_bone) {
	limb_arm_right->set_upper(skeleton, p_bone);
	right_shoulder_enabled =
			skeleton && limb_arm_right->is_valid_in_skeleton(skeleton) &&
			!spine_chain->contains_bone(
					skeleton,
					skeleton->get_bone_parent(limb_arm_right->get_lower_bone()));
}

void RenIK::set_hip_bone(BoneId p_bone) {
	hip = p_bone;
	spine_chain->set_root_bone(skeleton, p_bone);
	if (spine_chain->is_valid()) {
		calculate_hip_offset();
	}
}

void RenIK::set_foot_left_bone(BoneId p_bone) {
	limb_leg_left->set_leaf(skeleton, p_bone);
	if (lower_left_leg_bone_name.is_empty() && skeleton) {
		if (p_bone >= 0) {
			set_lower_leg_left_bone(skeleton->get_bone_parent(p_bone));
		}
	}
	placement.left_leg_length = limb_leg_left->is_valid_in_skeleton(skeleton)
			? limb_leg_left->lower.origin.length() +
					limb_leg_left->leaf.origin.length()
			: 0;
	placement.left_hip_offset = limb_leg_left->is_valid_in_skeleton(skeleton)
			? limb_leg_left->upper.origin
			: Vector3();
}

void RenIK::set_lower_leg_left_bone(BoneId p_bone) {
	limb_leg_left->set_lower(skeleton, p_bone);
	if (upper_left_leg_bone_name.is_empty() && skeleton) {
		if (p_bone >= 0) {
			set_upper_leg_left_bone(skeleton->get_bone_parent(p_bone));
		}
	}
	placement.left_leg_length = limb_leg_left->is_valid_in_skeleton(skeleton)
			? limb_leg_left->lower.origin.length() +
					limb_leg_left->leaf.origin.length()
			: 0;
	placement.left_hip_offset = limb_leg_left->is_valid_in_skeleton(skeleton)
			? limb_leg_left->upper.origin
			: Vector3();
}

void RenIK::set_upper_leg_left_bone(BoneId p_bone) {
	limb_leg_left->set_upper(skeleton, p_bone);
	placement.left_leg_length = limb_leg_left->is_valid_in_skeleton(skeleton)
			? limb_leg_left->lower.origin.length() +
					limb_leg_left->leaf.origin.length()
			: 0;
	placement.left_hip_offset = limb_leg_left->is_valid_in_skeleton(skeleton)
			? limb_leg_left->upper.origin
			: Vector3();
}

void RenIK::set_foot_right_bone(BoneId p_bone) {
	limb_leg_right->set_leaf(skeleton, p_bone);
	if (lower_right_leg_bone_name.is_empty() && skeleton) {
		if (p_bone >= 0) {
			set_lower_leg_right_bone(skeleton->get_bone_parent(p_bone));
		}
	}
	placement.right_leg_length = limb_leg_right->is_valid_in_skeleton(skeleton)
			? limb_leg_right->lower.origin.length() +
					limb_leg_right->leaf.origin.length()
			: 0;
	placement.right_hip_offset = limb_leg_right->is_valid_in_skeleton(skeleton)
			? limb_leg_right->upper.origin
			: Vector3();
}

void RenIK::set_lower_leg_right_bone(BoneId p_bone) {
	limb_leg_right->set_lower(skeleton, p_bone);
	if (upper_right_leg_bone_name.is_empty() && skeleton) {
		if (p_bone >= 0) {
			set_upper_leg_right_bone(skeleton->get_bone_parent(p_bone));
		}
	}
	placement.right_leg_length = limb_leg_right->is_valid_in_skeleton(skeleton)
			? limb_leg_right->lower.origin.length() +
					limb_leg_right->leaf.origin.length()
			: 0;
	placement.right_hip_offset = limb_leg_right->is_valid_in_skeleton(skeleton)
			? limb_leg_right->upper.origin
			: Vector3();
}

void RenIK::set_upper_leg_right_bone(BoneId p_bone) {
	limb_leg_right->set_upper(skeleton, p_bone);
	placement.right_leg_length = limb_leg_right->is_valid_in_skeleton(skeleton)
			? limb_leg_right->lower.origin.length() +
					limb_leg_right->leaf.origin.length()
			: 0;
	placement.right_hip_offset = limb_leg_right->is_valid_in_skeleton(skeleton)
			? limb_leg_right->upper.origin
			: Vector3();
}

int64_t RenIK::get_hip_bone() { return hip; }
int64_t RenIK::get_head_bone() { return head; }
int64_t RenIK::get_hand_left_bone() { return limb_arm_left->get_leaf_bone(); }
int64_t RenIK::get_lower_arm_left_bone() {
	return limb_arm_left->get_lower_bone();
}
int64_t RenIK::get_upper_arm_left_bone() {
	return limb_arm_left->get_upper_bone();
}
int64_t RenIK::get_hand_right_bone() { return limb_arm_right->get_leaf_bone(); }
int64_t RenIK::get_lower_arm_right_bone() {
	return limb_arm_right->get_lower_bone();
}
int64_t RenIK::get_upper_arm_right_bone() {
	return limb_arm_right->get_upper_bone();
}
int64_t RenIK::get_foot_left_bone() { return limb_leg_left->get_leaf_bone(); }
int64_t RenIK::get_lower_leg_left_bone() {
	return limb_leg_left->get_lower_bone();
}
int64_t RenIK::get_upper_leg_left_bone() {
	return limb_leg_left->get_upper_bone();
}
int64_t RenIK::get_foot_right_bone() { return limb_leg_right->get_leaf_bone(); }
int64_t RenIK::get_lower_leg_right_bone() {
	return limb_leg_right->get_lower_bone();
}
int64_t RenIK::get_upper_leg_right_bone() {
	return limb_leg_right->get_upper_bone();
}

String RenIK::get_hip_bone_name() {
	String l_hip_bone_name = hip_bone_name;
	if (skeleton != nullptr && !spine_chain.is_null() && spine_chain.is_valid() &&
			spine_chain->get_root_bone() >= 0) {
		l_hip_bone_name = skeleton->get_bone_name(spine_chain->get_root_bone());
	}
	return l_hip_bone_name;
}
String RenIK::get_head_bone_name() {
	String l_head_bone_name = head_bone_name;
	if (skeleton != nullptr && !spine_chain.is_null() && spine_chain.is_valid() &&
			spine_chain->get_leaf_bone() >= 0) {
		l_head_bone_name = skeleton->get_bone_name(spine_chain->get_leaf_bone());
	}
	return l_head_bone_name;
}
String RenIK::get_hand_left_bone_name() {
	String l_hand_left_bone_name = hand_left_bone_name;
	if (skeleton != nullptr && limb_arm_left->leaf_id >= 0) {
		l_hand_left_bone_name = skeleton->get_bone_name(limb_arm_left->leaf_id);
	}
	return l_hand_left_bone_name;
}
String RenIK::get_lower_arm_left_bone_name() {
	String l_lower_left_arm_bone_name = lower_left_arm_bone_name;
	if (skeleton != nullptr && limb_arm_left->lower_id >= 0) {
		l_lower_left_arm_bone_name =
				skeleton->get_bone_name(limb_arm_left->lower_id);
	}
	return l_lower_left_arm_bone_name;
}
String RenIK::get_upper_arm_left_bone_name() {
	String l_upper_left_arm_bone_name = upper_left_arm_bone_name;
	if (skeleton != nullptr && limb_arm_left->upper_id >= 0) {
		l_upper_left_arm_bone_name =
				skeleton->get_bone_name(limb_arm_left->upper_id);
	}
	return l_upper_left_arm_bone_name;
}
String RenIK::get_hand_right_bone_name() {
	String l_hand_right_bone_name = hand_right_bone_name;
	if (skeleton != nullptr && limb_arm_right->leaf_id >= 0) {
		l_hand_right_bone_name = skeleton->get_bone_name(limb_arm_right->leaf_id);
	}
	return l_hand_right_bone_name;
}
String RenIK::get_lower_arm_right_bone_name() {
	String l_lower_right_arm_bone_name = lower_right_arm_bone_name;
	if (skeleton != nullptr && limb_arm_right->lower_id >= 0) {
		l_lower_right_arm_bone_name =
				skeleton->get_bone_name(limb_arm_right->lower_id);
	}
	return l_lower_right_arm_bone_name;
}
String RenIK::get_upper_arm_right_bone_name() {
	String l_upper_right_arm_bone_name = upper_right_arm_bone_name;
	if (skeleton != nullptr && limb_arm_right->upper_id >= 0) {
		l_upper_right_arm_bone_name =
				skeleton->get_bone_name(limb_arm_right->upper_id);
	}
	return l_upper_right_arm_bone_name;
}
String RenIK::get_foot_left_bone_name() {
	String l_foot_left_bone_name = foot_left_bone_name;
	if (skeleton != nullptr && limb_leg_left->leaf_id >= 0) {
		l_foot_left_bone_name = skeleton->get_bone_name(limb_leg_left->leaf_id);
	}
	return l_foot_left_bone_name;
}
String RenIK::get_lower_leg_left_bone_name() {
	String l_lower_left_leg_bone_name = lower_left_leg_bone_name;
	if (skeleton != nullptr && limb_leg_left->lower_id >= 0) {
		l_lower_left_leg_bone_name =
				skeleton->get_bone_name(limb_leg_left->lower_id);
	}
	return l_lower_left_leg_bone_name;
}
String RenIK::get_upper_leg_left_bone_name() {
	String l_upper_left_leg_bone_name = upper_left_leg_bone_name;
	if (skeleton != nullptr && limb_leg_left->upper_id >= 0) {
		l_upper_left_leg_bone_name =
				skeleton->get_bone_name(limb_leg_left->upper_id);
	}
	return l_upper_left_leg_bone_name;
}
String RenIK::get_foot_right_bone_name() {
	String l_foot_right_bone_name = foot_right_bone_name;
	if (skeleton != nullptr && limb_leg_right->leaf_id >= 0) {
		l_foot_right_bone_name = skeleton->get_bone_name(limb_leg_right->leaf_id);
	}
	return l_foot_right_bone_name;
}
String RenIK::get_lower_leg_right_bone_name() {
	String l_lower_right_leg_bone_name = lower_right_leg_bone_name;
	if (skeleton != nullptr && limb_leg_right->lower_id >= 0) {
		l_lower_right_leg_bone_name =
				skeleton->get_bone_name(limb_leg_right->lower_id);
	}
	return l_lower_right_leg_bone_name;
}
String RenIK::get_upper_leg_right_bone_name() {
	String l_upper_right_leg_bone_name = upper_right_leg_bone_name;
	if (skeleton != nullptr && limb_leg_right->upper_id >= 0) {
		l_upper_right_leg_bone_name =
				skeleton->get_bone_name(limb_leg_right->upper_id);
	}
	return l_upper_right_leg_bone_name;
}

NodePath RenIK::get_head_target_path() { return head_target_path; }
void RenIK::set_head_target_path(NodePath p_path) {
	head_target_path = p_path;
	if(skeleton == nullptr){
		return;
	}
	//if (is_inside_tree()) {
		Node3D *new_node = Object::cast_to<Node3D>(skeleton->get_owner()-> get_node_or_null(p_path));
		if (new_node || p_path.is_empty()) {
			head_target_spatial = new_node;
		}
	//}
}

NodePath RenIK::get_hand_left_target_path() { return hand_left_target_path; }

void RenIK::set_hand_left_target_path(NodePath p_path) {
	hand_left_target_path = p_path;
	if(skeleton == nullptr){
		return;
	}
	//if (is_inside_tree()) {
		Node3D *new_node = Object::cast_to<Node3D>(skeleton->get_owner()-> get_node_or_null(p_path));
		if (new_node || p_path.is_empty()) {
			hand_left_target_spatial = new_node;
		}
	//}
}

NodePath RenIK::get_hand_right_target_path() { return hand_right_target_path; }

void RenIK::set_hand_right_target_path(NodePath p_path) {
	hand_right_target_path = p_path;
	if(skeleton == nullptr){
		return;
	}
	//if (is_inside_tree()) {
		Node3D *new_node = Object::cast_to<Node3D>(skeleton->get_owner()-> get_node_or_null(p_path));
		if (new_node || p_path.is_empty()) {
			hand_right_target_spatial = new_node;
		}
	//}
}

NodePath RenIK::get_hip_target_path() { return hip_target_path; }

void RenIK::set_hip_target_path(NodePath p_path) {
	hip_target_path = p_path;
	if(skeleton == nullptr){
		return;
	}
	//if (is_inside_tree()) {
		Node3D *new_node = Object::cast_to<Node3D>(skeleton->get_owner()->get_node_or_null(p_path));
		if (new_node || p_path.is_empty()) {
			hip_target_spatial = new_node;
		}
	//}
}

NodePath RenIK::get_foot_left_target_path() { return foot_left_target_path; }

void RenIK::set_foot_left_target_path(NodePath p_path) {
	foot_left_target_path = p_path;
	if(skeleton == nullptr){
		return;
	}
	//if (is_inside_tree()) {
		Node3D *new_node = Object::cast_to<Node3D>(skeleton->get_owner()->get_node_or_null(p_path));
		if (new_node || p_path.is_empty()) {
			foot_left_target_spatial = new_node;
		}
	//}
}

NodePath RenIK::get_foot_right_target_path() { return foot_right_target_path; }

void RenIK::set_foot_right_target_path(NodePath p_path) {
	foot_right_target_path = p_path;
	if(skeleton == nullptr){
		return;
	}
	//if (is_inside_tree()) {
		Node3D *new_node = Object::cast_to<Node3D>(skeleton->get_owner()->get_node_or_null(p_path));
		if (new_node || p_path.is_empty()) {
			foot_right_target_spatial = new_node;
		}
	//}
}
float RenIK::get_arm_upper_twist_offset() {
	return Math::rad_to_deg(limb_arm_left->upper_twist_offset);
}
void RenIK::set_arm_upper_twist_offset(float degrees) {
	limb_arm_left->upper_twist_offset = Math::deg_to_rad(degrees);
	limb_arm_right->upper_twist_offset = Math::deg_to_rad(-degrees);
}
float RenIK::get_arm_lower_twist_offset() {
	return Math::rad_to_deg(limb_arm_left->lower_twist_offset);
}
void RenIK::set_arm_lower_twist_offset(float degrees) {
	limb_arm_left->lower_twist_offset = Math::deg_to_rad(degrees);
	limb_arm_right->lower_twist_offset = Math::deg_to_rad(-degrees);
}
float RenIK::get_arm_roll_offset() {
	return Math::rad_to_deg(limb_arm_left->roll_offset);
}
void RenIK::set_arm_roll_offset(float degrees) {
	limb_arm_left->roll_offset = Math::deg_to_rad(degrees);
	limb_arm_right->roll_offset = Math::deg_to_rad(-degrees);
}
float RenIK::get_arm_upper_limb_twist() {
	return limb_arm_left->upper_limb_twist * 100;
}
void RenIK::set_arm_upper_limb_twist(float ratio) {
	limb_arm_left->upper_limb_twist = ratio / 100.0;
	limb_arm_right->upper_limb_twist = ratio / 100.0;
}
float RenIK::get_arm_lower_limb_twist() {
	return limb_arm_left->lower_limb_twist * 100;
}
void RenIK::set_arm_lower_limb_twist(float ratio) {
	limb_arm_left->lower_limb_twist = ratio / 100.0;
	limb_arm_right->lower_limb_twist = ratio / 100.0;
}
float RenIK::get_arm_twist_inflection_point_offset() {
	return Math::rad_to_deg(limb_arm_left->twist_inflection_point_offset);
}
void RenIK::set_arm_twist_inflection_point_offset(float degrees) {
	limb_arm_left->twist_inflection_point_offset = Math::deg_to_rad(degrees);
	limb_arm_right->twist_inflection_point_offset = Math::deg_to_rad(-degrees);
}
float RenIK::get_arm_twist_overflow() {
	return Math::rad_to_deg(limb_arm_left->twist_overflow);
}
void RenIK::set_arm_twist_overflow(float degrees) {
	limb_arm_left->twist_overflow = Math::deg_to_rad(degrees);
	limb_arm_right->twist_overflow = Math::deg_to_rad(degrees);
}

Vector3 RenIK::get_arm_pole_offset() {
	Vector3 v = limb_arm_left->pole_offset;
	return Vector3(Math::rad_to_deg(v[0]), Math::rad_to_deg(v[1]), Math::rad_to_deg(v[2]));
}
void RenIK::set_arm_pole_offset(Vector3 euler) {
	limb_arm_left->pole_offset = Vector3(Math::deg_to_rad(euler[0]), Math::deg_to_rad(euler[1]),
			Math::deg_to_rad(euler[2]));
	limb_arm_right->pole_offset = Vector3(Math::deg_to_rad(euler[0]), Math::deg_to_rad(-euler[1]),
			Math::deg_to_rad(-euler[2]));
}
Vector3 RenIK::get_arm_target_position_influence() {
	return limb_arm_left->target_position_influence * 10.0;
}
void RenIK::set_arm_target_position_influence(Vector3 xyz) {
	limb_arm_left->target_position_influence = xyz / 10.0;
	limb_arm_right->target_position_influence =
			Vector3(xyz[0], -xyz[1], -xyz[2]) / 10.0;
}
float RenIK::get_arm_target_rotation_influence() {
	return limb_arm_left->target_rotation_influence * 100.0;
}
void RenIK::set_arm_target_rotation_influence(float influence) {
	limb_arm_left->target_rotation_influence = influence / 100.0;
	limb_arm_right->target_rotation_influence = influence / 100.0;
}

float RenIK::get_leg_upper_twist_offset() {
	return Math::rad_to_deg(limb_leg_left->upper_twist_offset);
}
void RenIK::set_leg_upper_twist_offset(float degrees) {
	limb_leg_left->upper_twist_offset = Math::deg_to_rad(degrees);
	limb_leg_right->upper_twist_offset = Math::deg_to_rad(-degrees);
}
float RenIK::get_leg_lower_twist_offset() {
	return Math::rad_to_deg(limb_leg_left->lower_twist_offset);
}
void RenIK::set_leg_lower_twist_offset(float degrees) {
	limb_leg_left->lower_twist_offset = Math::deg_to_rad(degrees);
	limb_leg_right->lower_twist_offset = Math::deg_to_rad(-degrees);
}
float RenIK::get_leg_roll_offset() {
	return Math::rad_to_deg(limb_leg_left->roll_offset);
}
void RenIK::set_leg_roll_offset(float degrees) {
	limb_leg_left->roll_offset = Math::deg_to_rad(degrees);
	limb_leg_right->roll_offset = Math::deg_to_rad(-degrees);
}
float RenIK::get_leg_upper_limb_twist() {
	return limb_leg_left->upper_limb_twist * 100;
}
void RenIK::set_leg_upper_limb_twist(float ratio) {
	limb_leg_left->upper_limb_twist = ratio / 100.0;
	limb_leg_right->upper_limb_twist = ratio / 100.0;
}
float RenIK::get_leg_lower_limb_twist() {
	return limb_leg_left->lower_limb_twist * 100;
}
void RenIK::set_leg_lower_limb_twist(float ratio) {
	limb_leg_left->lower_limb_twist = ratio / 100.0;
	limb_leg_right->lower_limb_twist = ratio / 100.0;
}
float RenIK::get_leg_twist_inflection_point_offset() {
	return Math::rad_to_deg(limb_leg_left->twist_inflection_point_offset);
}
void RenIK::set_leg_twist_inflection_point_offset(float degrees) {
	limb_leg_left->twist_inflection_point_offset = Math::deg_to_rad(degrees);
	limb_leg_right->twist_inflection_point_offset = Math::deg_to_rad(-degrees);
}
float RenIK::get_leg_twist_overflow() {
	return Math::rad_to_deg(limb_leg_left->twist_overflow);
}
void RenIK::set_leg_twist_overflow(float degrees) {
	limb_leg_left->twist_overflow = Math::deg_to_rad(degrees);
	limb_leg_right->twist_overflow = Math::deg_to_rad(degrees);
}

Vector3 RenIK::get_leg_pole_offset() {
	Vector3 v = limb_leg_left->pole_offset;
	return Vector3(Math::rad_to_deg(v[0]), Math::rad_to_deg(v[1]), Math::rad_to_deg(v[2]));
}

void RenIK::set_leg_pole_offset(Vector3 euler) {
	limb_leg_left->pole_offset = Vector3(Math::deg_to_rad(euler[0]), Math::deg_to_rad(euler[1]),
			Math::deg_to_rad(euler[2]));
	limb_leg_right->pole_offset = Vector3(Math::deg_to_rad(euler[0]), Math::deg_to_rad(-euler[1]),
			Math::deg_to_rad(-euler[2]));
}

Vector3 RenIK::get_leg_target_position_influence() {
	return limb_leg_left->target_position_influence * 10.0;
}

void RenIK::set_leg_target_position_influence(Vector3 xyz) {
	limb_leg_left->target_position_influence = xyz / 10.0;
	limb_leg_right->target_position_influence =
			Vector3(xyz[0], -xyz[1], -xyz[2]) / 10.0;
}

float RenIK::get_leg_target_rotation_influence() {
	return limb_leg_left->target_rotation_influence * 100.0;
}
void RenIK::set_leg_target_rotation_influence(float influence) {
	limb_leg_left->target_rotation_influence = influence / 100.0;
	limb_leg_right->target_rotation_influence = influence / 100.0;
}

Vector3 RenIK::get_spine_curve() { return spine_chain->chain_curve_direction; }
void RenIK::set_spine_curve(Vector3 direction) {
	spine_chain->chain_curve_direction = direction;
}
float RenIK::get_upper_spine_stiffness() {
	return spine_chain->get_leaf_stiffness() * 100.0;
}
void RenIK::set_upper_spine_stiffness(float influence) {
	spine_chain->set_leaf_stiffness(skeleton, influence / 100.0);
}
float RenIK::get_lower_spine_stiffness() {
	return spine_chain->get_root_stiffness() * 100.0;
}
void RenIK::set_lower_spine_stiffness(float influence) {
	spine_chain->set_root_stiffness(skeleton, influence / 100.0);
}
float RenIK::get_spine_twist() { return spine_chain->get_twist() * 100.0; }
void RenIK::set_spine_twist(float influence) {
	spine_chain->set_twist(skeleton, influence / 100.0);
}
float RenIK::get_spine_twist_start() {
	return spine_chain->get_twist_start() * 100.0;
}
void RenIK::set_spine_twist_start(float influence) {
	spine_chain->set_twist_start(skeleton, influence / 100.0);
}

float RenIK::get_shoulder_influence() { return shoulder_influence * 100.0; }
void RenIK::set_shoulder_influence(float influence) {
	shoulder_influence = influence / 100;
}

Vector3 RenIK::get_shoulder_offset() {
	return Vector3(Math::rad_to_deg(left_shoulder_offset[0]),
			Math::rad_to_deg(left_shoulder_offset[1]),
			Math::rad_to_deg(left_shoulder_offset[2]));
}

void RenIK::set_shoulder_offset(Vector3 euler) {
	left_shoulder_offset =
			Vector3(Math::deg_to_rad(euler[0]), Math::deg_to_rad(euler[1]),
					Math::deg_to_rad(euler[2]));
	right_shoulder_offset =
			Vector3(Math::deg_to_rad(euler[0]), -Math::deg_to_rad(euler[1]),
					-Math::deg_to_rad(euler[2]));
}

Vector3 RenIK::get_shoulder_pole_offset() {
	return Vector3(Math::rad_to_deg(left_shoulder_pole_offset[0]),
			Math::rad_to_deg(left_shoulder_pole_offset[1]),
			Math::rad_to_deg(left_shoulder_pole_offset[2]));
}

void RenIK::set_shoulder_pole_offset(Vector3 euler) {
	left_shoulder_pole_offset =
			Vector3(Math::deg_to_rad(euler[0]), Math::deg_to_rad(euler[1]),
					Math::deg_to_rad(euler[2]));
	right_shoulder_pole_offset =
			Vector3(Math::deg_to_rad(euler[0]), -Math::deg_to_rad(euler[1]),
					-Math::deg_to_rad(euler[2]));
}

// Placement
void RenIK::set_falling(bool p_value) { placement.set_falling(p_value); }

void RenIK::set_collision_mask(uint32_t p_mask) {
	placement.set_collision_mask(p_mask);
}

uint32_t RenIK::get_collision_mask() const {
	return placement.get_collision_mask();
}

void RenIK::set_collision_mask_bit(int p_bit, bool p_value) {
	placement.set_collision_mask_bit(p_bit, p_value);
}

bool RenIK::get_collision_mask_bit(int p_bit) const {
	return placement.get_collision_mask_bit(p_bit);
}

void RenIK::set_collide_with_areas(bool p_clip) {
	placement.set_collide_with_areas(p_clip);
}

bool RenIK::is_collide_with_areas_enabled() const {
	return placement.is_collide_with_areas_enabled();
}

void RenIK::set_collide_with_bodies(bool p_clip) {
	placement.set_collide_with_bodies(p_clip);
}

bool RenIK::is_collide_with_bodies_enabled() const {
	return placement.is_collide_with_bodies_enabled();
}

void RenIK::set_forward_speed_scalar_min(float speed_scalar_min) {
	placement.forward_gait.speed_scalar_min = speed_scalar_min / 100.0;
}
float RenIK::get_forward_speed_scalar_min() const {
	return placement.forward_gait.speed_scalar_min * 100.0;
}
void RenIK::set_forward_speed_scalar_max(float speed_scalar_max) {
	placement.forward_gait.speed_scalar_max = speed_scalar_max / 100.0;
}
float RenIK::get_forward_speed_scalar_max() const {
	return placement.forward_gait.speed_scalar_max * 100.0;
}

void RenIK::set_forward_ground_time(float ground_time) {
	placement.forward_gait.ground_time = ground_time;
}
float RenIK::get_forward_ground_time() const {
	return placement.forward_gait.ground_time;
}
void RenIK::set_forward_lift_time_base(float lift_time_base) {
	placement.forward_gait.lift_time_base = lift_time_base;
}
float RenIK::get_forward_lift_time_base() const {
	return placement.forward_gait.lift_time_base;
}
void RenIK::set_forward_lift_time_scalar(float lift_time_scalar) {
	placement.forward_gait.lift_time_scalar = lift_time_scalar;
}
float RenIK::get_forward_lift_time_scalar() const {
	return placement.forward_gait.lift_time_scalar;
}
void RenIK::set_forward_apex_in_time_base(float apex_in_time_base) {
	placement.forward_gait.apex_in_time_base = apex_in_time_base;
}
float RenIK::get_forward_apex_in_time_base() const {
	return placement.forward_gait.apex_in_time_base;
}
void RenIK::set_forward_apex_in_time_scalar(float apex_in_time_scalar) {
	placement.forward_gait.apex_in_time_scalar = apex_in_time_scalar;
}
float RenIK::get_forward_apex_in_time_scalar() const {
	return placement.forward_gait.apex_in_time_scalar;
}
void RenIK::set_forward_apex_out_time_base(float apex_out_time_base) {
	placement.forward_gait.apex_out_time_base = apex_out_time_base;
}
float RenIK::get_forward_apex_out_time_base() const {
	return placement.forward_gait.apex_out_time_base;
}
void RenIK::set_forward_apex_out_time_scalar(float apex_out_time_scalar) {
	placement.forward_gait.apex_out_time_scalar = apex_out_time_scalar;
}
float RenIK::get_forward_apex_out_time_scalar() const {
	return placement.forward_gait.apex_out_time_scalar;
}
void RenIK::set_forward_drop_time_base(float drop_time_base) {
	placement.forward_gait.drop_time_base = drop_time_base;
}
float RenIK::get_forward_drop_time_base() const {
	return placement.forward_gait.drop_time_base;
}
void RenIK::set_forward_drop_time_scalar(float drop_time_scalar) {
	placement.forward_gait.drop_time_scalar = drop_time_scalar;
}
float RenIK::get_forward_drop_time_scalar() const {
	return placement.forward_gait.drop_time_scalar;
}

void RenIK::set_forward_tip_toe_distance_scalar(float tip_toe_distance_scalar) {
	placement.forward_gait.tip_toe_distance_scalar =
			Math::deg_to_rad(tip_toe_distance_scalar);
}

float RenIK::get_forward_tip_toe_distance_scalar() const {
	return Math::rad_to_deg(placement.forward_gait.tip_toe_distance_scalar);
}

void RenIK::set_forward_tip_toe_speed_scalar(float tip_toe_speed_scalar) {
	placement.forward_gait.tip_toe_speed_scalar =
			Math::deg_to_rad(tip_toe_speed_scalar);
}
float RenIK::get_forward_tip_toe_speed_scalar() const {
	return Math::rad_to_deg(placement.forward_gait.tip_toe_speed_scalar);
}
void RenIK::set_forward_tip_toe_angle_max(float tip_toe_angle_max) {
	placement.forward_gait.tip_toe_angle_max = Math::deg_to_rad(tip_toe_angle_max);
}
float RenIK::get_forward_tip_toe_angle_max() const {
	return Math::rad_to_deg(placement.forward_gait.tip_toe_angle_max);
}

void RenIK::set_forward_lift_vertical(float lift_vertical) {
	placement.forward_gait.lift_vertical = lift_vertical / 100.0;
}
float RenIK::get_forward_lift_vertical() const {
	return placement.forward_gait.lift_vertical * 100.0;
}
void RenIK::set_forward_lift_vertical_scalar(float lift_vertical_scalar) {
	placement.forward_gait.lift_vertical_scalar = lift_vertical_scalar / 100.0;
}
float RenIK::get_forward_lift_vertical_scalar() const {
	return placement.forward_gait.lift_vertical_scalar * 100.0;
}
void RenIK::set_forward_lift_horizontal_scalar(float lift_horizontal_scalar) {
	placement.forward_gait.lift_horizontal_scalar =
			lift_horizontal_scalar / 100.0;
}
float RenIK::get_forward_lift_horizontal_scalar() const {
	return placement.forward_gait.lift_horizontal_scalar * 100.0;
}
void RenIK::set_forward_lift_angle(float lift_angle) {
	placement.forward_gait.lift_angle = Math::deg_to_rad(lift_angle);
}
float RenIK::get_forward_lift_angle() const {
	return Math::rad_to_deg(placement.forward_gait.lift_angle);
}

void RenIK::set_forward_apex_vertical(float apex_vertical) {
	placement.forward_gait.apex_vertical = apex_vertical / 100.0;
}
float RenIK::get_forward_apex_vertical() const {
	return placement.forward_gait.apex_vertical * 100.0;
}
void RenIK::set_forward_apex_vertical_scalar(float apex_vertical_scalar) {
	placement.forward_gait.apex_vertical_scalar = apex_vertical_scalar / 100.0;
}
float RenIK::get_forward_apex_vertical_scalar() const {
	return placement.forward_gait.apex_vertical_scalar * 100.0;
}
void RenIK::set_forward_apex_angle(float apex_angle) {
	placement.forward_gait.apex_angle = Math::deg_to_rad(apex_angle);
}
float RenIK::get_forward_apex_angle() const {
	return Math::rad_to_deg(placement.forward_gait.apex_angle);
}

void RenIK::set_forward_drop_vertical(float drop_vertical) {
	placement.forward_gait.drop_vertical = drop_vertical / 100.0;
}
float RenIK::get_forward_drop_vertical() const {
	return placement.forward_gait.drop_vertical * 100.0;
}
void RenIK::set_forward_drop_vertical_scalar(float drop_vertical_scalar) {
	placement.forward_gait.drop_vertical_scalar = drop_vertical_scalar / 100.0;
}
float RenIK::get_forward_drop_vertical_scalar() const {
	return placement.forward_gait.drop_vertical_scalar * 100.0;
}
void RenIK::set_forward_drop_horizontal_scalar(float drop_horizontal_scalar) {
	placement.forward_gait.drop_horizontal_scalar =
			drop_horizontal_scalar / 100.0;
}
float RenIK::get_forward_drop_horizontal_scalar() const {
	return placement.forward_gait.drop_horizontal_scalar * 100.0;
}
void RenIK::set_forward_drop_angle(float drop_angle) {
	placement.forward_gait.drop_angle = Math::deg_to_rad(drop_angle);
}
float RenIK::get_forward_drop_angle() const {
	return Math::rad_to_deg(placement.forward_gait.drop_angle);
}

void RenIK::set_forward_contact_point_ease(float contact_point_ease) {
	placement.forward_gait.contact_point_ease = contact_point_ease / 100.0;
}
float RenIK::get_forward_contact_point_ease() const {
	return placement.forward_gait.contact_point_ease * 100.0;
}
void RenIK::set_forward_contact_point_ease_scalar(
		float contact_point_ease_scalar) {
	placement.forward_gait.contact_point_ease_scalar =
			contact_point_ease_scalar / 100.0;
}
float RenIK::get_forward_contact_point_ease_scalar() const {
	return placement.forward_gait.contact_point_ease_scalar * 100.0;
}
void RenIK::set_forward_scaling_ease(float scaling_ease) {
	placement.forward_gait.scaling_ease = scaling_ease / 100.0;
}
float RenIK::get_forward_scaling_ease() const {
	return placement.forward_gait.scaling_ease * 100.0;
}

void RenIK::set_backward_speed_scalar_min(float speed_scalar_min) {
	placement.backward_gait.speed_scalar_min = speed_scalar_min / 100.0;
}
float RenIK::get_backward_speed_scalar_min() const {
	return placement.backward_gait.speed_scalar_min * 100.0;
}
void RenIK::set_backward_speed_scalar_max(float speed_scalar_max) {
	placement.backward_gait.speed_scalar_max = speed_scalar_max / 100.0;
}
float RenIK::get_backward_speed_scalar_max() const {
	return placement.backward_gait.speed_scalar_max * 100.0;
}

void RenIK::set_backward_ground_time(float ground_time) {
	placement.backward_gait.ground_time = ground_time;
}
float RenIK::get_backward_ground_time() const {
	return placement.backward_gait.ground_time;
}
void RenIK::set_backward_lift_time_base(float lift_time_base) {
	placement.backward_gait.lift_time_base = lift_time_base;
}
float RenIK::get_backward_lift_time_base() const {
	return placement.backward_gait.lift_time_base;
}
void RenIK::set_backward_lift_time_scalar(float lift_time_scalar) {
	placement.backward_gait.lift_time_scalar = lift_time_scalar;
}
float RenIK::get_backward_lift_time_scalar() const {
	return placement.backward_gait.lift_time_scalar;
}
void RenIK::set_backward_apex_in_time_base(float apex_in_time_base) {
	placement.backward_gait.apex_in_time_base = apex_in_time_base;
}
float RenIK::get_backward_apex_in_time_base() const {
	return placement.backward_gait.apex_in_time_base;
}
void RenIK::set_backward_apex_in_time_scalar(float apex_in_time_scalar) {
	placement.backward_gait.apex_in_time_scalar = apex_in_time_scalar;
}
float RenIK::get_backward_apex_in_time_scalar() const {
	return placement.backward_gait.apex_in_time_scalar;
}
void RenIK::set_backward_apex_out_time_base(float apex_out_time_base) {
	placement.backward_gait.apex_out_time_base = apex_out_time_base;
}
float RenIK::get_backward_apex_out_time_base() const {
	return placement.backward_gait.apex_out_time_base;
}
void RenIK::set_backward_apex_out_time_scalar(float apex_out_time_scalar) {
	placement.backward_gait.apex_out_time_scalar = apex_out_time_scalar;
}
float RenIK::get_backward_apex_out_time_scalar() const {
	return placement.backward_gait.apex_out_time_scalar;
}
void RenIK::set_backward_drop_time_base(float drop_time_base) {
	placement.backward_gait.drop_time_base = drop_time_base;
}
float RenIK::get_backward_drop_time_base() const {
	return placement.backward_gait.drop_time_base;
}
void RenIK::set_backward_drop_time_scalar(float drop_time_scalar) {
	placement.backward_gait.drop_time_scalar = drop_time_scalar;
}
float RenIK::get_backward_drop_time_scalar() const {
	return placement.backward_gait.drop_time_scalar;
}

void RenIK::set_backward_tip_toe_distance_scalar(
		float tip_toe_distance_scalar) {
	placement.backward_gait.tip_toe_distance_scalar =
			Math::deg_to_rad(tip_toe_distance_scalar);
}
float RenIK::get_backward_tip_toe_distance_scalar() const {
	return Math::rad_to_deg(placement.backward_gait.tip_toe_distance_scalar);
}
void RenIK::set_backward_tip_toe_speed_scalar(float tip_toe_speed_scalar) {
	placement.backward_gait.tip_toe_speed_scalar =
			Math::deg_to_rad(tip_toe_speed_scalar);
}
float RenIK::get_backward_tip_toe_speed_scalar() const {
	return Math::rad_to_deg(placement.backward_gait.tip_toe_speed_scalar);
}
void RenIK::set_backward_tip_toe_angle_max(float tip_toe_angle_max) {
	placement.backward_gait.tip_toe_angle_max = Math::deg_to_rad(tip_toe_angle_max);
}
float RenIK::get_backward_tip_toe_angle_max() const {
	return Math::rad_to_deg(placement.backward_gait.tip_toe_angle_max);
}

void RenIK::set_backward_lift_vertical(float lift_vertical) {
	placement.backward_gait.lift_vertical = lift_vertical / 100.0;
}
float RenIK::get_backward_lift_vertical() const {
	return placement.backward_gait.lift_vertical * 100.0;
}
void RenIK::set_backward_lift_vertical_scalar(float lift_vertical_scalar) {
	placement.backward_gait.lift_vertical_scalar = lift_vertical_scalar / 100.0;
}
float RenIK::get_backward_lift_vertical_scalar() const {
	return placement.backward_gait.lift_vertical_scalar * 100.0;
}
void RenIK::set_backward_lift_horizontal_scalar(float lift_horizontal_scalar) {
	placement.backward_gait.lift_horizontal_scalar =
			lift_horizontal_scalar / 100.0;
}
float RenIK::get_backward_lift_horizontal_scalar() const {
	return placement.backward_gait.lift_horizontal_scalar * 100.0;
}
void RenIK::set_backward_lift_angle(float lift_angle) {
	placement.backward_gait.lift_angle = Math::deg_to_rad(lift_angle);
}
float RenIK::get_backward_lift_angle() const {
	return Math::rad_to_deg(placement.backward_gait.lift_angle);
}

void RenIK::set_backward_apex_vertical(float apex_vertical) {
	placement.backward_gait.apex_vertical = apex_vertical / 100.0;
}
float RenIK::get_backward_apex_vertical() const {
	return placement.backward_gait.apex_vertical * 100.0;
}
void RenIK::set_backward_apex_vertical_scalar(float apex_vertical_scalar) {
	placement.backward_gait.apex_vertical_scalar = apex_vertical_scalar / 100.0;
}
float RenIK::get_backward_apex_vertical_scalar() const {
	return placement.backward_gait.apex_vertical_scalar * 100.0;
}
void RenIK::set_backward_apex_angle(float apex_angle) {
	placement.backward_gait.apex_angle = Math::deg_to_rad(apex_angle);
}
float RenIK::get_backward_apex_angle() const {
	return Math::rad_to_deg(placement.backward_gait.apex_angle);
}

void RenIK::set_backward_drop_vertical(float drop_vertical) {
	placement.backward_gait.drop_vertical = drop_vertical / 100.0;
}
float RenIK::get_backward_drop_vertical() const {
	return placement.backward_gait.drop_vertical * 100.0;
}
void RenIK::set_backward_drop_vertical_scalar(float drop_vertical_scalar) {
	placement.backward_gait.drop_vertical_scalar = drop_vertical_scalar / 100.0;
}
float RenIK::get_backward_drop_vertical_scalar() const {
	return placement.backward_gait.drop_vertical_scalar * 100.0;
}
void RenIK::set_backward_drop_horizontal_scalar(float drop_horizontal_scalar) {
	placement.backward_gait.drop_horizontal_scalar =
			drop_horizontal_scalar / 100.0;
}
float RenIK::get_backward_drop_horizontal_scalar() const {
	return placement.backward_gait.drop_horizontal_scalar * 100.0;
}
void RenIK::set_backward_drop_angle(float drop_angle) {
	placement.backward_gait.drop_angle = Math::deg_to_rad(drop_angle);
}
float RenIK::get_backward_drop_angle() const {
	return Math::rad_to_deg(placement.backward_gait.drop_angle);
}

void RenIK::set_backward_contact_point_ease(float contact_point_ease) {
	placement.backward_gait.contact_point_ease = contact_point_ease / 100.0;
}
float RenIK::get_backward_contact_point_ease() const {
	return placement.backward_gait.contact_point_ease * 100.0;
}
void RenIK::set_backward_contact_point_ease_scalar(
		float contact_point_ease_scalar) {
	placement.backward_gait.contact_point_ease_scalar =
			contact_point_ease_scalar / 100.0;
}
float RenIK::get_backward_contact_point_ease_scalar() const {
	return placement.backward_gait.contact_point_ease_scalar * 100.0;
}
void RenIK::set_backward_scaling_ease(float scaling_ease) {
	placement.backward_gait.scaling_ease = scaling_ease / 100.0;
}
float RenIK::get_backward_scaling_ease() const {
	return placement.backward_gait.scaling_ease * 100.0;
}

void RenIK::set_sideways_speed_scalar_min(float speed_scalar_min) {
	placement.sideways_gait.speed_scalar_min = speed_scalar_min / 100.0;
}
float RenIK::get_sideways_speed_scalar_min() const {
	return placement.sideways_gait.speed_scalar_min * 100.0;
}
void RenIK::set_sideways_speed_scalar_max(float speed_scalar_max) {
	placement.sideways_gait.speed_scalar_max = speed_scalar_max / 100.0;
}
float RenIK::get_sideways_speed_scalar_max() const {
	return placement.sideways_gait.speed_scalar_max * 100.0;
}

void RenIK::set_sideways_ground_time(float ground_time) {
	placement.sideways_gait.ground_time = ground_time;
}
float RenIK::get_sideways_ground_time() const {
	return placement.sideways_gait.ground_time;
}
void RenIK::set_sideways_lift_time_base(float lift_time_base) {
	placement.sideways_gait.lift_time_base = lift_time_base;
}
float RenIK::get_sideways_lift_time_base() const {
	return placement.sideways_gait.lift_time_base;
}
void RenIK::set_sideways_lift_time_scalar(float lift_time_scalar) {
	placement.sideways_gait.lift_time_scalar = lift_time_scalar;
}
float RenIK::get_sideways_lift_time_scalar() const {
	return placement.sideways_gait.lift_time_scalar;
}
void RenIK::set_sideways_apex_in_time_base(float apex_in_time_base) {
	placement.sideways_gait.apex_in_time_base = apex_in_time_base;
}
float RenIK::get_sideways_apex_in_time_base() const {
	return placement.sideways_gait.apex_in_time_base;
}
void RenIK::set_sideways_apex_in_time_scalar(float apex_in_time_scalar) {
	placement.sideways_gait.apex_in_time_scalar = apex_in_time_scalar;
}
float RenIK::get_sideways_apex_in_time_scalar() const {
	return placement.sideways_gait.apex_in_time_scalar;
}
void RenIK::set_sideways_apex_out_time_base(float apex_out_time_base) {
	placement.sideways_gait.apex_out_time_base = apex_out_time_base;
}
float RenIK::get_sideways_apex_out_time_base() const {
	return placement.sideways_gait.apex_out_time_base;
}
void RenIK::set_sideways_apex_out_time_scalar(float apex_out_time_scalar) {
	placement.sideways_gait.apex_out_time_scalar = apex_out_time_scalar;
}
float RenIK::get_sideways_apex_out_time_scalar() const {
	return placement.sideways_gait.apex_out_time_scalar;
}
void RenIK::set_sideways_drop_time_base(float drop_time_base) {
	placement.sideways_gait.drop_time_base = drop_time_base;
}
float RenIK::get_sideways_drop_time_base() const {
	return placement.sideways_gait.drop_time_base;
}
void RenIK::set_sideways_drop_time_scalar(float drop_time_scalar) {
	placement.sideways_gait.drop_time_scalar = drop_time_scalar;
}
float RenIK::get_sideways_drop_time_scalar() const {
	return placement.sideways_gait.drop_time_scalar;
}

void RenIK::set_sideways_tip_toe_distance_scalar(
		float tip_toe_distance_scalar) {
	placement.sideways_gait.tip_toe_distance_scalar =
			Math::deg_to_rad(tip_toe_distance_scalar);
}
float RenIK::get_sideways_tip_toe_distance_scalar() const {
	return Math::rad_to_deg(placement.sideways_gait.tip_toe_distance_scalar);
}
void RenIK::set_sideways_tip_toe_speed_scalar(float tip_toe_speed_scalar) {
	placement.sideways_gait.tip_toe_speed_scalar =
			Math::deg_to_rad(tip_toe_speed_scalar);
}
float RenIK::get_sideways_tip_toe_speed_scalar() const {
	return Math::rad_to_deg(placement.sideways_gait.tip_toe_speed_scalar);
}
void RenIK::set_sideways_tip_toe_angle_max(float tip_toe_angle_max) {
	placement.sideways_gait.tip_toe_angle_max = Math::deg_to_rad(tip_toe_angle_max);
}
float RenIK::get_sideways_tip_toe_angle_max() const {
	return Math::rad_to_deg(placement.sideways_gait.tip_toe_angle_max);
}

void RenIK::set_sideways_lift_vertical(float lift_vertical) {
	placement.sideways_gait.lift_vertical = lift_vertical / 100.0;
}
float RenIK::get_sideways_lift_vertical() const {
	return placement.sideways_gait.lift_vertical * 100.0;
}
void RenIK::set_sideways_lift_vertical_scalar(float lift_vertical_scalar) {
	placement.sideways_gait.lift_vertical_scalar = lift_vertical_scalar / 100.0;
}
float RenIK::get_sideways_lift_vertical_scalar() const {
	return placement.sideways_gait.lift_vertical_scalar * 100.0;
}
void RenIK::set_sideways_lift_horizontal_scalar(float lift_horizontal_scalar) {
	placement.sideways_gait.lift_horizontal_scalar =
			lift_horizontal_scalar / 100.0;
}
float RenIK::get_sideways_lift_horizontal_scalar() const {
	return placement.sideways_gait.lift_horizontal_scalar * 100.0;
}
void RenIK::set_sideways_lift_angle(float lift_angle) {
	placement.sideways_gait.lift_angle = Math::deg_to_rad(lift_angle);
}
float RenIK::get_sideways_lift_angle() const {
	return Math::rad_to_deg(placement.sideways_gait.lift_angle);
}

void RenIK::set_sideways_apex_vertical(float apex_vertical) {
	placement.sideways_gait.apex_vertical = apex_vertical / 100.0;
}
float RenIK::get_sideways_apex_vertical() const {
	return placement.sideways_gait.apex_vertical * 100.0;
}
void RenIK::set_sideways_apex_vertical_scalar(float apex_vertical_scalar) {
	placement.sideways_gait.apex_vertical_scalar = apex_vertical_scalar / 100.0;
}
float RenIK::get_sideways_apex_vertical_scalar() const {
	return placement.sideways_gait.apex_vertical_scalar * 100.0;
}
void RenIK::set_sideways_apex_angle(float apex_angle) {
	placement.sideways_gait.apex_angle = Math::deg_to_rad(apex_angle);
}
float RenIK::get_sideways_apex_angle() const {
	return Math::rad_to_deg(placement.sideways_gait.apex_angle);
}

void RenIK::set_sideways_drop_vertical(float drop_vertical) {
	placement.sideways_gait.drop_vertical = drop_vertical / 100.0;
}
float RenIK::get_sideways_drop_vertical() const {
	return placement.sideways_gait.drop_vertical * 100.0;
}
void RenIK::set_sideways_drop_vertical_scalar(float drop_vertical_scalar) {
	placement.sideways_gait.drop_vertical_scalar = drop_vertical_scalar / 100.0;
}
float RenIK::get_sideways_drop_vertical_scalar() const {
	return placement.sideways_gait.drop_vertical_scalar * 100.0;
}
void RenIK::set_sideways_drop_horizontal_scalar(float drop_horizontal_scalar) {
	placement.sideways_gait.drop_horizontal_scalar =
			drop_horizontal_scalar / 100.0;
}
float RenIK::get_sideways_drop_horizontal_scalar() const {
	return placement.sideways_gait.drop_horizontal_scalar * 100.0;
}
void RenIK::set_sideways_drop_angle(float drop_angle) {
	placement.sideways_gait.drop_angle = Math::deg_to_rad(drop_angle);
}
float RenIK::get_sideways_drop_angle() const {
	return Math::rad_to_deg(placement.sideways_gait.drop_angle);
}

void RenIK::set_sideways_contact_point_ease(float contact_point_ease) {
	placement.sideways_gait.contact_point_ease = contact_point_ease / 100.0;
}
float RenIK::get_sideways_contact_point_ease() const {
	return placement.sideways_gait.contact_point_ease * 100.0;
}
void RenIK::set_sideways_contact_point_ease_scalar(
		float contact_point_ease_scalar) {
	placement.sideways_gait.contact_point_ease_scalar =
			contact_point_ease_scalar / 100.0;
}
float RenIK::get_sideways_contact_point_ease_scalar() const {
	return placement.sideways_gait.contact_point_ease_scalar * 100.0;
}
void RenIK::set_sideways_scaling_ease(float scaling_ease) {
	placement.sideways_gait.scaling_ease = scaling_ease / 100.0;
}
float RenIK::get_sideways_scaling_ease() const {
	return placement.sideways_gait.scaling_ease * 100.0;
}

Vector<Transform3D> RenIK::compute_global_transforms(const Vector<RenIKChain::Joint> &joints, const Transform3D &root, const Transform3D &true_root) {
	Vector<Transform3D> global_transforms;
	global_transforms.resize(joints.size());
	Transform3D current_global_transform = true_root;

	for (int i = 0; i < joints.size(); i++) {
		Transform3D local_transform;
		local_transform.basis = Basis(joints[i].rotation);
		if (i == 0) {
			local_transform.origin = root.origin;
		} else {
			local_transform.origin = joints[i - 1].relative_next;
		}
		current_global_transform *= local_transform;
		global_transforms.write[i] = current_global_transform;
	}

	return global_transforms;
}

void RenIK::compute_rest_and_target_positions(const Vector<Transform3D> &p_global_transforms, const Transform3D &p_target, const Vector3 &p_priority, Vector<Vector3> &p_reference_positions, Vector<Vector3> &p_target_positions, Vector<double> &r_weights) {
	for (int joint_i = 0; joint_i < p_global_transforms.size(); joint_i++) {
		Transform3D bone_direction_global_transform = p_global_transforms[joint_i];
		real_t pin_weight = r_weights[joint_i];
		int32_t rest_index = joint_i * 7;

		Basis tip_basis = bone_direction_global_transform.basis.orthogonalized();

		Quaternion quaternion = tip_basis.get_rotation_quaternion();
		tip_basis.set_quaternion_scale(quaternion, tip_basis.get_scale());

		p_reference_positions.write[rest_index] = p_target.origin - bone_direction_global_transform.origin;
		rest_index++;

		double scale_by = pin_weight;

		Vector3 target_global_space = p_target.origin;
		if (!quaternion.is_equal_approx(Quaternion())) {
			target_global_space = bone_direction_global_transform.xform(p_target.origin);
		}
		double distance = target_global_space.distance_to(bone_direction_global_transform.origin);
		scale_by = MAX(1.0, distance);

		for (int axis_i = Vector3::AXIS_X; axis_i <= Vector3::AXIS_Z; ++axis_i) {
			if (p_priority[axis_i] > 0.0) {
				Vector3 column = p_target.basis.get_column(axis_i);
				p_reference_positions.write[rest_index] = bone_direction_global_transform.affine_inverse().xform((column + p_target.origin) - bone_direction_global_transform.origin);
				p_reference_positions.write[rest_index] *= scale_by;
				rest_index++;
				p_reference_positions.write[rest_index] = bone_direction_global_transform.affine_inverse().xform((p_target.origin - column) - bone_direction_global_transform.origin);
				p_reference_positions.write[rest_index] *= scale_by;
				rest_index++;
			}
		}

		int32_t target_index = joint_i * 7;
		p_target_positions.write[target_index] = p_target.origin - bone_direction_global_transform.origin;
		target_index++;

		scale_by = pin_weight;

		for (int axis_j = Vector3::AXIS_X; axis_j <= Vector3::AXIS_Z; ++axis_j) {
			if (p_priority[axis_j] > 0.0) {
				real_t w = r_weights[rest_index];
				Vector3 column = tip_basis.get_column(axis_j) * p_priority[axis_j];
				p_target_positions.write[target_index] = bone_direction_global_transform.xform((column + p_target.origin) - bone_direction_global_transform.origin);
				p_target_positions.write[target_index] *= Vector3(w, w, w);
				target_index++;
				p_target_positions.write[target_index] = bone_direction_global_transform.xform((p_target.origin - column) - bone_direction_global_transform.origin);
				p_target_positions.write[target_index] *= Vector3(w, w, w);
				target_index++;
			}
		}
	}
}

HashMap<BoneId, Quaternion> RenIK::solve_ik_qcp(Ref<RenIKChain> chain,
		Transform3D root,
		Transform3D target) {
	HashMap<BoneId, Quaternion> map;

	if (!chain->is_valid()) {
		return map;
	}

	Vector<RenIKChain::Joint> joints = chain->get_joints();
	const Transform3D true_root = root.translated_local(joints[0].relative_prev);
	Vector<Vector3> rest_positions;
	Vector<Vector3> target_positions;
	Vector<double> weights;
	constexpr int TRANSFORM_TO_HEADINGS_COUNT = 7;
	rest_positions.resize(TRANSFORM_TO_HEADINGS_COUNT * joints.size());
	target_positions.resize(TRANSFORM_TO_HEADINGS_COUNT * joints.size());
	weights.resize(TRANSFORM_TO_HEADINGS_COUNT * joints.size());
	weights.fill(1.0);
	const Vector3 priority = Vector3(0.2, 0, 0.2);

	Vector<Transform3D> global_transforms = compute_global_transforms(joints, root, true_root);

	static constexpr double evec_prec = static_cast<double>(1E-6);
	QCP qcp = QCP(evec_prec);

	compute_rest_and_target_positions(global_transforms, target, priority, rest_positions, target_positions, weights);

	for (int joint_i = 0; joint_i < joints.size(); joint_i++) {
		Quaternion solved_global_pose = qcp.weighted_superpose(rest_positions, target_positions, weights, false);

		int parent_index = joint_i > 0 ? joint_i - 1 : 0;
		const Basis new_rot = global_transforms[parent_index].basis;

		const Quaternion local_pose = new_rot.inverse() * solved_global_pose * new_rot;
		map.insert(joints[joint_i].id, local_pose);

		global_transforms.write[joint_i] = global_transforms[parent_index] * Transform3D(Basis(local_pose));
	}

	return map;
}

void RenIK::setup_humanoid_bones(bool set_targets) {
	ERR_FAIL_NULL(skeleton);
	static const String HEAD = "Head";
	static const String LEFT_HAND = "LeftHand";
	static const String RIGHT_HAND = "RightHand";
	static const String HIPS = "Hips";
	static const String LEFT_FOOT = "LeftFoot";
	static const String RIGHT_FOOT = "RightFoot";
	HashMap<String, Node3D *> ik_target_spatials;
	HashMap<String, NodePath> bone_node_paths;
	bone_node_paths["Head"] = NodePath("Head");
	bone_node_paths["LeftHand"] = NodePath("LeftHand");
	bone_node_paths["RightHand"] = NodePath("RightHand");
	bone_node_paths["Hips"] = NodePath("Hips");
	bone_node_paths["LeftFoot"] = NodePath("LeftFoot");
	bone_node_paths["RightFoot"] = NodePath("RightFoot");
	Node* root = skeleton->get_owner();
	for (const KeyValue<String, NodePath> &entry : bone_node_paths) {
		String bone_name = entry.key;
		int bone_idx = -1;
		if (skeleton && set_targets) {
			bone_idx = skeleton->find_bone(bone_name);
		}
		if (skeleton->has_node(entry.value)) {
			ik_target_spatials[bone_name] = Object::cast_to<Node3D>(skeleton->get_node(entry.value));
			bone_node_paths[bone_name] = entry.value;
		} else {
			Marker3D *new_position_3d = memnew(Marker3D);
			new_position_3d->set_dont_save(true);
			new_position_3d->set_name(set_targets ? bone_name : String(""));
			skeleton->add_child(new_position_3d, true);
			new_position_3d->set_owner(root);
			ik_target_spatials[bone_name] = new_position_3d;
			bone_node_paths[bone_name] = new_position_3d->get_path();
		}
		if (bone_name == HEAD) {
			set_head_bone(bone_idx);
		} else if (bone_name == LEFT_HAND) {
			set_hand_left_bone(bone_idx);
		} else if (bone_name == RIGHT_HAND) {
			set_hand_right_bone(bone_idx);
		} else if (bone_name == HIPS) {
			set_hip_bone(bone_idx);
		} else if (bone_name == LEFT_FOOT) {
			set_foot_left_bone(bone_idx);
		} else if (bone_name == RIGHT_FOOT) {
			set_foot_right_bone(bone_idx);
		}
	}
	set_head_target_path(bone_node_paths[HEAD]);
	set_hand_left_target_path(bone_node_paths[LEFT_HAND]);
	set_hand_right_target_path(bone_node_paths[RIGHT_HAND]);
	set_hip_target_path(bone_node_paths[HIPS]);
	set_foot_left_target_path(bone_node_paths[LEFT_FOOT]);
	set_foot_right_target_path(bone_node_paths[RIGHT_FOOT]);
	for (const KeyValue<String, Node3D *> &entry : ik_target_spatials) {
		String bone_name = entry.key;

		if (skeleton && set_targets) {
			int bone_idx = skeleton->find_bone(bone_name);

			if (bone_idx != -1) {
				Transform3D global_pose = skeleton->get_bone_global_pose_no_override(bone_idx);

				if (ik_target_spatials[bone_name]) {
					ik_target_spatials[bone_name]->set_transform(global_pose);
				}
			}
		} else {
			if (ik_target_spatials[bone_name]) {
				ik_target_spatials[bone_name]->set_transform(Transform3D());
			}
		}
	}
}

void RenIK::set_setup_humanoid_bones(bool set_targets) {
	is_setup_humanoid_bones = set_targets;
	setup_humanoid_bones(is_setup_humanoid_bones);
}

bool RenIK::get_setup_humanoid_bones() const {
	return is_setup_humanoid_bones;
}

#endif // _3D_DISABLED

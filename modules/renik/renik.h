/**************************************************************************/
/*  renik.h                                                               */
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

#ifndef RENIK_H
#define RENIK_H

#ifndef _3D_DISABLED

#include "renik/renik_chain.h"
#include "renik/renik_helper.h"
#include "renik/renik_limb.h"
#include "renik/renik_placement.h"
#include "servers/physics_server_3d.h"
#include <core/config/engine.h>
#include <core/variant/variant.h>
#include <scene/3d/skeleton_3d.h>
#include <scene/main/node.h>
#include <memory>
#include <vector>

class RenIK : public RefCounted {
	GDCLASS(RenIK, RefCounted);

public:
	RenIK();

	void _initialize(Skeleton3D* p_skeleton);
	void on_post_initialize();

	struct SpineTransforms {
		Transform3D hip_transform;
		Transform3D left_arm_parent_transform;
		Transform3D right_arm_parent_transform;
		Transform3D arm_parent_transform;
		Transform3D head_transform;
		SpineTransforms(Transform3D hip = Transform3D(),
				Transform3D left_arm = Transform3D(),
				Transform3D right_arm = Transform3D(),
				Transform3D head = Transform3D()) {
			hip_transform = hip;
			left_arm_parent_transform = left_arm;
			right_arm_parent_transform = right_arm;
			head_transform = head;
		}
	};
	void setup_humanoid_bones(bool set_targets);
	bool is_setup_humanoid_bones = false;

protected:
	void _validate_property(PropertyInfo &property) const;
	void _notification(int p_what);
	static void _bind_methods();

public:
	Vector<Transform3D> compute_global_transforms(const Vector<RenIKChain::Joint> &joints, const Transform3D &root, const Transform3D &true_root);
	void compute_rest_and_target_positions(const Vector<Transform3D> &p_global_transforms, const Transform3D &p_target, const Vector3 &p_priority, Vector<Vector3> &p_reference_positions, Vector<Vector3> &p_target_positions, Vector<double> &r_weights);
	HashMap<BoneId, Quaternion> solve_ik_qcp(Ref<RenIKChain> chain,
			Transform3D root,
			Transform3D target);
	void set_setup_humanoid_bones(bool set_targets);
	bool get_setup_humanoid_bones() const;
	void update_ik();
	void update_placement(float delta);

	void apply_ik_map(HashMap<BoneId, Quaternion> ik_map, Transform3D global_parent,
			Vector<BoneId> apply_order);
	void apply_ik_map(HashMap<BoneId, Basis> ik_map, Transform3D global_parent,
			Vector<BoneId> apply_order);
	Vector<BoneId> bone_id_order(Ref<RenIKChain> chain);
	Vector<BoneId> bone_id_order(Ref<RenIKLimb> limb);

	Transform3D get_global_parent_pose(BoneId child,
			HashMap<BoneId, Quaternion> ik_map,
			Transform3D map_global_parent);

	SpineTransforms perform_torso_ik();
	void perform_hand_left_ik(Transform3D global_parent, Transform3D target);
	void perform_hand_right_ik(Transform3D global_parent, Transform3D target);
	void perform_foot_left_ik(Transform3D global_parent, Transform3D target);
	void perform_foot_right_ik(Transform3D global_parent, Transform3D target);
	void reset_chain(Ref<RenIKChain> chain);
	void reset_limb(Ref<RenIKLimb> limb);

	bool get_live_preview();
	void set_live_preview(bool p_enable);

	//NodePath get_skeleton_path();
	//void set_skeleton_path(NodePath p_path);
	void set_skeleton(Node *p_path);

	void set_head_bone_by_name(String p_bone);

	void set_hand_left_bone_by_name(String p_bone);
	void set_lower_arm_left_bone_by_name(String p_bone);
	void set_upper_arm_left_bone_by_name(String p_bone);

	void set_hand_right_bone_by_name(String p_bone);
	void set_lower_arm_right_bone_by_name(String p_bone);
	void set_upper_arm_right_bone_by_name(String p_bone);

	void set_hip_bone_by_name(String p_bone);

	void set_foot_left_bone_by_name(String p_bone);
	void set_lower_leg_left_bone_by_name(String p_bone);
	void set_upper_leg_left_bone_by_name(String p_bone);

	void set_foot_right_bone_by_name(String p_bone);
	void set_lower_leg_right_bone_by_name(String p_bone);
	void set_upper_leg_right_bone_by_name(String p_bone);

	void set_head_bone(BoneId p_bone);

	void set_hand_left_bone(BoneId p_bone);
	void set_lower_arm_left_bone(BoneId p_bone);
	void set_upper_arm_left_bone(BoneId p_bone);

	void set_hand_right_bone(BoneId p_bone);
	void set_lower_arm_right_bone(BoneId p_bone);
	void set_upper_arm_right_bone(BoneId p_bone);

	void set_hip_bone(BoneId p_bone);

	void set_foot_left_bone(BoneId p_bone);
	void set_lower_leg_left_bone(BoneId p_bone);
	void set_upper_leg_left_bone(BoneId p_bone);

	void set_foot_right_bone(BoneId p_bone);
	void set_lower_leg_right_bone(BoneId p_bone);
	void set_upper_leg_right_bone(BoneId p_bone);

	int64_t get_hip_bone();
	int64_t get_head_bone();

	int64_t get_hand_left_bone();
	int64_t get_lower_arm_left_bone();
	int64_t get_upper_arm_left_bone();

	int64_t get_hand_right_bone();
	int64_t get_lower_arm_right_bone();
	int64_t get_upper_arm_right_bone();

	int64_t get_foot_left_bone();
	int64_t get_lower_leg_left_bone();
	int64_t get_upper_leg_left_bone();

	int64_t get_foot_right_bone();
	int64_t get_lower_leg_right_bone();
	int64_t get_upper_leg_right_bone();

	String get_hip_bone_name();
	String get_head_bone_name();

	String get_hand_left_bone_name();
	String get_lower_arm_left_bone_name();
	String get_upper_arm_left_bone_name();

	String get_hand_right_bone_name();
	String get_lower_arm_right_bone_name();
	String get_upper_arm_right_bone_name();

	String get_foot_left_bone_name();
	String get_lower_leg_left_bone_name();
	String get_upper_leg_left_bone_name();

	String get_foot_right_bone_name();
	String get_lower_leg_right_bone_name();
	String get_upper_leg_right_bone_name();

	void set_head_target_path(NodePath p_path);
	void set_hand_left_target_path(NodePath p_path);
	void set_hand_right_target_path(NodePath p_path);
	void set_hip_target_path(NodePath p_path);
	void set_foot_left_target_path(NodePath p_path);
	void set_foot_right_target_path(NodePath p_path);
	NodePath get_head_target_path();
	NodePath get_hand_left_target_path();
	NodePath get_hand_right_target_path();
	NodePath get_hip_target_path();
	NodePath get_foot_left_target_path();
	NodePath get_foot_right_target_path();

	float get_arm_upper_twist_offset();
	void set_arm_upper_twist_offset(float degrees);
	float get_arm_lower_twist_offset();
	void set_arm_lower_twist_offset(float degrees);
	float get_arm_roll_offset();
	void set_arm_roll_offset(float degrees);
	float get_arm_upper_limb_twist();
	void set_arm_upper_limb_twist(float ratio);
	float get_arm_lower_limb_twist();
	void set_arm_lower_limb_twist(float ratio);
	float get_arm_twist_inflection_point_offset();
	void set_arm_twist_inflection_point_offset(float degrees);
	float get_arm_twist_overflow();
	void set_arm_twist_overflow(float degrees);

	Vector3 get_arm_pole_offset();
	void set_arm_pole_offset(Vector3 euler);
	Vector3 get_arm_target_position_influence();
	void set_arm_target_position_influence(Vector3 xyz);
	float get_arm_target_rotation_influence();
	void set_arm_target_rotation_influence(float influence);
	float get_arm_target_twist_influence();
	void set_arm_target_twist_influence(float influence);

	float get_leg_upper_twist_offset();
	void set_leg_upper_twist_offset(float degrees);
	float get_leg_lower_twist_offset();
	void set_leg_lower_twist_offset(float degrees);
	float get_leg_roll_offset();
	void set_leg_roll_offset(float degrees);
	float get_leg_upper_limb_twist();
	void set_leg_upper_limb_twist(float ratio);
	float get_leg_lower_limb_twist();
	void set_leg_lower_limb_twist(float ratio);
	float get_leg_twist_inflection_point_offset();
	void set_leg_twist_inflection_point_offset(float degrees);
	float get_leg_twist_overflow();
	void set_leg_twist_overflow(float degrees);

	Vector3 get_leg_pole_offset();
	void set_leg_pole_offset(Vector3 euler);
	Vector3 get_leg_target_position_influence();
	void set_leg_target_position_influence(Vector3 xyz);
	float get_leg_target_rotation_influence();
	void set_leg_target_rotation_influence(float influence);
	float get_leg_target_twist_influence();
	void set_leg_target_twist_influence(float influence);

	Vector3 get_spine_curve();
	void set_spine_curve(Vector3 influence);
	float get_upper_spine_stiffness();
	void set_upper_spine_stiffness(float influence);
	float get_lower_spine_stiffness();
	void set_lower_spine_stiffness(float influence);
	float get_spine_twist();
	void set_spine_twist(float influence);
	float get_spine_twist_start();
	void set_spine_twist_start(float influence);

	float get_shoulder_influence();
	void set_shoulder_influence(float influence);

	Vector3 get_shoulder_offset();
	void set_shoulder_offset(Vector3 euler);
	Vector3 get_shoulder_pole_offset();
	void set_shoulder_pole_offset(Vector3 euler);

	bool get_use_editor_speed();
	void set_use_editor_speed(bool enable);

	void set_falling(bool falling);
	void enable_solve_ik_every_frame(bool automatically_update_ik);
	void enable_foot_placement(bool enabled);
	void enable_hip_placement(bool enabled);
	void set_collision_mask_bit(int p_bit, bool p_value);
	bool get_collision_mask_bit(int p_bit) const;
	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;
	void set_collide_with_areas(bool p_clip);
	bool is_collide_with_areas_enabled() const;
	void set_collide_with_bodies(bool p_clip);
	bool is_collide_with_bodies_enabled() const;

	void set_forward_speed_scalar_min(float speed_scalar_min);
	float get_forward_speed_scalar_min() const;
	void set_forward_speed_scalar_max(float speed_scalar_max);
	float get_forward_speed_scalar_max() const;
	void set_forward_ground_time(float ground_time);
	float get_forward_ground_time() const;
	void set_forward_lift_time_base(float lift_time_base);
	float get_forward_lift_time_base() const;
	void set_forward_lift_time_scalar(float lift_time_scalar);
	float get_forward_lift_time_scalar() const;
	void set_forward_apex_in_time_base(float apex_in_time_base);
	float get_forward_apex_in_time_base() const;
	void set_forward_apex_in_time_scalar(float apex_in_time_scalar);
	float get_forward_apex_in_time_scalar() const;
	void set_forward_apex_out_time_base(float apex_out_time_base);
	float get_forward_apex_out_time_base() const;
	void set_forward_apex_out_time_scalar(float apex_out_time_scalar);
	float get_forward_apex_out_time_scalar() const;
	void set_forward_drop_time_base(float drop_time_base);
	float get_forward_drop_time_base() const;
	void set_forward_drop_time_scalar(float drop_time_scalar);
	float get_forward_drop_time_scalar() const;
	void set_forward_tip_toe_distance_scalar(float tip_toe_distance_scalar);
	float get_forward_tip_toe_distance_scalar() const;
	void set_forward_tip_toe_speed_scalar(float tip_toe_speed_scalar);
	float get_forward_tip_toe_speed_scalar() const;
	void set_forward_tip_toe_angle_max(float tip_toe_angle_max);
	float get_forward_tip_toe_angle_max() const;
	void set_forward_lift_vertical(float lift_vertical);
	float get_forward_lift_vertical() const;
	void set_forward_lift_vertical_scalar(float lift_vertical_scalar);
	float get_forward_lift_vertical_scalar() const;
	void set_forward_lift_horizontal_scalar(float lift_horizontal_scalar);
	float get_forward_lift_horizontal_scalar() const;
	void set_forward_lift_angle(float lift_angle);
	float get_forward_lift_angle() const;
	void set_forward_apex_vertical(float apex_vertical);
	float get_forward_apex_vertical() const;
	void set_forward_apex_vertical_scalar(float apex_vertical_scalar);
	float get_forward_apex_vertical_scalar() const;
	void set_forward_apex_angle(float apex_angle);
	float get_forward_apex_angle() const;
	void set_forward_drop_vertical(float drop_vertical);
	float get_forward_drop_vertical() const;
	void set_forward_drop_vertical_scalar(float drop_vertical_scalar);
	float get_forward_drop_vertical_scalar() const;
	void set_forward_drop_horizontal_scalar(float drop_horizontal_scalar);
	float get_forward_drop_horizontal_scalar() const;
	void set_forward_drop_angle(float drop_angle);
	float get_forward_drop_angle() const;
	void set_forward_contact_point_ease(float contact_point_ease);
	float get_forward_contact_point_ease() const;
	void set_forward_contact_point_ease_scalar(float contact_point_ease_scalar);
	float get_forward_contact_point_ease_scalar() const;
	void set_forward_scaling_ease(float scaling_ease);
	float get_forward_scaling_ease() const;

	void set_backward_speed_scalar_min(float speed_scalar_min);
	float get_backward_speed_scalar_min() const;
	void set_backward_speed_scalar_max(float speed_scalar_max);
	float get_backward_speed_scalar_max() const;
	void set_backward_ground_time(float ground_time);
	float get_backward_ground_time() const;
	void set_backward_lift_time_base(float lift_time_base);
	float get_backward_lift_time_base() const;
	void set_backward_lift_time_scalar(float lift_time_scalar);
	float get_backward_lift_time_scalar() const;
	void set_backward_apex_in_time_base(float apex_in_time_base);
	float get_backward_apex_in_time_base() const;
	void set_backward_apex_in_time_scalar(float apex_in_time_scalar);
	float get_backward_apex_in_time_scalar() const;
	void set_backward_apex_out_time_base(float apex_out_time_base);
	float get_backward_apex_out_time_base() const;
	void set_backward_apex_out_time_scalar(float apex_out_time_scalar);
	float get_backward_apex_out_time_scalar() const;
	void set_backward_drop_time_base(float drop_time_base);
	float get_backward_drop_time_base() const;
	void set_backward_drop_time_scalar(float drop_time_scalar);
	float get_backward_drop_time_scalar() const;
	void set_backward_tip_toe_distance_scalar(float tip_toe_distance_scalar);
	float get_backward_tip_toe_distance_scalar() const;
	void set_backward_tip_toe_speed_scalar(float tip_toe_speed_scalar);
	float get_backward_tip_toe_speed_scalar() const;
	void set_backward_tip_toe_angle_max(float tip_toe_angle_max);
	float get_backward_tip_toe_angle_max() const;
	void set_backward_lift_vertical(float lift_vertical);
	float get_backward_lift_vertical() const;
	void set_backward_lift_vertical_scalar(float lift_vertical_scalar);
	float get_backward_lift_vertical_scalar() const;
	void set_backward_lift_horizontal_scalar(float lift_horizontal_scalar);
	float get_backward_lift_horizontal_scalar() const;
	void set_backward_lift_angle(float lift_angle);
	float get_backward_lift_angle() const;
	void set_backward_apex_vertical(float apex_vertical);
	float get_backward_apex_vertical() const;
	void set_backward_apex_vertical_scalar(float apex_vertical_scalar);
	float get_backward_apex_vertical_scalar() const;
	void set_backward_apex_angle(float apex_angle);
	float get_backward_apex_angle() const;
	void set_backward_drop_vertical(float drop_vertical);
	float get_backward_drop_vertical() const;
	void set_backward_drop_vertical_scalar(float drop_vertical_scalar);
	float get_backward_drop_vertical_scalar() const;
	void set_backward_drop_horizontal_scalar(float drop_horizontal_scalar);
	float get_backward_drop_horizontal_scalar() const;
	void set_backward_drop_angle(float drop_angle);
	float get_backward_drop_angle() const;
	void set_backward_contact_point_ease(float contact_point_ease);
	float get_backward_contact_point_ease() const;
	void set_backward_contact_point_ease_scalar(float contact_point_ease_scalar);
	float get_backward_contact_point_ease_scalar() const;
	void set_backward_scaling_ease(float scaling_ease);
	float get_backward_scaling_ease() const;

	void set_sideways_speed_scalar_min(float speed_scalar_min);
	float get_sideways_speed_scalar_min() const;
	void set_sideways_speed_scalar_max(float speed_scalar_max);
	float get_sideways_speed_scalar_max() const;
	void set_sideways_ground_time(float ground_time);
	float get_sideways_ground_time() const;
	void set_sideways_lift_time_base(float lift_time_base);
	float get_sideways_lift_time_base() const;
	void set_sideways_lift_time_scalar(float lift_time_scalar);
	float get_sideways_lift_time_scalar() const;
	void set_sideways_apex_in_time_base(float apex_in_time_base);
	float get_sideways_apex_in_time_base() const;
	void set_sideways_apex_in_time_scalar(float apex_in_time_scalar);
	float get_sideways_apex_in_time_scalar() const;
	void set_sideways_apex_out_time_base(float apex_out_time_base);
	float get_sideways_apex_out_time_base() const;
	void set_sideways_apex_out_time_scalar(float apex_out_time_scalar);
	float get_sideways_apex_out_time_scalar() const;
	void set_sideways_drop_time_base(float drop_time_base);
	float get_sideways_drop_time_base() const;
	void set_sideways_drop_time_scalar(float drop_time_scalar);
	float get_sideways_drop_time_scalar() const;
	void set_sideways_tip_toe_distance_scalar(float tip_toe_distance_scalar);
	float get_sideways_tip_toe_distance_scalar() const;
	void set_sideways_tip_toe_speed_scalar(float tip_toe_speed_scalar);
	float get_sideways_tip_toe_speed_scalar() const;
	void set_sideways_tip_toe_angle_max(float tip_toe_angle_max);
	float get_sideways_tip_toe_angle_max() const;
	void set_sideways_lift_vertical(float lift_vertical);
	float get_sideways_lift_vertical() const;
	void set_sideways_lift_vertical_scalar(float lift_vertical_scalar);
	float get_sideways_lift_vertical_scalar() const;
	void set_sideways_lift_horizontal_scalar(float lift_horizontal_scalar);
	float get_sideways_lift_horizontal_scalar() const;
	void set_sideways_lift_angle(float lift_angle);
	float get_sideways_lift_angle() const;
	void set_sideways_apex_vertical(float apex_vertical);
	float get_sideways_apex_vertical() const;
	void set_sideways_apex_vertical_scalar(float apex_vertical_scalar);
	float get_sideways_apex_vertical_scalar() const;
	void set_sideways_apex_angle(float apex_angle);
	float get_sideways_apex_angle() const;
	void set_sideways_drop_vertical(float drop_vertical);
	float get_sideways_drop_vertical() const;
	void set_sideways_drop_vertical_scalar(float drop_vertical_scalar);
	float get_sideways_drop_vertical_scalar() const;
	void set_sideways_drop_horizontal_scalar(float drop_horizontal_scalar);
	float get_sideways_drop_horizontal_scalar() const;
	void set_sideways_drop_angle(float drop_angle);
	float get_sideways_drop_angle() const;
	void set_sideways_contact_point_ease(float contact_point_ease);
	float get_sideways_contact_point_ease() const;
	void set_sideways_contact_point_ease_scalar(float contact_point_ease_scalar);
	float get_sideways_contact_point_ease_scalar() const;
	void set_sideways_scaling_ease(float scaling_ease);
	float get_sideways_scaling_ease() const;

	void reset_all_bones();

	static std::pair<float, float> trig_angles(Vector3 const &length1,
			Vector3 const &length2,
			Vector3 const &length3);
	static HashMap<BoneId, Quaternion>
	solve_trig_ik(Ref<RenIKLimb> limb, Transform3D limb_parent_transform,
			Transform3D target);

	static HashMap<BoneId, Basis>
	solve_trig_ik_redux(Ref<RenIKLimb> limb, Transform3D limb_parent_transform,
			Transform3D target);

	static HashMap<BoneId, Quaternion>
	solve_ifabrik(Ref<RenIKChain> chain, Transform3D chain_parent_transform,
			Transform3D target, float threshold, int loopLimit);

private:
	// Setup -------------------------
	bool live_preview = false;
	// The Skeleton3D
	//NodePath skeleton_path = NodePath("..");
	Skeleton3D *skeleton = nullptr;

	// IK Targets
	NodePath head_target_path;
	NodePath hand_left_target_path;
	NodePath hand_right_target_path;
	NodePath hip_target_path;
	NodePath foot_left_target_path;
	NodePath foot_right_target_path;


	Node3D *head_target_spatial = nullptr;
	Node3D *hand_left_target_spatial = nullptr;
	Node3D *hand_right_target_spatial = nullptr;
	Node3D *hip_target_spatial = nullptr;

	// 脚
	Node3D *foot_left_target_spatial = nullptr;
	Node3D *foot_right_target_spatial = nullptr;

	// IK ADJUSTMENTS --------------------
	RenIKPlacement placement;
	String head_bone_name;
	String hip_bone_name;

	String hand_left_bone_name;
	String lower_left_arm_bone_name;
	String upper_left_arm_bone_name;

	String hand_right_bone_name;
	String lower_right_arm_bone_name;
	String upper_right_arm_bone_name;

	String foot_left_bone_name;
	String lower_left_leg_bone_name;
	String upper_left_leg_bone_name;

	String foot_right_bone_name;
	String lower_right_leg_bone_name;
	String upper_right_leg_bone_name;

	BoneId hip = -1;
	BoneId head = -1;
	// 脊柱
	Ref<RenIKChain> spine_chain;

	// 手
	Ref<RenIKLimb> limb_arm_left;
	Ref<RenIKLimb> limb_arm_right;

	// 腿
	Ref<RenIKLimb> limb_leg_left;
	Ref<RenIKLimb> limb_leg_right;
	float shoulder_influence = 0.25;
	bool left_shoulder_enabled = false;
	bool right_shoulder_enabled = false;
	Vector3 left_shoulder_offset;
	Vector3 right_shoulder_offset;
	Vector3 left_shoulder_pole_offset;
	Vector3 right_shoulder_pole_offset;

	bool hip_placement = true;
	bool foot_placement = true;
	bool headTrackerEnabled = true;
	bool leftHandTrackerEnabled = true;
	bool rightHandTrackerEnabled = true;
	bool hipTrackerEnabled = true;
	bool leftFootTrackerEnabled = true;
	bool rightFootTrackerEnabled = true;

	void calculate_hip_offset();
	Vector<BoneId> calculate_bone_chain(BoneId root, BoneId leaf);
};

#endif

#endif // RENIK_H

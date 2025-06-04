/**************************************************************************/
/*  test_convert_transform_modifier_3d.h                                  */
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

#pragma once

#include "tests/test_macros.h"

#include "scene/3d/bone_attachment_3d.h"
#include "scene/3d/convert_transform_modifier_3d.h"

namespace TestConvertTransformModifier3D {

Transform3D make_random_transform_3d(int p_seed) {
	RandomNumberGenerator rng;
	rng.set_seed(p_seed);

	Vector3 pos;
	pos.x = rng.randf_range(-10.0, 10.0);
	rng.set_seed(++p_seed);
	pos.y = rng.randf_range(-10.0, 10.0);
	rng.set_seed(++p_seed);
	pos.z = rng.randf_range(-10.0, 10.0);
	rng.set_seed(++p_seed);

	Quaternion rot;
	rot.x = rng.randf_range(-1.0, 1.0);
	rng.set_seed(++p_seed);
	rot.y = rng.randf_range(-1.0, 1.0);
	rng.set_seed(++p_seed);
	rot.z = rng.randf_range(-1.0, 1.0);
	rng.set_seed(++p_seed);
	rot.w = rng.randf_range(-1.0, 1.0);
	rng.set_seed(++p_seed);
	rot = rot.normalized();

	Vector3 scl;
	scl.x = rng.randf_range(0.5, 2.0);
	rng.set_seed(++p_seed);
	scl.y = rng.randf_range(0.5, 2.0);
	rng.set_seed(++p_seed);
	scl.z = rng.randf_range(0.5, 2.0);
	rng.set_seed(++p_seed);

	return Transform3D(Basis(rot).scaled(scl), pos);
}

TEST_CASE("[SceneTree][ConvertTransformModifier3D]") {
	SceneTree *tree = SceneTree::get_singleton();
	int seed = 12345;
	Skeleton3D *skeleton = memnew(Skeleton3D);
	ConvertTransformModifier3D *mod = memnew(ConvertTransformModifier3D);

	// Instead of awaiting the process to wait to finish deferred process and watch "skeleton_updated" signal,
	// force notify NOTIFICATION_UPDATE_SKELETON and get the modified pose from the BoneAttachment's transform.
	BoneAttachment3D *modified = memnew(BoneAttachment3D);

	tree->get_root()->add_child(skeleton);

	int root = skeleton->add_bone("root");
	skeleton->set_bone_rest(root, make_random_transform_3d(++seed));
	skeleton->set_bone_pose(root, make_random_transform_3d(++seed));

	int apl_root = skeleton->add_bone("apl_root");
	skeleton->set_bone_parent(apl_root, root);
	skeleton->set_bone_rest(apl_root, make_random_transform_3d(++seed));
	skeleton->set_bone_pose(apl_root, make_random_transform_3d(++seed));

	int apl_bone = skeleton->add_bone("apl_bone");
	skeleton->set_bone_parent(apl_bone, apl_root);
	skeleton->set_bone_rest(apl_bone, make_random_transform_3d(++seed));
	skeleton->set_bone_pose(apl_bone, make_random_transform_3d(++seed));

	int tgt_root = skeleton->add_bone("tgt_root");
	skeleton->set_bone_parent(tgt_root, root);
	skeleton->set_bone_rest(tgt_root, make_random_transform_3d(++seed));
	skeleton->set_bone_pose(tgt_root, make_random_transform_3d(++seed));

	int tgt_bone = skeleton->add_bone("tgt_bone");
	skeleton->set_bone_parent(tgt_bone, tgt_root);
	skeleton->set_bone_rest(tgt_bone, make_random_transform_3d(++seed));
	skeleton->set_bone_pose(tgt_bone, make_random_transform_3d(++seed));

	skeleton->add_child(mod);
	skeleton->add_child(modified);
	modified->set_rotation_edit_mode(Node3D::ROTATION_EDIT_MODE_QUATERNION);
	modified->set_bone_idx(apl_bone);

	mod->set_setting_count(1);
	mod->set_reference_bone(0, tgt_bone);
	mod->set_apply_bone(0, apl_bone);

	mod->set_reference_axis(0, Vector3::AXIS_X);
	mod->set_apply_axis(0, Vector3::AXIS_Y);

	// ===== [ConvertTransformModifier3D] Position x to y =====
	mod->set_reference_transform_mode(0, ConvertTransformModifier3D::TRANSFORM_MODE_POSITION);
	mod->set_reference_range_min(0, -100.0);
	mod->set_reference_range_max(0, 100.0);
	mod->set_apply_transform_mode(0, ConvertTransformModifier3D::TRANSFORM_MODE_POSITION);
	mod->set_apply_range_min(0, -100.0);
	mod->set_apply_range_max(0, 100.0);

	SUBCASE("[ConvertTransformModifier3D] Position x to y, additive=false, relative=false") {
		mod->set_additive(0, false);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK(Math::is_equal_approx(
				skeleton->get_bone_pose_position(tgt_bone).x,
				(skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin.y));
	}

	SUBCASE("[ConvertTransformModifier3D] Position x to y, additive=true, relative=false") {
		mod->set_additive(0, true);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK(Math::is_equal_approx(
				skeleton->get_bone_pose_position(tgt_bone).x,
				((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)).y));
	}

	SUBCASE("[ConvertTransformModifier3D] Position x to y, additive=false, relative=true") {
		mod->set_additive(0, false);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK(Math::is_equal_approx(
				(skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).x,
				((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_rest(apl_bone).origin).y));
	}

	SUBCASE("[ConvertTransformModifier3D] Position x to y, additive=true, relative=true") {
		mod->set_additive(0, true);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK(Math::is_equal_approx(
				(skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).x,
				((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)).y));
	}

	// ===== [ConvertTransformModifier3D] Rotation (roll) x to y =====
	mod->set_reference_transform_mode(0, ConvertTransformModifier3D::TRANSFORM_MODE_ROTATION);
	mod->set_reference_range_min(0, -180.0);
	mod->set_reference_range_max(0, 180.0);
	mod->set_apply_transform_mode(0, ConvertTransformModifier3D::TRANSFORM_MODE_ROTATION);
	mod->set_apply_range_min(0, -180.0);
	mod->set_apply_range_max(0, 180.0);

	SUBCASE("[ConvertTransformModifier3D] Rotation (roll) x to y, additive=false, relative=false") {
		mod->set_additive(0, false);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK(Math::is_equal_approx(
				BoneConstraint3D::get_roll_angle(skeleton->get_bone_pose_rotation(tgt_bone), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_X)),
				BoneConstraint3D::get_roll_angle((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion(), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_Y))));
	}

	SUBCASE("[ConvertTransformModifier3D] Rotation (roll) x to y, additive=true, relative=false") {
		mod->set_additive(0, true);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK(Math::is_equal_approx(
				BoneConstraint3D::get_roll_angle(skeleton->get_bone_pose_rotation(tgt_bone), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_X)),
				BoneConstraint3D::get_roll_angle(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion(), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_Y))));
	}

	SUBCASE("[ConvertTransformModifier3D] Rotation (roll) x to y, additive=false, relative=true") {
		mod->set_additive(0, false);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK(Math::is_equal_approx(
				BoneConstraint3D::get_roll_angle(skeleton->get_bone_rest(tgt_bone).basis.get_rotation_quaternion().inverse() * skeleton->get_bone_pose_rotation(tgt_bone), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_X)),
				BoneConstraint3D::get_roll_angle(skeleton->get_bone_rest(apl_bone).basis.get_rotation_quaternion().inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion(), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_Y))));
	}

	SUBCASE("[ConvertTransformModifier3D] Rotation (roll) x to y, additive=true, relative=true") {
		mod->set_additive(0, true);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK(Math::is_equal_approx(
				BoneConstraint3D::get_roll_angle(skeleton->get_bone_rest(tgt_bone).basis.get_rotation_quaternion().inverse() * skeleton->get_bone_pose_rotation(tgt_bone), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_X)),
				BoneConstraint3D::get_roll_angle(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion(), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_Y))));
	}

	// ===== [ConvertTransformModifier3D] Scale x to y =====
	mod->set_reference_transform_mode(0, ConvertTransformModifier3D::TRANSFORM_MODE_SCALE);
	mod->set_reference_range_min(0, 0);
	mod->set_reference_range_max(0, 10.0);
	mod->set_apply_transform_mode(0, ConvertTransformModifier3D::TRANSFORM_MODE_SCALE);
	mod->set_apply_range_min(0, 0);
	mod->set_apply_range_max(0, 10.0);

	SUBCASE("[ConvertTransformModifier3D] Scale x to y, additive=false, relative=false") {
		mod->set_additive(0, false);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK(Math::is_equal_approx(
				skeleton->get_bone_pose_scale(tgt_bone).x,
				(skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale().y));
	}

	SUBCASE("[ConvertTransformModifier3D] Scale x to y, additive=true, relative=false") {
		mod->set_additive(0, true);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK(Math::is_equal_approx(
				skeleton->get_bone_pose_scale(tgt_bone).x,
				((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)).y));
	}

	SUBCASE("[ConvertTransformModifier3D] Scale x to y, additive=false, relative=true") {
		mod->set_additive(0, false);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK(Math::is_equal_approx(
				(skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).x,
				((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_rest(apl_bone).basis.get_scale()).y));
	}

	SUBCASE("[ConvertTransformModifier3D] Scale x to y, additive=true, relative=true") {
		mod->set_additive(0, true);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK(Math::is_equal_approx(
				(skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).x,
				((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)).y));
	}

	memdelete(modified);
	memdelete(mod);
	memdelete(skeleton);
}
} // namespace TestConvertTransformModifier3D

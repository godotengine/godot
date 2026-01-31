/**************************************************************************/
/*  test_copy_transform_modifier_3d.h                                     */
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
#include "scene/3d/copy_transform_modifier_3d.h"

namespace TestCopyTransformModifier3D {

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

Vector3 flip_x(Vector3 p_pos) {
	return Vector3(-p_pos.x, p_pos.y, p_pos.z);
}

Vector3 flip_xy(Vector3 p_pos) {
	return Vector3(-p_pos.x, -p_pos.y, p_pos.z);
}

Vector3 flip_all(Vector3 p_pos) {
	return -p_pos;
}

// Quaternion's phase can be confused by inversion, it is aligned via casting to Basis.

Quaternion flip_x(Quaternion p_rot) {
	return Basis(Quaternion(-p_rot.x, p_rot.y, p_rot.z, p_rot.w).normalized()).get_rotation_quaternion();
}

Quaternion flip_xy(Quaternion p_rot) {
	return Basis(Quaternion(-p_rot.x, -p_rot.y, p_rot.z, p_rot.w).normalized()).get_rotation_quaternion();
}

Quaternion flip_all(Quaternion p_rot) {
	return Basis(p_rot.inverse()).get_rotation_quaternion();
}

Vector3 inv_x(Vector3 p_scl) {
	return Vector3(1.0 / p_scl.x, p_scl.y, p_scl.z);
}

Vector3 inv_xy(Vector3 p_scl) {
	return Vector3(1.0 / p_scl.x, 1.0 / p_scl.y, p_scl.z);
}

Vector3 inv_all(Vector3 p_scl) {
	return Vector3(1.0, 1.0, 1.0) / p_scl;
}

TEST_CASE("[SceneTree][CopyTransformModifier3D]") {
	SceneTree *tree = SceneTree::get_singleton();
	int seed = 12345;
	Skeleton3D *skeleton = memnew(Skeleton3D);
	CopyTransformModifier3D *mod = memnew(CopyTransformModifier3D);

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

	mod->set_copy_position(0, true);
	mod->set_copy_rotation(0, true);
	mod->set_copy_scale(0, true);

	// ===== [CopyTransformModifier3D] Enable 1 axis =====
	mod->set_axis_x_enabled(0, true);
	mod->set_axis_y_enabled(0, false);
	mod->set_axis_z_enabled(0, false);
	mod->set_axis_x_inverted(0, false);
	mod->set_axis_y_inverted(0, false);
	mod->set_axis_z_inverted(0, false);

	SUBCASE("[CopyTransformModifier3D] Enable 1 axis, additive=false, relative=false") {
		mod->set_additive(0, false);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(Math::is_equal_approx(
							  skeleton->get_bone_pose_position(tgt_bone).x,
							  (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin.x),
				"Position x is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  BoneConstraint3D::get_roll_angle(skeleton->get_bone_pose_rotation(tgt_bone), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_X)),
							  BoneConstraint3D::get_roll_angle((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion(), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_X))),
				"Rotation x (roll x) is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  skeleton->get_bone_pose_scale(tgt_bone).x,
							  (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale().x),
				"Scale x is copied correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable 1 axis, additive=true, relative=false") {
		mod->set_additive(0, true);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(Math::is_equal_approx(
							  skeleton->get_bone_pose_position(tgt_bone).x,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)).x),
				"Position x is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  BoneConstraint3D::get_roll_angle(skeleton->get_bone_pose_rotation(tgt_bone), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_X)),
							  BoneConstraint3D::get_roll_angle(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion(), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_X))),
				"Rotation x (roll x) is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  skeleton->get_bone_pose_scale(tgt_bone).x,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)).x),
				"Scale x is copied correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable 1 axis, additive=false, relative=true") {
		mod->set_additive(0, false);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(Math::is_equal_approx(
							  (skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).x,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_rest(apl_bone).origin).x),
				"Position x is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  BoneConstraint3D::get_roll_angle(skeleton->get_bone_rest(tgt_bone).basis.get_rotation_quaternion().inverse() * skeleton->get_bone_pose_rotation(tgt_bone), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_X)),
							  BoneConstraint3D::get_roll_angle(skeleton->get_bone_rest(apl_bone).basis.get_rotation_quaternion().inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion(), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_X))),
				"Rotation x (roll x) is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  (skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).x,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_rest(apl_bone).basis.get_scale()).x),
				"Scale x is copied correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable 1 axis, additive=true, relative=true") {
		mod->set_additive(0, true);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(Math::is_equal_approx(
							  (skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).x,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)).x),
				"Position x is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  BoneConstraint3D::get_roll_angle(skeleton->get_bone_rest(tgt_bone).basis.get_rotation_quaternion().inverse() * skeleton->get_bone_pose_rotation(tgt_bone), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_X)),
							  BoneConstraint3D::get_roll_angle(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion(), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_X))),
				"Rotation x (roll x) is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  (skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).x,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)).x),
				"Scale x is copied correctly.");
	}

	// ===== [CopyTransformModifier3D] Enable 2 axes =====
	mod->set_axis_x_enabled(0, true);
	mod->set_axis_y_enabled(0, true);
	mod->set_axis_z_enabled(0, false);
	mod->set_axis_x_inverted(0, false);
	mod->set_axis_y_inverted(0, false);
	mod->set_axis_z_inverted(0, false);

	SUBCASE("[CopyTransformModifier3D] Enable 2 axes, additive=false, relative=false") {
		mod->set_additive(0, false);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(Math::is_equal_approx(
							  skeleton->get_bone_pose_position(tgt_bone).x,
							  (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin.x),
				"Position x is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  skeleton->get_bone_pose_position(tgt_bone).y,
							  (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin.y),
				"Position y is copied correctly.");
		CHECK_MESSAGE(Math::is_zero_approx(
							  BoneConstraint3D::get_roll_angle((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion(), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_Z))),
				"Rotation z (roll z) is zero correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  skeleton->get_bone_pose_scale(tgt_bone).x,
							  (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale().x),
				"Scale x is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  skeleton->get_bone_pose_scale(tgt_bone).y,
							  (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale().y),
				"Scale y is copied correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable 2 axes, additive=true, relative=false") {
		mod->set_additive(0, true);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(Math::is_equal_approx(
							  skeleton->get_bone_pose_position(tgt_bone).x,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)).x),
				"Position x is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  skeleton->get_bone_pose_position(tgt_bone).y,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)).y),
				"Position y is copied correctly.");
		CHECK_MESSAGE(Math::is_zero_approx(
							  BoneConstraint3D::get_roll_angle(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion(), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_Z))),
				"Rotation z (roll z) is zero correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  skeleton->get_bone_pose_scale(tgt_bone).x,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)).x),
				"Scale x is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  skeleton->get_bone_pose_scale(tgt_bone).y,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)).y),
				"Scale y is copied correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable 2 axes, additive=false, relative=true") {
		mod->set_additive(0, false);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(Math::is_equal_approx(
							  (skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).x,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_rest(apl_bone).origin).x),
				"Position x is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  (skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).y,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_rest(apl_bone).origin).y),
				"Position y is copied correctly.");
		CHECK_MESSAGE(Math::is_zero_approx(
							  BoneConstraint3D::get_roll_angle(skeleton->get_bone_rest(apl_bone).basis.get_rotation_quaternion().inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion(), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_Z))),
				"Rotation z (roll z) is zero correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  (skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).x,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_rest(apl_bone).basis.get_scale()).x),
				"Scale x is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  (skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).y,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_rest(apl_bone).basis.get_scale()).y),
				"Scale y is copied correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable 2 axes, additive=true, relative=true") {
		mod->set_additive(0, true);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(Math::is_equal_approx(
							  (skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).x,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)).x),
				"Position x is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  (skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).y,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)).y),
				"Position y is copied correctly.");
		CHECK_MESSAGE(Math::is_zero_approx(
							  BoneConstraint3D::get_roll_angle(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion(), BoneConstraint3D::get_vector_from_axis(Vector3::AXIS_Z))),
				"Rotation z (roll z) is zero correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  (skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).x,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)).x),
				"Scale x is copied correctly.");
		CHECK_MESSAGE(Math::is_equal_approx(
							  (skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).y,
							  ((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)).y),
				"Scale y is copied correctly.");
	}

	// ===== [CopyTransformModifier3D] Enable all axes =====
	mod->set_axis_x_enabled(0, true);
	mod->set_axis_y_enabled(0, true);
	mod->set_axis_z_enabled(0, true);
	mod->set_axis_x_inverted(0, false);
	mod->set_axis_y_inverted(0, false);
	mod->set_axis_z_inverted(0, false);

	SUBCASE("[CopyTransformModifier3D] Enable all axes, additive=false, relative=false") {
		mod->set_additive(0, false);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(skeleton->get_bone_pose_position(tgt_bone).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin), "Position is copied correctly.");
		CHECK_MESSAGE(Basis(skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(Basis((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion())), "Rotation is copied correctly.");
		CHECK_MESSAGE(skeleton->get_bone_pose_scale(tgt_bone).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale()), "Scale is copied correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable all axes, additive=true, relative=false") {
		mod->set_additive(0, true);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(skeleton->get_bone_pose_position(tgt_bone).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)), "Position is copied correctly.");
		CHECK_MESSAGE(Basis(skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()), "Rotation is copied correctly.");
		CHECK_MESSAGE(skeleton->get_bone_pose_scale(tgt_bone).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)), "Scale is copied correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable all axes, additive=false, relative=true") {
		mod->set_additive(0, false);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE((skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_rest(apl_bone).origin), "Position is copied correctly.");
		CHECK_MESSAGE(Basis(skeleton->get_bone_rest(tgt_bone).basis.get_rotation_quaternion().inverse() * skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(Basis(skeleton->get_bone_rest(apl_bone).basis.get_rotation_quaternion().inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion())), "Rotation is copied correctly.");
		CHECK_MESSAGE((skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_rest(apl_bone).basis.get_scale()), "Scale is copied correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable all axes, additive=true, relative=true") {
		mod->set_additive(0, true);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE((skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)), "Position is copied correctly.");
		CHECK_MESSAGE(Basis(skeleton->get_bone_rest(tgt_bone).basis.get_rotation_quaternion().inverse() * skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()), "Rotation is copied correctly.");
		CHECK_MESSAGE((skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)), "Scale is copied correctly.");
	}

	// ===== [CopyTransformModifier3D] Enable all axes, invert 1 axis =====
	mod->set_axis_x_enabled(0, true);
	mod->set_axis_y_enabled(0, true);
	mod->set_axis_z_enabled(0, true);
	mod->set_axis_x_inverted(0, true);
	mod->set_axis_y_inverted(0, false);
	mod->set_axis_z_inverted(0, false);

	SUBCASE("[CopyTransformModifier3D] Enable all axes, invert 1 axis, additive=false, relative=false") {
		mod->set_additive(0, false);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(flip_x(skeleton->get_bone_pose_position(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin), "Position is copied/inverted correctly.");
		CHECK_MESSAGE(flip_x(skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()), "Rotation is copied/inverted correctly.");
		CHECK_MESSAGE(inv_x(skeleton->get_bone_pose_scale(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale()), "Scale is copied/inverted correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable all axes, invert 1 axis, additive=true, relative=false") {
		mod->set_additive(0, true);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(flip_x(skeleton->get_bone_pose_position(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)), "Position is copied/inverted correctly.");
		CHECK_MESSAGE(flip_x(skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(Basis(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()).get_rotation_quaternion()), "Rotation is copied/inverted correctly.");
		CHECK_MESSAGE(inv_x(skeleton->get_bone_pose_scale(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)), "Scale is copied/inverted correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable all axes, invert 1 axis, additive=false, relative=true") {
		mod->set_additive(0, false);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(flip_x(skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_rest(apl_bone).origin), "Position is copied/inverted correctly.");
		CHECK_MESSAGE(flip_x(skeleton->get_bone_rest(tgt_bone).basis.get_rotation_quaternion().inverse() * skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(Basis(skeleton->get_bone_rest(apl_bone).basis.get_rotation_quaternion().inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()).get_rotation_quaternion()), "Rotation is copied/inverted correctly.");
		CHECK_MESSAGE(inv_x(skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_rest(apl_bone).basis.get_scale()), "Scale is copied/inverted correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable all axes, invert 1 axis, additive=true, relative=true") {
		mod->set_additive(0, true);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(flip_x(skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)), "Position is copied/inverted correctly.");
		CHECK_MESSAGE(flip_x(skeleton->get_bone_rest(tgt_bone).basis.get_rotation_quaternion().inverse() * skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(Basis(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()).get_rotation_quaternion()), "Rotation is copied/inverted correctly.");
		CHECK_MESSAGE(inv_x(skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)), "Scale is copied/inverted correctly.");
	}

	// ===== [CopyTransformModifier3D] Enable all axes, invert 2 axes =====
	mod->set_axis_x_enabled(0, true);
	mod->set_axis_y_enabled(0, true);
	mod->set_axis_z_enabled(0, true);
	mod->set_axis_x_inverted(0, true);
	mod->set_axis_y_inverted(0, true);
	mod->set_axis_z_inverted(0, false);

	SUBCASE("[CopyTransformModifier3D] Enable all axes, invert 2 axes, additive=false, relative=false") {
		mod->set_additive(0, false);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(flip_xy(skeleton->get_bone_pose_position(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin), "Position is copied/inverted correctly.");
		CHECK_MESSAGE(flip_xy(skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()), "Rotation is copied/inverted correctly.");
		CHECK_MESSAGE(inv_xy(skeleton->get_bone_pose_scale(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale()), "Scale is copied/inverted correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable all axes, invert 2 axes, additive=true, relative=false") {
		mod->set_additive(0, true);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(flip_xy(skeleton->get_bone_pose_position(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)), "Position is copied/inverted correctly.");
		CHECK_MESSAGE(flip_xy(skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(Basis(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()).get_rotation_quaternion()), "Rotation is copied/inverted correctly.");
		CHECK_MESSAGE(inv_xy(skeleton->get_bone_pose_scale(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)), "Scale is copied/inverted correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable all axes, invert 2 axes, additive=false, relative=true") {
		mod->set_additive(0, false);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(flip_xy(skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_rest(apl_bone).origin), "Position is copied/inverted correctly.");
		CHECK_MESSAGE(flip_xy(skeleton->get_bone_rest(tgt_bone).basis.get_rotation_quaternion().inverse() * skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(Basis(skeleton->get_bone_rest(apl_bone).basis.get_rotation_quaternion().inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()).get_rotation_quaternion()), "Rotation is copied/inverted correctly.");
		CHECK_MESSAGE(inv_xy(skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_rest(apl_bone).basis.get_scale()), "Scale is copied/inverted correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable all axes, invert 2 axes, additive=true, relative=true") {
		mod->set_additive(0, true);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(flip_xy(skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)), "Position is copied/inverted correctly.");
		CHECK_MESSAGE(flip_xy(skeleton->get_bone_rest(tgt_bone).basis.get_rotation_quaternion().inverse() * skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(Basis(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()).get_rotation_quaternion()), "Rotation is copied/inverted correctly.");
		CHECK_MESSAGE(inv_xy(skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)), "Scale is copied/inverted correctly.");
	}

	// ===== [CopyTransformModifier3D] Enable all axes, invert all axes =====
	mod->set_axis_x_enabled(0, true);
	mod->set_axis_y_enabled(0, true);
	mod->set_axis_z_enabled(0, true);
	mod->set_axis_x_inverted(0, true);
	mod->set_axis_y_inverted(0, true);
	mod->set_axis_z_inverted(0, true);

	SUBCASE("[CopyTransformModifier3D] Enable all axes, invert all axes, additive=false, relative=false") {
		mod->set_additive(0, false);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(flip_all(skeleton->get_bone_pose_position(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin), "Position is copied/inverted correctly.");
		CHECK_MESSAGE(flip_all(skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()), "Rotation is copied/inverted correctly.");
		CHECK_MESSAGE(inv_all(skeleton->get_bone_pose_scale(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale()), "Scale is copied/inverted correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable all axes, invert all axes, additive=true, relative=false") {
		mod->set_additive(0, true);
		mod->set_relative(0, false);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(flip_all(skeleton->get_bone_pose_position(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)), "Position is copied/inverted correctly.");
		CHECK_MESSAGE(flip_all(skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(Basis(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()).get_rotation_quaternion()), "Rotation is copied/inverted correctly.");
		CHECK_MESSAGE(inv_all(skeleton->get_bone_pose_scale(tgt_bone)).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)), "Scale is copied/inverted correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable all axes, invert all axes, additive=false, relative=true") {
		mod->set_additive(0, false);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(flip_all(skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_rest(apl_bone).origin), "Position is copied/inverted correctly.");
		CHECK_MESSAGE(flip_all(skeleton->get_bone_rest(tgt_bone).basis.get_rotation_quaternion().inverse() * skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(Basis(skeleton->get_bone_rest(apl_bone).basis.get_rotation_quaternion().inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()).get_rotation_quaternion()), "Rotation is copied/inverted correctly.");
		CHECK_MESSAGE(inv_all(skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_rest(apl_bone).basis.get_scale()), "Scale is copied/inverted correctly.");
	}

	SUBCASE("[CopyTransformModifier3D] Enable all axes, invert all axes, additive=true, relative=true") {
		mod->set_additive(0, true);
		mod->set_relative(0, true);
		skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
		CHECK_MESSAGE(flip_all(skeleton->get_bone_pose_position(tgt_bone) - skeleton->get_bone_rest(tgt_bone).origin).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).origin - skeleton->get_bone_pose_position(apl_bone)), "Position is copied/inverted correctly.");
		CHECK_MESSAGE(flip_all(skeleton->get_bone_rest(tgt_bone).basis.get_rotation_quaternion().inverse() * skeleton->get_bone_pose_rotation(tgt_bone)).is_equal_approx(Basis(skeleton->get_bone_pose_rotation(apl_bone).inverse() * (skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_rotation_quaternion()).get_rotation_quaternion()), "Rotation is copied/inverted correctly.");
		CHECK_MESSAGE(inv_all(skeleton->get_bone_pose_scale(tgt_bone) / skeleton->get_bone_rest(tgt_bone).basis.get_scale()).is_equal_approx((skeleton->get_bone_global_pose(apl_root).affine_inverse() * modified->get_transform()).basis.get_scale() / skeleton->get_bone_pose_scale(apl_bone)), "Scale is copied/inverted correctly.");
	}

	memdelete(modified);
	memdelete(mod);
	memdelete(skeleton);
}
} // namespace TestCopyTransformModifier3D

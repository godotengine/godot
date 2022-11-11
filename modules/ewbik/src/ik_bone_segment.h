/*************************************************************************/
/*  ik_bone_segment.h                                                    */
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

#ifndef IK_BONE_SEGMENT_H
#define IK_BONE_SEGMENT_H

#include "ik_bone_3d.h"
#include "ik_effector_3d.h"
#include "ik_effector_template.h"
#include "math/qcp.h"
#include "scene/3d/skeleton_3d.h"

#include "core/io/resource.h"
#include "core/object/ref_counted.h"

class IKEffector3D;
class IKBone3D;
class IKLimitCone;

class IKBoneSegment : public Resource {
	GDCLASS(IKBoneSegment, Resource);
	Ref<IKBone3D> root;
	Ref<IKBone3D> tip;
	Vector<Ref<IKBone3D>> bones;
	Vector<Ref<IKBone3D>> pinned_bones;
	Vector<Ref<IKBoneSegment>> child_segments; // Contains only direct child chains that end with effectors or have child that end with effectors
	Ref<IKBoneSegment> parent_segment;
	Ref<IKBoneSegment> root_segment;
	Vector<Ref<IKEffector3D>> effector_list;
	PackedVector3Array target_headings;
	PackedVector3Array tip_headings;
	Vector<real_t> heading_weights;
	BoneId ewbik_root_bone = -1;
	BoneId ewbik_tip_bone = -1;
	int32_t idx_eff_i = -1, idx_eff_f = -1;
	Skeleton3D *skeleton = nullptr;
	bool pinned_descendants = false;
	bool has_pinned_descendants();
	void enable_pinned_descendants();
	BoneId find_root_bone_id(BoneId p_bone);
	void update_target_headings(Ref<IKBone3D> p_for_bone, Vector<real_t> *r_weights, PackedVector3Array *r_htarget);
	void update_tip_headings(Ref<IKBone3D> p_for_bone, PackedVector3Array *r_heading_tip);
	void set_optimal_rotation(Ref<IKBone3D> p_for_bone, PackedVector3Array *r_htip, PackedVector3Array *r_heading_tip, Vector<real_t> *r_weights, float p_dampening = -1, bool p_translate = false);
	void qcp_solver(real_t p_damp, bool p_translate);
	void update_optimal_rotation(Ref<IKBone3D> p_for_bone, real_t p_damp, bool p_translate);
	float get_manual_msd(const PackedVector3Array &r_htip, const PackedVector3Array &r_htarget, const Vector<real_t> &p_weights);
	HashMap<BoneId, Ref<IKBone3D>> bone_map;
	// This orientation angle is a cos(angle/2) representation.
	Quaternion set_quadrance_angle(Quaternion p_quat, real_t p_cos_half_angle) const;

protected:
	static void _bind_methods();

public:
	static Quaternion clamp_to_angle(Quaternion p_quat, real_t p_angle);
	static Quaternion clamp_to_quadrance_angle(Quaternion p_quat, real_t p_cos_half_angle);
	_FORCE_INLINE_ static real_t cos(real_t p_angle) {
		// https://stackoverflow.com/questions/18662261/fastest-implementation-of-sine-cosine-and-square-root-in-c-doesnt-need-to-b/28050328#28050328
		real_t x = real_t(0.5) * p_angle;
		constexpr real_t tp = 1. / (2. * Math_PI);
		x *= tp;
		x -= real_t(.25) + Math::floor(x + real_t(.25));
		x *= real_t(16.) * (Math::abs(x) - real_t(.5));
		// BEGIN EXTRA_PRECISION
		x += real_t(.225) * x * (Math::abs(x) - real_t(1.));
		// END EXTRA_PRECISION
		return x;
	}
	static void recursive_create_headings_arrays_for(Ref<IKBoneSegment> p_bone_segment);
	void create_headings_arrays();
	void recursive_create_penalty_array(Ref<IKBoneSegment> p_bone_segment, Vector<Vector<real_t>> &r_penalty_array, Vector<Ref<IKBone3D>> &r_pinned_bones, real_t p_falloff);
	Ref<IKBoneSegment> get_parent_segment();
	void segment_solver(real_t p_damp);
	Ref<IKBone3D> get_root() const;
	Ref<IKBone3D> get_tip() const;
	bool is_pinned() const;
	Vector<Ref<IKBoneSegment>> get_child_segments() const;
	void create_bone_list(Vector<Ref<IKBone3D>> &p_list, bool p_recursive = false, bool p_debug_skeleton = false) const;
	Vector<Ref<IKBone3D>> get_bone_list() const;
	Ref<IKBone3D> get_ik_bone(BoneId p_bone);
	void generate_default_segments_from_root(Vector<Ref<IKEffectorTemplate>> &p_pins, BoneId p_root_bone, BoneId p_tip_bone);
	void update_pinned_list(Vector<Vector<real_t>> &r_weights);
	IKBoneSegment() {}
	IKBoneSegment(Skeleton3D *p_skeleton, StringName p_root_bone_name, Vector<Ref<IKEffectorTemplate>> &p_pins, const Ref<IKBoneSegment> &p_parent = nullptr,
			BoneId root = -1, BoneId tip = -1);
	~IKBoneSegment() {}
};

#endif // IK_BONE_SEGMENT_H

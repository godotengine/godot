/**************************************************************************/
/*  ik_bone_segment_3d.h                                                  */
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

#ifndef IK_BONE_SEGMENT_3D_H
#define IK_BONE_SEGMENT_3D_H

#include "ik_bone_3d.h"
#include "ik_effector_3d.h"
#include "ik_effector_template_3d.h"
#include "math/qcp.h"
#include "scene/3d/skeleton_3d.h"

#include "core/io/resource.h"
#include "core/object/ref_counted.h"

class IKEffector3D;
class IKBone3D;
class IKLimitCone3D;

class IKBoneSegment3D : public Resource {
	GDCLASS(IKBoneSegment3D, Resource);
	Ref<IKBone3D> root;
	Ref<IKBone3D> tip;
	Vector<Ref<IKBone3D>> bones;
	Vector<Ref<IKBone3D>> pinned_bones;
	TypedArray<IKBoneSegment3D> child_segments; // Contains only direct child chains that end with effectors or have child that end with effectors
	Ref<IKBoneSegment3D> parent_segment;
	Vector<Ref<IKEffector3D>> effector_list;
	PackedVector3Array target_headings;
	PackedVector3Array tip_headings;
	PackedVector3Array tip_headings_uniform;
	Vector<real_t> heading_weights;
	BoneId many_bone_ik_tip_bone = -1;
	int32_t idx_eff_i = -1, idx_eff_f = -1;
	Skeleton3D *skeleton = nullptr;
	bool pinned_descendants = false;
	real_t previous_deviation = 0;
	int32_t default_stabilizing_pass_count = 1; // Move to the stabilizing pass to the ik solver.
	bool has_pinned_descendants();
	void enable_pinned_descendants();
	void update_target_headings(Ref<IKBone3D> p_for_bone, Vector<real_t> *r_weights, PackedVector3Array *r_htarget);
	void update_tip_headings(Ref<IKBone3D> p_for_bone, PackedVector3Array *r_heading_tip, bool p_uniform);
	void set_optimal_rotation(Ref<IKBone3D> p_for_bone, PackedVector3Array *r_htip, PackedVector3Array *r_heading_tip, Vector<real_t> *r_weights, float p_dampening = -1, bool p_translate = false, bool p_constraint_mode = false);
	void qcp_solver(const Vector<float> &p_damp, float p_default_damp, bool p_translate, bool p_constraint_mode);
	void update_optimal_rotation(Ref<IKBone3D> p_for_bone, real_t p_damp, bool p_translate, bool p_constraint_mode);
	float get_manual_msd(const PackedVector3Array &r_htip, const PackedVector3Array &r_htarget, const Vector<real_t> &p_weights);

protected:
	static void _bind_methods();

public:
	Ref<IKBone3D> get_root();
	int32_t get_default_stabilizing_pass_count();
	void set_default_stabilizing_pass_count(int32_t p_count);
	const double evec_prec = static_cast<double>(1E-6);
	const double eval_prec = static_cast<double>(1E-11);
	static Quaternion clamp_to_quadrance_angle(Quaternion p_quat, real_t p_cos_half_angle);
	static void recursive_create_headings_arrays_for(Ref<IKBoneSegment3D> p_bone_segment);
	void create_headings_arrays();
	void recursive_create_penalty_array(Ref<IKBoneSegment3D> p_bone_segment, Vector<Vector<real_t>> &r_penalty_array, Vector<Ref<IKBone3D>> &r_pinned_bones, real_t p_falloff);
	Ref<IKBoneSegment3D> get_parent_segment();
	void set_parent_segment(Ref<IKBoneSegment3D> p_parent_segment);
	void segment_solver(const Vector<float> &p_damp, float p_default_damp, bool p_constraint_mode);
	Ref<IKBone3D> get_tip() const;
	void set_tip(Ref<IKBone3D> p_tip);
	bool is_pinned() const;
	TypedArray<IKBoneSegment3D> get_child_segments() const;
	void set_child_segments(TypedArray<IKBoneSegment3D> p_child_segments);
	void create_bone_list(Vector<Ref<IKBone3D>> &p_list, bool p_recursive = false, bool p_debug_skeleton = false) const;
	Vector<Ref<IKBone3D>> get_bone_list() const;
	void set_bone_list(Vector<Ref<IKBone3D>> p_bone_list) {
		bones = p_bone_list;
	}
	Ref<IKBone3D> find_ik_bone(BoneId p_bone) const;
	void generate_default_segments_from_root(Vector<Ref<IKEffectorTemplate3D>> &p_pins, BoneId p_root_bone, BoneId p_tip_bone, ManyBoneIK3D *p_many_bone_ik);
	void update_pinned_list(Vector<Vector<real_t>> &r_weights);
	IKBoneSegment3D() {}
	IKBoneSegment3D(Skeleton3D *p_skeleton, StringName p_root_bone_name, Vector<Ref<IKEffectorTemplate3D>> &p_pins, ManyBoneIK3D *p_many_bone_ik, const Ref<IKBoneSegment3D> &p_parent = nullptr,
			BoneId root = -1, BoneId tip = -1);
	~IKBoneSegment3D() {}
};

#endif // IK_BONE_SEGMENT_3D_H

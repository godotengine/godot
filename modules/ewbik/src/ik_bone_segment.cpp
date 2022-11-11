/*************************************************************************/
/*  ik_bone_segment.cpp                                                  */
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

#include "ik_bone_segment.h"
#include "ik_effector_3d.h"
#include "ik_limit_cone.h"
#include "math/ik_node_3d.h"
#include "scene/3d/skeleton_3d.h"

Ref<IKBone3D> IKBoneSegment::get_root() const {
	return root;
}
Ref<IKBone3D> IKBoneSegment::get_tip() const {
	return tip;
}

bool IKBoneSegment::is_pinned() const {
	ERR_FAIL_NULL_V(tip, false);
	return tip->is_pinned();
}

Vector<Ref<IKBoneSegment>> IKBoneSegment::get_child_segments() const {
	return child_segments;
}

BoneId IKBoneSegment::find_root_bone_id(BoneId p_bone) {
	BoneId root_id = p_bone;
	while (skeleton->get_bone_parent(root_id) != -1) {
		root_id = skeleton->get_bone_parent(root_id);
	}
	return root_id;
}

void IKBoneSegment::generate_default_segments_from_root(Vector<Ref<IKEffectorTemplate>> &p_pins, BoneId p_root_bone, BoneId p_tip_bone) {
	Ref<IKBone3D> temp_tip = root;
	while (true) {
		if (skeleton->get_bone_parent(temp_tip->get_bone_id()) >= p_tip_bone && p_tip_bone != -1) {
			break;
		}
		Vector<BoneId> children = skeleton->get_bone_children(temp_tip->get_bone_id());
		if (children.size() > 1 || temp_tip->is_pinned()) {
			tip = temp_tip;
			Ref<IKBoneSegment> parent(this);
			for (int32_t child_i = 0; child_i < children.size(); child_i++) {
				BoneId child_bone = children[child_i];
				String child_name = skeleton->get_bone_name(child_bone);
				Ref<IKBoneSegment> child_segment = Ref<IKBoneSegment>(memnew(IKBoneSegment(skeleton, child_name, p_pins, parent, p_root_bone, p_tip_bone)));
				child_segment->generate_default_segments_from_root(p_pins, p_root_bone, p_tip_bone);
				if (child_segment->has_pinned_descendants()) {
					enable_pinned_descendants();
					child_segments.push_back(child_segment);
				}
			}
			break;
		} else if (children.size() == 1) {
			BoneId bone_id = children[0];
			Ref<IKBone3D> next = Ref<IKBone3D>(memnew(IKBone3D(skeleton->get_bone_name(bone_id), skeleton, temp_tip, p_pins)));
			root_segment->bone_map[bone_id] = next;
			if (ewbik_root_bone != bone_id) {
				temp_tip = next;
			}
		} else {
			break;
		}
	}
	tip = temp_tip;
	if (tip->is_pinned()) {
		enable_pinned_descendants();
	}
	set_name(vformat("IKBoneSegment%sRoot%sTip", root->get_name(), tip->get_name()));
	bones.clear();
	create_bone_list(bones, false);
}

void IKBoneSegment::create_bone_list(Vector<Ref<IKBone3D>> &p_list, bool p_recursive, bool p_debug_skeleton) const {
	if (p_recursive) {
		for (int32_t child_i = 0; child_i < child_segments.size(); child_i++) {
			child_segments[child_i]->create_bone_list(p_list, p_recursive, p_debug_skeleton);
		}
	}
	Ref<IKBone3D> current_bone = tip;
	Vector<Ref<IKBone3D>> list;
	while (current_bone.is_valid()) {
		list.push_back(current_bone);
		if (current_bone == root) {
			break;
		}
		current_bone = current_bone->get_parent();
	}
	if (p_debug_skeleton) {
		for (int32_t name_i = 0; name_i < list.size(); name_i++) {
			BoneId bone = list[name_i]->get_bone_id();

			String bone_name = skeleton->get_bone_name(bone);
			String effector;
			if (list[name_i]->is_pinned()) {
				effector += "Effector ";
			}
			String prefix;
			if (list[name_i] == root) {
				prefix += "(" + effector + "Root) ";
			}
			if (list[name_i] == tip) {
				prefix += "(" + effector + "Tip) ";
			}
			print_line(vformat("%s%s (%s)", prefix, bone_name, itos(bone)));
		}
	}
	p_list.append_array(list);
}

void IKBoneSegment::update_pinned_list(Vector<Vector<real_t>> &r_weights) {
	real_t depth_falloff = is_pinned() ? tip->get_pin()->depth_falloff : 1.0;
	for (int32_t chain_i = 0; chain_i < child_segments.size(); chain_i++) {
		Ref<IKBoneSegment> chain = child_segments[chain_i];
		chain->update_pinned_list(r_weights);
	}
	if (is_pinned()) {
		effector_list.push_back(tip->get_pin());
	}
	if (depth_falloff > 0.0) {
		for (Ref<IKBoneSegment> child : child_segments) {
			effector_list.append_array(child->effector_list);
		}
	}
}

void IKBoneSegment::update_optimal_rotation(Ref<IKBone3D> p_for_bone, real_t p_damp, bool p_translate) {
	ERR_FAIL_NULL(p_for_bone);
	update_target_headings(p_for_bone, &heading_weights, &target_headings);
	update_tip_headings(p_for_bone, &tip_headings);
	set_optimal_rotation(p_for_bone, &tip_headings, &target_headings, &heading_weights, p_damp, p_translate);
}

Quaternion IKBoneSegment::set_quadrance_angle(Quaternion p_quat, real_t p_cos_half_angle) const {
	double squared_sine = p_quat.get_axis().length_squared();
	Quaternion rot = p_quat;
	if (!Math::is_zero_approx(squared_sine)) {
		double inverse_coeff = Math::sqrt(((1.0f - (p_cos_half_angle * p_cos_half_angle)) / squared_sine));
		rot.x = inverse_coeff * p_quat.x;
		rot.y = inverse_coeff * p_quat.y;
		rot.z = inverse_coeff * p_quat.z;
		rot.w = p_quat.w < 0 ? -p_cos_half_angle : p_cos_half_angle;
	}
	return rot;
}

Quaternion IKBoneSegment::clamp_to_angle(Quaternion p_quat, real_t p_angle) {
	real_t x = cos(p_angle);
	real_t cos_half_angle = x;
	return clamp_to_quadrance_angle(p_quat, cos_half_angle);
}

Quaternion IKBoneSegment::clamp_to_quadrance_angle(Quaternion p_quat, real_t p_cos_half_angle) {
	double newCoeff = 1.0f - (p_cos_half_angle * p_cos_half_angle);
	Quaternion rot = p_quat;
	double currentCoeff = rot.x * rot.x + rot.y * rot.y + rot.z * rot.z;
	if (newCoeff > currentCoeff) {
		return rot;
	} else {
		rot.w = rot.w < 0.0f ? -p_cos_half_angle : p_cos_half_angle;
		double compositeCoeff = Math::sqrt(newCoeff / currentCoeff);
		rot.x *= compositeCoeff;
		rot.y *= compositeCoeff;
		rot.z *= compositeCoeff;
	}
	return rot;
}

float IKBoneSegment::get_manual_msd(const PackedVector3Array &r_htip, const PackedVector3Array &r_htarget, const Vector<real_t> &p_weights) {
	float manual_RMSD = 0.0f;
	float w_sum = 0.0f;
	for (int i = 0; i < r_htarget.size(); i++) {
		float x_d = r_htarget[i].x - r_htip[i].x;
		float y_d = r_htarget[i].y - r_htip[i].y;
		float z_d = r_htarget[i].z - r_htip[i].z;
		float mag_sq = p_weights[i] * (x_d * x_d + y_d * y_d + z_d * z_d);
		manual_RMSD += mag_sq;
		w_sum += p_weights[i];
	}
	manual_RMSD /= w_sum;
	return manual_RMSD;
}

void IKBoneSegment::set_optimal_rotation(Ref<IKBone3D> p_for_bone, PackedVector3Array *r_htip, PackedVector3Array *r_htarget, Vector<real_t> *r_weights, float p_dampening, bool p_translate) {
	ERR_FAIL_NULL(p_for_bone);
	ERR_FAIL_NULL(r_htip);
	ERR_FAIL_NULL(r_htarget);
	ERR_FAIL_NULL(r_weights);
	double bone_damp = p_for_bone->get_cos_half_dampen();
	{
		// Solved ik transform and apply it.
		QCP qcp = QCP(DBL_EPSILON, DBL_EPSILON);
		Quaternion rot = qcp.weighted_superpose(*r_htip, *r_htarget, *r_weights, p_translate);
		Vector3 translation = qcp.get_translation();
		if (p_dampening != -1.0f) {
			bone_damp = p_dampening;
			rot = clamp_to_angle(rot, bone_damp);
		} else {
			rot = clamp_to_quadrance_angle(rot, bone_damp);
		}
		p_for_bone->get_ik_transform()->rotate_local_with_global(rot);
		Transform3D result = Transform3D(p_for_bone->get_global_pose().basis, p_for_bone->get_global_pose().origin + translation);
		p_for_bone->set_global_pose(result);
	}

	// If the solved transform is outside the hard constraints, move it back into range.
	if (p_for_bone->get_constraint().is_valid() && p_for_bone->get_constraint_transform().is_valid()) {
		if (!p_for_bone->get_constraint()->get_limit_cones().is_empty()) {
			TypedArray<IKLimitCone> limit_cones = p_for_bone->get_constraint()->get_limit_cones();
			if (limit_cones.size()) {
				Ref<IKLimitCone> cone = limit_cones[0];
				if (cone.is_valid()) {
					Vector3 control_point = p_for_bone->get_constraint_transform()->to_global(cone->get_control_point());
					control_point -= p_for_bone->get_constraint_transform()->get_global_transform().origin;
				}
			}
		}
		if (p_for_bone->get_constraint()->is_axially_constrained()) {
			p_for_bone->get_constraint()->_update_constraint();
			p_for_bone->get_constraint()->set_snap_to_twist_limit(p_for_bone->get_ik_transform(), p_for_bone->get_constraint_transform(), bone_damp, p_for_bone->get_cos_half_dampen());
		}
		if (p_for_bone->get_constraint()->is_orientationally_constrained()) {
			p_for_bone->get_constraint()->set_axes_to_orientation_snap(p_for_bone->get_ik_transform(), p_for_bone->get_constraint_transform(), bone_damp, p_for_bone->get_cos_half_dampen());
		}
	}
}

void IKBoneSegment::update_target_headings(Ref<IKBone3D> p_for_bone, Vector<real_t> *r_weights, PackedVector3Array *r_target_headings) {
	ERR_FAIL_NULL(p_for_bone);
	ERR_FAIL_NULL(r_weights);
	ERR_FAIL_NULL(r_target_headings);
	int32_t last_index = 0;
	for (int32_t effector_i = 0; effector_i < effector_list.size(); effector_i++) {
		Ref<IKEffector3D> effector = effector_list[effector_i];
		last_index = effector->update_effector_target_headings(r_target_headings, last_index, p_for_bone, &heading_weights);
	}
}

void IKBoneSegment::update_tip_headings(Ref<IKBone3D> p_for_bone, PackedVector3Array *r_heading_tip) {
	ERR_FAIL_NULL(r_heading_tip);
	ERR_FAIL_NULL(p_for_bone);
	int32_t last_index = 0;
	for (int32_t effector_i = 0; effector_i < effector_list.size(); effector_i++) {
		Ref<IKEffector3D> effector = effector_list[effector_i];
		last_index = effector->update_effector_tip_headings(r_heading_tip, last_index, p_for_bone);
	}
}

void IKBoneSegment::segment_solver(real_t p_damp) {
	for (Ref<IKBoneSegment> child : child_segments) {
		if (child.is_null()) {
			continue;
		}
		child->segment_solver(p_damp);
	}
	bool is_translate = parent_segment.is_null();
	if (is_translate) {
		p_damp = Math_PI;
	}
	qcp_solver(p_damp, is_translate);
}

void IKBoneSegment::qcp_solver(real_t p_damp, bool p_translate) {
	for (Ref<IKBone3D> current_bone : bones) {
		update_optimal_rotation(current_bone, p_damp, p_translate && current_bone == root);
	}
}

void IKBoneSegment::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_pinned"), &IKBoneSegment::is_pinned);
	ClassDB::bind_method(D_METHOD("get_ik_bone", "bone"), &IKBoneSegment::get_ik_bone);
}

Ref<IKBoneSegment> IKBoneSegment::get_parent_segment() {
	return parent_segment;
}

IKBoneSegment::IKBoneSegment(Skeleton3D *p_skeleton, StringName p_root_bone_name, Vector<Ref<IKEffectorTemplate>> &p_pins, const Ref<IKBoneSegment> &p_parent,
		BoneId p_root, BoneId p_tip) {
	root = p_root;
	tip = p_tip;
	skeleton = p_skeleton;
	root = Ref<IKBone3D>(memnew(IKBone3D(p_root_bone_name, p_skeleton, p_parent, p_pins, Math_PI)));
	if (p_parent.is_valid()) {
		root_segment = p_parent->root_segment;
	} else {
		root_segment = Ref<IKBoneSegment>(this);
	}
	root_segment->bone_map[root->get_bone_id()] = root;
	if (p_parent.is_valid()) {
		parent_segment = p_parent;
		root->set_parent(p_parent->get_tip());
	}
}

void IKBoneSegment::enable_pinned_descendants() {
	pinned_descendants = true;
}

bool IKBoneSegment::has_pinned_descendants() {
	return pinned_descendants;
}

Vector<Ref<IKBone3D>> IKBoneSegment::get_bone_list() const {
	return bones;
}

Ref<IKBone3D> IKBoneSegment::get_ik_bone(BoneId p_bone) {
	if (!bone_map.has(p_bone)) {
		return Ref<IKBone3D>();
	}
	return bone_map[p_bone];
}

void IKBoneSegment::create_headings_arrays() {
	Vector<Vector<real_t>> penalty_array;
	Vector<Ref<IKBone3D>> new_pinned_bones;
	recursive_create_penalty_array(this, penalty_array, new_pinned_bones, 1.0);
	pinned_bones.resize(new_pinned_bones.size());
	int32_t total_headings = 0;
	for (const Vector<real_t> &current_penalty_array : penalty_array) {
		total_headings += current_penalty_array.size();
	}
	for (int32_t bone_i = 0; bone_i < new_pinned_bones.size(); bone_i++) {
		pinned_bones.write[bone_i] = new_pinned_bones[bone_i];
	}
	target_headings.resize(total_headings);
	tip_headings.resize(total_headings);
	heading_weights.resize(total_headings);
	int currentHeading = 0;
	for (const Vector<real_t> &current_penalty_array : penalty_array) {
		for (real_t ad : current_penalty_array) {
			heading_weights.write[currentHeading] = ad;
			target_headings.write[currentHeading] = Vector3();
			tip_headings.write[currentHeading] = Vector3();
			currentHeading++;
		}
	}
}

void IKBoneSegment::recursive_create_penalty_array(Ref<IKBoneSegment> p_bone_segment, Vector<Vector<real_t>> &r_penalty_array, Vector<Ref<IKBone3D>> &r_pinned_bones, real_t p_falloff) {
	if (p_falloff <= 0.0) {
		return;
	} else {
		real_t current_falloff = 1.0;
		if (p_bone_segment->is_pinned()) {
			Ref<IKBone3D> current_tip = p_bone_segment->get_tip();
			Ref<IKEffector3D> pin = current_tip->get_pin();
			real_t weight = pin->get_weight();
			Vector<real_t> inner_weight_array;
			inner_weight_array.push_back(weight * p_falloff);
			real_t max_pin_weight = 0.0;
			Vector3 priority = pin->get_direction_priorities();
			if (priority.x > 0.0) {
				max_pin_weight = MAX(max_pin_weight, priority.x);
			}
			if (priority.y > 0.0) {
				max_pin_weight = MAX(max_pin_weight, priority.y);
			}
			if (priority.z > 0.0) {
				max_pin_weight = MAX(max_pin_weight, priority.z);
			}
			if (max_pin_weight == 0.0) {
				max_pin_weight = 1.0;
			}
			max_pin_weight = 1.0;
			if (priority.x > 0.0) {
				double sub_target_weight = pin->get_weight() * (priority.x / max_pin_weight) * p_falloff;
				inner_weight_array.push_back(sub_target_weight);
				inner_weight_array.push_back(sub_target_weight);
			}
			if (priority.y > 0.0) {
				double sub_target_weight = pin->get_weight() * (priority.y / max_pin_weight) * p_falloff;
				inner_weight_array.push_back(sub_target_weight);
				inner_weight_array.push_back(sub_target_weight);
			}
			if (priority.z > 0.0) {
				double sub_target_weight = pin->get_weight() * (priority.z / max_pin_weight) * p_falloff;
				inner_weight_array.push_back(sub_target_weight);
				inner_weight_array.push_back(sub_target_weight);
			}
			r_penalty_array.push_back(inner_weight_array);
			r_pinned_bones.push_back(current_tip);
			current_falloff = pin->get_depth_falloff();
		}
		for (Ref<IKBoneSegment> s : p_bone_segment->get_child_segments()) {
			recursive_create_penalty_array(s, r_penalty_array, r_pinned_bones, p_falloff * current_falloff);
		}
	}
}

void IKBoneSegment::recursive_create_headings_arrays_for(Ref<IKBoneSegment> p_bone_segment) {
	p_bone_segment->create_headings_arrays();
	for (Ref<IKBoneSegment> segments : p_bone_segment->get_child_segments()) {
		recursive_create_headings_arrays_for(segments);
	}
}

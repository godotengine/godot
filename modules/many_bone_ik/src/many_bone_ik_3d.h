/**************************************************************************/
/*  many_bone_ik_3d.h                                                     */
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

#ifndef MANY_BONE_IK_3D_H
#define MANY_BONE_IK_3D_H

#include "core/math/math_defs.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "ik_bone_3d.h"
#include "ik_effector_template_3d.h"
#include "math/ik_node_3d.h"

#include "core/object/ref_counted.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/main/scene_tree.h"

class ManyBoneIK3DState;
class ManyBoneIK3D : public Node3D {
	GDCLASS(ManyBoneIK3D, Node3D);

	bool is_constraint_mode = false;
	NodePath skeleton_path;
	Vector<Ref<IKBoneSegment3D>> segmented_skeletons;
	int32_t constraint_count = 0, pin_count = 0, bone_count = 0;
	Vector<StringName> constraint_names;
	Vector<Ref<IKEffectorTemplate3D>> pins;
	Vector<Ref<IKBone3D>> bone_list;
	Vector<Vector2> kusudama_twist;
	Vector<float> bone_damp;
	Vector<Vector<Vector4>> kusudama_limit_cones;
	Vector<int> kusudama_limit_cone_count;
	float MAX_KUSUDAMA_LIMIT_CONES = 10;
	int32_t iterations_per_frame = 15;
	float default_damp = Math::deg_to_rad(5.0f);
	bool queue_debug_skeleton = false;
	Ref<IKNode3D> godot_skeleton_transform;
	Transform3D godot_skeleton_transform_inverse;
	Ref<IKNode3D> ik_origin;
	bool is_dirty = true;
	NodePath skeleton_node_path = NodePath("..");
	int32_t ui_selected_bone = -1, stabilize_passes = 0;

	void _on_timer_timeout();
	void _update_ik_bones_transform();
	void _update_skeleton_bones_transform();
	Vector<Ref<IKEffectorTemplate3D>> _get_bone_effectors() const;
	void _set_constraint_name(int32_t p_index, String p_name);
	void _set_pin_count(int32_t p_value);
	void _set_constraint_count(int32_t p_count);
	void _remove_pin(int32_t p_index);
	void _set_bone_count(int32_t p_count);
	void _set_pin_root_bone(int32_t p_pin_index, const String &p_root_bone);
	String _get_pin_root_bone(int32_t p_pin_index) const;
	bool _is_descendant_of(int bone_i, int parent_bone_i) const;
	bool _is_ancestor_of(int potential_ancestor, int bone_idx) const;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	static void _bind_methods();
	virtual void _skeleton_changed(Skeleton3D *skeleton);
	virtual void _execute(real_t delta);
	void _notification(int p_what);

public:
	void set_pin_target_static(int32_t p_effector_index, bool p_force_ignore);
	bool get_pin_target_static(int32_t p_effector_index);
	void set_pin_bone_name(int32_t p_effector_index, StringName p_name) const;
	void set_state(Ref<ManyBoneIK3DState> p_state);
	Ref<ManyBoneIK3DState> get_state() const;
	void add_constraint();
	void set_stabilization_passes(int32_t p_passes);
	int32_t get_stabilization_passes();
	Transform3D get_godot_skeleton_transform_inverse();
	Ref<IKNode3D> get_godot_skeleton_transform();
	void set_ui_selected_bone(int32_t p_ui_selected_bone);
	int32_t get_ui_selected_bone() const;
	void set_constraint_mode(bool p_enabled);
	bool get_constraint_mode() const;
	bool get_pin_enabled(int32_t p_effector_index) const;
	void set_skeleton_node_path(NodePath p_skeleton_node_path);
	void register_skeleton();
	void reset_constraints();
	NodePath get_skeleton_node_path();
	Skeleton3D *get_skeleton() const;
	Vector<Ref<IKBone3D>> get_bone_list() const;
	Vector<Ref<IKBoneSegment3D>> get_segmented_skeletons();
	float get_iterations_per_frame() const;
	void set_iterations_per_frame(const float &p_iterations_per_frame);
	void queue_print_skeleton();
	int32_t get_pin_count() const;
	void remove_constraint(int32_t p_index);
	void set_pin_bone(int32_t p_pin_index, const String &p_bone);
	StringName get_pin_bone_name(int32_t p_effector_index) const;
	void set_pin_nodepath(int32_t p_effector_index, NodePath p_node_path);
	NodePath get_pin_nodepath(int32_t p_effector_index) const;
	int32_t find_effector_id(StringName p_bone_name);
	void set_pin_target_nodepath(int32_t p_effector_index, const NodePath &p_target_node);
	void set_pin_weight(int32_t p_pin_index, const real_t &p_weight);
	real_t get_pin_weight(int32_t p_pin_index) const;
	void set_pin_direction_priorities(int32_t p_pin_index, const Vector3 &p_priority_direction);
	Vector3 get_pin_direction_priorities(int32_t p_pin_index) const;
	NodePath get_pin_target_nodepath(int32_t p_pin_index);
	void set_pin_passthrough_factor(int32_t p_effector_index, const float p_passthrough_factor);
	float get_pin_passthrough_factor(int32_t p_effector_index) const;
	real_t get_default_damp() const;
	void set_default_damp(float p_default_damp);
	int32_t find_constraint(String p_string) const;
	int32_t find_pin(String p_string) const;
	int32_t get_constraint_count() const;
	StringName get_constraint_name(int32_t p_index) const;
	void set_constraint_twist_transform(int32_t p_index, Transform3D p_transform);
	Transform3D get_constraint_twist_transform(int32_t p_index) const;
	void set_constraint_orientation_transform(int32_t p_index, Transform3D p_transform);
	Transform3D get_constraint_orientation_transform(int32_t p_index) const;
	void set_bone_direction_transform(int32_t p_index, Transform3D p_transform);
	Transform3D get_bone_direction_transform(int32_t p_index) const;
	Vector2 get_kusudama_twist(int32_t p_index) const;
	void set_kusudama_limit_cone(int32_t p_bone, int32_t p_index,
			Vector3 p_center, float p_radius);
	Vector3 get_kusudama_limit_cone_center(int32_t p_constraint_index, int32_t p_index) const;
	float get_kusudama_limit_cone_radius(int32_t p_constraint_index, int32_t p_index) const;
	int32_t get_kusudama_limit_cone_count(int32_t p_constraint_index) const;
	int32_t get_bone_count() const;
	void set_kusudama_twist_from_range(int32_t p_index, float from, float range);
	void set_kusudama_twist(int32_t p_index, Vector2 p_limit);
	void set_kusudama_limit_cone_count(int32_t p_constraint_index, int32_t p_count);
	void set_kusudama_limit_cone_center(int32_t p_constraint_index, int32_t p_index, Vector3 p_center);
	void set_kusudama_limit_cone_radius(int32_t p_constraint_index, int32_t p_index, float p_radius);
	ManyBoneIK3D();
	~ManyBoneIK3D();
	void set_dirty();
};

#endif // MANY_BONE_IK_3D_H

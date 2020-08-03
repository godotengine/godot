/*************************************************************************/
/*  skeleton_modification_2d.h                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SKELETONMODIFICATION2D_H
#define SKELETONMODIFICATION2D_H

#include "scene/2d/skeleton_2d.h"

///////////////////////////////////////
// SkeletonModificationStack2D
///////////////////////////////////////

class Skeleton2D;
class SkeletonModification2D;

class SkeletonModificationStack2D : public Resource {
	GDCLASS(SkeletonModificationStack2D, Resource);
	friend class Skeleton2D;
	friend class SkeletonModification2D;

protected:
	static void _bind_methods();
	void _get_property_list(List<PropertyInfo> *p_list) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	bool _get(const StringName &p_path, Variant &r_ret) const;

public:
	Skeleton2D *skeleton;
	bool is_setup = false;
	bool enabled = true;
	float strength = 0.0;

	enum EXECUTION_MODE {
		execution_mode_process,
		execution_mode_physics_process
	};
	int execution_mode = execution_mode_process;

	void set_execution_mode(int p_mode);
	int get_execution_mode();

	Vector<Ref<SkeletonModification2D>> modifications;
	int modifications_count = 0;

	void setup();
	void execute(float delta);

	void enable_all_modifications(bool p_enable);
	Ref<SkeletonModification2D> get_modification(int p_mod_idx) const;
	void add_modification(Ref<SkeletonModification2D> p_mod);
	void delete_modification(int p_mod_idx);
	void set_modification(int p_mod_idx, Ref<SkeletonModification2D> p_mod);

	void set_modification_count(int p_count);
	int get_modification_count() const;

	void set_skeleton(Skeleton2D *p_skeleton);
	Skeleton2D *get_skeleton() const;

	bool get_is_setup() const;

	void set_enabled(bool p_enabled);
	bool get_enabled() const;

	void set_strength(float p_strength);
	float get_strength() const;

	SkeletonModificationStack2D();
};

///////////////////////////////////////
// SkeletonModification2D
///////////////////////////////////////

class SkeletonModification2D : public Resource {
	GDCLASS(SkeletonModification2D, Resource);
	friend class Skeleton2D;

protected:
	static void _bind_methods();

	SkeletonModificationStack2D *stack;

	bool enabled = false;
	bool is_setup = false;

public:
	virtual void execute(float delta);
	virtual void setup_modification(SkeletonModificationStack2D *p_stack);

	void set_enabled(bool p_enabled);
	bool get_enabled();

	float clamp_angle(float angle, float min_bound, float max_bound, bool invert_clamp = false);

	SkeletonModification2D();
};

///////////////////////////////////////
// SkeletonModification2DLookAt
///////////////////////////////////////

class SkeletonModification2DLookAt : public SkeletonModification2D {
	GDCLASS(SkeletonModification2DLookAt, SkeletonModification2D);

private:
	int bone_idx = -1;
	NodePath bone2d_node;
	ObjectID bone2d_node_cache;

	NodePath target_node;
	ObjectID target_node_cache;

	float additional_rotation = 0;
	bool enable_constraint = false;
	float constraint_angle_min = 0;
	float constraint_angle_max = (2.0 * Math_PI);
	bool constraint_angle_invert = false;
	bool constraint_in_localspace = true;

	void update_bone2d_cache();
	void update_target_cache();

protected:
	static void _bind_methods();
	bool _set(const StringName &p_path, const Variant &p_value);
	bool _get(const StringName &p_path, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void execute(float delta) override;
	void setup_modification(SkeletonModificationStack2D *p_stack) override;

	void set_bone2d_node(const NodePath &p_target_node);
	NodePath get_bone2d_node() const;
	void set_bone_index(int p_idx);
	int get_bone_index() const;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;

	void set_additional_rotation(float p_rotation);
	float get_additional_rotation() const;

	void set_enable_constraint(bool p_constraint);
	bool get_enable_constraint() const;
	void set_constraint_angle_min(float p_angle_min);
	float get_constraint_angle_min() const;
	void set_constraint_angle_max(float p_angle_max);
	float get_constraint_angle_max() const;
	void set_constraint_angle_invert(bool p_invert);
	bool get_constraint_angle_invert() const;
	void set_constraint_in_localspace(bool p_constraint_in_localspace);
	bool get_constraint_in_localspace() const;

	SkeletonModification2DLookAt();
	~SkeletonModification2DLookAt();
};

///////////////////////////////////////
// SkeletonModification2DCCDIK
///////////////////////////////////////

class SkeletonModification2DCCDIK : public SkeletonModification2D {
	GDCLASS(SkeletonModification2DCCDIK, SkeletonModification2D);

private:
	struct CCDIK_Joint_Data2D {
		int bone_idx = -1;
		NodePath bone2d_node;
		ObjectID bone2d_node_cache;
		bool rotate_from_joint = false;

		bool enable_constraint = false;
		float constraint_angle_min = 0;
		float constraint_angle_max = (2.0 * Math_PI);
		bool constraint_angle_invert = false;
		bool constraint_in_localspace = true;
	};

	Vector<CCDIK_Joint_Data2D> ccdik_data_chain;

	NodePath target_node;
	ObjectID target_node_cache;
	void update_target_cache();

	NodePath tip_node;
	ObjectID tip_node_cache;
	void update_tip_cache();

	void ccdik_joint_update_bone2d_cache(int p_joint_idx);
	void _execute_ccdik_joint(int p_joint_idx, Node2D *target, Node2D *tip);

protected:
	static void _bind_methods();
	bool _set(const StringName &p_path, const Variant &p_value);
	bool _get(const StringName &p_path, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void execute(float delta) override;
	void setup_modification(SkeletonModificationStack2D *p_stack) override;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;
	void set_tip_node(const NodePath &p_tip_node);
	NodePath get_tip_node() const;

	int get_ccdik_data_chain_length();
	void set_ccdik_data_chain_length(int p_new_length);

	void ccdik_joint_set_bone2d_node(int p_joint_idx, const NodePath &p_target_node);
	NodePath ccdik_joint_get_bone2d_node(int p_joint_idx) const;
	void ccdik_joint_set_bone_index(int p_joint_idx, int p_bone_idx);
	int ccdik_joint_get_bone_index(int p_joint_idx) const;

	void ccdik_joint_set_rotate_from_joint(int p_joint_idx, bool p_rotate_from_joint);
	bool ccdik_joint_get_rotate_from_joint(int p_joint_idx) const;
	void ccdik_joint_set_enable_constraint(int p_joint_idx, bool p_constraint);
	bool ccdik_joint_get_enable_constraint(int p_joint_idx) const;
	void ccdik_joint_set_constraint_angle_min(int p_joint_idx, float p_angle_min);
	float ccdik_joint_get_constraint_angle_min(int p_joint_idx) const;
	void ccdik_joint_set_constraint_angle_max(int p_joint_idx, float p_angle_max);
	float ccdik_joint_get_constraint_angle_max(int p_joint_idx) const;
	void ccdik_joint_set_constraint_angle_invert(int p_joint_idx, bool p_invert);
	bool ccdik_joint_get_constraint_angle_invert(int p_joint_idx) const;
	void ccdik_joint_set_constraint_in_localspace(int p_joint_idx, bool p_constraint_in_localspace);
	bool ccdik_joint_get_constraint_in_localspace(int p_joint_idx) const;

	SkeletonModification2DCCDIK();
	~SkeletonModification2DCCDIK();
};

#endif // SKELETONMODIFICATION2D_H

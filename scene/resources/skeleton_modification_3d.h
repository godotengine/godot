/*************************************************************************/
/*  skeleton_modification_3d.h                                           */
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

#ifndef SKELETONMODIFICATION3D_H
#define SKELETONMODIFICATION3D_H

#include "scene/3d/skeleton_3d.h"

///////////////////////////////////////
// SkeletonModificationStack3D
///////////////////////////////////////

class Skeleton3D;
class SkeletonModification3D;

class SkeletonModificationStack3D : public Resource {
	GDCLASS(SkeletonModificationStack3D, Resource);
	friend class Skeleton3D;
	friend class SkeletonModification3D;

protected:
	static void _bind_methods();
	void _get_property_list(List<PropertyInfo> *p_list) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	bool _get(const StringName &p_path, Variant &r_ret) const;

public:
	Skeleton3D *skeleton;
	bool is_setup = false;
	bool enabled = true;
	float strength = 1.0;

	enum EXECUTION_MODE {
		execution_mode_process,
		execution_mode_physics_process,
	};

	Vector<Ref<SkeletonModification3D>> modifications;
	int modifications_count = 0;

	void setup();
	void execute(float delta, int execution_mode);

	void enable_all_modifications(bool p_enable);
	Ref<SkeletonModification3D> get_modification(int p_mod_idx) const;
	void add_modification(Ref<SkeletonModification3D> p_mod);
	void delete_modification(int p_mod_idx);
	void set_modification(int p_mod_idx, Ref<SkeletonModification3D> p_mod);

	void set_modification_count(int p_count);
	int get_modification_count() const;

	void set_skeleton(Skeleton3D *p_skeleton);
	Skeleton3D *get_skeleton() const;

	bool get_is_setup() const;

	void set_enabled(bool p_enabled);
	bool get_enabled() const;

	void set_strength(float p_strength);
	float get_strength() const;

	SkeletonModificationStack3D();
};

///////////////////////////////////////
// SkeletonModification3D
///////////////////////////////////////

class SkeletonModification3D : public Resource {
	GDCLASS(SkeletonModification3D, Resource);
	friend class Skeleton3D;

protected:
	static void _bind_methods();

	SkeletonModificationStack3D *stack;
	int execution_mode = SkeletonModificationStack3D::EXECUTION_MODE::execution_mode_process;

	bool enabled = true;
	bool is_setup = false;
	bool execution_error_found = false;

	bool _print_execution_error(bool p_condition, String p_message);

public:
	virtual void execute(float delta);
	virtual void setup_modification(SkeletonModificationStack3D *p_stack);

	float clamp_angle(float angle, float min_bound, float max_bound, bool invert);

	void set_enabled(bool p_enabled);
	bool get_enabled();

	void set_execution_mode(int p_mode);
	int get_execution_mode() const;

	SkeletonModificationStack3D *get_modification_stack();

	void set_is_setup(bool p_setup);
	bool get_is_setup() const;

	SkeletonModification3D();
};

///////////////////////////////////////
// SkeletonModification3DLookAt
///////////////////////////////////////

class SkeletonModification3DLookAt : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DLookAt, SkeletonModification3D);

private:
	String bone_name = "";
	int bone_idx = -1;
	NodePath target_node;
	ObjectID target_node_cache;

	Vector3 additional_rotation = Vector3(1, 0, 0);
	bool lock_rotation_to_plane = false;
	int lock_rotation_plane = ROTATION_PLANE_X;

	void update_cache();

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	enum ROTATION_PLANE {
		ROTATION_PLANE_X,
		ROTATION_PLANE_Y,
		ROTATION_PLANE_Z
	};

	void execute(float delta) override;
	void setup_modification(SkeletonModificationStack3D *p_stack) override;

	void set_bone_name(String p_name);
	String get_bone_name() const;

	void set_bone_index(int p_idx);
	int get_bone_index() const;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;

	void set_additional_rotation(Vector3 p_offset);
	Vector3 get_additional_rotation() const;

	void set_lock_rotation_to_plane(bool p_lock_to_plane);
	bool get_lock_rotation_to_plane() const;
	void set_lock_rotation_plane(int p_plane);
	int get_lock_rotation_plane() const;

	SkeletonModification3DLookAt();
	~SkeletonModification3DLookAt();
};

///////////////////////////////////////
// SkeletonModification3DCCDIK
///////////////////////////////////////

class SkeletonModification3DCCDIK : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DCCDIK, SkeletonModification3D);

private:
	enum CCDIK_Axes {
		AXIS_X,
		AXIS_Y,
		AXIS_Z
	};

	struct CCDIK_Joint_Data {
		String bone_name = "";
		int bone_idx = -1;
		int ccdik_axis = 0;

		bool enable_constraint = false;
		float constraint_angle_min = 0;
		float constraint_angle_max = (2.0 * Math_PI);
		bool constraint_angles_invert = false;
	};

	Vector<CCDIK_Joint_Data> ccdik_data_chain;
	NodePath target_node;
	ObjectID target_node_cache;

	NodePath tip_node;
	ObjectID tip_node_cache;

	bool use_high_quality_solve = true;

	void update_target_cache();
	void update_tip_cache();

	void _execute_ccdik_joint(int p_joint_idx, Node3D *target, Node3D *tip);

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void execute(float delta) override;
	void setup_modification(SkeletonModificationStack3D *p_stack) override;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;

	void set_tip_node(const NodePath &p_tip_node);
	NodePath get_tip_node() const;

	void set_use_high_quality_solve(bool p_solve);
	bool get_use_high_quality_solve() const;

	String ccdik_joint_get_bone_name(int p_joint_idx) const;
	void ccdik_joint_set_bone_name(int p_joint_idx, String p_bone_name);
	int ccdik_joint_get_bone_index(int p_joint_idx) const;
	void ccdik_joint_set_bone_index(int p_joint_idx, int p_bone_idx);
	int ccdik_joint_get_ccdik_axis(int p_joint_idx) const;
	void ccdik_joint_set_ccdik_axis(int p_joint_idx, int p_axis);
	bool ccdik_joint_get_enable_constraint(int p_joint_idx) const;
	void ccdik_joint_set_enable_constraint(int p_joint_idx, bool p_enable);
	float ccdik_joint_get_constraint_angle_min(int p_joint_idx) const;
	void ccdik_joint_set_constraint_angle_min(int p_joint_idx, float p_angle_min);
	void ccdik_joint_set_constraint_angle_degrees_min(int p_joint_idx, float p_angle_min);
	float ccdik_joint_get_constraint_angle_max(int p_joint_idx) const;
	void ccdik_joint_set_constraint_angle_max(int p_joint_idx, float p_angle_max);
	void ccdik_joint_set_constraint_angle_degrees_max(int p_joint_idx, float p_angle_max);
	bool ccdik_joint_get_constraint_invert(int p_joint_idx) const;
	void ccdik_joint_set_constraint_invert(int p_joint_idx, bool p_invert);

	int get_ccdik_data_chain_length();
	void set_ccdik_data_chain_length(int p_new_length);

	SkeletonModification3DCCDIK();
	~SkeletonModification3DCCDIK();
};

///////////////////////////////////////
// SkeletonModification3DFABRIK
///////////////////////////////////////

class SkeletonModification3DFABRIK : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DFABRIK, SkeletonModification3D);

private:
	struct FABRIK_Joint_Data {
		String bone_name = "";
		int bone_idx = -1;
		float length = -1;
		Vector3 magnet_position = Vector3(0, 0, 0);

		bool auto_calculate_length = true;
		bool use_tip_node = false;
		NodePath tip_node = NodePath();
		ObjectID tip_node_cache;

		bool use_target_basis = false;
		float roll = 0;
	};

	Vector<FABRIK_Joint_Data> fabrik_data_chain;
	NodePath target_node;
	ObjectID target_node_cache;

	float chain_tolerance = 0.01;
	int chain_max_iterations = 10;
	int chain_iterations = 0;

	void update_target_cache();
	void update_joint_tip_cache(int p_joint_idx);

	int final_joint_idx = 0;
	Transform target_global_pose = Transform();
	Transform origin_global_pose = Transform();

	void chain_backwards();
	void chain_apply();

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void execute(float delta) override;
	void setup_modification(SkeletonModificationStack3D *p_stack) override;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;

	int get_fabrik_data_chain_length();
	void set_fabrik_data_chain_length(int p_new_length);

	float get_chain_tolerance();
	void set_chain_tolerance(float p_tolerance);

	int get_chain_max_iterations();
	void set_chain_max_iterations(int p_iterations);

	String fabrik_joint_get_bone_name(int p_joint_idx) const;
	void fabrik_joint_set_bone_name(int p_joint_idx, String p_bone_name);
	int fabrik_joint_get_bone_index(int p_joint_idx) const;
	void fabrik_joint_set_bone_index(int p_joint_idx, int p_bone_idx);
	float fabrik_joint_get_length(int p_joint_idx) const;
	void fabrik_joint_set_length(int p_joint_idx, float p_bone_length);
	Vector3 fabrik_joint_get_magnet(int p_joint_idx) const;
	void fabrik_joint_set_magnet(int p_joint_idx, Vector3 p_magnet);
	bool fabrik_joint_get_auto_calculate_length(int p_joint_idx) const;
	void fabrik_joint_set_auto_calculate_length(int p_joint_idx, bool p_auto_calculate);
	void fabrik_joint_auto_calculate_length(int p_joint_idx);
	bool fabrik_joint_get_use_tip_node(int p_joint_idx) const;
	void fabrik_joint_set_use_tip_node(int p_joint_idx, bool p_use_tip_node);
	NodePath fabrik_joint_get_tip_node(int p_joint_idx) const;
	void fabrik_joint_set_tip_node(int p_joint_idx, NodePath p_tip_node);
	bool fabrik_joint_get_use_target_basis(int p_joint_idx) const;
	void fabrik_joint_set_use_target_basis(int p_joint_idx, bool p_use_basis);
	float fabrik_joint_get_roll(int p_joint_idx) const;
	void fabrik_joint_set_roll(int p_joint_idx, float p_roll);

	SkeletonModification3DFABRIK();
	~SkeletonModification3DFABRIK();
};

///////////////////////////////////////
// SkeletonModification3DJiggle
///////////////////////////////////////

class SkeletonModification3DJiggle : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DJiggle, SkeletonModification3D);

private:
	struct Jiggle_Joint_Data {
		String bone_name = "";
		int bone_idx = -1;

		bool override_defaults = false;
		float stiffness = 3;
		float mass = 0.75;
		float damping = 0.75;
		bool use_gravity = false;
		Vector3 gravity = Vector3(0, -6.0, 0);
		float roll = 0;

		Vector3 cached_rotation = Vector3(0, 0, 0);
		Vector3 force = Vector3(0, 0, 0);
		Vector3 acceleration = Vector3(0, 0, 0);
		Vector3 velocity = Vector3(0, 0, 0);
		Vector3 last_position = Vector3(0, 0, 0);
		Vector3 dynamic_position = Vector3(0, 0, 0);

		Vector3 last_noncollision_position = Vector3(0, 0, 0);
	};

	NodePath target_node;
	ObjectID target_node_cache;
	Vector<Jiggle_Joint_Data> jiggle_data_chain;

	float stiffness = 3;
	float mass = 0.75;
	float damping = 0.75;
	bool use_gravity = false;
	Vector3 gravity = Vector3(0, -6.0, 0);

	bool use_colliders = false;
	uint32_t collision_mask = 1;

	void update_cache();
	void _execute_jiggle_joint(int p_joint_idx, Node3D *target, float delta);
	void _update_jiggle_joint_data();

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void execute(float delta) override;
	void setup_modification(SkeletonModificationStack3D *p_stack) override;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;

	void set_stiffness(float p_stiffness);
	float get_stiffness() const;
	void set_mass(float p_mass);
	float get_mass() const;
	void set_damping(float p_damping);
	float get_damping() const;

	void set_use_gravity(bool p_use_gravity);
	bool get_use_gravity() const;
	void set_gravity(Vector3 p_gravity);
	Vector3 get_gravity() const;

	void set_use_colliders(bool p_use_colliders);
	bool get_use_colliders() const;
	void set_collision_mask(int p_mask);
	int get_collision_mask() const;

	int get_jiggle_data_chain_length();
	void set_jiggle_data_chain_length(int p_new_length);

	void jiggle_joint_set_bone_name(int joint_idx, String p_name);
	String jiggle_joint_get_bone_name(int joint_idx) const;
	void jiggle_joint_set_bone_index(int joint_idx, int p_idx);
	int jiggle_joint_get_bone_index(int joint_idx) const;

	void jiggle_joint_set_override(int joint_idx, bool p_override);
	bool jiggle_joint_get_override(int joint_idx) const;
	void jiggle_joint_set_stiffness(int joint_idx, float p_stiffness);
	float jiggle_joint_get_stiffness(int joint_idx) const;
	void jiggle_joint_set_mass(int joint_idx, float p_mass);
	float jiggle_joint_get_mass(int joint_idx) const;
	void jiggle_joint_set_damping(int joint_idx, float p_damping);
	float jiggle_joint_get_damping(int joint_idx) const;
	void jiggle_joint_set_use_gravity(int joint_idx, bool p_use_gravity);
	bool jiggle_joint_get_use_gravity(int joint_idx) const;
	void jiggle_joint_set_gravity(int joint_idx, Vector3 p_gravity);
	Vector3 jiggle_joint_get_gravity(int joint_idx) const;
	void jiggle_joint_set_roll(int joint_idx, float p_roll);
	float jiggle_joint_get_roll(int joint_idx) const;

	SkeletonModification3DJiggle();
	~SkeletonModification3DJiggle();
};

///////////////////////////////////////
// SkeletonModification3DTwoBoneIK
///////////////////////////////////////

class SkeletonModification3DTwoBoneIK : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DTwoBoneIK, SkeletonModification3D);

private:
	NodePath target_node;
	ObjectID target_node_cache;

	bool use_tip_node = false;
	NodePath tip_node;
	ObjectID tip_node_cache;

	bool use_pole_node = false;
	NodePath pole_node;
	ObjectID pole_node_cache;

	String joint_one_bone_name = "";
	int joint_one_bone_idx = -1;
	String joint_two_bone_name = "";
	int joint_two_bone_idx = -1;

	bool auto_calculate_joint_length = false;
	float joint_one_length = -1;
	float joint_two_length = -1;

	float joint_one_roll = 0;
	float joint_two_roll = 0;

	void update_cache_target();
	void update_cache_tip();
	void update_cache_pole();

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void execute(float delta) override;
	void setup_modification(SkeletonModificationStack3D *p_stack) override;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;

	void set_use_tip_node(const bool p_use_tip_node);
	bool get_use_tip_node() const;
	void set_tip_node(const NodePath &p_tip_node);
	NodePath get_tip_node() const;

	void set_use_pole_node(const bool p_use_pole_node);
	bool get_use_pole_node() const;
	void set_pole_node(const NodePath &p_pole_node);
	NodePath get_pole_node() const;

	void set_auto_calculate_joint_length(bool p_calculate);
	bool get_auto_calculate_joint_length() const;
	void calculate_joint_lengths();

	void set_joint_one_bone_name(String p_bone_name);
	String get_joint_one_bone_name() const;
	void set_joint_one_bone_idx(int p_bone_idx);
	int get_joint_one_bone_idx() const;
	void set_joint_one_length(float p_length);
	float get_joint_one_length() const;

	void set_joint_two_bone_name(String p_bone_name);
	String get_joint_two_bone_name() const;
	void set_joint_two_bone_idx(int p_bone_idx);
	int get_joint_two_bone_idx() const;
	void set_joint_two_length(float p_length);
	float get_joint_two_length() const;

	void set_joint_one_roll(float p_roll);
	float get_joint_one_roll() const;
	void set_joint_two_roll(float p_roll);
	float get_joint_two_roll() const;

	SkeletonModification3DTwoBoneIK();
	~SkeletonModification3DTwoBoneIK();
};

///////////////////////////////////////
// SkeletonModification3DStackHolder
///////////////////////////////////////

class SkeletonModification3DStackHolder : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DStackHolder, SkeletonModification3D);

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	Ref<SkeletonModificationStack3D> held_modification_stack;

	void execute(float delta) override;
	void setup_modification(SkeletonModificationStack3D *p_stack) override;

	void set_held_modification_stack(Ref<SkeletonModificationStack3D> p_held_stack);
	Ref<SkeletonModificationStack3D> get_held_modification_stack() const;

	SkeletonModification3DStackHolder();
	~SkeletonModification3DStackHolder();
};

#endif // SKELETONMODIFICATION3D_H

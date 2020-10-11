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
class Bone2D;

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

	Vector<Ref<SkeletonModification2D>> modifications;
	int modifications_count = 0;

	void setup();
	void execute(float delta, int p_execution_mode);

	bool editor_gizmo_dirty = false;
	void draw_editor_gizmos();
	void set_editor_gizmos_dirty(bool p_dirty);

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
	friend class Bone2D;

protected:
	static void _bind_methods();

	SkeletonModificationStack2D *stack;
	int execution_mode = SkeletonModificationStack2D::EXECUTION_MODE::execution_mode_process;

	bool enabled = true;
	bool is_setup = false;
	bool execution_error_found = false;

	bool _print_execution_error(bool p_condition, String p_message);

public:
	virtual void execute(float delta);
	virtual void setup_modification(SkeletonModificationStack2D *p_stack);
	virtual void draw_editor_gizmo();

	bool editor_draw_gizmo = false;
	void set_editor_draw_gizmo(bool p_draw_gizmo);
	bool get_editor_draw_gizmo() const;

	void set_enabled(bool p_enabled);
	bool get_enabled();

	Ref<SkeletonModificationStack2D> get_modification_stack();
	void set_is_setup(bool p_setup);
	bool get_is_setup() const;

	void set_execution_mode(int p_mode);
	int get_execution_mode() const;

	float clamp_angle(float angle, float min_bound, float max_bound, bool invert_clamp = false);
	void editor_draw_angle_constraints(Bone2D *operation_bone, float min_bound, float max_bound, bool constraint_enabled, bool constraint_in_localspace, bool constraint_inverted);

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
	void draw_editor_gizmo() override;

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

		bool editor_draw_gizmo = true;
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
	void draw_editor_gizmo() override;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;
	void set_tip_node(const NodePath &p_tip_node);
	NodePath get_tip_node() const;

	int get_ccdik_data_chain_length();
	void set_ccdik_data_chain_length(int p_new_length);

	void set_ccdik_joint_bone2d_node(int p_joint_idx, const NodePath &p_target_node);
	NodePath get_ccdik_joint_bone2d_node(int p_joint_idx) const;
	void set_ccdik_joint_bone_index(int p_joint_idx, int p_bone_idx);
	int get_ccdik_joint_bone_index(int p_joint_idx) const;

	void set_ccdik_joint_rotate_from_joint(int p_joint_idx, bool p_rotate_from_joint);
	bool get_ccdik_joint_rotate_from_joint(int p_joint_idx) const;
	void set_ccdik_joint_enable_constraint(int p_joint_idx, bool p_constraint);
	bool get_ccdik_joint_enable_constraint(int p_joint_idx) const;
	void set_ccdik_joint_constraint_angle_min(int p_joint_idx, float p_angle_min);
	float get_ccdik_joint_constraint_angle_min(int p_joint_idx) const;
	void set_ccdik_joint_constraint_angle_max(int p_joint_idx, float p_angle_max);
	float get_ccdik_joint_constraint_angle_max(int p_joint_idx) const;
	void set_ccdik_joint_constraint_angle_invert(int p_joint_idx, bool p_invert);
	bool get_ccdik_joint_constraint_angle_invert(int p_joint_idx) const;
	void set_ccdik_joint_constraint_in_localspace(int p_joint_idx, bool p_constraint_in_localspace);
	bool get_ccdik_joint_constraint_in_localspace(int p_joint_idx) const;
	void set_ccdik_joint_editor_draw_gizmo(int p_joint_idx, bool p_draw_gizmo);
	bool get_ccdik_joint_editor_draw_gizmo(int p_joint_idx) const;

	SkeletonModification2DCCDIK();
	~SkeletonModification2DCCDIK();
};

///////////////////////////////////////
// SkeletonModification2DFABRIK
///////////////////////////////////////

class SkeletonModification2DFABRIK : public SkeletonModification2D {
	GDCLASS(SkeletonModification2DFABRIK, SkeletonModification2D);

private:
	struct FABRIK_Joint_Data2D {
		int bone_idx = -1;
		NodePath bone2d_node;
		ObjectID bone2d_node_cache;

		Vector2 magnet_position = Vector2(0, 0);
		bool use_target_rotation = false;

		bool enable_constraint = false;
		float constraint_angle_min = 0;
		float constraint_angle_max = (2.0 * Math_PI);
		bool constraint_angle_invert = false;
		bool constraint_in_localspace = true;

		bool editor_draw_gizmo = true;
	};

	Vector<FABRIK_Joint_Data2D> fabrik_data_chain;

	// Unlike in 3D, we need a vector of Transform2D objects to perform FABRIK.
	// This is because FABRIK (unlike CCDIK) needs to operate on transforms that are NOT
	// affected by each other, making the transforms stored in Bone2D unusable, as well as those in Skeleton2D.
	// For this reason, this modification stores a vector of Transform2Ds used for the calculations, which are then applied at the end.
	Vector<Transform2D> fabrik_transform_chain;

	NodePath target_node;
	ObjectID target_node_cache;
	void update_target_cache();

	float chain_tolarance = 0.01;
	int chain_max_iterations = 10;
	int chain_iterations = 0;
	Transform2D target_global_pose = Transform2D();
	Transform2D origin_global_pose = Transform2D();

	void fabrik_joint_update_bone2d_cache(int p_joint_idx);
	void chain_backwards();
	void chain_forwards();

protected:
	static void _bind_methods();
	bool _set(const StringName &p_path, const Variant &p_value);
	bool _get(const StringName &p_path, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void execute(float delta) override;
	void setup_modification(SkeletonModificationStack2D *p_stack) override;
	void draw_editor_gizmo() override;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;

	int get_fabrik_data_chain_length();
	void set_fabrik_data_chain_length(int p_new_length);

	void set_fabrik_joint_bone2d_node(int p_joint_idx, const NodePath &p_target_node);
	NodePath get_fabrik_joint_bone2d_node(int p_joint_idx) const;
	void set_fabrik_joint_bone_index(int p_joint_idx, int p_bone_idx);
	int get_fabrik_joint_bone_index(int p_joint_idx) const;

	void set_fabrik_joint_magnet_position(int p_joint_idx, Vector2 p_magnet_position);
	Vector2 get_fabrik_joint_magnet_position(int p_joint_idx) const;
	void set_fabrik_joint_use_target_rotation(int p_joint_idx, bool p_use_target_rotation);
	bool get_fabrik_joint_use_target_rotation(int p_joint_idx) const;
	void set_fabrik_joint_enable_constraint(int p_joint_idx, bool p_constraint);
	bool get_fabrik_joint_enable_constraint(int p_joint_idx) const;
	void set_fabrik_joint_constraint_angle_min(int p_joint_idx, float p_angle_min);
	float get_fabrik_joint_constraint_angle_min(int p_joint_idx) const;
	void set_fabrik_joint_constraint_angle_max(int p_joint_idx, float p_angle_max);
	float get_fabrik_joint_constraint_angle_max(int p_joint_idx) const;
	void set_fabrik_joint_constraint_angle_invert(int p_joint_idx, bool p_invert);
	bool get_fabrik_joint_constraint_angle_invert(int p_joint_idx) const;
	void set_fabrik_joint_constraint_in_localspace(int p_joint_idx, bool p_constraint_in_localspace);
	bool get_fabrik_joint_constraint_in_localspace(int p_joint_idx) const;
	void set_fabrik_joint_editor_draw_gizmo(int p_joint_idx, bool p_draw_gizmo);
	bool get_fabrik_joint_editor_draw_gizmo(int p_joint_idx) const;

	SkeletonModification2DFABRIK();
	~SkeletonModification2DFABRIK();
};

///////////////////////////////////////
// SkeletonModification2DJiggle
///////////////////////////////////////

class SkeletonModification2DJiggle : public SkeletonModification2D {
	GDCLASS(SkeletonModification2DJiggle, SkeletonModification2D);

private:
	struct Jiggle_Joint_Data2D {
		int bone_idx = -1;
		NodePath bone2d_node;
		ObjectID bone2d_node_cache;

		bool override_defaults = false;
		float stiffness = 3;
		float mass = 0.75;
		float damping = 0.75;
		bool use_gravity = false;
		Vector2 gravity = Vector2(0, 6.0);

		Vector2 force = Vector2(0, 0);
		Vector2 acceleration = Vector2(0, 0);
		Vector2 velocity = Vector2(0, 0);
		Vector2 last_position = Vector2(0, 0);
		Vector2 dynamic_position = Vector2(0, 0);

		Vector2 last_noncollision_position = Vector2(0, 0);
	};

	Vector<Jiggle_Joint_Data2D> jiggle_data_chain;

	NodePath target_node;
	ObjectID target_node_cache;
	void update_target_cache();

	float stiffness = 3;
	float mass = 0.75;
	float damping = 0.75;
	bool use_gravity = false;
	Vector2 gravity = Vector2(0, 6);

	bool use_colliders = false;
	uint32_t collision_mask = 1;

	void jiggle_joint_update_bone2d_cache(int p_joint_idx);
	void _execute_jiggle_joint(int p_joint_idx, Node2D *target, float delta);
	void _update_jiggle_joint_data();

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

	void set_stiffness(float p_stiffness);
	float get_stiffness() const;
	void set_mass(float p_mass);
	float get_mass() const;
	void set_damping(float p_damping);
	float get_damping() const;
	void set_use_gravity(bool p_use_gravity);
	bool get_use_gravity() const;
	void set_gravity(Vector2 p_gravity);
	Vector2 get_gravity() const;

	void set_use_colliders(bool p_use_colliders);
	bool get_use_colliders() const;
	void set_collision_mask(int p_mask);
	int get_collision_mask() const;

	int get_jiggle_data_chain_length();
	void set_jiggle_data_chain_length(int p_new_length);

	void set_jiggle_joint_bone2d_node(int p_joint_idx, const NodePath &p_target_node);
	NodePath get_jiggle_joint_bone2d_node(int p_joint_idx) const;
	void set_jiggle_joint_bone_index(int p_joint_idx, int p_bone_idx);
	int get_jiggle_joint_bone_index(int p_joint_idx) const;

	void set_jiggle_joint_override(int p_joint_idx, bool p_override);
	bool get_jiggle_joint_override(int p_joint_idx) const;
	void set_jiggle_joint_stiffness(int p_joint_idx, float p_stiffness);
	float get_jiggle_joint_stiffness(int p_joint_idx) const;
	void set_jiggle_joint_mass(int p_joint_idx, float p_mass);
	float get_jiggle_joint_mass(int p_joint_idx) const;
	void set_jiggle_joint_damping(int p_joint_idx, float p_damping);
	float get_jiggle_joint_damping(int p_joint_idx) const;
	void set_jiggle_joint_use_gravity(int p_joint_idx, bool p_use_gravity);
	bool get_jiggle_joint_use_gravity(int p_joint_idx) const;
	void set_jiggle_joint_gravity(int p_joint_idx, Vector2 p_gravity);
	Vector2 get_jiggle_joint_gravity(int p_joint_idx) const;

	SkeletonModification2DJiggle();
	~SkeletonModification2DJiggle();
};

///////////////////////////////////////
// SkeletonModification2DTwoBoneIK
///////////////////////////////////////

class SkeletonModification2DTwoBoneIK : public SkeletonModification2D {
	GDCLASS(SkeletonModification2DTwoBoneIK, SkeletonModification2D);

private:
	NodePath target_node;
	ObjectID target_node_cache;
	float target_minimum_distance = 0;
	float target_maximum_distance = 0;
	bool flip_bend_direction = false;

	NodePath joint_one_bone2d_node;
	ObjectID joint_one_bone2d_node_cache;
	int joint_one_bone_idx = -1;

	NodePath joint_two_bone2d_node;
	ObjectID joint_two_bone2d_node_cache;
	int joint_two_bone_idx = -1;

#ifdef TOOLS_ENABLED
	bool editor_draw_min_max = false;
#endif // TOOLS_ENABLED

	void update_target_cache();
	void update_joint_one_bone2d_cache();
	void update_joint_two_bone2d_cache();

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void execute(float delta) override;
	void setup_modification(SkeletonModificationStack2D *p_stack) override;
	void draw_editor_gizmo() override;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;

	void set_target_minimum_distance(float p_minimum_distance);
	float get_target_minimum_distance() const;
	void set_target_maximum_distance(float p_maximum_distance);
	float get_target_maximum_distance() const;
	void set_flip_bend_direction(bool p_flip_direction);
	bool get_flip_bend_direction() const;

	void set_joint_one_bone2d_node(const NodePath &p_node);
	NodePath get_joint_one_bone2d_node() const;
	void set_joint_one_bone_idx(int p_bone_idx);
	int get_joint_one_bone_idx() const;

	void set_joint_two_bone2d_node(const NodePath &p_node);
	NodePath get_joint_two_bone2d_node() const;
	void set_joint_two_bone_idx(int p_bone_idx);
	int get_joint_two_bone_idx() const;

#ifdef TOOLS_ENABLED
	void set_editor_draw_min_max(bool p_draw);
	bool get_editor_draw_min_max() const;
#endif // TOOLS_ENABLED

	SkeletonModification2DTwoBoneIK();
	~SkeletonModification2DTwoBoneIK();
};

///////////////////////////////////////
// SkeletonModification2DPhysicalBones
///////////////////////////////////////

class SkeletonModification2DPhysicalBones : public SkeletonModification2D {
	GDCLASS(SkeletonModification2DPhysicalBones, SkeletonModification2D);

private:
	struct PhysicalBone_Data2D {
		NodePath physical_bone_node;
		ObjectID physical_bone_node_cache;
	};
	Vector<PhysicalBone_Data2D> physical_bone_chain;

	void _physical_bone_update_cache(int p_joint_idx);

	bool _simulation_state_dirty = false;
	TypedArray<StringName> _simulation_state_dirty_names;
	bool _simulation_state_dirty_process;
	void _update_simulation_state();

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void execute(float delta) override;
	void setup_modification(SkeletonModificationStack2D *p_stack) override;

	int get_physical_bone_chain_length();
	void set_physical_bone_chain_length(int p_new_length);

	void set_physical_bone_node(int p_joint_idx, const NodePath &p_path);
	NodePath get_physical_bone_node(int p_joint_idx) const;

	void fetch_physical_bones();
	void start_simulation(const TypedArray<StringName> &p_bones);
	void stop_simulation(const TypedArray<StringName> &p_bones);

	SkeletonModification2DPhysicalBones();
	~SkeletonModification2DPhysicalBones();
};

///////////////////////////////////////
// SkeletonModification2DStackHolder
///////////////////////////////////////

class SkeletonModification2DStackHolder : public SkeletonModification2D {
	GDCLASS(SkeletonModification2DStackHolder, SkeletonModification2D);

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	Ref<SkeletonModificationStack2D> held_modification_stack;

	void execute(float delta) override;
	void setup_modification(SkeletonModificationStack2D *p_stack) override;
	void draw_editor_gizmo() override;

	void set_held_modification_stack(Ref<SkeletonModificationStack2D> p_held_stack);
	Ref<SkeletonModificationStack2D> get_held_modification_stack() const;

	SkeletonModification2DStackHolder();
	~SkeletonModification2DStackHolder();
};

#endif // SKELETONMODIFICATION2D_H

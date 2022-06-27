/*************************************************************************/
/*  skeleton_modification_3d_jiggle.h                                    */
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

#include "core/templates/local_vector.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/skeleton_modification_3d.h"

#ifndef SKELETONMODIFICATION3DJIGGLE_H
#define SKELETONMODIFICATION3DJIGGLE_H

class SkeletonModification3DJiggle : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DJiggle, SkeletonModification3D);

private:
	struct Jiggle_Joint_Data {
		String bone_name = "";
		int bone_idx = -1;

		bool override_defaults = false;
		real_t stiffness = 3;
		real_t mass = 0.75;
		real_t damping = 0.75;
		bool use_gravity = false;
		Vector3 gravity = Vector3(0, -6.0, 0);
		real_t roll = 0;

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
	LocalVector<Jiggle_Joint_Data> jiggle_data_chain;

	real_t stiffness = 3;
	real_t mass = 0.75;
	real_t damping = 0.75;
	bool use_gravity = false;
	Vector3 gravity = Vector3(0, -6.0, 0);

	bool use_colliders = false;
	uint32_t collision_mask = 1;

	void update_cache();
	void _execute_jiggle_joint(int p_joint_idx, Node3D *p_target, real_t p_delta);
	void _update_jiggle_joint_data();

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	virtual void _execute(real_t p_delta) override;
	virtual void _setup_modification(SkeletonModificationStack3D *p_stack) override;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;

	void set_stiffness(real_t p_stiffness);
	real_t get_stiffness() const;
	void set_mass(real_t p_mass);
	real_t get_mass() const;
	void set_damping(real_t p_damping);
	real_t get_damping() const;

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

	void set_jiggle_joint_bone_name(int p_joint_idx, String p_name);
	String get_jiggle_joint_bone_name(int p_joint_idx) const;
	void set_jiggle_joint_bone_index(int p_joint_idx, int p_idx);
	int get_jiggle_joint_bone_index(int p_joint_idx) const;

	void set_jiggle_joint_override(int p_joint_idx, bool p_override);
	bool get_jiggle_joint_override(int p_joint_idx) const;
	void set_jiggle_joint_stiffness(int p_joint_idx, real_t p_stiffness);
	real_t get_jiggle_joint_stiffness(int p_joint_idx) const;
	void set_jiggle_joint_mass(int p_joint_idx, real_t p_mass);
	real_t get_jiggle_joint_mass(int p_joint_idx) const;
	void set_jiggle_joint_damping(int p_joint_idx, real_t p_damping);
	real_t get_jiggle_joint_damping(int p_joint_idx) const;
	void set_jiggle_joint_use_gravity(int p_joint_idx, bool p_use_gravity);
	bool get_jiggle_joint_use_gravity(int p_joint_idx) const;
	void set_jiggle_joint_gravity(int p_joint_idx, Vector3 p_gravity);
	Vector3 get_jiggle_joint_gravity(int p_joint_idx) const;
	void set_jiggle_joint_roll(int p_joint_idx, real_t p_roll);
	real_t get_jiggle_joint_roll(int p_joint_idx) const;

	SkeletonModification3DJiggle();
	~SkeletonModification3DJiggle();
};

#endif //SKELETONMODIFICATION3DJIGGLE_H

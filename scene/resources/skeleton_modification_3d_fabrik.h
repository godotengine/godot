/*************************************************************************/
/*  skeleton_modification_3d_fabrik.h                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SKELETONMODIFICATION3DFABRIK_H
#define SKELETONMODIFICATION3DFABRIK_H

class SkeletonModification3DFABRIK : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DFABRIK, SkeletonModification3D);

private:
	struct FABRIK_Joint_Data {
		String bone_name = "";
		int bone_idx = -1;
		real_t length = -1;
		Vector3 magnet_position = Vector3(0, 0, 0);

		bool auto_calculate_length = true;
		bool use_tip_node = false;
		NodePath tip_node = NodePath();
		ObjectID tip_node_cache;

		bool use_target_basis = false;
		real_t roll = 0;
	};

	LocalVector<FABRIK_Joint_Data> fabrik_data_chain;
	LocalVector<Transform3D> fabrik_transforms;

	NodePath target_node;
	ObjectID target_node_cache;

	real_t chain_tolerance = 0.01;
	int chain_max_iterations = 10;
	int chain_iterations = 0;

	void update_target_cache();
	void update_joint_tip_cache(int p_joint_idx);

	int final_joint_idx = 0;
	Transform3D target_global_pose = Transform3D();
	Transform3D origin_global_pose = Transform3D();

	void chain_backwards();
	void chain_forwards();
	void chain_apply();

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

	int get_fabrik_data_chain_length();
	void set_fabrik_data_chain_length(int p_new_length);

	real_t get_chain_tolerance();
	void set_chain_tolerance(real_t p_tolerance);

	int get_chain_max_iterations();
	void set_chain_max_iterations(int p_iterations);

	String get_fabrik_joint_bone_name(int p_joint_idx) const;
	void set_fabrik_joint_bone_name(int p_joint_idx, String p_bone_name);
	int get_fabrik_joint_bone_index(int p_joint_idx) const;
	void set_fabrik_joint_bone_index(int p_joint_idx, int p_bone_idx);
	real_t get_fabrik_joint_length(int p_joint_idx) const;
	void set_fabrik_joint_length(int p_joint_idx, real_t p_bone_length);
	Vector3 get_fabrik_joint_magnet(int p_joint_idx) const;
	void set_fabrik_joint_magnet(int p_joint_idx, Vector3 p_magnet);
	bool get_fabrik_joint_auto_calculate_length(int p_joint_idx) const;
	void set_fabrik_joint_auto_calculate_length(int p_joint_idx, bool p_auto_calculate);
	void fabrik_joint_auto_calculate_length(int p_joint_idx);
	bool get_fabrik_joint_use_tip_node(int p_joint_idx) const;
	void set_fabrik_joint_use_tip_node(int p_joint_idx, bool p_use_tip_node);
	NodePath get_fabrik_joint_tip_node(int p_joint_idx) const;
	void set_fabrik_joint_tip_node(int p_joint_idx, NodePath p_tip_node);
	bool get_fabrik_joint_use_target_basis(int p_joint_idx) const;
	void set_fabrik_joint_use_target_basis(int p_joint_idx, bool p_use_basis);
	real_t get_fabrik_joint_roll(int p_joint_idx) const;
	void set_fabrik_joint_roll(int p_joint_idx, real_t p_roll);

	SkeletonModification3DFABRIK();
	~SkeletonModification3DFABRIK();
};

#endif //SKELETONMODIFICATION3DFABRIK_H

/*************************************************************************/
/*  skeleton_modification_3d_twoboneik.h                                 */
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

#include "scene/3d/skeleton_3d.h"
#include "scene/resources/skeleton_modification_3d.h"

#ifndef SKELETONMODIFICATION3DTWOBONEIK_H
#define SKELETONMODIFICATION3DTWOBONEIK_H

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
	real_t joint_one_length = -1;
	real_t joint_two_length = -1;

	real_t joint_one_roll = 0;
	real_t joint_two_roll = 0;

	void update_cache_target();
	void update_cache_tip();
	void update_cache_pole();

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
	void set_joint_one_length(real_t p_length);
	real_t get_joint_one_length() const;

	void set_joint_two_bone_name(String p_bone_name);
	String get_joint_two_bone_name() const;
	void set_joint_two_bone_idx(int p_bone_idx);
	int get_joint_two_bone_idx() const;
	void set_joint_two_length(real_t p_length);
	real_t get_joint_two_length() const;

	void set_joint_one_roll(real_t p_roll);
	real_t get_joint_one_roll() const;
	void set_joint_two_roll(real_t p_roll);
	real_t get_joint_two_roll() const;

	SkeletonModification3DTwoBoneIK();
	~SkeletonModification3DTwoBoneIK();
};

#endif //SKELETONMODIFICATION3DTWOBONEIK_H

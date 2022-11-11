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

#ifndef SKELETON_MODIFICATION_3D_TWOBONEIK_H
#define SKELETON_MODIFICATION_3D_TWOBONEIK_H

#include "scene/3d/skeleton_3d.h"
#include "scene/3d/skeleton_modification_3d.h"

class SkeletonModification3DTwoBoneIK : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DTwoBoneIK, SkeletonModification3D);

private:
	NodePath target_node;
	String target_bone;
	mutable Variant target_cache;

	NodePath tip_node;
	String tip_bone;
	mutable Variant tip_cache;

	NodePath pole_node;
	String pole_bone;
	mutable Variant pole_cache;

	mutable int bone_idx = UNCACHED_BONE_IDX;
	String joint_one_bone_name;
	mutable int joint_one_bone_idx = UNCACHED_BONE_IDX;
	String joint_two_bone_name;
	mutable int joint_two_bone_idx = UNCACHED_BONE_IDX;

	bool auto_calculate_joint_length = false;
	real_t joint_one_length = -1;
	real_t joint_two_length = -1;

	real_t joint_one_roll = 0;
	real_t joint_two_roll = 0;

protected:
	void execute(real_t delta) override;
	static void _bind_methods();
	void skeleton_changed(Skeleton3D *skeleton) override;
	bool is_property_hidden(String property_name) const override;
	bool is_bone_property(String property_name) const override;
	PackedStringArray get_configuration_warnings() const override;

public:
	void set_target_bone(const String &p_target_node);
	String get_target_bone() const;
	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;

	void set_tip_bone(const String &p_tip_bone);
	String get_tip_bone() const;
	void set_tip_node(const NodePath &p_tip_node);
	NodePath get_tip_node() const;

	void set_pole_bone(const String &p_pole_bone);
	String get_pole_bone() const;
	void set_pole_node(const NodePath &p_pole_node);
	NodePath get_pole_node() const;

	void set_auto_calculate_joint_length(bool p_calculate);
	bool get_auto_calculate_joint_length() const;
	void calculate_joint_lengths();

	void set_joint_one_bone(String p_bone_name);
	String get_joint_one_bone() const;
	void set_joint_one_length(real_t p_length);
	real_t get_joint_one_length() const;

	void set_joint_two_bone(String p_bone_name);
	String get_joint_two_bone() const;
	void set_joint_two_length(real_t p_length);
	real_t get_joint_two_length() const;

	void set_joint_one_roll(real_t p_roll);
	real_t get_joint_one_roll() const;
	void set_joint_two_roll(real_t p_roll);
	real_t get_joint_two_roll() const;

	SkeletonModification3DTwoBoneIK();
	~SkeletonModification3DTwoBoneIK();
};

#endif // SKELETON_MODIFICATION_3D_TWOBONEIK_H

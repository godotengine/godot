/*************************************************************************/
/*  skeleton.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef SKELETON_H
#define SKELETON_H

#include "core/rid.h"
#include "scene/3d/spatial.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

#ifndef _3D_DISABLED
typedef int BoneId;

class PhysicalBone;
#endif // _3D_DISABLED

class Skeleton : public Spatial {

	GDCLASS(Skeleton, Spatial);

	struct Bone {

		String name;

		bool enabled;
		int parent;
		int sort_index; //used for re-sorting process order

		bool ignore_animation;

		bool disable_rest;
		Transform rest;
		Transform rest_global_inverse;

		Transform pose;
		Transform pose_global;

		bool custom_pose_enable;
		Transform custom_pose;

		Transform transform_final;

#ifndef _3D_DISABLED
		PhysicalBone *physical_bone;
		PhysicalBone *cache_parent_physical_bone;
#endif // _3D_DISABLED

		List<uint32_t> nodes_bound;

		Bone() {
			parent = -1;
			enabled = true;
			ignore_animation = false;
			custom_pose_enable = false;
			disable_rest = false;
#ifndef _3D_DISABLED
			physical_bone = NULL;
			cache_parent_physical_bone = NULL;
#endif // _3D_DISABLED
		}
	};

	bool rest_global_inverse_dirty;

	Vector<Bone> bones;
	Vector<int> process_order;
	bool process_order_dirty;

	RID skeleton;

	void _make_dirty();
	bool dirty;
	bool use_bones_in_world_transform;

	// bind helpers
	Array _get_bound_child_nodes_to_bone(int p_bone) const {

		Array bound;
		List<Node *> children;
		get_bound_child_nodes_to_bone(p_bone, &children);

		for (int i = 0; i < children.size(); i++) {

			bound.push_back(children[i]);
		}
		return bound;
	}

	void _update_process_order();

protected:
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);
	static void _bind_methods();

public:
	enum {

		NOTIFICATION_UPDATE_SKELETON = 50
	};

	RID get_skeleton() const;

	// skeleton creation api
	void add_bone(const String &p_name);
	int find_bone(const String &p_name) const;
	String get_bone_name(int p_bone) const;

	bool is_bone_parent_of(int p_bone_id, int p_parent_bone_id) const;

	void set_bone_parent(int p_bone, int p_parent);
	int get_bone_parent(int p_bone) const;

	void unparent_bone_and_rest(int p_bone);

	void set_bone_ignore_animation(int p_bone, bool p_ignore);
	bool is_bone_ignore_animation(int p_bone) const;

	void set_bone_disable_rest(int p_bone, bool p_disable);
	bool is_bone_rest_disabled(int p_bone) const;

	int get_bone_count() const;

	void set_bone_rest(int p_bone, const Transform &p_rest);
	Transform get_bone_rest(int p_bone) const;
	Transform get_bone_transform(int p_bone) const;
	Transform get_bone_global_pose(int p_bone) const;

	void set_bone_global_pose(int p_bone, const Transform &p_pose);

	void set_bone_enabled(int p_bone, bool p_enabled);
	bool is_bone_enabled(int p_bone) const;

	void bind_child_node_to_bone(int p_bone, Node *p_node);
	void unbind_child_node_from_bone(int p_bone, Node *p_node);
	void get_bound_child_nodes_to_bone(int p_bone, List<Node *> *p_bound) const;

	void clear_bones();

	// posing api

	void set_bone_pose(int p_bone, const Transform &p_pose);
	Transform get_bone_pose(int p_bone) const;

	void set_bone_custom_pose(int p_bone, const Transform &p_custom_pose);
	Transform get_bone_custom_pose(int p_bone) const;

	void localize_rests(); // used for loaders and tools
	int get_process_order(int p_idx);

	void set_use_bones_in_world_transform(bool p_enable);
	bool is_using_bones_in_world_transform() const;

#ifndef _3D_DISABLED
	// Physical bone API

	void bind_physical_bone_to_bone(int p_bone, PhysicalBone *p_physical_bone);
	void unbind_physical_bone_from_bone(int p_bone);

	PhysicalBone *get_physical_bone(int p_bone);
	PhysicalBone *get_physical_bone_parent(int p_bone);

private:
	/// This is a slow API os it's cached
	PhysicalBone *_get_physical_bone_parent(int p_bone);
	void _rebuild_physical_bones_cache();

public:
	void physical_bones_stop_simulation();
	void physical_bones_start_simulation_on(const Array &p_bones);
	void physical_bones_add_collision_exception(RID p_exception);
	void physical_bones_remove_collision_exception(RID p_exception);
#endif // _3D_DISABLED

public:
	Skeleton();
	~Skeleton();
};

#endif

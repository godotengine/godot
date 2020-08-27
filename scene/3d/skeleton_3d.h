/*************************************************************************/
/*  skeleton_3d.h                                                        */
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

#ifndef SKELETON_3D_H
#define SKELETON_3D_H

#include "core/rid.h"
#include "scene/3d/node_3d.h"
#include "scene/resources/skin.h"

#ifndef _3D_DISABLED
typedef int BoneId;

class PhysicalBone3D;
#endif // _3D_DISABLED

class Skeleton3D;

class SkinReference : public Reference {
	GDCLASS(SkinReference, Reference)
	friend class Skeleton3D;

	Skeleton3D *skeleton_node;
	RID skeleton;
	Ref<Skin> skin;
	uint32_t bind_count = 0;
	uint64_t skeleton_version = 0;
	Vector<uint32_t> skin_bone_indices;
	uint32_t *skin_bone_indices_ptrs;
	void _skin_changed();

protected:
	static void _bind_methods();

public:
	RID get_skeleton() const;
	Ref<Skin> get_skin() const;
	~SkinReference();
};

class Skeleton3D : public Node3D {
	GDCLASS(Skeleton3D, Node3D);

private:
	friend class SkinReference;

	struct Bone {
		String name;

		bool enabled;
		int parent;
		int sort_index; //used for re-sorting process order

		bool disable_rest;
		Transform rest;

		Transform pose;
		Transform pose_global;

		bool custom_pose_enable;
		Transform custom_pose;

		float global_pose_override_amount;
		bool global_pose_override_reset;
		Transform global_pose_override;

#ifndef _3D_DISABLED
		PhysicalBone3D *physical_bone;
		PhysicalBone3D *cache_parent_physical_bone;
#endif // _3D_DISABLED

		List<ObjectID> nodes_bound;

		Bone() {
			parent = -1;
			enabled = true;
			disable_rest = false;
			custom_pose_enable = false;
			global_pose_override_amount = 0;
			global_pose_override_reset = false;
#ifndef _3D_DISABLED
			physical_bone = nullptr;
			cache_parent_physical_bone = nullptr;
#endif // _3D_DISABLED
		}
	};

	Set<SkinReference *> skin_bindings;

	void _skin_changed();

	bool animate_physical_bones;
	Vector<Bone> bones;
	Vector<int> process_order;
	bool process_order_dirty;

	void _make_dirty();
	bool dirty;

	uint64_t version;

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

	// skeleton creation api
	void add_bone(const String &p_name);
	int find_bone(const String &p_name) const;
	String get_bone_name(int p_bone) const;

	bool is_bone_parent_of(int p_bone_id, int p_parent_bone_id) const;

	void set_bone_parent(int p_bone, int p_parent);
	int get_bone_parent(int p_bone) const;

	void unparent_bone_and_rest(int p_bone);

	void set_bone_disable_rest(int p_bone, bool p_disable);
	bool is_bone_rest_disabled(int p_bone) const;

	int get_bone_count() const;

	void set_bone_rest(int p_bone, const Transform &p_rest);
	Transform get_bone_rest(int p_bone) const;
	Transform get_bone_global_pose(int p_bone) const;

	void clear_bones_global_pose_override();
	void set_bone_global_pose_override(int p_bone, const Transform &p_pose, float p_amount, bool p_persistent = false);

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
	Vector<int> get_bone_process_orders();

	Ref<SkinReference> register_skin(const Ref<Skin> &p_skin);

	// Helper functions
	Transform bone_transform_to_world_transform(Transform p_transform);
	Transform world_transform_to_bone_transform(Transform p_transform);

#ifndef _3D_DISABLED
	// Physical bone API

	void set_animate_physical_bones(bool p_animate);
	bool get_animate_physical_bones() const;

	void bind_physical_bone_to_bone(int p_bone, PhysicalBone3D *p_physical_bone);
	void unbind_physical_bone_from_bone(int p_bone);

	PhysicalBone3D *get_physical_bone(int p_bone);
	PhysicalBone3D *get_physical_bone_parent(int p_bone);

private:
	/// This is a slow API, so it's cached
	PhysicalBone3D *_get_physical_bone_parent(int p_bone);
	void _rebuild_physical_bones_cache();

public:
	void physical_bones_stop_simulation();
	void physical_bones_start_simulation_on(const TypedArray<StringName> &p_bones);
	void physical_bones_add_collision_exception(RID p_exception);
	void physical_bones_remove_collision_exception(RID p_exception);
#endif // _3D_DISABLED

public:
	Skeleton3D();
	~Skeleton3D();
};

#endif

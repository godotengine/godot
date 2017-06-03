/*************************************************************************/
/*  skeleton.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "rid.h"
#include "scene/3d/spatial.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class Skeleton : public Spatial {

	GDCLASS(Skeleton, Spatial);

	struct Bone {

		String name;

		bool enabled;
		int parent;

		bool disable_rest;
		Transform rest;
		Transform rest_global_inverse;

		Transform pose;
		Transform pose_global;

		bool custom_pose_enable;
		Transform custom_pose;

		List<uint32_t> nodes_bound;

		Bone() {
			parent = -1;
			enabled = true;
			custom_pose_enable = false;
			disable_rest = false;
		}
	};

	bool rest_global_inverse_dirty;

	Vector<Bone> bones;

	RID skeleton;

	void _make_dirty();
	bool dirty;

	//bind helpers
	Array _get_bound_child_nodes_to_bone(int p_bone) const {

		Array bound;
		List<Node *> childs;
		get_bound_child_nodes_to_bone(p_bone, &childs);

		for (int i = 0; i < childs.size(); i++) {

			bound.push_back(childs[i]);
		}
		return bound;
	}

protected:
	bool _get(const StringName &p_name, Variant &r_ret) const;
	bool _set(const StringName &p_name, const Variant &p_value);
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
	int find_bone(String p_name) const;
	String get_bone_name(int p_bone) const;

	void set_bone_parent(int p_bone, int p_parent);
	int get_bone_parent(int p_bone) const;

	void unparent_bone_and_rest(int p_idx);

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

	Skeleton();
	~Skeleton();
};

#endif

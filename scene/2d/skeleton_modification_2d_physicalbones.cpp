/*************************************************************************/
/*  skeleton_modification_2d_physicalbones.cpp                           */
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

#include "skeleton_modification_2d_physicalbones.h"
#include "scene/2d/physical_bone_2d.h"
#include "scene/2d/skeleton_2d.h"

bool SkeletonModification2DPhysicalBones::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

#ifdef TOOLS_ENABLED
	// Exposes a way to fetch the PhysicalBone2D nodes from the Godot editor.
	if (Engine::get_singleton()->is_editor_hint()) {
		if (path.begins_with("fetch_bones")) {
			_fetch_physical_bones();
			notify_property_list_changed();
			return true;
		}
	}
#endif //TOOLS_ENABLED

	if (path.begins_with("joint_")) {
		int which = path.get_slicec('_', 0).substr(6).to_int();
		String what = path.get_slicec('_', 1);
		ERR_FAIL_INDEX_V(which, physical_bone_chain.size(), false);

		if (what == "node") {
			set_physical_bone_node(which, p_value);
		}
		return true;
	}
	return true;
}

bool SkeletonModification2DPhysicalBones::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		if (path.begins_with("fetch_bones")) {
			return true; // Do nothing!
		}
	}
#endif //TOOLS_ENABLED

	if (path.begins_with("joint_")) {
		int which = path.get_slicec('_', 0).substr(6).to_int();
		String what = path.get_slicec('_', 1);
		ERR_FAIL_INDEX_V(which, physical_bone_chain.size(), false);

		if (what == "node") {
			r_ret = get_physical_bone_node(which);
		}
		return true;
	}
	return true;
}

void SkeletonModification2DPhysicalBones::_get_property_list(List<PropertyInfo> *p_list) const {
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint()) {
		p_list->push_back(PropertyInfo(Variant::BOOL, "fetch_bones", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
	}
#endif //TOOLS_ENABLED

	for (int i = 0; i < physical_bone_chain.size(); i++) {
		String base_string = "joint_" + itos(i) + "/";

		p_list->push_back(PropertyInfo(Variant::NODE_PATH, base_string + "node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "PhysicalBone2D", PROPERTY_USAGE_DEFAULT));
	}
}

int SkeletonModification2DPhysicalBones::get_joint_count() {
	return physical_bone_chain.size();
}

void SkeletonModification2DPhysicalBones::set_joint_count(int p_length) {
	ERR_FAIL_COND(p_length < 0);
	physical_bone_chain.resize(p_length);
	notify_property_list_changed();
}

#ifdef TOOLS_ENABLED
void SkeletonModification2DPhysicalBones::_fetch_physical_bones() {
	Skeleton2D *skeleton = get_skeleton();
	ERR_FAIL_COND_MSG(!skeleton, "No skeleton found! Cannot fetch physical bones!");

	physical_bone_chain.clear();

	List<Node *> node_queue = List<Node *>();
	node_queue.push_back(skeleton);

	while (node_queue.size() > 0) {
		Node *node_to_process = node_queue[0];
		node_queue.pop_front();

		if (node_to_process != nullptr) {
			PhysicalBone2D *potential_bone = Object::cast_to<PhysicalBone2D>(node_to_process);
			if (potential_bone) {
				PhysicalBone_Data2D new_data = PhysicalBone_Data2D();
				new_data.physical_bone_node = skeleton->get_path_to(potential_bone);
				new_data.physical_bone_node_cache = potential_bone->get_instance_id();
				physical_bone_chain.push_back(new_data);
			}
			for (int i = 0; i < node_to_process->get_child_count(); i++) {
				node_queue.push_back(node_to_process->get_child(i));
			}
		}
	}
}
#endif

void SkeletonModification2DPhysicalBones::set_physical_bone_node(int p_joint_idx, const NodePath &p_nodepath) {
	ERR_FAIL_INDEX_MSG(p_joint_idx, physical_bone_chain.size(), "Joint index out of range!");
	physical_bone_chain.write[p_joint_idx].physical_bone_node = p_nodepath;
	physical_bone_chain.write[p_joint_idx].physical_bone_node_cache = Variant();
}

NodePath SkeletonModification2DPhysicalBones::get_physical_bone_node(int p_joint_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_joint_idx, physical_bone_chain.size(), NodePath(), "Joint index out of range!");
	return physical_bone_chain[p_joint_idx].physical_bone_node;
}

void SkeletonModification2DPhysicalBones::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_joint_count", "length"), &SkeletonModification2DPhysicalBones::set_joint_count);
	ClassDB::bind_method(D_METHOD("get_joint_count"), &SkeletonModification2DPhysicalBones::get_joint_count);

	ClassDB::bind_method(D_METHOD("set_physical_bone_node", "joint_idx", "physicalbone2d_node"), &SkeletonModification2DPhysicalBones::set_physical_bone_node);
	ClassDB::bind_method(D_METHOD("get_physical_bone_node", "joint_idx"), &SkeletonModification2DPhysicalBones::get_physical_bone_node);

	ADD_ARRAY_COUNT("Physical bone chain", "joint_count", "set_joint_count", "get_joint_count", "joint_");
}

SkeletonModification2DPhysicalBones::SkeletonModification2DPhysicalBones() {
	physical_bone_chain = Vector<PhysicalBone_Data2D>();
}

SkeletonModification2DPhysicalBones::~SkeletonModification2DPhysicalBones() {
}

void SkeletonModification2DPhysicalBones::execute(real_t delta) {
	SkeletonModification2D::execute(delta);

	for (int i = 0; i < physical_bone_chain.size(); i++) {
		const PhysicalBone_Data2D &bone_data = physical_bone_chain[i];
		if (_cache_bone(bone_data.physical_bone_node_cache, bone_data.physical_bone_node)) {
			WARN_PRINT_ONCE("2DPhysicalBones unable to get a physical bone");
			return;
		}
	}

	for (int i = 0; i < physical_bone_chain.size(); i++) {
		const PhysicalBone_Data2D &bone_data = physical_bone_chain[i];
		PhysicalBone2D *physical_bone = Object::cast_to<PhysicalBone2D>(_cache_bone(bone_data.physical_bone_node_cache, bone_data.physical_bone_node));
		if (physical_bone) {
			Node2D *bone_2d = physical_bone->get_cached_bone_node();
			if (bone_2d && physical_bone->get_simulate_physics() && !physical_bone->get_follow_bone_when_simulating()) {
				bone_2d->set_global_transform(physical_bone->get_global_transform());
			}
		}
	}
}

PackedStringArray SkeletonModification2DPhysicalBones::get_configuration_warnings() const {
	PackedStringArray ret = SkeletonModification2D::get_configuration_warnings();
	if (!get_skeleton()) {
		return ret;
	}
	for (int i = 0; i < physical_bone_chain.size(); i++) {
		if (!_cache_node(physical_bone_chain[i].physical_bone_node_cache, physical_bone_chain[i].physical_bone_node)) {
			ret.append(vformat("Joint %d physical bone path %s was not found.", i, physical_bone_chain[i].physical_bone_node));
		}
		PhysicalBone2D *physbone = cast_to<PhysicalBone2D>((Object *)physical_bone_chain[i].physical_bone_node_cache);
		if (!physbone) {
			ret.append(vformat("Joint %d physical bone path %s is not a PhysicalBone2D.", i, physical_bone_chain[i].physical_bone_node));
		} else {
			Node2D *bone = physbone->get_cached_bone_node();
			if (!bone) {
				ret.append(vformat("Joint %d physical bone %s not connected to a bone.", i, physical_bone_chain[i].physical_bone_node));
			}
		}
	}
	return ret;
}

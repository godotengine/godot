/**************************************************************************/
/*  skeleton.cpp                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "skeleton.h"

#include "core/message_queue.h"

#include "core/project_settings.h"
#include "scene/3d/physics_body.h"
#include "scene/resources/surface_tool.h"

void SkinReference::_skin_changed() {
	if (skeleton_node) {
		skeleton_node->_make_dirty();
	}
	skeleton_version = 0;
}

void SkinReference::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_skin_changed"), &SkinReference::_skin_changed);
	ClassDB::bind_method(D_METHOD("get_skeleton"), &SkinReference::get_skeleton);
	ClassDB::bind_method(D_METHOD("get_skin"), &SkinReference::get_skin);
}

RID SkinReference::get_skeleton() const {
	return skeleton;
}

Skeleton *SkinReference::get_skeleton_node() const {
	return skeleton_node;
}

Ref<Skin> SkinReference::get_skin() const {
	return skin;
}

SkinReference::~SkinReference() {
	if (skeleton_node) {
		skeleton_node->skin_bindings.erase(this);
	}

	VS::get_singleton()->free(skeleton);
}

bool Skeleton::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (!path.begins_with("bones/")) {
		return false;
	}

	int which = path.get_slicec('/', 1).to_int();
	String what = path.get_slicec('/', 2);

	if (which == bones.size() && what == "name") {
		add_bone(p_value);
		return true;
	}

	ERR_FAIL_INDEX_V(which, bones.size(), false);

	if (what == "parent") {
		set_bone_parent(which, p_value);
	} else if (what == "rest") {
		set_bone_rest(which, p_value);
	} else if (what == "enabled") {
		set_bone_enabled(which, p_value);
	} else if (what == "pose") {
		set_bone_pose(which, p_value);
	} else if (what == "bound_children") {
		Array children = p_value;

		if (is_inside_tree()) {
			bones.write[which].nodes_bound.clear();

			for (int i = 0; i < children.size(); i++) {
				NodePath npath = children[i];
				ERR_CONTINUE(npath.operator String() == "");
				Node *node = get_node(npath);
				ERR_CONTINUE(!node);
				bind_child_node_to_bone(which, node);
			}
		}
	} else {
		return false;
	}

	return true;
}

bool Skeleton::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (!path.begins_with("bones/")) {
		return false;
	}

	int which = path.get_slicec('/', 1).to_int();
	String what = path.get_slicec('/', 2);

	ERR_FAIL_INDEX_V(which, bones.size(), false);

	if (what == "name") {
		r_ret = get_bone_name(which);
	} else if (what == "parent") {
		r_ret = get_bone_parent(which);
	} else if (what == "rest") {
		r_ret = get_bone_rest(which);
	} else if (what == "enabled") {
		r_ret = is_bone_enabled(which);
	} else if (what == "pose") {
		r_ret = get_bone_pose(which);
	} else if (what == "bound_children") {
		Array children;

		const LocalVectori<uint32_t> &nodes = bones[which].nodes_bound;
		for (int i = 0; i < nodes.size(); i++) {
			Object *obj = ObjectDB::get_instance(nodes[i]);
			ERR_CONTINUE(!obj);
			Node *node = Object::cast_to<Node>(obj);
			ERR_CONTINUE(!node);
			NodePath npath = get_path_to(node);
			children.push_back(npath);
		}

		r_ret = children;
	} else {
		return false;
	}

	return true;
}
void Skeleton::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < bones.size(); i++) {
		const String prep = vformat("%s/%d/", PNAME("bones"), i);
		p_list->push_back(PropertyInfo(Variant::STRING, prep + PNAME("name")));
		p_list->push_back(PropertyInfo(Variant::INT, prep + PNAME("parent"), PROPERTY_HINT_RANGE, "-1," + itos(bones.size() - 1) + ",1"));
		p_list->push_back(PropertyInfo(Variant::TRANSFORM, prep + PNAME("rest")));
		p_list->push_back(PropertyInfo(Variant::BOOL, prep + PNAME("enabled")));
		p_list->push_back(PropertyInfo(Variant::TRANSFORM, prep + PNAME("pose"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::ARRAY, prep + PNAME("bound_children")));
	}
}

void Skeleton::_update_process_order() {
	if (!process_order_dirty) {
		return;
	}

	Bone *bonesptr = bones.ptrw();
	int len = bones.size();

	process_order.resize(len);
	int *order = process_order.ptrw();
	for (int i = 0; i < len; i++) {
		if (bonesptr[i].parent >= len) {
			//validate this just in case
			ERR_PRINT("Bone " + itos(i) + " has invalid parent: " + itos(bonesptr[i].parent));
			bonesptr[i].parent = -1;
		}
		order[i] = i;
		bonesptr[i].sort_index = i;
	}
	//now check process order
	int pass_count = 0;
	while (pass_count < len * len) {
		//using bubblesort because of simplicity, it won't run every frame though.
		//bublesort worst case is O(n^2), and this may be an infinite loop if cyclic
		bool swapped = false;
		for (int i = 0; i < len; i++) {
			int parent_idx = bonesptr[order[i]].parent;
			if (parent_idx < 0) {
				continue; //do nothing because it has no parent
			}
			//swap indices
			int parent_order = bonesptr[parent_idx].sort_index;
			if (parent_order > i) {
				bonesptr[order[i]].sort_index = parent_order;
				bonesptr[parent_idx].sort_index = i;
				//swap order
				SWAP(order[i], order[parent_order]);
				swapped = true;
			}
		}

		if (!swapped) {
			break;
		}
		pass_count++;
	}

	if (pass_count == len * len) {
		ERR_PRINT("Skeleton parenthood graph is cyclic");
	}

	process_order_dirty = false;
}

void Skeleton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_UPDATE_SKELETON: {
			VisualServer *vs = VisualServer::get_singleton();
			Bone *bonesptr = bones.ptrw();
			int len = bones.size();

			_update_process_order();

			const int *order = process_order.ptr();

			for (int i = 0; i < len; i++) {
				Bone &b = bonesptr[order[i]];

				if (b.disable_rest) {
					if (b.enabled) {
						Transform pose = b.pose;
						if (b.custom_pose_enable) {
							pose = b.custom_pose * pose;
						}
						if (b.parent >= 0) {
							b.pose_global = bonesptr[b.parent].pose_global * pose;
							b.pose_global_no_override = bonesptr[b.parent].pose_global_no_override * pose;
						} else {
							b.pose_global = pose;
							b.pose_global_no_override = pose;
						}
					} else {
						if (b.parent >= 0) {
							b.pose_global = bonesptr[b.parent].pose_global;
							b.pose_global_no_override = bonesptr[b.parent].pose_global_no_override;
						} else {
							b.pose_global = Transform();
							b.pose_global_no_override = Transform();
						}
					}
				} else {
					if (b.enabled) {
						Transform pose = b.pose;
						if (b.custom_pose_enable) {
							pose = b.custom_pose * pose;
						}
						if (b.parent >= 0) {
							b.pose_global = bonesptr[b.parent].pose_global * (b.rest * pose);
							b.pose_global_no_override = bonesptr[b.parent].pose_global_no_override * (b.rest * pose);
						} else {
							b.pose_global = b.rest * pose;
							b.pose_global_no_override = b.rest * pose;
						}
					} else {
						if (b.parent >= 0) {
							b.pose_global = bonesptr[b.parent].pose_global * b.rest;
							b.pose_global_no_override = bonesptr[b.parent].pose_global_no_override * b.rest;
						} else {
							b.pose_global = b.rest;
							b.pose_global_no_override = b.rest;
						}
					}
				}

				if (b.global_pose_override_amount >= CMP_EPSILON) {
					b.pose_global = b.pose_global.interpolate_with(b.global_pose_override, b.global_pose_override_amount);
				}

				if (b.global_pose_override_reset) {
					b.global_pose_override_amount = 0.0;
				}

				const LocalVectori<uint32_t> &nodes = b.nodes_bound;
				for (int j = 0; j < nodes.size(); j++) {
					Object *obj = ObjectDB::get_instance(nodes[j]);
					ERR_CONTINUE(!obj);
					Spatial *sp = Object::cast_to<Spatial>(obj);
					ERR_CONTINUE(!sp);
					sp->set_transform(b.pose_global);
				}
			}

			//update skins
			for (Set<SkinReference *>::Element *E = skin_bindings.front(); E; E = E->next()) {
				const Skin *skin = E->get()->skin.operator->();
				RID skeleton = E->get()->skeleton;
				uint32_t bind_count = skin->get_bind_count();

				if (E->get()->bind_count != bind_count) {
					VS::get_singleton()->skeleton_allocate(skeleton, bind_count);
					E->get()->bind_count = bind_count;
					E->get()->skin_bone_indices.resize(bind_count);
					E->get()->skin_bone_indices_ptrs = E->get()->skin_bone_indices.ptrw();
				}

				if (E->get()->skeleton_version != version) {
					for (uint32_t i = 0; i < bind_count; i++) {
						StringName bind_name = skin->get_bind_name(i);

						if (bind_name != StringName()) {
							//bind name used, use this
							bool found = false;
							for (int j = 0; j < len; j++) {
								if (bonesptr[j].name == bind_name) {
									E->get()->skin_bone_indices_ptrs[i] = j;
									found = true;
									break;
								}
							}

							if (!found) {
								ERR_PRINT("Skin bind #" + itos(i) + " contains named bind '" + String(bind_name) + "' but Skeleton has no bone by that name.");
								E->get()->skin_bone_indices_ptrs[i] = 0;
							}
						} else if (skin->get_bind_bone(i) >= 0) {
							int bind_index = skin->get_bind_bone(i);
							if (bind_index >= len) {
								ERR_PRINT("Skin bind #" + itos(i) + " contains bone index bind: " + itos(bind_index) + " , which is greater than the skeleton bone count: " + itos(len) + ".");
								E->get()->skin_bone_indices_ptrs[i] = 0;
							} else {
								E->get()->skin_bone_indices_ptrs[i] = bind_index;
							}
						} else {
							ERR_PRINT("Skin bind #" + itos(i) + " does not contain a name nor a bone index.");
							E->get()->skin_bone_indices_ptrs[i] = 0;
						}
					}

					E->get()->skeleton_version = version;
				}

				for (uint32_t i = 0; i < bind_count; i++) {
					uint32_t bone_index = E->get()->skin_bone_indices_ptrs[i];
					ERR_CONTINUE(bone_index >= (uint32_t)len);
					vs->skeleton_bone_set_transform(skeleton, i, bonesptr[bone_index].pose_global * skin->get_bind_pose(i));
				}
			}

			dirty = false;
			emit_signal("skeleton_updated");
		} break;
	}
}

void Skeleton::clear_bones_global_pose_override() {
	for (int i = 0; i < bones.size(); i += 1) {
		bones.write[i].global_pose_override_amount = 0;
		bones.write[i].global_pose_override_reset = true;
	}
	_make_dirty();
}

void Skeleton::set_bone_global_pose_override(int p_bone, const Transform &p_pose, float p_amount, bool p_persistent) {
	ERR_FAIL_INDEX(p_bone, bones.size());
	bones.write[p_bone].global_pose_override_amount = p_amount;
	bones.write[p_bone].global_pose_override = p_pose;
	bones.write[p_bone].global_pose_override_reset = !p_persistent;
	_make_dirty();
}

Transform Skeleton::get_bone_global_pose(int p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), Transform());
	if (dirty) {
		const_cast<Skeleton *>(this)->notification(NOTIFICATION_UPDATE_SKELETON);
	}
	return bones[p_bone].pose_global;
}

Transform Skeleton::get_bone_global_pose_no_override(int p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), Transform());
	if (dirty)
		const_cast<Skeleton *>(this)->notification(NOTIFICATION_UPDATE_SKELETON);
	return bones[p_bone].pose_global_no_override;
}

// skeleton creation api
void Skeleton::add_bone(const String &p_name) {
	ERR_FAIL_COND(p_name == "" || p_name.find(":") != -1 || p_name.find("/") != -1);

	for (int i = 0; i < bones.size(); i++) {
		ERR_FAIL_COND(bones[i].name == p_name);
	}

	Bone b;
	b.name = p_name;
	bones.push_back(b);
	process_order_dirty = true;
	version++;
	_make_dirty();
	update_gizmo();
}
int Skeleton::find_bone(const String &p_name) const {
	for (int i = 0; i < bones.size(); i++) {
		if (bones[i].name == p_name) {
			return i;
		}
	}

	return -1;
}
String Skeleton::get_bone_name(int p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), "");

	return bones[p_bone].name;
}

void Skeleton::set_bone_name(int p_bone, const String &p_name) {
	ERR_FAIL_INDEX(p_bone, bones.size());

	for (int i = 0; i < bones.size(); i++) {
		if (i != p_bone) {
			ERR_FAIL_COND(bones[i].name == p_name);
		}
	}

	bones.write[p_bone].name = p_name;
}

bool Skeleton::is_bone_parent_of(int p_bone, int p_parent_bone_id) const {
	int parent_of_bone = get_bone_parent(p_bone);

	if (-1 == parent_of_bone) {
		return false;
	}

	if (parent_of_bone == p_parent_bone_id) {
		return true;
	}

	return is_bone_parent_of(parent_of_bone, p_parent_bone_id);
}

int Skeleton::get_bone_count() const {
	return bones.size();
}

void Skeleton::set_bone_parent(int p_bone, int p_parent) {
	ERR_FAIL_INDEX(p_bone, bones.size());
	ERR_FAIL_COND(p_parent != -1 && (p_parent < 0));
	ERR_FAIL_COND(p_bone == p_parent);

	bones.write[p_bone].parent = p_parent;
	process_order_dirty = true;
	_make_dirty();
}

void Skeleton::unparent_bone_and_rest(int p_bone) {
	ERR_FAIL_INDEX(p_bone, bones.size());

	_update_process_order();

	int parent = bones[p_bone].parent;
	while (parent >= 0) {
		bones.write[p_bone].rest = bones[parent].rest * bones[p_bone].rest;
		parent = bones[parent].parent;
	}

	bones.write[p_bone].parent = -1;
	process_order_dirty = true;

	_make_dirty();
}

void Skeleton::set_bone_disable_rest(int p_bone, bool p_disable) {
	ERR_FAIL_INDEX(p_bone, bones.size());
	bones.write[p_bone].disable_rest = p_disable;
}

bool Skeleton::is_bone_rest_disabled(int p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), false);
	return bones[p_bone].disable_rest;
}

int Skeleton::get_bone_parent(int p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), -1);

	return bones[p_bone].parent;
}

void Skeleton::set_bone_rest(int p_bone, const Transform &p_rest) {
	ERR_FAIL_INDEX(p_bone, bones.size());

	bones.write[p_bone].rest = p_rest;
	_make_dirty();
}
Transform Skeleton::get_bone_rest(int p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), Transform());

	return bones[p_bone].rest;
}

void Skeleton::set_bone_enabled(int p_bone, bool p_enabled) {
	ERR_FAIL_INDEX(p_bone, bones.size());

	bones.write[p_bone].enabled = p_enabled;
	_make_dirty();
}
bool Skeleton::is_bone_enabled(int p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), false);
	return bones[p_bone].enabled;
}

void Skeleton::bind_child_node_to_bone(int p_bone, Node *p_node) {
	ERR_FAIL_NULL(p_node);
	ERR_FAIL_INDEX(p_bone, bones.size());

	uint32_t id = p_node->get_instance_id();

	if (bones[p_bone].nodes_bound.find(id) != -1) {
		return; // Already here.
	}

	bones.write[p_bone].nodes_bound.push_back(id);
}
void Skeleton::unbind_child_node_from_bone(int p_bone, Node *p_node) {
	ERR_FAIL_NULL(p_node);
	ERR_FAIL_INDEX(p_bone, bones.size());

	uint32_t id = p_node->get_instance_id();

	int index = bones[p_bone].nodes_bound.find(id);
	if (index == -1) {
		return; // Doesn't exist in the first place.
	}

	bones.write[p_bone].nodes_bound.remove(index);
}
void Skeleton::get_bound_child_nodes_to_bone(int p_bone, List<Node *> *p_bound) const {
	ERR_FAIL_INDEX(p_bone, bones.size());

	const LocalVectori<uint32_t> &nodes = bones[p_bone].nodes_bound;
	for (int i = 0; i < nodes.size(); i++) {
		Object *obj = ObjectDB::get_instance(nodes[i]);
		ERR_CONTINUE(!obj);
		p_bound->push_back(Object::cast_to<Node>(obj));
	}
}

void Skeleton::clear_bones() {
	bones.clear();
	process_order_dirty = true;
	version++;
	_make_dirty();
}

// posing api

void Skeleton::set_bone_pose(int p_bone, const Transform &p_pose) {
	ERR_FAIL_INDEX(p_bone, bones.size());

	bones.write[p_bone].pose = p_pose;
	if (is_inside_tree()) {
		_make_dirty();
	}
}
Transform Skeleton::get_bone_pose(int p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), Transform());
	return bones[p_bone].pose;
}

void Skeleton::set_bone_custom_pose(int p_bone, const Transform &p_custom_pose) {
	ERR_FAIL_INDEX(p_bone, bones.size());
	//ERR_FAIL_COND( !is_inside_scene() );

	bones.write[p_bone].custom_pose_enable = (p_custom_pose != Transform());
	bones.write[p_bone].custom_pose = p_custom_pose;

	_make_dirty();
}

Transform Skeleton::get_bone_custom_pose(int p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), Transform());
	return bones[p_bone].custom_pose;
}

void Skeleton::_make_dirty() {
	if (dirty) {
		return;
	}

	MessageQueue::get_singleton()->push_notification(this, NOTIFICATION_UPDATE_SKELETON);
	dirty = true;
}

int Skeleton::get_process_order(int p_idx) {
	ERR_FAIL_INDEX_V(p_idx, bones.size(), -1);
	_update_process_order();
	return process_order[p_idx];
}

void Skeleton::localize_rests() {
	_update_process_order();

	for (int i = bones.size() - 1; i >= 0; i--) {
		int idx = process_order[i];
		if (bones[idx].parent >= 0) {
			set_bone_rest(idx, bones[bones[idx].parent].rest.affine_inverse() * bones[idx].rest);
		}
	}
}

#ifndef _3D_DISABLED

void Skeleton::bind_physical_bone_to_bone(int p_bone, PhysicalBone *p_physical_bone) {
	ERR_FAIL_INDEX(p_bone, bones.size());
	ERR_FAIL_COND(bones[p_bone].physical_bone);
	ERR_FAIL_COND(!p_physical_bone);
	bones.write[p_bone].physical_bone = p_physical_bone;

	_rebuild_physical_bones_cache();
}

void Skeleton::unbind_physical_bone_from_bone(int p_bone) {
	ERR_FAIL_INDEX(p_bone, bones.size());
	bones.write[p_bone].physical_bone = nullptr;

	_rebuild_physical_bones_cache();
}

PhysicalBone *Skeleton::get_physical_bone(int p_bone) {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), nullptr);

	return bones[p_bone].physical_bone;
}

PhysicalBone *Skeleton::get_physical_bone_parent(int p_bone) {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), nullptr);

	if (bones[p_bone].cache_parent_physical_bone) {
		return bones[p_bone].cache_parent_physical_bone;
	}

	return _get_physical_bone_parent(p_bone);
}

PhysicalBone *Skeleton::_get_physical_bone_parent(int p_bone) {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), nullptr);

	const int parent_bone = bones[p_bone].parent;
	if (0 > parent_bone) {
		return nullptr;
	}

	PhysicalBone *pb = bones[parent_bone].physical_bone;
	if (pb) {
		return pb;
	} else {
		return get_physical_bone_parent(parent_bone);
	}
}

void Skeleton::_rebuild_physical_bones_cache() {
	const int b_size = bones.size();
	for (int i = 0; i < b_size; ++i) {
		PhysicalBone *parent_pb = _get_physical_bone_parent(i);
		if (parent_pb != bones[i].cache_parent_physical_bone) {
			bones.write[i].cache_parent_physical_bone = parent_pb;
			if (bones[i].physical_bone) {
				bones[i].physical_bone->_on_bone_parent_changed();
			}
		}
	}
}

void _pb_stop_simulation(Node *p_node) {
	for (int i = p_node->get_child_count() - 1; 0 <= i; --i) {
		_pb_stop_simulation(p_node->get_child(i));
	}

	PhysicalBone *pb = Object::cast_to<PhysicalBone>(p_node);
	if (pb) {
		pb->set_simulate_physics(false);
		pb->set_static_body(false);
	}
}

void Skeleton::physical_bones_stop_simulation() {
	_pb_stop_simulation(this);
}

void _pb_start_simulation(const Skeleton *p_skeleton, Node *p_node, const Vector<int> &p_sim_bones) {
	for (int i = p_node->get_child_count() - 1; 0 <= i; --i) {
		_pb_start_simulation(p_skeleton, p_node->get_child(i), p_sim_bones);
	}

	PhysicalBone *pb = Object::cast_to<PhysicalBone>(p_node);
	if (pb) {
		bool sim = false;
		if (p_sim_bones.empty()) { // If no bones is specified, activate ragdoll on full body.
			sim = true;
		} else {
			for (int i = p_sim_bones.size() - 1; 0 <= i; --i) {
				if (p_sim_bones[i] == pb->get_bone_id() || p_skeleton->is_bone_parent_of(pb->get_bone_id(), p_sim_bones[i])) {
					sim = true;
					break;
				}
			}
		}

		pb->set_simulate_physics(true);
		if (sim) {
			pb->set_static_body(false);
		} else {
			pb->set_static_body(true);
		}
	}
}

void Skeleton::physical_bones_start_simulation_on(const Array &p_bones) {
	Vector<int> sim_bones;
	if (p_bones.size() > 0) {
		sim_bones.resize(p_bones.size());
		int c = 0;
		for (int i = sim_bones.size() - 1; 0 <= i; --i) {
			if (Variant::STRING == p_bones.get(i).get_type()) {
				int bone_id = find_bone(p_bones.get(i));
				if (bone_id != -1) {
					sim_bones.write[c++] = bone_id;
				}
			}
		}
		sim_bones.resize(c);
	}

	_pb_start_simulation(this, this, sim_bones);
}

void _physical_bones_add_remove_collision_exception(bool p_add, Node *p_node, RID p_exception) {
	for (int i = p_node->get_child_count() - 1; 0 <= i; --i) {
		_physical_bones_add_remove_collision_exception(p_add, p_node->get_child(i), p_exception);
	}

	CollisionObject *co = Object::cast_to<CollisionObject>(p_node);
	if (co) {
		if (p_add) {
			PhysicsServer::get_singleton()->body_add_collision_exception(co->get_rid(), p_exception);
		} else {
			PhysicsServer::get_singleton()->body_remove_collision_exception(co->get_rid(), p_exception);
		}
	}
}

void Skeleton::physical_bones_add_collision_exception(RID p_exception) {
	_physical_bones_add_remove_collision_exception(true, this, p_exception);
}

void Skeleton::physical_bones_remove_collision_exception(RID p_exception) {
	_physical_bones_add_remove_collision_exception(false, this, p_exception);
}

#endif // _3D_DISABLED

void Skeleton::_skin_changed() {
	_make_dirty();
}

Ref<SkinReference> Skeleton::register_skin(const Ref<Skin> &p_skin) {
	for (Set<SkinReference *>::Element *E = skin_bindings.front(); E; E = E->next()) {
		if (E->get()->skin == p_skin) {
			return Ref<SkinReference>(E->get());
		}
	}

	Ref<Skin> skin = p_skin;

	if (skin.is_null()) {
		//need to create one from existing code, this is for compatibility only
		//when skeletons did not support skins. It is also used by gizmo
		//to display the skeleton.

		skin.instance();
		skin->set_bind_count(bones.size());
		_update_process_order(); //just in case

		// pose changed, rebuild cache of inverses
		const Bone *bonesptr = bones.ptr();
		int len = bones.size();
		const int *order = process_order.ptr();

		// calculate global rests and invert them
		for (int i = 0; i < len; i++) {
			const Bone &b = bonesptr[order[i]];
			if (b.parent >= 0) {
				skin->set_bind_pose(order[i], skin->get_bind_pose(b.parent) * b.rest);
			} else {
				skin->set_bind_pose(order[i], b.rest);
			}
		}

		for (int i = 0; i < len; i++) {
			//the inverse is what is actually required
			skin->set_bind_bone(i, i);
			skin->set_bind_pose(i, skin->get_bind_pose(i).affine_inverse());
		}
	}

	ERR_FAIL_COND_V(skin.is_null(), Ref<SkinReference>());

	Ref<SkinReference> skin_ref;
	skin_ref.instance();

	skin_ref->skeleton_node = this;
	skin_ref->bind_count = 0;
	skin_ref->skeleton = RID_PRIME(VisualServer::get_singleton()->skeleton_create());
	skin_ref->skeleton_node = this;
	skin_ref->skin = skin;

	skin_bindings.insert(skin_ref.operator->());

	skin->connect("changed", skin_ref.operator->(), "_skin_changed");
	_make_dirty();
	return skin_ref;
}

void Skeleton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_bone", "name"), &Skeleton::add_bone);
	ClassDB::bind_method(D_METHOD("find_bone", "name"), &Skeleton::find_bone);
	ClassDB::bind_method(D_METHOD("get_bone_name", "bone_idx"), &Skeleton::get_bone_name);
	ClassDB::bind_method(D_METHOD("set_bone_name", "bone_idx", "name"), &Skeleton::set_bone_name);

	ClassDB::bind_method(D_METHOD("get_bone_parent", "bone_idx"), &Skeleton::get_bone_parent);
	ClassDB::bind_method(D_METHOD("set_bone_parent", "bone_idx", "parent_idx"), &Skeleton::set_bone_parent);

	ClassDB::bind_method(D_METHOD("get_bone_count"), &Skeleton::get_bone_count);

	ClassDB::bind_method(D_METHOD("unparent_bone_and_rest", "bone_idx"), &Skeleton::unparent_bone_and_rest);

	ClassDB::bind_method(D_METHOD("get_bone_rest", "bone_idx"), &Skeleton::get_bone_rest);
	ClassDB::bind_method(D_METHOD("set_bone_rest", "bone_idx", "rest"), &Skeleton::set_bone_rest);

	ClassDB::bind_method(D_METHOD("register_skin", "skin"), &Skeleton::register_skin);

	ClassDB::bind_method(D_METHOD("localize_rests"), &Skeleton::localize_rests);

	ClassDB::bind_method(D_METHOD("set_bone_disable_rest", "bone_idx", "disable"), &Skeleton::set_bone_disable_rest);
	ClassDB::bind_method(D_METHOD("is_bone_rest_disabled", "bone_idx"), &Skeleton::is_bone_rest_disabled);

	ClassDB::bind_method(D_METHOD("bind_child_node_to_bone", "bone_idx", "node"), &Skeleton::bind_child_node_to_bone);
	ClassDB::bind_method(D_METHOD("unbind_child_node_from_bone", "bone_idx", "node"), &Skeleton::unbind_child_node_from_bone);
	ClassDB::bind_method(D_METHOD("get_bound_child_nodes_to_bone", "bone_idx"), &Skeleton::_get_bound_child_nodes_to_bone);

	ClassDB::bind_method(D_METHOD("clear_bones"), &Skeleton::clear_bones);

	ClassDB::bind_method(D_METHOD("get_bone_pose", "bone_idx"), &Skeleton::get_bone_pose);
	ClassDB::bind_method(D_METHOD("set_bone_pose", "bone_idx", "pose"), &Skeleton::set_bone_pose);

	ClassDB::bind_method(D_METHOD("clear_bones_global_pose_override"), &Skeleton::clear_bones_global_pose_override);
	ClassDB::bind_method(D_METHOD("set_bone_global_pose_override", "bone_idx", "pose", "amount", "persistent"), &Skeleton::set_bone_global_pose_override, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_bone_global_pose", "bone_idx"), &Skeleton::get_bone_global_pose);
	ClassDB::bind_method(D_METHOD("get_bone_global_pose_no_override", "bone_idx"), &Skeleton::get_bone_global_pose_no_override);

	ClassDB::bind_method(D_METHOD("get_bone_custom_pose", "bone_idx"), &Skeleton::get_bone_custom_pose);
	ClassDB::bind_method(D_METHOD("set_bone_custom_pose", "bone_idx", "custom_pose"), &Skeleton::set_bone_custom_pose);

#ifndef _3D_DISABLED

	ClassDB::bind_method(D_METHOD("physical_bones_stop_simulation"), &Skeleton::physical_bones_stop_simulation);
	ClassDB::bind_method(D_METHOD("physical_bones_start_simulation", "bones"), &Skeleton::physical_bones_start_simulation_on, DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("physical_bones_add_collision_exception", "exception"), &Skeleton::physical_bones_add_collision_exception);
	ClassDB::bind_method(D_METHOD("physical_bones_remove_collision_exception", "exception"), &Skeleton::physical_bones_remove_collision_exception);

#endif // _3D_DISABLED

	ADD_SIGNAL(MethodInfo("skeleton_updated"));

	BIND_CONSTANT(NOTIFICATION_UPDATE_SKELETON);
}

Skeleton::Skeleton() {
	dirty = false;
	version = 1;
	process_order_dirty = true;
}

Skeleton::~Skeleton() {
	//some skins may remain bound
	for (Set<SkinReference *>::Element *E = skin_bindings.front(); E; E = E->next()) {
		E->get()->skeleton_node = nullptr;
	}
}

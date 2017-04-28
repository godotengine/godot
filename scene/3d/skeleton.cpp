/*************************************************************************/
/*  skeleton.cpp                                                         */
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
#include "skeleton.h"

#include "message_queue.h"

#include "core/global_config.h"
#include "scene/resources/surface_tool.h"

bool Skeleton::_set(const StringName &p_path, const Variant &p_value) {

	String path = p_path;

	if (!path.begins_with("bones/"))
		return false;

	int which = path.get_slicec('/', 1).to_int();
	String what = path.get_slicec('/', 2);

	if (which == bones.size() && what == "name") {

		add_bone(p_value);
		return true;
	}

	ERR_FAIL_INDEX_V(which, bones.size(), false);

	if (what == "parent")
		set_bone_parent(which, p_value);
	else if (what == "rest")
		set_bone_rest(which, p_value);
	else if (what == "enabled")
		set_bone_enabled(which, p_value);
	else if (what == "pose")
		set_bone_pose(which, p_value);
	else if (what == "bound_childs") {
		Array children = p_value;

		if (is_inside_tree()) {
			bones[which].nodes_bound.clear();

			for (int i = 0; i < children.size(); i++) {

				NodePath path = children[i];
				ERR_CONTINUE(path.operator String() == "");
				Node *node = get_node(path);
				ERR_CONTINUE(!node);
				bind_child_node_to_bone(which, node);
			}
		}
	} else {
		return false;
	}

	return true;
}

bool Skeleton::_get(const StringName &p_name, Variant &r_ret) const {

	String path = p_name;

	if (!path.begins_with("bones/"))
		return false;

	int which = path.get_slicec('/', 1).to_int();
	String what = path.get_slicec('/', 2);

	ERR_FAIL_INDEX_V(which, bones.size(), false);

	if (what == "name")
		r_ret = get_bone_name(which);
	else if (what == "parent")
		r_ret = get_bone_parent(which);
	else if (what == "rest")
		r_ret = get_bone_rest(which);
	else if (what == "enabled")
		r_ret = is_bone_enabled(which);
	else if (what == "pose")
		r_ret = get_bone_pose(which);
	else if (what == "bound_childs") {
		Array children;

		for (const List<uint32_t>::Element *E = bones[which].nodes_bound.front(); E; E = E->next()) {

			Object *obj = ObjectDB::get_instance(E->get());
			ERR_CONTINUE(!obj);
			Node *node = obj->cast_to<Node>();
			ERR_CONTINUE(!node);
			NodePath path = get_path_to(node);
			children.push_back(path);
		}

		r_ret = children;
	} else
		return false;

	return true;
}
void Skeleton::_get_property_list(List<PropertyInfo> *p_list) const {

	for (int i = 0; i < bones.size(); i++) {

		String prep = "bones/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::STRING, prep + "name"));
		p_list->push_back(PropertyInfo(Variant::INT, prep + "parent", PROPERTY_HINT_RANGE, "-1," + itos(i - 1) + ",1"));
		p_list->push_back(PropertyInfo(Variant::TRANSFORM, prep + "rest"));
		p_list->push_back(PropertyInfo(Variant::BOOL, prep + "enabled"));
		p_list->push_back(PropertyInfo(Variant::TRANSFORM, prep + "pose", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR));
		p_list->push_back(PropertyInfo(Variant::ARRAY, prep + "bound_childs"));
	}
}

void Skeleton::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_WORLD: {

			if (dirty) {

				dirty = false;
				_make_dirty(); // property make it dirty
			}

		} break;
		case NOTIFICATION_EXIT_WORLD: {

		} break;
		case NOTIFICATION_UPDATE_SKELETON: {

			VisualServer *vs = VisualServer::get_singleton();
			Bone *bonesptr = &bones[0];
			int len = bones.size();

			vs->skeleton_allocate(skeleton, len); // if same size, nothin really happens

			// pose changed, rebuild cache of inverses
			if (rest_global_inverse_dirty) {

				// calculate global rests and invert them
				for (int i = 0; i < len; i++) {
					Bone &b = bonesptr[i];
					if (b.parent >= 0)
						b.rest_global_inverse = bonesptr[b.parent].rest_global_inverse * b.rest;
					else
						b.rest_global_inverse = b.rest;
				}
				for (int i = 0; i < len; i++) {
					Bone &b = bonesptr[i];
					b.rest_global_inverse.affine_invert();
				}

				rest_global_inverse_dirty = false;
			}

			for (int i = 0; i < len; i++) {

				Bone &b = bonesptr[i];

				if (b.disable_rest) {
					if (b.enabled) {

						Transform pose = b.pose;
						if (b.custom_pose_enable) {

							pose = b.custom_pose * pose;
						}

						if (b.parent >= 0) {

							b.pose_global = bonesptr[b.parent].pose_global * pose;
						} else {

							b.pose_global = pose;
						}
					} else {

						if (b.parent >= 0) {

							b.pose_global = bonesptr[b.parent].pose_global;
						} else {

							b.pose_global = Transform();
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
						} else {

							b.pose_global = b.rest * pose;
						}
					} else {

						if (b.parent >= 0) {

							b.pose_global = bonesptr[b.parent].pose_global * b.rest;
						} else {

							b.pose_global = b.rest;
						}
					}
				}

				vs->skeleton_bone_set_transform(skeleton, i, b.pose_global * b.rest_global_inverse);

				for (List<uint32_t>::Element *E = b.nodes_bound.front(); E; E = E->next()) {

					Object *obj = ObjectDB::get_instance(E->get());
					ERR_CONTINUE(!obj);
					Spatial *sp = obj->cast_to<Spatial>();
					ERR_CONTINUE(!sp);
					sp->set_transform(b.pose_global);
				}
			}

			dirty = false;
		} break;
	}
}

Transform Skeleton::get_bone_transform(int p_bone) const {
	ERR_FAIL_INDEX_V(p_bone, bones.size(), Transform());
	if (dirty)
		const_cast<Skeleton *>(this)->notification(NOTIFICATION_UPDATE_SKELETON);
	return bones[p_bone].pose_global * bones[p_bone].rest_global_inverse;
}

void Skeleton::set_bone_global_pose(int p_bone, const Transform &p_pose) {

	ERR_FAIL_INDEX(p_bone, bones.size());
	if (bones[p_bone].parent == -1) {

		set_bone_pose(p_bone, bones[p_bone].rest_global_inverse * p_pose); //fast
	} else {

		set_bone_pose(p_bone, bones[p_bone].rest.affine_inverse() * (get_bone_global_pose(bones[p_bone].parent).affine_inverse() * p_pose)); //slow
	}
}

Transform Skeleton::get_bone_global_pose(int p_bone) const {

	ERR_FAIL_INDEX_V(p_bone, bones.size(), Transform());
	if (dirty)
		const_cast<Skeleton *>(this)->notification(NOTIFICATION_UPDATE_SKELETON);
	return bones[p_bone].pose_global;
}

RID Skeleton::get_skeleton() const {

	return skeleton;
}

// skeleton creation api
void Skeleton::add_bone(const String &p_name) {

	ERR_FAIL_COND(p_name == "" || p_name.find(":") != -1 || p_name.find("/") != -1);

	for (int i = 0; i < bones.size(); i++) {

		ERR_FAIL_COND(bones[i].name == "p_name");
	}

	Bone b;
	b.name = p_name;
	bones.push_back(b);

	rest_global_inverse_dirty = true;
	_make_dirty();
	update_gizmo();
}
int Skeleton::find_bone(String p_name) const {

	for (int i = 0; i < bones.size(); i++) {

		if (bones[i].name == p_name)
			return i;
	}

	return -1;
}
String Skeleton::get_bone_name(int p_bone) const {

	ERR_FAIL_INDEX_V(p_bone, bones.size(), "");

	return bones[p_bone].name;
}

int Skeleton::get_bone_count() const {

	return bones.size();
}

void Skeleton::set_bone_parent(int p_bone, int p_parent) {

	ERR_FAIL_INDEX(p_bone, bones.size());
	ERR_FAIL_COND(p_parent != -1 && (p_parent < 0 || p_parent >= p_bone));

	bones[p_bone].parent = p_parent;
	rest_global_inverse_dirty = true;
	_make_dirty();
}

void Skeleton::unparent_bone_and_rest(int p_bone) {

	ERR_FAIL_INDEX(p_bone, bones.size());

	int parent = bones[p_bone].parent;
	while (parent >= 0) {
		bones[p_bone].rest = bones[parent].rest * bones[p_bone].rest;
		parent = bones[parent].parent;
	}

	bones[p_bone].parent = -1;
	bones[p_bone].rest_global_inverse = bones[p_bone].rest.affine_inverse(); //same thing

	_make_dirty();
}

void Skeleton::set_bone_disable_rest(int p_bone, bool p_disable) {

	ERR_FAIL_INDEX(p_bone, bones.size());
	bones[p_bone].disable_rest = p_disable;
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

	bones[p_bone].rest = p_rest;
	rest_global_inverse_dirty = true;
	_make_dirty();
}
Transform Skeleton::get_bone_rest(int p_bone) const {

	ERR_FAIL_INDEX_V(p_bone, bones.size(), Transform());

	return bones[p_bone].rest;
}

void Skeleton::set_bone_enabled(int p_bone, bool p_enabled) {

	ERR_FAIL_INDEX(p_bone, bones.size());

	bones[p_bone].enabled = p_enabled;
	rest_global_inverse_dirty = true;
	_make_dirty();
}
bool Skeleton::is_bone_enabled(int p_bone) const {

	ERR_FAIL_INDEX_V(p_bone, bones.size(), false);
	return bones[p_bone].enabled;
}

void Skeleton::bind_child_node_to_bone(int p_bone, Node *p_node) {

	ERR_FAIL_NULL(p_node);
	ERR_FAIL_INDEX(p_bone, bones.size());

	uint32_t id = p_node->get_instance_ID();

	for (List<uint32_t>::Element *E = bones[p_bone].nodes_bound.front(); E; E = E->next()) {

		if (E->get() == id)
			return; // already here
	}

	bones[p_bone].nodes_bound.push_back(id);
}
void Skeleton::unbind_child_node_from_bone(int p_bone, Node *p_node) {

	ERR_FAIL_NULL(p_node);
	ERR_FAIL_INDEX(p_bone, bones.size());

	uint32_t id = p_node->get_instance_ID();
	bones[p_bone].nodes_bound.erase(id);
}
void Skeleton::get_bound_child_nodes_to_bone(int p_bone, List<Node *> *p_bound) const {

	ERR_FAIL_INDEX(p_bone, bones.size());

	for (const List<uint32_t>::Element *E = bones[p_bone].nodes_bound.front(); E; E = E->next()) {

		Object *obj = ObjectDB::get_instance(E->get());
		ERR_CONTINUE(!obj);
		p_bound->push_back(obj->cast_to<Node>());
	}
}

void Skeleton::clear_bones() {

	bones.clear();
	rest_global_inverse_dirty = true;
	_make_dirty();
}

// posing api

void Skeleton::set_bone_pose(int p_bone, const Transform &p_pose) {

	ERR_FAIL_INDEX(p_bone, bones.size());
	ERR_FAIL_COND(!is_inside_tree());

	bones[p_bone].pose = p_pose;
	_make_dirty();
}
Transform Skeleton::get_bone_pose(int p_bone) const {

	ERR_FAIL_INDEX_V(p_bone, bones.size(), Transform());
	return bones[p_bone].pose;
}

void Skeleton::set_bone_custom_pose(int p_bone, const Transform &p_custom_pose) {

	ERR_FAIL_INDEX(p_bone, bones.size());
	//ERR_FAIL_COND( !is_inside_scene() );

	bones[p_bone].custom_pose_enable = (p_custom_pose != Transform());
	bones[p_bone].custom_pose = p_custom_pose;

	_make_dirty();
}

Transform Skeleton::get_bone_custom_pose(int p_bone) const {

	ERR_FAIL_INDEX_V(p_bone, bones.size(), Transform());
	return bones[p_bone].custom_pose;
}

void Skeleton::_make_dirty() {

	if (dirty)
		return;

	if (!is_inside_tree()) {
		dirty = true;
		return;
	}
	MessageQueue::get_singleton()->push_notification(this, NOTIFICATION_UPDATE_SKELETON);
	dirty = true;
}

void Skeleton::localize_rests() {

	for (int i = bones.size() - 1; i >= 0; i--) {

		if (bones[i].parent >= 0)
			set_bone_rest(i, bones[bones[i].parent].rest.affine_inverse() * bones[i].rest);
	}
}

void Skeleton::_bind_methods() {

	ClassDB::bind_method(D_METHOD("add_bone", "name"), &Skeleton::add_bone);
	ClassDB::bind_method(D_METHOD("find_bone", "name"), &Skeleton::find_bone);
	ClassDB::bind_method(D_METHOD("get_bone_name", "bone_idx"), &Skeleton::get_bone_name);

	ClassDB::bind_method(D_METHOD("get_bone_parent", "bone_idx"), &Skeleton::get_bone_parent);
	ClassDB::bind_method(D_METHOD("set_bone_parent", "bone_idx", "parent_idx"), &Skeleton::set_bone_parent);

	ClassDB::bind_method(D_METHOD("get_bone_count"), &Skeleton::get_bone_count);

	ClassDB::bind_method(D_METHOD("unparent_bone_and_rest", "bone_idx"), &Skeleton::unparent_bone_and_rest);

	ClassDB::bind_method(D_METHOD("get_bone_rest", "bone_idx"), &Skeleton::get_bone_rest);
	ClassDB::bind_method(D_METHOD("set_bone_rest", "bone_idx", "rest"), &Skeleton::set_bone_rest);

	ClassDB::bind_method(D_METHOD("set_bone_disable_rest", "bone_idx", "disable"), &Skeleton::set_bone_disable_rest);
	ClassDB::bind_method(D_METHOD("is_bone_rest_disabled", "bone_idx"), &Skeleton::is_bone_rest_disabled);

	ClassDB::bind_method(D_METHOD("bind_child_node_to_bone", "bone_idx", "node:Node"), &Skeleton::bind_child_node_to_bone);
	ClassDB::bind_method(D_METHOD("unbind_child_node_from_bone", "bone_idx", "node:Node"), &Skeleton::unbind_child_node_from_bone);
	ClassDB::bind_method(D_METHOD("get_bound_child_nodes_to_bone", "bone_idx"), &Skeleton::_get_bound_child_nodes_to_bone);

	ClassDB::bind_method(D_METHOD("clear_bones"), &Skeleton::clear_bones);

	ClassDB::bind_method(D_METHOD("get_bone_pose", "bone_idx"), &Skeleton::get_bone_pose);
	ClassDB::bind_method(D_METHOD("set_bone_pose", "bone_idx", "pose"), &Skeleton::set_bone_pose);

	ClassDB::bind_method(D_METHOD("set_bone_global_pose", "bone_idx", "pose"), &Skeleton::set_bone_global_pose);
	ClassDB::bind_method(D_METHOD("get_bone_global_pose", "bone_idx"), &Skeleton::get_bone_global_pose);

	ClassDB::bind_method(D_METHOD("get_bone_custom_pose", "bone_idx"), &Skeleton::get_bone_custom_pose);
	ClassDB::bind_method(D_METHOD("set_bone_custom_pose", "bone_idx", "custom_pose"), &Skeleton::set_bone_custom_pose);

	ClassDB::bind_method(D_METHOD("get_bone_transform", "bone_idx"), &Skeleton::get_bone_transform);

	BIND_CONSTANT(NOTIFICATION_UPDATE_SKELETON);
}

Skeleton::Skeleton() {

	rest_global_inverse_dirty = true;
	dirty = false;
	skeleton = VisualServer::get_singleton()->skeleton_create();
}

Skeleton::~Skeleton() {

	VisualServer::get_singleton()->free(skeleton);
}

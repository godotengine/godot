/**************************************************************************/
/*  skeleton_3d.cpp                                                       */
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

#include "skeleton_3d.h"

#include "core/object/message_queue.h"
#include "core/variant/type_info.h"
#include "scene/3d/physics_body_3d.h"
#include "scene/resources/surface_tool.h"
#include "scene/scene_string_names.h"

void SkinReference::_skin_changed() {
	if (skeleton_node) {
		skeleton_node->_make_dirty();
	}
	skeleton_version = 0;
}

void SkinReference::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_skeleton"), &SkinReference::get_skeleton);
	ClassDB::bind_method(D_METHOD("get_skin"), &SkinReference::get_skin);
}

RID SkinReference::get_skeleton() const {
	return skeleton;
}

Ref<Skin> SkinReference::get_skin() const {
	return skin;
}

SkinReference::~SkinReference() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	if (skeleton_node) {
		skeleton_node->skin_bindings.erase(this);
	}
	RS::get_singleton()->free(skeleton);
}

///////////////////////////////////////

bool Skeleton3D::_set(const StringName &p_path, const Variant &p_value) {
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
	} else if (what == "position") {
		set_bone_pose_position(which, p_value);
	} else if (what == "rotation") {
		set_bone_pose_rotation(which, p_value);
	} else if (what == "scale") {
		set_bone_pose_scale(which, p_value);
	} else {
		return false;
	}

	return true;
}

bool Skeleton3D::_get(const StringName &p_path, Variant &r_ret) const {
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
	} else if (what == "position") {
		r_ret = get_bone_pose_position(which);
	} else if (what == "rotation") {
		r_ret = get_bone_pose_rotation(which);
	} else if (what == "scale") {
		r_ret = get_bone_pose_scale(which);
	} else {
		return false;
	}

	return true;
}

void Skeleton3D::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < bones.size(); i++) {
		const String prep = vformat("%s/%d/", PNAME("bones"), i);
		p_list->push_back(PropertyInfo(Variant::STRING, prep + PNAME("name"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::INT, prep + PNAME("parent"), PROPERTY_HINT_RANGE, "-1," + itos(bones.size() - 1) + ",1", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::TRANSFORM3D, prep + PNAME("rest"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::BOOL, prep + PNAME("enabled"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::VECTOR3, prep + PNAME("position"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::QUATERNION, prep + PNAME("rotation"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::VECTOR3, prep + PNAME("scale"), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	}

	for (PropertyInfo &E : *p_list) {
		_validate_property(E);
	}
}

void Skeleton3D::_validate_property(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() == 3 && split[0] == "bones") {
		if (split[2] == "rest") {
			p_property.usage |= PROPERTY_USAGE_READ_ONLY;
		}
		if (is_show_rest_only()) {
			if (split[2] == "enabled") {
				p_property.usage |= PROPERTY_USAGE_READ_ONLY;
			}
			if (split[2] == "position") {
				p_property.usage |= PROPERTY_USAGE_READ_ONLY;
			}
			if (split[2] == "rotation") {
				p_property.usage |= PROPERTY_USAGE_READ_ONLY;
			}
			if (split[2] == "scale") {
				p_property.usage |= PROPERTY_USAGE_READ_ONLY;
			}
		} else if (!is_bone_enabled(split[1].to_int())) {
			if (split[2] == "position") {
				p_property.usage |= PROPERTY_USAGE_READ_ONLY;
			}
			if (split[2] == "rotation") {
				p_property.usage |= PROPERTY_USAGE_READ_ONLY;
			}
			if (split[2] == "scale") {
				p_property.usage |= PROPERTY_USAGE_READ_ONLY;
			}
		}
	}
}

void Skeleton3D::_update_process_order() {
	if (!process_order_dirty) {
		return;
	}

	Bone *bonesptr = bones.ptrw();
	int len = bones.size();

	parentless_bones.clear();

	for (int i = 0; i < len; i++) {
		bonesptr[i].child_bones.clear();
	}

	for (int i = 0; i < len; i++) {
		if (bonesptr[i].parent >= len) {
			// Validate this just in case.
			ERR_PRINT("Bone " + itos(i) + " has invalid parent: " + itos(bonesptr[i].parent));
			bonesptr[i].parent = -1;
		}

		if (bonesptr[i].parent != -1) {
			int parent_bone_idx = bonesptr[i].parent;

			// Check to see if this node is already added to the parent.
			if (bonesptr[parent_bone_idx].child_bones.find(i) < 0) {
				// Add the child node.
				bonesptr[parent_bone_idx].child_bones.push_back(i);
			} else {
				ERR_PRINT("Skeleton3D parenthood graph is cyclic");
			}
		} else {
			parentless_bones.push_back(i);
		}
	}

	process_order_dirty = false;
}

void Skeleton3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (dirty) {
				notification(NOTIFICATION_UPDATE_SKELETON);
			}
		} break;
		case NOTIFICATION_UPDATE_SKELETON: {
			RenderingServer *rs = RenderingServer::get_singleton();
			Bone *bonesptr = bones.ptrw();

			int len = bones.size();
			dirty = false;

			// Update bone transforms.
			force_update_all_bone_transforms();

			// Update skins.
			for (SkinReference *E : skin_bindings) {
				const Skin *skin = E->skin.operator->();
				RID skeleton = E->skeleton;
				uint32_t bind_count = skin->get_bind_count();

				if (E->bind_count != bind_count) {
					RS::get_singleton()->skeleton_allocate_data(skeleton, bind_count);
					E->bind_count = bind_count;
					E->skin_bone_indices.resize(bind_count);
					E->skin_bone_indices_ptrs = E->skin_bone_indices.ptrw();
				}

				if (E->skeleton_version != version) {
					for (uint32_t i = 0; i < bind_count; i++) {
						StringName bind_name = skin->get_bind_name(i);

						if (bind_name != StringName()) {
							// Bind name used, use this.
							bool found = false;
							for (int j = 0; j < len; j++) {
								if (bonesptr[j].name == bind_name) {
									E->skin_bone_indices_ptrs[i] = j;
									found = true;
									break;
								}
							}

							if (!found) {
								ERR_PRINT("Skin bind #" + itos(i) + " contains named bind '" + String(bind_name) + "' but Skeleton3D has no bone by that name.");
								E->skin_bone_indices_ptrs[i] = 0;
							}
						} else if (skin->get_bind_bone(i) >= 0) {
							int bind_index = skin->get_bind_bone(i);
							if (bind_index >= len) {
								ERR_PRINT("Skin bind #" + itos(i) + " contains bone index bind: " + itos(bind_index) + " , which is greater than the skeleton bone count: " + itos(len) + ".");
								E->skin_bone_indices_ptrs[i] = 0;
							} else {
								E->skin_bone_indices_ptrs[i] = bind_index;
							}
						} else {
							ERR_PRINT("Skin bind #" + itos(i) + " does not contain a name nor a bone index.");
							E->skin_bone_indices_ptrs[i] = 0;
						}
					}

					E->skeleton_version = version;
				}

				for (uint32_t i = 0; i < bind_count; i++) {
					uint32_t bone_index = E->skin_bone_indices_ptrs[i];
					ERR_CONTINUE(bone_index >= (uint32_t)len);
					rs->skeleton_bone_set_transform(skeleton, i, bonesptr[bone_index].pose_global * skin->get_bind_pose(i));
				}
			}
			emit_signal(SceneStringNames::get_singleton()->pose_updated);
		} break;

#ifndef _3D_DISABLED
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			// This is active only if the skeleton animates the physical bones
			// and the state of the bone is not active.
			if (animate_physical_bones) {
				for (int i = 0; i < bones.size(); i += 1) {
					if (bones[i].physical_bone) {
						if (bones[i].physical_bone->is_simulating_physics() == false) {
							bones[i].physical_bone->reset_to_rest_position();
						}
					}
				}
			}
		} break;
		case NOTIFICATION_READY: {
			if (Engine::get_singleton()->is_editor_hint()) {
				set_physics_process_internal(true);
			}
		} break;
#endif // _3D_DISABLED
	}
}

void Skeleton3D::clear_bones_global_pose_override() {
	for (int i = 0; i < bones.size(); i += 1) {
		bones.write[i].global_pose_override_amount = 0;
		bones.write[i].global_pose_override_reset = true;
	}
	_make_dirty();
}

void Skeleton3D::set_bone_global_pose_override(int p_bone, const Transform3D &p_pose, real_t p_amount, bool p_persistent) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone, bone_size);
	bones.write[p_bone].global_pose_override_amount = p_amount;
	bones.write[p_bone].global_pose_override = p_pose;
	bones.write[p_bone].global_pose_override_reset = !p_persistent;
	_make_dirty();
}

Transform3D Skeleton3D::get_bone_global_pose_override(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, Transform3D());
	return bones[p_bone].global_pose_override;
}

Transform3D Skeleton3D::get_bone_global_pose(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, Transform3D());
	if (dirty) {
		const_cast<Skeleton3D *>(this)->notification(NOTIFICATION_UPDATE_SKELETON);
	}
	return bones[p_bone].pose_global;
}

Transform3D Skeleton3D::get_bone_global_pose_no_override(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, Transform3D());
	if (dirty) {
		const_cast<Skeleton3D *>(this)->notification(NOTIFICATION_UPDATE_SKELETON);
	}
	return bones[p_bone].pose_global_no_override;
}

void Skeleton3D::set_motion_scale(float p_motion_scale) {
	if (p_motion_scale <= 0) {
		motion_scale = 1;
		ERR_FAIL_MSG("Motion scale must be larger than 0.");
	}
	motion_scale = p_motion_scale;
}

float Skeleton3D::get_motion_scale() const {
	ERR_FAIL_COND_V(motion_scale <= 0, 1);
	return motion_scale;
}

// Skeleton creation api

uint64_t Skeleton3D::get_version() const {
	return version;
}

void Skeleton3D::add_bone(const String &p_name) {
	ERR_FAIL_COND(p_name.is_empty() || p_name.contains(":") || p_name.contains("/") || name_to_bone_index.has(p_name));

	Bone b;
	b.name = p_name;
	bones.push_back(b);
	name_to_bone_index.insert(p_name, bones.size() - 1);
	process_order_dirty = true;
	version++;
	rest_dirty = true;
	_make_dirty();
	update_gizmos();
}

int Skeleton3D::find_bone(const String &p_name) const {
	const int *bone_index_ptr = name_to_bone_index.getptr(p_name);
	return bone_index_ptr != nullptr ? *bone_index_ptr : -1;
}

String Skeleton3D::get_bone_name(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, "");
	return bones[p_bone].name;
}

void Skeleton3D::set_bone_name(int p_bone, const String &p_name) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone, bone_size);

	const int *bone_index_ptr = name_to_bone_index.getptr(p_name);
	if (bone_index_ptr != nullptr) {
		ERR_FAIL_COND_MSG(*bone_index_ptr != p_bone, "Skeleton3D: '" + get_name() + "', bone name:  '" + p_name + "' already exists.");
		return; // No need to rename, the bone already has the given name.
	}

	name_to_bone_index.erase(bones[p_bone].name);
	bones.write[p_bone].name = p_name;
	name_to_bone_index.insert(p_name, p_bone);

	version++;
}

bool Skeleton3D::is_bone_parent_of(int p_bone, int p_parent_bone_id) const {
	int parent_of_bone = get_bone_parent(p_bone);

	if (-1 == parent_of_bone) {
		return false;
	}

	if (parent_of_bone == p_parent_bone_id) {
		return true;
	}

	return is_bone_parent_of(parent_of_bone, p_parent_bone_id);
}

int Skeleton3D::get_bone_count() const {
	return bones.size();
}

void Skeleton3D::set_bone_parent(int p_bone, int p_parent) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone, bone_size);
	ERR_FAIL_COND(p_parent != -1 && (p_parent < 0));
	ERR_FAIL_COND(p_bone == p_parent);

	bones.write[p_bone].parent = p_parent;
	process_order_dirty = true;
	rest_dirty = true;
	_make_dirty();
}

void Skeleton3D::unparent_bone_and_rest(int p_bone) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone, bone_size);

	_update_process_order();

	int parent = bones[p_bone].parent;
	while (parent >= 0) {
		bones.write[p_bone].rest = bones[parent].rest * bones[p_bone].rest;
		parent = bones[parent].parent;
	}

	bones.write[p_bone].parent = -1;
	process_order_dirty = true;

	rest_dirty = true;
	_make_dirty();
}

int Skeleton3D::get_bone_parent(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, -1);
	if (process_order_dirty) {
		const_cast<Skeleton3D *>(this)->_update_process_order();
	}
	return bones[p_bone].parent;
}

Vector<int> Skeleton3D::get_bone_children(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, Vector<int>());
	if (process_order_dirty) {
		const_cast<Skeleton3D *>(this)->_update_process_order();
	}
	return bones[p_bone].child_bones;
}

Vector<int> Skeleton3D::get_parentless_bones() const {
	if (process_order_dirty) {
		const_cast<Skeleton3D *>(this)->_update_process_order();
	}
	return parentless_bones;
}

void Skeleton3D::set_bone_rest(int p_bone, const Transform3D &p_rest) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone, bone_size);

	bones.write[p_bone].rest = p_rest;
	rest_dirty = true;
	_make_dirty();
}
Transform3D Skeleton3D::get_bone_rest(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, Transform3D());

	return bones[p_bone].rest;
}
Transform3D Skeleton3D::get_bone_global_rest(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, Transform3D());
	if (rest_dirty) {
		const_cast<Skeleton3D *>(this)->notification(NOTIFICATION_UPDATE_SKELETON);
	}
	return bones[p_bone].global_rest;
}

void Skeleton3D::set_bone_enabled(int p_bone, bool p_enabled) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone, bone_size);

	bones.write[p_bone].enabled = p_enabled;
	emit_signal(SceneStringNames::get_singleton()->bone_enabled_changed, p_bone);
	_make_dirty();
}

bool Skeleton3D::is_bone_enabled(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, false);
	return bones[p_bone].enabled;
}

void Skeleton3D::set_show_rest_only(bool p_enabled) {
	show_rest_only = p_enabled;
	emit_signal(SceneStringNames::get_singleton()->show_rest_only_changed);
	_make_dirty();
}

bool Skeleton3D::is_show_rest_only() const {
	return show_rest_only;
}

void Skeleton3D::clear_bones() {
	bones.clear();
	name_to_bone_index.clear();
	process_order_dirty = true;
	version++;
	_make_dirty();
}

// Posing api

void Skeleton3D::set_bone_pose_position(int p_bone, const Vector3 &p_position) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone, bone_size);

	bones.write[p_bone].pose_position = p_position;
	bones.write[p_bone].pose_cache_dirty = true;
	if (is_inside_tree()) {
		_make_dirty();
	}
}
void Skeleton3D::set_bone_pose_rotation(int p_bone, const Quaternion &p_rotation) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone, bone_size);

	bones.write[p_bone].pose_rotation = p_rotation;
	bones.write[p_bone].pose_cache_dirty = true;
	if (is_inside_tree()) {
		_make_dirty();
	}
}
void Skeleton3D::set_bone_pose_scale(int p_bone, const Vector3 &p_scale) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone, bone_size);

	bones.write[p_bone].pose_scale = p_scale;
	bones.write[p_bone].pose_cache_dirty = true;
	if (is_inside_tree()) {
		_make_dirty();
	}
}

Vector3 Skeleton3D::get_bone_pose_position(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, Vector3());
	return bones[p_bone].pose_position;
}

Quaternion Skeleton3D::get_bone_pose_rotation(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, Quaternion());
	return bones[p_bone].pose_rotation;
}

Vector3 Skeleton3D::get_bone_pose_scale(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, Vector3());
	return bones[p_bone].pose_scale;
}

void Skeleton3D::reset_bone_pose(int p_bone) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone, bone_size);
	set_bone_pose_position(p_bone, bones[p_bone].rest.origin);
	set_bone_pose_rotation(p_bone, bones[p_bone].rest.basis.get_rotation_quaternion());
	set_bone_pose_scale(p_bone, bones[p_bone].rest.basis.get_scale());
}

void Skeleton3D::reset_bone_poses() {
	for (int i = 0; i < bones.size(); i++) {
		reset_bone_pose(i);
	}
}

Transform3D Skeleton3D::get_bone_pose(int p_bone) const {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, Transform3D());
	((Skeleton3D *)this)->bones.write[p_bone].update_pose_cache();
	return bones[p_bone].pose_cache;
}

void Skeleton3D::_make_dirty() {
	if (dirty) {
		return;
	}

	if (is_inside_tree()) {
		notify_deferred_thread_group(NOTIFICATION_UPDATE_SKELETON);
	}
	dirty = true;
}

void Skeleton3D::localize_rests() {
	Vector<int> bones_to_process = get_parentless_bones();
	while (bones_to_process.size() > 0) {
		int current_bone_idx = bones_to_process[0];
		bones_to_process.erase(current_bone_idx);

		if (bones[current_bone_idx].parent >= 0) {
			set_bone_rest(current_bone_idx, bones[bones[current_bone_idx].parent].rest.affine_inverse() * bones[current_bone_idx].rest);
		}

		// Add the bone's children to the list of bones to be processed.
		int child_bone_size = bones[current_bone_idx].child_bones.size();
		for (int i = 0; i < child_bone_size; i++) {
			bones_to_process.push_back(bones[current_bone_idx].child_bones[i]);
		}
	}
}

void Skeleton3D::set_animate_physical_bones(bool p_enabled) {
	animate_physical_bones = p_enabled;

	if (Engine::get_singleton()->is_editor_hint() == false) {
		bool sim = false;
		for (int i = 0; i < bones.size(); i += 1) {
			if (bones[i].physical_bone) {
				bones[i].physical_bone->reset_physics_simulation_state();
				if (bones[i].physical_bone->is_simulating_physics()) {
					sim = true;
				}
			}
		}
		set_physics_process_internal(sim == false && p_enabled);
	}
}

bool Skeleton3D::get_animate_physical_bones() const {
	return animate_physical_bones;
}

void Skeleton3D::bind_physical_bone_to_bone(int p_bone, PhysicalBone3D *p_physical_bone) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone, bone_size);
	ERR_FAIL_COND(bones[p_bone].physical_bone);
	ERR_FAIL_NULL(p_physical_bone);
	bones.write[p_bone].physical_bone = p_physical_bone;

	_rebuild_physical_bones_cache();
}

void Skeleton3D::unbind_physical_bone_from_bone(int p_bone) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone, bone_size);
	bones.write[p_bone].physical_bone = nullptr;

	_rebuild_physical_bones_cache();
}

PhysicalBone3D *Skeleton3D::get_physical_bone(int p_bone) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, nullptr);

	return bones[p_bone].physical_bone;
}

PhysicalBone3D *Skeleton3D::get_physical_bone_parent(int p_bone) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, nullptr);

	if (bones[p_bone].cache_parent_physical_bone) {
		return bones[p_bone].cache_parent_physical_bone;
	}

	return _get_physical_bone_parent(p_bone);
}

PhysicalBone3D *Skeleton3D::_get_physical_bone_parent(int p_bone) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX_V(p_bone, bone_size, nullptr);

	const int parent_bone = bones[p_bone].parent;
	if (0 > parent_bone) {
		return nullptr;
	}

	PhysicalBone3D *pb = bones[parent_bone].physical_bone;
	if (pb) {
		return pb;
	} else {
		return get_physical_bone_parent(parent_bone);
	}
}

void Skeleton3D::_rebuild_physical_bones_cache() {
	const int b_size = bones.size();
	for (int i = 0; i < b_size; ++i) {
		PhysicalBone3D *parent_pb = _get_physical_bone_parent(i);
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

	PhysicalBone3D *pb = Object::cast_to<PhysicalBone3D>(p_node);
	if (pb) {
		pb->set_simulate_physics(false);
	}
}

void Skeleton3D::physical_bones_stop_simulation() {
	_pb_stop_simulation(this);
	if (Engine::get_singleton()->is_editor_hint() == false && animate_physical_bones) {
		set_physics_process_internal(true);
	}
}

void _pb_start_simulation(const Skeleton3D *p_skeleton, Node *p_node, const Vector<int> &p_sim_bones) {
	for (int i = p_node->get_child_count() - 1; 0 <= i; --i) {
		_pb_start_simulation(p_skeleton, p_node->get_child(i), p_sim_bones);
	}

	PhysicalBone3D *pb = Object::cast_to<PhysicalBone3D>(p_node);
	if (pb) {
		if (p_sim_bones.is_empty()) { // If no bones is specified, activate ragdoll on full body.
			pb->set_simulate_physics(true);
		} else {
			for (int i = p_sim_bones.size() - 1; 0 <= i; --i) {
				if (p_sim_bones[i] == pb->get_bone_id() || p_skeleton->is_bone_parent_of(pb->get_bone_id(), p_sim_bones[i])) {
					pb->set_simulate_physics(true);
					break;
				}
			}
		}
	}
}

void Skeleton3D::physical_bones_start_simulation_on(const TypedArray<StringName> &p_bones) {
	set_physics_process_internal(false);

	Vector<int> sim_bones;
	if (p_bones.size() > 0) {
		sim_bones.resize(p_bones.size());
		int c = 0;
		for (int i = sim_bones.size() - 1; 0 <= i; --i) {
			int bone_id = find_bone(p_bones[i]);
			if (bone_id != -1) {
				sim_bones.write[c++] = bone_id;
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

	CollisionObject3D *co = Object::cast_to<CollisionObject3D>(p_node);
	if (co) {
		if (p_add) {
			PhysicsServer3D::get_singleton()->body_add_collision_exception(co->get_rid(), p_exception);
		} else {
			PhysicsServer3D::get_singleton()->body_remove_collision_exception(co->get_rid(), p_exception);
		}
	}
}

void Skeleton3D::physical_bones_add_collision_exception(RID p_exception) {
	_physical_bones_add_remove_collision_exception(true, this, p_exception);
}

void Skeleton3D::physical_bones_remove_collision_exception(RID p_exception) {
	_physical_bones_add_remove_collision_exception(false, this, p_exception);
}

void Skeleton3D::_skin_changed() {
	_make_dirty();
}

Ref<Skin> Skeleton3D::create_skin_from_rest_transforms() {
	Ref<Skin> skin;

	skin.instantiate();
	skin->set_bind_count(bones.size());

	// Pose changed, rebuild cache of inverses.
	const Bone *bonesptr = bones.ptr();
	int len = bones.size();

	// Calculate global rests and invert them.
	LocalVector<int> bones_to_process;
	bones_to_process = get_parentless_bones();
	while (bones_to_process.size() > 0) {
		int current_bone_idx = bones_to_process[0];
		const Bone &b = bonesptr[current_bone_idx];
		bones_to_process.erase(current_bone_idx);
		LocalVector<int> child_bones_vector;
		child_bones_vector = get_bone_children(current_bone_idx);
		int child_bones_size = child_bones_vector.size();
		if (b.parent < 0) {
			skin->set_bind_pose(current_bone_idx, b.rest);
		}
		for (int i = 0; i < child_bones_size; i++) {
			int child_bone_idx = child_bones_vector[i];
			const Bone &cb = bonesptr[child_bone_idx];
			skin->set_bind_pose(child_bone_idx, skin->get_bind_pose(current_bone_idx) * cb.rest);
			// Add the bone's children to the list of bones to be processed.
			bones_to_process.push_back(child_bones_vector[i]);
		}
	}

	for (int i = 0; i < len; i++) {
		// The inverse is what is actually required.
		skin->set_bind_bone(i, i);
		skin->set_bind_pose(i, skin->get_bind_pose(i).affine_inverse());
	}

	return skin;
}

Ref<SkinReference> Skeleton3D::register_skin(const Ref<Skin> &p_skin) {
	ERR_FAIL_COND_V(p_skin.is_null(), Ref<SkinReference>());

	for (const SkinReference *E : skin_bindings) {
		if (E->skin == p_skin) {
			return Ref<SkinReference>(E);
		}
	}

	Ref<SkinReference> skin_ref;
	skin_ref.instantiate();

	skin_ref->skeleton_node = this;
	skin_ref->bind_count = 0;
	skin_ref->skeleton = RenderingServer::get_singleton()->skeleton_create();
	skin_ref->skeleton_node = this;
	skin_ref->skin = p_skin;

	skin_bindings.insert(skin_ref.operator->());

	skin_ref->skin->connect_changed(callable_mp(skin_ref.operator->(), &SkinReference::_skin_changed));

	_make_dirty(); // Skin needs to be updated, so update skeleton.

	return skin_ref;
}

void Skeleton3D::force_update_all_dirty_bones() {
	if (dirty) {
		const_cast<Skeleton3D *>(this)->notification(NOTIFICATION_UPDATE_SKELETON);
	}
}

void Skeleton3D::force_update_all_bone_transforms() {
	_update_process_order();

	for (int i = 0; i < parentless_bones.size(); i++) {
		force_update_bone_children_transforms(parentless_bones[i]);
	}
	rest_dirty = false;
}

void Skeleton3D::force_update_bone_children_transforms(int p_bone_idx) {
	const int bone_size = bones.size();
	ERR_FAIL_INDEX(p_bone_idx, bone_size);

	Bone *bonesptr = bones.ptrw();
	List<int> bones_to_process = List<int>();
	bones_to_process.push_back(p_bone_idx);

	while (bones_to_process.size() > 0) {
		int current_bone_idx = bones_to_process[0];
		bones_to_process.erase(current_bone_idx);

		Bone &b = bonesptr[current_bone_idx];
		bool bone_enabled = b.enabled && !show_rest_only;

		if (bone_enabled) {
			b.update_pose_cache();
			Transform3D pose = b.pose_cache;

			if (b.parent >= 0) {
				b.pose_global = bonesptr[b.parent].pose_global * pose;
				b.pose_global_no_override = bonesptr[b.parent].pose_global_no_override * pose;
			} else {
				b.pose_global = pose;
				b.pose_global_no_override = pose;
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
		if (rest_dirty) {
			b.global_rest = b.parent >= 0 ? bonesptr[b.parent].global_rest * b.rest : b.rest;
		}

		if (b.global_pose_override_amount >= CMP_EPSILON) {
			b.pose_global = b.pose_global.interpolate_with(b.global_pose_override, b.global_pose_override_amount);
		}

		if (b.global_pose_override_reset) {
			b.global_pose_override_amount = 0.0;
		}

		// Add the bone's children to the list of bones to be processed.
		int child_bone_size = b.child_bones.size();
		for (int i = 0; i < child_bone_size; i++) {
			bones_to_process.push_back(b.child_bones[i]);
		}

		emit_signal(SceneStringNames::get_singleton()->bone_pose_changed, current_bone_idx);
	}
}

void Skeleton3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_bone", "name"), &Skeleton3D::add_bone);
	ClassDB::bind_method(D_METHOD("find_bone", "name"), &Skeleton3D::find_bone);
	ClassDB::bind_method(D_METHOD("get_bone_name", "bone_idx"), &Skeleton3D::get_bone_name);
	ClassDB::bind_method(D_METHOD("set_bone_name", "bone_idx", "name"), &Skeleton3D::set_bone_name);

	ClassDB::bind_method(D_METHOD("get_bone_parent", "bone_idx"), &Skeleton3D::get_bone_parent);
	ClassDB::bind_method(D_METHOD("set_bone_parent", "bone_idx", "parent_idx"), &Skeleton3D::set_bone_parent);

	ClassDB::bind_method(D_METHOD("get_bone_count"), &Skeleton3D::get_bone_count);
	ClassDB::bind_method(D_METHOD("get_version"), &Skeleton3D::get_version);

	ClassDB::bind_method(D_METHOD("unparent_bone_and_rest", "bone_idx"), &Skeleton3D::unparent_bone_and_rest);

	ClassDB::bind_method(D_METHOD("get_bone_children", "bone_idx"), &Skeleton3D::get_bone_children);

	ClassDB::bind_method(D_METHOD("get_parentless_bones"), &Skeleton3D::get_parentless_bones);

	ClassDB::bind_method(D_METHOD("get_bone_rest", "bone_idx"), &Skeleton3D::get_bone_rest);
	ClassDB::bind_method(D_METHOD("set_bone_rest", "bone_idx", "rest"), &Skeleton3D::set_bone_rest);
	ClassDB::bind_method(D_METHOD("get_bone_global_rest", "bone_idx"), &Skeleton3D::get_bone_global_rest);

	ClassDB::bind_method(D_METHOD("create_skin_from_rest_transforms"), &Skeleton3D::create_skin_from_rest_transforms);
	ClassDB::bind_method(D_METHOD("register_skin", "skin"), &Skeleton3D::register_skin);

	ClassDB::bind_method(D_METHOD("localize_rests"), &Skeleton3D::localize_rests);

	ClassDB::bind_method(D_METHOD("clear_bones"), &Skeleton3D::clear_bones);

	ClassDB::bind_method(D_METHOD("get_bone_pose", "bone_idx"), &Skeleton3D::get_bone_pose);
	ClassDB::bind_method(D_METHOD("set_bone_pose_position", "bone_idx", "position"), &Skeleton3D::set_bone_pose_position);
	ClassDB::bind_method(D_METHOD("set_bone_pose_rotation", "bone_idx", "rotation"), &Skeleton3D::set_bone_pose_rotation);
	ClassDB::bind_method(D_METHOD("set_bone_pose_scale", "bone_idx", "scale"), &Skeleton3D::set_bone_pose_scale);

	ClassDB::bind_method(D_METHOD("get_bone_pose_position", "bone_idx"), &Skeleton3D::get_bone_pose_position);
	ClassDB::bind_method(D_METHOD("get_bone_pose_rotation", "bone_idx"), &Skeleton3D::get_bone_pose_rotation);
	ClassDB::bind_method(D_METHOD("get_bone_pose_scale", "bone_idx"), &Skeleton3D::get_bone_pose_scale);

	ClassDB::bind_method(D_METHOD("reset_bone_pose", "bone_idx"), &Skeleton3D::reset_bone_pose);
	ClassDB::bind_method(D_METHOD("reset_bone_poses"), &Skeleton3D::reset_bone_poses);

	ClassDB::bind_method(D_METHOD("is_bone_enabled", "bone_idx"), &Skeleton3D::is_bone_enabled);
	ClassDB::bind_method(D_METHOD("set_bone_enabled", "bone_idx", "enabled"), &Skeleton3D::set_bone_enabled, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("clear_bones_global_pose_override"), &Skeleton3D::clear_bones_global_pose_override);
	ClassDB::bind_method(D_METHOD("set_bone_global_pose_override", "bone_idx", "pose", "amount", "persistent"), &Skeleton3D::set_bone_global_pose_override, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_bone_global_pose_override", "bone_idx"), &Skeleton3D::get_bone_global_pose_override);
	ClassDB::bind_method(D_METHOD("get_bone_global_pose", "bone_idx"), &Skeleton3D::get_bone_global_pose);
	ClassDB::bind_method(D_METHOD("get_bone_global_pose_no_override", "bone_idx"), &Skeleton3D::get_bone_global_pose_no_override);

	ClassDB::bind_method(D_METHOD("force_update_all_bone_transforms"), &Skeleton3D::force_update_all_bone_transforms);
	ClassDB::bind_method(D_METHOD("force_update_bone_child_transform", "bone_idx"), &Skeleton3D::force_update_bone_children_transforms);

	ClassDB::bind_method(D_METHOD("set_motion_scale", "motion_scale"), &Skeleton3D::set_motion_scale);
	ClassDB::bind_method(D_METHOD("get_motion_scale"), &Skeleton3D::get_motion_scale);

	ClassDB::bind_method(D_METHOD("set_show_rest_only", "enabled"), &Skeleton3D::set_show_rest_only);
	ClassDB::bind_method(D_METHOD("is_show_rest_only"), &Skeleton3D::is_show_rest_only);

	ClassDB::bind_method(D_METHOD("set_animate_physical_bones", "enabled"), &Skeleton3D::set_animate_physical_bones);
	ClassDB::bind_method(D_METHOD("get_animate_physical_bones"), &Skeleton3D::get_animate_physical_bones);

	ClassDB::bind_method(D_METHOD("physical_bones_stop_simulation"), &Skeleton3D::physical_bones_stop_simulation);
	ClassDB::bind_method(D_METHOD("physical_bones_start_simulation", "bones"), &Skeleton3D::physical_bones_start_simulation_on, DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("physical_bones_add_collision_exception", "exception"), &Skeleton3D::physical_bones_add_collision_exception);
	ClassDB::bind_method(D_METHOD("physical_bones_remove_collision_exception", "exception"), &Skeleton3D::physical_bones_remove_collision_exception);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "motion_scale", PROPERTY_HINT_RANGE, "0.001,10,0.001,or_greater"), "set_motion_scale", "get_motion_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_rest_only"), "set_show_rest_only", "is_show_rest_only");
#ifndef _3D_DISABLED
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "animate_physical_bones"), "set_animate_physical_bones", "get_animate_physical_bones");
#endif // _3D_DISABLED

	ADD_SIGNAL(MethodInfo("pose_updated"));
	ADD_SIGNAL(MethodInfo("bone_pose_changed", PropertyInfo(Variant::INT, "bone_idx")));
	ADD_SIGNAL(MethodInfo("bone_enabled_changed", PropertyInfo(Variant::INT, "bone_idx")));
	ADD_SIGNAL(MethodInfo("show_rest_only_changed"));

	BIND_CONSTANT(NOTIFICATION_UPDATE_SKELETON);
}

Skeleton3D::Skeleton3D() {
}

Skeleton3D::~Skeleton3D() {
	// Some skins may remain bound.
	for (SkinReference *E : skin_bindings) {
		E->skeleton_node = nullptr;
	}
}

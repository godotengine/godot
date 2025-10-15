/**************************************************************************/
/*  bone_expander_3d.cpp                                                  */
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

#include "bone_expander_3d.h"

bool BoneExpander3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "bone_name") {
			set_bone_name(which, p_value);
		} else if (what == "bone") {
			set_bone(which, p_value);
		} else if (what == "bone_scale") {
			set_bone_scale(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool BoneExpander3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "bone_name") {
			r_ret = get_bone_name(which);
		} else if (what == "bone") {
			r_ret = get_bone(which);
		} else if (what == "bone_scale") {
			r_ret = get_bone_scale(which);
		} else {
			return false;
		}
	}
	return true;
}

void BoneExpander3D::_get_property_list(List<PropertyInfo> *p_list) const {
	String enum_hint;
	Skeleton3D *skeleton = get_skeleton();
	if (skeleton) {
		enum_hint = skeleton->get_concatenated_bone_names();
	}

	for (uint32_t i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::STRING_NAME, path + "bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		p_list->push_back(PropertyInfo(Variant::INT, path + "bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::VECTOR3, path + "bone_scale"));
	}
}

int BoneExpander3D::get_setting_size() {
	return settings.size();
}

void BoneExpander3D::set_setting_size(int p_size) {
	ERR_FAIL_COND(p_size < 0);
	settings.resize(p_size);
	notify_property_list_changed();
}

void BoneExpander3D::clear_settings() {
	settings.clear();
}

void BoneExpander3D::set_bone_name(int p_index, const StringName &p_bone_name) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index].bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_bone(p_index, sk->find_bone(settings[p_index].bone_name));
	}
}

StringName BoneExpander3D::get_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), StringName());
	return settings[p_index].bone_name;
}

void BoneExpander3D::set_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	settings[p_index].bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index].bone <= -1 || settings[p_index].bone >= sk->get_bone_count()) {
			WARN_PRINT("apply bone index out of range!");
			settings[p_index].bone = -1;
			callable_mp(this, &BoneExpander3D::_force_render_skin).call_deferred(sk); // Force update to reset.
		} else {
			settings[p_index].bone_name = sk->get_bone_name(settings[p_index].bone);
		}
	}
	bone_changed = true;
	_make_skin_dirty();
}

int BoneExpander3D::get_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), -1);
	return settings[p_index].bone;
}

void BoneExpander3D::set_bone_scale(int p_index, Vector3 p_scale) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ERR_FAIL_COND_MSG(Math::is_zero_approx(p_scale.x * p_scale.y * p_scale.z), "Scale must not be zero.");
	settings[p_index].scale = p_scale;
}

Vector3 BoneExpander3D::get_bone_scale(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3(1, 1, 1));
	return settings[p_index].scale;
}

void BoneExpander3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_make_skin_dirty();
			_skeleton_changed(nullptr, get_skeleton()); // Bind skeleton.
			_set_active(active); // Force update rendered skin.
		} break;
	}
}

void BoneExpander3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bone_name", "index", "bone_name"), &BoneExpander3D::set_bone_name);
	ClassDB::bind_method(D_METHOD("get_bone_name", "index"), &BoneExpander3D::get_bone_name);
	ClassDB::bind_method(D_METHOD("set_bone", "index", "bone"), &BoneExpander3D::set_bone);
	ClassDB::bind_method(D_METHOD("get_bone", "index"), &BoneExpander3D::get_bone);
	ClassDB::bind_method(D_METHOD("set_bone_scale", "index", "scale"), &BoneExpander3D::set_bone_scale);
	ClassDB::bind_method(D_METHOD("get_bone_scale", "index"), &BoneExpander3D::get_bone_scale);

	ClassDB::bind_method(D_METHOD("set_setting_size", "size"), &BoneExpander3D::set_setting_size);
	ClassDB::bind_method(D_METHOD("get_setting_size"), &BoneExpander3D::get_setting_size);
	ClassDB::bind_method(D_METHOD("clear_setting"), &BoneExpander3D::clear_settings);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "setting_size", PROPERTY_HINT_RANGE, "0,1000,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Settings,settings/"), "set_setting_size", "get_setting_size");
}

void BoneExpander3D::_force_render_skin(Skeleton3D *p_skeleton) {
	p_skeleton->force_update_deferred();
	p_skeleton->notification(Skeleton3D::NOTIFICATION_UPDATE_SKELETON);
}

void BoneExpander3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
	if (p_old) {
		if (p_old->is_connected(SceneStringName(skeleton_updated), callable_mp(this, &BoneExpander3D::_apply_skin))) {
			p_old->disconnect(SceneStringName(skeleton_updated), callable_mp(this, &BoneExpander3D::_apply_skin));
		}
		if (p_old->is_connected(SceneStringName(skeleton_rendered), callable_mp(this, &BoneExpander3D::_restore_skin))) {
			p_old->disconnect(SceneStringName(skeleton_rendered), callable_mp(this, &BoneExpander3D::_restore_skin));
		}
		if (p_old->is_connected(SceneStringName(skin_changed), callable_mp(this, &BoneExpander3D::_make_skin_dirty))) {
			p_old->disconnect(SceneStringName(skin_changed), callable_mp(this, &BoneExpander3D::_make_skin_dirty));
		}
		// Hack:
		// Re-rendering is not called randomly, maybe depend on the processing order with RenderingServer?
		// So call it twice for the proof.
		_force_render_skin(p_old);
		callable_mp(this, &BoneExpander3D::_force_render_skin).call_deferred(p_old);
	}
	if (p_new) {
		if (!p_new->is_connected(SceneStringName(skeleton_updated), callable_mp(this, &BoneExpander3D::_apply_skin))) {
			p_new->connect(SceneStringName(skeleton_updated), callable_mp(this, &BoneExpander3D::_apply_skin));
		}
		if (!p_new->is_connected(SceneStringName(skeleton_rendered), callable_mp(this, &BoneExpander3D::_restore_skin))) {
			p_new->connect(SceneStringName(skeleton_rendered), callable_mp(this, &BoneExpander3D::_restore_skin));
		}
		if (!p_new->is_connected(SceneStringName(skin_changed), callable_mp(this, &BoneExpander3D::_make_skin_dirty))) {
			p_new->connect(SceneStringName(skin_changed), callable_mp(this, &BoneExpander3D::_make_skin_dirty));
		}
	}
	_make_skin_dirty();
}

void BoneExpander3D::_make_skin_dirty() {
	skin_info.clear();
	skin_dirty = true;
}

void BoneExpander3D::_set_active(bool p_active) {
	if (!p_active) {
		Skeleton3D *skeleton = get_skeleton();
		if (!skeleton) {
			return;
		}
		_force_render_skin(skeleton);
	}
}

void BoneExpander3D::_process_modification(double p_delta) {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	if (skin_dirty) {
		_map_skin();
	}

	bool processed = false;
	for (const BoneExpander3DSetting &setting : settings) {
		int bone = setting.bone;
		if (bone < 0) {
			continue;
		}

		Vector3 scl = setting.scale;
		Vector<int> children = skeleton->get_bone_children(bone);
		for (int i = 0; i < children.size(); i++) {
			int c = children[i];
			skeleton->set_bone_pose_position(c, skeleton->get_bone_pose_position(c) * scl);
		}

		for (KeyValue<ObjectID, HashMap<int, BindInfo>> &E : skin_info) {
			Ref<Skin> skin = ObjectDB::get_ref<Skin>(E.key);
			if (skin.is_null()) {
				continue;
			}
			HashMap<int, BindInfo> &binds = E.value;
			if (binds.has(bone)) {
				binds[bone].apply_scale = scl;
				binds[bone].original_matrix = skin->get_bind_pose(binds[bone].index);
			}
		}

		processed = true;
	}

	if (processed || bone_changed) {
		skeleton->force_update_deferred();
		bone_changed = false;
	}
}

void BoneExpander3D::_map_skin() {
	skin_info.clear();
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	for (SkinReference *E : skeleton->get_skin_bindings()) {
		Ref<Skin> skin = E->get_skin();
		HashMap<int, BindInfo> info;
		skin_info.insert(skin->get_instance_id(), info);
	}
	for (const BoneExpander3DSetting &setting : settings) {
		int bone = setting.bone;
		if (bone < 0) {
			continue;
		}
		for (KeyValue<ObjectID, HashMap<int, BindInfo>> &E : skin_info) {
			Ref<Skin> skin = ObjectDB::get_ref<Skin>(E.key);
			if (skin.is_null()) {
				continue;
			}
			int skin_len = skin->get_bind_count();
			for (int i = 0; i < skin_len; i++) {
				StringName bn = skin->get_bind_name(i);
				int bone_idx = skeleton->find_bone(bn);
				if (bone_idx >= 0 && bn == setting.bone_name) {
					BindInfo st;
					st.index = i;
					E.value.insert(bone, st);
				}
			}
		}
	}
}

void BoneExpander3D::_apply_skin() {
	for (const KeyValue<ObjectID, HashMap<int, BindInfo>> &E : skin_info) {
		Ref<Skin> skin = ObjectDB::get_ref<Skin>(E.key);
		if (skin.is_null()) {
			continue;
		}
		for (const KeyValue<int, BindInfo> &st : E.value) {
			int idx = st.value.index;
			if (idx < 0) {
				continue;
			}
			skin->set_bind_pose(idx, skin->get_bind_pose(idx).scaled(st.value.apply_scale));
		}
	}
}

void BoneExpander3D::_restore_skin() {
	for (const KeyValue<ObjectID, HashMap<int, BindInfo>> &E : skin_info) {
		Ref<Skin> skin = ObjectDB::get_ref<Skin>(E.key);
		if (skin.is_null()) {
			continue;
		}
		for (const KeyValue<int, BindInfo> &st : E.value) {
			int idx = st.value.index;
			if (idx < 0) {
				continue;
			}
			skin->set_bind_pose(st.value.index, st.value.original_matrix);
		}
	}
}

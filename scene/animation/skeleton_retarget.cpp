/*************************************************************************/
/*  skeleton_retarget.cpp                                                */
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

#include "skeleton_retarget.h"

#include "editor/plugins/skeleton_retarget_editor_plugin.h"

// Retarget profile

bool RetargetProfile::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (!path.begins_with("intermediate_bones/")) {
		return false;
	}

	int which = path.get_slicec('/', 1).to_int();
	String what = path.get_slicec('/', 2);

	if (which == intermediate_bone_names.size() && what == "bone_name") {
		add_intermediate_bone(p_value);
		return true;
	}

	ERR_FAIL_INDEX_V(which, intermediate_bone_names.size(), false);

	if (what == "bone_name") {
		set_intermediate_bone_name(which, p_value);
	} else {
		return false;
	}

	return true;
}

bool RetargetProfile::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (!path.begins_with("intermediate_bones/")) {
		return false;
	}

	int which = path.get_slicec('/', 1).to_int();
	String what = path.get_slicec('/', 2);

	ERR_FAIL_INDEX_V(which, intermediate_bone_names.size(), false);

	if (what == "bone_name") {
		r_ret = get_intermediate_bone_name(which);
	} else {
		return false;
	}

	return true;
}

void RetargetProfile::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < intermediate_bone_names.size(); i++) {
		String prep = "intermediate_bones/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::STRING, prep + "bone_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	}

	for (PropertyInfo &E : *p_list) {
		_validate_property(E);
	}
}

void RetargetProfile::_validate_property(PropertyInfo &property) const {
}

#ifdef TOOLS_ENABLED
void RetargetProfile::redraw() {
	emit_signal("redraw_needed");
}
#endif // TOOLS_ENABLED

void RetargetProfile::add_intermediate_bone(const StringName &p_intermediate_bone_name, int to_index) {
	if (to_index == -1) {
		(const_cast<RetargetProfile *>(this)->intermediate_bone_names).push_back(p_intermediate_bone_name);
	} else {
		ERR_FAIL_INDEX(to_index, intermediate_bone_names.size() + 1);
		(const_cast<RetargetProfile *>(this)->intermediate_bone_names).insert(to_index, p_intermediate_bone_name);
	}
#ifdef TOOLS_ENABLED
	redraw();
#endif // TOOLS_ENABLED
	emit_signal("intermediate_bone_updated");
	emit_signal("profile_updated");
}

void RetargetProfile::remove_intermediate_bone(int p_id) {
	ERR_FAIL_INDEX(p_id, intermediate_bone_names.size());
	(const_cast<RetargetProfile *>(this)->intermediate_bone_names).remove_at(p_id);
#ifdef TOOLS_ENABLED
	redraw();
#endif // TOOLS_ENABLED
	emit_signal("intermediate_bone_updated");
	emit_signal("profile_updated");
}

int RetargetProfile::find_intermediate_bone(const StringName &p_intermediate_bone_name) {
	return intermediate_bone_names.find(p_intermediate_bone_name);
}

int RetargetProfile::get_intermediate_bones_size() {
	return intermediate_bone_names.size();
}

void RetargetProfile::set_intermediate_bone_name(int p_id, const StringName &p_intermediate_bone_name) {
	ERR_FAIL_INDEX(p_id, intermediate_bone_names.size());
	intermediate_bone_names.write[p_id] = p_intermediate_bone_name;
	emit_signal("intermediate_bone_updated");
	emit_signal("profile_updated");
}

StringName RetargetProfile::get_intermediate_bone_name(int p_id) const {
	ERR_FAIL_INDEX_V(p_id, intermediate_bone_names.size(), StringName());
	return intermediate_bone_names[p_id];
}

void RetargetProfile::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_intermediate_bone", "intermediate_bone_name", "to_index"), &RetargetProfile::add_intermediate_bone, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("remove_intermediate_bone", "id"), &RetargetProfile::remove_intermediate_bone);
	ClassDB::bind_method(D_METHOD("find_intermediate_bone", "intermediate_bone_name"), &RetargetProfile::find_intermediate_bone);
	ClassDB::bind_method(D_METHOD("get_intermediate_bones_size"), &RetargetProfile::get_intermediate_bones_size);
	ClassDB::bind_method(D_METHOD("set_intermediate_bone_name", "id", "intermediate_bone_name"), &RetargetProfile::set_intermediate_bone_name);
	ClassDB::bind_method(D_METHOD("get_intermediate_bone_name", "id"), &RetargetProfile::get_intermediate_bone_name);
	ADD_SIGNAL(MethodInfo("intermediate_bone_updated"));
	ADD_SIGNAL(MethodInfo("profile_updated"));
#ifdef TOOLS_ENABLED
	ADD_SIGNAL(MethodInfo("redraw_needed"));
#endif // TOOLS_ENABLED
}

RetargetProfile::RetargetProfile() {
}

RetargetProfile::~RetargetProfile() {
}

// Retarget rich profile

bool RetargetRichProfile::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("groups/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);

		if (which == group_names.size() && what == "group_name") {
			add_group(p_value);
			return true;
		}

		ERR_FAIL_INDEX_V(which, group_names.size(), false);

		if (what == "group_name") {
			set_group_name(which, p_value);
		} else if (what == "group_texture") {
			set_group_texture(which, p_value);
		} else {
			return false;
		}

		return true;
	} else if (path.begins_with("intermediate_bones/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);

		if (which == intermediate_bone_names.size() && what == "bone_name") {
			add_intermediate_bone(p_value);
			return true;
		}

		ERR_FAIL_INDEX_V(which, intermediate_bone_names.size(), false);

		if (what == "bone_name") {
			set_intermediate_bone_name(which, p_value);
		} else if (what == "handle_offset") {
			set_intermediate_bone_handle_offset(which, p_value);
		} else if (what == "group_id") {
			set_intermediate_bone_group_id(which, p_value);
		} else {
			return false;
		}

		return true;
	}

	return false;
}

bool RetargetRichProfile::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("groups/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);

		ERR_FAIL_INDEX_V(which, group_names.size(), false);

		if (what == "group_name") {
			r_ret = get_group_name(which);
		} else if (what == "group_texture") {
			r_ret = get_group_texture(which);
		} else {
			return false;
		}

		return true;
	} else if (path.begins_with("intermediate_bones/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);

		ERR_FAIL_INDEX_V(which, intermediate_bone_names.size(), false);

		if (what == "bone_name") {
			r_ret = get_intermediate_bone_name(which);
		} else if (what == "handle_offset") {
			r_ret = get_intermediate_bone_handle_offset(which);
		} else if (what == "group_id") {
			r_ret = get_intermediate_bone_group_id(which);
		} else {
			return false;
		}

		return true;
	}

	return false;
}

void RetargetRichProfile::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < group_names.size(); i++) {
		String prep = "groups/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::STRING, prep + "group_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::OBJECT, prep + "group_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", PROPERTY_USAGE_NO_EDITOR));
	}

	for (int i = 0; i < intermediate_bone_names.size(); i++) {
		String prep = "intermediate_bones/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::STRING, prep + "bone_name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::VECTOR2, prep + "handle_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::INT, prep + "group_id", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	}

	for (PropertyInfo &E : *p_list) {
		_validate_property(E);
	}
}

void RetargetRichProfile::_validate_property(PropertyInfo &property) const {
}

void RetargetRichProfile::add_group(const StringName &p_group_name, int p_to_index) {
	if (p_to_index == -1) {
		(const_cast<RetargetRichProfile *>(this)->group_names).push_back(p_group_name);
		(const_cast<RetargetRichProfile *>(this)->group_textures).push_back(Ref<Texture2D>());
	} else {
		ERR_FAIL_INDEX(p_to_index, group_names.size() + 1);
		(const_cast<RetargetRichProfile *>(this)->group_names).insert(p_to_index, p_group_name);
		(const_cast<RetargetRichProfile *>(this)->group_textures).insert(p_to_index, Ref<Texture2D>());
	}
#ifdef TOOLS_ENABLED
	redraw();
#endif // TOOLS_ENABLED
	emit_signal("group_updated");
	emit_signal("profile_updated");
}

void RetargetRichProfile::remove_group(int p_id) {
	ERR_FAIL_INDEX(p_id, group_names.size());
	(const_cast<RetargetRichProfile *>(this)->group_names).remove_at(p_id);
	(const_cast<RetargetRichProfile *>(this)->group_textures).remove_at(p_id);
	int len = intermediate_bone_names.size();
	for (int i = 0; i < len; i++) {
		set_intermediate_bone_group_id(i, get_intermediate_bone_group_id(i), false);
	}
#ifdef TOOLS_ENABLED
	redraw();
#endif // TOOLS_ENABLED
	emit_signal("group_updated");
	emit_signal("intermediate_bone_updated");
	emit_signal("profile_updated");
}

int RetargetRichProfile::find_group(const StringName &p_group_name) {
	return group_names.find(p_group_name);
}

int RetargetRichProfile::get_groups_size() const {
	return group_names.size();
}

void RetargetRichProfile::set_group_name(int p_id, const StringName &p_group_name) {
	ERR_FAIL_INDEX(p_id, group_names.size());
	group_names.write[p_id] = p_group_name;
	emit_signal("group_updated");
	emit_signal("profile_updated");
}

StringName RetargetRichProfile::get_group_name(int p_id) const {
	ERR_FAIL_INDEX_V(p_id, group_names.size(), StringName());
	return group_names[p_id];
}

void RetargetRichProfile::set_group_texture(int p_id, const Ref<Texture2D> &p_group_texture) {
	ERR_FAIL_INDEX(p_id, group_names.size());
	group_textures.write[p_id] = p_group_texture;
	emit_signal("group_updated");
	emit_signal("profile_updated");
}

Ref<Texture2D> RetargetRichProfile::get_group_texture(int p_id) const {
	ERR_FAIL_INDEX_V(p_id, group_names.size(), Ref<Texture2D>());
	return group_textures[p_id];
}

void RetargetRichProfile::add_intermediate_bone(const StringName &p_intermediate_bone_name, int to_index) {
	if (to_index == -1) {
		(const_cast<RetargetRichProfile *>(this)->intermediate_bone_names).push_back(p_intermediate_bone_name);
		(const_cast<RetargetRichProfile *>(this)->intermediate_bone_handle_offsets).push_back(Vector2());
		(const_cast<RetargetRichProfile *>(this)->intermediate_bone_group_ids).push_back(0);
	} else {
		ERR_FAIL_INDEX(to_index, intermediate_bone_names.size() + 1);
		(const_cast<RetargetRichProfile *>(this)->intermediate_bone_names).insert(to_index, p_intermediate_bone_name);
		(const_cast<RetargetRichProfile *>(this)->intermediate_bone_handle_offsets).insert(to_index, Vector2());
		(const_cast<RetargetRichProfile *>(this)->intermediate_bone_group_ids).insert(to_index, 0);
	}
#ifdef TOOLS_ENABLED
	redraw();
#endif // TOOLS_ENABLED
	emit_signal("intermediate_bone_updated");
	emit_signal("profile_updated");
}

void RetargetRichProfile::remove_intermediate_bone(int p_id) {
	ERR_FAIL_INDEX(p_id, intermediate_bone_names.size());
	(const_cast<RetargetRichProfile *>(this)->intermediate_bone_names).remove_at(p_id);
	(const_cast<RetargetRichProfile *>(this)->intermediate_bone_handle_offsets).remove_at(p_id);
	(const_cast<RetargetRichProfile *>(this)->intermediate_bone_group_ids).remove_at(p_id);
#ifdef TOOLS_ENABLED
	redraw();
#endif // TOOLS_ENABLED
	emit_signal("intermediate_bone_updated");
	emit_signal("profile_updated");
}

void RetargetRichProfile::set_intermediate_bone_handle_offset(int p_id, Vector2 p_handle_offset) {
	ERR_FAIL_INDEX(p_id, intermediate_bone_names.size());
	intermediate_bone_handle_offsets.write[p_id] = p_handle_offset;
	emit_signal("intermediate_bone_updated");
	emit_signal("profile_updated");
}

Vector2 RetargetRichProfile::get_intermediate_bone_handle_offset(int p_id) const {
	ERR_FAIL_INDEX_V(p_id, intermediate_bone_names.size(), Vector2());
	return intermediate_bone_handle_offsets[p_id];
}

void RetargetRichProfile::set_intermediate_bone_group_id(int p_id, int p_group_id, bool p_emit_signal) {
	ERR_FAIL_INDEX(p_id, intermediate_bone_names.size());
	intermediate_bone_group_ids.write[p_id] = p_group_id;
	if (p_emit_signal) {
		emit_signal("intermediate_bone_updated");
		emit_signal("profile_updated");
	}
}

int RetargetRichProfile::get_intermediate_bone_group_id(int p_id) const {
	ERR_FAIL_INDEX_V(p_id, intermediate_bone_names.size(), 0);
	return MIN(MAX(0, group_names.size() - 1), intermediate_bone_group_ids[p_id]);
}

void RetargetRichProfile::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_group", "group_name", "to_index"), &RetargetRichProfile::add_group, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("remove_group", "id"), &RetargetRichProfile::remove_group);
	ClassDB::bind_method(D_METHOD("find_group", "group_name"), &RetargetRichProfile::find_group);
	ClassDB::bind_method(D_METHOD("get_groups_size"), &RetargetRichProfile::get_groups_size);
	ClassDB::bind_method(D_METHOD("set_group_name", "id", "group_name"), &RetargetRichProfile::set_group_name);
	ClassDB::bind_method(D_METHOD("get_group_name", "id"), &RetargetRichProfile::get_group_name);
	ClassDB::bind_method(D_METHOD("set_group_texture", "id", "group_texture"), &RetargetRichProfile::set_group_texture);
	ClassDB::bind_method(D_METHOD("get_group_texture", "id"), &RetargetRichProfile::get_group_texture);
	ClassDB::bind_method(D_METHOD("set_intermediate_bone_handle_offset", "id", "handle_offset"), &RetargetRichProfile::set_intermediate_bone_handle_offset);
	ClassDB::bind_method(D_METHOD("get_intermediate_bone_handle_offset", "id"), &RetargetRichProfile::get_intermediate_bone_handle_offset);
	ClassDB::bind_method(D_METHOD("set_intermediate_bone_group_id", "id", "group_id", "emit_signal"), &RetargetRichProfile::set_intermediate_bone_group_id, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_intermediate_bone_group_id", "id"), &RetargetRichProfile::get_intermediate_bone_group_id);
	ADD_SIGNAL(MethodInfo("group_updated"));
}

RetargetRichProfile::RetargetRichProfile() {
}

RetargetRichProfile::~RetargetRichProfile() {
}

// Source setting

bool RetargetBoneOption::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.split("/").size() < 1) {
		return false;
	}

	String which = path.get_slicec('/', 0);
	String what = path.get_slicec('/', 1);

	if (!retarget_options.has(which)) {
		add_key(which);
	}

	if (what == "retarget_mode") {
		set_retarget_mode(which, Animation::RetargetMode(p_value.operator int()));
	} else {
		return false;
	}

	return true;
}

bool RetargetBoneOption::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.split("/").size() < 1) {
		return false;
	}

	String which = path.get_slicec('/', 0);
	String what = path.get_slicec('/', 1);

	ERR_FAIL_COND_V(!retarget_options.has(which), false);

	if (what == "retarget_mode") {
		r_ret = get_retarget_mode(which);
	} else {
		return false;
	}

	return true;
}

void RetargetBoneOption::_get_property_list(List<PropertyInfo> *p_list) const {
	for (Map<StringName, RetargetBoneOptionParams>::Element *E = retarget_options.front(); E; E = E->next()) {
		String prep = String(E->key()) + "/";
		p_list->push_back(PropertyInfo(Variant::INT, prep + "retarget_mode", PROPERTY_HINT_ENUM, "Global,Local,Absolute", PROPERTY_USAGE_NO_EDITOR));
	}

	for (PropertyInfo &E : *p_list) {
		_validate_property(E);
	}
}

void RetargetBoneOption::_validate_property(PropertyInfo &property) const {
}

Vector<StringName> RetargetBoneOption::get_keys() const {
	Vector<StringName> arr;
	for (Map<StringName, RetargetBoneOptionParams>::Element *E = retarget_options.front(); E; E = E->next()) {
		arr.push_back(E->key());
	}
	return arr;
}

bool RetargetBoneOption::has_key(const StringName &p_intermediate_bone_name) {
	return (const_cast<RetargetBoneOption *>(this)->retarget_options).has(p_intermediate_bone_name);
}

void RetargetBoneOption::add_key(const StringName &p_intermediate_bone_name) {
	RetargetBoneOptionParams params = RetargetBoneOptionParams();
	(const_cast<RetargetBoneOption *>(this)->retarget_options).insert(p_intermediate_bone_name, params);
#ifdef TOOLS_ENABLED
	redraw();
#endif // TOOLS_ENABLED
	emit_signal("retarget_option_updated");
}

void RetargetBoneOption::remove_key(const StringName &p_intermediate_bone_name) {
	(const_cast<RetargetBoneOption *>(this)->retarget_options).erase(p_intermediate_bone_name);
#ifdef TOOLS_ENABLED
	redraw();
#endif // TOOLS_ENABLED
	emit_signal("retarget_option_updated");
}

void RetargetBoneOption::set_retarget_mode(const StringName &p_intermediate_bone_name, Animation::RetargetMode p_retarget_mode) {
	ERR_FAIL_COND(!retarget_options.has(p_intermediate_bone_name));
	retarget_options[p_intermediate_bone_name].retarget_mode = p_retarget_mode;
	emit_signal("retarget_option_updated");
}

Animation::RetargetMode RetargetBoneOption::get_retarget_mode(const StringName &p_intermediate_bone_name) const {
	ERR_FAIL_COND_V(!retarget_options.has(p_intermediate_bone_name), Animation::RETARGET_MODE_ABSOLUTE);
	return retarget_options[p_intermediate_bone_name].retarget_mode;
}

#ifdef TOOLS_ENABLED
void RetargetBoneOption::redraw() {
	emit_signal("redraw_needed");
}
#endif // TOOLS_ENABLED

void RetargetBoneOption::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_keys"), &RetargetBoneOption::get_keys);
	ClassDB::bind_method(D_METHOD("has_key", "intermediate_bone_name"), &RetargetBoneOption::has_key);
	ClassDB::bind_method(D_METHOD("add_key", "intermediate_bone_name"), &RetargetBoneOption::add_key);
	ClassDB::bind_method(D_METHOD("remove_key", "intermediate_bone_name"), &RetargetBoneOption::remove_key);
	ClassDB::bind_method(D_METHOD("set_retarget_mode", "intermediate_bone_name", "retarget_mode"), &RetargetBoneOption::set_retarget_mode);
	ClassDB::bind_method(D_METHOD("get_retarget_mode", "intermediate_bone_name"), &RetargetBoneOption::get_retarget_mode);
	ADD_SIGNAL(MethodInfo("retarget_option_updated"));
#ifdef TOOLS_ENABLED
	ADD_SIGNAL(MethodInfo("redraw_needed"));
#endif // TOOLS_ENABLED
}

RetargetBoneOption::RetargetBoneOption() {
}

RetargetBoneOption::~RetargetBoneOption() {
}

// Target setting

bool RetargetBoneMap::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.split("/").size() < 1) {
		return false;
	}

	String which = path.get_slicec('/', 0);
	String what = path.get_slicec('/', 1);

	if (!retarget_map.has(which)) {
		add_key(which);
	}

	set_bone_name(which, p_value);

	return true;
}

bool RetargetBoneMap::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.split("/").size() < 1) {
		return false;
	}

	String which = path.get_slicec('/', 0);
	String what = path.get_slicec('/', 1);

	ERR_FAIL_COND_V(!retarget_map.has(which), false);

	r_ret = get_bone_name(which);

	return true;
}

void RetargetBoneMap::_get_property_list(List<PropertyInfo> *p_list) const {
	for (Map<StringName, StringName>::Element *E = retarget_map.front(); E; E = E->next()) {
		p_list->push_back(PropertyInfo(Variant::STRING, E->key(), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
	}

	for (PropertyInfo &E : *p_list) {
		_validate_property(E);
	}
}

void RetargetBoneMap::_validate_property(PropertyInfo &property) const {
}

Vector<StringName> RetargetBoneMap::get_keys() const {
	Vector<StringName> arr;
	for (Map<StringName, StringName>::Element *E = retarget_map.front(); E; E = E->next()) {
		arr.push_back(E->key());
	}
	return arr;
}

bool RetargetBoneMap::has_key(const StringName &p_intermediate_bone_name) {
	return (const_cast<RetargetBoneMap *>(this)->retarget_map).has(p_intermediate_bone_name);
}

void RetargetBoneMap::add_key(const StringName &p_intermediate_bone_name) {
	(const_cast<RetargetBoneMap *>(this)->retarget_map).insert(p_intermediate_bone_name, StringName());
#ifdef TOOLS_ENABLED
	redraw();
#endif // TOOLS_ENABLED
	emit_signal("retarget_map_updated");
}

void RetargetBoneMap::remove_key(const StringName &p_intermediate_bone_name) {
	(const_cast<RetargetBoneMap *>(this)->retarget_map).erase(p_intermediate_bone_name);
#ifdef TOOLS_ENABLED
	redraw();
#endif // TOOLS_ENABLED
	emit_signal("retarget_map_updated");
}

void RetargetBoneMap::set_bone_name(const StringName &p_intermediate_bone_name, const StringName &p_bone_name) {
	ERR_FAIL_COND(!retarget_map.has(p_intermediate_bone_name));
	retarget_map[p_intermediate_bone_name] = p_bone_name;
	emit_signal("retarget_map_updated");
}

StringName RetargetBoneMap::get_bone_name(const StringName &p_intermediate_bone_name) const {
	ERR_FAIL_COND_V(!retarget_map.has(p_intermediate_bone_name), StringName());
	return retarget_map[p_intermediate_bone_name];
}

StringName RetargetBoneMap::find_key(const StringName &p_bone_name) const {
	StringName found_key = "";
	for (Map<StringName, StringName>::Element *E = retarget_map.front(); E; E = E->next()) {
		if (E->get() == p_bone_name) {
			found_key = E->key();
			break;
		}
	}
	return found_key;
};

#ifdef TOOLS_ENABLED
void RetargetBoneMap::redraw() {
	emit_signal("redraw_needed");
}
#endif // TOOLS_ENABLED

void RetargetBoneMap::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_keys"), &RetargetBoneMap::get_keys);
	ClassDB::bind_method(D_METHOD("has_key", "intermediate_bone_name"), &RetargetBoneMap::has_key);
	ClassDB::bind_method(D_METHOD("add_key", "intermediate_bone_name"), &RetargetBoneMap::add_key);
	ClassDB::bind_method(D_METHOD("remove_key", "intermediate_bone_name"), &RetargetBoneMap::remove_key);
	ClassDB::bind_method(D_METHOD("find_key", "bone_name"), &RetargetBoneMap::find_key);
	ClassDB::bind_method(D_METHOD("set_bone_name", "intermediate_bone_name", "bone_name"), &RetargetBoneMap::set_bone_name);
	ClassDB::bind_method(D_METHOD("get_bone_name", "intermediate_bone_name"), &RetargetBoneMap::get_bone_name);
	ADD_SIGNAL(MethodInfo("retarget_map_updated"));
#ifdef TOOLS_ENABLED
	ADD_SIGNAL(MethodInfo("redraw_needed"));
#endif // TOOLS_ENABLED
}

RetargetBoneMap::RetargetBoneMap() {
}

RetargetBoneMap::~RetargetBoneMap() {
}

// Reatrget Node

void SkeletonRetarget::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			source_skeleton = Object::cast_to<Skeleton3D>(get_node_or_null(source_skeleton_path));
			if (source_skeleton && !source_skeleton->is_connected("pose_updated", callable_mp(this, &SkeletonRetarget::_transpote_pose))) {
				source_skeleton->connect("pose_updated", callable_mp(this, &SkeletonRetarget::_transpote_pose));
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
#ifdef TOOLS_ENABLED
			if (retarget_profile.is_valid() && retarget_profile->is_connected("profile_updated", callable_mp(this, &SkeletonRetarget::_redraw))) {
				retarget_profile->disconnect("profile_updated", callable_mp(this, &SkeletonRetarget::_redraw));
			}
#endif // TOOLS_ENABLED
			if (retarget_option.is_valid() && retarget_option->is_connected("retarget_option_updated", callable_mp(this, &SkeletonRetarget::_clear_override))) {
				retarget_option->disconnect("retarget_option_updated", callable_mp(this, &SkeletonRetarget::_clear_override));
			}
			if (source_skeleton && source_skeleton->is_connected("pose_updated", callable_mp(this, &SkeletonRetarget::_transpote_pose))) {
				source_skeleton->disconnect("pose_updated", callable_mp(this, &SkeletonRetarget::_transpote_pose));
			}
			if (source_map.is_valid() && source_map->is_connected("retarget_map_updated", callable_mp(this, &SkeletonRetarget::_clear_override))) {
				source_map->disconnect("retarget_map_updated", callable_mp(this, &SkeletonRetarget::_clear_override));
			}
			if (target_map.is_valid() && target_map->is_connected("retarget_map_updated", callable_mp(this, &SkeletonRetarget::_clear_override))) {
				target_map->disconnect("retarget_map_updated", callable_mp(this, &SkeletonRetarget::_clear_override));
			}
			_clear_override();
		} break;
	}
}

#ifdef TOOLS_ENABLED
void SkeletonRetarget::_redraw() {
	if (source_map.is_valid()) {
		source_map->redraw();
	}
	if (target_map.is_valid()) {
		target_map->redraw();
	}
}
#endif // TOOLS_ENABLED

void SkeletonRetarget::_clear_override() {
	Skeleton3D *target_skeleton = Object::cast_to<Skeleton3D>(get_node_or_null(target_skeleton_path));
	if (target_skeleton) {
		target_skeleton->clear_bones_local_pose_override();
	}
}

void SkeletonRetarget::_transpote_pose() {
	Skeleton3D *s_sk = Object::cast_to<Skeleton3D>(get_node_or_null(source_skeleton_path));
	Skeleton3D *t_sk = Object::cast_to<Skeleton3D>(get_node_or_null(target_skeleton_path));
	if (s_sk && t_sk && source_map.is_valid() && target_map.is_valid()) {
		Vector<StringName> intermediate_bones = target_map->get_keys();
		int len = intermediate_bones.size();
		for (int i = 0; i < len; i++) {
			StringName imbone_name = intermediate_bones[i];
			if (!source_map->has_key(imbone_name) || !target_map->has_key(imbone_name)) {
				continue; // Bone is not found in settings.
			}
			int source_bone = s_sk->find_bone(source_map->get_bone_name(imbone_name));
			int target_bone = t_sk->find_bone(target_map->get_bone_name(imbone_name));
			if (source_bone < 0 || target_bone < 0) {
				continue; // Bone is not found in skeletons.
			}

			// If use extract_global_retarget_position/rotation/scale(), you get bone_pose_no_override.
			// Most of modifications use bone_pose_override, so use extract_global_retarget_transform()
			// since it get bone_pose_override.
			Transform3D pose = Transform3D();
			if (retarget_option.is_valid() && retarget_option->has_key(imbone_name)) {
				switch (retarget_option->get_retarget_mode(imbone_name)) {
					case Animation::RETARGET_MODE_GLOBAL: {
						pose = t_sk->global_retarget_transform_to_local_pose(target_bone, s_sk->extract_global_retarget_transform(source_bone));
					} break;
					case Animation::RETARGET_MODE_LOCAL: {
						pose = t_sk->local_retarget_transform_to_local_pose(target_bone, s_sk->extract_local_retarget_transform(source_bone));
					} break;
					case Animation::RETARGET_MODE_ABSOLUTE: {
						pose = s_sk->get_bone_pose(source_bone);
					} break;
					default: {
					} break;
				}
			} else {
				pose = t_sk->global_retarget_transform_to_local_pose(target_bone, s_sk->extract_global_retarget_transform(source_bone));
			}

			Transform3D final_pose = t_sk->get_bone_rest(target_bone);
			if (retarget_position) {
				final_pose.origin = pose.get_origin();
			}
			if (retarget_rotation) {
				final_pose.basis = pose.basis.orthonormalized();
			}
			if (retarget_scale) {
				final_pose.basis.set_quaternion_scale(final_pose.basis.get_rotation_quaternion(), pose.basis.get_scale());
			}
			t_sk->set_bone_local_pose_override(target_bone, final_pose, 1, true);
		}
	}
}

void SkeletonRetarget::set_retarget_profile(const Ref<RetargetProfile> &p_retarget_profile) {
#ifdef TOOLS_ENABLED
	if (retarget_profile.is_valid() && retarget_profile->is_connected("profile_updated", callable_mp(this, &SkeletonRetarget::_redraw))) {
		retarget_profile->disconnect("profile_updated", callable_mp(this, &SkeletonRetarget::_redraw));
	}
#endif // TOOLS_ENABLED
	retarget_profile = p_retarget_profile;
#ifdef TOOLS_ENABLED
	if (retarget_profile.is_valid() && !retarget_profile->is_connected("profile_updated", callable_mp(this, &SkeletonRetarget::_redraw))) {
		retarget_profile->connect("profile_updated", callable_mp(this, &SkeletonRetarget::_redraw));
	}
#endif // TOOLS_ENABLED
	notify_property_list_changed();
}

Ref<RetargetProfile> SkeletonRetarget::get_retarget_profile() const {
	return retarget_profile;
}

void SkeletonRetarget::set_retarget_option(const Ref<RetargetBoneOption> &p_retarget_option) {
	if (retarget_option.is_valid() && retarget_option->is_connected("retarget_option_updated", callable_mp(this, &SkeletonRetarget::_clear_override))) {
		retarget_option->disconnect("retarget_option_updated", callable_mp(this, &SkeletonRetarget::_clear_override));
	}
	_clear_override();
	retarget_option = p_retarget_option;
	if (retarget_option.is_valid() && !retarget_option->is_connected("retarget_option_updated", callable_mp(this, &SkeletonRetarget::_clear_override))) {
		retarget_option->connect("retarget_option_updated", callable_mp(this, &SkeletonRetarget::_clear_override));
	}
	notify_property_list_changed();
}

Ref<RetargetBoneOption> SkeletonRetarget::get_retarget_option() const {
	return retarget_option;
}

void SkeletonRetarget::set_source_skeleton(const NodePath &p_skeleton) {
	ERR_FAIL_COND_MSG(!p_skeleton.is_empty() && !target_skeleton_path.is_empty() && p_skeleton == target_skeleton_path, "The source and target skeletons are not allowed to be the same.");
	// Init.
	if (source_skeleton && source_skeleton->is_connected("pose_updated", callable_mp(this, &SkeletonRetarget::_transpote_pose))) {
		source_skeleton->disconnect("pose_updated", callable_mp(this, &SkeletonRetarget::_transpote_pose));
	}
	_clear_override();
	// Set.
	source_skeleton_path = p_skeleton;
	source_skeleton = Object::cast_to<Skeleton3D>(get_node_or_null(source_skeleton_path));
	if (source_skeleton && !source_skeleton->is_connected("pose_updated", callable_mp(this, &SkeletonRetarget::_transpote_pose))) {
		source_skeleton->connect("pose_updated", callable_mp(this, &SkeletonRetarget::_transpote_pose));
	}
	notify_property_list_changed();
}

NodePath SkeletonRetarget::get_source_skeleton() const {
	return source_skeleton_path;
}

void SkeletonRetarget::set_source_map(const Ref<RetargetBoneMap> &p_source_map) {
	ERR_FAIL_COND_MSG(p_source_map.is_valid() && target_map.is_valid() && p_source_map == target_map, "The source and target maps are not allowed to be the same. You should make one of them unique or duplicate it.");
	if (source_map.is_valid() && source_map->is_connected("retarget_map_updated", callable_mp(this, &SkeletonRetarget::_clear_override))) {
		source_map->disconnect("retarget_map_updated", callable_mp(this, &SkeletonRetarget::_clear_override));
	}
	_clear_override();
	source_map = p_source_map;
	if (source_map.is_valid() && !source_map->is_connected("retarget_map_updated", callable_mp(this, &SkeletonRetarget::_clear_override))) {
		source_map->connect("retarget_map_updated", callable_mp(this, &SkeletonRetarget::_clear_override));
	}
}

Ref<RetargetBoneMap> SkeletonRetarget::get_source_map() const {
	return source_map;
}

void SkeletonRetarget::set_target_skeleton(const NodePath &p_skeleton) {
	ERR_FAIL_COND_MSG(!p_skeleton.is_empty() && !source_skeleton_path.is_empty() && p_skeleton == source_skeleton_path, "The source and target skeletons are not allowed to be the same.");
	// Init.
	_clear_override();
	// Set.
	target_skeleton_path = p_skeleton;
	notify_property_list_changed();
}

NodePath SkeletonRetarget::get_target_skeleton() const {
	return target_skeleton_path;
}

void SkeletonRetarget::set_target_map(const Ref<RetargetBoneMap> &p_target_map) {
	ERR_FAIL_COND_MSG(p_target_map.is_valid() && source_map.is_valid() && p_target_map == source_map, "The source and target maps are not allowed to be the same. You should make one of them unique or duplicate it.");
	if (target_map.is_valid() && target_map->is_connected("retarget_map_updated", callable_mp(this, &SkeletonRetarget::_clear_override))) {
		target_map->disconnect("retarget_map_updated", callable_mp(this, &SkeletonRetarget::_clear_override));
	}
	_clear_override();
	target_map = p_target_map;
	if (target_map.is_valid() && !target_map->is_connected("retarget_map_updated", callable_mp(this, &SkeletonRetarget::_clear_override))) {
		target_map->connect("retarget_map_updated", callable_mp(this, &SkeletonRetarget::_clear_override));
	}
}

Ref<RetargetBoneMap> SkeletonRetarget::get_target_map() const {
	return target_map;
}

void SkeletonRetarget::set_retarget_position(bool p_enabled) {
	retarget_position = p_enabled;
}

bool SkeletonRetarget::is_retarget_position() const {
	return retarget_position;
}

void SkeletonRetarget::set_retarget_rotation(bool p_enabled) {
	retarget_rotation = p_enabled;
}

bool SkeletonRetarget::is_retarget_rotation() const {
	return retarget_rotation;
}

void SkeletonRetarget::set_retarget_scale(bool p_enabled) {
	retarget_scale = p_enabled;
}

bool SkeletonRetarget::is_retarget_scale() const {
	return retarget_scale;
}

void SkeletonRetarget::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_retarget_profile", "retarget_profile"), &SkeletonRetarget::set_retarget_profile);
	ClassDB::bind_method(D_METHOD("get_retarget_profile"), &SkeletonRetarget::get_retarget_profile);
	ClassDB::bind_method(D_METHOD("set_retarget_option", "retarget_option"), &SkeletonRetarget::set_retarget_option);
	ClassDB::bind_method(D_METHOD("get_retarget_option"), &SkeletonRetarget::get_retarget_option);
	ClassDB::bind_method(D_METHOD("set_source_skeleton", "source_skeleton_path"), &SkeletonRetarget::set_source_skeleton);
	ClassDB::bind_method(D_METHOD("get_source_skeleton"), &SkeletonRetarget::get_source_skeleton);
	ClassDB::bind_method(D_METHOD("set_source_map", "source_map"), &SkeletonRetarget::set_source_map);
	ClassDB::bind_method(D_METHOD("get_source_map"), &SkeletonRetarget::get_source_map);
	ClassDB::bind_method(D_METHOD("set_target_skeleton", "target_skeleton_path"), &SkeletonRetarget::set_target_skeleton);
	ClassDB::bind_method(D_METHOD("get_target_skeleton"), &SkeletonRetarget::get_target_skeleton);
	ClassDB::bind_method(D_METHOD("set_target_map", "target_map"), &SkeletonRetarget::set_target_map);
	ClassDB::bind_method(D_METHOD("get_target_map"), &SkeletonRetarget::get_target_map);
	ClassDB::bind_method(D_METHOD("set_retarget_position", "retarget_position"), &SkeletonRetarget::set_retarget_position);
	ClassDB::bind_method(D_METHOD("is_retarget_position"), &SkeletonRetarget::is_retarget_position);
	ClassDB::bind_method(D_METHOD("set_retarget_rotation", "retarget_rotation"), &SkeletonRetarget::set_retarget_rotation);
	ClassDB::bind_method(D_METHOD("is_retarget_rotation"), &SkeletonRetarget::is_retarget_rotation);
	ClassDB::bind_method(D_METHOD("set_retarget_scale", "retarget_scale"), &SkeletonRetarget::set_retarget_scale);
	ClassDB::bind_method(D_METHOD("is_retarget_scale"), &SkeletonRetarget::is_retarget_scale);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "retarget_profile", PROPERTY_HINT_RESOURCE_TYPE, "RetargetProfile"), "set_retarget_profile", "get_retarget_profile");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "retarget_option", PROPERTY_HINT_RESOURCE_TYPE, "RetargetBoneOption"), "set_retarget_option", "get_retarget_option");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "source_skeleton", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D"), "set_source_skeleton", "get_source_skeleton");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "source_map", PROPERTY_HINT_RESOURCE_TYPE, "RetargetBoneMap"), "set_source_map", "get_source_map");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_skeleton", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Skeleton3D"), "set_target_skeleton", "get_target_skeleton");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "target_map", PROPERTY_HINT_RESOURCE_TYPE, "RetargetBoneMap"), "set_target_map", "get_target_map");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "retarget_position"), "set_retarget_position", "is_retarget_position");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "retarget_rotation"), "set_retarget_rotation", "is_retarget_rotation");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "retarget_scale"), "set_retarget_scale", "is_retarget_scale");
}

SkeletonRetarget::SkeletonRetarget() {
}

SkeletonRetarget::~SkeletonRetarget() {
}

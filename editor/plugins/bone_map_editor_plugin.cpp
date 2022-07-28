/*************************************************************************/
/*  bone_map_editor_plugin.cpp                                           */
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

#include "bone_map_editor_plugin.h"

#include "editor/editor_scale.h"
#include "editor/import/post_import_plugin_skeleton_renamer.h"
#include "editor/import/post_import_plugin_skeleton_rest_fixer.h"
#include "editor/import/post_import_plugin_skeleton_track_organizer.h"
#include "editor/import/scene_import_settings.h"

void BoneMapperButton::fetch_textures() {
	if (selected) {
		set_normal_texture(get_theme_icon(SNAME("BoneMapperHandleSelected"), SNAME("EditorIcons")));
	} else {
		set_normal_texture(get_theme_icon(SNAME("BoneMapperHandle"), SNAME("EditorIcons")));
	}
	set_offset(SIDE_LEFT, 0);
	set_offset(SIDE_RIGHT, 0);
	set_offset(SIDE_TOP, 0);
	set_offset(SIDE_BOTTOM, 0);

	circle = memnew(TextureRect);
	circle->set_texture(get_theme_icon(SNAME("BoneMapperHandleCircle"), SNAME("EditorIcons")));
	add_child(circle);
	set_state(BONE_MAP_STATE_UNSET);
}

StringName BoneMapperButton::get_profile_bone_name() const {
	return profile_bone_name;
}

void BoneMapperButton::set_state(BoneMapState p_state) {
	switch (p_state) {
		case BONE_MAP_STATE_UNSET: {
			circle->set_modulate(EditorSettings::get_singleton()->get("editors/bone_mapper/handle_colors/unset"));
		} break;
		case BONE_MAP_STATE_SET: {
			circle->set_modulate(EditorSettings::get_singleton()->get("editors/bone_mapper/handle_colors/set"));
		} break;
		case BONE_MAP_STATE_MISSING: {
			circle->set_modulate(EditorSettings::get_singleton()->get("editors/bone_mapper/handle_colors/missing"));
		} break;
		case BONE_MAP_STATE_ERROR: {
			circle->set_modulate(EditorSettings::get_singleton()->get("editors/bone_mapper/handle_colors/error"));
		} break;
		default: {
		} break;
	}
}

bool BoneMapperButton::is_require() const {
	return require;
}

void BoneMapperButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			fetch_textures();
		} break;
	}
}

BoneMapperButton::BoneMapperButton(const StringName p_profile_bone_name, bool p_require, bool p_selected) {
	profile_bone_name = p_profile_bone_name;
	require = p_require;
	selected = p_selected;
}

BoneMapperButton::~BoneMapperButton() {
}

void BoneMapperItem::create_editor() {
	skeleton_bone_selector = memnew(EditorPropertyTextEnum);
	skeleton_bone_selector->setup(skeleton_bone_names, false, true);
	skeleton_bone_selector->set_label(profile_bone_name);
	skeleton_bone_selector->set_selectable(false);
	skeleton_bone_selector->set_object_and_property(bone_map.ptr(), "bone_map/" + String(profile_bone_name));
	skeleton_bone_selector->update_property();
	skeleton_bone_selector->connect("property_changed", callable_mp(this, &BoneMapperItem::_value_changed));
	add_child(skeleton_bone_selector);
}

void BoneMapperItem::_update_property() {
	if (skeleton_bone_selector->get_edited_object() && skeleton_bone_selector->get_edited_property()) {
		skeleton_bone_selector->update_property();
	}
}

void BoneMapperItem::_value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	bone_map->set(p_property, p_value);
}

void BoneMapperItem::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_editor();
			bone_map->connect("bone_map_updated", callable_mp(this, &BoneMapperItem::_update_property));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (!bone_map.is_null() && bone_map->is_connected("bone_map_updated", callable_mp(this, &BoneMapperItem::_update_property))) {
				bone_map->disconnect("bone_map_updated", callable_mp(this, &BoneMapperItem::_update_property));
			}
		} break;
	}
}

void BoneMapperItem::_bind_methods() {
}

BoneMapperItem::BoneMapperItem(Ref<BoneMap> &p_bone_map, PackedStringArray p_skeleton_bone_names, const StringName &p_profile_bone_name) {
	bone_map = p_bone_map;
	skeleton_bone_names = p_skeleton_bone_names;
	profile_bone_name = p_profile_bone_name;
}

BoneMapperItem::~BoneMapperItem() {
}

void BoneMapper::create_editor() {
	profile_group_selector = memnew(EditorPropertyEnum);
	profile_group_selector->set_label("Group");
	profile_group_selector->set_selectable(false);
	profile_group_selector->set_object_and_property(this, "current_group_idx");
	profile_group_selector->update_property();
	profile_group_selector->connect("property_changed", callable_mp(this, &BoneMapper::_value_changed));
	add_child(profile_group_selector);

	bone_mapper_field = memnew(AspectRatioContainer);
	bone_mapper_field->set_stretch_mode(AspectRatioContainer::STRETCH_FIT);
	bone_mapper_field->set_custom_minimum_size(Vector2(0, 256.0) * EDSCALE);
	bone_mapper_field->set_h_size_flags(Control::SIZE_FILL);
	add_child(bone_mapper_field);

	profile_bg = memnew(ColorRect);
	profile_bg->set_color(Color(0, 0, 0, 1));
	profile_bg->set_h_size_flags(Control::SIZE_FILL);
	profile_bg->set_v_size_flags(Control::SIZE_FILL);
	bone_mapper_field->add_child(profile_bg);

	profile_texture = memnew(TextureRect);
	profile_texture->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	profile_texture->set_ignore_texture_size(true);
	profile_texture->set_h_size_flags(Control::SIZE_FILL);
	profile_texture->set_v_size_flags(Control::SIZE_FILL);
	bone_mapper_field->add_child(profile_texture);

	mapper_item_vbox = memnew(VBoxContainer);
	add_child(mapper_item_vbox);

	separator = memnew(HSeparator);
	add_child(separator);

	recreate_items();
}

void BoneMapper::update_group_idx() {
	if (!bone_map->get_profile().is_valid()) {
		return;
	}

	PackedStringArray group_names;
	int len = bone_map->get_profile()->get_group_size();
	for (int i = 0; i < len; i++) {
		group_names.push_back(bone_map->get_profile()->get_group_name(i));
	}
	if (current_group_idx >= len) {
		current_group_idx = 0;
	}
	if (len > 0) {
		profile_group_selector->setup(group_names);
		profile_group_selector->update_property();
		profile_group_selector->set_read_only(false);
	}
}

void BoneMapper::set_current_group_idx(int p_group_idx) {
	current_group_idx = p_group_idx;
	recreate_editor();
}

int BoneMapper::get_current_group_idx() const {
	return current_group_idx;
}

void BoneMapper::set_current_bone_idx(int p_bone_idx) {
	current_bone_idx = p_bone_idx;
	recreate_editor();
}

int BoneMapper::get_current_bone_idx() const {
	return current_bone_idx;
}

void BoneMapper::recreate_editor() {
	// Clear buttons.
	int len = bone_mapper_buttons.size();
	for (int i = 0; i < len; i++) {
		profile_texture->remove_child(bone_mapper_buttons[i]);
		memdelete(bone_mapper_buttons[i]);
	}
	bone_mapper_buttons.clear();

	// Organize mapper items.
	len = bone_mapper_items.size();
	for (int i = 0; i < len; i++) {
		bone_mapper_items[i]->set_visible(current_bone_idx == i);
	}

	Ref<SkeletonProfile> profile = bone_map->get_profile();
	if (profile.is_valid()) {
		SkeletonProfileHumanoid *hmn = Object::cast_to<SkeletonProfileHumanoid>(profile.ptr());
		if (hmn) {
			StringName hmn_group_name = profile->get_group_name(current_group_idx);
			if (hmn_group_name == "Body") {
				profile_texture->set_texture(get_theme_icon(SNAME("BoneMapHumanBody"), SNAME("EditorIcons")));
			} else if (hmn_group_name == "Face") {
				profile_texture->set_texture(get_theme_icon(SNAME("BoneMapHumanFace"), SNAME("EditorIcons")));
			} else if (hmn_group_name == "LeftHand") {
				profile_texture->set_texture(get_theme_icon(SNAME("BoneMapHumanLeftHand"), SNAME("EditorIcons")));
			} else if (hmn_group_name == "RightHand") {
				profile_texture->set_texture(get_theme_icon(SNAME("BoneMapHumanRightHand"), SNAME("EditorIcons")));
			}
		} else {
			profile_texture->set_texture(profile->get_texture(current_group_idx));
		}
	} else {
		profile_texture->set_texture(Ref<Texture2D>());
	}

	if (!profile.is_valid()) {
		return;
	}

	for (int i = 0; i < len; i++) {
		if (profile->get_group(i) == profile->get_group_name(current_group_idx)) {
			BoneMapperButton *mb = memnew(BoneMapperButton(profile->get_bone_name(i), profile->is_require(i), current_bone_idx == i));
			mb->connect("pressed", callable_mp(this, &BoneMapper::set_current_bone_idx).bind(i), CONNECT_DEFERRED);
			mb->set_h_grow_direction(GROW_DIRECTION_BOTH);
			mb->set_v_grow_direction(GROW_DIRECTION_BOTH);
			Vector2 vc = profile->get_handle_offset(i);
			bone_mapper_buttons.push_back(mb);
			profile_texture->add_child(mb);
			mb->set_anchor(SIDE_LEFT, vc.x);
			mb->set_anchor(SIDE_RIGHT, vc.x);
			mb->set_anchor(SIDE_TOP, vc.y);
			mb->set_anchor(SIDE_BOTTOM, vc.y);
		}
	}

	_update_state();
}

void BoneMapper::clear_items() {
	// Clear items.
	int len = bone_mapper_items.size();
	for (int i = 0; i < len; i++) {
		mapper_item_vbox->remove_child(bone_mapper_items[i]);
		memdelete(bone_mapper_items[i]);
	}
	bone_mapper_items.clear();
}

void BoneMapper::recreate_items() {
	clear_items();
	// Create items by profile.
	Ref<SkeletonProfile> profile = bone_map->get_profile();
	if (profile.is_valid()) {
		PackedStringArray skeleton_bone_names;
		int len = skeleton->get_bone_count();
		for (int i = 0; i < len; i++) {
			skeleton_bone_names.push_back(skeleton->get_bone_name(i));
		}

		len = profile->get_bone_size();
		for (int i = 0; i < len; i++) {
			StringName bn = profile->get_bone_name(i);
			bone_mapper_items.append(memnew(BoneMapperItem(bone_map, skeleton_bone_names, bn)));
			mapper_item_vbox->add_child(bone_mapper_items[i]);
		}
	}

	update_group_idx();
	recreate_editor();
}

void BoneMapper::_update_state() {
	int len = bone_mapper_buttons.size();
	for (int i = 0; i < len; i++) {
		StringName pbn = bone_mapper_buttons[i]->get_profile_bone_name();
		StringName sbn = bone_map->get_skeleton_bone_name(pbn);
		int bone_idx = skeleton->find_bone(sbn);
		if (bone_idx >= 0) {
			if (bone_map->get_skeleton_bone_name_count(sbn) == 1) {
				Ref<SkeletonProfile> prof = bone_map->get_profile();

				StringName parent_name = prof->get_bone_parent(prof->find_bone(pbn));
				Vector<int> prof_parent_bones;
				while (parent_name != StringName()) {
					prof_parent_bones.push_back(skeleton->find_bone(bone_map->get_skeleton_bone_name(parent_name)));
					if (prof->find_bone(parent_name) == -1) {
						break;
					}
					parent_name = prof->get_bone_parent(prof->find_bone(parent_name));
				}

				int parent_id = skeleton->get_bone_parent(bone_idx);
				Vector<int> skel_parent_bones;
				while (parent_id >= 0) {
					skel_parent_bones.push_back(parent_id);
					parent_id = skeleton->get_bone_parent(parent_id);
				}

				bool is_broken = false;
				for (int j = 0; j < prof_parent_bones.size(); j++) {
					if (prof_parent_bones[j] != -1 && !skel_parent_bones.has(prof_parent_bones[j])) {
						is_broken = true;
					}
				}

				if (is_broken) {
					bone_mapper_buttons[i]->set_state(BoneMapperButton::BONE_MAP_STATE_ERROR);
				} else {
					bone_mapper_buttons[i]->set_state(BoneMapperButton::BONE_MAP_STATE_SET);
				}
			} else {
				bone_mapper_buttons[i]->set_state(BoneMapperButton::BONE_MAP_STATE_ERROR);
			}
		} else {
			if (bone_mapper_buttons[i]->is_require()) {
				bone_mapper_buttons[i]->set_state(BoneMapperButton::BONE_MAP_STATE_MISSING);
			} else {
				bone_mapper_buttons[i]->set_state(BoneMapperButton::BONE_MAP_STATE_UNSET);
			}
		}
	}
}

void BoneMapper::_value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	set(p_property, p_value);
	recreate_editor();
}

void BoneMapper::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_current_group_idx", "current_group_idx"), &BoneMapper::set_current_group_idx);
	ClassDB::bind_method(D_METHOD("get_current_group_idx"), &BoneMapper::get_current_group_idx);
	ClassDB::bind_method(D_METHOD("set_current_bone_idx", "current_bone_idx"), &BoneMapper::set_current_bone_idx);
	ClassDB::bind_method(D_METHOD("get_current_bone_idx"), &BoneMapper::get_current_bone_idx);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_group_idx"), "set_current_group_idx", "get_current_group_idx");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_bone_idx"), "set_current_bone_idx", "get_current_bone_idx");
}

void BoneMapper::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_editor();
			bone_map->connect("bone_map_updated", callable_mp(this, &BoneMapper::_update_state));
			bone_map->connect("profile_updated", callable_mp(this, &BoneMapper::recreate_items));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			clear_items();
			if (!bone_map.is_null()) {
				if (bone_map->is_connected("bone_map_updated", callable_mp(this, &BoneMapper::_update_state))) {
					bone_map->disconnect("bone_map_updated", callable_mp(this, &BoneMapper::_update_state));
				}
				if (bone_map->is_connected("profile_updated", callable_mp(this, &BoneMapper::recreate_items))) {
					bone_map->disconnect("profile_updated", callable_mp(this, &BoneMapper::recreate_items));
				}
			}
		}
	}
}

BoneMapper::BoneMapper(Skeleton3D *p_skeleton, Ref<BoneMap> &p_bone_map) {
	skeleton = p_skeleton;
	bone_map = p_bone_map;
}

BoneMapper::~BoneMapper() {
}

void BoneMapEditor::create_editors() {
	if (!skeleton) {
		return;
	}
	bone_mapper = memnew(BoneMapper(skeleton, bone_map));
	add_child(bone_mapper);
}

void BoneMapEditor::fetch_objects() {
	skeleton = nullptr;
	// Hackey... but it may be the easist way to get a selected object from "ImporterScene".
	SceneImportSettings *si = SceneImportSettings::get_singleton();
	if (!si) {
		return;
	}
	if (!si->is_visible()) {
		return;
	}
	Node *selected = si->get_selected_node();
	if (selected) {
		Skeleton3D *sk = Object::cast_to<Skeleton3D>(selected);
		if (!sk) {
			return;
		}
		skeleton = sk;
	} else {
		// Editor should not exist.
		skeleton = nullptr;
	}
}

void BoneMapEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			fetch_objects();
			create_editors();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (bone_mapper) {
				remove_child(bone_mapper);
				bone_mapper->queue_delete();
			}
			skeleton = nullptr;
		} break;
	}
}

BoneMapEditor::BoneMapEditor(Ref<BoneMap> &p_bone_map) {
	bone_map = p_bone_map;
}

BoneMapEditor::~BoneMapEditor() {
}

bool EditorInspectorPluginBoneMap::can_handle(Object *p_object) {
	return Object::cast_to<BoneMap>(p_object) != nullptr;
}

void EditorInspectorPluginBoneMap::parse_begin(Object *p_object) {
	BoneMap *bm = Object::cast_to<BoneMap>(p_object);
	if (!bm) {
		return;
	}
	Ref<BoneMap> r(bm);
	editor = memnew(BoneMapEditor(r));
	add_custom_control(editor);
}

BoneMapEditorPlugin::BoneMapEditorPlugin() {
	// Register properties in editor settings.
	EDITOR_DEF("editors/bone_mapper/handle_colors/unset", Color(0.3, 0.3, 0.3));
	EDITOR_DEF("editors/bone_mapper/handle_colors/set", Color(0.1, 0.6, 0.25));
	EDITOR_DEF("editors/bone_mapper/handle_colors/missing", Color(0.8, 0.2, 0.8));
	EDITOR_DEF("editors/bone_mapper/handle_colors/error", Color(0.8, 0.2, 0.2));

	Ref<EditorInspectorPluginBoneMap> inspector_plugin;
	inspector_plugin.instantiate();
	add_inspector_plugin(inspector_plugin);

	Ref<PostImportPluginSkeletonTrackOrganizer> post_import_plugin_track_organizer;
	post_import_plugin_track_organizer.instantiate();
	add_scene_post_import_plugin(post_import_plugin_track_organizer);

	Ref<PostImportPluginSkeletonRenamer> post_import_plugin_renamer;
	post_import_plugin_renamer.instantiate();
	add_scene_post_import_plugin(post_import_plugin_renamer);

	Ref<PostImportPluginSkeletonRestFixer> post_import_plugin_rest_fixer;
	post_import_plugin_rest_fixer.instantiate();
	add_scene_post_import_plugin(post_import_plugin_rest_fixer);
}

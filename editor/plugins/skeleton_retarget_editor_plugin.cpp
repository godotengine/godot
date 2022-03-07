/*************************************************************************/
/*  skeleton_retarget_editor_plugin.cpp                                  */
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

#include "skeleton_retarget_editor_plugin.h"

#include "editor/editor_scale.h"
#include "scene/animation/animation_player.h"

// Base classes

void RetargetEditorForm::create_editors() {
}

void RetargetEditorForm::submit() {
}

void RetargetEditorForm::create_button_submit() {
	button_submit = memnew(Button);
	button_submit->set_text(TTR("Append Item"));
	button_submit->connect("pressed", callable_mp(this, &RetargetEditorForm::submit));
	add_child(button_submit);
}

void RetargetEditorForm::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_editors();
			create_button_submit();
		} break;
	}
}

RetargetEditorForm::RetargetEditorForm() {
}

RetargetEditorForm::~RetargetEditorForm() {
}

VBoxContainer *RetargetEditorItem::get_vbox() {
	return vbox;
}

void RetargetEditorItem::create_editors() {
	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);

	Label *label = memnew(Label);
	label->set_text(itos(index));
	hb->add_child(label);
	vbox = memnew(VBoxContainer);
	vbox->set_h_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(vbox);
	button_remove = memnew(Button);
	button_remove->set_icon(get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
	button_remove->connect("pressed", callable_mp(this, &RetargetEditorItem::fire_remove));
	hb->add_child(button_remove);

	HSeparator *separator = memnew(HSeparator);
	add_child(separator);
}

void RetargetEditorItem::fire_remove() {
	emit_signal("remove", index);
}

void RetargetEditorItem::_bind_methods() {
	ADD_SIGNAL(MethodInfo("remove", PropertyInfo(Variant::INT, "index")));
}

void RetargetEditorItem::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_editors();
		} break;
	}
}

RetargetEditorItem::RetargetEditorItem(const int p_index) {
	index = p_index;
}

RetargetEditorItem::~RetargetEditorItem() {
}

// Mapper base

void MapperButton::fetch_textures() {
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

	set_state(state);
}

void MapperButton::set_state(MapperState p_state) {
	state = p_state;
	switch (state) {
		case MAPPER_STATE_UNSET: {
			circle->set_modulate(EditorSettings::get_singleton()->get("editors/retarget_mapper/button_colors/unset"));
		} break;
		case MAPPER_STATE_SET: {
			circle->set_modulate(EditorSettings::get_singleton()->get("editors/retarget_mapper/button_colors/set"));
		} break;
		case MAPPER_STATE_ERROR: {
			circle->set_modulate(EditorSettings::get_singleton()->get("editors/retarget_mapper/button_colors/error"));
		} break;
		default: {
		} break;
	}
}

StringName MapperButton::get_name() {
	return name;
}

void MapperButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			fetch_textures();
		} break;
	}
}

MapperButton::MapperButton(const StringName &p_name, bool p_selected, MapperState p_state) {
	name = p_name;
	selected = p_selected;
	state = p_state;
}

MapperButton::~MapperButton() {
}

void RetargetEditorMapperItem::create_editors() {
}

void RetargetEditorMapperItem::create_buttons() {
	HBoxContainer *label_box = memnew(HBoxContainer);
	label_box->add_theme_color_override("background_color", Color(0.5, 0.5, 0.5));
	add_child(label_box);
	Label *label = memnew(Label);
	label->set_text("[" + String(key_name) + "]");
	label->set_h_size_flags(SIZE_EXPAND_FILL);
	label_box->add_child(label);

	button_enable = memnew(Button);
	button_enable->set_text(TTR("Register"));
	button_enable->connect("pressed", callable_mp(this, &RetargetEditorMapperItem::fire_enable));

	button_remove = memnew(Button);
	button_remove->set_icon(get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
	button_remove->connect("pressed", callable_mp(this, &RetargetEditorMapperItem::fire_remove));

	inputs_vbox = memnew(VBoxContainer);

	if (enabled) {
		button_enable->set_visible(false);
	} else {
		button_remove->set_visible(false);
		inputs_vbox->set_visible(false);
	}

	label_box->add_child(button_remove);
	label_box->add_child(button_enable);
	add_child(inputs_vbox);

	HSeparator *separator = memnew(HSeparator);
	add_child(separator);
}

void RetargetEditorMapperItem::fire_remove() {
	emit_signal("remove", key_name);
}

void RetargetEditorMapperItem::fire_enable() {
	emit_signal("enable", key_name);
}

void RetargetEditorMapperItem::assign_button_id(int p_button_id) {
	button_id = p_button_id;
}

void RetargetEditorMapperItem::_value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
}

void RetargetEditorMapperItem::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_buttons();
			create_editors();
		} break;
	}
}

void RetargetEditorMapperItem::_bind_methods() {
	ADD_SIGNAL(MethodInfo("remove", PropertyInfo(Variant::STRING_NAME, "intermediate_bone_name")));
	ADD_SIGNAL(MethodInfo("enable", PropertyInfo(Variant::STRING_NAME, "intermediate_bone_name")));
	ADD_SIGNAL(MethodInfo("pick", PropertyInfo(Variant::STRING_NAME, "intermediate_bone_name")));
}

RetargetEditorMapperItem::RetargetEditorMapperItem(const StringName &p_name, const bool p_enabled) {
	key_name = p_name;
	enabled = p_enabled;
}

RetargetEditorMapperItem::~RetargetEditorMapperItem() {
}

void RetargetEditorMapper::create_rich_editor() {
	if (!use_rich_profile) {
		return;
	}

	profile_group_selector = memnew(EditorPropertyEnum);
	profile_group_selector->set_label("Group");
	profile_group_selector->set_selectable(false);
	profile_group_selector->set_object_and_property(this, "current_group");
	profile_group_selector->update_property();
	profile_group_selector->connect("property_changed", callable_mp(this, &RetargetEditorMapper::_value_changed));
	add_child(profile_group_selector);

	rich_profile_field = memnew(AspectRatioContainer);
	rich_profile_field->set_stretch_mode(AspectRatioContainer::STRETCH_FIT);
	rich_profile_field->set_custom_minimum_size(Vector2(0, 256.0) * EDSCALE);
	rich_profile_field->set_h_size_flags(Control::SIZE_FILL);
	add_child(rich_profile_field);

	profile_bg = memnew(ColorRect);
	profile_bg->set_color(Color(0, 0, 0, 1));
	profile_bg->set_h_size_flags(Control::SIZE_FILL);
	profile_bg->set_v_size_flags(Control::SIZE_FILL);
	rich_profile_field->add_child(profile_bg);

	profile_texture = memnew(TextureRect);
	profile_texture->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	profile_texture->set_ignore_texture_size(true);
	profile_texture->set_h_size_flags(Control::SIZE_FILL);
	profile_texture->set_v_size_flags(Control::SIZE_FILL);
	rich_profile_field->add_child(profile_texture);
}

void RetargetEditorMapper::update_group_ids() {
	if (!use_rich_profile) {
		return;
	}

	Ref<RetargetRichProfile> rich_profile = static_cast<Ref<RetargetRichProfile>>(profile);
	PackedStringArray group_names;
	int len = rich_profile->get_groups_size();
	for (int i = 0; i < len; i++) {
		group_names.push_back(rich_profile->get_group_name(i));
	}
	if (current_group >= len) {
		current_group = 0;
	}
	if (len > 0) {
		profile_group_selector->setup(group_names);
		profile_group_selector->update_property();
		profile_group_selector->set_read_only(false);
	} else {
		group_names.push_back("--");
		profile_group_selector->setup(group_names);
		profile_group_selector->set_read_only(true);
	}
}

void RetargetEditorMapper::set_current_group(int p_group) {
	current_group = p_group;
	recreate_rich_editor();
}

int RetargetEditorMapper::get_current_group() const {
	return current_group;
}

void RetargetEditorMapper::set_current_intermediate_bone(int p_bone) {
	current_intermediate_bone = p_bone;
	recreate_rich_editor();
}

int RetargetEditorMapper::get_current_intermediate_bone() const {
	return current_intermediate_bone;
}

void RetargetEditorMapper::recreate_rich_editor() {
	if (!use_rich_profile) {
		return;
	}

	// Clear buttons.
	int len = mapper_buttons.size();
	for (int i = 0; i < len; i++) {
		profile_texture->remove_child(mapper_buttons[i]);
		memdelete(mapper_buttons[i]);
	}
	mapper_buttons.clear();

	// Organize mapper items.
	len = mapper_items.size();
	for (int i = 0; i < len; i++) {
		mapper_items[i]->set_visible(current_intermediate_bone == i);
	}

	Ref<RetargetRichProfile> rich_profile = static_cast<Ref<RetargetRichProfile>>(profile);
	if (rich_profile->get_groups_size() > 0) {
		profile_texture->set_texture(rich_profile->get_group_texture(current_group));
	} else {
		profile_texture->set_texture(Ref<Texture2D>());
	}

	int j = 0;
	for (int i = 0; i < len; i++) {
		if (rich_profile->get_intermediate_bone_group_id(i) == current_group) {
			MapperButton *mb = memnew(MapperButton(rich_profile->get_intermediate_bone_name(i), current_intermediate_bone == i, get_mapper_state(rich_profile->get_intermediate_bone_name(i))));
			mb->connect("pressed", callable_mp(this, &RetargetEditorMapper::set_current_intermediate_bone), varray(i), CONNECT_DEFERRED);
			mb->set_h_grow_direction(GROW_DIRECTION_BOTH);
			mb->set_v_grow_direction(GROW_DIRECTION_BOTH);
			Vector2 vc = rich_profile->get_intermediate_bone_handle_offset(i);
			mapper_buttons.push_back(mb);
			profile_texture->add_child(mb);
			mb->set_anchor(SIDE_LEFT, vc.x);
			mb->set_anchor(SIDE_RIGHT, vc.x);
			mb->set_anchor(SIDE_TOP, vc.y);
			mb->set_anchor(SIDE_BOTTOM, vc.y);
			mapper_items[i]->assign_button_id(j);
			j++;
		}
	}
}

MapperButton::MapperState RetargetEditorMapper::get_mapper_state(const StringName &p_bone_name) {
	return MapperButton::MAPPER_STATE_UNSET;
}

void RetargetEditorMapper::set_mapper_state(int p_bone, MapperButton::MapperState p_state) {
	ERR_FAIL_INDEX(p_bone, mapper_buttons.size());
	mapper_buttons[p_bone]->set_state(p_state);
}

void RetargetEditorMapper::create_editors() {
	create_rich_editor();

	map_vbox = memnew(VBoxContainer);
	add_child(map_vbox);

	const Color section_color = get_theme_color(SNAME("prop_subsection"), SNAME("Editor"));
	section_unprofiled = memnew(EditorInspectorSection);
	section_unprofiled->setup("unprofiled_bones", "Unprofiled Bones", this, section_color, true);
	add_child(section_unprofiled);

	unprofiled_vbox = memnew(VBoxContainer);
	section_unprofiled->get_vbox()->add_child(unprofiled_vbox);

	recreate_items();
}

void RetargetEditorMapper::set_profile(const Ref<RetargetProfile> &p_profile) {
	profile = p_profile;
	Ref<RetargetRichProfile> rrp = static_cast<Ref<RetargetRichProfile>>(profile);
	use_rich_profile = rrp.is_valid();
}

void RetargetEditorMapper::clear_items() {
}

void RetargetEditorMapper::recreate_items() {
}

void RetargetEditorMapper::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_current_group", "current_group"), &RetargetEditorMapper::set_current_group);
	ClassDB::bind_method(D_METHOD("get_current_group"), &RetargetEditorMapper::get_current_group);
	ClassDB::bind_method(D_METHOD("set_current_intermediate_bone", "current_intermediate_bone"), &RetargetEditorMapper::set_current_intermediate_bone);
	ClassDB::bind_method(D_METHOD("get_current_intermediate_bone"), &RetargetEditorMapper::get_current_intermediate_bone);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_group"), "set_current_group", "get_current_group");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_intermediate_bone"), "set_current_intermediate_bone", "get_current_intermediate_bone");
}

void RetargetEditorMapper::_value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	set(p_property, p_value);
	recreate_rich_editor();
}

void RetargetEditorMapper::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_editors();
		} break;
	}
}

RetargetEditorMapper::RetargetEditorMapper() {
}

RetargetEditorMapper::~RetargetEditorMapper() {
}

// Bone picker

void Skeleton3DBonePicker::create_editors() {
	set_title(TTR("Bone Picker:"));

	VBoxContainer *vbox = memnew(VBoxContainer);
	add_child(vbox);

	bones = memnew(Tree);
	bones->set_select_mode(Tree::SELECT_SINGLE);
	bones->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	bones->set_hide_root(true);
	vbox->add_child(bones);
}

void Skeleton3DBonePicker::create_bones_tree(Skeleton3D *p_skeleton) {
	bones->clear();

	if (!p_skeleton) {
		return;
	}

	TreeItem *root = bones->create_item();

	Map<int, TreeItem *> items;

	items.insert(-1, root);

	Ref<Texture> bone_icon = get_theme_icon(SNAME("BoneAttachment3D"), SNAME("EditorIcons"));

	Vector<int> bones_to_process = p_skeleton->get_parentless_bones();
	while (bones_to_process.size() > 0) {
		int current_bone_idx = bones_to_process[0];
		bones_to_process.erase(current_bone_idx);

		const int parent_idx = p_skeleton->get_bone_parent(current_bone_idx);
		TreeItem *parent_item = items.find(parent_idx)->get();

		TreeItem *joint_item = bones->create_item(parent_item);
		items.insert(current_bone_idx, joint_item);

		joint_item->set_text(0, p_skeleton->get_bone_name(current_bone_idx));
		joint_item->set_icon(0, bone_icon);
		joint_item->set_selectable(0, true);
		joint_item->set_metadata(0, "bones/" + itos(current_bone_idx));

		// Add the bone's children to the list of bones to be processed.
		Vector<int> current_bone_child_bones = p_skeleton->get_bone_children(current_bone_idx);
		int child_bone_size = current_bone_child_bones.size();
		for (int i = 0; i < child_bone_size; i++) {
			bones_to_process.push_back(current_bone_child_bones[i]);
		}
	}
}

void Skeleton3DBonePicker::popup_bones_tree(Skeleton3D *p_skeleton, const Size2i &p_minsize) {
	create_bones_tree(p_skeleton);
	popup_centered(p_minsize);
}

bool Skeleton3DBonePicker::has_selected_bone() {
	TreeItem *selected = bones->get_selected();
	if (!selected) {
		return false;
	}
	return true;
}

StringName Skeleton3DBonePicker::get_selected_bone() {
	TreeItem *selected = bones->get_selected();
	if (!selected) {
		return StringName();
	}
	return selected->get_text(0);
}

void Skeleton3DBonePicker::_bind_methods() {
}

void Skeleton3DBonePicker::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_editors();
		} break;
	}
}

Skeleton3DBonePicker::Skeleton3DBonePicker() {
}

Skeleton3DBonePicker::~Skeleton3DBonePicker() {
}

// Retarget profile

void RetargetProfileEditorForm::create_editors() {
	key_name = memnew(EditorPropertyText);
	key_name->set_label("Name");
	key_name->set_selectable(false);
	key_name->set_object_and_property(this, "key_name");
	key_name->update_property();
	key_name->connect("property_changed", callable_mp(this, &RetargetProfileEditorForm::_value_changed));
	add_child(key_name);
}

void RetargetProfileEditorForm::submit() {
	emit_signal("submit", prop_key_name);
	set("key_name", StringName()); // Initialize.
	key_name->update_property();
}

void RetargetProfileEditorForm::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_key_name", "key_name"), &RetargetProfileEditorForm::set_key_name);
	ClassDB::bind_method(D_METHOD("get_key_name"), &RetargetProfileEditorForm::get_key_name);
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "key_name"), "set_key_name", "get_key_name");
	ADD_SIGNAL(MethodInfo("submit", PropertyInfo(Variant::STRING_NAME, "key_name")));
}

void RetargetProfileEditorForm::_value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	set(p_property, p_value);
}

void RetargetProfileEditorForm::set_key_name(const StringName &p_key_name) {
	prop_key_name = p_key_name;
}

StringName RetargetProfileEditorForm::get_key_name() const {
	return prop_key_name;
}

RetargetProfileEditorForm::RetargetProfileEditorForm() {
}

RetargetProfileEditorForm::~RetargetProfileEditorForm() {
}

void RetargetProfileEditor::create_editors() {
	imb_vbox = memnew(VBoxContainer);
	add_child(imb_vbox);

	imb_form = memnew(RetargetProfileEditorForm);
	imb_form->connect("submit", callable_mp(this, &RetargetProfileEditor::_add_intermediate_bone));
	add_child(imb_form);

	recreate_items();
}

void RetargetProfileEditor::recreate_items() {
	clear_items();

	// Create items.
	int len = retarget_profile->get_intermediate_bones_size();
	for (int i = 0; i < len; i++) {
		intermediate_bones.append(memnew(RetargetEditorItem(i)));
		intermediate_bones[i]->connect("remove", callable_mp(this, &RetargetProfileEditor::_remove_intermediate_bone), varray(), CONNECT_DEFERRED);
		imb_vbox->add_child(intermediate_bones[i]);

		String prep = "intermediate_bones/" + itos(i) + "/";
		intermediate_bone_names.append(memnew(EditorPropertyText()));
		intermediate_bone_names[i]->set_label("Bone Name");
		intermediate_bone_names[i]->set_selectable(false);
		intermediate_bone_names[i]->set_object_and_property(retarget_profile, prep + "bone_name");
		intermediate_bone_names[i]->update_property();
		intermediate_bone_names[i]->connect("property_changed", callable_mp(this, &RetargetProfileEditor::_value_changed), varray(), CONNECT_DEFERRED);
		intermediate_bones[i]->get_vbox()->add_child(intermediate_bone_names[i]);
	}
}

void RetargetProfileEditor::clear_items() {
	// Clear items.
	int len = intermediate_bones.size();
	for (int i = 0; i < len; i++) {
		intermediate_bone_names[i]->disconnect("property_changed", callable_mp(this, &RetargetProfileEditor::_value_changed));
		intermediate_bones[i]->get_vbox()->remove_child(intermediate_bone_names[i]);
		memdelete(intermediate_bone_names[i]);
		imb_vbox->remove_child(intermediate_bones[i]);
		memdelete(intermediate_bones[i]);
	}
	intermediate_bone_names.clear();
	intermediate_bones.clear();
}

void RetargetProfileEditor::_value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	ERR_FAIL_COND(!retarget_profile);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Retarget Profile Property"), UndoRedo::MERGE_ENDS);
	ur->add_undo_property(retarget_profile, p_property, retarget_profile->get(p_property));
	ur->add_do_property(retarget_profile, p_property, p_value);
	ur->commit_action();
}

void RetargetProfileEditor::_update_intermediate_bone_property() {
	int len = intermediate_bone_names.size();
	for (int i = 0; i < len; i++) {
		if (intermediate_bone_names[i]->get_edited_object() && intermediate_bone_names[i]->get_edited_property()) {
			intermediate_bone_names[i]->update_property();
		}
	}
}

void RetargetProfileEditor::_add_intermediate_bone(const StringName &p_bone_name) {
	ERR_FAIL_COND(!retarget_profile);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Add Intermediate Bone"));
	ur->add_undo_method(retarget_profile, "remove_intermediate_bone", retarget_profile->get_intermediate_bones_size());
	ur->add_do_method(retarget_profile, "add_intermediate_bone", p_bone_name);
	ur->commit_action();
}

void RetargetProfileEditor::_remove_intermediate_bone(const int p_id) {
	ERR_FAIL_COND(!retarget_profile);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Remove Intermediate Bone"));
	ur->add_undo_method(retarget_profile, "add_intermediate_bone", retarget_profile->get_intermediate_bone_name(p_id), p_id);
	ur->add_do_method(retarget_profile, "remove_intermediate_bone", p_id);
	ur->commit_action();
}

void RetargetProfileEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_editors();
		} break;
		case NOTIFICATION_POST_ENTER_TREE: {
			retarget_profile->connect("intermediate_bone_updated", callable_mp(this, &RetargetProfileEditor::_update_intermediate_bone_property));
			retarget_profile->connect("redraw_needed", callable_mp(this, &RetargetProfileEditor::recreate_items));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (retarget_profile) {
				if (retarget_profile->is_connected("intermediate_bone_updated", callable_mp(this, &RetargetProfileEditor::_update_intermediate_bone_property))) {
					retarget_profile->disconnect("intermediate_bone_updated", callable_mp(this, &RetargetProfileEditor::_update_intermediate_bone_property));
				}
				if (retarget_profile->is_connected("redraw_needed", callable_mp(this, &RetargetProfileEditor::recreate_items))) {
					retarget_profile->disconnect("redraw_needed", callable_mp(this, &RetargetProfileEditor::recreate_items));
				}
			}
		} break;
	}
}

RetargetProfileEditor::RetargetProfileEditor(RetargetProfile *p_retarget_profile) {
	retarget_profile = p_retarget_profile;
}

RetargetProfileEditor::~RetargetProfileEditor() {
}

void RetargetRichProfileEditor::create_editors() {
	const Color section_color = get_theme_color(SNAME("prop_subsection"), SNAME("Editor"));

	// Group settings.
	section_grp = memnew(EditorInspectorSection);
	section_grp->setup("groups", "Groups", this, section_color, true);
	add_child(section_grp);

	grp_vbox = memnew(VBoxContainer);
	section_grp->get_vbox()->add_child(grp_vbox);

	grp_form = memnew(RetargetProfileEditorForm);
	grp_form->connect("submit", callable_mp(this, &RetargetRichProfileEditor::_add_group));
	section_grp->get_vbox()->add_child(grp_form);

	// Intermediate bones.
	section_imb = memnew(EditorInspectorSection);
	section_imb->setup("intermediate_bones", "Intermediate Bones", this, section_color, true);
	add_child(section_imb);

	imb_vbox = memnew(VBoxContainer);
	section_imb->get_vbox()->add_child(imb_vbox);

	imb_form = memnew(RetargetProfileEditorForm);
	imb_form->connect("submit", callable_mp(this, &RetargetRichProfileEditor::_add_intermediate_bone));
	section_imb->get_vbox()->add_child(imb_form);

	recreate_items();
}

void RetargetRichProfileEditor::recreate_items() {
	clear_items();

	// Create items.
	// Group settings.
	PackedStringArray group_names_array = PackedStringArray();
	PackedStringArray dummy_group_names_array = PackedStringArray();
	int grp_len = retarget_profile->get_groups_size();
	for (int i = 0; i < grp_len; i++) {
		groups.append(memnew(RetargetEditorItem(i)));
		groups[i]->connect("remove", callable_mp(this, &RetargetRichProfileEditor::_remove_group), varray(), CONNECT_DEFERRED);
		grp_vbox->add_child(groups[i]);

		String prep = "groups/" + itos(i) + "/";
		group_names.append(memnew(EditorPropertyText()));
		group_names[i]->set_label("Group Name");
		group_names[i]->set_selectable(false);
		group_names[i]->set_object_and_property(retarget_profile, prep + "group_name");
		group_names[i]->update_property();
		group_names[i]->connect("property_changed", callable_mp(this, &RetargetRichProfileEditor::_value_changed), varray(), CONNECT_DEFERRED);
		groups[i]->get_vbox()->add_child(group_names[i]);
		group_textures.append(memnew(EditorPropertyResource()));
		group_textures[i]->setup(retarget_profile, prep + "group_texture", "Texture2D");
		group_textures[i]->set_label("Group Texture");
		group_textures[i]->set_selectable(false);
		group_textures[i]->set_object_and_property(retarget_profile, prep + "group_texture");
		group_textures[i]->update_property();
		group_textures[i]->connect("property_changed", callable_mp(this, &RetargetRichProfileEditor::_value_changed), varray(), CONNECT_DEFERRED);
		groups[i]->get_vbox()->add_child(group_textures[i]);

		group_names_array.push_back(retarget_profile->get_group_name(i));
	}

	bool has_group = grp_len > 0;
	if (!has_group) {
		dummy_group_names_array.push_back("--");
	}

	// Intermediate bones.
	int imb_len = retarget_profile->get_intermediate_bones_size();
	for (int i = 0; i < imb_len; i++) {
		intermediate_bones.append(memnew(RetargetEditorItem(i)));
		intermediate_bones[i]->connect("remove", callable_mp(this, &RetargetRichProfileEditor::_remove_intermediate_bone), varray(), CONNECT_DEFERRED);
		imb_vbox->add_child(intermediate_bones[i]);

		String prep = "intermediate_bones/" + itos(i) + "/";
		intermediate_bone_names.append(memnew(EditorPropertyText()));
		intermediate_bone_names[i]->set_label("Bone Name");
		intermediate_bone_names[i]->set_selectable(false);
		intermediate_bone_names[i]->set_object_and_property(retarget_profile, prep + "bone_name");
		intermediate_bone_names[i]->update_property();
		intermediate_bone_names[i]->connect("property_changed", callable_mp(this, &RetargetRichProfileEditor::_value_changed), varray(), CONNECT_DEFERRED);
		intermediate_bones[i]->get_vbox()->add_child(intermediate_bone_names[i]);
		intermediate_bone_handle_offsets.append(memnew(EditorPropertyVector2()));
		intermediate_bone_handle_offsets[i]->setup(0.0, 1.0, 0.001, false);
		intermediate_bone_handle_offsets[i]->set_label("Handle Offset");
		intermediate_bone_handle_offsets[i]->set_selectable(false);
		intermediate_bone_handle_offsets[i]->set_object_and_property(retarget_profile, prep + "handle_offset");
		intermediate_bone_handle_offsets[i]->update_property();
		intermediate_bone_handle_offsets[i]->connect("property_changed", callable_mp(this, &RetargetRichProfileEditor::_value_changed), varray(), CONNECT_DEFERRED);
		intermediate_bones[i]->get_vbox()->add_child(intermediate_bone_handle_offsets[i]);
		intermediate_bone_group_ids.append(memnew(EditorPropertyEnum()));
		intermediate_bone_group_ids[i]->set_label("Group");
		intermediate_bone_group_ids[i]->set_selectable(false);
		if (has_group) {
			intermediate_bone_group_ids[i]->setup(group_names_array);
			intermediate_bone_group_ids[i]->set_object_and_property(retarget_profile, prep + "group_id");
			intermediate_bone_group_ids[i]->update_property();
		} else {
			intermediate_bone_group_ids[i]->setup(dummy_group_names_array);
			intermediate_bone_group_ids[i]->set_read_only(true);
		}
		intermediate_bone_group_ids[i]->connect("property_changed", callable_mp(this, &RetargetRichProfileEditor::_value_changed), varray(), CONNECT_DEFERRED);
		intermediate_bones[i]->get_vbox()->add_child(intermediate_bone_group_ids[i]);
	}
}

void RetargetRichProfileEditor::clear_items() {
	// Clear items.
	// Group settings.
	int len = groups.size();
	for (int i = 0; i < len; i++) {
		group_names[i]->disconnect("property_changed", callable_mp(this, &RetargetRichProfileEditor::_value_changed));
		groups[i]->get_vbox()->remove_child(group_names[i]);
		memdelete(group_names[i]);
		group_textures[i]->disconnect("property_changed", callable_mp(this, &RetargetRichProfileEditor::_value_changed));
		groups[i]->get_vbox()->remove_child(group_textures[i]);
		memdelete(group_textures[i]);
		grp_vbox->remove_child(groups[i]);
		memdelete(groups[i]);
	}
	group_names.clear();
	group_textures.clear();
	groups.clear();
	// Intermediate bones.
	len = intermediate_bones.size();
	for (int i = 0; i < len; i++) {
		intermediate_bone_names[i]->disconnect("property_changed", callable_mp(this, &RetargetRichProfileEditor::_value_changed));
		intermediate_bones[i]->get_vbox()->remove_child(intermediate_bone_names[i]);
		memdelete(intermediate_bone_names[i]);
		intermediate_bone_handle_offsets[i]->disconnect("property_changed", callable_mp(this, &RetargetRichProfileEditor::_value_changed));
		intermediate_bones[i]->get_vbox()->remove_child(intermediate_bone_handle_offsets[i]);
		memdelete(intermediate_bone_handle_offsets[i]);
		intermediate_bone_group_ids[i]->disconnect("property_changed", callable_mp(this, &RetargetRichProfileEditor::_value_changed));
		intermediate_bones[i]->get_vbox()->remove_child(intermediate_bone_group_ids[i]);
		memdelete(intermediate_bone_group_ids[i]);
		imb_vbox->remove_child(intermediate_bones[i]);
		memdelete(intermediate_bones[i]);
	}
	intermediate_bone_names.clear();
	intermediate_bone_handle_offsets.clear();
	intermediate_bone_group_ids.clear();
	intermediate_bones.clear();
}

void RetargetRichProfileEditor::_update_group_ids() {
	ERR_FAIL_COND(!retarget_profile);
	PackedStringArray group_names_array = PackedStringArray();
	int len = retarget_profile->get_groups_size();
	for (int i = 0; i < len; i++) {
		group_names_array.push_back(retarget_profile->get_group_name(i));
	}
	if (len > 0) {
		len = intermediate_bone_group_ids.size();
		for (int i = 0; i < len; i++) {
			intermediate_bone_group_ids[i]->setup(group_names_array);
			intermediate_bone_group_ids[i]->set_read_only(false);
		}
	} else {
		group_names_array.push_back("--");
		len = intermediate_bone_group_ids.size();
		for (int i = 0; i < len; i++) {
			intermediate_bone_group_ids[i]->setup(group_names_array);
			intermediate_bone_group_ids[i]->set_read_only(true);
		}
	}
}

void RetargetRichProfileEditor::_value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	ERR_FAIL_COND(!retarget_profile);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Retarget Profile Property"), UndoRedo::MERGE_ENDS);
	ur->add_undo_property(retarget_profile, p_property, retarget_profile->get(p_property));
	ur->add_do_property(retarget_profile, p_property, p_value);
	ur->commit_action();
}

void RetargetRichProfileEditor::_update_group_property() {
	int len = group_names.size();
	for (int i = 0; i < len; i++) {
		if (group_names[i]->get_edited_object() && group_names[i]->get_edited_property()) {
			group_names[i]->update_property();
		}
		if (group_textures[i]->get_edited_object() && group_textures[i]->get_edited_property()) {
			group_textures[i]->update_property();
		}
	}
	_update_group_ids();
}

void RetargetRichProfileEditor::_update_intermediate_bone_property() {
	int len = intermediate_bone_names.size();
	for (int i = 0; i < len; i++) {
		if (intermediate_bone_names[i]->get_edited_object() && intermediate_bone_names[i]->get_edited_property()) {
			intermediate_bone_names[i]->update_property();
		}
		if (intermediate_bone_handle_offsets[i]->get_edited_object() && intermediate_bone_handle_offsets[i]->get_edited_property()) {
			intermediate_bone_handle_offsets[i]->update_property();
		}
		if (intermediate_bone_group_ids[i]->get_edited_object() && intermediate_bone_group_ids[i]->get_edited_property()) {
			intermediate_bone_group_ids[i]->update_property();
		}
	}
}

void RetargetRichProfileEditor::_add_intermediate_bone(const StringName &p_bone_name) {
	ERR_FAIL_COND(!retarget_profile);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Add Intermediate Bone"));
	ur->add_undo_method(retarget_profile, "remove_intermediate_bone", retarget_profile->get_intermediate_bones_size());
	ur->add_do_method(retarget_profile, "add_intermediate_bone", p_bone_name);
	ur->commit_action();
}

void RetargetRichProfileEditor::_remove_intermediate_bone(const int p_id) {
	ERR_FAIL_COND(!retarget_profile);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Remove Intermediate Bone"));
	ur->add_undo_method(retarget_profile, "add_intermediate_bone", retarget_profile->get_intermediate_bone_name(p_id), p_id);
	ur->add_undo_method(retarget_profile, "set_intermediate_bone_handle_offset", p_id, retarget_profile->get_intermediate_bone_handle_offset(p_id));
	ur->add_undo_method(retarget_profile, "set_intermediate_bone_group_id", p_id, retarget_profile->get_intermediate_bone_group_id(p_id));
	ur->add_do_method(retarget_profile, "remove_intermediate_bone", p_id);
	ur->commit_action();
}

void RetargetRichProfileEditor::_add_group(const StringName &p_group_name) {
	ERR_FAIL_COND(!retarget_profile);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Add Group"));
	ur->add_undo_method(retarget_profile, "remove_group", retarget_profile->get_groups_size());
	ur->add_do_method(retarget_profile, "add_group", p_group_name);
	ur->commit_action();
}

void RetargetRichProfileEditor::_remove_group(const int p_id) {
	ERR_FAIL_COND(!retarget_profile);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Remove Group"));
	ur->add_undo_method(retarget_profile, "add_group", retarget_profile->get_group_name(p_id), p_id);
	ur->add_undo_method(retarget_profile, "set_group_texture", p_id, retarget_profile->get_group_texture(p_id));
	int len = retarget_profile->get_intermediate_bones_size();
	if (len > 0) {
		len--;
		for (int i = 0; i < len; i++) {
			// Don't emit signal again.
			ur->add_undo_method(retarget_profile, "set_intermediate_bone_group_id", i, retarget_profile->get_intermediate_bone_group_id(i), false);
		}
		// Emit signal.
		ur->add_undo_method(retarget_profile, "set_intermediate_bone_group_id", len, retarget_profile->get_intermediate_bone_group_id(len));
	}
	ur->add_do_method(retarget_profile, "remove_group", p_id);
	ur->commit_action();
}

void RetargetRichProfileEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			create_editors();
		} break;
		case NOTIFICATION_POST_ENTER_TREE: {
			retarget_profile->connect("group_updated", callable_mp(this, &RetargetRichProfileEditor::_update_group_property));
			retarget_profile->connect("intermediate_bone_updated", callable_mp(this, &RetargetRichProfileEditor::_update_intermediate_bone_property));
			retarget_profile->connect("redraw_needed", callable_mp(this, &RetargetRichProfileEditor::recreate_items));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (retarget_profile) {
				if (retarget_profile->is_connected("group_updated", callable_mp(this, &RetargetRichProfileEditor::_update_group_property))) {
					retarget_profile->disconnect("group_updated", callable_mp(this, &RetargetRichProfileEditor::_update_group_property));
				}
				if (retarget_profile->is_connected("intermediate_bone_updated", callable_mp(this, &RetargetRichProfileEditor::_update_intermediate_bone_property))) {
					retarget_profile->disconnect("intermediate_bone_updated", callable_mp(this, &RetargetRichProfileEditor::_update_intermediate_bone_property));
				}
				if (retarget_profile->is_connected("redraw_needed", callable_mp(this, &RetargetRichProfileEditor::recreate_items))) {
					retarget_profile->disconnect("redraw_needed", callable_mp(this, &RetargetRichProfileEditor::recreate_items));
				}
			}
		} break;
	}
}

RetargetRichProfileEditor::RetargetRichProfileEditor(RetargetRichProfile *p_retarget_profile) {
	retarget_profile = p_retarget_profile;
}

RetargetRichProfileEditor::~RetargetRichProfileEditor() {
}

bool EditorInspectorPluginRetargetProfile::can_handle(Object *p_object) {
	return Object::cast_to<RetargetProfile>(p_object) != nullptr;
}

void EditorInspectorPluginRetargetProfile::parse_begin(Object *p_object) {
	RetargetRichProfile *rrp = Object::cast_to<RetargetRichProfile>(p_object);
	if (rrp) {
		rp_editor = memnew(RetargetRichProfileEditor(rrp));
		add_custom_control(rp_editor);
		return;
	}

	RetargetProfile *rp = Object::cast_to<RetargetProfile>(p_object);
	if (rp) {
		rp_editor = memnew(RetargetProfileEditor(rp));
		add_custom_control(rp_editor);
		return;
	}
}

RetargetProfileEditorPlugin::RetargetProfileEditorPlugin() {
	// Register properties in editor settings.
	EDITOR_DEF("editors/retarget_mapper/button_colors/set", Color(0.1, 0.6, 0.25));
	EDITOR_DEF("editors/retarget_mapper/button_colors/error", Color(0.8, 0.2, 0.2));
	EDITOR_DEF("editors/retarget_mapper/button_colors/unset", Color(0.3, 0.3, 0.3));

	Ref<EditorInspectorPluginRetargetProfile> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}

// Retarget source setting

void RetargetBoneOptionMapperItem::create_editors() {
	retarget_mode = memnew(EditorPropertyEnum);
	retarget_mode->setup(retarget_mode_arr);
	retarget_mode->set_label("Retarget Mode");
	retarget_mode->set_selectable(false);
	retarget_mode->connect("property_changed", callable_mp(this, &RetargetBoneOptionMapperItem::_value_changed));
	inputs_vbox->add_child(retarget_mode);

	if (enabled) {
		String prep = String(key_name) + "/";
		retarget_mode->set_object_and_property(retarget_option, prep + "retarget_mode");
		retarget_mode->update_property();
	}
}

void RetargetBoneOptionMapperItem::_value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	ERR_FAIL_COND(!retarget_option);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Retarget Option Property"), UndoRedo::MERGE_ENDS);
	ur->add_undo_property(retarget_option, p_property, retarget_option->get(p_property));
	ur->add_do_property(retarget_option, p_property, p_value);
	ur->commit_action();
}

void RetargetBoneOptionMapperItem::_update_property() {
	if (retarget_mode->get_edited_object() && retarget_mode->get_edited_property()) {
		retarget_mode->update_property();
	}
}

void RetargetBoneOptionMapperItem::_bind_methods() {
}

void RetargetBoneOptionMapperItem::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POST_ENTER_TREE: {
			retarget_option->connect("retarget_option_updated", callable_mp(this, &RetargetBoneOptionMapperItem::_update_property));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (retarget_option && retarget_option->is_connected("retarget_option_updated", callable_mp(this, &RetargetBoneOptionMapperItem::_update_property))) {
				retarget_option->disconnect("retarget_option_updated", callable_mp(this, &RetargetBoneOptionMapperItem::_update_property));
			}
		} break;
	}
}

RetargetBoneOptionMapperItem::RetargetBoneOptionMapperItem(RetargetBoneOption *p_retarget_option, const StringName &p_name, const bool p_enabled) {
	retarget_option = p_retarget_option;
	key_name = p_name;
	enabled = p_enabled;
	retarget_mode_arr.push_back("Global");
	retarget_mode_arr.push_back("Local");
	retarget_mode_arr.push_back("Absolute");
}

RetargetBoneOptionMapperItem::~RetargetBoneOptionMapperItem() {
}

void RetargetBoneOptionMapper::clear_items() {
	// Clear items.
	int len = mapper_items.size();
	for (int i = 0; i < len; i++) {
		mapper_items[i]->disconnect("remove", callable_mp(this, &RetargetBoneOptionMapper::_remove_item));
		mapper_items[i]->disconnect("enable", callable_mp(this, &RetargetBoneOptionMapper::_add_item));
		map_vbox->remove_child(mapper_items[i]);
		memdelete(mapper_items[i]);
	}
	mapper_items.clear();

	len = unprofiled_items.size();
	for (int i = 0; i < len; i++) {
		unprofiled_items[i]->disconnect("remove", callable_mp(this, &RetargetBoneOptionMapper::_remove_item));
		unprofiled_items[i]->disconnect("enable", callable_mp(this, &RetargetBoneOptionMapper::_add_item));
		unprofiled_vbox->remove_child(unprofiled_items[i]);
		memdelete(unprofiled_items[i]);
	}
	unprofiled_items.clear();
}

void RetargetBoneOptionMapper::recreate_items() {
	clear_items();
	// Create items by profile.
	Vector<StringName> found_keys;
	if (profile.is_valid()) {
		int len = profile->get_intermediate_bones_size();
		for (int i = 0; i < len; i++) {
			StringName bn = profile->get_intermediate_bone_name(i);
			bool is_found = retarget_option->has_key(bn);
			if (is_found) {
				found_keys.push_back(bn);
			}
			mapper_items.append(memnew(RetargetBoneOptionMapperItem(retarget_option, bn, is_found)));
			mapper_items[i]->connect("remove", callable_mp(this, &RetargetBoneOptionMapper::_remove_item), varray(), CONNECT_DEFERRED);
			mapper_items[i]->connect("enable", callable_mp(this, &RetargetBoneOptionMapper::_add_item), varray(), CONNECT_DEFERRED);
			map_vbox->add_child(mapper_items[i]);
		}
	}

	// Create items by setting.
	Vector<StringName> keys = retarget_option->get_keys();
	int len = keys.size();
	int j = 0;
	for (int i = 0; i < len; i++) {
		StringName bn = keys[i];
		if (!found_keys.has(bn)) {
			unprofiled_items.append(memnew(RetargetBoneOptionMapperItem(retarget_option, bn, true)));
			unprofiled_items[j]->connect("remove", callable_mp(this, &RetargetBoneOptionMapper::_remove_item), varray(), CONNECT_DEFERRED);
			unprofiled_items[j]->connect("enable", callable_mp(this, &RetargetBoneOptionMapper::_add_item), varray(), CONNECT_DEFERRED);
			unprofiled_vbox->add_child(unprofiled_items[j]);
			j++;
		}
	}

	update_group_ids();
	recreate_rich_editor();
}

void RetargetBoneOptionMapper::_update_mapper_state() {
	int len = mapper_buttons.size();
	for (int i = 0; i < len; i++) {
		set_mapper_state(i, get_mapper_state(mapper_buttons[i]->get_name()));
	}
}

void RetargetBoneOptionMapper::_add_item(const StringName &p_intermediate_bone_name) {
	ERR_FAIL_COND(!retarget_option);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Add Retarget Option Item"));
	ur->add_undo_method(retarget_option, "remove_key", p_intermediate_bone_name);
	ur->add_do_method(retarget_option, "add_key", p_intermediate_bone_name);
	ur->commit_action();
	recreate_items();
}

void RetargetBoneOptionMapper::_remove_item(const StringName &p_intermediate_bone_name) {
	ERR_FAIL_COND(!retarget_option);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Remove Retarget Map Item"));
	ur->add_undo_method(retarget_option, "add_key", p_intermediate_bone_name);
	ur->add_undo_method(retarget_option, "set_retarget_mode", p_intermediate_bone_name, retarget_option->get_retarget_mode(p_intermediate_bone_name));
	ur->add_do_method(retarget_option, "remove_key", p_intermediate_bone_name);
	ur->commit_action();
	recreate_items();
}

MapperButton::MapperState RetargetBoneOptionMapper::get_mapper_state(const StringName &p_intermediate_bone_name) {
	ERR_FAIL_COND_V(!retarget_option, MapperButton::MAPPER_STATE_UNSET);
	if (retarget_option->has_key(p_intermediate_bone_name)) {
		return MapperButton::MAPPER_STATE_SET;
	}
	return MapperButton::MAPPER_STATE_UNSET;
}

RetargetBoneOptionMapper::RetargetBoneOptionMapper(RetargetBoneOption *p_retarget_option) {
	retarget_option = p_retarget_option;
}

void RetargetBoneOptionMapper::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POST_ENTER_TREE: {
			retarget_option->connect("retarget_option_updated", callable_mp(this, &RetargetBoneOptionMapper::_update_mapper_state));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (retarget_option && retarget_option->is_connected("retarget_option_updated", callable_mp(this, &RetargetBoneOptionMapper::_update_mapper_state))) {
				retarget_option->disconnect("retarget_option_updated", callable_mp(this, &RetargetBoneOptionMapper::_update_mapper_state));
			}
		} break;
	}
}

RetargetBoneOptionMapper::~RetargetBoneOptionMapper() {
}

void RetargetBoneOptionEditor::create_editors() {
	mapper = memnew(RetargetBoneOptionMapper(retarget_option));
	mapper->set_profile(profile);
	add_child(mapper);
}

void RetargetBoneOptionEditor::clear_editors() {
	remove_child(mapper);
	memdelete(mapper);
}

void RetargetBoneOptionEditor::set_profile(const Ref<RetargetProfile> &p_profile) {
	profile = p_profile;
	clear_editors();
	create_editors();
}

Ref<RetargetProfile> RetargetBoneOptionEditor::get_profile() const {
	return profile;
}

void RetargetBoneOptionEditor::fetch_objects() {
	EditorSelection *es = EditorInterface::get_singleton()->get_selection();
	if (es->get_selected_nodes().size() == 1) {
		Node *nd = Object::cast_to<Node>(es->get_selected_nodes()[0]);
		if (!nd) {
			return;
		}
		// SkeletonRetarget
		SkeletonRetarget *sr = Object::cast_to<SkeletonRetarget>(nd);
		if (sr) {
			profile = sr->get_retarget_profile();
		}
		// AnimationPlayer
		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(nd);
		if (ap) {
			profile = ap->get_retarget_profile();
		}
	} else {
		// Editor should not exist.
		profile = Ref<RetargetProfile>();
	}
}

void RetargetBoneOptionEditor::redraw() {
	mapper->recreate_items();
}

void RetargetBoneOptionEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			fetch_objects();
			create_editors();
		} break;
		case NOTIFICATION_POST_ENTER_TREE: {
			retarget_option->connect("redraw_needed", callable_mp(this, &RetargetBoneOptionEditor::redraw));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (retarget_option && retarget_option->is_connected("redraw_needed", callable_mp(this, &RetargetBoneOptionEditor::redraw))) {
				retarget_option->disconnect("redraw_needed", callable_mp(this, &RetargetBoneOptionEditor::redraw));
			}
		} break;
	}
}

RetargetBoneOptionEditor::RetargetBoneOptionEditor(RetargetBoneOption *p_retarget_option) {
	retarget_option = p_retarget_option;
}

RetargetBoneOptionEditor::~RetargetBoneOptionEditor() {
}

bool EditorInspectorPluginRetargetBoneOption::can_handle(Object *p_object) {
	return Object::cast_to<RetargetBoneOption>(p_object) != nullptr;
}

void EditorInspectorPluginRetargetBoneOption::parse_begin(Object *p_object) {
	RetargetBoneOption *rs = Object::cast_to<RetargetBoneOption>(p_object);
	rs_editor = memnew(RetargetBoneOptionEditor(rs));
	add_custom_control(rs_editor);
}

RetargetBoneOptionEditorPlugin::RetargetBoneOptionEditorPlugin() {
	Ref<EditorInspectorPluginRetargetBoneOption> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}

// Retarget target setting

void RetargetBoneMapMapperItem::create_editors() {
	HBoxContainer *bone_name_box = memnew(HBoxContainer);
	bone_name = memnew(EditorPropertyText);
	bone_name->set_label("Retarget Bone");
	bone_name->set_h_size_flags(SIZE_EXPAND_FILL);
	bone_name->set_selectable(false);
	bone_name->connect("property_changed", callable_mp(this, &RetargetBoneMapMapperItem::_value_changed));
	bone_name_box->add_child(bone_name);

	button_pick = memnew(Button);
	button_pick->set_icon(get_theme_icon(SNAME("ColorPick"), SNAME("EditorIcons")));
	button_pick->connect("pressed", callable_mp(this, &RetargetBoneMapMapperItem::fire_pick));
	bone_name_box->add_child(button_pick);

	inputs_vbox->add_child(bone_name_box);

	if (enabled) {
		String prep = String(key_name) + "/";
		bone_name->set_object_and_property(retarget_map, prep);
		bone_name->update_property();
	}
}

void RetargetBoneMapMapperItem::_value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) {
	ERR_FAIL_COND(!retarget_map);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Retarget Map Property"), UndoRedo::MERGE_ENDS);
	ur->add_undo_property(retarget_map, p_property, retarget_map->get(p_property));
	ur->add_do_property(retarget_map, p_property, p_value);
	ur->commit_action();
}

void RetargetBoneMapMapperItem::_update_property() {
	if (bone_name->get_edited_object() && bone_name->get_edited_property()) {
		bone_name->update_property();
	}
}

void RetargetBoneMapMapperItem::fire_pick() {
	emit_signal("pick", key_name);
}

void RetargetBoneMapMapperItem::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POST_ENTER_TREE: {
			retarget_map->connect("retarget_map_updated", callable_mp(this, &RetargetBoneMapMapperItem::_update_property));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (retarget_map && retarget_map->is_connected("retarget_map_updated", callable_mp(this, &RetargetBoneMapMapperItem::_update_property))) {
				retarget_map->disconnect("retarget_map_updated", callable_mp(this, &RetargetBoneMapMapperItem::_update_property));
			}
		} break;
	}
}

void RetargetBoneMapMapperItem::_bind_methods() {
}

RetargetBoneMapMapperItem::RetargetBoneMapMapperItem(RetargetBoneMap *p_retarget_map, const StringName &p_name, const bool p_enabled) {
	retarget_map = p_retarget_map;
	key_name = p_name;
	enabled = p_enabled;
}

RetargetBoneMapMapperItem::~RetargetBoneMapMapperItem() {
}

void RetargetBoneMapMapper::apply_picker_selection() {
	if (!picker->has_selected_bone()) {
		return;
	}
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Set Bone Name"));
	ur->add_undo_method(retarget_map, "set_bone_name", picker_key_name, retarget_map->get_bone_name(picker_key_name));
	ur->add_do_method(retarget_map, "set_bone_name", picker_key_name, picker->get_selected_bone());
	ur->commit_action();
	recreate_items();
}

void RetargetBoneMapMapper::clear_items() {
	// Clear items.
	int len = mapper_items.size();
	for (int i = 0; i < len; i++) {
		mapper_items[i]->disconnect("remove", callable_mp(this, &RetargetBoneMapMapper::_remove_item));
		mapper_items[i]->disconnect("enable", callable_mp(this, &RetargetBoneMapMapper::_add_item));
		mapper_items[i]->disconnect("pick", callable_mp(this, &RetargetBoneMapMapper::_pick_bone));
		map_vbox->remove_child(mapper_items[i]);
		memdelete(mapper_items[i]);
	}
	mapper_items.clear();

	len = unprofiled_items.size();
	for (int i = 0; i < len; i++) {
		unprofiled_items[i]->disconnect("remove", callable_mp(this, &RetargetBoneMapMapper::_remove_item));
		unprofiled_items[i]->disconnect("enable", callable_mp(this, &RetargetBoneMapMapper::_add_item));
		unprofiled_items[i]->disconnect("pick", callable_mp(this, &RetargetBoneMapMapper::_pick_bone));
		unprofiled_vbox->remove_child(unprofiled_items[i]);
		memdelete(unprofiled_items[i]);
	}
	unprofiled_items.clear();
}

void RetargetBoneMapMapper::recreate_items() {
	clear_items();
	// Create items by profile.
	Vector<StringName> found_keys;
	if (profile.is_valid()) {
		int len = profile->get_intermediate_bones_size();
		for (int i = 0; i < len; i++) {
			StringName bn = profile->get_intermediate_bone_name(i);
			bool is_found = retarget_map->has_key(bn);
			if (is_found) {
				found_keys.push_back(bn);
			}
			mapper_items.append(memnew(RetargetBoneMapMapperItem(retarget_map, bn, is_found)));
			mapper_items[i]->connect("remove", callable_mp(this, &RetargetBoneMapMapper::_remove_item), varray(), CONNECT_DEFERRED);
			mapper_items[i]->connect("enable", callable_mp(this, &RetargetBoneMapMapper::_add_item), varray(), CONNECT_DEFERRED);
			mapper_items[i]->connect("pick", callable_mp(this, &RetargetBoneMapMapper::_pick_bone), varray(), CONNECT_DEFERRED);
			map_vbox->add_child(mapper_items[i]);
		}
	}

	// Create items by setting.
	Vector<StringName> keys = retarget_map->get_keys();
	int len = keys.size();
	int j = 0;
	for (int i = 0; i < len; i++) {
		StringName bn = keys[i];
		if (!found_keys.has(bn)) {
			unprofiled_items.append(memnew(RetargetBoneMapMapperItem(retarget_map, bn, true)));
			unprofiled_items[j]->connect("remove", callable_mp(this, &RetargetBoneMapMapper::_remove_item), varray(), CONNECT_DEFERRED);
			unprofiled_items[j]->connect("enable", callable_mp(this, &RetargetBoneMapMapper::_add_item), varray(), CONNECT_DEFERRED);
			unprofiled_items[j]->connect("pick", callable_mp(this, &RetargetBoneMapMapper::_pick_bone), varray(), CONNECT_DEFERRED);
			unprofiled_vbox->add_child(unprofiled_items[j]);
			j++;
		}
	}

	update_group_ids();
	recreate_rich_editor();
}

void RetargetBoneMapMapper::_update_mapper_state() {
	int len = mapper_buttons.size();
	for (int i = 0; i < len; i++) {
		set_mapper_state(i, get_mapper_state(mapper_buttons[i]->get_name()));
	}
}

void RetargetBoneMapMapper::_pick_bone(const StringName &p_intermediate_bone_name) {
	picker_key_name = p_intermediate_bone_name;
	// Get bone names.
	ERR_FAIL_COND_MSG(!skeleton, "Skeleton is not found.");
	picker->popup_bones_tree(skeleton, Size2(500, 500) * EDSCALE);
}

void RetargetBoneMapMapper::_add_item(const StringName &p_intermediate_bone_name) {
	ERR_FAIL_COND(!retarget_map);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Add Retarget Map Item"));
	ur->add_undo_method(retarget_map, "remove_key", p_intermediate_bone_name);
	ur->add_do_method(retarget_map, "add_key", p_intermediate_bone_name);
	ur->commit_action();
	recreate_items();
}

void RetargetBoneMapMapper::_remove_item(const StringName &p_intermediate_bone_name) {
	ERR_FAIL_COND(!retarget_map);
	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Remove Retarget Map Item"));
	ur->add_undo_method(retarget_map, "add_key", p_intermediate_bone_name);
	ur->add_undo_method(retarget_map, "set_bone_name", p_intermediate_bone_name, retarget_map->get_bone_name(p_intermediate_bone_name));
	ur->add_do_method(retarget_map, "remove_key", p_intermediate_bone_name);
	ur->commit_action();
	recreate_items();
}

MapperButton::MapperState RetargetBoneMapMapper::get_mapper_state(const StringName &p_intermediate_bone_name) {
	ERR_FAIL_COND_V(!retarget_map, MapperButton::MAPPER_STATE_UNSET);
	if (retarget_map->has_key(p_intermediate_bone_name)) {
		if (!skeleton) {
			return MapperButton::MAPPER_STATE_ERROR;
		}
		if (skeleton->find_bone(retarget_map->get_bone_name(p_intermediate_bone_name)) >= 0) {
			return MapperButton::MAPPER_STATE_SET;
		}
		return MapperButton::MAPPER_STATE_ERROR;
	}
	return MapperButton::MAPPER_STATE_UNSET;
}

void RetargetBoneMapMapper::set_skeleton(Skeleton3D *p_skeleton) {
	skeleton = p_skeleton;
}

void RetargetBoneMapMapper::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			picker = memnew(Skeleton3DBonePicker);
			picker->connect("confirmed", callable_mp(this, &RetargetBoneMapMapper::apply_picker_selection));
			add_child(picker, false, INTERNAL_MODE_FRONT);
		} break;
		case NOTIFICATION_POST_ENTER_TREE: {
			retarget_map->connect("retarget_map_updated", callable_mp(this, &RetargetBoneMapMapper::_update_mapper_state));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (picker && picker->is_connected("confirmed", callable_mp(this, &RetargetBoneMapMapper::apply_picker_selection))) {
				picker->disconnect("confirmed", callable_mp(this, &RetargetBoneMapMapper::apply_picker_selection));
			}
			if (retarget_map && retarget_map->is_connected("retarget_map_updated", callable_mp(this, &RetargetBoneMapMapper::_update_mapper_state))) {
				retarget_map->disconnect("retarget_map_updated", callable_mp(this, &RetargetBoneMapMapper::_update_mapper_state));
			}
		} break;
	}
}

RetargetBoneMapMapper::RetargetBoneMapMapper(RetargetBoneMap *p_retarget_map) {
	retarget_map = p_retarget_map;
}

RetargetBoneMapMapper::~RetargetBoneMapMapper() {
}

void RetargetBoneMapEditor::create_editors() {
	mapper = memnew(RetargetBoneMapMapper(retarget_map));
	mapper->set_skeleton(skeleton);
	mapper->set_profile(profile);
	add_child(mapper);
}

void RetargetBoneMapEditor::clear_editors() {
	remove_child(mapper);
	memdelete(mapper);
}

void RetargetBoneMapEditor::set_skeleton(Skeleton3D *p_skeleton) {
	skeleton = p_skeleton;
	clear_editors();
	create_editors();
}

Skeleton3D *RetargetBoneMapEditor::get_skeleton() {
	return skeleton;
}

void RetargetBoneMapEditor::set_profile(const Ref<RetargetProfile> &p_profile) {
	profile = p_profile;
	clear_editors();
	create_editors();
}

Ref<RetargetProfile> RetargetBoneMapEditor::get_profile() const {
	return profile;
}

void RetargetBoneMapEditor::fetch_objects() {
	EditorSelection *es = EditorInterface::get_singleton()->get_selection();
	if (es->get_selected_nodes().size() == 1) {
		Node *nd = Object::cast_to<Node>(es->get_selected_nodes()[0]);
		if (!nd) {
			return;
		}
		// SkeletonRetarget
		SkeletonRetarget *sr = Object::cast_to<SkeletonRetarget>(nd);
		if (sr) {
			Object *obj = nullptr;
			// Note:
			// If maps are same, maps only refer to source skeleton.
			// That's why the SkeletonRetarget doesn't allow to set same maps in set_target_map() and set_source_map().
			if (retarget_map == sr->get_source_map().ptr()) {
				obj = sr->get_node_or_null(sr->get_source_skeleton());
			} else {
				obj = sr->get_node_or_null(sr->get_target_skeleton());
			}
			skeleton = Object::cast_to<Skeleton3D>(obj);
			profile = sr->get_retarget_profile();
		}
		// AnimationPlayer
		AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(nd);
		if (ap) {
			Object *obj = ap->get_node_or_null(ap->get_retarget_skeleton());
			skeleton = Object::cast_to<Skeleton3D>(obj);
			profile = ap->get_retarget_profile();
		}
	} else {
		// Editor should not exist.
		skeleton = nullptr;
		profile = Ref<RetargetProfile>();
	}
}

void RetargetBoneMapEditor::redraw() {
	mapper->recreate_items();
}

void RetargetBoneMapEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			fetch_objects();
			create_editors();
		} break;
		case NOTIFICATION_POST_ENTER_TREE: {
			retarget_map->connect("redraw_needed", callable_mp(this, &RetargetBoneMapEditor::redraw));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (retarget_map && retarget_map->is_connected("redraw_needed", callable_mp(this, &RetargetBoneMapEditor::redraw))) {
				retarget_map->disconnect("redraw_needed", callable_mp(this, &RetargetBoneMapEditor::redraw));
			}
		} break;
	}
}

RetargetBoneMapEditor::RetargetBoneMapEditor(RetargetBoneMap *p_retarget_map) {
	retarget_map = p_retarget_map;
}

RetargetBoneMapEditor::~RetargetBoneMapEditor() {
}

bool EditorInspectorPluginRetargetBoneMap::can_handle(Object *p_object) {
	return Object::cast_to<RetargetBoneMap>(p_object) != nullptr;
}

void EditorInspectorPluginRetargetBoneMap::parse_begin(Object *p_object) {
	RetargetBoneMap *rt = Object::cast_to<RetargetBoneMap>(p_object);
	rt_editor = memnew(RetargetBoneMapEditor(rt));
	add_custom_control(rt_editor);
}

RetargetBoneMapEditorPlugin::RetargetBoneMapEditorPlugin() {
	Ref<EditorInspectorPluginRetargetBoneMap> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}

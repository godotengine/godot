/*************************************************************************/
/*  openxr_action_map_editor.cpp                                         */
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

#include "openxr_action_map_editor.h"

#include "core/config/project_settings.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"

// TODO implement redo/undo system

void OpenXRActionMapEditor::_bind_methods() {
	ClassDB::bind_method("_add_action_set_editor", &OpenXRActionMapEditor::_add_action_set_editor);
	ClassDB::bind_method("_update_action_sets", &OpenXRActionMapEditor::_update_action_sets);

	ClassDB::bind_method("_add_interaction_profile_editor", &OpenXRActionMapEditor::_add_interaction_profile_editor);
	ClassDB::bind_method("_update_interaction_profiles", &OpenXRActionMapEditor::_update_interaction_profiles);

	ClassDB::bind_method(D_METHOD("_add_action_set", "name"), &OpenXRActionMapEditor::_add_action_set);
	ClassDB::bind_method(D_METHOD("_set_focus_on_action_set", "action_set"), &OpenXRActionMapEditor::_set_focus_on_action_set);
	ClassDB::bind_method(D_METHOD("_remove_action_set", "name"), &OpenXRActionMapEditor::_remove_action_set);
}

void OpenXRActionMapEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			for (int i = 0; i < tabs->get_child_count(); i++) {
				Control *tab = static_cast<Control *>(tabs->get_child(i));
				if (tab) {
					tab->add_theme_style_override("bg", get_theme_stylebox(SNAME("bg"), SNAME("Tree")));
				}
			}
		} break;

		case NOTIFICATION_READY: {
			_update_action_sets();
			_update_interaction_profiles();
		} break;
	}
}

OpenXRActionSetEditor *OpenXRActionMapEditor::_add_action_set_editor(Ref<OpenXRActionSet> p_action_set) {
	ERR_FAIL_COND_V(p_action_set.is_null(), nullptr);

	OpenXRActionSetEditor *action_set_editor = memnew(OpenXRActionSetEditor(action_map, p_action_set));
	action_set_editor->connect("remove", callable_mp(this, &OpenXRActionMapEditor::_on_remove_action_set));
	action_set_editor->connect("action_removed", callable_mp(this, &OpenXRActionMapEditor::_on_action_removed));
	actionsets_vb->add_child(action_set_editor);

	return action_set_editor;
}

void OpenXRActionMapEditor::_update_action_sets() {
	// out with the old...
	while (actionsets_vb->get_child_count() > 0) {
		memdelete(actionsets_vb->get_child(0));
	}

	// in with the new...
	if (action_map.is_valid()) {
		Array action_sets = action_map->get_action_sets();
		for (int i = 0; i < action_sets.size(); i++) {
			Ref<OpenXRActionSet> action_set = action_sets[i];
			_add_action_set_editor(action_set);
		}
	}
}

OpenXRInteractionProfileEditorBase *OpenXRActionMapEditor::_add_interaction_profile_editor(Ref<OpenXRInteractionProfile> p_interaction_profile) {
	ERR_FAIL_COND_V(p_interaction_profile.is_null(), nullptr);

	String profile_path = p_interaction_profile->get_interaction_profile_path();

	// need to instance the correct editor for our profile
	OpenXRInteractionProfileEditorBase *new_profile_editor = nullptr;
	if (profile_path == "placeholder_text") {
		// instance specific editor for this type
	} else {
		// instance generic editor
		new_profile_editor = memnew(OpenXRInteractionProfileEditor(action_map, p_interaction_profile));
	}

	// now add it in..
	ERR_FAIL_NULL_V(new_profile_editor, nullptr);
	tabs->add_child(new_profile_editor);
	new_profile_editor->add_theme_style_override("bg", get_theme_stylebox(SNAME("bg"), SNAME("Tree")));
	tabs->set_tab_button_icon(tabs->get_tab_count() - 1, get_theme_icon(SNAME("close"), SNAME("TabBar")));

	interaction_profiles.push_back(new_profile_editor);

	return new_profile_editor;
}

void OpenXRActionMapEditor::_update_interaction_profiles() {
	// out with the old...
	while (interaction_profiles.size() > 0) {
		Node *interaction_profile = interaction_profiles[0];
		interaction_profiles.remove_at(0);

		tabs->remove_child(interaction_profile);
		interaction_profile->queue_delete();
	}

	// in with the new...
	if (action_map.is_valid()) {
		Array new_interaction_profiles = action_map->get_interaction_profiles();
		for (int i = 0; i < new_interaction_profiles.size(); i++) {
			Ref<OpenXRInteractionProfile> interaction_profile = new_interaction_profiles[i];
			_add_interaction_profile_editor(interaction_profile);
		}
	}
}

OpenXRActionSetEditor *OpenXRActionMapEditor::_add_action_set(String p_name) {
	ERR_FAIL_COND_V(action_map.is_null(), nullptr);
	Ref<OpenXRActionSet> new_action_set;

	// add our new action set
	new_action_set.instantiate();
	new_action_set->set_name(p_name);
	new_action_set->set_localized_name(p_name);
	action_map->add_action_set(new_action_set);

	// update our editor right away
	return _add_action_set_editor(new_action_set);
}

void OpenXRActionMapEditor::_remove_action_set(String p_name) {
	ERR_FAIL_COND(action_map.is_null());
	Ref<OpenXRActionSet> action_set = action_map->find_action_set(p_name);
	ERR_FAIL_COND(action_set.is_null());

	if (action_set->get_action_count() > 0) {
		// we should remove these and add to our redo/undo step before calling _remove_action_set
		WARN_PRINT("Action set still has associated actions before being removed!");
	}

	// now we remove it
	action_map->remove_action_set(action_set);
}

void OpenXRActionMapEditor::_on_add_action_set() {
	ERR_FAIL_COND(action_map.is_null());
	String new_name = "New";
	int count = 0;

	while (action_map->find_action_set(new_name).is_valid()) {
		new_name = "New_" + itos(count++);
	}

	OpenXRActionSetEditor *new_action_set_editor = _add_action_set(new_name);

	// Make sure our action set is the current tab
	tabs->set_current_tab(0);

	call_deferred("_set_focus_on_action_set", new_action_set_editor);
}

void OpenXRActionMapEditor::_set_focus_on_action_set(OpenXRActionSetEditor *p_action_set_editor) {
	// Scroll down to our new entry
	actionsets_scroll->ensure_control_visible(p_action_set_editor);

	// Set focus on this entry
	p_action_set_editor->set_focus_on_entry();
}

void OpenXRActionMapEditor::_on_remove_action_set(Object *p_action_set_editor) {
	ERR_FAIL_COND(action_map.is_null());

	OpenXRActionSetEditor *action_set_editor = Object::cast_to<OpenXRActionSetEditor>(p_action_set_editor);
	ERR_FAIL_NULL(action_set_editor);
	ERR_FAIL_COND(action_set_editor->get_parent() != actionsets_vb);
	Ref<OpenXRActionSet> action_set = action_set_editor->get_action_set();
	ERR_FAIL_COND(action_set.is_null());

	action_map->remove_action_set(action_set);
	actionsets_vb->remove_child(action_set_editor);
	action_set_editor->queue_delete();
}

void OpenXRActionMapEditor::_on_action_removed() {
	// make sure our interaction profiles are updated
	_update_interaction_profiles();
}

void OpenXRActionMapEditor::_on_add_interaction_profile() {
	ERR_FAIL_COND(action_map.is_null());

	PackedStringArray already_selected;

	for (int i = 0; i < action_map->get_interaction_profile_count(); i++) {
		already_selected.push_back(action_map->get_interaction_profile(i)->get_interaction_profile_path());
	}

	select_interaction_profile_dialog->open(already_selected);
}

void OpenXRActionMapEditor::_on_interaction_profile_selected(const String p_path) {
	ERR_FAIL_COND(action_map.is_null());

	Ref<OpenXRInteractionProfile> new_profile;
	new_profile.instantiate();
	new_profile->set_interaction_profile_path(p_path);
	action_map->add_interaction_profile(new_profile);

	_add_interaction_profile_editor(new_profile);

	tabs->set_current_tab(tabs->get_tab_count() - 1);
}

void OpenXRActionMapEditor::_load_action_map(const String p_path, bool p_create_new_if_missing) {
	action_map = ResourceLoader::load(p_path, "", ResourceFormatLoader::CACHE_MODE_IGNORE);
	if (action_map.is_null()) {
		if (p_create_new_if_missing) {
			action_map.instantiate();
			action_map->create_default_action_sets();
		} else {
			EditorNode::get_singleton()->show_warning(TTR("Invalid file, not an OpenXR action map."));

			edited_path = "";
			header_label->set_text("");
			return;
		}
	}

	edited_path = p_path;
	header_label->set_text(TTR("OpenXR Action map:") + " " + p_path.get_file());
}

void OpenXRActionMapEditor::_on_save_action_map() {
	Error err = ResourceSaver::save(action_map, edited_path);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("Error saving file: %s"), edited_path));
		return;
	}

	_update_action_sets();
	_update_interaction_profiles();
}

void OpenXRActionMapEditor::_on_reset_to_default_layout() {
	// create a new one
	action_map.unref();
	action_map.instantiate();
	action_map->create_default_action_sets();

	_update_action_sets();
	_update_interaction_profiles();
}

void OpenXRActionMapEditor::_on_tabs_tab_changed(int p_tab) {
}

void OpenXRActionMapEditor::_on_tab_button_pressed(int p_tab) {
	OpenXRInteractionProfileEditorBase *profile_editor = static_cast<OpenXRInteractionProfileEditorBase *>(tabs->get_tab_control(p_tab));
	ERR_FAIL_NULL(profile_editor);

	Ref<OpenXRInteractionProfile> interaction_profile = profile_editor->get_interaction_profile();
	ERR_FAIL_COND(interaction_profile.is_null());

	action_map->remove_interaction_profile(interaction_profile);
	tabs->remove_child(profile_editor);
	profile_editor->queue_delete();
}

void OpenXRActionMapEditor::open_action_map(String p_path) {
	EditorNode::get_singleton()->make_bottom_panel_item_visible(this);

	_load_action_map(p_path);

	_update_action_sets();
	_update_interaction_profiles();
}

OpenXRActionMapEditor::OpenXRActionMapEditor() {
	set_custom_minimum_size(Size2(0.0, 300.0));

	top_hb = memnew(HBoxContainer);
	add_child(top_hb);

	header_label = memnew(Label);
	header_label->set_text(String(TTR("Action Map")));
	header_label->set_clip_text(true);
	header_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	top_hb->add_child(header_label);

	add_action_set = memnew(Button);
	add_action_set->set_text(TTR("Add Action Set"));
	add_action_set->set_tooltip(TTR("Add an action set."));
	add_action_set->connect("pressed", callable_mp(this, &OpenXRActionMapEditor::_on_add_action_set));
	top_hb->add_child(add_action_set);

	add_interaction_profile = memnew(Button);
	add_interaction_profile->set_text(TTR("Add profile"));
	add_interaction_profile->set_tooltip(TTR("Add an interaction profile."));
	add_interaction_profile->connect("pressed", callable_mp(this, &OpenXRActionMapEditor::_on_add_interaction_profile));
	top_hb->add_child(add_interaction_profile);

	VSeparator *vseparator = memnew(VSeparator);
	top_hb->add_child(vseparator);

	save_as = memnew(Button);
	save_as->set_text(TTR("Save"));
	save_as->set_tooltip(TTR("Save this OpenXR action map."));
	save_as->connect("pressed", callable_mp(this, &OpenXRActionMapEditor::_on_save_action_map));
	top_hb->add_child(save_as);

	_default = memnew(Button);
	_default->set_text(TTR("Reset to Default"));
	_default->set_tooltip(TTR("Reset to default OpenXR action map."));
	_default->connect("pressed", callable_mp(this, &OpenXRActionMapEditor::_on_reset_to_default_layout));
	top_hb->add_child(_default);

	tabs = memnew(TabContainer);
	tabs->set_h_size_flags(SIZE_EXPAND_FILL);
	tabs->set_v_size_flags(SIZE_EXPAND_FILL);
	tabs->set_theme_type_variation("TabContainerOdd");
	tabs->connect("tab_changed", callable_mp(this, &OpenXRActionMapEditor::_on_tabs_tab_changed));
	tabs->connect("tab_button_pressed", callable_mp(this, &OpenXRActionMapEditor::_on_tab_button_pressed));
	add_child(tabs);

	actionsets_scroll = memnew(ScrollContainer);
	actionsets_scroll->set_h_size_flags(SIZE_EXPAND_FILL);
	actionsets_scroll->set_v_size_flags(SIZE_EXPAND_FILL);
	actionsets_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	tabs->add_child(actionsets_scroll);
	actionsets_scroll->set_name(TTR("Action Sets"));

	actionsets_vb = memnew(VBoxContainer);
	actionsets_vb->set_h_size_flags(SIZE_EXPAND_FILL);
	actionsets_scroll->add_child(actionsets_vb);

	select_interaction_profile_dialog = memnew(OpenXRSelectInteractionProfileDialog);
	select_interaction_profile_dialog->connect("interaction_profile_selected", callable_mp(this, &OpenXRActionMapEditor::_on_interaction_profile_selected));
	add_child(select_interaction_profile_dialog);

	_load_action_map(ProjectSettings::get_singleton()->get("xr/openxr/default_action_map"));
}

OpenXRActionMapEditor::~OpenXRActionMapEditor() {
}

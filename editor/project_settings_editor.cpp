/*************************************************************************/
/*  project_settings_editor.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "project_settings_editor.h"

#include "core/config/project_settings.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"

ProjectSettingsEditor *ProjectSettingsEditor::singleton = nullptr;

void ProjectSettingsEditor::popup_project_settings() {
	// Restore valid window bounds or pop up at default size.
	Rect2 saved_size = EditorSettings::get_singleton()->get_project_metadata("dialog_bounds", "project_settings", Rect2());
	if (saved_size != Rect2()) {
		popup(saved_size);
	} else {
		popup_centered_clamped(Size2(900, 700) * EDSCALE, 0.8);
	}

	_add_feature_overrides();
	inspector->update_category_list();

	localization_editor->update_translations();
	autoload_settings->update_autoload();
	plugin_settings->update_plugins();
	import_defaults_editor->clear();
}

void ProjectSettingsEditor::queue_save() {
	EditorNode::get_singleton()->notify_settings_changed();
	timer->start();
}

void ProjectSettingsEditor::set_plugins_page() {
	tab_container->set_current_tab(plugin_settings->get_index());
}

void ProjectSettingsEditor::update_plugins() {
	plugin_settings->update_plugins();
}

void ProjectSettingsEditor::_setting_edited(const String &p_name) {
	queue_save();
}

void ProjectSettingsEditor::_advanced_toggled(bool p_button_pressed) {
	EditorSettings::get_singleton()->set_project_metadata("project_settings", "advanced_mode", p_button_pressed);
	inspector->set_restrict_to_basic_settings(!p_button_pressed);
}

void ProjectSettingsEditor::_setting_selected(const String &p_path) {
	if (p_path == String()) {
		return;
	}

	property_box->set_text(inspector->get_current_section() + "/" + p_path);

	_update_property_box(); // set_text doesn't trigger text_changed
}

void ProjectSettingsEditor::_add_setting() {
	String setting = _get_setting_name();

	// Initialize the property with the default value for the given type.
	Callable::CallError ce;
	Variant value;
	Variant::construct(Variant::Type(type_box->get_selected_id()), value, nullptr, 0, ce);

	undo_redo->create_action(TTR("Add Project Setting"));
	undo_redo->add_do_property(ps, setting, value);
	undo_redo->add_undo_property(ps, setting, ps->has_setting(setting) ? ps->get(setting) : Variant());

	undo_redo->add_do_method(inspector, "update_category_list");
	undo_redo->add_undo_method(inspector, "update_category_list");
	undo_redo->add_do_method(this, "queue_save");
	undo_redo->add_undo_method(this, "queue_save");
	undo_redo->commit_action();

	inspector->set_current_section(setting.get_slice("/", 1));
	add_button->release_focus();
}

void ProjectSettingsEditor::_delete_setting() {
	String setting = _get_setting_name();
	Variant value = ps->get(setting);
	int order = ps->get_order(setting);

	undo_redo->create_action(TTR("Delete Item"));

	undo_redo->add_do_method(ps, "clear", setting);
	undo_redo->add_undo_method(ps, "set", setting, value);
	undo_redo->add_undo_method(ps, "set_order", setting, order);

	undo_redo->add_do_method(inspector, "update_category_list");
	undo_redo->add_undo_method(inspector, "update_category_list");
	undo_redo->add_do_method(this, "queue_save");
	undo_redo->add_undo_method(this, "queue_save");

	undo_redo->commit_action();

	property_box->clear();
	del_button->release_focus();
}

void ProjectSettingsEditor::_property_box_changed(const String &p_text) {
	_update_property_box();
}

void ProjectSettingsEditor::_feature_selected(int p_index) {
	Vector<String> t = property_box->get_text().strip_edges().split(".", true, 1);
	const String feature = p_index ? "." + feature_box->get_item_text(p_index) : "";
	property_box->set_text(t[0] + feature);
	_update_property_box();
}

void ProjectSettingsEditor::_update_property_box() {
	const String setting = _get_setting_name();
	const Vector<String> t = setting.split(".", true, 1);
	const String name = t[0];
	const String feature = (t.size() == 2) ? t[1] : "";
	bool feature_invalid = (t.size() == 2) && (t[1] == "");

	add_button->set_disabled(true);
	del_button->set_disabled(true);

	if (feature != "") {
		feature_invalid = true;
		for (int i = 1; i < feature_box->get_item_count(); i++) {
			if (feature == feature_box->get_item_text(i)) {
				feature_invalid = false;
				feature_box->select(i);
				break;
			}
		}
	}

	if (feature == "" || feature_invalid) {
		feature_box->select(0);
	}

	if (property_box->get_text() == "") {
		return;
	}

	if (ps->has_setting(setting)) {
		del_button->set_disabled(ps->is_builtin_setting(setting));
		_select_type(ps->get_setting(setting).get_type());
	} else {
		if (ps->has_setting(name)) {
			_select_type(ps->get_setting(name).get_type());
		} else {
			type_box->select(0);
		}

		if (feature_invalid) {
			return;
		}

		const Vector<String> names = name.split("/");
		for (int i = 0; i < names.size(); i++) {
			if (!names[i].is_valid_identifier()) {
				return;
			}
		}

		add_button->set_disabled(false);
	}
}

void ProjectSettingsEditor::_select_type(Variant::Type p_type) {
	type_box->select(type_box->get_item_index(p_type));
}

String ProjectSettingsEditor::_get_setting_name() const {
	String name = property_box->get_text().strip_edges();
	if (name.find("/") == -1) {
		name = "global/" + name;
	}
	return name;
}

void ProjectSettingsEditor::_add_feature_overrides() {
	Set<String> presets;

	presets.insert("bptc");
	presets.insert("s3tc");
	presets.insert("etc");
	presets.insert("etc2");
	presets.insert("pvrtc");
	presets.insert("debug");
	presets.insert("release");
	presets.insert("editor");
	presets.insert("standalone");
	presets.insert("32");
	presets.insert("64");
	presets.insert("Server"); // Not available as an export platform yet, so it needs to be added manually

	EditorExport *ee = EditorExport::get_singleton();

	for (int i = 0; i < ee->get_export_platform_count(); i++) {
		List<String> p;
		ee->get_export_platform(i)->get_platform_features(&p);
		for (List<String>::Element *E = p.front(); E; E = E->next()) {
			presets.insert(E->get());
		}
	}

	for (int i = 0; i < ee->get_export_preset_count(); i++) {
		List<String> p;
		ee->get_export_preset(i)->get_platform()->get_preset_features(ee->get_export_preset(i), &p);
		for (List<String>::Element *E = p.front(); E; E = E->next()) {
			presets.insert(E->get());
		}

		String custom = ee->get_export_preset(i)->get_custom_features();
		Vector<String> custom_list = custom.split(",");
		for (int j = 0; j < custom_list.size(); j++) {
			String f = custom_list[j].strip_edges();
			if (f != String()) {
				presets.insert(f);
			}
		}
	}

	feature_box->clear();
	feature_box->add_item(TTR("(All)"), 0); // So it is always on top.
	int id = 1;
	for (Set<String>::Element *E = presets.front(); E; E = E->next()) {
		feature_box->add_item(E->get(), id++);
	}
}

void ProjectSettingsEditor::_editor_restart() {
	ProjectSettings::get_singleton()->save();
	EditorNode::get_singleton()->save_all_scenes();
	EditorNode::get_singleton()->restart_editor();
}

void ProjectSettingsEditor::_editor_restart_request() {
	restart_container->show();
}

void ProjectSettingsEditor::_editor_restart_close() {
	restart_container->hide();
}

void ProjectSettingsEditor::_action_added(const String &p_name) {
	String name = "input/" + p_name;

	if (ProjectSettings::get_singleton()->has_setting(name)) {
		action_map->show_message(vformat(TTR("An action with the name '%s' already exists."), name));
		return;
	}

	Dictionary action;
	action["events"] = Array();
	action["deadzone"] = 0.5f;

	undo_redo->create_action(TTR("Add Input Action"));
	undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", name, action);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "clear", name);

	undo_redo->add_do_method(this, "_update_action_map_editor");
	undo_redo->add_undo_method(this, "_update_action_map_editor");
	undo_redo->add_do_method(this, "queue_save");
	undo_redo->add_undo_method(this, "queue_save");
	undo_redo->commit_action();
}

void ProjectSettingsEditor::_action_edited(const String &p_name, const Dictionary &p_action) {
	const String property_name = "input/" + p_name;
	Dictionary old_val = ProjectSettings::get_singleton()->get(property_name);

	if (old_val["deadzone"] != p_action["deadzone"]) {
		// Deadzone Changed
		undo_redo->create_action(TTR("Change Action deadzone"));
		undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", property_name, p_action);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", property_name, old_val);

	} else {
		// Events changed
		int event_count = ((Array)p_action["events"]).size();
		int old_event_count = ((Array)old_val["events"]).size();

		if (event_count == old_event_count) {
			undo_redo->create_action(TTR("Edit Input Action Event"));
		} else if (event_count > old_event_count) {
			undo_redo->create_action(TTR("Add Input Action Event"));
		} else if (event_count < old_event_count) {
			undo_redo->create_action(TTR("Remove Input Action Event"));
		}

		undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", property_name, p_action);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", property_name, old_val);
	}

	undo_redo->add_do_method(this, "_update_action_map_editor");
	undo_redo->add_undo_method(this, "_update_action_map_editor");
	undo_redo->add_do_method(this, "queue_save");
	undo_redo->add_undo_method(this, "queue_save");
	undo_redo->commit_action();
}

void ProjectSettingsEditor::_action_removed(const String &p_name) {
	const String property_name = "input/" + p_name;

	Dictionary old_val = ProjectSettings::get_singleton()->get(property_name);
	int order = ProjectSettings::get_singleton()->get_order(property_name);

	undo_redo->create_action(TTR("Erase Input Action"));
	undo_redo->add_do_method(ProjectSettings::get_singleton(), "clear", property_name);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", property_name, old_val);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", property_name, order);

	undo_redo->add_do_method(this, "_update_action_map_editor");
	undo_redo->add_undo_method(this, "_update_action_map_editor");
	undo_redo->add_do_method(this, "queue_save");
	undo_redo->add_undo_method(this, "queue_save");
	undo_redo->commit_action();
}

void ProjectSettingsEditor::_action_renamed(const String &p_old_name, const String &p_new_name) {
	const String old_property_name = "input/" + p_old_name;
	const String new_property_name = "input/" + p_new_name;

	if (ProjectSettings::get_singleton()->has_setting(new_property_name)) {
		action_map->show_message(vformat(TTR("An action with the name '%s' already exists."), new_property_name));
		return;
	}

	int order = ProjectSettings::get_singleton()->get_order(old_property_name);
	Dictionary action = ProjectSettings::get_singleton()->get(old_property_name);

	undo_redo->create_action(TTR("Rename Input Action Event"));
	// Do: clear old, set new
	undo_redo->add_do_method(ProjectSettings::get_singleton(), "clear", old_property_name);
	undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", new_property_name, action);
	undo_redo->add_do_method(ProjectSettings::get_singleton(), "set_order", new_property_name, order);
	// Undo: clear new, set old
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "clear", new_property_name);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", old_property_name, action);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", old_property_name, order);

	undo_redo->add_do_method(this, "_update_action_map_editor");
	undo_redo->add_undo_method(this, "_update_action_map_editor");
	undo_redo->add_do_method(this, "queue_save");
	undo_redo->add_undo_method(this, "queue_save");
	undo_redo->commit_action();
}

void ProjectSettingsEditor::_action_reordered(const String &p_action_name, const String &p_relative_to, bool p_before) {
	const String action_name = "input/" + p_action_name;
	const String target_name = "input/" + p_relative_to;

	// It is much easier to rebuild the custom "input" properties rather than messing around with the "order" values of them.
	Variant action_value = ps->get(action_name);
	Variant target_value = ps->get(target_name);

	List<PropertyInfo> props;
	OrderedHashMap<String, Variant> action_values;
	ProjectSettings::get_singleton()->get_property_list(&props);

	undo_redo->create_action(TTR("Update Input Action Order"));

	for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
		PropertyInfo prop = E->get();
		// Skip builtins and non-inputs
		if (ProjectSettings::get_singleton()->is_builtin_setting(prop.name) || !prop.name.begins_with("input/")) {
			continue;
		}

		action_values.insert(prop.name, ps->get(prop.name));

		undo_redo->add_do_method(ProjectSettings::get_singleton(), "clear", prop.name);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "clear", prop.name);
	}

	for (OrderedHashMap<String, Variant>::Element E = action_values.front(); E; E = E.next()) {
		String name = E.key();
		Variant value = E.get();

		if (name == target_name) {
			if (p_before) {
				// Insert before target
				undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", action_name, action_value);
				undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", target_name, target_value);

				undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", target_name, target_value);
				undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", action_name, action_value);
			} else {
				// Insert after target
				undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", target_name, target_value);
				undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", action_name, action_value);

				undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", action_name, action_value);
				undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", target_name, target_value);
			}

		} else if (name != action_name) {
			undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", name, value);
			undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", name, value);
		}
	}

	undo_redo->add_do_method(this, "_update_action_map_editor");
	undo_redo->add_undo_method(this, "_update_action_map_editor");
	undo_redo->add_do_method(this, "queue_save");
	undo_redo->add_undo_method(this, "queue_save");
	undo_redo->commit_action();
}

void ProjectSettingsEditor::_update_action_map_editor() {
	Vector<ActionMapEditor::ActionInfo> actions;

	List<PropertyInfo> props;
	ProjectSettings::get_singleton()->get_property_list(&props);

	const Ref<Texture2D> builtin_icon = get_theme_icon("PinPressed", "EditorIcons");
	for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {
		const String property_name = E->get().name;

		if (!property_name.begins_with("input/")) {
			continue;
		}

		// Strip the "input/" from the left.
		String display_name = property_name.substr(String("input/").size() - 1);
		Dictionary action = ProjectSettings::get_singleton()->get(property_name);

		ActionMapEditor::ActionInfo action_info;
		action_info.action = action;
		action_info.editable = true;
		action_info.name = display_name;

		const bool is_builtin_input = ProjectSettings::get_singleton()->get_input_presets().find(property_name) != nullptr;
		if (is_builtin_input) {
			action_info.editable = false;
			action_info.icon = builtin_icon;
		}

		actions.push_back(action_info);
	}

	action_map->update_action_list(actions);
}

void ProjectSettingsEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				EditorSettings::get_singleton()->set_project_metadata("dialog_bounds", "project_settings", Rect2(get_position(), get_size()));
			}
		} break;
		case NOTIFICATION_ENTER_TREE: {
			inspector->edit(ps);

			search_box->set_right_icon(get_theme_icon("Search", "EditorIcons"));
			search_box->set_clear_button_enabled(true);

			restart_close_button->set_icon(get_theme_icon("Close", "EditorIcons"));
			restart_container->add_theme_style_override("panel", get_theme_stylebox("bg", "Tree"));
			restart_icon->set_texture(get_theme_icon("StatusWarning", "EditorIcons"));
			restart_label->add_theme_color_override("font_color", get_theme_color("warning_color", "Editor"));

			_update_action_map_editor();
		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			search_box->set_right_icon(get_theme_icon("Search", "EditorIcons"));
			search_box->set_clear_button_enabled(true);
		} break;
	}
}

void ProjectSettingsEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("queue_save"), &ProjectSettingsEditor::queue_save);

	ClassDB::bind_method(D_METHOD("_update_action_map_editor"), &ProjectSettingsEditor::_update_action_map_editor);
}

ProjectSettingsEditor::ProjectSettingsEditor(EditorData *p_data) {
	singleton = this;
	set_title(TTR("Project Settings (project.godot)"));

	ps = ProjectSettings::get_singleton();
	undo_redo = &p_data->get_undo_redo();
	data = p_data;

	tab_container = memnew(TabContainer);
	tab_container->set_tab_align(TabContainer::ALIGN_LEFT);
	tab_container->set_use_hidden_tabs_for_min_size(true);
	add_child(tab_container);

	VBoxContainer *general_editor = memnew(VBoxContainer);
	general_editor->set_name(TTR("General"));
	general_editor->set_alignment(BoxContainer::ALIGN_BEGIN);
	general_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tab_container->add_child(general_editor);

	HBoxContainer *search_bar = memnew(HBoxContainer);
	general_editor->add_child(search_bar);

	search_box = memnew(LineEdit);
	search_box->set_placeholder(TTR("Filter Settings"));
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_bar->add_child(search_box);

	advanced = memnew(CheckButton);
	advanced->set_text(TTR("Advanced Settings"));
	advanced->connect("toggled", callable_mp(this, &ProjectSettingsEditor::_advanced_toggled));
	search_bar->add_child(advanced);

	HBoxContainer *header = memnew(HBoxContainer);
	general_editor->add_child(header);

	property_box = memnew(LineEdit);
	property_box->set_placeholder(TTR("Select a setting or type its name"));
	property_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	property_box->connect("text_changed", callable_mp(this, &ProjectSettingsEditor::_property_box_changed));
	header->add_child(property_box);

	feature_box = memnew(OptionButton);
	feature_box->set_custom_minimum_size(Size2(120, 0) * EDSCALE);
	feature_box->connect("item_selected", callable_mp(this, &ProjectSettingsEditor::_feature_selected));
	header->add_child(feature_box);

	type_box = memnew(OptionButton);
	type_box->set_custom_minimum_size(Size2(120, 0) * EDSCALE);
	header->add_child(type_box);

	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		// There's no point in adding Nil types, and Object types
		// can't be serialized correctly in the project settings.
		if (i != Variant::NIL && i != Variant::OBJECT) {
			type_box->add_item(Variant::get_type_name(Variant::Type(i)), i);
		}
	}

	add_button = memnew(Button);
	add_button->set_text(TTR("Add"));
	add_button->set_disabled(true);
	add_button->connect("pressed", callable_mp(this, &ProjectSettingsEditor::_add_setting));
	header->add_child(add_button);

	del_button = memnew(Button);
	del_button->set_text(TTR("Delete"));
	del_button->set_disabled(true);
	del_button->connect("pressed", callable_mp(this, &ProjectSettingsEditor::_delete_setting));
	header->add_child(del_button);

	inspector = memnew(SectionedInspector);
	inspector->get_inspector()->set_undo_redo(EditorNode::get_singleton()->get_undo_redo());
	inspector->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	inspector->register_search_box(search_box);
	inspector->get_inspector()->connect("property_selected", callable_mp(this, &ProjectSettingsEditor::_setting_selected));
	inspector->get_inspector()->connect("property_edited", callable_mp(this, &ProjectSettingsEditor::_setting_edited));
	inspector->get_inspector()->connect("restart_requested", callable_mp(this, &ProjectSettingsEditor::_editor_restart_request));
	general_editor->add_child(inspector);

	restart_container = memnew(PanelContainer);
	general_editor->add_child(restart_container);

	HBoxContainer *restart_hb = memnew(HBoxContainer);
	restart_container->hide();
	restart_container->add_child(restart_hb);

	restart_icon = memnew(TextureRect);
	restart_icon->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	restart_hb->add_child(restart_icon);

	restart_label = memnew(Label);
	restart_label->set_text(TTR("Changed settings will be applied to the editor after restarting."));
	restart_hb->add_child(restart_label);
	restart_hb->add_spacer();

	Button *restart_button = memnew(Button);
	restart_button->connect("pressed", callable_mp(this, &ProjectSettingsEditor::_editor_restart));
	restart_hb->add_child(restart_button);
	restart_button->set_text(TTR("Save & Restart"));

	restart_close_button = memnew(Button);
	restart_close_button->set_flat(true);
	restart_close_button->connect("pressed", callable_mp(this, &ProjectSettingsEditor::_editor_restart_close));
	restart_hb->add_child(restart_close_button);

	action_map = memnew(ActionMapEditor);
	action_map->set_name(TTR("Input Map"));
	action_map->connect("action_added", callable_mp(this, &ProjectSettingsEditor::_action_added));
	action_map->connect("action_edited", callable_mp(this, &ProjectSettingsEditor::_action_edited));
	action_map->connect("action_removed", callable_mp(this, &ProjectSettingsEditor::_action_removed));
	action_map->connect("action_renamed", callable_mp(this, &ProjectSettingsEditor::_action_renamed));
	action_map->connect("action_reordered", callable_mp(this, &ProjectSettingsEditor::_action_reordered));
	tab_container->add_child(action_map);

	localization_editor = memnew(LocalizationEditor);
	localization_editor->set_name(TTR("Localization"));
	localization_editor->connect("localization_changed", callable_mp(this, &ProjectSettingsEditor::queue_save));
	tab_container->add_child(localization_editor);

	autoload_settings = memnew(EditorAutoloadSettings);
	autoload_settings->set_name(TTR("AutoLoad"));
	autoload_settings->connect("autoload_changed", callable_mp(this, &ProjectSettingsEditor::queue_save));
	tab_container->add_child(autoload_settings);

	shaders_global_variables_editor = memnew(ShaderGlobalsEditor);
	shaders_global_variables_editor->set_name(TTR("Shader Globals"));
	shaders_global_variables_editor->connect("globals_changed", callable_mp(this, &ProjectSettingsEditor::queue_save));
	tab_container->add_child(shaders_global_variables_editor);

	plugin_settings = memnew(EditorPluginSettings);
	plugin_settings->set_name(TTR("Plugins"));
	tab_container->add_child(plugin_settings);

	timer = memnew(Timer);
	timer->set_wait_time(1.5);
	timer->connect("timeout", callable_mp(ps, &ProjectSettings::save));
	timer->set_one_shot(true);
	add_child(timer);

	get_ok_button()->set_text(TTR("Close"));
	set_hide_on_ok(true);

	bool use_advanced = EditorSettings::get_singleton()->get_project_metadata("project_settings", "advanced_mode", false);

	if (use_advanced) {
		advanced->set_pressed(true);
	}

	inspector->set_restrict_to_basic_settings(!use_advanced);

	import_defaults_editor = memnew(ImportDefaultsEditor);
	import_defaults_editor->set_name(TTR("Import Defaults"));
	tab_container->add_child(import_defaults_editor);
	import_defaults_editor->connect("project_settings_changed", callable_mp(this, &ProjectSettingsEditor::queue_save));
}

/*************************************************************************/
/*  project_settings_editor.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/global_constants.h"
#include "core/os/keyboard.h"
#include "core/project_settings.h"
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

	globals_editor->update_category_list();
	localization_editor->update_translations();
	autoload_settings->update_autoload();
	plugin_settings->update_plugins();
	set_process_unhandled_input(true);
}

void ProjectSettingsEditor::_unhandled_input(const Ref<InputEvent> &p_event) {
	const Ref<InputEventKey> k = p_event;

	if (k.is_valid() && k->is_pressed()) {
		if (k->get_keycode_with_modifiers() == (KEY_MASK_CMD | KEY_F)) {
			if (search_button->is_pressed()) {
				search_box->grab_focus();
				search_box->select_all();
			} else {
				// This toggles the search bar display while giving the button its "pressed" appearance
				search_button->set_pressed(true);
			}

			set_input_as_handled();
		}
	}
}

void ProjectSettingsEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				EditorSettings::get_singleton()->set_project_metadata("dialog_bounds", "project_settings", Rect2(get_position(), get_size()));
				set_process_unhandled_input(false);
			}
		} break;
		case NOTIFICATION_ENTER_TREE: {
			globals_editor->edit(ProjectSettings::get_singleton());

			search_button->set_icon(get_theme_icon("Search", "EditorIcons"));
			search_box->set_right_icon(get_theme_icon("Search", "EditorIcons"));
			search_box->set_clear_button_enabled(true);

			restart_close_button->set_icon(get_theme_icon("Close", "EditorIcons"));
			restart_container->add_theme_style_override("panel", get_theme_stylebox("bg", "Tree"));
			restart_icon->set_texture(get_theme_icon("StatusWarning", "EditorIcons"));
			restart_label->add_theme_color_override("font_color", get_theme_color("warning_color", "Editor"));

		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			search_button->set_icon(get_theme_icon("Search", "EditorIcons"));
			search_box->set_right_icon(get_theme_icon("Search", "EditorIcons"));
			search_box->set_clear_button_enabled(true);
		} break;
	}
}

void ProjectSettingsEditor::update_plugins() {
	plugin_settings->update_plugins();
}

void ProjectSettingsEditor::_item_selected(const String &p_path) {
	const String &selected_path = p_path;
	if (selected_path == String()) {
		return;
	}
	category->set_text(globals_editor->get_current_section());
	property->set_text(selected_path);
	popup_copy_to_feature->set_disabled(false);
}

void ProjectSettingsEditor::_item_adds(String) {
	_item_add();
}

void ProjectSettingsEditor::_item_add() {
	// Initialize the property with the default value for the given type.
	// The type list starts at 1 (as we exclude Nil), so add 1 to the selected value.
	Callable::CallError ce;
	const Variant value = Variant::construct(Variant::Type(type->get_selected() + 1), nullptr, 0, ce);

	String catname = category->get_text().strip_edges();
	String propname = property->get_text().strip_edges();

	if (propname.empty()) {
		return;
	}

	if (catname.empty()) {
		catname = "global";
	}

	String name = catname + "/" + propname;

	undo_redo->create_action(TTR("Add Global Property"));

	undo_redo->add_do_property(ProjectSettings::get_singleton(), name, value);

	if (ProjectSettings::get_singleton()->has_setting(name)) {
		undo_redo->add_undo_property(ProjectSettings::get_singleton(), name, ProjectSettings::get_singleton()->get(name));
	} else {
		undo_redo->add_undo_property(ProjectSettings::get_singleton(), name, Variant());
	}

	undo_redo->add_do_method(globals_editor, "update_category_list");
	undo_redo->add_undo_method(globals_editor, "update_category_list");
	undo_redo->add_do_method(this, "_settings_changed");
	undo_redo->add_undo_method(this, "_settings_changed");
	undo_redo->commit_action();

	globals_editor->set_current_section(catname);

	_settings_changed();
}

void ProjectSettingsEditor::_item_del() {
	String path = globals_editor->get_inspector()->get_selected_path();
	if (path == String()) {
		EditorNode::get_singleton()->show_warning(TTR("Select a setting item first!"));
		return;
	}

	String property = globals_editor->get_current_section().plus_file(path);

	if (!ProjectSettings::get_singleton()->has_setting(property)) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("No property '%s' exists."), property));
		return;
	}

	if (ProjectSettings::get_singleton()->get_order(property) < ProjectSettings::NO_BUILTIN_ORDER_BASE) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("Setting '%s' is internal, and it can't be deleted."), property));
		return;
	}

	undo_redo->create_action(TTR("Delete Item"));

	Variant value = ProjectSettings::get_singleton()->get(property);
	int order = ProjectSettings::get_singleton()->get_order(property);

	undo_redo->add_do_method(ProjectSettings::get_singleton(), "clear", property);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", property, value);
	undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set_order", property, order);

	undo_redo->add_do_method(globals_editor, "update_category_list");
	undo_redo->add_undo_method(globals_editor, "update_category_list");

	undo_redo->add_do_method(this, "_settings_changed");
	undo_redo->add_undo_method(this, "_settings_changed");

	undo_redo->commit_action();
}

void ProjectSettingsEditor::_save() {
	Error err = ProjectSettings::get_singleton()->save();
	message->set_text(err != OK ? TTR("Error saving settings.") : TTR("Settings saved OK."));
	message->popup_centered(Size2(300, 100) * EDSCALE);
}

void ProjectSettingsEditor::_settings_prop_edited(const String &p_name) {
	// Method needed to discard the mandatory argument of the property_edited signal
	_settings_changed();
}

void ProjectSettingsEditor::_settings_changed() {
	timer->start();
}

void ProjectSettingsEditor::queue_save() {
	_settings_changed();
}

void ProjectSettingsEditor::_copy_to_platform_about_to_show() {
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
	// Not available as an export platform yet, so it needs to be added manually
	presets.insert("Server");

	for (int i = 0; i < EditorExport::get_singleton()->get_export_platform_count(); i++) {
		List<String> p;
		EditorExport::get_singleton()->get_export_platform(i)->get_platform_features(&p);
		for (List<String>::Element *E = p.front(); E; E = E->next()) {
			presets.insert(E->get());
		}
	}

	for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); i++) {
		List<String> p;
		EditorExport::get_singleton()->get_export_preset(i)->get_platform()->get_preset_features(EditorExport::get_singleton()->get_export_preset(i), &p);
		for (List<String>::Element *E = p.front(); E; E = E->next()) {
			presets.insert(E->get());
		}

		String custom = EditorExport::get_singleton()->get_export_preset(i)->get_custom_features();
		Vector<String> custom_list = custom.split(",");
		for (int j = 0; j < custom_list.size(); j++) {
			String f = custom_list[j].strip_edges();
			if (f != String()) {
				presets.insert(f);
			}
		}
	}

	popup_copy_to_feature->get_popup()->clear();
	int id = 0;
	for (Set<String>::Element *E = presets.front(); E; E = E->next()) {
		popup_copy_to_feature->get_popup()->add_item(E->get(), id++);
	}
}

void ProjectSettingsEditor::_copy_to_platform(int p_which) {
	String path = globals_editor->get_inspector()->get_selected_path();
	if (path == String()) {
		EditorNode::get_singleton()->show_warning(TTR("Select a setting item first!"));
		return;
	}

	String property = globals_editor->get_current_section().plus_file(path);

	undo_redo->create_action(TTR("Override for Feature"));

	Variant value = ProjectSettings::get_singleton()->get(property);
	if (property.find(".") != -1) { //overwriting overwrite, keep overwrite
		undo_redo->add_do_method(ProjectSettings::get_singleton(), "clear", property);
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", property, value);
	}

	String feature = popup_copy_to_feature->get_popup()->get_item_text(p_which);
	String new_path = property + "." + feature;

	undo_redo->add_do_method(ProjectSettings::get_singleton(), "set", new_path, value);
	if (ProjectSettings::get_singleton()->has_setting(new_path)) {
		undo_redo->add_undo_method(ProjectSettings::get_singleton(), "set", new_path, ProjectSettings::get_singleton()->get(new_path));
	}

	undo_redo->add_do_method(globals_editor, "update_category_list");
	undo_redo->add_undo_method(globals_editor, "update_category_list");

	undo_redo->add_do_method(this, "_settings_changed");
	undo_redo->add_undo_method(this, "_settings_changed");

	undo_redo->commit_action();
}

void ProjectSettingsEditor::_toggle_search_bar(bool p_pressed) {
	globals_editor->get_inspector()->set_use_filter(p_pressed);

	if (p_pressed) {
		search_bar->show();
		add_prop_bar->hide();
		search_box->grab_focus();
		search_box->select_all();
	} else {
		search_box->clear();
		search_bar->hide();
		add_prop_bar->show();
	}
}

void ProjectSettingsEditor::set_plugins_page() {
	tab_container->set_current_tab(plugin_settings->get_index());
}

TabContainer *ProjectSettingsEditor::get_tabs() {
	return tab_container;
}

void ProjectSettingsEditor::_editor_restart() {
	EditorNode::get_singleton()->save_all_scenes();
	EditorNode::get_singleton()->restart_editor();
}

void ProjectSettingsEditor::_editor_restart_request() {
	restart_container->show();
}

void ProjectSettingsEditor::_editor_restart_close() {
	restart_container->hide();
}

void ProjectSettingsEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_unhandled_input"), &ProjectSettingsEditor::_unhandled_input);
	ClassDB::bind_method(D_METHOD("_save"), &ProjectSettingsEditor::_save);

	ClassDB::bind_method(D_METHOD("get_tabs"), &ProjectSettingsEditor::get_tabs);
}

ProjectSettingsEditor::ProjectSettingsEditor(EditorData *p_data) {
	singleton = this;
	set_title(TTR("Project Settings (project.godot)"));

	undo_redo = &p_data->get_undo_redo();
	data = p_data;

	tab_container = memnew(TabContainer);
	tab_container->set_tab_align(TabContainer::ALIGN_LEFT);
	tab_container->set_use_hidden_tabs_for_min_size(true);
	add_child(tab_container);

	VBoxContainer *props_base = memnew(VBoxContainer);
	props_base->set_name(TTR("General"));
	props_base->set_alignment(BoxContainer::ALIGN_BEGIN);
	props_base->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tab_container->add_child(props_base);

	HBoxContainer *hbc = memnew(HBoxContainer);
	hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	props_base->add_child(hbc);

	search_button = memnew(Button);
	search_button->set_text(TTR("Search"));
	search_button->set_toggle_mode(true);
	search_button->set_pressed(false);
	search_button->connect("toggled", callable_mp(this, &ProjectSettingsEditor::_toggle_search_bar));
	hbc->add_child(search_button);

	hbc->add_child(memnew(VSeparator));

	add_prop_bar = memnew(HBoxContainer);
	add_prop_bar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbc->add_child(add_prop_bar);

	Label *l = memnew(Label);
	l->set_text(TTR("Category:"));
	add_prop_bar->add_child(l);

	category = memnew(LineEdit);
	category->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	category->connect("text_entered", callable_mp(this, &ProjectSettingsEditor::_item_adds));
	add_prop_bar->add_child(category);

	l = memnew(Label);
	l->set_text(TTR("Property:"));
	add_prop_bar->add_child(l);

	property = memnew(LineEdit);
	property->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	property->connect("text_entered", callable_mp(this, &ProjectSettingsEditor::_item_adds));
	add_prop_bar->add_child(property);

	l = memnew(Label);
	l->set_text(TTR("Type:"));
	add_prop_bar->add_child(l);

	type = memnew(OptionButton);
	type->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_prop_bar->add_child(type);

	// Start at 1 to avoid adding "Nil" as an option
	for (int i = 1; i < Variant::VARIANT_MAX; i++) {
		type->add_item(Variant::get_type_name(Variant::Type(i)));
	}

	Button *add = memnew(Button);
	add->set_text(TTR("Add"));
	add->connect("pressed", callable_mp(this, &ProjectSettingsEditor::_item_add));
	add_prop_bar->add_child(add);

	search_bar = memnew(HBoxContainer);
	search_bar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_bar->hide();
	hbc->add_child(search_bar);

	search_box = memnew(LineEdit);
	search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	search_bar->add_child(search_box);

	globals_editor = memnew(SectionedInspector);
	globals_editor->get_inspector()->set_undo_redo(EditorNode::get_singleton()->get_undo_redo());
	globals_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	globals_editor->register_search_box(search_box);
	globals_editor->get_inspector()->connect("property_selected", callable_mp(this, &ProjectSettingsEditor::_item_selected));
	globals_editor->get_inspector()->connect("property_edited", callable_mp(this, &ProjectSettingsEditor::_settings_prop_edited));
	globals_editor->get_inspector()->connect("restart_requested", callable_mp(this, &ProjectSettingsEditor::_editor_restart_request));
	props_base->add_child(globals_editor);

	Button *del = memnew(Button);
	del->set_text(TTR("Delete"));
	del->connect("pressed", callable_mp(this, &ProjectSettingsEditor::_item_del));
	hbc->add_child(del);

	add_prop_bar->add_child(memnew(VSeparator));

	popup_copy_to_feature = memnew(MenuButton);
	popup_copy_to_feature->set_text(TTR("Override For..."));
	popup_copy_to_feature->set_disabled(true);
	add_prop_bar->add_child(popup_copy_to_feature);

	popup_copy_to_feature->get_popup()->connect("id_pressed", callable_mp(this, &ProjectSettingsEditor::_copy_to_platform));
	popup_copy_to_feature->get_popup()->connect("about_to_popup", callable_mp(this, &ProjectSettingsEditor::_copy_to_platform_about_to_show));

	get_ok()->set_text(TTR("Close"));
	set_hide_on_ok(true);

	restart_container = memnew(PanelContainer);
	props_base->add_child(restart_container);

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

	message = memnew(AcceptDialog);
	add_child(message);

	inputmap_editor = memnew(InputMapEditor);
	inputmap_editor->set_name(TTR("Input Map"));
	inputmap_editor->connect("inputmap_changed", callable_mp(this, &ProjectSettingsEditor::_settings_changed));
	tab_container->add_child(inputmap_editor);

	localization_editor = memnew(LocalizationEditor);
	localization_editor->set_name(TTR("Localization"));
	localization_editor->connect("localization_changed", callable_mp(this, &ProjectSettingsEditor::_settings_changed));
	tab_container->add_child(localization_editor);

	autoload_settings = memnew(EditorAutoloadSettings);
	autoload_settings->set_name(TTR("AutoLoad"));
	autoload_settings->connect("autoload_changed", callable_mp(this, &ProjectSettingsEditor::_settings_changed));
	tab_container->add_child(autoload_settings);

	shaders_global_variables_editor = memnew(ShaderGlobalsEditor);
	shaders_global_variables_editor->set_name(TTR("Shader Globals"));
	shaders_global_variables_editor->connect("globals_changed", callable_mp(this, &ProjectSettingsEditor::_settings_changed));
	tab_container->add_child(shaders_global_variables_editor);

	plugin_settings = memnew(EditorPluginSettings);
	plugin_settings->set_name(TTR("Plugins"));
	tab_container->add_child(plugin_settings);

	timer = memnew(Timer);
	timer->set_wait_time(1.5);
	timer->connect("timeout", callable_mp(ProjectSettings::get_singleton(), &ProjectSettings::save));
	timer->set_one_shot(true);
	add_child(timer);
}

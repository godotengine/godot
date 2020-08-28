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

	_add_feature_overrides();
	inspector->update_category_list();

	localization_editor->update_translations();
	autoload_settings->update_autoload();
	plugin_settings->update_plugins();
}

void ProjectSettingsEditor::queue_save() {
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

void ProjectSettingsEditor::_advanced_pressed() {
	if (advanced->is_pressed()) {
		_update_advanced_bar();
		advanced_bar->show();
	} else {
		advanced_bar->hide();
	}
}

void ProjectSettingsEditor::_setting_selected(const String &p_path) {
	if (p_path == String()) {
		return;
	}

	category_box->set_text(inspector->get_current_section());
	property_box->set_text(p_path);

	if (advanced_bar->is_visible()) {
		_update_advanced_bar(); // set_text doesn't trigger text_changed
	}
}

void ProjectSettingsEditor::_add_setting() {
	String setting = _get_setting_name();

	// Initialize the property with the default value for the given type.
	// The type list starts at 1 (as we exclude Nil), so add 1 to the selected value.
	Callable::CallError ce;
	const Variant value = Variant::construct(Variant::Type(type->get_selected() + 1), nullptr, 0, ce);

	undo_redo->create_action(TTR("Add Project Setting"));
	undo_redo->add_do_property(ps, setting, value);
	undo_redo->add_undo_property(ps, setting, ps->has_setting(setting) ? ps->get(setting) : Variant());

	undo_redo->add_do_method(inspector, "update_category_list");
	undo_redo->add_undo_method(inspector, "update_category_list");
	undo_redo->add_do_method(this, "queue_save");
	undo_redo->add_undo_method(this, "queue_save");
	undo_redo->commit_action();

	inspector->set_current_section(setting.get_slice("/", 1));
}

void ProjectSettingsEditor::_delete_setting(bool p_confirmed) {
	String setting = _get_setting_name();
	Variant value = ps->get(setting);
	int order = ps->get_order(setting);

	if (!p_confirmed) {
		del_confirmation->set_text(vformat(TTR("Are you sure you want to delete '%s'?"), setting));
		del_confirmation->popup_centered();
		return;
	}

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
}

void ProjectSettingsEditor::_text_field_changed(const String &p_text) {
	_update_advanced_bar();
}

void ProjectSettingsEditor::_feature_selected(int p_index) {
	_update_advanced_bar();
}

void ProjectSettingsEditor::_update_advanced_bar() {
	const String property_text = property_box->get_text().strip_edges();

	String error_msg = "";
	bool disable_add = true;
	bool disable_del = true;

	if (!property_box->get_text().empty()) {
		const String setting = _get_setting_name();
		bool setting_exists = ps->has_setting(setting);
		if (setting_exists) {
			error_msg = TTR(" - Cannot add already existing setting.");

			disable_del = ps->is_builtin_setting(setting);
			if (disable_del) {
				String msg = TTR(" - Cannot delete built-in setting.");
				error_msg += (error_msg == "") ? msg : "\n" + msg;
			}
		} else {
			bool bad_category = false; // Allow empty string.
			Vector<String> cats = category_box->get_text().strip_edges().split("/");
			for (int i = 0; i < cats.size(); i++) {
				if (!cats[i].is_valid_identifier()) {
					bad_category = true;
					error_msg = TTR(" - Invalid category name.");
					break;
				}
			}

			disable_add = bad_category;

			if (!property_text.is_valid_identifier()) {
				disable_add = true;
				String msg = TTR(" - Invalid property name.");
				error_msg += (error_msg == "") ? msg : "\n" + msg;
			}
		}
	}

	add_button->set_disabled(disable_add);
	del_button->set_disabled(disable_del);

	error_label->set_text(error_msg);
	error_label->set_visible(error_msg != "");
}

String ProjectSettingsEditor::_get_setting_name() const {
	const String cat = category_box->get_text();
	const String name = (cat.empty() ? "global" : cat.strip_edges()).plus_file(property_box->get_text().strip_edges());
	const String feature = feature_override->get_item_text(feature_override->get_selected());

	return (feature == "") ? name : (name + "." + feature);
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

	feature_override->clear();
	feature_override->add_item("", 0); // So it is always on top.
	int id = 1;
	for (Set<String>::Element *E = presets.front(); E; E = E->next()) {
		feature_override->add_item(E->get(), id++);
	}
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

void ProjectSettingsEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				EditorSettings::get_singleton()->set_project_metadata("dialog_bounds", "project_settings", Rect2(get_position(), get_size()));
				if (advanced->is_pressed()) {
					advanced->set_pressed(false);
					advanced_bar->hide();
				}
			}
		} break;
		case NOTIFICATION_ENTER_TREE: {
			inspector->edit(ps);

			error_label->add_theme_color_override("font_color", error_label->get_theme_color("error_color", "Editor"));
			add_button->set_icon(get_theme_icon("Add", "EditorIcons"));
			del_button->set_icon(get_theme_icon("Remove", "EditorIcons"));

			search_box->set_right_icon(get_theme_icon("Search", "EditorIcons"));
			search_box->set_clear_button_enabled(true);

			restart_close_button->set_icon(get_theme_icon("Close", "EditorIcons"));
			restart_container->add_theme_style_override("panel", get_theme_stylebox("bg", "Tree"));
			restart_icon->set_texture(get_theme_icon("StatusWarning", "EditorIcons"));
			restart_label->add_theme_color_override("font_color", get_theme_color("warning_color", "Editor"));
		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			search_box->set_right_icon(get_theme_icon("Search", "EditorIcons"));
			search_box->set_clear_button_enabled(true);
		} break;
	}
}

void ProjectSettingsEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("queue_save"), &ProjectSettingsEditor::queue_save);
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

	VBoxContainer *header = memnew(VBoxContainer);
	header->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	general_editor->add_child(header);

	{
		// Search bar.
		search_bar = memnew(HBoxContainer);
		search_bar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		header->add_child(search_bar);

		search_box = memnew(LineEdit);
		search_box->set_placeholder(TTR("Search"));
		search_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		search_bar->add_child(search_box);

		advanced = memnew(CheckButton);
		advanced->set_text(TTR("Advanced"));
		advanced->connect("pressed", callable_mp(this, &ProjectSettingsEditor::_advanced_pressed));
		search_bar->add_child(advanced);
	}

	{
		// Advanced bar.
		advanced_bar = memnew(VBoxContainer);
		advanced_bar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		advanced_bar->hide();
		header->add_child(advanced_bar);

		advanced_bar->add_child(memnew(HSeparator));

		HBoxContainer *hbc = memnew(HBoxContainer);
		hbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		advanced_bar->add_margin_child(TTR("Add or Remove Custom Project Settings:"), hbc, true);

		category_box = memnew(LineEdit);
		category_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		category_box->connect("text_changed", callable_mp(this, &ProjectSettingsEditor::_text_field_changed));
		category_box->set_placeholder(TTR("Category"));
		hbc->add_child(category_box);

		Label *l = memnew(Label);
		l->set_text("/");
		hbc->add_child(l);

		property_box = memnew(LineEdit);
		property_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		property_box->set_placeholder(TTR("Property"));
		property_box->connect("text_changed", callable_mp(this, &ProjectSettingsEditor::_text_field_changed));
		hbc->add_child(property_box);

		l = memnew(Label);
		l->set_text(TTR("Type:"));
		hbc->add_child(l);

		type = memnew(OptionButton);
		type->set_custom_minimum_size(Size2(100, 0) * EDSCALE);
		hbc->add_child(type);

		// Start at 1 to avoid adding "Nil" as an option
		for (int i = 1; i < Variant::VARIANT_MAX; i++) {
			type->add_item(Variant::get_type_name(Variant::Type(i)));
		}

		l = memnew(Label);
		l->set_text(TTR("Feature Override:"));
		hbc->add_child(l);

		feature_override = memnew(OptionButton);
		feature_override->set_custom_minimum_size(Size2(100, 0) * EDSCALE);
		feature_override->connect("item_selected", callable_mp(this, &ProjectSettingsEditor::_feature_selected));
		hbc->add_child(feature_override);

		add_button = memnew(Button);
		add_button->set_flat(true);
		add_button->connect("pressed", callable_mp(this, &ProjectSettingsEditor::_add_setting));
		hbc->add_child(add_button);

		del_button = memnew(Button);
		del_button->set_flat(true);
		del_button->connect("pressed", callable_mp(this, &ProjectSettingsEditor::_delete_setting), varray(false));
		hbc->add_child(del_button);

		error_label = memnew(Label);
		advanced_bar->add_child(error_label);
	}

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

	inputmap_editor = memnew(InputMapEditor);
	inputmap_editor->set_name(TTR("Input Map"));
	inputmap_editor->connect("inputmap_changed", callable_mp(this, &ProjectSettingsEditor::queue_save));
	tab_container->add_child(inputmap_editor);

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

	del_confirmation = memnew(ConfirmationDialog);
	del_confirmation->connect("confirmed", callable_mp(this, &ProjectSettingsEditor::_delete_setting), varray(true));
	add_child(del_confirmation);

	get_ok()->set_text(TTR("Close"));
	set_hide_on_ok(true);
}

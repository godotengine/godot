/**************************************************************************/
/*  extension_class_create_dialog.cpp                                     */
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

#include "extension_class_create_dialog.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/extension/extension_source_code_manager.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/file_system/editor_paths.h"
#include "editor/gui/create_dialog.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/gui/editor_validation_panel.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/line_edit.h"
#include "scene/theme/theme_db.h"

bool ExtensionClassCreateDialog::TemplateOptions::_set(const StringName &p_name, const Variant &p_value) {
	if (!options.has(p_name)) {
		return false;
	}
	options[p_name].variant = p_value;
	return true;
}

bool ExtensionClassCreateDialog::TemplateOptions::_get(const StringName &p_name, Variant &r_ret) const {
	if (!options.has(p_name)) {
		return false;
	}
	r_ret = options[p_name].variant;
	return true;
}

void ExtensionClassCreateDialog::TemplateOptions::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const KeyValue<StringName, VariantContainer> &E : options) {
		const VariantContainer &option = E.value;
		p_list->push_back(option.info);
	}
}

bool ExtensionClassCreateDialog::TemplateOptions::_property_can_revert(const StringName &p_name) const {
	return options.has(p_name);
}

bool ExtensionClassCreateDialog::TemplateOptions::_property_get_revert(const StringName &p_name, Variant &r_property) const {
	if (!options.has(p_name)) {
		return false;
	}
	r_property = options[p_name].initial.duplicate();
	return true;
}

void ExtensionClassCreateDialog::TemplateOptions::config(TypedArray<Dictionary> p_options) {
	options.clear();

	for (Dictionary option_dict : p_options) {
		PropertyInfo pinfo = PropertyInfo::from_dict(option_dict);
		Variant default_value = option_dict["default_value"];

		TemplateOptions::VariantContainer option;
		option.info = pinfo;
		option.variant = default_value.duplicate();
		option.initial = default_value;
		options[pinfo.name] = option;
	}
}

void ExtensionClassCreateDialog::_language_changed(int p_language_index) {
	int plugin_idx = language_menu->get_item_metadata(p_language_index);
	selected_source_code_plugin = ExtensionSourceCodeManager::get_singleton()->get_plugin_at_index(plugin_idx);

	validation_context->clear_scope(VALIDATION_SCOPE_PATH);
	_update_path_edits();
	_adjust_paths();

	_update_template_menu();

	EditorSettings::get_singleton()->set_project_metadata("class_setup", "last_selected_language", selected_source_code_plugin->get_language_name());
}

void ExtensionClassCreateDialog::_class_name_changed(const String &p_class_name) {
	validation_context->set_current_scope(VALIDATION_SCOPE_CLASS_NAME);
	validation_context->clear_scope();
	_validate_class_name(validation_context, p_class_name);
	validation_panel->update();
	_adjust_paths();
}

void ExtensionClassCreateDialog::_inherited_class_name_changed(const String &p_class_name) {
	validation_context->set_current_scope(VALIDATION_SCOPE_INHERITED_CLASS_NAME);
	validation_context->clear_scope();
	_validate_inherited_class_name(validation_context, p_class_name);
	validation_panel->update();

	_update_template_menu();
}

void ExtensionClassCreateDialog::_use_template_pressed() {
	is_using_templates = use_templates->is_pressed();
	// We use the same setting as the script templates intentionally, since it's the same feature
	// just applied to extension classes so we want to remember the user's choice.
	EditorSettings::get_singleton()->set("_script_setup_use_script_templates", is_using_templates);
	validation_panel->update();

	_update_template_menu();
}

void ExtensionClassCreateDialog::_template_changed(int p_template_index) {
	validation_context->clear_scope(VALIDATION_SCOPE_TEMPLATE_OPTIONS);

	if (p_template_index == -1 || !is_using_templates) {
		// The template can be unselected when switching to a plugin that doesn't provide templates
		// or the user explicitly unchecks the checkbox.
		template_options_label->hide();
		template_options_inspector->hide();
		_update_template_label(TTRC("Empty"), TTRC("Empty template suitable for all Objects."));
		return;
	}

	Dictionary template_metadata = template_menu->get_item_metadata(p_template_index);
	const String template_id_or_path = template_metadata["template_id_or_path"];
	const TemplateLocation template_location = template_metadata["template_location"];

	// Update last used dictionaries.
	const String inherited_class_name = inherited_class_name_edit->get_text();
	if (is_using_templates) {
		switch (template_location) {
			case TEMPLATE_LOCATION_PROJECT: {
				// Save the last used template for this type into the project dictionary.
				Dictionary last_used_templates_project = EditorSettings::get_singleton()->get_project_metadata("class_setup", "templates_dictionary", Dictionary());
				last_used_templates_project[inherited_class_name] = template_id_or_path;
				EditorSettings::get_singleton()->set_project_metadata("class_setup", "templates_dictionary", last_used_templates_project);
			} break;

			case TEMPLATE_LOCATION_EDITOR:
			case TEMPLATE_LOCATION_PLUGIN: {
				// Save the last used template for this type into the editor dictionary.
				Dictionary last_used_templates_editor = EDITOR_GET("_class_setup_templates_dictionary");
				last_used_templates_editor[inherited_class_name] = template_id_or_path;
				EditorSettings::get_singleton()->set("_class_setup_templates_dictionary", last_used_templates_editor);
				// Remove template from project dictionary as we last used an editor level template.
				Dictionary last_used_templates_project = EditorSettings::get_singleton()->get_project_metadata("class_setup", "templates_dictionary", Dictionary());
				if (last_used_templates_project.has(inherited_class_name)) {
					last_used_templates_project.erase(inherited_class_name);
					EditorSettings::get_singleton()->set_project_metadata("class_setup", "templates_dictionary", last_used_templates_project);
				}
			} break;
		}
	}

	switch (template_location) {
		case TEMPLATE_LOCATION_PLUGIN: {
			const String template_id = template_id_or_path;
			const String template_name = selected_source_code_plugin->get_template_display_name(template_id);
			const String template_description = selected_source_code_plugin->get_template_description(template_id);
			_update_template_label(template_name, template_description);

			// Update template options fields.
			TypedArray<Dictionary> template_options = selected_source_code_plugin->get_template_options(template_id);
			if (template_options.is_empty()) {
				template_options_label->hide();
				template_options_inspector->hide();
			} else {
				template_options_label->show();
				template_options_inspector->show();
				template_options_object->config(template_options);
				template_options_inspector->update_tree();
			}
		} break;

		case TEMPLATE_LOCATION_EDITOR:
		case TEMPLATE_LOCATION_PROJECT: {
			const String template_path = template_id_or_path;
			const String template_name = selected_source_code_plugin->get_template_file_display_name(template_path);
			const String template_description = selected_source_code_plugin->get_template_file_description(template_path);
			_update_template_label(template_name, template_description);

			// Template files don't have options.
			template_options_label->hide();
			template_options_inspector->hide();
		} break;
	}
}

void ExtensionClassCreateDialog::_template_option_changed(const String &p_option_name) {
	int template_index = template_menu->get_selected();
	ERR_FAIL_COND_MSG(template_index == -1, "Template option changed without a template selected.");
	Dictionary template_metadata = template_menu->get_item_metadata(template_index);
	const String template_id = template_metadata["template_id_or_path"];

	validation_context->set_current_scope(VALIDATION_SCOPE_TEMPLATE_OPTIONS + "/" + p_option_name);
	validation_context->clear_scope();
	_validate_template_option(validation_context, template_id, p_option_name, template_options_object->get(p_option_name));
	validation_panel->update();
}

void ExtensionClassCreateDialog::_path_changed(int p_path_index, const String &p_path) {
	ERR_FAIL_COND_MSG(p_path_index < 0 || (uint32_t)p_path_index >= path_edits.size(), "Invalid path index.");
	const String path_label = path_edits[p_path_index].label->get_text().trim_suffix(":");

	validation_context->set_current_scope(VALIDATION_SCOPE_PATH + "/" + path_label);
	validation_context->clear_scope();
	_validate_path(validation_context, p_path_index, p_path);
	validation_panel->update();
}

void ExtensionClassCreateDialog::_path_changed_bind(const String &p_path, int p_path_index) {
	_path_changed(p_path_index, p_path);
}

void ExtensionClassCreateDialog::_validate_class_name(ValidationContext *p_validation_context, const String &p_class_name) {
	if (ClassDB::class_exists(p_class_name)) {
		p_validation_context->add_validation(ValidationContext::VALIDATION_SEVERITY_ERROR, TTR("A class with the same name already exists."));
	}

	selected_source_code_plugin->validate_class_name(p_validation_context, p_class_name);
}

void ExtensionClassCreateDialog::_validate_inherited_class_name(ValidationContext *p_validation_context, const String &p_class_name) {
	if (p_class_name.length() == 0) {
		p_validation_context->add_validation(ValidationContext::VALIDATION_SEVERITY_ERROR, TTR("Inherited class name cannot be empty."));
		return;
	}

	if (!ClassDB::class_exists(p_class_name)) {
		p_validation_context->add_validation(ValidationContext::VALIDATION_SEVERITY_ERROR, TTR("Inherited class name does not exist."));
	}
}

void ExtensionClassCreateDialog::_validate_template_option(ValidationContext *p_validation_context, const String &p_template_id, const String &p_option_name, const Variant &p_value) {
	selected_source_code_plugin->validate_template_option(p_validation_context, p_template_id, p_option_name, p_value);
}

void ExtensionClassCreateDialog::_validate_path(ValidationContext *p_validation_context, int p_path_index, const String &p_path) {
	const String path = p_path.strip_edges();

	if (path.is_empty()) {
		p_validation_context->add_validation(ValidationContext::VALIDATION_SEVERITY_ERROR, TTR("Path is empty."));
		return;
	}
	if (path.get_file().get_basename().is_empty()) {
		p_validation_context->add_validation(ValidationContext::VALIDATION_SEVERITY_ERROR, TTR("Filename is empty."));
		return;
	}
	if (!path.get_file().get_basename().is_valid_filename()) {
		p_validation_context->add_validation(ValidationContext::VALIDATION_SEVERITY_ERROR, TTR("Filename is invalid."));
		return;
	}
	if (path.get_file().begins_with(".")) {
		p_validation_context->add_validation(ValidationContext::VALIDATION_SEVERITY_ERROR, TTR("Name begins with a dot."));
		return;
	}

	if (DirAccess::dir_exists_absolute(path)) {
		p_validation_context->add_validation(ValidationContext::VALIDATION_SEVERITY_ERROR, TTR("A directory with the same name exists."));
		return;
	}
	if (FileAccess::exists(path)) {
		p_validation_context->add_validation(ValidationContext::VALIDATION_SEVERITY_WARNING, TTR("File already exists, it will be overwritten."));
	}

	// Let the plugin do additional validation.
	selected_source_code_plugin->validate_path(p_validation_context, p_path_index, path);
}

void ExtensionClassCreateDialog::_browse_inherited_class() {
	// GDExtension classes can't inherit from Scripts but they still show up in the dialog
	// if they are registered as a "global class", so we need to make sure we filter them.
	HashSet<StringName> type_blocklist;
	LocalVector<StringName> script_global_class_list;
	ScriptServer::get_global_class_list(script_global_class_list);
	for (const StringName &class_name : script_global_class_list) {
		type_blocklist.insert(class_name);
	}
	select_inherited_class_dialog->set_type_blocklist(type_blocklist);

	select_inherited_class_dialog->set_base_type(base_type);
	select_inherited_class_dialog->popup_create(true);
	select_inherited_class_dialog->set_title(vformat(TTR("Inherit %s"), base_type));
	select_inherited_class_dialog->set_ok_button_text(TTR("Inherit"));
}

void ExtensionClassCreateDialog::_inherited_class_selected() {
	inherited_class_name_edit->set_text(select_inherited_class_dialog->get_selected_type().get_slice(" ", 0));
	_inherited_class_name_changed(inherited_class_name_edit->get_text());
}

void ExtensionClassCreateDialog::_browse_path(int p_path_index) {
	browsing_path_index = p_path_index;

	select_path_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	select_path_dialog->set_title(TTR("Choose Location"));
	select_path_dialog->set_ok_button_text(TTR("Save"));
	select_path_dialog->clear_filters();

	// Let the plugin configure the dialog. For example, to add extension filters.
	selected_source_code_plugin->configure_select_path_dialog(p_path_index, select_path_dialog);

	select_path_dialog->set_current_path(select_path_dialog->get_text());
	select_path_dialog->popup_file_dialog();
}

void ExtensionClassCreateDialog::_path_selected(const String &p_path) {
	LineEdit *path_edit = path_edits[browsing_path_index].line_edit;

	const String path = ProjectSettings::get_singleton()->localize_path(p_path);
	path_edit->set_text(path);
	_path_changed(browsing_path_index, path);

	// Select the filename part of the path so the user can quickly edit it.
	const String filename = path.get_file().get_basename();
	int select_start = path.rfind(filename);
	path_edit->select(select_start, select_start + filename.length());
	path_edit->set_caret_column(select_start + filename.length());
	path_edit->grab_focus();
}

void ExtensionClassCreateDialog::ok_pressed() {
	_create_new();
	validation_panel->update();
}

void ExtensionClassCreateDialog::_create_new() {
	const String class_name = class_name_edit->get_text();

	const String parent_class = inherited_class_name_edit->get_text();
	if (!ClassDB::class_exists(parent_class)) {
		ERR_FAIL_MSG("Parent class does not exist: " + parent_class);
	}

	PackedStringArray paths;
	for (uint32_t i = 0; i < path_edits.size(); i++) {
		LineEdit *path = path_edits[i].line_edit;
		paths.push_back(path->get_text());
	}
	ERR_FAIL_COND_MSG(paths.is_empty(), "No paths specified for the class files.");

	Error err = ERR_UNAVAILABLE;
	if (is_using_templates) {
		int template_index = template_menu->get_selected();
		Dictionary template_metadata = template_menu->get_item_metadata(template_index);
		const TemplateLocation template_location = template_metadata["template_location"];

		switch (template_location) {
			case TEMPLATE_LOCATION_PLUGIN: {
				// Using a template provided by the plugin, so we need to retrieve the ID and options data.
				// The plugin provided the ID and what options the template has so it must support this.
				const String template_id = template_metadata["template_id_or_path"];

				Dictionary template_options;
				List<PropertyInfo> template_options_plist;
				template_options_object->get_property_list(&template_options_plist);
				for (const PropertyInfo &info : template_options_plist) {
					template_options[info.name] = template_options_object->get(info.name);
				}

				err = selected_source_code_plugin->create_class_source_from_template_id(class_name, parent_class, paths, template_id, template_options);
			} break;

			case TEMPLATE_LOCATION_EDITOR:
			case TEMPLATE_LOCATION_PROJECT: {
				// Using a template file, so we need to retrieve the template file path for the plugin to parse.
				// The plugin is responsible from parsing the template file (and caching if needed), then using the
				// parsed data to create the class source.
				const String template_path = template_metadata["template_id_or_path"];

				err = selected_source_code_plugin->create_class_source_from_template_file(class_name, parent_class, paths, template_path);
			} break;
		}
	} else {
		// Not using templates, so we can call the plugin directly and it can use some default scaffolding
		// or whatever is appropriate.
		err = selected_source_code_plugin->create_class_source(class_name, parent_class, paths);
	}
	if (err != OK) {
		alert->set_text(TTR("Error - Could not create class in filesystem."));
		alert->popup_centered();
		return;
	}

	EditorFileSystem::get_singleton()->update_files(paths);

	emit_signal(SNAME("class_created"), class_name);
	hide();

	if (selected_source_code_plugin->overrides_external_editor()) {
		selected_source_code_plugin->open_in_external_editor(paths[0], 0, 0);
	} else {
		EditorNode::get_singleton()->load_resource(paths[0]);
	}
}

static Vector<String> _get_hierarchy(const String &p_class_name) {
	Vector<String> hierarchy;

	String class_name = p_class_name;
	while (true) {
		if (ClassDB::class_exists(class_name)) {
			hierarchy.push_back(class_name);
			class_name = ClassDB::get_parent_class(class_name);
			continue;
		}

		break;
	}

	if (hierarchy.is_empty()) {
		hierarchy.push_back("Object");
	}

	return hierarchy;
}

String ExtensionClassCreateDialog::_get_template_location_label(const ExtensionClassCreateDialog::TemplateLocation &p_template_location) const {
	switch (p_template_location) {
		case TEMPLATE_LOCATION_PLUGIN:
			return TTRC("Built-in");
		case TEMPLATE_LOCATION_EDITOR:
			return TTRC("Editor");
		case TEMPLATE_LOCATION_PROJECT:
			return TTRC("Project");
	}
	return "";
}

String ExtensionClassCreateDialog::_get_template_display_name(const String &p_template_id_or_path, const TemplateLocation &p_template_location) const {
	switch (p_template_location) {
		case TEMPLATE_LOCATION_PLUGIN: {
			const String template_id = p_template_id_or_path;
			const String display_name = selected_source_code_plugin->get_template_display_name(template_id);
			if (!display_name.is_empty()) {
				return display_name;
			}
			// Fallback to the template ID.
			return template_id.capitalize();
		}
		case TEMPLATE_LOCATION_EDITOR:
		case TEMPLATE_LOCATION_PROJECT: {
			const String template_path = p_template_id_or_path;
			const String display_name = selected_source_code_plugin->get_template_file_display_name(template_path);
			if (!display_name.is_empty()) {
				return display_name;
			}
			// Fallback to the file name.
			return template_path.get_file().get_basename().capitalize();
		}
	}
	ERR_FAIL_V_MSG(String(), "Unknown template location.");
}

PackedStringArray ExtensionClassCreateDialog::_get_user_template_files(const String &p_base_class_name, const String &p_root_path) const {
	PackedStringArray user_templates;
	const String templates_path = p_root_path.path_join(p_base_class_name);

	Error err;
	Ref<DirAccess> directory = DirAccess::open(templates_path, &err);
	if (directory.is_null() || err != OK) {
		// Directory may not exists if there are no templates for the specific base class.
		return user_templates;
	}

	directory->set_include_navigational(false);
	directory->list_dir_begin();
	String file_name = directory->get_next();
	while (!file_name.is_empty()) {
		if (!directory->current_is_dir()) {
			const String template_file_path = templates_path.path_join(file_name);
			if (selected_source_code_plugin->can_handle_template_file(template_file_path)) {
				user_templates.append(template_file_path);
			}
		}
		file_name = directory->get_next();
	}
	directory->list_dir_end();

	return user_templates;
}

PackedStringArray ExtensionClassCreateDialog::_get_templates_from_location(const ExtensionClassCreateDialog::TemplateLocation &p_template_location, const String &p_base_class_name) const {
	switch (p_template_location) {
		case TEMPLATE_LOCATION_PLUGIN: {
			return selected_source_code_plugin->get_available_templates(p_base_class_name);
		}
		case TEMPLATE_LOCATION_EDITOR: {
			// We use the script templates directory for backwards compatibility.
			const String templates_directory = EditorPaths::get_singleton()->get_script_templates_dir();
			return _get_user_template_files(p_base_class_name, templates_directory);
		}
		case TEMPLATE_LOCATION_PROJECT: {
			// We use the script templates directory for backwards compatibility.
			const String templates_directory = EditorPaths::get_singleton()->get_project_script_templates_dir();
			return _get_user_template_files(p_base_class_name, templates_directory);
		}
	}
	return PackedStringArray();
}

void ExtensionClassCreateDialog::_update_language_menu() {
	language_menu->clear();

	int previous_language = -1;
	default_language = -1;
	const String last_language = EditorSettings::get_singleton()->get_project_metadata("class_setup", "last_selected_language", "");

	for (int i = 0; i < ExtensionSourceCodeManager::get_singleton()->get_plugin_count(); i++) {
		const Ref<EditorExtensionSourceCodePlugin> &source_code_plugin = ExtensionSourceCodeManager::get_singleton()->get_plugin_at_index(i);
		if (!source_code_plugin->can_create_class_source()) {
			continue;
		}

		String lang = source_code_plugin->get_language_name();
		language_menu->add_item(lang);

		int lang_idx = language_menu->get_item_count() - 1;
		language_menu->set_item_metadata(lang_idx, i);

		if (initialized) {
			const int icon_size = get_theme_constant(SNAME("class_icon_size"), EditorStringName(Editor));
			Ref<Texture2D> language_icon = source_code_plugin->get_language_icon();
			if (language_icon.is_valid()) {
				language_menu->get_popup()->set_item_icon_max_width(lang_idx, icon_size);
				language_menu->set_item_icon(lang_idx, language_icon);
			}
		}

		if (lang == "C#") {
			default_language = lang_idx;
		}

		if (selected_source_code_plugin.is_valid()) {
			// If we have a selected source code plugin, see if it matches the current plugin.
			if (selected_source_code_plugin == source_code_plugin) {
				previous_language = lang_idx;
			}
		} else {
			// Otherwise, check if it matches the last used language saved in project settings
			// which would survive editor restarts.
			if (lang == last_language) {
				previous_language = lang_idx;
			}
		}
	}

	// If there was a previous language selection, reselect it.
	// Otherwise, if we found the default language, select that.
	if (previous_language >= 0) {
		language_menu->select(previous_language);
	} else if (default_language >= 0) {
		language_menu->select(default_language);
	}

	_language_changed(language_menu->get_selected());
}

void ExtensionClassCreateDialog::_update_template_menu() {
	bool is_language_using_templates = selected_source_code_plugin->is_using_templates();
	template_menu->set_disabled(false);
	template_menu->clear();

	if (is_language_using_templates) {
		// Get the latest templates used for each type from the project settings then global settings.
		Dictionary last_used_templates_project = EditorSettings::get_singleton()->get_project_metadata("class_setup", "templates_dictionary", Dictionary());
		Dictionary last_used_templates_editor = EDITOR_GET("_class_setup_templates_dictionary");
		const String inherited_class_name = inherited_class_name_edit->get_text();

		// Get all ancestor types for the selected base type.
		// Their templates will also fit the base type.
		Vector<String> hierarchy = _get_hierarchy(inherited_class_name);
		int last_used_template = -1;
		int preselected_template = -1;
		int previous_ancestor_level = -1;

		// Templates can be stored in three different locations.
		Vector<TemplateLocation> template_locations;
		if (selected_source_code_plugin->can_use_template_files()) {
			template_locations.push_back(TEMPLATE_LOCATION_PROJECT);
			template_locations.push_back(TEMPLATE_LOCATION_EDITOR);
		}
		template_locations.push_back(TEMPLATE_LOCATION_PLUGIN);

		for (const TemplateLocation &template_location : template_locations) {
			const String display_name = _get_template_location_label(template_location);
			bool separator = false;
			int ancestor_level = 0;
			for (const String &current_type : hierarchy) {
				PackedStringArray available_templates = _get_templates_from_location(template_location, current_type);
				if (!available_templates.is_empty()) {
					if (!separator) {
						template_menu->add_separator();
						template_menu->set_item_text(-1, display_name);
						template_menu->set_item_auto_translate_mode(-1, AUTO_TRANSLATE_MODE_ALWAYS);
						separator = true;
					}
					for (const String &template_id_or_path : available_templates) {
						const String template_name = _get_template_display_name(template_id_or_path, template_location);
						template_menu->add_item(template_name);
						Dictionary template_metadata;
						template_metadata["template_id_or_path"] = template_id_or_path;
						template_metadata["template_location"] = template_location;
						template_menu->set_item_metadata(template_menu->get_item_count() - 1, template_metadata);
						int idx = template_menu->get_item_count() - 1;
						// Check if this template should be preselected if the type isn't in the last used dictionary.
						if (ancestor_level < previous_ancestor_level || previous_ancestor_level == -1) {
							previous_ancestor_level = ancestor_level;
							preselected_template = idx;
						}
						// Check for last used templates for this type in project settings then in editor settings.
						if (last_used_templates_project.has(inherited_class_name) && template_id_or_path == String(last_used_templates_project[inherited_class_name])) {
							last_used_template = idx;
						} else if (last_used_template == -1 && last_used_templates_editor.has(inherited_class_name) && template_id_or_path == String(last_used_templates_editor[inherited_class_name])) {
							last_used_template = idx;
						}
						String icon = has_theme_icon(current_type, EditorStringName(EditorIcons)) ? current_type : "Object";
						template_menu->set_item_icon(idx, get_editor_theme_icon(icon));
					}
				}
				ancestor_level++;
			}
		}

		if (last_used_template != -1) {
			template_menu->select(last_used_template);
		} else if (preselected_template != -1) {
			template_menu->select(preselected_template);
		}
	}

	_template_changed(template_menu->get_selected());
}

void ExtensionClassCreateDialog::_update_template_label(const String &p_template_name, const String &p_template_description) {
	String template_info = U"•  ";
	template_info += TTR("Template:");
	template_info += " " + p_template_name;
	if (!p_template_description.is_empty()) {
		template_info += " - " + p_template_description;
	}
	validation_panel->set_message(MSG_ID_TEMPLATE, template_info, EditorValidationPanel::MSG_INFO, false);
}

void ExtensionClassCreateDialog::_update_path_edits() {
	uint32_t path_count = selected_source_code_plugin->get_path_count();
	if (path_edits.size() != path_count) {
		_update_path_edit(0, path_edits[0]);
		for (uint32_t i = 1; i < path_edits.size(); i++) {
			_free_path_edit(path_edits[i]);
		}
		path_edits.resize(path_count);

		int insert_index = path_edit_insert_index;
		for (uint32_t i = 1; i < path_edits.size(); i++) {
			path_edits[i] = _create_path_edit(i);
			gc->add_child(path_edits[i].label);
			gc->add_child(path_edits[i].hb);
			gc->move_child(path_edits[i].label, insert_index++);
			gc->move_child(path_edits[i].hb, insert_index++);
		}
	} else {
		for (uint32_t i = 0; i < path_edits.size(); i++) {
			_update_path_edit(i, path_edits[i]);
		}
	}
}

void ExtensionClassCreateDialog::_adjust_paths() {
	for (uint32_t i = 0; i < path_edits.size(); i++) {
		LineEdit *path_edit = path_edits[i].line_edit;
		String new_path = selected_source_code_plugin->adjust_path(i, class_name_edit->get_text(), base_path, path_edit->get_text());
		path_edit->set_text(new_path);
		_path_changed(i, new_path);
	}
}

void ExtensionClassCreateDialog::_update_validation_messages(const String &p_scope, int p_msg_id, const String &p_ok_message) {
	ValidationContext::ValidationSeverity highest_severity = ValidationContext::VALIDATION_SEVERITY_INFO;

	Vector<ValidationContext::ValidationInfo> validations = validation_context->get_validations_for_scope(p_scope);
	if (!validations.is_empty()) {
		String messages;
		for (const ValidationContext::ValidationInfo &validation : validations) {
			String prefix;
			if (validation.scope.begins_with(p_scope + "/")) {
				prefix = validation.scope.substr(p_scope.length() + 1);
			}

			if (!validation.message.is_empty()) {
				if (!messages.is_empty()) {
					messages += "\n";
				}
				messages += U"•  ";
				if (!prefix.is_empty()) {
					messages += prefix.capitalize() + " - ";
				}
				messages += validation.message;
			}

			if (validation.severity > highest_severity) {
				highest_severity = validation.severity;
			}
		}
		if (!messages.is_empty()) {
			EditorValidationPanel::MessageType msg_type = EditorValidationPanel::MSG_OK;
			switch (highest_severity) {
				case ValidationContext::VALIDATION_SEVERITY_WARNING:
					msg_type = EditorValidationPanel::MSG_WARNING;
					break;
				case ValidationContext::VALIDATION_SEVERITY_ERROR:
					msg_type = EditorValidationPanel::MSG_ERROR;
					break;
				case ValidationContext::VALIDATION_SEVERITY_INFO:
					msg_type = EditorValidationPanel::MSG_INFO;
					break;
			}
			validation_panel->set_message(p_msg_id, messages, msg_type, false);
			return;
		}
	}

	validation_panel->set_message(p_msg_id, p_ok_message, EditorValidationPanel::MSG_OK);
}

void ExtensionClassCreateDialog::_update_dialog() {
	_update_validation_messages(VALIDATION_SCOPE_CLASS_NAME, MSG_ID_CLASS, TTR("Class name is valid."));
	_update_validation_messages(VALIDATION_SCOPE_INHERITED_CLASS_NAME, MSG_ID_INHERITED_CLASS);
	_update_validation_messages(VALIDATION_SCOPE_PATH, MSG_ID_PATH, TTR("File paths are valid."));
	_update_validation_messages(VALIDATION_SCOPE_TEMPLATE_OPTIONS, MSG_ID_TEMPLATE_OPTIONS);

	// Show templates list if needed.
	String template_inactive_message = "";
	if (is_using_templates) {
		// Check if at least one suitable template has been found.
		if (template_menu->get_item_count() == 0) {
			template_inactive_message = TTRC("No suitable template.");
		}
	} else {
		template_inactive_message = TTRC("Empty");
	}
	if (!template_inactive_message.is_empty()) {
		template_menu->set_disabled(true);
		template_menu->clear();
		template_menu->add_item(template_inactive_message);
		template_menu->set_item_auto_translate_mode(-1, AUTO_TRANSLATE_MODE_ALWAYS);
		validation_panel->set_message(MSG_ID_TEMPLATE, "", EditorValidationPanel::MSG_INFO);
	}
}

void ExtensionClassCreateDialog::_focus_on_class_name_edit() {
	if (is_visible()) {
		class_name_edit->select(0, class_name_edit->get_text().length());
		class_name_edit->grab_focus();
	}
}

ExtensionClassCreateDialog::PathEdit ExtensionClassCreateDialog::_create_path_edit(int p_path_index) {
	PathEdit path_edit;

	path_edit.label = memnew(Label);
	path_edit.hb = memnew(HBoxContainer);
	path_edit.hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	path_edit.line_edit = memnew(LineEdit);
	path_edit.line_edit->connect(SceneStringName(text_changed), callable_mp(this, &ExtensionClassCreateDialog::_path_changed_bind).bind(p_path_index));
	path_edit.line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	path_edit.hb->add_child(path_edit.line_edit);
	register_text_enter(path_edit.line_edit);

	path_edit.button = memnew(Button);
	path_edit.button->set_accessibility_name(TTRC("Select File"));
	path_edit.button->connect(SceneStringName(pressed), callable_mp(this, &ExtensionClassCreateDialog::_browse_path).bind(p_path_index));
	path_edit.hb->add_child(path_edit.button);

	_update_path_edit(p_path_index, path_edit);

	return path_edit;
}

void ExtensionClassCreateDialog::_update_path_edit(int p_path_index, const PathEdit &p_path_edit) {
	String path_label;
	if (selected_source_code_plugin.is_valid()) {
		path_label = selected_source_code_plugin->get_path_label(p_path_index);
	}
	if (path_label.is_empty()) {
		path_label = "Path:";
	}

	p_path_edit.label->set_text(TTR(path_label));
	p_path_edit.line_edit->set_accessibility_name(TTRC(path_label));
	p_path_edit.line_edit->set_text("");

	if (initialized) {
		p_path_edit.button->set_button_icon(get_editor_theme_icon(SNAME("Folder")));
	}
}

void ExtensionClassCreateDialog::_free_path_edit(const PathEdit &p_path_edit) {
	p_path_edit.label->queue_free();
	p_path_edit.hb->queue_free();
}

void ExtensionClassCreateDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			initialized = true;
		} break;

		case NOTIFICATION_ENTER_TREE: {
			// We use the same setting as the script templates intentionally, since it's the same feature
			// just applied to extension classes so we want to remember the user's choice.
			is_using_templates = EDITOR_GET("_script_setup_use_script_templates");
			use_templates->set_pressed(is_using_templates);
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			const int icon_size = get_theme_constant(SNAME("class_icon_size"), EditorStringName(Editor));
			for (int lang_idx = 0; lang_idx < language_menu->get_item_count(); lang_idx++) {
				int i = language_menu->get_item_metadata(lang_idx);
				const Ref<EditorExtensionSourceCodePlugin> source_code_plugin = ExtensionSourceCodeManager::get_singleton()->get_plugin_at_index(i);
				Ref<Texture2D> language_icon = source_code_plugin->get_language_icon();
				if (language_icon.is_valid()) {
					language_menu->get_popup()->set_item_icon_max_width(lang_idx, icon_size);
					language_menu->set_item_icon(lang_idx, language_icon);
				}
			}

			for (uint32_t i = 0; i < path_edits.size(); i++) {
				path_edits[i].button->set_button_icon(get_editor_theme_icon(SNAME("Folder")));
			}
			inherited_class_search_button->set_button_icon(get_editor_theme_icon(SNAME("ClassList")));
		} break;
	}
}

void ExtensionClassCreateDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("config", "inherits", "path"), &ExtensionClassCreateDialog::config, DEFVAL(true), DEFVAL(true));

	ADD_SIGNAL(MethodInfo("class_created", PropertyInfo(Variant::STRING, "class_name")));
}

void ExtensionClassCreateDialog::config(const String &p_base_name, const String &p_base_path) {
	class_name_edit->set_text("NewClass");

	inherited_class_name_edit->set_text(p_base_name);
	inherited_class_name_edit->deselect();

	base_path = p_base_path;

	_update_language_menu();
	_class_name_changed(class_name_edit->get_text());
	_inherited_class_name_changed(inherited_class_name_edit->get_text());
}

ExtensionClassCreateDialog::ExtensionClassCreateDialog() {
	if (EditorSettings::get_singleton()) {
		EDITOR_DEF("_class_setup_templates_dictionary", Dictionary());
	}

	/* Main Controls */

	gc = memnew(GridContainer);
	gc->set_columns(2);

	/* Information Messages Field */

	validation_context = memnew(ValidationContext);

	validation_panel = memnew(EditorValidationPanel);
	validation_panel->add_line(MSG_ID_CLASS, TTR("Class name is valid."));
	validation_panel->add_line(MSG_ID_INHERITED_CLASS);
	validation_panel->add_line(MSG_ID_PATH, TTR("File paths are valid."));
	validation_panel->add_line(MSG_ID_TEMPLATE);
	validation_panel->add_line(MSG_ID_TEMPLATE_OPTIONS);
	validation_panel->set_update_callback(callable_mp(this, &ExtensionClassCreateDialog::_update_dialog));
	validation_panel->set_accept_button(get_ok_button());

	/* Spacing */

	Control *spacing = memnew(Control);
	spacing->set_custom_minimum_size(Size2(0, 10 * EDSCALE));

	VBoxContainer *vb = memnew(VBoxContainer);
	vb->add_child(gc);
	vb->add_child(spacing);
	vb->add_child(validation_panel);
	add_child(vb);

	/* Language */

	gc->add_child(memnew(Label(TTR("Language:"))));
	language_menu = memnew(OptionButton);
	language_menu->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	language_menu->set_custom_minimum_size(Size2(350, 0) * EDSCALE);
	language_menu->set_expand_icon(true);
	language_menu->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	language_menu->set_accessibility_name(TTRC("Language:"));
	gc->add_child(language_menu);

	language_menu->connect(SceneStringName(item_selected), callable_mp(this, &ExtensionClassCreateDialog::_language_changed));

	/* Class name */

	gc->add_child(memnew(Label(TTR("Class Name:"))));
	class_name_edit = memnew(LineEdit);
	class_name_edit->set_accessibility_name(TTRC("Class Name"));
	class_name_edit->connect(SceneStringName(text_changed), callable_mp(this, &ExtensionClassCreateDialog::_class_name_changed));
	class_name_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	register_text_enter(class_name_edit);
	gc->add_child(class_name_edit);

	connect(SceneStringName(visibility_changed), callable_mp(this, &ExtensionClassCreateDialog::_focus_on_class_name_edit));

	/* Inherits */

	base_type = "Object";

	gc->add_child(memnew(Label(TTR("Inherits:"))));
	HBoxContainer *inherits_hb = memnew(HBoxContainer);
	inherits_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	inherited_class_name_edit = memnew(LineEdit);
	inherited_class_name_edit->set_accessibility_name(TTRC("Parent Name"));
	inherited_class_name_edit->connect(SceneStringName(text_changed), callable_mp(this, &ExtensionClassCreateDialog::_inherited_class_name_changed));
	inherited_class_name_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	inherits_hb->add_child(inherited_class_name_edit);
	register_text_enter(inherited_class_name_edit);

	inherited_class_search_button = memnew(Button);
	inherited_class_search_button->set_accessibility_name(TTRC("Search Parent"));
	inherited_class_search_button->connect(SceneStringName(pressed), callable_mp(this, &ExtensionClassCreateDialog::_browse_inherited_class));
	inherits_hb->add_child(inherited_class_search_button);

	gc->add_child(inherits_hb);

	/* Templates */

	gc->add_child(memnew(Label(TTRC("Template:"))));
	HBoxContainer *template_hb = memnew(HBoxContainer);
	template_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	use_templates = memnew(CheckBox);
	use_templates->set_pressed(is_using_templates);
	use_templates->set_accessibility_name(TTRC("Use Template"));
	use_templates->connect(SceneStringName(pressed), callable_mp(this, &ExtensionClassCreateDialog::_use_template_pressed));
	template_hb->add_child(use_templates);

	template_menu = memnew(OptionButton);
	template_menu->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	template_menu->set_accessibility_name(TTRC("Template"));
	template_menu->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	template_menu->connect(SceneStringName(item_selected), callable_mp(this, &ExtensionClassCreateDialog::_template_changed));
	template_hb->add_child(template_menu);

	gc->add_child(template_hb);

	template_options_label = memnew(Label(TTR("Template options:")));
	template_options_label->set_v_size_flags(Control::SIZE_SHRINK_BEGIN);
	gc->add_child(template_options_label);
	template_options_inspector = memnew(EditorInspector);
	template_options_inspector->set_custom_minimum_size(Size2(0, 100) * EDSCALE);
	template_options_inspector->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	template_options_inspector->connect(SNAME("property_edited"), callable_mp(this, &ExtensionClassCreateDialog::_template_option_changed));
	template_options_object = memnew(TemplateOptions);
	template_options_inspector->edit(template_options_object);
	gc->add_child(template_options_inspector);

	/* Path */

	base_path = "res://";

	PathEdit path_edit = _create_path_edit(0);
	path_edits.push_back(path_edit);

	gc->add_child(path_edit.label);
	gc->add_child(path_edit.hb);

	path_edit_insert_index = gc->get_child_count();

	/* Dialog Setup */

	select_inherited_class_dialog = memnew(CreateDialog);
	select_inherited_class_dialog->connect("create", callable_mp(this, &ExtensionClassCreateDialog::_inherited_class_selected));
	add_child(select_inherited_class_dialog);

	select_path_dialog = memnew(EditorFileDialog);
	select_path_dialog->connect("file_selected", callable_mp(this, &ExtensionClassCreateDialog::_path_selected));
	add_child(select_path_dialog);

	alert = memnew(AcceptDialog);
	alert->get_label()->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	alert->get_label()->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	alert->get_label()->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	alert->get_label()->set_custom_minimum_size(Size2(325, 60) * EDSCALE);
	add_child(alert);

	set_ok_button_text(TTR("Create"));
	set_hide_on_ok(false);
	set_title(TTR("Create Extension Class"));
}

ExtensionClassCreateDialog::~ExtensionClassCreateDialog() {
	memdelete(validation_context);
	memdelete(template_options_object);
}

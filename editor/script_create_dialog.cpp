/*************************************************************************/
/*  script_create_dialog.cpp                                             */
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

#include "script_create_dialog.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "core/string/string_builder.h"
#include "editor/create_dialog.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor_file_system.h"

void ScriptCreateDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			for (int i = 0; i < ScriptServer::get_language_count(); i++) {
				Ref<Texture2D> language_icon = get_theme_icon(ScriptServer::get_language(i)->get_type(), SNAME("EditorIcons"));
				if (language_icon.is_valid()) {
					language_menu->set_item_icon(i, language_icon);
				}
			}

			String last_language = EditorSettings::get_singleton()->get_project_metadata("script_setup", "last_selected_language", "");
			if (!last_language.is_empty()) {
				for (int i = 0; i < language_menu->get_item_count(); i++) {
					if (language_menu->get_item_text(i) == last_language) {
						language_menu->select(i);
						current_language = i;
						break;
					}
				}
			} else {
				language_menu->select(default_language);
			}
			if (EditorSettings::get_singleton()->has_meta("script_setup/use_script_templates")) {
				is_using_templates = bool(EditorSettings::get_singleton()->get_meta("script_setup/use_script_templates"));
				use_templates->set_pressed(is_using_templates);
			}

			path_button->set_icon(get_theme_icon(SNAME("Folder"), SNAME("EditorIcons")));
			parent_browse_button->set_icon(get_theme_icon(SNAME("Folder"), SNAME("EditorIcons")));
			parent_search_button->set_icon(get_theme_icon(SNAME("ClassList"), SNAME("EditorIcons")));
			status_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("bg"), SNAME("Tree")));
		} break;
	}
}

void ScriptCreateDialog::_path_hbox_sorted() {
	if (is_visible()) {
		int filename_start_pos = initial_bp.rfind("/") + 1;
		int filename_end_pos = initial_bp.length();

		if (!is_built_in) {
			file_path->select(filename_start_pos, filename_end_pos);
		}

		// First set cursor to the end of line to scroll LineEdit view
		// to the right and then set the actual cursor position.
		file_path->set_caret_column(file_path->get_text().length());
		file_path->set_caret_column(filename_start_pos);

		file_path->grab_focus();
	}
}

bool ScriptCreateDialog::_can_be_built_in() {
	return (supports_built_in && built_in_enabled);
}

void ScriptCreateDialog::config(const String &p_base_name, const String &p_base_path, bool p_built_in_enabled, bool p_load_enabled) {
	class_name->set_text("");
	class_name->deselect();
	parent_name->set_text(p_base_name);
	parent_name->deselect();

	if (!p_base_path.is_empty()) {
		initial_bp = p_base_path.get_basename();
		file_path->set_text(initial_bp + "." + ScriptServer::get_language(language_menu->get_selected())->get_extension());
		current_language = language_menu->get_selected();
	} else {
		initial_bp = "";
		file_path->set_text("");
	}
	file_path->deselect();

	built_in_enabled = p_built_in_enabled;
	load_enabled = p_load_enabled;

	_language_changed(current_language);
	_class_name_changed("");
	_path_changed(file_path->get_text());
}

void ScriptCreateDialog::set_inheritance_base_type(const String &p_base) {
	base_type = p_base;
}

bool ScriptCreateDialog::_validate_parent(const String &p_string) {
	if (p_string.length() == 0) {
		return false;
	}

	if (can_inherit_from_file && p_string.is_quoted()) {
		String p = p_string.substr(1, p_string.length() - 2);
		if (_validate_path(p, true) == "") {
			return true;
		}
	}

	return ClassDB::class_exists(p_string) || ScriptServer::is_global_class(p_string);
}

bool ScriptCreateDialog::_validate_class(const String &p_string) {
	if (p_string.length() == 0) {
		return false;
	}

	for (int i = 0; i < p_string.length(); i++) {
		if (i == 0) {
			// Cannot start with a number.
			if (p_string[0] >= '0' && p_string[0] <= '9') {
				return false;
			}
		}

		bool valid_char = (p_string[i] >= '0' && p_string[i] <= '9') || (p_string[i] >= 'a' && p_string[i] <= 'z') || (p_string[i] >= 'A' && p_string[i] <= 'Z') || p_string[i] == '_' || p_string[i] == '.';

		if (!valid_char) {
			return false;
		}
	}

	return true;
}

String ScriptCreateDialog::_validate_path(const String &p_path, bool p_file_must_exist) {
	String p = p_path.strip_edges();

	if (p.is_empty()) {
		return TTR("Path is empty.");
	}
	if (p.get_file().get_basename().is_empty()) {
		return TTR("Filename is empty.");
	}

	if (!p.get_file().get_basename().is_valid_filename()) {
		return TTR("Filename is invalid.");
	}

	p = ProjectSettings::get_singleton()->localize_path(p);
	if (!p.begins_with("res://")) {
		return TTR("Path is not local.");
	}

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (d->change_dir(p.get_base_dir()) != OK) {
		memdelete(d);
		return TTR("Base path is invalid.");
	}
	memdelete(d);

	// Check if file exists.
	DirAccess *f = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (f->dir_exists(p)) {
		memdelete(f);
		return TTR("A directory with the same name exists.");
	} else if (p_file_must_exist && !f->file_exists(p)) {
		memdelete(f);
		return TTR("File does not exist.");
	}
	memdelete(f);

	// Check file extension.
	String extension = p.get_extension();
	List<String> extensions;

	// Get all possible extensions for script.
	for (int l = 0; l < language_menu->get_item_count(); l++) {
		ScriptServer::get_language(l)->get_recognized_extensions(&extensions);
	}

	bool found = false;
	bool match = false;
	int index = 0;
	for (const String &E : extensions) {
		if (E.nocasecmp_to(extension) == 0) {
			found = true;
			if (E == ScriptServer::get_language(language_menu->get_selected())->get_extension()) {
				match = true;
			}
			break;
		}
		index++;
	}

	if (!found) {
		return TTR("Invalid extension.");
	}
	if (!match) {
		return TTR("Extension doesn't match chosen language.");
	}

	// Let ScriptLanguage do custom validation.
	String path_error = ScriptServer::get_language(language_menu->get_selected())->validate_path(p);
	if (!path_error.is_empty()) {
		return path_error;
	}

	// All checks passed.
	return "";
}

String ScriptCreateDialog::_get_class_name() const {
	if (has_named_classes) {
		return class_name->get_text();
	} else {
		return ProjectSettings::get_singleton()->localize_path(file_path->get_text()).get_file().get_basename();
	}
}

void ScriptCreateDialog::_class_name_changed(const String &p_name) {
	is_class_name_valid = _validate_class(class_name->get_text());
	_update_dialog();
}

void ScriptCreateDialog::_parent_name_changed(const String &p_parent) {
	is_parent_name_valid = _validate_parent(parent_name->get_text());
	_update_dialog();
}

void ScriptCreateDialog::_template_changed(int p_template) {
	const ScriptLanguage::ScriptTemplate &sinfo = _get_current_template();
	// Update last used dictionaries
	if (is_using_templates && !parent_name->get_text().begins_with("\"res:")) {
		if (sinfo.origin == ScriptLanguage::TemplateLocation::TEMPLATE_PROJECT) {
			// Save the last used template for this node into the project dictionary.
			Dictionary dic_templates_project = EditorSettings::get_singleton()->get_project_metadata("script_setup", "templates_dictionary", Dictionary());
			dic_templates_project[parent_name->get_text()] = sinfo.get_hash();
			EditorSettings::get_singleton()->set_project_metadata("script_setup", "templates_dictionary", dic_templates_project);
		} else {
			// Save template into to editor dictionary (not a project template).
			Dictionary dic_templates;
			if (EditorSettings::get_singleton()->has_meta("script_setup/templates_dictionary")) {
				dic_templates = (Dictionary)EditorSettings::get_singleton()->get_meta("script_setup/templates_dictionary");
			}
			dic_templates[parent_name->get_text()] = sinfo.get_hash();
			EditorSettings::get_singleton()->set_meta("script_setup/templates_dictionary", dic_templates);
			// Remove template from project dictionary as we last used an editor level template.
			Dictionary dic_templates_project = EditorSettings::get_singleton()->get_project_metadata("script_setup", "templates_dictionary", Dictionary());
			if (dic_templates_project.has(parent_name->get_text())) {
				dic_templates_project.erase(parent_name->get_text());
				EditorSettings::get_singleton()->set_project_metadata("script_setup", "templates_dictionary", dic_templates_project);
			}
		}
	}
	// Update template label information.
	String template_info = String::utf8("•  ");
	template_info += TTR("Template:");
	template_info += " " + sinfo.name;
	if (!sinfo.description.is_empty()) {
		template_info += " - " + sinfo.description;
	}
	template_info_label->set_text(template_info);
	template_info_label->add_theme_color_override("font_color", get_theme_color(SNAME("success_color"), SNAME("Editor")));
}

void ScriptCreateDialog::ok_pressed() {
	if (is_new_script_created) {
		_create_new();
	} else {
		_load_exist();
	}

	EditorSettings::get_singleton()->save();
	is_new_script_created = true;
	_update_dialog();
}

void ScriptCreateDialog::_create_new() {
	String cname_param = _get_class_name();

	Ref<Script> scr;

	const ScriptLanguage::ScriptTemplate sinfo = _get_current_template();

	scr = ScriptServer::get_language(language_menu->get_selected())->make_template(sinfo.content, cname_param, parent_name->get_text());

	if (has_named_classes) {
		String cname = class_name->get_text();
		if (cname.length()) {
			scr->set_name(cname);
		}
	}

	if (is_built_in) {
		scr->set_name(internal_name->get_text());
	} else {
		String lpath = ProjectSettings::get_singleton()->localize_path(file_path->get_text());
		scr->set_path(lpath);
		Error err = ResourceSaver::save(lpath, scr, ResourceSaver::FLAG_CHANGE_PATH);
		if (err != OK) {
			alert->set_text(TTR("Error - Could not create script in filesystem."));
			alert->popup_centered();
			return;
		}
	}

	emit_signal(SNAME("script_created"), scr);
	hide();
}

void ScriptCreateDialog::_load_exist() {
	String path = file_path->get_text();
	RES p_script = ResourceLoader::load(path, "Script");
	if (p_script.is_null()) {
		alert->set_text(vformat(TTR("Error loading script from %s"), path));
		alert->popup_centered();
		return;
	}

	emit_signal(SNAME("script_created"), p_script);
	hide();
}

Vector<String> ScriptCreateDialog::get_hierarchy(String p_object) const {
	Vector<String> hierarchy;
	hierarchy.append(p_object);

	String parent_class = ClassDB::get_parent_class(p_object);
	while (parent_class.is_valid_identifier()) {
		hierarchy.append(parent_class);
		parent_class = ClassDB::get_parent_class(parent_class);
	}
	return hierarchy;
}

void ScriptCreateDialog::_language_changed(int l) {
	language = ScriptServer::get_language(l);

	has_named_classes = language->has_named_classes();
	can_inherit_from_file = language->can_inherit_from_file();
	supports_built_in = language->supports_builtin_mode();
	if (!supports_built_in) {
		is_built_in = false;
	}

	String selected_ext = "." + language->get_extension();
	String path = file_path->get_text();
	String extension = "";
	if (!path.is_empty()) {
		if (path.find(".") != -1) {
			extension = path.get_extension();
		}

		if (extension.length() == 0) {
			// Add extension if none.
			path += selected_ext;
			_path_changed(path);
		} else {
			// Change extension by selected language.
			List<String> extensions;
			// Get all possible extensions for script.
			for (int m = 0; m < language_menu->get_item_count(); m++) {
				ScriptServer::get_language(m)->get_recognized_extensions(&extensions);
			}

			for (const String &E : extensions) {
				if (E.nocasecmp_to(extension) == 0) {
					path = path.get_basename() + selected_ext;
					_path_changed(path);
					break;
				}
			}
		}
	} else {
		path = "class" + selected_ext;
		_path_changed(path);
	}
	file_path->set_text(path);

	EditorSettings::get_singleton()->set_project_metadata("script_setup", "last_selected_language", language_menu->get_item_text(language_menu->get_selected()));

	_parent_name_changed(parent_name->get_text());
	_update_dialog();
}

void ScriptCreateDialog::_built_in_pressed() {
	if (internal->is_pressed()) {
		is_built_in = true;
		is_new_script_created = true;
	} else {
		is_built_in = false;
		_path_changed(file_path->get_text());
	}
	_update_dialog();
}

void ScriptCreateDialog::_use_template_pressed() {
	is_using_templates = use_templates->is_pressed();
	EditorSettings::get_singleton()->set_meta("script_setup/use_script_templates", is_using_templates);
	_update_dialog();
}

void ScriptCreateDialog::_browse_path(bool browse_parent, bool p_save) {
	is_browsing_parent = browse_parent;

	if (p_save) {
		file_browse->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
		file_browse->set_title(TTR("Open Script / Choose Location"));
		file_browse->get_ok_button()->set_text(TTR("Open"));
	} else {
		file_browse->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
		file_browse->set_title(TTR("Open Script"));
	}

	file_browse->set_disable_overwrite_warning(true);
	file_browse->clear_filters();
	List<String> extensions;

	int lang = language_menu->get_selected();
	ScriptServer::get_language(lang)->get_recognized_extensions(&extensions);

	for (const String &E : extensions) {
		file_browse->add_filter("*." + E);
	}

	file_browse->set_current_path(file_path->get_text());
	file_browse->popup_file_dialog();
}

void ScriptCreateDialog::_file_selected(const String &p_file) {
	String path = ProjectSettings::get_singleton()->localize_path(p_file);
	if (is_browsing_parent) {
		parent_name->set_text("\"" + path + "\"");
		_parent_name_changed(parent_name->get_text());
	} else {
		file_path->set_text(path);
		_path_changed(path);

		String filename = path.get_file().get_basename();
		int select_start = path.rfind(filename);
		file_path->select(select_start, select_start + filename.length());
		file_path->set_caret_column(select_start + filename.length());
		file_path->grab_focus();
	}
}

void ScriptCreateDialog::_create() {
	parent_name->set_text(select_class->get_selected_type().split(" ")[0]);
	_parent_name_changed(parent_name->get_text());
}

void ScriptCreateDialog::_browse_class_in_tree() {
	select_class->set_base_type(base_type);
	select_class->popup_create(true);
	select_class->set_title(vformat(TTR("Inherit %s"), base_type));
	select_class->get_ok_button()->set_text(TTR("Inherit"));
}

void ScriptCreateDialog::_path_changed(const String &p_path) {
	if (is_built_in) {
		return;
	}

	is_path_valid = false;
	is_new_script_created = true;

	String path_error = _validate_path(p_path, false);
	if (!path_error.is_empty()) {
		_msg_path_valid(false, path_error);
		_update_dialog();
		return;
	}

	// Check if file exists.
	DirAccess *f = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	String p = ProjectSettings::get_singleton()->localize_path(p_path.strip_edges());
	if (f->file_exists(p)) {
		is_new_script_created = false;
		_msg_path_valid(true, TTR("File exists, it will be reused."));
	}
	memdelete(f);

	is_path_valid = true;
	_update_dialog();
}

void ScriptCreateDialog::_path_submitted(const String &p_path) {
	ok_pressed();
}

void ScriptCreateDialog::_msg_script_valid(bool valid, const String &p_msg) {
	error_label->set_text(String::utf8("•  ") + p_msg);
	if (valid) {
		error_label->add_theme_color_override("font_color", get_theme_color(SNAME("success_color"), SNAME("Editor")));
	} else {
		error_label->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
	}
}

void ScriptCreateDialog::_msg_path_valid(bool valid, const String &p_msg) {
	path_error_label->set_text(String::utf8("•  ") + p_msg);
	if (valid) {
		path_error_label->add_theme_color_override("font_color", get_theme_color(SNAME("success_color"), SNAME("Editor")));
	} else {
		path_error_label->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
	}
}

void ScriptCreateDialog::_update_template_menu() {
	bool is_language_using_templates = language->is_using_templates();
	template_menu->set_disabled(false);
	template_menu->clear();
	template_list.clear();

	if (is_language_using_templates) {
		// Get the latest templates used for each type of node from project settings then global settings.
		Dictionary last_local_templates = EditorSettings::get_singleton()->get_project_metadata("script_setup", "templates_dictionary", Dictionary());
		Dictionary last_global_templates;
		if (EditorSettings::get_singleton()->has_meta("script_setup/templates_dictionary")) {
			last_global_templates = (Dictionary)EditorSettings::get_singleton()->get_meta("script_setup/templates_dictionary");
		}
		String inherits_base_type = parent_name->get_text();

		// If it inherits from a script, select Object instead.
		if (inherits_base_type[0] == '"') {
			inherits_base_type = "Object";
		}

		// Get all ancestor node for selected base node.
		// There templates will also fit the base node.
		Vector<String> hierarchy = get_hierarchy(inherits_base_type);
		int last_used_template = -1;
		int preselected_template = -1;
		int previous_ancestor_level = -1;

		// Templates can be stored in tree different locations.
		Vector<ScriptLanguage::TemplateLocation> template_locations;
		template_locations.append(ScriptLanguage::TEMPLATE_PROJECT);
		template_locations.append(ScriptLanguage::TEMPLATE_EDITOR);
		template_locations.append(ScriptLanguage::TEMPLATE_BUILT_IN);

		for (const ScriptLanguage::TemplateLocation &template_location : template_locations) {
			String display_name = _get_script_origin_label(template_location);
			bool separator = false;
			int ancestor_level = 0;
			for (const String &current_node : hierarchy) {
				Vector<ScriptLanguage::ScriptTemplate> templates_found;
				if (template_location == ScriptLanguage::TEMPLATE_BUILT_IN) {
					templates_found = language->get_built_in_templates(current_node);
				} else {
					String template_directory;
					if (template_location == ScriptLanguage::TEMPLATE_PROJECT) {
						template_directory = EditorSettings::get_singleton()->get_project_script_templates_dir();
					} else {
						template_directory = EditorSettings::get_singleton()->get_script_templates_dir();
					}
					templates_found = _get_user_templates(language, current_node, template_directory, template_location);
				}
				if (!templates_found.is_empty()) {
					if (!separator) {
						template_menu->add_separator();
						template_menu->set_item_text(template_menu->get_item_count() - 1, display_name);
						separator = true;
					}
					for (ScriptLanguage::ScriptTemplate &t : templates_found) {
						template_menu->add_item(t.inherit + ": " + t.name);
						int id = template_menu->get_item_count() - 1;
						// Check if this template should be preselected if node isn't in the last used dictionary.
						if (ancestor_level < previous_ancestor_level || previous_ancestor_level == -1) {
							previous_ancestor_level = ancestor_level;
							preselected_template = id;
						}
						// Check for last used template for this node in project settings then in global settings.
						if (last_local_templates.has(parent_name->get_text()) && t.get_hash() == String(last_local_templates[parent_name->get_text()])) {
							last_used_template = id;
						} else if (last_used_template == -1 && last_global_templates.has(parent_name->get_text()) && t.get_hash() == String(last_global_templates[parent_name->get_text()])) {
							last_used_template = id;
						}
						t.id = id;
						template_list.push_back(t);
						String icon = has_theme_icon(t.inherit, SNAME("EditorIcons")) ? t.inherit : "Object";
						template_menu->set_item_icon(id, get_theme_icon(icon, SNAME("EditorIcons")));
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

void ScriptCreateDialog::_update_dialog() {
	// "Add Script Dialog" GUI logic and script checks.
	_update_template_menu();
	bool script_ok = true;

	// Is script path/name valid (order from top to bottom)?

	if (!is_built_in && !is_path_valid) {
		_msg_script_valid(false, TTR("Invalid path."));
		script_ok = false;
	}
	if (has_named_classes && (is_new_script_created && !is_class_name_valid)) {
		_msg_script_valid(false, TTR("Invalid class name."));
		script_ok = false;
	}
	if (!is_parent_name_valid && is_new_script_created) {
		_msg_script_valid(false, TTR("Invalid inherited parent name or path."));
		script_ok = false;
	}

	if (script_ok) {
		_msg_script_valid(true, TTR("Script path/name is valid."));
	}

	// Does script have named classes?

	if (has_named_classes) {
		if (is_new_script_created) {
			class_name->set_editable(true);
			class_name->set_placeholder(TTR("Allowed: a-z, A-Z, 0-9, _ and ."));
			class_name->set_placeholder_alpha(0.3);
		} else {
			class_name->set_editable(false);
		}
	} else {
		class_name->set_editable(false);
		class_name->set_placeholder(TTR("N/A"));
		class_name->set_placeholder_alpha(1);
		class_name->set_text("");
	}

	// Is script Built-in?

	if (is_built_in) {
		file_path->set_editable(false);
		path_button->set_disabled(true);
		re_check_path = true;
	} else {
		file_path->set_editable(true);
		path_button->set_disabled(false);
		if (re_check_path) {
			re_check_path = false;
			_path_changed(file_path->get_text());
		}
	}

	if (!_can_be_built_in()) {
		internal->set_pressed(false);
	}
	internal->set_disabled(!_can_be_built_in());

	// Is Script created or loaded from existing file?

	builtin_warning_label->set_visible(is_built_in);

	path_controls[0]->set_visible(!is_built_in);
	path_controls[1]->set_visible(!is_built_in);
	name_controls[0]->set_visible(is_built_in);
	name_controls[1]->set_visible(is_built_in);

	// Check if the script name is the same as the parent class.
	// This warning isn't relevant if the script is built-in.
	script_name_warning_label->set_visible(!is_built_in && _get_class_name() == parent_name->get_text());

	bool is_new_file = is_built_in || is_new_script_created;

	parent_name->set_editable(is_new_file);
	parent_search_button->set_disabled(!is_new_file);
	parent_browse_button->set_disabled(!is_new_file || !can_inherit_from_file);
	template_inactive_message = "";
	String button_text = is_new_file ? TTR("Create") : TTR("Load");
	get_ok_button()->set_text(button_text);

	if (is_new_file) {
		if (is_built_in) {
			_msg_path_valid(true, TTR("Built-in script (into scene file)."));
		}
		if (is_new_script_created && is_path_valid) {
			_msg_path_valid(true, TTR("Will create a new script file."));
		}
	} else {
		if (load_enabled) {
			template_inactive_message = TTR("Using existing script file.");
			if (is_path_valid) {
				_msg_path_valid(true, TTR("Will load an existing script file."));
			}
		} else {
			template_inactive_message = TTR("Using existing script file.");
			_msg_path_valid(false, TTR("Script file already exists."));
			script_ok = false;
		}
	}

	// Show templates list if needed.
	if (is_using_templates) {
		// Check if at least one suitable template has been found.
		if (template_menu->get_item_count() == 0 && template_inactive_message.is_empty()) {
			template_inactive_message = TTR("No suitable template.");
		}
	} else {
		template_inactive_message = TTR("Empty");
	}

	if (!template_inactive_message.is_empty()) {
		template_menu->set_disabled(true);
		template_menu->clear();
		template_menu->add_item(template_inactive_message);
	}
	template_info_label->set_visible(!template_menu->is_disabled());

	get_ok_button()->set_disabled(!script_ok);

	Callable entered_call = callable_mp(this, &ScriptCreateDialog::_path_submitted);
	if (script_ok) {
		if (!file_path->is_connected("text_submitted", entered_call)) {
			file_path->connect("text_submitted", entered_call);
		}
	} else if (file_path->is_connected("text_submitted", entered_call)) {
		file_path->disconnect("text_submitted", entered_call);
	}
}

ScriptLanguage::ScriptTemplate ScriptCreateDialog::_get_current_template() const {
	int selected_id = template_menu->get_selected_id();
	for (const ScriptLanguage::ScriptTemplate &t : template_list) {
		if (is_using_templates) {
			if (t.id == selected_id) {
				return t;
			}
		} else {
			// Using empty built-in template if templates are disabled.
			if (t.origin == ScriptLanguage::TemplateLocation::TEMPLATE_BUILT_IN && t.name == "Empty") {
				return t;
			}
		}
	}
	return ScriptLanguage::ScriptTemplate();
}

Vector<ScriptLanguage::ScriptTemplate> ScriptCreateDialog::_get_user_templates(const ScriptLanguage *language, const StringName &p_object, const String &p_dir, const ScriptLanguage::TemplateLocation &p_origin) const {
	Vector<ScriptLanguage::ScriptTemplate> user_templates;
	String extension = language->get_extension();

	String dir_path = p_dir.plus_file(p_object);

	DirAccess *d = DirAccess::open(dir_path);
	if (d) {
		d->list_dir_begin();
		String file = d->get_next();
		while (file != String()) {
			if (file.get_extension() == extension) {
				user_templates.append(_parse_template(language, dir_path, file, p_origin, p_object));
			}
			file = d->get_next();
		}
		d->list_dir_end();
		memdelete(d);
	}
	return user_templates;
}

ScriptLanguage::ScriptTemplate ScriptCreateDialog::_parse_template(const ScriptLanguage *language, const String &p_path, const String &p_filename, const ScriptLanguage::TemplateLocation &p_origin, const String &p_inherits) const {
	ScriptLanguage::ScriptTemplate script_template = ScriptLanguage::ScriptTemplate();
	script_template.origin = p_origin;
	script_template.inherit = p_inherits;
	String space_indent = "    ";
	// Get meta delimiter
	String meta_delimiter = String();
	List<String> comment_delimiters;
	language->get_comment_delimiters(&comment_delimiters);
	for (const String &script_delimiter : comment_delimiters) {
		if (script_delimiter.find(" ") == -1) {
			meta_delimiter = script_delimiter;
			break;
		}
	}
	String meta_prefix = meta_delimiter + " meta-";

	// Parse file for meta-information and script content
	Error err;
	FileAccess *file = FileAccess::open(p_path.plus_file(p_filename), FileAccess::READ, &err);
	if (!err) {
		while (!file->eof_reached()) {
			String line = file->get_line();
			if (line.begins_with(meta_prefix)) {
				// Store meta information
				line = line.substr(meta_prefix.length(), -1);
				if (line.begins_with("name")) {
					script_template.name = line.substr(5, -1).strip_edges();
				}
				if (line.begins_with("description")) {
					script_template.description = line.substr(12, -1).strip_edges();
				}
				if (line.begins_with("space-indent")) {
					String indent_value = line.substr(17, -1).strip_edges();
					if (indent_value.is_valid_int()) {
						space_indent = "";
						for (int i = 0; i < indent_value.to_int(); i++) {
							space_indent += " ";
						}
					} else {
						WARN_PRINT(vformat("Template meta-use_space_indent need to be a valid integer value. Found %s.", indent_value));
					}
				}
			} else {
				// Store script
				if (space_indent != "") {
					line = line.replace(space_indent, "_TS_");
				}
				script_template.content += line.replace("\t", "_TS_") + "\n";
			}
		}
		file->close();
		memdelete(file);
	}

	script_template.content = script_template.content.lstrip("\n");

	// Get name from file name if no name in meta information
	if (script_template.name == String()) {
		script_template.name = p_filename.get_basename().replace("_", " ").capitalize();
	}

	return script_template;
}

String ScriptCreateDialog::_get_script_origin_label(const ScriptLanguage::TemplateLocation &p_origin) const {
	switch (p_origin) {
		case ScriptLanguage::TEMPLATE_BUILT_IN:
			return TTR("Built-in");
		case ScriptLanguage::TEMPLATE_EDITOR:
			return TTR("Editor");
		case ScriptLanguage::TEMPLATE_PROJECT:
			return TTR("Project");
	}
	return "";
}

void ScriptCreateDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("config", "inherits", "path", "built_in_enabled", "load_enabled"), &ScriptCreateDialog::config, DEFVAL(true), DEFVAL(true));

	ADD_SIGNAL(MethodInfo("script_created", PropertyInfo(Variant::OBJECT, "script", PROPERTY_HINT_RESOURCE_TYPE, "Script")));
}

ScriptCreateDialog::ScriptCreateDialog() {
	/* Main Controls */

	GridContainer *gc = memnew(GridContainer);
	gc->set_columns(2);

	/* Information Messages Field */

	VBoxContainer *vb = memnew(VBoxContainer);

	error_label = memnew(Label);
	vb->add_child(error_label);

	path_error_label = memnew(Label);
	vb->add_child(path_error_label);

	builtin_warning_label = memnew(Label);
	builtin_warning_label->set_text(
			TTR("Note: Built-in scripts have some limitations and can't be edited using an external editor."));
	vb->add_child(builtin_warning_label);
	builtin_warning_label->set_autowrap_mode(Label::AUTOWRAP_WORD_SMART);
	builtin_warning_label->hide();

	script_name_warning_label = memnew(Label);
	script_name_warning_label->set_text(
			TTR("Warning: Having the script name be the same as a built-in type is usually not desired."));
	vb->add_child(script_name_warning_label);
	script_name_warning_label->add_theme_color_override("font_color", Color(1, 0.85, 0.4));
	script_name_warning_label->set_autowrap_mode(Label::AUTOWRAP_WORD_SMART);
	script_name_warning_label->hide();

	template_info_label = memnew(Label);
	vb->add_child(template_info_label);
	template_info_label->set_autowrap_mode(Label::AUTOWRAP_WORD_SMART);

	status_panel = memnew(PanelContainer);
	status_panel->set_h_size_flags(Control::SIZE_FILL);
	status_panel->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	status_panel->add_child(vb);

	/* Spacing */

	Control *spacing = memnew(Control);
	spacing->set_custom_minimum_size(Size2(0, 10 * EDSCALE));

	vb = memnew(VBoxContainer);
	vb->add_child(gc);
	vb->add_child(spacing);
	vb->add_child(status_panel);
	add_child(vb);

	/* Language */

	language_menu = memnew(OptionButton);
	language_menu->set_custom_minimum_size(Size2(350, 0) * EDSCALE);
	language_menu->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	gc->add_child(memnew(Label(TTR("Language:"))));
	gc->add_child(language_menu);

	default_language = -1;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		String lang = ScriptServer::get_language(i)->get_name();
		language_menu->add_item(lang);
		if (lang == "GDScript") {
			default_language = i;
		}
	}
	if (default_language >= 0) {
		language_menu->select(default_language);
	}
	current_language = default_language;

	language_menu->connect("item_selected", callable_mp(this, &ScriptCreateDialog::_language_changed));

	/* Inherits */

	base_type = "Object";

	HBoxContainer *hb = memnew(HBoxContainer);
	hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	parent_name = memnew(LineEdit);
	parent_name->connect("text_changed", callable_mp(this, &ScriptCreateDialog::_parent_name_changed));
	parent_name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb->add_child(parent_name);
	parent_search_button = memnew(Button);
	parent_search_button->connect("pressed", callable_mp(this, &ScriptCreateDialog::_browse_class_in_tree));
	hb->add_child(parent_search_button);
	parent_browse_button = memnew(Button);
	parent_browse_button->connect("pressed", callable_mp(this, &ScriptCreateDialog::_browse_path), varray(true, false));
	hb->add_child(parent_browse_button);
	gc->add_child(memnew(Label(TTR("Inherits:"))));
	gc->add_child(hb);
	is_browsing_parent = false;

	/* Class Name */

	class_name = memnew(LineEdit);
	class_name->connect("text_changed", callable_mp(this, &ScriptCreateDialog::_class_name_changed));
	class_name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	gc->add_child(memnew(Label(TTR("Class Name:"))));
	gc->add_child(class_name);

	/* Templates */

	is_using_templates = true;
	gc->add_child(memnew(Label(TTR("Template:"))));
	HBoxContainer *template_hb = memnew(HBoxContainer);
	template_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);

	use_templates = memnew(CheckBox);
	use_templates->set_pressed(is_using_templates);
	use_templates->connect("pressed", callable_mp(this, &ScriptCreateDialog::_use_template_pressed));
	template_hb->add_child(use_templates);

	template_inactive_message = "";

	template_menu = memnew(OptionButton);
	template_menu->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	template_menu->connect("item_selected", callable_mp(this, &ScriptCreateDialog::_template_changed));
	template_hb->add_child(template_menu);

	gc->add_child(template_hb);

	/* Built-in Script */

	internal = memnew(CheckBox);
	internal->set_text(TTR("On"));
	internal->connect("pressed", callable_mp(this, &ScriptCreateDialog::_built_in_pressed));
	gc->add_child(memnew(Label(TTR("Built-in Script:"))));
	gc->add_child(internal);

	/* Path */

	hb = memnew(HBoxContainer);
	hb->connect("sort_children", callable_mp(this, &ScriptCreateDialog::_path_hbox_sorted));
	file_path = memnew(LineEdit);
	file_path->connect("text_changed", callable_mp(this, &ScriptCreateDialog::_path_changed));
	file_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb->add_child(file_path);
	path_button = memnew(Button);
	path_button->connect("pressed", callable_mp(this, &ScriptCreateDialog::_browse_path), varray(false, true));
	hb->add_child(path_button);
	Label *label = memnew(Label(TTR("Path:")));
	gc->add_child(label);
	gc->add_child(hb);
	re_check_path = false;
	path_controls[0] = label;
	path_controls[1] = hb;

	/* Name */

	internal_name = memnew(LineEdit);
	internal_name->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	label = memnew(Label(TTR("Name:")));
	gc->add_child(label);
	gc->add_child(internal_name);
	name_controls[0] = label;
	name_controls[1] = internal_name;
	label->hide();
	internal_name->hide();

	/* Dialog Setup */

	select_class = memnew(CreateDialog);
	select_class->connect("create", callable_mp(this, &ScriptCreateDialog::_create));
	add_child(select_class);

	file_browse = memnew(EditorFileDialog);
	file_browse->connect("file_selected", callable_mp(this, &ScriptCreateDialog::_file_selected));
	file_browse->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	add_child(file_browse);
	get_ok_button()->set_text(TTR("Create"));
	alert = memnew(AcceptDialog);
	alert->get_label()->set_autowrap_mode(Label::AUTOWRAP_WORD_SMART);
	alert->get_label()->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	alert->get_label()->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	alert->get_label()->set_custom_minimum_size(Size2(325, 60) * EDSCALE);
	add_child(alert);

	set_hide_on_ok(false);
	set_title(TTR("Attach Node Script"));

	is_parent_name_valid = false;
	is_class_name_valid = false;
	is_path_valid = false;

	has_named_classes = false;
	supports_built_in = false;
	can_inherit_from_file = false;
	is_built_in = false;
	built_in_enabled = true;
	load_enabled = true;

	is_new_script_created = true;
}

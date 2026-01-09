/**************************************************************************/
/*  shader_create_dialog.cpp                                              */
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

#include "shader_create_dialog.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "editor/editor_node.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/gui/editor_validation_panel.h"
#include "editor/shader/editor_shader_language_plugin.h"
#include "editor/themes/editor_scale.h"
#include "scene/resources/shader_include.h"
#include "servers/rendering/shader_types.h"

void ShaderCreateDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			current_mode = EditorSettings::get_singleton()->get_project_metadata("shader_setup", "last_selected_mode", 0);
			mode_menu->select(current_mode);
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			path_button->set_button_icon(get_editor_theme_icon(SNAME("Folder")));
			// Note that some of the theme logic happens in `config()` when opening the dialog.
		} break;
	}
}

void ShaderCreateDialog::_refresh_type_icons() {
	for (int i = 0; i < type_menu->get_item_count(); i++) {
		const String item_name = type_menu->get_item_text(i);
		Ref<Texture2D> icon = get_editor_theme_icon(item_name);
		if (icon.is_valid()) {
			type_menu->set_item_icon(i, icon);
		} else {
			icon = get_editor_theme_icon("TextFile");
			if (icon.is_valid()) {
				type_menu->set_item_icon(i, icon);
			}
		}
	}
}

void ShaderCreateDialog::_update_language_info() {
	type_data.clear();

	for (int i = 0; i < EditorShaderLanguagePlugin::get_shader_language_variation_count(); i++) {
		ShaderTypeData shader_type_data;
		if (i == 0) {
			// HACK: The ShaderCreateDialog class currently only shows templates for text shaders. Generalize this later.
			shader_type_data.use_templates = true;
		}
		shader_type_data.default_extension = EditorShaderLanguagePlugin::get_file_extension_for_index(i);
		shader_type_data.extensions.push_back(shader_type_data.default_extension);
		if (shader_type_data.default_extension != "tres") {
			shader_type_data.extensions.push_back("tres");
		}
		shader_type_data.extensions.push_back("res");
		type_data.push_back(shader_type_data);
	}
}

void ShaderCreateDialog::_path_hbox_sorted() {
	if (is_visible()) {
		int filename_start_pos = initial_base_path.rfind_char('/') + 1;
		int filename_end_pos = initial_base_path.length();

		if (!is_built_in) {
			file_path->select(filename_start_pos, filename_end_pos);
		}

		file_path->set_caret_column(file_path->get_text().length());
		file_path->set_caret_column(filename_start_pos);

		file_path->grab_focus();
	}
}

void ShaderCreateDialog::_mode_changed(int p_mode) {
	current_mode = p_mode;
	EditorSettings::get_singleton()->set_project_metadata("shader_setup", "last_selected_mode", p_mode);
}

void ShaderCreateDialog::_template_changed(int p_template) {
	current_template = p_template;
	EditorSettings::get_singleton()->set_project_metadata("shader_setup", "last_selected_template", p_template);
}

void ShaderCreateDialog::ok_pressed() {
	if (is_new_shader_created) {
		_create_new();
		if (built_in_enabled) {
			// Only save state of built-in checkbox if it's enabled.
			EditorSettings::get_singleton()->set_project_metadata("shader_setup", "create_built_in_shader", internal->is_pressed());
		}
	} else {
		_load_exist();
	}

	is_new_shader_created = true;
	validation_panel->update();
}

void ShaderCreateDialog::_create_new() {
	Ref<Resource> shader;
	Ref<Resource> shader_inc;

	const int type_index = type_menu->get_selected();
	Ref<EditorShaderLanguagePlugin> shader_lang = EditorShaderLanguagePlugin::get_shader_language_for_index(type_index);
	// A bit of an unfortunate hack because Shader and ShaderInclude do not share a common base class.
	// We need duplicate code paths for includes vs non-includes, so just check for the string "Include".
	if (type_menu->get_item_text(type_index).contains("Include")) {
		shader_inc = shader_lang->create_new_shader_include();
	} else {
		shader = shader_lang->create_new_shader(0, Shader::Mode(current_mode), current_template);
	}

	if (shader.is_null()) {
		String lpath = ProjectSettings::get_singleton()->localize_path(file_path->get_text());
		shader_inc->set_path(lpath);

		Error error = ResourceSaver::save(shader_inc, lpath, ResourceSaver::FLAG_CHANGE_PATH);
		if (error != OK) {
			alert->set_text(TTR("Error - Could not create shader include in filesystem."));
			alert->popup_centered();
			return;
		}

		emit_signal(SNAME("shader_include_created"), shader_inc);
	} else {
		if (is_built_in) {
			Node *edited_scene = get_tree()->get_edited_scene_root();
			if (likely(edited_scene)) {
				shader->set_path(edited_scene->get_scene_file_path() + "::" + shader->generate_scene_unique_id());
			}
		} else {
			String lpath = ProjectSettings::get_singleton()->localize_path(file_path->get_text());
			shader->set_path(lpath);

			Error error = ResourceSaver::save(shader, lpath, ResourceSaver::FLAG_CHANGE_PATH);
			if (error != OK) {
				alert->set_text(TTR("Error - Could not create shader in filesystem."));
				alert->popup_centered();
				return;
			}
		}

		emit_signal(SNAME("shader_created"), shader);
	}

	file_path->set_text(file_path->get_text().get_base_dir());
	hide();
}

void ShaderCreateDialog::_load_exist() {
	String path = file_path->get_text();
	Ref<Resource> p_shader = ResourceLoader::load(path, "Shader");
	if (p_shader.is_null()) {
		alert->set_text(vformat(TTR("Error loading shader from %s"), path));
		alert->popup_centered();
		return;
	}

	emit_signal(SNAME("shader_created"), p_shader);
	hide();
}

void ShaderCreateDialog::_type_changed(int p_language) {
	current_type = p_language;
	ShaderTypeData shader_type_data = type_data.get(p_language);

	String selected_ext = "." + shader_type_data.default_extension;
	String path = file_path->get_text();
	String extension = "";

	if (!path.is_empty()) {
		if (path.contains_char('.')) {
			extension = path.get_extension();
		}
		if (extension.length() == 0) {
			path += selected_ext;
		} else {
			path = path.get_basename() + selected_ext;
		}
	} else {
		path = "shader" + selected_ext;
	}
	_path_changed(path);
	file_path->set_text(path);

	mode_menu->set_disabled(false);
	for (int i = 0; i < type_menu->get_item_count(); i++) {
		const String item_name = type_menu->get_item_text(i);
		if (item_name.contains("Include")) {
			type_menu->set_item_disabled(i, load_enabled);
			if (i == p_language) {
				mode_menu->set_disabled(true);
			}
		} else {
			type_menu->set_item_disabled(i, false);
		}
	}
	template_menu->set_disabled(!shader_type_data.use_templates);
	template_menu->clear();

	if (shader_type_data.use_templates) {
		int last_template = EditorSettings::get_singleton()->get_project_metadata("shader_setup", "last_selected_template", 0);

		template_menu->add_item(TTRC("Default"));
		template_menu->add_item(TTRC("Empty"));

		template_menu->select(last_template);
		current_template = last_template;
	} else {
		template_menu->add_item(TTRC("N/A"));
	}

	EditorSettings::get_singleton()->set_project_metadata("shader_setup", "last_selected_language", type_menu->get_item_text(type_menu->get_selected()));
	validation_panel->update();
}

void ShaderCreateDialog::_built_in_toggled(bool p_enabled) {
	is_built_in = p_enabled;
	if (p_enabled) {
		is_new_shader_created = true;
	} else {
		_path_changed(file_path->get_text());
	}
	validation_panel->update();
}

void ShaderCreateDialog::_browse_path() {
	file_browse->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	file_browse->set_title(TTR("Open Shader / Choose Location"));
	file_browse->set_ok_button_text(TTR("Open"));

	file_browse->set_customization_flag_enabled(FileDialog::CUSTOMIZATION_OVERWRITE_WARNING, false);
	file_browse->clear_filters();

	List<String> extensions = type_data.get(type_menu->get_selected()).extensions;

	for (const String &E : extensions) {
		file_browse->add_filter("*." + E);
	}

	file_browse->set_current_path(file_path->get_text());
	file_browse->popup_file_dialog();
}

void ShaderCreateDialog::_file_selected(const String &p_file) {
	String p = ProjectSettings::get_singleton()->localize_path(p_file);
	file_path->set_text(p);
	_path_changed(p);

	String filename = p.get_file().get_basename();
	int select_start = p.rfind(filename);
	file_path->select(select_start, select_start + filename.length());
	file_path->set_caret_column(select_start + filename.length());
	file_path->grab_focus();
}

void ShaderCreateDialog::_path_changed(const String &p_path) {
	if (is_built_in) {
		return;
	}

	is_path_valid = false;
	is_new_shader_created = true;

	path_error = _validate_path(p_path);
	if (!path_error.is_empty()) {
		validation_panel->update();
		return;
	}

	Ref<DirAccess> f = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	String p = ProjectSettings::get_singleton()->localize_path(p_path.strip_edges());
	if (f->file_exists(p)) {
		is_new_shader_created = false;
	}

	is_path_valid = true;
	validation_panel->update();
}

void ShaderCreateDialog::_path_submitted(const String &p_path) {
	if (!get_ok_button()->is_disabled()) {
		ok_pressed();
	}
}

void ShaderCreateDialog::config(const String &p_base_path, bool p_built_in_enabled, bool p_load_enabled, const String &p_preferred_type, int p_preferred_mode) {
	_update_language_info();
	type_menu->clear();
	const Vector<Ref<EditorShaderLanguagePlugin>> shader_langs = EditorShaderLanguagePlugin::get_shader_languages_read_only();
	ERR_FAIL_COND_MSG(shader_langs.is_empty(), "ShaderCreateDialog: Unable to load any shader languages!");
	for (Ref<EditorShaderLanguagePlugin> shader_lang : shader_langs) {
		const PackedStringArray variations = shader_lang->get_language_variations();
		for (const String &variation : variations) {
			type_menu->add_item(variation);
		}
	}
	_refresh_type_icons();

	int preferred_type = -1;
	// Select preferred type if specified.
	for (int i = 0; i < type_menu->get_item_count(); i++) {
		if (type_menu->get_item_text(i) == p_preferred_type) {
			preferred_type = i;
			break;
		}
	}
	if (preferred_type < 0 || preferred_type >= type_menu->get_item_count()) {
		preferred_type = 0;
		// Select the last selected language if possible, otherwise default to the first language.
		String last_lang = EditorSettings::get_singleton()->get_project_metadata("shader_setup", "last_selected_language", "");
		if (!last_lang.is_empty()) {
			for (int i = 0; i < type_menu->get_item_count(); i++) {
				if (type_menu->get_item_text(i) == last_lang) {
					preferred_type = i;
					break;
				}
			}
		}
	}
	type_menu->select(preferred_type);
	current_type = 0;

	if (!p_base_path.is_empty()) {
		initial_base_path = p_base_path.get_basename();
		file_path->set_text(initial_base_path + "." + type_data.get(type_menu->get_selected()).default_extension);
		current_type = type_menu->get_selected();
	} else {
		initial_base_path = "";
		file_path->set_text("");
	}
	file_path->deselect();

	built_in_enabled = p_built_in_enabled;
	load_enabled = p_load_enabled;

	if (built_in_enabled) {
		internal->set_pressed(EditorSettings::get_singleton()->get_project_metadata("shader_setup", "create_built_in_shader", false));
	}

	if (p_preferred_mode > -1) {
		mode_menu->select(p_preferred_mode);
		_mode_changed(p_preferred_mode);
	}

	_type_changed(current_type);
	_path_changed(file_path->get_text());
}

String ShaderCreateDialog::_validate_path(const String &p_path) {
	ERR_FAIL_COND_V(current_type >= type_data.size(), TTR("Invalid shader type selected."));
	String stripped_file_path = p_path.strip_edges();

	if (stripped_file_path.is_empty()) {
		return TTR("Path is empty.");
	}
	if (stripped_file_path.get_file().get_basename().is_empty()) {
		return TTR("Filename is empty.");
	}

	stripped_file_path = ProjectSettings::get_singleton()->localize_path(stripped_file_path);
	if (!stripped_file_path.begins_with("res://")) {
		return TTR("Path is not local.");
	}

	Ref<DirAccess> d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (d->change_dir(stripped_file_path.get_base_dir()) != OK) {
		return TTR("Invalid base path.");
	}

	Ref<DirAccess> f = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (f->dir_exists(stripped_file_path)) {
		return TTR("A directory with the same name exists.");
	}

	const ShaderCreateDialog::ShaderTypeData &current_type_data = type_data.get(current_type);
	const String file_extension = stripped_file_path.get_extension();

	for (const String &type_ext : current_type_data.extensions) {
		if (type_ext.nocasecmp_to(file_extension) == 0) {
			return "";
		}
	}

	return TTR("Invalid extension for selected shader type.");
}

void ShaderCreateDialog::_update_dialog() {
	if (!is_built_in && !is_path_valid) {
		validation_panel->set_message(MSG_ID_SHADER, TTR("Invalid path."), EditorValidationPanel::MSG_ERROR);
	}
	if (!is_built_in && !path_error.is_empty()) {
		validation_panel->set_message(MSG_ID_PATH, path_error, EditorValidationPanel::MSG_ERROR);
	} else if (validation_panel->is_valid() && !is_new_shader_created) {
		validation_panel->set_message(MSG_ID_SHADER, TTR("File exists, it will be reused."), EditorValidationPanel::MSG_OK);
	}
	if (!built_in_enabled) {
		internal->set_pressed(false);
	}

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

	internal->set_disabled(!built_in_enabled);

	if (is_built_in) {
		validation_panel->set_message(MSG_ID_BUILT_IN, TTR("Note: Built-in shaders can't be edited using an external editor."), EditorValidationPanel::MSG_INFO, false);
	}

	if (is_built_in) {
		set_ok_button_text(TTR("Create"));
		validation_panel->set_message(MSG_ID_PATH, TTR("Built-in shader (into scene file)."), EditorValidationPanel::MSG_OK);
	} else if (is_new_shader_created) {
		set_ok_button_text(TTR("Create"));
	} else if (load_enabled) {
		set_ok_button_text(TTR("Load"));
		if (is_path_valid) {
			validation_panel->set_message(MSG_ID_PATH, TTR("Will load an existing shader file."), EditorValidationPanel::MSG_OK);
		}
	} else {
		set_ok_button_text(TTR("Create"));
		validation_panel->set_message(MSG_ID_PATH, TTR("Shader file already exists."), EditorValidationPanel::MSG_ERROR);
	}
}

void ShaderCreateDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("config", "path", "built_in_enabled", "load_enabled"), &ShaderCreateDialog::config, DEFVAL(true), DEFVAL(true));

	ADD_SIGNAL(MethodInfo("shader_created", PropertyInfo(Variant::OBJECT, "shader", PROPERTY_HINT_RESOURCE_TYPE, "Shader")));
	ADD_SIGNAL(MethodInfo("shader_include_created", PropertyInfo(Variant::OBJECT, "shader_include", PROPERTY_HINT_RESOURCE_TYPE, "ShaderInclude")));
}

ShaderCreateDialog::ShaderCreateDialog() {
	_update_language_info();

	// Main Controls.

	gc = memnew(GridContainer);
	gc->set_columns(2);

	// Error Fields.

	validation_panel = memnew(EditorValidationPanel);
	validation_panel->add_line(MSG_ID_SHADER, TTR("Shader path/name is valid."));
	validation_panel->add_line(MSG_ID_PATH, TTR("Will create a new shader file."));
	validation_panel->add_line(MSG_ID_BUILT_IN);
	validation_panel->set_update_callback(callable_mp(this, &ShaderCreateDialog::_update_dialog));
	validation_panel->set_accept_button(get_ok_button());

	// Spacing.

	Control *spacing = memnew(Control);
	spacing->set_custom_minimum_size(Size2(0, 10 * EDSCALE));

	VBoxContainer *vb = memnew(VBoxContainer);
	vb->add_child(gc);
	vb->add_child(spacing);
	vb->add_child(validation_panel);
	add_child(vb);

	// Type.

	type_menu = memnew(OptionButton);
	type_menu->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	type_menu->set_accessibility_name(TTRC("Type:"));
	type_menu->set_custom_minimum_size(Size2(250, 0) * EDSCALE);
	type_menu->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	gc->add_child(memnew(Label(TTR("Type:"))));
	gc->add_child(type_menu);

	type_menu->connect(SceneStringName(item_selected), callable_mp(this, &ShaderCreateDialog::_type_changed));

	// Modes.

	mode_menu = memnew(OptionButton);
	mode_menu->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	mode_menu->set_accessibility_name(TTRC("Mode:"));
	for (const String &type_name : ShaderTypes::get_singleton()->get_types_list()) {
		mode_menu->add_item(type_name.capitalize());
	}
	gc->add_child(memnew(Label(TTR("Mode:"))));
	gc->add_child(mode_menu);
	mode_menu->connect(SceneStringName(item_selected), callable_mp(this, &ShaderCreateDialog::_mode_changed));

	// Templates.

	template_menu = memnew(OptionButton);
	template_menu->set_accessibility_name(TTRC("Template:"));
	gc->add_child(memnew(Label(TTR("Template:"))));
	gc->add_child(template_menu);
	template_menu->connect(SceneStringName(item_selected), callable_mp(this, &ShaderCreateDialog::_template_changed));

	// Built-in Shader.

	internal = memnew(CheckBox);
	internal->set_text(TTR("On"));
	internal->set_accessibility_name(TTRC("Built-in Shader:"));
	internal->connect(SceneStringName(toggled), callable_mp(this, &ShaderCreateDialog::_built_in_toggled));
	gc->add_child(memnew(Label(TTR("Built-in Shader:"))));
	gc->add_child(internal);

	// Path.

	HBoxContainer *hb = memnew(HBoxContainer);
	hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb->connect(SceneStringName(sort_children), callable_mp(this, &ShaderCreateDialog::_path_hbox_sorted));
	file_path = memnew(LineEdit);
	file_path->connect(SceneStringName(text_changed), callable_mp(this, &ShaderCreateDialog::_path_changed));
	file_path->set_accessibility_name(TTRC("Path:"));
	file_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb->add_child(file_path);
	register_text_enter(file_path);
	path_button = memnew(Button);
	path_button->set_accessibility_name(TTRC("Select"));
	path_button->connect(SceneStringName(pressed), callable_mp(this, &ShaderCreateDialog::_browse_path));
	hb->add_child(path_button);
	gc->add_child(memnew(Label(TTR("Path:"))));
	gc->add_child(hb);

	// Dialog Setup.

	file_browse = memnew(EditorFileDialog);
	file_browse->connect("file_selected", callable_mp(this, &ShaderCreateDialog::_file_selected));
	file_browse->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	add_child(file_browse);

	alert = memnew(AcceptDialog);
	alert->get_label()->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	alert->get_label()->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	alert->get_label()->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	alert->get_label()->set_custom_minimum_size(Size2(325, 60) * EDSCALE);
	add_child(alert);

	set_ok_button_text(TTR("Create"));
	set_hide_on_ok(false);

	set_title(TTR("Create Shader"));
}

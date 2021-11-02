/*************************************************************************/
/*  shader_create_dialog.cpp                                             */
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

#include "shader_create_dialog.h"
#include "editor/editor_scale.h"
#include "scene/resources/visual_shader.h"
#include "servers/rendering/shader_types.h"

void ShaderCreateDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_theme();

			String last_lang = EditorSettings::get_singleton()->get_project_metadata("shader_setup", "last_selected_language", "");
			if (!last_lang.is_empty()) {
				for (int i = 0; i < language_menu->get_item_count(); i++) {
					if (language_menu->get_item_text(i) == last_lang) {
						language_menu->select(i);
						current_language = i;
						break;
					}
				}
			} else {
				language_menu->select(default_language);
			}

			current_mode = EditorSettings::get_singleton()->get_project_metadata("shader_setup", "last_selected_mode", 0);
			mode_menu->select(current_mode);
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_update_theme();
		} break;
	}
}

void ShaderCreateDialog::_update_theme() {
	Ref<Texture2D> shader_icon = gc->get_theme_icon(SNAME("Shader"), SNAME("EditorIcons"));
	if (shader_icon.is_valid()) {
		language_menu->set_item_icon(0, shader_icon);
	}

	Ref<Texture2D> visual_shader_icon = gc->get_theme_icon(SNAME("VisualShader"), SNAME("EditorIcons"));
	if (visual_shader_icon.is_valid()) {
		language_menu->set_item_icon(1, visual_shader_icon);
	}

	path_button->set_icon(get_theme_icon(SNAME("Folder"), SNAME("EditorIcons")));
	status_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("bg"), SNAME("Tree")));
}

void ShaderCreateDialog::_update_language_info() {
	language_data.clear();

	for (int i = 0; i < SHADER_TYPE_MAX; i++) {
		ShaderTypeData data;
		if (i == int(SHADER_TYPE_TEXT)) {
			data.use_templates = true;
			data.extensions.push_back("gdshader");
			data.default_extension = "gdshader";
		} else {
			data.default_extension = "tres";
		}
		data.extensions.push_back("res");
		data.extensions.push_back("tres");
		language_data.push_back(data);
	}
}

void ShaderCreateDialog::_path_hbox_sorted() {
	if (is_visible()) {
		int filename_start_pos = initial_base_path.rfind("/") + 1;
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
	} else {
		_load_exist();
	}

	is_new_shader_created = true;
	_update_dialog();
}

void ShaderCreateDialog::_create_new() {
	RES shader;

	if (language_menu->get_selected() == int(SHADER_TYPE_TEXT)) {
		Ref<Shader> text_shader;
		text_shader.instantiate();
		shader = text_shader;

		StringBuilder code;
		code += vformat("shader_type %s;\n", mode_menu->get_text().replace(" ", "").camelcase_to_underscore());

		if (current_template == 0) { // Default template.
			code += "\n";
			switch (current_mode) {
				case Shader::MODE_SPATIAL:
					code += "void fragment() {\n";
					code += "\t// Place fragment code here.\n";
					code += "}\n";
					break;
				case Shader::MODE_CANVAS_ITEM:
					code += "void fragment() {\n";
					code += "\t// Place fragment code here.\n";
					code += "}\n";
					break;
				case Shader::MODE_PARTICLES:
					code += "void start() {\n";
					code += "\t// Place start code here.\n";
					code += "}\n";
					code += "\n";
					code += "void process() {\n";
					code += "\t// Place process code here.\n";
					code += "}\n";
					break;
				case Shader::MODE_SKY:
					code += "void sky() {\n";
					code += "\t// Place sky code here.\n";
					code += "}\n";
					break;
				case Shader::MODE_FOG:
					code += "void fog() {\n";
					code += "\t// Place fog code here.\n";
					code += "}\n";
					break;
			}
		}
		text_shader->set_code(code.as_string());
	} else {
		Ref<VisualShader> visual_shader;
		visual_shader.instantiate();
		shader = visual_shader;
		visual_shader->set_engine_version(Engine::get_singleton()->get_version_info());
		visual_shader->set_mode(Shader::Mode(current_mode));
	}

	if (!is_built_in) {
		String lpath = ProjectSettings::get_singleton()->localize_path(file_path->get_text());
		shader->set_path(lpath);
		Error err = ResourceSaver::save(lpath, shader, ResourceSaver::FLAG_CHANGE_PATH);
		if (err != OK) {
			alert->set_text(TTR("Error - Could not create shader in filesystem."));
			alert->popup_centered();
			return;
		}
	}

	emit_signal(SNAME("shader_created"), shader);
	hide();
}

void ShaderCreateDialog::_load_exist() {
	String path = file_path->get_text();
	RES p_shader = ResourceLoader::load(path, "Shader");
	if (p_shader.is_null()) {
		alert->set_text(vformat(TTR("Error loading shader from %s"), path));
		alert->popup_centered();
		return;
	}

	emit_signal(SNAME("shader_created"), p_shader);
	hide();
}

void ShaderCreateDialog::_language_changed(int p_language) {
	current_language = p_language;
	ShaderTypeData data = language_data[p_language];

	String selected_ext = "." + data.default_extension;
	String path = file_path->get_text();
	String extension = "";

	if (path != "") {
		if (path.find(".") != -1) {
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

	template_menu->set_disabled(!data.use_templates);
	template_menu->clear();

	if (data.use_templates) {
		int last_template = EditorSettings::get_singleton()->get_project_metadata("shader_setup", "last_selected_template", 0);

		template_menu->add_item(TTR("Default"));
		template_menu->add_item(TTR("Empty"));

		template_menu->select(last_template);
		current_template = last_template;
	} else {
		template_menu->add_item(TTR("N/A"));
	}

	EditorSettings::get_singleton()->set_project_metadata("shader_setup", "last_selected_language", language_menu->get_item_text(language_menu->get_selected()));
	_update_dialog();
}

void ShaderCreateDialog::_built_in_toggled(bool p_enabled) {
	is_built_in = p_enabled;
	if (p_enabled) {
		is_new_shader_created = true;
	} else {
		_path_changed(file_path->get_text());
	}
	_update_dialog();
}

void ShaderCreateDialog::_browse_path() {
	file_browse->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	file_browse->set_title(TTR("Open Shader / Choose Location"));
	file_browse->get_ok_button()->set_text(TTR("Open"));

	file_browse->set_disable_overwrite_warning(true);
	file_browse->clear_filters();

	List<String> extensions = language_data[language_menu->get_selected()].extensions;

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

	String path_error = _validate_path(p_path);
	if (path_error != "") {
		_msg_path_valid(false, path_error);
		_update_dialog();
		return;
	}

	DirAccessRef f = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	String p = ProjectSettings::get_singleton()->localize_path(p_path.strip_edges());
	if (f->file_exists(p)) {
		is_new_shader_created = false;
		_msg_path_valid(true, TTR("File exists, it will be reused."));
	}

	is_path_valid = true;
	_update_dialog();
}

void ShaderCreateDialog::_path_submitted(const String &p_path) {
	ok_pressed();
}

void ShaderCreateDialog::config(const String &p_base_path, bool p_built_in_enabled, bool p_load_enabled) {
	if (p_base_path != "") {
		initial_base_path = p_base_path.get_basename();
		file_path->set_text(initial_base_path + "." + language_data[language_menu->get_selected()].default_extension);
		current_language = language_menu->get_selected();
	} else {
		initial_base_path = "";
		file_path->set_text("");
	}
	file_path->deselect();

	built_in_enabled = p_built_in_enabled;
	load_enabled = p_load_enabled;

	_language_changed(current_language);
	_path_changed(file_path->get_text());
}

String ShaderCreateDialog::_validate_path(const String &p_path) {
	String p = p_path.strip_edges();

	if (p == "") {
		return TTR("Path is empty.");
	}
	if (p.get_file().get_basename() == "") {
		return TTR("Filename is empty.");
	}

	p = ProjectSettings::get_singleton()->localize_path(p);
	if (!p.begins_with("res://")) {
		return TTR("Path is not local.");
	}

	DirAccessRef d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (d->change_dir(p.get_base_dir()) != OK) {
		return TTR("Invalid base path.");
	}

	DirAccessRef f = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (f->dir_exists(p)) {
		return TTR("A directory with the same name exists.");
	}

	String extension = p.get_extension();
	Set<String> extensions;

	for (int i = 0; i < SHADER_TYPE_MAX; i++) {
		for (const String &ext : language_data[i].extensions) {
			if (!extensions.has(ext)) {
				extensions.insert(ext);
			}
		}
	}

	ShaderTypeData data = language_data[language_menu->get_selected()];

	bool found = false;
	bool match = false;

	for (const String &ext : extensions) {
		if (ext.nocasecmp_to(extension) == 0) {
			found = true;
			for (const String &lang_ext : language_data[current_language].extensions) {
				if (lang_ext.nocasecmp_to(extension) == 0) {
					match = true;
					break;
				}
			}
			break;
		}
	}

	if (!found) {
		return TTR("Invalid extension.");
	}
	if (!match) {
		return TTR("Wrong extension chosen.");
	}

	return "";
}

void ShaderCreateDialog::_msg_script_valid(bool valid, const String &p_msg) {
	error_label->set_text("- " + p_msg);
	if (valid) {
		error_label->add_theme_color_override("font_color", gc->get_theme_color(SNAME("success_color"), SNAME("Editor")));
	} else {
		error_label->add_theme_color_override("font_color", gc->get_theme_color(SNAME("error_color"), SNAME("Editor")));
	}
}

void ShaderCreateDialog::_msg_path_valid(bool valid, const String &p_msg) {
	path_error_label->set_text("- " + p_msg);
	if (valid) {
		path_error_label->add_theme_color_override("font_color", gc->get_theme_color(SNAME("success_color"), SNAME("Editor")));
	} else {
		path_error_label->add_theme_color_override("font_color", gc->get_theme_color(SNAME("error_color"), SNAME("Editor")));
	}
}

void ShaderCreateDialog::_update_dialog() {
	bool shader_ok = true;

	if (!is_built_in && !is_path_valid) {
		_msg_script_valid(false, TTR("Invalid path."));
		shader_ok = false;
	}
	if (shader_ok) {
		_msg_script_valid(true, TTR("Shader path/name is valid."));
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

	builtin_warning_label->set_visible(is_built_in);

	if (is_built_in) {
		get_ok_button()->set_text(TTR("Create"));
		_msg_path_valid(true, TTR("Built-in shader (into scene file)."));
	} else if (is_new_shader_created) {
		get_ok_button()->set_text(TTR("Create"));
		if (is_path_valid) {
			_msg_path_valid(true, TTR("Will create a new shader file."));
		}
	} else if (load_enabled) {
		get_ok_button()->set_text(TTR("Load"));
		if (is_path_valid) {
			_msg_path_valid(true, TTR("Will load an existing shader file."));
		}
	} else {
		get_ok_button()->set_text(TTR("Create"));
		_msg_path_valid(false, TTR("Shader file already exists."));

		shader_ok = false;
	}

	get_ok_button()->set_disabled(!shader_ok);

	Callable entered_call = callable_mp(this, &ShaderCreateDialog::_path_submitted);
	if (shader_ok) {
		if (!file_path->is_connected("text_submitted", entered_call)) {
			file_path->connect("text_submitted", entered_call);
		}
	} else if (file_path->is_connected("text_submitted", entered_call)) {
		file_path->disconnect("text_submitted", entered_call);
	}
}

void ShaderCreateDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("config", "path", "built_in_enabled", "load_enabled"), &ShaderCreateDialog::config, DEFVAL(true), DEFVAL(true));

	ADD_SIGNAL(MethodInfo("shader_created", PropertyInfo(Variant::OBJECT, "shader", PROPERTY_HINT_RESOURCE_TYPE, "Shader")));
}

ShaderCreateDialog::ShaderCreateDialog() {
	_update_language_info();

	// Main Controls.

	gc = memnew(GridContainer);
	gc->set_columns(2);

	// Error Fields.

	VBoxContainer *vb = memnew(VBoxContainer);

	error_label = memnew(Label);
	vb->add_child(error_label);

	path_error_label = memnew(Label);
	vb->add_child(path_error_label);

	builtin_warning_label = memnew(Label);
	builtin_warning_label->set_text(
			TTR("Note: Built-in shaders can't be edited using an external editor."));
	vb->add_child(builtin_warning_label);
	builtin_warning_label->set_autowrap_mode(Label::AUTOWRAP_WORD_SMART);
	builtin_warning_label->hide();

	status_panel = memnew(PanelContainer);
	status_panel->set_h_size_flags(Control::SIZE_FILL);
	status_panel->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	status_panel->add_child(vb);

	// Spacing.

	Control *spacing = memnew(Control);
	spacing->set_custom_minimum_size(Size2(0, 10 * EDSCALE));

	vb = memnew(VBoxContainer);
	vb->add_child(gc);
	vb->add_child(spacing);
	vb->add_child(status_panel);
	add_child(vb);

	// Language.

	language_menu = memnew(OptionButton);
	language_menu->set_custom_minimum_size(Size2(250, 0) * EDSCALE);
	language_menu->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	gc->add_child(memnew(Label(TTR("Language:"))));
	gc->add_child(language_menu);

	for (int i = 0; i < SHADER_TYPE_MAX; i++) {
		String language;
		bool invalid = false;
		switch (i) {
			case SHADER_TYPE_TEXT:
				language = "Shader";
				default_language = i;
				break;
			case SHADER_TYPE_VISUAL:
				language = "VisualShader";
				break;
			case SHADER_TYPE_MAX:
				invalid = true;
				break;
			default:
				invalid = true;
				break;
		}
		if (invalid) {
			continue;
		}
		language_menu->add_item(language);
	}
	if (default_language >= 0) {
		language_menu->select(default_language);
	}
	current_language = default_language;
	language_menu->connect("item_selected", callable_mp(this, &ShaderCreateDialog::_language_changed));

	// Modes.

	mode_menu = memnew(OptionButton);
	for (const String &type_name : ShaderTypes::get_singleton()->get_types_list()) {
		mode_menu->add_item(type_name.capitalize());
	}
	gc->add_child(memnew(Label(TTR("Mode:"))));
	gc->add_child(mode_menu);
	mode_menu->connect("item_selected", callable_mp(this, &ShaderCreateDialog::_mode_changed));

	// Templates.

	template_menu = memnew(OptionButton);
	gc->add_child(memnew(Label(TTR("Template:"))));
	gc->add_child(template_menu);
	template_menu->connect("item_selected", callable_mp(this, &ShaderCreateDialog::_template_changed));

	// Built-in Shader.

	internal = memnew(CheckBox);
	internal->set_text(TTR("On"));
	internal->connect("toggled", callable_mp(this, &ShaderCreateDialog::_built_in_toggled));
	gc->add_child(memnew(Label(TTR("Built-in Shader:"))));
	gc->add_child(internal);

	// Path.

	HBoxContainer *hb = memnew(HBoxContainer);
	hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb->connect("sort_children", callable_mp(this, &ShaderCreateDialog::_path_hbox_sorted));
	file_path = memnew(LineEdit);
	file_path->connect("text_changed", callable_mp(this, &ShaderCreateDialog::_path_changed));
	file_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb->add_child(file_path);
	path_button = memnew(Button);
	path_button->connect("pressed", callable_mp(this, &ShaderCreateDialog::_browse_path));
	hb->add_child(path_button);
	gc->add_child(memnew(Label(TTR("Path:"))));
	gc->add_child(hb);

	// Dialog Setup.

	file_browse = memnew(EditorFileDialog);
	file_browse->connect("file_selected", callable_mp(this, &ShaderCreateDialog::_file_selected));
	file_browse->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	add_child(file_browse);

	alert = memnew(AcceptDialog);
	alert->get_label()->set_autowrap_mode(Label::AUTOWRAP_WORD_SMART);
	alert->get_label()->set_align(Label::ALIGN_CENTER);
	alert->get_label()->set_valign(Label::VALIGN_CENTER);
	alert->get_label()->set_custom_minimum_size(Size2(325, 60) * EDSCALE);
	add_child(alert);

	get_ok_button()->set_text(TTR("Create"));
	set_hide_on_ok(false);

	set_title(TTR("Create Shader"));
}

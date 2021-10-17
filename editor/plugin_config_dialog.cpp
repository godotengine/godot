/*************************************************************************/
/*  plugin_config_dialog.cpp                                             */
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

#include "plugin_config_dialog.h"

#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/editor_scale.h"
#include "editor/project_settings_editor.h"
#include "scene/gui/grid_container.h"

#include "modules/modules_enabled.gen.h"
#ifdef MODULE_GDSCRIPT_ENABLED
#include "modules/gdscript/gdscript.h"
#endif

void PluginConfigDialog::_clear_fields() {
	name_edit->set_text("");
	subfolder_edit->set_text("");
	desc_edit->set_text("");
	author_edit->set_text("");
	version_edit->set_text("");
	script_edit->set_text("");
}

void PluginConfigDialog::_on_confirmed() {
	String path = "res://addons/" + subfolder_edit->get_text();

	if (!_edit_mode) {
		DirAccess *d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		if (!d || d->make_dir_recursive(path) != OK) {
			return;
		}
	}

	Ref<ConfigFile> cf = memnew(ConfigFile);
	cf->set_value("plugin", "name", name_edit->get_text());
	cf->set_value("plugin", "description", desc_edit->get_text());
	cf->set_value("plugin", "author", author_edit->get_text());
	cf->set_value("plugin", "version", version_edit->get_text());
	cf->set_value("plugin", "script", script_edit->get_text());

	cf->save(path.plus_file("plugin.cfg"));

	if (!_edit_mode) {
		int lang_idx = script_option_edit->get_selected();
		String lang_name = ScriptServer::get_language(lang_idx)->get_name();

		Ref<Script> script;

		// TODO Use script templates. Right now, this code won't add the 'tool' annotation to other languages.
		// TODO Better support script languages with named classes (has_named_classes).

		// FIXME: It's hacky to have hardcoded access to the GDScript module here.
		// The editor code should not have to know what languages are enabled.
#ifdef MODULE_GDSCRIPT_ENABLED
		if (lang_name == GDScriptLanguage::get_singleton()->get_name()) {
			// Hard-coded GDScript template to keep usability until we use script templates.
			Ref<Script> gdscript = memnew(GDScript);
			gdscript->set_source_code(
					"@tool\n"
					"extends EditorPlugin\n"
					"\n"
					"\n"
					"func _enter_tree()%VOID_RETURN%:\n"
					"%TS%pass\n"
					"\n"
					"\n"
					"func _exit_tree()%VOID_RETURN%:\n"
					"%TS%pass\n");
			GDScriptLanguage::get_singleton()->make_template("", "", gdscript);
			String script_path = path.plus_file(script_edit->get_text());
			gdscript->set_path(script_path);
			ResourceSaver::save(script_path, gdscript);
			script = gdscript;
		} else {
#endif
			String script_path = path.plus_file(script_edit->get_text());
			String class_name = script_path.get_file().get_basename();
			script = ScriptServer::get_language(lang_idx)->get_template(class_name, "EditorPlugin");
			script->set_path(script_path);
			ResourceSaver::save(script_path, script);
#ifdef MODULE_GDSCRIPT_ENABLED
		}
#endif

		emit_signal(SNAME("plugin_ready"), script.operator->(), active_edit->is_pressed() ? _to_absolute_plugin_path(subfolder_edit->get_text()) : "");
	} else {
		EditorNode::get_singleton()->get_project_settings()->update_plugins();
	}
	_clear_fields();
}

void PluginConfigDialog::_on_cancelled() {
	_clear_fields();
}

void PluginConfigDialog::_on_language_changed(const int) {
	_on_required_text_changed(String());
}

void PluginConfigDialog::_on_required_text_changed(const String &) {
	int lang_idx = script_option_edit->get_selected();
	String ext = ScriptServer::get_language(lang_idx)->get_extension();

	Ref<Texture2D> valid_icon = get_theme_icon(SNAME("StatusSuccess"), SNAME("EditorIcons"));
	Ref<Texture2D> invalid_icon = get_theme_icon(SNAME("StatusWarning"), SNAME("EditorIcons"));

	// Set variables to assume all is valid
	bool is_valid = true;
	name_validation->set_texture(valid_icon);
	subfolder_validation->set_texture(valid_icon);
	script_validation->set_texture(valid_icon);
	name_validation->set_tooltip("");
	subfolder_validation->set_tooltip("");
	script_validation->set_tooltip("");

	// Change valid status to invalid depending on conditions.
	Vector<String> errors;
	if (name_edit->get_text().is_empty()) {
		is_valid = false;
		name_validation->set_texture(invalid_icon);
		name_validation->set_tooltip(TTR("Plugin name cannot not be blank."));
	}
	if (script_edit->get_text().get_extension() != ext) {
		is_valid = false;
		script_validation->set_texture(invalid_icon);
		script_validation->set_tooltip(vformat(TTR("Script extension must match chosen language extension (.%s)."), ext));
	}
	if (script_edit->get_text().get_basename().is_empty()) {
		is_valid = false;
		script_validation->set_texture(invalid_icon);
		script_validation->set_tooltip(TTR("Script name cannot not be blank."));
	}
	if (subfolder_edit->get_text().is_empty()) {
		is_valid = false;
		subfolder_validation->set_texture(invalid_icon);
		subfolder_validation->set_tooltip(TTR("Subfolder cannot be blank."));
	} else if (!subfolder_edit->get_text().is_valid_filename()) {
		subfolder_validation->set_texture(invalid_icon);
		subfolder_validation->set_tooltip(TTR("Subfolder name is not a valid folder name."));
	} else {
		DirAccessRef dir = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		String path = "res://addons/" + subfolder_edit->get_text();
		if (dir->dir_exists(path) && !_edit_mode) { // Only show this error if in "create" mode.
			is_valid = false;
			subfolder_validation->set_texture(invalid_icon);
			subfolder_validation->set_tooltip(TTR("Subfolder cannot be one which already exists."));
		}
	}

	get_ok_button()->set_disabled(!is_valid);
}

String PluginConfigDialog::_to_absolute_plugin_path(const String &p_plugin_name) {
	return "res://addons/" + p_plugin_name + "/plugin.cfg";
}

void PluginConfigDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				name_edit->grab_focus();
			}
		} break;
		case NOTIFICATION_READY: {
			connect("confirmed", callable_mp(this, &PluginConfigDialog::_on_confirmed));
			get_cancel_button()->connect("pressed", callable_mp(this, &PluginConfigDialog::_on_cancelled));
		} break;
	}
}

void PluginConfigDialog::config(const String &p_config_path) {
	if (p_config_path.length()) {
		Ref<ConfigFile> cf = memnew(ConfigFile);
		Error err = cf->load(p_config_path);
		ERR_FAIL_COND_MSG(err != OK, "Cannot load config file from path '" + p_config_path + "'.");

		name_edit->set_text(cf->get_value("plugin", "name", ""));
		subfolder_edit->set_text(p_config_path.get_base_dir().get_basename().get_file());
		desc_edit->set_text(cf->get_value("plugin", "description", ""));
		author_edit->set_text(cf->get_value("plugin", "author", ""));
		version_edit->set_text(cf->get_value("plugin", "version", ""));
		script_edit->set_text(cf->get_value("plugin", "script", ""));

		_edit_mode = true;
		active_edit->hide();
		Object::cast_to<Label>(active_edit->get_parent()->get_child(active_edit->get_index() - 2))->hide();
		subfolder_edit->hide();
		Object::cast_to<Label>(subfolder_edit->get_parent()->get_child(subfolder_edit->get_index() - 2))->hide();
		set_title(TTR("Edit a Plugin"));
	} else {
		_clear_fields();
		_edit_mode = false;
		active_edit->show();
		Object::cast_to<Label>(active_edit->get_parent()->get_child(active_edit->get_index() - 2))->show();
		subfolder_edit->show();
		Object::cast_to<Label>(subfolder_edit->get_parent()->get_child(subfolder_edit->get_index() - 2))->show();
		set_title(TTR("Create a Plugin"));
	}
	// Simulate text changing so the errors populate.
	_on_required_text_changed("");

	get_ok_button()->set_disabled(!_edit_mode);
	get_ok_button()->set_text(_edit_mode ? TTR("Update") : TTR("Create"));
}

void PluginConfigDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo("plugin_ready", PropertyInfo(Variant::STRING, "script_path", PROPERTY_HINT_NONE, ""), PropertyInfo(Variant::STRING, "activate_name")));
}

PluginConfigDialog::PluginConfigDialog() {
	get_ok_button()->set_disabled(true);
	set_hide_on_ok(true);

	VBoxContainer *vbox = memnew(VBoxContainer);
	vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	vbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(vbox);

	GridContainer *grid = memnew(GridContainer);
	grid->set_columns(3);
	vbox->add_child(grid);

	// Plugin Name
	Label *name_lb = memnew(Label);
	name_lb->set_text(TTR("Plugin Name:"));
	grid->add_child(name_lb);

	name_validation = memnew(TextureRect);
	name_validation->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	grid->add_child(name_validation);

	name_edit = memnew(LineEdit);
	name_edit->connect("text_changed", callable_mp(this, &PluginConfigDialog::_on_required_text_changed));
	name_edit->set_placeholder("MyPlugin");
	grid->add_child(name_edit);

	// Subfolder
	Label *subfolder_lb = memnew(Label);
	subfolder_lb->set_text(TTR("Subfolder:"));
	grid->add_child(subfolder_lb);

	subfolder_validation = memnew(TextureRect);
	subfolder_validation->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	grid->add_child(subfolder_validation);

	subfolder_edit = memnew(LineEdit);
	subfolder_edit->set_placeholder("\"my_plugin\" -> res://addons/my_plugin");
	subfolder_edit->connect("text_changed", callable_mp(this, &PluginConfigDialog::_on_required_text_changed));
	grid->add_child(subfolder_edit);

	// Description
	Label *desc_lb = memnew(Label);
	desc_lb->set_text(TTR("Description:"));
	grid->add_child(desc_lb);

	Control *desc_spacer = memnew(Control);
	grid->add_child(desc_spacer);

	desc_edit = memnew(TextEdit);
	desc_edit->set_custom_minimum_size(Size2(400, 80) * EDSCALE);
	desc_edit->set_line_wrapping_mode(TextEdit::LineWrappingMode::LINE_WRAPPING_BOUNDARY);
	grid->add_child(desc_edit);

	// Author
	Label *author_lb = memnew(Label);
	author_lb->set_text(TTR("Author:"));
	grid->add_child(author_lb);

	Control *author_spacer = memnew(Control);
	grid->add_child(author_spacer);

	author_edit = memnew(LineEdit);
	author_edit->set_placeholder("Godette");
	grid->add_child(author_edit);

	// Version
	Label *version_lb = memnew(Label);
	version_lb->set_text(TTR("Version:"));
	grid->add_child(version_lb);

	Control *version_spacer = memnew(Control);
	grid->add_child(version_spacer);

	version_edit = memnew(LineEdit);
	version_edit->set_placeholder("1.0");
	grid->add_child(version_edit);

	// Language dropdown
	Label *script_option_lb = memnew(Label);
	script_option_lb->set_text(TTR("Language:"));
	grid->add_child(script_option_lb);

	Control *script_opt_spacer = memnew(Control);
	grid->add_child(script_opt_spacer);

	script_option_edit = memnew(OptionButton);
	int default_lang = 0;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptLanguage *lang = ScriptServer::get_language(i);
		script_option_edit->add_item(lang->get_name());
#ifdef MODULE_GDSCRIPT_ENABLED
		if (lang == GDScriptLanguage::get_singleton()) {
			default_lang = i;
		}
#endif
	}
	script_option_edit->select(default_lang);
	grid->add_child(script_option_edit);
	script_option_edit->connect("item_selected", callable_mp(this, &PluginConfigDialog::_on_language_changed));

	// Plugin Script Name
	Label *script_lb = memnew(Label);
	script_lb->set_text(TTR("Script Name:"));
	grid->add_child(script_lb);

	script_validation = memnew(TextureRect);
	script_validation->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	grid->add_child(script_validation);

	script_edit = memnew(LineEdit);
	script_edit->connect("text_changed", callable_mp(this, &PluginConfigDialog::_on_required_text_changed));
	script_edit->set_placeholder("\"plugin.gd\" -> res://addons/my_plugin/plugin.gd");
	grid->add_child(script_edit);

	// Activate now checkbox
	// TODO Make this option work better with languages like C#. Right now, it does not work because the C# project must be compiled first.
	Label *active_lb = memnew(Label);
	active_lb->set_text(TTR("Activate now?"));
	grid->add_child(active_lb);

	Control *active_spacer = memnew(Control);
	grid->add_child(active_spacer);

	active_edit = memnew(CheckBox);
	active_edit->set_pressed(true);
	grid->add_child(active_edit);
}

PluginConfigDialog::~PluginConfigDialog() {
}

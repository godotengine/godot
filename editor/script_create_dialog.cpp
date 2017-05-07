/*************************************************************************/
/*  script_create_dialog.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor/editor_scale.h"
#include "editor_file_system.h"
#include "global_config.h"
#include "io/resource_saver.h"
#include "os/file_access.h"
#include "script_language.h"

void ScriptCreateDialog::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			path_button->set_icon(get_icon("Folder", "EditorIcons"));
			parent_browse_button->set_icon(get_icon("Folder", "EditorIcons"));
		}
	}
}

void ScriptCreateDialog::config(const String &p_base_name, const String &p_base_path) {

	class_name->set_text("");
	parent_name->set_text(p_base_name);
	if (p_base_path != "") {
		initial_bp = p_base_path.get_basename();
		file_path->set_text(initial_bp + "." + ScriptServer::get_language(language_menu->get_selected())->get_extension());
	} else {
		initial_bp = "";
		file_path->set_text("");
	}
	_lang_changed(current_language);
	_parent_name_changed(parent_name->get_text());
	_class_name_changed("");
	_path_changed(file_path->get_text());
}

bool ScriptCreateDialog::_validate(const String &p_string) {

	if (p_string.length() == 0)
		return false;

	String path_chars = "\"res://";
	bool is_val_path = ScriptServer::get_language(language_menu->get_selected())->can_inherit_from_file();
	for (int i = 0; i < p_string.length(); i++) {

		if (i == 0) {
			if (p_string[0] >= '0' && p_string[0] <= '9')
				return false; // no start with number plz
		}

		if (i == p_string.length() - 1 && is_val_path)
			return p_string[i] == '\"';

		if (is_val_path && i < path_chars.length()) {
			if (p_string[i] != path_chars[i])
				is_val_path = false;
			else
				continue;
		}

		bool valid_char = (p_string[i] >= '0' && p_string[i] <= '9') || (p_string[i] >= 'a' && p_string[i] <= 'z') || (p_string[i] >= 'A' && p_string[i] <= 'Z') || p_string[i] == '_' || (is_val_path && (p_string[i] == '/' || p_string[i] == '.'));

		if (!valid_char)
			return false;
	}

	return true;
}

void ScriptCreateDialog::_class_name_changed(const String &p_name) {

	if (_validate(class_name->get_text())) {
		is_class_name_valid = true;
	} else {
		is_class_name_valid = false;
	}
	_update_dialog();
}

void ScriptCreateDialog::_parent_name_changed(const String &p_parent) {

	if (_validate(parent_name->get_text())) {
		is_parent_name_valid = true;
	} else {
		is_parent_name_valid = false;
	}
	_update_dialog();
}

void ScriptCreateDialog::ok_pressed() {

	if (is_new_script_created) {
		_create_new();
	} else {
		_load_exist();
	}

	is_new_script_created = true;
	_update_dialog();
}

void ScriptCreateDialog::_create_new() {

	String cname;
	if (has_named_classes)
		cname = class_name->get_text();

	Ref<Script> scr = ScriptServer::get_language(language_menu->get_selected())->get_template(cname, parent_name->get_text());

	String selected_language = language_menu->get_item_text(language_menu->get_selected());
	editor_settings->set_project_metadata("script_setup", "last_selected_language", selected_language);

	if (cname != "")
		scr->set_name(cname);

	if (!is_built_in) {
		String lpath = GlobalConfig::get_singleton()->localize_path(file_path->get_text());
		scr->set_path(lpath);
		Error err = ResourceSaver::save(lpath, scr, ResourceSaver::FLAG_CHANGE_PATH);
		if (err != OK) {
			alert->set_text(TTR("Error - Could not create script in filesystem."));
			alert->popup_centered();
			return;
		}
	}

	hide();
	emit_signal("script_created", scr);
}

void ScriptCreateDialog::_load_exist() {

	String path = file_path->get_text();
	RES p_script = ResourceLoader::load(path, "Script");
	if (p_script.is_null()) {
		alert->get_ok()->set_text(TTR("OK"));
		alert->set_text(vformat(TTR("Error loading script from %s"), path));
		alert->popup_centered();
		return;
	}

	hide();
	emit_signal("script_created", p_script.get_ref_ptr());
}

void ScriptCreateDialog::_lang_changed(int l) {

	l = language_menu->get_selected();
	if (ScriptServer::get_language(l)->has_named_classes()) {
		has_named_classes = true;
	} else {
		has_named_classes = false;
	}

	if (ScriptServer::get_language(l)->can_inherit_from_file()) {
		can_inherit_from_file = true;
	} else {
		can_inherit_from_file = false;
	}

	String selected_ext = "." + ScriptServer::get_language(l)->get_extension();
	String path = file_path->get_text();
	String extension = "";
	if (path != "") {
		if (path.find(".") >= 0) {
			extension = path.get_extension();
		}

		if (extension.length() == 0) {
			// add extension if none
			path += selected_ext;
			_path_changed(path);
		} else {
			// change extension by selected language
			List<String> extensions;
			// get all possible extensions for script
			for (int l = 0; l < language_menu->get_item_count(); l++) {
				ScriptServer::get_language(l)->get_recognized_extensions(&extensions);
			}

			for (List<String>::Element *E = extensions.front(); E; E = E->next()) {
				if (E->get().nocasecmp_to(extension) == 0) {
					path = path.get_basename() + selected_ext;
					_path_changed(path);
					break;
				}
			}
		}
		file_path->set_text(path);
	}

	_update_dialog();
}

void ScriptCreateDialog::_built_in_pressed() {

	if (internal->is_pressed()) {
		is_built_in = true;
	} else {
		is_built_in = false;
	}
	_update_dialog();
}

void ScriptCreateDialog::_browse_path(bool browse_parent) {

	is_browsing_parent = browse_parent;

	file_browse->set_mode(EditorFileDialog::MODE_SAVE_FILE);
	file_browse->set_disable_overwrite_warning(true);
	file_browse->clear_filters();
	List<String> extensions;

	// get all possible extensions for script
	for (int l = 0; l < language_menu->get_item_count(); l++) {
		ScriptServer::get_language(l)->get_recognized_extensions(&extensions);
	}

	for (List<String>::Element *E = extensions.front(); E; E = E->next()) {
		file_browse->add_filter("*." + E->get());
	}

	file_browse->set_current_path(file_path->get_text());
	file_browse->popup_centered_ratio();
}

void ScriptCreateDialog::_file_selected(const String &p_file) {

	String p = GlobalConfig::get_singleton()->localize_path(p_file);
	if (is_browsing_parent) {
		parent_name->set_text("\"" + p + "\"");
		_class_name_changed("\"" + p + "\"");
	} else {
		file_path->set_text(p);
		_path_changed(p);
	}
}

void ScriptCreateDialog::_path_changed(const String &p_path) {

	is_path_valid = false;
	is_new_script_created = true;
	String p = p_path;

	if (p == "") {
		_msg_path_valid(false, TTR("Path is empty"));
		_update_dialog();
		return;
	}

	p = GlobalConfig::get_singleton()->localize_path(p);
	if (!p.begins_with("res://")) {
		_msg_path_valid(false, TTR("Path is not local"));
		_update_dialog();
		return;
	}

	if (p.find("/") || p.find("\\")) {
		DirAccess *d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		if (d->change_dir(p.get_base_dir()) != OK) {
			_msg_path_valid(false, TTR("Invalid base path"));
			memdelete(d);
			_update_dialog();
			return;
		}
		memdelete(d);
	}

	/* Does file already exist */

	DirAccess *f = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (f->file_exists(p) && !(f->current_is_dir())) {
		is_new_script_created = false;
		is_path_valid = true;
	}
	memdelete(f);
	_update_dialog();

	/* Check file extension */

	String extension = p.get_extension();
	List<String> extensions;

	// get all possible extensions for script
	for (int l = 0; l < language_menu->get_item_count(); l++) {
		ScriptServer::get_language(l)->get_recognized_extensions(&extensions);
	}

	bool found = false;
	bool match = false;
	int index = 0;
	for (List<String>::Element *E = extensions.front(); E; E = E->next()) {
		if (E->get().nocasecmp_to(extension) == 0) {
			//FIXME (?) - changing language this way doesn't update controls, needs rework
			//language_menu->select(index); // change Language option by extension
			found = true;
			if (E->get() == ScriptServer::get_language(language_menu->get_selected())->get_extension()) {
				match = true;
			}
			break;
		}
		index++;
	}

	if (!found) {
		_msg_path_valid(false, TTR("Invalid extension"));
		_update_dialog();
		return;
	}

	if (!match) {
		_msg_path_valid(false, TTR("Wrong extension chosen"));
		_update_dialog();
		return;
	}

	/* All checks passed */

	is_path_valid = true;
	_update_dialog();
}

void ScriptCreateDialog::_msg_script_valid(bool valid, const String &p_msg) {

	error_label->set_text(TTR(p_msg));
	if (valid) {
		error_label->add_color_override("font_color", Color(0, 1.0, 0.8, 0.8));
	} else {
		error_label->add_color_override("font_color", Color(1, 0.2, 0.2, 0.8));
	}
}

void ScriptCreateDialog::_msg_path_valid(bool valid, const String &p_msg) {

	path_error_label->set_text(TTR(p_msg));
	if (valid) {
		path_error_label->add_color_override("font_color", Color(0, 1.0, 0.8, 0.8));
	} else {
		path_error_label->add_color_override("font_color", Color(1, 0.4, 0.0, 0.8));
	}
}

void ScriptCreateDialog::_update_dialog() {

	bool script_ok = true;

	/* "Add Script Dialog" gui logic and script checks */

	// Is Script Valid (order from top to bottom)
	get_ok()->set_disabled(true);
	if (!is_built_in) {
		if (!is_path_valid) {
			_msg_script_valid(false, TTR("Invalid Path"));
			script_ok = false;
		}
	}
	if (has_named_classes && (!is_class_name_valid)) {
		_msg_script_valid(false, TTR("Invalid class name"));
		script_ok = false;
	}
	if (!is_parent_name_valid) {
		_msg_script_valid(false, TTR("Invalid inherited parent name or path"));
		script_ok = false;
	}
	if (script_ok) {
		_msg_script_valid(true, TTR("Script valid"));
		get_ok()->set_disabled(false);
	}

	/* Does script have named classes */

	if (has_named_classes) {
		if (is_new_script_created) {
			class_name->set_editable(true);
			class_name->set_placeholder(TTR("Allowed: a-z, A-Z, 0-9 and _"));
			class_name->set_placeholder_alpha(0.3);
		} else {
			class_name->set_editable(false);
		}
	} else {
		class_name->set_editable(false);
		class_name->set_placeholder(TTR("N/A"));
		class_name->set_placeholder_alpha(1);
	}

	/* Can script inherit from a file */

	if (can_inherit_from_file) {
		parent_browse_button->set_disabled(false);
	} else {
		parent_browse_button->set_disabled(true);
	}

	/* Is script Built-in */

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

	/* Is Script created or loaded from existing file */

	if (is_new_script_created) {
		// New Script Created
		get_ok()->set_text(TTR("Create"));
		parent_name->set_editable(true);
		parent_browse_button->set_disabled(false);
		internal->set_disabled(false);
		if (is_built_in) {
			_msg_path_valid(true, TTR("Built-in script (into scene file)"));
		} else {
			if (script_ok) {
				_msg_path_valid(true, TTR("Create new script file"));
			}
		}
	} else {
		// Script Loaded
		get_ok()->set_text(TTR("Load"));
		parent_name->set_editable(false);
		parent_browse_button->set_disabled(true);
		internal->set_disabled(true);
		if (script_ok) {
			_msg_path_valid(true, TTR("Load existing script file"));
		}
	}
}

void ScriptCreateDialog::_bind_methods() {

	ClassDB::bind_method("_class_name_changed", &ScriptCreateDialog::_class_name_changed);
	ClassDB::bind_method("_parent_name_changed", &ScriptCreateDialog::_parent_name_changed);
	ClassDB::bind_method("_lang_changed", &ScriptCreateDialog::_lang_changed);
	ClassDB::bind_method("_built_in_pressed", &ScriptCreateDialog::_built_in_pressed);
	ClassDB::bind_method("_browse_path", &ScriptCreateDialog::_browse_path);
	ClassDB::bind_method("_file_selected", &ScriptCreateDialog::_file_selected);
	ClassDB::bind_method("_path_changed", &ScriptCreateDialog::_path_changed);
	ADD_SIGNAL(MethodInfo("script_created", PropertyInfo(Variant::OBJECT, "script", PROPERTY_HINT_RESOURCE_TYPE, "Script")));
}

ScriptCreateDialog::ScriptCreateDialog() {

	editor_settings = EditorSettings::get_singleton();

	GridContainer *gc = memnew(GridContainer);
	VBoxContainer *vb = memnew(VBoxContainer);
	HBoxContainer *hb = memnew(HBoxContainer);
	Label *l = memnew(Label);
	Control *empty = memnew(Control);
	Control *empty_h = memnew(Control);
	Control *empty_v = memnew(Control);
	PanelContainer *pc = memnew(PanelContainer);

	/* DIALOG */

	/* Main Controls */

	gc = memnew(GridContainer);
	gc->set_columns(2);

	/* Error Stylebox Background */

	StyleBoxFlat *sb = memnew(StyleBoxFlat);
	sb->set_bg_color(Color(0, 0, 0, 0.05));
	sb->set_light_color(Color(1, 1, 1, 0.05));
	sb->set_dark_color(Color(1, 1, 1, 0.05));
	sb->set_border_blend(false);
	sb->set_border_size(1);
	sb->set_default_margin(MARGIN_TOP, 10.0 * EDSCALE);
	sb->set_default_margin(MARGIN_BOTTOM, 10.0 * EDSCALE);
	sb->set_default_margin(MARGIN_LEFT, 10.0 * EDSCALE);
	sb->set_default_margin(MARGIN_RIGHT, 10.0 * EDSCALE);

	/* Error Messages Field */

	vb = memnew(VBoxContainer);

	hb = memnew(HBoxContainer);
	l = memnew(Label);
	l->set_text(" - ");
	hb->add_child(l);
	error_label = memnew(Label);
	error_label->set_text(TTR("Error!"));
	error_label->set_align(Label::ALIGN_LEFT);
	hb->add_child(error_label);
	vb->add_child(hb);

	hb = memnew(HBoxContainer);
	l = memnew(Label);
	l->set_text(" - ");
	hb->add_child(l);
	path_error_label = memnew(Label);
	path_error_label->set_text(TTR("Error!"));
	path_error_label->set_align(Label::ALIGN_LEFT);
	hb->add_child(path_error_label);
	vb->add_child(hb);

	pc = memnew(PanelContainer);
	pc->set_h_size_flags(Control::SIZE_FILL);
	pc->add_style_override("panel", sb);
	pc->add_child(vb);

	/* Margins */

	empty_h = memnew(Control);
	empty_h->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	empty_h->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	empty_h->set_custom_minimum_size(Size2(0, 10 * EDSCALE));
	empty_v = memnew(Control);
	empty_v->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	empty_v->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	empty_v->set_custom_minimum_size(Size2(10, 0 * EDSCALE));

	vb = memnew(VBoxContainer);
	vb->add_child(empty_h->duplicate());
	vb->add_child(gc);
	vb->add_child(empty_h->duplicate());
	vb->add_child(pc);
	vb->add_child(empty_h->duplicate());
	hb = memnew(HBoxContainer);
	hb->add_child(empty_v->duplicate());
	hb->add_child(vb);
	hb->add_child(empty_v->duplicate());

	add_child(hb);

	/* Language */

	language_menu = memnew(OptionButton);
	language_menu->set_custom_minimum_size(Size2(250, 0) * EDSCALE);
	language_menu->set_h_size_flags(SIZE_EXPAND_FILL);
	l = memnew(Label);
	l->set_text(TTR("Language"));
	l->set_align(Label::ALIGN_RIGHT);
	gc->add_child(l);
	gc->add_child(language_menu);

	int default_lang = 0;
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {

		String lang = ScriptServer::get_language(i)->get_name();
		language_menu->add_item(lang);
		if (lang == "GDScript") {
			default_lang = i;
		}
	}

	String last_selected_language = editor_settings->get_project_metadata("script_setup", "last_selected_language", "");
	if (last_selected_language != "") {
		for (int i = 0; i < language_menu->get_item_count(); i++) {
			if (language_menu->get_item_text(i) == last_selected_language) {
				language_menu->select(i);
				current_language = i;
				break;
			}
		}
	} else {
		language_menu->select(default_lang);
		current_language = default_lang;
	}

	language_menu->connect("item_selected", this, "_lang_changed");

	/* Inherits */

	hb = memnew(HBoxContainer);
	hb->set_h_size_flags(SIZE_EXPAND_FILL);
	parent_name = memnew(LineEdit);
	parent_name->connect("text_changed", this, "_parent_name_changed");
	parent_name->set_h_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(parent_name);
	parent_browse_button = memnew(Button);
	parent_browse_button->set_flat(true);
	parent_browse_button->connect("pressed", this, "_browse_path", varray(true));
	hb->add_child(parent_browse_button);
	l = memnew(Label);
	l->set_text(TTR("Inherits"));
	l->set_align(Label::ALIGN_RIGHT);
	gc->add_child(l);
	gc->add_child(hb);
	is_browsing_parent = false;

	/* Class Name */

	class_name = memnew(LineEdit);
	class_name->connect("text_changed", this, "_class_name_changed");
	class_name->set_h_size_flags(SIZE_EXPAND_FILL);
	l = memnew(Label);
	l->set_text(TTR("Class Name"));
	l->set_align(Label::ALIGN_RIGHT);
	gc->add_child(l);
	gc->add_child(class_name);

	/* Built-in Script */

	internal = memnew(CheckButton);
	internal->connect("pressed", this, "_built_in_pressed");
	hb = memnew(HBoxContainer);
	empty = memnew(Control);
	hb->add_child(internal);
	hb->add_child(empty);
	l = memnew(Label);
	l->set_text(TTR("Built-in Script"));
	l->set_align(Label::ALIGN_RIGHT);
	gc->add_child(l);
	gc->add_child(hb);

	/* Path */

	hb = memnew(HBoxContainer);
	file_path = memnew(LineEdit);
	file_path->connect("text_changed", this, "_path_changed");
	file_path->set_h_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(file_path);
	path_button = memnew(Button);
	path_button->set_flat(true);
	path_button->connect("pressed", this, "_browse_path", varray(false));
	hb->add_child(path_button);
	l = memnew(Label);
	l->set_text(TTR("Path"));
	l->set_align(Label::ALIGN_RIGHT);
	gc->add_child(l);
	gc->add_child(hb);

	/* Dialog Setup */

	file_browse = memnew(EditorFileDialog);
	file_browse->connect("file_selected", this, "_file_selected");
	add_child(file_browse);
	get_ok()->set_text(TTR("Create"));
	alert = memnew(AcceptDialog);
	alert->set_as_minsize();
	alert->get_label()->set_autowrap(true);
	alert->get_label()->set_align(Label::ALIGN_CENTER);
	alert->get_label()->set_valign(Label::VALIGN_CENTER);
	alert->get_label()->set_custom_minimum_size(Size2(325, 60) * EDSCALE);
	add_child(alert);

	set_as_minsize();
	set_hide_on_ok(false);
	set_title(TTR("Attach Node Script"));

	is_parent_name_valid = false;
	is_class_name_valid = false;
	is_path_valid = false;

	has_named_classes = false;
	can_inherit_from_file = false;
	is_built_in = false;

	is_new_script_created = true;
}

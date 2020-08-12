/*************************************************************************/
/*  documentation_generation_dialog.cpp                                  */
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

#include "documentation_generation_dialog.h"

#include "core/script_language.h"
// TODO: editor help is for debugging, remove
#include "editor/editor_help.h"
#include "editor/editor_scale.h"
// TODO(require core): remove gdscript here (added for testing)
#include "modules/gdscript/gdscript.h"
#include "scene/resources/text_file.h"

void DocumentationGenerationDialog::_path_changed(const String &p_path, bool p_is_input) {
	String path = p_path.strip_edges();

	String path_target;
	Label *error_label = nullptr;
	bool *path_valid_bool = nullptr;

	if (p_is_input) {
		path_target = "Input";
		error_label = input_error_label;
		path_valid_bool = &is_input_path_valid;
	} else {
		path_target = "Target";
		error_label = target_error_label;
		path_valid_bool = &is_target_path_valid;
	}

	if (path == "") {
		// TODO: does it make sence in other languages.
		error_label->set_text("- " + path_target + " " + TTR("path is empty."));
		error_label->add_theme_color_override("font_color", gc->get_theme_color("error_color", "Editor"));
		*path_valid_bool = false;
		_update_dialog();
		return;
	}

	path = ProjectSettings::get_singleton()->localize_path(path);
	if (!path.begins_with("res://")) {
		error_label->set_text("- " + path_target + " " + TTR("path is not local."));
		error_label->add_theme_color_override("font_color", gc->get_theme_color("error_color", "Editor"));
		*path_valid_bool = false;
		_update_dialog();
		return;
	}

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (d->change_dir(path) != OK) {
		memdelete(d);
		error_label->set_text("- " + path_target + " " + TTR("path is invalid."));
		error_label->add_theme_color_override("font_color", gc->get_theme_color("error_color", "Editor"));
		*path_valid_bool = false;
		_update_dialog();
		return;
	}

	if (!p_is_input) {
		d->list_dir_begin();
		bool is_empty = true;
		String n = d->get_next();
		while (n != String()) {
			if (!n.begins_with(".")) {
				is_empty = false;
				break;
			}
			n = d->get_next();
		}
		d->list_dir_end();

		if (!is_empty) {
			error_label->set_text("- " + TTR("Target path is not empty, files will be overriden."));
			error_label->add_theme_color_override("font_color", gc->get_theme_color("warning_color", "Editor"));
			*path_valid_bool = true;
			_update_dialog();
			return;
		}
	}
	memdelete(d);

	error_label->set_text("- " + path_target + " " + TTR("path is valid."));
	error_label->add_theme_color_override("font_color", gc->get_theme_color("success_color", "Editor"));
	*path_valid_bool = true;
	_update_dialog();
	return;
}

#define EXCLUDE_OPTIONS(m_doc)                                            \
	if ((m_doc.name.begins_with("_") && exclude_private->is_pressed()) || \
			(m_doc.description == "" && documented_only->is_pressed())) { \
		continue;                                                         \
	} else                                                                \
		((void)0)

DocData::ClassDoc DocumentationGenerationDialog::_apply_options_filter(const DocData::ClassDoc &p_class) {
	DocData::ClassDoc cd;

	cd.name = p_class.name;
	cd.inherits = p_class.inherits;
	cd.brief_description = p_class.brief_description;
	cd.description = p_class.description;
	cd.tutorials = p_class.tutorials;

	/* Constants */
	for (int i = 0; i < p_class.constants.size(); i++) {
		EXCLUDE_OPTIONS(p_class.constants[i]);
		cd.constants.append(p_class.constants[i]);
	}
	/* Methods */
	for (int i = 0; i < p_class.methods.size(); i++) {
		EXCLUDE_OPTIONS(p_class.methods[i]);
		cd.methods.append(p_class.methods[i]);
	}
	/* Signals */
	for (int i = 0; i < p_class.signals.size(); i++) {
		EXCLUDE_OPTIONS(p_class.signals[i]);
		cd.signals.append(p_class.signals[i]);
	}
	/* Variables */
	for (int i = 0; i < p_class.properties.size(); i++) {
		EXCLUDE_OPTIONS(p_class.properties[i]);
		cd.properties.append(p_class.properties[i]);
	}
	/* Enums */
	/* Unnamed_enums */
	// TODO(require core): currently enums are const dictionary in the core documentation implementation
	//                     contiains the constant to enums map.
	return cd;
}
#undef EXCLUDE_OPTIONS

bool DocumentationGenerationDialog::_generate(const String &p_input, const String &p_target, bool p_recursive_call) {
	static List<String> extensions;
	static String target;
	if (!p_recursive_call) {
		extensions.clear();
		int lang = language_menu->get_selected();
		ScriptServer::get_language(lang)->get_recognized_extensions(&extensions);
		target = ProjectSettings::get_singleton()->localize_path(p_target);
	}

	Ref<GDScript> scr; // TODO(require core): change this to Ref<Script>
	// Ref<Script> scr;
	DirAccess *dir_access = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	if (dir_access->change_dir(p_input) != OK) {
		memdelete(dir_access);
		alert->set_text(vformat("Error opening directory %s", p_input));
		alert->popup_centered();
		return false;
	}
	dir_access->list_dir_begin();
	String next_dir = dir_access->get_next();
	while (next_dir != String()) {
		if (next_dir.begins_with(".")) {
			next_dir = dir_access->get_next();
			continue;
		}
		if (dir_access->current_is_dir()) {
			if (ProjectSettings::get_singleton()->localize_path(p_input + "/" + next_dir) == target) {
				next_dir = dir_access->get_next();
				continue;
			}

			DirAccessRef dir_access_ref = DirAccess::open(p_target);
			if (!dir_access_ref) {
				alert->set_text(vformat("%s \"%s\".", TTR("Error - Cannot open directory at"), p_target));
				alert->popup_centered();
				return false;
			}

			if (!dir_access_ref->dir_exists(next_dir)) {
				Error err = dir_access_ref->make_dir(dir_access_ref->get_current_dir() + "/" + next_dir);
				if (err != OK) {
					alert->set_text(vformat("%s \"%s\".", TTR("Error - Cannot create directory at"), dir_access_ref->get_current_dir()));
					alert->popup_centered();
					return false;
				}
			}

			if (!_generate(p_input + "/" + next_dir, p_target + "/" + next_dir, true)) {
				return false;
			}

		} else {
			bool is_file_extension_valid = false;
			for (List<String>::Element *E = extensions.front(); E; E = E->next()) {
				if (next_dir.ends_with(E->get())) {
					is_file_extension_valid = true;
					break;
				}
			}
			if (is_file_extension_valid) {
				// TODO(require core): change gdscript to script after core merged.
				scr = ResourceLoader::load(dir_access->get_current_dir() + "/" + next_dir);
				if (scr.is_valid() && scr->is_valid()) {
					// TODO(require core): get the class doc from the script, now hardcoded with "Node".
					// Vector<DocData::ClassDoc> docs = scr->get_documentation();
					DocData::ClassDoc cd = _apply_options_filter(EditorHelp::get_doc_data()->class_list["Node"]);
					String source, output_name;

					switch (output_format->get_selected()) {
						case OutputFormats::FMT_XML: {
							XmlWriteStream xws(&source);
							DocData::write_class(cd, xws);
							output_name = next_dir.substr(0, next_dir.rfind(".")) + ".xml";
						} break;
						case OutputFormats::FMT_JSON: {
							source = DocData::json_from_class_doc(cd);
							output_name = next_dir.substr(0, next_dir.rfind(".")) + ".json";
						} break;
					}

					/* Save file */
					Error err;
					FileAccess *file = FileAccess::open(p_target + "/" + output_name, FileAccess::WRITE, &err);
					if (err != OK) {
						alert->set_text(vformat("%s \"%s\".", TTR("Error - Cannot open file at"), output_name));
						alert->popup_centered();
						return false;
					}
					file->store_string(source);
					if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
						memdelete(file);
						alert->set_text(vformat("%s \"%s\".", TTR("Error - Cannot store file at"), output_name));
						alert->popup_centered();
						return false;
					}
					file->close();
					memdelete(file);

				} else {
					if (!ignore_invalid_scripts->is_pressed()) {
						alert->set_text(vformat("%s \"%s\".", TTR("Error - Invalid script at"), dir_access->get_current_dir() + "/" + next_dir));
						alert->popup_centered();
						return false;
					}
				}
			}
		}
		next_dir = dir_access->get_next();
	}
	dir_access->list_dir_end();
	memdelete(dir_access);
	return true;
}

void DocumentationGenerationDialog::_theme_changed() {
	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		String lang = ScriptServer::get_language(i)->get_type();
		Ref<Texture2D> lang_icon = gc->get_theme_icon(lang, "EditorIcons");
		if (lang_icon.is_valid()) {
			language_menu->set_item_icon(i, lang_icon);
		}
	}

	input_dir_button->set_icon(gc->get_theme_icon("Folder", "EditorIcons"));
	target_dir_button->set_icon(gc->get_theme_icon("Folder", "EditorIcons"));
	status_panel->add_theme_style_override("panel", gc->get_theme_stylebox("bg", "Tree"));
	language_not_supported->add_theme_color_override("font_color", gc->get_theme_color("error_color", "Editor"));
}

void DocumentationGenerationDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_theme_changed();
		} break;
	}
}

void DocumentationGenerationDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("config", "input", "target", "options"), &DocumentationGenerationDialog::config, DEFVAL(0));
	ADD_SIGNAL(MethodInfo("documentations_generated"));
}

void DocumentationGenerationDialog::_lang_changed(int p_lang) {
	ScriptLanguage *language = ScriptServer::get_language(p_lang);

	// TODO(require core) to check if the selected language supports documentation.
	// workaround:
	is_language_supported = language->get_type() == "GDScript";
	if (is_language_supported) {
		language_not_supported->set_visible(false);
	} else {
		language_not_supported->set_text("- " + TTR("The selected language isn't support documentation."));
		language_not_supported->set_visible(true);
	}
	_update_dialog();
}

void DocumentationGenerationDialog::_path_entered(const String &p_path, bool p_is_input) {
	ok_pressed();
	_update_dialog();
}

void DocumentationGenerationDialog::_browse_path(bool p_is_input) {
	if (p_is_input) {
		file_browse->set_title(TTR("Choose Input Location"));
		file_browse->set_current_dir(EditorSettings::get_singleton()->get_project_metadata("documentation_generation", "last_source_path", input_dir_path->get_text().strip_edges()));
		is_browsing_input = true;
	} else {
		file_browse->set_title(TTR("Choose Target Location"));
		file_browse->set_current_dir(EditorSettings::get_singleton()->get_project_metadata("documentation_generation", "last_target_path", target_dir_path->get_text().strip_edges()));
		is_browsing_input = false;
	}
	file_browse->popup_file_dialog();
}

void DocumentationGenerationDialog::_dir_selected(const String &p_dir) {
	if (is_browsing_input) {
		input_dir_path->set_text(p_dir);
		_path_changed(p_dir, true);
		EditorSettings::get_singleton()->set_project_metadata("documentation_generation", "last_source_path", p_dir);
	} else {
		target_dir_path->set_text(p_dir);
		_path_changed(p_dir, false);
		EditorSettings::get_singleton()->set_project_metadata("documentation_generation", "last_target_path", p_dir);
	}
}

void DocumentationGenerationDialog::_update_dialog() {
	get_ok()->set_disabled(!(is_input_path_valid && is_target_path_valid && is_language_supported));
}

void DocumentationGenerationDialog::ok_pressed() {
	if (_generate(ProjectSettings::get_singleton()->localize_path(input_dir_path->get_text().strip_edges()), ProjectSettings::get_singleton()->localize_path(target_dir_path->get_text().strip_edges()))) {
		EditorSettings::get_singleton()->set_project_metadata("documentation_generation", "exclude_private", exclude_private->is_pressed());
		EditorSettings::get_singleton()->set_project_metadata("documentation_generation", "documented_only", documented_only->is_pressed());
		EditorSettings::get_singleton()->set_project_metadata("documentation_generation", "ignore_invalid_scripts", ignore_invalid_scripts->is_pressed());
		EditorSettings::get_singleton()->set_project_metadata("documentation_generation", "last_output_format", output_format->get_selected());
		emit_signal("documentations_generated");
		hide();
	}
	_update_dialog();
}

void DocumentationGenerationDialog::config(const String &p_input_dir, const String &p_target_dir, int p_output_format, int p_options) {
	input_error_label->set_text("");
	target_error_label->set_text("");

	if (p_input_dir == "") {
		input_dir_path->set_text(EditorSettings::get_singleton()->get_project_metadata("documentation_generation", "last_source_path", "res://"));
	} else {
		input_dir_path->set_text(p_input_dir);
	}
	input_dir_path->deselect();

	if (p_target_dir == "") {
		target_dir_path->set_text(EditorSettings::get_singleton()->get_project_metadata("documentation_generation", "last_target_path", "res://"));
	} else {
		target_dir_path->set_text(p_target_dir);
	}
	target_dir_path->deselect();

	if (p_output_format == FMT_UNKNOWN) {
		output_format->select(EditorSettings::get_singleton()->get_project_metadata("documentation_generation", "last_output_format", FMT_XML));
	} else {
		output_format->select(p_output_format);
	}

	if (p_options == OPT_UNKNOWN) {
		ignore_invalid_scripts->set_pressed(EditorSettings::get_singleton()->get_project_metadata("documentation_generation", "ignore_invalid_scripts", false));
		exclude_private->set_pressed(EditorSettings::get_singleton()->get_project_metadata("documentation_generation", "exclude_private", false));
		documented_only->set_pressed(EditorSettings::get_singleton()->get_project_metadata("documentation_generation", "documented_only", false));
	} else {
		ignore_invalid_scripts->set_pressed((bool)(p_options & IGNORE_INVALID));
		exclude_private->set_pressed((bool)(p_options & EXCLUDE_PRIVATE));
		documented_only->set_pressed((bool)(p_options & DOCUMENTED_ONLY));
	}

	_lang_changed(language_menu->get_selected());
	_path_changed(input_dir_path->get_text().strip_edges(), true);
	_path_changed(target_dir_path->get_text().strip_edges(), false);
}

DocumentationGenerationDialog::DocumentationGenerationDialog() {
	gc = memnew(GridContainer);
	gc->set_columns(2);
	gc->connect("theme_changed", callable_mp(this, &DocumentationGenerationDialog::_theme_changed));

	VBoxContainer *vb = memnew(VBoxContainer);

	/* Error Messages Field */
	input_error_label = memnew(Label);
	target_error_label = memnew(Label);
	language_not_supported = memnew(Label);
	language_not_supported->set_autowrap(true);
	language_not_supported->hide();
	vb->add_child(input_error_label);
	vb->add_child(target_error_label);
	vb->add_child(language_not_supported);

	/* Status Field */
	status_panel = memnew(PanelContainer);
	status_panel->set_h_size_flags(Control::SIZE_FILL);
	status_panel->add_child(vb);

	/* Spacing */
	Control *spacing = memnew(Control);
	spacing->set_custom_minimum_size(Size2(0, 10 * EDSCALE));

	/* Layout */
	vb = memnew(VBoxContainer);
	vb->add_child(gc);
	vb->add_child(spacing);
	vb->add_child(status_panel);
	HBoxContainer *hb = memnew(HBoxContainer);
	hb->add_child(vb);

	add_child(hb);

	/* Language */
	language_menu = memnew(OptionButton);
	language_menu->set_custom_minimum_size(Size2(250, 0) * EDSCALE);
	language_menu->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	language_menu->connect("item_selected", callable_mp(this, &DocumentationGenerationDialog::_lang_changed));
	gc->add_child(memnew(Label(TTR("Language:"))));
	gc->add_child(language_menu);

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		String lang_name = ScriptServer::get_language(i)->get_name();
		language_menu->add_item(lang_name);
		if (lang_name == "GDScript") {
			default_language = i;
		}
	}
	language_menu->select(default_language);

	/* Input path */
	hb = memnew(HBoxContainer);
	input_dir_path = memnew(LineEdit);
	input_dir_path->connect("text_changed", callable_mp(this, &DocumentationGenerationDialog::_path_changed), varray(true));
	input_dir_path->connect("text_entered", callable_mp(this, &DocumentationGenerationDialog::_path_entered), varray(true));
	input_dir_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb->add_child(input_dir_path);
	input_dir_button = memnew(Button);
	input_dir_button->set_flat(true);
	input_dir_button->connect("pressed", callable_mp(this, &DocumentationGenerationDialog::_browse_path), varray(true));
	hb->add_child(input_dir_button);
	gc->add_child(memnew(Label(TTR("Input Path:"))));
	gc->add_child(hb);

	/* Target path */
	hb = memnew(HBoxContainer);
	target_dir_path = memnew(LineEdit);
	target_dir_path->connect("text_changed", callable_mp(this, &DocumentationGenerationDialog::_path_changed), varray(false));
	target_dir_path->connect("text_entered", callable_mp(this, &DocumentationGenerationDialog::_path_entered), varray(false));
	target_dir_path->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hb->add_child(target_dir_path);
	target_dir_button = memnew(Button);
	target_dir_button->set_flat(true);
	target_dir_button->connect("pressed", callable_mp(this, &DocumentationGenerationDialog::_browse_path), varray(false));
	hb->add_child(target_dir_button);
	gc->add_child(memnew(Label(TTR("Target Path:"))));
	gc->add_child(hb);

	/* Output Format */
	output_format = memnew(OptionButton);
	output_format->set_custom_minimum_size(Size2(250, 0) * EDSCALE);
	output_format->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	gc->add_child(memnew(Label(TTR("Output Format:"))));
	gc->add_child(output_format);
	output_format->add_item("XML");
	output_format->add_item("JSON");

	/* Options */
	ignore_invalid_scripts = memnew(CheckBox);
	ignore_invalid_scripts->set_text(TTR("On"));
	gc->add_child(memnew(Label(TTR("Ignore Invalid Scripts:"))));
	gc->add_child(ignore_invalid_scripts);

	exclude_private = memnew(CheckBox);
	exclude_private->set_text(TTR("On"));
	gc->add_child(memnew(Label(TTR("Exclude Private:"))));
	gc->add_child(exclude_private);

	documented_only = memnew(CheckBox);
	documented_only->set_text(TTR("On"));
	gc->add_child(memnew(Label(TTR("Documented only:"))));
	gc->add_child(documented_only);

	/* Dialog Setup */
	file_browse = memnew(EditorFileDialog);
	file_browse->connect("dir_selected", callable_mp(this, &DocumentationGenerationDialog::_dir_selected));
	file_browse->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_DIR);
	add_child(file_browse);

	alert = memnew(AcceptDialog);
	alert->get_label()->set_autowrap(true);
	alert->get_label()->set_align(Label::ALIGN_CENTER);
	alert->get_label()->set_valign(Label::VALIGN_CENTER);
	alert->get_label()->set_custom_minimum_size(Size2(325, 60) * EDSCALE);
	add_child(alert);

	get_ok()->set_text(TTR("Generate"));
	get_ok()->set_disabled(true);
	set_hide_on_ok(false);
}

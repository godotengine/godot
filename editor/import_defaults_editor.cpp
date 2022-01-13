/*************************************************************************/
/*  import_defaults_editor.cpp                                           */
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

#include "import_defaults_editor.h"

class ImportDefaultsEditorSettings : public Object {
	GDCLASS(ImportDefaultsEditorSettings, Object)
	friend class ImportDefaultsEditor;
	List<PropertyInfo> properties;
	Map<StringName, Variant> values;
	Map<StringName, Variant> default_values;

	Ref<ResourceImporter> importer;

protected:
	bool _set(const StringName &p_name, const Variant &p_value) {
		if (values.has(p_name)) {
			values[p_name] = p_value;
			return true;
		} else {
			return false;
		}
	}
	bool _get(const StringName &p_name, Variant &r_ret) const {
		if (values.has(p_name)) {
			r_ret = values[p_name];
			return true;
		} else {
			r_ret = Variant();
			return false;
		}
	}
	void _get_property_list(List<PropertyInfo> *p_list) const {
		if (importer.is_null()) {
			return;
		}
		for (const List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
			if (importer->get_option_visibility(E->get().name, values)) {
				p_list->push_back(E->get());
			}
		}
	}
};

void ImportDefaultsEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_PREDELETE) {
		if (inspector) {
			inspector->edit(nullptr);
		}
	}
}

void ImportDefaultsEditor::_reset() {
	if (settings->importer.is_valid()) {
		settings->values = settings->default_values;
		settings->_change_notify();
	}
}

void ImportDefaultsEditor::_save() {
	if (settings->importer.is_valid()) {
		Dictionary modified;

		for (Map<StringName, Variant>::Element *E = settings->values.front(); E; E = E->next()) {
			if (E->get() != settings->default_values[E->key()]) {
				modified[E->key()] = E->get();
			}
		}

		if (modified.size()) {
			ProjectSettings::get_singleton()->set("importer_defaults/" + settings->importer->get_importer_name(), modified);
		} else {
			ProjectSettings::get_singleton()->set("importer_defaults/" + settings->importer->get_importer_name(), Variant());
		}

		emit_signal("project_settings_changed");
	}
}

void ImportDefaultsEditor::_update_importer() {
	List<Ref<ResourceImporter>> importer_list;
	ResourceFormatImporter::get_singleton()->get_importers(&importer_list);
	Ref<ResourceImporter> importer;
	for (List<Ref<ResourceImporter>>::Element *E = importer_list.front(); E; E = E->next()) {
		if (E->get()->get_visible_name() == importers->get_item_text(importers->get_selected())) {
			importer = E->get();
			break;
		}
	}

	settings->properties.clear();
	settings->values.clear();
	settings->importer = importer;

	if (importer.is_valid()) {
		List<ResourceImporter::ImportOption> options;
		importer->get_import_options(&options);
		Dictionary d;
		if (ProjectSettings::get_singleton()->has_setting("importer_defaults/" + importer->get_importer_name())) {
			d = ProjectSettings::get_singleton()->get("importer_defaults/" + importer->get_importer_name());
		}

		for (List<ResourceImporter::ImportOption>::Element *E = options.front(); E; E = E->next()) {
			settings->properties.push_back(E->get().option);
			if (d.has(E->get().option.name)) {
				settings->values[E->get().option.name] = d[E->get().option.name];
			} else {
				settings->values[E->get().option.name] = E->get().default_value;
			}
			settings->default_values[E->get().option.name] = E->get().default_value;
		}

		save_defaults->set_disabled(false);
		reset_defaults->set_disabled(false);

	} else {
		save_defaults->set_disabled(true);
		reset_defaults->set_disabled(true);
	}

	settings->_change_notify();

	inspector->edit(settings);
}

void ImportDefaultsEditor::_importer_selected(int p_index) {
	_update_importer();
}

void ImportDefaultsEditor::clear() {
	String last_selected;
	if (importers->get_selected() > 0) {
		last_selected = importers->get_item_text(importers->get_selected());
	}

	importers->clear();

	importers->add_item("<" + TTR("Select Importer") + ">");
	importers->set_item_disabled(0, true);

	List<Ref<ResourceImporter>> importer_list;
	ResourceFormatImporter::get_singleton()->get_importers(&importer_list);
	Vector<String> names;
	for (List<Ref<ResourceImporter>>::Element *E = importer_list.front(); E; E = E->next()) {
		String vn = E->get()->get_visible_name();
		names.push_back(vn);
	}
	names.sort();

	for (int i = 0; i < names.size(); i++) {
		importers->add_item(names[i]);

		if (names[i] == last_selected) {
			importers->select(i + 1);
		}
	}
}

void ImportDefaultsEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_reset"), &ImportDefaultsEditor::_reset);
	ClassDB::bind_method(D_METHOD("_save"), &ImportDefaultsEditor::_save);
	ClassDB::bind_method(D_METHOD("_importer_selected"), &ImportDefaultsEditor::_importer_selected);

	ADD_SIGNAL(MethodInfo("project_settings_changed"));
}

ImportDefaultsEditor::ImportDefaultsEditor() {
	HBoxContainer *hb = memnew(HBoxContainer);
	hb->add_child(memnew(Label(TTR("Importer:"))));
	importers = memnew(OptionButton);
	hb->add_child(importers);
	hb->add_spacer();
	importers->connect("item_selected", this, "_importer_selected");
	reset_defaults = memnew(Button);
	reset_defaults->set_text(TTR("Reset to Defaults"));
	reset_defaults->set_disabled(true);
	reset_defaults->connect("pressed", this, "_reset");
	hb->add_child(reset_defaults);
	add_child(hb);
	inspector = memnew(EditorInspector);
	add_child(inspector);
	inspector->set_v_size_flags(SIZE_EXPAND_FILL);
	CenterContainer *cc = memnew(CenterContainer);
	save_defaults = memnew(Button);
	save_defaults->set_text(TTR("Save"));
	save_defaults->connect("pressed", this, "_save");
	cc->add_child(save_defaults);
	add_child(cc);

	settings = memnew(ImportDefaultsEditorSettings);
}

ImportDefaultsEditor::~ImportDefaultsEditor() {
	memdelete(settings);
}

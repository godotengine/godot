/**************************************************************************/
/*  import_defaults_editor.cpp                                            */
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

#include "import_defaults_editor.h"

#include "core/config/project_settings.h"
#include "core/io/resource_importer.h"
#include "editor/action_map_editor.h"
#include "editor/editor_autoload_settings.h"
#include "editor/editor_sectioned_inspector.h"
#include "editor/localization_editor.h"
#include "editor/plugins/editor_plugin_settings.h"
#include "editor/shader_globals_editor.h"
#include "scene/gui/center_container.h"

class ImportDefaultsEditorSettings : public Object {
	GDCLASS(ImportDefaultsEditorSettings, Object)
	friend class ImportDefaultsEditor;
	List<PropertyInfo> properties;
	HashMap<StringName, Variant> values;
	HashMap<StringName, Variant> default_values;

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
		for (const PropertyInfo &E : properties) {
			if (importer->get_option_visibility("", E.name, values)) {
				p_list->push_back(E);
			}
		}
	}
};

void ImportDefaultsEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PREDELETE: {
			inspector->edit(nullptr);
		} break;
	}
}

void ImportDefaultsEditor::_reset() {
	if (settings->importer.is_valid()) {
		settings->values = settings->default_values;
		settings->notify_property_list_changed();
	}
}

void ImportDefaultsEditor::_save() {
	if (settings->importer.is_valid()) {
		Dictionary modified;

		for (const KeyValue<StringName, Variant> &E : settings->values) {
			if (E.value != settings->default_values[E.key]) {
				modified[E.key] = E.value;
			}
		}

		if (modified.size()) {
			ProjectSettings::get_singleton()->set("importer_defaults/" + settings->importer->get_importer_name(), modified);
		} else {
			ProjectSettings::get_singleton()->set("importer_defaults/" + settings->importer->get_importer_name(), Variant());
		}
		ProjectSettings::get_singleton()->save();
	}
}

void ImportDefaultsEditor::_update_importer() {
	List<Ref<ResourceImporter>> importer_list;
	ResourceFormatImporter::get_singleton()->get_importers(&importer_list);
	Ref<ResourceImporter> importer;
	for (const Ref<ResourceImporter> &E : importer_list) {
		if (E->get_visible_name() == importers->get_item_text(importers->get_selected())) {
			importer = E;
			break;
		}
	}

	settings->properties.clear();
	settings->values.clear();
	settings->importer = importer;

	if (importer.is_valid()) {
		List<ResourceImporter::ImportOption> options;
		importer->get_import_options("", &options);
		Dictionary d;
		if (ProjectSettings::get_singleton()->has_setting("importer_defaults/" + importer->get_importer_name())) {
			d = GLOBAL_GET("importer_defaults/" + importer->get_importer_name());
		}

		for (const ResourceImporter::ImportOption &E : options) {
			settings->properties.push_back(E.option);
			if (d.has(E.option.name)) {
				settings->values[E.option.name] = d[E.option.name];
			} else {
				settings->values[E.option.name] = E.default_value;
			}
			settings->default_values[E.option.name] = E.default_value;
		}

		save_defaults->set_disabled(false);
		reset_defaults->set_disabled(false);

	} else {
		save_defaults->set_disabled(true);
		reset_defaults->set_disabled(true);
	}

	settings->notify_property_list_changed();

	// Set the importer class to fetch the correct class in the XML class reference.
	// This allows tooltips to display when hovering properties.
	inspector->set_object_class(importer->get_class_name());
	inspector->edit(settings);
}

void ImportDefaultsEditor::_importer_selected(int p_index) {
	_update_importer();
}

void ImportDefaultsEditor::clear() {
	String last_selected;

	if (importers->get_selected() >= 0) {
		last_selected = importers->get_item_text(importers->get_selected());
	}

	importers->clear();

	List<Ref<ResourceImporter>> importer_list;
	ResourceFormatImporter::get_singleton()->get_importers(&importer_list);
	Vector<String> names;
	for (const Ref<ResourceImporter> &E : importer_list) {
		String vn = E->get_visible_name();
		names.push_back(vn);
	}
	names.sort();

	// `last_selected.is_empty()` means it's the first time being called.
	if (last_selected.is_empty() && !names.is_empty()) {
		last_selected = names[0];
	}

	for (int i = 0; i < names.size(); i++) {
		importers->add_item(names[i]);

		if (names[i] == last_selected) {
			importers->select(i);
			_update_importer();
		}
	}
}

ImportDefaultsEditor::ImportDefaultsEditor() {
	ProjectSettings::get_singleton()->add_hidden_prefix("importer_defaults/");

	HBoxContainer *hb = memnew(HBoxContainer);
	hb->add_child(memnew(Label(TTR("Importer:"))));
	importers = memnew(OptionButton);
	hb->add_child(importers);
	hb->add_spacer();
	importers->connect("item_selected", callable_mp(this, &ImportDefaultsEditor::_importer_selected));
	reset_defaults = memnew(Button);
	reset_defaults->set_text(TTR("Reset to Defaults"));
	reset_defaults->set_disabled(true);
	reset_defaults->connect(SceneStringName(pressed), callable_mp(this, &ImportDefaultsEditor::_reset));
	hb->add_child(reset_defaults);
	add_child(hb);

	inspector = memnew(EditorInspector);
	add_child(inspector);
	inspector->set_v_size_flags(SIZE_EXPAND_FILL);
	// Make it possible to display tooltips stored in the XML class reference.
	// The object name is set when the importer changes in `_update_importer()`.
	inspector->set_use_doc_hints(true);

	CenterContainer *cc = memnew(CenterContainer);
	save_defaults = memnew(Button);
	save_defaults->set_text(TTR("Save"));
	save_defaults->connect(SceneStringName(pressed), callable_mp(this, &ImportDefaultsEditor::_save));
	cc->add_child(save_defaults);
	add_child(cc);

	settings = memnew(ImportDefaultsEditorSettings);
}

ImportDefaultsEditor::~ImportDefaultsEditor() {
	memdelete(settings);
}

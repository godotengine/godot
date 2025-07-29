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
#include "editor/inspector/editor_inspector.h"
#include "editor/inspector/editor_sectioned_inspector.h"
#include "editor/settings/action_map_editor.h"
#include "scene/gui/center_container.h"
#include "scene/gui/label.h"

// Importer-specific default options (e.g., texture import settings).
class ImporterDefaultSettings : public Object {
	GDCLASS(ImporterDefaultSettings, Object)
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

// File extension default importers.
class ExtensionImporterSettings : public Object {
	GDCLASS(ExtensionImporterSettings, Object)

	friend class ImportDefaultsEditor;

	List<PropertyInfo> properties;
	HashMap<StringName, Variant> values;

public:
	ExtensionImporterSettings() {
		// Add hidden prefix for default resource importers.
		ProjectSettings::get_singleton()->add_hidden_prefix("default_resource_importers/");
	}

private:
	void _refresh_extensions() {
		properties.clear();
		values.clear();

		List<String> all_extensions;
		ResourceFormatImporter::get_singleton()->get_recognized_extensions(&all_extensions);
		all_extensions.sort();

		for (const String &extension : all_extensions) {
			// Retrieve the current default importer (if any).
			Ref<ResourceImporter> default_importer = ResourceFormatImporter::get_singleton()->get_default_importer_by_extension(extension);
			values[extension] = 0; // Initialize a default fallback value.

			// Create a property for this extension.
			PropertyInfo pinfo;
			pinfo.name = extension;
			pinfo.type = Variant::INT;
			pinfo.hint = PropertyHint::PROPERTY_HINT_ENUM;

			// Construct the hint string.
			List<Ref<ResourceImporter>> importers;
			ResourceFormatImporter::get_singleton()->get_importers_for_file("dummy." + extension, &importers);

			// Build the default option text with default importer name.
			String default_text = "Default Importer";
			if (!importers.is_empty()) {
				String auto_default = importers.front()->get()->get_visible_name();
				default_text = vformat(TTR("Default (%s)"), auto_default);
			}
			pinfo.hint_string = default_text;

			int i = 1; // We start at 1 to allow for the empty "Default" choice.
			for (const Ref<ResourceImporter> &importer : importers) {
				// Add the importer to the selectbox.
				pinfo.hint_string += "," + importer->get_visible_name();
				// If that's our valid default importer, set the default value.
				if (default_importer.is_valid() && default_importer->get_importer_name() == importer->get_importer_name()) {
					values[extension] = i;
				}
				++i;
			}

			properties.push_back(pinfo); // Finally add the property to the object.
		}
	}

	bool save_extension_importer(const String &p_extension) const {
		Variant index;
		if (_get(p_extension, index)) {
			Variant value; // Fall back to default (unset the setting).
			if ((int)index > 0) {
				int real_index = (int)index - 1; // Account for the default empty value.
				List<Ref<ResourceImporter>> importers;
				ResourceFormatImporter::get_singleton()->get_importers_for_file("dummy." + p_extension, &importers);
				if (importers.size() > real_index) {
					// If we have a valid importer at this index, that's the one.
					int current = 0;
					for (const Ref<ResourceImporter> &importer : importers) {
						if (current == real_index) {
							value = importer->get_importer_name();
							break;
						}
						current++;
					}
				}
			}

			const String setting_name = "default_resource_importers/" + p_extension;
			ProjectSettings::get_singleton()->set(setting_name, value);
			ProjectSettings::get_singleton()->save();

			return true;
		}

		return false;
	}

protected:
	bool _set(const StringName &p_name, const Variant &p_value) {
		if (values.has(p_name)) {
			values[p_name] = p_value;
			return true;
		}
		return false;
	}

	bool _get(const StringName &p_name, Variant &r_ret) const {
		if (values.has(p_name)) {
			r_ret = values[p_name];
			return true;
		}
		r_ret = Variant();
		return false;
	}

	void _get_property_list(List<PropertyInfo> *p_list) const {
		*p_list = properties;
	}
};

void ImportDefaultsEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PREDELETE: {
			importer_defaults_inspector->edit(nullptr);
			extension_importer_inspector->edit(nullptr);
		} break;
	}
}

void ImportDefaultsEditor::_reset() {
	if (importer_settings->importer.is_valid()) {
		importer_settings->values = importer_settings->default_values;
		importer_settings->notify_property_list_changed();
	}
}

void ImportDefaultsEditor::_save() {
	if (importer_settings->importer.is_valid()) {
		Dictionary modified;

		for (const KeyValue<StringName, Variant> &E : importer_settings->values) {
			if (E.value != importer_settings->default_values[E.key]) {
				modified[E.key] = E.value;
			}
		}

		if (modified.size()) {
			ProjectSettings::get_singleton()->set("importer_defaults/" + importer_settings->importer->get_importer_name(), modified);
		} else {
			ProjectSettings::get_singleton()->set("importer_defaults/" + importer_settings->importer->get_importer_name(), Variant());
		}
		ProjectSettings::get_singleton()->save();
	}
}

void ImportDefaultsEditor::_save_extension_importer(const String &p_property_name) {
	if (extension_settings->save_extension_importer(p_property_name)) {
		emit_signal(SNAME("project_settings_changed"));
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

	importer_settings->properties.clear();
	importer_settings->values.clear();
	importer_settings->importer = importer;

	if (importer.is_valid()) {
		List<ResourceImporter::ImportOption> options;
		importer->get_import_options("", &options);
		Dictionary d;
		if (ProjectSettings::get_singleton()->has_setting("importer_defaults/" + importer->get_importer_name())) {
			d = GLOBAL_GET("importer_defaults/" + importer->get_importer_name());
		}

		for (const ResourceImporter::ImportOption &E : options) {
			importer_settings->properties.push_back(E.option);
			if (d.has(E.option.name)) {
				importer_settings->values[E.option.name] = d[E.option.name];
			} else {
				importer_settings->values[E.option.name] = E.default_value;
			}
			importer_settings->default_values[E.option.name] = E.default_value;
		}

		save_defaults->set_disabled(false);
		reset_defaults->set_disabled(false);

	} else {
		save_defaults->set_disabled(true);
		reset_defaults->set_disabled(true);
	}

	importer_settings->notify_property_list_changed();

	// Set the importer class to fetch the correct class in the XML class reference.
	// This allows tooltips to display when hovering properties.
	importer_defaults_inspector->set_object_class(importer->get_class_name());
	importer_defaults_inspector->edit(importer_settings);
}

void ImportDefaultsEditor::_importer_selected(int p_index) {
	_update_importer();
}

void ImportDefaultsEditor::update_editors() {
	extension_settings->_refresh_extensions();
	extension_importer_inspector->edit(extension_settings);

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
	// Only triggers if anything is selected.
	if (importers->get_selected() == -1) {
		_update_importer();
	}
}

ImportDefaultsEditor::ImportDefaultsEditor() {
	ProjectSettings::get_singleton()->add_hidden_prefix("importer_defaults/");

	// Initialize settings objects.
	extension_settings = memnew(ExtensionImporterSettings);
	importer_settings = memnew(ImporterDefaultSettings);

	tabs = memnew(TabContainer);
	tabs->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(tabs);

	// Importer defaults.
	VBoxContainer *importer_defaults_tab = memnew(VBoxContainer);
	importer_defaults_tab->set_name(TTRC("Importer Defaults"));
	tabs->add_child(importer_defaults_tab);

	HBoxContainer *hb = memnew(HBoxContainer);
	hb->add_child(memnew(Label(TTRC("Importer:"))));
	importers = memnew(OptionButton);
	hb->add_child(importers);
	hb->add_spacer();
	importers->connect(SceneStringName(item_selected), callable_mp(this, &ImportDefaultsEditor::_importer_selected));
	reset_defaults = memnew(Button);
	reset_defaults->set_text(TTRC("Reset to Defaults"));
	reset_defaults->set_disabled(true);
	reset_defaults->connect(SceneStringName(pressed), callable_mp(this, &ImportDefaultsEditor::_reset));
	hb->add_child(reset_defaults);
	importer_defaults_tab->add_child(hb);

	importer_defaults_inspector = memnew(EditorInspector);
	importer_defaults_tab->add_child(importer_defaults_inspector);
	importer_defaults_inspector->set_v_size_flags(SIZE_EXPAND_FILL);
	// Make it possible to display tooltips stored in the XML class reference.
	// The object name is set when the importer changes in `_update_importer()`.
	importer_defaults_inspector->set_use_doc_hints(true);

	// Default Importers by extension (second tab).
	VBoxContainer *default_importers_tab = memnew(VBoxContainer);
	default_importers_tab->set_name(TTRC("Extension Defaults"));
	tabs->add_child(default_importers_tab);

	Label *extension_importers_label = memnew(Label);
	extension_importers_label->set_text(TTRC("Select default importer for each file extension:"));
	default_importers_tab->add_child(extension_importers_label);

	extension_importer_inspector = memnew(EditorInspector);
	extension_importer_inspector->set_v_size_flags(SIZE_EXPAND_FILL);
	extension_importer_inspector->set_property_name_style(EditorPropertyNameProcessor::STYLE_RAW);
	extension_importer_inspector->set_use_settings_name_style(false);
	default_importers_tab->add_child(extension_importer_inspector);
	extension_importer_inspector->connect("property_edited", callable_mp(this, &ImportDefaultsEditor::_save_extension_importer));

	CenterContainer *cc = memnew(CenterContainer);
	save_defaults = memnew(Button);
	save_defaults->set_text(TTRC("Save"));
	save_defaults->connect(SceneStringName(pressed), callable_mp(this, &ImportDefaultsEditor::_save));
	cc->add_child(save_defaults);
	importer_defaults_tab->add_child(cc);
}

ImportDefaultsEditor::~ImportDefaultsEditor() {
	memdelete(importer_settings);
	memdelete(extension_settings);
}

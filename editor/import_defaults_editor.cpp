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
#include "editor/editor_plugin_settings.h"
#include "editor/editor_sectioned_inspector.h"
#include "editor/editor_settings.h"
#include "editor/localization_editor.h"
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

class DefaultImportersEditorSettings : public Object {
	GDCLASS(DefaultImportersEditorSettings, Object)
	friend class ImportDefaultsEditor;
	List<PropertyInfo> properties;
	HashMap<StringName, Variant> values;

	void initialize() {
		properties.clear();
		values.clear();

		List<String> all_extensions;
		ResourceFormatImporter::get_singleton()->get_recognized_extensions(&all_extensions);
		all_extensions.sort();
		for (const String &extension : all_extensions) {
			// Retrieve the current default importer (if any).
			Ref<ResourceImporter> default = ResourceFormatImporter::get_singleton()->get_default_importer_by_extension(extension);
			values[extension] = 0; // Initialize a default fallback value.

			// Create a property for this extension.
			PropertyInfo pinfo;
			pinfo.name = extension;
			pinfo.type = Variant::INT;
			pinfo.hint = PropertyHint::PROPERTY_HINT_ENUM;

			// Construct the hint string.
			pinfo.hint_string = "";
			List<Ref<ResourceImporter>> importers;
			ResourceFormatImporter::get_singleton()->get_importers_for_extension(extension, &importers);
			int i = 1; // We start at 1 to allow for the empty "Default" choice.
			for (const Ref<ResourceImporter> &importer : importers) {
				// Add the importer to the selectbox.
				pinfo.hint_string += "," + importer->get_visible_name();
				// If that's our valid default importer, set the default value.
				if (default.is_valid() && default->get_importer_name() == importer->get_importer_name()) {
					values[extension] = i;//get_value_from_importer(importer);
				}
				++i;
			}

			properties.push_back(pinfo); // Finally add the property to the object.
		}
	}

	bool save_default_importer_for_extension(const String &p_extension) const {
		Variant index;
		print_line(vformat("Attempting so save '%s'", p_extension));
		if (_get(p_extension, index)) {
			Variant value; // Fallback to default (unset the setting).
			if ((int)index > 0) {
				index = (int)index - 1; // Account for the default empty value.
				List<Ref<ResourceImporter>> importers;
				ResourceFormatImporter::get_singleton()->get_importers_for_extension(p_extension, &importers);
				if (importers.size() > (int)index) {
					// If we have a valid importer at this index, that's the one.
					value = importers[(int)index]->get_importer_name();
				}
			}

			const String setting_name = "default_resource_importers/" + p_extension;
			ProjectSettings::get_singleton()->set(setting_name, value);

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
		case NOTIFICATION_ENTER_TREE:
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			importers_inspector->set_property_name_style(EditorPropertyNameProcessor::Style::STYLE_RAW);
			settings_inspector->set_property_name_style(EditorPropertyNameProcessor::get_settings_style());
		} break;

		case NOTIFICATION_PREDELETE: {
			importers_inspector->edit(nullptr);
			settings_inspector->edit(nullptr);
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

		emit_signal(SNAME("project_settings_changed"));
	}
}

void ImportDefaultsEditor::_save_default_importer(const String &p_property_name) {
	if (default_importers->save_default_importer_for_extension(p_property_name)) {
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

		save_default_settings->set_disabled(false);
		reset_default_settings->set_disabled(false);

	} else {
		save_default_settings->set_disabled(true);
		reset_default_settings->set_disabled(true);
	}

	settings->notify_property_list_changed();

	settings_inspector->edit(settings);
}

void ImportDefaultsEditor::_importer_selected(int p_index) {
	_update_importer();
}

void ImportDefaultsEditor::initialize() {
	default_importers->initialize();
	importers_inspector->edit(default_importers);

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
	for (const Ref<ResourceImporter> &E : importer_list) {
		String vn = E->get_visible_name();
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
	ADD_SIGNAL(MethodInfo("project_settings_changed"));
}

ImportDefaultsEditor::ImportDefaultsEditor() {
	Label *top_label = memnew(Label(TTR("Default importers by extension:")));
	top_label->set_theme_type_variation("HeaderSmall");
	add_child(top_label);
	importers_inspector = memnew(EditorInspector);
	importers_inspector->set_v_size_flags(SIZE_EXPAND_FILL);
	importers_inspector->connect("property_edited", callable_mp(this, &ImportDefaultsEditor::_save_default_importer));
	add_child(importers_inspector);

	default_importers = memnew(DefaultImportersEditorSettings);

	HBoxContainer *bottom_hb = memnew(HBoxContainer);
	Label *bottom_label = memnew(Label(TTR("Default settings for:")));
	bottom_label->set_theme_type_variation("HeaderSmall");
	bottom_hb->add_child(bottom_label);
	importers = memnew(OptionButton);
	bottom_hb->add_child(importers);
	bottom_hb->add_spacer();
	importers->connect("item_selected", callable_mp(this, &ImportDefaultsEditor::_importer_selected));
	reset_default_settings = memnew(Button);
	reset_default_settings->set_text(TTR("Reset to Defaults"));
	reset_default_settings->set_disabled(true);
	reset_default_settings->connect("pressed", callable_mp(this, &ImportDefaultsEditor::_reset));
	bottom_hb->add_child(reset_default_settings);
	add_child(bottom_hb);
	settings_inspector = memnew(EditorInspector);
	add_child(settings_inspector);
	settings_inspector->set_v_size_flags(SIZE_EXPAND_FILL);
	CenterContainer *cc = memnew(CenterContainer);
	save_default_settings = memnew(Button);
	save_default_settings->set_text(TTR("Save"));
	save_default_settings->connect("pressed", callable_mp(this, &ImportDefaultsEditor::_save));
	cc->add_child(save_default_settings);
	add_child(cc);

	settings = memnew(ImportDefaultsEditorSettings);
}

ImportDefaultsEditor::~ImportDefaultsEditor() {
	memdelete(default_importers);
	memdelete(settings);
}

/*************************************************************************/
/*  editor_export.cpp                                                    */
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

#include "editor_export.h"

#include "core/config/project_settings.h"
#include "core/io/config_file.h"

EditorExport *EditorExport::singleton = nullptr;

void EditorExport::_save() {
	Ref<ConfigFile> config;
	config.instantiate();
	for (int i = 0; i < export_presets.size(); i++) {
		Ref<EditorExportPreset> preset = export_presets[i];
		String section = "preset." + itos(i);

		config->set_value(section, "name", preset->get_name());
		config->set_value(section, "platform", preset->get_platform()->get_name());
		config->set_value(section, "runnable", preset->is_runnable());
		config->set_value(section, "custom_features", preset->get_custom_features());

		bool save_files = false;
		switch (preset->get_export_filter()) {
			case EditorExportPreset::EXPORT_ALL_RESOURCES: {
				config->set_value(section, "export_filter", "all_resources");
			} break;
			case EditorExportPreset::EXPORT_SELECTED_SCENES: {
				config->set_value(section, "export_filter", "scenes");
				save_files = true;
			} break;
			case EditorExportPreset::EXPORT_SELECTED_RESOURCES: {
				config->set_value(section, "export_filter", "resources");
				save_files = true;
			} break;
			case EditorExportPreset::EXCLUDE_SELECTED_RESOURCES: {
				config->set_value(section, "export_filter", "exclude");
				save_files = true;
			} break;
		}

		if (save_files) {
			Vector<String> export_files = preset->get_files_to_export();
			config->set_value(section, "export_files", export_files);
		}
		config->set_value(section, "include_filter", preset->get_include_filter());
		config->set_value(section, "exclude_filter", preset->get_exclude_filter());
		config->set_value(section, "export_path", preset->get_export_path());
		config->set_value(section, "encryption_include_filters", preset->get_enc_in_filter());
		config->set_value(section, "encryption_exclude_filters", preset->get_enc_ex_filter());
		config->set_value(section, "encrypt_pck", preset->get_enc_pck());
		config->set_value(section, "encrypt_directory", preset->get_enc_directory());
		config->set_value(section, "script_export_mode", preset->get_script_export_mode());
		config->set_value(section, "script_encryption_key", preset->get_script_encryption_key());

		String option_section = "preset." + itos(i) + ".options";

		for (const PropertyInfo &E : preset->get_properties()) {
			config->set_value(option_section, E.name, preset->get(E.name));
		}
	}

	config->save("res://export_presets.cfg");
}

void EditorExport::save_presets() {
	if (block_save) {
		return;
	}
	save_timer->start();
}

void EditorExport::_bind_methods() {
	ADD_SIGNAL(MethodInfo("export_presets_updated"));
}

void EditorExport::add_export_platform(const Ref<EditorExportPlatform> &p_platform) {
	export_platforms.push_back(p_platform);
}

int EditorExport::get_export_platform_count() {
	return export_platforms.size();
}

Ref<EditorExportPlatform> EditorExport::get_export_platform(int p_idx) {
	ERR_FAIL_INDEX_V(p_idx, export_platforms.size(), Ref<EditorExportPlatform>());

	return export_platforms[p_idx];
}

void EditorExport::add_export_preset(const Ref<EditorExportPreset> &p_preset, int p_at_pos) {
	if (p_at_pos < 0) {
		export_presets.push_back(p_preset);
	} else {
		export_presets.insert(p_at_pos, p_preset);
	}
}

String EditorExportPlatform::test_etc2() const {
	const bool etc2_supported = ProjectSettings::get_singleton()->get("rendering/textures/vram_compression/import_etc2");

	if (!etc2_supported) {
		return TTR("Target platform requires 'ETC2' texture compression. Enable 'Import Etc 2' in Project Settings.");
	}

	return String();
}

int EditorExport::get_export_preset_count() const {
	return export_presets.size();
}

Ref<EditorExportPreset> EditorExport::get_export_preset(int p_idx) {
	ERR_FAIL_INDEX_V(p_idx, export_presets.size(), Ref<EditorExportPreset>());
	return export_presets[p_idx];
}

void EditorExport::remove_export_preset(int p_idx) {
	export_presets.remove_at(p_idx);
	save_presets();
}

void EditorExport::add_export_plugin(const Ref<EditorExportPlugin> &p_plugin) {
	if (!export_plugins.has(p_plugin)) {
		export_plugins.push_back(p_plugin);
	}
}

void EditorExport::remove_export_plugin(const Ref<EditorExportPlugin> &p_plugin) {
	export_plugins.erase(p_plugin);
}

Vector<Ref<EditorExportPlugin>> EditorExport::get_export_plugins() {
	return export_plugins;
}

void EditorExport::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			load_config();
		} break;

		case NOTIFICATION_PROCESS: {
			update_export_presets();
		} break;
	}
}

void EditorExport::load_config() {
	Ref<ConfigFile> config;
	config.instantiate();
	Error err = config->load("res://export_presets.cfg");
	if (err != OK) {
		return;
	}

	block_save = true;

	int index = 0;
	while (true) {
		String section = "preset." + itos(index);
		if (!config->has_section(section)) {
			break;
		}

		String platform = config->get_value(section, "platform");

		Ref<EditorExportPreset> preset;

		for (int i = 0; i < export_platforms.size(); i++) {
			if (export_platforms[i]->get_name() == platform) {
				preset = export_platforms.write[i]->create_preset();
				break;
			}
		}

		if (!preset.is_valid()) {
			index++;
			ERR_CONTINUE(!preset.is_valid());
		}

		preset->set_name(config->get_value(section, "name"));
		preset->set_runnable(config->get_value(section, "runnable"));

		if (config->has_section_key(section, "custom_features")) {
			preset->set_custom_features(config->get_value(section, "custom_features"));
		}

		String export_filter = config->get_value(section, "export_filter");

		bool get_files = false;

		if (export_filter == "all_resources") {
			preset->set_export_filter(EditorExportPreset::EXPORT_ALL_RESOURCES);
		} else if (export_filter == "scenes") {
			preset->set_export_filter(EditorExportPreset::EXPORT_SELECTED_SCENES);
			get_files = true;
		} else if (export_filter == "resources") {
			preset->set_export_filter(EditorExportPreset::EXPORT_SELECTED_RESOURCES);
			get_files = true;
		} else if (export_filter == "exclude") {
			preset->set_export_filter(EditorExportPreset::EXCLUDE_SELECTED_RESOURCES);
			get_files = true;
		}

		if (get_files) {
			Vector<String> files = config->get_value(section, "export_files");

			for (int i = 0; i < files.size(); i++) {
				if (!FileAccess::exists(files[i])) {
					preset->remove_export_file(files[i]);
				} else {
					preset->add_export_file(files[i]);
				}
			}
		}

		preset->set_include_filter(config->get_value(section, "include_filter"));
		preset->set_exclude_filter(config->get_value(section, "exclude_filter"));
		preset->set_export_path(config->get_value(section, "export_path", ""));

		if (config->has_section_key(section, "encrypt_pck")) {
			preset->set_enc_pck(config->get_value(section, "encrypt_pck"));
		}
		if (config->has_section_key(section, "encrypt_directory")) {
			preset->set_enc_directory(config->get_value(section, "encrypt_directory"));
		}
		if (config->has_section_key(section, "encryption_include_filters")) {
			preset->set_enc_in_filter(config->get_value(section, "encryption_include_filters"));
		}
		if (config->has_section_key(section, "encryption_exclude_filters")) {
			preset->set_enc_ex_filter(config->get_value(section, "encryption_exclude_filters"));
		}
		if (config->has_section_key(section, "script_export_mode")) {
			preset->set_script_export_mode(config->get_value(section, "script_export_mode"));
		}
		if (config->has_section_key(section, "script_encryption_key")) {
			preset->set_script_encryption_key(config->get_value(section, "script_encryption_key"));
		}

		String option_section = "preset." + itos(index) + ".options";

		List<String> options;

		config->get_section_keys(option_section, &options);

		for (const String &E : options) {
			Variant value = config->get_value(option_section, E);

			preset->set(E, value);
		}

		add_export_preset(preset);
		index++;
	}

	block_save = false;
}

void EditorExport::update_export_presets() {
	HashMap<StringName, List<EditorExportPlatform::ExportOption>> platform_options;

	for (int i = 0; i < export_platforms.size(); i++) {
		Ref<EditorExportPlatform> platform = export_platforms[i];

		if (platform->should_update_export_options()) {
			List<EditorExportPlatform::ExportOption> options;
			platform->get_export_options(&options);

			platform_options[platform->get_name()] = options;
		}
	}

	bool export_presets_updated = false;
	for (int i = 0; i < export_presets.size(); i++) {
		Ref<EditorExportPreset> preset = export_presets[i];
		if (platform_options.has(preset->get_platform()->get_name())) {
			export_presets_updated = true;

			List<EditorExportPlatform::ExportOption> options = platform_options[preset->get_platform()->get_name()];

			// Copy the previous preset values
			HashMap<StringName, Variant> previous_values = preset->values;

			// Clear the preset properties and values prior to reloading
			preset->properties.clear();
			preset->values.clear();
			preset->update_visibility.clear();

			for (const EditorExportPlatform::ExportOption &E : options) {
				preset->properties.push_back(E.option);

				StringName option_name = E.option.name;
				preset->values[option_name] = previous_values.has(option_name) ? previous_values[option_name] : E.default_value;
				preset->update_visibility[option_name] = E.update_visibility;
			}
		}
	}

	if (export_presets_updated) {
		emit_signal(_export_presets_updated);
	}
}

bool EditorExport::poll_export_platforms() {
	bool changed = false;
	for (int i = 0; i < export_platforms.size(); i++) {
		if (export_platforms.write[i]->poll_export()) {
			changed = true;
		}
	}

	return changed;
}

EditorExport::EditorExport() {
	save_timer = memnew(Timer);
	add_child(save_timer);
	save_timer->set_wait_time(0.8);
	save_timer->set_one_shot(true);
	save_timer->connect("timeout", callable_mp(this, &EditorExport::_save));

	_export_presets_updated = "export_presets_updated";

	singleton = this;
	set_process(true);

	GLOBAL_DEF("editor/export/convert_text_resources_to_binary", true);
}

EditorExport::~EditorExport() {
}

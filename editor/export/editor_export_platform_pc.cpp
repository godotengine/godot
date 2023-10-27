/**************************************************************************/
/*  editor_export_platform_pc.cpp                                         */
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

#include "editor_export_platform_pc.h"

#include "core/config/project_settings.h"
#include "scene/resources/image_texture.h"

void EditorExportPlatformPC::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) const {
	if (p_preset->get("texture_format/bptc")) {
		r_features->push_back("bptc");
	}
	if (p_preset->get("texture_format/s3tc")) {
		r_features->push_back("s3tc");
	}
	if (p_preset->get("texture_format/etc")) {
		r_features->push_back("etc");
	}
	if (p_preset->get("texture_format/etc2")) {
		r_features->push_back("etc2");
	}
	// PC platforms only have one architecture per export, since
	// we export a single executable instead of a bundle.
	r_features->push_back(p_preset->get("binary_format/architecture"));
}

void EditorExportPlatformPC::get_export_options(List<ExportOption> *r_options) const {
	String ext_filter = (get_os_name() == "Windows") ? "*.exe" : "";
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/debug", PROPERTY_HINT_GLOBAL_FILE, ext_filter), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/release", PROPERTY_HINT_GLOBAL_FILE, ext_filter), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "debug/export_console_wrapper", PROPERTY_HINT_ENUM, "No,Debug Only,Debug and Release"), 1));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "binary_format/embed_pck"), false));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/bptc"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/s3tc"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc2"), false));
}

String EditorExportPlatformPC::get_name() const {
	return name;
}

String EditorExportPlatformPC::get_os_name() const {
	return os_name;
}

Ref<Texture2D> EditorExportPlatformPC::get_logo() const {
	return logo;
}

bool EditorExportPlatformPC::has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates, bool p_debug) const {
	String err;
	bool valid = false;

	// Look for export templates (first official, and if defined custom templates).
	String arch = p_preset->get("binary_format/architecture");
	bool dvalid = exists_export_template(get_template_file_name("debug", arch), &err);
	bool rvalid = exists_export_template(get_template_file_name("release", arch), &err);

	if (p_preset->get("custom_template/debug") != "") {
		dvalid = FileAccess::exists(p_preset->get("custom_template/debug"));
		if (!dvalid) {
			err += TTR("Custom debug template not found.") + "\n";
		}
	}
	if (p_preset->get("custom_template/release") != "") {
		rvalid = FileAccess::exists(p_preset->get("custom_template/release"));
		if (!rvalid) {
			err += TTR("Custom release template not found.") + "\n";
		}
	}

	valid = dvalid || rvalid;
	r_missing_templates = !valid;

	if (!err.is_empty()) {
		r_error = err;
	}
	return valid;
}

bool EditorExportPlatformPC::has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error) const {
	return true;
}

Error EditorExportPlatformPC::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

	Error err = prepare_template(p_preset, p_debug, p_path, p_flags);
	if (err == OK) {
		err = modify_template(p_preset, p_debug, p_path, p_flags);
	}
	if (err == OK) {
		err = export_project_data(p_preset, p_debug, p_path, p_flags);
	}

	return err;
}

Error EditorExportPlatformPC::prepare_template(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	if (!DirAccess::exists(p_path.get_base_dir())) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Prepare Template"), TTR("The given export path doesn't exist."));
		return ERR_FILE_BAD_PATH;
	}

	String custom_debug = p_preset->get("custom_template/debug");
	String custom_release = p_preset->get("custom_template/release");

	String template_path = p_debug ? custom_debug : custom_release;

	template_path = template_path.strip_edges();

	if (template_path.is_empty()) {
		template_path = find_export_template(get_template_file_name(p_debug ? "debug" : "release", p_preset->get("binary_format/architecture")));
	}

	if (!template_path.is_empty() && !FileAccess::exists(template_path)) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Prepare Template"), vformat(TTR("Template file not found: \"%s\"."), template_path));
		return ERR_FILE_NOT_FOUND;
	}

	String wrapper_template_path = template_path.get_basename() + "_console.exe";
	int con_wrapper_mode = p_preset->get("debug/export_console_wrapper");
	bool copy_wrapper = (con_wrapper_mode == 1 && p_debug) || (con_wrapper_mode == 2);

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	da->make_dir_recursive(p_path.get_base_dir());
	Error err = da->copy(template_path, p_path, get_chmod_flags());
	if (err == OK && copy_wrapper && FileAccess::exists(wrapper_template_path)) {
		err = da->copy(wrapper_template_path, p_path.get_basename() + ".console.exe", get_chmod_flags());
	}
	if (err != OK) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Prepare Template"), TTR("Failed to copy export template."));
	}

	return err;
}

Error EditorExportPlatformPC::export_project_data(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	String pck_path;
	if (p_preset->get("binary_format/embed_pck")) {
		pck_path = p_path;
	} else {
		pck_path = p_path.get_basename() + ".pck";
	}

	Vector<SharedObject> so_files;

	int64_t embedded_pos;
	int64_t embedded_size;
	Error err = save_pack(p_preset, p_debug, pck_path, &so_files, p_preset->get("binary_format/embed_pck"), &embedded_pos, &embedded_size);
	if (err == OK && p_preset->get("binary_format/embed_pck")) {
		if (embedded_size >= 0x100000000 && String(p_preset->get("binary_format/architecture")).contains("32")) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("PCK Embedding"), TTR("On 32-bit exports the embedded PCK cannot be bigger than 4 GiB."));
			return ERR_INVALID_PARAMETER;
		}

		err = fixup_embedded_pck(p_path, embedded_pos, embedded_size);
	}

	if (err == OK && !so_files.is_empty()) {
		// If shared object files, copy them.
		Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		for (int i = 0; i < so_files.size() && err == OK; i++) {
			String src_path = ProjectSettings::get_singleton()->globalize_path(so_files[i].path);
			String target_path;
			if (so_files[i].target.is_empty()) {
				target_path = p_path.get_base_dir();
			} else {
				target_path = p_path.get_base_dir().path_join(so_files[i].target);
				da->make_dir_recursive(target_path);
			}
			target_path = target_path.path_join(src_path.get_file());

			if (da->dir_exists(src_path)) {
				err = da->make_dir_recursive(target_path);
				if (err == OK) {
					err = da->copy_dir(src_path, target_path, -1, true);
				}
			} else {
				err = da->copy(src_path, target_path);
				if (err == OK) {
					err = sign_shared_object(p_preset, p_debug, target_path);
				}
			}
		}
	}

	return err;
}

Error EditorExportPlatformPC::sign_shared_object(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path) {
	return OK;
}

void EditorExportPlatformPC::set_name(const String &p_name) {
	name = p_name;
}

void EditorExportPlatformPC::set_os_name(const String &p_name) {
	os_name = p_name;
}

void EditorExportPlatformPC::set_logo(const Ref<Texture2D> &p_logo) {
	logo = p_logo;
}

void EditorExportPlatformPC::get_platform_features(List<String> *r_features) const {
	r_features->push_back("pc"); //all pcs support "pc"
	r_features->push_back("s3tc"); //all pcs support "s3tc" compression
	r_features->push_back(get_os_name().to_lower()); //OS name is a feature
}

void EditorExportPlatformPC::resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, HashSet<String> &p_features) {
}

int EditorExportPlatformPC::get_chmod_flags() const {
	return chmod_flags;
}

void EditorExportPlatformPC::set_chmod_flags(int p_flags) {
	chmod_flags = p_flags;
}

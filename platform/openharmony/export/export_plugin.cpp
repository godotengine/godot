/**************************************************************************/
/*  export_plugin.cpp                                                     */
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

#include "export_plugin.h"

#include "logo_svg.gen.h"
#include "run_icon_svg.gen.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/image.h"
#include "core/io/json.h"
#include "core/io/marshalls.h"
#include "core/io/zip_io.h"
#include "core/templates/safe_refcount.h"
#include "core/version.h"
#include "editor/editor_node.h"
#include "editor/export/editor_export.h"
#include "editor/file_system/editor_paths.h"
#include "editor/import/resource_importer_texture.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "main/splash.gen.h"
#include "scene/resources/image_texture.h"

#include "modules/modules_enabled.gen.h" // For regex.
#include "modules/svg/image_loader_svg.h"
#ifdef MODULE_REGEX_ENABLED
#include "modules/regex/regex.h"
#endif

#include <string.h>

// OpenHarmony permissions
static const char *OPENHARMONY_PERMISSIONS[] = {
	"ohos.permission.INTERNET",
	"ohos.permission.MICROPHONE",
	nullptr
};

// OpenHarmony user permissions
static const char *OPENHARMONY_USER_PERMISSIONS[] = {
	"ohos.permission.MICROPHONE",
	nullptr
};

static const char *OPENHARMONY_DEFAULT_SDK_VERSION = "5.1.0(18)";
static const char *OPENHARMONY_DEFAULT_BUNDLE_ID = "org.godotengine.template";
static const char *OPENHARMONY_ORIENTATION_ENUMS = "landscape,landscape_inverted,auto_rotation_landscape,auto_rotation_landscape_restricted,portrait,portrait_inverted,auto_rotation_portrait,auto_rotation_portrait_restricted,auto_rotation_unspecified,auto_rotation_restricted,follow_recent,follow_desktop";

void EditorExportPlatformOpenHarmony::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) const {
	r_features->push_back("etc2");
	r_features->push_back("astc");
	if (p_preset->get("architectures/arm64-v8a")) {
		r_features->push_back("arm64");
	} else if (p_preset->get("architectures/x86_64")) {
		r_features->push_back("x86_64");
	}
}

void EditorExportPlatformOpenHarmony::get_export_options(List<ExportOption> *r_options) const {
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/debug", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/release", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, vformat("%s/%s", PNAME("architectures"), "arm64")), true, true, true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, vformat("%s/%s", PNAME("architectures"), "x86_64")), false, true, true));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "build/export_project_only"), false, true, true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "build/override_project_dir"), false, true, true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "build/sdk_version", PROPERTY_HINT_PLACEHOLDER_TEXT, vformat("%s (default)", OPENHARMONY_DEFAULT_SDK_VERSION)), "", false, true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "build/bundle_id", PROPERTY_HINT_PLACEHOLDER_TEXT, vformat("%s (default)", OPENHARMONY_DEFAULT_BUNDLE_ID)), "", false, true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "build/default_orientation", PROPERTY_HINT_ENUM, OPENHARMONY_ORIENTATION_ENUMS), 0, true, true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "build/background_image", PROPERTY_HINT_GLOBAL_FILE, "*.png"), "", false, false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "build/foreground_image", PROPERTY_HINT_GLOBAL_FILE, "*.png"), "", false, false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "build/sign"), false, true, true));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "sign/store_file", PROPERTY_HINT_GLOBAL_FILE, "*.p12", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_SECRET), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "sign/store_password", PROPERTY_HINT_PASSWORD, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_SECRET), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "sign/key_alias", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_SECRET), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "sign/key_password", PROPERTY_HINT_PASSWORD, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_SECRET), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "sign/sign_alg", PROPERTY_HINT_NONE, "SHA256withECDSA", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_SECRET), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "sign/profile_file", PROPERTY_HINT_GLOBAL_FILE, "*.p7b", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_SECRET), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "sign/certpath_file", PROPERTY_HINT_GLOBAL_FILE, "*.cer", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_SECRET), ""));

	const char **perms = OPENHARMONY_PERMISSIONS;
	while (*perms) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, vformat("%s/%s", PNAME("permissions"), String(*perms))), false));
		perms++;
	}
}

bool EditorExportPlatformOpenHarmony::get_export_option_visibility(const EditorExportPreset *p_preset, const String &p_option) const {
	if (p_preset == nullptr) {
		return true;
	}

	bool advanced_options_enabled = p_preset->are_advanced_options_enabled();

	// Hide custom template options unless advanced options are enabled
	if (p_option == "custom_template/debug" || p_option == "custom_template/release") {
		return advanced_options_enabled;
	}

	// Hide architecture options unless advanced options are enabled
	if (p_option.begins_with("architectures/")) {
		return advanced_options_enabled;
	}

	// Hide sign options unless build/sign is enabled
	bool sign_enabled = p_preset->get("build/sign");
	if (p_option.begins_with("sign/")) {
		return sign_enabled;
	}

	return true;
}

String EditorExportPlatformOpenHarmony::get_export_option_warning(const EditorExportPreset *p_preset, const StringName &p_name) const {
	if (p_preset == nullptr) {
		return String();
	}

	// Check architecture selection - only one should be selected
	if (String(p_name).begins_with("architectures/")) {
		bool arm64_selected = p_preset->get("architectures/arm64");
		bool x86_64_selected = p_preset->get("architectures/x86_64");

		int selected_count = 0;
		if (arm64_selected) {
			selected_count++;
		}
		if (x86_64_selected) {
			selected_count++;
		}

		if (selected_count == 0) {
			return TTR("At least one architecture must be selected.");
		} else if (selected_count > 1) {
			return TTR("Only one architecture can be selected at a time.");
		}
	}

	// Check sign options when build/sign is enabled
	bool sign_enabled = p_preset->get("build/sign");
	if (sign_enabled && String(p_name).begins_with("sign/")) {
		String value = p_preset->get(p_name);
		if (value.is_empty()) {
			if (p_name == "sign/store_file") {
				return TTR("Store file path is required when signing is enabled.");
			} else if (p_name == "sign/store_password") {
				return TTR("Store password is required when signing is enabled.");
			} else if (p_name == "sign/key_alias") {
				return TTR("Key alias is required when signing is enabled.");
			} else if (p_name == "sign/key_password") {
				return TTR("Key password is required when signing is enabled.");
			} else if (p_name == "sign/sign_alg") {
				return TTR("Sign algorithm is required when signing is enabled.");
			} else if (p_name == "sign/profile_file") {
				return TTR("Profile file path is required when signing is enabled.");
			} else if (p_name == "sign/certpath_file") {
				return TTR("Certificate path file is required when signing is enabled.");
			}
		}
	}

	return String();
}

String EditorExportPlatformOpenHarmony::get_name() const {
	return "OpenHarmony";
}

String EditorExportPlatformOpenHarmony::get_os_name() const {
	return "OpenHarmony";
}

Ref<Texture2D> EditorExportPlatformOpenHarmony::get_logo() const {
	return logo;
}

Ref<Texture2D> EditorExportPlatformOpenHarmony::get_run_icon() const {
	return run_icon;
}

bool EditorExportPlatformOpenHarmony::poll_export() {
	bool dc = devices_changed.is_set();
	if (dc) {
		// don't clear unless we're reporting true, to avoid race
		devices_changed.clear();
	}
	return dc;
}

int EditorExportPlatformOpenHarmony::get_options_count() const {
	MutexLock lock(device_lock);
	return devices.size();
}

String EditorExportPlatformOpenHarmony::get_options_tooltip() const {
	return TTR("Select device from the list");
}

String EditorExportPlatformOpenHarmony::get_option_label(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, devices.size(), "");
	MutexLock lock(device_lock);
	return devices[p_index];
}

String EditorExportPlatformOpenHarmony::get_option_tooltip(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, devices.size(), "");
	MutexLock lock(device_lock);
	return "Device ID: " + devices[p_index];
}

String EditorExportPlatformOpenHarmony::get_device_architecture(int p_index) const {
	// Only arm64 is supported for now.
	return "arm64";
}

List<String> EditorExportPlatformOpenHarmony::get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const {
	List<String> list;
	list.push_back("hap");
	list.push_back("app");
	return list;
}

Error EditorExportPlatformOpenHarmony::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	bool should_sign = p_preset->get("build/sign");
	bool export_project_only = p_preset->get("build/export_project_only");
	return export_project_helper(p_preset, p_debug, p_path, should_sign, export_project_only, p_flags);
}

Error EditorExportPlatformOpenHarmony::export_project_helper(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, bool should_sign, bool export_project_only, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

	EditorProgress ep("export", TTR("Exporting OpenHarmony Project"), 7, true);

	bool has_sign = p_preset->get("build/sign");
	if (should_sign && !has_sign) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Code Signing"), TTR("Signing is not enabled in the export preset."));
		return ERR_CANT_CREATE;
	}

	if (ep.step(TTR("Preparing templates..."), 0)) {
		return ERR_SKIP;
	}

	String custom_debug = p_preset->get("custom_template/debug");
	String custom_release = p_preset->get("custom_template/release");
	String template_path = p_debug ? custom_debug : custom_release;
	template_path = template_path.strip_edges();

	if (template_path.is_empty()) {
		String template_file_name = p_debug ? "openharmony_debug_arm64-v8a.zip" : "openharmony_release_arm64-v8a.zip";
		String err;
		template_path = find_export_template(template_file_name, &err);
		if (template_path.is_empty()) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Prepare Templates"), TTR("Export template not found.") + "\n" + err);
			return ERR_FILE_NOT_FOUND;
		}
	}

	if (!FileAccess::exists(template_path)) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Prepare Templates"), vformat(TTR("Template file not found: \"%s\"."), template_path));
		return ERR_FILE_NOT_FOUND;
	}

	if (ep.step(TTR("Creating project directory..."), 1)) {
		return ERR_SKIP;
	}

	String base_dir = p_path.get_base_dir();

	if (base_dir.is_relative_path()) {
		base_dir = OS::get_singleton()->get_resource_dir().path_join(base_dir);
	}
	base_dir = ProjectSettings::get_singleton()->globalize_path(base_dir).simplify_path();
	String project_name = p_path.get_file().get_basename();
	String project_dir = base_dir.path_join(project_name);
	String file_ext = p_path.get_extension();

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (!da->dir_exists(base_dir)) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Target folder does not exist or is inaccessible: \"%s\""), base_dir));
		return ERR_FILE_BAD_PATH;
	}

	if (da->dir_exists(project_dir)) {
		bool override_project = p_preset->get("build/override_project_dir");
		if (override_project) {
			Error err = da->change_dir(project_dir);
			if (err == OK) {
				da->erase_contents_recursive();
				da->change_dir("..");
				da->remove(project_name);
			}
		} else {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Project dir is already exists (Enable \"Override Project Dir\" to force override): \"%s\"."), project_dir));
			return ERR_ALREADY_EXISTS;
		}
	}

	Error err = da->make_dir_recursive(project_dir);
	if (err != OK) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not create project directory: \"%s\""), project_dir));
		return err;
	}

	if (ep.step(TTR("Extracting template files..."), 2)) {
		return ERR_SKIP;
	}

	Ref<FileAccess> io_fa;
	zlib_filefunc_def io = zipio_create_io(&io_fa);
	unzFile pkg = unzOpen2(template_path.utf8().get_data(), &io);
	if (!pkg) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not open template for export: \"%s\"."), template_path));
		return ERR_FILE_NOT_FOUND;
	}

	int ret = unzGoToFirstFile(pkg);
	while (ret == UNZ_OK) {
		unz_file_info info;
		char filename[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, filename, 16384, nullptr, 0, nullptr, 0);
		if (ret != UNZ_OK) {
			break;
		}

		String file = String::utf8(filename);
		String full_path = project_dir.path_join(file);

		if (file.ends_with("/")) {
			da->make_dir_recursive(full_path);
		} else {
			da->make_dir_recursive(full_path.get_base_dir());

			ret = unzOpenCurrentFile(pkg);
			if (ret == UNZ_OK) {
				Ref<FileAccess> f = FileAccess::open(full_path, FileAccess::WRITE);
				if (f.is_valid()) {
					const int buffer_size = 65536;
					uint8_t buffer[buffer_size];

					while (true) {
						int bytes_read = unzReadCurrentFile(pkg, buffer, buffer_size);
						if (bytes_read == 0) {
							break;
						}
						if (bytes_read < 0) {
							add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not read from template: \"%s\"."), template_path));
							unzCloseCurrentFile(pkg);
							unzClose(pkg);
							return ERR_FILE_CORRUPT;
						}
						f->store_buffer(buffer, bytes_read);
					}
				} else {
					add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write file: \"%s\"."), full_path));
					unzCloseCurrentFile(pkg);
					unzClose(pkg);
					return ERR_FILE_CANT_WRITE;
				}
				unzCloseCurrentFile(pkg);
			}
		}

		ret = unzGoToNextFile(pkg);
	}
	unzClose(pkg);

	if (ep.step(TTR("Configuring project files..."), 3)) {
		return ERR_SKIP;
	}

	Vector<String> command_line_flags = gen_export_flags(p_flags);
	String cl_file_path = project_dir.path_join("entry/src/main/resources/rawfile/_cl_");
	da->make_dir_recursive(cl_file_path.get_base_dir());

	Ref<FileAccess> cl_file = FileAccess::open(cl_file_path, FileAccess::WRITE);
	if (cl_file.is_null()) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write command line file: \"%s\"."), cl_file_path));
		return ERR_FILE_CANT_WRITE;
	}

	for (const String &flag : command_line_flags) {
		CharString cs = (flag + "\n").utf8();
		cl_file->store_buffer((const uint8_t *)cs.get_data(), cs.length());
	}
	cl_file.unref();

	String bundle_id = p_preset->get("build/bundle_id");
	if (bundle_id.is_empty()) {
		bundle_id = OPENHARMONY_DEFAULT_BUNDLE_ID;
	}
	String app_json_path = project_dir.path_join("AppScope/app.json5");
	if (FileAccess::exists(app_json_path)) {
		Ref<FileAccess> app_json_file = FileAccess::open(app_json_path, FileAccess::READ);
		if (app_json_file.is_valid()) {
			String content = app_json_file->get_as_text();
			content = content.replace(OPENHARMONY_DEFAULT_BUNDLE_ID, bundle_id);
			app_json_file = FileAccess::open(app_json_path, FileAccess::WRITE);
			if (app_json_file.is_valid()) {
				app_json_file->store_string(content);
			} else {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write app json: \"%s\"."), app_json_path));
				return ERR_FILE_CANT_WRITE;
			}
		} else {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not read app json: \"%s\"."), app_json_path));
			return ERR_FILE_CANT_READ;
		}
	}

	String app_name = GLOBAL_GET("application/config/name");
	if (app_name.is_empty()) {
		app_name = "template";
	}
	String string_json_path = project_dir.path_join("entry/src/main/resources/base/element/string.json");
	if (FileAccess::exists(string_json_path)) {
		Ref<FileAccess> string_json_file = FileAccess::open(string_json_path, FileAccess::READ);
		if (string_json_file.is_valid()) {
			String content = string_json_file->get_as_text();
			String key = "\"EntryAbility_label\"";
			int pos = content.find(key);
			if (pos >= 0) {
				String value = "\"label\"";
				pos = content.find(value, pos + key.length());
				if (pos >= 0) {
					content = content.left(pos) + "\"" + app_name + "\"" + content.right(content.length() - pos - value.length());
				}
			}

			key = "\"user_permissions\"";
			pos = content.find(key);
			if (pos >= 0) {
				String value = "\"\"";
				pos = content.find(value, pos + key.length());
				if (pos >= 0) {
					const char **perms = OPENHARMONY_USER_PERMISSIONS;
					String user_permissions;
					while (*perms) {
						String perm_name = String(*perms);
						String perm_option = vformat("%s/%s", PNAME("permissions"), perm_name);
						bool perm_enabled = p_preset->get(perm_option);
						if (perm_enabled) {
							if (user_permissions != "") {
								user_permissions += ",";
							}
							user_permissions += perm_name;
						}
						perms++;
					}
					content = content.left(pos) + "\"" + user_permissions + "\"" + content.right(content.length() - pos - value.length());
				}
			}

			string_json_file = FileAccess::open(string_json_path, FileAccess::WRITE);
			if (string_json_file.is_valid()) {
				string_json_file->store_string(content);
			} else {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write string json: \"%s\"."), string_json_path));
				return ERR_FILE_CANT_WRITE;
			}
		} else {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not read string json: \"%s\"."), string_json_path));
			return ERR_FILE_CANT_READ;
		}
	}

	String sdk_version = p_preset->get("build/sdk_version");
	if (sdk_version.is_empty()) {
		sdk_version = OPENHARMONY_DEFAULT_SDK_VERSION;
	}

	String build_profile_path = project_dir.path_join("build-profile.json5");
	if (FileAccess::exists(build_profile_path)) {
		Ref<FileAccess> build_file = FileAccess::open(build_profile_path, FileAccess::READ);
		if (build_file.is_valid()) {
			String content = build_file->get_as_text();
			if (should_sign) {
				String certpath_file = p_preset->get("sign/certpath_file");
				String key_alias = p_preset->get("sign/key_alias");
				String key_password = p_preset->get("sign/key_password");
				String profile_file = p_preset->get("sign/profile_file");
				String sign_alg = p_preset->get("sign/sign_alg");
				String store_file = p_preset->get("sign/store_file");
				String store_password = p_preset->get("sign/store_password");
				content = content.replace("\"signingConfigs\": [],", String("\"signingConfigs\": [\n") + "      {\n" + "        \"name\": \"default\",\n" + "        \"type\": \"HarmonyOS\",\n" + "        \"material\": {\n" + "          \"certpath\": \"" + certpath_file + "\",\n" + "          \"keyAlias\": \"" + key_alias + "\",\n" + "          \"keyPassword\": \"" + key_password + "\",\n" + "          \"profile\": \"" + profile_file + "\",\n" + "          \"signAlg\": \"" + sign_alg + "\",\n" + "          \"storeFile\": \"" + store_file + "\",\n" + "          \"storePassword\": \"" + store_password + "\"\n" + "        }\n" + "      }\n" + "    ],");
			}

			content = content.replace("\"targetSdkVersion\": \"5.1.0(18)\"", "\"targetSdkVersion\": \"" + sdk_version + "\"");
			content = content.replace("\"compatibleSdkVersion\": \"5.1.0(18)\"", "\"compatibleSdkVersion\": \"" + sdk_version + "\"");

			build_file = FileAccess::open(build_profile_path, FileAccess::WRITE);
			if (build_file.is_valid()) {
				build_file->store_string(content);
			} else {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write build profile: \"%s\"."), build_profile_path));
				return ERR_FILE_CANT_WRITE;
			}
		} else {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not read build profile: \"%s\"."), build_profile_path));
			return ERR_FILE_CANT_READ;
		}
	}

	String background_image = p_preset->get("build/background_image");
	if (!background_image.is_empty() && FileAccess::exists(background_image)) {
		String dest_bg_path = project_dir.path_join("entry/src/main/resources/base/media/background.png");
		da->make_dir_recursive(dest_bg_path.get_base_dir());
		err = da->copy(background_image, dest_bg_path);
		if (err != OK) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not copy background image: \"%s\" to \"%s\"."), background_image, dest_bg_path));
			return err;
		}
		dest_bg_path = project_dir.path_join("AppScope/resources/base/media/background.png");
		da->make_dir_recursive(dest_bg_path.get_base_dir());
		err = da->copy(background_image, dest_bg_path);
		if (err != OK) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not copy background image: \"%s\" to \"%s\"."), background_image, dest_bg_path));
			return err;
		}
	}

	String foreground_image = p_preset->get("build/foreground_image");
	if (!foreground_image.is_empty() && FileAccess::exists(foreground_image)) {
		String dest_fg_path = project_dir.path_join("entry/src/main/resources/base/media/foreground.png");
		da->make_dir_recursive(dest_fg_path.get_base_dir());
		err = da->copy(foreground_image, dest_fg_path);
		if (err != OK) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not copy foreground image: \"%s\" to \"%s\"."), foreground_image, dest_fg_path));
			return err;
		}
		dest_fg_path = project_dir.path_join("AppScope/resources/base/media/foreground.png");
		da->make_dir_recursive(dest_fg_path.get_base_dir());
		err = da->copy(foreground_image, dest_fg_path);
		if (err != OK) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not copy foreground image: \"%s\" to \"%s\"."), foreground_image, dest_fg_path));
			return err;
		}
	}

	String entry_build_profile_path = project_dir.path_join("entry/build-profile.json5");
	if (FileAccess::exists(entry_build_profile_path)) {
		Ref<FileAccess> entry_build_file = FileAccess::open(entry_build_profile_path, FileAccess::READ);
		if (entry_build_file.is_valid()) {
			String content = entry_build_file->get_as_text();

			String selected_arch;
			if (p_preset->get("architectures/arm64")) {
				selected_arch = "arm64-v8a";
			} else if (p_preset->get("architectures/x86_64")) {
				selected_arch = "x86_64";
			} else {
				selected_arch = "arm64-v8a";
			}

#ifdef MODULE_REGEX_ENABLED
			RegEx regex;
			regex.compile("\"abiFilters\"\\s*:\\s*\\[[^\\]]*\\]");
			content = regex.sub(content, "\"abiFilters\": [\"" + selected_arch + "\"]", true);
#else
			int start = content.find("\"abiFilters\"");
			if (start != -1) {
				int bracket_start = content.find("[", start);
				int bracket_end = content.find("]", bracket_start);
				if (bracket_start != -1 && bracket_end != -1) {
					String before = content.substr(0, bracket_start + 1);
					String after = content.substr(bracket_end);
					content = before + "\"" + selected_arch + "\"" + after;
				}
			}
#endif

			entry_build_file = FileAccess::open(entry_build_profile_path, FileAccess::WRITE);
			if (entry_build_file.is_valid()) {
				entry_build_file->store_string(content);
			} else {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write entry build profile: \"%s\"."), entry_build_profile_path));
				return ERR_FILE_CANT_WRITE;
			}
		} else {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not read entry build profile: \"%s\"."), entry_build_profile_path));
			return ERR_FILE_CANT_READ;
		}
	}

	String module_json_path = project_dir.path_join("entry/src/main/module.json5");
	if (FileAccess::exists(module_json_path)) {
		Ref<FileAccess> module_json_file = FileAccess::open(module_json_path, FileAccess::READ);
		if (module_json_file.is_valid()) {
			String content = module_json_file->get_as_text();

			String permissions_json = "  \"requestPermissions\": [\n";
			const char **perms = OPENHARMONY_PERMISSIONS;
			while (*perms) {
				String perm_name = String(*perms);
				String perm_option = vformat("%s/%s", PNAME("permissions"), perm_name);
				bool perm_enabled = p_preset->get(perm_option);
				if (perm_enabled) {
					permissions_json += "    {\n";
					permissions_json += "      \"name\": \"" + perm_name + "\",\n";
					permissions_json += "      \"reason\": \"$string:" + perm_name.trim_prefix("ohos.permission.") + "_reason\",\n";
					permissions_json += "      \"usedScene\": {\n";
					permissions_json += "        \"abilities\": [\n";
					permissions_json += "          \"FormAbility\"\n";
					permissions_json += "        ],\n";
					permissions_json += "        \"when\": \"always\"\n";
					permissions_json += "      }\n";
					permissions_json += "    },\n";
				}
				perms++;
			}
			permissions_json += "  ],\n";
			content = content.replace("\"requestPermissions\": [],", permissions_json);

			uint32_t orientation_index = p_preset->get("build/default_orientation");
			String orientation = String(OPENHARMONY_ORIENTATION_ENUMS).split(",")[orientation_index];
			content = content.replace("\"orientation\": \"portrait\",", "\"orientation\": \"" + orientation + "\",");

			module_json_file = FileAccess::open(module_json_path, FileAccess::WRITE);
			if (module_json_file.is_valid()) {
				module_json_file->store_string(content);
			} else {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not write module json: \"%s\"."), module_json_path));
				return ERR_FILE_CANT_WRITE;
			}
		} else {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), vformat(TTR("Could not read module json: \"%s\"."), module_json_path));
			return ERR_FILE_CANT_READ;
		}
	}

	if (ep.step(TTR("Saving project data..."), 4)) {
		return ERR_SKIP;
	}

	String pck_path = project_dir.path_join("/entry/src/main/resources/rawfile/template.pck");
	err = save_pack(p_preset, p_debug, pck_path);
	if (err != OK) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Export"), TTR("Could not write package file."));
		return err;
	}

	print_line(vformat("Project exported pck successfully. %s", pck_path));

	if (export_project_only) {
		print_line(vformat("Project exported successfully. Build skipped as requested."));
		return OK;
	}

	if (ep.step(TTR("Building project..."), 5)) {
		return ERR_SKIP;
	}

	String tool_path = get_tool_path();
	if (tool_path.is_empty()) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Build"), TTR("OpenHarmony tool path not configured. Please set it in the Editor Settings (Export > OpenHarmony > OpenHarmony Tool Path)."));
		return ERR_UNCONFIGURED;
	}

	if (!DirAccess::dir_exists_absolute(tool_path)) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Build"), vformat(TTR("OpenHarmony tool path does not exist: \"%s\"."), tool_path));
		return ERR_FILE_NOT_FOUND;
	}

	String hvigor_cmd = get_hvigor_path();
	if (!FileAccess::exists(hvigor_cmd)) {
		String hvigor_cmd_ide = get_hvigor_path_ide();
		if (!FileAccess::exists(hvigor_cmd_ide)) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Build"), vformat(TTR("Hvigor command not found: \"%s\" or \"%s\"."), hvigor_cmd, hvigor_cmd_ide));
			return ERR_FILE_NOT_FOUND;
		}
		hvigor_cmd = hvigor_cmd_ide;
	}

	bool is_hap = file_ext == "hap";

	List<String> args;
	args.push_back(is_hap ? "assembleHap" : "assembleApp");
	args.push_back("-p");
	args.push_back(String("buildMode=") + (p_debug ? "debug" : "release"));
	args.push_back("-p");
	args.push_back("product=default");
	if (is_hap) {
		args.push_back("-p");
		args.push_back("module=entry@default");
	}
	args.push_back("--mode");
	args.push_back(is_hap ? "module" : "project");
	args.push_back("--analyze=normal");
	args.push_back("--parallel");
	args.push_back("--incremental");
	args.push_back("--sync");
	args.push_back("--no-daemon");

	err = OS::get_singleton()->set_cwd(project_dir);
	if (err != OK) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Build"), vformat(TTR("Could not change to project directory: \"%s\"."), project_dir));
		return err;
	}
	OS::get_singleton()->set_environment("DEVECO_SDK_HOME", get_sdk_path());
	String output;
	int exit_code;
	err = OS::get_singleton()->execute(hvigor_cmd, args, &output, &exit_code, true, nullptr, false);
	OS::get_singleton()->set_cwd(EditorPaths::get_singleton()->get_project_data_dir());
	if (err != OK) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Build"), vformat(TTR("Failed to execute build command: %s"), hvigor_cmd));
		return err;
	}

	if (exit_code != 0) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Build"), vformat(TTR("Build failed with exit code %d:\n%s"), exit_code, output));
		return ERR_COMPILATION_FAILED;
	}

	if (ep.step(TTR("Copying output files..."), 6)) {
		return ERR_SKIP;
	}

	String output_dir = project_dir.path_join(is_hap ? "entry/build/default/outputs/default" : "build/outputs/default");
	Ref<DirAccess> bundle_dir = DirAccess::open(output_dir);
	if (bundle_dir.is_valid()) {
		bundle_dir->list_dir_begin();
		String file_name = bundle_dir->get_next();
		String bundle_file;

		String bundle_ext = String(should_sign ? "-signed" : "-unsigned") + (is_hap ? ".hap" : ".app");

		while (!file_name.is_empty()) {
			if (file_name.ends_with(bundle_ext)) {
				bundle_file = output_dir.path_join(file_name);
				break;
			}
			file_name = bundle_dir->get_next();
		}
		bundle_dir->list_dir_end();

		if (!bundle_file.is_empty() && FileAccess::exists(bundle_file)) {
			err = da->copy(bundle_file, base_dir.path_join(p_path.get_file()));
			if (err != OK) {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Build"), vformat(TTR("Could not copy bundle file from \"%s\" to \"%s\"."), bundle_file, p_path));
				return err;
			}
			print_line(vformat("Build completed successfully."));
		} else {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Build"), vformat(TTR("bundle file not found in output directory: \"%s\"."), output_dir));
			return ERR_FILE_NOT_FOUND;
		}
	} else {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Build"), vformat(TTR("Build output directory not found: \"%s\"."), output_dir));
		return ERR_FILE_NOT_FOUND;
	}

	return OK;
}

void EditorExportPlatformOpenHarmony::_remove_dir_recursive(const String &p_dir) {
	Ref<DirAccess> da = DirAccess::open(p_dir);
	if (da.is_valid()) {
		Error err = da->erase_contents_recursive();
		ERR_FAIL_COND_MSG(err != OK, "Could not remove directory: " + p_dir);
		err = DirAccess::remove_absolute(p_dir);
		ERR_FAIL_COND_MSG(err != OK, "Could not remove directory: " + p_dir);
	}
}

Error EditorExportPlatformOpenHarmony::run(const Ref<EditorExportPreset> &p_preset, int p_device, BitField<EditorExportPlatform::DebugFlags> p_debug_flags) {
	ERR_FAIL_INDEX_V(p_device, devices.size(), ERR_INVALID_PARAMETER);

	String can_export_error;
	bool can_export_missing_templates;
	if (!can_export(p_preset, can_export_error, can_export_missing_templates)) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Run"), can_export_error);
		return ERR_UNCONFIGURED;
	}

	MutexLock lock(device_lock);

	EditorProgress ep("run", vformat(TTR("Running on %s"), devices[p_device]), 4);

	String hdc = get_hdc_path();
	if (hdc.is_empty() || !FileAccess::exists(hdc)) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Run"), TTR("HDC command not found."));
		return ERR_FILE_NOT_FOUND;
	}

	// Export temporary HAP file
	if (ep.step(TTR("Exporting HAP..."), 0)) {
		return ERR_SKIP;
	}

	String tmp_export_path = EditorPaths::get_singleton()->get_temp_dir().path_join("tmpexport." + uitos(OS::get_singleton()->get_unix_time()) + ".hap");

#define CLEANUP_AND_RETURN(m_err)                              \
	{                                                          \
		_remove_dir_recursive(tmp_export_path.get_basename()); \
		DirAccess::remove_file_or_error(tmp_export_path);      \
		return m_err;                                          \
	}                                                          \
	((void)0)

	// Export to temporary HAP with signing forced to true
	Error err = export_project_helper(p_preset, true, tmp_export_path, true, false, p_debug_flags);
	if (err != OK) {
		CLEANUP_AND_RETURN(err);
	}
	print_line("HAP package path: " + tmp_export_path);

	List<String> args;
	int rv;
	String output;
	String device_id = devices[p_device];

	// Install HAP to device
	if (ep.step(TTR("Installing to device, please wait..."), 1)) {
		CLEANUP_AND_RETURN(ERR_SKIP);
	}

	print_line("Installing to device: " + device_id);

	err = OS::get_singleton()->set_cwd(tmp_export_path.get_base_dir());
	if (err != OK) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Run"), vformat(TTR("Could not change to hap directory: \"%s\"."), tmp_export_path.get_base_dir()));
		return err;
	}
	args.clear();
	args.push_back("-t");
	args.push_back(device_id);
	args.push_back("install");
	args.push_back(tmp_export_path.get_file());

	output.clear();
	err = OS::get_singleton()->execute(hdc, args, &output, &rv, true);
	OS::get_singleton()->set_cwd(EditorPaths::get_singleton()->get_project_data_dir());
	print_verbose(output);
	if (err || rv != 0 || output.contains("error")) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Run"), vformat(TTR("Could not install to device: %s"), output));
		CLEANUP_AND_RETURN(ERR_CANT_CREATE);
	}

	// Setup port forwarding for debugging
	if (p_debug_flags.has_flag(DEBUG_FLAG_REMOTE_DEBUG)) {
		if (ep.step(TTR("Setting up debugging..."), 2)) {
			CLEANUP_AND_RETURN(ERR_SKIP);
		}

		int dbg_port = EDITOR_GET("network/debug/remote_port");

		// remove rport with `hdc fport rm tcp:1234 tcp:1234`
		args.clear();
		args.push_back("fport");
		args.push_back("rm");
		args.push_back("tcp:" + itos(dbg_port));
		args.push_back("tcp:" + itos(dbg_port));

		output.clear();
		OS::get_singleton()->execute(hdc, args, &output, &rv, true);
		print_verbose(output);

		args.clear();
		args.push_back("rport");
		args.push_back("tcp:" + itos(dbg_port));
		args.push_back("tcp:" + itos(dbg_port));

		output.clear();
		OS::get_singleton()->execute(hdc, args, &output, &rv, true);
		print_verbose(output);
		print_line("Debug port forwarding: " + itos(dbg_port));
	}

	// Launch application
	if (ep.step(TTR("Running on device..."), 3)) {
		CLEANUP_AND_RETURN(ERR_SKIP);
	}

	args.clear();
	args.push_back("-t");
	args.push_back(device_id);
	args.push_back("shell");
	args.push_back("aa");
	args.push_back("start");
	args.push_back("-b");
	String bundle_id = p_preset->get("build/bundle_id");
	if (bundle_id.is_empty()) {
		bundle_id = OPENHARMONY_DEFAULT_BUNDLE_ID;
	}
	args.push_back(bundle_id);
	args.push_back("-a");
	args.push_back("EntryAbility");

	output.clear();
	err = OS::get_singleton()->execute(hdc, args, &output, &rv, true);
	print_verbose(output);
	if (err || rv != 0 || output.contains("error")) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Run"), vformat(TTR("Could not start application on device: %s"), output));
		CLEANUP_AND_RETURN(ERR_CANT_CREATE);
	}

	print_line("Application started successfully on device: " + device_id);

	CLEANUP_AND_RETURN(OK);
#undef CLEANUP_AND_RETURN
}

void EditorExportPlatformOpenHarmony::get_platform_features(List<String> *r_features) const {
	r_features->push_back("mobile");
	r_features->push_back("openharmony");
}

bool EditorExportPlatformOpenHarmony::has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates, bool p_debug) const {
	String err;
	bool valid = false;

	bool dvalid = false;
	bool rvalid = false;
	bool has_export_templates = false;

	if (p_preset->get("custom_template/debug") != "") {
		dvalid = FileAccess::exists(p_preset->get("custom_template/debug"));
		if (!dvalid) {
			err += TTR("Custom debug template not found.") + "\n";
		}
	} else {
		has_export_templates |= exists_export_template("openharmony_debug_arm64-v8a.zip", &err);
	}

	if (p_preset->get("custom_template/release") != "") {
		rvalid = FileAccess::exists(p_preset->get("custom_template/release"));
		if (!rvalid) {
			err += TTR("Custom release template not found.") + "\n";
		}
	} else {
		has_export_templates |= exists_export_template("openharmony_release_arm64-v8a.zip", &err);
	}

	r_missing_templates = !(dvalid || rvalid || has_export_templates);
	valid = dvalid || rvalid || has_export_templates;

	bool sign_enabled = p_preset->get("build/sign");
	if (sign_enabled) {
		String store_file = p_preset->get("sign/store_file");
		String store_password = p_preset->get("sign/store_password");
		String key_alias = p_preset->get("sign/key_alias");
		String key_password = p_preset->get("sign/key_password");
		String sign_alg = p_preset->get("sign/sign_alg");
		String profile_file = p_preset->get("sign/profile_file");
		String certpath_file = p_preset->get("sign/certpath_file");

		if (store_file.is_empty()) {
			valid = false;
			err += TTR("Store file path is required when signing is enabled.") + "\n";
		} else if (!FileAccess::exists(store_file)) {
			valid = false;
			err += TTR("Store file does not exist.") + "\n";
		}

		if (store_password.is_empty()) {
			valid = false;
			err += TTR("Store password is required when signing is enabled.") + "\n";
		}

		if (key_alias.is_empty()) {
			valid = false;
			err += TTR("Key alias is required when signing is enabled.") + "\n";
		}

		if (key_password.is_empty()) {
			valid = false;
			err += TTR("Key password is required when signing is enabled.") + "\n";
		}

		if (sign_alg.is_empty()) {
			valid = false;
			err += TTR("Sign algorithm is required when signing is enabled.") + "\n";
		}

		if (profile_file.is_empty()) {
			valid = false;
			err += TTR("Profile file path is required when signing is enabled.") + "\n";
		} else if (!FileAccess::exists(profile_file)) {
			valid = false;
			err += TTR("Profile file does not exist.") + "\n";
		}

		if (certpath_file.is_empty()) {
			valid = false;
			err += TTR("Certificate path file is required when signing is enabled.") + "\n";
		} else if (!FileAccess::exists(certpath_file)) {
			valid = false;
			err += TTR("Certificate path file does not exist.") + "\n";
		}
	}

	String background_image = p_preset->get("build/background_image");
	String foreground_image = p_preset->get("build/foreground_image");

	if (!background_image.is_empty()) {
		if (!FileAccess::exists(background_image)) {
			valid = false;
			err += TTR("Background image file does not exist.") + "\n";
		} else {
			if (background_image.get_extension().to_lower() != "png") {
				valid = false;
				err += TTR("Background image must be a PNG file.") + "\n";
			} else {
				Ref<Image> img = Image::load_from_file(background_image);
				if (img.is_null()) {
					valid = false;
					err += TTR("Failed to load background image.") + "\n";
				} else if (img->get_width() != 1024 || img->get_height() != 1024) {
					valid = false;
					err += TTR("Background image must be 1024x1024 pixels.") + "\n";
				}
			}
		}
	}

	if (!foreground_image.is_empty()) {
		if (!FileAccess::exists(foreground_image)) {
			valid = false;
			err += TTR("Foreground image file does not exist.") + "\n";
		} else {
			if (foreground_image.get_extension().to_lower() != "png") {
				valid = false;
				err += TTR("Foreground image must be a PNG file.") + "\n";
			} else {
				Ref<Image> img = Image::load_from_file(foreground_image);
				if (img.is_null()) {
					valid = false;
					err += TTR("Failed to load foreground image.") + "\n";
				} else if (img->get_width() != 1024 || img->get_height() != 1024) {
					valid = false;
					err += TTR("Foreground image must be 1024x1024 pixels.") + "\n";
				}
			}
		}
	}

	String tool_path = get_tool_path();
	if (tool_path.is_empty()) {
		valid = false;
		err += TTR("OpenHarmony tool path not configured. Please set it in the Editor Settings (Export > OpenHarmony > OpenHarmony Tool Path).") + "\n";
	}

	if (!err.is_empty()) {
		r_error = err;
	}

	return valid;
}

bool EditorExportPlatformOpenHarmony::has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error) const {
	String err;
	bool valid = true;

	// Validate preset options using our visibility and warning methods
	List<ExportOption> options;
	get_export_options(&options);
	for (const EditorExportPlatform::ExportOption &E : options) {
		if (get_export_option_visibility(p_preset.ptr(), E.option.name)) {
			String warn = get_export_option_warning(p_preset.ptr(), E.option.name);
			if (!warn.is_empty()) {
				err += warn + "\n";
				if (E.required) {
					valid = false;
				}
			}
		}
	}

	// Check if ETC2/ASTC texture compression is enabled (required for OpenHarmony)
	if (!ResourceImporterTextureSettings::should_import_etc2_astc()) {
		valid = false;
		err += TTR("ETC2/ASTC texture compression must be enabled for OpenHarmony export. Enable it in Project Settings (Rendering > Textures > VRAM Compression > Import ETC2 ASTC).") + "\n";
	}

	// Check if Vulkan renderer is being used (required for OpenHarmony)
	String rendering_method = GLOBAL_GET("rendering/renderer/rendering_method.mobile");
	String rendering_driver = GLOBAL_GET("rendering/rendering_device/driver.openharmony");

	bool uses_vulkan = (rendering_method == "forward_plus" || rendering_method == "mobile") && rendering_driver == "vulkan";
	if (!uses_vulkan) {
		valid = false;
		err += TTR("OpenHarmony export requires Vulkan renderer. Set rendering method to 'Forward+' or 'Mobile' and rendering driver to 'Vulkan' in Project Settings.") + "\n";
	}

	if (!err.is_empty()) {
		r_error = err;
	}

	return valid;
}

String EditorExportPlatformOpenHarmony::get_tool_path() const {
	return EDITOR_GET("export/openharmony/openharmony_tool_path");
}

String EditorExportPlatformOpenHarmony::get_sdk_path() const {
	String tool_path = get_tool_path();
	if (tool_path.is_empty()) {
		return "";
	}
	return tool_path.path_join("/sdk");
}

String EditorExportPlatformOpenHarmony::get_hvigor_path() const {
	String tool_path = get_tool_path();
	if (tool_path.is_empty()) {
		return "";
	}
	String exe_ext;
	if (OS::get_singleton()->get_name() == "Windows") {
		exe_ext = ".bat";
	}
	return tool_path.path_join("/hvigor/bin/hvigorw" + exe_ext);
}

String EditorExportPlatformOpenHarmony::get_hvigor_path_ide() const {
	String tool_path = get_tool_path();
	if (tool_path.is_empty()) {
		return "";
	}
	String exe_ext;
	if (OS::get_singleton()->get_name() == "Windows") {
		exe_ext = ".bat";
	}
	return tool_path.path_join("/tools/hvigor/bin/hvigorw" + exe_ext);
}

String EditorExportPlatformOpenHarmony::get_hdc_path() const {
	String sdk_path = get_sdk_path();
	if (sdk_path.is_empty()) {
		return "";
	}
	String exe_ext;
	if (OS::get_singleton()->get_name() == "Windows") {
		exe_ext = ".exe";
	}
	return sdk_path.path_join("/default/openharmony/toolchains/hdc" + exe_ext);
}

EditorExportPlatformOpenHarmony::EditorExportPlatformOpenHarmony() {
	if (EditorNode::get_singleton()) {
		Ref<Image> img = memnew(Image);
		const bool upsample = !Math::is_equal_approx(Math::round(EDSCALE), EDSCALE);

		ImageLoaderSVG::create_image_from_string(img, _openharmony_logo_svg, EDSCALE, upsample, false);
		logo = ImageTexture::create_from_image(img);

		ImageLoaderSVG::create_image_from_string(img, _openharmony_run_icon_svg, EDSCALE, upsample, false);
		run_icon = ImageTexture::create_from_image(img);

		devices_changed.set();
		_update_preset_status();
		check_for_changes_thread.start(_check_for_changes_poll_thread, this);
	}
}

void EditorExportPlatformOpenHarmony::_check_for_changes_poll_thread(void *ud) {
	EditorExportPlatformOpenHarmony *ea = static_cast<EditorExportPlatformOpenHarmony *>(ud);

	while (!ea->quit_request.is_set()) {
		String hdc = ea->get_hdc_path();
		if (ea->has_runnable_preset.is_set() && FileAccess::exists(hdc) && EditorNode::get_singleton()->is_editor_ready()) {
			String devices_output;
			List<String> args;
			args.push_back("list");
			args.push_back("targets");
			int ec;
			OS::get_singleton()->execute(hdc, args, &devices_output, &ec);

			Vector<String> ds = devices_output.split("\n");
			Vector<String> ldevices;

			for (int i = 0; i < ds.size(); i++) {
				String d = ds[i].strip_edges();
				if (d.is_empty() || d == "[Empty]") {
					continue;
				}
				ldevices.push_back(d);
			}

			MutexLock lock(ea->device_lock);

			bool different = false;

			if (ea->devices.size() != ldevices.size()) {
				different = true;
			} else {
				for (int i = 0; i < ea->devices.size(); i++) {
					if (ea->devices[i] != ldevices[i]) {
						different = true;
						break;
					}
				}
			}

			if (different) {
				ea->devices = ldevices;
				ea->devices_changed.set();
			}
		}

		uint64_t sleep = 200;
		uint64_t wait = 3000000;
		uint64_t time = OS::get_singleton()->get_ticks_usec();
		while (OS::get_singleton()->get_ticks_usec() - time < wait) {
			OS::get_singleton()->delay_usec(1000 * sleep);
			if (ea->quit_request.is_set()) {
				break;
			}
		}
	}
}

void EditorExportPlatformOpenHarmony::_update_preset_status() {
	const int preset_count = EditorExport::get_singleton()->get_export_preset_count();
	bool has_runnable = false;

	for (int i = 0; i < preset_count; i++) {
		const Ref<EditorExportPreset> &preset = EditorExport::get_singleton()->get_export_preset(i);
		if (preset->get_platform() == this && preset->is_runnable()) {
			has_runnable = true;
			break;
		}
	}

	if (has_runnable) {
		has_runnable_preset.set();
	} else {
		has_runnable_preset.clear();
	}
	devices_changed.set();
}

void EditorExportPlatformOpenHarmony::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			if (EditorExport::get_singleton()) {
				EditorExport::get_singleton()->connect_presets_runnable_updated(callable_mp(this, &EditorExportPlatformOpenHarmony::_update_preset_status));
			}
		} break;
	}
}

EditorExportPlatformOpenHarmony::~EditorExportPlatformOpenHarmony() {
	quit_request.set();
	if (check_for_changes_thread.is_started()) {
		check_for_changes_thread.wait_to_finish();
	}
}

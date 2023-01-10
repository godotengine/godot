/**************************************************************************/
/*  export_plugin.h                                                       */
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

#ifndef ANDROID_EXPORT_PLUGIN_H
#define ANDROID_EXPORT_PLUGIN_H

#include "core/io/image_loader.h"
#include "core/io/json.h"
#include "core/io/marshalls.h"
#include "core/io/zip_io.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "core/safe_refcount.h"
#include "core/version.h"
#include "drivers/png/png_driver_common.h"
#include "editor/editor_export.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "main/splash.gen.h"
#include "platform/android/logo.gen.h"
#include "platform/android/run_icon.gen.h"

#include "godot_plugin_config.h"
#include "gradle_export_util.h"

#include <string.h>

const String SPLASH_CONFIG_XML_CONTENT = R"SPLASH(<?xml version="1.0" encoding="utf-8"?>
<layer-list xmlns:android="http://schemas.android.com/apk/res/android">
	<item android:drawable="@drawable/splash_bg_color" />
	<item>
		<bitmap
				android:gravity="center"
				android:filter="%s"
				android:src="@drawable/splash" />
	</item>
</layer-list>
)SPLASH";

struct LauncherIcon {
	const char *export_path;
	int dimensions;
};

class EditorExportPlatformAndroid : public EditorExportPlatform {
	GDCLASS(EditorExportPlatformAndroid, EditorExportPlatform);

	Ref<ImageTexture> logo;
	Ref<ImageTexture> run_icon;

	struct Device {
		String id;
		String name;
		String description;
		int api_level;
	};

	struct APKExportData {
		zipFile apk;
		EditorProgress *ep;
	};

	Vector<PluginConfigAndroid> plugins;
	String last_plugin_names;
	uint64_t last_custom_build_time = 0;
	SafeFlag plugins_changed;
	Mutex plugins_lock;
	Vector<Device> devices;
	SafeFlag devices_changed;
	Mutex device_lock;
#ifndef ANDROID_ENABLED
	Thread check_for_changes_thread;
	SafeFlag quit_request;

	static void _check_for_changes_poll_thread(void *ud);
#endif

	String get_project_name(const String &p_name) const;

	String get_package_name(const String &p_package) const;

	String get_assets_directory(const Ref<EditorExportPreset> &p_preset, int p_export_format) const;

	bool is_package_name_valid(const String &p_package, String *r_error = nullptr) const;

	static bool _should_compress_asset(const String &p_path, const Vector<uint8_t> &p_data);

	static zip_fileinfo get_zip_fileinfo();

	static Vector<String> get_abis();

	/// List the gdap files in the directory specified by the p_path parameter.
	static Vector<String> list_gdap_files(const String &p_path);

	static Vector<PluginConfigAndroid> get_plugins();

	static Vector<PluginConfigAndroid> get_enabled_plugins(const Ref<EditorExportPreset> &p_presets);

	static Error store_in_apk(APKExportData *ed, const String &p_path, const Vector<uint8_t> &p_data, int compression_method = Z_DEFLATED);

	static Error save_apk_so(void *p_userdata, const SharedObject &p_so);

	static Error save_apk_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);

	static Error ignore_apk_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);

	static Error copy_gradle_so(void *p_userdata, const SharedObject &p_so);

	bool _has_read_write_storage_permission(const Vector<String> &p_permissions);

	bool _has_manage_external_storage_permission(const Vector<String> &p_permissions);

	void _get_permissions(const Ref<EditorExportPreset> &p_preset, bool p_give_internet, Vector<String> &r_permissions);

	void _write_tmp_manifest(const Ref<EditorExportPreset> &p_preset, bool p_give_internet, bool p_debug);

	void _fix_manifest(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &p_manifest, bool p_give_internet);

	static String _parse_string(const uint8_t *p_bytes, bool p_utf8);

	void _fix_resources(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &r_manifest);

	void _load_image_data(const Ref<Image> &p_splash_image, Vector<uint8_t> &p_data);

	void _process_launcher_icons(const String &p_file_name, const Ref<Image> &p_source_image, int dimension, Vector<uint8_t> &p_data);

	String load_splash_refs(Ref<Image> &splash_image, Ref<Image> &splash_bg_color_image);

	void load_icon_refs(const Ref<EditorExportPreset> &p_preset, Ref<Image> &icon, Ref<Image> &foreground, Ref<Image> &background);

	void store_image(const LauncherIcon launcher_icon, const Vector<uint8_t> &data);

	void store_image(const String &export_path, const Vector<uint8_t> &data);

	void _copy_icons_to_gradle_project(const Ref<EditorExportPreset> &p_preset,
			const String &processed_splash_config_xml,
			const Ref<Image> &splash_image,
			const Ref<Image> &splash_bg_color_image,
			const Ref<Image> &main_image,
			const Ref<Image> &foreground,
			const Ref<Image> &background);

	static Vector<String> get_enabled_abis(const Ref<EditorExportPreset> &p_preset);

public:
	typedef Error (*EditorExportSaveFunction)(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);

public:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features);

	virtual void get_export_options(List<ExportOption> *r_options);

	virtual String get_name() const;

	virtual String get_os_name() const;

	virtual Ref<Texture> get_logo() const;

	virtual bool should_update_export_options();

	virtual bool poll_export();

	virtual int get_options_count() const;

	virtual String get_options_tooltip() const;

	virtual String get_option_label(int p_index) const;

	virtual String get_option_tooltip(int p_index) const;

	virtual Error run(const Ref<EditorExportPreset> &p_preset, int p_device, int p_debug_flags);

	virtual Ref<Texture> get_run_icon() const;

	static String get_adb_path();

	static String get_apksigner_path();

	virtual bool has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const;
	virtual bool has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error) const;

	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const;

	void _update_custom_build_project();

	inline bool is_clean_build_required(Vector<PluginConfigAndroid> enabled_plugins) {
		String plugin_names = PluginConfigAndroid::get_plugins_names(enabled_plugins);
		bool first_build = last_custom_build_time == 0;
		bool have_plugins_changed = false;

		if (!first_build) {
			have_plugins_changed = plugin_names != last_plugin_names;
			if (!have_plugins_changed) {
				for (int i = 0; i < enabled_plugins.size(); i++) {
					if (enabled_plugins.get(i).last_updated > last_custom_build_time) {
						have_plugins_changed = true;
						break;
					}
				}
			}
		}

		last_custom_build_time = OS::get_singleton()->get_unix_time();
		last_plugin_names = plugin_names;

		return have_plugins_changed || first_build;
	}

	String get_apk_expansion_fullpath(const Ref<EditorExportPreset> &p_preset, const String &p_path);

	Error save_apk_expansion_file(const Ref<EditorExportPreset> &p_preset, const String &p_path);

	void get_command_line_flags(const Ref<EditorExportPreset> &p_preset, const String &p_path, int p_flags, Vector<uint8_t> &r_command_line_flags);

	Error sign_apk(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &export_path, EditorProgress &ep);

	void _clear_assets_directory();

	void _remove_copied_libs();

	String join_list(List<String> parts, const String &separator) const;

	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);

	Error export_project_helper(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int export_format, bool should_sign, int p_flags);

	virtual void get_platform_features(List<String> *r_features);

	virtual void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features);

	EditorExportPlatformAndroid();

	~EditorExportPlatformAndroid();
};

#endif // ANDROID_EXPORT_PLUGIN_H

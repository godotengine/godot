/**************************************************************************/
/*  editor_export_platform_apple_embedded.h                               */
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

#pragma once

#include "plugin_config_apple_embedded.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/image_loader.h"
#include "core/io/marshalls.h"
#include "core/io/resource_saver.h"
#include "core/io/zip_io.h"
#include "core/os/os.h"
#include "core/templates/safe_refcount.h"
#include "editor/editor_settings.h"
#include "editor/export/editor_export_platform.h"
#include "main/splash.gen.h"
#include "scene/resources/image_texture.h"

#include <sys/stat.h>

// Optional environment variables for defining confidential information. If any
// of these is set, they will override the values set in the credentials file.
const String ENV_APPLE_PLATFORM_PROFILE_UUID_DEBUG = "GODOT_APPLE_PLATFORM_PROVISIONING_PROFILE_UUID_DEBUG";
const String ENV_APPLE_PLATFORM_PROFILE_UUID_RELEASE = "GODOT_APPLE_PLATFORM_PROVISIONING_PROFILE_UUID_RELEASE";
const String ENV_APPLE_PLATFORM_PROFILE_SPECIFIER_DEBUG = "GODOT_APPLE_PLATFORM_PROFILE_SPECIFIER_DEBUG";
const String ENV_APPLE_PLATFORM_PROFILE_SPECIFIER_RELEASE = "GODOT_APPLE_PLATFORM_PROFILE_SPECIFIER_RELEASE";

static const String storyboard_image_scale_mode[] = {
	"center",
	"scaleAspectFit",
	"scaleAspectFill",
	"scaleToFill",
};

class EditorExportPlatformAppleEmbedded : public EditorExportPlatform {
	GDCLASS(EditorExportPlatformAppleEmbedded, EditorExportPlatform);

	Ref<ImageTexture> logo;
	Ref<ImageTexture> run_icon;

	// Plugins
	mutable SafeFlag plugins_changed;
	SafeFlag devices_changed;

	struct Device {
		String id;
		String name;
		bool wifi = false;
		bool use_ios_deploy = false;
	};

	Vector<Device> devices;
	Mutex device_lock;

	Mutex plugins_lock;
	mutable Vector<PluginConfigAppleEmbedded> plugins;
#ifdef MACOS_ENABLED
	Thread check_for_changes_thread;
	SafeFlag quit_request;
	SafeFlag has_runnable_preset;

	static bool _check_xcode_install();
	static void _check_for_changes_poll_thread(void *ud);
	void _update_preset_status();
#endif

	typedef Error (*FileHandler)(String p_file, void *p_userdata);
	static Error _walk_dir_recursive(Ref<DirAccess> &p_da, FileHandler p_handler, void *p_userdata);
	static Error _codesign(String p_file, void *p_userdata);

	struct AppleEmbeddedConfigData {
		String pkg_name;
		String binary_name;
		String plist_content;
		String architectures;
		String linker_flags;
		String cpp_code;
		String modules_buildfile;
		String modules_fileref;
		String modules_buildphase;
		String modules_buildgrp;
		Vector<String> capabilities;
		bool use_swift_runtime;
	};

	struct ExportArchitecture {
		String name;
		bool is_default = false;

		ExportArchitecture() {}

		ExportArchitecture(String p_name, bool p_is_default) {
			name = p_name;
			is_default = p_is_default;
		}
	};

	struct AppleEmbeddedExportAsset {
		String exported_path;
		bool is_framework = false; // framework is anything linked to the binary, otherwise it's a resource
		bool should_embed = false;
	};

	String _get_additional_plist_content();
	String _get_linker_flags();
	String _get_cpp_code();
	void _fix_config_file(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &pfile, const AppleEmbeddedConfigData &p_config, bool p_debug);

	Vector<ExportArchitecture> _get_supported_architectures() const;
	Vector<String> _get_preset_architectures(const Ref<EditorExportPreset> &p_preset) const;

	void _check_xcframework_content(const String &p_path, int &r_total_libs, int &r_static_libs, int &r_dylibs, int &r_frameworks) const;
	Error _convert_to_framework(const String &p_source, const String &p_destination, const String &p_id) const;

	void _add_assets_to_project(const String &p_out_dir, const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &p_project_data, const Vector<AppleEmbeddedExportAsset> &p_additional_assets);
	Error _export_additional_assets(const Ref<EditorExportPreset> &p_preset, const String &p_out_dir, const Vector<String> &p_assets, bool p_is_framework, bool p_should_embed, Vector<AppleEmbeddedExportAsset> &r_exported_assets);
	Error _copy_asset(const Ref<EditorExportPreset> &p_preset, const String &p_out_dir, const String &p_asset, const String *p_custom_file_name, bool p_is_framework, bool p_should_embed, Vector<AppleEmbeddedExportAsset> &r_exported_assets);
	Error _export_additional_assets(const Ref<EditorExportPreset> &p_preset, const String &p_out_dir, const Vector<SharedObject> &p_libraries, Vector<AppleEmbeddedExportAsset> &r_exported_assets);
	Error _export_apple_embedded_plugins(const Ref<EditorExportPreset> &p_preset, AppleEmbeddedConfigData &p_config_data, const String &dest_dir, Vector<AppleEmbeddedExportAsset> &r_exported_assets, bool p_debug);

	Error _export_project_helper(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags, bool p_oneclick);

	bool is_package_name_valid(const String &p_package, String *r_error = nullptr) const;

protected:
	struct IconInfo {
		const char *preset_key;
		const char *idiom;
		const char *export_name;
		const char *actual_size_side;
		const char *scale;
		const char *unscaled_size;
		bool force_opaque;
	};

	void _blend_and_rotate(Ref<Image> &p_dst, Ref<Image> &p_src, bool p_rot);

	virtual Error _export_loading_screen_file(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir) { return OK; }
	virtual Error _export_icons(const Ref<EditorExportPreset> &p_preset, const String &p_iconset_dir) { return OK; }

	virtual String get_platform_name() const = 0;
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) const override;
	virtual void get_export_options(List<ExportOption> *r_options) const override;
	virtual bool get_export_option_visibility(const EditorExportPreset *p_preset, const String &p_option) const override;
	virtual String get_export_option_warning(const EditorExportPreset *p_preset, const StringName &p_name) const override;

	virtual Vector<IconInfo> get_icon_infos() const = 0;

	void _notification(int p_what);

	virtual void get_platform_features(List<String> *r_features) const override {
		r_features->push_back("mobile");
		r_features->push_back("apple_embedded");
	}

public:
	virtual Ref<Texture2D> get_logo() const override { return logo; }
	virtual Ref<Texture2D> get_run_icon() const override { return run_icon; }

	virtual int get_options_count() const override;
	virtual String get_options_tooltip() const override;
	virtual Ref<ImageTexture> get_option_icon(int p_index) const override;
	virtual String get_option_label(int p_index) const override;
	virtual String get_option_tooltip(int p_index) const override;
	virtual Error run(const Ref<EditorExportPreset> &p_preset, int p_device, BitField<EditorExportPlatform::DebugFlags> p_debug_flags) override;

	virtual bool poll_export() override {
		bool dc = devices_changed.is_set();
		if (dc) {
			// don't clear unless we're reporting true, to avoid race
			devices_changed.clear();
		}
		return dc;
	}

	virtual bool should_update_export_options() override {
		bool export_options_changed = plugins_changed.is_set();
		if (export_options_changed) {
			// don't clear unless we're reporting true, to avoid race
			plugins_changed.clear();
		}
		return export_options_changed;
	}

	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const override {
		List<String> list;
		if (p_preset.is_valid()) {
			bool project_only = p_preset->get("application/export_project_only");
			if (project_only) {
				list.push_back("xcodeproj");
			} else {
				list.push_back("ipa");
			}
		}
		return list;
	}

	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags = 0) override;

	virtual bool has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates, bool p_debug = false) const override;
	virtual bool has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error) const override;

	virtual void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, HashSet<String> &p_features) override {
	}

	EditorExportPlatformAppleEmbedded(const char *p_platform_logo_svg, const char *p_run_icon_svg);
	~EditorExportPlatformAppleEmbedded();

	/// List the gdip files in the directory specified by the p_path parameter.
	static Vector<String> list_plugin_config_files(const String &p_path, bool p_check_directories) {
		Vector<String> dir_files;
		Ref<DirAccess> da = DirAccess::open(p_path);
		if (da.is_valid()) {
			da->list_dir_begin();
			while (true) {
				String file = da->get_next();
				if (file.is_empty()) {
					break;
				}

				if (file == "." || file == "..") {
					continue;
				}

				if (da->current_is_hidden()) {
					continue;
				}

				if (da->current_is_dir()) {
					if (p_check_directories) {
						Vector<String> directory_files = list_plugin_config_files(p_path.path_join(file), false);
						for (int i = 0; i < directory_files.size(); ++i) {
							dir_files.push_back(file.path_join(directory_files[i]));
						}
					}

					continue;
				}

				if (file.ends_with(PluginConfigAppleEmbedded::PLUGIN_CONFIG_EXT)) {
					dir_files.push_back(file);
				}
			}
			da->list_dir_end();
		}

		return dir_files;
	}

	static Vector<PluginConfigAppleEmbedded> get_plugins(const String &p_platform_name) {
		Vector<PluginConfigAppleEmbedded> loaded_plugins;

		String plugins_dir = ProjectSettings::get_singleton()->get_resource_path().path_join(p_platform_name + "/plugins");

		if (DirAccess::exists(plugins_dir)) {
			Vector<String> plugins_filenames = list_plugin_config_files(plugins_dir, true);

			if (!plugins_filenames.is_empty()) {
				Ref<ConfigFile> config_file;
				for (int i = 0; i < plugins_filenames.size(); i++) {
					PluginConfigAppleEmbedded config = PluginConfigAppleEmbedded::load_plugin_config(config_file, plugins_dir.path_join(plugins_filenames[i]));
					if (config.valid_config) {
						loaded_plugins.push_back(config);
					} else {
						print_error("Invalid plugin config file " + plugins_filenames[i]);
					}
				}
			}
		}

		return loaded_plugins;
	}

	static Vector<PluginConfigAppleEmbedded> get_enabled_plugins(const String &p_platform_name, const Ref<EditorExportPreset> &p_presets) {
		Vector<PluginConfigAppleEmbedded> enabled_plugins;
		Vector<PluginConfigAppleEmbedded> all_plugins = get_plugins(p_platform_name);
		for (int i = 0; i < all_plugins.size(); i++) {
			PluginConfigAppleEmbedded plugin = all_plugins[i];
			bool enabled = p_presets->get("plugins/" + plugin.name);
			if (enabled) {
				enabled_plugins.push_back(plugin);
			}
		}

		return enabled_plugins;
	}
};

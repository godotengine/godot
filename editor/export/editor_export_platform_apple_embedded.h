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
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/image_loader.h"
#include "core/io/marshalls.h"
#include "core/io/resource_saver.h"
#include "core/io/zip_io.h"
#include "core/os/os.h"
#include "core/templates/safe_refcount.h"
#include "editor/export/editor_export_platform.h"
#include "editor/settings/editor_settings.h"
#include "main/splash.gen.h"
#include "scene/resources/image_texture.h"

#include <sys/stat.h>
#include <functional>

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

protected:
	void _start_remote_device_poller_thread() {
		check_for_changes_thread.start(_check_for_changes_poll_thread, this);
	}

	void _stop_remote_device_poller_thread() {
		quit_request.set();
		if (check_for_changes_thread.is_started()) {
			check_for_changes_thread.wait_to_finish();
		}
	}

	int _execute(const String &p_path, const List<String> &p_arguments, std::function<void(const String &)> p_on_data);

private:
#endif

	typedef Error (*FileHandler)(String p_file, void *p_userdata);
	static Error _walk_dir_recursive(Ref<DirAccess> &p_da, FileHandler p_handler, void *p_userdata);
	static Error _codesign(String p_file, void *p_userdata);

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

protected:
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
	};

	struct CodeSigningDetails {
		String debug_signing_identity;
		String release_signing_identity;
		String debug_provisioning_profile_uuid;
		String release_provisioning_profile_uuid;
		String debug_provisioning_profile_specifier;
		String release_provisioning_profile_specifier;
		bool debug_manual_signing = false;
		bool release_manual_signing = false;

		CodeSigningDetails(const Ref<EditorExportPreset> &p_preset) {
			debug_signing_identity = p_preset->get("application/code_sign_identity_debug").operator String().is_empty() ? "Apple Development" : p_preset->get("application/code_sign_identity_debug");
			release_signing_identity = p_preset->get("application/code_sign_identity_release").operator String().is_empty() ? "Apple Distribution" : p_preset->get("application/code_sign_identity_release");

			debug_provisioning_profile_uuid = p_preset->get_or_env("application/provisioning_profile_uuid_debug", ENV_APPLE_PLATFORM_PROFILE_UUID_DEBUG).operator String();
			release_provisioning_profile_uuid = p_preset->get_or_env("application/provisioning_profile_uuid_release", ENV_APPLE_PLATFORM_PROFILE_UUID_DEBUG).operator String();

			debug_manual_signing = !debug_provisioning_profile_uuid.is_empty() || (debug_signing_identity != "Apple Development" && debug_signing_identity != "Apple Distribution");
			release_manual_signing = !release_provisioning_profile_uuid.is_empty() || (release_signing_identity != "Apple Development" && release_signing_identity != "Apple Distribution");

			debug_provisioning_profile_specifier = p_preset->get_or_env("application/provisioning_profile_specifier_debug", ENV_APPLE_PLATFORM_PROFILE_SPECIFIER_DEBUG).operator String();
			debug_manual_signing |= !debug_provisioning_profile_specifier.is_empty();

			release_provisioning_profile_specifier = p_preset->get_or_env("application/provisioning_profile_specifier_release", ENV_APPLE_PLATFORM_PROFILE_SPECIFIER_RELEASE).operator String();
			release_manual_signing |= !release_provisioning_profile_specifier.is_empty();
		}
	};

	struct IconInfo {
		const char *preset_key;
		const char *idiom;
		const char *export_name;
		const char *actual_size_side;
		const char *scale;
		const char *unscaled_size;
		bool force_opaque;
	};

private:
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
	virtual String _process_config_file_line(const Ref<EditorExportPreset> &p_preset, const String &p_line, const AppleEmbeddedConfigData &p_config, bool p_debug, const CodeSigningDetails &p_code_signing);

	void _blend_and_rotate(Ref<Image> &p_dst, Ref<Image> &p_src, bool p_rot);

	virtual Error _export_loading_screen_file(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir) { return OK; }
	virtual Error _export_icons(const Ref<EditorExportPreset> &p_preset, const String &p_iconset_dir) { return OK; }

	virtual String get_platform_name() const = 0;
	virtual String get_sdk_name() const = 0;
	virtual const Vector<String> get_device_types() const = 0;
	virtual String get_minimum_deployment_target() const = 0;

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

	void _initialize(const char *p_platform_logo_svg, const char *p_run_icon_svg);

public:
	virtual Ref<Texture2D> get_logo() const override { return logo; }
	virtual Ref<Texture2D> get_run_icon() const override { return run_icon; }

	virtual int get_options_count() const override;
	virtual String get_options_tooltip() const override;
	virtual Ref<Texture2D> get_option_icon(int p_index) const override;
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
				config_file.instantiate();
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

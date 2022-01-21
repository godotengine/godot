/*************************************************************************/
/*  export_plugin.h                                                      */
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

#ifndef IPHONE_EXPORT_PLUGIN_H
#define IPHONE_EXPORT_PLUGIN_H

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/image_loader.h"
#include "core/io/marshalls.h"
#include "core/io/resource_saver.h"
#include "core/io/zip_io.h"
#include "core/os/os.h"
#include "core/templates/safe_refcount.h"
#include "core/version.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "main/splash.gen.h"
#include "platform/iphone/logo.gen.h"
#include "string.h"

#include "godot_plugin_config.h"

#include <sys/stat.h>

class EditorExportPlatformIOS : public EditorExportPlatform {
	GDCLASS(EditorExportPlatformIOS, EditorExportPlatform);

	int version_code;

	Ref<ImageTexture> logo;

	// Plugins
	SafeFlag plugins_changed;
	Thread check_for_changes_thread;
	SafeFlag quit_request;
	Mutex plugins_lock;
	Vector<PluginConfigIOS> plugins;

	typedef Error (*FileHandler)(String p_file, void *p_userdata);
	static Error _walk_dir_recursive(DirAccess *p_da, FileHandler p_handler, void *p_userdata);
	static Error _codesign(String p_file, void *p_userdata);
	void _blend_and_rotate(Ref<Image> &p_dst, Ref<Image> &p_src, bool p_rot);

	struct IOSConfigData {
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
	struct ExportArchitecture {
		String name;
		bool is_default = false;

		ExportArchitecture() {}

		ExportArchitecture(String p_name, bool p_is_default) {
			name = p_name;
			is_default = p_is_default;
		}
	};

	struct IOSExportAsset {
		String exported_path;
		bool is_framework = false; // framework is anything linked to the binary, otherwise it's a resource
		bool should_embed = false;
	};

	String _get_additional_plist_content();
	String _get_linker_flags();
	String _get_cpp_code();
	void _fix_config_file(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &pfile, const IOSConfigData &p_config, bool p_debug);
	Error _export_loading_screen_images(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir);
	Error _export_loading_screen_file(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir);
	Error _export_icons(const Ref<EditorExportPreset> &p_preset, const String &p_iconset_dir);

	Vector<ExportArchitecture> _get_supported_architectures();
	Vector<String> _get_preset_architectures(const Ref<EditorExportPreset> &p_preset);

	void _add_assets_to_project(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &p_project_data, const Vector<IOSExportAsset> &p_additional_assets);
	Error _export_additional_assets(const String &p_out_dir, const Vector<String> &p_assets, bool p_is_framework, bool p_should_embed, Vector<IOSExportAsset> &r_exported_assets);
	Error _copy_asset(const String &p_out_dir, const String &p_asset, const String *p_custom_file_name, bool p_is_framework, bool p_should_embed, Vector<IOSExportAsset> &r_exported_assets);
	Error _export_additional_assets(const String &p_out_dir, const Vector<SharedObject> &p_libraries, Vector<IOSExportAsset> &r_exported_assets);
	Error _export_ios_plugins(const Ref<EditorExportPreset> &p_preset, IOSConfigData &p_config_data, const String &dest_dir, Vector<IOSExportAsset> &r_exported_assets, bool p_debug);

	bool is_package_name_valid(const String &p_package, String *r_error = nullptr) const {
		String pname = p_package;

		if (pname.length() == 0) {
			if (r_error) {
				*r_error = TTR("Identifier is missing.");
			}
			return false;
		}

		for (int i = 0; i < pname.length(); i++) {
			char32_t c = pname[i];
			if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '.')) {
				if (r_error) {
					*r_error = vformat(TTR("The character '%s' is not allowed in Identifier."), String::chr(c));
				}
				return false;
			}
		}

		return true;
	}

	static void _check_for_changes_poll_thread(void *ud) {
		EditorExportPlatformIOS *ea = (EditorExportPlatformIOS *)ud;

		while (!ea->quit_request.is_set()) {
			// Nothing to do if we already know the plugins have changed.
			if (!ea->plugins_changed.is_set()) {
				MutexLock lock(ea->plugins_lock);

				Vector<PluginConfigIOS> loaded_plugins = get_plugins();

				if (ea->plugins.size() != loaded_plugins.size()) {
					ea->plugins_changed.set();
				} else {
					for (int i = 0; i < ea->plugins.size(); i++) {
						if (ea->plugins[i].name != loaded_plugins[i].name || ea->plugins[i].last_updated != loaded_plugins[i].last_updated) {
							ea->plugins_changed.set();
							break;
						}
					}
				}
			}

			uint64_t wait = 3000000;
			uint64_t time = OS::get_singleton()->get_ticks_usec();
			while (OS::get_singleton()->get_ticks_usec() - time < wait) {
				OS::get_singleton()->delay_usec(300000);

				if (ea->quit_request.is_set()) {
					break;
				}
			}
		}
	}

protected:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) override;
	virtual void get_export_options(List<ExportOption> *r_options) override;

public:
	virtual String get_name() const override { return "iOS"; }
	virtual String get_os_name() const override { return "iOS"; }
	virtual Ref<Texture2D> get_logo() const override { return logo; }

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
		list.push_back("ipa");
		return list;
	}
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0) override;

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const override;

	virtual void get_platform_features(List<String> *r_features) override {
		r_features->push_back("mobile");
		r_features->push_back("ios");
	}

	virtual void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features) override {
	}

	EditorExportPlatformIOS();
	~EditorExportPlatformIOS();

	/// List the gdip files in the directory specified by the p_path parameter.
	static Vector<String> list_plugin_config_files(const String &p_path, bool p_check_directories) {
		Vector<String> dir_files;
		DirAccessRef da = DirAccess::open(p_path);
		if (da) {
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
						Vector<String> directory_files = list_plugin_config_files(p_path.plus_file(file), false);
						for (int i = 0; i < directory_files.size(); ++i) {
							dir_files.push_back(file.plus_file(directory_files[i]));
						}
					}

					continue;
				}

				if (file.ends_with(PluginConfigIOS::PLUGIN_CONFIG_EXT)) {
					dir_files.push_back(file);
				}
			}
			da->list_dir_end();
		}

		return dir_files;
	}

	static Vector<PluginConfigIOS> get_plugins() {
		Vector<PluginConfigIOS> loaded_plugins;

		String plugins_dir = ProjectSettings::get_singleton()->get_resource_path().plus_file("ios/plugins");

		if (DirAccess::exists(plugins_dir)) {
			Vector<String> plugins_filenames = list_plugin_config_files(plugins_dir, true);

			if (!plugins_filenames.is_empty()) {
				Ref<ConfigFile> config_file = memnew(ConfigFile);
				for (int i = 0; i < plugins_filenames.size(); i++) {
					PluginConfigIOS config = PluginConfigIOS::load_plugin_config(config_file, plugins_dir.plus_file(plugins_filenames[i]));
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

	static Vector<PluginConfigIOS> get_enabled_plugins(const Ref<EditorExportPreset> &p_presets) {
		Vector<PluginConfigIOS> enabled_plugins;
		Vector<PluginConfigIOS> all_plugins = get_plugins();
		for (int i = 0; i < all_plugins.size(); i++) {
			PluginConfigIOS plugin = all_plugins[i];
			bool enabled = p_presets->get("plugins/" + plugin.name);
			if (enabled) {
				enabled_plugins.push_back(plugin);
			}
		}

		return enabled_plugins;
	}
};

#endif

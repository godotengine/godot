/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "export.h"
#include "core/io/marshalls.h"
#include "core/io/resource_saver.h"
#include "core/io/zip_io.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "core/version.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "main/splash.gen.h"
#include "platform/iphone/logo.gen.h"
#include "platform/iphone/plugin/godot_plugin_config.h"
#include "string.h"

#include <sys/stat.h>

class EditorExportPlatformIOS : public EditorExportPlatform {

	GDCLASS(EditorExportPlatformIOS, EditorExportPlatform);

	int version_code;

	Ref<ImageTexture> logo;

	// Plugins
	volatile bool plugins_changed;
	Thread *check_for_changes_thread;
	volatile bool quit_request;
	Mutex *plugins_lock;
	Vector<PluginConfigIOS> plugins;

	typedef Error (*FileHandler)(String p_file, void *p_userdata);
	static Error _walk_dir_recursive(DirAccess *p_da, FileHandler p_handler, void *p_userdata);
	static Error _codesign(String p_file, void *p_userdata);

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
		bool is_default;

		ExportArchitecture() :
				name(""),
				is_default(false) {
		}

		ExportArchitecture(String p_name, bool p_is_default) {
			name = p_name;
			is_default = p_is_default;
		}
	};

	struct IOSExportAsset {
		String exported_path;
		bool is_framework; // framework is anything linked to the binary, otherwise it's a resource
		bool should_embed;
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
	Error _copy_asset(const String &p_out_dir, const String &p_asset, const String *p_custom_file_name, bool p_is_framework, bool p_should_embed, Vector<IOSExportAsset> &r_exported_assets);
	Error _export_additional_assets(const String &p_out_dir, const Vector<String> &p_assets, bool p_is_framework, bool p_should_embed, Vector<IOSExportAsset> &r_exported_assets);
	Error _export_additional_assets(const String &p_out_dir, const Vector<SharedObject> &p_libraries, Vector<IOSExportAsset> &r_exported_assets);
	Error _export_ios_plugins(const Ref<EditorExportPreset> &p_preset, IOSConfigData &p_config_data, const String &dest_dir, Vector<IOSExportAsset> &r_exported_assets, bool p_debug);

	bool is_package_name_valid(const String &p_package, String *r_error = NULL) const {

		String pname = p_package;

		if (pname.length() == 0) {
			if (r_error) {
				*r_error = TTR("Identifier is missing.");
			}
			return false;
		}

		for (int i = 0; i < pname.length(); i++) {
			CharType c = pname[i];
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

		while (!ea->quit_request) {
			// Nothing to do if we already know the plugins have changed.
			if (!ea->plugins_changed) {

				ea->plugins_lock->lock();

				Vector<PluginConfigIOS> loaded_plugins = get_plugins();

				if (ea->plugins.size() != loaded_plugins.size()) {
					ea->plugins_changed = true;
				} else {
					for (int i = 0; i < ea->plugins.size(); i++) {
						if (ea->plugins[i].name != loaded_plugins[i].name || ea->plugins[i].last_updated != loaded_plugins[i].last_updated) {
							ea->plugins_changed = true;
							break;
						}
					}
				}

				ea->plugins_lock->unlock();
			}

			uint64_t wait = 3000000;
			uint64_t time = OS::get_singleton()->get_ticks_usec();
			while (OS::get_singleton()->get_ticks_usec() - time < wait) {
				OS::get_singleton()->delay_usec(300000);

				if (ea->quit_request) {
					break;
				}
			}
		}
	}

protected:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features);
	virtual void get_export_options(List<ExportOption> *r_options);

public:
	virtual String get_name() const { return "iOS"; }
	virtual String get_os_name() const { return "iOS"; }
	virtual Ref<Texture> get_logo() const { return logo; }

	virtual bool should_update_export_options() {
		bool export_options_changed = plugins_changed;
		if (export_options_changed) {
			// don't clear unless we're reporting true, to avoid race
			plugins_changed = false;
		}
		return export_options_changed;
	}

	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const {
		List<String> list;
		list.push_back("ipa");
		return list;
	}

	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const;

	virtual void get_platform_features(List<String> *r_features) {

		r_features->push_back("mobile");
		r_features->push_back("iOS");
	}

	virtual void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features) {
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
				if (file.empty()) {
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

			if (!plugins_filenames.empty()) {
				Ref<ConfigFile> config_file = memnew(ConfigFile);
				for (int i = 0; i < plugins_filenames.size(); i++) {
					PluginConfigIOS config = load_plugin_config(config_file, plugins_dir.plus_file(plugins_filenames[i]));
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

void EditorExportPlatformIOS::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) {

	String driver = ProjectSettings::get_singleton()->get("rendering/quality/driver/driver_name");
	r_features->push_back("pvrtc");
	if (driver == "GLES3") {
		r_features->push_back("etc2");
	}

	Vector<String> architectures = _get_preset_architectures(p_preset);
	for (int i = 0; i < architectures.size(); ++i) {
		r_features->push_back(architectures[i]);
	}
}

Vector<EditorExportPlatformIOS::ExportArchitecture> EditorExportPlatformIOS::_get_supported_architectures() {
	Vector<ExportArchitecture> archs;
	archs.push_back(ExportArchitecture("armv7", false)); // Disabled by default, not included in official templates.
	archs.push_back(ExportArchitecture("arm64", true));
	return archs;
}

struct LoadingScreenInfo {
	const char *preset_key;
	const char *export_name;
};

static const LoadingScreenInfo loading_screen_infos[] = {
	{ "landscape_launch_screens/iphone_2436x1125", "Default-Landscape-X.png" },
	{ "landscape_launch_screens/iphone_2208x1242", "Default-Landscape-736h@3x.png" },
	{ "landscape_launch_screens/ipad_1024x768", "Default-Landscape.png" },
	{ "landscape_launch_screens/ipad_2048x1536", "Default-Landscape@2x.png" },

	{ "portrait_launch_screens/iphone_640x960", "Default-480h@2x.png" },
	{ "portrait_launch_screens/iphone_640x1136", "Default-568h@2x.png" },
	{ "portrait_launch_screens/iphone_750x1334", "Default-667h@2x.png" },
	{ "portrait_launch_screens/iphone_1125x2436", "Default-Portrait-X.png" },
	{ "portrait_launch_screens/ipad_768x1024", "Default-Portrait.png" },
	{ "portrait_launch_screens/ipad_1536x2048", "Default-Portrait@2x.png" },
	{ "portrait_launch_screens/iphone_1242x2208", "Default-Portrait-736h@3x.png" }
};

void EditorExportPlatformIOS::get_export_options(List<ExportOption> *r_options) {

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/debug", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/release", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));

	Vector<ExportArchitecture> architectures = _get_supported_architectures();
	for (int i = 0; i < architectures.size(); ++i) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "architectures/" + architectures[i].name), architectures[i].is_default));
	}

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/app_store_team_id"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/provisioning_profile_uuid_debug"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/code_sign_identity_debug", PROPERTY_HINT_PLACEHOLDER_TEXT, "iPhone Developer"), "iPhone Developer"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "application/export_method_debug", PROPERTY_HINT_ENUM, "App Store,Development,Ad-Hoc,Enterprise"), 1));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/provisioning_profile_uuid_release"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/code_sign_identity_release", PROPERTY_HINT_PLACEHOLDER_TEXT, "iPhone Distribution"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "application/export_method_release", PROPERTY_HINT_ENUM, "App Store,Development,Ad-Hoc,Enterprise"), 0));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/info"), "Made with Godot Engine"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/identifier", PROPERTY_HINT_PLACEHOLDER_TEXT, "com.example.game"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/signature"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/short_version"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/version"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/copyright"), ""));

	Vector<PluginConfigIOS> found_plugins = get_plugins();
	for (int i = 0; i < found_plugins.size(); i++) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "plugins/" + found_plugins[i].name), false));
	}

	plugins_changed = false;
	plugins = found_plugins;

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/access_wifi"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/push_notifications"), false));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "user_data/accessible_from_files_app"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "user_data/accessible_from_itunes_sharing"), false));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "privacy/camera_usage_description", PROPERTY_HINT_PLACEHOLDER_TEXT, "Provide a message if you need to use the camera"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "privacy/microphone_usage_description", PROPERTY_HINT_PLACEHOLDER_TEXT, "Provide a message if you need to use the microphone"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "privacy/photolibrary_usage_description", PROPERTY_HINT_PLACEHOLDER_TEXT, "Provide a message if you need access to the photo library"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "orientation/portrait"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "orientation/landscape_left"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "orientation/landscape_right"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "orientation/portrait_upside_down"), true));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "required_icons/iphone_120x120", PROPERTY_HINT_FILE, "*.png"), "")); // Home screen on iPhone/iPod Touch with retina display
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "required_icons/ipad_76x76", PROPERTY_HINT_FILE, "*.png"), "")); // Home screen on iPad
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "required_icons/app_store_1024x1024", PROPERTY_HINT_FILE, "*.png"), "")); // App Store

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "optional_icons/iphone_180x180", PROPERTY_HINT_FILE, "*.png"), "")); // Home screen on iPhone with retina HD display
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "optional_icons/ipad_152x152", PROPERTY_HINT_FILE, "*.png"), "")); // Home screen on iPad with retina display
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "optional_icons/ipad_167x167", PROPERTY_HINT_FILE, "*.png"), "")); // Home screen on iPad Pro
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "optional_icons/spotlight_40x40", PROPERTY_HINT_FILE, "*.png"), "")); // Spotlight
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "optional_icons/spotlight_80x80", PROPERTY_HINT_FILE, "*.png"), "")); // Spotlight on devices with retina display

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "storyboard/use_launch_screen_storyboard"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "storyboard/image_scale_mode", PROPERTY_HINT_ENUM, "Same as Logo,Center,Scale To Fit,Scale To Fill,Scale"), 0));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "storyboard/custom_image@2x", PROPERTY_HINT_FILE, "*.png"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "storyboard/custom_image@3x", PROPERTY_HINT_FILE, "*.png"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "storyboard/use_custom_bg_color"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::COLOR, "storyboard/custom_bg_color"), Color()));

	for (uint64_t i = 0; i < sizeof(loading_screen_infos) / sizeof(loading_screen_infos[0]); ++i) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, loading_screen_infos[i].preset_key, PROPERTY_HINT_FILE, "*.png"), ""));
	}
}

void EditorExportPlatformIOS::_fix_config_file(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &pfile, const IOSConfigData &p_config, bool p_debug) {
	static const String export_method_string[] = {
		"app-store",
		"development",
		"ad-hoc",
		"enterprise"
	};
	static const String storyboard_image_scale_mode[] = {
		"center",
		"scaleAspectFit",
		"scaleAspectFill",
		"scaleToFill"
	};
	String str;
	String strnew;
	str.parse_utf8((const char *)pfile.ptr(), pfile.size());
	Vector<String> lines = str.split("\n");
	for (int i = 0; i < lines.size(); i++) {
		if (lines[i].find("$binary") != -1) {
			strnew += lines[i].replace("$binary", p_config.binary_name) + "\n";
		} else if (lines[i].find("$modules_buildfile") != -1) {
			strnew += lines[i].replace("$modules_buildfile", p_config.modules_buildfile) + "\n";
		} else if (lines[i].find("$modules_fileref") != -1) {
			strnew += lines[i].replace("$modules_fileref", p_config.modules_fileref) + "\n";
		} else if (lines[i].find("$modules_buildphase") != -1) {
			strnew += lines[i].replace("$modules_buildphase", p_config.modules_buildphase) + "\n";
		} else if (lines[i].find("$modules_buildgrp") != -1) {
			strnew += lines[i].replace("$modules_buildgrp", p_config.modules_buildgrp) + "\n";
		} else if (lines[i].find("$name") != -1) {
			strnew += lines[i].replace("$name", p_config.pkg_name) + "\n";
		} else if (lines[i].find("$info") != -1) {
			strnew += lines[i].replace("$info", p_preset->get("application/info")) + "\n";
		} else if (lines[i].find("$identifier") != -1) {
			strnew += lines[i].replace("$identifier", p_preset->get("application/identifier")) + "\n";
		} else if (lines[i].find("$short_version") != -1) {
			strnew += lines[i].replace("$short_version", p_preset->get("application/short_version")) + "\n";
		} else if (lines[i].find("$version") != -1) {
			strnew += lines[i].replace("$version", p_preset->get("application/version")) + "\n";
		} else if (lines[i].find("$signature") != -1) {
			strnew += lines[i].replace("$signature", p_preset->get("application/signature")) + "\n";
		} else if (lines[i].find("$copyright") != -1) {
			strnew += lines[i].replace("$copyright", p_preset->get("application/copyright")) + "\n";
		} else if (lines[i].find("$team_id") != -1) {
			strnew += lines[i].replace("$team_id", p_preset->get("application/app_store_team_id")) + "\n";
		} else if (lines[i].find("$export_method") != -1) {
			int export_method = p_preset->get(p_debug ? "application/export_method_debug" : "application/export_method_release");
			strnew += lines[i].replace("$export_method", export_method_string[export_method]) + "\n";
		} else if (lines[i].find("$provisioning_profile_uuid_release") != -1) {
			strnew += lines[i].replace("$provisioning_profile_uuid_release", p_preset->get("application/provisioning_profile_uuid_release")) + "\n";
		} else if (lines[i].find("$provisioning_profile_uuid_debug") != -1) {
			strnew += lines[i].replace("$provisioning_profile_uuid_debug", p_preset->get("application/provisioning_profile_uuid_debug")) + "\n";
		} else if (lines[i].find("$provisioning_profile_uuid") != -1) {
			String uuid = p_debug ? p_preset->get("application/provisioning_profile_uuid_debug") : p_preset->get("application/provisioning_profile_uuid_release");
			strnew += lines[i].replace("$provisioning_profile_uuid", uuid) + "\n";
		} else if (lines[i].find("$code_sign_identity_debug") != -1) {
			strnew += lines[i].replace("$code_sign_identity_debug", p_preset->get("application/code_sign_identity_debug")) + "\n";
		} else if (lines[i].find("$code_sign_identity_release") != -1) {
			strnew += lines[i].replace("$code_sign_identity_release", p_preset->get("application/code_sign_identity_release")) + "\n";
		} else if (lines[i].find("$additional_plist_content") != -1) {
			strnew += lines[i].replace("$additional_plist_content", p_config.plist_content) + "\n";
		} else if (lines[i].find("$godot_archs") != -1) {
			strnew += lines[i].replace("$godot_archs", p_config.architectures) + "\n";
		} else if (lines[i].find("$linker_flags") != -1) {
			strnew += lines[i].replace("$linker_flags", p_config.linker_flags) + "\n";
		} else if (lines[i].find("$cpp_code") != -1) {
			strnew += lines[i].replace("$cpp_code", p_config.cpp_code) + "\n";
		} else if (lines[i].find("$docs_in_place") != -1) {
			strnew += lines[i].replace("$docs_in_place", ((bool)p_preset->get("user_data/accessible_from_files_app")) ? "<true/>" : "<false/>") + "\n";
		} else if (lines[i].find("$docs_sharing") != -1) {
			strnew += lines[i].replace("$docs_sharing", ((bool)p_preset->get("user_data/accessible_from_itunes_sharing")) ? "<true/>" : "<false/>") + "\n";
		} else if (lines[i].find("$entitlements_push_notifications") != -1) {
			bool is_on = p_preset->get("capabilities/push_notifications");
			strnew += lines[i].replace("$entitlements_push_notifications", is_on ? "<key>aps-environment</key><string>development</string>" : "") + "\n";
		} else if (lines[i].find("$required_device_capabilities") != -1) {
			String capabilities;

			// I've removed armv7 as we can run on 64bit only devices
			// Note that capabilities listed here are requirements for the app to be installed.
			// They don't enable anything.
			Vector<String> capabilities_list = p_config.capabilities;

			if ((bool)p_preset->get("capabilities/access_wifi") && capabilities_list.find("wifi") != -1) {
				capabilities_list.push_back("wifi");
			}

			for (int idx = 0; idx < capabilities_list.size(); idx++) {
				capabilities += "<string>" + capabilities_list[idx] + "</string>\n";
			}

			strnew += lines[i].replace("$required_device_capabilities", capabilities);
		} else if (lines[i].find("$interface_orientations") != -1) {
			String orientations;

			if ((bool)p_preset->get("orientation/portrait")) {
				orientations += "<string>UIInterfaceOrientationPortrait</string>\n";
			}
			if ((bool)p_preset->get("orientation/landscape_left")) {
				orientations += "<string>UIInterfaceOrientationLandscapeLeft</string>\n";
			}
			if ((bool)p_preset->get("orientation/landscape_right")) {
				orientations += "<string>UIInterfaceOrientationLandscapeRight</string>\n";
			}
			if ((bool)p_preset->get("orientation/portrait_upside_down")) {
				orientations += "<string>UIInterfaceOrientationPortraitUpsideDown</string>\n";
			}

			strnew += lines[i].replace("$interface_orientations", orientations);
		} else if (lines[i].find("$camera_usage_description") != -1) {
			String description = p_preset->get("privacy/camera_usage_description");
			strnew += lines[i].replace("$camera_usage_description", description) + "\n";
		} else if (lines[i].find("$microphone_usage_description") != -1) {
			String description = p_preset->get("privacy/microphone_usage_description");
			strnew += lines[i].replace("$microphone_usage_description", description) + "\n";
		} else if (lines[i].find("$photolibrary_usage_description") != -1) {
			String description = p_preset->get("privacy/photolibrary_usage_description");
			strnew += lines[i].replace("$photolibrary_usage_description", description) + "\n";
		} else if (lines[i].find("$plist_launch_screen_name") != -1) {
			bool is_on = p_preset->get("storyboard/use_launch_screen_storyboard");
			String value = is_on ? "<key>UILaunchStoryboardName</key>\n<string>Launch Screen</string>" : "";
			strnew += lines[i].replace("$plist_launch_screen_name", value) + "\n";
		} else if (lines[i].find("$pbx_launch_screen_file_reference") != -1) {
			bool is_on = p_preset->get("storyboard/use_launch_screen_storyboard");
			String value = is_on ? "90DD2D9D24B36E8000717FE1 = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = file.storyboard; path = \"Launch Screen.storyboard\"; sourceTree = \"<group>\"; };" : "";
			strnew += lines[i].replace("$pbx_launch_screen_file_reference", value) + "\n";
		} else if (lines[i].find("$pbx_launch_screen_copy_files") != -1) {
			bool is_on = p_preset->get("storyboard/use_launch_screen_storyboard");
			String value = is_on ? "90DD2D9D24B36E8000717FE1 /* Launch Screen.storyboard */," : "";
			strnew += lines[i].replace("$pbx_launch_screen_copy_files", value) + "\n";
		} else if (lines[i].find("$pbx_launch_screen_build_phase") != -1) {
			bool is_on = p_preset->get("storyboard/use_launch_screen_storyboard");
			String value = is_on ? "90DD2D9E24B36E8000717FE1 /* Launch Screen.storyboard in Resources */," : "";
			strnew += lines[i].replace("$pbx_launch_screen_build_phase", value) + "\n";
		} else if (lines[i].find("$pbx_launch_screen_build_reference") != -1) {
			bool is_on = p_preset->get("storyboard/use_launch_screen_storyboard");
			String value = is_on ? "90DD2D9E24B36E8000717FE1 /* Launch Screen.storyboard in Resources */ = {isa = PBXBuildFile; fileRef = 90DD2D9D24B36E8000717FE1 /* Launch Screen.storyboard */; };" : "";
			strnew += lines[i].replace("$pbx_launch_screen_build_reference", value) + "\n";
		} else if (lines[i].find("$pbx_launch_image_usage_setting") != -1) {
			bool is_on = p_preset->get("storyboard/use_launch_screen_storyboard");
			String value = is_on ? "" : "ASSETCATALOG_COMPILER_LAUNCHIMAGE_NAME = LaunchImage;";
			strnew += lines[i].replace("$pbx_launch_image_usage_setting", value) + "\n";
		} else if (lines[i].find("$launch_screen_image_mode") != -1) {
			int image_scale_mode = p_preset->get("storyboard/image_scale_mode");
			String value;

			switch (image_scale_mode) {
				case 0: {
					String logo_path = ProjectSettings::get_singleton()->get("application/boot_splash/image");
					bool is_on = ProjectSettings::get_singleton()->get("application/boot_splash/fullsize");
					// If custom logo is not specified, Godot does not scale default one, so we should do the same.
					value = (is_on && logo_path.length() > 0) ? "scaleAspectFit" : "center";
				} break;
				default: {
					value = storyboard_image_scale_mode[image_scale_mode - 1];
				}
			}

			strnew += lines[i].replace("$launch_screen_image_mode", value) + "\n";
		} else if (lines[i].find("$launch_screen_background_color") != -1) {
			bool use_custom = p_preset->get("storyboard/use_custom_bg_color");
			Color color = use_custom ? p_preset->get("storyboard/custom_bg_color") : ProjectSettings::get_singleton()->get("application/boot_splash/bg_color");
			const String value_format = "red=\"$red\" green=\"$green\" blue=\"$blue\" alpha=\"$alpha\"";

			Dictionary value_dictionary;
			value_dictionary["red"] = color.r;
			value_dictionary["green"] = color.g;
			value_dictionary["blue"] = color.b;
			value_dictionary["alpha"] = color.a;
			String value = value_format.format(value_dictionary, "$_");

			strnew += lines[i].replace("$launch_screen_background_color", value) + "\n";
		} else {
			strnew += lines[i] + "\n";
		}
	}

	// !BAS! I'm assuming the 9 in the original code was a typo. I've added -1 or else it seems to also be adding our terminating zero...
	// should apply the same fix in our OSX export.
	CharString cs = strnew.utf8();
	pfile.resize(cs.size() - 1);
	for (int i = 0; i < cs.size() - 1; i++) {
		pfile.write[i] = cs[i];
	}
}

String EditorExportPlatformIOS::_get_additional_plist_content() {
	Vector<Ref<EditorExportPlugin> > export_plugins = EditorExport::get_singleton()->get_export_plugins();
	String result;
	for (int i = 0; i < export_plugins.size(); ++i) {
		result += export_plugins[i]->get_ios_plist_content();
	}
	return result;
}

String EditorExportPlatformIOS::_get_linker_flags() {
	Vector<Ref<EditorExportPlugin> > export_plugins = EditorExport::get_singleton()->get_export_plugins();
	String result;
	for (int i = 0; i < export_plugins.size(); ++i) {
		String flags = export_plugins[i]->get_ios_linker_flags();
		if (flags.length() == 0) continue;
		if (result.length() > 0) {
			result += ' ';
		}
		result += flags;
	}
	// the flags will be enclosed in quotes, so need to escape them
	return result.replace("\"", "\\\"");
}

String EditorExportPlatformIOS::_get_cpp_code() {
	Vector<Ref<EditorExportPlugin> > export_plugins = EditorExport::get_singleton()->get_export_plugins();
	String result;
	for (int i = 0; i < export_plugins.size(); ++i) {
		result += export_plugins[i]->get_ios_cpp_code();
	}
	return result;
}

struct IconInfo {
	const char *preset_key;
	const char *idiom;
	const char *export_name;
	const char *actual_size_side;
	const char *scale;
	const char *unscaled_size;
	bool is_required;
};

static const IconInfo icon_infos[] = {
	{ "required_icons/iphone_120x120", "iphone", "Icon-120.png", "120", "2x", "60x60", true },
	{ "required_icons/iphone_120x120", "iphone", "Icon-120.png", "120", "3x", "40x40", true },

	{ "required_icons/ipad_76x76", "ipad", "Icon-76.png", "76", "1x", "76x76", false },
	{ "required_icons/app_store_1024x1024", "ios-marketing", "Icon-1024.png", "1024", "1x", "1024x1024", false },

	{ "optional_icons/iphone_180x180", "iphone", "Icon-180.png", "180", "3x", "60x60", false },

	{ "optional_icons/ipad_152x152", "ipad", "Icon-152.png", "152", "2x", "76x76", false },

	{ "optional_icons/ipad_167x167", "ipad", "Icon-167.png", "167", "2x", "83.5x83.5", false },

	{ "optional_icons/spotlight_40x40", "ipad", "Icon-40.png", "40", "1x", "40x40", false },

	{ "optional_icons/spotlight_80x80", "iphone", "Icon-80.png", "80", "2x", "40x40", false },
	{ "optional_icons/spotlight_80x80", "ipad", "Icon-80.png", "80", "2x", "40x40", false }

};

Error EditorExportPlatformIOS::_export_icons(const Ref<EditorExportPreset> &p_preset, const String &p_iconset_dir) {
	String json_description = "{\"images\":[";
	String sizes;

	DirAccess *da = DirAccess::open(p_iconset_dir);
	ERR_FAIL_COND_V_MSG(!da, ERR_CANT_OPEN, "Cannot open directory '" + p_iconset_dir + "'.");

	for (uint64_t i = 0; i < (sizeof(icon_infos) / sizeof(icon_infos[0])); ++i) {
		IconInfo info = icon_infos[i];
		String icon_path = p_preset->get(info.preset_key);
		if (icon_path.length() == 0) {
			if (info.is_required) {
				ERR_PRINT("Required icon is not specified in the preset");
				return ERR_UNCONFIGURED;
			}
			continue;
		}
		Error err = da->copy(icon_path, p_iconset_dir + info.export_name);
		if (err) {
			memdelete(da);
			String err_str = String("Failed to export icon: ") + icon_path;
			ERR_PRINT(err_str.utf8().get_data());
			return err;
		}
		sizes += String(info.actual_size_side) + "\n";
		if (i > 0) {
			json_description += ",";
		}
		json_description += String("{");
		json_description += String("\"idiom\":") + "\"" + info.idiom + "\",";
		json_description += String("\"size\":") + "\"" + info.unscaled_size + "\",";
		json_description += String("\"scale\":") + "\"" + info.scale + "\",";
		json_description += String("\"filename\":") + "\"" + info.export_name + "\"";
		json_description += String("}");
	}
	json_description += "]}";
	memdelete(da);

	FileAccess *json_file = FileAccess::open(p_iconset_dir + "Contents.json", FileAccess::WRITE);
	ERR_FAIL_COND_V(!json_file, ERR_CANT_CREATE);
	CharString json_utf8 = json_description.utf8();
	json_file->store_buffer((const uint8_t *)json_utf8.get_data(), json_utf8.length());
	memdelete(json_file);

	FileAccess *sizes_file = FileAccess::open(p_iconset_dir + "sizes", FileAccess::WRITE);
	ERR_FAIL_COND_V(!sizes_file, ERR_CANT_CREATE);
	CharString sizes_utf8 = sizes.utf8();
	sizes_file->store_buffer((const uint8_t *)sizes_utf8.get_data(), sizes_utf8.length());
	memdelete(sizes_file);

	return OK;
}

Error EditorExportPlatformIOS::_export_loading_screen_file(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir) {
	const String custom_launch_image_2x = p_preset->get("storyboard/custom_image@2x");
	const String custom_launch_image_3x = p_preset->get("storyboard/custom_image@3x");

	if (custom_launch_image_2x.length() > 0 && custom_launch_image_3x.length() > 0) {
		Ref<Image> image;
		String image_path = p_dest_dir.plus_file("splash@2x.png");
		image.instance();
		Error err = image->load(custom_launch_image_2x);

		if (err) {
			image.unref();
			return err;
		}

		if (image->save_png(image_path) != OK) {
			return ERR_FILE_CANT_WRITE;
		}

		image.unref();
		image_path = p_dest_dir.plus_file("splash@3x.png");
		image.instance();
		err = image->load(custom_launch_image_3x);

		if (err) {
			image.unref();
			return err;
		}

		if (image->save_png(image_path) != OK) {
			return ERR_FILE_CANT_WRITE;
		}
	} else {
		Ref<Image> splash;

		const String splash_path = ProjectSettings::get_singleton()->get("application/boot_splash/image");

		if (!splash_path.empty()) {
			splash.instance();
			const Error err = splash->load(splash_path);
			if (err) {
				splash.unref();
			}
		}

		if (splash.is_null()) {
			splash = Ref<Image>(memnew(Image(boot_splash_png)));
		}

		// Using same image for both @2x and @3x
		// because Godot's own boot logo uses single image for all resolutions.
		// Also not using @1x image, because devices using this image variant
		// are not supported by iOS 9, which is minimal target.
		const String splash_png_path_2x = p_dest_dir.plus_file("splash@2x.png");
		const String splash_png_path_3x = p_dest_dir.plus_file("splash@3x.png");

		if (splash->save_png(splash_png_path_2x) != OK) {
			return ERR_FILE_CANT_WRITE;
		}

		if (splash->save_png(splash_png_path_3x) != OK) {
			return ERR_FILE_CANT_WRITE;
		}
	}

	return OK;
}

Error EditorExportPlatformIOS::_export_loading_screen_images(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir) {
	DirAccess *da = DirAccess::open(p_dest_dir);
	ERR_FAIL_COND_V_MSG(!da, ERR_CANT_OPEN, "Cannot open directory '" + p_dest_dir + "'.");

	for (uint64_t i = 0; i < sizeof(loading_screen_infos) / sizeof(loading_screen_infos[0]); ++i) {
		LoadingScreenInfo info = loading_screen_infos[i];
		String loading_screen_file = p_preset->get(info.preset_key);
		if (loading_screen_file.size() > 0) {
			Error err = da->copy(loading_screen_file, p_dest_dir + info.export_name);
			if (err) {
				memdelete(da);
				String err_str = String("Failed to export loading screen (") + info.preset_key + ") from path '" + loading_screen_file + "'.";
				ERR_PRINT(err_str.utf8().get_data());
				return err;
			}
		}
	}
	memdelete(da);

	return OK;
}

Error EditorExportPlatformIOS::_walk_dir_recursive(DirAccess *p_da, FileHandler p_handler, void *p_userdata) {
	Vector<String> dirs;
	String path;
	String current_dir = p_da->get_current_dir();
	p_da->list_dir_begin();
	while ((path = p_da->get_next()).length() != 0) {
		if (p_da->current_is_dir()) {
			if (path != "." && path != "..") {
				dirs.push_back(path);
			}
		} else {
			Error err = p_handler(current_dir.plus_file(path), p_userdata);
			if (err) {
				p_da->list_dir_end();
				return err;
			}
		}
	}
	p_da->list_dir_end();

	for (int i = 0; i < dirs.size(); ++i) {
		String dir = dirs[i];
		p_da->change_dir(dir);
		Error err = _walk_dir_recursive(p_da, p_handler, p_userdata);
		p_da->change_dir("..");
		if (err) {
			return err;
		}
	}

	return OK;
}

struct CodesignData {
	const Ref<EditorExportPreset> &preset;
	bool debug;

	CodesignData(const Ref<EditorExportPreset> &p_preset, bool p_debug) :
			preset(p_preset),
			debug(p_debug) {
	}
};

Error EditorExportPlatformIOS::_codesign(String p_file, void *p_userdata) {
	if (p_file.ends_with(".dylib")) {
		CodesignData *data = (CodesignData *)p_userdata;
		print_line(String("Signing ") + p_file);
		List<String> codesign_args;
		codesign_args.push_back("-f");
		codesign_args.push_back("-s");
		codesign_args.push_back(data->preset->get(data->debug ? "application/code_sign_identity_debug" : "application/code_sign_identity_release"));
		codesign_args.push_back(p_file);
		return OS::get_singleton()->execute("codesign", codesign_args, true);
	}
	return OK;
}

struct PbxId {
private:
	static char _hex_char(uint8_t four_bits) {
		if (four_bits < 10) {
			return ('0' + four_bits);
		}
		return 'A' + (four_bits - 10);
	}

	static String _hex_pad(uint32_t num) {
		Vector<char> ret;
		ret.resize(sizeof(num) * 2);
		for (uint64_t i = 0; i < sizeof(num) * 2; ++i) {
			uint8_t four_bits = (num >> (sizeof(num) * 8 - (i + 1) * 4)) & 0xF;
			ret.write[i] = _hex_char(four_bits);
		}
		return String::utf8(ret.ptr(), ret.size());
	}

public:
	uint32_t high_bits;
	uint32_t mid_bits;
	uint32_t low_bits;

	String str() const {
		return _hex_pad(high_bits) + _hex_pad(mid_bits) + _hex_pad(low_bits);
	}

	PbxId &operator++() {
		low_bits++;
		if (!low_bits) {
			mid_bits++;
			if (!mid_bits) {
				high_bits++;
			}
		}

		return *this;
	}
};

struct ExportLibsData {
	Vector<String> lib_paths;
	String dest_dir;
};

void EditorExportPlatformIOS::_add_assets_to_project(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &p_project_data, const Vector<IOSExportAsset> &p_additional_assets) {
	// that is just a random number, we just need Godot IDs not to clash with
	// existing IDs in the project.
	PbxId current_id = { 0x58938401, 0, 0 };
	String pbx_files;
	String pbx_frameworks_build;
	String pbx_frameworks_refs;
	String pbx_resources_build;
	String pbx_resources_refs;
	String pbx_embeded_frameworks;

	const String file_info_format = String("$build_id = {isa = PBXBuildFile; fileRef = $ref_id; };\n") +
									"$ref_id = {isa = PBXFileReference; lastKnownFileType = $file_type; name = \"$name\"; path = \"$file_path\"; sourceTree = \"<group>\"; };\n";

	for (int i = 0; i < p_additional_assets.size(); ++i) {
		String additional_asset_info_format = file_info_format;

		String build_id = (++current_id).str();
		String ref_id = (++current_id).str();
		String framework_id = "";

		const IOSExportAsset &asset = p_additional_assets[i];

		String type;
		if (asset.exported_path.ends_with(".framework")) {
			if (asset.should_embed) {
				additional_asset_info_format += "$framework_id = {isa = PBXBuildFile; fileRef = $ref_id; settings = {ATTRIBUTES = (CodeSignOnCopy, ); }; };\n";
				framework_id = (++current_id).str();
				pbx_embeded_frameworks += framework_id + ",\n";
			}

			type = "wrapper.framework";
		} else if (asset.exported_path.ends_with(".xcframework")) {
			if (asset.should_embed) {
				additional_asset_info_format += "$framework_id = {isa = PBXBuildFile; fileRef = $ref_id; settings = {ATTRIBUTES = (CodeSignOnCopy, ); }; };\n";
				framework_id = (++current_id).str();
				pbx_embeded_frameworks += framework_id + ",\n";
			}

			type = "wrapper.xcframework";
		} else if (asset.exported_path.ends_with(".dylib")) {
			type = "compiled.mach-o.dylib";
		} else if (asset.exported_path.ends_with(".a")) {
			type = "archive.ar";
		} else {
			type = "file";
		}

		String &pbx_build = asset.is_framework ? pbx_frameworks_build : pbx_resources_build;
		String &pbx_refs = asset.is_framework ? pbx_frameworks_refs : pbx_resources_refs;

		if (pbx_build.length() > 0) {
			pbx_build += ",\n";
			pbx_refs += ",\n";
		}
		pbx_build += build_id;
		pbx_refs += ref_id;

		Dictionary format_dict;
		format_dict["build_id"] = build_id;
		format_dict["ref_id"] = ref_id;
		format_dict["name"] = asset.exported_path.get_file();
		format_dict["file_path"] = asset.exported_path;
		format_dict["file_type"] = type;
		if (framework_id.length() > 0) {
			format_dict["framework_id"] = framework_id;
		}
		pbx_files += additional_asset_info_format.format(format_dict, "$_");
	}

	// Note, frameworks like gamekit are always included in our project.pbxprof file
	// even if turned off in capabilities.

	String str = String::utf8((const char *)p_project_data.ptr(), p_project_data.size());
	str = str.replace("$additional_pbx_files", pbx_files);
	str = str.replace("$additional_pbx_frameworks_build", pbx_frameworks_build);
	str = str.replace("$additional_pbx_frameworks_refs", pbx_frameworks_refs);
	str = str.replace("$additional_pbx_resources_build", pbx_resources_build);
	str = str.replace("$additional_pbx_resources_refs", pbx_resources_refs);
	str = str.replace("$pbx_embeded_frameworks", pbx_embeded_frameworks);

	CharString cs = str.utf8();
	p_project_data.resize(cs.size() - 1);
	for (int i = 0; i < cs.size() - 1; i++) {
		p_project_data.write[i] = cs[i];
	}
}

Error EditorExportPlatformIOS::_copy_asset(const String &p_out_dir, const String &p_asset, const String *p_custom_file_name, bool p_is_framework, bool p_should_embed, Vector<IOSExportAsset> &r_exported_assets) {
	DirAccess *filesystem_da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	ERR_FAIL_COND_V_MSG(!filesystem_da, ERR_CANT_CREATE, "Cannot create DirAccess for path '" + p_out_dir + "'.");

	String binary_name = p_out_dir.get_file().get_basename();

	DirAccess *da = DirAccess::create_for_path(p_asset);
	if (!da) {
		memdelete(filesystem_da);
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Can't create directory: " + p_asset + ".");
	}
	bool file_exists = da->file_exists(p_asset);
	bool dir_exists = da->dir_exists(p_asset);
	if (!file_exists && !dir_exists) {
		memdelete(da);
		memdelete(filesystem_da);
		return ERR_FILE_NOT_FOUND;
	}

	String base_dir = p_asset.get_base_dir().replace("res://", "");
	String destination_dir;
	String destination;
	String asset_path;

	bool create_framework = false;

	if (p_is_framework && p_asset.ends_with(".dylib")) {
		// For iOS we need to turn .dylib into .framework
		// to be able to send application to AppStore
		asset_path = String("dylibs").plus_file(base_dir);

		String file_name;

		if (!p_custom_file_name) {
			file_name = p_asset.get_basename().get_file();
		} else {
			file_name = *p_custom_file_name;
		}

		String framework_name = file_name + ".framework";

		asset_path = asset_path.plus_file(framework_name);
		destination_dir = p_out_dir.plus_file(asset_path);
		destination = destination_dir.plus_file(file_name);
		create_framework = true;
	} else if (p_is_framework && (p_asset.ends_with(".framework") || p_asset.ends_with(".xcframework"))) {
		asset_path = String("dylibs").plus_file(base_dir);

		String file_name;

		if (!p_custom_file_name) {
			file_name = p_asset.get_file();
		} else {
			file_name = *p_custom_file_name;
		}

		asset_path = asset_path.plus_file(file_name);
		destination_dir = p_out_dir.plus_file(asset_path);
		destination = destination_dir;
	} else {
		asset_path = base_dir;

		String file_name;

		if (!p_custom_file_name) {
			file_name = p_asset.get_file();
		} else {
			file_name = *p_custom_file_name;
		}

		destination_dir = p_out_dir.plus_file(asset_path);
		asset_path = asset_path.plus_file(file_name);
		destination = p_out_dir.plus_file(asset_path);
	}

	if (!filesystem_da->dir_exists(destination_dir)) {
		Error make_dir_err = filesystem_da->make_dir_recursive(destination_dir);
		if (make_dir_err) {
			memdelete(da);
			memdelete(filesystem_da);
			return make_dir_err;
		}
	}

	Error err = dir_exists ? da->copy_dir(p_asset, destination) : da->copy(p_asset, destination);
	memdelete(da);
	if (err) {
		memdelete(filesystem_da);
		return err;
	}
	IOSExportAsset exported_asset = { binary_name.plus_file(asset_path), p_is_framework, p_should_embed };
	r_exported_assets.push_back(exported_asset);

	if (create_framework) {
		String file_name;

		if (!p_custom_file_name) {
			file_name = p_asset.get_basename().get_file();
		} else {
			file_name = *p_custom_file_name;
		}

		String framework_name = file_name + ".framework";

		// Performing `install_name_tool -id @rpath/{name}.framework/{name} ./{name}` on dylib
		{
			List<String> install_name_args;
			install_name_args.push_back("-id");
			install_name_args.push_back(String("@rpath").plus_file(framework_name).plus_file(file_name));
			install_name_args.push_back(destination);

			OS::get_singleton()->execute("install_name_tool", install_name_args, true);
		}

		// Creating Info.plist
		{
			String info_plist_format = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
									   "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
									   "<plist version=\"1.0\">\n"
									   "<dict>\n"
									   "<key>CFBundleShortVersionString</key>\n"
									   "<string>1.0</string>\n"
									   "<key>CFBundleIdentifier</key>\n"
									   "<string>com.gdnative.framework.$name</string>\n"
									   "<key>CFBundleName</key>\n"
									   "<string>$name</string>\n"
									   "<key>CFBundleExecutable</key>\n"
									   "<string>$name</string>\n"
									   "<key>DTPlatformName</key>\n"
									   "<string>iphoneos</string>\n"
									   "<key>CFBundleInfoDictionaryVersion</key>\n"
									   "<string>6.0</string>\n"
									   "<key>CFBundleVersion</key>\n"
									   "<string>1</string>\n"
									   "<key>CFBundlePackageType</key>\n"
									   "<string>FMWK</string>\n"
									   "<key>MinimumOSVersion</key>\n"
									   "<string>10.0</string>\n"
									   "</dict>\n"
									   "</plist>";

			String info_plist = info_plist_format.replace("$name", file_name);

			FileAccess *f = FileAccess::open(destination_dir.plus_file("Info.plist"), FileAccess::WRITE);
			if (f) {
				f->store_string(info_plist);
				f->close();
				memdelete(f);
			}
		}
	}

	memdelete(filesystem_da);

	return OK;
}

Error EditorExportPlatformIOS::_export_additional_assets(const String &p_out_dir, const Vector<String> &p_assets, bool p_is_framework, bool p_should_embed, Vector<IOSExportAsset> &r_exported_assets) {
	for (int f_idx = 0; f_idx < p_assets.size(); ++f_idx) {
		String asset = p_assets[f_idx];
		if (!asset.begins_with("res://")) {
			// either SDK-builtin or already a part of the export template
			IOSExportAsset exported_asset = { asset, p_is_framework, p_should_embed };
			r_exported_assets.push_back(exported_asset);
		} else {
			Error err = _copy_asset(p_out_dir, asset, nullptr, p_is_framework, p_should_embed, r_exported_assets);
			ERR_FAIL_COND_V(err, err);
		}
	}

	return OK;
}

Error EditorExportPlatformIOS::_export_additional_assets(const String &p_out_dir, const Vector<SharedObject> &p_libraries, Vector<IOSExportAsset> &r_exported_assets) {
	Vector<Ref<EditorExportPlugin> > export_plugins = EditorExport::get_singleton()->get_export_plugins();
	for (int i = 0; i < export_plugins.size(); i++) {
		Vector<String> linked_frameworks = export_plugins[i]->get_ios_frameworks();
		Error err = _export_additional_assets(p_out_dir, linked_frameworks, true, false, r_exported_assets);
		ERR_FAIL_COND_V(err, err);

		Vector<String> embedded_frameworks = export_plugins[i]->get_ios_embedded_frameworks();
		err = _export_additional_assets(p_out_dir, embedded_frameworks, true, true, r_exported_assets);
		ERR_FAIL_COND_V(err, err);

		Vector<String> project_static_libs = export_plugins[i]->get_ios_project_static_libs();
		for (int j = 0; j < project_static_libs.size(); j++)
			project_static_libs.write[j] = project_static_libs[j].get_file(); // Only the file name as it's copied to the project
		err = _export_additional_assets(p_out_dir, project_static_libs, true, true, r_exported_assets);
		ERR_FAIL_COND_V(err, err);

		Vector<String> ios_bundle_files = export_plugins[i]->get_ios_bundle_files();
		err = _export_additional_assets(p_out_dir, ios_bundle_files, false, false, r_exported_assets);
		ERR_FAIL_COND_V(err, err);
	}

	Vector<String> library_paths;
	for (int i = 0; i < p_libraries.size(); ++i) {
		library_paths.push_back(p_libraries[i].path);
	}
	Error err = _export_additional_assets(p_out_dir, library_paths, true, true, r_exported_assets);
	ERR_FAIL_COND_V(err, err);

	return OK;
}

Vector<String> EditorExportPlatformIOS::_get_preset_architectures(const Ref<EditorExportPreset> &p_preset) {
	Vector<ExportArchitecture> all_archs = _get_supported_architectures();
	Vector<String> enabled_archs;
	for (int i = 0; i < all_archs.size(); ++i) {
		bool is_enabled = p_preset->get("architectures/" + all_archs[i].name);
		if (is_enabled) {
			enabled_archs.push_back(all_archs[i].name);
		}
	}
	return enabled_archs;
}

Error EditorExportPlatformIOS::_export_ios_plugins(const Ref<EditorExportPreset> &p_preset, IOSConfigData &p_config_data, const String &dest_dir, Vector<IOSExportAsset> &r_exported_assets, bool p_debug) {
	String plugin_definition_cpp_code;
	String plugin_initialization_cpp_code;
	String plugin_deinitialization_cpp_code;

	Vector<String> plugin_linked_dependencies;
	Vector<String> plugin_embedded_dependencies;
	Vector<String> plugin_files;

	Vector<PluginConfigIOS> enabled_plugins = get_enabled_plugins(p_preset);

	Vector<String> added_linked_dependenciy_names;
	Vector<String> added_embedded_dependenciy_names;
	HashMap<String, String> plist_values;

	Error err;

	for (int i = 0; i < enabled_plugins.size(); i++) {
		PluginConfigIOS plugin = enabled_plugins[i];

		// Export plugin binary.
		String plugin_main_binary = get_plugin_main_binary(plugin, p_debug);
		String plugin_binary_result_file = plugin.binary.get_file();
		// We shouldn't embed .xcframework that contains static libraries.
		// Static libraries are not embedded anyway.
		err = _copy_asset(dest_dir, plugin_main_binary, &plugin_binary_result_file, true, false, r_exported_assets);

		ERR_FAIL_COND_V(err, err);

		// Adding dependencies.
		// Use separate container for names to check for duplicates.
		for (int j = 0; j < plugin.linked_dependencies.size(); j++) {
			String dependency = plugin.linked_dependencies[j];
			String name = dependency.get_file();

			if (added_linked_dependenciy_names.find(name) != -1) {
				continue;
			}

			added_linked_dependenciy_names.push_back(name);
			plugin_linked_dependencies.push_back(dependency);
		}

		for (int j = 0; j < plugin.system_dependencies.size(); j++) {
			String dependency = plugin.system_dependencies[j];
			String name = dependency.get_file();

			if (added_linked_dependenciy_names.find(name) != -1) {
				continue;
			}

			added_linked_dependenciy_names.push_back(name);
			plugin_linked_dependencies.push_back(dependency);
		}

		for (int j = 0; j < plugin.embedded_dependencies.size(); j++) {
			String dependency = plugin.embedded_dependencies[j];
			String name = dependency.get_file();

			if (added_embedded_dependenciy_names.find(name) != -1) {
				continue;
			}

			added_embedded_dependenciy_names.push_back(name);
			plugin_embedded_dependencies.push_back(dependency);
		}

		plugin_files.append_array(plugin.files_to_copy);

		// Capabilities
		// Also checking for duplicates.
		for (int j = 0; j < plugin.capabilities.size(); j++) {
			String capability = plugin.capabilities[j];

			if (p_config_data.capabilities.find(capability) != -1) {
				continue;
			}

			p_config_data.capabilities.push_back(capability);
		}

		// Plist
		// Using hash map container to remove duplicates
		const String *K = nullptr;

		while ((K = plugin.plist.next(K))) {
			String key = *K;
			String value = plugin.plist[key];

			if (key.empty() || value.empty()) {
				continue;
			}

			plist_values[key] = value;
		}

		// CPP Code
		String definition_comment = "// Plugin: " + plugin.name + "\n";
		String initialization_method = plugin.initialization_method + "();\n";
		String deinitialization_method = plugin.deinitialization_method + "();\n";

		plugin_definition_cpp_code += definition_comment +
									  "extern void " + initialization_method +
									  "extern void " + deinitialization_method + "\n";

		plugin_initialization_cpp_code += "\t" + initialization_method;
		plugin_deinitialization_cpp_code += "\t" + deinitialization_method;
	}

	// Updating `Info.plist`
	{
		const String *K = nullptr;
		while ((K = plist_values.next(K))) {
			String key = *K;
			String value = plist_values[key];

			if (key.empty() || value.empty()) {
				continue;
			}

			p_config_data.plist_content += "<key>" + key + "</key><string>" + value + "</string>\n";
		}
	}

	// Export files
	{
		// Export linked plugin dependency
		err = _export_additional_assets(dest_dir, plugin_linked_dependencies, true, false, r_exported_assets);
		ERR_FAIL_COND_V(err, err);

		// Export embedded plugin dependency
		err = _export_additional_assets(dest_dir, plugin_embedded_dependencies, true, true, r_exported_assets);
		ERR_FAIL_COND_V(err, err);

		// Export plugin files
		err = _export_additional_assets(dest_dir, plugin_files, false, false, r_exported_assets);
		ERR_FAIL_COND_V(err, err);
	}

	// Update CPP
	{
		Dictionary plugin_format;
		plugin_format["definition"] = plugin_definition_cpp_code;
		plugin_format["initialization"] = plugin_initialization_cpp_code;
		plugin_format["deinitialization"] = plugin_deinitialization_cpp_code;

		String plugin_cpp_code = "\n// Godot Plugins\n"
								 "void godot_ios_plugins_initialize();\n"
								 "void godot_ios_plugins_deinitialize();\n"
								 "// Exported Plugins\n\n"
								 "$definition"
								 "// Use Plugins\n"
								 "void godot_ios_plugins_initialize() {\n"
								 "$initialization"
								 "}\n\n"
								 "void godot_ios_plugins_deinitialize() {\n"
								 "$deinitialization"
								 "}\n";

		p_config_data.cpp_code += plugin_cpp_code.format(plugin_format, "$_");
	}
	return OK;
}

Error EditorExportPlatformIOS::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

	String src_pkg_name;
	String dest_dir = p_path.get_base_dir() + "/";
	String binary_name = p_path.get_file().get_basename();

	EditorProgress ep("export", "Exporting for iOS", 5, true);

	String team_id = p_preset->get("application/app_store_team_id");
	ERR_FAIL_COND_V_MSG(team_id.length() == 0, ERR_CANT_OPEN, "App Store Team ID not specified - cannot configure the project.");

	if (p_debug)
		src_pkg_name = p_preset->get("custom_template/debug");
	else
		src_pkg_name = p_preset->get("custom_template/release");

	if (src_pkg_name == "") {
		String err;
		src_pkg_name = find_export_template("iphone.zip", &err);
		if (src_pkg_name == "") {
			EditorNode::add_io_error(err);
			return ERR_FILE_NOT_FOUND;
		}
	}

	if (!DirAccess::exists(dest_dir)) {
		return ERR_FILE_BAD_PATH;
	}

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (da) {
		String current_dir = da->get_current_dir();

		// remove leftovers from last export so they don't interfere
		// in case some files are no longer needed
		if (da->change_dir(dest_dir + binary_name + ".xcodeproj") == OK) {
			da->erase_contents_recursive();
		}
		if (da->change_dir(dest_dir + binary_name) == OK) {
			da->erase_contents_recursive();
		}

		da->change_dir(current_dir);

		if (!da->dir_exists(dest_dir + binary_name)) {
			Error err = da->make_dir(dest_dir + binary_name);
			if (err) {
				memdelete(da);
				return err;
			}
		}
		memdelete(da);
	}

	if (ep.step("Making .pck", 0)) {
		return ERR_SKIP;
	}
	String pack_path = dest_dir + binary_name + ".pck";
	Vector<SharedObject> libraries;
	Error err = save_pack(p_preset, pack_path, &libraries);
	if (err)
		return err;

	if (ep.step("Extracting and configuring Xcode project", 1)) {
		return ERR_SKIP;
	}

	String library_to_use = "libgodot.iphone." + String(p_debug ? "debug" : "release") + ".xcframework";

	print_line("Static framework: " + library_to_use);
	String pkg_name;
	if (p_preset->get("application/name") != "")
		pkg_name = p_preset->get("application/name"); // app_name
	else if (String(ProjectSettings::get_singleton()->get("application/config/name")) != "")
		pkg_name = String(ProjectSettings::get_singleton()->get("application/config/name"));
	else
		pkg_name = "Unnamed";

	bool found_library = false;
	int total_size = 0;

	const String project_file = "godot_ios.xcodeproj/project.pbxproj";
	Set<String> files_to_parse;
	files_to_parse.insert("godot_ios/godot_ios-Info.plist");
	files_to_parse.insert(project_file);
	files_to_parse.insert("godot_ios/export_options.plist");
	files_to_parse.insert("godot_ios/dummy.cpp");
	files_to_parse.insert("godot_ios.xcodeproj/project.xcworkspace/contents.xcworkspacedata");
	files_to_parse.insert("godot_ios.xcodeproj/xcshareddata/xcschemes/godot_ios.xcscheme");
	files_to_parse.insert("godot_ios/godot_ios.entitlements");
	files_to_parse.insert("godot_ios/Launch Screen.storyboard");

	IOSConfigData config_data = {
		pkg_name,
		binary_name,
		_get_additional_plist_content(),
		String(" ").join(_get_preset_architectures(p_preset)),
		_get_linker_flags(),
		_get_cpp_code(),
		"",
		"",
		"",
		"",
		Vector<String>()
	};

	Vector<IOSExportAsset> assets;

	DirAccess *tmp_app_path = DirAccess::create_for_path(dest_dir);
	ERR_FAIL_COND_V(!tmp_app_path, ERR_CANT_CREATE);

	print_line("Unzipping...");
	FileAccess *src_f = NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);
	unzFile src_pkg_zip = unzOpen2(src_pkg_name.utf8().get_data(), &io);
	if (!src_pkg_zip) {
		EditorNode::add_io_error("Could not open export template (not a zip file?):\n" + src_pkg_name);
		return ERR_CANT_OPEN;
	}

	err = _export_ios_plugins(p_preset, config_data, dest_dir + binary_name, assets, p_debug);
	ERR_FAIL_COND_V(err, err);

	//export rest of the files
	int ret = unzGoToFirstFile(src_pkg_zip);
	Vector<uint8_t> project_file_data;
	while (ret == UNZ_OK) {
#if defined(OSX_ENABLED) || defined(X11_ENABLED)
		bool is_execute = false;
#endif

		//get filename
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(src_pkg_zip, &info, fname, 16384, NULL, 0, NULL, 0);

		String file = fname;

		print_line("READ: " + file);
		Vector<uint8_t> data;
		data.resize(info.uncompressed_size);

		//read
		unzOpenCurrentFile(src_pkg_zip);
		unzReadCurrentFile(src_pkg_zip, data.ptrw(), data.size());
		unzCloseCurrentFile(src_pkg_zip);

		//write

		file = file.replace_first("iphone/", "");

		if (files_to_parse.has(file)) {
			_fix_config_file(p_preset, data, config_data, p_debug);
		} else if (file.begins_with("libgodot.iphone")) {
			if (!file.begins_with(library_to_use) || file.ends_with(String("/empty"))) {
				ret = unzGoToNextFile(src_pkg_zip);
				continue; //ignore!
			}
			found_library = true;
#if defined(OSX_ENABLED) || defined(X11_ENABLED)
			is_execute = true;
#endif
			file = file.replace(library_to_use, binary_name + ".xcframework");
		}

		if (file == project_file) {
			project_file_data = data;
		}

		///@TODO need to parse logo files

		if (data.size() > 0) {
			file = file.replace("godot_ios", binary_name);

			print_line("ADDING: " + file + " size: " + itos(data.size()));
			total_size += data.size();

			/* write it into our folder structure */
			file = dest_dir + file;

			/* make sure this folder exists */
			String dir_name = file.get_base_dir();
			if (!tmp_app_path->dir_exists(dir_name)) {
				print_line("Creating " + dir_name);
				Error dir_err = tmp_app_path->make_dir_recursive(dir_name);
				if (dir_err) {
					ERR_PRINTS("Can't create '" + dir_name + "'.");
					unzClose(src_pkg_zip);
					memdelete(tmp_app_path);
					return ERR_CANT_CREATE;
				}
			}

			/* write the file */
			FileAccess *f = FileAccess::open(file, FileAccess::WRITE);
			if (!f) {
				ERR_PRINTS("Can't write '" + file + "'.");
				unzClose(src_pkg_zip);
				memdelete(tmp_app_path);
				return ERR_CANT_CREATE;
			};
			f->store_buffer(data.ptr(), data.size());
			f->close();
			memdelete(f);

#if defined(OSX_ENABLED) || defined(X11_ENABLED)
			if (is_execute) {
				// we need execute rights on this file
				chmod(file.utf8().get_data(), 0755);
			}
#endif
		}

		ret = unzGoToNextFile(src_pkg_zip);
	}

	/* we're done with our source zip */
	unzClose(src_pkg_zip);

	if (!found_library) {
		ERR_PRINTS("Requested template library '" + library_to_use + "' not found. It might be missing from your template archive.");
		memdelete(tmp_app_path);
		return ERR_FILE_NOT_FOUND;
	}

	// Copy project static libs to the project
	Vector<Ref<EditorExportPlugin> > export_plugins = EditorExport::get_singleton()->get_export_plugins();
	for (int i = 0; i < export_plugins.size(); i++) {
		Vector<String> project_static_libs = export_plugins[i]->get_ios_project_static_libs();
		for (int j = 0; j < project_static_libs.size(); j++) {
			const String &static_lib_path = project_static_libs[j];
			String dest_lib_file_path = dest_dir + static_lib_path.get_file();
			Error lib_copy_err = tmp_app_path->copy(static_lib_path, dest_lib_file_path);
			if (lib_copy_err != OK) {
				ERR_PRINTS("Can't copy '" + static_lib_path + "'.");
				memdelete(tmp_app_path);
				return lib_copy_err;
			}
		}
	}

	String iconset_dir = dest_dir + binary_name + "/Images.xcassets/AppIcon.appiconset/";
	err = OK;
	if (!tmp_app_path->dir_exists(iconset_dir)) {
		err = tmp_app_path->make_dir_recursive(iconset_dir);
	}
	memdelete(tmp_app_path);
	if (err)
		return err;

	err = _export_icons(p_preset, iconset_dir);
	if (err)
		return err;

	bool use_storyboard = p_preset->get("storyboard/use_launch_screen_storyboard");

	String launch_image_path = dest_dir + binary_name + "/Images.xcassets/LaunchImage.launchimage/";
	String splash_image_path = dest_dir + binary_name + "/Images.xcassets/SplashImage.imageset/";

	DirAccess *launch_screen_da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	if (!launch_screen_da) {
		return ERR_CANT_CREATE;
	}

	if (use_storyboard) {
		print_line("Using Launch Storyboard");

		if (launch_screen_da->change_dir(launch_image_path) == OK) {
			launch_screen_da->erase_contents_recursive();
			launch_screen_da->remove(launch_image_path);
		}

		err = _export_loading_screen_file(p_preset, splash_image_path);
	} else {
		print_line("Using Launch Images");

		const String launch_screen_path = dest_dir + binary_name + "/Launch Screen.storyboard";

		launch_screen_da->remove(launch_screen_path);

		if (launch_screen_da->change_dir(splash_image_path) == OK) {
			launch_screen_da->erase_contents_recursive();
			launch_screen_da->remove(splash_image_path);
		}

		err = _export_loading_screen_images(p_preset, launch_image_path);
	}

	memdelete(launch_screen_da);

	if (err) {
		return err;
	}

	print_line("Exporting additional assets");
	_export_additional_assets(dest_dir + binary_name, libraries, assets);
	_add_assets_to_project(p_preset, project_file_data, assets);
	String project_file_name = dest_dir + binary_name + ".xcodeproj/project.pbxproj";
	FileAccess *f = FileAccess::open(project_file_name, FileAccess::WRITE);
	if (!f) {
		ERR_PRINTS("Can't write '" + project_file_name + "'.");
		return ERR_CANT_CREATE;
	};
	f->store_buffer(project_file_data.ptr(), project_file_data.size());
	f->close();
	memdelete(f);

#ifdef OSX_ENABLED
	if (ep.step("Code-signing dylibs", 2)) {
		return ERR_SKIP;
	}
	DirAccess *dylibs_dir = DirAccess::open(dest_dir + binary_name + "/dylibs");
	ERR_FAIL_COND_V(!dylibs_dir, ERR_CANT_OPEN);
	CodesignData codesign_data(p_preset, p_debug);
	err = _walk_dir_recursive(dylibs_dir, _codesign, &codesign_data);
	memdelete(dylibs_dir);
	ERR_FAIL_COND_V(err, err);

	if (ep.step("Making .xcarchive", 3)) {
		return ERR_SKIP;
	}
	String archive_path = p_path.get_basename() + ".xcarchive";
	List<String> archive_args;
	archive_args.push_back("-project");
	archive_args.push_back(dest_dir + binary_name + ".xcodeproj");
	archive_args.push_back("-scheme");
	archive_args.push_back(binary_name);
	archive_args.push_back("-sdk");
	archive_args.push_back("iphoneos");
	archive_args.push_back("-configuration");
	archive_args.push_back(p_debug ? "Debug" : "Release");
	archive_args.push_back("-destination");
	archive_args.push_back("generic/platform=iOS");
	archive_args.push_back("archive");
	archive_args.push_back("-archivePath");
	archive_args.push_back(archive_path);
	err = OS::get_singleton()->execute("xcodebuild", archive_args, true);
	ERR_FAIL_COND_V(err, err);

	if (ep.step("Making .ipa", 4)) {
		return ERR_SKIP;
	}
	List<String> export_args;
	export_args.push_back("-exportArchive");
	export_args.push_back("-archivePath");
	export_args.push_back(archive_path);
	export_args.push_back("-exportOptionsPlist");
	export_args.push_back(dest_dir + binary_name + "/export_options.plist");
	export_args.push_back("-allowProvisioningUpdates");
	export_args.push_back("-exportPath");
	export_args.push_back(dest_dir);
	err = OS::get_singleton()->execute("xcodebuild", export_args, true);
	ERR_FAIL_COND_V(err, err);
#else
	print_line(".ipa can only be built on macOS. Leaving Xcode project without building the package.");
#endif

	return OK;
}

bool EditorExportPlatformIOS::can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {

	String err;
	bool valid = false;

	// Look for export templates (first official, and if defined custom templates).

	bool dvalid = exists_export_template("iphone.zip", &err);
	bool rvalid = dvalid; // Both in the same ZIP.

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

	// Validate the rest of the configuration.

	String team_id = p_preset->get("application/app_store_team_id");
	if (team_id.length() == 0) {
		err += TTR("App Store Team ID not specified - cannot configure the project.") + "\n";
		valid = false;
	}

	String identifier = p_preset->get("application/identifier");
	String pn_err;
	if (!is_package_name_valid(identifier, &pn_err)) {
		err += TTR("Invalid Identifier:") + " " + pn_err + "\n";
		valid = false;
	}

	for (uint64_t i = 0; i < (sizeof(icon_infos) / sizeof(icon_infos[0])); ++i) {
		IconInfo info = icon_infos[i];
		String icon_path = p_preset->get(info.preset_key);
		if (icon_path.length() == 0) {
			if (info.is_required) {
				err += TTR("Required icon is not specified in the preset.") + "\n";
				valid = false;
			}
			break;
		}
	}

	String etc_error = test_etc2_or_pvrtc();
	if (etc_error != String()) {
		valid = false;
		err += etc_error;
	}

	if (!err.empty())
		r_error = err;

	return valid;
}

EditorExportPlatformIOS::EditorExportPlatformIOS() {

	Ref<Image> img = memnew(Image(_iphone_logo));
	logo.instance();
	logo->create_from_image(img);

	plugins_changed = true;
	quit_request = false;
	plugins_lock = Mutex::create();

	check_for_changes_thread = Thread::create(_check_for_changes_poll_thread, this);
}

EditorExportPlatformIOS::~EditorExportPlatformIOS() {
	quit_request = true;
	Thread::wait_to_finish(check_for_changes_thread);
	memdelete(plugins_lock);
	memdelete(check_for_changes_thread);
}

void register_iphone_exporter() {

	Ref<EditorExportPlatformIOS> platform;
	platform.instance();

	EditorExport::get_singleton()->add_export_platform(platform);
}

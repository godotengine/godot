/*************************************************************************/
/*  export_plugin.cpp                                                    */
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

#include "export_plugin.h"

static const char *android_perms[] = {
	"ACCESS_CHECKIN_PROPERTIES",
	"ACCESS_COARSE_LOCATION",
	"ACCESS_FINE_LOCATION",
	"ACCESS_LOCATION_EXTRA_COMMANDS",
	"ACCESS_MOCK_LOCATION",
	"ACCESS_NETWORK_STATE",
	"ACCESS_SURFACE_FLINGER",
	"ACCESS_WIFI_STATE",
	"ACCOUNT_MANAGER",
	"ADD_VOICEMAIL",
	"AUTHENTICATE_ACCOUNTS",
	"BATTERY_STATS",
	"BIND_ACCESSIBILITY_SERVICE",
	"BIND_APPWIDGET",
	"BIND_DEVICE_ADMIN",
	"BIND_INPUT_METHOD",
	"BIND_NFC_SERVICE",
	"BIND_NOTIFICATION_LISTENER_SERVICE",
	"BIND_PRINT_SERVICE",
	"BIND_REMOTEVIEWS",
	"BIND_TEXT_SERVICE",
	"BIND_VPN_SERVICE",
	"BIND_WALLPAPER",
	"BLUETOOTH",
	"BLUETOOTH_ADMIN",
	"BLUETOOTH_PRIVILEGED",
	"BRICK",
	"BROADCAST_PACKAGE_REMOVED",
	"BROADCAST_SMS",
	"BROADCAST_STICKY",
	"BROADCAST_WAP_PUSH",
	"CALL_PHONE",
	"CALL_PRIVILEGED",
	"CAMERA",
	"CAPTURE_AUDIO_OUTPUT",
	"CAPTURE_SECURE_VIDEO_OUTPUT",
	"CAPTURE_VIDEO_OUTPUT",
	"CHANGE_COMPONENT_ENABLED_STATE",
	"CHANGE_CONFIGURATION",
	"CHANGE_NETWORK_STATE",
	"CHANGE_WIFI_MULTICAST_STATE",
	"CHANGE_WIFI_STATE",
	"CLEAR_APP_CACHE",
	"CLEAR_APP_USER_DATA",
	"CONTROL_LOCATION_UPDATES",
	"DELETE_CACHE_FILES",
	"DELETE_PACKAGES",
	"DEVICE_POWER",
	"DIAGNOSTIC",
	"DISABLE_KEYGUARD",
	"DUMP",
	"EXPAND_STATUS_BAR",
	"FACTORY_TEST",
	"FLASHLIGHT",
	"FORCE_BACK",
	"GET_ACCOUNTS",
	"GET_PACKAGE_SIZE",
	"GET_TASKS",
	"GET_TOP_ACTIVITY_INFO",
	"GLOBAL_SEARCH",
	"HARDWARE_TEST",
	"INJECT_EVENTS",
	"INSTALL_LOCATION_PROVIDER",
	"INSTALL_PACKAGES",
	"INSTALL_SHORTCUT",
	"INTERNAL_SYSTEM_WINDOW",
	"INTERNET",
	"KILL_BACKGROUND_PROCESSES",
	"LOCATION_HARDWARE",
	"MANAGE_ACCOUNTS",
	"MANAGE_APP_TOKENS",
	"MANAGE_DOCUMENTS",
	"MASTER_CLEAR",
	"MEDIA_CONTENT_CONTROL",
	"MODIFY_AUDIO_SETTINGS",
	"MODIFY_PHONE_STATE",
	"MOUNT_FORMAT_FILESYSTEMS",
	"MOUNT_UNMOUNT_FILESYSTEMS",
	"NFC",
	"PERSISTENT_ACTIVITY",
	"PROCESS_OUTGOING_CALLS",
	"READ_CALENDAR",
	"READ_CALL_LOG",
	"READ_CONTACTS",
	"READ_EXTERNAL_STORAGE",
	"READ_FRAME_BUFFER",
	"READ_HISTORY_BOOKMARKS",
	"READ_INPUT_STATE",
	"READ_LOGS",
	"READ_PHONE_STATE",
	"READ_PROFILE",
	"READ_SMS",
	"READ_SOCIAL_STREAM",
	"READ_SYNC_SETTINGS",
	"READ_SYNC_STATS",
	"READ_USER_DICTIONARY",
	"REBOOT",
	"RECEIVE_BOOT_COMPLETED",
	"RECEIVE_MMS",
	"RECEIVE_SMS",
	"RECEIVE_WAP_PUSH",
	"RECORD_AUDIO",
	"REORDER_TASKS",
	"RESTART_PACKAGES",
	"SEND_RESPOND_VIA_MESSAGE",
	"SEND_SMS",
	"SET_ACTIVITY_WATCHER",
	"SET_ALARM",
	"SET_ALWAYS_FINISH",
	"SET_ANIMATION_SCALE",
	"SET_DEBUG_APP",
	"SET_ORIENTATION",
	"SET_POINTER_SPEED",
	"SET_PREFERRED_APPLICATIONS",
	"SET_PROCESS_LIMIT",
	"SET_TIME",
	"SET_TIME_ZONE",
	"SET_WALLPAPER",
	"SET_WALLPAPER_HINTS",
	"SIGNAL_PERSISTENT_PROCESSES",
	"STATUS_BAR",
	"SUBSCRIBED_FEEDS_READ",
	"SUBSCRIBED_FEEDS_WRITE",
	"SYSTEM_ALERT_WINDOW",
	"TRANSMIT_IR",
	"UNINSTALL_SHORTCUT",
	"UPDATE_DEVICE_STATS",
	"USE_CREDENTIALS",
	"USE_SIP",
	"VIBRATE",
	"WAKE_LOCK",
	"WRITE_APN_SETTINGS",
	"WRITE_CALENDAR",
	"WRITE_CALL_LOG",
	"WRITE_CONTACTS",
	"WRITE_EXTERNAL_STORAGE",
	"WRITE_GSERVICES",
	"WRITE_HISTORY_BOOKMARKS",
	"WRITE_PROFILE",
	"WRITE_SECURE_SETTINGS",
	"WRITE_SETTINGS",
	"WRITE_SMS",
	"WRITE_SOCIAL_STREAM",
	"WRITE_SYNC_SETTINGS",
	"WRITE_USER_DICTIONARY",
	nullptr
};

static const char *SPLASH_IMAGE_EXPORT_PATH = "res/drawable-nodpi/splash.png";
static const char *LEGACY_BUILD_SPLASH_IMAGE_EXPORT_PATH = "res/drawable-nodpi-v4/splash.png";
static const char *SPLASH_BG_COLOR_PATH = "res/drawable-nodpi/splash_bg_color.png";
static const char *LEGACY_BUILD_SPLASH_BG_COLOR_PATH = "res/drawable-nodpi-v4/splash_bg_color.png";
static const char *SPLASH_CONFIG_PATH = "res://android/build/res/drawable/splash_drawable.xml";
static const char *GDNATIVE_LIBS_PATH = "res://android/build/libs/gdnativelibs.json";

static const int icon_densities_count = 6;
static const char *launcher_icon_option = "launcher_icons/main_192x192";
static const char *launcher_adaptive_icon_foreground_option = "launcher_icons/adaptive_foreground_432x432";
static const char *launcher_adaptive_icon_background_option = "launcher_icons/adaptive_background_432x432";

static const LauncherIcon launcher_icons[icon_densities_count] = {
	{ "res/mipmap-xxxhdpi-v4/icon.png", 192 },
	{ "res/mipmap-xxhdpi-v4/icon.png", 144 },
	{ "res/mipmap-xhdpi-v4/icon.png", 96 },
	{ "res/mipmap-hdpi-v4/icon.png", 72 },
	{ "res/mipmap-mdpi-v4/icon.png", 48 },
	{ "res/mipmap/icon.png", 192 }
};

static const LauncherIcon launcher_adaptive_icon_foregrounds[icon_densities_count] = {
	{ "res/mipmap-xxxhdpi-v4/icon_foreground.png", 432 },
	{ "res/mipmap-xxhdpi-v4/icon_foreground.png", 324 },
	{ "res/mipmap-xhdpi-v4/icon_foreground.png", 216 },
	{ "res/mipmap-hdpi-v4/icon_foreground.png", 162 },
	{ "res/mipmap-mdpi-v4/icon_foreground.png", 108 },
	{ "res/mipmap/icon_foreground.png", 432 }
};

static const LauncherIcon launcher_adaptive_icon_backgrounds[icon_densities_count] = {
	{ "res/mipmap-xxxhdpi-v4/icon_background.png", 432 },
	{ "res/mipmap-xxhdpi-v4/icon_background.png", 324 },
	{ "res/mipmap-xhdpi-v4/icon_background.png", 216 },
	{ "res/mipmap-hdpi-v4/icon_background.png", 162 },
	{ "res/mipmap-mdpi-v4/icon_background.png", 108 },
	{ "res/mipmap/icon_background.png", 432 }
};

static const int EXPORT_FORMAT_APK = 0;
static const int EXPORT_FORMAT_AAB = 1;

static const char *APK_ASSETS_DIRECTORY = "res://android/build/assets";
static const char *AAB_ASSETS_DIRECTORY = "res://android/build/assetPacks/installTime/src/main/assets";

void EditorExportPlatformAndroid::_check_for_changes_poll_thread(void *ud) {
	EditorExportPlatformAndroid *ea = (EditorExportPlatformAndroid *)ud;

	while (!ea->quit_request.is_set()) {
		// Check for plugins updates
		{
			// Nothing to do if we already know the plugins have changed.
			if (!ea->plugins_changed.is_set()) {
				Vector<PluginConfigAndroid> loaded_plugins = get_plugins();

				MutexLock lock(ea->plugins_lock);

				if (ea->plugins.size() != loaded_plugins.size()) {
					ea->plugins_changed.set();
				} else {
					for (int i = 0; i < ea->plugins.size(); i++) {
						if (ea->plugins[i].name != loaded_plugins[i].name) {
							ea->plugins_changed.set();
							break;
						}
					}
				}

				if (ea->plugins_changed.is_set()) {
					ea->plugins = loaded_plugins;
				}
			}
		}

		// Check for devices updates
		String adb = get_adb_path();
		if (FileAccess::exists(adb)) {
			String devices;
			List<String> args;
			args.push_back("devices");
			int ec;
			OS::get_singleton()->execute(adb, args, &devices, &ec);

			Vector<String> ds = devices.split("\n");
			Vector<String> ldevices;
			for (int i = 1; i < ds.size(); i++) {
				String d = ds[i];
				int dpos = d.find("device");
				if (dpos == -1) {
					continue;
				}
				d = d.substr(0, dpos).strip_edges();
				ldevices.push_back(d);
			}

			MutexLock lock(ea->device_lock);

			bool different = false;

			if (ea->devices.size() != ldevices.size()) {
				different = true;
			} else {
				for (int i = 0; i < ea->devices.size(); i++) {
					if (ea->devices[i].id != ldevices[i]) {
						different = true;
						break;
					}
				}
			}

			if (different) {
				Vector<Device> ndevices;

				for (int i = 0; i < ldevices.size(); i++) {
					Device d;
					d.id = ldevices[i];
					for (int j = 0; j < ea->devices.size(); j++) {
						if (ea->devices[j].id == ldevices[i]) {
							d.description = ea->devices[j].description;
							d.name = ea->devices[j].name;
							d.api_level = ea->devices[j].api_level;
						}
					}

					if (d.description == "") {
						//in the oven, request!
						args.clear();
						args.push_back("-s");
						args.push_back(d.id);
						args.push_back("shell");
						args.push_back("getprop");
						int ec2;
						String dp;

						OS::get_singleton()->execute(adb, args, &dp, &ec2);

						Vector<String> props = dp.split("\n");
						String vendor;
						String device;
						d.description = "Device ID: " + d.id + "\n";
						d.api_level = 0;
						for (int j = 0; j < props.size(); j++) {
							// got information by `shell cat /system/build.prop` before and its format is "property=value"
							// it's now changed to use `shell getporp` because of permission issue with Android 8.0 and above
							// its format is "[property]: [value]" so changed it as like build.prop
							String p = props[j];
							p = p.replace("]: ", "=");
							p = p.replace("[", "");
							p = p.replace("]", "");

							if (p.begins_with("ro.product.model=")) {
								device = p.get_slice("=", 1).strip_edges();
							} else if (p.begins_with("ro.product.brand=")) {
								vendor = p.get_slice("=", 1).strip_edges().capitalize();
							} else if (p.begins_with("ro.build.display.id=")) {
								d.description += "Build: " + p.get_slice("=", 1).strip_edges() + "\n";
							} else if (p.begins_with("ro.build.version.release=")) {
								d.description += "Release: " + p.get_slice("=", 1).strip_edges() + "\n";
							} else if (p.begins_with("ro.build.version.sdk=")) {
								d.api_level = p.get_slice("=", 1).to_int();
							} else if (p.begins_with("ro.product.cpu.abi=")) {
								d.description += "CPU: " + p.get_slice("=", 1).strip_edges() + "\n";
							} else if (p.begins_with("ro.product.manufacturer=")) {
								d.description += "Manufacturer: " + p.get_slice("=", 1).strip_edges() + "\n";
							} else if (p.begins_with("ro.board.platform=")) {
								d.description += "Chipset: " + p.get_slice("=", 1).strip_edges() + "\n";
							} else if (p.begins_with("ro.opengles.version=")) {
								uint32_t opengl = p.get_slice("=", 1).to_int();
								d.description += "OpenGL: " + itos(opengl >> 16) + "." + itos((opengl >> 8) & 0xFF) + "." + itos((opengl)&0xFF) + "\n";
							}
						}

						d.name = vendor + " " + device;
						if (device == String()) {
							continue;
						}
					}

					ndevices.push_back(d);
				}

				ea->devices = ndevices;
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

	if (EditorSettings::get_singleton()->get("export/android/shutdown_adb_on_exit")) {
		String adb = get_adb_path();
		if (!FileAccess::exists(adb)) {
			return; //adb not configured
		}

		List<String> args;
		args.push_back("kill-server");
		OS::get_singleton()->execute(adb, args);
	};
}

String EditorExportPlatformAndroid::get_project_name(const String &p_name) const {
	String aname;
	if (p_name != "") {
		aname = p_name;
	} else {
		aname = ProjectSettings::get_singleton()->get("application/config/name");
	}

	if (aname == "") {
		aname = VERSION_NAME;
	}

	return aname;
}

String EditorExportPlatformAndroid::get_package_name(const String &p_package) const {
	String pname = p_package;
	String basename = ProjectSettings::get_singleton()->get("application/config/name");
	basename = basename.to_lower();

	String name;
	bool first = true;
	for (int i = 0; i < basename.length(); i++) {
		char32_t c = basename[i];
		if (c >= '0' && c <= '9' && first) {
			continue;
		}
		if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
			name += String::chr(c);
			first = false;
		}
	}
	if (name == "") {
		name = "noname";
	}

	pname = pname.replace("$genname", name);

	return pname;
}

String EditorExportPlatformAndroid::get_assets_directory(const Ref<EditorExportPreset> &p_preset, int p_export_format) const {
	return p_export_format == EXPORT_FORMAT_AAB ? AAB_ASSETS_DIRECTORY : APK_ASSETS_DIRECTORY;
}

bool EditorExportPlatformAndroid::is_package_name_valid(const String &p_package, String *r_error) const {
	String pname = p_package;

	if (pname.length() == 0) {
		if (r_error) {
			*r_error = TTR("Package name is missing.");
		}
		return false;
	}

	int segments = 0;
	bool first = true;
	for (int i = 0; i < pname.length(); i++) {
		char32_t c = pname[i];
		if (first && c == '.') {
			if (r_error) {
				*r_error = TTR("Package segments must be of non-zero length.");
			}
			return false;
		}
		if (c == '.') {
			segments++;
			first = true;
			continue;
		}
		if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) {
			if (r_error) {
				*r_error = vformat(TTR("The character '%s' is not allowed in Android application package names."), String::chr(c));
			}
			return false;
		}
		if (first && (c >= '0' && c <= '9')) {
			if (r_error) {
				*r_error = TTR("A digit cannot be the first character in a package segment.");
			}
			return false;
		}
		if (first && c == '_') {
			if (r_error) {
				*r_error = vformat(TTR("The character '%s' cannot be the first character in a package segment."), String::chr(c));
			}
			return false;
		}
		first = false;
	}

	if (segments == 0) {
		if (r_error) {
			*r_error = TTR("The package must have at least one '.' separator.");
		}
		return false;
	}

	if (first) {
		if (r_error) {
			*r_error = TTR("Package segments must be of non-zero length.");
		}
		return false;
	}

	return true;
}

bool EditorExportPlatformAndroid::_should_compress_asset(const String &p_path, const Vector<uint8_t> &p_data) {
	/*
     *  By not compressing files with little or not benefit in doing so,
     *  a performance gain is expected attime. Moreover, if the APK is
     *  zip-aligned, assets stored as they are can be efficiently read by
     *  Android by memory-mapping them.
     */

	// -- Unconditional uncompress to mimic AAPT plus some other

	static const char *unconditional_compress_ext[] = {
		// From https://github.com/android/platform_frameworks_base/blob/master/tools/aapt/Package.cpp
		// These formats are already compressed, or don't compress well:
		".jpg", ".jpeg", ".png", ".gif",
		".wav", ".mp2", ".mp3", ".ogg", ".aac",
		".mpg", ".mpeg", ".mid", ".midi", ".smf", ".jet",
		".rtttl", ".imy", ".xmf", ".mp4", ".m4a",
		".m4v", ".3gp", ".3gpp", ".3g2", ".3gpp2",
		".amr", ".awb", ".wma", ".wmv",
		// Godot-specific:
		".webp", // Same reasoning as .png
		".cfb", // Don't let small config files slow-down startup
		".scn", // Binary scenes are usually already compressed
		".stex", // Streamable textures are usually already compressed
		// Trailer for easier processing
		nullptr
	};

	for (const char **ext = unconditional_compress_ext; *ext; ++ext) {
		if (p_path.to_lower().ends_with(String(*ext))) {
			return false;
		}
	}

	// -- Compressed resource?

	if (p_data.size() >= 4 && p_data[0] == 'R' && p_data[1] == 'S' && p_data[2] == 'C' && p_data[3] == 'C') {
		// Already compressed
		return false;
	}

	// --- TODO: Decide on texture resources according to their image compression setting

	return true;
}

zip_fileinfo EditorExportPlatformAndroid::get_zip_fileinfo() {
	OS::Time time = OS::get_singleton()->get_time();
	OS::Date date = OS::get_singleton()->get_date();

	zip_fileinfo zipfi;
	zipfi.tmz_date.tm_hour = time.hour;
	zipfi.tmz_date.tm_mday = date.day;
	zipfi.tmz_date.tm_min = time.minute;
	zipfi.tmz_date.tm_mon = date.month - 1; // tm_mon is zero indexed
	zipfi.tmz_date.tm_sec = time.second;
	zipfi.tmz_date.tm_year = date.year;
	zipfi.dosDate = 0;
	zipfi.external_fa = 0;
	zipfi.internal_fa = 0;

	return zipfi;
}

Vector<String> EditorExportPlatformAndroid::get_abis() {
	Vector<String> abis;
	abis.push_back("armeabi-v7a");
	abis.push_back("arm64-v8a");
	abis.push_back("x86");
	abis.push_back("x86_64");
	return abis;
}

/// List the gdap files in the directory specified by the p_path parameter.
Vector<String> EditorExportPlatformAndroid::list_gdap_files(const String &p_path) {
	Vector<String> dir_files;
	DirAccessRef da = DirAccess::open(p_path);
	if (da) {
		da->list_dir_begin();
		while (true) {
			String file = da->get_next();
			if (file == "") {
				break;
			}

			if (da->current_is_dir() || da->current_is_hidden()) {
				continue;
			}

			if (file.ends_with(PluginConfigAndroid::PLUGIN_CONFIG_EXT)) {
				dir_files.push_back(file);
			}
		}
		da->list_dir_end();
	}

	return dir_files;
}

Vector<PluginConfigAndroid> EditorExportPlatformAndroid::get_plugins() {
	Vector<PluginConfigAndroid> loaded_plugins;

	String plugins_dir = ProjectSettings::get_singleton()->get_resource_path().plus_file("android/plugins");

	// Add the prebuilt plugins
	loaded_plugins.append_array(PluginConfigAndroid::get_prebuilt_plugins(plugins_dir));

	if (DirAccess::exists(plugins_dir)) {
		Vector<String> plugins_filenames = list_gdap_files(plugins_dir);

		if (!plugins_filenames.is_empty()) {
			Ref<ConfigFile> config_file = memnew(ConfigFile);
			for (int i = 0; i < plugins_filenames.size(); i++) {
				PluginConfigAndroid config = PluginConfigAndroid::load_plugin_config(config_file, plugins_dir.plus_file(plugins_filenames[i]));
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

Vector<PluginConfigAndroid> EditorExportPlatformAndroid::get_enabled_plugins(const Ref<EditorExportPreset> &p_presets) {
	Vector<PluginConfigAndroid> enabled_plugins;
	Vector<PluginConfigAndroid> all_plugins = get_plugins();
	for (int i = 0; i < all_plugins.size(); i++) {
		PluginConfigAndroid plugin = all_plugins[i];
		bool enabled = p_presets->get("plugins/" + plugin.name);
		if (enabled) {
			enabled_plugins.push_back(plugin);
		}
	}

	return enabled_plugins;
}

Error EditorExportPlatformAndroid::store_in_apk(APKExportData *ed, const String &p_path, const Vector<uint8_t> &p_data, int compression_method) {
	zip_fileinfo zipfi = get_zip_fileinfo();
	zipOpenNewFileInZip(ed->apk,
			p_path.utf8().get_data(),
			&zipfi,
			nullptr,
			0,
			nullptr,
			0,
			nullptr,
			compression_method,
			Z_DEFAULT_COMPRESSION);

	zipWriteInFileInZip(ed->apk, p_data.ptr(), p_data.size());
	zipCloseFileInZip(ed->apk);

	return OK;
}

Error EditorExportPlatformAndroid::save_apk_so(void *p_userdata, const SharedObject &p_so) {
	if (!p_so.path.get_file().begins_with("lib")) {
		String err = "Android .so file names must start with \"lib\", but got: " + p_so.path;
		ERR_PRINT(err);
		return FAILED;
	}
	APKExportData *ed = (APKExportData *)p_userdata;
	Vector<String> abis = get_abis();
	bool exported = false;
	for (int i = 0; i < p_so.tags.size(); ++i) {
		// shared objects can be fat (compatible with multiple ABIs)
		int abi_index = abis.find(p_so.tags[i]);
		if (abi_index != -1) {
			exported = true;
			String abi = abis[abi_index];
			String dst_path = String("lib").plus_file(abi).plus_file(p_so.path.get_file());
			Vector<uint8_t> array = FileAccess::get_file_as_array(p_so.path);
			Error store_err = store_in_apk(ed, dst_path, array);
			ERR_FAIL_COND_V_MSG(store_err, store_err, "Cannot store in apk file '" + dst_path + "'.");
		}
	}
	if (!exported) {
		String abis_string = String(" ").join(abis);
		String err = "Cannot determine ABI for library \"" + p_so.path + "\". One of the supported ABIs must be used as a tag: " + abis_string;
		ERR_PRINT(err);
		return FAILED;
	}
	return OK;
}

Error EditorExportPlatformAndroid::save_apk_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const Vector<uint8_t> &p_key) {
	APKExportData *ed = (APKExportData *)p_userdata;
	String dst_path = p_path.replace_first("res://", "assets/");

	store_in_apk(ed, dst_path, p_data, _should_compress_asset(p_path, p_data) ? Z_DEFLATED : 0);
	return OK;
}

Error EditorExportPlatformAndroid::ignore_apk_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const Vector<uint8_t> &p_key) {
	return OK;
}

Error EditorExportPlatformAndroid::copy_gradle_so(void *p_userdata, const SharedObject &p_so) {
	ERR_FAIL_COND_V_MSG(!p_so.path.get_file().begins_with("lib"), FAILED,
			"Android .so file names must start with \"lib\", but got: " + p_so.path);
	Vector<String> abis = get_abis();
	CustomExportData *export_data = (CustomExportData *)p_userdata;
	bool exported = false;
	for (int i = 0; i < p_so.tags.size(); ++i) {
		int abi_index = abis.find(p_so.tags[i]);
		if (abi_index != -1) {
			exported = true;
			String base = "res://android/build/libs";
			String type = export_data->debug ? "debug" : "release";
			String abi = abis[abi_index];
			String filename = p_so.path.get_file();
			String dst_path = base.plus_file(type).plus_file(abi).plus_file(filename);
			Vector<uint8_t> data = FileAccess::get_file_as_array(p_so.path);
			print_verbose("Copying .so file from " + p_so.path + " to " + dst_path);
			Error err = store_file_at_path(dst_path, data);
			ERR_FAIL_COND_V_MSG(err, err, "Failed to copy .so file from " + p_so.path + " to " + dst_path);
			export_data->libs.push_back(dst_path);
		}
	}
	ERR_FAIL_COND_V_MSG(!exported, FAILED,
			"Cannot determine ABI for library \"" + p_so.path + "\". One of the supported ABIs must be used as a tag: " + String(" ").join(abis));
	return OK;
}

bool EditorExportPlatformAndroid::_has_storage_permission(const Vector<String> &p_permissions) {
	return p_permissions.find("android.permission.READ_EXTERNAL_STORAGE") != -1 || p_permissions.find("android.permission.WRITE_EXTERNAL_STORAGE") != -1;
}

void EditorExportPlatformAndroid::_get_permissions(const Ref<EditorExportPreset> &p_preset, bool p_give_internet, Vector<String> &r_permissions) {
	const char **aperms = android_perms;
	while (*aperms) {
		bool enabled = p_preset->get("permissions/" + String(*aperms).to_lower());
		if (enabled) {
			r_permissions.push_back("android.permission." + String(*aperms));
		}
		aperms++;
	}
	PackedStringArray user_perms = p_preset->get("permissions/custom_permissions");
	for (int i = 0; i < user_perms.size(); i++) {
		String user_perm = user_perms[i].strip_edges();
		if (!user_perm.is_empty()) {
			r_permissions.push_back(user_perm);
		}
	}
	if (p_give_internet) {
		if (r_permissions.find("android.permission.INTERNET") == -1) {
			r_permissions.push_back("android.permission.INTERNET");
		}
	}

	int xr_mode_index = p_preset->get("xr_features/xr_mode");
	if (xr_mode_index == 1 /* XRMode.OVR */) {
		int hand_tracking_index = p_preset->get("xr_features/hand_tracking"); // 0: none, 1: optional, 2: required
		if (hand_tracking_index > 0) {
			if (r_permissions.find("com.oculus.permission.HAND_TRACKING") == -1) {
				r_permissions.push_back("com.oculus.permission.HAND_TRACKING");
			}
		}
	}
}

void EditorExportPlatformAndroid::_write_tmp_manifest(const Ref<EditorExportPreset> &p_preset, bool p_give_internet, bool p_debug) {
	print_verbose("Building temporary manifest..");
	String manifest_text =
			"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
			"<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\"\n"
			"    xmlns:tools=\"http://schemas.android.com/tools\">\n";

	manifest_text += _get_screen_sizes_tag(p_preset);
	manifest_text += _get_gles_tag();

	Vector<String> perms;
	_get_permissions(p_preset, p_give_internet, perms);
	for (int i = 0; i < perms.size(); i++) {
		String permission = perms.get(i);
		if (permission == "android.permission.WRITE_EXTERNAL_STORAGE" || permission == "android.permission.READ_EXTERNAL_STORAGE") {
			manifest_text += vformat("    <uses-permission android:name=\"%s\" android:maxSdkVersion=\"29\" />\n", permission);
		} else {
			manifest_text += vformat("    <uses-permission android:name=\"%s\" />\n", permission);
		}
	}

	manifest_text += _get_xr_features_tag(p_preset);
	manifest_text += _get_instrumentation_tag(p_preset);
	manifest_text += _get_application_tag(p_preset, _has_storage_permission(perms));
	manifest_text += "</manifest>\n";
	String manifest_path = vformat("res://android/build/src/%s/AndroidManifest.xml", (p_debug ? "debug" : "release"));

	print_verbose("Storing manifest into " + manifest_path + ": " + "\n" + manifest_text);
	store_string_at_path(manifest_path, manifest_text);
}

void EditorExportPlatformAndroid::_fix_manifest(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &p_manifest, bool p_give_internet) {
	// Leaving the unused types commented because looking these constants up
	// again later would be annoying
	// const int CHUNK_AXML_FILE = 0x00080003;
	// const int CHUNK_RESOURCEIDS = 0x00080180;
	const int CHUNK_STRINGS = 0x001C0001;
	// const int CHUNK_XML_END_NAMESPACE = 0x00100101;
	const int CHUNK_XML_END_TAG = 0x00100103;
	// const int CHUNK_XML_START_NAMESPACE = 0x00100100;
	const int CHUNK_XML_START_TAG = 0x00100102;
	// const int CHUNK_XML_TEXT = 0x00100104;
	const int UTF8_FLAG = 0x00000100;

	Vector<String> string_table;

	uint32_t ofs = 8;

	uint32_t string_count = 0;
	//uint32_t styles_count = 0;
	uint32_t string_flags = 0;
	uint32_t string_data_offset = 0;

	//uint32_t styles_offset = 0;
	uint32_t string_table_begins = 0;
	uint32_t string_table_ends = 0;
	Vector<uint8_t> stable_extra;

	String version_name = p_preset->get("version/name");
	int version_code = p_preset->get("version/code");
	String package_name = p_preset->get("package/unique_name");

	const int screen_orientation =
			_get_android_orientation_value(DisplayServer::ScreenOrientation(int(GLOBAL_GET("display/window/handheld/orientation"))));

	bool screen_support_small = p_preset->get("screen/support_small");
	bool screen_support_normal = p_preset->get("screen/support_normal");
	bool screen_support_large = p_preset->get("screen/support_large");
	bool screen_support_xlarge = p_preset->get("screen/support_xlarge");

	int xr_mode_index = p_preset->get("xr_features/xr_mode");

	bool backup_allowed = p_preset->get("user_data_backup/allow");
	bool classify_as_game = p_preset->get("package/classify_as_game");
	bool retain_data_on_uninstall = p_preset->get("package/retain_data_on_uninstall");

	Vector<String> perms;
	// Write permissions into the perms variable.
	_get_permissions(p_preset, p_give_internet, perms);
	bool has_storage_permission = _has_storage_permission(perms);

	while (ofs < (uint32_t)p_manifest.size()) {
		uint32_t chunk = decode_uint32(&p_manifest[ofs]);
		uint32_t size = decode_uint32(&p_manifest[ofs + 4]);

		switch (chunk) {
			case CHUNK_STRINGS: {
				int iofs = ofs + 8;

				string_count = decode_uint32(&p_manifest[iofs]);
				//styles_count = decode_uint32(&p_manifest[iofs + 4]);
				string_flags = decode_uint32(&p_manifest[iofs + 8]);
				string_data_offset = decode_uint32(&p_manifest[iofs + 12]);
				//styles_offset = decode_uint32(&p_manifest[iofs + 16]);
				/*
                printf("string count: %i\n",string_count);
                printf("flags: %i\n",string_flags);
                printf("sdata ofs: %i\n",string_data_offset);
                printf("styles ofs: %i\n",styles_offset);
                */
				uint32_t st_offset = iofs + 20;
				string_table.resize(string_count);
				uint32_t string_end = 0;

				string_table_begins = st_offset;

				for (uint32_t i = 0; i < string_count; i++) {
					uint32_t string_at = decode_uint32(&p_manifest[st_offset + i * 4]);
					string_at += st_offset + string_count * 4;

					ERR_FAIL_COND_MSG(string_flags & UTF8_FLAG, "Unimplemented, can't read UTF-8 string table.");

					if (string_flags & UTF8_FLAG) {
					} else {
						uint32_t len = decode_uint16(&p_manifest[string_at]);
						Vector<char32_t> ucstring;
						ucstring.resize(len + 1);
						for (uint32_t j = 0; j < len; j++) {
							uint16_t c = decode_uint16(&p_manifest[string_at + 2 + 2 * j]);
							ucstring.write[j] = c;
						}
						string_end = MAX(string_at + 2 + 2 * len, string_end);
						ucstring.write[len] = 0;
						string_table.write[i] = ucstring.ptr();
					}
				}

				for (uint32_t i = string_end; i < (ofs + size); i++) {
					stable_extra.push_back(p_manifest[i]);
				}

				string_table_ends = ofs + size;

			} break;
			case CHUNK_XML_START_TAG: {
				int iofs = ofs + 8;
				uint32_t name = decode_uint32(&p_manifest[iofs + 12]);

				String tname = string_table[name];
				uint32_t attrcount = decode_uint32(&p_manifest[iofs + 20]);
				iofs += 28;

				for (uint32_t i = 0; i < attrcount; i++) {
					uint32_t attr_nspace = decode_uint32(&p_manifest[iofs]);
					uint32_t attr_name = decode_uint32(&p_manifest[iofs + 4]);
					uint32_t attr_value = decode_uint32(&p_manifest[iofs + 8]);
					uint32_t attr_resid = decode_uint32(&p_manifest[iofs + 16]);

					const String value = (attr_value != 0xFFFFFFFF) ? string_table[attr_value] : "Res #" + itos(attr_resid);
					String attrname = string_table[attr_name];
					const String nspace = (attr_nspace != 0xFFFFFFFF) ? string_table[attr_nspace] : "";

					//replace project information
					if (tname == "manifest" && attrname == "package") {
						string_table.write[attr_value] = get_package_name(package_name);
					}

					if (tname == "manifest" && attrname == "versionCode") {
						encode_uint32(version_code, &p_manifest.write[iofs + 16]);
					}

					if (tname == "manifest" && attrname == "versionName") {
						if (attr_value == 0xFFFFFFFF) {
							WARN_PRINT("Version name in a resource, should be plain text");
						} else {
							string_table.write[attr_value] = version_name;
						}
					}

					if (tname == "application" && attrname == "requestLegacyExternalStorage") {
						encode_uint32(has_storage_permission ? 0xFFFFFFFF : 0, &p_manifest.write[iofs + 16]);
					}

					if (tname == "application" && attrname == "allowBackup") {
						encode_uint32(backup_allowed, &p_manifest.write[iofs + 16]);
					}

					if (tname == "application" && attrname == "isGame") {
						encode_uint32(classify_as_game, &p_manifest.write[iofs + 16]);
					}

					if (tname == "application" && attrname == "hasFragileUserData") {
						encode_uint32(retain_data_on_uninstall, &p_manifest.write[iofs + 16]);
					}

					if (tname == "instrumentation" && attrname == "targetPackage") {
						string_table.write[attr_value] = get_package_name(package_name);
					}

					if (tname == "activity" && attrname == "screenOrientation") {
						encode_uint32(screen_orientation, &p_manifest.write[iofs + 16]);
					}

					if (tname == "supports-screens") {
						if (attrname == "smallScreens") {
							encode_uint32(screen_support_small ? 0xFFFFFFFF : 0, &p_manifest.write[iofs + 16]);

						} else if (attrname == "normalScreens") {
							encode_uint32(screen_support_normal ? 0xFFFFFFFF : 0, &p_manifest.write[iofs + 16]);

						} else if (attrname == "largeScreens") {
							encode_uint32(screen_support_large ? 0xFFFFFFFF : 0, &p_manifest.write[iofs + 16]);

						} else if (attrname == "xlargeScreens") {
							encode_uint32(screen_support_xlarge ? 0xFFFFFFFF : 0, &p_manifest.write[iofs + 16]);
						}
					}

					iofs += 20;
				}

			} break;
			case CHUNK_XML_END_TAG: {
				int iofs = ofs + 8;
				uint32_t name = decode_uint32(&p_manifest[iofs + 12]);
				String tname = string_table[name];

				if (tname == "uses-feature") {
					Vector<String> feature_names;
					Vector<bool> feature_required_list;
					Vector<int> feature_versions;

					if (xr_mode_index == 1 /* XRMode.OVR */) {
						// Set degrees of freedom
						feature_names.push_back("android.hardware.vr.headtracking");
						feature_required_list.push_back(true);
						feature_versions.push_back(1);

						// Check for hand tracking
						int hand_tracking_index = p_preset->get("xr_features/hand_tracking"); // 0: none, 1: optional, 2: required
						if (hand_tracking_index > 0) {
							feature_names.push_back("oculus.software.handtracking");
							feature_required_list.push_back(hand_tracking_index == 2);
							feature_versions.push_back(-1); // no version attribute should be added.
						}
					}

					if (feature_names.size() > 0) {
						ofs += 24; // skip over end tag

						// save manifest ending so we can restore it
						Vector<uint8_t> manifest_end;
						uint32_t manifest_cur_size = p_manifest.size();

						manifest_end.resize(p_manifest.size() - ofs);
						memcpy(manifest_end.ptrw(), &p_manifest[ofs], manifest_end.size());

						int32_t attr_name_string = string_table.find("name");
						ERR_FAIL_COND_MSG(attr_name_string == -1, "Template does not have 'name' attribute.");

						int32_t ns_android_string = string_table.find("http://schemas.android.com/apk/res/android");
						if (ns_android_string == -1) {
							string_table.push_back("http://schemas.android.com/apk/res/android");
							ns_android_string = string_table.size() - 1;
						}

						int32_t attr_uses_feature_string = string_table.find("uses-feature");
						if (attr_uses_feature_string == -1) {
							string_table.push_back("uses-feature");
							attr_uses_feature_string = string_table.size() - 1;
						}

						int32_t attr_required_string = string_table.find("required");
						if (attr_required_string == -1) {
							string_table.push_back("required");
							attr_required_string = string_table.size() - 1;
						}

						for (int i = 0; i < feature_names.size(); i++) {
							String feature_name = feature_names[i];
							bool feature_required = feature_required_list[i];
							int feature_version = feature_versions[i];
							bool has_version_attribute = feature_version != -1;

							print_line("Adding feature " + feature_name);

							int32_t feature_string = string_table.find(feature_name);
							if (feature_string == -1) {
								string_table.push_back(feature_name);
								feature_string = string_table.size() - 1;
							}

							String required_value_string = feature_required ? "true" : "false";
							int32_t required_value = string_table.find(required_value_string);
							if (required_value == -1) {
								string_table.push_back(required_value_string);
								required_value = string_table.size() - 1;
							}

							int32_t attr_version_string = -1;
							int32_t version_value = -1;
							int tag_size;
							int attr_count;
							if (has_version_attribute) {
								attr_version_string = string_table.find("version");
								if (attr_version_string == -1) {
									string_table.push_back("version");
									attr_version_string = string_table.size() - 1;
								}

								version_value = string_table.find(itos(feature_version));
								if (version_value == -1) {
									string_table.push_back(itos(feature_version));
									version_value = string_table.size() - 1;
								}

								tag_size = 96; // node and three attrs + end node
								attr_count = 3;
							} else {
								tag_size = 76; // node and two attrs + end node
								attr_count = 2;
							}
							manifest_cur_size += tag_size + 24;
							p_manifest.resize(manifest_cur_size);

							// start tag
							encode_uint16(0x102, &p_manifest.write[ofs]); // type
							encode_uint16(16, &p_manifest.write[ofs + 2]); // headersize
							encode_uint32(tag_size, &p_manifest.write[ofs + 4]); // size
							encode_uint32(0, &p_manifest.write[ofs + 8]); // lineno
							encode_uint32(-1, &p_manifest.write[ofs + 12]); // comment
							encode_uint32(-1, &p_manifest.write[ofs + 16]); // ns
							encode_uint32(attr_uses_feature_string, &p_manifest.write[ofs + 20]); // name
							encode_uint16(20, &p_manifest.write[ofs + 24]); // attr_start
							encode_uint16(20, &p_manifest.write[ofs + 26]); // attr_size
							encode_uint16(attr_count, &p_manifest.write[ofs + 28]); // num_attrs
							encode_uint16(0, &p_manifest.write[ofs + 30]); // id_index
							encode_uint16(0, &p_manifest.write[ofs + 32]); // class_index
							encode_uint16(0, &p_manifest.write[ofs + 34]); // style_index

							// android:name attribute
							encode_uint32(ns_android_string, &p_manifest.write[ofs + 36]); // ns
							encode_uint32(attr_name_string, &p_manifest.write[ofs + 40]); // 'name'
							encode_uint32(feature_string, &p_manifest.write[ofs + 44]); // raw_value
							encode_uint16(8, &p_manifest.write[ofs + 48]); // typedvalue_size
							p_manifest.write[ofs + 50] = 0; // typedvalue_always0
							p_manifest.write[ofs + 51] = 0x03; // typedvalue_type (string)
							encode_uint32(feature_string, &p_manifest.write[ofs + 52]); // typedvalue reference

							// android:required attribute
							encode_uint32(ns_android_string, &p_manifest.write[ofs + 56]); // ns
							encode_uint32(attr_required_string, &p_manifest.write[ofs + 60]); // 'name'
							encode_uint32(required_value, &p_manifest.write[ofs + 64]); // raw_value
							encode_uint16(8, &p_manifest.write[ofs + 68]); // typedvalue_size
							p_manifest.write[ofs + 70] = 0; // typedvalue_always0
							p_manifest.write[ofs + 71] = 0x03; // typedvalue_type (string)
							encode_uint32(required_value, &p_manifest.write[ofs + 72]); // typedvalue reference

							ofs += 76;

							if (has_version_attribute) {
								// android:version attribute
								encode_uint32(ns_android_string, &p_manifest.write[ofs]); // ns
								encode_uint32(attr_version_string, &p_manifest.write[ofs + 4]); // 'name'
								encode_uint32(version_value, &p_manifest.write[ofs + 8]); // raw_value
								encode_uint16(8, &p_manifest.write[ofs + 12]); // typedvalue_size
								p_manifest.write[ofs + 14] = 0; // typedvalue_always0
								p_manifest.write[ofs + 15] = 0x03; // typedvalue_type (string)
								encode_uint32(version_value, &p_manifest.write[ofs + 16]); // typedvalue reference

								ofs += 20;
							}

							// end tag
							encode_uint16(0x103, &p_manifest.write[ofs]); // type
							encode_uint16(16, &p_manifest.write[ofs + 2]); // headersize
							encode_uint32(24, &p_manifest.write[ofs + 4]); // size
							encode_uint32(0, &p_manifest.write[ofs + 8]); // lineno
							encode_uint32(-1, &p_manifest.write[ofs + 12]); // comment
							encode_uint32(-1, &p_manifest.write[ofs + 16]); // ns
							encode_uint32(attr_uses_feature_string, &p_manifest.write[ofs + 20]); // name

							ofs += 24;
						}
						memcpy(&p_manifest.write[ofs], manifest_end.ptr(), manifest_end.size());
						ofs -= 24; // go back over back end
					}
				}
				if (tname == "manifest") {
					// save manifest ending so we can restore it
					Vector<uint8_t> manifest_end;
					uint32_t manifest_cur_size = p_manifest.size();

					manifest_end.resize(p_manifest.size() - ofs);
					memcpy(manifest_end.ptrw(), &p_manifest[ofs], manifest_end.size());

					int32_t attr_name_string = string_table.find("name");
					ERR_FAIL_COND_MSG(attr_name_string == -1, "Template does not have 'name' attribute.");

					int32_t ns_android_string = string_table.find("android");
					ERR_FAIL_COND_MSG(ns_android_string == -1, "Template does not have 'android' namespace.");

					int32_t attr_uses_permission_string = string_table.find("uses-permission");
					if (attr_uses_permission_string == -1) {
						string_table.push_back("uses-permission");
						attr_uses_permission_string = string_table.size() - 1;
					}

					for (int i = 0; i < perms.size(); ++i) {
						print_line("Adding permission " + perms[i]);

						manifest_cur_size += 56 + 24; // node + end node
						p_manifest.resize(manifest_cur_size);

						// Add permission to the string pool
						int32_t perm_string = string_table.find(perms[i]);
						if (perm_string == -1) {
							string_table.push_back(perms[i]);
							perm_string = string_table.size() - 1;
						}

						// start tag
						encode_uint16(0x102, &p_manifest.write[ofs]); // type
						encode_uint16(16, &p_manifest.write[ofs + 2]); // headersize
						encode_uint32(56, &p_manifest.write[ofs + 4]); // size
						encode_uint32(0, &p_manifest.write[ofs + 8]); // lineno
						encode_uint32(-1, &p_manifest.write[ofs + 12]); // comment
						encode_uint32(-1, &p_manifest.write[ofs + 16]); // ns
						encode_uint32(attr_uses_permission_string, &p_manifest.write[ofs + 20]); // name
						encode_uint16(20, &p_manifest.write[ofs + 24]); // attr_start
						encode_uint16(20, &p_manifest.write[ofs + 26]); // attr_size
						encode_uint16(1, &p_manifest.write[ofs + 28]); // num_attrs
						encode_uint16(0, &p_manifest.write[ofs + 30]); // id_index
						encode_uint16(0, &p_manifest.write[ofs + 32]); // class_index
						encode_uint16(0, &p_manifest.write[ofs + 34]); // style_index

						// attribute
						encode_uint32(ns_android_string, &p_manifest.write[ofs + 36]); // ns
						encode_uint32(attr_name_string, &p_manifest.write[ofs + 40]); // 'name'
						encode_uint32(perm_string, &p_manifest.write[ofs + 44]); // raw_value
						encode_uint16(8, &p_manifest.write[ofs + 48]); // typedvalue_size
						p_manifest.write[ofs + 50] = 0; // typedvalue_always0
						p_manifest.write[ofs + 51] = 0x03; // typedvalue_type (string)
						encode_uint32(perm_string, &p_manifest.write[ofs + 52]); // typedvalue reference

						ofs += 56;

						// end tag
						encode_uint16(0x103, &p_manifest.write[ofs]); // type
						encode_uint16(16, &p_manifest.write[ofs + 2]); // headersize
						encode_uint32(24, &p_manifest.write[ofs + 4]); // size
						encode_uint32(0, &p_manifest.write[ofs + 8]); // lineno
						encode_uint32(-1, &p_manifest.write[ofs + 12]); // comment
						encode_uint32(-1, &p_manifest.write[ofs + 16]); // ns
						encode_uint32(attr_uses_permission_string, &p_manifest.write[ofs + 20]); // name

						ofs += 24;
					}

					// copy footer back in
					memcpy(&p_manifest.write[ofs], manifest_end.ptr(), manifest_end.size());
				}
			} break;
		}

		ofs += size;
	}

	//create new andriodmanifest binary

	Vector<uint8_t> ret;
	ret.resize(string_table_begins + string_table.size() * 4);

	for (uint32_t i = 0; i < string_table_begins; i++) {
		ret.write[i] = p_manifest[i];
	}

	ofs = 0;
	for (int i = 0; i < string_table.size(); i++) {
		encode_uint32(ofs, &ret.write[string_table_begins + i * 4]);
		ofs += string_table[i].length() * 2 + 2 + 2;
	}

	ret.resize(ret.size() + ofs);
	string_data_offset = ret.size() - ofs;
	uint8_t *chars = &ret.write[string_data_offset];
	for (int i = 0; i < string_table.size(); i++) {
		String s = string_table[i];
		encode_uint16(s.length(), chars);
		chars += 2;
		for (int j = 0; j < s.length(); j++) {
			encode_uint16(s[j], chars);
			chars += 2;
		}
		encode_uint16(0, chars);
		chars += 2;
	}

	for (int i = 0; i < stable_extra.size(); i++) {
		ret.push_back(stable_extra[i]);
	}

	//pad
	while (ret.size() % 4) {
		ret.push_back(0);
	}

	uint32_t new_stable_end = ret.size();

	uint32_t extra = (p_manifest.size() - string_table_ends);
	ret.resize(new_stable_end + extra);
	for (uint32_t i = 0; i < extra; i++) {
		ret.write[new_stable_end + i] = p_manifest[string_table_ends + i];
	}

	while (ret.size() % 4) {
		ret.push_back(0);
	}
	encode_uint32(ret.size(), &ret.write[4]); //update new file size

	encode_uint32(new_stable_end - 8, &ret.write[12]); //update new string table size
	encode_uint32(string_table.size(), &ret.write[16]); //update new number of strings
	encode_uint32(string_data_offset - 8, &ret.write[28]); //update new string data offset

	p_manifest = ret;
}

String EditorExportPlatformAndroid::_parse_string(const uint8_t *p_bytes, bool p_utf8) {
	uint32_t offset = 0;
	uint32_t len = 0;

	if (p_utf8) {
		uint8_t byte = p_bytes[offset];
		if (byte & 0x80) {
			offset += 2;
		} else {
			offset += 1;
		}
		byte = p_bytes[offset];
		offset++;
		if (byte & 0x80) {
			len = byte & 0x7F;
			len = (len << 8) + p_bytes[offset];
			offset++;
		} else {
			len = byte;
		}
	} else {
		len = decode_uint16(&p_bytes[offset]);
		offset += 2;
		if (len & 0x8000) {
			len &= 0x7FFF;
			len = (len << 16) + decode_uint16(&p_bytes[offset]);
			offset += 2;
		}
	}

	if (p_utf8) {
		Vector<uint8_t> str8;
		str8.resize(len + 1);
		for (uint32_t i = 0; i < len; i++) {
			str8.write[i] = p_bytes[offset + i];
		}
		str8.write[len] = 0;
		String str;
		str.parse_utf8((const char *)str8.ptr());
		return str;
	} else {
		String str;
		for (uint32_t i = 0; i < len; i++) {
			char32_t c = decode_uint16(&p_bytes[offset + i * 2]);
			if (c == 0) {
				break;
			}
			str += String::chr(c);
		}
		return str;
	}
}

void EditorExportPlatformAndroid::_fix_resources(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &r_manifest) {
	const int UTF8_FLAG = 0x00000100;

	uint32_t string_block_len = decode_uint32(&r_manifest[16]);
	uint32_t string_count = decode_uint32(&r_manifest[20]);
	uint32_t string_flags = decode_uint32(&r_manifest[28]);
	const uint32_t string_table_begins = 40;

	Vector<String> string_table;

	String package_name = p_preset->get("package/name");

	for (uint32_t i = 0; i < string_count; i++) {
		uint32_t offset = decode_uint32(&r_manifest[string_table_begins + i * 4]);
		offset += string_table_begins + string_count * 4;

		String str = _parse_string(&r_manifest[offset], string_flags & UTF8_FLAG);

		if (str.begins_with("godot-project-name")) {
			if (str == "godot-project-name") {
				//project name
				str = get_project_name(package_name);

			} else {
				String lang = str.substr(str.rfind("-") + 1, str.length()).replace("-", "_");
				String prop = "application/config/name_" + lang;
				if (ProjectSettings::get_singleton()->has_setting(prop)) {
					str = ProjectSettings::get_singleton()->get(prop);
				} else {
					str = get_project_name(package_name);
				}
			}
		}

		string_table.push_back(str);
	}

	//write a new string table, but use 16 bits
	Vector<uint8_t> ret;
	ret.resize(string_table_begins + string_table.size() * 4);

	for (uint32_t i = 0; i < string_table_begins; i++) {
		ret.write[i] = r_manifest[i];
	}

	int ofs = 0;
	for (int i = 0; i < string_table.size(); i++) {
		encode_uint32(ofs, &ret.write[string_table_begins + i * 4]);
		ofs += string_table[i].length() * 2 + 2 + 2;
	}

	ret.resize(ret.size() + ofs);
	uint8_t *chars = &ret.write[ret.size() - ofs];
	for (int i = 0; i < string_table.size(); i++) {
		String s = string_table[i];
		encode_uint16(s.length(), chars);
		chars += 2;
		for (int j = 0; j < s.length(); j++) {
			encode_uint16(s[j], chars);
			chars += 2;
		}
		encode_uint16(0, chars);
		chars += 2;
	}

	//pad
	while (ret.size() % 4) {
		ret.push_back(0);
	}

	//change flags to not use utf8
	encode_uint32(string_flags & ~0x100, &ret.write[28]);
	//change length
	encode_uint32(ret.size() - 12, &ret.write[16]);
	//append the rest...
	int rest_from = 12 + string_block_len;
	int rest_to = ret.size();
	int rest_len = (r_manifest.size() - rest_from);
	ret.resize(ret.size() + (r_manifest.size() - rest_from));
	for (int i = 0; i < rest_len; i++) {
		ret.write[rest_to + i] = r_manifest[rest_from + i];
	}
	//finally update the size
	encode_uint32(ret.size(), &ret.write[4]);

	r_manifest = ret;
	//printf("end\n");
}

void EditorExportPlatformAndroid::_load_image_data(const Ref<Image> &p_splash_image, Vector<uint8_t> &p_data) {
	Vector<uint8_t> png_buffer;
	Error err = PNGDriverCommon::image_to_png(p_splash_image, png_buffer);
	if (err == OK) {
		p_data.resize(png_buffer.size());
		memcpy(p_data.ptrw(), png_buffer.ptr(), p_data.size());
	} else {
		String err_str = String("Failed to convert splash image to png.");
		WARN_PRINT(err_str.utf8().get_data());
	}
}

void EditorExportPlatformAndroid::_process_launcher_icons(const String &p_file_name, const Ref<Image> &p_source_image, int dimension, Vector<uint8_t> &p_data) {
	Ref<Image> working_image = p_source_image;

	if (p_source_image->get_width() != dimension || p_source_image->get_height() != dimension) {
		working_image = p_source_image->duplicate();
		working_image->resize(dimension, dimension, Image::Interpolation::INTERPOLATE_LANCZOS);
	}

	Vector<uint8_t> png_buffer;
	Error err = PNGDriverCommon::image_to_png(working_image, png_buffer);
	if (err == OK) {
		p_data.resize(png_buffer.size());
		memcpy(p_data.ptrw(), png_buffer.ptr(), p_data.size());
	} else {
		String err_str = String("Failed to convert resized icon (") + p_file_name + ") to png.";
		WARN_PRINT(err_str.utf8().get_data());
	}
}

String EditorExportPlatformAndroid::load_splash_refs(Ref<Image> &splash_image, Ref<Image> &splash_bg_color_image) {
	bool scale_splash = ProjectSettings::get_singleton()->get("application/boot_splash/fullsize");
	bool apply_filter = ProjectSettings::get_singleton()->get("application/boot_splash/use_filter");
	String project_splash_path = ProjectSettings::get_singleton()->get("application/boot_splash/image");

	if (!project_splash_path.is_empty()) {
		splash_image.instantiate();
		print_verbose("Loading splash image: " + project_splash_path);
		const Error err = ImageLoader::load_image(project_splash_path, splash_image);
		if (err) {
			if (OS::get_singleton()->is_stdout_verbose()) {
				print_error("- unable to load splash image from " + project_splash_path + " (" + itos(err) + ")");
			}
			splash_image.unref();
		}
	}

	if (splash_image.is_null()) {
		// Use the default
		print_verbose("Using default splash image.");
		splash_image = Ref<Image>(memnew(Image(boot_splash_png)));
	}

	if (scale_splash) {
		Size2 screen_size = Size2(ProjectSettings::get_singleton()->get("display/window/size/width"), ProjectSettings::get_singleton()->get("display/window/size/height"));
		int width, height;
		if (screen_size.width > screen_size.height) {
			// scale horizontally
			height = screen_size.height;
			width = splash_image->get_width() * screen_size.height / splash_image->get_height();
		} else {
			// scale vertically
			width = screen_size.width;
			height = splash_image->get_height() * screen_size.width / splash_image->get_width();
		}
		splash_image->resize(width, height);
	}

	// Setup the splash bg color
	bool bg_color_valid;
	Color bg_color = ProjectSettings::get_singleton()->get("application/boot_splash/bg_color", &bg_color_valid);
	if (!bg_color_valid) {
		bg_color = boot_splash_bg_color;
	}

	print_verbose("Creating splash background color image.");
	splash_bg_color_image.instantiate();
	splash_bg_color_image->create(splash_image->get_width(), splash_image->get_height(), false, splash_image->get_format());
	splash_bg_color_image->fill(bg_color);

	String processed_splash_config_xml = vformat(SPLASH_CONFIG_XML_CONTENT, bool_to_string(apply_filter));
	return processed_splash_config_xml;
}

void EditorExportPlatformAndroid::load_icon_refs(const Ref<EditorExportPreset> &p_preset, Ref<Image> &icon, Ref<Image> &foreground, Ref<Image> &background) {
	String project_icon_path = ProjectSettings::get_singleton()->get("application/config/icon");

	icon.instantiate();
	foreground.instantiate();
	background.instantiate();

	// Regular icon: user selection -> project icon -> default.
	String path = static_cast<String>(p_preset->get(launcher_icon_option)).strip_edges();
	print_verbose("Loading regular icon from " + path);
	if (path.is_empty() || ImageLoader::load_image(path, icon) != OK) {
		print_verbose("- falling back to project icon: " + project_icon_path);
		ImageLoader::load_image(project_icon_path, icon);
	}

	// Adaptive foreground: user selection -> regular icon (user selection -> project icon -> default).
	path = static_cast<String>(p_preset->get(launcher_adaptive_icon_foreground_option)).strip_edges();
	print_verbose("Loading adaptive foreground icon from " + path);
	if (path.is_empty() || ImageLoader::load_image(path, foreground) != OK) {
		print_verbose("- falling back to using the regular icon");
		foreground = icon;
	}

	// Adaptive background: user selection -> default.
	path = static_cast<String>(p_preset->get(launcher_adaptive_icon_background_option)).strip_edges();
	if (!path.is_empty()) {
		print_verbose("Loading adaptive background icon from " + path);
		ImageLoader::load_image(path, background);
	}
}

void EditorExportPlatformAndroid::store_image(const LauncherIcon launcher_icon, const Vector<uint8_t> &data) {
	store_image(launcher_icon.export_path, data);
}

void EditorExportPlatformAndroid::store_image(const String &export_path, const Vector<uint8_t> &data) {
	String img_path = export_path.insert(0, "res://android/build/");
	store_file_at_path(img_path, data);
}

void EditorExportPlatformAndroid::_copy_icons_to_gradle_project(const Ref<EditorExportPreset> &p_preset,
		const String &processed_splash_config_xml,
		const Ref<Image> &splash_image,
		const Ref<Image> &splash_bg_color_image,
		const Ref<Image> &main_image,
		const Ref<Image> &foreground,
		const Ref<Image> &background) {
	// Store the splash configuration
	if (!processed_splash_config_xml.is_empty()) {
		print_verbose("Storing processed splash configuration: " + String("\n") + processed_splash_config_xml);
		store_string_at_path(SPLASH_CONFIG_PATH, processed_splash_config_xml);
	}

	// Store the splash image
	if (splash_image.is_valid() && !splash_image->is_empty()) {
		print_verbose("Storing splash image in " + String(SPLASH_IMAGE_EXPORT_PATH));
		Vector<uint8_t> data;
		_load_image_data(splash_image, data);
		store_image(SPLASH_IMAGE_EXPORT_PATH, data);
	}

	// Store the splash bg color image
	if (splash_bg_color_image.is_valid() && !splash_bg_color_image->is_empty()) {
		print_verbose("Storing splash background image in " + String(SPLASH_BG_COLOR_PATH));
		Vector<uint8_t> data;
		_load_image_data(splash_bg_color_image, data);
		store_image(SPLASH_BG_COLOR_PATH, data);
	}

	// Prepare images to be resized for the icons. If some image ends up being uninitialized,
	// the default image from the export template will be used.

	for (int i = 0; i < icon_densities_count; ++i) {
		if (main_image.is_valid() && !main_image->is_empty()) {
			print_verbose("Processing launcher icon for dimension " + itos(launcher_icons[i].dimensions) + " into " + launcher_icons[i].export_path);
			Vector<uint8_t> data;
			_process_launcher_icons(launcher_icons[i].export_path, main_image, launcher_icons[i].dimensions, data);
			store_image(launcher_icons[i], data);
		}

		if (foreground.is_valid() && !foreground->is_empty()) {
			print_verbose("Processing launcher adaptive icon foreground for dimension " + itos(launcher_adaptive_icon_foregrounds[i].dimensions) + " into " + launcher_adaptive_icon_foregrounds[i].export_path);
			Vector<uint8_t> data;
			_process_launcher_icons(launcher_adaptive_icon_foregrounds[i].export_path, foreground,
					launcher_adaptive_icon_foregrounds[i].dimensions, data);
			store_image(launcher_adaptive_icon_foregrounds[i], data);
		}

		if (background.is_valid() && !background->is_empty()) {
			print_verbose("Processing launcher adaptive icon background for dimension " + itos(launcher_adaptive_icon_backgrounds[i].dimensions) + " into " + launcher_adaptive_icon_backgrounds[i].export_path);
			Vector<uint8_t> data;
			_process_launcher_icons(launcher_adaptive_icon_backgrounds[i].export_path, background,
					launcher_adaptive_icon_backgrounds[i].dimensions, data);
			store_image(launcher_adaptive_icon_backgrounds[i], data);
		}
	}
}

Vector<String> EditorExportPlatformAndroid::get_enabled_abis(const Ref<EditorExportPreset> &p_preset) {
	Vector<String> abis = get_abis();
	Vector<String> enabled_abis;
	for (int i = 0; i < abis.size(); ++i) {
		bool is_enabled = p_preset->get("architectures/" + abis[i]);
		if (is_enabled) {
			enabled_abis.push_back(abis[i]);
		}
	}
	return enabled_abis;
}

void EditorExportPlatformAndroid::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) {
	String driver = ProjectSettings::get_singleton()->get("rendering/driver/driver_name");
	if (driver == "GLES2") {
		r_features->push_back("etc");
	}
	// FIXME: Review what texture formats are used for Vulkan.
	if (driver == "Vulkan") {
		r_features->push_back("etc2");
	}

	Vector<String> abis = get_enabled_abis(p_preset);
	for (int i = 0; i < abis.size(); ++i) {
		r_features->push_back(abis[i]);
	}
}

void EditorExportPlatformAndroid::get_export_options(List<ExportOption> *r_options) {
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/debug", PROPERTY_HINT_GLOBAL_FILE, "*.apk"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/release", PROPERTY_HINT_GLOBAL_FILE, "*.apk"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "custom_template/use_custom_build"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "custom_template/export_format", PROPERTY_HINT_ENUM, "Export APK,Export AAB"), EXPORT_FORMAT_APK));

	Vector<PluginConfigAndroid> plugins_configs = get_plugins();
	for (int i = 0; i < plugins_configs.size(); i++) {
		print_verbose("Found Android plugin " + plugins_configs[i].name);
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "plugins/" + plugins_configs[i].name), false));
	}
	plugins_changed.clear();

	const Vector<String> abis = get_abis();
	for (int i = 0; i < abis.size(); ++i) {
		const String abi = abis[i];
		// All Android devices supporting Vulkan run 64-bit Android,
		// so there is usually no point in exporting for 32-bit Android.
		const bool is_default = abi == "arm64-v8a";
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "architectures/" + abi), is_default));
	}

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "keystore/debug", PROPERTY_HINT_GLOBAL_FILE, "*.keystore,*.jks"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "keystore/debug_user"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "keystore/debug_password"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "keystore/release", PROPERTY_HINT_GLOBAL_FILE, "*.keystore,*.jks"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "keystore/release_user"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "keystore/release_password"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "version/code", PROPERTY_HINT_RANGE, "1,4096,1,or_greater"), 1));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "version/name"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/unique_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "ext.domain.name"), "org.godotengine.$genname"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name [default if blank]"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "package/signed"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "package/classify_as_game"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "package/retain_data_on_uninstall"), false));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, launcher_icon_option, PROPERTY_HINT_FILE, "*.png"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, launcher_adaptive_icon_foreground_option, PROPERTY_HINT_FILE, "*.png"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, launcher_adaptive_icon_background_option, PROPERTY_HINT_FILE, "*.png"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "graphics/32_bits_framebuffer"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "graphics/opengl_debug"), false));

	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "xr_features/xr_mode", PROPERTY_HINT_ENUM, "Regular,Oculus Mobile VR"), 0));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "xr_features/hand_tracking", PROPERTY_HINT_ENUM, "None,Optional,Required"), 0));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "screen/immersive_mode"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "screen/support_small"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "screen/support_normal"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "screen/support_large"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "screen/support_xlarge"), true));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "user_data_backup/allow"), false));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "command_line/extra_args"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "apk_expansion/enable"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "apk_expansion/SALT"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "apk_expansion/public_key", PROPERTY_HINT_MULTILINE_TEXT), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "permissions/custom_permissions"), PackedStringArray()));

	const char **perms = android_perms;
	while (*perms) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "permissions/" + String(*perms).to_lower()), false));
		perms++;
	}
}

String EditorExportPlatformAndroid::get_name() const {
	return "Android";
}

String EditorExportPlatformAndroid::get_os_name() const {
	return "Android";
}

Ref<Texture2D> EditorExportPlatformAndroid::get_logo() const {
	return logo;
}

bool EditorExportPlatformAndroid::should_update_export_options() {
	bool export_options_changed = plugins_changed.is_set();
	if (export_options_changed) {
		// don't clear unless we're reporting true, to avoid race
		plugins_changed.clear();
	}
	return export_options_changed;
}

bool EditorExportPlatformAndroid::poll_export() {
	bool dc = devices_changed.is_set();
	if (dc) {
		// don't clear unless we're reporting true, to avoid race
		devices_changed.clear();
	}
	return dc;
}

int EditorExportPlatformAndroid::get_options_count() const {
	MutexLock lock(device_lock);
	return devices.size();
}

String EditorExportPlatformAndroid::get_options_tooltip() const {
	return TTR("Select device from the list");
}

String EditorExportPlatformAndroid::get_option_label(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, devices.size(), "");
	MutexLock lock(device_lock);
	return devices[p_index].name;
}

String EditorExportPlatformAndroid::get_option_tooltip(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, devices.size(), "");
	MutexLock lock(device_lock);
	String s = devices[p_index].description;
	if (devices.size() == 1) {
		// Tooltip will be:
		// Name
		// Description
		s = devices[p_index].name + "\n\n" + s;
	}
	return s;
}

Error EditorExportPlatformAndroid::run(const Ref<EditorExportPreset> &p_preset, int p_device, int p_debug_flags) {
	ERR_FAIL_INDEX_V(p_device, devices.size(), ERR_INVALID_PARAMETER);

	String can_export_error;
	bool can_export_missing_templates;
	if (!can_export(p_preset, can_export_error, can_export_missing_templates)) {
		EditorNode::add_io_error(can_export_error);
		return ERR_UNCONFIGURED;
	}

	MutexLock lock(device_lock);

	EditorProgress ep("run", vformat(TTR("Running on %s"), devices[p_device].name), 3);

	String adb = get_adb_path();

	// Export_temp APK.
	if (ep.step(TTR("Exporting APK..."), 0)) {
		return ERR_SKIP;
	}

	const bool use_remote = (p_debug_flags & DEBUG_FLAG_REMOTE_DEBUG) || (p_debug_flags & DEBUG_FLAG_DUMB_CLIENT);
	const bool use_reverse = devices[p_device].api_level >= 21;

	if (use_reverse) {
		p_debug_flags |= DEBUG_FLAG_REMOTE_DEBUG_LOCALHOST;
	}

	String tmp_export_path = EditorPaths::get_singleton()->get_cache_dir().plus_file("tmpexport." + uitos(OS::get_singleton()->get_unix_time()) + ".apk");

#define CLEANUP_AND_RETURN(m_err)                         \
	{                                                     \
		DirAccess::remove_file_or_error(tmp_export_path); \
		return m_err;                                     \
	}

	// Export to temporary APK before sending to device.
	Error err = export_project_helper(p_preset, true, tmp_export_path, EXPORT_FORMAT_APK, true, p_debug_flags);

	if (err != OK) {
		CLEANUP_AND_RETURN(err);
	}

	List<String> args;
	int rv;
	String output;

	bool remove_prev = EDITOR_GET("export/android/one_click_deploy_clear_previous_install");
	String version_name = p_preset->get("version/name");
	String package_name = p_preset->get("package/unique_name");

	if (remove_prev) {
		if (ep.step(TTR("Uninstalling..."), 1)) {
			CLEANUP_AND_RETURN(ERR_SKIP);
		}

		print_line("Uninstalling previous version: " + devices[p_device].name);

		args.push_back("-s");
		args.push_back(devices[p_device].id);
		args.push_back("uninstall");
		args.push_back(get_package_name(package_name));

		output.clear();
		err = OS::get_singleton()->execute(adb, args, &output, &rv, true);
		print_verbose(output);
	}

	print_line("Installing to device (please wait...): " + devices[p_device].name);
	if (ep.step(TTR("Installing to device, please wait..."), 2)) {
		CLEANUP_AND_RETURN(ERR_SKIP);
	}

	args.clear();
	args.push_back("-s");
	args.push_back(devices[p_device].id);
	args.push_back("install");
	args.push_back("-r");
	args.push_back(tmp_export_path);

	output.clear();
	err = OS::get_singleton()->execute(adb, args, &output, &rv, true);
	print_verbose(output);
	if (err || rv != 0) {
		EditorNode::add_io_error(vformat(TTR("Could not install to device: %s"), output));
		CLEANUP_AND_RETURN(ERR_CANT_CREATE);
	}

	if (use_remote) {
		if (use_reverse) {
			static const char *const msg = "--- Device API >= 21; debugging over USB ---";
			EditorNode::get_singleton()->get_log()->add_message(msg, EditorLog::MSG_TYPE_EDITOR);
			print_line(String(msg).to_upper());

			args.clear();
			args.push_back("-s");
			args.push_back(devices[p_device].id);
			args.push_back("reverse");
			args.push_back("--remove-all");
			output.clear();
			OS::get_singleton()->execute(adb, args, &output, &rv, true);
			print_verbose(output);

			if (p_debug_flags & DEBUG_FLAG_REMOTE_DEBUG) {
				int dbg_port = EditorSettings::get_singleton()->get("network/debug/remote_port");
				args.clear();
				args.push_back("-s");
				args.push_back(devices[p_device].id);
				args.push_back("reverse");
				args.push_back("tcp:" + itos(dbg_port));
				args.push_back("tcp:" + itos(dbg_port));

				output.clear();
				OS::get_singleton()->execute(adb, args, &output, &rv, true);
				print_verbose(output);
				print_line("Reverse result: " + itos(rv));
			}

			if (p_debug_flags & DEBUG_FLAG_DUMB_CLIENT) {
				int fs_port = EditorSettings::get_singleton()->get("filesystem/file_server/port");

				args.clear();
				args.push_back("-s");
				args.push_back(devices[p_device].id);
				args.push_back("reverse");
				args.push_back("tcp:" + itos(fs_port));
				args.push_back("tcp:" + itos(fs_port));

				output.clear();
				err = OS::get_singleton()->execute(adb, args, &output, &rv, true);
				print_verbose(output);
				print_line("Reverse result2: " + itos(rv));
			}
		} else {
			static const char *const msg = "--- Device API < 21; debugging over Wi-Fi ---";
			EditorNode::get_singleton()->get_log()->add_message(msg, EditorLog::MSG_TYPE_EDITOR);
			print_line(String(msg).to_upper());
		}
	}

	if (ep.step(TTR("Running on device..."), 3)) {
		CLEANUP_AND_RETURN(ERR_SKIP);
	}
	args.clear();
	args.push_back("-s");
	args.push_back(devices[p_device].id);
	args.push_back("shell");
	args.push_back("am");
	args.push_back("start");
	if ((bool)EditorSettings::get_singleton()->get("export/android/force_system_user") && devices[p_device].api_level >= 17) { // Multi-user introduced in Android 17
		args.push_back("--user");
		args.push_back("0");
	}
	args.push_back("-a");
	args.push_back("android.intent.action.MAIN");
	args.push_back("-n");
	args.push_back(get_package_name(package_name) + "/com.godot.game.GodotApp");

	output.clear();
	err = OS::get_singleton()->execute(adb, args, &output, &rv, true);
	print_verbose(output);
	if (err || rv != 0) {
		EditorNode::add_io_error(TTR("Could not execute on device."));
		CLEANUP_AND_RETURN(ERR_CANT_CREATE);
	}

	CLEANUP_AND_RETURN(OK);
#undef CLEANUP_AND_RETURN
}

Ref<Texture2D> EditorExportPlatformAndroid::get_run_icon() const {
	return run_icon;
}

String EditorExportPlatformAndroid::get_adb_path() {
	String exe_ext = "";
	if (OS::get_singleton()->get_name() == "Windows") {
		exe_ext = ".exe";
	}
	String sdk_path = EditorSettings::get_singleton()->get("export/android/android_sdk_path");
	return sdk_path.plus_file("platform-tools/adb" + exe_ext);
}

String EditorExportPlatformAndroid::get_apksigner_path() {
	String exe_ext = "";
	if (OS::get_singleton()->get_name() == "Windows") {
		exe_ext = ".bat";
	}
	String apksigner_command_name = "apksigner" + exe_ext;
	String sdk_path = EditorSettings::get_singleton()->get("export/android/android_sdk_path");
	String apksigner_path = "";

	Error errn;
	String build_tools_dir = sdk_path.plus_file("build-tools");
	DirAccessRef da = DirAccess::open(build_tools_dir, &errn);
	if (errn != OK) {
		print_error("Unable to open Android 'build-tools' directory.");
		return apksigner_path;
	}

	// There are additional versions directories we need to go through.
	da->list_dir_begin();
	String sub_dir = da->get_next();
	while (!sub_dir.is_empty()) {
		if (!sub_dir.begins_with(".") && da->current_is_dir()) {
			// Check if the tool is here.
			String tool_path = build_tools_dir.plus_file(sub_dir).plus_file(apksigner_command_name);
			if (FileAccess::exists(tool_path)) {
				apksigner_path = tool_path;
				break;
			}
		}
		sub_dir = da->get_next();
	}
	da->list_dir_end();

	if (apksigner_path.is_empty()) {
		EditorNode::get_singleton()->show_warning(TTR("Unable to find the 'apksigner' tool."));
	}

	return apksigner_path;
}

bool EditorExportPlatformAndroid::can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {
	String err;
	bool valid = false;

	// Look for export templates (first official, and if defined custom templates).

	if (!bool(p_preset->get("custom_template/use_custom_build"))) {
		String template_err;
		bool dvalid = false;
		bool rvalid = false;
		bool has_export_templates = false;

		if (p_preset->get("custom_template/debug") != "") {
			dvalid = FileAccess::exists(p_preset->get("custom_template/debug"));
			if (!dvalid) {
				template_err += TTR("Custom debug template not found.") + "\n";
			}
		} else {
			has_export_templates |= exists_export_template("android_debug.apk", &template_err);
		}

		if (p_preset->get("custom_template/release") != "") {
			rvalid = FileAccess::exists(p_preset->get("custom_template/release"));
			if (!rvalid) {
				template_err += TTR("Custom release template not found.") + "\n";
			}
		} else {
			has_export_templates |= exists_export_template("android_release.apk", &template_err);
		}

		r_missing_templates = !has_export_templates;
		valid = dvalid || rvalid || has_export_templates;
		if (!valid) {
			err += template_err;
		}
	} else {
		bool installed_android_build_template = FileAccess::exists("res://android/build/build.gradle");
		if (!installed_android_build_template) {
			r_missing_templates = !exists_export_template("android_source.zip", &err);
			err += TTR("Android build template not installed in the project. Install it from the Project menu.") + "\n";
		} else {
			r_missing_templates = false;
		}

		valid = installed_android_build_template && !r_missing_templates;
	}

	// Validate the rest of the configuration.

	String dk = p_preset->get("keystore/debug");
	String dk_user = p_preset->get("keystore/debug_user");
	String dk_password = p_preset->get("keystore/debug_password");

	if ((dk.is_empty() || dk_user.is_empty() || dk_password.is_empty()) && (!dk.is_empty() || !dk_user.is_empty() || !dk_password.is_empty())) {
		valid = false;
		err += TTR("Either Debug Keystore, Debug User AND Debug Password settings must be configured OR none of them.") + "\n";
	}

	if (!FileAccess::exists(dk)) {
		dk = EditorSettings::get_singleton()->get("export/android/debug_keystore");
		if (!FileAccess::exists(dk)) {
			valid = false;
			err += TTR("Debug keystore not configured in the Editor Settings nor in the preset.") + "\n";
		}
	}

	String rk = p_preset->get("keystore/release");
	String rk_user = p_preset->get("keystore/release_user");
	String rk_password = p_preset->get("keystore/release_password");

	if ((rk.is_empty() || rk_user.is_empty() || rk_password.is_empty()) && (!rk.is_empty() || !rk_user.is_empty() || !rk_password.is_empty())) {
		valid = false;
		err += TTR("Either Release Keystore, Release User AND Release Password settings must be configured OR none of them.") + "\n";
	}

	if (!rk.is_empty() && !FileAccess::exists(rk)) {
		valid = false;
		err += TTR("Release keystore incorrectly configured in the export preset.") + "\n";
	}

	String sdk_path = EditorSettings::get_singleton()->get("export/android/android_sdk_path");
	if (sdk_path == "") {
		err += TTR("A valid Android SDK path is required in Editor Settings.") + "\n";
		valid = false;
	} else {
		Error errn;
		// Check for the platform-tools directory.
		DirAccessRef da = DirAccess::open(sdk_path.plus_file("platform-tools"), &errn);
		if (errn != OK) {
			err += TTR("Invalid Android SDK path in Editor Settings.");
			err += TTR("Missing 'platform-tools' directory!");
			err += "\n";
			valid = false;
		}

		// Validate that adb is available
		String adb_path = get_adb_path();
		if (!FileAccess::exists(adb_path)) {
			err += TTR("Unable to find Android SDK platform-tools' adb command.");
			err += TTR("Please check in the Android SDK directory specified in Editor Settings.");
			err += "\n";
			valid = false;
		}

		// Check for the build-tools directory.
		DirAccessRef build_tools_da = DirAccess::open(sdk_path.plus_file("build-tools"), &errn);
		if (errn != OK) {
			err += TTR("Invalid Android SDK path in Editor Settings.");
			err += TTR("Missing 'build-tools' directory!");
			err += "\n";
			valid = false;
		}

		// Validate that apksigner is available
		String apksigner_path = get_apksigner_path();
		if (!FileAccess::exists(apksigner_path)) {
			err += TTR("Unable to find Android SDK build-tools' apksigner command.");
			err += TTR("Please check in the Android SDK directory specified in Editor Settings.");
			err += "\n";
			valid = false;
		}
	}

	bool apk_expansion = p_preset->get("apk_expansion/enable");

	if (apk_expansion) {
		String apk_expansion_pkey = p_preset->get("apk_expansion/public_key");

		if (apk_expansion_pkey == "") {
			valid = false;

			err += TTR("Invalid public key for APK expansion.") + "\n";
		}
	}

	String pn = p_preset->get("package/unique_name");
	String pn_err;

	if (!is_package_name_valid(get_package_name(pn), &pn_err)) {
		valid = false;
		err += TTR("Invalid package name:") + " " + pn_err + "\n";
	}

	String etc_error = test_etc2();
	if (etc_error != String()) {
		valid = false;
		err += etc_error;
	}

	// Ensure that `Use Custom Build` is enabled if a plugin is selected.
	String enabled_plugins_names = PluginConfigAndroid::get_plugins_names(get_enabled_plugins(p_preset));
	bool custom_build_enabled = p_preset->get("custom_template/use_custom_build");
	if (!enabled_plugins_names.is_empty() && !custom_build_enabled) {
		valid = false;
		err += TTR("\"Use Custom Build\" must be enabled to use the plugins.");
		err += "\n";
	}

	// Validate the Xr features are properly populated
	int xr_mode_index = p_preset->get("xr_features/xr_mode");
	int hand_tracking = p_preset->get("xr_features/hand_tracking");
	if (xr_mode_index != /* XRMode.OVR*/ 1) {
		if (hand_tracking > 0) {
			valid = false;
			err += TTR("\"Hand Tracking\" is only valid when \"Xr Mode\" is \"Oculus Mobile VR\".");
			err += "\n";
		}
	}

	if (int(p_preset->get("custom_template/export_format")) == EXPORT_FORMAT_AAB &&
			!bool(p_preset->get("custom_template/use_custom_build"))) {
		valid = false;
		err += TTR("\"Export AAB\" is only valid when \"Use Custom Build\" is enabled.");
		err += "\n";
	}

	r_error = err;
	return valid;
}

List<String> EditorExportPlatformAndroid::get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const {
	List<String> list;
	list.push_back("apk");
	list.push_back("aab");
	return list;
}

String EditorExportPlatformAndroid::get_apk_expansion_fullpath(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
	int version_code = p_preset->get("version/code");
	String package_name = p_preset->get("package/unique_name");
	String apk_file_name = "main." + itos(version_code) + "." + get_package_name(package_name) + ".obb";
	String fullpath = p_path.get_base_dir().plus_file(apk_file_name);
	return fullpath;
}

Error EditorExportPlatformAndroid::save_apk_expansion_file(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
	String fullpath = get_apk_expansion_fullpath(p_preset, p_path);
	Error err = save_pack(p_preset, fullpath);
	return err;
}

void EditorExportPlatformAndroid::get_command_line_flags(const Ref<EditorExportPreset> &p_preset, const String &p_path, int p_flags, Vector<uint8_t> &r_command_line_flags) {
	String cmdline = p_preset->get("command_line/extra_args");
	Vector<String> command_line_strings = cmdline.strip_edges().split(" ");
	for (int i = 0; i < command_line_strings.size(); i++) {
		if (command_line_strings[i].strip_edges().length() == 0) {
			command_line_strings.remove(i);
			i--;
		}
	}

	gen_export_flags(command_line_strings, p_flags);

	bool apk_expansion = p_preset->get("apk_expansion/enable");
	if (apk_expansion) {
		String fullpath = get_apk_expansion_fullpath(p_preset, p_path);
		String apk_expansion_public_key = p_preset->get("apk_expansion/public_key");

		command_line_strings.push_back("--use_apk_expansion");
		command_line_strings.push_back("--apk_expansion_md5");
		command_line_strings.push_back(FileAccess::get_md5(fullpath));
		command_line_strings.push_back("--apk_expansion_key");
		command_line_strings.push_back(apk_expansion_public_key.strip_edges());
	}

	int xr_mode_index = p_preset->get("xr_features/xr_mode");
	if (xr_mode_index == 1) {
		command_line_strings.push_back("--xr_mode_ovr");
	} else { // XRMode.REGULAR is the default.
		command_line_strings.push_back("--xr_mode_regular");
	}

	bool use_32_bit_framebuffer = p_preset->get("graphics/32_bits_framebuffer");
	if (use_32_bit_framebuffer) {
		command_line_strings.push_back("--use_depth_32");
	}

	bool immersive = p_preset->get("screen/immersive_mode");
	if (immersive) {
		command_line_strings.push_back("--use_immersive");
	}

	bool debug_opengl = p_preset->get("graphics/opengl_debug");
	if (debug_opengl) {
		command_line_strings.push_back("--debug_opengl");
	}

	if (command_line_strings.size()) {
		r_command_line_flags.resize(4);
		encode_uint32(command_line_strings.size(), &r_command_line_flags.write[0]);
		for (int i = 0; i < command_line_strings.size(); i++) {
			print_line(itos(i) + " param: " + command_line_strings[i]);
			CharString command_line_argument = command_line_strings[i].utf8();
			int base = r_command_line_flags.size();
			int length = command_line_argument.length();
			if (length == 0) {
				continue;
			}
			r_command_line_flags.resize(base + 4 + length);
			encode_uint32(length, &r_command_line_flags.write[base]);
			memcpy(&r_command_line_flags.write[base + 4], command_line_argument.ptr(), length);
		}
	}
}

Error EditorExportPlatformAndroid::sign_apk(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &export_path, EditorProgress &ep) {
	int export_format = int(p_preset->get("custom_template/export_format"));
	String export_label = export_format == EXPORT_FORMAT_AAB ? "AAB" : "APK";
	String release_keystore = p_preset->get("keystore/release");
	String release_username = p_preset->get("keystore/release_user");
	String release_password = p_preset->get("keystore/release_password");

	String apksigner = get_apksigner_path();
	print_verbose("Starting signing of the " + export_label + " binary using " + apksigner);
	if (!FileAccess::exists(apksigner)) {
		EditorNode::add_io_error(vformat(TTR("'apksigner' could not be found.\nPlease check the command is available in the Android SDK build-tools directory.\nThe resulting %s is unsigned."), export_label));
		return OK;
	}

	String keystore;
	String password;
	String user;
	if (p_debug) {
		keystore = p_preset->get("keystore/debug");
		password = p_preset->get("keystore/debug_password");
		user = p_preset->get("keystore/debug_user");

		if (keystore.is_empty()) {
			keystore = EditorSettings::get_singleton()->get("export/android/debug_keystore");
			password = EditorSettings::get_singleton()->get("export/android/debug_keystore_pass");
			user = EditorSettings::get_singleton()->get("export/android/debug_keystore_user");
		}

		if (ep.step(vformat(TTR("Signing debug %s..."), export_label), 104)) {
			return ERR_SKIP;
		}

	} else {
		keystore = release_keystore;
		password = release_password;
		user = release_username;

		if (ep.step(vformat(TTR("Signing release %s..."), export_label), 104)) {
			return ERR_SKIP;
		}
	}

	if (!FileAccess::exists(keystore)) {
		EditorNode::add_io_error(TTR("Could not find keystore, unable to export."));
		return ERR_FILE_CANT_OPEN;
	}

	String output;
	List<String> args;
	args.push_back("sign");
	args.push_back("--verbose");
	args.push_back("--ks");
	args.push_back(keystore);
	args.push_back("--ks-pass");
	args.push_back("pass:" + password);
	args.push_back("--ks-key-alias");
	args.push_back(user);
	args.push_back(export_path);
	if (p_debug) {
		// We only print verbose logs for debug builds to avoid leaking release keystore credentials.
		print_verbose("Signing debug binary using: " + String("\n") + apksigner + " " + join_list(args, String(" ")));
	}
	int retval;
	output.clear();
	OS::get_singleton()->execute(apksigner, args, &output, &retval, true);
	print_verbose(output);
	if (retval) {
		EditorNode::add_io_error(vformat(TTR("'apksigner' returned with error #%d"), retval));
		return ERR_CANT_CREATE;
	}

	if (ep.step(vformat(TTR("Verifying %s..."), export_label), 105)) {
		return ERR_SKIP;
	}

	args.clear();
	args.push_back("verify");
	args.push_back("--verbose");
	args.push_back(export_path);
	if (p_debug) {
		print_verbose("Verifying signed build using: " + String("\n") + apksigner + " " + join_list(args, String(" ")));
	}

	output.clear();
	OS::get_singleton()->execute(apksigner, args, &output, &retval, true);
	print_verbose(output);
	if (retval) {
		EditorNode::add_io_error(vformat(TTR("'apksigner' verification of %s failed."), export_label));
		return ERR_CANT_CREATE;
	}

	print_verbose("Successfully completed signing build.");
	return OK;
}

void EditorExportPlatformAndroid::_clear_assets_directory() {
	DirAccessRef da_res = DirAccess::create(DirAccess::ACCESS_RESOURCES);

	// Clear the APK assets directory
	if (da_res->dir_exists(APK_ASSETS_DIRECTORY)) {
		print_verbose("Clearing APK assets directory..");
		DirAccessRef da_assets = DirAccess::open(APK_ASSETS_DIRECTORY);
		da_assets->erase_contents_recursive();
		da_res->remove(APK_ASSETS_DIRECTORY);
	}

	// Clear the AAB assets directory
	if (da_res->dir_exists(AAB_ASSETS_DIRECTORY)) {
		print_verbose("Clearing AAB assets directory..");
		DirAccessRef da_assets = DirAccess::open(AAB_ASSETS_DIRECTORY);
		da_assets->erase_contents_recursive();
		da_res->remove(AAB_ASSETS_DIRECTORY);
	}
}

void EditorExportPlatformAndroid::_remove_copied_libs() {
	print_verbose("Removing previously installed libraries...");
	Error error;
	String libs_json = FileAccess::get_file_as_string(GDNATIVE_LIBS_PATH, &error);
	if (error || libs_json.is_empty()) {
		print_verbose("No previously installed libraries found");
		return;
	}

	JSON json;
	error = json.parse(libs_json);
	ERR_FAIL_COND_MSG(error, "Error parsing \"" + libs_json + "\" on line " + itos(json.get_error_line()) + ": " + json.get_error_message());

	Vector<String> libs = json.get_data();
	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	for (int i = 0; i < libs.size(); i++) {
		print_verbose("Removing previously installed library " + libs[i]);
		da->remove(libs[i]);
	}
	da->remove(GDNATIVE_LIBS_PATH);
}

String EditorExportPlatformAndroid::join_list(List<String> parts, const String &separator) const {
	String ret;
	for (int i = 0; i < parts.size(); ++i) {
		if (i > 0) {
			ret += separator;
		}
		ret += parts[i];
	}
	return ret;
}

Error EditorExportPlatformAndroid::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	int export_format = int(p_preset->get("custom_template/export_format"));
	bool should_sign = p_preset->get("package/signed");
	return export_project_helper(p_preset, p_debug, p_path, export_format, should_sign, p_flags);
}

Error EditorExportPlatformAndroid::export_project_helper(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int export_format, bool should_sign, int p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

	String src_apk;
	Error err;

	EditorProgress ep("export", TTR("Exporting for Android"), 105, true);

	bool use_custom_build = bool(p_preset->get("custom_template/use_custom_build"));
	bool p_give_internet = p_flags & (DEBUG_FLAG_DUMB_CLIENT | DEBUG_FLAG_REMOTE_DEBUG);
	bool apk_expansion = p_preset->get("apk_expansion/enable");
	Vector<String> enabled_abis = get_enabled_abis(p_preset);

	print_verbose("Exporting for Android...");
	print_verbose("- debug build: " + bool_to_string(p_debug));
	print_verbose("- export path: " + p_path);
	print_verbose("- export format: " + itos(export_format));
	print_verbose("- sign build: " + bool_to_string(should_sign));
	print_verbose("- custom build enabled: " + bool_to_string(use_custom_build));
	print_verbose("- apk expansion enabled: " + bool_to_string(apk_expansion));
	print_verbose("- enabled abis: " + String(",").join(enabled_abis));
	print_verbose("- export filter: " + itos(p_preset->get_export_filter()));
	print_verbose("- include filter: " + p_preset->get_include_filter());
	print_verbose("- exclude filter: " + p_preset->get_exclude_filter());

	Ref<Image> splash_image;
	Ref<Image> splash_bg_color_image;
	String processed_splash_config_xml = load_splash_refs(splash_image, splash_bg_color_image);

	Ref<Image> main_image;
	Ref<Image> foreground;
	Ref<Image> background;

	load_icon_refs(p_preset, main_image, foreground, background);

	Vector<uint8_t> command_line_flags;
	// Write command line flags into the command_line_flags variable.
	get_command_line_flags(p_preset, p_path, p_flags, command_line_flags);

	if (export_format == EXPORT_FORMAT_AAB) {
		if (!p_path.ends_with(".aab")) {
			EditorNode::get_singleton()->show_warning(TTR("Invalid filename! Android App Bundle requires the *.aab extension."));
			return ERR_UNCONFIGURED;
		}
		if (apk_expansion) {
			EditorNode::get_singleton()->show_warning(TTR("APK Expansion not compatible with Android App Bundle."));
			return ERR_UNCONFIGURED;
		}
	}
	if (export_format == EXPORT_FORMAT_APK && !p_path.ends_with(".apk")) {
		EditorNode::get_singleton()->show_warning(
				TTR("Invalid filename! Android APK requires the *.apk extension."));
		return ERR_UNCONFIGURED;
	}
	if (export_format > EXPORT_FORMAT_AAB || export_format < EXPORT_FORMAT_APK) {
		EditorNode::add_io_error(TTR("Unsupported export format!\n"));
		return ERR_UNCONFIGURED; //TODO: is this the right error?
	}

	if (use_custom_build) {
		print_verbose("Starting custom build..");
		//test that installed build version is alright
		{
			print_verbose("Checking build version..");
			FileAccessRef f = FileAccess::open("res://android/.build_version", FileAccess::READ);
			if (!f) {
				EditorNode::get_singleton()->show_warning(TTR("Trying to build from a custom built template, but no version info for it exists. Please reinstall from the 'Project' menu."));
				return ERR_UNCONFIGURED;
			}
			String version = f->get_line().strip_edges();
			print_verbose("- build version: " + version);
			f->close();
			if (version != VERSION_FULL_CONFIG) {
				EditorNode::get_singleton()->show_warning(vformat(TTR("Android build version mismatch:\n   Template installed: %s\n   Godot Version: %s\nPlease reinstall Android build template from 'Project' menu."), version, VERSION_FULL_CONFIG));
				return ERR_UNCONFIGURED;
			}
		}
		const String assets_directory = get_assets_directory(p_preset, export_format);
		String sdk_path = EDITOR_GET("export/android/android_sdk_path");
		ERR_FAIL_COND_V_MSG(sdk_path.is_empty(), ERR_UNCONFIGURED, "Android SDK path must be configured in Editor Settings at 'export/android/android_sdk_path'.");
		print_verbose("Android sdk path: " + sdk_path);

		// TODO: should we use "package/name" or "application/config/name"?
		String project_name = get_project_name(p_preset->get("package/name"));
		err = _create_project_name_strings_files(p_preset, project_name); //project name localization.
		if (err != OK) {
			EditorNode::add_io_error(TTR("Unable to overwrite res://android/build/res/*.xml files with project name"));
		}
		// Copies the project icon files into the appropriate Gradle project directory.
		_copy_icons_to_gradle_project(p_preset, processed_splash_config_xml, splash_image, splash_bg_color_image, main_image, foreground, background);
		// Write an AndroidManifest.xml file into the Gradle project directory.
		_write_tmp_manifest(p_preset, p_give_internet, p_debug);

		//stores all the project files inside the Gradle project directory. Also includes all ABIs
		_clear_assets_directory();
		_remove_copied_libs();
		if (!apk_expansion) {
			print_verbose("Exporting project files..");
			CustomExportData user_data;
			user_data.assets_directory = assets_directory;
			user_data.debug = p_debug;
			err = export_project_files(p_preset, rename_and_store_file_in_gradle_project, &user_data, copy_gradle_so);
			if (err != OK) {
				EditorNode::add_io_error(TTR("Could not export project files to gradle project\n"));
				return err;
			}
			if (user_data.libs.size() > 0) {
				FileAccessRef fa = FileAccess::open(GDNATIVE_LIBS_PATH, FileAccess::WRITE);
				JSON json;
				fa->store_string(json.stringify(user_data.libs, "\t"));
				fa->close();
			}
		} else {
			print_verbose("Saving apk expansion file..");
			err = save_apk_expansion_file(p_preset, p_path);
			if (err != OK) {
				EditorNode::add_io_error(TTR("Could not write expansion package file!"));
				return err;
			}
		}
		print_verbose("Storing command line flags..");
		store_file_at_path(assets_directory + "/_cl_", command_line_flags);

		print_verbose("Updating ANDROID_HOME environment to " + sdk_path);
		OS::get_singleton()->set_environment("ANDROID_HOME", sdk_path); //set and overwrite if required
		String build_command;

#ifdef WINDOWS_ENABLED
		build_command = "gradlew.bat";
#else
		build_command = "gradlew";
#endif

		String build_path = ProjectSettings::get_singleton()->get_resource_path().plus_file("android/build");
		build_command = build_path.plus_file(build_command);

		String package_name = get_package_name(p_preset->get("package/unique_name"));
		String version_code = itos(p_preset->get("version/code"));
		String version_name = p_preset->get("version/name");
		String enabled_abi_string = String("|").join(enabled_abis);
		String sign_flag = should_sign ? "true" : "false";
		String zipalign_flag = "true";

		Vector<PluginConfigAndroid> enabled_plugins = get_enabled_plugins(p_preset);
		String local_plugins_binaries = PluginConfigAndroid::get_plugins_binaries(PluginConfigAndroid::BINARY_TYPE_LOCAL, enabled_plugins);
		String remote_plugins_binaries = PluginConfigAndroid::get_plugins_binaries(PluginConfigAndroid::BINARY_TYPE_REMOTE, enabled_plugins);
		String custom_maven_repos = PluginConfigAndroid::get_plugins_custom_maven_repos(enabled_plugins);
		bool clean_build_required = is_clean_build_required(enabled_plugins);

		List<String> cmdline;
		if (clean_build_required) {
			cmdline.push_back("clean");
		}

		String build_type = p_debug ? "Debug" : "Release";
		if (export_format == EXPORT_FORMAT_AAB) {
			String bundle_build_command = vformat("bundle%s", build_type);
			cmdline.push_back(bundle_build_command);
		} else if (export_format == EXPORT_FORMAT_APK) {
			String apk_build_command = vformat("assemble%s", build_type);
			cmdline.push_back(apk_build_command);
		}

		cmdline.push_back("-p"); // argument to specify the start directory.
		cmdline.push_back(build_path); // start directory.
		cmdline.push_back("-Pexport_package_name=" + package_name); // argument to specify the package name.
		cmdline.push_back("-Pexport_version_code=" + version_code); // argument to specify the version code.
		cmdline.push_back("-Pexport_version_name=" + version_name); // argument to specify the version name.
		cmdline.push_back("-Pexport_enabled_abis=" + enabled_abi_string); // argument to specify enabled ABIs.
		cmdline.push_back("-Pplugins_local_binaries=" + local_plugins_binaries); // argument to specify the list of plugins local dependencies.
		cmdline.push_back("-Pplugins_remote_binaries=" + remote_plugins_binaries); // argument to specify the list of plugins remote dependencies.
		cmdline.push_back("-Pplugins_maven_repos=" + custom_maven_repos); // argument to specify the list of custom maven repos for the plugins dependencies.
		cmdline.push_back("-Pperform_zipalign=" + zipalign_flag); // argument to specify whether the build should be zipaligned.
		cmdline.push_back("-Pperform_signing=" + sign_flag); // argument to specify whether the build should be signed.
		cmdline.push_back("-Pgodot_editor_version=" + String(VERSION_FULL_CONFIG));

		// NOTE: The release keystore is not included in the verbose logging
		// to avoid accidentally leaking sensitive information when sharing verbose logs for troubleshooting.
		// Any non-sensitive additions to the command line arguments must be done above this section.
		// Sensitive additions must be done below the logging statement.
		print_verbose("Build Android project using gradle command: " + String("\n") + build_command + " " + join_list(cmdline, String(" ")));

		if (should_sign) {
			if (p_debug) {
				String debug_keystore = p_preset->get("keystore/debug");
				String debug_password = p_preset->get("keystore/debug_password");
				String debug_user = p_preset->get("keystore/debug_user");

				if (debug_keystore.is_empty()) {
					debug_keystore = EditorSettings::get_singleton()->get("export/android/debug_keystore");
					debug_password = EditorSettings::get_singleton()->get("export/android/debug_keystore_pass");
					debug_user = EditorSettings::get_singleton()->get("export/android/debug_keystore_user");
				}

				cmdline.push_back("-Pdebug_keystore_file=" + debug_keystore); // argument to specify the debug keystore file.
				cmdline.push_back("-Pdebug_keystore_alias=" + debug_user); // argument to specify the debug keystore alias.
				cmdline.push_back("-Pdebug_keystore_password=" + debug_password); // argument to specify the debug keystore password.
			} else {
				// Pass the release keystore info as well
				String release_keystore = p_preset->get("keystore/release");
				String release_username = p_preset->get("keystore/release_user");
				String release_password = p_preset->get("keystore/release_password");
				if (!FileAccess::exists(release_keystore)) {
					EditorNode::add_io_error(TTR("Could not find keystore, unable to export."));
					return ERR_FILE_CANT_OPEN;
				}

				cmdline.push_back("-Prelease_keystore_file=" + release_keystore); // argument to specify the release keystore file.
				cmdline.push_back("-Prelease_keystore_alias=" + release_username); // argument to specify the release keystore alias.
				cmdline.push_back("-Prelease_keystore_password=" + release_password); // argument to specify the release keystore password.
			}
		}

		int result = EditorNode::get_singleton()->execute_and_show_output(TTR("Building Android Project (gradle)"), build_command, cmdline);
		if (result != 0) {
			EditorNode::get_singleton()->show_warning(TTR("Building of Android project failed, check output for the error.\nAlternatively visit docs.godotengine.org for Android build documentation."));
			return ERR_CANT_CREATE;
		}

		List<String> copy_args;
		String copy_command;
		if (export_format == EXPORT_FORMAT_AAB) {
			copy_command = vformat("copyAndRename%sAab", build_type);
		} else if (export_format == EXPORT_FORMAT_APK) {
			copy_command = vformat("copyAndRename%sApk", build_type);
		}

		copy_args.push_back(copy_command);

		copy_args.push_back("-p"); // argument to specify the start directory.
		copy_args.push_back(build_path); // start directory.

		String export_filename = p_path.get_file();
		String export_path = p_path.get_base_dir();
		if (export_path.is_relative_path()) {
			export_path = OS::get_singleton()->get_resource_dir().plus_file(export_path);
		}
		export_path = ProjectSettings::get_singleton()->globalize_path(export_path).simplify_path();

		copy_args.push_back("-Pexport_path=file:" + export_path);
		copy_args.push_back("-Pexport_filename=" + export_filename);

		print_verbose("Copying Android binary using gradle command: " + String("\n") + build_command + " " + join_list(copy_args, String(" ")));
		int copy_result = EditorNode::get_singleton()->execute_and_show_output(TTR("Moving output"), build_command, copy_args);
		if (copy_result != 0) {
			EditorNode::get_singleton()->show_warning(TTR("Unable to copy and rename export file, check gradle project directory for outputs."));
			return ERR_CANT_CREATE;
		}

		print_verbose("Successfully completed Android custom build.");
		return OK;
	}
	// This is the start of the Legacy build system
	print_verbose("Starting legacy build system..");
	if (p_debug) {
		src_apk = p_preset->get("custom_template/debug");
	} else {
		src_apk = p_preset->get("custom_template/release");
	}
	src_apk = src_apk.strip_edges();
	if (src_apk == "") {
		if (p_debug) {
			src_apk = find_export_template("android_debug.apk");
		} else {
			src_apk = find_export_template("android_release.apk");
		}
		if (src_apk == "") {
			EditorNode::add_io_error(vformat(TTR("Package not found: %s"), src_apk));
			return ERR_FILE_NOT_FOUND;
		}
	}

	if (!DirAccess::exists(p_path.get_base_dir())) {
		return ERR_FILE_BAD_PATH;
	}

	FileAccess *src_f = nullptr;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

	if (ep.step(TTR("Creating APK..."), 0)) {
		return ERR_SKIP;
	}

	unzFile pkg = unzOpen2(src_apk.utf8().get_data(), &io);
	if (!pkg) {
		EditorNode::add_io_error(vformat(TTR("Could not find template APK to export:\n%s"), src_apk));
		return ERR_FILE_NOT_FOUND;
	}

	int ret = unzGoToFirstFile(pkg);

	zlib_filefunc_def io2 = io;
	FileAccess *dst_f = nullptr;
	io2.opaque = &dst_f;

	String tmp_unaligned_path = EditorPaths::get_singleton()->get_cache_dir().plus_file("tmpexport-unaligned." + uitos(OS::get_singleton()->get_unix_time()) + ".apk");

#define CLEANUP_AND_RETURN(m_err)                            \
	{                                                        \
		DirAccess::remove_file_or_error(tmp_unaligned_path); \
		return m_err;                                        \
	}

	zipFile unaligned_apk = zipOpen2(tmp_unaligned_path.utf8().get_data(), APPEND_STATUS_CREATE, nullptr, &io2);

	String cmdline = p_preset->get("command_line/extra_args");

	String version_name = p_preset->get("version/name");
	String package_name = p_preset->get("package/unique_name");

	String apk_expansion_pkey = p_preset->get("apk_expansion/public_key");

	Vector<String> invalid_abis(enabled_abis);
	while (ret == UNZ_OK) {
		//get filename
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, nullptr, 0, nullptr, 0);

		bool skip = false;

		String file = fname;

		Vector<uint8_t> data;
		data.resize(info.uncompressed_size);

		//read
		unzOpenCurrentFile(pkg);
		unzReadCurrentFile(pkg, data.ptrw(), data.size());
		unzCloseCurrentFile(pkg);

		//write
		if (file == "AndroidManifest.xml") {
			_fix_manifest(p_preset, data, p_give_internet);
		}
		if (file == "resources.arsc") {
			_fix_resources(p_preset, data);
		}

		// Process the splash image
		if ((file == SPLASH_IMAGE_EXPORT_PATH || file == LEGACY_BUILD_SPLASH_IMAGE_EXPORT_PATH) && splash_image.is_valid() && !splash_image->is_empty()) {
			_load_image_data(splash_image, data);
		}

		// Process the splash bg color image
		if ((file == SPLASH_BG_COLOR_PATH || file == LEGACY_BUILD_SPLASH_BG_COLOR_PATH) && splash_bg_color_image.is_valid() && !splash_bg_color_image->is_empty()) {
			_load_image_data(splash_bg_color_image, data);
		}

		for (int i = 0; i < icon_densities_count; ++i) {
			if (main_image.is_valid() && !main_image->is_empty()) {
				if (file == launcher_icons[i].export_path) {
					_process_launcher_icons(file, main_image, launcher_icons[i].dimensions, data);
				}
			}
			if (foreground.is_valid() && !foreground->is_empty()) {
				if (file == launcher_adaptive_icon_foregrounds[i].export_path) {
					_process_launcher_icons(file, foreground, launcher_adaptive_icon_foregrounds[i].dimensions, data);
				}
			}
			if (background.is_valid() && !background->is_empty()) {
				if (file == launcher_adaptive_icon_backgrounds[i].export_path) {
					_process_launcher_icons(file, background, launcher_adaptive_icon_backgrounds[i].dimensions, data);
				}
			}
		}

		if (file.ends_with(".so")) {
			bool enabled = false;
			for (int i = 0; i < enabled_abis.size(); ++i) {
				if (file.begins_with("lib/" + enabled_abis[i] + "/")) {
					invalid_abis.erase(enabled_abis[i]);
					enabled = true;
					break;
				}
			}
			if (!enabled) {
				skip = true;
			}
		}

		if (file.begins_with("META-INF") && should_sign) {
			skip = true;
		}

		if (!skip) {
			print_line("ADDING: " + file);

			// Respect decision on compression made by AAPT for the export template
			const bool uncompressed = info.compression_method == 0;

			zip_fileinfo zipfi = get_zip_fileinfo();

			zipOpenNewFileInZip(unaligned_apk,
					file.utf8().get_data(),
					&zipfi,
					nullptr,
					0,
					nullptr,
					0,
					nullptr,
					uncompressed ? 0 : Z_DEFLATED,
					Z_DEFAULT_COMPRESSION);

			zipWriteInFileInZip(unaligned_apk, data.ptr(), data.size());
			zipCloseFileInZip(unaligned_apk);
		}

		ret = unzGoToNextFile(pkg);
	}

	if (!invalid_abis.is_empty()) {
		String unsupported_arch = String(", ").join(invalid_abis);
		EditorNode::add_io_error(vformat(TTR("Missing libraries in the export template for the selected architectures: %s.\nPlease build a template with all required libraries, or uncheck the missing architectures in the export preset."), unsupported_arch));
		CLEANUP_AND_RETURN(ERR_FILE_NOT_FOUND);
	}

	if (ep.step(TTR("Adding files..."), 1)) {
		CLEANUP_AND_RETURN(ERR_SKIP);
	}
	err = OK;

	if (p_flags & DEBUG_FLAG_DUMB_CLIENT) {
		APKExportData ed;
		ed.ep = &ep;
		ed.apk = unaligned_apk;
		err = export_project_files(p_preset, ignore_apk_file, &ed, save_apk_so);
	} else {
		if (apk_expansion) {
			err = save_apk_expansion_file(p_preset, p_path);
			if (err != OK) {
				EditorNode::add_io_error(TTR("Could not write expansion package file!"));
				return err;
			}
		} else {
			APKExportData ed;
			ed.ep = &ep;
			ed.apk = unaligned_apk;
			err = export_project_files(p_preset, save_apk_file, &ed, save_apk_so);
		}
	}

	if (err != OK) {
		unzClose(pkg);
		EditorNode::add_io_error(TTR("Could not export project files"));
		CLEANUP_AND_RETURN(ERR_SKIP);
	}

	zip_fileinfo zipfi = get_zip_fileinfo();
	zipOpenNewFileInZip(unaligned_apk,
			"assets/_cl_",
			&zipfi,
			nullptr,
			0,
			nullptr,
			0,
			nullptr,
			0, // No compress (little size gain and potentially slower startup)
			Z_DEFAULT_COMPRESSION);
	zipWriteInFileInZip(unaligned_apk, command_line_flags.ptr(), command_line_flags.size());
	zipCloseFileInZip(unaligned_apk);
	zipClose(unaligned_apk, nullptr);
	unzClose(pkg);

	if (err != OK) {
		CLEANUP_AND_RETURN(err);
	}

	// Let's zip-align (must be done before signing)

	static const int ZIP_ALIGNMENT = 4;

	// If we're not signing the apk, then the next step should be the last.
	const int next_step = should_sign ? 103 : 105;
	if (ep.step(TTR("Aligning APK..."), next_step)) {
		CLEANUP_AND_RETURN(ERR_SKIP);
	}

	unzFile tmp_unaligned = unzOpen2(tmp_unaligned_path.utf8().get_data(), &io);
	if (!tmp_unaligned) {
		EditorNode::add_io_error(TTR("Could not unzip temporary unaligned APK."));
		CLEANUP_AND_RETURN(ERR_FILE_NOT_FOUND);
	}

	ret = unzGoToFirstFile(tmp_unaligned);

	io2 = io;
	dst_f = nullptr;
	io2.opaque = &dst_f;
	zipFile final_apk = zipOpen2(p_path.utf8().get_data(), APPEND_STATUS_CREATE, nullptr, &io2);

	// Take files from the unaligned APK and write them out to the aligned one
	// in raw mode, i.e. not uncompressing and recompressing, aligning them as needed,
	// following what is done in https://github.com/android/platform_build/blob/master/tools/zipalign/ZipAlign.cpp
	int bias = 0;
	while (ret == UNZ_OK) {
		unz_file_info info;
		memset(&info, 0, sizeof(info));

		char fname[16384];
		char extra[16384];
		ret = unzGetCurrentFileInfo(tmp_unaligned, &info, fname, 16384, extra, 16384 - ZIP_ALIGNMENT, nullptr, 0);

		String file = fname;

		Vector<uint8_t> data;
		data.resize(info.compressed_size);

		// read
		int method, level;
		unzOpenCurrentFile2(tmp_unaligned, &method, &level, 1); // raw read
		long file_offset = unzGetCurrentFileZStreamPos64(tmp_unaligned);
		unzReadCurrentFile(tmp_unaligned, data.ptrw(), data.size());
		unzCloseCurrentFile(tmp_unaligned);

		// align
		int padding = 0;
		if (!info.compression_method) {
			// Uncompressed file => Align
			long new_offset = file_offset + bias;
			padding = (ZIP_ALIGNMENT - (new_offset % ZIP_ALIGNMENT)) % ZIP_ALIGNMENT;
		}

		memset(extra + info.size_file_extra, 0, padding);

		zip_fileinfo fileinfo = get_zip_fileinfo();
		zipOpenNewFileInZip2(final_apk,
				file.utf8().get_data(),
				&fileinfo,
				extra,
				info.size_file_extra + padding,
				nullptr,
				0,
				nullptr,
				method,
				level,
				1); // raw write
		zipWriteInFileInZip(final_apk, data.ptr(), data.size());
		zipCloseFileInZipRaw(final_apk, info.uncompressed_size, info.crc);

		bias += padding;

		ret = unzGoToNextFile(tmp_unaligned);
	}

	zipClose(final_apk, nullptr);
	unzClose(tmp_unaligned);

	if (should_sign) {
		// Signing must be done last as any additional modifications to the
		// file will invalidate the signature.
		err = sign_apk(p_preset, p_debug, p_path, ep);
		if (err != OK) {
			CLEANUP_AND_RETURN(err);
		}
	}

	CLEANUP_AND_RETURN(OK);
}

void EditorExportPlatformAndroid::get_platform_features(List<String> *r_features) {
	r_features->push_back("mobile");
	r_features->push_back("android");
}

void EditorExportPlatformAndroid::resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features) {
}

EditorExportPlatformAndroid::EditorExportPlatformAndroid() {
	Ref<Image> img = memnew(Image(_android_logo));
	logo.instantiate();
	logo->create_from_image(img);

	img = Ref<Image>(memnew(Image(_android_run_icon)));
	run_icon.instantiate();
	run_icon->create_from_image(img);

	devices_changed.set();
	plugins_changed.set();
	check_for_changes_thread.start(_check_for_changes_poll_thread, this);
}

EditorExportPlatformAndroid::~EditorExportPlatformAndroid() {
	quit_request.set();
	check_for_changes_thread.wait_to_finish();
}

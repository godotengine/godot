/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "io/marshalls.h"
#include "io/zip_io.h"
#include "os/file_access.h"
#include "os/os.h"
#include "platform/android/logo.gen.h"
#include "platform/android/run_icon.gen.h"
#include "project_settings.h"
#include "version.h"

#include <string.h>

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
	NULL
};

class EditorExportAndroid : public EditorExportPlatform {

	GDCLASS(EditorExportAndroid, EditorExportPlatform)

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

	Vector<Device> devices;
	bool devices_changed;
	Mutex *device_lock;
	Thread *device_thread;
	volatile bool quit_request;

	static void _device_poll_thread(void *ud) {

		EditorExportAndroid *ea = (EditorExportAndroid *)ud;

		while (!ea->quit_request) {

			String adb = EditorSettings::get_singleton()->get("export/android/adb");
			if (FileAccess::exists(adb)) {

				String devices;
				List<String> args;
				args.push_back("devices");
				int ec;
				OS::get_singleton()->execute(adb, args, true, NULL, &devices, &ec);

				Vector<String> ds = devices.split("\n");
				Vector<String> ldevices;
				for (int i = 1; i < ds.size(); i++) {

					String d = ds[i];
					int dpos = d.find("device");
					if (dpos == -1)
						continue;
					d = d.substr(0, dpos).strip_edges();
					//print_line("found devuce: "+d);
					ldevices.push_back(d);
				}

				ea->device_lock->lock();

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

					print_line("DIFFERENT!");
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
							args.push_back("cat");
							args.push_back("/system/build.prop");
							int ec;
							String dp;

							OS::get_singleton()->execute(adb, args, true, NULL, &dp, &ec);

							Vector<String> props = dp.split("\n");
							String vendor;
							String device;
							d.description + "Device ID: " + d.id + "\n";
							d.api_level = 0;
							for (int j = 0; j < props.size(); j++) {

								String p = props[j];
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
							//print_line("name: "+d.name);
							//print_line("description: "+d.description);
						}

						ndevices.push_back(d);
					}

					ea->devices = ndevices;
					ea->devices_changed = true;
				}

				ea->device_lock->unlock();
			}

			uint64_t wait = 3000000;
			uint64_t time = OS::get_singleton()->get_ticks_usec();
			while (OS::get_singleton()->get_ticks_usec() - time < wait) {
				OS::get_singleton()->delay_usec(1000);
				if (ea->quit_request)
					break;
			}
		}

		if (EditorSettings::get_singleton()->get("export/android/shutdown_adb_on_exit")) {
			String adb = EditorSettings::get_singleton()->get("export/android/adb");
			if (!FileAccess::exists(adb)) {
				return; //adb not configured
			}

			List<String> args;
			args.push_back("kill-server");
			OS::get_singleton()->execute(adb, args, true);
		};
	}

	String get_project_name(const String &p_name) const {

		String aname;
		if (p_name != "") {
			aname = p_name;
		} else {
			aname = ProjectSettings::get_singleton()->get("application/config/name");
		}

		if (aname == "") {
			aname = _MKSTR(VERSION_NAME);
		}

		return aname;
	}

	String get_package_name(const String &p_package) {

		String pname = p_package;
		String basename = ProjectSettings::get_singleton()->get("application/config/name");
		basename = basename.to_lower();

		String name;
		bool first = true;
		for (int i = 0; i < basename.length(); i++) {
			CharType c = basename[i];
			if (c >= '0' && c <= '9' && first) {
				continue;
			}
			if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
				name += String::chr(c);
				first = false;
			}
		}
		if (name == "")
			name = "noname";

		pname = pname.replace("$genname", name);
		return pname;
	}

	static bool _should_compress_asset(const String &p_path, const Vector<uint8_t> &p_data) {

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
			NULL
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

	static zip_fileinfo get_zip_fileinfo() {

		OS::Time time = OS::get_singleton()->get_time();
		OS::Date date = OS::get_singleton()->get_date();

		zip_fileinfo zipfi;
		zipfi.tmz_date.tm_hour = time.hour;
		zipfi.tmz_date.tm_mday = date.day;
		zipfi.tmz_date.tm_min = time.min;
		zipfi.tmz_date.tm_mon = date.month;
		zipfi.tmz_date.tm_sec = time.sec;
		zipfi.tmz_date.tm_year = date.year;
		zipfi.dosDate = 0;
		zipfi.external_fa = 0;
		zipfi.internal_fa = 0;

		return zipfi;
	}

	static Set<String> get_abis() {
		Set<String> abis;
		abis.insert("armeabi");
		abis.insert("armeabi-v7a");
		abis.insert("arm64-v8a");
		abis.insert("x86");
		abis.insert("x86_64");
		abis.insert("mips");
		abis.insert("mips64");
		return abis;
	}

	static Error save_apk_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total) {
		APKExportData *ed = (APKExportData *)p_userdata;
		String dst_path = p_path;
		static Set<String> android_abis = get_abis();

		if (dst_path.ends_with(".so")) {
			String abi = dst_path.get_base_dir().get_file().strip_edges(); // parent dir name
			if (android_abis.has(abi)) {
				dst_path = "lib/" + abi + "/" + dst_path.get_file();
			} else {
				String err = "Dynamic libraries must be located in the folder named after Android ABI they were compiled for. " +
							 p_path + " does not follow this convention.";
				ERR_PRINT(err.utf8().get_data());
				return ERR_FILE_BAD_PATH;
			}
		} else {
			dst_path = dst_path.replace_first("res://", "assets/");
		}

		zip_fileinfo zipfi = get_zip_fileinfo();

		zipOpenNewFileInZip(ed->apk,
				dst_path.utf8().get_data(),
				&zipfi,
				NULL,
				0,
				NULL,
				0,
				NULL,
				_should_compress_asset(p_path, p_data) ? Z_DEFLATED : 0,
				Z_DEFAULT_COMPRESSION);

		zipWriteInFileInZip(ed->apk, p_data.ptr(), p_data.size());
		zipCloseFileInZip(ed->apk);
		ed->ep->step("File: " + p_path, 3 + p_file * 100 / p_total);
		return OK;
	}

	void _fix_manifest(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &p_manifest, bool p_give_internet) {

		// Leaving the unused types commented because looking these constants up
		// again later would be annoying
		// const int CHUNK_AXML_FILE = 0x00080003;
		// const int CHUNK_RESOURCEIDS = 0x00080180;
		const int CHUNK_STRINGS = 0x001C0001;
		// const int CHUNK_XML_END_NAMESPACE = 0x00100101;
		// const int CHUNK_XML_END_TAG = 0x00100103;
		// const int CHUNK_XML_START_NAMESPACE = 0x00100100;
		const int CHUNK_XML_START_TAG = 0x00100102;
		// const int CHUNK_XML_TEXT = 0x00100104;
		const int UTF8_FLAG = 0x00000100;

		Vector<String> string_table;

		uint32_t ofs = 8;

		uint32_t string_count = 0;
		uint32_t styles_count = 0;
		uint32_t string_flags = 0;
		uint32_t string_data_offset = 0;

		uint32_t styles_offset = 0;
		uint32_t string_table_begins = 0;
		uint32_t string_table_ends = 0;
		Vector<uint8_t> stable_extra;

		String version_name = p_preset->get("version/name");
		int version_code = p_preset->get("version/code");
		String package_name = p_preset->get("package/unique_name");

		int orientation = p_preset->get("screen/orientation");

		bool screen_support_small = p_preset->get("screen/support_small");
		bool screen_support_normal = p_preset->get("screen/support_normal");
		bool screen_support_large = p_preset->get("screen/support_large");
		bool screen_support_xlarge = p_preset->get("screen/support_xlarge");

		String user_perms[MAX_USER_PERMISSIONS];

		for (int i = 0; i < MAX_USER_PERMISSIONS; i++) {

			user_perms[i] = p_preset->get("user_permissions/" + itos(i));
		}

		Set<String> perms;

		const char **aperms = android_perms;
		while (*aperms) {

			bool enabled = p_preset->get("permissions/" + String(*aperms).to_lower());
			if (enabled)
				perms.insert(String(*aperms));
			aperms++;
		}

		while (ofs < (uint32_t)p_manifest.size()) {

			uint32_t chunk = decode_uint32(&p_manifest[ofs]);
			uint32_t size = decode_uint32(&p_manifest[ofs + 4]);

			switch (chunk) {

				case CHUNK_STRINGS: {

					int iofs = ofs + 8;

					string_count = decode_uint32(&p_manifest[iofs]);
					styles_count = decode_uint32(&p_manifest[iofs + 4]);
					string_flags = decode_uint32(&p_manifest[iofs + 8]);
					string_data_offset = decode_uint32(&p_manifest[iofs + 12]);
					styles_offset = decode_uint32(&p_manifest[iofs + 16]);
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

						ERR_EXPLAIN("Unimplemented, can't read utf8 string table.");
						ERR_FAIL_COND(string_flags & UTF8_FLAG);

						if (string_flags & UTF8_FLAG) {

						} else {
							uint32_t len = decode_uint16(&p_manifest[string_at]);
							Vector<CharType> ucstring;
							ucstring.resize(len + 1);
							for (uint32_t j = 0; j < len; j++) {
								uint16_t c = decode_uint16(&p_manifest[string_at + 2 + 2 * j]);
								ucstring[j] = c;
							}
							string_end = MAX(string_at + 2 + 2 * len, string_end);
							ucstring[len] = 0;
							string_table[i] = ucstring.ptr();
						}

						//print_line("String "+itos(i)+": "+string_table[i]);
					}

					for (uint32_t i = string_end; i < (ofs + size); i++) {
						stable_extra.push_back(p_manifest[i]);
					}

					//printf("stable extra: %i\n",int(stable_extra.size()));
					string_table_ends = ofs + size;

					//print_line("STABLE SIZE: "+itos(size)+" ACTUAL: "+itos(string_table_ends));

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

						String value;
						if (attr_value != 0xFFFFFFFF)
							value = string_table[attr_value];
						else
							value = "Res #" + itos(attr_resid);
						String attrname = string_table[attr_name];
						String nspace;
						if (attr_nspace != 0xFFFFFFFF)
							nspace = string_table[attr_nspace];
						else
							nspace = "";

						//replace project information
						if (tname == "manifest" && attrname == "package") {

							print_line("FOUND package");
							string_table[attr_value] = get_package_name(package_name);
						}

						if (tname == "manifest" && /*nspace=="android" &&*/ attrname == "versionCode") {

							print_line("FOUND versionCode");
							encode_uint32(version_code, &p_manifest[iofs + 16]);
						}

						if (tname == "manifest" && /*nspace=="android" &&*/ attrname == "versionName") {

							print_line("FOUND versionName");
							if (attr_value == 0xFFFFFFFF) {
								WARN_PRINT("Version name in a resource, should be plaintext")
							} else
								string_table[attr_value] = version_name;
						}

						if (tname == "activity" && /*nspace=="android" &&*/ attrname == "screenOrientation") {

							encode_uint32(orientation == 0 ? 0 : 1, &p_manifest[iofs + 16]);
						}

						if (tname == "uses-feature" && /*nspace=="android" &&*/ attrname == "glEsVersion") {
							print_line("version number: " + itos(decode_uint32(&p_manifest[iofs + 16])));
						}

						if (tname == "uses-permission" && /*nspace=="android" &&*/ attrname == "name") {

							if (value.begins_with("godot.custom")) {

								int which = value.get_slice(".", 2).to_int();
								if (which >= 0 && which < MAX_USER_PERMISSIONS && user_perms[which].strip_edges() != "") {

									string_table[attr_value] = user_perms[which].strip_edges();
								}

							} else if (value.begins_with("godot.")) {
								String perm = value.get_slice(".", 1);

								if (perms.has(perm) || (p_give_internet && perm == "INTERNET")) {

									print_line("PERM: " + perm);
									string_table[attr_value] = "android.permission." + perm;
								}
							}
						}

						if (tname == "supports-screens") {

							if (attrname == "smallScreens") {

								encode_uint32(screen_support_small ? 0xFFFFFFFF : 0, &p_manifest[iofs + 16]);

							} else if (attrname == "normalScreens") {

								encode_uint32(screen_support_normal ? 0xFFFFFFFF : 0, &p_manifest[iofs + 16]);

							} else if (attrname == "largeScreens") {

								encode_uint32(screen_support_large ? 0xFFFFFFFF : 0, &p_manifest[iofs + 16]);

							} else if (attrname == "xlargeScreens") {

								encode_uint32(screen_support_xlarge ? 0xFFFFFFFF : 0, &p_manifest[iofs + 16]);
							}
						}

						iofs += 20;
					}

				} break;
			}

			ofs += size;
		}

		//create new andriodmanifest binary

		Vector<uint8_t> ret;
		ret.resize(string_table_begins + string_table.size() * 4);

		for (uint32_t i = 0; i < string_table_begins; i++) {

			ret[i] = p_manifest[i];
		}

		ofs = 0;
		for (int i = 0; i < string_table.size(); i++) {

			encode_uint32(ofs, &ret[string_table_begins + i * 4]);
			ofs += string_table[i].length() * 2 + 2 + 2;
			//print_line("ofs: "+itos(i)+": "+itos(ofs));
		}
		ret.resize(ret.size() + ofs);
		uint8_t *chars = &ret[ret.size() - ofs];
		for (int i = 0; i < string_table.size(); i++) {

			String s = string_table[i];
			//print_line("savint string :"+s);
			encode_uint16(s.length(), chars);
			chars += 2;
			for (int j = 0; j < s.length(); j++) { //include zero?
				encode_uint16(s[j], chars);
				chars += 2;
			}
			encode_uint16(0, chars);
			chars += 2;
		}

		for (int i = 0; i < stable_extra.size(); i++) {
			ret.push_back(stable_extra[i]);
		}

		while (ret.size() % 4)
			ret.push_back(0);

		uint32_t new_stable_end = ret.size();

		uint32_t extra = (p_manifest.size() - string_table_ends);
		ret.resize(new_stable_end + extra);
		for (uint32_t i = 0; i < extra; i++)
			ret[new_stable_end + i] = p_manifest[string_table_ends + i];

		while (ret.size() % 4)
			ret.push_back(0);
		encode_uint32(ret.size(), &ret[4]); //update new file size

		encode_uint32(new_stable_end - 8, &ret[12]); //update new string table size

		//print_line("file size: "+itos(ret.size()));

		p_manifest = ret;
	}

	static String _parse_string(const uint8_t *p_bytes, bool p_utf8) {

		uint32_t offset = 0;
		uint32_t len = decode_uint16(&p_bytes[offset]);

		if (p_utf8) {
			//don't know how to read extended utf8, this will have to be for now
			len >>= 8;
		}
		offset += 2;
		//printf("len %i, unicode: %i\n",len,int(p_utf8));

		if (p_utf8) {

			Vector<uint8_t> str8;
			str8.resize(len + 1);
			for (uint32_t i = 0; i < len; i++) {
				str8[i] = p_bytes[offset + i];
			}
			str8[len] = 0;
			String str;
			str.parse_utf8((const char *)str8.ptr());
			return str;
		} else {

			String str;
			for (uint32_t i = 0; i < len; i++) {
				CharType c = decode_uint16(&p_bytes[offset + i * 2]);
				if (c == 0)
					break;
				str += String::chr(c);
			}
			return str;
		}
	}
	void _fix_resources(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &p_manifest) {

		const int UTF8_FLAG = 0x00000100;
		print_line("*******************GORRRGLE***********************");

		uint32_t string_block_len = decode_uint32(&p_manifest[16]);
		uint32_t string_count = decode_uint32(&p_manifest[20]);
		uint32_t string_flags = decode_uint32(&p_manifest[28]);
		const uint32_t string_table_begins = 40;

		Vector<String> string_table;

		String package_name = p_preset->get("package/name");

		for (uint32_t i = 0; i < string_count; i++) {

			uint32_t offset = decode_uint32(&p_manifest[string_table_begins + i * 4]);
			offset += string_table_begins + string_count * 4;

			String str = _parse_string(&p_manifest[offset], string_flags & UTF8_FLAG);

			if (str.begins_with("godot-project-name")) {

				if (str == "godot-project-name") {
					//project name
					str = get_project_name(package_name);

				} else {

					String lang = str.substr(str.find_last("-") + 1, str.length()).replace("-", "_");
					String prop = "application/config/name_" + lang;
					if (ProjectSettings::get_singleton()->has(prop)) {
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

			ret[i] = p_manifest[i];
		}

		int ofs = 0;
		for (int i = 0; i < string_table.size(); i++) {

			encode_uint32(ofs, &ret[string_table_begins + i * 4]);
			ofs += string_table[i].length() * 2 + 2 + 2;
		}

		ret.resize(ret.size() + ofs);
		uint8_t *chars = &ret[ret.size() - ofs];
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
		while (ret.size() % 4)
			ret.push_back(0);

		//change flags to not use utf8
		encode_uint32(string_flags & ~0x100, &ret[28]);
		//change length
		encode_uint32(ret.size() - 12, &ret[16]);
		//append the rest...
		int rest_from = 12 + string_block_len;
		int rest_to = ret.size();
		int rest_len = (p_manifest.size() - rest_from);
		ret.resize(ret.size() + (p_manifest.size() - rest_from));
		for (int i = 0; i < rest_len; i++) {
			ret[rest_to + i] = p_manifest[rest_from + i];
		}
		//finally update the size
		encode_uint32(ret.size(), &ret[4]);

		p_manifest = ret;
		//printf("end\n");
	}

public:
	enum {
		MAX_USER_PERMISSIONS = 20
	};

	typedef Error (*EditorExportSaveFunction)(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);

public:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) {

		int api = p_preset->get("graphics/api");
		if (api == 0)
			r_features->push_back("etc");
		else
			r_features->push_back("etc2");
	}

	virtual void get_export_options(List<ExportOption> *r_options) {

		r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "graphics/api", PROPERTY_HINT_ENUM, "OpenGL ES 2.0,OpenGL ES 3.0"), 1));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "graphics/32_bits_framebuffer"), true));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "one_click_deploy/clear_previous_install"), true));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_package/debug", PROPERTY_HINT_GLOBAL_FILE, "apk"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_package/release", PROPERTY_HINT_GLOBAL_FILE, "apk"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "command_line/extra_args"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "version/code", PROPERTY_HINT_RANGE, "1,65535,1"), 1));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "version/name"), "1.0"));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/unique_name"), "org.godotengine.$genname"));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/name"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/icon", PROPERTY_HINT_FILE, "png"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "package/signed"), true));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "architecture/arm"), true));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "architecture/x86"), false));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "screen/immersive_mode"), true));
		r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "screen/orientation", PROPERTY_HINT_ENUM, "Landscape,Portrait"), 0));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "screen/support_small"), true));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "screen/support_normal"), true));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "screen/support_large"), true));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "screen/support_xlarge"), true));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "keystore/release", PROPERTY_HINT_GLOBAL_FILE, "keystore"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "keystore/release_user"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "keystore/release_password"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "apk_expansion/enable"), false));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "apk_expansion/SALT"), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "apk_expansion/public_key", PROPERTY_HINT_MULTILINE_TEXT), ""));

		const char **perms = android_perms;
		while (*perms) {

			r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "permissions/" + String(*perms).to_lower()), false));
			perms++;
		}

		for (int i = 0; i < MAX_USER_PERMISSIONS; i++) {

			r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "user_permissions/" + itos(i)), false));
		}

		//r_options->push_back( PropertyInfo( Variant::INT, "resources/pack_mode", PROPERTY_HINT_ENUM,"Copy,Single Exec.,Pack (.pck),Bundles (Optical)"));
	}

	virtual String get_name() const {
		return "Android";
	}

	virtual String get_os_name() const {
		return "Android";
	}

	virtual Ref<Texture> get_logo() const {
		return logo;
	}

	virtual bool poll_devices() {

		bool dc = devices_changed;
		devices_changed = false;
		return dc;
	}

	virtual int get_device_count() const {

		device_lock->lock();
		int dc = devices.size();
		device_lock->unlock();

		return dc;
	}

	virtual String get_device_name(int p_device) const {

		ERR_FAIL_INDEX_V(p_device, devices.size(), "");
		device_lock->lock();
		String s = devices[p_device].name;
		device_lock->unlock();
		return s;
	}

	virtual String get_device_info(int p_device) const {

		ERR_FAIL_INDEX_V(p_device, devices.size(), "");
		device_lock->lock();
		String s = devices[p_device].description;
		device_lock->unlock();
		return s;
	}

	virtual Error run(const Ref<EditorExportPreset> &p_preset, int p_device, int p_debug_flags) {

		ERR_FAIL_INDEX_V(p_device, devices.size(), ERR_INVALID_PARAMETER);
		device_lock->lock();

		EditorProgress ep("run", "Running on " + devices[p_device].name, 3);

		String adb = EditorSettings::get_singleton()->get("export/android/adb");
		if (adb == "") {

			EditorNode::add_io_error("ADB executable not configured in settings, can't run.");
			device_lock->unlock();
			return ERR_UNCONFIGURED;
		}

		//export_temp
		ep.step("Exporting APK", 0);

		const bool use_remote = (p_debug_flags & DEBUG_FLAG_REMOTE_DEBUG) || (p_debug_flags & DEBUG_FLAG_DUMB_CLIENT);
		const bool use_reverse = devices[p_device].api_level >= 21;

		if (use_reverse)
			p_debug_flags |= DEBUG_FLAG_REMOTE_DEBUG_LOCALHOST;

		String export_to = EditorSettings::get_singleton()->get_settings_path() + "/tmp/tmpexport.apk";
		Error err = export_project(p_preset, true, export_to, p_debug_flags);
		if (err) {
			device_lock->unlock();
			return err;
		}

		List<String> args;
		int rv;

		bool remove_prev = p_preset->get("one_click_deploy/clear_previous_install");
		String version_name = p_preset->get("version/name");
		String package_name = p_preset->get("package/unique_name");

		if (remove_prev) {
			ep.step("Uninstalling..", 1);

			print_line("Uninstalling previous version: " + devices[p_device].name);

			args.push_back("-s");
			args.push_back(devices[p_device].id);
			args.push_back("uninstall");
			args.push_back(get_package_name(package_name));

			err = OS::get_singleton()->execute(adb, args, true, NULL, NULL, &rv);
		}

		print_line("Installing into device (please wait..): " + devices[p_device].name);
		ep.step("Installing to Device (please wait..)..", 2);

		args.clear();
		args.push_back("-s");
		args.push_back(devices[p_device].id);
		args.push_back("install");
		args.push_back("-r");
		args.push_back(export_to);

		err = OS::get_singleton()->execute(adb, args, true, NULL, NULL, &rv);
		if (err || rv != 0) {
			EditorNode::add_io_error("Could not install to device.");
			device_lock->unlock();
			return ERR_CANT_CREATE;
		}

		if (use_remote) {
			if (use_reverse) {

				static const char *const msg = "** Device API >= 21; debugging over USB **";
				EditorNode::get_singleton()->get_log()->add_message(msg);
				print_line(String(msg).to_upper());

				args.clear();
				args.push_back("-s");
				args.push_back(devices[p_device].id);
				args.push_back("reverse");
				args.push_back("--remove-all");
				OS::get_singleton()->execute(adb, args, true, NULL, NULL, &rv);

				if (p_debug_flags & DEBUG_FLAG_REMOTE_DEBUG) {

					int dbg_port = EditorSettings::get_singleton()->get("network/debug/remote_port");
					args.clear();
					args.push_back("-s");
					args.push_back(devices[p_device].id);
					args.push_back("reverse");
					args.push_back("tcp:" + itos(dbg_port));
					args.push_back("tcp:" + itos(dbg_port));

					OS::get_singleton()->execute(adb, args, true, NULL, NULL, &rv);
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

					err = OS::get_singleton()->execute(adb, args, true, NULL, NULL, &rv);
					print_line("Reverse result2: " + itos(rv));
				}
			} else {

				static const char *const msg = "** Device API < 21; debugging over Wi-Fi **";
				EditorNode::get_singleton()->get_log()->add_message(msg);
				print_line(String(msg).to_upper());
			}
		}

		ep.step("Running on Device..", 3);
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
		args.push_back(get_package_name(package_name) + "/org.godotengine.godot.Godot");

		err = OS::get_singleton()->execute(adb, args, true, NULL, NULL, &rv);
		if (err || rv != 0) {
			EditorNode::add_io_error("Could not execute on device.");
			device_lock->unlock();
			return ERR_CANT_CREATE;
		}
		device_lock->unlock();
		return OK;
	}

	virtual Ref<Texture> get_run_icon() const {
		return run_icon;
	}

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {

		r_missing_templates = find_export_template("android_debug.apk") == String() || find_export_template("android_release.apk") == String();

		bool valid = !r_missing_templates;

		String adb = EditorSettings::get_singleton()->get("export/android/adb");
		String err;

		if (!FileAccess::exists(adb)) {

			valid = false;
			err += "ADB executable not configured in editor settings.\n";
		}

		String js = EditorSettings::get_singleton()->get("export/android/jarsigner");

		if (!FileAccess::exists(js)) {

			valid = false;
			err += "OpenJDK 6 jarsigner not configured in editor settings.\n";
		}

		String dk = EditorSettings::get_singleton()->get("export/android/debug_keystore");

		if (!FileAccess::exists(dk)) {

			valid = false;
			err += "Debug Keystore not configured in editor settings.\n";
		}

		bool apk_expansion = p_preset->get("apk_expansion/enable");

		if (apk_expansion) {

			/*
			 if (apk_expansion_salt=="") {
				 valid=false;
				 err+="Invalid SALT for apk expansion.\n";
			 }
			 */

			String apk_expansion_pkey = p_preset->get("apk_expansion/public_key");

			if (apk_expansion_pkey == "") {
				valid = false;

				err += "Invalid public key for apk expansion.\n";
			}
		}

		r_error = err;
		return valid;
	}

	virtual String get_binary_extension() const {
		return "apk";
	}

	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0) {

		String src_apk;

		EditorProgress ep("export", "Exporting for Android", 105);

		if (p_debug)
			src_apk = p_preset->get("custom_package/debug");
		else
			src_apk = p_preset->get("custom_package/release");

		src_apk = src_apk.strip_edges();
		if (src_apk == "") {
			if (p_debug) {
				src_apk = find_export_template("android_debug.apk");
			} else {
				src_apk = find_export_template("android_release.apk");
			}
			if (src_apk == "") {
				EditorNode::add_io_error("Package not found: " + src_apk);
				return ERR_FILE_NOT_FOUND;
			}
		}

		FileAccess *src_f = NULL;
		zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

		ep.step("Creating APK", 0);

		unzFile pkg = unzOpen2(src_apk.utf8().get_data(), &io);
		if (!pkg) {

			EditorNode::add_io_error("Could not find template APK to export:\n" + src_apk);
			return ERR_FILE_NOT_FOUND;
		}

		ERR_FAIL_COND_V(!pkg, ERR_CANT_OPEN);
		int ret = unzGoToFirstFile(pkg);

		zlib_filefunc_def io2 = io;
		FileAccess *dst_f = NULL;
		io2.opaque = &dst_f;
		String unaligned_path = EditorSettings::get_singleton()->get_settings_path() + "/tmp/tmpexport-unaligned.apk";
		zipFile unaligned_apk = zipOpen2(unaligned_path.utf8().get_data(), APPEND_STATUS_CREATE, NULL, &io2);

		bool export_x86 = p_preset->get("architecture/x86");
		bool export_arm = p_preset->get("architecture/arm");
		bool export_arm64 = p_preset->get("architecture/arm64");

		bool use_32_fb = p_preset->get("graphics/32_bits_framebuffer");
		bool immersive = p_preset->get("screen/immersive_mode");

		bool _signed = p_preset->get("package/signed");

		bool apk_expansion = p_preset->get("apk_expansion/enable");

		String cmdline = p_preset->get("command_line/extra_args");

		int version_code = p_preset->get("version/code");
		String version_name = p_preset->get("version/name");
		String package_name = p_preset->get("package/unique_name");

		String apk_expansion_pkey = p_preset->get("apk_expansion/public_key");

		String release_keystore = p_preset->get("keystore/release");
		String release_username = p_preset->get("keystore/release_user");
		String release_password = p_preset->get("keystore/release_password");

		while (ret == UNZ_OK) {

			//get filename
			unz_file_info info;
			char fname[16384];
			ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, NULL, 0, NULL, 0);

			bool skip = false;

			String file = fname;

			Vector<uint8_t> data;
			data.resize(info.uncompressed_size);

			//read
			unzOpenCurrentFile(pkg);
			unzReadCurrentFile(pkg, data.ptr(), data.size());
			unzCloseCurrentFile(pkg);

			//write

			if (file == "AndroidManifest.xml") {

				_fix_manifest(p_preset, data, p_flags & (DEBUG_FLAG_DUMB_CLIENT | DEBUG_FLAG_REMOTE_DEBUG));
			}

			if (file == "resources.arsc") {

				_fix_resources(p_preset, data);
			}

			if (file == "res/drawable/icon.png") {

				String icon = p_preset->get("package/icon");
				icon = icon.strip_edges();
				bool found = false;

				if (icon != "" && icon.ends_with(".png")) {

					FileAccess *f = FileAccess::open(icon, FileAccess::READ);
					if (f) {

						data.resize(f->get_len());
						f->get_buffer(data.ptr(), data.size());
						memdelete(f);
						found = true;
					}
				}

				if (!found) {

					String appicon = ProjectSettings::get_singleton()->get("application/config/icon");
					if (appicon != "" && appicon.ends_with(".png")) {
						FileAccess *f = FileAccess::open(appicon, FileAccess::READ);
						if (f) {
							data.resize(f->get_len());
							f->get_buffer(data.ptr(), data.size());
							memdelete(f);
						}
					}
				}
			}

			if (file == "lib/x86/*.so" && !export_x86) {
				skip = true;
			}

			if (file.match("lib/armeabi*/*.so") && !export_arm) {
				skip = true;
			}

			if (file.match("lib/arm64*/*.so") && !export_arm64) {
				skip = true;
			}

			if (file.begins_with("META-INF") && _signed) {
				skip = true;
			}

			print_line("ADDING: " + file);

			if (!skip) {

				// Respect decision on compression made by AAPT for the export template
				const bool uncompressed = info.compression_method == 0;

				zip_fileinfo zipfi = get_zip_fileinfo();

				zipOpenNewFileInZip(unaligned_apk,
						file.utf8().get_data(),
						&zipfi,
						NULL,
						0,
						NULL,
						0,
						NULL,
						uncompressed ? 0 : Z_DEFLATED,
						Z_DEFAULT_COMPRESSION);

				zipWriteInFileInZip(unaligned_apk, data.ptr(), data.size());
				zipCloseFileInZip(unaligned_apk);
			}

			ret = unzGoToNextFile(pkg);
		}

		ep.step("Adding Files..", 1);
		Error err = OK;
		Vector<String> cl = cmdline.strip_edges().split(" ");
		for (int i = 0; i < cl.size(); i++) {
			if (cl[i].strip_edges().length() == 0) {
				cl.remove(i);
				i--;
			}
		}

		gen_export_flags(cl, p_flags);

		if (p_flags & DEBUG_FLAG_DUMB_CLIENT) {

			/*String host = EditorSettings::get_singleton()->get("filesystem/file_server/host");
			int port = EditorSettings::get_singleton()->get("filesystem/file_server/post");
			String passwd = EditorSettings::get_singleton()->get("filesystem/file_server/password");
			cl.push_back("--remote-fs");
			cl.push_back(host+":"+itos(port));
			if (passwd!="") {
				cl.push_back("--remote-fs-password");
				cl.push_back(passwd);
			}*/

		} else {
			//all files

			if (apk_expansion) {

				String apkfname = "main." + itos(version_code) + "." + get_package_name(package_name) + ".obb";
				String fullpath = p_path.get_base_dir().plus_file(apkfname);
				err = save_pack(p_preset, fullpath);
				if (err != OK) {
					EditorNode::add_io_error("Could not write expansion package file: " + apkfname);
					return OK;
				}

				cl.push_back("--use_apk_expansion");
				cl.push_back("--apk_expansion_md5");
				cl.push_back(FileAccess::get_md5(fullpath));
				cl.push_back("--apk_expansion_key");
				cl.push_back(apk_expansion_pkey.strip_edges());

			} else {

				APKExportData ed;
				ed.ep = &ep;
				ed.apk = unaligned_apk;

				err = export_project_files(p_preset, save_apk_file, &ed);
			}
		}

		if (use_32_fb)
			cl.push_back("--use_depth_32");

		if (immersive)
			cl.push_back("--use_immersive");

		if (cl.size()) {
			//add comandline
			Vector<uint8_t> clf;
			clf.resize(4);
			encode_uint32(cl.size(), &clf[0]);
			for (int i = 0; i < cl.size(); i++) {

				CharString txt = cl[i].utf8();
				int base = clf.size();
				clf.resize(base + 4 + txt.length());
				encode_uint32(txt.length(), &clf[base]);
				copymem(&clf[base + 4], txt.ptr(), txt.length());
				print_line(itos(i) + " param: " + cl[i]);
			}

			zip_fileinfo zipfi = get_zip_fileinfo();

			zipOpenNewFileInZip(unaligned_apk,
					"assets/_cl_",
					&zipfi,
					NULL,
					0,
					NULL,
					0,
					NULL,
					0, // No compress (little size gain and potentially slower startup)
					Z_DEFAULT_COMPRESSION);

			zipWriteInFileInZip(unaligned_apk, clf.ptr(), clf.size());
			zipCloseFileInZip(unaligned_apk);
		}

		zipClose(unaligned_apk, NULL);
		unzClose(pkg);

		if (err) {
			return err;
		}

		if (_signed) {

			String jarsigner = EditorSettings::get_singleton()->get("export/android/jarsigner");
			if (!FileAccess::exists(jarsigner)) {
				EditorNode::add_io_error("'jarsigner' could not be found.\nPlease supply a path in the editor settings.\nResulting apk is unsigned.");
				return OK;
			}

			String keystore;
			String password;
			String user;
			if (p_debug) {
				keystore = EditorSettings::get_singleton()->get("export/android/debug_keystore");
				password = EditorSettings::get_singleton()->get("export/android/debug_keystore_pass");
				user = EditorSettings::get_singleton()->get("export/android/debug_keystore_user");

				ep.step("Signing Debug APK..", 103);

			} else {
				keystore = release_keystore;
				password = release_password;
				user = release_username;

				ep.step("Signing Release APK..", 103);
			}

			if (!FileAccess::exists(keystore)) {
				EditorNode::add_io_error("Could not find keystore, unable to export.");
				return ERR_FILE_CANT_OPEN;
			}

			List<String> args;
			args.push_back("-digestalg");
			args.push_back("SHA1");
			args.push_back("-sigalg");
			args.push_back("MD5withRSA");
			String tsa_url = EditorSettings::get_singleton()->get("export/android/timestamping_authority_url");
			if (tsa_url != "") {
				args.push_back("-tsa");
				args.push_back(tsa_url);
			}
			args.push_back("-verbose");
			args.push_back("-keystore");
			args.push_back(keystore);
			args.push_back("-storepass");
			args.push_back(password);
			args.push_back(unaligned_path);
			args.push_back(user);
			int retval;
			OS::get_singleton()->execute(jarsigner, args, true, NULL, NULL, &retval);
			if (retval) {
				EditorNode::add_io_error("'jarsigner' returned with error #" + itos(retval));
				return ERR_CANT_CREATE;
			}

			ep.step("Verifying APK..", 104);

			args.clear();
			args.push_back("-verify");
			args.push_back("-keystore");
			args.push_back(keystore);
			args.push_back(unaligned_path);
			args.push_back("-verbose");

			OS::get_singleton()->execute(jarsigner, args, true, NULL, NULL, &retval);
			if (retval) {
				EditorNode::add_io_error("'jarsigner' verification of APK failed. Make sure to use jarsigner from Java 6.");
				return ERR_CANT_CREATE;
			}
		}

		// Let's zip-align (must be done after signing)

		static const int ZIP_ALIGNMENT = 4;

		ep.step("Aligning APK..", 105);

		unzFile tmp_unaligned = unzOpen2(unaligned_path.utf8().get_data(), &io);
		if (!tmp_unaligned) {

			EditorNode::add_io_error("Could not find temp unaligned APK.");
			return ERR_FILE_NOT_FOUND;
		}

		ERR_FAIL_COND_V(!tmp_unaligned, ERR_CANT_OPEN);
		ret = unzGoToFirstFile(tmp_unaligned);

		io2 = io;
		dst_f = NULL;
		io2.opaque = &dst_f;
		zipFile final_apk = zipOpen2(p_path.utf8().get_data(), APPEND_STATUS_CREATE, NULL, &io2);

		// Take files from the unaligned APK and write them out to the aligned one
		// in raw mode, i.e. not uncompressing and recompressing, aligning them as needed,
		// following what is done in https://github.com/android/platform_build/blob/master/tools/zipalign/ZipAlign.cpp
		int bias = 0;
		while (ret == UNZ_OK) {

			unz_file_info info;
			memset(&info, 0, sizeof(info));

			char fname[16384];
			char extra[16384];
			ret = unzGetCurrentFileInfo(tmp_unaligned, &info, fname, 16384, extra, 16384 - ZIP_ALIGNMENT, NULL, 0);

			String file = fname;

			Vector<uint8_t> data;
			data.resize(info.compressed_size);

			// read
			int method, level;
			unzOpenCurrentFile2(tmp_unaligned, &method, &level, 1); // raw read
			long file_offset = unzGetCurrentFileZStreamPos64(tmp_unaligned);
			unzReadCurrentFile(tmp_unaligned, data.ptr(), data.size());
			unzCloseCurrentFile(tmp_unaligned);

			// align
			int padding = 0;
			if (!info.compression_method) {
				// Uncompressed file => Align
				long new_offset = file_offset + bias;
				padding = (ZIP_ALIGNMENT - (new_offset % ZIP_ALIGNMENT)) % ZIP_ALIGNMENT;
			}

			memset(extra + info.size_file_extra, 0, padding);

			// write
			zip_fileinfo zipfi = get_zip_fileinfo();

			zipOpenNewFileInZip2(final_apk,
					file.utf8().get_data(),
					&zipfi,
					extra,
					info.size_file_extra + padding,
					NULL,
					0,
					NULL,
					method,
					level,
					1); // raw write
			zipWriteInFileInZip(final_apk, data.ptr(), data.size());
			zipCloseFileInZipRaw(final_apk, info.uncompressed_size, info.crc);

			bias += padding;

			ret = unzGoToNextFile(tmp_unaligned);
		}

		zipClose(final_apk, NULL);
		unzClose(tmp_unaligned);

		if (err) {
			return err;
		}

		return OK;
	}

	virtual void get_platform_features(List<String> *r_features) {

		r_features->push_back("mobile");
		r_features->push_back("Android");
	}

	EditorExportAndroid() {

		Ref<Image> img = memnew(Image(_android_logo));
		logo.instance();
		logo->create_from_image(img);

		img = Ref<Image>(memnew(Image(_android_run_icon)));
		run_icon.instance();
		run_icon->create_from_image(img);

		device_lock = Mutex::create();
		device_thread = Thread::create(_device_poll_thread, this);
		devices_changed = true;
		quit_request = false;
	}

	~EditorExportAndroid() {
		quit_request = true;
		Thread::wait_to_finish(device_thread);
		memdelete(device_lock);
		memdelete(device_thread);
	}
};

void register_android_exporter() {

	String exe_ext;
	if (OS::get_singleton()->get_name() == "Windows") {
		exe_ext = "*.exe";
	}

	EDITOR_DEF("export/android/adb", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/android/adb", PROPERTY_HINT_GLOBAL_FILE, exe_ext));
	EDITOR_DEF("export/android/jarsigner", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/android/jarsigner", PROPERTY_HINT_GLOBAL_FILE, exe_ext));
	EDITOR_DEF("export/android/debug_keystore", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/android/debug_keystore", PROPERTY_HINT_GLOBAL_FILE, "keystore"));
	EDITOR_DEF("export/android/debug_keystore_user", "androiddebugkey");
	EDITOR_DEF("export/android/debug_keystore_pass", "android");
	EDITOR_DEF("export/android/force_system_user", false);

	EDITOR_DEF("export/android/timestamping_authority_url", "");
	EDITOR_DEF("export/android/shutdown_adb_on_exit", true);

	Ref<EditorExportAndroid> exporter = Ref<EditorExportAndroid>(memnew(EditorExportAndroid));
	EditorExport::get_singleton()->add_export_platform(exporter);
}

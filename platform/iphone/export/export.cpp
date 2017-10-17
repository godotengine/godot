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
#include "io/resource_saver.h"
#include "io/zip_io.h"
#include "os/file_access.h"
#include "os/os.h"
#include "platform/osx/logo.gen.h"
#include "project_settings.h"
#include "string.h"
#include "version.h"

#include <sys/stat.h>

class EditorExportPlatformIOS : public EditorExportPlatform {

	GDCLASS(EditorExportPlatformIOS, EditorExportPlatform);

	int version_code;

	Ref<ImageTexture> logo;

	typedef Error (*FileHandler)(String p_file, void *p_userdata);
	static Error _walk_dir_recursive(DirAccess *p_da, FileHandler p_handler, void *p_userdata);
	static Error _codesign(String p_file, void *p_userdata);

	void _fix_config_file(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &pfile, const String &p_name, const String &p_binary, bool p_debug);
	static Error _export_dylibs(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);
	Error _export_loading_screens(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir);
	Error _export_icons(const Ref<EditorExportPreset> &p_preset, const String &p_iconset_dir);

protected:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features);
	virtual void get_export_options(List<ExportOption> *r_options);

public:
	virtual String get_name() const { return "iOS"; }
	virtual String get_os_name() const { return "iOS"; }
	virtual Ref<Texture> get_logo() const { return logo; }

	virtual String get_binary_extension() const { return "ipa"; }
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const;

	virtual void get_platform_features(List<String> *r_features) {

		r_features->push_back("mobile");
		r_features->push_back("iOS");
	}

	EditorExportPlatformIOS();
	~EditorExportPlatformIOS();
};

void EditorExportPlatformIOS::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) {

	if (p_preset->get("texture_format/s3tc")) {
		r_features->push_back("s3tc");
	}
	if (p_preset->get("texture_format/etc")) {
		r_features->push_back("etc");
	}
	if (p_preset->get("texture_format/etc2")) {
		r_features->push_back("etc2");
	}
}

void EditorExportPlatformIOS::get_export_options(List<ExportOption> *r_options) {

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_package/debug", PROPERTY_HINT_GLOBAL_FILE, "zip"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_package/release", PROPERTY_HINT_GLOBAL_FILE, "zip"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/app_store_team_id"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/provisioning_profile_uuid_debug"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/code_sign_identity_debug"), "iPhone Developer"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "application/export_method_debug", PROPERTY_HINT_ENUM, "App Store,Development,Ad-Hoc,Enterprise"), 1));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/provisioning_profile_uuid_release"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/code_sign_identity_release"), "iPhone Distribution"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "application/export_method_release", PROPERTY_HINT_ENUM, "App Store,Development,Ad-Hoc,Enterprise"), 0));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/info"), "Made with Godot Engine"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/identifier"), "org.godotengine.iosgame"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/signature"), "????"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/short_version"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/version"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/copyright"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "application/bits_mode", PROPERTY_HINT_ENUM, "Fat (32 & 64 bits),64 bits,32 bits"), 1));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "required_icons/iphone_120x120", PROPERTY_HINT_FILE, "png"), "")); // Home screen on iPhone/iPod Touch with retina display
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "required_icons/ipad_76x76", PROPERTY_HINT_FILE, "png"), "")); // Home screen on iPad

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "optional_icons/iphone_180x180", PROPERTY_HINT_FILE, "png"), "")); // Home screen on iPhone with retina HD display
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "optional_icons/ipad_152x152", PROPERTY_HINT_FILE, "png"), "")); // Home screen on iPad with retina display
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "optional_icons/ipad_167x167", PROPERTY_HINT_FILE, "png"), "")); // Home screen on iPad Pro
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "optional_icons/spotlight_40x40", PROPERTY_HINT_FILE, "png"), "")); // Spotlight
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "optional_icons/spotlight_80x80", PROPERTY_HINT_FILE, "png"), "")); // Spotlight on devices with retina display

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "landscape_launch_screens/iphone_2208x1242", PROPERTY_HINT_FILE, "png"), "")); // iPhone 6 Plus, 6s Plus, 7 Plus
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "landscape_launch_screens/ipad_2732x2048", PROPERTY_HINT_FILE, "png"), "")); // 12.9-inch iPad Pro
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "landscape_launch_screens/ipad_2048x1536", PROPERTY_HINT_FILE, "png"), "")); // Other iPads

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "portrait_launch_screens/iphone_640x1136", PROPERTY_HINT_FILE, "png"), "")); // iPhone 5, 5s, SE
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "portrait_launch_screens/iphone_750x1334", PROPERTY_HINT_FILE, "png"), "")); // iPhone 6, 6s, 7
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "portrait_launch_screens/iphone_1242x2208", PROPERTY_HINT_FILE, "png"), "")); // iPhone 6 Plus, 6s Plus, 7 Plus
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "portrait_launch_screens/ipad_2048x2732", PROPERTY_HINT_FILE, "png"), "")); // 12.9-inch iPad Pro
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "portrait_launch_screens/ipad_1536x2048", PROPERTY_HINT_FILE, "png"), "")); // Other iPads

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/s3tc"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc2"), true));

	/* probably need some more info */
}

void EditorExportPlatformIOS::_fix_config_file(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &pfile, const String &p_name, const String &p_binary, bool p_debug) {
	static const String export_method_string[] = {
		"app-store",
		"development",
		"ad-hoc",
		"enterprise"
	};
	String str;
	String strnew;
	str.parse_utf8((const char *)pfile.ptr(), pfile.size());
	print_line(str);
	Vector<String> lines = str.split("\n");
	for (int i = 0; i < lines.size(); i++) {
		if (lines[i].find("$binary") != -1) {
			strnew += lines[i].replace("$binary", p_binary) + "\n";
		} else if (lines[i].find("$name") != -1) {
			strnew += lines[i].replace("$name", p_name) + "\n";
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
		} else if (lines[i].find("$code_sign_identity_debug") != -1) {
			strnew += lines[i].replace("$code_sign_identity_debug", p_preset->get("application/code_sign_identity_debug")) + "\n";
		} else if (lines[i].find("$code_sign_identity_release") != -1) {
			strnew += lines[i].replace("$code_sign_identity_release", p_preset->get("application/code_sign_identity_release")) + "\n";
		} else {
			strnew += lines[i] + "\n";
		}
	}

	// !BAS! I'm assuming the 9 in the original code was a typo. I've added -1 or else it seems to also be adding our terminating zero...
	// should apply the same fix in our OSX export.
	CharString cs = strnew.utf8();
	pfile.resize(cs.size() - 1);
	for (int i = 0; i < cs.size() - 1; i++) {
		pfile[i] = cs[i];
	}
}

Error EditorExportPlatformIOS::_export_dylibs(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total) {
	if (!p_path.ends_with(".dylib")) return OK;
	const String &dest_dir = *(String *)p_userdata;
	String rel_path = p_path.replace_first("res://", "dylibs/");
	DirAccess *dest_dir_access = DirAccess::open(dest_dir);
	ERR_FAIL_COND_V(!dest_dir_access, ERR_CANT_OPEN);

	String base_dir = rel_path.get_base_dir();
	Error make_dir_err = OK;
	if (!dest_dir_access->dir_exists(base_dir)) {
		make_dir_err = dest_dir_access->make_dir_recursive(base_dir);
	}
	if (make_dir_err != OK) {
		memdelete(dest_dir_access);
		return make_dir_err;
	}

	Error copy_err = dest_dir_access->copy(p_path, dest_dir + rel_path);
	memdelete(dest_dir_access);

	return copy_err;
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
	ERR_FAIL_COND_V(!da, ERR_CANT_OPEN);

	for (int i = 0; i < (sizeof(icon_infos) / sizeof(icon_infos[0])); ++i) {
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

struct LoadingScreenInfo {
	const char *preset_key;
	const char *export_name;
};

static const LoadingScreenInfo loading_screen_infos[] = {
	{ "landscape_launch_screens/iphone_2208x1242", "Default-Landscape-736h@3x.png" },
	{ "landscape_launch_screens/ipad_2732x2048", "Default-Landscape-1366h@2x.png" },
	{ "landscape_launch_screens/ipad_2048x1536", "Default-Landscape@2x.png" },

	{ "portrait_launch_screens/iphone_640x1136", "Default-568h@2x.png" },
	{ "portrait_launch_screens/iphone_750x1334", "Default-667h@2x.png" },
	{ "portrait_launch_screens/iphone_1242x2208", "Default-Portrait-736h@3x.png" },
	{ "portrait_launch_screens/ipad_2048x2732", "Default-Portrait-1366h@2x.png" },
	{ "portrait_launch_screens/ipad_1536x2048", "Default-Portrait@2x.png" }
};

Error EditorExportPlatformIOS::_export_loading_screens(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir) {
	DirAccess *da = DirAccess::open(p_dest_dir);
	ERR_FAIL_COND_V(!da, ERR_CANT_OPEN);

	for (int i = 0; i < sizeof(loading_screen_infos) / sizeof(loading_screen_infos[0]); ++i) {
		LoadingScreenInfo info = loading_screen_infos[i];
		String loading_screen_file = p_preset->get(info.preset_key);
		Error err = da->copy(loading_screen_file, p_dest_dir + info.export_name);
		if (err) {
			memdelete(da);
			String err_str = String("Failed to export loading screen: ") + loading_screen_file;
			ERR_PRINT(err_str.utf8().get_data());
			return err;
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
			Error err = p_handler(current_dir + "/" + path, p_userdata);
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

	CodesignData(const Ref<EditorExportPreset> &p_preset, bool p_debug)
		: preset(p_preset), debug(p_debug) {
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

Error EditorExportPlatformIOS::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	String src_pkg_name;
	String dest_dir = p_path.get_base_dir() + "/";
	String binary_name = p_path.get_file().get_basename();

	EditorProgress ep("export", "Exporting for iOS", 5);

	String team_id = p_preset->get("application/app_store_team_id");
	ERR_EXPLAIN("App Store Team ID not specified - cannot configure the project.");
	ERR_FAIL_COND_V(team_id.length() == 0, ERR_CANT_OPEN);

	if (p_debug)
		src_pkg_name = p_preset->get("custom_package/debug");
	else
		src_pkg_name = p_preset->get("custom_package/release");

	if (src_pkg_name == "") {
		String err;
		src_pkg_name = find_export_template("iphone.zip", &err);
		if (src_pkg_name == "") {
			EditorNode::add_io_error(err);
			return ERR_FILE_NOT_FOUND;
		}
	}

	FileAccess *src_f = NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

	ep.step("Creating app", 0);

	unzFile src_pkg_zip = unzOpen2(src_pkg_name.utf8().get_data(), &io);
	if (!src_pkg_zip) {

		EditorNode::add_io_error("Could not find template app to export:\n" + src_pkg_name);
		return ERR_FILE_NOT_FOUND;
	}

	ERR_FAIL_COND_V(!src_pkg_zip, ERR_CANT_OPEN);
	int ret = unzGoToFirstFile(src_pkg_zip);

	String binary_to_use = "godot.iphone." + String(p_debug ? "debug" : "release") + ".";
	int bits_mode = p_preset->get("application/bits_mode");
	binary_to_use += String(bits_mode == 0 ? "fat" : bits_mode == 1 ? "arm64" : "armv7");

	print_line("binary: " + binary_to_use);
	String pkg_name;
	if (p_preset->get("application/name") != "")
		pkg_name = p_preset->get("application/name"); // app_name
	else if (String(ProjectSettings::get_singleton()->get("application/config/name")) != "")
		pkg_name = String(ProjectSettings::get_singleton()->get("application/config/name"));
	else
		pkg_name = "Unnamed";

	DirAccess *tmp_app_path = DirAccess::create_for_path(dest_dir);
	ERR_FAIL_COND_V(!tmp_app_path, ERR_CANT_CREATE)

	/* Now process our template */
	bool found_binary = false;
	int total_size = 0;

	Set<String> files_to_parse;
	files_to_parse.insert("godot_ios/godot_ios-Info.plist");
	files_to_parse.insert("godot_ios.xcodeproj/project.pbxproj");
	files_to_parse.insert("export_options.plist");
	files_to_parse.insert("godot_ios.xcodeproj/project.xcworkspace/contents.xcworkspacedata");
	files_to_parse.insert("godot_ios.xcodeproj/xcshareddata/xcschemes/godot_ios.xcscheme");

	print_line("Unzipping...");

	while (ret == UNZ_OK) {
		bool is_execute = false;

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
		unzReadCurrentFile(src_pkg_zip, data.ptr(), data.size());
		unzCloseCurrentFile(src_pkg_zip);

		//write

		file = file.replace_first("iphone/", "");

		if (files_to_parse.has(file)) {
			print_line(String("parse ") + file);
			_fix_config_file(p_preset, data, pkg_name, binary_name, p_debug);
		} else if (file.begins_with("godot.iphone")) {
			if (file != binary_to_use) {
				ret = unzGoToNextFile(src_pkg_zip);
				continue; //ignore!
			}
			found_binary = true;
			is_execute = true;
			file = "godot_ios.iphone";
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

#ifdef OSX_ENABLED
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

	if (!found_binary) {
		ERR_PRINTS("Requested template binary '" + binary_to_use + "' not found. It might be missing from your template archive.");
		memdelete(tmp_app_path);
		return ERR_FILE_NOT_FOUND;
	}

	String iconset_dir = dest_dir + binary_name + "/Images.xcassets/AppIcon.appiconset/";
	Error err = OK;
	if (!tmp_app_path->dir_exists(iconset_dir)) {
		Error err = tmp_app_path->make_dir_recursive(iconset_dir);
	}
	memdelete(tmp_app_path);
	if (err)
		return err;

	err = _export_icons(p_preset, iconset_dir);
	if (err)
		return err;

	err = _export_loading_screens(p_preset, dest_dir + binary_name + "/");
	if (err)
		return err;

	ep.step("Making .pck", 1);

	String pack_path = dest_dir + binary_name + ".pck";
	err = save_pack(p_preset, pack_path);
	if (err)
		return err;

	err = export_project_files(p_preset, _export_dylibs, &dest_dir);
	if (err)
		return err;

#ifdef OSX_ENABLED
	ep.step("Code-signing dylibs", 2);
	DirAccess *dylibs_dir = DirAccess::open(dest_dir + "dylibs");
	ERR_FAIL_COND_V(!dylibs_dir, ERR_CANT_OPEN);
	CodesignData codesign_data(p_preset, p_debug);
	err = _walk_dir_recursive(dylibs_dir, _codesign, &codesign_data);
	memdelete(dylibs_dir);
	ERR_FAIL_COND_V(err, err);

	ep.step("Making .xcarchive", 3);
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

	ep.step("Making .ipa", 4);
	List<String> export_args;
	export_args.push_back("-exportArchive");
	export_args.push_back("-archivePath");
	export_args.push_back(archive_path);
	export_args.push_back("-exportOptionsPlist");
	export_args.push_back(dest_dir + "export_options.plist");
	export_args.push_back("-exportPath");
	export_args.push_back(dest_dir);
	err = OS::get_singleton()->execute("xcodebuild", export_args, true);
	ERR_FAIL_COND_V(err, err);
#else
	print_line(".ipa can only be built on macOS. Leaving XCode project without building the package.");
#endif

	return OK;
}

bool EditorExportPlatformIOS::can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {

	bool valid = true;
	String err;

	if (!exists_export_template("iphone.zip", &err)) {
		valid = false;
	}

	if (p_preset->get("custom_package/debug") != "" && !FileAccess::exists(p_preset->get("custom_package/debug"))) {
		valid = false;
		err += "Custom debug package not found.\n";
	}

	if (p_preset->get("custom_package/release") != "" && !FileAccess::exists(p_preset->get("custom_package/release"))) {
		valid = false;
		err += "Custom release package not found.\n";
	}

	if (!err.empty())
		r_error = err;

	return valid;
}

EditorExportPlatformIOS::EditorExportPlatformIOS() {

	///@TODO need to create the correct logo
	//  Ref<Image> img = memnew(Image(_iphone_logo));
	Ref<Image> img = memnew(Image(_osx_logo));
	logo.instance();
	logo->create_from_image(img);
}

EditorExportPlatformIOS::~EditorExportPlatformIOS() {
}

void register_iphone_exporter() {

	Ref<EditorExportPlatformIOS> platform;
	platform.instance();

	EditorExport::get_singleton()->add_export_platform(platform);
}

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
#include "platform/iphone/logo.gen.h"
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

	struct IOSConfigData {
		String pkg_name;
		String binary_name;
		String plist_content;
		String architectures;
		String linker_flags;
		String cpp_code;
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
	};

	String _get_additional_plist_content();
	String _get_linker_flags();
	String _get_cpp_code();
	void _fix_config_file(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &pfile, const IOSConfigData &p_config, bool p_debug);
	Error _export_loading_screens(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir);
	Error _export_icons(const Ref<EditorExportPreset> &p_preset, const String &p_iconset_dir);

	Vector<ExportArchitecture> _get_supported_architectures();
	Vector<String> _get_preset_architectures(const Ref<EditorExportPreset> &p_preset);

	void _add_assets_to_project(Vector<uint8_t> &p_project_data, const Vector<IOSExportAsset> &p_additional_assets);
	Error _export_additional_assets(const String &p_out_dir, const Vector<String> &p_assets, bool p_is_framework, Vector<IOSExportAsset> &r_exported_assets);
	Error _export_additional_assets(const String &p_out_dir, const Vector<SharedObject> &p_libraries, Vector<IOSExportAsset> &r_exported_assets);

protected:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features);
	virtual void get_export_options(List<ExportOption> *r_options);

public:
	virtual String get_name() const { return "iOS"; }
	virtual String get_os_name() const { return "iOS"; }
	virtual Ref<Texture> get_logo() const { return logo; }

	virtual String get_binary_extension(const Ref<EditorExportPreset> &p_preset) const { return "ipa"; }
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
	Vector<String> architectures = _get_preset_architectures(p_preset);
	for (int i = 0; i < architectures.size(); ++i) {
		r_features->push_back(architectures[i]);
	}
}

Vector<EditorExportPlatformIOS::ExportArchitecture> EditorExportPlatformIOS::_get_supported_architectures() {
	Vector<ExportArchitecture> archs;
	archs.push_back(ExportArchitecture("armv7", true));
	archs.push_back(ExportArchitecture("arm64", true));
	return archs;
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

	Vector<ExportArchitecture> architectures = _get_supported_architectures();
	for (int i = 0; i < architectures.size(); ++i) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "architectures/" + architectures[i].name), architectures[i].is_default));
	}
}

void EditorExportPlatformIOS::_fix_config_file(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &pfile, const IOSConfigData &p_config, bool p_debug) {
	static const String export_method_string[] = {
		"app-store",
		"development",
		"ad-hoc",
		"enterprise"
	};
	String str;
	String strnew;
	str.parse_utf8((const char *)pfile.ptr(), pfile.size());
	Vector<String> lines = str.split("\n");
	for (int i = 0; i < lines.size(); i++) {
		if (lines[i].find("$binary") != -1) {
			strnew += lines[i].replace("$binary", p_config.binary_name) + "\n";
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
		for (int i = 0; i < sizeof(num) * 2; ++i) {
			uint8_t four_bits = (num >> (sizeof(num) * 8 - (i + 1) * 4)) & 0xF;
			ret[i] = _hex_char(four_bits);
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

void EditorExportPlatformIOS::_add_assets_to_project(Vector<uint8_t> &p_project_data, const Vector<IOSExportAsset> &p_additional_assets) {
	Vector<Ref<EditorExportPlugin> > export_plugins = EditorExport::get_singleton()->get_export_plugins();
	Vector<String> frameworks;
	for (int i = 0; i < export_plugins.size(); ++i) {
		Vector<String> plugin_frameworks = export_plugins[i]->get_ios_frameworks();
		for (int j = 0; j < plugin_frameworks.size(); ++j) {
			frameworks.push_back(plugin_frameworks[j]);
		}
	}

	// that is just a random number, we just need Godot IDs not to clash with
	// existing IDs in the project.
	PbxId current_id = { 0x58938401, 0, 0 };
	String pbx_files;
	String pbx_frameworks_build;
	String pbx_frameworks_refs;
	String pbx_resources_build;
	String pbx_resources_refs;

	const String file_info_format = String("$build_id = {isa = PBXBuildFile; fileRef = $ref_id; };\n") +
									"$ref_id = {isa = PBXFileReference; lastKnownFileType = $file_type; name = $name; path = \"$file_path\"; sourceTree = \"<group>\"; };\n";
	for (int i = 0; i < p_additional_assets.size(); ++i) {
		String build_id = (++current_id).str();
		String ref_id = (++current_id).str();
		const IOSExportAsset &asset = p_additional_assets[i];

		String type;
		if (asset.exported_path.ends_with(".framework")) {
			type = "wrapper.framework";
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
		pbx_files += file_info_format.format(format_dict, "$_");
	}

	String str = String::utf8((const char *)p_project_data.ptr(), p_project_data.size());
	str = str.replace("$additional_pbx_files", pbx_files);
	str = str.replace("$additional_pbx_frameworks_build", pbx_frameworks_build);
	str = str.replace("$additional_pbx_frameworks_refs", pbx_frameworks_refs);
	str = str.replace("$additional_pbx_resources_build", pbx_resources_build);
	str = str.replace("$additional_pbx_resources_refs", pbx_resources_refs);

	CharString cs = str.utf8();
	p_project_data.resize(cs.size() - 1);
	for (int i = 0; i < cs.size() - 1; i++) {
		p_project_data[i] = cs[i];
	}
}

Error EditorExportPlatformIOS::_export_additional_assets(const String &p_out_dir, const Vector<String> &p_assets, bool p_is_framework, Vector<IOSExportAsset> &r_exported_assets) {
	DirAccess *filesystem_da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	ERR_FAIL_COND_V(!filesystem_da, ERR_CANT_CREATE);
	for (int f_idx = 0; f_idx < p_assets.size(); ++f_idx) {
		String asset = p_assets[f_idx];
		if (!asset.begins_with("res://")) {
			// either SDK-builtin or already a part of the export template
			IOSExportAsset exported_asset = { asset, p_is_framework };
			r_exported_assets.push_back(exported_asset);
		} else {
			DirAccess *da = DirAccess::create_for_path(asset);
			if (!da) {
				memdelete(filesystem_da);
				ERR_FAIL_COND_V(!da, ERR_CANT_CREATE);
			}
			bool file_exists = da->file_exists(asset);
			bool dir_exists = da->dir_exists(asset);
			if (!file_exists && !dir_exists) {
				memdelete(da);
				memdelete(filesystem_da);
				return ERR_FILE_NOT_FOUND;
			}
			String additional_dir = p_is_framework && asset.ends_with(".dylib") ? "/dylibs/" : "/";
			String destination_dir = p_out_dir + additional_dir + asset.get_base_dir().replace("res://", "");
			if (!filesystem_da->dir_exists(destination_dir)) {
				Error make_dir_err = filesystem_da->make_dir_recursive(destination_dir);
				if (make_dir_err) {
					memdelete(da);
					memdelete(filesystem_da);
					return make_dir_err;
				}
			}

			String destination = destination_dir + "/" + asset.get_file();
			Error err = dir_exists ? da->copy_dir(asset, destination) : da->copy(asset, destination);
			memdelete(da);
			if (err) {
				memdelete(filesystem_da);
				return err;
			}
			IOSExportAsset exported_asset = { destination, p_is_framework };
			r_exported_assets.push_back(exported_asset);
		}
	}
	memdelete(filesystem_da);

	return OK;
}

Error EditorExportPlatformIOS::_export_additional_assets(const String &p_out_dir, const Vector<SharedObject> &p_libraries, Vector<IOSExportAsset> &r_exported_assets) {
	Vector<Ref<EditorExportPlugin> > export_plugins = EditorExport::get_singleton()->get_export_plugins();
	for (int i = 0; i < export_plugins.size(); i++) {
		Vector<String> frameworks = export_plugins[i]->get_ios_frameworks();
		Error err = _export_additional_assets(p_out_dir, frameworks, true, r_exported_assets);
		ERR_FAIL_COND_V(err, err);
		Vector<String> ios_bundle_files = export_plugins[i]->get_ios_bundle_files();
		err = _export_additional_assets(p_out_dir, ios_bundle_files, false, r_exported_assets);
		ERR_FAIL_COND_V(err, err);
	}

	Vector<String> library_paths;
	for (int i = 0; i < p_libraries.size(); ++i) {
		library_paths.push_back(p_libraries[i].path);
	}
	Error err = _export_additional_assets(p_out_dir, library_paths, true, r_exported_assets);
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

Error EditorExportPlatformIOS::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

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

	ep.step("Making .pck", 0);
	String pack_path = dest_dir + binary_name + ".pck";
	Vector<SharedObject> libraries;
	Error err = save_pack(p_preset, pack_path, &libraries);
	if (err)
		return err;

	ep.step("Extracting and configuring Xcode project", 1);

	String library_to_use = "libgodot.iphone." + String(p_debug ? "debug" : "release") + ".fat.a";

	print_line("static library: " + library_to_use);
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

	IOSConfigData config_data = {
		pkg_name,
		binary_name,
		_get_additional_plist_content(),
		String(" ").join(_get_preset_architectures(p_preset)),
		_get_linker_flags(),
		_get_cpp_code()
	};

	DirAccess *tmp_app_path = DirAccess::create_for_path(dest_dir);
	ERR_FAIL_COND_V(!tmp_app_path, ERR_CANT_CREATE)

	print_line("Unzipping...");
	FileAccess *src_f = NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);
	unzFile src_pkg_zip = unzOpen2(src_pkg_name.utf8().get_data(), &io);
	if (!src_pkg_zip) {
		EditorNode::add_io_error("Could not open export template (not a zip file?):\n" + src_pkg_name);
		return ERR_CANT_OPEN;
	}
	ERR_FAIL_COND_V(!src_pkg_zip, ERR_CANT_OPEN);
	int ret = unzGoToFirstFile(src_pkg_zip);
	Vector<uint8_t> project_file_data;
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
		unzReadCurrentFile(src_pkg_zip, data.ptrw(), data.size());
		unzCloseCurrentFile(src_pkg_zip);

		//write

		file = file.replace_first("iphone/", "");

		if (files_to_parse.has(file)) {
			print_line(String("parse ") + file);
			_fix_config_file(p_preset, data, config_data, p_debug);
		} else if (file.begins_with("libgodot.iphone")) {
			if (file != library_to_use) {
				ret = unzGoToNextFile(src_pkg_zip);
				continue; //ignore!
			}
			found_library = true;
			is_execute = true;
			file = "godot_ios.a";
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

	if (!found_library) {
		ERR_PRINTS("Requested template library '" + library_to_use + "' not found. It might be missing from your template archive.");
		memdelete(tmp_app_path);
		return ERR_FILE_NOT_FOUND;
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

	err = _export_loading_screens(p_preset, dest_dir + binary_name + "/");
	if (err)
		return err;

	print_line("Exporting additional assets");
	Vector<IOSExportAsset> assets;
	_export_additional_assets(dest_dir + binary_name, libraries, assets);
	_add_assets_to_project(project_file_data, assets);
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
	ep.step("Code-signing dylibs", 2);
	DirAccess *dylibs_dir = DirAccess::open(dest_dir + binary_name + "/dylibs");
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

	r_missing_templates = !valid;
	return valid;
}

EditorExportPlatformIOS::EditorExportPlatformIOS() {

	Ref<Image> img = memnew(Image(_iphone_logo));
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

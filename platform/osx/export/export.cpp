/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "core/version.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "platform/osx/logo.gen.h"
#include "string.h"
#include <sys/stat.h>

class EditorExportPlatformOSX : public EditorExportPlatform {
	GDCLASS(EditorExportPlatformOSX, EditorExportPlatform);

	int version_code;

	Ref<ImageTexture> logo;

	void _fix_plist(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &plist, const String &p_binary);
	void _make_icon(const Ref<Image> &p_icon, Vector<uint8_t> &p_data);

	Error _notarize(const Ref<EditorExportPreset> &p_preset, const String &p_path);
	Error _code_sign(const Ref<EditorExportPreset> &p_preset, const String &p_path);
	Error _create_dmg(const String &p_dmg_path, const String &p_pkg_name, const String &p_app_path_name);
	void _zip_folder_recursive(zipFile &p_zip, const String &p_root_path, const String &p_folder, const String &p_pkg_name);

#ifdef OSX_ENABLED
	bool use_codesign() const { return true; }
	bool use_dmg() const { return true; }
#else
	bool use_codesign() const { return false; }
	bool use_dmg() const { return false; }
#endif
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

protected:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) override;
	virtual void get_export_options(List<ExportOption> *r_options) override;

public:
	virtual String get_name() const override { return "macOS"; }
	virtual String get_os_name() const override { return "macOS"; }
	virtual Ref<Texture2D> get_logo() const override { return logo; }

	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const override {
		List<String> list;
		if (use_dmg()) {
			list.push_back("dmg");
		}
		list.push_back("zip");
		return list;
	}
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0) override;

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const override;

	virtual void get_platform_features(List<String> *r_features) override {
		r_features->push_back("pc");
		r_features->push_back("s3tc");
		r_features->push_back("OSX");
	}

	virtual void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features) override {
	}

	EditorExportPlatformOSX();
	~EditorExportPlatformOSX();
};

void EditorExportPlatformOSX::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) {
	if (p_preset->get("texture_format/s3tc")) {
		r_features->push_back("s3tc");
	}
	if (p_preset->get("texture_format/etc")) {
		r_features->push_back("etc");
	}
	if (p_preset->get("texture_format/etc2")) {
		r_features->push_back("etc2");
	}

	r_features->push_back("64");
}

void EditorExportPlatformOSX::get_export_options(List<ExportOption> *r_options) {
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/debug", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/release", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/info"), "Made with Godot Engine"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/icon", PROPERTY_HINT_FILE, "*.png,*.icns"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/bundle_identifier", PROPERTY_HINT_PLACEHOLDER_TEXT, "com.example.game"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/signature"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/short_version"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/version"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/copyright"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "display/high_res"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "privacy/camera_usage_description", PROPERTY_HINT_PLACEHOLDER_TEXT, "Provide a message if you need to use the camera"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "privacy/microphone_usage_description", PROPERTY_HINT_PLACEHOLDER_TEXT, "Provide a message if you need to use the microphone"), ""));

#ifdef OSX_ENABLED
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "codesign/enable"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "codesign/identity", PROPERTY_HINT_PLACEHOLDER_TEXT, "Type: Name (ID)"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "codesign/timestamp"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "codesign/hardened_runtime"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "codesign/entitlements", PROPERTY_HINT_GLOBAL_FILE, "*.plist"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "codesign/custom_options"), PackedStringArray()));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "notarization/enable"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "notarization/apple_id_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Apple ID email"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "notarization/apple_id_password", PROPERTY_HINT_PLACEHOLDER_TEXT, "Enable two-factor authentication and provide app-specific password"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "notarization/apple_team_id", PROPERTY_HINT_PLACEHOLDER_TEXT, "Provide team ID if your Apple ID belongs to multiple teams"), ""));
#endif

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/s3tc"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc2"), false));
}

void _rgba8_to_packbits_encode(int p_ch, int p_size, Vector<uint8_t> &p_source, Vector<uint8_t> &p_dest) {
	int src_len = p_size * p_size;

	Vector<uint8_t> result;
	result.resize(src_len * 1.25); //temp vector for rle encoded data, make it 25% larger for worst case scenario
	int res_size = 0;

	uint8_t buf[128];
	int buf_size = 0;

	int i = 0;
	while (i < src_len) {
		uint8_t cur = p_source.ptr()[i * 4 + p_ch];

		if (i < src_len - 2) {
			if ((p_source.ptr()[(i + 1) * 4 + p_ch] == cur) && (p_source.ptr()[(i + 2) * 4 + p_ch] == cur)) {
				if (buf_size > 0) {
					result.write[res_size++] = (uint8_t)(buf_size - 1);
					copymem(&result.write[res_size], &buf, buf_size);
					res_size += buf_size;
					buf_size = 0;
				}

				uint8_t lim = i + 130 >= src_len ? src_len - i - 1 : 130;
				bool hit_lim = true;

				for (int j = 3; j <= lim; j++) {
					if (p_source.ptr()[(i + j) * 4 + p_ch] != cur) {
						hit_lim = false;
						i = i + j - 1;
						result.write[res_size++] = (uint8_t)(j - 3 + 0x80);
						result.write[res_size++] = cur;
						break;
					}
				}
				if (hit_lim) {
					result.write[res_size++] = (uint8_t)(lim - 3 + 0x80);
					result.write[res_size++] = cur;
					i = i + lim;
				}
			} else {
				buf[buf_size++] = cur;
				if (buf_size == 128) {
					result.write[res_size++] = (uint8_t)(buf_size - 1);
					copymem(&result.write[res_size], &buf, buf_size);
					res_size += buf_size;
					buf_size = 0;
				}
			}
		} else {
			buf[buf_size++] = cur;
			result.write[res_size++] = (uint8_t)(buf_size - 1);
			copymem(&result.write[res_size], &buf, buf_size);
			res_size += buf_size;
			buf_size = 0;
		}

		i++;
	}

	int ofs = p_dest.size();
	p_dest.resize(p_dest.size() + res_size);
	copymem(&p_dest.write[ofs], result.ptr(), res_size);
}

void EditorExportPlatformOSX::_make_icon(const Ref<Image> &p_icon, Vector<uint8_t> &p_data) {
	Ref<ImageTexture> it = memnew(ImageTexture);

	Vector<uint8_t> data;

	data.resize(8);
	data.write[0] = 'i';
	data.write[1] = 'c';
	data.write[2] = 'n';
	data.write[3] = 's';

	struct MacOSIconInfo {
		const char *name;
		const char *mask_name;
		bool is_png;
		int size;
	};

	static const MacOSIconInfo icon_infos[] = {
		{ "ic10", "", true, 1024 }, //1024x1024 32-bit PNG and 512x512@2x 32-bit "retina" PNG
		{ "ic09", "", true, 512 }, //512×512 32-bit PNG
		{ "ic14", "", true, 512 }, //256x256@2x 32-bit "retina" PNG
		{ "ic08", "", true, 256 }, //256×256 32-bit PNG
		{ "ic13", "", true, 256 }, //128x128@2x 32-bit "retina" PNG
		{ "ic07", "", true, 128 }, //128x128 32-bit PNG
		{ "ic12", "", true, 64 }, //32x32@2x 32-bit "retina" PNG
		{ "ic11", "", true, 32 }, //16x16@2x 32-bit "retina" PNG
		{ "il32", "l8mk", false, 32 }, //32x32 24-bit RLE + 8-bit uncompressed mask
		{ "is32", "s8mk", false, 16 } //16x16 24-bit RLE + 8-bit uncompressed mask
	};

	for (uint64_t i = 0; i < (sizeof(icon_infos) / sizeof(icon_infos[0])); ++i) {
		Ref<Image> copy = p_icon; // does this make sense? doesn't this just increase the reference count instead of making a copy? Do we even need a copy?
		copy->convert(Image::FORMAT_RGBA8);
		copy->resize(icon_infos[i].size, icon_infos[i].size);

		if (icon_infos[i].is_png) {
			// Encode PNG icon.
			it->create_from_image(copy);
			String path = EditorSettings::get_singleton()->get_cache_dir().plus_file("icon.png");
			ResourceSaver::save(path, it);

			FileAccess *f = FileAccess::open(path, FileAccess::READ);
			if (!f) {
				// Clean up generated file.
				DirAccess::remove_file_or_error(path);
				ERR_FAIL();
			}

			int ofs = data.size();
			uint32_t len = f->get_len();
			data.resize(data.size() + len + 8);
			f->get_buffer(&data.write[ofs + 8], len);
			memdelete(f);
			len += 8;
			len = BSWAP32(len);
			copymem(&data.write[ofs], icon_infos[i].name, 4);
			encode_uint32(len, &data.write[ofs + 4]);

			// Clean up generated file.
			DirAccess::remove_file_or_error(path);

		} else {
			Vector<uint8_t> src_data = copy->get_data();

			//encode 24bit RGB RLE icon
			{
				int ofs = data.size();
				data.resize(data.size() + 8);

				_rgba8_to_packbits_encode(0, icon_infos[i].size, src_data, data); // encode R
				_rgba8_to_packbits_encode(1, icon_infos[i].size, src_data, data); // encode G
				_rgba8_to_packbits_encode(2, icon_infos[i].size, src_data, data); // encode B

				int len = data.size() - ofs;
				len = BSWAP32(len);
				copymem(&data.write[ofs], icon_infos[i].name, 4);
				encode_uint32(len, &data.write[ofs + 4]);
			}

			//encode 8bit mask uncompressed icon
			{
				int ofs = data.size();
				int len = copy->get_width() * copy->get_height();
				data.resize(data.size() + len + 8);

				for (int j = 0; j < len; j++) {
					data.write[ofs + 8 + j] = src_data.ptr()[j * 4 + 3];
				}
				len += 8;
				len = BSWAP32(len);
				copymem(&data.write[ofs], icon_infos[i].mask_name, 4);
				encode_uint32(len, &data.write[ofs + 4]);
			}
		}
	}

	uint32_t total_len = data.size();
	total_len = BSWAP32(total_len);
	encode_uint32(total_len, &data.write[4]);

	p_data = data;
}

void EditorExportPlatformOSX::_fix_plist(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &plist, const String &p_binary) {
	String str;
	String strnew;
	str.parse_utf8((const char *)plist.ptr(), plist.size());
	Vector<String> lines = str.split("\n");
	for (int i = 0; i < lines.size(); i++) {
		if (lines[i].find("$binary") != -1) {
			strnew += lines[i].replace("$binary", p_binary) + "\n";
		} else if (lines[i].find("$name") != -1) {
			strnew += lines[i].replace("$name", p_binary) + "\n";
		} else if (lines[i].find("$info") != -1) {
			strnew += lines[i].replace("$info", p_preset->get("application/info")) + "\n";
		} else if (lines[i].find("$bundle_identifier") != -1) {
			strnew += lines[i].replace("$bundle_identifier", p_preset->get("application/bundle_identifier")) + "\n";
		} else if (lines[i].find("$short_version") != -1) {
			strnew += lines[i].replace("$short_version", p_preset->get("application/short_version")) + "\n";
		} else if (lines[i].find("$version") != -1) {
			strnew += lines[i].replace("$version", p_preset->get("application/version")) + "\n";
		} else if (lines[i].find("$signature") != -1) {
			strnew += lines[i].replace("$signature", p_preset->get("application/signature")) + "\n";
		} else if (lines[i].find("$copyright") != -1) {
			strnew += lines[i].replace("$copyright", p_preset->get("application/copyright")) + "\n";
		} else if (lines[i].find("$highres") != -1) {
			strnew += lines[i].replace("$highres", p_preset->get("display/high_res") ? "<true/>" : "<false/>") + "\n";
		} else if (lines[i].find("$camera_usage_description") != -1) {
			String description = p_preset->get("privacy/camera_usage_description");
			strnew += lines[i].replace("$camera_usage_description", description) + "\n";
		} else if (lines[i].find("$microphone_usage_description") != -1) {
			String description = p_preset->get("privacy/microphone_usage_description");
			strnew += lines[i].replace("$microphone_usage_description", description) + "\n";
		} else {
			strnew += lines[i] + "\n";
		}
	}

	CharString cs = strnew.utf8();
	plist.resize(cs.size() - 1);
	for (int i = 0; i < cs.size() - 1; i++) {
		plist.write[i] = cs[i];
	}
}

/**
	If we're running the OSX version of the Godot editor we'll:
	- export our application bundle to a temporary folder
	- attempt to code sign it
	- and then wrap it up in a DMG
**/

Error EditorExportPlatformOSX::_notarize(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
#ifdef OSX_ENABLED
	List<String> args;

	args.push_back("altool");
	args.push_back("--notarize-app");

	args.push_back("--primary-bundle-id");
	args.push_back(p_preset->get("application/bundle_identifier"));

	args.push_back("--username");
	args.push_back(p_preset->get("notarization/apple_id_name"));

	args.push_back("--password");
	args.push_back(p_preset->get("notarization/apple_id_password"));

	args.push_back("--type");
	args.push_back("osx");

	if (p_preset->get("notarization/apple_team_id")) {
		args.push_back("--asc-provider");
		args.push_back(p_preset->get("notarization/apple_team_id"));
	}

	args.push_back("--file");
	args.push_back(p_path);

	String str;
	Error err = OS::get_singleton()->execute("xcrun", args, true, nullptr, &str, nullptr, true);
	ERR_FAIL_COND_V(err != OK, err);

	print_line("altool (" + p_path + "):\n" + str);
	if (str.find("RequestUUID") == -1) {
		EditorNode::add_io_error("altool: " + str);
		return FAILED;
	} else {
		print_line("Note: The notarization process generally takes less than an hour. When the process is completed, you'll receive an email.");
		print_line("      You can check progress manually by opening a Terminal and running the following command:");
		print_line("      \"xcrun altool --notarization-history 0 -u <your email> -p <app-specific pwd>\"");
	}

#endif

	return OK;
}

Error EditorExportPlatformOSX::_code_sign(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
#ifdef OSX_ENABLED
	List<String> args;

	if (p_preset->get("codesign/timestamp")) {
		args.push_back("--timestamp");
	}
	if (p_preset->get("codesign/hardened_runtime")) {
		args.push_back("--options");
		args.push_back("runtime");
	}

	if ((p_preset->get("codesign/entitlements") != "") && (p_path.get_extension() != "dmg")) {
		args.push_back("--entitlements");
		args.push_back(p_preset->get("codesign/entitlements"));
	}

	PackedStringArray user_args = p_preset->get("codesign/custom_options");
	for (int i = 0; i < user_args.size(); i++) {
		String user_arg = user_args[i].strip_edges();
		if (!user_arg.empty()) {
			args.push_back(user_arg);
		}
	}

	args.push_back("-s");
	args.push_back(p_preset->get("codesign/identity"));

	args.push_back("-v"); /* provide some more feedback */

	args.push_back(p_path);

	String str;
	Error err = OS::get_singleton()->execute("codesign", args, true, nullptr, &str, nullptr, true);
	ERR_FAIL_COND_V(err != OK, err);

	print_line("codesign (" + p_path + "):\n" + str);
	if (str.find("no identity found") != -1) {
		EditorNode::add_io_error("codesign: no identity found");
		return FAILED;
	}
	if ((str.find("unrecognized blob type") != -1) || (str.find("cannot read entitlement data") != -1)) {
		EditorNode::add_io_error("codesign: invalid entitlements file");
		return FAILED;
	}
#endif

	return OK;
}

Error EditorExportPlatformOSX::_create_dmg(const String &p_dmg_path, const String &p_pkg_name, const String &p_app_path_name) {
	List<String> args;

	if (FileAccess::exists(p_dmg_path)) {
		OS::get_singleton()->move_to_trash(p_dmg_path);
	}

	args.push_back("create");
	args.push_back(p_dmg_path);
	args.push_back("-volname");
	args.push_back(p_pkg_name);
	args.push_back("-fs");
	args.push_back("HFS+");
	args.push_back("-srcfolder");
	args.push_back(p_app_path_name);

	String str;
	Error err = OS::get_singleton()->execute("hdiutil", args, true, nullptr, &str, nullptr, true);
	ERR_FAIL_COND_V(err != OK, err);

	print_line("hdiutil returned: " + str);
	if (str.find("create failed") != -1) {
		if (str.find("File exists") != -1) {
			EditorNode::add_io_error("hdiutil: create failed - file exists");
		} else {
			EditorNode::add_io_error("hdiutil: create failed");
		}
		return FAILED;
	}

	return OK;
}

Error EditorExportPlatformOSX::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

	String src_pkg_name;

	EditorProgress ep("export", "Exporting for OSX", 3, true);

	if (p_debug) {
		src_pkg_name = p_preset->get("custom_template/debug");
	} else {
		src_pkg_name = p_preset->get("custom_template/release");
	}

	if (src_pkg_name == "") {
		String err;
		src_pkg_name = find_export_template("osx.zip", &err);
		if (src_pkg_name == "") {
			EditorNode::add_io_error(err);
			return ERR_FILE_NOT_FOUND;
		}
	}

	if (!DirAccess::exists(p_path.get_base_dir())) {
		return ERR_FILE_BAD_PATH;
	}

	FileAccess *src_f = nullptr;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

	if (ep.step("Creating app", 0)) {
		return ERR_SKIP;
	}

	unzFile src_pkg_zip = unzOpen2(src_pkg_name.utf8().get_data(), &io);
	if (!src_pkg_zip) {
		EditorNode::add_io_error("Could not find template app to export:\n" + src_pkg_name);
		return ERR_FILE_NOT_FOUND;
	}

	int ret = unzGoToFirstFile(src_pkg_zip);

	String binary_to_use = "godot_osx_" + String(p_debug ? "debug" : "release") + ".64";

	String pkg_name;
	if (p_preset->get("application/name") != "") {
		pkg_name = p_preset->get("application/name"); // app_name
	} else if (String(ProjectSettings::get_singleton()->get("application/config/name")) != "") {
		pkg_name = String(ProjectSettings::get_singleton()->get("application/config/name"));
	} else {
		pkg_name = "Unnamed";
	}

	String pkg_name_safe = OS::get_singleton()->get_safe_dir_name(pkg_name);

	Error err = OK;
	String tmp_app_path_name = "";

	DirAccess *tmp_app_path = nullptr;
	String export_format = use_dmg() && p_path.ends_with("dmg") ? "dmg" : "zip";

	// Create our application bundle.
	tmp_app_path_name = EditorSettings::get_singleton()->get_cache_dir().plus_file(pkg_name + ".app");
	print_line("Exporting to " + tmp_app_path_name);
	tmp_app_path = DirAccess::create_for_path(tmp_app_path_name);
	if (!tmp_app_path) {
		err = ERR_CANT_CREATE;
	}

	// Create our folder structure.
	if (err == OK) {
		print_line("Creating " + tmp_app_path_name + "/Contents/MacOS");
		err = tmp_app_path->make_dir_recursive(tmp_app_path_name + "/Contents/MacOS");
	}

	if (err == OK) {
		print_line("Creating " + tmp_app_path_name + "/Contents/Frameworks");
		err = tmp_app_path->make_dir_recursive(tmp_app_path_name + "/Contents/Frameworks");
	}

	if (err == OK) {
		print_line("Creating " + tmp_app_path_name + "/Contents/Resources");
		err = tmp_app_path->make_dir_recursive(tmp_app_path_name + "/Contents/Resources");
	}

	// Now process our template.
	bool found_binary = false;
	int total_size = 0;

	while (ret == UNZ_OK && err == OK) {
		bool is_execute = false;

		// Get filename.
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(src_pkg_zip, &info, fname, 16384, nullptr, 0, nullptr, 0);

		String file = fname;

		Vector<uint8_t> data;
		data.resize(info.uncompressed_size);

		// Read.
		unzOpenCurrentFile(src_pkg_zip);
		unzReadCurrentFile(src_pkg_zip, data.ptrw(), data.size());
		unzCloseCurrentFile(src_pkg_zip);

		// Write.
		file = file.replace_first("osx_template.app/", "");

		if (file == "Contents/Info.plist") {
			_fix_plist(p_preset, data, pkg_name);
		}

		if (file.begins_with("Contents/MacOS/godot_")) {
			if (file != "Contents/MacOS/" + binary_to_use) {
				ret = unzGoToNextFile(src_pkg_zip);
				continue; // skip
			}
			found_binary = true;
			is_execute = true;
			file = "Contents/MacOS/" + pkg_name;
		}

		if (file == "Contents/Resources/icon.icns") {
			// See if there is an icon.
			String iconpath;
			if (p_preset->get("application/icon") != "") {
				iconpath = p_preset->get("application/icon");
			} else {
				iconpath = ProjectSettings::get_singleton()->get("application/config/icon");
			}

			if (iconpath != "") {
				if (iconpath.get_extension() == "icns") {
					FileAccess *icon = FileAccess::open(iconpath, FileAccess::READ);
					if (icon) {
						data.resize(icon->get_len());
						icon->get_buffer(&data.write[0], icon->get_len());
						icon->close();
						memdelete(icon);
					}
				} else {
					Ref<Image> icon;
					icon.instance();
					icon->load(iconpath);
					if (!icon->empty()) {
						_make_icon(icon, data);
					}
				}
			}
		}

		if (data.size() > 0) {
			if (file.find("/data.mono.osx.64.release_debug/") != -1) {
				if (!p_debug) {
					ret = unzGoToNextFile(src_pkg_zip);
					continue; // skip
				}
				file = file.replace("/data.mono.osx.64.release_debug/", "/data_" + pkg_name_safe + "/");
			}
			if (file.find("/data.mono.osx.64.release/") != -1) {
				if (p_debug) {
					ret = unzGoToNextFile(src_pkg_zip);
					continue; // skip
				}
				file = file.replace("/data.mono.osx.64.release/", "/data_" + pkg_name_safe + "/");
			}

			print_line("ADDING: " + file + " size: " + itos(data.size()));
			total_size += data.size();

			// Write it into our application bundle.
			file = tmp_app_path_name.plus_file(file);
			if (err == OK) {
				err = tmp_app_path->make_dir_recursive(file.get_base_dir());
			}
			if (err == OK) {
				FileAccess *f = FileAccess::open(file, FileAccess::WRITE);
				if (f) {
					f->store_buffer(data.ptr(), data.size());
					f->close();
					if (is_execute) {
						// chmod with 0755 if the file is executable.
						FileAccess::set_unix_permissions(file, 0755);
					}
					memdelete(f);
				} else {
					err = ERR_CANT_CREATE;
				}
			}
		}

		ret = unzGoToNextFile(src_pkg_zip);
	}

	// We're done with our source zip.
	unzClose(src_pkg_zip);

	if (!found_binary) {
		ERR_PRINT("Requested template binary '" + binary_to_use + "' not found. It might be missing from your template archive.");
		err = ERR_FILE_NOT_FOUND;
	}

	if (err == OK) {
		if (ep.step("Making PKG", 1)) {
			return ERR_SKIP;
		}

		String pack_path = tmp_app_path_name + "/Contents/Resources/" + pkg_name + ".pck";
		Vector<SharedObject> shared_objects;
		err = save_pack(p_preset, pack_path, &shared_objects);

		// See if we can code sign our new package.
		bool sign_enabled = p_preset->get("codesign/enable");

		if (err == OK) {
			DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
			for (int i = 0; i < shared_objects.size(); i++) {
				err = da->copy(shared_objects[i].path, tmp_app_path_name + "/Contents/Frameworks/" + shared_objects[i].path.get_file());
				if (err == OK && sign_enabled) {
					err = _code_sign(p_preset, tmp_app_path_name + "/Contents/Frameworks/" + shared_objects[i].path.get_file());
				}
			}
			memdelete(da);
		}

		if (err == OK && sign_enabled) {
			if (ep.step("Code signing bundle", 2)) {
				return ERR_SKIP;
			}
			err = _code_sign(p_preset, tmp_app_path_name + "/Contents/MacOS/" + pkg_name);
		}

		if (export_format == "dmg") {
			// Create a DMG.
			if (err == OK) {
				if (ep.step("Making DMG", 3)) {
					return ERR_SKIP;
				}
				err = _create_dmg(p_path, pkg_name, tmp_app_path_name);
			}
			// Sign DMG.
			if (err == OK && sign_enabled) {
				if (ep.step("Code signing DMG", 3)) {
					return ERR_SKIP;
				}
				err = _code_sign(p_preset, p_path);
			}
		} else {
			// Create ZIP.
			if (err == OK) {
				if (ep.step("Making ZIP", 3)) {
					return ERR_SKIP;
				}
				if (FileAccess::exists(p_path)) {
					OS::get_singleton()->move_to_trash(p_path);
				}

				FileAccess *dst_f = nullptr;
				zlib_filefunc_def io_dst = zipio_create_io_from_file(&dst_f);
				zipFile zip = zipOpen2(p_path.utf8().get_data(), APPEND_STATUS_CREATE, nullptr, &io_dst);

				_zip_folder_recursive(zip, EditorSettings::get_singleton()->get_cache_dir(), pkg_name + ".app", pkg_name);

				zipClose(zip, nullptr);
			}
		}

		bool noto_enabled = p_preset->get("notarization/enable");
		if (err == OK && noto_enabled) {
			if (ep.step("Sending archive for notarization", 4)) {
				return ERR_SKIP;
			}
			err = _notarize(p_preset, p_path);
		}

		// Clean up temporary .app dir.
		OS::get_singleton()->move_to_trash(tmp_app_path_name);
	}

	return err;
}

void EditorExportPlatformOSX::_zip_folder_recursive(zipFile &p_zip, const String &p_root_path, const String &p_folder, const String &p_pkg_name) {
	String dir = p_root_path.plus_file(p_folder);

	DirAccess *da = DirAccess::open(dir);
	da->list_dir_begin();
	String f;
	while ((f = da->get_next()) != "") {
		if (f == "." || f == "..") {
			continue;
		}
		if (da->current_is_dir()) {
			_zip_folder_recursive(p_zip, p_root_path, p_folder.plus_file(f), p_pkg_name);
		} else {
			bool is_executable = (p_folder.ends_with("MacOS") && (f == p_pkg_name));

			OS::Time time = OS::get_singleton()->get_time();
			OS::Date date = OS::get_singleton()->get_date();

			zip_fileinfo zipfi;
			zipfi.tmz_date.tm_hour = time.hour;
			zipfi.tmz_date.tm_mday = date.day;
			zipfi.tmz_date.tm_min = time.min;
			zipfi.tmz_date.tm_mon = date.month - 1; // Note: "tm" month range - 0..11, Godot month range - 1..12, http://www.cplusplus.com/reference/ctime/tm/
			zipfi.tmz_date.tm_sec = time.sec;
			zipfi.tmz_date.tm_year = date.year;
			zipfi.dosDate = 0;
			// 0100000: regular file type
			// 0000755: permissions rwxr-xr-x
			// 0000644: permissions rw-r--r--
			uint32_t _mode = (is_executable ? 0100755 : 0100644);
			zipfi.external_fa = (_mode << 16L) | !(_mode & 0200);
			zipfi.internal_fa = 0;

			zipOpenNewFileInZip4(p_zip,
					p_folder.plus_file(f).utf8().get_data(),
					&zipfi,
					nullptr,
					0,
					nullptr,
					0,
					nullptr,
					Z_DEFLATED,
					Z_DEFAULT_COMPRESSION,
					0,
					-MAX_WBITS,
					DEF_MEM_LEVEL,
					Z_DEFAULT_STRATEGY,
					nullptr,
					0,
					0x0314, // "version made by", 0x03 - Unix, 0x14 - ZIP specification version 2.0, required to store Unix file permissions
					0);

			Vector<uint8_t> array = FileAccess::get_file_as_array(dir.plus_file(f));
			zipWriteInFileInZip(p_zip, array.ptr(), array.size());
			zipCloseFileInZip(p_zip);
		}
	}
	da->list_dir_end();
	memdelete(da);
}

bool EditorExportPlatformOSX::can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {
	String err;
	bool valid = false;

	// Look for export templates (first official, and if defined custom templates).

	bool dvalid = exists_export_template("osx.zip", &err);
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

	String identifier = p_preset->get("application/bundle_identifier");
	String pn_err;
	if (!is_package_name_valid(identifier, &pn_err)) {
		err += TTR("Invalid bundle identifier:") + " " + pn_err + "\n";
		valid = false;
	}

	bool sign_enabled = p_preset->get("codesign/enable");
	if (sign_enabled) {
		if (p_preset->get("codesign/identity") == "") {
			err += TTR("Codesign: identity not specified.") + "\n";
			valid = false;
		}
	}
	bool noto_enabled = p_preset->get("notarization/enable");
	if (noto_enabled) {
		if (!sign_enabled) {
			err += TTR("Notarization: code signing required.") + "\n";
			valid = false;
		}
		bool hr_enabled = p_preset->get("codesign/hardened_runtime");
		if (!hr_enabled) {
			err += TTR("Notarization: hardened runtime required.") + "\n";
			valid = false;
		}
		if (p_preset->get("notarization/apple_id_name") == "") {
			err += TTR("Notarization: Apple ID name not specified.") + "\n";
			valid = false;
		}
		if (p_preset->get("notarization/apple_id_password") == "") {
			err += TTR("Notarization: Apple ID password not specified.") + "\n";
			valid = false;
		}
	}

	if (!err.empty()) {
		r_error = err;
	}
	return valid;
}

EditorExportPlatformOSX::EditorExportPlatformOSX() {
	Ref<Image> img = memnew(Image(_osx_logo));
	logo.instance();
	logo->create_from_image(img);
}

EditorExportPlatformOSX::~EditorExportPlatformOSX() {
}

void register_osx_exporter() {
	Ref<EditorExportPlatformOSX> platform;
	platform.instance();

	EditorExport::get_singleton()->add_export_platform(platform);
}

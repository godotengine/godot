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

	void _fix_config_file(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &pfile, const String &p_name, const String &p_binary);

protected:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features);
	virtual void get_export_options(List<ExportOption> *r_options);

public:
	virtual String get_name() const { return "iOS"; }
	virtual String get_os_name() const { return "iOS"; }
	virtual Ref<Texture> get_logo() const { return logo; }

	virtual String get_binary_extension() const { return "xcodeproj"; }
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

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/info"), "Made with Godot Engine"));
	//  r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/icon", PROPERTY_HINT_FILE, "png"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/identifier"), "org.godotengine.iosgame"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/signature"), "godotiosgame"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/short_version"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/version"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/copyright"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "application/bits_mode", PROPERTY_HINT_ENUM, "Fat (32 & 64 bits),64 bits,32 bits"), 1));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/s3tc"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc2"), true));

	/* probably need some more info */
}

void EditorExportPlatformIOS::_fix_config_file(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &pfile, const String &p_name, const String &p_binary) {

	String str;
	String strnew;
	str.parse_utf8((const char *)pfile.ptr(), pfile.size());
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

Error EditorExportPlatformIOS::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	String src_pkg_name;
	String dest_dir = p_path.get_base_dir() + "/";
	String binary_name = p_path.get_file().get_basename();

	EditorProgress ep("export", "Exporting for iOS", 3);

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

		if (file == "godot_ios.xcodeproj/project.pbxproj") {
			print_line("parse pbxproj");
			_fix_config_file(p_preset, data, pkg_name, binary_name);
		} else if (file == "godot_ios/godot_ios-Info.plist") {
			print_line("parse plist");
			_fix_config_file(p_preset, data, pkg_name, binary_name);
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
					return ERR_CANT_CREATE;
				}
			}

			/* write the file */
			FileAccess *f = FileAccess::open(file, FileAccess::WRITE);
			if (!f) {
				ERR_PRINTS("Can't write '" + file + "'.");
				unzClose(src_pkg_zip);
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
		unzClose(src_pkg_zip);
		return ERR_FILE_NOT_FOUND;
	}

	ep.step("Making PKG", 1);

	String pack_path = dest_dir + binary_name + ".pck";
	Error err = save_pack(p_preset, pack_path);

	if (err) {
		return err;
	}

#ifdef OSX_ENABLED
	/* and open up xcode with our new project.... */
	List<String> args;
	args.push_back(p_path);
	err = OS::get_singleton()->execute("/usr/bin/open", args, false);
	ERR_FAIL_COND_V(err, err);

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

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

class EditorExportPlatformOSX : public EditorExportPlatform {

	GDCLASS(EditorExportPlatformOSX, EditorExportPlatform);

	int version_code;

	Ref<ImageTexture> logo;

	void _fix_plist(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &plist, const String &p_binary);
	void _make_icon(const Ref<Image> &p_icon, Vector<uint8_t> &p_data);

	Error _code_sign(const Ref<EditorExportPreset> &p_preset, const String &p_path);
	Error _create_dmg(const String &p_dmg_path, const String &p_pkg_name, const String &p_app_path_name);

#ifdef OSX_ENABLED
	bool use_codesign() const { return true; }
	bool use_dmg() const { return true; }
#else
	bool use_codesign() const { return false; }
	bool use_dmg() const { return false; }
#endif

protected:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features);
	virtual void get_export_options(List<ExportOption> *r_options);

public:
	virtual String get_name() const { return "Mac OSX"; }
	virtual String get_os_name() const { return "OSX"; }
	virtual Ref<Texture> get_logo() const { return logo; }

	virtual String get_binary_extension(const Ref<EditorExportPreset> &p_preset) const { return use_dmg() ? "dmg" : "zip"; }
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const;

	virtual void get_platform_features(List<String> *r_features) {

		r_features->push_back("pc");
		r_features->push_back("s3tc");
		r_features->push_back("OSX");
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

	int bits = p_preset->get("application/bits_mode");

	if (bits == 0 || bits == 1) {
		r_features->push_back("64");
	}

	if (bits == 0 || bits == 2) {
		r_features->push_back("32");
	}
}

void EditorExportPlatformOSX::get_export_options(List<ExportOption> *r_options) {

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_package/debug", PROPERTY_HINT_GLOBAL_FILE, "zip"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_package/release", PROPERTY_HINT_GLOBAL_FILE, "zip"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/info"), "Made with Godot Engine"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/icon", PROPERTY_HINT_FILE, "png"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/identifier"), "org.godotengine.macgame"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/signature"), "godotmacgame"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/short_version"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/version"), "1.0"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/copyright"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "application/bits_mode", PROPERTY_HINT_ENUM, "Fat (32 & 64 bits),64 bits,32 bits"), 0));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "display/high_res"), false));

#ifdef OSX_ENABLED
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "codesign/identity"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "codesign/entitlements"), ""));
#endif

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/s3tc"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "texture_format/etc2"), false));
}

void EditorExportPlatformOSX::_make_icon(const Ref<Image> &p_icon, Vector<uint8_t> &p_data) {

	Ref<ImageTexture> it = memnew(ImageTexture);
	int size = 512;

	Vector<uint8_t> data;

	data.resize(8);
	data[0] = 'i';
	data[1] = 'c';
	data[2] = 'n';
	data[3] = 's';

	const char *name[] = { "ic09", "ic08", "ic07", "icp6", "icp5", "icp4" };
	int index = 0;

	while (size >= 16) {

		Ref<Image> copy = p_icon; // does this make sense? doesn't this just increase the reference count instead of making a copy? Do we even need a copy?
		copy->convert(Image::FORMAT_RGBA8);
		copy->resize(size, size);
		it->create_from_image(copy);
		String path = EditorSettings::get_singleton()->get_cache_dir().plus_file("icon.png");
		ResourceSaver::save(path, it);

		FileAccess *f = FileAccess::open(path, FileAccess::READ);
		ERR_FAIL_COND(!f);

		int ofs = data.size();
		uint32_t len = f->get_len();
		data.resize(data.size() + len + 8);
		f->get_buffer(&data[ofs + 8], len);
		memdelete(f);
		len += 8;
		len = BSWAP32(len);
		copymem(&data[ofs], name[index], 4);
		encode_uint32(len, &data[ofs + 4]);
		index++;
		size /= 2;
	}

	uint32_t total_len = data.size();
	total_len = BSWAP32(total_len);
	encode_uint32(total_len, &data[4]);

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
		} else if (lines[i].find("$highres") != -1) {
			strnew += lines[i].replace("$highres", p_preset->get("display/high_res") ? "<true/>" : "<false/>") + "\n";
		} else {
			strnew += lines[i] + "\n";
		}
	}

	CharString cs = strnew.utf8();
	plist.resize(cs.size() - 1);
	for (int i = 0; i < cs.size() - 1; i++) {
		plist[i] = cs[i];
	}
}

/**
	If we're running the OSX version of the Godot editor we'll:
	- export our application bundle to a temporary folder
	- attempt to code sign it
	- and then wrap it up in a DMG
**/

Error EditorExportPlatformOSX::_code_sign(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
	List<String> args;

	if (p_preset->get("codesign/entitlements") != "") {
		/* this should point to our entitlements.plist file that sandboxes our application, I don't know if this should also be placed in our app bundle */
		args.push_back("-entitlements");
		args.push_back(p_preset->get("codesign/entitlements"));
	}
	args.push_back("-s");
	args.push_back(p_preset->get("codesign/identity"));
	args.push_back("-v"); /* provide some more feedback */
	args.push_back(p_path);

	String str;
	Error err = OS::get_singleton()->execute("codesign", args, true, NULL, &str, NULL, true);
	ERR_FAIL_COND_V(err != OK, err);

	print_line("codesign: " + str);
	if (str.find("no identity found") != -1) {
		EditorNode::add_io_error("codesign: no identity found");
		return FAILED;
	}

	return OK;
}

Error EditorExportPlatformOSX::_create_dmg(const String &p_dmg_path, const String &p_pkg_name, const String &p_app_path_name) {
	List<String> args;

	OS::get_singleton()->move_to_trash(p_dmg_path);

	args.push_back("create");
	args.push_back(p_dmg_path);
	args.push_back("-volname");
	args.push_back(p_pkg_name);
	args.push_back("-fs");
	args.push_back("HFS+");
	args.push_back("-srcfolder");
	args.push_back(p_app_path_name);

	String str;
	Error err = OS::get_singleton()->execute("hdiutil", args, true, NULL, &str, NULL, true);
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

	EditorProgress ep("export", "Exporting for OSX", 3);

	if (p_debug)
		src_pkg_name = p_preset->get("custom_package/debug");
	else
		src_pkg_name = p_preset->get("custom_package/release");

	if (src_pkg_name == "") {
		String err;
		src_pkg_name = find_export_template("osx.zip", &err);
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

	String binary_to_use = "godot_osx_" + String(p_debug ? "debug" : "release") + ".";
	int bits_mode = p_preset->get("application/bits_mode");
	binary_to_use += String(bits_mode == 0 ? "fat" : bits_mode == 1 ? "64" : "32");

	print_line("binary: " + binary_to_use);
	String pkg_name;
	if (p_preset->get("application/name") != "")
		pkg_name = p_preset->get("application/name"); // app_name
	else if (String(ProjectSettings::get_singleton()->get("application/config/name")) != "")
		pkg_name = String(ProjectSettings::get_singleton()->get("application/config/name"));
	else
		pkg_name = "Unnamed";

	Error err = OK;
	String tmp_app_path_name = "";
	zlib_filefunc_def io2 = io;
	FileAccess *dst_f = NULL;
	io2.opaque = &dst_f;
	zipFile dst_pkg_zip = NULL;

	if (use_dmg()) {
		// We're on OSX so we can export to DMG, but first we create our application bundle
		tmp_app_path_name = EditorSettings::get_singleton()->get_cache_dir().plus_file(pkg_name + ".app");
		print_line("Exporting to " + tmp_app_path_name);
		DirAccess *tmp_app_path = DirAccess::create_for_path(tmp_app_path_name);
		if (!tmp_app_path) {
			err = ERR_CANT_CREATE;
		}

		// Create our folder structure or rely on unzip?
		if (err == OK) {
			print_line("Creating " + tmp_app_path_name + "/Contents/MacOS");
			err = tmp_app_path->make_dir_recursive(tmp_app_path_name + "/Contents/MacOS");
		}

		if (err == OK) {
			print_line("Creating " + tmp_app_path_name + "/Contents/Resources");
			err = tmp_app_path->make_dir_recursive(tmp_app_path_name + "/Contents/Resources");
		}
	} else {
		// Open our destination zip file
		dst_pkg_zip = zipOpen2(p_path.utf8().get_data(), APPEND_STATUS_CREATE, NULL, &io2);
		if (!dst_pkg_zip) {
			err = ERR_CANT_CREATE;
		}
	}

	// Now process our template
	bool found_binary = false;
	int total_size = 0;

	while (ret == UNZ_OK && err == OK) {
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

		file = file.replace_first("osx_template.app/", "");

		if (file == "Contents/Info.plist") {
			print_line("parse plist");
			_fix_plist(p_preset, data, pkg_name);
		}

		if (file.begins_with("Contents/MacOS/godot_")) {
			if (file != "Contents/MacOS/" + binary_to_use) {
				ret = unzGoToNextFile(src_pkg_zip);
				continue; //ignore!
			}
			found_binary = true;
			is_execute = true;
			file = "Contents/MacOS/" + pkg_name;
		}

		if (file == "Contents/Resources/icon.icns") {
			//see if there is an icon
			String iconpath;
			if (p_preset->get("application/icon") != "")
				iconpath = p_preset->get("application/icon");
			else
				iconpath = ProjectSettings::get_singleton()->get("application/config/icon");
			print_line("icon? " + iconpath);
			if (iconpath != "") {
				Ref<Image> icon;
				icon.instance();
				icon->load(iconpath);
				if (!icon->empty()) {
					print_line("loaded?");
					_make_icon(icon, data);
				}
			}
			//bleh?
		}

		if (data.size() > 0) {
			print_line("ADDING: " + file + " size: " + itos(data.size()));
			total_size += data.size();

			if (use_dmg()) {
				// write it into our application bundle
				file = tmp_app_path_name + "/" + file;

				// write the file, need to add chmod
				FileAccess *f = FileAccess::open(file, FileAccess::WRITE);
				if (f) {
					f->store_buffer(data.ptr(), data.size());
					f->close();
					if (is_execute) {
						// Chmod with 0755 if the file is executable
						f->_chmod(file, 0755);
					}
					memdelete(f);
				} else {
					err = ERR_CANT_CREATE;
				}
			} else {
				// add it to our zip file
				file = pkg_name + ".app/" + file;

				zip_fileinfo fi;
				fi.tmz_date.tm_hour = info.tmu_date.tm_hour;
				fi.tmz_date.tm_min = info.tmu_date.tm_min;
				fi.tmz_date.tm_sec = info.tmu_date.tm_sec;
				fi.tmz_date.tm_mon = info.tmu_date.tm_mon;
				fi.tmz_date.tm_mday = info.tmu_date.tm_mday;
				fi.tmz_date.tm_year = info.tmu_date.tm_year;
				fi.dosDate = info.dosDate;
				fi.internal_fa = info.internal_fa;
				fi.external_fa = info.external_fa;

				int zerr = zipOpenNewFileInZip(dst_pkg_zip,
						file.utf8().get_data(),
						&fi,
						NULL,
						0,
						NULL,
						0,
						NULL,
						Z_DEFLATED,
						Z_DEFAULT_COMPRESSION);

				print_line("OPEN ERR: " + itos(zerr));
				zerr = zipWriteInFileInZip(dst_pkg_zip, data.ptr(), data.size());
				print_line("WRITE ERR: " + itos(zerr));
				zipCloseFileInZip(dst_pkg_zip);
			}
		}

		ret = unzGoToNextFile(src_pkg_zip);
	}

	// we're done with our source zip
	unzClose(src_pkg_zip);

	if (!found_binary) {
		ERR_PRINTS("Requested template binary '" + binary_to_use + "' not found. It might be missing from your template archive.");
		err = ERR_FILE_NOT_FOUND;
	}

	if (err == OK) {
		ep.step("Making PKG", 1);

		if (use_dmg()) {
			String pack_path = tmp_app_path_name + "/Contents/Resources/" + pkg_name + ".pck";
			err = save_pack(p_preset, pack_path);

			// see if we can code sign our new package
			String identity = p_preset->get("codesign/identity");
			if (err == OK && identity != "") {
				ep.step("Code signing bundle", 2);

				// the order in which we code sign is important, this is a bit of a shame or we could do this in our loop that extracts the files from our ZIP

				// start with our application
				err = _code_sign(p_preset, tmp_app_path_name + "/Contents/MacOS/" + pkg_name);

				///@TODO we should check the contents of /Contents/Frameworks for frameworks to sign
			}

			if (err == OK && identity != "") {
				// we should probably loop through all resources and sign them?
				err = _code_sign(p_preset, tmp_app_path_name + "/Contents/Resources/icon.icns");
			}

			if (err == OK && identity != "") {
				err = _code_sign(p_preset, pack_path);
			}

			if (err == OK && identity != "") {
				err = _code_sign(p_preset, tmp_app_path_name + "/Contents/Info.plist");
			}

			// and finally create a DMG
			if (err == OK) {
				ep.step("Making DMG", 3);
				err = _create_dmg(p_path, pkg_name, tmp_app_path_name);
			}

			// Clean up temporary .app dir
			OS::get_singleton()->move_to_trash(tmp_app_path_name);
		} else {

			String pack_path = EditorSettings::get_singleton()->get_cache_dir().plus_file(pkg_name + ".pck");
			Error err = save_pack(p_preset, pack_path);

			if (err == OK) {
				zipOpenNewFileInZip(dst_pkg_zip,
						(pkg_name + ".app/Contents/Resources/" + pkg_name + ".pck").utf8().get_data(),
						NULL,
						NULL,
						0,
						NULL,
						0,
						NULL,
						Z_DEFLATED,
						Z_DEFAULT_COMPRESSION);

				FileAccess *pf = FileAccess::open(pack_path, FileAccess::READ);
				if (pf) {
					const int BSIZE = 16384;
					uint8_t buf[BSIZE];

					while (true) {

						int r = pf->get_buffer(buf, BSIZE);
						if (r <= 0)
							break;
						zipWriteInFileInZip(dst_pkg_zip, buf, r);
					}
					zipCloseFileInZip(dst_pkg_zip);
					memdelete(pf);
				} else {
					err = ERR_CANT_OPEN;
				}
			}
		}
	}

	if (dst_pkg_zip) {
		zipClose(dst_pkg_zip, NULL);
	}

	return OK;
}

bool EditorExportPlatformOSX::can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {

	bool valid = true;
	String err;

	if (!exists_export_template("osx.zip", &err)) {
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

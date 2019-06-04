/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor/editor_import_export.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "globals.h"
#include "io/marshalls.h"
#include "io/resource_saver.h"
#include "io/zip_io.h"
#include "os/file_access.h"
#include "os/os.h"
#include "platform/osx/logo.gen.h"
#include "string.h"
#include "version.h"

class EditorExportPlatformOSX : public EditorExportPlatform {

	OBJ_TYPE(EditorExportPlatformOSX, EditorExportPlatform);

	String custom_release_package;
	String custom_debug_package;

	enum BitsMode {
		BITS_FAT,
		BITS_64,
		BITS_32
	};

	int version_code;

	String app_name;
	String info;
	String icon;
	String identifier;
	String short_version;
	String version;
	String signature;
	String copyright;
	String identity;
	String entitlements;
	BitsMode bits_mode;
	bool high_resolution;

	Ref<ImageTexture> logo;

	void _fix_plist(Vector<uint8_t> &plist, const String &p_binary);
	void _make_icon(const Image &p_icon, Vector<uint8_t> &data);

	Error _code_sign(const String &p_path);
	Error _create_dmg(const String &p_dmg_path, const String &p_pkg_name, const String &p_app_path_name);

#ifdef OSX_ENABLED
	bool use_codesign() const { return true; }
	bool use_dmg() const { return true; }
#else
	bool use_codesign() const { return false; }
	bool use_dmg() const { return false; }
#endif

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	virtual String get_name() const { return "Mac OSX"; }
	virtual ImageCompression get_image_compression() const { return IMAGE_COMPRESSION_BC; }
	virtual Ref<Texture> get_logo() const { return logo; }

	virtual bool poll_devices() { return false; }
	virtual int get_device_count() const { return 0; }
	virtual String get_device_name(int p_device) const { return String(); }
	virtual String get_device_info(int p_device) const { return String(); }
	virtual Error run(int p_device, int p_flags = 0);

	virtual bool requires_password(bool p_debug) const { return false; }
	virtual String get_binary_extension() const { return use_dmg() ? "dmg" : "zip"; }
	virtual Error export_project(const String &p_path, bool p_debug, int p_flags = 0);

	virtual bool can_export(String *r_error = NULL) const;

	EditorExportPlatformOSX();
	~EditorExportPlatformOSX();
};

bool EditorExportPlatformOSX::_set(const StringName &p_name, const Variant &p_value) {

	String n = p_name;

	if (n == "custom_package/debug")
		custom_debug_package = p_value;
	else if (n == "custom_package/release")
		custom_release_package = p_value;
	else if (n == "application/name")
		app_name = p_value;
	else if (n == "application/info")
		info = p_value;
	else if (n == "application/icon")
		icon = p_value;
	else if (n == "application/identifier")
		identifier = p_value;
	else if (n == "application/signature")
		signature = p_value;
	else if (n == "application/short_version")
		short_version = p_value;
	else if (n == "application/version")
		version = p_value;
	else if (n == "application/copyright")
		copyright = p_value;
	else if (n == "application/bits_mode")
		bits_mode = BitsMode(int(p_value));
	else if (n == "display/high_res")
		high_resolution = p_value;
	else if (n == "codesign/identity")
		identity = p_value;
	else if (n == "codesign/entitlements")
		entitlements = p_value;
	else
		return false;

	return true;
}

bool EditorExportPlatformOSX::_get(const StringName &p_name, Variant &r_ret) const {

	String n = p_name;

	if (n == "custom_package/debug")
		r_ret = custom_debug_package;
	else if (n == "custom_package/release")
		r_ret = custom_release_package;
	else if (n == "application/name")
		r_ret = app_name;
	else if (n == "application/info")
		r_ret = info;
	else if (n == "application/icon")
		r_ret = icon;
	else if (n == "application/identifier")
		r_ret = identifier;
	else if (n == "application/signature")
		r_ret = signature;
	else if (n == "application/short_version")
		r_ret = short_version;
	else if (n == "application/version")
		r_ret = version;
	else if (n == "application/copyright")
		r_ret = copyright;
	else if (n == "application/bits_mode")
		r_ret = bits_mode;
	else if (n == "display/high_res")
		r_ret = high_resolution;
	else if (n == "codesign/identity")
		r_ret = identity;
	else if (n == "codesign/entitlements")
		r_ret = entitlements;
	else
		return false;

	return true;
}
void EditorExportPlatformOSX::_get_property_list(List<PropertyInfo> *p_list) const {

	p_list->push_back(PropertyInfo(Variant::STRING, "custom_package/debug", PROPERTY_HINT_GLOBAL_FILE, "zip"));
	p_list->push_back(PropertyInfo(Variant::STRING, "custom_package/release", PROPERTY_HINT_GLOBAL_FILE, "zip"));

	p_list->push_back(PropertyInfo(Variant::STRING, "application/name"));
	p_list->push_back(PropertyInfo(Variant::STRING, "application/info"));
	p_list->push_back(PropertyInfo(Variant::STRING, "application/icon", PROPERTY_HINT_FILE, "png"));
	p_list->push_back(PropertyInfo(Variant::STRING, "application/identifier"));
	p_list->push_back(PropertyInfo(Variant::STRING, "application/signature"));
	p_list->push_back(PropertyInfo(Variant::STRING, "application/short_version"));
	p_list->push_back(PropertyInfo(Variant::STRING, "application/version"));
	p_list->push_back(PropertyInfo(Variant::STRING, "application/copyright"));
	p_list->push_back(PropertyInfo(Variant::INT, "application/bits_mode", PROPERTY_HINT_ENUM, "Fat (32 & 64 bits),64 bits,32 bits"));
	p_list->push_back(PropertyInfo(Variant::BOOL, "display/high_res"));

	p_list->push_back(PropertyInfo(Variant::STRING, "codesign/identity"));
	p_list->push_back(PropertyInfo(Variant::STRING, "codesign/entitlements"));
}

void EditorExportPlatformOSX::_make_icon(const Image &p_icon, Vector<uint8_t> &icon) {

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

		Image copy = p_icon;
		copy.convert(Image::FORMAT_RGBA);
		copy.resize(size, size);
		it->create_from_image(copy);
		String path = EditorSettings::get_singleton()->get_settings_path() + "/tmp/icon.png";
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

	icon = data;
}

void EditorExportPlatformOSX::_fix_plist(Vector<uint8_t> &plist, const String &p_binary) {

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
			strnew += lines[i].replace("$info", info) + "\n";
		} else if (lines[i].find("$identifier") != -1) {
			strnew += lines[i].replace("$identifier", identifier) + "\n";
		} else if (lines[i].find("$short_version") != -1) {
			strnew += lines[i].replace("$short_version", short_version) + "\n";
		} else if (lines[i].find("$version") != -1) {
			strnew += lines[i].replace("$version", version) + "\n";
		} else if (lines[i].find("$signature") != -1) {
			strnew += lines[i].replace("$signature", signature) + "\n";
		} else if (lines[i].find("$copyright") != -1) {
			strnew += lines[i].replace("$copyright", copyright) + "\n";
		} else if (lines[i].find("$highres") != -1) {
			strnew += lines[i].replace("$highres", high_resolution ? "<true/>" : "<false/>") + "\n";
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

Error EditorExportPlatformOSX::_code_sign(const String &p_path) {
	List<String> args;

	if (entitlements != "") {
		// this should point to our entitlements.plist file that sandboxes our application, I don't know if this should also be placed in our app bundle
		args.push_back("-entitlements");
		args.push_back(entitlements);
	}
	args.push_back("-s");
	args.push_back(identity);
	args.push_back("-v"); // provide some more feedback
	args.push_back(p_path);

	String str;
	Error err = OS::get_singleton()->execute("/usr/bin/codesign", args, true, NULL, &str, NULL, true);
	ERR_FAIL_COND_V(err != OK, err);

	print_line("codesign: " + str);
	if (str.find("no identity found") != -1) {
		EditorNode::add_io_error("codesign: no identity found");
		return FAILED;
	}

	return OK;
}

Error EditorExportPlatformOSX::_create_dmg(const String &p_dmg_path, const String &p_pkg_name, const String &p_app_path_name) {

	OS::get_singleton()->move_path_to_trash(p_dmg_path);

	List<String> args;

	args.push_back("create");
	args.push_back(p_dmg_path);
	args.push_back("-volname");
	args.push_back(p_pkg_name);
	args.push_back("-fs");
	args.push_back("HFS+");
	args.push_back("-srcfolder");
	args.push_back(p_app_path_name);

	String str;
	Error err = OS::get_singleton()->execute("/usr/bin/hdiutil", args, true, NULL, &str, NULL, true);
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

Error EditorExportPlatformOSX::export_project(const String &p_path, bool p_debug, int p_flags) {

	EditorProgress ep("export", "Exporting for OSX", 104);

	String src_pkg = p_debug ? custom_debug_package : custom_release_package;
	if (src_pkg == "") {
		String err;

		src_pkg = find_export_template("osx.zip", &err);
		if (src_pkg == "") {
			EditorNode::add_io_error(err);
			return ERR_FILE_NOT_FOUND;
		}
	}

	FileAccess *src_f = NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

	ep.step("Creating app", 0);

	unzFile pkg = unzOpen2(src_pkg.utf8().get_data(), &io);
	if (!pkg) {

		EditorNode::add_io_error("Could not find template app to export:\n" + src_pkg);
		return ERR_FILE_NOT_FOUND;
	}

	int ret = unzGoToFirstFile(pkg);

	zlib_filefunc_def io2 = io;
	FileAccess *dst_f = NULL;
	io2.opaque = &dst_f;
	zipFile dpkg = NULL;

	if (!use_dmg()) {
		dpkg = zipOpen2(p_path.utf8().get_data(), APPEND_STATUS_CREATE, NULL, &io2);
		if (!dpkg) {
			unzClose(pkg);
			return ERR_CANT_OPEN;
		}
	}

	String binary_to_use = "godot_osx_" + String(p_debug ? "debug" : "release") + ".";
	binary_to_use += String(bits_mode == BITS_FAT ? "fat" : bits_mode == BITS_64 ? "64" : "32");

	print_line("binary: " + binary_to_use);
	String pkg_name;
	if (app_name != "")
		pkg_name = app_name;
	else if (String(Globals::get_singleton()->get("application/name")) != "")
		pkg_name = String(Globals::get_singleton()->get("application/name"));
	else
		pkg_name = "Unnamed";

	Error err = OK;
	String tmp_app_path_name = "";
	if (use_dmg()) {
		// We're on OSX so we can export to DMG, but first we create our application bundle
		tmp_app_path_name = EditorSettings::get_singleton()->get_settings_path() + "/tmp/" + pkg_name + ".app";
		print_line("Exporting to " + tmp_app_path_name);
		DirAccess *da_tmp_app = DirAccess::create_for_path(tmp_app_path_name);
		if (!da_tmp_app) {
			err = ERR_CANT_CREATE;
		}

		// Create our folder structure or rely on unzip?
		if (err == OK) {
			print_line("Creating " + tmp_app_path_name + "/Contents/MacOS");
			err = da_tmp_app->make_dir_recursive(tmp_app_path_name + "/Contents/MacOS");
		}

		if (err == OK) {
			print_line("Creating " + tmp_app_path_name + "/Contents/Resources");
			err = da_tmp_app->make_dir_recursive(tmp_app_path_name + "/Contents/Resources");
		}
	}

	bool found_binary = false;

	while (ret == UNZ_OK && err == OK) {
		bool is_execute = false;

		//get filename
		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, NULL, 0, NULL, 0);

		String file = fname;

		print_line("READ: " + file);
		Vector<uint8_t> data;
		data.resize(info.uncompressed_size);

		//read
		unzOpenCurrentFile(pkg);
		unzReadCurrentFile(pkg, data.ptr(), data.size());
		unzCloseCurrentFile(pkg);

		//write
		file = file.replace_first("osx_template.app/", "");
		if (file == "Contents/Info.plist") {
			print_line("parse plist");
			_fix_plist(data, pkg_name);
		}

		if (file.begins_with("Contents/MacOS/godot_")) {
			if (file != "Contents/MacOS/" + binary_to_use) {
				ret = unzGoToNextFile(pkg);
				continue; //ignore!
			}
			found_binary = true;
			is_execute = true;
			file = "Contents/MacOS/" + pkg_name;
		}

		if (file == "Contents/Resources/icon.icns") {
			//see if there is an icon
			String iconpath = Globals::get_singleton()->get("application/icon");
			print_line("icon? " + iconpath);
			if (iconpath != "") {
				Image icon;
				icon.load(iconpath);
				if (!icon.empty()) {
					print_line("loaded?");
					_make_icon(icon, data);
				}
			}
		}

		if (data.size() > 0) {
			print_line("ADDING: " + file + " size: " + itos(data.size()));

			if (use_dmg()) {
				// write it into our application bundle
				file = tmp_app_path_name + "/" + file;

				// write the file
				FileAccess *f = FileAccess::open(file, FileAccess::WRITE);
				if (f) {
					f->store_buffer(data.ptr(), data.size());
					f->close();
					if (is_execute) {
						// Chmod with 0755 if the file is executable
						err = f->_chmod(file, 0755);
					}
					memdelete(f);
				} else {
					err = ERR_CANT_CREATE;
				}
			} else {
				zip_fileinfo fi;

				file = pkg_name + ".app/" + file;

				fi.tmz_date.tm_hour = info.tmu_date.tm_hour;
				fi.tmz_date.tm_min = info.tmu_date.tm_min;
				fi.tmz_date.tm_sec = info.tmu_date.tm_sec;
				fi.tmz_date.tm_mon = info.tmu_date.tm_mon;
				fi.tmz_date.tm_mday = info.tmu_date.tm_mday;
				fi.tmz_date.tm_year = info.tmu_date.tm_year;
				fi.dosDate = info.dosDate;
				fi.internal_fa = info.internal_fa;
				fi.external_fa = info.external_fa;

				int zerr = zipOpenNewFileInZip(dpkg,
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
				zerr = zipWriteInFileInZip(dpkg, data.ptr(), data.size());
				print_line("WRITE ERR: " + itos(zerr));
				zipCloseFileInZip(dpkg);
			}
		}

		ret = unzGoToNextFile(pkg);
	}

	if (!found_binary) {
		ERR_PRINTS("Requested template binary '" + binary_to_use + "' not found. It might be missing from your template archive.");
		err = ERR_FILE_NOT_FOUND;
	}

	if (err == OK) {
		ep.step("Making PKG", 1);

		String pack_path;

		if (use_dmg()) {
			pack_path = tmp_app_path_name + "/Contents/Resources/" + pkg_name + ".pck";
		} else {
			pack_path = EditorSettings::get_singleton()->get_settings_path() + "/tmp/data.pck";
		}

		FileAccess *pfs = FileAccess::open(pack_path, FileAccess::WRITE);
		if (pfs) {
			err = save_pack(pfs);
			memdelete(pfs);
		} else {
			err = ERR_CANT_OPEN;
		}

		if (use_dmg()) {
			if (err == OK && use_codesign()) {
				/* see if we can code sign our new package */
				if (err == OK && identity != "") {
					ep.step("Code signing bundle", 2);

					/* the order in which we code sign is important, this is a bit of a shame or we could do this in our loop that extracts the files from our ZIP */

					// start with our application
					err = _code_sign(tmp_app_path_name + "/Contents/MacOS/" + pkg_name);
				}

				///@TODO we should check the contents of /Contents/Frameworks for frameworks to sign

				if (err == OK && identity != "") {
					// we should probably loop through all resources and sign them?
					err = _code_sign(tmp_app_path_name + "/Contents/Resources/icon.icns");
				}

				if (err == OK && identity != "") {
					err = _code_sign(pack_path);
				}

				if (err == OK && identity != "") {
					err = _code_sign(tmp_app_path_name + "/Contents/Info.plist");
				}
			}

			if (err == OK) {
				// and finally create a DMG
				ep.step("Making DMG", 3);
				err = _create_dmg(p_path, pkg_name, tmp_app_path_name);
			}

			// Clean up temporary .app dir
			OS::get_singleton()->move_path_to_trash(tmp_app_path_name);
		} else if (err == OK) {
			//write datapack

			int zerr = zipOpenNewFileInZip(dpkg,
					(pkg_name + ".app/Contents/Resources/data.pck").utf8().get_data(),
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
					zipWriteInFileInZip(dpkg, buf, r);
				}

				zipCloseFileInZip(dpkg);
				memdelete(pf);
			} else {
				err = ERR_CANT_OPEN;
			}
		}
	}

	if (dpkg) {
		zipClose(dpkg, NULL);
	}
	unzClose(pkg);

	return err;
}

Error EditorExportPlatformOSX::run(int p_device, int p_flags) {

	return OK;
}

EditorExportPlatformOSX::EditorExportPlatformOSX() {

	Image img(_osx_logo);
	logo = Ref<ImageTexture>(memnew(ImageTexture));
	logo->create_from_image(img);

	info = "Made with Godot Engine";
	identifier = "org.godotengine.macgame";
	signature = "godotmacgame";
	short_version = "1.0";
	version = "1.0";
	bits_mode = BITS_FAT;
	high_resolution = false;
	identity = "";
	entitlements = "";
}

bool EditorExportPlatformOSX::can_export(String *r_error) const {

	bool valid = true;
	String err;

	if (!exists_export_template("osx.zip")) {
		valid = false;
		err += "No export templates found.\nDownload and install export templates.\n";
	}

	if (custom_debug_package != "" && !FileAccess::exists(custom_debug_package)) {
		valid = false;
		err += "Custom debug package not found.\n";
	}

	if (custom_release_package != "" && !FileAccess::exists(custom_release_package)) {
		valid = false;
		err += "Custom release package not found.\n";
	}

	if (r_error)
		*r_error = err;

	return valid;
}

EditorExportPlatformOSX::~EditorExportPlatformOSX() {
}

void register_osx_exporter() {

	Ref<EditorExportPlatformOSX> exporter = Ref<EditorExportPlatformOSX>(memnew(EditorExportPlatformOSX));
	EditorImportExport::get_singleton()->add_export_platform(exporter);
}

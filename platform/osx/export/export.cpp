/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "global_config.h"
#include "io/marshalls.h"
#include "io/resource_saver.h"
#include "io/zip_io.h"
#include "os/file_access.h"
#include "os/os.h"
#include "platform/osx/logo.gen.h"
#include "string.h"
#include "version.h"

class EditorExportPlatformOSX : public EditorExportPlatform {

	GDCLASS(EditorExportPlatformOSX, EditorExportPlatform);

	int version_code;

	Ref<ImageTexture> logo;

	void _fix_plist(const Ref<EditorExportPreset> &p_preset, Vector<uint8_t> &plist, const String &p_binary);
	void _make_icon(const Ref<Image> &p_icon, Vector<uint8_t> &p_data);

protected:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features);
	virtual void get_export_options(List<ExportOption> *r_options);

public:
	virtual String get_name() const { return "Mac OSX"; }
	virtual Ref<Texture> get_logo() const { return logo; }

	virtual String get_binary_extension() const { return "zip"; }
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const;

	EditorExportPlatformOSX();
	~EditorExportPlatformOSX();
};

void EditorExportPlatformOSX::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) {

	// what does this need to do?
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
	plist.resize(cs.size());
	for (int i = 9; i < cs.size(); i++) {
		plist[i] = cs[i];
	}
}

Error EditorExportPlatformOSX::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {

	String src_pkg;

	EditorProgress ep("export", "Exporting for OSX", 104);

	if (p_debug)
		src_pkg = p_preset->get("custom_package/debug");
	else
		src_pkg = p_preset->get("custom_package/release");

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

	ERR_FAIL_COND_V(!pkg, ERR_CANT_OPEN);
	int ret = unzGoToFirstFile(pkg);

	zlib_filefunc_def io2 = io;
	FileAccess *dst_f = NULL;
	io2.opaque = &dst_f;
	zipFile dpkg = zipOpen2(p_path.utf8().get_data(), APPEND_STATUS_CREATE, NULL, &io2);

	String binary_to_use = "godot_osx_" + String(p_debug ? "debug" : "release") + ".";
	int bits_mode = p_preset->get("application/bits_mode");
	binary_to_use += String(bits_mode == 0 ? "fat" : bits_mode == 1 ? "64" : "32");

	print_line("binary: " + binary_to_use);
	String pkg_name;
	if (p_preset->get("application/name") != "")
		pkg_name = p_preset->get("application/name"); // app_name
	else if (String(GlobalConfig::get_singleton()->get("application/name")) != "")
		pkg_name = String(GlobalConfig::get_singleton()->get("application/name"));
	else
		pkg_name = "Unnamed";

	bool found_binary = false;

	while (ret == UNZ_OK) {

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
			_fix_plist(p_preset, data, pkg_name);
		}

		if (file.begins_with("Contents/MacOS/godot_")) {
			if (file != "Contents/MacOS/" + binary_to_use) {
				ret = unzGoToNextFile(pkg);
				continue; //ignore!
			}
			found_binary = true;
			file = "Contents/MacOS/" + pkg_name;
		}

		if (file == "Contents/Resources/icon.icns") {
			//see if there is an icon
			String iconpath;
			if (p_preset->get("application/icon") != "")
				iconpath = p_preset->get("application/icon");
			else
				iconpath = GlobalConfig::get_singleton()->get("application/icon");
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

		file = pkg_name + ".app/" + file;

		if (data.size() > 0) {
			print_line("ADDING: " + file + " size: " + itos(data.size()));

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

			int err = zipOpenNewFileInZip(dpkg,
					file.utf8().get_data(),
					&fi,
					NULL,
					0,
					NULL,
					0,
					NULL,
					Z_DEFLATED,
					Z_DEFAULT_COMPRESSION);

			print_line("OPEN ERR: " + itos(err));
			err = zipWriteInFileInZip(dpkg, data.ptr(), data.size());
			print_line("WRITE ERR: " + itos(err));
			zipCloseFileInZip(dpkg);
		}

		ret = unzGoToNextFile(pkg);
	}

	if (!found_binary) {
		ERR_PRINTS("Requested template binary '" + binary_to_use + "' not found. It might be missing from your template archive.");
		zipClose(dpkg, NULL);
		unzClose(pkg);
		return ERR_FILE_NOT_FOUND;
	}

	ep.step("Making PKG", 1);

	String pack_path = EditorSettings::get_singleton()->get_settings_path() + "/tmp/data.pck";
	Error err = save_pack(p_preset, pack_path);

	if (err) {
		zipClose(dpkg, NULL);
		unzClose(pkg);
		return err;
	}

	{
		//write datapack

		zipOpenNewFileInZip(dpkg,
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
		ERR_FAIL_COND_V(!pf, ERR_CANT_OPEN);
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
	}

	zipClose(dpkg, NULL);
	unzClose(pkg);

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

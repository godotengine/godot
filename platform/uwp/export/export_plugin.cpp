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

#include "platform/uwp/logo.gen.h"

String EditorExportPlatformUWP::get_name() const {
	return "UWP";
}
String EditorExportPlatformUWP::get_os_name() const {
	return "UWP";
}

List<String> EditorExportPlatformUWP::get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const {
	List<String> list;
	list.push_back("appx");
	return list;
}

Ref<Texture2D> EditorExportPlatformUWP::get_logo() const {
	return logo;
}

void EditorExportPlatformUWP::get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) {
	r_features->push_back("s3tc");
	r_features->push_back("etc");
	switch ((int)p_preset->get("architecture/target")) {
		case EditorExportPlatformUWP::ARM: {
			r_features->push_back("arm");
		} break;
		case EditorExportPlatformUWP::X86: {
			r_features->push_back("32");
		} break;
		case EditorExportPlatformUWP::X64: {
			r_features->push_back("64");
		} break;
	}
}

void EditorExportPlatformUWP::get_export_options(List<ExportOption> *r_options) {
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/debug", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_template/release", PROPERTY_HINT_GLOBAL_FILE, "*.zip"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "architecture/target", PROPERTY_HINT_ENUM, "arm,x86,x64"), 1));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "command_line/extra_args"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/display_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/short_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/unique_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game.Name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/description"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/publisher", PROPERTY_HINT_PLACEHOLDER_TEXT, "CN=CompanyName"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/publisher_display_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Company Name"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "identity/product_guid", PROPERTY_HINT_PLACEHOLDER_TEXT, "00000000-0000-0000-0000-000000000000"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "identity/publisher_guid", PROPERTY_HINT_PLACEHOLDER_TEXT, "00000000-0000-0000-0000-000000000000"), ""));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "signing/certificate", PROPERTY_HINT_GLOBAL_FILE, "*.pfx"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "signing/password"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "signing/algorithm", PROPERTY_HINT_ENUM, "MD5,SHA1,SHA256"), 2));

	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "version/major"), 1));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "version/minor"), 0));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "version/build"), 0));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "version/revision"), 0));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "orientation/landscape"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "orientation/portrait"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "orientation/landscape_flipped"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "orientation/portrait_flipped"), true));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "images/background_color"), "transparent"));
	r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/store_logo", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture2D"), Variant()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/square44x44_logo", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture2D"), Variant()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/square71x71_logo", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture2D"), Variant()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/square150x150_logo", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture2D"), Variant()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/square310x310_logo", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture2D"), Variant()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/wide310x150_logo", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture2D"), Variant()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::OBJECT, "images/splash_screen", PROPERTY_HINT_RESOURCE_TYPE, "StreamTexture2D"), Variant()));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "tiles/show_name_on_square150x150"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "tiles/show_name_on_wide310x150"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "tiles/show_name_on_square310x310"), false));

	// Capabilities
	const char **basic = uwp_capabilities;
	while (*basic) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/" + String(*basic)), false));
		basic++;
	}

	const char **uap = uwp_uap_capabilities;
	while (*uap) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/" + String(*uap)), false));
		uap++;
	}

	const char **device = uwp_device_capabilities;
	while (*device) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "capabilities/" + String(*device)), false));
		device++;
	}
}

bool EditorExportPlatformUWP::can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {
	String err;
	bool valid = false;

	// Look for export templates (first official, and if defined custom templates).

	Platform arch = (Platform)(int)(p_preset->get("architecture/target"));
	String platform_infix;
	switch (arch) {
		case EditorExportPlatformUWP::ARM: {
			platform_infix = "arm";
		} break;
		case EditorExportPlatformUWP::X86: {
			platform_infix = "x86";
		} break;
		case EditorExportPlatformUWP::X64: {
			platform_infix = "x64";
		} break;
	}

	bool dvalid = exists_export_template("uwp_" + platform_infix + "_debug.zip", &err);
	bool rvalid = exists_export_template("uwp_" + platform_infix + "_release.zip", &err);

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

	if (!_valid_resource_name(p_preset->get("package/short_name"))) {
		valid = false;
		err += TTR("Invalid package short name.") + "\n";
	}

	if (!_valid_resource_name(p_preset->get("package/unique_name"))) {
		valid = false;
		err += TTR("Invalid package unique name.") + "\n";
	}

	if (!_valid_resource_name(p_preset->get("package/publisher_display_name"))) {
		valid = false;
		err += TTR("Invalid package publisher display name.") + "\n";
	}

	if (!_valid_guid(p_preset->get("identity/product_guid"))) {
		valid = false;
		err += TTR("Invalid product GUID.") + "\n";
	}

	if (!_valid_guid(p_preset->get("identity/publisher_guid"))) {
		valid = false;
		err += TTR("Invalid publisher GUID.") + "\n";
	}

	if (!_valid_bgcolor(p_preset->get("images/background_color"))) {
		valid = false;
		err += TTR("Invalid background color.") + "\n";
	}

	if (!p_preset->get("images/store_logo").is_zero() && !_valid_image((Object::cast_to<StreamTexture2D>((Object *)p_preset->get("images/store_logo"))), 50, 50)) {
		valid = false;
		err += TTR("Invalid Store Logo image dimensions (should be 50x50).") + "\n";
	}

	if (!p_preset->get("images/square44x44_logo").is_zero() && !_valid_image((Object::cast_to<StreamTexture2D>((Object *)p_preset->get("images/square44x44_logo"))), 44, 44)) {
		valid = false;
		err += TTR("Invalid square 44x44 logo image dimensions (should be 44x44).") + "\n";
	}

	if (!p_preset->get("images/square71x71_logo").is_zero() && !_valid_image((Object::cast_to<StreamTexture2D>((Object *)p_preset->get("images/square71x71_logo"))), 71, 71)) {
		valid = false;
		err += TTR("Invalid square 71x71 logo image dimensions (should be 71x71).") + "\n";
	}

	if (!p_preset->get("images/square150x150_logo").is_zero() && !_valid_image((Object::cast_to<StreamTexture2D>((Object *)p_preset->get("images/square150x150_logo"))), 150, 150)) {
		valid = false;
		err += TTR("Invalid square 150x150 logo image dimensions (should be 150x150).") + "\n";
	}

	if (!p_preset->get("images/square310x310_logo").is_zero() && !_valid_image((Object::cast_to<StreamTexture2D>((Object *)p_preset->get("images/square310x310_logo"))), 310, 310)) {
		valid = false;
		err += TTR("Invalid square 310x310 logo image dimensions (should be 310x310).") + "\n";
	}

	if (!p_preset->get("images/wide310x150_logo").is_zero() && !_valid_image((Object::cast_to<StreamTexture2D>((Object *)p_preset->get("images/wide310x150_logo"))), 310, 150)) {
		valid = false;
		err += TTR("Invalid wide 310x150 logo image dimensions (should be 310x150).") + "\n";
	}

	if (!p_preset->get("images/splash_screen").is_zero() && !_valid_image((Object::cast_to<StreamTexture2D>((Object *)p_preset->get("images/splash_screen"))), 620, 300)) {
		valid = false;
		err += TTR("Invalid splash screen image dimensions (should be 620x300).") + "\n";
	}

	r_error = err;
	return valid;
}

Error EditorExportPlatformUWP::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	ExportNotifier notifier(*this, p_preset, p_debug, p_path, p_flags);

	String src_appx;

	EditorProgress ep("export", "Exporting for UWP", 7, true);

	if (p_debug) {
		src_appx = p_preset->get("custom_template/debug");
	} else {
		src_appx = p_preset->get("custom_template/release");
	}

	src_appx = src_appx.strip_edges();

	Platform arch = (Platform)(int)p_preset->get("architecture/target");

	if (src_appx.is_empty()) {
		String err, infix;
		switch (arch) {
			case ARM: {
				infix = "_arm_";
			} break;
			case X86: {
				infix = "_x86_";
			} break;
			case X64: {
				infix = "_x64_";
			} break;
		}
		if (p_debug) {
			src_appx = find_export_template("uwp" + infix + "debug.zip", &err);
		} else {
			src_appx = find_export_template("uwp" + infix + "release.zip", &err);
		}
		if (src_appx.is_empty()) {
			EditorNode::add_io_error(err);
			return ERR_FILE_NOT_FOUND;
		}
	}

	if (!DirAccess::exists(p_path.get_base_dir())) {
		return ERR_FILE_BAD_PATH;
	}

	Error err = OK;

	FileAccess *fa_pack = FileAccess::open(p_path, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V_MSG(err != OK, ERR_CANT_CREATE, "Cannot create file '" + p_path + "'.");

	AppxPackager packager;
	packager.init(fa_pack);

	FileAccess *src_f = nullptr;
	zlib_filefunc_def io = zipio_create_io_from_file(&src_f);

	if (ep.step("Creating package...", 0)) {
		return ERR_SKIP;
	}

	unzFile pkg = unzOpen2(src_appx.utf8().get_data(), &io);

	if (!pkg) {
		EditorNode::add_io_error("Could not find template appx to export:\n" + src_appx);
		return ERR_FILE_NOT_FOUND;
	}

	int ret = unzGoToFirstFile(pkg);

	if (ep.step("Copying template files...", 1)) {
		return ERR_SKIP;
	}

	EditorNode::progress_add_task("template_files", "Template files", 100);
	packager.set_progress_task("template_files");

	int template_files_amount = 9;
	int template_file_no = 1;

	while (ret == UNZ_OK) {
		// get file name
		unz_file_info info;
		char fname[16834];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16834, nullptr, 0, nullptr, 0);

		String path = fname;

		if (path.ends_with("/")) {
			// Ignore directories
			ret = unzGoToNextFile(pkg);
			continue;
		}

		Vector<uint8_t> data;
		bool do_read = true;

		if (path.begins_with("Assets/")) {
			path = path.replace(".scale-100", "");

			data = _get_image_data(p_preset, path);
			if (data.size() > 0) {
				do_read = false;
			}
		}

		//read
		if (do_read) {
			data.resize(info.uncompressed_size);
			unzOpenCurrentFile(pkg);
			unzReadCurrentFile(pkg, data.ptrw(), data.size());
			unzCloseCurrentFile(pkg);
		}

		if (path == "AppxManifest.xml") {
			data = _fix_manifest(p_preset, data, p_flags & (DEBUG_FLAG_DUMB_CLIENT | DEBUG_FLAG_REMOTE_DEBUG));
		}

		print_line("ADDING: " + path);

		err = packager.add_file(path, data.ptr(), data.size(), template_file_no++, template_files_amount, _should_compress_asset(path, data));
		if (err != OK) {
			return err;
		}

		ret = unzGoToNextFile(pkg);
	}

	EditorNode::progress_end_task("template_files");

	if (ep.step("Creating command line...", 2)) {
		return ERR_SKIP;
	}

	Vector<String> cl = ((String)p_preset->get("command_line/extra_args")).strip_edges().split(" ");
	for (int i = 0; i < cl.size(); i++) {
		if (cl[i].strip_edges().length() == 0) {
			cl.remove_at(i);
			i--;
		}
	}

	if (!(p_flags & DEBUG_FLAG_DUMB_CLIENT)) {
		cl.push_back("--path");
		cl.push_back("game");
	}

	gen_export_flags(cl, p_flags);

	// Command line file
	Vector<uint8_t> clf;

	// Argc
	clf.resize(4);
	encode_uint32(cl.size(), clf.ptrw());

	for (int i = 0; i < cl.size(); i++) {
		CharString txt = cl[i].utf8();
		int base = clf.size();
		clf.resize(base + 4 + txt.length());
		encode_uint32(txt.length(), &clf.write[base]);
		memcpy(&clf.write[base + 4], txt.ptr(), txt.length());
		print_line(itos(i) + " param: " + cl[i]);
	}

	err = packager.add_file("__cl__.cl", clf.ptr(), clf.size(), -1, -1, false);
	if (err != OK) {
		return err;
	}

	if (ep.step("Adding project files...", 3)) {
		return ERR_SKIP;
	}

	EditorNode::progress_add_task("project_files", "Project Files", 100);
	packager.set_progress_task("project_files");

	err = export_project_files(p_preset, save_appx_file, &packager);

	EditorNode::progress_end_task("project_files");

	if (ep.step("Closing package...", 7)) {
		return ERR_SKIP;
	}

	unzClose(pkg);

	packager.finish();

#ifdef WINDOWS_ENABLED
	// Sign with signtool
	String signtool_path = EditorSettings::get_singleton()->get("export/uwp/signtool");
	if (signtool_path.is_empty()) {
		return OK;
	}

	if (!FileAccess::exists(signtool_path)) {
		ERR_PRINT("Could not find signtool executable at " + signtool_path + ", aborting.");
		return ERR_FILE_NOT_FOUND;
	}

	static String algs[] = { "MD5", "SHA1", "SHA256" };

	String cert_path = EditorSettings::get_singleton()->get("export/uwp/debug_certificate");
	String cert_pass = EditorSettings::get_singleton()->get("export/uwp/debug_password");
	int cert_alg = EditorSettings::get_singleton()->get("export/uwp/debug_algorithm");

	if (!p_debug) {
		cert_path = p_preset->get("signing/certificate");
		cert_pass = p_preset->get("signing/password");
		cert_alg = p_preset->get("signing/algorithm");
	}

	if (cert_path.is_empty()) {
		return OK; // Certificate missing, don't try to sign
	}

	if (!FileAccess::exists(cert_path)) {
		ERR_PRINT("Could not find certificate file at " + cert_path + ", aborting.");
		return ERR_FILE_NOT_FOUND;
	}

	if (cert_alg < 0 || cert_alg > 2) {
		ERR_PRINT("Invalid certificate algorithm " + itos(cert_alg) + ", aborting.");
		return ERR_INVALID_DATA;
	}

	List<String> args;
	args.push_back("sign");
	args.push_back("/fd");
	args.push_back(algs[cert_alg]);
	args.push_back("/a");
	args.push_back("/f");
	args.push_back(cert_path);
	args.push_back("/p");
	args.push_back(cert_pass);
	args.push_back(p_path);

	OS::get_singleton()->execute(signtool_path, args);
#endif // WINDOWS_ENABLED

	return OK;
}

void EditorExportPlatformUWP::get_platform_features(List<String> *r_features) {
	r_features->push_back("pc");
	r_features->push_back("uwp");
}

void EditorExportPlatformUWP::resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features) {
}

EditorExportPlatformUWP::EditorExportPlatformUWP() {
	Ref<Image> img = memnew(Image(_uwp_logo));
	logo.instantiate();
	logo->create_from_image(img);
}

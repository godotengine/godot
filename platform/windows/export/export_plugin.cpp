/*************************************************************************/
/*  export_plugin.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/config/project_settings.h"
#include "editor/editor_node.h"

Error EditorExportPlatformWindows::sign_shared_object(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path) {
	if (p_preset->get("codesign/enable")) {
		return _code_sign(p_preset, p_path);
	} else {
		return OK;
	}
}

Error EditorExportPlatformWindows::_export_debug_script(const Ref<EditorExportPreset> &p_preset, const String &p_app_name, const String &p_pkg_name, const String &p_path) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V(f.is_null(), ERR_CANT_CREATE);

	f->store_line("@echo off");
	f->store_line("title \"" + p_app_name + "\"");
	f->store_line("\"%~dp0" + p_pkg_name + "\" \"%*\"");
	f->store_line("pause > nul");

	return OK;
}

Error EditorExportPlatformWindows::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	String pck_path = p_path;
	if (p_preset->get("binary_format/embed_pck")) {
		pck_path = p_path.get_basename() + ".tmp";
	}
	Error err = EditorExportPlatformPC::prepare_template(p_preset, p_debug, pck_path, p_flags);
	if (p_preset->get("application/modify_resources") && err == OK) {
		err = _rcedit_add_data(p_preset, pck_path);
	}
	if (err == OK) {
		err = EditorExportPlatformPC::export_project_data(p_preset, p_debug, pck_path, p_flags);
	}
	if (p_preset->get("codesign/enable") && err == OK) {
		err = _code_sign(p_preset, pck_path);
	}
	if (p_preset->get("binary_format/embed_pck") && err == OK) {
		Ref<DirAccess> tmp_dir = DirAccess::create_for_path(p_path.get_base_dir());
		err = tmp_dir->rename(pck_path, p_path);
	}

	String app_name;
	if (String(ProjectSettings::get_singleton()->get("application/config/name")) != "") {
		app_name = String(ProjectSettings::get_singleton()->get("application/config/name"));
	} else {
		app_name = "Unnamed";
	}
	app_name = OS::get_singleton()->get_safe_dir_name(app_name);

	// Save console script.
	if (err == OK) {
		int con_scr = p_preset->get("debug/export_console_script");
		if ((con_scr == 1 && p_debug) || (con_scr == 2)) {
			String scr_path = p_path.get_basename() + ".cmd";
			err = _export_debug_script(p_preset, app_name, p_path.get_file(), scr_path);
		}
	}

	return err;
}

String EditorExportPlatformWindows::get_template_file_name(const String &p_target, const String &p_arch) const {
	return "windows_" + p_arch + "_" + p_target + ".exe";
}

List<String> EditorExportPlatformWindows::get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const {
	List<String> list;
	list.push_back("exe");
	return list;
}

bool EditorExportPlatformWindows::get_export_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {
	// This option is not supported by "osslsigncode", used on non-Windows host.
	if (!OS::get_singleton()->has_feature("windows") && p_option == "codesign/identity_type") {
		return false;
	}
	return true;
}

void EditorExportPlatformWindows::get_export_options(List<ExportOption> *r_options) {
	EditorExportPlatformPC::get_export_options(r_options);

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "codesign/enable"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "codesign/identity_type", PROPERTY_HINT_ENUM, "Select automatically,Use PKCS12 file (specify *.PFX/*.P12 file),Use certificate store (specify SHA1 hash)"), 0));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "codesign/identity", PROPERTY_HINT_GLOBAL_FILE, "*.pfx,*.p12"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "codesign/password"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "codesign/timestamp"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "codesign/timestamp_server_url"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "codesign/digest_algorithm", PROPERTY_HINT_ENUM, "SHA1,SHA256"), 1));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "codesign/description"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "codesign/custom_options"), PackedStringArray()));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "application/modify_resources"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/icon", PROPERTY_HINT_FILE, "*.ico"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/file_version", PROPERTY_HINT_PLACEHOLDER_TEXT, "1.0.0.0"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/product_version", PROPERTY_HINT_PLACEHOLDER_TEXT, "1.0.0.0"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/company_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Company Name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/product_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/file_description"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/copyright"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/trademarks"), ""));
}

Error EditorExportPlatformWindows::_rcedit_add_data(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
	String rcedit_path = EditorSettings::get_singleton()->get("export/windows/rcedit");

	if (rcedit_path.is_empty()) {
		WARN_PRINT("The rcedit tool is not configured in the Editor Settings (Export > Windows > Rcedit). No custom icon or app information data will be embedded in the exported executable.");
		return ERR_FILE_NOT_FOUND;
	}

	if (!FileAccess::exists(rcedit_path)) {
		ERR_PRINT("Could not find rcedit executable at " + rcedit_path + ", no icon or app information data will be included.");
		return ERR_FILE_NOT_FOUND;
	}
	if (rcedit_path.is_empty()) {
		rcedit_path = "rcedit"; // try to run signtool from PATH
	}

#ifndef WINDOWS_ENABLED
	// On non-Windows we need WINE to run rcedit
	String wine_path = EditorSettings::get_singleton()->get("export/windows/wine");

	if (!wine_path.is_empty() && !FileAccess::exists(wine_path)) {
		ERR_PRINT("Could not find wine executable at " + wine_path + ", no icon or app information data will be included.");
		return ERR_FILE_NOT_FOUND;
	}

	if (wine_path.is_empty()) {
		wine_path = "wine"; // try to run wine from PATH
	}
#endif

	String icon_path = ProjectSettings::get_singleton()->globalize_path(p_preset->get("application/icon"));
	String file_verion = p_preset->get("application/file_version");
	String product_version = p_preset->get("application/product_version");
	String company_name = p_preset->get("application/company_name");
	String product_name = p_preset->get("application/product_name");
	String file_description = p_preset->get("application/file_description");
	String copyright = p_preset->get("application/copyright");
	String trademarks = p_preset->get("application/trademarks");
	String comments = p_preset->get("application/comments");

	List<String> args;
	args.push_back(p_path);
	if (!icon_path.is_empty()) {
		args.push_back("--set-icon");
		args.push_back(icon_path);
	}
	if (!file_verion.is_empty()) {
		args.push_back("--set-file-version");
		args.push_back(file_verion);
	}
	if (!product_version.is_empty()) {
		args.push_back("--set-product-version");
		args.push_back(product_version);
	}
	if (!company_name.is_empty()) {
		args.push_back("--set-version-string");
		args.push_back("CompanyName");
		args.push_back(company_name);
	}
	if (!product_name.is_empty()) {
		args.push_back("--set-version-string");
		args.push_back("ProductName");
		args.push_back(product_name);
	}
	if (!file_description.is_empty()) {
		args.push_back("--set-version-string");
		args.push_back("FileDescription");
		args.push_back(file_description);
	}
	if (!copyright.is_empty()) {
		args.push_back("--set-version-string");
		args.push_back("LegalCopyright");
		args.push_back(copyright);
	}
	if (!trademarks.is_empty()) {
		args.push_back("--set-version-string");
		args.push_back("LegalTrademarks");
		args.push_back(trademarks);
	}

#ifndef WINDOWS_ENABLED
	// On non-Windows we need WINE to run rcedit
	args.push_front(rcedit_path);
	rcedit_path = wine_path;
#endif

	String str;
	Error err = OS::get_singleton()->execute(rcedit_path, args, &str, nullptr, true);
	ERR_FAIL_COND_V(err != OK, err);
	print_line("rcedit (" + p_path + "): " + str);

	if (str.find("Fatal error") != -1) {
		return FAILED;
	}

	return OK;
}

Error EditorExportPlatformWindows::_code_sign(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
	List<String> args;

#ifdef WINDOWS_ENABLED
	String signtool_path = EditorSettings::get_singleton()->get("export/windows/signtool");
	if (!signtool_path.is_empty() && !FileAccess::exists(signtool_path)) {
		ERR_PRINT("Could not find signtool executable at " + signtool_path + ", aborting.");
		return ERR_FILE_NOT_FOUND;
	}
	if (signtool_path.is_empty()) {
		signtool_path = "signtool"; // try to run signtool from PATH
	}
#else
	String signtool_path = EditorSettings::get_singleton()->get("export/windows/osslsigncode");
	if (!signtool_path.is_empty() && !FileAccess::exists(signtool_path)) {
		ERR_PRINT("Could not find osslsigncode executable at " + signtool_path + ", aborting.");
		return ERR_FILE_NOT_FOUND;
	}
	if (signtool_path.is_empty()) {
		signtool_path = "osslsigncode"; // try to run signtool from PATH
	}
#endif

	args.push_back("sign");

	//identity
#ifdef WINDOWS_ENABLED
	int id_type = p_preset->get("codesign/identity_type");
	if (id_type == 0) { //auto select
		args.push_back("/a");
	} else if (id_type == 1) { //pkcs12
		if (p_preset->get("codesign/identity") != "") {
			args.push_back("/f");
			args.push_back(p_preset->get("codesign/identity"));
		} else {
			EditorNode::add_io_error("codesign: no identity found");
			return FAILED;
		}
	} else if (id_type == 2) { //Windows certificate store
		if (p_preset->get("codesign/identity") != "") {
			args.push_back("/sha1");
			args.push_back(p_preset->get("codesign/identity"));
		} else {
			EditorNode::add_io_error("codesign: no identity found");
			return FAILED;
		}
	} else {
		EditorNode::add_io_error("codesign: invalid identity type");
		return FAILED;
	}
#else
	if (p_preset->get("codesign/identity") != "") {
		args.push_back("-pkcs12");
		args.push_back(p_preset->get("codesign/identity"));
	} else {
		EditorNode::add_io_error("codesign: no identity found");
		return FAILED;
	}
#endif

	//password
	if (p_preset->get("codesign/password") != "") {
#ifdef WINDOWS_ENABLED
		args.push_back("/p");
#else
		args.push_back("-pass");
#endif
		args.push_back(p_preset->get("codesign/password"));
	}

	//timestamp
	if (p_preset->get("codesign/timestamp")) {
		if (p_preset->get("codesign/timestamp_server") != "") {
#ifdef WINDOWS_ENABLED
			args.push_back("/tr");
			args.push_back(p_preset->get("codesign/timestamp_server_url"));
			args.push_back("/td");
			if ((int)p_preset->get("codesign/digest_algorithm") == 0) {
				args.push_back("sha1");
			} else {
				args.push_back("sha256");
			}
#else
			args.push_back("-ts");
			args.push_back(p_preset->get("codesign/timestamp_server_url"));
#endif
		} else {
			EditorNode::add_io_error("codesign: invalid timestamp server");
			return FAILED;
		}
	}

	//digest
#ifdef WINDOWS_ENABLED
	args.push_back("/fd");
#else
	args.push_back("-h");
#endif
	if ((int)p_preset->get("codesign/digest_algorithm") == 0) {
		args.push_back("sha1");
	} else {
		args.push_back("sha256");
	}

	//description
	if (p_preset->get("codesign/description") != "") {
#ifdef WINDOWS_ENABLED
		args.push_back("/d");
#else
		args.push_back("-n");
#endif
		args.push_back(p_preset->get("codesign/description"));
	}

	//user options
	PackedStringArray user_args = p_preset->get("codesign/custom_options");
	for (int i = 0; i < user_args.size(); i++) {
		String user_arg = user_args[i].strip_edges();
		if (!user_arg.is_empty()) {
			args.push_back(user_arg);
		}
	}

#ifndef WINDOWS_ENABLED
	args.push_back("-in");
#endif
	args.push_back(p_path);
#ifndef WINDOWS_ENABLED
	args.push_back("-out");
	args.push_back(p_path + "_signed");
#endif

	String str;
	Error err = OS::get_singleton()->execute(signtool_path, args, &str, nullptr, true);
	ERR_FAIL_COND_V(err != OK, err);

	print_line("codesign (" + p_path + "): " + str);
#ifndef WINDOWS_ENABLED
	if (str.find("SignTool Error") != -1) {
#else
	if (str.find("Failed") != -1) {
#endif
		return FAILED;
	}

#ifndef WINDOWS_ENABLED
	Ref<DirAccess> tmp_dir = DirAccess::create_for_path(p_path.get_base_dir());

	err = tmp_dir->remove(p_path);
	ERR_FAIL_COND_V(err != OK, err);

	err = tmp_dir->rename(p_path + "_signed", p_path);
	ERR_FAIL_COND_V(err != OK, err);
#endif

	return OK;
}

bool EditorExportPlatformWindows::can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {
	String err = "";
	bool valid = EditorExportPlatformPC::can_export(p_preset, err, r_missing_templates);

	String rcedit_path = EditorSettings::get_singleton()->get("export/windows/rcedit");
	if (rcedit_path.is_empty()) {
		err += TTR("The rcedit tool must be configured in the Editor Settings (Export > Windows > Rcedit) to change the icon or app information data.") + "\n";
	}

	String icon_path = ProjectSettings::get_singleton()->globalize_path(p_preset->get("application/icon"));
	if (!icon_path.is_empty() && !FileAccess::exists(icon_path)) {
		err += TTR("Invalid icon path:") + " " + icon_path + "\n";
	}

	// Only non-negative integers can exist in the version string.

	String file_version = p_preset->get("application/file_version");
	if (!file_version.is_empty()) {
		PackedStringArray version_array = file_version.split(".", false);
		if (version_array.size() != 4 || !version_array[0].is_valid_int() ||
				!version_array[1].is_valid_int() || !version_array[2].is_valid_int() ||
				!version_array[3].is_valid_int() || file_version.find("-") > -1) {
			err += TTR("Invalid file version:") + " " + file_version + "\n";
		}
	}

	String product_version = p_preset->get("application/product_version");
	if (!product_version.is_empty()) {
		PackedStringArray version_array = product_version.split(".", false);
		if (version_array.size() != 4 || !version_array[0].is_valid_int() ||
				!version_array[1].is_valid_int() || !version_array[2].is_valid_int() ||
				!version_array[3].is_valid_int() || product_version.find("-") > -1) {
			err += TTR("Invalid product version:") + " " + product_version + "\n";
		}
	}

	if (!err.is_empty()) {
		r_error = err;
	}

	return valid;
}

Error EditorExportPlatformWindows::fixup_embedded_pck(const String &p_path, int64_t p_embedded_start, int64_t p_embedded_size) const {
	// Patch the header of the "pck" section in the PE file so that it corresponds to the embedded data

	if (p_embedded_size + p_embedded_start >= 0x100000000) { // Check for total executable size
		ERR_FAIL_V_MSG(ERR_INVALID_DATA, "Windows executables cannot be >= 4 GiB.");
	}

	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ_WRITE);
	if (f.is_null()) {
		return ERR_CANT_OPEN;
	}

	// Jump to the PE header and check the magic number
	{
		f->seek(0x3c);
		uint32_t pe_pos = f->get_32();

		f->seek(pe_pos);
		uint32_t magic = f->get_32();
		if (magic != 0x00004550) {
			return ERR_FILE_CORRUPT;
		}
	}

	// Process header

	int num_sections;
	{
		int64_t header_pos = f->get_position();

		f->seek(header_pos + 2);
		num_sections = f->get_16();
		f->seek(header_pos + 16);
		uint16_t opt_header_size = f->get_16();

		// Skip rest of header + optional header to go to the section headers
		f->seek(f->get_position() + 2 + opt_header_size);
	}

	// Search for the "pck" section

	int64_t section_table_pos = f->get_position();

	bool found = false;
	for (int i = 0; i < num_sections; ++i) {
		int64_t section_header_pos = section_table_pos + i * 40;
		f->seek(section_header_pos);

		uint8_t section_name[9];
		f->get_buffer(section_name, 8);
		section_name[8] = '\0';

		if (strcmp((char *)section_name, "pck") == 0) {
			// "pck" section found, let's patch!

			// Set virtual size to a little to avoid it taking memory (zero would give issues)
			f->seek(section_header_pos + 8);
			f->store_32(8);

			f->seek(section_header_pos + 16);
			f->store_32(p_embedded_size);
			f->seek(section_header_pos + 20);
			f->store_32(p_embedded_start);

			found = true;
			break;
		}
	}

	return found ? OK : ERR_FILE_CORRUPT;
}

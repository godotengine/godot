/*************************************************************************/
/*  export.cpp                                                           */
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

#include "core/os/file_access.h"
#include "core/os/os.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "platform/windows/logo.gen.h"

static Error fixup_embedded_pck(const String &p_path, int64_t p_embedded_start, int64_t p_embedded_size);

class EditorExportPlatformWindows : public EditorExportPlatformPC {
	void _rcedit_add_data(const Ref<EditorExportPreset> &p_preset, const String &p_path);
	Error _code_sign(const Ref<EditorExportPreset> &p_preset, const String &p_path);

public:
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);
	virtual Error sign_shared_object(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path);
	virtual void get_export_options(List<ExportOption> *r_options);
};

Error EditorExportPlatformWindows::sign_shared_object(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path) {
	if (p_preset->get("codesign/enable")) {
		return _code_sign(p_preset, p_path);
	} else {
		return OK;
	}
}

Error EditorExportPlatformWindows::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	Error err = EditorExportPlatformPC::export_project(p_preset, p_debug, p_path, p_flags);

	if (err != OK) {
		return err;
	}

	_rcedit_add_data(p_preset, p_path);

	if (p_preset->get("codesign/enable") && err == OK) {
		err = _code_sign(p_preset, p_path);
	}

	return err;
}

void EditorExportPlatformWindows::get_export_options(List<ExportOption> *r_options) {
	EditorExportPlatformPC::get_export_options(r_options);

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "codesign/enable"), false));
#ifdef WINDOWS_ENABLED
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "codesign/identity_type", PROPERTY_HINT_ENUM, "Select automatically,Use PKCS12 file (specify *.PFX/*.P12 file),Use certificate store (specify SHA1 hash)"), 0));
#endif
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "codesign/identity", PROPERTY_HINT_GLOBAL_FILE, "*.pfx,*.p12"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "codesign/password"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "codesign/timestamp"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "codesign/timestamp_server_url"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "codesign/digest_algorithm", PROPERTY_HINT_ENUM, "SHA1,SHA256"), 1));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "codesign/description"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::POOL_STRING_ARRAY, "codesign/custom_options"), PoolStringArray()));

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/icon", PROPERTY_HINT_FILE, "*.ico"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/file_version", PROPERTY_HINT_PLACEHOLDER_TEXT, "1.0.0"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/product_version", PROPERTY_HINT_PLACEHOLDER_TEXT, "1.0.0"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/company_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Company Name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/product_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/file_description"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/copyright"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/trademarks"), ""));
}

void EditorExportPlatformWindows::_rcedit_add_data(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
	String rcedit_path = EditorSettings::get_singleton()->get("export/windows/rcedit");

	if (rcedit_path == String()) {
		return;
	}

	if (!FileAccess::exists(rcedit_path)) {
		ERR_PRINT("Could not find rcedit executable at " + rcedit_path + ", no icon or app information data will be included.");
		return;
	}

#ifndef WINDOWS_ENABLED
	// On non-Windows we need WINE to run rcedit
	String wine_path = EditorSettings::get_singleton()->get("export/windows/wine");

	if (wine_path != String() && !FileAccess::exists(wine_path)) {
		ERR_PRINT("Could not find wine executable at " + wine_path + ", no icon or app information data will be included.");
		return;
	}

	if (wine_path == String()) {
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
	if (icon_path != String()) {
		args.push_back("--set-icon");
		args.push_back(icon_path);
	}
	if (file_verion != String()) {
		args.push_back("--set-file-version");
		args.push_back(file_verion);
	}
	if (product_version != String()) {
		args.push_back("--set-product-version");
		args.push_back(product_version);
	}
	if (company_name != String()) {
		args.push_back("--set-version-string");
		args.push_back("CompanyName");
		args.push_back(company_name);
	}
	if (product_name != String()) {
		args.push_back("--set-version-string");
		args.push_back("ProductName");
		args.push_back(product_name);
	}
	if (file_description != String()) {
		args.push_back("--set-version-string");
		args.push_back("FileDescription");
		args.push_back(file_description);
	}
	if (copyright != String()) {
		args.push_back("--set-version-string");
		args.push_back("LegalCopyright");
		args.push_back(copyright);
	}
	if (trademarks != String()) {
		args.push_back("--set-version-string");
		args.push_back("LegalTrademarks");
		args.push_back(trademarks);
	}

#ifdef WINDOWS_ENABLED
	OS::get_singleton()->execute(rcedit_path, args, true);
#else
	// On non-Windows we need WINE to run rcedit
	args.push_front(rcedit_path);
	OS::get_singleton()->execute(wine_path, args, true);
#endif
}

Error EditorExportPlatformWindows::_code_sign(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
	List<String> args;

#ifdef WINDOWS_ENABLED
	String signtool_path = EditorSettings::get_singleton()->get("export/windows/signtool");
	if (signtool_path != String() && !FileAccess::exists(signtool_path)) {
		ERR_PRINT("Could not find signtool executable at " + signtool_path + ", aborting.");
		return ERR_FILE_NOT_FOUND;
	}
	if (signtool_path == String()) {
		signtool_path = "signtool"; // try to run signtool from PATH
	}
#else
	String signtool_path = EditorSettings::get_singleton()->get("export/windows/osslsigncode");
	if (signtool_path != String() && !FileAccess::exists(signtool_path)) {
		ERR_PRINT("Could not find osslsigncode executable at " + signtool_path + ", aborting.");
		return ERR_FILE_NOT_FOUND;
	}
	if (signtool_path == String()) {
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
	PoolStringArray user_args = p_preset->get("codesign/custom_options");
	for (int i = 0; i < user_args.size(); i++) {
		String user_arg = user_args[i].strip_edges();
		if (!user_arg.empty()) {
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
	Error err = OS::get_singleton()->execute(signtool_path, args, true, nullptr, &str, nullptr, true);
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
	DirAccessRef tmp_dir = DirAccess::create_for_path(p_path.get_base_dir());

	err = tmp_dir->remove(p_path);
	ERR_FAIL_COND_V(err != OK, err);

	err = tmp_dir->rename(p_path + "_signed", p_path);
	ERR_FAIL_COND_V(err != OK, err);
#endif

	return OK;
}

void register_windows_exporter() {
	EDITOR_DEF("export/windows/rcedit", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/windows/rcedit", PROPERTY_HINT_GLOBAL_FILE, "*.exe"));
#ifdef WINDOWS_ENABLED
	EDITOR_DEF("export/windows/signtool", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/windows/signtool", PROPERTY_HINT_GLOBAL_FILE, "*.exe"));
#else
	EDITOR_DEF("export/windows/osslsigncode", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/windows/osslsigncode", PROPERTY_HINT_GLOBAL_FILE));
	// On non-Windows we need WINE to run rcedit
	EDITOR_DEF("export/windows/wine", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/windows/wine", PROPERTY_HINT_GLOBAL_FILE));
#endif

	Ref<EditorExportPlatformWindows> platform;
	platform.instance();

	Ref<Image> img = memnew(Image(_windows_logo));
	Ref<ImageTexture> logo;
	logo.instance();
	logo->create_from_image(img);
	platform->set_logo(logo);
	platform->set_name("Windows Desktop");
	platform->set_extension("exe");
	platform->set_release_32("windows_32_release.exe");
	platform->set_debug_32("windows_32_debug.exe");
	platform->set_release_64("windows_64_release.exe");
	platform->set_debug_64("windows_64_debug.exe");
	platform->set_os_name("Windows");
	platform->set_fixup_embedded_pck_func(&fixup_embedded_pck);

	EditorExport::get_singleton()->add_export_platform(platform);
}

static Error fixup_embedded_pck(const String &p_path, int64_t p_embedded_start, int64_t p_embedded_size) {
	// Patch the header of the "pck" section in the PE file so that it corresponds to the embedded data

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ_WRITE);
	if (!f) {
		return ERR_CANT_OPEN;
	}

	// Jump to the PE header and check the magic number
	{
		f->seek(0x3c);
		uint32_t pe_pos = f->get_32();

		f->seek(pe_pos);
		uint32_t magic = f->get_32();
		if (magic != 0x00004550) {
			f->close();
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

	f->close();

	return found ? OK : ERR_FILE_CORRUPT;
}

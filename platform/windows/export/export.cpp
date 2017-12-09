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
#include "editor/editor_export.h"
#include "editor/editor_settings.h"
#include "os/file_access.h"
#include "os/os.h"
#include "platform/windows/logo.gen.h"

class EditorExportPlatformWindows : public EditorExportPlatformPC {

public:
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);
	virtual void get_export_options(List<ExportOption> *r_options);
};

Error EditorExportPlatformWindows::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	Error err = EditorExportPlatformPC::export_project(p_preset, p_debug, p_path, p_flags);

	if (err != OK) {
		return err;
	}

	String rcedit_path = EditorSettings::get_singleton()->get("export/windows/rcedit");

	if (rcedit_path == String()) {
		return OK;
	}

	if (!FileAccess::exists(rcedit_path)) {
		ERR_PRINTS("Could not find rcedit executable at " + rcedit_path + ", aborting.");
		return ERR_FILE_NOT_FOUND;
	}

#ifndef WINDOWS_ENABLED
	// On non-Windows we need WINE to run rcedit
	String wine_path = EditorSettings::get_singleton()->get("export/windows/wine");

	if (wine_path != String() && !FileAccess::exists(wine_path)) {
		ERR_PRINTS("Could not find wine executable at " + wine_path + ", aborting.");
		return ERR_FILE_NOT_FOUND;
	}

	if (wine_path == String()) {
		wine_path = "wine"; // try to run wine from PATH
	}
#endif

	String icon_path = p_preset->get("application/icon");
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

	return OK;
}

void EditorExportPlatformWindows::get_export_options(List<ExportOption> *r_options) {
	EditorExportPlatformPC::get_export_options(r_options);

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/icon", PROPERTY_HINT_GLOBAL_FILE, "*.ico"), String()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/file_version"), String()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/product_version"), String()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/company_name"), String()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/product_name"), String()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/file_description"), String()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/copyright"), String()));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/trademarks"), String()));
}

void register_windows_exporter() {

	EDITOR_DEF("export/windows/rcedit", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/windows/rcedit", PROPERTY_HINT_GLOBAL_FILE, "*.exe"));
#ifndef WINDOWS_ENABLED
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

	EditorExport::get_singleton()->add_export_platform(platform);
}

/**************************************************************************/
/*  export.cpp                                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "export.h"

#include "template_modifier.h"

#include "core/io/image_loader.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "platform/windows/logo.gen.h"

class EditorExportPlatformWindows : public EditorExportPlatformPC {
	Error _process_icon(const Ref<EditorExportPreset> &p_preset, const String &p_src_path, const String &p_dst_path);
	Error _add_data(const Ref<EditorExportPreset> &p_preset, const String &p_path);
	Error _code_sign(const Ref<EditorExportPreset> &p_preset, const String &p_path);

public:
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);
	virtual Error sign_shared_object(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path);
	virtual Error modify_template(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags);
	virtual Error fixup_embedded_pck(const String &p_path, int64_t p_embedded_start, int64_t p_embedded_size);
	virtual void get_export_options(List<ExportOption> *r_options);
	virtual bool get_option_visibility(const EditorExportPreset *p_preset, const String &p_option, const Map<StringName, Variant> &p_options) const;
	virtual bool has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const;
	virtual bool has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error) const;
};

Error EditorExportPlatformWindows::_process_icon(const Ref<EditorExportPreset> &p_preset, const String &p_src_path, const String &p_dst_path) {
	static const uint8_t icon_size[] = { 16, 32, 48, 64, 128, 0 /*256*/ };

	struct IconData {
		PoolVector<uint8_t> data;
		uint8_t pal_colors = 0;
		uint16_t planes = 0;
		uint16_t bpp = 32;
	};

	HashMap<uint8_t, IconData> images;
	Error err;

	if (p_src_path.get_extension() == "ico") {
		FileAccess *f = FileAccess::open(p_src_path, FileAccess::READ, &err);
		if (err != OK) {
			return err;
		}

		// Read ICONDIR.
		f->get_16(); // Reserved.
		uint16_t icon_type = f->get_16(); // Image type: 1 - ICO.
		uint16_t icon_count = f->get_16(); // Number of images.
		if (icon_type != 1) {
			f->close();
			memdelete(f);

			ERR_FAIL_V(ERR_CANT_OPEN);
		}

		for (uint16_t i = 0; i < icon_count; i++) {
			// Read ICONDIRENTRY.
			uint16_t w = f->get_8(); // Width in pixels.
			uint16_t h = f->get_8(); // Height in pixels.
			uint8_t pal_colors = f->get_8(); // Number of colors in the palette (0 - no palette).
			f->get_8(); // Reserved.
			uint16_t planes = f->get_16(); // Number of color planes.
			uint16_t bpp = f->get_16(); // Bits per pixel.
			uint32_t img_size = f->get_32(); // Image data size in bytes.
			uint32_t img_offset = f->get_32(); // Image data offset.
			if (w != h) {
				continue;
			}

			// Read image data.
			uint64_t prev_offset = f->get_position();
			images[w].pal_colors = pal_colors;
			images[w].planes = planes;
			images[w].bpp = bpp;
			images[w].data.resize(img_size);
			PoolVector<uint8_t>::Write wr = images[w].data.write();
			f->seek(img_offset);
			f->get_buffer(wr.ptr(), img_size);
			f->seek(prev_offset);
		}
		f->close();
		memdelete(f);
	} else {
		Ref<Image> src_image;
		src_image.instance();
		err = ImageLoader::load_image(p_src_path, src_image.ptr());
		ERR_FAIL_COND_V(err != OK || src_image->empty(), ERR_CANT_OPEN);
		for (size_t i = 0; i < sizeof(icon_size) / sizeof(icon_size[0]); ++i) {
			int size = (icon_size[i] == 0) ? 256 : icon_size[i];

			Ref<Image> res_image = src_image->duplicate();
			ERR_FAIL_COND_V(res_image.is_null() || res_image->empty(), ERR_CANT_OPEN);
			res_image->resize(size, size, (Image::Interpolation)(p_preset->get("application/icon_interpolation").operator int()));
			images[icon_size[i]].data = res_image->save_png_to_buffer();
		}
	}

	uint16_t valid_icon_count = 0;
	for (size_t i = 0; i < sizeof(icon_size) / sizeof(icon_size[0]); ++i) {
		if (images.has(icon_size[i])) {
			valid_icon_count++;
		} else {
			int size = (icon_size[i] == 0) ? 256 : icon_size[i];
			add_message(EXPORT_MESSAGE_WARNING, TTR("Resources Modification"), vformat(TTR("Icon size \"%d\" is missing."), size));
		}
	}
	ERR_FAIL_COND_V(valid_icon_count == 0, ERR_CANT_OPEN);

	FileAccess *fw = FileAccess::open(p_dst_path, FileAccess::WRITE, &err);
	if (err != OK) {
		return err;
	}

	// Write ICONDIR.
	fw->store_16(0); // Reserved.
	fw->store_16(1); // Image type: 1 - ICO.
	fw->store_16(valid_icon_count); // Number of images.

	// Write ICONDIRENTRY.
	uint32_t img_offset = 6 + 16 * valid_icon_count;
	for (size_t i = 0; i < sizeof(icon_size) / sizeof(icon_size[0]); ++i) {
		if (images.has(icon_size[i])) {
			const IconData &di = images[icon_size[i]];
			fw->store_8(icon_size[i]); // Width in pixels.
			fw->store_8(icon_size[i]); // Height in pixels.
			fw->store_8(di.pal_colors); // Number of colors in the palette (0 - no palette).
			fw->store_8(0); // Reserved.
			fw->store_16(di.planes); // Number of color planes.
			fw->store_16(di.bpp); // Bits per pixel.
			fw->store_32(di.data.size()); // Image data size in bytes.
			fw->store_32(img_offset); // Image data offset.

			img_offset += di.data.size();
		}
	}

	// Write image data.
	for (size_t i = 0; i < sizeof(icon_size) / sizeof(icon_size[0]); ++i) {
		if (images.has(icon_size[i])) {
			const IconData &di = images[icon_size[i]];
			PoolVector<uint8_t>::Read r = di.data.read();
			fw->store_buffer(r.ptr(), di.data.size());
		}
	}
	fw->close();
	memdelete(fw);

	return OK;
}

Error EditorExportPlatformWindows::sign_shared_object(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path) {
	if (p_preset->get("codesign/enable")) {
		return _code_sign(p_preset, p_path);
	} else {
		return OK;
	}
}

Error EditorExportPlatformWindows::modify_template(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	if (p_preset->get("application/modify_resources")) {
		_add_data(p_preset, p_path);
	}
	return OK;
}

Error EditorExportPlatformWindows::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	String pck_path = p_path;
	if (p_preset->get("binary_format/embed_pck")) {
		pck_path = p_path.get_basename() + ".tmp";
	}

	Error err = EditorExportPlatformPC::export_project(p_preset, p_debug, pck_path, p_flags);
	if (p_preset->get("codesign/enable") && err == OK) {
		_code_sign(p_preset, pck_path);
	}

	if (p_preset->get("binary_format/embed_pck") && err == OK) {
		DirAccessRef tmp_dir = DirAccess::create_for_path(p_path.get_base_dir());
		err = tmp_dir->rename(pck_path, p_path);
		if (err != OK) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("PCK Embedding"), vformat(TTR("Failed to rename temporary file \"%s\"."), pck_path));
		}
	}

	return err;
}

bool EditorExportPlatformWindows::get_option_visibility(const EditorExportPreset *p_preset, const String &p_option, const Map<StringName, Variant> &p_options) const {
	// This option is not supported by "osslsigncode", used on non-Windows host.
	if (!OS::get_singleton()->has_feature("Windows") && p_option == "codesign/identity_type") {
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
	r_options->push_back(ExportOption(PropertyInfo(Variant::POOL_STRING_ARRAY, "codesign/custom_options"), PoolStringArray()));

	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "application/modify_resources"), true));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/icon", PROPERTY_HINT_FILE, "*.ico,*.png,*.webp,*.svg"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "application/icon_interpolation", PROPERTY_HINT_ENUM, "Nearest neighbor,Bilinear,Cubic,Trilinear,Lanczos"), 4));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/file_version", PROPERTY_HINT_PLACEHOLDER_TEXT, "1.0.0.0"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/product_version", PROPERTY_HINT_PLACEHOLDER_TEXT, "1.0.0.0"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/company_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Company Name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/product_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/file_description"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/copyright"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/trademarks"), ""));
}

Error EditorExportPlatformWindows::_add_data(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
	String icon_path = ProjectSettings::get_singleton()->globalize_path(p_preset->get("application/icon"));
	String tmp_icon_path = EditorSettings::get_singleton()->get_cache_dir().plus_file("_tmp.ico");
	if (!icon_path.empty()) {
		if (_process_icon(p_preset, icon_path, tmp_icon_path) != OK) {
			add_message(EXPORT_MESSAGE_WARNING, TTR("Resources Modification"), vformat(TTR("Invalid icon file \"%s\"."), icon_path));
			icon_path = String();
		}
	}

	TemplateModifier::modify(p_preset, p_path, tmp_icon_path);

	return OK;
}

Error EditorExportPlatformWindows::_code_sign(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
	List<String> args;

#ifdef WINDOWS_ENABLED
	String signtool_path = EditorSettings::get_singleton()->get("export/windows/signtool");
	if (signtool_path != String() && !FileAccess::exists(signtool_path)) {
		add_message(EXPORT_MESSAGE_WARNING, TTR("Code Signing"), vformat(TTR("Could not find signtool executable at \"%s\"."), signtool_path));
		return ERR_FILE_NOT_FOUND;
	}
	if (signtool_path == String()) {
		signtool_path = "signtool"; // try to run signtool from PATH
	}
#else
	String signtool_path = EditorSettings::get_singleton()->get("export/windows/osslsigncode");
	if (signtool_path != String() && !FileAccess::exists(signtool_path)) {
		add_message(EXPORT_MESSAGE_WARNING, TTR("Code Signing"), vformat(TTR("Could not find osslsigncode executable at \"%s\"."), signtool_path));
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
			add_message(EXPORT_MESSAGE_WARNING, TTR("Code Signing"), TTR("No identity found."));
			return FAILED;
		}
	} else if (id_type == 2) { //Windows certificate store
		if (p_preset->get("codesign/identity") != "") {
			args.push_back("/sha1");
			args.push_back(p_preset->get("codesign/identity"));
		} else {
			add_message(EXPORT_MESSAGE_WARNING, TTR("Code Signing"), TTR("No identity found."));
			return FAILED;
		}
	} else {
		add_message(EXPORT_MESSAGE_WARNING, TTR("Code Signing"), TTR("Invalid identity type."));
		return FAILED;
	}
#else
	if (p_preset->get("codesign/identity") != "") {
		args.push_back("-pkcs12");
		args.push_back(p_preset->get("codesign/identity"));
	} else {
		add_message(EXPORT_MESSAGE_WARNING, TTR("Code Signing"), TTR("No identity found."));
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
			add_message(EXPORT_MESSAGE_WARNING, TTR("Code Signing"), TTR("Invalid timestamp server."));
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
	if (err != OK || (str.find("not found") != -1) || (str.find("not recognized") != -1)) {
#ifndef WINDOWS_ENABLED
		add_message(EXPORT_MESSAGE_WARNING, TTR("Code Signing"), TTR("Could not start signtool executable. Configure signtool path in the Editor Settings (Export > Windows > signtool), or disable \"Codesign\" in the export preset."));
#else
		add_message(EXPORT_MESSAGE_WARNING, TTR("Code Signing"), TTR("Could not start osslsigncode executable. Configure signtool path in the Editor Settings (Export > Windows > osslsigncode), or disable \"Codesign\" in the export preset."));
#endif
		return err;
	}

	print_line("codesign (" + p_path + "): " + str);
#ifndef WINDOWS_ENABLED
	if (str.find("SignTool Error") != -1) {
#else
	if (str.find("Failed") != -1) {
#endif
		add_message(EXPORT_MESSAGE_WARNING, TTR("Code Signing"), vformat(TTR("Signtool failed to sign executable: %s."), str));
		return FAILED;
	}

#ifndef WINDOWS_ENABLED
	DirAccessRef tmp_dir = DirAccess::create_for_path(p_path.get_base_dir());

	err = tmp_dir->remove(p_path);
	if (err != OK) {
		add_message(EXPORT_MESSAGE_WARNING, TTR("Code Signing"), vformat(TTR("Failed to remove temporary file \"%s\"."), p_path));
		return err;
	}

	err = tmp_dir->rename(p_path + "_signed", p_path);
	if (err != OK) {
		add_message(EXPORT_MESSAGE_WARNING, TTR("Code Signing"), vformat(TTR("Failed to rename temporary file \"%s\"."), p_path + "_signed"));
		return err;
	}
#endif

	return OK;
}

bool EditorExportPlatformWindows::has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const {
	String err = "";
	bool valid = EditorExportPlatformPC::has_valid_export_configuration(p_preset, err, r_missing_templates);

	if (!err.empty()) {
		r_error = err;
	}

	return valid;
}

bool EditorExportPlatformWindows::has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error) const {
	String err = "";
	bool valid = true;

	String icon_path = ProjectSettings::get_singleton()->globalize_path(p_preset->get("application/icon"));
	if (!icon_path.empty() && !FileAccess::exists(icon_path)) {
		err += TTR("Invalid icon path:") + " " + icon_path + "\n";
	}

	// Only non-negative integers can exist in the version string.

	String file_version = p_preset->get("application/file_version");
	if (!file_version.empty()) {
		Vector<String> version_array = file_version.split(".", false);
		if (version_array.size() != 4 || !version_array[0].is_valid_integer() ||
				!version_array[1].is_valid_integer() || !version_array[2].is_valid_integer() ||
				!version_array[3].is_valid_integer() || file_version.find("-") > -1) {
			err += TTR("Invalid file version:") + " " + file_version + "\n";
		}
	}

	String product_version = p_preset->get("application/product_version");
	if (!product_version.empty()) {
		Vector<String> version_array = product_version.split(".", false);
		if (version_array.size() != 4 || !version_array[0].is_valid_integer() ||
				!version_array[1].is_valid_integer() || !version_array[2].is_valid_integer() ||
				!version_array[3].is_valid_integer() || product_version.find("-") > -1) {
			err += TTR("Invalid product version:") + " " + product_version + "\n";
		}
	}

	if (!err.empty()) {
		r_error = err;
	}

	return valid;
}

Error EditorExportPlatformWindows::fixup_embedded_pck(const String &p_path, int64_t p_embedded_start, int64_t p_embedded_size) {
	// Patch the header of the "pck" section in the PE file so that it corresponds to the embedded data.

	if (p_embedded_size + p_embedded_start >= 0x100000000) { // Check for total executable size.
		add_message(EXPORT_MESSAGE_ERROR, TTR("PCK Embedding"), TTR("Windows executables cannot be >= 4 GiB."));
		return ERR_INVALID_DATA;
	}

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ_WRITE);
	if (!f) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("PCK Embedding"), vformat(TTR("Failed to open executable file \"%s\"."), p_path));
		return ERR_CANT_OPEN;
	}

	// Jump to the PE header and check the magic number.
	{
		f->seek(0x3c);
		uint32_t pe_pos = f->get_32();

		f->seek(pe_pos);
		uint32_t magic = f->get_32();
		if (magic != 0x00004550) {
			f->close();
			add_message(EXPORT_MESSAGE_ERROR, TTR("PCK Embedding"), TTR("Executable file header corrupted."));
			return ERR_FILE_CORRUPT;
		}
	}

	// Process header.
	uint32_t sect_alignment = 0x1000;
	uint32_t image_size = 0;
	int num_sections;

	int64_t header_pos = f->get_position();
	int64_t opt_header_pos = 0;
	{
		f->seek(header_pos + 2);
		num_sections = f->get_16();
		f->seek(header_pos + 16);
		uint16_t opt_header_size = f->get_16();
		opt_header_pos = f->get_position() + 2;

		f->seek(opt_header_pos + 32);
		sect_alignment = f->get_32();

		f->seek(opt_header_pos + 56);
		image_size = f->get_32();

		// Skip rest of header + optional header to go to the section headers.
		f->seek(opt_header_pos + opt_header_size);
	}

	// Search for the "pck" section.

	int64_t section_table_pos = f->get_position();

	int pck_old_pos = -1;
	for (int i = 0; i < num_sections; i++) {
		int64_t section_header_pos = section_table_pos + i * 40;
		f->seek(section_header_pos);

		uint8_t section_name[9];
		f->get_buffer(section_name, 8);
		section_name[8] = '\0';

		if (strcmp((char *)section_name, "pck") == 0) {
			pck_old_pos = i;

			// Update virtual size of previous section to avoid gaps in the virtual addresses.
			f->seek(section_table_pos + (i - 1) * 40 + 8);
			uint32_t virt_size = f->get_32();
			f->seek(section_table_pos + (i - 1) * 40 + 8);
			f->store_32(virt_size + sect_alignment);
			break;
		}
	}
	if (pck_old_pos >= 0) {
		// Move section data.
		uint8_t section_data[40];
		for (int i = pck_old_pos; i < num_sections - 1; i++) {
			f->seek(section_table_pos + (i + 1) * 40);
			f->get_buffer(section_data, 40);
			f->seek(section_table_pos + i * 40);
			f->store_buffer(section_data, 40);
		}

		// Add "pck" at the end.
		f->seek(section_table_pos + (num_sections - 1) * 40);
		uint8_t section_name[8] = { 'p', 'c', 'k', '\0', '\0', '\0', '\0', '\0' };
		f->store_buffer(section_name, 8); // Name.
		f->store_32(8); // VirtualSize, set to a little to avoid it taking memory (zero would give issues).
		f->store_32(image_size); // VirtualAddress.
		f->store_32(p_embedded_size); // SizeOfRawData.
		f->store_32(p_embedded_start); // PointerToRawData.
		f->store_32(0); // PointerToRelocations, not used.
		f->store_32(0); // PointerToLinenumbers, not used.
		f->store_16(0); // NumberOfRelocations.
		f->store_16(0); // NumberOfLinenumbers.
		f->store_32(0x40000000); // Characteristics: Read.

		// Update image virtual size.
		f->seek(opt_header_pos + 56);
		f->store_32(image_size + sect_alignment);
	}

	f->close();

	if (pck_old_pos == -1) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("PCK Embedding"), TTR("Executable \"pck\" section not found."));
		return ERR_FILE_CORRUPT;
	}
	return OK;
}

void register_windows_exporter() {
#ifndef ANDROID_ENABLED
#ifdef WINDOWS_ENABLED
	EDITOR_DEF("export/windows/signtool", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/windows/signtool", PROPERTY_HINT_GLOBAL_FILE, "*.exe"));
#else
	EDITOR_DEF("export/windows/osslsigncode", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/windows/osslsigncode", PROPERTY_HINT_GLOBAL_FILE));
#endif
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

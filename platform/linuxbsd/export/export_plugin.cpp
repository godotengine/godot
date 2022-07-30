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

Error EditorExportPlatformLinuxBSD::_export_debug_script(const Ref<EditorExportPreset> &p_preset, const String &p_app_name, const String &p_pkg_name, const String &p_path) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	if (f.is_null()) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Debug Script Export"), vformat(TTR("Could not open file \"%s\"."), p_path));
		return ERR_CANT_CREATE;
	}

	f->store_line("#!/bin/sh");
	f->store_line("echo -ne '\\033c\\033]0;" + p_app_name + "\\a'");
	f->store_line("base_path=\"$(dirname \"$(realpath \"$0\")\")\"");
	f->store_line("\"$base_path/" + p_pkg_name + "\" \"$@\"");

	return OK;
}

Error EditorExportPlatformLinuxBSD::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) {
	Error err = EditorExportPlatformPC::export_project(p_preset, p_debug, p_path, p_flags);

	if (err != OK) {
		return err;
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
			String scr_path = p_path.get_basename() + ".sh";
			err = _export_debug_script(p_preset, app_name, p_path.get_file(), scr_path);
			FileAccess::set_unix_permissions(scr_path, 0755);
			if (err != OK) {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Debug Script Export"), TTR("Could not create console script."));
			}
		}
	}

	return err;
}

String EditorExportPlatformLinuxBSD::get_template_file_name(const String &p_target, const String &p_arch) const {
	return "linux_" + p_target + "." + p_arch;
}

List<String> EditorExportPlatformLinuxBSD::get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const {
	List<String> list;
	list.push_back(p_preset->get("binary_format/architecture"));
	return list;
}

void EditorExportPlatformLinuxBSD::get_export_options(List<ExportOption> *r_options) {
	EditorExportPlatformPC::get_export_options(r_options);
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "binary_format/architecture", PROPERTY_HINT_ENUM, "x86_64,x86_32,arm64,arm32,rv64,ppc64,ppc32"), "x86_64"));
}

Error EditorExportPlatformLinuxBSD::fixup_embedded_pck(const String &p_path, int64_t p_embedded_start, int64_t p_embedded_size) {
	// Patch the header of the "pck" section in the ELF file so that it corresponds to the embedded data

	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ_WRITE);
	if (f.is_null()) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("PCK Embedding"), vformat(TTR("Failed to open executable file \"%s\"."), p_path));
		return ERR_CANT_OPEN;
	}

	// Read and check ELF magic number
	{
		uint32_t magic = f->get_32();
		if (magic != 0x464c457f) { // 0x7F + "ELF"
			add_message(EXPORT_MESSAGE_ERROR, TTR("PCK Embedding"), TTR("Executable file header corrupted."));
			return ERR_FILE_CORRUPT;
		}
	}

	// Read program architecture bits from class field

	int bits = f->get_8() * 32;

	if (bits == 32 && p_embedded_size >= 0x100000000) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("PCK Embedding"), TTR("32-bit executables cannot have embedded data >= 4 GiB."));
	}

	// Get info about the section header table

	int64_t section_table_pos;
	int64_t section_header_size;
	if (bits == 32) {
		section_header_size = 40;
		f->seek(0x20);
		section_table_pos = f->get_32();
		f->seek(0x30);
	} else { // 64
		section_header_size = 64;
		f->seek(0x28);
		section_table_pos = f->get_64();
		f->seek(0x3c);
	}
	int num_sections = f->get_16();
	int string_section_idx = f->get_16();

	// Load the strings table
	uint8_t *strings;
	{
		// Jump to the strings section header
		f->seek(section_table_pos + string_section_idx * section_header_size);

		// Read strings data size and offset
		int64_t string_data_pos;
		int64_t string_data_size;
		if (bits == 32) {
			f->seek(f->get_position() + 0x10);
			string_data_pos = f->get_32();
			string_data_size = f->get_32();
		} else { // 64
			f->seek(f->get_position() + 0x18);
			string_data_pos = f->get_64();
			string_data_size = f->get_64();
		}

		// Read strings data
		f->seek(string_data_pos);
		strings = (uint8_t *)memalloc(string_data_size);
		if (!strings) {
			return ERR_OUT_OF_MEMORY;
		}
		f->get_buffer(strings, string_data_size);
	}

	// Search for the "pck" section

	bool found = false;
	for (int i = 0; i < num_sections; ++i) {
		int64_t section_header_pos = section_table_pos + i * section_header_size;
		f->seek(section_header_pos);

		uint32_t name_offset = f->get_32();
		if (strcmp((char *)strings + name_offset, "pck") == 0) {
			// "pck" section found, let's patch!

			if (bits == 32) {
				f->seek(section_header_pos + 0x10);
				f->store_32(p_embedded_start);
				f->store_32(p_embedded_size);
			} else { // 64
				f->seek(section_header_pos + 0x18);
				f->store_64(p_embedded_start);
				f->store_64(p_embedded_size);
			}

			found = true;
			break;
		}
	}

	memfree(strings);

	if (!found) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("PCK Embedding"), TTR("Executable \"pck\" section not found."));
		return ERR_FILE_CORRUPT;
	}
	return OK;
}

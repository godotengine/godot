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

#include "export.h"

#include "core/io/file_access.h"
#include "editor/editor_export.h"
#include "platform/linuxbsd/logo.gen.h"
#include "scene/resources/texture.h"

static Error fixup_embedded_pck(const String &p_path, int64_t p_embedded_start, int64_t p_embedded_size);

void register_linuxbsd_exporter() {
	Ref<EditorExportPlatformPC> platform;
	platform.instantiate();

	Ref<Image> img = memnew(Image(_linuxbsd_logo));
	Ref<ImageTexture> logo;
	logo.instantiate();
	logo->create_from_image(img);
	platform->set_logo(logo);
	platform->set_name("Linux/X11");
	platform->set_extension("x86");
	platform->set_extension("x86_64", "binary_format/64_bits");
	platform->set_release_32("linux_x11_32_release");
	platform->set_debug_32("linux_x11_32_debug");
	platform->set_release_64("linux_x11_64_release");
	platform->set_debug_64("linux_x11_64_debug");
	platform->set_os_name("LinuxBSD");
	platform->set_chmod_flags(0755);
	platform->set_fixup_embedded_pck_func(&fixup_embedded_pck);

	EditorExport::get_singleton()->add_export_platform(platform);
}

static Error fixup_embedded_pck(const String &p_path, int64_t p_embedded_start, int64_t p_embedded_size) {
	// Patch the header of the "pck" section in the ELF file so that it corresponds to the embedded data

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ_WRITE);
	if (!f) {
		return ERR_CANT_OPEN;
	}

	// Read and check ELF magic number
	{
		uint32_t magic = f->get_32();
		if (magic != 0x464c457f) { // 0x7F + "ELF"
			f->close();
			return ERR_FILE_CORRUPT;
		}
	}

	// Read program architecture bits from class field

	int bits = f->get_8() * 32;

	if (bits == 32 && p_embedded_size >= 0x100000000) {
		f->close();
		ERR_FAIL_V_MSG(ERR_INVALID_DATA, "32-bit executables cannot have embedded data >= 4 GiB.");
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
			f->close();
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
	f->close();

	return found ? OK : ERR_FILE_CORRUPT;
}

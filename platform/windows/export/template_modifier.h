/**************************************************************************/
/*  template_modifier.h                                                   */
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

#ifndef WINDOWS_TEMPLATE_MODIFIER_H
#define WINDOWS_TEMPLATE_MODIFIER_H

#include "core/io/file_access.h"
#include "editor/export/editor_export_platform_pc.h"

class TemplateModifier {
	const uint32_t PAGE_SIZE = 4096;
	const uint32_t BLOCK_SIZE = 512;
	const uint32_t POINTER_TO_PE_HEADER_OFFSET = 0x3c;
	const uint32_t SIZE_OF_OPTIONAL_HEADER_OFFSET = 20;
	const uint32_t COFF_HEADER_SIZE = 24;
	const uint32_t SIZE_OF_IMAGE_OFFSET = 56;

	const uint8_t RESOURCE_DIRECTORY_TABLES[0x164] = {
		/*000*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, // root table 3 entries
		/*010*/ 0x03, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x80, // RT_ICON
		/*018*/ 0x0e, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x00, 0x80, // RT_GROUP_ICON
		/*020*/ 0x10, 0x00, 0x00, 0x00, 0x28, 0x01, 0x00, 0x80, // RT_VERSION
		/*028*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, // icon ids table 6 entries
		/*038*/ 0x01, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x80, // ICON 1 id
		/*040*/ 0x02, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x80, // ICON 2 id
		/*048*/ 0x03, 0x00, 0x00, 0x00, 0x98, 0x00, 0x00, 0x80, // ICON 3 id
		/*050*/ 0x04, 0x00, 0x00, 0x00, 0xb0, 0x00, 0x00, 0x80, // ICON 4 id
		/*058*/ 0x05, 0x00, 0x00, 0x00, 0xc8, 0x00, 0x00, 0x80, // ICON 5 id
		/*060*/ 0x06, 0x00, 0x00, 0x00, 0xe0, 0x00, 0x00, 0x80, // ICON 6 id
		/*068*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, // icon 1 language table
		/*078*/ 0x09, 0x04, 0x00, 0x00, 0x64, 0x01, 0x00, 0x00, // icon 1 language 1033
		/*080*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, // icon 2 language table
		/*088*/ 0x09, 0x04, 0x00, 0x00, 0x74, 0x01, 0x00, 0x00, // icon 2 language 1033
		/*098*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, // icon 3 language table
		/*0a8*/ 0x09, 0x04, 0x00, 0x00, 0x84, 0x01, 0x00, 0x00, // icon 3 language 1033
		/*0b0*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, // icon 4 language table
		/*0c0*/ 0x09, 0x04, 0x00, 0x00, 0x94, 0x01, 0x00, 0x00, // icon 4 language 1033
		/*0c8*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, // icon 5 language table
		/*0d8*/ 0x09, 0x04, 0x00, 0x00, 0xa4, 0x01, 0x00, 0x00, // icon 5 language 1033
		/*0e0*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, // icon 6 language table
		/*0f0*/ 0x09, 0x04, 0x00, 0x00, 0xb4, 0x01, 0x00, 0x00, // icon 6 language 1033
		/*0f8*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // group icon name table 1 entry
		/*108*/ 0x58, 0x01, 0x00, 0x80, 0x10, 0x01, 0x00, 0x80, // group icon name entry
		/*110*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, // group icon language table
		/*120*/ 0x09, 0x04, 0x00, 0x00, 0xc4, 0x01, 0x00, 0x00, // group icon language 1033
		/*128*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, // version ids table 1 entry
		/*138*/ 0x01, 0x00, 0x00, 0x00, 0x40, 0x01, 0x00, 0x80, // version id entry
		/*140*/ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, // version language table
		/*150*/ 0x09, 0x04, 0x00, 0x00, 0xd4, 0x01, 0x00, 0x00, // version language 1033
		/*158*/ 0x04, 0x00, 0x49, 0x00, 0x43, 0x00, 0x4f, 0x00, 0x4e, 0x00, 0x00, 0x00 // (4 0) as "icon" length I.C.O.N. and padding to 32 bits
		// TODO
		/*164   insert rva, size, codepage = 0, reserved = 0, 8 times */
	};

	struct ByteStream {
		void save(uint8_t p_value, Vector<uint8_t> &r_bytes) const;
		void save(uint16_t p_value, Vector<uint8_t> &r_bytes) const;
		void save(uint32_t p_value, Vector<uint8_t> &r_bytes) const;
		void save(const String &p_value, Vector<uint8_t> &r_bytes) const;
		void save(uint32_t p_value, Vector<uint8_t> &r_bytes, uint32_t p_count) const;
		Vector<uint8_t> save() const;
	};

	struct FixedFileInfo : ByteStream {
		uint32_t signature = 0xfeef04bd;
		uint32_t struct_version = 0x10000;
		uint32_t file_version_ms = 0;
		uint32_t file_version_ls = 0;
		uint32_t product_version_ms = 0;
		uint32_t product_version_ls = 0;
		uint32_t file_flags_mask = 0;
		uint32_t file_flags = 0;
		uint32_t file_os = 0x00000004;
		uint32_t file_type = 0x00000001;
		uint32_t file_subtype = 0;
		uint32_t file_date_ms = 0;
		uint32_t file_date_ls = 0;

		Vector<uint8_t> save() const;
		void set_file_version(const String &p_file_version);
		void set_product_version(const String &p_product_version);
	};

	struct Structure : ByteStream {
		uint16_t length = 0;
		uint16_t value_length = 0;
		uint16_t type = 0;
		String key;

		Vector<uint8_t> save() const;
		Vector<uint8_t> &add_length(Vector<uint8_t> &bytes) const;
	};

	struct StringStructure : Structure {
		String value;

		Vector<uint8_t> save() const;
		StringStructure();
		StringStructure(String p_key, String p_value);
	};

	struct StringTable : Structure {
		Vector<StringStructure> strings;

		Vector<uint8_t> save() const;
		void put(const String &p_key, const String &p_value);
		StringTable();
	};

	struct StringFileInfo : Structure {
		StringTable string_table;

		Vector<uint8_t> save() const;
		StringFileInfo();
	};

	struct Var : Structure {
		const uint32_t value = 0x04b00409;

		Vector<uint8_t> save() const;
		Var();
	};

	struct VarFileInfo : Structure {
		Var var;

		Vector<uint8_t> save() const;
		VarFileInfo();
	};

	struct VersionInfo : Structure {
		FixedFileInfo value;
		StringFileInfo string_file_info;
		VarFileInfo var_file_info;

		Vector<uint8_t> save() const;
		VersionInfo();
	};

	struct IconEntry : ByteStream {
		static const uint32_t SIZE = 16;

		uint8_t width;
		uint8_t height;
		uint8_t colors = 0;
		uint8_t reserved = 0;
		uint16_t planes = 0;
		uint16_t bits_per_pixel = 32;
		uint32_t image_size;
		uint32_t image_offset;
		Vector<uint8_t> data;

		Vector<uint8_t> save() const;
		void load(Ref<FileAccess> p_file);
	};

	struct GroupIcon : ByteStream {
		static const uint16_t IMAGE_COUNT = 6;
		static constexpr uint8_t SIZES[6] = { 16, 32, 48, 64, 128, 0 };

		uint16_t reserved = 0;
		uint16_t type = 1;
		uint16_t image_count = IMAGE_COUNT;
		Vector<IconEntry> icon_entries;
		Vector<Vector<uint8_t>> images;

		Vector<uint8_t> save() const;
		void load(Ref<FileAccess> p_icon_file);
		void fill_with_godot_blue();
	};

	struct SectionEntry : ByteStream {
		static const uint32_t SIZE = 40;

		String name;
		uint32_t virtual_size;
		uint32_t virtual_address;
		uint32_t size_of_raw_data;
		uint32_t pointer_to_raw_data;
		uint32_t pointer_to_relocations;
		uint32_t pointer_to_line_numbers;
		uint16_t number_of_relocations;
		uint16_t number_of_line_numbers;
		uint32_t characteristics;

		Vector<uint8_t> save() const;
		void load(Ref<FileAccess> p_file);
	};

	struct ResourceDataEntry : ByteStream {
		static const uint16_t SIZE = 16;

		uint32_t rva;
		uint32_t size;

		Vector<uint8_t> save() const;
	};

	uint32_t _snap(uint32_t p_value, uint32_t p_size) const;
	uint32_t _get_pe_header_offset(const String &p_executable_path) const;
	Vector<SectionEntry> _get_section_entries(const String &p_executable_path) const;
	String _get_icon_path(const Ref<EditorExportPreset> &p_preset, bool p_console_icon) const;
	GroupIcon _create_group_icon(const String &p_icon_path) const;
	VersionInfo _create_version_info(const HashMap<String, String> &p_strings) const;
	Vector<uint8_t> _create_resources(uint32_t p_virtual_address, const GroupIcon &p_group_icon, const VersionInfo &p_version_info) const;
	Error _truncate(const String &p_executable_path, uint32_t p_size) const;
	HashMap<String, String> _get_strings(const Ref<EditorExportPreset> &p_preset) const;
	Error _modify_template(const Ref<EditorExportPreset> &p_preset, const String &p_template_path, bool p_console_icon) const;

public:
	// TODO would be nice to remove p_console_icon arg
	static Error modify(const Ref<EditorExportPreset> &p_preset, const String &p_template_path, bool p_console_icon);
};

#endif // WINDOWS_TEMPLATE_MODIFIER_H

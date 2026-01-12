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

#pragma once

#include "core/io/file_access.h"
#include "editor/export/editor_export_platform_pc.h"

class TemplateModifier {
	const uint32_t PE_PAGE_SIZE = 4096;
	const uint32_t BLOCK_SIZE = 512;
	const uint32_t COFF_HEADER_SIZE = 24;
	const uint32_t POINTER_TO_PE_HEADER_OFFSET = 0x3c;
	// all offsets below are calculated from POINTER_TO_PE_HEADER_OFFSET value, at 0 is magic string PE (0x50450000)
	const uint32_t MAGIC_NUMBER_OFFSET = 24;
	const uint32_t SIZE_OF_INITIALIZED_DATA_OFFSET = 32;
	const uint32_t SIZE_OF_IMAGE_OFFSET = 80;

	struct ByteStream {
		void save(uint8_t p_value, Vector<uint8_t> &r_bytes) const;
		void save(uint16_t p_value, Vector<uint8_t> &r_bytes) const;
		void save(uint32_t p_value, Vector<uint8_t> &r_bytes) const;
		void save(const String &p_value, Vector<uint8_t> &r_bytes) const;
		void save(uint32_t p_value, Vector<uint8_t> &r_bytes, uint32_t p_count) const;
		Vector<uint8_t> save() const;
	};

	struct ResourceDirectoryTable : ByteStream {
		static const uint16_t SIZE = 16;

		uint16_t name_entry_count = 0;
		uint16_t id_entry_count = 0;

		Vector<uint8_t> save() const;
	};

	struct ResourceDirectoryEntry : ByteStream {
		static const uint16_t SIZE = 8;
		static const uint32_t ICON = 0x03;
		static const uint32_t GROUP_ICON = 0x0e;
		static const uint32_t MANIFEST = 0x18;
		static const uint32_t VERSION = 0x10;
		static const uint32_t ENGLISH = 0x0409;
		static const uint32_t HIGH_BIT = 0x80000000;

		uint32_t id = 0;
		uint32_t data_offset = 0;
		bool name = false;
		bool subdirectory = false;

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
		Vector<uint8_t> &add_length(Vector<uint8_t> &r_bytes) const;
	};

	struct StringStructure : Structure {
		String value;

		Vector<uint8_t> save() const;
		StringStructure();
		StringStructure(const String &p_key, const String &p_value);
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

	struct ManifestInfo : Structure {
		String manifest;

		Vector<uint8_t> save() const;
		ManifestInfo() {}
	};

	struct IconEntry : ByteStream {
		static const uint32_t SIZE = 16;

		uint8_t width = 0;
		uint8_t height = 0;
		uint8_t colors = 0;
		uint8_t reserved = 0;
		uint16_t planes = 0;
		uint16_t bits_per_pixel = 32;
		uint32_t image_size = 0;
		uint32_t image_offset = 0;
		Vector<uint8_t> data;

		Vector<uint8_t> save() const;
		void load(Ref<FileAccess> p_file);
	};

	struct GroupIcon : ByteStream {
		static constexpr uint8_t SIZES[6] = { 16, 32, 48, 64, 128, 0 };

		uint16_t reserved = 0;
		uint16_t type = 1;
		uint16_t image_count = 0;
		Vector<IconEntry> icon_entries;
		Vector<Vector<uint8_t>> images;

		Vector<uint8_t> save() const;
		void load(Ref<FileAccess> p_icon_file);
		void fill_with_godot_blue();
	};

	struct SectionEntry : ByteStream {
		static const uint32_t SIZE = 40;

		String name;
		uint32_t virtual_size = 0;
		uint32_t virtual_address = 0;
		uint32_t size_of_raw_data = 0;
		uint32_t pointer_to_raw_data = 0;
		uint32_t pointer_to_relocations = 0;
		uint32_t pointer_to_line_numbers = 0;
		uint16_t number_of_relocations = 0;
		uint16_t number_of_line_numbers = 0;
		uint32_t characteristics = 0;

		Vector<uint8_t> save() const;
		void load(Ref<FileAccess> p_file);
	};

	struct ResourceDataEntry : ByteStream {
		static const uint16_t SIZE = 16;

		uint32_t rva = 0;
		uint32_t size = 0;

		Vector<uint8_t> save() const;
	};

	uint32_t _snap(uint32_t p_value, uint32_t p_size) const;
	uint32_t _get_pe_header_offset(Ref<FileAccess> p_executable) const;
	Vector<SectionEntry> _get_section_entries(Ref<FileAccess> p_executable) const;
	GroupIcon _create_group_icon(const String &p_icon_path) const;
	ManifestInfo _create_manifest_info() const;
	VersionInfo _create_version_info(const HashMap<String, String> &p_strings) const;
	Vector<uint8_t> _create_resources(uint32_t p_virtual_address, const GroupIcon &p_group_icon, const VersionInfo &p_version_info, const ManifestInfo &p_manifest_info) const;
	Error _truncate(const String &p_executable_path, uint32_t p_size) const;
	HashMap<String, String> _get_strings(const Ref<EditorExportPreset> &p_preset) const;
	Error _modify_template(const Ref<EditorExportPreset> &p_preset, const String &p_template_path, const String &p_icon_path) const;

public:
	static Error modify(const Ref<EditorExportPreset> &p_preset, const String &p_template_path, const String &p_icon_path);
};

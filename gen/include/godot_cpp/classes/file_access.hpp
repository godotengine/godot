/**************************************************************************/
/*  file_access.hpp                                                       */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class FileAccess : public RefCounted {
	GDEXTENSION_CLASS(FileAccess, RefCounted)

public:
	enum ModeFlags {
		READ = 1,
		WRITE = 2,
		READ_WRITE = 3,
		WRITE_READ = 7,
	};

	enum CompressionMode {
		COMPRESSION_FASTLZ = 0,
		COMPRESSION_DEFLATE = 1,
		COMPRESSION_ZSTD = 2,
		COMPRESSION_GZIP = 3,
		COMPRESSION_BROTLI = 4,
	};

	enum UnixPermissionFlags : uint64_t {
		UNIX_READ_OWNER = 256,
		UNIX_WRITE_OWNER = 128,
		UNIX_EXECUTE_OWNER = 64,
		UNIX_READ_GROUP = 32,
		UNIX_WRITE_GROUP = 16,
		UNIX_EXECUTE_GROUP = 8,
		UNIX_READ_OTHER = 4,
		UNIX_WRITE_OTHER = 2,
		UNIX_EXECUTE_OTHER = 1,
		UNIX_SET_USER_ID = 2048,
		UNIX_SET_GROUP_ID = 1024,
		UNIX_RESTRICTED_DELETE = 512,
	};

	static Ref<FileAccess> open(const String &p_path, FileAccess::ModeFlags p_flags);
	static Ref<FileAccess> open_encrypted(const String &p_path, FileAccess::ModeFlags p_mode_flags, const PackedByteArray &p_key, const PackedByteArray &p_iv = PackedByteArray());
	static Ref<FileAccess> open_encrypted_with_pass(const String &p_path, FileAccess::ModeFlags p_mode_flags, const String &p_pass);
	static Ref<FileAccess> open_compressed(const String &p_path, FileAccess::ModeFlags p_mode_flags, FileAccess::CompressionMode p_compression_mode = (FileAccess::CompressionMode)0);
	static Error get_open_error();
	static Ref<FileAccess> create_temp(FileAccess::ModeFlags p_mode_flags, const String &p_prefix = String(), const String &p_extension = String(), bool p_keep = false);
	static PackedByteArray get_file_as_bytes(const String &p_path);
	static String get_file_as_string(const String &p_path);
	Error resize(int64_t p_length);
	void flush();
	String get_path() const;
	String get_path_absolute() const;
	bool is_open() const;
	void seek(uint64_t p_position);
	void seek_end(int64_t p_position = 0);
	uint64_t get_position() const;
	uint64_t get_length() const;
	bool eof_reached() const;
	uint8_t get_8() const;
	uint16_t get_16() const;
	uint32_t get_32() const;
	uint64_t get_64() const;
	float get_half() const;
	float get_float() const;
	double get_double() const;
	float get_real() const;
	PackedByteArray get_buffer(int64_t p_length) const;
	String get_line() const;
	PackedStringArray get_csv_line(const String &p_delim = ",") const;
	String get_as_text() const;
	static String get_md5(const String &p_path);
	static String get_sha256(const String &p_path);
	bool is_big_endian() const;
	void set_big_endian(bool p_big_endian);
	Error get_error() const;
	Variant get_var(bool p_allow_objects = false) const;
	bool store_8(uint8_t p_value);
	bool store_16(uint16_t p_value);
	bool store_32(uint32_t p_value);
	bool store_64(uint64_t p_value);
	bool store_half(float p_value);
	bool store_float(float p_value);
	bool store_double(double p_value);
	bool store_real(float p_value);
	bool store_buffer(const PackedByteArray &p_buffer);
	bool store_line(const String &p_line);
	bool store_csv_line(const PackedStringArray &p_values, const String &p_delim = ",");
	bool store_string(const String &p_string);
	bool store_var(const Variant &p_value, bool p_full_objects = false);
	bool store_pascal_string(const String &p_string);
	String get_pascal_string();
	void close();
	static bool file_exists(const String &p_path);
	static uint64_t get_modified_time(const String &p_file);
	static uint64_t get_access_time(const String &p_file);
	static int64_t get_size(const String &p_file);
	static BitField<FileAccess::UnixPermissionFlags> get_unix_permissions(const String &p_file);
	static Error set_unix_permissions(const String &p_file, BitField<FileAccess::UnixPermissionFlags> p_permissions);
	static bool get_hidden_attribute(const String &p_file);
	static Error set_hidden_attribute(const String &p_file, bool p_hidden);
	static Error set_read_only_attribute(const String &p_file, bool p_ro);
	static bool get_read_only_attribute(const String &p_file);
	static PackedByteArray get_extended_attribute(const String &p_file, const String &p_attribute_name);
	static String get_extended_attribute_string(const String &p_file, const String &p_attribute_name);
	static Error set_extended_attribute(const String &p_file, const String &p_attribute_name, const PackedByteArray &p_data);
	static Error set_extended_attribute_string(const String &p_file, const String &p_attribute_name, const String &p_data);
	static Error remove_extended_attribute(const String &p_file, const String &p_attribute_name);
	static PackedStringArray get_extended_attributes_list(const String &p_file);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
	uint64_t get_buffer(uint8_t *p_dst, uint64_t p_length) const;
	void store_buffer(const uint8_t *p_src, uint64_t p_length);
};

} // namespace godot

VARIANT_ENUM_CAST(FileAccess::ModeFlags);
VARIANT_ENUM_CAST(FileAccess::CompressionMode);
VARIANT_BITFIELD_CAST(FileAccess::UnixPermissionFlags);


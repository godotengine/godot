/**************************************************************************/
/*  dir_access.hpp                                                        */
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
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class DirAccess : public RefCounted {
	GDEXTENSION_CLASS(DirAccess, RefCounted)

public:
	static Ref<DirAccess> open(const String &p_path);
	static Error get_open_error();
	static Ref<DirAccess> create_temp(const String &p_prefix = String(), bool p_keep = false);
	Error list_dir_begin();
	String get_next();
	bool current_is_dir() const;
	void list_dir_end();
	PackedStringArray get_files();
	static PackedStringArray get_files_at(const String &p_path);
	PackedStringArray get_directories();
	static PackedStringArray get_directories_at(const String &p_path);
	static int32_t get_drive_count();
	static String get_drive_name(int32_t p_idx);
	int32_t get_current_drive();
	Error change_dir(const String &p_to_dir);
	String get_current_dir(bool p_include_drive = true) const;
	Error make_dir(const String &p_path);
	static Error make_dir_absolute(const String &p_path);
	Error make_dir_recursive(const String &p_path);
	static Error make_dir_recursive_absolute(const String &p_path);
	bool file_exists(const String &p_path);
	bool dir_exists(const String &p_path);
	static bool dir_exists_absolute(const String &p_path);
	uint64_t get_space_left();
	Error copy(const String &p_from, const String &p_to, int32_t p_chmod_flags = -1);
	static Error copy_absolute(const String &p_from, const String &p_to, int32_t p_chmod_flags = -1);
	Error rename(const String &p_from, const String &p_to);
	static Error rename_absolute(const String &p_from, const String &p_to);
	Error remove(const String &p_path);
	static Error remove_absolute(const String &p_path);
	bool is_link(const String &p_path);
	String read_link(const String &p_path);
	Error create_link(const String &p_source, const String &p_target);
	bool is_bundle(const String &p_path) const;
	void set_include_navigational(bool p_enable);
	bool get_include_navigational() const;
	void set_include_hidden(bool p_enable);
	bool get_include_hidden() const;
	String get_filesystem_type() const;
	bool is_case_sensitive(const String &p_path) const;
	bool is_equivalent(const String &p_path_a, const String &p_path_b) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot


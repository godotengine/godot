/**************************************************************************/
/*  project_settings.hpp                                                  */
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
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class StringName;

class ProjectSettings : public Object {
	GDEXTENSION_CLASS(ProjectSettings, Object)

	static ProjectSettings *singleton;

public:
	static ProjectSettings *get_singleton();

	bool has_setting(const String &p_name) const;
	void set_setting(const String &p_name, const Variant &p_value);
	Variant get_setting(const String &p_name, const Variant &p_default_value = nullptr) const;
	Variant get_setting_with_override(const StringName &p_name) const;
	TypedArray<Dictionary> get_global_class_list();
	Variant get_setting_with_override_and_custom_features(const StringName &p_name, const PackedStringArray &p_features) const;
	void set_order(const String &p_name, int32_t p_position);
	int32_t get_order(const String &p_name) const;
	void set_initial_value(const String &p_name, const Variant &p_value);
	void set_as_basic(const String &p_name, bool p_basic);
	void set_as_internal(const String &p_name, bool p_internal);
	void add_property_info(const Dictionary &p_hint);
	void set_restart_if_changed(const String &p_name, bool p_restart);
	void clear(const String &p_name);
	String localize_path(const String &p_path) const;
	String globalize_path(const String &p_path) const;
	Error save();
	bool load_resource_pack(const String &p_pack, bool p_replace_files = true, int32_t p_offset = 0);
	Error save_custom(const String &p_file);
	PackedStringArray get_changed_settings() const;
	bool check_changed_settings_in_group(const String &p_setting_prefix) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~ProjectSettings();

public:
};

} // namespace godot


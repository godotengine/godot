/**************************************************************************/
/*  editor_settings.hpp                                                   */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Dictionary;
class InputEvent;
class Shortcut;
class String;
class StringName;

class EditorSettings : public Resource {
	GDEXTENSION_CLASS(EditorSettings, Resource)

public:
	static const int NOTIFICATION_EDITOR_SETTINGS_CHANGED = 10000;

	bool has_setting(const String &p_name) const;
	void set_setting(const String &p_name, const Variant &p_value);
	Variant get_setting(const String &p_name) const;
	void erase(const String &p_property);
	void set_initial_value(const StringName &p_name, const Variant &p_value, bool p_update_current);
	void add_property_info(const Dictionary &p_info);
	void set_project_metadata(const String &p_section, const String &p_key, const Variant &p_data);
	Variant get_project_metadata(const String &p_section, const String &p_key, const Variant &p_default = nullptr) const;
	void set_favorites(const PackedStringArray &p_dirs);
	PackedStringArray get_favorites() const;
	void set_recent_dirs(const PackedStringArray &p_dirs);
	PackedStringArray get_recent_dirs() const;
	void set_builtin_action_override(const String &p_name, const TypedArray<Ref<InputEvent>> &p_actions_list);
	void add_shortcut(const String &p_path, const Ref<Shortcut> &p_shortcut);
	void remove_shortcut(const String &p_path);
	bool is_shortcut(const String &p_path, const Ref<InputEvent> &p_event) const;
	bool has_shortcut(const String &p_path) const;
	Ref<Shortcut> get_shortcut(const String &p_path) const;
	PackedStringArray get_shortcut_list();
	bool check_changed_settings_in_group(const String &p_setting_prefix) const;
	PackedStringArray get_changed_settings() const;
	void mark_setting_changed(const String &p_setting);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot


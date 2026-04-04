/**************************************************************************/
/*  translation.hpp                                                       */
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
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Translation : public Resource {
	GDEXTENSION_CLASS(Translation, Resource)

public:
	void set_locale(const String &p_locale);
	String get_locale() const;
	void add_message(const StringName &p_src_message, const StringName &p_xlated_message, const StringName &p_context = StringName());
	void add_plural_message(const StringName &p_src_message, const PackedStringArray &p_xlated_messages, const StringName &p_context = StringName());
	StringName get_message(const StringName &p_src_message, const StringName &p_context = StringName()) const;
	StringName get_plural_message(const StringName &p_src_message, const StringName &p_src_plural_message, int32_t p_n, const StringName &p_context = StringName()) const;
	void erase_message(const StringName &p_src_message, const StringName &p_context = StringName());
	PackedStringArray get_message_list() const;
	PackedStringArray get_translated_message_list() const;
	int32_t get_message_count() const;
	void set_plural_rules_override(const String &p_rules);
	String get_plural_rules_override() const;
	virtual StringName _get_plural_message(const StringName &p_src_message, const StringName &p_src_plural_message, int32_t p_n, const StringName &p_context) const;
	virtual StringName _get_message(const StringName &p_src_message, const StringName &p_context) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_plural_message), decltype(&T::_get_plural_message)>) {
			BIND_VIRTUAL_METHOD(T, _get_plural_message, 1970324172);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_message), decltype(&T::_get_message)>) {
			BIND_VIRTUAL_METHOD(T, _get_message, 3639719779);
		}
	}

public:
};

} // namespace godot


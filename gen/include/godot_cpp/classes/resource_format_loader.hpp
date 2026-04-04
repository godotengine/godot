/**************************************************************************/
/*  resource_format_loader.hpp                                            */
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
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Dictionary;
class StringName;

class ResourceFormatLoader : public RefCounted {
	GDEXTENSION_CLASS(ResourceFormatLoader, RefCounted)

public:
	enum CacheMode {
		CACHE_MODE_IGNORE = 0,
		CACHE_MODE_REUSE = 1,
		CACHE_MODE_REPLACE = 2,
		CACHE_MODE_IGNORE_DEEP = 3,
		CACHE_MODE_REPLACE_DEEP = 4,
	};

	virtual PackedStringArray _get_recognized_extensions() const;
	virtual bool _recognize_path(const String &p_path, const StringName &p_type) const;
	virtual bool _handles_type(const StringName &p_type) const;
	virtual String _get_resource_type(const String &p_path) const;
	virtual String _get_resource_script_class(const String &p_path) const;
	virtual int64_t _get_resource_uid(const String &p_path) const;
	virtual PackedStringArray _get_dependencies(const String &p_path, bool p_add_types) const;
	virtual Error _rename_dependencies(const String &p_path, const Dictionary &p_renames) const;
	virtual bool _exists(const String &p_path) const;
	virtual PackedStringArray _get_classes_used(const String &p_path) const;
	virtual Variant _load(const String &p_path, const String &p_original_path, bool p_use_sub_threads, int32_t p_cache_mode) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_recognized_extensions), decltype(&T::_get_recognized_extensions)>) {
			BIND_VIRTUAL_METHOD(T, _get_recognized_extensions, 1139954409);
		}
		if constexpr (!std::is_same_v<decltype(&B::_recognize_path), decltype(&T::_recognize_path)>) {
			BIND_VIRTUAL_METHOD(T, _recognize_path, 2594487047);
		}
		if constexpr (!std::is_same_v<decltype(&B::_handles_type), decltype(&T::_handles_type)>) {
			BIND_VIRTUAL_METHOD(T, _handles_type, 2619796661);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_resource_type), decltype(&T::_get_resource_type)>) {
			BIND_VIRTUAL_METHOD(T, _get_resource_type, 3135753539);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_resource_script_class), decltype(&T::_get_resource_script_class)>) {
			BIND_VIRTUAL_METHOD(T, _get_resource_script_class, 3135753539);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_resource_uid), decltype(&T::_get_resource_uid)>) {
			BIND_VIRTUAL_METHOD(T, _get_resource_uid, 1321353865);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_dependencies), decltype(&T::_get_dependencies)>) {
			BIND_VIRTUAL_METHOD(T, _get_dependencies, 6257701);
		}
		if constexpr (!std::is_same_v<decltype(&B::_rename_dependencies), decltype(&T::_rename_dependencies)>) {
			BIND_VIRTUAL_METHOD(T, _rename_dependencies, 223715120);
		}
		if constexpr (!std::is_same_v<decltype(&B::_exists), decltype(&T::_exists)>) {
			BIND_VIRTUAL_METHOD(T, _exists, 3927539163);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_classes_used), decltype(&T::_get_classes_used)>) {
			BIND_VIRTUAL_METHOD(T, _get_classes_used, 4291131558);
		}
		if constexpr (!std::is_same_v<decltype(&B::_load), decltype(&T::_load)>) {
			BIND_VIRTUAL_METHOD(T, _load, 2885906527);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(ResourceFormatLoader::CacheMode);


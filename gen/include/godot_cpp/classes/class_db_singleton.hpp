/**************************************************************************/
/*  class_db_singleton.hpp                                                */
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
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/binder_common.hpp>

namespace godot {

class ClassDBSingleton : public Object {
	GDEXTENSION_CLASS_ALIAS(ClassDBSingleton, ClassDB, Object)

	static ClassDBSingleton *singleton;

public:
	enum APIType {
		API_CORE = 0,
		API_EDITOR = 1,
		API_EXTENSION = 2,
		API_EDITOR_EXTENSION = 3,
		API_NONE = 4,
	};

	static ClassDBSingleton *get_singleton();

	PackedStringArray get_class_list() const;
	PackedStringArray get_inheriters_from_class(const StringName &p_class) const;
	StringName get_parent_class(const StringName &p_class) const;
	bool class_exists(const StringName &p_class) const;
	bool is_parent_class(const StringName &p_class, const StringName &p_inherits) const;
	bool can_instantiate(const StringName &p_class) const;
	Variant instantiate(const StringName &p_class) const;
	ClassDBSingleton::APIType class_get_api_type(const StringName &p_class) const;
	bool class_has_signal(const StringName &p_class, const StringName &p_signal) const;
	Dictionary class_get_signal(const StringName &p_class, const StringName &p_signal) const;
	TypedArray<Dictionary> class_get_signal_list(const StringName &p_class, bool p_no_inheritance = false) const;
	TypedArray<Dictionary> class_get_property_list(const StringName &p_class, bool p_no_inheritance = false) const;
	StringName class_get_property_getter(const StringName &p_class, const StringName &p_property);
	StringName class_get_property_setter(const StringName &p_class, const StringName &p_property);
	Variant class_get_property(Object *p_object, const StringName &p_property) const;
	Error class_set_property(Object *p_object, const StringName &p_property, const Variant &p_value) const;
	Variant class_get_property_default_value(const StringName &p_class, const StringName &p_property) const;
	bool class_has_method(const StringName &p_class, const StringName &p_method, bool p_no_inheritance = false) const;
	int32_t class_get_method_argument_count(const StringName &p_class, const StringName &p_method, bool p_no_inheritance = false) const;
	TypedArray<Dictionary> class_get_method_list(const StringName &p_class, bool p_no_inheritance = false) const;

private:
	Variant class_call_static_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	Variant class_call_static(const StringName &p_class, const StringName &p_method, const Args &...p_args) {
		std::array<Variant, 2 + sizeof...(Args)> variant_args{{ Variant(p_class), Variant(p_method), Variant(p_args)... }};
		std::array<const Variant *, 2 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		return class_call_static_internal(call_args.data(), variant_args.size());
	}
	PackedStringArray class_get_integer_constant_list(const StringName &p_class, bool p_no_inheritance = false) const;
	bool class_has_integer_constant(const StringName &p_class, const StringName &p_name) const;
	int64_t class_get_integer_constant(const StringName &p_class, const StringName &p_name) const;
	bool class_has_enum(const StringName &p_class, const StringName &p_name, bool p_no_inheritance = false) const;
	PackedStringArray class_get_enum_list(const StringName &p_class, bool p_no_inheritance = false) const;
	PackedStringArray class_get_enum_constants(const StringName &p_class, const StringName &p_enum, bool p_no_inheritance = false) const;
	StringName class_get_integer_constant_enum(const StringName &p_class, const StringName &p_name, bool p_no_inheritance = false) const;
	bool is_class_enum_bitfield(const StringName &p_class, const StringName &p_enum, bool p_no_inheritance = false) const;
	bool is_class_enabled(const StringName &p_class) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~ClassDBSingleton();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(ClassDBSingleton::APIType);

#define CLASSDB_SINGLETON_FORWARD_METHODS \
	enum APIType { \
		API_CORE = 0, \
		API_EDITOR = 1, \
		API_EXTENSION = 2, \
		API_EDITOR_EXTENSION = 3, \
		API_NONE = 4, \
	}; \
	 \
	static PackedStringArray get_class_list() { \
		return ClassDBSingleton::get_singleton()->get_class_list(); \
	} \
	static PackedStringArray get_inheriters_from_class(const StringName &p_class) { \
		return ClassDBSingleton::get_singleton()->get_inheriters_from_class(p_class); \
	} \
	static StringName get_parent_class(const StringName &p_class) { \
		return ClassDBSingleton::get_singleton()->get_parent_class(p_class); \
	} \
	static bool class_exists(const StringName &p_class) { \
		return ClassDBSingleton::get_singleton()->class_exists(p_class); \
	} \
	static bool is_parent_class(const StringName &p_class, const StringName &p_inherits) { \
		return ClassDBSingleton::get_singleton()->is_parent_class(p_class, p_inherits); \
	} \
	static bool can_instantiate(const StringName &p_class) { \
		return ClassDBSingleton::get_singleton()->can_instantiate(p_class); \
	} \
	static Variant instantiate(const StringName &p_class) { \
		return ClassDBSingleton::get_singleton()->instantiate(p_class); \
	} \
	static ClassDB::APIType class_get_api_type(const StringName &p_class) { \
		return (ClassDB::APIType)ClassDBSingleton::get_singleton()->class_get_api_type(p_class); \
	} \
	static bool class_has_signal(const StringName &p_class, const StringName &p_signal) { \
		return ClassDBSingleton::get_singleton()->class_has_signal(p_class, p_signal); \
	} \
	static Dictionary class_get_signal(const StringName &p_class, const StringName &p_signal) { \
		return ClassDBSingleton::get_singleton()->class_get_signal(p_class, p_signal); \
	} \
	static TypedArray<Dictionary> class_get_signal_list(const StringName &p_class, bool p_no_inheritance = false) { \
		return ClassDBSingleton::get_singleton()->class_get_signal_list(p_class, p_no_inheritance); \
	} \
	static TypedArray<Dictionary> class_get_property_list(const StringName &p_class, bool p_no_inheritance = false) { \
		return ClassDBSingleton::get_singleton()->class_get_property_list(p_class, p_no_inheritance); \
	} \
	static StringName class_get_property_getter(const StringName &p_class, const StringName &p_property) { \
		return ClassDBSingleton::get_singleton()->class_get_property_getter(p_class, p_property); \
	} \
	static StringName class_get_property_setter(const StringName &p_class, const StringName &p_property) { \
		return ClassDBSingleton::get_singleton()->class_get_property_setter(p_class, p_property); \
	} \
	static Variant class_get_property(Object *p_object, const StringName &p_property) { \
		return ClassDBSingleton::get_singleton()->class_get_property(p_object, p_property); \
	} \
	static Error class_set_property(Object *p_object, const StringName &p_property, const Variant &p_value) { \
		return ClassDBSingleton::get_singleton()->class_set_property(p_object, p_property, p_value); \
	} \
	static Variant class_get_property_default_value(const StringName &p_class, const StringName &p_property) { \
		return ClassDBSingleton::get_singleton()->class_get_property_default_value(p_class, p_property); \
	} \
	static bool class_has_method(const StringName &p_class, const StringName &p_method, bool p_no_inheritance = false) { \
		return ClassDBSingleton::get_singleton()->class_has_method(p_class, p_method, p_no_inheritance); \
	} \
	static int32_t class_get_method_argument_count(const StringName &p_class, const StringName &p_method, bool p_no_inheritance = false) { \
		return ClassDBSingleton::get_singleton()->class_get_method_argument_count(p_class, p_method, p_no_inheritance); \
	} \
	static TypedArray<Dictionary> class_get_method_list(const StringName &p_class, bool p_no_inheritance = false) { \
		return ClassDBSingleton::get_singleton()->class_get_method_list(p_class, p_no_inheritance); \
	} \
	template <typename... Args> static Variant class_call_static(const StringName &p_class, const StringName &p_method, const Args &...p_args) { \
		return ClassDBSingleton::get_singleton()->class_call_static(p_class, p_method, p_args...); \
	} \
	static PackedStringArray class_get_integer_constant_list(const StringName &p_class, bool p_no_inheritance = false) { \
		return ClassDBSingleton::get_singleton()->class_get_integer_constant_list(p_class, p_no_inheritance); \
	} \
	static bool class_has_integer_constant(const StringName &p_class, const StringName &p_name) { \
		return ClassDBSingleton::get_singleton()->class_has_integer_constant(p_class, p_name); \
	} \
	static int64_t class_get_integer_constant(const StringName &p_class, const StringName &p_name) { \
		return ClassDBSingleton::get_singleton()->class_get_integer_constant(p_class, p_name); \
	} \
	static bool class_has_enum(const StringName &p_class, const StringName &p_name, bool p_no_inheritance = false) { \
		return ClassDBSingleton::get_singleton()->class_has_enum(p_class, p_name, p_no_inheritance); \
	} \
	static PackedStringArray class_get_enum_list(const StringName &p_class, bool p_no_inheritance = false) { \
		return ClassDBSingleton::get_singleton()->class_get_enum_list(p_class, p_no_inheritance); \
	} \
	static PackedStringArray class_get_enum_constants(const StringName &p_class, const StringName &p_enum, bool p_no_inheritance = false) { \
		return ClassDBSingleton::get_singleton()->class_get_enum_constants(p_class, p_enum, p_no_inheritance); \
	} \
	static StringName class_get_integer_constant_enum(const StringName &p_class, const StringName &p_name, bool p_no_inheritance = false) { \
		return ClassDBSingleton::get_singleton()->class_get_integer_constant_enum(p_class, p_name, p_no_inheritance); \
	} \
	static bool is_class_enum_bitfield(const StringName &p_class, const StringName &p_enum, bool p_no_inheritance = false) { \
		return ClassDBSingleton::get_singleton()->is_class_enum_bitfield(p_class, p_enum, p_no_inheritance); \
	} \
	static bool is_class_enabled(const StringName &p_class) { \
		return ClassDBSingleton::get_singleton()->is_class_enabled(p_class); \
	} \
	

#define CLASSDB_SINGLETON_VARIANT_CAST \
	VARIANT_ENUM_CAST(ClassDB::APIType); \
	


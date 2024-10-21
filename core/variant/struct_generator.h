/**************************************************************************/
/*  struct_generator.h                                                    */
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

#ifndef STRUCT_GENERATOR_H
#define STRUCT_GENERATOR_H

#include "core/variant/variant.h"

template <typename T>
class TypedArray;

template <typename T>
class Struct;

class Script;

/* The following set of macros let you seamlessly add reflection data to
 * a C++ struct or class so that it can be exposed as a Godot Struct.
 * The StructInfo struct below uses these macros and serves as a good
 * example of how these macros should be used.*/

// Goes at the top of every exposed C++ struct.
#define STRUCT_DECLARE(m_struct_name) using StructType = m_struct_name

// Creates the typedef for a non-pointer struct member, along with some helper functions.
// "Alias" means that the exposed name can differ from the internal name.
#define STRUCT_MEMBER_TYPEDEF_ALIAS(m_type, m_member_name, m_member_name_alias, m_default)                                                       \
	using Type = m_type;                                                                                                                         \
	_FORCE_INLINE_ static const StringName get_name() { return SNAME(m_member_name_alias); }                                                     \
	_FORCE_INLINE_ static Variant get_variant(const StructType &p_struct) { return to_variant(p_struct.m_member_name); }                         \
	_FORCE_INLINE_ static const Variant get_default_value_variant() { return to_variant(m_default); }                                            \
	_FORCE_INLINE_ static Type get(const StructType &p_struct) { return p_struct.m_member_name; }                                                \
	_FORCE_INLINE_ static Type get_default_value() { return m_default; }                                                                         \
	_FORCE_INLINE_ static void set_variant(StructType &p_struct, const Variant &p_variant) { p_struct.m_member_name = from_variant(p_variant); } \
	_FORCE_INLINE_ static void set(StructType &p_struct, const Type &p_value) { p_struct.m_member_name = p_value; }

// Shorter macro you can use when the exposed name is the same as the internal name.
#define STRUCT_MEMBER_TYPEDEF(m_type, m_member_name, m_default) \
	STRUCT_MEMBER_TYPEDEF_ALIAS(m_type, m_member_name, #m_member_name, m_default)

// Same as above, but for pointer members.
#define STRUCT_MEMBER_TYPEDEF_POINTER_ALIAS(m_type, m_member_name, m_member_name_alias, m_default)                                               \
	using Type = m_type *;                                                                                                                       \
	_FORCE_INLINE_ static const StringName get_name() { return SNAME(m_member_name_alias); }                                                     \
	_FORCE_INLINE_ static Variant get_variant(const StructType &p_struct) { return to_variant(p_struct.m_member_name); }                         \
	_FORCE_INLINE_ static const Variant get_default_value_variant() { return to_variant(m_default); }                                            \
	_FORCE_INLINE_ static Type get(const StructType &p_struct) { return p_struct.m_member_name; }                                                \
	_FORCE_INLINE_ static const m_type *get_default_value() { return m_default; }                                                                \
	_FORCE_INLINE_ static void set_variant(StructType &p_struct, const Variant &p_variant) { p_struct.m_member_name = from_variant(p_variant); } \
	_FORCE_INLINE_ static void set(StructType &p_struct, Type p_value) { p_struct.m_member_name = p_value; }

// Creates the typedef for a pointer struct member, along with some helper functions.
#define STRUCT_MEMBER_TYPEDEF_POINTER(m_type, m_member_name, m_default) \
	STRUCT_MEMBER_TYPEDEF_POINTER_ALIAS(m_type, m_member_name, #m_member_name, m_default)

// Creates all the reflection data for a struct member and stores it as static properties of a struct with
// the same name as the member. This particular macro only works for primitive Variant types. For more
// complicated types, such as class values, class pointers, or structs, use the corresponding macro below.
#define STRUCT_MEMBER_ALIAS(m_type, m_member_name, m_member_name_alias, m_default)                          \
	m_type m_member_name = m_default;                                                                       \
	struct m_member_name {                                                                                  \
		_FORCE_INLINE_ static m_type from_variant(const Variant &p_variant) { return p_variant; }           \
		_FORCE_INLINE_ static Variant to_variant(const m_type &p_value) { return p_value; }                 \
		STRUCT_MEMBER_TYPEDEF_ALIAS(m_type, m_member_name, m_member_name_alias, m_default);                 \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return to_variant(m_default).get_type(); } \
		_FORCE_INLINE_ static StringName get_type_name() { return StringName(); }                           \
		_FORCE_INLINE_ static const Script *get_script() { return nullptr; }                                \
	}

#define STRUCT_MEMBER(m_type, m_member_name, m_default) \
	STRUCT_MEMBER_ALIAS(m_type, m_member_name, #m_member_name, m_default)

// Macros that include _FROM allow you to customize the way the struct is converted from a Variant. You must
// implement the from_variant(const Variant &p_variant) function in the corresponding .cpp file.
#define STRUCT_MEMBER_FROM(m_type, m_member_name, m_default)                                                \
	m_type m_member_name = m_default;                                                                       \
	struct m_member_name {                                                                                  \
		static m_type from_variant(const Variant &p_variant);                                               \
		_FORCE_INLINE_ static Variant to_variant(const m_type &p_value) { return p_value; }                 \
		STRUCT_MEMBER_TYPEDEF(m_type, m_member_name, m_default);                                            \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return to_variant(m_default).get_type(); } \
		_FORCE_INLINE_ static StringName get_type_name() { return StringName(); }                           \
		_FORCE_INLINE_ static const Script *get_script() { return nullptr; }                                \
	}

// Macros that include _TO allow you to customize the way the struct is converted to a Variant. You must
// implement the Variant to_variant function in the corresponding .cpp file.
#define STRUCT_MEMBER_FROM_TO_ALIAS(m_type, m_member_name, m_member_name_alias, m_default)                  \
	m_type m_member_name = m_default;                                                                       \
	struct m_member_name {                                                                                  \
		static m_type from_variant(const Variant &p_variant);                                               \
		static Variant to_variant(const m_type &p_value);                                                   \
		STRUCT_MEMBER_TYPEDEF_ALIAS(m_type, m_member_name, m_member_name_alias, m_default);                 \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return to_variant(m_default).get_type(); } \
		_FORCE_INLINE_ static StringName get_type_name() { return StringName(); }                           \
		_FORCE_INLINE_ static const Script *get_script() { return nullptr; }                                \
	}

#define STRUCT_MEMBER_FROM_TO(m_type, m_member_name, m_default) \
	STRUCT_MEMBER_FROM_TO_ALIAS(m_type, m_member_name, #m_member_name, m_default)

#define STRUCT_MEMBER_CLASS_POINTER(m_type, m_member_name, m_default)                                                       \
	m_type *m_member_name = m_default;                                                                                      \
	struct m_member_name {                                                                                                  \
		_FORCE_INLINE_ static m_type *from_variant(const Variant &p_variant) { return Object::cast_to<m_type>(p_variant); } \
		_FORCE_INLINE_ static Variant to_variant(m_type *p_value) { return p_value; }                                       \
		STRUCT_MEMBER_TYPEDEF_POINTER(m_type, m_member_name, m_default);                                                    \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return Variant::OBJECT; }                                  \
		_FORCE_INLINE_ static StringName get_type_name() { return SNAME(#m_type); }                                         \
		_FORCE_INLINE_ static const Script *get_script() { return nullptr; }                                                \
	}

#define STRUCT_MEMBER_CLASS_VALUE(m_type, m_member_name, m_default)                               \
	m_type m_member_name = m_default;                                                             \
	struct m_member_name {                                                                        \
		_FORCE_INLINE_ static m_type from_variant(const Variant &p_variant) { return p_variant; } \
		_FORCE_INLINE_ static Variant to_variant(const m_type &p_value) { return p_value; }       \
		STRUCT_MEMBER_TYPEDEF(m_type, m_member_name, m_default);                                  \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return Variant::OBJECT; }        \
		_FORCE_INLINE_ static StringName get_type_name() { return SNAME(#m_type); }               \
		_FORCE_INLINE_ static const Script *get_script() { return nullptr; }                      \
	}

#define STRUCT_MEMBER_STRUCT(m_type, m_member_name, m_default)                                                    \
	m_type m_member_name = m_default;                                                                             \
	struct m_member_name {                                                                                        \
		_FORCE_INLINE_ static m_type from_variant(const Variant &p_variant) { return Struct<m_type>(p_variant); } \
		_FORCE_INLINE_ static Variant to_variant(const m_type &p_value) { return Struct<m_type>(p_value); }       \
		STRUCT_MEMBER_TYPEDEF(m_type, m_member_name, m_default);                                                  \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return Variant::ARRAY; }                         \
		_FORCE_INLINE_ static StringName get_type_name() { return m_type::get_struct_name(); }                    \
		_FORCE_INLINE_ static const Script *get_script() { return nullptr; }                                      \
	}

#define STRUCT_MEMBER_STRUCT_FROM_TO_ALIAS(m_type, m_member_name, m_member_name_alias, m_default) \
	m_type m_member_name = m_default;                                                             \
	struct m_member_name {                                                                        \
		static m_type from_variant(const Variant &p_variant);                                     \
		static Variant to_variant(const m_type &p_value);                                         \
		STRUCT_MEMBER_TYPEDEF_ALIAS(m_type, m_member_name, m_member_name_alias, m_default);       \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return Variant::ARRAY; }         \
		_FORCE_INLINE_ static StringName get_type_name() { return m_type::get_struct_name(); }    \
		_FORCE_INLINE_ static const Script *get_script() { return nullptr; }                      \
	}

#define STRUCT_MEMBER_STRUCT_FROM_TO(m_type, m_member_name, m_default) \
	STRUCT_MEMBER_STRUCT_FROM_TO_ALIAS(m_type, m_member_name, #m_member_name, m_default)

// Use after all the struct members have been declared to specialize the StructLayout Template
// and define some helper functions.
#define STRUCT_LAYOUT_ALIAS(m_struct_name, m_struct, ...) \
	static const StringName get_struct_name() {           \
		return SNAME(m_struct_name);                      \
	}                                                     \
	static const StructInfo &get_struct_info() {          \
		return Layout::get_struct_info();                 \
	}                                                     \
	using Layout = StructLayout<m_struct, __VA_ARGS__>;   \
	m_struct(const Dictionary &p_dict) {                  \
		Layout::fill_struct(p_dict, *this);               \
	}                                                     \
	m_struct(const Array &p_array) {                      \
		Layout::fill_struct(p_array, *this);              \
	}

// Most of the time, the exposed name of a struct follows the pattern "OwningClass.StructName".
#define STRUCT_LAYOUT(m_owner, m_struct, ...) STRUCT_LAYOUT_ALIAS(#m_owner "." #m_struct, m_struct, __VA_ARGS__)

template <typename StructType, typename... StructMembers>
struct StructLayout;

/* Represents the type data of both native and user Godot Structs.
 * StructInfo is itself exposed as a Godot Struct, so it serves as
 * a good example for how to expose other structs. */
struct StructInfo {
	STRUCT_DECLARE(StructInfo);
	STRUCT_MEMBER(StringName, name, StringName());
	STRUCT_MEMBER(int32_t, count, 0);
	STRUCT_MEMBER(Vector<StringName>, names, Vector<StringName>());
	STRUCT_MEMBER_FROM_TO(Vector<Variant::Type>, types, Vector<Variant::Type>());
	STRUCT_MEMBER(Vector<StringName>, type_names, Vector<StringName>());
	STRUCT_MEMBER_FROM_TO(Vector<const Script *>, scripts, Vector<const Script *>()); // wants to be Vector<Ref<Script>> but can't include Ref here.
	STRUCT_MEMBER(Vector<Variant>, default_values, Vector<Variant>());

	/* Normally, you would write
	 * STRUCT_LAYOUT(StructInfo, struct name, struct count, struct names, struct types, struct type_names, struct scripts, struct default_values);
	 * but we can't do that for StructInfo or it will create a circular dependency. */
	static const StringName get_struct_name() {
		return SNAME("StructInfo");
	}
	static const StructInfo &get_struct_info();
	using Layout = StructLayout<StructInfo, struct name, struct count, struct names, struct types, struct type_names, struct scripts, struct default_values>;
	StructInfo(const Dictionary &p_dict);
	StructInfo(const Array &p_array);

	StructInfo() {}
	StructInfo(const StringName &p_name, const int32_t p_count) :
			name(p_name), count(p_count) {
		names.resize(p_count);
		types.resize(p_count);
		type_names.resize(p_count);
		scripts.resize(p_count);
		default_values.resize(p_count);
	}
	StructInfo(const StringName &p_name, const int32_t p_count, const Vector<StringName> &p_names, const Vector<Variant::Type> &p_types, const Vector<StringName> &p_type_names, const Vector<const Script *> &p_scripts, const Vector<Variant> &p_default_values) :
			name(p_name),
			count(p_count),
			names(p_names),
			types(p_types),
			type_names(p_type_names),
			scripts(p_scripts),
			default_values(p_default_values) {}

	_FORCE_INLINE_ void set(int32_t p_index, const StringName &p_name, const Variant::Type &p_type, const StringName &p_type_name, const Script *p_script, const Variant &p_default_value) {
		names.write[p_index] = p_name;
		types.write[p_index] = p_type;
		type_names.write[p_index] = p_type_name;
		scripts.write[p_index] = p_script;
		default_values.write[p_index] = p_default_value;
	}

	_FORCE_INLINE_ bool operator==(const StructInfo &p_struct_info) const {
		return name == p_struct_info.name;
	}
	_FORCE_INLINE_ bool operator!=(const StructInfo &p_struct_info) const {
		return name != p_struct_info.name;
	}
	_FORCE_INLINE_ static bool is_compatible(const StructInfo *p_struct_info_1, const StructInfo *p_struct_info_2) {
		if (p_struct_info_1) {
			if (p_struct_info_2) {
				return *p_struct_info_1 == *p_struct_info_2;
			}
			return false;
		}
		return p_struct_info_2 == nullptr;
	}
};

/* The StructLayout template manages all the reflection data for a native struct. It automatically generates
 * functions for converting the C++ struct to and from a Godot Struct, Array, or Dictionary.
 * The StructType argument is expected to be a struct declared with STRUCT_DECLARE and the StructMember
 * arguments are expected to be the reflection structs created by any of the various STRUCT_MEMBER macros. */
template <typename StructType, typename... StructMembers>
struct StructLayout {
	static constexpr int32_t struct_member_count = sizeof...(StructMembers);
	_FORCE_INLINE_ static const StringName get_struct_name() {
		return StructType::get_struct_name();
	}
	_FORCE_INLINE_ static const StructInfo &get_struct_info() {
		static const Vector<StringName> names = { StructMembers::get_name()... };
		static const Vector<Variant::Type> types = { StructMembers::get_variant_type()... };
		static const Vector<StringName> type_names = { StructMembers::get_type_name()... };
		static const Vector<const Script *> scripts = { StructMembers::get_script()... };
		static const Vector<Variant> default_values = { StructMembers::get_default_value_variant()... };
		static const StructInfo info = StructInfo(get_struct_name(), struct_member_count, names, types, type_names, scripts, default_values);
		return info;
	}

	static Array to_array(const StructType &p_struct) {
		Array array;
		fill_array(array, p_struct);
		return array;
	}
	static void fill_array(Array &p_array, const StructType &p_struct) {
		p_array.resize(struct_member_count);
		Variant vals[struct_member_count] = { StructMembers::get_variant(p_struct)... };
		for (int32_t i = 0; i < struct_member_count; i++) {
			p_array.set(i, vals[i]);
		}
	}
	static void fill_struct(const Array &p_array, StructType &p_struct) {
		int32_t i = 0;
		int temp[] = { 0, (StructMembers::set_variant(p_struct, p_array.get(i)), i++, 0)... };
		(void)temp; // Suppress unused variable warning
	}
	static void fill_struct_array(Struct<StructType> &p_array, const StructType &p_struct);

	static Dictionary to_dict(const StructType &p_struct) {
		Dictionary dict;
		fill_dict(dict, p_struct);
		return dict;
	}
	static void fill_dict(Dictionary &p_dict, const StructType &p_struct) {
		int temp[] = { 0, (p_dict[StructMembers::get_name()] = StructMembers::get_variant(p_struct), 0)... };
		(void)temp; // Suppress unused variable warning
	}
	static void fill_struct(const Dictionary &p_dict, StructType &p_struct) {
		int temp[] = { 0, (StructMembers::set_variant(p_struct, p_dict[StructMembers::get_name()]), 0)... };
		(void)temp; // Suppress unused variable warning
	}

	template <typename T>
	_FORCE_INLINE_ static constexpr int get_member_index() {
		return sizeof...(StructMembers) - TypeFinder<T, StructMembers...>::remaining_count - 1;
	}

	// Provides random access member lookup for native Godot Structs.
	// It uses recursive types to force the compiler to perform a linear member search
	// so that member access is O(1) at runtime.
private:
	template <typename TypeToFind, typename... TypesToSearch>
	struct TypeFinder;

	template <typename TypeFound, typename... TypesRemaining>
	struct TypeFinder<TypeFound, TypeFound, TypesRemaining...> {
		static constexpr int remaining_count = sizeof...(TypesRemaining);
	};

	template <typename TypeNotFound>
	struct TypeFinder<TypeNotFound> {
		static constexpr int remaining_count = 0;
	};

	template <typename TypeToFind, typename TypeToCheck, typename... TypesRemaining>
	struct TypeFinder<TypeToFind, TypeToCheck, TypesRemaining...> {
		static constexpr int remaining_count = TypeFinder<TypeToFind, TypesRemaining...>::remaining_count;
	};
};

#endif // STRUCT_GENERATOR_H

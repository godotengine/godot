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

#define STRUCT_DECLARE(m_struct_name) using StructType = m_struct_name

#define STRUCT_MEMBER_TYPEDEF_ALIAS(m_type, m_member_name, m_member_name_alias, m_default)                                                       \
	using Type = m_type;                                                                                                                         \
	_FORCE_INLINE_ static const StringName get_name() { return SNAME(m_member_name_alias); }                                                     \
	_FORCE_INLINE_ static Variant get_variant(const StructType &p_struct) { return to_variant(p_struct.m_member_name); }                         \
	_FORCE_INLINE_ static const Variant get_default_value_variant() { return to_variant(m_default); }                                            \
	_FORCE_INLINE_ static Type get(const StructType &p_struct) { return p_struct.m_member_name; }                                                \
	_FORCE_INLINE_ static Type get_default_value() { return m_default; }                                                                         \
	_FORCE_INLINE_ static void set_variant(StructType &p_struct, const Variant &p_variant) { p_struct.m_member_name = from_variant(p_variant); } \
	_FORCE_INLINE_ static void set(StructType &p_struct, const m_type &p_value) { p_struct.m_member_name = p_value; }

#define STRUCT_MEMBER_TYPEDEF(m_type, m_member_name, m_default) \
	STRUCT_MEMBER_TYPEDEF_ALIAS(m_type, m_member_name, #m_member_name, m_default)

#define STRUCT_MEMBER_TYPEDEF_POINTER(m_type, m_member_name, m_default)                                                                          \
	using Type = m_type *;                                                                                                                       \
	_FORCE_INLINE_ static const StringName get_name() { return SNAME(#m_member_name); }                                                          \
	_FORCE_INLINE_ static Variant get_variant(const StructType &p_struct) { return to_variant(p_struct.m_member_name); }                         \
	_FORCE_INLINE_ static const Variant get_default_value_variant() { return to_variant(m_default); }                                            \
	_FORCE_INLINE_ static Type get(const StructType &p_struct) { return p_struct.m_member_name; }                                                \
	_FORCE_INLINE_ static const m_type *get_default_value() { return m_default; }                                                                \
	_FORCE_INLINE_ static void set_variant(StructType &p_struct, const Variant &p_variant) { p_struct.m_member_name = from_variant(p_variant); } \
	_FORCE_INLINE_ static void set(StructType &p_struct, Type p_value) { p_struct.m_member_name = p_value; }

#define STRUCT_MEMBER_PRIMITIVE_ALIAS(m_type, m_member_name, m_member_name_alias, m_default)                \
	m_type m_member_name = m_default;                                                                       \
	struct m_member_name {                                                                                  \
		_FORCE_INLINE_ static m_type from_variant(const Variant &p_variant) { return p_variant; }           \
		_FORCE_INLINE_ static Variant to_variant(const m_type &p_value) { return p_value; }                 \
		STRUCT_MEMBER_TYPEDEF_ALIAS(m_type, m_member_name, m_member_name_alias, m_default);                 \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return to_variant(m_default).get_type(); } \
		_FORCE_INLINE_ static const StringName get_class_name() { return StringName(); }                    \
		static const StructInfo *get_struct_member_info() { return nullptr; }                               \
	}

#define STRUCT_MEMBER_PRIMITIVE_FROM(m_type, m_member_name, m_default)                                      \
	m_type m_member_name = m_default;                                                                       \
	struct m_member_name {                                                                                  \
		static m_type from_variant(const Variant &p_variant);                                               \
		_FORCE_INLINE_ static Variant to_variant(const m_type &p_value) { return p_value; }                 \
		STRUCT_MEMBER_TYPEDEF(m_type, m_member_name, m_default);                                            \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return to_variant(m_default).get_type(); } \
		_FORCE_INLINE_ static const StringName get_class_name() { return StringName(); }                    \
		static const StructInfo *get_struct_member_info() { return nullptr; }                               \
	}

#define STRUCT_MEMBER_PRIMITIVE(m_type, m_member_name, m_default) \
	STRUCT_MEMBER_PRIMITIVE_ALIAS(m_type, m_member_name, #m_member_name, m_default)

#define STRUCT_MEMBER_PRIMITIVE_FROM_TO_ALIAS(m_type, m_member_name, m_member_name_alias, m_default)        \
	m_type m_member_name = m_default;                                                                       \
	struct m_member_name {                                                                                  \
		static m_type from_variant(const Variant &p_variant);                                               \
		static Variant to_variant(const m_type &p_value);                                                   \
		STRUCT_MEMBER_TYPEDEF_ALIAS(m_type, m_member_name, m_member_name_alias, m_default);                 \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return to_variant(m_default).get_type(); } \
		_FORCE_INLINE_ static const StringName get_class_name() { return StringName(); }                    \
		static const StructInfo *get_struct_member_info() { return nullptr; }                               \
	}

#define STRUCT_MEMBER_PRIMITIVE_FROM_TO(m_type, m_member_name, m_default) \
	STRUCT_MEMBER_PRIMITIVE_FROM_TO_ALIAS(m_type, m_member_name, #m_member_name, m_default)

#define STRUCT_MEMBER_CLASS_POINTER(m_type, m_member_name, m_default)                                                       \
	m_type *m_member_name = m_default;                                                                                      \
	struct m_member_name {                                                                                                  \
		_FORCE_INLINE_ static m_type *from_variant(const Variant &p_variant) { return Object::cast_to<m_type>(p_variant); } \
		_FORCE_INLINE_ static Variant to_variant(m_type *p_value) { return p_value; }                                       \
		STRUCT_MEMBER_TYPEDEF_POINTER(m_type, m_member_name, m_default);                                                    \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return Variant::OBJECT; }                                  \
		_FORCE_INLINE_ static const StringName get_class_name() { return SNAME(#m_type); }                                  \
		static const StructInfo *get_struct_member_info() { return nullptr; }                                               \
	}

#define STRUCT_MEMBER_CLASS_VALUE(m_type, m_member_name, m_default)                               \
	m_type m_member_name = m_default;                                                             \
	struct m_member_name {                                                                        \
		_FORCE_INLINE_ static m_type from_variant(const Variant &p_variant) { return p_variant; } \
		_FORCE_INLINE_ static Variant to_variant(const m_type &p_value) { return p_value; }       \
		STRUCT_MEMBER_TYPEDEF(m_type, m_member_name, m_default);                                  \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return Variant::OBJECT; }        \
		_FORCE_INLINE_ static const StringName get_class_name() { return SNAME(#m_type); }        \
		static const StructInfo *get_struct_member_info() { return nullptr; }                     \
	}

#define STRUCT_MEMBER_STRUCT(m_type, m_member_name, m_default)                                                          \
	m_type m_member_name = m_default;                                                                                   \
	struct m_member_name {                                                                                              \
		_FORCE_INLINE_ static m_type from_variant(const Variant &p_variant) { return Struct<m_type>(p_variant); }       \
		_FORCE_INLINE_ static Variant to_variant(const m_type &p_value) { return Struct<m_type>(p_value); }             \
		STRUCT_MEMBER_TYPEDEF(m_type, m_member_name, m_default);                                                        \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return Variant::ARRAY; }                               \
		_FORCE_INLINE_ static const StringName get_class_name() { return StringName(); }                                \
		_FORCE_INLINE_ static const StructInfo *get_struct_member_info() { return &m_type::Layout::get_struct_info(); } \
	}

#define STRUCT_MEMBER_STRUCT_FROM_TO_ALIAS(m_type, m_member_name, m_member_name_alias, m_default) \
	m_type m_member_name = m_default;                                                             \
	struct m_member_name {                                                                        \
		static m_type from_variant(const Variant &p_variant);                                     \
		static Variant to_variant(const m_type &p_value);                                         \
		STRUCT_MEMBER_TYPEDEF_ALIAS(m_type, m_member_name, m_member_name_alias, m_default);       \
		_FORCE_INLINE_ static Variant::Type get_variant_type() { return Variant::ARRAY; }         \
		_FORCE_INLINE_ static const StringName get_class_name() { return StringName(); }          \
		_FORCE_INLINE_ static const StructInfo *get_struct_member_info() {                        \
			return &m_type::Layout::get_struct_info();                                            \
		}                                                                                         \
	}

#define STRUCT_MEMBER_STRUCT_FROM_TO(m_type, m_member_name, m_default) \
	STRUCT_MEMBER_STRUCT_FROM_TO_ALIAS(m_type, m_member_name, #m_member_name, m_default)

#define STRUCT_LAYOUT_WITH_NAME(m_struct_name, m_struct, ...) \
	static const StringName get_struct_name() {               \
		return SNAME(m_struct_name);                          \
	}                                                         \
	static const StructInfo &get_struct_info() {              \
		return Layout::get_struct_info();                     \
	}                                                         \
	using Layout = StructLayout<m_struct, __VA_ARGS__>;       \
	m_struct(const Dictionary &p_dict) {                      \
		Layout::fill_struct(p_dict, *this);                   \
	}                                                         \
	m_struct(const Array &p_array) {                          \
		Layout::fill_struct(p_array, *this);                  \
	}

#define STRUCT_LAYOUT_OWNER(m_owner, m_struct, ...) STRUCT_LAYOUT_WITH_NAME(#m_owner "." #m_struct, m_struct, __VA_ARGS__)

#define STRUCT_LAYOUT(m_struct, ...) STRUCT_LAYOUT_WITH_NAME(#m_struct, m_struct, __VA_ARGS__)

struct StructInfo {
	StringName name = StringName();
	int32_t count = 0;

	Vector<StringName> names;
	Vector<Variant::Type> types;
	Vector<StringName> class_names;
	Vector<const StructInfo *> struct_member_infos;
	Vector<Variant> default_values;

	StructInfo() {};
	StructInfo(const StringName &p_name, const int32_t p_count) :
			name(p_name), count(p_count) {
		names.resize(p_count);
		types.resize(p_count);
		class_names.resize(p_count);
		struct_member_infos.resize(p_count);
		default_values.resize(p_count);
	}
	StructInfo(const StringName &p_name, const int32_t p_count, const Vector<StringName> &p_names, const Vector<Variant::Type> &p_types, const Vector<StringName> &p_class_names, const Vector<const StructInfo *> &p_struct_member_infos, const Vector<Variant> &p_default_values) :
			name(p_name),
			count(p_count),
			names(p_names),
			types(p_types),
			class_names(p_class_names),
			struct_member_infos(p_struct_member_infos),
			default_values(p_default_values) {};

	Dictionary to_dict() const;

	_FORCE_INLINE_ void set(int32_t p_index, const StringName &p_name, const Variant::Type &p_type, const StringName &p_class_name, const StructInfo *p_struct_member_info, const Variant &p_default_value) {
		names.write[p_index] = p_name;
		types.write[p_index] = p_type;
		class_names.write[p_index] = p_class_name;
		struct_member_infos.write[p_index] = p_struct_member_info;
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

template <typename StructType, typename... StructMembers>
struct StructLayout {
	static constexpr int32_t struct_member_count = sizeof...(StructMembers);
	_FORCE_INLINE_ static const StringName get_struct_name() {
		return StructType::get_struct_name();
	}
	_FORCE_INLINE_ static const StructInfo &get_struct_info() {
		static const Vector<StringName> names = { StructMembers::get_name()... };
		static const Vector<Variant::Type> types = { StructMembers::get_variant_type()... };
		static const Vector<StringName> class_names = { StructMembers::get_class_name()... };
		static const Vector<const StructInfo *> struct_member_infos = { StructMembers::get_struct_member_info()... };
		static const Vector<Variant> default_values = { StructMembers::get_default_value_variant()... };
		static const StructInfo info = StructInfo(get_struct_name(), struct_member_count, names, types, class_names, struct_member_infos, default_values);
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

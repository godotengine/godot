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

#undef STRUCT_MEMBER
#undef STRUCT_LAYOUT

#define STRUCT_MEMBER_PRIMITIVE(m_struct_name, m_struct_member, m_type, m_variant_type, m_member_name, m_default) \
	m_type m_member_name = m_default;                                                                             \
	struct m_struct_member {                                                                                      \
		using type = m_type;                                                                                      \
		static constexpr const char *name = #m_member_name;                                                       \
		static constexpr m_type m_struct_name::*ptr = &m_struct_name::m_member_name;                              \
		static constexpr Variant::Type variant_type = m_variant_type;                                             \
		static constexpr const char *class_name = "";                                                             \
		static constexpr m_type default_value = m_default;                                                        \
		static constexpr StructInfo2 *struct_member_info = nullptr;                                                      \
	}

#define STRUCT_MEMBER_CLASS(m_struct_name, m_struct_member, m_type, m_class_name, m_member_name, m_default) \
	m_type m_member_name = m_default;                                                                       \
	struct m_struct_member {                                                                                \
		using type = m_type;                                                                                \
		static constexpr const char *name = #m_member_name;                                                 \
		static constexpr m_type m_struct_name::*ptr = &m_struct_name::m_member_name;                        \
		static constexpr Variant::Type variant_type = Variant::OBJECT;                                      \
		static constexpr const char *class_name = m_class_name;                                             \
		static constexpr m_type default_value = m_default;                                                  \
		static constexpr StructInfo2 *struct_info = nullptr;                                                \
	}

#define STRUCT_MEMBER_STRUCT(m_struct_name, m_struct_member, m_type, m_member_name, m_default) \
	m_type m_member_name = m_default;                                                          \
	struct m_struct_member {                                                                   \
		using type = m_type;                                                                   \
		static constexpr const char *name = #m_member_name;                                    \
		static constexpr m_type m_struct_name::*ptr = &m_struct_name::m_member_name;           \
		static constexpr Variant::Type variant_type = Variant::ARRAY;                          \
		static constexpr const char *class_name = m_class_name;                                \
		static constexpr m_type default_value = m_default;                                     \
		static constexpr StructInfo2 *struct_info = &m_type::get_struct_info();                \
	}

#define STRUCT_LAYOUT(m_struct, ...)                      \
	static constexpr const char *struct_name = #m_struct; \
	using Layout = StructLayout<m_struct, __VA_ARGS__>;   \
	m_struct(){};                                         \
	m_struct(const Struct<m_struct> &p_struct_array) {    \
		Layout::fill_struct(p_struct_array, *this);       \
	}
//	operator Struct<m_struct>() const {                   \
//		Struct<m_struct> ret;                             \
//		Layout::fill_array(ret, *this);                   \
//		return ret;                                       \
//	}

struct StructInfo2 {
	StringName name = StringName();
	uint32_t count = 0;

	StringName *names = nullptr;
	Variant::Type *types = nullptr;
	StringName *class_names = nullptr;
	StructInfo2 *struct_member_info = nullptr;
	Variant *default_values = nullptr;

	StructInfo2(){};
	StructInfo2(const StringName &p_name, uint32_t p_count, StringName *p_names, Variant::Type *p_types, StringName *p_class_names, StructInfo2 *p_struct_member_info, Variant *p_default_values) {
		name = p_name;
		count = p_count;
		names = p_names;
		types = p_types;
		class_names = p_class_names;
		struct_member_info = p_struct_member_info;
		default_values = p_default_values;
	};

	_FORCE_INLINE_ const bool operator==(const StructInfo2 &p_struct_info) const {
		return name == p_struct_info.name;
	}
	_FORCE_INLINE_ const bool operator!=(const StructInfo2 &p_struct_info) const {
		return name != p_struct_info.name;
	}
};

template <typename StructType, typename... StructMembers>
struct StructLayout {
	static constexpr uint32_t struct_member_count = sizeof...(StructMembers);
	_FORCE_INLINE_ static constexpr uint32_t get_struct_member_count() {
		return struct_member_count;
	}
	_FORCE_INLINE_ static const StringName get_struct_name() {
		static const StringName struct_name = SNAME(StructType::struct_name);
		return struct_name;
	}
	_FORCE_INLINE_ static const StructInfo2 get_struct_info() {
		static const StringName names[struct_member_count] = { StructMembers::name... };
		static const Variant::Type types[struct_member_count] = { StructMembers::variant_type... };
		static const StringName class_names[struct_member_count] = { SNAME(StructMembers::class_name)... };
		static const StructInfo2 struct_member_infos[struct_member_count] = { StructMembers::struct_member_info... };
		static const Variant default_values[struct_member_count] = { Variant(StructMembers::default_value)... };
		static const StructInfo2 info = StructInfo2(get_struct_name(), struct_member_count, names, types, class_names, struct_member_infos, default_values);
		return info;
	}

	static const Array to_array(const StructType &p_struct) {
		Array array;
		fill_array(array, p_struct);
		return array;
	}
	static constexpr void fill_array(Array &p_array, const StructType &p_struct) {
		p_array.resize(struct_member_count);
		Variant vals[struct_member_count] = { Variant(get_member_value<StructMembers>(p_struct))... };
		for (uint32_t i = 0; i < struct_member_count; i++) {
			p_array.set(i, vals[i]);
		}
	}
	static constexpr void fill_struct_array(Struct<StructType> &p_array, const StructType &p_struct) {
		int dummy[] = { 0, (p_array.set_member_value<StructMembers>(get_member_value<StructMembers>(p_struct)), 0)... };
		(void)dummy; // Suppress unused variable warning
	}
	static constexpr void fill_struct(const Struct<StructType> &p_struct_array, StructType &p_struct) {
		int dummy[] = { 0, (set_member_value<StructMembers>(p_struct, p_struct_array.get_member_value<StructMembers>()), 0)... };
		(void)dummy; // Suppress unused variable warning
	}

	template <typename StructMember>
	_FORCE_INLINE_ static constexpr StructMember::type get_member_value(const StructType &p_struct) {
		return p_struct.*StructMember::ptr;
	}
	template <typename StructMember>
	_FORCE_INLINE_ static constexpr void set_member_value(StructType &p_struct, const StructMember::type &p_value) {
		p_struct.*StructMember::ptr = p_value;
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

struct BasicStruct {
	STRUCT_MEMBER_PRIMITIVE(BasicStruct, IntVal, int, Variant::INT, int_val, 4);
	STRUCT_MEMBER_PRIMITIVE(BasicStruct, FloatVal, float, Variant::FLOAT, float_val, 5.5f);
	STRUCT_LAYOUT(BasicStruct, IntVal, FloatVal);
};

//BasicStruct basic_struct;
//Struct<BasicStruct> basic_struct_array;

#endif //STRUCT_GENERATOR_H

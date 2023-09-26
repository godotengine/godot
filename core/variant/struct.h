/**************************************************************************/
/*  struct.h                                                              */
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

#ifndef STRUCT_H
#define STRUCT_H

#include "core/string/string_name.h"
#include "core/variant/variant.h"

class Array;

struct StructMember {
	StringName name;
	Variant::Type type; // TODO: should this be a union of type and struct_name?
	StringName class_name;
	Variant default_value;

	StructMember(const StringName &p_name = StringName(), const Variant::Type p_type = Variant::NIL, const StringName &p_class_name = StringName(), const Variant &p_default_value = Variant()) {
		name = p_name;
		type = p_type;
		class_name = p_class_name;
		default_value = p_default_value;
	}
};

#define STRUCT_MEMBER(m_name, m_type) StructMember(SNAME(m_name), m_type)
#define STRUCT_CLASS_MEMBER(m_name, m_class) StructMember(SNAME(m_name), Variant::OBJECT, m_class)

// TODO: is there a way to define this so that the member count doesn't have to be passed?
#define STRUCT_LAYOUT(m_struct, m_name, ...)                                                 \
	struct m_struct {                                                                        \
		_FORCE_INLINE_ static const StringName get_class() {                                 \
			return SNAME(#m_struct);                                                         \
		}                                                                                    \
		_FORCE_INLINE_ static const StringName get_name() {                                  \
			return SNAME(m_name);                                                            \
		}                                                                                    \
		_FORCE_INLINE_ static const StructMember &get_member(uint32_t p_index) {             \
			static const StructMember members[] = { __VA_ARGS__ };                           \
			static constexpr uint32_t member_count = sizeof(members) / sizeof(StructMember); \
			CRASH_BAD_INDEX(p_index, member_count);                                          \
			return members[p_index];                                                         \
		}                                                                                    \
		_FORCE_INLINE_ static const uint32_t get_member_count() {                            \
			static const StructMember members[] = { __VA_ARGS__ };                           \
			static constexpr uint32_t member_count = sizeof(members) / sizeof(StructMember); \
			return member_count;                                                             \
		}                                                                                    \
	};

//// TODO: A different version I was playing around with
//#define STRUCT_LAYOUT(m_name, m_member_count, ...)                                                                                          \
//	struct m_name {                                                                                                                         \
//		static const uint32_t member_count = m_member_count;                                                                                \
//		_FORCE_INLINE_ static const StructInfo get_struct_info() {                                                                       \
//			static const StructMember members[member_count] = { __VA_ARGS__ };                                                                     \
//			StringName names[member_count];                                                                                                 \
//			uint32_t types[member_count];                                                                                                   \
//			StringName class_names[member_count];                                                                                                      \
//			Variant default_values[member_count];                                                                                                   \
//			for (uint32_t i = 0; i < member_count; i++) {                                                                                   \
//				StructMember member = members[i];                                                                                            \
//				names[i] = member.name;                                                                                                     \
//				types[i] = member.type;                                                                                                     \
//				class_names[i] = member.class_name;                                                                                         \
//				default_values[i] = member.default_value;                                                                                   \
//			}                                                                                                                               \
//			static const StructInfo struct_info = StructInfo(StringName(#m_name), member_count, names, types, class_names, default_values); \
//			return struct_info;                                                                                                             \
//		}                                                                                                                                   \
//	};

//// TODO: Another different version
//#define STRUCT_LAYOUT(m_name, m_member_count, ...)                             \
//	struct m_name {                                                            \
//		static const uint32_t member_count = m_member_count;                   \
//		_FORCE_INLINE_ static const StructMember *get_members() {              \
//			static const StructMember members[member_count] = { __VA_ARGS__ }; \
//			return members;                                                    \
//		}                                                                      \
//		_FORCE_INLINE_ static const StringName *get_member_names() {           \
//			const StructMember members[member_count] = get_members();          \
//			StringName member_names[member_count];                             \
//			for (uint32_t i = 0; i < member_count; i++) {                      \
//				member_names[i] = members[i].name;                             \
//			}                                                                  \
//			return member_names;                                               \
//		}                                                                      \
//		_FORCE_INLINE_ static const uint32_t *get_member_types() {             \
//			const StructMember members[member_count] = get_members();          \
//			uint32_t member_types[member_count];                               \
//			for (uint32_t i = 0; i < member_count; i++) {                      \
//				member_types[i] = members[i].type;                             \
//			}                                                                  \
//			return member_types;                                               \
//		}                                                                      \
//		_FORCE_INLINE_ static const StringName *get_member_class_names() {     \
//			const StructMember members[member_count] = get_members();          \
//			StringName member_class_names[member_count];                       \
//			for (uint32_t i = 0; i < member_count; i++) {                      \
//				member_class_names[i] = members[i].class_name;                 \
//			}                                                                  \
//			return member_class_names;                                         \
//		}                                                                      \
//		_FORCE_INLINE_ static const Variant *get_member_default_values() {     \
//			const StructMember members[member_count] = get_members();          \
//			Variant member_default_values[member_count];                       \
//			for (uint32_t i = 0; i < member_count; i++) {                      \
//				member_default_values[i] = members[i].default_value;           \
//			}                                                                  \
//			return member_default_values;                                      \
//		}                                                                      \
//	};

template <class T>
class Struct : public Array {
public:
	_FORCE_INLINE_ void operator=(const Array &p_array) {
		ERR_FAIL_COND_MSG(!is_same_typed(p_array), "Cannot assign a Struct from array with a different format.");
		_ref(p_array);
	}
	_FORCE_INLINE_ Struct(const Variant &p_variant) :
			Array(Array(p_variant), T::get_member_count(), T::get_name(), T::get_member) {
	}
	_FORCE_INLINE_ Struct(const Array &p_array) :
			Array(p_array, T::get_member_count(), T::get_name(), T::get_member) {
	}
	_FORCE_INLINE_ Struct() :
			Array(T::get_member_count(), T::get_name(), T::get_member) {
	}
};

#endif // STRUCT_H

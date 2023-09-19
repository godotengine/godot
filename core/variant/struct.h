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

#include "array.h"

struct StructMember {
	StringName name;
	Variant::Type type; // TODO: should this be a union of type and class_name?
	StringName class_name;

	StructMember(const StringName &p_name, const Variant::Type p_type, const StringName &p_class_name = StringName()) {
		name = p_name;
		type = p_type;
		class_name = p_class_name;
	}
};

#define STRUCT_MEMBER(m_name, m_type) StructMember(SNAME(m_name), m_type)
#define STRUCT_CLASS_MEMBER(m_name, m_class) StructMember(SNAME(m_name), Variant::OBJECT, m_class)

#define STRUCT_LAYOUT(m_name,...) \
struct m_name { \
	static constexpr uint32_t member_count = GET_ARGUMENT_COUNT;\
	_FORCE_INLINE_ static const StructMember& get_member(uint32_t p_index) {\
		CRASH_BAD_INDEX(p_index,member_count)\
		static StructMember members[member_count]={ __VA_ARGS__ };\
		return members[p_index];\
	}\
};

template <class T>
class Struct : public Array {
public:
	typedef Type T;

	_FORCE_INLINE_ void operator=(const Array &p_array) {
		ERR_FAIL_COND_MSG(!is_same_typed(p_array), "Cannot assign a Struct from array with a different format.");
		_ref(p_array);
	}
	_FORCE_INLINE_ Struct(const Variant &p_variant) :
			Array(T::member_count, T::get_member,Array(p_variant)) {
	}
	_FORCE_INLINE_ Struct(const Array &p_array) :
			Array(T::member_count, T::get_member,p_array) {
	}
	_FORCE_INLINE_ Struct() {
		Array(T::member_count, T::get_member) {
		}
	};

#endif // STRUCT_H

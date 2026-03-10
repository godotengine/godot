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

#pragma once

#include "core/variant/array.h"
#include "core/variant/variant.h"

struct StructMember {
	StringName name;
	StringName class_name;
	Variant::Type type;
	Variant default_value;
	Variant script;

	StructMember(const StringName &p_name, Variant::Type p_type, const Variant &p_default_value = Variant(), const StringName &p_class_name = StringName(), const Variant &p_script = Variant()) {
		name = p_name;
		type = p_type;
		default_value = p_default_value;
		class_name = p_class_name;
		script = p_script;
	}
};

#define STRUCT_MEMBER(m_name, m_type, ...) StructMember(SNAME(m_name), m_type, ##__VA_ARGS__)
#define STRUCT_CLASS_MEMBER(m_name, m_class) StructMember(SNAME(m_name), Variant::OBJECT, Variant(), m_class)

#define STRUCT_LAYOUT(m_class, m_name, ...) \
	struct m_name { \
		_FORCE_INLINE_ static StringName get_class() { return SNAME(#m_class); } \
		_FORCE_INLINE_ static StringName get_name() { return SNAME(#m_name); } \
		_FORCE_INLINE_ static uint32_t get_member_count() { \
			static const StructMember members[] = { __VA_ARGS__ }; \
			return std_size(members); \
		} \
		_FORCE_INLINE_ static const StructMember *get_members() { \
			static const StructMember members[] = { __VA_ARGS__ }; \
			return members; \
		} \
		_FORCE_INLINE_ static const StructMember &get_member(uint32_t p_index) { \
			CRASH_BAD_INDEX(p_index, get_member_count()); \
			return get_members()[p_index]; \
		} \
	};

template <class T>
class Struct : public Array {
public:
	typedef T Layout;

	_FORCE_INLINE_ void operator=(const Array &p_array) { Array::operator=(p_array); }
	_FORCE_INLINE_ Struct(const Variant &p_variant) : Array(T::get_member_count(), T::get_member, Array(p_variant)) {}
	_FORCE_INLINE_ Struct(const Array &p_array) : Array(T::get_member_count(), T::get_member, p_array) {}
	_FORCE_INLINE_ Struct() : Array(T::get_member_count(), T::get_member) {}
};

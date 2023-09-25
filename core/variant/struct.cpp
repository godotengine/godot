/**************************************************************************/
/*  struct.cpp                                                            */
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

#include "core/variant/struct.h"

StructInfo::StructInfo(const StringName &p_name, uint32_t p_count, const LocalVector<StringName> &p_names, const LocalVector<Variant::Type> &p_types, const LocalVector<StringName> &p_class_names, const LocalVector<Variant> &p_default_values) {
	name = p_name;
	count = p_count;
	names = p_names;
	types = p_types;
	class_names = p_class_names;
	default_values = p_default_values;
}

StructInfo::StructInfo(const StringName &p_name, uint32_t p_count, const StructMember *p_members) {
	name = p_name;
	count = p_count;
	names.resize(count);
	types.resize(count);
	class_names.resize(count);
	default_values.resize(count);
	for (uint32_t i = 0; i < p_count; i++) {
		StructMember member = p_members[i];
		names[i] = member.name;
		types[i] = member.type;
		class_names[i] = member.class_name;
		default_values[i] = member.default_value;
	}
}

/**************************************************************************/
/*  struct_generator.cpp                                                  */
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

#include "struct_generator.h"

#include "core/object/script_language.h"
#include "core/variant/struct.h"
#include "core/variant/typed_array.h"

Variant StructInfo::types::to_variant(const Vector<Variant::Type> &p_value) {
	PackedInt32Array packed_types;
	const Vector<Variant::Type>::Size size = p_value.size();
	packed_types.resize(size);
	for (int i = 0; i < size; i++) {
		packed_types.write[i] = p_value[i];
	}
	return packed_types;
}

Vector<Variant::Type> StructInfo::types::from_variant(const Variant &p_variant) {
	const PackedInt32Array &packed_types = p_variant;
	Vector<Variant::Type> types;
	const PackedInt32Array::Size size = packed_types.size();
	types.resize(size);
	for (int i = 0; i < size; i++) {
		types.write[i] = static_cast<Variant::Type>(packed_types[i]);
	}
	return types;
}

Variant StructInfo::scripts::to_variant(const Vector<const Script *> &p_value) {
	TypedArray<Script> scripts;
	const Vector<const Script *>::Size size = p_value.size();
	scripts.resize(size);
	for (int i = 0; i < size; i++) {
		scripts[i] = p_value[i];
	}
	return scripts;
}

Vector<const Script *> StructInfo::scripts::from_variant(const Variant &p_variant) {
	const TypedArray<Script> &script_array = p_variant;
	Vector<const Script *> script_vector;
	const int size = script_array.size();
	script_vector.resize(size);
	for (int i = 0; i < size; i++) {
		script_vector.ptrw()[i] = static_cast<Ref<Script>>(script_array[i]).ptr();
	}
	return script_vector;
}

// The following three methods need to be .cpp to avoid circular dependency between StructLayout and StructInfo
const StructInfo &StructInfo::get_struct_info() {
	return Layout::get_struct_info();
}

StructInfo::StructInfo(const Dictionary &p_dict) {
	Layout::fill_struct(p_dict, *this);
}

StructInfo::StructInfo(const Array &p_array) {
	Layout::fill_struct(p_array, *this);
}

// Needs to be in .cpp so struct.h can be included
template <typename StructType, typename... StructMembers>
void StructLayout<StructType, StructMembers...>::fill_struct_array(Struct<StructType> &p_array, const StructType &p_struct) {
	int dummy[] = { 0, (p_array.template set_member_value<StructMembers>(StructMembers::get_variant(p_struct)), 0)... };
	(void)dummy; // Suppress unused variable warning
}

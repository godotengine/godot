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

#include "core/variant/struct.h"
#include "core/variant/typed_array.h"

Dictionary StructInfo::to_dict() const {
	Dictionary dict;
	dict["name"] = name;
	dict["count"] = count;
	dict["names"] = names;
	PackedInt32Array packed_types;
	packed_types.resize(count);
	for (int i = 0; i < count; i++) {
		packed_types.write[i] = types[i];
	}
	dict["types"] = packed_types;
	dict["class_names"] = class_names;
	TypedArray<Dictionary> member_info_dictionaries;
	member_info_dictionaries.resize(count);
	for (int i = 0; i < count; i++) {
		const StructInfo *member_info = struct_member_infos[i];
		// TODO: Recursion limit?
		member_info_dictionaries[i] = member_info ? member_info->to_dict() : Dictionary();
	}
	dict["struct_member_infos"] = member_info_dictionaries;
	dict["default_values"] = default_values;
	return dict;
}

// Needs to be in .cpp so struct.h can be included
template <typename StructType, typename... StructMembers>
void StructLayout<StructType, StructMembers...>::fill_struct_array(Struct<StructType> &p_array, const StructType &p_struct) {
	int dummy[] = { 0, (p_array.template set_member_value<StructMembers>(StructMembers::get_variant(p_struct)), 0)... };
	(void)dummy; // Suppress unused variable warning
}

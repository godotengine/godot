/**************************************************************************/
/*  variant_struct_native.cpp                                             */
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

#include "variant_struct_native.h"

#include "core/object/class_db.h"

HashMap<StringName, StructDefinition *> in_built_struct_definitions;
Vector<StructDefinition **> in_built_struct_pointers;

void StructDefinition::_register_native_definition(StructDefinition **p_definition) {
	in_built_struct_definitions.insert((*p_definition)->qualified_name, *p_definition);
	in_built_struct_pointers.push_back(p_definition);
	StructDefinition::_register_struct_definition(*p_definition, false);
}
void unregister_inbuilt_data_structures() {
	for (StructDefinition **E : in_built_struct_pointers) {
		memdelete(*E);
		*E = nullptr;
	}
	in_built_struct_definitions.clear();
	in_built_struct_pointers.clear();
}

const StructDefinition *StructDefinition::get_native(const StringName &p_name) {
	return in_built_struct_definitions.get(p_name);
}

void StructDefinition::unregister_native_types() {
	unregister_inbuilt_data_structures();
}

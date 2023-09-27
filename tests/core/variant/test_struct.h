/**************************************************************************/
/*  test_struct.h                                                         */
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

#ifndef TEST_STRUCT_H
#define TEST_STRUCT_H

#include "core/variant/array.h"
#include "core/variant/struct.h"
#include "scene/main/node.h"
#include "tests/test_macros.h"
#include "tests/test_tools.h"

namespace TestStruct {

TEST_CASE("[Struct] PropertyInfo") {
	Node *my_node = memnew(Node);
	List<PropertyInfo> list;
	my_node->get_property_list(&list);
	PropertyInfo info = list[0];
	print_line((Dictionary)info);

	Struct<PropertyInfoLayout> prop = my_node->_get_property_struct(0);
	prop.set_named(SNAME("name"), info.name);
	prop.set_named(SNAME("class_name"), info.class_name);
	prop.set_named(SNAME("type"), info.type);
	prop.set_named(SNAME("hint"), info.hint);
	prop.set_named(SNAME("hint_string"), info.hint_string);
	prop.set_named(SNAME("usage"), info.usage);

	SUBCASE("Equality") {
		CHECK_EQ(info.name, String(prop.get_named(SNAME("name"))));
		CHECK_EQ(info.class_name, StringName(prop.get_named(SNAME("class_name"))));
		CHECK_EQ(info.type, (Variant::Type)(int)prop.get_named(SNAME("type")));
		CHECK_EQ(info.hint, (PropertyHint)(int)prop.get_named(SNAME("hint")));
		CHECK_EQ(info.hint_string, String(prop.get_named(SNAME("name"))));
		CHECK_EQ(info.usage, (PropertyUsageFlags)(int)prop.get_named(SNAME("usage")));
	}

	SUBCASE("Duplication") {
		Variant var = prop;
		CHECK_EQ(var.get_type(), Variant::ARRAY);
		Variant var_dup = prop.duplicate();
		CHECK_EQ(var_dup.get_type(), Variant::ARRAY);

		print_line(var);
		print_line(var_dup);
	}

	SUBCASE("Type Validation") {
		CHECK(prop.is_same_typed(prop));
		CHECK(prop.is_same_typed((Variant)prop));
		Variant var = prop;
		Struct<PropertyInfoLayout> prop2 = var;
		print_line(prop2);

		CHECK_THROWS(prop.set_named(SNAME("name"), 4));
		CHECK_NOTHROW(prop.set_named(SNAME("name"), "Node")); // TODO: not sure if these tests are working correctly
	}
}

TEST_CASE("[Struct] ClassDB") {
	List<StructMember> ls;
	::ClassDB::get_struct_members(SNAME("Object"), SNAME("PropertyInfo"), &ls);
	for (const StructMember &E : ls) {
		print_line(vformat("name: %s, type: %s, class_name: %s, default: %s.", E.name, E.type, E.class_name, E.default_value));
	}
}

} // namespace TestStruct

#endif // TEST_STRUCT_H

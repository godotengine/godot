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
#include "core/variant/variant.h"
#include "scene/main/node.h"
#include "tests/test_macros.h"
#include "tests/test_tools.h"

namespace TestStruct {

TEST_CASE("[Struct] PropertyInfo") {
	Node *my_node = memnew(Node);
	List<PropertyInfo> list;
	my_node->get_property_list(&list);
	PropertyInfo info = list[0];
	CHECK_EQ((((Variant)(Dictionary)info)).stringify(), "{ \"name\": \"Node\", \"class_name\": &\"\", \"type\": 0, \"hint\": 0, \"hint_string\": \"Node\", \"usage\": 128 }");
	CHECK_EQ((((Variant)(Array)info)).stringify(), "[name: \"Node\", class_name: &\"\", type: 0, hint: 0, hint_string: \"Node\", usage: 128]");

	Struct<PropertyInfoLayout> prop = my_node->_get_property_struct(0);
	CHECK_EQ(((Variant)prop).stringify(), "[name: \"Node\", class_name: &\"\", type: 0, hint: 0, hint_string: \"Node\", usage: 128]");

	SUBCASE("Equality") {
		CHECK_EQ(info.name, String(prop[SNAME("name")]));
		CHECK_EQ(info.class_name, StringName(prop[SNAME("class_name")]));
		CHECK_EQ(info.type, (Variant::Type)(int)prop[SNAME("type")]);
		CHECK_EQ(info.hint, (PropertyHint)(int)prop[SNAME("hint")]);
		CHECK_EQ(info.hint_string, String(prop[SNAME("name")]));
		CHECK_EQ(info.usage, (PropertyUsageFlags)(int)prop[SNAME("usage")]);
	}

	SUBCASE("Duplication") {
		Variant var = prop;
		CHECK_EQ(var.get_type(), Variant::ARRAY);
		Variant var_dup = prop.duplicate();
		CHECK_EQ(var_dup.get_type(), Variant::ARRAY);
	}

	SUBCASE("Type Validation") {
		CHECK(prop.is_same_typed(prop));
		CHECK(prop.is_same_typed((Variant)prop));
		Variant var = prop;
		Struct<PropertyInfoLayout> prop2 = var;
		CHECK_EQ(prop2, var);

		CHECK_THROWS(prop.set_named(SNAME("name"), 4));
		CHECK_NOTHROW(prop.set_named(SNAME("name"), "Node")); // TODO: not sure if these tests are working correctly
	}

	SUBCASE("Setget Named") {
		Variant variant_prop = prop;
		bool valid = false;
		Variant changed = SNAME("Changed");
		variant_prop.set_named(SNAME("name"), SNAME("Changed"), valid);
		CHECK_EQ(valid, true);
		Variant val = variant_prop.get_named(SNAME("name"), valid);
		CHECK_EQ(valid, true);
		CHECK_EQ((StringName)val, SNAME("Changed"));

		val = variant_prop.get_named(SNAME("oops"), valid);
		CHECK_EQ(valid, false);
		CHECK_EQ(val, Variant());

		variant_prop.set_named(SNAME("oops"), SNAME("oh no"), valid);
		CHECK_EQ(valid, false);
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

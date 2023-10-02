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

TEST_CASE("[Struct] Validation") {
	struct NamedInt {
		StringName name;
		int value;
	};
	STRUCT_LAYOUT(NamedIntLayout, "NamedInt",
			STRUCT_MEMBER("name", Variant::STRING_NAME),
			STRUCT_MEMBER("value", Variant::INT));

	Struct<NamedIntLayout> named_int;
	named_int["name"] = "Godot";
	named_int["value"] = 4;
	CHECK_EQ(((Variant)named_int).stringify(), "[name: \"Godot\", value: 4]");

	SUBCASE("Self Equal") {
		CHECK(named_int.is_same_typed(named_int));
		Variant variant_named_int = named_int;
		CHECK(named_int.is_same_typed(variant_named_int));
		Struct<NamedIntLayout> same_named_int = variant_named_int;
		CHECK_EQ(named_int, same_named_int);
	}

	SUBCASE("Assignment") {
		ERR_PRINT_OFF;
		named_int.set_named("name", 4);
		CHECK_EQ(named_int["name"], "Godot");

		named_int.set_named("value", "Godot");
		CHECK_EQ((int)named_int["value"], 4);
		ERR_PRINT_ON;

		Struct<NamedIntLayout> a_match = named_int;
		CHECK_EQ(named_int, a_match);
		Array not_a_match;

		ERR_PRINT_OFF;
		named_int = not_a_match;
		CHECK_EQ(named_int, a_match);

		not_a_match.resize(2);
		named_int = not_a_match;
		CHECK_EQ(named_int, a_match);

		not_a_match[0] = 4;
		not_a_match[1] = "Godot";
		named_int = not_a_match;
		CHECK_EQ(named_int, a_match);

		not_a_match[0] = "Godooot";
		not_a_match[1] = 5;
		named_int = not_a_match;
		CHECK_EQ(named_int, a_match);
		ERR_PRINT_ON;

		named_int.assign(not_a_match);
		CHECK_EQ(named_int["name"], "Godooot");
		CHECK_EQ((int)named_int["value"], 5);
	}
}

TEST_CASE("[Struct] Nesting") {
	struct BasicStruct {
		int int_val;
		float float_val;
	};
	struct BasicStructLookalike {
		int int_val;
		float float_val;
	};
	struct NestedStruct {
		Node *node;
		BasicStruct value;
	};
	STRUCT_LAYOUT(BasicStructLayout, "BasicStruct",
			STRUCT_MEMBER("int_val", Variant::INT),
			STRUCT_MEMBER("float_val", Variant::FLOAT));
	STRUCT_LAYOUT(BasicStructLookalikeLayout, "BasicStructLookalike",
			STRUCT_MEMBER("int_val", Variant::INT),
			STRUCT_MEMBER("float_val", Variant::FLOAT));
	STRUCT_LAYOUT(NestedStructLayout, "NestedStruct",
			STRUCT_CLASS_MEMBER("node", "Node"),
			STRUCT_STRUCT_MEMBER("value", BasicStructLayout));

	SUBCASE("Assignment") {
		Struct<BasicStructLayout> basic_struct;
		basic_struct["int_val"] = 1;
		basic_struct["float_val"] = 3.14;

		Struct<BasicStructLookalikeLayout> basic_struct_lookalike;
		basic_struct_lookalike["int_val"] = 2;
		basic_struct_lookalike["float_val"] = 2.7;

		Struct<NestedStructLayout> nested_struct;
		Node *node = memnew(Node);
		nested_struct.set_named("node", node);
		nested_struct.set_named("value", basic_struct);

		CHECK_EQ(nested_struct["node"], Variant(node));
		CHECK_EQ(nested_struct["value"], basic_struct);
		CHECK_EQ(((Struct<BasicStructLayout>)nested_struct["value"])["int_val"], basic_struct["int_val"]);
		CHECK_EQ(((Struct<BasicStructLayout>)nested_struct["value"])["float_val"], basic_struct["float_val"]);

		ERR_PRINT_OFF;
		nested_struct.set_named("value", basic_struct_lookalike);
		CHECK_EQ(nested_struct["value"], basic_struct);
		ERR_PRINT_ON;
	}
}

TEST_CASE("[Struct] ClassDB") {
	StructInfo *struct_info = ::ClassDB::get_struct_info(SNAME("Object"), SNAME("PropertyInfo"));
	REQUIRE(struct_info);
	CHECK_EQ(struct_info->count, 6);
	CHECK_EQ(struct_info->name, "PropertyInfo");
	CHECK_EQ(struct_info->names[3], "hint");
	CHECK_EQ(struct_info->types[3], Variant::INT);
	CHECK_EQ(struct_info->class_names[3], "");
	CHECK_EQ(struct_info->default_values[3], Variant());
}

} // namespace TestStruct

#endif // TEST_STRUCT_H

/**************************************************************************/
/*  test_string_name.cpp                                                   */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_string_name)

#include "core/string/string_name.h"

namespace TestStringName {

TEST_CASE("[StringName] Default construction") {
	StringName name;
	CHECK(name.is_empty());
	CHECK(!name);
}

TEST_CASE("[StringName] Construction from string") {
	StringName name_char = "abc";
	StringName name_string = String("abc");

	CHECK(!name_char.is_empty());
	CHECK(!name_string.is_empty());
	CHECK(name_char == name_string);
}

TEST_CASE("[StringName] Interning") {
	// StringNames with equal contents share the same interned buffer.
	StringName name_a = "abc";
	StringName name_b = "abc";
	StringName name_c = String("abc");

	CHECK(name_a == name_b);
	CHECK(name_a == name_c);
	CHECK(name_a.data_unique_pointer() == name_b.data_unique_pointer());
	CHECK(name_a.data_unique_pointer() == name_c.data_unique_pointer());

	StringName name_d = "def";
	CHECK(name_a != name_d);
	CHECK(name_a.data_unique_pointer() != name_d.data_unique_pointer());
}

TEST_CASE("[StringName] Equality and inequality") {
	StringName name = "abc";

	CHECK(name == "abc");
	CHECK("abc" == name);
	CHECK(name == String("abc"));
	CHECK(String("abc") == name);

	CHECK(name != "def");
	CHECK("def" != name);
	CHECK(name != String("def"));
	CHECK(String("def") != name);

	CHECK(name == StringName("abc"));
	CHECK(name != StringName("def"));
}

TEST_CASE("[StringName] Hash") {
	StringName name_a = "abc";
	StringName name_b = "abc";

	CHECK(name_a.hash() == name_b.hash());

	// Empty StringNames share the empty hash.
	CHECK(StringName().hash() == StringName::EMPTY.hash());
}

TEST_CASE("[StringName] Length and indexing") {
	StringName name = "abc";

	CHECK(name.length() == 3);
	CHECK(name[0] == U'a');
	CHECK(name[1] == U'b');
	CHECK(name[2] == U'c');
}

TEST_CASE("[StringName] String conversion") {
	StringName name = "abc";

	CHECK(name.string() == "abc");
	CHECK(String(name) == "abc");
}

TEST_CASE("[StringName] is_node_unique_name") {
	CHECK(StringName("%Node").is_node_unique_name());
	CHECK(!StringName("Node").is_node_unique_name());
	CHECK(!StringName().is_node_unique_name());
}

TEST_CASE("[StringName] Copy and assignment") {
	StringName name = "abc";
	StringName copy(name);
	CHECK(copy == name);

	StringName assigned = "def";
	assigned = name;
	CHECK(assigned == name);
	CHECK(assigned == "abc");

	// Modifying a copy must not affect the original.
	StringName other = name;
	other = "xyz";
	CHECK(other == "xyz");
	CHECK(name == "abc");
}

TEST_CASE("[StringName] AlphCompare") {
	StringName name_a = "a";
	StringName name_b = "b";

	// AlphCompare::compare() returns true when the left operand sorts before the right.
	CHECK(StringName::AlphCompare::compare(name_a, name_b));
	CHECK(!StringName::AlphCompare::compare(name_b, name_a));
	CHECK(!StringName::AlphCompare::compare(name_a, name_a));
}

} // namespace TestStringName

/**************************************************************************/
/*  test_string_name.cpp                                                  */
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
#include "core/string/ustring.h"
#include "core/variant/variant.h"

namespace TestStringName {

TEST_CASE("[StringName] Variant indexed get") {
	Variant s = StringName("abcd");
	bool valid = false;
	bool oob = true;

	String r = s.get_indexed(1, valid, oob);

	CHECK(valid);
	CHECK_FALSE(oob);
	CHECK_EQ(r, String("b"));
}

TEST_CASE("[StringName] Variant validated indexed get") {
	Variant s = StringName("abcd");

	Variant::ValidatedIndexedGetter getter = Variant::get_member_validated_indexed_getter(Variant::STRING_NAME);

	Variant r;
	bool oob = true;
	getter(&s, 1, &r, &oob);

	CHECK_FALSE(oob);
	CHECK_EQ(r, String("b"));
}

TEST_CASE("[StringName] Variant ptr indexed get") {
	StringName s("abcd");

	Variant::PTRIndexedGetter getter = Variant::get_member_ptr_indexed_getter(Variant::STRING_NAME);

	String r;
	getter(&s, 1, &r);

	CHECK_EQ(r, String("b"));
}

TEST_CASE("[StringName] Variant indexed set") {
	Variant s = StringName("abcd");
	bool valid = false;
	bool oob = false;

	s.set_indexed(1, String("z"), valid, oob);

	CHECK_FALSE(valid);
	CHECK(oob);
	CHECK_EQ(StringName(s), StringName("abcd"));
}

TEST_CASE("[StringName] Variant validated indexed set") {
	Variant s = StringName("abcd");

	Variant::ValidatedIndexedSetter setter = Variant::get_member_validated_indexed_setter(Variant::STRING_NAME);

	Variant v = String("z");
	bool oob = false;
	setter(&s, 1, &v, &oob);

	CHECK(oob);
	CHECK_EQ(StringName(s), StringName("abcd"));
}

TEST_CASE("[StringName] Variant ptr indexed set") {
	StringName s("abcd");

	Variant::PTRIndexedSetter setter = Variant::get_member_ptr_indexed_setter(Variant::STRING_NAME);

	String v("z");
	setter(&s, 1, &v);

	CHECK_EQ(StringName(s), StringName("abcd"));
}

} // namespace TestStringName

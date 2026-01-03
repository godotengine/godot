/**************************************************************************/
/*  test_property_utils.h                                                 */
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

#include "scene/property_utils.h"

#include "tests/test_macros.h"

namespace TestPropertyUtils {

TEST_CASE("[PropertyUtils] Empty Array Hint") {
	String hint_string;
	Variant::Type subtype;
	PropertyHint subtype_hint;
	String subtype_hint_string;
	PropertyUtils::parse_array_hint_string(hint_string, subtype, subtype_hint, &subtype_hint_string);

	CHECK(subtype == Variant::NIL);
	CHECK(subtype_hint == PROPERTY_HINT_NONE);
	CHECK(subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Type Only Array Hint") {
	String hint_string = vformat("%d:", Variant::INT);
	Variant::Type subtype;
	PropertyHint subtype_hint;
	String subtype_hint_string;
	PropertyUtils::parse_array_hint_string(hint_string, subtype, subtype_hint, &subtype_hint_string);

	CHECK(subtype == Variant::INT);
	CHECK(subtype_hint == PROPERTY_HINT_NONE);
	CHECK(subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Type String Array Hint") {
	String hint_string = "int";
	Variant::Type subtype;
	PropertyHint subtype_hint;
	String subtype_hint_string;
	PropertyUtils::parse_array_hint_string(hint_string, subtype, subtype_hint, &subtype_hint_string);

	CHECK(subtype == Variant::INT);
	CHECK(subtype_hint == PROPERTY_HINT_NONE);
	CHECK(subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Object Type Array Hint") {
	String hint_string = "Texture";
	Variant::Type subtype;
	PropertyHint subtype_hint;
	String subtype_hint_string;
	PropertyUtils::parse_array_hint_string(hint_string, subtype, subtype_hint, &subtype_hint_string);

	CHECK(subtype == Variant::OBJECT);
	CHECK(subtype_hint == PROPERTY_HINT_RESOURCE_TYPE);
	CHECK(subtype_hint_string == "Texture");
}

TEST_CASE("[PropertyUtils] Type And Hint Array Hint") {
	String hint_string = vformat("%d/%d:", Variant::INT, PROPERTY_HINT_ENUM);
	Variant::Type subtype;
	PropertyHint subtype_hint;
	String subtype_hint_string;
	PropertyUtils::parse_array_hint_string(hint_string, subtype, subtype_hint, &subtype_hint_string);

	CHECK(subtype == Variant::INT);
	CHECK(subtype_hint == PROPERTY_HINT_ENUM);
	CHECK(subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Complete Array Hint") {
	String hint_string = vformat("%d/%d:%s", Variant::INT, PROPERTY_HINT_ENUM, "One,Two,Three");
	Variant::Type subtype;
	PropertyHint subtype_hint;
	String subtype_hint_string;
	PropertyUtils::parse_array_hint_string(hint_string, subtype, subtype_hint, &subtype_hint_string);

	CHECK(subtype == Variant::INT);
	CHECK(subtype_hint == PROPERTY_HINT_ENUM);
	CHECK(subtype_hint_string == "One,Two,Three");
}

TEST_CASE("[PropertyUtils] Empty Dictionary Hint") {
	String hint_string;
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::NIL);
	CHECK(key_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(key_subtype_hint_string == String());
	CHECK(value_subtype == Variant::NIL);
	CHECK(value_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(value_subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Key Type Only Dictionary Hint") {
	String hint_string = vformat("%d:;", Variant::INT);
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::INT);
	CHECK(key_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(key_subtype_hint_string == String());
	CHECK(value_subtype == Variant::NIL);
	CHECK(value_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(value_subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Key Type String Dictionary Hint") {
	String hint_string = "int";
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::INT);
	CHECK(key_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(key_subtype_hint_string == String());
	CHECK(value_subtype == Variant::NIL);
	CHECK(value_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(value_subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Key Object Type Dictionary Hint") {
	String hint_string = "Texture";
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::OBJECT);
	CHECK(key_subtype_hint == PROPERTY_HINT_RESOURCE_TYPE);
	CHECK(key_subtype_hint_string == "Texture");
	CHECK(value_subtype == Variant::NIL);
	CHECK(value_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(value_subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Key Type And Hint Dictionary Hint") {
	String hint_string = vformat("%d/%d:;", Variant::INT, PROPERTY_HINT_ENUM);
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::INT);
	CHECK(key_subtype_hint == PROPERTY_HINT_ENUM);
	CHECK(key_subtype_hint_string == String());
	CHECK(value_subtype == Variant::NIL);
	CHECK(value_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(value_subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Complete Key Dictionary Hint") {
	String hint_string = vformat("%d/%d:%s;", Variant::INT, PROPERTY_HINT_ENUM, "One,Two,Three");
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::INT);
	CHECK(key_subtype_hint == PROPERTY_HINT_ENUM);
	CHECK(key_subtype_hint_string == "One,Two,Three");
	CHECK(value_subtype == Variant::NIL);
	CHECK(value_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(value_subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Value Type Only Dictionary Hint") {
	String hint_string = vformat(";%d:", Variant::INT);
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::NIL);
	CHECK(key_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(key_subtype_hint_string == String());
	CHECK(value_subtype == Variant::INT);
	CHECK(value_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(value_subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Value Type String Dictionary Hint") {
	String hint_string = ";int";
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::NIL);
	CHECK(key_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(key_subtype_hint_string == String());
	CHECK(value_subtype == Variant::INT);
	CHECK(value_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(value_subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Value Object Type Dictionary Hint") {
	String hint_string = ";Texture";
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::NIL);
	CHECK(key_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(key_subtype_hint_string == String());
	CHECK(value_subtype == Variant::OBJECT);
	CHECK(value_subtype_hint == PROPERTY_HINT_RESOURCE_TYPE);
	CHECK(value_subtype_hint_string == "Texture");
}

TEST_CASE("[PropertyUtils] Value Type And Hint Dictionary Hint") {
	String hint_string = vformat(";%d/%d:", Variant::INT, PROPERTY_HINT_ENUM);
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::NIL);
	CHECK(key_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(key_subtype_hint_string == String());
	CHECK(value_subtype == Variant::INT);
	CHECK(value_subtype_hint == PROPERTY_HINT_ENUM);
	CHECK(value_subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Complete Value Dictionary Hint") {
	String hint_string = vformat(";%d/%d:%s", Variant::INT, PROPERTY_HINT_ENUM, "One,Two,Three");
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::NIL);
	CHECK(key_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(key_subtype_hint_string == String());
	CHECK(value_subtype == Variant::INT);
	CHECK(value_subtype_hint == PROPERTY_HINT_ENUM);
	CHECK(value_subtype_hint_string == "One,Two,Three");
}

TEST_CASE("[PropertyUtils] Key And Value Type Only Dictionary Hint") {
	String hint_string = vformat("%d:;%d:", Variant::INT, Variant::STRING);
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::INT);
	CHECK(key_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(key_subtype_hint_string == String());
	CHECK(value_subtype == Variant::STRING);
	CHECK(value_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(value_subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Key And Value Type Strings Dictionary Hint") {
	String hint_string = "int;float";
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::INT);
	CHECK(key_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(key_subtype_hint_string == String());
	CHECK(value_subtype == Variant::FLOAT);
	CHECK(value_subtype_hint == PROPERTY_HINT_NONE);
	CHECK(value_subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Key And Value Object Type Dictionary Hint") {
	String hint_string = "Texture;Texture2D";
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::OBJECT);
	CHECK(key_subtype_hint == PROPERTY_HINT_RESOURCE_TYPE);
	CHECK(key_subtype_hint_string == "Texture");
	CHECK(value_subtype == Variant::OBJECT);
	CHECK(value_subtype_hint == PROPERTY_HINT_RESOURCE_TYPE);
	CHECK(value_subtype_hint_string == "Texture2D");
}

TEST_CASE("[PropertyUtils] Key And Value Type And Hint Dictionary Hint") {
	String hint_string = vformat("%d/%d:;%d/%d:", Variant::INT, PROPERTY_HINT_ENUM, Variant::STRING, PROPERTY_HINT_FILE);
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::INT);
	CHECK(key_subtype_hint == PROPERTY_HINT_ENUM);
	CHECK(key_subtype_hint_string == String());
	CHECK(value_subtype == Variant::STRING);
	CHECK(value_subtype_hint == PROPERTY_HINT_FILE);
	CHECK(value_subtype_hint_string == String());
}

TEST_CASE("[PropertyUtils] Complete Key And Value Dictionary Hint") {
	String hint_string = vformat("%d/%d:%s;%d/%d:%s", Variant::INT, PROPERTY_HINT_ENUM, "One,Two,Three", Variant::STRING, PROPERTY_HINT_FILE, "*.png");
	Variant::Type key_subtype;
	PropertyHint key_subtype_hint;
	String key_subtype_hint_string;
	Variant::Type value_subtype;
	PropertyHint value_subtype_hint;
	String value_subtype_hint_string;
	PropertyUtils::parse_dictionary_hint_string(hint_string, key_subtype, key_subtype_hint, value_subtype, value_subtype_hint, &key_subtype_hint_string, &value_subtype_hint_string);

	CHECK(key_subtype == Variant::INT);
	CHECK(key_subtype_hint == PROPERTY_HINT_ENUM);
	CHECK(key_subtype_hint_string == "One,Two,Three");
	CHECK(value_subtype == Variant::STRING);
	CHECK(value_subtype_hint == PROPERTY_HINT_FILE);
	CHECK(value_subtype_hint_string == "*.png");
}

} //namespace TestPropertyUtils

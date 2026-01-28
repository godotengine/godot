/**************************************************************************/
/*  test_json.h                                                           */
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

#include "core/io/json.h"

#include "thirdparty/doctest/doctest.h"

namespace TestJSON {

TEST_CASE("[JSON] Stringify single data types") {
	CHECK(JSON::stringify(Variant()) == "null");
	CHECK(JSON::stringify(false) == "false");
	CHECK(JSON::stringify(true) == "true");
	CHECK(JSON::stringify(0) == "0");
	CHECK(JSON::stringify(12345) == "12345");
	CHECK(JSON::stringify(0.75) == "0.75");
	CHECK(JSON::stringify("test") == "\"test\"");
	CHECK(JSON::stringify("\\\b\f\n\r\t\v\"") == "\"\\\\\\b\\f\\n\\r\\t\\v\\\"\"");
}

TEST_CASE("[JSON] Stringify arrays") {
	CHECK(JSON::stringify(Array()) == "[]");

	Array int_array;
	for (int i = 0; i < 10; i++) {
		int_array.push_back(i);
	}
	CHECK(JSON::stringify(int_array) == "[0,1,2,3,4,5,6,7,8,9]");

	Array str_array;
	str_array.push_back("Hello");
	str_array.push_back("World");
	str_array.push_back("!");
	CHECK(JSON::stringify(str_array) == "[\"Hello\",\"World\",\"!\"]");

	Array indented_array;
	Array nested_array;
	for (int i = 0; i < 5; i++) {
		indented_array.push_back(i);
		nested_array.push_back(i);
	}
	indented_array.push_back(nested_array);
	CHECK(JSON::stringify(indented_array, "\t") == "[\n\t0,\n\t1,\n\t2,\n\t3,\n\t4,\n\t[\n\t\t0,\n\t\t1,\n\t\t2,\n\t\t3,\n\t\t4\n\t]\n]");

	Array full_precision_array;
	full_precision_array.push_back(0.12345678901234568);
	CHECK(JSON::stringify(full_precision_array, "", true, true) == "[0.12345678901234568]");

	Array non_finite_array;
	non_finite_array.push_back(Math::INF);
	non_finite_array.push_back(-Math::INF);
	non_finite_array.push_back(Math::NaN);
	ERR_PRINT_OFF
	CHECK(JSON::stringify(non_finite_array) == "[1e99999,-1e99999,null]");

	Array non_finite_round_trip = JSON::parse_string(JSON::stringify(non_finite_array));
	CHECK(non_finite_round_trip[0] == Variant(Math::INF));
	CHECK(non_finite_round_trip[1] == Variant(-Math::INF));
	CHECK(non_finite_round_trip[2].get_type() == Variant::NIL);

	Array self_array;
	self_array.push_back(self_array);
	CHECK(JSON::stringify(self_array) == "[\"[...]\"]");
	self_array.clear();

	Array max_recursion_array;
	for (int i = 0; i < Variant::MAX_RECURSION_DEPTH + 1; i++) {
		Array next;
		next.push_back(max_recursion_array);
		max_recursion_array = next;
	}
	CHECK(JSON::stringify(max_recursion_array).contains("[...]"));
	ERR_PRINT_ON
}

TEST_CASE("[JSON] Stringify dictionaries") {
	CHECK(JSON::stringify(Dictionary()) == "{}");

	Dictionary single_entry;
	single_entry["key"] = "value";
	CHECK(JSON::stringify(single_entry) == "{\"key\":\"value\"}");

	Dictionary indented;
	indented["key1"] = "value1";
	indented["key2"] = 2;
	CHECK(JSON::stringify(indented, "\t") == "{\n\t\"key1\": \"value1\",\n\t\"key2\": 2\n}");

	Dictionary outer;
	Dictionary inner;
	inner["key"] = "value";
	outer["inner"] = inner;
	CHECK(JSON::stringify(outer) == "{\"inner\":{\"key\":\"value\"}}");

	Dictionary full_precision_dictionary;
	full_precision_dictionary["key"] = 0.12345678901234568;
	CHECK(JSON::stringify(full_precision_dictionary, "", true, true) == "{\"key\":0.12345678901234568}");

	Dictionary non_finite_dictionary;
	non_finite_dictionary["-inf"] = -Math::INF;
	non_finite_dictionary["inf"] = Math::INF;
	non_finite_dictionary["nan"] = Math::NaN;
	ERR_PRINT_OFF
	CHECK(JSON::stringify(non_finite_dictionary) == "{\"-inf\":-1e99999,\"inf\":1e99999,\"nan\":null}");

	Dictionary non_finite_round_trip = JSON::parse_string(JSON::stringify(non_finite_dictionary));
	CHECK(non_finite_round_trip["-inf"] == Variant(-Math::INF));
	CHECK(non_finite_round_trip["inf"] == Variant(Math::INF));
	CHECK(non_finite_round_trip["nan"].get_type() == Variant::NIL);

	Dictionary self_dictionary;
	self_dictionary["key"] = self_dictionary;
	CHECK(JSON::stringify(self_dictionary) == "{\"key\":\"{...}\"}");
	self_dictionary.clear();

	Dictionary max_recursion_dictionary;
	for (int i = 0; i < Variant::MAX_RECURSION_DEPTH + 1; i++) {
		Dictionary next;
		next["key"] = max_recursion_dictionary;
		max_recursion_dictionary = next;
	}
	CHECK(JSON::stringify(max_recursion_dictionary).contains("{...:...}"));
	ERR_PRINT_ON
}

// NOTE: The current JSON parser accepts many non-conformant strings such as
// single-quoted strings, duplicate commas and trailing commas.
// This is intentionally not tested as users shouldn't rely on this behavior.

TEST_CASE("[JSON] Parsing single data types") {
	// Parsing a single data type as JSON is valid per the JSON specification.

	JSON json;

	json.parse("null");
	CHECK_MESSAGE(
			json.get_error_line() == 0,
			"Parsing `null` as JSON should parse successfully.");
	CHECK_MESSAGE(
			json.get_data() == Variant(),
			"Parsing a double quoted string as JSON should return the expected value.");

	json.parse("true");
	CHECK_MESSAGE(
			json.get_error_line() == 0,
			"Parsing boolean `true` as JSON should parse successfully.");
	CHECK_MESSAGE(
			json.get_data(),
			"Parsing boolean `true` as JSON should return the expected value.");

	json.parse("false");
	CHECK_MESSAGE(
			json.get_error_line() == 0,
			"Parsing boolean `false` as JSON should parse successfully.");
	CHECK_MESSAGE(
			!json.get_data(),
			"Parsing boolean `false` as JSON should return the expected value.");

	json.parse("123456");
	CHECK_MESSAGE(
			json.get_error_line() == 0,
			"Parsing an integer number as JSON should parse successfully.");
	CHECK_MESSAGE(
			(int)(json.get_data()) == 123456,
			"Parsing an integer number as JSON should return the expected value.");

	json.parse("0.123456");
	CHECK_MESSAGE(
			json.get_error_line() == 0,
			"Parsing a floating-point number as JSON should parse successfully.");
	CHECK_MESSAGE(
			double(json.get_data()) == doctest::Approx(0.123456),
			"Parsing a floating-point number as JSON should return the expected value.");

	json.parse("\"hello\"");
	CHECK_MESSAGE(
			json.get_error_line() == 0,
			"Parsing a double quoted string as JSON should parse successfully.");
	CHECK_MESSAGE(
			json.get_data() == "hello",
			"Parsing a double quoted string as JSON should return the expected value.");
}

TEST_CASE("[JSON] Parsing arrays") {
	JSON json;

	// JSON parsing fails if it's split over several lines (even if leading indentation is removed).
	json.parse(R"(["Hello", "world.", "This is",["a","json","array.",[]], "Empty arrays ahoy:", [[["Gotcha!"]]]])");

	const Array array = json.get_data();
	CHECK_MESSAGE(
			json.get_error_line() == 0,
			"Parsing a JSON array should parse successfully.");
	CHECK_MESSAGE(
			array[0] == "Hello",
			"The parsed JSON should contain the expected values.");
	const Array sub_array = array[3];
	CHECK_MESSAGE(
			sub_array.size() == 4,
			"The parsed JSON should contain the expected values.");
	CHECK_MESSAGE(
			sub_array[1] == "json",
			"The parsed JSON should contain the expected values.");
	CHECK_MESSAGE(
			sub_array[3].hash() == Array().hash(),
			"The parsed JSON should contain the expected values.");
	const Array deep_array = Array(Array(array[5])[0])[0];
	CHECK_MESSAGE(
			deep_array[0] == "Gotcha!",
			"The parsed JSON should contain the expected values.");
}

TEST_CASE("[JSON] Parsing objects (dictionaries)") {
	JSON json;

	json.parse(R"({"name": "Godot Engine", "is_free": true, "bugs": null, "apples": {"red": 500, "green": 0, "blue": -20}, "empty_object": {}})");

	const Dictionary dictionary = json.get_data();
	CHECK_MESSAGE(
			dictionary["name"] == "Godot Engine",
			"The parsed JSON should contain the expected values.");
	CHECK_MESSAGE(
			dictionary["is_free"],
			"The parsed JSON should contain the expected values.");
	CHECK_MESSAGE(
			dictionary["bugs"] == Variant(),
			"The parsed JSON should contain the expected values.");
	CHECK_MESSAGE(
			(int)Dictionary(dictionary["apples"])["blue"] == -20,
			"The parsed JSON should contain the expected values.");
	CHECK_MESSAGE(
			dictionary["empty_object"].hash() == Dictionary().hash(),
			"The parsed JSON should contain the expected values.");
}

TEST_CASE("[JSON] Parsing escape sequences") {
	// Only certain escape sequences are valid according to the JSON specification.
	// Others must result in a parsing error instead.

	JSON json;

	TypedArray<String> valid_escapes = { "\";\"", "\\;\\", "/;/", "b;\b", "f;\f", "n;\n", "r;\r", "t;\t" };

	SUBCASE("Basic valid escape sequences") {
		for (int i = 0; i < valid_escapes.size(); i++) {
			String valid_escape = valid_escapes[i];
			String valid_escape_string = valid_escape.get_slicec(';', 0);
			String valid_escape_value = valid_escape.get_slicec(';', 1);

			String json_string = "\"\\";
			json_string += valid_escape_string;
			json_string += "\"";
			json.parse(json_string);

			CHECK_MESSAGE(
					json.get_error_line() == 0,
					vformat("Parsing valid escape sequence `%s` as JSON should parse successfully.", valid_escape_string));

			String json_value = json.get_data();
			CHECK_MESSAGE(
					json_value == valid_escape_value,
					vformat("Parsing valid escape sequence `%s` as JSON should return the expected value.", valid_escape_string));
		}
	}

	SUBCASE("Valid unicode escape sequences") {
		String json_string = "\"\\u0020\"";
		json.parse(json_string);

		CHECK_MESSAGE(
				json.get_error_line() == 0,
				vformat("Parsing valid unicode escape sequence with value `0020` as JSON should parse successfully."));

		String json_value = json.get_data();
		CHECK_MESSAGE(
				json_value == " ",
				vformat("Parsing valid unicode escape sequence with value `0020` as JSON should return the expected value."));
	}

	SUBCASE("Invalid escape sequences") {
		ERR_PRINT_OFF
		for (char32_t i = 0; i < 128; i++) {
			bool skip = false;
			for (int j = 0; j < valid_escapes.size(); j++) {
				String valid_escape = valid_escapes[j];
				String valid_escape_string = valid_escape.get_slicec(';', 0);
				if (valid_escape_string[0] == i) {
					skip = true;
					break;
				}
			}

			if (skip) {
				continue;
			}

			String json_string = "\"\\";
			json_string += i;
			json_string += "\"";
			Error err = json.parse(json_string);

			// TODO: Line number is currently kept on 0, despite an error occurring. This should be fixed in the JSON parser.
			// CHECK_MESSAGE(
			// 		json.get_error_line() != 0,
			// 		vformat("Parsing invalid escape sequence with ASCII value `%d` as JSON should fail to parse.", i));
			CHECK_MESSAGE(
					err == ERR_PARSE_ERROR,
					vformat("Parsing invalid escape sequence with ASCII value `%d` as JSON should fail to parse with ERR_PARSE_ERROR.", i));
		}
		ERR_PRINT_ON
	}
}

TEST_CASE("[JSON] Serialization") {
	JSON json;

	struct FpTestCase {
		double number;
		String json;
	};

	struct IntTestCase {
		int64_t number;
		String json;
	};

	struct UIntTestCase {
		uint64_t number;
		String json;
	};

	static FpTestCase fp_tests_default_precision[] = {
		{ 0.0, "0.0" },
		{ 1000.1234567890123456789, "1000.12345678901" },
		{ -1000.1234567890123456789, "-1000.12345678901" },
		{ DBL_MAX, "179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0" },
		{ DBL_MAX - 1, "179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.0" },
		{ std::pow(2, 53), "9007199254740992.0" },
		{ -std::pow(2, 53), "-9007199254740992.0" },
		{ 0.00000000000000011, "0.00000000000000011" },
		{ -0.00000000000000011, "-0.00000000000000011" },
		{ 1.0 / 3.0, "0.333333333333333" },
		{ 0.9999999999999999, "1.0" },
		{ 1.0000000000000001, "1.0" },
	};

	static FpTestCase fp_tests_full_precision[] = {
		{ 0.0, "0.0" },
		{ 1000.1234567890123456789, "1000.1234567890124" },
		{ -1000.1234567890123456789, "-1000.1234567890124" },
		{ DBL_MAX, "1.7976931348623157e+308" },
		{ DBL_MAX - 1, "1.7976931348623157e+308" },
		{ std::pow(2, 53), "9.007199254740992e+15" },
		{ -std::pow(2, 53), "-9.007199254740992e+15" },
		{ 0.00000000000000011, "1.1e-16" },
		{ -0.00000000000000011, "-1.1e-16" },
		{ 1.0 / 3.0, "0.3333333333333333" },
		{ 0.9999999999999999, "0.9999999999999999" },
		{ 1.0000000000000001, "1.0" },
	};

	static IntTestCase int_tests[] = {
		{ 0, "0" },
		{ INT64_MAX, "9223372036854775807" },
		{ INT64_MIN, "-9223372036854775808" },
	};

	SUBCASE("Floating point default precision") {
		for (FpTestCase &test : fp_tests_default_precision) {
			String json_value = json.stringify(test.number, "", true, false);

			CHECK_MESSAGE(
					json_value == test.json,
					vformat("Serializing `%.20d` to JSON should return the expected value.", test.number));
		}
	}

	SUBCASE("Floating point full precision") {
		for (FpTestCase &test : fp_tests_full_precision) {
			String json_value = json.stringify(test.number, "", true, true);

			CHECK_MESSAGE(
					json_value == test.json,
					vformat("Serializing `%20f` to JSON should return the expected value.", test.number));
		}
	}

	SUBCASE("Signed integer") {
		for (IntTestCase &test : int_tests) {
			String json_value = json.stringify(test.number, "", true, true);

			CHECK_MESSAGE(
					json_value == test.json,
					vformat("Serializing `%d` to JSON should return the expected value.", test.number));
		}
	}
}
} // namespace TestJSON

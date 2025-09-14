/**************************************************************************/
/*  test_jsonrpc.h                                                        */
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

#include "tests/test_macros.h"
#include "tests/test_utils.h"

#include "../jsonrpc.h"

namespace TestJSONRPC {

void check_invalid(const Dictionary &p_dict);

TEST_CASE("[JSONRPC] process_action invalid") {
	JSONRPC json_rpc = JSONRPC();

	check_invalid(json_rpc.process_action("String is invalid"));
	check_invalid(json_rpc.process_action(1234));
	check_invalid(json_rpc.process_action(false));
	check_invalid(json_rpc.process_action(3.14159));
	check_invalid(json_rpc.process_action(Array()));
}

void check_invalid_string(const String &p_str);

TEST_CASE("[JSONRPC] process_string invalid") {
	JSONRPC json_rpc = JSONRPC();

	check_invalid_string(json_rpc.process_string("\"String is invalid\""));
	check_invalid_string(json_rpc.process_string("1234"));
	check_invalid_string(json_rpc.process_string("false"));
	check_invalid_string(json_rpc.process_string("3.14159"));
	check_invalid_string(json_rpc.process_string("[]"));
}

class TestClassJSONRPC : public JSONRPC {
public:
	TestClassJSONRPC() {
		set_method("something", callable_mp(this, &TestClassJSONRPC::something));
	}

	String something(const String &p_in);
};

void test_process_action(const Variant &p_in, const Variant &p_expected, bool p_process_array_elements = false);

TEST_CASE("[JSONRPC] process_action Dictionary") {
	Dictionary in_dict = Dictionary();
	in_dict["method"] = "something";
	in_dict["id"] = "ID";
	in_dict["params"] = "yes";

	Dictionary expected_dict = Dictionary();
	expected_dict["jsonrpc"] = "2.0";
	expected_dict["id"] = "ID";
	expected_dict["result"] = "yes, please";

	test_process_action(in_dict, expected_dict);
}

TEST_CASE("[JSONRPC] process_action Array") {
	Array in;
	Dictionary in_1;
	in_1["method"] = "something";
	in_1["id"] = 1;
	in_1["params"] = "more";
	in.push_back(in_1);
	Dictionary in_2;
	in_2["method"] = "something";
	in_2["id"] = 2;
	in_2["params"] = "yes";
	in.push_back(in_2);

	Array expected;
	Dictionary expected_1;
	expected_1["jsonrpc"] = "2.0";
	expected_1["id"] = 1;
	expected_1["result"] = "more, please";
	expected.push_back(expected_1);
	Dictionary expected_2;
	expected_2["jsonrpc"] = "2.0";
	expected_2["id"] = 2;
	expected_2["result"] = "yes, please";
	expected.push_back(expected_2);

	test_process_action(in, expected, true);
}

void test_process_string(const String &p_in, const String &p_expected);

TEST_CASE("[JSONRPC] process_string Dictionary") {
	const String in = "{\"method\":\"something\",\"id\":\"ID\",\"params\":\"yes\"}";
	const String expected = "{\"id\":\"ID\",\"jsonrpc\":\"2.0\",\"result\":\"yes, please\"}";

	test_process_string(in, expected);
}

void test_process_action_bad_method(const Dictionary &p_in);

TEST_CASE("[JSONRPC] process_action bad method") {
	Dictionary in_dict;
	in_dict["id"] = 1;
	in_dict["method"] = "nothing";

	test_process_action_bad_method(in_dict);
}

void test_no_response(const Variant &p_in);

TEST_CASE("[JSONRPC] process_action notification") {
	Dictionary in_dict = Dictionary();
	in_dict["method"] = "something";
	in_dict["params"] = "yes";

	test_no_response(in_dict);
}

TEST_CASE("[JSONRPC] process_action notification bad method") {
	Dictionary in_dict;
	in_dict["method"] = "nothing";

	test_no_response(in_dict);
}

TEST_CASE("[JSONRPC] process_action notification batch") {
	Array in;
	Dictionary in_1;
	in_1["method"] = "something";
	in_1["params"] = "more";
	in.push_back(in_1);
	Dictionary in_2;
	in_2["method"] = "something";
	in_2["params"] = "yes";
	in.push_back(in_2);

	test_no_response(in);
}

TEST_CASE("[JSONRPC] mixed batch") {
	Array in;
	Dictionary in_1;
	in_1["method"] = "something";
	in_1["id"] = 1;
	in_1["params"] = "more";
	in.push_back(in_1);
	Dictionary in_2;
	in_2["method"] = "something";
	in_2["id"] = 2;
	in_2["params"] = "yes";
	in.push_back(in_2);
	Dictionary in_3;
	in_3["method"] = "something";
	in_3["params"] = "yes";
	in.push_back(in_3);

	Array expected;
	Dictionary expected_1;
	expected_1["jsonrpc"] = "2.0";
	expected_1["id"] = 1;
	expected_1["result"] = "more, please";
	expected.push_back(expected_1);
	Dictionary expected_2;
	expected_2["jsonrpc"] = "2.0";
	expected_2["id"] = 2;
	expected_2["result"] = "yes, please";
	expected.push_back(expected_2);

	test_process_action(in, expected, true);
}

class TestHandlerJSONRPC : public Object {
public:
	bool called_1 = false;
	bool called_2 = false;

	void method1(const Variant &p_arg) {
		called_1 = true;
	}

	void method2(const Variant &p_arg) {
		called_2 = true;
	}
};

TEST_CASE("[JSONRPC] process response") {
	TestHandlerJSONRPC *handler = memnew(TestHandlerJSONRPC);
	JSONRPC jsonrpc = JSONRPC();

	jsonrpc.set_response_handler(1, callable_mp(handler, &TestHandlerJSONRPC::method1));
	jsonrpc.set_response_handler(2, callable_mp(handler, &TestHandlerJSONRPC::method2));

	jsonrpc.process_action(jsonrpc.make_response(Variant(), 1));

	CHECK(handler->called_1);
	CHECK_FALSE(handler->called_2);

	memdelete(handler);
}

} // namespace TestJSONRPC

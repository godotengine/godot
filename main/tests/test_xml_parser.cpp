/*************************************************************************/
/*  test_xml_parser.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "test_xml_parser.h"

#include "core/os/os.h"

namespace TestXMLParser {

#define CHECK(X)                                          \
	if (!(X)) {                                           \
		OS::get_singleton()->print("\tFAIL at %s\n", #X); \
		return false;                                     \
	} else {                                              \
		OS::get_singleton()->print("\tPASS\n");           \
	}
#define REQUIRE_EQ(X, Y)                                            \
	if ((X) != (Y)) {                                               \
		OS::get_singleton()->print("\tFAIL at %s != %s\n", #X, #Y); \
		return false;                                               \
	} else {                                                        \
		OS::get_singleton()->print("\tPASS\n");                     \
	}

Vector<uint8_t> _to_buffer(const String &p_text) {
	Vector<uint8_t> buff;
	for (int i = 0; i < p_text.length(); i++) {
		buff.push_back(p_text.get(i));
	}
	return buff;
}

bool test_1() {
	String source = "<?xml version = \"1.0\" encoding=\"UTF-8\" ?>\
<top attr=\"attr value\">\
  Text&lt;&#65;&#x42;&gt;\
</top>";
	Vector<uint8_t> buff = _to_buffer(source);
	XMLParser parser;
	parser.open_buffer(buff);

	// <?xml ...?> gets parsed as NODE_UNKNOWN
	CHECK(parser.read() == OK);
	CHECK(parser.get_node_type() == XMLParser::NodeType::NODE_UNKNOWN);

	CHECK(parser.read() == OK);
	CHECK(parser.get_node_type() == XMLParser::NodeType::NODE_ELEMENT);
	CHECK(parser.get_node_name() == "top");
	CHECK(parser.has_attribute("attr"));
	CHECK(parser.get_attribute_value("attr") == "attr value");

	CHECK(parser.read() == OK);
	CHECK(parser.get_node_type() == XMLParser::NodeType::NODE_TEXT);
	CHECK(parser.get_node_data().lstrip(" \t") == "Text<AB>");

	CHECK(parser.read() == OK);
	CHECK(parser.get_node_type() == XMLParser::NodeType::NODE_ELEMENT_END);
	CHECK(parser.get_node_name() == "top");

	parser.close();
	return true;
}

bool test_comments() {
	XMLParser parser;

	// Missing end of comment.
	{
		const String input = "<first></first><!-- foo";
		REQUIRE_EQ(parser.open_buffer(_to_buffer(input)), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_ELEMENT);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_ELEMENT_END);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_COMMENT);
		REQUIRE_EQ(parser.get_node_name(), " foo");
	}

	// Bad start of comment.
	{
		const String input = "<first></first><!-";
		REQUIRE_EQ(parser.open_buffer(_to_buffer(input)), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_ELEMENT);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_ELEMENT_END);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_COMMENT);
		REQUIRE_EQ(parser.get_node_name(), "-");
	}

	// Unblanced angle brackets in comment.
	{
		const String input = "<!-- example << --><next-tag></next-tag>";
		REQUIRE_EQ(parser.open_buffer(_to_buffer(input)), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_COMMENT);
		REQUIRE_EQ(parser.get_node_name(), " example << ");
	}

	// Doctype.
	{
		const String input = "<!DOCTYPE greeting [<!ELEMENT greeting (#PCDATA)>]>";
		REQUIRE_EQ(parser.open_buffer(_to_buffer(input)), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_COMMENT);
		REQUIRE_EQ(parser.get_node_name(), "DOCTYPE greeting [<!ELEMENT greeting (#PCDATA)>]");
	}

	return true;
}

bool test_premature_endings() {
	struct Case {
		String input;
		String expected_name;
		XMLParser::NodeType expected_type;
	} const cases[] = {
		// Incomplete unknown.
		{ "<first></first><?xml", "?xml", XMLParser::NodeType::NODE_UNKNOWN },
		// Incomplete CDStart.
		{ "<first></first><![CD", "", XMLParser::NodeType::NODE_CDATA },
		// Incomplete CData.
		{ "<first></first><![CDATA[example", "example", XMLParser::NodeType::NODE_CDATA },
		// Incomplete CDEnd.
		{ "<first></first><![CDATA[example]]", "example]]", XMLParser::NodeType::NODE_CDATA },
		// Incomplete start-tag name.
		{ "<first></first><second", "second", XMLParser::NodeType::NODE_ELEMENT },
	};

	XMLParser parser;

	for (unsigned long i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
		const Case &test_case = cases[i];

		REQUIRE_EQ(parser.open_buffer(_to_buffer(test_case.input)), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_ELEMENT);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_ELEMENT_END);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), test_case.expected_type);
		REQUIRE_EQ(parser.get_node_name(), test_case.expected_name);
	}

	// Incomplete start-tag attribute name.
	{
		const String input = "<first></first><second attr1=\"foo\" attr2";
		REQUIRE_EQ(parser.open_buffer(_to_buffer(input)), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_ELEMENT);
		REQUIRE_EQ(parser.get_node_name(), "second");
		REQUIRE_EQ(parser.get_attribute_count(), 1);
		REQUIRE_EQ(parser.get_attribute_name(0), "attr1");
		REQUIRE_EQ(parser.get_attribute_value(0), "foo");
	}

	// Incomplete start-tag attribute unquoted value.
	{
		const String input = "<first></first><second attr1=\"foo\" attr2=bar";
		REQUIRE_EQ(parser.open_buffer(_to_buffer(input)), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_ELEMENT);
		REQUIRE_EQ(parser.get_node_name(), "second");
		REQUIRE_EQ(parser.get_attribute_count(), 1);
		REQUIRE_EQ(parser.get_attribute_name(0), "attr1");
		REQUIRE_EQ(parser.get_attribute_value(0), "foo");
	}

	// Incomplete start-tag attribute quoted value.
	{
		const String input = "<first></first><second attr1=\"foo\" attr2=\"bar";
		REQUIRE_EQ(parser.open_buffer(_to_buffer(input)), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_ELEMENT);
		REQUIRE_EQ(parser.get_node_name(), "second");
		REQUIRE_EQ(parser.get_attribute_count(), 2);
		REQUIRE_EQ(parser.get_attribute_name(0), "attr1");
		REQUIRE_EQ(parser.get_attribute_value(0), "foo");
		REQUIRE_EQ(parser.get_attribute_name(1), "attr2");
		REQUIRE_EQ(parser.get_attribute_value(1), "bar");
	}

	// Incomplete end-tag name.
	{
		const String input = "<first></fir";
		REQUIRE_EQ(parser.open_buffer(_to_buffer(input)), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_ELEMENT_END);
		REQUIRE_EQ(parser.get_node_name(), "fir");
	}

	// Trailing text.
	{
		const String input = "<first></first>example";
		REQUIRE_EQ(parser.open_buffer(_to_buffer(input)), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.read(), OK);
		REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_TEXT);
		REQUIRE_EQ(parser.get_node_data(), "example");
	}

	return true;
}

bool test_cdata() {
	const String input = "<a><![CDATA[my cdata content goes here]]></a>";
	XMLParser parser;
	REQUIRE_EQ(parser.open_buffer(_to_buffer(input)), OK);
	REQUIRE_EQ(parser.read(), OK);
	REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_ELEMENT);
	REQUIRE_EQ(parser.get_node_name(), "a");
	REQUIRE_EQ(parser.read(), OK);
	REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_CDATA);
	REQUIRE_EQ(parser.get_node_name(), "my cdata content goes here");
	REQUIRE_EQ(parser.read(), OK);
	REQUIRE_EQ(parser.get_node_type(), XMLParser::NodeType::NODE_ELEMENT_END);
	REQUIRE_EQ(parser.get_node_name(), "a");

	return true;
}

typedef bool (*TestFunc)();
TestFunc test_funcs[] = {
	test_1,
	test_comments,
	test_premature_endings,
	test_cdata,
	nullptr
};

MainLoop *test() {
	int count = 0;
	int passed = 0;

	while (true) {
		if (!test_funcs[count]) {
			break;
		}
		bool pass = test_funcs[count]();
		if (pass) {
			passed++;
		}
		OS::get_singleton()->print("\t%s\n", pass ? "PASS" : "FAILED");

		count++;
	}

	OS::get_singleton()->print("\n\n\n");
	OS::get_singleton()->print("*************\n");
	OS::get_singleton()->print("***TOTALS!***\n");
	OS::get_singleton()->print("*************\n");

	OS::get_singleton()->print("Passed %i of %i tests\n", passed, count);

	return nullptr;
}
} // namespace TestXMLParser

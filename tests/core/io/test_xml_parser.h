/*************************************************************************/
/*  test_xml_parser.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_XML_PARSER_H
#define TEST_XML_PARSER_H

#include "core/io/xml_parser.h"

#include "tests/test_macros.h"

namespace TestXMLParser {
TEST_CASE("[XMLParser] End-to-end") {
	String source = "<?xml version = \"1.0\" encoding=\"UTF-8\" ?>\
<top attr=\"attr value\">\
  Text&lt;&#65;&#x42;&gt;\
</top>";
	Vector<uint8_t> buff = source.to_utf8_buffer();

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
}
} // namespace TestXMLParser

#endif // TEST_XML_PARSER_H

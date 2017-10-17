/*************************************************************************/
/*  xml_parser.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "xml_parser.h"
#include "print_string.h"
//#define DEBUG_XML

VARIANT_ENUM_CAST(XMLParser::NodeType);

static bool _equalsn(const CharType *str1, const CharType *str2, int len) {
	int i;
	for (i = 0; i < len && str1[i] && str2[i]; ++i)
		if (str1[i] != str2[i])
			return false;

	// if one (or both) of the strings was smaller then they
	// are only equal if they have the same length
	return (i == len) || (str1[i] == 0 && str2[i] == 0);
}

String XMLParser::_replace_special_characters(const String &origstr) {

	int pos = origstr.find("&");
	int oldPos = 0;

	if (pos == -1)
		return origstr;

	String newstr;

	while (pos != -1 && pos < origstr.length() - 2) {
		// check if it is one of the special characters

		int specialChar = -1;
		for (int i = 0; i < (int)special_characters.size(); ++i) {
			const CharType *p = &origstr[pos] + 1;

			if (_equalsn(&special_characters[i][1], p, special_characters[i].length() - 1)) {
				specialChar = i;
				break;
			}
		}

		if (specialChar != -1) {
			newstr += (origstr.substr(oldPos, pos - oldPos));
			newstr += (special_characters[specialChar][0]);
			pos += special_characters[specialChar].length();
		} else {
			newstr += (origstr.substr(oldPos, pos - oldPos + 1));
			pos += 1;
		}

		// find next &
		oldPos = pos;
		pos = origstr.find("&", pos);
	}

	if (oldPos < origstr.length() - 1)
		newstr += (origstr.substr(oldPos, origstr.length() - oldPos));

	return newstr;
}

static inline bool _is_white_space(char c) {
	return (c == ' ' || c == '\t' || c == '\n' || c == '\r');
}

//! sets the state that text was found. Returns true if set should be set
bool XMLParser::_set_text(char *start, char *end) {
	// check if text is more than 2 characters, and if not, check if there is
	// only white space, so that this text won't be reported
	if (end - start < 3) {
		char *p = start;
		for (; p != end; ++p)
			if (!_is_white_space(*p))
				break;

		if (p == end)
			return false;
	}

	// set current text to the parsed text, and replace xml special characters
	String s = String::utf8(start, (int)(end - start));
	node_name = _replace_special_characters(s);

	// current XML node type is text
	node_type = NODE_TEXT;

	return true;
}

void XMLParser::_parse_closing_xml_element() {
	node_type = NODE_ELEMENT_END;
	node_empty = false;
	attributes.clear();

	++P;
	const char *pBeginClose = P;

	while (*P != '>')
		++P;

	node_name = String::utf8(pBeginClose, (int)(P - pBeginClose));
#ifdef DEBUG_XML
	print_line("XML CLOSE: " + node_name);
#endif
	++P;
}

void XMLParser::_ignore_definition() {
	node_type = NODE_UNKNOWN;

	char *F = P;
	// move until end marked with '>' reached
	while (*P != '>')
		++P;
	node_name.parse_utf8(F, P - F);
	++P;
}

bool XMLParser::_parse_cdata() {

	if (*(P + 1) != '[')
		return false;

	node_type = NODE_CDATA;

	// skip '<![CDATA['
	int count = 0;
	while (*P && count < 8) {
		++P;
		++count;
	}

	if (!*P)
		return true;

	char *cDataBegin = P;
	char *cDataEnd = 0;

	// find end of CDATA
	while (*P && !cDataEnd) {
		if (*P == '>' &&
				(*(P - 1) == ']') &&
				(*(P - 2) == ']')) {
			cDataEnd = P - 2;
		}

		++P;
	}

	if (cDataEnd)
		node_name = String::utf8(cDataBegin, (int)(cDataEnd - cDataBegin));
	else
		node_name = "";
#ifdef DEBUG_XML
	print_line("XML CDATA: " + node_name);
#endif

	return true;
}

void XMLParser::_parse_comment() {

	node_type = NODE_COMMENT;
	P += 1;

	char *pCommentBegin = P;

	int count = 1;

	// move until end of comment reached
	while (count) {
		if (*P == '>')
			--count;
		else if (*P == '<')
			++count;

		++P;
	}

	P -= 3;
	node_name = String::utf8(pCommentBegin + 2, (int)(P - pCommentBegin - 2));
	P += 3;
#ifdef DEBUG_XML
	print_line("XML COMMENT: " + node_name);
#endif
}

void XMLParser::_parse_opening_xml_element() {

	node_type = NODE_ELEMENT;
	node_empty = false;
	attributes.clear();

	// find name
	const char *startName = P;

	// find end of element
	while (*P != '>' && !_is_white_space(*P))
		++P;

	const char *endName = P;

	// find attributes
	while (*P != '>') {
		if (_is_white_space(*P))
			++P;
		else {
			if (*P != '/') {
				// we've got an attribute

				// read the attribute names
				const char *attributeNameBegin = P;

				while (!_is_white_space(*P) && *P != '=')
					++P;

				const char *attributeNameEnd = P;
				++P;

				// read the attribute value
				// check for quotes and single quotes, thx to murphy
				while ((*P != '\"') && (*P != '\'') && *P)
					++P;

				if (!*P) // malformatted xml file
					return;

				const char attributeQuoteChar = *P;

				++P;
				const char *attributeValueBegin = P;

				while (*P != attributeQuoteChar && *P)
					++P;

				if (!*P) // malformatted xml file
					return;

				const char *attributeValueEnd = P;
				++P;

				Attribute attr;
				attr.name = String::utf8(attributeNameBegin,
						(int)(attributeNameEnd - attributeNameBegin));

				String s = String::utf8(attributeValueBegin,
						(int)(attributeValueEnd - attributeValueBegin));

				attr.value = _replace_special_characters(s);
				attributes.push_back(attr);
			} else {
				// tag is closed directly
				++P;
				node_empty = true;
				break;
			}
		}
	}

	// check if this tag is closing directly
	if (endName > startName && *(endName - 1) == '/') {
		// directly closing tag
		node_empty = true;
		endName--;
	}

	node_name = String::utf8(startName, (int)(endName - startName));
#ifdef DEBUG_XML
	print_line("XML OPEN: " + node_name);
#endif

	++P;
}

void XMLParser::_parse_current_node() {

	char *start = P;
	node_offset = P - data;

	// more forward until '<' found
	while (*P != '<' && *P)
		++P;

	if (!*P)
		return;

	if (P - start > 0) {
		// we found some text, store it
		if (_set_text(start, P))
			return;
	}

	++P;

	// based on current token, parse and report next element
	switch (*P) {
		case '/':
			_parse_closing_xml_element();
			break;
		case '?':
			_ignore_definition();
			break;
		case '!':
			if (!_parse_cdata())
				_parse_comment();
			break;
		default:
			_parse_opening_xml_element();
			break;
	}
}

uint64_t XMLParser::get_node_offset() const {

	return node_offset;
};

Error XMLParser::seek(uint64_t p_pos) {

	ERR_FAIL_COND_V(!data, ERR_FILE_EOF)
	ERR_FAIL_COND_V(p_pos >= length, ERR_FILE_EOF);

	P = data + p_pos;

	return read();
};

void XMLParser::_bind_methods() {

	ClassDB::bind_method(D_METHOD("read"), &XMLParser::read);
	ClassDB::bind_method(D_METHOD("get_node_type"), &XMLParser::get_node_type);
	ClassDB::bind_method(D_METHOD("get_node_name"), &XMLParser::get_node_name);
	ClassDB::bind_method(D_METHOD("get_node_data"), &XMLParser::get_node_data);
	ClassDB::bind_method(D_METHOD("get_node_offset"), &XMLParser::get_node_offset);
	ClassDB::bind_method(D_METHOD("get_attribute_count"), &XMLParser::get_attribute_count);
	ClassDB::bind_method(D_METHOD("get_attribute_name", "idx"), &XMLParser::get_attribute_name);
	ClassDB::bind_method(D_METHOD("get_attribute_value", "idx"), (String(XMLParser::*)(int) const) & XMLParser::get_attribute_value);
	ClassDB::bind_method(D_METHOD("has_attribute", "name"), &XMLParser::has_attribute);
	ClassDB::bind_method(D_METHOD("get_named_attribute_value", "name"), (String(XMLParser::*)(const String &) const) & XMLParser::get_attribute_value);
	ClassDB::bind_method(D_METHOD("get_named_attribute_value_safe", "name"), &XMLParser::get_attribute_value_safe);
	ClassDB::bind_method(D_METHOD("is_empty"), &XMLParser::is_empty);
	ClassDB::bind_method(D_METHOD("get_current_line"), &XMLParser::get_current_line);
	ClassDB::bind_method(D_METHOD("skip_section"), &XMLParser::skip_section);
	ClassDB::bind_method(D_METHOD("seek", "position"), &XMLParser::seek);
	ClassDB::bind_method(D_METHOD("open", "file"), &XMLParser::open);
	ClassDB::bind_method(D_METHOD("open_buffer", "buffer"), &XMLParser::open_buffer);

	BIND_ENUM_CONSTANT(NODE_NONE);
	BIND_ENUM_CONSTANT(NODE_ELEMENT);
	BIND_ENUM_CONSTANT(NODE_ELEMENT_END);
	BIND_ENUM_CONSTANT(NODE_TEXT);
	BIND_ENUM_CONSTANT(NODE_COMMENT);
	BIND_ENUM_CONSTANT(NODE_CDATA);
	BIND_ENUM_CONSTANT(NODE_UNKNOWN);
};

Error XMLParser::read() {

	// if not end reached, parse the node
	if (P && (P - data) < (int64_t)length - 1 && *P != 0) {
		_parse_current_node();
		return OK;
	}

	return ERR_FILE_EOF;
}

XMLParser::NodeType XMLParser::get_node_type() {

	return node_type;
}
String XMLParser::get_node_data() const {

	ERR_FAIL_COND_V(node_type != NODE_TEXT, "");
	return node_name;
}

String XMLParser::get_node_name() const {
	ERR_FAIL_COND_V(node_type == NODE_TEXT, "");
	return node_name;
}
int XMLParser::get_attribute_count() const {

	return attributes.size();
}
String XMLParser::get_attribute_name(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, attributes.size(), "");
	return attributes[p_idx].name;
}
String XMLParser::get_attribute_value(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, attributes.size(), "");
	return attributes[p_idx].value;
}
bool XMLParser::has_attribute(const String &p_name) const {

	for (int i = 0; i < attributes.size(); i++) {
		if (attributes[i].name == p_name)
			return true;
	}

	return false;
}
String XMLParser::get_attribute_value(const String &p_name) const {

	int idx = -1;
	for (int i = 0; i < attributes.size(); i++) {
		if (attributes[i].name == p_name) {
			idx = i;
			break;
		}
	}

	if (idx < 0) {
		ERR_EXPLAIN("Attribute not found: " + p_name);
	}
	ERR_FAIL_COND_V(idx < 0, "");
	return attributes[idx].value;
}

String XMLParser::get_attribute_value_safe(const String &p_name) const {

	int idx = -1;
	for (int i = 0; i < attributes.size(); i++) {
		if (attributes[i].name == p_name) {
			idx = i;
			break;
		}
	}

	if (idx < 0)
		return "";
	return attributes[idx].value;
}
bool XMLParser::is_empty() const {

	return node_empty;
}

Error XMLParser::open_buffer(const Vector<uint8_t> &p_buffer) {

	ERR_FAIL_COND_V(p_buffer.size() == 0, ERR_INVALID_DATA);

	length = p_buffer.size();
	data = memnew_arr(char, length + 1);
	copymem(data, p_buffer.ptr(), length);
	data[length] = 0;
	P = data;
	return OK;
}

Error XMLParser::open(const String &p_path) {

	Error err;
	FileAccess *file = FileAccess::open(p_path, FileAccess::READ, &err);

	if (err) {
		ERR_FAIL_COND_V(err != OK, err);
	}

	length = file->get_len();
	ERR_FAIL_COND_V(length < 1, ERR_FILE_CORRUPT);

	data = memnew_arr(char, length + 1);
	file->get_buffer((uint8_t *)data, length);
	data[length] = 0;
	P = data;

	memdelete(file);

	return OK;
}

void XMLParser::skip_section() {

	// skip if this element is empty anyway.
	if (is_empty())
		return;

	// read until we've reached the last element in this section
	int tagcount = 1;

	while (tagcount && read() == OK) {
		if (get_node_type() == XMLParser::NODE_ELEMENT &&
				!is_empty()) {
			++tagcount;
		} else if (get_node_type() == XMLParser::NODE_ELEMENT_END)
			--tagcount;
	}
}

void XMLParser::close() {

	if (data)
		memdelete_arr(data);
	data = NULL;
	length = 0;
	P = NULL;
	node_empty = false;
	node_type = NODE_NONE;
	node_offset = 0;
}

int XMLParser::get_current_line() const {

	return 0;
}

XMLParser::XMLParser() {

	data = NULL;
	close();
	special_characters.push_back("&amp;");
	special_characters.push_back("<lt;");
	special_characters.push_back(">gt;");
	special_characters.push_back("\"quot;");
	special_characters.push_back("'apos;");
}
XMLParser::~XMLParser() {

	if (data)
		memdelete_arr(data);
}

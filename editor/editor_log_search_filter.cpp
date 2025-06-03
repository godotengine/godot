/**************************************************************************/
/*  editor_log_search_filter.cpp                                          */
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

#include "editor/editor_log_search_filter.h"

EditorLogSearchFilter::Node::~Node() {
}

EditorLogSearchFilter::ContainsNode::ContainsNode(const String &p_contained, bool p_exact) :
		Node(), contained(p_contained), exact(p_exact) {}

bool EditorLogSearchFilter::ContainsNode::match(const String &p_text) const {
	if (exact) {
		return p_text.contains(contained);
	}
	return p_text.containsn(contained);
}

EditorLogSearchFilter::NotNode::NotNode() {}

EditorLogSearchFilter::NotNode::NotNode(Node *p_argument) :
		Node(), argument(p_argument) {}

EditorLogSearchFilter::NotNode::NotNode(NotNode &&p_other) :
		Node(), argument(p_other.argument) {
	p_other.argument = nullptr;
}

void EditorLogSearchFilter::NotNode::operator=(NotNode &&p_other) {
	if (this == &p_other) {
		return;
	}
	if (argument) {
		memdelete(argument);
	}
	argument = p_other.argument;
	p_other.argument = nullptr;
}

EditorLogSearchFilter::NotNode::~NotNode() {
	if (argument) {
		memdelete(argument);
		argument = nullptr;
	}
}

bool EditorLogSearchFilter::NotNode::match(const String &p_text) const {
	return !argument->match(p_text);
}

EditorLogSearchFilter::BinaryOpNode::BinaryOpNode() :
		Node() {}

EditorLogSearchFilter::BinaryOpNode::BinaryOpNode(Node *p_left, Node *p_right) :
		Node(), left(p_left), right(p_right) {}

EditorLogSearchFilter::BinaryOpNode::BinaryOpNode(BinaryOpNode &&p_other) :
		Node(), left(p_other.left), right(p_other.right) {
	p_other.left = nullptr;
	p_other.right = nullptr;
}

void EditorLogSearchFilter::BinaryOpNode::operator=(BinaryOpNode &&p_other) {
	if (this == &p_other) {
		return;
	}
	if (left) {
		memdelete(left);
	}
	if (right) {
		memdelete(right);
	}
	left = p_other.left;
	right = p_other.right;
	p_other.left = nullptr;
	p_other.right = nullptr;
}

EditorLogSearchFilter::BinaryOpNode::~BinaryOpNode() {
	if (left) {
		memdelete(left);
		left = nullptr;
	}
	if (right) {
		memdelete(right);
		right = nullptr;
	}
}

void EditorLogSearchFilter::BinaryOpNode::set_left(Node *p_left) {
	if (left && left != p_left) {
		memdelete(left);
	}
	left = p_left;
}

void EditorLogSearchFilter::BinaryOpNode::set_right(Node *p_right) {
	if (right && right != p_right) {
		memdelete(right);
	}
	right = p_right;
}

const EditorLogSearchFilter::Node *EditorLogSearchFilter::BinaryOpNode::get_left() const {
	return left;
}

const EditorLogSearchFilter::Node *EditorLogSearchFilter::BinaryOpNode::get_right() const {
	return right;
}

EditorLogSearchFilter::AndNode::AndNode() :
		BinaryOpNode() {}

EditorLogSearchFilter::AndNode::AndNode(Node *p_left, Node *p_right) :
		BinaryOpNode(p_left, p_right) {}

bool EditorLogSearchFilter::AndNode::match(const String &p_text) const {
	return get_left()->match(p_text) && get_right()->match(p_text);
}

EditorLogSearchFilter::OrNode::OrNode() :
		BinaryOpNode() {}

EditorLogSearchFilter::OrNode::OrNode(Node *p_left, Node *p_right) :
		BinaryOpNode(p_left, p_right) {}

bool EditorLogSearchFilter::OrNode::match(const String &p_text) const {
	return get_left()->match(p_text) || get_right()->match(p_text);
}

EditorLogSearchFilter::EditorLogSearchFilter() :
		RefCounted() {}

EditorLogSearchFilter::EditorLogSearchFilter(Node *p_filter) :
		RefCounted(), filter(p_filter) {}

EditorLogSearchFilter::~EditorLogSearchFilter() {
	if (filter) {
		memdelete(filter);
	}
}

bool EditorLogSearchFilter::match(const String &p_text) const {
	return !filter || filter->match(p_text);
}

EditorLogSearchParser::Tokenizer::Token::Token() {}

EditorLogSearchParser::Tokenizer::Token::Token(Type p_type) :
		type(p_type) {}

EditorLogSearchParser::Tokenizer::Token::Token(Type p_type, const String &p_value) :
		type(p_type), value(p_value) {}

EditorLogSearchParser::Tokenizer::Token::Type EditorLogSearchParser::Tokenizer::Token::get_type() const {
	return type;
}

void EditorLogSearchParser::Tokenizer::Token::set_type(Type p_type) {
	type = p_type;
}

const String &EditorLogSearchParser::Tokenizer::Token::get_value() const {
	return value;
}

void EditorLogSearchParser::Tokenizer::Token::set_value(const String &p_value) {
	value = p_value;
}

void EditorLogSearchParser::Tokenizer::Token::append(const String &p_suffix) {
	value += p_suffix;
}

void EditorLogSearchParser::Tokenizer::Token::append(char32_t p_char) {
	value += p_char;
}

String EditorLogSearchParser::Tokenizer::Token::get_name(Type p_type) {
	String name;
	switch (p_type) {
		case EMPTY: {
			name = "";
			break;
		}

		case LITERAL: {
			name = "LITERAL";
			break;
		}

		case QUOTED_LITERAL: {
			name = "QUOTED LITERAL";
			break;
		}

		case DOUBLE_PIPE: {
			name = "||";
			break;
		}

		case MINUS: {
			name = "-";
			break;
		}

		case OPEN_PAREN: {
			name = "(";
			break;
		}

		case CLOSE_PAREN: {
			name = ")";
			break;
		}

		case TK_EOF: {
			name = "EOF";
			break;
		}

		case ERROR: {
			name = "ERROR";
			break;
		}
	}
	return name;
}

char32_t EditorLogSearchParser::Tokenizer::_peek(int p_offset) {
	int idx = position + p_offset;
	if (idx < 0 || idx >= source.length()) {
		return '\0';
	}
	return source[idx];
}

char32_t EditorLogSearchParser::Tokenizer::_advance(unsigned int p_offset) {
	position += p_offset;
	if (position >= source.length()) {
		position = source.length() + 1;
		return '\0';
	}
	return source[position++];
}

char32_t EditorLogSearchParser::Tokenizer::_back(unsigned int p_offset) {
	position -= p_offset;
	if (position <= 0) {
		position = 0;
		return '\0';
	}
	if (position > source.length()) {
		position = source.length();
		return '\0';
	}
	return source[--position];
}

bool EditorLogSearchParser::Tokenizer::_is_whitespace(char32_t p_char) {
	return p_char == ' ';
}

bool EditorLogSearchParser::Tokenizer::_is_delimiter(char32_t p_char) {
	switch (p_char) {
		case '(':
		case ')':
		case '"':
		case '\'':
		case '\0': {
			return true;
		}

		default: {
			return _is_whitespace(p_char);
		}
	}
}

void EditorLogSearchParser::Tokenizer::_consume_whitespace() {
	while (true) {
		if (_is_whitespace(_advance())) {
			continue;
		}
		_back();
		return;
	}
}

/**
 * Parses a token that starts after a delimiter.
 * Note that the delimiter itself should have already been consumed before
 * calling the function.
 */
EditorLogSearchParser::Tokenizer::Token EditorLogSearchParser::Tokenizer::_delimiter_start() {
	char32_t start = _advance();
	switch (start) {
		case '-': {
			return Token(Token::MINUS);
		}

		case '|': {
			if (_peek() != '|' || !_is_delimiter(_peek(1))) {
				break;
			}
			_advance();
			return Token(Token::DOUBLE_PIPE);
		}

		default: {
			break;
		}
	}
	_back();
	return _non_delimiter_start();
}

/**
 * Parses a token that doesn't necessarily come after a delimiter (although
 * the token itself may begin with one).
 */
EditorLogSearchParser::Tokenizer::Token EditorLogSearchParser::Tokenizer::_non_delimiter_start() {
	char32_t start = _advance();
	Token scanned;
	switch (start) {
		case '(': {
			scanned = Token(Token::OPEN_PAREN);
			break;
		}

		case ')': {
			scanned = Token(Token::CLOSE_PAREN);
			break;
		}

		case '\0': {
			scanned = Token(Token::TK_EOF);
			break;
		}

		default: {
			_back();
			_parse_literal(scanned);
			break;
		}
	}
	return scanned;
}

/**
 * Parses a literal in the form: <literal>, "<literal>" or '<literal>'.
 */
void EditorLogSearchParser::Tokenizer::_parse_literal(Token &r_token) {
	char32_t start = _peek();
	bool is_quote = start == '\'' || start == '"';
	if (_is_delimiter(start) && !is_quote) {
		r_token.set_type(Token::EMPTY);
		r_token.set_value("");
		return;
	}

	if (!is_quote) {
		r_token.set_type(Token::LITERAL);
		while (true) {
			char32_t c = _advance();
			if (_is_delimiter(c)) {
				_back();
				return;
			}
			r_token.append(c);
		}
	}

	r_token.set_type(Token::QUOTED_LITERAL);
	_advance();
	while (true) {
		char32_t c = _advance();
		if (c == start) {
			return;
		}

		if (c == '\0') {
			_back();
			r_token.set_type(Token::ERROR);
			r_token.set_value(String("Expected to find closing ") + start);
			return;
		}

		if (c == '\\') {
			c = _advance();
			if (c == '\0') {
				_back(1);
				r_token.set_type(Token::ERROR);
				r_token.set_value(String("Expected to find closing ") + start);
				return;
			}
			r_token.append(c);
			continue;
		}

		r_token.append(c);
	}
}

EditorLogSearchParser::Tokenizer::Tokenizer(const String &p_source) :
		source(p_source) {}

/**
 * Consumes the next token and returns it.
 * It's important to note that the '-' token expects a delimiter (i.e. \0, (, ), ", ' or whitespace)
 * to appear before it and '||' expects delimiters before and after.
 * This means that the string "--a" will read as "-" "-a", "a||b" as "a||b",
 * "a|| b" as "a||" "b" and "a || b" as "a" "||" "b".
 */
EditorLogSearchParser::Tokenizer::Token EditorLogSearchParser::Tokenizer::next() {
	if (peeked) {
		peeked = false;
		return emitted_last;
	}

	if (emitted_last.get_type() == Token::ERROR) {
		return emitted_last;
	}

	Token scanned = Token();
	_consume_whitespace();
	if (_is_delimiter(_peek(-1))) {
		scanned = _delimiter_start();
	} else {
		scanned = _non_delimiter_start();
	}

	emitted_last = scanned;
	return scanned;
}

EditorLogSearchParser::Tokenizer::Token EditorLogSearchParser::Tokenizer::peek() {
	if (peeked) {
		return emitted_last;
	}

	if (emitted_last.get_type() == Token::ERROR) {
		return emitted_last;
	}

	next();
	peeked = true;
	return emitted_last;
}

int EditorLogSearchParser::_get_left_precedence(LeftPrecedence p_op) {
	return 2 * p_op + 1;
}

int EditorLogSearchParser::_get_right_precedence(RightPrecedence p_op) {
	return 2 * (p_op + 1);
}

bool EditorLogSearchParser::_can_begin_expression(const Tokenizer::Token &p_token) {
	switch (p_token.get_type()) {
		case Tokenizer::Token::OPEN_PAREN:
		case Tokenizer::Token::QUOTED_LITERAL:
		case Tokenizer::Token::LITERAL:
		case Tokenizer::Token::MINUS: {
			return true;
		}

		default: {
			return false;
		}
	}
}

EditorLogSearchFilter::Node *EditorLogSearchParser::_parse(int p_min_prec) {
	Tokenizer::Token tlhs = tokenizer.next();
	EditorLogSearchFilter::Node *lhs = nullptr;
	switch (tlhs.get_type()) {
		case Tokenizer::Token::LITERAL: {
			lhs = memnew(EditorLogSearchFilter::ContainsNode(tlhs.get_value(), false));
			break;
		}

		case Tokenizer::Token::QUOTED_LITERAL: {
			lhs = memnew(EditorLogSearchFilter::ContainsNode(tlhs.get_value(), true));
			break;
		}

		case Tokenizer::Token::OPEN_PAREN: {
			lhs = _parse(0);
			if (!lhs) {
				return nullptr;
			}
			Tokenizer::Token close_paren = tokenizer.next().get_type();
			if (close_paren.get_type() != Tokenizer::Token::CLOSE_PAREN) {
				memdelete(lhs);
				error_message = vformat("Expected %s token, got %s instead", Tokenizer::Token::get_name(Tokenizer::Token::CLOSE_PAREN), Tokenizer::Token::get_name(close_paren.get_type()));
				return nullptr;
			}
			break;
		}

		case Tokenizer::Token::MINUS: {
			EditorLogSearchFilter::Node *arg = _parse(_get_right_precedence(R_NOT));
			if (!arg) {
				return nullptr;
			}
			lhs = memnew(EditorLogSearchFilter::NotNode(arg));
			break;
		}

		case Tokenizer::Token::ERROR: {
			error_message = tlhs.get_value();
			return nullptr;
		}

		default: {
			error_message = vformat("Found unexpected %s token", Tokenizer::Token::get_name(tlhs.get_type()));
			return nullptr;
		}
	}

	while (true) {
		int left_prec = 0;
		int right_prec = 0;
		bool injected = false;
		Tokenizer::Token top = tokenizer.peek();
		EditorLogSearchFilter::BinaryOpNode *op = nullptr;

		switch (top.get_type()) {
			case Tokenizer::Token::CLOSE_PAREN:
			case Tokenizer::Token::TK_EOF: {
				return lhs;
			}

			case Tokenizer::Token::DOUBLE_PIPE: {
				op = memnew(EditorLogSearchFilter::OrNode());
				left_prec = _get_left_precedence(L_OR);
				right_prec = _get_right_precedence(R_OR);
				break;
			}

			default: {
				if (!_can_begin_expression(top)) {
					memdelete(lhs);
					error_message = vformat("Found unexpected \"%s\" token", Tokenizer::Token::get_name(top.get_type()));
					return nullptr;
				}
				op = memnew(EditorLogSearchFilter::AndNode());
				injected = true;
				left_prec = _get_left_precedence(L_AND);
				right_prec = _get_right_precedence(R_AND);
				break;
			}
		}

		if (left_prec < p_min_prec) {
			memdelete(op);
			break;
		}
		if (!injected) {
			tokenizer.next();
		}

		EditorLogSearchFilter::Node *rhs = _parse(right_prec);
		if (!rhs) {
			memdelete(lhs);
			memdelete(op);
			return nullptr;
		}
		op->set_left(lhs);
		op->set_right(rhs);
		lhs = op;
	}

	return lhs;
}

EditorLogSearchParser::EditorLogSearchParser(const String &p_filter) :
		tokenizer(p_filter) {}

/**
 * Parses the source string into a filter.
 * The available operations are:
 * 		literal			(message must contain the specified literal, ignoring case)
 * 		"literal"		(message must contain the specified literal, case matters)
 * 		'literal'		(same as above)
 * 		expr || expr	(logical OR);
 * 		-expr			(logical NOT)
 * 		(expr)
 * 		expr expr		(implicitly ANDs both expressions)
 *
 * Note that, unless quoted, literals must not contain spaces. For instance,
 * the string: hello world; will match hello AND world, not hello <space> world.
 */
Error EditorLogSearchParser::parse(Ref<EditorLogSearchFilter> &p_filter) {
	if (tokenizer.peek().get_type() == Tokenizer::Token::TK_EOF) {
		p_filter = memnew(EditorLogSearchFilter());
		return OK;
	}
	EditorLogSearchFilter::Node *parsed = _parse(0);
	if (!parsed) {
		return ERR_PARSE_ERROR;
	}
	p_filter = memnew(EditorLogSearchFilter(parsed));
	return OK;
}

String EditorLogSearchParser::get_error_message() {
	return error_message;
}

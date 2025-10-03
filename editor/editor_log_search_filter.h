/**************************************************************************/
/*  editor_log_search_filter.h                                            */
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

#include "core/object/ref_counted.h"

class EditorLogSearchFilter : public RefCounted {
	GDCLASS(EditorLogSearchFilter, RefCounted);

public:
	class Node {
	public:
		virtual ~Node();

		virtual bool match(const String &p_text) const = 0;
	};

	class ContainsNode : public Node {
		String contained;
		bool exact;

	public:
		ContainsNode(const String &p_contained, bool p_exact);

		bool match(const String &p_text) const override;
	};

	class NotNode : public Node {
		Node *argument = nullptr;

	public:
		NotNode();

		NotNode(Node *p_argument);

		NotNode(NotNode &&p_other);

		void operator=(NotNode &&p_other);

		~NotNode();

		bool match(const String &p_text) const override;

		NotNode(const NotNode &p_other) = delete;

		void operator=(const NotNode &p_other) = delete;
	};

	class BinaryOpNode : public Node {
		Node *left = nullptr;
		Node *right = nullptr;

	public:
		BinaryOpNode();

		BinaryOpNode(Node *p_left, Node *p_right);

		BinaryOpNode(BinaryOpNode &&p_other);

		void operator=(BinaryOpNode &&p_other);

		~BinaryOpNode();

		void set_left(Node *p_left);

		void set_right(Node *p_right);

		const Node *get_left() const;

		const Node *get_right() const;

		BinaryOpNode(const BinaryOpNode &p_other) = delete;

		void operator=(const BinaryOpNode &p_other) = delete;
	};

	class AndNode : public BinaryOpNode {
	public:
		AndNode();

		AndNode(Node *p_left, Node *p_right);

		bool match(const String &p_text) const override;
	};

	class OrNode : public BinaryOpNode {
	public:
		OrNode();

		OrNode(Node *p_left, Node *p_right);

		bool match(const String &p_text) const override;
	};

private:
	Node *filter = nullptr;

public:
	EditorLogSearchFilter();

	EditorLogSearchFilter(Node *p_filter);

	~EditorLogSearchFilter();

	bool match(const String &p_text) const;
};

class EditorLogSearchParser {
	class Tokenizer {
	public:
		struct Token {
			enum Type {
				EMPTY,
				LITERAL,
				QUOTED_LITERAL,
				DOUBLE_PIPE,
				MINUS,
				OPEN_PAREN,
				CLOSE_PAREN,
				TK_EOF,
				ERROR
			};

		private:
			Type type = EMPTY;
			String value = "";

		public:
			Token();

			Token(Type p_type);

			Token(Type p_type, const String &p_value);

			Type get_type() const;

			void set_type(Type p_type);

			const String &get_value() const;

			void set_value(const String &p_value);

			void append(const String &p_suffix);

			void append(char32_t p_char);

			static String get_name(Type p_type);
		};

	private:
		String source;
		int position = 0;
		bool peeked = false;
		Token emitted_last = Token();

		char32_t _peek(int p_offset = 0);

		char32_t _advance(unsigned int p_offset = 0);

		char32_t _back(unsigned int p_offset = 0);

		bool _is_whitespace(char32_t p_char);

		bool _is_delimiter(char32_t p_char);

		void _consume_whitespace();

		Token _delimiter_start();

		Token _non_delimiter_start();

		void _parse_literal(Token &r_token);

	public:
		Tokenizer(const String &p_source);

		Token next();

		Token peek();
	};

	Tokenizer tokenizer;
	String error_message = "";

	enum LeftPrecedence {
		L_OR,
		L_AND
	};

	enum RightPrecedence {
		R_OR,
		R_AND,
		R_NOT
	};

	int _get_left_precedence(LeftPrecedence p_op);

	int _get_right_precedence(RightPrecedence p_op);

	bool _can_begin_expression(const Tokenizer::Token &p_token);

	EditorLogSearchFilter::Node *_parse(int p_min_prec);

public:
	EditorLogSearchParser(const String &p_filter);

	Error parse(Ref<EditorLogSearchFilter> &r_filter);

	String get_error_message();
};

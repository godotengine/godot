/**************************************************************************/
/*  gdscript_tokenizer.h                                                  */
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

#include "core/templates/hash_map.h"
#include "core/templates/list.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

class GDScriptTokenizer {
public:
	enum CursorPlace {
		CURSOR_NONE,
		CURSOR_BEGINNING,
		CURSOR_MIDDLE,
		CURSOR_END,
	};

	typedef Pair<int, int> LineColumn;

	struct CodeArea {
		LineColumn start;
		LineColumn end;

	public:
		constexpr bool is_overlapping(const LineColumn &p_other) const {
			return !(is_before(p_other) || is_after(p_other));
		}
		constexpr bool is_overlapping(const CodeArea &p_other) const {
			return !(is_before(p_other) || is_after(p_other));
		}
		constexpr bool is_overlapping(int p_line, int p_column) const {
			return is_overlapping(LineColumn{ p_line, p_column });
		}
		constexpr bool is_overlapping(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return is_overlapping({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool is_overlapping(Vector2i p_position) const {
			return is_overlapping(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool is_overlapping(Vector2i p_start, Vector2i p_end) const {
			return is_overlapping({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool is_before(const LineColumn &p_other) const {
			return end < p_other;
		}
		constexpr bool is_before(const CodeArea &p_other) const {
			return is_before(p_other.start);
		}
		constexpr bool is_before(int p_line, int p_column) const {
			return is_before(LineColumn{ p_line, p_column });
		}
		constexpr bool is_before(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return is_before({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool is_before(Vector2i p_position) const {
			return is_before(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool is_before(Vector2i p_start, Vector2i p_end) const {
			return is_before({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool is_after(const LineColumn &p_other) const {
			return p_other < start;
		}
		constexpr bool is_after(const CodeArea &p_other) const {
			return p_other.end < start;
		}
		constexpr bool is_after(int p_line, int p_column) const {
			return is_after(LineColumn{ p_line, p_column });
		}
		constexpr bool is_after(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return is_after({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool is_after(Vector2i p_position) const {
			return is_after(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool is_after(Vector2i p_start, Vector2i p_end) const {
			return is_after({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool starts_together(const LineColumn &p_other) const {
			return start == p_other;
		}
		constexpr bool starts_together(const CodeArea &p_other) const {
			return starts_together(p_other.start);
		}
		constexpr bool starts_together(int p_line, int p_column) const {
			return starts_together(LineColumn{ p_line, p_column });
		}
		constexpr bool starts_together(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return starts_together({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool starts_together(Vector2i p_position) const {
			return starts_together(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool starts_together(Vector2i p_start, Vector2i p_end) const {
			return starts_together({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool starts_before(const LineColumn &p_other) const {
			return start < p_other;
		}
		constexpr bool starts_before(const CodeArea &p_other) const {
			return starts_before(p_other.start);
		}
		constexpr bool starts_before(int p_line, int p_column) const {
			return starts_before(LineColumn{ p_line, p_column });
		}
		constexpr bool starts_before(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return starts_before({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool starts_before(Vector2i p_position) const {
			return starts_before(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool starts_before(Vector2i p_start, Vector2i p_end) const {
			return starts_before({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool starts_before_or_together(const LineColumn &p_other) const {
			return start <= p_other;
		}
		constexpr bool starts_before_or_together(const CodeArea &p_other) const {
			return starts_before_or_together(p_other.start);
		}
		constexpr bool starts_before_or_together(int p_line, int p_column) const {
			return starts_before_or_together(LineColumn{ p_line, p_column });
		}
		constexpr bool starts_before_or_together(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return starts_before_or_together({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool starts_before_or_together(Vector2i p_position) const {
			return starts_before_or_together(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool starts_before_or_together(Vector2i p_start, Vector2i p_end) const {
			return starts_before_or_together({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool starts_after(const LineColumn &p_other) const {
			return start > p_other;
		}
		constexpr bool starts_after(const CodeArea &p_other) const {
			return starts_after(p_other.start);
		}
		constexpr bool starts_after(int p_line, int p_column) const {
			return starts_after(LineColumn{ p_line, p_column });
		}
		constexpr bool starts_after(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return starts_after({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool starts_after(Vector2i p_position) const {
			return starts_after(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool starts_after(Vector2i p_start, Vector2i p_end) const {
			return starts_after({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool starts_after_or_together(const LineColumn &p_other) const {
			return start >= p_other;
		}
		constexpr bool starts_after_or_together(const CodeArea &p_other) const {
			return starts_after_or_together(p_other.start);
		}
		constexpr bool starts_after_or_together(int p_line, int p_column) const {
			return starts_after_or_together(LineColumn{ p_line, p_column });
		}
		constexpr bool starts_after_or_together(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return starts_after_or_together({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool starts_after_or_together(Vector2i p_position) const {
			return starts_after_or_together(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool starts_after_or_together(Vector2i p_start, Vector2i p_end) const {
			return starts_after_or_together({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool ends_together(const LineColumn &p_other) const {
			return end == p_other;
		}
		constexpr bool ends_together(const CodeArea &p_other) const {
			return ends_together(p_other.end);
		}
		constexpr bool ends_together(int p_line, int p_column) const {
			return ends_together(LineColumn{ p_line, p_column });
		}
		constexpr bool ends_together(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return ends_together({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool ends_together(Vector2i p_position) const {
			return ends_together(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool ends_together(Vector2i p_start, Vector2i p_end) const {
			return ends_together({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool ends_before(const LineColumn &p_other) const {
			return end < p_other;
		}
		constexpr bool ends_before(const CodeArea &p_other) const {
			return ends_before(p_other.end);
		}
		constexpr bool ends_before(int p_line, int p_column) const {
			return ends_before(LineColumn{ p_line, p_column });
		}
		constexpr bool ends_before(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return ends_before({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool ends_before(Vector2i p_position) const {
			return ends_before(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool ends_before(Vector2i p_start, Vector2i p_end) const {
			return ends_before({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool ends_before_or_together(const LineColumn &p_other) const {
			return end <= p_other;
		}
		constexpr bool ends_before_or_together(const CodeArea &p_other) const {
			return ends_before_or_together(p_other.end);
		}
		constexpr bool ends_before_or_together(int p_line, int p_column) const {
			return ends_before_or_together(LineColumn{ p_line, p_column });
		}
		constexpr bool ends_before_or_together(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return ends_before_or_together({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool ends_before_or_together(Vector2i p_position) const {
			return ends_before_or_together(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool ends_before_or_together(Vector2i p_start, Vector2i p_end) const {
			return ends_before_or_together({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool ends_after(const LineColumn &p_other) const {
			return end > p_other;
		}
		constexpr bool ends_after(const CodeArea &p_other) const {
			return ends_after(p_other.end);
		}
		constexpr bool ends_after(int p_line, int p_column) const {
			return ends_after(LineColumn{ p_line, p_column });
		}
		constexpr bool ends_after(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return ends_after({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool ends_after(Vector2i p_position) const {
			return ends_after(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool ends_after(Vector2i p_start, Vector2i p_end) const {
			return ends_after({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool ends_after_or_together(const LineColumn &p_other) const {
			return end >= p_other;
		}
		constexpr bool ends_after_or_together(const CodeArea &p_other) const {
			return ends_after_or_together(p_other.end);
		}
		constexpr bool ends_after_or_together(int p_line, int p_column) const {
			return ends_after_or_together(LineColumn{ p_line, p_column });
		}
		constexpr bool ends_after_or_together(int p_start_line, int p_start_column, int p_end_line, int p_end_column) const {
			return ends_after_or_together({ { p_start_line, p_start_column }, { p_end_line, p_end_column } });
		}
		constexpr bool ends_after_or_together(Vector2i p_position) const {
			return ends_after_or_together(LineColumn{ p_position.y, p_position.x });
		}
		constexpr bool ends_after_or_together(Vector2i p_start, Vector2i p_end) const {
			return ends_after_or_together({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool contains(const LineColumn &p_other, bool p_allow_start_overlap = true, bool p_allow_end_overlap = false) const {
			if (p_allow_start_overlap) {
				if (!starts_before_or_together(p_other)) {
					return false;
				}
			} else if (!starts_before(p_other)) {
				return false;
			}

			if (p_allow_end_overlap) {
				return ends_after_or_together(p_other);
			}
			return ends_after(p_other);
		}
		constexpr bool contains(const CodeArea &p_other, bool p_allow_overlap = true) const {
			return contains(p_other.start, p_allow_overlap, p_allow_overlap) && contains(p_other.end, p_allow_overlap, p_allow_overlap);
		}
		constexpr bool contains(int p_line, int p_column, bool p_allow_overlap = true) const {
			return contains(LineColumn{ p_line, p_column }, p_allow_overlap);
		}
		constexpr bool contains(int p_start_line, int p_start_column, int p_end_line, int p_end_column, bool p_allow_overlap = true) const {
			return contains({ { p_start_line, p_start_column }, { p_end_line, p_end_column } }, p_allow_overlap);
		}
		constexpr bool contains(Vector2i p_position, bool p_allow_start_overlap = true, bool p_allow_end_overlap = false) const {
			return contains(LineColumn{ p_position.y, p_position.x }, p_allow_start_overlap, p_allow_end_overlap);
		}
		constexpr bool contains(Vector2i p_start, Vector2i p_end, bool p_allow_overlap = true) const {
			return contains({ { p_start.y, p_start.x }, { p_end.y, p_end.x } });
		}

		constexpr bool operator==(const CodeArea &p_other) const {
			return start == p_other.start && end == p_other.end;
		}

		constexpr int get_start_line() const { return start.first; }
		constexpr int get_start_column() const { return start.second; }
		constexpr int get_end_line() const { return end.first; }
		constexpr int get_end_column() const { return end.second; }

		operator String() const {
			return vformat("CodeArea(L%sC%s, L%sC%s)",
					start.first, start.second,
					end.first, end.second);
		}

		constexpr CodeArea() = default;
		constexpr CodeArea(LineColumn p_start, LineColumn p_end) :
				start(p_start), end(p_end) {
			ERR_FAIL_COND(start > end);
		}
		constexpr CodeArea(int p_start_line, int p_start_column, int p_end_line, int p_end_column) :
				start({ p_start_line, p_start_column }), end({ p_end_line, p_end_column }) {
			ERR_FAIL_COND(start > end);
		}
	};

	struct Token {
		// If this enum changes, please increment the TOKENIZER_VERSION in gdscript_tokenizer_buffer.h
		enum Type {
			EMPTY,
			// Basic
			ANNOTATION,
			IDENTIFIER,
			LITERAL,
			// Comparison
			LESS,
			LESS_EQUAL,
			GREATER,
			GREATER_EQUAL,
			EQUAL_EQUAL,
			BANG_EQUAL,
			// Logical
			AND,
			OR,
			NOT,
			AMPERSAND_AMPERSAND,
			PIPE_PIPE,
			BANG,
			// Bitwise
			AMPERSAND,
			PIPE,
			TILDE,
			CARET,
			LESS_LESS,
			GREATER_GREATER,
			// Math
			PLUS,
			MINUS,
			STAR,
			STAR_STAR,
			SLASH,
			PERCENT,
			// Assignment
			EQUAL,
			PLUS_EQUAL,
			MINUS_EQUAL,
			STAR_EQUAL,
			STAR_STAR_EQUAL,
			SLASH_EQUAL,
			PERCENT_EQUAL,
			LESS_LESS_EQUAL,
			GREATER_GREATER_EQUAL,
			AMPERSAND_EQUAL,
			PIPE_EQUAL,
			CARET_EQUAL,
			// Control flow
			IF,
			ELIF,
			ELSE,
			FOR,
			WHILE,
			BREAK,
			CONTINUE,
			PASS,
			RETURN,
			MATCH,
			WHEN,
			// Keywords
			AS,
			ASSERT,
			AWAIT,
			BREAKPOINT,
			CLASS,
			CLASS_NAME,
			TK_CONST, // Conflict with WinAPI.
			ENUM,
			EXTENDS,
			FUNC,
			TK_IN, // Conflict with WinAPI.
			IS,
			NAMESPACE,
			PRELOAD,
			SELF,
			SIGNAL,
			STATIC,
			SUPER,
			TRAIT,
			VAR,
			TK_VOID, // Conflict with WinAPI.
			YIELD,
			// Punctuation
			BRACKET_OPEN,
			BRACKET_CLOSE,
			BRACE_OPEN,
			BRACE_CLOSE,
			PARENTHESIS_OPEN,
			PARENTHESIS_CLOSE,
			COMMA,
			SEMICOLON,
			PERIOD,
			PERIOD_PERIOD,
			PERIOD_PERIOD_PERIOD,
			COLON,
			DOLLAR,
			FORWARD_ARROW,
			UNDERSCORE,
			// Whitespace
			NEWLINE,
			INDENT,
			DEDENT,
			// Constants
			CONST_PI,
			CONST_TAU,
			CONST_INF,
			CONST_NAN,
			// Error message improvement
			VCS_CONFLICT_MARKER,
			BACKTICK,
			QUESTION_MARK,
			// Special
			ERROR,
			TK_EOF, // "EOF" is reserved
			TK_MAX
		};

		Type type = EMPTY;
		Variant literal;
		int start_line = 0, end_line = 0, start_column = 0, end_column = 0;
		int cursor_position = -1;
		CursorPlace cursor_place = CURSOR_NONE;
		String source;

		const char *get_name() const;
		String get_debug_name() const;
		bool can_precede_bin_op() const;
		bool is_identifier() const;
		bool is_node_name() const;
		bool has_cursor() const;
		StringName get_identifier() const { return literal; }

		constexpr LineColumn get_start() const { return { start_line, start_column }; }
		constexpr LineColumn get_end() const { return { end_line, end_column }; }
		constexpr CodeArea get_code_area() const { return { get_start(), get_end() }; }

		constexpr bool operator==(const Token &p_right) {
			if (type != p_right.type) {
				return false;
			}
			if (literal != p_right.literal) {
				return false;
			}
			if (start_line != p_right.start_line || start_column != p_right.start_column) {
				return false;
			}
			if (end_line != p_right.end_line || end_column != p_right.end_column) {
				return false;
			}
			if (cursor_position != p_right.cursor_position) {
				return false;
			}
			if (cursor_place != p_right.cursor_place) {
				return false;
			}
			if (source != p_right.source) {
				return false;
			}
			return true;
		}
		constexpr bool operator!=(const Token &p_right) {
			return !operator==(p_right);
		}

		Token(Type p_type) {
			type = p_type;
		}

		Token() {}
	};

#ifdef TOOLS_ENABLED
	struct CommentData {
		String comment;
		// true: Comment starts at beginning of line or after indentation.
		// false: Inline comment (starts after some code).
		bool new_line = false;
		CommentData() {}
		CommentData(const String &p_comment, bool p_new_line) {
			comment = p_comment;
			new_line = p_new_line;
		}
	};
	virtual const HashMap<int, CommentData> &get_comments() const = 0;
#endif // TOOLS_ENABLED

	static String get_token_name(Token::Type p_token_type);

#ifdef TOOLS_ENABLED
	// This is a temporary solution, as Tokens are not able to store their position, only lines and columns.
	virtual int get_current_position() const { return 0; }
	virtual String get_source_code() const { return ""; }
#endif // TOOLS_ENABLED

	virtual int get_cursor_line() const = 0;
	virtual int get_cursor_column() const = 0;
	virtual void set_cursor_position(int p_line, int p_column) = 0;
	virtual void set_multiline_mode(bool p_state) = 0;
	virtual bool is_past_cursor() const = 0;
	virtual void push_expression_indented_block() = 0; // For lambdas, or blocks inside expressions.
	virtual void pop_expression_indented_block() = 0; // For lambdas, or blocks inside expressions.
	virtual bool is_text() = 0;

	virtual Token scan() = 0;

	virtual ~GDScriptTokenizer() {}
};

class GDScriptTokenizerText : public GDScriptTokenizer {
	String source;
	const char32_t *_source = nullptr;
	const char32_t *_current = nullptr;
	int line = -1, column = -1;
	int cursor_line = -1, cursor_column = -1;
	int tab_size = 4;

	// Keep track of multichar tokens.
	const char32_t *_start = nullptr;
	int start_line = 0, start_column = 0;

	// Info cache.
	bool line_continuation = false; // Whether this line is a continuation of the previous, like when using '\'.
	bool multiline_mode = false;
	List<Token> error_stack;
	bool pending_newline = false;
	Token last_token;
	Token last_newline;
	int pending_indents = 0;
	List<int> indent_stack;
	List<List<int>> indent_stack_stack; // For lambdas, which require manipulating the indentation point.
	List<char32_t> paren_stack;
	char32_t indent_char = '\0';
	int position = 0;
	int length = 0;
	Vector<int> continuation_lines;
#ifdef DEBUG_ENABLED
	Vector<String> keyword_list;
#endif // DEBUG_ENABLED

#ifdef TOOLS_ENABLED
	HashMap<int, CommentData> comments;
#endif // TOOLS_ENABLED

	_FORCE_INLINE_ bool _is_at_end() { return position >= length; }
	_FORCE_INLINE_ char32_t _peek(int p_offset = 0) { return position + p_offset >= 0 && position + p_offset < length ? _current[p_offset] : '\0'; }
	int indent_level() const { return indent_stack.size(); }
	bool has_error() const { return !error_stack.is_empty(); }
	Token pop_error();
	char32_t _advance();
	String _get_indent_char_name(char32_t ch);
	void _skip_whitespace();
	void check_indent();

#ifdef DEBUG_ENABLED
	void make_keyword_list();
#endif // DEBUG_ENABLED

	Token make_error(const String &p_message);
	void push_error(const String &p_message);
	void push_error(const Token &p_error);
	Token make_paren_error(char32_t p_paren);
	Token make_token(Token::Type p_type);
	Token make_literal(const Variant &p_literal);
	Token make_identifier(const StringName &p_identifier);
	Token check_vcs_marker(char32_t p_test, Token::Type p_double_type);
	void push_paren(char32_t p_char);
	bool pop_paren(char32_t p_expected);

	void newline(bool p_make_token);
	Token number();
	Token potential_identifier();
	Token string();
	Token annotation();

public:
	void set_source_code(const String &p_source_code);

	const Vector<int> &get_continuation_lines() const { return continuation_lines; }

#ifdef TOOLS_ENABLED
	virtual int get_current_position() const override { return position; }
	virtual String get_source_code() const override { return source; }
#endif // TOOLS_ENABLED

	virtual int get_cursor_line() const override;
	virtual int get_cursor_column() const override;
	virtual void set_cursor_position(int p_line, int p_column) override;
	virtual void set_multiline_mode(bool p_state) override;
	virtual bool is_past_cursor() const override;
	virtual void push_expression_indented_block() override; // For lambdas, or blocks inside expressions.
	virtual void pop_expression_indented_block() override; // For lambdas, or blocks inside expressions.
	virtual bool is_text() override { return true; }

#ifdef TOOLS_ENABLED
	virtual const HashMap<int, CommentData> &get_comments() const override {
		return comments;
	}
#endif // TOOLS_ENABLED

	virtual Token scan() override;

	GDScriptTokenizerText();
};

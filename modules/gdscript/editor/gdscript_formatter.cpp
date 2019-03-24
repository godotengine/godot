/*************************************************************************/
/*  gdscript_formatter.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "gdscript_formatter.h"

static bool is_end_of_expression(GDScriptTokenizer::Token p_token) {
	switch (p_token) {
		// Any identifier or constant, built-in or otherwise, indicates an expression
		// A closing bracket, curly brace, or parenthesis also indicates an expression
		case GDScriptTokenizer::TK_IDENTIFIER:
		case GDScriptTokenizer::TK_CONSTANT:
		case GDScriptTokenizer::TK_SELF:
		case GDScriptTokenizer::TK_BUILT_IN_TYPE:
		case GDScriptTokenizer::TK_BUILT_IN_FUNC:
		case GDScriptTokenizer::TK_BRACKET_CLOSE:
		case GDScriptTokenizer::TK_CURLY_BRACKET_CLOSE:
		case GDScriptTokenizer::TK_PARENTHESIS_CLOSE:
		case GDScriptTokenizer::TK_CONST_PI:
		case GDScriptTokenizer::TK_CONST_TAU:
		case GDScriptTokenizer::TK_WILDCARD:
		case GDScriptTokenizer::TK_CONST_INF:
		case GDScriptTokenizer::TK_CONST_NAN:
			return true;
		default:
			// Anything else is a separator of some kind
			return false;
	}
}

static bool should_insert_space(GDScriptTokenizer::Token p_left, GDScriptTokenizer::Token p_cur, GDScriptTokenizer::Token p_right) {
	// New lines and EOF

	if (p_right == GDScriptTokenizer::TK_EMPTY || p_right == GDScriptTokenizer::TK_EOF) {
		return false;
	}

	if (p_cur == GDScriptTokenizer::TK_NEWLINE || p_right == GDScriptTokenizer::TK_NEWLINE) {
		return false;
	}

	if (p_cur == GDScriptTokenizer::TK_EOF) {
		return false;
	}

    // Unspaced binary operators
    if (p_cur == GDScriptTokenizer::TK_OP_MUL || p_cur == GDScriptTokenizer::TK_OP_DIV || p_cur == GDScriptTokenizer::TK_OP_MOD) {
        return false;
    }

    if (p_right == GDScriptTokenizer::TK_OP_MUL || p_right == GDScriptTokenizer::TK_OP_DIV || p_right == GDScriptTokenizer::TK_OP_MOD) {
        return false;
    }

	// Parentheses

	if (p_cur == GDScriptTokenizer::TK_PARENTHESIS_OPEN) {
		return false;
	}

	if (p_right == GDScriptTokenizer::TK_PARENTHESIS_OPEN) {
		// identifier(
		if (p_cur == GDScriptTokenizer::TK_IDENTIFIER) {
			return false;
		}

		// self(
		// built_in_type(
		// built_in_func(
		if (p_cur == GDScriptTokenizer::TK_SELF || p_cur == GDScriptTokenizer::TK_BUILT_IN_TYPE || p_cur == GDScriptTokenizer::TK_BUILT_IN_FUNC) {
			return false;
		}

		// preload(
		// assert(
		if (p_cur == GDScriptTokenizer::TK_PR_PRELOAD || p_cur == GDScriptTokenizer::TK_PR_ASSERT) {
			return false;
		}

		// )(
		if (p_cur == GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
			return false;
		}
	}

	if (p_right == GDScriptTokenizer::TK_PARENTHESIS_CLOSE) {
		return false;
	}

	// Brackets

	if (p_cur == GDScriptTokenizer::TK_BRACKET_OPEN) {
		return false;
	}

	if (p_right == GDScriptTokenizer::TK_BRACKET_OPEN) {
		// Check whether we are indexing into an expression, or we have the start of a list literal.
		// We should not insert a space in the former case, but we should in the latter.
		return !is_end_of_expression(p_cur);
	}

	if (p_right == GDScriptTokenizer::TK_BRACKET_CLOSE) {
		return false;
	}

	// Curly brackets

	if (p_cur == GDScriptTokenizer::TK_CURLY_BRACKET_OPEN) {
		// We always want a space after a curly bracket, unless it's immediately followed by a
		// closing culy bracket.
		return p_right != GDScriptTokenizer::TK_CURLY_BRACKET_CLOSE;
	}

	// Unary operators

	if (p_cur == GDScriptTokenizer::TK_OP_BIT_INVERT) {
		return false;
	}

	if (p_cur == GDScriptTokenizer::TK_DOLLAR) {
		return false;
	}

	if (p_cur == GDScriptTokenizer::TK_OP_SUB) {
		// Check if we have a unary minus sign by looking at the previous token.
		// If the previous token is the end of an expression, then we have a binary minus.
		// Otherwise, we have a unary minus.
		return is_end_of_expression(p_left);
	}

	// Other separators

	if (p_right == GDScriptTokenizer::TK_COMMA) {
		return false;
	}

	if (p_cur == GDScriptTokenizer::TK_PERIOD || p_right == GDScriptTokenizer::TK_PERIOD) {
		return false;
	}

	if (p_right == GDScriptTokenizer::TK_COLON) {
		return false;
	}

	return true;
}

GDScriptFormatter::GDScriptFormatter(const String &p_source) :
		_source(&p_source[0]),
		_tokenizer(memnew(GDScriptTokenizerText)) {
	_tokenizer->set_code(p_source);
}

GDScriptFormatter::~GDScriptFormatter() {
	memdelete(_tokenizer);
}

bool GDScriptFormatter::format() {
	while (_tokenizer->get_token() != GDScriptTokenizer::TK_EMPTY) {
		// Comments and '\\' (and any whitespace afterwards) are skipped by the tokenizer, so in order to retain them,
		// check the 'between_text' for non-whitespace, then if there is any, copy all of it.
		int prev_end = 0;
		if (_tokenizer->get_token(-1) != GDScriptTokenizer::TK_EMPTY) {
			prev_end = _tokenizer->get_token_code_pos(-1) + _tokenizer->get_token_code_len(-1);
		}

		int start = _tokenizer->get_token_code_pos();
		String between_text;
		for (int i = prev_end; i < start; i++) {
			between_text += _source[i];
		}
		if (between_text.strip_edges().length() > 0) {
			_formatted += between_text;
		}

		switch (_tokenizer->get_token()) {
			case GDScriptTokenizer::TK_EOF:
				return true;
			default: {
				// Special case for unary '!' operator which requires no space after it
				if (_tokenizer->get_token() == GDScriptTokenizer::TK_OP_NOT && _source[_tokenizer->get_token_code_pos()] == '!') {
					_formatted += '!';
					break;
				}

				int end = start + _tokenizer->get_token_code_len();

				for (int i = start; i < end; i++) {
					_formatted += _source[i];
				}

				if (should_insert_space(_tokenizer->get_token(-1), _tokenizer->get_token(), _tokenizer->get_token(1))) {
					_formatted += " ";
				}

				break;
			}
		}

		_tokenizer->advance();
		if (_tokenizer->get_token() == GDScriptTokenizer::TK_ERROR) {
			return false;
		}
	}

	return false;
}

String GDScriptFormatter::formatted() {
	return _formatted.as_string().rstrip("\n \t");
}

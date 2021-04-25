/*************************************************************************/
/*  FBXTokenizer.cpp                                                     */
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

/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/

/** @file  FBXTokenizer.cpp
 *  @brief Implementation of the FBX broadphase lexer
 */

// tab width for logging columns
#define ASSIMP_FBX_TAB_WIDTH 4

#include "FBXTokenizer.h"
#include "core/string/print_string.h"

namespace FBXDocParser {

// ------------------------------------------------------------------------------------------------
Token::Token(const char *p_sbegin, const char *p_send, TokenType p_type, unsigned int p_line, unsigned int p_column) :
		sbegin(p_sbegin),
		send(p_send),
		type(p_type),
		line(p_line),
		column(p_column) {
#ifdef DEBUG_ENABLED
	contents = std::string(sbegin, static_cast<size_t>(send - sbegin));
#endif
}

// ------------------------------------------------------------------------------------------------
Token::~Token() {
}

namespace {

// ------------------------------------------------------------------------------------------------
void TokenizeError(const std::string &message, unsigned int line, unsigned int column) {
	print_error("[FBX-Tokenize]" + String(message.c_str()) + " " + itos(line) + ":" + itos(column));
}

// process a potential data token up to 'cur', adding it to 'output_tokens'.
// ------------------------------------------------------------------------------------------------
void ProcessDataToken(TokenList &output_tokens, const char *&start, const char *&end,
		unsigned int line,
		unsigned int column,
		TokenType type = TokenType_DATA,
		bool must_have_token = false) {
	if (start && end) {
		// sanity check:
		// tokens should have no whitespace outside quoted text and [start,end] should
		// properly delimit the valid range.
		bool in_double_quotes = false;
		for (const char *c = start; c != end + 1; ++c) {
			if (*c == '\"') {
				in_double_quotes = !in_double_quotes;
			}

			if (!in_double_quotes && IsSpaceOrNewLine(*c)) {
				TokenizeError("unexpected whitespace in token", line, column);
			}
		}

		if (in_double_quotes) {
			TokenizeError("non-terminated double quotes", line, column);
		}

		output_tokens.push_back(new_Token(start, end + 1, type, line, column));
	} else if (must_have_token) {
		TokenizeError("unexpected character, expected data token", line, column);
	}

	start = end = nullptr;
}
} // namespace

// ------------------------------------------------------------------------------------------------
void Tokenize(TokenList &output_tokens, const char *input, size_t length, bool &corrupt) {
	// line and column numbers numbers are one-based
	unsigned int line = 1;
	unsigned int column = 1;

	bool comment = false;
	bool in_double_quotes = false;
	bool pending_data_token = false;

	const char *token_begin = nullptr, *token_end = nullptr;

	// input (starting string), *cur the current string, column +=
	// modified to fix strlen() and stop buffer overflow
	for (size_t x = 0; x < length; x++) {
		const char c = input[x];
		const char *cur = &input[x];
		column += (c == '\t' ? ASSIMP_FBX_TAB_WIDTH : 1);

		if (IsLineEnd(c)) {
			comment = false;

			column = 0;
			++line;
		}

		if (comment) {
			continue;
		}

		if (in_double_quotes) {
			if (c == '\"') {
				in_double_quotes = false;
				token_end = cur;

				ProcessDataToken(output_tokens, token_begin, token_end, line, column);
				pending_data_token = false;
			}
			continue;
		}

		switch (c) {
			case '\"':
				if (token_begin) {
					TokenizeError("unexpected double-quote", line, column);
					corrupt = true;
					return;
				}
				token_begin = cur;
				in_double_quotes = true;
				continue;

			case ';':
				ProcessDataToken(output_tokens, token_begin, token_end, line, column);
				comment = true;
				continue;

			case '{':
				ProcessDataToken(output_tokens, token_begin, token_end, line, column);
				output_tokens.push_back(new_Token(cur, cur + 1, TokenType_OPEN_BRACKET, line, column));
				continue;

			case '}':
				ProcessDataToken(output_tokens, token_begin, token_end, line, column);
				output_tokens.push_back(new_Token(cur, cur + 1, TokenType_CLOSE_BRACKET, line, column));
				continue;

			case ',':
				if (pending_data_token) {
					ProcessDataToken(output_tokens, token_begin, token_end, line, column, TokenType_DATA, true);
				}
				output_tokens.push_back(new_Token(cur, cur + 1, TokenType_COMMA, line, column));
				continue;

			case ':':
				if (pending_data_token) {
					ProcessDataToken(output_tokens, token_begin, token_end, line, column, TokenType_KEY, true);
				} else {
					TokenizeError("unexpected colon", line, column);
				}
				continue;
		}

		if (IsSpaceOrNewLine(c)) {
			if (token_begin) {
				// peek ahead and check if the next token is a colon in which
				// case this counts as KEY token.
				TokenType type = TokenType_DATA;
				for (const char *peek = cur; *peek && IsSpaceOrNewLine(*peek); ++peek) {
					if (*peek == ':') {
						type = TokenType_KEY;
						cur = peek;
						break;
					}
				}

				ProcessDataToken(output_tokens, token_begin, token_end, line, column, type);
			}

			pending_data_token = false;
		} else {
			token_end = cur;
			if (!token_begin) {
				token_begin = cur;
			}

			pending_data_token = true;
		}
	}
}
} // namespace FBXDocParser

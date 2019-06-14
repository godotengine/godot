/*************************************************************************/
/*  gdscript_extend_parser.h                                             */
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

#ifndef GDSCRIPT_EXTEND_PARSER_H
#define GDSCRIPT_EXTEND_PARSER_H

#include "../gdscript_parser.h"
#include "core/variant.h"
#include "lsp.hpp"

class ExtendGDScriptParser : public GDScriptParser {
	String path;
	String code;
	Vector<String> lines;

	lsp::DocumentSymbol class_symbol;
	Vector<lsp::Diagnostic> diagnostics;

	void update_diagnostics();
	void update_symbols();

	void parse_class_symbol(const GDScriptParser::ClassNode *p_class, lsp::DocumentSymbol &r_symbol);
	void parse_function_symbol(const GDScriptParser::FunctionNode *p_func, lsp::DocumentSymbol &r_symbol);

public:
	_FORCE_INLINE_ const String &get_path() const { return path; }
	_FORCE_INLINE_ const String &get_code() const { return code; }
	_FORCE_INLINE_ const Vector<String> &get_lines() const { return lines; }
	_FORCE_INLINE_ const lsp::DocumentSymbol &get_symbols() const { return class_symbol; }
	_FORCE_INLINE_ const Vector<lsp::Diagnostic> &get_diagnostics() const { return diagnostics; }

	String get_text_for_completion(const lsp::Position &p_cursor);

	Error parse(const String &p_code, const String &p_path);
};

#endif

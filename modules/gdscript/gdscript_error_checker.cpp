/*************************************************************************/
/*  gdscript_error_checker.cpp                                           */
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

#include "gdscript_error_checker.h"

#include "gdscript_parser.h"

void GDScriptErrorChecker::_bind_methods() {
	ClassDB::bind_method("has_errors", &GDScriptErrorChecker::has_errors);
	ClassDB::bind_method("get_error_count", &GDScriptErrorChecker::get_error_count);
	ClassDB::bind_method(D_METHOD("set_source", "source_code"), &GDScriptErrorChecker::set_source);
	ClassDB::bind_method(D_METHOD("get_error", "idx"), &GDScriptErrorChecker::get_error);
	ClassDB::bind_method(D_METHOD("get_error_line", "idx"), &GDScriptErrorChecker::get_error_line);
	ClassDB::bind_method(D_METHOD("get_error_column", "idx"), &GDScriptErrorChecker::get_error_column);
}

bool GDScriptErrorChecker::has_errors() const {
	ERR_FAIL_COND_V_MSG(parser == nullptr, false, "No source code provided.");
	return !parser->get_errors().is_empty();
}

int GDScriptErrorChecker::get_error_count() const {
	ERR_FAIL_COND_V_MSG(parser == nullptr, false, "No source code provided.");
	return parser->get_errors().size();
}

String GDScriptErrorChecker::get_error(const int p_idx) const {
	ERR_FAIL_COND_V_MSG(parser == nullptr, String(), "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, parser->get_errors().size(), String());
	return parser->get_errors()[p_idx].message;
}

int GDScriptErrorChecker::get_error_line(const int p_idx) const {
	ERR_FAIL_COND_V_MSG(parser == nullptr, -1, "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, parser->get_errors().size(), -1);
	return parser->get_errors()[p_idx].line;
}

int GDScriptErrorChecker::get_error_column(const int p_idx) const {
	ERR_FAIL_COND_V_MSG(parser == nullptr, -1, "No source code provided.");
	ERR_FAIL_INDEX_V(p_idx, parser->get_errors().size(), -1);
	return parser->get_errors()[p_idx].column;
}

Error GDScriptErrorChecker::set_source(const String &p_source) {
	if (parser != nullptr) {
		memdelete(parser);
		parser = nullptr;
	}
	parser = memnew(GDScriptParser);
	return parser->parse(p_source, "", false);
}

GDScriptErrorChecker::GDScriptErrorChecker() = default;

GDScriptErrorChecker::~GDScriptErrorChecker() {
	if (parser != nullptr) {
		memdelete(parser);
		parser = nullptr;
	}
}

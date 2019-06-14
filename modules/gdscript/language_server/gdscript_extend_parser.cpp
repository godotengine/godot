/*************************************************************************/
/*  gdscript_extend_parser.cpp                                           */
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

#include "gdscript_extend_parser.h"
#include "../gdscript.h"

void ExtendGDScriptParser::update_diagnostics() {

	diagnostics.clear();

	if (has_error()) {
		lsp::Diagnostic diagnostic;
		diagnostic.severity = lsp::DiagnosticSeverity::Error;
		diagnostic.message = get_error();
		diagnostic.source = "gdscript";
		diagnostic.code = -1;
		lsp::Range range;
		lsp::Position pos;
		int line = get_error_line() - 1;
		const String &line_text = get_lines()[line];
		pos.line = line;
		pos.character = line_text.length() - line_text.strip_edges(true, false).length();
		range.start = pos;
		range.end = range.start;
		range.end.character = line_text.strip_edges(false).length();
		diagnostic.range = range;
		diagnostics.push_back(diagnostic);
	}

	const List<GDScriptWarning> &warnings = get_warnings();
	for (const List<GDScriptWarning>::Element *E = warnings.front(); E; E = E->next()) {
		const GDScriptWarning &warning = E->get();
		lsp::Diagnostic diagnostic;
		diagnostic.severity = lsp::DiagnosticSeverity::Warning;
		diagnostic.message = warning.get_message();
		diagnostic.source = "gdscript";
		diagnostic.code = warning.code;
		lsp::Range range;
		lsp::Position pos;
		int line = warning.line - 1;
		const String &line_text = get_lines()[line];
		pos.line = line;
		pos.character = line_text.length() - line_text.strip_edges(true, false).length();
		range.start = pos;
		range.end = pos;
		range.end.character = line_text.strip_edges(false).length();
		diagnostic.range = range;
		diagnostics.push_back(diagnostic);
	}
}

void ExtendGDScriptParser::update_symbols() {
	const GDScriptParser::Node *head = get_parse_tree();
	if (const GDScriptParser::ClassNode *gdclass = dynamic_cast<const GDScriptParser::ClassNode *>(head)) {
		parse_class_symbol(gdclass, class_symbol);
	}
}

void ExtendGDScriptParser::parse_class_symbol(const GDScriptParser::ClassNode *p_class, lsp::DocumentSymbol &r_symbol) {
	r_symbol.children.clear();
	r_symbol.name = p_class->name;
	if (r_symbol.name.empty())
		r_symbol.name = path.get_file();
	r_symbol.kind = lsp::SymbolKind::Class;
	r_symbol.detail = p_class->get_datatype().to_string();
	r_symbol.deprecated = false;
	r_symbol.range.start.line = p_class->line - 1;
	r_symbol.range.start.character = p_class->column;
	r_symbol.range.end.line = p_class->end_line - 1;
	r_symbol.selectionRange.start.line = r_symbol.range.start.line;

	for (int i = 0; i < p_class->variables.size(); ++i) {

		const GDScriptParser::ClassNode::Member &m = p_class->variables[i];

		lsp::DocumentSymbol symbol;
		symbol.name = m.identifier;
		symbol.kind = lsp::SymbolKind::Variable;
		symbol.detail = m.data_type.to_string();
		symbol.deprecated = false;
		const int line = m.line - 1;
		symbol.range.start.line = line;
		symbol.range.start.character = lines[line].length() - lines[line].strip_edges(true, false).length();
		symbol.range.end.line = line;
		symbol.range.end.character = lines[line].length();
		symbol.selectionRange.start.line = symbol.range.start.line;

		r_symbol.children.push_back(symbol);
	}

	for (int i = 0; i < p_class->_signals.size(); ++i) {
		const GDScriptParser::ClassNode::Signal &signal = p_class->_signals[i];

		lsp::DocumentSymbol symbol;
		symbol.name = signal.name;
		symbol.kind = lsp::SymbolKind::Event;
		symbol.deprecated = false;
		const int line = signal.line - 1;
		symbol.range.start.line = line;
		symbol.range.start.character = lines[line].length() - lines[line].strip_edges(true, false).length();
		symbol.range.end.line = symbol.range.start.line;
		symbol.range.end.character = lines[line].length();
		symbol.selectionRange.start.line = symbol.range.start.line;

		r_symbol.children.push_back(symbol);
	}

	for (Map<StringName, GDScriptParser::ClassNode::Constant>::Element *E = p_class->constant_expressions.front(); E; E = E->next()) {
		lsp::DocumentSymbol symbol;
		symbol.name = E->key();
		symbol.kind = lsp::SymbolKind::Constant;
		symbol.deprecated = false;
		const int line = E->get().expression->line - 1;
		symbol.range.start.line = line;
		symbol.range.start.character = E->get().expression->column;
		symbol.range.end.line = symbol.range.start.line;
		symbol.range.end.character = lines[line].length();
		symbol.selectionRange.start.line = symbol.range.start.line;

		r_symbol.children.push_back(symbol);
	}

	for (int i = 0; i < p_class->functions.size(); ++i) {
		const GDScriptParser::FunctionNode *func = p_class->functions[i];
		lsp::DocumentSymbol symbol;
		parse_function_symbol(func, symbol);
		r_symbol.children.push_back(symbol);
	}

	for (int i = 0; i < p_class->static_functions.size(); ++i) {
		const GDScriptParser::FunctionNode *func = p_class->static_functions[i];
		lsp::DocumentSymbol symbol;
		parse_function_symbol(func, symbol);
		r_symbol.children.push_back(symbol);
	}

	for (int i = 0; i < p_class->subclasses.size(); ++i) {
		const GDScriptParser::ClassNode *subclass = p_class->subclasses[i];
		lsp::DocumentSymbol symbol;
		parse_class_symbol(subclass, symbol);
		r_symbol.children.push_back(symbol);
	}
}

void ExtendGDScriptParser::parse_function_symbol(const GDScriptParser::FunctionNode *p_func, lsp::DocumentSymbol &r_symbol) {
	r_symbol.name = p_func->name;
	r_symbol.kind = lsp::SymbolKind::Function;
	r_symbol.detail = p_func->get_datatype().to_string();
	r_symbol.deprecated = false;
	const int line = p_func->line - 1;
	r_symbol.range.start.line = line;
	r_symbol.range.start.character = p_func->column;
	r_symbol.range.end.line = MAX(p_func->body->end_line - 2, p_func->body->line);
	r_symbol.range.end.character = lines[r_symbol.range.end.line].length();
	r_symbol.selectionRange.start.line = r_symbol.range.start.line;

	for (const Map<StringName, LocalVarNode *>::Element *E = p_func->body->variables.front(); E; E = E->next()) {
		lsp::DocumentSymbol symbol;
		symbol.name = E->key();
		symbol.kind = lsp::SymbolKind::Variable;
		symbol.range.start.line = E->get()->line - 1;
		symbol.range.start.character = E->get()->column;
		symbol.range.end.line = symbol.range.start.line;
		symbol.range.end.character = lines[symbol.range.end.line].length();
		r_symbol.children.push_back(symbol);
	}
	for (int i = 0; i < p_func->arguments.size(); i++) {
		lsp::DocumentSymbol symbol;
		symbol.kind = lsp::SymbolKind::Variable;
		symbol.name = p_func->arguments[i];
		symbol.range.start.line = p_func->body->line - 1;
		symbol.range.start.character = p_func->body->column;
		symbol.range.end = symbol.range.start;
		r_symbol.children.push_back(symbol);
	}
}

String ExtendGDScriptParser::get_text_for_completion(const lsp::Position &p_cursor) {

	String longthing;
	int len = lines.size();
	for (int i = 0; i < len; i++) {

		if (i == p_cursor.line) {
			longthing += lines[i].substr(0, p_cursor.character);
			longthing += String::chr(0xFFFF); //not unicode, represents the cursor
			longthing += lines[i].substr(p_cursor.character, lines[i].size());
		} else {

			longthing += lines[i];
		}

		if (i != len - 1)
			longthing += "\n";
	}

	return longthing;
}

Error ExtendGDScriptParser::parse(const String &p_code, const String &p_path) {
	path = p_path;
	code = p_code;
	lines = p_code.split("\n");

	Error err = GDScriptParser::parse(p_code, p_path.get_base_dir(), false, p_path, false, NULL, false);
	update_diagnostics();
	update_symbols();

	return err;
}

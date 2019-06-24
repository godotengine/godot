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
#include "core/io/json.h"
#include "gdscript_language_protocol.h"
#include "gdscript_workspace.h"

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
		int line = LINE_NUMBER_TO_INDEX(get_error_line());
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
		int line = LINE_NUMBER_TO_INDEX(warning.line);
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

	members.clear();

	const GDScriptParser::Node *head = get_parse_tree();
	if (const GDScriptParser::ClassNode *gdclass = dynamic_cast<const GDScriptParser::ClassNode *>(head)) {

		parse_class_symbol(gdclass, class_symbol);

		for (int i = 0; i < class_symbol.children.size(); i++) {
			const lsp::DocumentSymbol &symbol = class_symbol.children[i];
			members.set(symbol.name, &symbol);
		}
	}
}

void ExtendGDScriptParser::parse_class_symbol(const GDScriptParser::ClassNode *p_class, lsp::DocumentSymbol &r_symbol) {

	const String uri = get_uri();

	r_symbol.uri = uri;
	r_symbol.script_path = path;
	r_symbol.children.clear();
	r_symbol.name = p_class->name;
	if (r_symbol.name.empty())
		r_symbol.name = path.get_file();
	r_symbol.kind = lsp::SymbolKind::Class;
	r_symbol.deprecated = false;
	r_symbol.range.start.line = LINE_NUMBER_TO_INDEX(p_class->line);
	r_symbol.range.start.character = p_class->column;
	r_symbol.range.end.line = LINE_NUMBER_TO_INDEX(p_class->end_line);
	r_symbol.selectionRange.start.line = r_symbol.range.start.line;
	r_symbol.detail = "class " + r_symbol.name;
	r_symbol.documentation = parse_documentation(LINE_NUMBER_TO_INDEX(p_class->line));

	for (int i = 0; i < p_class->variables.size(); ++i) {

		const GDScriptParser::ClassNode::Member &m = p_class->variables[i];

		lsp::DocumentSymbol symbol;
		symbol.name = m.identifier;
		symbol.kind = lsp::SymbolKind::Variable;
		symbol.deprecated = false;
		const int line = LINE_NUMBER_TO_INDEX(m.line);
		symbol.range.start.line = line;
		symbol.range.start.character = lines[line].length() - lines[line].strip_edges(true, false).length();
		symbol.range.end.line = line;
		symbol.range.end.character = lines[line].length();
		symbol.selectionRange.start.line = symbol.range.start.line;
		symbol.detail = "var " + m.identifier;
		if (m.data_type.kind != GDScriptParser::DataType::UNRESOLVED) {
			symbol.detail += ": " + m.data_type.to_string();
		}
		if (m.default_value.get_type() != Variant::NIL) {
			symbol.detail += " = " + JSON::print(m.default_value);
		}

		symbol.documentation = parse_documentation(line);
		symbol.uri = uri;
		symbol.script_path = path;

		r_symbol.children.push_back(symbol);
	}

	for (int i = 0; i < p_class->_signals.size(); ++i) {
		const GDScriptParser::ClassNode::Signal &signal = p_class->_signals[i];

		lsp::DocumentSymbol symbol;
		symbol.name = signal.name;
		symbol.kind = lsp::SymbolKind::Event;
		symbol.deprecated = false;
		const int line = LINE_NUMBER_TO_INDEX(signal.line);
		symbol.range.start.line = line;
		symbol.range.start.character = lines[line].length() - lines[line].strip_edges(true, false).length();
		symbol.range.end.line = symbol.range.start.line;
		symbol.range.end.character = lines[line].length();
		symbol.selectionRange.start.line = symbol.range.start.line;
		symbol.documentation = parse_documentation(line);
		symbol.uri = uri;
		symbol.script_path = path;
		symbol.detail = "signal " + signal.name + "(";
		for (int j = 0; j < signal.arguments.size(); j++) {
			if (j > 0) {
				symbol.detail += ", ";
			}
			symbol.detail += signal.arguments[j];
		}
		symbol.detail += ")";

		r_symbol.children.push_back(symbol);
	}

	for (Map<StringName, GDScriptParser::ClassNode::Constant>::Element *E = p_class->constant_expressions.front(); E; E = E->next()) {
		lsp::DocumentSymbol symbol;
		const GDScriptParser::ClassNode::Constant &c = E->value();
		const GDScriptParser::ConstantNode *node = dynamic_cast<const GDScriptParser::ConstantNode *>(c.expression);
		symbol.name = E->key();
		symbol.kind = lsp::SymbolKind::Constant;
		symbol.deprecated = false;
		const int line = LINE_NUMBER_TO_INDEX(E->get().expression->line);
		symbol.range.start.line = line;
		symbol.range.start.character = E->get().expression->column;
		symbol.range.end.line = symbol.range.start.line;
		symbol.range.end.character = lines[line].length();
		symbol.selectionRange.start.line = symbol.range.start.line;
		symbol.documentation = parse_documentation(line);
		symbol.uri = uri;
		symbol.script_path = path;

		symbol.detail = "const " + symbol.name;
		if (c.type.kind != GDScriptParser::DataType::UNRESOLVED) {
			symbol.detail += ": " + c.type.to_string();
		}
		symbol.detail += " = " + String(node->value);

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

	const String uri = get_uri();

	r_symbol.name = p_func->name;
	r_symbol.kind = lsp::SymbolKind::Function;
	r_symbol.detail = "func " + p_func->name + "(";
	r_symbol.deprecated = false;
	const int line = LINE_NUMBER_TO_INDEX(p_func->line);
	r_symbol.range.start.line = line;
	r_symbol.range.start.character = p_func->column;
	r_symbol.range.end.line = MAX(p_func->body->end_line - 2, p_func->body->line);
	r_symbol.range.end.character = lines[r_symbol.range.end.line].length();
	r_symbol.selectionRange.start.line = r_symbol.range.start.line;
	r_symbol.documentation = GDScriptWorkspace::marked_documentation(parse_documentation(line));
	r_symbol.uri = uri;
	r_symbol.script_path = path;

	String arguments;
	for (int i = 0; i < p_func->arguments.size(); i++) {
		lsp::DocumentSymbol symbol;
		symbol.kind = lsp::SymbolKind::Variable;
		symbol.name = p_func->arguments[i];
		symbol.range.start.line = LINE_NUMBER_TO_INDEX(p_func->body->line);
		symbol.range.start.character = p_func->body->column;
		symbol.range.end = symbol.range.start;
		symbol.uri = uri;
		symbol.script_path = path;
		r_symbol.children.push_back(symbol);
		if (i > 0) {
			arguments += ", ";
		}
		arguments += String(p_func->arguments[i]);
		if (p_func->argument_types[i].kind != GDScriptParser::DataType::UNRESOLVED) {
			arguments += ": " + p_func->argument_types[i].to_string();
		}
		int default_value_idx = i - (p_func->arguments.size() - p_func->default_values.size());
		if (default_value_idx >= 0) {
			const GDScriptParser::ConstantNode *const_node = dynamic_cast<const GDScriptParser::ConstantNode *>(p_func->default_values[default_value_idx]);
			if (const_node == NULL) {
				const GDScriptParser::OperatorNode *operator_node = dynamic_cast<const GDScriptParser::OperatorNode *>(p_func->default_values[default_value_idx]);
				if (operator_node) {
					const_node = dynamic_cast<const GDScriptParser::ConstantNode *>(operator_node->next);
				}
			}

			if (const_node) {
				String value = JSON::print(const_node->value);
				arguments += " = " + value;
			}
		}
	}
	r_symbol.detail += arguments + ")";
	if (p_func->return_type.kind != GDScriptParser::DataType::UNRESOLVED) {
		r_symbol.detail += " -> " + p_func->return_type.to_string();
	}

	for (const Map<StringName, LocalVarNode *>::Element *E = p_func->body->variables.front(); E; E = E->next()) {
		lsp::DocumentSymbol symbol;
		const GDScriptParser::LocalVarNode *var = E->value();
		symbol.name = E->key();
		symbol.kind = lsp::SymbolKind::Variable;
		symbol.range.start.line = LINE_NUMBER_TO_INDEX(E->get()->line);
		symbol.range.start.character = E->get()->column;
		symbol.range.end.line = symbol.range.start.line;
		symbol.range.end.character = lines[symbol.range.end.line].length();
		symbol.uri = uri;
		symbol.script_path = path;
		symbol.detail = "var " + symbol.name;
		if (var->datatype.kind != GDScriptParser::DataType::UNRESOLVED) {
			symbol.detail += ": " + var->datatype.to_string();
		}
		symbol.documentation = GDScriptWorkspace::marked_documentation(parse_documentation(line));
		r_symbol.children.push_back(symbol);
	}
}

String ExtendGDScriptParser::parse_documentation(int p_line) {
	ERR_FAIL_INDEX_V(p_line, lines.size(), String());

	List<String> doc_lines;

	// inline comment
	String inline_comment = lines[p_line];
	int comment_start = inline_comment.find("#");
	if (comment_start != -1) {
		inline_comment = inline_comment.substr(comment_start, inline_comment.length());
		if (inline_comment.length() > 1) {
			doc_lines.push_back(inline_comment.substr(1, inline_comment.length()));
		}
	}

	// upper line comments
	for (int i = p_line - 1; i >= 0; --i) {
		String line_comment = lines[i].strip_edges(true, false);
		if (line_comment.begins_with("#")) {
			if (line_comment.length() > 1) {
				doc_lines.push_front(line_comment.substr(1, line_comment.length()));
			} else {
				doc_lines.push_front("");
			}
		} else {
			break;
		}
	}

	String doc;
	for (List<String>::Element *E = doc_lines.front(); E; E = E->next()) {
		String content = E->get();
		doc += content + "\n";
	}
	return doc;
}

String ExtendGDScriptParser::get_text_for_completion(const lsp::Position &p_cursor) const {

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

String ExtendGDScriptParser::get_text_for_lookup_symbol(const lsp::Position &p_cursor, const String &p_symbol, bool p_func_requred) const {
	String longthing;
	int len = lines.size();
	for (int i = 0; i < len; i++) {

		if (i == p_cursor.line) {
			String line = lines[i];
			String first_part = line.substr(0, p_cursor.character);
			String last_part = line.substr(p_cursor.character, lines[i].size());
			if (!p_symbol.empty()) {
				String left_cursor_text;
				for (int c = p_cursor.character - 1; c >= 0; c--) {
					left_cursor_text = line.substr(c, p_cursor.character - c);
					if (p_symbol.begins_with(left_cursor_text)) {
						first_part = line.substr(0, c);
						first_part += p_symbol;
						break;
					}
				}
			}

			longthing += first_part;
			longthing += String::chr(0xFFFF); //not unicode, represents the cursor
			if (p_func_requred) {
				longthing += "("; // tell the parser this is a function call
			}
			longthing += last_part;
		} else {

			longthing += lines[i];
		}

		if (i != len - 1)
			longthing += "\n";
	}

	return longthing;
}

String ExtendGDScriptParser::get_identifier_under_position(const lsp::Position &p_position, Vector2i &p_offset) const {

	ERR_FAIL_INDEX_V(p_position.line, lines.size(), "");
	String line = lines[p_position.line];
	ERR_FAIL_INDEX_V(p_position.character, line.size(), "");

	int start_pos = p_position.character;
	for (int c = p_position.character; c >= 0; c--) {
		start_pos = c;
		CharType ch = line[c];
		bool valid_char = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_';
		if (!valid_char) {
			break;
		}
	}

	int end_pos = p_position.character;
	for (int c = p_position.character; c < line.length(); c++) {
		CharType ch = line[c];
		bool valid_char = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_';
		if (!valid_char) {
			break;
		}
		end_pos = c;
	}
	if (start_pos < end_pos) {
		p_offset.x = start_pos - p_position.character;
		p_offset.y = end_pos - p_position.character;
		return line.substr(start_pos + 1, end_pos - start_pos);
	}

	return "";
}

String ExtendGDScriptParser::get_uri() const {
	return GDScriptLanguageProtocol::get_singleton()->get_workspace().get_file_uri(path);
}

const lsp::DocumentSymbol *ExtendGDScriptParser::search_symbol_defined_at_line(int p_line, const lsp::DocumentSymbol &p_parent) const {
	const lsp::DocumentSymbol *ret = NULL;
	if (p_line < p_parent.range.start.line) {
		return ret;
	} else if (p_parent.range.start.line == p_line) {
		return &p_parent;
	} else {
		for (int i = 0; i < p_parent.children.size(); i++) {
			ret = search_symbol_defined_at_line(p_line, p_parent.children[i]);
			if (ret) {
				break;
			}
		}
	}
	return ret;
}

const lsp::DocumentSymbol *ExtendGDScriptParser::get_symbol_defined_at_line(int p_line) const {
	if (p_line <= 0) {
		return &class_symbol;
	}
	return search_symbol_defined_at_line(p_line, class_symbol);
}

const lsp::DocumentSymbol *ExtendGDScriptParser::get_member_symbol(const String &p_name) const {

	const lsp::DocumentSymbol *const *ptr = members.getptr(p_name);
	if (ptr) {
		return *ptr;
	}

	return NULL;
}

const Array &ExtendGDScriptParser::get_member_completions() {

	if (member_completions.empty()) {

		const String *name = members.next(NULL);
		while (name) {

			const lsp::DocumentSymbol *symbol = members.get(*name);
			lsp::CompletionItem item = symbol->make_completion_item(false);
			item.data = JOIN_SYMBOLS(path, *name);
			member_completions.push_back(item.to_json());

			name = members.next(name);
		}
	}

	return member_completions;
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

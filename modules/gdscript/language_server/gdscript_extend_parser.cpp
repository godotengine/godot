/*************************************************************************/
/*  gdscript_extend_parser.cpp                                           */
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

#include "gdscript_extend_parser.h"

#include "../gdscript.h"
#include "../gdscript_analyzer.h"
#include "core/io/json.h"
#include "gdscript_language_protocol.h"
#include "gdscript_workspace.h"

void ExtendGDScriptParser::update_diagnostics() {
	diagnostics.clear();

	const List<ParserError> &errors = get_errors();
	for (const List<ParserError>::Element *E = errors.front(); E != nullptr; E = E->next()) {
		const ParserError &error = E->get();
		lsp::Diagnostic diagnostic;
		diagnostic.severity = lsp::DiagnosticSeverity::Error;
		diagnostic.message = error.message;
		diagnostic.source = "gdscript";
		diagnostic.code = -1;
		lsp::Range range;
		lsp::Position pos;
		const PackedStringArray lines = get_lines();
		int line = CLAMP(LINE_NUMBER_TO_INDEX(error.line), 0, lines.size() - 1);
		const String &line_text = lines[line];
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
		diagnostic.message = "(" + warning.get_name() + "): " + warning.get_message();
		diagnostic.source = "gdscript";
		diagnostic.code = warning.code;
		lsp::Range range;
		lsp::Position pos;
		int line = LINE_NUMBER_TO_INDEX(warning.start_line);
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

	const GDScriptParser::Node *head = get_tree();
	if (const GDScriptParser::ClassNode *gdclass = dynamic_cast<const GDScriptParser::ClassNode *>(head)) {
		parse_class_symbol(gdclass, class_symbol);

		for (int i = 0; i < class_symbol.children.size(); i++) {
			const lsp::DocumentSymbol &symbol = class_symbol.children[i];
			members.set(symbol.name, &symbol);

			// cache level one inner classes
			if (symbol.kind == lsp::SymbolKind::Class) {
				ClassMembers inner_class;
				for (int j = 0; j < symbol.children.size(); j++) {
					const lsp::DocumentSymbol &s = symbol.children[j];
					inner_class.set(s.name, &s);
				}
				inner_classes.set(symbol.name, inner_class);
			}
		}
	}
}

void ExtendGDScriptParser::update_document_links(const String &p_code) {
	document_links.clear();

	GDScriptTokenizer tokenizer;
	FileAccessRef fs = FileAccess::create(FileAccess::ACCESS_RESOURCES);
	tokenizer.set_source_code(p_code);
	while (true) {
		GDScriptTokenizer::Token token = tokenizer.scan();
		if (token.type == GDScriptTokenizer::Token::TK_EOF) {
			break;
		} else if (token.type == GDScriptTokenizer::Token::LITERAL) {
			const Variant &const_val = token.literal;
			if (const_val.get_type() == Variant::STRING) {
				String path = const_val;
				bool exists = fs->file_exists(path);
				if (!exists) {
					path = get_path().get_base_dir() + "/" + path;
					exists = fs->file_exists(path);
				}
				if (exists) {
					String value = const_val;
					lsp::DocumentLink link;
					link.target = GDScriptLanguageProtocol::get_singleton()->get_workspace()->get_file_uri(path);
					link.range.start.line = LINE_NUMBER_TO_INDEX(token.start_line);
					link.range.end.line = LINE_NUMBER_TO_INDEX(token.end_line);
					link.range.start.character = LINE_NUMBER_TO_INDEX(token.start_column);
					link.range.end.character = LINE_NUMBER_TO_INDEX(token.end_column);
					document_links.push_back(link);
				}
			}
		}
	}
}

void ExtendGDScriptParser::parse_class_symbol(const GDScriptParser::ClassNode *p_class, lsp::DocumentSymbol &r_symbol) {
	const String uri = get_uri();

	r_symbol.uri = uri;
	r_symbol.script_path = path;
	r_symbol.children.clear();
	r_symbol.name = p_class->identifier != nullptr ? String(p_class->identifier->name) : String();
	if (r_symbol.name.is_empty()) {
		r_symbol.name = path.get_file();
	}
	r_symbol.kind = lsp::SymbolKind::Class;
	r_symbol.deprecated = false;
	r_symbol.range.start.line = LINE_NUMBER_TO_INDEX(p_class->start_line);
	r_symbol.range.start.character = LINE_NUMBER_TO_INDEX(p_class->start_column);
	r_symbol.range.end.line = LINE_NUMBER_TO_INDEX(p_class->end_line);
	r_symbol.selectionRange.start.line = r_symbol.range.start.line;
	r_symbol.detail = "class " + r_symbol.name;
	bool is_root_class = &r_symbol == &class_symbol;
	r_symbol.documentation = parse_documentation(is_root_class ? 0 : LINE_NUMBER_TO_INDEX(p_class->start_line), is_root_class);

	for (int i = 0; i < p_class->members.size(); i++) {
		const ClassNode::Member &m = p_class->members[i];

		switch (m.type) {
			case ClassNode::Member::VARIABLE: {
				lsp::DocumentSymbol symbol;
				symbol.name = m.variable->identifier->name;
				symbol.kind = lsp::SymbolKind::Variable;
				symbol.deprecated = false;
				symbol.range.start.line = LINE_NUMBER_TO_INDEX(m.variable->start_line);
				symbol.range.start.character = LINE_NUMBER_TO_INDEX(m.variable->start_column);
				symbol.range.end.line = LINE_NUMBER_TO_INDEX(m.variable->end_line);
				symbol.range.end.character = LINE_NUMBER_TO_INDEX(m.variable->end_column);
				symbol.selectionRange.start.line = symbol.range.start.line;
				if (m.variable->exported) {
					symbol.detail += "@export ";
				}
				symbol.detail += "var " + m.variable->identifier->name;
				if (m.get_datatype().is_hard_type()) {
					symbol.detail += ": " + m.get_datatype().to_string();
				}
				if (m.variable->initializer != nullptr && m.variable->initializer->is_constant) {
					symbol.detail += " = " + JSON::print(m.variable->initializer->reduced_value);
				}

				symbol.documentation = parse_documentation(LINE_NUMBER_TO_INDEX(m.variable->start_line));
				symbol.uri = uri;
				symbol.script_path = path;

				r_symbol.children.push_back(symbol);
			} break;
			case ClassNode::Member::CONSTANT: {
				lsp::DocumentSymbol symbol;

				symbol.name = m.constant->identifier->name;
				symbol.kind = lsp::SymbolKind::Constant;
				symbol.deprecated = false;
				symbol.range.start.line = LINE_NUMBER_TO_INDEX(m.constant->start_line);
				symbol.range.start.character = LINE_NUMBER_TO_INDEX(m.constant->start_column);
				symbol.range.end.line = LINE_NUMBER_TO_INDEX(m.constant->end_line);
				symbol.range.end.character = LINE_NUMBER_TO_INDEX(m.constant->start_column);
				symbol.selectionRange.start.line = LINE_NUMBER_TO_INDEX(m.constant->start_line);
				symbol.documentation = parse_documentation(LINE_NUMBER_TO_INDEX(m.constant->start_line));
				symbol.uri = uri;
				symbol.script_path = path;

				symbol.detail = "const " + symbol.name;
				if (m.constant->get_datatype().is_hard_type()) {
					symbol.detail += ": " + m.constant->get_datatype().to_string();
				}

				const Variant &default_value = m.constant->initializer->reduced_value;
				String value_text;
				if (default_value.get_type() == Variant::OBJECT) {
					RES res = default_value;
					if (res.is_valid() && !res->get_path().is_empty()) {
						value_text = "preload(\"" + res->get_path() + "\")";
						if (symbol.documentation.is_empty()) {
							if (Map<String, ExtendGDScriptParser *>::Element *S = GDScriptLanguageProtocol::get_singleton()->get_workspace()->scripts.find(res->get_path())) {
								symbol.documentation = S->get()->class_symbol.documentation;
							}
						}
					} else {
						value_text = JSON::print(default_value);
					}
				} else {
					value_text = JSON::print(default_value);
				}
				if (!value_text.is_empty()) {
					symbol.detail += " = " + value_text;
				}

				r_symbol.children.push_back(symbol);
			} break;
			case ClassNode::Member::ENUM_VALUE: {
				lsp::DocumentSymbol symbol;

				symbol.name = m.enum_value.identifier->name;
				symbol.kind = lsp::SymbolKind::EnumMember;
				symbol.deprecated = false;
				symbol.range.start.line = LINE_NUMBER_TO_INDEX(m.enum_value.line);
				symbol.range.start.character = LINE_NUMBER_TO_INDEX(m.enum_value.leftmost_column);
				symbol.range.end.line = LINE_NUMBER_TO_INDEX(m.enum_value.line);
				symbol.range.end.character = LINE_NUMBER_TO_INDEX(m.enum_value.rightmost_column);
				symbol.selectionRange.start.line = LINE_NUMBER_TO_INDEX(m.enum_value.line);
				symbol.documentation = parse_documentation(LINE_NUMBER_TO_INDEX(m.enum_value.line));
				symbol.uri = uri;
				symbol.script_path = path;

				symbol.detail = symbol.name + " = " + itos(m.enum_value.value);

				r_symbol.children.push_back(symbol);
			} break;
			case ClassNode::Member::SIGNAL: {
				lsp::DocumentSymbol symbol;
				symbol.name = m.signal->identifier->name;
				symbol.kind = lsp::SymbolKind::Event;
				symbol.deprecated = false;
				symbol.range.start.line = LINE_NUMBER_TO_INDEX(m.signal->start_line);
				symbol.range.start.character = LINE_NUMBER_TO_INDEX(m.signal->start_column);
				symbol.range.end.line = LINE_NUMBER_TO_INDEX(m.signal->end_line);
				symbol.range.end.character = LINE_NUMBER_TO_INDEX(m.signal->end_column);
				symbol.selectionRange.start.line = symbol.range.start.line;
				symbol.documentation = parse_documentation(LINE_NUMBER_TO_INDEX(m.signal->start_line));
				symbol.uri = uri;
				symbol.script_path = path;
				symbol.detail = "signal " + String(m.signal->identifier->name) + "(";
				for (int j = 0; j < m.signal->parameters.size(); j++) {
					if (j > 0) {
						symbol.detail += ", ";
					}
					symbol.detail += m.signal->parameters[i]->identifier->name;
				}
				symbol.detail += ")";

				r_symbol.children.push_back(symbol);
			} break;
			case ClassNode::Member::ENUM: {
				lsp::DocumentSymbol symbol;
				symbol.kind = lsp::SymbolKind::Enum;
				symbol.range.start.line = LINE_NUMBER_TO_INDEX(m.m_enum->start_line);
				symbol.range.start.character = LINE_NUMBER_TO_INDEX(m.m_enum->start_column);
				symbol.range.end.line = LINE_NUMBER_TO_INDEX(m.m_enum->end_line);
				symbol.range.end.character = LINE_NUMBER_TO_INDEX(m.m_enum->end_column);
				symbol.selectionRange.start.line = symbol.range.start.line;
				symbol.documentation = parse_documentation(LINE_NUMBER_TO_INDEX(m.m_enum->start_line));
				symbol.uri = uri;
				symbol.script_path = path;

				symbol.detail = "enum " + String(m.m_enum->identifier->name) + "{";
				for (int j = 0; j < m.m_enum->values.size(); j++) {
					if (j > 0) {
						symbol.detail += ", ";
					}
					symbol.detail += String(m.m_enum->values[j].identifier->name) + " = " + itos(m.m_enum->values[j].value);
				}
				symbol.detail += "}";
				r_symbol.children.push_back(symbol);
			} break;
			case ClassNode::Member::FUNCTION: {
				lsp::DocumentSymbol symbol;
				parse_function_symbol(m.function, symbol);
				r_symbol.children.push_back(symbol);
			} break;
			case ClassNode::Member::CLASS: {
				lsp::DocumentSymbol symbol;
				parse_class_symbol(m.m_class, symbol);
				r_symbol.children.push_back(symbol);
			} break;
			case ClassNode::Member::UNDEFINED:
				break; // Unreachable.
		}
	}
}

void ExtendGDScriptParser::parse_function_symbol(const GDScriptParser::FunctionNode *p_func, lsp::DocumentSymbol &r_symbol) {
	const String uri = get_uri();

	r_symbol.name = p_func->identifier->name;
	r_symbol.kind = lsp::SymbolKind::Function;
	r_symbol.detail = "func " + String(p_func->identifier->name) + "(";
	r_symbol.deprecated = false;
	r_symbol.range.start.line = LINE_NUMBER_TO_INDEX(p_func->start_line);
	r_symbol.range.start.character = LINE_NUMBER_TO_INDEX(p_func->start_column);
	r_symbol.range.end.line = LINE_NUMBER_TO_INDEX(p_func->start_line);
	r_symbol.range.end.character = LINE_NUMBER_TO_INDEX(p_func->end_column);
	r_symbol.selectionRange.start.line = r_symbol.range.start.line;
	r_symbol.documentation = parse_documentation(LINE_NUMBER_TO_INDEX(p_func->start_line));
	r_symbol.uri = uri;
	r_symbol.script_path = path;

	String parameters;
	for (int i = 0; i < p_func->parameters.size(); i++) {
		const ParameterNode *parameter = p_func->parameters[i];
		lsp::DocumentSymbol symbol;
		symbol.kind = lsp::SymbolKind::Variable;
		symbol.name = parameter->identifier->name;
		symbol.range.start.line = LINE_NUMBER_TO_INDEX(parameter->start_line);
		symbol.range.start.character = LINE_NUMBER_TO_INDEX(parameter->start_line);
		symbol.range.end.line = LINE_NUMBER_TO_INDEX(parameter->end_line);
		symbol.range.end.character = LINE_NUMBER_TO_INDEX(parameter->end_column);
		symbol.uri = uri;
		symbol.script_path = path;
		r_symbol.children.push_back(symbol);
		if (i > 0) {
			parameters += ", ";
		}
		parameters += String(parameter->identifier->name);
		if (parameter->get_datatype().is_hard_type()) {
			parameters += ": " + parameter->get_datatype().to_string();
		}
		if (parameter->default_value != nullptr) {
			String value = JSON::print(parameter->default_value->reduced_value);
			parameters += " = " + value;
		}
	}
	r_symbol.detail += parameters + ")";
	if (p_func->get_datatype().is_hard_type()) {
		r_symbol.detail += " -> " + p_func->get_datatype().to_string();
	}

	List<GDScriptParser::SuiteNode *> function_nodes;

	List<GDScriptParser::Node *> node_stack;
	node_stack.push_back(p_func->body);

	while (!node_stack.is_empty()) {
		GDScriptParser::Node *node = node_stack[0];
		node_stack.pop_front();

		switch (node->type) {
			case GDScriptParser::TypeNode::IF: {
				GDScriptParser::IfNode *if_node = (GDScriptParser::IfNode *)node;
				node_stack.push_back(if_node->true_block);
				if (if_node->false_block) {
					node_stack.push_back(if_node->false_block);
				}
			} break;

			case GDScriptParser::TypeNode::FOR: {
				GDScriptParser::ForNode *for_node = (GDScriptParser::ForNode *)node;
				node_stack.push_back(for_node->loop);
			} break;

			case GDScriptParser::TypeNode::WHILE: {
				GDScriptParser::WhileNode *while_node = (GDScriptParser::WhileNode *)node;
				node_stack.push_back(while_node->loop);
			} break;

			case GDScriptParser::TypeNode::MATCH_BRANCH: {
				GDScriptParser::MatchBranchNode *match_node = (GDScriptParser::MatchBranchNode *)node;
				node_stack.push_back(match_node->block);
			} break;

			case GDScriptParser::TypeNode::SUITE: {
				GDScriptParser::SuiteNode *suite_node = (GDScriptParser::SuiteNode *)node;
				function_nodes.push_back(suite_node);
				for (int i = 0; i < suite_node->statements.size(); ++i) {
					node_stack.push_back(suite_node->statements[i]);
				}
			} break;

			default:
				continue;
		}
	}

	for (List<GDScriptParser::SuiteNode *>::Element *N = function_nodes.front(); N; N = N->next()) {
		const GDScriptParser::SuiteNode *suite_node = N->get();
		for (int i = 0; i < suite_node->locals.size(); i++) {
			const SuiteNode::Local &local = suite_node->locals[i];
			lsp::DocumentSymbol symbol;
			symbol.name = local.name;
			symbol.kind = local.type == SuiteNode::Local::CONSTANT ? lsp::SymbolKind::Constant : lsp::SymbolKind::Variable;
			symbol.range.start.line = LINE_NUMBER_TO_INDEX(local.start_line);
			symbol.range.start.character = LINE_NUMBER_TO_INDEX(local.start_column);
			symbol.range.end.line = LINE_NUMBER_TO_INDEX(local.end_line);
			symbol.range.end.character = LINE_NUMBER_TO_INDEX(local.end_column);
			symbol.uri = uri;
			symbol.script_path = path;
			symbol.detail = local.type == SuiteNode::Local::CONSTANT ? "const " : "var ";
			symbol.detail += symbol.name;
			if (local.get_datatype().is_hard_type()) {
				symbol.detail += ": " + local.get_datatype().to_string();
			}
			symbol.documentation = parse_documentation(LINE_NUMBER_TO_INDEX(local.start_line));
			r_symbol.children.push_back(symbol);
		}
	}
}

String ExtendGDScriptParser::parse_documentation(int p_line, bool p_docs_down) {
	ERR_FAIL_INDEX_V(p_line, lines.size(), String());

	List<String> doc_lines;

	if (!p_docs_down) { // inline comment
		String inline_comment = lines[p_line];
		int comment_start = inline_comment.find("#");
		if (comment_start != -1) {
			inline_comment = inline_comment.substr(comment_start, inline_comment.length()).strip_edges();
			if (inline_comment.length() > 1) {
				doc_lines.push_back(inline_comment.substr(1, inline_comment.length()));
			}
		}
	}

	int step = p_docs_down ? 1 : -1;
	int start_line = p_docs_down ? p_line : p_line - 1;
	for (int i = start_line; true; i += step) {
		if (i < 0 || i >= lines.size()) {
			break;
		}

		String line_comment = lines[i].strip_edges(true, false);
		if (line_comment.begins_with("#")) {
			line_comment = line_comment.substr(1, line_comment.length());
			if (p_docs_down) {
				doc_lines.push_back(line_comment);
			} else {
				doc_lines.push_front(line_comment);
			}
		} else {
			break;
		}
	}

	String doc;
	for (List<String>::Element *E = doc_lines.front(); E; E = E->next()) {
		doc += E->get() + "\n";
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

		if (i != len - 1) {
			longthing += "\n";
		}
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
			String last_part = line.substr(p_cursor.character + 1, lines[i].length());
			if (!p_symbol.is_empty()) {
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

		if (i != len - 1) {
			longthing += "\n";
		}
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
		char32_t ch = line[c];
		bool valid_char = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_';
		if (!valid_char) {
			break;
		}
	}

	int end_pos = p_position.character;
	for (int c = p_position.character; c < line.length(); c++) {
		char32_t ch = line[c];
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
	return GDScriptLanguageProtocol::get_singleton()->get_workspace()->get_file_uri(path);
}

const lsp::DocumentSymbol *ExtendGDScriptParser::search_symbol_defined_at_line(int p_line, const lsp::DocumentSymbol &p_parent) const {
	const lsp::DocumentSymbol *ret = nullptr;
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

Error ExtendGDScriptParser::get_left_function_call(const lsp::Position &p_position, lsp::Position &r_func_pos, int &r_arg_index) const {
	ERR_FAIL_INDEX_V(p_position.line, lines.size(), ERR_INVALID_PARAMETER);

	int bracket_stack = 0;
	int index = 0;

	bool found = false;
	for (int l = p_position.line; l >= 0; --l) {
		String line = lines[l];
		int c = line.length() - 1;
		if (l == p_position.line) {
			c = MIN(c, p_position.character - 1);
		}

		while (c >= 0) {
			const char32_t &character = line[c];
			if (character == ')') {
				++bracket_stack;
			} else if (character == '(') {
				--bracket_stack;
				if (bracket_stack < 0) {
					found = true;
				}
			}
			if (bracket_stack <= 0 && character == ',') {
				++index;
			}
			--c;
			if (found) {
				r_func_pos.character = c;
				break;
			}
		}

		if (found) {
			r_func_pos.line = l;
			r_arg_index = index;
			return OK;
		}
	}

	return ERR_METHOD_NOT_FOUND;
}

const lsp::DocumentSymbol *ExtendGDScriptParser::get_symbol_defined_at_line(int p_line) const {
	if (p_line <= 0) {
		return &class_symbol;
	}
	return search_symbol_defined_at_line(p_line, class_symbol);
}

const lsp::DocumentSymbol *ExtendGDScriptParser::get_member_symbol(const String &p_name, const String &p_subclass) const {
	if (p_subclass.is_empty()) {
		const lsp::DocumentSymbol *const *ptr = members.getptr(p_name);
		if (ptr) {
			return *ptr;
		}
	} else {
		if (const ClassMembers *_class = inner_classes.getptr(p_subclass)) {
			const lsp::DocumentSymbol *const *ptr = _class->getptr(p_name);
			if (ptr) {
				return *ptr;
			}
		}
	}

	return nullptr;
}

const List<lsp::DocumentLink> &ExtendGDScriptParser::get_document_links() const {
	return document_links;
}

const Array &ExtendGDScriptParser::get_member_completions() {
	if (member_completions.is_empty()) {
		const String *name = members.next(nullptr);
		while (name) {
			const lsp::DocumentSymbol *symbol = members.get(*name);
			lsp::CompletionItem item = symbol->make_completion_item();
			item.data = JOIN_SYMBOLS(path, *name);
			member_completions.push_back(item.to_json());

			name = members.next(name);
		}

		const String *_class = inner_classes.next(nullptr);
		while (_class) {
			const ClassMembers *inner_class = inner_classes.getptr(*_class);
			const String *member_name = inner_class->next(nullptr);
			while (member_name) {
				const lsp::DocumentSymbol *symbol = inner_class->get(*member_name);
				lsp::CompletionItem item = symbol->make_completion_item();
				item.data = JOIN_SYMBOLS(path, JOIN_SYMBOLS(*_class, *member_name));
				member_completions.push_back(item.to_json());

				member_name = inner_class->next(member_name);
			}

			_class = inner_classes.next(_class);
		}
	}

	return member_completions;
}

Dictionary ExtendGDScriptParser::dump_function_api(const GDScriptParser::FunctionNode *p_func) const {
	Dictionary func;
	ERR_FAIL_NULL_V(p_func, func);
	func["name"] = p_func->identifier->name;
	func["return_type"] = p_func->get_datatype().to_string();
	func["rpc_mode"] = p_func->rpc_mode;
	Array parameters;
	for (int i = 0; i < p_func->parameters.size(); i++) {
		Dictionary arg;
		arg["name"] = p_func->parameters[i]->identifier->name;
		arg["type"] = p_func->parameters[i]->get_datatype().to_string();
		if (p_func->parameters[i]->default_value != nullptr) {
			arg["default_value"] = p_func->parameters[i]->default_value->reduced_value;
		}
		parameters.push_back(arg);
	}
	if (const lsp::DocumentSymbol *symbol = get_symbol_defined_at_line(LINE_NUMBER_TO_INDEX(p_func->start_line))) {
		func["signature"] = symbol->detail;
		func["description"] = symbol->documentation;
	}
	func["arguments"] = parameters;
	return func;
}

Dictionary ExtendGDScriptParser::dump_class_api(const GDScriptParser::ClassNode *p_class) const {
	Dictionary class_api;

	ERR_FAIL_NULL_V(p_class, class_api);

	class_api["name"] = p_class->identifier != nullptr ? String(p_class->identifier->name) : String();
	class_api["path"] = path;
	Array extends_class;
	for (int i = 0; i < p_class->extends.size(); i++) {
		extends_class.append(String(p_class->extends[i]));
	}
	class_api["extends_class"] = extends_class;
	class_api["extends_file"] = String(p_class->extends_path);
	class_api["icon"] = String(p_class->icon_path);

	if (const lsp::DocumentSymbol *symbol = get_symbol_defined_at_line(LINE_NUMBER_TO_INDEX(p_class->start_line))) {
		class_api["signature"] = symbol->detail;
		class_api["description"] = symbol->documentation;
	}

	Array nested_classes;
	Array constants;
	Array members;
	Array signals;
	Array methods;
	Array static_functions;

	for (int i = 0; i < p_class->members.size(); i++) {
		const ClassNode::Member &m = p_class->members[i];
		switch (m.type) {
			case ClassNode::Member::CLASS:
				nested_classes.push_back(dump_class_api(m.m_class));
				break;
			case ClassNode::Member::CONSTANT: {
				Dictionary api;
				api["name"] = m.constant->identifier->name;
				api["value"] = m.constant->initializer->reduced_value;
				api["data_type"] = m.constant->get_datatype().to_string();
				if (const lsp::DocumentSymbol *symbol = get_symbol_defined_at_line(LINE_NUMBER_TO_INDEX(m.constant->start_line))) {
					api["signature"] = symbol->detail;
					api["description"] = symbol->documentation;
				}
				constants.push_back(api);
			} break;
			case ClassNode::Member::ENUM_VALUE: {
				Dictionary api;
				api["name"] = m.enum_value.identifier->name;
				api["value"] = m.enum_value.value;
				api["data_type"] = m.get_datatype().to_string();
				if (const lsp::DocumentSymbol *symbol = get_symbol_defined_at_line(LINE_NUMBER_TO_INDEX(m.enum_value.line))) {
					api["signature"] = symbol->detail;
					api["description"] = symbol->documentation;
				}
				constants.push_back(api);
			} break;
			case ClassNode::Member::ENUM: {
				Dictionary enum_dict;
				for (int j = 0; j < m.m_enum->values.size(); j++) {
					enum_dict[m.m_enum->values[j].identifier->name] = m.m_enum->values[j].value;
				}

				Dictionary api;
				api["name"] = m.m_enum->identifier->name;
				api["value"] = enum_dict;
				api["data_type"] = m.get_datatype().to_string();
				if (const lsp::DocumentSymbol *symbol = get_symbol_defined_at_line(LINE_NUMBER_TO_INDEX(m.m_enum->start_line))) {
					api["signature"] = symbol->detail;
					api["description"] = symbol->documentation;
				}
				constants.push_back(api);
			} break;
			case ClassNode::Member::VARIABLE: {
				Dictionary api;
				api["name"] = m.variable->identifier->name;
				api["data_type"] = m.variable->get_datatype().to_string();
				api["default_value"] = m.variable->initializer != nullptr ? m.variable->initializer->reduced_value : Variant();
				api["setter"] = m.variable->setter ? ("@" + String(m.variable->identifier->name) + "_setter") : (m.variable->setter_pointer != nullptr ? String(m.variable->setter_pointer->name) : String());
				api["getter"] = m.variable->getter ? ("@" + String(m.variable->identifier->name) + "_getter") : (m.variable->getter_pointer != nullptr ? String(m.variable->getter_pointer->name) : String());
				api["export"] = m.variable->exported;
				if (const lsp::DocumentSymbol *symbol = get_symbol_defined_at_line(LINE_NUMBER_TO_INDEX(m.variable->start_line))) {
					api["signature"] = symbol->detail;
					api["description"] = symbol->documentation;
				}
				members.push_back(api);
			} break;
			case ClassNode::Member::SIGNAL: {
				Dictionary api;
				api["name"] = m.signal->identifier->name;
				Array pars;
				for (int j = 0; j < m.signal->parameters.size(); j++) {
					pars.append(String(m.signal->parameters[i]->identifier->name));
				}
				api["arguments"] = pars;
				if (const lsp::DocumentSymbol *symbol = get_symbol_defined_at_line(LINE_NUMBER_TO_INDEX(m.signal->start_line))) {
					api["signature"] = symbol->detail;
					api["description"] = symbol->documentation;
				}
				signals.push_back(api);
			} break;
			case ClassNode::Member::FUNCTION: {
				if (m.function->is_static) {
					static_functions.append(dump_function_api(m.function));
				} else {
					methods.append(dump_function_api(m.function));
				}
			} break;
			case ClassNode::Member::UNDEFINED:
				break; // Unreachable.
		}
	}

	class_api["sub_classes"] = nested_classes;
	class_api["constants"] = constants;
	class_api["members"] = members;
	class_api["signals"] = signals;
	class_api["methods"] = methods;
	class_api["static_functions"] = static_functions;

	return class_api;
}

Dictionary ExtendGDScriptParser::generate_api() const {
	Dictionary api;
	const GDScriptParser::Node *head = get_tree();
	if (const GDScriptParser::ClassNode *gdclass = dynamic_cast<const GDScriptParser::ClassNode *>(head)) {
		api = dump_class_api(gdclass);
	}
	return api;
}

Error ExtendGDScriptParser::parse(const String &p_code, const String &p_path) {
	path = p_path;
	lines = p_code.split("\n");

	Error err = GDScriptParser::parse(p_code, p_path, false);
	if (err == OK) {
		GDScriptAnalyzer analyzer(this);
		err = analyzer.analyze();
	}
	update_diagnostics();
	update_symbols();
	update_document_links(p_code);
	return err;
}

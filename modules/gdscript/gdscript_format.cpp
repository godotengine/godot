/**************************************************************************/
/*  gdscript_format.cpp                                                   */
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

#include "gdscript_format.h"
#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif
#include "core/string/string_builder.h"
#include "gdscript_parser.h"

GDScriptFormat::GDScriptFormat() :
		line_length_maximum(100),
		tab_size(4),
		tab_type(0),
		lines_between_functions(2),
		indent_in_multiline_block(2) {
#ifdef TOOLS_ENABLED
	if (EditorSettings::get_singleton()) {
		line_length_maximum = EditorSettings::get_singleton()->get_setting("text_editor/appearance/guidelines/line_length_guideline_hard_column");
		lines_between_functions = EditorSettings::get_singleton()->get_setting("text_editor/behavior/formatter/lines_between_functions");
		indent_in_multiline_block = EditorSettings::get_singleton()->get_setting("text_editor/behavior/formatter/indent_in_multiline_block");
		tab_size = EditorSettings::get_singleton()->get_setting("text_editor/behavior/indent/size");
		tab_type = EditorSettings::get_singleton()->get_setting("text_editor/behavior/indent/type");
	}
#endif // TOOLS_ENABLED
}

Error GDScriptFormat::format(const String &p_code, String &r_formatted_code) {
	GDScriptParser parser;
	Error err = parser.parse(p_code, "", false);

	if (err != OK) {
		parser_errors.clear();
		for (GDScriptParser::ParserError err : parser.get_errors()) {
			parser_errors.push_back({ vformat("%s", err.message), err.line, err.column });
		}
		return FAILED;
	}

	find_custom_newlines(p_code);

	GDP::ClassNode *root = parser.get_tree();

	StringBuilder code_block;
	if (parser.is_tool()) {
		if (!root->tool_header_comment.is_empty()) {
			for (const String &i : root->tool_header_comment) {
				code_block += "# ";
				code_block += i;
				code_block += "\n";
			}
		}
		code_block += "@tool";
		if (!root->tool_inline_comment.is_empty()) {
			code_block += " # ";
			code_block += root->tool_inline_comment;
		}
		code_block += "\n";
	}

	code_block += parse_class(root, 0);

	String output = code_block.as_string();

	while (output.ends_with("\n\n")) {
		output = output.substr(0, output.length() - 1);
	}

	output = make_disabled_lines_from_headers(output);

	r_formatted_code.clear();
	r_formatted_code += output;

	return OK;
}

void GDScriptFormat::find_custom_newlines(const String &p_code) {
	new_lines.clear();
	const Vector<String> split_code = p_code.replace("\t", "").replace(" ", "").split("\n");
	for (int i = 0; i < split_code.size() - 1; ++i) {
		if (split_code[i].is_empty()) {
			new_lines.insert(i);
		}
	}
}

bool GDScriptFormat::node_has_comments(const GDP::Node *p_node) {
	return p_node != nullptr && (!p_node->header_comment.is_empty() || !p_node->inline_comment.is_empty());
}

bool GDScriptFormat::children_have_comments(const GDP::Node *p_parent) {
	if (p_parent == nullptr) {
		return false;
	}
	switch (p_parent->type) {
		case GDP::Node::Type::CLASS: {
			const GDP::ClassNode *m_class = dynamic_cast<const GDP::ClassNode *>(p_parent);
			for (const GDScriptParser::ClassNode::Member &member : m_class->members) {
				bool children = false;
				switch (member.type) {
					case GDP::ClassNode::Member::Type::ENUM:
						children = node_has_comments(member.m_enum) || children_have_comments(member.m_enum);
						break;
					case GDP::ClassNode::Member::Type::CONSTANT:
						children = node_has_comments(member.constant) || children_have_comments(member.constant);
						break;
					case GDP::ClassNode::Member::Type::SIGNAL:
						children = node_has_comments(member.signal) || children_have_comments(member.signal);
						break;
					case GDP::ClassNode::Member::Type::FUNCTION:
						children = node_has_comments(member.function) || children_have_comments(member.function);
						break;
					case GDP::ClassNode::Member::Type::ENUM_VALUE:
						children = node_has_comments(member.enum_value.parent_enum) || children_have_comments(member.enum_value.parent_enum);
						break;
					case GDP::ClassNode::Member::Type::VARIABLE:
						children = node_has_comments(member.variable) || children_have_comments(member.variable);
						break;
					case GDP::ClassNode::Member::Type::UNDEFINED:
						children = false;
						break;
					case GDP::ClassNode::Member::Type::CLASS:
					case GDScriptParser::ClassNode::Member::GROUP:
						break;
				}
				if (children) {
					return true;
				}
			}
			return false;
		}
		case GDP::Node::Type::ENUM: {
			const GDP::EnumNode *m_enum = dynamic_cast<const GDP::EnumNode *>(p_parent);
			for (const GDScriptParser::EnumNode::Value &value : m_enum->values) {
				if (!value.identifier->inline_comment.is_empty() || !value.identifier->header_comment.is_empty()) {
					return true;
				}
			}
			return false;
		}
		case GDP::Node::Type::CONSTANT: {
			const GDP::ConstantNode *constant = dynamic_cast<const GDP::ConstantNode *>(p_parent);
			return node_has_comments(constant->initializer) || children_have_comments(constant->initializer);
		}

		case GDP::Node::Type::VARIABLE: {
			const GDP::VariableNode *variable = dynamic_cast<const GDP::VariableNode *>(p_parent);
			return node_has_comments(variable->initializer) || children_have_comments(variable->initializer);
		}
		case GDP::Node::Type::ARRAY: {
			const GDP::ArrayNode *m_array = dynamic_cast<const GDP::ArrayNode *>(p_parent);
			for (const GDScriptParser::ExpressionNode *element : m_array->elements) {
				if (node_has_comments(element) || children_have_comments(element)) {
					return true;
				}
			}
			return false;
		}
		case GDP::Node::Type::ASSERT: {
			const GDP::AssertNode *assert = dynamic_cast<const GDP::AssertNode *>(p_parent);
			return node_has_comments(assert->condition) || children_have_comments(assert->condition);
		}
		case GDP::Node::Type::ASSIGNMENT: {
			const GDP::AssignmentNode *assignment = dynamic_cast<const GDP::AssignmentNode *>(p_parent);
			return node_has_comments(assignment->assigned_value) || children_have_comments(assignment->assigned_value);
		}
		case GDP::Node::Type::BINARY_OPERATOR: {
			const GDP::BinaryOpNode *binary_op_node = dynamic_cast<const GDP::BinaryOpNode *>(p_parent);
			return node_has_comments(binary_op_node->left_operand) || node_has_comments(binary_op_node->right_operand) || children_have_comments(binary_op_node->left_operand) || children_have_comments(binary_op_node->right_operand);
		}
		case GDP::Node::Type::CALL: {
			const GDP::CallNode *call_node = dynamic_cast<const GDP::CallNode *>(p_parent);
			for (const GDScriptParser::ExpressionNode *argument : call_node->arguments) {
				if (node_has_comments(argument) || children_have_comments(argument)) {
					return true;
				}
			}
			return false;
		}
		case GDP::Node::Type::DICTIONARY: {
			const GDP::DictionaryNode *dictionary_node = dynamic_cast<const GDP::DictionaryNode *>(p_parent);
			for (const GDScriptParser::DictionaryNode::Pair element : dictionary_node->elements) {
				if (node_has_comments(element.key) || node_has_comments(element.value) || children_have_comments(element.key) || children_have_comments(element.value)) {
					return true;
				}
			}
			return false;
		}
		case GDP::Node::Type::FOR: {
			const GDP::ForNode *for_node = dynamic_cast<const GDP::ForNode *>(p_parent);
			return node_has_comments(for_node->list) || children_have_comments(for_node->list);
		}
		case GDP::Node::Type::FUNCTION: {
			const GDP::FunctionNode *function_node = dynamic_cast<const GDP::FunctionNode *>(p_parent);
			for (const GDScriptParser::ParameterNode *parameter : function_node->parameters) {
				if (node_has_comments(parameter) || children_have_comments(parameter)) {
					return true;
				}
			}
			return false;
		}
		case GDP::Node::Type::IF: {
			const GDP::IfNode *if_node = dynamic_cast<const GDP::IfNode *>(p_parent);
			return node_has_comments(if_node->condition) || children_have_comments(if_node->condition);
		}
		case GDP::Node::Type::LAMBDA: {
			const GDP::LambdaNode *lambda_node = dynamic_cast<const GDP::LambdaNode *>(p_parent);
			return children_have_comments(lambda_node->function);
		}
		case GDP::Node::Type::MATCH: {
			const GDP::MatchNode *match_node = dynamic_cast<const GDP::MatchNode *>(p_parent);
			return node_has_comments(match_node->test) || children_have_comments(match_node->test);
		}
		case GDP::Node::Type::MATCH_BRANCH: {
			const GDP::MatchBranchNode *match_branch_node = dynamic_cast<const GDP::MatchBranchNode *>(p_parent);
			for (const GDScriptParser::PatternNode *pattern : match_branch_node->patterns) {
				if (node_has_comments(pattern) || children_have_comments(pattern)) {
					return true;
				}
			}
			return false;
		}
		case GDP::Node::Type::SUBSCRIPT: {
			const GDP::SubscriptNode *subscript_node = dynamic_cast<const GDP::SubscriptNode *>(p_parent);
			return node_has_comments(subscript_node->index) || children_have_comments(subscript_node->index);
		}
		case GDP::Node::Type::TERNARY_OPERATOR: {
			const GDP::TernaryOpNode *ternary_op_node = dynamic_cast<const GDP::TernaryOpNode *>(p_parent);
			return node_has_comments(ternary_op_node->condition) || node_has_comments(ternary_op_node->false_expr) || children_have_comments(ternary_op_node->condition) || children_have_comments(ternary_op_node->false_expr);
		}
		case GDP::Node::Type::SIGNAL: {
			const GDP::SignalNode *signal_node = dynamic_cast<const GDP::SignalNode *>(p_parent);
			for (const GDScriptParser::ParameterNode *parameter : signal_node->parameters) {
				if (node_has_comments(parameter) || children_have_comments(parameter)) {
					return true;
				}
			}
			return false;
		}
		default:
			break;
	}

	return false;
}

String GDScriptFormat::make_disabled_lines_from_headers(const String &p_input) const {
	Vector<String> split_lines = p_input.split("\n");
	for (int i = 0; i < split_lines.size(); ++i) {
		String dedented = split_lines[i].dedent();
		if (dedented.begins_with("# DIS#")) {
			String line = dedented.substr(6);
			if (line.begins_with(tab_type == 0 ? "\t" : String(" ").repeat(tab_size))) {
				split_lines.write[i] = "#" + line;
			}
		}
	}

	return String("\n").join(split_lines);
}

String GDScriptFormat::print_comment(const GDP::Node *p_node, const bool p_headers, const int p_indent_level) {
	StringBuilder output;
	if (p_headers) {
		for (const String &i : p_node->header_comment) {
			output += "#";
			if (!i.begins_with("#")) {
				output += " ";
			}
			output += i;
			output += indent(p_indent_level);
		}
	} else if (!p_node->inline_comment.is_empty()) {
		output += " # ";
		output += p_node->inline_comment;
	}
	return output;
}

String GDScriptFormat::parse_class_variable(const GDP::ClassNode *p_node, const int p_indent_level, const int p_member_index) {
	StringBuilder output;
	String variable_string;
	bool did_break = false;

	if (children_have_comments(p_node->members[p_member_index].variable) && is_nestable_statement(p_node->members[p_member_index].variable->initializer)) {
		variable_string = parse_variable(p_node->members[p_member_index].variable, p_indent_level, NONE);
		if (get_length_without_comments(variable_string) > line_length_maximum || variable_string.contains("\n")) {
			variable_string = parse_variable(p_node->members[p_member_index].variable, p_indent_level, WRAP);
			did_break = get_length_without_comments(variable_string) > line_length_maximum || variable_string.contains("\n");
		}
	} else {
		variable_string = parse_variable(p_node->members[p_member_index].variable, p_indent_level, NONE);
		if (get_length_without_comments(variable_string) > line_length_maximum) {
			did_break = true;
			variable_string = parse_variable(p_node->members[p_member_index].variable, p_indent_level, WRAP);
		}
	}

	output += print_comment(p_node->members[p_member_index].variable, true, p_indent_level);

	if (!p_node->members[p_member_index].variable->annotations.is_empty()) {
		String annotation_string;
		if (!did_break) {
			annotation_string = parse_annotations(p_node->members[p_member_index].variable->annotations, p_indent_level);
			if (get_length_without_comments(annotation_string) + get_length_without_comments(variable_string) > line_length_maximum) {
				annotation_string = parse_annotations(p_node->members[p_member_index].variable->annotations, p_indent_level, SPLIT);
			}
		} else {
			annotation_string = parse_annotations(p_node->members[p_member_index].variable->annotations, p_indent_level, SPLIT);
		}
		output += annotation_string;
	}
	output += variable_string;
	output += print_comment(p_node->members[p_member_index].variable, false);

	const String property_string = parse_property(p_node->members[p_member_index].variable, p_indent_level);
	output += property_string;
	if ((!property_string.is_empty() && p_member_index < p_node->members.size() - 1) || new_lines.has(p_node->members[p_member_index].variable->end_line)) {
		output += "\n";
	}

	return output;
}

String GDScriptFormat::parse_class(const GDP::ClassNode *p_node, const int p_indent_level) {
	StringBuilder output;

	bool has_section = false;
	if (!p_node->icon_path.is_empty()) {
		if (!p_node->icon_header_comment.is_empty()) {
			for (const String &i : p_node->icon_header_comment) {
				output += "# ";
				output += i;
				output += indent(p_indent_level);
			}
		}
		output += "@icon(\"";
		output += p_node->icon_path;
		output += "\")";
		if (!p_node->icon_inline_comment.is_empty()) {
			output += " # ";
			output += p_node->icon_inline_comment;
		}
		output += "\n";
		has_section = true;
	}
	if (p_node->identifier != nullptr) {
		if (p_node->outer != nullptr) {
			const String pre_comment = print_comment(p_node, true, p_indent_level);
			const String comment = print_comment(p_node, false);
			output += pre_comment;
			output += "class ";
			output += p_node->identifier->name;
			output += ":";
			output += comment + "\n";
		} else {
			output += print_comment(p_node, true, p_indent_level);
			output += "class_name " + p_node->identifier->name;
			output += print_comment(p_node, false);
			output += "\n";
			has_section = true;
		}
	}
	if (p_node->extends_used) {
		if (p_node->outer != nullptr) {
			output += indent(p_indent_level + 1, false);
		}
		if (!p_node->extends_header_comment.is_empty()) {
			for (const String &i : p_node->extends_header_comment) {
				output += "# ";
				output += i;
				output += indent(p_indent_level);
			}
		}
		output += "extends ";
		if (!p_node->extends_path.is_empty()) {
			output += "\"";
			output += p_node->extends_path;
			output += "\"";
		}
		if (!p_node->extends.is_empty()) {
			if (!p_node->extends_path.is_empty()) {
				output += ".";
			}
			for (int i = 0; i < p_node->extends.size(); ++i) {
				output += p_node->extends[i]->name;
				if (i < p_node->extends.size() - 1) {
					output += ".";
				}
			}
		}

		if (!p_node->extends_inline_comment.is_empty()) {
			output += " # ";
			output += p_node->extends_inline_comment;
		}

		output += "\n";
		has_section = true;
	}

	if (!p_node->members.is_empty()) {
		Vector<GDP::EnumNode *> completed_values;

		if (has_section) {
			while (!output.as_string().ends_with("\n\n\n")) {
				output += "\n";
			}
		}
		int indent_mod = p_node->outer != nullptr ? 1 : 0;
		for (int i = 0; i < p_node->members.size(); ++i) {
			// Section newlines
			switch (p_node->members[i].type) {
				case GDP::ClassNode::Member::Type::FUNCTION: {
					if (i > 0 && p_node->members[i - 1].type != GDP::ClassNode::Member::Type::FUNCTION) {
						while (!output.as_string().ends_with("\n\n\n")) {
							output += "\n";
						}
					}
				} break;
				default: {
					// do nothing
				}
			}

			bool skip_newline = false;
			if (p_node->outer != nullptr) {
				output += indent(p_indent_level + 1, false);
			}

			switch (p_node->members[i].type) {
				case GDP::ClassNode::Member::Type::VARIABLE:
					output += parse_class_variable(p_node, p_indent_level + indent_mod, i);
					break;
				case GDP::ClassNode::Member::Type::CONSTANT: {
					String constant_string;
					if (children_have_comments(p_node->members[i].constant) && is_nestable_statement(p_node->members[i].constant->initializer)) {
						constant_string = parse_constant(p_node->members[i].constant, p_indent_level + indent_mod, WRAP);
					} else {
						constant_string = parse_constant(p_node->members[i].constant, p_indent_level + indent_mod);
						if (get_length_without_comments(constant_string) > line_length_maximum) {
							constant_string = parse_constant(p_node->members[i].constant, p_indent_level + indent_mod, WRAP);
						}
					}
					output += constant_string;
					if ((i < p_node->members.size() - 1 && p_node->members[i + 1].type != GDP::ClassNode::Member::Type::CONSTANT && p_node->members[i + 1].type != GDP::ClassNode::Member::Type::ENUM && p_node->members[i + 1].type != GDP::ClassNode::Member::Type::ENUM_VALUE) || new_lines.has(p_node->members[i].constant->end_line)) {
						output += "\n";
					}
				} break;
				case GDP::ClassNode::Member::Type::FUNCTION:
					if (!p_node->members[i].function->annotations.is_empty()) {
						output += parse_annotations(p_node->members[i].function->annotations, p_indent_level + indent_mod, SPLIT);
					}
					output += parse_function(p_node->members[i].function, p_indent_level + indent_mod);
					if (i < p_node->members.size() - 1) {
						while (!output.as_string().ends_with(String("\n").repeat(lines_between_functions))) {
							output += "\n";
						}
					}
					break;
				case GDP::ClassNode::Member::Type::SIGNAL: {
					String signal_string;
					if (children_have_comments(p_node->members[i].signal)) {
						signal_string = parse_signal(p_node->members[i].signal, p_indent_level + indent_mod, WRAP);
					} else {
						signal_string = parse_signal(p_node->members[i].signal, p_indent_level + indent_mod);
						if (get_length_without_comments(signal_string) > line_length_maximum) {
							signal_string = parse_signal(p_node->members[i].signal, p_indent_level + indent_mod, WRAP);
						}
					}
					output += signal_string;
					if ((i < p_node->members.size() - 1 && p_node->members[i + 1].type != GDP::ClassNode::Member::Type::SIGNAL) || new_lines.has(p_node->members[i].signal->end_line)) {
						output += "\n";
					}
				} break;
				case GDP::ClassNode::Member::Type::CLASS:
					output += parse_class(p_node->members[i].m_class, p_indent_level + indent_mod);
					skip_newline = p_node->members[i].m_class->outer != nullptr;
					break;
				case GDP::ClassNode::Member::Type::ENUM: {
					output += parse_enum(p_node->members[i].m_enum, p_indent_level + indent_mod);
					if ((i < p_node->members.size() - 1 && p_node->members[i + 1].type != GDP::ClassNode::Member::Type::CONSTANT && p_node->members[i + 1].type != GDP::ClassNode::Member::Type::ENUM && p_node->members[i + 1].type != GDP::ClassNode::Member::Type::ENUM_VALUE) || new_lines.has(p_node->members[i].m_enum->end_line)) {
						output += "\n";
					}
				} break;
				case GDP::ClassNode::Member::Type::ENUM_VALUE: {
					if (completed_values.find(p_node->members[i].enum_value.parent_enum) == -1) {
						output += parse_enum(p_node->members[i].enum_value.parent_enum, p_indent_level + indent_mod);
						completed_values.push_back(p_node->members[i].enum_value.parent_enum);
						if ((i < p_node->members.size() - 1 && p_node->members[i + 1].type != GDP::ClassNode::Member::Type::CONSTANT && p_node->members[i + 1].type != GDP::ClassNode::Member::Type::ENUM && p_node->members[i + 1].type != GDP::ClassNode::Member::Type::ENUM_VALUE) || new_lines.has(p_node->members[i].enum_value.parent_enum->end_line)) {
							output += "\n";
						}
					} else {
						skip_newline = true;
					}
				} break;
				case GDP::ClassNode::Member::Type::UNDEFINED:
				case GDScriptParser::ClassNode::Member::GROUP:
					break;
			}
			if (!skip_newline) {
				output += "\n";
			}
		}
	} else if (p_node->outer != nullptr && !p_node->extends_used) {
		output += indent(p_indent_level + 1, false) + "pass\n";
	}

	const int indent_mod = p_node->outer == nullptr ? 0 : 1;
	for (int i = 0; i < p_node->footer_comment.size(); ++i) {
		if (i == p_node->footer_comment.size() - 1 && indent_mod == 0 && !output.as_string().ends_with("\n")) {
			output += "\n";
		}
		output += indent(p_indent_level + indent_mod);
		output += "# ";
		output += p_node->footer_comment[i];
		if (i == p_node->footer_comment.size() - 1 && indent_mod == 0) {
			output += "\n";
		}
	}

	return output;
}

String GDScriptFormat::parse_enum_elements(const GDP::EnumNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	for (int i = 0; i < p_node->values.size(); ++i) {
		if (p_break_type != NONE) {
			output += indent(p_indent_level);
		}
		output += print_comment(p_node->values[i].identifier, true, p_indent_level);

		output += p_node->values[i].identifier->name;
		if (p_node->values[i].custom_value != nullptr) {
			output += " = ";
			output += parse_expression(p_node->values[i].custom_value, p_indent_level);
		}

		if (p_break_type != NONE || i < p_node->values.size() - 1) {
			output += ",";
		}
		if (p_break_type == NONE && i < p_node->values.size() - 1) {
			output += " ";
		}
		output += print_comment(p_node->values[i].identifier, false);
	}

	return output;
}

String GDScriptFormat::parse_enum(const GDP::EnumNode *p_node, const int p_indent_level) {
	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);

	output += "enum";
	if (p_node->identifier != nullptr) {
		output += " ";
		output += p_node->identifier->name;
	}
	output += " {";
	bool break_line = false;

	String enum_elements_string;
	if (children_have_comments(p_node)) {
		enum_elements_string = parse_enum_elements(p_node, p_indent_level + 1, WRAP);
		break_line = true;
	} else {
		enum_elements_string = parse_enum_elements(p_node, p_indent_level);
		if (get_length_without_comments(enum_elements_string) > line_length_maximum) {
			enum_elements_string = parse_enum_elements(p_node, p_indent_level + 1, WRAP);
			break_line = true;
		}
	}

	if (!break_line && get_length_without_comments(output) + get_length_without_comments(enum_elements_string) > line_length_maximum) {
		output += indent(p_indent_level + 1);
		break_line = true;
	} else if (!break_line) {
		output += " ";
	}

	output += enum_elements_string;

	if (break_line) {
		output += indent(p_indent_level);
	} else {
		output += " ";
	}

	output += "}";
	output += print_comment(p_node, false);

	return output;
}

String GDScriptFormat::parse_type(const GDP::TypeNode *p_node) {
	StringBuilder output;
	if (p_node->type_chain.size() > 0) {
		for (int i = 0; i < p_node->type_chain.size(); ++i) {
			output += p_node->type_chain[i]->name;
			if (i < p_node->type_chain.size() - 1) {
				output += ".";
			}
		}
		if (p_node->container_type != nullptr) {
			output += "[";
			output += parse_type(p_node->container_type);
			output += "]";
		}
		output += print_comment(p_node->type_chain[p_node->type_chain.size() - 1], false);
	} else {
		output += "void";
	}

	return output;
}

String GDScriptFormat::parse_var_value(const GDP::ExpressionNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	const String comment_headers = print_comment(p_node, true, p_indent_level);

	StringBuilder value;
	if (p_break_type != NONE) {
		bool add_wrap;
		switch (p_node->type) {
			case GDP::Node::Type::ARRAY:
			case GDP::Node::Type::CALL:
			case GDP::Node::Type::PRELOAD:
			case GDP::Node::Type::DICTIONARY:
			case GDP::Node::Type::SUBSCRIPT:
			case GDP::Node::Type::TERNARY_OPERATOR:
			case GDP::Node::Type::BINARY_OPERATOR:
			case GDP::Node::Type::LAMBDA:
				add_wrap = false;
				break;
			default:
				add_wrap = true;
				break;
		}

		if (add_wrap) {
			value += "(";
			value += indent(p_indent_level + indent_in_multiline_block);

			String value_string;
			if (children_have_comments(p_node)) {
				value_string = parse_expression(p_node, p_indent_level + indent_in_multiline_block, p_break_type);
			} else {
				value_string = parse_expression(p_node, p_indent_level + indent_in_multiline_block);
				if (get_length_without_comments(value_string) > line_length_maximum) {
					value_string = parse_expression(p_node, p_indent_level + indent_in_multiline_block, p_break_type);
				}
			}

			value += value_string;
			value += indent(p_indent_level) + ")";
		} else {
			value += parse_expression(p_node, p_indent_level, p_break_type);
		}
	} else {
		if (children_have_comments(p_node)) {
			value += parse_expression(p_node, p_indent_level, p_break_type);
		} else {
			value += parse_expression(p_node, p_indent_level);
		}
	}

	const String comment = print_comment(p_node, false);

	StringBuilder output;
	output += comment_headers;
	output += value;

	// Expressions can themselves write comments (eg. function calls comments)
	if (!output.as_string().ends_with(comment)) {
		output += comment;
	}

	return output;
}

String GDScriptFormat::parse_constant(const GDP::ConstantNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder declaration;
	declaration += "const ";
	declaration += p_node->identifier->name;

	String value;
	if (p_node->initializer != nullptr) {
		if (p_node->infer_datatype) {
			declaration += " := ";
		} else if (p_node->datatype_specifier != nullptr) {
			declaration += ": ";
			declaration += parse_type(p_node->datatype_specifier);
			declaration += " = ";
		} else {
			declaration += " = ";
		}

		value = parse_var_value(p_node->initializer, p_indent_level, p_break_type);
	}

	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);
	output += declaration;
	output += value;
	output += print_comment(p_node, false);
	return output;
}

String GDScriptFormat::parse_property(const GDP::VariableNode *p_node, const int p_indent_level) {
	StringBuilder output;

	switch (p_node->property) {
		case GDP::VariableNode::PropertyStyle::PROP_INLINE:
			output += indent(p_indent_level + 1);
			if (p_node->setter != nullptr) {
				output += print_comment(p_node->setter, true, p_indent_level + 1);
				output += "set(";
				output += p_node->setter->parameters[0]->identifier->name;
				output += "):";
				output += print_comment(p_node->setter, false);
				output += parse_suite(p_node->setter->body, p_indent_level + 2);
				if (p_node->getter != nullptr) {
					output += indent(p_indent_level + 1);
				}
			}
			if (p_node->getter != nullptr) {
				output += print_comment(p_node->getter, true, p_indent_level + 1);
				output += "get:";
				output += print_comment(p_node->getter, false);
				output += parse_suite(p_node->getter->body, p_indent_level + 2);
			}
			break;
		case GDP::VariableNode::PropertyStyle::PROP_SETGET:
			output += " : ";
			if (p_node->setter_pointer != nullptr) {
				output += "set = ";
				output += p_node->setter_pointer->name;
			}
			if (p_node->getter_pointer != nullptr) {
				output += p_node->setter_pointer != nullptr ? ", get = " : "get = ";
				output += p_node->getter_pointer->name;
			}
			break;
		case GDP::VariableNode::PropertyStyle::PROP_NONE:
			break;
	}

	return output;
}

String GDScriptFormat::parse_variable(const GDP::VariableNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder declaration;
	declaration += "var ";
	declaration += p_node->identifier->name;

	String value = "";
	if (p_node->initializer != nullptr) {
		if (p_node->infer_datatype) {
			declaration += " := ";
		} else if (p_node->datatype_specifier != nullptr) {
			declaration += ": ";
			declaration += parse_type(p_node->datatype_specifier);
			declaration += " = ";
		} else {
			declaration += " = ";
		}

		value = parse_var_value(p_node->initializer, p_indent_level, p_break_type);
	} else if (p_node->datatype_specifier != nullptr) {
		declaration += ": ";
		declaration += parse_type(p_node->datatype_specifier);
	}
	const String prop_append = p_node->property == GDP::VariableNode::PROP_INLINE ? ":" : "";

	StringBuilder output;
	output += declaration;
	output += value;
	output += prop_append;

	return output;
}

String GDScriptFormat::parse_variable_with_comments(const GDP::VariableNode *p_node, int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);
	output += parse_variable(p_node, p_indent_level, p_break_type);
	output += print_comment(p_node, false);
	return output;
}

String GDScriptFormat::parse_annotations(const List<GDP::AnnotationNode *> &p_annotations, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	for (const GDScriptParser::AnnotationNode *p_annotation : p_annotations) {
		output += print_comment(p_annotation, true, p_indent_level);
		output += p_annotation->name;

		if (!p_annotation->arguments.is_empty()) {
			output += "(";
			output += parse_call_arguments(p_annotation->arguments, p_indent_level);
			output += ")";
		}

		output += print_comment(p_annotation, false);

		if (p_break_type != NONE) {
			output += "\n";
		} else {
			output += " ";
		}
	}
	return output;
}

String GDScriptFormat::parse_signal(const GDP::SignalNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder signal_string;
	signal_string += "signal ";
	signal_string += parse_expression(p_node->identifier, p_indent_level);

	if (!p_node->parameters.is_empty()) {
		signal_string += "(";
		if (p_break_type != NONE) {
			signal_string += indent(p_indent_level + 1);
		}
		String parameters;
		if (children_have_comments(p_node)) {
			parameters = parse_parameters(p_node->parameters, p_indent_level, p_break_type);
		} else {
			parameters = parse_parameters(p_node->parameters, p_indent_level);
			if (p_break_type != NONE && get_length_without_comments(parameters) > line_length_maximum) {
				parameters = parse_parameters(p_node->parameters, p_indent_level, p_break_type);
			}
		}
		signal_string += parameters;
		if (p_break_type != NONE) {
			signal_string += indent(p_indent_level);
		}
		signal_string += ")";
	} else if (p_node->has_empty_parameter_list) {
		signal_string += "()";
	}

	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);
	output += signal_string;
	output += print_comment(p_node, false);

	return output;
}

String GDScriptFormat::parse_cast(const GDP::CastNode *p_node, const int p_indent_level) {
	StringBuilder output;
	output += parse_expression(p_node->operand, p_indent_level);
	output += " as ";
	output += parse_type(p_node->cast_type);

	return output;
}

String GDScriptFormat::parse_expression(const GDP::ExpressionNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	String output = "";
	switch (p_node->type) {
		case GDP::Node::Type::ASSIGNMENT:
			output = parse_assignment(dynamic_cast<const GDP::AssignmentNode *>(p_node), p_indent_level, p_break_type);
			break;
		case GDP::Node::Type::AWAIT:
			output = parse_await(dynamic_cast<const GDP::AwaitNode *>(p_node), p_indent_level);
			break;
		case GDP::Node::Type::ARRAY:
			output = parse_array(dynamic_cast<const GDP::ArrayNode *>(p_node), p_indent_level, p_break_type);
			break;
		case GDP::Node::Type::BINARY_OPERATOR:
			output = parse_binary_operator(dynamic_cast<const GDP::BinaryOpNode *>(p_node), p_indent_level, p_break_type);
			break;
		case GDP::Node::Type::LITERAL:
			output = parse_literal(dynamic_cast<const GDP::LiteralNode *>(p_node));
			break;
		case GDP::Node::Type::CAST:
			output = parse_cast(dynamic_cast<const GDP::CastNode *>(p_node), p_indent_level);
			break;
		case GDP::Node::Type::IDENTIFIER:
			output = dynamic_cast<const GDP::IdentifierNode *>(p_node)->name;
			break;
		case GDP::Node::Type::CALL:
			output = parse_call(dynamic_cast<const GDP::CallNode *>(p_node), p_indent_level, p_break_type);
			break;
		case GDP::Node::Type::DICTIONARY:
			output = parse_dictionary(dynamic_cast<const GDP::DictionaryNode *>(p_node), p_indent_level, p_break_type);
			break;
		case GDP::Node::Type::GET_NODE:
			output = parse_get_node(dynamic_cast<const GDP::GetNodeNode *>(p_node), p_indent_level);
			break;
		case GDP::Node::Type::PRELOAD:
			output = parse_preload(dynamic_cast<const GDP::PreloadNode *>(p_node), p_indent_level, p_break_type);
			break;
		case GDP::Node::Type::SELF:
			output = "self";
			break;
		case GDP::Node::Type::SUBSCRIPT:
			output = parse_subscript(dynamic_cast<const GDP::SubscriptNode *>(p_node), p_indent_level, p_break_type);
			break;
		case GDP::Node::Type::TERNARY_OPERATOR:
			output = parse_ternary_op(dynamic_cast<const GDP::TernaryOpNode *>(p_node), p_indent_level, p_break_type);
			break;
		case GDP::Node::Type::TYPE_TEST:
			output = parse_type_test(dynamic_cast<const GDP::TypeTestNode *>(p_node), p_indent_level, p_break_type);
			break;
		case GDP::Node::Type::UNARY_OPERATOR:
			output = parse_unary_op(dynamic_cast<const GDP::UnaryOpNode *>(p_node), p_indent_level, p_break_type);
			break;
		case GDP::Node::Type::LAMBDA:
			output = parse_lambda(dynamic_cast<const GDP::LambdaNode *>(p_node), p_indent_level);
			break;
		default:
			break;
	}

	return output;
}

String GDScriptFormat::parse_parameter(const GDP::ParameterNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder param_string;
	param_string += parse_expression(p_node->identifier, p_indent_level, p_break_type);
	if (p_node->datatype_specifier != nullptr) {
		param_string += ": ";
		param_string += parse_type(p_node->datatype_specifier);
	} else if (p_node->infer_datatype) {
		param_string += " :";
	}

	if (p_node->initializer != nullptr) {
		if (p_node->datatype_specifier != nullptr || !p_node->infer_datatype) {
			param_string += " ";
		}
		param_string += "= " + parse_expression(p_node->initializer, p_indent_level, p_break_type);
	}

	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);
	output += param_string;

	return output;
}

String GDScriptFormat::parse_parameters(const Vector<GDP::ParameterNode *> &p_nodes, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	for (int i = 0; i < p_nodes.size(); ++i) {
		const GDP::ParameterNode *parameter_node = p_nodes[i];

		int indent_level_mod = 0;
		if (p_break_type != NONE && i > 0) {
			indent_level_mod = 1;
			output += indent(p_indent_level + 1);
		}
		String parameter;
		if (children_have_comments(parameter_node)) {
			parameter = parse_parameter(parameter_node, p_indent_level + indent_level_mod, p_break_type);
		} else {
			parameter = parse_parameter(parameter_node, p_indent_level + indent_level_mod);
			if (p_break_type != NONE && get_length_without_comments(parameter) > line_length_maximum) {
				parameter = parse_parameter(parameter_node, p_indent_level + indent_level_mod, p_break_type);
			}
		}

		output += parameter;
		if (i < p_nodes.size() - 1) {
			output += ",";
			if (p_break_type == NONE) {
				output += " ";
			}
		}

		output += print_comment(parameter_node, false);
	}

	return output;
}

String GDScriptFormat::parse_function_signature(const GDP::FunctionNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder signature;
	if (p_node->is_static) {
		signature += "static ";
	}

	signature += "func";

	if (p_node->identifier != nullptr) {
		signature += " ";
		signature += parse_expression(p_node->identifier, p_indent_level, p_break_type);
	}
	signature += "(";

	if (p_break_type != NONE) {
		signature += indent(p_indent_level + 1);
	}

	String parameters;
	if (children_have_comments(p_node)) {
		parameters = parse_parameters(p_node->parameters, p_indent_level, p_break_type);
	} else {
		parameters = parse_parameters(p_node->parameters, p_indent_level);
		if (p_break_type != NONE && get_length_without_comments(parameters) > line_length_maximum) {
			parameters = parse_parameters(p_node->parameters, p_indent_level, p_break_type);
		}
	}

	signature += parameters;

	for (int i = 0; i < p_node->params_footer_comments.size(); ++i) {
		signature += indent(p_indent_level + 1);
		signature += "# ";
		signature += p_node->params_footer_comments[i];
		if (i == p_node->params_footer_comments.size() - 1) {
			signature += indent(p_indent_level);
		}
	}
	if (p_break_type != NONE) {
		signature += indent(p_indent_level);
	}
	signature += ")";

	if (p_node->return_type != nullptr) {
		signature += " -> ";
		signature += parse_type(p_node->return_type);
	}
	signature += ":";

	return signature;
}

String GDScriptFormat::parse_lambda(const GDP::LambdaNode *p_node, const int p_indent_level) {
	String lambda = parse_function(p_node->function, p_indent_level);
	if (lambda.ends_with("\n")) {
		lambda = lambda.substr(0, lambda.length() - 1);
	}
	return lambda;
}

String GDScriptFormat::parse_function(const GDP::FunctionNode *p_node, const int p_indent_level) {
	StringBuilder output;
	String signature;
	if (children_have_comments(p_node)) {
		signature = parse_function_signature(p_node, p_indent_level, WRAP);
	} else {
		signature = parse_function_signature(p_node, p_indent_level);
		if (get_length_without_comments(signature) > line_length_maximum) {
			signature = parse_function_signature(p_node, p_indent_level, WRAP);
		}
	}

	output += print_comment(p_node, true, p_indent_level);
	output += signature;
	output += print_comment(p_node, false);

	output += parse_suite(p_node->body, p_indent_level + 1);

	return output;
}

String GDScriptFormat::parse_return(const GDP::ReturnNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);
	output += "return";
	if (p_node->return_value != nullptr) {
		output += " ";
		output += parse_expression(p_node->return_value, p_indent_level, p_break_type);
	}
	output += print_comment(p_node, false);

	return output;
}

String GDScriptFormat::parse_assert(const GDP::AssertNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder comment;
	comment += print_comment(p_node, false);
	if (p_node->message != nullptr) {
		comment += print_comment(p_node->message, false);
	}

	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);
	output += "assert(";

	String condition_string;
	if (children_have_comments(p_node)) {
		condition_string = parse_expression(p_node->condition, p_indent_level, WRAP);
	} else {
		condition_string = parse_expression(p_node->condition, p_indent_level);
		if (p_break_type != NONE && get_length_without_comments(condition_string) > line_length_maximum) {
			condition_string = parse_expression(p_node->condition, p_indent_level, p_break_type);
		}
	}
	output += condition_string;

	if (p_node->message != nullptr) {
		output += ", ";
		output += parse_expression(p_node->message, p_indent_level);
	}
	output += ")";
	output += comment;

	return output;
}

String GDScriptFormat::parse_if(const GDP::IfNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;

	StringBuilder true_string;
	true_string += "if ";
	String true_condition = parse_expression(p_node->condition, p_indent_level);
	if (p_break_type != NONE && get_length_without_comments(true_condition) + 4 > line_length_maximum) {
		true_string += "(";
		true_string += indent(p_indent_level + 1);

		if (get_length_without_comments(true_condition) > line_length_maximum) {
			BreakType break_type = p_break_type;
			if (p_node->condition->type == GDP::Node::Type::BINARY_OPERATOR) {
				break_type = SPLIT;
			}
			true_condition = parse_expression(p_node->condition, p_indent_level + 1, break_type);
		}

		true_string += true_condition;
		true_string += indent(p_indent_level);

		true_string += "):";
	} else {
		true_string += true_condition;
		true_string += ":";
	}
	output += true_string;
	output += print_comment(p_node->condition, false);
	output += parse_suite(p_node->true_block, p_indent_level + 1);

	if (p_node->false_block != nullptr) {
		if (p_node->false_block->statements.size() == 1 && p_node->false_block->statements[0]->type == GDP::Node::Type::IF) {
			output += indent(p_indent_level);
			output += print_comment(p_node->false_block->statements[0], true, p_indent_level);
			output += "el";
			output += parse_if(dynamic_cast<const GDP::IfNode *>(p_node->false_block->statements[0]), p_indent_level);
		} else {
			output += indent(p_indent_level) + print_comment(p_node->false_block, true, p_indent_level);
			output += "else:";
			output += print_comment(p_node->false_block, false);
			output += parse_suite(p_node->false_block, p_indent_level + 1);
		}
	}

	return output;
}

String GDScriptFormat::parse_match_pattern(const GDP::PatternNode *p_node, const int p_indent_level) {
	StringBuilder output;
	switch (p_node->pattern_type) {
		case GDP::PatternNode::Type::PT_LITERAL:
			output += parse_literal(p_node->literal);
			break;
		case GDP::PatternNode::Type::PT_WILDCARD:
			output += "_";
			break;
		case GDP::PatternNode::Type::PT_EXPRESSION:
			output += parse_expression(p_node->expression, p_indent_level);
			break;
		case GDP::PatternNode::Type::PT_BIND:
			output += "var ";
			output += parse_expression(p_node->bind, p_indent_level);
			break;
		case GDP::PatternNode::Type::PT_ARRAY: {
			output += "[";
			for (int i = 0; i < p_node->array.size(); ++i) {
				output += parse_match_pattern(p_node->array[i], p_indent_level);
				if (i < p_node->array.size() - 1) {
					output += ", ";
				}
			}
			output += "]";
		} break;
		case GDP::PatternNode::Type::PT_DICTIONARY: {
			output += "{";
			for (int i = 0; i < p_node->dictionary.size(); ++i) {
				output += parse_expression(p_node->dictionary[i].key);
				output += ": ";
				output += parse_match_pattern(p_node->dictionary[i].value_pattern);
				if (i < p_node->dictionary.size() - 1) {
					output += ", ";
				}
			}
			output += "}";
		} break;
		case GDP::PatternNode::Type::PT_REST:
			output += "..";
			break;
	}
	return output;
}

String GDScriptFormat::parse_match_branch(const GDP::MatchBranchNode *p_node, const int p_indent_level) {
	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);
	for (int i = 0; i < p_node->patterns.size(); ++i) {
		output += parse_match_pattern(p_node->patterns[i], p_indent_level);
		if (i < p_node->patterns.size() - 1) {
			output += ", ";
		}
	}

	output += ":";
	output += print_comment(p_node, false);

	output += parse_suite(p_node->block, p_indent_level + 1);

	return output;
}

String GDScriptFormat::parse_match(const GDP::MatchNode *p_node, const int p_indent_level, BreakType p_break_type) {
	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);
	output += "match ";
	output += parse_expression(p_node->test, p_indent_level);
	output += ":" + print_comment(p_node->test, false);
	output += indent(p_indent_level + 1);
	for (int i = 0; i < p_node->branches.size(); ++i) {
		output += parse_match_branch(p_node->branches[i], p_indent_level + 1);
		if (i < p_node->branches.size() - 1) {
			output += indent(p_indent_level + 1);
		}
	}

	return output;
}

String GDScriptFormat::parse_for(const GDP::ForNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);

	output += "for ";
	output += parse_expression(p_node->variable, p_indent_level);
	output += " in ";

	String condition = parse_expression(p_node->list, p_indent_level);
	if (p_break_type != NONE && get_length_without_comments(output) + get_length_without_comments(condition) + 1 > line_length_maximum) {
		condition = parse_expression(p_node->list, p_indent_level, WRAP);
	}
	output += condition;
	output += ":";
	output += print_comment(p_node, false);

	output += parse_suite(p_node->loop, p_indent_level + 1);

	return output;
}

String GDScriptFormat::parse_while(const GDP::WhileNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);

	output += "while ";

	String condition = parse_expression(p_node->condition, p_indent_level);
	if (p_break_type != NONE && get_length_without_comments(condition) + 6 > line_length_maximum) {
		output += "(";
		output += indent(p_indent_level + 1);

		if (get_length_without_comments(condition) > line_length_maximum) {
			condition = parse_expression(p_node->condition, p_indent_level + 1, p_break_type);
		}
		output += condition;
		output += indent(p_indent_level);

		output += "):";
	} else {
		output += condition;
		output += ":";
	}

	output += print_comment(p_node, false);
	output += parse_suite(p_node->loop, p_indent_level + 1);

	return output;
}

String GDScriptFormat::parse_suite(const GDP::SuiteNode *p_node, const int p_indent_level) {
	StringBuilder output;
	for (GDScriptParser::Node *statement : p_node->statements) {
		output += indent(p_indent_level);

		ParserFunc parser = nullptr;

		String raw_statement = "";

		switch (statement->type) {
			case GDP::Node::Type::PASS:
				raw_statement = "pass";
				break;
			case GDP::Node::Type::BREAKPOINT:
				raw_statement = "breakpoint";
				break;
			case GDP::Node::Type::BREAK:
				raw_statement = "break";
				break;
			case GDP::Node::Type::CONTINUE:
				raw_statement = "continue";
				break;
			case GDP::Node::Type::RETURN:
				parser = reinterpret_cast<ParserFunc>(&GDScriptFormat::parse_return);
				break;
			case GDP::Node::Type::VARIABLE:
				parser = reinterpret_cast<ParserFunc>(&GDScriptFormat::parse_variable_with_comments);
				break;
			case GDP::Node::Type::CONSTANT:
				parser = reinterpret_cast<ParserFunc>(&GDScriptFormat::parse_constant);
				break;
			case GDP::Node::Type::ASSERT:
				parser = reinterpret_cast<ParserFunc>(&GDScriptFormat::parse_assert);
				break;
			case GDP::Node::Type::IF:
				parser = reinterpret_cast<ParserFunc>(&GDScriptFormat::parse_if);
				output += print_comment(statement, true, p_indent_level);
				break;
			case GDP::Node::Type::WHILE:
				parser = reinterpret_cast<ParserFunc>(&GDScriptFormat::parse_while);
				break;
			case GDP::Node::Type::FOR:
				parser = reinterpret_cast<ParserFunc>(&GDScriptFormat::parse_for);
				break;
			case GDP::Node::Type::MATCH:
				parser = reinterpret_cast<ParserFunc>(&GDScriptFormat::parse_match);
				break;

			case GDP::Node::Type::IDENTIFIER:
			case GDP::Node::Type::ASSIGNMENT:
			case GDP::Node::Type::AWAIT:
			case GDP::Node::Type::CALL:
			case GDP::Node::Type::BINARY_OPERATOR:
			case GDP::Node::Type::ARRAY:
			case GDP::Node::Type::DICTIONARY:
			case GDP::Node::Type::GET_NODE:
			case GDP::Node::Type::LITERAL:
			case GDP::Node::Type::PRELOAD:
			case GDP::Node::Type::SELF:
			case GDP::Node::Type::SUBSCRIPT:
			case GDP::Node::Type::TERNARY_OPERATOR:
			case GDP::Node::Type::UNARY_OPERATOR:
				parser = reinterpret_cast<ParserFunc>(&GDScriptFormat::parse_expression);
				break;
			default:
				break;
		}

		if (parser == nullptr) {
			if (!raw_statement.is_empty()) {
				output += print_comment(statement, true, p_indent_level);
				output += raw_statement;
				output += print_comment(statement, false);
				if (new_lines.has(statement->end_line)) {
					output += "\n";
				}
			}
			continue;
		}

		String statement_string = (this->*parser)(statement, p_indent_level, NONE);
		if (get_length_without_comments(statement_string) > line_length_maximum) {
			statement_string = (this->*parser)(statement, p_indent_level, WRAP);
		}
		output += statement_string;

		if (new_lines.has(statement->end_line)) {
			output += "\n";
		}
	}

	for (int i = 0; i < p_node->footer_comment.size(); ++i) {
		if (i == 0 && !output.as_string().ends_with("\n")) {
			output += "\n";
		}
		output += indent(p_indent_level);
		output += "# ";
		output += p_node->footer_comment[i];
	}

	return output;
}

String GDScriptFormat::parse_type_test(const GDP::TypeTestNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	output += parse_expression(p_node->operand, p_indent_level, p_break_type);
	output += " is ";
	output += parse_type(p_node->test_type);
	return output;
}

String GDScriptFormat::parse_unary_op(const GDP::UnaryOpNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	switch (p_node->operation) {
		case GDP::UnaryOpNode::OpType::OP_NEGATIVE:
			output += "-";
			break;
		case GDP::UnaryOpNode::OpType::OP_POSITIVE:
			output += "+";
			break;
		case GDP::UnaryOpNode::OpType::OP_LOGIC_NOT:
			output += "not ";
			break;
		case GDP::UnaryOpNode::OpType::OP_COMPLEMENT:
			output += "~";
			break;
	}

	output += parse_expression(p_node->operand, p_indent_level, p_break_type);

	return output;
}

String GDScriptFormat::parse_ternary_op(const GDP::TernaryOpNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	if (p_break_type != NONE) {
		output += "(";
	}

	StringBuilder true_condition;
	true_condition += parse_expression(p_node->true_expr, p_indent_level + 1);
	true_condition += " if ";
	true_condition += parse_expression(p_node->condition, p_indent_level + 1);
	if (p_break_type != NONE) {
		String true_condition_string = true_condition.as_string();
		true_condition = StringBuilder();
		true_condition += indent(p_indent_level + 1) + true_condition_string;
		if (get_length_without_comments(true_condition) > line_length_maximum) {
			true_condition = StringBuilder();
			true_condition += indent(p_indent_level + 1);
			true_condition += parse_expression(p_node->true_expr, p_indent_level + 1, p_break_type);
			true_condition += " if ";
			true_condition += parse_expression(p_node->condition, p_indent_level + 1, p_break_type);
		}
	}
	output += true_condition;
	output += print_comment(p_node->condition, false);

	StringBuilder false_condition;
	false_condition += "else ";
	false_condition += parse_expression(p_node->false_expr, p_indent_level + 1);
	if (p_break_type != NONE) {
		String false_condition_sting = false_condition.as_string();
		false_condition = StringBuilder();
		false_condition += indent(p_indent_level + 1);
		false_condition += false_condition_sting;
		if (get_length_without_comments(false_condition) > line_length_maximum) {
			false_condition = StringBuilder();
			false_condition += indent(p_indent_level + 1);
			false_condition += "else ";
			false_condition += parse_expression(p_node->false_expr, p_indent_level + 1, p_break_type);
		}

	} else {
		String false_condition_sting = false_condition.as_string();
		false_condition = StringBuilder();
		false_condition += " ";
		false_condition += false_condition_sting;
	}
	output += false_condition;
	output += print_comment(p_node->false_expr, false);

	if (p_break_type != NONE) {
		output += indent(p_indent_level) + ")";
	}

	return output;
}

String GDScriptFormat::parse_await(const GDP::AwaitNode *p_node, const int p_indent_level) {
	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level) + print_comment(p_node->to_await, true, p_indent_level);
	output += "await ";

	StringBuilder await_string;
	await_string += parse_expression(p_node->to_await, p_indent_level);
	if (get_length_without_comments(await_string) > line_length_maximum) {
		await_string = StringBuilder();
		await_string += "(";
		await_string += indent(p_indent_level + 1);
		await_string += parse_expression(p_node->to_await, p_indent_level + 1, WRAP);
		await_string += indent(p_indent_level);
		await_string += ")";
	}
	output += await_string;
	output += print_comment(p_node, false);
	output += print_comment(p_node->to_await, false);

	return output;
}

String GDScriptFormat::parse_subscript(const GDP::SubscriptNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	if (p_node->base->type == GDP::Node::Type::CAST || p_node->base->type == GDP::Node::Type::BINARY_OPERATOR) {
		output += "(";
	}
	String base = parse_expression(p_node->base, p_indent_level);
	if (p_break_type != NONE && get_length_without_comments(base) > line_length_maximum) {
		base = parse_expression(p_node->base, p_indent_level, p_break_type);
	}
	output += base;
	if (p_node->base->type == GDP::Node::Type::CAST || p_node->base->type == GDP::Node::Type::BINARY_OPERATOR) {
		output += ")";
	}
	if (p_node->is_attribute) {
		String attribute = parse_expression(p_node->attribute, p_indent_level);
		if (p_break_type != NONE && get_length_without_comments(attribute) + 1 > line_length_maximum) {
			attribute = parse_expression(p_node->attribute, p_indent_level, p_break_type);
		}
		output += ".";
		output += attribute;
	} else {
		output += "[";
		if (p_break_type != NONE) {
			output += indent(p_indent_level + 1);
		}
		String subscript_string = print_comment(p_node->index, true) + parse_expression(p_node->index, p_indent_level);
		if (get_length_without_comments(subscript_string) > line_length_maximum) {
			subscript_string = parse_expression(p_node->index, p_indent_level, p_break_type);
		}
		output += subscript_string;
		output += print_comment(p_node->index, false);

		if (p_break_type != NONE) {
			output += indent(p_indent_level);
		}
		output += "]";
	}

	return output;
}

String GDScriptFormat::parse_get_node(const GDP::GetNodeNode *p_node, const int p_indent_level) {
	StringBuilder output;
	String append;
	String full_path = p_node->full_path;
	bool wrap = false;

	if (full_path.get(0) == '%') {
		output += "%";
		full_path = full_path.substr(1);
	} else {
		output += "$";
	}
	char32_t *full_path_chars = full_path.ptrw();

	if (!is_unicode_identifier_start(full_path_chars[0]) && full_path_chars[0] != '%') {
		wrap = true;
	}
	if (!wrap) {
		for (int i = 1; i < full_path.length(); i++) {
			if (!is_unicode_identifier_continue(full_path_chars[i]) && full_path_chars[i] != '/') {
				wrap = true;
				break;
			}
		}
	}
	if (wrap) {
		output += "\"";
		append = "\"";
	}

	output += full_path;
	output += append;
	return output;
}

String GDScriptFormat::parse_binary_operator_element(const GDP::ExpressionNode *p_node, int p_indent_level, const BreakType p_break_type) {
	String element;
	if (p_break_type == SPLIT) {
		element = parse_expression(p_node, p_indent_level, p_break_type);
	} else {
		element = parse_expression(p_node, p_indent_level);
		if (p_break_type != NONE && get_length_without_comments(element) > line_length_maximum) {
			int indent_mod = 0;
			if (p_break_type == WRAP) {
				indent_mod = 1;
			}
			element = parse_expression(p_node, p_indent_level + indent_mod, p_break_type);
		}
	}

	return element;
}

String GDScriptFormat::parse_binary_operator(const GDP::BinaryOpNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	String operator_string = "";
	switch (p_node->operation) {
		case GDP::BinaryOpNode::OpType::OP_ADDITION:
			operator_string = "+ ";
			break;
		case GDP::BinaryOpNode::OpType::OP_SUBTRACTION:
			operator_string = "- ";
			break;
		case GDP::BinaryOpNode::OpType::OP_MULTIPLICATION:
			operator_string = "* ";
			break;
		case GDP::BinaryOpNode::OpType::OP_DIVISION:
			operator_string = "/ ";
			break;
		case GDP::BinaryOpNode::OpType::OP_MODULO:
			operator_string = "% ";
			break;
		case GDP::BinaryOpNode::OpType::OP_BIT_LEFT_SHIFT:
			operator_string = "<< ";
			break;
		case GDP::BinaryOpNode::OpType::OP_BIT_RIGHT_SHIFT:
			operator_string = ">> ";
			break;
		case GDP::BinaryOpNode::OpType::OP_BIT_AND:
			operator_string = "& ";
			break;
		case GDP::BinaryOpNode::OpType::OP_BIT_OR:
			operator_string = "| ";
			break;
		case GDP::BinaryOpNode::OpType::OP_BIT_XOR:
			operator_string = "^ ";
			break;
		case GDP::BinaryOpNode::OpType::OP_LOGIC_AND:
			operator_string = "and ";
			break;
		case GDP::BinaryOpNode::OpType::OP_LOGIC_OR:
			operator_string = "or ";
			break;
		case GDP::BinaryOpNode::OpType::OP_CONTENT_TEST:
			operator_string = "in ";
			break;
		case GDP::BinaryOpNode::OpType::OP_COMP_EQUAL:
			operator_string = "== ";
			break;
		case GDP::BinaryOpNode::OpType::OP_COMP_NOT_EQUAL:
			operator_string = "!= ";
			break;
		case GDP::BinaryOpNode::OpType::OP_COMP_LESS:
			operator_string = "< ";
			break;
		case GDP::BinaryOpNode::OpType::OP_COMP_LESS_EQUAL:
			operator_string = "<= ";
			break;
		case GDP::BinaryOpNode::OpType::OP_COMP_GREATER:
			operator_string = "> ";
			break;
		case GDP::BinaryOpNode::OpType::OP_COMP_GREATER_EQUAL:
			operator_string = ">= ";
			break;
		case GDScriptParser::BinaryOpNode::OP_POWER:
			operator_string += "** ";
			break;
	}

	StringBuilder output;

	bool break_string = false;
	bool split = false;
	StringBuilder left_operand, right_operand;

	// Check if the binary operands have comments (if we need to split the expression)
	// In the binary tree, only the rightmost node can have a comment to not split
	if (children_have_comments(p_node)) {
		const GDP::BinaryOpNode *right_node = p_node;
		while (right_node) {
			if (node_has_comments(right_node->left_operand) || children_have_comments(right_node->left_operand)) {
				split = true;
				break;
			}
			right_node = dynamic_cast<const GDP::BinaryOpNode *>(right_node->right_operand);
		}
	}

	if (split) {
		break_string = true;

		if (p_node->left_operand->type == GDP::Node::Type::BINARY_OPERATOR) {
			left_operand += parse_binary_operator_element(p_node->left_operand, p_indent_level + (p_indent_level == 0 ? indent_in_multiline_block : 0), FORCE_SPLIT);
		} else {
			left_operand += parse_binary_operator_element(p_node->left_operand, p_indent_level, WRAP);
		}
		String left_operand_string = left_operand.as_string();
		left_operand = StringBuilder();
		left_operand += print_comment(p_node->left_operand, true, p_indent_level);
		left_operand += left_operand_string;
		left_operand += print_comment(p_node->left_operand, false);

		if (p_node->right_operand->type == GDP::Node::Type::BINARY_OPERATOR) {
			right_operand += parse_binary_operator_element(p_node->right_operand, p_indent_level + (p_indent_level == 0 ? indent_in_multiline_block : 0), FORCE_SPLIT);
		} else {
			right_operand += parse_binary_operator_element(p_node->right_operand, p_indent_level, WRAP);
		}
		String right_operand_string = right_operand.as_string();
		right_operand = StringBuilder();
		right_operand += print_comment(p_node->right_operand, true, p_indent_level);
		right_operand += right_operand_string;
		right_operand += print_comment(p_node->right_operand, false);
	} else {
		left_operand += parse_binary_operator_element(p_node->left_operand, p_indent_level);
		right_operand += parse_binary_operator_element(p_node->right_operand, p_indent_level);
		if (p_break_type != NONE && get_length_without_comments(left_operand) + get_length_without_comments(right_operand) + get_length_without_comments(operator_string) + 1 > line_length_maximum) {
			break_string = true;
			left_operand = StringBuilder();
			right_operand = StringBuilder();

			if (p_node->left_operand->type == GDP::Node::Type::BINARY_OPERATOR) {
				left_operand += parse_expression(p_node->left_operand, p_indent_level + (p_indent_level == 0 ? indent_in_multiline_block : 0), FORCE_SPLIT);
			} else {
				left_operand += parse_binary_operator_element(p_node->left_operand, p_indent_level, WRAP);
			}
			if (p_node->right_operand->type == GDP::Node::Type::BINARY_OPERATOR) {
				right_operand += parse_expression(p_node->right_operand, p_indent_level + (p_indent_level == 0 ? indent_in_multiline_block : 0), FORCE_SPLIT);
			} else {
				right_operand += parse_binary_operator_element(p_node->right_operand, p_indent_level, WRAP);
			}
		} else {
			String right_operand_string = right_operand.as_string();
			right_operand = StringBuilder();
			right_operand += right_operand_string;
			right_operand += print_comment(p_node->right_operand, false);

			if (p_break_type == FORCE_SPLIT) {
				break_string = true;
			}
		}
	}

	if (p_break_type == WRAP) {
		output += "(";
		output += indent(p_indent_level + indent_in_multiline_block);
	}

	bool wrap_left = false;
	if (p_node->left_operand->type == GDP::Node::Type::BINARY_OPERATOR) {
		const int left_priority = get_operation_priority(dynamic_cast<GDP::BinaryOpNode *>(p_node->left_operand)->operation);
		const int right_priority = get_operation_priority(p_node->operation);

		wrap_left = left_priority > right_priority;
	}

	if (wrap_left) {
		output += "(";
		output += left_operand;
		output += ")";
	} else {
		output += left_operand;
	}

	if (break_string) {
		output += indent(p_indent_level + (p_break_type == WRAP ? indent_in_multiline_block : 0));
	} else {
		output += " ";
	}

	bool wrap_right = false;
	if (p_node->right_operand->type == GDP::Node::Type::BINARY_OPERATOR) {
		const int left_priority = get_operation_priority(p_node->operation);
		const int right_priority = get_operation_priority(dynamic_cast<GDP::BinaryOpNode *>(p_node->right_operand)->operation);

		wrap_right = p_node->operation == GDP::BinaryOpNode::OpType::OP_DIVISION || right_priority > left_priority;
	}

	if (wrap_right) {
		output += operator_string;
		output += "(";
		output += right_operand;
		output += ")";
	} else {
		output += operator_string;
		output += right_operand;
	}

	if (p_break_type == WRAP) {
		output += indent(p_indent_level) + ")";
	}

	return output;
}

int GDScriptFormat::get_length_without_comments(const String &p_string) {
	Vector<String> lines = p_string.split("\n");
	if (lines.is_empty()) {
		return 0;
	}
	while (!lines.is_empty() && lines[0].dedent().begins_with("#")) {
		lines.remove_at(0);
	}
	int character_count = 0;
	for (const String &line : lines) {
		const int inline_comment_idx = line.rfind("#");
		int quote_string_blocks = 0;
		int quote_single_string_blocks = 0;
		for (int c = inline_comment_idx; c >= 0; --c) {
			if (line[c] == '"') {
				quote_string_blocks += 1;
			}
			if (line[c] == '\'') {
				quote_single_string_blocks += 1;
			}
		}
		if (quote_string_blocks > 0 && quote_string_blocks % 2) {
			continue;
		}
		if (quote_single_string_blocks > 0 && quote_single_string_blocks % 2) {
			continue;
		}
		if (inline_comment_idx > -1) {
			character_count += line.substr(inline_comment_idx).length();
		} else {
			character_count += line.length();
		}
	}
	return character_count;
}

String GDScriptFormat::parse_literal(const GDP::LiteralNode *p_node) {
	StringBuilder value_string;

	if (p_node->is_builtin_constant) {
		switch (p_node->constant_type) {
			case GDScriptTokenizer::Token::Type::CONST_PI:
				value_string += "PI";
				break;
			case GDScriptTokenizer::Token::Type::CONST_TAU:
				value_string += "TAU";
				break;
			case GDScriptTokenizer::Token::Type::CONST_INF:
				value_string += "INF";
				break;
			case GDScriptTokenizer::Token::Type::CONST_NAN:
				value_string += "NAN";
				break;
			default:
				break;
		}
	}

	switch (p_node->value.get_type()) {
		case Variant::Type::STRING:
		case Variant::Type::NODE_PATH:
		case Variant::Type::STRING_NAME:
		case Variant::Type::INT:
		case Variant::Type::FLOAT: {
			value_string += p_node->source;
		} break;

		case Variant::Type::NIL: {
			value_string += "null";
		} break;

		default: {
			value_string += String(p_node->value);
		}
	}

	return value_string;
}

String GDScriptFormat::parse_array_element(const GDP::ExpressionNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	return parse_expression(p_node, p_indent_level, p_break_type);
}

String GDScriptFormat::parse_array_elements(const GDP::ArrayNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;

	for (int i = 0; i < p_node->elements.size(); ++i) {
		if (i > 0 && p_break_type != NONE) {
			output += indent(p_indent_level);
		}

		String element;
		if (children_have_comments(p_node)) {
			element = print_comment(p_node->elements[i], true, p_indent_level) + parse_array_element(p_node->elements[i], p_indent_level, p_break_type);
		} else {
			element = parse_array_element(p_node->elements[i], p_indent_level);
			if (p_break_type != NONE && element.length() > line_length_maximum) {
				element = parse_array_element(p_node->elements[i], p_indent_level, p_break_type);
			}
		}

		output += element;
		if (i < p_node->elements.size() - 1 || p_break_type != NONE) {
			output += ",";
		}
		if (p_break_type == NONE && i < p_node->elements.size() - 1) {
			output += " ";
		}
		output += print_comment(p_node->elements[i], false);
	}

	return output;
}

String GDScriptFormat::parse_array(const GDP::ArrayNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder array_string;
	array_string += "[";
	int indent_mod = 0;
	if (p_break_type != NONE) {
		indent_mod = 1;
		array_string += indent(p_indent_level + 1);
	}

	String elements;
	if (children_have_comments(p_node) || !p_node->footer_comments.is_empty()) {
		if (p_break_type == NONE && !p_node->elements.is_empty()) {
			array_string += indent(p_indent_level + 1);
		}
		elements = parse_array_elements(p_node, p_indent_level + 1, WRAP);
	} else {
		elements = parse_array_elements(p_node, p_indent_level + indent_mod);
		if (p_break_type != NONE && elements.length() > line_length_maximum) {
			elements = parse_array_elements(p_node, p_indent_level + indent_mod, p_break_type);
		}
	}

	array_string += elements;

	for (int i = 0; i < p_node->footer_comments.size(); ++i) {
		array_string += indent(p_indent_level + 1);
		array_string += "# ";
		array_string += p_node->footer_comments[i];
	}
	if (p_break_type != NONE || !p_node->footer_comments.is_empty()) {
		array_string += indent(p_indent_level);
	}
	array_string += "]";

	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);
	output += array_string;
	output += print_comment(p_node, false);

	return output;
}

String GDScriptFormat::parse_dictionary_element(const GDP::ExpressionNode *p_key, const GDP::ExpressionNode *p_value, const bool p_is_lua, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	output += print_comment(p_key, true, p_indent_level);
	output += print_comment(p_value, true, p_indent_level);
	output += parse_expression(p_key, p_indent_level);
	if (p_is_lua) {
		output += " = ";
	} else {
		output += ": ";
	}

	output += parse_expression(p_value, p_indent_level, p_break_type);

	return output;
}

String GDScriptFormat::parse_dictionary_elements(const GDP::DictionaryNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;

	const bool is_lua = p_node->style == GDP::DictionaryNode::Style::LUA_TABLE;
	for (int i = 0; i < p_node->elements.size(); ++i) {
		const GDP::ExpressionNode *key = p_node->elements[i].key;
		const GDP::ExpressionNode *value = p_node->elements[i].value;

		if (p_break_type != NONE) {
			output += indent(p_indent_level);
		}

		String element_string;
		if (children_have_comments(p_node)) {
			element_string = parse_dictionary_element(key, value, is_lua, p_indent_level, p_break_type);
		} else {
			element_string = parse_dictionary_element(key, value, is_lua, p_indent_level);
			if (p_break_type != NONE && element_string.length() > line_length_maximum) {
				element_string = parse_dictionary_element(key, value, is_lua, p_indent_level, p_break_type);
			}
		}

		if (p_break_type != NONE || i < p_node->elements.size() - 1) {
			element_string += ",";
		}
		if (p_break_type == NONE && i < p_node->elements.size() - 1) {
			element_string += " ";
		}

		output += element_string;
		output += print_comment(value, false);
	}

	return output;
}

String GDScriptFormat::parse_dictionary(const GDP::DictionaryNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder dictionary_string;
	dictionary_string += "{";

	int indent_mod = 0;
	if (p_break_type != NONE) {
		indent_mod = 1;
	}

	String elements;
	if (children_have_comments(p_node) || !p_node->footer_comments.is_empty()) {
		elements = parse_dictionary_elements(p_node, p_indent_level + 1, WRAP);
	} else {
		elements = parse_dictionary_elements(p_node, p_indent_level + indent_mod);
		if (p_break_type != NONE && get_length_without_comments(elements) > line_length_maximum) {
			elements = parse_dictionary_elements(p_node, p_indent_level + indent_mod, p_break_type);
		} else if (p_break_type != NONE) {
			dictionary_string += indent(p_indent_level + indent_mod);
		}
	}
	dictionary_string += elements;

	for (int i = 0; i < p_node->footer_comments.size(); ++i) {
		dictionary_string += indent(p_indent_level + 1);
		dictionary_string += "# ";
		dictionary_string += p_node->footer_comments[i];
	}
	if (p_break_type == WRAP || !p_node->footer_comments.is_empty()) {
		dictionary_string += indent(p_indent_level);
	}
	dictionary_string += "}";

	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);
	output += dictionary_string;
	output += print_comment(p_node, false);

	return output;
}

String GDScriptFormat::parse_assignment(const GDP::AssignmentNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);
	output += parse_expression(p_node->assignee, p_indent_level, p_break_type);

	switch (p_node->operation) {
		case GDP::AssignmentNode::Operation::OP_NONE:
			output += " = ";
			break;
		case GDP::AssignmentNode::Operation::OP_ADDITION:
			output += " += ";
			break;
		case GDP::AssignmentNode::Operation::OP_SUBTRACTION:
			output += " -= ";
			break;
		case GDP::AssignmentNode::Operation::OP_MULTIPLICATION:
			output += " *= ";
			break;
		case GDP::AssignmentNode::Operation::OP_DIVISION:
			output += " /= ";
			break;
		case GDP::AssignmentNode::Operation::OP_MODULO:
			output += " %= ";
			break;
		case GDP::AssignmentNode::Operation::OP_BIT_SHIFT_LEFT:
			output += " <<= ";
			break;
		case GDP::AssignmentNode::Operation::OP_BIT_SHIFT_RIGHT:
			output += " >>= ";
			break;
		case GDP::AssignmentNode::Operation::OP_BIT_AND:
			output += " &= ";
			break;
		case GDP::AssignmentNode::Operation::OP_BIT_OR:
			output += " |= ";
			break;
		case GDP::AssignmentNode::Operation::OP_BIT_XOR:
			output += " ^= ";
			break;
		case GDScriptParser::AssignmentNode::OP_POWER:
			output += " **= ";
			break;
	}
	output += parse_expression(p_node->assigned_value, p_indent_level, p_break_type);
	output += print_comment(p_node, false);
	if (p_node->assigned_value->type == GDP::Node::Type::LITERAL || p_node->assigned_value->type == GDP::Node::Type::IDENTIFIER || p_node->assigned_value->type == GDP::Node::Type::SELF) {
		output += print_comment(p_node->assigned_value, false);
	}

	if (p_break_type == NONE && output.as_string().length() > line_length_maximum) {
		output = StringBuilder();
		output += parse_assignment(p_node, p_indent_level, WRAP);
	}

	return output;
}

String GDScriptFormat::indent(const int p_count, const bool p_newline) {
	StringBuilder output;
	if (p_newline) {
		output += "\n";
	}
	if (tab_type == 0) {
		for (int i = 0; i < p_count; ++i) {
			output += "\t";
		}
	} else {
		for (int i = 0; i < p_count * tab_size; ++i) {
			output += " ";
		}
	}
	return output;
}

String GDScriptFormat::parse_preload(const GDP::PreloadNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	output += "preload(";
	const bool path_has_comment = !p_node->path->inline_comment.is_empty();
	if (path_has_comment || p_break_type != NONE) {
		output += indent(p_indent_level + 1);
	}
	output += parse_expression(p_node->path, p_indent_level, p_break_type);
	output += print_comment(p_node->path, false);
	if (path_has_comment || p_break_type != NONE) {
		output += indent(p_indent_level);
	}
	output += ")";

	output += print_comment(p_node, false);

	return output;
}

String GDScriptFormat::parse_call_arguments(const Vector<GDP::ExpressionNode *> &p_nodes, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	for (int i = 0; i < p_nodes.size(); ++i) {
		GDP::ExpressionNode *node = p_nodes[i];

		if (p_break_type != NONE) {
			if (node->type != GDP::Node::ARRAY && node->type != GDP::Node::DICTIONARY) {
				output += indent(p_indent_level);
			}
		}

		int indent_mod = 0;
		if (node->type == GDP::Node::ARRAY || node->type == GDP::Node::DICTIONARY) {
			indent_mod = 1;
		}

		String parameter_string;
		if (children_have_comments(node)) {
			parameter_string = parse_expression(node, p_indent_level - indent_mod, p_break_type);
		} else {
			parameter_string = parse_expression(node, p_indent_level);
			if (p_break_type != NONE && parameter_string.length() > line_length_maximum) {
				parameter_string = parse_expression(node, p_indent_level - indent_mod, p_break_type);
			}
		}
		output += print_comment(node, true, p_indent_level);
		output += parameter_string;

		// TODO: handle trailing comma here:
		if (i < p_nodes.size() - 1) {
			output += ",";
			if (p_break_type == NONE) {
				output += " ";
			}
		}
		output += print_comment(node, false);
	}

	return output;
}

String GDScriptFormat::parse_call(const GDP::CallNode *p_node, const int p_indent_level, const BreakType p_break_type) {
	StringBuilder output;
	output += print_comment(p_node, true, p_indent_level);

	if (p_node->is_super) {
		output += "super";
		if (p_node->get_callee_type() != GDP::Node::Type::NONE) {
			output += ".";
			output += parse_expression(p_node->callee, p_indent_level);
		}
	} else {
		output += parse_expression(p_node->callee, p_indent_level);
	}
	output += "(";

	bool break_string = false;
	int indent_mod = 0;
	if (p_break_type != NONE) {
		indent_mod = indent_in_multiline_block;
	}

	String parameters_string;
	if (children_have_comments(p_node) || !p_node->params_footer_comment.is_empty()) {
		parameters_string = parse_call_arguments(p_node->arguments, p_indent_level + indent_in_multiline_block, WRAP);
		break_string = true;
	} else {
		parameters_string = parse_call_arguments(p_node->arguments, p_indent_level + indent_mod);
		if (p_break_type != NONE && parameters_string.length() > line_length_maximum) {
			break_string = true;
			parameters_string = parse_call_arguments(p_node->arguments, p_indent_level + indent_mod, p_break_type);
		}
	}

	if (!break_string && p_break_type != NONE) {
		output += indent(p_indent_level + indent_in_multiline_block);
	}
	output += parameters_string;

	for (int i = 0; i < p_node->params_footer_comment.size(); ++i) {
		output += indent(p_indent_level + indent_in_multiline_block);
		output += "# ";
		output += p_node->params_footer_comment[i];
		if (i == p_node->params_footer_comment.size() - 1) {
			output += indent(p_indent_level);
		}
	}

	if (p_break_type != NONE || break_string) {
		const String indent_checker = p_indent_level > 0 ? "\t" : "\n";
		if (!output.as_string().ends_with(indent_checker)) {
			bool skip_indent = false;
			if (!p_node->arguments.is_empty()) {
				const GDP::Node *last_argument = p_node->arguments[p_node->arguments.size() - 1];
				if (last_argument->type == GDP::Node::ARRAY || last_argument->type == GDP::Node::DICTIONARY) {
					skip_indent = true;
				}
			}
			if (!skip_indent) {
				output += indent(p_indent_level);
			}
		}
	}
	output += ")";
	output += print_comment(p_node, false);

	return output;
}

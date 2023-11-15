/**************************************************************************/
/*  gdscript_translation_parser_plugin.cpp                                */
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

#include "gdscript_translation_parser_plugin.h"

#include "../gdscript.h"
#include "../gdscript_analyzer.h"

#include "core/io/resource_loader.h"

void GDScriptEditorTranslationParserPlugin::get_recognized_extensions(List<String> *r_extensions) const {
	GDScriptLanguage::get_singleton()->get_recognized_extensions(r_extensions);
}

Error GDScriptEditorTranslationParserPlugin::parse_file(const String &p_path, Vector<String> *r_ids, Vector<Vector<String>> *r_ids_ctx_plural) {
	// Extract all translatable strings using the parsed tree from GDScriptParser.
	// The strategy is to find all ExpressionNode and AssignmentNode from the tree and extract strings if relevant, i.e
	// Search strings in ExpressionNode -> CallNode -> tr(), set_text(), set_placeholder() etc.
	// Search strings in AssignmentNode -> text = "__", tooltip_text = "__" etc.

	Error err;
	Ref<Resource> loaded_res = ResourceLoader::load(p_path, "", ResourceFormatLoader::CACHE_MODE_REUSE, &err);
	if (err) {
		ERR_PRINT("Failed to load " + p_path);
		return err;
	}

	ids = r_ids;
	ids_ctx_plural = r_ids_ctx_plural;
	Ref<GDScript> gdscript = loaded_res;
	String source_code = gdscript->get_source_code();

	GDScriptParser parser;
	err = parser.parse(source_code, p_path, false);
	ERR_FAIL_COND_V_MSG(err != OK, err, "Failed to parse GDScript with GDScriptParser.");

	GDScriptAnalyzer analyzer(&parser);
	err = analyzer.analyze();
	ERR_FAIL_COND_V_MSG(err != OK, err, "Failed to analyze GDScript with GDScriptAnalyzer.");

	// Traverse through the parsed tree from GDScriptParser.
	GDScriptParser::ClassNode *c = parser.get_tree();
	_traverse_class(c);

	return OK;
}

bool GDScriptEditorTranslationParserPlugin::_is_constant_string(const GDScriptParser::ExpressionNode *p_expression) {
	ERR_FAIL_NULL_V(p_expression, false);
	return p_expression->is_constant && (p_expression->reduced_value.get_type() == Variant::STRING || p_expression->reduced_value.get_type() == Variant::STRING_NAME);
}

void GDScriptEditorTranslationParserPlugin::_traverse_class(const GDScriptParser::ClassNode *p_class) {
	for (int i = 0; i < p_class->members.size(); i++) {
		const GDScriptParser::ClassNode::Member &m = p_class->members[i];
		// There are 7 types of Member, but only class, function and variable can contain translatable strings.
		switch (m.type) {
			case GDScriptParser::ClassNode::Member::CLASS:
				_traverse_class(m.m_class);
				break;
			case GDScriptParser::ClassNode::Member::FUNCTION:
				_traverse_function(m.function);
				break;
			case GDScriptParser::ClassNode::Member::VARIABLE:
				_read_variable(m.variable);
				break;
			default:
				break;
		}
	}
}

void GDScriptEditorTranslationParserPlugin::_traverse_function(const GDScriptParser::FunctionNode *p_func) {
	_traverse_block(p_func->body);
}

void GDScriptEditorTranslationParserPlugin::_read_variable(const GDScriptParser::VariableNode *p_var) {
	_assess_expression(p_var->initializer);
}

void GDScriptEditorTranslationParserPlugin::_traverse_block(const GDScriptParser::SuiteNode *p_suite) {
	if (!p_suite) {
		return;
	}

	const Vector<GDScriptParser::Node *> &statements = p_suite->statements;
	for (int i = 0; i < statements.size(); i++) {
		const GDScriptParser::Node *statement = statements[i];

		// Statements with Node type constant, break, continue, pass, breakpoint are skipped because they can't contain translatable strings.
		switch (statement->type) {
			case GDScriptParser::Node::VARIABLE:
				_assess_expression(static_cast<const GDScriptParser::VariableNode *>(statement)->initializer);
				break;
			case GDScriptParser::Node::IF: {
				const GDScriptParser::IfNode *if_node = static_cast<const GDScriptParser::IfNode *>(statement);
				_assess_expression(if_node->condition);
				//FIXME : if the elif logic is changed in GDScriptParser, then this probably will have to change as well. See GDScriptParser::TreePrinter::print_if().
				_traverse_block(if_node->true_block);
				_traverse_block(if_node->false_block);
				break;
			}
			case GDScriptParser::Node::FOR: {
				const GDScriptParser::ForNode *for_node = static_cast<const GDScriptParser::ForNode *>(statement);
				_assess_expression(for_node->list);
				_traverse_block(for_node->loop);
				break;
			}
			case GDScriptParser::Node::WHILE: {
				const GDScriptParser::WhileNode *while_node = static_cast<const GDScriptParser::WhileNode *>(statement);
				_assess_expression(while_node->condition);
				_traverse_block(while_node->loop);
				break;
			}
			case GDScriptParser::Node::MATCH: {
				const GDScriptParser::MatchNode *match_node = static_cast<const GDScriptParser::MatchNode *>(statement);
				_assess_expression(match_node->test);
				for (int j = 0; j < match_node->branches.size(); j++) {
					_traverse_block(match_node->branches[j]->block);
				}
				break;
			}
			case GDScriptParser::Node::RETURN:
				_assess_expression(static_cast<const GDScriptParser::ReturnNode *>(statement)->return_value);
				break;
			case GDScriptParser::Node::ASSERT:
				_assess_expression((static_cast<const GDScriptParser::AssertNode *>(statement))->condition);
				break;
			case GDScriptParser::Node::ASSIGNMENT:
				_assess_assignment(static_cast<const GDScriptParser::AssignmentNode *>(statement));
				break;
			default:
				if (statement->is_expression()) {
					_assess_expression(static_cast<const GDScriptParser::ExpressionNode *>(statement));
				}
				break;
		}
	}
}

void GDScriptEditorTranslationParserPlugin::_assess_expression(const GDScriptParser::ExpressionNode *p_expression) {
	// Explore all ExpressionNodes to find CallNodes which contain translation strings, such as tr(), set_text() etc.
	// tr() can be embedded quite deep within multiple ExpressionNodes so need to dig down to search through all ExpressionNodes.
	if (!p_expression) {
		return;
	}

	// ExpressionNode of type await, cast, get_node, identifier, literal, preload, self, subscript, unary are ignored as they can't be CallNode
	// containing translation strings.
	switch (p_expression->type) {
		case GDScriptParser::Node::ARRAY: {
			const GDScriptParser::ArrayNode *array_node = static_cast<const GDScriptParser::ArrayNode *>(p_expression);
			for (int i = 0; i < array_node->elements.size(); i++) {
				_assess_expression(array_node->elements[i]);
			}
			break;
		}
		case GDScriptParser::Node::ASSIGNMENT:
			_assess_assignment(static_cast<const GDScriptParser::AssignmentNode *>(p_expression));
			break;
		case GDScriptParser::Node::BINARY_OPERATOR: {
			const GDScriptParser::BinaryOpNode *binary_op_node = static_cast<const GDScriptParser::BinaryOpNode *>(p_expression);
			_assess_expression(binary_op_node->left_operand);
			_assess_expression(binary_op_node->right_operand);
			break;
		}
		case GDScriptParser::Node::CALL: {
			const GDScriptParser::CallNode *call_node = static_cast<const GDScriptParser::CallNode *>(p_expression);
			_extract_from_call(call_node);
			for (int i = 0; i < call_node->arguments.size(); i++) {
				_assess_expression(call_node->arguments[i]);
			}
		} break;
		case GDScriptParser::Node::DICTIONARY: {
			const GDScriptParser::DictionaryNode *dict_node = static_cast<const GDScriptParser::DictionaryNode *>(p_expression);
			for (int i = 0; i < dict_node->elements.size(); i++) {
				_assess_expression(dict_node->elements[i].key);
				_assess_expression(dict_node->elements[i].value);
			}
			break;
		}
		case GDScriptParser::Node::TERNARY_OPERATOR: {
			const GDScriptParser::TernaryOpNode *ternary_op_node = static_cast<const GDScriptParser::TernaryOpNode *>(p_expression);
			_assess_expression(ternary_op_node->condition);
			_assess_expression(ternary_op_node->true_expr);
			_assess_expression(ternary_op_node->false_expr);
			break;
		}
		default:
			break;
	}
}

void GDScriptEditorTranslationParserPlugin::_assess_assignment(const GDScriptParser::AssignmentNode *p_assignment) {
	// Extract the translatable strings coming from assignments. For example, get_node("Label").text = "____"

	StringName assignee_name;
	if (p_assignment->assignee->type == GDScriptParser::Node::IDENTIFIER) {
		assignee_name = static_cast<const GDScriptParser::IdentifierNode *>(p_assignment->assignee)->name;
	} else if (p_assignment->assignee->type == GDScriptParser::Node::SUBSCRIPT) {
		const GDScriptParser::SubscriptNode *subscript = static_cast<const GDScriptParser::SubscriptNode *>(p_assignment->assignee);
		if (subscript->is_attribute && subscript->attribute) {
			assignee_name = subscript->attribute->name;
		} else if (subscript->index && _is_constant_string(subscript->index)) {
			assignee_name = subscript->index->reduced_value;
		}
	}

	if (assignee_name != StringName() && assignment_patterns.has(assignee_name) && _is_constant_string(p_assignment->assigned_value)) {
		// If the assignment is towards one of the extract patterns (text, tooltip_text etc.), and the value is a constant string, we collect the string.
		ids->push_back(p_assignment->assigned_value->reduced_value);
	} else if (assignee_name == fd_filters && p_assignment->assigned_value->type == GDScriptParser::Node::CALL) {
		// FileDialog.filters accepts assignment in the form of PackedStringArray. For example,
		// get_node("FileDialog").filters = PackedStringArray(["*.png ; PNG Images","*.gd ; GDScript Files"]).

		const GDScriptParser::CallNode *call_node = static_cast<const GDScriptParser::CallNode *>(p_assignment->assigned_value);
		if (!call_node->arguments.is_empty() && call_node->arguments[0]->type == GDScriptParser::Node::ARRAY) {
			const GDScriptParser::ArrayNode *array_node = static_cast<const GDScriptParser::ArrayNode *>(call_node->arguments[0]);

			// Extract the name in "extension ; name" of PackedStringArray.
			for (int i = 0; i < array_node->elements.size(); i++) {
				_extract_fd_constant_strings(array_node->elements[i]);
			}
		}
	} else {
		// If the assignee is not in extract patterns or the assigned_value is not a constant string, try to see if the assigned_value contains tr().
		_assess_expression(p_assignment->assigned_value);
	}
}

void GDScriptEditorTranslationParserPlugin::_extract_from_call(const GDScriptParser::CallNode *p_call) {
	// Extract the translatable strings coming from function calls. For example:
	// tr("___"), get_node("Label").set_text("____"), get_node("LineEdit").set_placeholder("____").

	StringName function_name = p_call->function_name;

	// Variables for extracting tr() and tr_n().
	Vector<String> id_ctx_plural;
	id_ctx_plural.resize(3);
	bool extract_id_ctx_plural = true;

	if (function_name == tr_func) {
		// Extract from tr(id, ctx).
		for (int i = 0; i < p_call->arguments.size(); i++) {
			if (_is_constant_string(p_call->arguments[i])) {
				id_ctx_plural.write[i] = p_call->arguments[i]->reduced_value;
			} else {
				// Avoid adding something like tr("Flying dragon", var_context_level_1). We want to extract both id and context together.
				extract_id_ctx_plural = false;
			}
		}
		if (extract_id_ctx_plural) {
			ids_ctx_plural->push_back(id_ctx_plural);
		}
	} else if (function_name == trn_func) {
		// Extract from tr_n(id, plural, n, ctx).
		Vector<int> indices;
		indices.push_back(0);
		indices.push_back(3);
		indices.push_back(1);
		for (int i = 0; i < indices.size(); i++) {
			if (indices[i] >= p_call->arguments.size()) {
				continue;
			}

			if (_is_constant_string(p_call->arguments[indices[i]])) {
				id_ctx_plural.write[i] = p_call->arguments[indices[i]]->reduced_value;
			} else {
				extract_id_ctx_plural = false;
			}
		}
		if (extract_id_ctx_plural) {
			ids_ctx_plural->push_back(id_ctx_plural);
		}
	} else if (first_arg_patterns.has(function_name)) {
		if (_is_constant_string(p_call->arguments[0])) {
			ids->push_back(p_call->arguments[0]->reduced_value);
		}
	} else if (second_arg_patterns.has(function_name)) {
		if (_is_constant_string(p_call->arguments[1])) {
			ids->push_back(p_call->arguments[1]->reduced_value);
		}
	} else if (function_name == fd_add_filter) {
		// Extract the 'JPE Images' in this example - get_node("FileDialog").add_filter("*.jpg; JPE Images").
		_extract_fd_constant_strings(p_call->arguments[0]);
	} else if (function_name == fd_set_filter && p_call->arguments[0]->type == GDScriptParser::Node::CALL) {
		// FileDialog.set_filters() accepts assignment in the form of PackedStringArray. For example,
		// get_node("FileDialog").set_filters( PackedStringArray(["*.png ; PNG Images","*.gd ; GDScript Files"])).

		const GDScriptParser::CallNode *call_node = static_cast<const GDScriptParser::CallNode *>(p_call->arguments[0]);
		if (call_node->arguments[0]->type == GDScriptParser::Node::ARRAY) {
			const GDScriptParser::ArrayNode *array_node = static_cast<const GDScriptParser::ArrayNode *>(call_node->arguments[0]);
			for (int i = 0; i < array_node->elements.size(); i++) {
				_extract_fd_constant_strings(array_node->elements[i]);
			}
		}
	}

	if (p_call->callee && p_call->callee->type == GDScriptParser::Node::SUBSCRIPT) {
		const GDScriptParser::SubscriptNode *subscript_node = static_cast<const GDScriptParser::SubscriptNode *>(p_call->callee);
		if (subscript_node->base && subscript_node->base->type == GDScriptParser::Node::CALL) {
			const GDScriptParser::CallNode *call_node = static_cast<const GDScriptParser::CallNode *>(subscript_node->base);
			_extract_from_call(call_node);
		}
	}
}

void GDScriptEditorTranslationParserPlugin::_extract_fd_constant_strings(const GDScriptParser::ExpressionNode *p_expression) {
	// Extract the name in "extension ; name".

	if (_is_constant_string(p_expression)) {
		String arg_val = p_expression->reduced_value;
		PackedStringArray arr = arg_val.split(";", true);
		if (arr.size() != 2) {
			ERR_PRINT("Argument for setting FileDialog has bad format.");
			return;
		}
		ids->push_back(arr[1].strip_edges());
	}
}

GDScriptEditorTranslationParserPlugin::GDScriptEditorTranslationParserPlugin() {
	assignment_patterns.insert("text");
	assignment_patterns.insert("placeholder_text");
	assignment_patterns.insert("tooltip_text");

	first_arg_patterns.insert("set_text");
	first_arg_patterns.insert("set_tooltip_text");
	first_arg_patterns.insert("set_placeholder");
	first_arg_patterns.insert("add_tab");
	first_arg_patterns.insert("add_check_item");
	first_arg_patterns.insert("add_item");
	first_arg_patterns.insert("add_multistate_item");
	first_arg_patterns.insert("add_radio_check_item");
	first_arg_patterns.insert("add_separator");
	first_arg_patterns.insert("add_submenu_item");

	second_arg_patterns.insert("set_tab_title");
	second_arg_patterns.insert("add_icon_check_item");
	second_arg_patterns.insert("add_icon_item");
	second_arg_patterns.insert("add_icon_radio_check_item");
	second_arg_patterns.insert("set_item_text");
}

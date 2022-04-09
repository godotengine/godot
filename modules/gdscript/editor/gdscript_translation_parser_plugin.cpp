/*************************************************************************/
/*  gdscript_translation_parser_plugin.cpp                               */
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

#include "gdscript_translation_parser_plugin.h"

#include "core/io/resource_loader.h"
#include "modules/gdscript/gdscript.h"

void GDScriptEditorTranslationParserPlugin::get_recognized_extensions(List<String> *r_extensions) const {
	GDScriptLanguage::get_singleton()->get_recognized_extensions(r_extensions);
}

Error GDScriptEditorTranslationParserPlugin::parse_file(const String &p_path, Vector<String> *r_ids, Vector<Vector<String>> *r_ids_ctx_plural) {
	// Extract all translatable strings using the parsed tree from GDSriptParser.
	// The strategy is to find all ExpressionNode and AssignmentNode from the tree and extract strings if relevant, i.e
	// Search strings in ExpressionNode -> CallNode -> tr(), set_text(), set_placeholder() etc.
	// Search strings in AssignmentNode -> text = "__", hint_tooltip = "__" etc.

	Error err;
	RES loaded_res = ResourceLoader::load(p_path, "", ResourceFormatLoader::CACHE_MODE_REUSE, &err);
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
	if (err != OK) {
		ERR_PRINT("Failed to parse with GDScript with GDScriptParser.");
		return err;
	}

	// Traverse through the parsed tree from GDScriptParser.
	GDScriptParser::ClassNode *c = parser.get_tree();
	_traverse_class(c);

	return OK;
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
		GDScriptParser::Node *statement = statements[i];

		// Statements with Node type constant, break, continue, pass, breakpoint are skipped because they can't contain translatable strings.
		switch (statement->type) {
			case GDScriptParser::Node::VARIABLE:
				_assess_expression(static_cast<GDScriptParser::VariableNode *>(statement)->initializer);
				break;
			case GDScriptParser::Node::IF: {
				GDScriptParser::IfNode *if_node = static_cast<GDScriptParser::IfNode *>(statement);
				_assess_expression(if_node->condition);
				//FIXME : if the elif logic is changed in GDScriptParser, then this probably will have to change as well. See GDScriptParser::TreePrinter::print_if().
				_traverse_block(if_node->true_block);
				_traverse_block(if_node->false_block);
				break;
			}
			case GDScriptParser::Node::FOR: {
				GDScriptParser::ForNode *for_node = static_cast<GDScriptParser::ForNode *>(statement);
				_assess_expression(for_node->list);
				_traverse_block(for_node->loop);
				break;
			}
			case GDScriptParser::Node::WHILE: {
				GDScriptParser::WhileNode *while_node = static_cast<GDScriptParser::WhileNode *>(statement);
				_assess_expression(while_node->condition);
				_traverse_block(while_node->loop);
				break;
			}
			case GDScriptParser::Node::MATCH: {
				GDScriptParser::MatchNode *match_node = static_cast<GDScriptParser::MatchNode *>(statement);
				_assess_expression(match_node->test);
				for (int j = 0; j < match_node->branches.size(); j++) {
					_traverse_block(match_node->branches[j]->block);
				}
				break;
			}
			case GDScriptParser::Node::RETURN:
				_assess_expression(static_cast<GDScriptParser::ReturnNode *>(statement)->return_value);
				break;
			case GDScriptParser::Node::ASSERT:
				_assess_expression((static_cast<GDScriptParser::AssertNode *>(statement))->condition);
				break;
			case GDScriptParser::Node::ASSIGNMENT:
				_assess_assignment(static_cast<GDScriptParser::AssignmentNode *>(statement));
				break;
			default:
				if (statement->is_expression()) {
					_assess_expression(static_cast<GDScriptParser::ExpressionNode *>(statement));
				}
				break;
		}
	}
}

void GDScriptEditorTranslationParserPlugin::_assess_expression(GDScriptParser::ExpressionNode *p_expression) {
	// Explore all ExpressionNodes to find CallNodes which contain translation strings, such as tr(), set_text() etc.
	// tr() can be embedded quite deep within multiple ExpressionNodes so need to dig down to search through all ExpressionNodes.
	if (!p_expression) {
		return;
	}

	// ExpressionNode of type await, cast, get_node, identifier, literal, preload, self, subscript, unary are ignored as they can't be CallNode
	// containing translation strings.
	switch (p_expression->type) {
		case GDScriptParser::Node::ARRAY: {
			GDScriptParser::ArrayNode *array_node = static_cast<GDScriptParser::ArrayNode *>(p_expression);
			for (int i = 0; i < array_node->elements.size(); i++) {
				_assess_expression(array_node->elements[i]);
			}
			break;
		}
		case GDScriptParser::Node::ASSIGNMENT:
			_assess_assignment(static_cast<GDScriptParser::AssignmentNode *>(p_expression));
			break;
		case GDScriptParser::Node::BINARY_OPERATOR: {
			GDScriptParser::BinaryOpNode *binary_op_node = static_cast<GDScriptParser::BinaryOpNode *>(p_expression);
			_assess_expression(binary_op_node->left_operand);
			_assess_expression(binary_op_node->right_operand);
			break;
		}
		case GDScriptParser::Node::CALL: {
			GDScriptParser::CallNode *call_node = static_cast<GDScriptParser::CallNode *>(p_expression);
			_extract_from_call(call_node);
			for (int i = 0; i < call_node->arguments.size(); i++) {
				_assess_expression(call_node->arguments[i]);
			}
		} break;
		case GDScriptParser::Node::DICTIONARY: {
			GDScriptParser::DictionaryNode *dict_node = static_cast<GDScriptParser::DictionaryNode *>(p_expression);
			for (int i = 0; i < dict_node->elements.size(); i++) {
				_assess_expression(dict_node->elements[i].key);
				_assess_expression(dict_node->elements[i].value);
			}
			break;
		}
		case GDScriptParser::Node::TERNARY_OPERATOR: {
			GDScriptParser::TernaryOpNode *ternary_op_node = static_cast<GDScriptParser::TernaryOpNode *>(p_expression);
			_assess_expression(ternary_op_node->condition);
			_assess_expression(ternary_op_node->true_expr);
			_assess_expression(ternary_op_node->false_expr);
			break;
		}
		default:
			break;
	}
}

void GDScriptEditorTranslationParserPlugin::_assess_assignment(GDScriptParser::AssignmentNode *p_assignment) {
	// Extract the translatable strings coming from assignments. For example, get_node("Label").text = "____"

	StringName assignee_name;
	if (p_assignment->assignee->type == GDScriptParser::Node::IDENTIFIER) {
		assignee_name = static_cast<GDScriptParser::IdentifierNode *>(p_assignment->assignee)->name;
	} else if (p_assignment->assignee->type == GDScriptParser::Node::SUBSCRIPT) {
		assignee_name = static_cast<GDScriptParser::SubscriptNode *>(p_assignment->assignee)->attribute->name;
	}

	if (assignment_patterns.has(assignee_name) && p_assignment->assigned_value->type == GDScriptParser::Node::LITERAL) {
		// If the assignment is towards one of the extract patterns (text, hint_tooltip etc.), and the value is a string literal, we collect the string.
		ids->push_back(static_cast<GDScriptParser::LiteralNode *>(p_assignment->assigned_value)->value);
	} else if (assignee_name == fd_filters && p_assignment->assigned_value->type == GDScriptParser::Node::CALL) {
		// FileDialog.filters accepts assignment in the form of PackedStringArray. For example,
		// get_node("FileDialog").filters = PackedStringArray(["*.png ; PNG Images","*.gd ; GDScript Files"]).

		GDScriptParser::CallNode *call_node = static_cast<GDScriptParser::CallNode *>(p_assignment->assigned_value);
		if (call_node->arguments[0]->type == GDScriptParser::Node::ARRAY) {
			GDScriptParser::ArrayNode *array_node = static_cast<GDScriptParser::ArrayNode *>(call_node->arguments[0]);

			// Extract the name in "extension ; name" of PackedStringArray.
			for (int i = 0; i < array_node->elements.size(); i++) {
				_extract_fd_literals(array_node->elements[i]);
			}
		}
	} else {
		// If the assignee is not in extract patterns or the assigned_value is not Literal type, try to see if the assigned_value contains tr().
		_assess_expression(p_assignment->assigned_value);
	}
}

void GDScriptEditorTranslationParserPlugin::_extract_from_call(GDScriptParser::CallNode *p_call) {
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
			if (p_call->arguments[i]->type == GDScriptParser::Node::LITERAL) {
				id_ctx_plural.write[i] = static_cast<GDScriptParser::LiteralNode *>(p_call->arguments[i])->value;
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

			if (p_call->arguments[indices[i]]->type == GDScriptParser::Node::LITERAL) {
				id_ctx_plural.write[i] = static_cast<GDScriptParser::LiteralNode *>(p_call->arguments[indices[i]])->value;
			} else {
				extract_id_ctx_plural = false;
			}
		}
		if (extract_id_ctx_plural) {
			ids_ctx_plural->push_back(id_ctx_plural);
		}
	} else if (first_arg_patterns.has(function_name)) {
		// Extracting argument with only string literals. In other words, not extracting something like set_text("hello " + some_var).
		if (p_call->arguments[0]->type == GDScriptParser::Node::LITERAL) {
			ids->push_back(static_cast<GDScriptParser::LiteralNode *>(p_call->arguments[0])->value);
		}
	} else if (second_arg_patterns.has(function_name)) {
		if (p_call->arguments[1]->type == GDScriptParser::Node::LITERAL) {
			ids->push_back(static_cast<GDScriptParser::LiteralNode *>(p_call->arguments[1])->value);
		}
	} else if (function_name == fd_add_filter) {
		// Extract the 'JPE Images' in this example - get_node("FileDialog").add_filter("*.jpg; JPE Images").
		_extract_fd_literals(p_call->arguments[0]);

	} else if (function_name == fd_set_filter && p_call->arguments[0]->type == GDScriptParser::Node::CALL) {
		// FileDialog.set_filters() accepts assignment in the form of PackedStringArray. For example,
		// get_node("FileDialog").set_filters( PackedStringArray(["*.png ; PNG Images","*.gd ; GDScript Files"])).

		GDScriptParser::CallNode *call_node = static_cast<GDScriptParser::CallNode *>(p_call->arguments[0]);
		if (call_node->arguments[0]->type == GDScriptParser::Node::ARRAY) {
			GDScriptParser::ArrayNode *array_node = static_cast<GDScriptParser::ArrayNode *>(call_node->arguments[0]);
			for (int i = 0; i < array_node->elements.size(); i++) {
				_extract_fd_literals(array_node->elements[i]);
			}
		}
	}
}

void GDScriptEditorTranslationParserPlugin::_extract_fd_literals(GDScriptParser::ExpressionNode *p_expression) {
	// Extract the name in "extension ; name".

	if (p_expression->type == GDScriptParser::Node::LITERAL) {
		String arg_val = String(static_cast<GDScriptParser::LiteralNode *>(p_expression)->value);
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
	assignment_patterns.insert("hint_tooltip");

	first_arg_patterns.insert("set_text");
	first_arg_patterns.insert("set_tooltip");
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

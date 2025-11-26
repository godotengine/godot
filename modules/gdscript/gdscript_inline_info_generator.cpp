/**************************************************************************/
/*  gdscript_inline_info_generator.cpp                                    */
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

#include "gdscript_inline_info_generator.h"

void GDScriptInlineInfoGenerator::add_color_info(GDScriptParser::Node *p_node) {
	Dictionary info;
	info["type"] = SNAME("color");
	info["line"] = p_node->start_line;
	info["column"] = p_node->start_column;

	(*inline_info)[p_node->start_line].push_back(info);
}

void GDScriptInlineInfoGenerator::add_parameters_info(GDScriptParser::Node *p_node, const PropertyInfo &p_info) {
	Dictionary info;
	info["type"] = SNAME("parameter");
	info["line"] = p_node->start_line;
	info["column"] = p_node->start_column;
	info["name"] = p_info.name;

	(*inline_info)[p_node->start_line].push_back(info);
}

void GDScriptInlineInfoGenerator::visit_class(GDScriptParser::ClassNode *p_class) {
	parser->current_class = p_class;

	for (GDScriptParser::ClassNode::Member &member : p_class->members) {
		switch (member.type) {
			case GDScriptParser::ClassNode::Member::CLASS:
				visit_class(member.m_class);
				break;
			case GDScriptParser::ClassNode::Member::VARIABLE:
				visit_variable(member.variable);
				break;
			case GDScriptParser::ClassNode::Member::CONSTANT:
				visit_constant(member.constant);
				break;
			case GDScriptParser::ClassNode::Member::FUNCTION:
				visit_function(member.function);
				break;
			default:
				break;
		}
	}

	if (p_class->get_datatype().class_type->outer) {
		parser->current_class = p_class->get_datatype().class_type->outer;
	}
}

void GDScriptInlineInfoGenerator::visit_variable(GDScriptParser::VariableNode *p_variable) {
	if (p_variable->initializer) {
		visit_expression(p_variable->initializer);
	}
	if (p_variable->property == GDScriptParser::VariableNode::PROP_SETGET) {
		if (p_variable->setter) {
			visit_function(p_variable->setter);
		}
		if (p_variable->getter) {
			visit_function(p_variable->getter);
		}
	}
}

void GDScriptInlineInfoGenerator::visit_constant(GDScriptParser::ConstantNode *p_constant) {
	if (p_constant->initializer) {
		visit_expression(p_constant->initializer);
	}
}

void GDScriptInlineInfoGenerator::visit_function(GDScriptParser::FunctionNode *p_function) {
	if (p_function->body) {
		visit_suite(p_function->body);
	}
}

void GDScriptInlineInfoGenerator::visit_suite(GDScriptParser::SuiteNode *p_suite) {
	for (GDScriptParser::Node *statement : p_suite->statements) {
		visit_statement(statement);
	}
}

void GDScriptInlineInfoGenerator::visit_statement(GDScriptParser::Node *p_statement) {
	switch (p_statement->type) {
		case GDScriptParser::Node::VARIABLE:
			visit_variable(static_cast<GDScriptParser::VariableNode *>(p_statement));
			break;
		case GDScriptParser::Node::CONSTANT:
			visit_constant(static_cast<GDScriptParser::ConstantNode *>(p_statement));
			break;
		case GDScriptParser::Node::IF:
			visit_if(static_cast<GDScriptParser::IfNode *>(p_statement));
			break;
		case GDScriptParser::Node::FOR:
			visit_for(static_cast<GDScriptParser::ForNode *>(p_statement));
			break;
		case GDScriptParser::Node::WHILE:
			visit_while(static_cast<GDScriptParser::WhileNode *>(p_statement));
			break;
		case GDScriptParser::Node::MATCH:
			visit_match(static_cast<GDScriptParser::MatchNode *>(p_statement));
			break;
		case GDScriptParser::Node::RETURN:
			visit_return(static_cast<GDScriptParser::ReturnNode *>(p_statement));
			break;
		case GDScriptParser::Node::ASSERT:
			visit_assert(static_cast<GDScriptParser::AssertNode *>(p_statement));
			break;
		case GDScriptParser::Node::BREAK:
		case GDScriptParser::Node::CONTINUE:
		case GDScriptParser::Node::PASS:
		case GDScriptParser::Node::BREAKPOINT:
			break;
		default:
			if (p_statement->is_expression()) {
				visit_expression(static_cast<GDScriptParser::ExpressionNode *>(p_statement));
			}
			break;
	}
}

void GDScriptInlineInfoGenerator::visit_if(GDScriptParser::IfNode *p_if) {
	if (p_if->condition) {
		visit_expression(p_if->condition);
	}
	if (p_if->true_block) {
		visit_suite(p_if->true_block);
	}
	if (p_if->false_block) {
		visit_suite(p_if->false_block);
	}
}

void GDScriptInlineInfoGenerator::visit_for(GDScriptParser::ForNode *p_for) {
	if (p_for->list) {
		visit_expression(p_for->list);
	}
	if (p_for->loop) {
		visit_suite(p_for->loop);
	}
}

void GDScriptInlineInfoGenerator::visit_while(GDScriptParser::WhileNode *p_while) {
	if (p_while->condition) {
		visit_expression(p_while->condition);
	}
	if (p_while->loop) {
		visit_suite(p_while->loop);
	}
}

void GDScriptInlineInfoGenerator::visit_match(GDScriptParser::MatchNode *p_match) {
	visit_expression(p_match->test);

	for (GDScriptParser::MatchBranchNode *branch : p_match->branches) {
		for (GDScriptParser::PatternNode *pattern : branch->patterns) {
			if (pattern->pattern_type == GDScriptParser::PatternNode::PT_EXPRESSION) {
				visit_expression(pattern->expression);
			}
		}
		if (branch->guard_body) {
			visit_suite(branch->guard_body);
		}
		if (branch->block) {
			visit_suite(branch->block);
		}
	}
}

void GDScriptInlineInfoGenerator::visit_return(GDScriptParser::ReturnNode *p_return) {
	if (p_return->return_value) {
		visit_expression(p_return->return_value);
	}
}

void GDScriptInlineInfoGenerator::visit_assert(GDScriptParser::AssertNode *p_assert) {
	if (p_assert->condition) {
		visit_expression(p_assert->condition);
	}
	if (p_assert->message) {
		visit_expression(p_assert->message);
	}
}

void GDScriptInlineInfoGenerator::visit_expression(GDScriptParser::ExpressionNode *p_expression) {
	if (p_expression == nullptr) {
		return;
	}

	switch (p_expression->type) {
		case GDScriptParser::Node::CALL:
			visit_call(static_cast<GDScriptParser::CallNode *>(p_expression));
			break;
		case GDScriptParser::Node::AWAIT:
			if (static_cast<GDScriptParser::AwaitNode *>(p_expression)->to_await) {
				visit_expression(static_cast<GDScriptParser::AwaitNode *>(p_expression)->to_await);
			}
			break;
		case GDScriptParser::Node::ASSIGNMENT: {
			GDScriptParser::AssignmentNode *assign = static_cast<GDScriptParser::AssignmentNode *>(p_expression);
			if (assign->assignee) {
				visit_expression(assign->assignee);
			}
			if (assign->assigned_value) {
				visit_expression(assign->assigned_value);
			}
			break;
		}
		case GDScriptParser::Node::BINARY_OPERATOR: {
			GDScriptParser::BinaryOpNode *binop = static_cast<GDScriptParser::BinaryOpNode *>(p_expression);
			if (binop->left_operand) {
				visit_expression(binop->left_operand);
			}
			if (binop->right_operand) {
				visit_expression(binop->right_operand);
			}
			break;
		}
		case GDScriptParser::Node::UNARY_OPERATOR: {
			GDScriptParser::UnaryOpNode *unop = static_cast<GDScriptParser::UnaryOpNode *>(p_expression);
			if (unop->operand) {
				visit_expression(unop->operand);
			}
			break;
		}
		case GDScriptParser::Node::TERNARY_OPERATOR: {
			GDScriptParser::TernaryOpNode *ternary = static_cast<GDScriptParser::TernaryOpNode *>(p_expression);
			if (ternary->condition) {
				visit_expression(ternary->condition);
			}
			if (ternary->true_expr) {
				visit_expression(ternary->true_expr);
			}
			if (ternary->false_expr) {
				visit_expression(ternary->false_expr);
			}
			break;
		}
		case GDScriptParser::Node::SUBSCRIPT: {
			GDScriptParser::SubscriptNode *subscript = static_cast<GDScriptParser::SubscriptNode *>(p_expression);
			if (subscript->base) {
				visit_expression(subscript->base);
			}
			if (subscript->index) {
				visit_expression(subscript->index);
			}
			break;
		}
		case GDScriptParser::Node::ARRAY: {
			GDScriptParser::ArrayNode *array = static_cast<GDScriptParser::ArrayNode *>(p_expression);
			for (GDScriptParser::ExpressionNode *element : array->elements) {
				visit_expression(element);
			}
			break;
		}
		case GDScriptParser::Node::DICTIONARY: {
			GDScriptParser::DictionaryNode *dict = static_cast<GDScriptParser::DictionaryNode *>(p_expression);
			for (const GDScriptParser::DictionaryNode::Pair &pair : dict->elements) {
				visit_expression(pair.key);
				visit_expression(pair.value);
			}
			break;
		}
		case GDScriptParser::Node::CAST: {
			GDScriptParser::CastNode *cast = static_cast<GDScriptParser::CastNode *>(p_expression);
			if (cast->operand) {
				visit_expression(cast->operand);
			}
			break;
		}
		case GDScriptParser::Node::LAMBDA: {
			GDScriptParser::LambdaNode *lambda = static_cast<GDScriptParser::LambdaNode *>(p_expression);
			visit_function(lambda->function);
			break;
		}

		default:
			break;
	}
}

void GDScriptInlineInfoGenerator::visit_call(GDScriptParser::CallNode *p_call) {
	if (p_call->callee) {
		visit_expression(p_call->callee);
	}

	MethodInfo function_info;

	GDScriptParser::Node::Type callee_type = p_call->get_callee_type();

	if (callee_type == GDScriptParser::Node::SUBSCRIPT) {
		GDScriptParser::SubscriptNode *subscript = static_cast<GDScriptParser::SubscriptNode *>(p_call->callee);
		function_info = get_method_info_on_base(subscript->base, p_call->function_name);

		if (p_call->is_static && subscript->base->type == GDScriptParser::Node::IDENTIFIER) {
			GDScriptParser::IdentifierNode *identifier = static_cast<GDScriptParser::IdentifierNode *>(subscript->base);
			if (identifier->name == SNAME("Color")) {
				add_color_info(identifier);
			}
		}

	} else if (p_call->is_super || callee_type == GDScriptParser::Node::IDENTIFIER) {
		Variant::Type builtin_type = GDScriptParser::get_builtin_type(p_call->function_name);
		bool is_valid_builtin_type = builtin_type < Variant::VARIANT_MAX;

		if (is_valid_builtin_type) {
			if (builtin_type == Variant::COLOR) {
				add_color_info(p_call);
			}

			List<MethodInfo> constructors;
			Variant::get_constructor_list(builtin_type, &constructors);

			for (const MethodInfo &constructor_info : constructors) {
				bool arity_matches = p_call->arguments.size() <= constructor_info.arguments.size() && p_call->arguments.size() >= constructor_info.arguments.size() - constructor_info.default_arguments.size();
				if (!arity_matches) {
					continue;
				}

				bool type_matches = true;
				for (int i = 0; i < p_call->arguments.size() && type_matches; i++) {
					const PropertyInfo &parameter = constructor_info.arguments[i];
					const PropertyInfo argument = p_call->arguments[i]->get_datatype().to_property_info(parameter.name);

					type_matches = Variant::can_convert_strict(argument.type, parameter.type) && parameter.class_name == argument.class_name;
				}

				if (type_matches) {
					function_info = constructor_info;
					break;
				}
			}

		} else if (Variant::has_utility_function(p_call->function_name)) {
			function_info = Variant::get_utility_function_info(p_call->function_name);
		} else if (GDScriptUtilityFunctions::function_exists(p_call->function_name)) {
			function_info = GDScriptUtilityFunctions::get_function_info(p_call->function_name);
		} else {
			function_info = get_method_info_on_base(parser->current_class, p_call->function_name);
		}
	}

	long hint_count = MIN(p_call->arguments.size(), function_info.arguments.size());
	for (int i = 0; i < p_call->arguments.size(); i++) {
		GDScriptParser::ExpressionNode *argument = p_call->arguments[i];
		if (i < hint_count) {
			const PropertyInfo &parameter = function_info.arguments[i];
			add_parameters_info(argument, parameter);
		}

		visit_expression(argument);
	}
}

MethodInfo GDScriptInlineInfoGenerator::get_method_info_on_base(GDScriptParser::Node *p_base, const StringName &p_function) {
	GDScriptParser::DataType base_datatype = p_base->get_datatype();

	MethodInfo info;

	switch (base_datatype.kind) {
		case GDScriptParser::DataType::BUILTIN: {
			if (Variant::has_builtin_method(base_datatype.builtin_type, p_function)) {
				info = Variant::get_builtin_method_info(base_datatype.builtin_type, p_function);
			}
			break;
		}
		case GDScriptParser::DataType::SCRIPT: {
			Ref<GDScript> script = base_datatype.script_type;

			if (script.is_valid()) {
				info = script->get_method_info(p_function);
			}
			break;
		}
		case GDScriptParser::DataType::UNRESOLVED:
		case GDScriptParser::DataType::NATIVE:
		case GDScriptParser::DataType::CLASS: {
			StringName function_to_find = p_function;
			StringName base_name;

			if (base_datatype.kind == GDScriptParser::DataType::NATIVE) {
				base_name = base_datatype.native_type;
			} else if (base_datatype.kind == GDScriptParser::DataType::CLASS) {
				base_name = base_datatype.to_string();
			} else if (base_datatype.kind == GDScriptParser::DataType::UNRESOLVED && p_base->type == GDScriptParser::Node::IDENTIFIER) {
				base_name = static_cast<GDScriptParser::IdentifierNode *>(p_base)->name;
			}

			// Special case for static methods on builtins, which are UNRESOLVED.
			Variant::Type builtin_type = GDScriptParser::get_builtin_type(base_name);
			bool is_valid_builtin_type = builtin_type < Variant::VARIANT_MAX;

			// Special case for the constructor, that refers to the _init method.
			if (p_function == SNAME("new")) {
				function_to_find = GDScriptLanguage::get_singleton()->strings._init;
			}

			if (is_valid_builtin_type) {
				info = Variant::get_builtin_method_info(builtin_type, function_to_find);

			} else if (ClassDB::class_exists(base_name)) {
				while (ClassDB::class_exists(base_name)) {
					if (ClassDB::has_method(base_name, function_to_find, true)) {
						ClassDB::get_method_info(base_name, function_to_find, &info);
						break;
					}
					base_name = ClassDB::get_parent_class(base_name);
				}

			} else {
				GDScriptParser::ClassNode *base_class = base_datatype.class_type;
				bool function_found = false;

				while (!function_found && base_class != nullptr) {
					if (base_class->has_member(function_to_find)) {
						GDScriptParser::ClassNode::Member member = base_class->get_member(function_to_find);

						if (member.type == GDScriptParser::ClassNode::Member::FUNCTION) {
							info = member.function->info;
						}
						function_found = true;
					}

					base_class = base_class->base_type.class_type;
				}
			}
			break;
		}
		case GDScriptParser::DataType::ENUM:
		case GDScriptParser::DataType::VARIANT:
		case GDScriptParser::DataType::RESOLVING:
			break;
	}

	return info;
}

Error GDScriptInlineInfoGenerator::generate(HashMap<int, TypedArray<Dictionary>> *p_info) {
	ERR_FAIL_NULL_V(p_info, FAILED);

	inline_info = p_info;
	inline_info->clear();

	GDScriptParser::ClassNode *root_node = parser->get_tree();
	ERR_FAIL_NULL_V_MSG(root_node, FAILED, "Couldn't generate inline info. The code must be parsed before generation.");
	ERR_FAIL_COND_V_MSG(!root_node->resolved_body || !root_node->resolved_interface, FAILED, "Couldn't generate inline info. The code must be analyzed before generation.");

	visit_class(root_node);

	return OK;
}

GDScriptInlineInfoGenerator::GDScriptInlineInfoGenerator(GDScriptParser *p_parser) {
	parser = p_parser;
}

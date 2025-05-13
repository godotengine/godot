/**************************************************************************/
/*  gdscript_tree_printer.cpp                                             */
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

/*---------- PRETTY PRINT FOR DEBUG ----------*/
#ifdef DEBUG_ENABLED

#include "gdscript_tree_printer.h"

void GDScriptTreePrinter::increase_indent() {
	indent_level++;
	indent = "";
	for (int i = 0; i < indent_level * 4; i++) {
		if (i % 4 == 0) {
			indent += "|";
		} else {
			indent += " ";
		}
	}
}

void GDScriptTreePrinter::decrease_indent() {
	indent_level--;
	indent = "";
	for (int i = 0; i < indent_level * 4; i++) {
		if (i % 4 == 0) {
			indent += "|";
		} else {
			indent += " ";
		}
	}
}

void GDScriptTreePrinter::push_line(const String &p_line) {
	if (!p_line.is_empty()) {
		push_text(p_line);
	}
	printed += "\n";
	pending_indent = true;
}

void GDScriptTreePrinter::push_text(const String &p_text) {
	if (pending_indent) {
		printed += indent;
		pending_indent = false;
	}
	printed += p_text;
}

void GDScriptTreePrinter::print_annotation(const GDScriptParser::AnnotationNode *p_annotation) {
	push_text(p_annotation->name);
	push_text(" (");
	for (int i = 0; i < p_annotation->arguments.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_expression(p_annotation->arguments[i]);
	}
	push_line(")");
}

void GDScriptTreePrinter::print_array(GDScriptParser::ArrayNode *p_array) {
	push_text("[ ");
	for (int i = 0; i < p_array->elements.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_expression(p_array->elements[i]);
	}
	push_text(" ]");
}

void GDScriptTreePrinter::print_assert(GDScriptParser::AssertNode *p_assert) {
	push_text("Assert ( ");
	print_expression(p_assert->condition);
	push_line(" )");
}

void GDScriptTreePrinter::print_assignment(GDScriptParser::AssignmentNode *p_assignment) {
	switch (p_assignment->assignee->type) {
		case GDScriptParser::Node::IDENTIFIER:
			print_identifier(static_cast<GDScriptParser::IdentifierNode *>(p_assignment->assignee));
			break;
		case GDScriptParser::Node::SUBSCRIPT:
			print_subscript(static_cast<GDScriptParser::SubscriptNode *>(p_assignment->assignee));
			break;
		default:
			break; // Unreachable.
	}

	push_text(" ");
	switch (p_assignment->operation) {
		case GDScriptParser::AssignmentNode::OP_ADDITION:
			push_text("+");
			break;
		case GDScriptParser::AssignmentNode::OP_SUBTRACTION:
			push_text("-");
			break;
		case GDScriptParser::AssignmentNode::OP_MULTIPLICATION:
			push_text("*");
			break;
		case GDScriptParser::AssignmentNode::OP_DIVISION:
			push_text("/");
			break;
		case GDScriptParser::AssignmentNode::OP_MODULO:
			push_text("%");
			break;
		case GDScriptParser::AssignmentNode::OP_POWER:
			push_text("**");
			break;
		case GDScriptParser::AssignmentNode::OP_BIT_SHIFT_LEFT:
			push_text("<<");
			break;
		case GDScriptParser::AssignmentNode::OP_BIT_SHIFT_RIGHT:
			push_text(">>");
			break;
		case GDScriptParser::AssignmentNode::OP_BIT_AND:
			push_text("&");
			break;
		case GDScriptParser::AssignmentNode::OP_BIT_OR:
			push_text("|");
			break;
		case GDScriptParser::AssignmentNode::OP_BIT_XOR:
			push_text("^");
			break;
		case GDScriptParser::AssignmentNode::OP_NONE:
			break;
	}
	push_text("= ");
	print_expression(p_assignment->assigned_value);
	push_line();
}

void GDScriptTreePrinter::print_await(GDScriptParser::AwaitNode *p_await) {
	push_text("Await ");
	print_expression(p_await->to_await);
}

void GDScriptTreePrinter::print_binary_op(GDScriptParser::BinaryOpNode *p_binary_op) {
	// Surround in parenthesis for disambiguation.
	push_text("(");
	print_expression(p_binary_op->left_operand);
	switch (p_binary_op->operation) {
		case GDScriptParser::BinaryOpNode::OP_ADDITION:
			push_text(" + ");
			break;
		case GDScriptParser::BinaryOpNode::OP_SUBTRACTION:
			push_text(" - ");
			break;
		case GDScriptParser::BinaryOpNode::OP_MULTIPLICATION:
			push_text(" * ");
			break;
		case GDScriptParser::BinaryOpNode::OP_DIVISION:
			push_text(" / ");
			break;
		case GDScriptParser::BinaryOpNode::OP_MODULO:
			push_text(" % ");
			break;
		case GDScriptParser::BinaryOpNode::OP_POWER:
			push_text(" ** ");
			break;
		case GDScriptParser::BinaryOpNode::OP_BIT_LEFT_SHIFT:
			push_text(" << ");
			break;
		case GDScriptParser::BinaryOpNode::OP_BIT_RIGHT_SHIFT:
			push_text(" >> ");
			break;
		case GDScriptParser::BinaryOpNode::OP_BIT_AND:
			push_text(" & ");
			break;
		case GDScriptParser::BinaryOpNode::OP_BIT_OR:
			push_text(" | ");
			break;
		case GDScriptParser::BinaryOpNode::OP_BIT_XOR:
			push_text(" ^ ");
			break;
		case GDScriptParser::BinaryOpNode::OP_LOGIC_AND:
			push_text(" AND ");
			break;
		case GDScriptParser::BinaryOpNode::OP_LOGIC_OR:
			push_text(" OR ");
			break;
		case GDScriptParser::BinaryOpNode::OP_CONTENT_TEST:
			push_text(" IN ");
			break;
		case GDScriptParser::BinaryOpNode::OP_COMP_EQUAL:
			push_text(" == ");
			break;
		case GDScriptParser::BinaryOpNode::OP_COMP_NOT_EQUAL:
			push_text(" != ");
			break;
		case GDScriptParser::BinaryOpNode::OP_COMP_LESS:
			push_text(" < ");
			break;
		case GDScriptParser::BinaryOpNode::OP_COMP_LESS_EQUAL:
			push_text(" <= ");
			break;
		case GDScriptParser::BinaryOpNode::OP_COMP_GREATER:
			push_text(" > ");
			break;
		case GDScriptParser::BinaryOpNode::OP_COMP_GREATER_EQUAL:
			push_text(" >= ");
			break;
	}
	print_expression(p_binary_op->right_operand);
	// Surround in parenthesis for disambiguation.
	push_text(")");
}

void GDScriptTreePrinter::print_call(GDScriptParser::CallNode *p_call) {
	if (p_call->is_super) {
		push_text("super");
		if (p_call->callee != nullptr) {
			push_text(".");
			print_expression(p_call->callee);
		}
	} else {
		print_expression(p_call->callee);
	}
	push_text("( ");
	for (int i = 0; i < p_call->arguments.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_expression(p_call->arguments[i]);
	}
	push_text(" )");
}

void GDScriptTreePrinter::print_cast(GDScriptParser::CastNode *p_cast) {
	print_expression(p_cast->operand);
	push_text(" AS ");
	print_type(p_cast->cast_type);
}

void GDScriptTreePrinter::print_class(GDScriptParser::ClassNode *p_class) {
	push_text("Class ");
	if (p_class->identifier == nullptr) {
		push_text("<unnamed>");
	} else {
		print_identifier(p_class->identifier);
	}

	if (p_class->extends_used) {
		bool first = true;
		push_text(" Extends ");
		if (!p_class->extends_path.is_empty()) {
			push_text(vformat(R"("%s")", p_class->extends_path));
			first = false;
		}
		for (int i = 0; i < p_class->extends.size(); i++) {
			if (!first) {
				push_text(".");
			} else {
				first = false;
			}
			push_text(p_class->extends[i]->name);
		}
	}

	push_line(" :");

	increase_indent();

	for (int i = 0; i < p_class->members.size(); i++) {
		const GDScriptParser::ClassNode::Member &m = p_class->members[i];

		switch (m.type) {
			case GDScriptParser::ClassNode::Member::CLASS:
				print_class(m.m_class);
				break;
			case GDScriptParser::ClassNode::Member::VARIABLE:
				print_variable(m.variable);
				break;
			case GDScriptParser::ClassNode::Member::CONSTANT:
				print_constant(m.constant);
				break;
			case GDScriptParser::ClassNode::Member::SIGNAL:
				print_signal(m.signal);
				break;
			case GDScriptParser::ClassNode::Member::FUNCTION:
				print_function(m.function);
				break;
			case GDScriptParser::ClassNode::Member::ENUM:
				print_enum(m.m_enum);
				break;
			case GDScriptParser::ClassNode::Member::ENUM_VALUE:
				break; // Nothing. Will be printed by enum.
			case GDScriptParser::ClassNode::Member::GROUP:
				break; // Nothing. Groups are only used by inspector.
			case GDScriptParser::ClassNode::Member::UNDEFINED:
				push_line("<unknown member>");
				break;
		}
	}

	decrease_indent();
}

void GDScriptTreePrinter::print_constant(GDScriptParser::ConstantNode *p_constant) {
	push_text("Constant ");
	print_identifier(p_constant->identifier);

	increase_indent();

	push_line();
	push_text("= ");
	if (p_constant->initializer == nullptr) {
		push_text("<missing value>");
	} else {
		print_expression(p_constant->initializer);
	}
	decrease_indent();
	push_line();
}

void GDScriptTreePrinter::print_dictionary(GDScriptParser::DictionaryNode *p_dictionary) {
	push_line("{");
	increase_indent();
	for (int i = 0; i < p_dictionary->elements.size(); i++) {
		print_expression(p_dictionary->elements[i].key);
		if (p_dictionary->style == GDScriptParser::DictionaryNode::PYTHON_DICT) {
			push_text(" : ");
		} else {
			push_text(" = ");
		}
		print_expression(p_dictionary->elements[i].value);
		push_line(" ,");
	}
	decrease_indent();
	push_text("}");
}

void GDScriptTreePrinter::print_expression(GDScriptParser::ExpressionNode *p_expression) {
	if (p_expression == nullptr) {
		push_text("<invalid expression>");
		return;
	}
	switch (p_expression->type) {
		case GDScriptParser::Node::ARRAY:
			print_array(static_cast<GDScriptParser::ArrayNode *>(p_expression));
			break;
		case GDScriptParser::Node::ASSIGNMENT:
			print_assignment(static_cast<GDScriptParser::AssignmentNode *>(p_expression));
			break;
		case GDScriptParser::Node::AWAIT:
			print_await(static_cast<GDScriptParser::AwaitNode *>(p_expression));
			break;
		case GDScriptParser::Node::BINARY_OPERATOR:
			print_binary_op(static_cast<GDScriptParser::BinaryOpNode *>(p_expression));
			break;
		case GDScriptParser::Node::CALL:
			print_call(static_cast<GDScriptParser::CallNode *>(p_expression));
			break;
		case GDScriptParser::Node::CAST:
			print_cast(static_cast<GDScriptParser::CastNode *>(p_expression));
			break;
		case GDScriptParser::Node::DICTIONARY:
			print_dictionary(static_cast<GDScriptParser::DictionaryNode *>(p_expression));
			break;
		case GDScriptParser::Node::GET_NODE:
			print_get_node(static_cast<GDScriptParser::GetNodeNode *>(p_expression));
			break;
		case GDScriptParser::Node::IDENTIFIER:
			print_identifier(static_cast<GDScriptParser::IdentifierNode *>(p_expression));
			break;
		case GDScriptParser::Node::LAMBDA:
			print_lambda(static_cast<GDScriptParser::LambdaNode *>(p_expression));
			break;
		case GDScriptParser::Node::LITERAL:
			print_literal(static_cast<GDScriptParser::LiteralNode *>(p_expression));
			break;
		case GDScriptParser::Node::PRELOAD:
			print_preload(static_cast<GDScriptParser::PreloadNode *>(p_expression));
			break;
		case GDScriptParser::Node::SELF:
			print_self(static_cast<GDScriptParser::SelfNode *>(p_expression));
			break;
		case GDScriptParser::Node::SUBSCRIPT:
			print_subscript(static_cast<GDScriptParser::SubscriptNode *>(p_expression));
			break;
		case GDScriptParser::Node::TERNARY_OPERATOR:
			print_ternary_op(static_cast<GDScriptParser::TernaryOpNode *>(p_expression));
			break;
		case GDScriptParser::Node::TYPE_TEST:
			print_type_test(static_cast<GDScriptParser::TypeTestNode *>(p_expression));
			break;
		case GDScriptParser::Node::UNARY_OPERATOR:
			print_unary_op(static_cast<GDScriptParser::UnaryOpNode *>(p_expression));
			break;
		default:
			push_text(vformat("<unknown expression %d>", p_expression->type));
			break;
	}
}

void GDScriptTreePrinter::print_enum(GDScriptParser::EnumNode *p_enum) {
	push_text("Enum ");
	if (p_enum->identifier != nullptr) {
		print_identifier(p_enum->identifier);
	} else {
		push_text("<unnamed>");
	}

	push_line(" {");
	increase_indent();
	for (int i = 0; i < p_enum->values.size(); i++) {
		const GDScriptParser::EnumNode::Value &item = p_enum->values[i];
		print_identifier(item.identifier);
		push_text(" = ");
		push_text(itos(item.value));
		push_line(" ,");
	}
	decrease_indent();
	push_line("}");
}

void GDScriptTreePrinter::print_for(GDScriptParser::ForNode *p_for) {
	push_text("For ");
	print_identifier(p_for->variable);
	push_text(" IN ");
	print_expression(p_for->list);
	push_line(" :");

	increase_indent();

	print_suite(p_for->loop);

	decrease_indent();
}

void GDScriptTreePrinter::print_function(GDScriptParser::FunctionNode *p_function, const String &p_context) {
	for (const GDScriptParser::AnnotationNode *E : p_function->annotations) {
		print_annotation(E);
	}
	if (p_function->is_static) {
		push_text("Static ");
	}
	push_text(p_context);
	push_text(" ");
	if (p_function->identifier) {
		print_identifier(p_function->identifier);
	} else {
		push_text("<anonymous>");
	}
	push_text("( ");
	for (int i = 0; i < p_function->parameters.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_parameter(p_function->parameters[i]);
	}
	push_line(" ) :");
	increase_indent();
	print_suite(p_function->body);
	decrease_indent();
}

void GDScriptTreePrinter::print_get_node(GDScriptParser::GetNodeNode *p_get_node) {
	if (p_get_node->use_dollar) {
		push_text("$");
	}
	push_text(p_get_node->full_path);
}

void GDScriptTreePrinter::print_identifier(GDScriptParser::IdentifierNode *p_identifier) {
	if (p_identifier != nullptr) {
		push_text(p_identifier->name);
	} else {
		push_text("<invalid identifier>");
	}
}

void GDScriptTreePrinter::print_if(GDScriptParser::IfNode *p_if, bool p_is_elif) {
	if (p_is_elif) {
		push_text("Elif ");
	} else {
		push_text("If ");
	}
	print_expression(p_if->condition);
	push_line(" :");

	increase_indent();
	print_suite(p_if->true_block);
	decrease_indent();

	// FIXME: Properly detect "elif" blocks.
	if (p_if->false_block != nullptr) {
		push_line("Else :");
		increase_indent();
		print_suite(p_if->false_block);
		decrease_indent();
	}
}

void GDScriptTreePrinter::print_lambda(GDScriptParser::LambdaNode *p_lambda) {
	print_function(p_lambda->function, "Lambda");
	push_text("| captures [ ");
	for (int i = 0; i < p_lambda->captures.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		push_text(p_lambda->captures[i]->name.operator String());
	}
	push_line(" ]");
}

void GDScriptTreePrinter::print_literal(GDScriptParser::LiteralNode *p_literal) {
	// Prefix for string types.
	switch (p_literal->value.get_type()) {
		case Variant::NODE_PATH:
			push_text("^\"");
			break;
		case Variant::STRING:
			push_text("\"");
			break;
		case Variant::STRING_NAME:
			push_text("&\"");
			break;
		default:
			break;
	}
	push_text(p_literal->value);
	// Suffix for string types.
	switch (p_literal->value.get_type()) {
		case Variant::NODE_PATH:
		case Variant::STRING:
		case Variant::STRING_NAME:
			push_text("\"");
			break;
		default:
			break;
	}
}

void GDScriptTreePrinter::print_match(GDScriptParser::MatchNode *p_match) {
	push_text("Match ");
	print_expression(p_match->test);
	push_line(" :");

	increase_indent();
	for (int i = 0; i < p_match->branches.size(); i++) {
		print_match_branch(p_match->branches[i]);
	}
	decrease_indent();
}

void GDScriptTreePrinter::print_match_branch(GDScriptParser::MatchBranchNode *p_match_branch) {
	for (int i = 0; i < p_match_branch->patterns.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_match_pattern(p_match_branch->patterns[i]);
	}

	push_line(" :");

	increase_indent();
	print_suite(p_match_branch->block);
	decrease_indent();
}

void GDScriptTreePrinter::print_match_pattern(GDScriptParser::PatternNode *p_match_pattern) {
	switch (p_match_pattern->pattern_type) {
		case GDScriptParser::PatternNode::PT_LITERAL:
			print_literal(p_match_pattern->literal);
			break;
		case GDScriptParser::PatternNode::PT_WILDCARD:
			push_text("_");
			break;
		case GDScriptParser::PatternNode::PT_REST:
			push_text("..");
			break;
		case GDScriptParser::PatternNode::PT_BIND:
			push_text("Var ");
			print_identifier(p_match_pattern->bind);
			break;
		case GDScriptParser::PatternNode::PT_EXPRESSION:
			print_expression(p_match_pattern->expression);
			break;
		case GDScriptParser::PatternNode::PT_ARRAY:
			push_text("[ ");
			for (int i = 0; i < p_match_pattern->array.size(); i++) {
				if (i > 0) {
					push_text(" , ");
				}
				print_match_pattern(p_match_pattern->array[i]);
			}
			push_text(" ]");
			break;
		case GDScriptParser::PatternNode::PT_DICTIONARY:
			push_text("{ ");
			for (int i = 0; i < p_match_pattern->dictionary.size(); i++) {
				if (i > 0) {
					push_text(" , ");
				}
				if (p_match_pattern->dictionary[i].key != nullptr) {
					// Key can be null for rest pattern.
					print_expression(p_match_pattern->dictionary[i].key);
					push_text(" : ");
				}
				print_match_pattern(p_match_pattern->dictionary[i].value_pattern);
			}
			push_text(" }");
			break;
	}
}

void GDScriptTreePrinter::print_parameter(GDScriptParser::ParameterNode *p_parameter) {
	print_identifier(p_parameter->identifier);
	if (p_parameter->datatype_specifier != nullptr) {
		push_text(" : ");
		print_type(p_parameter->datatype_specifier);
	}
	if (p_parameter->initializer != nullptr) {
		push_text(" = ");
		print_expression(p_parameter->initializer);
	}
}

void GDScriptTreePrinter::print_preload(GDScriptParser::PreloadNode *p_preload) {
	push_text(R"(Preload ( ")");
	push_text(p_preload->resolved_path);
	push_text(R"(" )");
}

void GDScriptTreePrinter::print_return(GDScriptParser::ReturnNode *p_return) {
	push_text("Return");
	if (p_return->return_value != nullptr) {
		push_text(" ");
		print_expression(p_return->return_value);
	}
	push_line();
}

void GDScriptTreePrinter::print_self(GDScriptParser::SelfNode *p_self) {
	push_text("Self(");
	if (p_self->current_class->identifier != nullptr) {
		print_identifier(p_self->current_class->identifier);
	} else {
		push_text("<main class>");
	}
	push_text(")");
}

void GDScriptTreePrinter::print_signal(GDScriptParser::SignalNode *p_signal) {
	push_text("Signal ");
	print_identifier(p_signal->identifier);
	push_text("( ");
	for (int i = 0; i < p_signal->parameters.size(); i++) {
		print_parameter(p_signal->parameters[i]);
	}
	push_line(" )");
}

void GDScriptTreePrinter::print_subscript(GDScriptParser::SubscriptNode *p_subscript) {
	print_expression(p_subscript->base);
	if (p_subscript->is_attribute) {
		push_text(".");
		print_identifier(p_subscript->attribute);
	} else {
		push_text("[ ");
		print_expression(p_subscript->index);
		push_text(" ]");
	}
}

void GDScriptTreePrinter::print_statement(GDScriptParser::Node *p_statement) {
	switch (p_statement->type) {
		case GDScriptParser::Node::ASSERT:
			print_assert(static_cast<GDScriptParser::AssertNode *>(p_statement));
			break;
		case GDScriptParser::Node::VARIABLE:
			print_variable(static_cast<GDScriptParser::VariableNode *>(p_statement));
			break;
		case GDScriptParser::Node::CONSTANT:
			print_constant(static_cast<GDScriptParser::ConstantNode *>(p_statement));
			break;
		case GDScriptParser::Node::IF:
			print_if(static_cast<GDScriptParser::IfNode *>(p_statement));
			break;
		case GDScriptParser::Node::FOR:
			print_for(static_cast<GDScriptParser::ForNode *>(p_statement));
			break;
		case GDScriptParser::Node::WHILE:
			print_while(static_cast<GDScriptParser::WhileNode *>(p_statement));
			break;
		case GDScriptParser::Node::MATCH:
			print_match(static_cast<GDScriptParser::MatchNode *>(p_statement));
			break;
		case GDScriptParser::Node::RETURN:
			print_return(static_cast<GDScriptParser::ReturnNode *>(p_statement));
			break;
		case GDScriptParser::Node::BREAK:
			push_line("Break");
			break;
		case GDScriptParser::Node::CONTINUE:
			push_line("Continue");
			break;
		case GDScriptParser::Node::PASS:
			push_line("Pass");
			break;
		case GDScriptParser::Node::BREAKPOINT:
			push_line("Breakpoint");
			break;
		case GDScriptParser::Node::ASSIGNMENT:
			print_assignment(static_cast<GDScriptParser::AssignmentNode *>(p_statement));
			break;
		default:
			if (p_statement->is_expression()) {
				print_expression(static_cast<GDScriptParser::ExpressionNode *>(p_statement));
				push_line();
			} else {
				push_line(vformat("<unknown statement %d>", p_statement->type));
			}
			break;
	}
}

void GDScriptTreePrinter::print_suite(GDScriptParser::SuiteNode *p_suite) {
	for (int i = 0; i < p_suite->statements.size(); i++) {
		print_statement(p_suite->statements[i]);
	}
}

void GDScriptTreePrinter::print_ternary_op(GDScriptParser::TernaryOpNode *p_ternary_op) {
	// Surround in parenthesis for disambiguation.
	push_text("(");
	print_expression(p_ternary_op->true_expr);
	push_text(") IF (");
	print_expression(p_ternary_op->condition);
	push_text(") ELSE (");
	print_expression(p_ternary_op->false_expr);
	push_text(")");
}

void GDScriptTreePrinter::print_type(GDScriptParser::TypeNode *p_type) {
	if (p_type->type_chain.is_empty()) {
		push_text("Void");
	} else {
		for (int i = 0; i < p_type->type_chain.size(); i++) {
			if (i > 0) {
				push_text(".");
			}
			print_identifier(p_type->type_chain[i]);
		}
	}
}

void GDScriptTreePrinter::print_type_test(GDScriptParser::TypeTestNode *p_test) {
	print_expression(p_test->operand);
	push_text(" IS ");
	print_type(p_test->test_type);
}

void GDScriptTreePrinter::print_unary_op(GDScriptParser::UnaryOpNode *p_unary_op) {
	// Surround in parenthesis for disambiguation.
	push_text("(");
	switch (p_unary_op->operation) {
		case GDScriptParser::UnaryOpNode::OP_POSITIVE:
			push_text("+");
			break;
		case GDScriptParser::UnaryOpNode::OP_NEGATIVE:
			push_text("-");
			break;
		case GDScriptParser::UnaryOpNode::OP_LOGIC_NOT:
			push_text("NOT");
			break;
		case GDScriptParser::UnaryOpNode::OP_COMPLEMENT:
			push_text("~");
			break;
	}
	print_expression(p_unary_op->operand);
	// Surround in parenthesis for disambiguation.
	push_text(")");
}

void GDScriptTreePrinter::print_variable(GDScriptParser::VariableNode *p_variable) {
	for (const GDScriptParser::AnnotationNode *E : p_variable->annotations) {
		print_annotation(E);
	}

	if (p_variable->is_static) {
		push_text("Static ");
	}
	push_text("Variable ");
	print_identifier(p_variable->identifier);

	push_text(" : ");
	if (p_variable->datatype_specifier != nullptr) {
		print_type(p_variable->datatype_specifier);
	} else if (p_variable->infer_datatype) {
		push_text("<inferred type>");
	} else {
		push_text("Variant");
	}

	increase_indent();

	push_line();
	push_text("= ");
	if (p_variable->initializer == nullptr) {
		push_text("<default value>");
	} else {
		print_expression(p_variable->initializer);
	}
	push_line();

	if (p_variable->property != GDScriptParser::VariableNode::PROP_NONE) {
		if (p_variable->getter != nullptr) {
			push_text("Get");
			if (p_variable->property == GDScriptParser::VariableNode::PROP_INLINE) {
				push_line(":");
				increase_indent();
				print_suite(p_variable->getter->body);
				decrease_indent();
			} else {
				push_line(" =");
				increase_indent();
				print_identifier(p_variable->getter_pointer);
				push_line();
				decrease_indent();
			}
		}
		if (p_variable->setter != nullptr) {
			push_text("Set (");
			if (p_variable->property == GDScriptParser::VariableNode::PROP_INLINE) {
				if (p_variable->setter_parameter != nullptr) {
					print_identifier(p_variable->setter_parameter);
				} else {
					push_text("<missing>");
				}
				push_line("):");
				increase_indent();
				print_suite(p_variable->setter->body);
				decrease_indent();
			} else {
				push_line(" =");
				increase_indent();
				print_identifier(p_variable->setter_pointer);
				push_line();
				decrease_indent();
			}
		}
	}

	decrease_indent();
	push_line();
}

void GDScriptTreePrinter::print_while(GDScriptParser::WhileNode *p_while) {
	push_text("While ");
	print_expression(p_while->condition);
	push_line(" :");

	increase_indent();
	print_suite(p_while->loop);
	decrease_indent();
}

void GDScriptTreePrinter::print_tree(const GDScriptParser &p_parser) {
	ERR_FAIL_NULL_MSG(p_parser.get_tree(), "Parse the code before printing the parse tree.");

	if (p_parser.is_tool()) {
		push_line("@tool");
	}
	if (!p_parser.get_tree()->icon_path.is_empty()) {
		push_text(R"(@icon (")");
		push_text(p_parser.get_tree()->icon_path);
		push_line("\")");
	}
	print_class(p_parser.get_tree());

	print_line(String(printed));
}
#endif // DEBUG_ENABLED

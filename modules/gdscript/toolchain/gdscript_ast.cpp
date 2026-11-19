/**************************************************************************/
/*  gdscript_ast.cpp                                                      */
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

#include "gdscript_ast.h"

#include "../gdscript.h"
#include "../gdscript_cache.h"
#include "../gdscript_parser.h"

#include "core/object/class_db.h"

bool GDScriptAST::PatternNode::has_bind(const StringName &p_name) {
	return binds.has(p_name);
}

GDScriptAST::IdentifierNode *GDScriptAST::PatternNode::get_bind(const StringName &p_name) {
	return binds[p_name];
}

bool GDScriptAST::SuiteNode::has_local(const StringName &p_name) const {
	if (locals_indices.has(p_name)) {
		return true;
	}
	if (parent_block != nullptr) {
		return parent_block->has_local(p_name);
	}
	return false;
}

const GDScriptAST::SuiteNode::Local &GDScriptAST::SuiteNode::get_local(const StringName &p_name) const {
	if (locals_indices.has(p_name)) {
		return locals[locals_indices[p_name]];
	}
	if (parent_block != nullptr) {
		return parent_block->get_local(p_name);
	}
	return empty;
}

bool GDScriptAST::AnnotationNode::apply(GDScriptParser *p_this, Node *p_target, ClassNode *p_class) {
	if (is_applied) {
		return true;
	}
	is_applied = true;
	return (p_this->*(p_this->valid_annotations[name].apply))(this, p_target, p_class);
}

GDScriptAST::DataType GDScriptAST::SuiteNode::Local::get_datatype() const {
	switch (type) {
		case CONSTANT:
			return constant->type_constraint;
		case VARIABLE:
			return variable->type_constraint;
		case PARAMETER:
			return parameter->type_constraint;
		case FOR_VARIABLE:
		case PATTERN_BIND:
			return bind->type_constraint;
		case UNDEFINED:
			return DataType();
	}
	return DataType();
}

String GDScriptAST::SuiteNode::Local::get_name() const {
	switch (type) {
		case SuiteNode::Local::PARAMETER:
			return "parameter";
		case SuiteNode::Local::CONSTANT:
			return "constant";
		case SuiteNode::Local::VARIABLE:
			return "variable";
		case SuiteNode::Local::FOR_VARIABLE:
			return "for loop iterator";
		case SuiteNode::Local::PATTERN_BIND:
			return "pattern bind";
		case SuiteNode::Local::UNDEFINED:
			return "<undefined>";
		default:
			return String();
	}
}

String GDScriptAST::DataType::to_string() const {
	switch (kind) {
		case VARIANT:
			return "Variant";
		case BUILTIN:
			if (builtin_type == Variant::NIL) {
				return "null";
			}
			if (builtin_type == Variant::ARRAY && has_container_element_type(0)) {
				return vformat("Array[%s]", get_container_element_type(0).to_string());
			}
			if (builtin_type == Variant::DICTIONARY && has_container_element_types()) {
				return vformat("Dictionary[%s, %s]", get_container_element_type_or_variant(0).to_string(), get_container_element_type_or_variant(1).to_string());
			}
			return Variant::get_type_name(builtin_type);
		case NATIVE:
			if (is_meta_type) {
				return GDScriptNativeClass::get_class_static();
			}
			return native_type.string();
		case CLASS:
			if (class_type->identifier != nullptr) {
				return class_type->identifier->name.string();
			}
			return class_type->fqcn;
		case SCRIPT: {
			if (is_meta_type) {
				return script_type.is_valid() ? script_type->get_class_name().string() : "";
			}
			String name = script_type.is_valid() ? script_type->get_name() : "";
			if (!name.is_empty()) {
				return name;
			}
			name = script_path;
			if (!name.is_empty()) {
				return name;
			}
			return native_type.string();
		}
		case ENUM: {
			// native_type contains either the native class defining the enum
			// or the fully qualified class name of the script defining the enum
			return String(native_type).get_file(); // Remove path, keep filename
		}
		case RESOLVING:
		case UNRESOLVED:
			return "<unresolved type>";
	}

	ERR_FAIL_V_MSG("<unresolved type>", "Kind set outside the enum range.");
}

String GDScriptAST::DataType::to_property_info_hint_string() const {
	switch (kind) {
		case BUILTIN:
			return Variant::get_type_name(builtin_type);
		case NATIVE:
			return native_type;
		case SCRIPT:
			if (script_type.is_valid() && script_type->get_global_name() != StringName()) {
				return script_type->get_global_name();
			} else {
				return native_type;
			}
		case CLASS:
			if (class_type != nullptr && class_type->get_global_name() != StringName()) {
				return class_type->get_global_name();
			} else {
				return native_type;
			}
		case ENUM:
			return String(native_type).replace("::", ".");
		case VARIANT:
			return "Variant";
		case RESOLVING:
		case UNRESOLVED:
			break;
	}
	ERR_FAIL_V_MSG("Variant", "GDScript bug: Unexpected type kind.");
}

PropertyInfo GDScriptAST::DataType::to_property_info(const String &p_name) const {
	PropertyInfo result;
	result.name = p_name;
	result.usage = PROPERTY_USAGE_NONE;

	if (!is_hard_type()) {
		result.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
		return result;
	}

	switch (kind) {
		case BUILTIN:
			result.type = builtin_type;
			if (builtin_type == Variant::ARRAY && has_container_element_type(0)) {
				const DataType elem_type = get_container_element_type(0);
				if (elem_type.is_variant()) {
					break;
				}

				result.hint = PROPERTY_HINT_ARRAY_TYPE;
				result.hint_string = elem_type.to_property_info_hint_string();
			} else if (builtin_type == Variant::DICTIONARY && has_container_element_types()) {
				const DataType key_type = get_container_element_type_or_variant(0);
				const DataType value_type = get_container_element_type_or_variant(1);
				if (key_type.is_variant() && value_type.is_variant()) {
					break;
				}

				result.hint = PROPERTY_HINT_DICTIONARY_TYPE;
				result.hint_string = key_type.to_property_info_hint_string() + ";" + value_type.to_property_info_hint_string();
			}
			break;
		case NATIVE:
			result.type = Variant::OBJECT;
			if (is_meta_type) {
				result.class_name = GDScriptNativeClass::get_class_static();
			} else {
				result.class_name = native_type;
			}
			break;
		case SCRIPT:
			result.type = Variant::OBJECT;
			if (is_meta_type) {
				result.class_name = script_type.is_valid() ? script_type->get_class_name() : Script::get_class_static();
			} else if (script_type.is_valid() && script_type->get_global_name() != StringName()) {
				result.class_name = script_type->get_global_name();
			} else {
				result.class_name = native_type;
			}
			break;
		case CLASS:
			result.type = Variant::OBJECT;
			if (is_meta_type) {
				result.class_name = GDScript::get_class_static();
			} else if (class_type != nullptr && class_type->get_global_name() != StringName()) {
				result.class_name = class_type->get_global_name();
			} else {
				result.class_name = native_type;
			}
			break;
		case ENUM:
			if (is_meta_type) {
				result.type = Variant::DICTIONARY;
			} else {
				result.type = Variant::INT;
				result.usage |= PROPERTY_USAGE_CLASS_IS_ENUM;
				result.class_name = String(native_type).replace("::", ".");
			}
			break;
		case VARIANT:
		case RESOLVING:
		case UNRESOLVED:
			result.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
			break;
	}

	return result;
}

static Variant::Type _variant_type_to_typed_array_element_type(Variant::Type p_type) {
	switch (p_type) {
		case Variant::PACKED_BYTE_ARRAY:
		case Variant::PACKED_INT32_ARRAY:
		case Variant::PACKED_INT64_ARRAY:
			return Variant::INT;
		case Variant::PACKED_FLOAT32_ARRAY:
		case Variant::PACKED_FLOAT64_ARRAY:
			return Variant::FLOAT;
		case Variant::PACKED_STRING_ARRAY:
			return Variant::STRING;
		case Variant::PACKED_VECTOR2_ARRAY:
			return Variant::VECTOR2;
		case Variant::PACKED_VECTOR3_ARRAY:
			return Variant::VECTOR3;
		case Variant::PACKED_COLOR_ARRAY:
			return Variant::COLOR;
		case Variant::PACKED_VECTOR4_ARRAY:
			return Variant::VECTOR4;
		default:
			return Variant::NIL;
	}
}

bool GDScriptAST::DataType::is_typed_container_type() const {
	return kind == GDScriptAST::DataType::BUILTIN && _variant_type_to_typed_array_element_type(builtin_type) != Variant::NIL;
}

GDScriptAST::DataType GDScriptAST::DataType::get_typed_container_type() const {
	GDScriptAST::DataType type;
	type.kind = GDScriptAST::DataType::BUILTIN;
	type.builtin_type = _variant_type_to_typed_array_element_type(builtin_type);
	return type;
}

bool GDScriptAST::DataType::can_reference(const GDScriptAST::DataType &p_other) const {
	if (p_other.is_meta_type) {
		return false;
	} else if (builtin_type != p_other.builtin_type) {
		return false;
	} else if (builtin_type != Variant::OBJECT) {
		return true;
	}

	if (native_type == StringName()) {
		return true;
	} else if (p_other.native_type == StringName()) {
		return false;
	} else if (native_type != p_other.native_type && !ClassDB::is_parent_class(p_other.native_type, native_type)) {
		return false;
	}

	Ref<Script> script = script_type;
	if (kind == GDScriptAST::DataType::CLASS && script.is_null()) {
		Error err = OK;
		Ref<GDScript> scr = GDScriptCache::get_shallow_script(script_path, err);
		ERR_FAIL_COND_V_MSG(err, false, vformat(R"(Error while getting cache for script "%s".)", script_path));
		script.reference_ptr(scr->find_class(class_type->fqcn));
	}

	Ref<Script> script_other = p_other.script_type;
	if (p_other.kind == GDScriptAST::DataType::CLASS && script_other.is_null()) {
		Error err = OK;
		Ref<GDScript> scr = GDScriptCache::get_shallow_script(p_other.script_path, err);
		ERR_FAIL_COND_V_MSG(err, false, vformat(R"(Error while getting cache for script "%s".)", p_other.script_path));
		script_other.reference_ptr(scr->find_class(p_other.class_type->fqcn));
	}

	if (script.is_null()) {
		return true;
	} else if (script_other.is_null()) {
		return false;
	} else if (script != script_other && !script_other->inherits_script(script)) {
		return false;
	}

	return true;
}

bool GDScriptAST::AnnotationNode::applies_to(uint32_t p_target_kinds) const {
	return (info->target_kind & p_target_kinds) > 0;
}

/*---------- PRETTY PRINT FOR DEBUG ----------*/

#ifdef DEBUG_ENABLED

void TreePrinter::increase_indent() {
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

void TreePrinter::decrease_indent() {
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

void TreePrinter::push_line(const String &p_line) {
	if (!p_line.is_empty()) {
		push_text(p_line);
	}
	printed += "\n";
	pending_indent = true;
}

void TreePrinter::push_text(const String &p_text) {
	if (pending_indent) {
		printed += indent;
		pending_indent = false;
	}
	printed += p_text;
}

void TreePrinter::print_annotation(const GDScriptAST::AnnotationNode *p_annotation) {
	push_text(p_annotation->name);
	push_text(" (");
	for (uint32_t i = 0; i < p_annotation->arguments.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_expression(p_annotation->arguments[i]);
	}
	push_line(")");
}

void TreePrinter::print_array(const GDScriptAST::ArrayNode *p_array) {
	push_text("[ ");
	for (uint32_t i = 0; i < p_array->elements.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_expression(p_array->elements[i]);
	}
	push_text(" ]");
}

void TreePrinter::print_assert(const GDScriptAST::AssertNode *p_assert) {
	push_text("Assert ( ");
	print_expression(p_assert->condition);
	push_line(" )");
}

void TreePrinter::print_assignment(const GDScriptAST::AssignmentNode *p_assignment) {
	switch (p_assignment->assignee->type) {
		case GDScriptAST::Node::IDENTIFIER:
			print_identifier(static_cast<GDScriptAST::IdentifierNode *>(p_assignment->assignee));
			break;
		case GDScriptAST::Node::SUBSCRIPT:
			print_subscript(static_cast<GDScriptAST::SubscriptNode *>(p_assignment->assignee));
			break;
		default:
			break; // Unreachable.
	}

	push_text(" ");
	switch (p_assignment->operation) {
		case GDScriptAST::AssignmentNode::OP_ADDITION:
			push_text("+");
			break;
		case GDScriptAST::AssignmentNode::OP_SUBTRACTION:
			push_text("-");
			break;
		case GDScriptAST::AssignmentNode::OP_MULTIPLICATION:
			push_text("*");
			break;
		case GDScriptAST::AssignmentNode::OP_DIVISION:
			push_text("/");
			break;
		case GDScriptAST::AssignmentNode::OP_MODULO:
			push_text("%");
			break;
		case GDScriptAST::AssignmentNode::OP_POWER:
			push_text("**");
			break;
		case GDScriptAST::AssignmentNode::OP_BIT_SHIFT_LEFT:
			push_text("<<");
			break;
		case GDScriptAST::AssignmentNode::OP_BIT_SHIFT_RIGHT:
			push_text(">>");
			break;
		case GDScriptAST::AssignmentNode::OP_BIT_AND:
			push_text("&");
			break;
		case GDScriptAST::AssignmentNode::OP_BIT_OR:
			push_text("|");
			break;
		case GDScriptAST::AssignmentNode::OP_BIT_XOR:
			push_text("^");
			break;
		case GDScriptAST::AssignmentNode::OP_NONE:
			break;
	}
	push_text("= ");
	print_expression(p_assignment->assigned_value);
	push_line();
}

void TreePrinter::print_await(const GDScriptAST::AwaitNode *p_await) {
	push_text("Await ");
	print_expression(p_await->to_await);
}

void TreePrinter::print_binary_op(const GDScriptAST::BinaryOpNode *p_binary_op) {
	// Surround in parenthesis for disambiguation.
	push_text("(");
	print_expression(p_binary_op->left_operand);
	switch (p_binary_op->operation) {
		case GDScriptAST::BinaryOpNode::OP_ADDITION:
			push_text(" + ");
			break;
		case GDScriptAST::BinaryOpNode::OP_SUBTRACTION:
			push_text(" - ");
			break;
		case GDScriptAST::BinaryOpNode::OP_MULTIPLICATION:
			push_text(" * ");
			break;
		case GDScriptAST::BinaryOpNode::OP_DIVISION:
			push_text(" / ");
			break;
		case GDScriptAST::BinaryOpNode::OP_MODULO:
			push_text(" % ");
			break;
		case GDScriptAST::BinaryOpNode::OP_POWER:
			push_text(" ** ");
			break;
		case GDScriptAST::BinaryOpNode::OP_BIT_LEFT_SHIFT:
			push_text(" << ");
			break;
		case GDScriptAST::BinaryOpNode::OP_BIT_RIGHT_SHIFT:
			push_text(" >> ");
			break;
		case GDScriptAST::BinaryOpNode::OP_BIT_AND:
			push_text(" & ");
			break;
		case GDScriptAST::BinaryOpNode::OP_BIT_OR:
			push_text(" | ");
			break;
		case GDScriptAST::BinaryOpNode::OP_BIT_XOR:
			push_text(" ^ ");
			break;
		case GDScriptAST::BinaryOpNode::OP_LOGIC_AND:
			push_text(" AND ");
			break;
		case GDScriptAST::BinaryOpNode::OP_LOGIC_OR:
			push_text(" OR ");
			break;
		case GDScriptAST::BinaryOpNode::OP_CONTENT_TEST:
			push_text(" IN ");
			break;
		case GDScriptAST::BinaryOpNode::OP_COMP_EQUAL:
			push_text(" == ");
			break;
		case GDScriptAST::BinaryOpNode::OP_COMP_NOT_EQUAL:
			push_text(" != ");
			break;
		case GDScriptAST::BinaryOpNode::OP_COMP_LESS:
			push_text(" < ");
			break;
		case GDScriptAST::BinaryOpNode::OP_COMP_LESS_EQUAL:
			push_text(" <= ");
			break;
		case GDScriptAST::BinaryOpNode::OP_COMP_GREATER:
			push_text(" > ");
			break;
		case GDScriptAST::BinaryOpNode::OP_COMP_GREATER_EQUAL:
			push_text(" >= ");
			break;
	}
	print_expression(p_binary_op->right_operand);
	// Surround in parenthesis for disambiguation.
	push_text(")");
}

void TreePrinter::print_call(const GDScriptAST::CallNode *p_call) {
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
	for (uint32_t i = 0; i < p_call->arguments.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		print_expression(p_call->arguments[i]);
	}
	push_text(" )");
}

void TreePrinter::print_cast(const GDScriptAST::CastNode *p_cast) {
	print_expression(p_cast->operand);
	push_text(" AS ");
	print_type(p_cast->cast_type);
}

void TreePrinter::print_class(const GDScriptAST::ClassNode *p_class) {
	for (const GDScriptAST::AnnotationNode *E : p_class->annotations) {
		print_annotation(E);
	}
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

	for (const GDScriptAST::ClassNode::Member &m : p_class->members) {
		switch (m.type) {
			case GDScriptAST::ClassNode::Member::CLASS:
				print_class(m.m_class);
				break;
			case GDScriptAST::ClassNode::Member::VARIABLE:
				print_variable(m.variable);
				break;
			case GDScriptAST::ClassNode::Member::CONSTANT:
				print_constant(m.constant);
				break;
			case GDScriptAST::ClassNode::Member::SIGNAL:
				print_signal(m.signal);
				break;
			case GDScriptAST::ClassNode::Member::FUNCTION:
				print_function(m.function);
				break;
			case GDScriptAST::ClassNode::Member::ENUM:
				print_enum(m.m_enum);
				break;
			case GDScriptAST::ClassNode::Member::ENUM_VALUE:
				break; // Nothing. Will be printed by enum.
			case GDScriptAST::ClassNode::Member::GROUP:
				break; // Nothing. Groups are only used by inspector.
			case GDScriptAST::ClassNode::Member::UNDEFINED:
				push_line("<unknown member>");
				break;
		}
	}

	decrease_indent();
}

void TreePrinter::print_constant(const GDScriptAST::ConstantNode *p_constant) {
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

void TreePrinter::print_dictionary(const GDScriptAST::DictionaryNode *p_dictionary) {
	push_line("{");
	increase_indent();
	for (const GDScriptAST::DictionaryNode::Pair &element : p_dictionary->elements) {
		print_expression(element.key);
		if (p_dictionary->style == GDScriptAST::DictionaryNode::PYTHON_DICT) {
			push_text(" : ");
		} else {
			push_text(" = ");
		}
		print_expression(element.value);
		push_line(" ,");
	}
	decrease_indent();
	push_text("}");
}

void TreePrinter::print_expression(const GDScriptAST::ExpressionNode *p_expression) {
	if (p_expression == nullptr) {
		push_text("<invalid expression>");
		return;
	}
	switch (p_expression->type) {
		case GDScriptAST::Node::ARRAY:
			print_array(static_cast<const GDScriptAST::ArrayNode *>(p_expression));
			break;
		case GDScriptAST::Node::ASSIGNMENT:
			print_assignment(static_cast<const GDScriptAST::AssignmentNode *>(p_expression));
			break;
		case GDScriptAST::Node::AWAIT:
			print_await(static_cast<const GDScriptAST::AwaitNode *>(p_expression));
			break;
		case GDScriptAST::Node::BINARY_OPERATOR:
			print_binary_op(static_cast<const GDScriptAST::BinaryOpNode *>(p_expression));
			break;
		case GDScriptAST::Node::CALL:
			print_call(static_cast<const GDScriptAST::CallNode *>(p_expression));
			break;
		case GDScriptAST::Node::CAST:
			print_cast(static_cast<const GDScriptAST::CastNode *>(p_expression));
			break;
		case GDScriptAST::Node::DICTIONARY:
			print_dictionary(static_cast<const GDScriptAST::DictionaryNode *>(p_expression));
			break;
		case GDScriptAST::Node::GET_NODE:
			print_get_node(static_cast<const GDScriptAST::GetNodeNode *>(p_expression));
			break;
		case GDScriptAST::Node::IDENTIFIER:
			print_identifier(static_cast<const GDScriptAST::IdentifierNode *>(p_expression));
			break;
		case GDScriptAST::Node::LAMBDA:
			print_lambda(static_cast<const GDScriptAST::LambdaNode *>(p_expression));
			break;
		case GDScriptAST::Node::LITERAL:
			print_literal(static_cast<const GDScriptAST::LiteralNode *>(p_expression));
			break;
		case GDScriptAST::Node::PRELOAD:
			print_preload(static_cast<const GDScriptAST::PreloadNode *>(p_expression));
			break;
		case GDScriptAST::Node::SELF:
			print_self(static_cast<const GDScriptAST::SelfNode *>(p_expression));
			break;
		case GDScriptAST::Node::SUBSCRIPT:
			print_subscript(static_cast<const GDScriptAST::SubscriptNode *>(p_expression));
			break;
		case GDScriptAST::Node::TERNARY_OPERATOR:
			print_ternary_op(static_cast<const GDScriptAST::TernaryOpNode *>(p_expression));
			break;
		case GDScriptAST::Node::TYPE_TEST:
			print_type_test(static_cast<const GDScriptAST::TypeTestNode *>(p_expression));
			break;
		case GDScriptAST::Node::UNARY_OPERATOR:
			print_unary_op(static_cast<const GDScriptAST::UnaryOpNode *>(p_expression));
			break;
		default:
			push_text(vformat("<unknown expression %d>", p_expression->type));
			break;
	}
}

void TreePrinter::print_enum(const GDScriptAST::EnumNode *p_enum) {
	push_text("Enum ");
	if (p_enum->identifier != nullptr) {
		print_identifier(p_enum->identifier);
	} else {
		push_text("<unnamed>");
	}

	push_line(" {");
	increase_indent();
	for (const GDScriptAST::EnumNode::Value &item : p_enum->values) {
		print_identifier(item.identifier);
		push_text(" = ");
		push_text(itos(item.value));
		push_line(" ,");
	}
	decrease_indent();
	push_line("}");
}

void TreePrinter::print_for(const GDScriptAST::ForNode *p_for) {
	push_text("For ");
	print_identifier(p_for->variable);
	push_text(" IN ");
	print_expression(p_for->list);
	push_line(" :");

	increase_indent();

	print_suite(p_for->loop);

	decrease_indent();
}

void TreePrinter::print_function(const GDScriptAST::FunctionNode *p_function, const String &p_context) {
	for (const GDScriptAST::AnnotationNode *E : p_function->annotations) {
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
	for (uint32_t i = 0; i < p_function->parameters.size(); i++) {
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

void TreePrinter::print_get_node(const GDScriptAST::GetNodeNode *p_get_node) {
	if (p_get_node->use_dollar) {
		push_text("$");
	}
	push_text(p_get_node->full_path);
}

void TreePrinter::print_identifier(const GDScriptAST::IdentifierNode *p_identifier) {
	if (p_identifier != nullptr) {
		push_text(p_identifier->name);
	} else {
		push_text("<invalid identifier>");
	}
}

void TreePrinter::print_if(const GDScriptAST::IfNode *p_if, bool p_is_elif) {
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

void TreePrinter::print_lambda(const GDScriptAST::LambdaNode *p_lambda) {
	print_function(p_lambda->function, "Lambda");
	push_text("| captures [ ");
	for (uint32_t i = 0; i < p_lambda->captures.size(); i++) {
		if (i > 0) {
			push_text(" , ");
		}
		push_text(p_lambda->captures[i]->name.string());
	}
	push_line(" ]");
}

void TreePrinter::print_literal(const GDScriptAST::LiteralNode *p_literal) {
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

void TreePrinter::print_match(const GDScriptAST::MatchNode *p_match) {
	push_text("Match ");
	print_expression(p_match->test);
	push_line(" :");

	increase_indent();
	for (const GDScriptAST::MatchBranchNode *branch : p_match->branches) {
		print_match_branch(branch);
	}
	decrease_indent();
}

void TreePrinter::print_match_branch(const GDScriptAST::MatchBranchNode *p_match_branch) {
	for (uint32_t i = 0; i < p_match_branch->patterns.size(); i++) {
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

void TreePrinter::print_match_pattern(const GDScriptAST::PatternNode *p_match_pattern) {
	switch (p_match_pattern->pattern_type) {
		case GDScriptAST::PatternNode::PT_LITERAL:
			print_literal(p_match_pattern->literal);
			break;
		case GDScriptAST::PatternNode::PT_WILDCARD:
			push_text("_");
			break;
		case GDScriptAST::PatternNode::PT_REST:
			push_text("..");
			break;
		case GDScriptAST::PatternNode::PT_BIND:
			push_text("Var ");
			print_identifier(p_match_pattern->bind);
			break;
		case GDScriptAST::PatternNode::PT_EXPRESSION:
			print_expression(p_match_pattern->expression);
			break;
		case GDScriptAST::PatternNode::PT_ARRAY:
			push_text("[ ");
			for (uint32_t i = 0; i < p_match_pattern->array.size(); i++) {
				if (i > 0) {
					push_text(" , ");
				}
				print_match_pattern(p_match_pattern->array[i]);
			}
			push_text(" ]");
			break;
		case GDScriptAST::PatternNode::PT_DICTIONARY:
			push_text("{ ");
			for (uint32_t i = 0; i < p_match_pattern->dictionary.size(); i++) {
				if (i > 0) {
					push_text(" , ");
				}
				if (p_match_pattern->dictionary[i].key != nullptr) {
					// Key can be null for rest pattern.
					print_expression(p_match_pattern->dictionary[i].key);
					if (p_match_pattern->dictionary[i].value_pattern != nullptr) {
						// Value can be null when only matching key.
						push_text(" : ");
						print_match_pattern(p_match_pattern->dictionary[i].value_pattern);
					}
				}
			}
			push_text(" }");
			break;
	}
}

void TreePrinter::print_parameter(const GDScriptAST::ParameterNode *p_parameter) {
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

void TreePrinter::print_preload(const GDScriptAST::PreloadNode *p_preload) {
	push_text(R"(Preload ( ")");
	push_text(p_preload->resolved_path);
	push_text(R"(" )");
}

void TreePrinter::print_return(const GDScriptAST::ReturnNode *p_return) {
	push_text("Return");
	if (p_return->return_value != nullptr) {
		push_text(" ");
		print_expression(p_return->return_value);
	}
	push_line();
}

void TreePrinter::print_self(const GDScriptAST::SelfNode *p_self) {
	push_text("Self(");
	if (p_self->current_class->identifier != nullptr) {
		print_identifier(p_self->current_class->identifier);
	} else {
		push_text("<main class>");
	}
	push_text(")");
}

void TreePrinter::print_signal(const GDScriptAST::SignalNode *p_signal) {
	push_text("Signal ");
	print_identifier(p_signal->identifier);
	push_text("( ");
	for (const GDScriptAST::ParameterNode *param : p_signal->parameters) {
		print_parameter(param);
	}
	push_line(" )");
}

void TreePrinter::print_subscript(const GDScriptAST::SubscriptNode *p_subscript) {
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

void TreePrinter::print_statement(const GDScriptAST::Node *p_statement) {
	switch (p_statement->type) {
		case GDScriptAST::Node::ASSERT:
			print_assert(static_cast<const GDScriptAST::AssertNode *>(p_statement));
			break;
		case GDScriptAST::Node::VARIABLE:
			print_variable(static_cast<const GDScriptAST::VariableNode *>(p_statement));
			break;
		case GDScriptAST::Node::CONSTANT:
			print_constant(static_cast<const GDScriptAST::ConstantNode *>(p_statement));
			break;
		case GDScriptAST::Node::IF:
			print_if(static_cast<const GDScriptAST::IfNode *>(p_statement));
			break;
		case GDScriptAST::Node::FOR:
			print_for(static_cast<const GDScriptAST::ForNode *>(p_statement));
			break;
		case GDScriptAST::Node::WHILE:
			print_while(static_cast<const GDScriptAST::WhileNode *>(p_statement));
			break;
		case GDScriptAST::Node::MATCH:
			print_match(static_cast<const GDScriptAST::MatchNode *>(p_statement));
			break;
		case GDScriptAST::Node::RETURN:
			print_return(static_cast<const GDScriptAST::ReturnNode *>(p_statement));
			break;
		case GDScriptAST::Node::BREAK:
			push_line("Break");
			break;
		case GDScriptAST::Node::CONTINUE:
			push_line("Continue");
			break;
		case GDScriptAST::Node::PASS:
			push_line("Pass");
			break;
		case GDScriptAST::Node::BREAKPOINT:
			push_line("Breakpoint");
			break;
		case GDScriptAST::Node::ASSIGNMENT:
			print_assignment(static_cast<const GDScriptAST::AssignmentNode *>(p_statement));
			break;
		default:
			if (p_statement->is_expression()) {
				print_expression(static_cast<const GDScriptAST::ExpressionNode *>(p_statement));
				push_line();
			} else {
				push_line(vformat("<unknown statement %d>", p_statement->type));
			}
			break;
	}
}

void TreePrinter::print_suite(const GDScriptAST::SuiteNode *p_suite) {
	for (const GDScriptAST::Node *stmt : p_suite->statements) {
		print_statement(stmt);
	}
}

void TreePrinter::print_ternary_op(const GDScriptAST::TernaryOpNode *p_ternary_op) {
	// Surround in parenthesis for disambiguation.
	push_text("(");
	print_expression(p_ternary_op->true_expr);
	push_text(") IF (");
	print_expression(p_ternary_op->condition);
	push_text(") ELSE (");
	print_expression(p_ternary_op->false_expr);
	push_text(")");
}

void TreePrinter::print_type(const GDScriptAST::TypeNode *p_type) {
	if (p_type->type_chain.is_empty()) {
		push_text("Void");
	} else {
		for (uint32_t i = 0; i < p_type->type_chain.size(); i++) {
			if (i > 0) {
				push_text(".");
			}
			print_identifier(p_type->type_chain[i]);
		}
	}
}

void TreePrinter::print_type_test(const GDScriptAST::TypeTestNode *p_test) {
	print_expression(p_test->operand);
	push_text(" IS ");
	print_type(p_test->test_type);
}

void TreePrinter::print_unary_op(const GDScriptAST::UnaryOpNode *p_unary_op) {
	// Surround in parenthesis for disambiguation.
	push_text("(");
	switch (p_unary_op->operation) {
		case GDScriptAST::UnaryOpNode::OP_POSITIVE:
			push_text("+");
			break;
		case GDScriptAST::UnaryOpNode::OP_NEGATIVE:
			push_text("-");
			break;
		case GDScriptAST::UnaryOpNode::OP_LOGIC_NOT:
			push_text("NOT");
			break;
		case GDScriptAST::UnaryOpNode::OP_COMPLEMENT:
			push_text("~");
			break;
	}
	print_expression(p_unary_op->operand);
	// Surround in parenthesis for disambiguation.
	push_text(")");
}

void TreePrinter::print_variable(const GDScriptAST::VariableNode *p_variable) {
	for (const GDScriptAST::AnnotationNode *E : p_variable->annotations) {
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

	if (p_variable->property != GDScriptAST::VariableNode::PROP_NONE) {
		if (p_variable->getter != nullptr) {
			push_text("Get");
			if (p_variable->property == GDScriptAST::VariableNode::PROP_INLINE) {
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
			if (p_variable->property == GDScriptAST::VariableNode::PROP_INLINE) {
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

void TreePrinter::print_while(const GDScriptAST::WhileNode *p_while) {
	push_text("While ");
	print_expression(p_while->condition);
	push_line(" :");

	increase_indent();
	print_suite(p_while->loop);
	decrease_indent();
}

void TreePrinter::print_tree(const GDScriptAST::ClassNode *class_tree, const bool is_tool) {
	ERR_FAIL_NULL_MSG(class_tree, "Parse the code before printing the parse tree.");

	if (is_tool) {
		push_line("@tool");
	}
	if (!class_tree->icon_path.is_empty()) {
		push_text(R"(@icon (")");
		push_text(class_tree->icon_path);
		push_line("\")");
	}
	print_class(class_tree);

	print_line(String(printed));
}

#endif // DEBUG_ENABLED

/**************************************************************************/
/*  gdscript_tree_converter.cpp                                           */
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

#ifdef TOOLS_ENABLED

#include "gdscript_tree_converter.h"
#include "core/variant/dictionary.h"

Dictionary GDScriptTreeConverter::to_dictionary(const PropertyInfo &self) {
	Dictionary dict;
	dict["type"] = self.type;
	dict["name"] = (String)self.name;
	dict["class_name"] = self.class_name;
	dict["hint"] = self.hint;
	dict["hint_string"] = self.hint_string;
	dict["usage"] = self.usage;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const MethodInfo &self) {
	Dictionary dict;

	dict["name"] = self.name;
	dict["return_val"] = to_dictionary(self.return_val);
	dict["flags"] = self.flags;
	dict["id"] = self.id;

	Array dict_arguments;
	dict["arguments"] = dict_arguments;
	for (const PropertyInfo &argument : self.arguments) {
		dict_arguments.push_back(to_dictionary(argument));
	}

	Array dict_default_arguments;
	dict["default_arguments"] = dict_default_arguments;
	for (const Variant &default_argument : self.default_arguments) {
		dict_default_arguments.push_back(default_argument);
	}

	dict["return_val_metadata"] = self.return_val_metadata;

	Array dict_arguments_metadata;
	dict["arguments_metadata"] = dict_arguments_metadata;
	for (const int &argument_metadata : self.arguments_metadata) {
		dict_arguments_metadata.push_back(argument_metadata);
	}

	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::DataType &self) {
	Dictionary dict;

	Array dict_container_element_types;
	dict["container_element_types"] = dict_container_element_types;
	for (const GDScriptParser::DataType &container_element_type : self.container_element_types) {
		dict_container_element_types.append(to_dictionary(container_element_type));
	}

	dict["kind"] = self.kind;
	dict["type_source"] = self.type_source;

	dict["is_read_only"] = self.is_read_only;
	dict["is_constant"] = self.is_constant;
	dict["is_meta_type"] = self.is_meta_type;
	dict["is_pseudo_type"] = self.is_pseudo_type;
	dict["is_coroutine"] = self.is_coroutine;

	dict["builtin_type"] = self.builtin_type;

	dict["native_type"] = (String)self.native_type;
	dict["enum_type"] = (String)self.enum_type;

	//dict["script_type"] = self.script_type;
	dict["script_path"] = self.script_path;

	////if(self.class_type != nullptr)
	////dict["class_type"] = to_dictionary(*self.class_type);

	dict["method_info"] = to_dictionary(self.method_info);

	Dictionary dict_enum_values;
	dict["enum_values"] = dict_enum_values;
	for (const KeyValue<StringName, int64_t> &kv : self.enum_values) {
		dict_enum_values[(String)kv.key] = kv.value;
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::ClassDocData &self) {
	Dictionary dict;
	dict["brief"] = self.brief;
	dict["description"] = self.description;

	Array dict_tutorials;
	dict["tutorials"] = dict_tutorials;
	for (const Pair<String, String> &p : self.tutorials) {
		Array dict_p;
		dict_p.push_back(p.first);
		dict_p.push_back(p.second);
		dict_tutorials.push_back(dict_p);
	}

	dict["is_deprecated"] = self.is_deprecated;
	dict["deprecated_message"] = self.deprecated_message;
	dict["is_experimental"] = self.is_experimental;
	dict["experimental_message"] = self.experimental_message;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::MemberDocData &self) {
	Dictionary dict;
	dict["description"] = self.description;
	dict["is_deprecated"] = self.is_deprecated;
	dict["deprecated_message"] = self.deprecated_message;
	dict["is_experimental"] = self.is_experimental;
	dict["experimental_message"] = self.experimental_message;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::Node &self) {
	switch (self.type) {
		case GDScriptParser::Node::Type::ANNOTATION:
			return to_dictionary(*((GDScriptParser::AnnotationNode *)&self));
		case GDScriptParser::Node::Type::ASSERT:
			return to_dictionary(*((GDScriptParser::AssertNode *)&self));
		case GDScriptParser::Node::Type::BREAK:
			return to_dictionary(*((GDScriptParser::BreakNode *)&self));
		case GDScriptParser::Node::Type::BREAKPOINT:
			return to_dictionary(*((GDScriptParser::BreakpointNode *)&self));
		case GDScriptParser::Node::Type::CLASS:
			return to_dictionary(*((GDScriptParser::ClassNode *)&self));
		case GDScriptParser::Node::Type::CONSTANT:
			return to_dictionary(*((GDScriptParser::ConstantNode *)&self));
		case GDScriptParser::Node::Type::CONTINUE:
			return to_dictionary(*((GDScriptParser::ContinueNode *)&self));
		case GDScriptParser::Node::Type::ENUM:
			return to_dictionary(*((GDScriptParser::EnumNode *)&self));
		case GDScriptParser::Node::Type::FOR:
			return to_dictionary(*((GDScriptParser::ForNode *)&self));
		case GDScriptParser::Node::Type::FUNCTION:
			return to_dictionary(*((GDScriptParser::FunctionNode *)&self));
		case GDScriptParser::Node::Type::IF:
			return to_dictionary(*((GDScriptParser::IfNode *)&self));
		case GDScriptParser::Node::Type::MATCH:
			return to_dictionary(*((GDScriptParser::MatchNode *)&self));
		case GDScriptParser::Node::Type::MATCH_BRANCH:
			return to_dictionary(*((GDScriptParser::MatchBranchNode *)&self));
		case GDScriptParser::Node::Type::PARAMETER:
			return to_dictionary(*((GDScriptParser::ParameterNode *)&self));
		case GDScriptParser::Node::Type::PASS:
			return to_dictionary(*((GDScriptParser::PassNode *)&self));
		case GDScriptParser::Node::Type::PATTERN:
			return to_dictionary(*((GDScriptParser::PatternNode *)&self));
		case GDScriptParser::Node::Type::RETURN:
			return to_dictionary(*((GDScriptParser::ReturnNode *)&self));
		case GDScriptParser::Node::Type::SIGNAL:
			return to_dictionary(*((GDScriptParser::SignalNode *)&self));
		case GDScriptParser::Node::Type::SUITE:
			return to_dictionary(*((GDScriptParser::SuiteNode *)&self));
		case GDScriptParser::Node::Type::TYPE:
			return to_dictionary(*((GDScriptParser::TypeNode *)&self));
		case GDScriptParser::Node::Type::VARIABLE:
			return to_dictionary(*((GDScriptParser::VariableNode *)&self));
		case GDScriptParser::Node::Type::WHILE:
			return to_dictionary(*((GDScriptParser::WhileNode *)&self));
		case GDScriptParser::Node::Type::ARRAY:
		case GDScriptParser::Node::Type::ASSIGNMENT:
		case GDScriptParser::Node::Type::AWAIT:
		case GDScriptParser::Node::Type::BINARY_OPERATOR:
		case GDScriptParser::Node::Type::CALL:
		case GDScriptParser::Node::Type::CAST:
		case GDScriptParser::Node::Type::DICTIONARY:
		case GDScriptParser::Node::Type::GET_NODE:
		case GDScriptParser::Node::Type::IDENTIFIER:
		case GDScriptParser::Node::Type::LAMBDA:
		case GDScriptParser::Node::Type::LITERAL:
		case GDScriptParser::Node::Type::PRELOAD:
		case GDScriptParser::Node::Type::SELF:
		case GDScriptParser::Node::Type::SUBSCRIPT:
		case GDScriptParser::Node::Type::TERNARY_OPERATOR:
		case GDScriptParser::Node::Type::TYPE_TEST:
		case GDScriptParser::Node::Type::UNARY_OPERATOR:
			return to_dictionary(*((GDScriptParser::ExpressionNode *)&self));
		default:
			ERR_FAIL_V_MSG(Dictionary(), "Bug!");
	}
}

Dictionary GDScriptTreeConverter::to_dictionary_without_elevation(const GDScriptParser::Node &self) {
	Dictionary dict;
	dict["type"] = self.type;
	dict["start_line"] = self.start_line;
	dict["end_line"] = self.end_line;
	dict["start_column"] = self.start_column;
	dict["end_column"] = self.end_column;
	dict["leftmost_column"] = self.leftmost_column;
	dict["rightmost_column"] = self.rightmost_column;
	////if(self.next != nullptr)
	////dict["next"] = to_dictionary(*self.next);
	Array dict_annotations;
	dict["annotations"] = dict_annotations;
	for (const GDScriptParser::AnnotationNode *annotation : self.annotations) {
		if (annotation != nullptr) {
			dict_annotations.push_back(to_dictionary(*annotation));
		}
	}
	dict["datatype"] = to_dictionary(self.datatype);
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::ExpressionNode &self) {
	switch (self.type) {
		case GDScriptParser::Node::Type::ARRAY:
			return to_dictionary(*((GDScriptParser::ArrayNode *)&self));
		case GDScriptParser::Node::Type::ASSIGNMENT:
			return to_dictionary(*((GDScriptParser::AssignmentNode *)&self));
		case GDScriptParser::Node::Type::AWAIT:
			return to_dictionary(*((GDScriptParser::AwaitNode *)&self));
		case GDScriptParser::Node::Type::BINARY_OPERATOR:
			return to_dictionary(*((GDScriptParser::BinaryOpNode *)&self));
		case GDScriptParser::Node::Type::CALL:
			return to_dictionary(*((GDScriptParser::CallNode *)&self));
		case GDScriptParser::Node::Type::CAST:
			return to_dictionary(*((GDScriptParser::CastNode *)&self));
		case GDScriptParser::Node::Type::DICTIONARY:
			return to_dictionary(*((GDScriptParser::DictionaryNode *)&self));
		case GDScriptParser::Node::Type::GET_NODE:
			return to_dictionary(*((GDScriptParser::GetNodeNode *)&self));
		case GDScriptParser::Node::Type::IDENTIFIER:
			return to_dictionary(*((GDScriptParser::IdentifierNode *)&self));
		case GDScriptParser::Node::Type::LAMBDA:
			return to_dictionary(*((GDScriptParser::LambdaNode *)&self));
		case GDScriptParser::Node::Type::LITERAL:
			return to_dictionary(*((GDScriptParser::LiteralNode *)&self));
		case GDScriptParser::Node::Type::PRELOAD:
			return to_dictionary(*((GDScriptParser::PreloadNode *)&self));
		case GDScriptParser::Node::Type::SELF:
			return to_dictionary(*((GDScriptParser::SelfNode *)&self));
		case GDScriptParser::Node::Type::SUBSCRIPT:
			return to_dictionary(*((GDScriptParser::SubscriptNode *)&self));
		case GDScriptParser::Node::Type::TERNARY_OPERATOR:
			return to_dictionary(*((GDScriptParser::TernaryOpNode *)&self));
		case GDScriptParser::Node::Type::TYPE_TEST:
			return to_dictionary(*((GDScriptParser::TypeTestNode *)&self));
		case GDScriptParser::Node::Type::UNARY_OPERATOR:
			return to_dictionary(*((GDScriptParser::UnaryOpNode *)&self));
		default:
			ERR_FAIL_V_MSG(Dictionary(), "Bug!");
	}
}

Dictionary GDScriptTreeConverter::to_dictionary_without_elevation(const GDScriptParser::ExpressionNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	dict["reduced"] = self.reduced;
	dict["is_constant"] = self.is_constant;
	dict["reduced_value"] = self.reduced_value;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::AnnotationNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	dict["name"] = self.name;

	Array dict_arguments;
	dict["arguments"] = dict_arguments;
	for (const GDScriptParser::ExpressionNode *argument : self.arguments) {
		if (argument != nullptr) {
			dict_arguments.push_back(to_dictionary(*argument));
		}
	}

	Array dict_resolved_arguments;
	dict["resolved_arguments"] = dict_resolved_arguments;
	for (const Variant &argument : self.resolved_arguments) {
		dict_resolved_arguments.push_back(argument);
	}

	////if(self.info != nullptr)
	////dict["info"] = to_dictionary(*self.info);
	dict["export_info"] = to_dictionary(self.export_info);
	dict["is_resolved"] = self.is_resolved;
	dict["is_applied"] = self.is_applied;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::ArrayNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	Array dict_elements;
	dict["elements"] = dict_elements;
	for (const GDScriptParser::ExpressionNode *element : self.elements) {
		if (element != nullptr) {
			dict_elements.push_back(to_dictionary(*element));
		}
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::AssertNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	if (self.condition != nullptr) {
		dict["condition"] = to_dictionary(*self.condition);
	}
	if (self.message != nullptr) {
		dict["message"] = to_dictionary(*self.message);
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::AssignableNode &self) {
	switch (self.type) {
		case GDScriptParser::Node::Type::CONSTANT:
			return to_dictionary(*((GDScriptParser::ConstantNode *)&self));
		case GDScriptParser::Node::Type::PARAMETER:
			return to_dictionary(*((GDScriptParser::ParameterNode *)&self));
		case GDScriptParser::Node::Type::VARIABLE:
			return to_dictionary(*((GDScriptParser::VariableNode *)&self));
		default:
			ERR_FAIL_V_MSG(Dictionary(), "Bug!");
	}
}

Dictionary GDScriptTreeConverter::to_dictionary_without_elevation(const GDScriptParser::AssignableNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	if (self.identifier != nullptr) {
		dict["identifier"] = to_dictionary(*self.identifier);
	}
	if (self.initializer != nullptr) {
		dict["initializer"] = to_dictionary(*self.initializer);
	}
	if (self.datatype_specifier != nullptr) {
		dict["datatype_specifier"] = to_dictionary(*self.datatype_specifier);
	}
	dict["infer_datatype"] = self.infer_datatype;
	dict["use_conversion_assign"] = self.use_conversion_assign;
	dict["usages"] = self.usages;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::AssignmentNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	dict["operation"] = self.operation;
	dict["variant_op"] = self.variant_op;
	if (self.assignee != nullptr) {
		dict["assignee"] = to_dictionary(*self.assignee);
	}
	if (self.assigned_value != nullptr) {
		dict["assigned_value"] = to_dictionary(*self.assigned_value);
	}
	dict["use_conversion_assign"] = self.use_conversion_assign;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::AwaitNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	if (self.to_await != nullptr) {
		dict["to_await"] = to_dictionary(*self.to_await);
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::BinaryOpNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	dict["operation"] = self.operation;
	dict["variant_op"] = self.variant_op;
	if (self.left_operand != nullptr) {
		dict["left_operand"] = to_dictionary(*self.left_operand);
	}
	if (self.right_operand != nullptr) {
		dict["right_operand"] = to_dictionary(*self.right_operand);
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::BreakNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::BreakpointNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::CallNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	if (self.callee != nullptr) {
		dict["callee"] = to_dictionary(*self.callee);
	}

	Array dict_arguments;
	dict["arguments"] = dict_arguments;
	for (GDScriptParser::ExpressionNode *arg : self.arguments) {
		dict_arguments.append(to_dictionary(*arg));
	}

	dict["function_name"] = self.function_name;
	dict["is_super"] = self.is_super;
	dict["is_static"] = self.is_static;

	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::CastNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	if (self.operand != nullptr) {
		dict["operand"] = to_dictionary(*self.operand);
	}
	if (self.cast_type != nullptr) {
		dict["cast_type"] = to_dictionary(*self.cast_type);
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::EnumNode::Value &self) {
	Dictionary dict;
	if (self.identifier != nullptr) {
		dict["identifier"] = to_dictionary(*self.identifier);
	}
	if (self.custom_value != nullptr) {
		dict["custom_value"] = to_dictionary(*self.custom_value);
	}
	////if(self.parent_enum != nullptr){
	////    dict["parent_enum"] = to_dictionary(*self.parent_enum);
	////}
	dict["index"] = self.index;
	dict["resolved"] = self.resolved;
	dict["value"] = self.value;
	dict["line"] = self.line;
	dict["leftmost_column"] = self.leftmost_column;
	dict["rightmost_column"] = self.rightmost_column;
	dict["doc_data"] = to_dictionary(self.doc_data);
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::EnumNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);

	if (self.identifier != nullptr) {
		dict["identifier"] = to_dictionary(*self.identifier);
	}

	Array dict_values;
	dict["values"] = dict_values;
	for (const GDScriptParser::EnumNode::Value &v : self.values) {
		dict_values.push_back(to_dictionary(v));
	}
	dict["dictionary"] = self.dictionary;
	dict["doc_data"] = to_dictionary(self.doc_data);

	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::ClassNode::Member &self) {
	switch (self.type) {
		case GDScriptParser::ClassNode::Member::Type::CLASS:
			return to_dictionary(*self.m_class);
		case GDScriptParser::ClassNode::Member::CONSTANT:
			return to_dictionary(*self.constant);
		case GDScriptParser::ClassNode::Member::FUNCTION:
			return to_dictionary(*self.function);
		case GDScriptParser::ClassNode::Member::VARIABLE:
			return to_dictionary(*self.variable);
		case GDScriptParser::ClassNode::Member::ENUM:
			return to_dictionary(*self.m_enum);
		case GDScriptParser::ClassNode::Member::ENUM_VALUE:
			return to_dictionary(self.enum_value);
		case GDScriptParser::ClassNode::Member::SIGNAL:
			return to_dictionary(*self.signal);
		case GDScriptParser::ClassNode::Member::GROUP:
			return to_dictionary(*self.annotation);
		case GDScriptParser::ClassNode::Member::UNDEFINED:
			return Dictionary();
	}
	return Dictionary();
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::ClassNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	if (self.identifier != nullptr) {
		dict["identifier"] = to_dictionary(*self.identifier);
	}
	dict["icon_path"] = self.icon_path;
	dict["simplified_icon_path"] = self.simplified_icon_path;

	Array dict_members;
	dict["members"] = dict_members;
	for (const GDScriptParser::ClassNode::Member &member : self.members) {
		dict_members.push_back(to_dictionary(member));
	}

	Dictionary dict_members_indices;
	dict["members_indices"] = dict_members_indices;
	for (const KeyValue<StringName, int> &kv : self.members_indices) {
		dict_members_indices[(String)kv.key] = kv.value;
	}
	////if(self.outer != nullptr)
	////dict["outer"] = to_dictionary(*self.outer);
	dict["extends_used"] = self.extends_used;
	dict["onready_used"] = self.onready_used;
	dict["has_static_data"] = self.has_static_data;
	dict["annotated_static_unload"] = self.annotated_static_unload;
	dict["extends_path"] = self.extends_path;
	Array dict_extends;
	dict["extends"] = dict_extends;
	for (const GDScriptParser::IdentifierNode *extend : self.extends) {
		if (extend != nullptr) {
			dict_extends.push_back(to_dictionary(*extend));
		}
	}
	dict["base_type"] = to_dictionary(self.base_type);
	dict["fqcn"] = self.fqcn;
	dict["doc_data"] = to_dictionary(self.doc_data);
	dict["resolved_interface"] = self.resolved_interface;
	dict["resolved_body"] = self.resolved_body;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::ConstantNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::AssignableNode)self);
	dict["doc_data"] = to_dictionary(self.doc_data);
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::ContinueNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::DictionaryNode::Pair &self) {
	Dictionary dict;
	if (self.key != nullptr) {
		dict["key"] = to_dictionary(*self.key);
	}
	if (self.value != nullptr) {
		dict["value"] = to_dictionary(*self.value);
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::DictionaryNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);

	dict["style"] = self.style;

	Array dict_elements;
	dict["elements"] = dict_elements;
	for (const GDScriptParser::DictionaryNode::Pair &pair : self.elements) {
		dict_elements.push_back(to_dictionary(pair));
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::ForNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	if (self.variable != nullptr) {
		dict["variable"] = to_dictionary(*self.variable);
	}
	if (self.datatype_specifier != nullptr) {
		dict["datatype_specifier"] = to_dictionary(*self.datatype_specifier);
	}
	dict["use_conversion_assign"] = self.use_conversion_assign;
	if (self.list != nullptr) {
		dict["list"] = to_dictionary(*self.list);
	}
	if (self.loop != nullptr) {
		dict["loop"] = to_dictionary(*self.loop);
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::FunctionNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	if (self.identifier != nullptr) {
		dict["identifier"] = to_dictionary(*self.identifier);
	}

	Array dict_parameters;
	dict["parameters"] = dict_parameters;
	for (const GDScriptParser::ParameterNode *parameter : self.parameters) {
		if (parameter != nullptr) {
			dict_parameters.push_back(to_dictionary(*parameter));
		}
	}
	Dictionary dict_parameters_indices;
	dict["parameters_indices"] = dict_parameters_indices;
	for (const KeyValue<StringName, int> &kv : self.parameters_indices) {
		dict_parameters_indices[(String)kv.key] = kv.value;
	}

	if (self.return_type != nullptr) {
		dict["return_type"] = to_dictionary(*self.return_type);
	}
	if (self.body != nullptr) {
		dict["body"] = to_dictionary(*self.body);
	}

	dict["is_static"] = self.is_static;
	dict["is_coroutine"] = self.is_coroutine;
	dict["rpc_config"] = self.rpc_config;

	dict["info"] = to_dictionary(self.info);

	////if(self.source_lambda != nullptr)
	////dict["source_lambda"] = to_dictionary(*self.source_lambda);

	Array dict_default_arg_values;
	dict["default_arg_values"] = dict_default_arg_values;
	for (const Variant &default_arg_value : self.default_arg_values) {
		dict_default_arg_values.push_back(default_arg_value);
	}

	dict["doc_data"] = to_dictionary(self.doc_data);
	dict["min_local_doc_line"] = self.min_local_doc_line;

	dict["resolved_signature"] = self.resolved_signature;
	dict["resolved_body"] = self.resolved_body;

	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::GetNodeNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	dict["full_path"] = self.full_path;
	dict["use_dollar"] = self.use_dollar;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::IdentifierNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);

	dict["name"] = self.name;
	////if(self.suite != nullptr)
	////dict["suite"] = to_dictionary(*self.suite);
	dict["source"] = self.source;

	switch (self.source) {
		case GDScriptParser::IdentifierNode::Source::FUNCTION_PARAMETER:
			////if(self.parameter_source != nullptr)
			////dict["parameter_source"] = to_dictionary(*self.parameter_source);
			break;
		case GDScriptParser::IdentifierNode::Source::LOCAL_ITERATOR:
		case GDScriptParser::IdentifierNode::Source::LOCAL_BIND:
			////if(self.bind_source != nullptr)
			////dict["bind_source"] = to_dictionary(*self.bind_source);
			break;
		case GDScriptParser::IdentifierNode::Source::LOCAL_VARIABLE:
		case GDScriptParser::IdentifierNode::Source::MEMBER_VARIABLE:
		case GDScriptParser::IdentifierNode::Source::STATIC_VARIABLE:
		case GDScriptParser::IdentifierNode::Source::INHERITED_VARIABLE:
			////if(self.variable_source != nullptr)
			////dict["variable_source"] = to_dictionary(*self.variable_source);
			break;
		case GDScriptParser::IdentifierNode::Source::LOCAL_CONSTANT:
		case GDScriptParser::IdentifierNode::Source::MEMBER_CONSTANT:
			////if(self.constant_source != nullptr)
			////dict["constant_source"] = to_dictionary(*self.constant_source);
			break;
		case GDScriptParser::IdentifierNode::Source::MEMBER_SIGNAL:
			////if(self.signal_source != nullptr)
			////dict["signal_source"] = to_dictionary(*self.signal_source);
			break;
		case GDScriptParser::IdentifierNode::Source::MEMBER_FUNCTION:
			////if(self.function_source != nullptr)
			////dict["function_source"] = to_dictionary(*self.function_source);
			break;
		case GDScriptParser::IdentifierNode::Source::UNDEFINED_SOURCE:
		case GDScriptParser::IdentifierNode::Source::MEMBER_CLASS:
			break;
	}
	dict["function_source_is_static"] = self.function_source_is_static;
	////if(self.source_function != nullptr)
	////dict["source_function"] = to_dictionary(*self.source_function);

	dict["usages"] = self.usages;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::IfNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	if (self.condition != nullptr) {
		dict["condition"] = to_dictionary(*self.condition);
	}
	if (self.true_block != nullptr) {
		dict["true_block"] = to_dictionary(*self.true_block);
	}
	if (self.false_block != nullptr) {
		dict["false_block"] = to_dictionary(*self.false_block);
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::LambdaNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);

	if (self.function != nullptr) {
		dict["function"] = to_dictionary(*self.function);
	}
	////if(self.parent_function != nullptr)
	////dict["parent_function"] = to_dictionary(*self.parent_function);
	////if(self.parent_lambda != nullptr)
	////dict["parent_lambda"] = to_dictionary(*self.parent_lambda);

	Array dict_captures;
	dict["captures"] = dict_captures;
	for (const GDScriptParser::IdentifierNode *capture : self.captures) {
		if (capture != nullptr) {
			dict_captures.append(to_dictionary(*capture));
		}
	}

	Dictionary dict_captures_indices;
	dict["captures_indices"] = dict_captures_indices;
	for (const KeyValue<StringName, int> &kv : self.captures_indices) {
		dict_captures_indices[(String)kv.key] = kv.value;
	}

	dict["use_self"] = self.use_self;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::LiteralNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	dict["value"] = self.value;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::MatchNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	if (self.test != nullptr) {
		dict["test"] = to_dictionary(*self.test);
	}

	Array dict_branches;
	dict["branches"] = dict_branches;
	for (const GDScriptParser::MatchBranchNode *branch : self.branches) {
		if (branch != nullptr) {
			dict_branches.push_back(to_dictionary(*branch));
		}
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::MatchBranchNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);

	Array dict_patterns;
	dict["patterns"] = dict_patterns;
	for (const GDScriptParser::PatternNode *pattern : self.patterns) {
		if (pattern != nullptr) {
			dict_patterns.append(to_dictionary(*pattern));
		}
	}
	if (self.block != nullptr) {
		dict["block"] = to_dictionary(*self.block);
	}
	dict["has_wildcard"] = self.has_wildcard;
	if (self.guard_body != nullptr) {
		dict["guard_body"] = to_dictionary(*self.guard_body);
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::ParameterNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::AssignableNode)self);
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::PassNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::PatternNode::Pair &self) {
	Dictionary dict;
	if (self.key != nullptr) {
		dict["key"] = to_dictionary(*self.key);
	}
	if (self.value_pattern != nullptr) {
		dict["value_pattern"] = to_dictionary(*self.value_pattern);
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::PatternNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	dict["pattern_type"] = self.pattern_type;

	switch (self.pattern_type) {
		case GDScriptParser::PatternNode::Type::PT_LITERAL:
			if (self.literal != nullptr) {
				dict["literal"] = to_dictionary(*self.literal);
			}
			break;
		case GDScriptParser::PatternNode::Type::PT_EXPRESSION:
			if (self.expression != nullptr) {
				dict["expression"] = to_dictionary(*self.expression);
			}
			break;
		case GDScriptParser::PatternNode::Type::PT_BIND:
			if (self.bind != nullptr) {
				dict["bind"] = to_dictionary(*self.bind);
			}
			break;
		case GDScriptParser::PatternNode::Type::PT_ARRAY:
		case GDScriptParser::PatternNode::Type::PT_DICTIONARY:
		case GDScriptParser::PatternNode::Type::PT_REST:
		case GDScriptParser::PatternNode::Type::PT_WILDCARD:
			break;
	}

	Array dict_array;
	dict["array"] = dict_array;
	for (const GDScriptParser::PatternNode *pattern : self.array) {
		if (pattern != nullptr) {
			dict_array.append(to_dictionary(*pattern));
		}
	}

	dict["rest_used"] = self.rest_used;

	Array dict_dictionary;
	dict["dictionary"] = dict_dictionary;
	for (const GDScriptParser::PatternNode::Pair &pair : self.dictionary) {
		dict_dictionary.append(to_dictionary(pair));
	}

	Dictionary dict_binds;
	dict["binds"] = dict_binds;
	for (const KeyValue<StringName, GDScriptParser::IdentifierNode *> &kv : self.binds) {
		if (kv.value != nullptr) {
			dict_binds[(String)kv.key] = to_dictionary(*kv.value);
		}
	}

	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::PreloadNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	if (self.path != nullptr) {
		dict["path"] = to_dictionary(*self.path);
	}
	dict["resolved_path"] = self.resolved_path;
	//if(self.resource.is_valid())
	//dict["resource"] = to_dictionary(*self.resource);
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::ReturnNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	if (self.return_value != nullptr) {
		dict["return_value"] = to_dictionary(*self.return_value);
	}
	dict["void_return"] = self.void_return;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::SelfNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	////if(self.current_class != nullptr)
	////dict["current_class"] = to_dictionary(*self.current_class);
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::SignalNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	if (self.identifier != nullptr) {
		dict["identifier"] = to_dictionary(*self.identifier);
	}
	Array dict_parameters;
	dict["parameters"] = dict_parameters;
	for (const GDScriptParser::ParameterNode *parameter : self.parameters) {
		if (parameter != nullptr) {
			dict_parameters.append(to_dictionary(*parameter));
		}
	}
	Dictionary dict_parameters_indices;
	dict["parameters_indices"] = dict_parameters_indices;
	for (const KeyValue<StringName, int> &kv : self.parameters_indices) {
		dict_parameters_indices[(String)kv.key] = kv.value;
	}
	dict["method_info"] = to_dictionary(self.method_info);
	dict["doc_data"] = to_dictionary(self.doc_data);
	dict["usages"] = self.usages;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::SubscriptNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	if (self.base != nullptr) {
		dict["base"] = to_dictionary(*self.base);
	}
	if (self.is_attribute) {
		if (self.attribute != nullptr) {
			dict["attribute"] = to_dictionary(*self.attribute);
		}
	} else {
		if (self.index != nullptr) {
			dict["index"] = to_dictionary(*self.index);
		}
	}
	dict["is_attribute"] = self.is_attribute;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::SuiteNode::Local &self) {
	Dictionary dict;
	dict["type"] = self.type;
	switch (self.type) {
		case GDScriptParser::SuiteNode::Local::Type::CONSTANT:
			////if(self.constant != nullptr)
			////dict["constant"] = to_dictionary(*self.constant);
			break;
		case GDScriptParser::SuiteNode::Local::Type::VARIABLE:
		case GDScriptParser::SuiteNode::Local::Type::FOR_VARIABLE:
			////if(self.variable != nullptr)
			////dict["variable"] = to_dictionary(*self.variable);
			break;
		case GDScriptParser::SuiteNode::Local::Type::PARAMETER:
			////if(self.parameter != nullptr)
			////dict["parameter"] = to_dictionary(*self.parameter);
			break;
		case GDScriptParser::SuiteNode::Local::Type::PATTERN_BIND:
			////if(self.bind != nullptr)
			////dict["bind"] = to_dictionary(*self.bind);
			break;
		case GDScriptParser::SuiteNode::Local::Type::UNDEFINED:
			break;
	}
	dict["name"] = (String)self.name;
	dict["start_line"] = self.start_line;
	dict["end_line"] = self.end_line;
	dict["start_column"] = self.start_column;
	dict["end_column"] = self.end_column;
	dict["leftmost_column"] = self.leftmost_column;
	dict["rightmost_column"] = self.rightmost_column;
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::SuiteNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	////if(self.parent_block != nullptr)
	////dict["parent_block"] = to_dictionary(*self.parent_block);

	Array dict_statements;
	dict["statements"] = dict_statements;
	for (const GDScriptParser::Node *statement : self.statements) {
		if (statement != nullptr) {
			dict_statements.append(to_dictionary(*statement)); //TODO:
		}
	}

	dict["empty"] = to_dictionary(self.empty);

	Array dict_locals;
	dict["locals"] = dict_locals;
	for (const GDScriptParser::SuiteNode::Local &local : self.locals) {
		dict_locals.push_back(to_dictionary(local));
	}

	Dictionary dict_locals_indices;
	dict["locals_indices"] = dict_locals_indices;
	for (const KeyValue<StringName, int> &kv : self.locals_indices) {
		dict_locals_indices[(String)kv.key] = kv.value;
	}

	////if(self.parent_function != nullptr)
	////dict["parent_function"] = to_dictionary(*self.parent_function);
	////if(self.parent_if != nullptr)
	////dict["parent_if"] = to_dictionary(*self.parent_if);

	dict["has_return"] = self.has_return;
	dict["has_continue"] = self.has_continue;
	dict["has_unreachable_code"] = self.has_unreachable_code;
	dict["is_in_loop"] = self.is_in_loop;

	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::TernaryOpNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	if (self.condition != nullptr) {
		dict["condition"] = to_dictionary(*self.condition);
	}
	if (self.true_expr != nullptr) {
		dict["true_expr"] = to_dictionary(*self.true_expr);
	}
	if (self.false_expr != nullptr) {
		dict["false_expr"] = to_dictionary(*self.false_expr);
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::TypeNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);

	Array dict_type_chain;
	dict["type_chain"] = dict_type_chain;
	for (const GDScriptParser::IdentifierNode *type_from_chain : self.type_chain) {
		if (type_from_chain != nullptr) {
			dict_type_chain.append(to_dictionary(*type_from_chain));
		}
	}

	Array dict_container_types;
	dict["container_types"] = dict_container_types;
	for (const GDScriptParser::TypeNode *container_type : self.container_types) {
		if (container_type != nullptr) {
			dict_container_types.append(to_dictionary(*container_type));
		}
	}

	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::TypeTestNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	if (self.operand != nullptr) {
		dict["operand"] = to_dictionary(*self.operand);
	}
	if (self.test_type != nullptr) {
		dict["test_type"] = to_dictionary(*self.test_type);
	}
	dict["test_datatype"] = to_dictionary(self.test_datatype);
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::UnaryOpNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::ExpressionNode)self);
	dict["operation"] = self.operation;
	dict["variant_op"] = self.variant_op;
	if (self.operand != nullptr) {
		dict["operand"] = to_dictionary(*self.operand);
	}
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::VariableNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::AssignableNode)self);
	dict["property"] = self.property;
	switch (self.property) {
		case GDScriptParser::VariableNode::PropertyStyle::PROP_INLINE:
			if (self.setter != nullptr) {
				dict["setter"] = to_dictionary(*self.setter);
			}
			if (self.setter_parameter != nullptr) {
				dict["setter_parameter"] = to_dictionary(*self.setter_parameter);
			}
			if (self.getter != nullptr) {
				dict["getter"] = to_dictionary(*self.getter);
			}
			break;
		case GDScriptParser::VariableNode::PropertyStyle::PROP_SETGET:
			if (self.setter_pointer != nullptr) {
				dict["setter"] = to_dictionary(*self.setter_pointer);
			}
			if (self.getter_pointer != nullptr) {
				dict["getter"] = to_dictionary(*self.getter_pointer);
			}
			break;
		case GDScriptParser::VariableNode::PropertyStyle::PROP_NONE:
			break;
	}
	dict["exported"] = self.exported;
	dict["onready"] = self.onready;
	dict["export_info"] = to_dictionary(self.export_info);
	dict["assignments"] = self.assignments;
	dict["is_static"] = self.is_static;
	dict["doc_data"] = to_dictionary(self.doc_data);
	return dict;
}

Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::WhileNode &self) {
	Dictionary dict = to_dictionary_without_elevation((GDScriptParser::Node)self);
	if (self.condition != nullptr) {
		dict["condition"] = to_dictionary(*self.condition);
	}
	if (self.loop != nullptr) {
		dict["loop"] = to_dictionary(*self.loop);
	}
	return dict;
}
/*
Dictionary GDScriptTreeConverter::to_dictionary(const GDScriptParser::AnnotationInfo &self) {
	Dictionary dict;
	dict["target_kind"] = self.target_kind;
	//dict["apply"] = self.apply;
	dict["info"] = to_dictionary(self.info);
	return dict;
}
*/
#endif // TOOLS_ENABLED

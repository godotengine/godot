#ifdef TOOLS_ENABLED

#include "modules/gdscript/gdscript_parser.h"

Dictionary to_dictionary(GDScriptParser::DataType& self) {
    Dictionary dict;

    Array dict_container_element_types;
    dict["container_element_types"] = dict_container_element_types;
    for (const GDScriptParser::DataType &container_element_type : self.container_element_types) {
        dict_container_element_types.append(container_element_type.to_dictionary());
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
    ////dict["class_type"] = self.class_type->to_dictionary();

    dict["method_info"] = self.method_info.to_dictionary();

    Dictionary dict_enum_values;
    dict["enum_values"] = dict_enum_values;
    for (const KeyValue<StringName, int64_t> &kv : self.enum_values) {
        dict_enum_values[(String)kv.key] = kv.value;
    }
    return dict;
}

Dictionary to_dictionary(GDScriptParser::ClassDocData &self) {
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

Dictionary to_dictionary(GDScriptParser::MemberDocData &self) {
    Dictionary dict;
    dict["description"] = self.description;
    dict["is_deprecated"] = self.is_deprecated;
    dict["deprecated_message"] = self.deprecated_message;
    dict["is_experimental"] = self.is_experimental;
    dict["experimental_message"] = self.experimental_message;
    return dict;
}

Dictionary to_dictionary(GDScriptParser::Node &self) {
    Dictionary dict;
    dict["type"] = self.type;
    dict["start_line"] = self.start_line;
    dict["end_line"] = self.end_line;
    dict["start_column"] = self.start_column;
    dict["end_column"] = self.end_column;
    dict["leftmost_column"] = self.leftmost_column;
    dict["rightmost_column"] = self.rightmost_column;
    ////if(self.next != nullptr)
    ////dict["next"] = self.next->to_dictionary();
    Array dict_annotations;
    dict["annotations"] = dict_annotations;
    for (const GDScriptParser::AnnotationNode *annotation : self.annotations) {
        if (annotation != nullptr) {
            dict_annotations.push_back(annotation->to_dictionary());
        }
    }
    dict["datatype"] = self.datatype.to_dictionary();
    return dict;
}

Dictionary to_dictionary(GDScriptParser::AnnotationNode &self) {
    Dictionary dict = GDScriptParser::Node::to_dictionary();
    dict["name"] = self.name;

    Array dict_arguments;
    dict["arguments"] = dict_arguments;
    for (const GDScriptParser::ExpressionNode *argument : self.arguments) {
        if (argument != nullptr) {
            dict_arguments.push_back(argument->to_dictionary());
        }
    }

    Array dict_resolved_arguments;
    dict["resolved_arguments"] = dict_resolved_arguments;
    for (const Variant &argument : self.resolved_arguments) {
        dict_resolved_arguments.push_back(argument);
    }

    ////if(self.info != nullptr)
    ////dict["info"] = self.info->to_dictionary();
    dict["export_info"] = self.export_info.to_dictionary();
    dict["is_resolved"] = self.is_resolved;
    dict["is_applied"] = self.is_applied;
    return dict;
}

Dictionary to_dictionary(GDScriptParser::ArrayNode &self) {
    Dictionary dict = GDScriptParser::ExpressionNode::to_dictionary();
    Array dict_elements;
    dict["elements"] = dict_elements;
    for (const GDScriptParser::ExpressionNode *element : self.elements) {
        if (element != nullptr) {
            dict_elements.push_back(element->to_dictionary());
        }
    }
    return dict;
}

Dictionary to_dictionary(GDScriptParser::AssignmentNode &self) {
    Dictionary dict = GDScriptParser::ExpressionNode::to_dictionary();
    dict["operation"] = self.operation;
    dict["variant_op"] = self.variant_op;
    if (self.assignee != nullptr) {
        dict["assignee"] = self.assignee->to_dictionary();
    }
    if (self.assigned_value != nullptr) {
        dict["assigned_value"] = self.assigned_value->to_dictionary();
    }
    dict["use_conversion_assign"] = self.use_conversion_assign;
    return dict;
}

Dictionary to_dictionary(GDScriptParser::AwaitNode &self) {
    Dictionary dict = GDScriptParser::ExpressionNode::to_dictionary();
    if (self.to_await != nullptr) {
        dict["to_await"] = self.to_await->to_dictionary();
    }
    return dict;
}

Dictionary to_dictionary(GDScriptParser::BinaryOpNode &self) {
    Dictionary dict = ExpressionNode::to_dictionary();
    dict["operation"] = self.operation;
    dict["variant_op"] = self.variant_op;
    if (self.left_operand != nullptr) {
        dict["left_operand"] = self.left_operand->to_dictionary();
    }
    if (self.right_operand != nullptr) {
        dict["right_operand"] = self.right_operand->to_dictionary();
    }
    return dict;
}

Dictionary to_dictionary(GDScriptParser::CallNode &self) {
    Dictionary dict = ExpressionNode::to_dictionary();
    if (self.callee != nullptr) {
        dict["callee"] = self.callee->to_dictionary();
    }

    Array dict_arguments;
    dict["arguments"] = dict_arguments;
    for (GDScriptParser::ExpressionNode *arg : self.arguments) {
        dict_arguments.append(arg->to_dictionary());
    }

    dict["function_name"] = self.function_name;
    dict["is_super"] = self.is_super;
    dict["is_static"] = self.is_static;

    return dict;
}

Dictionary to_dictionary(GDScriptParser::CastNode &self) {
    Dictionary dict = GDScriptParser::ExpressionNode::to_dictionary();
    if (self.operand != nullptr) {
        dict["operand"] = self.operand->to_dictionary();
    }
    if (self.cast_type != nullptr) {
        dict["cast_type"] = self.cast_type->to_dictionary();
    }
    return dict;
}

Dictionary to_dictionary(GDScriptParser::EnumNode &self) {
    Dictionary dict = GDScriptParser::Node::to_dictionary();

    if (self.identifier != nullptr) {
        dict["identifier"] = self.identifier->to_dictionary();
    }

    Array dict_values;
    dict["values"] = dict_values;
    for (const GDScriptParser::EnumNode::Value &v : self.values) {
        dict_values.push_back(v.identifier->to_dictionary());
    }
    dict["dictionary"] = self.dictionary;
    dict["doc_data"] = self.doc_data.to_dictionary();

    return dict;
}

Dictionary to_dictionary(GDScriptParser::ClassNode::Member &self) {
    return self.get_source_node()->to_dictionary();
}

Dictionary to_dictionary(GDScriptParser::ClassNode &self) {
    Dictionary dict = Node::to_dictionary();
    if (self.identifier != nullptr) {
        dict["identifier"] = self.identifier->to_dictionary();
    }
    dict["icon_path"] = self.icon_path;
    dict["simplified_icon_path"] = self.simplified_icon_path;

    Array dict_members;
    dict["members"] = dict_members;
    for (const GDScriptParser::ClassNode::Member &member : self.members) {
        dict_members.push_back(member.to_dictionary());
    }

    Dictionary dict_members_indices;
    dict["members_indices"] = dict_members_indices;
    for (const KeyValue<StringName, int> &kv : self.members_indices) {
        dict_members_indices[(String)kv.key] = kv.value;
    }
    ////if(self.outer != nullptr)
    ////dict["outer"] = self.outer->to_dictionary();
    dict["extends_used"] = self.extends_used;
    dict["onready_used"] = self.onready_used;
    dict["has_static_data"] = self.has_static_data;
    dict["annotated_static_unload"] = self.annotated_static_unload;
    dict["extends_path"] = self.extends_path;
    Array dict_extends;
    dict["extends"] = dict_extends;
    for (const GDScriptParser::IdentifierNode *extend : self.extends) {
        if (extend != nullptr) {
            dict_extends.push_back(extend->to_dictionary());
        }
    }
    dict["base_type"] = self.base_type.to_dictionary();
    dict["fqcn"] = self.fqcn;
    dict["doc_data"] = self.doc_data.to_dictionary();
    dict["resolved_interface"] = self.resolved_interface;
    dict["resolved_body"] = self.resolved_body;
    return dict;
}

Dictionary to_dictionary(GDScriptParser::ConstantNode &self) {
    Dictionary dict = AssignableNode::to_dictionary();
    dict["doc_data"] = self.doc_data.to_dictionary();
    return dict;
}

Dictionary to_dictionary(GDScriptParser::DictionaryNode::Pair &self) {
    Dictionary dict;
    if (self.key != nullptr) {
        dict["key"] = self.key->to_dictionary();
    }
    if (self.value != nullptr) {
        dict["value"] = self.value->to_dictionary();
    }
    return dict;
}

Dictionary to_dictionary(GDScriptParser::DictionaryNode &self) {
    Dictionary dict = ExpressionNode::to_dictionary();

    dict["style"] = self.style;

    Array dict_elements;
    dict["elements"] = dict_elements;
    for (const GDScriptParser::DictionaryNode::Pair &pair : self.elements) {
        dict_elements.push_back(pair.to_dictionary());
    }
    return dict;
}

Dictionary to_dictionary(GDScriptParser::ForNode &self) {
    Dictionary dict = Node::to_dictionary();
    if (self.variable != nullptr) {
        dict["variable"] = self.variable->to_dictionary();
    }
    if (self.datatype_specifier != nullptr) {
        dict["datatype_specifier"] = self.datatype_specifier->to_dictionary();
    }
    dict["use_conversion_assign"] = self.use_conversion_assign;
    if (self.list != nullptr) {
        dict["list"] = self.list->to_dictionary();
    }
    if (self.loop != nullptr) {
        dict["loop"] = self.loop->to_dictionary();
    }
    return dict;
}

Dictionary to_dictionary(GDScriptParser::FunctionNode &self) {
    Dictionary dict = Node::to_dictionary();
    if (self.identifier != nullptr) {
        dict["identifier"] = self.identifier->to_dictionary();
    }

    Array dict_parameters;
    dict["parameters"] = dict_parameters;
    for (const GDScriptParser::ParameterNode *parameter : self.parameters) {
        if (parameter != nullptr) {
            dict_parameters.push_back(parameter->to_dictionary());
        }
    }
    Dictionary dict_parameters_indices;
    dict["parameters_indices"] = dict_parameters_indices;
    for (const KeyValue<StringName, int> &kv : self.parameters_indices) {
        dict_parameters_indices[(String)kv.key] = kv.value;
    }

    if (self.return_type != nullptr) {
        dict["return_type"] = self.return_type->to_dictionary();
    }
    if (self.body != nullptr) {
        dict["body"] = self.body->to_dictionary();
    }

    dict["is_static"] = self.is_static;
    dict["is_coroutine"] = self.is_coroutine;
    dict["rpc_config"] = self.rpc_config;

    dict["info"] = self.info.to_dictionary();

    ////if(self.source_lambda != nullptr)
    ////dict["source_lambda"] = self.source_lambda->to_dictionary();

    Array dict_default_arg_values;
    dict["default_arg_values"] = dict_default_arg_values;
    for (const Variant &default_arg_value : self.default_arg_values) {
        dict_default_arg_values.push_back(default_arg_value);
    }

    dict["doc_data"] = self.doc_data.to_dictionary();
    dict["min_local_doc_line"] = self.min_local_doc_line;

    dict["resolved_signature"] = self.resolved_signature;
    dict["resolved_body"] = self.resolved_body;

    return dict;
}

Dictionary to_dictionary(GDScriptParser::GetNodeNode &self) {
    Dictionary dict = ExpressionNode::to_dictionary();
    dict["full_path"] = self.full_path;
    dict["use_dollar"] = self.use_dollar;
    return dict;
}

Dictionary to_dictionary(GDScriptParser::IdentifierNode &self) {
    Dictionary dict = ExpressionNode::to_dictionary();

    dict["name"] = self.name;
    ////if(self.suite != nullptr)
    ////dict["suite"] = self.suite->to_dictionary();
    dict["source"] = self.source;

    switch (self.source) {
        case GDScriptParser::IdentifierNode::Source::FUNCTION_PARAMETER:
            ////if(self.parameter_source != nullptr)
            ////dict["parameter_source"] = self.parameter_source->to_dictionary();
            break;
        case GDScriptParser::IdentifierNode::Source::LOCAL_ITERATOR:
        case GDScriptParser::IdentifierNode::Source::LOCAL_BIND:
            ////if(self.bind_source != nullptr)
            ////dict["bind_source"] = self.bind_source->to_dictionary();
            break;
        case GDScriptParser::IdentifierNode::Source::LOCAL_VARIABLE:
        case GDScriptParser::IdentifierNode::Source::MEMBER_VARIABLE:
        case GDScriptParser::IdentifierNode::Source::STATIC_VARIABLE:
        case GDScriptParser::IdentifierNode::Source::INHERITED_VARIABLE:
            ////if(self.variable_source != nullptr)
            ////dict["variable_source"] = self.variable_source->to_dictionary();
            break;
        case GDScriptParser::IdentifierNode::Source::LOCAL_CONSTANT:
        case GDScriptParser::IdentifierNode::Source::MEMBER_CONSTANT:
            ////if(self.constant_source != nullptr)
            ////dict["constant_source"] = self.constant_source->to_dictionary();
            break;
        case GDScriptParser::IdentifierNode::Source::MEMBER_SIGNAL:
            ////if(self.signal_source != nullptr)
            ////dict["signal_source"] = self.signal_source->to_dictionary();
            break;
        case GDScriptParser::IdentifierNode::Source::MEMBER_FUNCTION:
            ////if(self.function_source != nullptr)
            ////dict["function_source"] = self.function_source->to_dictionary();
            break;
        case GDScriptParser::IdentifierNode::Source::UNDEFINED_SOURCE:
        case GDScriptParser::IdentifierNode::Source::MEMBER_CLASS:
            break;
    }
    dict["function_source_is_static"] = self.function_source_is_static;
    ////if(self.source_function != nullptr)
    ////dict["source_function"] = self.source_function->to_dictionary();

    dict["usages"] = self.usages;
    return dict;
}

Dictionary to_dictionary(GDScriptParser::IfNode &self) {
    Dictionary dict = Node::to_dictionary();
    if (self.condition != nullptr) {
        dict["condition"] = self.condition->to_dictionary();
    }
    if (self.true_block != nullptr) {
        dict["true_block"] = self.true_block->to_dictionary();
    }
    if (self.false_block != nullptr) {
        dict["false_block"] = self.false_block->to_dictionary();
    }
    return dict;
}

Dictionary to_dictionary(GDScriptParser::LambdaNode &self) {
    Dictionary dict = ExpressionNode::to_dictionary();

    if (self.function != nullptr) {
        dict["function"] = self.function->to_dictionary();
    }
    ////if(self.parent_function != nullptr)
    ////dict["parent_function"] = self.parent_function->to_dictionary();
    ////if(self.parent_lambda != nullptr)
    ////dict["parent_lambda"] = self.parent_lambda->to_dictionary();

    Array dict_captures;
    dict["captures"] = dict_captures;
    for (const GDScriptParser::IdentifierNode *capture : self.captures) {
        if (capture != nullptr) {
            dict_captures.append(capture->to_dictionary());
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

Dictionary to_dictionary(GDScriptParser::LiteralNode &self) {
    Dictionary dict = ExpressionNode::to_dictionary();
    dict["value"] = self.value;
    return dict;
}

Dictionary to_dictionary(GDScriptParser::MatchNode &self) {
    Dictionary dict = Node::to_dictionary();
    if (self.test != nullptr) {
        dict["test"] = self.test->to_dictionary();
    }

    Array dict_branches;
    dict["branches"] = dict_branches;
    for (const GDScriptParser::MatchBranchNode *branch : self.branches) {
        if (branch != nullptr) {
            dict_branches.push_back(branch->to_dictionary());
        }
    }
    return dict;
}

Dictionary to_dictionary(GDScriptParser::MatchBranchNode &self) {
    Dictionary dict = Node::to_dictionary();

    Array dict_patterns;
    dict["patterns"] = dict_patterns;
    for (const GDScriptParser::PatternNode *pattern : self.patterns) {
        if (pattern != nullptr) {
            dict_patterns.append(pattern->to_dictionary());
        }
    }
    if (self.block != nullptr) {
        dict["block"] = self.block->to_dictionary();
    }
    dict["has_wildcard"] = self.has_wildcard;
    if (self.guard_body != nullptr) {
        dict["guard_body"] = self.guard_body->to_dictionary();
    }
    return dict;
}

Dictionary to_dictionary(GDScriptParser::PatternNode::Pair &self) {
    Dictionary dict;
    if (self.key != nullptr) {
        dict["key"] = self.key->to_dictionary();
    }
    if (self.value_pattern != nullptr) {
        dict["value_pattern"] = self.value_pattern->to_dictionary();
    }
    return dict;
}

Dictionary to_dictionary(GDScriptParser::PatternNode  &self) {
    Dictionary dict = Node::to_dictionary();
    dict["pattern_type"] = self.pattern_type;

    switch (self.pattern_type) {
        case GDScriptParser::PatternNode::Type::PT_LITERAL:
            if (self.literal != nullptr) {
                dict["literal"] = self.literal->to_dictionary();
            }
            break;
        case GDScriptParser::PatternNode::Type::PT_EXPRESSION:
            if (self.expression != nullptr) {
                dict["expression"] = self.expression->to_dictionary();
            }
            break;
        case GDScriptParser::PatternNode::Type::PT_BIND:
            if (self.bind != nullptr) {
                dict["bind"] = self.bind->to_dictionary();
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
            dict_array.append(pattern->to_dictionary());
        }
    }

    dict["rest_used"] = self.rest_used;

    Array dict_dictionary;
    dict["dictionary"] = dict_dictionary;
    for (const GDScriptParser::PatternNode::Pair &pair : self.dictionary) {
        dict_dictionary.append(pair.to_dictionary());
    }

    Dictionary dict_binds;
    dict["binds"] = dict_binds;
    for (const KeyValue<StringName, GDScriptParser::IdentifierNode *> &kv : self.binds) {
        if (kv.value != nullptr) {
            dict_binds[(String)kv.key] = kv.value->to_dictionary();
        }
    }

    return dict;
}

Dictionary to_dictionary(GDScriptParser::PreloadNode &self) {
    Dictionary dict = ExpressionNode::to_dictionary();
    if (self.path != nullptr) {
        dict["path"] = self.path->to_dictionary();
    }
    dict["resolved_path"] = self.resolved_path;
    //if(self.resource.is_valid())
    //dict["resource"] = self.resource->to_dictionary();
    return dict;
}

Dictionary to_dictionary(GDScriptParser::ReturnNode &self) {
    Dictionary dict = Node::to_dictionary();
    if (self.return_value != nullptr) {
        dict["return_value"] = self.return_value->to_dictionary();
    }
    dict["void_return"] = self.void_return;
    return dict;
}

Dictionary to_dictionary(GDScriptParser::SelfNode &self) {
    Dictionary dict = ExpressionNode::to_dictionary();
    ////if(self.current_class != nullptr)
    ////dict["current_class"] = self.current_class->to_dictionary();
    return dict;
}

Dictionary to_dictionary(GDScriptParser::SignalNode &self) {
    Dictionary dict = Node::to_dictionary();
    if (self.identifier != nullptr) {
        dict["identifier"] = self.identifier->to_dictionary();
    }
    Array dict_parameters;
    dict["parameters"] = dict_parameters;
    for (const GDScriptParser::ParameterNode *parameter : self.parameters) {
        if (parameter != nullptr) {
            dict_parameters.append(parameter->to_dictionary());
        }
    }
    Dictionary dict_parameters_indices;
    dict["parameters_indices"] = dict_parameters_indices;
    for (const KeyValue<StringName, int> &kv : self.parameters_indices) {
        dict_parameters_indices[(String)kv.key] = kv.value;
    }
    dict["method_info"] = self.method_info.to_dictionary();
    dict["doc_data"] = self.doc_data.to_dictionary();
    dict["usages"] = self.usages;
    return dict;
}

Dictionary to_dictionary(GDScriptParser::SubscriptNode &self) {
    Dictionary dict = ExpressionNode::to_dictionary();
    if (self.base != nullptr) {
        dict["base"] = self.base->to_dictionary();
    }
    if (self.is_attribute) {
        if (self.attribute != nullptr) {
            dict["attribute"] = self.attribute->to_dictionary();
        }
    } else {
        if (self.index != nullptr) {
            dict["index"] = self.index->to_dictionary();
        }
    }
    dict["is_attribute"] = self.is_attribute;
    return dict;
}

#endif // TOOLS_ENABLED
/**************************************************************************/
/*  gdscript_reconstructor.cpp                                            */
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

#include "gdscript_reconstructor.h"

const char *GDScriptReconstructor::OPStrings[] = {
	"call",
	"parent_call",
	"yield",
	"is",
	"is_builtin",
	"index",
	"index_named",
	"neg",
	"pos",
	"not",
	"bit_invert",
	"in",
	"==",
	"!=",
	"<",
	"<=",
	">",
	">=",
	"and",
	"or",
	"+",
	"-",
	"*",
	"/",
	"%",
	"<<",
	">>",
	"=",
	"=",
	"+=",
	"-=",
	"*=",
	"/=",
	"%=",
	"<<=",
	">>=",
	"&=",
	"|=",
	"~=",
	"&",
	"|",
	"~",
	"?",
	":"
};

bool GDScriptReconstructor::output_control_flow(const GDScriptParser::ControlFlowNode *p_node, Output out) const {
	bool colon = false;
	bool brackets = false;
	bool manual = false;
	String sz;

	switch (p_node->cf_type) {
		default:
			break;
		case GDScriptParser::ControlFlowNode::CF_IF: {
			sz += draw_keyword("if");
			// brackets = true;
			colon = true;
		} break;
		case GDScriptParser::ControlFlowNode::CF_FOR: {
			sz += draw_keyword("for");
			colon = true;
			manual = true;
			sz += " ";
			output_node(p_node->arguments[0], Output(out.indent, &sz));
			sz += " in range " + draw_operator("(");
			output_node(p_node->arguments[1], Output(out.indent, &sz));
			sz += draw_operator("):");
		} break;
		case GDScriptParser::ControlFlowNode::CF_WHILE: {
			sz += draw_keyword("while");
			colon = true;
			brackets = true;
		} break;
		case GDScriptParser::ControlFlowNode::CF_BREAK: {
			sz += draw_keyword("break");
		} break;
		case GDScriptParser::ControlFlowNode::CF_CONTINUE: {
			sz += draw_keyword("continue");
		} break;
		case GDScriptParser::ControlFlowNode::CF_RETURN: {
			sz += draw_keyword("return");
		} break;
		case GDScriptParser::ControlFlowNode::CF_MATCH: {
			sz += draw_keyword("match");
		} break;
	}

	if (!manual) {
		if (brackets) {
			sz += " (";
		} else {
			sz += " ";
		}
		for (int n = 0; n < p_node->arguments.size(); n++) {
			if (n != 0) {
				sz += " ";
			}
			output_node(p_node->arguments[n], Output(out.indent, &sz));
		}
		if (brackets) {
			sz += ")";
		}

		if (colon) {
			sz += ":";
		}
	}

	if (p_node->body) {
		output_block(p_node->body, Output(out.indent + 1, &sz));
	}

	if (p_node->body_else) {
		sz += "\n";
		output_indent(Output(out.indent, &sz), true);

		// Detect else if.
		String else_body;
		output_block(p_node->body_else, Output(out.indent + 1, &else_body));

		if (else_body.begins_with(draw_keyword("if"))) {
			sz += draw_keyword("else ");
			else_body = "";
			data.output_tab_required = false;
			output_block(p_node->body_else, Output(out.indent, &else_body));
		} else {
			sz += draw_keyword("else:");
		}
		sz += else_body;
	}

	if (p_node->match) {
		sz += String(" ");
		output_node(p_node->match->val_to_match, Output(out.indent, &sz));
		sz += String(":");
		for (int n = 0; n < p_node->match->branches.size(); n++) {
			GDScriptParser::PatternBranchNode *branch = p_node->match->branches[n];

			for (int p = 0; p < branch->patterns.size(); p++) {
				GDScriptParser::PatternNode *pat = branch->patterns[p];

				if (pat->pt_type == GDScriptParser::PatternNode::PT_CONSTANT) {
					sz += String("\n");
					data.output_tab_required = true;
					output_node(pat->constant, Output(out.indent + 1, &sz));
					sz += String(":");
				}
			}

			output_block(branch->body, Output(out.indent + 2, &sz));
		}
	}

	*out.text += sz;
	return true;
}

bool GDScriptReconstructor::output_block(const GDScriptParser::BlockNode *p_node, Output out) const {
	for (int n = 0; n < p_node->statements.size(); n++) {
		output_node(p_node->statements[n], out);
	}

	return true;
}

bool GDScriptReconstructor::output_function(const GDScriptParser::FunctionNode *p_node, Output out) const {
	output_indent(out);

	String sz;
	if (p_node->_static) {
		sz += draw_keyword("static ");
	}
	if (p_node->_inline_func) {
		sz += draw_keyword("inline ");
	}
	sz += draw_keyword("func ");
	sz += String(p_node->name) + draw_operator("(");

	for (int n = 0; n < p_node->arguments.size(); n++) {
		sz += p_node->arguments[n];
		if (p_node->argument_types.size()) {
			helper_data_type_to_string(p_node->argument_types[n], sz);
		}
		if (n != p_node->arguments.size() - 1) {
			sz += draw_keyword(", ");
		}
	}

	sz += draw_operator("):");

	output_block(p_node->body, Output(out.indent + 1, &sz));
	sz += "\n";
	data.output_tab_required = true;

	*out.text += sz;

	return true;
}

bool GDScriptReconstructor::output_class(const GDScriptParser::ClassNode *p_node, Output out) const {
	String sz;
	output_indent(Output(out.indent, &sz));

	for (int n = 0; n < p_node->extends_class.size(); n++) {
		output_indent(Output(out.indent, &sz));
		sz += draw_keyword("extends ") + p_node->extends_class[n] + "\n";
		data.output_tab_required = true;
	}
	if (p_node->extends_class.size()) {
		sz += "\n";
		data.output_tab_required = true;
	}

	bool has_constants = false;
	for (Map<StringName, GDScriptParser::ClassNode::Constant>::Element *E = p_node->constant_expressions.front(); E; E = E->next()) {
		GDScriptParser::ClassNode::Constant &c = E->get();
		output_indent(Output(out.indent, &sz));
		sz += draw_keyword("const ") + c.name + draw_operator(" = ");
		output_node(c.expression, Output(out.indent, &sz));
		sz += "\n";
		has_constants = true;
		data.output_tab_required = true;
	}
	if (has_constants) {
		sz += "\n";
		data.output_tab_required = true;
	}

	for (int n = 0; n < p_node->variables.size(); n++) {
		const GDScriptParser::ClassNode::Member &var = p_node->variables[n];
		output_indent(Output(out.indent, &sz));
		sz += draw_keyword("var ");
		output_node(var.initial_assignment, Output(out.indent, &sz));
		sz += "\n";
		data.output_tab_required = true;
	}
	if (p_node->variables.size()) {
		sz += "\n";
		data.output_tab_required = true;
	}

	for (int n = 0; n < p_node->subclasses.size(); n++) {
		output_indent(Output(out.indent, &sz));
		data.output_tab_required = true;
		sz += draw_keyword("class ") + p_node->subclasses[n]->name + draw_operator(":") + "\n";
		String sz_class;
		output_class(p_node->subclasses[n], Output(out.indent + 1, &sz_class));
		sz += sz_class;
		data.output_tab_required = true;
		sz += "\n";
	}

	for (int n = 0; n < p_node->static_functions.size(); n++) {
		output_function(p_node->static_functions[n], Output(out.indent, &sz));
	}

	for (int n = 0; n < p_node->functions.size(); n++) {
		output_function(p_node->functions[n], Output(out.indent, &sz));
	}

	*out.text += sz;
	data.output_tab_required = true;

	return true;
}

void GDScriptReconstructor::apply_emphasis(const GDScriptParser::Node *p_node, String &r_text) const {
	if (p_node == data.emphasis_node) {
		r_text += draw_bold(">>> ");
	}
}

bool GDScriptReconstructor::output_operator(const GDScriptParser::OperatorNode *p_node, Output out, bool p_apply_parent_brackets) const {
	bool binary_operator = true;
	bool commas = false;
	int first_arg = 0;
	bool brackets = false;
	bool add_child_brackets = true;
	bool prepend_operator = false;

	String sz;

	// Add brackets if we are inside the first level.
	brackets = p_apply_parent_brackets;

	switch (p_node->op) {
		default: {
		} break;
		case GDScriptParser::OperatorNode::OP_GREATER:
		case GDScriptParser::OperatorNode::OP_GREATER_EQUAL:
		case GDScriptParser::OperatorNode::OP_LESS:
		case GDScriptParser::OperatorNode::OP_LESS_EQUAL:
		case GDScriptParser::OperatorNode::OP_EQUAL:
		case GDScriptParser::OperatorNode::OP_NOT_EQUAL:
		case GDScriptParser::OperatorNode::OP_ASSIGN: {
			add_child_brackets = false;
		} break;
		case GDScriptParser::OperatorNode::OP_NEG: {
			prepend_operator = true;
		} break;
		case GDScriptParser::OperatorNode::OP_INIT_ASSIGN: {
			output_node(p_node->arguments[0], Output(out.indent, &sz));
			ERR_FAIL_COND_V(p_node->arguments[0]->type != GDScriptParser::Node::TYPE_IDENTIFIER, true);
			helper_data_type_to_string(((GDScriptParser::IdentifierNode *)p_node->arguments[0])->datatype, sz);
			sz += draw_operator(" = ");
			first_arg = 1;
		} break;
		case GDScriptParser::OperatorNode::OP_CALL: {
			binary_operator = false;
			commas = true;
			brackets = true;

			switch (p_node->arguments[0]->type) {
				default: {
					output_node(p_node->arguments[0], Output(out.indent, &sz));
					sz += draw_operator(".");
					output_node(p_node->arguments[1], Output(out.indent, &sz));
					first_arg = 2;
				} break;
				case GDScriptParser::Node::TYPE_TYPE: {
					first_arg = 1;
				} break;
				case GDScriptParser::Node::TYPE_SELF: {
					output_node(p_node->arguments[1], Output(out.indent, &sz));
					first_arg = 2;
				} break;
				case GDScriptParser::Node::TYPE_BUILT_IN_FUNCTION: {
					output_node(p_node->arguments[0], Output(out.indent, &sz));
					first_arg = 1;
				} break;
			}

		} break;
	}

	if (brackets) {
		sz += draw_operator("(");
	}

	bool close_index = false;

	for (int n = first_arg; n < p_node->arguments.size(); n++) {
		bool is_last = n == (p_node->arguments.size() - 1);

		if (!prepend_operator) {
			output_node(p_node->arguments[n], Output(out.indent, &sz), add_child_brackets);
		}

		// In a variable declaration, we should output the type info if available.
		if (data.var_declaration) {
			helper_data_type_to_string(p_node->arguments[n]->get_datatype(), sz);
			data.var_declaration = false;
		}

		if (binary_operator && (n == 0)) {
			String sz_op;
			switch (p_node->op) {
				default: {
					sz_op += String(" ") + OPStrings[p_node->op] + " ";
				} break;
				case GDScriptParser::OperatorNode::Operator::OP_INDEX_NAMED: {
					sz_op += ".";
				} break;
				case GDScriptParser::OperatorNode::Operator::OP_INDEX: {
					sz_op += "[";
					close_index = true;
				} break;
				case GDScriptParser::OperatorNode::Operator::OP_NEG: {
					sz_op += "-";
				} break;
			}
			sz += draw_operator(sz_op);
		}

		if (prepend_operator) {
			output_node(p_node->arguments[n], Output(out.indent, &sz), add_child_brackets);
		}

		if (commas && !is_last) {
			sz += draw_operator(", ");
		}
	}
	if (close_index) {
		sz += draw_operator("]");
	}
	if (brackets) {
		sz += draw_operator(")");
	}

	*out.text += sz;
	return true;
}

bool GDScriptReconstructor::debug_output_control_flow(const GDScriptParser::ControlFlowNode *p_node, Output out) const {
	bool colon = false;
	bool brackets = false;
	String sz;

	switch (p_node->cf_type) {
		default:
			break;
		case GDScriptParser::ControlFlowNode::CF_IF: {
			sz += "if";
			colon = true;
		} break;
		case GDScriptParser::ControlFlowNode::CF_FOR: {
			sz += "for";
			colon = true;
		} break;
		case GDScriptParser::ControlFlowNode::CF_WHILE: {
			sz += "while";
			colon = true;
			brackets = true;
		} break;
		case GDScriptParser::ControlFlowNode::CF_BREAK: {
			sz += "break";
		} break;
		case GDScriptParser::ControlFlowNode::CF_CONTINUE: {
			sz += "continue";
		} break;
		case GDScriptParser::ControlFlowNode::CF_RETURN: {
			sz += "return";
		} break;
		case GDScriptParser::ControlFlowNode::CF_MATCH: {
			sz += "match";
			colon = true;
		} break;
	}

	if (brackets) {
		sz += " (";
	} else {
		sz += " ";
	}
	for (int n = 0; n < p_node->arguments.size(); n++) {
		if (n != 0) {
			sz += " ";
		}
		output_node(p_node->arguments[n], Output(out.indent, &sz));
	}
	if (brackets) {
		sz += ")";
	}

	if (colon) {
		sz += ":";
	}

	if (p_node->body) {
		output_block(p_node->body, Output(out.indent + 1, &sz));
	}

	if (p_node->body_else) {
		sz += "\n";
		output_indent(Output(out.indent, &sz), true);

		// Detect else if.
		String else_body;
		output_block(p_node->body_else, Output(out.indent + 1, &else_body));

		if (else_body.begins_with("if ")) {
			sz += "else ";
			else_body = "";
			data.output_tab_required = false;
			output_block(p_node->body_else, Output(out.indent, &else_body));
		} else {
			sz += "else:";
		}
		sz += else_body;
	}

	*out.text += sz;
	return true;
}

bool GDScriptReconstructor::debug_output_operator(const GDScriptParser::OperatorNode *p_node, Output out) const {
	bool binary_operator = true;
	bool commas = false;
	int first_arg = 0;
	bool brackets = false;
	bool prepend_operator = false;

	String sz;

	switch (p_node->op) {
		default: {
		} break;
		case GDScriptParser::OperatorNode::OP_NEG: {
			prepend_operator = true;
		} break;
		case GDScriptParser::OperatorNode::OP_CALL: {
			binary_operator = false;
			commas = true;
			brackets = true;

			switch (p_node->arguments[0]->type) {
				default: {
					debug_output_node(p_node->arguments[0], Output(out.indent, &sz));
					sz += ".";
					debug_output_node(p_node->arguments[1], Output(out.indent, &sz));
					first_arg = 2;
				} break;
				case GDScriptParser::Node::TYPE_TYPE: {
					first_arg = 1;
				} break;
				case GDScriptParser::Node::TYPE_SELF: {
					debug_output_node(p_node->arguments[1], Output(out.indent, &sz));
					first_arg = 2;
				} break;
				case GDScriptParser::Node::TYPE_BUILT_IN_FUNCTION: {
					debug_output_node(p_node->arguments[0], Output(out.indent, &sz));
					first_arg = 1;
				} break;
			}

		} break;
	}

	if (brackets) {
		sz += "(";
	}

	bool close_index = false;

	for (int n = first_arg; n < p_node->arguments.size(); n++) {
		bool is_last = n == (p_node->arguments.size() - 1);

		if (!prepend_operator) {
			output_node(p_node->arguments[n], Output(out.indent, &sz));
		}

		if (binary_operator && (n == 0)) {
			switch (p_node->op) {
				default: {
					sz += String(" ") + OPStrings[p_node->op] + " ";
				} break;
				case GDScriptParser::OperatorNode::Operator::OP_INDEX_NAMED: {
					sz += ".";
				} break;
				case GDScriptParser::OperatorNode::Operator::OP_INDEX: {
					sz += "[";
					close_index = true;
				} break;
				case GDScriptParser::OperatorNode::Operator::OP_NEG: {
					sz += "-";
				} break;
			}
		}

		if (prepend_operator) {
			debug_output_node(p_node->arguments[n], Output(out.indent, &sz));
		}

		if (commas && !is_last) {
			sz += ", ";
		}
	}
	if (close_index) {
		sz += "]";
	}
	if (brackets) {
		sz += ")";
	}

	*out.text += sz;
	return true;
}

bool GDScriptReconstructor::debug_output_node(const GDScriptParser::Node *p_node, Output out) const {
	DEV_ASSERT(p_node);
	output_indent(out);

	String sz;

	switch (p_node->type) {
		default: {
			sz += "default";
		} break;
		case GDScriptParser::Node::TYPE_TYPE: {
			const GDScriptParser::TypeNode *tp = (const GDScriptParser::TypeNode *)p_node;
			sz += "TYPE " + Variant::get_type_name(tp->vtype);
		} break;
		case GDScriptParser::Node::TYPE_BLOCK: {
			const GDScriptParser::BlockNode *block = (const GDScriptParser::BlockNode *)p_node;
			for (int n = 0; n < block->statements.size(); n++) {
				debug_output_node(block->statements[n], out);
			}
		} break;
		case GDScriptParser::Node::TYPE_CLASS: {
			const GDScriptParser::ClassNode *cl = (const GDScriptParser::ClassNode *)p_node;
			for (int n = 0; n < cl->functions.size(); n++) {
				debug_output_node(cl->functions[n], out);
			}
		} break;
		case GDScriptParser::Node::TYPE_IDENTIFIER: {
			const GDScriptParser::IdentifierNode *ident = (const GDScriptParser::IdentifierNode *)p_node;
			sz += ident->name;
		} break;
		case GDScriptParser::Node::TYPE_OPERATOR: {
			return debug_output_operator((const GDScriptParser::OperatorNode *)p_node, out);
		} break;
		case GDScriptParser::Node::TYPE_CONSTANT: {
			const GDScriptParser::ConstantNode *cons = (const GDScriptParser::ConstantNode *)p_node;
			const Variant &val = cons->value;
			sz += "CONSTANT ";
			switch (val.get_type()) {
				default: {
					sz += String(cons->value);
				} break;
				case Variant::STRING: {
					sz += "\"" + String(cons->value) + "\"";
				} break;
				case Variant::NIL: {
					sz += "null";
				} break;
				case Variant::BOOL: {
					sz += cons->value ? "true" : "false";
				} break;
			}
		} break;
		case GDScriptParser::Node::TYPE_LOCAL_VAR: {
			// const GDScriptParser::LocalVarNode *var = (const GDScriptParser::LocalVarNode *)p_node;
			sz += "LOCAL_VAR ";
		} break;
		case GDScriptParser::Node::TYPE_FUNCTION: {
			const GDScriptParser::FunctionNode *func = (const GDScriptParser::FunctionNode *)p_node;
			sz += "FUNCTION " + func->name;
			debug_output_node(func->body, out);
		} break;
		case GDScriptParser::Node::TYPE_BUILT_IN_FUNCTION: {
			const GDScriptParser::BuiltInFunctionNode *func = (const GDScriptParser::BuiltInFunctionNode *)p_node;
			const char *func_name = GDScriptFunctions::get_func_name(func->function);
			sz += String("BUILTIN ") + func_name;
		} break;
		case GDScriptParser::Node::TYPE_SELF: {
			sz += "SELF";
		} break;
		case GDScriptParser::Node::TYPE_ASSERT: {
			sz += "ASSERT";
		} break;
		case GDScriptParser::Node::TYPE_NEWLINE: {
			sz += " ...\n";
			data.output_tab_required = true;
		} break;
		case GDScriptParser::Node::TYPE_CONTROL_FLOW: {
			return debug_output_control_flow((const GDScriptParser::ControlFlowNode *)p_node, out);
		} break;
	}

	*out.text += sz;
	return true;

	return true;
}

bool GDScriptReconstructor::output_node(const GDScriptParser::Node *p_node, Output out, bool p_apply_parent_brackets) const {
	DEV_ASSERT(p_node);
	output_indent(out);

	apply_emphasis(p_node, *out.text);
	String sz;

	switch (p_node->type) {
		default: {
			;
		} break;
		case GDScriptParser::Node::TYPE_INLINE_BLOCK: {
			GDScriptParser::InlineBlockNode *block = (GDScriptParser::InlineBlockNode *)p_node;
			for (int n = 0; n < block->statements.size(); n++) {
				output_node(block->statements[n], out);
			}
		} break;
		case GDScriptParser::Node::TYPE_DICTIONARY: {
			GDScriptParser::DictionaryNode *dic = (GDScriptParser::DictionaryNode *)p_node;
			sz += draw_operator("{");
			for (int n = 0; n < dic->elements.size(); n++) {
				data.output_tab_required = true;
				sz += "\n";
				output_node(dic->elements[n].key, Output(out.indent + 1, &sz));
				sz += draw_operator(" : ");
				output_node(dic->elements[n].value, Output(out.indent + 1, &sz));
				if (n && (n != dic->elements.size() - 1)) {
					sz += draw_operator(", ");
				}
			}
			sz += String("\n") + String("\t").repeat(out.indent) + draw_operator("}");
		} break;
		case GDScriptParser::Node::TYPE_ARRAY: {
			GDScriptParser::ArrayNode *arr = (GDScriptParser::ArrayNode *)p_node;
			sz += draw_operator("[");
			for (int n = 0; n < arr->elements.size(); n++) {
				output_node(arr->elements[n], Output(out.indent, &sz));
				if (n != arr->elements.size() - 1) {
					sz += draw_operator(", ");
				}
			}
			sz += draw_operator("]");
		} break;
		case GDScriptParser::Node::TYPE_TYPE: {
			const GDScriptParser::TypeNode *tp = (const GDScriptParser::TypeNode *)p_node;
			sz += draw_operator(" : ") + draw_keyword(Variant::get_type_name(tp->vtype));
		} break;
		case GDScriptParser::Node::TYPE_BLOCK: {
			return output_block((const GDScriptParser::BlockNode *)p_node, out);
		} break;
		case GDScriptParser::Node::TYPE_CLASS: {
			return output_class((const GDScriptParser::ClassNode *)p_node, out);
		} break;
		case GDScriptParser::Node::TYPE_IDENTIFIER: {
			const GDScriptParser::IdentifierNode *ident = (const GDScriptParser::IdentifierNode *)p_node;
			sz += draw_identifier(ident->name);
		} break;
		case GDScriptParser::Node::TYPE_OPERATOR: {
			return output_operator((const GDScriptParser::OperatorNode *)p_node, out, p_apply_parent_brackets);
		} break;
		case GDScriptParser::Node::TYPE_CONSTANT: {
			const GDScriptParser::ConstantNode *cons = (const GDScriptParser::ConstantNode *)p_node;
			sz += variant_to_string(cons->value);
		} break;
		case GDScriptParser::Node::TYPE_LOCAL_VAR: {
			// const GDScriptParser::LocalVarNode *var = (const GDScriptParser::LocalVarNode *)p_node;
			sz += draw_keyword("var ");
			data.var_declaration = true;
		} break;
		case GDScriptParser::Node::TYPE_FUNCTION: {
			const GDScriptParser::FunctionNode *func = (const GDScriptParser::FunctionNode *)p_node;
			output_function(func, out);
		} break;
		case GDScriptParser::Node::TYPE_BUILT_IN_FUNCTION: {
			const GDScriptParser::BuiltInFunctionNode *func = (const GDScriptParser::BuiltInFunctionNode *)p_node;
			const char *func_name = GDScriptFunctions::get_func_name(func->function);
			sz += func_name;
		} break;
		case GDScriptParser::Node::TYPE_SELF: {
			sz += draw_keyword("self");
		} break;
		case GDScriptParser::Node::TYPE_ASSERT: {
			sz += draw_keyword("assert ");
			const GDScriptParser::AssertNode *ass = (const GDScriptParser::AssertNode *)p_node;
			if (ass->condition) {
				output_node(ass->condition, Output(out.indent, &sz), true);
			}
		} break;
		case GDScriptParser::Node::TYPE_NEWLINE: {
#if 1
			sz += "\n";
#else
			sz += " ...\n";
#endif
			data.output_tab_required = true;
		} break;
		case GDScriptParser::Node::TYPE_CONTROL_FLOW: {
			return output_control_flow((const GDScriptParser::ControlFlowNode *)p_node, out);
		} break;
	}

	*out.text += sz;
	return true;
}

String GDScriptReconstructor::_variant_data_to_string(const Variant &p_val, bool p_style) {
	String sz = String(p_val);
	if (sz.begins_with("(") && sz.ends_with(")")) {
		int l = sz.length();
		if (l > 2) {
			sz = sz.substr(1, sz.length() - 2);
			sz = draw_operator("(") + draw_constant(sz) + draw_operator(")");
		} else {
			sz = draw_operator("()");
		}
	} else {
		if (p_style) {
			sz = draw_constant(sz);
		}
	}
	return sz;
}

String GDScriptReconstructor::variant_to_string(const Variant &p_val, bool p_style) {
	String sz;
	switch (p_val.get_type()) {
		default: {
			sz = _variant_data_to_string(p_val, p_style);
		} break;
		case Variant::VECTOR2:
		case Variant::RECT2:
		case Variant::TRANSFORM2D:
		case Variant::VECTOR3:
		case Variant::PLANE:
		case Variant::AABB:
		case Variant::QUAT:
		case Variant::BASIS:
		case Variant::TRANSFORM:
		case Variant::COLOR:
		case Variant::_RID:
		case Variant::OBJECT:
		case Variant::NODE_PATH:
		case Variant::POOL_BYTE_ARRAY:
		case Variant::POOL_INT_ARRAY:
		case Variant::POOL_REAL_ARRAY:
		case Variant::POOL_STRING_ARRAY:
		case Variant::POOL_VECTOR2_ARRAY:
		case Variant::POOL_VECTOR3_ARRAY:
		case Variant::POOL_COLOR_ARRAY: {
			sz = Variant::get_type_name(p_val.get_type());
			sz += _variant_data_to_string(p_val, p_style);
		} break;
		case Variant::NIL: {
			sz = p_style ? draw_constant("null") : "null";
		} break;
		case Variant::STRING: {
			sz = "\"" + String(p_val) + "\"";
			if (p_style) {
				sz = draw_constant(sz);
			}
		} break;
		case Variant::BOOL: {
			sz = p_val.booleanize() ? "true" : "false";
			if (p_style) {
				sz = draw_constant(sz);
			}
		} break;
	}
	return sz;
}

void GDScriptReconstructor::output_indent(const Output &p_out, bool p_force) const {
	if (data.output_tab_required || p_force) {
		data.output_tab_required = false;
		*p_out.text += String("\t").repeat(p_out.indent);
	}
}

bool GDScriptReconstructor::debug_output(GDScriptParser &r_parser, String &r_text) {
	Output out(0, &r_text);
	debug_output_node(r_parser.get_parse_tree(), out);

	return true;
}

bool GDScriptReconstructor::output_branch(const GDScriptParser::Node *p_branch, String &r_text, const GDScriptParser::Node *p_emphasis_node) {
	data.emphasis_node = p_emphasis_node;
	Output out(0, &r_text);
	output_node(p_branch, out);
	return true;
}

bool GDScriptReconstructor::output(GDScriptParser &r_parser, String &r_text) {
	return output_branch(r_parser.get_parse_tree(), r_text);
}

/*************************************************************************/
/*  test_gdscript.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "test_gdscript.h"

#include "os/file_access.h"
#include "os/main_loop.h"
#include "os/os.h"

#ifdef GDSCRIPT_ENABLED

#include "modules/gdscript/gd_compiler.h"
#include "modules/gdscript/gd_parser.h"
#include "modules/gdscript/gd_script.h"
#include "modules/gdscript/gd_tokenizer.h"

namespace TestGDScript {

static void _print_indent(int p_ident, const String &p_text) {

	String txt;
	for (int i = 0; i < p_ident; i++) {
		txt += '\t';
	}

	print_line(txt + p_text);
}

static String _parser_extends(const GDParser::ClassNode *p_class) {

	String txt = "extends ";
	if (String(p_class->extends_file) != "") {
		txt += "\"" + p_class->extends_file + "\"";
		if (p_class->extends_class.size())
			txt += ".";
	}

	for (int i = 0; i < p_class->extends_class.size(); i++) {

		if (i != 0)
			txt += ".";

		txt += p_class->extends_class[i];
	}

	return txt;
}

static String _parser_expr(const GDParser::Node *p_expr) {

	String txt;
	switch (p_expr->type) {

		case GDParser::Node::TYPE_IDENTIFIER: {

			const GDParser::IdentifierNode *id_node = static_cast<const GDParser::IdentifierNode *>(p_expr);
			txt = id_node->name;
		} break;
		case GDParser::Node::TYPE_CONSTANT: {
			const GDParser::ConstantNode *c_node = static_cast<const GDParser::ConstantNode *>(p_expr);
			if (c_node->value.get_type() == Variant::STRING)
				txt = "\"" + String(c_node->value) + "\"";
			else
				txt = c_node->value;

		} break;
		case GDParser::Node::TYPE_SELF: {
			txt = "self";
		} break;
		case GDParser::Node::TYPE_ARRAY: {
			const GDParser::ArrayNode *arr_node = static_cast<const GDParser::ArrayNode *>(p_expr);
			txt += "[";
			for (int i = 0; i < arr_node->elements.size(); i++) {

				if (i > 0)
					txt += ", ";
				txt += _parser_expr(arr_node->elements[i]);
			}
			txt += "]";
		} break;
		case GDParser::Node::TYPE_DICTIONARY: {
			const GDParser::DictionaryNode *dict_node = static_cast<const GDParser::DictionaryNode *>(p_expr);
			txt += "{";
			for (int i = 0; i < dict_node->elements.size(); i++) {

				if (i > 0)
					txt += ", ";

				const GDParser::DictionaryNode::Pair &p = dict_node->elements[i];
				txt += _parser_expr(p.key);
				txt += ":";
				txt += _parser_expr(p.value);
			}
			txt += "}";
		} break;
		case GDParser::Node::TYPE_OPERATOR: {

			const GDParser::OperatorNode *c_node = static_cast<const GDParser::OperatorNode *>(p_expr);
			switch (c_node->op) {

				case GDParser::OperatorNode::OP_PARENT_CALL:
					txt += ".";
				case GDParser::OperatorNode::OP_CALL: {

					ERR_FAIL_COND_V(c_node->arguments.size() < 1, "");
					String func_name;
					const GDParser::Node *nfunc = c_node->arguments[0];
					int arg_ofs = 0;
					if (nfunc->type == GDParser::Node::TYPE_BUILT_IN_FUNCTION) {

						const GDParser::BuiltInFunctionNode *bif_node = static_cast<const GDParser::BuiltInFunctionNode *>(nfunc);
						func_name = GDFunctions::get_func_name(bif_node->function);
						arg_ofs = 1;
					} else if (nfunc->type == GDParser::Node::TYPE_TYPE) {

						const GDParser::TypeNode *t_node = static_cast<const GDParser::TypeNode *>(nfunc);
						func_name = Variant::get_type_name(t_node->vtype);
						arg_ofs = 1;
					} else {

						ERR_FAIL_COND_V(c_node->arguments.size() < 2, "");
						nfunc = c_node->arguments[1];
						ERR_FAIL_COND_V(nfunc->type != GDParser::Node::TYPE_IDENTIFIER, "");

						if (c_node->arguments[0]->type != GDParser::Node::TYPE_SELF)
							func_name = _parser_expr(c_node->arguments[0]) + ".";

						func_name += _parser_expr(nfunc);
						arg_ofs = 2;
					}

					txt += func_name + "(";

					for (int i = arg_ofs; i < c_node->arguments.size(); i++) {

						const GDParser::Node *arg = c_node->arguments[i];
						if (i > arg_ofs)
							txt += ", ";
						txt += _parser_expr(arg);
					}

					txt += ")";

				} break;
				case GDParser::OperatorNode::OP_INDEX: {

					ERR_FAIL_COND_V(c_node->arguments.size() != 2, "");

					//index with []
					txt = _parser_expr(c_node->arguments[0]) + "[" + _parser_expr(c_node->arguments[1]) + "]";

				} break;
				case GDParser::OperatorNode::OP_INDEX_NAMED: {

					ERR_FAIL_COND_V(c_node->arguments.size() != 2, "");

					txt = _parser_expr(c_node->arguments[0]) + "." + _parser_expr(c_node->arguments[1]);

				} break;
				case GDParser::OperatorNode::OP_NEG: {
					txt = "-" + _parser_expr(c_node->arguments[0]);
				} break;
				case GDParser::OperatorNode::OP_NOT: {
					txt = "not " + _parser_expr(c_node->arguments[0]);
				} break;
				case GDParser::OperatorNode::OP_BIT_INVERT: {
					txt = "~" + _parser_expr(c_node->arguments[0]);
				} break;
				case GDParser::OperatorNode::OP_PREINC: {
				} break;
				case GDParser::OperatorNode::OP_PREDEC: {
				} break;
				case GDParser::OperatorNode::OP_INC: {
				} break;
				case GDParser::OperatorNode::OP_DEC: {
				} break;
				case GDParser::OperatorNode::OP_IN: {
					txt = _parser_expr(c_node->arguments[0]) + " in " + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_EQUAL: {
					txt = _parser_expr(c_node->arguments[0]) + "==" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_NOT_EQUAL: {
					txt = _parser_expr(c_node->arguments[0]) + "!=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_LESS: {
					txt = _parser_expr(c_node->arguments[0]) + "<" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_LESS_EQUAL: {
					txt = _parser_expr(c_node->arguments[0]) + "<=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_GREATER: {
					txt = _parser_expr(c_node->arguments[0]) + ">" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_GREATER_EQUAL: {
					txt = _parser_expr(c_node->arguments[0]) + ">=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_AND: {
					txt = _parser_expr(c_node->arguments[0]) + " and " + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_OR: {
					txt = _parser_expr(c_node->arguments[0]) + " or " + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_ADD: {
					txt = _parser_expr(c_node->arguments[0]) + "+" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_SUB: {
					txt = _parser_expr(c_node->arguments[0]) + "-" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_MUL: {
					txt = _parser_expr(c_node->arguments[0]) + "*" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_DIV: {
					txt = _parser_expr(c_node->arguments[0]) + "/" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_MOD: {
					txt = _parser_expr(c_node->arguments[0]) + "%" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_SHIFT_LEFT: {
					txt = _parser_expr(c_node->arguments[0]) + "<<" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_SHIFT_RIGHT: {
					txt = _parser_expr(c_node->arguments[0]) + ">>" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_ASSIGN: {
					txt = _parser_expr(c_node->arguments[0]) + "=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_ASSIGN_ADD: {
					txt = _parser_expr(c_node->arguments[0]) + "+=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_ASSIGN_SUB: {
					txt = _parser_expr(c_node->arguments[0]) + "-=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_ASSIGN_MUL: {
					txt = _parser_expr(c_node->arguments[0]) + "*=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_ASSIGN_DIV: {
					txt = _parser_expr(c_node->arguments[0]) + "/=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_ASSIGN_MOD: {
					txt = _parser_expr(c_node->arguments[0]) + "%=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_ASSIGN_SHIFT_LEFT: {
					txt = _parser_expr(c_node->arguments[0]) + "<<=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_ASSIGN_SHIFT_RIGHT: {
					txt = _parser_expr(c_node->arguments[0]) + ">>=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_ASSIGN_BIT_AND: {
					txt = _parser_expr(c_node->arguments[0]) + "&=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_ASSIGN_BIT_OR: {
					txt = _parser_expr(c_node->arguments[0]) + "|=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_ASSIGN_BIT_XOR: {
					txt = _parser_expr(c_node->arguments[0]) + "^=" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_BIT_AND: {
					txt = _parser_expr(c_node->arguments[0]) + "&" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_BIT_OR: {
					txt = _parser_expr(c_node->arguments[0]) + "|" + _parser_expr(c_node->arguments[1]);
				} break;
				case GDParser::OperatorNode::OP_BIT_XOR: {
					txt = _parser_expr(c_node->arguments[0]) + "^" + _parser_expr(c_node->arguments[1]);
				} break;
				default: {}
			}

		} break;
		case GDParser::Node::TYPE_NEWLINE: {

			//skippie
		} break;
		default: {

			String error = "Parser bug at " + itos(p_expr->line) + ", invalid expression type: " + itos(p_expr->type);
			ERR_EXPLAIN(error);
			ERR_FAIL_V("");
		}
	}

	return txt;
	//return "("+txt+")";
}

static void _parser_show_block(const GDParser::BlockNode *p_block, int p_indent) {

	for (int i = 0; i < p_block->statements.size(); i++) {

		const GDParser::Node *statement = p_block->statements[i];

		switch (statement->type) {

			case GDParser::Node::TYPE_CONTROL_FLOW: {

				const GDParser::ControlFlowNode *cf_node = static_cast<const GDParser::ControlFlowNode *>(statement);
				switch (cf_node->cf_type) {

					case GDParser::ControlFlowNode::CF_IF: {

						ERR_FAIL_COND(cf_node->arguments.size() != 1);
						String txt;
						txt += "if ";
						txt += _parser_expr(cf_node->arguments[0]);
						txt += ":";
						_print_indent(p_indent, txt);
						ERR_FAIL_COND(!cf_node->body);
						_parser_show_block(cf_node->body, p_indent + 1);
						if (cf_node->body_else) {
							_print_indent(p_indent, "else:");
							_parser_show_block(cf_node->body_else, p_indent + 1);
						}

					} break;
					case GDParser::ControlFlowNode::CF_FOR: {
						ERR_FAIL_COND(cf_node->arguments.size() != 2);
						String txt;
						txt += "for ";
						txt += _parser_expr(cf_node->arguments[0]);
						txt += " in ";
						txt += _parser_expr(cf_node->arguments[1]);
						txt += ":";
						_print_indent(p_indent, txt);
						ERR_FAIL_COND(!cf_node->body);
						_parser_show_block(cf_node->body, p_indent + 1);

					} break;
					case GDParser::ControlFlowNode::CF_WHILE: {

						ERR_FAIL_COND(cf_node->arguments.size() != 1);
						String txt;
						txt += "while ";
						txt += _parser_expr(cf_node->arguments[0]);
						txt += ":";
						_print_indent(p_indent, txt);
						ERR_FAIL_COND(!cf_node->body);
						_parser_show_block(cf_node->body, p_indent + 1);

					} break;
					case GDParser::ControlFlowNode::CF_SWITCH: {

					} break;
					case GDParser::ControlFlowNode::CF_CONTINUE: {

						_print_indent(p_indent, "continue");
					} break;
					case GDParser::ControlFlowNode::CF_BREAK: {

						_print_indent(p_indent, "break");
					} break;
					case GDParser::ControlFlowNode::CF_RETURN: {

						if (cf_node->arguments.size())
							_print_indent(p_indent, "return " + _parser_expr(cf_node->arguments[0]));
						else
							_print_indent(p_indent, "return ");
					} break;
				}

			} break;
			case GDParser::Node::TYPE_LOCAL_VAR: {

				const GDParser::LocalVarNode *lv_node = static_cast<const GDParser::LocalVarNode *>(statement);
				_print_indent(p_indent, "var " + String(lv_node->name));
			} break;
			default: {
				//expression i guess
				_print_indent(p_indent, _parser_expr(statement));
			}
		}
	}
}

static void _parser_show_function(const GDParser::FunctionNode *p_func, int p_indent, GDParser::BlockNode *p_initializer = NULL) {

	String txt;
	if (p_func->_static)
		txt = "static ";
	txt += "func ";
	if (p_func->name == "") // initializer
		txt += "[built-in-initializer]";
	else
		txt += String(p_func->name);
	txt += "(";

	for (int i = 0; i < p_func->arguments.size(); i++) {

		if (i != 0)
			txt += ", ";
		txt += "var " + String(p_func->arguments[i]);
		if (i >= (p_func->arguments.size() - p_func->default_values.size())) {
			int defarg = i - (p_func->arguments.size() - p_func->default_values.size());
			txt += "=";
			txt += _parser_expr(p_func->default_values[defarg]);
		}
	}

	txt += ")";

	//todo constructor check!

	txt += ":";

	_print_indent(p_indent, txt);
	if (p_initializer)
		_parser_show_block(p_initializer, p_indent + 1);
	_parser_show_block(p_func->body, p_indent + 1);
}

static void _parser_show_class(const GDParser::ClassNode *p_class, int p_indent, const Vector<String> &p_code) {

	if (p_indent == 0 && (String(p_class->extends_file) != "" || p_class->extends_class.size())) {

		_print_indent(p_indent, _parser_extends(p_class));
		print_line("\n");
	}

	for (int i = 0; i < p_class->subclasses.size(); i++) {

		const GDParser::ClassNode *subclass = p_class->subclasses[i];
		String line = "class " + subclass->name;
		if (String(subclass->extends_file) != "" || subclass->extends_class.size())
			line += " " + _parser_extends(subclass);
		line += ":";
		_print_indent(p_indent, line);
		_parser_show_class(subclass, p_indent + 1, p_code);
		print_line("\n");
	}

	for (int i = 0; i < p_class->constant_expressions.size(); i++) {

		const GDParser::ClassNode::Constant &constant = p_class->constant_expressions[i];
		_print_indent(p_indent, "const " + String(constant.identifier) + "=" + _parser_expr(constant.expression));
	}

	for (int i = 0; i < p_class->variables.size(); i++) {

		const GDParser::ClassNode::Member &m = p_class->variables[i];

		_print_indent(p_indent, "var " + String(m.identifier));
	}

	print_line("\n");

	for (int i = 0; i < p_class->static_functions.size(); i++) {

		_parser_show_function(p_class->static_functions[i], p_indent);
		print_line("\n");
	}

	for (int i = 0; i < p_class->functions.size(); i++) {

		if (String(p_class->functions[i]->name) == "_init") {
			_parser_show_function(p_class->functions[i], p_indent, p_class->initializer);
		} else
			_parser_show_function(p_class->functions[i], p_indent);
		print_line("\n");
	}
	//_parser_show_function(p_class->initializer,p_indent);
	print_line("\n");
}

static String _disassemble_addr(const Ref<GDScript> &p_script, const GDFunction &func, int p_addr) {

	int addr = p_addr & GDFunction::ADDR_MASK;

	switch (p_addr >> GDFunction::ADDR_BITS) {

		case GDFunction::ADDR_TYPE_SELF: {
			return "self";
		} break;
		case GDFunction::ADDR_TYPE_CLASS: {
			return "class";
		} break;
		case GDFunction::ADDR_TYPE_MEMBER: {

			return "member(" + p_script->debug_get_member_by_index(addr) + ")";
		} break;
		case GDFunction::ADDR_TYPE_CLASS_CONSTANT: {

			return "class_const(" + func.get_global_name(addr) + ")";
		} break;
		case GDFunction::ADDR_TYPE_LOCAL_CONSTANT: {

			Variant v = func.get_constant(addr);
			String txt;
			if (v.get_type() == Variant::STRING || v.get_type() == Variant::NODE_PATH)
				txt = "\"" + String(v) + "\"";
			else
				txt = v;
			return "const(" + txt + ")";
		} break;
		case GDFunction::ADDR_TYPE_STACK: {

			return "stack(" + itos(addr) + ")";
		} break;
		case GDFunction::ADDR_TYPE_STACK_VARIABLE: {

			return "var_stack(" + itos(addr) + ")";
		} break;
		case GDFunction::ADDR_TYPE_GLOBAL: {

			return "global(" + func.get_global_name(addr) + ")";
		} break;
		case GDFunction::ADDR_TYPE_NIL: {
			return "nil";
		} break;
	}

	return "<err>";
}

static void _disassemble_class(const Ref<GDScript> &p_class, const Vector<String> &p_code) {

	const Map<StringName, GDFunction *> &mf = p_class->debug_get_member_functions();

	for (const Map<StringName, GDFunction *>::Element *E = mf.front(); E; E = E->next()) {

		const GDFunction &func = *E->get();
		const int *code = func.get_code();
		int codelen = func.get_code_size();
		String defargs;
		if (func.get_default_argument_count()) {
			defargs = "defarg at: ";
			for (int i = 0; i < func.get_default_argument_count(); i++) {

				if (i > 0)
					defargs += ",";
				defargs += itos(func.get_default_argument_addr(i));
			}
			defargs += " ";
		}
		print_line("== function " + String(func.get_name()) + "() :: stack size: " + itos(func.get_max_stack_size()) + " " + defargs + "==");

#define DADDR(m_ip) (_disassemble_addr(p_class, func, code[ip + m_ip]))

		for (int ip = 0; ip < codelen;) {

			int incr = 0;
			String txt = itos(ip) + " ";

			switch (code[ip]) {

				case GDFunction::OPCODE_OPERATOR: {

					int op = code[ip + 1];
					txt += "op ";

					String opname = Variant::get_operator_name(Variant::Operator(op));

					txt += DADDR(4);
					txt += " = ";
					txt += DADDR(2);
					txt += " " + opname + " ";
					txt += DADDR(3);
					incr += 5;

				} break;
				case GDFunction::OPCODE_SET: {

					txt += "set ";
					txt += DADDR(1);
					txt += "[";
					txt += DADDR(2);
					txt += "]=";
					txt += DADDR(3);
					incr += 4;

				} break;
				case GDFunction::OPCODE_GET: {

					txt += " get ";
					txt += DADDR(3);
					txt += "=";
					txt += DADDR(1);
					txt += "[";
					txt += DADDR(2);
					txt += "]";
					incr += 4;

				} break;
				case GDFunction::OPCODE_SET_NAMED: {

					txt += " set_named ";
					txt += DADDR(1);
					txt += "[\"";
					txt += func.get_global_name(code[ip + 2]);
					txt += "\"]=";
					txt += DADDR(3);
					incr += 4;

				} break;
				case GDFunction::OPCODE_GET_NAMED: {

					txt += " get_named ";
					txt += DADDR(3);
					txt += "=";
					txt += DADDR(1);
					txt += "[\"";
					txt += func.get_global_name(code[ip + 2]);
					txt += "\"]";
					incr += 4;

				} break;
				case GDFunction::OPCODE_SET_MEMBER: {

					txt += " set_member ";
					txt += "[\"";
					txt += func.get_global_name(code[ip + 1]);
					txt += "\"]=";
					txt += DADDR(2);
					incr += 3;

				} break;
				case GDFunction::OPCODE_GET_MEMBER: {

					txt += " get_member ";
					txt += DADDR(2);
					txt += "=";
					txt += "[\"";
					txt += func.get_global_name(code[ip + 1]);
					txt += "\"]";
					incr += 3;

				} break;
				case GDFunction::OPCODE_ASSIGN: {

					txt += " assign ";
					txt += DADDR(1);
					txt += "=";
					txt += DADDR(2);
					incr += 3;

				} break;
				case GDFunction::OPCODE_ASSIGN_TRUE: {

					txt += " assign ";
					txt += DADDR(1);
					txt += "= true";
					incr += 2;

				} break;
				case GDFunction::OPCODE_ASSIGN_FALSE: {

					txt += " assign ";
					txt += DADDR(1);
					txt += "= false";
					incr += 2;

				} break;
				case GDFunction::OPCODE_CONSTRUCT: {

					Variant::Type t = Variant::Type(code[ip + 1]);
					int argc = code[ip + 2];

					txt += " construct ";
					txt += DADDR(3 + argc);
					txt += " = ";

					txt += Variant::get_type_name(t) + "(";
					for (int i = 0; i < argc; i++) {

						if (i > 0)
							txt += ", ";
						txt += DADDR(i + 3);
					}
					txt += ")";

					incr = 4 + argc;

				} break;
				case GDFunction::OPCODE_CONSTRUCT_ARRAY: {

					int argc = code[ip + 1];
					txt += " make_array ";
					txt += DADDR(2 + argc);
					txt += " = [ ";

					for (int i = 0; i < argc; i++) {
						if (i > 0)
							txt += ", ";
						txt += DADDR(2 + i);
					}

					txt += "]";

					incr += 3 + argc;

				} break;
				case GDFunction::OPCODE_CONSTRUCT_DICTIONARY: {

					int argc = code[ip + 1];
					txt += " make_dict ";
					txt += DADDR(2 + argc * 2);
					txt += " = { ";

					for (int i = 0; i < argc; i++) {
						if (i > 0)
							txt += ", ";
						txt += DADDR(2 + i * 2 + 0);
						txt += ":";
						txt += DADDR(2 + i * 2 + 1);
					}

					txt += "}";

					incr += 3 + argc * 2;

				} break;

				case GDFunction::OPCODE_CALL:
				case GDFunction::OPCODE_CALL_RETURN: {

					bool ret = code[ip] == GDFunction::OPCODE_CALL_RETURN;

					if (ret)
						txt += " call-ret ";
					else
						txt += " call ";

					int argc = code[ip + 1];
					if (ret) {
						txt += DADDR(4 + argc) + "=";
					}

					txt += DADDR(2) + ".";
					txt += String(func.get_global_name(code[ip + 3]));
					txt += "(";

					for (int i = 0; i < argc; i++) {
						if (i > 0)
							txt += ", ";
						txt += DADDR(4 + i);
					}
					txt += ")";

					incr = 5 + argc;

				} break;
				case GDFunction::OPCODE_CALL_BUILT_IN: {

					txt += " call-built-in ";

					int argc = code[ip + 2];
					txt += DADDR(3 + argc) + "=";

					txt += GDFunctions::get_func_name(GDFunctions::Function(code[ip + 1]));
					txt += "(";

					for (int i = 0; i < argc; i++) {
						if (i > 0)
							txt += ", ";
						txt += DADDR(3 + i);
					}
					txt += ")";

					incr = 4 + argc;

				} break;
				case GDFunction::OPCODE_CALL_SELF_BASE: {

					txt += " call-self-base ";

					int argc = code[ip + 2];
					txt += DADDR(3 + argc) + "=";

					txt += func.get_global_name(code[ip + 1]);
					txt += "(";

					for (int i = 0; i < argc; i++) {
						if (i > 0)
							txt += ", ";
						txt += DADDR(3 + i);
					}
					txt += ")";

					incr = 4 + argc;

				} break;
				case GDFunction::OPCODE_YIELD: {

					txt += " yield ";
					incr = 1;

				} break;
				case GDFunction::OPCODE_YIELD_SIGNAL: {

					txt += " yield_signal ";
					txt += DADDR(1);
					txt += ",";
					txt += DADDR(2);
					incr = 3;
				} break;
				case GDFunction::OPCODE_YIELD_RESUME: {

					txt += " yield resume: ";
					txt += DADDR(1);
					incr = 2;
				} break;
				case GDFunction::OPCODE_JUMP: {

					txt += " jump ";
					txt += itos(code[ip + 1]);

					incr = 2;

				} break;
				case GDFunction::OPCODE_JUMP_IF: {

					txt += " jump-if ";
					txt += DADDR(1);
					txt += " to ";
					txt += itos(code[ip + 2]);

					incr = 3;
				} break;
				case GDFunction::OPCODE_JUMP_IF_NOT: {

					txt += " jump-if-not ";
					txt += DADDR(1);
					txt += " to ";
					txt += itos(code[ip + 2]);

					incr = 3;
				} break;
				case GDFunction::OPCODE_JUMP_TO_DEF_ARGUMENT: {

					txt += " jump-to-default-argument ";
					incr = 1;
				} break;
				case GDFunction::OPCODE_RETURN: {

					txt += " return ";
					txt += DADDR(1);

					incr = 2;

				} break;
				case GDFunction::OPCODE_ITERATE_BEGIN: {

					txt += " for-init " + DADDR(4) + " in " + DADDR(2) + " counter " + DADDR(1) + " end " + itos(code[ip + 3]);
					incr += 5;

				} break;
				case GDFunction::OPCODE_ITERATE: {

					txt += " for-loop " + DADDR(4) + " in " + DADDR(2) + " counter " + DADDR(1) + " end " + itos(code[ip + 3]);
					incr += 5;

				} break;
				case GDFunction::OPCODE_LINE: {

					int line = code[ip + 1] - 1;
					if (line >= 0 && line < p_code.size())
						txt = "\n" + itos(line + 1) + ": " + p_code[line] + "\n";
					else
						txt = "";
					incr += 2;
				} break;
				case GDFunction::OPCODE_END: {

					txt += " end";
					incr += 1;
				} break;
				case GDFunction::OPCODE_ASSERT: {

					txt += " assert ";
					txt += DADDR(1);
					incr += 2;

				} break;
			}

			if (incr == 0) {

				ERR_EXPLAIN("unhandled opcode: " + itos(code[ip]));
				ERR_BREAK(incr == 0);
			}

			ip += incr;
			if (txt != "")
				print_line(txt);
		}
	}
}

MainLoop *test(TestType p_test) {

	List<String> cmdlargs = OS::get_singleton()->get_cmdline_args();

	if (cmdlargs.empty()) {
		//try editor!
		return NULL;
	}

	String test = cmdlargs.back()->get();

	FileAccess *fa = FileAccess::open(test, FileAccess::READ);

	if (!fa) {
		ERR_EXPLAIN("Could not open file: " + test);
		ERR_FAIL_V(NULL);
	}

	Vector<uint8_t> buf;
	int flen = fa->get_len();
	buf.resize(fa->get_len() + 1);
	fa->get_buffer(&buf[0], flen);
	buf[flen] = 0;

	String code;
	code.parse_utf8((const char *)&buf[0]);

	Vector<String> lines;
	int last = 0;

	for (int i = 0; i <= code.length(); i++) {

		if (code[i] == '\n' || code[i] == 0) {

			lines.push_back(code.substr(last, i - last));
			last = i + 1;
		}
	}

	if (p_test == TEST_TOKENIZER) {

		GDTokenizerText tk;
		tk.set_code(code);
		int line = -1;
		while (tk.get_token() != GDTokenizer::TK_EOF) {

			String text;
			if (tk.get_token() == GDTokenizer::TK_IDENTIFIER)
				text = "'" + tk.get_token_identifier() + "' (identifier)";
			else if (tk.get_token() == GDTokenizer::TK_CONSTANT) {
				Variant c = tk.get_token_constant();
				if (c.get_type() == Variant::STRING)
					text = "\"" + String(c) + "\"";
				else
					text = c;

				text = text + " (" + Variant::get_type_name(c.get_type()) + " constant)";
			} else if (tk.get_token() == GDTokenizer::TK_ERROR)
				text = "ERROR: " + tk.get_token_error();
			else if (tk.get_token() == GDTokenizer::TK_NEWLINE)
				text = "newline (" + itos(tk.get_token_line()) + ") + indent: " + itos(tk.get_token_line_indent());
			else if (tk.get_token() == GDTokenizer::TK_BUILT_IN_FUNC)
				text = "'" + String(GDFunctions::get_func_name(tk.get_token_built_in_func())) + "' (built-in function)";
			else
				text = tk.get_token_name(tk.get_token());

			if (tk.get_token_line() != line) {
				int from = line + 1;
				line = tk.get_token_line();

				for (int i = from; i <= line; i++) {
					int l = i - 1;
					if (l >= 0 && l < lines.size()) {
						print_line("\n" + itos(i) + ": " + lines[l] + "\n");
					}
				}
			}
			print_line("\t(" + itos(tk.get_token_column()) + "): " + text);
			tk.advance();
		}
	}

	if (p_test == TEST_PARSER) {

		GDParser parser;
		Error err = parser.parse(code);
		if (err) {
			print_line("Parse Error:\n" + itos(parser.get_error_line()) + ":" + itos(parser.get_error_column()) + ":" + parser.get_error());
			memdelete(fa);
			return NULL;
		}

		const GDParser::Node *root = parser.get_parse_tree();
		ERR_FAIL_COND_V(root->type != GDParser::Node::TYPE_CLASS, NULL);
		const GDParser::ClassNode *cnode = static_cast<const GDParser::ClassNode *>(root);

		_parser_show_class(cnode, 0, lines);
	}

	if (p_test == TEST_COMPILER) {

		GDParser parser;

		Error err = parser.parse(code);
		if (err) {
			print_line("Parse Error:\n" + itos(parser.get_error_line()) + ":" + itos(parser.get_error_column()) + ":" + parser.get_error());
			memdelete(fa);
			return NULL;
		}

		GDScript *script = memnew(GDScript);

		GDCompiler gdc;
		err = gdc.compile(&parser, script);
		if (err) {

			print_line("Compile Error:\n" + itos(gdc.get_error_line()) + ":" + itos(gdc.get_error_column()) + ":" + gdc.get_error());
			memdelete(script);
			return NULL;
		}

		Ref<GDScript> gds = Ref<GDScript>(script);

		Ref<GDScript> current = gds;

		while (current.is_valid()) {

			print_line("** CLASS **");
			_disassemble_class(current, lines);

			current = current->get_base();
		}

	} else if (p_test == TEST_BYTECODE) {

		Vector<uint8_t> buf = GDTokenizerBuffer::parse_code_string(code);
		String dst = test.get_basename() + ".gdc";
		FileAccess *fw = FileAccess::open(dst, FileAccess::WRITE);
		fw->store_buffer(buf.ptr(), buf.size());
		memdelete(fw);
	}

#if 0
	Parser parser;
	Error err = parser.parse(code);
	if (err) {
		print_line("error:"+itos(parser.get_error_line())+":"+itos(parser.get_error_column())+":"+parser.get_error());
	} else {
		print_line("Parse O-K!");
	}
#endif

	memdelete(fa);

	return NULL;
}
}

#else

namespace TestGDScript {

MainLoop *test(TestType p_test) {

	return NULL;
}
}

#endif

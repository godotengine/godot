/*************************************************************************/
/*  test_shader_lang.cpp                                                 */
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

#include "test_shader_lang.h"

#include "core/os/file_access.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"

#include "core/print_string.h"
#include "scene/gui/control.h"
#include "scene/gui/text_edit.h"
#include "servers/visual/shader_language.h"

typedef ShaderLanguage SL;

namespace TestShaderLang {

static String _mktab(int p_level) {
	String tb;
	for (int i = 0; i < p_level; i++) {
		tb += "\t";
	}

	return tb;
}

static String _typestr(SL::DataType p_type) {
	return ShaderLanguage::get_datatype_name(p_type);
}

static String _prestr(SL::DataPrecision p_pres) {
	switch (p_pres) {
		case SL::PRECISION_LOWP:
			return "lowp ";
		case SL::PRECISION_MEDIUMP:
			return "mediump ";
		case SL::PRECISION_HIGHP:
			return "highp ";
		case SL::PRECISION_DEFAULT:
			return "";
	}
	return "";
}

static String _opstr(SL::Operator p_op) {
	return ShaderLanguage::get_operator_text(p_op);
}

static String get_constant_text(SL::DataType p_type, const Vector<SL::ConstantNode::Value> &p_values) {
	switch (p_type) {
		case SL::TYPE_BOOL:
			return p_values[0].boolean ? "true" : "false";
		case SL::TYPE_BVEC2:
			return String() + "bvec2(" + (p_values[0].boolean ? "true" : "false") + (p_values[1].boolean ? "true" : "false") + ")";
		case SL::TYPE_BVEC3:
			return String() + "bvec3(" + (p_values[0].boolean ? "true" : "false") + "," + (p_values[1].boolean ? "true" : "false") + "," + (p_values[2].boolean ? "true" : "false") + ")";
		case SL::TYPE_BVEC4:
			return String() + "bvec4(" + (p_values[0].boolean ? "true" : "false") + "," + (p_values[1].boolean ? "true" : "false") + "," + (p_values[2].boolean ? "true" : "false") + "," + (p_values[3].boolean ? "true" : "false") + ")";
		case SL::TYPE_INT:
			return rtos(p_values[0].sint);
		case SL::TYPE_IVEC2:
			return String() + "ivec2(" + rtos(p_values[0].sint) + "," + rtos(p_values[1].sint) + ")";
		case SL::TYPE_IVEC3:
			return String() + "ivec3(" + rtos(p_values[0].sint) + "," + rtos(p_values[1].sint) + "," + rtos(p_values[2].sint) + ")";
		case SL::TYPE_IVEC4:
			return String() + "ivec4(" + rtos(p_values[0].sint) + "," + rtos(p_values[1].sint) + "," + rtos(p_values[2].sint) + "," + rtos(p_values[3].sint) + ")";
		case SL::TYPE_UINT:
			return rtos(p_values[0].real);
		case SL::TYPE_UVEC2:
			return String() + "uvec2(" + rtos(p_values[0].real) + "," + rtos(p_values[1].real) + ")";
		case SL::TYPE_UVEC3:
			return String() + "uvec3(" + rtos(p_values[0].real) + "," + rtos(p_values[1].real) + "," + rtos(p_values[2].real) + ")";
		case SL::TYPE_UVEC4:
			return String() + "uvec4(" + rtos(p_values[0].real) + "," + rtos(p_values[1].real) + "," + rtos(p_values[2].real) + "," + rtos(p_values[3].real) + ")";
		case SL::TYPE_FLOAT:
			return rtos(p_values[0].real);
		case SL::TYPE_VEC2:
			return String() + "vec2(" + rtos(p_values[0].real) + "," + rtos(p_values[1].real) + ")";
		case SL::TYPE_VEC3:
			return String() + "vec3(" + rtos(p_values[0].real) + "," + rtos(p_values[1].real) + "," + rtos(p_values[2].real) + ")";
		case SL::TYPE_VEC4:
			return String() + "vec4(" + rtos(p_values[0].real) + "," + rtos(p_values[1].real) + "," + rtos(p_values[2].real) + "," + rtos(p_values[3].real) + ")";
		default:
			ERR_FAIL_V(String());
	}
}

static String dump_node_code(SL::Node *p_node, int p_level) {
	String code;

	switch (p_node->type) {
		case SL::Node::TYPE_SHADER: {
			SL::ShaderNode *pnode = (SL::ShaderNode *)p_node;

			for (Map<StringName, SL::ShaderNode::Uniform>::Element *E = pnode->uniforms.front(); E; E = E->next()) {
				String ucode = "uniform ";
				ucode += _prestr(E->get().precision);
				ucode += _typestr(E->get().type);
				ucode += " " + String(E->key());

				if (E->get().default_value.size()) {
					ucode += " = " + get_constant_text(E->get().type, E->get().default_value);
				}

				static const char *hint_name[SL::ShaderNode::Uniform::HINT_MAX] = {
					"",
					"color",
					"range",
					"albedo",
					"normal",
					"black",
					"white"
				};

				if (E->get().hint) {
					ucode += " : " + String(hint_name[E->get().hint]);
				}

				code += ucode + "\n";
			}

			for (Map<StringName, SL::ShaderNode::Varying>::Element *E = pnode->varyings.front(); E; E = E->next()) {
				String vcode = "varying ";
				vcode += _prestr(E->get().precision);
				vcode += _typestr(E->get().type);
				vcode += " " + String(E->key());

				code += vcode + "\n";
			}
			for (int i = 0; i < pnode->functions.size(); i++) {
				SL::FunctionNode *fnode = pnode->functions[i].function;

				String header;
				header = _typestr(fnode->return_type) + " " + fnode->name + "(";
				for (int j = 0; j < fnode->arguments.size(); j++) {
					if (j > 0) {
						header += ", ";
					}
					header += _prestr(fnode->arguments[j].precision) + _typestr(fnode->arguments[j].type) + " " + fnode->arguments[j].name;
				}

				header += ")\n";
				code += header;
				code += dump_node_code(fnode->body, p_level + 1);
			}

			//code+=dump_node_code(pnode->body,p_level);
		} break;
		case SL::Node::TYPE_STRUCT: {
		} break;
		case SL::Node::TYPE_FUNCTION: {
		} break;
		case SL::Node::TYPE_BLOCK: {
			SL::BlockNode *bnode = (SL::BlockNode *)p_node;

			//variables
			code += _mktab(p_level - 1) + "{\n";
			for (Map<StringName, SL::BlockNode::Variable>::Element *E = bnode->variables.front(); E; E = E->next()) {
				code += _mktab(p_level) + _prestr(E->get().precision) + _typestr(E->get().type) + " " + E->key() + ";\n";
			}

			for (int i = 0; i < bnode->statements.size(); i++) {
				String scode = dump_node_code(bnode->statements[i], p_level);

				if (bnode->statements[i]->type == SL::Node::TYPE_CONTROL_FLOW) {
					code += scode; //use directly
				} else {
					code += _mktab(p_level) + scode + ";\n";
				}
			}
			code += _mktab(p_level - 1) + "}\n";

		} break;
		case SL::Node::TYPE_VARIABLE: {
			SL::VariableNode *vnode = (SL::VariableNode *)p_node;
			code = vnode->name;

		} break;
		case SL::Node::TYPE_VARIABLE_DECLARATION: {
			// FIXME: Implement
		} break;
		case SL::Node::TYPE_ARRAY: {
			SL::ArrayNode *vnode = (SL::ArrayNode *)p_node;
			code = vnode->name;
		} break;
		case SL::Node::TYPE_ARRAY_DECLARATION: {
			// FIXME: Implement
		} break;
		case SL::Node::TYPE_ARRAY_CONSTRUCT: {
			// FIXME: Implement
		} break;
		case SL::Node::TYPE_CONSTANT: {
			SL::ConstantNode *cnode = (SL::ConstantNode *)p_node;
			return get_constant_text(cnode->datatype, cnode->values);

		} break;
		case SL::Node::TYPE_OPERATOR: {
			SL::OperatorNode *onode = (SL::OperatorNode *)p_node;

			switch (onode->op) {
				case SL::OP_ASSIGN:
				case SL::OP_ASSIGN_ADD:
				case SL::OP_ASSIGN_SUB:
				case SL::OP_ASSIGN_MUL:
				case SL::OP_ASSIGN_DIV:
				case SL::OP_ASSIGN_SHIFT_LEFT:
				case SL::OP_ASSIGN_SHIFT_RIGHT:
				case SL::OP_ASSIGN_MOD:
				case SL::OP_ASSIGN_BIT_AND:
				case SL::OP_ASSIGN_BIT_OR:
				case SL::OP_ASSIGN_BIT_XOR:
					code = dump_node_code(onode->arguments[0], p_level) + _opstr(onode->op) + dump_node_code(onode->arguments[1], p_level);
					break;
				case SL::OP_BIT_INVERT:
				case SL::OP_NEGATE:
				case SL::OP_NOT:
				case SL::OP_DECREMENT:
				case SL::OP_INCREMENT:
					code = _opstr(onode->op) + dump_node_code(onode->arguments[0], p_level);
					break;
				case SL::OP_POST_DECREMENT:
				case SL::OP_POST_INCREMENT:
					code = dump_node_code(onode->arguments[0], p_level) + _opstr(onode->op);
					break;
				case SL::OP_CALL:
				case SL::OP_CONSTRUCT:
					code = dump_node_code(onode->arguments[0], p_level) + "(";
					for (int i = 1; i < onode->arguments.size(); i++) {
						if (i > 1) {
							code += ", ";
						}
						code += dump_node_code(onode->arguments[i], p_level);
					}
					code += ")";
					break;
				default: {
					code = "(" + dump_node_code(onode->arguments[0], p_level) + _opstr(onode->op) + dump_node_code(onode->arguments[1], p_level) + ")";
					break;
				}
			}

		} break;
		case SL::Node::TYPE_CONTROL_FLOW: {
			SL::ControlFlowNode *cfnode = (SL::ControlFlowNode *)p_node;
			if (cfnode->flow_op == SL::FLOW_OP_IF) {
				code += _mktab(p_level) + "if (" + dump_node_code(cfnode->expressions[0], p_level) + ")\n";
				code += dump_node_code(cfnode->blocks[0], p_level + 1);
				if (cfnode->blocks.size() == 2) {
					code += _mktab(p_level) + "else\n";
					code += dump_node_code(cfnode->blocks[1], p_level + 1);
				}

			} else if (cfnode->flow_op == SL::FLOW_OP_RETURN) {
				if (cfnode->blocks.size()) {
					code = "return " + dump_node_code(cfnode->blocks[0], p_level);
				} else {
					code = "return";
				}
			}

		} break;
		case SL::Node::TYPE_MEMBER: {
			SL::MemberNode *mnode = (SL::MemberNode *)p_node;
			code = dump_node_code(mnode->owner, p_level) + "." + mnode->name;

		} break;
	}

	return code;
}

static Error recreate_code(void *p_str, SL::ShaderNode *p_program) {
	String *str = (String *)p_str;

	*str = dump_node_code(p_program, 0);

	return OK;
}

MainLoop *test() {
	List<String> cmdlargs = OS::get_singleton()->get_cmdline_args();

	if (cmdlargs.empty()) {
		//try editor!
		print_line("usage: godot -test shader_lang <shader>");
		return nullptr;
	}

	String test = cmdlargs.back()->get();

	FileAccess *fa = FileAccess::open(test, FileAccess::READ);

	if (!fa) {
		ERR_FAIL_V(nullptr);
	}

	String code;

	while (true) {
		CharType c = fa->get_8();
		if (fa->eof_reached()) {
			break;
		}
		code += c;
	}

	SL sl;
	print_line("tokens:\n\n" + sl.token_debug(code));

	Map<StringName, SL::FunctionInfo> dt;
	dt["fragment"].built_ins["ALBEDO"] = SL::TYPE_VEC3;
	dt["fragment"].can_discard = true;

	Vector<StringName> rm;
	rm.push_back("popo");
	Set<String> types;
	types.insert("spatial");

	Error err = sl.compile(code, dt, rm, types);

	if (err) {
		print_line("Error at line: " + rtos(sl.get_error_line()) + ": " + sl.get_error_text());
		return nullptr;
	} else {
		String code2;
		recreate_code(&code2, sl.get_shader());
		print_line("code:\n\n" + code2);
	}

	return nullptr;
}
} // namespace TestShaderLang

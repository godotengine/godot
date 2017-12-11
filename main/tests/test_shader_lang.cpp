/*************************************************************************/
/*  test_shader_lang.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#include "os/file_access.h"
#include "os/main_loop.h"
#include "os/os.h"

#include "drivers/gles2/shader_compiler_gles2.h"
#include "print_string.h"
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

	switch (p_type) {

		case SL::TYPE_VOID: return "void";
		case SL::TYPE_BOOL: return "bool";
		case SL::TYPE_FLOAT: return "float";
		case SL::TYPE_VEC2: return "vec2";
		case SL::TYPE_VEC3: return "vec3";
		case SL::TYPE_VEC4: return "vec4";
		case SL::TYPE_MAT3: return "mat3";
		case SL::TYPE_MAT4: return "mat4";
		case SL::TYPE_TEXTURE: return "texture";
		case SL::TYPE_CUBEMAP: return "cubemap";
		default: {}
	}

	return "";
}

static String _opstr(SL::Operator p_op) {

	switch (p_op) {
		case SL::OP_ASSIGN: return "=";
		case SL::OP_ADD: return "+";
		case SL::OP_SUB: return "-";
		case SL::OP_MUL: return "*";
		case SL::OP_DIV: return "/";
		case SL::OP_ASSIGN_ADD: return "+=";
		case SL::OP_ASSIGN_SUB: return "-=";
		case SL::OP_ASSIGN_MUL: return "*=";
		case SL::OP_ASSIGN_DIV: return "/=";
		case SL::OP_NEG: return "-";
		case SL::OP_NOT: return "!";
		case SL::OP_CMP_EQ: return "==";
		case SL::OP_CMP_NEQ: return "!=";
		case SL::OP_CMP_LEQ: return "<=";
		case SL::OP_CMP_GEQ: return ">=";
		case SL::OP_CMP_LESS: return "<";
		case SL::OP_CMP_GREATER: return ">";
		case SL::OP_CMP_OR: return "||";
		case SL::OP_CMP_AND: return "&&";
		default: return "";
	}

	return "";
}

static String dump_node_code(SL::Node *p_node, int p_level) {

	String code;

	switch (p_node->type) {

		case SL::Node::TYPE_PROGRAM: {

			SL::ProgramNode *pnode = (SL::ProgramNode *)p_node;

			for (Map<StringName, SL::Uniform>::Element *E = pnode->uniforms.front(); E; E = E->next()) {

				String ucode = "uniform ";
				ucode += _typestr(E->get().type) + "=" + String(E->get().default_value) + "\n";
				code += ucode;
			}

			for (int i = 0; i < pnode->functions.size(); i++) {

				SL::FunctionNode *fnode = pnode->functions[i].function;

				String header;
				header = _typestr(fnode->return_type) + " " + fnode->name + "(";
				for (int i = 0; i < fnode->arguments.size(); i++) {

					if (i > 0)
						header += ", ";
					header += _typestr(fnode->arguments[i].type) + " " + fnode->arguments[i].name;
				}

				header += ") {\n";
				code += header;
				code += dump_node_code(fnode->body, p_level + 1);
				code += "}\n";
			}

			code += dump_node_code(pnode->body, p_level);
		} break;
		case SL::Node::TYPE_FUNCTION: {

		} break;
		case SL::Node::TYPE_BLOCK: {
			SL::BlockNode *bnode = (SL::BlockNode *)p_node;

			//variables
			for (Map<StringName, SL::DataType>::Element *E = bnode->variables.front(); E; E = E->next()) {

				code += _mktab(p_level) + _typestr(E->value()) + " " + E->key() + ";\n";
			}

			for (int i = 0; i < bnode->statements.size(); i++) {

				code += _mktab(p_level) + dump_node_code(bnode->statements[i], p_level) + ";\n";
			}

		} break;
		case SL::Node::TYPE_VARIABLE: {
			SL::VariableNode *vnode = (SL::VariableNode *)p_node;
			code = vnode->name;

		} break;
		case SL::Node::TYPE_CONSTANT: {
			SL::ConstantNode *cnode = (SL::ConstantNode *)p_node;
			switch (cnode->datatype) {

				case SL::TYPE_BOOL: code = cnode->value.operator bool() ? "true" : "false"; break;
				case SL::TYPE_FLOAT: code = cnode->value; break;
				case SL::TYPE_VEC2: {
					Vector2 v = cnode->value;
					code = "vec2(" + rtos(v.x) + ", " + rtos(v.y) + ")";
				} break;
				case SL::TYPE_VEC3: {
					Vector3 v = cnode->value;
					code = "vec3(" + rtos(v.x) + ", " + rtos(v.y) + ", " + rtos(v.z) + ")";
				} break;
				case SL::TYPE_VEC4: {
					Plane v = cnode->value;
					code = "vec4(" + rtos(v.normal.x) + ", " + rtos(v.normal.y) + ", " + rtos(v.normal.z) + ", " + rtos(v.d) + ")";
				} break;
				case SL::TYPE_MAT3: {
					Matrix3 x = cnode->value;
					code = "mat3( vec3(" + rtos(x.get_axis(0).x) + ", " + rtos(x.get_axis(0).y) + ", " + rtos(x.get_axis(0).z) + "), vec3(" + rtos(x.get_axis(1).x) + ", " + rtos(x.get_axis(1).y) + ", " + rtos(x.get_axis(1).z) + "), vec3(" + rtos(x.get_axis(2).x) + ", " + rtos(x.get_axis(2).y) + ", " + rtos(x.get_axis(2).z) + "))";
				} break;
				case SL::TYPE_MAT4: {
					Transform x = cnode->value;
					code = "mat4( vec3(" + rtos(x.basis.get_axis(0).x) + ", " + rtos(x.basis.get_axis(0).y) + ", " + rtos(x.basis.get_axis(0).z) + "), vec3(" + rtos(x.basis.get_axis(1).x) + ", " + rtos(x.basis.get_axis(1).y) + ", " + rtos(x.basis.get_axis(1).z) + "), vec3(" + rtos(x.basis.get_axis(2).x) + ", " + rtos(x.basis.get_axis(2).y) + ", " + rtos(x.basis.get_axis(2).z) + "), vec3(" + rtos(x.origin.x) + ", " + rtos(x.origin.y) + ", " + rtos(x.origin.z) + "))";
				} break;
				default: code = "<error: " + Variant::get_type_name(cnode->value.get_type()) + " (" + itos(cnode->datatype) + ">";
			}

		} break;
		case SL::Node::TYPE_OPERATOR: {
			SL::OperatorNode *onode = (SL::OperatorNode *)p_node;

			switch (onode->op) {

				case SL::OP_ASSIGN:
				case SL::OP_ASSIGN_ADD:
				case SL::OP_ASSIGN_SUB:
				case SL::OP_ASSIGN_MUL:
				case SL::OP_ASSIGN_DIV:
					code = dump_node_code(onode->arguments[0], p_level) + _opstr(onode->op) + dump_node_code(onode->arguments[1], p_level);
					break;

				case SL::OP_ADD:
				case SL::OP_SUB:
				case SL::OP_MUL:
				case SL::OP_DIV:
				case SL::OP_CMP_EQ:
				case SL::OP_CMP_NEQ:
				case SL::OP_CMP_LEQ:
				case SL::OP_CMP_GEQ:
				case SL::OP_CMP_LESS:
				case SL::OP_CMP_GREATER:
				case SL::OP_CMP_OR:
				case SL::OP_CMP_AND:

					code = "(" + dump_node_code(onode->arguments[0], p_level) + _opstr(onode->op) + dump_node_code(onode->arguments[1], p_level) + ")";
					break;
				case SL::OP_NEG:
				case SL::OP_NOT:
					code = _opstr(onode->op) + dump_node_code(onode->arguments[0], p_level);
					break;
				case SL::OP_CALL:
				case SL::OP_CONSTRUCT:
					code = dump_node_code(onode->arguments[0], p_level) + "(";
					for (int i = 1; i < onode->arguments.size(); i++) {
						if (i > 1)
							code += ", ";
						code += dump_node_code(onode->arguments[i], p_level);
					}
					code += ")";
					break;
				default: {}
			}

		} break;
		case SL::Node::TYPE_CONTROL_FLOW: {
			SL::ControlFlowNode *cfnode = (SL::ControlFlowNode *)p_node;
			if (cfnode->flow_op == SL::FLOW_OP_IF) {

				code += "if (" + dump_node_code(cfnode->statements[0], p_level) + ") {\n";
				code += dump_node_code(cfnode->statements[1], p_level + 1);
				if (cfnode->statements.size() == 3) {

					code += "} else {\n";
					code += dump_node_code(cfnode->statements[2], p_level + 1);
				}

				code += "}\n";

			} else if (cfnode->flow_op == SL::FLOW_OP_RETURN) {

				if (cfnode->statements.size()) {
					code = "return " + dump_node_code(cfnode->statements[0], p_level);
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

static Error recreate_code(void *p_str, SL::ProgramNode *p_program) {

	print_line("recr");
	String *str = (String *)p_str;

	*str = dump_node_code(p_program, 0);

	return OK;
}

MainLoop *test() {

	List<String> cmdlargs = OS::get_singleton()->get_cmdline_args();

	if (cmdlargs.empty()) {
		//try editor!
		return NULL;
	}

	String test = cmdlargs.back()->get();

	FileAccess *fa = FileAccess::open(test, FileAccess::READ);

	if (!fa) {
		ERR_FAIL_V(NULL);
	}

	String code;

	while (true) {
		CharType c = fa->get_8();
		if (fa->eof_reached())
			break;
		code += c;
	}

	int errline;
	int errcol;
	String error;
	print_line(SL::lex_debug(code));
	Error err = SL::compile(code, ShaderLanguage::SHADER_MATERIAL_FRAGMENT, NULL, NULL, &error, &errline, &errcol);

	if (err) {

		print_line("Error: " + itos(errline) + ":" + itos(errcol) + " " + error);
		return NULL;
	}

	print_line("Compile OK! - pretty printing");

	String rcode;
	err = SL::compile(code, ShaderLanguage::SHADER_MATERIAL_FRAGMENT, recreate_code, &rcode, &error, &errline, &errcol);

	if (!err) {
		print_line(rcode);
	}

	ShaderCompilerGLES2 comp;
	String codeline, globalsline;
	SL::VarInfo vi;
	vi.name = "mongs";
	vi.type = SL::TYPE_VEC3;

	ShaderCompilerGLES2::Flags fl;
	comp.compile(code, ShaderLanguage::SHADER_MATERIAL_FRAGMENT, codeline, globalsline, fl);

	return NULL;
}
} // namespace TestShaderLang

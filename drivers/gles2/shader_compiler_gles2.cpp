/*************************************************************************/
/*  shader_compiler_gles2.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "shader_compiler_gles2.h"
#ifdef GLES2_BACKEND_ENABLED

#include "core/os/os.h"

//#ifdef GODOT_3

#ifdef GODOT_3
#include "core/project_settings.h"
#include "core/string_buffer.h"
#include "core/string_builder.h"
#else
#include "core/config/project_settings.h"
#include "core/string/string_buffer.h"
#include "core/string/string_builder.h"

#endif

#define SL ShaderLanguage

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

static String _qualstr(SL::ArgumentQualifier p_qual) {
	switch (p_qual) {
		case SL::ARGUMENT_QUALIFIER_IN:
			return "in ";
		case SL::ARGUMENT_QUALIFIER_OUT:
			return "out ";
		case SL::ARGUMENT_QUALIFIER_INOUT:
			return "inout ";
	}
	return "";
}

static String _opstr(SL::Operator p_op) {
	return SL::get_operator_text(p_op);
}

static String _mkid(const String &p_id) {
	String id = "m_" + p_id.replace("__", "_dus_");
	return id.replace("__", "_dus_"); //doubleunderscore is reserved in glsl
}

static String f2sp0(float p_float) {
	String num = rtoss(p_float);
	if (num.find(".") == -1 && num.find("e") == -1) {
		num += ".0";
	}
	return num;
}

static String get_constant_text(SL::DataType p_type, const Vector<SL::ConstantNode::Value> &p_values) {
	switch (p_type) {
		case SL::TYPE_BOOL:
			return p_values[0].boolean ? "true" : "false";
		case SL::TYPE_BVEC2:
		case SL::TYPE_BVEC3:
		case SL::TYPE_BVEC4: {
			StringBuffer<> text;

			text += "bvec";
			text += itos(p_type - SL::TYPE_BOOL + 1);
			text += "(";

			for (int i = 0; i < p_values.size(); i++) {
				if (i > 0)
					text += ",";

				text += p_values[i].boolean ? "true" : "false";
			}
			text += ")";
			return text.as_string();
		}

		// GLSL ES 2 doesn't support uints, so we just use signed ints instead...
		case SL::TYPE_UINT:
			return itos(p_values[0].uint);
		case SL::TYPE_UVEC2:
		case SL::TYPE_UVEC3:
		case SL::TYPE_UVEC4: {
			StringBuffer<> text;

			text += "ivec";
			text += itos(p_type - SL::TYPE_UINT + 1);
			text += "(";

			for (int i = 0; i < p_values.size(); i++) {
				if (i > 0)
					text += ",";

				text += itos(p_values[i].uint);
			}
			text += ")";
			return text.as_string();

		} break;

		case SL::TYPE_INT:
			return itos(p_values[0].sint);
		case SL::TYPE_IVEC2:
		case SL::TYPE_IVEC3:
		case SL::TYPE_IVEC4: {
			StringBuffer<> text;

			text += "ivec";
			text += itos(p_type - SL::TYPE_INT + 1);
			text += "(";

			for (int i = 0; i < p_values.size(); i++) {
				if (i > 0)
					text += ",";

				text += itos(p_values[i].sint);
			}
			text += ")";
			return text.as_string();

		} break;
		case SL::TYPE_FLOAT:
			return f2sp0(p_values[0].real);
		case SL::TYPE_VEC2:
		case SL::TYPE_VEC3:
		case SL::TYPE_VEC4: {
			StringBuffer<> text;

			text += "vec";
			text += itos(p_type - SL::TYPE_FLOAT + 1);
			text += "(";

			for (int i = 0; i < p_values.size(); i++) {
				if (i > 0)
					text += ",";

				text += f2sp0(p_values[i].real);
			}
			text += ")";
			return text.as_string();

		} break;
		case SL::TYPE_MAT2:
		case SL::TYPE_MAT3:
		case SL::TYPE_MAT4: {
			StringBuffer<> text;

			text += "mat";
			text += itos(p_type - SL::TYPE_MAT2 + 2);
			text += "(";

			for (int i = 0; i < p_values.size(); i++) {
				if (i > 0)
					text += ",";

				text += f2sp0(p_values[i].real);
			}
			text += ")";
			return text.as_string();

		} break;
		default:
			ERR_FAIL_V(String());
	}
}

void ShaderCompilerGLES2::_dump_function_deps(SL::ShaderNode *p_node, const StringName &p_for_func, const Map<StringName, String> &p_func_code, StringBuilder &r_to_add, Set<StringName> &r_added) {
	int fidx = -1;

	for (int i = 0; i < p_node->functions.size(); i++) {
		if (p_node->functions[i].name == p_for_func) {
			fidx = i;
			break;
		}
	}

	ERR_FAIL_COND(fidx == -1);

	for (Set<StringName>::Element *E = p_node->functions[fidx].uses_function.front(); E; E = E->next()) {
		if (r_added.has(E->get())) {
			continue;
		}

		_dump_function_deps(p_node, E->get(), p_func_code, r_to_add, r_added);

		SL::FunctionNode *fnode = NULL;

		for (int i = 0; i < p_node->functions.size(); i++) {
			if (p_node->functions[i].name == E->get()) {
				fnode = p_node->functions[i].function;
				break;
			}
		}

		ERR_FAIL_COND(!fnode);

		r_to_add += "\n";

		StringBuffer<128> header;

		header += _typestr(fnode->return_type);
		header += " ";
		header += _mkid(fnode->name);
		header += "(";

		for (int i = 0; i < fnode->arguments.size(); i++) {
			if (i > 0)
				header += ", ";

			header += _qualstr(fnode->arguments[i].qualifier);
			header += _prestr(fnode->arguments[i].precision);
			header += _typestr(fnode->arguments[i].type);
			header += " ";
			header += _mkid(fnode->arguments[i].name);
		}

		header += ")\n";
		r_to_add += header.as_string();
		r_to_add += p_func_code[E->get()];

		r_added.insert(E->get());
	}
}

String ShaderCompilerGLES2::_dump_node_code(SL::Node *p_node, int p_level, GeneratedCode &r_gen_code, IdentifierActions &p_actions, const DefaultIdentifierActions &p_default_actions, bool p_assigning, bool p_use_scope) {
	StringBuilder code;

	switch (p_node->type) {
		default: {
		} break;
		case SL::Node::TYPE_SHADER: {
			SL::ShaderNode *snode = (SL::ShaderNode *)p_node;

			for (int i = 0; i < snode->render_modes.size(); i++) {
				if (p_default_actions.render_mode_defines.has(snode->render_modes[i]) && !used_rmode_defines.has(snode->render_modes[i])) {
					r_gen_code.custom_defines.push_back(p_default_actions.render_mode_defines[snode->render_modes[i]].utf8());
					used_rmode_defines.insert(snode->render_modes[i]);
				}

				if (p_actions.render_mode_flags.has(snode->render_modes[i])) {
					*p_actions.render_mode_flags[snode->render_modes[i]] = true;
				}

				if (p_actions.render_mode_values.has(snode->render_modes[i])) {
					Pair<int *, int> &p = p_actions.render_mode_values[snode->render_modes[i]];
					*p.first = p.second;
				}
			}

			int max_texture_uniforms = 0;
			int max_uniforms = 0;

			for (Map<StringName, SL::ShaderNode::Uniform>::Element *E = snode->uniforms.front(); E; E = E->next()) {
				if (SL::is_sampler_type(E->get().type))
					max_texture_uniforms++;
				else
					max_uniforms++;
			}

			r_gen_code.texture_uniforms.resize(max_texture_uniforms);
			r_gen_code.texture_hints.resize(max_texture_uniforms);

			r_gen_code.uniforms.resize(max_uniforms + max_texture_uniforms);

			StringBuilder vertex_global;
			StringBuilder fragment_global;

			// uniforms

			for (Map<StringName, SL::ShaderNode::Uniform>::Element *E = snode->uniforms.front(); E; E = E->next()) {
				StringBuffer<> uniform_code;

				// use highp if no precision is specified to prevent different default values in fragment and vertex shader
				SL::DataPrecision precision = E->get().precision;
				if (precision == SL::PRECISION_DEFAULT && E->get().type != SL::TYPE_BOOL) {
					precision = SL::PRECISION_HIGHP;
				}

				uniform_code += "uniform ";
				uniform_code += _prestr(precision);
				uniform_code += _typestr(E->get().type);
				uniform_code += " ";
				uniform_code += _mkid(E->key());
				uniform_code += ";\n";

				if (SL::is_sampler_type(E->get().type)) {
					r_gen_code.texture_uniforms.write[E->get().texture_order] = E->key();
					r_gen_code.texture_hints.write[E->get().texture_order] = E->get().hint;
				} else {
					r_gen_code.uniforms.write[E->get().order] = E->key();
				}

				vertex_global += uniform_code.as_string();
				fragment_global += uniform_code.as_string();

				p_actions.uniforms->insert(E->key(), E->get());
			}

			// varyings

			for (Map<StringName, SL::ShaderNode::Varying>::Element *E = snode->varyings.front(); E; E = E->next()) {
				StringBuffer<> varying_code;

				varying_code += "varying ";
				varying_code += _prestr(E->get().precision);
				varying_code += _typestr(E->get().type);
				varying_code += " ";
				varying_code += _mkid(E->key());
				if (E->get().array_size > 0) {
					varying_code += "[";
					varying_code += itos(E->get().array_size);
					varying_code += "]";
				}
				varying_code += ";\n";

				String final_code = varying_code.as_string();

				vertex_global += final_code;
				fragment_global += final_code;
			}

			// constants

			for (int i = 0; i < snode->vconstants.size(); i++) {
				String gcode;
				gcode += "const ";
				gcode += _prestr(snode->vconstants[i].precision);
				gcode += _typestr(snode->vconstants[i].type);
				gcode += " " + _mkid(String(snode->vconstants[i].name));
				gcode += "=";
				gcode += _dump_node_code(snode->vconstants[i].initializer, p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				gcode += ";\n";
				vertex_global += gcode;
				fragment_global += gcode;
			}

			// functions

			Map<StringName, String> function_code;

			for (int i = 0; i < snode->functions.size(); i++) {
				SL::FunctionNode *fnode = snode->functions[i].function;
				current_func_name = fnode->name;
				function_code[fnode->name] = _dump_node_code(fnode->body, 1, r_gen_code, p_actions, p_default_actions, p_assigning);
			}

			Set<StringName> added_vertex;
			Set<StringName> added_fragment;

			for (int i = 0; i < snode->functions.size(); i++) {
				SL::FunctionNode *fnode = snode->functions[i].function;

				current_func_name = fnode->name;

				if (fnode->name == vertex_name) {
					_dump_function_deps(snode, fnode->name, function_code, vertex_global, added_vertex);
					r_gen_code.vertex = function_code[vertex_name];

				} else if (fnode->name == fragment_name) {
					_dump_function_deps(snode, fnode->name, function_code, fragment_global, added_fragment);
					r_gen_code.fragment = function_code[fragment_name];

				} else if (fnode->name == light_name) {
					_dump_function_deps(snode, fnode->name, function_code, fragment_global, added_fragment);
					r_gen_code.light = function_code[light_name];
				}
			}

			r_gen_code.vertex_global = vertex_global.as_string();
			r_gen_code.fragment_global = fragment_global.as_string();

		} break;

		case SL::Node::TYPE_FUNCTION: {
		} break;

		case SL::Node::TYPE_BLOCK: {
			SL::BlockNode *bnode = (SL::BlockNode *)p_node;

			if (!bnode->single_statement) {
				code += _mktab(p_level - 1);
				code += "{\n";
			}

			for (int i = 0; i < bnode->statements.size(); i++) {
				String statement_code = _dump_node_code(bnode->statements[i], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);

				if (bnode->statements[i]->type == SL::Node::TYPE_CONTROL_FLOW || bnode->single_statement) {
					code += statement_code;
				} else {
					code += _mktab(p_level);
					code += statement_code;
					code += ";\n";
				}
			}

			if (!bnode->single_statement) {
				code += _mktab(p_level - 1);
				code += "}\n";
			}
		} break;

		case SL::Node::TYPE_VARIABLE_DECLARATION: {
			SL::VariableDeclarationNode *var_dec_node = (SL::VariableDeclarationNode *)p_node;

			StringBuffer<> declaration;
			if (var_dec_node->is_const) {
				declaration += "const ";
			}
			declaration += _prestr(var_dec_node->precision);
			declaration += _typestr(var_dec_node->datatype);

			for (int i = 0; i < var_dec_node->declarations.size(); i++) {
				if (i > 0) {
					declaration += ",";
				}

				declaration += " ";

				declaration += _mkid(var_dec_node->declarations[i].name);

				if (var_dec_node->declarations[i].initializer) {
					declaration += " = ";
					declaration += _dump_node_code(var_dec_node->declarations[i].initializer, p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				}
			}

			code += declaration.as_string();
		} break;

		case SL::Node::TYPE_VARIABLE: {
			SL::VariableNode *var_node = (SL::VariableNode *)p_node;

			if (p_assigning && p_actions.write_flag_pointers.has(var_node->name)) {
				*p_actions.write_flag_pointers[var_node->name] = true;
			}

			if (p_default_actions.usage_defines.has(var_node->name) && !used_name_defines.has(var_node->name)) {
				String define = p_default_actions.usage_defines[var_node->name];
				String node_name = define.substr(1, define.length());

				if (define.begins_with("@")) {
					define = p_default_actions.usage_defines[node_name];
				}

				if (!used_name_defines.has(node_name)) {
					r_gen_code.custom_defines.push_back(define.utf8());
				}
				used_name_defines.insert(var_node->name);
			}

			if (p_actions.usage_flag_pointers.has(var_node->name) && !used_flag_pointers.has(var_node->name)) {
				*p_actions.usage_flag_pointers[var_node->name] = true;
				used_flag_pointers.insert(var_node->name);
			}

			if (p_default_actions.renames.has(var_node->name)) {
				code += p_default_actions.renames[var_node->name];
			} else {
				code += _mkid(var_node->name);
			}

			if (var_node->name == time_name) {
				if (current_func_name == vertex_name) {
					r_gen_code.uses_vertex_time = true;
				}
				if (current_func_name == fragment_name || current_func_name == light_name) {
					r_gen_code.uses_fragment_time = true;
				}
			}
		} break;
		case SL::Node::TYPE_ARRAY_DECLARATION: {
			SL::ArrayDeclarationNode *arr_dec_node = (SL::ArrayDeclarationNode *)p_node;

			StringBuffer<> declaration;
			declaration += _prestr(arr_dec_node->precision);
			declaration += _typestr(arr_dec_node->datatype);

			for (int i = 0; i < arr_dec_node->declarations.size(); i++) {
				if (i > 0) {
					declaration += ",";
				}

				declaration += " ";

				declaration += _mkid(arr_dec_node->declarations[i].name);
				declaration += "[";
				declaration += itos(arr_dec_node->declarations[i].size);
				declaration += "]";
			}

			code += declaration.as_string();
		} break;
		case SL::Node::TYPE_ARRAY: {
			SL::ArrayNode *arr_node = (SL::ArrayNode *)p_node;

			if (p_assigning && p_actions.write_flag_pointers.has(arr_node->name)) {
				*p_actions.write_flag_pointers[arr_node->name] = true;
			}

			if (p_default_actions.usage_defines.has(arr_node->name) && !used_name_defines.has(arr_node->name)) {
				String define = p_default_actions.usage_defines[arr_node->name];
				String node_name = define.substr(1, define.length());

				if (define.begins_with("@")) {
					define = p_default_actions.usage_defines[node_name];
				}

				if (!used_name_defines.has(node_name)) {
					r_gen_code.custom_defines.push_back(define.utf8());
				}
				used_name_defines.insert(arr_node->name);
			}

			if (p_actions.usage_flag_pointers.has(arr_node->name) && !used_flag_pointers.has(arr_node->name)) {
				*p_actions.usage_flag_pointers[arr_node->name] = true;
				used_flag_pointers.insert(arr_node->name);
			}

			if (p_default_actions.renames.has(arr_node->name)) {
				code += p_default_actions.renames[arr_node->name];
			} else {
				code += _mkid(arr_node->name);
			}

			if (arr_node->call_expression != NULL) {
				code += ".";
				code += _dump_node_code(arr_node->call_expression, p_level, r_gen_code, p_actions, p_default_actions, p_assigning, false);
			}

			if (arr_node->index_expression != NULL) {
				code += "[";
				code += _dump_node_code(arr_node->index_expression, p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				code += "]";
			}

			if (arr_node->name == time_name) {
				if (current_func_name == vertex_name) {
					r_gen_code.uses_vertex_time = true;
				}
				if (current_func_name == fragment_name || current_func_name == light_name) {
					r_gen_code.uses_fragment_time = true;
				}
			}

		} break;
		case SL::Node::TYPE_CONSTANT: {
			SL::ConstantNode *const_node = (SL::ConstantNode *)p_node;

			return get_constant_text(const_node->datatype, const_node->values);
		} break;

		case SL::Node::TYPE_OPERATOR: {
			SL::OperatorNode *op_node = (SL::OperatorNode *)p_node;

			switch (op_node->op) {
				case SL::OP_ASSIGN:
				case SL::OP_ASSIGN_ADD:
				case SL::OP_ASSIGN_SUB:
				case SL::OP_ASSIGN_MUL:
				case SL::OP_ASSIGN_DIV:
				case SL::OP_ASSIGN_SHIFT_LEFT:
				case SL::OP_ASSIGN_SHIFT_RIGHT:
				case SL::OP_ASSIGN_BIT_AND:
				case SL::OP_ASSIGN_BIT_OR:
				case SL::OP_ASSIGN_BIT_XOR: {
					code += _dump_node_code(op_node->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, true);
					code += " ";
					code += _opstr(op_node->op);
					code += " ";
					code += _dump_node_code(op_node->arguments[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				} break;

				case SL::OP_ASSIGN_MOD: {
					String a = _dump_node_code(op_node->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					String n = _dump_node_code(op_node->arguments[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += a + " = " + n + " == 0 ? 0 : ";
					code += a + " - " + n + " * (" + a + " / " + n + ")";
				} break;

				case SL::OP_BIT_INVERT:
				case SL::OP_NEGATE:
				case SL::OP_NOT:
				case SL::OP_DECREMENT:
				case SL::OP_INCREMENT: {
					code += _opstr(op_node->op);
					code += _dump_node_code(op_node->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				} break;

				case SL::OP_POST_DECREMENT:
				case SL::OP_POST_INCREMENT: {
					code += _dump_node_code(op_node->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += _opstr(op_node->op);
				} break;

				case SL::OP_CALL:
				case SL::OP_CONSTRUCT: {
					ERR_FAIL_COND_V(op_node->arguments[0]->type != SL::Node::TYPE_VARIABLE, String());

					SL::VariableNode *var_node = (SL::VariableNode *)op_node->arguments[0];

					if (op_node->op == SL::OP_CONSTRUCT) {
						code += var_node->name;
					} else {
						if (var_node->name == "texture") {
							// emit texture call

							if (op_node->arguments[1]->get_datatype() == SL::TYPE_SAMPLER2D) { // ||
								//									op_node->arguments[1]->get_datatype() == SL::TYPE_SAMPLEREXT) {
								code += "texture2D";
							} else if (op_node->arguments[1]->get_datatype() == SL::TYPE_SAMPLERCUBE) {
								code += "textureCube";
							}

						} else if (var_node->name == "textureLod") {
							// emit texture call

							if (op_node->arguments[1]->get_datatype() == SL::TYPE_SAMPLER2D) {
								code += "texture2DLod";
							} else if (op_node->arguments[1]->get_datatype() == SL::TYPE_SAMPLERCUBE) {
								code += "textureCubeLod";
							}

						} else if (var_node->name == "mix") {
							switch (op_node->arguments[3]->get_datatype()) {
								case SL::TYPE_BVEC2: {
									code += "select2";
								} break;

								case SL::TYPE_BVEC3: {
									code += "select3";
								} break;

								case SL::TYPE_BVEC4: {
									code += "select4";
								} break;

								case SL::TYPE_VEC2:
								case SL::TYPE_VEC3:
								case SL::TYPE_VEC4:
								case SL::TYPE_FLOAT: {
									code += "mix";
								} break;

								default: {
									SL::DataType type = op_node->arguments[3]->get_datatype();
									// FIXME: Proper error print or graceful handling
									print_line(String("uhhhh invalid mix with type: ") + itos(type));
								} break;
							}

						} else if (p_default_actions.renames.has(var_node->name)) {
							code += p_default_actions.renames[var_node->name];
						} else if (internal_functions.has(var_node->name)) {
							code += var_node->name;
						} else {
							code += _mkid(var_node->name);
						}
					}

					code += "(";

					for (int i = 1; i < op_node->arguments.size(); i++) {
						if (i > 1) {
							code += ", ";
						}

						code += _dump_node_code(op_node->arguments[i], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					}

					code += ")";

					if (p_default_actions.usage_defines.has(var_node->name) && !used_name_defines.has(var_node->name)) {
						String define = p_default_actions.usage_defines[var_node->name];
						String node_name = define.substr(1, define.length());

						if (define.begins_with("@")) {
							define = p_default_actions.usage_defines[node_name];
						}

						if (!used_name_defines.has(node_name)) {
							r_gen_code.custom_defines.push_back(define.utf8());
						}
						used_name_defines.insert(var_node->name);
					}

				} break;

				case SL::OP_INDEX: {
					code += _dump_node_code(op_node->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += "[";
					code += _dump_node_code(op_node->arguments[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += "]";
				} break;

				case SL::OP_SELECT_IF: {
					code += "(";
					code += _dump_node_code(op_node->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += " ? ";
					code += _dump_node_code(op_node->arguments[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += " : ";
					code += _dump_node_code(op_node->arguments[2], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += ")";
				} break;

				case SL::OP_MOD: {
					String a = _dump_node_code(op_node->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					String n = _dump_node_code(op_node->arguments[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += "(" + n + " == 0 ? 0 : ";
					code += a + " - " + n + " * (" + a + " / " + n + "))";
				} break;

				default: {
					if (p_use_scope) {
						code += "(";
					}
					code += _dump_node_code(op_node->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += " ";
					code += _opstr(op_node->op);
					code += " ";
					code += _dump_node_code(op_node->arguments[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					if (p_use_scope) {
						code += ")";
					}
				} break;
			}
		} break;

		case SL::Node::TYPE_CONTROL_FLOW: {
			SL::ControlFlowNode *cf_node = (SL::ControlFlowNode *)p_node;

			if (cf_node->flow_op == SL::FLOW_OP_IF) {
				code += _mktab(p_level);
				code += "if (";
				code += _dump_node_code(cf_node->expressions[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				code += ")\n";
				code += _dump_node_code(cf_node->blocks[0], p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);

				if (cf_node->blocks.size() == 2) {
					code += _mktab(p_level);
					code += "else\n";
					code += _dump_node_code(cf_node->blocks[1], p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);
				}
			} else if (cf_node->flow_op == SL::FLOW_OP_DO) {
				code += _mktab(p_level);
				code += "do";
				code += _dump_node_code(cf_node->blocks[0], p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);
				code += _mktab(p_level);
				code += "while (";
				code += _dump_node_code(cf_node->expressions[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				code += ");";
			} else if (cf_node->flow_op == SL::FLOW_OP_WHILE) {
				code += _mktab(p_level);
				code += "while (";
				code += _dump_node_code(cf_node->expressions[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				code += ")\n";
				code += _dump_node_code(cf_node->blocks[0], p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);
			} else if (cf_node->flow_op == SL::FLOW_OP_FOR) {
				code += _mktab(p_level);
				code += "for (";
				code += _dump_node_code(cf_node->blocks[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				code += "; ";
				code += _dump_node_code(cf_node->expressions[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				code += "; ";
				code += _dump_node_code(cf_node->expressions[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				code += ")\n";

				code += _dump_node_code(cf_node->blocks[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);

			} else if (cf_node->flow_op == SL::FLOW_OP_RETURN) {
				code += _mktab(p_level);
				code += "return";

				if (cf_node->expressions.size()) {
					code += " ";
					code += _dump_node_code(cf_node->expressions[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				}
				code += ";\n";
			} else if (cf_node->flow_op == SL::FLOW_OP_DISCARD) {
				if (p_actions.usage_flag_pointers.has("DISCARD") && !used_flag_pointers.has("DISCARD")) {
					*p_actions.usage_flag_pointers["DISCARD"] = true;
					used_flag_pointers.insert("DISCARD");
				}
				code += "discard;";
			} else if (cf_node->flow_op == SL::FLOW_OP_CONTINUE) {
				code += "continue;";
			} else if (cf_node->flow_op == SL::FLOW_OP_BREAK) {
				code += "break;";
			}
		} break;

		case SL::Node::TYPE_MEMBER: {
			SL::MemberNode *member_node = (SL::MemberNode *)p_node;
			code += _dump_node_code(member_node->owner, p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
			code += ".";
			code += member_node->name;
		} break;
	}

	return code.as_string();
}

ShaderLanguage::DataType ShaderCompilerGLES2::_get_variable_type(const StringName &p_type) {
	//	RS::GlobalVariableType gvt = ((RasterizerStorageRD *)(RendererStorage::base_singleton))->global_variable_get_type_internal(p_type);
	RS::GlobalVariableType gvt = RS::GLOBAL_VAR_TYPE_MAX;
	return RS::global_variable_type_get_shader_datatype(gvt);
}

Error ShaderCompilerGLES2::compile(GD_VS::ShaderMode p_mode, const String &p_code, IdentifierActions *p_actions, const String &p_path, GeneratedCode &r_gen_code) {
	
	ShaderLanguage::VaryingFunctionNames var_names;
	
	Error err = parser.compile(p_code, ShaderTypes::get_singleton()->get_functions(p_mode), ShaderTypes::get_singleton()->get_modes(p_mode), var_names, ShaderTypes::get_singleton()->get_types(), _get_variable_type);

	//	Error ShaderLanguage::compile(const String &p_code, const Map<StringName, FunctionInfo> &p_functions, const Vector<StringName> &p_render_modes, const Set<String> &p_shader_types, GlobalVariableGetTypeFunc p_global_variable_type_func) {
	if (err != OK) {
		Vector<String> shader = p_code.split("\n");
		for (int i = 0; i < shader.size(); i++) {
			print_line(itos(i + 1) + " " + shader[i]);
		}

		_err_print_error(NULL, p_path.utf8().get_data(), parser.get_error_line(), parser.get_error_text().utf8().get_data(), ERR_HANDLER_SHADER);
		return err;
	}

	r_gen_code.custom_defines.clear();
	r_gen_code.uniforms.clear();
	r_gen_code.texture_uniforms.clear();
	r_gen_code.texture_hints.clear();
	r_gen_code.vertex = String();
	r_gen_code.vertex_global = String();
	r_gen_code.fragment = String();
	r_gen_code.fragment_global = String();
	r_gen_code.light = String();
	r_gen_code.uses_fragment_time = false;
	r_gen_code.uses_vertex_time = false;

	used_name_defines.clear();
	used_rmode_defines.clear();
	used_flag_pointers.clear();

	_dump_node_code(parser.get_shader(), 1, r_gen_code, *p_actions, actions[p_mode], false);

	return OK;
}

ShaderCompilerGLES2::ShaderCompilerGLES2() {
	/** CANVAS ITEM SHADER **/

	actions[GD_VS::SHADER_CANVAS_ITEM].renames["VERTEX"] = "outvec.xy";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["UV"] = "uv";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["POINT_SIZE"] = "point_size";

	actions[GD_VS::SHADER_CANVAS_ITEM].renames["WORLD_MATRIX"] = "modelview_matrix";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["PROJECTION_MATRIX"] = "projection_matrix";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["EXTRA_MATRIX"] = "extra_matrix_instance";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["TIME"] = "time";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["AT_LIGHT_PASS"] = "at_light_pass";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["INSTANCE_CUSTOM"] = "instance_custom";

	actions[GD_VS::SHADER_CANVAS_ITEM].renames["COLOR"] = "color";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["MODULATE"] = "final_modulate";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["NORMAL"] = "normal";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["NORMALMAP"] = "normal_map";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["NORMALMAP_DEPTH"] = "normal_depth";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["TEXTURE"] = "color_texture";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["TEXTURE_PIXEL_SIZE"] = "color_texpixel_size";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["NORMAL_TEXTURE"] = "normal_texture";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["SCREEN_UV"] = "screen_uv";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["SCREEN_TEXTURE"] = "screen_texture";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["SCREEN_PIXEL_SIZE"] = "screen_pixel_size";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["FRAGCOORD"] = "gl_FragCoord";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["POINT_COORD"] = "gl_PointCoord";

	actions[GD_VS::SHADER_CANVAS_ITEM].renames["LIGHT_VEC"] = "light_vec";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["LIGHT_HEIGHT"] = "light_height";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["LIGHT_COLOR"] = "light_color";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["LIGHT_UV"] = "light_uv";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["LIGHT"] = "light";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["SHADOW_COLOR"] = "shadow_color";
	actions[GD_VS::SHADER_CANVAS_ITEM].renames["SHADOW_VEC"] = "shadow_vec";

	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["COLOR"] = "#define COLOR_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["MODULATE"] = "#define MODULATE_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["SCREEN_TEXTURE"] = "#define SCREEN_TEXTURE_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["SCREEN_UV"] = "#define SCREEN_UV_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["SCREEN_PIXEL_SIZE"] = "@SCREEN_UV";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["NORMAL"] = "#define NORMAL_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["NORMALMAP"] = "#define NORMALMAP_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["LIGHT"] = "#define USE_LIGHT_SHADER_CODE\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].render_mode_defines["skip_vertex_transform"] = "#define SKIP_TRANSFORM_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["SHADOW_VEC"] = "#define SHADOW_VEC_USED\n";

	// Ported from GLES3

	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["sinh"] = "#define SINH_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["cosh"] = "#define COSH_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["tanh"] = "#define TANH_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["asinh"] = "#define ASINH_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["acosh"] = "#define ACOSH_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["atanh"] = "#define ATANH_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["determinant"] = "#define DETERMINANT_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["transpose"] = "#define TRANSPOSE_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["outerProduct"] = "#define OUTER_PRODUCT_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["round"] = "#define ROUND_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["roundEven"] = "#define ROUND_EVEN_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["inverse"] = "#define INVERSE_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["isinf"] = "#define IS_INF_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["isnan"] = "#define IS_NAN_USED\n";
	actions[GD_VS::SHADER_CANVAS_ITEM].usage_defines["trunc"] = "#define TRUNC_USED\n";

	/** SPATIAL SHADER **/

	actions[GD_VS::SHADER_SPATIAL].renames["WORLD_MATRIX"] = "world_transform";
	actions[GD_VS::SHADER_SPATIAL].renames["INV_CAMERA_MATRIX"] = "camera_inverse_matrix";
	actions[GD_VS::SHADER_SPATIAL].renames["CAMERA_MATRIX"] = "camera_matrix";
	actions[GD_VS::SHADER_SPATIAL].renames["PROJECTION_MATRIX"] = "projection_matrix";
	actions[GD_VS::SHADER_SPATIAL].renames["INV_PROJECTION_MATRIX"] = "projection_inverse_matrix";
	actions[GD_VS::SHADER_SPATIAL].renames["MODELVIEW_MATRIX"] = "modelview";

	actions[GD_VS::SHADER_SPATIAL].renames["VERTEX"] = "vertex.xyz";
	actions[GD_VS::SHADER_SPATIAL].renames["NORMAL"] = "normal";
	actions[GD_VS::SHADER_SPATIAL].renames["TANGENT"] = "tangent";
	actions[GD_VS::SHADER_SPATIAL].renames["BINORMAL"] = "binormal";
	actions[GD_VS::SHADER_SPATIAL].renames["POSITION"] = "position";
	actions[GD_VS::SHADER_SPATIAL].renames["UV"] = "uv_interp";
	actions[GD_VS::SHADER_SPATIAL].renames["UV2"] = "uv2_interp";
	actions[GD_VS::SHADER_SPATIAL].renames["COLOR"] = "color_interp";
	actions[GD_VS::SHADER_SPATIAL].renames["POINT_SIZE"] = "point_size";
	// gl_InstanceID is not available in OpenGL ES 2.0
	actions[GD_VS::SHADER_SPATIAL].renames["INSTANCE_ID"] = "0";

	//builtins

	actions[GD_VS::SHADER_SPATIAL].renames["TIME"] = "time";
	actions[GD_VS::SHADER_SPATIAL].renames["VIEWPORT_SIZE"] = "viewport_size";

	actions[GD_VS::SHADER_SPATIAL].renames["FRAGCOORD"] = "gl_FragCoord";
	actions[GD_VS::SHADER_SPATIAL].renames["FRONT_FACING"] = "gl_FrontFacing";
	actions[GD_VS::SHADER_SPATIAL].renames["NORMALMAP"] = "normalmap";
	actions[GD_VS::SHADER_SPATIAL].renames["NORMALMAP_DEPTH"] = "normaldepth";
	actions[GD_VS::SHADER_SPATIAL].renames["ALBEDO"] = "albedo";
	actions[GD_VS::SHADER_SPATIAL].renames["ALPHA"] = "alpha";
	actions[GD_VS::SHADER_SPATIAL].renames["METALLIC"] = "metallic";
	actions[GD_VS::SHADER_SPATIAL].renames["SPECULAR"] = "specular";
	actions[GD_VS::SHADER_SPATIAL].renames["ROUGHNESS"] = "roughness";
	actions[GD_VS::SHADER_SPATIAL].renames["RIM"] = "rim";
	actions[GD_VS::SHADER_SPATIAL].renames["RIM_TINT"] = "rim_tint";
	actions[GD_VS::SHADER_SPATIAL].renames["CLEARCOAT"] = "clearcoat";
	actions[GD_VS::SHADER_SPATIAL].renames["CLEARCOAT_GLOSS"] = "clearcoat_gloss";
	actions[GD_VS::SHADER_SPATIAL].renames["ANISOTROPY"] = "anisotropy";
	actions[GD_VS::SHADER_SPATIAL].renames["ANISOTROPY_FLOW"] = "anisotropy_flow";
	actions[GD_VS::SHADER_SPATIAL].renames["SSS_STRENGTH"] = "sss_strength";
	actions[GD_VS::SHADER_SPATIAL].renames["TRANSMISSION"] = "transmission";
	actions[GD_VS::SHADER_SPATIAL].renames["AO"] = "ao";
	actions[GD_VS::SHADER_SPATIAL].renames["AO_LIGHT_AFFECT"] = "ao_light_affect";
	actions[GD_VS::SHADER_SPATIAL].renames["EMISSION"] = "emission";
	actions[GD_VS::SHADER_SPATIAL].renames["POINT_COORD"] = "gl_PointCoord";
	actions[GD_VS::SHADER_SPATIAL].renames["INSTANCE_CUSTOM"] = "instance_custom";
	actions[GD_VS::SHADER_SPATIAL].renames["SCREEN_UV"] = "screen_uv";
	actions[GD_VS::SHADER_SPATIAL].renames["SCREEN_TEXTURE"] = "screen_texture";
	actions[GD_VS::SHADER_SPATIAL].renames["DEPTH_TEXTURE"] = "depth_texture";
	// Defined in GLES3, but not available in GLES2
	//actions[GD_VS::SHADER_SPATIAL].renames["DEPTH"] = "gl_FragDepth";
	actions[GD_VS::SHADER_SPATIAL].renames["ALPHA_SCISSOR"] = "alpha_scissor";
	actions[GD_VS::SHADER_SPATIAL].renames["OUTPUT_IS_SRGB"] = "SHADER_IS_SRGB";

	//for light
	actions[GD_VS::SHADER_SPATIAL].renames["VIEW"] = "view";
	actions[GD_VS::SHADER_SPATIAL].renames["LIGHT_COLOR"] = "light_color";
	actions[GD_VS::SHADER_SPATIAL].renames["LIGHT"] = "light";
	actions[GD_VS::SHADER_SPATIAL].renames["ATTENUATION"] = "attenuation";
	actions[GD_VS::SHADER_SPATIAL].renames["DIFFUSE_LIGHT"] = "diffuse_light";
	actions[GD_VS::SHADER_SPATIAL].renames["SPECULAR_LIGHT"] = "specular_light";

	actions[GD_VS::SHADER_SPATIAL].usage_defines["TANGENT"] = "#define ENABLE_TANGENT_INTERP\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["BINORMAL"] = "@TANGENT";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["RIM"] = "#define LIGHT_USE_RIM\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["RIM_TINT"] = "@RIM";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["CLEARCOAT"] = "#define LIGHT_USE_CLEARCOAT\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["CLEARCOAT_GLOSS"] = "@CLEARCOAT";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["ANISOTROPY"] = "#define LIGHT_USE_ANISOTROPY\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["ANISOTROPY_FLOW"] = "@ANISOTROPY";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["AO"] = "#define ENABLE_AO\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["AO_LIGHT_AFFECT"] = "#define ENABLE_AO\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["UV"] = "#define ENABLE_UV_INTERP\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["UV2"] = "#define ENABLE_UV2_INTERP\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["NORMALMAP"] = "#define ENABLE_NORMALMAP\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["NORMALMAP_DEPTH"] = "@NORMALMAP";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["COLOR"] = "#define ENABLE_COLOR_INTERP\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["INSTANCE_CUSTOM"] = "#define ENABLE_INSTANCE_CUSTOM\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["ALPHA_SCISSOR"] = "#define ALPHA_SCISSOR_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["POSITION"] = "#define OVERRIDE_POSITION\n";

	actions[GD_VS::SHADER_SPATIAL].usage_defines["SSS_STRENGTH"] = "#define ENABLE_SSS\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["TRANSMISSION"] = "#define TRANSMISSION_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["SCREEN_TEXTURE"] = "#define SCREEN_TEXTURE_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["DEPTH_TEXTURE"] = "#define DEPTH_TEXTURE_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["SCREEN_UV"] = "#define SCREEN_UV_USED\n";

	actions[GD_VS::SHADER_SPATIAL].usage_defines["DIFFUSE_LIGHT"] = "#define USE_LIGHT_SHADER_CODE\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["SPECULAR_LIGHT"] = "#define USE_LIGHT_SHADER_CODE\n";

	// Ported from GLES3

	actions[GD_VS::SHADER_SPATIAL].usage_defines["sinh"] = "#define SINH_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["cosh"] = "#define COSH_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["tanh"] = "#define TANH_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["asinh"] = "#define ASINH_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["acosh"] = "#define ACOSH_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["atanh"] = "#define ATANH_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["determinant"] = "#define DETERMINANT_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["transpose"] = "#define TRANSPOSE_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["outerProduct"] = "#define OUTER_PRODUCT_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["round"] = "#define ROUND_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["roundEven"] = "#define ROUND_EVEN_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["inverse"] = "#define INVERSE_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["isinf"] = "#define IS_INF_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["isnan"] = "#define IS_NAN_USED\n";
	actions[GD_VS::SHADER_SPATIAL].usage_defines["trunc"] = "#define TRUNC_USED\n";

	actions[GD_VS::SHADER_SPATIAL].render_mode_defines["skip_vertex_transform"] = "#define SKIP_TRANSFORM_USED\n";
	actions[GD_VS::SHADER_SPATIAL].render_mode_defines["world_vertex_coords"] = "#define VERTEX_WORLD_COORDS_USED\n";

	// Defined in GLES3, could be implemented in GLES2 too if there's a need for it
	//actions[GD_VS::SHADER_SPATIAL].render_mode_defines["ensure_correct_normals"] = "#define ENSURE_CORRECT_NORMALS\n";
	// Defined in GLES3, might not be possible in GLES2 as gl_FrontFacing is not available
	//actions[GD_VS::SHADER_SPATIAL].render_mode_defines["cull_front"] = "#define DO_SIDE_CHECK\n";
	//actions[GD_VS::SHADER_SPATIAL].render_mode_defines["cull_disabled"] = "#define DO_SIDE_CHECK\n";

	bool force_lambert = GLOBAL_GET("rendering/quality/shading/force_lambert_over_burley");

	if (!force_lambert) {
		actions[GD_VS::SHADER_SPATIAL].render_mode_defines["diffuse_burley"] = "#define DIFFUSE_BURLEY\n";
	}

	actions[GD_VS::SHADER_SPATIAL].render_mode_defines["diffuse_oren_nayar"] = "#define DIFFUSE_OREN_NAYAR\n";
	actions[GD_VS::SHADER_SPATIAL].render_mode_defines["diffuse_lambert_wrap"] = "#define DIFFUSE_LAMBERT_WRAP\n";
	actions[GD_VS::SHADER_SPATIAL].render_mode_defines["diffuse_toon"] = "#define DIFFUSE_TOON\n";

	bool force_blinn = GLOBAL_GET("rendering/quality/shading/force_blinn_over_ggx");

	if (!force_blinn) {
		actions[GD_VS::SHADER_SPATIAL].render_mode_defines["specular_schlick_ggx"] = "#define SPECULAR_SCHLICK_GGX\n";
	} else {
		actions[GD_VS::SHADER_SPATIAL].render_mode_defines["specular_schlick_ggx"] = "#define SPECULAR_BLINN\n";
	}

	actions[GD_VS::SHADER_SPATIAL].render_mode_defines["specular_blinn"] = "#define SPECULAR_BLINN\n";
	actions[GD_VS::SHADER_SPATIAL].render_mode_defines["specular_phong"] = "#define SPECULAR_PHONG\n";
	actions[GD_VS::SHADER_SPATIAL].render_mode_defines["specular_toon"] = "#define SPECULAR_TOON\n";
	actions[GD_VS::SHADER_SPATIAL].render_mode_defines["specular_disabled"] = "#define SPECULAR_DISABLED\n";
	actions[GD_VS::SHADER_SPATIAL].render_mode_defines["shadows_disabled"] = "#define SHADOWS_DISABLED\n";
	actions[GD_VS::SHADER_SPATIAL].render_mode_defines["ambient_light_disabled"] = "#define AMBIENT_LIGHT_DISABLED\n";
	actions[GD_VS::SHADER_SPATIAL].render_mode_defines["shadow_to_opacity"] = "#define USE_SHADOW_TO_OPACITY\n";

	// No defines for particle shaders in GLES2, there are no GPU particles

	vertex_name = "vertex";
	fragment_name = "fragment";
	light_name = "light";
	time_name = "TIME";

	List<String> func_list;

	ShaderLanguage::get_builtin_funcs(&func_list);

	for (List<String>::Element *E = func_list.front(); E; E = E->next()) {
		internal_functions.insert(E->get());
	}
}

#endif // GLES2_BACKEND_ENABLED

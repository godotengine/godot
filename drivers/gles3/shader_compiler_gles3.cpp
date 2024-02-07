/**************************************************************************/
/*  shader_compiler_gles3.cpp                                             */
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

#include "shader_compiler_gles3.h"

#include "core/os/os.h"
#include "core/project_settings.h"

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

static int _get_datatype_size(SL::DataType p_type) {
	switch (p_type) {
		case SL::TYPE_VOID:
			return 0;
		case SL::TYPE_BOOL:
			return 4;
		case SL::TYPE_BVEC2:
			return 8;
		case SL::TYPE_BVEC3:
			return 12;
		case SL::TYPE_BVEC4:
			return 16;
		case SL::TYPE_INT:
			return 4;
		case SL::TYPE_IVEC2:
			return 8;
		case SL::TYPE_IVEC3:
			return 12;
		case SL::TYPE_IVEC4:
			return 16;
		case SL::TYPE_UINT:
			return 4;
		case SL::TYPE_UVEC2:
			return 8;
		case SL::TYPE_UVEC3:
			return 12;
		case SL::TYPE_UVEC4:
			return 16;
		case SL::TYPE_FLOAT:
			return 4;
		case SL::TYPE_VEC2:
			return 8;
		case SL::TYPE_VEC3:
			return 12;
		case SL::TYPE_VEC4:
			return 16;
		case SL::TYPE_MAT2:
			return 32; //4 * 4 + 4 * 4
		case SL::TYPE_MAT3:
			return 48; // 4 * 4 + 4 * 4 + 4 * 4
		case SL::TYPE_MAT4:
			return 64;
		case SL::TYPE_SAMPLER2D:
			return 16;
		case SL::TYPE_ISAMPLER2D:
			return 16;
		case SL::TYPE_USAMPLER2D:
			return 16;
		case SL::TYPE_SAMPLER2DARRAY:
			return 16;
		case SL::TYPE_ISAMPLER2DARRAY:
			return 16;
		case SL::TYPE_USAMPLER2DARRAY:
			return 16;
		case SL::TYPE_SAMPLER3D:
			return 16;
		case SL::TYPE_ISAMPLER3D:
			return 16;
		case SL::TYPE_USAMPLER3D:
			return 16;
		case SL::TYPE_SAMPLERCUBE:
			return 16;
		case SL::TYPE_SAMPLEREXT:
			return 16;
		case SL::TYPE_STRUCT:
			return 0;
		default: {
		}
	}

	ERR_FAIL_V(0);
}

static int _get_datatype_alignment(SL::DataType p_type) {
	switch (p_type) {
		case SL::TYPE_VOID:
			return 0;
		case SL::TYPE_BOOL:
			return 4;
		case SL::TYPE_BVEC2:
			return 8;
		case SL::TYPE_BVEC3:
			return 16;
		case SL::TYPE_BVEC4:
			return 16;
		case SL::TYPE_INT:
			return 4;
		case SL::TYPE_IVEC2:
			return 8;
		case SL::TYPE_IVEC3:
			return 16;
		case SL::TYPE_IVEC4:
			return 16;
		case SL::TYPE_UINT:
			return 4;
		case SL::TYPE_UVEC2:
			return 8;
		case SL::TYPE_UVEC3:
			return 16;
		case SL::TYPE_UVEC4:
			return 16;
		case SL::TYPE_FLOAT:
			return 4;
		case SL::TYPE_VEC2:
			return 8;
		case SL::TYPE_VEC3:
			return 16;
		case SL::TYPE_VEC4:
			return 16;
		case SL::TYPE_MAT2:
			return 16;
		case SL::TYPE_MAT3:
			return 16;
		case SL::TYPE_MAT4:
			return 16;
		case SL::TYPE_SAMPLER2D:
			return 16;
		case SL::TYPE_ISAMPLER2D:
			return 16;
		case SL::TYPE_USAMPLER2D:
			return 16;
		case SL::TYPE_SAMPLER2DARRAY:
			return 16;
		case SL::TYPE_ISAMPLER2DARRAY:
			return 16;
		case SL::TYPE_USAMPLER2DARRAY:
			return 16;
		case SL::TYPE_SAMPLER3D:
			return 16;
		case SL::TYPE_ISAMPLER3D:
			return 16;
		case SL::TYPE_USAMPLER3D:
			return 16;
		case SL::TYPE_SAMPLERCUBE:
			return 16;
		case SL::TYPE_SAMPLEREXT:
			return 16;
		case SL::TYPE_STRUCT:
			return 0;
		default: {
		}
	}

	ERR_FAIL_V(0);
}
static String _interpstr(SL::DataInterpolation p_interp) {
	switch (p_interp) {
		case SL::INTERPOLATION_FLAT:
			return "flat ";
		case SL::INTERPOLATION_SMOOTH:
			return "";
	}
	return "";
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

static String _constr(bool p_is_const) {
	if (p_is_const) {
		return "const ";
	}
	return "";
}

static String _qualstr(SL::ArgumentQualifier p_qual) {
	switch (p_qual) {
		case SL::ARGUMENT_QUALIFIER_IN:
			return "";
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
			String text = "bvec" + itos(p_type - SL::TYPE_BOOL + 1) + "(";
			for (int i = 0; i < p_values.size(); i++) {
				if (i > 0) {
					text += ",";
				}

				text += p_values[i].boolean ? "true" : "false";
			}
			text += ")";
			return text;
		}

		case SL::TYPE_INT:
			return itos(p_values[0].sint);
		case SL::TYPE_IVEC2:
		case SL::TYPE_IVEC3:
		case SL::TYPE_IVEC4: {
			String text = "ivec" + itos(p_type - SL::TYPE_INT + 1) + "(";
			for (int i = 0; i < p_values.size(); i++) {
				if (i > 0) {
					text += ",";
				}

				text += itos(p_values[i].sint);
			}
			text += ")";
			return text;

		} break;
		case SL::TYPE_UINT:
			return itos(p_values[0].uint) + "u";
		case SL::TYPE_UVEC2:
		case SL::TYPE_UVEC3:
		case SL::TYPE_UVEC4: {
			String text = "uvec" + itos(p_type - SL::TYPE_UINT + 1) + "(";
			for (int i = 0; i < p_values.size(); i++) {
				if (i > 0) {
					text += ",";
				}

				text += itos(p_values[i].uint) + "u";
			}
			text += ")";
			return text;
		} break;
		case SL::TYPE_FLOAT:
			return f2sp0(p_values[0].real);
		case SL::TYPE_VEC2:
		case SL::TYPE_VEC3:
		case SL::TYPE_VEC4: {
			String text = "vec" + itos(p_type - SL::TYPE_FLOAT + 1) + "(";
			for (int i = 0; i < p_values.size(); i++) {
				if (i > 0) {
					text += ",";
				}

				text += f2sp0(p_values[i].real);
			}
			text += ")";
			return text;

		} break;
		case SL::TYPE_MAT2:
		case SL::TYPE_MAT3:
		case SL::TYPE_MAT4: {
			String text = "mat" + itos(p_type - SL::TYPE_MAT2 + 2) + "(";
			for (int i = 0; i < p_values.size(); i++) {
				if (i > 0) {
					text += ",";
				}

				text += f2sp0(p_values[i].real);
			}
			text += ")";
			return text;

		} break;
		default:
			ERR_FAIL_V(String());
	}
}

void ShaderCompilerGLES3::_dump_function_deps(const SL::ShaderNode *p_node, const StringName &p_for_func, const Map<StringName, String> &p_func_code, String &r_to_add, Set<StringName> &added) {
	int fidx = -1;

	for (int i = 0; i < p_node->functions.size(); i++) {
		if (p_node->functions[i].name == p_for_func) {
			fidx = i;
			break;
		}
	}

	ERR_FAIL_COND(fidx == -1);

	for (int ufidx = 0; ufidx < p_node->functions[fidx].uses_function.size(); ufidx++) {
		StringName function_name = p_node->functions[fidx].uses_function[ufidx];
		if (added.has(function_name)) {
			continue; //was added already
		}

		_dump_function_deps(p_node, function_name, p_func_code, r_to_add, added);

		SL::FunctionNode *fnode = nullptr;

		for (int i = 0; i < p_node->functions.size(); i++) {
			if (p_node->functions[i].name == function_name) {
				fnode = p_node->functions[i].function;
				break;
			}
		}

		ERR_FAIL_COND(!fnode);

		r_to_add += "\n";

		String header;
		if (fnode->return_type == SL::TYPE_STRUCT) {
			header = _mkid(fnode->return_struct_name) + " " + _mkid(fnode->name) + "(";
		} else {
			header = _typestr(fnode->return_type) + " " + _mkid(fnode->name) + "(";
		}
		for (int i = 0; i < fnode->arguments.size(); i++) {
			if (i > 0) {
				header += ", ";
			}

			header += _constr(fnode->arguments[i].is_const);

			if (fnode->arguments[i].type == SL::TYPE_STRUCT) {
				header += _qualstr(fnode->arguments[i].qualifier) + _mkid(fnode->arguments[i].type_str) + " " + _mkid(fnode->arguments[i].name);
			} else {
				header += _qualstr(fnode->arguments[i].qualifier) + _prestr(fnode->arguments[i].precision) + _typestr(fnode->arguments[i].type) + " " + _mkid(fnode->arguments[i].name);
			}
		}

		header += ")\n";
		r_to_add += header;
		r_to_add += p_func_code[function_name];

		added.insert(function_name);
	}
}

String ShaderCompilerGLES3::_dump_node_code(const SL::Node *p_node, int p_level, GeneratedCode &r_gen_code, IdentifierActions &p_actions, const DefaultIdentifierActions &p_default_actions, bool p_assigning, bool p_use_scope) {
	String code;

	switch (p_node->type) {
		case SL::Node::TYPE_SHADER: {
			SL::ShaderNode *pnode = (SL::ShaderNode *)p_node;

			for (int i = 0; i < pnode->render_modes.size(); i++) {
				if (p_default_actions.render_mode_defines.has(pnode->render_modes[i]) && !used_rmode_defines.has(pnode->render_modes[i])) {
					r_gen_code.defines.push_back(p_default_actions.render_mode_defines[pnode->render_modes[i]].utf8());
					used_rmode_defines.insert(pnode->render_modes[i]);
				}

				if (p_actions.render_mode_flags.has(pnode->render_modes[i])) {
					*p_actions.render_mode_flags[pnode->render_modes[i]] = true;
				}

				if (p_actions.render_mode_values.has(pnode->render_modes[i])) {
					Pair<int *, int> &p = p_actions.render_mode_values[pnode->render_modes[i]];
					*p.first = p.second;
				}
			}

			// structs

			for (int i = 0; i < pnode->vstructs.size(); i++) {
				SL::StructNode *st = pnode->vstructs[i].shader_struct;
				String struct_code;

				struct_code += "struct ";
				struct_code += _mkid(pnode->vstructs[i].name);
				struct_code += " ";
				struct_code += "{\n";
				for (int j = 0; j < st->members.size(); j++) {
					SL::MemberNode *m = st->members[j];
					if (m->datatype == SL::TYPE_STRUCT) {
						struct_code += _mkid(m->struct_name);
					} else {
						struct_code += _prestr(m->precision);
						struct_code += _typestr(m->datatype);
					}
					struct_code += " ";
					struct_code += m->name;
					if (m->array_size > 0) {
						struct_code += "[";
						struct_code += itos(m->array_size);
						struct_code += "]";
					}
					struct_code += ";\n";
				}
				struct_code += "}";
				struct_code += ";\n";

				r_gen_code.vertex_global += struct_code;
				r_gen_code.fragment_global += struct_code;
			}

			int max_texture_uniforms = 0;
			int max_uniforms = 0;

			for (OrderedHashMap<StringName, SL::ShaderNode::Uniform>::Element E = pnode->uniforms.front(); E; E = E.next()) {
				if (SL::is_sampler_type(E.get().type)) {
					max_texture_uniforms++;
				} else {
					max_uniforms++;
				}
			}

			r_gen_code.texture_uniforms.resize(max_texture_uniforms);
			r_gen_code.texture_hints.resize(max_texture_uniforms);
			r_gen_code.texture_types.resize(max_texture_uniforms);

			Vector<int> uniform_sizes;
			Vector<int> uniform_alignments;
			Vector<StringName> uniform_defines;
			uniform_sizes.resize(max_uniforms);
			uniform_alignments.resize(max_uniforms);
			uniform_defines.resize(max_uniforms);
			bool uses_uniforms = false;

			for (OrderedHashMap<StringName, SL::ShaderNode::Uniform>::Element E = pnode->uniforms.front(); E; E = E.next()) {
				String ucode;

				if (SL::is_sampler_type(E.get().type)) {
					ucode = "uniform ";
				}

				ucode += _prestr(E.get().precision);
				ucode += _typestr(E.get().type);
				ucode += " " + _mkid(E.key());
				ucode += ";\n";
				if (SL::is_sampler_type(E.get().type)) {
					r_gen_code.vertex_global += ucode;
					r_gen_code.fragment_global += ucode;
					r_gen_code.texture_uniforms.write[E.get().texture_order] = _mkid(E.key());
					r_gen_code.texture_hints.write[E.get().texture_order] = E.get().hint;
					r_gen_code.texture_types.write[E.get().texture_order] = E.get().type;
				} else {
					if (!uses_uniforms) {
						r_gen_code.defines.push_back(String("#define USE_MATERIAL\n").ascii());
						uses_uniforms = true;
					}
					uniform_defines.write[E.get().order] = ucode;
					uniform_sizes.write[E.get().order] = _get_datatype_size(E.get().type);
					uniform_alignments.write[E.get().order] = _get_datatype_alignment(E.get().type);
				}

				p_actions.uniforms->insert(E.key(), E.get());
			}

			for (int i = 0; i < max_uniforms; i++) {
				r_gen_code.uniforms += uniform_defines[i];
			}

			// add up
			int offset = 0;
			for (int i = 0; i < uniform_sizes.size(); i++) {
				int align = offset % uniform_alignments[i];

				if (align != 0) {
					offset += uniform_alignments[i] - align;
				}

				r_gen_code.uniform_offsets.push_back(offset);

				offset += uniform_sizes[i];
			}

			r_gen_code.uniform_total_size = offset;
			if (r_gen_code.uniform_total_size % 16 != 0) { //UBO sizes must be multiples of 16
				r_gen_code.uniform_total_size += r_gen_code.uniform_total_size % 16;
			}

			List<Pair<StringName, SL::ShaderNode::Varying>> var_frag_to_light;

			for (OrderedHashMap<StringName, SL::ShaderNode::Varying>::Element E = pnode->varyings.front(); E; E = E.next()) {
				if (E.get().stage == SL::ShaderNode::Varying::STAGE_FRAGMENT_TO_LIGHT || E.get().stage == SL::ShaderNode::Varying::STAGE_FRAGMENT) {
					var_frag_to_light.push_back(Pair<StringName, SL::ShaderNode::Varying>(E.key(), E.get()));
					fragment_varyings.insert(E.key());
					continue;
				}
				String vcode;
				String interp_mode = _interpstr(E.get().interpolation);
				vcode += _prestr(E.get().precision);
				vcode += _typestr(E.get().type);
				vcode += " " + _mkid(E.key());
				if (E.get().array_size > 0) {
					vcode += "[";
					vcode += itos(E.get().array_size);
					vcode += "]";
				}
				vcode += ";\n";
				r_gen_code.vertex_global += interp_mode + "out " + vcode;
				r_gen_code.fragment_global += interp_mode + "in " + vcode;
			}

			if (var_frag_to_light.size() > 0) {
				String gcode = "\n\nstruct {\n";
				for (List<Pair<StringName, SL::ShaderNode::Varying>>::Element *E = var_frag_to_light.front(); E; E = E->next()) {
					gcode += "\t" + _prestr(E->get().second.precision) + _typestr(E->get().second.type) + " " + _mkid(E->get().first);
					if (E->get().second.array_size > 0) {
						gcode += "[";
						gcode += itos(E->get().second.array_size);
						gcode += "]";
					}
					gcode += ";\n";
				}
				gcode += "} frag_to_light;\n";
				r_gen_code.fragment_global += gcode;
			}

			for (int i = 0; i < pnode->vconstants.size(); i++) {
				const SL::ShaderNode::Constant &cnode = pnode->vconstants[i];
				String gcode;
				gcode += _constr(true);
				if (pnode->vconstants[i].type == SL::TYPE_STRUCT) {
					gcode += _mkid(cnode.type_str);
				} else {
					gcode += _prestr(cnode.precision);
					gcode += _typestr(cnode.type);
				}
				gcode += " " + _mkid(String(cnode.name));
				if (cnode.array_size > 0) {
					gcode += "[";
					gcode += itos(cnode.array_size);
					gcode += "]";
				}
				gcode += "=";
				gcode += _dump_node_code(cnode.initializer, p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				gcode += ";\n";
				r_gen_code.vertex_global += gcode;
				r_gen_code.fragment_global += gcode;
			}

			Map<StringName, String> function_code;

			//code for functions
			for (int i = 0; i < pnode->functions.size(); i++) {
				SL::FunctionNode *fnode = pnode->functions[i].function;
				function = fnode;
				current_func_name = fnode->name;
				function_code[fnode->name] = _dump_node_code(fnode->body, p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);
				function = nullptr;
			}

			//place functions in actual code

			Set<StringName> added_vtx;
			Set<StringName> added_fragment; //share for light

			for (int i = 0; i < pnode->functions.size(); i++) {
				SL::FunctionNode *fnode = pnode->functions[i].function;

				function = fnode;

				current_func_name = fnode->name;

				if (fnode->name == vertex_name) {
					_dump_function_deps(pnode, fnode->name, function_code, r_gen_code.vertex_global, added_vtx);
					r_gen_code.vertex = function_code[vertex_name];
				}

				if (fnode->name == fragment_name) {
					_dump_function_deps(pnode, fnode->name, function_code, r_gen_code.fragment_global, added_fragment);
					r_gen_code.fragment = function_code[fragment_name];
				}

				if (fnode->name == light_name) {
					_dump_function_deps(pnode, fnode->name, function_code, r_gen_code.fragment_global, added_fragment);
					r_gen_code.light = function_code[light_name];
				}

				function = nullptr;
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
			if (!bnode->single_statement) {
				code += _mktab(p_level - 1) + "{\n";
			}

			for (int i = 0; i < bnode->statements.size(); i++) {
				String scode = _dump_node_code(bnode->statements[i], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);

				if (bnode->statements[i]->type == SL::Node::TYPE_CONTROL_FLOW || bnode->single_statement) {
					code += scode; //use directly
				} else {
					code += _mktab(p_level) + scode + ";\n";
				}
			}
			if (!bnode->single_statement) {
				code += _mktab(p_level - 1) + "}\n";
			}

		} break;
		case SL::Node::TYPE_VARIABLE_DECLARATION: {
			SL::VariableDeclarationNode *vdnode = (SL::VariableDeclarationNode *)p_node;

			String declaration;
			declaration += _constr(vdnode->is_const);
			if (vdnode->datatype == SL::TYPE_STRUCT) {
				declaration += _mkid(vdnode->struct_name);
			} else {
				declaration += _prestr(vdnode->precision);
				declaration += _typestr(vdnode->datatype);
			}
			for (int i = 0; i < vdnode->declarations.size(); i++) {
				if (i > 0) {
					declaration += ",";
				} else {
					declaration += " ";
				}
				declaration += _mkid(vdnode->declarations[i].name);
				if (vdnode->declarations[i].initializer) {
					declaration += "=";
					declaration += _dump_node_code(vdnode->declarations[i].initializer, p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				}
			}

			code += declaration;
		} break;
		case SL::Node::TYPE_VARIABLE: {
			SL::VariableNode *vnode = (SL::VariableNode *)p_node;
			bool use_fragment_varying = false;

			if (!vnode->is_local && current_func_name != vertex_name) {
				if (p_assigning) {
					if (shader->varyings.has(vnode->name)) {
						use_fragment_varying = true;
					}
				} else {
					if (fragment_varyings.has(vnode->name)) {
						use_fragment_varying = true;
					}
				}
			}

			if (p_assigning && p_actions.write_flag_pointers.has(vnode->name)) {
				*p_actions.write_flag_pointers[vnode->name] = true;
			}

			if (p_default_actions.usage_defines.has(vnode->name) && !used_name_defines.has(vnode->name)) {
				String define = p_default_actions.usage_defines[vnode->name];
				if (define.begins_with("@")) {
					define = p_default_actions.usage_defines[define.substr(1, define.length())];
				}
				r_gen_code.defines.push_back(define.utf8());
				used_name_defines.insert(vnode->name);
			}

			if (p_actions.usage_flag_pointers.has(vnode->name) && !used_flag_pointers.has(vnode->name)) {
				*p_actions.usage_flag_pointers[vnode->name] = true;
				used_flag_pointers.insert(vnode->name);
			}

			if (p_default_actions.renames.has(vnode->name)) {
				code = p_default_actions.renames[vnode->name];
			} else if (use_fragment_varying) {
				code = "frag_to_light." + _mkid(vnode->name);
			} else {
				code = _mkid(vnode->name);
			}

			if (vnode->name == time_name) {
				if (current_func_name == vertex_name) {
					r_gen_code.uses_vertex_time = true;
				}
				if (current_func_name == fragment_name || current_func_name == light_name) {
					r_gen_code.uses_fragment_time = true;
				}
			}

		} break;
		case SL::Node::TYPE_ARRAY_CONSTRUCT: {
			SL::ArrayConstructNode *acnode = (SL::ArrayConstructNode *)p_node;
			int sz = acnode->initializer.size();
			if (acnode->datatype == SL::TYPE_STRUCT) {
				code += _mkid(acnode->struct_name);
			} else {
				code += _typestr(acnode->datatype);
			}
			code += "[";
			code += itos(acnode->initializer.size());
			code += "]";
			code += "(";
			for (int i = 0; i < sz; i++) {
				code += _dump_node_code(acnode->initializer[i], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				if (i != sz - 1) {
					code += ", ";
				}
			}
			code += ")";
		} break;
		case SL::Node::TYPE_ARRAY_DECLARATION: {
			SL::ArrayDeclarationNode *adnode = (SL::ArrayDeclarationNode *)p_node;

			String declaration;
			if (adnode->is_const) {
				declaration += "const ";
			}
			if (adnode->datatype == SL::TYPE_STRUCT) {
				declaration += _mkid(adnode->struct_name);
			} else {
				declaration += _prestr(adnode->precision);
				declaration += _typestr(adnode->datatype);
			}
			for (int i = 0; i < adnode->declarations.size(); i++) {
				if (i > 0) {
					declaration += ",";
				} else {
					declaration += " ";
				}
				declaration += _mkid(adnode->declarations[i].name);
				declaration += "[";
				declaration += itos(adnode->declarations[i].size);
				declaration += "]";
				int sz = adnode->declarations[i].initializer.size();
				if (sz > 0) {
					declaration += "=";
					if (adnode->datatype == SL::TYPE_STRUCT) {
						declaration += _mkid(adnode->struct_name);
					} else {
						declaration += _typestr(adnode->datatype);
					}
					declaration += "[";
					declaration += itos(sz);
					declaration += "]";
					declaration += "(";
					for (int j = 0; j < sz; j++) {
						declaration += _dump_node_code(adnode->declarations[i].initializer[j], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
						if (j != sz - 1) {
							declaration += ", ";
						}
					}
					declaration += ")";
				}
			}

			code += declaration;
		} break;
		case SL::Node::TYPE_ARRAY: {
			SL::ArrayNode *anode = (SL::ArrayNode *)p_node;
			bool use_fragment_varying = false;

			if (!anode->is_local && current_func_name != vertex_name) {
				if (anode->assign_expression != nullptr) {
					use_fragment_varying = true;
				} else {
					if (p_assigning) {
						if (shader->varyings.has(anode->name)) {
							use_fragment_varying = true;
						}
					} else {
						if (fragment_varyings.has(anode->name)) {
							use_fragment_varying = true;
						}
					}
				}
			}

			if (p_assigning && p_actions.write_flag_pointers.has(anode->name)) {
				*p_actions.write_flag_pointers[anode->name] = true;
			}

			if (p_default_actions.usage_defines.has(anode->name) && !used_name_defines.has(anode->name)) {
				String define = p_default_actions.usage_defines[anode->name];
				if (define.begins_with("@")) {
					define = p_default_actions.usage_defines[define.substr(1, define.length())];
				}
				r_gen_code.defines.push_back(define.utf8());
				used_name_defines.insert(anode->name);
			}

			if (p_actions.usage_flag_pointers.has(anode->name) && !used_flag_pointers.has(anode->name)) {
				*p_actions.usage_flag_pointers[anode->name] = true;
				used_flag_pointers.insert(anode->name);
			}

			if (p_default_actions.renames.has(anode->name)) {
				code = p_default_actions.renames[anode->name];
			} else if (use_fragment_varying) {
				code = "frag_to_light." + _mkid(anode->name);
			} else {
				code = _mkid(anode->name);
			}

			if (anode->call_expression != nullptr) {
				code += ".";
				code += _dump_node_code(anode->call_expression, p_level, r_gen_code, p_actions, p_default_actions, p_assigning, false);
			} else if (anode->index_expression != nullptr) {
				code += "[";
				code += _dump_node_code(anode->index_expression, p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				code += "]";
			} else if (anode->assign_expression != nullptr) {
				code += "=";
				code += _dump_node_code(anode->assign_expression, p_level, r_gen_code, p_actions, p_default_actions, true, false);
			}

			if (anode->name == time_name) {
				if (current_func_name == vertex_name) {
					r_gen_code.uses_vertex_time = true;
				}
				if (current_func_name == fragment_name || current_func_name == light_name) {
					r_gen_code.uses_fragment_time = true;
				}
			}

		} break;
		case SL::Node::TYPE_CONSTANT: {
			SL::ConstantNode *cnode = (SL::ConstantNode *)p_node;

			if (cnode->array_size == 0) {
				return get_constant_text(cnode->datatype, cnode->values);
			} else {
				if (cnode->get_datatype() == SL::TYPE_STRUCT) {
					code += _mkid(cnode->struct_name);
				} else {
					code += _typestr(cnode->datatype);
				}
				code += "[";
				code += itos(cnode->array_size);
				code += "]";
				code += "(";
				for (int i = 0; i < cnode->array_size; i++) {
					if (i > 0) {
						code += ",";
					} else {
						code += "";
					}
					code += _dump_node_code(cnode->array_declarations[0].initializer[i], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				}
				code += ")";
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
				case SL::OP_ASSIGN_SHIFT_LEFT:
				case SL::OP_ASSIGN_SHIFT_RIGHT:
				case SL::OP_ASSIGN_MOD:
				case SL::OP_ASSIGN_BIT_AND:
				case SL::OP_ASSIGN_BIT_OR:
				case SL::OP_ASSIGN_BIT_XOR:
					code = _dump_node_code(onode->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, true) + _opstr(onode->op) + _dump_node_code(onode->arguments[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					break;
				case SL::OP_BIT_INVERT:
				case SL::OP_NEGATE:
				case SL::OP_NOT:
				case SL::OP_DECREMENT:
				case SL::OP_INCREMENT:
					code = _opstr(onode->op) + _dump_node_code(onode->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					break;
				case SL::OP_POST_DECREMENT:
				case SL::OP_POST_INCREMENT:
					code = _dump_node_code(onode->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning) + _opstr(onode->op);
					break;
				case SL::OP_CALL:
				case SL::OP_STRUCT:
				case SL::OP_CONSTRUCT: {
					ERR_FAIL_COND_V(onode->arguments[0]->type != SL::Node::TYPE_VARIABLE, String());

					SL::VariableNode *vnode = (SL::VariableNode *)onode->arguments[0];

					if (onode->op == SL::OP_STRUCT) {
						code += _mkid(vnode->name);
					} else if (onode->op == SL::OP_CONSTRUCT) {
						code += String(vnode->name);
					} else {
						if (internal_functions.has(vnode->name)) {
							code += vnode->name;
						} else if (p_default_actions.renames.has(vnode->name)) {
							code += p_default_actions.renames[vnode->name];
						} else {
							code += _mkid(vnode->name);
						}
					}

					code += "(";

					for (int i = 1; i < onode->arguments.size(); i++) {
						if (i > 1) {
							code += ", ";
						}
						code += _dump_node_code(onode->arguments[i], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					}
					code += ")";
				} break;
				case SL::OP_INDEX: {
					code += _dump_node_code(onode->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += "[";
					code += _dump_node_code(onode->arguments[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += "]";

				} break;
				case SL::OP_SELECT_IF: {
					code += "(";
					code += _dump_node_code(onode->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += "?";
					code += _dump_node_code(onode->arguments[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += ":";
					code += _dump_node_code(onode->arguments[2], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					code += ")";

				} break;

				default: {
					if (p_use_scope) {
						code += "(";
					}
					code += _dump_node_code(onode->arguments[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning) + _opstr(onode->op) + _dump_node_code(onode->arguments[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
					if (p_use_scope) {
						code += ")";
					}
					break;
				}
			}

		} break;
		case SL::Node::TYPE_CONTROL_FLOW: {
			SL::ControlFlowNode *cfnode = (SL::ControlFlowNode *)p_node;
			if (cfnode->flow_op == SL::FLOW_OP_IF) {
				code += _mktab(p_level) + "if (" + _dump_node_code(cfnode->expressions[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning) + ")\n";
				code += _dump_node_code(cfnode->blocks[0], p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);
				if (cfnode->blocks.size() == 2) {
					code += _mktab(p_level) + "else\n";
					code += _dump_node_code(cfnode->blocks[1], p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);
				}
			} else if (cfnode->flow_op == SL::FLOW_OP_SWITCH) {
				code += _mktab(p_level) + "switch (" + _dump_node_code(cfnode->expressions[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning) + ")\n";
				code += _dump_node_code(cfnode->blocks[0], p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);
			} else if (cfnode->flow_op == SL::FLOW_OP_CASE) {
				code += _mktab(p_level) + "case " + _dump_node_code(cfnode->expressions[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning) + ":\n";
				code += _dump_node_code(cfnode->blocks[0], p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);
			} else if (cfnode->flow_op == SL::FLOW_OP_DEFAULT) {
				code += _mktab(p_level) + "default:\n";
				code += _dump_node_code(cfnode->blocks[0], p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);
			} else if (cfnode->flow_op == SL::FLOW_OP_DO) {
				code += _mktab(p_level) + "do";
				code += _dump_node_code(cfnode->blocks[0], p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);
				code += _mktab(p_level) + "while (" + _dump_node_code(cfnode->expressions[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning) + ");";

			} else if (cfnode->flow_op == SL::FLOW_OP_WHILE) {
				code += _mktab(p_level) + "while (" + _dump_node_code(cfnode->expressions[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning) + ")\n";
				code += _dump_node_code(cfnode->blocks[0], p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);
			} else if (cfnode->flow_op == SL::FLOW_OP_FOR) {
				String left = _dump_node_code(cfnode->blocks[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				String middle = _dump_node_code(cfnode->expressions[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				String right = _dump_node_code(cfnode->expressions[1], p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				code += _mktab(p_level) + "for (" + left + ";" + middle + ";" + right + ")\n";
				code += _dump_node_code(cfnode->blocks[1], p_level + 1, r_gen_code, p_actions, p_default_actions, p_assigning);

			} else if (cfnode->flow_op == SL::FLOW_OP_RETURN) {
				if (cfnode->expressions.size()) {
					code = "return " + _dump_node_code(cfnode->expressions[0], p_level, r_gen_code, p_actions, p_default_actions, p_assigning) + ";";
				} else {
					code = "return;";
				}
			} else if (cfnode->flow_op == SL::FLOW_OP_DISCARD) {
				if (p_actions.usage_flag_pointers.has("DISCARD") && !used_flag_pointers.has("DISCARD")) {
					*p_actions.usage_flag_pointers["DISCARD"] = true;
					used_flag_pointers.insert("DISCARD");
				}

				code = "discard;";
			} else if (cfnode->flow_op == SL::FLOW_OP_CONTINUE) {
				code = "continue;";
			} else if (cfnode->flow_op == SL::FLOW_OP_BREAK) {
				code = "break;";
			}

		} break;
		case SL::Node::TYPE_MEMBER: {
			SL::MemberNode *mnode = (SL::MemberNode *)p_node;
			code = _dump_node_code(mnode->owner, p_level, r_gen_code, p_actions, p_default_actions, p_assigning) + "." + mnode->name;
			if (mnode->index_expression != nullptr) {
				code += "[";
				code += _dump_node_code(mnode->index_expression, p_level, r_gen_code, p_actions, p_default_actions, p_assigning);
				code += "]";
			} else if (mnode->assign_expression != nullptr) {
				code += "=";
				code += _dump_node_code(mnode->assign_expression, p_level, r_gen_code, p_actions, p_default_actions, true, false);
			}
		} break;
	}

	return code;
}

Error ShaderCompilerGLES3::compile(VS::ShaderMode p_mode, const String &p_code, IdentifierActions *p_actions, const String &p_path, GeneratedCode &r_gen_code) {
	Error err = parser.compile(p_code, ShaderTypes::get_singleton()->get_functions(p_mode), ShaderTypes::get_singleton()->get_modes(p_mode), ShaderTypes::get_singleton()->get_types());

	if (err != OK) {
		Vector<String> shader = p_code.split("\n");
		for (int i = 0; i < shader.size(); i++) {
			if (i + 1 == parser.get_error_line()) {
				// Mark the error line to be visible without having to look at
				// the trace at the end.
				print_line(vformat("E%4d-> %s", i + 1, shader[i]));
			} else {
				print_line(vformat("%5d | %s", i + 1, shader[i]));
			}
		}

		_err_print_error(nullptr, p_path.utf8().get_data(), parser.get_error_line(), parser.get_error_text().utf8().get_data(), ERR_HANDLER_SHADER);
		return err;
	}

	r_gen_code.defines.clear();
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
	fragment_varyings.clear();

	shader = parser.get_shader();
	function = nullptr;
	_dump_node_code(shader, 1, r_gen_code, *p_actions, actions[p_mode], false);

	if (r_gen_code.uniform_total_size) { //uniforms used?
		int md = sizeof(float) * 4;
		if (r_gen_code.uniform_total_size % md) {
			r_gen_code.uniform_total_size += md - (r_gen_code.uniform_total_size % md);
		}
		r_gen_code.uniform_total_size += md; //pad just in case
	}

	return OK;
}

ShaderCompilerGLES3::ShaderCompilerGLES3() {
	/** CANVAS ITEM SHADER **/

	actions[VS::SHADER_CANVAS_ITEM].renames["VERTEX"] = "outvec.xy";
	actions[VS::SHADER_CANVAS_ITEM].renames["UV"] = "uv";
	actions[VS::SHADER_CANVAS_ITEM].renames["POINT_SIZE"] = "point_size";

	actions[VS::SHADER_CANVAS_ITEM].renames["WORLD_MATRIX"] = "modelview_matrix";
	actions[VS::SHADER_CANVAS_ITEM].renames["PROJECTION_MATRIX"] = "projection_matrix";
	actions[VS::SHADER_CANVAS_ITEM].renames["EXTRA_MATRIX"] = "extra_matrix";
	actions[VS::SHADER_CANVAS_ITEM].renames["TIME"] = "time";
	actions[VS::SHADER_CANVAS_ITEM].renames["AT_LIGHT_PASS"] = "at_light_pass";
	actions[VS::SHADER_CANVAS_ITEM].renames["INSTANCE_CUSTOM"] = "instance_custom";

	actions[VS::SHADER_CANVAS_ITEM].renames["COLOR"] = "color";
	actions[VS::SHADER_CANVAS_ITEM].renames["MODULATE"] = "final_modulate_alias";
	actions[VS::SHADER_CANVAS_ITEM].renames["NORMAL"] = "normal";
	actions[VS::SHADER_CANVAS_ITEM].renames["NORMALMAP"] = "normal_map";
	actions[VS::SHADER_CANVAS_ITEM].renames["NORMALMAP_DEPTH"] = "normal_depth";
	actions[VS::SHADER_CANVAS_ITEM].renames["TEXTURE"] = "color_texture";
	actions[VS::SHADER_CANVAS_ITEM].renames["TEXTURE_PIXEL_SIZE"] = "color_texpixel_size";
	actions[VS::SHADER_CANVAS_ITEM].renames["NORMAL_TEXTURE"] = "normal_texture";
	actions[VS::SHADER_CANVAS_ITEM].renames["SCREEN_UV"] = "screen_uv";
	actions[VS::SHADER_CANVAS_ITEM].renames["SCREEN_TEXTURE"] = "screen_texture";
	actions[VS::SHADER_CANVAS_ITEM].renames["SCREEN_PIXEL_SIZE"] = "screen_pixel_size";
	actions[VS::SHADER_CANVAS_ITEM].renames["FRAGCOORD"] = "gl_FragCoord";
	actions[VS::SHADER_CANVAS_ITEM].renames["POINT_COORD"] = "gl_PointCoord";
	actions[VS::SHADER_CANVAS_ITEM].renames["INSTANCE_ID"] = "gl_InstanceID";
	actions[VS::SHADER_CANVAS_ITEM].renames["VERTEX_ID"] = "gl_VertexID";

	actions[VS::SHADER_CANVAS_ITEM].renames["LIGHT_VEC"] = "light_vec";
	actions[VS::SHADER_CANVAS_ITEM].renames["LIGHT_HEIGHT"] = "light_height";
	actions[VS::SHADER_CANVAS_ITEM].renames["LIGHT_COLOR"] = "light_color";
	actions[VS::SHADER_CANVAS_ITEM].renames["LIGHT_UV"] = "light_uv";
	actions[VS::SHADER_CANVAS_ITEM].renames["LIGHT"] = "light";
	actions[VS::SHADER_CANVAS_ITEM].renames["SHADOW_COLOR"] = "shadow_color";
	actions[VS::SHADER_CANVAS_ITEM].renames["SHADOW_VEC"] = "shadow_vec";

	actions[VS::SHADER_CANVAS_ITEM].usage_defines["COLOR"] = "#define COLOR_USED\n";
	actions[VS::SHADER_CANVAS_ITEM].usage_defines["MODULATE"] = "#define MODULATE_USED\n";
	actions[VS::SHADER_CANVAS_ITEM].usage_defines["SCREEN_TEXTURE"] = "#define SCREEN_TEXTURE_USED\n";
	actions[VS::SHADER_CANVAS_ITEM].usage_defines["SCREEN_UV"] = "#define SCREEN_UV_USED\n";
	actions[VS::SHADER_CANVAS_ITEM].usage_defines["SCREEN_PIXEL_SIZE"] = "@SCREEN_UV";
	actions[VS::SHADER_CANVAS_ITEM].usage_defines["NORMAL"] = "#define NORMAL_USED\n";
	actions[VS::SHADER_CANVAS_ITEM].usage_defines["NORMALMAP"] = "#define NORMALMAP_USED\n";
	actions[VS::SHADER_CANVAS_ITEM].usage_defines["LIGHT"] = "#define USE_LIGHT_SHADER_CODE\n";
	actions[VS::SHADER_CANVAS_ITEM].usage_defines["SHADOW_VEC"] = "#define SHADOW_VEC_USED\n";
	actions[VS::SHADER_CANVAS_ITEM].render_mode_defines["skip_vertex_transform"] = "#define SKIP_TRANSFORM_USED\n";

	/** SPATIAL SHADER **/

	actions[VS::SHADER_SPATIAL].renames["WORLD_MATRIX"] = "world_transform";
	actions[VS::SHADER_SPATIAL].renames["INV_CAMERA_MATRIX"] = "camera_inverse_matrix";
	actions[VS::SHADER_SPATIAL].renames["CAMERA_MATRIX"] = "camera_matrix";
	actions[VS::SHADER_SPATIAL].renames["PROJECTION_MATRIX"] = "projection_matrix";
	actions[VS::SHADER_SPATIAL].renames["INV_PROJECTION_MATRIX"] = "inv_projection_matrix";
	actions[VS::SHADER_SPATIAL].renames["MODELVIEW_MATRIX"] = "modelview";

	actions[VS::SHADER_SPATIAL].renames["VIEW_INDEX"] = "view_index";
	actions[VS::SHADER_SPATIAL].renames["VIEW_MONO_LEFT"] = "0";
	actions[VS::SHADER_SPATIAL].renames["VIEW_RIGHT"] = "1";

	actions[VS::SHADER_SPATIAL].renames["VERTEX"] = "vertex.xyz";
	actions[VS::SHADER_SPATIAL].renames["NORMAL"] = "normal";
	actions[VS::SHADER_SPATIAL].renames["TANGENT"] = "tangent";
	actions[VS::SHADER_SPATIAL].renames["BINORMAL"] = "binormal";
	actions[VS::SHADER_SPATIAL].renames["POSITION"] = "position";
	actions[VS::SHADER_SPATIAL].renames["UV"] = "uv_interp";
	actions[VS::SHADER_SPATIAL].renames["UV2"] = "uv2_interp";
	actions[VS::SHADER_SPATIAL].renames["COLOR"] = "color_interp";
	actions[VS::SHADER_SPATIAL].renames["POINT_SIZE"] = "point_size";
	actions[VS::SHADER_SPATIAL].renames["INSTANCE_ID"] = "gl_InstanceID";
	actions[VS::SHADER_SPATIAL].renames["VERTEX_ID"] = "gl_VertexID";

	//builtins

	actions[VS::SHADER_SPATIAL].renames["TIME"] = "time";
	actions[VS::SHADER_SPATIAL].renames["VIEWPORT_SIZE"] = "viewport_size";

	actions[VS::SHADER_SPATIAL].renames["FRAGCOORD"] = "gl_FragCoord";
	actions[VS::SHADER_SPATIAL].renames["FRONT_FACING"] = "gl_FrontFacing";
	actions[VS::SHADER_SPATIAL].renames["NORMALMAP"] = "normalmap";
	actions[VS::SHADER_SPATIAL].renames["NORMALMAP_DEPTH"] = "normaldepth";
	actions[VS::SHADER_SPATIAL].renames["ALBEDO"] = "albedo";
	actions[VS::SHADER_SPATIAL].renames["ALPHA"] = "alpha";
	actions[VS::SHADER_SPATIAL].renames["METALLIC"] = "metallic";
	actions[VS::SHADER_SPATIAL].renames["SPECULAR"] = "specular";
	actions[VS::SHADER_SPATIAL].renames["ROUGHNESS"] = "roughness";
	actions[VS::SHADER_SPATIAL].renames["RIM"] = "rim";
	actions[VS::SHADER_SPATIAL].renames["RIM_TINT"] = "rim_tint";
	actions[VS::SHADER_SPATIAL].renames["CLEARCOAT"] = "clearcoat";
	actions[VS::SHADER_SPATIAL].renames["CLEARCOAT_GLOSS"] = "clearcoat_gloss";
	actions[VS::SHADER_SPATIAL].renames["ANISOTROPY"] = "anisotropy";
	actions[VS::SHADER_SPATIAL].renames["ANISOTROPY_FLOW"] = "anisotropy_flow";
	actions[VS::SHADER_SPATIAL].renames["SSS_STRENGTH"] = "sss_strength";
	actions[VS::SHADER_SPATIAL].renames["TRANSMISSION"] = "transmission";
	actions[VS::SHADER_SPATIAL].renames["AO"] = "ao";
	actions[VS::SHADER_SPATIAL].renames["AO_LIGHT_AFFECT"] = "ao_light_affect";
	actions[VS::SHADER_SPATIAL].renames["EMISSION"] = "emission";
	actions[VS::SHADER_SPATIAL].renames["POINT_COORD"] = "gl_PointCoord";
	actions[VS::SHADER_SPATIAL].renames["INSTANCE_CUSTOM"] = "instance_custom";
	actions[VS::SHADER_SPATIAL].renames["SCREEN_UV"] = "screen_uv";
	actions[VS::SHADER_SPATIAL].renames["SCREEN_TEXTURE"] = "screen_texture";
	actions[VS::SHADER_SPATIAL].renames["DEPTH_TEXTURE"] = "depth_buffer";
	actions[VS::SHADER_SPATIAL].renames["DEPTH"] = "gl_FragDepth";
	actions[VS::SHADER_SPATIAL].renames["ALPHA_SCISSOR"] = "alpha_scissor";
	actions[VS::SHADER_SPATIAL].renames["OUTPUT_IS_SRGB"] = "SHADER_IS_SRGB";
	actions[VS::SHADER_SPATIAL].renames["NODE_POSITION_WORLD"] = "world_transform[3].xyz";
	actions[VS::SHADER_SPATIAL].renames["CAMERA_POSITION_WORLD"] = "camera_matrix[3].xyz";
	actions[VS::SHADER_SPATIAL].renames["CAMERA_DIRECTION_WORLD"] = "camera_inverse_matrix[3].xyz";
	actions[VS::SHADER_SPATIAL].renames["NODE_POSITION_VIEW"] = "(camera_inverse_matrix * world_transform)[3].xyz";

	//for light
	actions[VS::SHADER_SPATIAL].renames["VIEW"] = "view";
	actions[VS::SHADER_SPATIAL].renames["LIGHT_COLOR"] = "light_color";
	actions[VS::SHADER_SPATIAL].renames["LIGHT"] = "light";
	actions[VS::SHADER_SPATIAL].renames["ATTENUATION"] = "attenuation";
	actions[VS::SHADER_SPATIAL].renames["DIFFUSE_LIGHT"] = "diffuse_light";
	actions[VS::SHADER_SPATIAL].renames["SPECULAR_LIGHT"] = "specular_light";

	actions[VS::SHADER_SPATIAL].usage_defines["TANGENT"] = "#define ENABLE_TANGENT_INTERP\n";
	actions[VS::SHADER_SPATIAL].usage_defines["BINORMAL"] = "@TANGENT";
	actions[VS::SHADER_SPATIAL].usage_defines["RIM"] = "#define LIGHT_USE_RIM\n";
	actions[VS::SHADER_SPATIAL].usage_defines["RIM_TINT"] = "@RIM";
	actions[VS::SHADER_SPATIAL].usage_defines["CLEARCOAT"] = "#define LIGHT_USE_CLEARCOAT\n";
	actions[VS::SHADER_SPATIAL].usage_defines["CLEARCOAT_GLOSS"] = "@CLEARCOAT";
	actions[VS::SHADER_SPATIAL].usage_defines["ANISOTROPY"] = "#define LIGHT_USE_ANISOTROPY\n";
	actions[VS::SHADER_SPATIAL].usage_defines["ANISOTROPY_FLOW"] = "@ANISOTROPY";
	actions[VS::SHADER_SPATIAL].usage_defines["AO"] = "#define ENABLE_AO\n";
	actions[VS::SHADER_SPATIAL].usage_defines["AO_LIGHT_AFFECT"] = "#define ENABLE_AO\n";
	actions[VS::SHADER_SPATIAL].usage_defines["UV"] = "#define ENABLE_UV_INTERP\n";
	actions[VS::SHADER_SPATIAL].usage_defines["UV2"] = "#define ENABLE_UV2_INTERP\n";
	actions[VS::SHADER_SPATIAL].usage_defines["NORMALMAP"] = "#define ENABLE_NORMALMAP\n";
	actions[VS::SHADER_SPATIAL].usage_defines["NORMALMAP_DEPTH"] = "@NORMALMAP";
	actions[VS::SHADER_SPATIAL].usage_defines["COLOR"] = "#define ENABLE_COLOR_INTERP\n";
	actions[VS::SHADER_SPATIAL].usage_defines["INSTANCE_CUSTOM"] = "#define ENABLE_INSTANCE_CUSTOM\n";
	actions[VS::SHADER_SPATIAL].usage_defines["ALPHA_SCISSOR"] = "#define ALPHA_SCISSOR_USED\n";
	actions[VS::SHADER_SPATIAL].usage_defines["POSITION"] = "#define OVERRIDE_POSITION\n";

	actions[VS::SHADER_SPATIAL].usage_defines["SSS_STRENGTH"] = "#define ENABLE_SSS\n";
	actions[VS::SHADER_SPATIAL].usage_defines["TRANSMISSION"] = "#define TRANSMISSION_USED\n";
	actions[VS::SHADER_SPATIAL].usage_defines["SCREEN_TEXTURE"] = "#define SCREEN_TEXTURE_USED\n";
	actions[VS::SHADER_SPATIAL].usage_defines["SCREEN_UV"] = "#define SCREEN_UV_USED\n";

	actions[VS::SHADER_SPATIAL].usage_defines["DIFFUSE_LIGHT"] = "#define USE_LIGHT_SHADER_CODE\n";
	actions[VS::SHADER_SPATIAL].usage_defines["SPECULAR_LIGHT"] = "#define USE_LIGHT_SHADER_CODE\n";

	actions[VS::SHADER_SPATIAL].render_mode_defines["skip_vertex_transform"] = "#define SKIP_TRANSFORM_USED\n";
	actions[VS::SHADER_SPATIAL].render_mode_defines["world_vertex_coords"] = "#define VERTEX_WORLD_COORDS_USED\n";
	actions[VS::SHADER_SPATIAL].render_mode_defines["ensure_correct_normals"] = "#define ENSURE_CORRECT_NORMALS\n";
	actions[VS::SHADER_SPATIAL].render_mode_defines["cull_front"] = "#define DO_SIDE_CHECK\n";
	actions[VS::SHADER_SPATIAL].render_mode_defines["cull_disabled"] = "#define DO_SIDE_CHECK\n";

	bool force_lambert = GLOBAL_GET("rendering/quality/shading/force_lambert_over_burley");

	if (!force_lambert) {
		actions[VS::SHADER_SPATIAL].render_mode_defines["diffuse_burley"] = "#define DIFFUSE_BURLEY\n";
	}

	actions[VS::SHADER_SPATIAL].render_mode_defines["diffuse_oren_nayar"] = "#define DIFFUSE_OREN_NAYAR\n";
	actions[VS::SHADER_SPATIAL].render_mode_defines["diffuse_lambert_wrap"] = "#define DIFFUSE_LAMBERT_WRAP\n";
	actions[VS::SHADER_SPATIAL].render_mode_defines["diffuse_toon"] = "#define DIFFUSE_TOON\n";

	bool force_blinn = GLOBAL_GET("rendering/quality/shading/force_blinn_over_ggx");

	if (!force_blinn) {
		actions[VS::SHADER_SPATIAL].render_mode_defines["specular_schlick_ggx"] = "#define SPECULAR_SCHLICK_GGX\n";
	} else {
		actions[VS::SHADER_SPATIAL].render_mode_defines["specular_schlick_ggx"] = "#define SPECULAR_BLINN\n";
	}

	actions[VS::SHADER_SPATIAL].render_mode_defines["specular_blinn"] = "#define SPECULAR_BLINN\n";
	actions[VS::SHADER_SPATIAL].render_mode_defines["specular_phong"] = "#define SPECULAR_PHONG\n";
	actions[VS::SHADER_SPATIAL].render_mode_defines["specular_toon"] = "#define SPECULAR_TOON\n";
	actions[VS::SHADER_SPATIAL].render_mode_defines["specular_disabled"] = "#define SPECULAR_DISABLED\n";
	actions[VS::SHADER_SPATIAL].render_mode_defines["shadows_disabled"] = "#define SHADOWS_DISABLED\n";
	actions[VS::SHADER_SPATIAL].render_mode_defines["ambient_light_disabled"] = "#define AMBIENT_LIGHT_DISABLED\n";
	actions[VS::SHADER_SPATIAL].render_mode_defines["shadow_to_opacity"] = "#define USE_SHADOW_TO_OPACITY\n";

	/* PARTICLES SHADER */

	actions[VS::SHADER_PARTICLES].renames["COLOR"] = "out_color";
	actions[VS::SHADER_PARTICLES].renames["VELOCITY"] = "out_velocity_active.xyz";
	actions[VS::SHADER_PARTICLES].renames["MASS"] = "mass";
	actions[VS::SHADER_PARTICLES].renames["ACTIVE"] = "shader_active";
	actions[VS::SHADER_PARTICLES].renames["RESTART"] = "restart";
	actions[VS::SHADER_PARTICLES].renames["CUSTOM"] = "out_custom";
	actions[VS::SHADER_PARTICLES].renames["TRANSFORM"] = "xform";
	actions[VS::SHADER_PARTICLES].renames["TIME"] = "time";
	actions[VS::SHADER_PARTICLES].renames["LIFETIME"] = "lifetime";
	actions[VS::SHADER_PARTICLES].renames["DELTA"] = "local_delta";
	actions[VS::SHADER_PARTICLES].renames["NUMBER"] = "particle_number";
	actions[VS::SHADER_PARTICLES].renames["INDEX"] = "index";
	actions[VS::SHADER_PARTICLES].renames["GRAVITY"] = "current_gravity";
	actions[VS::SHADER_PARTICLES].renames["EMISSION_TRANSFORM"] = "emission_transform";
	actions[VS::SHADER_PARTICLES].renames["RANDOM_SEED"] = "random_seed";

	actions[VS::SHADER_PARTICLES].render_mode_defines["disable_force"] = "#define DISABLE_FORCE\n";
	actions[VS::SHADER_PARTICLES].render_mode_defines["disable_velocity"] = "#define DISABLE_VELOCITY\n";
	actions[VS::SHADER_PARTICLES].render_mode_defines["keep_data"] = "#define ENABLE_KEEP_DATA\n";

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

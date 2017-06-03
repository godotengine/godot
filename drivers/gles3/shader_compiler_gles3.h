/*************************************************************************/
/*  shader_compiler_gles3.h                                              */
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
#ifndef SHADERCOMPILERGLES3_H
#define SHADERCOMPILERGLES3_H

#include "pair.h"
#include "servers/visual/shader_language.h"
#include "servers/visual/shader_types.h"
#include "servers/visual_server.h"

class ShaderCompilerGLES3 {
public:
	struct IdentifierActions {

		Map<StringName, Pair<int *, int> > render_mode_values;
		Map<StringName, bool *> render_mode_flags;
		Map<StringName, bool *> usage_flag_pointers;
		Map<StringName, bool *> write_flag_pointers;

		Map<StringName, ShaderLanguage::ShaderNode::Uniform> *uniforms;
	};

	struct GeneratedCode {

		Vector<CharString> defines;
		Vector<StringName> texture_uniforms;
		Vector<ShaderLanguage::ShaderNode::Uniform::Hint> texture_hints;

		Vector<uint32_t> uniform_offsets;
		uint32_t uniform_total_size;
		String uniforms;
		String vertex_global;
		String vertex;
		String fragment_global;
		String fragment;
		String light;

		bool uses_fragment_time;
		bool uses_vertex_time;
	};

private:
	ShaderLanguage parser;

	struct DefaultIdentifierActions {

		Map<StringName, String> renames;
		Map<StringName, String> render_mode_defines;
		Map<StringName, String> usage_defines;
	};

	void _dump_function_deps(ShaderLanguage::ShaderNode *p_node, const StringName &p_for_func, const Map<StringName, String> &p_func_code, String &r_to_add, Set<StringName> &added);
	String _dump_node_code(ShaderLanguage::Node *p_node, int p_level, GeneratedCode &r_gen_code, IdentifierActions &p_actions, const DefaultIdentifierActions &p_default_actions);

	StringName current_func_name;
	StringName vertex_name;
	StringName fragment_name;
	StringName time_name;

	Set<StringName> used_name_defines;
	Set<StringName> used_flag_pointers;
	Set<StringName> used_rmode_defines;
	Set<StringName> internal_functions;

	DefaultIdentifierActions actions[VS::SHADER_MAX];

public:
	Error compile(VS::ShaderMode p_mode, const String &p_code, IdentifierActions *p_actions, const String &p_path, GeneratedCode &r_gen_code);

	ShaderCompilerGLES3();
};

#endif // SHADERCOMPILERGLES3_H

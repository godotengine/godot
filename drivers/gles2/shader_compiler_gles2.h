/*************************************************************************/
/*  shader_compiler_gles2.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#pragma once

#include "drivers/gles_common/rasterizer_platforms.h"
#ifdef GLES2_BACKEND_ENABLED

#include "drivers/gles_common/rasterizer_version.h"
#ifdef GODOT_3
#include "core/pair.h"
#include "core/string_builder.h"
#include "servers/visual/shader_language.h"
#include "servers/visual/shader_types.h"
#include "servers/visual_server.h"
#else
#include "core/string/string_builder.h"
#include "core/templates/pair.h"
#include "servers/rendering/shader_language.h"
#include "servers/rendering/shader_types.h"
#include "servers/rendering_server.h"
#endif

class ShaderCompilerGLES2 {
public:
	struct IdentifierActions {
		Map<StringName, Pair<int *, int>> render_mode_values;
		Map<StringName, bool *> render_mode_flags;
		Map<StringName, bool *> usage_flag_pointers;
		Map<StringName, bool *> write_flag_pointers;

		Map<StringName, ShaderLanguage::ShaderNode::Uniform> *uniforms;
	};

	struct GeneratedCode {
		Vector<CharString> custom_defines;
		Vector<StringName> uniforms;
		Vector<StringName> texture_uniforms;
		Vector<ShaderLanguage::ShaderNode::Uniform::Hint> texture_hints;

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

	void _dump_function_deps(ShaderLanguage::ShaderNode *p_node, const StringName &p_for_func, const Map<StringName, String> &p_func_code, StringBuilder &r_to_add, Set<StringName> &r_added);
	String _dump_node_code(ShaderLanguage::Node *p_node, int p_level, GeneratedCode &r_gen_code, IdentifierActions &p_actions, const DefaultIdentifierActions &p_default_actions, bool p_assigning, bool p_use_scope = true);

	StringName current_func_name;
	StringName vertex_name;
	StringName fragment_name;
	StringName light_name;
	StringName time_name;

	Set<StringName> used_name_defines;
	Set<StringName> used_flag_pointers;
	Set<StringName> used_rmode_defines;
	Set<StringName> internal_functions;

	DefaultIdentifierActions actions[GD_VS::SHADER_MAX];

	// compatibility with godot 4
	static ShaderLanguage::DataType _get_variable_type(const StringName &p_type);

public:
	Error compile(GD_VS::ShaderMode p_mode, const String &p_code, IdentifierActions *p_actions, const String &p_path, GeneratedCode &r_gen_code);

	ShaderCompilerGLES2();
};

#endif // GLES2_BACKEND_ENABLED

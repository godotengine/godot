/*************************************************************************/
/*  shader_compiler_gles2.h                                              */
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
#ifndef SHADER_COMPILER_GLES2_H
#define SHADER_COMPILER_GLES2_H

#include "servers/visual/shader_language.h"
class ShaderCompilerGLES2 {

	class Uniform;

public:
	struct Flags;

private:
	ShaderLanguage::ProgramNode *program_node;
	String dump_node_code(ShaderLanguage::Node *p_node, int p_level, bool p_assign_left = false);
	Error compile_node(ShaderLanguage::ProgramNode *p_program);
	static Error create_glsl_120_code(void *p_str, ShaderLanguage::ProgramNode *p_program);

	bool uses_light;
	bool uses_texscreen;
	bool uses_texpos;
	bool uses_alpha;
	bool uses_discard;
	bool uses_time;
	bool uses_screen_uv;
	bool uses_normalmap;
	bool uses_normal;
	bool uses_texpixel_size;
	bool uses_worldvec;
	bool vertex_code_writes_vertex;
	bool vertex_code_writes_position;
	bool uses_shadow_color;

	bool sinh_used;
	bool tanh_used;
	bool cosh_used;

	bool custom_h;

	Flags *flags;

	StringName vname_discard;
	StringName vname_screen_uv;
	StringName vname_diffuse_alpha;
	StringName vname_color_interp;
	StringName vname_uv_interp;
	StringName vname_uv2_interp;
	StringName vname_tangent_interp;
	StringName vname_binormal_interp;
	StringName vname_var1_interp;
	StringName vname_var2_interp;
	StringName vname_vertex;
	StringName vname_position;
	StringName vname_light;
	StringName vname_time;
	StringName vname_normalmap;
	StringName vname_normalmap_depth;
	StringName vname_normal;
	StringName vname_texpixel_size;
	StringName vname_world_vec;
	StringName vname_shadow;

	Map<StringName, ShaderLanguage::Uniform> *uniforms;

	StringName out_vertex_name;

	String global_code;
	String code;
	ShaderLanguage::ShaderType type;

	String replace_string(const StringName &p_string);

	Map<StringName, StringName> mode_replace_table[9];
	Map<StringName, StringName> replace_table;

public:
	struct Flags {

		bool uses_alpha;
		bool uses_texscreen;
		bool uses_texpos;
		bool uses_normalmap;
		bool vertex_code_writes_vertex;
		bool vertex_code_writes_position;
		bool uses_discard;
		bool uses_screen_uv;
		bool use_color_interp;
		bool use_uv_interp;
		bool use_uv2_interp;
		bool use_tangent_interp;
		bool use_var1_interp;
		bool use_var2_interp;
		bool uses_light;
		bool uses_time;
		bool uses_normal;
		bool uses_texpixel_size;
		bool uses_worldvec;
		bool uses_shadow_color;
	};

	Error compile(const String &p_code, ShaderLanguage::ShaderType p_type, String &r_code_line, String &r_globals_line, Flags &r_flags, Map<StringName, ShaderLanguage::Uniform> *r_uniforms = NULL);

	ShaderCompilerGLES2();
};

#endif // SHADER_COMPILERL_GL_H

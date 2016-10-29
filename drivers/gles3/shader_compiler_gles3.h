#ifndef SHADERCOMPILERGLES3_H
#define SHADERCOMPILERGLES3_H

#include "servers/visual/shader_language.h"
#include "servers/visual/shader_types.h"
#include "servers/visual_server.h"
#include "pair.h"

class ShaderCompilerGLES3 {
public:
	struct IdentifierActions {

		Map<StringName,Pair<int*,int> > render_mode_values;
		Map<StringName,bool*> render_mode_flags;
		Map<StringName,bool*> usage_flag_pointers;

		Map<StringName,ShaderLanguage::ShaderNode::Uniform> *uniforms;
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

	};

private:

	ShaderLanguage parser;

	struct DefaultIdentifierActions {

		Map<StringName,String> renames;
		Map<StringName,String> render_mode_defines;
		Map<StringName,String> usage_defines;
	};

	void _dump_function_deps(ShaderLanguage::ShaderNode *p_node, const StringName& p_for_func, const Map<StringName, String> &p_func_code, String& r_to_add,Set<StringName> &added);
	String _dump_node_code(ShaderLanguage::Node *p_node, int p_level, GeneratedCode &r_gen_code, IdentifierActions& p_actions, const DefaultIdentifierActions& p_default_actions);



	Set<StringName> used_name_defines;
	Set<StringName> used_rmode_defines;
	Set<StringName> internal_functions;


	DefaultIdentifierActions actions[VS::SHADER_MAX];

public:


	Error compile(VS::ShaderMode p_mode, const String& p_code, IdentifierActions* p_actions, const String& p_path, GeneratedCode& r_gen_code);


	ShaderCompilerGLES3();
};

#endif // SHADERCOMPILERGLES3_H

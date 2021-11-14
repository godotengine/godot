/*************************************************************************/
/*  shader_preprocessor.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef SHADER_PREPROCESSOR_H
#define SHADER_PREPROCESSOR_H

#include "core/templates/list.h"
#include "core/templates/map.h"
#include "core/templates/set.h"
#include "core/typedefs.h"
#include "core/string/ustring.h"

#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "scene/resources/shader.h"

class PreproprocessorTokenizer;
struct PPToken;

typedef char CharType;

struct PreprocessorDefine {
	Vector<String> arguments;
	String body;
};

struct SkippedPreprocessorCondition
{
	int start_line = -1;
	int end_line = -1;
};

struct PreprocessorState {
	Map<String, PreprocessorDefine *> defines;
	Vector<bool> skip_stack_else;
	int condition_depth;
	Set<String> includes;
	int include_depth;
	String current_include = "";
	String error;
	int error_line;
	Map<String, Vector<SkippedPreprocessorCondition*>> skipped_conditions;

	~PreprocessorState()
	{
		for (auto& kvp : skipped_conditions)
		{
			for (auto& skip_proc : kvp.value)
			{
				delete skip_proc;
			}

			kvp.value.clear();
		}

		skipped_conditions.clear();
	}
};

class ShaderPreprocessor {

public:
	~ShaderPreprocessor();
	ShaderPreprocessor(const String &p_code);

	String preprocess(PreprocessorState *p_state);
	String preprocess() { return preprocess(NULL); }

	PreprocessorState *get_state() { return state; }

	static void get_keyword_list(List<String> *keywords);

	static void refresh_shader_dependencies(Ref<Shader> shader);

private:
	void process_directive(PreproprocessorTokenizer *);

	void process_if(PreproprocessorTokenizer *);
	void process_ifdef(PreproprocessorTokenizer *);
	void process_ifndef(PreproprocessorTokenizer*);
	void start_branch_condition(PreproprocessorTokenizer *tokenizer, bool success);

	void process_else(PreproprocessorTokenizer *);
	void process_endif(PreproprocessorTokenizer *);

	void process_define(PreproprocessorTokenizer *);
	void process_undef(PreproprocessorTokenizer *);
	void process_include(PreproprocessorTokenizer *);

	void expand_output_macros(int start, int line);
	String expand_macros(const String &p_string, int p_line);
	String expand_macros_once(const String &p_line, int line, int *p_expanded);

	String evaluate_internal_conditions(const String& p_string, int p_line);

	String next_directive(PreproprocessorTokenizer *tokenizer, const Vector<String> &directives);
	void add_to_output(const String &p_str);
	void set_error(const String &error, int line);

	static PreprocessorState *create_state();
	void free_state();

	String code;
	Vector<CharType> output;
	PreprocessorState *state;
	bool state_owner;
};

struct ShaderDependencyNode {
	int line;
	int line_count;
	String code;

	String path;
	Ref<Shader> shader;
	Set<ShaderDependencyNode*> dependencies;

	ShaderDependencyNode() = default;
	ShaderDependencyNode(Ref<Shader>);
	ShaderDependencyNode(String code);
	ShaderDependencyNode(String path, String code);

	int GetContext(int line, ShaderDependencyNode** context);
	String get_path()
	{
		if (shader.is_null())
		{
			return path;
		}

		return shader->get_path();
	}

	int get_line_count()
	{
		int total_lines = line_count - 1;
		for (ShaderDependencyNode* node : dependencies)
		{
			total_lines += node->get_line_count();
		}

		return total_lines;
	}

	~ShaderDependencyNode();

	friend bool operator<(ShaderDependencyNode, ShaderDependencyNode);
	friend bool operator==(ShaderDependencyNode, ShaderDependencyNode);
};

class ShaderDependencyGraph {
public:
	~ShaderDependencyGraph();

	Set<ShaderDependencyNode*> nodes;

	void populate(Ref<Shader>);
	void populate(String code);
	void populate(String path, String code);
	void update_shaders();

private:
	List<ShaderDependencyNode*> cyclic_dep_tracker;
	//List<Ref<Shader>> visited_shaders;
	List<String> visited_shaders;

	Set<ShaderDependencyNode*>::Element* find(Ref<Shader>);
	Set<ShaderDependencyNode*>::Element* find(String path);
	void populate(ShaderDependencyNode*);
	void update_shaders(ShaderDependencyNode*);
};

#endif

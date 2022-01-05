/*************************************************************************/
/*  shader_preprocessor.h                                                */
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

#ifndef SHADER_PREPROCESSOR_H
#define SHADER_PREPROCESSOR_H

#include "core/string/ustring.h"
#include "core/templates/list.h"
#include "core/templates/map.h"
#include "core/templates/set.h"
#include "core/typedefs.h"

#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "scene/resources/shader.h"

class PreprocessorTokenizer;
struct PPToken;

typedef char CharType;

struct PreprocessorDefine {
	Vector<String> arguments;
	String body;
};

struct SkippedPreprocessorCondition {
	int start_line = -1;
	int end_line = -1;
};

struct PreprocessorState {
	Map<String, PreprocessorDefine *> defines;
	Vector<bool> skip_stack_else;
	int condition_depth;
	Set<String> includes;
	List<uint64_t> cyclic_include_hashes; // holds code hash of includes
	int include_depth;
	String current_include = "";
	String error;
	int error_line;
	Map<String, Vector<SkippedPreprocessorCondition *>> skipped_conditions;
	bool disabled;
};

class ShaderPreprocessor {
public:
	~ShaderPreprocessor();
	ShaderPreprocessor(const String &p_code);

	String preprocess(PreprocessorState *p_state);
	String preprocess() { return preprocess(NULL); }

	PreprocessorState *get_state() { return state; }

	static void get_keyword_list(List<String> *p_keywords);

	static void refresh_shader_dependencies(Ref<Shader> p_shader);

private:
	void process_directive(PreprocessorTokenizer *p_tokenizer);

	void process_if(PreprocessorTokenizer *p_tokenizer);
	void process_ifdef(PreprocessorTokenizer *p_tokenizer);
	void process_ifndef(PreprocessorTokenizer *p_tokenizer);
	void start_branch_condition(PreprocessorTokenizer *p_tokenizer, bool p_success);

	void process_else(PreprocessorTokenizer *p_tokenizer);
	void process_endif(PreprocessorTokenizer *p_tokenizer);

	void process_define(PreprocessorTokenizer *p_tokenizer);
	void process_undef(PreprocessorTokenizer *p_tokenizer);
	void process_include(PreprocessorTokenizer *p_tokenizer);

	void process_pragma(PreprocessorTokenizer *p_tokenizer);

	void expand_output_macros(int p_start, int p_line);
	String expand_macros(const String &p_string, int p_line);
	String expand_macros(const String &p_string, int p_line, Vector<Pair<String, PreprocessorDefine *>> p_defines);
	String expand_macros_once(const String &p_line, int p_line_number, Pair<String, PreprocessorDefine *> p_define_pair, bool &r_expanded);
	bool find_match(const String &p_string, const String &p_value, int &r_index, int &r_index_start);
	bool is_char_word(const CharType p_char);

	String evaluate_internal_conditions(const String &p_string, int p_line);

	String next_directive(PreprocessorTokenizer *p_tokenizer, const Vector<String> &p_directives);
	void add_to_output(const String &p_str);
	void set_error(const String &p_error, int p_line);

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

	Ref<Shader> shader;
	Set<ShaderDependencyNode *> dependencies;

	ShaderDependencyNode() = default;
	ShaderDependencyNode(Ref<Shader> p_shader);
	ShaderDependencyNode(String p_code);

	int GetContext(int p_line, ShaderDependencyNode **r_context);
	String get_code() {
		if (shader.is_null()) {
			return code;
		}

		return shader->get_code();
	}

	String get_path() {
		if (shader.is_null()) {
			return "";
		}

		return shader->get_path();
	}

	int get_line_count() {
		int total_lines = line_count - 1;
		for (ShaderDependencyNode *node : dependencies) {
			total_lines += node->get_line_count();
		}

		return total_lines;
	}

	int num_deps();

	~ShaderDependencyNode();

	friend bool operator<(ShaderDependencyNode p_left, ShaderDependencyNode p_right);
	friend bool operator==(ShaderDependencyNode p_left, ShaderDependencyNode p_right);
};

class ShaderDependencyGraph {
public:
	~ShaderDependencyGraph();

	Set<ShaderDependencyNode *> nodes;

	void populate(Ref<Shader> p_shader);
	void populate(String p_code);
	void update_shaders();
	void clear();

	// gets size of graph -- total nodes/deps in tree.
	int size();

private:
	List<ShaderDependencyNode *> cyclic_dep_tracker;
	List<String> visited_shaders;

	Set<ShaderDependencyNode *>::Element *find(Ref<Shader> p_shader);
	List<ShaderDependencyNode *>::Element *find(uint64_t hash);
	void populate(ShaderDependencyNode *p_node);
	void update_shaders(ShaderDependencyNode *p_node);
};

#endif

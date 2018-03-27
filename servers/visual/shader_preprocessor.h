/*************************************************************************/
/*  shader_preprocessor.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "list.h"
#include "map.h"
#include "set.h"
#include "typedefs.h"
#include "ustring.h"

#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "scene/resources/shader.h"

class PreproprocessorTokenizer;
class PPToken;

struct PreprocessorDefine {
	Vector<String> arguments;
	String body;
};

struct PreprocessorState {
	Map<String, PreprocessorDefine *> defines;
	Vector<bool> skip_stack_else;
	int condition_depth;
	Set<String> includes;
	int include_depth;
	String error;
	int error_line;
};

class ShaderPreprocessor {

public:
	~ShaderPreprocessor();
	ShaderPreprocessor(const String &p_code);

	String preprocess(PreprocessorState *p_state);
	String preprocess() { return preprocess(NULL); }

	PreprocessorState *get_state() { return state; }

	static void get_keyword_list(List<String> *keywords);

	static void refresh_shader_dependencies(Shader *p_shader);

private:
	void process_directive(PreproprocessorTokenizer *);

	void process_if(PreproprocessorTokenizer *);
	void process_ifdef(PreproprocessorTokenizer *);
	void start_branch_condition(PreproprocessorTokenizer *tokenizer, bool success);

	void process_else(PreproprocessorTokenizer *);
	void process_endif(PreproprocessorTokenizer *);

	void process_define(PreproprocessorTokenizer *);
	void process_undef(PreproprocessorTokenizer *);
	void process_include(PreproprocessorTokenizer *);

	void expand_output_macros(int start, int line);
	String expand_macros(const String &p_string, int p_line);
	String expand_macros_once(const String &p_line, int line, int *p_expanded);

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

#endif

/*************************************************************************/
/*  shader_preprocessor.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "core/templates/local_vector.h"
#include "core/templates/rb_map.h"
#include "core/templates/rb_set.h"
#include "core/typedefs.h"

#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "scene/resources/shader.h"
#include "scene/resources/shader_include.h"

class ShaderPreprocessor {
public:
	enum CompletionType {
		COMPLETION_TYPE_NONE,
		COMPLETION_TYPE_DIRECTIVE,
		COMPLETION_TYPE_PRAGMA_DIRECTIVE,
		COMPLETION_TYPE_PRAGMA,
		COMPLETION_TYPE_INCLUDE_PATH,
	};

	struct FilePosition {
		String file;
		int line = 0;
	};

private:
	struct Token {
		char32_t text;
		int line;

		Token();
		Token(char32_t p_text, int p_line);
	};

	// The real preprocessor that understands basic shader and preprocessor language syntax.
	class Tokenizer {
	public:
		String code;
		int line;
		int index;
		int size;
		Vector<Token> generated;

	private:
		void add_generated(const Token &p_t);
		char32_t next();

	public:
		int get_line() const;
		int get_index() const;
		char32_t peek();

		void get_and_clear_generated(Vector<Token> *r_out);
		void backtrack(char32_t p_what);
		LocalVector<Token> advance(char32_t p_what);
		void skip_whitespace();
		String get_identifier(bool *r_is_cursor = nullptr, bool p_started = false);
		String peek_identifier();
		Token get_token();

		Tokenizer(const String &p_code);
	};

	class CommentRemover {
	private:
		LocalVector<char32_t> stripped;
		String code;
		int index;
		int line;
		int comment_line_open;
		int comments_open;
		int strings_open;

	public:
		String get_error() const;
		int get_error_line() const;
		char32_t peek() const;

		bool advance(char32_t p_what);
		String strip();

		CommentRemover(const String &p_code);
	};

	struct Define {
		Vector<String> arguments;
		String body;
	};

	struct SkippedCondition {
		int start_line = -1;
		int end_line = -1;
	};

	struct State {
		RBMap<String, Define *> defines;
		Vector<bool> skip_stack_else;
		int condition_depth = 0;
		RBSet<String> includes;
		List<uint64_t> cyclic_include_hashes; // Holds code hash of includes.
		int include_depth = 0;
		String current_include;
		String current_shader_type;
		String error;
		List<FilePosition> include_positions;
		RBMap<String, Vector<SkippedCondition *>> skipped_conditions;
		bool disabled = false;
		CompletionType completion_type = COMPLETION_TYPE_NONE;
		HashSet<Ref<ShaderInclude>> shader_includes;
	};

private:
	LocalVector<char32_t> output;
	State *state = nullptr;
	bool state_owner = false;

private:
	static bool is_char_word(char32_t p_char);
	static bool is_char_space(char32_t p_char);
	static bool is_char_end(char32_t p_char);
	static String vector_to_string(const LocalVector<char32_t> &p_v, int p_start = 0, int p_end = -1);
	static String tokens_to_string(const LocalVector<Token> &p_tokens);

	void process_directive(Tokenizer *p_tokenizer);
	void process_define(Tokenizer *p_tokenizer);
	void process_else(Tokenizer *p_tokenizer);
	void process_endif(Tokenizer *p_tokenizer);
	void process_if(Tokenizer *p_tokenizer);
	void process_ifdef(Tokenizer *p_tokenizer);
	void process_ifndef(Tokenizer *p_tokenizer);
	void process_include(Tokenizer *p_tokenizer);
	void process_pragma(Tokenizer *p_tokenizer);
	void process_undef(Tokenizer *p_tokenizer);

	void start_branch_condition(Tokenizer *p_tokenizer, bool p_success);

	void expand_output_macros(int p_start, int p_line);
	Error expand_macros(const String &p_string, int p_line, String &r_result);
	Error expand_macros(const String &p_string, int p_line, Vector<Pair<String, Define *>> p_defines, String &r_result);
	Error expand_macros_once(const String &p_line, int p_line_number, Pair<String, Define *> p_define_pair, String &r_expanded);
	bool find_match(const String &p_string, const String &p_value, int &r_index, int &r_index_start);

	String next_directive(Tokenizer *p_tokenizer, const Vector<String> &p_directives);
	void add_to_output(const String &p_str);
	void set_error(const String &p_error, int p_line);

	static Define *create_define(const String &p_body);

	void clear();

	Error preprocess(State *p_state, const String &p_code, String &r_result);

public:
	typedef void (*IncludeCompletionFunction)(List<ScriptLanguage::CodeCompletionOption> *);

	Error preprocess(const String &p_code, String &r_result, String *r_error_text = nullptr, List<FilePosition> *r_error_position = nullptr, HashSet<Ref<ShaderInclude>> *r_includes = nullptr, List<ScriptLanguage::CodeCompletionOption> *r_completion_options = nullptr, IncludeCompletionFunction p_include_completion_func = nullptr);

	static void get_keyword_list(List<String> *r_keywords, bool p_include_shader_keywords);
	static void get_pragma_list(List<String> *r_pragmas);

	ShaderPreprocessor();
	~ShaderPreprocessor();
};

#endif // SHADER_PREPROCESSOR_H

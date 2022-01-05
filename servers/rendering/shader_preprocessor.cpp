/*************************************************************************/
/*  shader_preprocessor.cpp                                              */
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

#include "shader_preprocessor.h"
#include "core/math/expression.h"
#include "editor/property_editor.h"

static bool is_whitespace(CharType p_c) {
	return (p_c == ' ') || (p_c == '\t');
}

static String vector_to_string(const Vector<CharType> &p_v, int p_start = 0, int p_end = -1) {
	const int stop = (p_end == -1) ? p_v.size() : p_end;
	const int count = stop - p_start;

	String result;
	result.resize(count + 1);
	for (int i = 0; i < count; i++) {
		result[i] = p_v[p_start + i];
	}
	result[count] = 0; //Ensure string is null terminated for length() to work
	return result;
}

struct PPToken {
	CharType text;
	int line;

	PPToken() {
		text = 0;
		line = -1;
	}

	PPToken(CharType p_text, int p_line) {
		text = p_text;
		line = p_line;
	}
};

static String tokens_to_string(const Vector<PPToken> &p_tokens) {
	Vector<CharType> result;
	for (int i = 0; i < p_tokens.size(); i++) {
		result.push_back(p_tokens[i].text);
	}
	return vector_to_string(result);
}

//Simple processor that can strip away C-like comments from a text

class CommentRemover {
private:
	Vector<CharType> stripped;
	String code;
	int index;
	int line;
	int comment_line_open;
	int comments_open;
	int strings_open;

public:
	CommentRemover(const String &p_code) {
		code = p_code;
		index = 0;
		line = 0;
		comment_line_open = 0;
		comments_open = 0;
		strings_open = 0;
	}

	String get_error() {
		if (comments_open != 0) {
			return "Block comment mismatch";
		}
		return "";
	}

	int get_error_line() {
		if (comments_open != 0) {
			return comment_line_open;
		}
		return -1;
	}

	CharType peek() {
		if (index < code.size()) {
			return code[index];
		}
		return 0;
	}

	bool advance(CharType p_what) {
		while (index < code.size()) {
			CharType c = code[index++];

			if (c == '\n') {
				line++;
				stripped.push_back('\n');
			}

			if (c == p_what) {
				return true;
			}
		}
		return false;
	}

	String strip() {
		stripped.clear();
		index = 0;
		line = 0;
		comment_line_open = 0;
		comments_open = 0;
		strings_open = 0;

		while (index < code.size()) {
			CharType c = code[index++];

			if (c < 0) {
				// skip invalid chars.
				continue;
			} else if (c == '"') {
				if (strings_open <= 0) {
					strings_open++;
				} else {
					strings_open--;
				}
				stripped.push_back(c);
			} else if (c == '/' && strings_open == 0) {
				CharType p = peek();
				if (p == '/') { //Single line comment
					advance('\n');
				} else if (p == '*') { //Start of a block comment
					index++;
					comment_line_open = line;
					comments_open++;
					while (advance('*')) {
						if (peek() == '/') { //End of a block comment
							comments_open--;
							index++;
							break;
						}
					}
				} else {
					stripped.push_back(c);
				}
			} else if (c == '*' && strings_open == 0) {
				if (peek() == '/') { //Unmatched end of a block comment
					comment_line_open = line;
					comments_open--;
				} else {
					stripped.push_back(c);
				}
			} else if (c == '\n') {
				line++;
				stripped.push_back(c);
			} else {
				stripped.push_back(c);
			}
		}
		return vector_to_string(stripped);
	}
};

//The real preprocessor that understands basic shader and preprocessor language syntax
class PreprocessorTokenizer {
public:
	String code;
	int line;
	int index;
	int size;
	Vector<PPToken> generated;

private:
	void add_generated(const PPToken &p_t) {
		generated.push_back(p_t);
	}

	CharType next() {
		if (index < size) {
			return code[index++];
		}
		return 0;
	}

public:
	PreprocessorTokenizer(String p_code) {
		code = p_code;
		line = 0;
		index = 0;
		size = code.size();
	}

	int get_line() {
		return line;
	}

	int get_index() {
		return index;
	}

	void get_and_clear_generated(Vector<PPToken> *r_out) {
		for (int i = 0; i < generated.size(); i++) {
			r_out->push_back(generated[i]);
		}
		generated.clear();
	}

	void backtrack(CharType p_what) {
		while (index >= 0) {
			CharType c = code[index];
			if (c == p_what) {
				break;
			}
			index--;
		}
	}

	CharType peek() {
		if (index < size) {
			return code[index];
		}
		return 0;
	}

	Vector<PPToken> advance(CharType p_what) {
		Vector<PPToken> tokens;

		while (index < size) {
			CharType c = code[index++];

			tokens.push_back(PPToken(c, line));

			if (c == '\n') {
				add_generated(PPToken('\n', line));
				line++;
			}

			if (c == p_what) {
				return tokens;
			}
		}
		return Vector<PPToken>();
	}

	void skip_whitespace() {
		while (is_whitespace(peek())) {
			next();
		}
	}

	String get_identifier() {
		Vector<CharType> text;

		bool started = false;
		while (1) {
			CharType c = peek();
			if (c == 0 || c == '\n' || c == '(' || c == ')' || c == ',') {
				break;
			}

			if (is_whitespace(c) && started) {
				break;
			}
			if (!is_whitespace(c)) {
				started = true;
			}

			CharType n = next();
			if (started) {
				text.push_back(n);
			}
		}

		String id = vector_to_string(text);
		if (!id.is_valid_identifier()) {
			return "";
		}

		return id;
	}

	String peek_identifier() {
		const int original = index;
		String id = get_identifier();
		index = original;
		return id;
	}

	PPToken get_token() {
		while (index < size) {
			const CharType c = code[index++];
			const PPToken t = PPToken(c, line);

			switch (c) {
				case ' ':
				case '\t':
					skip_whitespace();
					return PPToken(' ', line);
				case '\n':
					line++;
					return t;
				default:
					return t;
			}
		}
		return PPToken(0, line);
	}
};

ShaderPreprocessor::~ShaderPreprocessor() {
	free_state();
}

ShaderPreprocessor::ShaderPreprocessor(const String &p_code) :
		code(p_code), state(nullptr), state_owner(false) {
}

String ShaderPreprocessor::preprocess(PreprocessorState *p_state) {
	free_state();

	output.clear();

	state = p_state;
	if (state == nullptr) {
		state = create_state();
		state_owner = true;
	}

	// track code hashes to prevent cyclic include.
	uint64_t code_hash = code.hash64();
	state->cyclic_include_hashes.push_back(code_hash);

	CommentRemover remover(code);
	String stripped = remover.strip();
	String error = remover.get_error();
	if (!error.is_empty()) {
		set_error(error, remover.get_error_line());
		return "<error>";
	}

	PreprocessorTokenizer p_tokenizer(stripped);
	int last_size = 0;

	while (1) {
		const PPToken &t = p_tokenizer.get_token();

		//Add autogenerated tokens
		Vector<PPToken> generated;
		p_tokenizer.get_and_clear_generated(&generated);
		for (int i = 0; i < generated.size(); i++) {
			output.push_back(generated[i].text);
		}

		if (t.text == 0) {
			break;
		}

		if (state->disabled) {
			// preprocessor was disabled
			// read the rest of the file into the output.
			output.push_back(t.text);
			continue;
		}

		if (t.text == '#') { //TODO check if at the beginning of line
			process_directive(&p_tokenizer);
		} else {
			if (t.text == '\n') {
				expand_output_macros(last_size, p_tokenizer.get_line());
				last_size = output.size();
			}
			output.push_back(t.text);
		}

		if (!state->error.is_empty()) {
			return "<error>";
		}
	}

	expand_output_macros(last_size, p_tokenizer.get_line());

	String result = vector_to_string(output);

	// remove this hash.
	state->cyclic_include_hashes.erase(code_hash);

	return result;
}

void ShaderPreprocessor::process_directive(PreprocessorTokenizer *p_tokenizer) {
	String directive = p_tokenizer->get_identifier();

	if (directive == "if") {
		process_if(p_tokenizer);
	} else if (directive == "ifdef") {
		process_ifdef(p_tokenizer);
	} else if (directive == "ifndef") {
		process_ifndef(p_tokenizer);
	} else if (directive == "else") {
		process_else(p_tokenizer);
	} else if (directive == "endif") {
		process_endif(p_tokenizer);
	} else if (directive == "define") {
		process_define(p_tokenizer);
	} else if (directive == "undef") {
		process_undef(p_tokenizer);
	} else if (directive == "include") {
		process_include(p_tokenizer);
	} else if (directive == "pragma") {
		process_pragma(p_tokenizer);
	} else {
		set_error("Unknown directive", p_tokenizer->get_line());
	}
}

void ShaderPreprocessor::process_if(PreprocessorTokenizer *p_tokenizer) {
	int line = p_tokenizer->get_line();

	String body = tokens_to_string(p_tokenizer->advance('\n')).strip_edges();
	if (body.is_empty()) {
		set_error("Missing condition", line);
		return;
	}

	body = expand_macros(body, line);
	if (!state->error.is_empty()) {
		return;
	}

	Expression expression;
	Vector<String> names;
	Error error = expression.parse(body, names);
	if (error != OK) {
		set_error(expression.get_error_text(), line);
		return;
	}

	Variant v = expression.execute(Array(), nullptr, false);
	if (v.get_type() == Variant::NIL) {
		set_error("Condition evaluation error", line);
		return;
	}

	bool success = v.booleanize();
	start_branch_condition(p_tokenizer, success);
}

void ShaderPreprocessor::process_ifdef(PreprocessorTokenizer *p_tokenizer) {
	const int line = p_tokenizer->get_line();

	String label = p_tokenizer->get_identifier();
	if (label.is_empty()) {
		set_error("Invalid macro name", line);
		return;
	}

	p_tokenizer->skip_whitespace();
	if (p_tokenizer->peek() != '\n') {
		set_error("Invalid ifdef", line);
		return;
	}
	p_tokenizer->advance('\n');

	bool success = state->defines.has(label);
	start_branch_condition(p_tokenizer, success);
}

void ShaderPreprocessor::process_ifndef(PreprocessorTokenizer *p_tokenizer) {
	const int line = p_tokenizer->get_line();

	String label = p_tokenizer->get_identifier();
	if (label.is_empty()) {
		set_error("Invalid macro name", line);
		return;
	}

	p_tokenizer->skip_whitespace();
	if (p_tokenizer->peek() != '\n') {
		set_error("Invalid ifndef", line);
		return;
	}
	p_tokenizer->advance('\n');

	bool success = !state->defines.has(label);
	start_branch_condition(p_tokenizer, success);
}

void ShaderPreprocessor::start_branch_condition(PreprocessorTokenizer *p_tokenizer, bool p_success) {
	state->condition_depth++;

	if (p_success) {
		state->skip_stack_else.push_back(true);
	} else {
		SkippedPreprocessorCondition *cond = memnew(SkippedPreprocessorCondition());
		cond->start_line = p_tokenizer->get_line();
		state->skipped_conditions[state->current_include].push_back(cond);

		Vector<String> ends;
		ends.push_back("else");
		ends.push_back("endif");
		if (next_directive(p_tokenizer, ends) == "else") {
			state->skip_stack_else.push_back(false);
		} else {
			state->skip_stack_else.push_back(true);
		}
	}
}

void ShaderPreprocessor::process_else(PreprocessorTokenizer *p_tokenizer) {
	if (state->skip_stack_else.is_empty()) {
		set_error("Unmatched else", p_tokenizer->get_line());
		return;
	}
	p_tokenizer->advance('\n');

	bool skip = state->skip_stack_else[state->skip_stack_else.size() - 1];
	state->skip_stack_else.remove_at(state->skip_stack_else.size() - 1);

	Vector<SkippedPreprocessorCondition *> vec = state->skipped_conditions[state->current_include];
	int index = vec.size() - 1;
	if (index >= 0) {
		SkippedPreprocessorCondition *cond = vec[index];
		if (cond->end_line == -1) {
			cond->end_line = p_tokenizer->get_line();
		}
	}

	if (skip) {
		Vector<String> ends;
		ends.push_back("endif");
		next_directive(p_tokenizer, ends);
	}
}

void ShaderPreprocessor::process_endif(PreprocessorTokenizer *p_tokenizer) {
	state->condition_depth--;
	if (state->condition_depth < 0) {
		set_error("Unmatched endif", p_tokenizer->get_line());
		return;
	}

	Vector<SkippedPreprocessorCondition *> vec = state->skipped_conditions[state->current_include];
	int index = vec.size() - 1;
	if (index >= 0) {
		SkippedPreprocessorCondition *cond = vec[index];
		if (cond->end_line == -1) {
			cond->end_line = p_tokenizer->get_line();
		}
	}

	p_tokenizer->advance('\n');
}

void ShaderPreprocessor::process_define(PreprocessorTokenizer *p_tokenizer) {
	const int line = p_tokenizer->get_line();

	String label = p_tokenizer->get_identifier();
	if (label.is_empty()) {
		set_error("Invalid macro name", line);
		return;
	}

	if (state->defines.has(label)) {
		set_error("Macro redefinition", line);
		return;
	}

	if (p_tokenizer->peek() == '(') {
		//Macro has arguments
		p_tokenizer->get_token();

		Vector<String> args;
		while (1) {
			String name = p_tokenizer->get_identifier();
			if (name.is_empty()) {
				set_error("Invalid argument name", line);
				return;
			}
			args.push_back(name);

			p_tokenizer->skip_whitespace();
			CharType next = p_tokenizer->get_token().text;
			if (next == ')') {
				break;
			} else if (next != ',') {
				set_error("Expected a comma in the macro argument list", line);
				return;
			}
		}

		PreprocessorDefine *define = memnew(PreprocessorDefine);
		define->arguments = args;
		define->body = tokens_to_string(p_tokenizer->advance('\n')).strip_edges();
		state->defines[label] = define;
	} else {
		//Simple substitution macro
		PreprocessorDefine *define = memnew(PreprocessorDefine);
		define->body = tokens_to_string(p_tokenizer->advance('\n')).strip_edges();
		state->defines[label] = define;
	}
}

void ShaderPreprocessor::process_undef(PreprocessorTokenizer *p_tokenizer) {
	const int line = p_tokenizer->get_line();
	const String label = p_tokenizer->get_identifier();
	if (label.is_empty()) {
		set_error("Invalid name", line);
		return;
	}

	p_tokenizer->skip_whitespace();
	if (p_tokenizer->peek() != '\n') {
		set_error("Invalid undef", line);
		return;
	}

	memdelete(state->defines[label]);
	state->defines.erase(label);
}

void ShaderPreprocessor::process_include(PreprocessorTokenizer *p_tokenizer) {
	const int line = p_tokenizer->get_line();

	p_tokenizer->advance('"');
	String path = tokens_to_string(p_tokenizer->advance('"'));
	path = path.substr(0, path.length() - 1);
	p_tokenizer->skip_whitespace();

	if (path.is_empty() || p_tokenizer->peek() != '\n') {
		set_error("Invalid path", line);
		return;
	}

	RES res = ResourceLoader::load(path);
	if (res.is_null()) {
		set_error("Shader include load failed. Does the shader exist? Is there a cyclic dependency?", line);
		return;
	}

	Ref<Shader> shader = Object::cast_to<Shader>(*res);
	if (shader.is_null()) {
		set_error("Shader include resource type is wrong", line);
		return;
	}

	String included = shader->get_code();
	if (included.is_empty()) {
		set_error("Shader include not found", line);
		return;
	}

	uint64_t code_hash = included.hash64();
	if (state->cyclic_include_hashes.find(code_hash)) {
		set_error("Cyclic include found.", line);
		return;
	}

	int type_end = included.find(";");
	if (type_end == -1) {
		set_error("Shader include shader_type not found", line);
		return;
	}

	const String real_path = shader->get_path();
	if (state->includes.has(real_path)) {
		//Already included, skip.
		//This is a valid check because 2 separate include paths could use some
		//of the same shared functions from a common shader include.
		return;
	}

	//Mark as included
	state->includes.insert(real_path);

	state->include_depth++;
	if (state->include_depth > 25) {
		set_error("Shader max include depth exceeded", line);
		return;
	}

	//Replace shader_type line with a comment. Easy to maintain total lines this way.
	included = included.replace("shader_type ", "//shader_type ");

	String old_include = state->current_include;
	state->current_include = real_path;
	ShaderPreprocessor processor(included);
	String result = processor.preprocess(state);
	add_to_output(result);

	// reset to last include if there are no errors. We want to use this as context.
	if (state->error.is_empty()) {
		state->current_include = old_include;
	}

	state->include_depth--;
}

void ShaderPreprocessor::process_pragma(PreprocessorTokenizer *p_tokenizer) {
	const int line = p_tokenizer->get_line();
	const String label = p_tokenizer->get_identifier();
	if (label.is_empty()) {
		set_error("Invalid pragma value", line);
		return;
	}

	// explicitly handle pragma values here.
	// if more pragma options are created, then refactor into a more defined structure.
	if (label == "disable_preprocessor") {
		state->disabled = true;
	}

	p_tokenizer->advance('\n');
}

void ShaderPreprocessor::expand_output_macros(int p_start, int p_line_number) {
	String line = vector_to_string(output, p_start, output.size());

	line = expand_macros(line, p_line_number - 1); //We are already on next line, so -1

	output.resize(p_start);

	add_to_output(line);
}

String ShaderPreprocessor::expand_macros(const String &p_string, int p_line) {
	Vector<Pair<String, PreprocessorDefine *>> active_defines;
	active_defines.resize(state->defines.size());
	int index = 0;
	for (const Map<String, PreprocessorDefine *>::Element *E = state->defines.front(); E; E = E->next()) {
		active_defines.set(index++, Pair<String, PreprocessorDefine *>(E->key(), E->get()));
	}

	return expand_macros(p_string, p_line, active_defines);
}

String ShaderPreprocessor::expand_macros(const String &p_string, int p_line, Vector<Pair<String, PreprocessorDefine *>> p_defines) {
	String result = p_string;
	bool expanded = false;
	// when expanding macros we must only evaluate them once.
	// later we continue expanding but with the already
	// evaluated macros removed.
	for (int i = 0; i < p_defines.size(); i++) {
		Pair<String, PreprocessorDefine *> define_pair = p_defines[i];

		result = expand_macros_once(result, p_line, define_pair, expanded);

		if (!state->error.is_empty()) {
			return "<error>";
		}

		if (expanded) {
			// remove expanded macro and recursively replace remaining.
			p_defines.remove_at(i);
			return expand_macros(result, p_line, p_defines);
		}
	}

	return result;
}

String ShaderPreprocessor::expand_macros_once(const String &p_line, int p_line_number, Pair<String, PreprocessorDefine *> p_define_pair, bool &r_expanded) {
	String result = p_line;
	r_expanded = false;

	const String &key = p_define_pair.first;
	const PreprocessorDefine *define = p_define_pair.second;

	int index_start = 0;
	int index = 0;
	while (find_match(result, key, index, index_start)) {
		r_expanded = true;
		String body = define->body;
		if (define->arguments.size() > 0) {
			//Complex macro with arguments
			int args_start = index + key.length();
			int args_end = p_line.find(")", args_start);
			if (args_start == -1 || args_end == -1) {
				r_expanded = false;
				set_error("Missing macro argument parenthesis", p_line_number);
				return "<error>";
			}

			String values = result.substr(args_start + 1, args_end - (args_start + 1));
			Vector<String> args = values.split(",");
			if (args.size() != define->arguments.size()) {
				r_expanded = false;
				set_error("Invalid macro argument count", p_line_number);
				return "<error>";
			}

			//Insert macro arguments into the body
			for (int i = 0; i < args.size(); i++) {
				String arg_name = define->arguments[i];
				int arg_index_start = 0;
				int arg_index = 0;
				while (find_match(body, arg_name, arg_index, arg_index_start)) {
					body = body.substr(0, arg_index) + args[i] + body.substr(arg_index + arg_name.length(), body.length() - (arg_index + arg_name.length()));
					// manually reset arg_index_start to where the arg value of the define finishes.
					// this ensures we don't skip the other args of this macro in the string.
					arg_index_start = arg_index + args[i].length() + 1;
					r_expanded = true;
				}
			}

			result = result.substr(0, index) + " " + body + " " + result.substr(args_end + 1, result.length());
		} else {
			result = result.substr(0, index) + body + result.substr(index + key.length(), result.length() - (index + key.length()));
			// manually reset index_start to where the body value of the define finishes.
			// this ensures we don't skip another instance of this macro in the string.
			index_start = index + body.length() + 1;
			r_expanded = true;
			break;
		}
	}

	return result;
}

bool ShaderPreprocessor::find_match(const String &p_string, const String &p_value, int &r_index, int &r_index_start) {
	// looks for value in string and then determines if the boundaries
	// are non-word characters. This method semi-emulates \b in regex.
	r_index = p_string.find(p_value, r_index_start);
	while (r_index > -1) {
		if (r_index > 0) {
			if (is_char_word(p_string[r_index - 1])) {
				r_index_start = r_index + 1;
				r_index = p_string.find(p_value, r_index_start);
				continue;
			}
		}

		if (r_index + p_value.length() < p_string.length()) {
			if (is_char_word(p_string[r_index + p_value.length()])) {
				r_index_start = r_index + p_value.length() + 1;
				r_index = p_string.find(p_value, r_index_start);
				continue;
			}
		}

		// return and shift index start automatically for next call.
		r_index_start = r_index + p_value.length() + 1;
		return true;
	}

	return false;
}

bool ShaderPreprocessor::is_char_word(const CharType p_char) {
	if ((p_char >= '0' && p_char <= '9') ||
			(p_char >= 'a' && p_char <= 'z') ||
			(p_char >= 'A' && p_char <= 'Z') ||
			p_char == '_') {
		return true;
	}

	return false;
}

String ShaderPreprocessor::next_directive(PreprocessorTokenizer *p_tokenizer, const Vector<String> &p_directives) {
	const int line = p_tokenizer->get_line();
	int nesting = 0;

	while (1) {
		p_tokenizer->advance('#');

		String id = p_tokenizer->peek_identifier();
		if (id.is_empty()) {
			break;
		}

		if (nesting == 0) {
			for (int i = 0; i < p_directives.size(); i++) {
				if (p_directives[i] == id) {
					p_tokenizer->backtrack('#');
					return id;
				}
			}
		}

		if (id == "ifdef" || id == "ifndef" || id == "if") {
			nesting++;
		} else if (id == "endif") {
			nesting--;
		}
	}

	set_error("Can't find matching branch directive", line);
	return "";
}

void ShaderPreprocessor::add_to_output(const String &p_str) {
	for (int i = 0; i < p_str.length(); i++) {
		output.push_back(p_str[i]);
	}
}

void ShaderPreprocessor::set_error(const String &p_error, int p_line) {
	if (state->error.is_empty()) {
		state->error = p_error;
		state->error_line = p_line + 1;
	}
}

void ShaderPreprocessor::free_state() {
	if (state_owner && state != nullptr) {
		for (const Map<String, PreprocessorDefine *>::Element *E = state->defines.front(); E; E = E->next()) {
			memdelete(E->get());
		}

		for (const Map<String, Vector<SkippedPreprocessorCondition *>>::Element *E = state->skipped_conditions.front(); E; E = E->next()) {
			for (SkippedPreprocessorCondition *condition : E->get()) {
				memdelete(condition);
			}
		}

		memdelete(state);
	}
	state_owner = false;
	state = nullptr;
}

PreprocessorDefine *create_define(const String &p_body) {
	PreprocessorDefine *define = memnew(PreprocessorDefine);
	define->body = p_body;
	return define;
}

PreprocessorState *ShaderPreprocessor::create_state() {
	PreprocessorState *new_state = memnew(PreprocessorState);
	new_state->condition_depth = 0;
	new_state->include_depth = 0;
	new_state->error = "";
	new_state->error_line = -1;
	new_state->disabled = false;

	OS *os = OS::get_singleton();

	String platform = os->get_name().replace(" ", "_").to_upper();
	new_state->defines[platform] = create_define("true");

	Engine *engine = Engine::get_singleton();
	new_state->defines["EDITOR"] = create_define(engine->is_editor_hint() ? "true" : "false");

	return new_state;
}

void ShaderPreprocessor::get_keyword_list(List<String> *r_keywords) {
	r_keywords->push_back("include");
	r_keywords->push_back("define");
	r_keywords->push_back("undef");
	//keywords->push_back("if");  //Already a keyword
	r_keywords->push_back("ifdef");
	r_keywords->push_back("ifndef");
	//keywords->push_back("else"); //Already a keyword
	r_keywords->push_back("endif");
	r_keywords->push_back("pragma");
}

void ShaderPreprocessor::refresh_shader_dependencies(Ref<Shader> p_shader) {
	ShaderDependencyGraph shaderDependencies;
	shaderDependencies.populate(p_shader);
	shaderDependencies.update_shaders();
}

ShaderDependencyNode::ShaderDependencyNode(Ref<Shader> p_shader) :
		line(0), line_count(0), code(p_shader->get_code()), shader(p_shader) {}

ShaderDependencyNode::ShaderDependencyNode(String p_code) :
		line(0), line_count(0), code(p_code) {}

int ShaderDependencyNode::GetContext(int p_line, ShaderDependencyNode **r_context) {
	int include_offset = 0;
	for (ShaderDependencyNode *include : dependencies) {
		if (p_line < include->line + include_offset) {
			// line shifted by offset is sufficient. break.
			break;
		} else if (p_line >= include->line + include_offset && p_line <= (include->line + include_offset + include->get_line_count())) {
			return include->GetContext(p_line - include->line + 1, r_context); // plus 1 to be fully inclusive in the skip
		}

		include_offset += include->get_line_count();
	}

	*r_context = this;
	return p_line - include_offset;
}

int ShaderDependencyNode::num_deps() {
	int deps = 0;
	deps += dependencies.size();
	for (ShaderDependencyNode *node : dependencies) {
		deps += node->num_deps();
	}

	return deps;
}

ShaderDependencyNode::~ShaderDependencyNode() {
	for (ShaderDependencyNode *node : dependencies) {
		memdelete(node);
	}
}

bool operator<(ShaderDependencyNode p_left, ShaderDependencyNode p_right) {
	return p_left.shader < p_right.shader;
}

bool operator==(ShaderDependencyNode p_left, ShaderDependencyNode p_right) {
	return p_left.shader == p_right.shader;
}

void ShaderDependencyGraph::populate(Ref<Shader> p_shader) {
	clear();

	ShaderDependencyNode *node = memnew(ShaderDependencyNode(p_shader));
	nodes.insert(node);
	populate(node);
}

void ShaderDependencyGraph::populate(String p_code) {
	clear();

	ShaderDependencyNode *node = memnew(ShaderDependencyNode(p_code));
	nodes.insert(node);
	populate(node);
}

ShaderDependencyGraph::~ShaderDependencyGraph() {
	clear();
}

void ShaderDependencyGraph::populate(ShaderDependencyNode *p_node) {
	ERR_FAIL_COND_MSG(find(p_node->code.hash64()), vformat("Shader %s contains a cyclic import. Skipping...", p_node->get_path()));

	cyclic_dep_tracker.push_back(p_node);
	String code = CommentRemover(p_node->get_code()).strip(); // Build dependency graph starting from edited shader. Strip comments

	PreprocessorTokenizer p_tokenizer(code);
	while (1) {
		if (!p_tokenizer.advance('#').is_empty()) {
			String directive = p_tokenizer.get_identifier();
			if (directive == "include") {
				if (!p_tokenizer.advance('"').is_empty()) {
					String path = tokens_to_string(p_tokenizer.advance('"'));
					path = path.substr(0, path.length() - 1);
					p_tokenizer.skip_whitespace();
					if (!path.is_empty()) {
						RES res = ResourceLoader::load(path);
						ERR_FAIL_COND_MSG(res.is_null(), vformat("Could not load included shader %s. Does the shader exist? Is there a cyclic dependency?", path));
						if (!res.is_null()) {
							Ref<Shader> shader_reference = Object::cast_to<Shader>(*res);
							if (!shader_reference.is_null()) {
								// skip this shader if we've picked it up as a dependency already
								String shader_path = shader_reference->get_path();
								if (!visited_shaders.find(shader_path)) {
									visited_shaders.push_back(shader_path);
									String included_code = shader_reference->get_code();
									ShaderDependencyNode *new_node = memnew(ShaderDependencyNode(shader_reference));
									new_node->line = p_tokenizer.get_line() + 1;
									Vector<String> shader = included_code.split("\n");
									new_node->line_count = shader.size();
									populate(new_node);
									p_node->dependencies.insert(new_node);
								}
							}
						}
					}
				}
			}
		} else {
			break;
		}
	}

	cyclic_dep_tracker.erase(p_node);
}

Set<ShaderDependencyNode *>::Element *ShaderDependencyGraph::find(Ref<Shader> p_shader) {
	for (Set<ShaderDependencyNode *>::Element *E = nodes.front(); E; E = E->next()) {
		if (E->get()->shader == p_shader) {
			return E;
		}
	}

	return nullptr;
}

List<ShaderDependencyNode *>::Element *ShaderDependencyGraph::find(uint64_t p_hash) {
	for (List<ShaderDependencyNode *>::Element *E = cyclic_dep_tracker.front(); E; E = E->next()) {
		if (E->get()->code.hash64() == p_hash) {
			return E;
		}
	}

	return nullptr;
}

void ShaderDependencyGraph::update_shaders() {
	for (ShaderDependencyNode *node : nodes) {
		update_shaders(node);
	}
}

void ShaderDependencyGraph::clear() {
	for (ShaderDependencyNode *node : nodes) {
		memdelete(node);
	}

	cyclic_dep_tracker.clear();
	nodes.clear();
	visited_shaders.clear();
}

int ShaderDependencyGraph::size() {
	int count = 0;
	for (ShaderDependencyNode *node : nodes) {
		count++;
		count += node->num_deps();
	}

	return count;
}

void ShaderDependencyGraph::update_shaders(ShaderDependencyNode *p_node) {
	for (ShaderDependencyNode *node : p_node->dependencies) {
		update_shaders(node);
	}

	if (!p_node->shader.is_null()) {
		p_node->shader->set_code(p_node->shader->get_code());
	}
}

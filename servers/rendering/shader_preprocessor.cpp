/*************************************************************************/
/*  shader_preprocessor.cpp                                              */
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

#include "shader_preprocessor.h"
#include "core/math/expression.h"

const char32_t CURSOR = 0xFFFF;

// Tokenizer

void ShaderPreprocessor::Tokenizer::add_generated(const ShaderPreprocessor::Token &p_t) {
	generated.push_back(p_t);
}

char32_t ShaderPreprocessor::Tokenizer::next() {
	if (index < size) {
		return code[index++];
	}
	return 0;
}

int ShaderPreprocessor::Tokenizer::get_line() const {
	return line;
}

int ShaderPreprocessor::Tokenizer::get_index() const {
	return index;
}

void ShaderPreprocessor::Tokenizer::get_and_clear_generated(Vector<ShaderPreprocessor::Token> *r_out) {
	for (int i = 0; i < generated.size(); i++) {
		r_out->push_back(generated[i]);
	}
	generated.clear();
}

void ShaderPreprocessor::Tokenizer::backtrack(char32_t p_what) {
	while (index >= 0) {
		char32_t c = code[index];
		if (c == p_what) {
			break;
		}
		index--;
	}
}

char32_t ShaderPreprocessor::Tokenizer::peek() {
	if (index < size) {
		return code[index];
	}
	return 0;
}

LocalVector<ShaderPreprocessor::Token> ShaderPreprocessor::Tokenizer::advance(char32_t p_what) {
	LocalVector<ShaderPreprocessor::Token> tokens;

	while (index < size) {
		char32_t c = code[index++];

		tokens.push_back(ShaderPreprocessor::Token(c, line));

		if (c == '\n') {
			add_generated(ShaderPreprocessor::Token('\n', line));
			line++;
		}

		if (c == p_what || c == 0) {
			return tokens;
		}
	}
	return LocalVector<ShaderPreprocessor::Token>();
}

void ShaderPreprocessor::Tokenizer::skip_whitespace() {
	while (is_char_space(peek())) {
		next();
	}
}

String ShaderPreprocessor::Tokenizer::get_identifier(bool *r_is_cursor, bool p_started) {
	if (r_is_cursor != nullptr) {
		*r_is_cursor = false;
	}

	LocalVector<char32_t> text;

	while (true) {
		char32_t c = peek();
		if (is_char_end(c) || c == '(' || c == ')' || c == ',' || c == ';') {
			break;
		}

		if (is_whitespace(c) && p_started) {
			break;
		}
		if (!is_whitespace(c)) {
			p_started = true;
		}

		char32_t n = next();
		if (n == CURSOR) {
			if (r_is_cursor != nullptr) {
				*r_is_cursor = true;
			}
		} else {
			if (p_started) {
				text.push_back(n);
			}
		}
	}

	String id = vector_to_string(text);
	if (!id.is_valid_identifier()) {
		return "";
	}

	return id;
}

String ShaderPreprocessor::Tokenizer::peek_identifier() {
	const int original = index;
	String id = get_identifier();
	index = original;
	return id;
}

ShaderPreprocessor::Token ShaderPreprocessor::Tokenizer::get_token() {
	while (index < size) {
		const char32_t c = code[index++];
		const Token t = ShaderPreprocessor::Token(c, line);

		switch (c) {
			case ' ':
			case '\t':
				skip_whitespace();
				return ShaderPreprocessor::Token(' ', line);
			case '\n':
				line++;
				return t;
			default:
				return t;
		}
	}
	return ShaderPreprocessor::Token(char32_t(0), line);
}

ShaderPreprocessor::Tokenizer::Tokenizer(const String &p_code) {
	code = p_code;
	line = 0;
	index = 0;
	size = code.size();
}

// ShaderPreprocessor::CommentRemover

String ShaderPreprocessor::CommentRemover::get_error() const {
	if (comments_open != 0) {
		return "Block comment mismatch";
	}
	return "";
}

int ShaderPreprocessor::CommentRemover::get_error_line() const {
	if (comments_open != 0) {
		return comment_line_open;
	}
	return -1;
}

char32_t ShaderPreprocessor::CommentRemover::peek() const {
	if (index < code.size()) {
		return code[index];
	}
	return 0;
}

bool ShaderPreprocessor::CommentRemover::advance(char32_t p_what) {
	while (index < code.size()) {
		char32_t c = code[index++];

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

String ShaderPreprocessor::CommentRemover::strip() {
	stripped.clear();
	index = 0;
	line = 0;
	comment_line_open = 0;
	comments_open = 0;
	strings_open = 0;

	while (index < code.size()) {
		char32_t c = code[index++];

		if (c == CURSOR) {
			// Cursor. Maintain.
			stripped.push_back(c);
		} else if (c == '"') {
			if (strings_open <= 0) {
				strings_open++;
			} else {
				strings_open--;
			}
			stripped.push_back(c);
		} else if (c == '/' && strings_open == 0) {
			char32_t p = peek();
			if (p == '/') { // Single line comment.
				advance('\n');
			} else if (p == '*') { // Start of a block comment.
				index++;
				comment_line_open = line;
				comments_open++;
				while (advance('*')) {
					if (peek() == '/') { // End of a block comment.
						comments_open--;
						index++;
						break;
					}
				}
			} else {
				stripped.push_back(c);
			}
		} else if (c == '*' && strings_open == 0) {
			if (peek() == '/') { // Unmatched end of a block comment.
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

ShaderPreprocessor::CommentRemover::CommentRemover(const String &p_code) {
	code = p_code;
	index = 0;
	line = 0;
	comment_line_open = 0;
	comments_open = 0;
	strings_open = 0;
}

// ShaderPreprocessor::Token

ShaderPreprocessor::Token::Token() {
	text = 0;
	line = -1;
}

ShaderPreprocessor::Token::Token(char32_t p_text, int p_line) {
	text = p_text;
	line = p_line;
}

// ShaderPreprocessor

bool ShaderPreprocessor::is_char_word(char32_t p_char) {
	if ((p_char >= '0' && p_char <= '9') ||
			(p_char >= 'a' && p_char <= 'z') ||
			(p_char >= 'A' && p_char <= 'Z') ||
			p_char == '_') {
		return true;
	}

	return false;
}

bool ShaderPreprocessor::is_char_space(char32_t p_char) {
	return p_char == ' ' || p_char == '\t';
}

bool ShaderPreprocessor::is_char_end(char32_t p_char) {
	return p_char == '\n' || p_char == 0;
}

String ShaderPreprocessor::vector_to_string(const LocalVector<char32_t> &p_v, int p_start, int p_end) {
	const int stop = (p_end == -1) ? p_v.size() : p_end;
	const int count = stop - p_start;

	String result;
	result.resize(count + 1);
	for (int i = 0; i < count; i++) {
		result[i] = p_v[p_start + i];
	}
	result[count] = 0; // Ensure string is null terminated for length() to work.
	return result;
}

String ShaderPreprocessor::tokens_to_string(const LocalVector<Token> &p_tokens) {
	LocalVector<char32_t> result;
	for (uint32_t i = 0; i < p_tokens.size(); i++) {
		result.push_back(p_tokens[i].text);
	}
	return vector_to_string(result);
}

void ShaderPreprocessor::process_directive(Tokenizer *p_tokenizer) {
	bool is_cursor;
	String directive = p_tokenizer->get_identifier(&is_cursor, true);
	if (is_cursor) {
		state->completion_type = COMPLETION_TYPE_DIRECTIVE;
	}

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
		set_error(RTR("Unknown directive."), p_tokenizer->get_line());
	}
}

void ShaderPreprocessor::process_define(Tokenizer *p_tokenizer) {
	const int line = p_tokenizer->get_line();

	String label = p_tokenizer->get_identifier();
	if (label.is_empty()) {
		set_error(RTR("Invalid macro name."), line);
		return;
	}

	if (state->defines.has(label)) {
		set_error(RTR("Macro redefinition."), line);
		return;
	}

	if (p_tokenizer->peek() == '(') {
		// Macro has arguments.
		p_tokenizer->get_token();

		Vector<String> args;
		while (true) {
			String name = p_tokenizer->get_identifier();
			if (name.is_empty()) {
				set_error(RTR("Invalid argument name."), line);
				return;
			}
			args.push_back(name);

			p_tokenizer->skip_whitespace();
			char32_t next = p_tokenizer->get_token().text;
			if (next == ')') {
				break;
			} else if (next != ',') {
				set_error(RTR("Expected a comma in the macro argument list."), line);
				return;
			}
		}

		Define *define = memnew(Define);
		define->arguments = args;
		define->body = tokens_to_string(p_tokenizer->advance('\n')).strip_edges();
		state->defines[label] = define;
	} else {
		// Simple substitution macro.
		Define *define = memnew(Define);
		define->body = tokens_to_string(p_tokenizer->advance('\n')).strip_edges();
		state->defines[label] = define;
	}
}

void ShaderPreprocessor::process_else(Tokenizer *p_tokenizer) {
	if (state->skip_stack_else.is_empty()) {
		set_error(RTR("Unmatched else."), p_tokenizer->get_line());
		return;
	}
	p_tokenizer->advance('\n');

	bool skip = state->skip_stack_else[state->skip_stack_else.size() - 1];
	state->skip_stack_else.remove_at(state->skip_stack_else.size() - 1);

	Vector<SkippedCondition *> vec = state->skipped_conditions[state->current_include];
	int index = vec.size() - 1;
	if (index >= 0) {
		SkippedCondition *cond = vec[index];
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

void ShaderPreprocessor::process_endif(Tokenizer *p_tokenizer) {
	state->condition_depth--;
	if (state->condition_depth < 0) {
		set_error(RTR("Unmatched endif."), p_tokenizer->get_line());
		return;
	}

	Vector<SkippedCondition *> vec = state->skipped_conditions[state->current_include];
	int index = vec.size() - 1;
	if (index >= 0) {
		SkippedCondition *cond = vec[index];
		if (cond->end_line == -1) {
			cond->end_line = p_tokenizer->get_line();
		}
	}

	p_tokenizer->advance('\n');
}

void ShaderPreprocessor::process_if(Tokenizer *p_tokenizer) {
	int line = p_tokenizer->get_line();

	String body = tokens_to_string(p_tokenizer->advance('\n')).strip_edges();
	if (body.is_empty()) {
		set_error(RTR("Missing condition."), line);
		return;
	}

	Error error = expand_macros(body, line, body);
	if (error != OK) {
		return;
	}

	Expression expression;
	Vector<String> names;
	error = expression.parse(body, names);
	if (error != OK) {
		set_error(expression.get_error_text(), line);
		return;
	}

	Variant v = expression.execute(Array(), nullptr, false);
	if (v.get_type() == Variant::NIL) {
		set_error(RTR("Condition evaluation error."), line);
		return;
	}

	bool success = v.booleanize();
	start_branch_condition(p_tokenizer, success);
}

void ShaderPreprocessor::process_ifdef(Tokenizer *p_tokenizer) {
	const int line = p_tokenizer->get_line();

	String label = p_tokenizer->get_identifier();
	if (label.is_empty()) {
		set_error(RTR("Invalid macro name."), line);
		return;
	}

	p_tokenizer->skip_whitespace();
	if (!is_char_end(p_tokenizer->peek())) {
		set_error(RTR("Invalid ifdef."), line);
		return;
	}
	p_tokenizer->advance('\n');

	bool success = state->defines.has(label);
	start_branch_condition(p_tokenizer, success);
}

void ShaderPreprocessor::process_ifndef(Tokenizer *p_tokenizer) {
	const int line = p_tokenizer->get_line();

	String label = p_tokenizer->get_identifier();
	if (label.is_empty()) {
		set_error(RTR("Invalid macro name."), line);
		return;
	}

	p_tokenizer->skip_whitespace();
	if (!is_char_end(p_tokenizer->peek())) {
		set_error(RTR("Invalid ifndef."), line);
		return;
	}
	p_tokenizer->advance('\n');

	bool success = !state->defines.has(label);
	start_branch_condition(p_tokenizer, success);
}

void ShaderPreprocessor::process_include(Tokenizer *p_tokenizer) {
	const int line = p_tokenizer->get_line();

	p_tokenizer->advance('"');
	String path = tokens_to_string(p_tokenizer->advance('"'));
	for (int i = 0; i < path.length(); i++) {
		if (path[i] == '\n') {
			break; //stop parsing
		}
		if (path[i] == CURSOR) {
			state->completion_type = COMPLETION_TYPE_INCLUDE_PATH;
			break;
		}
	}
	path = path.substr(0, path.length() - 1);
	p_tokenizer->skip_whitespace();

	if (path.is_empty() || !is_char_end(p_tokenizer->peek())) {
		set_error(RTR("Invalid path."), line);
		return;
	}

	Ref<Resource> res = ResourceLoader::load(path);
	if (res.is_null()) {
		set_error(RTR("Shader include load failed. Does the shader include exist? Is there a cyclic dependency?"), line);
		return;
	}

	Ref<ShaderInclude> shader_inc = res;
	if (shader_inc.is_null()) {
		set_error(RTR("Shader include resource type is wrong."), line);
		return;
	}

	String included = shader_inc->get_code();
	if (!included.is_empty()) {
		uint64_t code_hash = included.hash64();
		if (state->cyclic_include_hashes.find(code_hash)) {
			set_error(RTR("Cyclic include found."), line);
			return;
		}
	}

	state->shader_includes.insert(shader_inc);

	const String real_path = shader_inc->get_path();
	if (state->includes.has(real_path)) {
		// Already included, skip.
		// This is a valid check because 2 separate include paths could use some
		// of the same shared functions from a common shader include.
		return;
	}

	// Mark as included.
	state->includes.insert(real_path);

	state->include_depth++;
	if (state->include_depth > 25) {
		set_error(RTR("Shader max include depth exceeded."), line);
		return;
	}

	String old_include = state->current_include;
	state->current_include = real_path;
	ShaderPreprocessor processor;

	int prev_condition_depth = state->condition_depth;
	state->condition_depth = 0;

	FilePosition fp;
	fp.file = state->current_include;
	fp.line = line;
	state->include_positions.push_back(fp);

	String result;
	processor.preprocess(state, included, result);
	add_to_output("@@>" + real_path + "\n"); // Add token for enter include path
	add_to_output(result);
	add_to_output("\n@@<\n"); // Add token for exit include path

	// Reset to last include if there are no errors. We want to use this as context.
	if (state->error.is_empty()) {
		state->current_include = old_include;
		state->include_positions.pop_back();
	} else {
		return;
	}

	state->include_depth--;
	state->condition_depth = prev_condition_depth;
}

void ShaderPreprocessor::process_pragma(Tokenizer *p_tokenizer) {
	const int line = p_tokenizer->get_line();

	bool is_cursor;
	const String label = p_tokenizer->get_identifier(&is_cursor);
	if (is_cursor) {
		state->completion_type = COMPLETION_TYPE_PRAGMA;
	}

	if (label.is_empty()) {
		set_error(RTR("Invalid pragma directive."), line);
		return;
	}

	// Rxplicitly handle pragma values here.
	// If more pragma options are created, then refactor into a more defined structure.
	if (label == "disable_preprocessor") {
		state->disabled = true;
	} else {
		set_error(RTR("Invalid pragma directive."), line);
		return;
	}

	p_tokenizer->advance('\n');
}

void ShaderPreprocessor::process_undef(Tokenizer *p_tokenizer) {
	const int line = p_tokenizer->get_line();
	const String label = p_tokenizer->get_identifier();
	if (label.is_empty() || !state->defines.has(label)) {
		set_error(RTR("Invalid name."), line);
		return;
	}

	p_tokenizer->skip_whitespace();
	if (!is_char_end(p_tokenizer->peek())) {
		set_error(RTR("Invalid undef."), line);
		return;
	}

	memdelete(state->defines[label]);
	state->defines.erase(label);
}

void ShaderPreprocessor::start_branch_condition(Tokenizer *p_tokenizer, bool p_success) {
	state->condition_depth++;

	if (p_success) {
		state->skip_stack_else.push_back(true);
	} else {
		SkippedCondition *cond = memnew(SkippedCondition());
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

void ShaderPreprocessor::expand_output_macros(int p_start, int p_line_number) {
	String line = vector_to_string(output, p_start, output.size());

	Error error = expand_macros(line, p_line_number - 1, line); // We are already on next line, so -1.
	if (error != OK) {
		return;
	}

	output.resize(p_start);

	add_to_output(line);
}

Error ShaderPreprocessor::expand_macros(const String &p_string, int p_line, String &r_expanded) {
	Vector<Pair<String, Define *>> active_defines;
	active_defines.resize(state->defines.size());
	int index = 0;
	for (const RBMap<String, Define *>::Element *E = state->defines.front(); E; E = E->next()) {
		active_defines.set(index++, Pair<String, Define *>(E->key(), E->get()));
	}

	return expand_macros(p_string, p_line, active_defines, r_expanded);
}

Error ShaderPreprocessor::expand_macros(const String &p_string, int p_line, Vector<Pair<String, Define *>> p_defines, String &r_expanded) {
	r_expanded = p_string;
	// When expanding macros we must only evaluate them once.
	// Later we continue expanding but with the already
	// evaluated macros removed.
	for (int i = 0; i < p_defines.size(); i++) {
		Pair<String, Define *> define_pair = p_defines[i];

		Error error = expand_macros_once(r_expanded, p_line, define_pair, r_expanded);
		if (error != OK) {
			return error;
		}

		// Remove expanded macro and recursively replace remaining.
		p_defines.remove_at(i);
		return expand_macros(r_expanded, p_line, p_defines, r_expanded);
	}

	return OK;
}

Error ShaderPreprocessor::expand_macros_once(const String &p_line, int p_line_number, Pair<String, Define *> p_define_pair, String &r_expanded) {
	String result = p_line;

	const String &key = p_define_pair.first;
	const Define *define = p_define_pair.second;

	int index_start = 0;
	int index = 0;
	while (find_match(result, key, index, index_start)) {
		String body = define->body;
		if (define->arguments.size() > 0) {
			// Complex macro with arguments.
			int args_start = index + key.length();
			int args_end = p_line.find(")", args_start);
			if (args_start == -1 || args_end == -1) {
				set_error(RTR("Missing macro argument parenthesis."), p_line_number);
				return FAILED;
			}

			String values = result.substr(args_start + 1, args_end - (args_start + 1));
			Vector<String> args = values.split(",");
			if (args.size() != define->arguments.size()) {
				set_error(RTR("Invalid macro argument count."), p_line_number);
				return FAILED;
			}

			// Insert macro arguments into the body.
			for (int i = 0; i < args.size(); i++) {
				String arg_name = define->arguments[i];
				int arg_index_start = 0;
				int arg_index = 0;
				while (find_match(body, arg_name, arg_index, arg_index_start)) {
					body = body.substr(0, arg_index) + args[i] + body.substr(arg_index + arg_name.length(), body.length() - (arg_index + arg_name.length()));
					// Manually reset arg_index_start to where the arg value of the define finishes.
					// This ensures we don't skip the other args of this macro in the string.
					arg_index_start = arg_index + args[i].length() + 1;
				}
			}

			result = result.substr(0, index) + " " + body + " " + result.substr(args_end + 1, result.length());
		} else {
			result = result.substr(0, index) + body + result.substr(index + key.length(), result.length() - (index + key.length()));
			// Manually reset index_start to where the body value of the define finishes.
			// This ensures we don't skip another instance of this macro in the string.
			index_start = index + body.length() + 1;
			break;
		}
	}
	r_expanded = result;
	return OK;
}

bool ShaderPreprocessor::find_match(const String &p_string, const String &p_value, int &r_index, int &r_index_start) {
	// Looks for value in string and then determines if the boundaries
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

		// Return and shift index start automatically for next call.
		r_index_start = r_index + p_value.length() + 1;
		return true;
	}

	return false;
}

String ShaderPreprocessor::next_directive(Tokenizer *p_tokenizer, const Vector<String> &p_directives) {
	const int line = p_tokenizer->get_line();
	int nesting = 0;

	while (true) {
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

	set_error(RTR("Can't find matching branch directive."), line);
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
		FilePosition fp;
		fp.line = p_line + 1;
		state->include_positions.push_back(fp);
	}
}

ShaderPreprocessor::Define *ShaderPreprocessor::create_define(const String &p_body) {
	ShaderPreprocessor::Define *define = memnew(Define);
	define->body = p_body;
	return define;
}

void ShaderPreprocessor::clear() {
	if (state_owner && state != nullptr) {
		for (const RBMap<String, Define *>::Element *E = state->defines.front(); E; E = E->next()) {
			memdelete(E->get());
		}

		for (const RBMap<String, Vector<SkippedCondition *>>::Element *E = state->skipped_conditions.front(); E; E = E->next()) {
			for (SkippedCondition *condition : E->get()) {
				memdelete(condition);
			}
		}

		memdelete(state);
	}
	state_owner = false;
	state = nullptr;
}

Error ShaderPreprocessor::preprocess(State *p_state, const String &p_code, String &r_result) {
	clear();

	output.clear();

	state = p_state;

	CommentRemover remover(p_code);
	String stripped = remover.strip();
	String error = remover.get_error();
	if (!error.is_empty()) {
		set_error(error, remover.get_error_line());
		return FAILED;
	}

	// Track code hashes to prevent cyclic include.
	uint64_t code_hash = p_code.hash64();
	state->cyclic_include_hashes.push_back(code_hash);

	Tokenizer p_tokenizer(stripped);
	int last_size = 0;
	bool has_symbols_before_directive = false;

	while (true) {
		const Token &t = p_tokenizer.get_token();

		if (t.text == 0) {
			break;
		}

		if (state->disabled) {
			// Preprocessor was disabled.
			// Read the rest of the file into the output.
			output.push_back(t.text);
			continue;
		} else {
			// Add autogenerated tokens.
			Vector<Token> generated;
			p_tokenizer.get_and_clear_generated(&generated);
			for (int i = 0; i < generated.size(); i++) {
				output.push_back(generated[i].text);
			}
		}

		if (t.text == '#') {
			if (has_symbols_before_directive) {
				set_error(RTR("Invalid symbols placed before directive."), p_tokenizer.get_line());
				state->cyclic_include_hashes.erase(code_hash); // Remove this hash.
				return FAILED;
			}
			process_directive(&p_tokenizer);
		} else {
			if (is_char_end(t.text)) {
				expand_output_macros(last_size, p_tokenizer.get_line());
				last_size = output.size();
				has_symbols_before_directive = false;
			} else if (!is_char_space(t.text)) {
				has_symbols_before_directive = true;
			}
			output.push_back(t.text);
		}

		if (!state->error.is_empty()) {
			state->cyclic_include_hashes.erase(code_hash); // Remove this hash.
			return FAILED;
		}
	}
	state->cyclic_include_hashes.erase(code_hash); // Remove this hash.

	if (!state->disabled) {
		if (state->condition_depth != 0) {
			set_error(RTR("Unmatched conditional statement."), p_tokenizer.line);
			return FAILED;
		}

		expand_output_macros(last_size, p_tokenizer.get_line());
	}

	r_result = vector_to_string(output);

	return OK;
}

Error ShaderPreprocessor::preprocess(const String &p_code, String &r_result, String *r_error_text, List<FilePosition> *r_error_position, HashSet<Ref<ShaderInclude>> *r_includes, List<ScriptLanguage::CodeCompletionOption> *r_completion_options, IncludeCompletionFunction p_include_completion_func) {
	State pp_state;
	Error err = preprocess(&pp_state, p_code, r_result);
	if (err != OK) {
		if (r_error_text) {
			*r_error_text = pp_state.error;
		}
		if (r_error_position) {
			*r_error_position = pp_state.include_positions;
		}
	}
	if (r_includes) {
		*r_includes = pp_state.shader_includes;
	}

	if (r_completion_options) {
		switch (pp_state.completion_type) {
			case COMPLETION_TYPE_DIRECTIVE: {
				List<String> options;
				get_keyword_list(&options, true);

				for (const String &E : options) {
					ScriptLanguage::CodeCompletionOption option(E, ScriptLanguage::CODE_COMPLETION_KIND_PLAIN_TEXT);
					r_completion_options->push_back(option);
				}

			} break;
			case COMPLETION_TYPE_PRAGMA: {
				List<String> options;

				ShaderPreprocessor::get_pragma_list(&options);

				for (const String &E : options) {
					ScriptLanguage::CodeCompletionOption option(E, ScriptLanguage::CODE_COMPLETION_KIND_PLAIN_TEXT);
					r_completion_options->push_back(option);
				}

			} break;
			case COMPLETION_TYPE_INCLUDE_PATH: {
				if (p_include_completion_func && r_completion_options) {
					p_include_completion_func(r_completion_options);
				}

			} break;
			default: {
			}
		}
	}
	return err;
}

void ShaderPreprocessor::get_keyword_list(List<String> *r_keywords, bool p_include_shader_keywords) {
	r_keywords->push_back("define");
	if (p_include_shader_keywords) {
		r_keywords->push_back("else");
	}
	r_keywords->push_back("endif");
	if (p_include_shader_keywords) {
		r_keywords->push_back("if");
	}
	r_keywords->push_back("ifdef");
	r_keywords->push_back("ifndef");
	r_keywords->push_back("include");
	r_keywords->push_back("pragma");
	r_keywords->push_back("undef");
}

void ShaderPreprocessor::get_pragma_list(List<String> *r_pragmas) {
	r_pragmas->push_back("disable_preprocessor");
}

ShaderPreprocessor::ShaderPreprocessor() {
}

ShaderPreprocessor::~ShaderPreprocessor() {
	clear();
}

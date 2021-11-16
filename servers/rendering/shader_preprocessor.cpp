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
#include "modules/regex/regex.h"

static bool is_whitespace(CharType c) {
	return (c == ' ') || (c == '\t');
}

static String vector_to_string(const Vector<CharType> &v, int start = 0, int end = -1) {
	const int stop = (end == -1) ? v.size() : end;
	const int count = stop - start;

	String result;
	result.resize(count + 1);
	for (int i = 0; i < count; i++) {
		result[i] = v[start + i];
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

static String tokens_to_string(const Vector<PPToken> &tokens) {
	Vector<CharType> result;
	for (int i = 0; i < tokens.size(); i++) {
		result.push_back(tokens[i].text);
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

	bool advance(CharType what) {
		while (index < code.size()) {
			CharType c = code[index++];

			if (c == '\n') {
				line++;
				stripped.push_back('\n');
			}

			if (c == what) {
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
class PreproprocessorTokenizer {
public:
	String code;
	int line;
	int index;
	int SIZE;
	Vector<PPToken> generated;

private:
	void add_generated(const PPToken &t) {
		generated.push_back(t);
	}

	CharType next() {
		if (index < SIZE) {
			return code[index++];
		}
		return 0;
	}

public:
	PreproprocessorTokenizer(String p_code) {
		code = p_code;
		line = 0;
		index = 0;
		SIZE = code.size();
	}

	int get_line() {
		return line;
	}

	int get_index() {
		return index;
	}

	void get_and_clear_generated(Vector<PPToken> *out) {
		for (int i = 0; i < generated.size(); i++) {
			out->push_back(generated[i]);
		}
		generated.clear();
	}

	void backtrack(CharType what) {
		while (index >= 0) {
			CharType c = code[index];
			if (c == what) {
				break;
			}
			index--;
		}
	}

	CharType peek() {
		if (index < SIZE) {
			return code[index];
		}
		return 0;
	}

	Vector<PPToken> advance(CharType what) {
		Vector<PPToken> tokens;

		while (index < SIZE) {
			CharType c = code[index++];

			tokens.push_back(PPToken(c, line));

			if (c == '\n') {
				add_generated(PPToken('\n', line));
				line++;
			}

			if (c == what) {
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
		while (index < SIZE) {
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

ShaderPreprocessor::ShaderPreprocessor(const String &p_code) {
	code = p_code;
	state = NULL;
	state_owner = false;
}

String ShaderPreprocessor::preprocess(PreprocessorState *p_state) {
	free_state();

	output.clear();

	state = p_state;
	if (state == NULL) {
		state = create_state();
		state_owner = true;
	}

	CommentRemover remover(code);
	String stripped = remover.strip();
	String error = remover.get_error();
	if (!error.is_empty()) {
		set_error(error, remover.get_error_line());
		return "<error>";
	}

	PreproprocessorTokenizer tokenizer(stripped);
	int last_size = 0;

	while (1) {
		const PPToken &t = tokenizer.get_token();

		//Add autogenerated tokens
		Vector<PPToken> generated;
		tokenizer.get_and_clear_generated(&generated);
		for (int i = 0; i < generated.size(); i++) {
			output.push_back(generated[i].text);
		}

		if (t.text == 0) {
			break;
		}

		if (t.text == '#') { //TODO check if at the beginning of line
			process_directive(&tokenizer);
		} else {
			if (t.text == '\n') {
				expand_output_macros(last_size, tokenizer.get_line());
				last_size = output.size();
			}
			output.push_back(t.text);
		}

		if (!state->error.is_empty()) {
			return "<error>";
		}
	}

	expand_output_macros(last_size, tokenizer.get_line());

	String result = vector_to_string(output);

	//print_line("--------------------------------------");
	//print_line("Preprocessed shader:");
	//print_line(result);
	//print_line("--------------------------------------");

	return result;
}

void ShaderPreprocessor::process_directive(PreproprocessorTokenizer *tokenizer) {
	String directive = tokenizer->get_identifier();

	if (directive == "if") {
		process_if(tokenizer);
	} else if (directive == "ifdef") {
		process_ifdef(tokenizer);
	} else if (directive == "ifndef") {
		process_ifndef(tokenizer);
	} else if (directive == "else") {
		process_else(tokenizer);
	} else if (directive == "endif") {
		process_endif(tokenizer);
	} else if (directive == "define") {
		process_define(tokenizer);
	} else if (directive == "undef") {
		process_undef(tokenizer);
	} else if (directive == "include") {
		process_include(tokenizer);
	} else {
		set_error("Unknown directive", tokenizer->get_line());
	}
}

void ShaderPreprocessor::process_if(PreproprocessorTokenizer *tokenizer) {
	int line = tokenizer->get_line();

	String body = tokens_to_string(tokenizer->advance('\n')).strip_edges();
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

	Variant v = expression.execute(Array(), NULL, false);
	if (v.get_type() == Variant::NIL) {
		set_error("Condition evaluation error", line);
		return;
	}

	bool success = v.booleanize();
	start_branch_condition(tokenizer, success);
}

void ShaderPreprocessor::process_ifdef(PreproprocessorTokenizer *tokenizer) {
	const int line = tokenizer->get_line();

	String label = tokenizer->get_identifier();
	if (label.is_empty()) {
		set_error("Invalid macro name", line);
		return;
	}

	tokenizer->skip_whitespace();
	if (tokenizer->peek() != '\n') {
		set_error("Invalid ifdef", line);
		return;
	}
	tokenizer->advance('\n');

	bool success = state->defines.has(label);
	start_branch_condition(tokenizer, success);
}

void ShaderPreprocessor::process_ifndef(PreproprocessorTokenizer *tokenizer) {
	const int line = tokenizer->get_line();

	String label = tokenizer->get_identifier();
	if (label.is_empty()) {
		set_error("Invalid macro name", line);
		return;
	}

	tokenizer->skip_whitespace();
	if (tokenizer->peek() != '\n') {
		set_error("Invalid ifndef", line);
		return;
	}
	tokenizer->advance('\n');

	bool success = !state->defines.has(label);
	start_branch_condition(tokenizer, success);
}

void ShaderPreprocessor::start_branch_condition(PreproprocessorTokenizer *tokenizer, bool success) {
	state->condition_depth++;

	if (success) {
		state->skip_stack_else.push_back(true);
	} else {
		SkippedPreprocessorCondition *cond = new SkippedPreprocessorCondition();
		cond->start_line = tokenizer->get_line();
		state->skipped_conditions[state->current_include].push_back(cond);

		Vector<String> ends;
		ends.push_back("else");
		ends.push_back("endif");
		if (next_directive(tokenizer, ends) == "else") {
			state->skip_stack_else.push_back(false);
		} else {
			state->skip_stack_else.push_back(true);
		}
	}
}

void ShaderPreprocessor::process_else(PreproprocessorTokenizer *tokenizer) {
	if (state->skip_stack_else.is_empty()) {
		set_error("Unmatched else", tokenizer->get_line());
		return;
	}
	tokenizer->advance('\n');

	bool skip = state->skip_stack_else[state->skip_stack_else.size() - 1];
	state->skip_stack_else.remove(state->skip_stack_else.size() - 1);

	auto vec = state->skipped_conditions[state->current_include];
	int index = vec.size() - 1;
	if (index >= 0) {
		SkippedPreprocessorCondition *cond = vec[index];
		if (cond->end_line == -1) {
			cond->end_line = tokenizer->get_line();
		}
	}

	if (skip) {
		Vector<String> ends;
		ends.push_back("endif");
		next_directive(tokenizer, ends);
	}
}

void ShaderPreprocessor::process_endif(PreproprocessorTokenizer *tokenizer) {
	state->condition_depth--;
	if (state->condition_depth < 0) {
		set_error("Unmatched endif", tokenizer->get_line());
		return;
	}

	auto vec = state->skipped_conditions[state->current_include];
	int index = vec.size() - 1;
	if (index >= 0) {
		SkippedPreprocessorCondition *cond = vec[index];
		if (cond->end_line == -1) {
			cond->end_line = tokenizer->get_line();
		}
	}

	tokenizer->advance('\n');
}

void ShaderPreprocessor::process_define(PreproprocessorTokenizer *tokenizer) {
	const int line = tokenizer->get_line();

	String label = tokenizer->get_identifier();
	if (label.is_empty()) {
		set_error("Invalid macro name", line);
		return;
	}

	if (state->defines.has(label)) {
		set_error("Macro redefinition", line);
		return;
	}

	if (tokenizer->peek() == '(') {
		//Macro has arguments
		tokenizer->get_token();

		Vector<String> args;
		while (1) {
			String name = tokenizer->get_identifier();
			if (name.is_empty()) {
				set_error("Invalid argument name", line);
				return;
			}
			args.push_back(name);

			tokenizer->skip_whitespace();
			CharType next = tokenizer->get_token().text;
			if (next == ')') {
				break;
			} else if (next != ',') {
				set_error("Expected a comma in the macro argument list", line);
				return;
			}
		}

		PreprocessorDefine *define = memnew(PreprocessorDefine);
		define->arguments = args;
		define->body = tokens_to_string(tokenizer->advance('\n')).strip_edges();
		state->defines[label] = define;
	} else {
		//Simple substitution macro
		PreprocessorDefine *define = memnew(PreprocessorDefine);
		define->body = tokens_to_string(tokenizer->advance('\n')).strip_edges();
		state->defines[label] = define;
	}
}

void ShaderPreprocessor::process_undef(PreproprocessorTokenizer *tokenizer) {
	const int line = tokenizer->get_line();
	const String label = tokenizer->get_identifier();
	if (label.is_empty()) {
		set_error("Invalid name", line);
		return;
	}

	tokenizer->skip_whitespace();
	if (tokenizer->peek() != '\n') {
		set_error("Invalid undef", line);
		return;
	}

	state->defines.erase(label);
}

void ShaderPreprocessor::process_include(PreproprocessorTokenizer *tokenizer) {
	const int line = tokenizer->get_line();

	tokenizer->advance('"');
	String path = tokens_to_string(tokenizer->advance('"'));
	path.erase(path.length() - 1, 1);
	tokenizer->skip_whitespace();

	if (path.is_empty() /* || tokenizer->peek() != '\n'*/) {
		set_error("Invalid path", line);
		return;
	}

	RES res = ResourceLoader::load(path);
	if (res.is_null()) {
		set_error("Shader include load failed", line);
		return;
	}

	Shader *shader = Object::cast_to<Shader>(*res);
	if (shader == NULL) {
		set_error("Shader include resource type is wrong", line);
		return;
	}

	String included = shader->get_code();
	if (included.is_empty()) {
		set_error("Shader include not found", line);
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
		return;
	}

	//Mark as included
	state->includes.insert(real_path);

	state->include_depth++;
	if (state->include_depth > 25) {
		set_error("Shader max include depth exceeded", line);
		return;
	}

	//Remove "shader_type xyz;" prefix from included files
	//included = included.substr(type_end + 1, included.length());
	included = included.replace("shader_type ", "//shader_type ");

	String old_include = state->current_include;
	state->current_include = real_path;
	ShaderPreprocessor processor(included);
	String result = processor.preprocess(state); // .replace("\n", " "); //To preserve line numbers cram everything into a single line
	add_to_output(result);
	state->current_include = old_include;

	if (!state->error.is_empty() && state_owner) {
		//This is the root file, so force the line number to match this instead of the included file
		// state->error_line = line + 1;
	}

	state->include_depth--;
}

void ShaderPreprocessor::expand_output_macros(int start, int line_number) {
	String line = vector_to_string(output, start, output.size());

	line = expand_macros(line, line_number - 1); //We are already on next line, so -1

	output.resize(start);

	add_to_output(line);
}

String ShaderPreprocessor::expand_macros(const String &p_string, int p_line) {
	String result = p_string;

	int expanded = 1;
	while (expanded) {
		result = expand_macros_once(result, p_line, &expanded);

		if (!state->error.is_empty()) {
			return "<error>";
		}
	}
	return result;
}

String ShaderPreprocessor::expand_macros_once(const String &p_line, int line_number, int *p_expanded) {
	String result = p_line;
	*p_expanded = 0;

	//TODO Could use something better than regular expressions for this...
	for (const Map<String, PreprocessorDefine *>::Element *E = state->defines.front(); E; E = E->next()) {
		const String &key = E->key();
		const PreprocessorDefine *define = E->get();

		//Match against word boundaries
		RegEx label("\\b" + key + "\\b");

		Ref<RegExMatch> match = label.search(result);
		if (match.is_valid()) {
			//Substitute all macro content
			if (define->arguments.size() > 0) {
				//Complex macro with arguments
				int args_start = match->get_end(0);
				int args_end = p_line.find(")", args_start);
				if (args_start == -1 || args_end == -1) {
					*p_expanded = 0;
					set_error("Missing macro argument parenthesis", line_number);
					return "<error>";
				}

				String values = result.substr(args_start + 1, args_end - (args_start + 1));
				Vector<String> args = values.split(",");
				if (args.size() != define->arguments.size()) {
					*p_expanded = 0;
					set_error("Invalid macro argument count", line_number);
					return "<error>";
				}
				//Insert macro arguments into the body
				String body = define->body;
				for (int i = 0; i < args.size(); i++) {
					RegEx value("\\b" + define->arguments[i] + "\\b");
					body = value.sub(body, args[i], true);
				}

				result = result.substr(0, match->get_start(0)) + " " + body + " " + result.substr(args_end + 1, result.length());
			} else {
				//Simple substitution macro
				result = label.sub(result, define->body, true);
			}

			*p_expanded = 1;
		}
	}
	return result;
}

String ShaderPreprocessor::next_directive(PreproprocessorTokenizer *tokenizer, const Vector<String> &directives) {
	const int line = tokenizer->get_line();
	int nesting = 0;

	while (1) {
		tokenizer->advance('#');

		String id = tokenizer->peek_identifier();
		if (id.is_empty()) {
			break;
		}

		if (nesting == 0) {
			for (int i = 0; i < directives.size(); i++) {
				if (directives[i] == id) {
					tokenizer->backtrack('#');
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

void ShaderPreprocessor::set_error(const String &error, int line) {
	if (state->error.is_empty()) {
		state->error = error;
		state->error_line = line + 1;
	}
}

void ShaderPreprocessor::free_state() {
	if (state_owner && state != NULL) {
		for (const Map<String, PreprocessorDefine *>::Element *E = state->defines.front(); E; E = E->next()) {
			memdelete(E->get());
		}
		memdelete(state);
	}
	state_owner = false;
	state = NULL;
}

PreprocessorDefine *create_define(const String &body) {
	PreprocessorDefine *define = memnew(PreprocessorDefine);
	define->body = body;
	return define;
}

PreprocessorState *ShaderPreprocessor::create_state() {
	PreprocessorState *new_state = memnew(PreprocessorState);
	new_state->condition_depth = 0;
	new_state->include_depth = 0;
	new_state->error = "";
	new_state->error_line = -1;

	OS *os = OS::get_singleton();

	String platform = os->get_name().replace(" ", "_").to_upper();
	new_state->defines[platform] = create_define("true");

	Engine *engine = Engine::get_singleton();
	new_state->defines["EDITOR"] = create_define(engine->is_editor_hint() ? "true" : "false");

	return new_state;
}

void ShaderPreprocessor::get_keyword_list(List<String> *keywords) {
	keywords->push_back("include");
	keywords->push_back("define");
	keywords->push_back("undef");
	//keywords->push_back("if");  //Already a keyword
	keywords->push_back("ifdef");
	keywords->push_back("ifndef");
	//keywords->push_back("else"); //Already a keyword
	keywords->push_back("endif");
}

void ShaderPreprocessor::refresh_shader_dependencies(Ref<Shader> shader) {
	ShaderDependencyGraph shaderDependencies;
	shaderDependencies.populate(shader);
	shaderDependencies.update_shaders();
}

ShaderDependencyGraph::~ShaderDependencyGraph() {
	for (auto E = nodes.front(); E; E = E->next()) {
		delete E->get();
	}
}

ShaderDependencyNode::ShaderDependencyNode(Ref<Shader> s) :
		shader(s) {}

ShaderDependencyNode::ShaderDependencyNode(String code) :
		code(code) {}

ShaderDependencyNode::ShaderDependencyNode(String path, String code) :
		code(code), path(path) {}

int ShaderDependencyNode::GetContext(int line, ShaderDependencyNode **context) {
	int include_offset = 0;
	for (ShaderDependencyNode *include : dependencies) {
		if (line < include->line + include_offset) {
			// line shifted by offset is sufficient. break.
			break;
		} else if (line >= include->line + include_offset && line <= (include->line + include_offset + include->get_line_count())) {
			if (include->dependencies.is_empty()) {
				*context = include;
				return line - include->line;
			} else {
				return include->GetContext(line - include->line + 1, context); // plus 1 to be fully inclusive in the skip
			}
		}

		include_offset += include->get_line_count();
	}

	*context = this;
	return line - include_offset;
}

ShaderDependencyNode::~ShaderDependencyNode() {
	for (auto E = dependencies.front(); E; E = E->next()) {
		delete E->get();
	}
}

bool operator<(ShaderDependencyNode left, ShaderDependencyNode right) {
	return left.shader < right.shader;
}

bool operator==(ShaderDependencyNode left, ShaderDependencyNode right) {
	return left.shader == right.shader;
}

void ShaderDependencyGraph::populate(Ref<Shader> shader) {
	cyclic_dep_tracker.clear();
	nodes.clear(); // TODO actually delete these if we're refreshing
	visited_shaders.clear();

	ShaderDependencyNode *node = new ShaderDependencyNode(shader);
	nodes.insert(node);
	populate(node);
}

void ShaderDependencyGraph::populate(String code) {
	cyclic_dep_tracker.clear();
	nodes.clear(); // TODO actually delete these if we're refreshing
	visited_shaders.clear();

	ShaderDependencyNode *node = new ShaderDependencyNode(code);
	nodes.insert(node);
	populate(node);
}

void ShaderDependencyGraph::populate(String path, String code) {
	cyclic_dep_tracker.clear();
	nodes.clear(); // TODO actually delete these if we're refreshing
	visited_shaders.clear();

	ShaderDependencyNode *node = new ShaderDependencyNode(path, code);
	nodes.insert(node);
	populate(node);
}

void ShaderDependencyGraph::populate(ShaderDependencyNode *node) {
	if (!node->shader.is_null()) {
		ERR_FAIL_COND_MSG(cyclic_dep_tracker.find(node), vformat("Shader %s contains a cyclic import. Skipping...", node->shader->get_path()));
	}

	cyclic_dep_tracker.push_back(node);
	String code;
	if (!node->shader.is_null())
		code = CommentRemover(node->shader->get_code()).strip(); // Build dependency graph starting from edited shader. Strip comments
	else
		code = CommentRemover(node->code).strip();

	PreproprocessorTokenizer tokenizer(code);
	while (1) {
		if (!tokenizer.advance('#').is_empty()) {
			String directive = tokenizer.get_identifier();
			if (directive == "include") {
				if (!tokenizer.advance('"').is_empty()) {
					String path = tokens_to_string(tokenizer.advance('"'));
					path.erase(path.length() - 1, 1);
					tokenizer.skip_whitespace();
					if (!path.is_empty()) {
						RES res = ResourceLoader::load(path);
						if (!res.is_null()) {
							Ref<Shader> shader_reference = Object::cast_to<Shader>(*res);
							if (!shader_reference.is_null()) {
								// skip this shader if we've picked it up as a dependency already
								String shader_path = shader_reference->get_path();
								if (!visited_shaders.find(shader_path)) {
									String included_code = shader_reference->get_code();
									ShaderDependencyNode *new_node = new ShaderDependencyNode(shader_path, included_code);
									new_node->line = tokenizer.get_line() + 1;
									Vector<String> shader = included_code.split("\n");
									new_node->line_count = shader.size();
									visited_shaders.push_back(shader_path);
									populate(new_node);
									node->dependencies.insert(new_node);
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

	cyclic_dep_tracker.erase(node);
}

Set<ShaderDependencyNode *>::Element *ShaderDependencyGraph::find(Ref<Shader> shader) {
	for (auto E = nodes.front(); E; E = E->next()) {
		if (E->get()->shader == shader) {
			return E;
		}
	}

	return nullptr;
}

Set<ShaderDependencyNode *>::Element *ShaderDependencyGraph::find(String path) {
	for (auto E = nodes.front(); E; E = E->next()) {
		if (E->get()->path == path) {
			return E;
		}
	}

	return nullptr;
}

void ShaderDependencyGraph::update_shaders() {
	for (auto E = nodes.front(); E; E = E->next()) {
		update_shaders(E->get());
	}
}

void ShaderDependencyGraph::update_shaders(ShaderDependencyNode *node) {
	for (auto E = node->dependencies.front(); E; E = E->next()) {
		update_shaders(E->get());
	}

	if (!node->shader.is_null()) {
		node->shader->set_code(node->shader->get_code());
	}
}

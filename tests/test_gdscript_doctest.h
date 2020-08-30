/*************************************************************************/
/*  test_gdscript_doctest.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEST_GDSCRIPT_DOCTEST_H
#define TEST_GDSCRIPT_DOCTEST_H

#include "modules/modules_enabled.gen.h"
#ifdef MODULE_GDSCRIPT_ENABLED

#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/string_builder.h"

#include "modules/gdscript/gdscript_analyzer.h"
#include "modules/gdscript/gdscript_compiler.h"
#include "modules/gdscript/gdscript_parser.h"
#include "modules/gdscript/gdscript_tokenizer.h"

#include "thirdparty/doctest/doctest.h"

namespace TestGDScript {

// NOTE: would be great to avoid the code duplication here by using test() declared in
// /modules/gdscript/tests/test_gdscript.h BUT:
// the underlying functions print_tokenizer, ... print directly to the logger and
// they require a command line argument.
// Ideally, that could be factored, so we can pass the path of a .gd script and get a String.
// Or at least pass the .gd script as an argument, as the OS singleton doesn't allow setting _cmdline,
// then I could register a print-handler to capture the output.
// (see /core/print_string.h, disable printing temporarily with _print_line_enabled and _print_error_enabled)
// For now: duplicate the code, remove the use of "text_editor/indent/size" in the tokenizer output
// so the output is consistent, stringbuild-ify. Check with:
// > diff -u modules/gdscript/tests/test_gdscript.cpp tests/test_gdscript_doctest.h

static String check_tokenizer(const String &p_code, const Vector<String> &p_lines) {
	StringBuilder result;
	GDScriptTokenizer tokenizer;
	tokenizer.set_source_code(p_code);

	int tab_size = 4;
	String tab = String(" ").repeat(tab_size);

	GDScriptTokenizer::Token current = tokenizer.scan();
	while (current.type != GDScriptTokenizer::Token::TK_EOF) {
		StringBuilder token;
		token += " --> "; // Padding for line number.

		for (int l = current.start_line; l <= current.end_line; l++) {
			result += vformat("%04d %s", l, p_lines[l - 1]).replace("\t", tab) + "\n";
		}

		{
			// Print carets to point at the token.
			StringBuilder pointer;
			pointer += "     "; // Padding for line number.
			int rightmost_column = current.rightmost_column;
			if (current.end_line > current.start_line) {
				rightmost_column--; // Don't point to the newline as a column.
			}
			for (int col = 1; col < rightmost_column; col++) {
				if (col < current.leftmost_column) {
					pointer += " ";
				} else {
					pointer += "^";
				}
			}
			result += pointer.as_string() + "\n";
		}

		token += current.get_name();

		if (current.type == GDScriptTokenizer::Token::ERROR || current.type == GDScriptTokenizer::Token::LITERAL || current.type == GDScriptTokenizer::Token::IDENTIFIER || current.type == GDScriptTokenizer::Token::ANNOTATION) {
			token += "(";
			token += Variant::get_type_name(current.literal.get_type());
			token += ") ";
			token += current.literal;
		}

		result += token.as_string();
		result += "\n";

		result += "-------------------------------------------------------\n";

		current = tokenizer.scan();
	}

	result += current.get_name(); // Should be EOF
	return result;
}

static String check_parser(const String &p_code, const String &p_script_path) {
	StringBuilder result;
	GDScriptParser parser;
	Error err = parser.parse(p_code, p_script_path, false);

	if (err != OK) {
		const List<GDScriptParser::ParserError> &errors = parser.get_errors();
		for (const List<GDScriptParser::ParserError>::Element *E = errors.front(); E != nullptr; E = E->next()) {
			const GDScriptParser::ParserError &error = E->get();
			result += vformat("%02d:%02d: %s", error.line, error.column, error.message) + "\n";
		}
	}

	GDScriptParser::TreePrinter printer;

	result += printer.get_tree_string(parser);
	return result;
}

/*
static String check_compiler(const String &p_code, const String &p_script_path, const Vector<String> &p_lines) {
	StringBuilder result;
	GDScriptParser parser;
	Error err = parser.parse(p_code, p_script_path, false);

	if (err != OK) {
		result += "Error in parser:\n";
		const List<GDScriptParser::ParserError> &errors = parser.get_errors();
		for (const List<GDScriptParser::ParserError>::Element *E = errors.front(); E != nullptr; E = E->next()) {
			const GDScriptParser::ParserError &error = E->get();
			result += vformat("%02d:%02d: %s", error.line, error.column, error.message) + "\n";
		}
		return result;
	}

	GDScriptAnalyzer analyzer(&parser);
	err = analyzer.analyze();

	if (err != OK) {
		result += "Error in analyzer:\n";
		const List<GDScriptParser::ParserError> &errors = parser.get_errors();
		for (const List<GDScriptParser::ParserError>::Element *E = errors.front(); E != nullptr; E = E->next()) {
			const GDScriptParser::ParserError &error = E->get();
			result += vformat("%02d:%02d: %s", error.line, error.column, error.message) + "\n";
		}
		return result;
	}

	GDScriptCompiler compiler;
	Ref<GDScript> script;
	script.instance();
	script->set_path(p_script_path);

	err = compiler.compile(&parser, script.ptr(), false);

	if (err) {
		result += "Error in compiler:\n";
		result += vformat("%02d:%02d: %s", compiler.get_error_line(), compiler.get_error_column(), compiler.get_error()) + "\n";
		return result;
	}

	for (const Map<StringName, GDScriptFunction *>::Element *E = script->get_member_functions().front(); E; E = E->next()) {
		const GDScriptFunction *func = E->value();

		String signature = "Disassembling " + func->get_name().operator String() + "(";
		for (int i = 0; i < func->get_argument_count(); i++) {
			if (i > 0) {
				signature += ", ";
			}
			signature += func->get_argument_name(i);
		}
		result += signature + ")" + "\n";

		func->disassemble(p_lines);
		result += "\n";
		result += "\n";
	}
	return result;
}
*/

static bool get_file_contents(String &p_code, String &p_script_path) {
	FileAccessRef fa = FileAccess::open(p_script_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(!fa, false, "Could not open file: " + p_script_path);

	Vector<uint8_t> buf;
	int flen = fa->get_len();
	buf.resize(fa->get_len() + 1);
	fa->get_buffer(buf.ptrw(), flen);
	buf.write[flen] = 0;

	p_code.parse_utf8((const char *)&buf[0]);
	return true;
}

static bool get_file_contents_and_lines(String &p_code, String &p_script_path, Vector<String> &p_lines) {
	if (!get_file_contents(p_code, p_script_path)) {
		return false;
	}

	int last = 0;
	for (int i = 0; i <= p_code.length(); i++) {
		if (p_code[i] == '\n' || p_code[i] == 0) {
			p_lines.push_back(p_code.substr(last, i - last));
			last = i + 1;
		}
	}
	return true;
}

TEST_CASE("[GDScript] File-based testing of tokenizer/parser/compiler") {
	// this assumes tests are run from the repository root:
	const String doctests_dir = "./tests/test_gdscript_doctest/";
	DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	REQUIRE(da);
	REQUIRE(da->change_dir(doctests_dir) == OK);
	// iterate through the directory and test all .gd files against the .gd.tokenized/parsed/compiled files:
	da->list_dir_begin();
	String fname = da->get_next();
	while (fname != String()) {
		if (fname.ends_with(".gd")) {
			SUBCASE(fname.ascii().ptr()) {
				String test_file = doctests_dir + fname;
				String code;
				Vector<String> lines;
				REQUIRE(get_file_contents_and_lines(code, test_file, lines));
				String test_file_parsed = test_file + ".parsed";
				String result_parsed;
				REQUIRE(get_file_contents(result_parsed, test_file_parsed));
				String test_file_tokenized = test_file + ".tokenized";
				String result_tokenized;
				REQUIRE(get_file_contents(result_tokenized, test_file_tokenized));
				String test_file_compiled = test_file + ".compiled";
				String result_compiled;
				REQUIRE(get_file_contents(result_compiled, test_file_compiled));

				CHECK_MESSAGE(
						check_tokenizer(code, lines).strip_edges() == result_tokenized.strip_edges(),
						fname + " was not tokenized as expected.");
				CHECK_MESSAGE(
						check_parser(code, test_file).strip_edges() == result_parsed.strip_edges(),
						fname + " was not parsed as expected.");
				// wait with enabling this, still crashes for some input:
				//CHECK_MESSAGE(
				//		check_compiler(code, test_file, lines).strip_edges() == result_compiled.strip_edges(),
				//		fname + " was not compiled as expected.");
			}
		}
		fname = da->get_next();
	}
	da->list_dir_end();
	memdelete(da);
}

} // namespace TestGDScript
#endif // MODULE_GDSCRIPT_ENABLED

#endif // TEST_GDSCRIPT_DOCTEST_H

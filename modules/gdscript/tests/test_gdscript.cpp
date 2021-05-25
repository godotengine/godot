/*************************************************************************/
/*  test_gdscript.cpp                                                    */
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

#include "test_gdscript.h"

#include "core/config/project_settings.h"
#include "core/io/file_access_pack.h"
#include "core/os/file_access.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/string/string_builder.h"
#include "scene/resources/packed_scene.h"

#include "modules/gdscript/gdscript_analyzer.h"
#include "modules/gdscript/gdscript_compiler.h"
#include "modules/gdscript/gdscript_parser.h"
#include "modules/gdscript/gdscript_tokenizer.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#endif

namespace GDScriptTests {

static void test_tokenizer(const String &p_code, const Vector<String> &p_lines) {
	GDScriptTokenizer tokenizer;
	tokenizer.set_source_code(p_code);

	int tab_size = 4;
#ifdef TOOLS_ENABLED
	if (EditorSettings::get_singleton()) {
		tab_size = EditorSettings::get_singleton()->get_setting("text_editor/indent/size");
	}
#endif // TOOLS_ENABLED
	String tab = String(" ").repeat(tab_size);

	GDScriptTokenizer::Token current = tokenizer.scan();
	while (current.type != GDScriptTokenizer::Token::TK_EOF) {
		StringBuilder token;
		token += " --> "; // Padding for line number.

		for (int l = current.start_line; l <= current.end_line && l <= p_lines.size(); l++) {
			print_line(vformat("%04d %s", l, p_lines[l - 1]).replace("\t", tab));
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
			print_line(pointer.as_string());
		}

		token += current.get_name();

		if (current.type == GDScriptTokenizer::Token::ERROR || current.type == GDScriptTokenizer::Token::LITERAL || current.type == GDScriptTokenizer::Token::IDENTIFIER || current.type == GDScriptTokenizer::Token::ANNOTATION) {
			token += "(";
			token += Variant::get_type_name(current.literal.get_type());
			token += ") ";
			token += current.literal;
		}

		print_line(token.as_string());

		print_line("-------------------------------------------------------");

		current = tokenizer.scan();
	}

	print_line(current.get_name()); // Should be EOF
}

static void test_parser(const String &p_code, const String &p_script_path, const Vector<String> &p_lines) {
	GDScriptParser parser;
	Error err = parser.parse(p_code, p_script_path, false);

	if (err != OK) {
		const List<GDScriptParser::ParserError> &errors = parser.get_errors();
		for (const List<GDScriptParser::ParserError>::Element *E = errors.front(); E != nullptr; E = E->next()) {
			const GDScriptParser::ParserError &error = E->get();
			print_line(vformat("%02d:%02d: %s", error.line, error.column, error.message));
		}
	}

	GDScriptAnalyzer analyzer(&parser);
	analyzer.analyze();

	if (err != OK) {
		const List<GDScriptParser::ParserError> &errors = parser.get_errors();
		for (const List<GDScriptParser::ParserError>::Element *E = errors.front(); E != nullptr; E = E->next()) {
			const GDScriptParser::ParserError &error = E->get();
			print_line(vformat("%02d:%02d: %s", error.line, error.column, error.message));
		}
	}

#ifdef TOOLS_ENABLED
	GDScriptParser::TreePrinter printer;
	printer.print_tree(parser);
#endif
}

static void test_compiler(const String &p_code, const String &p_script_path, const Vector<String> &p_lines) {
	GDScriptParser parser;
	Error err = parser.parse(p_code, p_script_path, false);

	if (err != OK) {
		print_line("Error in parser:");
		const List<GDScriptParser::ParserError> &errors = parser.get_errors();
		for (const List<GDScriptParser::ParserError>::Element *E = errors.front(); E != nullptr; E = E->next()) {
			const GDScriptParser::ParserError &error = E->get();
			print_line(vformat("%02d:%02d: %s", error.line, error.column, error.message));
		}
		return;
	}

	GDScriptAnalyzer analyzer(&parser);
	err = analyzer.analyze();

	if (err != OK) {
		print_line("Error in analyzer:");
		const List<GDScriptParser::ParserError> &errors = parser.get_errors();
		for (const List<GDScriptParser::ParserError>::Element *E = errors.front(); E != nullptr; E = E->next()) {
			const GDScriptParser::ParserError &error = E->get();
			print_line(vformat("%02d:%02d: %s", error.line, error.column, error.message));
		}
		return;
	}

	GDScriptCompiler compiler;
	Ref<GDScript> script;
	script.instance();
	script->set_path(p_script_path);

	err = compiler.compile(&parser, script.ptr(), false);

	if (err) {
		print_line("Error in compiler:");
		print_line(vformat("%02d:%02d: %s", compiler.get_error_line(), compiler.get_error_column(), compiler.get_error()));
		return;
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
		print_line(signature + ")");
#ifdef TOOLS_ENABLED
		func->disassemble(p_lines);
#endif
		print_line("");
		print_line("");
	}
}

void test(TestType p_type) {
	List<String> cmdlargs = OS::get_singleton()->get_cmdline_args();

	if (cmdlargs.is_empty()) {
		return;
	}

	String test = cmdlargs.back()->get();
	if (!test.ends_with(".gd")) {
		print_line("This test expects a path to a GDScript file as its last parameter. Got: " + test);
		return;
	}

	FileAccessRef fa = FileAccess::open(test, FileAccess::READ);
	ERR_FAIL_COND_MSG(!fa, "Could not open file: " + test);

	// Initialize the language for the test routine.
	init_language(fa->get_path_absolute().get_base_dir());

	Vector<uint8_t> buf;
	uint64_t flen = fa->get_length();
	buf.resize(flen + 1);
	fa->get_buffer(buf.ptrw(), flen);
	buf.write[flen] = 0;

	String code;
	code.parse_utf8((const char *)&buf[0]);

	Vector<String> lines;
	int last = 0;
	for (int i = 0; i <= code.length(); i++) {
		if (code[i] == '\n' || code[i] == 0) {
			lines.push_back(code.substr(last, i - last));
			last = i + 1;
		}
	}

	switch (p_type) {
		case TEST_TOKENIZER:
			test_tokenizer(code, lines);
			break;
		case TEST_PARSER:
			test_parser(code, test, lines);
			break;
		case TEST_COMPILER:
			test_compiler(code, test, lines);
			break;
		case TEST_BYTECODE:
			print_line("Not implemented.");
	}

	finish_language();
}
} // namespace GDScriptTests

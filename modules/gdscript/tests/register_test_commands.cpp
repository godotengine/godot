/**************************************************************************/
/*  register_test_commands.cpp                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "test_gdscript.h"

#include "tests/test_macros.h"

void test_tokenizer() {
	GDScriptTests::test(GDScriptTests::TestType::TEST_TOKENIZER);
}

void test_tokenizer_buffer() {
	GDScriptTests::test(GDScriptTests::TestType::TEST_TOKENIZER_BUFFER);
}

void test_parser() {
	GDScriptTests::test(GDScriptTests::TestType::TEST_PARSER);
}

void test_compiler() {
	GDScriptTests::test(GDScriptTests::TestType::TEST_COMPILER);
}

void test_bytecode() {
	GDScriptTests::test(GDScriptTests::TestType::TEST_BYTECODE);
}

REGISTER_TEST_COMMAND("gdscript-tokenizer", &test_tokenizer);
REGISTER_TEST_COMMAND("gdscript-tokenizer-buffer", &test_tokenizer_buffer);
REGISTER_TEST_COMMAND("gdscript-parser", &test_parser);
REGISTER_TEST_COMMAND("gdscript-compiler", &test_compiler);
REGISTER_TEST_COMMAND("gdscript-bytecode", &test_bytecode);

/*************************************************************************/
/*  test_main.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/list.h"
#include "core/os/main_loop.h"

#ifdef DEBUG_ENABLED

#include <thirdparty/doctest/doctest.h>

#include "test_astar.h"
#include "test_gdscript.h"
#include "test_gui.h"
#include "test_math.h"
#include "test_oa_hash_map.h"
#include "test_ordered_hash_map.h"
#include "test_physics.h"
#include "test_physics_2d.h"
#include "test_render.h"
#include "test_shader_lang.h"
#include "test_string.h"
#include "test_fbx_import.h"

const char **tests_get_names() {

	static const char *test_names[] = {
		"doctest",
		"math",
		"physics",
		"physics_2d",
		"render",
		"oa_hash_map",
		"gui",
		"shaderlang",
		"gd_tokenizer",
		"gd_parser",
		"gd_compiler",
		"gd_bytecode",
		"ordered_hash_map",
		"astar",
		NULL
	};

	return test_names;
}

int test_main(String p_test, const List<String> &p_args) {
	doctest::Context test_context;

	if (p_test == "doctest" || p_test == "string") {
		// tests should be ordered by name for humans
		test_context.setOption("order-by", "name");
		test_context.setOption("abort-after", 5);
		test_context.setOption("no-breaks", true);
	
		int status = test_context.run(); // run tests

		if(test_context.shouldExit()) // wait for testing to complete
			return status;

		return status;
	}

	if (p_test == "math") {

		return TestMath::test() == NULL;
	}

	if (p_test == "physics") {

		return TestPhysics::test() == NULL;
	}

	if (p_test == "physics_2d") {

		return TestPhysics2D::test() == NULL;
	}

	if (p_test == "render") {

		return TestRender::test() == NULL;
	}

	if (p_test == "oa_hash_map") {

		return TestOAHashMap::test() == NULL;
	}

#ifndef _3D_DISABLED
	if (p_test == "gui") {

		return TestGUI::test() == NULL;
	}
#endif

	if (p_test == "shaderlang") {

		return TestShaderLang::test() == NULL;
	}

	if (p_test == "gd_tokenizer") {

		return TestGDScript::test(TestGDScript::TEST_TOKENIZER) == NULL;
	}

	if (p_test == "gd_parser") {

		return TestGDScript::test(TestGDScript::TEST_PARSER) == NULL;
	}

	if (p_test == "gd_compiler") {

		return TestGDScript::test(TestGDScript::TEST_COMPILER) == NULL;
	}

	if (p_test == "gd_bytecode") {

		return TestGDScript::test(TestGDScript::TEST_BYTECODE) == NULL;
	}

	if (p_test == "ordered_hash_map") {

		return TestOrderedHashMap::test() == NULL;
	}

	if (p_test == "astar") {

		return TestAStar::test() == NULL;
	}

	print_line("Unknown test: " + p_test);
	return 1;
}

#else

const char **tests_get_names() {

	static const char *test_names[] = {
		NULL
	};

	return test_names;
}

int test_main(String p_test, const List<String> &p_args) {

	return 0;
}

#endif

/*************************************************************************/
/*  test_main.cpp                                                        */
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

#include "test_main.h"

#include "core/list.h"

#include "test_astar.h"
#include "test_basis.h"
#include "test_class_db.h"
#include "test_color.h"
#include "test_command_queue.h"
#include "test_expression.h"
#include "test_gradient.h"
#include "test_gui.h"
#include "test_list.h"
#include "test_math.h"
#include "test_method_bind.h"
#include "test_oa_hash_map.h"
#include "test_ordered_hash_map.h"
#include "test_physics_2d.h"
#include "test_physics_3d.h"
#include "test_render.h"
#include "test_shader_lang.h"
#include "test_string.h"
#include "test_validate_testing.h"
#include "test_variant.h"

#include "modules/modules_tests.gen.h"

#include "tests/test_macros.h"

int test_main(int argc, char *argv[]) {
	bool run_tests = true;

	// Convert arguments to Godot's command-line.
	List<String> args;

	for (int i = 0; i < argc; i++) {
		args.push_back(String::utf8(argv[i]));
	}
	OS::get_singleton()->set_cmdline("", args);

	// Run custom test tools.
	if (test_commands) {
		for (Map<String, TestFunc>::Element *E = test_commands->front(); E; E = E->next()) {
			if (args.find(E->key())) {
				const TestFunc &test_func = E->get();
				test_func();
				run_tests = false;
				break;
			}
		}
		if (!run_tests) {
			delete test_commands;
			return 0;
		}
	}
	// Doctest runner.
	doctest::Context test_context;
	List<String> test_args;

	// Clean arguments of "--test" from the args.
	for (int x = 0; x < argc; x++) {
		String arg = String(argv[x]);
		if (arg != "--test") {
			test_args.push_back(arg);
		}
	}
	// Convert Godot command line arguments back to standard arguments.
	char **doctest_args = new char *[test_args.size()];
	for (int x = 0; x < test_args.size(); x++) {
		// Operation to convert Godot string to non wchar string.
		CharString cs = test_args[x].utf8();
		const char *str = cs.get_data();
		// Allocate the string copy.
		doctest_args[x] = new char[strlen(str) + 1];
		// Copy this into memory.
		memcpy(doctest_args[x], str, strlen(str) + 1);
	}

	test_context.applyCommandLine(test_args.size(), doctest_args);

	test_context.setOption("order-by", "name");
	test_context.setOption("abort-after", 5);
	test_context.setOption("no-breaks", true);

	for (int x = 0; x < test_args.size(); x++) {
		delete[] doctest_args[x];
	}
	delete[] doctest_args;

	return test_context.run();
}

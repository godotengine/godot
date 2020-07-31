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
#include "test_gdscript.h"
#include "test_gui.h"
#include "test_math.h"
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
	// Doctest runner.
	doctest::Context test_context;
	List<String> valid_arguments;

	// Clean arguments of "--test" from the args.
	int argument_count = 0;
	for (int x = 0; x < argc; x++) {
		if (strncmp(argv[x], "--test", 6) != 0) {
			valid_arguments.push_back(String(argv[x]));
			argument_count++;
		}
	}
	// Convert Godot command line arguments back to standard arguments.
	char **args = new char *[valid_arguments.size()];
	for (int x = 0; x < valid_arguments.size(); x++) {
		// Operation to convert Godot string to non wchar string.
		CharString cs = valid_arguments[x].utf8();
		const char *str = cs.get_data();
		// Allocate the string copy.
		args[x] = new char[strlen(str) + 1];
		// Copy this into memory.
		std::memcpy(args[x], str, strlen(str) + 1);
	}

	test_context.applyCommandLine(valid_arguments.size(), args);

	test_context.setOption("order-by", "name");
	test_context.setOption("abort-after", 5);
	test_context.setOption("no-breaks", true);

	for (int x = 0; x < valid_arguments.size(); x++) {
		delete[] args[x];
	}
	delete[] args;

	return test_context.run();
}

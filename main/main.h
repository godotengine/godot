/**************************************************************************/
/*  main.h                                                                */
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

#ifndef MAIN_H
#define MAIN_H

#include "core/error/error_list.h"
#include "core/os/thread.h"
#include "core/typedefs.h"

template <class T>
class Vector;

class Main {
	static void print_help(const char *p_binary);
	static uint64_t last_ticks;
	static uint32_t hide_print_fps_attempts;
	static uint32_t frames;
	static uint32_t frame;
	static bool force_redraw_requested;
	static int iterating;
	static bool agile_input_event_flushing;

public:
	static bool is_cmdline_tool();
#ifdef TOOLS_ENABLED
	enum CLIScope {
		CLI_SCOPE_TOOL, // Editor and project manager.
		CLI_SCOPE_PROJECT,
	};
	static const Vector<String> &get_forwardable_cli_arguments(CLIScope p_scope);
#endif

	static int test_entrypoint(int argc, char *argv[], bool &tests_need_run);
	static Error setup(const char *execpath, int argc, char *argv[], bool p_second_phase = true);
	static Error setup2(); // The thread calling setup2() will effectively become the main thread.
	static String get_rendering_driver_name();
#ifdef TESTS_ENABLED
	static Error test_setup();
	static void test_cleanup();
#endif
	static bool start();

	static bool iteration();
	static void force_redraw();

	static bool is_iterating();

	static void cleanup(bool p_force = false);
};

// Test main override is for the testing behavior.
#define TEST_MAIN_OVERRIDE                                         \
	bool run_test = false;                                         \
	int return_code = Main::test_entrypoint(argc, argv, run_test); \
	if (run_test) {                                                \
		return return_code;                                        \
	}

#define TEST_MAIN_PARAM_OVERRIDE(argc, argv)                       \
	bool run_test = false;                                         \
	int return_code = Main::test_entrypoint(argc, argv, run_test); \
	if (run_test) {                                                \
		return return_code;                                        \
	}

#endif // MAIN_H

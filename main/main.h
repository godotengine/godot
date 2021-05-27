/*************************************************************************/
/*  main.h                                                               */
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

#ifndef MAIN_H
#define MAIN_H

#include "core/error/error_list.h"
#include "core/os/thread.h"
#include "core/typedefs.h"

class Main {
	static Error load_project_settings();
	static void load_default_project_settings();

	static void initialize_core();
	static Error initialize_servers();

	static void load_boot_graphics();
	static String localize_scene_path(const String &p_scene_path);
	static void deinitialize_early(bool p_unregister_core = true);

public:
	static Error setup(const char *exec_path, int argc, char *argv[], bool p_finish_setup = true);
	static Error finish_setup(Thread::ID p_main_tid_override = 0);
	static bool start();

	static bool iteration();

	static void force_redraw();

	static void cleanup(bool p_force = false);

	static bool is_iterating();
	static bool is_project_manager();

	static int test_entrypoint(int argc, char *argv[], bool &finished);
#ifdef TESTS_ENABLED
	static Error test_setup();
	static void test_cleanup();
#endif
};

#define TEST_MAIN_OVERRIDE                                         \
	bool finished = false;                                         \
	int return_code = Main::test_entrypoint(argc, argv, finished); \
	if (finished) {                                                \
		return return_code;                                        \
	}

#define TEST_MAIN_PARAM_OVERRIDE(argc, argv)                       \
	bool finished = false;                                         \
	int return_code = Main::test_entrypoint(argc, argv, finished); \
	if (finished) {                                                \
		return return_code;                                        \
	}

#endif // MAIN_H

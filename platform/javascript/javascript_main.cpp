/*************************************************************************/
/*  javascript_main.cpp                                                  */
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

#include "core/io/resource_loader.h"
#include "main/main.h"
#include "platform/javascript/os_javascript.h"

#include "godot_js.h"

#include <emscripten/emscripten.h>
#include <stdlib.h>

static OS_JavaScript *os = NULL;
static uint64_t target_ticks = 0;

void exit_callback() {
	emscripten_cancel_main_loop(); // After this, we can exit!
	Main::cleanup();
	int exit_code = OS_JavaScript::get_singleton()->get_exit_code();
	memdelete(os);
	os = NULL;
	emscripten_force_exit(exit_code); // No matter that we call cancel_main_loop, regular "exit" will not work, forcing.
}

void cleanup_after_sync() {
	emscripten_set_main_loop(exit_callback, -1, false);
}

void main_loop_callback() {
	uint64_t current_ticks = os->get_ticks_usec();

	bool force_draw = os->check_size_force_redraw();
	if (force_draw) {
		Main::force_redraw();
	} else if (current_ticks < target_ticks && !force_draw) {
		return; // Skip frame.
	}

	int target_fps = Engine::get_singleton()->get_target_fps();
	if (target_fps > 0) {
		target_ticks += (uint64_t)(1000000 / target_fps);
	}
	if (os->main_loop_iterate()) {
		emscripten_cancel_main_loop(); // Cancel current loop and wait for finalize_async.
		os->get_main_loop()->finish();
		godot_js_os_finish_async(cleanup_after_sync);
	}
}

int main(int argc, char *argv[]) {
	// Set locale
	char locale_ptr[16];
	godot_js_config_locale_get(locale_ptr, sizeof(locale_ptr));
	setenv("LANG", locale_ptr, true);

	os = new OS_JavaScript();

	Main::setup(argv[0], argc - 1, &argv[1]);
	// Ease up compatibility.
	ResourceLoader::set_abort_on_missing_resources(false);
	Main::start();
	os->get_main_loop()->init();
	emscripten_set_main_loop(main_loop_callback, -1, false);
	// Immediately run the first iteration.
	// We are inside an animation frame, we want to immediately draw on the newly setup canvas.
	main_loop_callback();

	return 0;
}

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
#include "platform/javascript/display_server_javascript.h"
#include "platform/javascript/os_javascript.h"

#include <emscripten/emscripten.h>
#include <stdlib.h>

static OS_JavaScript *os = nullptr;
static uint64_t target_ticks = 0;

extern "C" EMSCRIPTEN_KEEPALIVE void _request_quit_callback(char *p_filev[], int p_filec) {
	DisplayServerJavaScript *ds = DisplayServerJavaScript::get_singleton();
	if (ds) {
		Variant event = int(DisplayServer::WINDOW_EVENT_CLOSE_REQUEST);
		Variant *eventp = &event;
		Variant ret;
		Callable::CallError ce;
		ds->window_event_callback.call((const Variant **)&eventp, 1, ret, ce);
	}
}

void exit_callback() {
	emscripten_cancel_main_loop(); // After this, we can exit!
	Main::cleanup();
	int exit_code = OS_JavaScript::get_singleton()->get_exit_code();
	memdelete(os);
	os = nullptr;
	emscripten_force_exit(exit_code); // No matter that we call cancel_main_loop, regular "exit" will not work, forcing.
}

void main_loop_callback() {
	uint64_t current_ticks = os->get_ticks_usec();

	bool force_draw = DisplayServerJavaScript::get_singleton()->check_size_force_redraw();
	if (force_draw) {
		Main::force_redraw();
	} else if (current_ticks < target_ticks) {
		return; // Skip frame.
	}

	int target_fps = Engine::get_singleton()->get_target_fps();
	if (target_fps > 0) {
		target_ticks += (uint64_t)(1000000 / target_fps);
	}
	if (os->main_loop_iterate()) {
		emscripten_cancel_main_loop(); // Cancel current loop and wait for finalize_async.
		/* clang-format off */
		EM_ASM({
			// This will contain the list of operations that need to complete before cleanup.
			Module.async_finish = [
				// Always contains at least one async promise, to avoid firing immediately if nothing is added.
				new Promise(function(accept, reject) {
					setTimeout(accept, 0);
				})
			];
		});
		/* clang-format on */
		os->get_main_loop()->finish();
		os->finalize_async(); // Will add all the async finish functions.
		/* clang-format off */
		EM_ASM({
			Promise.all(Module.async_finish).then(function() {
				Module.async_finish = [];
				return new Promise(function(accept, reject) {
					if (!Module.idbfs) {
						accept();
						return;
					}
					FS.syncfs(function(error) {
						if (error) {
							err('Failed to save IDB file system: ' + error.message);
						}
						accept();
					});
				});
			}).then(function() {
				ccall("cleanup_after_sync", null, []);
			});
		});
		/* clang-format on */
	}
}

extern "C" EMSCRIPTEN_KEEPALIVE void cleanup_after_sync() {
	emscripten_set_main_loop(exit_callback, -1, false);
}

/// When calling main, it is assumed FS is setup and synced.
int main(int argc, char *argv[]) {
	// Configure locale.
	char locale_ptr[16];
	/* clang-format off */
	EM_ASM({
		stringToUTF8(Module['locale'], $0, 16);
	}, locale_ptr);
	/* clang-format on */
	setenv("LANG", locale_ptr, true);

	// Ensure the canvas ID.
	/* clang-format off */
	EM_ASM({
		stringToUTF8("#" + Module['canvas'].id, $0, 255);
	}, DisplayServerJavaScript::canvas_id);
	/* clang-format on */

	os = new OS_JavaScript();
	os->set_idb_available((bool)EM_ASM_INT({ return Module.idbfs }));

	// We must override main when testing is enabled
	TEST_MAIN_OVERRIDE

	Main::setup(argv[0], argc - 1, &argv[1]);

	// Ease up compatibility.
	ResourceLoader::set_abort_on_missing_resources(false);

	Main::start();
	os->get_main_loop()->init();
	// Expose method for requesting quit.
	/* clang-format off */
	EM_ASM({
		Module['request_quit'] = function() {
			ccall("_request_quit_callback", null, []);
		};
	});
	/* clang-format on */
	emscripten_set_main_loop(main_loop_callback, -1, false);
	// Immediately run the first iteration.
	// We are inside an animation frame, we want to immediately draw on the newly setup canvas.
	main_loop_callback();

	return 0;
}

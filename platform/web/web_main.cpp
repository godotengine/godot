/**************************************************************************/
/*  web_main.cpp                                                          */
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

#include "display_server_web.h"
#include "godot_js.h"
#include "os_web.h"
#include "web_main_loop_pacing.h"

#include "core/config/engine.h"
#include "core/io/resource_loader.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/profiling/profiling.h"
#include "main/main.h"

#ifdef TOOLS_ENABLED
#include "core/io/file_access.h"
#include "editor/web_tools_editor_plugin.h"
#include "scene/main/scene_tree.h"
#include "scene/main/window.h" // SceneTree only forward declares it.
#endif

#include <emscripten/emscripten.h>

#include <cstdlib>

static OS_Web *os = nullptr;
static uint64_t target_ticks = 0;

static bool main_started = false;
static bool shutdown_complete = false;

void exit_callback() {
	if (!shutdown_complete) {
		return; // Still waiting.
	}
	if (main_started) {
		Main::cleanup();
		main_started = false;
	}
	int exit_code = OS_Web::get_singleton()->get_exit_code();
	memdelete(os);
	os = nullptr;
	godot_cleanup_profiler();
	emscripten_force_exit(exit_code); // Exit runtime.
}

void cleanup_after_sync() {
	shutdown_complete = true;
}

void main_loop_callback() {
	if (web_main_loop_should_skip_frame(os, target_ticks)) {
		return; // Skip frame.
	}

	if (os->main_loop_iterate()) {
		emscripten_cancel_main_loop(); // Cancel current loop and set the cleanup one.
		emscripten_set_main_loop(exit_callback, -1, false);
		godot_js_os_finish_async(cleanup_after_sync);
	}
}

void print_web_header() {
	// Emscripten.
	char *emscripten_version_char = godot_js_emscripten_get_version();
	String emscripten_version = vformat("Emscripten %s", emscripten_version_char);
	// `free()` is used here because it's not memory that was allocated by Godot.
	free(emscripten_version_char);

	// Build features.
	String thread_support = OS::get_singleton()->has_feature("threads")
			? "multi-threaded"
			: "single-threaded";
	String extensions_support = OS::get_singleton()->has_feature("web_extensions")
			? "GDExtension support"
			: "no GDExtension support";

	Vector<String> build_configuration = { emscripten_version, thread_support, extensions_support };
	print_line(vformat("Build configuration: %s.", String(", ").join(build_configuration)));
}

/// When calling main, it is assumed FS is setup and synced.
extern EMSCRIPTEN_KEEPALIVE int godot_web_main(int argc, char *argv[]) {
	godot_init_profiler();

	os = new OS_Web();

#ifdef TOOLS_ENABLED
	WebToolsEditorPlugin::initialize();
#endif

	// We must override main when testing is enabled
	TEST_MAIN_OVERRIDE

	Error err = Main::setup(argv[0], argc - 1, &argv[1]);

	// Proper shutdown in case of setup failure.
	if (err != OK) {
		// Will only exit after sync.
		emscripten_set_main_loop(exit_callback, -1, false);
		godot_js_os_finish_async(cleanup_after_sync);
		if (err == ERR_HELP) { // Returned by --help and --version, so success.
			return EXIT_SUCCESS;
		}
		return EXIT_FAILURE;
	}

	print_web_header();

	main_started = true;

	// Ease up compatibility.
	ResourceLoader::set_abort_on_missing_resources(false);

	int ret = Main::start();
	os->set_exit_code(ret);
	os->get_main_loop()->initialize();
#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_project_manager_hint() && FileAccess::exists("/tmp/preload.zip")) {
		PackedStringArray ps;
		ps.push_back("/tmp/preload.zip");
		SceneTree::get_singleton()->get_root()->emit_signal(SNAME("files_dropped"), ps);
	}
#endif
	emscripten_set_main_loop(main_loop_callback, -1, false);
	// Immediately run the first iteration.
	// We are inside an animation frame, we want to immediately draw on the newly setup canvas.
	main_loop_callback();

	return os->get_exit_code();
}

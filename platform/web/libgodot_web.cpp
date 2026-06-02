/**************************************************************************/
/*  libgodot_web.cpp                                                      */
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
#include "os_web.h"
#include "web_main_loop_pacing.h"

#include "core/config/engine.h"
#include "core/extension/godot_instance.h"
#include "core/extension/libgodot.h"
#include "main/main.h"

#include <emscripten/emscripten.h>

static OS_Web *os = nullptr;

static GodotInstance *instance = nullptr;

GDExtensionObjectPtr libgodot_create_godot_instance(int p_argc, char *p_argv[], GDExtensionInitializationFunction p_init_func) {
	ERR_FAIL_COND_V_MSG(instance != nullptr, nullptr, "Only one Godot Instance may be created.");

	os = memnew(OS_Web);

	Error err = Main::setup(p_argv[0], p_argc - 1, &p_argv[1], false);
	if (err != OK) {
		memdelete(os);
		os = nullptr;
		return nullptr;
	}

	instance = memnew(GodotInstance);
	if (p_init_func != nullptr && !instance->initialize(p_init_func)) {
		// The instance never started, but Main::setup() already succeeded, so tear
		// down through the same path as libgodot_destroy_godot_instance (minus stop).
		memdelete(instance);
		instance = nullptr;
		Main::cleanup();
		memdelete(os);
		os = nullptr;
		return nullptr;
	}

	return (GDExtensionObjectPtr)instance;
}

void libgodot_destroy_godot_instance(GDExtensionObjectPtr p_godot_instance) {
	GodotInstance *godot_instance = (GodotInstance *)p_godot_instance;
	if (instance == godot_instance) {
		godot_instance->stop();
		memdelete(godot_instance);
		instance = nullptr;
		Main::cleanup();
		memdelete(os);
		os = nullptr;
	}
}

static bool shutdown_complete = false;

// Captured from OS::get_exit_code() at teardown so the host can read it after the
// engine quits (the OS object is freed by libgodot_destroy_godot_instance).
static int exit_code = 0;

static void libgodot_web_cleanup_after_sync() {
	shutdown_complete = true;
}

static void libgodot_web_exit() {
	if (!shutdown_complete) {
		return; // Still flushing the filesystem.
	}
	emscripten_cancel_main_loop();
	if (instance != nullptr) {
		// Read the exit code before destroying the instance, while the OS still exists.
		exit_code = OS_Web::get_singleton()->get_exit_code();
		// Tear the engine down through the regular path (Main::cleanup, free OS). No
		// emscripten_force_exit, so the hosting .NET runtime keeps running afterwards.
		libgodot_destroy_godot_instance((GDExtensionObjectPtr)instance);
	}
}

static uint64_t target_ticks = 0;

static void libgodot_web_iterate() {
	if (instance == nullptr) {
		return;
	}

	if (web_main_loop_should_skip_frame(os, target_ticks)) {
		return; // Skip frame.
	}

	if (instance->iteration()) {
		emscripten_cancel_main_loop(); // Cancel current loop and set the cleanup one.
		emscripten_set_main_loop(libgodot_web_exit, -1, false);
		godot_js_os_finish_async(libgodot_web_cleanup_after_sync);
	}
}

extern "C" LIBGODOT_API bool libgodot_web_start_godot_instance(GDExtensionObjectPtr p_godot_instance) {
	GodotInstance *godot_instance = (GodotInstance *)p_godot_instance;
	if (instance == nullptr || instance != godot_instance) {
		return false;
	}
	if (!instance->start()) {
		return false;
	}
	emscripten_set_main_loop(libgodot_web_iterate, -1, false);
	return true;
}

// Page JS can read this as Module._libgodot_web_get_exit_code() after the engine quits.
extern "C" LIBGODOT_API int libgodot_web_get_exit_code() {
	return exit_code;
}

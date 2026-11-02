// SPDX-License-Identifier: MIT
// 2dog: this file is part of https://2dog.dev

#include "display_server_web.h"
#include "godot_js.h"
#include "os_web.h"

#include "core/core_globals.h"
#include "core/profiling/profiling.h"

#ifndef PROXY_TO_PTHREAD_ENABLED
#include "core/config/engine.h"
#endif
#include "core/extension/godot_instance.h"
#include "core/extension/libgodot.h"
#include "core/io/resource_loader.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/variant/variant.h"
#include "main/main.h"

#include <cstdlib>

static OS_Web *os = nullptr;

static GodotInstance *instance = nullptr;
#ifndef PROXY_TO_PTHREAD_ENABLED
uint64_t target_ticks = 0;
#endif

static void print_web_header() {
	char *emscripten_version_char = godot_js_emscripten_get_version();
	String emscripten_version = vformat("Emscripten %s", emscripten_version_char);
	// 2dog: this value is allocated outside Godot's memory API.
	free(emscripten_version_char);

	String thread_support = OS::get_singleton()->has_feature("threads")
			? "multi-threaded"
			: "single-threaded";
	String extensions_support = OS::get_singleton()->has_feature("web_extensions")
			? "GDExtension support"
			: "no GDExtension support";

	Vector<String> build_configuration = { emscripten_version, thread_support, extensions_support };
	print_line(vformat("Build configuration: %s.", String(", ").join(build_configuration)));
}

GDExtensionObjectPtr libgodot_create_godot_instance(int p_argc, char *p_argv[], GDExtensionInitializationFunction p_init_func) {
	ERR_FAIL_COND_V_MSG(instance != nullptr, nullptr, "Only one Godot instance may be created at a time.");

	godot_init_profiler();

	// 2dog: retain process metadata across restarts; each engine lifetime needs a fresh canvas.
	CoreGlobals::global_init_func_libgodot = p_init_func;
	CoreGlobals::engine_reinit_enabled = true;
	CoreGlobals::libgodot_host_instance = true;
#ifndef PROXY_TO_PTHREAD_ENABLED
	target_ticks = 0;
#endif

	os = new OS_Web();

	Error err = Main::setup(p_argv[0], p_argc - 1, &p_argv[1], false);
	if (err != OK) {
		delete os;
		os = nullptr;
		godot_cleanup_profiler();
		return nullptr;
	}

	instance = memnew(GodotInstance);
	if (!instance->initialize()) {
		memdelete(instance);
		instance = nullptr;
		Main::cleanup();
		delete os;
		os = nullptr;
		godot_cleanup_profiler();
		return nullptr;
	}

	print_web_header();

	// 2dog: embedded web hosts may provide resources outside the project pack.
	ResourceLoader::set_abort_on_missing_resources(false);

	return static_cast<GDExtensionObjectPtr>(instance);
}

void libgodot_destroy_godot_instance(GDExtensionObjectPtr p_godot_instance) {
	GodotInstance *godot_instance = static_cast<GodotInstance *>(p_godot_instance);
	if (instance == godot_instance) {
		godot_instance->stop();
		memdelete(godot_instance);
		instance = nullptr;
		Main::cleanup();
		delete os;
		os = nullptr;
		CoreGlobals::global_init_func_libgodot = nullptr;
		CoreGlobals::libgodot_host_instance = false;
		godot_cleanup_profiler();
#ifndef PROXY_TO_PTHREAD_ENABLED
		target_ticks = 0;
#endif
	}
}

// 2dog: entry point used by the managed browser main loop.
extern "C" LIBGODOT_API GDExtensionBool libgodot_web_iteration() {
	ERR_FAIL_NULL_V(instance, true);
	ERR_FAIL_NULL_V(os, true);
	ERR_FAIL_NULL_V(DisplayServerWeb::get_singleton(), true);
#ifndef PROXY_TO_PTHREAD_ENABLED
	uint64_t current_ticks = os->get_ticks_usec();
#endif

	bool force_draw = DisplayServerWeb::get_singleton()->check_size_force_redraw();
	if (force_draw) {
		Main::force_redraw();
#ifndef PROXY_TO_PTHREAD_ENABLED
	} else if (current_ticks < target_ticks) {
		return false;
#endif
	}

#ifndef PROXY_TO_PTHREAD_ENABLED
	int max_fps = Engine::get_singleton()->get_max_fps();
	if (max_fps > 0) {
		if (current_ticks - target_ticks > 1000000) {
			// 2dog: reset accumulated delay after focus loss.
			target_ticks = current_ticks;
		}
		target_ticks += (uint64_t)(1000000 / max_fps);
	}
#endif

	return os->main_loop_iterate();
}

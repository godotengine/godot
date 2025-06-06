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

#pragma once

#include "core/config/project_settings.h"
#include "core/os/thread.h"
#include "core/typedefs.h"

template <typename T>
class Vector;

class Main {
	struct LaunchOptions {
		struct Option {
			int id = -1;
			int availability = -1;
			String name;
			String alias;
			String params;
			String description;
		};

		enum {
			ARG_HELP,
			ARG_VERSION,
			ARG_VERBOSE,
			ARG_QUIET,
			ARG_NO_HEADER,
			ARG_SEPARATOR,
			ARG_EDITOR,
			ARG_PROJECT_MANAGER,
			ARG_RECOVERY_MODE,
			ARG_DEBUG_SERVER,
			ARG_DAP_PORT,
			ARG_LSP_PORT,
			ARG_QUIT,
			ARG_QUIT_AFTER,
			ARG_LANGUAGE,
			ARG_PATH,
			ARG_SCENE,
			ARG_UPWARDS,
			ARG_MAIN_PACK,
			ARG_RENDER_THREAD_MODE,
			ARG_RENDER_REMOTE_FS,
			ARG_RENDER_REMOTE_FS_PASSWORD,
			ARG_AUDIO_DRIVER,
			ARG_DISPLAY_DRIVER,
			ARG_AUDIO_OUTPUT_LATENCY,
			ARG_RENDERING_METHOD,
			ARG_RENDERING_DRIVER,
			ARG_GPU_INDEX,
			ARG_TEXT_DRIVER,
			ARG_TABLET_DRIVER,
			ARG_HEADLESS,
			ARG_LOG_FILE,
			ARG_WRITE_MOVE,
			ARG_FULLSCREEN,
			ARG_MAXIMIZED,
			ARG_WINDOWED,
			ARG_ALWAYS_ON_TOP,
			ARG_RESOLUTION,
			ARG_POSITION,
			ARG_SCREEN,
			ARG_SINGLE_WINDOW,
			ARG_XR_MODE,
			ARG_WINDOW_ID,
			ARG_ACCESSIBILITY,
			ARG_DEBUG,
			ARG_BREAKPOINTS,
			ARG_IGNORE_ERROR_BREAKS,
			ARG_PROFILING,
			ARG_GPU_PROFILE,
			ARG_GPU_VALIDATION,
			ARG_GPU_ABORT,
			ARG_GENERATE_SPIRV_DEBUG_INFO,
			ARG_EXTRA_GPU_MEMORY_TRACKING,
			ARG_ACCURATE_BREADCRUMBS,
			ARG_REMOTE_DEBUG,
			ARG_SINGLE_THREADED_SCENE,
			ARG_DEBUG_COLLISIONS,
			ARG_DEBUG_PATHS,
			ARG_DEBUG_NAVIGATION,
			ARG_DEBUG_AVOIDANCE,
			ARG_DEBUG_STRING_NAMES,
			ARG_DEBUG_CANVAS_ITEM_REDRAW,
			ARG_MAX_FPS,
			ARG_FRAME_DELAY,
			ARG_TIME_SCALE,
			ARG_DISABLE_VSYNC,
			ARG_DISABLE_RENDER_LOOP,
			ARG_DISABLE_CRASH_HANDLER,
			ARG_FIXED_FPS,
			ARG_DELTA_SMOOTHING,
			ARG_PRINT_FPS,
			ARG_EDITOR_PSEUDOLOCALIZATION,
			ARG_SCRIPT,
			ARG_MAIN_LOOP,
			ARG_CHECK_ONLY,
			ARG_IMPORT,
			ARG_EXPORT_RELEASE,
			ARG_EXPORT_DEBUG,
			ARG_EXPORT_PACK,
			ARG_EXPORT_PATCH,
			ARG_PATCHES,
			ARG_INSTALL_ANDROID_BUILD_TEMPLATE,
			ARG_CONVERT_3_TO_4,
			ARG_VALIDATE_CONVERSION_3_TO_4,
			ARG_DOCTOOL,
			ARG_NO_DOCBASE,
			ARG_GDEXTENSION_DOCS,
			ARG_GDSCRIPT_DOCS,
			ARG_BUILD_SOLUTIONS,
			ARG_DUMP_GDEXTENSION_INTERFACE,
			ARG_DUMP_EXTENSION_API,
			ARG_DUMP_EXTENSION_API_WITH_DOCS,
			ARG_VALIDATE_EXTENSION_API,
			ARG_BENCHMARK,
			ARG_BENCHMARK_FILE,
			ARG_TEST,
		};

		LocalVector<Option> registered_options;

		const List<String>::Element *I = nullptr;

		void register_option(int p_id, const String &p_name, const String &p_description, const String &p_params = String(), const String &p_alias = String());
		void register_option(int p_id, int p_availability, const String &p_name, const String &p_description, const String &p_params = String(), const String &p_alias = String());

		void print_help();
		
		void process_arguments(const List<String> &p_arguments);
		int get_next_option();
		bool is_finished() const;
		String get_next_argument();
	};

	enum CLIOptionAvailability {
		CLI_OPTION_AVAILABILITY_EDITOR,
		CLI_OPTION_AVAILABILITY_TEMPLATE_DEBUG,
		CLI_OPTION_AVAILABILITY_TEMPLATE_RELEASE,
		CLI_OPTION_AVAILABILITY_HIDDEN,
	};

	static void print_header(bool p_rich);
	static void print_help_copyright(const char *p_notice);
	static void print_help_title(const char *p_title);
	static void print_help_option(const char *p_option, const char *p_description, CLIOptionAvailability p_availability = CLI_OPTION_AVAILABILITY_TEMPLATE_RELEASE);
	static String format_help_option(const char *p_option);
	static void print_help(const char *p_binary);
	static uint64_t last_ticks;
	static uint32_t hide_print_fps_attempts;
	static uint32_t frames;
	static uint32_t frame;
	static bool force_redraw_requested;
	static int iterating;

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
	static Error setup2(bool p_show_boot_logo = true); // The thread calling setup2() will effectively become the main thread.
	static String get_rendering_driver_name();
	static void setup_boot_logo();
#ifdef TESTS_ENABLED
	static Error test_setup();
	static void test_cleanup();
#endif
	static int start();

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

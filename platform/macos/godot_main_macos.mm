/**************************************************************************/
/*  godot_main_macos.mm                                                   */
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

#import "os_macos.h"

#import "godot_application.h"

#include "core/profiling/profiling.h"
#include "main/main.h"

#ifdef defined(ASAN_ENABLED)
#include <sys/resource.h>
#endif // defined(ASAN_ENABLED)

int main(int argc, char **argv) {
	godot_init_profiler();

#if defined(VULKAN_ENABLED)
	setenv("MVK_CONFIG_FULL_IMAGE_VIEW_SWIZZLE", "1", 1); // MoltenVK - enable full component swizzling support.
	setenv("MVK_CONFIG_SWAPCHAIN_MIN_MAG_FILTER_USE_NEAREST", "0", 1); // MoltenVK - use linear surface scaling. TODO: remove when full DPI scaling is implemented.
#endif

#if defined(ASAN_ENABLED)
	// Note: Set stack size to be at least 30 MB (vs 8 MB default) to avoid overflow, address sanitizer can increase stack usage up to 3 times.
	struct rlimit stack_lim = { 0x1E00000, 0x1E00000 };
	setrlimit(RLIMIT_STACK, &stack_lim);
#endif // defined(ASAN_ENABLED)

	LocalVector<char *> args;
	args.resize(argc);
	uint32_t argsc = 0;

	int wait_for_debugger = 0; // wait 5 second by default
	bool is_embedded = false;
	bool is_headless = false;

	for (int i = 0; i < argc; i++) {
		if (strcmp("-NSDocumentRevisionsDebugMode", argv[i]) == 0) {
			// remove "-NSDocumentRevisionsDebugMode" and the next argument
			continue;
		}

		if (strcmp("--os-debug", argv[i]) == 0) {
			i++;
			wait_for_debugger = 5000; // wait 5 seconds by default
			if (i < argc && strncmp(argv[i], "--", 2) != 0) {
				wait_for_debugger = atoi(argv[i]);
			}
			continue;
		}

		if (strcmp("--embedded", argv[i]) == 0) {
			is_embedded = true;
		}
		for (size_t j = 0; j < std::size(OS_MacOS::headless_args); j++) {
			if (strcmp(OS_MacOS::headless_args[j], argv[i]) == 0) {
				is_headless = true;
				break;
			}
		}

		if (i < argc - 1 && strcmp("--display-driver", argv[i]) == 0 && strcmp("headless", argv[i + 1]) == 0) {
			is_headless = true;
		}

		args.ptr()[argsc] = argv[i];
		argsc++;
	}

	uint32_t remaining_args = argsc - 1;

	OS_MacOS *os = nullptr;
	if (is_embedded) {
#ifdef TOOLS_ENABLED
		os = memnew(OS_MacOS_Embedded(args[0], remaining_args, remaining_args > 0 ? &args[1] : nullptr));
#else
		WARN_PRINT("Embedded mode is not supported in release builds.");
		return EXIT_FAILURE;
#endif
	} else if (is_headless) {
		os = memnew(OS_MacOS_Headless(args[0], remaining_args, remaining_args > 0 ? &args[1] : nullptr));
	} else {
		os = memnew(OS_MacOS_NSApp(args[0], remaining_args, remaining_args > 0 ? &args[1] : nullptr));
	}

#ifdef TOOLS_ENABLED
	if (wait_for_debugger > 0) {
		os->wait_for_debugger(wait_for_debugger);
		print_verbose("Continuing execution.");
	}
#else
	if (wait_for_debugger > 0) {
		WARN_PRINT("--os-debug is not supported in release builds.");
	}
#endif

	if (is_embedded) {
		// No dock icon for the embedded process, as it is hosted in the Godot editor.
		ProcessSerialNumber psn = { 0, kCurrentProcess };
		(void)TransformProcessType(&psn, kProcessTransformToBackgroundApplication);
	}

	// We must override main when testing is enabled.
	TEST_MAIN_OVERRIDE

	os->run();

	// Note: `os->run()` will never return if `OS_MacOS_NSApp` is used. Use `OS_MacOS_NSApp::cleanup()` for cleanup.

	int exit_code = os->get_exit_code();

	memdelete(os);

	godot_cleanup_profiler();
	return exit_code;
}

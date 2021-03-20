/*************************************************************************/
/*  crash_handler_linuxbsd.cpp                                           */
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

#include "crash_handler_linuxbsd.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "main/main.h"

#ifdef DEBUG_ENABLED
#define CRASH_HANDLER_ENABLED 1
#endif

#ifdef CRASH_HANDLER_ENABLED
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>

static void handle_crash(int sig) {
	if (OS::get_singleton() == nullptr) {
		abort();
	}

	void *bt_buffer[256];
	size_t size = backtrace(bt_buffer, 256);
	String _execpath = OS::get_singleton()->get_executable_path();

	String msg;
	const ProjectSettings *proj_settings = ProjectSettings::get_singleton();
	if (proj_settings) {
		msg = proj_settings->get("debug/settings/crash_handler/message");
	}

	// Dump the backtrace to stderr with a message to the user
	fprintf(stderr, "%s: Program crashed with signal %d\n", __FUNCTION__, sig);

	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_CRASH);
	}

	fprintf(stderr, "Dumping the backtrace. %s\n", msg.utf8().get_data());
	char **strings = backtrace_symbols(bt_buffer, size);
	if (strings) {
		for (size_t i = 1; i < size; i++) {
			char fname[1024];
			Dl_info info;

			snprintf(fname, 1024, "%s", strings[i]);

			// Try to demangle the function name to provide a more readable one
			if (dladdr(bt_buffer[i], &info) && info.dli_sname) {
				if (info.dli_sname[0] == '_') {
					int status;
					char *demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);

					if (status == 0 && demangled) {
						snprintf(fname, 1024, "%s", demangled);
					}

					if (demangled) {
						free(demangled);
					}
				}
			}

			List<String> args;

			char str[1024];
			snprintf(str, 1024, "%p", bt_buffer[i]);
			args.push_back(str);
			args.push_back("-e");
			args.push_back(_execpath);

			String output = "";

			// Try to get the file/line number using addr2line
			int ret;
			Error err = OS::get_singleton()->execute(String("addr2line"), args, &output, &ret);
			if (err == OK) {
				output.erase(output.length() - 1, 1);
			}

			fprintf(stderr, "[%ld] %s (%s)\n", (long int)i, fname, output.utf8().get_data());
		}

		free(strings);
	}
	fprintf(stderr, "-- END OF BACKTRACE --\n");

	// Abort to pass the error to the OS
	abort();
}
#endif

CrashHandler::CrashHandler() {
	disabled = false;
}

CrashHandler::~CrashHandler() {
	disable();
}

void CrashHandler::disable() {
	if (disabled) {
		return;
	}

#ifdef CRASH_HANDLER_ENABLED
	signal(SIGSEGV, nullptr);
	signal(SIGFPE, nullptr);
	signal(SIGILL, nullptr);
#endif

	disabled = true;
}

void CrashHandler::initialize() {
#ifdef CRASH_HANDLER_ENABLED
	signal(SIGSEGV, handle_crash);
	signal(SIGFPE, handle_crash);
	signal(SIGILL, handle_crash);
#endif
}

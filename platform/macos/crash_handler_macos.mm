/**************************************************************************/
/*  crash_handler_macos.mm                                                */
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

#import "crash_handler_macos.h"

#include "core/config/project_settings.h"
#include "core/object/script_language.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/version.h"
#include "main/main.h"

#include <unistd.h>

#include <cstdio>

#if defined(DEBUG_ENABLED)
#define CRASH_HANDLER_ENABLED 1
#endif

#ifdef CRASH_HANDLER_ENABLED
#include "stack_trace_macos.h"

#include <cxxabi.h>
#include <execinfo.h>

static void handle_crash(int sig) {
	signal(SIGSEGV, SIG_DFL);
	signal(SIGFPE, SIG_DFL);
	signal(SIGILL, SIG_DFL);
	signal(SIGTRAP, SIG_DFL);

	if (OS::get_singleton() == nullptr) {
		abort();
	}

	if (OS::get_singleton()->is_crash_handler_silent()) {
		std::_Exit(0);
	}

	void *bt_buffer[256];
	size_t size = backtrace(bt_buffer, 256);
	String exec_path = OS::get_singleton()->get_executable_path();

	String msg;
	if (ProjectSettings::get_singleton()) {
		msg = GLOBAL_GET("debug/settings/crash_handler/message");
	}

	// Tell MainLoop about the crash. This can be handled by users too in Node.
	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_CRASH);
	}

	// Dump the backtrace to stderr with a message to the user.
	print_error("\n================================================================");
	print_error(vformat("%s: Program crashed with signal %d", __FUNCTION__, sig));

	// Print the engine version just before, so that people are reminded to include the version in backtrace reports.
	if (String(GODOT_VERSION_HASH).is_empty()) {
		print_error(vformat("Engine version: %s", GODOT_VERSION_FULL_NAME));
	} else {
		print_error(vformat("Engine version: %s (%s)", GODOT_VERSION_FULL_NAME, GODOT_VERSION_HASH));
	}
	print_error(vformat("Dumping the backtrace. %s", msg));

	const void *load_addr = StackTraceMacOS::find_executable_load_address();
	print_error(vformat("Load address: %x\n", (uint64_t)load_addr));

	const Vector<String> lines = StackTraceMacOS::symbolize_with_atos(exec_path, load_addr, bt_buffer, size);

	if (!lines.is_empty()) {
		// Get demangled names from dladdr for fallback.
		char **strings = backtrace_symbols(bt_buffer, size);

		for (int i = 1; i < lines.size() && i < (int)size; i++) {
			String output = lines[i];
			String mod_name = "main";
			uint64_t mod_off = (uint64_t)load_addr;

			// If atos failed for this address, fall back to dladdr.
			if (strings) {
				bool atos_fail = output.substr(0, 2) == "0x";
				if (atos_fail) {
					Vector<String> fname_spl = String(strings[i]).split(" ", false, 3);
					output = fname_spl[fname_spl.size() - 1];
				}

				Dl_info info;
				if (dladdr(bt_buffer[i], &info)) {
					mod_off = (uint64_t)info.dli_fbase;
					if (mod_off != (uint64_t)load_addr) {
						mod_name = String(info.dli_fname).get_file();
					}
					if (atos_fail && info.dli_sname && info.dli_sname[0] == '_') {
						int status;
						char *demangled = abi::__cxa_demangle(info.dli_sname, nullptr, 0, &status);

						if (status == 0 && demangled) {
							output = String(demangled);
						}

						if (demangled) {
							free(demangled);
						}
					}
				} else {
					mod_name = "<unknown module>";
				}
			}

			print_error(vformat("[%d] %x (%s+%x) - %s", (int64_t)i, (uint64_t)bt_buffer[i], mod_name, (uint64_t)bt_buffer[i] - mod_off, output));
		}

		if (strings) {
			free(strings);
		}
	} else {
		// Fallback if atos fails entirely
		char **strings = backtrace_symbols(bt_buffer, size);
		if (strings) {
			for (size_t i = 0; i < size; i++) {
				Vector<String> fname_spl = String(strings[i]).split(" ", false, 3);
				String output = fname_spl[fname_spl.size() - 1];

				String mod_name = "main";
				uint64_t mod_off = (uint64_t)load_addr;

				Dl_info info;
				if (dladdr(bt_buffer[i], &info)) {
					mod_off = (uint64_t)info.dli_fbase;
					if (mod_off != (uint64_t)load_addr) {
						mod_name = String(info.dli_fname).get_file();
					}
					if (info.dli_sname && info.dli_sname[0] == '_') {
						int status;
						char *demangled = abi::__cxa_demangle(info.dli_sname, nullptr, 0, &status);

						if (status == 0 && demangled) {
							output = String(demangled);
						}

						if (demangled) {
							free(demangled);
						}
					}
				} else {
					mod_name = "<unknown module>";
				}

				print_error(vformat("[%d] %x (%s+%x) - %s", (int64_t)i, (uint64_t)bt_buffer[i], mod_name, (uint64_t)bt_buffer[i] - mod_off, output));
			}
			free(strings);
		}
	}

	print_error("-- END OF C++ BACKTRACE --");
	print_error("================================================================");

	for (const Ref<ScriptBacktrace> &backtrace : ScriptServer::capture_script_backtraces(false)) {
		if (!backtrace->is_empty()) {
			print_error(backtrace->format());
			print_error(vformat("-- END OF %s BACKTRACE --", backtrace->get_language_name().to_upper()));
			print_error("================================================================");
		}
	}

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
	signal(SIGSEGV, SIG_DFL);
	signal(SIGFPE, SIG_DFL);
	signal(SIGILL, SIG_DFL);
	signal(SIGTRAP, SIG_DFL);
#endif

	disabled = true;
}

void CrashHandler::initialize() {
#ifdef CRASH_HANDLER_ENABLED
	signal(SIGSEGV, handle_crash);
	signal(SIGFPE, handle_crash);
	signal(SIGILL, handle_crash);
	signal(SIGTRAP, handle_crash);
#endif
}

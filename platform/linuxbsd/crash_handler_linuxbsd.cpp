/**************************************************************************/
/*  crash_handler_linuxbsd.cpp                                            */
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

#include "crash_handler_linuxbsd.h"

#include "stack_trace_linuxbsd.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/object/script_language.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/version.h"
#include "drivers/unix/stack_trace_unix.h"
#include "main/main.h"

#ifndef DEBUG_ENABLED
#undef CRASH_HANDLER_ENABLED
#endif

#ifdef CRASH_HANDLER_ENABLED
#include <dlfcn.h>

#include <csignal>
#include <cstdlib>

static void handle_crash(int sig) {
	signal(SIGSEGV, SIG_DFL);
	signal(SIGFPE, SIG_DFL);
	signal(SIGILL, SIG_DFL);

	if (OS::get_singleton() == nullptr) {
		abort();
	}

	if (OS::get_singleton()->is_crash_handler_silent()) {
		std::_Exit(0);
	}

	void *bt_buffer[StackTraceUnix::MAX_FRAMES];
	int size = backtrace(bt_buffer, StackTraceUnix::MAX_FRAMES);
	if (size <= 0) {
		abort();
	}
	String _execpath = OS::get_singleton()->get_executable_path();

	if (FileAccess::exists(_execpath + ".debugsymbols")) {
		_execpath = _execpath + ".debugsymbols";
	}

	String msg;
	if (ProjectSettings::get_singleton()) {
		msg = GLOBAL_GET("debug/settings/crash_handler/message");
	}

	// Tell MainLoop about the crash. This can be handled by users too in Node.
	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_CRASH);
	}

	// Dump the backtrace to stderr with a message to the user
	print_error("\n================================================================");
	print_error(vformat("%s: Program crashed with signal %d", __FUNCTION__, sig));

	// Print the engine version just before, so that people are reminded to include the version in backtrace reports.
	if (String(GODOT_VERSION_HASH).is_empty()) {
		print_error(vformat("Engine version: %s", GODOT_VERSION_FULL_NAME));
	} else {
		print_error(vformat("Engine version: %s (%s)", GODOT_VERSION_FULL_NAME, GODOT_VERSION_HASH));
	}
	print_error(vformat("Dumping the backtrace. %s", msg));
	const Vector<String> symbols = StackTraceUnix::collect_symbol_strings(bt_buffer, size);
	const uintptr_t relocation = StackTraceLinuxBSD::get_relocation_offset();

	print_error(vformat("Load address: %x\n", (uint64_t)relocation));

	const String addr2line_exe = StackTraceLinuxBSD::select_addr2line_executable(OS::get_singleton());
	const Vector<uint64_t> addresses = StackTraceUnix::collect_addresses(bt_buffer, size);
	const Vector<String> addr2line_results = StackTraceLinuxBSD::symbolize_with_addr2line(OS::get_singleton(), addr2line_exe, addresses, relocation, _execpath);

	for (int i = 1; i < size; i++) {
		String symbol_name = i < symbols.size() ? symbols[i] : String("<unknown symbol>");
		Dl_info info = {};
		// Try to use a better symbol name if available, but only if it adds new information compared to the one from backtrace_symbols.
		if (dladdr(bt_buffer[i], &info) != 0 && info.dli_sname != nullptr) {
			const String demangled_name = StackTraceUnix::demangle_symbol_name(info.dli_sname);
			if (!demangled_name.is_empty()) {
				// If demangling changed the name, it was a mangled C++ symbol: replace the raw text.
				// If unchanged (C-style name), preserve the raw text unless it's unhelpful, to keep library context.
				const bool was_mangled = demangled_name != String(info.dli_sname);
				if (symbol_name.is_empty() || symbol_name.contains("??") || was_mangled) {
					symbol_name = demangled_name;
				} else if (!symbol_name.contains(demangled_name)) {
					symbol_name = vformat("%s [%s]", symbol_name, demangled_name);
				}
			}
		}
		String location;
		if (i < addr2line_results.size()) {
			location = addr2line_results[i].replace("/./", "/");
		}
		print_error(vformat("[%s] %x - %s (%s)", String::num_int64(i).lpad(2), (uint64_t)bt_buffer[i], symbol_name, location));
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

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

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/object/script_language.h"
#include "core/os/main_loop.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/version.h"
#include "main/main.h"

#ifndef DEBUG_ENABLED
#undef CRASH_HANDLER_ENABLED
#endif

#ifdef CRASH_HANDLER_ENABLED
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <link.h>

#include <csignal>
#include <cstdio>
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

	void *bt_buffer[256];
	size_t size = backtrace(bt_buffer, 256);
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
	char **strings = backtrace_symbols(bt_buffer, size);
	// PIE executable relocation, zero for non-PIE executables
#ifdef __GLIBC__
	// This is a glibc only thing apparently.
	uintptr_t relocation = _r_debug.r_map->l_addr;
#else
	// Non glibc systems apparently don't give PIE relocation info.
	uintptr_t relocation = 0;
#endif //__GLIBC__

	void *load_addr = nullptr;
	{
		Dl_info info;
		if (dladdr(bt_buffer[size - 1], &info)) {
			load_addr = info.dli_fbase;
		}
	}

	print_error(vformat("Load address: %x\n", (uint64_t)load_addr));

	if (strings) {
		int ret;

		List<String> args;
		args.push_back("--version");
		String exe_name;

		if (exe_name.is_empty()) {
			String output;
			// Faster implementation from gimli-rs/addr2line.
			Error err = OS::get_singleton()->execute(OS::get_singleton()->get_environment("HOME").path_join(String("/.cargo/bin/addr2line")), args, &output, &ret);
			if (err == OK && ret == 0) {
				exe_name = OS::get_singleton()->get_environment("HOME").path_join(String("/.cargo/bin/addr2line"));
			}
		}
		if (exe_name.is_empty()) {
			String output;
			Error err = OS::get_singleton()->execute(String("llvm-addr2line"), args, &output, &ret);
			if (err == OK && ret == 0) {
				exe_name = String("llvm-addr2line");
			}
		}
		if (exe_name.is_empty()) {
			exe_name = String("addr2line");
		}

		args.clear();
		for (size_t i = 0; i < size; i++) {
			char str[1024];
			snprintf(str, 1024, "%p", (void *)((uintptr_t)bt_buffer[i] - relocation));
			args.push_back(str);
		}
		args.push_back("-e");
		args.push_back(_execpath);
		args.push_back("-f");
		args.push_back("-p");
		args.push_back("-C");

		// Try to get the file/line number using addr2line
		String addr2line_output;
		Error err = OS::get_singleton()->execute(exe_name, args, &addr2line_output, &ret);
		if (err == OK) {
			Vector<String> addr2line_results = addr2line_output.substr(0, addr2line_output.length() - 1).split("\n", false);

			for (size_t i = 1; i < size; i++) {
				String output = addr2line_results[i].replace("/./", "/");
				String mod_name = "main";
				uint64_t mod_off = (uint64_t)load_addr;
				bool addr2line_fail = output.strip_edges().ends_with("??:0");
				if (addr2line_fail) {
					output = String(strings[i]);
				}

				Dl_info info;
				if (dladdr(bt_buffer[i], &info)) {
					mod_off = (uint64_t)info.dli_fbase;
					if (mod_off != (uint64_t)load_addr) {
						mod_name = String(info.dli_fname).get_file();
					}
					if (addr2line_fail && info.dli_sname && info.dli_sname[0] == '_') {
						int status = 0;
						char *demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);

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

				// Simplify printed file paths to remove redundant `/./` sections (e.g. `/opt/godot/./core` -> `/opt/godot/core`).
				print_error(vformat("[%d] %x (%s+%x) - %s", (int64_t)i, (uint64_t)bt_buffer[i], mod_name, (uint64_t)bt_buffer[i] - mod_off, output));
			}
		} else {
			// Otherwise fall back to trace symbols.
			for (size_t i = 1; i < size; i++) {
				String output = String(strings[i]);
				String mod_name = "main";
				uint64_t mod_off = (uint64_t)load_addr;

				Dl_info info;
				// Try to demangle the function name to provide a more readable one.
				if (dladdr(bt_buffer[i], &info)) {
					mod_off = (uint64_t)info.dli_fbase;
					if (mod_off != (uint64_t)load_addr) {
						mod_name = String(info.dli_fname).get_file();
					}
					if (info.dli_sname && info.dli_sname[0] == '_') {
						int status = 0;
						char *demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);

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
		}

		free(strings);
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

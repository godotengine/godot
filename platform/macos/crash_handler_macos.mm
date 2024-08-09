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

#include "crash_handler_macos.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/version.h"
#include "main/main.h"

#include <string.h>
#include <unistd.h>

CrashHandler *CrashHandler::singleton = nullptr;

#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <stdlib.h>

static void handle_crash(int p_signal) {
	signal(SIGSEGV, SIG_DFL);
	signal(SIGFPE, SIG_DFL);
	signal(SIGILL, SIG_DFL);
	signal(SIGTRAP, SIG_DFL);

	if (OS::get_singleton() == nullptr) {
		abort();
	}

	// Tell MainLoop about the crash. This can be handled by users too in Node.
	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_CRASH);
	}

	CrashHandler *cs = CrashHandler::get_singleton();
	if (cs) {
		cs->print_header(p_signal);
		CrashHandler::TraceData td = cs->collect_trace(p_signal);
		String shortcut = cs->encode_trace(td);
		if (!shortcut.is_empty()) {
			print_error("================================================================");
			print_error(shortcut);
		}
		cs->print_trace(td);
	}

	// Abort to pass the error to the OS.
	abort();
}

CrashHandler::TraceData CrashHandler::collect_trace(int p_signal) const {
	TraceData td;

	td.signal = p_signal;

	void *bt_buffer[512];
	size_t size = backtrace(bt_buffer, 512);
	for (size_t i = 1; i < size; i++) {
		AddressData addr;
		addr.address = 0x7FFFFFFFFFFF & (uint64_t)bt_buffer[i];
		td.trace.push_back(addr);
		decode_address(td, td.trace.size() - 1, false);
	}

	return td;
}

void CrashHandler::decode_address(CrashHandler::TraceData &p_data, int p_address_idx, bool p_remap) const {
	if (p_address_idx < 0 || p_address_idx >= (int)p_data.trace.size()) {
		return;
	}

	AddressData &addr_data = p_data.trace[p_address_idx];
	addr_data.fname = "???";
	if (p_remap) {
		// Relative address, remap it to the loaded module base address.
		String module_name = p_data.modules[addr_data.module_idx].fname;

		bool found = false;
		uint32_t dyld_count = _dyld_image_count();
		for (uint32_t i = 0; i < dyld_count; i++) {
			String image_name = String(_dyld_get_image_name(i));
			if (image_name == module_name) {
				const struct mach_header *header_addr = _dyld_get_image_header(i);
				Dl_info info;
				if (dladdr((void *)header_addr, &info)) {
					addr_data.address += (uint64_t)info.dli_fbase;
					found = true;
					break;
				}
			}
		}
		if (!found) {
			return;
		}
	}
	Dl_info info;
	if (dladdr((void *)((uint64_t)addr_data.address), &info)) {
		String module_name = String(info.dli_fname);
		if (!p_remap) {
			// Absolute address from the current process, save module info.
			for (int i = 0; i < (int)p_data.modules.size(); i++) {
				if (p_data.modules[i].fname == module_name) {
					addr_data.module_idx = i;
					break;
				}
			}
			if (addr_data.module_idx == -1) {
				ModuleData md;
				md.fname = module_name;
				md.load_address = (uint64_t)info.dli_fbase;
				p_data.modules.push_back(md);
				addr_data.module_idx = p_data.modules.size() - 1;
			}
		}
		addr_data.system = (module_name.begins_with("/usr/lib") || module_name.begins_with("/Library") || module_name.begins_with("/System/Library"));
		addr_data.base = (uint64_t)info.dli_fbase;
		if (addr_data.system) {
			addr_data.faddress = (uint64_t)info.dli_saddr;
			addr_data.fname = String(info.dli_sname);
			if (info.dli_sname && info.dli_sname[0] == '_') {
				int status;
				char *demangled = abi::__cxa_demangle(info.dli_sname, nullptr, 0, &status);
				if (status == 0 && demangled) {
					addr_data.fname = String(demangled);
				}
				if (demangled) {
					free(demangled);
				}
			}
			addr_data.fname = vformat("%s + %ux", addr_data.fname, addr_data.address - addr_data.faddress);
		} else {
			List<String> args;
			args.push_back("-o");
			args.push_back(module_name);
#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64__)
			args.push_back("-arch");
			args.push_back("x86_64");
#elif defined(__aarch64__)
			args.push_back("-arch");
			args.push_back("arm64");
#endif
			args.push_back("-l");
			args.push_back(vformat("0x%ux", addr_data.base));
			args.push_back(vformat("0x%ux", addr_data.address));

			int ret;
			String out = "";
			Error err = OS::get_singleton()->execute(String("atos"), args, &out, &ret);
			if (err == OK && out.substr(0, 2) != "0x") {
				addr_data.fname = out.substr(0, out.length() - 1);
			}
		}
	}
}

void CrashHandler::initialize() {
	signal(SIGSEGV, handle_crash);
	signal(SIGFPE, handle_crash);
	signal(SIGILL, handle_crash);
	signal(SIGTRAP, handle_crash);
}

void CrashHandler::disable() {
	if (disabled) {
		return;
	}

	signal(SIGSEGV, SIG_DFL);
	signal(SIGFPE, SIG_DFL);
	signal(SIGILL, SIG_DFL);
	signal(SIGTRAP, SIG_DFL);

	disabled = true;
}

CrashHandler::CrashHandler() {
	singleton = this;
	disabled = false;
}

CrashHandler::~CrashHandler() {
	disable();
	singleton = nullptr;
}

/**************************************************************************/
/*  stack_trace_unix.h                                                    */
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

#if defined(__cplusplus)

// Unix-style stack trace printing works on macOS, iOS, Linux, and BSDs, but does NOT work on Android or Web.
#if defined(__APPLE__) || defined(LINUXBSD_ENABLED)

#include "core/os/os.h" // IWYU pragma: keep.

#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>

#include <cstdlib>

namespace StackTraceUnix {

constexpr int MAX_FRAMES = 256;

inline Vector<uint64_t> collect_addresses(void *const *p_callstack, int p_frames) {
	Vector<uint64_t> addresses;
	for (int i = 0; i < p_frames; i++) {
		addresses.push_back((uint64_t)p_callstack[i]);
	}
	return addresses;
}

inline Vector<String> collect_symbol_strings(void *const *p_callstack, int p_frames) {
	Vector<String> symbols;
	if (p_frames <= 0) {
		return symbols;
	}
	char **const raw_symbols = backtrace_symbols(const_cast<void **>(p_callstack), p_frames);
	if (raw_symbols == nullptr) {
		return symbols;
	}
	symbols.resize(p_frames);
	for (int i = 0; i < p_frames; i++) {
		symbols.write[i] = raw_symbols[i];
	}
	free(raw_symbols);
	return symbols;
}

inline String demangle_symbol_name(const char *p_symbol) {
	if (p_symbol == nullptr) {
		return String();
	}
	int status = 0;
	char *demangled = abi::__cxa_demangle(p_symbol, nullptr, nullptr, &status);
	if (status == 0 && demangled != nullptr) {
		const String result = demangled;
		free(demangled);
		return result;
	}
	if (demangled != nullptr) {
		free(demangled);
	}
	return p_symbol;
}

inline uint64_t find_executable_load_address(const String &p_exec_path) {
	void *const handle = dlopen(p_exec_path.utf8().get_data(), RTLD_LAZY | RTLD_NOLOAD);
	if (handle == nullptr) {
		return 0;
	}
	void *const addr = dlsym(handle, "main");
	uint64_t load_addr = 0;
	if (addr != nullptr) {
		Dl_info info = {};
		if (dladdr(addr, &info) != 0) {
			load_addr = (uint64_t)info.dli_fbase;
		}
	}
	dlclose(handle);
	return load_addr;
}

} // namespace StackTraceUnix

#endif // defined(__APPLE__) || defined(LINUXBSD_ENABLED)

#endif // defined(__cplusplus)

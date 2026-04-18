/**************************************************************************/
/*  stack_trace_linuxbsd.h                                                */
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

#ifdef __cplusplus

#include "core/os/os.h" // IWYU pragma: keep.

#include <cstdio>

#ifdef __GLIBC__
#include <link.h>
#endif

namespace StackTraceLinuxBSD {

inline String find_addr2line_executable() {
	List<String> args;
	args.push_back("--version");
	String output;
	OS *os = OS::get_singleton();
	int ret = 0;
	Error err = OK;
	// First, check for addr2line in the home directory's cargo bin.
	if (os->has_environment("HOME")) {
		// Faster implementation from gimli-rs/addr2line.
		const String cargo_addr2line = os->get_environment("HOME").path_join(String("/.cargo/bin/addr2line"));
		err = os->execute(cargo_addr2line, args, &output, &ret);
		if (err == OK && ret == 0) {
			return cargo_addr2line;
		}
	}
	// Otherwise, check for llvm-addr2line.
	err = os->execute(String("llvm-addr2line"), args, &output, &ret);
	if (err == OK && ret == 0) {
		return String("llvm-addr2line");
	}
	// Fallback guess if none of the above returned a definitive result.
	return String("addr2line");
}

inline Vector<String> symbolize_with_addr2line(OS *p_os, const String &p_addr2line_exe, const Vector<uint64_t> &p_addresses, uintptr_t p_relocation, const String &p_exec_path) {
	List<String> args;
	for (const uint64_t address_u64 : p_addresses) {
		char address[32];
		snprintf(address, sizeof(address), "%p", (void *)((uintptr_t)address_u64 - p_relocation));
		args.push_back(address);
	}
	args.push_back("-e");
	args.push_back(p_exec_path);
	args.push_back("-f");
	args.push_back("-p");
	args.push_back("-C");
	String output;
	int ret = 0;
	const Error err = p_os->execute(p_addr2line_exe, args, &output, &ret);
	if (err == OK && ret == 0 && !output.is_empty()) {
		return output.split("\n", false);
	}
	return Vector<String>();
}

inline uintptr_t get_relocation_offset() {
#ifdef __GLIBC__
	if (_r_debug.r_map != nullptr) {
		return _r_debug.r_map->l_addr;
	}
#endif
	return 0;
}

} // namespace StackTraceLinuxBSD

#endif // __cplusplus

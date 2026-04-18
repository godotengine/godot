/**************************************************************************/
/*  stack_trace_macos.h                                                   */
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

#include "core/os/os.h" // IWYU pragma: keep.

#include <cstdio>

namespace StackTraceMacOS {

inline Vector<String> symbolize_with_atos(OS *p_os, const String &p_exec_path, uint64_t p_load_addr, const Vector<uint64_t> &p_addresses) {
	List<String> args;
	args.push_back("-o");
	args.push_back(p_exec_path);
#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64__)
	args.push_back("-arch");
	args.push_back("x86_64");
#elif defined(__aarch64__) || defined(__arm64__)
	args.push_back("-arch");
	args.push_back("arm64");
#endif
	args.push_back("--fullPath");
	args.push_back("-l");
	{
		char str[32];
		snprintf(str, sizeof(str), "%p", (void *)p_load_addr);
		args.push_back(str);
	}
	for (uint64_t address_u64 : p_addresses) {
		char str[32];
		snprintf(str, sizeof(str), "%p", (void *)address_u64);
		args.push_back(str);
	}
	String atos_output;
	int ret = 0;
	const Error err = p_os->execute("atos", args, &atos_output, &ret);
	if (err == OK && ret == 0 && !atos_output.is_empty()) {
		return atos_output.split("\n", false);
	}
	return Vector<String>();
}

inline String extract_atos_location(const String &p_atos_line) {
	int at_pos = p_atos_line.rfind(" (at ");
	if (at_pos == -1) {
		return String();
	}
	int end_pos = p_atos_line.rfind(")");
	if (end_pos <= at_pos) {
		return String();
	}
	return p_atos_line.substr(at_pos + 5, end_pos - at_pos - 5);
}

inline bool is_unresolved_atos_line(const String &p_line) {
	return p_line.begins_with("0x");
}

} // namespace StackTraceMacOS

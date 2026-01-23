/**************************************************************************/
/*  bench_string_name.h                                                   */
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

#include "core/os/os.h"
#include "core/string/string_name.h"
#include "tests/test_macros.h"

namespace BenchStringName {

TEST_CASE("[StringName] Benchmark SNAME vs runtime StringName") {
	// Warmup
	for (int i = 0; i < 1000; i++) {
		StringName s("warmup");
		StringName s2 = SNAME("warmup_static");
	}

	const int iterations = 10000000;
	uint64_t start, end;

	// Benchmark 1: Runtime construction
	start = OS::get_singleton()->get_ticks_usec();
	for (int i = 0; i < iterations; i++) {
		StringName s("runtime_string");
	}
	end = OS::get_singleton()->get_ticks_usec();
	print_line(vformat("Runtime StringName construction: %d usec", end - start));

	// Benchmark 2: SNAME construction
	start = OS::get_singleton()->get_ticks_usec();
	for (int i = 0; i < iterations; i++) {
		StringName s = SNAME("static_string");
	}
	end = OS::get_singleton()->get_ticks_usec();
	print_line(vformat("SNAME construction: %d usec", end - start));
}

} // namespace BenchStringName

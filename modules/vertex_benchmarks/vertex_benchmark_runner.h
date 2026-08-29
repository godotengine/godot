/**************************************************************************/
/*  vertex_benchmark_runner.h                                            */
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

#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "vertex_benchmark_result.h"

// Collects per-frame performance counters for named workloads and computes
// aggregates (min/max/avg FPS, avg frame time, draw calls, RAM). Spawn the
// workload (sprites/particles/tilemaps/physics/UI) from scripting, call
// record_sample("workload") once per frame, then finalize_workload() to
// compute the aggregate. Also exposes startup_time_ms for cold-start timing.
class VertexBenchmarkRunner : public Object {
	GDCLASS(VertexBenchmarkRunner, Object)

private:
	Ref<VertexBenchmarkResult> result;
	uint64_t start_time_us = 0;

protected:
	static void _bind_methods();

public:
	Dictionary sample_now() const;
	void record_sample(const String &p_workload);
	Dictionary finalize_workload(const String &p_workload);

	void mark_startup_start();
	Dictionary mark_startup_end() const; // Returns {"startup_time_ms": float}.

	Ref<VertexBenchmarkResult> get_result() const { return result; }
	Dictionary to_dictionary() const;

	VertexBenchmarkRunner();
};

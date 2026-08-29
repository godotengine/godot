/**************************************************************************/
/*  vertex_benchmark_runner.cpp                                          */
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

#include "vertex_benchmark_runner.h"

#include "core/object/class_db.h"
#include "core/os/memory.h"
#include "core/os/os.h"
#include "main/performance.h"

Dictionary VertexBenchmarkRunner::sample_now() const {
	Dictionary s;
	Performance *perf = Performance::get_singleton();
	if (perf) {
		double fps = perf->get_monitor(Performance::TIME_FPS);
		s["fps"] = fps;
		s["frame_time_ms"] = fps > 0.0 ? 1000.0 / fps : 0.0;
		s["process_time_ms"] = perf->get_monitor(Performance::TIME_PROCESS);
		s["physics_time_ms"] = perf->get_monitor(Performance::TIME_PHYSICS_PROCESS);
		s["draw_calls"] = perf->get_monitor(Performance::RENDER_TOTAL_DRAW_CALLS_IN_FRAME);
		s["primitives"] = perf->get_monitor(Performance::RENDER_TOTAL_PRIMITIVES_IN_FRAME);
		s["texture_mem_mb"] = perf->get_monitor(Performance::RENDER_TEXTURE_MEM_USED) / (1024.0 * 1024.0);
		s["video_mem_mb"] = perf->get_monitor(Performance::RENDER_VIDEO_MEM_USED) / (1024.0 * 1024.0);
		s["node_count"] = perf->get_monitor(Performance::OBJECT_NODE_COUNT);
	}
	s["static_mem_mb"] = Memory::get_mem_usage() / (1024.0 * 1024.0);
	return s;
}

void VertexBenchmarkRunner::record_sample(const String &p_workload) {
	if (result.is_null()) {
		result.instantiate();
	}
	result->add_sample(p_workload, sample_now());
}

Dictionary VertexBenchmarkRunner::finalize_workload(const String &p_workload) {
	if (result.is_null()) {
		return Dictionary();
	}
	Dictionary samples = result->get_samples();
	if (!samples.has(p_workload)) {
		return Dictionary();
	}
	Array arr = samples[p_workload];
	int n = arr.size();
	if (n == 0) {
		return Dictionary();
	}
	double fps_min = 1e9, fps_max = 0, fps_sum = 0;
	double frame_sum = 0, draw_sum = 0, tex_sum = 0, vid_sum = 0, mem_sum = 0;
	for (int i = 0; i < n; i++) {
		Dictionary d = arr[i];
		double fps = d.has("fps") ? (double)d["fps"] : 0.0;
		fps_min = MIN(fps_min, fps);
		fps_max = MAX(fps_max, fps);
		fps_sum += fps;
		frame_sum += d.has("frame_time_ms") ? (double)d["frame_time_ms"] : 0.0;
		draw_sum += d.has("draw_calls") ? (double)d["draw_calls"] : 0.0;
		tex_sum += d.has("texture_mem_mb") ? (double)d["texture_mem_mb"] : 0.0;
		vid_sum += d.has("video_mem_mb") ? (double)d["video_mem_mb"] : 0.0;
		mem_sum += d.has("static_mem_mb") ? (double)d["static_mem_mb"] : 0.0;
	}
	Dictionary agg;
	agg["samples"] = n;
	agg["fps_min"] = fps_min;
	agg["fps_max"] = fps_max;
	agg["fps_avg"] = fps_sum / n;
	agg["frame_time_avg_ms"] = frame_sum / n;
	agg["draw_calls_avg"] = draw_sum / n;
	agg["texture_mem_avg_mb"] = tex_sum / n;
	agg["video_mem_avg_mb"] = vid_sum / n;
	agg["static_mem_avg_mb"] = mem_sum / n;
	result->set_aggregate(p_workload, agg);
	return agg;
}

void VertexBenchmarkRunner::mark_startup_start() {
	start_time_us = OS::get_singleton()->get_ticks_usec();
}

Dictionary VertexBenchmarkRunner::mark_startup_end() const {
	Dictionary d;
	if (start_time_us == 0) {
		d["startup_time_ms"] = 0.0;
		return d;
	}
	uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - start_time_us;
	d["startup_time_ms"] = elapsed / 1000.0;
	return d;
}

Dictionary VertexBenchmarkRunner::to_dictionary() const {
	if (result.is_null()) {
		return Dictionary();
	}
	return result->to_dictionary();
}

VertexBenchmarkRunner::VertexBenchmarkRunner() {
	if (result.is_null()) {
		result.instantiate();
	}
}

void VertexBenchmarkRunner::_bind_methods() {
	ClassDB::bind_method(D_METHOD("sample_now"), &VertexBenchmarkRunner::sample_now);
	ClassDB::bind_method(D_METHOD("record_sample", "workload"), &VertexBenchmarkRunner::record_sample);
	ClassDB::bind_method(D_METHOD("finalize_workload", "workload"), &VertexBenchmarkRunner::finalize_workload);
	ClassDB::bind_method(D_METHOD("mark_startup_start"), &VertexBenchmarkRunner::mark_startup_start);
	ClassDB::bind_method(D_METHOD("mark_startup_end"), &VertexBenchmarkRunner::mark_startup_end);
	ClassDB::bind_method(D_METHOD("get_result"), &VertexBenchmarkRunner::get_result);
	ClassDB::bind_method(D_METHOD("to_dictionary"), &VertexBenchmarkRunner::to_dictionary);
}

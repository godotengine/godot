/**************************************************************************/
/*  jxl_parallel_runner.cpp                                               */
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

#include "jxl_parallel_runner.h"

#include "core/object/worker_thread_pool.h"
#include "core/templates/local_vector.h"

struct JXLParallelJob {
	void *jpegxl_opaque = nullptr;
	JxlParallelRunFunction func = nullptr;
	uint32_t start = 0;
	uint32_t end = 0;
	uint32_t chunks = 0;
};

struct JXLParallelChunk {
	const JXLParallelJob *job = nullptr;
	uint32_t index = 0;
};

static void _jxl_run_chunk(void *p_userdata) {
	const JXLParallelChunk *chunk = static_cast<const JXLParallelChunk *>(p_userdata);
	const JXLParallelJob *job = chunk->job;
	const uint64_t count = job->end - job->start;
	const uint32_t from = job->start + uint32_t(count * chunk->index / job->chunks);
	const uint32_t to = job->start + uint32_t(count * (chunk->index + 1) / job->chunks);
	// Each chunk runs on one task, so its index works as a libjxl thread ID.
	for (uint32_t i = from; i < to; i++) {
		job->func(job->jpegxl_opaque, i, chunk->index);
	}
}

JxlParallelRetCode jxl_parallel_runner(void *p_runner_opaque, void *p_jpegxl_opaque, JxlParallelRunInit p_init, JxlParallelRunFunction p_func, uint32_t p_start_range, uint32_t p_end_range) {
	WorkerThreadPool *pool = WorkerThreadPool::get_singleton();
	const uint32_t count = p_end_range > p_start_range ? p_end_range - p_start_range : 0;
	const uint32_t chunks = MAX(1u, MIN(count, pool ? uint32_t(pool->get_thread_count()) : 1u));

	const JxlParallelRetCode ret = p_init(p_jpegxl_opaque, chunks);
	if (ret != 0) {
		return ret;
	}

	JXLParallelJob job;
	job.jpegxl_opaque = p_jpegxl_opaque;
	job.func = p_func;
	job.start = p_start_range;
	job.end = p_end_range;
	job.chunks = chunks;

	LocalVector<JXLParallelChunk> chunk_data;
	chunk_data.resize(chunks);
	for (uint32_t i = 0; i < chunks; i++) {
		chunk_data[i].job = &job;
		chunk_data[i].index = i;
	}

	// Plain tasks rather than a group task, as waiting on them from a pool thread
	// (e.g. during threaded resource loading) runs other tasks instead of blocking.
	LocalVector<WorkerThreadPool::TaskID> tasks;
	for (uint32_t i = 1; i < chunks; i++) {
		tasks.push_back(pool->add_native_task(&_jxl_run_chunk, &chunk_data[i], true, SNAME("JXL")));
	}
	_jxl_run_chunk(&chunk_data[0]);
	for (WorkerThreadPool::TaskID task : tasks) {
		pool->wait_for_task_completion(task);
	}
	return 0;
}

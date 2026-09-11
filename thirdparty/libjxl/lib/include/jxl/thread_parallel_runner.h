/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_threads
 * @{
 * @file thread_parallel_runner.h
 * @brief implementation using std::thread of a ::JxlParallelRunner.
 */

/** Implementation of JxlParallelRunner than can be used to enable
 * multithreading when using the JPEG XL library. This uses std::thread
 * internally and related synchronization functions. The number of threads
 * created is fixed at construction time and the threads are re-used for every
 * ThreadParallelRunner::Runner call. Only one concurrent
 * JxlThreadParallelRunner call per instance is allowed at a time.
 *
 * This is a scalable, lower-overhead thread pool runner, especially suitable
 * for data-parallel computations in the fork-join model, where clients need to
 * know when all tasks have completed.
 *
 * This thread pool can efficiently load-balance millions of tasks using an
 * atomic counter, thus avoiding per-task virtual or system calls. With 48
 * hyperthreads and 1M tasks that add to an atomic counter, overall runtime is
 * 10-20x higher when using std::async, and ~200x for a queue-based thread
 */

#ifndef JXL_THREAD_PARALLEL_RUNNER_H_
#define JXL_THREAD_PARALLEL_RUNNER_H_

#include <jxl/jxl_threads_export.h>
#include <jxl/memory_manager.h>
#include <jxl/parallel_runner.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Parallel runner internally using std::thread. Use as @ref JxlParallelRunner.
 */
JXL_THREADS_EXPORT JxlParallelRetCode JxlThreadParallelRunner(
    void* runner_opaque, void* jpegxl_opaque, JxlParallelRunInit init,
    JxlParallelRunFunction func, uint32_t start_range, uint32_t end_range);

/** Creates the runner for @ref JxlThreadParallelRunner. Use as the opaque
 * runner.
 */
JXL_THREADS_EXPORT void* JxlThreadParallelRunnerCreate(
    const JxlMemoryManager* memory_manager, size_t num_worker_threads);

/** Destroys the runner created by @ref JxlThreadParallelRunnerCreate.
 */
JXL_THREADS_EXPORT void JxlThreadParallelRunnerDestroy(void* runner_opaque);

/** Returns a default num_worker_threads value for
 * @ref JxlThreadParallelRunnerCreate.
 */
JXL_THREADS_EXPORT size_t JxlThreadParallelRunnerDefaultNumWorkerThreads(void);

#ifdef __cplusplus
}
#endif

#endif /* JXL_THREAD_PARALLEL_RUNNER_H_ */

/** @}*/

/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_threads
 * @{
 * @file resizable_parallel_runner.h
 * @brief implementation using std::thread of a resizeable ::JxlParallelRunner.
 */

/** Implementation of JxlParallelRunner than can be used to enable
 * multithreading when using the JPEG XL library. This uses std::thread
 * internally and related synchronization functions. The number of threads
 * created can be changed after creation of the thread pool; the threads
 * (including the main thread) are re-used for every
 * ResizableParallelRunner::Runner call. Only one concurrent
 * @ref JxlResizableParallelRunner call per instance is allowed at a time.
 *
 * This is a scalable, lower-overhead thread pool runner, especially suitable
 * for data-parallel computations in the fork-join model, where clients need to
 * know when all tasks have completed.
 *
 * Compared to the implementation in @ref thread_parallel_runner.h, this
 * implementation is tuned for execution on lower-powered systems, including
 * for example ARM CPUs with big.LITTLE computation models.
 */

#ifndef JXL_RESIZABLE_PARALLEL_RUNNER_H_
#define JXL_RESIZABLE_PARALLEL_RUNNER_H_

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
JXL_THREADS_EXPORT JxlParallelRetCode JxlResizableParallelRunner(
    void* runner_opaque, void* jpegxl_opaque, JxlParallelRunInit init,
    JxlParallelRunFunction func, uint32_t start_range, uint32_t end_range);

/** Creates the runner for @ref JxlResizableParallelRunner. Use as the opaque
 * runner. The runner will execute tasks on the calling thread until
 * @ref JxlResizableParallelRunnerSetThreads is called.
 */
JXL_THREADS_EXPORT void* JxlResizableParallelRunnerCreate(
    const JxlMemoryManager* memory_manager);

/** Changes the number of threads for @ref JxlResizableParallelRunner.
 */
JXL_THREADS_EXPORT void JxlResizableParallelRunnerSetThreads(
    void* runner_opaque, size_t num_threads);

/** Suggests a number of threads to use for an image of given size.
 */
JXL_THREADS_EXPORT uint32_t
JxlResizableParallelRunnerSuggestThreads(uint64_t xsize, uint64_t ysize);

/** Destroys the runner created by @ref JxlResizableParallelRunnerCreate.
 */
JXL_THREADS_EXPORT void JxlResizableParallelRunnerDestroy(void* runner_opaque);

#ifdef __cplusplus
}
#endif

#endif /* JXL_RESIZABLE_PARALLEL_RUNNER_H_ */

/** @}*/

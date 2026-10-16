/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_threads
 *  @{
 */
/**
 * @file parallel_runner.h
 */

/** API for running data operations in parallel in a multi-threaded environment.
 * This module allows the JPEG XL caller to define their own way of creating and
 * assigning threads.
 *
 * The JxlParallelRunner function type defines a parallel data processing
 * runner that may be implemented by the caller to allow the library to process
 * in multiple threads. The multi-threaded processing in this library only
 * requires to run the same function over each number of a range, possibly
 * running each call in a different thread. The JPEG XL caller is responsible
 * for implementing this logic using the thread APIs available in their system.
 * For convenience, a C++ implementation based on std::thread is provided in
 * jpegxl/parallel_runner_thread.h (part of the jpegxl_threads library).
 *
 * Thread pools usually store small numbers of heterogeneous tasks in a queue.
 * When tasks are identical or differ only by an integer input parameter, it is
 * much faster to store just one function of an integer parameter and call it
 * for each value. Conventional vector-of-tasks can be run in parallel using a
 * lambda function adapter that simply calls task_funcs[task].
 *
 * If no multi-threading is desired, a @c NULL value of JxlParallelRunner
 * will use an internal implementation without multi-threading.
 */

#ifndef JXL_PARALLEL_RUNNER_H_
#define JXL_PARALLEL_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Return code used in the JxlParallel* functions as return value. A value
 * of ::JXL_PARALLEL_RET_SUCCESS means success and any other value means error.
 * The special value ::JXL_PARALLEL_RET_RUNNER_ERROR can be used by the runner
 * to indicate any other error.
 */
typedef int JxlParallelRetCode;

/**
 * Code returned by the @ref JxlParallelRunInit function to indicate success.
 */
#define JXL_PARALLEL_RET_SUCCESS (0)

/**
 * Code returned by the @ref JxlParallelRunInit function to indicate a general
 * error.
 */
#define JXL_PARALLEL_RET_RUNNER_ERROR (-1)

/**
 * Parallel run initialization callback. See @ref JxlParallelRunner for details.
 *
 * This function MUST be called by the JxlParallelRunner only once, on the
 * same thread that called @ref JxlParallelRunner, before any parallel
 * execution. The purpose of this call is to provide the maximum number of
 * threads that the
 * @ref JxlParallelRunner will use, which can be used by JPEG XL to allocate
 * per-thread storage if needed.
 *
 * @param jpegxl_opaque the @p jpegxl_opaque handle provided to
 * @ref JxlParallelRunner() must be passed here.
 * @param num_threads the maximum number of threads. This value must be
 * positive.
 * @return 0 if the initialization process was successful.
 * @return an error code if there was an error, which should be returned by
 * @ref JxlParallelRunner().
 */
typedef JxlParallelRetCode (*JxlParallelRunInit)(void* jpegxl_opaque,
                                                 size_t num_threads);

/**
 * Parallel run data processing callback. See @ref JxlParallelRunner for
 * details.
 *
 * This function MUST be called once for every number in the range [start_range,
 * end_range) (including start_range but not including end_range) passing this
 * number as the @p value. Calls for different value may be executed from
 * different threads in parallel.
 *
 * @param jpegxl_opaque the @p jpegxl_opaque handle provided to
 * @ref JxlParallelRunner() must be passed here.
 * @param value the number in the range [start_range, end_range) of the call.
 * @param thread_id the thread number where this function is being called from.
 * This must be lower than the @p num_threads value passed to
 * @ref JxlParallelRunInit.
 */
typedef void (*JxlParallelRunFunction)(void* jpegxl_opaque, uint32_t value,
                                       size_t thread_id);

/**
 * JxlParallelRunner function type. A parallel runner implementation can be
 * provided by a JPEG XL caller to allow running computations in multiple
 * threads. This function must call the initialization function @p init in the
 * same thread that called it and then call the passed @p func once for every
 * number in the range [start_range, end_range) (including start_range but not
 * including end_range) possibly from different multiple threads in parallel.
 *
 * The @ref JxlParallelRunner function does not need to be re-entrant. This
 * means that the same @ref JxlParallelRunner function with the same
 * runner_opaque provided parameter will not be called from the library from
 * either @p init or
 * @p func in the same decoder or encoder instance. However, a single decoding
 * or encoding instance may call the provided @ref JxlParallelRunner multiple
 * times for different parts of the decoding or encoding process.
 *
 * @return 0 if the @p init call succeeded (returned 0) and no other error
 * occurred in the runner code.
 * @return JXL_PARALLEL_RET_RUNNER_ERROR if an error occurred in the runner
 * code, for example, setting up the threads.
 * @return the return value of @p init() if non-zero.
 */
typedef JxlParallelRetCode (*JxlParallelRunner)(
    void* runner_opaque, void* jpegxl_opaque, JxlParallelRunInit init,
    JxlParallelRunFunction func, uint32_t start_range, uint32_t end_range);

/* The following is an example of a @ref JxlParallelRunner that doesn't use any
 * multi-threading. Note that this implementation doesn't store any state
 * between multiple calls of the ExampleSequentialRunner function, so the
 * runner_opaque value is not used.

  JxlParallelRetCode ExampleSequentialRunner(void* runner_opaque,
                                                void* jpegxl_opaque,
                                                JxlParallelRunInit init,
                                                JxlParallelRunFunction func,
                                                uint32_t start_range,
                                                uint32_t end_range) {
    // We only use one thread (the currently running thread).
    JxlParallelRetCode init_ret = (*init)(jpegxl_opaque, 1);
    if (init_ret != 0) return init_ret;

    // In case of other initialization error (for example when initializing the
    // threads) one can return JXL_PARALLEL_RET_RUNNER_ERROR.

    for (uint32_t i = start_range; i < end_range; i++) {
      // Every call is in the thread number 0. These don't need to be in any
      // order.
      (*func)(jpegxl_opaque, i, 0);
    }
    return JXL_PARALLEL_RET_SUCCESS;
  }
 */

#ifdef __cplusplus
}
#endif

#endif /* JXL_PARALLEL_RUNNER_H_ */

/** @}*/

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// @addtogroup libjxl_cpp
/// @{
///
/// @file resizable_parallel_runner_cxx.h
/// @ingroup libjxl_threads
/// @brief C++ header-only helper for @ref resizable_parallel_runner.h.
///
/// There's no binary library associated with the header since this is a header
/// only library.

#ifndef JXL_RESIZABLE_PARALLEL_RUNNER_CXX_H_
#define JXL_RESIZABLE_PARALLEL_RUNNER_CXX_H_

#include <jxl/memory_manager.h>
#include <jxl/resizable_parallel_runner.h>

#include <memory>

#ifndef __cplusplus
#error \
    "This a C++ only header. Use jxl/jxl_resizable_parallel_runner.h from C" \
    "sources."
#endif

/// Struct to call JxlResizableParallelRunnerDestroy from the
/// JxlResizableParallelRunnerPtr unique_ptr.
struct JxlResizableParallelRunnerDestroyStruct {
  /// Calls @ref JxlResizableParallelRunnerDestroy() on the passed runner.
  void operator()(void* runner) { JxlResizableParallelRunnerDestroy(runner); }
};

/// std::unique_ptr<> type that calls JxlResizableParallelRunnerDestroy() when
/// releasing the runner.
///
/// Use this helper type from C++ sources to ensure the runner is destroyed and
/// their internal resources released.
typedef std::unique_ptr<void, JxlResizableParallelRunnerDestroyStruct>
    JxlResizableParallelRunnerPtr;

/// Creates an instance of JxlResizableParallelRunner into a
/// JxlResizableParallelRunnerPtr and initializes it.
///
/// This function returns a unique_ptr that will call
/// JxlResizableParallelRunnerDestroy() when releasing the pointer. See @ref
/// JxlResizableParallelRunnerCreate for details on the instance creation.
///
/// @param memory_manager custom allocator function. It may be NULL. The memory
///        manager will be copied internally.
/// @return a @c NULL JxlResizableParallelRunnerPtr if the instance can not be
/// allocated or initialized
/// @return initialized JxlResizableParallelRunnerPtr instance otherwise.
static inline JxlResizableParallelRunnerPtr JxlResizableParallelRunnerMake(
    const JxlMemoryManager* memory_manager) {
  return JxlResizableParallelRunnerPtr(
      JxlResizableParallelRunnerCreate(memory_manager));
}

#endif  // JXL_RESIZABLE_PARALLEL_RUNNER_CXX_H_

/// @}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// @addtogroup libjxl_cpp
///@{
///
/// @file encode_cxx.h
/// @brief C++ header-only helper for @ref encode.h.
///
/// There's no binary library associated with the header since this is a header
/// only library.

#ifndef JXL_ENCODE_CXX_H_
#define JXL_ENCODE_CXX_H_

#include <jxl/encode.h>
#include <jxl/memory_manager.h>

#include <memory>

#ifndef __cplusplus
#error "This a C++ only header. Use jxl/encode.h from C sources."
#endif

/// Struct to call JxlEncoderDestroy from the JxlEncoderPtr unique_ptr.
struct JxlEncoderDestroyStruct {
  /// Calls @ref JxlEncoderDestroy() on the passed encoder.
  void operator()(JxlEncoder* encoder) { JxlEncoderDestroy(encoder); }
};

/// std::unique_ptr<> type that calls JxlEncoderDestroy() when releasing the
/// encoder.
///
/// Use this helper type from C++ sources to ensure the encoder is destroyed and
/// their internal resources released.
typedef std::unique_ptr<JxlEncoder, JxlEncoderDestroyStruct> JxlEncoderPtr;

/// Creates an instance of JxlEncoder into a JxlEncoderPtr and initializes it.
///
/// This function returns a unique_ptr that will call JxlEncoderDestroy() when
/// releasing the pointer. See @ref JxlEncoderCreate for details on the
/// instance creation.
///
/// @param memory_manager custom allocator function. It may be NULL. The memory
///        manager will be copied internally.
/// @return a @c NULL JxlEncoderPtr if the instance can not be allocated or
///         initialized
/// @return initialized JxlEncoderPtr instance otherwise.
static inline JxlEncoderPtr JxlEncoderMake(
    const JxlMemoryManager* memory_manager) {
  return JxlEncoderPtr(JxlEncoderCreate(memory_manager));
}

#endif  // JXL_ENCODE_CXX_H_

/// @}

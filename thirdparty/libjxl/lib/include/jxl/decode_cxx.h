// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// @addtogroup libjxl_cpp
/// @{
///
/// @file decode_cxx.h
/// @brief C++ header-only helper for @ref decode.h.
///
/// There's no binary library associated with the header since this is a header
/// only library.

#ifndef JXL_DECODE_CXX_H_
#define JXL_DECODE_CXX_H_

#include <jxl/decode.h>
#include <jxl/memory_manager.h>

#include <memory>

#ifndef __cplusplus
#error "This a C++ only header. Use jxl/decode.h from C sources."
#endif

/// Struct to call JxlDecoderDestroy from the JxlDecoderPtr unique_ptr.
struct JxlDecoderDestroyStruct {
  /// Calls @ref JxlDecoderDestroy() on the passed decoder.
  void operator()(JxlDecoder* decoder) { JxlDecoderDestroy(decoder); }
};

/// std::unique_ptr<> type that calls JxlDecoderDestroy() when releasing the
/// decoder.
///
/// Use this helper type from C++ sources to ensure the decoder is destroyed and
/// their internal resources released.
typedef std::unique_ptr<JxlDecoder, JxlDecoderDestroyStruct> JxlDecoderPtr;

/// Creates an instance of JxlDecoder into a JxlDecoderPtr and initializes it.
///
/// This function returns a unique_ptr that will call JxlDecoderDestroy() when
/// releasing the pointer. See @ref JxlDecoderCreate for details on the
/// instance creation.
///
/// @param memory_manager custom allocator function. It may be NULL. The memory
///        manager will be copied internally.
/// @return a @c NULL JxlDecoderPtr if the instance can not be allocated or
///         initialized
/// @return initialized JxlDecoderPtr instance otherwise.
static inline JxlDecoderPtr JxlDecoderMake(
    const JxlMemoryManager* memory_manager) {
  return JxlDecoderPtr(JxlDecoderCreate(memory_manager));
}

#endif  // JXL_DECODE_CXX_H_

/// @}

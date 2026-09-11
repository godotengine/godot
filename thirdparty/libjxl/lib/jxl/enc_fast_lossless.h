// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_FAST_LOSSLESS_H_
#define LIB_JXL_ENC_FAST_LOSSLESS_H_
#include <stdlib.h>

// FJXL_STANDALONE=1 for a stand-alone jxl encoder
// FJXL_STANDALONE=0 for use in libjxl to encode frames (but no image header)
#ifndef FJXL_STANDALONE
#define FJXL_STANDALONE 0
#endif

#if !FJXL_STANDALONE
#include <jxl/encode.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if FJXL_STANDALONE
// Simplified version of the streaming input source from jxl/encode.h
// We only need this part to wrap the full image buffer in the standalone mode
// and this way we don't need to depend on the jxl headers.
struct JxlChunkedFrameInputSource {
  void* opaque;
  const void* (*get_color_channel_data_at)(void* opaque, size_t xpos,
                                           size_t ypos, size_t xsize,
                                           size_t ysize, size_t* row_offset);
  void (*release_buffer)(void* opaque, const void* buf);
};
// The standalone version does not use this struct, but we define it here so
// that we don't have to clutter all the function signatures with defines.
struct JxlEncoderOutputProcessorWrapper {
  int unused;
};
#endif

// Simple encoding API.

// A FJxlParallelRunner must call fun(opaque, i) for all i from 0 to count. It
// may do so in parallel.
typedef void(FJxlParallelRunner)(void* runner_opaque, void* opaque,
                                 void fun(void*, size_t), size_t count);

#if FJXL_STANDALONE
// You may pass `nullptr` as a runner: encoding will be sequential.
size_t JxlFastLosslessEncode(const unsigned char* rgba, size_t width,
                             size_t row_stride, size_t height, size_t nb_chans,
                             size_t bitdepth, bool big_endian, int effort,
                             unsigned char** output, void* runner_opaque,
                             FJxlParallelRunner runner);
#endif

// More complex API for cases in which you may want to allocate your own buffer
// and other advanced use cases.

// Opaque struct that represents an intermediate state of the computation.
struct JxlFastLosslessFrameState;

// Returned JxlFastLosslessFrameState must be freed by calling
// JxlFastLosslessFreeFrameState.
JxlFastLosslessFrameState* JxlFastLosslessPrepareFrame(
    JxlChunkedFrameInputSource input, size_t width, size_t height,
    size_t nb_chans, size_t bitdepth, bool big_endian, int effort, int oneshot);

#if !FJXL_STANDALONE
class JxlEncoderOutputProcessorWrapper;
#endif

bool JxlFastLosslessProcessFrame(
    JxlFastLosslessFrameState* frame_state, bool is_last, void* runner_opaque,
    FJxlParallelRunner runner,
    JxlEncoderOutputProcessorWrapper* output_processor);

// Prepare the (image/frame) header. You may encode animations by concatenating
// the output of multiple frames, of which the first one has add_image_header =
// 1 and subsequent ones have add_image_header = 0, and all frames but the last
// one have is_last = 0.
// (when FJXL_STANDALONE=0, add_image_header has to be 0)
void JxlFastLosslessPrepareHeader(JxlFastLosslessFrameState* frame,
                                  int add_image_header, int is_last);

// Upper bound on the required output size, including any padding that may be
// required by JxlFastLosslessWriteOutput. Cannot be called before
// JxlFastLosslessPrepareHeader.
size_t JxlFastLosslessMaxRequiredOutput(const JxlFastLosslessFrameState* frame);

// Actual size of the frame once it is encoded. This is not identical to
// JxlFastLosslessMaxRequiredOutput because JxlFastLosslessWriteOutput may
// require extra padding.
size_t JxlFastLosslessOutputSize(const JxlFastLosslessFrameState* frame);

// Writes the frame to the given output buffer. Returns the number of bytes that
// were written, which is at least 1 unless the entire output has been written
// already. It is required that `output_size >= 32` when calling this function.
// This function must be called repeatedly until it returns 0.
size_t JxlFastLosslessWriteOutput(JxlFastLosslessFrameState* frame,
                                  unsigned char* output, size_t output_size);

// Frees the provided frame state.
void JxlFastLosslessFreeFrameState(JxlFastLosslessFrameState* frame);

#ifdef __cplusplus
}  // extern "C"
#endif

#if !FJXL_STANDALONE
bool JxlFastLosslessOutputFrame(
    JxlFastLosslessFrameState* frame_state,
    JxlEncoderOutputProcessorWrapper* output_process);
#endif

#endif  // LIB_JXL_ENC_FAST_LOSSLESS_H_

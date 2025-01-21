// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Internal header: WebP decoding parameters and custom IO on buffer
//
// Author: somnath@google.com (Somnath Banerjee)

#ifndef WEBP_DEC_WEBPI_DEC_H_
#define WEBP_DEC_WEBPI_DEC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "src/utils/rescaler_utils.h"
#include "src/dec/vp8_dec.h"
#include "src/webp/decode.h"

//------------------------------------------------------------------------------
// WebPDecParams: Decoding output parameters. Transient internal object.

typedef struct WebPDecParams WebPDecParams;
typedef int (*OutputFunc)(const VP8Io* const io, WebPDecParams* const p);
typedef int (*OutputAlphaFunc)(const VP8Io* const io, WebPDecParams* const p,
                               int expected_num_out_lines);
typedef int (*OutputRowFunc)(WebPDecParams* const p, int y_pos,
                             int max_out_lines);

struct WebPDecParams {
  WebPDecBuffer* output;             // output buffer.
  uint8_t* tmp_y, *tmp_u, *tmp_v;    // cache for the fancy upsampler
                                     // or used for tmp rescaling

  int last_y;                 // coordinate of the line that was last output
  const WebPDecoderOptions* options;  // if not NULL, use alt decoding features

  WebPRescaler* scaler_y, *scaler_u, *scaler_v, *scaler_a;  // rescalers
  void* memory;                  // overall scratch memory for the output work.

  OutputFunc emit;               // output RGB or YUV samples
  OutputAlphaFunc emit_alpha;    // output alpha channel
  OutputRowFunc emit_alpha_row;  // output one line of rescaled alpha values
};

// Should be called first, before any use of the WebPDecParams object.
void WebPResetDecParams(WebPDecParams* const params);

//------------------------------------------------------------------------------
// Header parsing helpers

// Structure storing a description of the RIFF headers.
typedef struct {
  const uint8_t* data;         // input buffer
  size_t data_size;            // input buffer size
  int have_all_data;           // true if all data is known to be available
  size_t offset;               // offset to main data chunk (VP8 or VP8L)
  const uint8_t* alpha_data;   // points to alpha chunk (if present)
  size_t alpha_data_size;      // alpha chunk size
  size_t compressed_size;      // VP8/VP8L compressed data size
  size_t riff_size;            // size of the riff payload (or 0 if absent)
  int is_lossless;             // true if a VP8L chunk is present
} WebPHeaderStructure;

// Skips over all valid chunks prior to the first VP8/VP8L frame header.
// Returns: VP8_STATUS_OK, VP8_STATUS_BITSTREAM_ERROR (invalid header/chunk),
// VP8_STATUS_NOT_ENOUGH_DATA (partial input) or VP8_STATUS_UNSUPPORTED_FEATURE
// in the case of non-decodable features (animation for instance).
// In 'headers', compressed_size, offset, alpha_data, alpha_size, and lossless
// fields are updated appropriately upon success.
VP8StatusCode WebPParseHeaders(WebPHeaderStructure* const headers);

//------------------------------------------------------------------------------
// Misc utils

// Returns true if crop dimensions are within image bounds.
int WebPCheckCropDimensions(int image_width, int image_height,
                            int x, int y, int w, int h);

// Initializes VP8Io with custom setup, io and teardown functions. The default
// hooks will use the supplied 'params' as io->opaque handle.
void WebPInitCustomIo(WebPDecParams* const params, VP8Io* const io);

// Setup crop_xxx fields, mb_w and mb_h in io. 'src_colorspace' refers
// to the *compressed* format, not the output one.
WEBP_NODISCARD int WebPIoInitFromOptions(
    const WebPDecoderOptions* const options, VP8Io* const io,
    WEBP_CSP_MODE src_colorspace);

//------------------------------------------------------------------------------
// Internal functions regarding WebPDecBuffer memory (in buffer.c).
// Don't really need to be externally visible for now.

// Prepare 'buffer' with the requested initial dimensions width/height.
// If no external storage is supplied, initializes buffer by allocating output
// memory and setting up the stride information. Validate the parameters. Return
// an error code in case of problem (no memory, or invalid stride / size /
// dimension / etc.). If *options is not NULL, also verify that the options'
// parameters are valid and apply them to the width/height dimensions of the
// output buffer. This takes cropping / scaling / rotation into account.
// Also incorporates the options->flip flag to flip the buffer parameters if
// needed.
VP8StatusCode WebPAllocateDecBuffer(int width, int height,
                                    const WebPDecoderOptions* const options,
                                    WebPDecBuffer* const buffer);

// Flip buffer vertically by negating the various strides.
VP8StatusCode WebPFlipBuffer(WebPDecBuffer* const buffer);

// Copy 'src' into 'dst' buffer, making sure 'dst' is not marked as owner of the
// memory (still held by 'src'). No pixels are copied.
void WebPCopyDecBuffer(const WebPDecBuffer* const src,
                       WebPDecBuffer* const dst);

// Copy and transfer ownership from src to dst (beware of parameter order!)
void WebPGrabDecBuffer(WebPDecBuffer* const src, WebPDecBuffer* const dst);

// Copy pixels from 'src' into a *preallocated* 'dst' buffer. Returns
// VP8_STATUS_INVALID_PARAM if the 'dst' is not set up correctly for the copy.
VP8StatusCode WebPCopyDecBufferPixels(const WebPDecBuffer* const src,
                                      WebPDecBuffer* const dst);

// Returns true if decoding will be slow with the current configuration
// and bitstream features.
int WebPAvoidSlowMemory(const WebPDecBuffer* const output,
                        const WebPBitstreamFeatures* const features);

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  // WEBP_DEC_WEBPI_DEC_H_

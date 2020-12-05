// Copyright 2010 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
//  Low-level API for VP8 decoder
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WEBP_DEC_VP8_DEC_H_
#define WEBP_DEC_VP8_DEC_H_

#include "src/webp/decode.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Lower-level API
//
// These functions provide fine-grained control of the decoding process.
// The call flow should resemble:
//
//   VP8Io io;
//   VP8InitIo(&io);
//   io.data = data;
//   io.data_size = size;
//   /* customize io's functions (setup()/put()/teardown()) if needed. */
//
//   VP8Decoder* dec = VP8New();
//   int ok = VP8Decode(dec, &io);
//   if (!ok) printf("Error: %s\n", VP8StatusMessage(dec));
//   VP8Delete(dec);
//   return ok;

// Input / Output
typedef struct VP8Io VP8Io;
typedef int (*VP8IoPutHook)(const VP8Io* io);
typedef int (*VP8IoSetupHook)(VP8Io* io);
typedef void (*VP8IoTeardownHook)(const VP8Io* io);

struct VP8Io {
  // set by VP8GetHeaders()
  int width, height;         // picture dimensions, in pixels (invariable).
                             // These are the original, uncropped dimensions.
                             // The actual area passed to put() is stored
                             // in mb_w / mb_h fields.

  // set before calling put()
  int mb_y;                  // position of the current rows (in pixels)
  int mb_w;                  // number of columns in the sample
  int mb_h;                  // number of rows in the sample
  const uint8_t* y, *u, *v;  // rows to copy (in yuv420 format)
  int y_stride;              // row stride for luma
  int uv_stride;             // row stride for chroma

  void* opaque;              // user data

  // called when fresh samples are available. Currently, samples are in
  // YUV420 format, and can be up to width x 24 in size (depending on the
  // in-loop filtering level, e.g.). Should return false in case of error
  // or abort request. The actual size of the area to update is mb_w x mb_h
  // in size, taking cropping into account.
  VP8IoPutHook put;

  // called just before starting to decode the blocks.
  // Must return false in case of setup error, true otherwise. If false is
  // returned, teardown() will NOT be called. But if the setup succeeded
  // and true is returned, then teardown() will always be called afterward.
  VP8IoSetupHook setup;

  // Called just after block decoding is finished (or when an error occurred
  // during put()). Is NOT called if setup() failed.
  VP8IoTeardownHook teardown;

  // this is a recommendation for the user-side yuv->rgb converter. This flag
  // is set when calling setup() hook and can be overwritten by it. It then
  // can be taken into consideration during the put() method.
  int fancy_upsampling;

  // Input buffer.
  size_t data_size;
  const uint8_t* data;

  // If true, in-loop filtering will not be performed even if present in the
  // bitstream. Switching off filtering may speed up decoding at the expense
  // of more visible blocking. Note that output will also be non-compliant
  // with the VP8 specifications.
  int bypass_filtering;

  // Cropping parameters.
  int use_cropping;
  int crop_left, crop_right, crop_top, crop_bottom;

  // Scaling parameters.
  int use_scaling;
  int scaled_width, scaled_height;

  // If non NULL, pointer to the alpha data (if present) corresponding to the
  // start of the current row (That is: it is pre-offset by mb_y and takes
  // cropping into account).
  const uint8_t* a;
};

// Internal, version-checked, entry point
int VP8InitIoInternal(VP8Io* const, int);

// Set the custom IO function pointers and user-data. The setter for IO hooks
// should be called before initiating incremental decoding. Returns true if
// WebPIDecoder object is successfully modified, false otherwise.
int WebPISetIOHooks(WebPIDecoder* const idec,
                    VP8IoPutHook put,
                    VP8IoSetupHook setup,
                    VP8IoTeardownHook teardown,
                    void* user_data);

// Main decoding object. This is an opaque structure.
typedef struct VP8Decoder VP8Decoder;

// Create a new decoder object.
VP8Decoder* VP8New(void);

// Must be called to make sure 'io' is initialized properly.
// Returns false in case of version mismatch. Upon such failure, no other
// decoding function should be called (VP8Decode, VP8GetHeaders, ...)
static WEBP_INLINE int VP8InitIo(VP8Io* const io) {
  return VP8InitIoInternal(io, WEBP_DECODER_ABI_VERSION);
}

// Decode the VP8 frame header. Returns true if ok.
// Note: 'io->data' must be pointing to the start of the VP8 frame header.
int VP8GetHeaders(VP8Decoder* const dec, VP8Io* const io);

// Decode a picture. Will call VP8GetHeaders() if it wasn't done already.
// Returns false in case of error.
int VP8Decode(VP8Decoder* const dec, VP8Io* const io);

// Return current status of the decoder:
VP8StatusCode VP8Status(VP8Decoder* const dec);

// return readable string corresponding to the last status.
const char* VP8StatusMessage(VP8Decoder* const dec);

// Resets the decoder in its initial state, reclaiming memory.
// Not a mandatory call between calls to VP8Decode().
void VP8Clear(VP8Decoder* const dec);

// Destroy the decoder object.
void VP8Delete(VP8Decoder* const dec);

//------------------------------------------------------------------------------
// Miscellaneous VP8/VP8L bitstream probing functions.

// Returns true if the next 3 bytes in data contain the VP8 signature.
WEBP_EXTERN int VP8CheckSignature(const uint8_t* const data, size_t data_size);

// Validates the VP8 data-header and retrieves basic header information viz
// width and height. Returns 0 in case of formatting error. *width/*height
// can be passed NULL.
WEBP_EXTERN int VP8GetInfo(
    const uint8_t* data,
    size_t data_size,    // data available so far
    size_t chunk_size,   // total data size expected in the chunk
    int* const width, int* const height);

// Returns true if the next byte(s) in data is a VP8L signature.
WEBP_EXTERN int VP8LCheckSignature(const uint8_t* const data, size_t size);

// Validates the VP8L data-header and retrieves basic header information viz
// width, height and alpha. Returns 0 in case of formatting error.
// width/height/has_alpha can be passed NULL.
WEBP_EXTERN int VP8LGetInfo(
    const uint8_t* data, size_t data_size,  // data available so far
    int* const width, int* const height, int* const has_alpha);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  // WEBP_DEC_VP8_DEC_H_

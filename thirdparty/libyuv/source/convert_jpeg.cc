/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/convert.h"
#include "libyuv/convert_argb.h"

#ifdef HAVE_JPEG
#include "libyuv/mjpeg_decoder.h"
#endif

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#ifdef HAVE_JPEG
struct I420Buffers {
  uint8_t* y;
  int y_stride;
  uint8_t* u;
  int u_stride;
  uint8_t* v;
  int v_stride;
  int w;
  int h;
};

static void JpegCopyI420(void* opaque,
                         const uint8_t* const* data,
                         const int* strides,
                         int rows) {
  I420Buffers* dest = (I420Buffers*)(opaque);
  I420Copy(data[0], strides[0], data[1], strides[1], data[2], strides[2],
           dest->y, dest->y_stride, dest->u, dest->u_stride, dest->v,
           dest->v_stride, dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->u += ((rows + 1) >> 1) * dest->u_stride;
  dest->v += ((rows + 1) >> 1) * dest->v_stride;
  dest->h -= rows;
}

static void JpegI422ToI420(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  I420Buffers* dest = (I420Buffers*)(opaque);
  I422ToI420(data[0], strides[0], data[1], strides[1], data[2], strides[2],
             dest->y, dest->y_stride, dest->u, dest->u_stride, dest->v,
             dest->v_stride, dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->u += ((rows + 1) >> 1) * dest->u_stride;
  dest->v += ((rows + 1) >> 1) * dest->v_stride;
  dest->h -= rows;
}

static void JpegI444ToI420(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  I420Buffers* dest = (I420Buffers*)(opaque);
  I444ToI420(data[0], strides[0], data[1], strides[1], data[2], strides[2],
             dest->y, dest->y_stride, dest->u, dest->u_stride, dest->v,
             dest->v_stride, dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->u += ((rows + 1) >> 1) * dest->u_stride;
  dest->v += ((rows + 1) >> 1) * dest->v_stride;
  dest->h -= rows;
}

static void JpegI400ToI420(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  I420Buffers* dest = (I420Buffers*)(opaque);
  I400ToI420(data[0], strides[0], dest->y, dest->y_stride, dest->u,
             dest->u_stride, dest->v, dest->v_stride, dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->u += ((rows + 1) >> 1) * dest->u_stride;
  dest->v += ((rows + 1) >> 1) * dest->v_stride;
  dest->h -= rows;
}

// Query size of MJPG in pixels.
LIBYUV_API
int MJPGSize(const uint8_t* src_mjpg,
             size_t src_size_mjpg,
             int* width,
             int* height) {
  MJpegDecoder mjpeg_decoder;
  LIBYUV_BOOL ret = mjpeg_decoder.LoadFrame(src_mjpg, src_size_mjpg);
  if (ret) {
    *width = mjpeg_decoder.GetWidth();
    *height = mjpeg_decoder.GetHeight();
  }
  mjpeg_decoder.UnloadFrame();
  return ret ? 0 : -1;  // -1 for runtime failure.
}

// MJPG (Motion JPeg) to I420
// TODO(fbarchard): review src_width and src_height requirement. dst_width and
// dst_height may be enough.
LIBYUV_API
int MJPGToI420(const uint8_t* src_mjpg,
               size_t src_size_mjpg,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int src_width,
               int src_height,
               int dst_width,
               int dst_height) {
  if (src_size_mjpg == kUnknownDataSize) {
    // ERROR: MJPEG frame size unknown
    return -1;
  }

  // TODO(fbarchard): Port MJpeg to C.
  MJpegDecoder mjpeg_decoder;
  LIBYUV_BOOL ret = mjpeg_decoder.LoadFrame(src_mjpg, src_size_mjpg);
  if (ret && (mjpeg_decoder.GetWidth() != src_width ||
              mjpeg_decoder.GetHeight() != src_height)) {
    // ERROR: MJPEG frame has unexpected dimensions
    mjpeg_decoder.UnloadFrame();
    return 1;  // runtime failure
  }
  if (ret) {
    I420Buffers bufs = {dst_y, dst_stride_y, dst_u,     dst_stride_u,
                        dst_v, dst_stride_v, dst_width, dst_height};
    // YUV420
    if (mjpeg_decoder.GetColorSpace() == MJpegDecoder::kColorSpaceYCbCr &&
        mjpeg_decoder.GetNumComponents() == 3 &&
        mjpeg_decoder.GetVertSampFactor(0) == 2 &&
        mjpeg_decoder.GetHorizSampFactor(0) == 2 &&
        mjpeg_decoder.GetVertSampFactor(1) == 1 &&
        mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
        mjpeg_decoder.GetVertSampFactor(2) == 1 &&
        mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegCopyI420, &bufs, dst_width,
                                           dst_height);
      // YUV422
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceYCbCr &&
               mjpeg_decoder.GetNumComponents() == 3 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 2 &&
               mjpeg_decoder.GetVertSampFactor(1) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
               mjpeg_decoder.GetVertSampFactor(2) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI422ToI420, &bufs, dst_width,
                                           dst_height);
      // YUV444
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceYCbCr &&
               mjpeg_decoder.GetNumComponents() == 3 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 1 &&
               mjpeg_decoder.GetVertSampFactor(1) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
               mjpeg_decoder.GetVertSampFactor(2) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI444ToI420, &bufs, dst_width,
                                           dst_height);
      // YUV400
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceGrayscale &&
               mjpeg_decoder.GetNumComponents() == 1 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI400ToI420, &bufs, dst_width,
                                           dst_height);
    } else {
      // TODO(fbarchard): Implement conversion for any other
      // colorspace/subsample factors that occur in practice. ERROR: Unable to
      // convert MJPEG frame because format is not supported
      mjpeg_decoder.UnloadFrame();
      return 1;
    }
  }
  return ret ? 0 : 1;
}

struct NV21Buffers {
  uint8_t* y;
  int y_stride;
  uint8_t* vu;
  int vu_stride;
  int w;
  int h;
};

static void JpegI420ToNV21(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  NV21Buffers* dest = (NV21Buffers*)(opaque);
  I420ToNV21(data[0], strides[0], data[1], strides[1], data[2], strides[2],
             dest->y, dest->y_stride, dest->vu, dest->vu_stride, dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->vu += ((rows + 1) >> 1) * dest->vu_stride;
  dest->h -= rows;
}

static void JpegI422ToNV21(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  NV21Buffers* dest = (NV21Buffers*)(opaque);
  I422ToNV21(data[0], strides[0], data[1], strides[1], data[2], strides[2],
             dest->y, dest->y_stride, dest->vu, dest->vu_stride, dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->vu += ((rows + 1) >> 1) * dest->vu_stride;
  dest->h -= rows;
}

static void JpegI444ToNV21(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  NV21Buffers* dest = (NV21Buffers*)(opaque);
  I444ToNV21(data[0], strides[0], data[1], strides[1], data[2], strides[2],
             dest->y, dest->y_stride, dest->vu, dest->vu_stride, dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->vu += ((rows + 1) >> 1) * dest->vu_stride;
  dest->h -= rows;
}

static void JpegI400ToNV21(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  NV21Buffers* dest = (NV21Buffers*)(opaque);
  I400ToNV21(data[0], strides[0], dest->y, dest->y_stride, dest->vu,
             dest->vu_stride, dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->vu += ((rows + 1) >> 1) * dest->vu_stride;
  dest->h -= rows;
}

// MJPG (Motion JPeg) to NV21
LIBYUV_API
int MJPGToNV21(const uint8_t* src_mjpg,
               size_t src_size_mjpg,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int src_width,
               int src_height,
               int dst_width,
               int dst_height) {
  if (src_size_mjpg == kUnknownDataSize) {
    // ERROR: MJPEG frame size unknown
    return -1;
  }

  // TODO(fbarchard): Port MJpeg to C.
  MJpegDecoder mjpeg_decoder;
  LIBYUV_BOOL ret = mjpeg_decoder.LoadFrame(src_mjpg, src_size_mjpg);
  if (ret && (mjpeg_decoder.GetWidth() != src_width ||
              mjpeg_decoder.GetHeight() != src_height)) {
    // ERROR: MJPEG frame has unexpected dimensions
    mjpeg_decoder.UnloadFrame();
    return 1;  // runtime failure
  }
  if (ret) {
    NV21Buffers bufs = {dst_y,         dst_stride_y, dst_vu,
                        dst_stride_vu, dst_width,    dst_height};
    // YUV420
    if (mjpeg_decoder.GetColorSpace() == MJpegDecoder::kColorSpaceYCbCr &&
        mjpeg_decoder.GetNumComponents() == 3 &&
        mjpeg_decoder.GetVertSampFactor(0) == 2 &&
        mjpeg_decoder.GetHorizSampFactor(0) == 2 &&
        mjpeg_decoder.GetVertSampFactor(1) == 1 &&
        mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
        mjpeg_decoder.GetVertSampFactor(2) == 1 &&
        mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI420ToNV21, &bufs, dst_width,
                                           dst_height);
      // YUV422
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceYCbCr &&
               mjpeg_decoder.GetNumComponents() == 3 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 2 &&
               mjpeg_decoder.GetVertSampFactor(1) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
               mjpeg_decoder.GetVertSampFactor(2) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI422ToNV21, &bufs, dst_width,
                                           dst_height);
      // YUV444
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceYCbCr &&
               mjpeg_decoder.GetNumComponents() == 3 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 1 &&
               mjpeg_decoder.GetVertSampFactor(1) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
               mjpeg_decoder.GetVertSampFactor(2) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI444ToNV21, &bufs, dst_width,
                                           dst_height);
      // YUV400
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceGrayscale &&
               mjpeg_decoder.GetNumComponents() == 1 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI400ToNV21, &bufs, dst_width,
                                           dst_height);
    } else {
      // Unknown colorspace.
      mjpeg_decoder.UnloadFrame();
      return 1;
    }
  }
  return ret ? 0 : 1;
}

static void JpegI420ToNV12(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  NV21Buffers* dest = (NV21Buffers*)(opaque);
  // Use NV21 with VU swapped.
  I420ToNV21(data[0], strides[0], data[2], strides[2], data[1], strides[1],
             dest->y, dest->y_stride, dest->vu, dest->vu_stride, dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->vu += ((rows + 1) >> 1) * dest->vu_stride;
  dest->h -= rows;
}

static void JpegI422ToNV12(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  NV21Buffers* dest = (NV21Buffers*)(opaque);
  // Use NV21 with VU swapped.
  I422ToNV21(data[0], strides[0], data[2], strides[2], data[1], strides[1],
             dest->y, dest->y_stride, dest->vu, dest->vu_stride, dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->vu += ((rows + 1) >> 1) * dest->vu_stride;
  dest->h -= rows;
}

static void JpegI444ToNV12(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  NV21Buffers* dest = (NV21Buffers*)(opaque);
  // Use NV21 with VU swapped.
  I444ToNV21(data[0], strides[0], data[2], strides[2], data[1], strides[1],
             dest->y, dest->y_stride, dest->vu, dest->vu_stride, dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->vu += ((rows + 1) >> 1) * dest->vu_stride;
  dest->h -= rows;
}

static void JpegI400ToNV12(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  NV21Buffers* dest = (NV21Buffers*)(opaque);
  // Use NV21 since there is no UV plane.
  I400ToNV21(data[0], strides[0], dest->y, dest->y_stride, dest->vu,
             dest->vu_stride, dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->vu += ((rows + 1) >> 1) * dest->vu_stride;
  dest->h -= rows;
}

// MJPG (Motion JPEG) to NV12.
LIBYUV_API
int MJPGToNV12(const uint8_t* sample,
               size_t sample_size,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int src_width,
               int src_height,
               int dst_width,
               int dst_height) {
  if (sample_size == kUnknownDataSize) {
    // ERROR: MJPEG frame size unknown
    return -1;
  }

  // TODO(fbarchard): Port MJpeg to C.
  MJpegDecoder mjpeg_decoder;
  LIBYUV_BOOL ret = mjpeg_decoder.LoadFrame(sample, sample_size);
  if (ret && (mjpeg_decoder.GetWidth() != src_width ||
              mjpeg_decoder.GetHeight() != src_height)) {
    // ERROR: MJPEG frame has unexpected dimensions
    mjpeg_decoder.UnloadFrame();
    return 1;  // runtime failure
  }
  if (ret) {
    // Use NV21Buffers but with UV instead of VU.
    NV21Buffers bufs = {dst_y,         dst_stride_y, dst_uv,
                        dst_stride_uv, dst_width,    dst_height};
    // YUV420
    if (mjpeg_decoder.GetColorSpace() == MJpegDecoder::kColorSpaceYCbCr &&
        mjpeg_decoder.GetNumComponents() == 3 &&
        mjpeg_decoder.GetVertSampFactor(0) == 2 &&
        mjpeg_decoder.GetHorizSampFactor(0) == 2 &&
        mjpeg_decoder.GetVertSampFactor(1) == 1 &&
        mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
        mjpeg_decoder.GetVertSampFactor(2) == 1 &&
        mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI420ToNV12, &bufs, dst_width,
                                           dst_height);
      // YUV422
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceYCbCr &&
               mjpeg_decoder.GetNumComponents() == 3 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 2 &&
               mjpeg_decoder.GetVertSampFactor(1) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
               mjpeg_decoder.GetVertSampFactor(2) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI422ToNV12, &bufs, dst_width,
                                           dst_height);
      // YUV444
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceYCbCr &&
               mjpeg_decoder.GetNumComponents() == 3 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 1 &&
               mjpeg_decoder.GetVertSampFactor(1) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
               mjpeg_decoder.GetVertSampFactor(2) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI444ToNV12, &bufs, dst_width,
                                           dst_height);
      // YUV400
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceGrayscale &&
               mjpeg_decoder.GetNumComponents() == 1 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI400ToNV12, &bufs, dst_width,
                                           dst_height);
    } else {
      // Unknown colorspace.
      mjpeg_decoder.UnloadFrame();
      return 1;
    }
  }
  return ret ? 0 : 1;
}

struct ARGBBuffers {
  uint8_t* argb;
  int argb_stride;
  int w;
  int h;
};

static void JpegI420ToARGB(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  ARGBBuffers* dest = (ARGBBuffers*)(opaque);
  I420ToARGB(data[0], strides[0], data[1], strides[1], data[2], strides[2],
             dest->argb, dest->argb_stride, dest->w, rows);
  dest->argb += rows * dest->argb_stride;
  dest->h -= rows;
}

static void JpegI422ToARGB(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  ARGBBuffers* dest = (ARGBBuffers*)(opaque);
  I422ToARGB(data[0], strides[0], data[1], strides[1], data[2], strides[2],
             dest->argb, dest->argb_stride, dest->w, rows);
  dest->argb += rows * dest->argb_stride;
  dest->h -= rows;
}

static void JpegI444ToARGB(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  ARGBBuffers* dest = (ARGBBuffers*)(opaque);
  I444ToARGB(data[0], strides[0], data[1], strides[1], data[2], strides[2],
             dest->argb, dest->argb_stride, dest->w, rows);
  dest->argb += rows * dest->argb_stride;
  dest->h -= rows;
}

static void JpegI400ToARGB(void* opaque,
                           const uint8_t* const* data,
                           const int* strides,
                           int rows) {
  ARGBBuffers* dest = (ARGBBuffers*)(opaque);
  I400ToARGB(data[0], strides[0], dest->argb, dest->argb_stride, dest->w, rows);
  dest->argb += rows * dest->argb_stride;
  dest->h -= rows;
}

// MJPG (Motion JPeg) to ARGB
// TODO(fbarchard): review src_width and src_height requirement. dst_width and
// dst_height may be enough.
LIBYUV_API
int MJPGToARGB(const uint8_t* src_mjpg,
               size_t src_size_mjpg,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int src_width,
               int src_height,
               int dst_width,
               int dst_height) {
  if (src_size_mjpg == kUnknownDataSize) {
    // ERROR: MJPEG frame size unknown
    return -1;
  }

  // TODO(fbarchard): Port MJpeg to C.
  MJpegDecoder mjpeg_decoder;
  LIBYUV_BOOL ret = mjpeg_decoder.LoadFrame(src_mjpg, src_size_mjpg);
  if (ret && (mjpeg_decoder.GetWidth() != src_width ||
              mjpeg_decoder.GetHeight() != src_height)) {
    // ERROR: MJPEG frame has unexpected dimensions
    mjpeg_decoder.UnloadFrame();
    return 1;  // runtime failure
  }
  if (ret) {
    ARGBBuffers bufs = {dst_argb, dst_stride_argb, dst_width, dst_height};
    // YUV420
    if (mjpeg_decoder.GetColorSpace() == MJpegDecoder::kColorSpaceYCbCr &&
        mjpeg_decoder.GetNumComponents() == 3 &&
        mjpeg_decoder.GetVertSampFactor(0) == 2 &&
        mjpeg_decoder.GetHorizSampFactor(0) == 2 &&
        mjpeg_decoder.GetVertSampFactor(1) == 1 &&
        mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
        mjpeg_decoder.GetVertSampFactor(2) == 1 &&
        mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI420ToARGB, &bufs, dst_width,
                                           dst_height);
      // YUV422
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceYCbCr &&
               mjpeg_decoder.GetNumComponents() == 3 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 2 &&
               mjpeg_decoder.GetVertSampFactor(1) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
               mjpeg_decoder.GetVertSampFactor(2) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI422ToARGB, &bufs, dst_width,
                                           dst_height);
      // YUV444
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceYCbCr &&
               mjpeg_decoder.GetNumComponents() == 3 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 1 &&
               mjpeg_decoder.GetVertSampFactor(1) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
               mjpeg_decoder.GetVertSampFactor(2) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI444ToARGB, &bufs, dst_width,
                                           dst_height);
      // YUV400
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceGrayscale &&
               mjpeg_decoder.GetNumComponents() == 1 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI400ToARGB, &bufs, dst_width,
                                           dst_height);
    } else {
      // TODO(fbarchard): Implement conversion for any other
      // colorspace/subsample factors that occur in practice. ERROR: Unable to
      // convert MJPEG frame because format is not supported
      mjpeg_decoder.UnloadFrame();
      return 1;
    }
  }
  return ret ? 0 : 1;
}

#endif  // HAVE_JPEG

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

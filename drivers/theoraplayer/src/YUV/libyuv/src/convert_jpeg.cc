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

#ifdef HAVE_JPEG
#include "libyuv/mjpeg_decoder.h"
#endif

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#ifdef HAVE_JPEG
struct I420Buffers {
  uint8* y;
  int y_stride;
  uint8* u;
  int u_stride;
  uint8* v;
  int v_stride;
  int w;
  int h;
};

static void JpegCopyI420(void* opaque,
                         const uint8* const* data,
                         const int* strides,
                         int rows) {
  I420Buffers* dest = (I420Buffers*)(opaque);
  I420Copy(data[0], strides[0],
           data[1], strides[1],
           data[2], strides[2],
           dest->y, dest->y_stride,
           dest->u, dest->u_stride,
           dest->v, dest->v_stride,
           dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->u += ((rows + 1) >> 1) * dest->u_stride;
  dest->v += ((rows + 1) >> 1) * dest->v_stride;
  dest->h -= rows;
}

static void JpegI422ToI420(void* opaque,
                           const uint8* const* data,
                           const int* strides,
                           int rows) {
  I420Buffers* dest = (I420Buffers*)(opaque);
  I422ToI420(data[0], strides[0],
             data[1], strides[1],
             data[2], strides[2],
             dest->y, dest->y_stride,
             dest->u, dest->u_stride,
             dest->v, dest->v_stride,
             dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->u += ((rows + 1) >> 1) * dest->u_stride;
  dest->v += ((rows + 1) >> 1) * dest->v_stride;
  dest->h -= rows;
}

static void JpegI444ToI420(void* opaque,
                           const uint8* const* data,
                           const int* strides,
                           int rows) {
  I420Buffers* dest = (I420Buffers*)(opaque);
  I444ToI420(data[0], strides[0],
             data[1], strides[1],
             data[2], strides[2],
             dest->y, dest->y_stride,
             dest->u, dest->u_stride,
             dest->v, dest->v_stride,
             dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->u += ((rows + 1) >> 1) * dest->u_stride;
  dest->v += ((rows + 1) >> 1) * dest->v_stride;
  dest->h -= rows;
}

static void JpegI411ToI420(void* opaque,
                           const uint8* const* data,
                           const int* strides,
                           int rows) {
  I420Buffers* dest = (I420Buffers*)(opaque);
  I411ToI420(data[0], strides[0],
             data[1], strides[1],
             data[2], strides[2],
             dest->y, dest->y_stride,
             dest->u, dest->u_stride,
             dest->v, dest->v_stride,
             dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->u += ((rows + 1) >> 1) * dest->u_stride;
  dest->v += ((rows + 1) >> 1) * dest->v_stride;
  dest->h -= rows;
}

static void JpegI400ToI420(void* opaque,
                           const uint8* const* data,
                           const int* strides,
                           int rows) {
  I420Buffers* dest = (I420Buffers*)(opaque);
  I400ToI420(data[0], strides[0],
             dest->y, dest->y_stride,
             dest->u, dest->u_stride,
             dest->v, dest->v_stride,
             dest->w, rows);
  dest->y += rows * dest->y_stride;
  dest->u += ((rows + 1) >> 1) * dest->u_stride;
  dest->v += ((rows + 1) >> 1) * dest->v_stride;
  dest->h -= rows;
}

// Query size of MJPG in pixels.
LIBYUV_API
int MJPGSize(const uint8* sample, size_t sample_size,
             int* width, int* height) {
  MJpegDecoder mjpeg_decoder;
  LIBYUV_BOOL ret = mjpeg_decoder.LoadFrame(sample, sample_size);
  if (ret) {
    *width = mjpeg_decoder.GetWidth();
    *height = mjpeg_decoder.GetHeight();
  }
  mjpeg_decoder.UnloadFrame();
  return ret ? 0 : -1;  // -1 for runtime failure.
}

// MJPG (Motion JPeg) to I420
// TODO(fbarchard): review w and h requirement. dw and dh may be enough.
LIBYUV_API
int MJPGToI420(const uint8* sample,
               size_t sample_size,
               uint8* y, int y_stride,
               uint8* u, int u_stride,
               uint8* v, int v_stride,
               int w, int h,
               int dw, int dh) {
  if (sample_size == kUnknownDataSize) {
    // ERROR: MJPEG frame size unknown
    return -1;
  }

  // TODO(fbarchard): Port MJpeg to C.
  MJpegDecoder mjpeg_decoder;
  LIBYUV_BOOL ret = mjpeg_decoder.LoadFrame(sample, sample_size);
  if (ret && (mjpeg_decoder.GetWidth() != w ||
              mjpeg_decoder.GetHeight() != h)) {
    // ERROR: MJPEG frame has unexpected dimensions
    mjpeg_decoder.UnloadFrame();
    return 1;  // runtime failure
  }
  if (ret) {
    I420Buffers bufs = { y, y_stride, u, u_stride, v, v_stride, dw, dh };
    // YUV420
    if (mjpeg_decoder.GetColorSpace() ==
            MJpegDecoder::kColorSpaceYCbCr &&
        mjpeg_decoder.GetNumComponents() == 3 &&
        mjpeg_decoder.GetVertSampFactor(0) == 2 &&
        mjpeg_decoder.GetHorizSampFactor(0) == 2 &&
        mjpeg_decoder.GetVertSampFactor(1) == 1 &&
        mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
        mjpeg_decoder.GetVertSampFactor(2) == 1 &&
        mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegCopyI420, &bufs, dw, dh);
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
      ret = mjpeg_decoder.DecodeToCallback(&JpegI422ToI420, &bufs, dw, dh);
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
      ret = mjpeg_decoder.DecodeToCallback(&JpegI444ToI420, &bufs, dw, dh);
    // YUV411
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceYCbCr &&
               mjpeg_decoder.GetNumComponents() == 3 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 4 &&
               mjpeg_decoder.GetVertSampFactor(1) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
               mjpeg_decoder.GetVertSampFactor(2) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI411ToI420, &bufs, dw, dh);
    // YUV400
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceGrayscale &&
               mjpeg_decoder.GetNumComponents() == 1 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI400ToI420, &bufs, dw, dh);
    } else {
      // TODO(fbarchard): Implement conversion for any other colorspace/sample
      // factors that occur in practice. 411 is supported by libjpeg
      // ERROR: Unable to convert MJPEG frame because format is not supported
      mjpeg_decoder.UnloadFrame();
      return 1;
    }
  }
  return ret ? 0 : 1;
}

#ifdef HAVE_JPEG
struct ARGBBuffers {
  uint8* argb;
  int argb_stride;
  int w;
  int h;
};

static void JpegI420ToARGB(void* opaque,
                         const uint8* const* data,
                         const int* strides,
                         int rows) {
  ARGBBuffers* dest = (ARGBBuffers*)(opaque);
  I420ToARGB(data[0], strides[0],
             data[1], strides[1],
             data[2], strides[2],
             dest->argb, dest->argb_stride,
             dest->w, rows);
  dest->argb += rows * dest->argb_stride;
  dest->h -= rows;
}

static void JpegI422ToARGB(void* opaque,
                           const uint8* const* data,
                           const int* strides,
                           int rows) {
  ARGBBuffers* dest = (ARGBBuffers*)(opaque);
  I422ToARGB(data[0], strides[0],
             data[1], strides[1],
             data[2], strides[2],
             dest->argb, dest->argb_stride,
             dest->w, rows);
  dest->argb += rows * dest->argb_stride;
  dest->h -= rows;
}

static void JpegI444ToARGB(void* opaque,
                           const uint8* const* data,
                           const int* strides,
                           int rows) {
  ARGBBuffers* dest = (ARGBBuffers*)(opaque);
  I444ToARGB(data[0], strides[0],
             data[1], strides[1],
             data[2], strides[2],
             dest->argb, dest->argb_stride,
             dest->w, rows);
  dest->argb += rows * dest->argb_stride;
  dest->h -= rows;
}

static void JpegI411ToARGB(void* opaque,
                           const uint8* const* data,
                           const int* strides,
                           int rows) {
  ARGBBuffers* dest = (ARGBBuffers*)(opaque);
  I411ToARGB(data[0], strides[0],
             data[1], strides[1],
             data[2], strides[2],
             dest->argb, dest->argb_stride,
             dest->w, rows);
  dest->argb += rows * dest->argb_stride;
  dest->h -= rows;
}

static void JpegI400ToARGB(void* opaque,
                           const uint8* const* data,
                           const int* strides,
                           int rows) {
  ARGBBuffers* dest = (ARGBBuffers*)(opaque);
  I400ToARGB(data[0], strides[0],
             dest->argb, dest->argb_stride,
             dest->w, rows);
  dest->argb += rows * dest->argb_stride;
  dest->h -= rows;
}

// MJPG (Motion JPeg) to ARGB
// TODO(fbarchard): review w and h requirement. dw and dh may be enough.
LIBYUV_API
int MJPGToARGB(const uint8* sample,
               size_t sample_size,
               uint8* argb, int argb_stride,
               int w, int h,
               int dw, int dh) {
  if (sample_size == kUnknownDataSize) {
    // ERROR: MJPEG frame size unknown
    return -1;
  }

  // TODO(fbarchard): Port MJpeg to C.
  MJpegDecoder mjpeg_decoder;
  LIBYUV_BOOL ret = mjpeg_decoder.LoadFrame(sample, sample_size);
  if (ret && (mjpeg_decoder.GetWidth() != w ||
              mjpeg_decoder.GetHeight() != h)) {
    // ERROR: MJPEG frame has unexpected dimensions
    mjpeg_decoder.UnloadFrame();
    return 1;  // runtime failure
  }
  if (ret) {
    ARGBBuffers bufs = { argb, argb_stride, dw, dh };
    // YUV420
    if (mjpeg_decoder.GetColorSpace() ==
            MJpegDecoder::kColorSpaceYCbCr &&
        mjpeg_decoder.GetNumComponents() == 3 &&
        mjpeg_decoder.GetVertSampFactor(0) == 2 &&
        mjpeg_decoder.GetHorizSampFactor(0) == 2 &&
        mjpeg_decoder.GetVertSampFactor(1) == 1 &&
        mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
        mjpeg_decoder.GetVertSampFactor(2) == 1 &&
        mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI420ToARGB, &bufs, dw, dh);
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
      ret = mjpeg_decoder.DecodeToCallback(&JpegI422ToARGB, &bufs, dw, dh);
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
      ret = mjpeg_decoder.DecodeToCallback(&JpegI444ToARGB, &bufs, dw, dh);
    // YUV411
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceYCbCr &&
               mjpeg_decoder.GetNumComponents() == 3 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 4 &&
               mjpeg_decoder.GetVertSampFactor(1) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(1) == 1 &&
               mjpeg_decoder.GetVertSampFactor(2) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(2) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI411ToARGB, &bufs, dw, dh);
    // YUV400
    } else if (mjpeg_decoder.GetColorSpace() ==
                   MJpegDecoder::kColorSpaceGrayscale &&
               mjpeg_decoder.GetNumComponents() == 1 &&
               mjpeg_decoder.GetVertSampFactor(0) == 1 &&
               mjpeg_decoder.GetHorizSampFactor(0) == 1) {
      ret = mjpeg_decoder.DecodeToCallback(&JpegI400ToARGB, &bufs, dw, dh);
    } else {
      // TODO(fbarchard): Implement conversion for any other colorspace/sample
      // factors that occur in practice. 411 is supported by libjpeg
      // ERROR: Unable to convert MJPEG frame because format is not supported
      mjpeg_decoder.UnloadFrame();
      return 1;
    }
  }
  return ret ? 0 : 1;
}
#endif

#endif

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

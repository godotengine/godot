/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_MJPEG_DECODER_H_  // NOLINT
#define INCLUDE_LIBYUV_MJPEG_DECODER_H_

#include "libyuv/basic_types.h"

#ifdef __cplusplus
// NOTE: For a simplified public API use convert.h MJPGToI420().

struct jpeg_common_struct;
struct jpeg_decompress_struct;
struct jpeg_source_mgr;

namespace libyuv {

#ifdef __cplusplus
extern "C" {
#endif

LIBYUV_BOOL ValidateJpeg(const uint8* sample, size_t sample_size);

#ifdef __cplusplus
}  // extern "C"
#endif

static const uint32 kUnknownDataSize = 0xFFFFFFFF;

enum JpegSubsamplingType {
  kJpegYuv420,
  kJpegYuv422,
  kJpegYuv411,
  kJpegYuv444,
  kJpegYuv400,
  kJpegUnknown
};

struct SetJmpErrorMgr;

// MJPEG ("Motion JPEG") is a pseudo-standard video codec where the frames are
// simply independent JPEG images with a fixed huffman table (which is omitted).
// It is rarely used in video transmission, but is common as a camera capture
// format, especially in Logitech devices. This class implements a decoder for
// MJPEG frames.
//
// See http://tools.ietf.org/html/rfc2435
class LIBYUV_API MJpegDecoder {
 public:
  typedef void (*CallbackFunction)(void* opaque,
                                   const uint8* const* data,
                                   const int* strides,
                                   int rows);

  static const int kColorSpaceUnknown;
  static const int kColorSpaceGrayscale;
  static const int kColorSpaceRgb;
  static const int kColorSpaceYCbCr;
  static const int kColorSpaceCMYK;
  static const int kColorSpaceYCCK;

  MJpegDecoder();
  ~MJpegDecoder();

  // Loads a new frame, reads its headers, and determines the uncompressed
  // image format.
  // Returns LIBYUV_TRUE if image looks valid and format is supported.
  // If return value is LIBYUV_TRUE, then the values for all the following
  // getters are populated.
  // src_len is the size of the compressed mjpeg frame in bytes.
  LIBYUV_BOOL LoadFrame(const uint8* src, size_t src_len);

  // Returns width of the last loaded frame in pixels.
  int GetWidth();

  // Returns height of the last loaded frame in pixels.
  int GetHeight();

  // Returns format of the last loaded frame. The return value is one of the
  // kColorSpace* constants.
  int GetColorSpace();

  // Number of color components in the color space.
  int GetNumComponents();

  // Sample factors of the n-th component.
  int GetHorizSampFactor(int component);

  int GetVertSampFactor(int component);

  int GetHorizSubSampFactor(int component);

  int GetVertSubSampFactor(int component);

  // Public for testability.
  int GetImageScanlinesPerImcuRow();

  // Public for testability.
  int GetComponentScanlinesPerImcuRow(int component);

  // Width of a component in bytes.
  int GetComponentWidth(int component);

  // Height of a component.
  int GetComponentHeight(int component);

  // Width of a component in bytes with padding for DCTSIZE. Public for testing.
  int GetComponentStride(int component);

  // Size of a component in bytes.
  int GetComponentSize(int component);

  // Call this after LoadFrame() if you decide you don't want to decode it
  // after all.
  LIBYUV_BOOL UnloadFrame();

  // Decodes the entire image into a one-buffer-per-color-component format.
  // dst_width must match exactly. dst_height must be <= to image height; if
  // less, the image is cropped. "planes" must have size equal to at least
  // GetNumComponents() and they must point to non-overlapping buffers of size
  // at least GetComponentSize(i). The pointers in planes are incremented
  // to point to after the end of the written data.
  // TODO(fbarchard): Add dst_x, dst_y to allow specific rect to be decoded.
  LIBYUV_BOOL DecodeToBuffers(uint8** planes, int dst_width, int dst_height);

  // Decodes the entire image and passes the data via repeated calls to a
  // callback function. Each call will get the data for a whole number of
  // image scanlines.
  // TODO(fbarchard): Add dst_x, dst_y to allow specific rect to be decoded.
  LIBYUV_BOOL DecodeToCallback(CallbackFunction fn, void* opaque,
                        int dst_width, int dst_height);

  // The helper function which recognizes the jpeg sub-sampling type.
  static JpegSubsamplingType JpegSubsamplingTypeHelper(
     int* subsample_x, int* subsample_y, int number_of_components);

 private:
  struct Buffer {
    const uint8* data;
    int len;
  };

  struct BufferVector {
    Buffer* buffers;
    int len;
    int pos;
  };

  // Methods that are passed to jpeglib.
  static int fill_input_buffer(jpeg_decompress_struct* cinfo);
  static void init_source(jpeg_decompress_struct* cinfo);
  static void skip_input_data(jpeg_decompress_struct* cinfo,
                              long num_bytes);  // NOLINT
  static void term_source(jpeg_decompress_struct* cinfo);

  static void ErrorHandler(jpeg_common_struct* cinfo);

  void AllocOutputBuffers(int num_outbufs);
  void DestroyOutputBuffers();

  LIBYUV_BOOL StartDecode();
  LIBYUV_BOOL FinishDecode();

  void SetScanlinePointers(uint8** data);
  LIBYUV_BOOL DecodeImcuRow();

  int GetComponentScanlinePadding(int component);

  // A buffer holding the input data for a frame.
  Buffer buf_;
  BufferVector buf_vec_;

  jpeg_decompress_struct* decompress_struct_;
  jpeg_source_mgr* source_mgr_;
  SetJmpErrorMgr* error_mgr_;

  // LIBYUV_TRUE iff at least one component has scanline padding. (i.e.,
  // GetComponentScanlinePadding() != 0.)
  LIBYUV_BOOL has_scanline_padding_;

  // Temporaries used to point to scanline outputs.
  int num_outbufs_;  // Outermost size of all arrays below.
  uint8*** scanlines_;
  int* scanlines_sizes_;
  // Temporary buffer used for decoding when we can't decode directly to the
  // output buffers. Large enough for just one iMCU row.
  uint8** databuf_;
  int* databuf_strides_;
};

}  // namespace libyuv

#endif  //  __cplusplus
#endif  // INCLUDE_LIBYUV_MJPEG_DECODER_H_  NOLINT

/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/mjpeg_decoder.h"

#ifdef HAVE_JPEG
#include <assert.h>

#if !defined(__pnacl__) && !defined(__CLR_VER) && !defined(COVERAGE_ENABLED) &&\
    !defined(TARGET_IPHONE_SIMULATOR)
// Must be included before jpeglib.
#include <setjmp.h>
#define HAVE_SETJMP
#endif
struct FILE;  // For jpeglib.h.

// C++ build requires extern C for jpeg internals.
#ifdef __cplusplus
extern "C" {
#endif

#include <jpeglib.h>

#ifdef __cplusplus
}  // extern "C"
#endif

#include "libyuv/planar_functions.h"  // For CopyPlane().

namespace libyuv {

#ifdef HAVE_SETJMP
struct SetJmpErrorMgr {
  jpeg_error_mgr base;  // Must be at the top
  jmp_buf setjmp_buffer;
};
#endif

const int MJpegDecoder::kColorSpaceUnknown = JCS_UNKNOWN;
const int MJpegDecoder::kColorSpaceGrayscale = JCS_GRAYSCALE;
const int MJpegDecoder::kColorSpaceRgb = JCS_RGB;
const int MJpegDecoder::kColorSpaceYCbCr = JCS_YCbCr;
const int MJpegDecoder::kColorSpaceCMYK = JCS_CMYK;
const int MJpegDecoder::kColorSpaceYCCK = JCS_YCCK;

MJpegDecoder::MJpegDecoder()
    : has_scanline_padding_(LIBYUV_FALSE),
      num_outbufs_(0),
      scanlines_(NULL),
      scanlines_sizes_(NULL),
      databuf_(NULL),
      databuf_strides_(NULL) {
  decompress_struct_ = new jpeg_decompress_struct;
  source_mgr_ = new jpeg_source_mgr;
#ifdef HAVE_SETJMP
  error_mgr_ = new SetJmpErrorMgr;
  decompress_struct_->err = jpeg_std_error(&error_mgr_->base);
  // Override standard exit()-based error handler.
  error_mgr_->base.error_exit = &ErrorHandler;
#endif
  decompress_struct_->client_data = NULL;
  source_mgr_->init_source = &init_source;
  source_mgr_->fill_input_buffer = &fill_input_buffer;
  source_mgr_->skip_input_data = &skip_input_data;
  source_mgr_->resync_to_restart = &jpeg_resync_to_restart;
  source_mgr_->term_source = &term_source;
  jpeg_create_decompress(decompress_struct_);
  decompress_struct_->src = source_mgr_;
  buf_vec_.buffers = &buf_;
  buf_vec_.len = 1;
}

MJpegDecoder::~MJpegDecoder() {
  jpeg_destroy_decompress(decompress_struct_);
  delete decompress_struct_;
  delete source_mgr_;
#ifdef HAVE_SETJMP
  delete error_mgr_;
#endif
  DestroyOutputBuffers();
}

LIBYUV_BOOL MJpegDecoder::LoadFrame(const uint8* src, size_t src_len) {
  if (!ValidateJpeg(src, src_len)) {
    return LIBYUV_FALSE;
  }

  buf_.data = src;
  buf_.len = (int)(src_len);
  buf_vec_.pos = 0;
  decompress_struct_->client_data = &buf_vec_;
#ifdef HAVE_SETJMP
  if (setjmp(error_mgr_->setjmp_buffer)) {
    // We called jpeg_read_header, it experienced an error, and we called
    // longjmp() and rewound the stack to here. Return error.
    return LIBYUV_FALSE;
  }
#endif
  if (jpeg_read_header(decompress_struct_, TRUE) != JPEG_HEADER_OK) {
    // ERROR: Bad MJPEG header
    return LIBYUV_FALSE;
  }
  AllocOutputBuffers(GetNumComponents());
  for (int i = 0; i < num_outbufs_; ++i) {
    int scanlines_size = GetComponentScanlinesPerImcuRow(i);
    if (scanlines_sizes_[i] != scanlines_size) {
      if (scanlines_[i]) {
        delete scanlines_[i];
      }
      scanlines_[i] = new uint8* [scanlines_size];
      scanlines_sizes_[i] = scanlines_size;
    }

    // We allocate padding for the final scanline to pad it up to DCTSIZE bytes
    // to avoid memory errors, since jpeglib only reads full MCUs blocks. For
    // the preceding scanlines, the padding is not needed/wanted because the
    // following addresses will already be valid (they are the initial bytes of
    // the next scanline) and will be overwritten when jpeglib writes out that
    // next scanline.
    int databuf_stride = GetComponentStride(i);
    int databuf_size = scanlines_size * databuf_stride;
    if (databuf_strides_[i] != databuf_stride) {
      if (databuf_[i]) {
        delete databuf_[i];
      }
      databuf_[i] = new uint8[databuf_size];
      databuf_strides_[i] = databuf_stride;
    }

    if (GetComponentStride(i) != GetComponentWidth(i)) {
      has_scanline_padding_ = LIBYUV_TRUE;
    }
  }
  return LIBYUV_TRUE;
}

static int DivideAndRoundUp(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

static int DivideAndRoundDown(int numerator, int denominator) {
  return numerator / denominator;
}

// Returns width of the last loaded frame.
int MJpegDecoder::GetWidth() {
  return decompress_struct_->image_width;
}

// Returns height of the last loaded frame.
int MJpegDecoder::GetHeight() {
  return decompress_struct_->image_height;
}

// Returns format of the last loaded frame. The return value is one of the
// kColorSpace* constants.
int MJpegDecoder::GetColorSpace() {
  return decompress_struct_->jpeg_color_space;
}

// Number of color components in the color space.
int MJpegDecoder::GetNumComponents() {
  return decompress_struct_->num_components;
}

// Sample factors of the n-th component.
int MJpegDecoder::GetHorizSampFactor(int component) {
  return decompress_struct_->comp_info[component].h_samp_factor;
}

int MJpegDecoder::GetVertSampFactor(int component) {
  return decompress_struct_->comp_info[component].v_samp_factor;
}

int MJpegDecoder::GetHorizSubSampFactor(int component) {
  return decompress_struct_->max_h_samp_factor /
      GetHorizSampFactor(component);
}

int MJpegDecoder::GetVertSubSampFactor(int component) {
  return decompress_struct_->max_v_samp_factor /
      GetVertSampFactor(component);
}

int MJpegDecoder::GetImageScanlinesPerImcuRow() {
  return decompress_struct_->max_v_samp_factor * DCTSIZE;
}

int MJpegDecoder::GetComponentScanlinesPerImcuRow(int component) {
  int vs = GetVertSubSampFactor(component);
  return DivideAndRoundUp(GetImageScanlinesPerImcuRow(), vs);
}

int MJpegDecoder::GetComponentWidth(int component) {
  int hs = GetHorizSubSampFactor(component);
  return DivideAndRoundUp(GetWidth(), hs);
}

int MJpegDecoder::GetComponentHeight(int component) {
  int vs = GetVertSubSampFactor(component);
  return DivideAndRoundUp(GetHeight(), vs);
}

// Get width in bytes padded out to a multiple of DCTSIZE
int MJpegDecoder::GetComponentStride(int component) {
  return (GetComponentWidth(component) + DCTSIZE - 1) & ~(DCTSIZE - 1);
}

int MJpegDecoder::GetComponentSize(int component) {
  return GetComponentWidth(component) * GetComponentHeight(component);
}

LIBYUV_BOOL MJpegDecoder::UnloadFrame() {
#ifdef HAVE_SETJMP
  if (setjmp(error_mgr_->setjmp_buffer)) {
    // We called jpeg_abort_decompress, it experienced an error, and we called
    // longjmp() and rewound the stack to here. Return error.
    return LIBYUV_FALSE;
  }
#endif
  jpeg_abort_decompress(decompress_struct_);
  return LIBYUV_TRUE;
}

// TODO(fbarchard): Allow rectangle to be specified: x, y, width, height.
LIBYUV_BOOL MJpegDecoder::DecodeToBuffers(
    uint8** planes, int dst_width, int dst_height) {
  if (dst_width != GetWidth() ||
      dst_height > GetHeight()) {
    // ERROR: Bad dimensions
    return LIBYUV_FALSE;
  }
#ifdef HAVE_SETJMP
  if (setjmp(error_mgr_->setjmp_buffer)) {
    // We called into jpeglib, it experienced an error sometime during this
    // function call, and we called longjmp() and rewound the stack to here.
    // Return error.
    return LIBYUV_FALSE;
  }
#endif
  if (!StartDecode()) {
    return LIBYUV_FALSE;
  }
  SetScanlinePointers(databuf_);
  int lines_left = dst_height;
  // Compute amount of lines to skip to implement vertical crop.
  // TODO(fbarchard): Ensure skip is a multiple of maximum component
  // subsample. ie 2
  int skip = (GetHeight() - dst_height) / 2;
  if (skip > 0) {
    // There is no API to skip lines in the output data, so we read them
    // into the temp buffer.
    while (skip >= GetImageScanlinesPerImcuRow()) {
      if (!DecodeImcuRow()) {
        FinishDecode();
        return LIBYUV_FALSE;
      }
      skip -= GetImageScanlinesPerImcuRow();
    }
    if (skip > 0) {
      // Have a partial iMCU row left over to skip. Must read it and then
      // copy the parts we want into the destination.
      if (!DecodeImcuRow()) {
        FinishDecode();
        return LIBYUV_FALSE;
      }
      for (int i = 0; i < num_outbufs_; ++i) {
        // TODO(fbarchard): Compute skip to avoid this
        assert(skip % GetVertSubSampFactor(i) == 0);
        int rows_to_skip =
            DivideAndRoundDown(skip, GetVertSubSampFactor(i));
        int scanlines_to_copy = GetComponentScanlinesPerImcuRow(i) -
                                rows_to_skip;
        int data_to_skip = rows_to_skip * GetComponentStride(i);
        CopyPlane(databuf_[i] + data_to_skip, GetComponentStride(i),
                  planes[i], GetComponentWidth(i),
                  GetComponentWidth(i), scanlines_to_copy);
        planes[i] += scanlines_to_copy * GetComponentWidth(i);
      }
      lines_left -= (GetImageScanlinesPerImcuRow() - skip);
    }
  }

  // Read full MCUs but cropped horizontally
  for (; lines_left > GetImageScanlinesPerImcuRow();
         lines_left -= GetImageScanlinesPerImcuRow()) {
    if (!DecodeImcuRow()) {
      FinishDecode();
      return LIBYUV_FALSE;
    }
    for (int i = 0; i < num_outbufs_; ++i) {
      int scanlines_to_copy = GetComponentScanlinesPerImcuRow(i);
      CopyPlane(databuf_[i], GetComponentStride(i),
                planes[i], GetComponentWidth(i),
                GetComponentWidth(i), scanlines_to_copy);
      planes[i] += scanlines_to_copy * GetComponentWidth(i);
    }
  }

  if (lines_left > 0) {
    // Have a partial iMCU row left over to decode.
    if (!DecodeImcuRow()) {
      FinishDecode();
      return LIBYUV_FALSE;
    }
    for (int i = 0; i < num_outbufs_; ++i) {
      int scanlines_to_copy =
          DivideAndRoundUp(lines_left, GetVertSubSampFactor(i));
      CopyPlane(databuf_[i], GetComponentStride(i),
                planes[i], GetComponentWidth(i),
                GetComponentWidth(i), scanlines_to_copy);
      planes[i] += scanlines_to_copy * GetComponentWidth(i);
    }
  }
  return FinishDecode();
}

LIBYUV_BOOL MJpegDecoder::DecodeToCallback(CallbackFunction fn, void* opaque,
    int dst_width, int dst_height) {
  if (dst_width != GetWidth() ||
      dst_height > GetHeight()) {
    // ERROR: Bad dimensions
    return LIBYUV_FALSE;
  }
#ifdef HAVE_SETJMP
  if (setjmp(error_mgr_->setjmp_buffer)) {
    // We called into jpeglib, it experienced an error sometime during this
    // function call, and we called longjmp() and rewound the stack to here.
    // Return error.
    return LIBYUV_FALSE;
  }
#endif
  if (!StartDecode()) {
    return LIBYUV_FALSE;
  }
  SetScanlinePointers(databuf_);
  int lines_left = dst_height;
  // TODO(fbarchard): Compute amount of lines to skip to implement vertical crop
  int skip = (GetHeight() - dst_height) / 2;
  if (skip > 0) {
    while (skip >= GetImageScanlinesPerImcuRow()) {
      if (!DecodeImcuRow()) {
        FinishDecode();
        return LIBYUV_FALSE;
      }
      skip -= GetImageScanlinesPerImcuRow();
    }
    if (skip > 0) {
      // Have a partial iMCU row left over to skip.
      if (!DecodeImcuRow()) {
        FinishDecode();
        return LIBYUV_FALSE;
      }
      for (int i = 0; i < num_outbufs_; ++i) {
        // TODO(fbarchard): Compute skip to avoid this
        assert(skip % GetVertSubSampFactor(i) == 0);
        int rows_to_skip = DivideAndRoundDown(skip, GetVertSubSampFactor(i));
        int data_to_skip = rows_to_skip * GetComponentStride(i);
        // Change our own data buffer pointers so we can pass them to the
        // callback.
        databuf_[i] += data_to_skip;
      }
      int scanlines_to_copy = GetImageScanlinesPerImcuRow() - skip;
      (*fn)(opaque, databuf_, databuf_strides_, scanlines_to_copy);
      // Now change them back.
      for (int i = 0; i < num_outbufs_; ++i) {
        int rows_to_skip = DivideAndRoundDown(skip, GetVertSubSampFactor(i));
        int data_to_skip = rows_to_skip * GetComponentStride(i);
        databuf_[i] -= data_to_skip;
      }
      lines_left -= scanlines_to_copy;
    }
  }
  // Read full MCUs until we get to the crop point.
  for (; lines_left >= GetImageScanlinesPerImcuRow();
         lines_left -= GetImageScanlinesPerImcuRow()) {
    if (!DecodeImcuRow()) {
      FinishDecode();
      return LIBYUV_FALSE;
    }
    (*fn)(opaque, databuf_, databuf_strides_, GetImageScanlinesPerImcuRow());
  }
  if (lines_left > 0) {
    // Have a partial iMCU row left over to decode.
    if (!DecodeImcuRow()) {
      FinishDecode();
      return LIBYUV_FALSE;
    }
    (*fn)(opaque, databuf_, databuf_strides_, lines_left);
  }
  return FinishDecode();
}

void MJpegDecoder::init_source(j_decompress_ptr cinfo) {
  fill_input_buffer(cinfo);
}

boolean MJpegDecoder::fill_input_buffer(j_decompress_ptr cinfo) {
  BufferVector* buf_vec = (BufferVector*)(cinfo->client_data);
  if (buf_vec->pos >= buf_vec->len) {
    assert(0 && "No more data");
    // ERROR: No more data
    return FALSE;
  }
  cinfo->src->next_input_byte = buf_vec->buffers[buf_vec->pos].data;
  cinfo->src->bytes_in_buffer = buf_vec->buffers[buf_vec->pos].len;
  ++buf_vec->pos;
  return TRUE;
}

void MJpegDecoder::skip_input_data(j_decompress_ptr cinfo,
                                   long num_bytes) {  // NOLINT
  cinfo->src->next_input_byte += num_bytes;
}

void MJpegDecoder::term_source(j_decompress_ptr cinfo) {
  // Nothing to do.
}

#ifdef HAVE_SETJMP
void MJpegDecoder::ErrorHandler(j_common_ptr cinfo) {
  // This is called when a jpeglib command experiences an error. Unfortunately
  // jpeglib's error handling model is not very flexible, because it expects the
  // error handler to not return--i.e., it wants the program to terminate. To
  // recover from errors we use setjmp() as shown in their example. setjmp() is
  // C's implementation for the "call with current continuation" functionality
  // seen in some functional programming languages.
  // A formatted message can be output, but is unsafe for release.
#ifdef DEBUG
  char buf[JMSG_LENGTH_MAX];
  (*cinfo->err->format_message)(cinfo, buf);
  // ERROR: Error in jpeglib: buf
#endif

  SetJmpErrorMgr* mgr = (SetJmpErrorMgr*)(cinfo->err);
  // This rewinds the call stack to the point of the corresponding setjmp()
  // and causes it to return (for a second time) with value 1.
  longjmp(mgr->setjmp_buffer, 1);
}
#endif

void MJpegDecoder::AllocOutputBuffers(int num_outbufs) {
  if (num_outbufs != num_outbufs_) {
    // We could perhaps optimize this case to resize the output buffers without
    // necessarily having to delete and recreate each one, but it's not worth
    // it.
    DestroyOutputBuffers();

    scanlines_ = new uint8** [num_outbufs];
    scanlines_sizes_ = new int[num_outbufs];
    databuf_ = new uint8* [num_outbufs];
    databuf_strides_ = new int[num_outbufs];

    for (int i = 0; i < num_outbufs; ++i) {
      scanlines_[i] = NULL;
      scanlines_sizes_[i] = 0;
      databuf_[i] = NULL;
      databuf_strides_[i] = 0;
    }

    num_outbufs_ = num_outbufs;
  }
}

void MJpegDecoder::DestroyOutputBuffers() {
  for (int i = 0; i < num_outbufs_; ++i) {
    delete [] scanlines_[i];
    delete [] databuf_[i];
  }
  delete [] scanlines_;
  delete [] databuf_;
  delete [] scanlines_sizes_;
  delete [] databuf_strides_;
  scanlines_ = NULL;
  databuf_ = NULL;
  scanlines_sizes_ = NULL;
  databuf_strides_ = NULL;
  num_outbufs_ = 0;
}

// JDCT_IFAST and do_block_smoothing improve performance substantially.
LIBYUV_BOOL MJpegDecoder::StartDecode() {
  decompress_struct_->raw_data_out = TRUE;
  decompress_struct_->dct_method = JDCT_IFAST;  // JDCT_ISLOW is default
  decompress_struct_->dither_mode = JDITHER_NONE;
  // Not applicable to 'raw':
  decompress_struct_->do_fancy_upsampling = LIBYUV_FALSE;
  // Only for buffered mode:
  decompress_struct_->enable_2pass_quant = LIBYUV_FALSE;
  // Blocky but fast:
  decompress_struct_->do_block_smoothing = LIBYUV_FALSE;

  if (!jpeg_start_decompress(decompress_struct_)) {
    // ERROR: Couldn't start JPEG decompressor";
    return LIBYUV_FALSE;
  }
  return LIBYUV_TRUE;
}

LIBYUV_BOOL MJpegDecoder::FinishDecode() {
  // jpeglib considers it an error if we finish without decoding the whole
  // image, so we call "abort" rather than "finish".
  jpeg_abort_decompress(decompress_struct_);
  return LIBYUV_TRUE;
}

void MJpegDecoder::SetScanlinePointers(uint8** data) {
  for (int i = 0; i < num_outbufs_; ++i) {
    uint8* data_i = data[i];
    for (int j = 0; j < scanlines_sizes_[i]; ++j) {
      scanlines_[i][j] = data_i;
      data_i += GetComponentStride(i);
    }
  }
}

inline LIBYUV_BOOL MJpegDecoder::DecodeImcuRow() {
  return (unsigned int)(GetImageScanlinesPerImcuRow()) ==
      jpeg_read_raw_data(decompress_struct_,
                         scanlines_,
                         GetImageScanlinesPerImcuRow());
}

// The helper function which recognizes the jpeg sub-sampling type.
JpegSubsamplingType MJpegDecoder::JpegSubsamplingTypeHelper(
    int* subsample_x, int* subsample_y, int number_of_components) {
  if (number_of_components == 3) {  // Color images.
    if (subsample_x[0] == 1 && subsample_y[0] == 1 &&
        subsample_x[1] == 2 && subsample_y[1] == 2 &&
        subsample_x[2] == 2 && subsample_y[2] == 2) {
      return kJpegYuv420;
    } else if (subsample_x[0] == 1 && subsample_y[0] == 1 &&
        subsample_x[1] == 2 && subsample_y[1] == 1 &&
        subsample_x[2] == 2 && subsample_y[2] == 1) {
      return kJpegYuv422;
    } else if (subsample_x[0] == 1 && subsample_y[0] == 1 &&
        subsample_x[1] == 1 && subsample_y[1] == 1 &&
        subsample_x[2] == 1 && subsample_y[2] == 1) {
      return kJpegYuv444;
    }
  } else if (number_of_components == 1) {  // Grey-scale images.
    if (subsample_x[0] == 1 && subsample_y[0] == 1) {
      return kJpegYuv400;
    }
  }
  return kJpegUnknown;
}

}  // namespace libyuv
#endif  // HAVE_JPEG


// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_EXTERNAL_IMAGE_H_
#define LIB_JXL_DEC_EXTERNAL_IMAGE_H_

// Interleaved image for color transforms and Codec.

#include <jxl/types.h>
#include <stddef.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"

namespace jxl {

// Maximum number of channels for the ConvertChannelsToExternal function.
const size_t kConvertMaxChannels = 4;

// Converts a list of channels to an interleaved image, applying transformations
// when needed.
// The input channels are given as a (non-const!) array of channel pointers and
// interleaved in that order.
//
// Note: if a pointer in channels[] is nullptr, a 1.0 value will be used
// instead. This is useful for handling when a user requests an alpha channel
// from an image that doesn't have one. The first channel in the list may not
// be nullptr, since it is used to determine the image size.
Status ConvertChannelsToExternal(const ImageF* in_channels[],
                                 size_t num_channels, size_t bits_per_sample,
                                 bool float_out, JxlEndianness endianness,
                                 size_t stride, jxl::ThreadPool* pool,
                                 void* out_image, size_t out_size,
                                 const PixelCallback& out_callback,
                                 jxl::Orientation undo_orientation);

// Converts ib to interleaved void* pixel buffer with the given format.
// bits_per_sample: must be 16 or 32 if float_out is true, and at most 16
// if it is false. No bit packing is done.
// num_channels: must be 1, 2, 3 or 4 for gray, gray+alpha, RGB, RGB+alpha.
// This supports the features needed for the C API and does not perform
// color space conversion.
// TODO(lode): support rectangle crop.
// stride_out is output scanline size in bytes, must be >=
// output_xsize * output_bytes_per_pixel.
// undo_orientation is an EXIF orientation to undo. Depending on the
// orientation, the output xsize and ysize are swapped compared to input
// xsize and ysize.
Status ConvertToExternal(const jxl::ImageBundle& ib, size_t bits_per_sample,
                         bool float_out, size_t num_channels,
                         JxlEndianness endianness, size_t stride_out,
                         jxl::ThreadPool* thread_pool, void* out_image,
                         size_t out_size, const PixelCallback& out_callback,
                         jxl::Orientation undo_orientation,
                         bool unpremul_alpha = false);

}  // namespace jxl

#endif  // LIB_JXL_DEC_EXTERNAL_IMAGE_H_

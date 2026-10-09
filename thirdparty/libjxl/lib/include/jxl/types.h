/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_common
 * @{
 * @file types.h
 * @brief Data types for the JPEG XL API, for both encoding and decoding.
 */

#ifndef JXL_TYPES_H_
#define JXL_TYPES_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A portable @c bool replacement.
 *
 * ::JXL_BOOL is a "documentation" type: actually it is @c int, but in API it
 * denotes a type, whose only values are ::JXL_TRUE and ::JXL_FALSE.
 */
#define JXL_BOOL int
/** Portable @c true replacement. */
#define JXL_TRUE 1
/** Portable @c false replacement. */
#define JXL_FALSE 0
/** Converts of bool-like value to either ::JXL_TRUE or ::JXL_FALSE. */
#define TO_JXL_BOOL(C) (!!(C) ? JXL_TRUE : JXL_FALSE)
/** Converts JXL_BOOL to C++ bool. */
#define FROM_JXL_BOOL(C) (static_cast<bool>(C))

/** Data type for the sample values per channel per pixel.
 */
typedef enum {
  /** Use 32-bit single-precision floating point values, with range 0.0-1.0
   * (within gamut, may go outside this range for wide color gamut). Floating
   * point output, either ::JXL_TYPE_FLOAT or ::JXL_TYPE_FLOAT16, is recommended
   * for HDR and wide gamut images when color profile conversion is required. */
  JXL_TYPE_FLOAT = 0,

  /** Use type uint8_t. May clip wide color gamut data.
   */
  JXL_TYPE_UINT8 = 2,

  /** Use type uint16_t. May clip wide color gamut data.
   */
  JXL_TYPE_UINT16 = 3,

  /** Use 16-bit IEEE 754 half-precision floating point values */
  JXL_TYPE_FLOAT16 = 5,
} JxlDataType;

/** Ordering of multi-byte data.
 */
typedef enum {
  /** Use the endianness of the system, either little endian or big endian,
   * without forcing either specific endianness. Do not use if pixel data
   * should be exported to a well defined format.
   */
  JXL_NATIVE_ENDIAN = 0,
  /** Force little endian */
  JXL_LITTLE_ENDIAN = 1,
  /** Force big endian */
  JXL_BIG_ENDIAN = 2,
} JxlEndianness;

/** Data type for the sample values per channel per pixel for the output buffer
 * for pixels. This is not necessarily the same as the data type encoded in the
 * codestream. The channels are interleaved per pixel. The pixels are
 * organized row by row, left to right, top to bottom.
 * TODO(lode): support different channel orders if needed (RGB, BGR, ...)
 */
typedef struct {
  /** Amount of channels available in a pixel buffer.
   * 1: single-channel data, e.g. grayscale or a single extra channel
   * 2: single-channel + alpha
   * 3: trichromatic, e.g. RGB
   * 4: trichromatic + alpha
   * TODO(lode): this needs finetuning. It is not yet defined how the user
   * chooses output color space. CMYK+alpha needs 5 channels.
   */
  uint32_t num_channels;

  /** Data type of each channel.
   */
  JxlDataType data_type;

  /** Whether multi-byte data types are represented in big endian or little
   * endian format. This applies to ::JXL_TYPE_UINT16 and ::JXL_TYPE_FLOAT.
   */
  JxlEndianness endianness;

  /** Align scanlines to a multiple of align bytes, or 0 to require no
   * alignment at all (which has the same effect as value 1)
   */
  size_t align;
} JxlPixelFormat;

/** Settings for the interpretation of UINT input and output buffers.
 *  (buffers using a FLOAT data type are not affected by this)
 */
typedef enum {
  /** This is the default setting, where the encoder expects the input pixels
   * to use the full range of the pixel format data type (e.g. for UINT16, the
   * input range is 0 .. 65535 and the value 65535 is mapped to 1.0 when
   * converting to float), and the decoder uses the full range to output
   * pixels. If the bit depth in the basic info is different from this, the
   * encoder expects the values to be rescaled accordingly (e.g. multiplied by
   * 65535/4095 for a 12-bit image using UINT16 input data type). */
  JXL_BIT_DEPTH_FROM_PIXEL_FORMAT = 0,

  /** If this setting is selected, the encoder expects the input pixels to be
   * in the range defined by the bits_per_sample value of the basic info (e.g.
   * for 12-bit images using UINT16 input data types, the allowed range is
   * 0 .. 4095 and the value 4095 is mapped to 1.0 when converting to float),
   * and the decoder outputs pixels in this range. */
  JXL_BIT_DEPTH_FROM_CODESTREAM = 1,

  /** This setting can only be used in the decoder to select a custom range for
   * pixel output */
  JXL_BIT_DEPTH_CUSTOM = 2,
} JxlBitDepthType;

/** Data type for describing the interpretation of the input and output buffers
 * in terms of the range of allowed input and output pixel values. */
typedef struct {
  /** Bit depth setting, see comment on @ref JxlBitDepthType */
  JxlBitDepthType type;

  /** Custom bits per sample */
  uint32_t bits_per_sample;

  /** Custom exponent bits per sample */
  uint32_t exponent_bits_per_sample;
} JxlBitDepth;

/** Data type holding the 4-character type name of an ISOBMFF box.
 */
typedef char JxlBoxType[4];

#ifdef __cplusplus
}
#endif

#endif /* JXL_TYPES_H_ */

/** @}*/

/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_color
 * @{
 * @file color_encoding.h
 * @brief Color Encoding definitions used by JPEG XL.
 * All CIE units are for the standard 1931 2 degree observer.
 */

#ifndef JXL_COLOR_ENCODING_H_
#define JXL_COLOR_ENCODING_H_

#ifdef __cplusplus
extern "C" {
#endif

/** Color space of the image data. */
typedef enum {
  /** Tristimulus RGB */
  JXL_COLOR_SPACE_RGB,
  /** Luminance based, the primaries in @ref JxlColorEncoding must be ignored.
   * This value implies that num_color_channels in @ref JxlBasicInfo is 1, any
   * other value implies num_color_channels is 3. */
  JXL_COLOR_SPACE_GRAY,
  /** XYB (opsin) color space */
  JXL_COLOR_SPACE_XYB,
  /** None of the other table entries describe the color space appropriately */
  JXL_COLOR_SPACE_UNKNOWN,
} JxlColorSpace;

/** Built-in white points for color encoding. When decoding, the numerical xy
 * white point value can be read from the @ref JxlColorEncoding white_point
 * field regardless of the enum value. When encoding, enum values except
 * ::JXL_WHITE_POINT_CUSTOM override the numerical fields. Some enum values
 * match a subset of CICP (Rec. ITU-T H.273 | ISO/IEC 23091-2:2019(E)), however
 * the white point and RGB primaries are separate enums here.
 */
typedef enum {
  /** CIE Standard Illuminant D65: 0.3127, 0.3290 */
  JXL_WHITE_POINT_D65 = 1,
  /** White point must be read from the @ref JxlColorEncoding white_point field,
   * or as ICC profile. This enum value is not an exact match of the
   * corresponding CICP value. */
  JXL_WHITE_POINT_CUSTOM = 2,
  /** CIE Standard Illuminant E (equal-energy): 1/3, 1/3 */
  JXL_WHITE_POINT_E = 10,
  /** DCI-P3 from SMPTE RP 431-2: 0.314, 0.351 */
  JXL_WHITE_POINT_DCI = 11,
} JxlWhitePoint;

/** Built-in primaries for color encoding. When decoding, the primaries can be
 * read from the @ref JxlColorEncoding primaries_red_xy, primaries_green_xy and
 * primaries_blue_xy fields regardless of the enum value. When encoding, the
 * enum values except ::JXL_PRIMARIES_CUSTOM override the numerical fields.
 * Some enum values match a subset of CICP (Rec. ITU-T H.273 | ISO/IEC
 * 23091-2:2019(E)), however the white point and RGB primaries are separate
 * enums here.
 */
typedef enum {
  /** The CIE xy values of the red, green and blue primaries are: 0.639998686,
     0.330010138; 0.300003784, 0.600003357; 0.150002046, 0.059997204 */
  JXL_PRIMARIES_SRGB = 1,
  /** Primaries must be read from the @ref JxlColorEncoding primaries_red_xy,
   * primaries_green_xy and primaries_blue_xy fields, or as ICC profile. This
   * enum value is not an exact match of the corresponding CICP value. */
  JXL_PRIMARIES_CUSTOM = 2,
  /** As specified in Rec. ITU-R BT.2100-1 */
  JXL_PRIMARIES_2100 = 9,
  /** As specified in SMPTE RP 431-2 */
  JXL_PRIMARIES_P3 = 11,
} JxlPrimaries;

/** Built-in transfer functions for color encoding. Enum values match a subset
 * of CICP (Rec. ITU-T H.273 | ISO/IEC 23091-2:2019(E)) unless specified
 * otherwise. */
typedef enum {
  /** As specified in ITU-R BT.709-6 */
  JXL_TRANSFER_FUNCTION_709 = 1,
  /** None of the other table entries describe the transfer function. */
  JXL_TRANSFER_FUNCTION_UNKNOWN = 2,
  /** The gamma exponent is 1 */
  JXL_TRANSFER_FUNCTION_LINEAR = 8,
  /** As specified in IEC 61966-2-1 sRGB */
  JXL_TRANSFER_FUNCTION_SRGB = 13,
  /** As specified in SMPTE ST 2084 */
  JXL_TRANSFER_FUNCTION_PQ = 16,
  /** As specified in SMPTE ST 428-1 */
  JXL_TRANSFER_FUNCTION_DCI = 17,
  /** As specified in Rec. ITU-R BT.2100-1 (HLG) */
  JXL_TRANSFER_FUNCTION_HLG = 18,
  /** Transfer function follows power law given by the gamma value in @ref
     JxlColorEncoding. Not a CICP value. */
  JXL_TRANSFER_FUNCTION_GAMMA = 65535,
} JxlTransferFunction;

/** Rendering intent for color encoding, as specified in ISO 15076-1:2010 */
typedef enum {
  /** vendor-specific */
  JXL_RENDERING_INTENT_PERCEPTUAL = 0,
  /** media-relative */
  JXL_RENDERING_INTENT_RELATIVE,
  /** vendor-specific */
  JXL_RENDERING_INTENT_SATURATION,
  /** ICC-absolute */
  JXL_RENDERING_INTENT_ABSOLUTE,
} JxlRenderingIntent;

/** Color encoding of the image as structured information.
 */
typedef struct {
  /** Color space of the image data.
   */
  JxlColorSpace color_space;

  /** Built-in white point. If this value is ::JXL_WHITE_POINT_CUSTOM, must
   * use the numerical white point values from white_point_xy.
   */
  JxlWhitePoint white_point;

  /** Numerical whitepoint values in CIE xy space. */
  double white_point_xy[2];

  /** Built-in RGB primaries. If this value is ::JXL_PRIMARIES_CUSTOM, must
   * use the numerical primaries values below. This field and the custom values
   * below are unused and must be ignored if the color space is
   * ::JXL_COLOR_SPACE_GRAY or ::JXL_COLOR_SPACE_XYB.
   */
  JxlPrimaries primaries;

  /** Numerical red primary values in CIE xy space. */
  double primaries_red_xy[2];

  /** Numerical green primary values in CIE xy space. */
  double primaries_green_xy[2];

  /** Numerical blue primary values in CIE xy space. */
  double primaries_blue_xy[2];

  /** Transfer function if have_gamma is 0 */
  JxlTransferFunction transfer_function;

  /** Gamma value used when transfer_function is @ref
   * JXL_TRANSFER_FUNCTION_GAMMA
   */
  double gamma;

  /** Rendering intent defined for the color profile. */
  JxlRenderingIntent rendering_intent;
} JxlColorEncoding;

#ifdef __cplusplus
}
#endif

#endif /* JXL_COLOR_ENCODING_H_ */

/** @}*/

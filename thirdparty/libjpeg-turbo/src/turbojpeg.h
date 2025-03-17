/*
 * Copyright (C)2009-2015, 2017, 2020-2024 D. R. Commander.
 *                                         All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the libjpeg-turbo Project nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS",
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __TURBOJPEG_H__
#define __TURBOJPEG_H__

#include <stddef.h>

#if defined(_WIN32) && defined(DLLDEFINE)
#define DLLEXPORT  __declspec(dllexport)
#else
#define DLLEXPORT
#endif
#define DLLCALL


/**
 * @addtogroup TurboJPEG
 * TurboJPEG API.  This API provides an interface for generating, decoding, and
 * transforming planar YUV and JPEG images in memory.
 *
 * @anchor YUVnotes
 * YUV Image Format Notes
 * ----------------------
 * Technically, the JPEG format uses the YCbCr colorspace (which is technically
 * not a colorspace but a color transform), but per the convention of the
 * digital video community, the TurboJPEG API uses "YUV" to refer to an image
 * format consisting of Y, Cb, and Cr image planes.
 *
 * Each plane is simply a 2D array of bytes, each byte representing the value
 * of one of the components (Y, Cb, or Cr) at a particular location in the
 * image.  The width and height of each plane are determined by the image
 * width, height, and level of chrominance subsampling.  The luminance plane
 * width is the image width padded to the nearest multiple of the horizontal
 * subsampling factor (1 in the case of 4:4:4, grayscale, 4:4:0, or 4:4:1; 2 in
 * the case of 4:2:2 or 4:2:0; 4 in the case of 4:1:1.)  Similarly, the
 * luminance plane height is the image height padded to the nearest multiple of
 * the vertical subsampling factor (1 in the case of 4:4:4, 4:2:2, grayscale,
 * or 4:1:1; 2 in the case of 4:2:0 or 4:4:0; 4 in the case of 4:4:1.)  This is
 * irrespective of any additional padding that may be specified as an argument
 * to the various YUV functions.  The chrominance plane width is equal to the
 * luminance plane width divided by the horizontal subsampling factor, and the
 * chrominance plane height is equal to the luminance plane height divided by
 * the vertical subsampling factor.
 *
 * For example, if the source image is 35 x 35 pixels and 4:2:2 subsampling is
 * used, then the luminance plane would be 36 x 35 bytes, and each of the
 * chrominance planes would be 18 x 35 bytes.  If you specify a row alignment
 * of 4 bytes on top of this, then the luminance plane would be 36 x 35 bytes,
 * and each of the chrominance planes would be 20 x 35 bytes.
 *
 * @{
 */


/**
 * The number of initialization options
 */
#define TJ_NUMINIT  3

/**
 * Initialization options
 */
enum TJINIT {
  /**
   * Initialize the TurboJPEG instance for compression.
   */
  TJINIT_COMPRESS,
  /**
   * Initialize the TurboJPEG instance for decompression.
   */
  TJINIT_DECOMPRESS,
  /**
   * Initialize the TurboJPEG instance for lossless transformation (both
   * compression and decompression.)
   */
  TJINIT_TRANSFORM
};


/**
 * The number of chrominance subsampling options
 */
#define TJ_NUMSAMP  7

/**
 * Chrominance subsampling options
 *
 * When pixels are converted from RGB to YCbCr (see #TJCS_YCbCr) or from CMYK
 * to YCCK (see #TJCS_YCCK) as part of the JPEG compression process, some of
 * the Cb and Cr (chrominance) components can be discarded or averaged together
 * to produce a smaller image with little perceptible loss of image quality.
 * (The human eye is more sensitive to small changes in brightness than to
 * small changes in color.)  This is called "chrominance subsampling".
 */
enum TJSAMP {
  /**
   * 4:4:4 chrominance subsampling (no chrominance subsampling)
   *
   * The JPEG or YUV image will contain one chrominance component for every
   * pixel in the source image.
   */
  TJSAMP_444,
  /**
   * 4:2:2 chrominance subsampling
   *
   * The JPEG or YUV image will contain one chrominance component for every 2x1
   * block of pixels in the source image.
   */
  TJSAMP_422,
  /**
   * 4:2:0 chrominance subsampling
   *
   * The JPEG or YUV image will contain one chrominance component for every 2x2
   * block of pixels in the source image.
   */
  TJSAMP_420,
  /**
   * Grayscale
   *
   * The JPEG or YUV image will contain no chrominance components.
   */
  TJSAMP_GRAY,
  /**
   * 4:4:0 chrominance subsampling
   *
   * The JPEG or YUV image will contain one chrominance component for every 1x2
   * block of pixels in the source image.
   *
   * @note 4:4:0 subsampling is not fully accelerated in libjpeg-turbo.
   */
  TJSAMP_440,
  /**
   * 4:1:1 chrominance subsampling
   *
   * The JPEG or YUV image will contain one chrominance component for every 4x1
   * block of pixels in the source image.  All else being equal, a JPEG image
   * with 4:1:1 subsampling is almost exactly the same size as a JPEG image
   * with 4:2:0 subsampling, and in the aggregate, both subsampling methods
   * produce approximately the same perceptual quality.  However, 4:1:1 is
   * better able to reproduce sharp horizontal features.
   *
   * @note 4:1:1 subsampling is not fully accelerated in libjpeg-turbo.
   */
  TJSAMP_411,
  /**
   * 4:4:1 chrominance subsampling
   *
   * The JPEG or YUV image will contain one chrominance component for every 1x4
   * block of pixels in the source image.  All else being equal, a JPEG image
   * with 4:4:1 subsampling is almost exactly the same size as a JPEG image
   * with 4:2:0 subsampling, and in the aggregate, both subsampling methods
   * produce approximately the same perceptual quality.  However, 4:4:1 is
   * better able to reproduce sharp vertical features.
   *
   * @note 4:4:1 subsampling is not fully accelerated in libjpeg-turbo.
   */
  TJSAMP_441,
  /**
   * Unknown subsampling
   *
   * The JPEG image uses an unusual type of chrominance subsampling.  Such
   * images can be decompressed into packed-pixel images, but they cannot be
   * - decompressed into planar YUV images,
   * - losslessly transformed if #TJXOPT_CROP is specified and #TJXOPT_GRAY is
   * not specified, or
   * - partially decompressed using a cropping region.
   */
  TJSAMP_UNKNOWN = -1
};

/**
 * iMCU width (in pixels) for a given level of chrominance subsampling
 *
 * In a typical lossy JPEG image, 8x8 blocks of DCT coefficients for each
 * component are interleaved in a single scan.  If the image uses chrominance
 * subsampling, then multiple luminance blocks are stored together, followed by
 * a single block for each chrominance component.  The minimum set of
 * full-resolution luminance block(s) and corresponding (possibly subsampled)
 * chrominance blocks necessary to represent at least one DCT block per
 * component is called a "Minimum Coded Unit" or "MCU".  (For example, an MCU
 * in an interleaved lossy JPEG image that uses 4:2:2 subsampling consists of
 * two luminance blocks followed by one block for each chrominance component.)
 * In a non-interleaved lossy JPEG image, each component is stored in a
 * separate scan, and an MCU is a single DCT block, so we use the term "iMCU"
 * (interleaved MCU) to refer to the equivalent of an MCU in an interleaved
 * JPEG image.  For the common case of interleaved JPEG images, an iMCU is the
 * same as an MCU.
 *
 * iMCU sizes:
 * - 8x8 for no subsampling or grayscale
 * - 16x8 for 4:2:2
 * - 8x16 for 4:4:0
 * - 16x16 for 4:2:0
 * - 32x8 for 4:1:1
 * - 8x32 for 4:4:1
 */
static const int tjMCUWidth[TJ_NUMSAMP]  = { 8, 16, 16, 8, 8, 32, 8 };

/**
 * iMCU height (in pixels) for a given level of chrominance subsampling
 *
 * In a typical lossy JPEG image, 8x8 blocks of DCT coefficients for each
 * component are interleaved in a single scan.  If the image uses chrominance
 * subsampling, then multiple luminance blocks are stored together, followed by
 * a single block for each chrominance component.  The minimum set of
 * full-resolution luminance block(s) and corresponding (possibly subsampled)
 * chrominance blocks necessary to represent at least one DCT block per
 * component is called a "Minimum Coded Unit" or "MCU".  (For example, an MCU
 * in an interleaved lossy JPEG image that uses 4:2:2 subsampling consists of
 * two luminance blocks followed by one block for each chrominance component.)
 * In a non-interleaved lossy JPEG image, each component is stored in a
 * separate scan, and an MCU is a single DCT block, so we use the term "iMCU"
 * (interleaved MCU) to refer to the equivalent of an MCU in an interleaved
 * JPEG image.  For the common case of interleaved JPEG images, an iMCU is the
 * same as an MCU.
 *
 * iMCU sizes:
 * - 8x8 for no subsampling or grayscale
 * - 16x8 for 4:2:2
 * - 8x16 for 4:4:0
 * - 16x16 for 4:2:0
 * - 32x8 for 4:1:1
 * - 8x32 for 4:4:1
 */
static const int tjMCUHeight[TJ_NUMSAMP] = { 8, 8, 16, 8, 16, 8, 32 };


/**
 * The number of pixel formats
 */
#define TJ_NUMPF  12

/**
 * Pixel formats
 */
enum TJPF {
  /**
   * RGB pixel format
   *
   * The red, green, and blue components in the image are stored in 3-sample
   * pixels in the order R, G, B from lowest to highest memory address within
   * each pixel.
   */
  TJPF_RGB,
  /**
   * BGR pixel format
   *
   * The red, green, and blue components in the image are stored in 3-sample
   * pixels in the order B, G, R from lowest to highest memory address within
   * each pixel.
   */
  TJPF_BGR,
  /**
   * RGBX pixel format
   *
   * The red, green, and blue components in the image are stored in 4-sample
   * pixels in the order R, G, B from lowest to highest memory address within
   * each pixel.  The X component is ignored when compressing/encoding and
   * undefined when decompressing/decoding.
   */
  TJPF_RGBX,
  /**
   * BGRX pixel format
   *
   * The red, green, and blue components in the image are stored in 4-sample
   * pixels in the order B, G, R from lowest to highest memory address within
   * each pixel.  The X component is ignored when compressing/encoding and
   * undefined when decompressing/decoding.
   */
  TJPF_BGRX,
  /**
   * XBGR pixel format
   *
   * The red, green, and blue components in the image are stored in 4-sample
   * pixels in the order R, G, B from highest to lowest memory address within
   * each pixel.  The X component is ignored when compressing/encoding and
   * undefined when decompressing/decoding.
   */
  TJPF_XBGR,
  /**
   * XRGB pixel format
   *
   * The red, green, and blue components in the image are stored in 4-sample
   * pixels in the order B, G, R from highest to lowest memory address within
   * each pixel.  The X component is ignored when compressing/encoding and
   * undefined when decompressing/decoding.
   */
  TJPF_XRGB,
  /**
   * Grayscale pixel format
   *
   * Each 1-sample pixel represents a luminance (brightness) level from 0 to
   * the maximum sample value (which is, for instance, 255 for 8-bit samples or
   * 4095 for 12-bit samples or 65535 for 16-bit samples.)
   */
  TJPF_GRAY,
  /**
   * RGBA pixel format
   *
   * This is the same as @ref TJPF_RGBX, except that when
   * decompressing/decoding, the X component is guaranteed to be equal to the
   * maximum sample value, which can be interpreted as an opaque alpha channel.
   */
  TJPF_RGBA,
  /**
   * BGRA pixel format
   *
   * This is the same as @ref TJPF_BGRX, except that when
   * decompressing/decoding, the X component is guaranteed to be equal to the
   * maximum sample value, which can be interpreted as an opaque alpha channel.
   */
  TJPF_BGRA,
  /**
   * ABGR pixel format
   *
   * This is the same as @ref TJPF_XBGR, except that when
   * decompressing/decoding, the X component is guaranteed to be equal to the
   * maximum sample value, which can be interpreted as an opaque alpha channel.
   */
  TJPF_ABGR,
  /**
   * ARGB pixel format
   *
   * This is the same as @ref TJPF_XRGB, except that when
   * decompressing/decoding, the X component is guaranteed to be equal to the
   * maximum sample value, which can be interpreted as an opaque alpha channel.
   */
  TJPF_ARGB,
  /**
   * CMYK pixel format
   *
   * Unlike RGB, which is an additive color model used primarily for display,
   * CMYK (Cyan/Magenta/Yellow/Key) is a subtractive color model used primarily
   * for printing.  In the CMYK color model, the value of each color component
   * typically corresponds to an amount of cyan, magenta, yellow, or black ink
   * that is applied to a white background.  In order to convert between CMYK
   * and RGB, it is necessary to use a color management system (CMS.)  A CMS
   * will attempt to map colors within the printer's gamut to perceptually
   * similar colors in the display's gamut and vice versa, but the mapping is
   * typically not 1:1 or reversible, nor can it be defined with a simple
   * formula.  Thus, such a conversion is out of scope for a codec library.
   * However, the TurboJPEG API allows for compressing packed-pixel CMYK images
   * into YCCK JPEG images (see #TJCS_YCCK) and decompressing YCCK JPEG images
   * into packed-pixel CMYK images.
   */
  TJPF_CMYK,
  /**
   * Unknown pixel format
   *
   * Currently this is only used by #tj3LoadImage8(), #tj3LoadImage12(), and
   * #tj3LoadImage16().
   */
  TJPF_UNKNOWN = -1
};

/**
 * Red offset (in samples) for a given pixel format
 *
 * This specifies the number of samples that the red component is offset from
 * the start of the pixel.  For instance, if an 8-bit-per-component pixel of
 * format TJPF_BGRX is stored in `unsigned char pixel[]`, then the red
 * component is `pixel[tjRedOffset[TJPF_BGRX]]`.  The offset is -1 if the pixel
 * format does not have a red component.
 */
static const int tjRedOffset[TJ_NUMPF] = {
  0, 2, 0, 2, 3, 1, -1, 0, 2, 3, 1, -1
};
/**
 * Green offset (in samples) for a given pixel format
 *
 * This specifies the number of samples that the green component is offset from
 * the start of the pixel.  For instance, if an 8-bit-per-component pixel of
 * format TJPF_BGRX is stored in `unsigned char pixel[]`, then the green
 * component is `pixel[tjGreenOffset[TJPF_BGRX]]`.  The offset is -1 if the
 * pixel format does not have a green component.
 */
static const int tjGreenOffset[TJ_NUMPF] = {
  1, 1, 1, 1, 2, 2, -1, 1, 1, 2, 2, -1
};
/**
 * Blue offset (in samples) for a given pixel format
 *
 * This specifies the number of samples that the blue component is offset from
 * the start of the pixel.  For instance, if an 8-bit-per-component pixel of
 * format TJPF_BGRX is stored in `unsigned char pixel[]`, then the blue
 * component is `pixel[tjBlueOffset[TJPF_BGRX]]`.  The offset is -1 if the
 * pixel format does not have a blue component.
 */
static const int tjBlueOffset[TJ_NUMPF] = {
  2, 0, 2, 0, 1, 3, -1, 2, 0, 1, 3, -1
};
/**
 * Alpha offset (in samples) for a given pixel format
 *
 * This specifies the number of samples that the alpha component is offset from
 * the start of the pixel.  For instance, if an 8-bit-per-component pixel of
 * format TJPF_BGRA is stored in `unsigned char pixel[]`, then the alpha
 * component is `pixel[tjAlphaOffset[TJPF_BGRA]]`.  The offset is -1 if the
 * pixel format does not have an alpha component.
 */
static const int tjAlphaOffset[TJ_NUMPF] = {
  -1, -1, -1, -1, -1, -1, -1, 3, 3, 0, 0, -1
};
/**
 * Pixel size (in samples) for a given pixel format
 */
static const int tjPixelSize[TJ_NUMPF] = {
  3, 3, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4
};


/**
 * The number of JPEG colorspaces
 */
#define TJ_NUMCS  5

/**
 * JPEG colorspaces
 */
enum TJCS {
  /**
   * RGB colorspace
   *
   * When generating the JPEG image, the R, G, and B components in the source
   * image are reordered into image planes, but no colorspace conversion or
   * subsampling is performed.  RGB JPEG images can be generated from and
   * decompressed to packed-pixel images with any of the extended RGB or
   * grayscale pixel formats, but they cannot be generated from or
   * decompressed to planar YUV images.
   */
  TJCS_RGB,
  /**
   * YCbCr colorspace
   *
   * YCbCr is not an absolute colorspace but rather a mathematical
   * transformation of RGB designed solely for storage and transmission.  YCbCr
   * images must be converted to RGB before they can be displayed.  In the
   * YCbCr colorspace, the Y (luminance) component represents the black & white
   * portion of the original image, and the Cb and Cr (chrominance) components
   * represent the color portion of the original image.  Historically, the
   * analog equivalent of this transformation allowed the same signal to be
   * displayed to both black & white and color televisions, but JPEG images use
   * YCbCr primarily because it allows the color data to be optionally
   * subsampled in order to reduce network and disk usage.  YCbCr is the most
   * common JPEG colorspace, and YCbCr JPEG images can be generated from and
   * decompressed to packed-pixel images with any of the extended RGB or
   * grayscale pixel formats.  YCbCr JPEG images can also be generated from
   * and decompressed to planar YUV images.
   */
  TJCS_YCbCr,
  /**
   * Grayscale colorspace
   *
   * The JPEG image retains only the luminance data (Y component), and any
   * color data from the source image is discarded.  Grayscale JPEG images can
   * be generated from and decompressed to packed-pixel images with any of the
   * extended RGB or grayscale pixel formats, or they can be generated from
   * and decompressed to planar YUV images.
   */
  TJCS_GRAY,
  /**
   * CMYK colorspace
   *
   * When generating the JPEG image, the C, M, Y, and K components in the
   * source image are reordered into image planes, but no colorspace conversion
   * or subsampling is performed.  CMYK JPEG images can only be generated from
   * and decompressed to packed-pixel images with the CMYK pixel format.
   */
  TJCS_CMYK,
  /**
   * YCCK colorspace
   *
   * YCCK (AKA "YCbCrK") is not an absolute colorspace but rather a
   * mathematical transformation of CMYK designed solely for storage and
   * transmission.  It is to CMYK as YCbCr is to RGB.  CMYK pixels can be
   * reversibly transformed into YCCK, and as with YCbCr, the chrominance
   * components in the YCCK pixels can be subsampled without incurring major
   * perceptual loss.  YCCK JPEG images can only be generated from and
   * decompressed to packed-pixel images with the CMYK pixel format.
   */
  TJCS_YCCK
};


/**
 * Parameters
 */
enum TJPARAM {
  /**
   * Error handling behavior
   *
   * **Value**
   * - `0` *[default]* Allow the current compression/decompression/transform
   * operation to complete unless a fatal error is encountered.
   * - `1` Immediately discontinue the current
   * compression/decompression/transform operation if a warning (non-fatal
   * error) occurs.
   */
  TJPARAM_STOPONWARNING,
  /**
   * Row order in packed-pixel source/destination images
   *
   * **Value**
   * - `0` *[default]* top-down (X11) order
   * - `1` bottom-up (Windows, OpenGL) order
   */
  TJPARAM_BOTTOMUP,
  /**
   * JPEG destination buffer (re)allocation [compression, lossless
   * transformation]
   *
   * **Value**
   * - `0` *[default]* Attempt to allocate or reallocate the JPEG destination
   * buffer as needed.
   * - `1` Generate an error if the JPEG destination buffer is invalid or too
   * small.
   */
  TJPARAM_NOREALLOC,
  /**
   * Perceptual quality of lossy JPEG images [compression only]
   *
   * **Value**
   * - `1`-`100` (`1` = worst quality but best compression, `100` = best
   * quality but worst compression) *[no default; must be explicitly
   * specified]*
   */
  TJPARAM_QUALITY,
  /**
   * Chrominance subsampling level
   *
   * The JPEG or YUV image uses (decompression, decoding) or will use (lossy
   * compression, encoding) the specified level of chrominance subsampling.
   *
   * **Value**
   * - One of the @ref TJSAMP "chrominance subsampling options" *[no default;
   * must be explicitly specified for lossy compression, encoding, and
   * decoding]*
   */
  TJPARAM_SUBSAMP,
  /**
   * JPEG width (in pixels) [decompression only, read-only]
   */
  TJPARAM_JPEGWIDTH,
  /**
   * JPEG height (in pixels) [decompression only, read-only]
   */
  TJPARAM_JPEGHEIGHT,
  /**
   * Data precision (bits per sample)
   *
   * The JPEG image uses (decompression) or will use (lossless compression) the
   * specified number of bits per sample.  This parameter also specifies the
   * target data precision when loading a PBMPLUS file with #tj3LoadImage8(),
   * #tj3LoadImage12(), or #tj3LoadImage16() and the source data precision when
   * saving a PBMPLUS file with #tj3SaveImage8(), #tj3SaveImage12(), or
   * #tj3SaveImage16().
   *
   * The data precision is the number of bits in the maximum sample value,
   * which may not be the same as the width of the data type used to store the
   * sample.
   *
   * **Value**
   * - `8` or `12` for lossy JPEG images; `2` to `16` for lossless JPEG and
   * PBMPLUS images
   *
   * 12-bit JPEG data precision implies #TJPARAM_OPTIMIZE unless
   * #TJPARAM_ARITHMETIC is set.
   */
  TJPARAM_PRECISION,
  /**
   * JPEG colorspace
   *
   * The JPEG image uses (decompression) or will use (lossy compression) the
   * specified colorspace.
   *
   * **Value**
   * - One of the @ref TJCS "JPEG colorspaces" *[default for lossy compression:
   * automatically selected based on the subsampling level and pixel format]*
   */
  TJPARAM_COLORSPACE,
  /**
   * Chrominance upsampling algorithm [lossy decompression only]
   *
   * **Value**
   * - `0` *[default]* Use smooth upsampling when decompressing a JPEG image
   * that was generated using chrominance subsampling.  This creates a smooth
   * transition between neighboring chrominance components in order to reduce
   * upsampling artifacts in the decompressed image.
   * - `1` Use the fastest chrominance upsampling algorithm available, which
   * may combine upsampling with color conversion.
   */
  TJPARAM_FASTUPSAMPLE,
  /**
   * DCT/IDCT algorithm [lossy compression and decompression]
   *
   * **Value**
   * - `0` *[default]* Use the most accurate DCT/IDCT algorithm available.
   * - `1` Use the fastest DCT/IDCT algorithm available.
   *
   * This parameter is provided mainly for backward compatibility with libjpeg,
   * which historically implemented several different DCT/IDCT algorithms
   * because of performance limitations with 1990s CPUs.  In the libjpeg-turbo
   * implementation of the TurboJPEG API:
   * - The "fast" and "accurate" DCT/IDCT algorithms perform similarly on
   * modern x86/x86-64 CPUs that support AVX2 instructions.
   * - The "fast" algorithm is generally only about 5-15% faster than the
   * "accurate" algorithm on other types of CPUs.
   * - The difference in accuracy between the "fast" and "accurate" algorithms
   * is the most pronounced at JPEG quality levels above 90 and tends to be
   * more pronounced with decompression than with compression.
   * - For JPEG quality levels above 97, the "fast" algorithm degrades and is
   * not fully accelerated, so it is slower than the "accurate" algorithm.
   */
  TJPARAM_FASTDCT,
  /**
   * Huffman table optimization [lossy compression, lossless transformation]
   *
   * **Value**
   * - `0` *[default]* The JPEG image will use the default Huffman tables.
   * - `1` Optimal Huffman tables will be computed for the JPEG image.  For
   * lossless transformation, this can also be specified using
   * #TJXOPT_OPTIMIZE.
   *
   * Huffman table optimization improves compression slightly (generally 5% or
   * less), but it reduces compression performance considerably.
   */
  TJPARAM_OPTIMIZE,
  /**
   * Progressive JPEG
   *
   * In a progressive JPEG image, the DCT coefficients are split across
   * multiple "scans" of increasing quality.  Thus, a low-quality scan
   * containing the lowest-frequency DCT coefficients can be transmitted first
   * and refined with subsequent higher-quality scans containing
   * higher-frequency DCT coefficients.  When using Huffman entropy coding, the
   * progressive JPEG format also provides an "end-of-bands (EOB) run" feature
   * that allows large groups of zeroes, potentially spanning multiple MCUs,
   * to be represented using only a few bytes.
   *
   * **Value**
   * - `0` *[default for compression, lossless transformation]* The lossy JPEG
   * image is (decompression) or will be (compression, lossless transformation)
   * single-scan.
   * - `1` The lossy JPEG image is (decompression) or will be (compression,
   * lossless transformation) progressive.  For lossless transformation, this
   * can also be specified using #TJXOPT_PROGRESSIVE.
   *
   * Progressive JPEG images generally have better compression ratios than
   * single-scan JPEG images (much better if the image has large areas of solid
   * color), but progressive JPEG compression and decompression is considerably
   * slower than single-scan JPEG compression and decompression.  Can be
   * combined with #TJPARAM_ARITHMETIC.  Implies #TJPARAM_OPTIMIZE unless
   * #TJPARAM_ARITHMETIC is also set.
   */
  TJPARAM_PROGRESSIVE,
  /**
   * Progressive JPEG scan limit for lossy JPEG images [decompression, lossless
   * transformation]
   *
   * Setting this parameter causes the decompression and transform functions to
   * return an error if the number of scans in a progressive JPEG image exceeds
   * the specified limit.  The primary purpose of this is to allow
   * security-critical applications to guard against an exploit of the
   * progressive JPEG format described in
   * <a href="https://libjpeg-turbo.org/pmwiki/uploads/About/TwoIssueswiththeJPEGStandard.pdf" target="_blank">this report</a>.
   *
   * **Value**
   * - maximum number of progressive JPEG scans that the decompression and
   * transform functions will process *[default: `0` (no limit)]*
   *
   * @see #TJPARAM_PROGRESSIVE
   */
  TJPARAM_SCANLIMIT,
  /**
   * Arithmetic entropy coding
   *
   * **Value**
   * - `0` *[default for compression, lossless transformation]* The lossy JPEG
   * image uses (decompression) or will use (compression, lossless
   * transformation) Huffman entropy coding.
   * - `1` The lossy JPEG image uses (decompression) or will use (compression,
   * lossless transformation) arithmetic entropy coding.  For lossless
   * transformation, this can also be specified using #TJXOPT_ARITHMETIC.
   *
   * Arithmetic entropy coding generally improves compression relative to
   * Huffman entropy coding, but it reduces compression and decompression
   * performance considerably.  Can be combined with #TJPARAM_PROGRESSIVE.
   */
  TJPARAM_ARITHMETIC,
  /**
   * Lossless JPEG
   *
   * **Value**
   * - `0` *[default for compression]* The JPEG image is (decompression) or
   * will be (compression) lossy/DCT-based.
   * - `1` The JPEG image is (decompression) or will be (compression)
   * lossless/predictive.
   *
   * In most cases, lossless JPEG compression and decompression is considerably
   * slower than lossy JPEG compression and decompression, and lossless JPEG
   * images are much larger than lossy JPEG images.  Thus, lossless JPEG images
   * are typically used only for applications that require mathematically
   * lossless compression.  Also note that the following features are not
   * available with lossless JPEG images:
   * - Colorspace conversion (lossless JPEG images always use #TJCS_RGB,
   * #TJCS_GRAY, or #TJCS_CMYK, depending on the pixel format of the source
   * image)
   * - Chrominance subsampling (lossless JPEG images always use #TJSAMP_444)
   * - JPEG quality selection
   * - DCT/IDCT algorithm selection
   * - Progressive JPEG
   * - Arithmetic entropy coding
   * - Compression from/decompression to planar YUV images
   * - Decompression scaling
   * - Lossless transformation
   *
   * @see #TJPARAM_LOSSLESSPSV, #TJPARAM_LOSSLESSPT
   */
  TJPARAM_LOSSLESS,
  /**
   * Lossless JPEG predictor selection value (PSV)
   *
   * **Value**
   * - `1`-`7` *[default for compression: `1`]*
   *
   * Lossless JPEG compression shares no algorithms with lossy JPEG
   * compression.  Instead, it uses differential pulse-code modulation (DPCM),
   * an algorithm whereby each sample is encoded as the difference between the
   * sample's value and a "predictor", which is based on the values of
   * neighboring samples.  If Ra is the sample immediately to the left of the
   * current sample, Rb is the sample immediately above the current sample, and
   * Rc is the sample diagonally to the left and above the current sample, then
   * the relationship between the predictor selection value and the predictor
   * is as follows:
   *
   * PSV | Predictor
   * ----|----------
   * 1   | Ra
   * 2   | Rb
   * 3   | Rc
   * 4   | Ra + Rb – Rc
   * 5   | Ra + (Rb – Rc) / 2
   * 6   | Rb + (Ra – Rc) / 2
   * 7   | (Ra + Rb) / 2
   *
   * Predictors 1-3 are 1-dimensional predictors, whereas Predictors 4-7 are
   * 2-dimensional predictors.  The best predictor for a particular image
   * depends on the image.
   *
   * @see #TJPARAM_LOSSLESS
   */
  TJPARAM_LOSSLESSPSV,
  /**
   * Lossless JPEG point transform (Pt)
   *
   * **Value**
   * - `0` through ***precision*** *- 1*, where ***precision*** is the JPEG
   * data precision in bits *[default for compression: `0`]*
   *
   * A point transform value of `0` is necessary in order to generate a fully
   * lossless JPEG image.  (A non-zero point transform value right-shifts the
   * input samples by the specified number of bits, which is effectively a form
   * of lossy color quantization.)
   *
   * @see #TJPARAM_LOSSLESS, #TJPARAM_PRECISION
   */
  TJPARAM_LOSSLESSPT,
  /**
   * JPEG restart marker interval in MCUs [lossy compression,
   * lossless transformation]
   *
   * The nature of entropy coding is such that a corrupt JPEG image cannot
   * be decompressed beyond the point of corruption unless it contains restart
   * markers.  A restart marker stops and restarts the entropy coding algorithm
   * so that, if a JPEG image is corrupted, decompression can resume at the
   * next marker.  Thus, adding more restart markers improves the fault
   * tolerance of the JPEG image, but adding too many restart markers can
   * adversely affect the compression ratio and performance.
   *
   * In typical JPEG images, an MCU (Minimum Coded Unit) is the minimum set of
   * interleaved "data units" (8x8 DCT blocks if the image is lossy or samples
   * if the image is lossless) necessary to represent at least one data unit
   * per component.  (For example, an MCU in an interleaved lossy JPEG image
   * that uses 4:2:2 subsampling consists of two luminance blocks followed by
   * one block for each chrominance component.)  In single-component or
   * non-interleaved JPEG images, an MCU is the same as a data unit.
   *
   * **Value**
   * - the number of MCUs between each restart marker *[default: `0` (no
   * restart markers)]*
   *
   * Setting this parameter to a non-zero value sets #TJPARAM_RESTARTROWS to 0.
   */
  TJPARAM_RESTARTBLOCKS,
  /**
   * JPEG restart marker interval in MCU rows [compression,
   * lossless transformation]
   *
   * See #TJPARAM_RESTARTBLOCKS for a description of restart markers and MCUs.
   * An MCU row is a row of MCUs spanning the entire width of the image.
   *
   * **Value**
   * - the number of MCU rows between each restart marker *[default: `0` (no
   * restart markers)]*
   *
   * Setting this parameter to a non-zero value sets #TJPARAM_RESTARTBLOCKS to
   * 0.
   */
  TJPARAM_RESTARTROWS,
  /**
   * JPEG horizontal pixel density
   *
   * **Value**
   * - The JPEG image has (decompression) or will have (compression) the
   * specified horizontal pixel density *[default for compression: `1`]*.
   *
   * This value is stored in or read from the JPEG header.  It does not affect
   * the contents of the JPEG image.  Note that this parameter is set by
   * #tj3LoadImage8() when loading a Windows BMP file that contains pixel
   * density information, and the value of this parameter is stored to a
   * Windows BMP file by #tj3SaveImage8() if the value of #TJPARAM_DENSITYUNITS
   * is `2`.
   *
   * This parameter has no effect unless the JPEG colorspace (see
   * #TJPARAM_COLORSPACE) is #TJCS_YCbCr or #TJCS_GRAY.
   *
   * @see TJPARAM_DENSITYUNITS
   */
  TJPARAM_XDENSITY,
  /**
   * JPEG vertical pixel density
   *
   * **Value**
   * - The JPEG image has (decompression) or will have (compression) the
   * specified vertical pixel density *[default for compression: `1`]*.
   *
   * This value is stored in or read from the JPEG header.  It does not affect
   * the contents of the JPEG image.  Note that this parameter is set by
   * #tj3LoadImage8() when loading a Windows BMP file that contains pixel
   * density information, and the value of this parameter is stored to a
   * Windows BMP file by #tj3SaveImage8() if the value of #TJPARAM_DENSITYUNITS
   * is `2`.
   *
   * This parameter has no effect unless the JPEG colorspace (see
   * #TJPARAM_COLORSPACE) is #TJCS_YCbCr or #TJCS_GRAY.
   *
   * @see TJPARAM_DENSITYUNITS
   */
  TJPARAM_YDENSITY,
  /**
   * JPEG pixel density units
   *
   * **Value**
   * - `0` *[default for compression]* The pixel density of the JPEG image is
   * expressed (decompression) or will be expressed (compression) in unknown
   * units.
   * - `1` The pixel density of the JPEG image is expressed (decompression) or
   * will be expressed (compression) in units of pixels/inch.
   * - `2` The pixel density of the JPEG image is expressed (decompression) or
   * will be expressed (compression) in units of pixels/cm.
   *
   * This value is stored in or read from the JPEG header.  It does not affect
   * the contents of the JPEG image.  Note that this parameter is set by
   * #tj3LoadImage8() when loading a Windows BMP file that contains pixel
   * density information, and the value of this parameter is stored to a
   * Windows BMP file by #tj3SaveImage8() if the value is `2`.
   *
   * This parameter has no effect unless the JPEG colorspace (see
   * #TJPARAM_COLORSPACE) is #TJCS_YCbCr or #TJCS_GRAY.
   *
   * @see TJPARAM_XDENSITY, TJPARAM_YDENSITY
   */
  TJPARAM_DENSITYUNITS,
  /**
   * Memory limit for intermediate buffers
   *
   * **Value**
   * - the maximum amount of memory (in megabytes) that will be allocated for
   * intermediate buffers, which are used with progressive JPEG compression and
   * decompression, Huffman table optimization, lossless JPEG compression, and
   * lossless transformation *[default: `0` (no limit)]*
   */
  TJPARAM_MAXMEMORY,
  /**
   * Image size limit [decompression, lossless transformation, packed-pixel
   * image loading]
   *
   * Setting this parameter causes the decompression, transform, and image
   * loading functions to return an error if the number of pixels in the source
   * image exceeds the specified limit.  This allows security-critical
   * applications to guard against excessive memory consumption.
   *
   * **Value**
   * - maximum number of pixels that the decompression, transform, and image
   * loading functions will process *[default: `0` (no limit)]*
   */
  TJPARAM_MAXPIXELS,
  /**
   * Marker copying behavior [decompression, lossless transformation]
   *
   * **Value [lossless transformation]**
   * - `0` Do not copy any extra markers (including comments, JFIF thumbnails,
   * Exif data, and ICC profile data) from the source image to the destination
   * image.
   * - `1` Do not copy any extra markers, except comment (COM) markers, from
   * the source image to the destination image.
   * - `2` *[default]* Copy all extra markers from the source image to the
   * destination image.
   * - `3` Copy all extra markers, except ICC profile data (APP2 markers), from
   * the source image to the destination image.
   * - `4` Do not copy any extra markers, except ICC profile data (APP2
   * markers), from the source image to the destination image.
   *
   * #TJXOPT_COPYNONE overrides this parameter for a particular transform.
   * This parameter overrides any ICC profile that was previously associated
   * with the TurboJPEG instance using #tj3SetICCProfile().
   *
   * When decompressing, #tj3DecompressHeader() extracts the ICC profile from a
   * JPEG image if this parameter is set to `2` or `4`.  #tj3GetICCProfile()
   * can then be used to retrieve the profile.
   */
  TJPARAM_SAVEMARKERS
};


/**
 * The number of error codes
 */
#define TJ_NUMERR  2

/**
 * Error codes
 */
enum TJERR {
  /**
   * The error was non-fatal and recoverable, but the destination image may
   * still be corrupt.
   */
  TJERR_WARNING,
  /**
   * The error was fatal and non-recoverable.
   */
  TJERR_FATAL
};


/**
 * The number of transform operations
 */
#define TJ_NUMXOP  8

/**
 * Transform operations for #tj3Transform()
 */
enum TJXOP {
  /**
   * Do not transform the position of the image pixels.
   */
  TJXOP_NONE,
  /**
   * Flip (mirror) image horizontally.  This transform is imperfect if there
   * are any partial iMCUs on the right edge (see #TJXOPT_PERFECT.)
   */
  TJXOP_HFLIP,
  /**
   * Flip (mirror) image vertically.  This transform is imperfect if there are
   * any partial iMCUs on the bottom edge (see #TJXOPT_PERFECT.)
   */
  TJXOP_VFLIP,
  /**
   * Transpose image (flip/mirror along upper left to lower right axis.)  This
   * transform is always perfect.
   */
  TJXOP_TRANSPOSE,
  /**
   * Transverse transpose image (flip/mirror along upper right to lower left
   * axis.)  This transform is imperfect if there are any partial iMCUs in the
   * image (see #TJXOPT_PERFECT.)
   */
  TJXOP_TRANSVERSE,
  /**
   * Rotate image clockwise by 90 degrees.  This transform is imperfect if
   * there are any partial iMCUs on the bottom edge (see #TJXOPT_PERFECT.)
   */
  TJXOP_ROT90,
  /**
   * Rotate image 180 degrees.  This transform is imperfect if there are any
   * partial iMCUs in the image (see #TJXOPT_PERFECT.)
   */
  TJXOP_ROT180,
  /**
   * Rotate image counter-clockwise by 90 degrees.  This transform is imperfect
   * if there are any partial iMCUs on the right edge (see #TJXOPT_PERFECT.)
   */
  TJXOP_ROT270
};


/**
 * This option causes #tj3Transform() to return an error if the transform is
 * not perfect.  Lossless transforms operate on iMCUs, the size of which
 * depends on the level of chrominance subsampling used (see #tjMCUWidth and
 * #tjMCUHeight.)  If the image's width or height is not evenly divisible by
 * the iMCU size, then there will be partial iMCUs on the right and/or bottom
 * edges.  It is not possible to move these partial iMCUs to the top or left of
 * the image, so any transform that would require that is "imperfect."  If this
 * option is not specified, then any partial iMCUs that cannot be transformed
 * will be left in place, which will create odd-looking strips on the right or
 * bottom edge of the image.
 */
#define TJXOPT_PERFECT  (1 << 0)
/**
 * Discard any partial iMCUs that cannot be transformed.
 */
#define TJXOPT_TRIM  (1 << 1)
/**
 * Enable lossless cropping.  See #tj3Transform() for more information.
 */
#define TJXOPT_CROP  (1 << 2)
/**
 * Discard the color data in the source image, and generate a grayscale
 * destination image.
 */
#define TJXOPT_GRAY  (1 << 3)
/**
 * Do not generate a destination image.  (This can be used in conjunction with
 * a custom filter to capture the transformed DCT coefficients without
 * transcoding them.)
 */
#define TJXOPT_NOOUTPUT  (1 << 4)
/**
 * Generate a progressive destination image instead of a single-scan
 * destination image.  Progressive JPEG images generally have better
 * compression ratios than single-scan JPEG images (much better if the image
 * has large areas of solid color), but progressive JPEG decompression is
 * considerably slower than single-scan JPEG decompression.  Can be combined
 * with #TJXOPT_ARITHMETIC.  Implies #TJXOPT_OPTIMIZE unless #TJXOPT_ARITHMETIC
 * is also specified.
 */
#define TJXOPT_PROGRESSIVE  (1 << 5)
/**
 * Do not copy any extra markers (including Exif and ICC profile data) from the
 * source image to the destination image.
 */
#define TJXOPT_COPYNONE  (1 << 6)
/**
 * Enable arithmetic entropy coding in the destination image.  Arithmetic
 * entropy coding generally improves compression relative to Huffman entropy
 * coding (the default), but it reduces decompression performance considerably.
 * Can be combined with #TJXOPT_PROGRESSIVE.
 */
#define TJXOPT_ARITHMETIC  (1 << 7)
/**
 * Enable Huffman table optimization for the destination image.  Huffman table
 * optimization improves compression slightly (generally 5% or less.)
 */
#define TJXOPT_OPTIMIZE  (1 << 8)


/**
 * Scaling factor
 */
typedef struct {
  /**
   * Numerator
   */
  int num;
  /**
   * Denominator
   */
  int denom;
} tjscalingfactor;

/**
 * Cropping region
 */
typedef struct {
  /**
   * The left boundary of the cropping region.  For lossless transformation,
   * this must be evenly divisible by the iMCU width (see #tjMCUWidth) of the
   * destination image.  For decompression, this must be evenly divisible by
   * the scaled iMCU width of the source image.
   */
  int x;
  /**
   * The upper boundary of the cropping region.  For lossless transformation,
   * this must be evenly divisible by the iMCU height (see #tjMCUHeight) of the
   * destination image.
   */
  int y;
  /**
   * The width of the cropping region.  Setting this to 0 is the equivalent of
   * setting it to the width of the source JPEG image - x.
   */
  int w;
  /**
   * The height of the cropping region.  Setting this to 0 is the equivalent of
   * setting it to the height of the source JPEG image - y.
   */
  int h;
} tjregion;

/**
 * A #tjregion structure that specifies no cropping
 */
static const tjregion TJUNCROPPED = { 0, 0, 0, 0 };

/**
 * Lossless transform
 */
typedef struct tjtransform {
  /**
   * Cropping region
   */
  tjregion r;
  /**
   * One of the @ref TJXOP "transform operations"
   */
  int op;
  /**
   * The bitwise OR of one of more of the @ref TJXOPT_ARITHMETIC
   * "transform options"
   */
  int options;
  /**
   * Arbitrary data that can be accessed within the body of the callback
   * function
   */
  void *data;
  /**
   * A callback function that can be used to modify the DCT coefficients after
   * they are losslessly transformed but before they are transcoded to a new
   * JPEG image.  This allows for custom filters or other transformations to be
   * applied in the frequency domain.
   *
   * @param coeffs pointer to an array of transformed DCT coefficients.  (NOTE:
   * This pointer is not guaranteed to be valid once the callback returns, so
   * applications wishing to hand off the DCT coefficients to another function
   * or library should make a copy of them within the body of the callback.)
   *
   * @param arrayRegion #tjregion structure containing the width and height of
   * the array pointed to by `coeffs` as well as its offset relative to the
   * component plane.  TurboJPEG implementations may choose to split each
   * component plane into multiple DCT coefficient arrays and call the callback
   * function once for each array.
   *
   * @param planeRegion #tjregion structure containing the width and height of
   * the component plane to which `coeffs` belongs
   *
   * @param componentID ID number of the component plane to which `coeffs`
   * belongs.  (Y, Cb, and Cr have, respectively, ID's of 0, 1, and 2 in
   * typical JPEG images.)
   *
   * @param transformID ID number of the transformed image to which `coeffs`
   * belongs.  This is the same as the index of the transform in the
   * `transforms` array that was passed to #tj3Transform().
   *
   * @param transform a pointer to a #tjtransform structure that specifies the
   * parameters and/or cropping region for this transform
   *
   * @return 0 if the callback was successful, or -1 if an error occurred.
   */
  int (*customFilter) (short *coeffs, tjregion arrayRegion,
                       tjregion planeRegion, int componentID, int transformID,
                       struct tjtransform *transform);
} tjtransform;

/**
 * TurboJPEG instance handle
 */
typedef void *tjhandle;


/**
 * Compute the scaled value of `dimension` using the given scaling factor.
 * This macro performs the integer equivalent of `ceil(dimension *
 * scalingFactor)`.
 */
#define TJSCALED(dimension, scalingFactor) \
  (((dimension) * scalingFactor.num + scalingFactor.denom - 1) / \
   scalingFactor.denom)

/**
 * A #tjscalingfactor structure that specifies a scaling factor of 1/1 (no
 * scaling)
 */
static const tjscalingfactor TJUNSCALED = { 1, 1 };


#ifdef __cplusplus
extern "C" {
#endif


/**
 * Create a new TurboJPEG instance.
 *
 * @param initType one of the @ref TJINIT "initialization options"
 *
 * @return a handle to the newly-created instance, or NULL if an error occurred
 * (see #tj3GetErrorStr().)
 */
DLLEXPORT tjhandle tj3Init(int initType);


/**
 * Destroy a TurboJPEG instance.
 *
 * @param handle handle to a TurboJPEG instance.  If the handle is NULL, then
 * this function has no effect.
 */
DLLEXPORT void tj3Destroy(tjhandle handle);


/**
 * Returns a descriptive error message explaining why the last command failed.
 *
 * @param handle handle to a TurboJPEG instance, or NULL if the error was
 * generated by a global function (but note that retrieving the error message
 * for a global function is thread-safe only on platforms that support
 * thread-local storage.)
 *
 * @return a descriptive error message explaining why the last command failed.
 */
DLLEXPORT char *tj3GetErrorStr(tjhandle handle);


/**
 * Returns a code indicating the severity of the last error.  See
 * @ref TJERR "Error codes".
 *
 * @param handle handle to a TurboJPEG instance
 *
 * @return a code indicating the severity of the last error.  See
 * @ref TJERR "Error codes".
 */
DLLEXPORT int tj3GetErrorCode(tjhandle handle);


/**
 * Set the value of a parameter.
 *
 * @param handle handle to a TurboJPEG instance
 *
 * @param param one of the @ref TJPARAM "parameters"
 *
 * @param value value of the parameter (refer to @ref TJPARAM
 * "parameter documentation")
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr().)
 */
DLLEXPORT int tj3Set(tjhandle handle, int param, int value);


/**
 * Get the value of a parameter.
 *
 * @param handle handle to a TurboJPEG instance
 *
 * @param param one of the @ref TJPARAM "parameters"
 *
 * @return the value of the specified parameter, or -1 if the value is unknown.
 */
DLLEXPORT int tj3Get(tjhandle handle, int param);


/**
 * Allocate a byte buffer for use with TurboJPEG.  You should always use this
 * function to allocate the JPEG destination buffer(s) for the compression and
 * transform functions unless you are disabling automatic buffer (re)allocation
 * (by setting #TJPARAM_NOREALLOC.)
 *
 * @param bytes the number of bytes to allocate
 *
 * @return a pointer to a newly-allocated buffer with the specified number of
 * bytes.
 *
 * @see tj3Free()
 */
DLLEXPORT void *tj3Alloc(size_t bytes);


/**
 * Free a byte buffer previously allocated by TurboJPEG.  You should always use
 * this function to free JPEG destination buffer(s) that were automatically
 * (re)allocated by the compression and transform functions or that were
 * manually allocated using #tj3Alloc().
 *
 * @param buffer address of the buffer to free.  If the address is NULL, then
 * this function has no effect.
 *
 * @see tj3Alloc()
 */
DLLEXPORT void tj3Free(void *buffer);


/**
 * The maximum size of the buffer (in bytes) required to hold a JPEG image with
 * the given parameters.  The number of bytes returned by this function is
 * larger than the size of the uncompressed source image.  The reason for this
 * is that the JPEG format uses 16-bit coefficients, so it is possible for a
 * very high-quality source image with very high-frequency content to expand
 * rather than compress when converted to the JPEG format.  Such images
 * represent very rare corner cases, but since there is no way to predict the
 * size of a JPEG image prior to compression, the corner cases have to be
 * handled.
 *
 * @param width width (in pixels) of the image
 *
 * @param height height (in pixels) of the image
 *
 * @param jpegSubsamp the level of chrominance subsampling to be used when
 * generating the JPEG image (see @ref TJSAMP
 * "Chrominance subsampling options".)  #TJSAMP_UNKNOWN is treated like
 * #TJSAMP_444, since a buffer large enough to hold a JPEG image with no
 * subsampling should also be large enough to hold a JPEG image with an
 * arbitrary level of subsampling.  Note that lossless JPEG images always
 * use #TJSAMP_444.
 *
 * @return the maximum size of the buffer (in bytes) required to hold the
 * image, or 0 if the arguments are out of bounds.
 */
DLLEXPORT size_t tj3JPEGBufSize(int width, int height, int jpegSubsamp);


/**
 * The size of the buffer (in bytes) required to hold a unified planar YUV
 * image with the given parameters.
 *
 * @param width width (in pixels) of the image
 *
 * @param align row alignment (in bytes) of the image (must be a power of 2.)
 * Setting this parameter to n specifies that each row in each plane of the
 * image will be padded to the nearest multiple of n bytes (1 = unpadded.)
 *
 * @param height height (in pixels) of the image
 *
 * @param subsamp level of chrominance subsampling in the image (see
 * @ref TJSAMP "Chrominance subsampling options".)
 *
 * @return the size of the buffer (in bytes) required to hold the image, or 0
 * if the arguments are out of bounds.
 */
DLLEXPORT size_t tj3YUVBufSize(int width, int align, int height, int subsamp);


/**
 * The size of the buffer (in bytes) required to hold a YUV image plane with
 * the given parameters.
 *
 * @param componentID ID number of the image plane (0 = Y, 1 = U/Cb, 2 = V/Cr)
 *
 * @param width width (in pixels) of the YUV image.  NOTE: This is the width of
 * the whole image, not the plane width.
 *
 * @param stride bytes per row in the image plane.  Setting this to 0 is the
 * equivalent of setting it to the plane width.
 *
 * @param height height (in pixels) of the YUV image.  NOTE: This is the height
 * of the whole image, not the plane height.
 *
 * @param subsamp level of chrominance subsampling in the image (see
 * @ref TJSAMP "Chrominance subsampling options".)
 *
 * @return the size of the buffer (in bytes) required to hold the YUV image
 * plane, or 0 if the arguments are out of bounds.
 */
DLLEXPORT size_t tj3YUVPlaneSize(int componentID, int width, int stride,
                                 int height, int subsamp);


/**
 * The plane width of a YUV image plane with the given parameters.  Refer to
 * @ref YUVnotes "YUV Image Format Notes" for a description of plane width.
 *
 * @param componentID ID number of the image plane (0 = Y, 1 = U/Cb, 2 = V/Cr)
 *
 * @param width width (in pixels) of the YUV image
 *
 * @param subsamp level of chrominance subsampling in the image (see
 * @ref TJSAMP "Chrominance subsampling options".)
 *
 * @return the plane width of a YUV image plane with the given parameters, or 0
 * if the arguments are out of bounds.
 */
DLLEXPORT int tj3YUVPlaneWidth(int componentID, int width, int subsamp);


/**
 * The plane height of a YUV image plane with the given parameters.  Refer to
 * @ref YUVnotes "YUV Image Format Notes" for a description of plane height.
 *
 * @param componentID ID number of the image plane (0 = Y, 1 = U/Cb, 2 = V/Cr)
 *
 * @param height height (in pixels) of the YUV image
 *
 * @param subsamp level of chrominance subsampling in the image (see
 * @ref TJSAMP "Chrominance subsampling options".)
 *
 * @return the plane height of a YUV image plane with the given parameters, or
 * 0 if the arguments are out of bounds.
 */
DLLEXPORT int tj3YUVPlaneHeight(int componentID, int height, int subsamp);


/**
 * Embed an ICC (International Color Consortium) color management profile in
 * JPEG images generated by subsequent compression and lossless transformation
 * operations.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * compression
 *
 * @param iccBuf pointer to a byte buffer containing an ICC profile.  A copy is
 * made of the ICC profile, so this buffer can be freed or reused as soon as
 * this function returns.  Setting this parameter to NULL or setting `iccSize`
 * to 0 removes any ICC profile that was previously associated with the
 * TurboJPEG instance.
 *
 * @param iccSize size of the ICC profile (in bytes.)  Setting this parameter
 * to 0 or setting `iccBuf` to NULL removes any ICC profile that was previously
 * associated with the TurboJPEG instance.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr().)
 */
DLLEXPORT int tj3SetICCProfile(tjhandle handle, unsigned char *iccBuf,
                               size_t iccSize);


/**
 * Compress a packed-pixel RGB, grayscale, or CMYK image with 2 to 8 bits of
 * data precision per sample into a JPEG image with the same data precision.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * compression
 *
 * @param srcBuf pointer to a buffer containing a packed-pixel RGB, grayscale,
 * or CMYK source image to be compressed.  This buffer should normally be
 * `pitch * height` samples in size.  However, you can also use this parameter
 * to compress from a specific region of a larger buffer.  The data precision
 * of the source image (from 2 to 8 bits per sample) can be specified using
 * #TJPARAM_PRECISION and defaults to 8 if #TJPARAM_PRECISION is unset or out
 * of range.
 *
 * @param width width (in pixels) of the source image
 *
 * @param pitch samples per row in the source image.  Normally this should be
 * <tt>width * #tjPixelSize[pixelFormat]</tt>, if the image is unpadded.
 * (Setting this parameter to 0 is the equivalent of setting it to
 * <tt>width * #tjPixelSize[pixelFormat]</tt>.)  However, you can also use this
 * parameter to specify the row alignment/padding of the source image, to skip
 * rows, or to compress from a specific region of a larger buffer.
 *
 * @param height height (in pixels) of the source image
 *
 * @param pixelFormat pixel format of the source image (see @ref TJPF
 * "Pixel formats".)
 *
 * @param jpegBuf address of a pointer to a byte buffer that will receive the
 * JPEG image.  TurboJPEG has the ability to reallocate the JPEG buffer to
 * accommodate the size of the JPEG image.  Thus, you can choose to:
 * -# pre-allocate the JPEG buffer with an arbitrary size using #tj3Alloc() and
 * let TurboJPEG grow the buffer as needed,
 * -# set `*jpegBuf` to NULL to tell TurboJPEG to allocate the buffer for you,
 * or
 * -# pre-allocate the buffer to a "worst case" size determined by calling
 * #tj3JPEGBufSize() and adding the return value to the size of the ICC profile
 * (if any) that was previously associated with the TurboJPEG instance (see
 * #tj3SetICCProfile().)  This should ensure that the buffer never has to be
 * re-allocated.  (Setting #TJPARAM_NOREALLOC guarantees that it won't be.)
 * .
 * If you choose option 1 or 3, then `*jpegSize` should be set to the size of
 * your pre-allocated buffer.  In any case, unless you have set
 * #TJPARAM_NOREALLOC, you should always check `*jpegBuf` upon return from this
 * function, as it may have changed.
 *
 * @param jpegSize pointer to a size_t variable that holds the size of the JPEG
 * buffer.  If `*jpegBuf` points to a pre-allocated buffer, then `*jpegSize`
 * should be set to the size of the buffer.  Upon return, `*jpegSize` will
 * contain the size of the JPEG image (in bytes.)  If `*jpegBuf` points to a
 * JPEG buffer that is being reused from a previous call to one of the JPEG
 * compression functions, then `*jpegSize` is ignored.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3Compress8(tjhandle handle, const unsigned char *srcBuf,
                           int width, int pitch, int height, int pixelFormat,
                           unsigned char **jpegBuf, size_t *jpegSize);

/**
 * Compress a packed-pixel RGB, grayscale, or CMYK image with 9 to 12 bits of
 * data precision per sample into a JPEG image with the same data precision.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * compression
 *
 * @param srcBuf pointer to a buffer containing a packed-pixel RGB, grayscale,
 * or CMYK source image to be compressed.  This buffer should normally be
 * `pitch * height` samples in size.  However, you can also use this parameter
 * to compress from a specific region of a larger buffer.  The data precision
 * of the source image (from 9 to 12 bits per sample) can be specified using
 * #TJPARAM_PRECISION and defaults to 12 if #TJPARAM_PRECISION is unset or out
 * of range.
 *
 * @param width width (in pixels) of the source image
 *
 * @param pitch samples per row in the source image.  Normally this should be
 * <tt>width * #tjPixelSize[pixelFormat]</tt>, if the image is unpadded.
 * (Setting this parameter to 0 is the equivalent of setting it to
 * <tt>width * #tjPixelSize[pixelFormat]</tt>.)  However, you can also use this
 * parameter to specify the row alignment/padding of the source image, to skip
 * rows, or to compress from a specific region of a larger buffer.
 *
 * @param height height (in pixels) of the source image
 *
 * @param pixelFormat pixel format of the source image (see @ref TJPF
 * "Pixel formats".)
 *
 * @param jpegBuf address of a pointer to a byte buffer that will receive the
 * JPEG image.  TurboJPEG has the ability to reallocate the JPEG buffer to
 * accommodate the size of the JPEG image.  Thus, you can choose to:
 * -# pre-allocate the JPEG buffer with an arbitrary size using #tj3Alloc() and
 * let TurboJPEG grow the buffer as needed,
 * -# set `*jpegBuf` to NULL to tell TurboJPEG to allocate the buffer for you,
 * or
 * -# pre-allocate the buffer to a "worst case" size determined by calling
 * #tj3JPEGBufSize() and adding the return value to the size of the ICC profile
 * (if any) that was previously associated with the TurboJPEG instance (see
 * #tj3SetICCProfile().)  This should ensure that the buffer never has to be
 * re-allocated.  (Setting #TJPARAM_NOREALLOC guarantees that it won't be.)
 * .
 * If you choose option 1 or 3, then `*jpegSize` should be set to the size of
 * your pre-allocated buffer.  In any case, unless you have set
 * #TJPARAM_NOREALLOC, you should always check `*jpegBuf` upon return from this
 * function, as it may have changed.
 *
 * @param jpegSize pointer to a size_t variable that holds the size of the JPEG
 * buffer.  If `*jpegBuf` points to a pre-allocated buffer, then `*jpegSize`
 * should be set to the size of the buffer.  Upon return, `*jpegSize` will
 * contain the size of the JPEG image (in bytes.)  If `*jpegBuf` points to a
 * JPEG buffer that is being reused from a previous call to one of the JPEG
 * compression functions, then `*jpegSize` is ignored.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3Compress12(tjhandle handle, const short *srcBuf, int width,
                            int pitch, int height, int pixelFormat,
                            unsigned char **jpegBuf, size_t *jpegSize);

/**
 * Compress a packed-pixel RGB, grayscale, or CMYK image with 13 to 16 bits of
 * data precision per sample into a lossless JPEG image with the same data
 * precision.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * compression
 *
 * @param srcBuf pointer to a buffer containing a packed-pixel RGB, grayscale,
 * or CMYK source image to be compressed.  This buffer should normally be
 * `pitch * height` samples in size.  However, you can also use this parameter
 * to compress from a specific region of a larger buffer.  The data precision
 * of the source image (from 13 to 16 bits per sample) can be specified using
 * #TJPARAM_PRECISION and defaults to 16 if #TJPARAM_PRECISION is unset or out
 * of range.
 *
 * @param width width (in pixels) of the source image
 *
 * @param pitch samples per row in the source image.  Normally this should be
 * <tt>width * #tjPixelSize[pixelFormat]</tt>, if the image is unpadded.
 * (Setting this parameter to 0 is the equivalent of setting it to
 * <tt>width * #tjPixelSize[pixelFormat]</tt>.)  However, you can also use this
 * parameter to specify the row alignment/padding of the source image, to skip
 * rows, or to compress from a specific region of a larger buffer.
 *
 * @param height height (in pixels) of the source image
 *
 * @param pixelFormat pixel format of the source image (see @ref TJPF
 * "Pixel formats".)
 *
 * @param jpegBuf address of a pointer to a byte buffer that will receive the
 * JPEG image.  TurboJPEG has the ability to reallocate the JPEG buffer to
 * accommodate the size of the JPEG image.  Thus, you can choose to:
 * -# pre-allocate the JPEG buffer with an arbitrary size using #tj3Alloc() and
 * let TurboJPEG grow the buffer as needed,
 * -# set `*jpegBuf` to NULL to tell TurboJPEG to allocate the buffer for you,
 * or
 * -# pre-allocate the buffer to a "worst case" size determined by calling
 * #tj3JPEGBufSize() and adding the return value to the size of the ICC profile
 * (if any) that was previously associated with the TurboJPEG instance (see
 * #tj3SetICCProfile().)  This should ensure that the buffer never has to be
 * re-allocated.  (Setting #TJPARAM_NOREALLOC guarantees that it won't be.)
 * .
 * If you choose option 1 or 3, then `*jpegSize` should be set to the size of
 * your pre-allocated buffer.  In any case, unless you have set
 * #TJPARAM_NOREALLOC, you should always check `*jpegBuf` upon return from this
 * function, as it may have changed.
 *
 * @param jpegSize pointer to a size_t variable that holds the size of the JPEG
 * buffer.  If `*jpegBuf` points to a pre-allocated buffer, then `*jpegSize`
 * should be set to the size of the buffer.  Upon return, `*jpegSize` will
 * contain the size of the JPEG image (in bytes.)  If `*jpegBuf` points to a
 * JPEG buffer that is being reused from a previous call to one of the JPEG
 * compression functions, then `*jpegSize` is ignored.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3Compress16(tjhandle handle, const unsigned short *srcBuf,
                            int width, int pitch, int height, int pixelFormat,
                            unsigned char **jpegBuf, size_t *jpegSize);


/**
 * Compress a set of 8-bit-per-sample Y, U (Cb), and V (Cr) image planes into
 * an 8-bit-per-sample JPEG image.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * compression
 *
 * @param srcPlanes an array of pointers to Y, U (Cb), and V (Cr) image planes
 * (or just a Y plane, if compressing a grayscale image) that contain a YUV
 * source image to be compressed.  These planes can be contiguous or
 * non-contiguous in memory.  The size of each plane should match the value
 * returned by #tj3YUVPlaneSize() for the given image width, height, strides,
 * and level of chrominance subsampling (see #TJPARAM_SUBSAMP.)  Refer to
 * @ref YUVnotes "YUV Image Format Notes" for more details.
 *
 * @param width width (in pixels) of the source image.  If the width is not an
 * even multiple of the iMCU width (see #tjMCUWidth), then an intermediate
 * buffer copy will be performed.
 *
 * @param strides an array of integers, each specifying the number of bytes per
 * row in the corresponding plane of the YUV source image.  Setting the stride
 * for any plane to 0 is the same as setting it to the plane width (see
 * @ref YUVnotes "YUV Image Format Notes".)  If `strides` is NULL, then the
 * strides for all planes will be set to their respective plane widths.  You
 * can adjust the strides in order to specify an arbitrary amount of row
 * padding in each plane or to create a JPEG image from a subregion of a larger
 * planar YUV image.
 *
 * @param height height (in pixels) of the source image.  If the height is not
 * an even multiple of the iMCU height (see #tjMCUHeight), then an intermediate
 * buffer copy will be performed.
 *
 * @param jpegBuf address of a pointer to a byte buffer that will receive the
 * JPEG image.  TurboJPEG has the ability to reallocate the JPEG buffer to
 * accommodate the size of the JPEG image.  Thus, you can choose to:
 * -# pre-allocate the JPEG buffer with an arbitrary size using #tj3Alloc() and
 * let TurboJPEG grow the buffer as needed,
 * -# set `*jpegBuf` to NULL to tell TurboJPEG to allocate the buffer for you,
 * or
 * -# pre-allocate the buffer to a "worst case" size determined by calling
 * #tj3JPEGBufSize() and adding the return value to the size of the ICC profile
 * (if any) that was previously associated with the TurboJPEG instance (see
 * #tj3SetICCProfile().)  This should ensure that the buffer never has to be
 * re-allocated.  (Setting #TJPARAM_NOREALLOC guarantees that it won't be.)
 * .
 * If you choose option 1 or 3, then `*jpegSize` should be set to the size of
 * your pre-allocated buffer.  In any case, unless you have set
 * #TJPARAM_NOREALLOC, you should always check `*jpegBuf` upon return from this
 * function, as it may have changed.
 *
 * @param jpegSize pointer to a size_t variable that holds the size of the JPEG
 * buffer.  If `*jpegBuf` points to a pre-allocated buffer, then `*jpegSize`
 * should be set to the size of the buffer.  Upon return, `*jpegSize` will
 * contain the size of the JPEG image (in bytes.)  If `*jpegBuf` points to a
 * JPEG buffer that is being reused from a previous call to one of the JPEG
 * compression functions, then `*jpegSize` is ignored.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3CompressFromYUVPlanes8(tjhandle handle,
                                        const unsigned char * const *srcPlanes,
                                        int width, const int *strides,
                                        int height, unsigned char **jpegBuf,
                                        size_t *jpegSize);


/**
 * Compress an 8-bit-per-sample unified planar YUV image into an
 * 8-bit-per-sample JPEG image.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * compression
 *
 * @param srcBuf pointer to a buffer containing a unified planar YUV source
 * image to be compressed.  The size of this buffer should match the value
 * returned by #tj3YUVBufSize() for the given image width, height, row
 * alignment, and level of chrominance subsampling (see #TJPARAM_SUBSAMP.)  The
 * Y, U (Cb), and V (Cr) image planes should be stored sequentially in the
 * buffer.  (Refer to @ref YUVnotes "YUV Image Format Notes".)
 *
 * @param width width (in pixels) of the source image.  If the width is not an
 * even multiple of the iMCU width (see #tjMCUWidth), then an intermediate
 * buffer copy will be performed.
 *
 * @param align row alignment (in bytes) of the source image (must be a power
 * of 2.)  Setting this parameter to n indicates that each row in each plane of
 * the source image is padded to the nearest multiple of n bytes
 * (1 = unpadded.)
 *
 * @param height height (in pixels) of the source image.  If the height is not
 * an even multiple of the iMCU height (see #tjMCUHeight), then an intermediate
 * buffer copy will be performed.
 *
 * @param jpegBuf address of a pointer to a byte buffer that will receive the
 * JPEG image.  TurboJPEG has the ability to reallocate the JPEG buffer to
 * accommodate the size of the JPEG image.  Thus, you can choose to:
 * -# pre-allocate the JPEG buffer with an arbitrary size using #tj3Alloc() and
 * let TurboJPEG grow the buffer as needed,
 * -# set `*jpegBuf` to NULL to tell TurboJPEG to allocate the buffer for you,
 * or
 * -# pre-allocate the buffer to a "worst case" size determined by calling
 * #tj3JPEGBufSize() and adding the return value to the size of the ICC profile
 * (if any) that was previously associated with the TurboJPEG instance (see
 * #tj3SetICCProfile().)  This should ensure that the buffer never has to be
 * re-allocated.  (Setting #TJPARAM_NOREALLOC guarantees that it won't be.)
 * .
 * If you choose option 1 or 3, then `*jpegSize` should be set to the size of
 * your pre-allocated buffer.  In any case, unless you have set
 * #TJPARAM_NOREALLOC, you should always check `*jpegBuf` upon return from this
 * function, as it may have changed.
 *
 * @param jpegSize pointer to a size_t variable that holds the size of the JPEG
 * buffer.  If `*jpegBuf` points to a pre-allocated buffer, then `*jpegSize`
 * should be set to the size of the buffer.  Upon return, `*jpegSize` will
 * contain the size of the JPEG image (in bytes.)  If `*jpegBuf` points to a
 * JPEG buffer that is being reused from a previous call to one of the JPEG
 * compression functions, then `*jpegSize` is ignored.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3CompressFromYUV8(tjhandle handle,
                                  const unsigned char *srcBuf, int width,
                                  int align, int height,
                                  unsigned char **jpegBuf, size_t *jpegSize);


/**
 * Encode an 8-bit-per-sample packed-pixel RGB or grayscale image into separate
 * 8-bit-per-sample Y, U (Cb), and V (Cr) image planes.  This function performs
 * color conversion (which is accelerated in the libjpeg-turbo implementation)
 * but does not execute any of the other steps in the JPEG compression process.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * compression
 *
 * @param srcBuf pointer to a buffer containing a packed-pixel RGB or grayscale
 * source image to be encoded.  This buffer should normally be `pitch * height`
 * bytes in size.  However, you can also use this parameter to encode from a
 * specific region of a larger buffer.
 *
 *
 * @param width width (in pixels) of the source image
 *
 * @param pitch bytes per row in the source image.  Normally this should be
 * <tt>width * #tjPixelSize[pixelFormat]</tt>, if the image is unpadded.
 * (Setting this parameter to 0 is the equivalent of setting it to
 * <tt>width * #tjPixelSize[pixelFormat]</tt>.)  However, you can also use this
 * parameter to specify the row alignment/padding of the source image, to skip
 * rows, or to encode from a specific region of a larger packed-pixel image.
 *
 * @param height height (in pixels) of the source image
 *
 * @param pixelFormat pixel format of the source image (see @ref TJPF
 * "Pixel formats".)
 *
 * @param dstPlanes an array of pointers to Y, U (Cb), and V (Cr) image planes
 * (or just a Y plane, if generating a grayscale image) that will receive the
 * encoded image.  These planes can be contiguous or non-contiguous in memory.
 * Use #tj3YUVPlaneSize() to determine the appropriate size for each plane
 * based on the image width, height, strides, and level of chrominance
 * subsampling (see #TJPARAM_SUBSAMP.)  Refer to @ref YUVnotes
 * "YUV Image Format Notes" for more details.
 *
 * @param strides an array of integers, each specifying the number of bytes per
 * row in the corresponding plane of the YUV image.  Setting the stride for any
 * plane to 0 is the same as setting it to the plane width (see @ref YUVnotes
 * "YUV Image Format Notes".)  If `strides` is NULL, then the strides for all
 * planes will be set to their respective plane widths.  You can adjust the
 * strides in order to add an arbitrary amount of row padding to each plane or
 * to encode an RGB or grayscale image into a subregion of a larger planar YUV
 * image.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3EncodeYUVPlanes8(tjhandle handle, const unsigned char *srcBuf,
                                  int width, int pitch, int height,
                                  int pixelFormat, unsigned char **dstPlanes,
                                  int *strides);


/**
 * Encode an 8-bit-per-sample packed-pixel RGB or grayscale image into an
 * 8-bit-per-sample unified planar YUV image.  This function performs color
 * conversion (which is accelerated in the libjpeg-turbo implementation) but
 * does not execute any of the other steps in the JPEG compression process.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * compression
 *
 * @param srcBuf pointer to a buffer containing a packed-pixel RGB or grayscale
 * source image to be encoded.  This buffer should normally be `pitch * height`
 * bytes in size.  However, you can also use this parameter to encode from a
 * specific region of a larger buffer.
 *
 * @param width width (in pixels) of the source image
 *
 * @param pitch bytes per row in the source image.  Normally this should be
 * <tt>width * #tjPixelSize[pixelFormat]</tt>, if the image is unpadded.
 * (Setting this parameter to 0 is the equivalent of setting it to
 * <tt>width * #tjPixelSize[pixelFormat]</tt>.)  However, you can also use this
 * parameter to specify the row alignment/padding of the source image, to skip
 * rows, or to encode from a specific region of a larger packed-pixel image.
 *
 * @param height height (in pixels) of the source image
 *
 * @param pixelFormat pixel format of the source image (see @ref TJPF
 * "Pixel formats".)
 *
 * @param dstBuf pointer to a buffer that will receive the unified planar YUV
 * image.  Use #tj3YUVBufSize() to determine the appropriate size for this
 * buffer based on the image width, height, row alignment, and level of
 * chrominance subsampling (see #TJPARAM_SUBSAMP.)  The Y, U (Cb), and V (Cr)
 * image planes will be stored sequentially in the buffer.  (Refer to
 * @ref YUVnotes "YUV Image Format Notes".)
 *
 * @param align row alignment (in bytes) of the YUV image (must be a power of
 * 2.)  Setting this parameter to n will cause each row in each plane of the
 * YUV image to be padded to the nearest multiple of n bytes (1 = unpadded.)
 * To generate images suitable for X Video, `align` should be set to 4.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3EncodeYUV8(tjhandle handle, const unsigned char *srcBuf,
                            int width, int pitch, int height, int pixelFormat,
                            unsigned char *dstBuf, int align);


/**
 * Retrieve information about a JPEG image without decompressing it, or prime
 * the decompressor with quantization and Huffman tables.  If a JPEG image is
 * passed to this function, then the @ref TJPARAM "parameters" that describe
 * the JPEG image will be set when the function returns.  If a JPEG image is
 * passed to this function and #TJPARAM_SAVEMARKERS is set to `2` or `4`, then
 * the ICC profile (if any) will be extracted from the JPEG image.
 * (#tj3GetICCProfile() can then be used to retrieve the profile.)
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * decompression
 *
 * @param jpegBuf pointer to a byte buffer containing a JPEG image or an
 * "abbreviated table specification" (AKA "tables-only") datastream.  Passing a
 * tables-only datastream to this function primes the decompressor with
 * quantization and Huffman tables that can be used when decompressing
 * subsequent "abbreviated image" datastreams.  This is useful, for instance,
 * when decompressing video streams in which all frames share the same
 * quantization and Huffman tables.
 *
 * @param jpegSize size of the JPEG image or tables-only datastream (in bytes)
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3DecompressHeader(tjhandle handle,
                                  const unsigned char *jpegBuf,
                                  size_t jpegSize);


/**
 * Retrieve the ICC (International Color Consortium) color management profile
 * (if any) that was previously extracted from a JPEG image.
 *
 * @note To extract the ICC profile from a JPEG image, call
 * #tj3DecompressHeader() with #TJPARAM_SAVEMARKERS set to `2` or `4`.  Once
 * the ICC profile is retrieved, it must be re-extracted before it can be
 * retrieved again.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * decompression
 *
 * @param iccBuf address of a pointer to a byte buffer.  Upon return:
 * - If `iccBuf` is not NULL and there is an ICC profile to retrieve, then
 * `*iccBuf` will point to a byte buffer containing the ICC profile.  This
 * buffer should be freed using #tj3Free().
 * - If `iccBuf` is not NULL and there is no ICC profile to retrieve, then
 * `*iccBuf` will be NULL.
 * - If `iccBuf` is NULL, then only the ICC profile size will be retrieved, and
 * the ICC profile can be retrieved later.
 *
 * @param iccSize address of a size_t variable.  Upon return, the variable will
 * contain the ICC profile size (or 0 if there is no ICC profile to retrieve.)
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3GetICCProfile(tjhandle handle, unsigned char **iccBuf,
                               size_t *iccSize);


/**
 * Returns a list of fractional scaling factors that the JPEG decompressor
 * supports.
 *
 * @param numScalingFactors pointer to an integer variable that will receive
 * the number of elements in the list
 *
 * @return a pointer to a list of fractional scaling factors, or NULL if an
 * error is encountered (see #tj3GetErrorStr().)
 */
DLLEXPORT tjscalingfactor *tj3GetScalingFactors(int *numScalingFactors);


/**
 * Set the scaling factor for subsequent lossy decompression operations.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * decompression
 *
 * @param scalingFactor #tjscalingfactor structure that specifies a fractional
 * scaling factor that the decompressor supports (see #tj3GetScalingFactors()),
 * or <tt>#TJUNSCALED</tt> for no scaling.  Decompression scaling is a function
 * of the IDCT algorithm, so scaling factors are generally limited to multiples
 * of 1/8.  If the entire JPEG image will be decompressed, then the width and
 * height of the scaled destination image can be determined by calling
 * #TJSCALED() with the JPEG width and height (see #TJPARAM_JPEGWIDTH and
 * #TJPARAM_JPEGHEIGHT) and the specified scaling factor.  When decompressing
 * into a planar YUV image, an intermediate buffer copy will be performed if
 * the width or height of the scaled destination image is not an even multiple
 * of the iMCU size (see #tjMCUWidth and #tjMCUHeight.)  Note that
 * decompression scaling is not available (and the specified scaling factor is
 * ignored) when decompressing lossless JPEG images (see #TJPARAM_LOSSLESS),
 * since the IDCT algorithm is not used with those images.  Note also that
 * #TJPARAM_FASTDCT is ignored when decompression scaling is enabled.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr().)
 */
DLLEXPORT int tj3SetScalingFactor(tjhandle handle,
                                  tjscalingfactor scalingFactor);


/**
 * Set the cropping region for partially decompressing a lossy JPEG image into
 * a packed-pixel image
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * decompression
 *
 * @param croppingRegion #tjregion structure that specifies a subregion of the
 * JPEG image to decompress, or <tt>#TJUNCROPPED</tt> for no cropping.  The
 * left boundary of the cropping region must be evenly divisible by the scaled
 * iMCU width-- <tt>#TJSCALED(#tjMCUWidth[subsamp], scalingFactor)</tt>, where
 * `subsamp` is the level of chrominance subsampling in the JPEG image (see
 * #TJPARAM_SUBSAMP) and `scalingFactor` is the decompression scaling factor
 * (see #tj3SetScalingFactor().)  The cropping region should be specified
 * relative to the scaled image dimensions.  Unless `croppingRegion` is
 * <tt>#TJUNCROPPED</tt>, the JPEG header must be read (see
 * #tj3DecompressHeader()) prior to calling this function.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr().)
 */
DLLEXPORT int tj3SetCroppingRegion(tjhandle handle, tjregion croppingRegion);


/**
 * Decompress a JPEG image with 2 to 8 bits of data precision per sample into a
 * packed-pixel RGB, grayscale, or CMYK image with the same data precision.
 * The @ref TJPARAM "parameters" that describe the JPEG image will be set when
 * this function returns.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * decompression
 *
 * @param jpegBuf pointer to a byte buffer containing the JPEG image to
 * decompress
 *
 * @param jpegSize size of the JPEG image (in bytes)
 *
 * @param dstBuf pointer to a buffer that will receive the packed-pixel
 * decompressed image.  This buffer should normally be
 * `pitch * destinationHeight` samples in size.  However, you can also use this
 * parameter to decompress into a specific region of a larger buffer.  NOTE:
 * If the JPEG image is lossy, then `destinationHeight` is either the scaled
 * JPEG height (see #TJSCALED(), #TJPARAM_JPEGHEIGHT, and
 * #tj3SetScalingFactor()) or the height of the cropping region (see
 * #tj3SetCroppingRegion().)  If the JPEG image is lossless, then
 * `destinationHeight` is the JPEG height.
 *
 * @param pitch samples per row in the destination image.  Normally this should
 * be set to <tt>destinationWidth * #tjPixelSize[pixelFormat]</tt>, if the
 * destination image should be unpadded.  (Setting this parameter to 0 is the
 * equivalent of setting it to
 * <tt>destinationWidth * #tjPixelSize[pixelFormat]</tt>.)  However, you can
 * also use this parameter to specify the row alignment/padding of the
 * destination image, to skip rows, or to decompress into a specific region of
 * a larger buffer.  NOTE: If the JPEG image is lossy, then `destinationWidth`
 * is either the scaled JPEG width (see #TJSCALED(), #TJPARAM_JPEGWIDTH, and
 * #tj3SetScalingFactor()) or the width of the cropping region (see
 * #tj3SetCroppingRegion().)  If the JPEG image is lossless, then
 * `destinationWidth` is the JPEG width.
 *
 * @param pixelFormat pixel format of the destination image (see @ref
 * TJPF "Pixel formats".)
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3Decompress8(tjhandle handle, const unsigned char *jpegBuf,
                             size_t jpegSize, unsigned char *dstBuf, int pitch,
                             int pixelFormat);

/**
 * Decompress a JPEG image with 9 to 12 bits of data precision per sample into
 * a packed-pixel RGB, grayscale, or CMYK image with the same data precision.
 *
 * \details \copydetails tj3Decompress8()
 */
DLLEXPORT int tj3Decompress12(tjhandle handle, const unsigned char *jpegBuf,
                              size_t jpegSize, short *dstBuf, int pitch,
                              int pixelFormat);

/**
 * Decompress a lossless JPEG image with 13 to 16 bits of data precision per
 * sample into a packed-pixel RGB, grayscale, or CMYK image with the same
 * data precision.
 *
 * \details \copydetails tj3Decompress8()
 */
DLLEXPORT int tj3Decompress16(tjhandle handle, const unsigned char *jpegBuf,
                              size_t jpegSize, unsigned short *dstBuf,
                              int pitch, int pixelFormat);


/**
 * Decompress an 8-bit-per-sample JPEG image into separate 8-bit-per-sample Y,
 * U (Cb), and V (Cr) image planes.  This function performs JPEG decompression
 * but leaves out the color conversion step, so a planar YUV image is generated
 * instead of a packed-pixel image.  The @ref TJPARAM "parameters" that
 * describe the JPEG image will be set when this function returns.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * decompression
 *
 * @param jpegBuf pointer to a byte buffer containing the JPEG image to
 * decompress
 *
 * @param jpegSize size of the JPEG image (in bytes)
 *
 * @param dstPlanes an array of pointers to Y, U (Cb), and V (Cr) image planes
 * (or just a Y plane, if decompressing a grayscale image) that will receive
 * the decompressed image.  These planes can be contiguous or non-contiguous in
 * memory.  Use #tj3YUVPlaneSize() to determine the appropriate size for each
 * plane based on the scaled JPEG width and height (see #TJSCALED(),
 * #TJPARAM_JPEGWIDTH, #TJPARAM_JPEGHEIGHT, and #tj3SetScalingFactor()),
 * strides, and level of chrominance subsampling (see #TJPARAM_SUBSAMP.)  Refer
 * to @ref YUVnotes "YUV Image Format Notes" for more details.
 *
 * @param strides an array of integers, each specifying the number of bytes per
 * row in the corresponding plane of the YUV image.  Setting the stride for any
 * plane to 0 is the same as setting it to the scaled plane width (see
 * @ref YUVnotes "YUV Image Format Notes".)  If `strides` is NULL, then the
 * strides for all planes will be set to their respective scaled plane widths.
 * You can adjust the strides in order to add an arbitrary amount of row
 * padding to each plane or to decompress the JPEG image into a subregion of a
 * larger planar YUV image.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3DecompressToYUVPlanes8(tjhandle handle,
                                        const unsigned char *jpegBuf,
                                        size_t jpegSize,
                                        unsigned char **dstPlanes,
                                        int *strides);


/**
 * Decompress an 8-bit-per-sample JPEG image into an 8-bit-per-sample unified
 * planar YUV image.  This function performs JPEG decompression but leaves out
 * the color conversion step, so a planar YUV image is generated instead of a
 * packed-pixel image.  The @ref TJPARAM "parameters" that describe the JPEG
 * image will be set when this function returns.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * decompression
 *
 * @param jpegBuf pointer to a byte buffer containing the JPEG image to
 * decompress
 *
 * @param jpegSize size of the JPEG image (in bytes)
 *
 * @param dstBuf pointer to a buffer that will receive the unified planar YUV
 * decompressed image.  Use #tj3YUVBufSize() to determine the appropriate size
 * for this buffer based on the scaled JPEG width and height (see #TJSCALED(),
 * #TJPARAM_JPEGWIDTH, #TJPARAM_JPEGHEIGHT, and #tj3SetScalingFactor()), row
 * alignment, and level of chrominance subsampling (see #TJPARAM_SUBSAMP.)  The
 * Y, U (Cb), and V (Cr) image planes will be stored sequentially in the
 * buffer.  (Refer to @ref YUVnotes "YUV Image Format Notes".)
 *
 * @param align row alignment (in bytes) of the YUV image (must be a power of
 * 2.)  Setting this parameter to n will cause each row in each plane of the
 * YUV image to be padded to the nearest multiple of n bytes (1 = unpadded.)
 * To generate images suitable for X Video, `align` should be set to 4.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3DecompressToYUV8(tjhandle handle,
                                  const unsigned char *jpegBuf,
                                  size_t jpegSize,
                                  unsigned char *dstBuf, int align);


/**
 * Decode a set of 8-bit-per-sample Y, U (Cb), and V (Cr) image planes into an
 * 8-bit-per-sample packed-pixel RGB or grayscale image.  This function
 * performs color conversion (which is accelerated in the libjpeg-turbo
 * implementation) but does not execute any of the other steps in the JPEG
 * decompression process.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * decompression
 *
 * @param srcPlanes an array of pointers to Y, U (Cb), and V (Cr) image planes
 * (or just a Y plane, if decoding a grayscale image) that contain a YUV image
 * to be decoded.  These planes can be contiguous or non-contiguous in memory.
 * The size of each plane should match the value returned by #tj3YUVPlaneSize()
 * for the given image width, height, strides, and level of chrominance
 * subsampling (see #TJPARAM_SUBSAMP.)  Refer to @ref YUVnotes
 * "YUV Image Format Notes" for more details.
 *
 * @param strides an array of integers, each specifying the number of bytes per
 * row in the corresponding plane of the YUV source image.  Setting the stride
 * for any plane to 0 is the same as setting it to the plane width (see
 * @ref YUVnotes "YUV Image Format Notes".)  If `strides` is NULL, then the
 * strides for all planes will be set to their respective plane widths.  You
 * can adjust the strides in order to specify an arbitrary amount of row
 * padding in each plane or to decode a subregion of a larger planar YUV image.
 *
 * @param dstBuf pointer to a buffer that will receive the packed-pixel decoded
 * image.  This buffer should normally be `pitch * height` bytes in size.
 * However, you can also use this parameter to decode into a specific region of
 * a larger buffer.
 *
 * @param width width (in pixels) of the source and destination images
 *
 * @param pitch bytes per row in the destination image.  Normally this should
 * be set to <tt>width * #tjPixelSize[pixelFormat]</tt>, if the destination
 * image should be unpadded.  (Setting this parameter to 0 is the equivalent of
 * setting it to <tt>width * #tjPixelSize[pixelFormat]</tt>.)  However, you can
 * also use this parameter to specify the row alignment/padding of the
 * destination image, to skip rows, or to decode into a specific region of a
 * larger buffer.
 *
 * @param height height (in pixels) of the source and destination images
 *
 * @param pixelFormat pixel format of the destination image (see @ref TJPF
 * "Pixel formats".)
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3DecodeYUVPlanes8(tjhandle handle,
                                  const unsigned char * const *srcPlanes,
                                  const int *strides, unsigned char *dstBuf,
                                  int width, int pitch, int height,
                                  int pixelFormat);


/**
 * Decode an 8-bit-per-sample unified planar YUV image into an 8-bit-per-sample
 * packed-pixel RGB or grayscale image.  This function performs color
 * conversion (which is accelerated in the libjpeg-turbo implementation) but
 * does not execute any of the other steps in the JPEG decompression process.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * decompression
 *
 * @param srcBuf pointer to a buffer containing a unified planar YUV source
 * image to be decoded.  The size of this buffer should match the value
 * returned by #tj3YUVBufSize() for the given image width, height, row
 * alignment, and level of chrominance subsampling (see #TJPARAM_SUBSAMP.)  The
 * Y, U (Cb), and V (Cr) image planes should be stored sequentially in the
 * source buffer.  (Refer to @ref YUVnotes "YUV Image Format Notes".)
 *
 * @param align row alignment (in bytes) of the YUV source image (must be a
 * power of 2.)  Setting this parameter to n indicates that each row in each
 * plane of the YUV source image is padded to the nearest multiple of n bytes
 * (1 = unpadded.)
 *
 * @param dstBuf pointer to a buffer that will receive the packed-pixel decoded
 * image.  This buffer should normally be `pitch * height` bytes in size.
 * However, you can also use this parameter to decode into a specific region of
 * a larger buffer.
 *
 * @param width width (in pixels) of the source and destination images
 *
 * @param pitch bytes per row in the destination image.  Normally this should
 * be set to <tt>width * #tjPixelSize[pixelFormat]</tt>, if the destination
 * image should be unpadded.  (Setting this parameter to 0 is the equivalent of
 * setting it to <tt>width * #tjPixelSize[pixelFormat]</tt>.)  However, you can
 * also use this parameter to specify the row alignment/padding of the
 * destination image, to skip rows, or to decode into a specific region of a
 * larger buffer.
 *
 * @param height height (in pixels) of the source and destination images
 *
 * @param pixelFormat pixel format of the destination image (see @ref TJPF
 * "Pixel formats".)
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3DecodeYUV8(tjhandle handle, const unsigned char *srcBuf,
                            int align, unsigned char *dstBuf, int width,
                            int pitch, int height, int pixelFormat);


/**
 * The maximum size of the buffer (in bytes) required to hold a JPEG image
 * transformed with the given transform parameters and/or cropping region.
 * This function is a wrapper for #tj3JPEGBufSize() that takes into account
 * cropping, transposition of the width and height (which affects the
 * destination image dimensions and level of chrominance subsampling),
 * grayscale conversion, and the ICC profile (if any) that was previously
 * associated with the TurboJPEG instance (see #tj3SetICCProfile()) or
 * extracted from the source image (see #tj3GetICCProfile() and
 * #TJPARAM_SAVEMARKERS.)  The JPEG header must be read (see
 * tj3DecompressHeader()) prior to calling this function.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * lossless transformation
 *
 * @param transform pointer to a #tjtransform structure that specifies the
 * transform parameters and/or cropping region for the JPEG image.
 *
 * @return the maximum size of the buffer (in bytes) required to hold the
 * transformed image, or 0 if an error occurred (see #tj3GetErrorStr() and
 * #tj3GetErrorCode().)
 */
DLLEXPORT size_t tj3TransformBufSize(tjhandle handle,
                                     const tjtransform *transform);


/**
 * Losslessly transform a JPEG image into another JPEG image.  Lossless
 * transforms work by moving the raw DCT coefficients from one JPEG image
 * structure to another without altering the values of the coefficients.  While
 * this is typically faster than decompressing the image, transforming it, and
 * re-compressing it, lossless transforms are not free.  Each lossless
 * transform requires reading and performing entropy decoding on all of the
 * coefficients in the source image, regardless of the size of the destination
 * image.  Thus, this function provides a means of generating multiple
 * transformed images from the same source or applying multiple transformations
 * simultaneously, in order to eliminate the need to read the source
 * coefficients multiple times.
 *
 * @param handle handle to a TurboJPEG instance that has been initialized for
 * lossless transformation
 *
 * @param jpegBuf pointer to a byte buffer containing the JPEG source image to
 * transform
 *
 * @param jpegSize size of the JPEG source image (in bytes)
 *
 * @param n the number of transformed JPEG images to generate
 *
 * @param dstBufs pointer to an array of n byte buffers.  `dstBufs[i]` will
 * receive a JPEG image that has been transformed using the parameters in
 * `transforms[i]`.  TurboJPEG has the ability to reallocate the JPEG
 * destination buffer to accommodate the size of the transformed JPEG image.
 * Thus, you can choose to:
 * -# pre-allocate the JPEG destination buffer with an arbitrary size using
 * #tj3Alloc() and let TurboJPEG grow the buffer as needed,
 * -# set `dstBufs[i]` to NULL to tell TurboJPEG to allocate the buffer for
 * you, or
 * -# pre-allocate the buffer to a "worst case" size determined by calling
 * #tj3TransformBufSize().  Under normal circumstances, this should ensure that
 * the buffer never has to be re-allocated.  (Setting #TJPARAM_NOREALLOC
 * guarantees that it won't be.)  Note, however, that there are some rare cases
 * (such as transforming images with a large amount of embedded Exif data) in
 * which the transformed JPEG image will be larger than the worst-case size,
 * and #TJPARAM_NOREALLOC cannot be used in those cases unless the embedded
 * data is discarded using #TJXOPT_COPYNONE or #TJPARAM_SAVEMARKERS.
 * .
 * If you choose option 1 or 3, then `dstSizes[i]` should be set to the size of
 * your pre-allocated buffer.  In any case, unless you have set
 * #TJPARAM_NOREALLOC, you should always check `dstBufs[i]` upon return from
 * this function, as it may have changed.
 *
 * @param dstSizes pointer to an array of n size_t variables that will receive
 * the actual sizes (in bytes) of each transformed JPEG image.  If `dstBufs[i]`
 * points to a pre-allocated buffer, then `dstSizes[i]` should be set to the
 * size of the buffer.  Upon return, `dstSizes[i]` will contain the size of the
 * transformed JPEG image (in bytes.)
 *
 * @param transforms pointer to an array of n #tjtransform structures, each of
 * which specifies the transform parameters and/or cropping region for the
 * corresponding transformed JPEG image.
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr()
 * and #tj3GetErrorCode().)
 */
DLLEXPORT int tj3Transform(tjhandle handle, const unsigned char *jpegBuf,
                           size_t jpegSize, int n, unsigned char **dstBufs,
                           size_t *dstSizes, const tjtransform *transforms);


/**
 * Load a packed-pixel image with 2 to 8 bits of data precision per sample from
 * disk into memory.
 *
 * @param handle handle to a TurboJPEG instance
 *
 * @param filename name of a file containing a packed-pixel image in Windows
 * BMP or PBMPLUS (PPM/PGM) format.  Windows BMP files require 8-bit-per-sample
 * data precision.  When loading a PBMPLUS file, the target data precision
 * (from 2 to 8 bits per sample) can be specified using #TJPARAM_PRECISION and
 * defaults to 8 if #TJPARAM_PRECISION is unset or out of range.  If the data
 * precision of the PBMPLUS file does not match the target data precision, then
 * upconverting or downconverting will be performed.
 *
 * @param width pointer to an integer variable that will receive the width (in
 * pixels) of the packed-pixel image
 *
 * @param align row alignment (in samples) of the packed-pixel buffer to be
 * returned (must be a power of 2.)  Setting this parameter to n will cause all
 * rows in the buffer to be padded to the nearest multiple of n samples
 * (1 = unpadded.)
 *
 * @param height pointer to an integer variable that will receive the height
 * (in pixels) of the packed-pixel image
 *
 * @param pixelFormat pointer to an integer variable that specifies or will
 * receive the pixel format of the packed-pixel buffer.  The behavior of this
 * function varies depending on the value of `*pixelFormat` passed to the
 * function:
 * - @ref TJPF_UNKNOWN : The packed-pixel buffer returned by this function will
 * use the most optimal pixel format for the file type, and `*pixelFormat` will
 * contain the ID of that pixel format upon successful return from this
 * function.
 * - @ref TJPF_GRAY : Only PGM files and 8-bit-per-pixel BMP files with a
 * grayscale colormap can be loaded.
 * - @ref TJPF_CMYK : The RGB or grayscale pixels stored in the file will be
 * converted using a quick & dirty algorithm that is suitable only for testing
 * purposes.  (Proper conversion between CMYK and other formats requires a
 * color management system.)
 * - Other @ref TJPF "pixel formats" : The packed-pixel buffer will use the
 * specified pixel format, and pixel format conversion will be performed if
 * necessary.
 *
 * @return a pointer to a newly-allocated buffer containing the packed-pixel
 * image, converted to the chosen pixel format and with the chosen row
 * alignment, or NULL if an error occurred (see #tj3GetErrorStr().)  This
 * buffer should be freed using #tj3Free().
 */
DLLEXPORT unsigned char *tj3LoadImage8(tjhandle handle, const char *filename,
                                       int *width, int align, int *height,
                                       int *pixelFormat);

/**
 * Load a packed-pixel image with 9 to 12 bits of data precision per sample
 * from disk into memory.
 *
 * @param handle handle to a TurboJPEG instance
 *
 * @param filename name of a file containing a packed-pixel image in PBMPLUS
 * (PPM/PGM) format.  The target data precision (from 9 to 12 bits per sample)
 * can be specified using #TJPARAM_PRECISION and defaults to 12 if
 * #TJPARAM_PRECISION is unset or out of range.  If the data precision of the
 * PBMPLUS file does not match the target data precision, then upconverting or
 * downconverting will be performed.
 *
 * @param width pointer to an integer variable that will receive the width (in
 * pixels) of the packed-pixel image
 *
 * @param align row alignment (in samples) of the packed-pixel buffer to be
 * returned (must be a power of 2.)  Setting this parameter to n will cause all
 * rows in the buffer to be padded to the nearest multiple of n samples
 * (1 = unpadded.)
 *
 * @param height pointer to an integer variable that will receive the height
 * (in pixels) of the packed-pixel image
 *
 * @param pixelFormat pointer to an integer variable that specifies or will
 * receive the pixel format of the packed-pixel buffer.  The behavior of this
 * function will vary depending on the value of `*pixelFormat` passed to the
 * function:
 * - @ref TJPF_UNKNOWN : The packed-pixel buffer returned by this function will
 * use the most optimal pixel format for the file type, and `*pixelFormat` will
 * contain the ID of that pixel format upon successful return from this
 * function.
 * - @ref TJPF_GRAY : Only PGM files can be loaded.
 * - @ref TJPF_CMYK : The RGB or grayscale pixels stored in the file will be
 * converted using a quick & dirty algorithm that is suitable only for testing
 * purposes.  (Proper conversion between CMYK and other formats requires a
 * color management system.)
 * - Other @ref TJPF "pixel formats" : The packed-pixel buffer will use the
 * specified pixel format, and pixel format conversion will be performed if
 * necessary.
 *
 * @return a pointer to a newly-allocated buffer containing the packed-pixel
 * image, converted to the chosen pixel format and with the chosen row
 * alignment, or NULL if an error occurred (see #tj3GetErrorStr().)  This
 * buffer should be freed using #tj3Free().
 */
DLLEXPORT short *tj3LoadImage12(tjhandle handle, const char *filename,
                                int *width, int align, int *height,
                                int *pixelFormat);

/**
 * Load a packed-pixel image with 13 to 16 bits of data precision per sample
 * from disk into memory.
 *
 * @param handle handle to a TurboJPEG instance
 *
 * @param filename name of a file containing a packed-pixel image in PBMPLUS
 * (PPM/PGM) format.  The target data precision (from 13 to 16 bits per sample)
 * can be specified using #TJPARAM_PRECISION and defaults to 16 if
 * #TJPARAM_PRECISION is unset or out of range.  If the data precision of the
 * PBMPLUS file does not match the target data precision, then upconverting or
 * downconverting will be performed.
 *
 * @param width pointer to an integer variable that will receive the width (in
 * pixels) of the packed-pixel image
 *
 * @param align row alignment (in samples) of the packed-pixel buffer to be
 * returned (must be a power of 2.)  Setting this parameter to n will cause all
 * rows in the buffer to be padded to the nearest multiple of n samples
 * (1 = unpadded.)
 *
 * @param height pointer to an integer variable that will receive the height
 * (in pixels) of the packed-pixel image
 *
 * @param pixelFormat pointer to an integer variable that specifies or will
 * receive the pixel format of the packed-pixel buffer.  The behavior of this
 * function will vary depending on the value of `*pixelFormat` passed to the
 * function:
 * - @ref TJPF_UNKNOWN : The packed-pixel buffer returned by this function will
 * use the most optimal pixel format for the file type, and `*pixelFormat` will
 * contain the ID of that pixel format upon successful return from this
 * function.
 * - @ref TJPF_GRAY : Only PGM files can be loaded.
 * - @ref TJPF_CMYK : The RGB or grayscale pixels stored in the file will be
 * converted using a quick & dirty algorithm that is suitable only for testing
 * purposes.  (Proper conversion between CMYK and other formats requires a
 * color management system.)
 * - Other @ref TJPF "pixel formats" : The packed-pixel buffer will use the
 * specified pixel format, and pixel format conversion will be performed if
 * necessary.
 *
 * @return a pointer to a newly-allocated buffer containing the packed-pixel
 * image, converted to the chosen pixel format and with the chosen row
 * alignment, or NULL if an error occurred (see #tj3GetErrorStr().)  This
 * buffer should be freed using #tj3Free().
 */
DLLEXPORT unsigned short *tj3LoadImage16(tjhandle handle, const char *filename,
                                         int *width, int align, int *height,
                                         int *pixelFormat);


/**
 * Save a packed-pixel image with 2 to 8 bits of data precision per sample from
 * memory to disk.
 *
 * @param handle handle to a TurboJPEG instance
 *
 * @param filename name of a file to which to save the packed-pixel image.  The
 * image will be stored in Windows BMP or PBMPLUS (PPM/PGM) format, depending
 * on the file extension.  Windows BMP files require 8-bit-per-sample data
 * precision.  When saving a PBMPLUS file, the source data precision (from 2 to
 * 8 bits per sample) can be specified using #TJPARAM_PRECISION and defaults to
 * 8 if #TJPARAM_PRECISION is unset or out of range.
 *
 * @param buffer pointer to a buffer containing a packed-pixel RGB, grayscale,
 * or CMYK image to be saved
 *
 * @param width width (in pixels) of the packed-pixel image
 *
 * @param pitch samples per row in the packed-pixel image.  Setting this
 * parameter to 0 is the equivalent of setting it to
 * <tt>width * #tjPixelSize[pixelFormat]</tt>.
 *
 * @param height height (in pixels) of the packed-pixel image
 *
 * @param pixelFormat pixel format of the packed-pixel image (see @ref TJPF
 * "Pixel formats".)  If this parameter is set to @ref TJPF_GRAY, then the
 * image will be stored in PGM or 8-bit-per-pixel (indexed color) BMP format.
 * Otherwise, the image will be stored in PPM or 24-bit-per-pixel BMP format.
 * If this parameter is set to @ref TJPF_CMYK, then the CMYK pixels will be
 * converted to RGB using a quick & dirty algorithm that is suitable only for
 * testing purposes.  (Proper conversion between CMYK and other formats
 * requires a color management system.)
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr().)
 */
DLLEXPORT int tj3SaveImage8(tjhandle handle, const char *filename,
                            const unsigned char *buffer, int width, int pitch,
                            int height, int pixelFormat);

/**
 * Save a packed-pixel image with 9 to 12 bits of data precision per sample
 * from memory to disk.
 *
 * @param handle handle to a TurboJPEG instance
 *
 * @param filename name of a file to which to save the packed-pixel image,
 * which will be stored in PBMPLUS (PPM/PGM) format.  The source data precision
 * (from 9 to 12 bits per sample) can be specified using #TJPARAM_PRECISION and
 * defaults to 12 if #TJPARAM_PRECISION is unset or out of range.
 *
 * @param buffer pointer to a buffer containing a packed-pixel RGB, grayscale,
 * or CMYK image to be saved
 *
 * @param width width (in pixels) of the packed-pixel image
 *
 * @param pitch samples per row in the packed-pixel image.  Setting this
 * parameter to 0 is the equivalent of setting it to
 * <tt>width * #tjPixelSize[pixelFormat]</tt>.
 *
 * @param height height (in pixels) of the packed-pixel image
 *
 * @param pixelFormat pixel format of the packed-pixel image (see @ref TJPF
 * "Pixel formats".)  If this parameter is set to @ref TJPF_GRAY, then the
 * image will be stored in PGM format.  Otherwise, the image will be stored in
 * PPM format.  If this parameter is set to @ref TJPF_CMYK, then the CMYK
 * pixels will be converted to RGB using a quick & dirty algorithm that is
 * suitable only for testing purposes.  (Proper conversion between CMYK and
 * other formats requires a color management system.)
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr().)
 */
DLLEXPORT int tj3SaveImage12(tjhandle handle, const char *filename,
                             const short *buffer, int width, int pitch,
                             int height, int pixelFormat);

/**
 * Save a packed-pixel image with 13 to 16 bits of data precision per sample
 * from memory to disk.
 *
 * @param handle handle to a TurboJPEG instance
 *
 * @param filename name of a file to which to save the packed-pixel image,
 * which will be stored in PBMPLUS (PPM/PGM) format.  The source data precision
 * (from 13 to 16 bits per sample) can be specified using #TJPARAM_PRECISION
 * and defaults to 16 if #TJPARAM_PRECISION is unset or out of range.
 *
 * @param buffer pointer to a buffer containing a packed-pixel RGB, grayscale,
 * or CMYK image to be saved
 *
 * @param width width (in pixels) of the packed-pixel image
 *
 * @param pitch samples per row in the packed-pixel image.  Setting this
 * parameter to 0 is the equivalent of setting it to
 * <tt>width * #tjPixelSize[pixelFormat]</tt>.
 *
 * @param height height (in pixels) of the packed-pixel image
 *
 * @param pixelFormat pixel format of the packed-pixel image (see @ref TJPF
 * "Pixel formats".)  If this parameter is set to @ref TJPF_GRAY, then the
 * image will be stored in PGM format.  Otherwise, the image will be stored in
 * PPM format.  If this parameter is set to @ref TJPF_CMYK, then the CMYK
 * pixels will be converted to RGB using a quick & dirty algorithm that is
 * suitable only for testing purposes.  (Proper conversion between CMYK and
 * other formats requires a color management system.)
 *
 * @return 0 if successful, or -1 if an error occurred (see #tj3GetErrorStr().)
 */
DLLEXPORT int tj3SaveImage16(tjhandle handle, const char *filename,
                             const unsigned short *buffer, int width,
                             int pitch, int height, int pixelFormat);


/* Backward compatibility functions and macros (nothing to see here) */

/* TurboJPEG 1.0+ */

#define NUMSUBOPT  TJ_NUMSAMP
#define TJ_444  TJSAMP_444
#define TJ_422  TJSAMP_422
#define TJ_420  TJSAMP_420
#define TJ_411  TJSAMP_420
#define TJ_GRAYSCALE  TJSAMP_GRAY

#define TJ_BGR  1
#define TJ_BOTTOMUP  TJFLAG_BOTTOMUP
#define TJ_FORCEMMX  TJFLAG_FORCEMMX
#define TJ_FORCESSE  TJFLAG_FORCESSE
#define TJ_FORCESSE2  TJFLAG_FORCESSE2
#define TJ_ALPHAFIRST  64
#define TJ_FORCESSE3  TJFLAG_FORCESSE3
#define TJ_FASTUPSAMPLE  TJFLAG_FASTUPSAMPLE

#define TJPAD(width)  (((width) + 3) & (~3))

DLLEXPORT unsigned long TJBUFSIZE(int width, int height);

DLLEXPORT int tjCompress(tjhandle handle, unsigned char *srcBuf, int width,
                         int pitch, int height, int pixelSize,
                         unsigned char *dstBuf, unsigned long *compressedSize,
                         int jpegSubsamp, int jpegQual, int flags);

DLLEXPORT int tjDecompress(tjhandle handle, unsigned char *jpegBuf,
                           unsigned long jpegSize, unsigned char *dstBuf,
                           int width, int pitch, int height, int pixelSize,
                           int flags);

DLLEXPORT int tjDecompressHeader(tjhandle handle, unsigned char *jpegBuf,
                                 unsigned long jpegSize, int *width,
                                 int *height);

DLLEXPORT int tjDestroy(tjhandle handle);

DLLEXPORT char *tjGetErrorStr(void);

DLLEXPORT tjhandle tjInitCompress(void);

DLLEXPORT tjhandle tjInitDecompress(void);

/* TurboJPEG 1.1+ */

#define TJ_YUV  512

DLLEXPORT unsigned long TJBUFSIZEYUV(int width, int height, int jpegSubsamp);

DLLEXPORT int tjDecompressHeader2(tjhandle handle, unsigned char *jpegBuf,
                                  unsigned long jpegSize, int *width,
                                  int *height, int *jpegSubsamp);

DLLEXPORT int tjDecompressToYUV(tjhandle handle, unsigned char *jpegBuf,
                                unsigned long jpegSize, unsigned char *dstBuf,
                                int flags);

DLLEXPORT int tjEncodeYUV(tjhandle handle, unsigned char *srcBuf, int width,
                          int pitch, int height, int pixelSize,
                          unsigned char *dstBuf, int subsamp, int flags);

/* TurboJPEG 1.2+ */

#define TJFLAG_BOTTOMUP  2
#define TJFLAG_FORCEMMX  8
#define TJFLAG_FORCESSE  16
#define TJFLAG_FORCESSE2  32
#define TJFLAG_FORCESSE3  128
#define TJFLAG_FASTUPSAMPLE  256
#define TJFLAG_NOREALLOC  1024

DLLEXPORT unsigned char *tjAlloc(int bytes);

DLLEXPORT unsigned long tjBufSize(int width, int height, int jpegSubsamp);

DLLEXPORT unsigned long tjBufSizeYUV(int width, int height, int subsamp);

DLLEXPORT int tjCompress2(tjhandle handle, const unsigned char *srcBuf,
                          int width, int pitch, int height, int pixelFormat,
                          unsigned char **jpegBuf, unsigned long *jpegSize,
                          int jpegSubsamp, int jpegQual, int flags);

DLLEXPORT int tjDecompress2(tjhandle handle, const unsigned char *jpegBuf,
                            unsigned long jpegSize, unsigned char *dstBuf,
                            int width, int pitch, int height, int pixelFormat,
                            int flags);

DLLEXPORT int tjEncodeYUV2(tjhandle handle, unsigned char *srcBuf, int width,
                           int pitch, int height, int pixelFormat,
                           unsigned char *dstBuf, int subsamp, int flags);

DLLEXPORT void tjFree(unsigned char *buffer);

DLLEXPORT tjscalingfactor *tjGetScalingFactors(int *numscalingfactors);

DLLEXPORT tjhandle tjInitTransform(void);

DLLEXPORT int tjTransform(tjhandle handle, const unsigned char *jpegBuf,
                            unsigned long jpegSize, int n,
                            unsigned char **dstBufs, unsigned long *dstSizes,
                            tjtransform *transforms, int flags);

/* TurboJPEG 1.2.1+ */

#define TJFLAG_FASTDCT  2048
#define TJFLAG_ACCURATEDCT  4096

/* TurboJPEG 1.4+ */

DLLEXPORT unsigned long tjBufSizeYUV2(int width, int align, int height,
                                      int subsamp);

DLLEXPORT int tjCompressFromYUV(tjhandle handle, const unsigned char *srcBuf,
                                int width, int align, int height, int subsamp,
                                unsigned char **jpegBuf,
                                unsigned long *jpegSize, int jpegQual,
                                int flags);

DLLEXPORT int tjCompressFromYUVPlanes(tjhandle handle,
                                      const unsigned char **srcPlanes,
                                      int width, const int *strides,
                                      int height, int subsamp,
                                      unsigned char **jpegBuf,
                                      unsigned long *jpegSize, int jpegQual,
                                      int flags);

DLLEXPORT int tjDecodeYUV(tjhandle handle, const unsigned char *srcBuf,
                          int align, int subsamp, unsigned char *dstBuf,
                          int width, int pitch, int height, int pixelFormat,
                          int flags);

DLLEXPORT int tjDecodeYUVPlanes(tjhandle handle,
                                const unsigned char **srcPlanes,
                                const int *strides, int subsamp,
                                unsigned char *dstBuf, int width, int pitch,
                                int height, int pixelFormat, int flags);

DLLEXPORT int tjDecompressHeader3(tjhandle handle,
                                  const unsigned char *jpegBuf,
                                  unsigned long jpegSize, int *width,
                                  int *height, int *jpegSubsamp,
                                  int *jpegColorspace);

DLLEXPORT int tjDecompressToYUV2(tjhandle handle, const unsigned char *jpegBuf,
                                 unsigned long jpegSize, unsigned char *dstBuf,
                                 int width, int align, int height, int flags);

DLLEXPORT int tjDecompressToYUVPlanes(tjhandle handle,
                                      const unsigned char *jpegBuf,
                                      unsigned long jpegSize,
                                      unsigned char **dstPlanes, int width,
                                      int *strides, int height, int flags);

DLLEXPORT int tjEncodeYUV3(tjhandle handle, const unsigned char *srcBuf,
                           int width, int pitch, int height, int pixelFormat,
                           unsigned char *dstBuf, int align, int subsamp,
                           int flags);

DLLEXPORT int tjEncodeYUVPlanes(tjhandle handle, const unsigned char *srcBuf,
                                int width, int pitch, int height,
                                int pixelFormat, unsigned char **dstPlanes,
                                int *strides, int subsamp, int flags);

DLLEXPORT int tjPlaneHeight(int componentID, int height, int subsamp);

DLLEXPORT unsigned long tjPlaneSizeYUV(int componentID, int width, int stride,
                                       int height, int subsamp);

DLLEXPORT int tjPlaneWidth(int componentID, int width, int subsamp);

/* TurboJPEG 2.0+ */

#define TJFLAG_STOPONWARNING  8192
#define TJFLAG_PROGRESSIVE  16384

DLLEXPORT int tjGetErrorCode(tjhandle handle);

DLLEXPORT char *tjGetErrorStr2(tjhandle handle);

DLLEXPORT unsigned char *tjLoadImage(const char *filename, int *width,
                                     int align, int *height, int *pixelFormat,
                                     int flags);

DLLEXPORT int tjSaveImage(const char *filename, unsigned char *buffer,
                          int width, int pitch, int height, int pixelFormat,
                          int flags);

/* TurboJPEG 2.1+ */

#define TJFLAG_LIMITSCANS  32768

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif

/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/**
 * # CategoryPixels
 *
 * SDL offers facilities for pixel management.
 *
 * Largely these facilities deal with pixel _format_: what does this set of
 * bits represent?
 *
 * If you mostly want to think of a pixel as some combination of red, green,
 * blue, and maybe alpha intensities, this is all pretty straightforward, and
 * in many cases, is enough information to build a perfectly fine game.
 *
 * However, the actual definition of a pixel is more complex than that:
 *
 * Pixels are a representation of a color in a particular color space.
 *
 * The first characteristic of a color space is the color type. SDL
 * understands two different color types, RGB and YCbCr, or in SDL also
 * referred to as YUV.
 *
 * RGB colors consist of red, green, and blue channels of color that are added
 * together to represent the colors we see on the screen.
 *
 * https://en.wikipedia.org/wiki/RGB_color_model
 *
 * YCbCr colors represent colors as a Y luma brightness component and red and
 * blue chroma color offsets. This color representation takes advantage of the
 * fact that the human eye is more sensitive to brightness than the color in
 * an image. The Cb and Cr components are often compressed and have lower
 * resolution than the luma component.
 *
 * https://en.wikipedia.org/wiki/YCbCr
 *
 * When the color information in YCbCr is compressed, the Y pixels are left at
 * full resolution and each Cr and Cb pixel represents an average of the color
 * information in a block of Y pixels. The chroma location determines where in
 * that block of pixels the color information is coming from.
 *
 * The color range defines how much of the pixel to use when converting a
 * pixel into a color on the display. When the full color range is used, the
 * entire numeric range of the pixel bits is significant. When narrow color
 * range is used, for historical reasons, the pixel uses only a portion of the
 * numeric range to represent colors.
 *
 * The color primaries and white point are a definition of the colors in the
 * color space relative to the standard XYZ color space.
 *
 * https://en.wikipedia.org/wiki/CIE_1931_color_space
 *
 * The transfer characteristic, or opto-electrical transfer function (OETF),
 * is the way a color is converted from mathematically linear space into a
 * non-linear output signals.
 *
 * https://en.wikipedia.org/wiki/Rec._709#Transfer_characteristics
 *
 * The matrix coefficients are used to convert between YCbCr and RGB colors.
 */

#ifndef SDL_pixels_h_
#define SDL_pixels_h_

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_endian.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * A fully opaque 8-bit alpha value.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_ALPHA_TRANSPARENT
 */
#define SDL_ALPHA_OPAQUE 255

/**
 * A fully opaque floating point alpha value.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_ALPHA_TRANSPARENT_FLOAT
 */
#define SDL_ALPHA_OPAQUE_FLOAT 1.0f

/**
 * A fully transparent 8-bit alpha value.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_ALPHA_OPAQUE
 */
#define SDL_ALPHA_TRANSPARENT 0

/**
 * A fully transparent floating point alpha value.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_ALPHA_OPAQUE_FLOAT
 */
#define SDL_ALPHA_TRANSPARENT_FLOAT 0.0f

/**
 * Pixel type.
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_PixelType
{
    SDL_PIXELTYPE_UNKNOWN,
    SDL_PIXELTYPE_INDEX1,
    SDL_PIXELTYPE_INDEX4,
    SDL_PIXELTYPE_INDEX8,
    SDL_PIXELTYPE_PACKED8,
    SDL_PIXELTYPE_PACKED16,
    SDL_PIXELTYPE_PACKED32,
    SDL_PIXELTYPE_ARRAYU8,
    SDL_PIXELTYPE_ARRAYU16,
    SDL_PIXELTYPE_ARRAYU32,
    SDL_PIXELTYPE_ARRAYF16,
    SDL_PIXELTYPE_ARRAYF32,
    /* appended at the end for compatibility with sdl2-compat:  */
    SDL_PIXELTYPE_INDEX2
} SDL_PixelType;

/**
 * Bitmap pixel order, high bit -> low bit.
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_BitmapOrder
{
    SDL_BITMAPORDER_NONE,
    SDL_BITMAPORDER_4321,
    SDL_BITMAPORDER_1234
} SDL_BitmapOrder;

/**
 * Packed component order, high bit -> low bit.
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_PackedOrder
{
    SDL_PACKEDORDER_NONE,
    SDL_PACKEDORDER_XRGB,
    SDL_PACKEDORDER_RGBX,
    SDL_PACKEDORDER_ARGB,
    SDL_PACKEDORDER_RGBA,
    SDL_PACKEDORDER_XBGR,
    SDL_PACKEDORDER_BGRX,
    SDL_PACKEDORDER_ABGR,
    SDL_PACKEDORDER_BGRA
} SDL_PackedOrder;

/**
 * Array component order, low byte -> high byte.
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_ArrayOrder
{
    SDL_ARRAYORDER_NONE,
    SDL_ARRAYORDER_RGB,
    SDL_ARRAYORDER_RGBA,
    SDL_ARRAYORDER_ARGB,
    SDL_ARRAYORDER_BGR,
    SDL_ARRAYORDER_BGRA,
    SDL_ARRAYORDER_ABGR
} SDL_ArrayOrder;

/**
 * Packed component layout.
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_PackedLayout
{
    SDL_PACKEDLAYOUT_NONE,
    SDL_PACKEDLAYOUT_332,
    SDL_PACKEDLAYOUT_4444,
    SDL_PACKEDLAYOUT_1555,
    SDL_PACKEDLAYOUT_5551,
    SDL_PACKEDLAYOUT_565,
    SDL_PACKEDLAYOUT_8888,
    SDL_PACKEDLAYOUT_2101010,
    SDL_PACKEDLAYOUT_1010102
} SDL_PackedLayout;

/**
 * A macro for defining custom FourCC pixel formats.
 *
 * For example, defining SDL_PIXELFORMAT_YV12 looks like this:
 *
 * ```c
 * SDL_DEFINE_PIXELFOURCC('Y', 'V', '1', '2')
 * ```
 *
 * \param A the first character of the FourCC code.
 * \param B the second character of the FourCC code.
 * \param C the third character of the FourCC code.
 * \param D the fourth character of the FourCC code.
 * \returns a format value in the style of SDL_PixelFormat.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_DEFINE_PIXELFOURCC(A, B, C, D) SDL_FOURCC(A, B, C, D)

/**
 * A macro for defining custom non-FourCC pixel formats.
 *
 * For example, defining SDL_PIXELFORMAT_RGBA8888 looks like this:
 *
 * ```c
 * SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_RGBA, SDL_PACKEDLAYOUT_8888, 32, 4)
 * ```
 *
 * \param type the type of the new format, probably a SDL_PixelType value.
 * \param order the order of the new format, probably a SDL_BitmapOrder,
 *              SDL_PackedOrder, or SDL_ArrayOrder value.
 * \param layout the layout of the new format, probably an SDL_PackedLayout
 *               value or zero.
 * \param bits the number of bits per pixel of the new format.
 * \param bytes the number of bytes per pixel of the new format.
 * \returns a format value in the style of SDL_PixelFormat.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_DEFINE_PIXELFORMAT(type, order, layout, bits, bytes) \
    ((1 << 28) | ((type) << 24) | ((order) << 20) | ((layout) << 16) | \
     ((bits) << 8) | ((bytes) << 0))

/**
 * A macro to retrieve the flags of an SDL_PixelFormat.
 *
 * This macro is generally not needed directly by an app, which should use
 * specific tests, like SDL_ISPIXELFORMAT_FOURCC, instead.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns the flags of `format`.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PIXELFLAG(format)    (((format) >> 28) & 0x0F)

/**
 * A macro to retrieve the type of an SDL_PixelFormat.
 *
 * This is usually a value from the SDL_PixelType enumeration.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns the type of `format`.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PIXELTYPE(format)    (((format) >> 24) & 0x0F)

/**
 * A macro to retrieve the order of an SDL_PixelFormat.
 *
 * This is usually a value from the SDL_BitmapOrder, SDL_PackedOrder, or
 * SDL_ArrayOrder enumerations, depending on the format type.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns the order of `format`.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PIXELORDER(format)   (((format) >> 20) & 0x0F)

/**
 * A macro to retrieve the layout of an SDL_PixelFormat.
 *
 * This is usually a value from the SDL_PackedLayout enumeration, or zero if a
 * layout doesn't make sense for the format type.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns the layout of `format`.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_PIXELLAYOUT(format)  (((format) >> 16) & 0x0F)

/**
 * A macro to determine an SDL_PixelFormat's bits per pixel.
 *
 * Note that this macro double-evaluates its parameter, so do not use
 * expressions with side-effects here.
 *
 * FourCC formats will report zero here, as it rarely makes sense to measure
 * them per-pixel.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns the bits-per-pixel of `format`.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_BYTESPERPIXEL
 */
#define SDL_BITSPERPIXEL(format) \
    (SDL_ISPIXELFORMAT_FOURCC(format) ? 0 : (((format) >> 8) & 0xFF))

/**
 * A macro to determine an SDL_PixelFormat's bytes per pixel.
 *
 * Note that this macro double-evaluates its parameter, so do not use
 * expressions with side-effects here.
 *
 * FourCC formats do their best here, but many of them don't have a meaningful
 * measurement of bytes per pixel.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns the bytes-per-pixel of `format`.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_BITSPERPIXEL
 */
#define SDL_BYTESPERPIXEL(format) \
    (SDL_ISPIXELFORMAT_FOURCC(format) ? \
        ((((format) == SDL_PIXELFORMAT_YUY2) || \
          ((format) == SDL_PIXELFORMAT_UYVY) || \
          ((format) == SDL_PIXELFORMAT_YVYU) || \
          ((format) == SDL_PIXELFORMAT_P010)) ? 2 : 1) : (((format) >> 0) & 0xFF))


/**
 * A macro to determine if an SDL_PixelFormat is an indexed format.
 *
 * Note that this macro double-evaluates its parameter, so do not use
 * expressions with side-effects here.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns true if the format is indexed, false otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ISPIXELFORMAT_INDEXED(format)   \
    (!SDL_ISPIXELFORMAT_FOURCC(format) && \
     ((SDL_PIXELTYPE(format) == SDL_PIXELTYPE_INDEX1) || \
      (SDL_PIXELTYPE(format) == SDL_PIXELTYPE_INDEX2) || \
      (SDL_PIXELTYPE(format) == SDL_PIXELTYPE_INDEX4) || \
      (SDL_PIXELTYPE(format) == SDL_PIXELTYPE_INDEX8)))

/**
 * A macro to determine if an SDL_PixelFormat is a packed format.
 *
 * Note that this macro double-evaluates its parameter, so do not use
 * expressions with side-effects here.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns true if the format is packed, false otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ISPIXELFORMAT_PACKED(format) \
    (!SDL_ISPIXELFORMAT_FOURCC(format) && \
     ((SDL_PIXELTYPE(format) == SDL_PIXELTYPE_PACKED8) || \
      (SDL_PIXELTYPE(format) == SDL_PIXELTYPE_PACKED16) || \
      (SDL_PIXELTYPE(format) == SDL_PIXELTYPE_PACKED32)))

/**
 * A macro to determine if an SDL_PixelFormat is an array format.
 *
 * Note that this macro double-evaluates its parameter, so do not use
 * expressions with side-effects here.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns true if the format is an array, false otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ISPIXELFORMAT_ARRAY(format) \
    (!SDL_ISPIXELFORMAT_FOURCC(format) && \
     ((SDL_PIXELTYPE(format) == SDL_PIXELTYPE_ARRAYU8) || \
      (SDL_PIXELTYPE(format) == SDL_PIXELTYPE_ARRAYU16) || \
      (SDL_PIXELTYPE(format) == SDL_PIXELTYPE_ARRAYU32) || \
      (SDL_PIXELTYPE(format) == SDL_PIXELTYPE_ARRAYF16) || \
      (SDL_PIXELTYPE(format) == SDL_PIXELTYPE_ARRAYF32)))

/**
 * A macro to determine if an SDL_PixelFormat is a 10-bit format.
 *
 * Note that this macro double-evaluates its parameter, so do not use
 * expressions with side-effects here.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns true if the format is 10-bit, false otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ISPIXELFORMAT_10BIT(format)    \
      (!SDL_ISPIXELFORMAT_FOURCC(format) && \
       ((SDL_PIXELTYPE(format) == SDL_PIXELTYPE_PACKED32) && \
        (SDL_PIXELLAYOUT(format) == SDL_PACKEDLAYOUT_2101010)))

/**
 * A macro to determine if an SDL_PixelFormat is a floating point format.
 *
 * Note that this macro double-evaluates its parameter, so do not use
 * expressions with side-effects here.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns true if the format is 10-bit, false otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ISPIXELFORMAT_FLOAT(format)    \
      (!SDL_ISPIXELFORMAT_FOURCC(format) && \
       ((SDL_PIXELTYPE(format) == SDL_PIXELTYPE_ARRAYF16) || \
        (SDL_PIXELTYPE(format) == SDL_PIXELTYPE_ARRAYF32)))

/**
 * A macro to determine if an SDL_PixelFormat has an alpha channel.
 *
 * Note that this macro double-evaluates its parameter, so do not use
 * expressions with side-effects here.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns true if the format has alpha, false otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ISPIXELFORMAT_ALPHA(format)   \
    ((SDL_ISPIXELFORMAT_PACKED(format) && \
      ((SDL_PIXELORDER(format) == SDL_PACKEDORDER_ARGB) || \
       (SDL_PIXELORDER(format) == SDL_PACKEDORDER_RGBA) || \
       (SDL_PIXELORDER(format) == SDL_PACKEDORDER_ABGR) || \
       (SDL_PIXELORDER(format) == SDL_PACKEDORDER_BGRA))) || \
     (SDL_ISPIXELFORMAT_ARRAY(format) && \
      ((SDL_PIXELORDER(format) == SDL_ARRAYORDER_ARGB) || \
       (SDL_PIXELORDER(format) == SDL_ARRAYORDER_RGBA) || \
       (SDL_PIXELORDER(format) == SDL_ARRAYORDER_ABGR) || \
       (SDL_PIXELORDER(format) == SDL_ARRAYORDER_BGRA))))


/**
 * A macro to determine if an SDL_PixelFormat is a "FourCC" format.
 *
 * This covers custom and other unusual formats.
 *
 * Note that this macro double-evaluates its parameter, so do not use
 * expressions with side-effects here.
 *
 * \param format an SDL_PixelFormat to check.
 * \returns true if the format has alpha, false otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ISPIXELFORMAT_FOURCC(format)  /* The flag is set to 1 because 0x1? is not in the printable ASCII range */ \
    ((format) && (SDL_PIXELFLAG(format) != 1))

/* Note: If you modify this enum, update SDL_GetPixelFormatName() */

/**
 * Pixel format.
 *
 * SDL's pixel formats have the following naming convention:
 *
 * - Names with a list of components and a single bit count, such as RGB24 and
 *   ABGR32, define a platform-independent encoding into bytes in the order
 *   specified. For example, in RGB24 data, each pixel is encoded in 3 bytes
 *   (red, green, blue) in that order, and in ABGR32 data, each pixel is
 *   encoded in 4 bytes alpha, blue, green, red) in that order. Use these
 *   names if the property of a format that is important to you is the order
 *   of the bytes in memory or on disk.
 * - Names with a bit count per component, such as ARGB8888 and XRGB1555, are
 *   "packed" into an appropriately-sized integer in the platform's native
 *   endianness. For example, ARGB8888 is a sequence of 32-bit integers; in
 *   each integer, the most significant bits are alpha, and the least
 *   significant bits are blue. On a little-endian CPU such as x86, the least
 *   significant bits of each integer are arranged first in memory, but on a
 *   big-endian CPU such as s390x, the most significant bits are arranged
 *   first. Use these names if the property of a format that is important to
 *   you is the meaning of each bit position within a native-endianness
 *   integer.
 * - In indexed formats such as INDEX4LSB, each pixel is represented by
 *   encoding an index into the palette into the indicated number of bits,
 *   with multiple pixels packed into each byte if appropriate. In LSB
 *   formats, the first (leftmost) pixel is stored in the least-significant
 *   bits of the byte; in MSB formats, it's stored in the most-significant
 *   bits. INDEX8 does not need LSB/MSB variants, because each pixel exactly
 *   fills one byte.
 *
 * The 32-bit byte-array encodings such as RGBA32 are aliases for the
 * appropriate 8888 encoding for the current platform. For example, RGBA32 is
 * an alias for ABGR8888 on little-endian CPUs like x86, or an alias for
 * RGBA8888 on big-endian CPUs.
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_PixelFormat
{
    SDL_PIXELFORMAT_UNKNOWN = 0,
    SDL_PIXELFORMAT_INDEX1LSB = 0x11100100u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_INDEX1, SDL_BITMAPORDER_4321, 0, 1, 0), */
    SDL_PIXELFORMAT_INDEX1MSB = 0x11200100u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_INDEX1, SDL_BITMAPORDER_1234, 0, 1, 0), */
    SDL_PIXELFORMAT_INDEX2LSB = 0x1c100200u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_INDEX2, SDL_BITMAPORDER_4321, 0, 2, 0), */
    SDL_PIXELFORMAT_INDEX2MSB = 0x1c200200u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_INDEX2, SDL_BITMAPORDER_1234, 0, 2, 0), */
    SDL_PIXELFORMAT_INDEX4LSB = 0x12100400u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_INDEX4, SDL_BITMAPORDER_4321, 0, 4, 0), */
    SDL_PIXELFORMAT_INDEX4MSB = 0x12200400u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_INDEX4, SDL_BITMAPORDER_1234, 0, 4, 0), */
    SDL_PIXELFORMAT_INDEX8 = 0x13000801u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_INDEX8, 0, 0, 8, 1), */
    SDL_PIXELFORMAT_RGB332 = 0x14110801u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED8, SDL_PACKEDORDER_XRGB, SDL_PACKEDLAYOUT_332, 8, 1), */
    SDL_PIXELFORMAT_XRGB4444 = 0x15120c02u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_XRGB, SDL_PACKEDLAYOUT_4444, 12, 2), */
    SDL_PIXELFORMAT_XBGR4444 = 0x15520c02u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_XBGR, SDL_PACKEDLAYOUT_4444, 12, 2), */
    SDL_PIXELFORMAT_XRGB1555 = 0x15130f02u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_XRGB, SDL_PACKEDLAYOUT_1555, 15, 2), */
    SDL_PIXELFORMAT_XBGR1555 = 0x15530f02u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_XBGR, SDL_PACKEDLAYOUT_1555, 15, 2), */
    SDL_PIXELFORMAT_ARGB4444 = 0x15321002u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_ARGB, SDL_PACKEDLAYOUT_4444, 16, 2), */
    SDL_PIXELFORMAT_RGBA4444 = 0x15421002u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_RGBA, SDL_PACKEDLAYOUT_4444, 16, 2), */
    SDL_PIXELFORMAT_ABGR4444 = 0x15721002u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_ABGR, SDL_PACKEDLAYOUT_4444, 16, 2), */
    SDL_PIXELFORMAT_BGRA4444 = 0x15821002u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_BGRA, SDL_PACKEDLAYOUT_4444, 16, 2), */
    SDL_PIXELFORMAT_ARGB1555 = 0x15331002u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_ARGB, SDL_PACKEDLAYOUT_1555, 16, 2), */
    SDL_PIXELFORMAT_RGBA5551 = 0x15441002u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_RGBA, SDL_PACKEDLAYOUT_5551, 16, 2), */
    SDL_PIXELFORMAT_ABGR1555 = 0x15731002u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_ABGR, SDL_PACKEDLAYOUT_1555, 16, 2), */
    SDL_PIXELFORMAT_BGRA5551 = 0x15841002u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_BGRA, SDL_PACKEDLAYOUT_5551, 16, 2), */
    SDL_PIXELFORMAT_RGB565 = 0x15151002u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_XRGB, SDL_PACKEDLAYOUT_565, 16, 2), */
    SDL_PIXELFORMAT_BGR565 = 0x15551002u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED16, SDL_PACKEDORDER_XBGR, SDL_PACKEDLAYOUT_565, 16, 2), */
    SDL_PIXELFORMAT_RGB24 = 0x17101803u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYU8, SDL_ARRAYORDER_RGB, 0, 24, 3), */
    SDL_PIXELFORMAT_BGR24 = 0x17401803u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYU8, SDL_ARRAYORDER_BGR, 0, 24, 3), */
    SDL_PIXELFORMAT_XRGB8888 = 0x16161804u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_XRGB, SDL_PACKEDLAYOUT_8888, 24, 4), */
    SDL_PIXELFORMAT_RGBX8888 = 0x16261804u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_RGBX, SDL_PACKEDLAYOUT_8888, 24, 4), */
    SDL_PIXELFORMAT_XBGR8888 = 0x16561804u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_XBGR, SDL_PACKEDLAYOUT_8888, 24, 4), */
    SDL_PIXELFORMAT_BGRX8888 = 0x16661804u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_BGRX, SDL_PACKEDLAYOUT_8888, 24, 4), */
    SDL_PIXELFORMAT_ARGB8888 = 0x16362004u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_ARGB, SDL_PACKEDLAYOUT_8888, 32, 4), */
    SDL_PIXELFORMAT_RGBA8888 = 0x16462004u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_RGBA, SDL_PACKEDLAYOUT_8888, 32, 4), */
    SDL_PIXELFORMAT_ABGR8888 = 0x16762004u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_ABGR, SDL_PACKEDLAYOUT_8888, 32, 4), */
    SDL_PIXELFORMAT_BGRA8888 = 0x16862004u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_BGRA, SDL_PACKEDLAYOUT_8888, 32, 4), */
    SDL_PIXELFORMAT_XRGB2101010 = 0x16172004u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_XRGB, SDL_PACKEDLAYOUT_2101010, 32, 4), */
    SDL_PIXELFORMAT_XBGR2101010 = 0x16572004u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_XBGR, SDL_PACKEDLAYOUT_2101010, 32, 4), */
    SDL_PIXELFORMAT_ARGB2101010 = 0x16372004u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_ARGB, SDL_PACKEDLAYOUT_2101010, 32, 4), */
    SDL_PIXELFORMAT_ABGR2101010 = 0x16772004u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_PACKED32, SDL_PACKEDORDER_ABGR, SDL_PACKEDLAYOUT_2101010, 32, 4), */
    SDL_PIXELFORMAT_RGB48 = 0x18103006u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYU16, SDL_ARRAYORDER_RGB, 0, 48, 6), */
    SDL_PIXELFORMAT_BGR48 = 0x18403006u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYU16, SDL_ARRAYORDER_BGR, 0, 48, 6), */
    SDL_PIXELFORMAT_RGBA64 = 0x18204008u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYU16, SDL_ARRAYORDER_RGBA, 0, 64, 8), */
    SDL_PIXELFORMAT_ARGB64 = 0x18304008u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYU16, SDL_ARRAYORDER_ARGB, 0, 64, 8), */
    SDL_PIXELFORMAT_BGRA64 = 0x18504008u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYU16, SDL_ARRAYORDER_BGRA, 0, 64, 8), */
    SDL_PIXELFORMAT_ABGR64 = 0x18604008u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYU16, SDL_ARRAYORDER_ABGR, 0, 64, 8), */
    SDL_PIXELFORMAT_RGB48_FLOAT = 0x1a103006u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYF16, SDL_ARRAYORDER_RGB, 0, 48, 6), */
    SDL_PIXELFORMAT_BGR48_FLOAT = 0x1a403006u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYF16, SDL_ARRAYORDER_BGR, 0, 48, 6), */
    SDL_PIXELFORMAT_RGBA64_FLOAT = 0x1a204008u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYF16, SDL_ARRAYORDER_RGBA, 0, 64, 8), */
    SDL_PIXELFORMAT_ARGB64_FLOAT = 0x1a304008u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYF16, SDL_ARRAYORDER_ARGB, 0, 64, 8), */
    SDL_PIXELFORMAT_BGRA64_FLOAT = 0x1a504008u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYF16, SDL_ARRAYORDER_BGRA, 0, 64, 8), */
    SDL_PIXELFORMAT_ABGR64_FLOAT = 0x1a604008u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYF16, SDL_ARRAYORDER_ABGR, 0, 64, 8), */
    SDL_PIXELFORMAT_RGB96_FLOAT = 0x1b10600cu,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYF32, SDL_ARRAYORDER_RGB, 0, 96, 12), */
    SDL_PIXELFORMAT_BGR96_FLOAT = 0x1b40600cu,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYF32, SDL_ARRAYORDER_BGR, 0, 96, 12), */
    SDL_PIXELFORMAT_RGBA128_FLOAT = 0x1b208010u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYF32, SDL_ARRAYORDER_RGBA, 0, 128, 16), */
    SDL_PIXELFORMAT_ARGB128_FLOAT = 0x1b308010u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYF32, SDL_ARRAYORDER_ARGB, 0, 128, 16), */
    SDL_PIXELFORMAT_BGRA128_FLOAT = 0x1b508010u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYF32, SDL_ARRAYORDER_BGRA, 0, 128, 16), */
    SDL_PIXELFORMAT_ABGR128_FLOAT = 0x1b608010u,
        /* SDL_DEFINE_PIXELFORMAT(SDL_PIXELTYPE_ARRAYF32, SDL_ARRAYORDER_ABGR, 0, 128, 16), */

    SDL_PIXELFORMAT_YV12 = 0x32315659u,      /**< Planar mode: Y + V + U  (3 planes) */
        /* SDL_DEFINE_PIXELFOURCC('Y', 'V', '1', '2'), */
    SDL_PIXELFORMAT_IYUV = 0x56555949u,      /**< Planar mode: Y + U + V  (3 planes) */
        /* SDL_DEFINE_PIXELFOURCC('I', 'Y', 'U', 'V'), */
    SDL_PIXELFORMAT_YUY2 = 0x32595559u,      /**< Packed mode: Y0+U0+Y1+V0 (1 plane) */
        /* SDL_DEFINE_PIXELFOURCC('Y', 'U', 'Y', '2'), */
    SDL_PIXELFORMAT_UYVY = 0x59565955u,      /**< Packed mode: U0+Y0+V0+Y1 (1 plane) */
        /* SDL_DEFINE_PIXELFOURCC('U', 'Y', 'V', 'Y'), */
    SDL_PIXELFORMAT_YVYU = 0x55595659u,      /**< Packed mode: Y0+V0+Y1+U0 (1 plane) */
        /* SDL_DEFINE_PIXELFOURCC('Y', 'V', 'Y', 'U'), */
    SDL_PIXELFORMAT_NV12 = 0x3231564eu,      /**< Planar mode: Y + U/V interleaved  (2 planes) */
        /* SDL_DEFINE_PIXELFOURCC('N', 'V', '1', '2'), */
    SDL_PIXELFORMAT_NV21 = 0x3132564eu,      /**< Planar mode: Y + V/U interleaved  (2 planes) */
        /* SDL_DEFINE_PIXELFOURCC('N', 'V', '2', '1'), */
    SDL_PIXELFORMAT_P010 = 0x30313050u,      /**< Planar mode: Y + U/V interleaved  (2 planes) */
        /* SDL_DEFINE_PIXELFOURCC('P', '0', '1', '0'), */
    SDL_PIXELFORMAT_EXTERNAL_OES = 0x2053454fu,     /**< Android video texture format */
        /* SDL_DEFINE_PIXELFOURCC('O', 'E', 'S', ' ') */

    SDL_PIXELFORMAT_MJPG = 0x47504a4du,     /**< Motion JPEG */
        /* SDL_DEFINE_PIXELFOURCC('M', 'J', 'P', 'G') */

    /* Aliases for RGBA byte arrays of color data, for the current platform */
    #if SDL_BYTEORDER == SDL_BIG_ENDIAN
    SDL_PIXELFORMAT_RGBA32 = SDL_PIXELFORMAT_RGBA8888,
    SDL_PIXELFORMAT_ARGB32 = SDL_PIXELFORMAT_ARGB8888,
    SDL_PIXELFORMAT_BGRA32 = SDL_PIXELFORMAT_BGRA8888,
    SDL_PIXELFORMAT_ABGR32 = SDL_PIXELFORMAT_ABGR8888,
    SDL_PIXELFORMAT_RGBX32 = SDL_PIXELFORMAT_RGBX8888,
    SDL_PIXELFORMAT_XRGB32 = SDL_PIXELFORMAT_XRGB8888,
    SDL_PIXELFORMAT_BGRX32 = SDL_PIXELFORMAT_BGRX8888,
    SDL_PIXELFORMAT_XBGR32 = SDL_PIXELFORMAT_XBGR8888
    #else
    SDL_PIXELFORMAT_RGBA32 = SDL_PIXELFORMAT_ABGR8888,
    SDL_PIXELFORMAT_ARGB32 = SDL_PIXELFORMAT_BGRA8888,
    SDL_PIXELFORMAT_BGRA32 = SDL_PIXELFORMAT_ARGB8888,
    SDL_PIXELFORMAT_ABGR32 = SDL_PIXELFORMAT_RGBA8888,
    SDL_PIXELFORMAT_RGBX32 = SDL_PIXELFORMAT_XBGR8888,
    SDL_PIXELFORMAT_XRGB32 = SDL_PIXELFORMAT_BGRX8888,
    SDL_PIXELFORMAT_BGRX32 = SDL_PIXELFORMAT_XRGB8888,
    SDL_PIXELFORMAT_XBGR32 = SDL_PIXELFORMAT_RGBX8888
    #endif
} SDL_PixelFormat;

/**
 * Colorspace color type.
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_ColorType
{
    SDL_COLOR_TYPE_UNKNOWN = 0,
    SDL_COLOR_TYPE_RGB = 1,
    SDL_COLOR_TYPE_YCBCR = 2
} SDL_ColorType;

/**
 * Colorspace color range, as described by
 * https://www.itu.int/rec/R-REC-BT.2100-2-201807-I/en
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_ColorRange
{
    SDL_COLOR_RANGE_UNKNOWN = 0,
    SDL_COLOR_RANGE_LIMITED = 1, /**< Narrow range, e.g. 16-235 for 8-bit RGB and luma, and 16-240 for 8-bit chroma */
    SDL_COLOR_RANGE_FULL = 2    /**< Full range, e.g. 0-255 for 8-bit RGB and luma, and 1-255 for 8-bit chroma */
} SDL_ColorRange;

/**
 * Colorspace color primaries, as described by
 * https://www.itu.int/rec/T-REC-H.273-201612-S/en
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_ColorPrimaries
{
    SDL_COLOR_PRIMARIES_UNKNOWN = 0,
    SDL_COLOR_PRIMARIES_BT709 = 1,                  /**< ITU-R BT.709-6 */
    SDL_COLOR_PRIMARIES_UNSPECIFIED = 2,
    SDL_COLOR_PRIMARIES_BT470M = 4,                 /**< ITU-R BT.470-6 System M */
    SDL_COLOR_PRIMARIES_BT470BG = 5,                /**< ITU-R BT.470-6 System B, G / ITU-R BT.601-7 625 */
    SDL_COLOR_PRIMARIES_BT601 = 6,                  /**< ITU-R BT.601-7 525, SMPTE 170M */
    SDL_COLOR_PRIMARIES_SMPTE240 = 7,               /**< SMPTE 240M, functionally the same as SDL_COLOR_PRIMARIES_BT601 */
    SDL_COLOR_PRIMARIES_GENERIC_FILM = 8,           /**< Generic film (color filters using Illuminant C) */
    SDL_COLOR_PRIMARIES_BT2020 = 9,                 /**< ITU-R BT.2020-2 / ITU-R BT.2100-0 */
    SDL_COLOR_PRIMARIES_XYZ = 10,                   /**< SMPTE ST 428-1 */
    SDL_COLOR_PRIMARIES_SMPTE431 = 11,              /**< SMPTE RP 431-2 */
    SDL_COLOR_PRIMARIES_SMPTE432 = 12,              /**< SMPTE EG 432-1 / DCI P3 */
    SDL_COLOR_PRIMARIES_EBU3213 = 22,               /**< EBU Tech. 3213-E */
    SDL_COLOR_PRIMARIES_CUSTOM = 31
} SDL_ColorPrimaries;

/**
 * Colorspace transfer characteristics.
 *
 * These are as described by https://www.itu.int/rec/T-REC-H.273-201612-S/en
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_TransferCharacteristics
{
    SDL_TRANSFER_CHARACTERISTICS_UNKNOWN = 0,
    SDL_TRANSFER_CHARACTERISTICS_BT709 = 1,         /**< Rec. ITU-R BT.709-6 / ITU-R BT1361 */
    SDL_TRANSFER_CHARACTERISTICS_UNSPECIFIED = 2,
    SDL_TRANSFER_CHARACTERISTICS_GAMMA22 = 4,       /**< ITU-R BT.470-6 System M / ITU-R BT1700 625 PAL & SECAM */
    SDL_TRANSFER_CHARACTERISTICS_GAMMA28 = 5,       /**< ITU-R BT.470-6 System B, G */
    SDL_TRANSFER_CHARACTERISTICS_BT601 = 6,         /**< SMPTE ST 170M / ITU-R BT.601-7 525 or 625 */
    SDL_TRANSFER_CHARACTERISTICS_SMPTE240 = 7,      /**< SMPTE ST 240M */
    SDL_TRANSFER_CHARACTERISTICS_LINEAR = 8,
    SDL_TRANSFER_CHARACTERISTICS_LOG100 = 9,
    SDL_TRANSFER_CHARACTERISTICS_LOG100_SQRT10 = 10,
    SDL_TRANSFER_CHARACTERISTICS_IEC61966 = 11,     /**< IEC 61966-2-4 */
    SDL_TRANSFER_CHARACTERISTICS_BT1361 = 12,       /**< ITU-R BT1361 Extended Colour Gamut */
    SDL_TRANSFER_CHARACTERISTICS_SRGB = 13,         /**< IEC 61966-2-1 (sRGB or sYCC) */
    SDL_TRANSFER_CHARACTERISTICS_BT2020_10BIT = 14, /**< ITU-R BT2020 for 10-bit system */
    SDL_TRANSFER_CHARACTERISTICS_BT2020_12BIT = 15, /**< ITU-R BT2020 for 12-bit system */
    SDL_TRANSFER_CHARACTERISTICS_PQ = 16,           /**< SMPTE ST 2084 for 10-, 12-, 14- and 16-bit systems */
    SDL_TRANSFER_CHARACTERISTICS_SMPTE428 = 17,     /**< SMPTE ST 428-1 */
    SDL_TRANSFER_CHARACTERISTICS_HLG = 18,          /**< ARIB STD-B67, known as "hybrid log-gamma" (HLG) */
    SDL_TRANSFER_CHARACTERISTICS_CUSTOM = 31
} SDL_TransferCharacteristics;

/**
 * Colorspace matrix coefficients.
 *
 * These are as described by https://www.itu.int/rec/T-REC-H.273-201612-S/en
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_MatrixCoefficients
{
    SDL_MATRIX_COEFFICIENTS_IDENTITY = 0,
    SDL_MATRIX_COEFFICIENTS_BT709 = 1,              /**< ITU-R BT.709-6 */
    SDL_MATRIX_COEFFICIENTS_UNSPECIFIED = 2,
    SDL_MATRIX_COEFFICIENTS_FCC = 4,                /**< US FCC Title 47 */
    SDL_MATRIX_COEFFICIENTS_BT470BG = 5,            /**< ITU-R BT.470-6 System B, G / ITU-R BT.601-7 625, functionally the same as SDL_MATRIX_COEFFICIENTS_BT601 */
    SDL_MATRIX_COEFFICIENTS_BT601 = 6,              /**< ITU-R BT.601-7 525 */
    SDL_MATRIX_COEFFICIENTS_SMPTE240 = 7,           /**< SMPTE 240M */
    SDL_MATRIX_COEFFICIENTS_YCGCO = 8,
    SDL_MATRIX_COEFFICIENTS_BT2020_NCL = 9,         /**< ITU-R BT.2020-2 non-constant luminance */
    SDL_MATRIX_COEFFICIENTS_BT2020_CL = 10,         /**< ITU-R BT.2020-2 constant luminance */
    SDL_MATRIX_COEFFICIENTS_SMPTE2085 = 11,         /**< SMPTE ST 2085 */
    SDL_MATRIX_COEFFICIENTS_CHROMA_DERIVED_NCL = 12,
    SDL_MATRIX_COEFFICIENTS_CHROMA_DERIVED_CL = 13,
    SDL_MATRIX_COEFFICIENTS_ICTCP = 14,             /**< ITU-R BT.2100-0 ICTCP */
    SDL_MATRIX_COEFFICIENTS_CUSTOM = 31
} SDL_MatrixCoefficients;

/**
 * Colorspace chroma sample location.
 *
 * \since This enum is available since SDL 3.2.0.
 */
typedef enum SDL_ChromaLocation
{
    SDL_CHROMA_LOCATION_NONE = 0,   /**< RGB, no chroma sampling */
    SDL_CHROMA_LOCATION_LEFT = 1,   /**< In MPEG-2, MPEG-4, and AVC, Cb and Cr are taken on midpoint of the left-edge of the 2x2 square. In other words, they have the same horizontal location as the top-left pixel, but is shifted one-half pixel down vertically. */
    SDL_CHROMA_LOCATION_CENTER = 2, /**< In JPEG/JFIF, H.261, and MPEG-1, Cb and Cr are taken at the center of the 2x2 square. In other words, they are offset one-half pixel to the right and one-half pixel down compared to the top-left pixel. */
    SDL_CHROMA_LOCATION_TOPLEFT = 3 /**< In HEVC for BT.2020 and BT.2100 content (in particular on Blu-rays), Cb and Cr are sampled at the same location as the group's top-left Y pixel ("co-sited", "co-located"). */
} SDL_ChromaLocation;


/* Colorspace definition */

/**
 * A macro for defining custom SDL_Colorspace formats.
 *
 * For example, defining SDL_COLORSPACE_SRGB looks like this:
 *
 * ```c
 * SDL_DEFINE_COLORSPACE(SDL_COLOR_TYPE_RGB,
 *                       SDL_COLOR_RANGE_FULL,
 *                       SDL_COLOR_PRIMARIES_BT709,
 *                       SDL_TRANSFER_CHARACTERISTICS_SRGB,
 *                       SDL_MATRIX_COEFFICIENTS_IDENTITY,
 *                       SDL_CHROMA_LOCATION_NONE)
 * ```
 *
 * \param type the type of the new format, probably an SDL_ColorType value.
 * \param range the range of the new format, probably a SDL_ColorRange value.
 * \param primaries the primaries of the new format, probably an
 *                  SDL_ColorPrimaries value.
 * \param transfer the transfer characteristics of the new format, probably an
 *                 SDL_TransferCharacteristics value.
 * \param matrix the matrix coefficients of the new format, probably an
 *               SDL_MatrixCoefficients value.
 * \param chroma the chroma sample location of the new format, probably an
 *               SDL_ChromaLocation value.
 * \returns a format value in the style of SDL_Colorspace.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_DEFINE_COLORSPACE(type, range, primaries, transfer, matrix, chroma) \
    (((Uint32)(type) << 28) | ((Uint32)(range) << 24) | ((Uint32)(chroma) << 20) | \
    ((Uint32)(primaries) << 10) | ((Uint32)(transfer) << 5) | ((Uint32)(matrix) << 0))

/**
 * A macro to retrieve the type of an SDL_Colorspace.
 *
 * \param cspace an SDL_Colorspace to check.
 * \returns the SDL_ColorType for `cspace`.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_COLORSPACETYPE(cspace)       (SDL_ColorType)(((cspace) >> 28) & 0x0F)

/**
 * A macro to retrieve the range of an SDL_Colorspace.
 *
 * \param cspace an SDL_Colorspace to check.
 * \returns the SDL_ColorRange of `cspace`.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_COLORSPACERANGE(cspace)      (SDL_ColorRange)(((cspace) >> 24) & 0x0F)

/**
 * A macro to retrieve the chroma sample location of an SDL_Colorspace.
 *
 * \param cspace an SDL_Colorspace to check.
 * \returns the SDL_ChromaLocation of `cspace`.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_COLORSPACECHROMA(cspace)     (SDL_ChromaLocation)(((cspace) >> 20) & 0x0F)

/**
 * A macro to retrieve the primaries of an SDL_Colorspace.
 *
 * \param cspace an SDL_Colorspace to check.
 * \returns the SDL_ColorPrimaries of `cspace`.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_COLORSPACEPRIMARIES(cspace)  (SDL_ColorPrimaries)(((cspace) >> 10) & 0x1F)

/**
 * A macro to retrieve the transfer characteristics of an SDL_Colorspace.
 *
 * \param cspace an SDL_Colorspace to check.
 * \returns the SDL_TransferCharacteristics of `cspace`.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_COLORSPACETRANSFER(cspace)   (SDL_TransferCharacteristics)(((cspace) >> 5) & 0x1F)

/**
 * A macro to retrieve the matrix coefficients of an SDL_Colorspace.
 *
 * \param cspace an SDL_Colorspace to check.
 * \returns the SDL_MatrixCoefficients of `cspace`.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_COLORSPACEMATRIX(cspace)     (SDL_MatrixCoefficients)((cspace) & 0x1F)

/**
 * A macro to determine if an SDL_Colorspace uses BT601 (or BT470BG) matrix
 * coefficients.
 *
 * Note that this macro double-evaluates its parameter, so do not use
 * expressions with side-effects here.
 *
 * \param cspace an SDL_Colorspace to check.
 * \returns true if BT601 or BT470BG, false otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ISCOLORSPACE_MATRIX_BT601(cspace)        (SDL_COLORSPACEMATRIX(cspace) == SDL_MATRIX_COEFFICIENTS_BT601 || SDL_COLORSPACEMATRIX(cspace) == SDL_MATRIX_COEFFICIENTS_BT470BG)

/**
 * A macro to determine if an SDL_Colorspace uses BT709 matrix coefficients.
 *
 * \param cspace an SDL_Colorspace to check.
 * \returns true if BT709, false otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ISCOLORSPACE_MATRIX_BT709(cspace)        (SDL_COLORSPACEMATRIX(cspace) == SDL_MATRIX_COEFFICIENTS_BT709)

/**
 * A macro to determine if an SDL_Colorspace uses BT2020_NCL matrix
 * coefficients.
 *
 * \param cspace an SDL_Colorspace to check.
 * \returns true if BT2020_NCL, false otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ISCOLORSPACE_MATRIX_BT2020_NCL(cspace)   (SDL_COLORSPACEMATRIX(cspace) == SDL_MATRIX_COEFFICIENTS_BT2020_NCL)

/**
 * A macro to determine if an SDL_Colorspace has a limited range.
 *
 * \param cspace an SDL_Colorspace to check.
 * \returns true if limited range, false otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ISCOLORSPACE_LIMITED_RANGE(cspace)       (SDL_COLORSPACERANGE(cspace) != SDL_COLOR_RANGE_FULL)

/**
 * A macro to determine if an SDL_Colorspace has a full range.
 *
 * \param cspace an SDL_Colorspace to check.
 * \returns true if full range, false otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ISCOLORSPACE_FULL_RANGE(cspace)          (SDL_COLORSPACERANGE(cspace) == SDL_COLOR_RANGE_FULL)

/**
 * Colorspace definitions.
 *
 * Since similar colorspaces may vary in their details (matrix, transfer
 * function, etc.), this is not an exhaustive list, but rather a
 * representative sample of the kinds of colorspaces supported in SDL.
 *
 * \since This enum is available since SDL 3.2.0.
 *
 * \sa SDL_ColorPrimaries
 * \sa SDL_ColorRange
 * \sa SDL_ColorType
 * \sa SDL_MatrixCoefficients
 * \sa SDL_TransferCharacteristics
 */
typedef enum SDL_Colorspace
{
    SDL_COLORSPACE_UNKNOWN = 0,

    /* sRGB is a gamma corrected colorspace, and the default colorspace for SDL rendering and 8-bit RGB surfaces */
    SDL_COLORSPACE_SRGB = 0x120005a0u, /**< Equivalent to DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709 */
        /* SDL_DEFINE_COLORSPACE(SDL_COLOR_TYPE_RGB,
                                 SDL_COLOR_RANGE_FULL,
                                 SDL_COLOR_PRIMARIES_BT709,
                                 SDL_TRANSFER_CHARACTERISTICS_SRGB,
                                 SDL_MATRIX_COEFFICIENTS_IDENTITY,
                                 SDL_CHROMA_LOCATION_NONE), */

    /* This is a linear colorspace and the default colorspace for floating point surfaces. On Windows this is the scRGB colorspace, and on Apple platforms this is kCGColorSpaceExtendedLinearSRGB for EDR content */
    SDL_COLORSPACE_SRGB_LINEAR = 0x12000500u, /**< Equivalent to DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709  */
        /* SDL_DEFINE_COLORSPACE(SDL_COLOR_TYPE_RGB,
                                 SDL_COLOR_RANGE_FULL,
                                 SDL_COLOR_PRIMARIES_BT709,
                                 SDL_TRANSFER_CHARACTERISTICS_LINEAR,
                                 SDL_MATRIX_COEFFICIENTS_IDENTITY,
                                 SDL_CHROMA_LOCATION_NONE), */

    /* HDR10 is a non-linear HDR colorspace and the default colorspace for 10-bit surfaces */
    SDL_COLORSPACE_HDR10 = 0x12002600u, /**< Equivalent to DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020  */
        /* SDL_DEFINE_COLORSPACE(SDL_COLOR_TYPE_RGB,
                                 SDL_COLOR_RANGE_FULL,
                                 SDL_COLOR_PRIMARIES_BT2020,
                                 SDL_TRANSFER_CHARACTERISTICS_PQ,
                                 SDL_MATRIX_COEFFICIENTS_IDENTITY,
                                 SDL_CHROMA_LOCATION_NONE), */

    SDL_COLORSPACE_JPEG = 0x220004c6u, /**< Equivalent to DXGI_COLOR_SPACE_YCBCR_FULL_G22_NONE_P709_X601 */
        /* SDL_DEFINE_COLORSPACE(SDL_COLOR_TYPE_YCBCR,
                                 SDL_COLOR_RANGE_FULL,
                                 SDL_COLOR_PRIMARIES_BT709,
                                 SDL_TRANSFER_CHARACTERISTICS_BT601,
                                 SDL_MATRIX_COEFFICIENTS_BT601,
                                 SDL_CHROMA_LOCATION_NONE), */

    SDL_COLORSPACE_BT601_LIMITED = 0x211018c6u, /**< Equivalent to DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601 */
        /* SDL_DEFINE_COLORSPACE(SDL_COLOR_TYPE_YCBCR,
                                 SDL_COLOR_RANGE_LIMITED,
                                 SDL_COLOR_PRIMARIES_BT601,
                                 SDL_TRANSFER_CHARACTERISTICS_BT601,
                                 SDL_MATRIX_COEFFICIENTS_BT601,
                                 SDL_CHROMA_LOCATION_LEFT), */

    SDL_COLORSPACE_BT601_FULL = 0x221018c6u, /**< Equivalent to DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601 */
        /* SDL_DEFINE_COLORSPACE(SDL_COLOR_TYPE_YCBCR,
                                 SDL_COLOR_RANGE_FULL,
                                 SDL_COLOR_PRIMARIES_BT601,
                                 SDL_TRANSFER_CHARACTERISTICS_BT601,
                                 SDL_MATRIX_COEFFICIENTS_BT601,
                                 SDL_CHROMA_LOCATION_LEFT), */

    SDL_COLORSPACE_BT709_LIMITED = 0x21100421u, /**< Equivalent to DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709 */
        /* SDL_DEFINE_COLORSPACE(SDL_COLOR_TYPE_YCBCR,
                                 SDL_COLOR_RANGE_LIMITED,
                                 SDL_COLOR_PRIMARIES_BT709,
                                 SDL_TRANSFER_CHARACTERISTICS_BT709,
                                 SDL_MATRIX_COEFFICIENTS_BT709,
                                 SDL_CHROMA_LOCATION_LEFT), */

    SDL_COLORSPACE_BT709_FULL = 0x22100421u, /**< Equivalent to DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709 */
        /* SDL_DEFINE_COLORSPACE(SDL_COLOR_TYPE_YCBCR,
                                 SDL_COLOR_RANGE_FULL,
                                 SDL_COLOR_PRIMARIES_BT709,
                                 SDL_TRANSFER_CHARACTERISTICS_BT709,
                                 SDL_MATRIX_COEFFICIENTS_BT709,
                                 SDL_CHROMA_LOCATION_LEFT), */

    SDL_COLORSPACE_BT2020_LIMITED = 0x21102609u, /**< Equivalent to DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P2020 */
        /* SDL_DEFINE_COLORSPACE(SDL_COLOR_TYPE_YCBCR,
                                 SDL_COLOR_RANGE_LIMITED,
                                 SDL_COLOR_PRIMARIES_BT2020,
                                 SDL_TRANSFER_CHARACTERISTICS_PQ,
                                 SDL_MATRIX_COEFFICIENTS_BT2020_NCL,
                                 SDL_CHROMA_LOCATION_LEFT), */

    SDL_COLORSPACE_BT2020_FULL = 0x22102609u, /**< Equivalent to DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P2020 */
        /* SDL_DEFINE_COLORSPACE(SDL_COLOR_TYPE_YCBCR,
                                 SDL_COLOR_RANGE_FULL,
                                 SDL_COLOR_PRIMARIES_BT2020,
                                 SDL_TRANSFER_CHARACTERISTICS_PQ,
                                 SDL_MATRIX_COEFFICIENTS_BT2020_NCL,
                                 SDL_CHROMA_LOCATION_LEFT), */

    SDL_COLORSPACE_RGB_DEFAULT = SDL_COLORSPACE_SRGB, /**< The default colorspace for RGB surfaces if no colorspace is specified */
    SDL_COLORSPACE_YUV_DEFAULT = SDL_COLORSPACE_JPEG  /**< The default colorspace for YUV surfaces if no colorspace is specified */
} SDL_Colorspace;

/**
 * A structure that represents a color as RGBA components.
 *
 * The bits of this structure can be directly reinterpreted as an
 * integer-packed color which uses the SDL_PIXELFORMAT_RGBA32 format
 * (SDL_PIXELFORMAT_ABGR8888 on little-endian systems and
 * SDL_PIXELFORMAT_RGBA8888 on big-endian systems).
 *
 * \since This struct is available since SDL 3.2.0.
 */
typedef struct SDL_Color
{
    Uint8 r;
    Uint8 g;
    Uint8 b;
    Uint8 a;
} SDL_Color;

/**
 * The bits of this structure can be directly reinterpreted as a float-packed
 * color which uses the SDL_PIXELFORMAT_RGBA128_FLOAT format
 *
 * \since This struct is available since SDL 3.2.0.
 */
typedef struct SDL_FColor
{
    float r;
    float g;
    float b;
    float a;
} SDL_FColor;

/**
 * A set of indexed colors representing a palette.
 *
 * \since This struct is available since SDL 3.2.0.
 *
 * \sa SDL_SetPaletteColors
 */
typedef struct SDL_Palette
{
    int ncolors;        /**< number of elements in `colors`. */
    SDL_Color *colors;  /**< an array of colors, `ncolors` long. */
    Uint32 version;     /**< internal use only, do not touch. */
    int refcount;       /**< internal use only, do not touch. */
} SDL_Palette;

/**
 * Details about the format of a pixel.
 *
 * \since This struct is available since SDL 3.2.0.
 */
typedef struct SDL_PixelFormatDetails
{
    SDL_PixelFormat format;
    Uint8 bits_per_pixel;
    Uint8 bytes_per_pixel;
    Uint8 padding[2];
    Uint32 Rmask;
    Uint32 Gmask;
    Uint32 Bmask;
    Uint32 Amask;
    Uint8 Rbits;
    Uint8 Gbits;
    Uint8 Bbits;
    Uint8 Abits;
    Uint8 Rshift;
    Uint8 Gshift;
    Uint8 Bshift;
    Uint8 Ashift;
} SDL_PixelFormatDetails;

/**
 * Get the human readable name of a pixel format.
 *
 * \param format the pixel format to query.
 * \returns the human readable name of the specified pixel format or
 *          "SDL_PIXELFORMAT_UNKNOWN" if the format isn't recognized.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC const char * SDLCALL SDL_GetPixelFormatName(SDL_PixelFormat format);

/**
 * Convert one of the enumerated pixel formats to a bpp value and RGBA masks.
 *
 * \param format one of the SDL_PixelFormat values.
 * \param bpp a bits per pixel value; usually 15, 16, or 32.
 * \param Rmask a pointer filled in with the red mask for the format.
 * \param Gmask a pointer filled in with the green mask for the format.
 * \param Bmask a pointer filled in with the blue mask for the format.
 * \param Amask a pointer filled in with the alpha mask for the format.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPixelFormatForMasks
 */
extern SDL_DECLSPEC bool SDLCALL SDL_GetMasksForPixelFormat(SDL_PixelFormat format, int *bpp, Uint32 *Rmask, Uint32 *Gmask, Uint32 *Bmask, Uint32 *Amask);

/**
 * Convert a bpp value and RGBA masks to an enumerated pixel format.
 *
 * This will return `SDL_PIXELFORMAT_UNKNOWN` if the conversion wasn't
 * possible.
 *
 * \param bpp a bits per pixel value; usually 15, 16, or 32.
 * \param Rmask the red mask for the format.
 * \param Gmask the green mask for the format.
 * \param Bmask the blue mask for the format.
 * \param Amask the alpha mask for the format.
 * \returns the SDL_PixelFormat value corresponding to the format masks, or
 *          SDL_PIXELFORMAT_UNKNOWN if there isn't a match.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetMasksForPixelFormat
 */
extern SDL_DECLSPEC SDL_PixelFormat SDLCALL SDL_GetPixelFormatForMasks(int bpp, Uint32 Rmask, Uint32 Gmask, Uint32 Bmask, Uint32 Amask);

/**
 * Create an SDL_PixelFormatDetails structure corresponding to a pixel format.
 *
 * Returned structure may come from a shared global cache (i.e. not newly
 * allocated), and hence should not be modified, especially the palette. Weird
 * errors such as `Blit combination not supported` may occur.
 *
 * \param format one of the SDL_PixelFormat values.
 * \returns a pointer to a SDL_PixelFormatDetails structure or NULL on
 *          failure; call SDL_GetError() for more information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC const SDL_PixelFormatDetails * SDLCALL SDL_GetPixelFormatDetails(SDL_PixelFormat format);

/**
 * Create a palette structure with the specified number of color entries.
 *
 * The palette entries are initialized to white.
 *
 * \param ncolors represents the number of color entries in the color palette.
 * \returns a new SDL_Palette structure on success or NULL on failure (e.g. if
 *          there wasn't enough memory); call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_DestroyPalette
 * \sa SDL_SetPaletteColors
 * \sa SDL_SetSurfacePalette
 */
extern SDL_DECLSPEC SDL_Palette * SDLCALL SDL_CreatePalette(int ncolors);

/**
 * Set a range of colors in a palette.
 *
 * \param palette the SDL_Palette structure to modify.
 * \param colors an array of SDL_Color structures to copy into the palette.
 * \param firstcolor the index of the first palette entry to modify.
 * \param ncolors the number of entries to modify.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \threadsafety It is safe to call this function from any thread, as long as
 *               the palette is not modified or destroyed in another thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_SetPaletteColors(SDL_Palette *palette, const SDL_Color *colors, int firstcolor, int ncolors);

/**
 * Free a palette created with SDL_CreatePalette().
 *
 * \param palette the SDL_Palette structure to be freed.
 *
 * \threadsafety It is safe to call this function from any thread, as long as
 *               the palette is not modified or destroyed in another thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CreatePalette
 */
extern SDL_DECLSPEC void SDLCALL SDL_DestroyPalette(SDL_Palette *palette);

/**
 * Map an RGB triple to an opaque pixel value for a given pixel format.
 *
 * This function maps the RGB color value to the specified pixel format and
 * returns the pixel value best approximating the given RGB color value for
 * the given pixel format.
 *
 * If the format has a palette (8-bit) the index of the closest matching color
 * in the palette will be returned.
 *
 * If the specified pixel format has an alpha component it will be returned as
 * all 1 bits (fully opaque).
 *
 * If the pixel format bpp (color depth) is less than 32-bpp then the unused
 * upper bits of the return value can safely be ignored (e.g., with a 16-bpp
 * format the return value can be assigned to a Uint16, and similarly a Uint8
 * for an 8-bpp format).
 *
 * \param format a pointer to SDL_PixelFormatDetails describing the pixel
 *               format.
 * \param palette an optional palette for indexed formats, may be NULL.
 * \param r the red component of the pixel in the range 0-255.
 * \param g the green component of the pixel in the range 0-255.
 * \param b the blue component of the pixel in the range 0-255.
 * \returns a pixel value.
 *
 * \threadsafety It is safe to call this function from any thread, as long as
 *               the palette is not modified.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPixelFormatDetails
 * \sa SDL_GetRGB
 * \sa SDL_MapRGBA
 * \sa SDL_MapSurfaceRGB
 */
extern SDL_DECLSPEC Uint32 SDLCALL SDL_MapRGB(const SDL_PixelFormatDetails *format, const SDL_Palette *palette, Uint8 r, Uint8 g, Uint8 b);

/**
 * Map an RGBA quadruple to a pixel value for a given pixel format.
 *
 * This function maps the RGBA color value to the specified pixel format and
 * returns the pixel value best approximating the given RGBA color value for
 * the given pixel format.
 *
 * If the specified pixel format has no alpha component the alpha value will
 * be ignored (as it will be in formats with a palette).
 *
 * If the format has a palette (8-bit) the index of the closest matching color
 * in the palette will be returned.
 *
 * If the pixel format bpp (color depth) is less than 32-bpp then the unused
 * upper bits of the return value can safely be ignored (e.g., with a 16-bpp
 * format the return value can be assigned to a Uint16, and similarly a Uint8
 * for an 8-bpp format).
 *
 * \param format a pointer to SDL_PixelFormatDetails describing the pixel
 *               format.
 * \param palette an optional palette for indexed formats, may be NULL.
 * \param r the red component of the pixel in the range 0-255.
 * \param g the green component of the pixel in the range 0-255.
 * \param b the blue component of the pixel in the range 0-255.
 * \param a the alpha component of the pixel in the range 0-255.
 * \returns a pixel value.
 *
 * \threadsafety It is safe to call this function from any thread, as long as
 *               the palette is not modified.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPixelFormatDetails
 * \sa SDL_GetRGBA
 * \sa SDL_MapRGB
 * \sa SDL_MapSurfaceRGBA
 */
extern SDL_DECLSPEC Uint32 SDLCALL SDL_MapRGBA(const SDL_PixelFormatDetails *format, const SDL_Palette *palette, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

/**
 * Get RGB values from a pixel in the specified format.
 *
 * This function uses the entire 8-bit [0..255] range when converting color
 * components from pixel formats with less than 8-bits per RGB component
 * (e.g., a completely white pixel in 16-bit RGB565 format would return [0xff,
 * 0xff, 0xff] not [0xf8, 0xfc, 0xf8]).
 *
 * \param pixel a pixel value.
 * \param format a pointer to SDL_PixelFormatDetails describing the pixel
 *               format.
 * \param palette an optional palette for indexed formats, may be NULL.
 * \param r a pointer filled in with the red component, may be NULL.
 * \param g a pointer filled in with the green component, may be NULL.
 * \param b a pointer filled in with the blue component, may be NULL.
 *
 * \threadsafety It is safe to call this function from any thread, as long as
 *               the palette is not modified.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPixelFormatDetails
 * \sa SDL_GetRGBA
 * \sa SDL_MapRGB
 * \sa SDL_MapRGBA
 */
extern SDL_DECLSPEC void SDLCALL SDL_GetRGB(Uint32 pixel, const SDL_PixelFormatDetails *format, const SDL_Palette *palette, Uint8 *r, Uint8 *g, Uint8 *b);

/**
 * Get RGBA values from a pixel in the specified format.
 *
 * This function uses the entire 8-bit [0..255] range when converting color
 * components from pixel formats with less than 8-bits per RGB component
 * (e.g., a completely white pixel in 16-bit RGB565 format would return [0xff,
 * 0xff, 0xff] not [0xf8, 0xfc, 0xf8]).
 *
 * If the surface has no alpha component, the alpha will be returned as 0xff
 * (100% opaque).
 *
 * \param pixel a pixel value.
 * \param format a pointer to SDL_PixelFormatDetails describing the pixel
 *               format.
 * \param palette an optional palette for indexed formats, may be NULL.
 * \param r a pointer filled in with the red component, may be NULL.
 * \param g a pointer filled in with the green component, may be NULL.
 * \param b a pointer filled in with the blue component, may be NULL.
 * \param a a pointer filled in with the alpha component, may be NULL.
 *
 * \threadsafety It is safe to call this function from any thread, as long as
 *               the palette is not modified.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetPixelFormatDetails
 * \sa SDL_GetRGB
 * \sa SDL_MapRGB
 * \sa SDL_MapRGBA
 */
extern SDL_DECLSPEC void SDLCALL SDL_GetRGBA(Uint32 pixel, const SDL_PixelFormatDetails *format, const SDL_Palette *palette, Uint8 *r, Uint8 *g, Uint8 *b, Uint8 *a);


/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_pixels_h_ */

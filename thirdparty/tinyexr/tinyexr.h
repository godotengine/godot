/*
Copyright (c) 2014 - 2019, Syoyo Fujita and many contributors.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Syoyo Fujita nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// TinyEXR contains some OpenEXR code, which is licensed under ------------

///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

// End of OpenEXR license -------------------------------------------------

#ifndef TINYEXR_H_
#define TINYEXR_H_

//
//
//   Do this:
//    #define TINYEXR_IMPLEMENTATION
//   before you include this file in *one* C or C++ file to create the
//   implementation.
//
//   // i.e. it should look like this:
//   #include ...
//   #include ...
//   #include ...
//   #define TINYEXR_IMPLEMENTATION
//   #include "tinyexr.h"
//
//

#include <stddef.h>  // for size_t
#include <stdint.h>  // guess stdint.h is available(C99)

#ifdef __cplusplus
extern "C" {
#endif

// Use embedded miniz or not to decode ZIP format pixel. Linking with zlib
// required if this flas is 0.
#ifndef TINYEXR_USE_MINIZ
#define TINYEXR_USE_MINIZ (1)
#endif

// Disable PIZ comporession when applying cpplint.
#ifndef TINYEXR_USE_PIZ
#define TINYEXR_USE_PIZ (1)
#endif

#ifndef TINYEXR_USE_ZFP
#define TINYEXR_USE_ZFP (0)  // TinyEXR extension.
// http://computation.llnl.gov/projects/floating-point-compression
#endif

#define TINYEXR_SUCCESS (0)
#define TINYEXR_ERROR_INVALID_MAGIC_NUMBER (-1)
#define TINYEXR_ERROR_INVALID_EXR_VERSION (-2)
#define TINYEXR_ERROR_INVALID_ARGUMENT (-3)
#define TINYEXR_ERROR_INVALID_DATA (-4)
#define TINYEXR_ERROR_INVALID_FILE (-5)
#define TINYEXR_ERROR_INVALID_PARAMETER (-6)
#define TINYEXR_ERROR_CANT_OPEN_FILE (-7)
#define TINYEXR_ERROR_UNSUPPORTED_FORMAT (-8)
#define TINYEXR_ERROR_INVALID_HEADER (-9)
#define TINYEXR_ERROR_UNSUPPORTED_FEATURE (-10)
#define TINYEXR_ERROR_CANT_WRITE_FILE (-11)
#define TINYEXR_ERROR_SERIALZATION_FAILED (-12)

// @note { OpenEXR file format: http://www.openexr.com/openexrfilelayout.pdf }

// pixel type: possible values are: UINT = 0 HALF = 1 FLOAT = 2
#define TINYEXR_PIXELTYPE_UINT (0)
#define TINYEXR_PIXELTYPE_HALF (1)
#define TINYEXR_PIXELTYPE_FLOAT (2)

#define TINYEXR_MAX_HEADER_ATTRIBUTES (1024)
#define TINYEXR_MAX_CUSTOM_ATTRIBUTES (128)

#define TINYEXR_COMPRESSIONTYPE_NONE (0)
#define TINYEXR_COMPRESSIONTYPE_RLE (1)
#define TINYEXR_COMPRESSIONTYPE_ZIPS (2)
#define TINYEXR_COMPRESSIONTYPE_ZIP (3)
#define TINYEXR_COMPRESSIONTYPE_PIZ (4)
#define TINYEXR_COMPRESSIONTYPE_ZFP (128)  // TinyEXR extension

#define TINYEXR_ZFP_COMPRESSIONTYPE_RATE (0)
#define TINYEXR_ZFP_COMPRESSIONTYPE_PRECISION (1)
#define TINYEXR_ZFP_COMPRESSIONTYPE_ACCURACY (2)

#define TINYEXR_TILE_ONE_LEVEL (0)
#define TINYEXR_TILE_MIPMAP_LEVELS (1)
#define TINYEXR_TILE_RIPMAP_LEVELS (2)

#define TINYEXR_TILE_ROUND_DOWN (0)
#define TINYEXR_TILE_ROUND_UP (1)

typedef struct _EXRVersion {
  int version;    // this must be 2
  int tiled;      // tile format image
  int long_name;  // long name attribute
  int non_image;  // deep image(EXR 2.0)
  int multipart;  // multi-part(EXR 2.0)
} EXRVersion;

typedef struct _EXRAttribute {
  char name[256];  // name and type are up to 255 chars long.
  char type[256];
  unsigned char *value;  // uint8_t*
  int size;
  int pad0;
} EXRAttribute;

typedef struct _EXRChannelInfo {
  char name[256];  // less than 255 bytes long
  int pixel_type;
  int x_sampling;
  int y_sampling;
  unsigned char p_linear;
  unsigned char pad[3];
} EXRChannelInfo;

typedef struct _EXRTile {
  int offset_x;
  int offset_y;
  int level_x;
  int level_y;

  int width;   // actual width in a tile.
  int height;  // actual height int a tile.

  unsigned char **images;  // image[channels][pixels]
} EXRTile;

typedef struct _EXRHeader {
  float pixel_aspect_ratio;
  int line_order;
  int data_window[4];
  int display_window[4];
  float screen_window_center[2];
  float screen_window_width;

  int chunk_count;

  // Properties for tiled format(`tiledesc`).
  int tiled;
  int tile_size_x;
  int tile_size_y;
  int tile_level_mode;
  int tile_rounding_mode;

  int long_name;
  int non_image;
  int multipart;
  unsigned int header_len;

  // Custom attributes(exludes required attributes(e.g. `channels`,
  // `compression`, etc)
  int num_custom_attributes;
  EXRAttribute *custom_attributes;  // array of EXRAttribute. size =
                                    // `num_custom_attributes`.

  EXRChannelInfo *channels;  // [num_channels]

  int *pixel_types;  // Loaded pixel type(TINYEXR_PIXELTYPE_*) of `images` for
  // each channel. This is overwritten with `requested_pixel_types` when
  // loading.
  int num_channels;

  int compression_type;        // compression type(TINYEXR_COMPRESSIONTYPE_*)
  int *requested_pixel_types;  // Filled initially by
                               // ParseEXRHeaderFrom(Meomory|File), then users
                               // can edit it(only valid for HALF pixel type
                               // channel)

} EXRHeader;

typedef struct _EXRMultiPartHeader {
  int num_headers;
  EXRHeader *headers;

} EXRMultiPartHeader;

typedef struct _EXRImage {
  EXRTile *tiles;  // Tiled pixel data. The application must reconstruct image
                   // from tiles manually. NULL if scanline format.
  unsigned char **images;  // image[channels][pixels]. NULL if tiled format.

  int width;
  int height;
  int num_channels;

  // Properties for tile format.
  int num_tiles;

} EXRImage;

typedef struct _EXRMultiPartImage {
  int num_images;
  EXRImage *images;

} EXRMultiPartImage;

typedef struct _DeepImage {
  const char **channel_names;
  float ***image;      // image[channels][scanlines][samples]
  int **offset_table;  // offset_table[scanline][offsets]
  int num_channels;
  int width;
  int height;
  int pad0;
} DeepImage;

// @deprecated { to be removed. }
// Loads single-frame OpenEXR image. Assume EXR image contains A(single channel
// alpha) or RGB(A) channels.
// Application must free image data as returned by `out_rgba`
// Result image format is: float x RGBA x width x hight
// Returns negative value and may set error string in `err` when there's an
// error
extern int LoadEXR(float **out_rgba, int *width, int *height,
                   const char *filename, const char **err);

// @deprecated { to be removed. }
// Simple wrapper API for ParseEXRHeaderFromFile.
// checking given file is a EXR file(by just look up header)
// @return TINYEXR_SUCCEES for EXR image, TINYEXR_ERROR_INVALID_HEADER for
// others
extern int IsEXR(const char *filename);

// @deprecated { to be removed. }
// Saves single-frame OpenEXR image. Assume EXR image contains RGB(A) channels.
// components must be 1(Grayscale), 3(RGB) or 4(RGBA).
// Input image format is: `float x width x height`, or `float x RGB(A) x width x
// hight`
// Save image as fp16(HALF) format when `save_as_fp16` is positive non-zero
// value.
// Save image as fp32(FLOAT) format when `save_as_fp16` is 0.
// Use ZIP compression by default.
// Returns negative value and may set error string in `err` when there's an
// error
extern int SaveEXR(const float *data, const int width, const int height,
                   const int components, const int save_as_fp16,
                   const char *filename, const char **err);

// Initialize EXRHeader struct
extern void InitEXRHeader(EXRHeader *exr_header);

// Initialize EXRImage struct
extern void InitEXRImage(EXRImage *exr_image);

// Free's internal data of EXRHeader struct
extern int FreeEXRHeader(EXRHeader *exr_header);

// Free's internal data of EXRImage struct
extern int FreeEXRImage(EXRImage *exr_image);

// Free's error message
extern void FreeEXRErrorMessage(const char *msg);

// Parse EXR version header of a file.
extern int ParseEXRVersionFromFile(EXRVersion *version, const char *filename);

// Parse EXR version header from memory-mapped EXR data.
extern int ParseEXRVersionFromMemory(EXRVersion *version,
                                     const unsigned char *memory, size_t size);

// Parse single-part OpenEXR header from a file and initialize `EXRHeader`.
// When there was an error message, Application must free `err` with
// FreeEXRErrorMessage()
extern int ParseEXRHeaderFromFile(EXRHeader *header, const EXRVersion *version,
                                  const char *filename, const char **err);

// Parse single-part OpenEXR header from a memory and initialize `EXRHeader`.
// When there was an error message, Application must free `err` with
// FreeEXRErrorMessage()
extern int ParseEXRHeaderFromMemory(EXRHeader *header,
                                    const EXRVersion *version,
                                    const unsigned char *memory, size_t size,
                                    const char **err);

// Parse multi-part OpenEXR headers from a file and initialize `EXRHeader*`
// array.
// When there was an error message, Application must free `err` with
// FreeEXRErrorMessage()
extern int ParseEXRMultipartHeaderFromFile(EXRHeader ***headers,
                                           int *num_headers,
                                           const EXRVersion *version,
                                           const char *filename,
                                           const char **err);

// Parse multi-part OpenEXR headers from a memory and initialize `EXRHeader*`
// array
// When there was an error message, Application must free `err` with
// FreeEXRErrorMessage()
extern int ParseEXRMultipartHeaderFromMemory(EXRHeader ***headers,
                                             int *num_headers,
                                             const EXRVersion *version,
                                             const unsigned char *memory,
                                             size_t size, const char **err);

// Loads single-part OpenEXR image from a file.
// Application must setup `ParseEXRHeaderFromFile` before calling this function.
// Application can free EXRImage using `FreeEXRImage`
// Returns negative value and may set error string in `err` when there's an
// error
// When there was an error message, Application must free `err` with
// FreeEXRErrorMessage()
extern int LoadEXRImageFromFile(EXRImage *image, const EXRHeader *header,
                                const char *filename, const char **err);

// Loads single-part OpenEXR image from a memory.
// Application must setup `EXRHeader` with
// `ParseEXRHeaderFromMemory` before calling this function.
// Application can free EXRImage using `FreeEXRImage`
// Returns negative value and may set error string in `err` when there's an
// error
// When there was an error message, Application must free `err` with
// FreeEXRErrorMessage()
extern int LoadEXRImageFromMemory(EXRImage *image, const EXRHeader *header,
                                  const unsigned char *memory,
                                  const size_t size, const char **err);

// Loads multi-part OpenEXR image from a file.
// Application must setup `ParseEXRMultipartHeaderFromFile` before calling this
// function.
// Application can free EXRImage using `FreeEXRImage`
// Returns negative value and may set error string in `err` when there's an
// error
// When there was an error message, Application must free `err` with
// FreeEXRErrorMessage()
extern int LoadEXRMultipartImageFromFile(EXRImage *images,
                                         const EXRHeader **headers,
                                         unsigned int num_parts,
                                         const char *filename,
                                         const char **err);

// Loads multi-part OpenEXR image from a memory.
// Application must setup `EXRHeader*` array with
// `ParseEXRMultipartHeaderFromMemory` before calling this function.
// Application can free EXRImage using `FreeEXRImage`
// Returns negative value and may set error string in `err` when there's an
// error
// When there was an error message, Application must free `err` with
// FreeEXRErrorMessage()
extern int LoadEXRMultipartImageFromMemory(EXRImage *images,
                                           const EXRHeader **headers,
                                           unsigned int num_parts,
                                           const unsigned char *memory,
                                           const size_t size, const char **err);

// Saves multi-channel, single-frame OpenEXR image to a file.
// Returns negative value and may set error string in `err` when there's an
// error
// When there was an error message, Application must free `err` with
// FreeEXRErrorMessage()
extern int SaveEXRImageToFile(const EXRImage *image,
                              const EXRHeader *exr_header, const char *filename,
                              const char **err);

// Saves multi-channel, single-frame OpenEXR image to a memory.
// Image is compressed using EXRImage.compression value.
// Return the number of bytes if success.
// Return zero and will set error string in `err` when there's an
// error.
// When there was an error message, Application must free `err` with
// FreeEXRErrorMessage()
extern size_t SaveEXRImageToMemory(const EXRImage *image,
                                   const EXRHeader *exr_header,
                                   unsigned char **memory, const char **err);

// Loads single-frame OpenEXR deep image.
// Application must free memory of variables in DeepImage(image, offset_table)
// Returns negative value and may set error string in `err` when there's an
// error
// When there was an error message, Application must free `err` with
// FreeEXRErrorMessage()
extern int LoadDeepEXR(DeepImage *out_image, const char *filename,
                       const char **err);

// NOT YET IMPLEMENTED:
// Saves single-frame OpenEXR deep image.
// Returns negative value and may set error string in `err` when there's an
// error
// extern int SaveDeepEXR(const DeepImage *in_image, const char *filename,
//                       const char **err);

// NOT YET IMPLEMENTED:
// Loads multi-part OpenEXR deep image.
// Application must free memory of variables in DeepImage(image, offset_table)
// extern int LoadMultiPartDeepEXR(DeepImage **out_image, int num_parts, const
// char *filename,
//                       const char **err);

// For emscripten.
// Loads single-frame OpenEXR image from memory. Assume EXR image contains
// RGB(A) channels.
// Returns negative value and may set error string in `err` when there's an
// error
// When there was an error message, Application must free `err` with
// FreeEXRErrorMessage()
extern int LoadEXRFromMemory(float **out_rgba, int *width, int *height,
                             const unsigned char *memory, size_t size,
                             const char **err);

#ifdef __cplusplus
}
#endif

#endif  // TINYEXR_H_

#ifdef TINYEXR_IMPLEMENTATION
#ifndef TINYEXR_IMPLEMENTATION_DEIFNED
#define TINYEXR_IMPLEMENTATION_DEIFNED

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

//#include <iostream> // debug

#include <limits>
#include <string>
#include <vector>

#if __cplusplus > 199711L
// C++11
#include <cstdint>
#endif  // __cplusplus > 199711L

#ifdef _OPENMP
#include <omp.h>
#endif

#if TINYEXR_USE_MINIZ
#else
//  Issue #46. Please include your own zlib-compatible API header before
//  including `tinyexr.h`
//#include "zlib.h"
#endif

#if TINYEXR_USE_ZFP
#include "zfp.h"
#endif

namespace tinyexr {

#if __cplusplus > 199711L
// C++11
typedef uint64_t tinyexr_uint64;
typedef int64_t tinyexr_int64;
#else
// Although `long long` is not a standard type pre C++11, assume it is defined
// as a compiler's extension.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-long-long"
#endif
typedef unsigned long long tinyexr_uint64;
typedef long long tinyexr_int64;
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif

#if TINYEXR_USE_MINIZ

namespace miniz {

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-long-long"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wc++11-extensions"
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wundef"

#if __has_warning("-Wcomma")
#pragma clang diagnostic ignored "-Wcomma"
#endif

#if __has_warning("-Wmacro-redefined")
#pragma clang diagnostic ignored "-Wmacro-redefined"
#endif

#if __has_warning("-Wcast-qual")
#pragma clang diagnostic ignored "-Wcast-qual"
#endif

#if __has_warning("-Wzero-as-null-pointer-constant")
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif

#if __has_warning("-Wtautological-constant-compare")
#pragma clang diagnostic ignored "-Wtautological-constant-compare"
#endif

#if __has_warning("-Wextra-semi-stmt")
#pragma clang diagnostic ignored "-Wextra-semi-stmt"
#endif

#endif

/* miniz.c v1.15 - public domain deflate/inflate, zlib-subset, ZIP
   reading/writing/appending, PNG writing
   See "unlicense" statement at the end of this file.
   Rich Geldreich <richgel99@gmail.com>, last updated Oct. 13, 2013
   Implements RFC 1950: http://www.ietf.org/rfc/rfc1950.txt and RFC 1951:
   http://www.ietf.org/rfc/rfc1951.txt

   Most API's defined in miniz.c are optional. For example, to disable the
   archive related functions just define
   MINIZ_NO_ARCHIVE_APIS, or to get rid of all stdio usage define MINIZ_NO_STDIO
   (see the list below for more macros).

   * Change History
     10/13/13 v1.15 r4 - Interim bugfix release while I work on the next major
   release with Zip64 support (almost there!):
       - Critical fix for the MZ_ZIP_FLAG_DO_NOT_SORT_CENTRAL_DIRECTORY bug
   (thanks kahmyong.moon@hp.com) which could cause locate files to not find
   files. This bug
        would only have occured in earlier versions if you explicitly used this
   flag, OR if you used mz_zip_extract_archive_file_to_heap() or
   mz_zip_add_mem_to_archive_file_in_place()
        (which used this flag). If you can't switch to v1.15 but want to fix
   this bug, just remove the uses of this flag from both helper funcs (and of
   course don't use the flag).
       - Bugfix in mz_zip_reader_extract_to_mem_no_alloc() from kymoon when
   pUser_read_buf is not NULL and compressed size is > uncompressed size
       - Fixing mz_zip_reader_extract_*() funcs so they don't try to extract
   compressed data from directory entries, to account for weird zipfiles which
   contain zero-size compressed data on dir entries.
         Hopefully this fix won't cause any issues on weird zip archives,
   because it assumes the low 16-bits of zip external attributes are DOS
   attributes (which I believe they always are in practice).
       - Fixing mz_zip_reader_is_file_a_directory() so it doesn't check the
   internal attributes, just the filename and external attributes
       - mz_zip_reader_init_file() - missing MZ_FCLOSE() call if the seek failed
       - Added cmake support for Linux builds which builds all the examples,
   tested with clang v3.3 and gcc v4.6.
       - Clang fix for tdefl_write_image_to_png_file_in_memory() from toffaletti
       - Merged MZ_FORCEINLINE fix from hdeanclark
       - Fix <time.h> include before config #ifdef, thanks emil.brink
       - Added tdefl_write_image_to_png_file_in_memory_ex(): supports Y flipping
   (super useful for OpenGL apps), and explicit control over the compression
   level (so you can
        set it to 1 for real-time compression).
       - Merged in some compiler fixes from paulharris's github repro.
       - Retested this build under Windows (VS 2010, including static analysis),
   tcc  0.9.26, gcc v4.6 and clang v3.3.
       - Added example6.c, which dumps an image of the mandelbrot set to a PNG
   file.
       - Modified example2 to help test the
   MZ_ZIP_FLAG_DO_NOT_SORT_CENTRAL_DIRECTORY flag more.
       - In r3: Bugfix to mz_zip_writer_add_file() found during merge: Fix
   possible src file fclose() leak if alignment bytes+local header file write
   faiiled
                 - In r4: Minor bugfix to mz_zip_writer_add_from_zip_reader():
   Was pushing the wrong central dir header offset, appears harmless in this
   release, but it became a problem in the zip64 branch
     5/20/12 v1.14 - MinGW32/64 GCC 4.6.1 compiler fixes: added MZ_FORCEINLINE,
   #include <time.h> (thanks fermtect).
     5/19/12 v1.13 - From jason@cornsyrup.org and kelwert@mtu.edu - Fix
   mz_crc32() so it doesn't compute the wrong CRC-32's when mz_ulong is 64-bit.
       - Temporarily/locally slammed in "typedef unsigned long mz_ulong" and
   re-ran a randomized regression test on ~500k files.
       - Eliminated a bunch of warnings when compiling with GCC 32-bit/64.
       - Ran all examples, miniz.c, and tinfl.c through MSVC 2008's /analyze
   (static analysis) option and fixed all warnings (except for the silly
        "Use of the comma-operator in a tested expression.." analysis warning,
   which I purposely use to work around a MSVC compiler warning).
       - Created 32-bit and 64-bit Codeblocks projects/workspace. Built and
   tested Linux executables. The codeblocks workspace is compatible with
   Linux+Win32/x64.
       - Added miniz_tester solution/project, which is a useful little app
   derived from LZHAM's tester app that I use as part of the regression test.
       - Ran miniz.c and tinfl.c through another series of regression testing on
   ~500,000 files and archives.
       - Modified example5.c so it purposely disables a bunch of high-level
   functionality (MINIZ_NO_STDIO, etc.). (Thanks to corysama for the
   MINIZ_NO_STDIO bug report.)
       - Fix ftell() usage in examples so they exit with an error on files which
   are too large (a limitation of the examples, not miniz itself).
     4/12/12 v1.12 - More comments, added low-level example5.c, fixed a couple
   minor level_and_flags issues in the archive API's.
      level_and_flags can now be set to MZ_DEFAULT_COMPRESSION. Thanks to Bruce
   Dawson <bruced@valvesoftware.com> for the feedback/bug report.
     5/28/11 v1.11 - Added statement from unlicense.org
     5/27/11 v1.10 - Substantial compressor optimizations:
      - Level 1 is now ~4x faster than before. The L1 compressor's throughput
   now varies between 70-110MB/sec. on a
      - Core i7 (actual throughput varies depending on the type of data, and x64
   vs. x86).
      - Improved baseline L2-L9 compression perf. Also, greatly improved
   compression perf. issues on some file types.
      - Refactored the compression code for better readability and
   maintainability.
      - Added level 10 compression level (L10 has slightly better ratio than
   level 9, but could have a potentially large
       drop in throughput on some files).
     5/15/11 v1.09 - Initial stable release.

   * Low-level Deflate/Inflate implementation notes:

     Compression: Use the "tdefl" API's. The compressor supports raw, static,
   and dynamic blocks, lazy or
     greedy parsing, match length filtering, RLE-only, and Huffman-only streams.
   It performs and compresses
     approximately as well as zlib.

     Decompression: Use the "tinfl" API's. The entire decompressor is
   implemented as a single function
     coroutine: see tinfl_decompress(). It supports decompression into a 32KB
   (or larger power of 2) wrapping buffer, or into a memory
     block large enough to hold the entire file.

     The low-level tdefl/tinfl API's do not make any use of dynamic memory
   allocation.

   * zlib-style API notes:

     miniz.c implements a fairly large subset of zlib. There's enough
   functionality present for it to be a drop-in
     zlib replacement in many apps:
        The z_stream struct, optional memory allocation callbacks
        deflateInit/deflateInit2/deflate/deflateReset/deflateEnd/deflateBound
        inflateInit/inflateInit2/inflate/inflateEnd
        compress, compress2, compressBound, uncompress
        CRC-32, Adler-32 - Using modern, minimal code size, CPU cache friendly
   routines.
        Supports raw deflate streams or standard zlib streams with adler-32
   checking.

     Limitations:
      The callback API's are not implemented yet. No support for gzip headers or
   zlib static dictionaries.
      I've tried to closely emulate zlib's various flavors of stream flushing
   and return status codes, but
      there are no guarantees that miniz.c pulls this off perfectly.

   * PNG writing: See the tdefl_write_image_to_png_file_in_memory() function,
   originally written by
     Alex Evans. Supports 1-4 bytes/pixel images.

   * ZIP archive API notes:

     The ZIP archive API's where designed with simplicity and efficiency in
   mind, with just enough abstraction to
     get the job done with minimal fuss. There are simple API's to retrieve file
   information, read files from
     existing archives, create new archives, append new files to existing
   archives, or clone archive data from
     one archive to another. It supports archives located in memory or the heap,
   on disk (using stdio.h),
     or you can specify custom file read/write callbacks.

     - Archive reading: Just call this function to read a single file from a
   disk archive:

      void *mz_zip_extract_archive_file_to_heap(const char *pZip_filename, const
   char *pArchive_name,
        size_t *pSize, mz_uint zip_flags);

     For more complex cases, use the "mz_zip_reader" functions. Upon opening an
   archive, the entire central
     directory is located and read as-is into memory, and subsequent file access
   only occurs when reading individual files.

     - Archives file scanning: The simple way is to use this function to scan a
   loaded archive for a specific file:

     int mz_zip_reader_locate_file(mz_zip_archive *pZip, const char *pName,
   const char *pComment, mz_uint flags);

     The locate operation can optionally check file comments too, which (as one
   example) can be used to identify
     multiple versions of the same file in an archive. This function uses a
   simple linear search through the central
     directory, so it's not very fast.

     Alternately, you can iterate through all the files in an archive (using
   mz_zip_reader_get_num_files()) and
     retrieve detailed info on each file by calling mz_zip_reader_file_stat().

     - Archive creation: Use the "mz_zip_writer" functions. The ZIP writer
   immediately writes compressed file data
     to disk and builds an exact image of the central directory in memory. The
   central directory image is written
     all at once at the end of the archive file when the archive is finalized.

     The archive writer can optionally align each file's local header and file
   data to any power of 2 alignment,
     which can be useful when the archive will be read from optical media. Also,
   the writer supports placing
     arbitrary data blobs at the very beginning of ZIP archives. Archives
   written using either feature are still
     readable by any ZIP tool.

     - Archive appending: The simple way to add a single file to an archive is
   to call this function:

      mz_bool mz_zip_add_mem_to_archive_file_in_place(const char *pZip_filename,
   const char *pArchive_name,
        const void *pBuf, size_t buf_size, const void *pComment, mz_uint16
   comment_size, mz_uint level_and_flags);

     The archive will be created if it doesn't already exist, otherwise it'll be
   appended to.
     Note the appending is done in-place and is not an atomic operation, so if
   something goes wrong
     during the operation it's possible the archive could be left without a
   central directory (although the local
     file headers and file data will be fine, so the archive will be
   recoverable).

     For more complex archive modification scenarios:
     1. The safest way is to use a mz_zip_reader to read the existing archive,
   cloning only those bits you want to
     preserve into a new archive using using the
   mz_zip_writer_add_from_zip_reader() function (which compiles the
     compressed file data as-is). When you're done, delete the old archive and
   rename the newly written archive, and
     you're done. This is safe but requires a bunch of temporary disk space or
   heap memory.

     2. Or, you can convert an mz_zip_reader in-place to an mz_zip_writer using
   mz_zip_writer_init_from_reader(),
     append new files as needed, then finalize the archive which will write an
   updated central directory to the
     original archive. (This is basically what
   mz_zip_add_mem_to_archive_file_in_place() does.) There's a
     possibility that the archive's central directory could be lost with this
   method if anything goes wrong, though.

     - ZIP archive support limitations:
     No zip64 or spanning support. Extraction functions can only handle
   unencrypted, stored or deflated files.
     Requires streams capable of seeking.

   * This is a header file library, like stb_image.c. To get only a header file,
   either cut and paste the
     below header, or create miniz.h, #define MINIZ_HEADER_FILE_ONLY, and then
   include miniz.c from it.

   * Important: For best perf. be sure to customize the below macros for your
   target platform:
     #define MINIZ_USE_UNALIGNED_LOADS_AND_STORES 1
     #define MINIZ_LITTLE_ENDIAN 1
     #define MINIZ_HAS_64BIT_REGISTERS 1

   * On platforms using glibc, Be sure to "#define _LARGEFILE64_SOURCE 1" before
   including miniz.c to ensure miniz
     uses the 64-bit variants: fopen64(), stat64(), etc. Otherwise you won't be
   able to process large files
     (i.e. 32-bit stat() fails for me on files > 0x7FFFFFFF bytes).
*/

#ifndef MINIZ_HEADER_INCLUDED
#define MINIZ_HEADER_INCLUDED

//#include <stdlib.h>

// Defines to completely disable specific portions of miniz.c:
// If all macros here are defined the only functionality remaining will be
// CRC-32, adler-32, tinfl, and tdefl.

// Define MINIZ_NO_STDIO to disable all usage and any functions which rely on
// stdio for file I/O.
//#define MINIZ_NO_STDIO

// If MINIZ_NO_TIME is specified then the ZIP archive functions will not be able
// to get the current time, or
// get/set file times, and the C run-time funcs that get/set times won't be
// called.
// The current downside is the times written to your archives will be from 1979.
#define MINIZ_NO_TIME

// Define MINIZ_NO_ARCHIVE_APIS to disable all ZIP archive API's.
#define MINIZ_NO_ARCHIVE_APIS

// Define MINIZ_NO_ARCHIVE_APIS to disable all writing related ZIP archive
// API's.
//#define MINIZ_NO_ARCHIVE_WRITING_APIS

// Define MINIZ_NO_ZLIB_APIS to remove all ZLIB-style compression/decompression
// API's.
//#define MINIZ_NO_ZLIB_APIS

// Define MINIZ_NO_ZLIB_COMPATIBLE_NAME to disable zlib names, to prevent
// conflicts against stock zlib.
//#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES

// Define MINIZ_NO_MALLOC to disable all calls to malloc, free, and realloc.
// Note if MINIZ_NO_MALLOC is defined then the user must always provide custom
// user alloc/free/realloc
// callbacks to the zlib and archive API's, and a few stand-alone helper API's
// which don't provide custom user
// functions (such as tdefl_compress_mem_to_heap() and
// tinfl_decompress_mem_to_heap()) won't work.
//#define MINIZ_NO_MALLOC

#if defined(__TINYC__) && (defined(__linux) || defined(__linux__))
// TODO: Work around "error: include file 'sys\utime.h' when compiling with tcc
// on Linux
#define MINIZ_NO_TIME
#endif

#if !defined(MINIZ_NO_TIME) && !defined(MINIZ_NO_ARCHIVE_APIS)
//#include <time.h>
#endif

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || \
    defined(__i386) || defined(__i486__) || defined(__i486) ||  \
    defined(i386) || defined(__ia64__) || defined(__x86_64__)
// MINIZ_X86_OR_X64_CPU is only used to help set the below macros.
#define MINIZ_X86_OR_X64_CPU 1
#endif

#if defined(__sparcv9)
// Big endian
#else
#if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) || MINIZ_X86_OR_X64_CPU
// Set MINIZ_LITTLE_ENDIAN to 1 if the processor is little endian.
#define MINIZ_LITTLE_ENDIAN 1
#endif
#endif

#if MINIZ_X86_OR_X64_CPU
// Set MINIZ_USE_UNALIGNED_LOADS_AND_STORES to 1 on CPU's that permit efficient
// integer loads and stores from unaligned addresses.
//#define MINIZ_USE_UNALIGNED_LOADS_AND_STORES 1
#define MINIZ_USE_UNALIGNED_LOADS_AND_STORES \
  0  // disable to suppress compiler warnings
#endif

#if defined(_M_X64) || defined(_WIN64) || defined(__MINGW64__) || \
    defined(_LP64) || defined(__LP64__) || defined(__ia64__) ||   \
    defined(__x86_64__)
// Set MINIZ_HAS_64BIT_REGISTERS to 1 if operations on 64-bit integers are
// reasonably fast (and don't involve compiler generated calls to helper
// functions).
#define MINIZ_HAS_64BIT_REGISTERS 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ------------------- zlib-style API Definitions.

// For more compatibility with zlib, miniz.c uses unsigned long for some
// parameters/struct members. Beware: mz_ulong can be either 32 or 64-bits!
typedef unsigned long mz_ulong;

// mz_free() internally uses the MZ_FREE() macro (which by default calls free()
// unless you've modified the MZ_MALLOC macro) to release a block allocated from
// the heap.
void mz_free(void *p);

#define MZ_ADLER32_INIT (1)
// mz_adler32() returns the initial adler-32 value to use when called with
// ptr==NULL.
mz_ulong mz_adler32(mz_ulong adler, const unsigned char *ptr, size_t buf_len);

#define MZ_CRC32_INIT (0)
// mz_crc32() returns the initial CRC-32 value to use when called with
// ptr==NULL.
mz_ulong mz_crc32(mz_ulong crc, const unsigned char *ptr, size_t buf_len);

// Compression strategies.
enum {
  MZ_DEFAULT_STRATEGY = 0,
  MZ_FILTERED = 1,
  MZ_HUFFMAN_ONLY = 2,
  MZ_RLE = 3,
  MZ_FIXED = 4
};

// Method
#define MZ_DEFLATED 8

#ifndef MINIZ_NO_ZLIB_APIS

// Heap allocation callbacks.
// Note that mz_alloc_func parameter types purpsosely differ from zlib's:
// items/size is size_t, not unsigned long.
typedef void *(*mz_alloc_func)(void *opaque, size_t items, size_t size);
typedef void (*mz_free_func)(void *opaque, void *address);
typedef void *(*mz_realloc_func)(void *opaque, void *address, size_t items,
                                 size_t size);

#define MZ_VERSION "9.1.15"
#define MZ_VERNUM 0x91F0
#define MZ_VER_MAJOR 9
#define MZ_VER_MINOR 1
#define MZ_VER_REVISION 15
#define MZ_VER_SUBREVISION 0

// Flush values. For typical usage you only need MZ_NO_FLUSH and MZ_FINISH. The
// other values are for advanced use (refer to the zlib docs).
enum {
  MZ_NO_FLUSH = 0,
  MZ_PARTIAL_FLUSH = 1,
  MZ_SYNC_FLUSH = 2,
  MZ_FULL_FLUSH = 3,
  MZ_FINISH = 4,
  MZ_BLOCK = 5
};

// Return status codes. MZ_PARAM_ERROR is non-standard.
enum {
  MZ_OK = 0,
  MZ_STREAM_END = 1,
  MZ_NEED_DICT = 2,
  MZ_ERRNO = -1,
  MZ_STREAM_ERROR = -2,
  MZ_DATA_ERROR = -3,
  MZ_MEM_ERROR = -4,
  MZ_BUF_ERROR = -5,
  MZ_VERSION_ERROR = -6,
  MZ_PARAM_ERROR = -10000
};

// Compression levels: 0-9 are the standard zlib-style levels, 10 is best
// possible compression (not zlib compatible, and may be very slow),
// MZ_DEFAULT_COMPRESSION=MZ_DEFAULT_LEVEL.
enum {
  MZ_NO_COMPRESSION = 0,
  MZ_BEST_SPEED = 1,
  MZ_BEST_COMPRESSION = 9,
  MZ_UBER_COMPRESSION = 10,
  MZ_DEFAULT_LEVEL = 6,
  MZ_DEFAULT_COMPRESSION = -1
};

// Window bits
#define MZ_DEFAULT_WINDOW_BITS 15

struct mz_internal_state;

// Compression/decompression stream struct.
typedef struct mz_stream_s {
  const unsigned char *next_in;  // pointer to next byte to read
  unsigned int avail_in;         // number of bytes available at next_in
  mz_ulong total_in;             // total number of bytes consumed so far

  unsigned char *next_out;  // pointer to next byte to write
  unsigned int avail_out;   // number of bytes that can be written to next_out
  mz_ulong total_out;       // total number of bytes produced so far

  char *msg;                        // error msg (unused)
  struct mz_internal_state *state;  // internal state, allocated by zalloc/zfree

  mz_alloc_func
      zalloc;          // optional heap allocation function (defaults to malloc)
  mz_free_func zfree;  // optional heap free function (defaults to free)
  void *opaque;        // heap alloc function user pointer

  int data_type;      // data_type (unused)
  mz_ulong adler;     // adler32 of the source or uncompressed data
  mz_ulong reserved;  // not used
} mz_stream;

typedef mz_stream *mz_streamp;

// Returns the version string of miniz.c.
const char *mz_version(void);

// mz_deflateInit() initializes a compressor with default options:
// Parameters:
//  pStream must point to an initialized mz_stream struct.
//  level must be between [MZ_NO_COMPRESSION, MZ_BEST_COMPRESSION].
//  level 1 enables a specially optimized compression function that's been
//  optimized purely for performance, not ratio.
//  (This special func. is currently only enabled when
//  MINIZ_USE_UNALIGNED_LOADS_AND_STORES and MINIZ_LITTLE_ENDIAN are defined.)
// Return values:
//  MZ_OK on success.
//  MZ_STREAM_ERROR if the stream is bogus.
//  MZ_PARAM_ERROR if the input parameters are bogus.
//  MZ_MEM_ERROR on out of memory.
int mz_deflateInit(mz_streamp pStream, int level);

// mz_deflateInit2() is like mz_deflate(), except with more control:
// Additional parameters:
//   method must be MZ_DEFLATED
//   window_bits must be MZ_DEFAULT_WINDOW_BITS (to wrap the deflate stream with
//   zlib header/adler-32 footer) or -MZ_DEFAULT_WINDOW_BITS (raw deflate/no
//   header or footer)
//   mem_level must be between [1, 9] (it's checked but ignored by miniz.c)
int mz_deflateInit2(mz_streamp pStream, int level, int method, int window_bits,
                    int mem_level, int strategy);

// Quickly resets a compressor without having to reallocate anything. Same as
// calling mz_deflateEnd() followed by mz_deflateInit()/mz_deflateInit2().
int mz_deflateReset(mz_streamp pStream);

// mz_deflate() compresses the input to output, consuming as much of the input
// and producing as much output as possible.
// Parameters:
//   pStream is the stream to read from and write to. You must initialize/update
//   the next_in, avail_in, next_out, and avail_out members.
//   flush may be MZ_NO_FLUSH, MZ_PARTIAL_FLUSH/MZ_SYNC_FLUSH, MZ_FULL_FLUSH, or
//   MZ_FINISH.
// Return values:
//   MZ_OK on success (when flushing, or if more input is needed but not
//   available, and/or there's more output to be written but the output buffer
//   is full).
//   MZ_STREAM_END if all input has been consumed and all output bytes have been
//   written. Don't call mz_deflate() on the stream anymore.
//   MZ_STREAM_ERROR if the stream is bogus.
//   MZ_PARAM_ERROR if one of the parameters is invalid.
//   MZ_BUF_ERROR if no forward progress is possible because the input and/or
//   output buffers are empty. (Fill up the input buffer or free up some output
//   space and try again.)
int mz_deflate(mz_streamp pStream, int flush);

// mz_deflateEnd() deinitializes a compressor:
// Return values:
//  MZ_OK on success.
//  MZ_STREAM_ERROR if the stream is bogus.
int mz_deflateEnd(mz_streamp pStream);

// mz_deflateBound() returns a (very) conservative upper bound on the amount of
// data that could be generated by deflate(), assuming flush is set to only
// MZ_NO_FLUSH or MZ_FINISH.
mz_ulong mz_deflateBound(mz_streamp pStream, mz_ulong source_len);

// Single-call compression functions mz_compress() and mz_compress2():
// Returns MZ_OK on success, or one of the error codes from mz_deflate() on
// failure.
int mz_compress(unsigned char *pDest, mz_ulong *pDest_len,
                const unsigned char *pSource, mz_ulong source_len);
int mz_compress2(unsigned char *pDest, mz_ulong *pDest_len,
                 const unsigned char *pSource, mz_ulong source_len, int level);

// mz_compressBound() returns a (very) conservative upper bound on the amount of
// data that could be generated by calling mz_compress().
mz_ulong mz_compressBound(mz_ulong source_len);

// Initializes a decompressor.
int mz_inflateInit(mz_streamp pStream);

// mz_inflateInit2() is like mz_inflateInit() with an additional option that
// controls the window size and whether or not the stream has been wrapped with
// a zlib header/footer:
// window_bits must be MZ_DEFAULT_WINDOW_BITS (to parse zlib header/footer) or
// -MZ_DEFAULT_WINDOW_BITS (raw deflate).
int mz_inflateInit2(mz_streamp pStream, int window_bits);

// Decompresses the input stream to the output, consuming only as much of the
// input as needed, and writing as much to the output as possible.
// Parameters:
//   pStream is the stream to read from and write to. You must initialize/update
//   the next_in, avail_in, next_out, and avail_out members.
//   flush may be MZ_NO_FLUSH, MZ_SYNC_FLUSH, or MZ_FINISH.
//   On the first call, if flush is MZ_FINISH it's assumed the input and output
//   buffers are both sized large enough to decompress the entire stream in a
//   single call (this is slightly faster).
//   MZ_FINISH implies that there are no more source bytes available beside
//   what's already in the input buffer, and that the output buffer is large
//   enough to hold the rest of the decompressed data.
// Return values:
//   MZ_OK on success. Either more input is needed but not available, and/or
//   there's more output to be written but the output buffer is full.
//   MZ_STREAM_END if all needed input has been consumed and all output bytes
//   have been written. For zlib streams, the adler-32 of the decompressed data
//   has also been verified.
//   MZ_STREAM_ERROR if the stream is bogus.
//   MZ_DATA_ERROR if the deflate stream is invalid.
//   MZ_PARAM_ERROR if one of the parameters is invalid.
//   MZ_BUF_ERROR if no forward progress is possible because the input buffer is
//   empty but the inflater needs more input to continue, or if the output
//   buffer is not large enough. Call mz_inflate() again
//   with more input data, or with more room in the output buffer (except when
//   using single call decompression, described above).
int mz_inflate(mz_streamp pStream, int flush);

// Deinitializes a decompressor.
int mz_inflateEnd(mz_streamp pStream);

// Single-call decompression.
// Returns MZ_OK on success, or one of the error codes from mz_inflate() on
// failure.
int mz_uncompress(unsigned char *pDest, mz_ulong *pDest_len,
                  const unsigned char *pSource, mz_ulong source_len);

// Returns a string description of the specified error code, or NULL if the
// error code is invalid.
const char *mz_error(int err);

// Redefine zlib-compatible names to miniz equivalents, so miniz.c can be used
// as a drop-in replacement for the subset of zlib that miniz.c supports.
// Define MINIZ_NO_ZLIB_COMPATIBLE_NAMES to disable zlib-compatibility if you
// use zlib in the same project.
#ifndef MINIZ_NO_ZLIB_COMPATIBLE_NAMES
typedef unsigned char Byte;
typedef unsigned int uInt;
typedef mz_ulong uLong;
typedef Byte Bytef;
typedef uInt uIntf;
typedef char charf;
typedef int intf;
typedef void *voidpf;
typedef uLong uLongf;
typedef void *voidp;
typedef void *const voidpc;
#define Z_NULL 0
#define Z_NO_FLUSH MZ_NO_FLUSH
#define Z_PARTIAL_FLUSH MZ_PARTIAL_FLUSH
#define Z_SYNC_FLUSH MZ_SYNC_FLUSH
#define Z_FULL_FLUSH MZ_FULL_FLUSH
#define Z_FINISH MZ_FINISH
#define Z_BLOCK MZ_BLOCK
#define Z_OK MZ_OK
#define Z_STREAM_END MZ_STREAM_END
#define Z_NEED_DICT MZ_NEED_DICT
#define Z_ERRNO MZ_ERRNO
#define Z_STREAM_ERROR MZ_STREAM_ERROR
#define Z_DATA_ERROR MZ_DATA_ERROR
#define Z_MEM_ERROR MZ_MEM_ERROR
#define Z_BUF_ERROR MZ_BUF_ERROR
#define Z_VERSION_ERROR MZ_VERSION_ERROR
#define Z_PARAM_ERROR MZ_PARAM_ERROR
#define Z_NO_COMPRESSION MZ_NO_COMPRESSION
#define Z_BEST_SPEED MZ_BEST_SPEED
#define Z_BEST_COMPRESSION MZ_BEST_COMPRESSION
#define Z_DEFAULT_COMPRESSION MZ_DEFAULT_COMPRESSION
#define Z_DEFAULT_STRATEGY MZ_DEFAULT_STRATEGY
#define Z_FILTERED MZ_FILTERED
#define Z_HUFFMAN_ONLY MZ_HUFFMAN_ONLY
#define Z_RLE MZ_RLE
#define Z_FIXED MZ_FIXED
#define Z_DEFLATED MZ_DEFLATED
#define Z_DEFAULT_WINDOW_BITS MZ_DEFAULT_WINDOW_BITS
#define alloc_func mz_alloc_func
#define free_func mz_free_func
#define internal_state mz_internal_state
#define z_stream mz_stream
#define deflateInit mz_deflateInit
#define deflateInit2 mz_deflateInit2
#define deflateReset mz_deflateReset
#define deflate mz_deflate
#define deflateEnd mz_deflateEnd
#define deflateBound mz_deflateBound
#define compress mz_compress
#define compress2 mz_compress2
#define compressBound mz_compressBound
#define inflateInit mz_inflateInit
#define inflateInit2 mz_inflateInit2
#define inflate mz_inflate
#define inflateEnd mz_inflateEnd
#define uncompress mz_uncompress
#define crc32 mz_crc32
#define adler32 mz_adler32
#define MAX_WBITS 15
#define MAX_MEM_LEVEL 9
#define zError mz_error
#define ZLIB_VERSION MZ_VERSION
#define ZLIB_VERNUM MZ_VERNUM
#define ZLIB_VER_MAJOR MZ_VER_MAJOR
#define ZLIB_VER_MINOR MZ_VER_MINOR
#define ZLIB_VER_REVISION MZ_VER_REVISION
#define ZLIB_VER_SUBREVISION MZ_VER_SUBREVISION
#define zlibVersion mz_version
#define zlib_version mz_version()
#endif  // #ifndef MINIZ_NO_ZLIB_COMPATIBLE_NAMES

#endif  // MINIZ_NO_ZLIB_APIS

// ------------------- Types and macros

typedef unsigned char mz_uint8;
typedef signed short mz_int16;
typedef unsigned short mz_uint16;
typedef unsigned int mz_uint32;
typedef unsigned int mz_uint;
typedef long long mz_int64;
typedef unsigned long long mz_uint64;
typedef int mz_bool;

#define MZ_FALSE (0)
#define MZ_TRUE (1)

// An attempt to work around MSVC's spammy "warning C4127: conditional
// expression is constant" message.
#ifdef _MSC_VER
#define MZ_MACRO_END while (0, 0)
#else
#define MZ_MACRO_END while (0)
#endif

// ------------------- ZIP archive reading/writing

#ifndef MINIZ_NO_ARCHIVE_APIS

enum {
  MZ_ZIP_MAX_IO_BUF_SIZE = 64 * 1024,
  MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE = 260,
  MZ_ZIP_MAX_ARCHIVE_FILE_COMMENT_SIZE = 256
};

typedef struct {
  mz_uint32 m_file_index;
  mz_uint32 m_central_dir_ofs;
  mz_uint16 m_version_made_by;
  mz_uint16 m_version_needed;
  mz_uint16 m_bit_flag;
  mz_uint16 m_method;
#ifndef MINIZ_NO_TIME
  time_t m_time;
#endif
  mz_uint32 m_crc32;
  mz_uint64 m_comp_size;
  mz_uint64 m_uncomp_size;
  mz_uint16 m_internal_attr;
  mz_uint32 m_external_attr;
  mz_uint64 m_local_header_ofs;
  mz_uint32 m_comment_size;
  char m_filename[MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE];
  char m_comment[MZ_ZIP_MAX_ARCHIVE_FILE_COMMENT_SIZE];
} mz_zip_archive_file_stat;

typedef size_t (*mz_file_read_func)(void *pOpaque, mz_uint64 file_ofs,
                                    void *pBuf, size_t n);
typedef size_t (*mz_file_write_func)(void *pOpaque, mz_uint64 file_ofs,
                                     const void *pBuf, size_t n);

struct mz_zip_internal_state_tag;
typedef struct mz_zip_internal_state_tag mz_zip_internal_state;

typedef enum {
  MZ_ZIP_MODE_INVALID = 0,
  MZ_ZIP_MODE_READING = 1,
  MZ_ZIP_MODE_WRITING = 2,
  MZ_ZIP_MODE_WRITING_HAS_BEEN_FINALIZED = 3
} mz_zip_mode;

typedef struct mz_zip_archive_tag {
  mz_uint64 m_archive_size;
  mz_uint64 m_central_directory_file_ofs;
  mz_uint m_total_files;
  mz_zip_mode m_zip_mode;

  mz_uint m_file_offset_alignment;

  mz_alloc_func m_pAlloc;
  mz_free_func m_pFree;
  mz_realloc_func m_pRealloc;
  void *m_pAlloc_opaque;

  mz_file_read_func m_pRead;
  mz_file_write_func m_pWrite;
  void *m_pIO_opaque;

  mz_zip_internal_state *m_pState;

} mz_zip_archive;

typedef enum {
  MZ_ZIP_FLAG_CASE_SENSITIVE = 0x0100,
  MZ_ZIP_FLAG_IGNORE_PATH = 0x0200,
  MZ_ZIP_FLAG_COMPRESSED_DATA = 0x0400,
  MZ_ZIP_FLAG_DO_NOT_SORT_CENTRAL_DIRECTORY = 0x0800
} mz_zip_flags;

// ZIP archive reading

// Inits a ZIP archive reader.
// These functions read and validate the archive's central directory.
mz_bool mz_zip_reader_init(mz_zip_archive *pZip, mz_uint64 size,
                           mz_uint32 flags);
mz_bool mz_zip_reader_init_mem(mz_zip_archive *pZip, const void *pMem,
                               size_t size, mz_uint32 flags);

#ifndef MINIZ_NO_STDIO
mz_bool mz_zip_reader_init_file(mz_zip_archive *pZip, const char *pFilename,
                                mz_uint32 flags);
#endif

// Returns the total number of files in the archive.
mz_uint mz_zip_reader_get_num_files(mz_zip_archive *pZip);

// Returns detailed information about an archive file entry.
mz_bool mz_zip_reader_file_stat(mz_zip_archive *pZip, mz_uint file_index,
                                mz_zip_archive_file_stat *pStat);

// Determines if an archive file entry is a directory entry.
mz_bool mz_zip_reader_is_file_a_directory(mz_zip_archive *pZip,
                                          mz_uint file_index);
mz_bool mz_zip_reader_is_file_encrypted(mz_zip_archive *pZip,
                                        mz_uint file_index);

// Retrieves the filename of an archive file entry.
// Returns the number of bytes written to pFilename, or if filename_buf_size is
// 0 this function returns the number of bytes needed to fully store the
// filename.
mz_uint mz_zip_reader_get_filename(mz_zip_archive *pZip, mz_uint file_index,
                                   char *pFilename, mz_uint filename_buf_size);

// Attempts to locates a file in the archive's central directory.
// Valid flags: MZ_ZIP_FLAG_CASE_SENSITIVE, MZ_ZIP_FLAG_IGNORE_PATH
// Returns -1 if the file cannot be found.
int mz_zip_reader_locate_file(mz_zip_archive *pZip, const char *pName,
                              const char *pComment, mz_uint flags);

// Extracts a archive file to a memory buffer using no memory allocation.
mz_bool mz_zip_reader_extract_to_mem_no_alloc(mz_zip_archive *pZip,
                                              mz_uint file_index, void *pBuf,
                                              size_t buf_size, mz_uint flags,
                                              void *pUser_read_buf,
                                              size_t user_read_buf_size);
mz_bool mz_zip_reader_extract_file_to_mem_no_alloc(
    mz_zip_archive *pZip, const char *pFilename, void *pBuf, size_t buf_size,
    mz_uint flags, void *pUser_read_buf, size_t user_read_buf_size);

// Extracts a archive file to a memory buffer.
mz_bool mz_zip_reader_extract_to_mem(mz_zip_archive *pZip, mz_uint file_index,
                                     void *pBuf, size_t buf_size,
                                     mz_uint flags);
mz_bool mz_zip_reader_extract_file_to_mem(mz_zip_archive *pZip,
                                          const char *pFilename, void *pBuf,
                                          size_t buf_size, mz_uint flags);

// Extracts a archive file to a dynamically allocated heap buffer.
void *mz_zip_reader_extract_to_heap(mz_zip_archive *pZip, mz_uint file_index,
                                    size_t *pSize, mz_uint flags);
void *mz_zip_reader_extract_file_to_heap(mz_zip_archive *pZip,
                                         const char *pFilename, size_t *pSize,
                                         mz_uint flags);

// Extracts a archive file using a callback function to output the file's data.
mz_bool mz_zip_reader_extract_to_callback(mz_zip_archive *pZip,
                                          mz_uint file_index,
                                          mz_file_write_func pCallback,
                                          void *pOpaque, mz_uint flags);
mz_bool mz_zip_reader_extract_file_to_callback(mz_zip_archive *pZip,
                                               const char *pFilename,
                                               mz_file_write_func pCallback,
                                               void *pOpaque, mz_uint flags);

#ifndef MINIZ_NO_STDIO
// Extracts a archive file to a disk file and sets its last accessed and
// modified times.
// This function only extracts files, not archive directory records.
mz_bool mz_zip_reader_extract_to_file(mz_zip_archive *pZip, mz_uint file_index,
                                      const char *pDst_filename, mz_uint flags);
mz_bool mz_zip_reader_extract_file_to_file(mz_zip_archive *pZip,
                                           const char *pArchive_filename,
                                           const char *pDst_filename,
                                           mz_uint flags);
#endif

// Ends archive reading, freeing all allocations, and closing the input archive
// file if mz_zip_reader_init_file() was used.
mz_bool mz_zip_reader_end(mz_zip_archive *pZip);

// ZIP archive writing

#ifndef MINIZ_NO_ARCHIVE_WRITING_APIS

// Inits a ZIP archive writer.
mz_bool mz_zip_writer_init(mz_zip_archive *pZip, mz_uint64 existing_size);
mz_bool mz_zip_writer_init_heap(mz_zip_archive *pZip,
                                size_t size_to_reserve_at_beginning,
                                size_t initial_allocation_size);

#ifndef MINIZ_NO_STDIO
mz_bool mz_zip_writer_init_file(mz_zip_archive *pZip, const char *pFilename,
                                mz_uint64 size_to_reserve_at_beginning);
#endif

// Converts a ZIP archive reader object into a writer object, to allow efficient
// in-place file appends to occur on an existing archive.
// For archives opened using mz_zip_reader_init_file, pFilename must be the
// archive's filename so it can be reopened for writing. If the file can't be
// reopened, mz_zip_reader_end() will be called.
// For archives opened using mz_zip_reader_init_mem, the memory block must be
// growable using the realloc callback (which defaults to realloc unless you've
// overridden it).
// Finally, for archives opened using mz_zip_reader_init, the mz_zip_archive's
// user provided m_pWrite function cannot be NULL.
// Note: In-place archive modification is not recommended unless you know what
// you're doing, because if execution stops or something goes wrong before
// the archive is finalized the file's central directory will be hosed.
mz_bool mz_zip_writer_init_from_reader(mz_zip_archive *pZip,
                                       const char *pFilename);

// Adds the contents of a memory buffer to an archive. These functions record
// the current local time into the archive.
// To add a directory entry, call this method with an archive name ending in a
// forwardslash with empty buffer.
// level_and_flags - compression level (0-10, see MZ_BEST_SPEED,
// MZ_BEST_COMPRESSION, etc.) logically OR'd with zero or more mz_zip_flags, or
// just set to MZ_DEFAULT_COMPRESSION.
mz_bool mz_zip_writer_add_mem(mz_zip_archive *pZip, const char *pArchive_name,
                              const void *pBuf, size_t buf_size,
                              mz_uint level_and_flags);
mz_bool mz_zip_writer_add_mem_ex(mz_zip_archive *pZip,
                                 const char *pArchive_name, const void *pBuf,
                                 size_t buf_size, const void *pComment,
                                 mz_uint16 comment_size,
                                 mz_uint level_and_flags, mz_uint64 uncomp_size,
                                 mz_uint32 uncomp_crc32);

#ifndef MINIZ_NO_STDIO
// Adds the contents of a disk file to an archive. This function also records
// the disk file's modified time into the archive.
// level_and_flags - compression level (0-10, see MZ_BEST_SPEED,
// MZ_BEST_COMPRESSION, etc.) logically OR'd with zero or more mz_zip_flags, or
// just set to MZ_DEFAULT_COMPRESSION.
mz_bool mz_zip_writer_add_file(mz_zip_archive *pZip, const char *pArchive_name,
                               const char *pSrc_filename, const void *pComment,
                               mz_uint16 comment_size, mz_uint level_and_flags);
#endif

// Adds a file to an archive by fully cloning the data from another archive.
// This function fully clones the source file's compressed data (no
// recompression), along with its full filename, extra data, and comment fields.
mz_bool mz_zip_writer_add_from_zip_reader(mz_zip_archive *pZip,
                                          mz_zip_archive *pSource_zip,
                                          mz_uint file_index);

// Finalizes the archive by writing the central directory records followed by
// the end of central directory record.
// After an archive is finalized, the only valid call on the mz_zip_archive
// struct is mz_zip_writer_end().
// An archive must be manually finalized by calling this function for it to be
// valid.
mz_bool mz_zip_writer_finalize_archive(mz_zip_archive *pZip);
mz_bool mz_zip_writer_finalize_heap_archive(mz_zip_archive *pZip, void **pBuf,
                                            size_t *pSize);

// Ends archive writing, freeing all allocations, and closing the output file if
// mz_zip_writer_init_file() was used.
// Note for the archive to be valid, it must have been finalized before ending.
mz_bool mz_zip_writer_end(mz_zip_archive *pZip);

// Misc. high-level helper functions:

// mz_zip_add_mem_to_archive_file_in_place() efficiently (but not atomically)
// appends a memory blob to a ZIP archive.
// level_and_flags - compression level (0-10, see MZ_BEST_SPEED,
// MZ_BEST_COMPRESSION, etc.) logically OR'd with zero or more mz_zip_flags, or
// just set to MZ_DEFAULT_COMPRESSION.
mz_bool mz_zip_add_mem_to_archive_file_in_place(
    const char *pZip_filename, const char *pArchive_name, const void *pBuf,
    size_t buf_size, const void *pComment, mz_uint16 comment_size,
    mz_uint level_and_flags);

// Reads a single file from an archive into a heap block.
// Returns NULL on failure.
void *mz_zip_extract_archive_file_to_heap(const char *pZip_filename,
                                          const char *pArchive_name,
                                          size_t *pSize, mz_uint zip_flags);

#endif  // #ifndef MINIZ_NO_ARCHIVE_WRITING_APIS

#endif  // #ifndef MINIZ_NO_ARCHIVE_APIS

// ------------------- Low-level Decompression API Definitions

// Decompression flags used by tinfl_decompress().
// TINFL_FLAG_PARSE_ZLIB_HEADER: If set, the input has a valid zlib header and
// ends with an adler32 checksum (it's a valid zlib stream). Otherwise, the
// input is a raw deflate stream.
// TINFL_FLAG_HAS_MORE_INPUT: If set, there are more input bytes available
// beyond the end of the supplied input buffer. If clear, the input buffer
// contains all remaining input.
// TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF: If set, the output buffer is large
// enough to hold the entire decompressed stream. If clear, the output buffer is
// at least the size of the dictionary (typically 32KB).
// TINFL_FLAG_COMPUTE_ADLER32: Force adler-32 checksum computation of the
// decompressed bytes.
enum {
  TINFL_FLAG_PARSE_ZLIB_HEADER = 1,
  TINFL_FLAG_HAS_MORE_INPUT = 2,
  TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF = 4,
  TINFL_FLAG_COMPUTE_ADLER32 = 8
};

// High level decompression functions:
// tinfl_decompress_mem_to_heap() decompresses a block in memory to a heap block
// allocated via malloc().
// On entry:
//  pSrc_buf, src_buf_len: Pointer and size of the Deflate or zlib source data
//  to decompress.
// On return:
//  Function returns a pointer to the decompressed data, or NULL on failure.
//  *pOut_len will be set to the decompressed data's size, which could be larger
//  than src_buf_len on uncompressible data.
//  The caller must call mz_free() on the returned block when it's no longer
//  needed.
void *tinfl_decompress_mem_to_heap(const void *pSrc_buf, size_t src_buf_len,
                                   size_t *pOut_len, int flags);

// tinfl_decompress_mem_to_mem() decompresses a block in memory to another block
// in memory.
// Returns TINFL_DECOMPRESS_MEM_TO_MEM_FAILED on failure, or the number of bytes
// written on success.
#define TINFL_DECOMPRESS_MEM_TO_MEM_FAILED ((size_t)(-1))
size_t tinfl_decompress_mem_to_mem(void *pOut_buf, size_t out_buf_len,
                                   const void *pSrc_buf, size_t src_buf_len,
                                   int flags);

// tinfl_decompress_mem_to_callback() decompresses a block in memory to an
// internal 32KB buffer, and a user provided callback function will be called to
// flush the buffer.
// Returns 1 on success or 0 on failure.
typedef int (*tinfl_put_buf_func_ptr)(const void *pBuf, int len, void *pUser);
int tinfl_decompress_mem_to_callback(const void *pIn_buf, size_t *pIn_buf_size,
                                     tinfl_put_buf_func_ptr pPut_buf_func,
                                     void *pPut_buf_user, int flags);

struct tinfl_decompressor_tag;
typedef struct tinfl_decompressor_tag tinfl_decompressor;

// Max size of LZ dictionary.
#define TINFL_LZ_DICT_SIZE 32768

// Return status.
typedef enum {
  TINFL_STATUS_BAD_PARAM = -3,
  TINFL_STATUS_ADLER32_MISMATCH = -2,
  TINFL_STATUS_FAILED = -1,
  TINFL_STATUS_DONE = 0,
  TINFL_STATUS_NEEDS_MORE_INPUT = 1,
  TINFL_STATUS_HAS_MORE_OUTPUT = 2
} tinfl_status;

// Initializes the decompressor to its initial state.
#define tinfl_init(r) \
  do {                \
    (r)->m_state = 0; \
  }                   \
  MZ_MACRO_END
#define tinfl_get_adler32(r) (r)->m_check_adler32

// Main low-level decompressor coroutine function. This is the only function
// actually needed for decompression. All the other functions are just
// high-level helpers for improved usability.
// This is a universal API, i.e. it can be used as a building block to build any
// desired higher level decompression API. In the limit case, it can be called
// once per every byte input or output.
tinfl_status tinfl_decompress(tinfl_decompressor *r,
                              const mz_uint8 *pIn_buf_next,
                              size_t *pIn_buf_size, mz_uint8 *pOut_buf_start,
                              mz_uint8 *pOut_buf_next, size_t *pOut_buf_size,
                              const mz_uint32 decomp_flags);

// Internal/private bits follow.
enum {
  TINFL_MAX_HUFF_TABLES = 3,
  TINFL_MAX_HUFF_SYMBOLS_0 = 288,
  TINFL_MAX_HUFF_SYMBOLS_1 = 32,
  TINFL_MAX_HUFF_SYMBOLS_2 = 19,
  TINFL_FAST_LOOKUP_BITS = 10,
  TINFL_FAST_LOOKUP_SIZE = 1 << TINFL_FAST_LOOKUP_BITS
};

typedef struct {
  mz_uint8 m_code_size[TINFL_MAX_HUFF_SYMBOLS_0];
  mz_int16 m_look_up[TINFL_FAST_LOOKUP_SIZE],
      m_tree[TINFL_MAX_HUFF_SYMBOLS_0 * 2];
} tinfl_huff_table;

#if MINIZ_HAS_64BIT_REGISTERS
#define TINFL_USE_64BIT_BITBUF 1
#endif

#if TINFL_USE_64BIT_BITBUF
typedef mz_uint64 tinfl_bit_buf_t;
#define TINFL_BITBUF_SIZE (64)
#else
typedef mz_uint32 tinfl_bit_buf_t;
#define TINFL_BITBUF_SIZE (32)
#endif

struct tinfl_decompressor_tag {
  mz_uint32 m_state, m_num_bits, m_zhdr0, m_zhdr1, m_z_adler32, m_final, m_type,
      m_check_adler32, m_dist, m_counter, m_num_extra,
      m_table_sizes[TINFL_MAX_HUFF_TABLES];
  tinfl_bit_buf_t m_bit_buf;
  size_t m_dist_from_out_buf_start;
  tinfl_huff_table m_tables[TINFL_MAX_HUFF_TABLES];
  mz_uint8 m_raw_header[4],
      m_len_codes[TINFL_MAX_HUFF_SYMBOLS_0 + TINFL_MAX_HUFF_SYMBOLS_1 + 137];
};

// ------------------- Low-level Compression API Definitions

// Set TDEFL_LESS_MEMORY to 1 to use less memory (compression will be slightly
// slower, and raw/dynamic blocks will be output more frequently).
#define TDEFL_LESS_MEMORY 0

// tdefl_init() compression flags logically OR'd together (low 12 bits contain
// the max. number of probes per dictionary search):
// TDEFL_DEFAULT_MAX_PROBES: The compressor defaults to 128 dictionary probes
// per dictionary search. 0=Huffman only, 1=Huffman+LZ (fastest/crap
// compression), 4095=Huffman+LZ (slowest/best compression).
enum {
  TDEFL_HUFFMAN_ONLY = 0,
  TDEFL_DEFAULT_MAX_PROBES = 128,
  TDEFL_MAX_PROBES_MASK = 0xFFF
};

// TDEFL_WRITE_ZLIB_HEADER: If set, the compressor outputs a zlib header before
// the deflate data, and the Adler-32 of the source data at the end. Otherwise,
// you'll get raw deflate data.
// TDEFL_COMPUTE_ADLER32: Always compute the adler-32 of the input data (even
// when not writing zlib headers).
// TDEFL_GREEDY_PARSING_FLAG: Set to use faster greedy parsing, instead of more
// efficient lazy parsing.
// TDEFL_NONDETERMINISTIC_PARSING_FLAG: Enable to decrease the compressor's
// initialization time to the minimum, but the output may vary from run to run
// given the same input (depending on the contents of memory).
// TDEFL_RLE_MATCHES: Only look for RLE matches (matches with a distance of 1)
// TDEFL_FILTER_MATCHES: Discards matches <= 5 chars if enabled.
// TDEFL_FORCE_ALL_STATIC_BLOCKS: Disable usage of optimized Huffman tables.
// TDEFL_FORCE_ALL_RAW_BLOCKS: Only use raw (uncompressed) deflate blocks.
// The low 12 bits are reserved to control the max # of hash probes per
// dictionary lookup (see TDEFL_MAX_PROBES_MASK).
enum {
  TDEFL_WRITE_ZLIB_HEADER = 0x01000,
  TDEFL_COMPUTE_ADLER32 = 0x02000,
  TDEFL_GREEDY_PARSING_FLAG = 0x04000,
  TDEFL_NONDETERMINISTIC_PARSING_FLAG = 0x08000,
  TDEFL_RLE_MATCHES = 0x10000,
  TDEFL_FILTER_MATCHES = 0x20000,
  TDEFL_FORCE_ALL_STATIC_BLOCKS = 0x40000,
  TDEFL_FORCE_ALL_RAW_BLOCKS = 0x80000
};

// High level compression functions:
// tdefl_compress_mem_to_heap() compresses a block in memory to a heap block
// allocated via malloc().
// On entry:
//  pSrc_buf, src_buf_len: Pointer and size of source block to compress.
//  flags: The max match finder probes (default is 128) logically OR'd against
//  the above flags. Higher probes are slower but improve compression.
// On return:
//  Function returns a pointer to the compressed data, or NULL on failure.
//  *pOut_len will be set to the compressed data's size, which could be larger
//  than src_buf_len on uncompressible data.
//  The caller must free() the returned block when it's no longer needed.
void *tdefl_compress_mem_to_heap(const void *pSrc_buf, size_t src_buf_len,
                                 size_t *pOut_len, int flags);

// tdefl_compress_mem_to_mem() compresses a block in memory to another block in
// memory.
// Returns 0 on failure.
size_t tdefl_compress_mem_to_mem(void *pOut_buf, size_t out_buf_len,
                                 const void *pSrc_buf, size_t src_buf_len,
                                 int flags);

// Compresses an image to a compressed PNG file in memory.
// On entry:
//  pImage, w, h, and num_chans describe the image to compress. num_chans may be
//  1, 2, 3, or 4.
//  The image pitch in bytes per scanline will be w*num_chans. The leftmost
//  pixel on the top scanline is stored first in memory.
//  level may range from [0,10], use MZ_NO_COMPRESSION, MZ_BEST_SPEED,
//  MZ_BEST_COMPRESSION, etc. or a decent default is MZ_DEFAULT_LEVEL
//  If flip is true, the image will be flipped on the Y axis (useful for OpenGL
//  apps).
// On return:
//  Function returns a pointer to the compressed data, or NULL on failure.
//  *pLen_out will be set to the size of the PNG image file.
//  The caller must mz_free() the returned heap block (which will typically be
//  larger than *pLen_out) when it's no longer needed.
void *tdefl_write_image_to_png_file_in_memory_ex(const void *pImage, int w,
                                                 int h, int num_chans,
                                                 size_t *pLen_out,
                                                 mz_uint level, mz_bool flip);
void *tdefl_write_image_to_png_file_in_memory(const void *pImage, int w, int h,
                                              int num_chans, size_t *pLen_out);

// Output stream interface. The compressor uses this interface to write
// compressed data. It'll typically be called TDEFL_OUT_BUF_SIZE at a time.
typedef mz_bool (*tdefl_put_buf_func_ptr)(const void *pBuf, int len,
                                          void *pUser);

// tdefl_compress_mem_to_output() compresses a block to an output stream. The
// above helpers use this function internally.
mz_bool tdefl_compress_mem_to_output(const void *pBuf, size_t buf_len,
                                     tdefl_put_buf_func_ptr pPut_buf_func,
                                     void *pPut_buf_user, int flags);

enum {
  TDEFL_MAX_HUFF_TABLES = 3,
  TDEFL_MAX_HUFF_SYMBOLS_0 = 288,
  TDEFL_MAX_HUFF_SYMBOLS_1 = 32,
  TDEFL_MAX_HUFF_SYMBOLS_2 = 19,
  TDEFL_LZ_DICT_SIZE = 32768,
  TDEFL_LZ_DICT_SIZE_MASK = TDEFL_LZ_DICT_SIZE - 1,
  TDEFL_MIN_MATCH_LEN = 3,
  TDEFL_MAX_MATCH_LEN = 258
};

// TDEFL_OUT_BUF_SIZE MUST be large enough to hold a single entire compressed
// output block (using static/fixed Huffman codes).
#if TDEFL_LESS_MEMORY
enum {
  TDEFL_LZ_CODE_BUF_SIZE = 24 * 1024,
  TDEFL_OUT_BUF_SIZE = (TDEFL_LZ_CODE_BUF_SIZE * 13) / 10,
  TDEFL_MAX_HUFF_SYMBOLS = 288,
  TDEFL_LZ_HASH_BITS = 12,
  TDEFL_LEVEL1_HASH_SIZE_MASK = 4095,
  TDEFL_LZ_HASH_SHIFT = (TDEFL_LZ_HASH_BITS + 2) / 3,
  TDEFL_LZ_HASH_SIZE = 1 << TDEFL_LZ_HASH_BITS
};
#else
enum {
  TDEFL_LZ_CODE_BUF_SIZE = 64 * 1024,
  TDEFL_OUT_BUF_SIZE = (TDEFL_LZ_CODE_BUF_SIZE * 13) / 10,
  TDEFL_MAX_HUFF_SYMBOLS = 288,
  TDEFL_LZ_HASH_BITS = 15,
  TDEFL_LEVEL1_HASH_SIZE_MASK = 4095,
  TDEFL_LZ_HASH_SHIFT = (TDEFL_LZ_HASH_BITS + 2) / 3,
  TDEFL_LZ_HASH_SIZE = 1 << TDEFL_LZ_HASH_BITS
};
#endif

// The low-level tdefl functions below may be used directly if the above helper
// functions aren't flexible enough. The low-level functions don't make any heap
// allocations, unlike the above helper functions.
typedef enum {
  TDEFL_STATUS_BAD_PARAM = -2,
  TDEFL_STATUS_PUT_BUF_FAILED = -1,
  TDEFL_STATUS_OKAY = 0,
  TDEFL_STATUS_DONE = 1
} tdefl_status;

// Must map to MZ_NO_FLUSH, MZ_SYNC_FLUSH, etc. enums
typedef enum {
  TDEFL_NO_FLUSH = 0,
  TDEFL_SYNC_FLUSH = 2,
  TDEFL_FULL_FLUSH = 3,
  TDEFL_FINISH = 4
} tdefl_flush;

// tdefl's compression state structure.
typedef struct {
  tdefl_put_buf_func_ptr m_pPut_buf_func;
  void *m_pPut_buf_user;
  mz_uint m_flags, m_max_probes[2];
  int m_greedy_parsing;
  mz_uint m_adler32, m_lookahead_pos, m_lookahead_size, m_dict_size;
  mz_uint8 *m_pLZ_code_buf, *m_pLZ_flags, *m_pOutput_buf, *m_pOutput_buf_end;
  mz_uint m_num_flags_left, m_total_lz_bytes, m_lz_code_buf_dict_pos, m_bits_in,
      m_bit_buffer;
  mz_uint m_saved_match_dist, m_saved_match_len, m_saved_lit,
      m_output_flush_ofs, m_output_flush_remaining, m_finished, m_block_index,
      m_wants_to_finish;
  tdefl_status m_prev_return_status;
  const void *m_pIn_buf;
  void *m_pOut_buf;
  size_t *m_pIn_buf_size, *m_pOut_buf_size;
  tdefl_flush m_flush;
  const mz_uint8 *m_pSrc;
  size_t m_src_buf_left, m_out_buf_ofs;
  mz_uint8 m_dict[TDEFL_LZ_DICT_SIZE + TDEFL_MAX_MATCH_LEN - 1];
  mz_uint16 m_huff_count[TDEFL_MAX_HUFF_TABLES][TDEFL_MAX_HUFF_SYMBOLS];
  mz_uint16 m_huff_codes[TDEFL_MAX_HUFF_TABLES][TDEFL_MAX_HUFF_SYMBOLS];
  mz_uint8 m_huff_code_sizes[TDEFL_MAX_HUFF_TABLES][TDEFL_MAX_HUFF_SYMBOLS];
  mz_uint8 m_lz_code_buf[TDEFL_LZ_CODE_BUF_SIZE];
  mz_uint16 m_next[TDEFL_LZ_DICT_SIZE];
  mz_uint16 m_hash[TDEFL_LZ_HASH_SIZE];
  mz_uint8 m_output_buf[TDEFL_OUT_BUF_SIZE];
} tdefl_compressor;

// Initializes the compressor.
// There is no corresponding deinit() function because the tdefl API's do not
// dynamically allocate memory.
// pBut_buf_func: If NULL, output data will be supplied to the specified
// callback. In this case, the user should call the tdefl_compress_buffer() API
// for compression.
// If pBut_buf_func is NULL the user should always call the tdefl_compress()
// API.
// flags: See the above enums (TDEFL_HUFFMAN_ONLY, TDEFL_WRITE_ZLIB_HEADER,
// etc.)
tdefl_status tdefl_init(tdefl_compressor *d,
                        tdefl_put_buf_func_ptr pPut_buf_func,
                        void *pPut_buf_user, int flags);

// Compresses a block of data, consuming as much of the specified input buffer
// as possible, and writing as much compressed data to the specified output
// buffer as possible.
tdefl_status tdefl_compress(tdefl_compressor *d, const void *pIn_buf,
                            size_t *pIn_buf_size, void *pOut_buf,
                            size_t *pOut_buf_size, tdefl_flush flush);

// tdefl_compress_buffer() is only usable when the tdefl_init() is called with a
// non-NULL tdefl_put_buf_func_ptr.
// tdefl_compress_buffer() always consumes the entire input buffer.
tdefl_status tdefl_compress_buffer(tdefl_compressor *d, const void *pIn_buf,
                                   size_t in_buf_size, tdefl_flush flush);

tdefl_status tdefl_get_prev_return_status(tdefl_compressor *d);
mz_uint32 tdefl_get_adler32(tdefl_compressor *d);

// Can't use tdefl_create_comp_flags_from_zip_params if MINIZ_NO_ZLIB_APIS isn't
// defined, because it uses some of its macros.
#ifndef MINIZ_NO_ZLIB_APIS
// Create tdefl_compress() flags given zlib-style compression parameters.
// level may range from [0,10] (where 10 is absolute max compression, but may be
// much slower on some files)
// window_bits may be -15 (raw deflate) or 15 (zlib)
// strategy may be either MZ_DEFAULT_STRATEGY, MZ_FILTERED, MZ_HUFFMAN_ONLY,
// MZ_RLE, or MZ_FIXED
mz_uint tdefl_create_comp_flags_from_zip_params(int level, int window_bits,
                                                int strategy);
#endif  // #ifndef MINIZ_NO_ZLIB_APIS

#ifdef __cplusplus
}
#endif

#endif  // MINIZ_HEADER_INCLUDED

// ------------------- End of Header: Implementation follows. (If you only want
// the header, define MINIZ_HEADER_FILE_ONLY.)

#ifndef MINIZ_HEADER_FILE_ONLY

typedef unsigned char mz_validate_uint16[sizeof(mz_uint16) == 2 ? 1 : -1];
typedef unsigned char mz_validate_uint32[sizeof(mz_uint32) == 4 ? 1 : -1];
typedef unsigned char mz_validate_uint64[sizeof(mz_uint64) == 8 ? 1 : -1];

//#include <assert.h>
//#include <string.h>

#define MZ_ASSERT(x) assert(x)

#ifdef MINIZ_NO_MALLOC
#define MZ_MALLOC(x) NULL
#define MZ_FREE(x) (void)x, ((void)0)
#define MZ_REALLOC(p, x) NULL
#else
#define MZ_MALLOC(x) malloc(x)
#define MZ_FREE(x) free(x)
#define MZ_REALLOC(p, x) realloc(p, x)
#endif

#define MZ_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MZ_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MZ_CLEAR_OBJ(obj) memset(&(obj), 0, sizeof(obj))

#if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
#define MZ_READ_LE16(p) *((const mz_uint16 *)(p))
#define MZ_READ_LE32(p) *((const mz_uint32 *)(p))
#else
#define MZ_READ_LE16(p)                      \
  ((mz_uint32)(((const mz_uint8 *)(p))[0]) | \
   ((mz_uint32)(((const mz_uint8 *)(p))[1]) << 8U))
#define MZ_READ_LE32(p)                               \
  ((mz_uint32)(((const mz_uint8 *)(p))[0]) |          \
   ((mz_uint32)(((const mz_uint8 *)(p))[1]) << 8U) |  \
   ((mz_uint32)(((const mz_uint8 *)(p))[2]) << 16U) | \
   ((mz_uint32)(((const mz_uint8 *)(p))[3]) << 24U))
#endif

#ifdef _MSC_VER
#define MZ_FORCEINLINE __forceinline
#elif defined(__GNUC__)
#define MZ_FORCEINLINE inline __attribute__((__always_inline__))
#else
#define MZ_FORCEINLINE inline
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ------------------- zlib-style API's

mz_ulong mz_adler32(mz_ulong adler, const unsigned char *ptr, size_t buf_len) {
  mz_uint32 i, s1 = (mz_uint32)(adler & 0xffff), s2 = (mz_uint32)(adler >> 16);
  size_t block_len = buf_len % 5552;
  if (!ptr) return MZ_ADLER32_INIT;
  while (buf_len) {
    for (i = 0; i + 7 < block_len; i += 8, ptr += 8) {
      s1 += ptr[0], s2 += s1;
      s1 += ptr[1], s2 += s1;
      s1 += ptr[2], s2 += s1;
      s1 += ptr[3], s2 += s1;
      s1 += ptr[4], s2 += s1;
      s1 += ptr[5], s2 += s1;
      s1 += ptr[6], s2 += s1;
      s1 += ptr[7], s2 += s1;
    }
    for (; i < block_len; ++i) s1 += *ptr++, s2 += s1;
    s1 %= 65521U, s2 %= 65521U;
    buf_len -= block_len;
    block_len = 5552;
  }
  return (s2 << 16) + s1;
}

// Karl Malbrain's compact CRC-32. See "A compact CCITT crc16 and crc32 C
// implementation that balances processor cache usage against speed":
// http://www.geocities.com/malbrain/
mz_ulong mz_crc32(mz_ulong crc, const mz_uint8 *ptr, size_t buf_len) {
  static const mz_uint32 s_crc32[16] = {
      0,          0x1db71064, 0x3b6e20c8, 0x26d930ac, 0x76dc4190, 0x6b6b51f4,
      0x4db26158, 0x5005713c, 0xedb88320, 0xf00f9344, 0xd6d6a3e8, 0xcb61b38c,
      0x9b64c2b0, 0x86d3d2d4, 0xa00ae278, 0xbdbdf21c};
  mz_uint32 crcu32 = (mz_uint32)crc;
  if (!ptr) return MZ_CRC32_INIT;
  crcu32 = ~crcu32;
  while (buf_len--) {
    mz_uint8 b = *ptr++;
    crcu32 = (crcu32 >> 4) ^ s_crc32[(crcu32 & 0xF) ^ (b & 0xF)];
    crcu32 = (crcu32 >> 4) ^ s_crc32[(crcu32 & 0xF) ^ (b >> 4)];
  }
  return ~crcu32;
}

void mz_free(void *p) { MZ_FREE(p); }

#ifndef MINIZ_NO_ZLIB_APIS

static void *def_alloc_func(void *opaque, size_t items, size_t size) {
  (void)opaque, (void)items, (void)size;
  return MZ_MALLOC(items * size);
}
static void def_free_func(void *opaque, void *address) {
  (void)opaque, (void)address;
  MZ_FREE(address);
}
// static void *def_realloc_func(void *opaque, void *address, size_t items,
//                              size_t size) {
//  (void)opaque, (void)address, (void)items, (void)size;
//  return MZ_REALLOC(address, items * size);
//}

const char *mz_version(void) { return MZ_VERSION; }

int mz_deflateInit(mz_streamp pStream, int level) {
  return mz_deflateInit2(pStream, level, MZ_DEFLATED, MZ_DEFAULT_WINDOW_BITS, 9,
                         MZ_DEFAULT_STRATEGY);
}

int mz_deflateInit2(mz_streamp pStream, int level, int method, int window_bits,
                    int mem_level, int strategy) {
  tdefl_compressor *pComp;
  mz_uint comp_flags =
      TDEFL_COMPUTE_ADLER32 |
      tdefl_create_comp_flags_from_zip_params(level, window_bits, strategy);

  if (!pStream) return MZ_STREAM_ERROR;
  if ((method != MZ_DEFLATED) || ((mem_level < 1) || (mem_level > 9)) ||
      ((window_bits != MZ_DEFAULT_WINDOW_BITS) &&
       (-window_bits != MZ_DEFAULT_WINDOW_BITS)))
    return MZ_PARAM_ERROR;

  pStream->data_type = 0;
  pStream->adler = MZ_ADLER32_INIT;
  pStream->msg = NULL;
  pStream->reserved = 0;
  pStream->total_in = 0;
  pStream->total_out = 0;
  if (!pStream->zalloc) pStream->zalloc = def_alloc_func;
  if (!pStream->zfree) pStream->zfree = def_free_func;

  pComp = (tdefl_compressor *)pStream->zalloc(pStream->opaque, 1,
                                              sizeof(tdefl_compressor));
  if (!pComp) return MZ_MEM_ERROR;

  pStream->state = (struct mz_internal_state *)pComp;

  if (tdefl_init(pComp, NULL, NULL, comp_flags) != TDEFL_STATUS_OKAY) {
    mz_deflateEnd(pStream);
    return MZ_PARAM_ERROR;
  }

  return MZ_OK;
}

int mz_deflateReset(mz_streamp pStream) {
  if ((!pStream) || (!pStream->state) || (!pStream->zalloc) ||
      (!pStream->zfree))
    return MZ_STREAM_ERROR;
  pStream->total_in = pStream->total_out = 0;
  tdefl_init((tdefl_compressor *)pStream->state, NULL, NULL,
             ((tdefl_compressor *)pStream->state)->m_flags);
  return MZ_OK;
}

int mz_deflate(mz_streamp pStream, int flush) {
  size_t in_bytes, out_bytes;
  mz_ulong orig_total_in, orig_total_out;
  int mz_status = MZ_OK;

  if ((!pStream) || (!pStream->state) || (flush < 0) || (flush > MZ_FINISH) ||
      (!pStream->next_out))
    return MZ_STREAM_ERROR;
  if (!pStream->avail_out) return MZ_BUF_ERROR;

  if (flush == MZ_PARTIAL_FLUSH) flush = MZ_SYNC_FLUSH;

  if (((tdefl_compressor *)pStream->state)->m_prev_return_status ==
      TDEFL_STATUS_DONE)
    return (flush == MZ_FINISH) ? MZ_STREAM_END : MZ_BUF_ERROR;

  orig_total_in = pStream->total_in;
  orig_total_out = pStream->total_out;
  for (;;) {
    tdefl_status defl_status;
    in_bytes = pStream->avail_in;
    out_bytes = pStream->avail_out;

    defl_status = tdefl_compress((tdefl_compressor *)pStream->state,
                                 pStream->next_in, &in_bytes, pStream->next_out,
                                 &out_bytes, (tdefl_flush)flush);
    pStream->next_in += (mz_uint)in_bytes;
    pStream->avail_in -= (mz_uint)in_bytes;
    pStream->total_in += (mz_uint)in_bytes;
    pStream->adler = tdefl_get_adler32((tdefl_compressor *)pStream->state);

    pStream->next_out += (mz_uint)out_bytes;
    pStream->avail_out -= (mz_uint)out_bytes;
    pStream->total_out += (mz_uint)out_bytes;

    if (defl_status < 0) {
      mz_status = MZ_STREAM_ERROR;
      break;
    } else if (defl_status == TDEFL_STATUS_DONE) {
      mz_status = MZ_STREAM_END;
      break;
    } else if (!pStream->avail_out)
      break;
    else if ((!pStream->avail_in) && (flush != MZ_FINISH)) {
      if ((flush) || (pStream->total_in != orig_total_in) ||
          (pStream->total_out != orig_total_out))
        break;
      return MZ_BUF_ERROR;  // Can't make forward progress without some input.
    }
  }
  return mz_status;
}

int mz_deflateEnd(mz_streamp pStream) {
  if (!pStream) return MZ_STREAM_ERROR;
  if (pStream->state) {
    pStream->zfree(pStream->opaque, pStream->state);
    pStream->state = NULL;
  }
  return MZ_OK;
}

mz_ulong mz_deflateBound(mz_streamp pStream, mz_ulong source_len) {
  (void)pStream;
  // This is really over conservative. (And lame, but it's actually pretty
  // tricky to compute a true upper bound given the way tdefl's blocking works.)
  return MZ_MAX(128 + (source_len * 110) / 100,
                128 + source_len + ((source_len / (31 * 1024)) + 1) * 5);
}

int mz_compress2(unsigned char *pDest, mz_ulong *pDest_len,
                 const unsigned char *pSource, mz_ulong source_len, int level) {
  int status;
  mz_stream stream;
  memset(&stream, 0, sizeof(stream));

  // In case mz_ulong is 64-bits (argh I hate longs).
  if ((source_len | *pDest_len) > 0xFFFFFFFFU) return MZ_PARAM_ERROR;

  stream.next_in = pSource;
  stream.avail_in = (mz_uint32)source_len;
  stream.next_out = pDest;
  stream.avail_out = (mz_uint32)*pDest_len;

  status = mz_deflateInit(&stream, level);
  if (status != MZ_OK) return status;

  status = mz_deflate(&stream, MZ_FINISH);
  if (status != MZ_STREAM_END) {
    mz_deflateEnd(&stream);
    return (status == MZ_OK) ? MZ_BUF_ERROR : status;
  }

  *pDest_len = stream.total_out;
  return mz_deflateEnd(&stream);
}

int mz_compress(unsigned char *pDest, mz_ulong *pDest_len,
                const unsigned char *pSource, mz_ulong source_len) {
  return mz_compress2(pDest, pDest_len, pSource, source_len,
                      MZ_DEFAULT_COMPRESSION);
}

mz_ulong mz_compressBound(mz_ulong source_len) {
  return mz_deflateBound(NULL, source_len);
}

typedef struct {
  tinfl_decompressor m_decomp;
  mz_uint m_dict_ofs, m_dict_avail, m_first_call, m_has_flushed;
  int m_window_bits;
  mz_uint8 m_dict[TINFL_LZ_DICT_SIZE];
  tinfl_status m_last_status;
} inflate_state;

int mz_inflateInit2(mz_streamp pStream, int window_bits) {
  inflate_state *pDecomp;
  if (!pStream) return MZ_STREAM_ERROR;
  if ((window_bits != MZ_DEFAULT_WINDOW_BITS) &&
      (-window_bits != MZ_DEFAULT_WINDOW_BITS))
    return MZ_PARAM_ERROR;

  pStream->data_type = 0;
  pStream->adler = 0;
  pStream->msg = NULL;
  pStream->total_in = 0;
  pStream->total_out = 0;
  pStream->reserved = 0;
  if (!pStream->zalloc) pStream->zalloc = def_alloc_func;
  if (!pStream->zfree) pStream->zfree = def_free_func;

  pDecomp = (inflate_state *)pStream->zalloc(pStream->opaque, 1,
                                             sizeof(inflate_state));
  if (!pDecomp) return MZ_MEM_ERROR;

  pStream->state = (struct mz_internal_state *)pDecomp;

  tinfl_init(&pDecomp->m_decomp);
  pDecomp->m_dict_ofs = 0;
  pDecomp->m_dict_avail = 0;
  pDecomp->m_last_status = TINFL_STATUS_NEEDS_MORE_INPUT;
  pDecomp->m_first_call = 1;
  pDecomp->m_has_flushed = 0;
  pDecomp->m_window_bits = window_bits;

  return MZ_OK;
}

int mz_inflateInit(mz_streamp pStream) {
  return mz_inflateInit2(pStream, MZ_DEFAULT_WINDOW_BITS);
}

int mz_inflate(mz_streamp pStream, int flush) {
  inflate_state *pState;
  mz_uint n, first_call, decomp_flags = TINFL_FLAG_COMPUTE_ADLER32;
  size_t in_bytes, out_bytes, orig_avail_in;
  tinfl_status status;

  if ((!pStream) || (!pStream->state)) return MZ_STREAM_ERROR;
  if (flush == MZ_PARTIAL_FLUSH) flush = MZ_SYNC_FLUSH;
  if ((flush) && (flush != MZ_SYNC_FLUSH) && (flush != MZ_FINISH))
    return MZ_STREAM_ERROR;

  pState = (inflate_state *)pStream->state;
  if (pState->m_window_bits > 0) decomp_flags |= TINFL_FLAG_PARSE_ZLIB_HEADER;
  orig_avail_in = pStream->avail_in;

  first_call = pState->m_first_call;
  pState->m_first_call = 0;
  if (pState->m_last_status < 0) return MZ_DATA_ERROR;

  if (pState->m_has_flushed && (flush != MZ_FINISH)) return MZ_STREAM_ERROR;
  pState->m_has_flushed |= (flush == MZ_FINISH);

  if ((flush == MZ_FINISH) && (first_call)) {
    // MZ_FINISH on the first call implies that the input and output buffers are
    // large enough to hold the entire compressed/decompressed file.
    decomp_flags |= TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
    in_bytes = pStream->avail_in;
    out_bytes = pStream->avail_out;
    status = tinfl_decompress(&pState->m_decomp, pStream->next_in, &in_bytes,
                              pStream->next_out, pStream->next_out, &out_bytes,
                              decomp_flags);
    pState->m_last_status = status;
    pStream->next_in += (mz_uint)in_bytes;
    pStream->avail_in -= (mz_uint)in_bytes;
    pStream->total_in += (mz_uint)in_bytes;
    pStream->adler = tinfl_get_adler32(&pState->m_decomp);
    pStream->next_out += (mz_uint)out_bytes;
    pStream->avail_out -= (mz_uint)out_bytes;
    pStream->total_out += (mz_uint)out_bytes;

    if (status < 0)
      return MZ_DATA_ERROR;
    else if (status != TINFL_STATUS_DONE) {
      pState->m_last_status = TINFL_STATUS_FAILED;
      return MZ_BUF_ERROR;
    }
    return MZ_STREAM_END;
  }
  // flush != MZ_FINISH then we must assume there's more input.
  if (flush != MZ_FINISH) decomp_flags |= TINFL_FLAG_HAS_MORE_INPUT;

  if (pState->m_dict_avail) {
    n = MZ_MIN(pState->m_dict_avail, pStream->avail_out);
    memcpy(pStream->next_out, pState->m_dict + pState->m_dict_ofs, n);
    pStream->next_out += n;
    pStream->avail_out -= n;
    pStream->total_out += n;
    pState->m_dict_avail -= n;
    pState->m_dict_ofs = (pState->m_dict_ofs + n) & (TINFL_LZ_DICT_SIZE - 1);
    return ((pState->m_last_status == TINFL_STATUS_DONE) &&
            (!pState->m_dict_avail))
               ? MZ_STREAM_END
               : MZ_OK;
  }

  for (;;) {
    in_bytes = pStream->avail_in;
    out_bytes = TINFL_LZ_DICT_SIZE - pState->m_dict_ofs;

    status = tinfl_decompress(
        &pState->m_decomp, pStream->next_in, &in_bytes, pState->m_dict,
        pState->m_dict + pState->m_dict_ofs, &out_bytes, decomp_flags);
    pState->m_last_status = status;

    pStream->next_in += (mz_uint)in_bytes;
    pStream->avail_in -= (mz_uint)in_bytes;
    pStream->total_in += (mz_uint)in_bytes;
    pStream->adler = tinfl_get_adler32(&pState->m_decomp);

    pState->m_dict_avail = (mz_uint)out_bytes;

    n = MZ_MIN(pState->m_dict_avail, pStream->avail_out);
    memcpy(pStream->next_out, pState->m_dict + pState->m_dict_ofs, n);
    pStream->next_out += n;
    pStream->avail_out -= n;
    pStream->total_out += n;
    pState->m_dict_avail -= n;
    pState->m_dict_ofs = (pState->m_dict_ofs + n) & (TINFL_LZ_DICT_SIZE - 1);

    if (status < 0)
      return MZ_DATA_ERROR;  // Stream is corrupted (there could be some
    // uncompressed data left in the output dictionary -
    // oh well).
    else if ((status == TINFL_STATUS_NEEDS_MORE_INPUT) && (!orig_avail_in))
      return MZ_BUF_ERROR;  // Signal caller that we can't make forward progress
                            // without supplying more input or by setting flush
                            // to MZ_FINISH.
    else if (flush == MZ_FINISH) {
      // The output buffer MUST be large to hold the remaining uncompressed data
      // when flush==MZ_FINISH.
      if (status == TINFL_STATUS_DONE)
        return pState->m_dict_avail ? MZ_BUF_ERROR : MZ_STREAM_END;
      // status here must be TINFL_STATUS_HAS_MORE_OUTPUT, which means there's
      // at least 1 more byte on the way. If there's no more room left in the
      // output buffer then something is wrong.
      else if (!pStream->avail_out)
        return MZ_BUF_ERROR;
    } else if ((status == TINFL_STATUS_DONE) || (!pStream->avail_in) ||
               (!pStream->avail_out) || (pState->m_dict_avail))
      break;
  }

  return ((status == TINFL_STATUS_DONE) && (!pState->m_dict_avail))
             ? MZ_STREAM_END
             : MZ_OK;
}

int mz_inflateEnd(mz_streamp pStream) {
  if (!pStream) return MZ_STREAM_ERROR;
  if (pStream->state) {
    pStream->zfree(pStream->opaque, pStream->state);
    pStream->state = NULL;
  }
  return MZ_OK;
}

int mz_uncompress(unsigned char *pDest, mz_ulong *pDest_len,
                  const unsigned char *pSource, mz_ulong source_len) {
  mz_stream stream;
  int status;
  memset(&stream, 0, sizeof(stream));

  // In case mz_ulong is 64-bits (argh I hate longs).
  if ((source_len | *pDest_len) > 0xFFFFFFFFU) return MZ_PARAM_ERROR;

  stream.next_in = pSource;
  stream.avail_in = (mz_uint32)source_len;
  stream.next_out = pDest;
  stream.avail_out = (mz_uint32)*pDest_len;

  status = mz_inflateInit(&stream);
  if (status != MZ_OK) return status;

  status = mz_inflate(&stream, MZ_FINISH);
  if (status != MZ_STREAM_END) {
    mz_inflateEnd(&stream);
    return ((status == MZ_BUF_ERROR) && (!stream.avail_in)) ? MZ_DATA_ERROR
                                                            : status;
  }
  *pDest_len = stream.total_out;

  return mz_inflateEnd(&stream);
}

const char *mz_error(int err) {
  static struct {
    int m_err;
    const char *m_pDesc;
  } s_error_descs[] = {{MZ_OK, ""},
                       {MZ_STREAM_END, "stream end"},
                       {MZ_NEED_DICT, "need dictionary"},
                       {MZ_ERRNO, "file error"},
                       {MZ_STREAM_ERROR, "stream error"},
                       {MZ_DATA_ERROR, "data error"},
                       {MZ_MEM_ERROR, "out of memory"},
                       {MZ_BUF_ERROR, "buf error"},
                       {MZ_VERSION_ERROR, "version error"},
                       {MZ_PARAM_ERROR, "parameter error"}};
  mz_uint i;
  for (i = 0; i < sizeof(s_error_descs) / sizeof(s_error_descs[0]); ++i)
    if (s_error_descs[i].m_err == err) return s_error_descs[i].m_pDesc;
  return NULL;
}

#endif  // MINIZ_NO_ZLIB_APIS

// ------------------- Low-level Decompression (completely independent from all
// compression API's)

#define TINFL_MEMCPY(d, s, l) memcpy(d, s, l)
#define TINFL_MEMSET(p, c, l) memset(p, c, l)

#define TINFL_CR_BEGIN  \
  switch (r->m_state) { \
    case 0:
#define TINFL_CR_RETURN(state_index, result) \
  do {                                       \
    status = result;                         \
    r->m_state = state_index;                \
    goto common_exit;                        \
    case state_index:;                       \
  }                                          \
  MZ_MACRO_END
#define TINFL_CR_RETURN_FOREVER(state_index, result) \
  do {                                               \
    for (;;) {                                       \
      TINFL_CR_RETURN(state_index, result);          \
    }                                                \
  }                                                  \
  MZ_MACRO_END
#define TINFL_CR_FINISH }

// TODO: If the caller has indicated that there's no more input, and we attempt
// to read beyond the input buf, then something is wrong with the input because
// the inflator never
// reads ahead more than it needs to. Currently TINFL_GET_BYTE() pads the end of
// the stream with 0's in this scenario.
#define TINFL_GET_BYTE(state_index, c)                                 \
  do {                                                                 \
    if (pIn_buf_cur >= pIn_buf_end) {                                  \
      for (;;) {                                                       \
        if (decomp_flags & TINFL_FLAG_HAS_MORE_INPUT) {                \
          TINFL_CR_RETURN(state_index, TINFL_STATUS_NEEDS_MORE_INPUT); \
          if (pIn_buf_cur < pIn_buf_end) {                             \
            c = *pIn_buf_cur++;                                        \
            break;                                                     \
          }                                                            \
        } else {                                                       \
          c = 0;                                                       \
          break;                                                       \
        }                                                              \
      }                                                                \
    } else                                                             \
      c = *pIn_buf_cur++;                                              \
  }                                                                    \
  MZ_MACRO_END

#define TINFL_NEED_BITS(state_index, n)            \
  do {                                             \
    mz_uint c;                                     \
    TINFL_GET_BYTE(state_index, c);                \
    bit_buf |= (((tinfl_bit_buf_t)c) << num_bits); \
    num_bits += 8;                                 \
  } while (num_bits < (mz_uint)(n))
#define TINFL_SKIP_BITS(state_index, n) \
  do {                                  \
    if (num_bits < (mz_uint)(n)) {      \
      TINFL_NEED_BITS(state_index, n);  \
    }                                   \
    bit_buf >>= (n);                    \
    num_bits -= (n);                    \
  }                                     \
  MZ_MACRO_END
#define TINFL_GET_BITS(state_index, b, n) \
  do {                                    \
    if (num_bits < (mz_uint)(n)) {        \
      TINFL_NEED_BITS(state_index, n);    \
    }                                     \
    b = bit_buf & ((1 << (n)) - 1);       \
    bit_buf >>= (n);                      \
    num_bits -= (n);                      \
  }                                       \
  MZ_MACRO_END

// TINFL_HUFF_BITBUF_FILL() is only used rarely, when the number of bytes
// remaining in the input buffer falls below 2.
// It reads just enough bytes from the input stream that are needed to decode
// the next Huffman code (and absolutely no more). It works by trying to fully
// decode a
// Huffman code by using whatever bits are currently present in the bit buffer.
// If this fails, it reads another byte, and tries again until it succeeds or
// until the
// bit buffer contains >=15 bits (deflate's max. Huffman code size).
#define TINFL_HUFF_BITBUF_FILL(state_index, pHuff)                     \
  do {                                                                 \
    temp = (pHuff)->m_look_up[bit_buf & (TINFL_FAST_LOOKUP_SIZE - 1)]; \
    if (temp >= 0) {                                                   \
      code_len = temp >> 9;                                            \
      if ((code_len) && (num_bits >= code_len)) break;                 \
    } else if (num_bits > TINFL_FAST_LOOKUP_BITS) {                    \
      code_len = TINFL_FAST_LOOKUP_BITS;                               \
      do {                                                             \
        temp = (pHuff)->m_tree[~temp + ((bit_buf >> code_len++) & 1)]; \
      } while ((temp < 0) && (num_bits >= (code_len + 1)));            \
      if (temp >= 0) break;                                            \
    }                                                                  \
    TINFL_GET_BYTE(state_index, c);                                    \
    bit_buf |= (((tinfl_bit_buf_t)c) << num_bits);                     \
    num_bits += 8;                                                     \
  } while (num_bits < 15);

// TINFL_HUFF_DECODE() decodes the next Huffman coded symbol. It's more complex
// than you would initially expect because the zlib API expects the decompressor
// to never read
// beyond the final byte of the deflate stream. (In other words, when this macro
// wants to read another byte from the input, it REALLY needs another byte in
// order to fully
// decode the next Huffman code.) Handling this properly is particularly
// important on raw deflate (non-zlib) streams, which aren't followed by a byte
// aligned adler-32.
// The slow path is only executed at the very end of the input buffer.
#define TINFL_HUFF_DECODE(state_index, sym, pHuff)                             \
  do {                                                                         \
    int temp;                                                                  \
    mz_uint code_len, c;                                                       \
    if (num_bits < 15) {                                                       \
      if ((pIn_buf_end - pIn_buf_cur) < 2) {                                   \
        TINFL_HUFF_BITBUF_FILL(state_index, pHuff);                            \
      } else {                                                                 \
        bit_buf |= (((tinfl_bit_buf_t)pIn_buf_cur[0]) << num_bits) |           \
                   (((tinfl_bit_buf_t)pIn_buf_cur[1]) << (num_bits + 8));      \
        pIn_buf_cur += 2;                                                      \
        num_bits += 16;                                                        \
      }                                                                        \
    }                                                                          \
    if ((temp = (pHuff)->m_look_up[bit_buf & (TINFL_FAST_LOOKUP_SIZE - 1)]) >= \
        0)                                                                     \
      code_len = temp >> 9, temp &= 511;                                       \
    else {                                                                     \
      code_len = TINFL_FAST_LOOKUP_BITS;                                       \
      do {                                                                     \
        temp = (pHuff)->m_tree[~temp + ((bit_buf >> code_len++) & 1)];         \
      } while (temp < 0);                                                      \
    }                                                                          \
    sym = temp;                                                                \
    bit_buf >>= code_len;                                                      \
    num_bits -= code_len;                                                      \
  }                                                                            \
  MZ_MACRO_END

tinfl_status tinfl_decompress(tinfl_decompressor *r,
                              const mz_uint8 *pIn_buf_next,
                              size_t *pIn_buf_size, mz_uint8 *pOut_buf_start,
                              mz_uint8 *pOut_buf_next, size_t *pOut_buf_size,
                              const mz_uint32 decomp_flags) {
  static const int s_length_base[31] = {
      3,  4,  5,  6,  7,  8,  9,  10,  11,  13,  15,  17,  19,  23, 27, 31,
      35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0,  0};
  static const int s_length_extra[31] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                                         1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4,
                                         4, 4, 5, 5, 5, 5, 0, 0, 0};
  static const int s_dist_base[32] = {
      1,    2,    3,    4,    5,    7,     9,     13,    17,  25,   33,
      49,   65,   97,   129,  193,  257,   385,   513,   769, 1025, 1537,
      2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0,   0};
  static const int s_dist_extra[32] = {0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                                       4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                                       9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
  static const mz_uint8 s_length_dezigzag[19] = {
      16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
  static const int s_min_table_sizes[3] = {257, 1, 4};

  tinfl_status status = TINFL_STATUS_FAILED;
  mz_uint32 num_bits, dist, counter, num_extra;
  tinfl_bit_buf_t bit_buf;
  const mz_uint8 *pIn_buf_cur = pIn_buf_next, *const pIn_buf_end =
                                                  pIn_buf_next + *pIn_buf_size;
  mz_uint8 *pOut_buf_cur = pOut_buf_next, *const pOut_buf_end =
                                              pOut_buf_next + *pOut_buf_size;
  size_t out_buf_size_mask =
             (decomp_flags & TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF)
                 ? (size_t)-1
                 : ((pOut_buf_next - pOut_buf_start) + *pOut_buf_size) - 1,
         dist_from_out_buf_start;

  // Ensure the output buffer's size is a power of 2, unless the output buffer
  // is large enough to hold the entire output file (in which case it doesn't
  // matter).
  if (((out_buf_size_mask + 1) & out_buf_size_mask) ||
      (pOut_buf_next < pOut_buf_start)) {
    *pIn_buf_size = *pOut_buf_size = 0;
    return TINFL_STATUS_BAD_PARAM;
  }

  num_bits = r->m_num_bits;
  bit_buf = r->m_bit_buf;
  dist = r->m_dist;
  counter = r->m_counter;
  num_extra = r->m_num_extra;
  dist_from_out_buf_start = r->m_dist_from_out_buf_start;
  TINFL_CR_BEGIN

  bit_buf = num_bits = dist = counter = num_extra = r->m_zhdr0 = r->m_zhdr1 = 0;
  r->m_z_adler32 = r->m_check_adler32 = 1;
  if (decomp_flags & TINFL_FLAG_PARSE_ZLIB_HEADER) {
    TINFL_GET_BYTE(1, r->m_zhdr0);
    TINFL_GET_BYTE(2, r->m_zhdr1);
    counter = (((r->m_zhdr0 * 256 + r->m_zhdr1) % 31 != 0) ||
               (r->m_zhdr1 & 32) || ((r->m_zhdr0 & 15) != 8));
    if (!(decomp_flags & TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF))
      counter |= (((1U << (8U + (r->m_zhdr0 >> 4))) > 32768U) ||
                  ((out_buf_size_mask + 1) <
                   (size_t)(1ULL << (8U + (r->m_zhdr0 >> 4)))));
    if (counter) {
      TINFL_CR_RETURN_FOREVER(36, TINFL_STATUS_FAILED);
    }
  }

  do {
    TINFL_GET_BITS(3, r->m_final, 3);
    r->m_type = r->m_final >> 1;
    if (r->m_type == 0) {
      TINFL_SKIP_BITS(5, num_bits & 7);
      for (counter = 0; counter < 4; ++counter) {
        if (num_bits)
          TINFL_GET_BITS(6, r->m_raw_header[counter], 8);
        else
          TINFL_GET_BYTE(7, r->m_raw_header[counter]);
      }
      if ((counter = (r->m_raw_header[0] | (r->m_raw_header[1] << 8))) !=
          (mz_uint)(0xFFFF ^
                    (r->m_raw_header[2] | (r->m_raw_header[3] << 8)))) {
        TINFL_CR_RETURN_FOREVER(39, TINFL_STATUS_FAILED);
      }
      while ((counter) && (num_bits)) {
        TINFL_GET_BITS(51, dist, 8);
        while (pOut_buf_cur >= pOut_buf_end) {
          TINFL_CR_RETURN(52, TINFL_STATUS_HAS_MORE_OUTPUT);
        }
        *pOut_buf_cur++ = (mz_uint8)dist;
        counter--;
      }
      while (counter) {
        size_t n;
        while (pOut_buf_cur >= pOut_buf_end) {
          TINFL_CR_RETURN(9, TINFL_STATUS_HAS_MORE_OUTPUT);
        }
        while (pIn_buf_cur >= pIn_buf_end) {
          if (decomp_flags & TINFL_FLAG_HAS_MORE_INPUT) {
            TINFL_CR_RETURN(38, TINFL_STATUS_NEEDS_MORE_INPUT);
          } else {
            TINFL_CR_RETURN_FOREVER(40, TINFL_STATUS_FAILED);
          }
        }
        n = MZ_MIN(MZ_MIN((size_t)(pOut_buf_end - pOut_buf_cur),
                          (size_t)(pIn_buf_end - pIn_buf_cur)),
                   counter);
        TINFL_MEMCPY(pOut_buf_cur, pIn_buf_cur, n);
        pIn_buf_cur += n;
        pOut_buf_cur += n;
        counter -= (mz_uint)n;
      }
    } else if (r->m_type == 3) {
      TINFL_CR_RETURN_FOREVER(10, TINFL_STATUS_FAILED);
    } else {
      if (r->m_type == 1) {
        mz_uint8 *p = r->m_tables[0].m_code_size;
        mz_uint i;
        r->m_table_sizes[0] = 288;
        r->m_table_sizes[1] = 32;
        TINFL_MEMSET(r->m_tables[1].m_code_size, 5, 32);
        for (i = 0; i <= 143; ++i) *p++ = 8;
        for (; i <= 255; ++i) *p++ = 9;
        for (; i <= 279; ++i) *p++ = 7;
        for (; i <= 287; ++i) *p++ = 8;
      } else {
        for (counter = 0; counter < 3; counter++) {
          TINFL_GET_BITS(11, r->m_table_sizes[counter], "\05\05\04"[counter]);
          r->m_table_sizes[counter] += s_min_table_sizes[counter];
        }
        MZ_CLEAR_OBJ(r->m_tables[2].m_code_size);
        for (counter = 0; counter < r->m_table_sizes[2]; counter++) {
          mz_uint s;
          TINFL_GET_BITS(14, s, 3);
          r->m_tables[2].m_code_size[s_length_dezigzag[counter]] = (mz_uint8)s;
        }
        r->m_table_sizes[2] = 19;
      }
      for (; (int)r->m_type >= 0; r->m_type--) {
        int tree_next, tree_cur;
        tinfl_huff_table *pTable;
        mz_uint i, j, used_syms, total, sym_index, next_code[17],
            total_syms[16];
        pTable = &r->m_tables[r->m_type];
        MZ_CLEAR_OBJ(total_syms);
        MZ_CLEAR_OBJ(pTable->m_look_up);
        MZ_CLEAR_OBJ(pTable->m_tree);
        for (i = 0; i < r->m_table_sizes[r->m_type]; ++i)
          total_syms[pTable->m_code_size[i]]++;
        used_syms = 0, total = 0;
        next_code[0] = next_code[1] = 0;
        for (i = 1; i <= 15; ++i) {
          used_syms += total_syms[i];
          next_code[i + 1] = (total = ((total + total_syms[i]) << 1));
        }
        if ((65536 != total) && (used_syms > 1)) {
          TINFL_CR_RETURN_FOREVER(35, TINFL_STATUS_FAILED);
        }
        for (tree_next = -1, sym_index = 0;
             sym_index < r->m_table_sizes[r->m_type]; ++sym_index) {
          mz_uint rev_code = 0, l, cur_code,
                  code_size = pTable->m_code_size[sym_index];
          if (!code_size) continue;
          cur_code = next_code[code_size]++;
          for (l = code_size; l > 0; l--, cur_code >>= 1)
            rev_code = (rev_code << 1) | (cur_code & 1);
          if (code_size <= TINFL_FAST_LOOKUP_BITS) {
            mz_int16 k = (mz_int16)((code_size << 9) | sym_index);
            while (rev_code < TINFL_FAST_LOOKUP_SIZE) {
              pTable->m_look_up[rev_code] = k;
              rev_code += (1 << code_size);
            }
            continue;
          }
          if (0 ==
              (tree_cur = pTable->m_look_up[rev_code &
                                            (TINFL_FAST_LOOKUP_SIZE - 1)])) {
            pTable->m_look_up[rev_code & (TINFL_FAST_LOOKUP_SIZE - 1)] =
                (mz_int16)tree_next;
            tree_cur = tree_next;
            tree_next -= 2;
          }
          rev_code >>= (TINFL_FAST_LOOKUP_BITS - 1);
          for (j = code_size; j > (TINFL_FAST_LOOKUP_BITS + 1); j--) {
            tree_cur -= ((rev_code >>= 1) & 1);
            if (!pTable->m_tree[-tree_cur - 1]) {
              pTable->m_tree[-tree_cur - 1] = (mz_int16)tree_next;
              tree_cur = tree_next;
              tree_next -= 2;
            } else
              tree_cur = pTable->m_tree[-tree_cur - 1];
          }
          tree_cur -= ((rev_code >>= 1) & 1);
          pTable->m_tree[-tree_cur - 1] = (mz_int16)sym_index;
        }
        if (r->m_type == 2) {
          for (counter = 0;
               counter < (r->m_table_sizes[0] + r->m_table_sizes[1]);) {
            mz_uint s;
            TINFL_HUFF_DECODE(16, dist, &r->m_tables[2]);
            if (dist < 16) {
              r->m_len_codes[counter++] = (mz_uint8)dist;
              continue;
            }
            if ((dist == 16) && (!counter)) {
              TINFL_CR_RETURN_FOREVER(17, TINFL_STATUS_FAILED);
            }
            num_extra = "\02\03\07"[dist - 16];
            TINFL_GET_BITS(18, s, num_extra);
            s += "\03\03\013"[dist - 16];
            TINFL_MEMSET(r->m_len_codes + counter,
                         (dist == 16) ? r->m_len_codes[counter - 1] : 0, s);
            counter += s;
          }
          if ((r->m_table_sizes[0] + r->m_table_sizes[1]) != counter) {
            TINFL_CR_RETURN_FOREVER(21, TINFL_STATUS_FAILED);
          }
          TINFL_MEMCPY(r->m_tables[0].m_code_size, r->m_len_codes,
                       r->m_table_sizes[0]);
          TINFL_MEMCPY(r->m_tables[1].m_code_size,
                       r->m_len_codes + r->m_table_sizes[0],
                       r->m_table_sizes[1]);
        }
      }
      for (;;) {
        mz_uint8 *pSrc;
        for (;;) {
          if (((pIn_buf_end - pIn_buf_cur) < 4) ||
              ((pOut_buf_end - pOut_buf_cur) < 2)) {
            TINFL_HUFF_DECODE(23, counter, &r->m_tables[0]);
            if (counter >= 256) break;
            while (pOut_buf_cur >= pOut_buf_end) {
              TINFL_CR_RETURN(24, TINFL_STATUS_HAS_MORE_OUTPUT);
            }
            *pOut_buf_cur++ = (mz_uint8)counter;
          } else {
            int sym2;
            mz_uint code_len;
#if TINFL_USE_64BIT_BITBUF
            if (num_bits < 30) {
              bit_buf |=
                  (((tinfl_bit_buf_t)MZ_READ_LE32(pIn_buf_cur)) << num_bits);
              pIn_buf_cur += 4;
              num_bits += 32;
            }
#else
            if (num_bits < 15) {
              bit_buf |=
                  (((tinfl_bit_buf_t)MZ_READ_LE16(pIn_buf_cur)) << num_bits);
              pIn_buf_cur += 2;
              num_bits += 16;
            }
#endif
            if ((sym2 =
                     r->m_tables[0]
                         .m_look_up[bit_buf & (TINFL_FAST_LOOKUP_SIZE - 1)]) >=
                0)
              code_len = sym2 >> 9;
            else {
              code_len = TINFL_FAST_LOOKUP_BITS;
              do {
                sym2 = r->m_tables[0]
                           .m_tree[~sym2 + ((bit_buf >> code_len++) & 1)];
              } while (sym2 < 0);
            }
            counter = sym2;
            bit_buf >>= code_len;
            num_bits -= code_len;
            if (counter & 256) break;

#if !TINFL_USE_64BIT_BITBUF
            if (num_bits < 15) {
              bit_buf |=
                  (((tinfl_bit_buf_t)MZ_READ_LE16(pIn_buf_cur)) << num_bits);
              pIn_buf_cur += 2;
              num_bits += 16;
            }
#endif
            if ((sym2 =
                     r->m_tables[0]
                         .m_look_up[bit_buf & (TINFL_FAST_LOOKUP_SIZE - 1)]) >=
                0)
              code_len = sym2 >> 9;
            else {
              code_len = TINFL_FAST_LOOKUP_BITS;
              do {
                sym2 = r->m_tables[0]
                           .m_tree[~sym2 + ((bit_buf >> code_len++) & 1)];
              } while (sym2 < 0);
            }
            bit_buf >>= code_len;
            num_bits -= code_len;

            pOut_buf_cur[0] = (mz_uint8)counter;
            if (sym2 & 256) {
              pOut_buf_cur++;
              counter = sym2;
              break;
            }
            pOut_buf_cur[1] = (mz_uint8)sym2;
            pOut_buf_cur += 2;
          }
        }
        if ((counter &= 511) == 256) break;

        num_extra = s_length_extra[counter - 257];
        counter = s_length_base[counter - 257];
        if (num_extra) {
          mz_uint extra_bits;
          TINFL_GET_BITS(25, extra_bits, num_extra);
          counter += extra_bits;
        }

        TINFL_HUFF_DECODE(26, dist, &r->m_tables[1]);
        num_extra = s_dist_extra[dist];
        dist = s_dist_base[dist];
        if (num_extra) {
          mz_uint extra_bits;
          TINFL_GET_BITS(27, extra_bits, num_extra);
          dist += extra_bits;
        }

        dist_from_out_buf_start = pOut_buf_cur - pOut_buf_start;
        if ((dist > dist_from_out_buf_start) &&
            (decomp_flags & TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF)) {
          TINFL_CR_RETURN_FOREVER(37, TINFL_STATUS_FAILED);
        }

        pSrc = pOut_buf_start +
               ((dist_from_out_buf_start - dist) & out_buf_size_mask);

        if ((MZ_MAX(pOut_buf_cur, pSrc) + counter) > pOut_buf_end) {
          while (counter--) {
            while (pOut_buf_cur >= pOut_buf_end) {
              TINFL_CR_RETURN(53, TINFL_STATUS_HAS_MORE_OUTPUT);
            }
            *pOut_buf_cur++ =
                pOut_buf_start[(dist_from_out_buf_start++ - dist) &
                               out_buf_size_mask];
          }
          continue;
        }
#if MINIZ_USE_UNALIGNED_LOADS_AND_STORES
        else if ((counter >= 9) && (counter <= dist)) {
          const mz_uint8 *pSrc_end = pSrc + (counter & ~7);
          do {
            ((mz_uint32 *)pOut_buf_cur)[0] = ((const mz_uint32 *)pSrc)[0];
            ((mz_uint32 *)pOut_buf_cur)[1] = ((const mz_uint32 *)pSrc)[1];
            pOut_buf_cur += 8;
          } while ((pSrc += 8) < pSrc_end);
          if ((counter &= 7) < 3) {
            if (counter) {
              pOut_buf_cur[0] = pSrc[0];
              if (counter > 1) pOut_buf_cur[1] = pSrc[1];
              pOut_buf_cur += counter;
            }
            continue;
          }
        }
#endif
        do {
          pOut_buf_cur[0] = pSrc[0];
          pOut_buf_cur[1] = pSrc[1];
          pOut_buf_cur[2] = pSrc[2];
          pOut_buf_cur += 3;
          pSrc += 3;
        } while ((int)(counter -= 3) > 2);
        if ((int)counter > 0) {
          pOut_buf_cur[0] = pSrc[0];
          if ((int)counter > 1) pOut_buf_cur[1] = pSrc[1];
          pOut_buf_cur += counter;
        }
      }
    }
  } while (!(r->m_final & 1));
  if (decomp_flags & TINFL_FLAG_PARSE_ZLIB_HEADER) {
    TINFL_SKIP_BITS(32, num_bits & 7);
    for (counter = 0; counter < 4; ++counter) {
      mz_uint s;
      if (num_bits)
        TINFL_GET_BITS(41, s, 8);
      else
        TINFL_GET_BYTE(42, s);
      r->m_z_adler32 = (r->m_z_adler32 << 8) | s;
    }
  }
  TINFL_CR_RETURN_FOREVER(34, TINFL_STATUS_DONE);
  TINFL_CR_FINISH

common_exit:
  r->m_num_bits = num_bits;
  r->m_bit_buf = bit_buf;
  r->m_dist = dist;
  r->m_counter = counter;
  r->m_num_extra = num_extra;
  r->m_dist_from_out_buf_start = dist_from_out_buf_start;
  *pIn_buf_size = pIn_buf_cur - pIn_buf_next;
  *pOut_buf_size = pOut_buf_cur - pOut_buf_next;
  if ((decomp_flags &
       (TINFL_FLAG_PARSE_ZLIB_HEADER | TINFL_FLAG_COMPUTE_ADLER32)) &&
      (status >= 0)) {
    const mz_uint8 *ptr = pOut_buf_next;
    size_t buf_len = *pOut_buf_size;
    mz_uint32 i, s1 = r->m_check_adler32 & 0xffff,
                 s2 = r->m_check_adler32 >> 16;
    size_t block_len = buf_len % 5552;
    while (buf_len) {
      for (i = 0; i + 7 < block_len; i += 8, ptr += 8) {
        s1 += ptr[0], s2 += s1;
        s1 += ptr[1], s2 += s1;
        s1 += ptr[2], s2 += s1;
        s1 += ptr[3], s2 += s1;
        s1 += ptr[4], s2 += s1;
        s1 += ptr[5], s2 += s1;
        s1 += ptr[6], s2 += s1;
        s1 += ptr[7], s2 += s1;
      }
      for (; i < block_len; ++i) s1 += *ptr++, s2 += s1;
      s1 %= 65521U, s2 %= 65521U;
      buf_len -= block_len;
      block_len = 5552;
    }
    r->m_check_adler32 = (s2 << 16) + s1;
    if ((status == TINFL_STATUS_DONE) &&
        (decomp_flags & TINFL_FLAG_PARSE_ZLIB_HEADER) &&
        (r->m_check_adler32 != r->m_z_adler32))
      status = TINFL_STATUS_ADLER32_MISMATCH;
  }
  return status;
}

// Higher level helper functions.
void *tinfl_decompress_mem_to_heap(const void *pSrc_buf, size_t src_buf_len,
                                   size_t *pOut_len, int flags) {
  tinfl_decompressor decomp;
  void *pBuf = NULL, *pNew_buf;
  size_t src_buf_ofs = 0, out_buf_capacity = 0;
  *pOut_len = 0;
  tinfl_init(&decomp);
  for (;;) {
    size_t src_buf_size = src_buf_len - src_buf_ofs,
           dst_buf_size = out_buf_capacity - *pOut_len, new_out_buf_capacity;
    tinfl_status status = tinfl_decompress(
        &decomp, (const mz_uint8 *)pSrc_buf + src_buf_ofs, &src_buf_size,
        (mz_uint8 *)pBuf, pBuf ? (mz_uint8 *)pBuf + *pOut_len : NULL,
        &dst_buf_size,
        (flags & ~TINFL_FLAG_HAS_MORE_INPUT) |
            TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF);
    if ((status < 0) || (status == TINFL_STATUS_NEEDS_MORE_INPUT)) {
      MZ_FREE(pBuf);
      *pOut_len = 0;
      return NULL;
    }
    src_buf_ofs += src_buf_size;
    *pOut_len += dst_buf_size;
    if (status == TINFL_STATUS_DONE) break;
    new_out_buf_capacity = out_buf_capacity * 2;
    if (new_out_buf_capacity < 128) new_out_buf_capacity = 128;
    pNew_buf = MZ_REALLOC(pBuf, new_out_buf_capacity);
    if (!pNew_buf) {
      MZ_FREE(pBuf);
      *pOut_len = 0;
      return NULL;
    }
    pBuf = pNew_buf;
    out_buf_capacity = new_out_buf_capacity;
  }
  return pBuf;
}

size_t tinfl_decompress_mem_to_mem(void *pOut_buf, size_t out_buf_len,
                                   const void *pSrc_buf, size_t src_buf_len,
                                   int flags) {
  tinfl_decompressor decomp;
  tinfl_status status;
  tinfl_init(&decomp);
  status =
      tinfl_decompress(&decomp, (const mz_uint8 *)pSrc_buf, &src_buf_len,
                       (mz_uint8 *)pOut_buf, (mz_uint8 *)pOut_buf, &out_buf_len,
                       (flags & ~TINFL_FLAG_HAS_MORE_INPUT) |
                           TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF);
  return (status != TINFL_STATUS_DONE) ? TINFL_DECOMPRESS_MEM_TO_MEM_FAILED
                                       : out_buf_len;
}

int tinfl_decompress_mem_to_callback(const void *pIn_buf, size_t *pIn_buf_size,
                                     tinfl_put_buf_func_ptr pPut_buf_func,
                                     void *pPut_buf_user, int flags) {
  int result = 0;
  tinfl_decompressor decomp;
  mz_uint8 *pDict = (mz_uint8 *)MZ_MALLOC(TINFL_LZ_DICT_SIZE);
  size_t in_buf_ofs = 0, dict_ofs = 0;
  if (!pDict) return TINFL_STATUS_FAILED;
  tinfl_init(&decomp);
  for (;;) {
    size_t in_buf_size = *pIn_buf_size - in_buf_ofs,
           dst_buf_size = TINFL_LZ_DICT_SIZE - dict_ofs;
    tinfl_status status =
        tinfl_decompress(&decomp, (const mz_uint8 *)pIn_buf + in_buf_ofs,
                         &in_buf_size, pDict, pDict + dict_ofs, &dst_buf_size,
                         (flags & ~(TINFL_FLAG_HAS_MORE_INPUT |
                                    TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF)));
    in_buf_ofs += in_buf_size;
    if ((dst_buf_size) &&
        (!(*pPut_buf_func)(pDict + dict_ofs, (int)dst_buf_size, pPut_buf_user)))
      break;
    if (status != TINFL_STATUS_HAS_MORE_OUTPUT) {
      result = (status == TINFL_STATUS_DONE);
      break;
    }
    dict_ofs = (dict_ofs + dst_buf_size) & (TINFL_LZ_DICT_SIZE - 1);
  }
  MZ_FREE(pDict);
  *pIn_buf_size = in_buf_ofs;
  return result;
}

// ------------------- Low-level Compression (independent from all decompression
// API's)

// Purposely making these tables static for faster init and thread safety.
static const mz_uint16 s_tdefl_len_sym[256] = {
    257, 258, 259, 260, 261, 262, 263, 264, 265, 265, 266, 266, 267, 267, 268,
    268, 269, 269, 269, 269, 270, 270, 270, 270, 271, 271, 271, 271, 272, 272,
    272, 272, 273, 273, 273, 273, 273, 273, 273, 273, 274, 274, 274, 274, 274,
    274, 274, 274, 275, 275, 275, 275, 275, 275, 275, 275, 276, 276, 276, 276,
    276, 276, 276, 276, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277,
    277, 277, 277, 277, 277, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278,
    278, 278, 278, 278, 278, 278, 279, 279, 279, 279, 279, 279, 279, 279, 279,
    279, 279, 279, 279, 279, 279, 279, 280, 280, 280, 280, 280, 280, 280, 280,
    280, 280, 280, 280, 280, 280, 280, 280, 281, 281, 281, 281, 281, 281, 281,
    281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281,
    281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 282, 282, 282, 282, 282,
    282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282,
    282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 283, 283, 283,
    283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283,
    283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 284,
    284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284,
    284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284,
    285};

static const mz_uint8 s_tdefl_len_extra[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0};

static const mz_uint8 s_tdefl_small_dist_sym[512] = {
    0,  1,  2,  3,  4,  4,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,
    8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17};

static const mz_uint8 s_tdefl_small_dist_extra[512] = {
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

static const mz_uint8 s_tdefl_large_dist_sym[128] = {
    0,  0,  18, 19, 20, 20, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24,
    24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26,
    26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27,
    27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
    28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
    28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29};

static const mz_uint8 s_tdefl_large_dist_extra[128] = {
    0,  0,  8,  8,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13};

// Radix sorts tdefl_sym_freq[] array by 16-bit key m_key. Returns ptr to sorted
// values.
typedef struct {
  mz_uint16 m_key, m_sym_index;
} tdefl_sym_freq;
static tdefl_sym_freq *tdefl_radix_sort_syms(mz_uint num_syms,
                                             tdefl_sym_freq *pSyms0,
                                             tdefl_sym_freq *pSyms1) {
  mz_uint32 total_passes = 2, pass_shift, pass, i, hist[256 * 2];
  tdefl_sym_freq *pCur_syms = pSyms0, *pNew_syms = pSyms1;
  MZ_CLEAR_OBJ(hist);
  for (i = 0; i < num_syms; i++) {
    mz_uint freq = pSyms0[i].m_key;
    hist[freq & 0xFF]++;
    hist[256 + ((freq >> 8) & 0xFF)]++;
  }
  while ((total_passes > 1) && (num_syms == hist[(total_passes - 1) * 256]))
    total_passes--;
  for (pass_shift = 0, pass = 0; pass < total_passes; pass++, pass_shift += 8) {
    const mz_uint32 *pHist = &hist[pass << 8];
    mz_uint offsets[256], cur_ofs = 0;
    for (i = 0; i < 256; i++) {
      offsets[i] = cur_ofs;
      cur_ofs += pHist[i];
    }
    for (i = 0; i < num_syms; i++)
      pNew_syms[offsets[(pCur_syms[i].m_key >> pass_shift) & 0xFF]++] =
          pCur_syms[i];
    {
      tdefl_sym_freq *t = pCur_syms;
      pCur_syms = pNew_syms;
      pNew_syms = t;
    }
  }
  return pCur_syms;
}

// tdefl_calculate_minimum_redundancy() originally written by: Alistair Moffat,
// alistair@cs.mu.oz.au, Jyrki Katajainen, jyrki@diku.dk, November 1996.
static void tdefl_calculate_minimum_redundancy(tdefl_sym_freq *A, int n) {
  int root, leaf, next, avbl, used, dpth;
  if (n == 0)
    return;
  else if (n == 1) {
    A[0].m_key = 1;
    return;
  }
  A[0].m_key += A[1].m_key;
  root = 0;
  leaf = 2;
  for (next = 1; next < n - 1; next++) {
    if (leaf >= n || A[root].m_key < A[leaf].m_key) {
      A[next].m_key = A[root].m_key;
      A[root++].m_key = (mz_uint16)next;
    } else
      A[next].m_key = A[leaf++].m_key;
    if (leaf >= n || (root < next && A[root].m_key < A[leaf].m_key)) {
      A[next].m_key = (mz_uint16)(A[next].m_key + A[root].m_key);
      A[root++].m_key = (mz_uint16)next;
    } else
      A[next].m_key = (mz_uint16)(A[next].m_key + A[leaf++].m_key);
  }
  A[n - 2].m_key = 0;
  for (next = n - 3; next >= 0; next--)
    A[next].m_key = A[A[next].m_key].m_key + 1;
  avbl = 1;
  used = dpth = 0;
  root = n - 2;
  next = n - 1;
  while (avbl > 0) {
    while (root >= 0 && (int)A[root].m_key == dpth) {
      used++;
      root--;
    }
    while (avbl > used) {
      A[next--].m_key = (mz_uint16)(dpth);
      avbl--;
    }
    avbl = 2 * used;
    dpth++;
    used = 0;
  }
}

// Limits canonical Huffman code table's max code size.
enum { TDEFL_MAX_SUPPORTED_HUFF_CODESIZE = 32 };
static void tdefl_huffman_enforce_max_code_size(int *pNum_codes,
                                                int code_list_len,
                                                int max_code_size) {
  int i;
  mz_uint32 total = 0;
  if (code_list_len <= 1) return;
  for (i = max_code_size + 1; i <= TDEFL_MAX_SUPPORTED_HUFF_CODESIZE; i++)
    pNum_codes[max_code_size] += pNum_codes[i];
  for (i = max_code_size; i > 0; i--)
    total += (((mz_uint32)pNum_codes[i]) << (max_code_size - i));
  while (total != (1UL << max_code_size)) {
    pNum_codes[max_code_size]--;
    for (i = max_code_size - 1; i > 0; i--)
      if (pNum_codes[i]) {
        pNum_codes[i]--;
        pNum_codes[i + 1] += 2;
        break;
      }
    total--;
  }
}

static void tdefl_optimize_huffman_table(tdefl_compressor *d, int table_num,
                                         int table_len, int code_size_limit,
                                         int static_table) {
  int i, j, l, num_codes[1 + TDEFL_MAX_SUPPORTED_HUFF_CODESIZE];
  mz_uint next_code[TDEFL_MAX_SUPPORTED_HUFF_CODESIZE + 1];
  MZ_CLEAR_OBJ(num_codes);
  if (static_table) {
    for (i = 0; i < table_len; i++)
      num_codes[d->m_huff_code_sizes[table_num][i]]++;
  } else {
    tdefl_sym_freq syms0[TDEFL_MAX_HUFF_SYMBOLS], syms1[TDEFL_MAX_HUFF_SYMBOLS],
        *pSyms;
    int num_used_syms = 0;
    const mz_uint16 *pSym_count = &d->m_huff_count[table_num][0];
    for (i = 0; i < table_len; i++)
      if (pSym_count[i]) {
        syms0[num_used_syms].m_key = (mz_uint16)pSym_count[i];
        syms0[num_used_syms++].m_sym_index = (mz_uint16)i;
      }

    pSyms = tdefl_radix_sort_syms(num_used_syms, syms0, syms1);
    tdefl_calculate_minimum_redundancy(pSyms, num_used_syms);

    for (i = 0; i < num_used_syms; i++) num_codes[pSyms[i].m_key]++;

    tdefl_huffman_enforce_max_code_size(num_codes, num_used_syms,
                                        code_size_limit);

    MZ_CLEAR_OBJ(d->m_huff_code_sizes[table_num]);
    MZ_CLEAR_OBJ(d->m_huff_codes[table_num]);
    for (i = 1, j = num_used_syms; i <= code_size_limit; i++)
      for (l = num_codes[i]; l > 0; l--)
        d->m_huff_code_sizes[table_num][pSyms[--j].m_sym_index] = (mz_uint8)(i);
  }

  next_code[1] = 0;
  for (j = 0, i = 2; i <= code_size_limit; i++)
    next_code[i] = j = ((j + num_codes[i - 1]) << 1);

  for (i = 0; i < table_len; i++) {
    mz_uint rev_code = 0, code, code_size;
    if ((code_size = d->m_huff_code_sizes[table_num][i]) == 0) continue;
    code = next_code[code_size]++;
    for (l = code_size; l > 0; l--, code >>= 1)
      rev_code = (rev_code << 1) | (code & 1);
    d->m_huff_codes[table_num][i] = (mz_uint16)rev_code;
  }
}

#define TDEFL_PUT_BITS(b, l)                               \
  do {                                                     \
    mz_uint bits = b;                                      \
    mz_uint len = l;                                       \
    MZ_ASSERT(bits <= ((1U << len) - 1U));                 \
    d->m_bit_buffer |= (bits << d->m_bits_in);             \
    d->m_bits_in += len;                                   \
    while (d->m_bits_in >= 8) {                            \
      if (d->m_pOutput_buf < d->m_pOutput_buf_end)         \
        *d->m_pOutput_buf++ = (mz_uint8)(d->m_bit_buffer); \
      d->m_bit_buffer >>= 8;                               \
      d->m_bits_in -= 8;                                   \
    }                                                      \
  }                                                        \
  MZ_MACRO_END

#define TDEFL_RLE_PREV_CODE_SIZE()                                        \
  {                                                                       \
    if (rle_repeat_count) {                                               \
      if (rle_repeat_count < 3) {                                         \
        d->m_huff_count[2][prev_code_size] = (mz_uint16)(                 \
            d->m_huff_count[2][prev_code_size] + rle_repeat_count);       \
        while (rle_repeat_count--)                                        \
          packed_code_sizes[num_packed_code_sizes++] = prev_code_size;    \
      } else {                                                            \
        d->m_huff_count[2][16] = (mz_uint16)(d->m_huff_count[2][16] + 1); \
        packed_code_sizes[num_packed_code_sizes++] = 16;                  \
        packed_code_sizes[num_packed_code_sizes++] =                      \
            (mz_uint8)(rle_repeat_count - 3);                             \
      }                                                                   \
      rle_repeat_count = 0;                                               \
    }                                                                     \
  }

#define TDEFL_RLE_ZERO_CODE_SIZE()                                            \
  {                                                                           \
    if (rle_z_count) {                                                        \
      if (rle_z_count < 3) {                                                  \
        d->m_huff_count[2][0] =                                               \
            (mz_uint16)(d->m_huff_count[2][0] + rle_z_count);                 \
        while (rle_z_count--) packed_code_sizes[num_packed_code_sizes++] = 0; \
      } else if (rle_z_count <= 10) {                                         \
        d->m_huff_count[2][17] = (mz_uint16)(d->m_huff_count[2][17] + 1);     \
        packed_code_sizes[num_packed_code_sizes++] = 17;                      \
        packed_code_sizes[num_packed_code_sizes++] =                          \
            (mz_uint8)(rle_z_count - 3);                                      \
      } else {                                                                \
        d->m_huff_count[2][18] = (mz_uint16)(d->m_huff_count[2][18] + 1);     \
        packed_code_sizes[num_packed_code_sizes++] = 18;                      \
        packed_code_sizes[num_packed_code_sizes++] =                          \
            (mz_uint8)(rle_z_count - 11);                                     \
      }                                                                       \
      rle_z_count = 0;                                                        \
    }                                                                         \
  }

static mz_uint8 s_tdefl_packed_code_size_syms_swizzle[] = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

static void tdefl_start_dynamic_block(tdefl_compressor *d) {
  int num_lit_codes, num_dist_codes, num_bit_lengths;
  mz_uint i, total_code_sizes_to_pack, num_packed_code_sizes, rle_z_count,
      rle_repeat_count, packed_code_sizes_index;
  mz_uint8
      code_sizes_to_pack[TDEFL_MAX_HUFF_SYMBOLS_0 + TDEFL_MAX_HUFF_SYMBOLS_1],
      packed_code_sizes[TDEFL_MAX_HUFF_SYMBOLS_0 + TDEFL_MAX_HUFF_SYMBOLS_1],
      prev_code_size = 0xFF;

  d->m_huff_count[0][256] = 1;

  tdefl_optimize_huffman_table(d, 0, TDEFL_MAX_HUFF_SYMBOLS_0, 15, MZ_FALSE);
  tdefl_optimize_huffman_table(d, 1, TDEFL_MAX_HUFF_SYMBOLS_1, 15, MZ_FALSE);

  for (num_lit_codes = 286; num_lit_codes > 257; num_lit_codes--)
    if (d->m_huff_code_sizes[0][num_lit_codes - 1]) break;
  for (num_dist_codes = 30; num_dist_codes > 1; num_dist_codes--)
    if (d->m_huff_code_sizes[1][num_dist_codes - 1]) break;

  memcpy(code_sizes_to_pack, &d->m_huff_code_sizes[0][0], num_lit_codes);
  memcpy(code_sizes_to_pack + num_lit_codes, &d->m_huff_code_sizes[1][0],
         num_dist_codes);
  total_code_sizes_to_pack = num_lit_codes + num_dist_codes;
  num_packed_code_sizes = 0;
  rle_z_count = 0;
  rle_repeat_count = 0;

  memset(&d->m_huff_count[2][0], 0,
         sizeof(d->m_huff_count[2][0]) * TDEFL_MAX_HUFF_SYMBOLS_2);
  for (i = 0; i < total_code_sizes_to_pack; i++) {
    mz_uint8 code_size = code_sizes_to_pack[i];
    if (!code_size) {
      TDEFL_RLE_PREV_CODE_SIZE();
      if (++rle_z_count == 138) {
        TDEFL_RLE_ZERO_CODE_SIZE();
      }
    } else {
      TDEFL_RLE_ZERO_CODE_SIZE();
      if (code_size != prev_code_size) {
        TDEFL_RLE_PREV_CODE_SIZE();
        d->m_huff_count[2][code_size] =
            (mz_uint16)(d->m_huff_count[2][code_size] + 1);
        packed_code_sizes[num_packed_code_sizes++] = code_size;
      } else if (++rle_repeat_count == 6) {
        TDEFL_RLE_PREV_CODE_SIZE();
      }
    }
    prev_code_size = code_size;
  }
  if (rle_repeat_count) {
    TDEFL_RLE_PREV_CODE_SIZE();
  } else {
    TDEFL_RLE_ZERO_CODE_SIZE();
  }

  tdefl_optimize_huffman_table(d, 2, TDEFL_MAX_HUFF_SYMBOLS_2, 7, MZ_FALSE);

  TDEFL_PUT_BITS(2, 2);

  TDEFL_PUT_BITS(num_lit_codes - 257, 5);
  TDEFL_PUT_BITS(num_dist_codes - 1, 5);

  for (num_bit_lengths = 18; num_bit_lengths >= 0; num_bit_lengths--)
    if (d->m_huff_code_sizes
            [2][s_tdefl_packed_code_size_syms_swizzle[num_bit_lengths]])
      break;
  num_bit_lengths = MZ_MAX(4, (num_bit_lengths + 1));
  TDEFL_PUT_BITS(num_bit_lengths - 4, 4);
  for (i = 0; (int)i < num_bit_lengths; i++)
    TDEFL_PUT_BITS(
        d->m_huff_code_sizes[2][s_tdefl_packed_code_size_syms_swizzle[i]], 3);

  for (packed_code_sizes_index = 0;
       packed_code_sizes_index < num_packed_code_sizes;) {
    mz_uint code = packed_code_sizes[packed_code_sizes_index++];
    MZ_ASSERT(code < TDEFL_MAX_HUFF_SYMBOLS_2);
    TDEFL_PUT_BITS(d->m_huff_codes[2][code], d->m_huff_code_sizes[2][code]);
    if (code >= 16)
      TDEFL_PUT_BITS(packed_code_sizes[packed_code_sizes_index++],
                     "\02\03\07"[code - 16]);
  }
}

static void tdefl_start_static_block(tdefl_compressor *d) {
  mz_uint i;
  mz_uint8 *p = &d->m_huff_code_sizes[0][0];

  for (i = 0; i <= 143; ++i) *p++ = 8;
  for (; i <= 255; ++i) *p++ = 9;
  for (; i <= 279; ++i) *p++ = 7;
  for (; i <= 287; ++i) *p++ = 8;

  memset(d->m_huff_code_sizes[1], 5, 32);

  tdefl_optimize_huffman_table(d, 0, 288, 15, MZ_TRUE);
  tdefl_optimize_huffman_table(d, 1, 32, 15, MZ_TRUE);

  TDEFL_PUT_BITS(1, 2);
}

static const mz_uint mz_bitmasks[17] = {
    0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF,
    0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF};

#if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN && \
    MINIZ_HAS_64BIT_REGISTERS
static mz_bool tdefl_compress_lz_codes(tdefl_compressor *d) {
  mz_uint flags;
  mz_uint8 *pLZ_codes;
  mz_uint8 *pOutput_buf = d->m_pOutput_buf;
  mz_uint8 *pLZ_code_buf_end = d->m_pLZ_code_buf;
  mz_uint64 bit_buffer = d->m_bit_buffer;
  mz_uint bits_in = d->m_bits_in;

#define TDEFL_PUT_BITS_FAST(b, l)                \
  {                                              \
    bit_buffer |= (((mz_uint64)(b)) << bits_in); \
    bits_in += (l);                              \
  }

  flags = 1;
  for (pLZ_codes = d->m_lz_code_buf; pLZ_codes < pLZ_code_buf_end;
       flags >>= 1) {
    if (flags == 1) flags = *pLZ_codes++ | 0x100;

    if (flags & 1) {
      mz_uint s0, s1, n0, n1, sym, num_extra_bits;
      mz_uint match_len = pLZ_codes[0],
              match_dist = *(const mz_uint16 *)(pLZ_codes + 1);
      pLZ_codes += 3;

      MZ_ASSERT(d->m_huff_code_sizes[0][s_tdefl_len_sym[match_len]]);
      TDEFL_PUT_BITS_FAST(d->m_huff_codes[0][s_tdefl_len_sym[match_len]],
                          d->m_huff_code_sizes[0][s_tdefl_len_sym[match_len]]);
      TDEFL_PUT_BITS_FAST(match_len & mz_bitmasks[s_tdefl_len_extra[match_len]],
                          s_tdefl_len_extra[match_len]);

      // This sequence coaxes MSVC into using cmov's vs. jmp's.
      s0 = s_tdefl_small_dist_sym[match_dist & 511];
      n0 = s_tdefl_small_dist_extra[match_dist & 511];
      s1 = s_tdefl_large_dist_sym[match_dist >> 8];
      n1 = s_tdefl_large_dist_extra[match_dist >> 8];
      sym = (match_dist < 512) ? s0 : s1;
      num_extra_bits = (match_dist < 512) ? n0 : n1;

      MZ_ASSERT(d->m_huff_code_sizes[1][sym]);
      TDEFL_PUT_BITS_FAST(d->m_huff_codes[1][sym],
                          d->m_huff_code_sizes[1][sym]);
      TDEFL_PUT_BITS_FAST(match_dist & mz_bitmasks[num_extra_bits],
                          num_extra_bits);
    } else {
      mz_uint lit = *pLZ_codes++;
      MZ_ASSERT(d->m_huff_code_sizes[0][lit]);
      TDEFL_PUT_BITS_FAST(d->m_huff_codes[0][lit],
                          d->m_huff_code_sizes[0][lit]);

      if (((flags & 2) == 0) && (pLZ_codes < pLZ_code_buf_end)) {
        flags >>= 1;
        lit = *pLZ_codes++;
        MZ_ASSERT(d->m_huff_code_sizes[0][lit]);
        TDEFL_PUT_BITS_FAST(d->m_huff_codes[0][lit],
                            d->m_huff_code_sizes[0][lit]);

        if (((flags & 2) == 0) && (pLZ_codes < pLZ_code_buf_end)) {
          flags >>= 1;
          lit = *pLZ_codes++;
          MZ_ASSERT(d->m_huff_code_sizes[0][lit]);
          TDEFL_PUT_BITS_FAST(d->m_huff_codes[0][lit],
                              d->m_huff_code_sizes[0][lit]);
        }
      }
    }

    if (pOutput_buf >= d->m_pOutput_buf_end) return MZ_FALSE;

    *(mz_uint64 *)pOutput_buf = bit_buffer;
    pOutput_buf += (bits_in >> 3);
    bit_buffer >>= (bits_in & ~7);
    bits_in &= 7;
  }

#undef TDEFL_PUT_BITS_FAST

  d->m_pOutput_buf = pOutput_buf;
  d->m_bits_in = 0;
  d->m_bit_buffer = 0;

  while (bits_in) {
    mz_uint32 n = MZ_MIN(bits_in, 16);
    TDEFL_PUT_BITS((mz_uint)bit_buffer & mz_bitmasks[n], n);
    bit_buffer >>= n;
    bits_in -= n;
  }

  TDEFL_PUT_BITS(d->m_huff_codes[0][256], d->m_huff_code_sizes[0][256]);

  return (d->m_pOutput_buf < d->m_pOutput_buf_end);
}
#else
static mz_bool tdefl_compress_lz_codes(tdefl_compressor *d) {
  mz_uint flags;
  mz_uint8 *pLZ_codes;

  flags = 1;
  for (pLZ_codes = d->m_lz_code_buf; pLZ_codes < d->m_pLZ_code_buf;
       flags >>= 1) {
    if (flags == 1) flags = *pLZ_codes++ | 0x100;
    if (flags & 1) {
      mz_uint sym, num_extra_bits;
      mz_uint match_len = pLZ_codes[0],
              match_dist = (pLZ_codes[1] | (pLZ_codes[2] << 8));
      pLZ_codes += 3;

      MZ_ASSERT(d->m_huff_code_sizes[0][s_tdefl_len_sym[match_len]]);
      TDEFL_PUT_BITS(d->m_huff_codes[0][s_tdefl_len_sym[match_len]],
                     d->m_huff_code_sizes[0][s_tdefl_len_sym[match_len]]);
      TDEFL_PUT_BITS(match_len & mz_bitmasks[s_tdefl_len_extra[match_len]],
                     s_tdefl_len_extra[match_len]);

      if (match_dist < 512) {
        sym = s_tdefl_small_dist_sym[match_dist];
        num_extra_bits = s_tdefl_small_dist_extra[match_dist];
      } else {
        sym = s_tdefl_large_dist_sym[match_dist >> 8];
        num_extra_bits = s_tdefl_large_dist_extra[match_dist >> 8];
      }
      MZ_ASSERT(d->m_huff_code_sizes[1][sym]);
      TDEFL_PUT_BITS(d->m_huff_codes[1][sym], d->m_huff_code_sizes[1][sym]);
      TDEFL_PUT_BITS(match_dist & mz_bitmasks[num_extra_bits], num_extra_bits);
    } else {
      mz_uint lit = *pLZ_codes++;
      MZ_ASSERT(d->m_huff_code_sizes[0][lit]);
      TDEFL_PUT_BITS(d->m_huff_codes[0][lit], d->m_huff_code_sizes[0][lit]);
    }
  }

  TDEFL_PUT_BITS(d->m_huff_codes[0][256], d->m_huff_code_sizes[0][256]);

  return (d->m_pOutput_buf < d->m_pOutput_buf_end);
}
#endif  // MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN &&
        // MINIZ_HAS_64BIT_REGISTERS

static mz_bool tdefl_compress_block(tdefl_compressor *d, mz_bool static_block) {
  if (static_block)
    tdefl_start_static_block(d);
  else
    tdefl_start_dynamic_block(d);
  return tdefl_compress_lz_codes(d);
}

static int tdefl_flush_block(tdefl_compressor *d, int flush) {
  mz_uint saved_bit_buf, saved_bits_in;
  mz_uint8 *pSaved_output_buf;
  mz_bool comp_block_succeeded = MZ_FALSE;
  int n, use_raw_block =
             ((d->m_flags & TDEFL_FORCE_ALL_RAW_BLOCKS) != 0) &&
             (d->m_lookahead_pos - d->m_lz_code_buf_dict_pos) <= d->m_dict_size;
  mz_uint8 *pOutput_buf_start =
      ((d->m_pPut_buf_func == NULL) &&
       ((*d->m_pOut_buf_size - d->m_out_buf_ofs) >= TDEFL_OUT_BUF_SIZE))
          ? ((mz_uint8 *)d->m_pOut_buf + d->m_out_buf_ofs)
          : d->m_output_buf;

  d->m_pOutput_buf = pOutput_buf_start;
  d->m_pOutput_buf_end = d->m_pOutput_buf + TDEFL_OUT_BUF_SIZE - 16;

  MZ_ASSERT(!d->m_output_flush_remaining);
  d->m_output_flush_ofs = 0;
  d->m_output_flush_remaining = 0;

  *d->m_pLZ_flags = (mz_uint8)(*d->m_pLZ_flags >> d->m_num_flags_left);
  d->m_pLZ_code_buf -= (d->m_num_flags_left == 8);

  if ((d->m_flags & TDEFL_WRITE_ZLIB_HEADER) && (!d->m_block_index)) {
    TDEFL_PUT_BITS(0x78, 8);
    TDEFL_PUT_BITS(0x01, 8);
  }

  TDEFL_PUT_BITS(flush == TDEFL_FINISH, 1);

  pSaved_output_buf = d->m_pOutput_buf;
  saved_bit_buf = d->m_bit_buffer;
  saved_bits_in = d->m_bits_in;

  if (!use_raw_block)
    comp_block_succeeded =
        tdefl_compress_block(d, (d->m_flags & TDEFL_FORCE_ALL_STATIC_BLOCKS) ||
                                    (d->m_total_lz_bytes < 48));

  // If the block gets expanded, forget the current contents of the output
  // buffer and send a raw block instead.
  if (((use_raw_block) ||
       ((d->m_total_lz_bytes) && ((d->m_pOutput_buf - pSaved_output_buf + 1U) >=
                                  d->m_total_lz_bytes))) &&
      ((d->m_lookahead_pos - d->m_lz_code_buf_dict_pos) <= d->m_dict_size)) {
    mz_uint i;
    d->m_pOutput_buf = pSaved_output_buf;
    d->m_bit_buffer = saved_bit_buf, d->m_bits_in = saved_bits_in;
    TDEFL_PUT_BITS(0, 2);
    if (d->m_bits_in) {
      TDEFL_PUT_BITS(0, 8 - d->m_bits_in);
    }
    for (i = 2; i; --i, d->m_total_lz_bytes ^= 0xFFFF) {
      TDEFL_PUT_BITS(d->m_total_lz_bytes & 0xFFFF, 16);
    }
    for (i = 0; i < d->m_total_lz_bytes; ++i) {
      TDEFL_PUT_BITS(
          d->m_dict[(d->m_lz_code_buf_dict_pos + i) & TDEFL_LZ_DICT_SIZE_MASK],
          8);
    }
  }
  // Check for the extremely unlikely (if not impossible) case of the compressed
  // block not fitting into the output buffer when using dynamic codes.
  else if (!comp_block_succeeded) {
    d->m_pOutput_buf = pSaved_output_buf;
    d->m_bit_buffer = saved_bit_buf, d->m_bits_in = saved_bits_in;
    tdefl_compress_block(d, MZ_TRUE);
  }

  if (flush) {
    if (flush == TDEFL_FINISH) {
      if (d->m_bits_in) {
        TDEFL_PUT_BITS(0, 8 - d->m_bits_in);
      }
      if (d->m_flags & TDEFL_WRITE_ZLIB_HEADER) {
        mz_uint i, a = d->m_adler32;
        for (i = 0; i < 4; i++) {
          TDEFL_PUT_BITS((a >> 24) & 0xFF, 8);
          a <<= 8;
        }
      }
    } else {
      mz_uint i, z = 0;
      TDEFL_PUT_BITS(0, 3);
      if (d->m_bits_in) {
        TDEFL_PUT_BITS(0, 8 - d->m_bits_in);
      }
      for (i = 2; i; --i, z ^= 0xFFFF) {
        TDEFL_PUT_BITS(z & 0xFFFF, 16);
      }
    }
  }

  MZ_ASSERT(d->m_pOutput_buf < d->m_pOutput_buf_end);

  memset(&d->m_huff_count[0][0], 0,
         sizeof(d->m_huff_count[0][0]) * TDEFL_MAX_HUFF_SYMBOLS_0);
  memset(&d->m_huff_count[1][0], 0,
         sizeof(d->m_huff_count[1][0]) * TDEFL_MAX_HUFF_SYMBOLS_1);

  d->m_pLZ_code_buf = d->m_lz_code_buf + 1;
  d->m_pLZ_flags = d->m_lz_code_buf;
  d->m_num_flags_left = 8;
  d->m_lz_code_buf_dict_pos += d->m_total_lz_bytes;
  d->m_total_lz_bytes = 0;
  d->m_block_index++;

  if ((n = (int)(d->m_pOutput_buf - pOutput_buf_start)) != 0) {
    if (d->m_pPut_buf_func) {
      *d->m_pIn_buf_size = d->m_pSrc - (const mz_uint8 *)d->m_pIn_buf;
      if (!(*d->m_pPut_buf_func)(d->m_output_buf, n, d->m_pPut_buf_user))
        return (d->m_prev_return_status = TDEFL_STATUS_PUT_BUF_FAILED);
    } else if (pOutput_buf_start == d->m_output_buf) {
      int bytes_to_copy = (int)MZ_MIN(
          (size_t)n, (size_t)(*d->m_pOut_buf_size - d->m_out_buf_ofs));
      memcpy((mz_uint8 *)d->m_pOut_buf + d->m_out_buf_ofs, d->m_output_buf,
             bytes_to_copy);
      d->m_out_buf_ofs += bytes_to_copy;
      if ((n -= bytes_to_copy) != 0) {
        d->m_output_flush_ofs = bytes_to_copy;
        d->m_output_flush_remaining = n;
      }
    } else {
      d->m_out_buf_ofs += n;
    }
  }

  return d->m_output_flush_remaining;
}

#if MINIZ_USE_UNALIGNED_LOADS_AND_STORES
#define TDEFL_READ_UNALIGNED_WORD(p) *(const mz_uint16 *)(p)
static MZ_FORCEINLINE void tdefl_find_match(
    tdefl_compressor *d, mz_uint lookahead_pos, mz_uint max_dist,
    mz_uint max_match_len, mz_uint *pMatch_dist, mz_uint *pMatch_len) {
  mz_uint dist, pos = lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK,
                match_len = *pMatch_len, probe_pos = pos, next_probe_pos,
                probe_len;
  mz_uint num_probes_left = d->m_max_probes[match_len >= 32];
  const mz_uint16 *s = (const mz_uint16 *)(d->m_dict + pos), *p, *q;
  mz_uint16 c01 = TDEFL_READ_UNALIGNED_WORD(&d->m_dict[pos + match_len - 1]),
            s01 = TDEFL_READ_UNALIGNED_WORD(s);
  MZ_ASSERT(max_match_len <= TDEFL_MAX_MATCH_LEN);
  if (max_match_len <= match_len) return;
  for (;;) {
    for (;;) {
      if (--num_probes_left == 0) return;
#define TDEFL_PROBE                                                            \
  next_probe_pos = d->m_next[probe_pos];                                       \
  if ((!next_probe_pos) ||                                                     \
      ((dist = (mz_uint16)(lookahead_pos - next_probe_pos)) > max_dist))       \
    return;                                                                    \
  probe_pos = next_probe_pos & TDEFL_LZ_DICT_SIZE_MASK;                        \
  if (TDEFL_READ_UNALIGNED_WORD(&d->m_dict[probe_pos + match_len - 1]) == c01) \
    break;
      TDEFL_PROBE;
      TDEFL_PROBE;
      TDEFL_PROBE;
    }
    if (!dist) break;
    q = (const mz_uint16 *)(d->m_dict + probe_pos);
    if (TDEFL_READ_UNALIGNED_WORD(q) != s01) continue;
    p = s;
    probe_len = 32;
    do {
    } while (
        (TDEFL_READ_UNALIGNED_WORD(++p) == TDEFL_READ_UNALIGNED_WORD(++q)) &&
        (TDEFL_READ_UNALIGNED_WORD(++p) == TDEFL_READ_UNALIGNED_WORD(++q)) &&
        (TDEFL_READ_UNALIGNED_WORD(++p) == TDEFL_READ_UNALIGNED_WORD(++q)) &&
        (TDEFL_READ_UNALIGNED_WORD(++p) == TDEFL_READ_UNALIGNED_WORD(++q)) &&
        (--probe_len > 0));
    if (!probe_len) {
      *pMatch_dist = dist;
      *pMatch_len = MZ_MIN(max_match_len, TDEFL_MAX_MATCH_LEN);
      break;
    } else if ((probe_len = ((mz_uint)(p - s) * 2) +
                            (mz_uint)(*(const mz_uint8 *)p ==
                                      *(const mz_uint8 *)q)) > match_len) {
      *pMatch_dist = dist;
      if ((*pMatch_len = match_len = MZ_MIN(max_match_len, probe_len)) ==
          max_match_len)
        break;
      c01 = TDEFL_READ_UNALIGNED_WORD(&d->m_dict[pos + match_len - 1]);
    }
  }
}
#else
static MZ_FORCEINLINE void tdefl_find_match(
    tdefl_compressor *d, mz_uint lookahead_pos, mz_uint max_dist,
    mz_uint max_match_len, mz_uint *pMatch_dist, mz_uint *pMatch_len) {
  mz_uint dist, pos = lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK,
                match_len = *pMatch_len, probe_pos = pos, next_probe_pos,
                probe_len;
  mz_uint num_probes_left = d->m_max_probes[match_len >= 32];
  const mz_uint8 *s = d->m_dict + pos, *p, *q;
  mz_uint8 c0 = d->m_dict[pos + match_len], c1 = d->m_dict[pos + match_len - 1];
  MZ_ASSERT(max_match_len <= TDEFL_MAX_MATCH_LEN);
  if (max_match_len <= match_len) return;
  for (;;) {
    for (;;) {
      if (--num_probes_left == 0) return;
#define TDEFL_PROBE                                                      \
  next_probe_pos = d->m_next[probe_pos];                                 \
  if ((!next_probe_pos) ||                                               \
      ((dist = (mz_uint16)(lookahead_pos - next_probe_pos)) > max_dist)) \
    return;                                                              \
  probe_pos = next_probe_pos & TDEFL_LZ_DICT_SIZE_MASK;                  \
  if ((d->m_dict[probe_pos + match_len] == c0) &&                        \
      (d->m_dict[probe_pos + match_len - 1] == c1))                      \
    break;
      TDEFL_PROBE;
      TDEFL_PROBE;
      TDEFL_PROBE;
    }
    if (!dist) break;
    p = s;
    q = d->m_dict + probe_pos;
    for (probe_len = 0; probe_len < max_match_len; probe_len++)
      if (*p++ != *q++) break;
    if (probe_len > match_len) {
      *pMatch_dist = dist;
      if ((*pMatch_len = match_len = probe_len) == max_match_len) return;
      c0 = d->m_dict[pos + match_len];
      c1 = d->m_dict[pos + match_len - 1];
    }
  }
}
#endif  // #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES

#if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
static mz_bool tdefl_compress_fast(tdefl_compressor *d) {
  // Faster, minimally featured LZRW1-style match+parse loop with better
  // register utilization. Intended for applications where raw throughput is
  // valued more highly than ratio.
  mz_uint lookahead_pos = d->m_lookahead_pos,
          lookahead_size = d->m_lookahead_size, dict_size = d->m_dict_size,
          total_lz_bytes = d->m_total_lz_bytes,
          num_flags_left = d->m_num_flags_left;
  mz_uint8 *pLZ_code_buf = d->m_pLZ_code_buf, *pLZ_flags = d->m_pLZ_flags;
  mz_uint cur_pos = lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK;

  while ((d->m_src_buf_left) || ((d->m_flush) && (lookahead_size))) {
    const mz_uint TDEFL_COMP_FAST_LOOKAHEAD_SIZE = 4096;
    mz_uint dst_pos =
        (lookahead_pos + lookahead_size) & TDEFL_LZ_DICT_SIZE_MASK;
    mz_uint num_bytes_to_process = (mz_uint)MZ_MIN(
        d->m_src_buf_left, TDEFL_COMP_FAST_LOOKAHEAD_SIZE - lookahead_size);
    d->m_src_buf_left -= num_bytes_to_process;
    lookahead_size += num_bytes_to_process;

    while (num_bytes_to_process) {
      mz_uint32 n = MZ_MIN(TDEFL_LZ_DICT_SIZE - dst_pos, num_bytes_to_process);
      memcpy(d->m_dict + dst_pos, d->m_pSrc, n);
      if (dst_pos < (TDEFL_MAX_MATCH_LEN - 1))
        memcpy(d->m_dict + TDEFL_LZ_DICT_SIZE + dst_pos, d->m_pSrc,
               MZ_MIN(n, (TDEFL_MAX_MATCH_LEN - 1) - dst_pos));
      d->m_pSrc += n;
      dst_pos = (dst_pos + n) & TDEFL_LZ_DICT_SIZE_MASK;
      num_bytes_to_process -= n;
    }

    dict_size = MZ_MIN(TDEFL_LZ_DICT_SIZE - lookahead_size, dict_size);
    if ((!d->m_flush) && (lookahead_size < TDEFL_COMP_FAST_LOOKAHEAD_SIZE))
      break;

    while (lookahead_size >= 4) {
      mz_uint cur_match_dist, cur_match_len = 1;
      mz_uint8 *pCur_dict = d->m_dict + cur_pos;
      mz_uint first_trigram = (*(const mz_uint32 *)pCur_dict) & 0xFFFFFF;
      mz_uint hash =
          (first_trigram ^ (first_trigram >> (24 - (TDEFL_LZ_HASH_BITS - 8)))) &
          TDEFL_LEVEL1_HASH_SIZE_MASK;
      mz_uint probe_pos = d->m_hash[hash];
      d->m_hash[hash] = (mz_uint16)lookahead_pos;

      if (((cur_match_dist = (mz_uint16)(lookahead_pos - probe_pos)) <=
           dict_size) &&
          ((*(const mz_uint32 *)(d->m_dict +
                                 (probe_pos &= TDEFL_LZ_DICT_SIZE_MASK)) &
            0xFFFFFF) == first_trigram)) {
        const mz_uint16 *p = (const mz_uint16 *)pCur_dict;
        const mz_uint16 *q = (const mz_uint16 *)(d->m_dict + probe_pos);
        mz_uint32 probe_len = 32;
        do {
        } while ((TDEFL_READ_UNALIGNED_WORD(++p) ==
                  TDEFL_READ_UNALIGNED_WORD(++q)) &&
                 (TDEFL_READ_UNALIGNED_WORD(++p) ==
                  TDEFL_READ_UNALIGNED_WORD(++q)) &&
                 (TDEFL_READ_UNALIGNED_WORD(++p) ==
                  TDEFL_READ_UNALIGNED_WORD(++q)) &&
                 (TDEFL_READ_UNALIGNED_WORD(++p) ==
                  TDEFL_READ_UNALIGNED_WORD(++q)) &&
                 (--probe_len > 0));
        cur_match_len = ((mz_uint)(p - (const mz_uint16 *)pCur_dict) * 2) +
                        (mz_uint)(*(const mz_uint8 *)p == *(const mz_uint8 *)q);
        if (!probe_len)
          cur_match_len = cur_match_dist ? TDEFL_MAX_MATCH_LEN : 0;

        if ((cur_match_len < TDEFL_MIN_MATCH_LEN) ||
            ((cur_match_len == TDEFL_MIN_MATCH_LEN) &&
             (cur_match_dist >= 8U * 1024U))) {
          cur_match_len = 1;
          *pLZ_code_buf++ = (mz_uint8)first_trigram;
          *pLZ_flags = (mz_uint8)(*pLZ_flags >> 1);
          d->m_huff_count[0][(mz_uint8)first_trigram]++;
        } else {
          mz_uint32 s0, s1;
          cur_match_len = MZ_MIN(cur_match_len, lookahead_size);

          MZ_ASSERT((cur_match_len >= TDEFL_MIN_MATCH_LEN) &&
                    (cur_match_dist >= 1) &&
                    (cur_match_dist <= TDEFL_LZ_DICT_SIZE));

          cur_match_dist--;

          pLZ_code_buf[0] = (mz_uint8)(cur_match_len - TDEFL_MIN_MATCH_LEN);
          *(mz_uint16 *)(&pLZ_code_buf[1]) = (mz_uint16)cur_match_dist;
          pLZ_code_buf += 3;
          *pLZ_flags = (mz_uint8)((*pLZ_flags >> 1) | 0x80);

          s0 = s_tdefl_small_dist_sym[cur_match_dist & 511];
          s1 = s_tdefl_large_dist_sym[cur_match_dist >> 8];
          d->m_huff_count[1][(cur_match_dist < 512) ? s0 : s1]++;

          d->m_huff_count[0][s_tdefl_len_sym[cur_match_len -
                                             TDEFL_MIN_MATCH_LEN]]++;
        }
      } else {
        *pLZ_code_buf++ = (mz_uint8)first_trigram;
        *pLZ_flags = (mz_uint8)(*pLZ_flags >> 1);
        d->m_huff_count[0][(mz_uint8)first_trigram]++;
      }

      if (--num_flags_left == 0) {
        num_flags_left = 8;
        pLZ_flags = pLZ_code_buf++;
      }

      total_lz_bytes += cur_match_len;
      lookahead_pos += cur_match_len;
      dict_size = MZ_MIN(dict_size + cur_match_len, TDEFL_LZ_DICT_SIZE);
      cur_pos = (cur_pos + cur_match_len) & TDEFL_LZ_DICT_SIZE_MASK;
      MZ_ASSERT(lookahead_size >= cur_match_len);
      lookahead_size -= cur_match_len;

      if (pLZ_code_buf > &d->m_lz_code_buf[TDEFL_LZ_CODE_BUF_SIZE - 8]) {
        int n;
        d->m_lookahead_pos = lookahead_pos;
        d->m_lookahead_size = lookahead_size;
        d->m_dict_size = dict_size;
        d->m_total_lz_bytes = total_lz_bytes;
        d->m_pLZ_code_buf = pLZ_code_buf;
        d->m_pLZ_flags = pLZ_flags;
        d->m_num_flags_left = num_flags_left;
        if ((n = tdefl_flush_block(d, 0)) != 0)
          return (n < 0) ? MZ_FALSE : MZ_TRUE;
        total_lz_bytes = d->m_total_lz_bytes;
        pLZ_code_buf = d->m_pLZ_code_buf;
        pLZ_flags = d->m_pLZ_flags;
        num_flags_left = d->m_num_flags_left;
      }
    }

    while (lookahead_size) {
      mz_uint8 lit = d->m_dict[cur_pos];

      total_lz_bytes++;
      *pLZ_code_buf++ = lit;
      *pLZ_flags = (mz_uint8)(*pLZ_flags >> 1);
      if (--num_flags_left == 0) {
        num_flags_left = 8;
        pLZ_flags = pLZ_code_buf++;
      }

      d->m_huff_count[0][lit]++;

      lookahead_pos++;
      dict_size = MZ_MIN(dict_size + 1, TDEFL_LZ_DICT_SIZE);
      cur_pos = (cur_pos + 1) & TDEFL_LZ_DICT_SIZE_MASK;
      lookahead_size--;

      if (pLZ_code_buf > &d->m_lz_code_buf[TDEFL_LZ_CODE_BUF_SIZE - 8]) {
        int n;
        d->m_lookahead_pos = lookahead_pos;
        d->m_lookahead_size = lookahead_size;
        d->m_dict_size = dict_size;
        d->m_total_lz_bytes = total_lz_bytes;
        d->m_pLZ_code_buf = pLZ_code_buf;
        d->m_pLZ_flags = pLZ_flags;
        d->m_num_flags_left = num_flags_left;
        if ((n = tdefl_flush_block(d, 0)) != 0)
          return (n < 0) ? MZ_FALSE : MZ_TRUE;
        total_lz_bytes = d->m_total_lz_bytes;
        pLZ_code_buf = d->m_pLZ_code_buf;
        pLZ_flags = d->m_pLZ_flags;
        num_flags_left = d->m_num_flags_left;
      }
    }
  }

  d->m_lookahead_pos = lookahead_pos;
  d->m_lookahead_size = lookahead_size;
  d->m_dict_size = dict_size;
  d->m_total_lz_bytes = total_lz_bytes;
  d->m_pLZ_code_buf = pLZ_code_buf;
  d->m_pLZ_flags = pLZ_flags;
  d->m_num_flags_left = num_flags_left;
  return MZ_TRUE;
}
#endif  // MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN

static MZ_FORCEINLINE void tdefl_record_literal(tdefl_compressor *d,
                                                mz_uint8 lit) {
  d->m_total_lz_bytes++;
  *d->m_pLZ_code_buf++ = lit;
  *d->m_pLZ_flags = (mz_uint8)(*d->m_pLZ_flags >> 1);
  if (--d->m_num_flags_left == 0) {
    d->m_num_flags_left = 8;
    d->m_pLZ_flags = d->m_pLZ_code_buf++;
  }
  d->m_huff_count[0][lit]++;
}

static MZ_FORCEINLINE void tdefl_record_match(tdefl_compressor *d,
                                              mz_uint match_len,
                                              mz_uint match_dist) {
  mz_uint32 s0, s1;

  MZ_ASSERT((match_len >= TDEFL_MIN_MATCH_LEN) && (match_dist >= 1) &&
            (match_dist <= TDEFL_LZ_DICT_SIZE));

  d->m_total_lz_bytes += match_len;

  d->m_pLZ_code_buf[0] = (mz_uint8)(match_len - TDEFL_MIN_MATCH_LEN);

  match_dist -= 1;
  d->m_pLZ_code_buf[1] = (mz_uint8)(match_dist & 0xFF);
  d->m_pLZ_code_buf[2] = (mz_uint8)(match_dist >> 8);
  d->m_pLZ_code_buf += 3;

  *d->m_pLZ_flags = (mz_uint8)((*d->m_pLZ_flags >> 1) | 0x80);
  if (--d->m_num_flags_left == 0) {
    d->m_num_flags_left = 8;
    d->m_pLZ_flags = d->m_pLZ_code_buf++;
  }

  s0 = s_tdefl_small_dist_sym[match_dist & 511];
  s1 = s_tdefl_large_dist_sym[(match_dist >> 8) & 127];
  d->m_huff_count[1][(match_dist < 512) ? s0 : s1]++;

  if (match_len >= TDEFL_MIN_MATCH_LEN)
    d->m_huff_count[0][s_tdefl_len_sym[match_len - TDEFL_MIN_MATCH_LEN]]++;
}

static mz_bool tdefl_compress_normal(tdefl_compressor *d) {
  const mz_uint8 *pSrc = d->m_pSrc;
  size_t src_buf_left = d->m_src_buf_left;
  tdefl_flush flush = d->m_flush;

  while ((src_buf_left) || ((flush) && (d->m_lookahead_size))) {
    mz_uint len_to_move, cur_match_dist, cur_match_len, cur_pos;
    // Update dictionary and hash chains. Keeps the lookahead size equal to
    // TDEFL_MAX_MATCH_LEN.
    if ((d->m_lookahead_size + d->m_dict_size) >= (TDEFL_MIN_MATCH_LEN - 1)) {
      mz_uint dst_pos = (d->m_lookahead_pos + d->m_lookahead_size) &
                        TDEFL_LZ_DICT_SIZE_MASK,
              ins_pos = d->m_lookahead_pos + d->m_lookahead_size - 2;
      mz_uint hash = (d->m_dict[ins_pos & TDEFL_LZ_DICT_SIZE_MASK]
                      << TDEFL_LZ_HASH_SHIFT) ^
                     d->m_dict[(ins_pos + 1) & TDEFL_LZ_DICT_SIZE_MASK];
      mz_uint num_bytes_to_process = (mz_uint)MZ_MIN(
          src_buf_left, TDEFL_MAX_MATCH_LEN - d->m_lookahead_size);
      const mz_uint8 *pSrc_end = pSrc + num_bytes_to_process;
      src_buf_left -= num_bytes_to_process;
      d->m_lookahead_size += num_bytes_to_process;
      while (pSrc != pSrc_end) {
        mz_uint8 c = *pSrc++;
        d->m_dict[dst_pos] = c;
        if (dst_pos < (TDEFL_MAX_MATCH_LEN - 1))
          d->m_dict[TDEFL_LZ_DICT_SIZE + dst_pos] = c;
        hash = ((hash << TDEFL_LZ_HASH_SHIFT) ^ c) & (TDEFL_LZ_HASH_SIZE - 1);
        d->m_next[ins_pos & TDEFL_LZ_DICT_SIZE_MASK] = d->m_hash[hash];
        d->m_hash[hash] = (mz_uint16)(ins_pos);
        dst_pos = (dst_pos + 1) & TDEFL_LZ_DICT_SIZE_MASK;
        ins_pos++;
      }
    } else {
      while ((src_buf_left) && (d->m_lookahead_size < TDEFL_MAX_MATCH_LEN)) {
        mz_uint8 c = *pSrc++;
        mz_uint dst_pos = (d->m_lookahead_pos + d->m_lookahead_size) &
                          TDEFL_LZ_DICT_SIZE_MASK;
        src_buf_left--;
        d->m_dict[dst_pos] = c;
        if (dst_pos < (TDEFL_MAX_MATCH_LEN - 1))
          d->m_dict[TDEFL_LZ_DICT_SIZE + dst_pos] = c;
        if ((++d->m_lookahead_size + d->m_dict_size) >= TDEFL_MIN_MATCH_LEN) {
          mz_uint ins_pos = d->m_lookahead_pos + (d->m_lookahead_size - 1) - 2;
          mz_uint hash = ((d->m_dict[ins_pos & TDEFL_LZ_DICT_SIZE_MASK]
                           << (TDEFL_LZ_HASH_SHIFT * 2)) ^
                          (d->m_dict[(ins_pos + 1) & TDEFL_LZ_DICT_SIZE_MASK]
                           << TDEFL_LZ_HASH_SHIFT) ^
                          c) &
                         (TDEFL_LZ_HASH_SIZE - 1);
          d->m_next[ins_pos & TDEFL_LZ_DICT_SIZE_MASK] = d->m_hash[hash];
          d->m_hash[hash] = (mz_uint16)(ins_pos);
        }
      }
    }
    d->m_dict_size =
        MZ_MIN(TDEFL_LZ_DICT_SIZE - d->m_lookahead_size, d->m_dict_size);
    if ((!flush) && (d->m_lookahead_size < TDEFL_MAX_MATCH_LEN)) break;

    // Simple lazy/greedy parsing state machine.
    len_to_move = 1;
    cur_match_dist = 0;
    cur_match_len =
        d->m_saved_match_len ? d->m_saved_match_len : (TDEFL_MIN_MATCH_LEN - 1);
    cur_pos = d->m_lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK;
    if (d->m_flags & (TDEFL_RLE_MATCHES | TDEFL_FORCE_ALL_RAW_BLOCKS)) {
      if ((d->m_dict_size) && (!(d->m_flags & TDEFL_FORCE_ALL_RAW_BLOCKS))) {
        mz_uint8 c = d->m_dict[(cur_pos - 1) & TDEFL_LZ_DICT_SIZE_MASK];
        cur_match_len = 0;
        while (cur_match_len < d->m_lookahead_size) {
          if (d->m_dict[cur_pos + cur_match_len] != c) break;
          cur_match_len++;
        }
        if (cur_match_len < TDEFL_MIN_MATCH_LEN)
          cur_match_len = 0;
        else
          cur_match_dist = 1;
      }
    } else {
      tdefl_find_match(d, d->m_lookahead_pos, d->m_dict_size,
                       d->m_lookahead_size, &cur_match_dist, &cur_match_len);
    }
    if (((cur_match_len == TDEFL_MIN_MATCH_LEN) &&
         (cur_match_dist >= 8U * 1024U)) ||
        (cur_pos == cur_match_dist) ||
        ((d->m_flags & TDEFL_FILTER_MATCHES) && (cur_match_len <= 5))) {
      cur_match_dist = cur_match_len = 0;
    }
    if (d->m_saved_match_len) {
      if (cur_match_len > d->m_saved_match_len) {
        tdefl_record_literal(d, (mz_uint8)d->m_saved_lit);
        if (cur_match_len >= 128) {
          tdefl_record_match(d, cur_match_len, cur_match_dist);
          d->m_saved_match_len = 0;
          len_to_move = cur_match_len;
        } else {
          d->m_saved_lit = d->m_dict[cur_pos];
          d->m_saved_match_dist = cur_match_dist;
          d->m_saved_match_len = cur_match_len;
        }
      } else {
        tdefl_record_match(d, d->m_saved_match_len, d->m_saved_match_dist);
        len_to_move = d->m_saved_match_len - 1;
        d->m_saved_match_len = 0;
      }
    } else if (!cur_match_dist)
      tdefl_record_literal(d,
                           d->m_dict[MZ_MIN(cur_pos, sizeof(d->m_dict) - 1)]);
    else if ((d->m_greedy_parsing) || (d->m_flags & TDEFL_RLE_MATCHES) ||
             (cur_match_len >= 128)) {
      tdefl_record_match(d, cur_match_len, cur_match_dist);
      len_to_move = cur_match_len;
    } else {
      d->m_saved_lit = d->m_dict[MZ_MIN(cur_pos, sizeof(d->m_dict) - 1)];
      d->m_saved_match_dist = cur_match_dist;
      d->m_saved_match_len = cur_match_len;
    }
    // Move the lookahead forward by len_to_move bytes.
    d->m_lookahead_pos += len_to_move;
    MZ_ASSERT(d->m_lookahead_size >= len_to_move);
    d->m_lookahead_size -= len_to_move;
    d->m_dict_size =
        MZ_MIN(d->m_dict_size + len_to_move, (mz_uint)TDEFL_LZ_DICT_SIZE);
    // Check if it's time to flush the current LZ codes to the internal output
    // buffer.
    if ((d->m_pLZ_code_buf > &d->m_lz_code_buf[TDEFL_LZ_CODE_BUF_SIZE - 8]) ||
        ((d->m_total_lz_bytes > 31 * 1024) &&
         (((((mz_uint)(d->m_pLZ_code_buf - d->m_lz_code_buf) * 115) >> 7) >=
           d->m_total_lz_bytes) ||
          (d->m_flags & TDEFL_FORCE_ALL_RAW_BLOCKS)))) {
      int n;
      d->m_pSrc = pSrc;
      d->m_src_buf_left = src_buf_left;
      if ((n = tdefl_flush_block(d, 0)) != 0)
        return (n < 0) ? MZ_FALSE : MZ_TRUE;
    }
  }

  d->m_pSrc = pSrc;
  d->m_src_buf_left = src_buf_left;
  return MZ_TRUE;
}

static tdefl_status tdefl_flush_output_buffer(tdefl_compressor *d) {
  if (d->m_pIn_buf_size) {
    *d->m_pIn_buf_size = d->m_pSrc - (const mz_uint8 *)d->m_pIn_buf;
  }

  if (d->m_pOut_buf_size) {
    size_t n = MZ_MIN(*d->m_pOut_buf_size - d->m_out_buf_ofs,
                      d->m_output_flush_remaining);
    memcpy((mz_uint8 *)d->m_pOut_buf + d->m_out_buf_ofs,
           d->m_output_buf + d->m_output_flush_ofs, n);
    d->m_output_flush_ofs += (mz_uint)n;
    d->m_output_flush_remaining -= (mz_uint)n;
    d->m_out_buf_ofs += n;

    *d->m_pOut_buf_size = d->m_out_buf_ofs;
  }

  return (d->m_finished && !d->m_output_flush_remaining) ? TDEFL_STATUS_DONE
                                                         : TDEFL_STATUS_OKAY;
}

tdefl_status tdefl_compress(tdefl_compressor *d, const void *pIn_buf,
                            size_t *pIn_buf_size, void *pOut_buf,
                            size_t *pOut_buf_size, tdefl_flush flush) {
  if (!d) {
    if (pIn_buf_size) *pIn_buf_size = 0;
    if (pOut_buf_size) *pOut_buf_size = 0;
    return TDEFL_STATUS_BAD_PARAM;
  }

  d->m_pIn_buf = pIn_buf;
  d->m_pIn_buf_size = pIn_buf_size;
  d->m_pOut_buf = pOut_buf;
  d->m_pOut_buf_size = pOut_buf_size;
  d->m_pSrc = (const mz_uint8 *)(pIn_buf);
  d->m_src_buf_left = pIn_buf_size ? *pIn_buf_size : 0;
  d->m_out_buf_ofs = 0;
  d->m_flush = flush;

  if (((d->m_pPut_buf_func != NULL) ==
       ((pOut_buf != NULL) || (pOut_buf_size != NULL))) ||
      (d->m_prev_return_status != TDEFL_STATUS_OKAY) ||
      (d->m_wants_to_finish && (flush != TDEFL_FINISH)) ||
      (pIn_buf_size && *pIn_buf_size && !pIn_buf) ||
      (pOut_buf_size && *pOut_buf_size && !pOut_buf)) {
    if (pIn_buf_size) *pIn_buf_size = 0;
    if (pOut_buf_size) *pOut_buf_size = 0;
    return (d->m_prev_return_status = TDEFL_STATUS_BAD_PARAM);
  }
  d->m_wants_to_finish |= (flush == TDEFL_FINISH);

  if ((d->m_output_flush_remaining) || (d->m_finished))
    return (d->m_prev_return_status = tdefl_flush_output_buffer(d));

#if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
  if (((d->m_flags & TDEFL_MAX_PROBES_MASK) == 1) &&
      ((d->m_flags & TDEFL_GREEDY_PARSING_FLAG) != 0) &&
      ((d->m_flags & (TDEFL_FILTER_MATCHES | TDEFL_FORCE_ALL_RAW_BLOCKS |
                      TDEFL_RLE_MATCHES)) == 0)) {
    if (!tdefl_compress_fast(d)) return d->m_prev_return_status;
  } else
#endif  // #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
  {
    if (!tdefl_compress_normal(d)) return d->m_prev_return_status;
  }

  if ((d->m_flags & (TDEFL_WRITE_ZLIB_HEADER | TDEFL_COMPUTE_ADLER32)) &&
      (pIn_buf))
    d->m_adler32 =
        (mz_uint32)mz_adler32(d->m_adler32, (const mz_uint8 *)pIn_buf,
                              d->m_pSrc - (const mz_uint8 *)pIn_buf);

  if ((flush) && (!d->m_lookahead_size) && (!d->m_src_buf_left) &&
      (!d->m_output_flush_remaining)) {
    if (tdefl_flush_block(d, flush) < 0) return d->m_prev_return_status;
    d->m_finished = (flush == TDEFL_FINISH);
    if (flush == TDEFL_FULL_FLUSH) {
      MZ_CLEAR_OBJ(d->m_hash);
      MZ_CLEAR_OBJ(d->m_next);
      d->m_dict_size = 0;
    }
  }

  return (d->m_prev_return_status = tdefl_flush_output_buffer(d));
}

tdefl_status tdefl_compress_buffer(tdefl_compressor *d, const void *pIn_buf,
                                   size_t in_buf_size, tdefl_flush flush) {
  MZ_ASSERT(d->m_pPut_buf_func);
  return tdefl_compress(d, pIn_buf, &in_buf_size, NULL, NULL, flush);
}

tdefl_status tdefl_init(tdefl_compressor *d,
                        tdefl_put_buf_func_ptr pPut_buf_func,
                        void *pPut_buf_user, int flags) {
  d->m_pPut_buf_func = pPut_buf_func;
  d->m_pPut_buf_user = pPut_buf_user;
  d->m_flags = (mz_uint)(flags);
  d->m_max_probes[0] = 1 + ((flags & 0xFFF) + 2) / 3;
  d->m_greedy_parsing = (flags & TDEFL_GREEDY_PARSING_FLAG) != 0;
  d->m_max_probes[1] = 1 + (((flags & 0xFFF) >> 2) + 2) / 3;
  if (!(flags & TDEFL_NONDETERMINISTIC_PARSING_FLAG)) MZ_CLEAR_OBJ(d->m_hash);
  d->m_lookahead_pos = d->m_lookahead_size = d->m_dict_size =
      d->m_total_lz_bytes = d->m_lz_code_buf_dict_pos = d->m_bits_in = 0;
  d->m_output_flush_ofs = d->m_output_flush_remaining = d->m_finished =
      d->m_block_index = d->m_bit_buffer = d->m_wants_to_finish = 0;
  d->m_pLZ_code_buf = d->m_lz_code_buf + 1;
  d->m_pLZ_flags = d->m_lz_code_buf;
  d->m_num_flags_left = 8;
  d->m_pOutput_buf = d->m_output_buf;
  d->m_pOutput_buf_end = d->m_output_buf;
  d->m_prev_return_status = TDEFL_STATUS_OKAY;
  d->m_saved_match_dist = d->m_saved_match_len = d->m_saved_lit = 0;
  d->m_adler32 = 1;
  d->m_pIn_buf = NULL;
  d->m_pOut_buf = NULL;
  d->m_pIn_buf_size = NULL;
  d->m_pOut_buf_size = NULL;
  d->m_flush = TDEFL_NO_FLUSH;
  d->m_pSrc = NULL;
  d->m_src_buf_left = 0;
  d->m_out_buf_ofs = 0;
  memset(&d->m_huff_count[0][0], 0,
         sizeof(d->m_huff_count[0][0]) * TDEFL_MAX_HUFF_SYMBOLS_0);
  memset(&d->m_huff_count[1][0], 0,
         sizeof(d->m_huff_count[1][0]) * TDEFL_MAX_HUFF_SYMBOLS_1);
  return TDEFL_STATUS_OKAY;
}

tdefl_status tdefl_get_prev_return_status(tdefl_compressor *d) {
  return d->m_prev_return_status;
}

mz_uint32 tdefl_get_adler32(tdefl_compressor *d) { return d->m_adler32; }

mz_bool tdefl_compress_mem_to_output(const void *pBuf, size_t buf_len,
                                     tdefl_put_buf_func_ptr pPut_buf_func,
                                     void *pPut_buf_user, int flags) {
  tdefl_compressor *pComp;
  mz_bool succeeded;
  if (((buf_len) && (!pBuf)) || (!pPut_buf_func)) return MZ_FALSE;
  pComp = (tdefl_compressor *)MZ_MALLOC(sizeof(tdefl_compressor));
  if (!pComp) return MZ_FALSE;
  succeeded = (tdefl_init(pComp, pPut_buf_func, pPut_buf_user, flags) ==
               TDEFL_STATUS_OKAY);
  succeeded =
      succeeded && (tdefl_compress_buffer(pComp, pBuf, buf_len, TDEFL_FINISH) ==
                    TDEFL_STATUS_DONE);
  MZ_FREE(pComp);
  return succeeded;
}

typedef struct {
  size_t m_size, m_capacity;
  mz_uint8 *m_pBuf;
  mz_bool m_expandable;
} tdefl_output_buffer;

static mz_bool tdefl_output_buffer_putter(const void *pBuf, int len,
                                          void *pUser) {
  tdefl_output_buffer *p = (tdefl_output_buffer *)pUser;
  size_t new_size = p->m_size + len;
  if (new_size > p->m_capacity) {
    size_t new_capacity = p->m_capacity;
    mz_uint8 *pNew_buf;
    if (!p->m_expandable) return MZ_FALSE;
    do {
      new_capacity = MZ_MAX(128U, new_capacity << 1U);
    } while (new_size > new_capacity);
    pNew_buf = (mz_uint8 *)MZ_REALLOC(p->m_pBuf, new_capacity);
    if (!pNew_buf) return MZ_FALSE;
    p->m_pBuf = pNew_buf;
    p->m_capacity = new_capacity;
  }
  memcpy((mz_uint8 *)p->m_pBuf + p->m_size, pBuf, len);
  p->m_size = new_size;
  return MZ_TRUE;
}

void *tdefl_compress_mem_to_heap(const void *pSrc_buf, size_t src_buf_len,
                                 size_t *pOut_len, int flags) {
  tdefl_output_buffer out_buf;
  MZ_CLEAR_OBJ(out_buf);
  if (!pOut_len)
    return MZ_FALSE;
  else
    *pOut_len = 0;
  out_buf.m_expandable = MZ_TRUE;
  if (!tdefl_compress_mem_to_output(
          pSrc_buf, src_buf_len, tdefl_output_buffer_putter, &out_buf, flags))
    return NULL;
  *pOut_len = out_buf.m_size;
  return out_buf.m_pBuf;
}

size_t tdefl_compress_mem_to_mem(void *pOut_buf, size_t out_buf_len,
                                 const void *pSrc_buf, size_t src_buf_len,
                                 int flags) {
  tdefl_output_buffer out_buf;
  MZ_CLEAR_OBJ(out_buf);
  if (!pOut_buf) return 0;
  out_buf.m_pBuf = (mz_uint8 *)pOut_buf;
  out_buf.m_capacity = out_buf_len;
  if (!tdefl_compress_mem_to_output(
          pSrc_buf, src_buf_len, tdefl_output_buffer_putter, &out_buf, flags))
    return 0;
  return out_buf.m_size;
}

#ifndef MINIZ_NO_ZLIB_APIS
static const mz_uint s_tdefl_num_probes[11] = {0,   1,   6,   32,  16,  32,
                                               128, 256, 512, 768, 1500};

// level may actually range from [0,10] (10 is a "hidden" max level, where we
// want a bit more compression and it's fine if throughput to fall off a cliff
// on some files).
mz_uint tdefl_create_comp_flags_from_zip_params(int level, int window_bits,
                                                int strategy) {
  mz_uint comp_flags =
      s_tdefl_num_probes[(level >= 0) ? MZ_MIN(10, level) : MZ_DEFAULT_LEVEL] |
      ((level <= 3) ? TDEFL_GREEDY_PARSING_FLAG : 0);
  if (window_bits > 0) comp_flags |= TDEFL_WRITE_ZLIB_HEADER;

  if (!level)
    comp_flags |= TDEFL_FORCE_ALL_RAW_BLOCKS;
  else if (strategy == MZ_FILTERED)
    comp_flags |= TDEFL_FILTER_MATCHES;
  else if (strategy == MZ_HUFFMAN_ONLY)
    comp_flags &= ~TDEFL_MAX_PROBES_MASK;
  else if (strategy == MZ_FIXED)
    comp_flags |= TDEFL_FORCE_ALL_STATIC_BLOCKS;
  else if (strategy == MZ_RLE)
    comp_flags |= TDEFL_RLE_MATCHES;

  return comp_flags;
}
#endif  // MINIZ_NO_ZLIB_APIS

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4204)  // nonstandard extension used : non-constant
                                 // aggregate initializer (also supported by GNU
                                 // C and C99, so no big deal)
#pragma warning(disable : 4244)  // 'initializing': conversion from '__int64' to
                                 // 'int', possible loss of data
#pragma warning(disable : 4267)  // 'argument': conversion from '__int64' to
                                 // 'int', possible loss of data
#pragma warning(disable : 4996)  // 'strdup': The POSIX name for this item is
                                 // deprecated. Instead, use the ISO C and C++
                                 // conformant name: _strdup.
#endif

// Simple PNG writer function by Alex Evans, 2011. Released into the public
// domain: https://gist.github.com/908299, more context at
// http://altdevblogaday.org/2011/04/06/a-smaller-jpg-encoder/.
// This is actually a modification of Alex's original code so PNG files
// generated by this function pass pngcheck.
void *tdefl_write_image_to_png_file_in_memory_ex(const void *pImage, int w,
                                                 int h, int num_chans,
                                                 size_t *pLen_out,
                                                 mz_uint level, mz_bool flip) {
  // Using a local copy of this array here in case MINIZ_NO_ZLIB_APIS was
  // defined.
  static const mz_uint s_tdefl_png_num_probes[11] = {
      0, 1, 6, 32, 16, 32, 128, 256, 512, 768, 1500};
  tdefl_compressor *pComp =
      (tdefl_compressor *)MZ_MALLOC(sizeof(tdefl_compressor));
  tdefl_output_buffer out_buf;
  int i, bpl = w * num_chans, y, z;
  mz_uint32 c;
  *pLen_out = 0;
  if (!pComp) return NULL;
  MZ_CLEAR_OBJ(out_buf);
  out_buf.m_expandable = MZ_TRUE;
  out_buf.m_capacity = 57 + MZ_MAX(64, (1 + bpl) * h);
  if (NULL == (out_buf.m_pBuf = (mz_uint8 *)MZ_MALLOC(out_buf.m_capacity))) {
    MZ_FREE(pComp);
    return NULL;
  }
  // write dummy header
  for (z = 41; z; --z) tdefl_output_buffer_putter(&z, 1, &out_buf);
  // compress image data
  tdefl_init(
      pComp, tdefl_output_buffer_putter, &out_buf,
      s_tdefl_png_num_probes[MZ_MIN(10, level)] | TDEFL_WRITE_ZLIB_HEADER);
  for (y = 0; y < h; ++y) {
    tdefl_compress_buffer(pComp, &z, 1, TDEFL_NO_FLUSH);
    tdefl_compress_buffer(pComp,
                          (mz_uint8 *)pImage + (flip ? (h - 1 - y) : y) * bpl,
                          bpl, TDEFL_NO_FLUSH);
  }
  if (tdefl_compress_buffer(pComp, NULL, 0, TDEFL_FINISH) !=
      TDEFL_STATUS_DONE) {
    MZ_FREE(pComp);
    MZ_FREE(out_buf.m_pBuf);
    return NULL;
  }
  // write real header
  *pLen_out = out_buf.m_size - 41;
  {
    static const mz_uint8 chans[] = {0x00, 0x00, 0x04, 0x02, 0x06};
    mz_uint8 pnghdr[41] = {0x89,
                           0x50,
                           0x4e,
                           0x47,
                           0x0d,
                           0x0a,
                           0x1a,
                           0x0a,
                           0x00,
                           0x00,
                           0x00,
                           0x0d,
                           0x49,
                           0x48,
                           0x44,
                           0x52,
                           0,
                           0,
                           (mz_uint8)(w >> 8),
                           (mz_uint8)w,
                           0,
                           0,
                           (mz_uint8)(h >> 8),
                           (mz_uint8)h,
                           8,
                           chans[num_chans],
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           (mz_uint8)(*pLen_out >> 24),
                           (mz_uint8)(*pLen_out >> 16),
                           (mz_uint8)(*pLen_out >> 8),
                           (mz_uint8)*pLen_out,
                           0x49,
                           0x44,
                           0x41,
                           0x54};
    c = (mz_uint32)mz_crc32(MZ_CRC32_INIT, pnghdr + 12, 17);
    for (i = 0; i < 4; ++i, c <<= 8)
      ((mz_uint8 *)(pnghdr + 29))[i] = (mz_uint8)(c >> 24);
    memcpy(out_buf.m_pBuf, pnghdr, 41);
  }
  // write footer (IDAT CRC-32, followed by IEND chunk)
  if (!tdefl_output_buffer_putter(
          "\0\0\0\0\0\0\0\0\x49\x45\x4e\x44\xae\x42\x60\x82", 16, &out_buf)) {
    *pLen_out = 0;
    MZ_FREE(pComp);
    MZ_FREE(out_buf.m_pBuf);
    return NULL;
  }
  c = (mz_uint32)mz_crc32(MZ_CRC32_INIT, out_buf.m_pBuf + 41 - 4,
                          *pLen_out + 4);
  for (i = 0; i < 4; ++i, c <<= 8)
    (out_buf.m_pBuf + out_buf.m_size - 16)[i] = (mz_uint8)(c >> 24);
  // compute final size of file, grab compressed data buffer and return
  *pLen_out += 57;
  MZ_FREE(pComp);
  return out_buf.m_pBuf;
}
void *tdefl_write_image_to_png_file_in_memory(const void *pImage, int w, int h,
                                              int num_chans, size_t *pLen_out) {
  // Level 6 corresponds to TDEFL_DEFAULT_MAX_PROBES or MZ_DEFAULT_LEVEL (but we
  // can't depend on MZ_DEFAULT_LEVEL being available in case the zlib API's
  // where #defined out)
  return tdefl_write_image_to_png_file_in_memory_ex(pImage, w, h, num_chans,
                                                    pLen_out, 6, MZ_FALSE);
}

// ------------------- .ZIP archive reading

#ifndef MINIZ_NO_ARCHIVE_APIS
#error "No arvhive APIs"

#ifdef MINIZ_NO_STDIO
#define MZ_FILE void *
#else
#include <stdio.h>
#include <sys/stat.h>

#if defined(_MSC_VER) || defined(__MINGW64__)
static FILE *mz_fopen(const char *pFilename, const char *pMode) {
  FILE *pFile = NULL;
  fopen_s(&pFile, pFilename, pMode);
  return pFile;
}
static FILE *mz_freopen(const char *pPath, const char *pMode, FILE *pStream) {
  FILE *pFile = NULL;
  if (freopen_s(&pFile, pPath, pMode, pStream)) return NULL;
  return pFile;
}
#ifndef MINIZ_NO_TIME
#include <sys/utime.h>
#endif
#define MZ_FILE FILE
#define MZ_FOPEN mz_fopen
#define MZ_FCLOSE fclose
#define MZ_FREAD fread
#define MZ_FWRITE fwrite
#define MZ_FTELL64 _ftelli64
#define MZ_FSEEK64 _fseeki64
#define MZ_FILE_STAT_STRUCT _stat
#define MZ_FILE_STAT _stat
#define MZ_FFLUSH fflush
#define MZ_FREOPEN mz_freopen
#define MZ_DELETE_FILE remove
#elif defined(__MINGW32__)
#ifndef MINIZ_NO_TIME
#include <sys/utime.h>
#endif
#define MZ_FILE FILE
#define MZ_FOPEN(f, m) fopen(f, m)
#define MZ_FCLOSE fclose
#define MZ_FREAD fread
#define MZ_FWRITE fwrite
#define MZ_FTELL64 ftello64
#define MZ_FSEEK64 fseeko64
#define MZ_FILE_STAT_STRUCT _stat
#define MZ_FILE_STAT _stat
#define MZ_FFLUSH fflush
#define MZ_FREOPEN(f, m, s) freopen(f, m, s)
#define MZ_DELETE_FILE remove
#elif defined(__TINYC__)
#ifndef MINIZ_NO_TIME
#include <sys/utime.h>
#endif
#define MZ_FILE FILE
#define MZ_FOPEN(f, m) fopen(f, m)
#define MZ_FCLOSE fclose
#define MZ_FREAD fread
#define MZ_FWRITE fwrite
#define MZ_FTELL64 ftell
#define MZ_FSEEK64 fseek
#define MZ_FILE_STAT_STRUCT stat
#define MZ_FILE_STAT stat
#define MZ_FFLUSH fflush
#define MZ_FREOPEN(f, m, s) freopen(f, m, s)
#define MZ_DELETE_FILE remove
#elif defined(__GNUC__) && defined(_LARGEFILE64_SOURCE) && _LARGEFILE64_SOURCE
#ifndef MINIZ_NO_TIME
#include <utime.h>
#endif
#define MZ_FILE FILE
#define MZ_FOPEN(f, m) fopen64(f, m)
#define MZ_FCLOSE fclose
#define MZ_FREAD fread
#define MZ_FWRITE fwrite
#define MZ_FTELL64 ftello64
#define MZ_FSEEK64 fseeko64
#define MZ_FILE_STAT_STRUCT stat64
#define MZ_FILE_STAT stat64
#define MZ_FFLUSH fflush
#define MZ_FREOPEN(p, m, s) freopen64(p, m, s)
#define MZ_DELETE_FILE remove
#else
#ifndef MINIZ_NO_TIME
#include <utime.h>
#endif
#define MZ_FILE FILE
#define MZ_FOPEN(f, m) fopen(f, m)
#define MZ_FCLOSE fclose
#define MZ_FREAD fread
#define MZ_FWRITE fwrite
#define MZ_FTELL64 ftello
#define MZ_FSEEK64 fseeko
#define MZ_FILE_STAT_STRUCT stat
#define MZ_FILE_STAT stat
#define MZ_FFLUSH fflush
#define MZ_FREOPEN(f, m, s) freopen(f, m, s)
#define MZ_DELETE_FILE remove
#endif  // #ifdef _MSC_VER
#endif  // #ifdef MINIZ_NO_STDIO

#define MZ_TOLOWER(c) ((((c) >= 'A') && ((c) <= 'Z')) ? ((c) - 'A' + 'a') : (c))

// Various ZIP archive enums. To completely avoid cross platform compiler
// alignment and platform endian issues, miniz.c doesn't use structs for any of
// this stuff.
enum {
  // ZIP archive identifiers and record sizes
  MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIG = 0x06054b50,
  MZ_ZIP_CENTRAL_DIR_HEADER_SIG = 0x02014b50,
  MZ_ZIP_LOCAL_DIR_HEADER_SIG = 0x04034b50,
  MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30,
  MZ_ZIP_CENTRAL_DIR_HEADER_SIZE = 46,
  MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE = 22,
  // Central directory header record offsets
  MZ_ZIP_CDH_SIG_OFS = 0,
  MZ_ZIP_CDH_VERSION_MADE_BY_OFS = 4,
  MZ_ZIP_CDH_VERSION_NEEDED_OFS = 6,
  MZ_ZIP_CDH_BIT_FLAG_OFS = 8,
  MZ_ZIP_CDH_METHOD_OFS = 10,
  MZ_ZIP_CDH_FILE_TIME_OFS = 12,
  MZ_ZIP_CDH_FILE_DATE_OFS = 14,
  MZ_ZIP_CDH_CRC32_OFS = 16,
  MZ_ZIP_CDH_COMPRESSED_SIZE_OFS = 20,
  MZ_ZIP_CDH_DECOMPRESSED_SIZE_OFS = 24,
  MZ_ZIP_CDH_FILENAME_LEN_OFS = 28,
  MZ_ZIP_CDH_EXTRA_LEN_OFS = 30,
  MZ_ZIP_CDH_COMMENT_LEN_OFS = 32,
  MZ_ZIP_CDH_DISK_START_OFS = 34,
  MZ_ZIP_CDH_INTERNAL_ATTR_OFS = 36,
  MZ_ZIP_CDH_EXTERNAL_ATTR_OFS = 38,
  MZ_ZIP_CDH_LOCAL_HEADER_OFS = 42,
  // Local directory header offsets
  MZ_ZIP_LDH_SIG_OFS = 0,
  MZ_ZIP_LDH_VERSION_NEEDED_OFS = 4,
  MZ_ZIP_LDH_BIT_FLAG_OFS = 6,
  MZ_ZIP_LDH_METHOD_OFS = 8,
  MZ_ZIP_LDH_FILE_TIME_OFS = 10,
  MZ_ZIP_LDH_FILE_DATE_OFS = 12,
  MZ_ZIP_LDH_CRC32_OFS = 14,
  MZ_ZIP_LDH_COMPRESSED_SIZE_OFS = 18,
  MZ_ZIP_LDH_DECOMPRESSED_SIZE_OFS = 22,
  MZ_ZIP_LDH_FILENAME_LEN_OFS = 26,
  MZ_ZIP_LDH_EXTRA_LEN_OFS = 28,
  // End of central directory offsets
  MZ_ZIP_ECDH_SIG_OFS = 0,
  MZ_ZIP_ECDH_NUM_THIS_DISK_OFS = 4,
  MZ_ZIP_ECDH_NUM_DISK_CDIR_OFS = 6,
  MZ_ZIP_ECDH_CDIR_NUM_ENTRIES_ON_DISK_OFS = 8,
  MZ_ZIP_ECDH_CDIR_TOTAL_ENTRIES_OFS = 10,
  MZ_ZIP_ECDH_CDIR_SIZE_OFS = 12,
  MZ_ZIP_ECDH_CDIR_OFS_OFS = 16,
  MZ_ZIP_ECDH_COMMENT_SIZE_OFS = 20,
};

typedef struct {
  void *m_p;
  size_t m_size, m_capacity;
  mz_uint m_element_size;
} mz_zip_array;

struct mz_zip_internal_state_tag {
  mz_zip_array m_central_dir;
  mz_zip_array m_central_dir_offsets;
  mz_zip_array m_sorted_central_dir_offsets;
  MZ_FILE *m_pFile;
  void *m_pMem;
  size_t m_mem_size;
  size_t m_mem_capacity;
};

#define MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(array_ptr, element_size) \
  (array_ptr)->m_element_size = element_size
#define MZ_ZIP_ARRAY_ELEMENT(array_ptr, element_type, index) \
  ((element_type *)((array_ptr)->m_p))[index]

static MZ_FORCEINLINE void mz_zip_array_clear(mz_zip_archive *pZip,
                                              mz_zip_array *pArray) {
  pZip->m_pFree(pZip->m_pAlloc_opaque, pArray->m_p);
  memset(pArray, 0, sizeof(mz_zip_array));
}

static mz_bool mz_zip_array_ensure_capacity(mz_zip_archive *pZip,
                                            mz_zip_array *pArray,
                                            size_t min_new_capacity,
                                            mz_uint growing) {
  void *pNew_p;
  size_t new_capacity = min_new_capacity;
  MZ_ASSERT(pArray->m_element_size);
  if (pArray->m_capacity >= min_new_capacity) return MZ_TRUE;
  if (growing) {
    new_capacity = MZ_MAX(1, pArray->m_capacity);
    while (new_capacity < min_new_capacity) new_capacity *= 2;
  }
  if (NULL == (pNew_p = pZip->m_pRealloc(pZip->m_pAlloc_opaque, pArray->m_p,
                                         pArray->m_element_size, new_capacity)))
    return MZ_FALSE;
  pArray->m_p = pNew_p;
  pArray->m_capacity = new_capacity;
  return MZ_TRUE;
}

static MZ_FORCEINLINE mz_bool mz_zip_array_reserve(mz_zip_archive *pZip,
                                                   mz_zip_array *pArray,
                                                   size_t new_capacity,
                                                   mz_uint growing) {
  if (new_capacity > pArray->m_capacity) {
    if (!mz_zip_array_ensure_capacity(pZip, pArray, new_capacity, growing))
      return MZ_FALSE;
  }
  return MZ_TRUE;
}

static MZ_FORCEINLINE mz_bool mz_zip_array_resize(mz_zip_archive *pZip,
                                                  mz_zip_array *pArray,
                                                  size_t new_size,
                                                  mz_uint growing) {
  if (new_size > pArray->m_capacity) {
    if (!mz_zip_array_ensure_capacity(pZip, pArray, new_size, growing))
      return MZ_FALSE;
  }
  pArray->m_size = new_size;
  return MZ_TRUE;
}

static MZ_FORCEINLINE mz_bool mz_zip_array_ensure_room(mz_zip_archive *pZip,
                                                       mz_zip_array *pArray,
                                                       size_t n) {
  return mz_zip_array_reserve(pZip, pArray, pArray->m_size + n, MZ_TRUE);
}

static MZ_FORCEINLINE mz_bool mz_zip_array_push_back(mz_zip_archive *pZip,
                                                     mz_zip_array *pArray,
                                                     const void *pElements,
                                                     size_t n) {
  size_t orig_size = pArray->m_size;
  if (!mz_zip_array_resize(pZip, pArray, orig_size + n, MZ_TRUE))
    return MZ_FALSE;
  memcpy((mz_uint8 *)pArray->m_p + orig_size * pArray->m_element_size,
         pElements, n * pArray->m_element_size);
  return MZ_TRUE;
}

#ifndef MINIZ_NO_TIME
static time_t mz_zip_dos_to_time_t(int dos_time, int dos_date) {
  struct tm tm;
  memset(&tm, 0, sizeof(tm));
  tm.tm_isdst = -1;
  tm.tm_year = ((dos_date >> 9) & 127) + 1980 - 1900;
  tm.tm_mon = ((dos_date >> 5) & 15) - 1;
  tm.tm_mday = dos_date & 31;
  tm.tm_hour = (dos_time >> 11) & 31;
  tm.tm_min = (dos_time >> 5) & 63;
  tm.tm_sec = (dos_time << 1) & 62;
  return mktime(&tm);
}

static void mz_zip_time_to_dos_time(time_t time, mz_uint16 *pDOS_time,
                                    mz_uint16 *pDOS_date) {
#ifdef _MSC_VER
  struct tm tm_struct;
  struct tm *tm = &tm_struct;
  errno_t err = localtime_s(tm, &time);
  if (err) {
    *pDOS_date = 0;
    *pDOS_time = 0;
    return;
  }
#else
  struct tm *tm = localtime(&time);
#endif
  *pDOS_time = (mz_uint16)(((tm->tm_hour) << 11) + ((tm->tm_min) << 5) +
                           ((tm->tm_sec) >> 1));
  *pDOS_date = (mz_uint16)(((tm->tm_year + 1900 - 1980) << 9) +
                           ((tm->tm_mon + 1) << 5) + tm->tm_mday);
}
#endif

#ifndef MINIZ_NO_STDIO
static mz_bool mz_zip_get_file_modified_time(const char *pFilename,
                                             mz_uint16 *pDOS_time,
                                             mz_uint16 *pDOS_date) {
#ifdef MINIZ_NO_TIME
  (void)pFilename;
  *pDOS_date = *pDOS_time = 0;
#else
  struct MZ_FILE_STAT_STRUCT file_stat;
  // On Linux with x86 glibc, this call will fail on large files (>= 0x80000000
  // bytes) unless you compiled with _LARGEFILE64_SOURCE. Argh.
  if (MZ_FILE_STAT(pFilename, &file_stat) != 0) return MZ_FALSE;
  mz_zip_time_to_dos_time(file_stat.st_mtime, pDOS_time, pDOS_date);
#endif  // #ifdef MINIZ_NO_TIME
  return MZ_TRUE;
}

#ifndef MINIZ_NO_TIME
static mz_bool mz_zip_set_file_times(const char *pFilename, time_t access_time,
                                     time_t modified_time) {
  struct utimbuf t;
  t.actime = access_time;
  t.modtime = modified_time;
  return !utime(pFilename, &t);
}
#endif  // #ifndef MINIZ_NO_TIME
#endif  // #ifndef MINIZ_NO_STDIO

static mz_bool mz_zip_reader_init_internal(mz_zip_archive *pZip,
                                           mz_uint32 flags) {
  (void)flags;
  if ((!pZip) || (pZip->m_pState) || (pZip->m_zip_mode != MZ_ZIP_MODE_INVALID))
    return MZ_FALSE;

  if (!pZip->m_pAlloc) pZip->m_pAlloc = def_alloc_func;
  if (!pZip->m_pFree) pZip->m_pFree = def_free_func;
  if (!pZip->m_pRealloc) pZip->m_pRealloc = def_realloc_func;

  pZip->m_zip_mode = MZ_ZIP_MODE_READING;
  pZip->m_archive_size = 0;
  pZip->m_central_directory_file_ofs = 0;
  pZip->m_total_files = 0;

  if (NULL == (pZip->m_pState = (mz_zip_internal_state *)pZip->m_pAlloc(
                   pZip->m_pAlloc_opaque, 1, sizeof(mz_zip_internal_state))))
    return MZ_FALSE;
  memset(pZip->m_pState, 0, sizeof(mz_zip_internal_state));
  MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(&pZip->m_pState->m_central_dir,
                                sizeof(mz_uint8));
  MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(&pZip->m_pState->m_central_dir_offsets,
                                sizeof(mz_uint32));
  MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(&pZip->m_pState->m_sorted_central_dir_offsets,
                                sizeof(mz_uint32));
  return MZ_TRUE;
}

static MZ_FORCEINLINE mz_bool
mz_zip_reader_filename_less(const mz_zip_array *pCentral_dir_array,
                            const mz_zip_array *pCentral_dir_offsets,
                            mz_uint l_index, mz_uint r_index) {
  const mz_uint8 *pL = &MZ_ZIP_ARRAY_ELEMENT(
                     pCentral_dir_array, mz_uint8,
                     MZ_ZIP_ARRAY_ELEMENT(pCentral_dir_offsets, mz_uint32,
                                          l_index)),
                 *pE;
  const mz_uint8 *pR = &MZ_ZIP_ARRAY_ELEMENT(
      pCentral_dir_array, mz_uint8,
      MZ_ZIP_ARRAY_ELEMENT(pCentral_dir_offsets, mz_uint32, r_index));
  mz_uint l_len = MZ_READ_LE16(pL + MZ_ZIP_CDH_FILENAME_LEN_OFS),
          r_len = MZ_READ_LE16(pR + MZ_ZIP_CDH_FILENAME_LEN_OFS);
  mz_uint8 l = 0, r = 0;
  pL += MZ_ZIP_CENTRAL_DIR_HEADER_SIZE;
  pR += MZ_ZIP_CENTRAL_DIR_HEADER_SIZE;
  pE = pL + MZ_MIN(l_len, r_len);
  while (pL < pE) {
    if ((l = MZ_TOLOWER(*pL)) != (r = MZ_TOLOWER(*pR))) break;
    pL++;
    pR++;
  }
  return (pL == pE) ? (l_len < r_len) : (l < r);
}

#define MZ_SWAP_UINT32(a, b) \
  do {                       \
    mz_uint32 t = a;         \
    a = b;                   \
    b = t;                   \
  }                          \
  MZ_MACRO_END

// Heap sort of lowercased filenames, used to help accelerate plain central
// directory searches by mz_zip_reader_locate_file(). (Could also use qsort(),
// but it could allocate memory.)
static void mz_zip_reader_sort_central_dir_offsets_by_filename(
    mz_zip_archive *pZip) {
  mz_zip_internal_state *pState = pZip->m_pState;
  const mz_zip_array *pCentral_dir_offsets = &pState->m_central_dir_offsets;
  const mz_zip_array *pCentral_dir = &pState->m_central_dir;
  mz_uint32 *pIndices = &MZ_ZIP_ARRAY_ELEMENT(
      &pState->m_sorted_central_dir_offsets, mz_uint32, 0);
  const int size = pZip->m_total_files;
  int start = (size - 2) >> 1, end;
  while (start >= 0) {
    int child, root = start;
    for (;;) {
      if ((child = (root << 1) + 1) >= size) break;
      child +=
          (((child + 1) < size) &&
           (mz_zip_reader_filename_less(pCentral_dir, pCentral_dir_offsets,
                                        pIndices[child], pIndices[child + 1])));
      if (!mz_zip_reader_filename_less(pCentral_dir, pCentral_dir_offsets,
                                       pIndices[root], pIndices[child]))
        break;
      MZ_SWAP_UINT32(pIndices[root], pIndices[child]);
      root = child;
    }
    start--;
  }

  end = size - 1;
  while (end > 0) {
    int child, root = 0;
    MZ_SWAP_UINT32(pIndices[end], pIndices[0]);
    for (;;) {
      if ((child = (root << 1) + 1) >= end) break;
      child +=
          (((child + 1) < end) &&
           mz_zip_reader_filename_less(pCentral_dir, pCentral_dir_offsets,
                                       pIndices[child], pIndices[child + 1]));
      if (!mz_zip_reader_filename_less(pCentral_dir, pCentral_dir_offsets,
                                       pIndices[root], pIndices[child]))
        break;
      MZ_SWAP_UINT32(pIndices[root], pIndices[child]);
      root = child;
    }
    end--;
  }
}

static mz_bool mz_zip_reader_read_central_dir(mz_zip_archive *pZip,
                                              mz_uint32 flags) {
  mz_uint cdir_size, num_this_disk, cdir_disk_index;
  mz_uint64 cdir_ofs;
  mz_int64 cur_file_ofs;
  const mz_uint8 *p;
  mz_uint32 buf_u32[4096 / sizeof(mz_uint32)];
  mz_uint8 *pBuf = (mz_uint8 *)buf_u32;
  mz_bool sort_central_dir =
      ((flags & MZ_ZIP_FLAG_DO_NOT_SORT_CENTRAL_DIRECTORY) == 0);
  // Basic sanity checks - reject files which are too small, and check the first
  // 4 bytes of the file to make sure a local header is there.
  if (pZip->m_archive_size < MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE)
    return MZ_FALSE;
  // Find the end of central directory record by scanning the file from the end
  // towards the beginning.
  cur_file_ofs =
      MZ_MAX((mz_int64)pZip->m_archive_size - (mz_int64)sizeof(buf_u32), 0);
  for (;;) {
    int i,
        n = (int)MZ_MIN(sizeof(buf_u32), pZip->m_archive_size - cur_file_ofs);
    if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pBuf, n) != (mz_uint)n)
      return MZ_FALSE;
    for (i = n - 4; i >= 0; --i)
      if (MZ_READ_LE32(pBuf + i) == MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIG) break;
    if (i >= 0) {
      cur_file_ofs += i;
      break;
    }
    if ((!cur_file_ofs) || ((pZip->m_archive_size - cur_file_ofs) >=
                            (0xFFFF + MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE)))
      return MZ_FALSE;
    cur_file_ofs = MZ_MAX(cur_file_ofs - (sizeof(buf_u32) - 3), 0);
  }
  // Read and verify the end of central directory record.
  if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pBuf,
                    MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE) !=
      MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE)
    return MZ_FALSE;
  if ((MZ_READ_LE32(pBuf + MZ_ZIP_ECDH_SIG_OFS) !=
       MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIG) ||
      ((pZip->m_total_files =
            MZ_READ_LE16(pBuf + MZ_ZIP_ECDH_CDIR_TOTAL_ENTRIES_OFS)) !=
       MZ_READ_LE16(pBuf + MZ_ZIP_ECDH_CDIR_NUM_ENTRIES_ON_DISK_OFS)))
    return MZ_FALSE;

  num_this_disk = MZ_READ_LE16(pBuf + MZ_ZIP_ECDH_NUM_THIS_DISK_OFS);
  cdir_disk_index = MZ_READ_LE16(pBuf + MZ_ZIP_ECDH_NUM_DISK_CDIR_OFS);
  if (((num_this_disk | cdir_disk_index) != 0) &&
      ((num_this_disk != 1) || (cdir_disk_index != 1)))
    return MZ_FALSE;

  if ((cdir_size = MZ_READ_LE32(pBuf + MZ_ZIP_ECDH_CDIR_SIZE_OFS)) <
      pZip->m_total_files * MZ_ZIP_CENTRAL_DIR_HEADER_SIZE)
    return MZ_FALSE;

  cdir_ofs = MZ_READ_LE32(pBuf + MZ_ZIP_ECDH_CDIR_OFS_OFS);
  if ((cdir_ofs + (mz_uint64)cdir_size) > pZip->m_archive_size) return MZ_FALSE;

  pZip->m_central_directory_file_ofs = cdir_ofs;

  if (pZip->m_total_files) {
    mz_uint i, n;

    // Read the entire central directory into a heap block, and allocate another
    // heap block to hold the unsorted central dir file record offsets, and
    // another to hold the sorted indices.
    if ((!mz_zip_array_resize(pZip, &pZip->m_pState->m_central_dir, cdir_size,
                              MZ_FALSE)) ||
        (!mz_zip_array_resize(pZip, &pZip->m_pState->m_central_dir_offsets,
                              pZip->m_total_files, MZ_FALSE)))
      return MZ_FALSE;

    if (sort_central_dir) {
      if (!mz_zip_array_resize(pZip,
                               &pZip->m_pState->m_sorted_central_dir_offsets,
                               pZip->m_total_files, MZ_FALSE))
        return MZ_FALSE;
    }

    if (pZip->m_pRead(pZip->m_pIO_opaque, cdir_ofs,
                      pZip->m_pState->m_central_dir.m_p,
                      cdir_size) != cdir_size)
      return MZ_FALSE;

    // Now create an index into the central directory file records, do some
    // basic sanity checking on each record, and check for zip64 entries (which
    // are not yet supported).
    p = (const mz_uint8 *)pZip->m_pState->m_central_dir.m_p;
    for (n = cdir_size, i = 0; i < pZip->m_total_files; ++i) {
      mz_uint total_header_size, comp_size, decomp_size, disk_index;
      if ((n < MZ_ZIP_CENTRAL_DIR_HEADER_SIZE) ||
          (MZ_READ_LE32(p) != MZ_ZIP_CENTRAL_DIR_HEADER_SIG))
        return MZ_FALSE;
      MZ_ZIP_ARRAY_ELEMENT(&pZip->m_pState->m_central_dir_offsets, mz_uint32,
                           i) =
          (mz_uint32)(p - (const mz_uint8 *)pZip->m_pState->m_central_dir.m_p);
      if (sort_central_dir)
        MZ_ZIP_ARRAY_ELEMENT(&pZip->m_pState->m_sorted_central_dir_offsets,
                             mz_uint32, i) = i;
      comp_size = MZ_READ_LE32(p + MZ_ZIP_CDH_COMPRESSED_SIZE_OFS);
      decomp_size = MZ_READ_LE32(p + MZ_ZIP_CDH_DECOMPRESSED_SIZE_OFS);
      if (((!MZ_READ_LE32(p + MZ_ZIP_CDH_METHOD_OFS)) &&
           (decomp_size != comp_size)) ||
          (decomp_size && !comp_size) || (decomp_size == 0xFFFFFFFF) ||
          (comp_size == 0xFFFFFFFF))
        return MZ_FALSE;
      disk_index = MZ_READ_LE16(p + MZ_ZIP_CDH_DISK_START_OFS);
      if ((disk_index != num_this_disk) && (disk_index != 1)) return MZ_FALSE;
      if (((mz_uint64)MZ_READ_LE32(p + MZ_ZIP_CDH_LOCAL_HEADER_OFS) +
           MZ_ZIP_LOCAL_DIR_HEADER_SIZE + comp_size) > pZip->m_archive_size)
        return MZ_FALSE;
      if ((total_header_size = MZ_ZIP_CENTRAL_DIR_HEADER_SIZE +
                               MZ_READ_LE16(p + MZ_ZIP_CDH_FILENAME_LEN_OFS) +
                               MZ_READ_LE16(p + MZ_ZIP_CDH_EXTRA_LEN_OFS) +
                               MZ_READ_LE16(p + MZ_ZIP_CDH_COMMENT_LEN_OFS)) >
          n)
        return MZ_FALSE;
      n -= total_header_size;
      p += total_header_size;
    }
  }

  if (sort_central_dir)
    mz_zip_reader_sort_central_dir_offsets_by_filename(pZip);

  return MZ_TRUE;
}

mz_bool mz_zip_reader_init(mz_zip_archive *pZip, mz_uint64 size,
                           mz_uint32 flags) {
  if ((!pZip) || (!pZip->m_pRead)) return MZ_FALSE;
  if (!mz_zip_reader_init_internal(pZip, flags)) return MZ_FALSE;
  pZip->m_archive_size = size;
  if (!mz_zip_reader_read_central_dir(pZip, flags)) {
    mz_zip_reader_end(pZip);
    return MZ_FALSE;
  }
  return MZ_TRUE;
}

static size_t mz_zip_mem_read_func(void *pOpaque, mz_uint64 file_ofs,
                                   void *pBuf, size_t n) {
  mz_zip_archive *pZip = (mz_zip_archive *)pOpaque;
  size_t s = (file_ofs >= pZip->m_archive_size)
                 ? 0
                 : (size_t)MZ_MIN(pZip->m_archive_size - file_ofs, n);
  memcpy(pBuf, (const mz_uint8 *)pZip->m_pState->m_pMem + file_ofs, s);
  return s;
}

mz_bool mz_zip_reader_init_mem(mz_zip_archive *pZip, const void *pMem,
                               size_t size, mz_uint32 flags) {
  if (!mz_zip_reader_init_internal(pZip, flags)) return MZ_FALSE;
  pZip->m_archive_size = size;
  pZip->m_pRead = mz_zip_mem_read_func;
  pZip->m_pIO_opaque = pZip;
#ifdef __cplusplus
  pZip->m_pState->m_pMem = const_cast<void *>(pMem);
#else
  pZip->m_pState->m_pMem = (void *)pMem;
#endif
  pZip->m_pState->m_mem_size = size;
  if (!mz_zip_reader_read_central_dir(pZip, flags)) {
    mz_zip_reader_end(pZip);
    return MZ_FALSE;
  }
  return MZ_TRUE;
}

#ifndef MINIZ_NO_STDIO
static size_t mz_zip_file_read_func(void *pOpaque, mz_uint64 file_ofs,
                                    void *pBuf, size_t n) {
  mz_zip_archive *pZip = (mz_zip_archive *)pOpaque;
  mz_int64 cur_ofs = MZ_FTELL64(pZip->m_pState->m_pFile);
  if (((mz_int64)file_ofs < 0) ||
      (((cur_ofs != (mz_int64)file_ofs)) &&
       (MZ_FSEEK64(pZip->m_pState->m_pFile, (mz_int64)file_ofs, SEEK_SET))))
    return 0;
  return MZ_FREAD(pBuf, 1, n, pZip->m_pState->m_pFile);
}

mz_bool mz_zip_reader_init_file(mz_zip_archive *pZip, const char *pFilename,
                                mz_uint32 flags) {
  mz_uint64 file_size;
  MZ_FILE *pFile = MZ_FOPEN(pFilename, "rb");
  if (!pFile) return MZ_FALSE;
  if (MZ_FSEEK64(pFile, 0, SEEK_END)) {
    MZ_FCLOSE(pFile);
    return MZ_FALSE;
  }
  file_size = MZ_FTELL64(pFile);
  if (!mz_zip_reader_init_internal(pZip, flags)) {
    MZ_FCLOSE(pFile);
    return MZ_FALSE;
  }
  pZip->m_pRead = mz_zip_file_read_func;
  pZip->m_pIO_opaque = pZip;
  pZip->m_pState->m_pFile = pFile;
  pZip->m_archive_size = file_size;
  if (!mz_zip_reader_read_central_dir(pZip, flags)) {
    mz_zip_reader_end(pZip);
    return MZ_FALSE;
  }
  return MZ_TRUE;
}
#endif  // #ifndef MINIZ_NO_STDIO

mz_uint mz_zip_reader_get_num_files(mz_zip_archive *pZip) {
  return pZip ? pZip->m_total_files : 0;
}

static MZ_FORCEINLINE const mz_uint8 *mz_zip_reader_get_cdh(
    mz_zip_archive *pZip, mz_uint file_index) {
  if ((!pZip) || (!pZip->m_pState) || (file_index >= pZip->m_total_files) ||
      (pZip->m_zip_mode != MZ_ZIP_MODE_READING))
    return NULL;
  return &MZ_ZIP_ARRAY_ELEMENT(
      &pZip->m_pState->m_central_dir, mz_uint8,
      MZ_ZIP_ARRAY_ELEMENT(&pZip->m_pState->m_central_dir_offsets, mz_uint32,
                           file_index));
}

mz_bool mz_zip_reader_is_file_encrypted(mz_zip_archive *pZip,
                                        mz_uint file_index) {
  mz_uint m_bit_flag;
  const mz_uint8 *p = mz_zip_reader_get_cdh(pZip, file_index);
  if (!p) return MZ_FALSE;
  m_bit_flag = MZ_READ_LE16(p + MZ_ZIP_CDH_BIT_FLAG_OFS);
  return (m_bit_flag & 1);
}

mz_bool mz_zip_reader_is_file_a_directory(mz_zip_archive *pZip,
                                          mz_uint file_index) {
  mz_uint filename_len, external_attr;
  const mz_uint8 *p = mz_zip_reader_get_cdh(pZip, file_index);
  if (!p) return MZ_FALSE;

  // First see if the filename ends with a '/' character.
  filename_len = MZ_READ_LE16(p + MZ_ZIP_CDH_FILENAME_LEN_OFS);
  if (filename_len) {
    if (*(p + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + filename_len - 1) == '/')
      return MZ_TRUE;
  }

  // Bugfix: This code was also checking if the internal attribute was non-zero,
  // which wasn't correct.
  // Most/all zip writers (hopefully) set DOS file/directory attributes in the
  // low 16-bits, so check for the DOS directory flag and ignore the source OS
  // ID in the created by field.
  // FIXME: Remove this check? Is it necessary - we already check the filename.
  external_attr = MZ_READ_LE32(p + MZ_ZIP_CDH_EXTERNAL_ATTR_OFS);
  if ((external_attr & 0x10) != 0) return MZ_TRUE;

  return MZ_FALSE;
}

mz_bool mz_zip_reader_file_stat(mz_zip_archive *pZip, mz_uint file_index,
                                mz_zip_archive_file_stat *pStat) {
  mz_uint n;
  const mz_uint8 *p = mz_zip_reader_get_cdh(pZip, file_index);
  if ((!p) || (!pStat)) return MZ_FALSE;

  // Unpack the central directory record.
  pStat->m_file_index = file_index;
  pStat->m_central_dir_ofs = MZ_ZIP_ARRAY_ELEMENT(
      &pZip->m_pState->m_central_dir_offsets, mz_uint32, file_index);
  pStat->m_version_made_by = MZ_READ_LE16(p + MZ_ZIP_CDH_VERSION_MADE_BY_OFS);
  pStat->m_version_needed = MZ_READ_LE16(p + MZ_ZIP_CDH_VERSION_NEEDED_OFS);
  pStat->m_bit_flag = MZ_READ_LE16(p + MZ_ZIP_CDH_BIT_FLAG_OFS);
  pStat->m_method = MZ_READ_LE16(p + MZ_ZIP_CDH_METHOD_OFS);
#ifndef MINIZ_NO_TIME
  pStat->m_time =
      mz_zip_dos_to_time_t(MZ_READ_LE16(p + MZ_ZIP_CDH_FILE_TIME_OFS),
                           MZ_READ_LE16(p + MZ_ZIP_CDH_FILE_DATE_OFS));
#endif
  pStat->m_crc32 = MZ_READ_LE32(p + MZ_ZIP_CDH_CRC32_OFS);
  pStat->m_comp_size = MZ_READ_LE32(p + MZ_ZIP_CDH_COMPRESSED_SIZE_OFS);
  pStat->m_uncomp_size = MZ_READ_LE32(p + MZ_ZIP_CDH_DECOMPRESSED_SIZE_OFS);
  pStat->m_internal_attr = MZ_READ_LE16(p + MZ_ZIP_CDH_INTERNAL_ATTR_OFS);
  pStat->m_external_attr = MZ_READ_LE32(p + MZ_ZIP_CDH_EXTERNAL_ATTR_OFS);
  pStat->m_local_header_ofs = MZ_READ_LE32(p + MZ_ZIP_CDH_LOCAL_HEADER_OFS);

  // Copy as much of the filename and comment as possible.
  n = MZ_READ_LE16(p + MZ_ZIP_CDH_FILENAME_LEN_OFS);
  n = MZ_MIN(n, MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE - 1);
  memcpy(pStat->m_filename, p + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE, n);
  pStat->m_filename[n] = '\0';

  n = MZ_READ_LE16(p + MZ_ZIP_CDH_COMMENT_LEN_OFS);
  n = MZ_MIN(n, MZ_ZIP_MAX_ARCHIVE_FILE_COMMENT_SIZE - 1);
  pStat->m_comment_size = n;
  memcpy(pStat->m_comment,
         p + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE +
             MZ_READ_LE16(p + MZ_ZIP_CDH_FILENAME_LEN_OFS) +
             MZ_READ_LE16(p + MZ_ZIP_CDH_EXTRA_LEN_OFS),
         n);
  pStat->m_comment[n] = '\0';

  return MZ_TRUE;
}

mz_uint mz_zip_reader_get_filename(mz_zip_archive *pZip, mz_uint file_index,
                                   char *pFilename, mz_uint filename_buf_size) {
  mz_uint n;
  const mz_uint8 *p = mz_zip_reader_get_cdh(pZip, file_index);
  if (!p) {
    if (filename_buf_size) pFilename[0] = '\0';
    return 0;
  }
  n = MZ_READ_LE16(p + MZ_ZIP_CDH_FILENAME_LEN_OFS);
  if (filename_buf_size) {
    n = MZ_MIN(n, filename_buf_size - 1);
    memcpy(pFilename, p + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE, n);
    pFilename[n] = '\0';
  }
  return n + 1;
}

static MZ_FORCEINLINE mz_bool mz_zip_reader_string_equal(const char *pA,
                                                         const char *pB,
                                                         mz_uint len,
                                                         mz_uint flags) {
  mz_uint i;
  if (flags & MZ_ZIP_FLAG_CASE_SENSITIVE) return 0 == memcmp(pA, pB, len);
  for (i = 0; i < len; ++i)
    if (MZ_TOLOWER(pA[i]) != MZ_TOLOWER(pB[i])) return MZ_FALSE;
  return MZ_TRUE;
}

static MZ_FORCEINLINE int mz_zip_reader_filename_compare(
    const mz_zip_array *pCentral_dir_array,
    const mz_zip_array *pCentral_dir_offsets, mz_uint l_index, const char *pR,
    mz_uint r_len) {
  const mz_uint8 *pL = &MZ_ZIP_ARRAY_ELEMENT(
                     pCentral_dir_array, mz_uint8,
                     MZ_ZIP_ARRAY_ELEMENT(pCentral_dir_offsets, mz_uint32,
                                          l_index)),
                 *pE;
  mz_uint l_len = MZ_READ_LE16(pL + MZ_ZIP_CDH_FILENAME_LEN_OFS);
  mz_uint8 l = 0, r = 0;
  pL += MZ_ZIP_CENTRAL_DIR_HEADER_SIZE;
  pE = pL + MZ_MIN(l_len, r_len);
  while (pL < pE) {
    if ((l = MZ_TOLOWER(*pL)) != (r = MZ_TOLOWER(*pR))) break;
    pL++;
    pR++;
  }
  return (pL == pE) ? (int)(l_len - r_len) : (l - r);
}

static int mz_zip_reader_locate_file_binary_search(mz_zip_archive *pZip,
                                                   const char *pFilename) {
  mz_zip_internal_state *pState = pZip->m_pState;
  const mz_zip_array *pCentral_dir_offsets = &pState->m_central_dir_offsets;
  const mz_zip_array *pCentral_dir = &pState->m_central_dir;
  mz_uint32 *pIndices = &MZ_ZIP_ARRAY_ELEMENT(
      &pState->m_sorted_central_dir_offsets, mz_uint32, 0);
  const int size = pZip->m_total_files;
  const mz_uint filename_len = (mz_uint)strlen(pFilename);
  int l = 0, h = size - 1;
  while (l <= h) {
    int m = (l + h) >> 1, file_index = pIndices[m],
        comp =
            mz_zip_reader_filename_compare(pCentral_dir, pCentral_dir_offsets,
                                           file_index, pFilename, filename_len);
    if (!comp)
      return file_index;
    else if (comp < 0)
      l = m + 1;
    else
      h = m - 1;
  }
  return -1;
}

int mz_zip_reader_locate_file(mz_zip_archive *pZip, const char *pName,
                              const char *pComment, mz_uint flags) {
  mz_uint file_index;
  size_t name_len, comment_len;
  if ((!pZip) || (!pZip->m_pState) || (!pName) ||
      (pZip->m_zip_mode != MZ_ZIP_MODE_READING))
    return -1;
  if (((flags & (MZ_ZIP_FLAG_IGNORE_PATH | MZ_ZIP_FLAG_CASE_SENSITIVE)) == 0) &&
      (!pComment) && (pZip->m_pState->m_sorted_central_dir_offsets.m_size))
    return mz_zip_reader_locate_file_binary_search(pZip, pName);
  name_len = strlen(pName);
  if (name_len > 0xFFFF) return -1;
  comment_len = pComment ? strlen(pComment) : 0;
  if (comment_len > 0xFFFF) return -1;
  for (file_index = 0; file_index < pZip->m_total_files; file_index++) {
    const mz_uint8 *pHeader = &MZ_ZIP_ARRAY_ELEMENT(
        &pZip->m_pState->m_central_dir, mz_uint8,
        MZ_ZIP_ARRAY_ELEMENT(&pZip->m_pState->m_central_dir_offsets, mz_uint32,
                             file_index));
    mz_uint filename_len = MZ_READ_LE16(pHeader + MZ_ZIP_CDH_FILENAME_LEN_OFS);
    const char *pFilename =
        (const char *)pHeader + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE;
    if (filename_len < name_len) continue;
    if (comment_len) {
      mz_uint file_extra_len = MZ_READ_LE16(pHeader + MZ_ZIP_CDH_EXTRA_LEN_OFS),
              file_comment_len =
                  MZ_READ_LE16(pHeader + MZ_ZIP_CDH_COMMENT_LEN_OFS);
      const char *pFile_comment = pFilename + filename_len + file_extra_len;
      if ((file_comment_len != comment_len) ||
          (!mz_zip_reader_string_equal(pComment, pFile_comment,
                                       file_comment_len, flags)))
        continue;
    }
    if ((flags & MZ_ZIP_FLAG_IGNORE_PATH) && (filename_len)) {
      int ofs = filename_len - 1;
      do {
        if ((pFilename[ofs] == '/') || (pFilename[ofs] == '\\') ||
            (pFilename[ofs] == ':'))
          break;
      } while (--ofs >= 0);
      ofs++;
      pFilename += ofs;
      filename_len -= ofs;
    }
    if ((filename_len == name_len) &&
        (mz_zip_reader_string_equal(pName, pFilename, filename_len, flags)))
      return file_index;
  }
  return -1;
}

mz_bool mz_zip_reader_extract_to_mem_no_alloc(mz_zip_archive *pZip,
                                              mz_uint file_index, void *pBuf,
                                              size_t buf_size, mz_uint flags,
                                              void *pUser_read_buf,
                                              size_t user_read_buf_size) {
  int status = TINFL_STATUS_DONE;
  mz_uint64 needed_size, cur_file_ofs, comp_remaining,
      out_buf_ofs = 0, read_buf_size, read_buf_ofs = 0, read_buf_avail;
  mz_zip_archive_file_stat file_stat;
  void *pRead_buf;
  mz_uint32
      local_header_u32[(MZ_ZIP_LOCAL_DIR_HEADER_SIZE + sizeof(mz_uint32) - 1) /
                       sizeof(mz_uint32)];
  mz_uint8 *pLocal_header = (mz_uint8 *)local_header_u32;
  tinfl_decompressor inflator;

  if ((buf_size) && (!pBuf)) return MZ_FALSE;

  if (!mz_zip_reader_file_stat(pZip, file_index, &file_stat)) return MZ_FALSE;

  // Empty file, or a directory (but not always a directory - I've seen odd zips
  // with directories that have compressed data which inflates to 0 bytes)
  if (!file_stat.m_comp_size) return MZ_TRUE;

  // Entry is a subdirectory (I've seen old zips with dir entries which have
  // compressed deflate data which inflates to 0 bytes, but these entries claim
  // to uncompress to 512 bytes in the headers).
  // I'm torn how to handle this case - should it fail instead?
  if (mz_zip_reader_is_file_a_directory(pZip, file_index)) return MZ_TRUE;

  // Encryption and patch files are not supported.
  if (file_stat.m_bit_flag & (1 | 32)) return MZ_FALSE;

  // This function only supports stored and deflate.
  if ((!(flags & MZ_ZIP_FLAG_COMPRESSED_DATA)) && (file_stat.m_method != 0) &&
      (file_stat.m_method != MZ_DEFLATED))
    return MZ_FALSE;

  // Ensure supplied output buffer is large enough.
  needed_size = (flags & MZ_ZIP_FLAG_COMPRESSED_DATA) ? file_stat.m_comp_size
                                                      : file_stat.m_uncomp_size;
  if (buf_size < needed_size) return MZ_FALSE;

  // Read and parse the local directory entry.
  cur_file_ofs = file_stat.m_local_header_ofs;
  if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pLocal_header,
                    MZ_ZIP_LOCAL_DIR_HEADER_SIZE) !=
      MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
    return MZ_FALSE;
  if (MZ_READ_LE32(pLocal_header) != MZ_ZIP_LOCAL_DIR_HEADER_SIG)
    return MZ_FALSE;

  cur_file_ofs += MZ_ZIP_LOCAL_DIR_HEADER_SIZE +
                  MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_FILENAME_LEN_OFS) +
                  MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_EXTRA_LEN_OFS);
  if ((cur_file_ofs + file_stat.m_comp_size) > pZip->m_archive_size)
    return MZ_FALSE;

  if ((flags & MZ_ZIP_FLAG_COMPRESSED_DATA) || (!file_stat.m_method)) {
    // The file is stored or the caller has requested the compressed data.
    if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pBuf,
                      (size_t)needed_size) != needed_size)
      return MZ_FALSE;
    return ((flags & MZ_ZIP_FLAG_COMPRESSED_DATA) != 0) ||
           (mz_crc32(MZ_CRC32_INIT, (const mz_uint8 *)pBuf,
                     (size_t)file_stat.m_uncomp_size) == file_stat.m_crc32);
  }

  // Decompress the file either directly from memory or from a file input
  // buffer.
  tinfl_init(&inflator);

  if (pZip->m_pState->m_pMem) {
    // Read directly from the archive in memory.
    pRead_buf = (mz_uint8 *)pZip->m_pState->m_pMem + cur_file_ofs;
    read_buf_size = read_buf_avail = file_stat.m_comp_size;
    comp_remaining = 0;
  } else if (pUser_read_buf) {
    // Use a user provided read buffer.
    if (!user_read_buf_size) return MZ_FALSE;
    pRead_buf = (mz_uint8 *)pUser_read_buf;
    read_buf_size = user_read_buf_size;
    read_buf_avail = 0;
    comp_remaining = file_stat.m_comp_size;
  } else {
    // Temporarily allocate a read buffer.
    read_buf_size =
        MZ_MIN(file_stat.m_comp_size, (mz_uint)MZ_ZIP_MAX_IO_BUF_SIZE);
#ifdef _MSC_VER
    if (((0, sizeof(size_t) == sizeof(mz_uint32))) &&
        (read_buf_size > 0x7FFFFFFF))
#else
    if (((sizeof(size_t) == sizeof(mz_uint32))) && (read_buf_size > 0x7FFFFFFF))
#endif
      return MZ_FALSE;
    if (NULL == (pRead_buf = pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1,
                                            (size_t)read_buf_size)))
      return MZ_FALSE;
    read_buf_avail = 0;
    comp_remaining = file_stat.m_comp_size;
  }

  do {
    size_t in_buf_size,
        out_buf_size = (size_t)(file_stat.m_uncomp_size - out_buf_ofs);
    if ((!read_buf_avail) && (!pZip->m_pState->m_pMem)) {
      read_buf_avail = MZ_MIN(read_buf_size, comp_remaining);
      if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pRead_buf,
                        (size_t)read_buf_avail) != read_buf_avail) {
        status = TINFL_STATUS_FAILED;
        break;
      }
      cur_file_ofs += read_buf_avail;
      comp_remaining -= read_buf_avail;
      read_buf_ofs = 0;
    }
    in_buf_size = (size_t)read_buf_avail;
    status = tinfl_decompress(
        &inflator, (mz_uint8 *)pRead_buf + read_buf_ofs, &in_buf_size,
        (mz_uint8 *)pBuf, (mz_uint8 *)pBuf + out_buf_ofs, &out_buf_size,
        TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF |
            (comp_remaining ? TINFL_FLAG_HAS_MORE_INPUT : 0));
    read_buf_avail -= in_buf_size;
    read_buf_ofs += in_buf_size;
    out_buf_ofs += out_buf_size;
  } while (status == TINFL_STATUS_NEEDS_MORE_INPUT);

  if (status == TINFL_STATUS_DONE) {
    // Make sure the entire file was decompressed, and check its CRC.
    if ((out_buf_ofs != file_stat.m_uncomp_size) ||
        (mz_crc32(MZ_CRC32_INIT, (const mz_uint8 *)pBuf,
                  (size_t)file_stat.m_uncomp_size) != file_stat.m_crc32))
      status = TINFL_STATUS_FAILED;
  }

  if ((!pZip->m_pState->m_pMem) && (!pUser_read_buf))
    pZip->m_pFree(pZip->m_pAlloc_opaque, pRead_buf);

  return status == TINFL_STATUS_DONE;
}

mz_bool mz_zip_reader_extract_file_to_mem_no_alloc(
    mz_zip_archive *pZip, const char *pFilename, void *pBuf, size_t buf_size,
    mz_uint flags, void *pUser_read_buf, size_t user_read_buf_size) {
  int file_index = mz_zip_reader_locate_file(pZip, pFilename, NULL, flags);
  if (file_index < 0) return MZ_FALSE;
  return mz_zip_reader_extract_to_mem_no_alloc(pZip, file_index, pBuf, buf_size,
                                               flags, pUser_read_buf,
                                               user_read_buf_size);
}

mz_bool mz_zip_reader_extract_to_mem(mz_zip_archive *pZip, mz_uint file_index,
                                     void *pBuf, size_t buf_size,
                                     mz_uint flags) {
  return mz_zip_reader_extract_to_mem_no_alloc(pZip, file_index, pBuf, buf_size,
                                               flags, NULL, 0);
}

mz_bool mz_zip_reader_extract_file_to_mem(mz_zip_archive *pZip,
                                          const char *pFilename, void *pBuf,
                                          size_t buf_size, mz_uint flags) {
  return mz_zip_reader_extract_file_to_mem_no_alloc(pZip, pFilename, pBuf,
                                                    buf_size, flags, NULL, 0);
}

void *mz_zip_reader_extract_to_heap(mz_zip_archive *pZip, mz_uint file_index,
                                    size_t *pSize, mz_uint flags) {
  mz_uint64 comp_size, uncomp_size, alloc_size;
  const mz_uint8 *p = mz_zip_reader_get_cdh(pZip, file_index);
  void *pBuf;

  if (pSize) *pSize = 0;
  if (!p) return NULL;

  comp_size = MZ_READ_LE32(p + MZ_ZIP_CDH_COMPRESSED_SIZE_OFS);
  uncomp_size = MZ_READ_LE32(p + MZ_ZIP_CDH_DECOMPRESSED_SIZE_OFS);

  alloc_size = (flags & MZ_ZIP_FLAG_COMPRESSED_DATA) ? comp_size : uncomp_size;
#ifdef _MSC_VER
  if (((0, sizeof(size_t) == sizeof(mz_uint32))) && (alloc_size > 0x7FFFFFFF))
#else
  if (((sizeof(size_t) == sizeof(mz_uint32))) && (alloc_size > 0x7FFFFFFF))
#endif
    return NULL;
  if (NULL ==
      (pBuf = pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, (size_t)alloc_size)))
    return NULL;

  if (!mz_zip_reader_extract_to_mem(pZip, file_index, pBuf, (size_t)alloc_size,
                                    flags)) {
    pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);
    return NULL;
  }

  if (pSize) *pSize = (size_t)alloc_size;
  return pBuf;
}

void *mz_zip_reader_extract_file_to_heap(mz_zip_archive *pZip,
                                         const char *pFilename, size_t *pSize,
                                         mz_uint flags) {
  int file_index = mz_zip_reader_locate_file(pZip, pFilename, NULL, flags);
  if (file_index < 0) {
    if (pSize) *pSize = 0;
    return MZ_FALSE;
  }
  return mz_zip_reader_extract_to_heap(pZip, file_index, pSize, flags);
}

mz_bool mz_zip_reader_extract_to_callback(mz_zip_archive *pZip,
                                          mz_uint file_index,
                                          mz_file_write_func pCallback,
                                          void *pOpaque, mz_uint flags) {
  int status = TINFL_STATUS_DONE;
  mz_uint file_crc32 = MZ_CRC32_INIT;
  mz_uint64 read_buf_size, read_buf_ofs = 0, read_buf_avail, comp_remaining,
                           out_buf_ofs = 0, cur_file_ofs;
  mz_zip_archive_file_stat file_stat;
  void *pRead_buf = NULL;
  void *pWrite_buf = NULL;
  mz_uint32
      local_header_u32[(MZ_ZIP_LOCAL_DIR_HEADER_SIZE + sizeof(mz_uint32) - 1) /
                       sizeof(mz_uint32)];
  mz_uint8 *pLocal_header = (mz_uint8 *)local_header_u32;

  if (!mz_zip_reader_file_stat(pZip, file_index, &file_stat)) return MZ_FALSE;

  // Empty file, or a directory (but not always a directory - I've seen odd zips
  // with directories that have compressed data which inflates to 0 bytes)
  if (!file_stat.m_comp_size) return MZ_TRUE;

  // Entry is a subdirectory (I've seen old zips with dir entries which have
  // compressed deflate data which inflates to 0 bytes, but these entries claim
  // to uncompress to 512 bytes in the headers).
  // I'm torn how to handle this case - should it fail instead?
  if (mz_zip_reader_is_file_a_directory(pZip, file_index)) return MZ_TRUE;

  // Encryption and patch files are not supported.
  if (file_stat.m_bit_flag & (1 | 32)) return MZ_FALSE;

  // This function only supports stored and deflate.
  if ((!(flags & MZ_ZIP_FLAG_COMPRESSED_DATA)) && (file_stat.m_method != 0) &&
      (file_stat.m_method != MZ_DEFLATED))
    return MZ_FALSE;

  // Read and parse the local directory entry.
  cur_file_ofs = file_stat.m_local_header_ofs;
  if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pLocal_header,
                    MZ_ZIP_LOCAL_DIR_HEADER_SIZE) !=
      MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
    return MZ_FALSE;
  if (MZ_READ_LE32(pLocal_header) != MZ_ZIP_LOCAL_DIR_HEADER_SIG)
    return MZ_FALSE;

  cur_file_ofs += MZ_ZIP_LOCAL_DIR_HEADER_SIZE +
                  MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_FILENAME_LEN_OFS) +
                  MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_EXTRA_LEN_OFS);
  if ((cur_file_ofs + file_stat.m_comp_size) > pZip->m_archive_size)
    return MZ_FALSE;

  // Decompress the file either directly from memory or from a file input
  // buffer.
  if (pZip->m_pState->m_pMem) {
    pRead_buf = (mz_uint8 *)pZip->m_pState->m_pMem + cur_file_ofs;
    read_buf_size = read_buf_avail = file_stat.m_comp_size;
    comp_remaining = 0;
  } else {
    read_buf_size =
        MZ_MIN(file_stat.m_comp_size, (mz_uint)MZ_ZIP_MAX_IO_BUF_SIZE);
    if (NULL == (pRead_buf = pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1,
                                            (size_t)read_buf_size)))
      return MZ_FALSE;
    read_buf_avail = 0;
    comp_remaining = file_stat.m_comp_size;
  }

  if ((flags & MZ_ZIP_FLAG_COMPRESSED_DATA) || (!file_stat.m_method)) {
    // The file is stored or the caller has requested the compressed data.
    if (pZip->m_pState->m_pMem) {
#ifdef _MSC_VER
      if (((0, sizeof(size_t) == sizeof(mz_uint32))) &&
          (file_stat.m_comp_size > 0xFFFFFFFF))
#else
      if (((sizeof(size_t) == sizeof(mz_uint32))) &&
          (file_stat.m_comp_size > 0xFFFFFFFF))
#endif
        return MZ_FALSE;
      if (pCallback(pOpaque, out_buf_ofs, pRead_buf,
                    (size_t)file_stat.m_comp_size) != file_stat.m_comp_size)
        status = TINFL_STATUS_FAILED;
      else if (!(flags & MZ_ZIP_FLAG_COMPRESSED_DATA))
        file_crc32 =
            (mz_uint32)mz_crc32(file_crc32, (const mz_uint8 *)pRead_buf,
                                (size_t)file_stat.m_comp_size);
      cur_file_ofs += file_stat.m_comp_size;
      out_buf_ofs += file_stat.m_comp_size;
      comp_remaining = 0;
    } else {
      while (comp_remaining) {
        read_buf_avail = MZ_MIN(read_buf_size, comp_remaining);
        if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pRead_buf,
                          (size_t)read_buf_avail) != read_buf_avail) {
          status = TINFL_STATUS_FAILED;
          break;
        }

        if (!(flags & MZ_ZIP_FLAG_COMPRESSED_DATA))
          file_crc32 = (mz_uint32)mz_crc32(
              file_crc32, (const mz_uint8 *)pRead_buf, (size_t)read_buf_avail);

        if (pCallback(pOpaque, out_buf_ofs, pRead_buf,
                      (size_t)read_buf_avail) != read_buf_avail) {
          status = TINFL_STATUS_FAILED;
          break;
        }
        cur_file_ofs += read_buf_avail;
        out_buf_ofs += read_buf_avail;
        comp_remaining -= read_buf_avail;
      }
    }
  } else {
    tinfl_decompressor inflator;
    tinfl_init(&inflator);

    if (NULL == (pWrite_buf = pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1,
                                             TINFL_LZ_DICT_SIZE)))
      status = TINFL_STATUS_FAILED;
    else {
      do {
        mz_uint8 *pWrite_buf_cur =
            (mz_uint8 *)pWrite_buf + (out_buf_ofs & (TINFL_LZ_DICT_SIZE - 1));
        size_t in_buf_size,
            out_buf_size =
                TINFL_LZ_DICT_SIZE - (out_buf_ofs & (TINFL_LZ_DICT_SIZE - 1));
        if ((!read_buf_avail) && (!pZip->m_pState->m_pMem)) {
          read_buf_avail = MZ_MIN(read_buf_size, comp_remaining);
          if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pRead_buf,
                            (size_t)read_buf_avail) != read_buf_avail) {
            status = TINFL_STATUS_FAILED;
            break;
          }
          cur_file_ofs += read_buf_avail;
          comp_remaining -= read_buf_avail;
          read_buf_ofs = 0;
        }

        in_buf_size = (size_t)read_buf_avail;
        status = tinfl_decompress(
            &inflator, (const mz_uint8 *)pRead_buf + read_buf_ofs, &in_buf_size,
            (mz_uint8 *)pWrite_buf, pWrite_buf_cur, &out_buf_size,
            comp_remaining ? TINFL_FLAG_HAS_MORE_INPUT : 0);
        read_buf_avail -= in_buf_size;
        read_buf_ofs += in_buf_size;

        if (out_buf_size) {
          if (pCallback(pOpaque, out_buf_ofs, pWrite_buf_cur, out_buf_size) !=
              out_buf_size) {
            status = TINFL_STATUS_FAILED;
            break;
          }
          file_crc32 =
              (mz_uint32)mz_crc32(file_crc32, pWrite_buf_cur, out_buf_size);
          if ((out_buf_ofs += out_buf_size) > file_stat.m_uncomp_size) {
            status = TINFL_STATUS_FAILED;
            break;
          }
        }
      } while ((status == TINFL_STATUS_NEEDS_MORE_INPUT) ||
               (status == TINFL_STATUS_HAS_MORE_OUTPUT));
    }
  }

  if ((status == TINFL_STATUS_DONE) &&
      (!(flags & MZ_ZIP_FLAG_COMPRESSED_DATA))) {
    // Make sure the entire file was decompressed, and check its CRC.
    if ((out_buf_ofs != file_stat.m_uncomp_size) ||
        (file_crc32 != file_stat.m_crc32))
      status = TINFL_STATUS_FAILED;
  }

  if (!pZip->m_pState->m_pMem) pZip->m_pFree(pZip->m_pAlloc_opaque, pRead_buf);
  if (pWrite_buf) pZip->m_pFree(pZip->m_pAlloc_opaque, pWrite_buf);

  return status == TINFL_STATUS_DONE;
}

mz_bool mz_zip_reader_extract_file_to_callback(mz_zip_archive *pZip,
                                               const char *pFilename,
                                               mz_file_write_func pCallback,
                                               void *pOpaque, mz_uint flags) {
  int file_index = mz_zip_reader_locate_file(pZip, pFilename, NULL, flags);
  if (file_index < 0) return MZ_FALSE;
  return mz_zip_reader_extract_to_callback(pZip, file_index, pCallback, pOpaque,
                                           flags);
}

#ifndef MINIZ_NO_STDIO
static size_t mz_zip_file_write_callback(void *pOpaque, mz_uint64 ofs,
                                         const void *pBuf, size_t n) {
  (void)ofs;
  return MZ_FWRITE(pBuf, 1, n, (MZ_FILE *)pOpaque);
}

mz_bool mz_zip_reader_extract_to_file(mz_zip_archive *pZip, mz_uint file_index,
                                      const char *pDst_filename,
                                      mz_uint flags) {
  mz_bool status;
  mz_zip_archive_file_stat file_stat;
  MZ_FILE *pFile;
  if (!mz_zip_reader_file_stat(pZip, file_index, &file_stat)) return MZ_FALSE;
  pFile = MZ_FOPEN(pDst_filename, "wb");
  if (!pFile) return MZ_FALSE;
  status = mz_zip_reader_extract_to_callback(
      pZip, file_index, mz_zip_file_write_callback, pFile, flags);
  if (MZ_FCLOSE(pFile) == EOF) return MZ_FALSE;
#ifndef MINIZ_NO_TIME
  if (status)
    mz_zip_set_file_times(pDst_filename, file_stat.m_time, file_stat.m_time);
#endif
  return status;
}
#endif  // #ifndef MINIZ_NO_STDIO

mz_bool mz_zip_reader_end(mz_zip_archive *pZip) {
  if ((!pZip) || (!pZip->m_pState) || (!pZip->m_pAlloc) || (!pZip->m_pFree) ||
      (pZip->m_zip_mode != MZ_ZIP_MODE_READING))
    return MZ_FALSE;

  if (pZip->m_pState) {
    mz_zip_internal_state *pState = pZip->m_pState;
    pZip->m_pState = NULL;
    mz_zip_array_clear(pZip, &pState->m_central_dir);
    mz_zip_array_clear(pZip, &pState->m_central_dir_offsets);
    mz_zip_array_clear(pZip, &pState->m_sorted_central_dir_offsets);

#ifndef MINIZ_NO_STDIO
    if (pState->m_pFile) {
      MZ_FCLOSE(pState->m_pFile);
      pState->m_pFile = NULL;
    }
#endif  // #ifndef MINIZ_NO_STDIO

    pZip->m_pFree(pZip->m_pAlloc_opaque, pState);
  }
  pZip->m_zip_mode = MZ_ZIP_MODE_INVALID;

  return MZ_TRUE;
}

#ifndef MINIZ_NO_STDIO
mz_bool mz_zip_reader_extract_file_to_file(mz_zip_archive *pZip,
                                           const char *pArchive_filename,
                                           const char *pDst_filename,
                                           mz_uint flags) {
  int file_index =
      mz_zip_reader_locate_file(pZip, pArchive_filename, NULL, flags);
  if (file_index < 0) return MZ_FALSE;
  return mz_zip_reader_extract_to_file(pZip, file_index, pDst_filename, flags);
}
#endif

// ------------------- .ZIP archive writing

#ifndef MINIZ_NO_ARCHIVE_WRITING_APIS

static void mz_write_le16(mz_uint8 *p, mz_uint16 v) {
  p[0] = (mz_uint8)v;
  p[1] = (mz_uint8)(v >> 8);
}
static void mz_write_le32(mz_uint8 *p, mz_uint32 v) {
  p[0] = (mz_uint8)v;
  p[1] = (mz_uint8)(v >> 8);
  p[2] = (mz_uint8)(v >> 16);
  p[3] = (mz_uint8)(v >> 24);
}
#define MZ_WRITE_LE16(p, v) mz_write_le16((mz_uint8 *)(p), (mz_uint16)(v))
#define MZ_WRITE_LE32(p, v) mz_write_le32((mz_uint8 *)(p), (mz_uint32)(v))

mz_bool mz_zip_writer_init(mz_zip_archive *pZip, mz_uint64 existing_size) {
  if ((!pZip) || (pZip->m_pState) || (!pZip->m_pWrite) ||
      (pZip->m_zip_mode != MZ_ZIP_MODE_INVALID))
    return MZ_FALSE;

  if (pZip->m_file_offset_alignment) {
    // Ensure user specified file offset alignment is a power of 2.
    if (pZip->m_file_offset_alignment & (pZip->m_file_offset_alignment - 1))
      return MZ_FALSE;
  }

  if (!pZip->m_pAlloc) pZip->m_pAlloc = def_alloc_func;
  if (!pZip->m_pFree) pZip->m_pFree = def_free_func;
  if (!pZip->m_pRealloc) pZip->m_pRealloc = def_realloc_func;

  pZip->m_zip_mode = MZ_ZIP_MODE_WRITING;
  pZip->m_archive_size = existing_size;
  pZip->m_central_directory_file_ofs = 0;
  pZip->m_total_files = 0;

  if (NULL == (pZip->m_pState = (mz_zip_internal_state *)pZip->m_pAlloc(
                   pZip->m_pAlloc_opaque, 1, sizeof(mz_zip_internal_state))))
    return MZ_FALSE;
  memset(pZip->m_pState, 0, sizeof(mz_zip_internal_state));
  MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(&pZip->m_pState->m_central_dir,
                                sizeof(mz_uint8));
  MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(&pZip->m_pState->m_central_dir_offsets,
                                sizeof(mz_uint32));
  MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(&pZip->m_pState->m_sorted_central_dir_offsets,
                                sizeof(mz_uint32));
  return MZ_TRUE;
}

static size_t mz_zip_heap_write_func(void *pOpaque, mz_uint64 file_ofs,
                                     const void *pBuf, size_t n) {
  mz_zip_archive *pZip = (mz_zip_archive *)pOpaque;
  mz_zip_internal_state *pState = pZip->m_pState;
  mz_uint64 new_size = MZ_MAX(file_ofs + n, pState->m_mem_size);
#ifdef _MSC_VER
  if ((!n) ||
      ((0, sizeof(size_t) == sizeof(mz_uint32)) && (new_size > 0x7FFFFFFF)))
#else
  if ((!n) ||
      ((sizeof(size_t) == sizeof(mz_uint32)) && (new_size > 0x7FFFFFFF)))
#endif
    return 0;
  if (new_size > pState->m_mem_capacity) {
    void *pNew_block;
    size_t new_capacity = MZ_MAX(64, pState->m_mem_capacity);
    while (new_capacity < new_size) new_capacity *= 2;
    if (NULL == (pNew_block = pZip->m_pRealloc(
                     pZip->m_pAlloc_opaque, pState->m_pMem, 1, new_capacity)))
      return 0;
    pState->m_pMem = pNew_block;
    pState->m_mem_capacity = new_capacity;
  }
  memcpy((mz_uint8 *)pState->m_pMem + file_ofs, pBuf, n);
  pState->m_mem_size = (size_t)new_size;
  return n;
}

mz_bool mz_zip_writer_init_heap(mz_zip_archive *pZip,
                                size_t size_to_reserve_at_beginning,
                                size_t initial_allocation_size) {
  pZip->m_pWrite = mz_zip_heap_write_func;
  pZip->m_pIO_opaque = pZip;
  if (!mz_zip_writer_init(pZip, size_to_reserve_at_beginning)) return MZ_FALSE;
  if (0 != (initial_allocation_size = MZ_MAX(initial_allocation_size,
                                             size_to_reserve_at_beginning))) {
    if (NULL == (pZip->m_pState->m_pMem = pZip->m_pAlloc(
                     pZip->m_pAlloc_opaque, 1, initial_allocation_size))) {
      mz_zip_writer_end(pZip);
      return MZ_FALSE;
    }
    pZip->m_pState->m_mem_capacity = initial_allocation_size;
  }
  return MZ_TRUE;
}

#ifndef MINIZ_NO_STDIO
static size_t mz_zip_file_write_func(void *pOpaque, mz_uint64 file_ofs,
                                     const void *pBuf, size_t n) {
  mz_zip_archive *pZip = (mz_zip_archive *)pOpaque;
  mz_int64 cur_ofs = MZ_FTELL64(pZip->m_pState->m_pFile);
  if (((mz_int64)file_ofs < 0) ||
      (((cur_ofs != (mz_int64)file_ofs)) &&
       (MZ_FSEEK64(pZip->m_pState->m_pFile, (mz_int64)file_ofs, SEEK_SET))))
    return 0;
  return MZ_FWRITE(pBuf, 1, n, pZip->m_pState->m_pFile);
}

mz_bool mz_zip_writer_init_file(mz_zip_archive *pZip, const char *pFilename,
                                mz_uint64 size_to_reserve_at_beginning) {
  MZ_FILE *pFile;
  pZip->m_pWrite = mz_zip_file_write_func;
  pZip->m_pIO_opaque = pZip;
  if (!mz_zip_writer_init(pZip, size_to_reserve_at_beginning)) return MZ_FALSE;
  if (NULL == (pFile = MZ_FOPEN(pFilename, "wb"))) {
    mz_zip_writer_end(pZip);
    return MZ_FALSE;
  }
  pZip->m_pState->m_pFile = pFile;
  if (size_to_reserve_at_beginning) {
    mz_uint64 cur_ofs = 0;
    char buf[4096];
    MZ_CLEAR_OBJ(buf);
    do {
      size_t n = (size_t)MZ_MIN(sizeof(buf), size_to_reserve_at_beginning);
      if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_ofs, buf, n) != n) {
        mz_zip_writer_end(pZip);
        return MZ_FALSE;
      }
      cur_ofs += n;
      size_to_reserve_at_beginning -= n;
    } while (size_to_reserve_at_beginning);
  }
  return MZ_TRUE;
}
#endif  // #ifndef MINIZ_NO_STDIO

mz_bool mz_zip_writer_init_from_reader(mz_zip_archive *pZip,
                                       const char *pFilename) {
  mz_zip_internal_state *pState;
  if ((!pZip) || (!pZip->m_pState) || (pZip->m_zip_mode != MZ_ZIP_MODE_READING))
    return MZ_FALSE;
  // No sense in trying to write to an archive that's already at the support max
  // size
  if ((pZip->m_total_files == 0xFFFF) ||
      ((pZip->m_archive_size + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE +
        MZ_ZIP_LOCAL_DIR_HEADER_SIZE) > 0xFFFFFFFF))
    return MZ_FALSE;

  pState = pZip->m_pState;

  if (pState->m_pFile) {
#ifdef MINIZ_NO_STDIO
    pFilename;
    return MZ_FALSE;
#else
    // Archive is being read from stdio - try to reopen as writable.
    if (pZip->m_pIO_opaque != pZip) return MZ_FALSE;
    if (!pFilename) return MZ_FALSE;
    pZip->m_pWrite = mz_zip_file_write_func;
    if (NULL ==
        (pState->m_pFile = MZ_FREOPEN(pFilename, "r+b", pState->m_pFile))) {
      // The mz_zip_archive is now in a bogus state because pState->m_pFile is
      // NULL, so just close it.
      mz_zip_reader_end(pZip);
      return MZ_FALSE;
    }
#endif  // #ifdef MINIZ_NO_STDIO
  } else if (pState->m_pMem) {
    // Archive lives in a memory block. Assume it's from the heap that we can
    // resize using the realloc callback.
    if (pZip->m_pIO_opaque != pZip) return MZ_FALSE;
    pState->m_mem_capacity = pState->m_mem_size;
    pZip->m_pWrite = mz_zip_heap_write_func;
  }
  // Archive is being read via a user provided read function - make sure the
  // user has specified a write function too.
  else if (!pZip->m_pWrite)
    return MZ_FALSE;

  // Start writing new files at the archive's current central directory
  // location.
  pZip->m_archive_size = pZip->m_central_directory_file_ofs;
  pZip->m_zip_mode = MZ_ZIP_MODE_WRITING;
  pZip->m_central_directory_file_ofs = 0;

  return MZ_TRUE;
}

mz_bool mz_zip_writer_add_mem(mz_zip_archive *pZip, const char *pArchive_name,
                              const void *pBuf, size_t buf_size,
                              mz_uint level_and_flags) {
  return mz_zip_writer_add_mem_ex(pZip, pArchive_name, pBuf, buf_size, NULL, 0,
                                  level_and_flags, 0, 0);
}

typedef struct {
  mz_zip_archive *m_pZip;
  mz_uint64 m_cur_archive_file_ofs;
  mz_uint64 m_comp_size;
} mz_zip_writer_add_state;

static mz_bool mz_zip_writer_add_put_buf_callback(const void *pBuf, int len,
                                                  void *pUser) {
  mz_zip_writer_add_state *pState = (mz_zip_writer_add_state *)pUser;
  if ((int)pState->m_pZip->m_pWrite(pState->m_pZip->m_pIO_opaque,
                                    pState->m_cur_archive_file_ofs, pBuf,
                                    len) != len)
    return MZ_FALSE;
  pState->m_cur_archive_file_ofs += len;
  pState->m_comp_size += len;
  return MZ_TRUE;
}

static mz_bool mz_zip_writer_create_local_dir_header(
    mz_zip_archive *pZip, mz_uint8 *pDst, mz_uint16 filename_size,
    mz_uint16 extra_size, mz_uint64 uncomp_size, mz_uint64 comp_size,
    mz_uint32 uncomp_crc32, mz_uint16 method, mz_uint16 bit_flags,
    mz_uint16 dos_time, mz_uint16 dos_date) {
  (void)pZip;
  memset(pDst, 0, MZ_ZIP_LOCAL_DIR_HEADER_SIZE);
  MZ_WRITE_LE32(pDst + MZ_ZIP_LDH_SIG_OFS, MZ_ZIP_LOCAL_DIR_HEADER_SIG);
  MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_VERSION_NEEDED_OFS, method ? 20 : 0);
  MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_BIT_FLAG_OFS, bit_flags);
  MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_METHOD_OFS, method);
  MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_FILE_TIME_OFS, dos_time);
  MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_FILE_DATE_OFS, dos_date);
  MZ_WRITE_LE32(pDst + MZ_ZIP_LDH_CRC32_OFS, uncomp_crc32);
  MZ_WRITE_LE32(pDst + MZ_ZIP_LDH_COMPRESSED_SIZE_OFS, comp_size);
  MZ_WRITE_LE32(pDst + MZ_ZIP_LDH_DECOMPRESSED_SIZE_OFS, uncomp_size);
  MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_FILENAME_LEN_OFS, filename_size);
  MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_EXTRA_LEN_OFS, extra_size);
  return MZ_TRUE;
}

static mz_bool mz_zip_writer_create_central_dir_header(
    mz_zip_archive *pZip, mz_uint8 *pDst, mz_uint16 filename_size,
    mz_uint16 extra_size, mz_uint16 comment_size, mz_uint64 uncomp_size,
    mz_uint64 comp_size, mz_uint32 uncomp_crc32, mz_uint16 method,
    mz_uint16 bit_flags, mz_uint16 dos_time, mz_uint16 dos_date,
    mz_uint64 local_header_ofs, mz_uint32 ext_attributes) {
  (void)pZip;
  memset(pDst, 0, MZ_ZIP_CENTRAL_DIR_HEADER_SIZE);
  MZ_WRITE_LE32(pDst + MZ_ZIP_CDH_SIG_OFS, MZ_ZIP_CENTRAL_DIR_HEADER_SIG);
  MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_VERSION_NEEDED_OFS, method ? 20 : 0);
  MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_BIT_FLAG_OFS, bit_flags);
  MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_METHOD_OFS, method);
  MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_FILE_TIME_OFS, dos_time);
  MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_FILE_DATE_OFS, dos_date);
  MZ_WRITE_LE32(pDst + MZ_ZIP_CDH_CRC32_OFS, uncomp_crc32);
  MZ_WRITE_LE32(pDst + MZ_ZIP_CDH_COMPRESSED_SIZE_OFS, comp_size);
  MZ_WRITE_LE32(pDst + MZ_ZIP_CDH_DECOMPRESSED_SIZE_OFS, uncomp_size);
  MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_FILENAME_LEN_OFS, filename_size);
  MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_EXTRA_LEN_OFS, extra_size);
  MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_COMMENT_LEN_OFS, comment_size);
  MZ_WRITE_LE32(pDst + MZ_ZIP_CDH_EXTERNAL_ATTR_OFS, ext_attributes);
  MZ_WRITE_LE32(pDst + MZ_ZIP_CDH_LOCAL_HEADER_OFS, local_header_ofs);
  return MZ_TRUE;
}

static mz_bool mz_zip_writer_add_to_central_dir(
    mz_zip_archive *pZip, const char *pFilename, mz_uint16 filename_size,
    const void *pExtra, mz_uint16 extra_size, const void *pComment,
    mz_uint16 comment_size, mz_uint64 uncomp_size, mz_uint64 comp_size,
    mz_uint32 uncomp_crc32, mz_uint16 method, mz_uint16 bit_flags,
    mz_uint16 dos_time, mz_uint16 dos_date, mz_uint64 local_header_ofs,
    mz_uint32 ext_attributes) {
  mz_zip_internal_state *pState = pZip->m_pState;
  mz_uint32 central_dir_ofs = (mz_uint32)pState->m_central_dir.m_size;
  size_t orig_central_dir_size = pState->m_central_dir.m_size;
  mz_uint8 central_dir_header[MZ_ZIP_CENTRAL_DIR_HEADER_SIZE];

  // No zip64 support yet
  if ((local_header_ofs > 0xFFFFFFFF) ||
      (((mz_uint64)pState->m_central_dir.m_size +
        MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + filename_size + extra_size +
        comment_size) > 0xFFFFFFFF))
    return MZ_FALSE;

  if (!mz_zip_writer_create_central_dir_header(
          pZip, central_dir_header, filename_size, extra_size, comment_size,
          uncomp_size, comp_size, uncomp_crc32, method, bit_flags, dos_time,
          dos_date, local_header_ofs, ext_attributes))
    return MZ_FALSE;

  if ((!mz_zip_array_push_back(pZip, &pState->m_central_dir, central_dir_header,
                               MZ_ZIP_CENTRAL_DIR_HEADER_SIZE)) ||
      (!mz_zip_array_push_back(pZip, &pState->m_central_dir, pFilename,
                               filename_size)) ||
      (!mz_zip_array_push_back(pZip, &pState->m_central_dir, pExtra,
                               extra_size)) ||
      (!mz_zip_array_push_back(pZip, &pState->m_central_dir, pComment,
                               comment_size)) ||
      (!mz_zip_array_push_back(pZip, &pState->m_central_dir_offsets,
                               &central_dir_ofs, 1))) {
    // Try to push the central directory array back into its original state.
    mz_zip_array_resize(pZip, &pState->m_central_dir, orig_central_dir_size,
                        MZ_FALSE);
    return MZ_FALSE;
  }

  return MZ_TRUE;
}

static mz_bool mz_zip_writer_validate_archive_name(const char *pArchive_name) {
  // Basic ZIP archive filename validity checks: Valid filenames cannot start
  // with a forward slash, cannot contain a drive letter, and cannot use
  // DOS-style backward slashes.
  if (*pArchive_name == '/') return MZ_FALSE;
  while (*pArchive_name) {
    if ((*pArchive_name == '\\') || (*pArchive_name == ':')) return MZ_FALSE;
    pArchive_name++;
  }
  return MZ_TRUE;
}

static mz_uint mz_zip_writer_compute_padding_needed_for_file_alignment(
    mz_zip_archive *pZip) {
  mz_uint32 n;
  if (!pZip->m_file_offset_alignment) return 0;
  n = (mz_uint32)(pZip->m_archive_size & (pZip->m_file_offset_alignment - 1));
  return (pZip->m_file_offset_alignment - n) &
         (pZip->m_file_offset_alignment - 1);
}

static mz_bool mz_zip_writer_write_zeros(mz_zip_archive *pZip,
                                         mz_uint64 cur_file_ofs, mz_uint32 n) {
  char buf[4096];
  memset(buf, 0, MZ_MIN(sizeof(buf), n));
  while (n) {
    mz_uint32 s = MZ_MIN(sizeof(buf), n);
    if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_file_ofs, buf, s) != s)
      return MZ_FALSE;
    cur_file_ofs += s;
    n -= s;
  }
  return MZ_TRUE;
}

mz_bool mz_zip_writer_add_mem_ex(mz_zip_archive *pZip,
                                 const char *pArchive_name, const void *pBuf,
                                 size_t buf_size, const void *pComment,
                                 mz_uint16 comment_size,
                                 mz_uint level_and_flags, mz_uint64 uncomp_size,
                                 mz_uint32 uncomp_crc32) {
  mz_uint16 method = 0, dos_time = 0, dos_date = 0;
  mz_uint level, ext_attributes = 0, num_alignment_padding_bytes;
  mz_uint64 local_dir_header_ofs = pZip->m_archive_size,
            cur_archive_file_ofs = pZip->m_archive_size, comp_size = 0;
  size_t archive_name_size;
  mz_uint8 local_dir_header[MZ_ZIP_LOCAL_DIR_HEADER_SIZE];
  tdefl_compressor *pComp = NULL;
  mz_bool store_data_uncompressed;
  mz_zip_internal_state *pState;

  if ((int)level_and_flags < 0) level_and_flags = MZ_DEFAULT_LEVEL;
  level = level_and_flags & 0xF;
  store_data_uncompressed =
      ((!level) || (level_and_flags & MZ_ZIP_FLAG_COMPRESSED_DATA));

  if ((!pZip) || (!pZip->m_pState) ||
      (pZip->m_zip_mode != MZ_ZIP_MODE_WRITING) || ((buf_size) && (!pBuf)) ||
      (!pArchive_name) || ((comment_size) && (!pComment)) ||
      (pZip->m_total_files == 0xFFFF) || (level > MZ_UBER_COMPRESSION))
    return MZ_FALSE;

  pState = pZip->m_pState;

  if ((!(level_and_flags & MZ_ZIP_FLAG_COMPRESSED_DATA)) && (uncomp_size))
    return MZ_FALSE;
  // No zip64 support yet
  if ((buf_size > 0xFFFFFFFF) || (uncomp_size > 0xFFFFFFFF)) return MZ_FALSE;
  if (!mz_zip_writer_validate_archive_name(pArchive_name)) return MZ_FALSE;

#ifndef MINIZ_NO_TIME
  {
    time_t cur_time;
    time(&cur_time);
    mz_zip_time_to_dos_time(cur_time, &dos_time, &dos_date);
  }
#endif  // #ifndef MINIZ_NO_TIME

  archive_name_size = strlen(pArchive_name);
  if (archive_name_size > 0xFFFF) return MZ_FALSE;

  num_alignment_padding_bytes =
      mz_zip_writer_compute_padding_needed_for_file_alignment(pZip);

  // no zip64 support yet
  if ((pZip->m_total_files == 0xFFFF) ||
      ((pZip->m_archive_size + num_alignment_padding_bytes +
        MZ_ZIP_LOCAL_DIR_HEADER_SIZE + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE +
        comment_size + archive_name_size) > 0xFFFFFFFF))
    return MZ_FALSE;

  if ((archive_name_size) && (pArchive_name[archive_name_size - 1] == '/')) {
    // Set DOS Subdirectory attribute bit.
    ext_attributes |= 0x10;
    // Subdirectories cannot contain data.
    if ((buf_size) || (uncomp_size)) return MZ_FALSE;
  }

  // Try to do any allocations before writing to the archive, so if an
  // allocation fails the file remains unmodified. (A good idea if we're doing
  // an in-place modification.)
  if ((!mz_zip_array_ensure_room(
          pZip, &pState->m_central_dir,
          MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + archive_name_size + comment_size)) ||
      (!mz_zip_array_ensure_room(pZip, &pState->m_central_dir_offsets, 1)))
    return MZ_FALSE;

  if ((!store_data_uncompressed) && (buf_size)) {
    if (NULL == (pComp = (tdefl_compressor *)pZip->m_pAlloc(
                     pZip->m_pAlloc_opaque, 1, sizeof(tdefl_compressor))))
      return MZ_FALSE;
  }

  if (!mz_zip_writer_write_zeros(
          pZip, cur_archive_file_ofs,
          num_alignment_padding_bytes + sizeof(local_dir_header))) {
    pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);
    return MZ_FALSE;
  }
  local_dir_header_ofs += num_alignment_padding_bytes;
  if (pZip->m_file_offset_alignment) {
    MZ_ASSERT((local_dir_header_ofs & (pZip->m_file_offset_alignment - 1)) ==
              0);
  }
  cur_archive_file_ofs +=
      num_alignment_padding_bytes + sizeof(local_dir_header);

  MZ_CLEAR_OBJ(local_dir_header);
  if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, pArchive_name,
                     archive_name_size) != archive_name_size) {
    pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);
    return MZ_FALSE;
  }
  cur_archive_file_ofs += archive_name_size;

  if (!(level_and_flags & MZ_ZIP_FLAG_COMPRESSED_DATA)) {
    uncomp_crc32 =
        (mz_uint32)mz_crc32(MZ_CRC32_INIT, (const mz_uint8 *)pBuf, buf_size);
    uncomp_size = buf_size;
    if (uncomp_size <= 3) {
      level = 0;
      store_data_uncompressed = MZ_TRUE;
    }
  }

  if (store_data_uncompressed) {
    if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, pBuf,
                       buf_size) != buf_size) {
      pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);
      return MZ_FALSE;
    }

    cur_archive_file_ofs += buf_size;
    comp_size = buf_size;

    if (level_and_flags & MZ_ZIP_FLAG_COMPRESSED_DATA) method = MZ_DEFLATED;
  } else if (buf_size) {
    mz_zip_writer_add_state state;

    state.m_pZip = pZip;
    state.m_cur_archive_file_ofs = cur_archive_file_ofs;
    state.m_comp_size = 0;

    if ((tdefl_init(pComp, mz_zip_writer_add_put_buf_callback, &state,
                    tdefl_create_comp_flags_from_zip_params(
                        level, -15, MZ_DEFAULT_STRATEGY)) !=
         TDEFL_STATUS_OKAY) ||
        (tdefl_compress_buffer(pComp, pBuf, buf_size, TDEFL_FINISH) !=
         TDEFL_STATUS_DONE)) {
      pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);
      return MZ_FALSE;
    }

    comp_size = state.m_comp_size;
    cur_archive_file_ofs = state.m_cur_archive_file_ofs;

    method = MZ_DEFLATED;
  }

  pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);
  pComp = NULL;

  // no zip64 support yet
  if ((comp_size > 0xFFFFFFFF) || (cur_archive_file_ofs > 0xFFFFFFFF))
    return MZ_FALSE;

  if (!mz_zip_writer_create_local_dir_header(
          pZip, local_dir_header, (mz_uint16)archive_name_size, 0, uncomp_size,
          comp_size, uncomp_crc32, method, 0, dos_time, dos_date))
    return MZ_FALSE;

  if (pZip->m_pWrite(pZip->m_pIO_opaque, local_dir_header_ofs, local_dir_header,
                     sizeof(local_dir_header)) != sizeof(local_dir_header))
    return MZ_FALSE;

  if (!mz_zip_writer_add_to_central_dir(
          pZip, pArchive_name, (mz_uint16)archive_name_size, NULL, 0, pComment,
          comment_size, uncomp_size, comp_size, uncomp_crc32, method, 0,
          dos_time, dos_date, local_dir_header_ofs, ext_attributes))
    return MZ_FALSE;

  pZip->m_total_files++;
  pZip->m_archive_size = cur_archive_file_ofs;

  return MZ_TRUE;
}

#ifndef MINIZ_NO_STDIO
mz_bool mz_zip_writer_add_file(mz_zip_archive *pZip, const char *pArchive_name,
                               const char *pSrc_filename, const void *pComment,
                               mz_uint16 comment_size,
                               mz_uint level_and_flags) {
  mz_uint uncomp_crc32 = MZ_CRC32_INIT, level, num_alignment_padding_bytes;
  mz_uint16 method = 0, dos_time = 0, dos_date = 0, ext_attributes = 0;
  mz_uint64 local_dir_header_ofs = pZip->m_archive_size,
            cur_archive_file_ofs = pZip->m_archive_size, uncomp_size = 0,
            comp_size = 0;
  size_t archive_name_size;
  mz_uint8 local_dir_header[MZ_ZIP_LOCAL_DIR_HEADER_SIZE];
  MZ_FILE *pSrc_file = NULL;

  if ((int)level_and_flags < 0) level_and_flags = MZ_DEFAULT_LEVEL;
  level = level_and_flags & 0xF;

  if ((!pZip) || (!pZip->m_pState) ||
      (pZip->m_zip_mode != MZ_ZIP_MODE_WRITING) || (!pArchive_name) ||
      ((comment_size) && (!pComment)) || (level > MZ_UBER_COMPRESSION))
    return MZ_FALSE;
  if (level_and_flags & MZ_ZIP_FLAG_COMPRESSED_DATA) return MZ_FALSE;
  if (!mz_zip_writer_validate_archive_name(pArchive_name)) return MZ_FALSE;

  archive_name_size = strlen(pArchive_name);
  if (archive_name_size > 0xFFFF) return MZ_FALSE;

  num_alignment_padding_bytes =
      mz_zip_writer_compute_padding_needed_for_file_alignment(pZip);

  // no zip64 support yet
  if ((pZip->m_total_files == 0xFFFF) ||
      ((pZip->m_archive_size + num_alignment_padding_bytes +
        MZ_ZIP_LOCAL_DIR_HEADER_SIZE + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE +
        comment_size + archive_name_size) > 0xFFFFFFFF))
    return MZ_FALSE;

  if (!mz_zip_get_file_modified_time(pSrc_filename, &dos_time, &dos_date))
    return MZ_FALSE;

  pSrc_file = MZ_FOPEN(pSrc_filename, "rb");
  if (!pSrc_file) return MZ_FALSE;
  MZ_FSEEK64(pSrc_file, 0, SEEK_END);
  uncomp_size = MZ_FTELL64(pSrc_file);
  MZ_FSEEK64(pSrc_file, 0, SEEK_SET);

  if (uncomp_size > 0xFFFFFFFF) {
    // No zip64 support yet
    MZ_FCLOSE(pSrc_file);
    return MZ_FALSE;
  }
  if (uncomp_size <= 3) level = 0;

  if (!mz_zip_writer_write_zeros(
          pZip, cur_archive_file_ofs,
          num_alignment_padding_bytes + sizeof(local_dir_header))) {
    MZ_FCLOSE(pSrc_file);
    return MZ_FALSE;
  }
  local_dir_header_ofs += num_alignment_padding_bytes;
  if (pZip->m_file_offset_alignment) {
    MZ_ASSERT((local_dir_header_ofs & (pZip->m_file_offset_alignment - 1)) ==
              0);
  }
  cur_archive_file_ofs +=
      num_alignment_padding_bytes + sizeof(local_dir_header);

  MZ_CLEAR_OBJ(local_dir_header);
  if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, pArchive_name,
                     archive_name_size) != archive_name_size) {
    MZ_FCLOSE(pSrc_file);
    return MZ_FALSE;
  }
  cur_archive_file_ofs += archive_name_size;

  if (uncomp_size) {
    mz_uint64 uncomp_remaining = uncomp_size;
    void *pRead_buf =
        pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, MZ_ZIP_MAX_IO_BUF_SIZE);
    if (!pRead_buf) {
      MZ_FCLOSE(pSrc_file);
      return MZ_FALSE;
    }

    if (!level) {
      while (uncomp_remaining) {
        mz_uint n =
            (mz_uint)MZ_MIN((mz_uint)MZ_ZIP_MAX_IO_BUF_SIZE, uncomp_remaining);
        if ((MZ_FREAD(pRead_buf, 1, n, pSrc_file) != n) ||
            (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, pRead_buf,
                            n) != n)) {
          pZip->m_pFree(pZip->m_pAlloc_opaque, pRead_buf);
          MZ_FCLOSE(pSrc_file);
          return MZ_FALSE;
        }
        uncomp_crc32 =
            (mz_uint32)mz_crc32(uncomp_crc32, (const mz_uint8 *)pRead_buf, n);
        uncomp_remaining -= n;
        cur_archive_file_ofs += n;
      }
      comp_size = uncomp_size;
    } else {
      mz_bool result = MZ_FALSE;
      mz_zip_writer_add_state state;
      tdefl_compressor *pComp = (tdefl_compressor *)pZip->m_pAlloc(
          pZip->m_pAlloc_opaque, 1, sizeof(tdefl_compressor));
      if (!pComp) {
        pZip->m_pFree(pZip->m_pAlloc_opaque, pRead_buf);
        MZ_FCLOSE(pSrc_file);
        return MZ_FALSE;
      }

      state.m_pZip = pZip;
      state.m_cur_archive_file_ofs = cur_archive_file_ofs;
      state.m_comp_size = 0;

      if (tdefl_init(pComp, mz_zip_writer_add_put_buf_callback, &state,
                     tdefl_create_comp_flags_from_zip_params(
                         level, -15, MZ_DEFAULT_STRATEGY)) !=
          TDEFL_STATUS_OKAY) {
        pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);
        pZip->m_pFree(pZip->m_pAlloc_opaque, pRead_buf);
        MZ_FCLOSE(pSrc_file);
        return MZ_FALSE;
      }

      for (;;) {
        size_t in_buf_size = (mz_uint32)MZ_MIN(uncomp_remaining,
                                               (mz_uint)MZ_ZIP_MAX_IO_BUF_SIZE);
        tdefl_status status;

        if (MZ_FREAD(pRead_buf, 1, in_buf_size, pSrc_file) != in_buf_size)
          break;

        uncomp_crc32 = (mz_uint32)mz_crc32(
            uncomp_crc32, (const mz_uint8 *)pRead_buf, in_buf_size);
        uncomp_remaining -= in_buf_size;

        status = tdefl_compress_buffer(
            pComp, pRead_buf, in_buf_size,
            uncomp_remaining ? TDEFL_NO_FLUSH : TDEFL_FINISH);
        if (status == TDEFL_STATUS_DONE) {
          result = MZ_TRUE;
          break;
        } else if (status != TDEFL_STATUS_OKAY)
          break;
      }

      pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);

      if (!result) {
        pZip->m_pFree(pZip->m_pAlloc_opaque, pRead_buf);
        MZ_FCLOSE(pSrc_file);
        return MZ_FALSE;
      }

      comp_size = state.m_comp_size;
      cur_archive_file_ofs = state.m_cur_archive_file_ofs;

      method = MZ_DEFLATED;
    }

    pZip->m_pFree(pZip->m_pAlloc_opaque, pRead_buf);
  }

  MZ_FCLOSE(pSrc_file);
  pSrc_file = NULL;

  // no zip64 support yet
  if ((comp_size > 0xFFFFFFFF) || (cur_archive_file_ofs > 0xFFFFFFFF))
    return MZ_FALSE;

  if (!mz_zip_writer_create_local_dir_header(
          pZip, local_dir_header, (mz_uint16)archive_name_size, 0, uncomp_size,
          comp_size, uncomp_crc32, method, 0, dos_time, dos_date))
    return MZ_FALSE;

  if (pZip->m_pWrite(pZip->m_pIO_opaque, local_dir_header_ofs, local_dir_header,
                     sizeof(local_dir_header)) != sizeof(local_dir_header))
    return MZ_FALSE;

  if (!mz_zip_writer_add_to_central_dir(
          pZip, pArchive_name, (mz_uint16)archive_name_size, NULL, 0, pComment,
          comment_size, uncomp_size, comp_size, uncomp_crc32, method, 0,
          dos_time, dos_date, local_dir_header_ofs, ext_attributes))
    return MZ_FALSE;

  pZip->m_total_files++;
  pZip->m_archive_size = cur_archive_file_ofs;

  return MZ_TRUE;
}
#endif  // #ifndef MINIZ_NO_STDIO

mz_bool mz_zip_writer_add_from_zip_reader(mz_zip_archive *pZip,
                                          mz_zip_archive *pSource_zip,
                                          mz_uint file_index) {
  mz_uint n, bit_flags, num_alignment_padding_bytes;
  mz_uint64 comp_bytes_remaining, local_dir_header_ofs;
  mz_uint64 cur_src_file_ofs, cur_dst_file_ofs;
  mz_uint32
      local_header_u32[(MZ_ZIP_LOCAL_DIR_HEADER_SIZE + sizeof(mz_uint32) - 1) /
                       sizeof(mz_uint32)];
  mz_uint8 *pLocal_header = (mz_uint8 *)local_header_u32;
  mz_uint8 central_header[MZ_ZIP_CENTRAL_DIR_HEADER_SIZE];
  size_t orig_central_dir_size;
  mz_zip_internal_state *pState;
  void *pBuf;
  const mz_uint8 *pSrc_central_header;

  if ((!pZip) || (!pZip->m_pState) || (pZip->m_zip_mode != MZ_ZIP_MODE_WRITING))
    return MZ_FALSE;
  if (NULL ==
      (pSrc_central_header = mz_zip_reader_get_cdh(pSource_zip, file_index)))
    return MZ_FALSE;
  pState = pZip->m_pState;

  num_alignment_padding_bytes =
      mz_zip_writer_compute_padding_needed_for_file_alignment(pZip);

  // no zip64 support yet
  if ((pZip->m_total_files == 0xFFFF) ||
      ((pZip->m_archive_size + num_alignment_padding_bytes +
        MZ_ZIP_LOCAL_DIR_HEADER_SIZE + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE) >
       0xFFFFFFFF))
    return MZ_FALSE;

  cur_src_file_ofs =
      MZ_READ_LE32(pSrc_central_header + MZ_ZIP_CDH_LOCAL_HEADER_OFS);
  cur_dst_file_ofs = pZip->m_archive_size;

  if (pSource_zip->m_pRead(pSource_zip->m_pIO_opaque, cur_src_file_ofs,
                           pLocal_header, MZ_ZIP_LOCAL_DIR_HEADER_SIZE) !=
      MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
    return MZ_FALSE;
  if (MZ_READ_LE32(pLocal_header) != MZ_ZIP_LOCAL_DIR_HEADER_SIG)
    return MZ_FALSE;
  cur_src_file_ofs += MZ_ZIP_LOCAL_DIR_HEADER_SIZE;

  if (!mz_zip_writer_write_zeros(pZip, cur_dst_file_ofs,
                                 num_alignment_padding_bytes))
    return MZ_FALSE;
  cur_dst_file_ofs += num_alignment_padding_bytes;
  local_dir_header_ofs = cur_dst_file_ofs;
  if (pZip->m_file_offset_alignment) {
    MZ_ASSERT((local_dir_header_ofs & (pZip->m_file_offset_alignment - 1)) ==
              0);
  }

  if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_dst_file_ofs, pLocal_header,
                     MZ_ZIP_LOCAL_DIR_HEADER_SIZE) !=
      MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
    return MZ_FALSE;
  cur_dst_file_ofs += MZ_ZIP_LOCAL_DIR_HEADER_SIZE;

  n = MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_FILENAME_LEN_OFS) +
      MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_EXTRA_LEN_OFS);
  comp_bytes_remaining =
      n + MZ_READ_LE32(pSrc_central_header + MZ_ZIP_CDH_COMPRESSED_SIZE_OFS);

  if (NULL == (pBuf = pZip->m_pAlloc(
                   pZip->m_pAlloc_opaque, 1,
                   (size_t)MZ_MAX(sizeof(mz_uint32) * 4,
                                  MZ_MIN((mz_uint)MZ_ZIP_MAX_IO_BUF_SIZE,
                                         comp_bytes_remaining)))))
    return MZ_FALSE;

  while (comp_bytes_remaining) {
    n = (mz_uint)MZ_MIN((mz_uint)MZ_ZIP_MAX_IO_BUF_SIZE, comp_bytes_remaining);
    if (pSource_zip->m_pRead(pSource_zip->m_pIO_opaque, cur_src_file_ofs, pBuf,
                             n) != n) {
      pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);
      return MZ_FALSE;
    }
    cur_src_file_ofs += n;

    if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_dst_file_ofs, pBuf, n) != n) {
      pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);
      return MZ_FALSE;
    }
    cur_dst_file_ofs += n;

    comp_bytes_remaining -= n;
  }

  bit_flags = MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_BIT_FLAG_OFS);
  if (bit_flags & 8) {
    // Copy data descriptor
    if (pSource_zip->m_pRead(pSource_zip->m_pIO_opaque, cur_src_file_ofs, pBuf,
                             sizeof(mz_uint32) * 4) != sizeof(mz_uint32) * 4) {
      pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);
      return MZ_FALSE;
    }

    n = sizeof(mz_uint32) * ((MZ_READ_LE32(pBuf) == 0x08074b50) ? 4 : 3);
    if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_dst_file_ofs, pBuf, n) != n) {
      pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);
      return MZ_FALSE;
    }

    cur_src_file_ofs += n;
    cur_dst_file_ofs += n;
  }
  pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);

  // no zip64 support yet
  if (cur_dst_file_ofs > 0xFFFFFFFF) return MZ_FALSE;

  orig_central_dir_size = pState->m_central_dir.m_size;

  memcpy(central_header, pSrc_central_header, MZ_ZIP_CENTRAL_DIR_HEADER_SIZE);
  MZ_WRITE_LE32(central_header + MZ_ZIP_CDH_LOCAL_HEADER_OFS,
                local_dir_header_ofs);
  if (!mz_zip_array_push_back(pZip, &pState->m_central_dir, central_header,
                              MZ_ZIP_CENTRAL_DIR_HEADER_SIZE))
    return MZ_FALSE;

  n = MZ_READ_LE16(pSrc_central_header + MZ_ZIP_CDH_FILENAME_LEN_OFS) +
      MZ_READ_LE16(pSrc_central_header + MZ_ZIP_CDH_EXTRA_LEN_OFS) +
      MZ_READ_LE16(pSrc_central_header + MZ_ZIP_CDH_COMMENT_LEN_OFS);
  if (!mz_zip_array_push_back(
          pZip, &pState->m_central_dir,
          pSrc_central_header + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE, n)) {
    mz_zip_array_resize(pZip, &pState->m_central_dir, orig_central_dir_size,
                        MZ_FALSE);
    return MZ_FALSE;
  }

  if (pState->m_central_dir.m_size > 0xFFFFFFFF) return MZ_FALSE;
  n = (mz_uint32)orig_central_dir_size;
  if (!mz_zip_array_push_back(pZip, &pState->m_central_dir_offsets, &n, 1)) {
    mz_zip_array_resize(pZip, &pState->m_central_dir, orig_central_dir_size,
                        MZ_FALSE);
    return MZ_FALSE;
  }

  pZip->m_total_files++;
  pZip->m_archive_size = cur_dst_file_ofs;

  return MZ_TRUE;
}

mz_bool mz_zip_writer_finalize_archive(mz_zip_archive *pZip) {
  mz_zip_internal_state *pState;
  mz_uint64 central_dir_ofs, central_dir_size;
  mz_uint8 hdr[MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE];

  if ((!pZip) || (!pZip->m_pState) || (pZip->m_zip_mode != MZ_ZIP_MODE_WRITING))
    return MZ_FALSE;

  pState = pZip->m_pState;

  // no zip64 support yet
  if ((pZip->m_total_files > 0xFFFF) ||
      ((pZip->m_archive_size + pState->m_central_dir.m_size +
        MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE) > 0xFFFFFFFF))
    return MZ_FALSE;

  central_dir_ofs = 0;
  central_dir_size = 0;
  if (pZip->m_total_files) {
    // Write central directory
    central_dir_ofs = pZip->m_archive_size;
    central_dir_size = pState->m_central_dir.m_size;
    pZip->m_central_directory_file_ofs = central_dir_ofs;
    if (pZip->m_pWrite(pZip->m_pIO_opaque, central_dir_ofs,
                       pState->m_central_dir.m_p,
                       (size_t)central_dir_size) != central_dir_size)
      return MZ_FALSE;
    pZip->m_archive_size += central_dir_size;
  }

  // Write end of central directory record
  MZ_CLEAR_OBJ(hdr);
  MZ_WRITE_LE32(hdr + MZ_ZIP_ECDH_SIG_OFS,
                MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIG);
  MZ_WRITE_LE16(hdr + MZ_ZIP_ECDH_CDIR_NUM_ENTRIES_ON_DISK_OFS,
                pZip->m_total_files);
  MZ_WRITE_LE16(hdr + MZ_ZIP_ECDH_CDIR_TOTAL_ENTRIES_OFS, pZip->m_total_files);
  MZ_WRITE_LE32(hdr + MZ_ZIP_ECDH_CDIR_SIZE_OFS, central_dir_size);
  MZ_WRITE_LE32(hdr + MZ_ZIP_ECDH_CDIR_OFS_OFS, central_dir_ofs);

  if (pZip->m_pWrite(pZip->m_pIO_opaque, pZip->m_archive_size, hdr,
                     sizeof(hdr)) != sizeof(hdr))
    return MZ_FALSE;
#ifndef MINIZ_NO_STDIO
  if ((pState->m_pFile) && (MZ_FFLUSH(pState->m_pFile) == EOF)) return MZ_FALSE;
#endif  // #ifndef MINIZ_NO_STDIO

  pZip->m_archive_size += sizeof(hdr);

  pZip->m_zip_mode = MZ_ZIP_MODE_WRITING_HAS_BEEN_FINALIZED;
  return MZ_TRUE;
}

mz_bool mz_zip_writer_finalize_heap_archive(mz_zip_archive *pZip, void **pBuf,
                                            size_t *pSize) {
  if ((!pZip) || (!pZip->m_pState) || (!pBuf) || (!pSize)) return MZ_FALSE;
  if (pZip->m_pWrite != mz_zip_heap_write_func) return MZ_FALSE;
  if (!mz_zip_writer_finalize_archive(pZip)) return MZ_FALSE;

  *pBuf = pZip->m_pState->m_pMem;
  *pSize = pZip->m_pState->m_mem_size;
  pZip->m_pState->m_pMem = NULL;
  pZip->m_pState->m_mem_size = pZip->m_pState->m_mem_capacity = 0;
  return MZ_TRUE;
}

mz_bool mz_zip_writer_end(mz_zip_archive *pZip) {
  mz_zip_internal_state *pState;
  mz_bool status = MZ_TRUE;
  if ((!pZip) || (!pZip->m_pState) || (!pZip->m_pAlloc) || (!pZip->m_pFree) ||
      ((pZip->m_zip_mode != MZ_ZIP_MODE_WRITING) &&
       (pZip->m_zip_mode != MZ_ZIP_MODE_WRITING_HAS_BEEN_FINALIZED)))
    return MZ_FALSE;

  pState = pZip->m_pState;
  pZip->m_pState = NULL;
  mz_zip_array_clear(pZip, &pState->m_central_dir);
  mz_zip_array_clear(pZip, &pState->m_central_dir_offsets);
  mz_zip_array_clear(pZip, &pState->m_sorted_central_dir_offsets);

#ifndef MINIZ_NO_STDIO
  if (pState->m_pFile) {
    MZ_FCLOSE(pState->m_pFile);
    pState->m_pFile = NULL;
  }
#endif  // #ifndef MINIZ_NO_STDIO

  if ((pZip->m_pWrite == mz_zip_heap_write_func) && (pState->m_pMem)) {
    pZip->m_pFree(pZip->m_pAlloc_opaque, pState->m_pMem);
    pState->m_pMem = NULL;
  }

  pZip->m_pFree(pZip->m_pAlloc_opaque, pState);
  pZip->m_zip_mode = MZ_ZIP_MODE_INVALID;
  return status;
}

#ifndef MINIZ_NO_STDIO
mz_bool mz_zip_add_mem_to_archive_file_in_place(
    const char *pZip_filename, const char *pArchive_name, const void *pBuf,
    size_t buf_size, const void *pComment, mz_uint16 comment_size,
    mz_uint level_and_flags) {
  mz_bool status, created_new_archive = MZ_FALSE;
  mz_zip_archive zip_archive;
  struct MZ_FILE_STAT_STRUCT file_stat;
  MZ_CLEAR_OBJ(zip_archive);
  if ((int)level_and_flags < 0) level_and_flags = MZ_DEFAULT_LEVEL;
  if ((!pZip_filename) || (!pArchive_name) || ((buf_size) && (!pBuf)) ||
      ((comment_size) && (!pComment)) ||
      ((level_and_flags & 0xF) > MZ_UBER_COMPRESSION))
    return MZ_FALSE;
  if (!mz_zip_writer_validate_archive_name(pArchive_name)) return MZ_FALSE;
  if (MZ_FILE_STAT(pZip_filename, &file_stat) != 0) {
    // Create a new archive.
    if (!mz_zip_writer_init_file(&zip_archive, pZip_filename, 0))
      return MZ_FALSE;
    created_new_archive = MZ_TRUE;
  } else {
    // Append to an existing archive.
    if (!mz_zip_reader_init_file(
            &zip_archive, pZip_filename,
            level_and_flags | MZ_ZIP_FLAG_DO_NOT_SORT_CENTRAL_DIRECTORY))
      return MZ_FALSE;
    if (!mz_zip_writer_init_from_reader(&zip_archive, pZip_filename)) {
      mz_zip_reader_end(&zip_archive);
      return MZ_FALSE;
    }
  }
  status =
      mz_zip_writer_add_mem_ex(&zip_archive, pArchive_name, pBuf, buf_size,
                               pComment, comment_size, level_and_flags, 0, 0);
  // Always finalize, even if adding failed for some reason, so we have a valid
  // central directory. (This may not always succeed, but we can try.)
  if (!mz_zip_writer_finalize_archive(&zip_archive)) status = MZ_FALSE;
  if (!mz_zip_writer_end(&zip_archive)) status = MZ_FALSE;
  if ((!status) && (created_new_archive)) {
    // It's a new archive and something went wrong, so just delete it.
    int ignoredStatus = MZ_DELETE_FILE(pZip_filename);
    (void)ignoredStatus;
  }
  return status;
}

void *mz_zip_extract_archive_file_to_heap(const char *pZip_filename,
                                          const char *pArchive_name,
                                          size_t *pSize, mz_uint flags) {
  int file_index;
  mz_zip_archive zip_archive;
  void *p = NULL;

  if (pSize) *pSize = 0;

  if ((!pZip_filename) || (!pArchive_name)) return NULL;

  MZ_CLEAR_OBJ(zip_archive);
  if (!mz_zip_reader_init_file(
          &zip_archive, pZip_filename,
          flags | MZ_ZIP_FLAG_DO_NOT_SORT_CENTRAL_DIRECTORY))
    return NULL;

  if ((file_index = mz_zip_reader_locate_file(&zip_archive, pArchive_name, NULL,
                                              flags)) >= 0)
    p = mz_zip_reader_extract_to_heap(&zip_archive, file_index, pSize, flags);

  mz_zip_reader_end(&zip_archive);
  return p;
}

#endif  // #ifndef MINIZ_NO_STDIO

#endif  // #ifndef MINIZ_NO_ARCHIVE_WRITING_APIS

#endif  // #ifndef MINIZ_NO_ARCHIVE_APIS

#ifdef __cplusplus
}
#endif

#endif  // MINIZ_HEADER_FILE_ONLY

/*
  This is free and unencumbered software released into the public domain.

  Anyone is free to copy, modify, publish, use, compile, sell, or
  distribute this software, either in source code form or as a compiled
  binary, for any purpose, commercial or non-commercial, and by any
  means.

  In jurisdictions that recognize copyright laws, the author or authors
  of this software dedicate any and all copyright interest in the
  software to the public domain. We make this dedication for the benefit
  of the public at large and to the detriment of our heirs and
  successors. We intend this dedication to be an overt act of
  relinquishment in perpetuity of all present and future rights to this
  software under copyright law.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
  OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
  OTHER DEALINGS IN THE SOFTWARE.

  For more information, please refer to <http://unlicense.org/>
*/

// ---------------------- end of miniz ----------------------------------------

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif
}  // namespace miniz
#else

// Reuse MINIZ_LITTE_ENDIAN macro

#if defined(__sparcv9)
// Big endian
#else
#if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) || MINIZ_X86_OR_X64_CPU
// Set MINIZ_LITTLE_ENDIAN to 1 if the processor is little endian.
#define MINIZ_LITTLE_ENDIAN 1
#endif
#endif

#endif  // TINYEXR_USE_MINIZ

// static bool IsBigEndian(void) {
//  union {
//    unsigned int i;
//    char c[4];
//  } bint = {0x01020304};
//
//  return bint.c[0] == 1;
//}

static void SetErrorMessage(const std::string &msg, const char **err) {
  if (err) {
#ifdef _WIN32
    (*err) = _strdup(msg.c_str());
#else
    (*err) = strdup(msg.c_str());
#endif
  }
}

static const int kEXRVersionSize = 8;

static void cpy2(unsigned short *dst_val, const unsigned short *src_val) {
  unsigned char *dst = reinterpret_cast<unsigned char *>(dst_val);
  const unsigned char *src = reinterpret_cast<const unsigned char *>(src_val);

  dst[0] = src[0];
  dst[1] = src[1];
}

static void swap2(unsigned short *val) {
#ifdef MINIZ_LITTLE_ENDIAN
  (void)val;
#else
  unsigned short tmp = *val;
  unsigned char *dst = reinterpret_cast<unsigned char *>(val);
  unsigned char *src = reinterpret_cast<unsigned char *>(&tmp);

  dst[0] = src[1];
  dst[1] = src[0];
#endif
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
static void cpy4(int *dst_val, const int *src_val) {
  unsigned char *dst = reinterpret_cast<unsigned char *>(dst_val);
  const unsigned char *src = reinterpret_cast<const unsigned char *>(src_val);

  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
  dst[3] = src[3];
}

static void cpy4(unsigned int *dst_val, const unsigned int *src_val) {
  unsigned char *dst = reinterpret_cast<unsigned char *>(dst_val);
  const unsigned char *src = reinterpret_cast<const unsigned char *>(src_val);

  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
  dst[3] = src[3];
}

static void cpy4(float *dst_val, const float *src_val) {
  unsigned char *dst = reinterpret_cast<unsigned char *>(dst_val);
  const unsigned char *src = reinterpret_cast<const unsigned char *>(src_val);

  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
  dst[3] = src[3];
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

static void swap4(unsigned int *val) {
#ifdef MINIZ_LITTLE_ENDIAN
  (void)val;
#else
  unsigned int tmp = *val;
  unsigned char *dst = reinterpret_cast<unsigned char *>(val);
  unsigned char *src = reinterpret_cast<unsigned char *>(&tmp);

  dst[0] = src[3];
  dst[1] = src[2];
  dst[2] = src[1];
  dst[3] = src[0];
#endif
}

#if 0
static void cpy8(tinyexr::tinyexr_uint64 *dst_val, const tinyexr::tinyexr_uint64 *src_val) {
  unsigned char *dst = reinterpret_cast<unsigned char *>(dst_val);
  const unsigned char *src = reinterpret_cast<const unsigned char *>(src_val);

  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
  dst[3] = src[3];
  dst[4] = src[4];
  dst[5] = src[5];
  dst[6] = src[6];
  dst[7] = src[7];
}
#endif

static void swap8(tinyexr::tinyexr_uint64 *val) {
#ifdef MINIZ_LITTLE_ENDIAN
  (void)val;
#else
  tinyexr::tinyexr_uint64 tmp = (*val);
  unsigned char *dst = reinterpret_cast<unsigned char *>(val);
  unsigned char *src = reinterpret_cast<unsigned char *>(&tmp);

  dst[0] = src[7];
  dst[1] = src[6];
  dst[2] = src[5];
  dst[3] = src[4];
  dst[4] = src[3];
  dst[5] = src[2];
  dst[6] = src[1];
  dst[7] = src[0];
#endif
}

// https://gist.github.com/rygorous/2156668
// Reuse MINIZ_LITTLE_ENDIAN flag from miniz.
union FP32 {
  unsigned int u;
  float f;
  struct {
#if MINIZ_LITTLE_ENDIAN
    unsigned int Mantissa : 23;
    unsigned int Exponent : 8;
    unsigned int Sign : 1;
#else
    unsigned int Sign : 1;
    unsigned int Exponent : 8;
    unsigned int Mantissa : 23;
#endif
  } s;
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

union FP16 {
  unsigned short u;
  struct {
#if MINIZ_LITTLE_ENDIAN
    unsigned int Mantissa : 10;
    unsigned int Exponent : 5;
    unsigned int Sign : 1;
#else
    unsigned int Sign : 1;
    unsigned int Exponent : 5;
    unsigned int Mantissa : 10;
#endif
  } s;
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

static FP32 half_to_float(FP16 h) {
  static const FP32 magic = {113 << 23};
  static const unsigned int shifted_exp = 0x7c00
                                          << 13;  // exponent mask after shift
  FP32 o;

  o.u = (h.u & 0x7fffU) << 13U;           // exponent/mantissa bits
  unsigned int exp_ = shifted_exp & o.u;  // just the exponent
  o.u += (127 - 15) << 23;                // exponent adjust

  // handle exponent special cases
  if (exp_ == shifted_exp)    // Inf/NaN?
    o.u += (128 - 16) << 23;  // extra exp adjust
  else if (exp_ == 0)         // Zero/Denormal?
  {
    o.u += 1 << 23;  // extra exp adjust
    o.f -= magic.f;  // renormalize
  }

  o.u |= (h.u & 0x8000U) << 16U;  // sign bit
  return o;
}

static FP16 float_to_half_full(FP32 f) {
  FP16 o = {0};

  // Based on ISPC reference code (with minor modifications)
  if (f.s.Exponent == 0)  // Signed zero/denormal (which will underflow)
    o.s.Exponent = 0;
  else if (f.s.Exponent == 255)  // Inf or NaN (all exponent bits set)
  {
    o.s.Exponent = 31;
    o.s.Mantissa = f.s.Mantissa ? 0x200 : 0;  // NaN->qNaN and Inf->Inf
  } else                                      // Normalized number
  {
    // Exponent unbias the single, then bias the halfp
    int newexp = f.s.Exponent - 127 + 15;
    if (newexp >= 31)  // Overflow, return signed infinity
      o.s.Exponent = 31;
    else if (newexp <= 0)  // Underflow
    {
      if ((14 - newexp) <= 24)  // Mantissa might be non-zero
      {
        unsigned int mant = f.s.Mantissa | 0x800000;  // Hidden 1 bit
        o.s.Mantissa = mant >> (14 - newexp);
        if ((mant >> (13 - newexp)) & 1)  // Check for rounding
          o.u++;  // Round, might overflow into exp bit, but this is OK
      }
    } else {
      o.s.Exponent = static_cast<unsigned int>(newexp);
      o.s.Mantissa = f.s.Mantissa >> 13;
      if (f.s.Mantissa & 0x1000)  // Check for rounding
        o.u++;                    // Round, might overflow to inf, this is OK
    }
  }

  o.s.Sign = f.s.Sign;
  return o;
}

// NOTE: From OpenEXR code
// #define IMF_INCREASING_Y  0
// #define IMF_DECREASING_Y  1
// #define IMF_RAMDOM_Y    2
//
// #define IMF_NO_COMPRESSION  0
// #define IMF_RLE_COMPRESSION 1
// #define IMF_ZIPS_COMPRESSION  2
// #define IMF_ZIP_COMPRESSION 3
// #define IMF_PIZ_COMPRESSION 4
// #define IMF_PXR24_COMPRESSION 5
// #define IMF_B44_COMPRESSION 6
// #define IMF_B44A_COMPRESSION  7

#ifdef __clang__
#pragma clang diagnostic push

#if __has_warning("-Wzero-as-null-pointer-constant")
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif

#endif

static const char *ReadString(std::string *s, const char *ptr, size_t len) {
  // Read untile NULL(\0).
  const char *p = ptr;
  const char *q = ptr;
  while ((size_t(q - ptr) < len) && (*q) != 0) {
    q++;
  }

  if (size_t(q - ptr) >= len) {
    (*s) = std::string();
    return NULL;
  }

  (*s) = std::string(p, q);

  return q + 1;  // skip '\0'
}

static bool ReadAttribute(std::string *name, std::string *type,
                          std::vector<unsigned char> *data, size_t *marker_size,
                          const char *marker, size_t size) {
  size_t name_len = strnlen(marker, size);
  if (name_len == size) {
    // String does not have a terminating character.
    return false;
  }
  *name = std::string(marker, name_len);

  marker += name_len + 1;
  size -= name_len + 1;

  size_t type_len = strnlen(marker, size);
  if (type_len == size) {
    return false;
  }
  *type = std::string(marker, type_len);

  marker += type_len + 1;
  size -= type_len + 1;

  if (size < sizeof(uint32_t)) {
    return false;
  }

  uint32_t data_len;
  memcpy(&data_len, marker, sizeof(uint32_t));
  tinyexr::swap4(reinterpret_cast<unsigned int *>(&data_len));

  if (data_len == 0) {
    if ((*type).compare("string") == 0) {
      // Accept empty string attribute.

      marker += sizeof(uint32_t);
      size -= sizeof(uint32_t);

      *marker_size = name_len + 1 + type_len + 1 + sizeof(uint32_t);

      data->resize(1);
      (*data)[0] = '\0';

      return true;
    } else {
      return false;
    }
  }

  marker += sizeof(uint32_t);
  size -= sizeof(uint32_t);

  if (size < data_len) {
    return false;
  }

  data->resize(static_cast<size_t>(data_len));
  memcpy(&data->at(0), marker, static_cast<size_t>(data_len));

  *marker_size = name_len + 1 + type_len + 1 + sizeof(uint32_t) + data_len;
  return true;
}

static void WriteAttributeToMemory(std::vector<unsigned char> *out,
                                   const char *name, const char *type,
                                   const unsigned char *data, int len) {
  out->insert(out->end(), name, name + strlen(name) + 1);
  out->insert(out->end(), type, type + strlen(type) + 1);

  int outLen = len;
  tinyexr::swap4(reinterpret_cast<unsigned int *>(&outLen));
  out->insert(out->end(), reinterpret_cast<unsigned char *>(&outLen),
              reinterpret_cast<unsigned char *>(&outLen) + sizeof(int));
  out->insert(out->end(), data, data + len);
}

typedef struct {
  std::string name;  // less than 255 bytes long
  int pixel_type;
  int x_sampling;
  int y_sampling;
  unsigned char p_linear;
  unsigned char pad[3];
} ChannelInfo;

typedef struct {
  std::vector<tinyexr::ChannelInfo> channels;
  std::vector<EXRAttribute> attributes;

  int data_window[4];
  int line_order;
  int display_window[4];
  float screen_window_center[2];
  float screen_window_width;
  float pixel_aspect_ratio;

  int chunk_count;

  // Tiled format
  int tile_size_x;
  int tile_size_y;
  int tile_level_mode;
  int tile_rounding_mode;

  unsigned int header_len;

  int compression_type;

  void clear() {
    channels.clear();
    attributes.clear();

    data_window[0] = 0;
    data_window[1] = 0;
    data_window[2] = 0;
    data_window[3] = 0;
    line_order = 0;
    display_window[0] = 0;
    display_window[1] = 0;
    display_window[2] = 0;
    display_window[3] = 0;
    screen_window_center[0] = 0.0f;
    screen_window_center[1] = 0.0f;
    screen_window_width = 0.0f;
    pixel_aspect_ratio = 0.0f;

    chunk_count = 0;

    // Tiled format
    tile_size_x = 0;
    tile_size_y = 0;
    tile_level_mode = 0;
    tile_rounding_mode = 0;

    header_len = 0;
    compression_type = 0;
  }
} HeaderInfo;

static bool ReadChannelInfo(std::vector<ChannelInfo> &channels,
                            const std::vector<unsigned char> &data) {
  const char *p = reinterpret_cast<const char *>(&data.at(0));

  for (;;) {
    if ((*p) == 0) {
      break;
    }
    ChannelInfo info;

    tinyexr_int64 data_len = static_cast<tinyexr_int64>(data.size()) -
                             (p - reinterpret_cast<const char *>(data.data()));
    if (data_len < 0) {
      return false;
    }

    p = ReadString(&info.name, p, size_t(data_len));
    if ((p == NULL) && (info.name.empty())) {
      // Buffer overrun. Issue #51.
      return false;
    }

    const unsigned char *data_end =
        reinterpret_cast<const unsigned char *>(p) + 16;
    if (data_end >= (data.data() + data.size())) {
      return false;
    }

    memcpy(&info.pixel_type, p, sizeof(int));
    p += 4;
    info.p_linear = static_cast<unsigned char>(p[0]);  // uchar
    p += 1 + 3;                                        // reserved: uchar[3]
    memcpy(&info.x_sampling, p, sizeof(int));          // int
    p += 4;
    memcpy(&info.y_sampling, p, sizeof(int));  // int
    p += 4;

    tinyexr::swap4(reinterpret_cast<unsigned int *>(&info.pixel_type));
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&info.x_sampling));
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&info.y_sampling));

    channels.push_back(info);
  }

  return true;
}

static void WriteChannelInfo(std::vector<unsigned char> &data,
                             const std::vector<ChannelInfo> &channels) {
  size_t sz = 0;

  // Calculate total size.
  for (size_t c = 0; c < channels.size(); c++) {
    sz += strlen(channels[c].name.c_str()) + 1;  // +1 for \0
    sz += 16;                                    // 4 * int
  }
  data.resize(sz + 1);

  unsigned char *p = &data.at(0);

  for (size_t c = 0; c < channels.size(); c++) {
    memcpy(p, channels[c].name.c_str(), strlen(channels[c].name.c_str()));
    p += strlen(channels[c].name.c_str());
    (*p) = '\0';
    p++;

    int pixel_type = channels[c].pixel_type;
    int x_sampling = channels[c].x_sampling;
    int y_sampling = channels[c].y_sampling;
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&pixel_type));
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&x_sampling));
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&y_sampling));

    memcpy(p, &pixel_type, sizeof(int));
    p += sizeof(int);

    (*p) = channels[c].p_linear;
    p += 4;

    memcpy(p, &x_sampling, sizeof(int));
    p += sizeof(int);

    memcpy(p, &y_sampling, sizeof(int));
    p += sizeof(int);
  }

  (*p) = '\0';
}

static void CompressZip(unsigned char *dst,
                        tinyexr::tinyexr_uint64 &compressedSize,
                        const unsigned char *src, unsigned long src_size) {
  std::vector<unsigned char> tmpBuf(src_size);

  //
  // Apply EXR-specific? postprocess. Grabbed from OpenEXR's
  // ImfZipCompressor.cpp
  //

  //
  // Reorder the pixel data.
  //

  const char *srcPtr = reinterpret_cast<const char *>(src);

  {
    char *t1 = reinterpret_cast<char *>(&tmpBuf.at(0));
    char *t2 = reinterpret_cast<char *>(&tmpBuf.at(0)) + (src_size + 1) / 2;
    const char *stop = srcPtr + src_size;

    for (;;) {
      if (srcPtr < stop)
        *(t1++) = *(srcPtr++);
      else
        break;

      if (srcPtr < stop)
        *(t2++) = *(srcPtr++);
      else
        break;
    }
  }

  //
  // Predictor.
  //

  {
    unsigned char *t = &tmpBuf.at(0) + 1;
    unsigned char *stop = &tmpBuf.at(0) + src_size;
    int p = t[-1];

    while (t < stop) {
      int d = int(t[0]) - p + (128 + 256);
      p = t[0];
      t[0] = static_cast<unsigned char>(d);
      ++t;
    }
  }

#if TINYEXR_USE_MINIZ
  //
  // Compress the data using miniz
  //

  miniz::mz_ulong outSize = miniz::mz_compressBound(src_size);
  int ret = miniz::mz_compress(
      dst, &outSize, static_cast<const unsigned char *>(&tmpBuf.at(0)),
      src_size);
  assert(ret == miniz::MZ_OK);
  (void)ret;

  compressedSize = outSize;
#else
  uLong outSize = compressBound(static_cast<uLong>(src_size));
  int ret = compress(dst, &outSize, static_cast<const Bytef *>(&tmpBuf.at(0)),
                     src_size);
  assert(ret == Z_OK);

  compressedSize = outSize;
#endif

  // Use uncompressed data when compressed data is larger than uncompressed.
  // (Issue 40)
  if (compressedSize >= src_size) {
    compressedSize = src_size;
    memcpy(dst, src, src_size);
  }
}

static bool DecompressZip(unsigned char *dst,
                          unsigned long *uncompressed_size /* inout */,
                          const unsigned char *src, unsigned long src_size) {
  if ((*uncompressed_size) == src_size) {
    // Data is not compressed(Issue 40).
    memcpy(dst, src, src_size);
    return true;
  }
  std::vector<unsigned char> tmpBuf(*uncompressed_size);

#if TINYEXR_USE_MINIZ
  int ret =
      miniz::mz_uncompress(&tmpBuf.at(0), uncompressed_size, src, src_size);
  if (miniz::MZ_OK != ret) {
    return false;
  }
#else
  int ret = uncompress(&tmpBuf.at(0), uncompressed_size, src, src_size);
  if (Z_OK != ret) {
    return false;
  }
#endif

  //
  // Apply EXR-specific? postprocess. Grabbed from OpenEXR's
  // ImfZipCompressor.cpp
  //

  // Predictor.
  {
    unsigned char *t = &tmpBuf.at(0) + 1;
    unsigned char *stop = &tmpBuf.at(0) + (*uncompressed_size);

    while (t < stop) {
      int d = int(t[-1]) + int(t[0]) - 128;
      t[0] = static_cast<unsigned char>(d);
      ++t;
    }
  }

  // Reorder the pixel data.
  {
    const char *t1 = reinterpret_cast<const char *>(&tmpBuf.at(0));
    const char *t2 = reinterpret_cast<const char *>(&tmpBuf.at(0)) +
                     (*uncompressed_size + 1) / 2;
    char *s = reinterpret_cast<char *>(dst);
    char *stop = s + (*uncompressed_size);

    for (;;) {
      if (s < stop)
        *(s++) = *(t1++);
      else
        break;

      if (s < stop)
        *(s++) = *(t2++);
      else
        break;
    }
  }

  return true;
}

// RLE code from OpenEXR --------------------------------------

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#if __has_warning("-Wextra-semi-stmt")
#pragma clang diagnostic ignored "-Wextra-semi-stmt"
#endif
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4204)  // nonstandard extension used : non-constant
                                 // aggregate initializer (also supported by GNU
                                 // C and C99, so no big deal)
#pragma warning(disable : 4244)  // 'initializing': conversion from '__int64' to
                                 // 'int', possible loss of data
#pragma warning(disable : 4267)  // 'argument': conversion from '__int64' to
                                 // 'int', possible loss of data
#pragma warning(disable : 4996)  // 'strdup': The POSIX name for this item is
                                 // deprecated. Instead, use the ISO C and C++
                                 // conformant name: _strdup.
#endif

const int MIN_RUN_LENGTH = 3;
const int MAX_RUN_LENGTH = 127;

//
// Compress an array of bytes, using run-length encoding,
// and return the length of the compressed data.
//

static int rleCompress(int inLength, const char in[], signed char out[]) {
  const char *inEnd = in + inLength;
  const char *runStart = in;
  const char *runEnd = in + 1;
  signed char *outWrite = out;

  while (runStart < inEnd) {
    while (runEnd < inEnd && *runStart == *runEnd &&
           runEnd - runStart - 1 < MAX_RUN_LENGTH) {
      ++runEnd;
    }

    if (runEnd - runStart >= MIN_RUN_LENGTH) {
      //
      // Compressable run
      //

      *outWrite++ = static_cast<char>(runEnd - runStart) - 1;
      *outWrite++ = *(reinterpret_cast<const signed char *>(runStart));
      runStart = runEnd;
    } else {
      //
      // Uncompressable run
      //

      while (runEnd < inEnd &&
             ((runEnd + 1 >= inEnd || *runEnd != *(runEnd + 1)) ||
              (runEnd + 2 >= inEnd || *(runEnd + 1) != *(runEnd + 2))) &&
             runEnd - runStart < MAX_RUN_LENGTH) {
        ++runEnd;
      }

      *outWrite++ = static_cast<char>(runStart - runEnd);

      while (runStart < runEnd) {
        *outWrite++ = *(reinterpret_cast<const signed char *>(runStart++));
      }
    }

    ++runEnd;
  }

  return static_cast<int>(outWrite - out);
}

//
// Uncompress an array of bytes compressed with rleCompress().
// Returns the length of the oncompressed data, or 0 if the
// length of the uncompressed data would be more than maxLength.
//

static int rleUncompress(int inLength, int maxLength, const signed char in[],
                         char out[]) {
  char *outStart = out;

  while (inLength > 0) {
    if (*in < 0) {
      int count = -(static_cast<int>(*in++));
      inLength -= count + 1;

      // Fixes #116: Add bounds check to in buffer.
      if ((0 > (maxLength -= count)) || (inLength < 0)) return 0;

      memcpy(out, in, count);
      out += count;
      in += count;
    } else {
      int count = *in++;
      inLength -= 2;

      if (0 > (maxLength -= count + 1)) return 0;

      memset(out, *reinterpret_cast<const char *>(in), count + 1);
      out += count + 1;

      in++;
    }
  }

  return static_cast<int>(out - outStart);
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

// End of RLE code from OpenEXR -----------------------------------

static void CompressRle(unsigned char *dst,
                        tinyexr::tinyexr_uint64 &compressedSize,
                        const unsigned char *src, unsigned long src_size) {
  std::vector<unsigned char> tmpBuf(src_size);

  //
  // Apply EXR-specific? postprocess. Grabbed from OpenEXR's
  // ImfRleCompressor.cpp
  //

  //
  // Reorder the pixel data.
  //

  const char *srcPtr = reinterpret_cast<const char *>(src);

  {
    char *t1 = reinterpret_cast<char *>(&tmpBuf.at(0));
    char *t2 = reinterpret_cast<char *>(&tmpBuf.at(0)) + (src_size + 1) / 2;
    const char *stop = srcPtr + src_size;

    for (;;) {
      if (srcPtr < stop)
        *(t1++) = *(srcPtr++);
      else
        break;

      if (srcPtr < stop)
        *(t2++) = *(srcPtr++);
      else
        break;
    }
  }

  //
  // Predictor.
  //

  {
    unsigned char *t = &tmpBuf.at(0) + 1;
    unsigned char *stop = &tmpBuf.at(0) + src_size;
    int p = t[-1];

    while (t < stop) {
      int d = int(t[0]) - p + (128 + 256);
      p = t[0];
      t[0] = static_cast<unsigned char>(d);
      ++t;
    }
  }

  // outSize will be (srcSiz * 3) / 2 at max.
  int outSize = rleCompress(static_cast<int>(src_size),
                            reinterpret_cast<const char *>(&tmpBuf.at(0)),
                            reinterpret_cast<signed char *>(dst));
  assert(outSize > 0);

  compressedSize = static_cast<tinyexr::tinyexr_uint64>(outSize);

  // Use uncompressed data when compressed data is larger than uncompressed.
  // (Issue 40)
  if (compressedSize >= src_size) {
    compressedSize = src_size;
    memcpy(dst, src, src_size);
  }
}

static bool DecompressRle(unsigned char *dst,
                          const unsigned long uncompressed_size,
                          const unsigned char *src, unsigned long src_size) {
  if (uncompressed_size == src_size) {
    // Data is not compressed(Issue 40).
    memcpy(dst, src, src_size);
    return true;
  }

  // Workaround for issue #112.
  // TODO(syoyo): Add more robust out-of-bounds check in `rleUncompress`.
  if (src_size <= 2) {
    return false;
  }

  std::vector<unsigned char> tmpBuf(uncompressed_size);

  int ret = rleUncompress(static_cast<int>(src_size),
                          static_cast<int>(uncompressed_size),
                          reinterpret_cast<const signed char *>(src),
                          reinterpret_cast<char *>(&tmpBuf.at(0)));
  if (ret != static_cast<int>(uncompressed_size)) {
    return false;
  }

  //
  // Apply EXR-specific? postprocess. Grabbed from OpenEXR's
  // ImfRleCompressor.cpp
  //

  // Predictor.
  {
    unsigned char *t = &tmpBuf.at(0) + 1;
    unsigned char *stop = &tmpBuf.at(0) + uncompressed_size;

    while (t < stop) {
      int d = int(t[-1]) + int(t[0]) - 128;
      t[0] = static_cast<unsigned char>(d);
      ++t;
    }
  }

  // Reorder the pixel data.
  {
    const char *t1 = reinterpret_cast<const char *>(&tmpBuf.at(0));
    const char *t2 = reinterpret_cast<const char *>(&tmpBuf.at(0)) +
                     (uncompressed_size + 1) / 2;
    char *s = reinterpret_cast<char *>(dst);
    char *stop = s + uncompressed_size;

    for (;;) {
      if (s < stop)
        *(s++) = *(t1++);
      else
        break;

      if (s < stop)
        *(s++) = *(t2++);
      else
        break;
    }
  }

  return true;
}

#if TINYEXR_USE_PIZ

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-long-long"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wc++11-extensions"
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"

#if __has_warning("-Wcast-qual")
#pragma clang diagnostic ignored "-Wcast-qual"
#endif

#if __has_warning("-Wextra-semi-stmt")
#pragma clang diagnostic ignored "-Wextra-semi-stmt"
#endif

#endif

//
// PIZ compress/uncompress, based on OpenEXR's ImfPizCompressor.cpp
//
// -----------------------------------------------------------------
// Copyright (c) 2004, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC)
// (3 clause BSD license)
//

struct PIZChannelData {
  unsigned short *start;
  unsigned short *end;
  int nx;
  int ny;
  int ys;
  int size;
};

//-----------------------------------------------------------------------------
//
//  16-bit Haar Wavelet encoding and decoding
//
//  The source code in this file is derived from the encoding
//  and decoding routines written by Christian Rouet for his
//  PIZ image file format.
//
//-----------------------------------------------------------------------------

//
// Wavelet basis functions without modulo arithmetic; they produce
// the best compression ratios when the wavelet-transformed data are
// Huffman-encoded, but the wavelet transform works only for 14-bit
// data (untransformed data values must be less than (1 << 14)).
//

inline void wenc14(unsigned short a, unsigned short b, unsigned short &l,
                   unsigned short &h) {
  short as = static_cast<short>(a);
  short bs = static_cast<short>(b);

  short ms = (as + bs) >> 1;
  short ds = as - bs;

  l = static_cast<unsigned short>(ms);
  h = static_cast<unsigned short>(ds);
}

inline void wdec14(unsigned short l, unsigned short h, unsigned short &a,
                   unsigned short &b) {
  short ls = static_cast<short>(l);
  short hs = static_cast<short>(h);

  int hi = hs;
  int ai = ls + (hi & 1) + (hi >> 1);

  short as = static_cast<short>(ai);
  short bs = static_cast<short>(ai - hi);

  a = static_cast<unsigned short>(as);
  b = static_cast<unsigned short>(bs);
}

//
// Wavelet basis functions with modulo arithmetic; they work with full
// 16-bit data, but Huffman-encoding the wavelet-transformed data doesn't
// compress the data quite as well.
//

const int NBITS = 16;
const int A_OFFSET = 1 << (NBITS - 1);
const int M_OFFSET = 1 << (NBITS - 1);
const int MOD_MASK = (1 << NBITS) - 1;

inline void wenc16(unsigned short a, unsigned short b, unsigned short &l,
                   unsigned short &h) {
  int ao = (a + A_OFFSET) & MOD_MASK;
  int m = ((ao + b) >> 1);
  int d = ao - b;

  if (d < 0) m = (m + M_OFFSET) & MOD_MASK;

  d &= MOD_MASK;

  l = static_cast<unsigned short>(m);
  h = static_cast<unsigned short>(d);
}

inline void wdec16(unsigned short l, unsigned short h, unsigned short &a,
                   unsigned short &b) {
  int m = l;
  int d = h;
  int bb = (m - (d >> 1)) & MOD_MASK;
  int aa = (d + bb - A_OFFSET) & MOD_MASK;
  b = static_cast<unsigned short>(bb);
  a = static_cast<unsigned short>(aa);
}

//
// 2D Wavelet encoding:
//

static void wav2Encode(
    unsigned short *in,  // io: values are transformed in place
    int nx,              // i : x size
    int ox,              // i : x offset
    int ny,              // i : y size
    int oy,              // i : y offset
    unsigned short mx)   // i : maximum in[x][y] value
{
  bool w14 = (mx < (1 << 14));
  int n = (nx > ny) ? ny : nx;
  int p = 1;   // == 1 <<  level
  int p2 = 2;  // == 1 << (level+1)

  //
  // Hierachical loop on smaller dimension n
  //

  while (p2 <= n) {
    unsigned short *py = in;
    unsigned short *ey = in + oy * (ny - p2);
    int oy1 = oy * p;
    int oy2 = oy * p2;
    int ox1 = ox * p;
    int ox2 = ox * p2;
    unsigned short i00, i01, i10, i11;

    //
    // Y loop
    //

    for (; py <= ey; py += oy2) {
      unsigned short *px = py;
      unsigned short *ex = py + ox * (nx - p2);

      //
      // X loop
      //

      for (; px <= ex; px += ox2) {
        unsigned short *p01 = px + ox1;
        unsigned short *p10 = px + oy1;
        unsigned short *p11 = p10 + ox1;

        //
        // 2D wavelet encoding
        //

        if (w14) {
          wenc14(*px, *p01, i00, i01);
          wenc14(*p10, *p11, i10, i11);
          wenc14(i00, i10, *px, *p10);
          wenc14(i01, i11, *p01, *p11);
        } else {
          wenc16(*px, *p01, i00, i01);
          wenc16(*p10, *p11, i10, i11);
          wenc16(i00, i10, *px, *p10);
          wenc16(i01, i11, *p01, *p11);
        }
      }

      //
      // Encode (1D) odd column (still in Y loop)
      //

      if (nx & p) {
        unsigned short *p10 = px + oy1;

        if (w14)
          wenc14(*px, *p10, i00, *p10);
        else
          wenc16(*px, *p10, i00, *p10);

        *px = i00;
      }
    }

    //
    // Encode (1D) odd line (must loop in X)
    //

    if (ny & p) {
      unsigned short *px = py;
      unsigned short *ex = py + ox * (nx - p2);

      for (; px <= ex; px += ox2) {
        unsigned short *p01 = px + ox1;

        if (w14)
          wenc14(*px, *p01, i00, *p01);
        else
          wenc16(*px, *p01, i00, *p01);

        *px = i00;
      }
    }

    //
    // Next level
    //

    p = p2;
    p2 <<= 1;
  }
}

//
// 2D Wavelet decoding:
//

static void wav2Decode(
    unsigned short *in,  // io: values are transformed in place
    int nx,              // i : x size
    int ox,              // i : x offset
    int ny,              // i : y size
    int oy,              // i : y offset
    unsigned short mx)   // i : maximum in[x][y] value
{
  bool w14 = (mx < (1 << 14));
  int n = (nx > ny) ? ny : nx;
  int p = 1;
  int p2;

  //
  // Search max level
  //

  while (p <= n) p <<= 1;

  p >>= 1;
  p2 = p;
  p >>= 1;

  //
  // Hierarchical loop on smaller dimension n
  //

  while (p >= 1) {
    unsigned short *py = in;
    unsigned short *ey = in + oy * (ny - p2);
    int oy1 = oy * p;
    int oy2 = oy * p2;
    int ox1 = ox * p;
    int ox2 = ox * p2;
    unsigned short i00, i01, i10, i11;

    //
    // Y loop
    //

    for (; py <= ey; py += oy2) {
      unsigned short *px = py;
      unsigned short *ex = py + ox * (nx - p2);

      //
      // X loop
      //

      for (; px <= ex; px += ox2) {
        unsigned short *p01 = px + ox1;
        unsigned short *p10 = px + oy1;
        unsigned short *p11 = p10 + ox1;

        //
        // 2D wavelet decoding
        //

        if (w14) {
          wdec14(*px, *p10, i00, i10);
          wdec14(*p01, *p11, i01, i11);
          wdec14(i00, i01, *px, *p01);
          wdec14(i10, i11, *p10, *p11);
        } else {
          wdec16(*px, *p10, i00, i10);
          wdec16(*p01, *p11, i01, i11);
          wdec16(i00, i01, *px, *p01);
          wdec16(i10, i11, *p10, *p11);
        }
      }

      //
      // Decode (1D) odd column (still in Y loop)
      //

      if (nx & p) {
        unsigned short *p10 = px + oy1;

        if (w14)
          wdec14(*px, *p10, i00, *p10);
        else
          wdec16(*px, *p10, i00, *p10);

        *px = i00;
      }
    }

    //
    // Decode (1D) odd line (must loop in X)
    //

    if (ny & p) {
      unsigned short *px = py;
      unsigned short *ex = py + ox * (nx - p2);

      for (; px <= ex; px += ox2) {
        unsigned short *p01 = px + ox1;

        if (w14)
          wdec14(*px, *p01, i00, *p01);
        else
          wdec16(*px, *p01, i00, *p01);

        *px = i00;
      }
    }

    //
    // Next level
    //

    p2 = p;
    p >>= 1;
  }
}

//-----------------------------------------------------------------------------
//
//  16-bit Huffman compression and decompression.
//
//  The source code in this file is derived from the 8-bit
//  Huffman compression and decompression routines written
//  by Christian Rouet for his PIZ image file format.
//
//-----------------------------------------------------------------------------

// Adds some modification for tinyexr.

const int HUF_ENCBITS = 16;  // literal (value) bit length
const int HUF_DECBITS = 14;  // decoding bit size (>= 8)

const int HUF_ENCSIZE = (1 << HUF_ENCBITS) + 1;  // encoding table size
const int HUF_DECSIZE = 1 << HUF_DECBITS;        // decoding table size
const int HUF_DECMASK = HUF_DECSIZE - 1;

struct HufDec {  // short code    long code
  //-------------------------------
  int len : 8;   // code length    0
  int lit : 24;  // lit      p size
  int *p;        // 0      lits
};

inline long long hufLength(long long code) { return code & 63; }

inline long long hufCode(long long code) { return code >> 6; }

inline void outputBits(int nBits, long long bits, long long &c, int &lc,
                       char *&out) {
  c <<= nBits;
  lc += nBits;

  c |= bits;

  while (lc >= 8) *out++ = static_cast<char>((c >> (lc -= 8)));
}

inline long long getBits(int nBits, long long &c, int &lc, const char *&in) {
  while (lc < nBits) {
    c = (c << 8) | *(reinterpret_cast<const unsigned char *>(in++));
    lc += 8;
  }

  lc -= nBits;
  return (c >> lc) & ((1 << nBits) - 1);
}

//
// ENCODING TABLE BUILDING & (UN)PACKING
//

//
// Build a "canonical" Huffman code table:
//  - for each (uncompressed) symbol, hcode contains the length
//    of the corresponding code (in the compressed data)
//  - canonical codes are computed and stored in hcode
//  - the rules for constructing canonical codes are as follows:
//    * shorter codes (if filled with zeroes to the right)
//      have a numerically higher value than longer codes
//    * for codes with the same length, numerical values
//      increase with numerical symbol values
//  - because the canonical code table can be constructed from
//    symbol lengths alone, the code table can be transmitted
//    without sending the actual code values
//  - see http://www.compressconsult.com/huffman/
//

static void hufCanonicalCodeTable(long long hcode[HUF_ENCSIZE]) {
  long long n[59];

  //
  // For each i from 0 through 58, count the
  // number of different codes of length i, and
  // store the count in n[i].
  //

  for (int i = 0; i <= 58; ++i) n[i] = 0;

  for (int i = 0; i < HUF_ENCSIZE; ++i) n[hcode[i]] += 1;

  //
  // For each i from 58 through 1, compute the
  // numerically lowest code with length i, and
  // store that code in n[i].
  //

  long long c = 0;

  for (int i = 58; i > 0; --i) {
    long long nc = ((c + n[i]) >> 1);
    n[i] = c;
    c = nc;
  }

  //
  // hcode[i] contains the length, l, of the
  // code for symbol i.  Assign the next available
  // code of length l to the symbol and store both
  // l and the code in hcode[i].
  //

  for (int i = 0; i < HUF_ENCSIZE; ++i) {
    int l = static_cast<int>(hcode[i]);

    if (l > 0) hcode[i] = l | (n[l]++ << 6);
  }
}

//
// Compute Huffman codes (based on frq input) and store them in frq:
//  - code structure is : [63:lsb - 6:msb] | [5-0: bit length];
//  - max code length is 58 bits;
//  - codes outside the range [im-iM] have a null length (unused values);
//  - original frequencies are destroyed;
//  - encoding tables are used by hufEncode() and hufBuildDecTable();
//

struct FHeapCompare {
  bool operator()(long long *a, long long *b) { return *a > *b; }
};

static void hufBuildEncTable(
    long long *frq,  // io: input frequencies [HUF_ENCSIZE], output table
    int *im,         //  o: min frq index
    int *iM)         //  o: max frq index
{
  //
  // This function assumes that when it is called, array frq
  // indicates the frequency of all possible symbols in the data
  // that are to be Huffman-encoded.  (frq[i] contains the number
  // of occurrences of symbol i in the data.)
  //
  // The loop below does three things:
  //
  // 1) Finds the minimum and maximum indices that point
  //    to non-zero entries in frq:
  //
  //     frq[im] != 0, and frq[i] == 0 for all i < im
  //     frq[iM] != 0, and frq[i] == 0 for all i > iM
  //
  // 2) Fills array fHeap with pointers to all non-zero
  //    entries in frq.
  //
  // 3) Initializes array hlink such that hlink[i] == i
  //    for all array entries.
  //

  std::vector<int> hlink(HUF_ENCSIZE);
  std::vector<long long *> fHeap(HUF_ENCSIZE);

  *im = 0;

  while (!frq[*im]) (*im)++;

  int nf = 0;

  for (int i = *im; i < HUF_ENCSIZE; i++) {
    hlink[i] = i;

    if (frq[i]) {
      fHeap[nf] = &frq[i];
      nf++;
      *iM = i;
    }
  }

  //
  // Add a pseudo-symbol, with a frequency count of 1, to frq;
  // adjust the fHeap and hlink array accordingly.  Function
  // hufEncode() uses the pseudo-symbol for run-length encoding.
  //

  (*iM)++;
  frq[*iM] = 1;
  fHeap[nf] = &frq[*iM];
  nf++;

  //
  // Build an array, scode, such that scode[i] contains the number
  // of bits assigned to symbol i.  Conceptually this is done by
  // constructing a tree whose leaves are the symbols with non-zero
  // frequency:
  //
  //     Make a heap that contains all symbols with a non-zero frequency,
  //     with the least frequent symbol on top.
  //
  //     Repeat until only one symbol is left on the heap:
  //
  //         Take the two least frequent symbols off the top of the heap.
  //         Create a new node that has first two nodes as children, and
  //         whose frequency is the sum of the frequencies of the first
  //         two nodes.  Put the new node back into the heap.
  //
  // The last node left on the heap is the root of the tree.  For each
  // leaf node, the distance between the root and the leaf is the length
  // of the code for the corresponding symbol.
  //
  // The loop below doesn't actually build the tree; instead we compute
  // the distances of the leaves from the root on the fly.  When a new
  // node is added to the heap, then that node's descendants are linked
  // into a single linear list that starts at the new node, and the code
  // lengths of the descendants (that is, their distance from the root
  // of the tree) are incremented by one.
  //

  std::make_heap(&fHeap[0], &fHeap[nf], FHeapCompare());

  std::vector<long long> scode(HUF_ENCSIZE);
  memset(scode.data(), 0, sizeof(long long) * HUF_ENCSIZE);

  while (nf > 1) {
    //
    // Find the indices, mm and m, of the two smallest non-zero frq
    // values in fHeap, add the smallest frq to the second-smallest
    // frq, and remove the smallest frq value from fHeap.
    //

    int mm = fHeap[0] - frq;
    std::pop_heap(&fHeap[0], &fHeap[nf], FHeapCompare());
    --nf;

    int m = fHeap[0] - frq;
    std::pop_heap(&fHeap[0], &fHeap[nf], FHeapCompare());

    frq[m] += frq[mm];
    std::push_heap(&fHeap[0], &fHeap[nf], FHeapCompare());

    //
    // The entries in scode are linked into lists with the
    // entries in hlink serving as "next" pointers and with
    // the end of a list marked by hlink[j] == j.
    //
    // Traverse the lists that start at scode[m] and scode[mm].
    // For each element visited, increment the length of the
    // corresponding code by one bit. (If we visit scode[j]
    // during the traversal, then the code for symbol j becomes
    // one bit longer.)
    //
    // Merge the lists that start at scode[m] and scode[mm]
    // into a single list that starts at scode[m].
    //

    //
    // Add a bit to all codes in the first list.
    //

    for (int j = m;; j = hlink[j]) {
      scode[j]++;

      assert(scode[j] <= 58);

      if (hlink[j] == j) {
        //
        // Merge the two lists.
        //

        hlink[j] = mm;
        break;
      }
    }

    //
    // Add a bit to all codes in the second list
    //

    for (int j = mm;; j = hlink[j]) {
      scode[j]++;

      assert(scode[j] <= 58);

      if (hlink[j] == j) break;
    }
  }

  //
  // Build a canonical Huffman code table, replacing the code
  // lengths in scode with (code, code length) pairs.  Copy the
  // code table from scode into frq.
  //

  hufCanonicalCodeTable(scode.data());
  memcpy(frq, scode.data(), sizeof(long long) * HUF_ENCSIZE);
}

//
// Pack an encoding table:
//  - only code lengths, not actual codes, are stored
//  - runs of zeroes are compressed as follows:
//
//    unpacked    packed
//    --------------------------------
//    1 zero    0  (6 bits)
//    2 zeroes    59
//    3 zeroes    60
//    4 zeroes    61
//    5 zeroes    62
//    n zeroes (6 or more)  63 n-6  (6 + 8 bits)
//

const int SHORT_ZEROCODE_RUN = 59;
const int LONG_ZEROCODE_RUN = 63;
const int SHORTEST_LONG_RUN = 2 + LONG_ZEROCODE_RUN - SHORT_ZEROCODE_RUN;
const int LONGEST_LONG_RUN = 255 + SHORTEST_LONG_RUN;

static void hufPackEncTable(
    const long long *hcode,  // i : encoding table [HUF_ENCSIZE]
    int im,                  // i : min hcode index
    int iM,                  // i : max hcode index
    char **pcode)            //  o: ptr to packed table (updated)
{
  char *p = *pcode;
  long long c = 0;
  int lc = 0;

  for (; im <= iM; im++) {
    int l = hufLength(hcode[im]);

    if (l == 0) {
      int zerun = 1;

      while ((im < iM) && (zerun < LONGEST_LONG_RUN)) {
        if (hufLength(hcode[im + 1]) > 0) break;
        im++;
        zerun++;
      }

      if (zerun >= 2) {
        if (zerun >= SHORTEST_LONG_RUN) {
          outputBits(6, LONG_ZEROCODE_RUN, c, lc, p);
          outputBits(8, zerun - SHORTEST_LONG_RUN, c, lc, p);
        } else {
          outputBits(6, SHORT_ZEROCODE_RUN + zerun - 2, c, lc, p);
        }
        continue;
      }
    }

    outputBits(6, l, c, lc, p);
  }

  if (lc > 0) *p++ = (unsigned char)(c << (8 - lc));

  *pcode = p;
}

//
// Unpack an encoding table packed by hufPackEncTable():
//

static bool hufUnpackEncTable(
    const char **pcode,  // io: ptr to packed table (updated)
    int ni,              // i : input size (in bytes)
    int im,              // i : min hcode index
    int iM,              // i : max hcode index
    long long *hcode)    //  o: encoding table [HUF_ENCSIZE]
{
  memset(hcode, 0, sizeof(long long) * HUF_ENCSIZE);

  const char *p = *pcode;
  long long c = 0;
  int lc = 0;

  for (; im <= iM; im++) {
    if (p - *pcode >= ni) {
      return false;
    }

    long long l = hcode[im] = getBits(6, c, lc, p);  // code length

    if (l == (long long)LONG_ZEROCODE_RUN) {
      if (p - *pcode > ni) {
        return false;
      }

      int zerun = getBits(8, c, lc, p) + SHORTEST_LONG_RUN;

      if (im + zerun > iM + 1) {
        return false;
      }

      while (zerun--) hcode[im++] = 0;

      im--;
    } else if (l >= (long long)SHORT_ZEROCODE_RUN) {
      int zerun = l - SHORT_ZEROCODE_RUN + 2;

      if (im + zerun > iM + 1) {
        return false;
      }

      while (zerun--) hcode[im++] = 0;

      im--;
    }
  }

  *pcode = const_cast<char *>(p);

  hufCanonicalCodeTable(hcode);

  return true;
}

//
// DECODING TABLE BUILDING
//

//
// Clear a newly allocated decoding table so that it contains only zeroes.
//

static void hufClearDecTable(HufDec *hdecod)  // io: (allocated by caller)
//     decoding table [HUF_DECSIZE]
{
  for (int i = 0; i < HUF_DECSIZE; i++) {
    hdecod[i].len = 0;
    hdecod[i].lit = 0;
    hdecod[i].p = NULL;
  }
  // memset(hdecod, 0, sizeof(HufDec) * HUF_DECSIZE);
}

//
// Build a decoding hash table based on the encoding table hcode:
//  - short codes (<= HUF_DECBITS) are resolved with a single table access;
//  - long code entry allocations are not optimized, because long codes are
//    unfrequent;
//  - decoding tables are used by hufDecode();
//

static bool hufBuildDecTable(const long long *hcode,  // i : encoding table
                             int im,                  // i : min index in hcode
                             int iM,                  // i : max index in hcode
                             HufDec *hdecod)  //  o: (allocated by caller)
//     decoding table [HUF_DECSIZE]
{
  //
  // Init hashtable & loop on all codes.
  // Assumes that hufClearDecTable(hdecod) has already been called.
  //

  for (; im <= iM; im++) {
    long long c = hufCode(hcode[im]);
    int l = hufLength(hcode[im]);

    if (c >> l) {
      //
      // Error: c is supposed to be an l-bit code,
      // but c contains a value that is greater
      // than the largest l-bit number.
      //

      // invalidTableEntry();
      return false;
    }

    if (l > HUF_DECBITS) {
      //
      // Long code: add a secondary entry
      //

      HufDec *pl = hdecod + (c >> (l - HUF_DECBITS));

      if (pl->len) {
        //
        // Error: a short code has already
        // been stored in table entry *pl.
        //

        // invalidTableEntry();
        return false;
      }

      pl->lit++;

      if (pl->p) {
        int *p = pl->p;
        pl->p = new int[pl->lit];

        for (int i = 0; i < pl->lit - 1; ++i) pl->p[i] = p[i];

        delete[] p;
      } else {
        pl->p = new int[1];
      }

      pl->p[pl->lit - 1] = im;
    } else if (l) {
      //
      // Short code: init all primary entries
      //

      HufDec *pl = hdecod + (c << (HUF_DECBITS - l));

      for (long long i = 1ULL << (HUF_DECBITS - l); i > 0; i--, pl++) {
        if (pl->len || pl->p) {
          //
          // Error: a short code or a long code has
          // already been stored in table entry *pl.
          //

          // invalidTableEntry();
          return false;
        }

        pl->len = l;
        pl->lit = im;
      }
    }
  }

  return true;
}

//
// Free the long code entries of a decoding table built by hufBuildDecTable()
//

static void hufFreeDecTable(HufDec *hdecod)  // io: Decoding table
{
  for (int i = 0; i < HUF_DECSIZE; i++) {
    if (hdecod[i].p) {
      delete[] hdecod[i].p;
      hdecod[i].p = 0;
    }
  }
}

//
// ENCODING
//

inline void outputCode(long long code, long long &c, int &lc, char *&out) {
  outputBits(hufLength(code), hufCode(code), c, lc, out);
}

inline void sendCode(long long sCode, int runCount, long long runCode,
                     long long &c, int &lc, char *&out) {
  //
  // Output a run of runCount instances of the symbol sCount.
  // Output the symbols explicitly, or if that is shorter, output
  // the sCode symbol once followed by a runCode symbol and runCount
  // expressed as an 8-bit number.
  //

  if (hufLength(sCode) + hufLength(runCode) + 8 < hufLength(sCode) * runCount) {
    outputCode(sCode, c, lc, out);
    outputCode(runCode, c, lc, out);
    outputBits(8, runCount, c, lc, out);
  } else {
    while (runCount-- >= 0) outputCode(sCode, c, lc, out);
  }
}

//
// Encode (compress) ni values based on the Huffman encoding table hcode:
//

static int hufEncode            // return: output size (in bits)
    (const long long *hcode,    // i : encoding table
     const unsigned short *in,  // i : uncompressed input buffer
     const int ni,              // i : input buffer size (in bytes)
     int rlc,                   // i : rl code
     char *out)                 //  o: compressed output buffer
{
  char *outStart = out;
  long long c = 0;  // bits not yet written to out
  int lc = 0;       // number of valid bits in c (LSB)
  int s = in[0];
  int cs = 0;

  //
  // Loop on input values
  //

  for (int i = 1; i < ni; i++) {
    //
    // Count same values or send code
    //

    if (s == in[i] && cs < 255) {
      cs++;
    } else {
      sendCode(hcode[s], cs, hcode[rlc], c, lc, out);
      cs = 0;
    }

    s = in[i];
  }

  //
  // Send remaining code
  //

  sendCode(hcode[s], cs, hcode[rlc], c, lc, out);

  if (lc) *out = (c << (8 - lc)) & 0xff;

  return (out - outStart) * 8 + lc;
}

//
// DECODING
//

//
// In order to force the compiler to inline them,
// getChar() and getCode() are implemented as macros
// instead of "inline" functions.
//

#define getChar(c, lc, in)                   \
  {                                          \
    c = (c << 8) | *(unsigned char *)(in++); \
    lc += 8;                                 \
  }

#if 0
#define getCode(po, rlc, c, lc, in, out, ob, oe) \
  {                                              \
    if (po == rlc) {                             \
      if (lc < 8) getChar(c, lc, in);            \
                                                 \
      lc -= 8;                                   \
                                                 \
      unsigned char cs = (c >> lc);              \
                                                 \
      if (out + cs > oe) return false;           \
                                                 \
      /* TinyEXR issue 78 */                     \
      unsigned short s = out[-1];                \
                                                 \
      while (cs-- > 0) *out++ = s;               \
    } else if (out < oe) {                       \
      *out++ = po;                               \
    } else {                                     \
      return false;                              \
    }                                            \
  }
#else
static bool getCode(int po, int rlc, long long &c, int &lc, const char *&in,
                    const char *in_end, unsigned short *&out,
                    const unsigned short *ob, const unsigned short *oe) {
  (void)ob;
  if (po == rlc) {
    if (lc < 8) {
      /* TinyEXR issue 78 */
      if ((in + 1) >= in_end) {
        return false;
      }

      getChar(c, lc, in);
    }

    lc -= 8;

    unsigned char cs = (c >> lc);

    if (out + cs > oe) return false;

    // Bounds check for safety
    // Issue 100.
    if ((out - 1) < ob) return false;
    unsigned short s = out[-1];

    while (cs-- > 0) *out++ = s;
  } else if (out < oe) {
    *out++ = po;
  } else {
    return false;
  }
  return true;
}
#endif

//
// Decode (uncompress) ni bits based on encoding & decoding tables:
//

static bool hufDecode(const long long *hcode,  // i : encoding table
                      const HufDec *hdecod,    // i : decoding table
                      const char *in,          // i : compressed input buffer
                      int ni,                  // i : input size (in bits)
                      int rlc,                 // i : run-length code
                      int no,  // i : expected output size (in bytes)
                      unsigned short *out)  //  o: uncompressed output buffer
{
  long long c = 0;
  int lc = 0;
  unsigned short *outb = out;          // begin
  unsigned short *oe = out + no;       // end
  const char *ie = in + (ni + 7) / 8;  // input byte size

  //
  // Loop on input bytes
  //

  while (in < ie) {
    getChar(c, lc, in);

    //
    // Access decoding table
    //

    while (lc >= HUF_DECBITS) {
      const HufDec pl = hdecod[(c >> (lc - HUF_DECBITS)) & HUF_DECMASK];

      if (pl.len) {
        //
        // Get short code
        //

        lc -= pl.len;
        // std::cout << "lit = " << pl.lit << std::endl;
        // std::cout << "rlc = " << rlc << std::endl;
        // std::cout << "c = " << c << std::endl;
        // std::cout << "lc = " << lc << std::endl;
        // std::cout << "in = " << in << std::endl;
        // std::cout << "out = " << out << std::endl;
        // std::cout << "oe = " << oe << std::endl;
        if (!getCode(pl.lit, rlc, c, lc, in, ie, out, outb, oe)) {
          return false;
        }
      } else {
        if (!pl.p) {
          return false;
        }
        // invalidCode(); // wrong code

        //
        // Search long code
        //

        int j;

        for (j = 0; j < pl.lit; j++) {
          int l = hufLength(hcode[pl.p[j]]);

          while (lc < l && in < ie)  // get more bits
            getChar(c, lc, in);

          if (lc >= l) {
            if (hufCode(hcode[pl.p[j]]) ==
                ((c >> (lc - l)) & (((long long)(1) << l) - 1))) {
              //
              // Found : get long code
              //

              lc -= l;
              if (!getCode(pl.p[j], rlc, c, lc, in, ie, out, outb, oe)) {
                return false;
              }
              break;
            }
          }
        }

        if (j == pl.lit) {
          return false;
          // invalidCode(); // Not found
        }
      }
    }
  }

  //
  // Get remaining (short) codes
  //

  int i = (8 - ni) & 7;
  c >>= i;
  lc -= i;

  while (lc > 0) {
    const HufDec pl = hdecod[(c << (HUF_DECBITS - lc)) & HUF_DECMASK];

    if (pl.len) {
      lc -= pl.len;
      if (!getCode(pl.lit, rlc, c, lc, in, ie, out, outb, oe)) {
        return false;
      }
    } else {
      return false;
      // invalidCode(); // wrong (long) code
    }
  }

  if (out - outb != no) {
    return false;
  }
  // notEnoughData ();

  return true;
}

static void countFrequencies(std::vector<long long> &freq,
                             const unsigned short data[/*n*/], int n) {
  for (int i = 0; i < HUF_ENCSIZE; ++i) freq[i] = 0;

  for (int i = 0; i < n; ++i) ++freq[data[i]];
}

static void writeUInt(char buf[4], unsigned int i) {
  unsigned char *b = (unsigned char *)buf;

  b[0] = i;
  b[1] = i >> 8;
  b[2] = i >> 16;
  b[3] = i >> 24;
}

static unsigned int readUInt(const char buf[4]) {
  const unsigned char *b = (const unsigned char *)buf;

  return (b[0] & 0x000000ff) | ((b[1] << 8) & 0x0000ff00) |
         ((b[2] << 16) & 0x00ff0000) | ((b[3] << 24) & 0xff000000);
}

//
// EXTERNAL INTERFACE
//

static int hufCompress(const unsigned short raw[], int nRaw,
                       char compressed[]) {
  if (nRaw == 0) return 0;

  std::vector<long long> freq(HUF_ENCSIZE);

  countFrequencies(freq, raw, nRaw);

  int im = 0;
  int iM = 0;
  hufBuildEncTable(freq.data(), &im, &iM);

  char *tableStart = compressed + 20;
  char *tableEnd = tableStart;
  hufPackEncTable(freq.data(), im, iM, &tableEnd);
  int tableLength = tableEnd - tableStart;

  char *dataStart = tableEnd;
  int nBits = hufEncode(freq.data(), raw, nRaw, iM, dataStart);
  int data_length = (nBits + 7) / 8;

  writeUInt(compressed, im);
  writeUInt(compressed + 4, iM);
  writeUInt(compressed + 8, tableLength);
  writeUInt(compressed + 12, nBits);
  writeUInt(compressed + 16, 0);  // room for future extensions

  return dataStart + data_length - compressed;
}

static bool hufUncompress(const char compressed[], int nCompressed,
                          std::vector<unsigned short> *raw) {
  if (nCompressed == 0) {
    if (raw->size() != 0) return false;

    return false;
  }

  int im = readUInt(compressed);
  int iM = readUInt(compressed + 4);
  // int tableLength = readUInt (compressed + 8);
  int nBits = readUInt(compressed + 12);

  if (im < 0 || im >= HUF_ENCSIZE || iM < 0 || iM >= HUF_ENCSIZE) return false;

  const char *ptr = compressed + 20;

  //
  // Fast decoder needs at least 2x64-bits of compressed data, and
  // needs to be run-able on this platform. Otherwise, fall back
  // to the original decoder
  //

  // if (FastHufDecoder::enabled() && nBits > 128)
  //{
  //    FastHufDecoder fhd (ptr, nCompressed - (ptr - compressed), im, iM, iM);
  //    fhd.decode ((unsigned char*)ptr, nBits, raw, nRaw);
  //}
  // else
  {
    std::vector<long long> freq(HUF_ENCSIZE);
    std::vector<HufDec> hdec(HUF_DECSIZE);

    hufClearDecTable(&hdec.at(0));

    hufUnpackEncTable(&ptr, nCompressed - (ptr - compressed), im, iM,
                      &freq.at(0));

    {
      if (nBits > 8 * (nCompressed - (ptr - compressed))) {
        return false;
      }

      hufBuildDecTable(&freq.at(0), im, iM, &hdec.at(0));
      hufDecode(&freq.at(0), &hdec.at(0), ptr, nBits, iM, raw->size(),
                raw->data());
    }
    // catch (...)
    //{
    //    hufFreeDecTable (hdec);
    //    throw;
    //}

    hufFreeDecTable(&hdec.at(0));
  }

  return true;
}

//
// Functions to compress the range of values in the pixel data
//

const int USHORT_RANGE = (1 << 16);
const int BITMAP_SIZE = (USHORT_RANGE >> 3);

static void bitmapFromData(const unsigned short data[/*nData*/], int nData,
                           unsigned char bitmap[BITMAP_SIZE],
                           unsigned short &minNonZero,
                           unsigned short &maxNonZero) {
  for (int i = 0; i < BITMAP_SIZE; ++i) bitmap[i] = 0;

  for (int i = 0; i < nData; ++i) bitmap[data[i] >> 3] |= (1 << (data[i] & 7));

  bitmap[0] &= ~1;  // zero is not explicitly stored in
                    // the bitmap; we assume that the
                    // data always contain zeroes
  minNonZero = BITMAP_SIZE - 1;
  maxNonZero = 0;

  for (int i = 0; i < BITMAP_SIZE; ++i) {
    if (bitmap[i]) {
      if (minNonZero > i) minNonZero = i;
      if (maxNonZero < i) maxNonZero = i;
    }
  }
}

static unsigned short forwardLutFromBitmap(
    const unsigned char bitmap[BITMAP_SIZE], unsigned short lut[USHORT_RANGE]) {
  int k = 0;

  for (int i = 0; i < USHORT_RANGE; ++i) {
    if ((i == 0) || (bitmap[i >> 3] & (1 << (i & 7))))
      lut[i] = k++;
    else
      lut[i] = 0;
  }

  return k - 1;  // maximum value stored in lut[],
}  // i.e. number of ones in bitmap minus 1

static unsigned short reverseLutFromBitmap(
    const unsigned char bitmap[BITMAP_SIZE], unsigned short lut[USHORT_RANGE]) {
  int k = 0;

  for (int i = 0; i < USHORT_RANGE; ++i) {
    if ((i == 0) || (bitmap[i >> 3] & (1 << (i & 7)))) lut[k++] = i;
  }

  int n = k - 1;

  while (k < USHORT_RANGE) lut[k++] = 0;

  return n;  // maximum k where lut[k] is non-zero,
}  // i.e. number of ones in bitmap minus 1

static void applyLut(const unsigned short lut[USHORT_RANGE],
                     unsigned short data[/*nData*/], int nData) {
  for (int i = 0; i < nData; ++i) data[i] = lut[data[i]];
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif  // __clang__

#ifdef _MSC_VER
#pragma warning(pop)
#endif

static bool CompressPiz(unsigned char *outPtr, unsigned int *outSize,
                        const unsigned char *inPtr, size_t inSize,
                        const std::vector<ChannelInfo> &channelInfo,
                        int data_width, int num_lines) {
  std::vector<unsigned char> bitmap(BITMAP_SIZE);
  unsigned short minNonZero;
  unsigned short maxNonZero;

#if !MINIZ_LITTLE_ENDIAN
  // @todo { PIZ compression on BigEndian architecture. }
  assert(0);
  return false;
#endif

  // Assume `inSize` is multiple of 2 or 4.
  std::vector<unsigned short> tmpBuffer(inSize / sizeof(unsigned short));

  std::vector<PIZChannelData> channelData(channelInfo.size());
  unsigned short *tmpBufferEnd = &tmpBuffer.at(0);

  for (size_t c = 0; c < channelData.size(); c++) {
    PIZChannelData &cd = channelData[c];

    cd.start = tmpBufferEnd;
    cd.end = cd.start;

    cd.nx = data_width;
    cd.ny = num_lines;
    // cd.ys = c.channel().ySampling;

    size_t pixelSize = sizeof(int);  // UINT and FLOAT
    if (channelInfo[c].pixel_type == TINYEXR_PIXELTYPE_HALF) {
      pixelSize = sizeof(short);
    }

    cd.size = static_cast<int>(pixelSize / sizeof(short));

    tmpBufferEnd += cd.nx * cd.ny * cd.size;
  }

  const unsigned char *ptr = inPtr;
  for (int y = 0; y < num_lines; ++y) {
    for (size_t i = 0; i < channelData.size(); ++i) {
      PIZChannelData &cd = channelData[i];

      // if (modp (y, cd.ys) != 0)
      //    continue;

      size_t n = static_cast<size_t>(cd.nx * cd.size);
      memcpy(cd.end, ptr, n * sizeof(unsigned short));
      ptr += n * sizeof(unsigned short);
      cd.end += n;
    }
  }

  bitmapFromData(&tmpBuffer.at(0), static_cast<int>(tmpBuffer.size()),
                 bitmap.data(), minNonZero, maxNonZero);

  std::vector<unsigned short> lut(USHORT_RANGE);
  unsigned short maxValue = forwardLutFromBitmap(bitmap.data(), lut.data());
  applyLut(lut.data(), &tmpBuffer.at(0), static_cast<int>(tmpBuffer.size()));

  //
  // Store range compression info in _outBuffer
  //

  char *buf = reinterpret_cast<char *>(outPtr);

  memcpy(buf, &minNonZero, sizeof(unsigned short));
  buf += sizeof(unsigned short);
  memcpy(buf, &maxNonZero, sizeof(unsigned short));
  buf += sizeof(unsigned short);

  if (minNonZero <= maxNonZero) {
    memcpy(buf, reinterpret_cast<char *>(&bitmap[0] + minNonZero),
           maxNonZero - minNonZero + 1);
    buf += maxNonZero - minNonZero + 1;
  }

  //
  // Apply wavelet encoding
  //

  for (size_t i = 0; i < channelData.size(); ++i) {
    PIZChannelData &cd = channelData[i];

    for (int j = 0; j < cd.size; ++j) {
      wav2Encode(cd.start + j, cd.nx, cd.size, cd.ny, cd.nx * cd.size,
                 maxValue);
    }
  }

  //
  // Apply Huffman encoding; append the result to _outBuffer
  //

  // length header(4byte), then huff data. Initialize length header with zero,
  // then later fill it by `length`.
  char *lengthPtr = buf;
  int zero = 0;
  memcpy(buf, &zero, sizeof(int));
  buf += sizeof(int);

  int length =
      hufCompress(&tmpBuffer.at(0), static_cast<int>(tmpBuffer.size()), buf);
  memcpy(lengthPtr, &length, sizeof(int));

  (*outSize) = static_cast<unsigned int>(
      (reinterpret_cast<unsigned char *>(buf) - outPtr) +
      static_cast<unsigned int>(length));

  // Use uncompressed data when compressed data is larger than uncompressed.
  // (Issue 40)
  if ((*outSize) >= inSize) {
    (*outSize) = static_cast<unsigned int>(inSize);
    memcpy(outPtr, inPtr, inSize);
  }
  return true;
}

static bool DecompressPiz(unsigned char *outPtr, const unsigned char *inPtr,
                          size_t tmpBufSize, size_t inLen, int num_channels,
                          const EXRChannelInfo *channels, int data_width,
                          int num_lines) {
  if (inLen == tmpBufSize) {
    // Data is not compressed(Issue 40).
    memcpy(outPtr, inPtr, inLen);
    return true;
  }

  std::vector<unsigned char> bitmap(BITMAP_SIZE);
  unsigned short minNonZero;
  unsigned short maxNonZero;

#if !MINIZ_LITTLE_ENDIAN
  // @todo { PIZ compression on BigEndian architecture. }
  assert(0);
  return false;
#endif

  memset(bitmap.data(), 0, BITMAP_SIZE);

  const unsigned char *ptr = inPtr;
  // minNonZero = *(reinterpret_cast<const unsigned short *>(ptr));
  tinyexr::cpy2(&minNonZero, reinterpret_cast<const unsigned short *>(ptr));
  // maxNonZero = *(reinterpret_cast<const unsigned short *>(ptr + 2));
  tinyexr::cpy2(&maxNonZero, reinterpret_cast<const unsigned short *>(ptr + 2));
  ptr += 4;

  if (maxNonZero >= BITMAP_SIZE) {
    return false;
  }

  if (minNonZero <= maxNonZero) {
    memcpy(reinterpret_cast<char *>(&bitmap[0] + minNonZero), ptr,
           maxNonZero - minNonZero + 1);
    ptr += maxNonZero - minNonZero + 1;
  }

  std::vector<unsigned short> lut(USHORT_RANGE);
  memset(lut.data(), 0, sizeof(unsigned short) * USHORT_RANGE);
  unsigned short maxValue = reverseLutFromBitmap(bitmap.data(), lut.data());

  //
  // Huffman decoding
  //

  int length;

  // length = *(reinterpret_cast<const int *>(ptr));
  tinyexr::cpy4(&length, reinterpret_cast<const int *>(ptr));
  ptr += sizeof(int);

  if (size_t((ptr - inPtr) + length) > inLen) {
    return false;
  }

  std::vector<unsigned short> tmpBuffer(tmpBufSize);
  hufUncompress(reinterpret_cast<const char *>(ptr), length, &tmpBuffer);

  //
  // Wavelet decoding
  //

  std::vector<PIZChannelData> channelData(static_cast<size_t>(num_channels));

  unsigned short *tmpBufferEnd = &tmpBuffer.at(0);

  for (size_t i = 0; i < static_cast<size_t>(num_channels); ++i) {
    const EXRChannelInfo &chan = channels[i];

    size_t pixelSize = sizeof(int);  // UINT and FLOAT
    if (chan.pixel_type == TINYEXR_PIXELTYPE_HALF) {
      pixelSize = sizeof(short);
    }

    channelData[i].start = tmpBufferEnd;
    channelData[i].end = channelData[i].start;
    channelData[i].nx = data_width;
    channelData[i].ny = num_lines;
    // channelData[i].ys = 1;
    channelData[i].size = static_cast<int>(pixelSize / sizeof(short));

    tmpBufferEnd += channelData[i].nx * channelData[i].ny * channelData[i].size;
  }

  for (size_t i = 0; i < channelData.size(); ++i) {
    PIZChannelData &cd = channelData[i];

    for (int j = 0; j < cd.size; ++j) {
      wav2Decode(cd.start + j, cd.nx, cd.size, cd.ny, cd.nx * cd.size,
                 maxValue);
    }
  }

  //
  // Expand the pixel data to their original range
  //

  applyLut(lut.data(), &tmpBuffer.at(0), static_cast<int>(tmpBufSize));

  for (int y = 0; y < num_lines; y++) {
    for (size_t i = 0; i < channelData.size(); ++i) {
      PIZChannelData &cd = channelData[i];

      // if (modp (y, cd.ys) != 0)
      //    continue;

      size_t n = static_cast<size_t>(cd.nx * cd.size);
      memcpy(outPtr, cd.end, static_cast<size_t>(n * sizeof(unsigned short)));
      outPtr += n * sizeof(unsigned short);
      cd.end += n;
    }
  }

  return true;
}
#endif  // TINYEXR_USE_PIZ

#if TINYEXR_USE_ZFP
struct ZFPCompressionParam {
  double rate;
  int precision;
  double tolerance;
  int type;  // TINYEXR_ZFP_COMPRESSIONTYPE_*

  ZFPCompressionParam() {
    type = TINYEXR_ZFP_COMPRESSIONTYPE_RATE;
    rate = 2.0;
    precision = 0;
    tolerance = 0.0f;
  }
};

bool FindZFPCompressionParam(ZFPCompressionParam *param,
                             const EXRAttribute *attributes,
                             int num_attributes) {
  bool foundType = false;

  for (int i = 0; i < num_attributes; i++) {
    if ((strcmp(attributes[i].name, "zfpCompressionType") == 0) &&
        (attributes[i].size == 1)) {
      param->type = static_cast<int>(attributes[i].value[0]);

      foundType = true;
    }
  }

  if (!foundType) {
    return false;
  }

  if (param->type == TINYEXR_ZFP_COMPRESSIONTYPE_RATE) {
    for (int i = 0; i < num_attributes; i++) {
      if ((strcmp(attributes[i].name, "zfpCompressionRate") == 0) &&
          (attributes[i].size == 8)) {
        param->rate = *(reinterpret_cast<double *>(attributes[i].value));
        return true;
      }
    }
  } else if (param->type == TINYEXR_ZFP_COMPRESSIONTYPE_PRECISION) {
    for (int i = 0; i < num_attributes; i++) {
      if ((strcmp(attributes[i].name, "zfpCompressionPrecision") == 0) &&
          (attributes[i].size == 4)) {
        param->rate = *(reinterpret_cast<int *>(attributes[i].value));
        return true;
      }
    }
  } else if (param->type == TINYEXR_ZFP_COMPRESSIONTYPE_ACCURACY) {
    for (int i = 0; i < num_attributes; i++) {
      if ((strcmp(attributes[i].name, "zfpCompressionTolerance") == 0) &&
          (attributes[i].size == 8)) {
        param->tolerance = *(reinterpret_cast<double *>(attributes[i].value));
        return true;
      }
    }
  } else {
    assert(0);
  }

  return false;
}

// Assume pixel format is FLOAT for all channels.
static bool DecompressZfp(float *dst, int dst_width, int dst_num_lines,
                          int num_channels, const unsigned char *src,
                          unsigned long src_size,
                          const ZFPCompressionParam &param) {
  size_t uncompressed_size = dst_width * dst_num_lines * num_channels;

  if (uncompressed_size == src_size) {
    // Data is not compressed(Issue 40).
    memcpy(dst, src, src_size);
  }

  zfp_stream *zfp = NULL;
  zfp_field *field = NULL;

  assert((dst_width % 4) == 0);
  assert((dst_num_lines % 4) == 0);

  if ((dst_width & 3U) || (dst_num_lines & 3U)) {
    return false;
  }

  field =
      zfp_field_2d(reinterpret_cast<void *>(const_cast<unsigned char *>(src)),
                   zfp_type_float, dst_width, dst_num_lines * num_channels);
  zfp = zfp_stream_open(NULL);

  if (param.type == TINYEXR_ZFP_COMPRESSIONTYPE_RATE) {
    zfp_stream_set_rate(zfp, param.rate, zfp_type_float, /* dimention */ 2,
                        /* write random access */ 0);
  } else if (param.type == TINYEXR_ZFP_COMPRESSIONTYPE_PRECISION) {
    zfp_stream_set_precision(zfp, param.precision, zfp_type_float);
  } else if (param.type == TINYEXR_ZFP_COMPRESSIONTYPE_ACCURACY) {
    zfp_stream_set_accuracy(zfp, param.tolerance, zfp_type_float);
  } else {
    assert(0);
  }

  size_t buf_size = zfp_stream_maximum_size(zfp, field);
  std::vector<unsigned char> buf(buf_size);
  memcpy(&buf.at(0), src, src_size);

  bitstream *stream = stream_open(&buf.at(0), buf_size);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  size_t image_size = dst_width * dst_num_lines;

  for (int c = 0; c < num_channels; c++) {
    // decompress 4x4 pixel block.
    for (int y = 0; y < dst_num_lines; y += 4) {
      for (int x = 0; x < dst_width; x += 4) {
        float fblock[16];
        zfp_decode_block_float_2(zfp, fblock);
        for (int j = 0; j < 4; j++) {
          for (int i = 0; i < 4; i++) {
            dst[c * image_size + ((y + j) * dst_width + (x + i))] =
                fblock[j * 4 + i];
          }
        }
      }
    }
  }

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);

  return true;
}

// Assume pixel format is FLOAT for all channels.
bool CompressZfp(std::vector<unsigned char> *outBuf, unsigned int *outSize,
                 const float *inPtr, int width, int num_lines, int num_channels,
                 const ZFPCompressionParam &param) {
  zfp_stream *zfp = NULL;
  zfp_field *field = NULL;

  assert((width % 4) == 0);
  assert((num_lines % 4) == 0);

  if ((width & 3U) || (num_lines & 3U)) {
    return false;
  }

  // create input array.
  field = zfp_field_2d(reinterpret_cast<void *>(const_cast<float *>(inPtr)),
                       zfp_type_float, width, num_lines * num_channels);

  zfp = zfp_stream_open(NULL);

  if (param.type == TINYEXR_ZFP_COMPRESSIONTYPE_RATE) {
    zfp_stream_set_rate(zfp, param.rate, zfp_type_float, 2, 0);
  } else if (param.type == TINYEXR_ZFP_COMPRESSIONTYPE_PRECISION) {
    zfp_stream_set_precision(zfp, param.precision, zfp_type_float);
  } else if (param.type == TINYEXR_ZFP_COMPRESSIONTYPE_ACCURACY) {
    zfp_stream_set_accuracy(zfp, param.tolerance, zfp_type_float);
  } else {
    assert(0);
  }

  size_t buf_size = zfp_stream_maximum_size(zfp, field);

  outBuf->resize(buf_size);

  bitstream *stream = stream_open(&outBuf->at(0), buf_size);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_field_free(field);

  size_t image_size = width * num_lines;

  for (int c = 0; c < num_channels; c++) {
    // compress 4x4 pixel block.
    for (int y = 0; y < num_lines; y += 4) {
      for (int x = 0; x < width; x += 4) {
        float fblock[16];
        for (int j = 0; j < 4; j++) {
          for (int i = 0; i < 4; i++) {
            fblock[j * 4 + i] =
                inPtr[c * image_size + ((y + j) * width + (x + i))];
          }
        }
        zfp_encode_block_float_2(zfp, fblock);
      }
    }
  }

  zfp_stream_flush(zfp);
  (*outSize) = zfp_stream_compressed_size(zfp);

  zfp_stream_close(zfp);

  return true;
}

#endif

//
// -----------------------------------------------------------------
//

// TODO(syoyo): Refactor function arguments.
static bool DecodePixelData(/* out */ unsigned char **out_images,
                            const int *requested_pixel_types,
                            const unsigned char *data_ptr, size_t data_len,
                            int compression_type, int line_order, int width,
                            int height, int x_stride, int y, int line_no,
                            int num_lines, size_t pixel_data_size,
                            size_t num_attributes,
                            const EXRAttribute *attributes, size_t num_channels,
                            const EXRChannelInfo *channels,
                            const std::vector<size_t> &channel_offset_list) {
  if (compression_type == TINYEXR_COMPRESSIONTYPE_PIZ) {  // PIZ
#if TINYEXR_USE_PIZ
    if ((width == 0) || (num_lines == 0) || (pixel_data_size == 0)) {
      // Invalid input #90
      return false;
    }

    // Allocate original data size.
    std::vector<unsigned char> outBuf(static_cast<size_t>(
        static_cast<size_t>(width * num_lines) * pixel_data_size));
    size_t tmpBufLen = outBuf.size();

    bool ret = tinyexr::DecompressPiz(
        reinterpret_cast<unsigned char *>(&outBuf.at(0)), data_ptr, tmpBufLen,
        data_len, static_cast<int>(num_channels), channels, width, num_lines);

    if (!ret) {
      return false;
    }

    // For PIZ_COMPRESSION:
    //   pixel sample data for channel 0 for scanline 0
    //   pixel sample data for channel 1 for scanline 0
    //   pixel sample data for channel ... for scanline 0
    //   pixel sample data for channel n for scanline 0
    //   pixel sample data for channel 0 for scanline 1
    //   pixel sample data for channel 1 for scanline 1
    //   pixel sample data for channel ... for scanline 1
    //   pixel sample data for channel n for scanline 1
    //   ...
    for (size_t c = 0; c < static_cast<size_t>(num_channels); c++) {
      if (channels[c].pixel_type == TINYEXR_PIXELTYPE_HALF) {
        for (size_t v = 0; v < static_cast<size_t>(num_lines); v++) {
          const unsigned short *line_ptr = reinterpret_cast<unsigned short *>(
              &outBuf.at(v * pixel_data_size * static_cast<size_t>(width) +
                         channel_offset_list[c] * static_cast<size_t>(width)));
          for (size_t u = 0; u < static_cast<size_t>(width); u++) {
            FP16 hf;

            // hf.u = line_ptr[u];
            // use `cpy` to avoid unaligned memory access when compiler's
            // optimization is on.
            tinyexr::cpy2(&(hf.u), line_ptr + u);

            tinyexr::swap2(reinterpret_cast<unsigned short *>(&hf.u));

            if (requested_pixel_types[c] == TINYEXR_PIXELTYPE_HALF) {
              unsigned short *image =
                  reinterpret_cast<unsigned short **>(out_images)[c];
              if (line_order == 0) {
                image += (static_cast<size_t>(line_no) + v) *
                             static_cast<size_t>(x_stride) +
                         u;
              } else {
                image += static_cast<size_t>(
                             (height - 1 - (line_no + static_cast<int>(v)))) *
                             static_cast<size_t>(x_stride) +
                         u;
              }
              *image = hf.u;
            } else {  // HALF -> FLOAT
              FP32 f32 = half_to_float(hf);
              float *image = reinterpret_cast<float **>(out_images)[c];
              size_t offset = 0;
              if (line_order == 0) {
                offset = (static_cast<size_t>(line_no) + v) *
                             static_cast<size_t>(x_stride) +
                         u;
              } else {
                offset = static_cast<size_t>(
                             (height - 1 - (line_no + static_cast<int>(v)))) *
                             static_cast<size_t>(x_stride) +
                         u;
              }
              image += offset;
              *image = f32.f;
            }
          }
        }
      } else if (channels[c].pixel_type == TINYEXR_PIXELTYPE_UINT) {
        assert(requested_pixel_types[c] == TINYEXR_PIXELTYPE_UINT);

        for (size_t v = 0; v < static_cast<size_t>(num_lines); v++) {
          const unsigned int *line_ptr = reinterpret_cast<unsigned int *>(
              &outBuf.at(v * pixel_data_size * static_cast<size_t>(width) +
                         channel_offset_list[c] * static_cast<size_t>(width)));
          for (size_t u = 0; u < static_cast<size_t>(width); u++) {
            unsigned int val;
            // val = line_ptr[u];
            tinyexr::cpy4(&val, line_ptr + u);

            tinyexr::swap4(&val);

            unsigned int *image =
                reinterpret_cast<unsigned int **>(out_images)[c];
            if (line_order == 0) {
              image += (static_cast<size_t>(line_no) + v) *
                           static_cast<size_t>(x_stride) +
                       u;
            } else {
              image += static_cast<size_t>(
                           (height - 1 - (line_no + static_cast<int>(v)))) *
                           static_cast<size_t>(x_stride) +
                       u;
            }
            *image = val;
          }
        }
      } else if (channels[c].pixel_type == TINYEXR_PIXELTYPE_FLOAT) {
        assert(requested_pixel_types[c] == TINYEXR_PIXELTYPE_FLOAT);
        for (size_t v = 0; v < static_cast<size_t>(num_lines); v++) {
          const float *line_ptr = reinterpret_cast<float *>(&outBuf.at(
              v * pixel_data_size * static_cast<size_t>(x_stride) +
              channel_offset_list[c] * static_cast<size_t>(x_stride)));
          for (size_t u = 0; u < static_cast<size_t>(width); u++) {
            float val;
            // val = line_ptr[u];
            tinyexr::cpy4(&val, line_ptr + u);

            tinyexr::swap4(reinterpret_cast<unsigned int *>(&val));

            float *image = reinterpret_cast<float **>(out_images)[c];
            if (line_order == 0) {
              image += (static_cast<size_t>(line_no) + v) *
                           static_cast<size_t>(x_stride) +
                       u;
            } else {
              image += static_cast<size_t>(
                           (height - 1 - (line_no + static_cast<int>(v)))) *
                           static_cast<size_t>(x_stride) +
                       u;
            }
            *image = val;
          }
        }
      } else {
        assert(0);
      }
    }
#else
    assert(0 && "PIZ is enabled in this build");
    return false;
#endif

  } else if (compression_type == TINYEXR_COMPRESSIONTYPE_ZIPS ||
             compression_type == TINYEXR_COMPRESSIONTYPE_ZIP) {
    // Allocate original data size.
    std::vector<unsigned char> outBuf(static_cast<size_t>(width) *
                                      static_cast<size_t>(num_lines) *
                                      pixel_data_size);

    unsigned long dstLen = static_cast<unsigned long>(outBuf.size());
    assert(dstLen > 0);
    if (!tinyexr::DecompressZip(
            reinterpret_cast<unsigned char *>(&outBuf.at(0)), &dstLen, data_ptr,
            static_cast<unsigned long>(data_len))) {
      return false;
    }

    // For ZIP_COMPRESSION:
    //   pixel sample data for channel 0 for scanline 0
    //   pixel sample data for channel 1 for scanline 0
    //   pixel sample data for channel ... for scanline 0
    //   pixel sample data for channel n for scanline 0
    //   pixel sample data for channel 0 for scanline 1
    //   pixel sample data for channel 1 for scanline 1
    //   pixel sample data for channel ... for scanline 1
    //   pixel sample data for channel n for scanline 1
    //   ...
    for (size_t c = 0; c < static_cast<size_t>(num_channels); c++) {
      if (channels[c].pixel_type == TINYEXR_PIXELTYPE_HALF) {
        for (size_t v = 0; v < static_cast<size_t>(num_lines); v++) {
          const unsigned short *line_ptr = reinterpret_cast<unsigned short *>(
              &outBuf.at(v * static_cast<size_t>(pixel_data_size) *
                             static_cast<size_t>(width) +
                         channel_offset_list[c] * static_cast<size_t>(width)));
          for (size_t u = 0; u < static_cast<size_t>(width); u++) {
            tinyexr::FP16 hf;

            // hf.u = line_ptr[u];
            tinyexr::cpy2(&(hf.u), line_ptr + u);

            tinyexr::swap2(reinterpret_cast<unsigned short *>(&hf.u));

            if (requested_pixel_types[c] == TINYEXR_PIXELTYPE_HALF) {
              unsigned short *image =
                  reinterpret_cast<unsigned short **>(out_images)[c];
              if (line_order == 0) {
                image += (static_cast<size_t>(line_no) + v) *
                             static_cast<size_t>(x_stride) +
                         u;
              } else {
                image += (static_cast<size_t>(height) - 1U -
                          (static_cast<size_t>(line_no) + v)) *
                             static_cast<size_t>(x_stride) +
                         u;
              }
              *image = hf.u;
            } else {  // HALF -> FLOAT
              tinyexr::FP32 f32 = half_to_float(hf);
              float *image = reinterpret_cast<float **>(out_images)[c];
              size_t offset = 0;
              if (line_order == 0) {
                offset = (static_cast<size_t>(line_no) + v) *
                             static_cast<size_t>(x_stride) +
                         u;
              } else {
                offset = (static_cast<size_t>(height) - 1U -
                          (static_cast<size_t>(line_no) + v)) *
                             static_cast<size_t>(x_stride) +
                         u;
              }
              image += offset;

              *image = f32.f;
            }
          }
        }
      } else if (channels[c].pixel_type == TINYEXR_PIXELTYPE_UINT) {
        assert(requested_pixel_types[c] == TINYEXR_PIXELTYPE_UINT);

        for (size_t v = 0; v < static_cast<size_t>(num_lines); v++) {
          const unsigned int *line_ptr = reinterpret_cast<unsigned int *>(
              &outBuf.at(v * pixel_data_size * static_cast<size_t>(width) +
                         channel_offset_list[c] * static_cast<size_t>(width)));
          for (size_t u = 0; u < static_cast<size_t>(width); u++) {
            unsigned int val;
            // val = line_ptr[u];
            tinyexr::cpy4(&val, line_ptr + u);

            tinyexr::swap4(&val);

            unsigned int *image =
                reinterpret_cast<unsigned int **>(out_images)[c];
            if (line_order == 0) {
              image += (static_cast<size_t>(line_no) + v) *
                           static_cast<size_t>(x_stride) +
                       u;
            } else {
              image += (static_cast<size_t>(height) - 1U -
                        (static_cast<size_t>(line_no) + v)) *
                           static_cast<size_t>(x_stride) +
                       u;
            }
            *image = val;
          }
        }
      } else if (channels[c].pixel_type == TINYEXR_PIXELTYPE_FLOAT) {
        assert(requested_pixel_types[c] == TINYEXR_PIXELTYPE_FLOAT);
        for (size_t v = 0; v < static_cast<size_t>(num_lines); v++) {
          const float *line_ptr = reinterpret_cast<float *>(
              &outBuf.at(v * pixel_data_size * static_cast<size_t>(width) +
                         channel_offset_list[c] * static_cast<size_t>(width)));
          for (size_t u = 0; u < static_cast<size_t>(width); u++) {
            float val;
            // val = line_ptr[u];
            tinyexr::cpy4(&val, line_ptr + u);

            tinyexr::swap4(reinterpret_cast<unsigned int *>(&val));

            float *image = reinterpret_cast<float **>(out_images)[c];
            if (line_order == 0) {
              image += (static_cast<size_t>(line_no) + v) *
                           static_cast<size_t>(x_stride) +
                       u;
            } else {
              image += (static_cast<size_t>(height) - 1U -
                        (static_cast<size_t>(line_no) + v)) *
                           static_cast<size_t>(x_stride) +
                       u;
            }
            *image = val;
          }
        }
      } else {
        assert(0);
        return false;
      }
    }
  } else if (compression_type == TINYEXR_COMPRESSIONTYPE_RLE) {
    // Allocate original data size.
    std::vector<unsigned char> outBuf(static_cast<size_t>(width) *
                                      static_cast<size_t>(num_lines) *
                                      pixel_data_size);

    unsigned long dstLen = static_cast<unsigned long>(outBuf.size());
    if (dstLen == 0) {
      return false;
    }

    if (!tinyexr::DecompressRle(reinterpret_cast<unsigned char *>(&outBuf.at(0)),
                           dstLen, data_ptr,
                           static_cast<unsigned long>(data_len))) {
      return false;
    }

    // For RLE_COMPRESSION:
    //   pixel sample data for channel 0 for scanline 0
    //   pixel sample data for channel 1 for scanline 0
    //   pixel sample data for channel ... for scanline 0
    //   pixel sample data for channel n for scanline 0
    //   pixel sample data for channel 0 for scanline 1
    //   pixel sample data for channel 1 for scanline 1
    //   pixel sample data for channel ... for scanline 1
    //   pixel sample data for channel n for scanline 1
    //   ...
    for (size_t c = 0; c < static_cast<size_t>(num_channels); c++) {
      if (channels[c].pixel_type == TINYEXR_PIXELTYPE_HALF) {
        for (size_t v = 0; v < static_cast<size_t>(num_lines); v++) {
          const unsigned short *line_ptr = reinterpret_cast<unsigned short *>(
              &outBuf.at(v * static_cast<size_t>(pixel_data_size) *
                             static_cast<size_t>(width) +
                         channel_offset_list[c] * static_cast<size_t>(width)));
          for (size_t u = 0; u < static_cast<size_t>(width); u++) {
            tinyexr::FP16 hf;

            // hf.u = line_ptr[u];
            tinyexr::cpy2(&(hf.u), line_ptr + u);

            tinyexr::swap2(reinterpret_cast<unsigned short *>(&hf.u));

            if (requested_pixel_types[c] == TINYEXR_PIXELTYPE_HALF) {
              unsigned short *image =
                  reinterpret_cast<unsigned short **>(out_images)[c];
              if (line_order == 0) {
                image += (static_cast<size_t>(line_no) + v) *
                             static_cast<size_t>(x_stride) +
                         u;
              } else {
                image += (static_cast<size_t>(height) - 1U -
                          (static_cast<size_t>(line_no) + v)) *
                             static_cast<size_t>(x_stride) +
                         u;
              }
              *image = hf.u;
            } else {  // HALF -> FLOAT
              tinyexr::FP32 f32 = half_to_float(hf);
              float *image = reinterpret_cast<float **>(out_images)[c];
              if (line_order == 0) {
                image += (static_cast<size_t>(line_no) + v) *
                             static_cast<size_t>(x_stride) +
                         u;
              } else {
                image += (static_cast<size_t>(height) - 1U -
                          (static_cast<size_t>(line_no) + v)) *
                             static_cast<size_t>(x_stride) +
                         u;
              }
              *image = f32.f;
            }
          }
        }
      } else if (channels[c].pixel_type == TINYEXR_PIXELTYPE_UINT) {
        assert(requested_pixel_types[c] == TINYEXR_PIXELTYPE_UINT);

        for (size_t v = 0; v < static_cast<size_t>(num_lines); v++) {
          const unsigned int *line_ptr = reinterpret_cast<unsigned int *>(
              &outBuf.at(v * pixel_data_size * static_cast<size_t>(width) +
                         channel_offset_list[c] * static_cast<size_t>(width)));
          for (size_t u = 0; u < static_cast<size_t>(width); u++) {
            unsigned int val;
            // val = line_ptr[u];
            tinyexr::cpy4(&val, line_ptr + u);

            tinyexr::swap4(&val);

            unsigned int *image =
                reinterpret_cast<unsigned int **>(out_images)[c];
            if (line_order == 0) {
              image += (static_cast<size_t>(line_no) + v) *
                           static_cast<size_t>(x_stride) +
                       u;
            } else {
              image += (static_cast<size_t>(height) - 1U -
                        (static_cast<size_t>(line_no) + v)) *
                           static_cast<size_t>(x_stride) +
                       u;
            }
            *image = val;
          }
        }
      } else if (channels[c].pixel_type == TINYEXR_PIXELTYPE_FLOAT) {
        assert(requested_pixel_types[c] == TINYEXR_PIXELTYPE_FLOAT);
        for (size_t v = 0; v < static_cast<size_t>(num_lines); v++) {
          const float *line_ptr = reinterpret_cast<float *>(
              &outBuf.at(v * pixel_data_size * static_cast<size_t>(width) +
                         channel_offset_list[c] * static_cast<size_t>(width)));
          for (size_t u = 0; u < static_cast<size_t>(width); u++) {
            float val;
            // val = line_ptr[u];
            tinyexr::cpy4(&val, line_ptr + u);

            tinyexr::swap4(reinterpret_cast<unsigned int *>(&val));

            float *image = reinterpret_cast<float **>(out_images)[c];
            if (line_order == 0) {
              image += (static_cast<size_t>(line_no) + v) *
                           static_cast<size_t>(x_stride) +
                       u;
            } else {
              image += (static_cast<size_t>(height) - 1U -
                        (static_cast<size_t>(line_no) + v)) *
                           static_cast<size_t>(x_stride) +
                       u;
            }
            *image = val;
          }
        }
      } else {
        assert(0);
        return false;
      }
    }
  } else if (compression_type == TINYEXR_COMPRESSIONTYPE_ZFP) {
#if TINYEXR_USE_ZFP
    tinyexr::ZFPCompressionParam zfp_compression_param;
    if (!FindZFPCompressionParam(&zfp_compression_param, attributes,
                                 num_attributes)) {
      assert(0);
      return false;
    }

    // Allocate original data size.
    std::vector<unsigned char> outBuf(static_cast<size_t>(width) *
                                      static_cast<size_t>(num_lines) *
                                      pixel_data_size);

    unsigned long dstLen = outBuf.size();
    assert(dstLen > 0);
    tinyexr::DecompressZfp(reinterpret_cast<float *>(&outBuf.at(0)), width,
                           num_lines, num_channels, data_ptr,
                           static_cast<unsigned long>(data_len),
                           zfp_compression_param);

    // For ZFP_COMPRESSION:
    //   pixel sample data for channel 0 for scanline 0
    //   pixel sample data for channel 1 for scanline 0
    //   pixel sample data for channel ... for scanline 0
    //   pixel sample data for channel n for scanline 0
    //   pixel sample data for channel 0 for scanline 1
    //   pixel sample data for channel 1 for scanline 1
    //   pixel sample data for channel ... for scanline 1
    //   pixel sample data for channel n for scanline 1
    //   ...
    for (size_t c = 0; c < static_cast<size_t>(num_channels); c++) {
      assert(channels[c].pixel_type == TINYEXR_PIXELTYPE_FLOAT);
      if (channels[c].pixel_type == TINYEXR_PIXELTYPE_FLOAT) {
        assert(requested_pixel_types[c] == TINYEXR_PIXELTYPE_FLOAT);
        for (size_t v = 0; v < static_cast<size_t>(num_lines); v++) {
          const float *line_ptr = reinterpret_cast<float *>(
              &outBuf.at(v * pixel_data_size * static_cast<size_t>(width) +
                         channel_offset_list[c] * static_cast<size_t>(width)));
          for (size_t u = 0; u < static_cast<size_t>(width); u++) {
            float val;
            tinyexr::cpy4(&val, line_ptr + u);

            tinyexr::swap4(reinterpret_cast<unsigned int *>(&val));

            float *image = reinterpret_cast<float **>(out_images)[c];
            if (line_order == 0) {
              image += (static_cast<size_t>(line_no) + v) *
                           static_cast<size_t>(x_stride) +
                       u;
            } else {
              image += (static_cast<size_t>(height) - 1U -
                        (static_cast<size_t>(line_no) + v)) *
                           static_cast<size_t>(x_stride) +
                       u;
            }
            *image = val;
          }
        }
      } else {
        assert(0);
        return false;
      }
    }
#else
    (void)attributes;
    (void)num_attributes;
    (void)num_channels;
    assert(0);
    return false;
#endif
  } else if (compression_type == TINYEXR_COMPRESSIONTYPE_NONE) {
    for (size_t c = 0; c < num_channels; c++) {
      for (size_t v = 0; v < static_cast<size_t>(num_lines); v++) {
        if (channels[c].pixel_type == TINYEXR_PIXELTYPE_HALF) {
          const unsigned short *line_ptr =
              reinterpret_cast<const unsigned short *>(
                  data_ptr + v * pixel_data_size * size_t(width) +
                  channel_offset_list[c] * static_cast<size_t>(width));

          if (requested_pixel_types[c] == TINYEXR_PIXELTYPE_HALF) {
            unsigned short *outLine =
                reinterpret_cast<unsigned short *>(out_images[c]);
            if (line_order == 0) {
              outLine += (size_t(y) + v) * size_t(x_stride);
            } else {
              outLine +=
                  (size_t(height) - 1 - (size_t(y) + v)) * size_t(x_stride);
            }

            for (int u = 0; u < width; u++) {
              tinyexr::FP16 hf;

              // hf.u = line_ptr[u];
              tinyexr::cpy2(&(hf.u), line_ptr + u);

              tinyexr::swap2(reinterpret_cast<unsigned short *>(&hf.u));

              outLine[u] = hf.u;
            }
          } else if (requested_pixel_types[c] == TINYEXR_PIXELTYPE_FLOAT) {
            float *outLine = reinterpret_cast<float *>(out_images[c]);
            if (line_order == 0) {
              outLine += (size_t(y) + v) * size_t(x_stride);
            } else {
              outLine +=
                  (size_t(height) - 1 - (size_t(y) + v)) * size_t(x_stride);
            }

            if (reinterpret_cast<const unsigned char *>(line_ptr + width) >
                (data_ptr + data_len)) {
              // Insufficient data size
              return false;
            }

            for (int u = 0; u < width; u++) {
              tinyexr::FP16 hf;

              // address may not be aliged. use byte-wise copy for safety.#76
              // hf.u = line_ptr[u];
              tinyexr::cpy2(&(hf.u), line_ptr + u);

              tinyexr::swap2(reinterpret_cast<unsigned short *>(&hf.u));

              tinyexr::FP32 f32 = half_to_float(hf);

              outLine[u] = f32.f;
            }
          } else {
            assert(0);
            return false;
          }
        } else if (channels[c].pixel_type == TINYEXR_PIXELTYPE_FLOAT) {
          const float *line_ptr = reinterpret_cast<const float *>(
              data_ptr + v * pixel_data_size * size_t(width) +
              channel_offset_list[c] * static_cast<size_t>(width));

          float *outLine = reinterpret_cast<float *>(out_images[c]);
          if (line_order == 0) {
            outLine += (size_t(y) + v) * size_t(x_stride);
          } else {
            outLine +=
                (size_t(height) - 1 - (size_t(y) + v)) * size_t(x_stride);
          }

          if (reinterpret_cast<const unsigned char *>(line_ptr + width) >
              (data_ptr + data_len)) {
            // Insufficient data size
            return false;
          }

          for (int u = 0; u < width; u++) {
            float val;
            tinyexr::cpy4(&val, line_ptr + u);

            tinyexr::swap4(reinterpret_cast<unsigned int *>(&val));

            outLine[u] = val;
          }
        } else if (channels[c].pixel_type == TINYEXR_PIXELTYPE_UINT) {
          const unsigned int *line_ptr = reinterpret_cast<const unsigned int *>(
              data_ptr + v * pixel_data_size * size_t(width) +
              channel_offset_list[c] * static_cast<size_t>(width));

          unsigned int *outLine =
              reinterpret_cast<unsigned int *>(out_images[c]);
          if (line_order == 0) {
            outLine += (size_t(y) + v) * size_t(x_stride);
          } else {
            outLine +=
                (size_t(height) - 1 - (size_t(y) + v)) * size_t(x_stride);
          }

          for (int u = 0; u < width; u++) {
            if (reinterpret_cast<const unsigned char *>(line_ptr + u) >=
                (data_ptr + data_len)) {
              // Corrupsed data?
              return false;
            }

            unsigned int val;
            tinyexr::cpy4(&val, line_ptr + u);

            tinyexr::swap4(reinterpret_cast<unsigned int *>(&val));

            outLine[u] = val;
          }
        }
      }
    }
  }

  return true;
}

static void DecodeTiledPixelData(
    unsigned char **out_images, int *width, int *height,
    const int *requested_pixel_types, const unsigned char *data_ptr,
    size_t data_len, int compression_type, int line_order, int data_width,
    int data_height, int tile_offset_x, int tile_offset_y, int tile_size_x,
    int tile_size_y, size_t pixel_data_size, size_t num_attributes,
    const EXRAttribute *attributes, size_t num_channels,
    const EXRChannelInfo *channels,
    const std::vector<size_t> &channel_offset_list) {
  assert(tile_offset_x * tile_size_x < data_width);
  assert(tile_offset_y * tile_size_y < data_height);

  // Compute actual image size in a tile.
  if ((tile_offset_x + 1) * tile_size_x >= data_width) {
    (*width) = data_width - (tile_offset_x * tile_size_x);
  } else {
    (*width) = tile_size_x;
  }

  if ((tile_offset_y + 1) * tile_size_y >= data_height) {
    (*height) = data_height - (tile_offset_y * tile_size_y);
  } else {
    (*height) = tile_size_y;
  }

  // Image size = tile size.
  DecodePixelData(out_images, requested_pixel_types, data_ptr, data_len,
                  compression_type, line_order, (*width), tile_size_y,
                  /* stride */ tile_size_x, /* y */ 0, /* line_no */ 0,
                  (*height), pixel_data_size, num_attributes, attributes,
                  num_channels, channels, channel_offset_list);
}

static bool ComputeChannelLayout(std::vector<size_t> *channel_offset_list,
                                 int *pixel_data_size, size_t *channel_offset,
                                 int num_channels,
                                 const EXRChannelInfo *channels) {
  channel_offset_list->resize(static_cast<size_t>(num_channels));

  (*pixel_data_size) = 0;
  (*channel_offset) = 0;

  for (size_t c = 0; c < static_cast<size_t>(num_channels); c++) {
    (*channel_offset_list)[c] = (*channel_offset);
    if (channels[c].pixel_type == TINYEXR_PIXELTYPE_HALF) {
      (*pixel_data_size) += sizeof(unsigned short);
      (*channel_offset) += sizeof(unsigned short);
    } else if (channels[c].pixel_type == TINYEXR_PIXELTYPE_FLOAT) {
      (*pixel_data_size) += sizeof(float);
      (*channel_offset) += sizeof(float);
    } else if (channels[c].pixel_type == TINYEXR_PIXELTYPE_UINT) {
      (*pixel_data_size) += sizeof(unsigned int);
      (*channel_offset) += sizeof(unsigned int);
    } else {
      // ???
      return false;
    }
  }
  return true;
}

static unsigned char **AllocateImage(int num_channels,
                                     const EXRChannelInfo *channels,
                                     const int *requested_pixel_types,
                                     int data_width, int data_height) {
  unsigned char **images =
      reinterpret_cast<unsigned char **>(static_cast<float **>(
          malloc(sizeof(float *) * static_cast<size_t>(num_channels))));

  for (size_t c = 0; c < static_cast<size_t>(num_channels); c++) {
    size_t data_len =
        static_cast<size_t>(data_width) * static_cast<size_t>(data_height);
    if (channels[c].pixel_type == TINYEXR_PIXELTYPE_HALF) {
      // pixel_data_size += sizeof(unsigned short);
      // channel_offset += sizeof(unsigned short);
      // Alloc internal image for half type.
      if (requested_pixel_types[c] == TINYEXR_PIXELTYPE_HALF) {
        images[c] =
            reinterpret_cast<unsigned char *>(static_cast<unsigned short *>(
                malloc(sizeof(unsigned short) * data_len)));
      } else if (requested_pixel_types[c] == TINYEXR_PIXELTYPE_FLOAT) {
        images[c] = reinterpret_cast<unsigned char *>(
            static_cast<float *>(malloc(sizeof(float) * data_len)));
      } else {
        assert(0);
      }
    } else if (channels[c].pixel_type == TINYEXR_PIXELTYPE_FLOAT) {
      // pixel_data_size += sizeof(float);
      // channel_offset += sizeof(float);
      images[c] = reinterpret_cast<unsigned char *>(
          static_cast<float *>(malloc(sizeof(float) * data_len)));
    } else if (channels[c].pixel_type == TINYEXR_PIXELTYPE_UINT) {
      // pixel_data_size += sizeof(unsigned int);
      // channel_offset += sizeof(unsigned int);
      images[c] = reinterpret_cast<unsigned char *>(
          static_cast<unsigned int *>(malloc(sizeof(unsigned int) * data_len)));
    } else {
      assert(0);
    }
  }

  return images;
}

static int ParseEXRHeader(HeaderInfo *info, bool *empty_header,
                          const EXRVersion *version, std::string *err,
                          const unsigned char *buf, size_t size) {
  const char *marker = reinterpret_cast<const char *>(&buf[0]);

  if (empty_header) {
    (*empty_header) = false;
  }

  if (version->multipart) {
    if (size > 0 && marker[0] == '\0') {
      // End of header list.
      if (empty_header) {
        (*empty_header) = true;
      }
      return TINYEXR_SUCCESS;
    }
  }

  // According to the spec, the header of every OpenEXR file must contain at
  // least the following attributes:
  //
  // channels chlist
  // compression compression
  // dataWindow box2i
  // displayWindow box2i
  // lineOrder lineOrder
  // pixelAspectRatio float
  // screenWindowCenter v2f
  // screenWindowWidth float
  bool has_channels = false;
  bool has_compression = false;
  bool has_data_window = false;
  bool has_display_window = false;
  bool has_line_order = false;
  bool has_pixel_aspect_ratio = false;
  bool has_screen_window_center = false;
  bool has_screen_window_width = false;

  info->data_window[0] = 0;
  info->data_window[1] = 0;
  info->data_window[2] = 0;
  info->data_window[3] = 0;
  info->line_order = 0;  // @fixme
  info->display_window[0] = 0;
  info->display_window[1] = 0;
  info->display_window[2] = 0;
  info->display_window[3] = 0;
  info->screen_window_center[0] = 0.0f;
  info->screen_window_center[1] = 0.0f;
  info->screen_window_width = -1.0f;
  info->pixel_aspect_ratio = -1.0f;

  info->tile_size_x = -1;
  info->tile_size_y = -1;
  info->tile_level_mode = -1;
  info->tile_rounding_mode = -1;

  info->attributes.clear();

  // Read attributes
  size_t orig_size = size;
  for (size_t nattr = 0; nattr < TINYEXR_MAX_HEADER_ATTRIBUTES; nattr++) {
    if (0 == size) {
      if (err) {
        (*err) += "Insufficient data size for attributes.\n";
      }
      return TINYEXR_ERROR_INVALID_DATA;
    } else if (marker[0] == '\0') {
      size--;
      break;
    }

    std::string attr_name;
    std::string attr_type;
    std::vector<unsigned char> data;
    size_t marker_size;
    if (!tinyexr::ReadAttribute(&attr_name, &attr_type, &data, &marker_size,
                                marker, size)) {
      if (err) {
        (*err) += "Failed to read attribute.\n";
      }
      return TINYEXR_ERROR_INVALID_DATA;
    }
    marker += marker_size;
    size -= marker_size;

    if (version->tiled && attr_name.compare("tiles") == 0) {
      unsigned int x_size, y_size;
      unsigned char tile_mode;
      assert(data.size() == 9);
      memcpy(&x_size, &data.at(0), sizeof(int));
      memcpy(&y_size, &data.at(4), sizeof(int));
      tile_mode = data[8];
      tinyexr::swap4(&x_size);
      tinyexr::swap4(&y_size);

      info->tile_size_x = static_cast<int>(x_size);
      info->tile_size_y = static_cast<int>(y_size);

      // mode = levelMode + roundingMode * 16
      info->tile_level_mode = tile_mode & 0x3;
      info->tile_rounding_mode = (tile_mode >> 4) & 0x1;

    } else if (attr_name.compare("compression") == 0) {
      bool ok = false;
      if (data[0] < TINYEXR_COMPRESSIONTYPE_PIZ) {
        ok = true;
      }

      if (data[0] == TINYEXR_COMPRESSIONTYPE_PIZ) {
#if TINYEXR_USE_PIZ
        ok = true;
#else
        if (err) {
          (*err) = "PIZ compression is not supported.";
        }
        return TINYEXR_ERROR_UNSUPPORTED_FORMAT;
#endif
      }

      if (data[0] == TINYEXR_COMPRESSIONTYPE_ZFP) {
#if TINYEXR_USE_ZFP
        ok = true;
#else
        if (err) {
          (*err) = "ZFP compression is not supported.";
        }
        return TINYEXR_ERROR_UNSUPPORTED_FORMAT;
#endif
      }

      if (!ok) {
        if (err) {
          (*err) = "Unknown compression type.";
        }
        return TINYEXR_ERROR_UNSUPPORTED_FORMAT;
      }

      info->compression_type = static_cast<int>(data[0]);
      has_compression = true;

    } else if (attr_name.compare("channels") == 0) {
      // name: zero-terminated string, from 1 to 255 bytes long
      // pixel type: int, possible values are: UINT = 0 HALF = 1 FLOAT = 2
      // pLinear: unsigned char, possible values are 0 and 1
      // reserved: three chars, should be zero
      // xSampling: int
      // ySampling: int

      if (!ReadChannelInfo(info->channels, data)) {
        if (err) {
          (*err) += "Failed to parse channel info.\n";
        }
        return TINYEXR_ERROR_INVALID_DATA;
      }

      if (info->channels.size() < 1) {
        if (err) {
          (*err) += "# of channels is zero.\n";
        }
        return TINYEXR_ERROR_INVALID_DATA;
      }

      has_channels = true;

    } else if (attr_name.compare("dataWindow") == 0) {
      if (data.size() >= 16) {
        memcpy(&info->data_window[0], &data.at(0), sizeof(int));
        memcpy(&info->data_window[1], &data.at(4), sizeof(int));
        memcpy(&info->data_window[2], &data.at(8), sizeof(int));
        memcpy(&info->data_window[3], &data.at(12), sizeof(int));
        tinyexr::swap4(reinterpret_cast<unsigned int *>(&info->data_window[0]));
        tinyexr::swap4(reinterpret_cast<unsigned int *>(&info->data_window[1]));
        tinyexr::swap4(reinterpret_cast<unsigned int *>(&info->data_window[2]));
        tinyexr::swap4(reinterpret_cast<unsigned int *>(&info->data_window[3]));
        has_data_window = true;
      }
    } else if (attr_name.compare("displayWindow") == 0) {
      if (data.size() >= 16) {
        memcpy(&info->display_window[0], &data.at(0), sizeof(int));
        memcpy(&info->display_window[1], &data.at(4), sizeof(int));
        memcpy(&info->display_window[2], &data.at(8), sizeof(int));
        memcpy(&info->display_window[3], &data.at(12), sizeof(int));
        tinyexr::swap4(
            reinterpret_cast<unsigned int *>(&info->display_window[0]));
        tinyexr::swap4(
            reinterpret_cast<unsigned int *>(&info->display_window[1]));
        tinyexr::swap4(
            reinterpret_cast<unsigned int *>(&info->display_window[2]));
        tinyexr::swap4(
            reinterpret_cast<unsigned int *>(&info->display_window[3]));

        has_display_window = true;
      }
    } else if (attr_name.compare("lineOrder") == 0) {
      if (data.size() >= 1) {
        info->line_order = static_cast<int>(data[0]);
        has_line_order = true;
      }
    } else if (attr_name.compare("pixelAspectRatio") == 0) {
      if (data.size() >= sizeof(float)) {
        memcpy(&info->pixel_aspect_ratio, &data.at(0), sizeof(float));
        tinyexr::swap4(
            reinterpret_cast<unsigned int *>(&info->pixel_aspect_ratio));
        has_pixel_aspect_ratio = true;
      }
    } else if (attr_name.compare("screenWindowCenter") == 0) {
      if (data.size() >= 8) {
        memcpy(&info->screen_window_center[0], &data.at(0), sizeof(float));
        memcpy(&info->screen_window_center[1], &data.at(4), sizeof(float));
        tinyexr::swap4(
            reinterpret_cast<unsigned int *>(&info->screen_window_center[0]));
        tinyexr::swap4(
            reinterpret_cast<unsigned int *>(&info->screen_window_center[1]));
        has_screen_window_center = true;
      }
    } else if (attr_name.compare("screenWindowWidth") == 0) {
      if (data.size() >= sizeof(float)) {
        memcpy(&info->screen_window_width, &data.at(0), sizeof(float));
        tinyexr::swap4(
            reinterpret_cast<unsigned int *>(&info->screen_window_width));

        has_screen_window_width = true;
      }
    } else if (attr_name.compare("chunkCount") == 0) {
      if (data.size() >= sizeof(int)) {
        memcpy(&info->chunk_count, &data.at(0), sizeof(int));
        tinyexr::swap4(reinterpret_cast<unsigned int *>(&info->chunk_count));
      }
    } else {
      // Custom attribute(up to TINYEXR_MAX_CUSTOM_ATTRIBUTES)
      if (info->attributes.size() < TINYEXR_MAX_CUSTOM_ATTRIBUTES) {
        EXRAttribute attrib;
#ifdef _MSC_VER
        strncpy_s(attrib.name, attr_name.c_str(), 255);
        strncpy_s(attrib.type, attr_type.c_str(), 255);
#else
        strncpy(attrib.name, attr_name.c_str(), 255);
        strncpy(attrib.type, attr_type.c_str(), 255);
#endif
        attrib.name[255] = '\0';
        attrib.type[255] = '\0';
        attrib.size = static_cast<int>(data.size());
        attrib.value = static_cast<unsigned char *>(malloc(data.size()));
        memcpy(reinterpret_cast<char *>(attrib.value), &data.at(0),
               data.size());
        info->attributes.push_back(attrib);
      }
    }
  }

  // Check if required attributes exist
  {
    std::stringstream ss_err;

    if (!has_compression) {
      ss_err << "\"compression\" attribute not found in the header."
             << std::endl;
    }

    if (!has_channels) {
      ss_err << "\"channels\" attribute not found in the header." << std::endl;
    }

    if (!has_line_order) {
      ss_err << "\"lineOrder\" attribute not found in the header." << std::endl;
    }

    if (!has_display_window) {
      ss_err << "\"displayWindow\" attribute not found in the header."
             << std::endl;
    }

    if (!has_data_window) {
      ss_err << "\"dataWindow\" attribute not found in the header or invalid."
             << std::endl;
    }

    if (!has_pixel_aspect_ratio) {
      ss_err << "\"pixelAspectRatio\" attribute not found in the header."
             << std::endl;
    }

    if (!has_screen_window_width) {
      ss_err << "\"screenWindowWidth\" attribute not found in the header."
             << std::endl;
    }

    if (!has_screen_window_center) {
      ss_err << "\"screenWindowCenter\" attribute not found in the header."
             << std::endl;
    }

    if (!(ss_err.str().empty())) {
      if (err) {
        (*err) += ss_err.str();
      }
      return TINYEXR_ERROR_INVALID_HEADER;
    }
  }

  info->header_len = static_cast<unsigned int>(orig_size - size);

  return TINYEXR_SUCCESS;
}

// C++ HeaderInfo to C EXRHeader conversion.
static void ConvertHeader(EXRHeader *exr_header, const HeaderInfo &info) {
  exr_header->pixel_aspect_ratio = info.pixel_aspect_ratio;
  exr_header->screen_window_center[0] = info.screen_window_center[0];
  exr_header->screen_window_center[1] = info.screen_window_center[1];
  exr_header->screen_window_width = info.screen_window_width;
  exr_header->chunk_count = info.chunk_count;
  exr_header->display_window[0] = info.display_window[0];
  exr_header->display_window[1] = info.display_window[1];
  exr_header->display_window[2] = info.display_window[2];
  exr_header->display_window[3] = info.display_window[3];
  exr_header->data_window[0] = info.data_window[0];
  exr_header->data_window[1] = info.data_window[1];
  exr_header->data_window[2] = info.data_window[2];
  exr_header->data_window[3] = info.data_window[3];
  exr_header->line_order = info.line_order;
  exr_header->compression_type = info.compression_type;

  exr_header->tile_size_x = info.tile_size_x;
  exr_header->tile_size_y = info.tile_size_y;
  exr_header->tile_level_mode = info.tile_level_mode;
  exr_header->tile_rounding_mode = info.tile_rounding_mode;

  exr_header->num_channels = static_cast<int>(info.channels.size());

  exr_header->channels = static_cast<EXRChannelInfo *>(malloc(
      sizeof(EXRChannelInfo) * static_cast<size_t>(exr_header->num_channels)));
  for (size_t c = 0; c < static_cast<size_t>(exr_header->num_channels); c++) {
#ifdef _MSC_VER
    strncpy_s(exr_header->channels[c].name, info.channels[c].name.c_str(), 255);
#else
    strncpy(exr_header->channels[c].name, info.channels[c].name.c_str(), 255);
#endif
    // manually add '\0' for safety.
    exr_header->channels[c].name[255] = '\0';

    exr_header->channels[c].pixel_type = info.channels[c].pixel_type;
    exr_header->channels[c].p_linear = info.channels[c].p_linear;
    exr_header->channels[c].x_sampling = info.channels[c].x_sampling;
    exr_header->channels[c].y_sampling = info.channels[c].y_sampling;
  }

  exr_header->pixel_types = static_cast<int *>(
      malloc(sizeof(int) * static_cast<size_t>(exr_header->num_channels)));
  for (size_t c = 0; c < static_cast<size_t>(exr_header->num_channels); c++) {
    exr_header->pixel_types[c] = info.channels[c].pixel_type;
  }

  // Initially fill with values of `pixel_types`
  exr_header->requested_pixel_types = static_cast<int *>(
      malloc(sizeof(int) * static_cast<size_t>(exr_header->num_channels)));
  for (size_t c = 0; c < static_cast<size_t>(exr_header->num_channels); c++) {
    exr_header->requested_pixel_types[c] = info.channels[c].pixel_type;
  }

  exr_header->num_custom_attributes = static_cast<int>(info.attributes.size());

  if (exr_header->num_custom_attributes > 0) {
    // TODO(syoyo): Report warning when # of attributes exceeds
    // `TINYEXR_MAX_CUSTOM_ATTRIBUTES`
    if (exr_header->num_custom_attributes > TINYEXR_MAX_CUSTOM_ATTRIBUTES) {
      exr_header->num_custom_attributes = TINYEXR_MAX_CUSTOM_ATTRIBUTES;
    }

    exr_header->custom_attributes = static_cast<EXRAttribute *>(malloc(
        sizeof(EXRAttribute) * size_t(exr_header->num_custom_attributes)));

    for (size_t i = 0; i < info.attributes.size(); i++) {
      memcpy(exr_header->custom_attributes[i].name, info.attributes[i].name,
             256);
      memcpy(exr_header->custom_attributes[i].type, info.attributes[i].type,
             256);
      exr_header->custom_attributes[i].size = info.attributes[i].size;
      // Just copy poiner
      exr_header->custom_attributes[i].value = info.attributes[i].value;
    }

  } else {
    exr_header->custom_attributes = NULL;
  }

  exr_header->header_len = info.header_len;
}

static int DecodeChunk(EXRImage *exr_image, const EXRHeader *exr_header,
                       const std::vector<tinyexr::tinyexr_uint64> &offsets,
                       const unsigned char *head, const size_t size,
                       std::string *err) {
  int num_channels = exr_header->num_channels;

  int num_scanline_blocks = 1;
  if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_ZIP) {
    num_scanline_blocks = 16;
  } else if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_PIZ) {
    num_scanline_blocks = 32;
  } else if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_ZFP) {
    num_scanline_blocks = 16;
  }

  int data_width = exr_header->data_window[2] - exr_header->data_window[0] + 1;
  int data_height = exr_header->data_window[3] - exr_header->data_window[1] + 1;

  if ((data_width < 0) || (data_height < 0)) {
    if (err) {
      std::stringstream ss;
      ss << "Invalid data width or data height: " << data_width << ", "
         << data_height << std::endl;
      (*err) += ss.str();
    }
    return TINYEXR_ERROR_INVALID_DATA;
  }

  // Do not allow too large data_width and data_height. header invalid?
  {
    const int threshold = 1024 * 8192;  // heuristics
    if ((data_width > threshold) || (data_height > threshold)) {
      if (err) {
        std::stringstream ss;
        ss << "data_with or data_height too large. data_width: " << data_width
           << ", "
           << "data_height = " << data_height << std::endl;
        (*err) += ss.str();
      }
      return TINYEXR_ERROR_INVALID_DATA;
    }
  }

  size_t num_blocks = offsets.size();

  std::vector<size_t> channel_offset_list;
  int pixel_data_size = 0;
  size_t channel_offset = 0;
  if (!tinyexr::ComputeChannelLayout(&channel_offset_list, &pixel_data_size,
                                     &channel_offset, num_channels,
                                     exr_header->channels)) {
    if (err) {
      (*err) += "Failed to compute channel layout.\n";
    }
    return TINYEXR_ERROR_INVALID_DATA;
  }

  bool invalid_data = false;  // TODO(LTE): Use atomic lock for MT safety.

  if (exr_header->tiled) {
    // value check
    if (exr_header->tile_size_x < 0) {
      if (err) {
        std::stringstream ss;
        ss << "Invalid tile size x : " << exr_header->tile_size_x << "\n";
        (*err) += ss.str();
      }
      return TINYEXR_ERROR_INVALID_HEADER;
    }

    if (exr_header->tile_size_y < 0) {
      if (err) {
        std::stringstream ss;
        ss << "Invalid tile size y : " << exr_header->tile_size_y << "\n";
        (*err) += ss.str();
      }
      return TINYEXR_ERROR_INVALID_HEADER;
    }

    size_t num_tiles = offsets.size();  // = # of blocks

    exr_image->tiles = static_cast<EXRTile *>(
        calloc(sizeof(EXRTile), static_cast<size_t>(num_tiles)));

    for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      // Allocate memory for each tile.
      exr_image->tiles[tile_idx].images = tinyexr::AllocateImage(
          num_channels, exr_header->channels, exr_header->requested_pixel_types,
          exr_header->tile_size_x, exr_header->tile_size_y);

      // 16 byte: tile coordinates
      // 4 byte : data size
      // ~      : data(uncompressed or compressed)
      if (offsets[tile_idx] + sizeof(int) * 5 > size) {
        if (err) {
          (*err) += "Insufficient data size.\n";
        }
        return TINYEXR_ERROR_INVALID_DATA;
      }

      size_t data_size = size_t(size - (offsets[tile_idx] + sizeof(int) * 5));
      const unsigned char *data_ptr =
          reinterpret_cast<const unsigned char *>(head + offsets[tile_idx]);

      int tile_coordinates[4];
      memcpy(tile_coordinates, data_ptr, sizeof(int) * 4);
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&tile_coordinates[0]));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&tile_coordinates[1]));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&tile_coordinates[2]));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&tile_coordinates[3]));

      // @todo{ LoD }
      if (tile_coordinates[2] != 0) {
        return TINYEXR_ERROR_UNSUPPORTED_FEATURE;
      }
      if (tile_coordinates[3] != 0) {
        return TINYEXR_ERROR_UNSUPPORTED_FEATURE;
      }

      int data_len;
      memcpy(&data_len, data_ptr + 16,
             sizeof(int));  // 16 = sizeof(tile_coordinates)
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&data_len));

      if (data_len < 4 || size_t(data_len) > data_size) {
        if (err) {
          (*err) += "Insufficient data length.\n";
        }
        return TINYEXR_ERROR_INVALID_DATA;
      }

      // Move to data addr: 20 = 16 + 4;
      data_ptr += 20;

      tinyexr::DecodeTiledPixelData(
          exr_image->tiles[tile_idx].images,
          &(exr_image->tiles[tile_idx].width),
          &(exr_image->tiles[tile_idx].height),
          exr_header->requested_pixel_types, data_ptr,
          static_cast<size_t>(data_len), exr_header->compression_type,
          exr_header->line_order, data_width, data_height, tile_coordinates[0],
          tile_coordinates[1], exr_header->tile_size_x, exr_header->tile_size_y,
          static_cast<size_t>(pixel_data_size),
          static_cast<size_t>(exr_header->num_custom_attributes),
          exr_header->custom_attributes,
          static_cast<size_t>(exr_header->num_channels), exr_header->channels,
          channel_offset_list);

      exr_image->tiles[tile_idx].offset_x = tile_coordinates[0];
      exr_image->tiles[tile_idx].offset_y = tile_coordinates[1];
      exr_image->tiles[tile_idx].level_x = tile_coordinates[2];
      exr_image->tiles[tile_idx].level_y = tile_coordinates[3];

      exr_image->num_tiles = static_cast<int>(num_tiles);
    }
  } else {  // scanline format

    // Don't allow too large image(256GB * pixel_data_size or more). Workaround
    // for #104.
    size_t total_data_len =
        size_t(data_width) * size_t(data_height) * size_t(num_channels);
    const bool total_data_len_overflown = sizeof(void*) == 8 ? (total_data_len >= 0x4000000000) : false;
    if ((total_data_len == 0) || total_data_len_overflown ) {
      if (err) {
        std::stringstream ss;
        ss << "Image data size is zero or too large: width = " << data_width
           << ", height = " << data_height << ", channels = " << num_channels
           << std::endl;
        (*err) += ss.str();
      }
      return TINYEXR_ERROR_INVALID_DATA;
    }

    exr_image->images = tinyexr::AllocateImage(
        num_channels, exr_header->channels, exr_header->requested_pixel_types,
        data_width, data_height);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int y = 0; y < static_cast<int>(num_blocks); y++) {
      size_t y_idx = static_cast<size_t>(y);

      if (offsets[y_idx] + sizeof(int) * 2 > size) {
        invalid_data = true;
      } else {
        // 4 byte: scan line
        // 4 byte: data size
        // ~     : pixel data(uncompressed or compressed)
        size_t data_size = size_t(size - (offsets[y_idx] + sizeof(int) * 2));
        const unsigned char *data_ptr =
            reinterpret_cast<const unsigned char *>(head + offsets[y_idx]);

        int line_no;
        memcpy(&line_no, data_ptr, sizeof(int));
        int data_len;
        memcpy(&data_len, data_ptr + 4, sizeof(int));
        tinyexr::swap4(reinterpret_cast<unsigned int *>(&line_no));
        tinyexr::swap4(reinterpret_cast<unsigned int *>(&data_len));

        if (size_t(data_len) > data_size) {
          invalid_data = true;

        } else if ((line_no > (2 << 20)) || (line_no < -(2 << 20))) {
          // Too large value. Assume this is invalid
          // 2**20 = 1048576 = heuristic value.
          invalid_data = true;
        } else if (data_len == 0) {
          // TODO(syoyo): May be ok to raise the threshold for example `data_len
          // < 4`
          invalid_data = true;
        } else {
          // line_no may be negative.
          int end_line_no = (std::min)(line_no + num_scanline_blocks,
                                       (exr_header->data_window[3] + 1));

          int num_lines = end_line_no - line_no;

          if (num_lines <= 0) {
            invalid_data = true;
          } else {
            // Move to data addr: 8 = 4 + 4;
            data_ptr += 8;

            // Adjust line_no with data_window.bmin.y

            // overflow check
            tinyexr_int64 lno = static_cast<tinyexr_int64>(line_no) - static_cast<tinyexr_int64>(exr_header->data_window[1]);
            if (lno > std::numeric_limits<int>::max()) {
              line_no = -1; // invalid
            } else if (lno < -std::numeric_limits<int>::max()) {
              line_no = -1; // invalid
            } else {
              line_no -= exr_header->data_window[1];
            }

            if (line_no < 0) {
              invalid_data = true;
            } else {
              if (!tinyexr::DecodePixelData(
                      exr_image->images, exr_header->requested_pixel_types,
                      data_ptr, static_cast<size_t>(data_len),
                      exr_header->compression_type, exr_header->line_order,
                      data_width, data_height, data_width, y, line_no,
                      num_lines, static_cast<size_t>(pixel_data_size),
                      static_cast<size_t>(exr_header->num_custom_attributes),
                      exr_header->custom_attributes,
                      static_cast<size_t>(exr_header->num_channels),
                      exr_header->channels, channel_offset_list)) {
                invalid_data = true;
              }
            }
          }
        }
      }
    }  // omp parallel
  }

  if (invalid_data) {
    if (err) {
      std::stringstream ss;
      (*err) += "Invalid data found when decoding pixels.\n";
    }
    return TINYEXR_ERROR_INVALID_DATA;
  }

  // Overwrite `pixel_type` with `requested_pixel_type`.
  {
    for (int c = 0; c < exr_header->num_channels; c++) {
      exr_header->pixel_types[c] = exr_header->requested_pixel_types[c];
    }
  }

  {
    exr_image->num_channels = num_channels;

    exr_image->width = data_width;
    exr_image->height = data_height;
  }

  return TINYEXR_SUCCESS;
}

static bool ReconstructLineOffsets(
    std::vector<tinyexr::tinyexr_uint64> *offsets, size_t n,
    const unsigned char *head, const unsigned char *marker, const size_t size) {
  assert(head < marker);
  assert(offsets->size() == n);

  for (size_t i = 0; i < n; i++) {
    size_t offset = static_cast<size_t>(marker - head);
    // Offset should not exceed whole EXR file/data size.
    if ((offset + sizeof(tinyexr::tinyexr_uint64)) >= size) {
      return false;
    }

    int y;
    unsigned int data_len;

    memcpy(&y, marker, sizeof(int));
    memcpy(&data_len, marker + 4, sizeof(unsigned int));

    if (data_len >= size) {
      return false;
    }

    tinyexr::swap4(reinterpret_cast<unsigned int *>(&y));
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&data_len));

    (*offsets)[i] = offset;

    marker += data_len + 8;  // 8 = 4 bytes(y) + 4 bytes(data_len)
  }

  return true;
}

static int DecodeEXRImage(EXRImage *exr_image, const EXRHeader *exr_header,
                          const unsigned char *head,
                          const unsigned char *marker, const size_t size,
                          const char **err) {
  if (exr_image == NULL || exr_header == NULL || head == NULL ||
      marker == NULL || (size <= tinyexr::kEXRVersionSize)) {
    tinyexr::SetErrorMessage("Invalid argument for DecodeEXRImage().", err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

  int num_scanline_blocks = 1;
  if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_ZIP) {
    num_scanline_blocks = 16;
  } else if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_PIZ) {
    num_scanline_blocks = 32;
  } else if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_ZFP) {
    num_scanline_blocks = 16;
  }

  int data_width = exr_header->data_window[2] - exr_header->data_window[0];
  if (data_width >= std::numeric_limits<int>::max()) {
    // Issue 63
    tinyexr::SetErrorMessage("Invalid data width value", err);
    return TINYEXR_ERROR_INVALID_DATA;
  }
  data_width++;

  int data_height = exr_header->data_window[3] - exr_header->data_window[1];
  if (data_height >= std::numeric_limits<int>::max()) {
    tinyexr::SetErrorMessage("Invalid data height value", err);
    return TINYEXR_ERROR_INVALID_DATA;
  }
  data_height++;

  if ((data_width < 0) || (data_height < 0)) {
    tinyexr::SetErrorMessage("data width or data height is negative.", err);
    return TINYEXR_ERROR_INVALID_DATA;
  }

  // Do not allow too large data_width and data_height. header invalid?
  {
    const int threshold = 1024 * 8192;  // heuristics
    if (data_width > threshold) {
      tinyexr::SetErrorMessage("data width too large.", err);
      return TINYEXR_ERROR_INVALID_DATA;
    }
    if (data_height > threshold) {
      tinyexr::SetErrorMessage("data height too large.", err);
      return TINYEXR_ERROR_INVALID_DATA;
    }
  }

  // Read offset tables.
  size_t num_blocks = 0;

  if (exr_header->chunk_count > 0) {
    // Use `chunkCount` attribute.
    num_blocks = static_cast<size_t>(exr_header->chunk_count);
  } else if (exr_header->tiled) {
    // @todo { LoD }
    size_t num_x_tiles = static_cast<size_t>(data_width) /
                         static_cast<size_t>(exr_header->tile_size_x);
    if (num_x_tiles * static_cast<size_t>(exr_header->tile_size_x) <
        static_cast<size_t>(data_width)) {
      num_x_tiles++;
    }
    size_t num_y_tiles = static_cast<size_t>(data_height) /
                         static_cast<size_t>(exr_header->tile_size_y);
    if (num_y_tiles * static_cast<size_t>(exr_header->tile_size_y) <
        static_cast<size_t>(data_height)) {
      num_y_tiles++;
    }

    num_blocks = num_x_tiles * num_y_tiles;
  } else {
    num_blocks = static_cast<size_t>(data_height) /
                 static_cast<size_t>(num_scanline_blocks);
    if (num_blocks * static_cast<size_t>(num_scanline_blocks) <
        static_cast<size_t>(data_height)) {
      num_blocks++;
    }
  }

  std::vector<tinyexr::tinyexr_uint64> offsets(num_blocks);

  for (size_t y = 0; y < num_blocks; y++) {
    tinyexr::tinyexr_uint64 offset;
    // Issue #81
    if ((marker + sizeof(tinyexr_uint64)) >= (head + size)) {
      tinyexr::SetErrorMessage("Insufficient data size in offset table.", err);
      return TINYEXR_ERROR_INVALID_DATA;
    }

    memcpy(&offset, marker, sizeof(tinyexr::tinyexr_uint64));
    tinyexr::swap8(&offset);
    if (offset >= size) {
      tinyexr::SetErrorMessage("Invalid offset value in DecodeEXRImage.", err);
      return TINYEXR_ERROR_INVALID_DATA;
    }
    marker += sizeof(tinyexr::tinyexr_uint64);  // = 8
    offsets[y] = offset;
  }

  // If line offsets are invalid, we try to reconstruct it.
  // See OpenEXR/IlmImf/ImfScanLineInputFile.cpp::readLineOffsets() for details.
  for (size_t y = 0; y < num_blocks; y++) {
    if (offsets[y] <= 0) {
      // TODO(syoyo) Report as warning?
      // if (err) {
      //  stringstream ss;
      //  ss << "Incomplete lineOffsets." << std::endl;
      //  (*err) += ss.str();
      //}
      bool ret =
          ReconstructLineOffsets(&offsets, num_blocks, head, marker, size);
      if (ret) {
        // OK
        break;
      } else {
        tinyexr::SetErrorMessage(
            "Cannot reconstruct lineOffset table in DecodeEXRImage.", err);
        return TINYEXR_ERROR_INVALID_DATA;
      }
    }
  }

  {
    std::string e;
    int ret = DecodeChunk(exr_image, exr_header, offsets, head, size, &e);

    if (ret != TINYEXR_SUCCESS) {
      if (!e.empty()) {
        tinyexr::SetErrorMessage(e, err);
      }

      // release memory(if exists)
      if ((exr_header->num_channels > 0) && exr_image && exr_image->images) {
        for (size_t c = 0; c < size_t(exr_header->num_channels); c++) {
          if (exr_image->images[c]) {
            free(exr_image->images[c]);
            exr_image->images[c] = NULL;
          }
        }
        free(exr_image->images);
        exr_image->images = NULL;
      }
    }

    return ret;
  }
}

}  // namespace tinyexr

int LoadEXR(float **out_rgba, int *width, int *height, const char *filename,
            const char **err) {
  if (out_rgba == NULL) {
    tinyexr::SetErrorMessage("Invalid argument for LoadEXR()", err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

  EXRVersion exr_version;
  EXRImage exr_image;
  EXRHeader exr_header;
  InitEXRHeader(&exr_header);
  InitEXRImage(&exr_image);

  {
    int ret = ParseEXRVersionFromFile(&exr_version, filename);
    if (ret != TINYEXR_SUCCESS) {
      tinyexr::SetErrorMessage("Invalid EXR header.", err);
      return ret;
    }

    if (exr_version.multipart || exr_version.non_image) {
      tinyexr::SetErrorMessage(
          "Loading multipart or DeepImage is not supported  in LoadEXR() API",
          err);
      return TINYEXR_ERROR_INVALID_DATA;  // @fixme.
    }
  }

  {
    int ret = ParseEXRHeaderFromFile(&exr_header, &exr_version, filename, err);
    if (ret != TINYEXR_SUCCESS) {
      FreeEXRHeader(&exr_header);
      return ret;
    }
  }

  // Read HALF channel as FLOAT.
  for (int i = 0; i < exr_header.num_channels; i++) {
    if (exr_header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) {
      exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }
  }

  {
    int ret = LoadEXRImageFromFile(&exr_image, &exr_header, filename, err);
    if (ret != TINYEXR_SUCCESS) {
      FreeEXRHeader(&exr_header);
      return ret;
    }
  }

  // RGBA
  int idxR = -1;
  int idxG = -1;
  int idxB = -1;
  int idxA = -1;
  for (int c = 0; c < exr_header.num_channels; c++) {
    if (strcmp(exr_header.channels[c].name, "R") == 0) {
      idxR = c;
    } else if (strcmp(exr_header.channels[c].name, "G") == 0) {
      idxG = c;
    } else if (strcmp(exr_header.channels[c].name, "B") == 0) {
      idxB = c;
    } else if (strcmp(exr_header.channels[c].name, "A") == 0) {
      idxA = c;
    }
  }

  if (exr_header.num_channels == 1) {
    // Grayscale channel only.

    (*out_rgba) = reinterpret_cast<float *>(
        malloc(4 * sizeof(float) * static_cast<size_t>(exr_image.width) *
               static_cast<size_t>(exr_image.height)));

    if (exr_header.tiled) {
      for (int it = 0; it < exr_image.num_tiles; it++) {
        for (int j = 0; j < exr_header.tile_size_y; j++) {
          for (int i = 0; i < exr_header.tile_size_x; i++) {
            const int ii =
                exr_image.tiles[it].offset_x * exr_header.tile_size_x + i;
            const int jj =
                exr_image.tiles[it].offset_y * exr_header.tile_size_y + j;
            const int idx = ii + jj * exr_image.width;

            // out of region check.
            if (ii >= exr_image.width) {
              continue;
            }
            if (jj >= exr_image.height) {
              continue;
            }
            const int srcIdx = i + j * exr_header.tile_size_x;
            unsigned char **src = exr_image.tiles[it].images;
            (*out_rgba)[4 * idx + 0] =
                reinterpret_cast<float **>(src)[0][srcIdx];
            (*out_rgba)[4 * idx + 1] =
                reinterpret_cast<float **>(src)[0][srcIdx];
            (*out_rgba)[4 * idx + 2] =
                reinterpret_cast<float **>(src)[0][srcIdx];
            (*out_rgba)[4 * idx + 3] =
                reinterpret_cast<float **>(src)[0][srcIdx];
          }
        }
      }
    } else {
      for (int i = 0; i < exr_image.width * exr_image.height; i++) {
        const float val = reinterpret_cast<float **>(exr_image.images)[0][i];
        (*out_rgba)[4 * i + 0] = val;
        (*out_rgba)[4 * i + 1] = val;
        (*out_rgba)[4 * i + 2] = val;
        (*out_rgba)[4 * i + 3] = val;
      }
    }
  } else {
    // Assume RGB(A)

    if (idxR == -1) {
      tinyexr::SetErrorMessage("R channel not found", err);

      // @todo { free exr_image }
      FreeEXRHeader(&exr_header);
      return TINYEXR_ERROR_INVALID_DATA;
    }

    if (idxG == -1) {
      tinyexr::SetErrorMessage("G channel not found", err);
      // @todo { free exr_image }
      FreeEXRHeader(&exr_header);
      return TINYEXR_ERROR_INVALID_DATA;
    }

    if (idxB == -1) {
      tinyexr::SetErrorMessage("B channel not found", err);
      // @todo { free exr_image }
      FreeEXRHeader(&exr_header);
      return TINYEXR_ERROR_INVALID_DATA;
    }

    (*out_rgba) = reinterpret_cast<float *>(
        malloc(4 * sizeof(float) * static_cast<size_t>(exr_image.width) *
               static_cast<size_t>(exr_image.height)));
    if (exr_header.tiled) {
      for (int it = 0; it < exr_image.num_tiles; it++) {
        for (int j = 0; j < exr_header.tile_size_y; j++) {
          for (int i = 0; i < exr_header.tile_size_x; i++) {
            const int ii =
                exr_image.tiles[it].offset_x * exr_header.tile_size_x + i;
            const int jj =
                exr_image.tiles[it].offset_y * exr_header.tile_size_y + j;
            const int idx = ii + jj * exr_image.width;

            // out of region check.
            if (ii >= exr_image.width) {
              continue;
            }
            if (jj >= exr_image.height) {
              continue;
            }
            const int srcIdx = i + j * exr_header.tile_size_x;
            unsigned char **src = exr_image.tiles[it].images;
            (*out_rgba)[4 * idx + 0] =
                reinterpret_cast<float **>(src)[idxR][srcIdx];
            (*out_rgba)[4 * idx + 1] =
                reinterpret_cast<float **>(src)[idxG][srcIdx];
            (*out_rgba)[4 * idx + 2] =
                reinterpret_cast<float **>(src)[idxB][srcIdx];
            if (idxA != -1) {
              (*out_rgba)[4 * idx + 3] =
                  reinterpret_cast<float **>(src)[idxA][srcIdx];
            } else {
              (*out_rgba)[4 * idx + 3] = 1.0;
            }
          }
        }
      }
    } else {
      for (int i = 0; i < exr_image.width * exr_image.height; i++) {
        (*out_rgba)[4 * i + 0] =
            reinterpret_cast<float **>(exr_image.images)[idxR][i];
        (*out_rgba)[4 * i + 1] =
            reinterpret_cast<float **>(exr_image.images)[idxG][i];
        (*out_rgba)[4 * i + 2] =
            reinterpret_cast<float **>(exr_image.images)[idxB][i];
        if (idxA != -1) {
          (*out_rgba)[4 * i + 3] =
              reinterpret_cast<float **>(exr_image.images)[idxA][i];
        } else {
          (*out_rgba)[4 * i + 3] = 1.0;
        }
      }
    }
  }

  (*width) = exr_image.width;
  (*height) = exr_image.height;

  FreeEXRHeader(&exr_header);
  FreeEXRImage(&exr_image);

  return TINYEXR_SUCCESS;
}

int IsEXR(const char *filename) {
  EXRVersion exr_version;

  int ret = ParseEXRVersionFromFile(&exr_version, filename);
  if (ret != TINYEXR_SUCCESS) {
    return TINYEXR_ERROR_INVALID_HEADER;
  }

  return TINYEXR_SUCCESS;
}

int ParseEXRHeaderFromMemory(EXRHeader *exr_header, const EXRVersion *version,
                             const unsigned char *memory, size_t size,
                             const char **err) {
  if (memory == NULL || exr_header == NULL) {
    tinyexr::SetErrorMessage(
        "Invalid argument. `memory` or `exr_header` argument is null in "
        "ParseEXRHeaderFromMemory()",
        err);

    // Invalid argument
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

  if (size < tinyexr::kEXRVersionSize) {
    tinyexr::SetErrorMessage("Insufficient header/data size.\n", err);
    return TINYEXR_ERROR_INVALID_DATA;
  }

  const unsigned char *marker = memory + tinyexr::kEXRVersionSize;
  size_t marker_size = size - tinyexr::kEXRVersionSize;

  tinyexr::HeaderInfo info;
  info.clear();

  std::string err_str;
  int ret = ParseEXRHeader(&info, NULL, version, &err_str, marker, marker_size);

  if (ret != TINYEXR_SUCCESS) {
    if (err && !err_str.empty()) {
      tinyexr::SetErrorMessage(err_str, err);
    }
  }

  ConvertHeader(exr_header, info);

  // transfoer `tiled` from version.
  exr_header->tiled = version->tiled;

  return ret;
}

int LoadEXRFromMemory(float **out_rgba, int *width, int *height,
                      const unsigned char *memory, size_t size,
                      const char **err) {
  if (out_rgba == NULL || memory == NULL) {
    tinyexr::SetErrorMessage("Invalid argument for LoadEXRFromMemory", err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

  EXRVersion exr_version;
  EXRImage exr_image;
  EXRHeader exr_header;

  InitEXRHeader(&exr_header);

  int ret = ParseEXRVersionFromMemory(&exr_version, memory, size);
  if (ret != TINYEXR_SUCCESS) {
    tinyexr::SetErrorMessage("Failed to parse EXR version", err);
    return ret;
  }

  ret = ParseEXRHeaderFromMemory(&exr_header, &exr_version, memory, size, err);
  if (ret != TINYEXR_SUCCESS) {
    return ret;
  }

  // Read HALF channel as FLOAT.
  for (int i = 0; i < exr_header.num_channels; i++) {
    if (exr_header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) {
      exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }
  }

  InitEXRImage(&exr_image);
  ret = LoadEXRImageFromMemory(&exr_image, &exr_header, memory, size, err);
  if (ret != TINYEXR_SUCCESS) {
    return ret;
  }

  // RGBA
  int idxR = -1;
  int idxG = -1;
  int idxB = -1;
  int idxA = -1;
  for (int c = 0; c < exr_header.num_channels; c++) {
    if (strcmp(exr_header.channels[c].name, "R") == 0) {
      idxR = c;
    } else if (strcmp(exr_header.channels[c].name, "G") == 0) {
      idxG = c;
    } else if (strcmp(exr_header.channels[c].name, "B") == 0) {
      idxB = c;
    } else if (strcmp(exr_header.channels[c].name, "A") == 0) {
      idxA = c;
    }
  }

  // TODO(syoyo): Refactor removing same code as used in LoadEXR().
  if (exr_header.num_channels == 1) {
    // Grayscale channel only.

    (*out_rgba) = reinterpret_cast<float *>(
        malloc(4 * sizeof(float) * static_cast<size_t>(exr_image.width) *
               static_cast<size_t>(exr_image.height)));

    if (exr_header.tiled) {
      for (int it = 0; it < exr_image.num_tiles; it++) {
        for (int j = 0; j < exr_header.tile_size_y; j++) {
          for (int i = 0; i < exr_header.tile_size_x; i++) {
            const int ii =
                exr_image.tiles[it].offset_x * exr_header.tile_size_x + i;
            const int jj =
                exr_image.tiles[it].offset_y * exr_header.tile_size_y + j;
            const int idx = ii + jj * exr_image.width;

            // out of region check.
            if (ii >= exr_image.width) {
              continue;
            }
            if (jj >= exr_image.height) {
              continue;
            }
            const int srcIdx = i + j * exr_header.tile_size_x;
            unsigned char **src = exr_image.tiles[it].images;
            (*out_rgba)[4 * idx + 0] =
                reinterpret_cast<float **>(src)[0][srcIdx];
            (*out_rgba)[4 * idx + 1] =
                reinterpret_cast<float **>(src)[0][srcIdx];
            (*out_rgba)[4 * idx + 2] =
                reinterpret_cast<float **>(src)[0][srcIdx];
            (*out_rgba)[4 * idx + 3] =
                reinterpret_cast<float **>(src)[0][srcIdx];
          }
        }
      }
    } else {
      for (int i = 0; i < exr_image.width * exr_image.height; i++) {
        const float val = reinterpret_cast<float **>(exr_image.images)[0][i];
        (*out_rgba)[4 * i + 0] = val;
        (*out_rgba)[4 * i + 1] = val;
        (*out_rgba)[4 * i + 2] = val;
        (*out_rgba)[4 * i + 3] = val;
      }
    }

  } else {
    // TODO(syoyo): Support non RGBA image.

    if (idxR == -1) {
      tinyexr::SetErrorMessage("R channel not found", err);

      // @todo { free exr_image }
      return TINYEXR_ERROR_INVALID_DATA;
    }

    if (idxG == -1) {
      tinyexr::SetErrorMessage("G channel not found", err);
      // @todo { free exr_image }
      return TINYEXR_ERROR_INVALID_DATA;
    }

    if (idxB == -1) {
      tinyexr::SetErrorMessage("B channel not found", err);
      // @todo { free exr_image }
      return TINYEXR_ERROR_INVALID_DATA;
    }

    (*out_rgba) = reinterpret_cast<float *>(
        malloc(4 * sizeof(float) * static_cast<size_t>(exr_image.width) *
               static_cast<size_t>(exr_image.height)));

    if (exr_header.tiled) {
      for (int it = 0; it < exr_image.num_tiles; it++) {
        for (int j = 0; j < exr_header.tile_size_y; j++)
          for (int i = 0; i < exr_header.tile_size_x; i++) {
            const int ii =
                exr_image.tiles[it].offset_x * exr_header.tile_size_x + i;
            const int jj =
                exr_image.tiles[it].offset_y * exr_header.tile_size_y + j;
            const int idx = ii + jj * exr_image.width;

            // out of region check.
            if (ii >= exr_image.width) {
              continue;
            }
            if (jj >= exr_image.height) {
              continue;
            }
            const int srcIdx = i + j * exr_header.tile_size_x;
            unsigned char **src = exr_image.tiles[it].images;
            (*out_rgba)[4 * idx + 0] =
                reinterpret_cast<float **>(src)[idxR][srcIdx];
            (*out_rgba)[4 * idx + 1] =
                reinterpret_cast<float **>(src)[idxG][srcIdx];
            (*out_rgba)[4 * idx + 2] =
                reinterpret_cast<float **>(src)[idxB][srcIdx];
            if (idxA != -1) {
              (*out_rgba)[4 * idx + 3] =
                  reinterpret_cast<float **>(src)[idxA][srcIdx];
            } else {
              (*out_rgba)[4 * idx + 3] = 1.0;
            }
          }
      }
    } else {
      for (int i = 0; i < exr_image.width * exr_image.height; i++) {
        (*out_rgba)[4 * i + 0] =
            reinterpret_cast<float **>(exr_image.images)[idxR][i];
        (*out_rgba)[4 * i + 1] =
            reinterpret_cast<float **>(exr_image.images)[idxG][i];
        (*out_rgba)[4 * i + 2] =
            reinterpret_cast<float **>(exr_image.images)[idxB][i];
        if (idxA != -1) {
          (*out_rgba)[4 * i + 3] =
              reinterpret_cast<float **>(exr_image.images)[idxA][i];
        } else {
          (*out_rgba)[4 * i + 3] = 1.0;
        }
      }
    }
  }

  (*width) = exr_image.width;
  (*height) = exr_image.height;

  FreeEXRHeader(&exr_header);
  FreeEXRImage(&exr_image);

  return TINYEXR_SUCCESS;
}

int LoadEXRImageFromFile(EXRImage *exr_image, const EXRHeader *exr_header,
                         const char *filename, const char **err) {
  if (exr_image == NULL) {
    tinyexr::SetErrorMessage("Invalid argument for LoadEXRImageFromFile", err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

#ifdef _WIN32
  FILE *fp = NULL;
  fopen_s(&fp, filename, "rb");
#else
  FILE *fp = fopen(filename, "rb");
#endif
  if (!fp) {
    tinyexr::SetErrorMessage("Cannot read file " + std::string(filename), err);
    return TINYEXR_ERROR_CANT_OPEN_FILE;
  }

  size_t filesize;
  // Compute size
  fseek(fp, 0, SEEK_END);
  filesize = static_cast<size_t>(ftell(fp));
  fseek(fp, 0, SEEK_SET);

  if (filesize < 16) {
    tinyexr::SetErrorMessage("File size too short " + std::string(filename),
                             err);
    return TINYEXR_ERROR_INVALID_FILE;
  }

  std::vector<unsigned char> buf(filesize);  // @todo { use mmap }
  {
    size_t ret;
    ret = fread(&buf[0], 1, filesize, fp);
    assert(ret == filesize);
    fclose(fp);
    (void)ret;
  }

  return LoadEXRImageFromMemory(exr_image, exr_header, &buf.at(0), filesize,
                                err);
}

int LoadEXRImageFromMemory(EXRImage *exr_image, const EXRHeader *exr_header,
                           const unsigned char *memory, const size_t size,
                           const char **err) {
  if (exr_image == NULL || memory == NULL ||
      (size < tinyexr::kEXRVersionSize)) {
    tinyexr::SetErrorMessage("Invalid argument for LoadEXRImageFromMemory",
                             err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

  if (exr_header->header_len == 0) {
    tinyexr::SetErrorMessage("EXRHeader variable is not initialized.", err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

  const unsigned char *head = memory;
  const unsigned char *marker = reinterpret_cast<const unsigned char *>(
      memory + exr_header->header_len +
      8);  // +8 for magic number + version header.
  return tinyexr::DecodeEXRImage(exr_image, exr_header, head, marker, size,
                                 err);
}

size_t SaveEXRImageToMemory(const EXRImage *exr_image,
                            const EXRHeader *exr_header,
                            unsigned char **memory_out, const char **err) {
  if (exr_image == NULL || memory_out == NULL ||
      exr_header->compression_type < 0) {
    tinyexr::SetErrorMessage("Invalid argument for SaveEXRImageToMemory", err);
    return 0;
  }

#if !TINYEXR_USE_PIZ
  if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_PIZ) {
    tinyexr::SetErrorMessage("PIZ compression is not supported in this build",
                             err);
    return 0;
  }
#endif

#if !TINYEXR_USE_ZFP
  if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_ZFP) {
    tinyexr::SetErrorMessage("ZFP compression is not supported in this build",
                             err);
    return 0;
  }
#endif

#if TINYEXR_USE_ZFP
  for (size_t i = 0; i < static_cast<size_t>(exr_header->num_channels); i++) {
    if (exr_header->requested_pixel_types[i] != TINYEXR_PIXELTYPE_FLOAT) {
      tinyexr::SetErrorMessage("Pixel type must be FLOAT for ZFP compression",
                               err);
      return 0;
    }
  }
#endif

  std::vector<unsigned char> memory;

  // Header
  {
    const char header[] = {0x76, 0x2f, 0x31, 0x01};
    memory.insert(memory.end(), header, header + 4);
  }

  // Version, scanline.
  {
    char marker[] = {2, 0, 0, 0};
    /* @todo
    if (exr_header->tiled) {
      marker[1] |= 0x2;
    }
    if (exr_header->long_name) {
      marker[1] |= 0x4;
    }
    if (exr_header->non_image) {
      marker[1] |= 0x8;
    }
    if (exr_header->multipart) {
      marker[1] |= 0x10;
    }
    */
    memory.insert(memory.end(), marker, marker + 4);
  }

  int num_scanlines = 1;
  if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_ZIP) {
    num_scanlines = 16;
  } else if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_PIZ) {
    num_scanlines = 32;
  } else if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_ZFP) {
    num_scanlines = 16;
  }

  // Write attributes.
  std::vector<tinyexr::ChannelInfo> channels;
  {
    std::vector<unsigned char> data;

    for (int c = 0; c < exr_header->num_channels; c++) {
      tinyexr::ChannelInfo info;
      info.p_linear = 0;
      info.pixel_type = exr_header->requested_pixel_types[c];
      info.x_sampling = 1;
      info.y_sampling = 1;
      info.name = std::string(exr_header->channels[c].name);
      channels.push_back(info);
    }

    tinyexr::WriteChannelInfo(data, channels);

    tinyexr::WriteAttributeToMemory(&memory, "channels", "chlist", &data.at(0),
                                    static_cast<int>(data.size()));
  }

  {
    int comp = exr_header->compression_type;
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&comp));
    tinyexr::WriteAttributeToMemory(
        &memory, "compression", "compression",
        reinterpret_cast<const unsigned char *>(&comp), 1);
  }

  {
    int data[4] = {0, 0, exr_image->width - 1, exr_image->height - 1};
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&data[0]));
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&data[1]));
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&data[2]));
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&data[3]));
    tinyexr::WriteAttributeToMemory(
        &memory, "dataWindow", "box2i",
        reinterpret_cast<const unsigned char *>(data), sizeof(int) * 4);
    tinyexr::WriteAttributeToMemory(
        &memory, "displayWindow", "box2i",
        reinterpret_cast<const unsigned char *>(data), sizeof(int) * 4);
  }

  {
    unsigned char line_order = 0;  // @fixme { read line_order from EXRHeader }
    tinyexr::WriteAttributeToMemory(&memory, "lineOrder", "lineOrder",
                                    &line_order, 1);
  }

  {
    float aspectRatio = 1.0f;
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&aspectRatio));
    tinyexr::WriteAttributeToMemory(
        &memory, "pixelAspectRatio", "float",
        reinterpret_cast<const unsigned char *>(&aspectRatio), sizeof(float));
  }

  {
    float center[2] = {0.0f, 0.0f};
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&center[0]));
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&center[1]));
    tinyexr::WriteAttributeToMemory(
        &memory, "screenWindowCenter", "v2f",
        reinterpret_cast<const unsigned char *>(center), 2 * sizeof(float));
  }

  {
    float w = static_cast<float>(exr_image->width);
    tinyexr::swap4(reinterpret_cast<unsigned int *>(&w));
    tinyexr::WriteAttributeToMemory(&memory, "screenWindowWidth", "float",
                                    reinterpret_cast<const unsigned char *>(&w),
                                    sizeof(float));
  }

  // Custom attributes
  if (exr_header->num_custom_attributes > 0) {
    for (int i = 0; i < exr_header->num_custom_attributes; i++) {
      tinyexr::WriteAttributeToMemory(
          &memory, exr_header->custom_attributes[i].name,
          exr_header->custom_attributes[i].type,
          reinterpret_cast<const unsigned char *>(
              exr_header->custom_attributes[i].value),
          exr_header->custom_attributes[i].size);
    }
  }

  {  // end of header
    unsigned char e = 0;
    memory.push_back(e);
  }

  int num_blocks = exr_image->height / num_scanlines;
  if (num_blocks * num_scanlines < exr_image->height) {
    num_blocks++;
  }

  std::vector<tinyexr::tinyexr_uint64> offsets(static_cast<size_t>(num_blocks));

  size_t headerSize = memory.size();
  tinyexr::tinyexr_uint64 offset =
      headerSize +
      static_cast<size_t>(num_blocks) *
          sizeof(
              tinyexr::tinyexr_int64);  // sizeof(header) + sizeof(offsetTable)

  std::vector<std::vector<unsigned char> > data_list(
      static_cast<size_t>(num_blocks));
  std::vector<size_t> channel_offset_list(
      static_cast<size_t>(exr_header->num_channels));

  int pixel_data_size = 0;
  size_t channel_offset = 0;
  for (size_t c = 0; c < static_cast<size_t>(exr_header->num_channels); c++) {
    channel_offset_list[c] = channel_offset;
    if (exr_header->requested_pixel_types[c] == TINYEXR_PIXELTYPE_HALF) {
      pixel_data_size += sizeof(unsigned short);
      channel_offset += sizeof(unsigned short);
    } else if (exr_header->requested_pixel_types[c] ==
               TINYEXR_PIXELTYPE_FLOAT) {
      pixel_data_size += sizeof(float);
      channel_offset += sizeof(float);
    } else if (exr_header->requested_pixel_types[c] == TINYEXR_PIXELTYPE_UINT) {
      pixel_data_size += sizeof(unsigned int);
      channel_offset += sizeof(unsigned int);
    } else {
      assert(0);
    }
  }

#if TINYEXR_USE_ZFP
  tinyexr::ZFPCompressionParam zfp_compression_param;

  // Use ZFP compression parameter from custom attributes(if such a parameter
  // exists)
  {
    bool ret = tinyexr::FindZFPCompressionParam(
        &zfp_compression_param, exr_header->custom_attributes,
        exr_header->num_custom_attributes);

    if (!ret) {
      // Use predefined compression parameter.
      zfp_compression_param.type = 0;
      zfp_compression_param.rate = 2;
    }
  }
#endif

// Use signed int since some OpenMP compiler doesn't allow unsigned type for
// `parallel for`
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < num_blocks; i++) {
    size_t ii = static_cast<size_t>(i);
    int start_y = num_scanlines * i;
    int endY = (std::min)(num_scanlines * (i + 1), exr_image->height);
    int h = endY - start_y;

    std::vector<unsigned char> buf(
        static_cast<size_t>(exr_image->width * h * pixel_data_size));

    for (size_t c = 0; c < static_cast<size_t>(exr_header->num_channels); c++) {
      if (exr_header->pixel_types[c] == TINYEXR_PIXELTYPE_HALF) {
        if (exr_header->requested_pixel_types[c] == TINYEXR_PIXELTYPE_FLOAT) {
          for (int y = 0; y < h; y++) {
            // Assume increasing Y
            float *line_ptr = reinterpret_cast<float *>(&buf.at(
                static_cast<size_t>(pixel_data_size * y * exr_image->width) +
                channel_offset_list[c] *
                    static_cast<size_t>(exr_image->width)));
            for (int x = 0; x < exr_image->width; x++) {
              tinyexr::FP16 h16;
              h16.u = reinterpret_cast<unsigned short **>(
                  exr_image->images)[c][(y + start_y) * exr_image->width + x];

              tinyexr::FP32 f32 = half_to_float(h16);

              tinyexr::swap4(reinterpret_cast<unsigned int *>(&f32.f));

              // line_ptr[x] = f32.f;
              tinyexr::cpy4(line_ptr + x, &(f32.f));
            }
          }
        } else if (exr_header->requested_pixel_types[c] ==
                   TINYEXR_PIXELTYPE_HALF) {
          for (int y = 0; y < h; y++) {
            // Assume increasing Y
            unsigned short *line_ptr = reinterpret_cast<unsigned short *>(
                &buf.at(static_cast<size_t>(pixel_data_size * y *
                                            exr_image->width) +
                        channel_offset_list[c] *
                            static_cast<size_t>(exr_image->width)));
            for (int x = 0; x < exr_image->width; x++) {
              unsigned short val = reinterpret_cast<unsigned short **>(
                  exr_image->images)[c][(y + start_y) * exr_image->width + x];

              tinyexr::swap2(&val);

              // line_ptr[x] = val;
              tinyexr::cpy2(line_ptr + x, &val);
            }
          }
        } else {
          assert(0);
        }

      } else if (exr_header->pixel_types[c] == TINYEXR_PIXELTYPE_FLOAT) {
        if (exr_header->requested_pixel_types[c] == TINYEXR_PIXELTYPE_HALF) {
          for (int y = 0; y < h; y++) {
            // Assume increasing Y
            unsigned short *line_ptr = reinterpret_cast<unsigned short *>(
                &buf.at(static_cast<size_t>(pixel_data_size * y *
                                            exr_image->width) +
                        channel_offset_list[c] *
                            static_cast<size_t>(exr_image->width)));
            for (int x = 0; x < exr_image->width; x++) {
              tinyexr::FP32 f32;
              f32.f = reinterpret_cast<float **>(
                  exr_image->images)[c][(y + start_y) * exr_image->width + x];

              tinyexr::FP16 h16;
              h16 = float_to_half_full(f32);

              tinyexr::swap2(reinterpret_cast<unsigned short *>(&h16.u));

              // line_ptr[x] = h16.u;
              tinyexr::cpy2(line_ptr + x, &(h16.u));
            }
          }
        } else if (exr_header->requested_pixel_types[c] ==
                   TINYEXR_PIXELTYPE_FLOAT) {
          for (int y = 0; y < h; y++) {
            // Assume increasing Y
            float *line_ptr = reinterpret_cast<float *>(&buf.at(
                static_cast<size_t>(pixel_data_size * y * exr_image->width) +
                channel_offset_list[c] *
                    static_cast<size_t>(exr_image->width)));
            for (int x = 0; x < exr_image->width; x++) {
              float val = reinterpret_cast<float **>(
                  exr_image->images)[c][(y + start_y) * exr_image->width + x];

              tinyexr::swap4(reinterpret_cast<unsigned int *>(&val));

              // line_ptr[x] = val;
              tinyexr::cpy4(line_ptr + x, &val);
            }
          }
        } else {
          assert(0);
        }
      } else if (exr_header->pixel_types[c] == TINYEXR_PIXELTYPE_UINT) {
        for (int y = 0; y < h; y++) {
          // Assume increasing Y
          unsigned int *line_ptr = reinterpret_cast<unsigned int *>(&buf.at(
              static_cast<size_t>(pixel_data_size * y * exr_image->width) +
              channel_offset_list[c] * static_cast<size_t>(exr_image->width)));
          for (int x = 0; x < exr_image->width; x++) {
            unsigned int val = reinterpret_cast<unsigned int **>(
                exr_image->images)[c][(y + start_y) * exr_image->width + x];

            tinyexr::swap4(&val);

            // line_ptr[x] = val;
            tinyexr::cpy4(line_ptr + x, &val);
          }
        }
      }
    }

    if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_NONE) {
      // 4 byte: scan line
      // 4 byte: data size
      // ~     : pixel data(uncompressed)
      std::vector<unsigned char> header(8);
      unsigned int data_len = static_cast<unsigned int>(buf.size());
      memcpy(&header.at(0), &start_y, sizeof(int));
      memcpy(&header.at(4), &data_len, sizeof(unsigned int));

      tinyexr::swap4(reinterpret_cast<unsigned int *>(&header.at(0)));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&header.at(4)));

      data_list[ii].insert(data_list[ii].end(), header.begin(), header.end());
      data_list[ii].insert(data_list[ii].end(), buf.begin(),
                           buf.begin() + data_len);

    } else if ((exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_ZIPS) ||
               (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_ZIP)) {
#if TINYEXR_USE_MINIZ
      std::vector<unsigned char> block(tinyexr::miniz::mz_compressBound(
          static_cast<unsigned long>(buf.size())));
#else
      std::vector<unsigned char> block(
          compressBound(static_cast<uLong>(buf.size())));
#endif
      tinyexr::tinyexr_uint64 outSize = block.size();

      tinyexr::CompressZip(&block.at(0), outSize,
                           reinterpret_cast<const unsigned char *>(&buf.at(0)),
                           static_cast<unsigned long>(buf.size()));

      // 4 byte: scan line
      // 4 byte: data size
      // ~     : pixel data(compressed)
      std::vector<unsigned char> header(8);
      unsigned int data_len = static_cast<unsigned int>(outSize);  // truncate
      memcpy(&header.at(0), &start_y, sizeof(int));
      memcpy(&header.at(4), &data_len, sizeof(unsigned int));

      tinyexr::swap4(reinterpret_cast<unsigned int *>(&header.at(0)));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&header.at(4)));

      data_list[ii].insert(data_list[ii].end(), header.begin(), header.end());
      data_list[ii].insert(data_list[ii].end(), block.begin(),
                           block.begin() + data_len);

    } else if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_RLE) {
      // (buf.size() * 3) / 2 would be enough.
      std::vector<unsigned char> block((buf.size() * 3) / 2);

      tinyexr::tinyexr_uint64 outSize = block.size();

      tinyexr::CompressRle(&block.at(0), outSize,
                           reinterpret_cast<const unsigned char *>(&buf.at(0)),
                           static_cast<unsigned long>(buf.size()));

      // 4 byte: scan line
      // 4 byte: data size
      // ~     : pixel data(compressed)
      std::vector<unsigned char> header(8);
      unsigned int data_len = static_cast<unsigned int>(outSize);  // truncate
      memcpy(&header.at(0), &start_y, sizeof(int));
      memcpy(&header.at(4), &data_len, sizeof(unsigned int));

      tinyexr::swap4(reinterpret_cast<unsigned int *>(&header.at(0)));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&header.at(4)));

      data_list[ii].insert(data_list[ii].end(), header.begin(), header.end());
      data_list[ii].insert(data_list[ii].end(), block.begin(),
                           block.begin() + data_len);

    } else if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_PIZ) {
#if TINYEXR_USE_PIZ
      unsigned int bufLen =
          8192 + static_cast<unsigned int>(
                     2 * static_cast<unsigned int>(
                             buf.size()));  // @fixme { compute good bound. }
      std::vector<unsigned char> block(bufLen);
      unsigned int outSize = static_cast<unsigned int>(block.size());

      CompressPiz(&block.at(0), &outSize,
                  reinterpret_cast<const unsigned char *>(&buf.at(0)),
                  buf.size(), channels, exr_image->width, h);

      // 4 byte: scan line
      // 4 byte: data size
      // ~     : pixel data(compressed)
      std::vector<unsigned char> header(8);
      unsigned int data_len = outSize;
      memcpy(&header.at(0), &start_y, sizeof(int));
      memcpy(&header.at(4), &data_len, sizeof(unsigned int));

      tinyexr::swap4(reinterpret_cast<unsigned int *>(&header.at(0)));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&header.at(4)));

      data_list[ii].insert(data_list[ii].end(), header.begin(), header.end());
      data_list[ii].insert(data_list[ii].end(), block.begin(),
                           block.begin() + data_len);

#else
      assert(0);
#endif
    } else if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_ZFP) {
#if TINYEXR_USE_ZFP
      std::vector<unsigned char> block;
      unsigned int outSize;

      tinyexr::CompressZfp(
          &block, &outSize, reinterpret_cast<const float *>(&buf.at(0)),
          exr_image->width, h, exr_header->num_channels, zfp_compression_param);

      // 4 byte: scan line
      // 4 byte: data size
      // ~     : pixel data(compressed)
      std::vector<unsigned char> header(8);
      unsigned int data_len = outSize;
      memcpy(&header.at(0), &start_y, sizeof(int));
      memcpy(&header.at(4), &data_len, sizeof(unsigned int));

      tinyexr::swap4(reinterpret_cast<unsigned int *>(&header.at(0)));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&header.at(4)));

      data_list[ii].insert(data_list[ii].end(), header.begin(), header.end());
      data_list[ii].insert(data_list[ii].end(), block.begin(),
                           block.begin() + data_len);

#else
      assert(0);
#endif
    } else {
      assert(0);
    }
  }  // omp parallel

  for (size_t i = 0; i < static_cast<size_t>(num_blocks); i++) {
    offsets[i] = offset;
    tinyexr::swap8(reinterpret_cast<tinyexr::tinyexr_uint64 *>(&offsets[i]));
    offset += data_list[i].size();
  }

  size_t totalSize = static_cast<size_t>(offset);
  {
    memory.insert(
        memory.end(), reinterpret_cast<unsigned char *>(&offsets.at(0)),
        reinterpret_cast<unsigned char *>(&offsets.at(0)) +
            sizeof(tinyexr::tinyexr_uint64) * static_cast<size_t>(num_blocks));
  }

  if (memory.size() == 0) {
    tinyexr::SetErrorMessage("Output memory size is zero", err);
    return 0;
  }

  (*memory_out) = static_cast<unsigned char *>(malloc(totalSize));
  memcpy((*memory_out), &memory.at(0), memory.size());
  unsigned char *memory_ptr = *memory_out + memory.size();

  for (size_t i = 0; i < static_cast<size_t>(num_blocks); i++) {
    memcpy(memory_ptr, &data_list[i].at(0), data_list[i].size());
    memory_ptr += data_list[i].size();
  }

  return totalSize;  // OK
}

int SaveEXRImageToFile(const EXRImage *exr_image, const EXRHeader *exr_header,
                       const char *filename, const char **err) {
  if (exr_image == NULL || filename == NULL ||
      exr_header->compression_type < 0) {
    tinyexr::SetErrorMessage("Invalid argument for SaveEXRImageToFile", err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

#if !TINYEXR_USE_PIZ
  if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_PIZ) {
    tinyexr::SetErrorMessage("PIZ compression is not supported in this build",
                             err);
    return TINYEXR_ERROR_UNSUPPORTED_FEATURE;
  }
#endif

#if !TINYEXR_USE_ZFP
  if (exr_header->compression_type == TINYEXR_COMPRESSIONTYPE_ZFP) {
    tinyexr::SetErrorMessage("ZFP compression is not supported in this build",
                             err);
    return TINYEXR_ERROR_UNSUPPORTED_FEATURE;
  }
#endif

#ifdef _WIN32
  FILE *fp = NULL;
  fopen_s(&fp, filename, "wb");
#else
  FILE *fp = fopen(filename, "wb");
#endif
  if (!fp) {
    tinyexr::SetErrorMessage("Cannot write a file", err);
    return TINYEXR_ERROR_CANT_WRITE_FILE;
  }

  unsigned char *mem = NULL;
  size_t mem_size = SaveEXRImageToMemory(exr_image, exr_header, &mem, err);
  if (mem_size == 0) {
    return TINYEXR_ERROR_SERIALZATION_FAILED;
  }

  size_t written_size = 0;
  if ((mem_size > 0) && mem) {
    written_size = fwrite(mem, 1, mem_size, fp);
  }
  free(mem);

  fclose(fp);

  if (written_size != mem_size) {
    tinyexr::SetErrorMessage("Cannot write a file", err);
    return TINYEXR_ERROR_CANT_WRITE_FILE;
  }

  return TINYEXR_SUCCESS;
}

int LoadDeepEXR(DeepImage *deep_image, const char *filename, const char **err) {
  if (deep_image == NULL) {
    tinyexr::SetErrorMessage("Invalid argument for LoadDeepEXR", err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

#ifdef _MSC_VER
  FILE *fp = NULL;
  errno_t errcode = fopen_s(&fp, filename, "rb");
  if ((0 != errcode) || (!fp)) {
    tinyexr::SetErrorMessage("Cannot read a file " + std::string(filename),
                             err);
    return TINYEXR_ERROR_CANT_OPEN_FILE;
  }
#else
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    tinyexr::SetErrorMessage("Cannot read a file " + std::string(filename),
                             err);
    return TINYEXR_ERROR_CANT_OPEN_FILE;
  }
#endif

  size_t filesize;
  // Compute size
  fseek(fp, 0, SEEK_END);
  filesize = static_cast<size_t>(ftell(fp));
  fseek(fp, 0, SEEK_SET);

  if (filesize == 0) {
    fclose(fp);
    tinyexr::SetErrorMessage("File size is zero : " + std::string(filename),
                             err);
    return TINYEXR_ERROR_INVALID_FILE;
  }

  std::vector<char> buf(filesize);  // @todo { use mmap }
  {
    size_t ret;
    ret = fread(&buf[0], 1, filesize, fp);
    assert(ret == filesize);
    (void)ret;
  }
  fclose(fp);

  const char *head = &buf[0];
  const char *marker = &buf[0];

  // Header check.
  {
    const char header[] = {0x76, 0x2f, 0x31, 0x01};

    if (memcmp(marker, header, 4) != 0) {
      tinyexr::SetErrorMessage("Invalid magic number", err);
      return TINYEXR_ERROR_INVALID_MAGIC_NUMBER;
    }
    marker += 4;
  }

  // Version, scanline.
  {
    // ver 2.0, scanline, deep bit on(0x800)
    // must be [2, 0, 0, 0]
    if (marker[0] != 2 || marker[1] != 8 || marker[2] != 0 || marker[3] != 0) {
      tinyexr::SetErrorMessage("Unsupported version or scanline", err);
      return TINYEXR_ERROR_UNSUPPORTED_FORMAT;
    }

    marker += 4;
  }

  int dx = -1;
  int dy = -1;
  int dw = -1;
  int dh = -1;
  int num_scanline_blocks = 1;  // 16 for ZIP compression.
  int compression_type = -1;
  int num_channels = -1;
  std::vector<tinyexr::ChannelInfo> channels;

  // Read attributes
  size_t size = filesize - tinyexr::kEXRVersionSize;
  for (;;) {
    if (0 == size) {
      return TINYEXR_ERROR_INVALID_DATA;
    } else if (marker[0] == '\0') {
      marker++;
      size--;
      break;
    }

    std::string attr_name;
    std::string attr_type;
    std::vector<unsigned char> data;
    size_t marker_size;
    if (!tinyexr::ReadAttribute(&attr_name, &attr_type, &data, &marker_size,
                                marker, size)) {
      std::stringstream ss;
      ss << "Failed to parse attribute\n";
      tinyexr::SetErrorMessage(ss.str(), err);
      return TINYEXR_ERROR_INVALID_DATA;
    }
    marker += marker_size;
    size -= marker_size;

    if (attr_name.compare("compression") == 0) {
      compression_type = data[0];
      if (compression_type > TINYEXR_COMPRESSIONTYPE_PIZ) {
        std::stringstream ss;
        ss << "Unsupported compression type : " << compression_type;
        tinyexr::SetErrorMessage(ss.str(), err);
        return TINYEXR_ERROR_UNSUPPORTED_FORMAT;
      }

      if (compression_type == TINYEXR_COMPRESSIONTYPE_ZIP) {
        num_scanline_blocks = 16;
      }

    } else if (attr_name.compare("channels") == 0) {
      // name: zero-terminated string, from 1 to 255 bytes long
      // pixel type: int, possible values are: UINT = 0 HALF = 1 FLOAT = 2
      // pLinear: unsigned char, possible values are 0 and 1
      // reserved: three chars, should be zero
      // xSampling: int
      // ySampling: int

      if (!tinyexr::ReadChannelInfo(channels, data)) {
        tinyexr::SetErrorMessage("Failed to parse channel info", err);
        return TINYEXR_ERROR_INVALID_DATA;
      }

      num_channels = static_cast<int>(channels.size());

      if (num_channels < 1) {
        tinyexr::SetErrorMessage("Invalid channels format", err);
        return TINYEXR_ERROR_INVALID_DATA;
      }

    } else if (attr_name.compare("dataWindow") == 0) {
      memcpy(&dx, &data.at(0), sizeof(int));
      memcpy(&dy, &data.at(4), sizeof(int));
      memcpy(&dw, &data.at(8), sizeof(int));
      memcpy(&dh, &data.at(12), sizeof(int));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&dx));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&dy));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&dw));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&dh));

    } else if (attr_name.compare("displayWindow") == 0) {
      int x;
      int y;
      int w;
      int h;
      memcpy(&x, &data.at(0), sizeof(int));
      memcpy(&y, &data.at(4), sizeof(int));
      memcpy(&w, &data.at(8), sizeof(int));
      memcpy(&h, &data.at(12), sizeof(int));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&x));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&y));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&w));
      tinyexr::swap4(reinterpret_cast<unsigned int *>(&h));
    }
  }

  assert(dx >= 0);
  assert(dy >= 0);
  assert(dw >= 0);
  assert(dh >= 0);
  assert(num_channels >= 1);

  int data_width = dw - dx + 1;
  int data_height = dh - dy + 1;

  std::vector<float> image(
      static_cast<size_t>(data_width * data_height * 4));  // 4 = RGBA

  // Read offset tables.
  int num_blocks = data_height / num_scanline_blocks;
  if (num_blocks * num_scanline_blocks < data_height) {
    num_blocks++;
  }

  std::vector<tinyexr::tinyexr_int64> offsets(static_cast<size_t>(num_blocks));

  for (size_t y = 0; y < static_cast<size_t>(num_blocks); y++) {
    tinyexr::tinyexr_int64 offset;
    memcpy(&offset, marker, sizeof(tinyexr::tinyexr_int64));
    tinyexr::swap8(reinterpret_cast<tinyexr::tinyexr_uint64 *>(&offset));
    marker += sizeof(tinyexr::tinyexr_int64);  // = 8
    offsets[y] = offset;
  }

#if TINYEXR_USE_PIZ
  if ((compression_type == TINYEXR_COMPRESSIONTYPE_NONE) ||
      (compression_type == TINYEXR_COMPRESSIONTYPE_RLE) ||
      (compression_type == TINYEXR_COMPRESSIONTYPE_ZIPS) ||
      (compression_type == TINYEXR_COMPRESSIONTYPE_ZIP) ||
      (compression_type == TINYEXR_COMPRESSIONTYPE_PIZ)) {
#else
  if ((compression_type == TINYEXR_COMPRESSIONTYPE_NONE) ||
      (compression_type == TINYEXR_COMPRESSIONTYPE_RLE) ||
      (compression_type == TINYEXR_COMPRESSIONTYPE_ZIPS) ||
      (compression_type == TINYEXR_COMPRESSIONTYPE_ZIP)) {
#endif
    // OK
  } else {
    tinyexr::SetErrorMessage("Unsupported compression format", err);
    return TINYEXR_ERROR_UNSUPPORTED_FORMAT;
  }

  deep_image->image = static_cast<float ***>(
      malloc(sizeof(float **) * static_cast<size_t>(num_channels)));
  for (int c = 0; c < num_channels; c++) {
    deep_image->image[c] = static_cast<float **>(
        malloc(sizeof(float *) * static_cast<size_t>(data_height)));
    for (int y = 0; y < data_height; y++) {
    }
  }

  deep_image->offset_table = static_cast<int **>(
      malloc(sizeof(int *) * static_cast<size_t>(data_height)));
  for (int y = 0; y < data_height; y++) {
    deep_image->offset_table[y] = static_cast<int *>(
        malloc(sizeof(int) * static_cast<size_t>(data_width)));
  }

  for (size_t y = 0; y < static_cast<size_t>(num_blocks); y++) {
    const unsigned char *data_ptr =
        reinterpret_cast<const unsigned char *>(head + offsets[y]);

    // int: y coordinate
    // int64: packed size of pixel offset table
    // int64: packed size of sample data
    // int64: unpacked size of sample data
    // compressed pixel offset table
    // compressed sample data
    int line_no;
    tinyexr::tinyexr_int64 packedOffsetTableSize;
    tinyexr::tinyexr_int64 packedSampleDataSize;
    tinyexr::tinyexr_int64 unpackedSampleDataSize;
    memcpy(&line_no, data_ptr, sizeof(int));
    memcpy(&packedOffsetTableSize, data_ptr + 4,
           sizeof(tinyexr::tinyexr_int64));
    memcpy(&packedSampleDataSize, data_ptr + 12,
           sizeof(tinyexr::tinyexr_int64));
    memcpy(&unpackedSampleDataSize, data_ptr + 20,
           sizeof(tinyexr::tinyexr_int64));

    tinyexr::swap4(reinterpret_cast<unsigned int *>(&line_no));
    tinyexr::swap8(
        reinterpret_cast<tinyexr::tinyexr_uint64 *>(&packedOffsetTableSize));
    tinyexr::swap8(
        reinterpret_cast<tinyexr::tinyexr_uint64 *>(&packedSampleDataSize));
    tinyexr::swap8(
        reinterpret_cast<tinyexr::tinyexr_uint64 *>(&unpackedSampleDataSize));

    std::vector<int> pixelOffsetTable(static_cast<size_t>(data_width));

    // decode pixel offset table.
    {
      unsigned long dstLen =
          static_cast<unsigned long>(pixelOffsetTable.size() * sizeof(int));
      if (!tinyexr::DecompressZip(
              reinterpret_cast<unsigned char *>(&pixelOffsetTable.at(0)),
              &dstLen, data_ptr + 28,
              static_cast<unsigned long>(packedOffsetTableSize))) {
        return false;
      }

      assert(dstLen == pixelOffsetTable.size() * sizeof(int));
      for (size_t i = 0; i < static_cast<size_t>(data_width); i++) {
        deep_image->offset_table[y][i] = pixelOffsetTable[i];
      }
    }

    std::vector<unsigned char> sample_data(
        static_cast<size_t>(unpackedSampleDataSize));

    // decode sample data.
    {
      unsigned long dstLen = static_cast<unsigned long>(unpackedSampleDataSize);
      if (dstLen) {
        if (!tinyexr::DecompressZip(
                reinterpret_cast<unsigned char *>(&sample_data.at(0)), &dstLen,
                data_ptr + 28 + packedOffsetTableSize,
                static_cast<unsigned long>(packedSampleDataSize))) {
          return false;
        }
        assert(dstLen == static_cast<unsigned long>(unpackedSampleDataSize));
      }
    }

    // decode sample
    int sampleSize = -1;
    std::vector<int> channel_offset_list(static_cast<size_t>(num_channels));
    {
      int channel_offset = 0;
      for (size_t i = 0; i < static_cast<size_t>(num_channels); i++) {
        channel_offset_list[i] = channel_offset;
        if (channels[i].pixel_type == TINYEXR_PIXELTYPE_UINT) {  // UINT
          channel_offset += 4;
        } else if (channels[i].pixel_type == TINYEXR_PIXELTYPE_HALF) {  // half
          channel_offset += 2;
        } else if (channels[i].pixel_type ==
                   TINYEXR_PIXELTYPE_FLOAT) {  // float
          channel_offset += 4;
        } else {
          assert(0);
        }
      }
      sampleSize = channel_offset;
    }
    assert(sampleSize >= 2);

    assert(static_cast<size_t>(
               pixelOffsetTable[static_cast<size_t>(data_width - 1)] *
               sampleSize) == sample_data.size());
    int samples_per_line = static_cast<int>(sample_data.size()) / sampleSize;

    //
    // Alloc memory
    //

    //
    // pixel data is stored as image[channels][pixel_samples]
    //
    {
      tinyexr::tinyexr_uint64 data_offset = 0;
      for (size_t c = 0; c < static_cast<size_t>(num_channels); c++) {
        deep_image->image[c][y] = static_cast<float *>(
            malloc(sizeof(float) * static_cast<size_t>(samples_per_line)));

        if (channels[c].pixel_type == 0) {  // UINT
          for (size_t x = 0; x < static_cast<size_t>(samples_per_line); x++) {
            unsigned int ui;
            unsigned int *src_ptr = reinterpret_cast<unsigned int *>(
                &sample_data.at(size_t(data_offset) + x * sizeof(int)));
            tinyexr::cpy4(&ui, src_ptr);
            deep_image->image[c][y][x] = static_cast<float>(ui);  // @fixme
          }
          data_offset +=
              sizeof(unsigned int) * static_cast<size_t>(samples_per_line);
        } else if (channels[c].pixel_type == 1) {  // half
          for (size_t x = 0; x < static_cast<size_t>(samples_per_line); x++) {
            tinyexr::FP16 f16;
            const unsigned short *src_ptr = reinterpret_cast<unsigned short *>(
                &sample_data.at(size_t(data_offset) + x * sizeof(short)));
            tinyexr::cpy2(&(f16.u), src_ptr);
            tinyexr::FP32 f32 = half_to_float(f16);
            deep_image->image[c][y][x] = f32.f;
          }
          data_offset += sizeof(short) * static_cast<size_t>(samples_per_line);
        } else {  // float
          for (size_t x = 0; x < static_cast<size_t>(samples_per_line); x++) {
            float f;
            const float *src_ptr = reinterpret_cast<float *>(
                &sample_data.at(size_t(data_offset) + x * sizeof(float)));
            tinyexr::cpy4(&f, src_ptr);
            deep_image->image[c][y][x] = f;
          }
          data_offset += sizeof(float) * static_cast<size_t>(samples_per_line);
        }
      }
    }
  }  // y

  deep_image->width = data_width;
  deep_image->height = data_height;

  deep_image->channel_names = static_cast<const char **>(
      malloc(sizeof(const char *) * static_cast<size_t>(num_channels)));
  for (size_t c = 0; c < static_cast<size_t>(num_channels); c++) {
#ifdef _WIN32
    deep_image->channel_names[c] = _strdup(channels[c].name.c_str());
#else
    deep_image->channel_names[c] = strdup(channels[c].name.c_str());
#endif
  }
  deep_image->num_channels = num_channels;

  return TINYEXR_SUCCESS;
}

void InitEXRImage(EXRImage *exr_image) {
  if (exr_image == NULL) {
    return;
  }

  exr_image->width = 0;
  exr_image->height = 0;
  exr_image->num_channels = 0;

  exr_image->images = NULL;
  exr_image->tiles = NULL;

  exr_image->num_tiles = 0;
}

void FreeEXRErrorMessage(const char *msg) {
  if (msg) {
    free(reinterpret_cast<void *>(const_cast<char *>(msg)));
  }
  return;
}

void InitEXRHeader(EXRHeader *exr_header) {
  if (exr_header == NULL) {
    return;
  }

  memset(exr_header, 0, sizeof(EXRHeader));
}

int FreeEXRHeader(EXRHeader *exr_header) {
  if (exr_header == NULL) {
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

  if (exr_header->channels) {
    free(exr_header->channels);
  }

  if (exr_header->pixel_types) {
    free(exr_header->pixel_types);
  }

  if (exr_header->requested_pixel_types) {
    free(exr_header->requested_pixel_types);
  }

  for (int i = 0; i < exr_header->num_custom_attributes; i++) {
    if (exr_header->custom_attributes[i].value) {
      free(exr_header->custom_attributes[i].value);
    }
  }

  if (exr_header->custom_attributes) {
    free(exr_header->custom_attributes);
  }

  return TINYEXR_SUCCESS;
}

int FreeEXRImage(EXRImage *exr_image) {
  if (exr_image == NULL) {
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

  for (int i = 0; i < exr_image->num_channels; i++) {
    if (exr_image->images && exr_image->images[i]) {
      free(exr_image->images[i]);
    }
  }

  if (exr_image->images) {
    free(exr_image->images);
  }

  if (exr_image->tiles) {
    for (int tid = 0; tid < exr_image->num_tiles; tid++) {
      for (int i = 0; i < exr_image->num_channels; i++) {
        if (exr_image->tiles[tid].images && exr_image->tiles[tid].images[i]) {
          free(exr_image->tiles[tid].images[i]);
        }
      }
      if (exr_image->tiles[tid].images) {
        free(exr_image->tiles[tid].images);
      }
    }
    free(exr_image->tiles);
  }

  return TINYEXR_SUCCESS;
}

int ParseEXRHeaderFromFile(EXRHeader *exr_header, const EXRVersion *exr_version,
                           const char *filename, const char **err) {
  if (exr_header == NULL || exr_version == NULL || filename == NULL) {
    tinyexr::SetErrorMessage("Invalid argument for ParseEXRHeaderFromFile",
                             err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

#ifdef _WIN32
  FILE *fp = NULL;
  fopen_s(&fp, filename, "rb");
#else
  FILE *fp = fopen(filename, "rb");
#endif
  if (!fp) {
    tinyexr::SetErrorMessage("Cannot read file " + std::string(filename), err);
    return TINYEXR_ERROR_CANT_OPEN_FILE;
  }

  size_t filesize;
  // Compute size
  fseek(fp, 0, SEEK_END);
  filesize = static_cast<size_t>(ftell(fp));
  fseek(fp, 0, SEEK_SET);

  std::vector<unsigned char> buf(filesize);  // @todo { use mmap }
  {
    size_t ret;
    ret = fread(&buf[0], 1, filesize, fp);
    assert(ret == filesize);
    fclose(fp);

    if (ret != filesize) {
      tinyexr::SetErrorMessage("fread() error on " + std::string(filename),
                               err);
      return TINYEXR_ERROR_INVALID_FILE;
    }
  }

  return ParseEXRHeaderFromMemory(exr_header, exr_version, &buf.at(0), filesize,
                                  err);
}

int ParseEXRMultipartHeaderFromMemory(EXRHeader ***exr_headers,
                                      int *num_headers,
                                      const EXRVersion *exr_version,
                                      const unsigned char *memory, size_t size,
                                      const char **err) {
  if (memory == NULL || exr_headers == NULL || num_headers == NULL ||
      exr_version == NULL) {
    // Invalid argument
    tinyexr::SetErrorMessage(
        "Invalid argument for ParseEXRMultipartHeaderFromMemory", err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

  if (size < tinyexr::kEXRVersionSize) {
    tinyexr::SetErrorMessage("Data size too short", err);
    return TINYEXR_ERROR_INVALID_DATA;
  }

  const unsigned char *marker = memory + tinyexr::kEXRVersionSize;
  size_t marker_size = size - tinyexr::kEXRVersionSize;

  std::vector<tinyexr::HeaderInfo> infos;

  for (;;) {
    tinyexr::HeaderInfo info;
    info.clear();

    std::string err_str;
    bool empty_header = false;
    int ret = ParseEXRHeader(&info, &empty_header, exr_version, &err_str,
                             marker, marker_size);

    if (ret != TINYEXR_SUCCESS) {
      tinyexr::SetErrorMessage(err_str, err);
      return ret;
    }

    if (empty_header) {
      marker += 1;  // skip '\0'
      break;
    }

    // `chunkCount` must exist in the header.
    if (info.chunk_count == 0) {
      tinyexr::SetErrorMessage(
          "`chunkCount' attribute is not found in the header.", err);
      return TINYEXR_ERROR_INVALID_DATA;
    }

    infos.push_back(info);

    // move to next header.
    marker += info.header_len;
    size -= info.header_len;
  }

  // allocate memory for EXRHeader and create array of EXRHeader pointers.
  (*exr_headers) =
      static_cast<EXRHeader **>(malloc(sizeof(EXRHeader *) * infos.size()));
  for (size_t i = 0; i < infos.size(); i++) {
    EXRHeader *exr_header = static_cast<EXRHeader *>(malloc(sizeof(EXRHeader)));

    ConvertHeader(exr_header, infos[i]);

    // transfoer `tiled` from version.
    exr_header->tiled = exr_version->tiled;

    (*exr_headers)[i] = exr_header;
  }

  (*num_headers) = static_cast<int>(infos.size());

  return TINYEXR_SUCCESS;
}

int ParseEXRMultipartHeaderFromFile(EXRHeader ***exr_headers, int *num_headers,
                                    const EXRVersion *exr_version,
                                    const char *filename, const char **err) {
  if (exr_headers == NULL || num_headers == NULL || exr_version == NULL ||
      filename == NULL) {
    tinyexr::SetErrorMessage(
        "Invalid argument for ParseEXRMultipartHeaderFromFile()", err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

#ifdef _WIN32
  FILE *fp = NULL;
  fopen_s(&fp, filename, "rb");
#else
  FILE *fp = fopen(filename, "rb");
#endif
  if (!fp) {
    tinyexr::SetErrorMessage("Cannot read file " + std::string(filename), err);
    return TINYEXR_ERROR_CANT_OPEN_FILE;
  }

  size_t filesize;
  // Compute size
  fseek(fp, 0, SEEK_END);
  filesize = static_cast<size_t>(ftell(fp));
  fseek(fp, 0, SEEK_SET);

  std::vector<unsigned char> buf(filesize);  // @todo { use mmap }
  {
    size_t ret;
    ret = fread(&buf[0], 1, filesize, fp);
    assert(ret == filesize);
    fclose(fp);

    if (ret != filesize) {
      tinyexr::SetErrorMessage("`fread' error. file may be corrupted.", err);
      return TINYEXR_ERROR_INVALID_FILE;
    }
  }

  return ParseEXRMultipartHeaderFromMemory(
      exr_headers, num_headers, exr_version, &buf.at(0), filesize, err);
}

int ParseEXRVersionFromMemory(EXRVersion *version, const unsigned char *memory,
                              size_t size) {
  if (version == NULL || memory == NULL) {
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

  if (size < tinyexr::kEXRVersionSize) {
    return TINYEXR_ERROR_INVALID_DATA;
  }

  const unsigned char *marker = memory;

  // Header check.
  {
    const char header[] = {0x76, 0x2f, 0x31, 0x01};

    if (memcmp(marker, header, 4) != 0) {
      return TINYEXR_ERROR_INVALID_MAGIC_NUMBER;
    }
    marker += 4;
  }

  version->tiled = false;
  version->long_name = false;
  version->non_image = false;
  version->multipart = false;

  // Parse version header.
  {
    // must be 2
    if (marker[0] != 2) {
      return TINYEXR_ERROR_INVALID_EXR_VERSION;
    }

    if (version == NULL) {
      return TINYEXR_SUCCESS;  // May OK
    }

    version->version = 2;

    if (marker[1] & 0x2) {  // 9th bit
      version->tiled = true;
    }
    if (marker[1] & 0x4) {  // 10th bit
      version->long_name = true;
    }
    if (marker[1] & 0x8) {        // 11th bit
      version->non_image = true;  // (deep image)
    }
    if (marker[1] & 0x10) {  // 12th bit
      version->multipart = true;
    }
  }

  return TINYEXR_SUCCESS;
}

int ParseEXRVersionFromFile(EXRVersion *version, const char *filename) {
  if (filename == NULL) {
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

#ifdef _WIN32
  FILE *fp = NULL;
  fopen_s(&fp, filename, "rb");
#else
  FILE *fp = fopen(filename, "rb");
#endif
  if (!fp) {
    return TINYEXR_ERROR_CANT_OPEN_FILE;
  }

  size_t file_size;
  // Compute size
  fseek(fp, 0, SEEK_END);
  file_size = static_cast<size_t>(ftell(fp));
  fseek(fp, 0, SEEK_SET);

  if (file_size < tinyexr::kEXRVersionSize) {
    return TINYEXR_ERROR_INVALID_FILE;
  }

  unsigned char buf[tinyexr::kEXRVersionSize];
  size_t ret = fread(&buf[0], 1, tinyexr::kEXRVersionSize, fp);
  fclose(fp);

  if (ret != tinyexr::kEXRVersionSize) {
    return TINYEXR_ERROR_INVALID_FILE;
  }

  return ParseEXRVersionFromMemory(version, buf, tinyexr::kEXRVersionSize);
}

int LoadEXRMultipartImageFromMemory(EXRImage *exr_images,
                                    const EXRHeader **exr_headers,
                                    unsigned int num_parts,
                                    const unsigned char *memory,
                                    const size_t size, const char **err) {
  if (exr_images == NULL || exr_headers == NULL || num_parts == 0 ||
      memory == NULL || (size <= tinyexr::kEXRVersionSize)) {
    tinyexr::SetErrorMessage(
        "Invalid argument for LoadEXRMultipartImageFromMemory()", err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

  // compute total header size.
  size_t total_header_size = 0;
  for (unsigned int i = 0; i < num_parts; i++) {
    if (exr_headers[i]->header_len == 0) {
      tinyexr::SetErrorMessage("EXRHeader variable is not initialized.", err);
      return TINYEXR_ERROR_INVALID_ARGUMENT;
    }

    total_header_size += exr_headers[i]->header_len;
  }

  const char *marker = reinterpret_cast<const char *>(
      memory + total_header_size + 4 +
      4);  // +8 for magic number and version header.

  marker += 1;  // Skip empty header.

  // NOTE 1:
  //   In multipart image, There is 'part number' before chunk data.
  //   4 byte : part number
  //   4+     : chunk
  //
  // NOTE 2:
  //   EXR spec says 'part number' is 'unsigned long' but actually this is
  //   'unsigned int(4 bytes)' in OpenEXR implementation...
  //   http://www.openexr.com/openexrfilelayout.pdf

  // Load chunk offset table.
  std::vector<std::vector<tinyexr::tinyexr_uint64> > chunk_offset_table_list;
  for (size_t i = 0; i < static_cast<size_t>(num_parts); i++) {
    std::vector<tinyexr::tinyexr_uint64> offset_table(
        static_cast<size_t>(exr_headers[i]->chunk_count));

    for (size_t c = 0; c < offset_table.size(); c++) {
      tinyexr::tinyexr_uint64 offset;
      memcpy(&offset, marker, 8);
      tinyexr::swap8(&offset);

      if (offset >= size) {
        tinyexr::SetErrorMessage("Invalid offset size in EXR header chunks.",
                                 err);
        return TINYEXR_ERROR_INVALID_DATA;
      }

      offset_table[c] = offset + 4;  // +4 to skip 'part number'
      marker += 8;
    }

    chunk_offset_table_list.push_back(offset_table);
  }

  // Decode image.
  for (size_t i = 0; i < static_cast<size_t>(num_parts); i++) {
    std::vector<tinyexr::tinyexr_uint64> &offset_table =
        chunk_offset_table_list[i];

    // First check 'part number' is identitical to 'i'
    for (size_t c = 0; c < offset_table.size(); c++) {
      const unsigned char *part_number_addr =
          memory + offset_table[c] - 4;  // -4 to move to 'part number' field.
      unsigned int part_no;
      memcpy(&part_no, part_number_addr, sizeof(unsigned int));  // 4
      tinyexr::swap4(&part_no);

      if (part_no != i) {
        tinyexr::SetErrorMessage("Invalid `part number' in EXR header chunks.",
                                 err);
        return TINYEXR_ERROR_INVALID_DATA;
      }
    }

    std::string e;
    int ret = tinyexr::DecodeChunk(&exr_images[i], exr_headers[i], offset_table,
                                   memory, size, &e);
    if (ret != TINYEXR_SUCCESS) {
      if (!e.empty()) {
        tinyexr::SetErrorMessage(e, err);
      }
      return ret;
    }
  }

  return TINYEXR_SUCCESS;
}

int LoadEXRMultipartImageFromFile(EXRImage *exr_images,
                                  const EXRHeader **exr_headers,
                                  unsigned int num_parts, const char *filename,
                                  const char **err) {
  if (exr_images == NULL || exr_headers == NULL || num_parts == 0) {
    tinyexr::SetErrorMessage(
        "Invalid argument for LoadEXRMultipartImageFromFile", err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

#ifdef _WIN32
  FILE *fp = NULL;
  fopen_s(&fp, filename, "rb");
#else
  FILE *fp = fopen(filename, "rb");
#endif
  if (!fp) {
    tinyexr::SetErrorMessage("Cannot read file " + std::string(filename), err);
    return TINYEXR_ERROR_CANT_OPEN_FILE;
  }

  size_t filesize;
  // Compute size
  fseek(fp, 0, SEEK_END);
  filesize = static_cast<size_t>(ftell(fp));
  fseek(fp, 0, SEEK_SET);

  std::vector<unsigned char> buf(filesize);  //  @todo { use mmap }
  {
    size_t ret;
    ret = fread(&buf[0], 1, filesize, fp);
    assert(ret == filesize);
    fclose(fp);
    (void)ret;
  }

  return LoadEXRMultipartImageFromMemory(exr_images, exr_headers, num_parts,
                                         &buf.at(0), filesize, err);
}

int SaveEXR(const float *data, int width, int height, int components,
            const int save_as_fp16, const char *outfilename, const char **err) {
  if ((components == 1) || components == 3 || components == 4) {
    // OK
  } else {
    std::stringstream ss;
    ss << "Unsupported component value : " << components << std::endl;

    tinyexr::SetErrorMessage(ss.str(), err);
    return TINYEXR_ERROR_INVALID_ARGUMENT;
  }

  EXRHeader header;
  InitEXRHeader(&header);

  if ((width < 16) && (height < 16)) {
    // No compression for small image.
    header.compression_type = TINYEXR_COMPRESSIONTYPE_NONE;
  } else {
    header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;
  }

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = components;

  std::vector<float> images[4];

  if (components == 1) {
    images[0].resize(static_cast<size_t>(width * height));
    memcpy(images[0].data(), data, sizeof(float) * size_t(width * height));
  } else {
    images[0].resize(static_cast<size_t>(width * height));
    images[1].resize(static_cast<size_t>(width * height));
    images[2].resize(static_cast<size_t>(width * height));
    images[3].resize(static_cast<size_t>(width * height));

    // Split RGB(A)RGB(A)RGB(A)... into R, G and B(and A) layers
    for (size_t i = 0; i < static_cast<size_t>(width * height); i++) {
      images[0][i] = data[static_cast<size_t>(components) * i + 0];
      images[1][i] = data[static_cast<size_t>(components) * i + 1];
      images[2][i] = data[static_cast<size_t>(components) * i + 2];
      if (components == 4) {
        images[3][i] = data[static_cast<size_t>(components) * i + 3];
      }
    }
  }

  float *image_ptr[4] = {0, 0, 0, 0};
  if (components == 4) {
    image_ptr[0] = &(images[3].at(0));  // A
    image_ptr[1] = &(images[2].at(0));  // B
    image_ptr[2] = &(images[1].at(0));  // G
    image_ptr[3] = &(images[0].at(0));  // R
  } else if (components == 3) {
    image_ptr[0] = &(images[2].at(0));  // B
    image_ptr[1] = &(images[1].at(0));  // G
    image_ptr[2] = &(images[0].at(0));  // R
  } else if (components == 1) {
    image_ptr[0] = &(images[0].at(0));  // A
  }

  image.images = reinterpret_cast<unsigned char **>(image_ptr);
  image.width = width;
  image.height = height;

  header.num_channels = components;
  header.channels = static_cast<EXRChannelInfo *>(malloc(
      sizeof(EXRChannelInfo) * static_cast<size_t>(header.num_channels)));
  // Must be (A)BGR order, since most of EXR viewers expect this channel order.
  if (components == 4) {
#ifdef _MSC_VER
    strncpy_s(header.channels[0].name, "A", 255);
    strncpy_s(header.channels[1].name, "B", 255);
    strncpy_s(header.channels[2].name, "G", 255);
    strncpy_s(header.channels[3].name, "R", 255);
#else
    strncpy(header.channels[0].name, "A", 255);
    strncpy(header.channels[1].name, "B", 255);
    strncpy(header.channels[2].name, "G", 255);
    strncpy(header.channels[3].name, "R", 255);
#endif
    header.channels[0].name[strlen("A")] = '\0';
    header.channels[1].name[strlen("B")] = '\0';
    header.channels[2].name[strlen("G")] = '\0';
    header.channels[3].name[strlen("R")] = '\0';
  } else if (components == 3) {
#ifdef _MSC_VER
    strncpy_s(header.channels[0].name, "B", 255);
    strncpy_s(header.channels[1].name, "G", 255);
    strncpy_s(header.channels[2].name, "R", 255);
#else
    strncpy(header.channels[0].name, "B", 255);
    strncpy(header.channels[1].name, "G", 255);
    strncpy(header.channels[2].name, "R", 255);
#endif
    header.channels[0].name[strlen("B")] = '\0';
    header.channels[1].name[strlen("G")] = '\0';
    header.channels[2].name[strlen("R")] = '\0';
  } else {
#ifdef _MSC_VER
    strncpy_s(header.channels[0].name, "A", 255);
#else
    strncpy(header.channels[0].name, "A", 255);
#endif
    header.channels[0].name[strlen("A")] = '\0';
  }

  header.pixel_types = static_cast<int *>(
      malloc(sizeof(int) * static_cast<size_t>(header.num_channels)));
  header.requested_pixel_types = static_cast<int *>(
      malloc(sizeof(int) * static_cast<size_t>(header.num_channels)));
  for (int i = 0; i < header.num_channels; i++) {
    header.pixel_types[i] =
        TINYEXR_PIXELTYPE_FLOAT;  // pixel type of input image

    if (save_as_fp16 > 0) {
      header.requested_pixel_types[i] =
          TINYEXR_PIXELTYPE_HALF;  // save with half(fp16) pixel format
    } else {
      header.requested_pixel_types[i] =
          TINYEXR_PIXELTYPE_FLOAT;  // save with float(fp32) pixel format(i.e.
                                    // no precision reduction)
    }
  }

  int ret = SaveEXRImageToFile(&image, &header, outfilename, err);
  if (ret != TINYEXR_SUCCESS) {
    return ret;
  }

  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);

  return ret;
}

#ifdef __clang__
// zero-as-null-ppinter-constant
#pragma clang diagnostic pop
#endif

#endif  // TINYEXR_IMPLEMENTATION_DEIFNED
#endif  // TINYEXR_IMPLEMENTATION

/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_TOOLS_COMMON_H_
#define VPX_TOOLS_COMMON_H_

#include <stdio.h>

#include "./vpx_config.h"
#include "vpx/vpx_codec.h"
#include "vpx/vpx_image.h"
#include "vpx/vpx_integer.h"

#if CONFIG_ENCODERS
#include "./y4minput.h"
#endif

#if defined(_MSC_VER)
/* MSVS uses _f{seek,tell}i64. */
#define fseeko _fseeki64
#define ftello _ftelli64
typedef int64_t FileOffset;
#elif defined(_WIN32)
/* MinGW uses f{seek,tell}o64 for large files. */
#define fseeko fseeko64
#define ftello ftello64
typedef off64_t FileOffset;
#elif CONFIG_OS_SUPPORT &&                                                  \
    !(defined(__ANDROID__) && __ANDROID_API__ < 24 && !defined(__LP64__) && \
      defined(_FILE_OFFSET_BITS) && _FILE_OFFSET_BITS == 64)
/* POSIX.1 has fseeko and ftello. fseeko and ftello are not available before
 * Android API level 24. See
 * https://android.googlesource.com/platform/bionic/+/main/docs/32-bit-abi.md */
#include <sys/types.h> /* NOLINT */
typedef off_t FileOffset;
/* Use 32-bit file operations in WebM file format when building ARM
 * executables (.axf) with RVCT. */
#else
#define fseeko fseek
#define ftello ftell
typedef long FileOffset; /* NOLINT */
#endif /* CONFIG_OS_SUPPORT */

#if CONFIG_OS_SUPPORT
#if defined(_MSC_VER)
#include <io.h> /* NOLINT */
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h> /* NOLINT */
#endif              /* _MSC_VER */
#endif              /* CONFIG_OS_SUPPORT */

#define LITERALU64(hi, lo) ((((uint64_t)hi) << 32) | lo)

#ifndef PATH_MAX
#define PATH_MAX 512
#endif

#define IVF_FRAME_HDR_SZ (4 + 8) /* 4 byte size + 8 byte timestamp */
#define IVF_FILE_HDR_SZ 32

#define RAW_FRAME_HDR_SZ sizeof(uint32_t)

#define VP8_FOURCC 0x30385056
#define VP9_FOURCC 0x30395056

enum VideoFileType {
  FILE_TYPE_RAW,
  FILE_TYPE_IVF,
  FILE_TYPE_Y4M,
  FILE_TYPE_WEBM
};

struct FileTypeDetectionBuffer {
  char buf[4];
  size_t buf_read;
  size_t position;
};

struct VpxRational {
  int numerator;
  int denominator;
};

struct VpxInputContext {
  const char *filename;
  FILE *file;
  int64_t length;
  struct FileTypeDetectionBuffer detect;
  enum VideoFileType file_type;
  uint32_t width;
  uint32_t height;
  struct VpxRational pixel_aspect_ratio;
  vpx_img_fmt_t fmt;
  vpx_bit_depth_t bit_depth;
  int only_i420;
  uint32_t fourcc;
  struct VpxRational framerate;
#if CONFIG_ENCODERS
  y4m_input y4m;
#endif
};

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define VPX_NO_RETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define VPX_NO_RETURN __declspec(noreturn)
#else
#define VPX_NO_RETURN
#endif

// Tells the compiler to perform `printf` format string checking if the
// compiler supports it; see the 'format' attribute in
// <https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html>.
#define VPX_TOOLS_FORMAT_PRINTF(string_index, first_to_check)
#if defined(__has_attribute)
#if __has_attribute(format)
#undef VPX_TOOLS_FORMAT_PRINTF
#define VPX_TOOLS_FORMAT_PRINTF(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))
#endif
#endif

/* Sets a stdio stream into binary mode */
FILE *set_binary_mode(FILE *stream);

VPX_NO_RETURN void die(const char *fmt, ...) VPX_TOOLS_FORMAT_PRINTF(1, 2);
VPX_NO_RETURN void fatal(const char *fmt, ...) VPX_TOOLS_FORMAT_PRINTF(1, 2);
void warn(const char *fmt, ...) VPX_TOOLS_FORMAT_PRINTF(1, 2);

VPX_NO_RETURN void die_codec(vpx_codec_ctx_t *ctx, const char *s);

/* The tool including this file must define usage_exit() */
VPX_NO_RETURN void usage_exit(void);

#undef VPX_NO_RETURN

int read_yuv_frame(struct VpxInputContext *input_ctx, vpx_image_t *yuv_frame);

typedef struct VpxInterface {
  const char *name;
  uint32_t fourcc;
  vpx_codec_iface_t *(*codec_interface)(void);
} VpxInterface;

int get_vpx_encoder_count(void);
const VpxInterface *get_vpx_encoder_by_index(int i);
const VpxInterface *get_vpx_encoder_by_name(const char *name);

int get_vpx_decoder_count(void);
const VpxInterface *get_vpx_decoder_by_index(int i);
const VpxInterface *get_vpx_decoder_by_name(const char *name);
const VpxInterface *get_vpx_decoder_by_fourcc(uint32_t fourcc);

int vpx_img_plane_width(const vpx_image_t *img, int plane);
int vpx_img_plane_height(const vpx_image_t *img, int plane);
void vpx_img_write(const vpx_image_t *img, FILE *file);
int vpx_img_read(vpx_image_t *img, FILE *file);

double sse_to_psnr(double samples, double peak, double mse);

#if CONFIG_ENCODERS
int read_frame(struct VpxInputContext *input_ctx, vpx_image_t *img);
int file_is_y4m(const char detect[4]);
int fourcc_is_ivf(const char detect[4]);
void open_input_file(struct VpxInputContext *input);
void close_input_file(struct VpxInputContext *input);
#endif

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_img_upshift(vpx_image_t *dst, vpx_image_t *src, int input_shift);
void vpx_img_downshift(vpx_image_t *dst, vpx_image_t *src, int down_shift);
void vpx_img_truncate_16_to_8(vpx_image_t *dst, vpx_image_t *src);
#endif

int compare_img(const vpx_image_t *const img1, const vpx_image_t *const img2);
#if CONFIG_VP9_HIGHBITDEPTH
void find_mismatch_high(const vpx_image_t *const img1,
                        const vpx_image_t *const img2, int yloc[4], int uloc[4],
                        int vloc[4]);
#endif
void find_mismatch(const vpx_image_t *const img1, const vpx_image_t *const img2,
                   int yloc[4], int uloc[4], int vloc[4]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // VPX_TOOLS_COMMON_H_

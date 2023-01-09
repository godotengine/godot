/*
 * Copyright 2017 Google
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef MESA_LOG_H
#define MESA_LOG_H

#include <stdarg.h>

#include "util/macros.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MESA_LOG_TAG
#define MESA_LOG_TAG "MESA"
#endif

enum mesa_log_level {
   MESA_LOG_ERROR,
   MESA_LOG_WARN,
   MESA_LOG_INFO,
   MESA_LOG_DEBUG,
};

void PRINTFLIKE(3, 4)
mesa_log(enum mesa_log_level, const char *tag, const char *format, ...);

void
mesa_log_v(enum mesa_log_level, const char *tag, const char *format,
            va_list va);

#define mesa_loge(fmt, ...) mesa_log(MESA_LOG_ERROR, (MESA_LOG_TAG), (fmt), ##__VA_ARGS__)
#define mesa_logw(fmt, ...) mesa_log(MESA_LOG_WARN, (MESA_LOG_TAG), (fmt), ##__VA_ARGS__)
#define mesa_logi(fmt, ...) mesa_log(MESA_LOG_INFO, (MESA_LOG_TAG), (fmt), ##__VA_ARGS__)
#ifdef DEBUG
#define mesa_logd(fmt, ...) mesa_log(MESA_LOG_DEBUG, (MESA_LOG_TAG), (fmt), ##__VA_ARGS__)
#else
#define mesa_logd(fmt, ...) __mesa_log_use_args((fmt), ##__VA_ARGS__)
#endif

#define mesa_loge_v(fmt, va) mesa_log_v(MESA_LOG_ERROR, (MESA_LOG_TAG), (fmt), (va))
#define mesa_logw_v(fmt, va) mesa_log_v(MESA_LOG_WARN, (MESA_LOG_TAG), (fmt), (va))
#define mesa_logi_v(fmt, va) mesa_log_v(MESA_LOG_INFO, (MESA_LOG_TAG), (fmt), (va))
#ifdef DEBUG
#define mesa_logd_v(fmt, va) mesa_log_v(MESA_LOG_DEBUG, (MESA_LOG_TAG), (fmt), (va))
#else
#define mesa_logd_v(fmt, va) __mesa_log_use_args((fmt), (va))
#endif

#define mesa_log_once(level, fmt, ...)        \
   do                                         \
   {                                          \
      static bool once;                       \
      if (!once) {                            \
         once = true;                         \
         mesa_log(level, (MESA_LOG_TAG), fmt, ##__VA_ARGS__); \
      }                                       \
   } while (0)

#define mesa_loge_once(fmt, ...) mesa_log_once(MESA_LOG_ERROR, fmt, ##__VA_ARGS__)
#define mesa_logw_once(fmt, ...) mesa_log_once(MESA_LOG_WARN, fmt, ##__VA_ARGS__)
#define mesa_logi_once(fmt, ...) mesa_log_once(MESA_LOG_INFO, fmt, ##__VA_ARGS__)
#define mesa_logd_once(fmt, ...) mesa_log_once(MESA_LOG_DEBUG, fmt, ##__VA_ARGS__)

struct log_stream {
   char *msg;
   const char *tag;
   size_t pos;
   enum mesa_log_level level;
};

struct log_stream *_mesa_log_stream_create(enum mesa_log_level level, char *tag);
#define mesa_log_streame() _mesa_log_stream_create(MESA_LOG_ERROR, (MESA_LOG_TAG))
#define mesa_log_streamw() _mesa_log_stream_create(MESA_LOG_WARN, (MESA_LOG_TAG))
#define mesa_log_streami() _mesa_log_stream_create(MESA_LOG_INFO, (MESA_LOG_TAG))
void mesa_log_stream_destroy(struct log_stream *stream);
void mesa_log_stream_printf(struct log_stream *stream, const char *format, ...);

void _mesa_log_multiline(enum mesa_log_level level, const char *tag, const char *lines);
#define mesa_log_multiline(level, lines) _mesa_log_multiline(level, (MESA_LOG_TAG), lines)

#ifndef DEBUG
/* Suppres -Wunused */
static inline void PRINTFLIKE(1, 2)
__mesa_log_use_args(UNUSED const char *format, ...) { }
#endif

#ifdef __cplusplus
}
#endif

#endif /* MESA_LOG_H */

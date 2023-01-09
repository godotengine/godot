/*
 * Copyright © 2017 Google, Inc.
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

#include <stdarg.h>

#ifdef ANDROID
#include <android/log.h>
#else
#include <stdio.h>
#endif

#include <stdlib.h>
#include <string.h>
#include "util/detect_os.h"
#include "util/log.h"
#include "util/ralloc.h"

#ifdef ANDROID
static inline android_LogPriority
level_to_android(enum mesa_log_level l)
{
   switch (l) {
   case MESA_LOG_ERROR: return ANDROID_LOG_ERROR;
   case MESA_LOG_WARN: return ANDROID_LOG_WARN;
   case MESA_LOG_INFO: return ANDROID_LOG_INFO;
   case MESA_LOG_DEBUG: return ANDROID_LOG_DEBUG;
   }

   unreachable("bad mesa_log_level");
}
#endif

#ifndef ANDROID
static inline const char *
level_to_str(enum mesa_log_level l)
{
   switch (l) {
   case MESA_LOG_ERROR: return "error";
   case MESA_LOG_WARN: return "warning";
   case MESA_LOG_INFO: return "info";
   case MESA_LOG_DEBUG: return "debug";
   }

   unreachable("bad mesa_log_level");
}
#endif

void
mesa_log(enum mesa_log_level level, const char *tag, const char *format, ...)
{
   va_list va;

   va_start(va, format);
   mesa_log_v(level, tag, format, va);
   va_end(va);
}

void
mesa_log_v(enum mesa_log_level level, const char *tag, const char *format,
            va_list va)
{
#ifdef ANDROID
   __android_log_vprint(level_to_android(level), tag, format, va);
#else
#if !DETECT_OS_WINDOWS
   flockfile(stderr);
#endif
   fprintf(stderr, "%s: %s: ", tag, level_to_str(level));
   vfprintf(stderr, format, va);
   if (format[strlen(format) - 1] != '\n')
      fprintf(stderr, "\n");
#if !DETECT_OS_WINDOWS
   funlockfile(stderr);
#endif
#endif
}

struct log_stream *
_mesa_log_stream_create(enum mesa_log_level level, char *tag)
{
   struct log_stream *stream = ralloc(NULL, struct log_stream);
   stream->level = level;
   stream->tag = tag;
   stream->msg = ralloc_strdup(stream, "");
   stream->pos = 0;
   return stream;
}

void
mesa_log_stream_destroy(struct log_stream *stream)
{
   /* If you left trailing stuff in the log stream, flush it out as a line. */
   if (stream->pos != 0)
      mesa_log(stream->level, stream->tag, "%s", stream->msg);

   ralloc_free(stream);
}

static void
mesa_log_stream_flush(struct log_stream *stream, size_t scan_offset)
{
   char *end;
   char *next = stream->msg;
   while ((end = strchr(stream->msg + scan_offset, '\n'))) {
      *end = 0;
      mesa_log(stream->level, stream->tag, "%s", next);
      next = end + 1;
      scan_offset = next - stream->msg;
   }
   if (next != stream->msg) {
      /* Clear out the lines we printed and move any trailing chars to the start. */
      size_t remaining = stream->msg + stream->pos - next;
      memmove(stream->msg, next, remaining);
      stream->pos = remaining;
   }
}

void mesa_log_stream_printf(struct log_stream *stream, const char *format, ...)
{
   size_t old_pos = stream->pos;

   va_list va;
   va_start(va, format);
   ralloc_vasprintf_rewrite_tail(&stream->msg, &stream->pos, format, va);
   va_end(va);

   mesa_log_stream_flush(stream, old_pos);
}

void
_mesa_log_multiline(enum mesa_log_level level, const char *tag, const char *lines)
{
   struct log_stream tmp = {
      .level = level,
      .tag = tag,
      .msg = strdup(lines),
      .pos = strlen(lines),
   };
   mesa_log_stream_flush(&tmp, 0);
   free(tmp.msg);
}

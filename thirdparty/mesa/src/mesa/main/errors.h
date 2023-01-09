/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */


/**
 * \file errors.h
 * Mesa debugging and error handling functions.
 *
 * This file provides functions to record errors, warnings, and miscellaneous
 * debug information.
 */


#ifndef ERRORS_H
#define ERRORS_H


#include <stdio.h>
#include <stdarg.h>
#include "util/glheader.h"
#include "menums.h"


#ifdef __cplusplus
extern "C" {
#endif

struct gl_context;

extern void
_mesa_warning( struct gl_context *gc, const char *fmtString, ... ) PRINTFLIKE(2, 3);

extern void
_mesa_problem( const struct gl_context *ctx, const char *fmtString, ... ) PRINTFLIKE(2, 3);

extern void
_mesa_error( struct gl_context *ctx, GLenum error, const char *fmtString, ... ) PRINTFLIKE(3, 4);

extern void
_mesa_error_no_memory(const char *caller);

extern void
_mesa_debug( const struct gl_context *ctx, const char *fmtString, ... ) PRINTFLIKE(2, 3);

extern void
_mesa_log(const char *fmtString, ...) PRINTFLIKE(1, 2);

extern void
_mesa_log_direct(const char *string);


extern FILE *
_mesa_get_log_file(void);

void
_mesa_shader_debug(struct gl_context *ctx, GLenum type, GLuint *id,
                   const char *msg);

extern void
_mesa_gl_vdebugf(struct gl_context *ctx,
                 GLuint *id,
                 enum mesa_debug_source source,
                 enum mesa_debug_type type,
                 enum mesa_debug_severity severity,
                 const char *fmtString,
                 va_list args);

extern void
_mesa_gl_debugf(struct gl_context *ctx,
                GLuint *id,
                enum mesa_debug_source source,
                enum mesa_debug_type type,
                enum mesa_debug_severity severity,
                const char *fmtString, ...) PRINTFLIKE(6, 7);

extern size_t
_mesa_gl_debug(struct gl_context *ctx,
               GLuint *id,
               enum mesa_debug_source source,
               enum mesa_debug_type type,
               enum mesa_debug_severity severity,
               const char *msg);

#define _mesa_perf_debug(ctx, sev, ...) do {                              \
   static GLuint msg_id = 0;                                              \
   if (unlikely(ctx->Const.ContextFlags & GL_CONTEXT_FLAG_DEBUG_BIT)) {   \
      _mesa_gl_debugf(ctx, &msg_id,                                       \
                      MESA_DEBUG_SOURCE_API,                              \
                      MESA_DEBUG_TYPE_PERFORMANCE,                        \
                      sev,                                                \
                      __VA_ARGS__);                                       \
   }                                                                      \
} while (0)

#ifdef __cplusplus
}
#endif


#endif /* ERRORS_H */

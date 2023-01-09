/**
 * \file extensions.h
 * Extension handling.
 * 
 * \if subset
 * (No-op)
 *
 * \endif
 */

/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2006  Brian Paul   All Rights Reserved.
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


#ifndef _EXTENSIONS_H_
#define _EXTENSIONS_H_

#include "mtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gl_context;
struct gl_extensions;

extern void _mesa_one_time_init_extension_overrides(const char *override);

extern void _mesa_init_extensions(struct gl_extensions *extentions);

extern GLubyte *_mesa_make_extension_string(struct gl_context *ctx);

extern void _mesa_override_extensions(struct gl_context *ctx);

extern GLuint
_mesa_get_extension_count(struct gl_context *ctx);

extern const GLubyte *
_mesa_get_enabled_extension(struct gl_context *ctx, GLuint index);


/**
 * \brief An element of the \c extension_table.
 */
struct mesa_extension {
   /** Name of extension, such as "GL_ARB_depth_clamp". */
   const char *name;

   /** Offset (in bytes) of the corresponding member in struct gl_extensions. */
   size_t offset;

   /** Minimum version the extension requires for the given API
    * (see gl_api defined in mtypes.h). The value is equal to:
    * 10 * major_version + minor_version
    */
   uint8_t version[API_OPENGL_LAST + 1];

   /** Year the extension was proposed or approved.  Used to sort the 
    * extension string chronologically. */
   uint16_t year;
};

extern const struct mesa_extension _mesa_extension_table[];


/* Generate enums for the functions below */
enum {
#define EXT(name_str, ...) MESA_EXTENSION_##name_str,
#include "extensions_table.h"
#undef EXT
MESA_EXTENSION_COUNT
};


/** Checks if the context supports a user-facing extension */
#define EXT(name_str, driver_cap, ...) \
static inline bool \
_mesa_has_##name_str(const struct gl_context *ctx) \
{ \
   return ctx->Extensions.driver_cap && (ctx->Extensions.Version >= \
          _mesa_extension_table[MESA_EXTENSION_##name_str].version[ctx->API]); \
}
#include "extensions_table.h"
#undef EXT

/* Sometimes the driver wants to query the extension override status before
 * a context is created. These variables are filled with extension override
 * information before context creation.
 *
 * This can be useful during extension bring-up when an extension is
 * partially implemented, but cannot yet be advertised as supported.
 *
 * Use it with care and keep access read-only.
 */
extern struct gl_extensions _mesa_extension_override_enables;
extern struct gl_extensions _mesa_extension_override_disables;

#ifdef __cplusplus
}
#endif

#endif

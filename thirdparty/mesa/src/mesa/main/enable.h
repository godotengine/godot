/**
 * \file enable.h
 * Enable/disable/query GL capabilities.
 */

/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2001  Brian Paul   All Rights Reserved.
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


#ifndef ENABLE_H
#define ENABLE_H


#include "util/glheader.h"

struct gl_context;


extern void
_mesa_update_derived_primitive_restart_state(struct gl_context *ctx);

extern void
_mesa_set_enable( struct gl_context* ctx, GLenum cap, GLboolean state );

extern void
_mesa_set_enablei(struct gl_context *ctx, GLenum cap, GLuint index, GLboolean state);

extern void
_mesa_set_multisample(struct gl_context *ctx, GLboolean state);

extern void
_mesa_set_framebuffer_srgb(struct gl_context *ctx, GLboolean state);

#endif

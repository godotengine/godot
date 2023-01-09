/**************************************************************************
 * 
 * Copyright 2003 VMware, Inc.
 * All Rights Reserved.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 **************************************************************************/


#ifndef PROG_CACHE_H
#define PROG_CACHE_H


#include "util/glheader.h"


#ifdef __cplusplus
extern "C" {
#endif


struct gl_context;

/** Opaque type */
struct gl_program_cache;


extern struct gl_program_cache *
_mesa_new_program_cache(void);

extern void
_mesa_delete_program_cache(struct gl_context *ctx, struct gl_program_cache *pc);

extern void
_mesa_delete_shader_cache(struct gl_context *ctx,
			  struct gl_program_cache *cache);

extern struct gl_program *
_mesa_search_program_cache(struct gl_program_cache *cache,
                           const void *key, GLuint keysize);

extern void
_mesa_program_cache_insert(struct gl_context *ctx,
                           struct gl_program_cache *cache,
                           const void *key, GLuint keysize,
                           struct gl_program *program);

void
_mesa_shader_cache_insert(struct gl_context *ctx,
			  struct gl_program_cache *cache,
			  const void *key, GLuint keysize,
			  struct gl_shader_program *program);


#ifdef __cplusplus
}
#endif


#endif /* PROG_CACHE_H */

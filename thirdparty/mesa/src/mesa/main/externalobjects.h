/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2017 Red Hat.
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Authors: Dave Airlie <airlied@gmail.com>
 * 	    Andres Rodriguez <andresx7@gmail.com>
 */

/**
 * \file externalobjects.h
 *
 * Declarations of functions related to the API interop extensions.
 */

#ifndef EXTERNALOBJECTS_H
#define EXTERNALOBJECTS_H

#include "util/glheader.h"
#include "hash.h"

static inline struct gl_memory_object *
_mesa_lookup_memory_object(struct gl_context *ctx, GLuint memory)
{
   if (!memory)
      return NULL;

   return (struct gl_memory_object *)
      _mesa_HashLookup(ctx->Shared->MemoryObjects, memory);
}

static inline struct gl_memory_object *
_mesa_lookup_memory_object_locked(struct gl_context *ctx, GLuint memory)
{
   if (!memory)
      return NULL;

   return (struct gl_memory_object *)
      _mesa_HashLookupLocked(ctx->Shared->MemoryObjects, memory);
}

static inline struct gl_semaphore_object *
_mesa_lookup_semaphore_object(struct gl_context *ctx, GLuint semaphore)
{
   if (!semaphore)
      return NULL;

   return (struct gl_semaphore_object *)
      _mesa_HashLookup(ctx->Shared->SemaphoreObjects, semaphore);
}

static inline struct gl_semaphore_object *
_mesa_lookup_semaphore_object_locked(struct gl_context *ctx, GLuint semaphore)
{
   if (!semaphore)
      return NULL;

   return (struct gl_semaphore_object *)
      _mesa_HashLookupLocked(ctx->Shared->SemaphoreObjects, semaphore);
}

extern void
_mesa_delete_memory_object(struct gl_context *ctx,
                           struct gl_memory_object *semObj);

extern void
_mesa_delete_semaphore_object(struct gl_context *ctx,
                              struct gl_semaphore_object *semObj);

#endif

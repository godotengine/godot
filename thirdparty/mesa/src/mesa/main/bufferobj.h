/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 * Copyright (C) 2009  VMware, Inc.  All Rights Reserved.
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



#ifndef BUFFEROBJ_H
#define BUFFEROBJ_H

#include <stdbool.h>
#include "mtypes.h"


/*
 * Internal functions
 */

static inline struct pipe_resource *
_mesa_get_bufferobj_reference(struct gl_context *ctx, struct gl_buffer_object *obj)
{
   if (unlikely(!obj))
      return NULL;

   struct pipe_resource *buffer = obj->buffer;

   if (unlikely(!buffer))
      return NULL;

   /* Only one context is using the fast path. All other contexts must use
    * the slow path.
    */
   if (unlikely(obj->private_refcount_ctx != ctx)) {
      p_atomic_inc(&buffer->reference.count);
      return buffer;
   }

   if (unlikely(obj->private_refcount <= 0)) {
      assert(obj->private_refcount == 0);

      /* This is the number of atomic increments we will skip. */
      obj->private_refcount = 100000000;
      p_atomic_add(&buffer->reference.count, obj->private_refcount);
   }

   /* Return a buffer reference while decrementing the private refcount. */
   obj->private_refcount--;
   return buffer;
}

void _mesa_bufferobj_subdata(struct gl_context *ctx,
                          GLintptrARB offset,
                          GLsizeiptrARB size,
                          const void * data, struct gl_buffer_object *obj);
GLboolean _mesa_bufferobj_data(struct gl_context *ctx,
                            GLenum target,
                            GLsizeiptrARB size,
                            const void *data,
                            GLenum usage,
                            GLbitfield storageFlags,
                            struct gl_buffer_object *obj);
void
_mesa_bufferobj_get_subdata(struct gl_context *ctx,
                            GLintptrARB offset,
                            GLsizeiptrARB size,
                            void *data, struct gl_buffer_object *obj);

void *_mesa_bufferobj_map_range(struct gl_context *ctx,
                                GLintptr offset, GLsizeiptr length,
                                GLbitfield access,
                                struct gl_buffer_object *obj,
                                gl_map_buffer_index index);

void _mesa_bufferobj_flush_mapped_range(struct gl_context *ctx,
                                        GLintptr offset, GLsizeiptr length,
                                        struct gl_buffer_object *obj,
                                        gl_map_buffer_index index);
GLboolean _mesa_bufferobj_unmap(struct gl_context *ctx, struct gl_buffer_object *obj,
                                 gl_map_buffer_index index);

struct gl_buffer_object *
_mesa_bufferobj_alloc(struct gl_context *ctx, GLuint id);
void
_mesa_bufferobj_release_buffer(struct gl_buffer_object *obj);

enum pipe_map_flags
_mesa_access_flags_to_transfer_flags(GLbitfield access, bool wholeBuffer);

/** Is the given buffer object currently mapped by the GL user? */
static inline GLboolean
_mesa_bufferobj_mapped(const struct gl_buffer_object *obj,
                       gl_map_buffer_index index)
{
   return obj->Mappings[index].Pointer != NULL;
}

/**
 * Check whether the given buffer object is illegally mapped prior to
 * drawing from (or reading back to) the buffer.
 * Note that it's legal for a buffer to be mapped at draw/readback time
 * if it was mapped persistently (See GL_ARB_buffer_storage spec).
 * \return true if the buffer is illegally mapped, false otherwise
 */
static inline bool
_mesa_check_disallowed_mapping(const struct gl_buffer_object *obj)
{
   return _mesa_bufferobj_mapped(obj, MAP_USER) &&
          !(obj->Mappings[MAP_USER].AccessFlags &
            GL_MAP_PERSISTENT_BIT);
}


extern void
_mesa_init_buffer_objects(struct gl_context *ctx);

extern void
_mesa_free_buffer_objects(struct gl_context *ctx);

extern bool
_mesa_handle_bind_buffer_gen(struct gl_context *ctx,
                             GLuint buffer,
                             struct gl_buffer_object **buf_handle,
                             const char *caller, bool no_error);

extern void
_mesa_update_default_objects_buffer_objects(struct gl_context *ctx);


extern struct gl_buffer_object *
_mesa_lookup_bufferobj(struct gl_context *ctx, GLuint buffer);

extern struct gl_buffer_object *
_mesa_lookup_bufferobj_locked(struct gl_context *ctx, GLuint buffer);

extern struct gl_buffer_object *
_mesa_lookup_bufferobj_err(struct gl_context *ctx, GLuint buffer,
                           const char *caller);

extern struct gl_buffer_object *
_mesa_multi_bind_lookup_bufferobj(struct gl_context *ctx,
                                  const GLuint *buffers,
                                  GLuint index, const char *caller,
                                  bool *error);

extern void
_mesa_delete_buffer_object(struct gl_context *ctx,
                           struct gl_buffer_object *bufObj);

/**
 * Set ptr to bufObj w/ reference counting.
 * This is normally only called from the _mesa_reference_buffer_object() macro
 * when there's a real pointer change.
 */
static inline void
_mesa_reference_buffer_object_(struct gl_context *ctx,
                               struct gl_buffer_object **ptr,
                               struct gl_buffer_object *bufObj,
                               bool shared_binding)
{
   if (*ptr) {
      /* Unreference the old buffer */
      struct gl_buffer_object *oldObj = *ptr;

      assert(oldObj->RefCount >= 1);

      /* Count references only if the context doesn't own the buffer or if
       * ptr is a binding point shared by multiple contexts (such as a texture
       * buffer object being a buffer bound within a texture object).
       */
      if (shared_binding || ctx != oldObj->Ctx) {
         if (p_atomic_dec_zero(&oldObj->RefCount)) {
            _mesa_delete_buffer_object(ctx, oldObj);
         }
      } else {
         /* Update the private ref count. */
         assert(oldObj->CtxRefCount >= 1);
         oldObj->CtxRefCount--;
      }
   }

   if (bufObj) {
      /* reference new buffer */
      if (shared_binding || ctx != bufObj->Ctx)
         p_atomic_inc(&bufObj->RefCount);
      else
         bufObj->CtxRefCount++;
   }

   *ptr = bufObj;
}

/**
 * Assign a buffer into a pointer with reference counting. The destination
 * must be private within a context.
 */
static inline void
_mesa_reference_buffer_object(struct gl_context *ctx,
                              struct gl_buffer_object **ptr,
                              struct gl_buffer_object *bufObj)
{
   if (*ptr != bufObj)
      _mesa_reference_buffer_object_(ctx, ptr, bufObj, false);
}

/**
 * Assign a buffer into a pointer with reference counting. The destination
 * must be shareable among multiple contexts.
 */
static inline void
_mesa_reference_buffer_object_shared(struct gl_context *ctx,
                                     struct gl_buffer_object **ptr,
                                     struct gl_buffer_object *bufObj)
{
   if (*ptr != bufObj)
      _mesa_reference_buffer_object_(ctx, ptr, bufObj, true);
}

extern GLuint
_mesa_total_buffer_object_memory(struct gl_context *ctx);

extern void
_mesa_buffer_data(struct gl_context *ctx, struct gl_buffer_object *bufObj,
                  GLenum target, GLsizeiptr size, const GLvoid *data,
                  GLenum usage, const char *func);

extern void
_mesa_buffer_sub_data(struct gl_context *ctx, struct gl_buffer_object *bufObj,
                      GLintptr offset, GLsizeiptr size, const GLvoid *data);

extern void
_mesa_buffer_unmap_all_mappings(struct gl_context *ctx,
                                struct gl_buffer_object *bufObj);

extern void
_mesa_ClearBufferSubData_sw(struct gl_context *ctx,
                            GLintptr offset, GLsizeiptr size,
                            const GLvoid *clearValue,
                            GLsizeiptr clearValueSize,
                            struct gl_buffer_object *bufObj);

void
_mesa_InternalBindElementBuffer(struct gl_context *ctx,
                                struct gl_buffer_object *buf);

#endif

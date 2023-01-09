/*
 * Copyright © 2010 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * \file ralloc.h
 *
 * ralloc: a recursive memory allocator
 *
 * The ralloc memory allocator creates a hierarchy of allocated
 * objects. Every allocation is in reference to some parent, and
 * every allocated object can in turn be used as the parent of a
 * subsequent allocation. This allows for extremely convenient
 * discarding of an entire tree/sub-tree of allocations by calling
 * ralloc_free on any particular object to free it and all of its
 * children.
 *
 * The conceptual working of ralloc was directly inspired by Andrew
 * Tridgell's talloc, but ralloc is an independent implementation
 * released under the MIT license and tuned for Mesa.
 *
 * talloc is more sophisticated than ralloc in that it includes reference
 * counting and useful debugging features.  However, it is released under
 * a non-permissive open source license.
 */

#ifndef RALLOC_H
#define RALLOC_H

#include <stddef.h>
#include <stdarg.h>
#include <stdbool.h>

#include "macros.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \def ralloc(ctx, type)
 * Allocate a new object chained off of the given context.
 *
 * This is equivalent to:
 * \code
 * ((type *) ralloc_size(ctx, sizeof(type))
 * \endcode
 */
#define ralloc(ctx, type)  ((type *) ralloc_size(ctx, sizeof(type)))

/**
 * \def rzalloc(ctx, type)
 * Allocate a new object out of the given context and initialize it to zero.
 *
 * This is equivalent to:
 * \code
 * ((type *) rzalloc_size(ctx, sizeof(type))
 * \endcode
 */
#define rzalloc(ctx, type) ((type *) rzalloc_size(ctx, sizeof(type)))

/**
 * Allocate a new ralloc context.
 *
 * While any ralloc'd pointer can be used as a context, sometimes it is useful
 * to simply allocate a context with no associated memory.
 *
 * It is equivalent to:
 * \code
 * ((type *) ralloc_size(ctx, 0)
 * \endcode
 */
void *ralloc_context(const void *ctx);

/**
 * Allocate memory chained off of the given context.
 *
 * This is the core allocation routine which is used by all others.  It
 * simply allocates storage for \p size bytes and returns the pointer,
 * similar to \c malloc.
 */
void *ralloc_size(const void *ctx, size_t size) MALLOCLIKE;

/**
 * Allocate zero-initialized memory chained off of the given context.
 *
 * This is similar to \c calloc with a size of 1.
 */
void *rzalloc_size(const void *ctx, size_t size) MALLOCLIKE;

/**
 * Resize a piece of ralloc-managed memory, preserving data.
 *
 * Similar to \c realloc.  Unlike C89, passing 0 for \p size does not free the
 * memory.  Instead, it resizes it to a 0-byte ralloc context, just like
 * calling ralloc_size(ctx, 0).  This is different from talloc.
 *
 * \param ctx  The context to use for new allocation.  If \p ptr != NULL,
 *             it must be the same as ralloc_parent(\p ptr).
 * \param ptr  Pointer to the memory to be resized.  May be NULL.
 * \param size The amount of memory to allocate, in bytes.
 */
void *reralloc_size(const void *ctx, void *ptr, size_t size);

/**
 * Resize a ralloc-managed array, preserving data and initializing any newly
 * allocated data to zero.
 *
 * Similar to \c realloc.  Unlike C89, passing 0 for \p size does not free the
 * memory.  Instead, it resizes it to a 0-byte ralloc context, just like
 * calling ralloc_size(ctx, 0).  This is different from talloc.
 *
 * \param ctx        The context to use for new allocation.  If \p ptr != NULL,
 *                   it must be the same as ralloc_parent(\p ptr).
 * \param ptr        Pointer to the memory to be resized.  May be NULL.
 * \param old_size   The amount of memory in the previous allocation, in bytes.
 * \param new_size   The amount of memory to allocate, in bytes.
 */
void *rerzalloc_size(const void *ctx, void *ptr,
                     size_t old_size, size_t new_size);

/// \defgroup array Array Allocators @{

/**
 * \def ralloc_array(ctx, type, count)
 * Allocate an array of objects chained off the given context.
 *
 * Similar to \c calloc, but does not initialize the memory to zero.
 *
 * More than a convenience function, this also checks for integer overflow when
 * multiplying \c sizeof(type) and \p count.  This is necessary for security.
 *
 * This is equivalent to:
 * \code
 * ((type *) ralloc_array_size(ctx, sizeof(type), count)
 * \endcode
 */
#define ralloc_array(ctx, type, count) \
   ((type *) ralloc_array_size(ctx, sizeof(type), count))

/**
 * \def rzalloc_array(ctx, type, count)
 * Allocate a zero-initialized array chained off the given context.
 *
 * Similar to \c calloc.
 *
 * More than a convenience function, this also checks for integer overflow when
 * multiplying \c sizeof(type) and \p count.  This is necessary for security.
 *
 * This is equivalent to:
 * \code
 * ((type *) rzalloc_array_size(ctx, sizeof(type), count)
 * \endcode
 */
#define rzalloc_array(ctx, type, count) \
   ((type *) rzalloc_array_size(ctx, sizeof(type), count))

/**
 * \def reralloc(ctx, ptr, type, count)
 * Resize a ralloc-managed array, preserving data.
 *
 * Similar to \c realloc.  Unlike C89, passing 0 for \p size does not free the
 * memory.  Instead, it resizes it to a 0-byte ralloc context, just like
 * calling ralloc_size(ctx, 0).  This is different from talloc.
 *
 * More than a convenience function, this also checks for integer overflow when
 * multiplying \c sizeof(type) and \p count.  This is necessary for security.
 *
 * \param ctx   The context to use for new allocation.  If \p ptr != NULL,
 *              it must be the same as ralloc_parent(\p ptr).
 * \param ptr   Pointer to the array to be resized.  May be NULL.
 * \param type  The element type.
 * \param count The number of elements to allocate.
 */
#define reralloc(ctx, ptr, type, count) \
   ((type *) reralloc_array_size(ctx, ptr, sizeof(type), count))

/**
 * \def rerzalloc(ctx, ptr, type, count)
 * Resize a ralloc-managed array, preserving data and initializing any newly
 * allocated data to zero.
 *
 * Similar to \c realloc.  Unlike C89, passing 0 for \p size does not free the
 * memory.  Instead, it resizes it to a 0-byte ralloc context, just like
 * calling ralloc_size(ctx, 0).  This is different from talloc.
 *
 * More than a convenience function, this also checks for integer overflow when
 * multiplying \c sizeof(type) and \p count.  This is necessary for security.
 *
 * \param ctx        The context to use for new allocation.  If \p ptr != NULL,
 *                   it must be the same as ralloc_parent(\p ptr).
 * \param ptr        Pointer to the array to be resized.  May be NULL.
 * \param type       The element type.
 * \param old_count  The number of elements in the previous allocation.
 * \param new_count  The number of elements to allocate.
 */
#define rerzalloc(ctx, ptr, type, old_count, new_count) \
   ((type *) rerzalloc_array_size(ctx, ptr, sizeof(type), old_count, new_count))

/**
 * Allocate memory for an array chained off the given context.
 *
 * Similar to \c calloc, but does not initialize the memory to zero.
 *
 * More than a convenience function, this also checks for integer overflow when
 * multiplying \p size and \p count.  This is necessary for security.
 */
void *ralloc_array_size(const void *ctx, size_t size, unsigned count) MALLOCLIKE;

/**
 * Allocate a zero-initialized array chained off the given context.
 *
 * Similar to \c calloc.
 *
 * More than a convenience function, this also checks for integer overflow when
 * multiplying \p size and \p count.  This is necessary for security.
 */
void *rzalloc_array_size(const void *ctx, size_t size, unsigned count) MALLOCLIKE;

/**
 * Resize a ralloc-managed array, preserving data.
 *
 * Similar to \c realloc.  Unlike C89, passing 0 for \p size does not free the
 * memory.  Instead, it resizes it to a 0-byte ralloc context, just like
 * calling ralloc_size(ctx, 0).  This is different from talloc.
 *
 * More than a convenience function, this also checks for integer overflow when
 * multiplying \c sizeof(type) and \p count.  This is necessary for security.
 *
 * \param ctx   The context to use for new allocation.  If \p ptr != NULL,
 *              it must be the same as ralloc_parent(\p ptr).
 * \param ptr   Pointer to the array to be resized.  May be NULL.
 * \param size  The size of an individual element.
 * \param count The number of elements to allocate.
 *
 * \return True unless allocation failed.
 */
void *reralloc_array_size(const void *ctx, void *ptr, size_t size,
			  unsigned count);

/**
 * Resize a ralloc-managed array, preserving data and initializing any newly
 * allocated data to zero.
 *
 * Similar to \c realloc.  Unlike C89, passing 0 for \p size does not free the
 * memory.  Instead, it resizes it to a 0-byte ralloc context, just like
 * calling ralloc_size(ctx, 0).  This is different from talloc.
 *
 * More than a convenience function, this also checks for integer overflow when
 * multiplying \c sizeof(type) and \p count.  This is necessary for security.
 *
 * \param ctx        The context to use for new allocation.  If \p ptr != NULL,
 *                   it must be the same as ralloc_parent(\p ptr).
 * \param ptr        Pointer to the array to be resized.  May be NULL.
 * \param size       The size of an individual element.
 * \param old_count  The number of elements in the previous allocation.
 * \param new_count  The number of elements to allocate.
 *
 * \return True unless allocation failed.
 */
void *rerzalloc_array_size(const void *ctx, void *ptr, size_t size,
			   unsigned old_count, unsigned new_count);
/// @}

/**
 * Free a piece of ralloc-managed memory.
 *
 * This will also free the memory of any children allocated this context.
 */
void ralloc_free(void *ptr);

/**
 * "Steal" memory from one context, changing it to another.
 *
 * This changes \p ptr's context to \p new_ctx.  This is quite useful if
 * memory is allocated out of a temporary context.
 */
void ralloc_steal(const void *new_ctx, void *ptr);

/**
 * Reparent all children from one context to another.
 *
 * This effectively calls ralloc_steal(new_ctx, child) for all children of \p old_ctx.
 */
void ralloc_adopt(const void *new_ctx, void *old_ctx);

/**
 * Return the given pointer's ralloc context.
 */
void *ralloc_parent(const void *ptr);

/**
 * Set a callback to occur just before an object is freed.
 */
void ralloc_set_destructor(const void *ptr, void(*destructor)(void *));

/// \defgroup array String Functions @{
/**
 * Duplicate a string, allocating the memory from the given context.
 */
char *ralloc_strdup(const void *ctx, const char *str) MALLOCLIKE;

/**
 * Duplicate a string, allocating the memory from the given context.
 *
 * Like \c strndup, at most \p n characters are copied.  If \p str is longer
 * than \p n characters, \p n are copied, and a termining \c '\0' byte is added.
 */
char *ralloc_strndup(const void *ctx, const char *str, size_t n) MALLOCLIKE;

/**
 * Concatenate two strings, allocating the necessary space.
 *
 * This appends \p str to \p *dest, similar to \c strcat, using ralloc_resize
 * to expand \p *dest to the appropriate size.  \p dest will be updated to the
 * new pointer unless allocation fails.
 *
 * The result will always be null-terminated.
 *
 * \return True unless allocation failed.
 */
bool ralloc_strcat(char **dest, const char *str);

/**
 * Concatenate two strings, allocating the necessary space.
 *
 * This appends at most \p n bytes of \p str to \p *dest, using ralloc_resize
 * to expand \p *dest to the appropriate size.  \p dest will be updated to the
 * new pointer unless allocation fails.
 *
 * The result will always be null-terminated; \p str does not need to be null
 * terminated if it is longer than \p n.
 *
 * \return True unless allocation failed.
 */
bool ralloc_strncat(char **dest, const char *str, size_t n);

/**
 * Concatenate two strings, allocating the necessary space.
 *
 * This appends \p n bytes of \p str to \p *dest, using ralloc_resize
 * to expand \p *dest to the appropriate size.  \p dest will be updated to the
 * new pointer unless allocation fails.
 *
 * The result will always be null-terminated.
 *
 * This function differs from ralloc_strcat() and ralloc_strncat() in that it
 * does not do any strlen() calls which can become costly on large strings.
 *
 * \return True unless allocation failed.
 */
bool
ralloc_str_append(char **dest, const char *str,
                  size_t existing_length, size_t str_size);

/**
 * Print to a string.
 *
 * This is analogous to \c sprintf, but allocates enough space (using \p ctx
 * as the context) for the resulting string.
 *
 * \return The newly allocated string.
 */
char *ralloc_asprintf (const void *ctx, const char *fmt, ...) PRINTFLIKE(2, 3) MALLOCLIKE;

/**
 * Print to a string, given a va_list.
 *
 * This is analogous to \c vsprintf, but allocates enough space (using \p ctx
 * as the context) for the resulting string.
 *
 * \return The newly allocated string.
 */
char *ralloc_vasprintf(const void *ctx, const char *fmt, va_list args) MALLOCLIKE;

/**
 * Rewrite the tail of an existing string, starting at a given index.
 *
 * Overwrites the contents of *str starting at \p start with newly formatted
 * text, including a new null-terminator.  Allocates more memory as necessary.
 *
 * This can be used to append formatted text when the length of the existing
 * string is already known, saving a strlen() call.
 *
 * \sa ralloc_asprintf_append
 *
 * \param str   The string to be updated.
 * \param start The index to start appending new data at.
 * \param fmt   A printf-style formatting string
 *
 * \p str will be updated to the new pointer unless allocation fails.
 * \p start will be increased by the length of the newly formatted text.
 *
 * \return True unless allocation failed.
 */
bool ralloc_asprintf_rewrite_tail(char **str, size_t *start,
				  const char *fmt, ...)
				  PRINTFLIKE(3, 4);

/**
 * Rewrite the tail of an existing string, starting at a given index.
 *
 * Overwrites the contents of *str starting at \p start with newly formatted
 * text, including a new null-terminator.  Allocates more memory as necessary.
 *
 * This can be used to append formatted text when the length of the existing
 * string is already known, saving a strlen() call.
 *
 * \sa ralloc_vasprintf_append
 *
 * \param str   The string to be updated.
 * \param start The index to start appending new data at.
 * \param fmt   A printf-style formatting string
 * \param args  A va_list containing the data to be formatted
 *
 * \p str will be updated to the new pointer unless allocation fails.
 * \p start will be increased by the length of the newly formatted text.
 *
 * \return True unless allocation failed.
 */
bool ralloc_vasprintf_rewrite_tail(char **str, size_t *start, const char *fmt,
				   va_list args);

/**
 * Append formatted text to the supplied string.
 *
 * This is equivalent to
 * \code
 * ralloc_asprintf_rewrite_tail(str, strlen(*str), fmt, ...)
 * \endcode
 *
 * \sa ralloc_asprintf
 * \sa ralloc_asprintf_rewrite_tail
 * \sa ralloc_strcat
 *
 * \p str will be updated to the new pointer unless allocation fails.
 *
 * \return True unless allocation failed.
 */
bool ralloc_asprintf_append (char **str, const char *fmt, ...)
			     PRINTFLIKE(2, 3);

/**
 * Append formatted text to the supplied string, given a va_list.
 *
 * This is equivalent to
 * \code
 * ralloc_vasprintf_rewrite_tail(str, strlen(*str), fmt, args)
 * \endcode
 *
 * \sa ralloc_vasprintf
 * \sa ralloc_vasprintf_rewrite_tail
 * \sa ralloc_strcat
 *
 * \p str will be updated to the new pointer unless allocation fails.
 *
 * \return True unless allocation failed.
 */
bool ralloc_vasprintf_append(char **str, const char *fmt, va_list args);
/// @}

typedef struct gc_ctx gc_ctx;

/**
 * Allocate a new garbage collection context. The children of the
 * context are not necessarily ralloc'd pointers and cannot be stolen to a ralloc context. Instead,
 * The user should use the mark-and-sweep interface below to free any unused children. Under the
 * hood, this restriction lets us manage allocations ourselves, using a freelist. This means that
 * GC contexts should be used for scenarios where there are many allocations and frees, most of
 * which use only a few different sizes.
 */
gc_ctx *gc_context(const void *parent);

#define gc_alloc(ctx, type, count) gc_alloc_size(ctx, sizeof(type) * (count), alignof(type))
#define gc_zalloc(ctx, type, count) gc_zalloc_size(ctx, sizeof(type) * (count), alignof(type))

#define gc_alloc_zla(ctx, type, type2, count) \
   gc_alloc_size(ctx, sizeof(type) + sizeof(type2) * (count), MAX2(alignof(type), alignof(type2)))
#define gc_zalloc_zla(ctx, type, type2, count) \
   gc_zalloc_size(ctx, sizeof(type) + sizeof(type2) * (count), MAX2(alignof(type), alignof(type2)))

void *gc_alloc_size(gc_ctx *ctx, size_t size, size_t align) MALLOCLIKE;
void *gc_zalloc_size(gc_ctx *ctx, size_t size, size_t align) MALLOCLIKE;
void gc_free(void *ptr);
gc_ctx *gc_get_context(void *ptr);

void gc_sweep_start(gc_ctx *ctx);
void gc_mark_live(gc_ctx *ctx, const void *mem);
void gc_sweep_end(gc_ctx *ctx);

/**
 * Declare C++ new and delete operators which use ralloc.
 *
 * Placing this macro in the body of a class makes it possible to do:
 *
 * TYPE *var = new(mem_ctx) TYPE(...);
 * delete var;
 *
 * which is more idiomatic in C++ than calling ralloc.
 */
#define DECLARE_ALLOC_CXX_OPERATORS_TEMPLATE(TYPE, ALLOC_FUNC)           \
private:                                                                 \
   static void _ralloc_destructor(void *p)                               \
   {                                                                     \
      reinterpret_cast<TYPE *>(p)->TYPE::~TYPE();                        \
   }                                                                     \
public:                                                                  \
   static void* operator new(size_t size, void *mem_ctx)                 \
   {                                                                     \
      void *p = ALLOC_FUNC(mem_ctx, size);                               \
      assert(p != NULL);                                                 \
      if (!HAS_TRIVIAL_DESTRUCTOR(TYPE))                                 \
         ralloc_set_destructor(p, _ralloc_destructor);                   \
      return p;                                                          \
   }                                                                     \
                                                                         \
   static void operator delete(void *p)                                  \
   {                                                                     \
      /* The object's destructor is guaranteed to have already been      \
       * called by the delete operator at this point -- Make sure it's   \
       * not called again.                                               \
       */                                                                \
      if (!HAS_TRIVIAL_DESTRUCTOR(TYPE))                                 \
         ralloc_set_destructor(p, NULL);                                 \
      ralloc_free(p);                                                    \
   }

#define DECLARE_RALLOC_CXX_OPERATORS(type) \
   DECLARE_ALLOC_CXX_OPERATORS_TEMPLATE(type, ralloc_size)

#define DECLARE_RZALLOC_CXX_OPERATORS(type) \
   DECLARE_ALLOC_CXX_OPERATORS_TEMPLATE(type, rzalloc_size)

#define DECLARE_LINEAR_ALLOC_CXX_OPERATORS(type) \
   DECLARE_ALLOC_CXX_OPERATORS_TEMPLATE(type, linear_alloc_child)

#define DECLARE_LINEAR_ZALLOC_CXX_OPERATORS(type) \
   DECLARE_ALLOC_CXX_OPERATORS_TEMPLATE(type, linear_zalloc_child)


/**
 * Do a fast allocation from the linear buffer, also known as the child node
 * from the allocator's point of view. It can't be freed directly. You have
 * to free the parent or the ralloc parent.
 *
 * \param parent   parent node of the linear allocator
 * \param size     size to allocate (max 32 bits)
 */
void *linear_alloc_child(void *parent, unsigned size);

/**
 * Allocate a parent node that will hold linear buffers. The returned
 * allocation is actually the first child node, but it's also the handle
 * of the parent node. Use it for all child node allocations.
 *
 * \param ralloc_ctx  ralloc context, must not be NULL
 * \param size        size to allocate (max 32 bits)
 */
void *linear_alloc_parent(void *ralloc_ctx, unsigned size);

/**
 * Same as linear_alloc_child, but also clears memory.
 */
void *linear_zalloc_child(void *parent, unsigned size);

/**
 * Same as linear_alloc_parent, but also clears memory.
 */
void *linear_zalloc_parent(void *ralloc_ctx, unsigned size);

/**
 * Free the linear parent node. This will free all child nodes too.
 * Freeing the ralloc parent will also free this.
 */
void linear_free_parent(void *ptr);

/**
 * Same as ralloc_steal, but steals the linear parent node.
 */
void ralloc_steal_linear_parent(void *new_ralloc_ctx, void *ptr);

/**
 * Return the ralloc parent of the linear parent node.
 */
void *ralloc_parent_of_linear_parent(void *ptr);

/**
 * Same as realloc except that the linear allocator doesn't free child nodes,
 * so it's reduced to memory duplication. It's used in places where
 * reallocation is required. Don't use it often. It's much slower than
 * realloc.
 */
void *linear_realloc(void *parent, void *old, unsigned new_size);

/* The functions below have the same semantics as their ralloc counterparts,
 * except that they always allocate a linear child node.
 */
char *linear_strdup(void *parent, const char *str);
char *linear_asprintf(void *parent, const char *fmt, ...);
char *linear_vasprintf(void *parent, const char *fmt, va_list args);
bool linear_asprintf_append(void *parent, char **str, const char *fmt, ...);
bool linear_vasprintf_append(void *parent, char **str, const char *fmt,
                             va_list args);
bool linear_asprintf_rewrite_tail(void *parent, char **str, size_t *start,
                                  const char *fmt, ...);
bool linear_vasprintf_rewrite_tail(void *parent, char **str, size_t *start,
                                   const char *fmt, va_list args);
bool linear_strcat(void *parent, char **dest, const char *str);

#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif

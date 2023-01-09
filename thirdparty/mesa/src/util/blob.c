/*
 * Copyright © 2014 Intel Corporation
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

#include <string.h>

#include "blob.h"
#include "u_math.h"

#ifdef HAVE_VALGRIND
#include <valgrind.h>
#include <memcheck.h>
#define VG(x) x
#else
#define VG(x)
#endif

#define BLOB_INITIAL_SIZE 4096

/* Ensure that \blob will be able to fit an additional object of size
 * \additional.  The growing (if any) will occur by doubling the existing
 * allocation.
 */
static bool
grow_to_fit(struct blob *blob, size_t additional)
{
   size_t to_allocate;
   uint8_t *new_data;

   if (blob->out_of_memory)
      return false;

   if (blob->size + additional <= blob->allocated)
      return true;

   if (blob->fixed_allocation) {
      blob->out_of_memory = true;
      return false;
   }

   if (blob->allocated == 0)
      to_allocate = BLOB_INITIAL_SIZE;
   else
      to_allocate = blob->allocated * 2;

   to_allocate = MAX2(to_allocate, blob->allocated + additional);

   new_data = realloc(blob->data, to_allocate);
   if (new_data == NULL) {
      blob->out_of_memory = true;
      return false;
   }

   blob->data = new_data;
   blob->allocated = to_allocate;

   return true;
}

/* Align the blob->size so that reading or writing a value at (blob->data +
 * blob->size) will result in an access aligned to a granularity of \alignment
 * bytes.
 *
 * \return True unless allocation fails
 */
bool
blob_align(struct blob *blob, size_t alignment)
{
   const size_t new_size = align64(blob->size, alignment);

   if (blob->size < new_size) {
      if (!grow_to_fit(blob, new_size - blob->size))
         return false;

      if (blob->data)
         memset(blob->data + blob->size, 0, new_size - blob->size);
      blob->size = new_size;
   }

   return true;
}

void
blob_reader_align(struct blob_reader *blob, size_t alignment)
{
   blob->current = blob->data + align64(blob->current - blob->data, alignment);
}

void
blob_init(struct blob *blob)
{
   blob->data = NULL;
   blob->allocated = 0;
   blob->size = 0;
   blob->fixed_allocation = false;
   blob->out_of_memory = false;
}

void
blob_init_fixed(struct blob *blob, void *data, size_t size)
{
   blob->data = data;
   blob->allocated = size;
   blob->size = 0;
   blob->fixed_allocation = true;
   blob->out_of_memory = false;
}

void
blob_finish_get_buffer(struct blob *blob, void **buffer, size_t *size)
{
   *buffer = blob->data;
   *size = blob->size;
   blob->data = NULL;

   /* Trim the buffer. */
   *buffer = realloc(*buffer, *size);
}

bool
blob_overwrite_bytes(struct blob *blob,
                     size_t offset,
                     const void *bytes,
                     size_t to_write)
{
   /* Detect an attempt to overwrite data out of bounds. */
   if (offset + to_write < offset || blob->size < offset + to_write)
      return false;

   VG(VALGRIND_CHECK_MEM_IS_DEFINED(bytes, to_write));

   if (blob->data)
      memcpy(blob->data + offset, bytes, to_write);

   return true;
}

bool
blob_write_bytes(struct blob *blob, const void *bytes, size_t to_write)
{
   if (! grow_to_fit(blob, to_write))
       return false;

   VG(VALGRIND_CHECK_MEM_IS_DEFINED(bytes, to_write));

   if (blob->data && to_write > 0)
      memcpy(blob->data + blob->size, bytes, to_write);
   blob->size += to_write;

   return true;
}

intptr_t
blob_reserve_bytes(struct blob *blob, size_t to_write)
{
   intptr_t ret;

   if (! grow_to_fit (blob, to_write))
      return -1;

   ret = blob->size;
   blob->size += to_write;

   return ret;
}

intptr_t
blob_reserve_uint32(struct blob *blob)
{
   blob_align(blob, sizeof(uint32_t));
   return blob_reserve_bytes(blob, sizeof(uint32_t));
}

intptr_t
blob_reserve_intptr(struct blob *blob)
{
   blob_align(blob, sizeof(intptr_t));
   return blob_reserve_bytes(blob, sizeof(intptr_t));
}

#define BLOB_WRITE_TYPE(name, type)                      \
bool                                                     \
name(struct blob *blob, type value)                      \
{                                                        \
   blob_align(blob, sizeof(value));                      \
   return blob_write_bytes(blob, &value, sizeof(value)); \
}

BLOB_WRITE_TYPE(blob_write_uint8, uint8_t)
BLOB_WRITE_TYPE(blob_write_uint16, uint16_t)
BLOB_WRITE_TYPE(blob_write_uint32, uint32_t)
BLOB_WRITE_TYPE(blob_write_uint64, uint64_t)
BLOB_WRITE_TYPE(blob_write_intptr, intptr_t)

#define ASSERT_ALIGNED(_offset, _align) \
   assert(align64((_offset), (_align)) == (_offset))

bool
blob_overwrite_uint8 (struct blob *blob,
                      size_t offset,
                      uint8_t value)
{
   ASSERT_ALIGNED(offset, sizeof(value));
   return blob_overwrite_bytes(blob, offset, &value, sizeof(value));
}

bool
blob_overwrite_uint32 (struct blob *blob,
                       size_t offset,
                       uint32_t value)
{
   ASSERT_ALIGNED(offset, sizeof(value));
   return blob_overwrite_bytes(blob, offset, &value, sizeof(value));
}

bool
blob_overwrite_intptr (struct blob *blob,
                       size_t offset,
                       intptr_t value)
{
   ASSERT_ALIGNED(offset, sizeof(value));
   return blob_overwrite_bytes(blob, offset, &value, sizeof(value));
}

bool
blob_write_string(struct blob *blob, const char *str)
{
   return blob_write_bytes(blob, str, strlen(str) + 1);
}

void
blob_reader_init(struct blob_reader *blob, const void *data, size_t size)
{
   blob->data = data;
   blob->end = blob->data + size;
   blob->current = data;
   blob->overrun = false;
}

/* Check that an object of size \size can be read from this blob.
 *
 * If not, set blob->overrun to indicate that we attempted to read too far.
 */
static bool
ensure_can_read(struct blob_reader *blob, size_t size)
{
   if (blob->overrun)
      return false;

   if (blob->current <= blob->end && blob->end - blob->current >= size)
      return true;

   blob->overrun = true;

   return false;
}

const void *
blob_read_bytes(struct blob_reader *blob, size_t size)
{
   const void *ret;

   if (! ensure_can_read (blob, size))
      return NULL;

   ret = blob->current;

   blob->current += size;

   return ret;
}

void
blob_copy_bytes(struct blob_reader *blob, void *dest, size_t size)
{
   const void *bytes;

   bytes = blob_read_bytes(blob, size);
   if (bytes == NULL || size == 0)
      return;

   memcpy(dest, bytes, size);
}

void
blob_skip_bytes(struct blob_reader *blob, size_t size)
{
   if (ensure_can_read (blob, size))
      blob->current += size;
}

#define BLOB_READ_TYPE(name, type)         \
type                                       \
name(struct blob_reader *blob)             \
{                                          \
   type ret = 0;                           \
   int size = sizeof(ret);                 \
   blob_reader_align(blob, size);          \
   blob_copy_bytes(blob, &ret, size);      \
   return ret;                             \
}

BLOB_READ_TYPE(blob_read_uint8, uint8_t)
BLOB_READ_TYPE(blob_read_uint16, uint16_t)
BLOB_READ_TYPE(blob_read_uint32, uint32_t)
BLOB_READ_TYPE(blob_read_uint64, uint64_t)
BLOB_READ_TYPE(blob_read_intptr, intptr_t)

char *
blob_read_string(struct blob_reader *blob)
{
   int size;
   char *ret;
   uint8_t *nul;

   /* If we're already at the end, then this is an overrun. */
   if (blob->current >= blob->end) {
      blob->overrun = true;
      return NULL;
   }

   /* Similarly, if there is no zero byte in the data remaining in this blob,
    * we also consider that an overrun.
    */
   nul = memchr(blob->current, 0, blob->end - blob->current);

   if (nul == NULL) {
      blob->overrun = true;
      return NULL;
   }

   size = nul - blob->current + 1;

   assert(ensure_can_read(blob, size));

   ret = (char *) blob->current;

   blob->current += size;

   return ret;
}

/*
 * Copyright © 2017 Thomas Helland
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
 *
 */

#include "string_buffer.h"

static bool
ensure_capacity(struct _mesa_string_buffer *str, uint32_t needed_capacity)
{
   if (needed_capacity <= str->capacity)
      return true;

   /* Too small, double until we can fit the new string */
   uint32_t new_capacity = str->capacity * 2;
   while (needed_capacity > new_capacity)
      new_capacity *= 2;

   str->buf = reralloc_array_size(str, str->buf, sizeof(char), new_capacity);
   if (str->buf == NULL)
      return false;

   str->capacity = new_capacity;
   return true;
}

struct _mesa_string_buffer *
_mesa_string_buffer_create(void *mem_ctx, uint32_t initial_capacity)
{
   struct _mesa_string_buffer *str;
   str = ralloc(mem_ctx, struct _mesa_string_buffer);

   if (str == NULL)
      return NULL;

   /* If no initial capacity is set then set it to something */
   str->capacity = initial_capacity ? initial_capacity : 32;
   str->buf = ralloc_array(str, char, str->capacity);

   if (!str->buf) {
      ralloc_free(str);
      return NULL;
   }

   str->length = 0;
   str->buf[str->length] = '\0';
   return str;
}

bool
_mesa_string_buffer_append_all(struct _mesa_string_buffer *str,
                               uint32_t num_args, ...)
{
   int i;
   char* s;
   va_list args;
   va_start(args, num_args);
   for (i = 0; i < num_args; i++) {
      s = va_arg(args, char*);
      if (!_mesa_string_buffer_append_len(str, s, strlen(s))) {
         va_end(args);
         return false;
      }
   }
   va_end(args);
   return true;
}

bool
_mesa_string_buffer_append_len(struct _mesa_string_buffer *str,
                               const char *c, uint32_t len)
{
   uint32_t needed_length = str->length + len + 1;

   /* Check if we're overflowing uint32_t */
   if (needed_length < str->length)
      return false;

   if (!ensure_capacity(str, needed_length))
      return false;

   memcpy(str->buf + str->length, c, len);
   str->length += len;
   str->buf[str->length] = '\0';
   return true;
}

bool
_mesa_string_buffer_vprintf(struct _mesa_string_buffer *str,
                            const char *format, va_list args)
{
   /* We're looping two times to avoid duplicating code */
   for (uint32_t i = 0; i < 2; i++) {
      va_list arg_copy;
      va_copy(arg_copy, args);
      uint32_t space_left = str->capacity - str->length;

      int32_t len = vsnprintf(str->buf + str->length,
                                   space_left, format, arg_copy);
      va_end(arg_copy);

      /* Error in vsnprintf() or measured len overflows size_t */
      if (unlikely(len < 0 || str->length + len + 1 < str->length))
         return false;

      /* There was enough space for the string; we're done */
      if (len < space_left) {
         str->length += len;
         return true;
      }

      /* Not enough space, resize and retry */
      ensure_capacity(str, str->length + len + 1);
   }

   return false;
}

bool
_mesa_string_buffer_printf(struct _mesa_string_buffer *str,
                            const char *format, ...)
{
   bool res;
   va_list args;
   va_start(args, format);
   res = _mesa_string_buffer_vprintf(str, format, args);
   va_end(args);
   return res;
}

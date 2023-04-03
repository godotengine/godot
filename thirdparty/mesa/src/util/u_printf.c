//
// Copyright 2020 Serge Martin
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// Extract from Serge's printf clover code by airlied.

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "macros.h"
#include "strndup.h"
#include "u_math.h"
#include "u_printf.h"

/* Some versions of MinGW are missing _vscprintf's declaration, although they
 * still provide the symbol in the import library. */
#ifdef __MINGW32__
_CRTIMP int _vscprintf(const char *format, va_list argptr);
#endif

static const char*
util_printf_prev_tok(const char *str)
{
   while (*str != '%')
      str--;
   return str;
}

size_t util_printf_next_spec_pos(const char *str, size_t pos)
{
   if (str == NULL)
      return -1;

   const char *str_found = str + pos;
   do {
      str_found = strchr(str_found, '%');
      if (str_found == NULL)
         return -1;

      ++str_found;
      if (*str_found == '%') {
         ++str_found;
         continue;
      }

      char *spec_pos = strpbrk(str_found, "cdieEfFgGaAosuxXp%");
      if (spec_pos == NULL) {
         return -1;
      } else if (*spec_pos == '%') {
         str_found = spec_pos;
      } else {
         return spec_pos - str;
      }
   } while (1);
}

size_t u_printf_length(const char *fmt, va_list untouched_args)
{
   int size;
   char junk;

   /* Make a copy of the va_list so the original caller can still use it */
   va_list args;
   va_copy(args, untouched_args);

#ifdef _WIN32
   /* We need to use _vcsprintf to calculate the size as vsnprintf returns -1
    * if the number of characters to write is greater than count.
    */
   size = _vscprintf(fmt, args);
   (void)junk;
#else
   size = vsnprintf(&junk, 1, fmt, args);
#endif
   assert(size >= 0);

   va_end(args);

   return size;
}

void u_printf(FILE *out, const char *buffer, size_t buffer_size,
              const u_printf_info *info, unsigned info_size)
{
   for (size_t buf_pos = 0; buf_pos < buffer_size;) {
      uint32_t fmt_idx = *(uint32_t*)&buffer[buf_pos];

      /* the idx is 1 based */
      assert(fmt_idx > 0);
      fmt_idx -= 1;

      /* The API allows more arguments than the format uses */
      if (fmt_idx >= info_size)
         return;

      const u_printf_info *fmt = &info[fmt_idx];
      const char *format = fmt->strings;
      buf_pos += sizeof(fmt_idx);

      if (!fmt->num_args) {
         fprintf(out, "%s", format);
         continue;
      }

      for (int i = 0; i < fmt->num_args; i++) {
         int arg_size = fmt->arg_sizes[i];
         size_t spec_pos = util_printf_next_spec_pos(format, 0);

         if (spec_pos == -1) {
            fprintf(out, "%s", format);
            continue;
         }

         const char *token = util_printf_prev_tok(&format[spec_pos]);
         const char *next_format = &format[spec_pos + 1];

         /* print the part before the format token */
         if (token != format)
            fwrite(format, token - format, 1, out);

         char *print_str = strndup(token, next_format - token);
         /* rebase spec_pos so we can use it with print_str */
         spec_pos += format - token;

         /* print the formated part */
         if (print_str[spec_pos] == 's') {
            uint64_t idx;
            memcpy(&idx, &buffer[buf_pos], 8);
            fprintf(out, print_str, &fmt->strings[idx]);

         /* Never pass a 'n' spec to the host printf */
         } else if (print_str[spec_pos] != 'n') {
            char *vec_pos = strchr(print_str, 'v');
            char *mod_pos = strpbrk(print_str, "hl");

            int component_count = 1;
            if (vec_pos != NULL) {
               /* non vector part of the format */
               size_t base = mod_pos ? mod_pos - print_str : spec_pos;
               size_t l = base - (vec_pos - print_str) - 1;
               char *vec = strndup(&vec_pos[1], l);
               component_count = atoi(vec);
               free(vec);

               /* remove the vector and precision stuff */
               memmove(&print_str[vec_pos - print_str], &print_str[spec_pos], 2);
            }

            /* in fact vec3 are vec4 */
            int men_components = component_count == 3 ? 4 : component_count;
            size_t elmt_size = arg_size / men_components;
            bool is_float = strpbrk(print_str, "fFeEgGaA") != NULL;

            for (int i = 0; i < component_count; i++) {
               size_t elmt_buf_pos = buf_pos + i * elmt_size;
               switch (elmt_size) {
               case 1: {
                  uint8_t v;
                  memcpy(&v, &buffer[elmt_buf_pos], elmt_size);
                  fprintf(out, print_str, v);
                  break;
               }
               case 2: {
                  uint16_t v;
                  memcpy(&v, &buffer[elmt_buf_pos], elmt_size);
                  fprintf(out, print_str, v);
                  break;
               }
               case 4: {
                  if (is_float) {
                     float v;
                     memcpy(&v, &buffer[elmt_buf_pos], elmt_size);
                     fprintf(out, print_str, v);
                  } else {
                     uint32_t v;
                     memcpy(&v, &buffer[elmt_buf_pos], elmt_size);
                     fprintf(out, print_str, v);
                  }
                  break;
               }
               case 8: {
                  if (is_float) {
                     double v;
                     memcpy(&v, &buffer[elmt_buf_pos], elmt_size);
                     fprintf(out, print_str, v);
                  } else {
                     uint64_t v;
                     memcpy(&v, &buffer[elmt_buf_pos], elmt_size);
                     fprintf(out, print_str, v);
                  }
                  break;
               }
               default:
                  assert(false);
                  break;
               }

               if (i < component_count - 1)
                  fprintf(out, ",");
            }
         }

         /* rebase format */
         format = next_format;
         free(print_str);

         buf_pos += arg_size;
         buf_pos = ALIGN(buf_pos, 4);
      }

      /* print remaining */
      fprintf(out, "%s", format);
   }
}

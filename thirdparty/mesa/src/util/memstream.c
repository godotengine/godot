/*
 * Copyright 2020 Lag Free Games, LLC
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
 */

#include "memstream.h"

#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#endif

bool
u_memstream_open(struct u_memstream *mem, char **bufp, size_t *sizep)
{
#ifdef _GAMING_XBOX
   int err = tmpfile_s(&mem->f);
   mem->bufp = bufp;
   mem->sizep = sizep;

   return err == 0;
#elif defined(_WIN32)
   bool success = false;

   char path[MAX_PATH];
   DWORD dwResult = GetTempPath(MAX_PATH, path);
   if ((dwResult > 0) && (dwResult < MAX_PATH)) {
      char *temp = mem->temp;
      UINT uResult = GetTempFileName(path, "MEMSTREAM", 0, temp);
      if (uResult != 0) {
         FILE *f = fopen(temp, "w+b");
         success = f != NULL;
         if (success)
         {
            mem->f = f;
            mem->bufp = bufp;
            mem->sizep = sizep;
         }
      }
   }

   return success;
#else
   FILE *const f = open_memstream(bufp, sizep);
   mem->f = f;
   return f != NULL;
#endif
}

void
u_memstream_close(struct u_memstream *mem)
{
   FILE *const f = mem->f;

#ifdef _WIN32
   long size = ftell(f);
   if (size > 0) {
      /* reserve space for the null terminator */
      char *buf = malloc(size + 1);
      fseek(f, 0, SEEK_SET);
      fread(buf, 1, size, f);
      /* insert null terminator */
      buf[size] = '\0';

      *mem->bufp = buf;
      *mem->sizep = size;
   }

   remove(mem->temp);
#endif

   fclose(f);
}

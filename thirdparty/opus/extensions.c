/* Copyright (c) 2022 Amazon */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include "opus_types.h"
#include "opus_defines.h"
#include "arch.h"
#include "os_support.h"
#include "opus_private.h"


/* Given an extension payload, advance data to the next extension and return the
   length of the remaining extensions. */
opus_int32 skip_extension(const unsigned char **data, opus_int32 len, opus_int32 *header_size)
{
   int id, L;
   if (len==0)
      return 0;
   id = **data>>1;
   L = **data&1;
   if (id == 0 && L == 1)
   {
      *header_size = 1;
      if (len < 1)
         return -1;
      (*data)++;
      len--;
      return len;
   } else if (id > 0 && id < 32)
   {
      if (len < 1+L)
         return -1;
      *data += 1+L;
      len -= 1+L;
      *header_size = 1;
      return len;
   } else {
      if (L==0)
      {
         *data += len;
         *header_size = 1;
         return 0;
      } else {
         opus_int32 bytes=0;
         *header_size = 1;
         do {
            (*data)++;
            len--;
            if (len == 0)
               return -1;
            bytes += **data;
            (*header_size)++;
         } while (**data == 255);
         (*data)++;
         len--;
         if (bytes <= len)
         {
            len -= bytes;
            *data += bytes;
         } else {
            return -1;
         }
         return len;
      }
   }
}

/* Count the number of extensions, excluding real padding and separators. */
opus_int32 opus_packet_extensions_count(const unsigned char *data, opus_int32 len)
{
   opus_int32 curr_len;
   opus_int32 count=0;
   const unsigned char *curr_data = data;

   celt_assert(len >= 0);
   celt_assert(data != NULL || len == 0);

   curr_len = len;
   while (curr_len > 0)
   {
      int id;
      opus_int32 header_size;
      id = *curr_data>>1;
      curr_len = skip_extension(&curr_data, curr_len, &header_size);
      if (curr_len < 0)
         return OPUS_INVALID_PACKET;
      if (id > 1)
         count++;
   }
   return count;
}

/* Extract extensions from Opus padding (excluding real padding and separators) */
opus_int32 opus_packet_extensions_parse(const unsigned char *data, opus_int32 len, opus_extension_data *extensions, opus_int32 *nb_extensions)
{
   const unsigned char *curr_data;
   opus_int32 curr_len;
   int curr_frame=0;
   opus_int32 count=0;

   celt_assert(len >= 0);
   celt_assert(data != NULL || len == 0);
   celt_assert(nb_extensions != NULL);
   celt_assert(extensions != NULL || *nb_extensions == 0);

   curr_data = data;
   curr_len = len;
   while (curr_len > 0)
   {
      int id;
      opus_int32 header_size;
      opus_extension_data curr_ext;
      id = *curr_data>>1;
      if (id > 1)
      {
         curr_ext.id = id;
         curr_ext.frame = curr_frame;
         curr_ext.data = curr_data;
      } else if (id == 1)
      {
         int L = *curr_data&1;
         if (L==0)
            curr_frame++;
         else {
            if (curr_len >= 2)
               curr_frame += curr_data[1];
            /* Else we're at the end and it doesn't matter. */
         }
         if (curr_frame >= 48)
         {
            *nb_extensions = count;
            return OPUS_INVALID_PACKET;
         }
      }
      curr_len = skip_extension(&curr_data, curr_len, &header_size);
      /* printf("curr_len = %d, header_size = %d\n", curr_len, header_size); */
      if (curr_len < 0)
      {
         *nb_extensions = count;
         return OPUS_INVALID_PACKET;
      }
      celt_assert(curr_data - data == len - curr_len);
      if (id > 1)
      {
         if (count == *nb_extensions)
         {
             return OPUS_BUFFER_TOO_SMALL;
         }
         curr_ext.len = curr_data - curr_ext.data - header_size;
         curr_ext.data += header_size;
         extensions[count++] = curr_ext;
      }
   }
   celt_assert(curr_len == 0);
   *nb_extensions = count;
   return OPUS_OK;
}

opus_int32 opus_packet_extensions_generate(unsigned char *data, opus_int32 len, const opus_extension_data  *extensions, opus_int32 nb_extensions, int pad)
{
   int max_frame=0;
   opus_int32 i;
   int frame;
   int curr_frame = 0;
   opus_int32 pos = 0;
   opus_int32 written = 0;

   celt_assert(len >= 0);

   for (i=0;i<nb_extensions;i++)
   {
      max_frame = IMAX(max_frame, extensions[i].frame);
      if (extensions[i].id < 2 || extensions[i].id > 127)
         return OPUS_BAD_ARG;
   }
   if (max_frame >= 48) return OPUS_BAD_ARG;
   for (frame=0;frame<=max_frame;frame++)
   {
      for (i=0;i<nb_extensions;i++)
      {
         if (extensions[i].frame == frame)
         {
            /* Insert separator when needed. */
            if (frame != curr_frame) {
               int diff = frame - curr_frame;
               if (len-pos < 2)
                  return OPUS_BUFFER_TOO_SMALL;
               if (diff == 1) {
                  if (data) data[pos] = 0x02;
                  pos++;
               } else {
                  if (data) data[pos] = 0x03;
                  pos++;
                  if (data) data[pos] = diff;
                  pos++;
               }
               curr_frame = frame;
            }
            if (extensions[i].id < 32)
            {
               if (extensions[i].len < 0 || extensions[i].len > 1)
                  return OPUS_BAD_ARG;
               if (len-pos < extensions[i].len+1)
                  return OPUS_BUFFER_TOO_SMALL;
               if (data) data[pos] = (extensions[i].id<<1) + extensions[i].len;
               pos++;
               if (extensions[i].len > 0) {
                  if (data) data[pos] = extensions[i].data[0];
                  pos++;
               }
            } else {
               int last;
               opus_int32 length_bytes;
               if (extensions[i].len < 0)
                  return OPUS_BAD_ARG;
               last = (written == nb_extensions - 1);
               length_bytes = 1 + extensions[i].len/255;
               if (last)
                  length_bytes = 0;
               if (len-pos < 1 + length_bytes + extensions[i].len)
                  return OPUS_BUFFER_TOO_SMALL;
               if (data) data[pos] = (extensions[i].id<<1) + !last;
               pos++;
               if (!last)
               {
                  opus_int32 j;
                  for (j=0;j<extensions[i].len/255;j++) {
                     if (data) data[pos] = 255;
                     pos++;
                  }
                  if (data) data[pos] = extensions[i].len % 255;
                  pos++;
               }
               if (data) OPUS_COPY(&data[pos], extensions[i].data, extensions[i].len);
               pos += extensions[i].len;
            }
            written++;
         }
      }
   }
   /* If we need to pad, just prepend 0x01 bytes. Even better would be to fill the
      end with zeros, but that requires checking that turning the last extesion into
      an L=1 case still fits. */
   if (pad && pos < len)
   {
      opus_int32 padding = len - pos;
      if (data) {
         OPUS_MOVE(data+padding, data, pos);
         for (i=0;i<padding;i++)
            data[i] = 0x01;
      }
      pos += padding;
   }
   return pos;
}

#if 0
#include <stdio.h>
int main()
{
   opus_extension_data ext[] = {{2, 0, (const unsigned char *)"a", 1},
   {32, 10, (const unsigned char *)"DRED", 4},
   {33, 1, (const unsigned char *)"NOT DRED", 8},
   {3, 4, (const unsigned char *)NULL, 0}
   };
   opus_extension_data ext2[10];
   int i, len;
   int nb_ext = 10;
   unsigned char packet[10000];
   len = opus_packet_extensions_generate(packet, 32, ext, 4, 1);
   for (i=0;i<len;i++)
   {
      printf("%#04x ", packet[i]);
      if (i%16 == 15)
         printf("\n");
   }
   printf("\n");
   printf("count = %d\n", opus_packet_extensions_count(packet, len));
   opus_packet_extensions_parse(packet, len, ext2, &nb_ext);
   for (i=0;i<nb_ext;i++)
   {
      int j;
      printf("%d %d {", ext2[i].id, ext2[i].frame);
      for (j=0;j<ext2[i].len;j++) printf("%#04x ", ext2[i].data[j]);
      printf("} %d\n", ext2[i].len);
   }
}
#endif

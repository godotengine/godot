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


/* Given an extension payload (i.e., excluding the initial ID byte), advance
    data to the next extension and return the length of the remaining
    extensions.
   N.B., a "Repeat These Extensions" extension (ID==2) does not advance past
    the repeated extension payloads.
   That requires higher-level logic. */
static opus_int32 skip_extension_payload(const unsigned char **pdata,
 opus_int32 len, opus_int32 *pheader_size, int id_byte,
 opus_int32 trailing_short_len)
{
   const unsigned char *data;
   opus_int32 header_size;
   int id, L;
   data = *pdata;
   header_size = 0;
   id = id_byte>>1;
   L = id_byte&1;
   if ((id == 0 && L == 1) || id == 2)
   {
      /* Nothing to do. */
   } else if (id > 0 && id < 32)
   {
      if (len < L)
         return -1;
      data += L;
      len -= L;
   } else {
      if (L==0)
      {
         if (len < trailing_short_len) return -1;
         data += len - trailing_short_len;
         len = trailing_short_len;
      } else {
         opus_int32 bytes=0;
         opus_int32 lacing;
         do {
            if (len < 1)
               return -1;
            lacing = *data++;
            bytes += lacing;
            header_size++;
            len -= lacing + 1;
         } while (lacing == 255);
         if (len < 0)
            return -1;
         data += bytes;
      }
   }
   *pdata = data;
   *pheader_size = header_size;
   return len;
}

/* Given an extension, advance data to the next extension and return the
   length of the remaining extensions.
   N.B., a "Repeat These Extensions" extension (ID==2) only advances past the
    extension ID byte.
   Higher-level logic is required to skip the extension payloads that come
    after it.*/
static opus_int32 skip_extension(const unsigned char **pdata, opus_int32 len,
 opus_int32 *pheader_size)
{
   const unsigned char *data;
   int id_byte;
   if (len == 0) {
      *pheader_size = 0;
      return 0;
   }
   if (len < 1)
      return -1;
   data = *pdata;
   id_byte = *data++;
   len--;
   len = skip_extension_payload(&data, len, pheader_size, id_byte, 0);
   if (len >= 0) {
      *pdata = data;
      (*pheader_size)++;
   }
   return len;
}

void opus_extension_iterator_init(OpusExtensionIterator *iter,
 const unsigned char *data, opus_int32 len, opus_int32 nb_frames) {
   celt_assert(len >= 0);
   celt_assert(data != NULL || len == 0);
   celt_assert(nb_frames >= 0 && nb_frames <= 48);
   iter->repeat_data = iter->curr_data = iter->data = data;
   iter->last_long = iter->src_data = NULL;
   iter->curr_len = iter->len = len;
   iter->repeat_len = iter->src_len = 0;
   iter->trailing_short_len = 0;
   iter->frame_max = iter->nb_frames = nb_frames;
   iter->repeat_frame = iter->curr_frame = 0;
   iter->repeat_l = 0;
}

/* Reset the iterator so it can start iterating again from the first
    extension. */
void opus_extension_iterator_reset(OpusExtensionIterator *iter) {
   iter->repeat_data = iter->curr_data = iter->data;
   iter->last_long = NULL;
   iter->curr_len = iter->len;
   iter->repeat_frame = iter->curr_frame = 0;
   iter->trailing_short_len = 0;
}

/* Tell the iterator not to return any extensions for frames of index
    frame_max or larger.
   This can allow it to stop iterating early if these extensions are not
    needed. */
void opus_extension_iterator_set_frame_max(OpusExtensionIterator *iter,
 int frame_max) {
   iter->frame_max = frame_max;
}

/* Return the next repeated extension.
   The return value is non-zero if one is found, negative on error, or 0 if we
    have finished repeating extensions. */
static int opus_extension_iterator_next_repeat(OpusExtensionIterator *iter,
 opus_extension_data *ext) {
   opus_int32 header_size;
   celt_assert(iter->repeat_frame > 0);
   for (;iter->repeat_frame < iter->nb_frames; iter->repeat_frame++) {
      while (iter->src_len > 0) {
         const unsigned char *curr_data0;
         int repeat_id_byte;
         repeat_id_byte = *iter->src_data;
         iter->src_len = skip_extension(&iter->src_data, iter->src_len,
          &header_size);
         /* We skipped this extension earlier, so it should not fail now. */
         celt_assert(iter->src_len >= 0);
         /* Don't repeat padding or frame separators with a 0 increment. */
         if (repeat_id_byte <= 3) continue;
         /* If the "Repeat These Extensions" extension had L == 0 and this
             is the last repeated long extension, then force decoding the
             payload with L = 0. */
         if (iter->repeat_l == 0
          && iter->repeat_frame + 1 >= iter->nb_frames
          && iter->src_data == iter->last_long) {
            repeat_id_byte &= ~1;
         }
         curr_data0 = iter->curr_data;
         iter->curr_len = skip_extension_payload(&iter->curr_data,
          iter->curr_len, &header_size, repeat_id_byte,
          iter->trailing_short_len);
         if (iter->curr_len < 0) {
            return OPUS_INVALID_PACKET;
         }
         celt_assert(iter->curr_data - iter->data
          == iter->len - iter->curr_len);
         /* If we were asked to stop at frame_max, skip extensions for later
             frames. */
         if (iter->repeat_frame >= iter->frame_max) {
            continue;
         }
         if (ext != NULL) {
            ext->id = repeat_id_byte >> 1;
            ext->frame = iter->repeat_frame;
            ext->data = curr_data0 + header_size;
            ext->len = iter->curr_data - curr_data0 - header_size;
         }
         return 1;
      }
      /* We finished repeating the extensions for this frame. */
      iter->src_data = iter->repeat_data;
      iter->src_len = iter->repeat_len;
   }
   /* We finished repeating extensions. */
   iter->repeat_data = iter->curr_data;
   iter->last_long = NULL;
   /* If L == 0, advance the frame number to handle the case where we did
       not consume all of the data with an L == 0 long extension. */
   if (iter->repeat_l == 0) {
      iter->curr_frame++;
      /* Ignore additional padding if this was already the last frame. */
      if (iter->curr_frame >= iter->nb_frames) {
         iter->curr_len = 0;
      }
   }
   iter->repeat_frame = 0;
   return 0;
}

/* Return the next extension (excluding real padding, separators, and repeat
    indicators, but including the repeated extensions) in bitstream order.
   Due to the extension repetition mechanism, extensions are not necessarily
    returned in frame order. */
int opus_extension_iterator_next(OpusExtensionIterator *iter,
 opus_extension_data *ext) {
   opus_int32 header_size;
   if (iter->curr_len < 0) {
      return OPUS_INVALID_PACKET;
   }
   if (iter->repeat_frame > 0) {
      int ret;
      /* We are in the process of repeating some extensions. */
      ret = opus_extension_iterator_next_repeat(iter, ext);
      if (ret) return ret;
   }
   /* Checking this here allows opus_extension_iterator_set_frame_max() to be
       called at any point. */
   if (iter->curr_frame >= iter->frame_max) {
      return 0;
   }
   while (iter->curr_len > 0) {
      const unsigned char *curr_data0;
      int id;
      int L;
      curr_data0 = iter->curr_data;
      id = *curr_data0>>1;
      L = *curr_data0&1;
      iter->curr_len = skip_extension(&iter->curr_data, iter->curr_len,
       &header_size);
      if (iter->curr_len < 0) {
         return OPUS_INVALID_PACKET;
      }
      celt_assert(iter->curr_data - iter->data == iter->len - iter->curr_len);
      if (id == 1) {
         if (L == 0) {
            iter->curr_frame++;
         }
         else {
            /* A frame increment of 0 is a no-op. */
            if (!curr_data0[1]) continue;
            iter->curr_frame += curr_data0[1];
         }
         if (iter->curr_frame >= iter->nb_frames) {
            iter->curr_len = -1;
            return OPUS_INVALID_PACKET;
         }
         /* If we were asked to stop at frame_max, skip extensions for later
             frames. */
         if (iter->curr_frame >= iter->frame_max) {
            iter->curr_len = 0;
         }
         iter->repeat_data = iter->curr_data;
         iter->last_long = NULL;
         iter->trailing_short_len = 0;
      }
      else if (id == 2) {
         int ret;
         iter->repeat_l = L;
         iter->repeat_frame = iter->curr_frame + 1;
         iter->repeat_len = curr_data0 - iter->repeat_data;
         iter->src_data = iter->repeat_data;
         iter->src_len = iter->repeat_len;
         ret = opus_extension_iterator_next_repeat(iter, ext);
         if (ret) return ret;
      }
      else if (id > 2) {
         /* Update the location of the last long extension.
            This lets us know when we need to modify the last L flag if we
             repeat these extensions with L=0. */
         if (id >= 32) {
           iter->last_long = iter->curr_data;
           iter->trailing_short_len = 0;
         }
         /* Otherwise, keep track of how many payload bytes follow the last
             long extension. */
         else iter->trailing_short_len += L;
         if (ext != NULL) {
            ext->id = id;
            ext->frame = iter->curr_frame;
            ext->data = curr_data0 + header_size;
            ext->len = iter->curr_data - curr_data0 - header_size;
         }
         return 1;
      }
   }
   return 0;
}

int opus_extension_iterator_find(OpusExtensionIterator *iter,
 opus_extension_data *ext, int id) {
   opus_extension_data curr_ext;
   int ret;
   for(;;) {
      ret = opus_extension_iterator_next(iter, &curr_ext);
      if (ret <= 0) {
         return ret;
      }
      if (curr_ext.id == id) {
         *ext = curr_ext;
         return ret;
      }
   }
}

/* Count the number of extensions, excluding real padding, separators, and
    repeat indicators, but including the repeated extensions. */
opus_int32 opus_packet_extensions_count(const unsigned char *data,
 opus_int32 len, int nb_frames)
{
   OpusExtensionIterator iter;
   int count;
   opus_extension_iterator_init(&iter, data, len, nb_frames);
   for (count=0; opus_extension_iterator_next(&iter, NULL) > 0; count++);
   return count;
}

/* Count the number of extensions for each frame, excluding real padding and
    separators and repeat indicators, but including the repeated extensions. */
opus_int32 opus_packet_extensions_count_ext(const unsigned char *data,
 opus_int32 len, opus_int32 *nb_frame_exts, int nb_frames) {
   OpusExtensionIterator iter;
   opus_extension_data ext;
   int count;
   opus_extension_iterator_init(&iter, data, len, nb_frames);
   OPUS_CLEAR(nb_frame_exts, nb_frames);
   for (count=0; opus_extension_iterator_next(&iter, &ext) > 0; count++) {
      nb_frame_exts[ext.frame]++;
   }
   return count;
}

/* Extract extensions from Opus padding (excluding real padding, separators,
    and repeat indicators, but including the repeated extensions) in bitstream
    order.
   Due to the extension repetition mechanism, extensions are not necessarily
    returned in frame order. */
opus_int32 opus_packet_extensions_parse(const unsigned char *data,
 opus_int32 len, opus_extension_data *extensions, opus_int32 *nb_extensions,
 int nb_frames) {
   OpusExtensionIterator iter;
   int count;
   int ret;
   celt_assert(nb_extensions != NULL);
   celt_assert(extensions != NULL || *nb_extensions == 0);
   opus_extension_iterator_init(&iter, data, len, nb_frames);
   for (count=0;; count++) {
      opus_extension_data ext;
      ret = opus_extension_iterator_next(&iter, &ext);
      if (ret <= 0) break;
      if (count == *nb_extensions) {
         return OPUS_BUFFER_TOO_SMALL;
      }
      extensions[count] = ext;
   }
   *nb_extensions = count;
   return ret;
}

/* Extract extensions from Opus padding (excluding real padding, separators,
    and repeat indicators, but including the repeated extensions) in frame
    order.
   nb_frame_exts must be filled with the output of
    opus_packet_extensions_count_ext(). */
opus_int32 opus_packet_extensions_parse_ext(const unsigned char *data,
 opus_int32 len, opus_extension_data *extensions, opus_int32 *nb_extensions,
 const opus_int32 *nb_frame_exts, int nb_frames) {
   OpusExtensionIterator iter;
   opus_extension_data ext;
   opus_int32 nb_frames_cum[49];
   int count;
   int prev_total;
   int ret;
   celt_assert(nb_extensions != NULL);
   celt_assert(extensions != NULL || *nb_extensions == 0);
   celt_assert(nb_frames <= 48);
   /* Convert the frame extension count array to a cumulative sum. */
   prev_total = 0;
   for (count=0; count<nb_frames; count++) {
      int total;
      total = nb_frame_exts[count] + prev_total;
      nb_frames_cum[count] = prev_total;
      prev_total = total;
   }
   nb_frames_cum[count] = prev_total;
   opus_extension_iterator_init(&iter, data, len, nb_frames);
   for (count=0;; count++) {
      opus_int32 idx;
      ret = opus_extension_iterator_next(&iter, &ext);
      if (ret <= 0) break;
      idx = nb_frames_cum[ext.frame]++;
      if (idx >= *nb_extensions) {
         return OPUS_BUFFER_TOO_SMALL;
      }
      celt_assert(idx < nb_frames_cum[ext.frame + 1]);
      extensions[idx] = ext;
   }
   *nb_extensions = count;
   return ret;
}

static int write_extension_payload(unsigned char *data, opus_int32 len,
 opus_int32 pos, const opus_extension_data *ext, int last) {
   celt_assert(ext->id >= 3 && ext->id <= 127);
   if (ext->id < 32)
   {
      if (ext->len < 0 || ext->len > 1)
         return OPUS_BAD_ARG;
      if (ext->len > 0) {
         if (len-pos < ext->len)
            return OPUS_BUFFER_TOO_SMALL;
         if (data) data[pos] = ext->data[0];
         pos++;
      }
   } else {
      opus_int32 length_bytes;
      if (ext->len < 0)
         return OPUS_BAD_ARG;
      length_bytes = 1 + ext->len/255;
      if (last)
         length_bytes = 0;
      if (len-pos < length_bytes + ext->len)
         return OPUS_BUFFER_TOO_SMALL;
      if (!last)
      {
         opus_int32 j;
         for (j=0;j<ext->len/255;j++) {
            if (data) data[pos] = 255;
            pos++;
         }
         if (data) data[pos] = ext->len % 255;
         pos++;
      }
      if (data) OPUS_COPY(&data[pos], ext->data, ext->len);
      pos += ext->len;
   }
   return pos;
}

static int write_extension(unsigned char *data, opus_int32 len, opus_int32 pos,
 const opus_extension_data *ext, int last) {
   if (len-pos < 1)
      return OPUS_BUFFER_TOO_SMALL;
   celt_assert(ext->id >= 3 && ext->id <= 127);
   if (data) data[pos] = (ext->id<<1) + (ext->id < 32 ? ext->len : !last);
   pos++;
   return write_extension_payload(data, len, pos, ext, last);
}

opus_int32 opus_packet_extensions_generate(unsigned char *data, opus_int32 len,
 const opus_extension_data  *extensions, opus_int32 nb_extensions,
 int nb_frames, int pad)
{
   opus_int32 frame_min_idx[48];
   opus_int32 frame_max_idx[48];
   opus_int32 frame_repeat_idx[48];
   opus_int32 i;
   int f;
   int curr_frame = 0;
   opus_int32 pos = 0;
   opus_int32 written = 0;

   celt_assert(len >= 0);
   if (nb_frames > 48) return OPUS_BAD_ARG;

   /* Do a little work up-front to make this O(nb_extensions) instead of
       O(nb_extensions*nb_frames) so long as the extensions are in frame
       order (without requiring that they be in frame order). */
   for (f=0;f<nb_frames;f++) frame_min_idx[f] = nb_extensions;
   OPUS_CLEAR(frame_max_idx, nb_frames);
   for (i=0;i<nb_extensions;i++)
   {
      f = extensions[i].frame;
      if (f < 0 || f >= nb_frames) return OPUS_BAD_ARG;
      if (extensions[i].id < 3 || extensions[i].id > 127) return OPUS_BAD_ARG;
      frame_min_idx[f] = IMIN(frame_min_idx[f], i);
      frame_max_idx[f] = IMAX(frame_max_idx[f], i+1);
   }
   for (f=0;f<nb_frames;f++) frame_repeat_idx[f] = frame_min_idx[f];
   for (f=0;f<nb_frames;f++)
   {
      opus_int32 last_long_idx;
      int repeat_count;
      repeat_count = 0;
      last_long_idx = -1;
      if (f + 1 < nb_frames)
      {
         for (i=frame_min_idx[f];i<frame_max_idx[f];i++)
         {
            if (extensions[i].frame == f)
            {
               int g;
               /* Test if we can repeat this extension in future frames. */
               for (g=f+1;g<nb_frames;g++)
               {
                  if (frame_repeat_idx[g] >= frame_max_idx[g]) break;
                  celt_assert(extensions[frame_repeat_idx[g]].frame == g);
                  if (extensions[frame_repeat_idx[g]].id != extensions[i].id)
                  {
                     break;
                  }
                  if (extensions[frame_repeat_idx[g]].id < 32
                    && extensions[frame_repeat_idx[g]].len
                    != extensions[i].len)
                  {
                     break;
                  }
               }
               if (g < nb_frames) break;
               /* We can! */
               /* If this is a long extension, save the index of the last
                   instance, so we can modify its L flag. */
               if (extensions[i].id >= 32) {
                  last_long_idx = frame_repeat_idx[nb_frames-1];
               }
               /* Using the repeat mechanism almost always makes the
                   encoding smaller (or at least no larger).
                  However, there's one case where that might not be true: if
                   the last repeated long extension in the last frame was
                   previously the last extension, but using the repeat
                   mechanism makes that no longer true (because there are other
                   non-repeated extensions in earlier frames that must now be
                   coded after it), and coding its length requires more bytes
                   than the repeat mechanism saves.
                  This can only be true if its length is at least 255 bytes
                   (although sometimes it requires even more).
                  Currently we do not check for that, and just always use the
                   repeat mechanism if we can.
                  See git history for code that does the check. */
               /* Advance the repeat pointers. */
               for (g=f+1; g<nb_frames; g++)
               {
                  int j;
                  for (j=frame_repeat_idx[g]+1; j<frame_max_idx[g]
                   && extensions[j].frame != g; j++);
                  frame_repeat_idx[g] = j;
               }
               repeat_count++;
               /* Point the repeat pointer for this frame to the current
                   extension, so we know when to trigger the repeats. */
               frame_repeat_idx[f] = i;
            }
         }
      }
      for (i=frame_min_idx[f];i<frame_max_idx[f];i++)
      {
         if (extensions[i].frame == f)
         {
            /* Insert separator when needed. */
            if (f != curr_frame) {
               int diff = f - curr_frame;
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
               curr_frame = f;
            }

            pos = write_extension(data, len, pos, extensions + i,
             written == nb_extensions - 1);
            if (pos < 0) return pos;
            written++;

            if (repeat_count > 0 && frame_repeat_idx[f] == i) {
               int nb_repeated;
               int last;
               int g;
               /* Add the repeat indicator. */
               nb_repeated = repeat_count*(nb_frames - (f + 1));
               last = written + nb_repeated == nb_extensions
                || (last_long_idx < 0 && i+1 >= frame_max_idx[f]);
               if (len-pos < 1)
                  return OPUS_BUFFER_TOO_SMALL;
               if (data) data[pos] = 0x04 + !last;
               pos++;
               for (g=f+1;g<nb_frames;g++)
               {
                  int j;
                  for (j=frame_min_idx[g];j<frame_repeat_idx[g];j++)
                  {
                     if (extensions[j].frame == g)
                     {
                        pos = write_extension_payload(data, len, pos,
                         extensions + j, last && j == last_long_idx);
                        if (pos < 0) return pos;
                        written++;
                     }
                  }
                  frame_min_idx[g] = j;
               }
               if (last) curr_frame++;
            }
         }
      }
   }
   celt_assert(written == nb_extensions);
   /* If we need to pad, just prepend 0x01 bytes. Even better would be to fill the
      end with zeros, but that requires checking that turning the last extension into
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

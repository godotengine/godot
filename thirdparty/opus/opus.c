/* Copyright (c) 2011 Xiph.Org Foundation, Skype Limited
   Written by Jean-Marc Valin and Koen Vos */
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

#include "opus.h"
#include "opus_private.h"

#ifndef DISABLE_FLOAT_API
OPUS_EXPORT void opus_pcm_soft_clip(float *_x, int N, int C, float *declip_mem)
{
   int c;
   int i;
   float *x;

   if (C<1 || N<1 || !_x || !declip_mem) return;

   /* First thing: saturate everything to +/- 2 which is the highest level our
      non-linearity can handle. At the point where the signal reaches +/-2,
      the derivative will be zero anyway, so this doesn't introduce any
      discontinuity in the derivative. */
   for (i=0;i<N*C;i++)
      _x[i] = MAX16(-2.f, MIN16(2.f, _x[i]));
   for (c=0;c<C;c++)
   {
      float a;
      float x0;
      int curr;

      x = _x+c;
      a = declip_mem[c];
      /* Continue applying the non-linearity from the previous frame to avoid
         any discontinuity. */
      for (i=0;i<N;i++)
      {
         if (x[i*C]*a>=0)
            break;
         x[i*C] = x[i*C]+a*x[i*C]*x[i*C];
      }

      curr=0;
      x0 = x[0];
      while(1)
      {
         int start, end;
         float maxval;
         int special=0;
         int peak_pos;
         for (i=curr;i<N;i++)
         {
            if (x[i*C]>1 || x[i*C]<-1)
               break;
         }
         if (i==N)
         {
            a=0;
            break;
         }
         peak_pos = i;
         start=end=i;
         maxval=ABS16(x[i*C]);
         /* Look for first zero crossing before clipping */
         while (start>0 && x[i*C]*x[(start-1)*C]>=0)
            start--;
         /* Look for first zero crossing after clipping */
         while (end<N && x[i*C]*x[end*C]>=0)
         {
            /* Look for other peaks until the next zero-crossing. */
            if (ABS16(x[end*C])>maxval)
            {
               maxval = ABS16(x[end*C]);
               peak_pos = end;
            }
            end++;
         }
         /* Detect the special case where we clip before the first zero crossing */
         special = (start==0 && x[i*C]*x[0]>=0);

         /* Compute a such that maxval + a*maxval^2 = 1 */
         a=(maxval-1)/(maxval*maxval);
         /* Slightly boost "a" by 2^-22. This is just enough to ensure -ffast-math
            does not cause output values larger than +/-1, but small enough not
            to matter even for 24-bit output.  */
         a += a*2.4e-7f;
         if (x[i*C]>0)
            a = -a;
         /* Apply soft clipping */
         for (i=start;i<end;i++)
            x[i*C] = x[i*C]+a*x[i*C]*x[i*C];

         if (special && peak_pos>=2)
         {
            /* Add a linear ramp from the first sample to the signal peak.
               This avoids a discontinuity at the beginning of the frame. */
            float delta;
            float offset = x0-x[0];
            delta = offset / peak_pos;
            for (i=curr;i<peak_pos;i++)
            {
               offset -= delta;
               x[i*C] += offset;
               x[i*C] = MAX16(-1.f, MIN16(1.f, x[i*C]));
            }
         }
         curr = end;
         if (curr==N)
            break;
      }
      declip_mem[c] = a;
   }
}
#endif

int encode_size(int size, unsigned char *data)
{
   if (size < 252)
   {
      data[0] = size;
      return 1;
   } else {
      data[0] = 252+(size&0x3);
      data[1] = (size-(int)data[0])>>2;
      return 2;
   }
}

static int parse_size(const unsigned char *data, opus_int32 len, opus_int16 *size)
{
   if (len<1)
   {
      *size = -1;
      return -1;
   } else if (data[0]<252)
   {
      *size = data[0];
      return 1;
   } else if (len<2)
   {
      *size = -1;
      return -1;
   } else {
      *size = 4*data[1] + data[0];
      return 2;
   }
}

int opus_packet_get_samples_per_frame(const unsigned char *data,
      opus_int32 Fs)
{
   int audiosize;
   if (data[0]&0x80)
   {
      audiosize = ((data[0]>>3)&0x3);
      audiosize = (Fs<<audiosize)/400;
   } else if ((data[0]&0x60) == 0x60)
   {
      audiosize = (data[0]&0x08) ? Fs/50 : Fs/100;
   } else {
      audiosize = ((data[0]>>3)&0x3);
      if (audiosize == 3)
         audiosize = Fs*60/1000;
      else
         audiosize = (Fs<<audiosize)/100;
   }
   return audiosize;
}

int opus_packet_parse_impl(const unsigned char *data, opus_int32 len,
      int self_delimited, unsigned char *out_toc,
      const unsigned char *frames[48], opus_int16 size[48],
      int *payload_offset, opus_int32 *packet_offset)
{
   int i, bytes;
   int count;
   int cbr;
   unsigned char ch, toc;
   int framesize;
   opus_int32 last_size;
   opus_int32 pad = 0;
   const unsigned char *data0 = data;

   if (size==NULL || len<0)
      return OPUS_BAD_ARG;
   if (len==0)
      return OPUS_INVALID_PACKET;

   framesize = opus_packet_get_samples_per_frame(data, 48000);

   cbr = 0;
   toc = *data++;
   len--;
   last_size = len;
   switch (toc&0x3)
   {
   /* One frame */
   case 0:
      count=1;
      break;
   /* Two CBR frames */
   case 1:
      count=2;
      cbr = 1;
      if (!self_delimited)
      {
         if (len&0x1)
            return OPUS_INVALID_PACKET;
         last_size = len/2;
         /* If last_size doesn't fit in size[0], we'll catch it later */
         size[0] = (opus_int16)last_size;
      }
      break;
   /* Two VBR frames */
   case 2:
      count = 2;
      bytes = parse_size(data, len, size);
      len -= bytes;
      if (size[0]<0 || size[0] > len)
         return OPUS_INVALID_PACKET;
      data += bytes;
      last_size = len-size[0];
      break;
   /* Multiple CBR/VBR frames (from 0 to 120 ms) */
   default: /*case 3:*/
      if (len<1)
         return OPUS_INVALID_PACKET;
      /* Number of frames encoded in bits 0 to 5 */
      ch = *data++;
      count = ch&0x3F;
      if (count <= 0 || framesize*(opus_int32)count > 5760)
         return OPUS_INVALID_PACKET;
      len--;
      /* Padding flag is bit 6 */
      if (ch&0x40)
      {
         int p;
         do {
            int tmp;
            if (len<=0)
               return OPUS_INVALID_PACKET;
            p = *data++;
            len--;
            tmp = p==255 ? 254: p;
            len -= tmp;
            pad += tmp;
         } while (p==255);
      }
      if (len<0)
         return OPUS_INVALID_PACKET;
      /* VBR flag is bit 7 */
      cbr = !(ch&0x80);
      if (!cbr)
      {
         /* VBR case */
         last_size = len;
         for (i=0;i<count-1;i++)
         {
            bytes = parse_size(data, len, size+i);
            len -= bytes;
            if (size[i]<0 || size[i] > len)
               return OPUS_INVALID_PACKET;
            data += bytes;
            last_size -= bytes+size[i];
         }
         if (last_size<0)
            return OPUS_INVALID_PACKET;
      } else if (!self_delimited)
      {
         /* CBR case */
         last_size = len/count;
         if (last_size*count!=len)
            return OPUS_INVALID_PACKET;
         for (i=0;i<count-1;i++)
            size[i] = (opus_int16)last_size;
      }
      break;
   }
   /* Self-delimited framing has an extra size for the last frame. */
   if (self_delimited)
   {
      bytes = parse_size(data, len, size+count-1);
      len -= bytes;
      if (size[count-1]<0 || size[count-1] > len)
         return OPUS_INVALID_PACKET;
      data += bytes;
      /* For CBR packets, apply the size to all the frames. */
      if (cbr)
      {
         if (size[count-1]*count > len)
            return OPUS_INVALID_PACKET;
         for (i=0;i<count-1;i++)
            size[i] = size[count-1];
      } else if (bytes+size[count-1] > last_size)
         return OPUS_INVALID_PACKET;
   } else
   {
      /* Because it's not encoded explicitly, it's possible the size of the
         last packet (or all the packets, for the CBR case) is larger than
         1275. Reject them here.*/
      if (last_size > 1275)
         return OPUS_INVALID_PACKET;
      size[count-1] = (opus_int16)last_size;
   }

   if (payload_offset)
      *payload_offset = (int)(data-data0);

   for (i=0;i<count;i++)
   {
      if (frames)
         frames[i] = data;
      data += size[i];
   }

   if (packet_offset)
      *packet_offset = pad+(opus_int32)(data-data0);

   if (out_toc)
      *out_toc = toc;

   return count;
}

int opus_packet_parse(const unsigned char *data, opus_int32 len,
      unsigned char *out_toc, const unsigned char *frames[48],
      opus_int16 size[48], int *payload_offset)
{
   return opus_packet_parse_impl(data, len, 0, out_toc,
                                 frames, size, payload_offset, NULL);
}


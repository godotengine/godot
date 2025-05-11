/* Copyright (c) 2011 Xiph.Org Foundation
   Written by Jean-Marc Valin */
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

#include "opus_multistream.h"
#include "opus.h"
#include "opus_private.h"
#include "stack_alloc.h"
#include <stdarg.h>
#include "float_cast.h"
#include "os_support.h"

/* DECODER */

#if defined(ENABLE_HARDENING) || defined(ENABLE_ASSERTIONS)
static void validate_ms_decoder(OpusMSDecoder *st)
{
   validate_layout(&st->layout);
}
#define VALIDATE_MS_DECODER(st) validate_ms_decoder(st)
#else
#define VALIDATE_MS_DECODER(st)
#endif


opus_int32 opus_multistream_decoder_get_size(int nb_streams, int nb_coupled_streams)
{
   int coupled_size;
   int mono_size;

   if(nb_streams<1||nb_coupled_streams>nb_streams||nb_coupled_streams<0)return 0;
   coupled_size = opus_decoder_get_size(2);
   mono_size = opus_decoder_get_size(1);
   return align(sizeof(OpusMSDecoder))
         + nb_coupled_streams * align(coupled_size)
         + (nb_streams-nb_coupled_streams) * align(mono_size);
}

int opus_multistream_decoder_init(
      OpusMSDecoder *st,
      opus_int32 Fs,
      int channels,
      int streams,
      int coupled_streams,
      const unsigned char *mapping
)
{
   int coupled_size;
   int mono_size;
   int i, ret;
   char *ptr;

   if ((channels>255) || (channels<1) || (coupled_streams>streams) ||
       (streams<1) || (coupled_streams<0) || (streams>255-coupled_streams))
      return OPUS_BAD_ARG;

   st->layout.nb_channels = channels;
   st->layout.nb_streams = streams;
   st->layout.nb_coupled_streams = coupled_streams;

   for (i=0;i<st->layout.nb_channels;i++)
      st->layout.mapping[i] = mapping[i];
   if (!validate_layout(&st->layout))
      return OPUS_BAD_ARG;

   ptr = (char*)st + align(sizeof(OpusMSDecoder));
   coupled_size = opus_decoder_get_size(2);
   mono_size = opus_decoder_get_size(1);

   for (i=0;i<st->layout.nb_coupled_streams;i++)
   {
      ret=opus_decoder_init((OpusDecoder*)ptr, Fs, 2);
      if(ret!=OPUS_OK)return ret;
      ptr += align(coupled_size);
   }
   for (;i<st->layout.nb_streams;i++)
   {
      ret=opus_decoder_init((OpusDecoder*)ptr, Fs, 1);
      if(ret!=OPUS_OK)return ret;
      ptr += align(mono_size);
   }
   return OPUS_OK;
}


OpusMSDecoder *opus_multistream_decoder_create(
      opus_int32 Fs,
      int channels,
      int streams,
      int coupled_streams,
      const unsigned char *mapping,
      int *error
)
{
   int ret;
   OpusMSDecoder *st;
   if ((channels>255) || (channels<1) || (coupled_streams>streams) ||
       (streams<1) || (coupled_streams<0) || (streams>255-coupled_streams))
   {
      if (error)
         *error = OPUS_BAD_ARG;
      return NULL;
   }
   st = (OpusMSDecoder *)opus_alloc(opus_multistream_decoder_get_size(streams, coupled_streams));
   if (st==NULL)
   {
      if (error)
         *error = OPUS_ALLOC_FAIL;
      return NULL;
   }
   ret = opus_multistream_decoder_init(st, Fs, channels, streams, coupled_streams, mapping);
   if (error)
      *error = ret;
   if (ret != OPUS_OK)
   {
      opus_free(st);
      st = NULL;
   }
   return st;
}

static int opus_multistream_packet_validate(const unsigned char *data,
      opus_int32 len, int nb_streams, opus_int32 Fs)
{
   int s;
   int count;
   unsigned char toc;
   opus_int16 size[48];
   int samples=0;
   opus_int32 packet_offset;

   for (s=0;s<nb_streams;s++)
   {
      int tmp_samples;
      if (len<=0)
         return OPUS_INVALID_PACKET;
      count = opus_packet_parse_impl(data, len, s!=nb_streams-1, &toc, NULL,
                                     size, NULL, &packet_offset, NULL, NULL);
      if (count<0)
         return count;
      tmp_samples = opus_packet_get_nb_samples(data, packet_offset, Fs);
      if (s!=0 && samples != tmp_samples)
         return OPUS_INVALID_PACKET;
      samples = tmp_samples;
      data += packet_offset;
      len -= packet_offset;
   }
   return samples;
}

int opus_multistream_decode_native(
      OpusMSDecoder *st,
      const unsigned char *data,
      opus_int32 len,
      void *pcm,
      opus_copy_channel_out_func copy_channel_out,
      int frame_size,
      int decode_fec,
      int soft_clip,
      void *user_data
)
{
   opus_int32 Fs;
   int coupled_size;
   int mono_size;
   int s, c;
   char *ptr;
   int do_plc=0;
   VARDECL(opus_val16, buf);
   ALLOC_STACK;

   VALIDATE_MS_DECODER(st);
   if (frame_size <= 0)
   {
      RESTORE_STACK;
      return OPUS_BAD_ARG;
   }
   /* Limit frame_size to avoid excessive stack allocations. */
   MUST_SUCCEED(opus_multistream_decoder_ctl(st, OPUS_GET_SAMPLE_RATE(&Fs)));
   frame_size = IMIN(frame_size, Fs/25*3);
   ALLOC(buf, 2*frame_size, opus_val16);
   ptr = (char*)st + align(sizeof(OpusMSDecoder));
   coupled_size = opus_decoder_get_size(2);
   mono_size = opus_decoder_get_size(1);

   if (len==0)
      do_plc = 1;
   if (len < 0)
   {
      RESTORE_STACK;
      return OPUS_BAD_ARG;
   }
   if (!do_plc && len < 2*st->layout.nb_streams-1)
   {
      RESTORE_STACK;
      return OPUS_INVALID_PACKET;
   }
   if (!do_plc)
   {
      int ret = opus_multistream_packet_validate(data, len, st->layout.nb_streams, Fs);
      if (ret < 0)
      {
         RESTORE_STACK;
         return ret;
      } else if (ret > frame_size)
      {
         RESTORE_STACK;
         return OPUS_BUFFER_TOO_SMALL;
      }
   }
   for (s=0;s<st->layout.nb_streams;s++)
   {
      OpusDecoder *dec;
      opus_int32 packet_offset;
      int ret;

      dec = (OpusDecoder*)ptr;
      ptr += (s < st->layout.nb_coupled_streams) ? align(coupled_size) : align(mono_size);

      if (!do_plc && len<=0)
      {
         RESTORE_STACK;
         return OPUS_INTERNAL_ERROR;
      }
      packet_offset = 0;
      ret = opus_decode_native(dec, data, len, buf, frame_size, decode_fec, s!=st->layout.nb_streams-1, &packet_offset, soft_clip, NULL, 0);
      if (!do_plc)
      {
        data += packet_offset;
        len -= packet_offset;
      }
      if (ret <= 0)
      {
         RESTORE_STACK;
         return ret;
      }
      frame_size = ret;
      if (s < st->layout.nb_coupled_streams)
      {
         int chan, prev;
         prev = -1;
         /* Copy "left" audio to the channel(s) where it belongs */
         while ( (chan = get_left_channel(&st->layout, s, prev)) != -1)
         {
            (*copy_channel_out)(pcm, st->layout.nb_channels, chan,
               buf, 2, frame_size, user_data);
            prev = chan;
         }
         prev = -1;
         /* Copy "right" audio to the channel(s) where it belongs */
         while ( (chan = get_right_channel(&st->layout, s, prev)) != -1)
         {
            (*copy_channel_out)(pcm, st->layout.nb_channels, chan,
               buf+1, 2, frame_size, user_data);
            prev = chan;
         }
      } else {
         int chan, prev;
         prev = -1;
         /* Copy audio to the channel(s) where it belongs */
         while ( (chan = get_mono_channel(&st->layout, s, prev)) != -1)
         {
            (*copy_channel_out)(pcm, st->layout.nb_channels, chan,
               buf, 1, frame_size, user_data);
            prev = chan;
         }
      }
   }
   /* Handle muted channels */
   for (c=0;c<st->layout.nb_channels;c++)
   {
      if (st->layout.mapping[c] == 255)
      {
         (*copy_channel_out)(pcm, st->layout.nb_channels, c,
            NULL, 0, frame_size, user_data);
      }
   }
   RESTORE_STACK;
   return frame_size;
}

#if !defined(DISABLE_FLOAT_API)
static void opus_copy_channel_out_float(
  void *dst,
  int dst_stride,
  int dst_channel,
  const opus_val16 *src,
  int src_stride,
  int frame_size,
  void *user_data
)
{
   float *float_dst;
   opus_int32 i;
   (void)user_data;
   float_dst = (float*)dst;
   if (src != NULL)
   {
      for (i=0;i<frame_size;i++)
#if defined(FIXED_POINT)
         float_dst[i*dst_stride+dst_channel] = (1/32768.f)*src[i*src_stride];
#else
         float_dst[i*dst_stride+dst_channel] = src[i*src_stride];
#endif
   }
   else
   {
      for (i=0;i<frame_size;i++)
         float_dst[i*dst_stride+dst_channel] = 0;
   }
}
#endif

static void opus_copy_channel_out_short(
  void *dst,
  int dst_stride,
  int dst_channel,
  const opus_val16 *src,
  int src_stride,
  int frame_size,
  void *user_data
)
{
   opus_int16 *short_dst;
   opus_int32 i;
   (void)user_data;
   short_dst = (opus_int16*)dst;
   if (src != NULL)
   {
      for (i=0;i<frame_size;i++)
#if defined(FIXED_POINT)
         short_dst[i*dst_stride+dst_channel] = src[i*src_stride];
#else
         short_dst[i*dst_stride+dst_channel] = FLOAT2INT16(src[i*src_stride]);
#endif
   }
   else
   {
      for (i=0;i<frame_size;i++)
         short_dst[i*dst_stride+dst_channel] = 0;
   }
}



#ifdef FIXED_POINT
int opus_multistream_decode(
      OpusMSDecoder *st,
      const unsigned char *data,
      opus_int32 len,
      opus_int16 *pcm,
      int frame_size,
      int decode_fec
)
{
   return opus_multistream_decode_native(st, data, len,
       pcm, opus_copy_channel_out_short, frame_size, decode_fec, 0, NULL);
}

#ifndef DISABLE_FLOAT_API
int opus_multistream_decode_float(OpusMSDecoder *st, const unsigned char *data,
      opus_int32 len, float *pcm, int frame_size, int decode_fec)
{
   return opus_multistream_decode_native(st, data, len,
       pcm, opus_copy_channel_out_float, frame_size, decode_fec, 0, NULL);
}
#endif

#else

int opus_multistream_decode(OpusMSDecoder *st, const unsigned char *data,
      opus_int32 len, opus_int16 *pcm, int frame_size, int decode_fec)
{
   return opus_multistream_decode_native(st, data, len,
       pcm, opus_copy_channel_out_short, frame_size, decode_fec, 1, NULL);
}

int opus_multistream_decode_float(
      OpusMSDecoder *st,
      const unsigned char *data,
      opus_int32 len,
      opus_val16 *pcm,
      int frame_size,
      int decode_fec
)
{
   return opus_multistream_decode_native(st, data, len,
       pcm, opus_copy_channel_out_float, frame_size, decode_fec, 0, NULL);
}
#endif

int opus_multistream_decoder_ctl_va_list(OpusMSDecoder *st, int request,
                                         va_list ap)
{
   int coupled_size, mono_size;
   char *ptr;
   int ret = OPUS_OK;

   coupled_size = opus_decoder_get_size(2);
   mono_size = opus_decoder_get_size(1);
   ptr = (char*)st + align(sizeof(OpusMSDecoder));
   switch (request)
   {
       case OPUS_GET_BANDWIDTH_REQUEST:
       case OPUS_GET_SAMPLE_RATE_REQUEST:
       case OPUS_GET_GAIN_REQUEST:
       case OPUS_GET_LAST_PACKET_DURATION_REQUEST:
       case OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST:
       {
          OpusDecoder *dec;
          /* For int32* GET params, just query the first stream */
          opus_int32 *value = va_arg(ap, opus_int32*);
          dec = (OpusDecoder*)ptr;
          ret = opus_decoder_ctl(dec, request, value);
       }
       break;
       case OPUS_GET_FINAL_RANGE_REQUEST:
       {
          int s;
          opus_uint32 *value = va_arg(ap, opus_uint32*);
          opus_uint32 tmp;
          if (!value)
          {
             goto bad_arg;
          }
          *value = 0;
          for (s=0;s<st->layout.nb_streams;s++)
          {
             OpusDecoder *dec;
             dec = (OpusDecoder*)ptr;
             if (s < st->layout.nb_coupled_streams)
                ptr += align(coupled_size);
             else
                ptr += align(mono_size);
             ret = opus_decoder_ctl(dec, request, &tmp);
             if (ret != OPUS_OK) break;
             *value ^= tmp;
          }
       }
       break;
       case OPUS_RESET_STATE:
       {
          int s;
          for (s=0;s<st->layout.nb_streams;s++)
          {
             OpusDecoder *dec;

             dec = (OpusDecoder*)ptr;
             if (s < st->layout.nb_coupled_streams)
                ptr += align(coupled_size);
             else
                ptr += align(mono_size);
             ret = opus_decoder_ctl(dec, OPUS_RESET_STATE);
             if (ret != OPUS_OK)
                break;
          }
       }
       break;
       case OPUS_MULTISTREAM_GET_DECODER_STATE_REQUEST:
       {
          int s;
          opus_int32 stream_id;
          OpusDecoder **value;
          stream_id = va_arg(ap, opus_int32);
          if (stream_id<0 || stream_id >= st->layout.nb_streams)
             goto bad_arg;
          value = va_arg(ap, OpusDecoder**);
          if (!value)
          {
             goto bad_arg;
          }
          for (s=0;s<stream_id;s++)
          {
             if (s < st->layout.nb_coupled_streams)
                ptr += align(coupled_size);
             else
                ptr += align(mono_size);
          }
          *value = (OpusDecoder*)ptr;
       }
       break;
       case OPUS_SET_GAIN_REQUEST:
       case OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST:
       {
          int s;
          /* This works for int32 params */
          opus_int32 value = va_arg(ap, opus_int32);
          for (s=0;s<st->layout.nb_streams;s++)
          {
             OpusDecoder *dec;

             dec = (OpusDecoder*)ptr;
             if (s < st->layout.nb_coupled_streams)
                ptr += align(coupled_size);
             else
                ptr += align(mono_size);
             ret = opus_decoder_ctl(dec, request, value);
             if (ret != OPUS_OK)
                break;
          }
       }
       break;
       default:
          ret = OPUS_UNIMPLEMENTED;
       break;
   }
   return ret;
bad_arg:
   return OPUS_BAD_ARG;
}

int opus_multistream_decoder_ctl(OpusMSDecoder *st, int request, ...)
{
   int ret;
   va_list ap;
   va_start(ap, request);
   ret = opus_multistream_decoder_ctl_va_list(st, request, ap);
   va_end(ap);
   return ret;
}

void opus_multistream_decoder_destroy(OpusMSDecoder *st)
{
    opus_free(st);
}

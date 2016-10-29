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

#include "opus.h"
#include "opus_private.h"
#include "os_support.h"


int opus_repacketizer_get_size(void)
{
   return sizeof(OpusRepacketizer);
}

OpusRepacketizer *opus_repacketizer_init(OpusRepacketizer *rp)
{
   rp->nb_frames = 0;
   return rp;
}

OpusRepacketizer *opus_repacketizer_create(void)
{
   OpusRepacketizer *rp;
   rp=(OpusRepacketizer *)opus_alloc(opus_repacketizer_get_size());
   if(rp==NULL)return NULL;
   return opus_repacketizer_init(rp);
}

void opus_repacketizer_destroy(OpusRepacketizer *rp)
{
   opus_free(rp);
}

static int opus_repacketizer_cat_impl(OpusRepacketizer *rp, const unsigned char *data, opus_int32 len, int self_delimited)
{
   unsigned char tmp_toc;
   int curr_nb_frames,ret;
   /* Set of check ToC */
   if (len<1) return OPUS_INVALID_PACKET;
   if (rp->nb_frames == 0)
   {
      rp->toc = data[0];
      rp->framesize = opus_packet_get_samples_per_frame(data, 8000);
   } else if ((rp->toc&0xFC) != (data[0]&0xFC))
   {
      /*fprintf(stderr, "toc mismatch: 0x%x vs 0x%x\n", rp->toc, data[0]);*/
      return OPUS_INVALID_PACKET;
   }
   curr_nb_frames = opus_packet_get_nb_frames(data, len);
   if(curr_nb_frames<1) return OPUS_INVALID_PACKET;

   /* Check the 120 ms maximum packet size */
   if ((curr_nb_frames+rp->nb_frames)*rp->framesize > 960)
   {
      return OPUS_INVALID_PACKET;
   }

   ret=opus_packet_parse_impl(data, len, self_delimited, &tmp_toc, &rp->frames[rp->nb_frames], &rp->len[rp->nb_frames], NULL, NULL);
   if(ret<1)return ret;

   rp->nb_frames += curr_nb_frames;
   return OPUS_OK;
}

int opus_repacketizer_cat(OpusRepacketizer *rp, const unsigned char *data, opus_int32 len)
{
   return opus_repacketizer_cat_impl(rp, data, len, 0);
}

int opus_repacketizer_get_nb_frames(OpusRepacketizer *rp)
{
   return rp->nb_frames;
}

opus_int32 opus_repacketizer_out_range_impl(OpusRepacketizer *rp, int begin, int end,
      unsigned char *data, opus_int32 maxlen, int self_delimited, int pad)
{
   int i, count;
   opus_int32 tot_size;
   opus_int16 *len;
   const unsigned char **frames;
   unsigned char * ptr;

   if (begin<0 || begin>=end || end>rp->nb_frames)
   {
      /*fprintf(stderr, "%d %d %d\n", begin, end, rp->nb_frames);*/
      return OPUS_BAD_ARG;
   }
   count = end-begin;

   len = rp->len+begin;
   frames = rp->frames+begin;
   if (self_delimited)
      tot_size = 1 + (len[count-1]>=252);
   else
      tot_size = 0;

   ptr = data;
   if (count==1)
   {
      /* Code 0 */
      tot_size += len[0]+1;
      if (tot_size > maxlen)
         return OPUS_BUFFER_TOO_SMALL;
      *ptr++ = rp->toc&0xFC;
   } else if (count==2)
   {
      if (len[1] == len[0])
      {
         /* Code 1 */
         tot_size += 2*len[0]+1;
         if (tot_size > maxlen)
            return OPUS_BUFFER_TOO_SMALL;
         *ptr++ = (rp->toc&0xFC) | 0x1;
      } else {
         /* Code 2 */
         tot_size += len[0]+len[1]+2+(len[0]>=252);
         if (tot_size > maxlen)
            return OPUS_BUFFER_TOO_SMALL;
         *ptr++ = (rp->toc&0xFC) | 0x2;
         ptr += encode_size(len[0], ptr);
      }
   }
   if (count > 2 || (pad && tot_size < maxlen))
   {
      /* Code 3 */
      int vbr;
      int pad_amount=0;

      /* Restart the process for the padding case */
      ptr = data;
      if (self_delimited)
         tot_size = 1 + (len[count-1]>=252);
      else
         tot_size = 0;
      vbr = 0;
      for (i=1;i<count;i++)
      {
         if (len[i] != len[0])
         {
            vbr=1;
            break;
         }
      }
      if (vbr)
      {
         tot_size += 2;
         for (i=0;i<count-1;i++)
            tot_size += 1 + (len[i]>=252) + len[i];
         tot_size += len[count-1];

         if (tot_size > maxlen)
            return OPUS_BUFFER_TOO_SMALL;
         *ptr++ = (rp->toc&0xFC) | 0x3;
         *ptr++ = count | 0x80;
      } else {
         tot_size += count*len[0]+2;
         if (tot_size > maxlen)
            return OPUS_BUFFER_TOO_SMALL;
         *ptr++ = (rp->toc&0xFC) | 0x3;
         *ptr++ = count;
      }
      pad_amount = pad ? (maxlen-tot_size) : 0;
      if (pad_amount != 0)
      {
         int nb_255s;
         data[1] |= 0x40;
         nb_255s = (pad_amount-1)/255;
         for (i=0;i<nb_255s;i++)
            *ptr++ = 255;
         *ptr++ = pad_amount-255*nb_255s-1;
         tot_size += pad_amount;
      }
      if (vbr)
      {
         for (i=0;i<count-1;i++)
            ptr += encode_size(len[i], ptr);
      }
   }
   if (self_delimited) {
      int sdlen = encode_size(len[count-1], ptr);
      ptr += sdlen;
   }
   /* Copy the actual data */
   for (i=0;i<count;i++)
   {
      /* Using OPUS_MOVE() instead of OPUS_COPY() in case we're doing in-place
         padding from opus_packet_pad or opus_packet_unpad(). */
      celt_assert(frames[i] + len[i] <= data || ptr <= frames[i]);
      OPUS_MOVE(ptr, frames[i], len[i]);
      ptr += len[i];
   }
   if (pad)
   {
      /* Fill padding with zeros. */
      while (ptr<data+maxlen)
         *ptr++=0;
   }
   return tot_size;
}

opus_int32 opus_repacketizer_out_range(OpusRepacketizer *rp, int begin, int end, unsigned char *data, opus_int32 maxlen)
{
   return opus_repacketizer_out_range_impl(rp, begin, end, data, maxlen, 0, 0);
}

opus_int32 opus_repacketizer_out(OpusRepacketizer *rp, unsigned char *data, opus_int32 maxlen)
{
   return opus_repacketizer_out_range_impl(rp, 0, rp->nb_frames, data, maxlen, 0, 0);
}

int opus_packet_pad(unsigned char *data, opus_int32 len, opus_int32 new_len)
{
   OpusRepacketizer rp;
   opus_int32 ret;
   if (len < 1)
      return OPUS_BAD_ARG;
   if (len==new_len)
      return OPUS_OK;
   else if (len > new_len)
      return OPUS_BAD_ARG;
   opus_repacketizer_init(&rp);
   /* Moving payload to the end of the packet so we can do in-place padding */
   OPUS_MOVE(data+new_len-len, data, len);
   opus_repacketizer_cat(&rp, data+new_len-len, len);
   ret = opus_repacketizer_out_range_impl(&rp, 0, rp.nb_frames, data, new_len, 0, 1);
   if (ret > 0)
      return OPUS_OK;
   else
      return ret;
}

opus_int32 opus_packet_unpad(unsigned char *data, opus_int32 len)
{
   OpusRepacketizer rp;
   opus_int32 ret;
   if (len < 1)
      return OPUS_BAD_ARG;
   opus_repacketizer_init(&rp);
   ret = opus_repacketizer_cat(&rp, data, len);
   if (ret < 0)
      return ret;
   ret = opus_repacketizer_out_range_impl(&rp, 0, rp.nb_frames, data, len, 0, 0);
   celt_assert(ret > 0 && ret <= len);
   return ret;
}

int opus_multistream_packet_pad(unsigned char *data, opus_int32 len, opus_int32 new_len, int nb_streams)
{
   int s;
   int count;
   unsigned char toc;
   opus_int16 size[48];
   opus_int32 packet_offset;
   opus_int32 amount;

   if (len < 1)
      return OPUS_BAD_ARG;
   if (len==new_len)
      return OPUS_OK;
   else if (len > new_len)
      return OPUS_BAD_ARG;
   amount = new_len - len;
   /* Seek to last stream */
   for (s=0;s<nb_streams-1;s++)
   {
      if (len<=0)
         return OPUS_INVALID_PACKET;
      count = opus_packet_parse_impl(data, len, 1, &toc, NULL,
                                     size, NULL, &packet_offset);
      if (count<0)
         return count;
      data += packet_offset;
      len -= packet_offset;
   }
   return opus_packet_pad(data, len, len+amount);
}

opus_int32 opus_multistream_packet_unpad(unsigned char *data, opus_int32 len, int nb_streams)
{
   int s;
   unsigned char toc;
   opus_int16 size[48];
   opus_int32 packet_offset;
   OpusRepacketizer rp;
   unsigned char *dst;
   opus_int32 dst_len;

   if (len < 1)
      return OPUS_BAD_ARG;
   dst = data;
   dst_len = 0;
   /* Unpad all frames */
   for (s=0;s<nb_streams;s++)
   {
      opus_int32 ret;
      int self_delimited = s!=nb_streams-1;
      if (len<=0)
         return OPUS_INVALID_PACKET;
      opus_repacketizer_init(&rp);
      ret = opus_packet_parse_impl(data, len, self_delimited, &toc, NULL,
                                     size, NULL, &packet_offset);
      if (ret<0)
         return ret;
      ret = opus_repacketizer_cat_impl(&rp, data, packet_offset, self_delimited);
      if (ret < 0)
         return ret;
      ret = opus_repacketizer_out_range_impl(&rp, 0, rp.nb_frames, dst, len, self_delimited, 0);
      if (ret < 0)
         return ret;
      else
         dst_len += ret;
      dst += ret;
      data += packet_offset;
      len -= packet_offset;
   }
   return dst_len;
}


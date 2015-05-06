/* Copyright (C) 2002 Jean-Marc Valin
   File speex_callbacks.c
   Callback handling and in-band signalling


   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/


#include "config.h"


#include <speex/speex_callbacks.h>
#include "arch.h"
#include "os_support.h"

EXPORT int speex_inband_handler(SpeexBits *bits, SpeexCallback *callback_list, void *state)
{
   int id;
   SpeexCallback *callback;
   /*speex_bits_advance(bits, 5);*/
   id=speex_bits_unpack_unsigned(bits, 4);
   callback = callback_list+id;

   if (callback->func)
   {
      return callback->func(bits, state, callback->data);
   } else
      /*If callback is not registered, skip the right number of bits*/
   {
      int adv;
      if (id<2)
         adv = 1;
      else if (id<8)
         adv = 4;
      else if (id<10)
         adv = 8;
      else if (id<12)
         adv = 16;
      else if (id<14)
         adv = 32;
      else 
         adv = 64;
      speex_bits_advance(bits, adv);
   }
   return 0;
}

EXPORT int speex_std_mode_request_handler(SpeexBits *bits, void *state, void *data)
{
   spx_int32_t m;
   m = speex_bits_unpack_unsigned(bits, 4);
   speex_encoder_ctl(data, SPEEX_SET_MODE, &m);
   return 0;
}

EXPORT int speex_std_low_mode_request_handler(SpeexBits *bits, void *state, void *data)
{
   spx_int32_t m;
   m = speex_bits_unpack_unsigned(bits, 4);
   speex_encoder_ctl(data, SPEEX_SET_LOW_MODE, &m);
   return 0;
}

EXPORT int speex_std_high_mode_request_handler(SpeexBits *bits, void *state, void *data)
{
   spx_int32_t m;
   m = speex_bits_unpack_unsigned(bits, 4);
   speex_encoder_ctl(data, SPEEX_SET_HIGH_MODE, &m);
   return 0;
}

#ifndef DISABLE_VBR
EXPORT int speex_std_vbr_request_handler(SpeexBits *bits, void *state, void *data)
{
   spx_int32_t vbr;
   vbr = speex_bits_unpack_unsigned(bits, 1);
   speex_encoder_ctl(data, SPEEX_SET_VBR, &vbr);
   return 0;
}
#endif /* #ifndef DISABLE_VBR */

EXPORT int speex_std_enh_request_handler(SpeexBits *bits, void *state, void *data)
{
   spx_int32_t enh;
   enh = speex_bits_unpack_unsigned(bits, 1);
   speex_decoder_ctl(data, SPEEX_SET_ENH, &enh);
   return 0;
}

#ifndef DISABLE_VBR
EXPORT int speex_std_vbr_quality_request_handler(SpeexBits *bits, void *state, void *data)
{
   float qual;
   qual = speex_bits_unpack_unsigned(bits, 4);
   speex_encoder_ctl(data, SPEEX_SET_VBR_QUALITY, &qual);
   return 0;
}
#endif /* #ifndef DISABLE_VBR */

EXPORT int speex_std_char_handler(SpeexBits *bits, void *state, void *data)
{
   unsigned char ch;
   ch = speex_bits_unpack_unsigned(bits, 8);
   _speex_putc(ch, data);
   /*printf("speex_std_char_handler ch=%x\n", ch);*/
   return 0;
}



/* Default handler for user callbacks: skip it */
EXPORT int speex_default_user_handler(SpeexBits *bits, void *state, void *data)
{
   int req_size = speex_bits_unpack_unsigned(bits, 4);
   speex_bits_advance(bits, 5+8*req_size);
   return 0;
}

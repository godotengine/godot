/* Copyright (C) 2002 Jean-Marc Valin 
   File: stereo.c

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


#include <speex/speex_stereo.h>
#include <speex/speex_callbacks.h>
#include "math_approx.h"
#include "vq.h"
#include <math.h>
#include "os_support.h"

typedef struct RealSpeexStereoState {
   spx_word32_t balance;      /**< Left/right balance info */
   spx_word32_t e_ratio;      /**< Ratio of energies: E(left+right)/[E(left)+E(right)]  */
   spx_word32_t smooth_left;  /**< Smoothed left channel gain */
   spx_word32_t smooth_right; /**< Smoothed right channel gain */
   spx_uint32_t reserved1;     /**< Reserved for future use */
   spx_int32_t reserved2;     /**< Reserved for future use */
} RealSpeexStereoState;


/*float e_ratio_quant[4] = {1, 1.26, 1.587, 2};*/
#ifndef FIXED_POINT
static const float e_ratio_quant[4] = {.25f, .315f, .397f, .5f};
static const float e_ratio_quant_bounds[3] = {0.2825f, 0.356f, 0.4485f};
#else
static const spx_word16_t e_ratio_quant[4] = {8192, 10332, 13009, 16384};
static const spx_word16_t e_ratio_quant_bounds[3] = {9257, 11665, 14696};
static const spx_word16_t balance_bounds[31] = {18, 23, 30, 38, 49, 63,  81, 104,
   134, 172, 221,  284, 364, 468, 600, 771,
   990, 1271, 1632, 2096, 2691, 3455, 4436, 5696,
   7314, 9392, 12059, 15484, 19882, 25529, 32766};
#endif

/* This is an ugly compatibility hack that properly resets the stereo state
   In case it it compiled in fixed-point, but initialised with the deprecated
   floating point static initialiser */
#ifdef FIXED_POINT
#define COMPATIBILITY_HACK(s) do {if ((s)->reserved1 != 0xdeadbeef) speex_stereo_state_reset((SpeexStereoState*)s); } while (0);
#else
#define COMPATIBILITY_HACK(s) 
#endif

EXPORT SpeexStereoState *speex_stereo_state_init()
{
   SpeexStereoState *stereo = speex_alloc(sizeof(SpeexStereoState));
   speex_stereo_state_reset(stereo);
   return stereo;
}

EXPORT void speex_stereo_state_reset(SpeexStereoState *_stereo)
{
   RealSpeexStereoState *stereo = (RealSpeexStereoState*)_stereo;
#ifdef FIXED_POINT
   stereo->balance = 65536;
   stereo->e_ratio = 16384;
   stereo->smooth_left = 16384;
   stereo->smooth_right = 16384;
   stereo->reserved1 = 0xdeadbeef;
   stereo->reserved2 = 0;
#else
   stereo->balance = 1.0f;
   stereo->e_ratio = .5f;
   stereo->smooth_left = 1.f;
   stereo->smooth_right = 1.f;
   stereo->reserved1 = 0;
   stereo->reserved2 = 0;
#endif   
}

EXPORT void speex_stereo_state_destroy(SpeexStereoState *stereo)
{
   speex_free(stereo);
}

#ifndef DISABLE_FLOAT_API
EXPORT void speex_encode_stereo(float *data, int frame_size, SpeexBits *bits)
{
   int i, tmp;
   float e_left=0, e_right=0, e_tot=0;
   float balance, e_ratio;
   for (i=0;i<frame_size;i++)
   {
      e_left  += ((float)data[2*i])*data[2*i];
      e_right += ((float)data[2*i+1])*data[2*i+1];
      data[i] =  .5*(((float)data[2*i])+data[2*i+1]);
      e_tot   += ((float)data[i])*data[i];
   }
   balance=(e_left+1)/(e_right+1);
   e_ratio = e_tot/(1+e_left+e_right);

   /*Quantization*/
   speex_bits_pack(bits, 14, 5);
   speex_bits_pack(bits, SPEEX_INBAND_STEREO, 4);
   
   balance=4*log(balance);

   /*Pack sign*/
   if (balance>0)
      speex_bits_pack(bits, 0, 1);
   else
      speex_bits_pack(bits, 1, 1);
   balance=floor(.5+fabs(balance));
   if (balance>30)
      balance=31;
   
   speex_bits_pack(bits, (int)balance, 5);
   
   /* FIXME: this is a hack */
   tmp=scal_quant(e_ratio*Q15_ONE, e_ratio_quant_bounds, 4);
   speex_bits_pack(bits, tmp, 2);
}
#endif /* #ifndef DISABLE_FLOAT_API */

EXPORT void speex_encode_stereo_int(spx_int16_t *data, int frame_size, SpeexBits *bits)
{
   int i, tmp;
   spx_word32_t e_left=0, e_right=0, e_tot=0;
   spx_word32_t balance, e_ratio;
   spx_word32_t largest, smallest;
   int balance_id;
#ifdef FIXED_POINT
   int shift;
#endif
   
   /* In band marker */
   speex_bits_pack(bits, 14, 5);
   /* Stereo marker */
   speex_bits_pack(bits, SPEEX_INBAND_STEREO, 4);

   for (i=0;i<frame_size;i++)
   {
      e_left  += SHR32(MULT16_16(data[2*i],data[2*i]),8);
      e_right += SHR32(MULT16_16(data[2*i+1],data[2*i+1]),8);
#ifdef FIXED_POINT
      /* I think this is actually unbiased */
      data[i] =  SHR16(data[2*i],1)+PSHR16(data[2*i+1],1);
#else
      data[i] =  .5*(((float)data[2*i])+data[2*i+1]);
#endif
      e_tot   += SHR32(MULT16_16(data[i],data[i]),8);
   }
   if (e_left > e_right)
   {
      speex_bits_pack(bits, 0, 1);
      largest = e_left;
      smallest = e_right;
   } else {
      speex_bits_pack(bits, 1, 1);
      largest = e_right;
      smallest = e_left;
   }

   /* Balance quantization */
#ifdef FIXED_POINT
   shift = spx_ilog2(largest)-15;
   largest = VSHR32(largest, shift-4);
   smallest = VSHR32(smallest, shift);
   balance = DIV32(largest, ADD32(smallest, 1));
   if (balance > 32767)
      balance = 32767;
   balance_id = scal_quant(EXTRACT16(balance), balance_bounds, 32);
#else
   balance=(largest+1.)/(smallest+1.);
   balance=4*log(balance);
   balance_id=floor(.5+fabs(balance));
   if (balance_id>30)
      balance_id=31;
#endif
   
   speex_bits_pack(bits, balance_id, 5);
   
   /* "coherence" quantisation */
#ifdef FIXED_POINT
   shift = spx_ilog2(e_tot);
   e_tot = VSHR32(e_tot, shift-25);
   e_left = VSHR32(e_left, shift-10);
   e_right = VSHR32(e_right, shift-10);
   e_ratio = DIV32(e_tot, e_left+e_right+1);
#else
   e_ratio = e_tot/(1.+e_left+e_right);
#endif
   
   tmp=scal_quant(EXTRACT16(e_ratio), e_ratio_quant_bounds, 4);
   /*fprintf (stderr, "%d %d %d %d\n", largest, smallest, balance_id, e_ratio);*/
   speex_bits_pack(bits, tmp, 2);
}

#ifndef DISABLE_FLOAT_API
EXPORT void speex_decode_stereo(float *data, int frame_size, SpeexStereoState *_stereo)
{
   int i;
   spx_word32_t balance;
   spx_word16_t e_left, e_right, e_ratio;
   RealSpeexStereoState *stereo = (RealSpeexStereoState*)_stereo;
   
   COMPATIBILITY_HACK(stereo);
   
   balance=stereo->balance;
   e_ratio=stereo->e_ratio;
   
   /* These two are Q14, with max value just below 2. */
   e_right = DIV32(QCONST32(1., 22), spx_sqrt(MULT16_32_Q15(e_ratio, ADD32(QCONST32(1., 16), balance))));
   e_left = SHR32(MULT16_16(spx_sqrt(balance), e_right), 8);

   for (i=frame_size-1;i>=0;i--)
   {
      spx_word16_t tmp=data[i];
      stereo->smooth_left = EXTRACT16(PSHR32(MAC16_16(MULT16_16(stereo->smooth_left, QCONST16(0.98, 15)), e_left, QCONST16(0.02, 15)), 15));
      stereo->smooth_right = EXTRACT16(PSHR32(MAC16_16(MULT16_16(stereo->smooth_right, QCONST16(0.98, 15)), e_right, QCONST16(0.02, 15)), 15));
      data[2*i] = (float)MULT16_16_P14(stereo->smooth_left, tmp);
      data[2*i+1] = (float)MULT16_16_P14(stereo->smooth_right, tmp);
   }
}
#endif /* #ifndef DISABLE_FLOAT_API */

EXPORT void speex_decode_stereo_int(spx_int16_t *data, int frame_size, SpeexStereoState *_stereo)
{
   int i;
   spx_word32_t balance;
   spx_word16_t e_left, e_right, e_ratio;
   RealSpeexStereoState *stereo = (RealSpeexStereoState*)_stereo;

   COMPATIBILITY_HACK(stereo);
   
   balance=stereo->balance;
   e_ratio=stereo->e_ratio;
   
   /* These two are Q14, with max value just below 2. */
   e_right = DIV32(QCONST32(1., 22), spx_sqrt(MULT16_32_Q15(e_ratio, ADD32(QCONST32(1., 16), balance))));
   e_left = SHR32(MULT16_16(spx_sqrt(balance), e_right), 8);

   for (i=frame_size-1;i>=0;i--)
   {
      spx_int16_t tmp=data[i];
      stereo->smooth_left = EXTRACT16(PSHR32(MAC16_16(MULT16_16(stereo->smooth_left, QCONST16(0.98, 15)), e_left, QCONST16(0.02, 15)), 15));
      stereo->smooth_right = EXTRACT16(PSHR32(MAC16_16(MULT16_16(stereo->smooth_right, QCONST16(0.98, 15)), e_right, QCONST16(0.02, 15)), 15));
      data[2*i] = (spx_int16_t)MULT16_16_P14(stereo->smooth_left, tmp);
      data[2*i+1] = (spx_int16_t)MULT16_16_P14(stereo->smooth_right, tmp);
   }
}

EXPORT int speex_std_stereo_request_handler(SpeexBits *bits, void *state, void *data)
{
   RealSpeexStereoState *stereo;
   spx_word16_t sign=1, dexp;
   int tmp;

   stereo = (RealSpeexStereoState*)data;
   
   COMPATIBILITY_HACK(stereo);

   if (speex_bits_unpack_unsigned(bits, 1))
      sign=-1;
   dexp = speex_bits_unpack_unsigned(bits, 5);
#ifndef FIXED_POINT
   stereo->balance = exp(sign*.25*dexp);
#else
   stereo->balance = spx_exp(MULT16_16(sign, SHL16(dexp, 9)));
#endif
   tmp = speex_bits_unpack_unsigned(bits, 2);
   stereo->e_ratio = e_ratio_quant[tmp];

   return 0;
}

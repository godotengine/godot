/* Copyright (c) 2010-2011 Xiph.Org Foundation, Skype Limited
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

#include <stdarg.h>
#include "celt.h"
#include "entenc.h"
#include "modes.h"
#include "API.h"
#include "stack_alloc.h"
#include "float_cast.h"
#include "opus.h"
#include "arch.h"
#include "pitch.h"
#include "opus_private.h"
#include "os_support.h"
#include "cpu_support.h"
#include "analysis.h"
#include "mathops.h"
#include "tuning_parameters.h"
#ifdef FIXED_POINT
#include "fixed/structs_FIX.h"
#else
#include "float/structs_FLP.h"
#endif

#define MAX_ENCODER_BUFFER 480

#ifndef DISABLE_FLOAT_API
#define PSEUDO_SNR_THRESHOLD 316.23f    /* 10^(25/10) */
#endif

typedef struct {
   opus_val32 XX, XY, YY;
   opus_val16 smoothed_width;
   opus_val16 max_follower;
} StereoWidthState;

struct OpusEncoder {
    int          celt_enc_offset;
    int          silk_enc_offset;
    silk_EncControlStruct silk_mode;
    int          application;
    int          channels;
    int          delay_compensation;
    int          force_channels;
    int          signal_type;
    int          user_bandwidth;
    int          max_bandwidth;
    int          user_forced_mode;
    int          voice_ratio;
    opus_int32   Fs;
    int          use_vbr;
    int          vbr_constraint;
    int          variable_duration;
    opus_int32   bitrate_bps;
    opus_int32   user_bitrate_bps;
    int          lsb_depth;
    int          encoder_buffer;
    int          lfe;
    int          arch;
    int          use_dtx;                 /* general DTX for both SILK and CELT */
#ifndef DISABLE_FLOAT_API
    TonalityAnalysisState analysis;
#endif

#define OPUS_ENCODER_RESET_START stream_channels
    int          stream_channels;
    opus_int16   hybrid_stereo_width_Q14;
    opus_int32   variable_HP_smth2_Q15;
    opus_val16   prev_HB_gain;
    opus_val32   hp_mem[4];
    int          mode;
    int          prev_mode;
    int          prev_channels;
    int          prev_framesize;
    int          bandwidth;
    /* Bandwidth determined automatically from the rate (before any other adjustment) */
    int          auto_bandwidth;
    int          silk_bw_switch;
    /* Sampling rate (at the API level) */
    int          first;
    opus_val16 * energy_masking;
    StereoWidthState width_mem;
    opus_val16   delay_buffer[MAX_ENCODER_BUFFER*2];
#ifndef DISABLE_FLOAT_API
    int          detected_bandwidth;
    int          nb_no_activity_frames;
    opus_val32   peak_signal_energy;
#endif
    int          nonfinal_frame; /* current frame is not the final in a packet */
    opus_uint32  rangeFinal;
};

/* Transition tables for the voice and music. First column is the
   middle (memoriless) threshold. The second column is the hysteresis
   (difference with the middle) */
static const opus_int32 mono_voice_bandwidth_thresholds[8] = {
         9000,  700, /* NB<->MB */
         9000,  700, /* MB<->WB */
        13500, 1000, /* WB<->SWB */
        14000, 2000, /* SWB<->FB */
};
static const opus_int32 mono_music_bandwidth_thresholds[8] = {
         9000,  700, /* NB<->MB */
         9000,  700, /* MB<->WB */
        11000, 1000, /* WB<->SWB */
        12000, 2000, /* SWB<->FB */
};
static const opus_int32 stereo_voice_bandwidth_thresholds[8] = {
         9000,  700, /* NB<->MB */
         9000,  700, /* MB<->WB */
        13500, 1000, /* WB<->SWB */
        14000, 2000, /* SWB<->FB */
};
static const opus_int32 stereo_music_bandwidth_thresholds[8] = {
         9000,  700, /* NB<->MB */
         9000,  700, /* MB<->WB */
        11000, 1000, /* WB<->SWB */
        12000, 2000, /* SWB<->FB */
};
/* Threshold bit-rates for switching between mono and stereo */
static const opus_int32 stereo_voice_threshold = 19000;
static const opus_int32 stereo_music_threshold = 17000;

/* Threshold bit-rate for switching between SILK/hybrid and CELT-only */
static const opus_int32 mode_thresholds[2][2] = {
      /* voice */ /* music */
      {  64000,      10000}, /* mono */
      {  44000,      10000}, /* stereo */
};

static const opus_int32 fec_thresholds[] = {
        12000, 1000, /* NB */
        14000, 1000, /* MB */
        16000, 1000, /* WB */
        20000, 1000, /* SWB */
        22000, 1000, /* FB */
};

int opus_encoder_get_size(int channels)
{
    int silkEncSizeBytes, celtEncSizeBytes;
    int ret;
    if (channels<1 || channels > 2)
        return 0;
    ret = silk_Get_Encoder_Size( &silkEncSizeBytes );
    if (ret)
        return 0;
    silkEncSizeBytes = align(silkEncSizeBytes);
    celtEncSizeBytes = celt_encoder_get_size(channels);
    return align(sizeof(OpusEncoder))+silkEncSizeBytes+celtEncSizeBytes;
}

int opus_encoder_init(OpusEncoder* st, opus_int32 Fs, int channels, int application)
{
    void *silk_enc;
    CELTEncoder *celt_enc;
    int err;
    int ret, silkEncSizeBytes;

   if((Fs!=48000&&Fs!=24000&&Fs!=16000&&Fs!=12000&&Fs!=8000)||(channels!=1&&channels!=2)||
        (application != OPUS_APPLICATION_VOIP && application != OPUS_APPLICATION_AUDIO
        && application != OPUS_APPLICATION_RESTRICTED_LOWDELAY))
        return OPUS_BAD_ARG;

    OPUS_CLEAR((char*)st, opus_encoder_get_size(channels));
    /* Create SILK encoder */
    ret = silk_Get_Encoder_Size( &silkEncSizeBytes );
    if (ret)
        return OPUS_BAD_ARG;
    silkEncSizeBytes = align(silkEncSizeBytes);
    st->silk_enc_offset = align(sizeof(OpusEncoder));
    st->celt_enc_offset = st->silk_enc_offset+silkEncSizeBytes;
    silk_enc = (char*)st+st->silk_enc_offset;
    celt_enc = (CELTEncoder*)((char*)st+st->celt_enc_offset);

    st->stream_channels = st->channels = channels;

    st->Fs = Fs;

    st->arch = opus_select_arch();

    ret = silk_InitEncoder( silk_enc, st->arch, &st->silk_mode );
    if(ret)return OPUS_INTERNAL_ERROR;

    /* default SILK parameters */
    st->silk_mode.nChannelsAPI              = channels;
    st->silk_mode.nChannelsInternal         = channels;
    st->silk_mode.API_sampleRate            = st->Fs;
    st->silk_mode.maxInternalSampleRate     = 16000;
    st->silk_mode.minInternalSampleRate     = 8000;
    st->silk_mode.desiredInternalSampleRate = 16000;
    st->silk_mode.payloadSize_ms            = 20;
    st->silk_mode.bitRate                   = 25000;
    st->silk_mode.packetLossPercentage      = 0;
    st->silk_mode.complexity                = 9;
    st->silk_mode.useInBandFEC              = 0;
    st->silk_mode.useDTX                    = 0;
    st->silk_mode.useCBR                    = 0;
    st->silk_mode.reducedDependency         = 0;

    /* Create CELT encoder */
    /* Initialize CELT encoder */
    err = celt_encoder_init(celt_enc, Fs, channels, st->arch);
    if(err!=OPUS_OK)return OPUS_INTERNAL_ERROR;

    celt_encoder_ctl(celt_enc, CELT_SET_SIGNALLING(0));
    celt_encoder_ctl(celt_enc, OPUS_SET_COMPLEXITY(st->silk_mode.complexity));

    st->use_vbr = 1;
    /* Makes constrained VBR the default (safer for real-time use) */
    st->vbr_constraint = 1;
    st->user_bitrate_bps = OPUS_AUTO;
    st->bitrate_bps = 3000+Fs*channels;
    st->application = application;
    st->signal_type = OPUS_AUTO;
    st->user_bandwidth = OPUS_AUTO;
    st->max_bandwidth = OPUS_BANDWIDTH_FULLBAND;
    st->force_channels = OPUS_AUTO;
    st->user_forced_mode = OPUS_AUTO;
    st->voice_ratio = -1;
    st->encoder_buffer = st->Fs/100;
    st->lsb_depth = 24;
    st->variable_duration = OPUS_FRAMESIZE_ARG;

    /* Delay compensation of 4 ms (2.5 ms for SILK's extra look-ahead
       + 1.5 ms for SILK resamplers and stereo prediction) */
    st->delay_compensation = st->Fs/250;

    st->hybrid_stereo_width_Q14 = 1 << 14;
    st->prev_HB_gain = Q15ONE;
    st->variable_HP_smth2_Q15 = silk_LSHIFT( silk_lin2log( VARIABLE_HP_MIN_CUTOFF_HZ ), 8 );
    st->first = 1;
    st->mode = MODE_HYBRID;
    st->bandwidth = OPUS_BANDWIDTH_FULLBAND;

#ifndef DISABLE_FLOAT_API
    tonality_analysis_init(&st->analysis, st->Fs);
    st->analysis.application = st->application;
#endif

    return OPUS_OK;
}

static unsigned char gen_toc(int mode, int framerate, int bandwidth, int channels)
{
   int period;
   unsigned char toc;
   period = 0;
   while (framerate < 400)
   {
       framerate <<= 1;
       period++;
   }
   if (mode == MODE_SILK_ONLY)
   {
       toc = (bandwidth-OPUS_BANDWIDTH_NARROWBAND)<<5;
       toc |= (period-2)<<3;
   } else if (mode == MODE_CELT_ONLY)
   {
       int tmp = bandwidth-OPUS_BANDWIDTH_MEDIUMBAND;
       if (tmp < 0)
           tmp = 0;
       toc = 0x80;
       toc |= tmp << 5;
       toc |= period<<3;
   } else /* Hybrid */
   {
       toc = 0x60;
       toc |= (bandwidth-OPUS_BANDWIDTH_SUPERWIDEBAND)<<4;
       toc |= (period-2)<<3;
   }
   toc |= (channels==2)<<2;
   return toc;
}

#ifndef FIXED_POINT
static void silk_biquad_float(
    const opus_val16      *in,            /* I:    Input signal                   */
    const opus_int32      *B_Q28,         /* I:    MA coefficients [3]            */
    const opus_int32      *A_Q28,         /* I:    AR coefficients [2]            */
    opus_val32            *S,             /* I/O:  State vector [2]               */
    opus_val16            *out,           /* O:    Output signal                  */
    const opus_int32      len,            /* I:    Signal length (must be even)   */
    int stride
)
{
    /* DIRECT FORM II TRANSPOSED (uses 2 element state vector) */
    opus_int   k;
    opus_val32 vout;
    opus_val32 inval;
    opus_val32 A[2], B[3];

    A[0] = (opus_val32)(A_Q28[0] * (1.f/((opus_int32)1<<28)));
    A[1] = (opus_val32)(A_Q28[1] * (1.f/((opus_int32)1<<28)));
    B[0] = (opus_val32)(B_Q28[0] * (1.f/((opus_int32)1<<28)));
    B[1] = (opus_val32)(B_Q28[1] * (1.f/((opus_int32)1<<28)));
    B[2] = (opus_val32)(B_Q28[2] * (1.f/((opus_int32)1<<28)));

    /* Negate A_Q28 values and split in two parts */

    for( k = 0; k < len; k++ ) {
        /* S[ 0 ], S[ 1 ]: Q12 */
        inval = in[ k*stride ];
        vout = S[ 0 ] + B[0]*inval;

        S[ 0 ] = S[1] - vout*A[0] + B[1]*inval;

        S[ 1 ] = - vout*A[1] + B[2]*inval + VERY_SMALL;

        /* Scale back to Q0 and saturate */
        out[ k*stride ] = vout;
    }
}
#endif

static void hp_cutoff(const opus_val16 *in, opus_int32 cutoff_Hz, opus_val16 *out, opus_val32 *hp_mem, int len, int channels, opus_int32 Fs, int arch)
{
   opus_int32 B_Q28[ 3 ], A_Q28[ 2 ];
   opus_int32 Fc_Q19, r_Q28, r_Q22;
   (void)arch;

   silk_assert( cutoff_Hz <= silk_int32_MAX / SILK_FIX_CONST( 1.5 * 3.14159 / 1000, 19 ) );
   Fc_Q19 = silk_DIV32_16( silk_SMULBB( SILK_FIX_CONST( 1.5 * 3.14159 / 1000, 19 ), cutoff_Hz ), Fs/1000 );
   silk_assert( Fc_Q19 > 0 && Fc_Q19 < 32768 );

   r_Q28 = SILK_FIX_CONST( 1.0, 28 ) - silk_MUL( SILK_FIX_CONST( 0.92, 9 ), Fc_Q19 );

   /* b = r * [ 1; -2; 1 ]; */
   /* a = [ 1; -2 * r * ( 1 - 0.5 * Fc^2 ); r^2 ]; */
   B_Q28[ 0 ] = r_Q28;
   B_Q28[ 1 ] = silk_LSHIFT( -r_Q28, 1 );
   B_Q28[ 2 ] = r_Q28;

   /* -r * ( 2 - Fc * Fc ); */
   r_Q22  = silk_RSHIFT( r_Q28, 6 );
   A_Q28[ 0 ] = silk_SMULWW( r_Q22, silk_SMULWW( Fc_Q19, Fc_Q19 ) - SILK_FIX_CONST( 2.0,  22 ) );
   A_Q28[ 1 ] = silk_SMULWW( r_Q22, r_Q22 );

#ifdef FIXED_POINT
   if( channels == 1 ) {
      silk_biquad_alt_stride1( in, B_Q28, A_Q28, hp_mem, out, len );
   } else {
      silk_biquad_alt_stride2( in, B_Q28, A_Q28, hp_mem, out, len, arch );
   }
#else
   silk_biquad_float( in, B_Q28, A_Q28, hp_mem, out, len, channels );
   if( channels == 2 ) {
       silk_biquad_float( in+1, B_Q28, A_Q28, hp_mem+2, out+1, len, channels );
   }
#endif
}

#ifdef FIXED_POINT
static void dc_reject(const opus_val16 *in, opus_int32 cutoff_Hz, opus_val16 *out, opus_val32 *hp_mem, int len, int channels, opus_int32 Fs)
{
   int c, i;
   int shift;

   /* Approximates -round(log2(6.3*cutoff_Hz/Fs)) */
   shift=celt_ilog2(Fs/(cutoff_Hz*4));
   for (c=0;c<channels;c++)
   {
      for (i=0;i<len;i++)
      {
         opus_val32 x, y;
         x = SHL32(EXTEND32(in[channels*i+c]), 14);
         y = x-hp_mem[2*c];
         hp_mem[2*c] = hp_mem[2*c] + PSHR32(x - hp_mem[2*c], shift);
         out[channels*i+c] = EXTRACT16(SATURATE(PSHR32(y, 14), 32767));
      }
   }
}

#else
static void dc_reject(const opus_val16 *in, opus_int32 cutoff_Hz, opus_val16 *out, opus_val32 *hp_mem, int len, int channels, opus_int32 Fs)
{
   int i;
   float coef, coef2;
   coef = 6.3f*cutoff_Hz/Fs;
   coef2 = 1-coef;
   if (channels==2)
   {
      float m0, m2;
      m0 = hp_mem[0];
      m2 = hp_mem[2];
      for (i=0;i<len;i++)
      {
         opus_val32 x0, x1, out0, out1;
         x0 = in[2*i+0];
         x1 = in[2*i+1];
         out0 = x0-m0;
         out1 = x1-m2;
         m0 = coef*x0 + VERY_SMALL + coef2*m0;
         m2 = coef*x1 + VERY_SMALL + coef2*m2;
         out[2*i+0] = out0;
         out[2*i+1] = out1;
      }
      hp_mem[0] = m0;
      hp_mem[2] = m2;
   } else {
      float m0;
      m0 = hp_mem[0];
      for (i=0;i<len;i++)
      {
         opus_val32 x, y;
         x = in[i];
         y = x-m0;
         m0 = coef*x + VERY_SMALL + coef2*m0;
         out[i] = y;
      }
      hp_mem[0] = m0;
   }
}
#endif

static void stereo_fade(const opus_val16 *in, opus_val16 *out, opus_val16 g1, opus_val16 g2,
        int overlap48, int frame_size, int channels, const opus_val16 *window, opus_int32 Fs)
{
    int i;
    int overlap;
    int inc;
    inc = 48000/Fs;
    overlap=overlap48/inc;
    g1 = Q15ONE-g1;
    g2 = Q15ONE-g2;
    for (i=0;i<overlap;i++)
    {
       opus_val32 diff;
       opus_val16 g, w;
       w = MULT16_16_Q15(window[i*inc], window[i*inc]);
       g = SHR32(MAC16_16(MULT16_16(w,g2),
             Q15ONE-w, g1), 15);
       diff = EXTRACT16(HALF32((opus_val32)in[i*channels] - (opus_val32)in[i*channels+1]));
       diff = MULT16_16_Q15(g, diff);
       out[i*channels] = out[i*channels] - diff;
       out[i*channels+1] = out[i*channels+1] + diff;
    }
    for (;i<frame_size;i++)
    {
       opus_val32 diff;
       diff = EXTRACT16(HALF32((opus_val32)in[i*channels] - (opus_val32)in[i*channels+1]));
       diff = MULT16_16_Q15(g2, diff);
       out[i*channels] = out[i*channels] - diff;
       out[i*channels+1] = out[i*channels+1] + diff;
    }
}

static void gain_fade(const opus_val16 *in, opus_val16 *out, opus_val16 g1, opus_val16 g2,
        int overlap48, int frame_size, int channels, const opus_val16 *window, opus_int32 Fs)
{
    int i;
    int inc;
    int overlap;
    int c;
    inc = 48000/Fs;
    overlap=overlap48/inc;
    if (channels==1)
    {
       for (i=0;i<overlap;i++)
       {
          opus_val16 g, w;
          w = MULT16_16_Q15(window[i*inc], window[i*inc]);
          g = SHR32(MAC16_16(MULT16_16(w,g2),
                Q15ONE-w, g1), 15);
          out[i] = MULT16_16_Q15(g, in[i]);
       }
    } else {
       for (i=0;i<overlap;i++)
       {
          opus_val16 g, w;
          w = MULT16_16_Q15(window[i*inc], window[i*inc]);
          g = SHR32(MAC16_16(MULT16_16(w,g2),
                Q15ONE-w, g1), 15);
          out[i*2] = MULT16_16_Q15(g, in[i*2]);
          out[i*2+1] = MULT16_16_Q15(g, in[i*2+1]);
       }
    }
    c=0;do {
       for (i=overlap;i<frame_size;i++)
       {
          out[i*channels+c] = MULT16_16_Q15(g2, in[i*channels+c]);
       }
    }
    while (++c<channels);
}

OpusEncoder *opus_encoder_create(opus_int32 Fs, int channels, int application, int *error)
{
   int ret;
   OpusEncoder *st;
   if((Fs!=48000&&Fs!=24000&&Fs!=16000&&Fs!=12000&&Fs!=8000)||(channels!=1&&channels!=2)||
       (application != OPUS_APPLICATION_VOIP && application != OPUS_APPLICATION_AUDIO
       && application != OPUS_APPLICATION_RESTRICTED_LOWDELAY))
   {
      if (error)
         *error = OPUS_BAD_ARG;
      return NULL;
   }
   st = (OpusEncoder *)opus_alloc(opus_encoder_get_size(channels));
   if (st == NULL)
   {
      if (error)
         *error = OPUS_ALLOC_FAIL;
      return NULL;
   }
   ret = opus_encoder_init(st, Fs, channels, application);
   if (error)
      *error = ret;
   if (ret != OPUS_OK)
   {
      opus_free(st);
      st = NULL;
   }
   return st;
}

static opus_int32 user_bitrate_to_bitrate(OpusEncoder *st, int frame_size, int max_data_bytes)
{
  if(!frame_size)frame_size=st->Fs/400;
  if (st->user_bitrate_bps==OPUS_AUTO)
    return 60*st->Fs/frame_size + st->Fs*st->channels;
  else if (st->user_bitrate_bps==OPUS_BITRATE_MAX)
    return max_data_bytes*8*st->Fs/frame_size;
  else
    return st->user_bitrate_bps;
}

#ifndef DISABLE_FLOAT_API
#ifdef FIXED_POINT
#define PCM2VAL(x) FLOAT2INT16(x)
#else
#define PCM2VAL(x) SCALEIN(x)
#endif

void downmix_float(const void *_x, opus_val32 *y, int subframe, int offset, int c1, int c2, int C)
{
   const float *x;
   int j;

   x = (const float *)_x;
   for (j=0;j<subframe;j++)
      y[j] = PCM2VAL(x[(j+offset)*C+c1]);
   if (c2>-1)
   {
      for (j=0;j<subframe;j++)
         y[j] += PCM2VAL(x[(j+offset)*C+c2]);
   } else if (c2==-2)
   {
      int c;
      for (c=1;c<C;c++)
      {
         for (j=0;j<subframe;j++)
            y[j] += PCM2VAL(x[(j+offset)*C+c]);
      }
   }
}
#endif

void downmix_int(const void *_x, opus_val32 *y, int subframe, int offset, int c1, int c2, int C)
{
   const opus_int16 *x;
   int j;

   x = (const opus_int16 *)_x;
   for (j=0;j<subframe;j++)
      y[j] = x[(j+offset)*C+c1];
   if (c2>-1)
   {
      for (j=0;j<subframe;j++)
         y[j] += x[(j+offset)*C+c2];
   } else if (c2==-2)
   {
      int c;
      for (c=1;c<C;c++)
      {
         for (j=0;j<subframe;j++)
            y[j] += x[(j+offset)*C+c];
      }
   }
}

opus_int32 frame_size_select(opus_int32 frame_size, int variable_duration, opus_int32 Fs)
{
   int new_size;
   if (frame_size<Fs/400)
      return -1;
   if (variable_duration == OPUS_FRAMESIZE_ARG)
      new_size = frame_size;
   else if (variable_duration >= OPUS_FRAMESIZE_2_5_MS && variable_duration <= OPUS_FRAMESIZE_120_MS)
   {
      if (variable_duration <= OPUS_FRAMESIZE_40_MS)
         new_size = (Fs/400)<<(variable_duration-OPUS_FRAMESIZE_2_5_MS);
      else
         new_size = (variable_duration-OPUS_FRAMESIZE_2_5_MS-2)*Fs/50;
   }
   else
      return -1;
   if (new_size>frame_size)
      return -1;
   if (400*new_size!=Fs   && 200*new_size!=Fs   && 100*new_size!=Fs   &&
        50*new_size!=Fs   &&  25*new_size!=Fs   &&  50*new_size!=3*Fs &&
        50*new_size!=4*Fs &&  50*new_size!=5*Fs &&  50*new_size!=6*Fs)
      return -1;
   return new_size;
}

opus_val16 compute_stereo_width(const opus_val16 *pcm, int frame_size, opus_int32 Fs, StereoWidthState *mem)
{
   opus_val32 xx, xy, yy;
   opus_val16 sqrt_xx, sqrt_yy;
   opus_val16 qrrt_xx, qrrt_yy;
   int frame_rate;
   int i;
   opus_val16 short_alpha;

   frame_rate = Fs/frame_size;
   short_alpha = Q15ONE - MULT16_16(25, Q15ONE)/IMAX(50,frame_rate);
   xx=xy=yy=0;
   /* Unroll by 4. The frame size is always a multiple of 4 *except* for
      2.5 ms frames at 12 kHz. Since this setting is very rare (and very
      stupid), we just discard the last two samples. */
   for (i=0;i<frame_size-3;i+=4)
   {
      opus_val32 pxx=0;
      opus_val32 pxy=0;
      opus_val32 pyy=0;
      opus_val16 x, y;
      x = pcm[2*i];
      y = pcm[2*i+1];
      pxx = SHR32(MULT16_16(x,x),2);
      pxy = SHR32(MULT16_16(x,y),2);
      pyy = SHR32(MULT16_16(y,y),2);
      x = pcm[2*i+2];
      y = pcm[2*i+3];
      pxx += SHR32(MULT16_16(x,x),2);
      pxy += SHR32(MULT16_16(x,y),2);
      pyy += SHR32(MULT16_16(y,y),2);
      x = pcm[2*i+4];
      y = pcm[2*i+5];
      pxx += SHR32(MULT16_16(x,x),2);
      pxy += SHR32(MULT16_16(x,y),2);
      pyy += SHR32(MULT16_16(y,y),2);
      x = pcm[2*i+6];
      y = pcm[2*i+7];
      pxx += SHR32(MULT16_16(x,x),2);
      pxy += SHR32(MULT16_16(x,y),2);
      pyy += SHR32(MULT16_16(y,y),2);

      xx += SHR32(pxx, 10);
      xy += SHR32(pxy, 10);
      yy += SHR32(pyy, 10);
   }
#ifndef FIXED_POINT
   if (!(xx < 1e9f) || celt_isnan(xx) || !(yy < 1e9f) || celt_isnan(yy))
   {
      xy = xx = yy = 0;
   }
#endif
   mem->XX += MULT16_32_Q15(short_alpha, xx-mem->XX);
   mem->XY += MULT16_32_Q15(short_alpha, xy-mem->XY);
   mem->YY += MULT16_32_Q15(short_alpha, yy-mem->YY);
   mem->XX = MAX32(0, mem->XX);
   mem->XY = MAX32(0, mem->XY);
   mem->YY = MAX32(0, mem->YY);
   if (MAX32(mem->XX, mem->YY)>QCONST16(8e-4f, 18))
   {
      opus_val16 corr;
      opus_val16 ldiff;
      opus_val16 width;
      sqrt_xx = celt_sqrt(mem->XX);
      sqrt_yy = celt_sqrt(mem->YY);
      qrrt_xx = celt_sqrt(sqrt_xx);
      qrrt_yy = celt_sqrt(sqrt_yy);
      /* Inter-channel correlation */
      mem->XY = MIN32(mem->XY, sqrt_xx*sqrt_yy);
      corr = SHR32(frac_div32(mem->XY,EPSILON+MULT16_16(sqrt_xx,sqrt_yy)),16);
      /* Approximate loudness difference */
      ldiff = MULT16_16(Q15ONE, ABS16(qrrt_xx-qrrt_yy))/(EPSILON+qrrt_xx+qrrt_yy);
      width = MULT16_16_Q15(celt_sqrt(QCONST32(1.f,30)-MULT16_16(corr,corr)), ldiff);
      /* Smoothing over one second */
      mem->smoothed_width += (width-mem->smoothed_width)/frame_rate;
      /* Peak follower */
      mem->max_follower = MAX16(mem->max_follower-QCONST16(.02f,15)/frame_rate, mem->smoothed_width);
   }
   /*printf("%f %f %f %f %f ", corr/(float)Q15ONE, ldiff/(float)Q15ONE, width/(float)Q15ONE, mem->smoothed_width/(float)Q15ONE, mem->max_follower/(float)Q15ONE);*/
   return EXTRACT16(MIN32(Q15ONE, MULT16_16(20, mem->max_follower)));
}

static int decide_fec(int useInBandFEC, int PacketLoss_perc, int last_fec, int mode, int *bandwidth, opus_int32 rate)
{
   int orig_bandwidth;
   if (!useInBandFEC || PacketLoss_perc == 0 || mode == MODE_CELT_ONLY)
      return 0;
   orig_bandwidth = *bandwidth;
   for (;;)
   {
      opus_int32 hysteresis;
      opus_int32 LBRR_rate_thres_bps;
      /* Compute threshold for using FEC at the current bandwidth setting */
      LBRR_rate_thres_bps = fec_thresholds[2*(*bandwidth - OPUS_BANDWIDTH_NARROWBAND)];
      hysteresis = fec_thresholds[2*(*bandwidth - OPUS_BANDWIDTH_NARROWBAND) + 1];
      if (last_fec == 1) LBRR_rate_thres_bps -= hysteresis;
      if (last_fec == 0) LBRR_rate_thres_bps += hysteresis;
      LBRR_rate_thres_bps = silk_SMULWB( silk_MUL( LBRR_rate_thres_bps,
            125 - silk_min( PacketLoss_perc, 25 ) ), SILK_FIX_CONST( 0.01, 16 ) );
      /* If loss <= 5%, we look at whether we have enough rate to enable FEC.
         If loss > 5%, we decrease the bandwidth until we can enable FEC. */
      if (rate > LBRR_rate_thres_bps)
         return 1;
      else if (PacketLoss_perc <= 5)
         return 0;
      else if (*bandwidth > OPUS_BANDWIDTH_NARROWBAND)
         (*bandwidth)--;
      else
         break;
   }
   /* Couldn't find any bandwidth to enable FEC, keep original bandwidth. */
   *bandwidth = orig_bandwidth;
   return 0;
}

static int compute_silk_rate_for_hybrid(int rate, int bandwidth, int frame20ms, int vbr, int fec, int channels) {
   int entry;
   int i;
   int N;
   int silk_rate;
   static int rate_table[][5] = {
  /*  |total| |-------- SILK------------|
              |-- No FEC -| |--- FEC ---|
               10ms   20ms   10ms   20ms */
      {    0,     0,     0,     0,     0},
      {12000, 10000, 10000, 11000, 11000},
      {16000, 13500, 13500, 15000, 15000},
      {20000, 16000, 16000, 18000, 18000},
      {24000, 18000, 18000, 21000, 21000},
      {32000, 22000, 22000, 28000, 28000},
      {64000, 38000, 38000, 50000, 50000}
   };
   /* Do the allocation per-channel. */
   rate /= channels;
   entry = 1 + frame20ms + 2*fec;
   N = sizeof(rate_table)/sizeof(rate_table[0]);
   for (i=1;i<N;i++)
   {
      if (rate_table[i][0] > rate) break;
   }
   if (i == N)
   {
      silk_rate = rate_table[i-1][entry];
      /* For now, just give 50% of the extra bits to SILK. */
      silk_rate += (rate-rate_table[i-1][0])/2;
   } else {
      opus_int32 lo, hi, x0, x1;
      lo = rate_table[i-1][entry];
      hi = rate_table[i][entry];
      x0 = rate_table[i-1][0];
      x1 = rate_table[i][0];
      silk_rate = (lo*(x1-rate) + hi*(rate-x0))/(x1-x0);
   }
   if (!vbr)
   {
      /* Tiny boost to SILK for CBR. We should probably tune this better. */
      silk_rate += 100;
   }
   if (bandwidth==OPUS_BANDWIDTH_SUPERWIDEBAND)
      silk_rate += 300;
   silk_rate *= channels;
   /* Small adjustment for stereo (calibrated for 32 kb/s, haven't tried other bitrates). */
   if (channels == 2 && rate >= 12000)
      silk_rate -= 1000;
   return silk_rate;
}

/* Returns the equivalent bitrate corresponding to 20 ms frames,
   complexity 10 VBR operation. */
static opus_int32 compute_equiv_rate(opus_int32 bitrate, int channels,
      int frame_rate, int vbr, int mode, int complexity, int loss)
{
   opus_int32 equiv;
   equiv = bitrate;
   /* Take into account overhead from smaller frames. */
   if (frame_rate > 50)
      equiv -= (40*channels+20)*(frame_rate - 50);
   /* CBR is about a 8% penalty for both SILK and CELT. */
   if (!vbr)
      equiv -= equiv/12;
   /* Complexity makes about 10% difference (from 0 to 10) in general. */
   equiv = equiv * (90+complexity)/100;
   if (mode == MODE_SILK_ONLY || mode == MODE_HYBRID)
   {
      /* SILK complexity 0-1 uses the non-delayed-decision NSQ, which
         costs about 20%. */
      if (complexity<2)
         equiv = equiv*4/5;
      equiv -= equiv*loss/(6*loss + 10);
   } else if (mode == MODE_CELT_ONLY) {
      /* CELT complexity 0-4 doesn't have the pitch filter, which costs
         about 10%. */
      if (complexity<5)
         equiv = equiv*9/10;
   } else {
      /* Mode not known yet */
      /* Half the SILK loss*/
      equiv -= equiv*loss/(12*loss + 20);
   }
   return equiv;
}

#ifndef DISABLE_FLOAT_API

int is_digital_silence(const opus_val16* pcm, int frame_size, int channels, int lsb_depth)
{
   int silence = 0;
   opus_val32 sample_max = 0;
#ifdef MLP_TRAINING
   return 0;
#endif
   sample_max = celt_maxabs16(pcm, frame_size*channels);

#ifdef FIXED_POINT
   silence = (sample_max == 0);
   (void)lsb_depth;
#else
   silence = (sample_max <= (opus_val16) 1 / (1 << lsb_depth));
#endif

   return silence;
}

#ifdef FIXED_POINT
static opus_val32 compute_frame_energy(const opus_val16 *pcm, int frame_size, int channels, int arch)
{
   int i;
   opus_val32 sample_max;
   int max_shift;
   int shift;
   opus_val32 energy = 0;
   int len = frame_size*channels;
   (void)arch;
   /* Max amplitude in the signal */
   sample_max = celt_maxabs16(pcm, len);

   /* Compute the right shift required in the MAC to avoid an overflow */
   max_shift = celt_ilog2(len);
   shift = IMAX(0, (celt_ilog2(sample_max) << 1) + max_shift - 28);

   /* Compute the energy */
   for (i=0; i<len; i++)
      energy += SHR32(MULT16_16(pcm[i], pcm[i]), shift);

   /* Normalize energy by the frame size and left-shift back to the original position */
   energy /= len;
   energy = SHL32(energy, shift);

   return energy;
}
#else
static opus_val32 compute_frame_energy(const opus_val16 *pcm, int frame_size, int channels, int arch)
{
   int len = frame_size*channels;
   return celt_inner_prod(pcm, pcm, len, arch)/len;
}
#endif

/* Decides if DTX should be turned on (=1) or off (=0) */
static int decide_dtx_mode(float activity_probability,    /* probability that current frame contains speech/music */
                           int *nb_no_activity_frames,    /* number of consecutive frames with no activity */
                           opus_val32 peak_signal_energy, /* peak energy of desired signal detected so far */
                           const opus_val16 *pcm,         /* input pcm signal */
                           int frame_size,                /* frame size */
                           int channels,
                           int is_silence,                 /* only digital silence detected in this frame */
                           int arch
                          )
{
   opus_val32 noise_energy;

   if (!is_silence)
   {
      if (activity_probability < DTX_ACTIVITY_THRESHOLD)  /* is noise */
      {
         noise_energy = compute_frame_energy(pcm, frame_size, channels, arch);

         /* but is sufficiently quiet */
         is_silence = peak_signal_energy >= (PSEUDO_SNR_THRESHOLD * noise_energy);
      }
   }

   if (is_silence)
   {
      /* The number of consecutive DTX frames should be within the allowed bounds */
      (*nb_no_activity_frames)++;

      if (*nb_no_activity_frames > NB_SPEECH_FRAMES_BEFORE_DTX)
      {
         if (*nb_no_activity_frames <= (NB_SPEECH_FRAMES_BEFORE_DTX + MAX_CONSECUTIVE_DTX))
            /* Valid frame for DTX! */
            return 1;
         else
            (*nb_no_activity_frames) = NB_SPEECH_FRAMES_BEFORE_DTX;
      }
   } else
      (*nb_no_activity_frames) = 0;

   return 0;
}

#endif

static opus_int32 encode_multiframe_packet(OpusEncoder *st,
                                           const opus_val16 *pcm,
                                           int nb_frames,
                                           int frame_size,
                                           unsigned char *data,
                                           opus_int32 out_data_bytes,
                                           int to_celt,
                                           int lsb_depth,
                                           int float_api)
{
   int i;
   int ret = 0;
   VARDECL(unsigned char, tmp_data);
   int bak_mode, bak_bandwidth, bak_channels, bak_to_mono;
   VARDECL(OpusRepacketizer, rp);
   int max_header_bytes;
   opus_int32 bytes_per_frame;
   opus_int32 cbr_bytes;
   opus_int32 repacketize_len;
   int tmp_len;
   ALLOC_STACK;

   /* Worst cases:
    * 2 frames: Code 2 with different compressed sizes
    * >2 frames: Code 3 VBR */
   max_header_bytes = nb_frames == 2 ? 3 : (2+(nb_frames-1)*2);

   if (st->use_vbr || st->user_bitrate_bps==OPUS_BITRATE_MAX)
      repacketize_len = out_data_bytes;
   else {
      cbr_bytes = 3*st->bitrate_bps/(3*8*st->Fs/(frame_size*nb_frames));
      repacketize_len = IMIN(cbr_bytes, out_data_bytes);
   }
   bytes_per_frame = IMIN(1276, 1+(repacketize_len-max_header_bytes)/nb_frames);

   ALLOC(tmp_data, nb_frames*bytes_per_frame, unsigned char);
   ALLOC(rp, 1, OpusRepacketizer);
   opus_repacketizer_init(rp);

   bak_mode = st->user_forced_mode;
   bak_bandwidth = st->user_bandwidth;
   bak_channels = st->force_channels;

   st->user_forced_mode = st->mode;
   st->user_bandwidth = st->bandwidth;
   st->force_channels = st->stream_channels;

   bak_to_mono = st->silk_mode.toMono;
   if (bak_to_mono)
      st->force_channels = 1;
   else
      st->prev_channels = st->stream_channels;

   for (i=0;i<nb_frames;i++)
   {
      st->silk_mode.toMono = 0;
      st->nonfinal_frame = i<(nb_frames-1);

      /* When switching from SILK/Hybrid to CELT, only ask for a switch at the last frame */
      if (to_celt && i==nb_frames-1)
         st->user_forced_mode = MODE_CELT_ONLY;

      tmp_len = opus_encode_native(st, pcm+i*(st->channels*frame_size), frame_size,
         tmp_data+i*bytes_per_frame, bytes_per_frame, lsb_depth, NULL, 0, 0, 0, 0,
         NULL, float_api);

      if (tmp_len<0)
      {
         RESTORE_STACK;
         return OPUS_INTERNAL_ERROR;
      }

      ret = opus_repacketizer_cat(rp, tmp_data+i*bytes_per_frame, tmp_len);

      if (ret<0)
      {
         RESTORE_STACK;
         return OPUS_INTERNAL_ERROR;
      }
   }

   ret = opus_repacketizer_out_range_impl(rp, 0, nb_frames, data, repacketize_len, 0, !st->use_vbr);

   if (ret<0)
   {
      RESTORE_STACK;
      return OPUS_INTERNAL_ERROR;
   }

   /* Discard configs that were forced locally for the purpose of repacketization */
   st->user_forced_mode = bak_mode;
   st->user_bandwidth = bak_bandwidth;
   st->force_channels = bak_channels;
   st->silk_mode.toMono = bak_to_mono;

   RESTORE_STACK;
   return ret;
}

static int compute_redundancy_bytes(opus_int32 max_data_bytes, opus_int32 bitrate_bps, int frame_rate, int channels)
{
   int redundancy_bytes_cap;
   int redundancy_bytes;
   opus_int32 redundancy_rate;
   int base_bits;
   opus_int32 available_bits;
   base_bits = (40*channels+20);

   /* Equivalent rate for 5 ms frames. */
   redundancy_rate = bitrate_bps + base_bits*(200 - frame_rate);
   /* For VBR, further increase the bitrate if we can afford it. It's pretty short
      and we'll avoid artefacts. */
   redundancy_rate = 3*redundancy_rate/2;
   redundancy_bytes = redundancy_rate/1600;

   /* Compute the max rate we can use given CBR or VBR with cap. */
   available_bits = max_data_bytes*8 - 2*base_bits;
   redundancy_bytes_cap = (available_bits*240/(240+48000/frame_rate) + base_bits)/8;
   redundancy_bytes = IMIN(redundancy_bytes, redundancy_bytes_cap);
   /* It we can't get enough bits for redundancy to be worth it, rely on the decoder PLC. */
   if (redundancy_bytes > 4 + 8*channels)
      redundancy_bytes = IMIN(257, redundancy_bytes);
   else
      redundancy_bytes = 0;
   return redundancy_bytes;
}

opus_int32 opus_encode_native(OpusEncoder *st, const opus_val16 *pcm, int frame_size,
                unsigned char *data, opus_int32 out_data_bytes, int lsb_depth,
                const void *analysis_pcm, opus_int32 analysis_size, int c1, int c2,
                int analysis_channels, downmix_func downmix, int float_api)
{
    void *silk_enc;
    CELTEncoder *celt_enc;
    int i;
    int ret=0;
    opus_int32 nBytes;
    ec_enc enc;
    int bytes_target;
    int prefill=0;
    int start_band = 0;
    int redundancy = 0;
    int redundancy_bytes = 0; /* Number of bytes to use for redundancy frame */
    int celt_to_silk = 0;
    VARDECL(opus_val16, pcm_buf);
    int nb_compr_bytes;
    int to_celt = 0;
    opus_uint32 redundant_rng = 0;
    int cutoff_Hz, hp_freq_smth1;
    int voice_est; /* Probability of voice in Q7 */
    opus_int32 equiv_rate;
    int delay_compensation;
    int frame_rate;
    opus_int32 max_rate; /* Max bitrate we're allowed to use */
    int curr_bandwidth;
    opus_val16 HB_gain;
    opus_int32 max_data_bytes; /* Max number of bytes we're allowed to use */
    int total_buffer;
    opus_val16 stereo_width;
    const CELTMode *celt_mode;
#ifndef DISABLE_FLOAT_API
    AnalysisInfo analysis_info;
    int analysis_read_pos_bak=-1;
    int analysis_read_subframe_bak=-1;
    int is_silence = 0;
#endif
    VARDECL(opus_val16, tmp_prefill);

    ALLOC_STACK;

    max_data_bytes = IMIN(1276, out_data_bytes);

    st->rangeFinal = 0;
    if (frame_size <= 0 || max_data_bytes <= 0)
    {
       RESTORE_STACK;
       return OPUS_BAD_ARG;
    }

    /* Cannot encode 100 ms in 1 byte */
    if (max_data_bytes==1 && st->Fs==(frame_size*10))
    {
      RESTORE_STACK;
      return OPUS_BUFFER_TOO_SMALL;
    }

    silk_enc = (char*)st+st->silk_enc_offset;
    celt_enc = (CELTEncoder*)((char*)st+st->celt_enc_offset);
    if (st->application == OPUS_APPLICATION_RESTRICTED_LOWDELAY)
       delay_compensation = 0;
    else
       delay_compensation = st->delay_compensation;

    lsb_depth = IMIN(lsb_depth, st->lsb_depth);

    celt_encoder_ctl(celt_enc, CELT_GET_MODE(&celt_mode));
#ifndef DISABLE_FLOAT_API
    analysis_info.valid = 0;
#ifdef FIXED_POINT
    if (st->silk_mode.complexity >= 10 && st->Fs>=16000)
#else
    if (st->silk_mode.complexity >= 7 && st->Fs>=16000)
#endif
    {
       is_silence = is_digital_silence(pcm, frame_size, st->channels, lsb_depth);
       analysis_read_pos_bak = st->analysis.read_pos;
       analysis_read_subframe_bak = st->analysis.read_subframe;
       run_analysis(&st->analysis, celt_mode, analysis_pcm, analysis_size, frame_size,
             c1, c2, analysis_channels, st->Fs,
             lsb_depth, downmix, &analysis_info);

       /* Track the peak signal energy */
       if (!is_silence && analysis_info.activity_probability > DTX_ACTIVITY_THRESHOLD)
          st->peak_signal_energy = MAX32(MULT16_32_Q15(QCONST16(0.999f, 15), st->peak_signal_energy),
                compute_frame_energy(pcm, frame_size, st->channels, st->arch));
    } else if (st->analysis.initialized) {
       tonality_analysis_reset(&st->analysis);
    }
#else
    (void)analysis_pcm;
    (void)analysis_size;
    (void)c1;
    (void)c2;
    (void)analysis_channels;
    (void)downmix;
#endif

#ifndef DISABLE_FLOAT_API
    /* Reset voice_ratio if this frame is not silent or if analysis is disabled.
     * Otherwise, preserve voice_ratio from the last non-silent frame */
    if (!is_silence)
      st->voice_ratio = -1;

    st->detected_bandwidth = 0;
    if (analysis_info.valid)
    {
       int analysis_bandwidth;
       if (st->signal_type == OPUS_AUTO)
       {
          float prob;
          if (st->prev_mode == 0)
             prob = analysis_info.music_prob;
          else if (st->prev_mode == MODE_CELT_ONLY)
             prob = analysis_info.music_prob_max;
          else
             prob = analysis_info.music_prob_min;
          st->voice_ratio = (int)floor(.5+100*(1-prob));
       }

       analysis_bandwidth = analysis_info.bandwidth;
       if (analysis_bandwidth<=12)
          st->detected_bandwidth = OPUS_BANDWIDTH_NARROWBAND;
       else if (analysis_bandwidth<=14)
          st->detected_bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
       else if (analysis_bandwidth<=16)
          st->detected_bandwidth = OPUS_BANDWIDTH_WIDEBAND;
       else if (analysis_bandwidth<=18)
          st->detected_bandwidth = OPUS_BANDWIDTH_SUPERWIDEBAND;
       else
          st->detected_bandwidth = OPUS_BANDWIDTH_FULLBAND;
    }
#else
    st->voice_ratio = -1;
#endif

    if (st->channels==2 && st->force_channels!=1)
       stereo_width = compute_stereo_width(pcm, frame_size, st->Fs, &st->width_mem);
    else
       stereo_width = 0;
    total_buffer = delay_compensation;
    st->bitrate_bps = user_bitrate_to_bitrate(st, frame_size, max_data_bytes);

    frame_rate = st->Fs/frame_size;
    if (!st->use_vbr)
    {
       int cbrBytes;
       /* Multiply by 12 to make sure the division is exact. */
       int frame_rate12 = 12*st->Fs/frame_size;
       /* We need to make sure that "int" values always fit in 16 bits. */
       cbrBytes = IMIN( (12*st->bitrate_bps/8 + frame_rate12/2)/frame_rate12, max_data_bytes);
       st->bitrate_bps = cbrBytes*(opus_int32)frame_rate12*8/12;
       /* Make sure we provide at least one byte to avoid failing. */
       max_data_bytes = IMAX(1, cbrBytes);
    }
    if (max_data_bytes<3 || st->bitrate_bps < 3*frame_rate*8
       || (frame_rate<50 && (max_data_bytes*frame_rate<300 || st->bitrate_bps < 2400)))
    {
       /*If the space is too low to do something useful, emit 'PLC' frames.*/
       int tocmode = st->mode;
       int bw = st->bandwidth == 0 ? OPUS_BANDWIDTH_NARROWBAND : st->bandwidth;
       int packet_code = 0;
       int num_multiframes = 0;

       if (tocmode==0)
          tocmode = MODE_SILK_ONLY;
       if (frame_rate>100)
          tocmode = MODE_CELT_ONLY;
       /* 40 ms -> 2 x 20 ms if in CELT_ONLY or HYBRID mode */
       if (frame_rate==25 && tocmode!=MODE_SILK_ONLY)
       {
          frame_rate = 50;
          packet_code = 1;
       }

       /* >= 60 ms frames */
       if (frame_rate<=16)
       {
          /* 1 x 60 ms, 2 x 40 ms, 2 x 60 ms */
          if (out_data_bytes==1 || (tocmode==MODE_SILK_ONLY && frame_rate!=10))
          {
             tocmode = MODE_SILK_ONLY;

             packet_code = frame_rate <= 12;
             frame_rate = frame_rate == 12 ? 25 : 16;
          }
          else
          {
             num_multiframes = 50/frame_rate;
             frame_rate = 50;
             packet_code = 3;
          }
       }

       if(tocmode==MODE_SILK_ONLY&&bw>OPUS_BANDWIDTH_WIDEBAND)
          bw=OPUS_BANDWIDTH_WIDEBAND;
       else if (tocmode==MODE_CELT_ONLY&&bw==OPUS_BANDWIDTH_MEDIUMBAND)
          bw=OPUS_BANDWIDTH_NARROWBAND;
       else if (tocmode==MODE_HYBRID&&bw<=OPUS_BANDWIDTH_SUPERWIDEBAND)
          bw=OPUS_BANDWIDTH_SUPERWIDEBAND;

       data[0] = gen_toc(tocmode, frame_rate, bw, st->stream_channels);
       data[0] |= packet_code;

       ret = packet_code <= 1 ? 1 : 2;

       max_data_bytes = IMAX(max_data_bytes, ret);

       if (packet_code==3)
          data[1] = num_multiframes;

       if (!st->use_vbr)
       {
          ret = opus_packet_pad(data, ret, max_data_bytes);
          if (ret == OPUS_OK)
             ret = max_data_bytes;
          else
             ret = OPUS_INTERNAL_ERROR;
       }
       RESTORE_STACK;
       return ret;
    }
    max_rate = frame_rate*max_data_bytes*8;

    /* Equivalent 20-ms rate for mode/channel/bandwidth decisions */
    equiv_rate = compute_equiv_rate(st->bitrate_bps, st->channels, st->Fs/frame_size,
          st->use_vbr, 0, st->silk_mode.complexity, st->silk_mode.packetLossPercentage);

    if (st->signal_type == OPUS_SIGNAL_VOICE)
       voice_est = 127;
    else if (st->signal_type == OPUS_SIGNAL_MUSIC)
       voice_est = 0;
    else if (st->voice_ratio >= 0)
    {
       voice_est = st->voice_ratio*327>>8;
       /* For AUDIO, never be more than 90% confident of having speech */
       if (st->application == OPUS_APPLICATION_AUDIO)
          voice_est = IMIN(voice_est, 115);
    } else if (st->application == OPUS_APPLICATION_VOIP)
       voice_est = 115;
    else
       voice_est = 48;

    if (st->force_channels!=OPUS_AUTO && st->channels == 2)
    {
        st->stream_channels = st->force_channels;
    } else {
#ifdef FUZZING
       /* Random mono/stereo decision */
       if (st->channels == 2 && (rand()&0x1F)==0)
          st->stream_channels = 3-st->stream_channels;
#else
       /* Rate-dependent mono-stereo decision */
       if (st->channels == 2)
       {
          opus_int32 stereo_threshold;
          stereo_threshold = stereo_music_threshold + ((voice_est*voice_est*(stereo_voice_threshold-stereo_music_threshold))>>14);
          if (st->stream_channels == 2)
             stereo_threshold -= 1000;
          else
             stereo_threshold += 1000;
          st->stream_channels = (equiv_rate > stereo_threshold) ? 2 : 1;
       } else {
          st->stream_channels = st->channels;
       }
#endif
    }
    /* Update equivalent rate for channels decision. */
    equiv_rate = compute_equiv_rate(st->bitrate_bps, st->stream_channels, st->Fs/frame_size,
          st->use_vbr, 0, st->silk_mode.complexity, st->silk_mode.packetLossPercentage);

    /* Allow SILK DTX if DTX is enabled but the generalized DTX cannot be used,
       e.g. because of the complexity setting or sample rate. */
#ifndef DISABLE_FLOAT_API
    st->silk_mode.useDTX = st->use_dtx && !(analysis_info.valid || is_silence);
#else
    st->silk_mode.useDTX = st->use_dtx;
#endif

    /* Mode selection depending on application and signal type */
    if (st->application == OPUS_APPLICATION_RESTRICTED_LOWDELAY)
    {
       st->mode = MODE_CELT_ONLY;
    } else if (st->user_forced_mode == OPUS_AUTO)
    {
#ifdef FUZZING
       /* Random mode switching */
       if ((rand()&0xF)==0)
       {
          if ((rand()&0x1)==0)
             st->mode = MODE_CELT_ONLY;
          else
             st->mode = MODE_SILK_ONLY;
       } else {
          if (st->prev_mode==MODE_CELT_ONLY)
             st->mode = MODE_CELT_ONLY;
          else
             st->mode = MODE_SILK_ONLY;
       }
#else
       opus_int32 mode_voice, mode_music;
       opus_int32 threshold;

       /* Interpolate based on stereo width */
       mode_voice = (opus_int32)(MULT16_32_Q15(Q15ONE-stereo_width,mode_thresholds[0][0])
             + MULT16_32_Q15(stereo_width,mode_thresholds[1][0]));
       mode_music = (opus_int32)(MULT16_32_Q15(Q15ONE-stereo_width,mode_thresholds[1][1])
             + MULT16_32_Q15(stereo_width,mode_thresholds[1][1]));
       /* Interpolate based on speech/music probability */
       threshold = mode_music + ((voice_est*voice_est*(mode_voice-mode_music))>>14);
       /* Bias towards SILK for VoIP because of some useful features */
       if (st->application == OPUS_APPLICATION_VOIP)
          threshold += 8000;

       /*printf("%f %d\n", stereo_width/(float)Q15ONE, threshold);*/
       /* Hysteresis */
       if (st->prev_mode == MODE_CELT_ONLY)
           threshold -= 4000;
       else if (st->prev_mode>0)
           threshold += 4000;

       st->mode = (equiv_rate >= threshold) ? MODE_CELT_ONLY: MODE_SILK_ONLY;

       /* When FEC is enabled and there's enough packet loss, use SILK */
       if (st->silk_mode.useInBandFEC && st->silk_mode.packetLossPercentage > (128-voice_est)>>4)
          st->mode = MODE_SILK_ONLY;
       /* When encoding voice and DTX is enabled but the generalized DTX cannot be used,
          use SILK in order to make use of its DTX. */
       if (st->silk_mode.useDTX && voice_est > 100)
          st->mode = MODE_SILK_ONLY;
#endif

       /* If max_data_bytes represents less than 6 kb/s, switch to CELT-only mode */
       if (max_data_bytes < (frame_rate > 50 ? 9000 : 6000)*frame_size / (st->Fs * 8))
          st->mode = MODE_CELT_ONLY;
    } else {
       st->mode = st->user_forced_mode;
    }

    /* Override the chosen mode to make sure we meet the requested frame size */
    if (st->mode != MODE_CELT_ONLY && frame_size < st->Fs/100)
       st->mode = MODE_CELT_ONLY;
    if (st->lfe)
       st->mode = MODE_CELT_ONLY;

    if (st->prev_mode > 0 &&
        ((st->mode != MODE_CELT_ONLY && st->prev_mode == MODE_CELT_ONLY) ||
    (st->mode == MODE_CELT_ONLY && st->prev_mode != MODE_CELT_ONLY)))
    {
        redundancy = 1;
        celt_to_silk = (st->mode != MODE_CELT_ONLY);
        if (!celt_to_silk)
        {
            /* Switch to SILK/hybrid if frame size is 10 ms or more*/
            if (frame_size >= st->Fs/100)
            {
                st->mode = st->prev_mode;
                to_celt = 1;
            } else {
                redundancy=0;
            }
        }
    }

    /* When encoding multiframes, we can ask for a switch to CELT only in the last frame. This switch
     * is processed above as the requested mode shouldn't interrupt stereo->mono transition. */
    if (st->stream_channels == 1 && st->prev_channels ==2 && st->silk_mode.toMono==0
          && st->mode != MODE_CELT_ONLY && st->prev_mode != MODE_CELT_ONLY)
    {
       /* Delay stereo->mono transition by two frames so that SILK can do a smooth downmix */
       st->silk_mode.toMono = 1;
       st->stream_channels = 2;
    } else {
       st->silk_mode.toMono = 0;
    }

    /* Update equivalent rate with mode decision. */
    equiv_rate = compute_equiv_rate(st->bitrate_bps, st->stream_channels, st->Fs/frame_size,
          st->use_vbr, st->mode, st->silk_mode.complexity, st->silk_mode.packetLossPercentage);

    if (st->mode != MODE_CELT_ONLY && st->prev_mode == MODE_CELT_ONLY)
    {
        silk_EncControlStruct dummy;
        silk_InitEncoder( silk_enc, st->arch, &dummy);
        prefill=1;
    }

    /* Automatic (rate-dependent) bandwidth selection */
    if (st->mode == MODE_CELT_ONLY || st->first || st->silk_mode.allowBandwidthSwitch)
    {
        const opus_int32 *voice_bandwidth_thresholds, *music_bandwidth_thresholds;
        opus_int32 bandwidth_thresholds[8];
        int bandwidth = OPUS_BANDWIDTH_FULLBAND;

        if (st->channels==2 && st->force_channels!=1)
        {
           voice_bandwidth_thresholds = stereo_voice_bandwidth_thresholds;
           music_bandwidth_thresholds = stereo_music_bandwidth_thresholds;
        } else {
           voice_bandwidth_thresholds = mono_voice_bandwidth_thresholds;
           music_bandwidth_thresholds = mono_music_bandwidth_thresholds;
        }
        /* Interpolate bandwidth thresholds depending on voice estimation */
        for (i=0;i<8;i++)
        {
           bandwidth_thresholds[i] = music_bandwidth_thresholds[i]
                    + ((voice_est*voice_est*(voice_bandwidth_thresholds[i]-music_bandwidth_thresholds[i]))>>14);
        }
        do {
            int threshold, hysteresis;
            threshold = bandwidth_thresholds[2*(bandwidth-OPUS_BANDWIDTH_MEDIUMBAND)];
            hysteresis = bandwidth_thresholds[2*(bandwidth-OPUS_BANDWIDTH_MEDIUMBAND)+1];
            if (!st->first)
            {
                if (st->auto_bandwidth >= bandwidth)
                    threshold -= hysteresis;
                else
                    threshold += hysteresis;
            }
            if (equiv_rate >= threshold)
                break;
        } while (--bandwidth>OPUS_BANDWIDTH_NARROWBAND);
        /* We don't use mediumband anymore, except when explicitly requested or during
           mode transitions. */
        if (bandwidth == OPUS_BANDWIDTH_MEDIUMBAND)
           bandwidth = OPUS_BANDWIDTH_WIDEBAND;
        st->bandwidth = st->auto_bandwidth = bandwidth;
        /* Prevents any transition to SWB/FB until the SILK layer has fully
           switched to WB mode and turned the variable LP filter off */
        if (!st->first && st->mode != MODE_CELT_ONLY && !st->silk_mode.inWBmodeWithoutVariableLP && st->bandwidth > OPUS_BANDWIDTH_WIDEBAND)
            st->bandwidth = OPUS_BANDWIDTH_WIDEBAND;
    }

    if (st->bandwidth>st->max_bandwidth)
       st->bandwidth = st->max_bandwidth;

    if (st->user_bandwidth != OPUS_AUTO)
        st->bandwidth = st->user_bandwidth;

    /* This prevents us from using hybrid at unsafe CBR/max rates */
    if (st->mode != MODE_CELT_ONLY && max_rate < 15000)
    {
       st->bandwidth = IMIN(st->bandwidth, OPUS_BANDWIDTH_WIDEBAND);
    }

    /* Prevents Opus from wasting bits on frequencies that are above
       the Nyquist rate of the input signal */
    if (st->Fs <= 24000 && st->bandwidth > OPUS_BANDWIDTH_SUPERWIDEBAND)
        st->bandwidth = OPUS_BANDWIDTH_SUPERWIDEBAND;
    if (st->Fs <= 16000 && st->bandwidth > OPUS_BANDWIDTH_WIDEBAND)
        st->bandwidth = OPUS_BANDWIDTH_WIDEBAND;
    if (st->Fs <= 12000 && st->bandwidth > OPUS_BANDWIDTH_MEDIUMBAND)
        st->bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
    if (st->Fs <= 8000 && st->bandwidth > OPUS_BANDWIDTH_NARROWBAND)
        st->bandwidth = OPUS_BANDWIDTH_NARROWBAND;
#ifndef DISABLE_FLOAT_API
    /* Use detected bandwidth to reduce the encoded bandwidth. */
    if (st->detected_bandwidth && st->user_bandwidth == OPUS_AUTO)
    {
       int min_detected_bandwidth;
       /* Makes bandwidth detection more conservative just in case the detector
          gets it wrong when we could have coded a high bandwidth transparently.
          When operating in SILK/hybrid mode, we don't go below wideband to avoid
          more complicated switches that require redundancy. */
       if (equiv_rate <= 18000*st->stream_channels && st->mode == MODE_CELT_ONLY)
          min_detected_bandwidth = OPUS_BANDWIDTH_NARROWBAND;
       else if (equiv_rate <= 24000*st->stream_channels && st->mode == MODE_CELT_ONLY)
          min_detected_bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
       else if (equiv_rate <= 30000*st->stream_channels)
          min_detected_bandwidth = OPUS_BANDWIDTH_WIDEBAND;
       else if (equiv_rate <= 44000*st->stream_channels)
          min_detected_bandwidth = OPUS_BANDWIDTH_SUPERWIDEBAND;
       else
          min_detected_bandwidth = OPUS_BANDWIDTH_FULLBAND;

       st->detected_bandwidth = IMAX(st->detected_bandwidth, min_detected_bandwidth);
       st->bandwidth = IMIN(st->bandwidth, st->detected_bandwidth);
    }
#endif
    st->silk_mode.LBRR_coded = decide_fec(st->silk_mode.useInBandFEC, st->silk_mode.packetLossPercentage,
          st->silk_mode.LBRR_coded, st->mode, &st->bandwidth, equiv_rate);
    celt_encoder_ctl(celt_enc, OPUS_SET_LSB_DEPTH(lsb_depth));

    /* CELT mode doesn't support mediumband, use wideband instead */
    if (st->mode == MODE_CELT_ONLY && st->bandwidth == OPUS_BANDWIDTH_MEDIUMBAND)
        st->bandwidth = OPUS_BANDWIDTH_WIDEBAND;
    if (st->lfe)
       st->bandwidth = OPUS_BANDWIDTH_NARROWBAND;

    curr_bandwidth = st->bandwidth;

    /* Chooses the appropriate mode for speech
       *NEVER* switch to/from CELT-only mode here as this will invalidate some assumptions */
    if (st->mode == MODE_SILK_ONLY && curr_bandwidth > OPUS_BANDWIDTH_WIDEBAND)
        st->mode = MODE_HYBRID;
    if (st->mode == MODE_HYBRID && curr_bandwidth <= OPUS_BANDWIDTH_WIDEBAND)
        st->mode = MODE_SILK_ONLY;

    /* Can't support higher than >60 ms frames, and >20 ms when in Hybrid or CELT-only modes */
    if ((frame_size > st->Fs/50 && (st->mode != MODE_SILK_ONLY)) || frame_size > 3*st->Fs/50)
    {
       int enc_frame_size;
       int nb_frames;

       if (st->mode == MODE_SILK_ONLY)
       {
         if (frame_size == 2*st->Fs/25)  /* 80 ms -> 2x 40 ms */
           enc_frame_size = st->Fs/25;
         else if (frame_size == 3*st->Fs/25)  /* 120 ms -> 2x 60 ms */
           enc_frame_size = 3*st->Fs/50;
         else                            /* 100 ms -> 5x 20 ms */
           enc_frame_size = st->Fs/50;
       }
       else
         enc_frame_size = st->Fs/50;

       nb_frames = frame_size/enc_frame_size;

#ifndef DISABLE_FLOAT_API
       if (analysis_read_pos_bak!= -1)
       {
          st->analysis.read_pos = analysis_read_pos_bak;
          st->analysis.read_subframe = analysis_read_subframe_bak;
       }
#endif

       ret = encode_multiframe_packet(st, pcm, nb_frames, enc_frame_size, data,
                                      out_data_bytes, to_celt, lsb_depth, float_api);

       RESTORE_STACK;
       return ret;
    }

    /* For the first frame at a new SILK bandwidth */
    if (st->silk_bw_switch)
    {
       redundancy = 1;
       celt_to_silk = 1;
       st->silk_bw_switch = 0;
       /* Do a prefill without reseting the sampling rate control. */
       prefill=2;
    }

    /* If we decided to go with CELT, make sure redundancy is off, no matter what
       we decided earlier. */
    if (st->mode == MODE_CELT_ONLY)
        redundancy = 0;

    if (redundancy)
    {
       redundancy_bytes = compute_redundancy_bytes(max_data_bytes, st->bitrate_bps, frame_rate, st->stream_channels);
       if (redundancy_bytes == 0)
          redundancy = 0;
    }

    /* printf("%d %d %d %d\n", st->bitrate_bps, st->stream_channels, st->mode, curr_bandwidth); */
    bytes_target = IMIN(max_data_bytes-redundancy_bytes, st->bitrate_bps * frame_size / (st->Fs * 8)) - 1;

    data += 1;

    ec_enc_init(&enc, data, max_data_bytes-1);

    ALLOC(pcm_buf, (total_buffer+frame_size)*st->channels, opus_val16);
    OPUS_COPY(pcm_buf, &st->delay_buffer[(st->encoder_buffer-total_buffer)*st->channels], total_buffer*st->channels);

    if (st->mode == MODE_CELT_ONLY)
       hp_freq_smth1 = silk_LSHIFT( silk_lin2log( VARIABLE_HP_MIN_CUTOFF_HZ ), 8 );
    else
       hp_freq_smth1 = ((silk_encoder*)silk_enc)->state_Fxx[0].sCmn.variable_HP_smth1_Q15;

    st->variable_HP_smth2_Q15 = silk_SMLAWB( st->variable_HP_smth2_Q15,
          hp_freq_smth1 - st->variable_HP_smth2_Q15, SILK_FIX_CONST( VARIABLE_HP_SMTH_COEF2, 16 ) );

    /* convert from log scale to Hertz */
    cutoff_Hz = silk_log2lin( silk_RSHIFT( st->variable_HP_smth2_Q15, 8 ) );

    if (st->application == OPUS_APPLICATION_VOIP)
    {
       hp_cutoff(pcm, cutoff_Hz, &pcm_buf[total_buffer*st->channels], st->hp_mem, frame_size, st->channels, st->Fs, st->arch);
    } else {
       dc_reject(pcm, 3, &pcm_buf[total_buffer*st->channels], st->hp_mem, frame_size, st->channels, st->Fs);
    }
#ifndef FIXED_POINT
    if (float_api)
    {
       opus_val32 sum;
       sum = celt_inner_prod(&pcm_buf[total_buffer*st->channels], &pcm_buf[total_buffer*st->channels], frame_size*st->channels, st->arch);
       /* This should filter out both NaNs and ridiculous signals that could
          cause NaNs further down. */
       if (!(sum < 1e9f) || celt_isnan(sum))
       {
          OPUS_CLEAR(&pcm_buf[total_buffer*st->channels], frame_size*st->channels);
          st->hp_mem[0] = st->hp_mem[1] = st->hp_mem[2] = st->hp_mem[3] = 0;
       }
    }
#endif


    /* SILK processing */
    HB_gain = Q15ONE;
    if (st->mode != MODE_CELT_ONLY)
    {
        opus_int32 total_bitRate, celt_rate;
        opus_int activity;
#ifdef FIXED_POINT
       const opus_int16 *pcm_silk;
#else
       VARDECL(opus_int16, pcm_silk);
       ALLOC(pcm_silk, st->channels*frame_size, opus_int16);
#endif

        activity = VAD_NO_DECISION;
#ifndef DISABLE_FLOAT_API
        if( analysis_info.valid ) {
            /* Inform SILK about the Opus VAD decision */
            activity = ( analysis_info.activity_probability >= DTX_ACTIVITY_THRESHOLD );
        }
#endif

        /* Distribute bits between SILK and CELT */
        total_bitRate = 8 * bytes_target * frame_rate;
        if( st->mode == MODE_HYBRID ) {
            /* Base rate for SILK */
            st->silk_mode.bitRate = compute_silk_rate_for_hybrid(total_bitRate,
                  curr_bandwidth, st->Fs == 50 * frame_size, st->use_vbr, st->silk_mode.LBRR_coded,
                  st->stream_channels);
            if (!st->energy_masking)
            {
               /* Increasingly attenuate high band when it gets allocated fewer bits */
               celt_rate = total_bitRate - st->silk_mode.bitRate;
               HB_gain = Q15ONE - SHR32(celt_exp2(-celt_rate * QCONST16(1.f/1024, 10)), 1);
            }
        } else {
            /* SILK gets all bits */
            st->silk_mode.bitRate = total_bitRate;
        }

        /* Surround masking for SILK */
        if (st->energy_masking && st->use_vbr && !st->lfe)
        {
           opus_val32 mask_sum=0;
           opus_val16 masking_depth;
           opus_int32 rate_offset;
           int c;
           int end = 17;
           opus_int16 srate = 16000;
           if (st->bandwidth == OPUS_BANDWIDTH_NARROWBAND)
           {
              end = 13;
              srate = 8000;
           } else if (st->bandwidth == OPUS_BANDWIDTH_MEDIUMBAND)
           {
              end = 15;
              srate = 12000;
           }
           for (c=0;c<st->channels;c++)
           {
              for(i=0;i<end;i++)
              {
                 opus_val16 mask;
                 mask = MAX16(MIN16(st->energy_masking[21*c+i],
                        QCONST16(.5f, DB_SHIFT)), -QCONST16(2.0f, DB_SHIFT));
                 if (mask > 0)
                    mask = HALF16(mask);
                 mask_sum += mask;
              }
           }
           /* Conservative rate reduction, we cut the masking in half */
           masking_depth = mask_sum / end*st->channels;
           masking_depth += QCONST16(.2f, DB_SHIFT);
           rate_offset = (opus_int32)PSHR32(MULT16_16(srate, masking_depth), DB_SHIFT);
           rate_offset = MAX32(rate_offset, -2*st->silk_mode.bitRate/3);
           /* Split the rate change between the SILK and CELT part for hybrid. */
           if (st->bandwidth==OPUS_BANDWIDTH_SUPERWIDEBAND || st->bandwidth==OPUS_BANDWIDTH_FULLBAND)
              st->silk_mode.bitRate += 3*rate_offset/5;
           else
              st->silk_mode.bitRate += rate_offset;
        }

        st->silk_mode.payloadSize_ms = 1000 * frame_size / st->Fs;
        st->silk_mode.nChannelsAPI = st->channels;
        st->silk_mode.nChannelsInternal = st->stream_channels;
        if (curr_bandwidth == OPUS_BANDWIDTH_NARROWBAND) {
            st->silk_mode.desiredInternalSampleRate = 8000;
        } else if (curr_bandwidth == OPUS_BANDWIDTH_MEDIUMBAND) {
            st->silk_mode.desiredInternalSampleRate = 12000;
        } else {
            celt_assert( st->mode == MODE_HYBRID || curr_bandwidth == OPUS_BANDWIDTH_WIDEBAND );
            st->silk_mode.desiredInternalSampleRate = 16000;
        }
        if( st->mode == MODE_HYBRID ) {
            /* Don't allow bandwidth reduction at lowest bitrates in hybrid mode */
            st->silk_mode.minInternalSampleRate = 16000;
        } else {
            st->silk_mode.minInternalSampleRate = 8000;
        }

        st->silk_mode.maxInternalSampleRate = 16000;
        if (st->mode == MODE_SILK_ONLY)
        {
           opus_int32 effective_max_rate = max_rate;
           if (frame_rate > 50)
              effective_max_rate = effective_max_rate*2/3;
           if (effective_max_rate < 8000)
           {
              st->silk_mode.maxInternalSampleRate = 12000;
              st->silk_mode.desiredInternalSampleRate = IMIN(12000, st->silk_mode.desiredInternalSampleRate);
           }
           if (effective_max_rate < 7000)
           {
              st->silk_mode.maxInternalSampleRate = 8000;
              st->silk_mode.desiredInternalSampleRate = IMIN(8000, st->silk_mode.desiredInternalSampleRate);
           }
        }

        st->silk_mode.useCBR = !st->use_vbr;

        /* Call SILK encoder for the low band */

        /* Max bits for SILK, counting ToC, redundancy bytes, and optionally redundancy. */
        st->silk_mode.maxBits = (max_data_bytes-1)*8;
        if (redundancy && redundancy_bytes >= 2)
        {
           /* Counting 1 bit for redundancy position and 20 bits for flag+size (only for hybrid). */
           st->silk_mode.maxBits -= redundancy_bytes*8 + 1;
           if (st->mode == MODE_HYBRID)
              st->silk_mode.maxBits -= 20;
        }
        if (st->silk_mode.useCBR)
        {
           if (st->mode == MODE_HYBRID)
           {
              st->silk_mode.maxBits = IMIN(st->silk_mode.maxBits, st->silk_mode.bitRate * frame_size / st->Fs);
           }
        } else {
           /* Constrained VBR. */
           if (st->mode == MODE_HYBRID)
           {
              /* Compute SILK bitrate corresponding to the max total bits available */
              opus_int32 maxBitRate = compute_silk_rate_for_hybrid(st->silk_mode.maxBits*st->Fs / frame_size,
                    curr_bandwidth, st->Fs == 50 * frame_size, st->use_vbr, st->silk_mode.LBRR_coded,
                    st->stream_channels);
              st->silk_mode.maxBits = maxBitRate * frame_size / st->Fs;
           }
        }

        if (prefill)
        {
            opus_int32 zero=0;
            int prefill_offset;
            /* Use a smooth onset for the SILK prefill to avoid the encoder trying to encode
               a discontinuity. The exact location is what we need to avoid leaving any "gap"
               in the audio when mixing with the redundant CELT frame. Here we can afford to
               overwrite st->delay_buffer because the only thing that uses it before it gets
               rewritten is tmp_prefill[] and even then only the part after the ramp really
               gets used (rather than sent to the encoder and discarded) */
            prefill_offset = st->channels*(st->encoder_buffer-st->delay_compensation-st->Fs/400);
            gain_fade(st->delay_buffer+prefill_offset, st->delay_buffer+prefill_offset,
                  0, Q15ONE, celt_mode->overlap, st->Fs/400, st->channels, celt_mode->window, st->Fs);
            OPUS_CLEAR(st->delay_buffer, prefill_offset);
#ifdef FIXED_POINT
            pcm_silk = st->delay_buffer;
#else
            for (i=0;i<st->encoder_buffer*st->channels;i++)
                pcm_silk[i] = FLOAT2INT16(st->delay_buffer[i]);
#endif
            silk_Encode( silk_enc, &st->silk_mode, pcm_silk, st->encoder_buffer, NULL, &zero, prefill, activity );
            /* Prevent a second switch in the real encode call. */
            st->silk_mode.opusCanSwitch = 0;
        }

#ifdef FIXED_POINT
        pcm_silk = pcm_buf+total_buffer*st->channels;
#else
        for (i=0;i<frame_size*st->channels;i++)
            pcm_silk[i] = FLOAT2INT16(pcm_buf[total_buffer*st->channels + i]);
#endif
        ret = silk_Encode( silk_enc, &st->silk_mode, pcm_silk, frame_size, &enc, &nBytes, 0, activity );
        if( ret ) {
            /*fprintf (stderr, "SILK encode error: %d\n", ret);*/
            /* Handle error */
           RESTORE_STACK;
           return OPUS_INTERNAL_ERROR;
        }

        /* Extract SILK internal bandwidth for signaling in first byte */
        if( st->mode == MODE_SILK_ONLY ) {
            if( st->silk_mode.internalSampleRate == 8000 ) {
               curr_bandwidth = OPUS_BANDWIDTH_NARROWBAND;
            } else if( st->silk_mode.internalSampleRate == 12000 ) {
               curr_bandwidth = OPUS_BANDWIDTH_MEDIUMBAND;
            } else if( st->silk_mode.internalSampleRate == 16000 ) {
               curr_bandwidth = OPUS_BANDWIDTH_WIDEBAND;
            }
        } else {
            celt_assert( st->silk_mode.internalSampleRate == 16000 );
        }

        st->silk_mode.opusCanSwitch = st->silk_mode.switchReady && !st->nonfinal_frame;

        if (nBytes==0)
        {
           st->rangeFinal = 0;
           data[-1] = gen_toc(st->mode, st->Fs/frame_size, curr_bandwidth, st->stream_channels);
           RESTORE_STACK;
           return 1;
        }

        /* FIXME: How do we allocate the redundancy for CBR? */
        if (st->silk_mode.opusCanSwitch)
        {
           redundancy_bytes = compute_redundancy_bytes(max_data_bytes, st->bitrate_bps, frame_rate, st->stream_channels);
           redundancy = (redundancy_bytes != 0);
           celt_to_silk = 0;
           st->silk_bw_switch = 1;
        }
    }

    /* CELT processing */
    {
        int endband=21;

        switch(curr_bandwidth)
        {
            case OPUS_BANDWIDTH_NARROWBAND:
                endband = 13;
                break;
            case OPUS_BANDWIDTH_MEDIUMBAND:
            case OPUS_BANDWIDTH_WIDEBAND:
                endband = 17;
                break;
            case OPUS_BANDWIDTH_SUPERWIDEBAND:
                endband = 19;
                break;
            case OPUS_BANDWIDTH_FULLBAND:
                endband = 21;
                break;
        }
        celt_encoder_ctl(celt_enc, CELT_SET_END_BAND(endband));
        celt_encoder_ctl(celt_enc, CELT_SET_CHANNELS(st->stream_channels));
    }
    celt_encoder_ctl(celt_enc, OPUS_SET_BITRATE(OPUS_BITRATE_MAX));
    if (st->mode != MODE_SILK_ONLY)
    {
        opus_val32 celt_pred=2;
        celt_encoder_ctl(celt_enc, OPUS_SET_VBR(0));
        /* We may still decide to disable prediction later */
        if (st->silk_mode.reducedDependency)
           celt_pred = 0;
        celt_encoder_ctl(celt_enc, CELT_SET_PREDICTION(celt_pred));

        if (st->mode == MODE_HYBRID)
        {
            if( st->use_vbr ) {
                celt_encoder_ctl(celt_enc, OPUS_SET_BITRATE(st->bitrate_bps-st->silk_mode.bitRate));
                celt_encoder_ctl(celt_enc, OPUS_SET_VBR_CONSTRAINT(0));
            }
        } else {
            if (st->use_vbr)
            {
                celt_encoder_ctl(celt_enc, OPUS_SET_VBR(1));
                celt_encoder_ctl(celt_enc, OPUS_SET_VBR_CONSTRAINT(st->vbr_constraint));
                celt_encoder_ctl(celt_enc, OPUS_SET_BITRATE(st->bitrate_bps));
            }
        }
    }

    ALLOC(tmp_prefill, st->channels*st->Fs/400, opus_val16);
    if (st->mode != MODE_SILK_ONLY && st->mode != st->prev_mode && st->prev_mode > 0)
    {
       OPUS_COPY(tmp_prefill, &st->delay_buffer[(st->encoder_buffer-total_buffer-st->Fs/400)*st->channels], st->channels*st->Fs/400);
    }

    if (st->channels*(st->encoder_buffer-(frame_size+total_buffer)) > 0)
    {
       OPUS_MOVE(st->delay_buffer, &st->delay_buffer[st->channels*frame_size], st->channels*(st->encoder_buffer-frame_size-total_buffer));
       OPUS_COPY(&st->delay_buffer[st->channels*(st->encoder_buffer-frame_size-total_buffer)],
             &pcm_buf[0],
             (frame_size+total_buffer)*st->channels);
    } else {
       OPUS_COPY(st->delay_buffer, &pcm_buf[(frame_size+total_buffer-st->encoder_buffer)*st->channels], st->encoder_buffer*st->channels);
    }
    /* gain_fade() and stereo_fade() need to be after the buffer copying
       because we don't want any of this to affect the SILK part */
    if( st->prev_HB_gain < Q15ONE || HB_gain < Q15ONE ) {
       gain_fade(pcm_buf, pcm_buf,
             st->prev_HB_gain, HB_gain, celt_mode->overlap, frame_size, st->channels, celt_mode->window, st->Fs);
    }
    st->prev_HB_gain = HB_gain;
    if (st->mode != MODE_HYBRID || st->stream_channels==1)
    {
       if (equiv_rate > 32000)
          st->silk_mode.stereoWidth_Q14 = 16384;
       else if (equiv_rate < 16000)
          st->silk_mode.stereoWidth_Q14 = 0;
       else
          st->silk_mode.stereoWidth_Q14 = 16384 - 2048*(opus_int32)(32000-equiv_rate)/(equiv_rate-14000);
    }
    if( !st->energy_masking && st->channels == 2 ) {
        /* Apply stereo width reduction (at low bitrates) */
        if( st->hybrid_stereo_width_Q14 < (1 << 14) || st->silk_mode.stereoWidth_Q14 < (1 << 14) ) {
            opus_val16 g1, g2;
            g1 = st->hybrid_stereo_width_Q14;
            g2 = (opus_val16)(st->silk_mode.stereoWidth_Q14);
#ifdef FIXED_POINT
            g1 = g1==16384 ? Q15ONE : SHL16(g1,1);
            g2 = g2==16384 ? Q15ONE : SHL16(g2,1);
#else
            g1 *= (1.f/16384);
            g2 *= (1.f/16384);
#endif
            stereo_fade(pcm_buf, pcm_buf, g1, g2, celt_mode->overlap,
                  frame_size, st->channels, celt_mode->window, st->Fs);
            st->hybrid_stereo_width_Q14 = st->silk_mode.stereoWidth_Q14;
        }
    }

    if ( st->mode != MODE_CELT_ONLY && ec_tell(&enc)+17+20*(st->mode == MODE_HYBRID) <= 8*(max_data_bytes-1))
    {
        /* For SILK mode, the redundancy is inferred from the length */
        if (st->mode == MODE_HYBRID)
           ec_enc_bit_logp(&enc, redundancy, 12);
        if (redundancy)
        {
            int max_redundancy;
            ec_enc_bit_logp(&enc, celt_to_silk, 1);
            if (st->mode == MODE_HYBRID)
            {
               /* Reserve the 8 bits needed for the redundancy length,
                  and at least a few bits for CELT if possible */
               max_redundancy = (max_data_bytes-1)-((ec_tell(&enc)+8+3+7)>>3);
            }
            else
               max_redundancy = (max_data_bytes-1)-((ec_tell(&enc)+7)>>3);
            /* Target the same bit-rate for redundancy as for the rest,
               up to a max of 257 bytes */
            redundancy_bytes = IMIN(max_redundancy, redundancy_bytes);
            redundancy_bytes = IMIN(257, IMAX(2, redundancy_bytes));
            if (st->mode == MODE_HYBRID)
                ec_enc_uint(&enc, redundancy_bytes-2, 256);
        }
    } else {
        redundancy = 0;
    }

    if (!redundancy)
    {
       st->silk_bw_switch = 0;
       redundancy_bytes = 0;
    }
    if (st->mode != MODE_CELT_ONLY)start_band=17;

    if (st->mode == MODE_SILK_ONLY)
    {
        ret = (ec_tell(&enc)+7)>>3;
        ec_enc_done(&enc);
        nb_compr_bytes = ret;
    } else {
       nb_compr_bytes = (max_data_bytes-1)-redundancy_bytes;
       ec_enc_shrink(&enc, nb_compr_bytes);
    }

#ifndef DISABLE_FLOAT_API
    if (redundancy || st->mode != MODE_SILK_ONLY)
       celt_encoder_ctl(celt_enc, CELT_SET_ANALYSIS(&analysis_info));
#endif
    if (st->mode == MODE_HYBRID) {
       SILKInfo info;
       info.signalType = st->silk_mode.signalType;
       info.offset = st->silk_mode.offset;
       celt_encoder_ctl(celt_enc, CELT_SET_SILK_INFO(&info));
    }

    /* 5 ms redundant frame for CELT->SILK */
    if (redundancy && celt_to_silk)
    {
        int err;
        celt_encoder_ctl(celt_enc, CELT_SET_START_BAND(0));
        celt_encoder_ctl(celt_enc, OPUS_SET_VBR(0));
        celt_encoder_ctl(celt_enc, OPUS_SET_BITRATE(OPUS_BITRATE_MAX));
        err = celt_encode_with_ec(celt_enc, pcm_buf, st->Fs/200, data+nb_compr_bytes, redundancy_bytes, NULL);
        if (err < 0)
        {
           RESTORE_STACK;
           return OPUS_INTERNAL_ERROR;
        }
        celt_encoder_ctl(celt_enc, OPUS_GET_FINAL_RANGE(&redundant_rng));
        celt_encoder_ctl(celt_enc, OPUS_RESET_STATE);
    }

    celt_encoder_ctl(celt_enc, CELT_SET_START_BAND(start_band));

    if (st->mode != MODE_SILK_ONLY)
    {
        if (st->mode != st->prev_mode && st->prev_mode > 0)
        {
           unsigned char dummy[2];
           celt_encoder_ctl(celt_enc, OPUS_RESET_STATE);

           /* Prefilling */
           celt_encode_with_ec(celt_enc, tmp_prefill, st->Fs/400, dummy, 2, NULL);
           celt_encoder_ctl(celt_enc, CELT_SET_PREDICTION(0));
        }
        /* If false, we already busted the budget and we'll end up with a "PLC frame" */
        if (ec_tell(&enc) <= 8*nb_compr_bytes)
        {
           /* Set the bitrate again if it was overridden in the redundancy code above*/
           if (redundancy && celt_to_silk && st->mode==MODE_HYBRID && st->use_vbr)
              celt_encoder_ctl(celt_enc, OPUS_SET_BITRATE(st->bitrate_bps-st->silk_mode.bitRate));
           celt_encoder_ctl(celt_enc, OPUS_SET_VBR(st->use_vbr));
           ret = celt_encode_with_ec(celt_enc, pcm_buf, frame_size, NULL, nb_compr_bytes, &enc);
           if (ret < 0)
           {
              RESTORE_STACK;
              return OPUS_INTERNAL_ERROR;
           }
           /* Put CELT->SILK redundancy data in the right place. */
           if (redundancy && celt_to_silk && st->mode==MODE_HYBRID && st->use_vbr)
           {
              OPUS_MOVE(data+ret, data+nb_compr_bytes, redundancy_bytes);
              nb_compr_bytes = nb_compr_bytes+redundancy_bytes;
           }
        }
    }

    /* 5 ms redundant frame for SILK->CELT */
    if (redundancy && !celt_to_silk)
    {
        int err;
        unsigned char dummy[2];
        int N2, N4;
        N2 = st->Fs/200;
        N4 = st->Fs/400;

        celt_encoder_ctl(celt_enc, OPUS_RESET_STATE);
        celt_encoder_ctl(celt_enc, CELT_SET_START_BAND(0));
        celt_encoder_ctl(celt_enc, CELT_SET_PREDICTION(0));
        celt_encoder_ctl(celt_enc, OPUS_SET_VBR(0));
        celt_encoder_ctl(celt_enc, OPUS_SET_BITRATE(OPUS_BITRATE_MAX));

        if (st->mode == MODE_HYBRID)
        {
           /* Shrink packet to what the encoder actually used. */
           nb_compr_bytes = ret;
           ec_enc_shrink(&enc, nb_compr_bytes);
        }
        /* NOTE: We could speed this up slightly (at the expense of code size) by just adding a function that prefills the buffer */
        celt_encode_with_ec(celt_enc, pcm_buf+st->channels*(frame_size-N2-N4), N4, dummy, 2, NULL);

        err = celt_encode_with_ec(celt_enc, pcm_buf+st->channels*(frame_size-N2), N2, data+nb_compr_bytes, redundancy_bytes, NULL);
        if (err < 0)
        {
           RESTORE_STACK;
           return OPUS_INTERNAL_ERROR;
        }
        celt_encoder_ctl(celt_enc, OPUS_GET_FINAL_RANGE(&redundant_rng));
    }



    /* Signalling the mode in the first byte */
    data--;
    data[0] = gen_toc(st->mode, st->Fs/frame_size, curr_bandwidth, st->stream_channels);

    st->rangeFinal = enc.rng ^ redundant_rng;

    if (to_celt)
        st->prev_mode = MODE_CELT_ONLY;
    else
        st->prev_mode = st->mode;
    st->prev_channels = st->stream_channels;
    st->prev_framesize = frame_size;

    st->first = 0;

    /* DTX decision */
#ifndef DISABLE_FLOAT_API
    if (st->use_dtx && (analysis_info.valid || is_silence))
    {
       if (decide_dtx_mode(analysis_info.activity_probability, &st->nb_no_activity_frames,
             st->peak_signal_energy, pcm, frame_size, st->channels, is_silence, st->arch))
       {
          st->rangeFinal = 0;
          data[0] = gen_toc(st->mode, st->Fs/frame_size, curr_bandwidth, st->stream_channels);
          RESTORE_STACK;
          return 1;
       }
    } else {
       st->nb_no_activity_frames = 0;
    }
#endif

    /* In the unlikely case that the SILK encoder busted its target, tell
       the decoder to call the PLC */
    if (ec_tell(&enc) > (max_data_bytes-1)*8)
    {
       if (max_data_bytes < 2)
       {
          RESTORE_STACK;
          return OPUS_BUFFER_TOO_SMALL;
       }
       data[1] = 0;
       ret = 1;
       st->rangeFinal = 0;
    } else if (st->mode==MODE_SILK_ONLY&&!redundancy)
    {
       /*When in LPC only mode it's perfectly
         reasonable to strip off trailing zero bytes as
         the required range decoder behavior is to
         fill these in. This can't be done when the MDCT
         modes are used because the decoder needs to know
         the actual length for allocation purposes.*/
       while(ret>2&&data[ret]==0)ret--;
    }
    /* Count ToC and redundancy */
    ret += 1+redundancy_bytes;
    if (!st->use_vbr)
    {
       if (opus_packet_pad(data, ret, max_data_bytes) != OPUS_OK)
       {
          RESTORE_STACK;
          return OPUS_INTERNAL_ERROR;
       }
       ret = max_data_bytes;
    }
    RESTORE_STACK;
    return ret;
}

#ifdef FIXED_POINT

#ifndef DISABLE_FLOAT_API
opus_int32 opus_encode_float(OpusEncoder *st, const float *pcm, int analysis_frame_size,
      unsigned char *data, opus_int32 max_data_bytes)
{
   int i, ret;
   int frame_size;
   VARDECL(opus_int16, in);
   ALLOC_STACK;

   frame_size = frame_size_select(analysis_frame_size, st->variable_duration, st->Fs);
   if (frame_size <= 0)
   {
      RESTORE_STACK;
      return OPUS_BAD_ARG;
   }
   ALLOC(in, frame_size*st->channels, opus_int16);

   for (i=0;i<frame_size*st->channels;i++)
      in[i] = FLOAT2INT16(pcm[i]);
   ret = opus_encode_native(st, in, frame_size, data, max_data_bytes, 16,
                            pcm, analysis_frame_size, 0, -2, st->channels, downmix_float, 1);
   RESTORE_STACK;
   return ret;
}
#endif

opus_int32 opus_encode(OpusEncoder *st, const opus_int16 *pcm, int analysis_frame_size,
                unsigned char *data, opus_int32 out_data_bytes)
{
   int frame_size;
   frame_size = frame_size_select(analysis_frame_size, st->variable_duration, st->Fs);
   return opus_encode_native(st, pcm, frame_size, data, out_data_bytes, 16,
                             pcm, analysis_frame_size, 0, -2, st->channels, downmix_int, 0);
}

#else
opus_int32 opus_encode(OpusEncoder *st, const opus_int16 *pcm, int analysis_frame_size,
      unsigned char *data, opus_int32 max_data_bytes)
{
   int i, ret;
   int frame_size;
   VARDECL(float, in);
   ALLOC_STACK;

   frame_size = frame_size_select(analysis_frame_size, st->variable_duration, st->Fs);
   if (frame_size <= 0)
   {
      RESTORE_STACK;
      return OPUS_BAD_ARG;
   }
   ALLOC(in, frame_size*st->channels, float);

   for (i=0;i<frame_size*st->channels;i++)
      in[i] = (1.0f/32768)*pcm[i];
   ret = opus_encode_native(st, in, frame_size, data, max_data_bytes, 16,
                            pcm, analysis_frame_size, 0, -2, st->channels, downmix_int, 0);
   RESTORE_STACK;
   return ret;
}
opus_int32 opus_encode_float(OpusEncoder *st, const float *pcm, int analysis_frame_size,
                      unsigned char *data, opus_int32 out_data_bytes)
{
   int frame_size;
   frame_size = frame_size_select(analysis_frame_size, st->variable_duration, st->Fs);
   return opus_encode_native(st, pcm, frame_size, data, out_data_bytes, 24,
                             pcm, analysis_frame_size, 0, -2, st->channels, downmix_float, 1);
}
#endif


int opus_encoder_ctl(OpusEncoder *st, int request, ...)
{
    int ret;
    CELTEncoder *celt_enc;
    va_list ap;

    ret = OPUS_OK;
    va_start(ap, request);

    celt_enc = (CELTEncoder*)((char*)st+st->celt_enc_offset);

    switch (request)
    {
        case OPUS_SET_APPLICATION_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if (   (value != OPUS_APPLICATION_VOIP && value != OPUS_APPLICATION_AUDIO
                 && value != OPUS_APPLICATION_RESTRICTED_LOWDELAY)
               || (!st->first && st->application != value))
            {
               ret = OPUS_BAD_ARG;
               break;
            }
            st->application = value;
#ifndef DISABLE_FLOAT_API
            st->analysis.application = value;
#endif
        }
        break;
        case OPUS_GET_APPLICATION_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->application;
        }
        break;
        case OPUS_SET_BITRATE_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if (value != OPUS_AUTO && value != OPUS_BITRATE_MAX)
            {
                if (value <= 0)
                    goto bad_arg;
                else if (value <= 500)
                    value = 500;
                else if (value > (opus_int32)300000*st->channels)
                    value = (opus_int32)300000*st->channels;
            }
            st->user_bitrate_bps = value;
        }
        break;
        case OPUS_GET_BITRATE_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = user_bitrate_to_bitrate(st, st->prev_framesize, 1276);
        }
        break;
        case OPUS_SET_FORCE_CHANNELS_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if((value<1 || value>st->channels) && value != OPUS_AUTO)
            {
               goto bad_arg;
            }
            st->force_channels = value;
        }
        break;
        case OPUS_GET_FORCE_CHANNELS_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->force_channels;
        }
        break;
        case OPUS_SET_MAX_BANDWIDTH_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if (value < OPUS_BANDWIDTH_NARROWBAND || value > OPUS_BANDWIDTH_FULLBAND)
            {
               goto bad_arg;
            }
            st->max_bandwidth = value;
            if (st->max_bandwidth == OPUS_BANDWIDTH_NARROWBAND) {
                st->silk_mode.maxInternalSampleRate = 8000;
            } else if (st->max_bandwidth == OPUS_BANDWIDTH_MEDIUMBAND) {
                st->silk_mode.maxInternalSampleRate = 12000;
            } else {
                st->silk_mode.maxInternalSampleRate = 16000;
            }
        }
        break;
        case OPUS_GET_MAX_BANDWIDTH_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->max_bandwidth;
        }
        break;
        case OPUS_SET_BANDWIDTH_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if ((value < OPUS_BANDWIDTH_NARROWBAND || value > OPUS_BANDWIDTH_FULLBAND) && value != OPUS_AUTO)
            {
               goto bad_arg;
            }
            st->user_bandwidth = value;
            if (st->user_bandwidth == OPUS_BANDWIDTH_NARROWBAND) {
                st->silk_mode.maxInternalSampleRate = 8000;
            } else if (st->user_bandwidth == OPUS_BANDWIDTH_MEDIUMBAND) {
                st->silk_mode.maxInternalSampleRate = 12000;
            } else {
                st->silk_mode.maxInternalSampleRate = 16000;
            }
        }
        break;
        case OPUS_GET_BANDWIDTH_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->bandwidth;
        }
        break;
        case OPUS_SET_DTX_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if(value<0 || value>1)
            {
               goto bad_arg;
            }
            st->use_dtx = value;
        }
        break;
        case OPUS_GET_DTX_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->use_dtx;
        }
        break;
        case OPUS_SET_COMPLEXITY_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if(value<0 || value>10)
            {
               goto bad_arg;
            }
            st->silk_mode.complexity = value;
            celt_encoder_ctl(celt_enc, OPUS_SET_COMPLEXITY(value));
        }
        break;
        case OPUS_GET_COMPLEXITY_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->silk_mode.complexity;
        }
        break;
        case OPUS_SET_INBAND_FEC_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if(value<0 || value>1)
            {
               goto bad_arg;
            }
            st->silk_mode.useInBandFEC = value;
        }
        break;
        case OPUS_GET_INBAND_FEC_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->silk_mode.useInBandFEC;
        }
        break;
        case OPUS_SET_PACKET_LOSS_PERC_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if (value < 0 || value > 100)
            {
               goto bad_arg;
            }
            st->silk_mode.packetLossPercentage = value;
            celt_encoder_ctl(celt_enc, OPUS_SET_PACKET_LOSS_PERC(value));
        }
        break;
        case OPUS_GET_PACKET_LOSS_PERC_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->silk_mode.packetLossPercentage;
        }
        break;
        case OPUS_SET_VBR_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if(value<0 || value>1)
            {
               goto bad_arg;
            }
            st->use_vbr = value;
            st->silk_mode.useCBR = 1-value;
        }
        break;
        case OPUS_GET_VBR_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->use_vbr;
        }
        break;
        case OPUS_SET_VOICE_RATIO_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if (value<-1 || value>100)
            {
               goto bad_arg;
            }
            st->voice_ratio = value;
        }
        break;
        case OPUS_GET_VOICE_RATIO_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->voice_ratio;
        }
        break;
        case OPUS_SET_VBR_CONSTRAINT_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if(value<0 || value>1)
            {
               goto bad_arg;
            }
            st->vbr_constraint = value;
        }
        break;
        case OPUS_GET_VBR_CONSTRAINT_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->vbr_constraint;
        }
        break;
        case OPUS_SET_SIGNAL_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if(value!=OPUS_AUTO && value!=OPUS_SIGNAL_VOICE && value!=OPUS_SIGNAL_MUSIC)
            {
               goto bad_arg;
            }
            st->signal_type = value;
        }
        break;
        case OPUS_GET_SIGNAL_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->signal_type;
        }
        break;
        case OPUS_GET_LOOKAHEAD_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->Fs/400;
            if (st->application != OPUS_APPLICATION_RESTRICTED_LOWDELAY)
                *value += st->delay_compensation;
        }
        break;
        case OPUS_GET_SAMPLE_RATE_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->Fs;
        }
        break;
        case OPUS_GET_FINAL_RANGE_REQUEST:
        {
            opus_uint32 *value = va_arg(ap, opus_uint32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->rangeFinal;
        }
        break;
        case OPUS_SET_LSB_DEPTH_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if (value<8 || value>24)
            {
               goto bad_arg;
            }
            st->lsb_depth=value;
        }
        break;
        case OPUS_GET_LSB_DEPTH_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->lsb_depth;
        }
        break;
        case OPUS_SET_EXPERT_FRAME_DURATION_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if (value != OPUS_FRAMESIZE_ARG    && value != OPUS_FRAMESIZE_2_5_MS &&
                value != OPUS_FRAMESIZE_5_MS   && value != OPUS_FRAMESIZE_10_MS  &&
                value != OPUS_FRAMESIZE_20_MS  && value != OPUS_FRAMESIZE_40_MS  &&
                value != OPUS_FRAMESIZE_60_MS  && value != OPUS_FRAMESIZE_80_MS  &&
                value != OPUS_FRAMESIZE_100_MS && value != OPUS_FRAMESIZE_120_MS)
            {
               goto bad_arg;
            }
            st->variable_duration = value;
        }
        break;
        case OPUS_GET_EXPERT_FRAME_DURATION_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            *value = st->variable_duration;
        }
        break;
        case OPUS_SET_PREDICTION_DISABLED_REQUEST:
        {
           opus_int32 value = va_arg(ap, opus_int32);
           if (value > 1 || value < 0)
              goto bad_arg;
           st->silk_mode.reducedDependency = value;
        }
        break;
        case OPUS_GET_PREDICTION_DISABLED_REQUEST:
        {
           opus_int32 *value = va_arg(ap, opus_int32*);
           if (!value)
              goto bad_arg;
           *value = st->silk_mode.reducedDependency;
        }
        break;
        case OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if(value<0 || value>1)
            {
               goto bad_arg;
            }
            celt_encoder_ctl(celt_enc, OPUS_SET_PHASE_INVERSION_DISABLED(value));
        }
        break;
        case OPUS_GET_PHASE_INVERSION_DISABLED_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
               goto bad_arg;
            }
            celt_encoder_ctl(celt_enc, OPUS_GET_PHASE_INVERSION_DISABLED(value));
        }
        break;
        case OPUS_RESET_STATE:
        {
           void *silk_enc;
           silk_EncControlStruct dummy;
           char *start;
           silk_enc = (char*)st+st->silk_enc_offset;
#ifndef DISABLE_FLOAT_API
           tonality_analysis_reset(&st->analysis);
#endif

           start = (char*)&st->OPUS_ENCODER_RESET_START;
           OPUS_CLEAR(start, sizeof(OpusEncoder) - (start - (char*)st));

           celt_encoder_ctl(celt_enc, OPUS_RESET_STATE);
           silk_InitEncoder( silk_enc, st->arch, &dummy );
           st->stream_channels = st->channels;
           st->hybrid_stereo_width_Q14 = 1 << 14;
           st->prev_HB_gain = Q15ONE;
           st->first = 1;
           st->mode = MODE_HYBRID;
           st->bandwidth = OPUS_BANDWIDTH_FULLBAND;
           st->variable_HP_smth2_Q15 = silk_LSHIFT( silk_lin2log( VARIABLE_HP_MIN_CUTOFF_HZ ), 8 );
        }
        break;
        case OPUS_SET_FORCE_MODE_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            if ((value < MODE_SILK_ONLY || value > MODE_CELT_ONLY) && value != OPUS_AUTO)
            {
               goto bad_arg;
            }
            st->user_forced_mode = value;
        }
        break;
        case OPUS_SET_LFE_REQUEST:
        {
            opus_int32 value = va_arg(ap, opus_int32);
            st->lfe = value;
            ret = celt_encoder_ctl(celt_enc, OPUS_SET_LFE(value));
        }
        break;
        case OPUS_SET_ENERGY_MASK_REQUEST:
        {
            opus_val16 *value = va_arg(ap, opus_val16*);
            st->energy_masking = value;
            ret = celt_encoder_ctl(celt_enc, OPUS_SET_ENERGY_MASK(value));
        }
        break;
        case OPUS_GET_IN_DTX_REQUEST:
        {
            opus_int32 *value = va_arg(ap, opus_int32*);
            if (!value)
            {
                goto bad_arg;
            }
            if (st->silk_mode.useDTX && (st->prev_mode == MODE_SILK_ONLY || st->prev_mode == MODE_HYBRID)) {
                /* DTX determined by Silk. */
                int n;
                void *silk_enc = (char*)st+st->silk_enc_offset;
                *value = 1;
                for (n=0;n<st->silk_mode.nChannelsInternal;n++) {
                    *value = *value && ((silk_encoder*)silk_enc)->state_Fxx[n].sCmn.noSpeechCounter >= NB_SPEECH_FRAMES_BEFORE_DTX;
                }
            }
#ifndef DISABLE_FLOAT_API
            else if (st->use_dtx) {
                /* DTX determined by Opus. */
                *value = st->nb_no_activity_frames >= NB_SPEECH_FRAMES_BEFORE_DTX;
            }
#endif
            else {
                *value = 0;
            }
        }
        break;

        case CELT_GET_MODE_REQUEST:
        {
           const CELTMode ** value = va_arg(ap, const CELTMode**);
           if (!value)
           {
              goto bad_arg;
           }
           ret = celt_encoder_ctl(celt_enc, CELT_GET_MODE(value));
        }
        break;
        default:
            /* fprintf(stderr, "unknown opus_encoder_ctl() request: %d", request);*/
            ret = OPUS_UNIMPLEMENTED;
            break;
    }
    va_end(ap);
    return ret;
bad_arg:
    va_end(ap);
    return OPUS_BAD_ARG;
}

void opus_encoder_destroy(OpusEncoder *st)
{
    opus_free(st);
}

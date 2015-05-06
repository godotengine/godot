/* Copyright (C) 2003-2008 Jean-Marc Valin

   File: mdf.c
   Echo canceller based on the MDF algorithm (see below)

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

/*
   The echo canceller is based on the MDF algorithm described in:

   J. S. Soo, K. K. Pang Multidelay block frequency adaptive filter, 
   IEEE Trans. Acoust. Speech Signal Process., Vol. ASSP-38, No. 2, 
   February 1990.
   
   We use the Alternatively Updated MDF (AUMDF) variant. Robustness to 
   double-talk is achieved using a variable learning rate as described in:
   
   Valin, J.-M., On Adjusting the Learning Rate in Frequency Domain Echo 
   Cancellation With Double-Talk. IEEE Transactions on Audio,
   Speech and Language Processing, Vol. 15, No. 3, pp. 1030-1034, 2007.
   http://people.xiph.org/~jm/papers/valin_taslp2006.pdf
   
   There is no explicit double-talk detection, but a continuous variation
   in the learning rate based on residual echo, double-talk and background
   noise.
   
   About the fixed-point version:
   All the signals are represented with 16-bit words. The filter weights 
   are represented with 32-bit words, but only the top 16 bits are used
   in most cases. The lower 16 bits are completely unreliable (due to the
   fact that the update is done only on the top bits), but help in the
   adaptation -- probably by removing a "threshold effect" due to
   quantization (rounding going to zero) when the gradient is small.
   
   Another kludge that seems to work good: when performing the weight
   update, we only move half the way toward the "goal" this seems to
   reduce the effect of quantization noise in the update phase. This
   can be seen as applying a gradient descent on a "soft constraint"
   instead of having a hard constraint.
   
*/


#include "config.h"


#include "arch.h"
#include "speex/speex_echo.h"
#include "fftwrap.h"
#include "pseudofloat.h"
#include "math_approx.h"
#include "os_support.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef FIXED_POINT
#define WEIGHT_SHIFT 11
#define NORMALIZE_SCALEDOWN 5
#define NORMALIZE_SCALEUP 3
#else
#define WEIGHT_SHIFT 0
#endif

#ifdef FIXED_POINT
#define WORD2INT(x) ((x) < -32767 ? -32768 : ((x) > 32766 ? 32767 : (x)))  
#else
#define WORD2INT(x) ((x) < -32767.5f ? -32768 : ((x) > 32766.5f ? 32767 : floor(.5+(x))))  
#endif

/* If enabled, the AEC will use a foreground filter and a background filter to be more robust to double-talk
   and difficult signals in general. The cost is an extra FFT and a matrix-vector multiply */
#define TWO_PATH

#ifdef FIXED_POINT
static const spx_float_t MIN_LEAK = {20972, -22};

/* Constants for the two-path filter */
static const spx_float_t VAR1_SMOOTH = {23593, -16};
static const spx_float_t VAR2_SMOOTH = {23675, -15};
static const spx_float_t VAR1_UPDATE = {16384, -15};
static const spx_float_t VAR2_UPDATE = {16384, -16};
static const spx_float_t VAR_BACKTRACK = {16384, -12};
#define TOP16(x) ((x)>>16)

#else

static const spx_float_t MIN_LEAK = .005f;

/* Constants for the two-path filter */
static const spx_float_t VAR1_SMOOTH = .36f;
static const spx_float_t VAR2_SMOOTH = .7225f;
static const spx_float_t VAR1_UPDATE = .5f;
static const spx_float_t VAR2_UPDATE = .25f;
static const spx_float_t VAR_BACKTRACK = 4.f;
#define TOP16(x) (x)
#endif


#define PLAYBACK_DELAY 2

void speex_echo_get_residual(SpeexEchoState *st, spx_word32_t *Yout, int len);


/** Speex echo cancellation state. */
struct SpeexEchoState_ {
   int frame_size;           /**< Number of samples processed each time */
   int window_size;
   int M;
   int cancel_count;
   int adapted;
   int saturated;
   int screwed_up;
   int C;                    /** Number of input channels (microphones) */
   int K;                    /** Number of output channels (loudspeakers) */
   spx_int32_t sampling_rate;
   spx_word16_t spec_average;
   spx_word16_t beta0;
   spx_word16_t beta_max;
   spx_word32_t sum_adapt;
   spx_word16_t leak_estimate;
   
   spx_word16_t *e;      /* scratch */
   spx_word16_t *x;      /* Far-end input buffer (2N) */
   spx_word16_t *X;      /* Far-end buffer (M+1 frames) in frequency domain */
   spx_word16_t *input;  /* scratch */
   spx_word16_t *y;      /* scratch */
   spx_word16_t *last_y;
   spx_word16_t *Y;      /* scratch */
   spx_word16_t *E;
   spx_word32_t *PHI;    /* scratch */
   spx_word32_t *W;      /* (Background) filter weights */
#ifdef TWO_PATH
   spx_word16_t *foreground; /* Foreground filter weights */
   spx_word32_t  Davg1;  /* 1st recursive average of the residual power difference */
   spx_word32_t  Davg2;  /* 2nd recursive average of the residual power difference */
   spx_float_t   Dvar1;  /* Estimated variance of 1st estimator */
   spx_float_t   Dvar2;  /* Estimated variance of 2nd estimator */
#endif
   spx_word32_t *power;  /* Power of the far-end signal */
   spx_float_t  *power_1;/* Inverse power of far-end */
   spx_word16_t *wtmp;   /* scratch */
#ifdef FIXED_POINT
   spx_word16_t *wtmp2;  /* scratch */
#endif
   spx_word32_t *Rf;     /* scratch */
   spx_word32_t *Yf;     /* scratch */
   spx_word32_t *Xf;     /* scratch */
   spx_word32_t *Eh;
   spx_word32_t *Yh;
   spx_float_t   Pey;
   spx_float_t   Pyy;
   spx_word16_t *window;
   spx_word16_t *prop;
   void *fft_table;
   spx_word16_t *memX, *memD, *memE;
   spx_word16_t preemph;
   spx_word16_t notch_radius;
   spx_mem_t *notch_mem;

   /* NOTE: If you only use speex_echo_cancel() and want to save some memory, remove this */
   spx_int16_t *play_buf;
   int play_buf_pos;
   int play_buf_started;
};

static SPEEX_INLINE void filter_dc_notch16(const spx_int16_t *in, spx_word16_t radius, spx_word16_t *out, int len, spx_mem_t *mem, int stride)
{
   int i;
   spx_word16_t den2;
#ifdef FIXED_POINT
   den2 = MULT16_16_Q15(radius,radius) + MULT16_16_Q15(QCONST16(.7,15),MULT16_16_Q15(32767-radius,32767-radius));
#else
   den2 = radius*radius + .7*(1-radius)*(1-radius);
#endif   
   /*printf ("%d %d %d %d %d %d\n", num[0], num[1], num[2], den[0], den[1], den[2]);*/
   for (i=0;i<len;i++)
   {
      spx_word16_t vin = in[i*stride];
      spx_word32_t vout = mem[0] + SHL32(EXTEND32(vin),15);
#ifdef FIXED_POINT
      mem[0] = mem[1] + SHL32(SHL32(-EXTEND32(vin),15) + MULT16_32_Q15(radius,vout),1);
#else
      mem[0] = mem[1] + 2*(-vin + radius*vout);
#endif
      mem[1] = SHL32(EXTEND32(vin),15) - MULT16_32_Q15(den2,vout);
      out[i] = SATURATE32(PSHR32(MULT16_32_Q15(radius,vout),15),32767);
   }
}

/* This inner product is slightly different from the codec version because of fixed-point */
static SPEEX_INLINE spx_word32_t mdf_inner_prod(const spx_word16_t *x, const spx_word16_t *y, int len)
{
   spx_word32_t sum=0;
   len >>= 1;
   while(len--)
   {
      spx_word32_t part=0;
      part = MAC16_16(part,*x++,*y++);
      part = MAC16_16(part,*x++,*y++);
      /* HINT: If you had a 40-bit accumulator, you could shift only at the end */
      sum = ADD32(sum,SHR32(part,6));
   }
   return sum;
}

/** Compute power spectrum of a half-complex (packed) vector */
static SPEEX_INLINE void power_spectrum(const spx_word16_t *X, spx_word32_t *ps, int N)
{
   int i, j;
   ps[0]=MULT16_16(X[0],X[0]);
   for (i=1,j=1;i<N-1;i+=2,j++)
   {
      ps[j] =  MULT16_16(X[i],X[i]) + MULT16_16(X[i+1],X[i+1]);
   }
   ps[j]=MULT16_16(X[i],X[i]);
}

/** Compute power spectrum of a half-complex (packed) vector and accumulate */
static SPEEX_INLINE void power_spectrum_accum(const spx_word16_t *X, spx_word32_t *ps, int N)
{
   int i, j;
   ps[0]+=MULT16_16(X[0],X[0]);
   for (i=1,j=1;i<N-1;i+=2,j++)
   {
      ps[j] +=  MULT16_16(X[i],X[i]) + MULT16_16(X[i+1],X[i+1]);
   }
   ps[j]+=MULT16_16(X[i],X[i]);
}

/** Compute cross-power spectrum of a half-complex (packed) vectors and add to acc */
#ifdef FIXED_POINT
static SPEEX_INLINE void spectral_mul_accum(const spx_word16_t *X, const spx_word32_t *Y, spx_word16_t *acc, int N, int M)
{
   int i,j;
   spx_word32_t tmp1=0,tmp2=0;
   for (j=0;j<M;j++)
   {
      tmp1 = MAC16_16(tmp1, X[j*N],TOP16(Y[j*N]));
   }
   acc[0] = PSHR32(tmp1,WEIGHT_SHIFT);
   for (i=1;i<N-1;i+=2)
   {
      tmp1 = tmp2 = 0;
      for (j=0;j<M;j++)
      {
         tmp1 = SUB32(MAC16_16(tmp1, X[j*N+i],TOP16(Y[j*N+i])), MULT16_16(X[j*N+i+1],TOP16(Y[j*N+i+1])));
         tmp2 = MAC16_16(MAC16_16(tmp2, X[j*N+i+1],TOP16(Y[j*N+i])), X[j*N+i], TOP16(Y[j*N+i+1]));
      }
      acc[i] = PSHR32(tmp1,WEIGHT_SHIFT);
      acc[i+1] = PSHR32(tmp2,WEIGHT_SHIFT);
   }
   tmp1 = tmp2 = 0;
   for (j=0;j<M;j++)
   {
      tmp1 = MAC16_16(tmp1, X[(j+1)*N-1],TOP16(Y[(j+1)*N-1]));
   }
   acc[N-1] = PSHR32(tmp1,WEIGHT_SHIFT);
}
static SPEEX_INLINE void spectral_mul_accum16(const spx_word16_t *X, const spx_word16_t *Y, spx_word16_t *acc, int N, int M)
{
   int i,j;
   spx_word32_t tmp1=0,tmp2=0;
   for (j=0;j<M;j++)
   {
      tmp1 = MAC16_16(tmp1, X[j*N],Y[j*N]);
   }
   acc[0] = PSHR32(tmp1,WEIGHT_SHIFT);
   for (i=1;i<N-1;i+=2)
   {
      tmp1 = tmp2 = 0;
      for (j=0;j<M;j++)
      {
         tmp1 = SUB32(MAC16_16(tmp1, X[j*N+i],Y[j*N+i]), MULT16_16(X[j*N+i+1],Y[j*N+i+1]));
         tmp2 = MAC16_16(MAC16_16(tmp2, X[j*N+i+1],Y[j*N+i]), X[j*N+i], Y[j*N+i+1]);
      }
      acc[i] = PSHR32(tmp1,WEIGHT_SHIFT);
      acc[i+1] = PSHR32(tmp2,WEIGHT_SHIFT);
   }
   tmp1 = tmp2 = 0;
   for (j=0;j<M;j++)
   {
      tmp1 = MAC16_16(tmp1, X[(j+1)*N-1],Y[(j+1)*N-1]);
   }
   acc[N-1] = PSHR32(tmp1,WEIGHT_SHIFT);
}

#else
static SPEEX_INLINE void spectral_mul_accum(const spx_word16_t *X, const spx_word32_t *Y, spx_word16_t *acc, int N, int M)
{
   int i,j;
   for (i=0;i<N;i++)
      acc[i] = 0;
   for (j=0;j<M;j++)
   {
      acc[0] += X[0]*Y[0];
      for (i=1;i<N-1;i+=2)
      {
         acc[i] += (X[i]*Y[i] - X[i+1]*Y[i+1]);
         acc[i+1] += (X[i+1]*Y[i] + X[i]*Y[i+1]);
      }
      acc[i] += X[i]*Y[i];
      X += N;
      Y += N;
   }
}
#define spectral_mul_accum16 spectral_mul_accum
#endif

/** Compute weighted cross-power spectrum of a half-complex (packed) vector with conjugate */
static SPEEX_INLINE void weighted_spectral_mul_conj(const spx_float_t *w, const spx_float_t p, const spx_word16_t *X, const spx_word16_t *Y, spx_word32_t *prod, int N)
{
   int i, j;
   spx_float_t W;
   W = FLOAT_AMULT(p, w[0]);
   prod[0] = FLOAT_MUL32(W,MULT16_16(X[0],Y[0]));
   for (i=1,j=1;i<N-1;i+=2,j++)
   {
      W = FLOAT_AMULT(p, w[j]);
      prod[i] = FLOAT_MUL32(W,MAC16_16(MULT16_16(X[i],Y[i]), X[i+1],Y[i+1]));
      prod[i+1] = FLOAT_MUL32(W,MAC16_16(MULT16_16(-X[i+1],Y[i]), X[i],Y[i+1]));
   }
   W = FLOAT_AMULT(p, w[j]);
   prod[i] = FLOAT_MUL32(W,MULT16_16(X[i],Y[i]));
}

static SPEEX_INLINE void mdf_adjust_prop(const spx_word32_t *W, int N, int M, int P, spx_word16_t *prop)
{
   int i, j, p;
   spx_word16_t max_sum = 1;
   spx_word32_t prop_sum = 1;
   for (i=0;i<M;i++)
   {
      spx_word32_t tmp = 1;
      for (p=0;p<P;p++)
         for (j=0;j<N;j++)
            tmp += MULT16_16(EXTRACT16(SHR32(W[p*N*M + i*N+j],18)), EXTRACT16(SHR32(W[p*N*M + i*N+j],18)));
#ifdef FIXED_POINT
      /* Just a security in case an overflow were to occur */
      tmp = MIN32(ABS32(tmp), 536870912);
#endif
      prop[i] = spx_sqrt(tmp);
      if (prop[i] > max_sum)
         max_sum = prop[i];
   }
   for (i=0;i<M;i++)
   {
      prop[i] += MULT16_16_Q15(QCONST16(.1f,15),max_sum);
      prop_sum += EXTEND32(prop[i]);
   }
   for (i=0;i<M;i++)
   {
      prop[i] = DIV32(MULT16_16(QCONST16(.99f,15), prop[i]),prop_sum);
      /*printf ("%f ", prop[i]);*/
   }
   /*printf ("\n");*/
}

#ifdef DUMP_ECHO_CANCEL_DATA
#include <stdio.h>
static FILE *rFile=NULL, *pFile=NULL, *oFile=NULL;

static void dump_audio(const spx_int16_t *rec, const spx_int16_t *play, const spx_int16_t *out, int len)
{
   if (!(rFile && pFile && oFile))
   {
      speex_fatal("Dump files not open");
   }
   fwrite(rec, sizeof(spx_int16_t), len, rFile);
   fwrite(play, sizeof(spx_int16_t), len, pFile);
   fwrite(out, sizeof(spx_int16_t), len, oFile);
}
#endif

/** Creates a new echo canceller state */
EXPORT SpeexEchoState *speex_echo_state_init(int frame_size, int filter_length)
{
   return speex_echo_state_init_mc(frame_size, filter_length, 1, 1);
}

EXPORT SpeexEchoState *speex_echo_state_init_mc(int frame_size, int filter_length, int nb_mic, int nb_speakers)
{
   int i,N,M, C, K;
   SpeexEchoState *st = (SpeexEchoState *)speex_alloc(sizeof(SpeexEchoState));

   st->K = nb_speakers;
   st->C = nb_mic;
   C=st->C;
   K=st->K;
#ifdef DUMP_ECHO_CANCEL_DATA
   if (rFile || pFile || oFile)
      speex_fatal("Opening dump files twice");
   rFile = fopen("aec_rec.sw", "wb");
   pFile = fopen("aec_play.sw", "wb");
   oFile = fopen("aec_out.sw", "wb");
#endif
   
   st->frame_size = frame_size;
   st->window_size = 2*frame_size;
   N = st->window_size;
   M = st->M = (filter_length+st->frame_size-1)/frame_size;
   st->cancel_count=0;
   st->sum_adapt = 0;
   st->saturated = 0;
   st->screwed_up = 0;
   /* This is the default sampling rate */
   st->sampling_rate = 8000;
   st->spec_average = DIV32_16(SHL32(EXTEND32(st->frame_size), 15), st->sampling_rate);
#ifdef FIXED_POINT
   st->beta0 = DIV32_16(SHL32(EXTEND32(st->frame_size), 16), st->sampling_rate);
   st->beta_max = DIV32_16(SHL32(EXTEND32(st->frame_size), 14), st->sampling_rate);
#else
   st->beta0 = (2.0f*st->frame_size)/st->sampling_rate;
   st->beta_max = (.5f*st->frame_size)/st->sampling_rate;
#endif
   st->leak_estimate = 0;

   st->fft_table = spx_fft_init(N);
   
   st->e = (spx_word16_t*)speex_alloc(C*N*sizeof(spx_word16_t));
   st->x = (spx_word16_t*)speex_alloc(K*N*sizeof(spx_word16_t));
   st->input = (spx_word16_t*)speex_alloc(C*st->frame_size*sizeof(spx_word16_t));
   st->y = (spx_word16_t*)speex_alloc(C*N*sizeof(spx_word16_t));
   st->last_y = (spx_word16_t*)speex_alloc(C*N*sizeof(spx_word16_t));
   st->Yf = (spx_word32_t*)speex_alloc((st->frame_size+1)*sizeof(spx_word32_t));
   st->Rf = (spx_word32_t*)speex_alloc((st->frame_size+1)*sizeof(spx_word32_t));
   st->Xf = (spx_word32_t*)speex_alloc((st->frame_size+1)*sizeof(spx_word32_t));
   st->Yh = (spx_word32_t*)speex_alloc((st->frame_size+1)*sizeof(spx_word32_t));
   st->Eh = (spx_word32_t*)speex_alloc((st->frame_size+1)*sizeof(spx_word32_t));

   st->X = (spx_word16_t*)speex_alloc(K*(M+1)*N*sizeof(spx_word16_t));
   st->Y = (spx_word16_t*)speex_alloc(C*N*sizeof(spx_word16_t));
   st->E = (spx_word16_t*)speex_alloc(C*N*sizeof(spx_word16_t));
   st->W = (spx_word32_t*)speex_alloc(C*K*M*N*sizeof(spx_word32_t));
#ifdef TWO_PATH
   st->foreground = (spx_word16_t*)speex_alloc(M*N*C*K*sizeof(spx_word16_t));
#endif
   st->PHI = (spx_word32_t*)speex_alloc(N*sizeof(spx_word32_t));
   st->power = (spx_word32_t*)speex_alloc((frame_size+1)*sizeof(spx_word32_t));
   st->power_1 = (spx_float_t*)speex_alloc((frame_size+1)*sizeof(spx_float_t));
   st->window = (spx_word16_t*)speex_alloc(N*sizeof(spx_word16_t));
   st->prop = (spx_word16_t*)speex_alloc(M*sizeof(spx_word16_t));
   st->wtmp = (spx_word16_t*)speex_alloc(N*sizeof(spx_word16_t));
#ifdef FIXED_POINT
   st->wtmp2 = (spx_word16_t*)speex_alloc(N*sizeof(spx_word16_t));
   for (i=0;i<N>>1;i++)
   {
      st->window[i] = (16383-SHL16(spx_cos(DIV32_16(MULT16_16(25736,i<<1),N)),1));
      st->window[N-i-1] = st->window[i];
   }
#else
   for (i=0;i<N;i++)
      st->window[i] = .5-.5*cos(2*M_PI*i/N);
#endif
   for (i=0;i<=st->frame_size;i++)
      st->power_1[i] = FLOAT_ONE;
   for (i=0;i<N*M*K*C;i++)
      st->W[i] = 0;
   {
      spx_word32_t sum = 0;
      /* Ratio of ~10 between adaptation rate of first and last block */
      spx_word16_t decay = SHR32(spx_exp(NEG16(DIV32_16(QCONST16(2.4,11),M))),1);
      st->prop[0] = QCONST16(.7, 15);
      sum = EXTEND32(st->prop[0]);
      for (i=1;i<M;i++)
      {
         st->prop[i] = MULT16_16_Q15(st->prop[i-1], decay);
         sum = ADD32(sum, EXTEND32(st->prop[i]));
      }
      for (i=M-1;i>=0;i--)
      {
         st->prop[i] = DIV32(MULT16_16(QCONST16(.8f,15), st->prop[i]),sum);
      }
   }
   
   st->memX = (spx_word16_t*)speex_alloc(K*sizeof(spx_word16_t));
   st->memD = (spx_word16_t*)speex_alloc(C*sizeof(spx_word16_t));
   st->memE = (spx_word16_t*)speex_alloc(C*sizeof(spx_word16_t));
   st->preemph = QCONST16(.9,15);
   if (st->sampling_rate<12000)
      st->notch_radius = QCONST16(.9, 15);
   else if (st->sampling_rate<24000)
      st->notch_radius = QCONST16(.982, 15);
   else
      st->notch_radius = QCONST16(.992, 15);

   st->notch_mem = (spx_mem_t*)speex_alloc(2*C*sizeof(spx_mem_t));
   st->adapted = 0;
   st->Pey = st->Pyy = FLOAT_ONE;
   
#ifdef TWO_PATH
   st->Davg1 = st->Davg2 = 0;
   st->Dvar1 = st->Dvar2 = FLOAT_ZERO;
#endif
   
   st->play_buf = (spx_int16_t*)speex_alloc(K*(PLAYBACK_DELAY+1)*st->frame_size*sizeof(spx_int16_t));
   st->play_buf_pos = PLAYBACK_DELAY*st->frame_size;
   st->play_buf_started = 0;
   
   return st;
}

/** Resets echo canceller state */
EXPORT void speex_echo_state_reset(SpeexEchoState *st)
{
   int i, M, N, C, K;
   st->cancel_count=0;
   st->screwed_up = 0;
   N = st->window_size;
   M = st->M;
   C=st->C;
   K=st->K;
   for (i=0;i<N*M;i++)
      st->W[i] = 0;
#ifdef TWO_PATH
   for (i=0;i<N*M;i++)
      st->foreground[i] = 0;
#endif
   for (i=0;i<N*(M+1);i++)
      st->X[i] = 0;
   for (i=0;i<=st->frame_size;i++)
   {
      st->power[i] = 0;
      st->power_1[i] = FLOAT_ONE;
      st->Eh[i] = 0;
      st->Yh[i] = 0;
   }
   for (i=0;i<st->frame_size;i++)
   {
      st->last_y[i] = 0;
   }
   for (i=0;i<N*C;i++)
   {
      st->E[i] = 0;
   }
   for (i=0;i<N*K;i++)
   {
      st->x[i] = 0;
   }
   for (i=0;i<2*C;i++)
      st->notch_mem[i] = 0;
   for (i=0;i<C;i++)
      st->memD[i]=st->memE[i]=0;
   for (i=0;i<K;i++)
      st->memX[i]=0;

   st->saturated = 0;
   st->adapted = 0;
   st->sum_adapt = 0;
   st->Pey = st->Pyy = FLOAT_ONE;
#ifdef TWO_PATH
   st->Davg1 = st->Davg2 = 0;
   st->Dvar1 = st->Dvar2 = FLOAT_ZERO;
#endif
   for (i=0;i<3*st->frame_size;i++)
      st->play_buf[i] = 0;
   st->play_buf_pos = PLAYBACK_DELAY*st->frame_size;
   st->play_buf_started = 0;

}

/** Destroys an echo canceller state */
EXPORT void speex_echo_state_destroy(SpeexEchoState *st)
{
   spx_fft_destroy(st->fft_table);

   speex_free(st->e);
   speex_free(st->x);
   speex_free(st->input);
   speex_free(st->y);
   speex_free(st->last_y);
   speex_free(st->Yf);
   speex_free(st->Rf);
   speex_free(st->Xf);
   speex_free(st->Yh);
   speex_free(st->Eh);

   speex_free(st->X);
   speex_free(st->Y);
   speex_free(st->E);
   speex_free(st->W);
#ifdef TWO_PATH
   speex_free(st->foreground);
#endif
   speex_free(st->PHI);
   speex_free(st->power);
   speex_free(st->power_1);
   speex_free(st->window);
   speex_free(st->prop);
   speex_free(st->wtmp);
#ifdef FIXED_POINT
   speex_free(st->wtmp2);
#endif
   speex_free(st->memX);
   speex_free(st->memD);
   speex_free(st->memE);
   speex_free(st->notch_mem);

   speex_free(st->play_buf);
   speex_free(st);
   
#ifdef DUMP_ECHO_CANCEL_DATA
   fclose(rFile);
   fclose(pFile);
   fclose(oFile);
   rFile = pFile = oFile = NULL;
#endif
}

EXPORT void speex_echo_capture(SpeexEchoState *st, const spx_int16_t *rec, spx_int16_t *out)
{
   int i;
   /*speex_warning_int("capture with fill level ", st->play_buf_pos/st->frame_size);*/
   st->play_buf_started = 1;
   if (st->play_buf_pos>=st->frame_size)
   {
      speex_echo_cancellation(st, rec, st->play_buf, out);
      st->play_buf_pos -= st->frame_size;
      for (i=0;i<st->play_buf_pos;i++)
         st->play_buf[i] = st->play_buf[i+st->frame_size];
   } else {
      speex_warning("No playback frame available (your application is buggy and/or got xruns)");
      if (st->play_buf_pos!=0)
      {
         speex_warning("internal playback buffer corruption?");
         st->play_buf_pos = 0;
      }
      for (i=0;i<st->frame_size;i++)
         out[i] = rec[i];
   }
}

EXPORT void speex_echo_playback(SpeexEchoState *st, const spx_int16_t *play)
{
   /*speex_warning_int("playback with fill level ", st->play_buf_pos/st->frame_size);*/
   if (!st->play_buf_started)
   {
      speex_warning("discarded first playback frame");
      return;
   }
   if (st->play_buf_pos<=PLAYBACK_DELAY*st->frame_size)
   {
      int i;
      for (i=0;i<st->frame_size;i++)
         st->play_buf[st->play_buf_pos+i] = play[i];
      st->play_buf_pos += st->frame_size;
      if (st->play_buf_pos <= (PLAYBACK_DELAY-1)*st->frame_size)
      {
         speex_warning("Auto-filling the buffer (your application is buggy and/or got xruns)");
         for (i=0;i<st->frame_size;i++)
            st->play_buf[st->play_buf_pos+i] = play[i];
         st->play_buf_pos += st->frame_size;
      }
   } else {
      speex_warning("Had to discard a playback frame (your application is buggy and/or got xruns)");
   }
}

/** Performs echo cancellation on a frame (deprecated, last arg now ignored) */
EXPORT void speex_echo_cancel(SpeexEchoState *st, const spx_int16_t *in, const spx_int16_t *far_end, spx_int16_t *out, spx_int32_t *Yout)
{
   speex_echo_cancellation(st, in, far_end, out);
}

/** Performs echo cancellation on a frame */
EXPORT void speex_echo_cancellation(SpeexEchoState *st, const spx_int16_t *in, const spx_int16_t *far_end, spx_int16_t *out)
{
   int i,j, chan, speak;
   int N,M, C, K;
   spx_word32_t Syy,See,Sxx,Sdd, Sff;
#ifdef TWO_PATH
   spx_word32_t Dbf;
   int update_foreground;
#endif
   spx_word32_t Sey;
   spx_word16_t ss, ss_1;
   spx_float_t Pey = FLOAT_ONE, Pyy=FLOAT_ONE;
   spx_float_t alpha, alpha_1;
   spx_word16_t RER;
   spx_word32_t tmp32;
   
   N = st->window_size;
   M = st->M;
   C = st->C;
   K = st->K;

   st->cancel_count++;
#ifdef FIXED_POINT
   ss=DIV32_16(11469,M);
   ss_1 = SUB16(32767,ss);
#else
   ss=.35/M;
   ss_1 = 1-ss;
#endif

   for (chan = 0; chan < C; chan++)
   {
      /* Apply a notch filter to make sure DC doesn't end up causing problems */
      filter_dc_notch16(in+chan, st->notch_radius, st->input+chan*st->frame_size, st->frame_size, st->notch_mem+2*chan, C);
      /* Copy input data to buffer and apply pre-emphasis */
      /* Copy input data to buffer */
      for (i=0;i<st->frame_size;i++)
      {
         spx_word32_t tmp32;
         /* FIXME: This core has changed a bit, need to merge properly */
         tmp32 = SUB32(EXTEND32(st->input[chan*st->frame_size+i]), EXTEND32(MULT16_16_P15(st->preemph, st->memD[chan])));
#ifdef FIXED_POINT
         if (tmp32 > 32767)
         {
            tmp32 = 32767;
            if (st->saturated == 0)
               st->saturated = 1;
         }      
         if (tmp32 < -32767)
         {
            tmp32 = -32767;
            if (st->saturated == 0)
               st->saturated = 1;
         }
#endif
         st->memD[chan] = st->input[chan*st->frame_size+i];
         st->input[chan*st->frame_size+i] = EXTRACT16(tmp32);
      }
   }

   for (speak = 0; speak < K; speak++)
   {
      for (i=0;i<st->frame_size;i++)
      {
         spx_word32_t tmp32;
         st->x[speak*N+i] = st->x[speak*N+i+st->frame_size];
         tmp32 = SUB32(EXTEND32(far_end[i*K+speak]), EXTEND32(MULT16_16_P15(st->preemph, st->memX[speak])));
#ifdef FIXED_POINT
         /*FIXME: If saturation occurs here, we need to freeze adaptation for M frames (not just one) */
         if (tmp32 > 32767)
         {
            tmp32 = 32767;
            st->saturated = M+1;
         }      
         if (tmp32 < -32767)
         {
            tmp32 = -32767;
            st->saturated = M+1;
         }      
#endif
         st->x[speak*N+i+st->frame_size] = EXTRACT16(tmp32);
         st->memX[speak] = far_end[i*K+speak];
      }
   }   
   
   for (speak = 0; speak < K; speak++)
   {
      /* Shift memory: this could be optimized eventually*/
      for (j=M-1;j>=0;j--)
      {
         for (i=0;i<N;i++)
            st->X[(j+1)*N*K+speak*N+i] = st->X[j*N*K+speak*N+i];
      }
      /* Convert x (echo input) to frequency domain */
      spx_fft(st->fft_table, st->x+speak*N, &st->X[speak*N]);
   }
   
   Sxx = 0;
   for (speak = 0; speak < K; speak++)
   {
      Sxx += mdf_inner_prod(st->x+speak*N+st->frame_size, st->x+speak*N+st->frame_size, st->frame_size);
      power_spectrum_accum(st->X+speak*N, st->Xf, N);
   }
   
   Sff = 0;  
   for (chan = 0; chan < C; chan++)
   {
#ifdef TWO_PATH
      /* Compute foreground filter */
      spectral_mul_accum16(st->X, st->foreground+chan*N*K*M, st->Y+chan*N, N, M*K);
      spx_ifft(st->fft_table, st->Y+chan*N, st->e+chan*N);
      for (i=0;i<st->frame_size;i++)
         st->e[chan*N+i] = SUB16(st->input[chan*st->frame_size+i], st->e[chan*N+i+st->frame_size]);
      Sff += mdf_inner_prod(st->e+chan*N, st->e+chan*N, st->frame_size);
#endif
   }
   
   /* Adjust proportional adaption rate */
   /* FIXME: Adjust that for C, K*/
   if (st->adapted)
      mdf_adjust_prop (st->W, N, M, C*K, st->prop);
   /* Compute weight gradient */
   if (st->saturated == 0)
   {
      for (chan = 0; chan < C; chan++)
      {
         for (speak = 0; speak < K; speak++)
         {
            for (j=M-1;j>=0;j--)
            {
               weighted_spectral_mul_conj(st->power_1, FLOAT_SHL(PSEUDOFLOAT(st->prop[j]),-15), &st->X[(j+1)*N*K+speak*N], st->E+chan*N, st->PHI, N);
               for (i=0;i<N;i++)
                  st->W[chan*N*K*M + j*N*K + speak*N + i] += st->PHI[i];
            }
         }
      }
   } else {
      st->saturated--;
   }
   
   /* FIXME: MC conversion required */ 
   /* Update weight to prevent circular convolution (MDF / AUMDF) */
   for (chan = 0; chan < C; chan++)
   {
      for (speak = 0; speak < K; speak++)
      {
         for (j=0;j<M;j++)
         {
            /* This is a variant of the Alternatively Updated MDF (AUMDF) */
            /* Remove the "if" to make this an MDF filter */
            if (j==0 || st->cancel_count%(M-1) == j-1)
            {
#ifdef FIXED_POINT
               for (i=0;i<N;i++)
                  st->wtmp2[i] = EXTRACT16(PSHR32(st->W[chan*N*K*M + j*N*K + speak*N + i],NORMALIZE_SCALEDOWN+16));
               spx_ifft(st->fft_table, st->wtmp2, st->wtmp);
               for (i=0;i<st->frame_size;i++)
               {
                  st->wtmp[i]=0;
               }
               for (i=st->frame_size;i<N;i++)
               {
                  st->wtmp[i]=SHL16(st->wtmp[i],NORMALIZE_SCALEUP);
               }
               spx_fft(st->fft_table, st->wtmp, st->wtmp2);
               /* The "-1" in the shift is a sort of kludge that trades less efficient update speed for decrease noise */
               for (i=0;i<N;i++)
                  st->W[chan*N*K*M + j*N*K + speak*N + i] -= SHL32(EXTEND32(st->wtmp2[i]),16+NORMALIZE_SCALEDOWN-NORMALIZE_SCALEUP-1);
#else
               spx_ifft(st->fft_table, &st->W[chan*N*K*M + j*N*K + speak*N], st->wtmp);
               for (i=st->frame_size;i<N;i++)
               {
                  st->wtmp[i]=0;
               }
               spx_fft(st->fft_table, st->wtmp, &st->W[chan*N*K*M + j*N*K + speak*N]);
#endif
            }
         }
      }
   }
   
   /* So we can use power_spectrum_accum */ 
   for (i=0;i<=st->frame_size;i++)
      st->Rf[i] = st->Yf[i] = st->Xf[i] = 0;
      
   Dbf = 0;
   See = 0;    
#ifdef TWO_PATH
   /* Difference in response, this is used to estimate the variance of our residual power estimate */
   for (chan = 0; chan < C; chan++)
   {
      spectral_mul_accum(st->X, st->W+chan*N*K*M, st->Y+chan*N, N, M*K);
      spx_ifft(st->fft_table, st->Y+chan*N, st->y+chan*N);
      for (i=0;i<st->frame_size;i++)
         st->e[chan*N+i] = SUB16(st->e[chan*N+i+st->frame_size], st->y[chan*N+i+st->frame_size]);
      Dbf += 10+mdf_inner_prod(st->e+chan*N, st->e+chan*N, st->frame_size);
      for (i=0;i<st->frame_size;i++)
         st->e[chan*N+i] = SUB16(st->input[chan*st->frame_size+i], st->y[chan*N+i+st->frame_size]);
      See += mdf_inner_prod(st->e+chan*N, st->e+chan*N, st->frame_size);
   }
#endif

#ifndef TWO_PATH
   Sff = See;
#endif

#ifdef TWO_PATH
   /* Logic for updating the foreground filter */
   
   /* For two time windows, compute the mean of the energy difference, as well as the variance */
   st->Davg1 = ADD32(MULT16_32_Q15(QCONST16(.6f,15),st->Davg1), MULT16_32_Q15(QCONST16(.4f,15),SUB32(Sff,See)));
   st->Davg2 = ADD32(MULT16_32_Q15(QCONST16(.85f,15),st->Davg2), MULT16_32_Q15(QCONST16(.15f,15),SUB32(Sff,See)));
   st->Dvar1 = FLOAT_ADD(FLOAT_MULT(VAR1_SMOOTH, st->Dvar1), FLOAT_MUL32U(MULT16_32_Q15(QCONST16(.4f,15),Sff), MULT16_32_Q15(QCONST16(.4f,15),Dbf)));
   st->Dvar2 = FLOAT_ADD(FLOAT_MULT(VAR2_SMOOTH, st->Dvar2), FLOAT_MUL32U(MULT16_32_Q15(QCONST16(.15f,15),Sff), MULT16_32_Q15(QCONST16(.15f,15),Dbf)));
   
   /* Equivalent float code:
   st->Davg1 = .6*st->Davg1 + .4*(Sff-See);
   st->Davg2 = .85*st->Davg2 + .15*(Sff-See);
   st->Dvar1 = .36*st->Dvar1 + .16*Sff*Dbf;
   st->Dvar2 = .7225*st->Dvar2 + .0225*Sff*Dbf;
   */
   
   update_foreground = 0;
   /* Check if we have a statistically significant reduction in the residual echo */
   /* Note that this is *not* Gaussian, so we need to be careful about the longer tail */
   if (FLOAT_GT(FLOAT_MUL32U(SUB32(Sff,See),ABS32(SUB32(Sff,See))), FLOAT_MUL32U(Sff,Dbf)))
      update_foreground = 1;
   else if (FLOAT_GT(FLOAT_MUL32U(st->Davg1, ABS32(st->Davg1)), FLOAT_MULT(VAR1_UPDATE,(st->Dvar1))))
      update_foreground = 1;
   else if (FLOAT_GT(FLOAT_MUL32U(st->Davg2, ABS32(st->Davg2)), FLOAT_MULT(VAR2_UPDATE,(st->Dvar2))))
      update_foreground = 1;
   
   /* Do we update? */
   if (update_foreground)
   {
      st->Davg1 = st->Davg2 = 0;
      st->Dvar1 = st->Dvar2 = FLOAT_ZERO;
      /* Copy background filter to foreground filter */
      for (i=0;i<N*M*C*K;i++)
         st->foreground[i] = EXTRACT16(PSHR32(st->W[i],16));
      /* Apply a smooth transition so as to not introduce blocking artifacts */
      for (chan = 0; chan < C; chan++)
         for (i=0;i<st->frame_size;i++)
            st->e[chan*N+i+st->frame_size] = MULT16_16_Q15(st->window[i+st->frame_size],st->e[chan*N+i+st->frame_size]) + MULT16_16_Q15(st->window[i],st->y[chan*N+i+st->frame_size]);
   } else {
      int reset_background=0;
      /* Otherwise, check if the background filter is significantly worse */
      if (FLOAT_GT(FLOAT_MUL32U(NEG32(SUB32(Sff,See)),ABS32(SUB32(Sff,See))), FLOAT_MULT(VAR_BACKTRACK,FLOAT_MUL32U(Sff,Dbf))))
         reset_background = 1;
      if (FLOAT_GT(FLOAT_MUL32U(NEG32(st->Davg1), ABS32(st->Davg1)), FLOAT_MULT(VAR_BACKTRACK,st->Dvar1)))
         reset_background = 1;
      if (FLOAT_GT(FLOAT_MUL32U(NEG32(st->Davg2), ABS32(st->Davg2)), FLOAT_MULT(VAR_BACKTRACK,st->Dvar2)))
         reset_background = 1;
      if (reset_background)
      {
         /* Copy foreground filter to background filter */
         for (i=0;i<N*M*C*K;i++)
            st->W[i] = SHL32(EXTEND32(st->foreground[i]),16);
         /* We also need to copy the output so as to get correct adaptation */
         for (chan = 0; chan < C; chan++)
         {        
            for (i=0;i<st->frame_size;i++)
               st->y[chan*N+i+st->frame_size] = st->e[chan*N+i+st->frame_size];
            for (i=0;i<st->frame_size;i++)
               st->e[chan*N+i] = SUB16(st->input[chan*st->frame_size+i], st->y[chan*N+i+st->frame_size]);
         }        
         See = Sff;
         st->Davg1 = st->Davg2 = 0;
         st->Dvar1 = st->Dvar2 = FLOAT_ZERO;
      }
   }
#endif

   Sey = Syy = Sdd = 0;  
   for (chan = 0; chan < C; chan++)
   {    
      /* Compute error signal (for the output with de-emphasis) */ 
      for (i=0;i<st->frame_size;i++)
      {
         spx_word32_t tmp_out;
#ifdef TWO_PATH
         tmp_out = SUB32(EXTEND32(st->input[chan*st->frame_size+i]), EXTEND32(st->e[chan*N+i+st->frame_size]));
#else
         tmp_out = SUB32(EXTEND32(st->input[chan*st->frame_size+i]), EXTEND32(st->y[chan*N+i+st->frame_size]));
#endif
         tmp_out = ADD32(tmp_out, EXTEND32(MULT16_16_P15(st->preemph, st->memE[chan])));
      /* This is an arbitrary test for saturation in the microphone signal */
         if (in[i*C+chan] <= -32000 || in[i*C+chan] >= 32000)
         {
         if (st->saturated == 0)
            st->saturated = 1;
         }
         out[i*C+chan] = WORD2INT(tmp_out);
         st->memE[chan] = tmp_out;
      }

#ifdef DUMP_ECHO_CANCEL_DATA
      dump_audio(in, far_end, out, st->frame_size);
#endif
   
      /* Compute error signal (filter update version) */ 
      for (i=0;i<st->frame_size;i++)
      {
         st->e[chan*N+i+st->frame_size] = st->e[chan*N+i];
         st->e[chan*N+i] = 0;
      }
      
      /* Compute a bunch of correlations */
      /* FIXME: bad merge */
      Sey += mdf_inner_prod(st->e+chan*N+st->frame_size, st->y+chan*N+st->frame_size, st->frame_size);
      Syy += mdf_inner_prod(st->y+chan*N+st->frame_size, st->y+chan*N+st->frame_size, st->frame_size);
      Sdd += mdf_inner_prod(st->input+chan*st->frame_size, st->input+chan*st->frame_size, st->frame_size);
      
      /* Convert error to frequency domain */
      spx_fft(st->fft_table, st->e+chan*N, st->E+chan*N);
      for (i=0;i<st->frame_size;i++)
         st->y[i+chan*N] = 0;
      spx_fft(st->fft_table, st->y+chan*N, st->Y+chan*N);
   
      /* Compute power spectrum of echo (X), error (E) and filter response (Y) */
      power_spectrum_accum(st->E+chan*N, st->Rf, N);
      power_spectrum_accum(st->Y+chan*N, st->Yf, N);
    
   }
   
   /*printf ("%f %f %f %f\n", Sff, See, Syy, Sdd, st->update_cond);*/
   
   /* Do some sanity check */
   if (!(Syy>=0 && Sxx>=0 && See >= 0)
#ifndef FIXED_POINT
       || !(Sff < N*1e9 && Syy < N*1e9 && Sxx < N*1e9)
#endif
      )
   {
      /* Things have gone really bad */
      st->screwed_up += 50;
      for (i=0;i<st->frame_size*C;i++)
         out[i] = 0;
   } else if (SHR32(Sff, 2) > ADD32(Sdd, SHR32(MULT16_16(N, 10000),6)))
   {
      /* AEC seems to add lots of echo instead of removing it, let's see if it will improve */
      st->screwed_up++;
   } else {
      /* Everything's fine */
      st->screwed_up=0;
   }
   if (st->screwed_up>=50)
   {
      speex_warning("The echo canceller started acting funny and got slapped (reset). It swears it will behave now.");
      speex_echo_state_reset(st);
      return;
   }

   /* Add a small noise floor to make sure not to have problems when dividing */
   See = MAX32(See, SHR32(MULT16_16(N, 100),6));
     
   for (speak = 0; speak < K; speak++)
   {
      Sxx += mdf_inner_prod(st->x+speak*N+st->frame_size, st->x+speak*N+st->frame_size, st->frame_size);
      power_spectrum_accum(st->X+speak*N, st->Xf, N);
   }

   
   /* Smooth far end energy estimate over time */
   for (j=0;j<=st->frame_size;j++)
      st->power[j] = MULT16_32_Q15(ss_1,st->power[j]) + 1 + MULT16_32_Q15(ss,st->Xf[j]);

   /* Compute filtered spectra and (cross-)correlations */
   for (j=st->frame_size;j>=0;j--)
   {
      spx_float_t Eh, Yh;
      Eh = PSEUDOFLOAT(st->Rf[j] - st->Eh[j]);
      Yh = PSEUDOFLOAT(st->Yf[j] - st->Yh[j]);
      Pey = FLOAT_ADD(Pey,FLOAT_MULT(Eh,Yh));
      Pyy = FLOAT_ADD(Pyy,FLOAT_MULT(Yh,Yh));
#ifdef FIXED_POINT
      st->Eh[j] = MAC16_32_Q15(MULT16_32_Q15(SUB16(32767,st->spec_average),st->Eh[j]), st->spec_average, st->Rf[j]);
      st->Yh[j] = MAC16_32_Q15(MULT16_32_Q15(SUB16(32767,st->spec_average),st->Yh[j]), st->spec_average, st->Yf[j]);
#else
      st->Eh[j] = (1-st->spec_average)*st->Eh[j] + st->spec_average*st->Rf[j];
      st->Yh[j] = (1-st->spec_average)*st->Yh[j] + st->spec_average*st->Yf[j];
#endif
   }
   
   Pyy = FLOAT_SQRT(Pyy);
   Pey = FLOAT_DIVU(Pey,Pyy);

   /* Compute correlation updatete rate */
   tmp32 = MULT16_32_Q15(st->beta0,Syy);
   if (tmp32 > MULT16_32_Q15(st->beta_max,See))
      tmp32 = MULT16_32_Q15(st->beta_max,See);
   alpha = FLOAT_DIV32(tmp32, See);
   alpha_1 = FLOAT_SUB(FLOAT_ONE, alpha);
   /* Update correlations (recursive average) */
   st->Pey = FLOAT_ADD(FLOAT_MULT(alpha_1,st->Pey) , FLOAT_MULT(alpha,Pey));
   st->Pyy = FLOAT_ADD(FLOAT_MULT(alpha_1,st->Pyy) , FLOAT_MULT(alpha,Pyy));
   if (FLOAT_LT(st->Pyy, FLOAT_ONE))
      st->Pyy = FLOAT_ONE;
   /* We don't really hope to get better than 33 dB (MIN_LEAK-3dB) attenuation anyway */
   if (FLOAT_LT(st->Pey, FLOAT_MULT(MIN_LEAK,st->Pyy)))
      st->Pey = FLOAT_MULT(MIN_LEAK,st->Pyy);
   if (FLOAT_GT(st->Pey, st->Pyy))
      st->Pey = st->Pyy;
   /* leak_estimate is the linear regression result */
   st->leak_estimate = FLOAT_EXTRACT16(FLOAT_SHL(FLOAT_DIVU(st->Pey, st->Pyy),14));
   /* This looks like a stupid bug, but it's right (because we convert from Q14 to Q15) */
   if (st->leak_estimate > 16383)
      st->leak_estimate = 32767;
   else
      st->leak_estimate = SHL16(st->leak_estimate,1);
   /*printf ("%f\n", st->leak_estimate);*/
   
   /* Compute Residual to Error Ratio */
#ifdef FIXED_POINT
   tmp32 = MULT16_32_Q15(st->leak_estimate,Syy);
   tmp32 = ADD32(SHR32(Sxx,13), ADD32(tmp32, SHL32(tmp32,1)));
   /* Check for y in e (lower bound on RER) */
   {
      spx_float_t bound = PSEUDOFLOAT(Sey);
      bound = FLOAT_DIVU(FLOAT_MULT(bound, bound), PSEUDOFLOAT(ADD32(1,Syy)));
      if (FLOAT_GT(bound, PSEUDOFLOAT(See)))
         tmp32 = See;
      else if (tmp32 < FLOAT_EXTRACT32(bound))
         tmp32 = FLOAT_EXTRACT32(bound);
   }
   if (tmp32 > SHR32(See,1))
      tmp32 = SHR32(See,1);
   RER = FLOAT_EXTRACT16(FLOAT_SHL(FLOAT_DIV32(tmp32,See),15));
#else
   RER = (.0001*Sxx + 3.*MULT16_32_Q15(st->leak_estimate,Syy)) / See;
   /* Check for y in e (lower bound on RER) */
   if (RER < Sey*Sey/(1+See*Syy))
      RER = Sey*Sey/(1+See*Syy);
   if (RER > .5)
      RER = .5;
#endif

   /* We consider that the filter has had minimal adaptation if the following is true*/
   if (!st->adapted && st->sum_adapt > SHL32(EXTEND32(M),15) && MULT16_32_Q15(st->leak_estimate,Syy) > MULT16_32_Q15(QCONST16(.03f,15),Syy))
   {
      st->adapted = 1;
   }

   if (st->adapted)
   {
      /* Normal learning rate calculation once we're past the minimal adaptation phase */
      for (i=0;i<=st->frame_size;i++)
      {
         spx_word32_t r, e;
         /* Compute frequency-domain adaptation mask */
         r = MULT16_32_Q15(st->leak_estimate,SHL32(st->Yf[i],3));
         e = SHL32(st->Rf[i],3)+1;
#ifdef FIXED_POINT
         if (r>SHR32(e,1))
            r = SHR32(e,1);
#else
         if (r>.5*e)
            r = .5*e;
#endif
         r = MULT16_32_Q15(QCONST16(.7,15),r) + MULT16_32_Q15(QCONST16(.3,15),(spx_word32_t)(MULT16_32_Q15(RER,e)));
         /*st->power_1[i] = adapt_rate*r/(e*(1+st->power[i]));*/
         st->power_1[i] = FLOAT_SHL(FLOAT_DIV32_FLOAT(r,FLOAT_MUL32U(e,st->power[i]+10)),WEIGHT_SHIFT+16);
      }
   } else {
      /* Temporary adaption rate if filter is not yet adapted enough */
      spx_word16_t adapt_rate=0;

      if (Sxx > SHR32(MULT16_16(N, 1000),6)) 
      {
         tmp32 = MULT16_32_Q15(QCONST16(.25f, 15), Sxx);
#ifdef FIXED_POINT
         if (tmp32 > SHR32(See,2))
            tmp32 = SHR32(See,2);
#else
         if (tmp32 > .25*See)
            tmp32 = .25*See;
#endif
         adapt_rate = FLOAT_EXTRACT16(FLOAT_SHL(FLOAT_DIV32(tmp32, See),15));
      }
      for (i=0;i<=st->frame_size;i++)
         st->power_1[i] = FLOAT_SHL(FLOAT_DIV32(EXTEND32(adapt_rate),ADD32(st->power[i],10)),WEIGHT_SHIFT+1);


      /* How much have we adapted so far? */
      st->sum_adapt = ADD32(st->sum_adapt,adapt_rate);
   }

   /* FIXME: MC conversion required */ 
      for (i=0;i<st->frame_size;i++)
         st->last_y[i] = st->last_y[st->frame_size+i];
   if (st->adapted)
   {
      /* If the filter is adapted, take the filtered echo */
      for (i=0;i<st->frame_size;i++)
         st->last_y[st->frame_size+i] = in[i]-out[i];
   } else {
      /* If filter isn't adapted yet, all we can do is take the far end signal directly */
      /* moved earlier: for (i=0;i<N;i++)
      st->last_y[i] = st->x[i];*/
   }

}

/* Compute spectrum of estimated echo for use in an echo post-filter */
void speex_echo_get_residual(SpeexEchoState *st, spx_word32_t *residual_echo, int len)
{
   int i;
   spx_word16_t leak2;
   int N;
   
   N = st->window_size;

   /* Apply hanning window (should pre-compute it)*/
   for (i=0;i<N;i++)
      st->y[i] = MULT16_16_Q15(st->window[i],st->last_y[i]);
      
   /* Compute power spectrum of the echo */
   spx_fft(st->fft_table, st->y, st->Y);
   power_spectrum(st->Y, residual_echo, N);
      
#ifdef FIXED_POINT
   if (st->leak_estimate > 16383)
      leak2 = 32767;
   else
      leak2 = SHL16(st->leak_estimate, 1);
#else
   if (st->leak_estimate>.5)
      leak2 = 1;
   else
      leak2 = 2*st->leak_estimate;
#endif
   /* Estimate residual echo */
   for (i=0;i<=st->frame_size;i++)
      residual_echo[i] = (spx_int32_t)MULT16_32_Q15(leak2,residual_echo[i]);
   
}

EXPORT int speex_echo_ctl(SpeexEchoState *st, int request, void *ptr)
{
   switch(request)
   {
      
      case SPEEX_ECHO_GET_FRAME_SIZE:
         (*(int*)ptr) = st->frame_size;
         break;
      case SPEEX_ECHO_SET_SAMPLING_RATE:
         st->sampling_rate = (*(int*)ptr);
         st->spec_average = DIV32_16(SHL32(EXTEND32(st->frame_size), 15), st->sampling_rate);
#ifdef FIXED_POINT
         st->beta0 = DIV32_16(SHL32(EXTEND32(st->frame_size), 16), st->sampling_rate);
         st->beta_max = DIV32_16(SHL32(EXTEND32(st->frame_size), 14), st->sampling_rate);
#else
         st->beta0 = (2.0f*st->frame_size)/st->sampling_rate;
         st->beta_max = (.5f*st->frame_size)/st->sampling_rate;
#endif
         if (st->sampling_rate<12000)
            st->notch_radius = QCONST16(.9, 15);
         else if (st->sampling_rate<24000)
            st->notch_radius = QCONST16(.982, 15);
         else
            st->notch_radius = QCONST16(.992, 15);
         break;
      case SPEEX_ECHO_GET_SAMPLING_RATE:
         (*(int*)ptr) = st->sampling_rate;
         break;
      case SPEEX_ECHO_GET_IMPULSE_RESPONSE_SIZE:
         /*FIXME: Implement this for multiple channels */
         *((spx_int32_t *)ptr) = st->M * st->frame_size;
         break;
      case SPEEX_ECHO_GET_IMPULSE_RESPONSE:
      {
         int M = st->M, N = st->window_size, n = st->frame_size, i, j;
         spx_int32_t *filt = (spx_int32_t *) ptr;
         for(j=0;j<M;j++)
         {
            /*FIXME: Implement this for multiple channels */
#ifdef FIXED_POINT
            for (i=0;i<N;i++)
               st->wtmp2[i] = EXTRACT16(PSHR32(st->W[j*N+i],16+NORMALIZE_SCALEDOWN));
            spx_ifft(st->fft_table, st->wtmp2, st->wtmp);
#else
            spx_ifft(st->fft_table, &st->W[j*N], st->wtmp);
#endif
            for(i=0;i<n;i++)
               filt[j*n+i] = PSHR32(MULT16_16(32767,st->wtmp[i]), WEIGHT_SHIFT-NORMALIZE_SCALEDOWN);
         }
      }
         break;
      default:
         speex_warning_int("Unknown speex_echo_ctl request: ", request);
         return -1;
   }
   return 0;
}

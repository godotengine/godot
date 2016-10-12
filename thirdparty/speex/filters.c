/* Copyright (C) 2002-2006 Jean-Marc Valin 
   File: filters.c
   Various analysis/synthesis filters

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


#include "filters.h"
#include "stack_alloc.h"
#include "arch.h"
#include "math_approx.h"
#include "ltp.h"
#include <math.h>

#ifdef _USE_SSE
#include "filters_sse.h"
#elif defined (ARM4_ASM) || defined(ARM5E_ASM)
#include "filters_arm4.h"
#elif defined (BFIN_ASM)
#include "filters_bfin.h"
#endif



void bw_lpc(spx_word16_t gamma, const spx_coef_t *lpc_in, spx_coef_t *lpc_out, int order)
{
   int i;
   spx_word16_t tmp=gamma;
   for (i=0;i<order;i++)
   {
      lpc_out[i] = MULT16_16_P15(tmp,lpc_in[i]);
      tmp = MULT16_16_P15(tmp, gamma);
   }
}

void sanitize_values32(spx_word32_t *vec, spx_word32_t min_val, spx_word32_t max_val, int len)
{
   int i;
   for (i=0;i<len;i++)
   {
      /* It's important we do the test that way so we can catch NaNs, which are neither greater nor smaller */
      if (!(vec[i]>=min_val && vec[i] <= max_val))
      {
         if (vec[i] < min_val)
            vec[i] = min_val;
         else if (vec[i] > max_val)
            vec[i] = max_val;
         else /* Has to be NaN */
            vec[i] = 0;
      }
   }
}

void highpass(const spx_word16_t *x, spx_word16_t *y, int len, int filtID, spx_mem_t *mem)
{
   int i;
#ifdef FIXED_POINT
   const spx_word16_t Pcoef[5][3] = {{16384, -31313, 14991}, {16384, -31569, 15249}, {16384, -31677, 15328}, {16384, -32313, 15947}, {16384, -22446, 6537}};
   const spx_word16_t Zcoef[5][3] = {{15672, -31344, 15672}, {15802, -31601, 15802}, {15847, -31694, 15847}, {16162, -32322, 16162}, {14418, -28836, 14418}};
#else
   const spx_word16_t Pcoef[5][3] = {{1.00000f, -1.91120f, 0.91498f}, {1.00000f, -1.92683f, 0.93071f}, {1.00000f, -1.93338f, 0.93553f}, {1.00000f, -1.97226f, 0.97332f}, {1.00000f, -1.37000f, 0.39900f}};
   const spx_word16_t Zcoef[5][3] = {{0.95654f, -1.91309f, 0.95654f}, {0.96446f, -1.92879f, 0.96446f}, {0.96723f, -1.93445f, 0.96723f}, {0.98645f, -1.97277f, 0.98645f}, {0.88000f, -1.76000f, 0.88000f}};
#endif
   const spx_word16_t *den, *num;
   if (filtID>4)
      filtID=4;
   
   den = Pcoef[filtID]; num = Zcoef[filtID];
   /*return;*/
   for (i=0;i<len;i++)
   {
      spx_word16_t yi;
      spx_word32_t vout = ADD32(MULT16_16(num[0], x[i]),mem[0]);
      yi = EXTRACT16(SATURATE(PSHR32(vout,14),32767));
      mem[0] = ADD32(MAC16_16(mem[1], num[1],x[i]), SHL32(MULT16_32_Q15(-den[1],vout),1));
      mem[1] = ADD32(MULT16_16(num[2],x[i]), SHL32(MULT16_32_Q15(-den[2],vout),1));
      y[i] = yi;
   }
}

#ifdef FIXED_POINT

/* FIXME: These functions are ugly and probably introduce too much error */
void signal_mul(const spx_sig_t *x, spx_sig_t *y, spx_word32_t scale, int len)
{
   int i;
   for (i=0;i<len;i++)
   {
      y[i] = SHL32(MULT16_32_Q14(EXTRACT16(SHR32(x[i],7)),scale),7);
   }
}

void signal_div(const spx_word16_t *x, spx_word16_t *y, spx_word32_t scale, int len)
{
   int i;
   if (scale > SHL32(EXTEND32(SIG_SCALING), 8))
   {
      spx_word16_t scale_1;
      scale = PSHR32(scale, SIG_SHIFT);
      scale_1 = EXTRACT16(PDIV32_16(SHL32(EXTEND32(SIG_SCALING),7),scale));
      for (i=0;i<len;i++)
      {
         y[i] = MULT16_16_P15(scale_1, x[i]);
      }
   } else if (scale > SHR32(EXTEND32(SIG_SCALING), 2)) {
      spx_word16_t scale_1;
      scale = PSHR32(scale, SIG_SHIFT-5);
      scale_1 = DIV32_16(SHL32(EXTEND32(SIG_SCALING),3),scale);
      for (i=0;i<len;i++)
      {
         y[i] = PSHR32(MULT16_16(scale_1, SHL16(x[i],2)),8);
      }
   } else {
      spx_word16_t scale_1;
      scale = PSHR32(scale, SIG_SHIFT-7);
      if (scale < 5)
         scale = 5;
      scale_1 = DIV32_16(SHL32(EXTEND32(SIG_SCALING),3),scale);
      for (i=0;i<len;i++)
      {
         y[i] = PSHR32(MULT16_16(scale_1, SHL16(x[i],2)),6);
      }
   }
}

#else

void signal_mul(const spx_sig_t *x, spx_sig_t *y, spx_word32_t scale, int len)
{
   int i;
   for (i=0;i<len;i++)
      y[i] = scale*x[i];
}

void signal_div(const spx_sig_t *x, spx_sig_t *y, spx_word32_t scale, int len)
{
   int i;
   float scale_1 = 1/scale;
   for (i=0;i<len;i++)
      y[i] = scale_1*x[i];
}
#endif



#ifdef FIXED_POINT



spx_word16_t compute_rms(const spx_sig_t *x, int len)
{
   int i;
   spx_word32_t sum=0;
   spx_sig_t max_val=1;
   int sig_shift;

   for (i=0;i<len;i++)
   {
      spx_sig_t tmp = x[i];
      if (tmp<0)
         tmp = -tmp;
      if (tmp > max_val)
         max_val = tmp;
   }

   sig_shift=0;
   while (max_val>16383)
   {
      sig_shift++;
      max_val >>= 1;
   }

   for (i=0;i<len;i+=4)
   {
      spx_word32_t sum2=0;
      spx_word16_t tmp;
      tmp = EXTRACT16(SHR32(x[i],sig_shift));
      sum2 = MAC16_16(sum2,tmp,tmp);
      tmp = EXTRACT16(SHR32(x[i+1],sig_shift));
      sum2 = MAC16_16(sum2,tmp,tmp);
      tmp = EXTRACT16(SHR32(x[i+2],sig_shift));
      sum2 = MAC16_16(sum2,tmp,tmp);
      tmp = EXTRACT16(SHR32(x[i+3],sig_shift));
      sum2 = MAC16_16(sum2,tmp,tmp);
      sum = ADD32(sum,SHR32(sum2,6));
   }
   
   return EXTRACT16(PSHR32(SHL32(EXTEND32(spx_sqrt(DIV32(sum,len))),(sig_shift+3)),SIG_SHIFT));
}

spx_word16_t compute_rms16(const spx_word16_t *x, int len)
{
   int i;
   spx_word16_t max_val=10; 

   for (i=0;i<len;i++)
   {
      spx_sig_t tmp = x[i];
      if (tmp<0)
         tmp = -tmp;
      if (tmp > max_val)
         max_val = tmp;
   }
   if (max_val>16383)
   {
      spx_word32_t sum=0;
      for (i=0;i<len;i+=4)
      {
         spx_word32_t sum2=0;
         sum2 = MAC16_16(sum2,SHR16(x[i],1),SHR16(x[i],1));
         sum2 = MAC16_16(sum2,SHR16(x[i+1],1),SHR16(x[i+1],1));
         sum2 = MAC16_16(sum2,SHR16(x[i+2],1),SHR16(x[i+2],1));
         sum2 = MAC16_16(sum2,SHR16(x[i+3],1),SHR16(x[i+3],1));
         sum = ADD32(sum,SHR32(sum2,6));
      }
      return SHL16(spx_sqrt(DIV32(sum,len)),4);
   } else {
      spx_word32_t sum=0;
      int sig_shift=0;
      if (max_val < 8192)
         sig_shift=1;
      if (max_val < 4096)
         sig_shift=2;
      if (max_val < 2048)
         sig_shift=3;
      for (i=0;i<len;i+=4)
      {
         spx_word32_t sum2=0;
         sum2 = MAC16_16(sum2,SHL16(x[i],sig_shift),SHL16(x[i],sig_shift));
         sum2 = MAC16_16(sum2,SHL16(x[i+1],sig_shift),SHL16(x[i+1],sig_shift));
         sum2 = MAC16_16(sum2,SHL16(x[i+2],sig_shift),SHL16(x[i+2],sig_shift));
         sum2 = MAC16_16(sum2,SHL16(x[i+3],sig_shift),SHL16(x[i+3],sig_shift));
         sum = ADD32(sum,SHR32(sum2,6));
      }
      return SHL16(spx_sqrt(DIV32(sum,len)),3-sig_shift);   
   }
}

#ifndef OVERRIDE_NORMALIZE16
int normalize16(const spx_sig_t *x, spx_word16_t *y, spx_sig_t max_scale, int len)
{
   int i;
   spx_sig_t max_val=1;
   int sig_shift;
   
   for (i=0;i<len;i++)
   {
      spx_sig_t tmp = x[i];
      if (tmp<0)
         tmp = NEG32(tmp);
      if (tmp >= max_val)
         max_val = tmp;
   }

   sig_shift=0;
   while (max_val>max_scale)
   {
      sig_shift++;
      max_val >>= 1;
   }

   for (i=0;i<len;i++)
      y[i] = EXTRACT16(SHR32(x[i], sig_shift));
   
   return sig_shift;
}
#endif

#else

spx_word16_t compute_rms(const spx_sig_t *x, int len)
{
   int i;
   float sum=0;
   for (i=0;i<len;i++)
   {
      sum += x[i]*x[i];
   }
   return sqrt(.1+sum/len);
}
spx_word16_t compute_rms16(const spx_word16_t *x, int len)
{
   return compute_rms(x, len);
}
#endif



#ifndef OVERRIDE_FILTER_MEM16
void filter_mem16(const spx_word16_t *x, const spx_coef_t *num, const spx_coef_t *den, spx_word16_t *y, int N, int ord, spx_mem_t *mem, char *stack)
{
   int i,j;
   spx_word16_t xi,yi,nyi;
   for (i=0;i<N;i++)
   {
      xi= x[i];
      yi = EXTRACT16(SATURATE(ADD32(EXTEND32(x[i]),PSHR32(mem[0],LPC_SHIFT)),32767));
      nyi = NEG16(yi);
      for (j=0;j<ord-1;j++)
      {
         mem[j] = MAC16_16(MAC16_16(mem[j+1], num[j],xi), den[j],nyi);
      }
      mem[ord-1] = ADD32(MULT16_16(num[ord-1],xi), MULT16_16(den[ord-1],nyi));
      y[i] = yi;
   }
}
#endif

#ifndef OVERRIDE_IIR_MEM16
void iir_mem16(const spx_word16_t *x, const spx_coef_t *den, spx_word16_t *y, int N, int ord, spx_mem_t *mem, char *stack)
{
   int i,j;
   spx_word16_t yi,nyi;

   for (i=0;i<N;i++)
   {
      yi = EXTRACT16(SATURATE(ADD32(EXTEND32(x[i]),PSHR32(mem[0],LPC_SHIFT)),32767));
      nyi = NEG16(yi);
      for (j=0;j<ord-1;j++)
      {
         mem[j] = MAC16_16(mem[j+1],den[j],nyi);
      }
      mem[ord-1] = MULT16_16(den[ord-1],nyi);
      y[i] = yi;
   }
}
#endif

#ifndef OVERRIDE_FIR_MEM16
void fir_mem16(const spx_word16_t *x, const spx_coef_t *num, spx_word16_t *y, int N, int ord, spx_mem_t *mem, char *stack)
{
   int i,j;
   spx_word16_t xi,yi;

   for (i=0;i<N;i++)
   {
      xi=x[i];
      yi = EXTRACT16(SATURATE(ADD32(EXTEND32(x[i]),PSHR32(mem[0],LPC_SHIFT)),32767));
      for (j=0;j<ord-1;j++)
      {
         mem[j] = MAC16_16(mem[j+1], num[j],xi);
      }
      mem[ord-1] = MULT16_16(num[ord-1],xi);
      y[i] = yi;
   }
}
#endif


void syn_percep_zero16(const spx_word16_t *xx, const spx_coef_t *ak, const spx_coef_t *awk1, const spx_coef_t *awk2, spx_word16_t *y, int N, int ord, char *stack)
{
   int i;
   VARDECL(spx_mem_t *mem);
   ALLOC(mem, ord, spx_mem_t);
   for (i=0;i<ord;i++)
      mem[i]=0;
   iir_mem16(xx, ak, y, N, ord, mem, stack);
   for (i=0;i<ord;i++)
      mem[i]=0;
   filter_mem16(y, awk1, awk2, y, N, ord, mem, stack);
}
void residue_percep_zero16(const spx_word16_t *xx, const spx_coef_t *ak, const spx_coef_t *awk1, const spx_coef_t *awk2, spx_word16_t *y, int N, int ord, char *stack)
{
   int i;
   VARDECL(spx_mem_t *mem);
   ALLOC(mem, ord, spx_mem_t);
   for (i=0;i<ord;i++)
      mem[i]=0;
   filter_mem16(xx, ak, awk1, y, N, ord, mem, stack);
   for (i=0;i<ord;i++)
      mem[i]=0;
   fir_mem16(y, awk2, y, N, ord, mem, stack);
}


#ifndef OVERRIDE_COMPUTE_IMPULSE_RESPONSE
void compute_impulse_response(const spx_coef_t *ak, const spx_coef_t *awk1, const spx_coef_t *awk2, spx_word16_t *y, int N, int ord, char *stack)
{
   int i,j;
   spx_word16_t y1, ny1i, ny2i;
   VARDECL(spx_mem_t *mem1);
   VARDECL(spx_mem_t *mem2);
   ALLOC(mem1, ord, spx_mem_t);
   ALLOC(mem2, ord, spx_mem_t);
   
   y[0] = LPC_SCALING;
   for (i=0;i<ord;i++)
      y[i+1] = awk1[i];
   i++;
   for (;i<N;i++)
      y[i] = VERY_SMALL;
   for (i=0;i<ord;i++)
      mem1[i] = mem2[i] = 0;
   for (i=0;i<N;i++)
   {
      y1 = ADD16(y[i], EXTRACT16(PSHR32(mem1[0],LPC_SHIFT)));
      ny1i = NEG16(y1);
      y[i] = PSHR32(ADD32(SHL32(EXTEND32(y1),LPC_SHIFT+1),mem2[0]),LPC_SHIFT);
      ny2i = NEG16(y[i]);
      for (j=0;j<ord-1;j++)
      {
         mem1[j] = MAC16_16(mem1[j+1], awk2[j],ny1i);
         mem2[j] = MAC16_16(mem2[j+1], ak[j],ny2i);
      }
      mem1[ord-1] = MULT16_16(awk2[ord-1],ny1i);
      mem2[ord-1] = MULT16_16(ak[ord-1],ny2i);
   }
}
#endif

/* Decomposes a signal into low-band and high-band using a QMF */
void qmf_decomp(const spx_word16_t *xx, const spx_word16_t *aa, spx_word16_t *y1, spx_word16_t *y2, int N, int M, spx_word16_t *mem, char *stack)
{
   int i,j,k,M2;
   VARDECL(spx_word16_t *a);
   VARDECL(spx_word16_t *x);
   spx_word16_t *x2;
   
   ALLOC(a, M, spx_word16_t);
   ALLOC(x, N+M-1, spx_word16_t);
   x2=x+M-1;
   M2=M>>1;
   for (i=0;i<M;i++)
      a[M-i-1]= aa[i];
   for (i=0;i<M-1;i++)
      x[i]=mem[M-i-2];
   for (i=0;i<N;i++)
      x[i+M-1]=SHR16(xx[i],1);
   for (i=0;i<M-1;i++)
      mem[i]=SHR16(xx[N-i-1],1);
   for (i=0,k=0;i<N;i+=2,k++)
   {
      spx_word32_t y1k=0, y2k=0;
      for (j=0;j<M2;j++)
      {
         y1k=ADD32(y1k,MULT16_16(a[j],ADD16(x[i+j],x2[i-j])));
         y2k=SUB32(y2k,MULT16_16(a[j],SUB16(x[i+j],x2[i-j])));
         j++;
         y1k=ADD32(y1k,MULT16_16(a[j],ADD16(x[i+j],x2[i-j])));
         y2k=ADD32(y2k,MULT16_16(a[j],SUB16(x[i+j],x2[i-j])));
      }
      y1[k] = EXTRACT16(SATURATE(PSHR32(y1k,15),32767));
      y2[k] = EXTRACT16(SATURATE(PSHR32(y2k,15),32767));
   }
}

/* Re-synthesised a signal from the QMF low-band and high-band signals */
void qmf_synth(const spx_word16_t *x1, const spx_word16_t *x2, const spx_word16_t *a, spx_word16_t *y, int N, int M, spx_word16_t *mem1, spx_word16_t *mem2, char *stack)
   /* assumptions:
      all odd x[i] are zero -- well, actually they are left out of the array now
      N and M are multiples of 4 */
{
   int i, j;
   int M2, N2;
   VARDECL(spx_word16_t *xx1);
   VARDECL(spx_word16_t *xx2);
   
   M2 = M>>1;
   N2 = N>>1;
   ALLOC(xx1, M2+N2, spx_word16_t);
   ALLOC(xx2, M2+N2, spx_word16_t);

   for (i = 0; i < N2; i++)
      xx1[i] = x1[N2-1-i];
   for (i = 0; i < M2; i++)
      xx1[N2+i] = mem1[2*i+1];
   for (i = 0; i < N2; i++)
      xx2[i] = x2[N2-1-i];
   for (i = 0; i < M2; i++)
      xx2[N2+i] = mem2[2*i+1];

   for (i = 0; i < N2; i += 2) {
      spx_sig_t y0, y1, y2, y3;
      spx_word16_t x10, x20;

      y0 = y1 = y2 = y3 = 0;
      x10 = xx1[N2-2-i];
      x20 = xx2[N2-2-i];

      for (j = 0; j < M2; j += 2) {
         spx_word16_t x11, x21;
         spx_word16_t a0, a1;

         a0 = a[2*j];
         a1 = a[2*j+1];
         x11 = xx1[N2-1+j-i];
         x21 = xx2[N2-1+j-i];

#ifdef FIXED_POINT
         /* We multiply twice by the same coef to avoid overflows */
         y0 = MAC16_16(MAC16_16(y0, a0, x11), NEG16(a0), x21);
         y1 = MAC16_16(MAC16_16(y1, a1, x11), a1, x21);
         y2 = MAC16_16(MAC16_16(y2, a0, x10), NEG16(a0), x20);
         y3 = MAC16_16(MAC16_16(y3, a1, x10), a1, x20);
#else
         y0 = ADD32(y0,MULT16_16(a0, x11-x21));
         y1 = ADD32(y1,MULT16_16(a1, x11+x21));
         y2 = ADD32(y2,MULT16_16(a0, x10-x20));
         y3 = ADD32(y3,MULT16_16(a1, x10+x20));
#endif
         a0 = a[2*j+2];
         a1 = a[2*j+3];
         x10 = xx1[N2+j-i];
         x20 = xx2[N2+j-i];

#ifdef FIXED_POINT
         /* We multiply twice by the same coef to avoid overflows */
         y0 = MAC16_16(MAC16_16(y0, a0, x10), NEG16(a0), x20);
         y1 = MAC16_16(MAC16_16(y1, a1, x10), a1, x20);
         y2 = MAC16_16(MAC16_16(y2, a0, x11), NEG16(a0), x21);
         y3 = MAC16_16(MAC16_16(y3, a1, x11), a1, x21);
#else
         y0 = ADD32(y0,MULT16_16(a0, x10-x20));
         y1 = ADD32(y1,MULT16_16(a1, x10+x20));
         y2 = ADD32(y2,MULT16_16(a0, x11-x21));
         y3 = ADD32(y3,MULT16_16(a1, x11+x21));
#endif
      }
#ifdef FIXED_POINT
      y[2*i] = EXTRACT16(SATURATE32(PSHR32(y0,15),32767));
      y[2*i+1] = EXTRACT16(SATURATE32(PSHR32(y1,15),32767));
      y[2*i+2] = EXTRACT16(SATURATE32(PSHR32(y2,15),32767));
      y[2*i+3] = EXTRACT16(SATURATE32(PSHR32(y3,15),32767));
#else
      /* Normalize up explicitly if we're in float */
      y[2*i] = 2.f*y0;
      y[2*i+1] = 2.f*y1;
      y[2*i+2] = 2.f*y2;
      y[2*i+3] = 2.f*y3;
#endif
   }

   for (i = 0; i < M2; i++)
      mem1[2*i+1] = xx1[i];
   for (i = 0; i < M2; i++)
      mem2[2*i+1] = xx2[i];
}

#ifdef FIXED_POINT
#if 0
const spx_word16_t shift_filt[3][7] = {{-33,    1043,   -4551,   19959,   19959,   -4551,    1043},
                                 {-98,    1133,   -4425,   29179,    8895,   -2328,     444},
                                 {444,   -2328,    8895,   29179,   -4425,    1133,     -98}};
#else
const spx_word16_t shift_filt[3][7] = {{-390,    1540,   -4993,   20123,   20123,   -4993,    1540},
                                {-1064,    2817,   -6694,   31589,    6837,    -990,    -209},
                                 {-209,    -990,    6837,   31589,   -6694,    2817,   -1064}};
#endif
#else
#if 0
const float shift_filt[3][7] = {{-9.9369e-04, 3.1831e-02, -1.3889e-01, 6.0910e-01, 6.0910e-01, -1.3889e-01, 3.1831e-02},
                          {-0.0029937, 0.0345613, -0.1350474, 0.8904793, 0.2714479, -0.0710304, 0.0135403},
                          {0.0135403, -0.0710304, 0.2714479, 0.8904793, -0.1350474, 0.0345613,  -0.0029937}};
#else
const float shift_filt[3][7] = {{-0.011915f, 0.046995f, -0.152373f, 0.614108f, 0.614108f, -0.152373f, 0.046995f},
                          {-0.0324855f, 0.0859768f, -0.2042986f, 0.9640297f, 0.2086420f, -0.0302054f, -0.0063646f},
                          {-0.0063646f, -0.0302054f, 0.2086420f, 0.9640297f, -0.2042986f, 0.0859768f, -0.0324855f}};
#endif
#endif

int interp_pitch(
spx_word16_t *exc,          /*decoded excitation*/
spx_word16_t *interp,          /*decoded excitation*/
int pitch,               /*pitch period*/
int len
)
{
   int i,j,k;
   spx_word32_t corr[4][7];
   spx_word32_t maxcorr;
   int maxi, maxj;
   for (i=0;i<7;i++)
   {
      corr[0][i] = inner_prod(exc, exc-pitch-3+i, len);
   }
   for (i=0;i<3;i++)
   {
      for (j=0;j<7;j++)
      {
         int i1, i2;
         spx_word32_t tmp=0;
         i1 = 3-j;
         if (i1<0)
            i1 = 0;
         i2 = 10-j;
         if (i2>7)
            i2 = 7;
         for (k=i1;k<i2;k++)
            tmp += MULT16_32_Q15(shift_filt[i][k],corr[0][j+k-3]);
         corr[i+1][j] = tmp;
      }
   }
   maxi=maxj=0;
   maxcorr = corr[0][0];
   for (i=0;i<4;i++)
   {
      for (j=0;j<7;j++)
      {
         if (corr[i][j] > maxcorr)
         {
            maxcorr = corr[i][j];
            maxi=i;
            maxj=j;
         }
      }
   }
   for (i=0;i<len;i++)
   {
      spx_word32_t tmp = 0;
      if (maxi>0)
      {
         for (k=0;k<7;k++)
         {
            tmp += MULT16_16(exc[i-(pitch-maxj+3)+k-3],shift_filt[maxi-1][k]);
         }
      } else {
         tmp = SHL32(exc[i-(pitch-maxj+3)],15);
      }
      interp[i] = PSHR32(tmp,15);
   }
   return pitch-maxj+3;
}

void multicomb(
spx_word16_t *exc,          /*decoded excitation*/
spx_word16_t *new_exc,      /*enhanced excitation*/
spx_coef_t *ak,           /*LPC filter coefs*/
int p,               /*LPC order*/
int nsf,             /*sub-frame size*/
int pitch,           /*pitch period*/
int max_pitch,
spx_word16_t  comb_gain,    /*gain of comb filter*/
char *stack
)
{
   int i; 
   VARDECL(spx_word16_t *iexc);
   spx_word16_t old_ener, new_ener;
   int corr_pitch;
   
   spx_word16_t iexc0_mag, iexc1_mag, exc_mag;
   spx_word32_t corr0, corr1;
   spx_word16_t gain0, gain1;
   spx_word16_t pgain1, pgain2;
   spx_word16_t c1, c2;
   spx_word16_t g1, g2;
   spx_word16_t ngain;
   spx_word16_t gg1, gg2;
#ifdef FIXED_POINT
   int scaledown=0;
#endif
#if 0 /* Set to 1 to enable full pitch search */
   int nol_pitch[6];
   spx_word16_t nol_pitch_coef[6];
   spx_word16_t ol_pitch_coef;
   open_loop_nbest_pitch(exc, 20, 120, nsf, 
                         nol_pitch, nol_pitch_coef, 6, stack);
   corr_pitch=nol_pitch[0];
   ol_pitch_coef = nol_pitch_coef[0];
   /*Try to remove pitch multiples*/
   for (i=1;i<6;i++)
   {
#ifdef FIXED_POINT
      if ((nol_pitch_coef[i]>MULT16_16_Q15(nol_pitch_coef[0],19661)) && 
#else
      if ((nol_pitch_coef[i]>.6*nol_pitch_coef[0]) && 
#endif
         (ABS(2*nol_pitch[i]-corr_pitch)<=2 || ABS(3*nol_pitch[i]-corr_pitch)<=3 || 
         ABS(4*nol_pitch[i]-corr_pitch)<=4 || ABS(5*nol_pitch[i]-corr_pitch)<=5))
      {
         corr_pitch = nol_pitch[i];
      }
   }
#else
   corr_pitch = pitch;
#endif
   
   ALLOC(iexc, 2*nsf, spx_word16_t);
   
   interp_pitch(exc, iexc, corr_pitch, 80);
   if (corr_pitch>max_pitch)
      interp_pitch(exc, iexc+nsf, 2*corr_pitch, 80);
   else
      interp_pitch(exc, iexc+nsf, -corr_pitch, 80);

#ifdef FIXED_POINT
   for (i=0;i<nsf;i++)
   {
      if (ABS16(exc[i])>16383)
      {
         scaledown = 1;
         break;
      }
   }
   if (scaledown)
   {
      for (i=0;i<nsf;i++)
         exc[i] = SHR16(exc[i],1);
      for (i=0;i<2*nsf;i++)
         iexc[i] = SHR16(iexc[i],1);
   }
#endif
   /*interp_pitch(exc, iexc+2*nsf, 2*corr_pitch, 80);*/
   
   /*printf ("%d %d %f\n", pitch, corr_pitch, max_corr*ener_1);*/
   iexc0_mag = spx_sqrt(1000+inner_prod(iexc,iexc,nsf));
   iexc1_mag = spx_sqrt(1000+inner_prod(iexc+nsf,iexc+nsf,nsf));
   exc_mag = spx_sqrt(1+inner_prod(exc,exc,nsf));
   corr0  = inner_prod(iexc,exc,nsf);
   if (corr0<0)
      corr0=0;
   corr1 = inner_prod(iexc+nsf,exc,nsf);
   if (corr1<0)
      corr1=0;
#ifdef FIXED_POINT
   /* Doesn't cost much to limit the ratio and it makes the rest easier */
   if (SHL32(EXTEND32(iexc0_mag),6) < EXTEND32(exc_mag))
      iexc0_mag = ADD16(1,PSHR16(exc_mag,6));
   if (SHL32(EXTEND32(iexc1_mag),6) < EXTEND32(exc_mag))
      iexc1_mag = ADD16(1,PSHR16(exc_mag,6));
#endif
   if (corr0 > MULT16_16(iexc0_mag,exc_mag))
      pgain1 = QCONST16(1., 14);
   else
      pgain1 = PDIV32_16(SHL32(PDIV32(corr0, exc_mag),14),iexc0_mag);
   if (corr1 > MULT16_16(iexc1_mag,exc_mag))
      pgain2 = QCONST16(1., 14);
   else
      pgain2 = PDIV32_16(SHL32(PDIV32(corr1, exc_mag),14),iexc1_mag);
   gg1 = PDIV32_16(SHL32(EXTEND32(exc_mag),8), iexc0_mag);
   gg2 = PDIV32_16(SHL32(EXTEND32(exc_mag),8), iexc1_mag);
   if (comb_gain>0)
   {
#ifdef FIXED_POINT
      c1 = (MULT16_16_Q15(QCONST16(.4,15),comb_gain)+QCONST16(.07,15));
      c2 = QCONST16(.5,15)+MULT16_16_Q14(QCONST16(1.72,14),(c1-QCONST16(.07,15)));
#else
      c1 = .4*comb_gain+.07;
      c2 = .5+1.72*(c1-.07);
#endif
   } else 
   {
      c1=c2=0;
   }
#ifdef FIXED_POINT
   g1 = 32767 - MULT16_16_Q13(MULT16_16_Q15(c2, pgain1),pgain1);
   g2 = 32767 - MULT16_16_Q13(MULT16_16_Q15(c2, pgain2),pgain2);
#else
   g1 = 1-c2*pgain1*pgain1;
   g2 = 1-c2*pgain2*pgain2;
#endif
   if (g1<c1)
      g1 = c1;
   if (g2<c1)
      g2 = c1;
   g1 = (spx_word16_t)PDIV32_16(SHL32(EXTEND32(c1),14),(spx_word16_t)g1);
   g2 = (spx_word16_t)PDIV32_16(SHL32(EXTEND32(c1),14),(spx_word16_t)g2);
   if (corr_pitch>max_pitch)
   {
      gain0 = MULT16_16_Q15(QCONST16(.7,15),MULT16_16_Q14(g1,gg1));
      gain1 = MULT16_16_Q15(QCONST16(.3,15),MULT16_16_Q14(g2,gg2));
   } else {
      gain0 = MULT16_16_Q15(QCONST16(.6,15),MULT16_16_Q14(g1,gg1));
      gain1 = MULT16_16_Q15(QCONST16(.6,15),MULT16_16_Q14(g2,gg2));
   }
   for (i=0;i<nsf;i++)
      new_exc[i] = ADD16(exc[i], EXTRACT16(PSHR32(ADD32(MULT16_16(gain0,iexc[i]), MULT16_16(gain1,iexc[i+nsf])),8)));
   /* FIXME: compute_rms16 is currently not quite accurate enough (but close) */
   new_ener = compute_rms16(new_exc, nsf);
   old_ener = compute_rms16(exc, nsf);
   
   if (old_ener < 1)
      old_ener = 1;
   if (new_ener < 1)
      new_ener = 1;
   if (old_ener > new_ener)
      old_ener = new_ener;
   ngain = PDIV32_16(SHL32(EXTEND32(old_ener),14),new_ener);
   
   for (i=0;i<nsf;i++)
      new_exc[i] = MULT16_16_Q14(ngain, new_exc[i]);
#ifdef FIXED_POINT
   if (scaledown)
   {
      for (i=0;i<nsf;i++)
         exc[i] = SHL16(exc[i],1);
      for (i=0;i<nsf;i++)
         new_exc[i] = SHL16(SATURATE16(new_exc[i],16383),1);
   }
#endif
}


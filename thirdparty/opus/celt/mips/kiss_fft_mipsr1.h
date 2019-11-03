/*Copyright (c) 2013, Xiph.Org Foundation and contributors.

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.*/

#ifndef KISS_FFT_MIPSR1_H
#define KISS_FFT_MIPSR1_H

#if !defined(KISS_FFT_GUTS_H)
#error "This file should only be included from _kiss_fft_guts.h"
#endif

#ifdef FIXED_POINT

#define S_MUL_ADD(a, b, c, d) (S_MUL(a,b)+S_MUL(c,d))
#define S_MUL_SUB(a, b, c, d) (S_MUL(a,b)-S_MUL(c,d))

#undef S_MUL_ADD
static inline int S_MUL_ADD(int a, int b, int c, int d) {
    int m;
    asm volatile("MULT $ac1, %0, %1" : : "r" ((int)a), "r" ((int)b));
    asm volatile("madd $ac1, %0, %1" : : "r" ((int)c), "r" ((int)d));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (m): "i" (15));
    return m;
}

#undef S_MUL_SUB
static inline int S_MUL_SUB(int a, int b, int c, int d) {
    int m;
    asm volatile("MULT $ac1, %0, %1" : : "r" ((int)a), "r" ((int)b));
    asm volatile("msub $ac1, %0, %1" : : "r" ((int)c), "r" ((int)d));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (m): "i" (15));
    return m;
}

#undef C_MUL
#   define C_MUL(m,a,b) (m=C_MUL_fun(a,b))
static inline kiss_fft_cpx C_MUL_fun(kiss_fft_cpx a, kiss_twiddle_cpx b) {
    kiss_fft_cpx m;

    asm volatile("MULT $ac1, %0, %1" : : "r" ((int)a.r), "r" ((int)b.r));
    asm volatile("msub $ac1, %0, %1" : : "r" ((int)a.i), "r" ((int)b.i));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (m.r): "i" (15));
    asm volatile("MULT $ac1, %0, %1" : : "r" ((int)a.r), "r" ((int)b.i));
    asm volatile("madd $ac1, %0, %1" : : "r" ((int)a.i), "r" ((int)b.r));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (m.i): "i" (15));

    return m;
}
#undef C_MULC
#   define C_MULC(m,a,b) (m=C_MULC_fun(a,b))
static inline kiss_fft_cpx C_MULC_fun(kiss_fft_cpx a, kiss_twiddle_cpx b) {
    kiss_fft_cpx m;

    asm volatile("MULT $ac1, %0, %1" : : "r" ((int)a.r), "r" ((int)b.r));
    asm volatile("madd $ac1, %0, %1" : : "r" ((int)a.i), "r" ((int)b.i));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (m.r): "i" (15));
    asm volatile("MULT $ac1, %0, %1" : : "r" ((int)a.i), "r" ((int)b.r));
    asm volatile("msub $ac1, %0, %1" : : "r" ((int)a.r), "r" ((int)b.i));
    asm volatile("EXTR.W %0,$ac1, %1" : "=r" (m.i): "i" (15));

    return m;
}

#endif /* FIXED_POINT */

#define OVERRIDE_kf_bfly5
static void kf_bfly5(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_state *st,
                     int m,
                     int N,
                     int mm
                    )
{
   kiss_fft_cpx *Fout0,*Fout1,*Fout2,*Fout3,*Fout4;
   int i, u;
   kiss_fft_cpx scratch[13];

   const kiss_twiddle_cpx *tw;
   kiss_twiddle_cpx ya,yb;
   kiss_fft_cpx * Fout_beg = Fout;

#ifdef FIXED_POINT
   ya.r = 10126;
   ya.i = -31164;
   yb.r = -26510;
   yb.i = -19261;
#else
   ya = st->twiddles[fstride*m];
   yb = st->twiddles[fstride*2*m];
#endif

   tw=st->twiddles;

   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      Fout0=Fout;
      Fout1=Fout0+m;
      Fout2=Fout0+2*m;
      Fout3=Fout0+3*m;
      Fout4=Fout0+4*m;

      /* For non-custom modes, m is guaranteed to be a multiple of 4. */
      for ( u=0; u<m; ++u ) {
         scratch[0] = *Fout0;


         C_MUL(scratch[1] ,*Fout1, tw[u*fstride]);
         C_MUL(scratch[2] ,*Fout2, tw[2*u*fstride]);
         C_MUL(scratch[3] ,*Fout3, tw[3*u*fstride]);
         C_MUL(scratch[4] ,*Fout4, tw[4*u*fstride]);

         C_ADD( scratch[7],scratch[1],scratch[4]);
         C_SUB( scratch[10],scratch[1],scratch[4]);
         C_ADD( scratch[8],scratch[2],scratch[3]);
         C_SUB( scratch[9],scratch[2],scratch[3]);

         Fout0->r += scratch[7].r + scratch[8].r;
         Fout0->i += scratch[7].i + scratch[8].i;
         scratch[5].r = scratch[0].r + S_MUL_ADD(scratch[7].r,ya.r,scratch[8].r,yb.r);
         scratch[5].i = scratch[0].i + S_MUL_ADD(scratch[7].i,ya.r,scratch[8].i,yb.r);

         scratch[6].r =  S_MUL_ADD(scratch[10].i,ya.i,scratch[9].i,yb.i);
         scratch[6].i =  -S_MUL_ADD(scratch[10].r,ya.i,scratch[9].r,yb.i);

         C_SUB(*Fout1,scratch[5],scratch[6]);
         C_ADD(*Fout4,scratch[5],scratch[6]);

         scratch[11].r = scratch[0].r + S_MUL_ADD(scratch[7].r,yb.r,scratch[8].r,ya.r);
         scratch[11].i = scratch[0].i + S_MUL_ADD(scratch[7].i,yb.r,scratch[8].i,ya.r);

         scratch[12].r =  S_MUL_SUB(scratch[9].i,ya.i,scratch[10].i,yb.i);
         scratch[12].i =  S_MUL_SUB(scratch[10].r,yb.i,scratch[9].r,ya.i);

         C_ADD(*Fout2,scratch[11],scratch[12]);
         C_SUB(*Fout3,scratch[11],scratch[12]);

         ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
      }
   }
}


#endif /* KISS_FFT_MIPSR1_H */

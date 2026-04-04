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

#if __mips == 32 && defined (__mips_dsp)

static inline int S_MUL_ADD(int a, int b, int c, int d) {
    long long acc = __builtin_mips_mult(a, b);
    acc = __builtin_mips_madd(acc, c, d);
    return __builtin_mips_extr_w(acc, 15);
}

static inline int S_MUL_SUB(int a, int b, int c, int d) {
    long long acc = __builtin_mips_mult(a, b);
    acc = __builtin_mips_msub(acc, c, d);
    return __builtin_mips_extr_w(acc, 15);
}

#undef C_MUL
#   define C_MUL(m,a,b) (m=C_MUL_fun(a,b))
static inline kiss_fft_cpx C_MUL_fun(kiss_fft_cpx a, kiss_twiddle_cpx b) {
    kiss_fft_cpx m;

    long long acc1 = __builtin_mips_mult((int)a.r, (int)b.r);
    long long acc2 = __builtin_mips_mult((int)a.r, (int)b.i);
    acc1 = __builtin_mips_msub(acc1, (int)a.i, (int)b.i);
    acc2 = __builtin_mips_madd(acc2, (int)a.i, (int)b.r);
    m.r = __builtin_mips_extr_w(acc1, 15);
    m.i = __builtin_mips_extr_w(acc2, 15);
    return m;
}
#undef C_MULC
#   define C_MULC(m,a,b) (m=C_MULC_fun(a,b))
static inline kiss_fft_cpx C_MULC_fun(kiss_fft_cpx a, kiss_twiddle_cpx b) {
    kiss_fft_cpx m;

    long long acc1 = __builtin_mips_mult((int)a.r, (int)b.r);
    long long acc2 = __builtin_mips_mult((int)a.i, (int)b.r);
    acc1 = __builtin_mips_madd(acc1, (int)a.i, (int)b.i);
    acc2 = __builtin_mips_msub(acc2, (int)a.r, (int)b.i);
    m.r = __builtin_mips_extr_w(acc1, 15);
    m.i = __builtin_mips_extr_w(acc2, 15);
    return m;
}

#define OVERRIDE_kf_bfly5

#elif __mips == 32 && defined(__mips_isa_rev) && __mips_isa_rev < 6

static inline int S_MUL_ADD(int a, int b, int c, int d) {
    long long acc;

    asm volatile (
            "mult %[a], %[b]  \n"
            "madd %[c], %[d]  \n"
        : [acc] "=x"(acc)
        : [a] "r"(a), [b] "r"(b), [c] "r"(c), [d] "r"(d)
        :
    );
    return (int)(acc >> 15);
}

static inline int S_MUL_SUB(int a, int b, int c, int d) {
    long long acc;

    asm volatile (
            "mult %[a], %[b]  \n"
            "msub %[c], %[d]  \n"
        : [acc] "=x"(acc)
        : [a] "r"(a), [b] "r"(b), [c] "r"(c), [d] "r"(d)
        :
    );
    return (int)(acc >> 15);
}

#undef C_MUL
#   define C_MUL(m,a,b) (m=C_MUL_fun(a,b))
static inline kiss_fft_cpx C_MUL_fun(kiss_fft_cpx a, kiss_twiddle_cpx b) {
    kiss_fft_cpx m;

    m.r = S_MUL_SUB(a.r, b.r, a.i, b.i);
    m.i = S_MUL_ADD(a.r, b.i, a.i, b.r);

    return m;
}

#undef C_MULC
#   define C_MULC(m,a,b) (m=C_MULC_fun(a,b))
static inline kiss_fft_cpx C_MULC_fun(kiss_fft_cpx a, kiss_twiddle_cpx b) {
    kiss_fft_cpx m;

    m.r = S_MUL_ADD(a.r, b.r, a.i, b.i);
    m.i = S_MUL_SUB(a.i, b.r, a.r, b.i);

    return m;
}

#define OVERRIDE_kf_bfly5

#endif

#endif /* FIXED_POINT */

#if defined(OVERRIDE_kf_bfly5)

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

#endif /* defined(OVERRIDE_kf_bfly5) */

#define OVERRIDE_fft_downshift
/* Just unroll tight loop, should be ok for any mips */
static void fft_downshift(kiss_fft_cpx *x, int N, int *total, int step) {
    int shift;
    shift = IMIN(step, *total);
    *total -= shift;
    if (shift == 1) {
        int i;
        for (i = 0; i < N - 1; i += 2) {
            x[i].r   = SHR32(x[i].r,   1);
            x[i].i   = SHR32(x[i].i,   1);
            x[i+1].r = SHR32(x[i+1].r, 1);
            x[i+1].i = SHR32(x[i+1].i, 1);
        }
        if (N & 1) {
            x[i].r = SHR32(x[i].r, 1);
            x[i].i = SHR32(x[i].i, 1);
        }
    } else if (shift > 0) {
        int i;
        for (i = 0; i < N - 3; i += 4) {
            x[i].r   = PSHR32(x[i].r,   shift);
            x[i].i   = PSHR32(x[i].i,   shift);
            x[i+1].r = PSHR32(x[i+1].r, shift);
            x[i+1].i = PSHR32(x[i+1].i, shift);
            x[i+2].r = PSHR32(x[i+2].r, shift);
            x[i+2].i = PSHR32(x[i+2].i, shift);
            x[i+3].r = PSHR32(x[i+3].r, shift);
            x[i+3].i = PSHR32(x[i+3].i, shift);
        }
        switch (N & 3) {
        case 3:
            x[i].r   = PSHR32(x[i].r,   shift);
            x[i].i   = PSHR32(x[i].i,   shift);
            x[i+1].r = PSHR32(x[i+1].r, shift);
            x[i+1].i = PSHR32(x[i+1].i, shift);
            x[i+2].r = PSHR32(x[i+2].r, shift);
            x[i+2].i = PSHR32(x[i+2].i, shift);
            break;
        case 2:
            x[i].r   = PSHR32(x[i].r,   shift);
            x[i].i   = PSHR32(x[i].i,   shift);
            x[i+1].r = PSHR32(x[i+1].r, shift);
            x[i+1].i = PSHR32(x[i+1].i, shift);
            break;
        case 1:
            x[i].r   = PSHR32(x[i].r,   shift);
            x[i].i   = PSHR32(x[i].i,   shift);
            break;
        case 0:
            break;
        }
    }
}

#endif /* KISS_FFT_MIPSR1_H */

/*Copyright (c) 2003-2004, Mark Borgerding
  Lots of modifications by Jean-Marc Valin
  Copyright (c) 2005-2007, Xiph.Org Foundation
  Copyright (c) 2008,      Xiph.Org Foundation, CSIRO

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

/* This code is originally from Mark Borgerding's KISS-FFT but has been
   heavily modified to better suit Opus */

#ifndef SKIP_CONFIG_H
#  ifdef OPUS_ENABLED
#include "opus/opus_config.h"
#  endif
#endif

#include "opus/celt/_kiss_fft_guts.h"
#include "opus/celt/arch.h"
#include "opus/celt/os_support.h"
#include "opus/celt/mathops.h"
#include "opus/celt/stack_alloc.h"

/* The guts header contains all the multiplication and addition macros that are defined for
   complex numbers.  It also delares the kf_ internal functions.
*/

static void kf_bfly2(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_state *st,
                     int m,
                     int N,
                     int mm
                    )
{
   kiss_fft_cpx * Fout2;
   const kiss_twiddle_cpx * tw1;
   int i,j;
   kiss_fft_cpx * Fout_beg = Fout;
   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      Fout2 = Fout + m;
      tw1 = st->twiddles;
      for(j=0;j<m;j++)
      {
         kiss_fft_cpx t;
         Fout->r = SHR32(Fout->r, 1);Fout->i = SHR32(Fout->i, 1);
         Fout2->r = SHR32(Fout2->r, 1);Fout2->i = SHR32(Fout2->i, 1);
         C_MUL (t,  *Fout2 , *tw1);
         tw1 += fstride;
         C_SUB( *Fout2 ,  *Fout , t );
         C_ADDTO( *Fout ,  t );
         ++Fout2;
         ++Fout;
      }
   }
}

static void ki_bfly2(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_state *st,
                     int m,
                     int N,
                     int mm
                    )
{
   kiss_fft_cpx * Fout2;
   const kiss_twiddle_cpx * tw1;
   kiss_fft_cpx t;
   int i,j;
   kiss_fft_cpx * Fout_beg = Fout;
   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      Fout2 = Fout + m;
      tw1 = st->twiddles;
      for(j=0;j<m;j++)
      {
         C_MULC (t,  *Fout2 , *tw1);
         tw1 += fstride;
         C_SUB( *Fout2 ,  *Fout , t );
         C_ADDTO( *Fout ,  t );
         ++Fout2;
         ++Fout;
      }
   }
}

static void kf_bfly4(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_state *st,
                     int m,
                     int N,
                     int mm
                    )
{
   const kiss_twiddle_cpx *tw1,*tw2,*tw3;
   kiss_fft_cpx scratch[6];
   const size_t m2=2*m;
   const size_t m3=3*m;
   int i, j;

   kiss_fft_cpx * Fout_beg = Fout;
   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      tw3 = tw2 = tw1 = st->twiddles;
      for (j=0;j<m;j++)
      {
         C_MUL4(scratch[0],Fout[m] , *tw1 );
         C_MUL4(scratch[1],Fout[m2] , *tw2 );
         C_MUL4(scratch[2],Fout[m3] , *tw3 );

         Fout->r = PSHR32(Fout->r, 2);
         Fout->i = PSHR32(Fout->i, 2);
         C_SUB( scratch[5] , *Fout, scratch[1] );
         C_ADDTO(*Fout, scratch[1]);
         C_ADD( scratch[3] , scratch[0] , scratch[2] );
         C_SUB( scratch[4] , scratch[0] , scratch[2] );
         C_SUB( Fout[m2], *Fout, scratch[3] );
         tw1 += fstride;
         tw2 += fstride*2;
         tw3 += fstride*3;
         C_ADDTO( *Fout , scratch[3] );

         Fout[m].r = scratch[5].r + scratch[4].i;
         Fout[m].i = scratch[5].i - scratch[4].r;
         Fout[m3].r = scratch[5].r - scratch[4].i;
         Fout[m3].i = scratch[5].i + scratch[4].r;
         ++Fout;
      }
   }
}

static void ki_bfly4(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_state *st,
                     int m,
                     int N,
                     int mm
                    )
{
   const kiss_twiddle_cpx *tw1,*tw2,*tw3;
   kiss_fft_cpx scratch[6];
   const size_t m2=2*m;
   const size_t m3=3*m;
   int i, j;

   kiss_fft_cpx * Fout_beg = Fout;
   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      tw3 = tw2 = tw1 = st->twiddles;
      for (j=0;j<m;j++)
      {
         C_MULC(scratch[0],Fout[m] , *tw1 );
         C_MULC(scratch[1],Fout[m2] , *tw2 );
         C_MULC(scratch[2],Fout[m3] , *tw3 );

         C_SUB( scratch[5] , *Fout, scratch[1] );
         C_ADDTO(*Fout, scratch[1]);
         C_ADD( scratch[3] , scratch[0] , scratch[2] );
         C_SUB( scratch[4] , scratch[0] , scratch[2] );
         C_SUB( Fout[m2], *Fout, scratch[3] );
         tw1 += fstride;
         tw2 += fstride*2;
         tw3 += fstride*3;
         C_ADDTO( *Fout , scratch[3] );

         Fout[m].r = scratch[5].r - scratch[4].i;
         Fout[m].i = scratch[5].i + scratch[4].r;
         Fout[m3].r = scratch[5].r + scratch[4].i;
         Fout[m3].i = scratch[5].i - scratch[4].r;
         ++Fout;
      }
   }
}

#ifndef RADIX_TWO_ONLY

static void kf_bfly3(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_state *st,
                     int m,
                     int N,
                     int mm
                    )
{
   int i;
   size_t k;
   const size_t m2 = 2*m;
   const kiss_twiddle_cpx *tw1,*tw2;
   kiss_fft_cpx scratch[5];
   kiss_twiddle_cpx epi3;

   kiss_fft_cpx * Fout_beg = Fout;
   epi3 = st->twiddles[fstride*m];
   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      tw1=tw2=st->twiddles;
      k=m;
      do {
         C_FIXDIV(*Fout,3); C_FIXDIV(Fout[m],3); C_FIXDIV(Fout[m2],3);

         C_MUL(scratch[1],Fout[m] , *tw1);
         C_MUL(scratch[2],Fout[m2] , *tw2);

         C_ADD(scratch[3],scratch[1],scratch[2]);
         C_SUB(scratch[0],scratch[1],scratch[2]);
         tw1 += fstride;
         tw2 += fstride*2;

         Fout[m].r = Fout->r - HALF_OF(scratch[3].r);
         Fout[m].i = Fout->i - HALF_OF(scratch[3].i);

         C_MULBYSCALAR( scratch[0] , epi3.i );

         C_ADDTO(*Fout,scratch[3]);

         Fout[m2].r = Fout[m].r + scratch[0].i;
         Fout[m2].i = Fout[m].i - scratch[0].r;

         Fout[m].r -= scratch[0].i;
         Fout[m].i += scratch[0].r;

         ++Fout;
      } while(--k);
   }
}

static void ki_bfly3(
                     kiss_fft_cpx * Fout,
                     const size_t fstride,
                     const kiss_fft_state *st,
                     int m,
                     int N,
                     int mm
                    )
{
   int i, k;
   const size_t m2 = 2*m;
   const kiss_twiddle_cpx *tw1,*tw2;
   kiss_fft_cpx scratch[5];
   kiss_twiddle_cpx epi3;

   kiss_fft_cpx * Fout_beg = Fout;
   epi3 = st->twiddles[fstride*m];
   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      tw1=tw2=st->twiddles;
      k=m;
      do{

         C_MULC(scratch[1],Fout[m] , *tw1);
         C_MULC(scratch[2],Fout[m2] , *tw2);

         C_ADD(scratch[3],scratch[1],scratch[2]);
         C_SUB(scratch[0],scratch[1],scratch[2]);
         tw1 += fstride;
         tw2 += fstride*2;

         Fout[m].r = Fout->r - HALF_OF(scratch[3].r);
         Fout[m].i = Fout->i - HALF_OF(scratch[3].i);

         C_MULBYSCALAR( scratch[0] , -epi3.i );

         C_ADDTO(*Fout,scratch[3]);

         Fout[m2].r = Fout[m].r + scratch[0].i;
         Fout[m2].i = Fout[m].i - scratch[0].r;

         Fout[m].r -= scratch[0].i;
         Fout[m].i += scratch[0].r;

         ++Fout;
      }while(--k);
   }
}

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
   const kiss_twiddle_cpx * twiddles = st->twiddles;
   const kiss_twiddle_cpx *tw;
   kiss_twiddle_cpx ya,yb;
   kiss_fft_cpx * Fout_beg = Fout;

   ya = twiddles[fstride*m];
   yb = twiddles[fstride*2*m];
   tw=st->twiddles;

   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      Fout0=Fout;
      Fout1=Fout0+m;
      Fout2=Fout0+2*m;
      Fout3=Fout0+3*m;
      Fout4=Fout0+4*m;

      for ( u=0; u<m; ++u ) {
         C_FIXDIV( *Fout0,5); C_FIXDIV( *Fout1,5); C_FIXDIV( *Fout2,5); C_FIXDIV( *Fout3,5); C_FIXDIV( *Fout4,5);
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

         scratch[5].r = scratch[0].r + S_MUL(scratch[7].r,ya.r) + S_MUL(scratch[8].r,yb.r);
         scratch[5].i = scratch[0].i + S_MUL(scratch[7].i,ya.r) + S_MUL(scratch[8].i,yb.r);

         scratch[6].r =  S_MUL(scratch[10].i,ya.i) + S_MUL(scratch[9].i,yb.i);
         scratch[6].i = -S_MUL(scratch[10].r,ya.i) - S_MUL(scratch[9].r,yb.i);

         C_SUB(*Fout1,scratch[5],scratch[6]);
         C_ADD(*Fout4,scratch[5],scratch[6]);

         scratch[11].r = scratch[0].r + S_MUL(scratch[7].r,yb.r) + S_MUL(scratch[8].r,ya.r);
         scratch[11].i = scratch[0].i + S_MUL(scratch[7].i,yb.r) + S_MUL(scratch[8].i,ya.r);
         scratch[12].r = - S_MUL(scratch[10].i,yb.i) + S_MUL(scratch[9].i,ya.i);
         scratch[12].i = S_MUL(scratch[10].r,yb.i) - S_MUL(scratch[9].r,ya.i);

         C_ADD(*Fout2,scratch[11],scratch[12]);
         C_SUB(*Fout3,scratch[11],scratch[12]);

         ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
      }
   }
}

static void ki_bfly5(
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
   const kiss_twiddle_cpx * twiddles = st->twiddles;
   const kiss_twiddle_cpx *tw;
   kiss_twiddle_cpx ya,yb;
   kiss_fft_cpx * Fout_beg = Fout;

   ya = twiddles[fstride*m];
   yb = twiddles[fstride*2*m];
   tw=st->twiddles;

   for (i=0;i<N;i++)
   {
      Fout = Fout_beg + i*mm;
      Fout0=Fout;
      Fout1=Fout0+m;
      Fout2=Fout0+2*m;
      Fout3=Fout0+3*m;
      Fout4=Fout0+4*m;

      for ( u=0; u<m; ++u ) {
         scratch[0] = *Fout0;

         C_MULC(scratch[1] ,*Fout1, tw[u*fstride]);
         C_MULC(scratch[2] ,*Fout2, tw[2*u*fstride]);
         C_MULC(scratch[3] ,*Fout3, tw[3*u*fstride]);
         C_MULC(scratch[4] ,*Fout4, tw[4*u*fstride]);

         C_ADD( scratch[7],scratch[1],scratch[4]);
         C_SUB( scratch[10],scratch[1],scratch[4]);
         C_ADD( scratch[8],scratch[2],scratch[3]);
         C_SUB( scratch[9],scratch[2],scratch[3]);

         Fout0->r += scratch[7].r + scratch[8].r;
         Fout0->i += scratch[7].i + scratch[8].i;

         scratch[5].r = scratch[0].r + S_MUL(scratch[7].r,ya.r) + S_MUL(scratch[8].r,yb.r);
         scratch[5].i = scratch[0].i + S_MUL(scratch[7].i,ya.r) + S_MUL(scratch[8].i,yb.r);

         scratch[6].r = -S_MUL(scratch[10].i,ya.i) - S_MUL(scratch[9].i,yb.i);
         scratch[6].i =  S_MUL(scratch[10].r,ya.i) + S_MUL(scratch[9].r,yb.i);

         C_SUB(*Fout1,scratch[5],scratch[6]);
         C_ADD(*Fout4,scratch[5],scratch[6]);

         scratch[11].r = scratch[0].r + S_MUL(scratch[7].r,yb.r) + S_MUL(scratch[8].r,ya.r);
         scratch[11].i = scratch[0].i + S_MUL(scratch[7].i,yb.r) + S_MUL(scratch[8].i,ya.r);
         scratch[12].r =  S_MUL(scratch[10].i,yb.i) - S_MUL(scratch[9].i,ya.i);
         scratch[12].i = -S_MUL(scratch[10].r,yb.i) + S_MUL(scratch[9].r,ya.i);

         C_ADD(*Fout2,scratch[11],scratch[12]);
         C_SUB(*Fout3,scratch[11],scratch[12]);

         ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
      }
   }
}

#endif


#ifdef CUSTOM_MODES

static
void compute_bitrev_table(
         int Fout,
         opus_int16 *f,
         const size_t fstride,
         int in_stride,
         opus_int16 * factors,
         const kiss_fft_state *st
            )
{
   const int p=*factors++; /* the radix  */
   const int m=*factors++; /* stage's fft length/p */

    /*printf ("fft %d %d %d %d %d %d\n", p*m, m, p, s2, fstride*in_stride, N);*/
   if (m==1)
   {
      int j;
      for (j=0;j<p;j++)
      {
         *f = Fout+j;
         f += fstride*in_stride;
      }
   } else {
      int j;
      for (j=0;j<p;j++)
      {
         compute_bitrev_table( Fout , f, fstride*p, in_stride, factors,st);
         f += fstride*in_stride;
         Fout += m;
      }
   }
}

/*  facbuf is populated by p1,m1,p2,m2, ...
    where
    p[i] * m[i] = m[i-1]
    m0 = n                  */
static
int kf_factor(int n,opus_int16 * facbuf)
{
    int p=4;

    /*factor out powers of 4, powers of 2, then any remaining primes */
    do {
        while (n % p) {
            switch (p) {
                case 4: p = 2; break;
                case 2: p = 3; break;
                default: p += 2; break;
            }
            if (p>32000 || (opus_int32)p*(opus_int32)p > n)
                p = n;          /* no more factors, skip to end */
        }
        n /= p;
#ifdef RADIX_TWO_ONLY
        if (p!=2 && p != 4)
#else
        if (p>5)
#endif
        {
           return 0;
        }
        *facbuf++ = p;
        *facbuf++ = n;
    } while (n > 1);
    return 1;
}

static void compute_twiddles(kiss_twiddle_cpx *twiddles, int nfft)
{
   int i;
#ifdef OPUS_FIXED_POINT
   for (i=0;i<nfft;++i) {
      opus_val32 phase = -i;
      kf_cexp2(twiddles+i, DIV32(SHL32(phase,17),nfft));
   }
#else
   for (i=0;i<nfft;++i) {
      const double pi=3.14159265358979323846264338327;
      double phase = ( -2*pi /nfft ) * i;
      kf_cexp(twiddles+i, phase );
   }
#endif
}

/*
 *
 * Allocates all necessary storage space for the fft and ifft.
 * The return value is a contiguous block of memory.  As such,
 * It can be freed with free().
 * */
kiss_fft_state *opus_fft_alloc_twiddles(int nfft,void * mem,size_t * lenmem,  const kiss_fft_state *base)
{
    kiss_fft_state *st=NULL;
    size_t memneeded = sizeof(struct kiss_fft_state); /* twiddle factors*/

    if ( lenmem==NULL ) {
        st = ( kiss_fft_state*)KISS_FFT_MALLOC( memneeded );
    }else{
        if (mem != NULL && *lenmem >= memneeded)
            st = (kiss_fft_state*)mem;
        *lenmem = memneeded;
    }
    if (st) {
        opus_int16 *bitrev;
        kiss_twiddle_cpx *twiddles;

        st->nfft=nfft;
#ifndef OPUS_FIXED_POINT
        st->scale = 1.f/nfft;
#endif
        if (base != NULL)
        {
           st->twiddles = base->twiddles;
           st->shift = 0;
           while (nfft<<st->shift != base->nfft && st->shift < 32)
              st->shift++;
           if (st->shift>=32)
              goto fail;
        } else {
           st->twiddles = twiddles = (kiss_twiddle_cpx*)KISS_FFT_MALLOC(sizeof(kiss_twiddle_cpx)*nfft);
           compute_twiddles(twiddles, nfft);
           st->shift = -1;
        }
        if (!kf_factor(nfft,st->factors))
        {
           goto fail;
        }

        /* bitrev */
        st->bitrev = bitrev = (opus_int16*)KISS_FFT_MALLOC(sizeof(opus_int16)*nfft);
        if (st->bitrev==NULL)
            goto fail;
        compute_bitrev_table(0, bitrev, 1,1, st->factors,st);
    }
    return st;
fail:
    opus_fft_free(st);
    return NULL;
}

kiss_fft_state *opus_fft_alloc(int nfft,void * mem,size_t * lenmem )
{
   return opus_fft_alloc_twiddles(nfft, mem, lenmem, NULL);
}

void opus_fft_free(const kiss_fft_state *cfg)
{
   if (cfg)
   {
      opus_free((opus_int16*)cfg->bitrev);
      if (cfg->shift < 0)
         opus_free((kiss_twiddle_cpx*)cfg->twiddles);
      opus_free((kiss_fft_state*)cfg);
   }
}

#endif /* CUSTOM_MODES */

void opus_fft(const kiss_fft_state *st,const kiss_fft_cpx *fin,kiss_fft_cpx *fout)
{
    int m2, m;
    int p;
    int L;
    int fstride[MAXFACTORS];
    int i;
    int shift;

    /* st->shift can be -1 */
    shift = st->shift>0 ? st->shift : 0;

    celt_assert2 (fin != fout, "In-place FFT not supported");
    /* Bit-reverse the input */
    for (i=0;i<st->nfft;i++)
    {
       fout[st->bitrev[i]] = fin[i];
#ifndef OPUS_FIXED_POINT
       fout[st->bitrev[i]].r *= st->scale;
       fout[st->bitrev[i]].i *= st->scale;
#endif
    }

    fstride[0] = 1;
    L=0;
    do {
       p = st->factors[2*L];
       m = st->factors[2*L+1];
       fstride[L+1] = fstride[L]*p;
       L++;
    } while(m!=1);
    m = st->factors[2*L-1];
    for (i=L-1;i>=0;i--)
    {
       if (i!=0)
          m2 = st->factors[2*i-1];
       else
          m2 = 1;
       switch (st->factors[2*i])
       {
       case 2:
          kf_bfly2(fout,fstride[i]<<shift,st,m, fstride[i], m2);
          break;
       case 4:
          kf_bfly4(fout,fstride[i]<<shift,st,m, fstride[i], m2);
          break;
 #ifndef RADIX_TWO_ONLY
       case 3:
          kf_bfly3(fout,fstride[i]<<shift,st,m, fstride[i], m2);
          break;
       case 5:
          kf_bfly5(fout,fstride[i]<<shift,st,m, fstride[i], m2);
          break;
 #endif
       }
       m = m2;
    }
}

void opus_ifft(const kiss_fft_state *st,const kiss_fft_cpx *fin,kiss_fft_cpx *fout)
{
   int m2, m;
   int p;
   int L;
   int fstride[MAXFACTORS];
   int i;
   int shift;

   /* st->shift can be -1 */
   shift = st->shift>0 ? st->shift : 0;
   celt_assert2 (fin != fout, "In-place FFT not supported");
   /* Bit-reverse the input */
   for (i=0;i<st->nfft;i++)
      fout[st->bitrev[i]] = fin[i];

   fstride[0] = 1;
   L=0;
   do {
      p = st->factors[2*L];
      m = st->factors[2*L+1];
      fstride[L+1] = fstride[L]*p;
      L++;
   } while(m!=1);
   m = st->factors[2*L-1];
   for (i=L-1;i>=0;i--)
   {
      if (i!=0)
         m2 = st->factors[2*i-1];
      else
         m2 = 1;
      switch (st->factors[2*i])
      {
      case 2:
         ki_bfly2(fout,fstride[i]<<shift,st,m, fstride[i], m2);
         break;
      case 4:
         ki_bfly4(fout,fstride[i]<<shift,st,m, fstride[i], m2);
         break;
#ifndef RADIX_TWO_ONLY
      case 3:
         ki_bfly3(fout,fstride[i]<<shift,st,m, fstride[i], m2);
         break;
      case 5:
         ki_bfly5(fout,fstride[i]<<shift,st,m, fstride[i], m2);
         break;
#endif
      }
      m = m2;
   }
}


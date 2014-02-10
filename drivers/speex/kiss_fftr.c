/*
Copyright (c) 2003-2004, Mark Borgerding

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "config.h"


#include "os_support.h"
#include "kiss_fftr.h"
#include "_kiss_fft_guts.h"

struct kiss_fftr_state{
    kiss_fft_cfg substate;
    kiss_fft_cpx * tmpbuf;
    kiss_fft_cpx * super_twiddles;
#ifdef USE_SIMD    
    long pad;
#endif    
};

kiss_fftr_cfg kiss_fftr_alloc(int nfft,int inverse_fft,void * mem,size_t * lenmem)
{
    int i;
    kiss_fftr_cfg st = NULL;
    size_t subsize, memneeded;

    if (nfft & 1) {
        speex_warning("Real FFT optimization must be even.\n");
        return NULL;
    }
    nfft >>= 1;

    kiss_fft_alloc (nfft, inverse_fft, NULL, &subsize);
    memneeded = sizeof(struct kiss_fftr_state) + subsize + sizeof(kiss_fft_cpx) * ( nfft * 2);

    if (lenmem == NULL) {
        st = (kiss_fftr_cfg) KISS_FFT_MALLOC (memneeded);
    } else {
        if (*lenmem >= memneeded)
            st = (kiss_fftr_cfg) mem;
        *lenmem = memneeded;
    }
    if (!st)
        return NULL;

    st->substate = (kiss_fft_cfg) (st + 1); /*just beyond kiss_fftr_state struct */
    st->tmpbuf = (kiss_fft_cpx *) (((char *) st->substate) + subsize);
    st->super_twiddles = st->tmpbuf + nfft;
    kiss_fft_alloc(nfft, inverse_fft, st->substate, &subsize);

#ifdef FIXED_POINT
    for (i=0;i<nfft;++i) {
       spx_word32_t phase = i+(nfft>>1);
       if (!inverse_fft)
          phase = -phase;
       kf_cexp2(st->super_twiddles+i, DIV32(SHL32(phase,16),nfft));
    }
#else
    for (i=0;i<nfft;++i) {
       const double pi=3.14159265358979323846264338327;
       double phase = pi*(((double)i) /nfft + .5);
       if (!inverse_fft)
          phase = -phase;
       kf_cexp(st->super_twiddles+i, phase );
    }
#endif
    return st;
}

void kiss_fftr(kiss_fftr_cfg st,const kiss_fft_scalar *timedata,kiss_fft_cpx *freqdata)
{
    /* input buffer timedata is stored row-wise */
    int k,ncfft;
    kiss_fft_cpx fpnk,fpk,f1k,f2k,tw,tdc;

    if ( st->substate->inverse) {
        speex_fatal("kiss fft usage error: improper alloc\n");
    }

    ncfft = st->substate->nfft;

    /*perform the parallel fft of two real signals packed in real,imag*/
    kiss_fft( st->substate , (const kiss_fft_cpx*)timedata, st->tmpbuf );
    /* The real part of the DC element of the frequency spectrum in st->tmpbuf
     * contains the sum of the even-numbered elements of the input time sequence
     * The imag part is the sum of the odd-numbered elements
     *
     * The sum of tdc.r and tdc.i is the sum of the input time sequence. 
     *      yielding DC of input time sequence
     * The difference of tdc.r - tdc.i is the sum of the input (dot product) [1,-1,1,-1... 
     *      yielding Nyquist bin of input time sequence
     */
 
    tdc.r = st->tmpbuf[0].r;
    tdc.i = st->tmpbuf[0].i;
    C_FIXDIV(tdc,2);
    CHECK_OVERFLOW_OP(tdc.r ,+, tdc.i);
    CHECK_OVERFLOW_OP(tdc.r ,-, tdc.i);
    freqdata[0].r = tdc.r + tdc.i;
    freqdata[ncfft].r = tdc.r - tdc.i;
#ifdef USE_SIMD    
    freqdata[ncfft].i = freqdata[0].i = _mm_set1_ps(0);
#else
    freqdata[ncfft].i = freqdata[0].i = 0;
#endif

    for ( k=1;k <= ncfft/2 ; ++k ) {
        fpk    = st->tmpbuf[k]; 
        fpnk.r =   st->tmpbuf[ncfft-k].r;
        fpnk.i = - st->tmpbuf[ncfft-k].i;
        C_FIXDIV(fpk,2);
        C_FIXDIV(fpnk,2);

        C_ADD( f1k, fpk , fpnk );
        C_SUB( f2k, fpk , fpnk );
        C_MUL( tw , f2k , st->super_twiddles[k]);

        freqdata[k].r = HALF_OF(f1k.r + tw.r);
        freqdata[k].i = HALF_OF(f1k.i + tw.i);
        freqdata[ncfft-k].r = HALF_OF(f1k.r - tw.r);
        freqdata[ncfft-k].i = HALF_OF(tw.i - f1k.i);
    }
}

void kiss_fftri(kiss_fftr_cfg st,const kiss_fft_cpx *freqdata, kiss_fft_scalar *timedata)
{
    /* input buffer timedata is stored row-wise */
    int k, ncfft;

    if (st->substate->inverse == 0) {
        speex_fatal("kiss fft usage error: improper alloc\n");
    }

    ncfft = st->substate->nfft;

    st->tmpbuf[0].r = freqdata[0].r + freqdata[ncfft].r;
    st->tmpbuf[0].i = freqdata[0].r - freqdata[ncfft].r;
    /*C_FIXDIV(st->tmpbuf[0],2);*/

    for (k = 1; k <= ncfft / 2; ++k) {
        kiss_fft_cpx fk, fnkc, fek, fok, tmp;
        fk = freqdata[k];
        fnkc.r = freqdata[ncfft - k].r;
        fnkc.i = -freqdata[ncfft - k].i;
        /*C_FIXDIV( fk , 2 );
        C_FIXDIV( fnkc , 2 );*/

        C_ADD (fek, fk, fnkc);
        C_SUB (tmp, fk, fnkc);
        C_MUL (fok, tmp, st->super_twiddles[k]);
        C_ADD (st->tmpbuf[k],     fek, fok);
        C_SUB (st->tmpbuf[ncfft - k], fek, fok);
#ifdef USE_SIMD        
        st->tmpbuf[ncfft - k].i *= _mm_set1_ps(-1.0);
#else
        st->tmpbuf[ncfft - k].i *= -1;
#endif
    }
    kiss_fft (st->substate, st->tmpbuf, (kiss_fft_cpx *) timedata);
}

void kiss_fftr2(kiss_fftr_cfg st,const kiss_fft_scalar *timedata,kiss_fft_scalar *freqdata)
{
   /* input buffer timedata is stored row-wise */
   int k,ncfft;
   kiss_fft_cpx f2k,tdc;
   spx_word32_t f1kr, f1ki, twr, twi;

   if ( st->substate->inverse) {
      speex_fatal("kiss fft usage error: improper alloc\n");
   }

   ncfft = st->substate->nfft;

   /*perform the parallel fft of two real signals packed in real,imag*/
   kiss_fft( st->substate , (const kiss_fft_cpx*)timedata, st->tmpbuf );
    /* The real part of the DC element of the frequency spectrum in st->tmpbuf
   * contains the sum of the even-numbered elements of the input time sequence
   * The imag part is the sum of the odd-numbered elements
   *
   * The sum of tdc.r and tdc.i is the sum of the input time sequence. 
   *      yielding DC of input time sequence
   * The difference of tdc.r - tdc.i is the sum of the input (dot product) [1,-1,1,-1... 
   *      yielding Nyquist bin of input time sequence
    */
 
   tdc.r = st->tmpbuf[0].r;
   tdc.i = st->tmpbuf[0].i;
   C_FIXDIV(tdc,2);
   CHECK_OVERFLOW_OP(tdc.r ,+, tdc.i);
   CHECK_OVERFLOW_OP(tdc.r ,-, tdc.i);
   freqdata[0] = tdc.r + tdc.i;
   freqdata[2*ncfft-1] = tdc.r - tdc.i;

   for ( k=1;k <= ncfft/2 ; ++k )
   {
      /*fpk    = st->tmpbuf[k]; 
      fpnk.r =   st->tmpbuf[ncfft-k].r;
      fpnk.i = - st->tmpbuf[ncfft-k].i;
      C_FIXDIV(fpk,2);
      C_FIXDIV(fpnk,2);

      C_ADD( f1k, fpk , fpnk );
      C_SUB( f2k, fpk , fpnk );
      
      C_MUL( tw , f2k , st->super_twiddles[k]);

      freqdata[2*k-1] = HALF_OF(f1k.r + tw.r);
      freqdata[2*k] = HALF_OF(f1k.i + tw.i);
      freqdata[2*(ncfft-k)-1] = HALF_OF(f1k.r - tw.r);
      freqdata[2*(ncfft-k)] = HALF_OF(tw.i - f1k.i);
      */

      /*f1k.r = PSHR32(ADD32(EXTEND32(st->tmpbuf[k].r), EXTEND32(st->tmpbuf[ncfft-k].r)),1);
      f1k.i = PSHR32(SUB32(EXTEND32(st->tmpbuf[k].i), EXTEND32(st->tmpbuf[ncfft-k].i)),1);
      f2k.r = PSHR32(SUB32(EXTEND32(st->tmpbuf[k].r), EXTEND32(st->tmpbuf[ncfft-k].r)),1);
      f2k.i = SHR32(ADD32(EXTEND32(st->tmpbuf[k].i), EXTEND32(st->tmpbuf[ncfft-k].i)),1);
      
      C_MUL( tw , f2k , st->super_twiddles[k]);

      freqdata[2*k-1] = HALF_OF(f1k.r + tw.r);
      freqdata[2*k] = HALF_OF(f1k.i + tw.i);
      freqdata[2*(ncfft-k)-1] = HALF_OF(f1k.r - tw.r);
      freqdata[2*(ncfft-k)] = HALF_OF(tw.i - f1k.i);
   */
      f2k.r = SHR32(SUB32(EXTEND32(st->tmpbuf[k].r), EXTEND32(st->tmpbuf[ncfft-k].r)),1);
      f2k.i = PSHR32(ADD32(EXTEND32(st->tmpbuf[k].i), EXTEND32(st->tmpbuf[ncfft-k].i)),1);
      
      f1kr = SHL32(ADD32(EXTEND32(st->tmpbuf[k].r), EXTEND32(st->tmpbuf[ncfft-k].r)),13);
      f1ki = SHL32(SUB32(EXTEND32(st->tmpbuf[k].i), EXTEND32(st->tmpbuf[ncfft-k].i)),13);
      
      twr = SHR32(SUB32(MULT16_16(f2k.r,st->super_twiddles[k].r),MULT16_16(f2k.i,st->super_twiddles[k].i)), 1);
      twi = SHR32(ADD32(MULT16_16(f2k.i,st->super_twiddles[k].r),MULT16_16(f2k.r,st->super_twiddles[k].i)), 1);
      
#ifdef FIXED_POINT
      freqdata[2*k-1] = PSHR32(f1kr + twr, 15);
      freqdata[2*k] = PSHR32(f1ki + twi, 15);
      freqdata[2*(ncfft-k)-1] = PSHR32(f1kr - twr, 15);
      freqdata[2*(ncfft-k)] = PSHR32(twi - f1ki, 15);
#else
      freqdata[2*k-1] = .5f*(f1kr + twr);
      freqdata[2*k] = .5f*(f1ki + twi);
      freqdata[2*(ncfft-k)-1] = .5f*(f1kr - twr);
      freqdata[2*(ncfft-k)] = .5f*(twi - f1ki);
      
#endif
   }
}

void kiss_fftri2(kiss_fftr_cfg st,const kiss_fft_scalar *freqdata,kiss_fft_scalar *timedata)
{
   /* input buffer timedata is stored row-wise */
   int k, ncfft;

   if (st->substate->inverse == 0) {
      speex_fatal ("kiss fft usage error: improper alloc\n");
   }

   ncfft = st->substate->nfft;

   st->tmpbuf[0].r = freqdata[0] + freqdata[2*ncfft-1];
   st->tmpbuf[0].i = freqdata[0] - freqdata[2*ncfft-1];
   /*C_FIXDIV(st->tmpbuf[0],2);*/

   for (k = 1; k <= ncfft / 2; ++k) {
      kiss_fft_cpx fk, fnkc, fek, fok, tmp;
      fk.r = freqdata[2*k-1];
      fk.i = freqdata[2*k];
      fnkc.r = freqdata[2*(ncfft - k)-1];
      fnkc.i = -freqdata[2*(ncfft - k)];
        /*C_FIXDIV( fk , 2 );
      C_FIXDIV( fnkc , 2 );*/

      C_ADD (fek, fk, fnkc);
      C_SUB (tmp, fk, fnkc);
      C_MUL (fok, tmp, st->super_twiddles[k]);
      C_ADD (st->tmpbuf[k],     fek, fok);
      C_SUB (st->tmpbuf[ncfft - k], fek, fok);
#ifdef USE_SIMD        
      st->tmpbuf[ncfft - k].i *= _mm_set1_ps(-1.0);
#else
      st->tmpbuf[ncfft - k].i *= -1;
#endif
   }
   kiss_fft (st->substate, st->tmpbuf, (kiss_fft_cpx *) timedata);
}

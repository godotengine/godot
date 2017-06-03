/* Copyright (c) 2008 Xiph.Org Foundation
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

#define SKIP_CONFIG_H

#ifndef CUSTOM_MODES
#define CUSTOM_MODES
#endif

#include <stdio.h>

#define CELT_C
#define TEST_UNIT_DFT_C
#include "stack_alloc.h"
#include "kiss_fft.h"
#include "kiss_fft.c"
#include "mathops.c"
#include "entcode.c"

#if defined(OPUS_X86_MAY_HAVE_SSE2) || defined(OPUS_X86_MAY_HAVE_SSE4_1)
# include "x86/x86cpu.c"
#elif defined(OPUS_ARM_ASM) || defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
# include "arm/armcpu.c"
# include "celt_lpc.c"
# include "pitch.c"
# if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
#  include "arm/celt_neon_intr.c"
#  if defined(HAVE_ARM_NE10)
#   include "mdct.c"
#   include "arm/celt_ne10_fft.c"
#   include "arm/celt_ne10_mdct.c"
#  endif
# endif
# include "arm/arm_celt_map.c"
#endif

#ifndef M_PI
#define M_PI 3.141592653
#endif

int ret = 0;

void check(kiss_fft_cpx  * in,kiss_fft_cpx  * out,int nfft,int isinverse)
{
    int bin,k;
    double errpow=0,sigpow=0, snr;

    for (bin=0;bin<nfft;++bin) {
        double ansr = 0;
        double ansi = 0;
        double difr;
        double difi;

        for (k=0;k<nfft;++k) {
            double phase = -2*M_PI*bin*k/nfft;
            double re = cos(phase);
            double im = sin(phase);
            if (isinverse)
                im = -im;

            if (!isinverse)
            {
               re /= nfft;
               im /= nfft;
            }

            ansr += in[k].r * re - in[k].i * im;
            ansi += in[k].r * im + in[k].i * re;
        }
        /*printf ("%d %d ", (int)ansr, (int)ansi);*/
        difr = ansr - out[bin].r;
        difi = ansi - out[bin].i;
        errpow += difr*difr + difi*difi;
        sigpow += ansr*ansr+ansi*ansi;
    }
    snr = 10*log10(sigpow/errpow);
    printf("nfft=%d inverse=%d,snr = %f\n",nfft,isinverse,snr );
    if (snr<60) {
       printf( "** poor snr: %f ** \n", snr);
       ret = 1;
    }
}

void test1d(int nfft,int isinverse,int arch)
{
    size_t buflen = sizeof(kiss_fft_cpx)*nfft;

    kiss_fft_cpx  * in = (kiss_fft_cpx*)malloc(buflen);
    kiss_fft_cpx  * out= (kiss_fft_cpx*)malloc(buflen);
    kiss_fft_state *cfg = opus_fft_alloc(nfft,0,0,arch);
    int k;

    for (k=0;k<nfft;++k) {
        in[k].r = (rand() % 32767) - 16384;
        in[k].i = (rand() % 32767) - 16384;
    }

    for (k=0;k<nfft;++k) {
       in[k].r *= 32768;
       in[k].i *= 32768;
    }

    if (isinverse)
    {
       for (k=0;k<nfft;++k) {
          in[k].r /= nfft;
          in[k].i /= nfft;
       }
    }

    /*for (k=0;k<nfft;++k) printf("%d %d ", in[k].r, in[k].i);printf("\n");*/

    if (isinverse)
       opus_ifft(cfg,in,out, arch);
    else
       opus_fft(cfg,in,out, arch);

    /*for (k=0;k<nfft;++k) printf("%d %d ", out[k].r, out[k].i);printf("\n");*/

    check(in,out,nfft,isinverse);

    free(in);
    free(out);
    opus_fft_free(cfg, arch);
}

int main(int argc,char ** argv)
{
    ALLOC_STACK;
    int arch = opus_select_arch();

    if (argc>1) {
        int k;
        for (k=1;k<argc;++k) {
            test1d(atoi(argv[k]),0,arch);
            test1d(atoi(argv[k]),1,arch);
        }
    }else{
        test1d(32,0,arch);
        test1d(32,1,arch);
        test1d(128,0,arch);
        test1d(128,1,arch);
        test1d(256,0,arch);
        test1d(256,1,arch);
#ifndef RADIX_TWO_ONLY
        test1d(36,0,arch);
        test1d(36,1,arch);
        test1d(50,0,arch);
        test1d(50,1,arch);
        test1d(60,0,arch);
        test1d(60,1,arch);
        test1d(120,0,arch);
        test1d(120,1,arch);
        test1d(240,0,arch);
        test1d(240,1,arch);
        test1d(480,0,arch);
        test1d(480,1,arch);
#endif
    }
    return ret;
}

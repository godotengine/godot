/* Copyright (c) 2015 Xiph.Org Foundation
   Written by Viswanath Puttagunta */
/**
   @file fft_arm.h
   @brief ARM Neon Intrinsic optimizations for fft using NE10 library
 */

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


#if !defined(FFT_ARM_H)
#define FFT_ARM_H

#include "config.h"
#include "kiss_fft.h"

#if defined(HAVE_ARM_NE10)

int opus_fft_alloc_arm_neon(kiss_fft_state *st);
void opus_fft_free_arm_neon(kiss_fft_state *st);

void opus_fft_neon(const kiss_fft_state *st,
                   const kiss_fft_cpx *fin,
                   kiss_fft_cpx *fout);

void opus_ifft_neon(const kiss_fft_state *st,
                    const kiss_fft_cpx *fin,
                    kiss_fft_cpx *fout);

#if !defined(OPUS_HAVE_RTCD)
#define OVERRIDE_OPUS_FFT (1)

#define opus_fft_alloc_arch(_st, arch) \
   ((void)(arch), opus_fft_alloc_arm_neon(_st))

#define opus_fft_free_arch(_st, arch) \
   ((void)(arch), opus_fft_free_arm_neon(_st))

#define opus_fft(_st, _fin, _fout, arch) \
   ((void)(arch), opus_fft_neon(_st, _fin, _fout))

#define opus_ifft(_st, _fin, _fout, arch) \
   ((void)(arch), opus_ifft_neon(_st, _fin, _fout))

#endif /* OPUS_HAVE_RTCD */

#endif /* HAVE_ARM_NE10 */

#endif

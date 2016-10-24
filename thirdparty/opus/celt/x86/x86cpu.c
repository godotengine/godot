/* Copyright (c) 2014, Cisco Systems, INC
   Written by XiangMingZhu WeiZhou MinPeng YanWang

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

#include "cpu_support.h"
#include "macros.h"
#include "main.h"
#include "pitch.h"
#include "x86cpu.h"

#if (defined(OPUS_X86_MAY_HAVE_SSE) && !defined(OPUS_X86_PRESUME_SSE)) || \
  (defined(OPUS_X86_MAY_HAVE_SSE2) && !defined(OPUS_X86_PRESUME_SSE2)) || \
  (defined(OPUS_X86_MAY_HAVE_SSE4_1) && !defined(OPUS_X86_PRESUME_SSE4_1)) || \
  (defined(OPUS_X86_MAY_HAVE_AVX) && !defined(OPUS_X86_PRESUME_AVX))


#if defined(_MSC_VER)

#include <intrin.h>
static _inline void cpuid(unsigned int CPUInfo[4], unsigned int InfoType)
{
	__cpuid((int*)CPUInfo, InfoType);
}

#else

#if defined(CPU_INFO_BY_C)
#include <cpuid.h>
#endif

static void cpuid(unsigned int CPUInfo[4], unsigned int InfoType)
{
#if defined(CPU_INFO_BY_ASM)
#if defined(__i386__) && defined(__PIC__)
/* %ebx is PIC register in 32-bit, so mustn't clobber it. */
    __asm__ __volatile__ (
        "xchg %%ebx, %1\n"
        "cpuid\n"
        "xchg %%ebx, %1\n":
        "=a" (CPUInfo[0]),
        "=r" (CPUInfo[1]),
        "=c" (CPUInfo[2]),
        "=d" (CPUInfo[3]) :
        "0" (InfoType)
    );
#else
    __asm__ __volatile__ (
        "cpuid":
        "=a" (CPUInfo[0]),
        "=b" (CPUInfo[1]),
        "=c" (CPUInfo[2]),
        "=d" (CPUInfo[3]) :
        "0" (InfoType)
    );
#endif
#elif defined(CPU_INFO_BY_C)
    __get_cpuid(InfoType, &(CPUInfo[0]), &(CPUInfo[1]), &(CPUInfo[2]), &(CPUInfo[3]));
#endif
}

#endif

typedef struct CPU_Feature{
    /*  SIMD: 128-bit */
    int HW_SSE;
    int HW_SSE2;
    int HW_SSE41;
    /*  SIMD: 256-bit */
    int HW_AVX;
} CPU_Feature;

static void opus_cpu_feature_check(CPU_Feature *cpu_feature)
{
    unsigned int info[4] = {0};
    unsigned int nIds = 0;

    cpuid(info, 0);
    nIds = info[0];

    if (nIds >= 1){
        cpuid(info, 1);
        cpu_feature->HW_SSE = (info[3] & (1 << 25)) != 0;
        cpu_feature->HW_SSE2 = (info[3] & (1 << 26)) != 0;
        cpu_feature->HW_SSE41 = (info[2] & (1 << 19)) != 0;
        cpu_feature->HW_AVX = (info[2] & (1 << 28)) != 0;
    }
    else {
        cpu_feature->HW_SSE = 0;
        cpu_feature->HW_SSE2 = 0;
        cpu_feature->HW_SSE41 = 0;
        cpu_feature->HW_AVX = 0;
    }
}

int opus_select_arch(void)
{
    CPU_Feature cpu_feature;
    int arch;

    opus_cpu_feature_check(&cpu_feature);

    arch = 0;
    if (!cpu_feature.HW_SSE)
    {
       return arch;
    }
    arch++;

    if (!cpu_feature.HW_SSE2)
    {
       return arch;
    }
    arch++;

    if (!cpu_feature.HW_SSE41)
    {
        return arch;
    }
    arch++;

    if (!cpu_feature.HW_AVX)
    {
        return arch;
    }
    arch++;

    return arch;
}

#endif

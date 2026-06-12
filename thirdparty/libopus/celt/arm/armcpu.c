/* Copyright (c) 2010 Xiph.Org Foundation
 * Copyright (c) 2013 Parrot */
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

/* Original code from libtheora modified to suit to Opus */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef OPUS_HAVE_RTCD

#include "armcpu.h"
#include "cpu_support.h"
#include "os_support.h"
#include "opus_types.h"
#include "arch.h"

#define OPUS_CPU_ARM_V4_FLAG    (1<<OPUS_ARCH_ARM_V4)
#define OPUS_CPU_ARM_EDSP_FLAG  (1<<OPUS_ARCH_ARM_EDSP)
#define OPUS_CPU_ARM_MEDIA_FLAG (1<<OPUS_ARCH_ARM_MEDIA)
#define OPUS_CPU_ARM_NEON_FLAG  (1<<OPUS_ARCH_ARM_NEON)
#define OPUS_CPU_ARM_DOTPROD_FLAG  (1<<OPUS_ARCH_ARM_DOTPROD)

#if defined(_MSC_VER)
/*For GetExceptionCode() and EXCEPTION_ILLEGAL_INSTRUCTION.*/
# define WIN32_LEAN_AND_MEAN
# define WIN32_EXTRA_LEAN
# include <windows.h>

static OPUS_INLINE opus_uint32 opus_cpu_capabilities(void){
  opus_uint32 flags;
  flags=0;
  /* MSVC has no OPUS_INLINE __asm support for ARM, but it does let you __emit
   * instructions via their assembled hex code.
   * All of these instructions should be essentially nops. */
# if defined(OPUS_ARM_MAY_HAVE_EDSP) || defined(OPUS_ARM_MAY_HAVE_MEDIA) \
 || defined(OPUS_ARM_MAY_HAVE_NEON) || defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
  __try{
    /*PLD [r13]*/
    __emit(0xF5DDF000);
    flags|=OPUS_CPU_ARM_EDSP_FLAG;
  }
  __except(GetExceptionCode()==EXCEPTION_ILLEGAL_INSTRUCTION){
    /*Ignore exception.*/
  }
#  if defined(OPUS_ARM_MAY_HAVE_MEDIA) \
 || defined(OPUS_ARM_MAY_HAVE_NEON) || defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
  __try{
    /*SHADD8 r3,r3,r3*/
    __emit(0xE6333F93);
    flags|=OPUS_CPU_ARM_MEDIA_FLAG;
  }
  __except(GetExceptionCode()==EXCEPTION_ILLEGAL_INSTRUCTION){
    /*Ignore exception.*/
  }
#   if defined(OPUS_ARM_MAY_HAVE_NEON) || defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
  __try{
    /*VORR q0,q0,q0*/
    __emit(0xF2200150);
    flags|=OPUS_CPU_ARM_NEON_FLAG;
  }
  __except(GetExceptionCode()==EXCEPTION_ILLEGAL_INSTRUCTION){
    /*Ignore exception.*/
  }
#   endif
#  endif
# endif
  return flags;
}

#elif defined(__linux__)
/* Linux based */
#include <stdio.h>

static opus_uint32 opus_cpu_capabilities(void)
{
  opus_uint32 flags = 0;
  FILE *cpuinfo;

  /* Reading /proc/self/auxv would be easier, but that doesn't work reliably on
   * Android */
  cpuinfo = fopen("/proc/cpuinfo", "r");

  if(cpuinfo != NULL)
  {
    /* 512 should be enough for anybody (it's even enough for all the flags that
     * x86 has accumulated... so far). */
    char buf[512];

    while(fgets(buf, 512, cpuinfo) != NULL)
    {
# if defined(OPUS_ARM_MAY_HAVE_EDSP) || defined(OPUS_ARM_MAY_HAVE_MEDIA) \
 || defined(OPUS_ARM_MAY_HAVE_NEON) || defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
      /* Search for edsp and neon flag */
      if(memcmp(buf, "Features", 8) == 0)
      {
        char *p;
        p = strstr(buf, " edsp");
        if(p != NULL && (p[5] == ' ' || p[5] == '\n'))
          flags |= OPUS_CPU_ARM_EDSP_FLAG;

#  if defined(OPUS_ARM_MAY_HAVE_NEON) || defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
        p = strstr(buf, " neon");
        if(p != NULL && (p[5] == ' ' || p[5] == '\n'))
          flags |= OPUS_CPU_ARM_NEON_FLAG;
        p = strstr(buf, " asimd");
        if(p != NULL && (p[6] == ' ' || p[6] == '\n'))
          flags |= OPUS_CPU_ARM_NEON_FLAG | OPUS_CPU_ARM_MEDIA_FLAG | OPUS_CPU_ARM_EDSP_FLAG;
#  endif
#  if defined(OPUS_ARM_MAY_HAVE_DOTPROD)
        p = strstr(buf, " asimddp");
        if(p != NULL && (p[8] == ' ' || p[8] == '\n'))
          flags |= OPUS_CPU_ARM_DOTPROD_FLAG;
#  endif
      }
# endif

# if defined(OPUS_ARM_MAY_HAVE_MEDIA) \
 || defined(OPUS_ARM_MAY_HAVE_NEON) || defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
      /* Search for media capabilities (>= ARMv6) */
      if(memcmp(buf, "CPU architecture:", 17) == 0)
      {
        int version;
        version = atoi(buf+17);

        if(version >= 6)
          flags |= OPUS_CPU_ARM_MEDIA_FLAG;
      }
# endif
    }

#if defined(OPUS_ARM_PRESUME_AARCH64_NEON_INTR)
    flags |= OPUS_CPU_ARM_EDSP_FLAG | OPUS_CPU_ARM_MEDIA_FLAG | OPUS_CPU_ARM_NEON_FLAG;
# if defined(OPUS_ARM_PRESUME_DOTPROD)
    flags |= OPUS_CPU_ARM_DOTPROD_FLAG;
# endif
#endif

    fclose(cpuinfo);
  }
  return flags;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>

static opus_uint32 opus_cpu_capabilities(void)
{
  opus_uint32 flags = 0;

#if defined(OPUS_ARM_MAY_HAVE_DOTPROD)
  size_t size = sizeof(uint32_t);
  uint32_t value = 0;
  if (!sysctlbyname("hw.optional.arm.FEAT_DotProd", &value, &size, NULL, 0) && value)
  {
    flags |= OPUS_CPU_ARM_DOTPROD_FLAG;
  }
#endif

#if defined(OPUS_ARM_PRESUME_AARCH64_NEON_INTR)
  flags |= OPUS_CPU_ARM_EDSP_FLAG | OPUS_CPU_ARM_MEDIA_FLAG | OPUS_CPU_ARM_NEON_FLAG;
# if defined(OPUS_ARM_PRESUME_DOTPROD)
  flags |= OPUS_CPU_ARM_DOTPROD_FLAG;
# endif
#endif
  return flags;
}

#elif defined(HAVE_ELF_AUX_INFO)
#include <sys/auxv.h>

static opus_uint32 opus_cpu_capabilities(void)
{
  long hwcap = 0;
  opus_uint32 flags = 0;

# if defined(OPUS_ARM_MAY_HAVE_MEDIA) \
 || defined(OPUS_ARM_MAY_HAVE_NEON) || defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
  /* FreeBSD requires armv6+, which always supports media instructions */
  flags |= OPUS_CPU_ARM_MEDIA_FLAG;
# endif

  elf_aux_info(AT_HWCAP, &hwcap, sizeof hwcap);

# if defined(OPUS_ARM_MAY_HAVE_EDSP) || defined(OPUS_ARM_MAY_HAVE_MEDIA) \
 || defined(OPUS_ARM_MAY_HAVE_NEON) || defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
#  ifdef HWCAP_EDSP
  if (hwcap & HWCAP_EDSP)
    flags |= OPUS_CPU_ARM_EDSP_FLAG;
#  endif

#  if defined(OPUS_ARM_MAY_HAVE_NEON) || defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
#   ifdef HWCAP_NEON
  if (hwcap & HWCAP_NEON)
    flags |= OPUS_CPU_ARM_NEON_FLAG;
#   elif defined(HWCAP_ASIMD)
  if (hwcap & HWCAP_ASIMD)
    flags |= OPUS_CPU_ARM_NEON_FLAG | OPUS_CPU_ARM_MEDIA_FLAG | OPUS_CPU_ARM_EDSP_FLAG;
#   endif
#  endif
#  if defined(OPUS_ARM_MAY_HAVE_DOTPROD) && defined(HWCAP_ASIMDDP)
  if (hwcap & HWCAP_ASIMDDP)
    flags |= OPUS_CPU_ARM_DOTPROD_FLAG;
#  endif
# endif

#if defined(OPUS_ARM_PRESUME_AARCH64_NEON_INTR)
    flags |= OPUS_CPU_ARM_EDSP_FLAG | OPUS_CPU_ARM_MEDIA_FLAG | OPUS_CPU_ARM_NEON_FLAG;
# if defined(OPUS_ARM_PRESUME_DOTPROD)
    flags |= OPUS_CPU_ARM_DOTPROD_FLAG;
# endif
#endif

  return (flags);
}

#elif defined(__OpenBSD__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <machine/armreg.h>
#include <machine/cpu.h>

static opus_uint32 opus_cpu_capabilities(void)
{
  opus_uint32 flags = 0;

#if defined(OPUS_ARM_MAY_HAVE_DOTPROD) && defined(CPU_ID_AA64ISAR0)
  const int isar0_mib[] = { CTL_MACHDEP, CPU_ID_AA64ISAR0 };
  uint64_t isar0;
  size_t len = sizeof(isar0);

  if (sysctl(isar0_mib, 2, &isar0, &len, NULL, 0) != -1)
  {
    if (ID_AA64ISAR0_DP(isar0) >= ID_AA64ISAR0_DP_IMPL)
      flags |= OPUS_CPU_ARM_DOTPROD_FLAG;
  }
#endif

#if defined(OPUS_ARM_PRESUME_NEON_INTR) \
 || defined(OPUS_ARM_PRESUME_AARCH64_NEON_INTR)
  flags |= OPUS_CPU_ARM_EDSP_FLAG | OPUS_CPU_ARM_MEDIA_FLAG | OPUS_CPU_ARM_NEON_FLAG;
# if defined(OPUS_ARM_PRESUME_DOTPROD)
  flags |= OPUS_CPU_ARM_DOTPROD_FLAG;
# endif
#endif
  return flags;
}

#else
/* The feature registers which can tell us what the processor supports are
 * accessible in privileged modes only, so we can't have a general user-space
 * detection method like on x86.*/
# error "Configured to use ARM asm but no CPU detection method available for " \
   "your platform.  Reconfigure with --disable-rtcd (or send patches)."
#endif

static int opus_select_arch_impl(void)
{
  opus_uint32 flags = opus_cpu_capabilities();
  int arch = 0;

  if(!(flags & OPUS_CPU_ARM_EDSP_FLAG)) {
    /* Asserts ensure arch values are sequential */
    celt_assert(arch == OPUS_ARCH_ARM_V4);
    return arch;
  }
  arch++;

  if(!(flags & OPUS_CPU_ARM_MEDIA_FLAG)) {
    celt_assert(arch == OPUS_ARCH_ARM_EDSP);
    return arch;
  }
  arch++;

  if(!(flags & OPUS_CPU_ARM_NEON_FLAG)) {
    celt_assert(arch == OPUS_ARCH_ARM_MEDIA);
    return arch;
  }
  arch++;

  if(!(flags & OPUS_CPU_ARM_DOTPROD_FLAG)) {
    celt_assert(arch == OPUS_ARCH_ARM_NEON);
    return arch;
  }
  arch++;

  celt_assert(arch == OPUS_ARCH_ARM_DOTPROD);
  return arch;
}

int opus_select_arch(void) {
  int arch = opus_select_arch_impl();
#ifdef FUZZING
  arch = rand()%(arch+1);
#endif
  return arch;
}
#endif

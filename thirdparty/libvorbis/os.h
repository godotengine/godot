#ifndef _OS_H
#define _OS_H
/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2015             *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

 function: #ifdef jail to whip a few platforms into the UNIX ideal.

 ********************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include <ogg/os_types.h>

#include "misc.h"

#ifndef _V_IFDEFJAIL_H_
#  define _V_IFDEFJAIL_H_

#  ifdef __GNUC__
#    define STIN static __inline__
#  elif defined(_WIN32)
#    define STIN static __inline
#  else
#    define STIN static
#  endif

#ifdef DJGPP
#  define rint(x)   (floor((x)+0.5f))
#endif

#ifndef M_PI
#  define M_PI (3.1415926536f)
#endif

#if defined(_WIN32) && !defined(__SYMBIAN32__)
#  include <malloc.h>
#  define rint(x)   (floor((x)+0.5f))
#  define NO_FLOAT_MATH_LIB
#  define FAST_HYPOT(a, b) sqrt((a)*(a) + (b)*(b))
#endif

#if defined(__SYMBIAN32__) && defined(__WINS__)
void *_alloca(size_t size);
#  define alloca _alloca
#endif

#ifndef FAST_HYPOT
#  define FAST_HYPOT hypot
#endif

#endif

#ifdef HAVE_ALLOCA_H
#  include <alloca.h>
#endif

#ifdef USE_MEMORY_H
#  include <memory.h>
#endif

#ifndef min
#  define min(x,y)  ((x)>(y)?(y):(x))
#endif

#ifndef max
#  define max(x,y)  ((x)<(y)?(y):(x))
#endif


/* Special i386 GCC implementation */
#if defined(__i386__) && defined(__GNUC__) && !defined(__BEOS__)
#  define VORBIS_FPU_CONTROL
/* both GCC and MSVC are kinda stupid about rounding/casting to int.
   Because of encapsulation constraints (GCC can't see inside the asm
   block and so we end up doing stupid things like a store/load that
   is collectively a noop), we do it this way */

/* we must set up the fpu before this works!! */

typedef ogg_int16_t vorbis_fpu_control;

static inline void vorbis_fpu_setround(vorbis_fpu_control *fpu){
  ogg_int16_t ret;
  ogg_int16_t temp;
  __asm__ __volatile__("fnstcw %0\n\t"
          "movw %0,%%dx\n\t"
          "andw $62463,%%dx\n\t"
          "movw %%dx,%1\n\t"
          "fldcw %1\n\t":"=m"(ret):"m"(temp): "dx");
  *fpu=ret;
}

static inline void vorbis_fpu_restore(vorbis_fpu_control fpu){
  __asm__ __volatile__("fldcw %0":: "m"(fpu));
}

/* assumes the FPU is in round mode! */
static inline int vorbis_ftoi(double f){  /* yes, double!  Otherwise,
                                             we get extra fst/fld to
                                             truncate precision */
  int i;
  __asm__("fistl %0": "=m"(i) : "t"(f));
  return(i);
}
#endif /* Special i386 GCC implementation */


/* MSVC inline assembly. 32 bit only; inline ASM isn't implemented in the
 * 64 bit compiler and doesn't work on arm. */
#if defined(_MSC_VER) && !defined(_WIN64) && \
      !defined(_WIN32_WCE) && !defined(_M_ARM)
#  define VORBIS_FPU_CONTROL

typedef ogg_int16_t vorbis_fpu_control;

static __inline int vorbis_ftoi(double f){
        int i;
        __asm{
                fld f
                fistp i
        }
        return i;
}

static __inline void vorbis_fpu_setround(vorbis_fpu_control *fpu){
  (void)fpu;
}

static __inline void vorbis_fpu_restore(vorbis_fpu_control fpu){
  (void)fpu;
}

#endif /* Special MSVC 32 bit implementation */


/* Optimized code path for x86_64 builds. Uses SSE2 intrinsics. This can be
   done safely because all x86_64 CPUs supports SSE2. */
#if (defined(_MSC_VER) && defined(_WIN64)) || (defined(__GNUC__) && defined (__x86_64__))
#  define VORBIS_FPU_CONTROL

typedef ogg_int16_t vorbis_fpu_control;

#include <emmintrin.h>
static __inline int vorbis_ftoi(double f){
        return _mm_cvtsd_si32(_mm_load_sd(&f));
}

static __inline void vorbis_fpu_setround(vorbis_fpu_control *fpu){
  (void)fpu;
}

static __inline void vorbis_fpu_restore(vorbis_fpu_control fpu){
  (void)fpu;
}

#endif /* Special MSVC x64 implementation */


/* If no special implementation was found for the current compiler / platform,
   use the default implementation here: */
#ifndef VORBIS_FPU_CONTROL

typedef int vorbis_fpu_control;

static int vorbis_ftoi(double f){
        /* Note: MSVC and GCC (at least on some systems) round towards zero, thus,
           the floor() call is required to ensure correct roudning of
           negative numbers */
        return (int)floor(f+.5);
}

/* We don't have special code for this compiler/arch, so do it the slow way */
#  define vorbis_fpu_setround(vorbis_fpu_control) {}
#  define vorbis_fpu_restore(vorbis_fpu_control) {}

#endif /* default implementation */

#endif /* _OS_H */

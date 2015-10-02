/* Copyright (C) 2001 Erik de Castro Lopo <erikd AT mega-nerd DOT com> */
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

/* Version 1.1 */

#ifndef FLOAT_CAST_H
#define FLOAT_CAST_H


#include "arch.h"

/*============================================================================
**      On Intel Pentium processors (especially PIII and probably P4), converting
**      from float to int is very slow. To meet the C specs, the code produced by
**      most C compilers targeting Pentium needs to change the FPU rounding mode
**      before the float to int conversion is performed.
**
**      Changing the FPU rounding mode causes the FPU pipeline to be flushed. It
**      is this flushing of the pipeline which is so slow.
**
**      Fortunately the ISO C99 specifications define the functions lrint, lrintf,
**      llrint and llrintf which fix this problem as a side effect.
**
**      On Unix-like systems, the configure process should have detected the
**      presence of these functions. If they weren't found we have to replace them
**      here with a standard C cast.
*/

/*
**      The C99 prototypes for lrint and lrintf are as follows:
**
**              long int lrintf (float x) ;
**              long int lrint  (double x) ;
*/

/*      The presence of the required functions are detected during the configure
**      process and the values HAVE_LRINT and HAVE_LRINTF are set accordingly in
**      the config.h file.
*/

#if (HAVE_LRINTF)

/*      These defines enable functionality introduced with the 1999 ISO C
**      standard. They must be defined before the inclusion of math.h to
**      engage them. If optimisation is enabled, these functions will be
**      inlined. With optimisation switched off, you have to link in the
**      maths library using -lm.
*/

#define _ISOC9X_SOURCE 1
#define _ISOC99_SOURCE 1

#define __USE_ISOC9X 1
#define __USE_ISOC99 1

#include <math.h>
#define float2int(x) lrintf(x)

#elif (defined(HAVE_LRINT))

#define _ISOC9X_SOURCE 1
#define _ISOC99_SOURCE 1

#define __USE_ISOC9X 1
#define __USE_ISOC99 1

#include <math.h>
#define float2int(x) lrint(x)

#elif (defined(_MSC_VER) && _MSC_VER >= 1400) && (defined (WIN64) || defined (_WIN64))
        #include <xmmintrin.h>

        __inline long int float2int(float value)
        {
                return _mm_cvtss_si32(_mm_load_ss(&value));
        }
#elif (defined(_MSC_VER) && _MSC_VER >= 1400) && (defined (WIN32) || defined (_WIN32))
        #include <math.h>

        /*      Win32 doesn't seem to have these functions.
        **      Therefore implement OPUS_INLINE versions of these functions here.
        */

        __inline long int
        float2int (float flt)
        {       int intgr;

                _asm
                {       fld flt
                        fistp intgr
                } ;

                return intgr ;
        }

#else

#if (defined(__GNUC__) && defined(__STDC__) && __STDC__ && __STDC_VERSION__ >= 199901L)
        /* supported by gcc in C99 mode, but not by all other compilers */
        #warning "Don't have the functions lrint() and lrintf ()."
        #warning "Replacing these functions with a standard C cast."
#endif /* __STDC_VERSION__ >= 199901L */
        #include <math.h>
        #define float2int(flt) ((int)(floor(.5+flt)))
#endif

#ifndef DISABLE_FLOAT_API
static OPUS_INLINE opus_int16 FLOAT2INT16(float x)
{
   x = x*CELT_SIG_SCALE;
   x = MAX32(x, -32768);
   x = MIN32(x, 32767);
   return (opus_int16)float2int(x);
}
#endif /* DISABLE_FLOAT_API */

#endif /* FLOAT_CAST_H */

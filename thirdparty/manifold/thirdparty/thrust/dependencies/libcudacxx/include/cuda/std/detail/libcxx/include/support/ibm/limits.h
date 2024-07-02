// -*- C++ -*-
//===--------------------- support/ibm/limits.h ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_SUPPORT_IBM_LIMITS_H
#define _LIBCUDACXX_SUPPORT_IBM_LIMITS_H

#if !defined(_AIX) // Linux
#include <math.h> // for HUGE_VAL, HUGE_VALF, HUGE_VALL, and NAN

static const unsigned int _QNAN_F = 0x7fc00000;
#define NANF (*((float *)(&_QNAN_F)))
static const unsigned int _QNAN_LDBL128[4] = {0x7ff80000, 0x0, 0x0, 0x0};
#define NANL (*((long double *)(&_QNAN_LDBL128)))
static const unsigned int _SNAN_F= 0x7f855555;
#define NANSF (*((float *)(&_SNAN_F)))
static const unsigned int _SNAN_D[2] = {0x7ff55555, 0x55555555};
#define NANS (*((double *)(&_SNAN_D)))
static const unsigned int _SNAN_LDBL128[4] = {0x7ff55555, 0x55555555, 0x0, 0x0};
#define NANSL (*((long double *)(&_SNAN_LDBL128)))

#define __builtin_huge_val()     HUGE_VAL
#define __builtin_huge_valf()    HUGE_VALF
#define __builtin_huge_vall()    HUGE_VALL
#define __builtin_nan(__dummy)   NAN
#define __builtin_nanf(__dummy)  NANF
#define __builtin_nanl(__dummy)  NANL
#define __builtin_nans(__dummy)  NANS
#define __builtin_nansf(__dummy) NANSF
#define __builtin_nansl(__dummy) NANSL

#else

#include <math.h>
#include <float.h> // limit constants

#define __builtin_huge_val()     HUGE_VAL  //0x7ff0000000000000
#define __builtin_huge_valf()    HUGE_VALF //0x7f800000
#define __builtin_huge_vall()    HUGE_VALL //0x7ff0000000000000
#define __builtin_nan(__dummy)   nan(__dummy) //0x7ff8000000000000
#define __builtin_nanf(__dummy)  nanf(__dummy) // 0x7ff80000
#define __builtin_nanl(__dummy)  nanl(__dummy) //0x7ff8000000000000
#define __builtin_nans(__dummy)  DBL_SNAN //0x7ff5555555555555
#define __builtin_nansf(__dummy) FLT_SNAN //0x7f855555
#define __builtin_nansl(__dummy) DBL_SNAN //0x7ff5555555555555

#define __FLT_MANT_DIG__   FLT_MANT_DIG
#define __FLT_DIG__        FLT_DIG
#define __FLT_RADIX__      FLT_RADIX
#define __FLT_MIN_EXP__    FLT_MIN_EXP
#define __FLT_MIN_10_EXP__ FLT_MIN_10_EXP
#define __FLT_MAX_EXP__    FLT_MAX_EXP
#define __FLT_MAX_10_EXP__ FLT_MAX_10_EXP
#define __FLT_MIN__        FLT_MIN
#define __FLT_MAX__        FLT_MAX
#define __FLT_EPSILON__    FLT_EPSILON
// predefined by XLC on LoP
#define __FLT_DENORM_MIN__ 1.40129846e-45F

#define __DBL_MANT_DIG__   DBL_MANT_DIG
#define __DBL_DIG__        DBL_DIG
#define __DBL_MIN_EXP__    DBL_MIN_EXP
#define __DBL_MIN_10_EXP__ DBL_MIN_10_EXP
#define __DBL_MAX_EXP__    DBL_MAX_EXP
#define __DBL_MAX_10_EXP__ DBL_MAX_10_EXP
#define __DBL_MIN__        DBL_MIN
#define __DBL_MAX__        DBL_MAX
#define __DBL_EPSILON__    DBL_EPSILON
// predefined by XLC on LoP
#define __DBL_DENORM_MIN__ 4.9406564584124654e-324

#define __LDBL_MANT_DIG__   LDBL_MANT_DIG
#define __LDBL_DIG__        LDBL_DIG
#define __LDBL_MIN_EXP__    LDBL_MIN_EXP
#define __LDBL_MIN_10_EXP__ LDBL_MIN_10_EXP
#define __LDBL_MAX_EXP__    LDBL_MAX_EXP
#define __LDBL_MAX_10_EXP__ LDBL_MAX_10_EXP
#define __LDBL_MIN__        LDBL_MIN
#define __LDBL_MAX__        LDBL_MAX
#define __LDBL_EPSILON__    LDBL_EPSILON
// predefined by XLC on LoP
#if __LONGDOUBLE128
#define __LDBL_DENORM_MIN__ 4.94065645841246544176568792868221e-324L
#else
#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
#endif

// predefined by XLC on LoP
#define __CHAR_BIT__    8

#endif // _AIX

#endif // _LIBCUDACXX_SUPPORT_IBM_LIMITS_H

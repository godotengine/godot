/*
  Copyright (c) 2005-2009, The Musepack Development Team
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.

  * Neither the name of the The Musepack Development Team nor the
  names of its contributors may be used to endorse or promote
  products derived from this software without specific prior
  written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/// \file math.h
/// Libmpcdec internal math routines.
#ifndef _MPCDEC_MATH_H_
#define _MPCDEC_MATH_H_
#ifdef WIN32
#pragma once
#endif

#include <mpc/mpc_types.h>

#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MPC_FIXED_POINT

#ifdef _WIN32_WCE
#include <cmnintrin.h>
#define MPC_HAVE_MULHIGH
#endif

//in fixedpoint mode, results in decode output buffer are in -MPC_FIXED_POINT_SCALE ... MPC_FIXED_POINT_SCALE range

typedef mpc_int64_t MPC_SAMPLE_FORMAT_MULTIPLY;

#define MAKE_MPC_SAMPLE(X) (MPC_SAMPLE_FORMAT)((double)(X) * (double)(((mpc_int64_t)1)<<MPC_FIXED_POINT_FRACTPART))
#define MAKE_MPC_SAMPLE_EX(X,Y) (MPC_SAMPLE_FORMAT)((double)(X) * (double)(((mpc_int64_t)1)<<(Y)))

#define MPC_MULTIPLY_NOTRUNCATE(X,Y) \
    (((MPC_SAMPLE_FORMAT_MULTIPLY)(X) * (MPC_SAMPLE_FORMAT_MULTIPLY)(Y)) >> MPC_FIXED_POINT_FRACTPART)

#define MPC_MULTIPLY_EX_NOTRUNCATE(X,Y,Z) \
    (((MPC_SAMPLE_FORMAT_MULTIPLY)(X) * (MPC_SAMPLE_FORMAT_MULTIPLY)(Y)) >> (Z))

#ifdef _DEBUG
static mpc_inline MPC_SAMPLE_FORMAT MPC_MULTIPLY(MPC_SAMPLE_FORMAT item1,MPC_SAMPLE_FORMAT item2)
{
    MPC_SAMPLE_FORMAT_MULTIPLY temp = MPC_MULTIPLY_NOTRUNCATE(item1,item2);
    assert(temp == (MPC_SAMPLE_FORMAT_MULTIPLY)(MPC_SAMPLE_FORMAT)temp);
    return (MPC_SAMPLE_FORMAT)temp;
}

static mpc_inline MPC_SAMPLE_FORMAT MPC_MULTIPLY_EX(MPC_SAMPLE_FORMAT item1,MPC_SAMPLE_FORMAT item2,unsigned shift)
{
    MPC_SAMPLE_FORMAT_MULTIPLY temp = MPC_MULTIPLY_EX_NOTRUNCATE(item1,item2,shift);
    assert(temp == (MPC_SAMPLE_FORMAT_MULTIPLY)(MPC_SAMPLE_FORMAT)temp);
    return (MPC_SAMPLE_FORMAT)temp;
}

#else

#define MPC_MULTIPLY(X,Y) ((MPC_SAMPLE_FORMAT)MPC_MULTIPLY_NOTRUNCATE(X,Y))
#define MPC_MULTIPLY_EX(X,Y,Z) ((MPC_SAMPLE_FORMAT)MPC_MULTIPLY_EX_NOTRUNCATE(X,Y,Z))

#endif

#ifdef MPC_HAVE_MULHIGH
#define MPC_MULTIPLY_FRACT(X,Y) _MulHigh(X,Y)
#else
//#define MPC_MULTIPLY_FRACT(X,Y) MPC_MULTIPLY_EX(X,Y,32)
//#define MPC_MULTIPLY_FRACT(X,Y) ((((MPC_SAMPLE_FORMAT_MULTIPLY)(X)*(MPC_SAMPLE_FORMAT_MULTIPLY)(Y))>>16)>>16)
//#if 1
#ifdef PSP_ENABLED
#define MPC_MULTIPLY_FRACT(X,Y) (((MPC_SAMPLE_FORMAT_MULTIPLY)(X)*(MPC_SAMPLE_FORMAT_MULTIPLY)(Y))/4294967296L)
#else
#define MPC_MULTIPLY_FRACT(X,Y) MPC_MULTIPLY_EX(X,Y,32)
#endif

#endif

//#define MPC_MAKE_FRACT_CONST(X) (MPC_SAMPLE_FORMAT)((X) * (double)(((mpc_int64_t)1)<<32) )
#define MPC_MAKE_FRACT_CONST(X) (MPC_SAMPLE_FORMAT)((X) * 4294967296.0 )


#define MPC_MULTIPLY_FRACT_CONST(X,Y) MPC_MULTIPLY_FRACT(X,MPC_MAKE_FRACT_CONST(Y))
#define MPC_MULTIPLY_FRACT_CONST_FIX(X,Y,Z) ( MPC_MULTIPLY_FRACT(X,MPC_MAKE_FRACT_CONST( Y / (1<<(Z)) )) << (Z) )
#define MPC_MULTIPLY_FRACT_CONST_SHR(X,Y,Z) MPC_MULTIPLY_FRACT(X,MPC_MAKE_FRACT_CONST( Y / (1<<(Z)) ))

#define MPC_MULTIPLY_FLOAT_INT(X,Y) ((X)*(Y))
#define MPC_SCALE_CONST(X,Y,Z) MPC_MULTIPLY_EX(X,MAKE_MPC_SAMPLE_EX(Y,Z),(Z))
#define MPC_SCALE_CONST_SHL(X,Y,Z,S) MPC_MULTIPLY_EX(X,MAKE_MPC_SAMPLE_EX(Y,Z),(Z)-(S))
#define MPC_SCALE_CONST_SHR(X,Y,Z,S) MPC_MULTIPLY_EX(X,MAKE_MPC_SAMPLE_EX(Y,Z),(Z)+(S))
#define MPC_SHR(X,Y) ((X)>>(Y))
#define MPC_SHL(X,Y) ((X)<<(Y))

#else

//in floating-point mode, decoded samples are in -1...1 range

#define MAKE_MPC_SAMPLE(X) ((MPC_SAMPLE_FORMAT)(X))
#define MAKE_MPC_SAMPLE_EX(X,Y) ((MPC_SAMPLE_FORMAT)(X))

#define MPC_MULTIPLY_FRACT(X,Y) ((X)*(Y))
#define MPC_MAKE_FRACT_CONST(X) (X)
#define MPC_MULTIPLY_FRACT_CONST(X,Y) MPC_MULTPLY_FRACT(X,MPC_MAKE_FRACT_CONST(Y))
#define MPC_MULTIPLY_FRACT_CONST_SHR(X,Y,Z) MPC_MULTIPLY_FRACT(X,MPC_MAKE_FRACT_CONST( Y ))
#define MPC_MULTIPLY_FRACT_CONST_FIX(X,Y,Z) MPC_MULTIPLY_FRACT(X,MPC_MAKE_FRACT_CONST( Y ))

#define MPC_MULTIPLY_FLOAT_INT(X,Y) ((X)*(Y))
#define MPC_MULTIPLY(X,Y) ((X)*(Y))
#define MPC_MULTIPLY_EX(X,Y,Z) ((X)*(Y))
#define MPC_SCALE_CONST(X,Y,Z) ((X)*(Y))
#define MPC_SCALE_CONST_SHL(X,Y,Z,S) ((X)*(Y))
#define MPC_SCALE_CONST_SHR(X,Y,Z,S) ((X)*(Y))
#define MPC_SHR(X,Y) (X)
#define MPC_SHL(X,Y) (X)

#endif

#ifdef __cplusplus
}
#endif
#endif

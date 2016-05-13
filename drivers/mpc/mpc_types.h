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
#ifndef _MPC_TYPES_H_
#define _MPC_TYPES_H_
#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>
//#include <memory.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
typedef __int8           mpc_int8_t;
typedef unsigned __int8  mpc_uint8_t;
typedef __int16          mpc_int16_t;
typedef unsigned __int16 mpc_uint16_t;
typedef __int32          mpc_int32_t;
typedef unsigned __int32 mpc_uint32_t;
typedef __int64          mpc_int64_t;
typedef unsigned __int64 mpc_uint64_t;
#define mpc_inline __inline
#else
#include <stdint.h>
typedef int8_t   mpc_int8_t;
typedef uint8_t  mpc_uint8_t;
typedef int16_t  mpc_int16_t;
typedef uint16_t mpc_uint16_t;
typedef int32_t  mpc_int32_t;
typedef uint32_t mpc_uint32_t;
typedef int64_t  mpc_int64_t;
typedef uint64_t  mpc_uint64_t;
#define mpc_inline inline
#endif

typedef int mpc_int_t;
typedef unsigned int mpc_uint_t;
typedef size_t mpc_size_t;
typedef mpc_uint8_t mpc_bool_t;

// #define LONG_SEEK_TABLE
#ifdef LONG_SEEK_TABLE  // define as needed (mpc_uint32_t supports files up to 512 MB)
typedef mpc_uint64_t mpc_seek_t;
#else
typedef mpc_uint32_t mpc_seek_t;
#endif

# define mpc_int64_min -9223372036854775808ll
# define mpc_int64_max 9223372036854775807ll

typedef struct mpc_quantizer {
	mpc_int16_t  L [36];
	mpc_int16_t  R [36];
} mpc_quantizer;

/// Libmpcdec error codes
typedef enum mpc_status {
    MPC_STATUS_OK        =  0,
    MPC_STATUS_FILE      = -1,
    MPC_STATUS_SV7BETA   = -2,
    MPC_STATUS_CBR       = -3,
    MPC_STATUS_IS        = -4,
    MPC_STATUS_BLOCKSIZE = -5,
    MPC_STATUS_INVALIDSV = -6
} mpc_status;


#define MPC_FIXED_POINT_SHIFT 16

#ifdef MPC_FIXED_POINT
# define MPC_FIXED_POINT_FRACTPART 14
# define MPC_FIXED_POINT_SCALE_SHIFT (MPC_FIXED_POINT_SHIFT + MPC_FIXED_POINT_FRACTPART)
# define MPC_FIXED_POINT_SCALE (1 << (MPC_FIXED_POINT_SCALE_SHIFT - 1))
typedef mpc_int32_t MPC_SAMPLE_FORMAT;
#else
typedef float       MPC_SAMPLE_FORMAT;
#endif

enum {
    MPC_FALSE = 0,
    MPC_TRUE  = !MPC_FALSE
};

//// 'Cdecl' forces the use of standard C/C++ calling convention ///////
#if   defined _WIN32
# define mpc_cdecl           __cdecl
#elif defined __ZTC__
# define mpc_cdecl           _cdecl
#elif defined __TURBOC__
# define mpc_cdecl           cdecl
#else
# define mpc_cdecl
#endif

#ifdef __GNUC__
# define MPC_API __attribute__ ((visibility("default")))
#else
# define MPC_API
#endif

#ifdef __cplusplus
}
#endif
#endif

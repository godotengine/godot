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
/// \file decoder.h
#ifndef _MPCDEC_DECODER_H_
#define _MPCDEC_DECODER_H_
#ifdef WIN32
#pragma once
#endif

#include "reader.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SEEKING_TABLE_SIZE  256u
// set it to SLOW_SEEKING_WINDOW to not use fast seeking
#define FAST_SEEKING_WINDOW 32
// set it to FAST_SEEKING_WINDOW to only use fast seeking
#define SLOW_SEEKING_WINDOW 0x80000000

enum {
    MPC_V_MEM           = 2304,
    MPC_DECODER_MEMSIZE = 16384,  // overall buffer size
};

struct mpc_decoder_t {
    /// @name internal state variables
    //@{
	mpc_uint32_t stream_version;     ///< Streamversion of stream
	mpc_int32_t max_band;           ///< Maximum band-index used in stream (0...31)
	mpc_uint32_t ms;                 ///< Mid/side stereo (0: off, 1: on)
	mpc_uint32_t channels;           ///< Number of channels in stream

	mpc_uint64_t samples;            ///< Number of samples in stream

	mpc_uint64_t decoded_samples;    ///< Number of samples decoded from file begining
	mpc_uint32_t samples_to_skip;    ///< Number samples to skip (used for seeking)
	mpc_int32_t last_max_band;       ///< number of bands used in the last frame

    // randomizer state variables
    mpc_uint32_t  __r1;
    mpc_uint32_t  __r2;

    mpc_int32_t   SCF_Index_L [32] [3];
    mpc_int32_t   SCF_Index_R [32] [3];       // holds scalefactor-indices
    mpc_quantizer Q [32];                     // holds quantized samples
    mpc_int32_t   Res_L [32];
    mpc_int32_t   Res_R [32];                 // holds the chosen quantizer for each subband
    mpc_bool_t    DSCF_Flag_L [32];
    mpc_bool_t    DSCF_Flag_R [32];           // differential SCF used?
    mpc_int32_t   SCFI_L [32];
    mpc_int32_t   SCFI_R [32];                // describes order of transmitted SCF
    mpc_bool_t    MS_Flag[32];                // MS used?
#ifdef MPC_FIXED_POINT
    mpc_uint8_t   SCF_shift[256];
#endif

    MPC_SAMPLE_FORMAT V_L[MPC_V_MEM + 960];
    MPC_SAMPLE_FORMAT V_R[MPC_V_MEM + 960];
    MPC_SAMPLE_FORMAT Y_L[36][32];
    MPC_SAMPLE_FORMAT Y_R[36][32];
    MPC_SAMPLE_FORMAT SCF[256]; ///< holds adapted scalefactors (for clipping prevention)
    //@}
};

#ifdef __cplusplus
}
#endif
#endif

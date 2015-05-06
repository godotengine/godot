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
/// \file huffman.h
/// Data structures and functions for huffman coding.

#ifndef _MPCDEC_HUFFMAN_H_
#define _MPCDEC_HUFFMAN_H_
#ifdef WIN32
#pragma once
#endif

#include <mpc/mpc_types.h>

#ifdef __cplusplus
extern "C" {
#endif

// LUT size parameter, LUT size is 1 << LUT_DEPTH
#define LUT_DEPTH 6

/// Huffman table entry.
typedef struct mpc_huffman_t {
    mpc_uint16_t  Code;
    mpc_uint8_t  Length;
    mpc_int8_t   Value;
} mpc_huffman;

/// Huffman LUT entry.
typedef struct mpc_huff_lut_t {
	mpc_uint8_t  Length;
	mpc_int8_t   Value;
} mpc_huff_lut;

/// Type used for huffman LUT decoding
typedef struct mpc_lut_data_t {
	mpc_huffman const * const table;
	mpc_huff_lut lut[1 << LUT_DEPTH];
} mpc_lut_data;

/// Type used for canonical huffman decoding
typedef struct mpc_can_data_t {
	mpc_huffman const * const table;
	mpc_int8_t const * const sym;
	mpc_huff_lut lut[1 << LUT_DEPTH];
} mpc_can_data;

void huff_init_lut(const int bits);

#ifdef __cplusplus
}
#endif
#endif

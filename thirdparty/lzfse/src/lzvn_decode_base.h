/*
Copyright (c) 2015-2016, Apple Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:  

1.  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
    in the documentation and/or other materials provided with the distribution.

3.  Neither the name of the copyright holder(s) nor the names of any contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// LZVN low-level decoder (v2)
// Functions in the low-level API should switch to these at some point.
// Apr 2014

#ifndef LZVN_DECODE_BASE_H
#define LZVN_DECODE_BASE_H

#include "lzfse_internal.h"

/*! @abstract Base decoder state. */
typedef struct {

  // Decoder I/O

  // Next byte to read in source buffer
  const unsigned char *src;
  // Next byte after source buffer
  const unsigned char *src_end;

  // Next byte to write in destination buffer (by decoder)
  unsigned char *dst;
  // Valid range for destination buffer is [dst_begin, dst_end - 1]
  unsigned char *dst_begin;
  unsigned char *dst_end;
  // Next byte to read in destination buffer (modified by caller)
  unsigned char *dst_current;

  // Decoder state

  // Partially expanded match, or 0,0,0.
  // In that case, src points to the next literal to copy, or the next op-code
  // if L==0.
  size_t L, M, D;

  // Distance for last emitted match, or 0
  lzvn_offset d_prev;

  // Did we decode end-of-stream?
  int end_of_stream;

} lzvn_decoder_state;

/*! @abstract Decode source to destination.
 *  Updates \p state (src,dst,d_prev). */
void lzvn_decode(lzvn_decoder_state *state);

#endif // LZVN_DECODE_BASE_H

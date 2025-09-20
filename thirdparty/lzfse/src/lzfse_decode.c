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

// LZFSE decode API

#include "lzfse.h"
#include "lzfse_internal.h"

size_t lzfse_decode_scratch_size() { return sizeof(lzfse_decoder_state); }

size_t lzfse_decode_buffer_with_scratch(uint8_t *__restrict dst_buffer, 
                         size_t dst_size, const uint8_t *__restrict src_buffer,
                         size_t src_size, void *__restrict scratch_buffer) {
  lzfse_decoder_state *s = (lzfse_decoder_state *)scratch_buffer;
  memset(s, 0x00, sizeof(*s));

  // Initialize state
  s->src = src_buffer;
  s->src_begin = src_buffer;
  s->src_end = s->src + src_size;
  s->dst = dst_buffer;
  s->dst_begin = dst_buffer;
  s->dst_end = dst_buffer + dst_size;

  // Decode
  int status = lzfse_decode(s);
  if (status == LZFSE_STATUS_DST_FULL)
    return dst_size;
  if (status != LZFSE_STATUS_OK)
    return 0;                           // failed
  return (size_t)(s->dst - dst_buffer); // bytes written
}

size_t lzfse_decode_buffer(uint8_t *__restrict dst_buffer, size_t dst_size,
                           const uint8_t *__restrict src_buffer,
                           size_t src_size, void *__restrict scratch_buffer) {
  int has_malloc = 0;
  size_t ret = 0;

  // Deal with the possible NULL pointer
  if (scratch_buffer == NULL) {
    // +1 in case scratch size could be zero
    scratch_buffer = malloc(lzfse_decode_scratch_size() + 1);
    has_malloc = 1;
  }
  if (scratch_buffer == NULL)
    return 0;
  ret = lzfse_decode_buffer_with_scratch(dst_buffer, 
                               dst_size, src_buffer, 
                               src_size, scratch_buffer);
  if (has_malloc)
    free(scratch_buffer);
  return ret;
} 

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

// LZFSE encode API

#include "lzfse.h"
#include "lzfse_internal.h"

size_t lzfse_encode_scratch_size() {
  size_t s1 = sizeof(lzfse_encoder_state);
  size_t s2 = lzvn_encode_scratch_size();
  return (s1 > s2) ? s1 : s2; // max(lzfse,lzvn)
}

size_t lzfse_encode_buffer_with_scratch(uint8_t *__restrict dst_buffer, 
                       size_t dst_size, const uint8_t *__restrict src_buffer,
                       size_t src_size, void *__restrict scratch_buffer) {
  const size_t original_size = src_size;

  // If input is really really small, go directly to uncompressed buffer
  // (because LZVN will refuse to encode it, and we will report a failure)
  if (src_size < LZVN_ENCODE_MIN_SRC_SIZE)
    goto try_uncompressed;

  // If input is too small, try encoding with LZVN
  if (src_size < LZFSE_ENCODE_LZVN_THRESHOLD) {
    // need header + end-of-stream marker
    size_t extra_size = 4 + sizeof(lzvn_compressed_block_header);
    if (dst_size <= extra_size)
      goto try_uncompressed; // DST is really too small, give up

    size_t sz = lzvn_encode_buffer(
        dst_buffer + sizeof(lzvn_compressed_block_header),
        dst_size - extra_size, src_buffer, src_size, scratch_buffer);
    if (sz == 0 || sz >= src_size)
      goto try_uncompressed; // failed, or no compression, fall back to
                             // uncompressed block

    // If we could encode, setup header and end-of-stream marker (we left room
    // for them, no need to test)
    lzvn_compressed_block_header header;
    header.magic = LZFSE_COMPRESSEDLZVN_BLOCK_MAGIC;
    header.n_raw_bytes = (uint32_t)src_size;
    header.n_payload_bytes = (uint32_t)sz;
    memcpy(dst_buffer, &header, sizeof(header));
    store4(dst_buffer + sizeof(lzvn_compressed_block_header) + sz,
           LZFSE_ENDOFSTREAM_BLOCK_MAGIC);

    return sz + extra_size;
  }

  // Try encoding with LZFSE
  {
    lzfse_encoder_state *state = scratch_buffer;
    memset(state, 0x00, sizeof *state);
    if (lzfse_encode_init(state) != LZFSE_STATUS_OK)
      goto try_uncompressed;
    state->dst = dst_buffer;
    state->dst_begin = dst_buffer;
    state->dst_end = &dst_buffer[dst_size];
    state->src = src_buffer;
    state->src_encode_i = 0;

    if (src_size >= 0xffffffffU) {
      //  lzfse only uses 32 bits for offsets internally, so if the input
      //  buffer is really huge, we need to process it in smaller chunks.
      //  Note that we switch over to this path for sizes much smaller
      //  2GB because it's actually faster to change algorithms well before
      //  it's necessary for correctness.
      //  The first chunk, we just process normally.
      const lzfse_offset encoder_block_size = 262144;
      state->src_end = encoder_block_size;
      if (lzfse_encode_base(state) != LZFSE_STATUS_OK)
        goto try_uncompressed;
      src_size -= encoder_block_size;
      while (src_size >= encoder_block_size) {
        //  All subsequent chunks require a translation to keep the offsets
        //  from getting too big.  Note that we are always going from
        //  encoder_block_size up to 2*encoder_block_size so that the
        //  offsets remain positive (as opposed to resetting to zero and
        //  having negative offsets).
        state->src_end = 2 * encoder_block_size;
        if (lzfse_encode_base(state) != LZFSE_STATUS_OK)
          goto try_uncompressed;
        lzfse_encode_translate(state, encoder_block_size);
        src_size -= encoder_block_size;
      }
      //  Set the end for the final chunk.
      state->src_end = encoder_block_size + (lzfse_offset)src_size;
    }
    //  If the source buffer is small enough to use 32-bit offsets, we simply
    //  encode the whole thing in a single chunk.
    else
      state->src_end = (lzfse_offset)src_size;
    //  This is either the trailing chunk (if the source file is huge), or
    //  the whole source file.
    if (lzfse_encode_base(state) != LZFSE_STATUS_OK)
      goto try_uncompressed;
    if (lzfse_encode_finish(state) != LZFSE_STATUS_OK)
      goto try_uncompressed;
    //  No error occured, return compressed size.
    return state->dst - dst_buffer;
  }

try_uncompressed:
  //  Compression failed for some reason.  If we can fit the data into the
  //  output buffer uncompressed, go ahead and do that instead.
  if (original_size + 12 <= dst_size && original_size < INT32_MAX) {
    uncompressed_block_header header = {.magic = LZFSE_UNCOMPRESSED_BLOCK_MAGIC,
                                        .n_raw_bytes = (uint32_t)src_size};
    uint8_t *dst_end = dst_buffer;
    memcpy(dst_end, &header, sizeof header);
    dst_end += sizeof header;
    memcpy(dst_end, src_buffer, original_size);
    dst_end += original_size;
    store4(dst_end, LZFSE_ENDOFSTREAM_BLOCK_MAGIC);
    dst_end += 4;
    return dst_end - dst_buffer;
  }

  //  Otherwise, there's nothing we can do, so return zero.
  return 0;
}

size_t lzfse_encode_buffer(uint8_t *__restrict dst_buffer, size_t dst_size,
                           const uint8_t *__restrict src_buffer,
                           size_t src_size, void *__restrict scratch_buffer) {
  int has_malloc = 0;
  size_t ret = 0;

  // Deal with the possible NULL pointer
  if (scratch_buffer == NULL) {
    // +1 in case scratch size could be zero
    scratch_buffer = malloc(lzfse_encode_scratch_size() + 1);
    has_malloc = 1;
  }
  if (scratch_buffer == NULL)
    return 0;
  ret = lzfse_encode_buffer_with_scratch(dst_buffer, 
                        dst_size, src_buffer, 
                        src_size, scratch_buffer);
  if (has_malloc)
    free(scratch_buffer);
  return ret;
} 

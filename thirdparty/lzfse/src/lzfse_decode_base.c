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

#include "lzfse_internal.h"
#include "lzvn_decode_base.h"

/*! @abstract Decode an entry value from next bits of stream.
 *  Return \p value, and set \p *nbits to the number of bits to consume
 *  (starting with LSB). */
static inline int lzfse_decode_v1_freq_value(uint32_t bits, int *nbits) {
  static const int8_t lzfse_freq_nbits_table[32] = {
      2, 3, 2, 5, 2, 3, 2, 8, 2, 3, 2, 5, 2, 3, 2, 14,
      2, 3, 2, 5, 2, 3, 2, 8, 2, 3, 2, 5, 2, 3, 2, 14};
  static const int8_t lzfse_freq_value_table[32] = {
      0, 2, 1, 4, 0, 3, 1, -1, 0, 2, 1, 5, 0, 3, 1, -1,
      0, 2, 1, 6, 0, 3, 1, -1, 0, 2, 1, 7, 0, 3, 1, -1};

  uint32_t b = bits & 31; // lower 5 bits
  int n = lzfse_freq_nbits_table[b];
  *nbits = n;

  // Special cases for > 5 bits encoding
  if (n == 8)
    return 8 + ((bits >> 4) & 0xf);
  if (n == 14)
    return 24 + ((bits >> 4) & 0x3ff);

  // <= 5 bits encoding from table
  return lzfse_freq_value_table[b];
}

/*! @abstract Extracts up to 32 bits from a 64-bit field beginning at
 *  \p offset, and zero-extends them to a \p uint32_t.
 *
 *  If we number the bits of \p v from 0 (least significant) to 63 (most
 *  significant), the result is bits \p offset to \p offset+nbits-1. */
static inline uint32_t get_field(uint64_t v, int offset, int nbits) {
  assert(offset + nbits < 64 && offset >= 0 && nbits <= 32);
  if (nbits == 32)
    return (uint32_t)(v >> offset);
  return (uint32_t)((v >> offset) & ((1 << nbits) - 1));
}

/*! @abstract Return \c header_size field from a \c lzfse_compressed_block_header_v2. */
static inline uint32_t
lzfse_decode_v2_header_size(const lzfse_compressed_block_header_v2 *in) {
  return get_field(in->packed_fields[2], 0, 32);
}

/*! @abstract Decode all fields from a \c lzfse_compressed_block_header_v2 to a
 * \c lzfse_compressed_block_header_v1.
 * @return 0 on success.
 * @return -1 on failure. */
static inline int lzfse_decode_v1(lzfse_compressed_block_header_v1 *out,
                                const lzfse_compressed_block_header_v2 *in) {
  // Clear all fields
  memset(out, 0x00, sizeof(lzfse_compressed_block_header_v1));

  uint64_t v0 = in->packed_fields[0];
  uint64_t v1 = in->packed_fields[1];
  uint64_t v2 = in->packed_fields[2];

  out->magic = LZFSE_COMPRESSEDV1_BLOCK_MAGIC;
  out->n_raw_bytes = in->n_raw_bytes;

  // Literal state
  out->n_literals = get_field(v0, 0, 20);
  out->n_literal_payload_bytes = get_field(v0, 20, 20);
  out->literal_bits = (int)get_field(v0, 60, 3) - 7;
  out->literal_state[0] = get_field(v1, 0, 10);
  out->literal_state[1] = get_field(v1, 10, 10);
  out->literal_state[2] = get_field(v1, 20, 10);
  out->literal_state[3] = get_field(v1, 30, 10);

  // L,M,D state
  out->n_matches = get_field(v0, 40, 20);
  out->n_lmd_payload_bytes = get_field(v1, 40, 20);
  out->lmd_bits = (int)get_field(v1, 60, 3) - 7;
  out->l_state = get_field(v2, 32, 10);
  out->m_state = get_field(v2, 42, 10);
  out->d_state = get_field(v2, 52, 10);

  // Total payload size
  out->n_payload_bytes =
      out->n_literal_payload_bytes + out->n_lmd_payload_bytes;

  // Freq tables
  uint16_t *dst = &(out->l_freq[0]);
  const uint8_t *src = &(in->freq[0]);
  const uint8_t *src_end =
      (const uint8_t *)in + get_field(v2, 0, 32); // first byte after header
  uint32_t accum = 0;
  int accum_nbits = 0;

  // No freq tables?
  if (src_end == src)
    return 0; // OK, freq tables were omitted

  for (int i = 0; i < LZFSE_ENCODE_L_SYMBOLS + LZFSE_ENCODE_M_SYMBOLS +
                          LZFSE_ENCODE_D_SYMBOLS + LZFSE_ENCODE_LITERAL_SYMBOLS;
       i++) {
    // Refill accum, one byte at a time, until we reach end of header, or accum
    // is full
    while (src < src_end && accum_nbits + 8 <= 32) {
      accum |= (uint32_t)(*src) << accum_nbits;
      accum_nbits += 8;
      src++;
    }

    // Decode and store value
    int nbits = 0;
    dst[i] = lzfse_decode_v1_freq_value(accum, &nbits);

    if (nbits > accum_nbits)
      return -1; // failed

    // Consume nbits bits
    accum >>= nbits;
    accum_nbits -= nbits;
  }

  if (accum_nbits >= 8 || src != src_end)
    return -1; // we need to end up exactly at the end of header, with less than
               // 8 bits in accumulator

  return 0;
}

static inline void copy(uint8_t *dst, const uint8_t *src, size_t length) {
  const uint8_t *dst_end = dst + length;
  do {
    copy8(dst, src);
    dst += 8;
    src += 8;
  } while (dst < dst_end);
}

static int lzfse_decode_lmd(lzfse_decoder_state *s) {
  lzfse_compressed_block_decoder_state *bs = &(s->compressed_lzfse_block_state);
  fse_state l_state = bs->l_state;
  fse_state m_state = bs->m_state;
  fse_state d_state = bs->d_state;
  fse_in_stream in = bs->lmd_in_stream;
  const uint8_t *src_start = s->src_begin;
  const uint8_t *src = s->src + bs->lmd_in_buf;
  const uint8_t *lit = bs->current_literal;
  uint8_t *dst = s->dst;
  uint32_t symbols = bs->n_matches;
  int32_t L = bs->l_value;
  int32_t M = bs->m_value;
  int32_t D = bs->d_value;

  assert(l_state < LZFSE_ENCODE_L_STATES);
  assert(m_state < LZFSE_ENCODE_M_STATES);
  assert(d_state < LZFSE_ENCODE_D_STATES);

  //  Number of bytes remaining in the destination buffer, minus 32 to
  //  provide a margin of safety for using overlarge copies on the fast path.
  //  This is a signed quantity, and may go negative when we are close to the
  //  end of the buffer.  That's OK; we're careful about how we handle it
  //  in the slow-and-careful match execution path.
  ptrdiff_t remaining_bytes = s->dst_end - dst - 32;

  //  If L or M is non-zero, that means that we have already started decoding
  //  this block, and that we needed to interrupt decoding to get more space
  //  from the caller.  There's a pending L, M, D triplet that we weren't
  //  able to completely process.  Jump ahead to finish executing that symbol
  //  before decoding new values.
  if (L || M)
    goto ExecuteMatch;

  while (symbols > 0) {
    int res;
    //  Decode the next L, M, D symbol from the input stream.
    res = fse_in_flush(&in, &src, src_start);
    if (res) {
      return LZFSE_STATUS_ERROR;
    }
    L = fse_value_decode(&l_state, bs->l_decoder, &in);
    assert(l_state < LZFSE_ENCODE_L_STATES);
    if ((lit + L) >= (bs->literals + LZFSE_LITERALS_PER_BLOCK + 64)) {
      return LZFSE_STATUS_ERROR;
    }
    res = fse_in_flush2(&in, &src, src_start);
    if (res) {
      return LZFSE_STATUS_ERROR;
    }
    M = fse_value_decode(&m_state, bs->m_decoder, &in);
    assert(m_state < LZFSE_ENCODE_M_STATES);
    res = fse_in_flush2(&in, &src, src_start);
    if (res) {
      return LZFSE_STATUS_ERROR;
    }
    int32_t new_d = fse_value_decode(&d_state, bs->d_decoder, &in);
    assert(d_state < LZFSE_ENCODE_D_STATES);
    D = new_d ? new_d : D;
    symbols--;

  ExecuteMatch:
    //  Error if D is out of range, so that we avoid passing through
    //  uninitialized data or accesssing memory out of the destination
    //  buffer.
    if ((uint32_t)D > dst + L - s->dst_begin)
      return LZFSE_STATUS_ERROR;

    if (L + M <= remaining_bytes) {
      //  If we have plenty of space remaining, we can copy the literal
      //  and match with 16- and 32-byte operations, without worrying
      //  about writing off the end of the buffer.
      remaining_bytes -= L + M;
      copy(dst, lit, L);
      dst += L;
      lit += L;
      //  For the match, we have two paths; a fast copy by 16-bytes if
      //  the match distance is large enough to allow it, and a more
      //  careful path that applies a permutation to account for the
      //  possible overlap between source and destination if the distance
      //  is small.
      if (D >= 8 || D >= M)
        copy(dst, dst - D, M);
      else
        for (size_t i = 0; i < M; i++)
          dst[i] = dst[i - D];
      dst += M;
    }

    else {
      //  Otherwise, we are very close to the end of the destination
      //  buffer, so we cannot use wide copies that slop off the end
      //  of the region that we are copying to. First, we restore
      //  the true length remaining, rather than the sham value we've
      //  been using so far.
      remaining_bytes += 32;
      //  Now, we process the literal. Either there's space for it
      //  or there isn't; if there is, we copy the whole thing and
      //  update all the pointers and lengths to reflect the copy.
      if (L <= remaining_bytes) {
        for (size_t i = 0; i < L; i++)
          dst[i] = lit[i];
        dst += L;
        lit += L;
        remaining_bytes -= L;
        L = 0;
      }
      //  There isn't enough space to fit the whole literal. Copy as
      //  much of it as we can, update the pointers and the value of
      //  L, and report that the destination buffer is full. Note that
      //  we always write right up to the end of the destination buffer.
      else {
        for (size_t i = 0; i < remaining_bytes; i++)
          dst[i] = lit[i];
        dst += remaining_bytes;
        lit += remaining_bytes;
        L -= remaining_bytes;
        goto DestinationBufferIsFull;
      }
      //  The match goes just like the literal does. We copy as much as
      //  we can byte-by-byte, and if we reach the end of the buffer
      //  before finishing, we return to the caller indicating that
      //  the buffer is full.
      if (M <= remaining_bytes) {
        for (size_t i = 0; i < M; i++)
          dst[i] = dst[i - D];
        dst += M;
        remaining_bytes -= M;
        M = 0;
        (void)M; // no dead store warning
                 //  We don't need to update M = 0, because there's no partial
                 //  symbol to continue executing. Either we're at the end of
                 //  the block, in which case we will never need to resume with
                 //  this state, or we're going to decode another L, M, D set,
                 //  which will overwrite M anyway.
                 //
                 // But we still set M = 0, to maintain the post-condition.
      } else {
        for (size_t i = 0; i < remaining_bytes; i++)
          dst[i] = dst[i - D];
        dst += remaining_bytes;
        M -= remaining_bytes;
      DestinationBufferIsFull:
        //  Because we want to be able to resume decoding where we've left
        //  off (even in the middle of a literal or match), we need to
        //  update all of the block state fields with the current values
        //  so that we can resume execution from this point once the
        //  caller has given us more space to write into.
        bs->l_value = L;
        bs->m_value = M;
        bs->d_value = D;
        bs->l_state = l_state;
        bs->m_state = m_state;
        bs->d_state = d_state;
        bs->lmd_in_stream = in;
        bs->n_matches = symbols;
        bs->lmd_in_buf = (uint32_t)(src - s->src);
        bs->current_literal = lit;
        s->dst = dst;
        return LZFSE_STATUS_DST_FULL;
      }
      //  Restore the "sham" decremented value of remaining_bytes and
      //  continue to the next L, M, D triple. We'll just be back in
      //  the careful path again, but this only happens at the very end
      //  of the buffer, so a little minor inefficiency here is a good
      //  tradeoff for simpler code.
      remaining_bytes -= 32;
    }
  }
  //  Because we've finished with the whole block, we don't need to update
  //  any of the blockstate fields; they will not be used again. We just
  //  update the destination pointer in the state object and return.
  s->dst = dst;
  return LZFSE_STATUS_OK;
}

int lzfse_decode(lzfse_decoder_state *s) {
  while (1) {
    // Are we inside a block?
    switch (s->block_magic) {
    case LZFSE_NO_BLOCK_MAGIC: {
      // We need at least 4 bytes of magic number to identify next block
      if (s->src + 4 > s->src_end)
        return LZFSE_STATUS_SRC_EMPTY; // SRC truncated
      uint32_t magic = load4(s->src);

      if (magic == LZFSE_ENDOFSTREAM_BLOCK_MAGIC) {
        s->src += 4;
        s->end_of_stream = 1;
        return LZFSE_STATUS_OK; // done
      }

      if (magic == LZFSE_UNCOMPRESSED_BLOCK_MAGIC) {
        if (s->src + sizeof(uncompressed_block_header) > s->src_end)
          return LZFSE_STATUS_SRC_EMPTY; // SRC truncated
        // Setup state for uncompressed block
        uncompressed_block_decoder_state *bs = &(s->uncompressed_block_state);
        bs->n_raw_bytes =
            load4(s->src + offsetof(uncompressed_block_header, n_raw_bytes));
        s->src += sizeof(uncompressed_block_header);
        s->block_magic = magic;
        break;
      }

      if (magic == LZFSE_COMPRESSEDLZVN_BLOCK_MAGIC) {
        if (s->src + sizeof(lzvn_compressed_block_header) > s->src_end)
          return LZFSE_STATUS_SRC_EMPTY; // SRC truncated
        // Setup state for compressed LZVN block
        lzvn_compressed_block_decoder_state *bs =
            &(s->compressed_lzvn_block_state);
        bs->n_raw_bytes =
            load4(s->src + offsetof(lzvn_compressed_block_header, n_raw_bytes));
        bs->n_payload_bytes = load4(
            s->src + offsetof(lzvn_compressed_block_header, n_payload_bytes));
        bs->d_prev = 0;
        s->src += sizeof(lzvn_compressed_block_header);
        s->block_magic = magic;
        break;
      }

      if (magic == LZFSE_COMPRESSEDV1_BLOCK_MAGIC ||
          magic == LZFSE_COMPRESSEDV2_BLOCK_MAGIC) {
        lzfse_compressed_block_header_v1 header1;
        size_t header_size = 0;

        // Decode compressed headers
        if (magic == LZFSE_COMPRESSEDV2_BLOCK_MAGIC) {
          // Check we have the fixed part of the structure
          if (s->src + offsetof(lzfse_compressed_block_header_v2, freq) > s->src_end)
            return LZFSE_STATUS_SRC_EMPTY; // SRC truncated

          // Get size, and check we have the entire structure
          const lzfse_compressed_block_header_v2 *header2 =
              (const lzfse_compressed_block_header_v2 *)s->src; // not aligned, OK
          header_size = lzfse_decode_v2_header_size(header2);
          if (s->src + header_size > s->src_end)
            return LZFSE_STATUS_SRC_EMPTY; // SRC truncated
          int decodeStatus = lzfse_decode_v1(&header1, header2);
          if (decodeStatus != 0)
            return LZFSE_STATUS_ERROR; // failed
        } else {
          if (s->src + sizeof(lzfse_compressed_block_header_v1) > s->src_end)
            return LZFSE_STATUS_SRC_EMPTY; // SRC truncated
          memcpy(&header1, s->src, sizeof(lzfse_compressed_block_header_v1));
          header_size = sizeof(lzfse_compressed_block_header_v1);
        }

        // We require the header + entire encoded block to be present in SRC
        // during the entire block decoding.
        // This can be relaxed somehow, if it becomes a limiting factor, at the
        // price of a more complex state maintenance.
        // For DST, we can't easily require space for the entire decoded block,
        // because it may expand to something very very large.
        if (s->src + header_size + header1.n_literal_payload_bytes +
                header1.n_lmd_payload_bytes >
            s->src_end)
          return LZFSE_STATUS_SRC_EMPTY; // need all encoded block

        // Sanity checks
        if (lzfse_check_block_header_v1(&header1) != 0) {
          return LZFSE_STATUS_ERROR;
        }

        // Skip header
        s->src += header_size;

        // Setup state for compressed V1 block from header
        lzfse_compressed_block_decoder_state *bs =
            &(s->compressed_lzfse_block_state);
        bs->n_lmd_payload_bytes = header1.n_lmd_payload_bytes;
        bs->n_matches = header1.n_matches;
        fse_init_decoder_table(LZFSE_ENCODE_LITERAL_STATES,
                               LZFSE_ENCODE_LITERAL_SYMBOLS,
                               header1.literal_freq, bs->literal_decoder);
        fse_init_value_decoder_table(
            LZFSE_ENCODE_L_STATES, LZFSE_ENCODE_L_SYMBOLS, header1.l_freq,
            l_extra_bits, l_base_value, bs->l_decoder);
        fse_init_value_decoder_table(
            LZFSE_ENCODE_M_STATES, LZFSE_ENCODE_M_SYMBOLS, header1.m_freq,
            m_extra_bits, m_base_value, bs->m_decoder);
        fse_init_value_decoder_table(
            LZFSE_ENCODE_D_STATES, LZFSE_ENCODE_D_SYMBOLS, header1.d_freq,
            d_extra_bits, d_base_value, bs->d_decoder);

        // Decode literals
        {
          fse_in_stream in;
          const uint8_t *buf_start = s->src_begin;
          s->src += header1.n_literal_payload_bytes; // skip literal payload
          const uint8_t *buf = s->src; // read bits backwards from the end
          if (fse_in_init(&in, header1.literal_bits, &buf, buf_start) != 0)
            return LZFSE_STATUS_ERROR;

          fse_state state0 = header1.literal_state[0];
          fse_state state1 = header1.literal_state[1];
          fse_state state2 = header1.literal_state[2];
          fse_state state3 = header1.literal_state[3];

          for (uint32_t i = 0; i < header1.n_literals; i += 4) // n_literals is multiple of 4
          {
#if FSE_IOSTREAM_64
            if (fse_in_flush(&in, &buf, buf_start) != 0)
              return LZFSE_STATUS_ERROR; // [57, 64] bits
            bs->literals[i + 0] =
                fse_decode(&state0, bs->literal_decoder, &in); // 10b max
            bs->literals[i + 1] =
                fse_decode(&state1, bs->literal_decoder, &in); // 10b max
            bs->literals[i + 2] =
                fse_decode(&state2, bs->literal_decoder, &in); // 10b max
            bs->literals[i + 3] =
                fse_decode(&state3, bs->literal_decoder, &in); // 10b max
#else
            if (fse_in_flush(&in, &buf, buf_start) != 0)
              return LZFSE_STATUS_ERROR; // [25, 23] bits
            bs->literals[i + 0] =
                fse_decode(&state0, bs->literal_decoder, &in); // 10b max
            bs->literals[i + 1] =
                fse_decode(&state1, bs->literal_decoder, &in); // 10b max
            if (fse_in_flush(&in, &buf, buf_start) != 0)
              return LZFSE_STATUS_ERROR; // [25, 23] bits
            bs->literals[i + 2] =
                fse_decode(&state2, bs->literal_decoder, &in); // 10b max
            bs->literals[i + 3] =
                fse_decode(&state3, bs->literal_decoder, &in); // 10b max
#endif
          }

          bs->current_literal = bs->literals;
        } // literals

        // SRC is not incremented to skip the LMD payload, since we need it
        // during block decode.
        // We will increment SRC at the end of the block only after this point.

        // Initialize the L,M,D decode stream, do not start decoding matches
        // yet, and store decoder state
        {
          fse_in_stream in;
          // read bits backwards from the end
          const uint8_t *buf = s->src + header1.n_lmd_payload_bytes;
          if (fse_in_init(&in, header1.lmd_bits, &buf, s->src) != 0)
            return LZFSE_STATUS_ERROR;

          bs->l_state = header1.l_state;
          bs->m_state = header1.m_state;
          bs->d_state = header1.d_state;
          bs->lmd_in_buf = (uint32_t)(buf - s->src);
          bs->l_value = bs->m_value = 0;
          //  Initialize D to an illegal value so we can't erroneously use
          //  an uninitialized "previous" value.
          bs->d_value = -1;
          bs->lmd_in_stream = in;
        }

        s->block_magic = magic;
        break;
      }

      // Here we have an invalid magic number
      return LZFSE_STATUS_ERROR;
    } // LZFSE_NO_BLOCK_MAGIC

    case LZFSE_UNCOMPRESSED_BLOCK_MAGIC: {
      uncompressed_block_decoder_state *bs = &(s->uncompressed_block_state);

      //  Compute the size (in bytes) of the data that we will actually copy.
      //  This size is minimum(bs->n_raw_bytes, space in src, space in dst).

      uint32_t copy_size = bs->n_raw_bytes; // bytes left to copy
      if (copy_size == 0) {
        s->block_magic = 0;
        break;
      } // end of block

      if (s->src_end <= s->src)
        return LZFSE_STATUS_SRC_EMPTY; // need more SRC data
      const size_t src_space = s->src_end - s->src;
      if (copy_size > src_space)
        copy_size = (uint32_t)src_space; // limit to SRC data (> 0)

      if (s->dst_end <= s->dst)
        return LZFSE_STATUS_DST_FULL; // need more DST capacity
      const size_t dst_space = s->dst_end - s->dst;
      if (copy_size > dst_space)
        copy_size = (uint32_t)dst_space; // limit to DST capacity (> 0)

      // Now that we know that the copy size is bounded to the source and
      // dest buffers, go ahead and copy the data.
      // We always have copy_size > 0 here
      memcpy(s->dst, s->src, copy_size);
      s->src += copy_size;
      s->dst += copy_size;
      bs->n_raw_bytes -= copy_size;

      break;
    } // LZFSE_UNCOMPRESSED_BLOCK_MAGIC

    case LZFSE_COMPRESSEDV1_BLOCK_MAGIC:
    case LZFSE_COMPRESSEDV2_BLOCK_MAGIC: {
      lzfse_compressed_block_decoder_state *bs =
          &(s->compressed_lzfse_block_state);
      // Require the entire LMD payload to be in SRC
      if (s->src_end <= s->src ||
          bs->n_lmd_payload_bytes > (size_t)(s->src_end - s->src))
        return LZFSE_STATUS_SRC_EMPTY;

      int status = lzfse_decode_lmd(s);
      if (status != LZFSE_STATUS_OK)
        return status;

      s->block_magic = LZFSE_NO_BLOCK_MAGIC;
      s->src += bs->n_lmd_payload_bytes; // to next block
      break;
    } // LZFSE_COMPRESSEDV1_BLOCK_MAGIC || LZFSE_COMPRESSEDV2_BLOCK_MAGIC

    case LZFSE_COMPRESSEDLZVN_BLOCK_MAGIC: {
      lzvn_compressed_block_decoder_state *bs =
          &(s->compressed_lzvn_block_state);
      if (bs->n_payload_bytes > 0 && s->src_end <= s->src)
        return LZFSE_STATUS_SRC_EMPTY; // need more SRC data

      // Init LZVN decoder state
      lzvn_decoder_state dstate;
      memset(&dstate, 0x00, sizeof(dstate));
      dstate.src = s->src;
      dstate.src_end = s->src_end;
      if (dstate.src_end - s->src > bs->n_payload_bytes)
        dstate.src_end = s->src + bs->n_payload_bytes; // limit to payload bytes
      dstate.dst_begin = s->dst_begin;
      dstate.dst = s->dst;
      dstate.dst_end = s->dst_end;
      if (dstate.dst_end - s->dst > bs->n_raw_bytes)
        dstate.dst_end = s->dst + bs->n_raw_bytes; // limit to raw bytes
      dstate.d_prev = bs->d_prev;
      dstate.end_of_stream = 0;

      // Run LZVN decoder
      lzvn_decode(&dstate);

      // Update our state
      size_t src_used = dstate.src - s->src;
      size_t dst_used = dstate.dst - s->dst;
      if (src_used > bs->n_payload_bytes || dst_used > bs->n_raw_bytes)
        return LZFSE_STATUS_ERROR; // sanity check
      s->src = dstate.src;
      s->dst = dstate.dst;
      bs->n_payload_bytes -= (uint32_t)src_used;
      bs->n_raw_bytes -= (uint32_t)dst_used;
      bs->d_prev = (uint32_t)dstate.d_prev;

      // Test end of block
      if (bs->n_payload_bytes == 0 && bs->n_raw_bytes == 0 &&
          dstate.end_of_stream) {
        s->block_magic = 0;
        break;
      } // block done

      // Check for invalid state
      if (bs->n_payload_bytes == 0 || bs->n_raw_bytes == 0 ||
          dstate.end_of_stream)
        return LZFSE_STATUS_ERROR;

      // Here, block is not done and state is valid, so we need more space in dst.
      return LZFSE_STATUS_DST_FULL;
    }

    default:
      return LZFSE_STATUS_ERROR; // invalid magic

    } // switch magic

  } // block loop

  return LZFSE_STATUS_OK;
}

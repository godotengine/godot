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

// LZFSE encoder

#include "lzfse_internal.h"
#include "lzfse_encode_tables.h"

/*! @abstract Get hash in range [0, LZFSE_ENCODE_HASH_VALUES-1] from 4 bytes in X. */
static inline uint32_t hashX(uint32_t x) {
  return (x * 2654435761U) >>
         (32 - LZFSE_ENCODE_HASH_BITS); // Knuth multiplicative hash
}

/*! @abstract Return value with all 0 except nbits<=32 unsigned bits from V
 * at bit offset OFFSET.
 * V is assumed to fit on nbits bits. */
static inline uint64_t setField(uint32_t v, int offset, int nbits) {
  assert(offset + nbits < 64 && offset >= 0 && nbits <= 32);
  assert(nbits == 32 || (v < (1U << nbits)));
  return ((uint64_t)v << (uint64_t)offset);
}

/*! @abstract Encode all fields, except freq, from a
 * lzfse_compressed_block_header_v1 to a lzfse_compressed_block_header_v2.
 * All but the header_size and freq fields of the output are modified. */
static inline void
lzfse_encode_v1_state(lzfse_compressed_block_header_v2 *out,
                      const lzfse_compressed_block_header_v1 *in) {
  out->magic = LZFSE_COMPRESSEDV2_BLOCK_MAGIC;
  out->n_raw_bytes = in->n_raw_bytes;

  // Literal state
  out->packed_fields[0] = setField(in->n_literals, 0, 20) |
              setField(in->n_literal_payload_bytes, 20, 20) |
              setField(in->n_matches, 40, 20) |
              setField(7 + in->literal_bits, 60, 3);
  out->packed_fields[1] = setField(in->literal_state[0], 0, 10) |
              setField(in->literal_state[1], 10, 10) |
              setField(in->literal_state[2], 20, 10) |
              setField(in->literal_state[3], 30, 10) |
              setField(in->n_lmd_payload_bytes, 40, 20) |
              setField(7 + in->lmd_bits, 60, 3);
  out->packed_fields[2] = out->packed_fields[2] // header_size already stored in v[2]
              | setField(in->l_state, 32, 10) | setField(in->m_state, 42, 10) |
              setField(in->d_state, 52, 10);
}

/*! @abstract Encode an entry value in a freq table. Return bits, and sets
 * *nbits to the number of bits to serialize. */
static inline uint32_t lzfse_encode_v1_freq_value(int value, int *nbits) {
  // Fixed Huffman code, bits are read from LSB.
  // Note that we rely on the position of the first '0' bit providing the number
  // of bits.
  switch (value) {
  case 0:
    *nbits = 2;
    return 0; //    0.0
  case 1:
    *nbits = 2;
    return 2; //    1.0
  case 2:
    *nbits = 3;
    return 1; //   0.01
  case 3:
    *nbits = 3;
    return 5; //   1.01
  case 4:
    *nbits = 5;
    return 3; // 00.011
  case 5:
    *nbits = 5;
    return 11; // 01.011
  case 6:
    *nbits = 5;
    return 19; // 10.011
  case 7:
    *nbits = 5;
    return 27; // 11.011
  default:
    break;
  }
  if (value < 24) {
    *nbits = 8;                    // 4+4
    return 7 + ((value - 8) << 4); // xxxx.0111
  }
  // 24..1047
  *nbits = 14;                     // 4+10
  return ((value - 24) << 4) + 15; // xxxxxxxxxx.1111
}

/*! @abstract Encode all tables from a lzfse_compressed_block_header_v1
 * to a lzfse_compressed_block_header_v2.
 * Only the header_size and freq fields of the output are modified.
 * @return Size of the lzfse_compressed_block_header_v2 */
static inline size_t
lzfse_encode_v1_freq_table(lzfse_compressed_block_header_v2 *out,
                           const lzfse_compressed_block_header_v1 *in) {
  uint32_t accum = 0;
  int accum_nbits = 0;
  const uint16_t *src = &(in->l_freq[0]); // first value of first table (struct
                                          // will not be modified, so this code
                                          // will remain valid)
  uint8_t *dst = &(out->freq[0]);
  for (int i = 0; i < LZFSE_ENCODE_L_SYMBOLS + LZFSE_ENCODE_M_SYMBOLS +
                          LZFSE_ENCODE_D_SYMBOLS + LZFSE_ENCODE_LITERAL_SYMBOLS;
       i++) {
    // Encode one value to accum
    int nbits = 0;
    uint32_t bits = lzfse_encode_v1_freq_value(src[i], &nbits);
    assert(bits < (1 << nbits));
    accum |= bits << accum_nbits;
    accum_nbits += nbits;

    // Store bytes from accum to output buffer
    while (accum_nbits >= 8) {
      *dst = (uint8_t)(accum & 0xff);
      accum >>= 8;
      accum_nbits -= 8;
      dst++;
    }
  }
  // Store final byte if needed
  if (accum_nbits > 0) {
    *dst = (uint8_t)(accum & 0xff);
    dst++;
  }

  // Return final size of out
  uint32_t header_size = (uint32_t)(dst - (uint8_t *)out);
  out->packed_fields[0] = 0;
  out->packed_fields[1] = 0;
  out->packed_fields[2] = setField(header_size, 0, 32);

  return header_size;
}

// We need to limit forward match length to make sure it won't split into a too
// large number of LMD.
// The limit itself is quite large, so it doesn't really impact compression
// ratio.
// The matches may still be expanded backwards by a few bytes, so the final
// length may be greater than this limit, which is OK.
#define LZFSE_ENCODE_MAX_MATCH_LENGTH (100 * LZFSE_ENCODE_MAX_M_VALUE)

// ===============================================================
// Encoder back end

/*! @abstract Encode matches stored in STATE into a compressed/uncompressed block.
 * @return LZFSE_STATUS_OK on success.
 * @return LZFSE_STATUS_DST_FULL and restore initial state if output buffer is
 * full. */
static int lzfse_encode_matches(lzfse_encoder_state *s) {
  if (s->n_literals == 0 && s->n_matches == 0)
    return LZFSE_STATUS_OK; // nothing to store, OK

  uint32_t l_occ[LZFSE_ENCODE_L_SYMBOLS];
  uint32_t m_occ[LZFSE_ENCODE_M_SYMBOLS];
  uint32_t d_occ[LZFSE_ENCODE_D_SYMBOLS];
  uint32_t literal_occ[LZFSE_ENCODE_LITERAL_SYMBOLS];
  fse_encoder_entry l_encoder[LZFSE_ENCODE_L_SYMBOLS];
  fse_encoder_entry m_encoder[LZFSE_ENCODE_M_SYMBOLS];
  fse_encoder_entry d_encoder[LZFSE_ENCODE_D_SYMBOLS];
  fse_encoder_entry literal_encoder[LZFSE_ENCODE_LITERAL_SYMBOLS];
  int ok = 1;
  lzfse_compressed_block_header_v1 header1 = {0};
  lzfse_compressed_block_header_v2 *header2 = 0;

  // Keep initial state to be able to restore it if DST full
  uint8_t *dst0 = s->dst;
  uint32_t n_literals0 = s->n_literals;

  // Add 0x00 literals until n_literals multiple of 4, since we encode 4
  // interleaved literal streams.
  while (s->n_literals & 3) {
    uint32_t n = s->n_literals++;
    s->literals[n] = 0;
  }

  // Encode previous distance
  uint32_t d_prev = 0;
  for (uint32_t i = 0; i < s->n_matches; i++) {
    uint32_t d = s->d_values[i];
    if (d == d_prev)
      s->d_values[i] = 0;
    else
      d_prev = d;
  }

  // Clear occurrence tables
  memset(l_occ, 0, sizeof(l_occ));
  memset(m_occ, 0, sizeof(m_occ));
  memset(d_occ, 0, sizeof(d_occ));
  memset(literal_occ, 0, sizeof(literal_occ));

  // Update occurrence tables in all 4 streams (L,M,D,literals)
  uint32_t l_sum = 0;
  uint32_t m_sum = 0;
  for (uint32_t i = 0; i < s->n_matches; i++) {
    uint32_t l = s->l_values[i];
    l_sum += l;
    l_occ[l_base_from_value(l)]++;
  }
  for (uint32_t i = 0; i < s->n_matches; i++) {
    uint32_t m = s->m_values[i];
    m_sum += m;
    m_occ[m_base_from_value(m)]++;
  }
  for (uint32_t i = 0; i < s->n_matches; i++)
    d_occ[d_base_from_value(s->d_values[i])]++;
  for (uint32_t i = 0; i < s->n_literals; i++)
    literal_occ[s->literals[i]]++;

  // Make sure we have enough room for a _full_ V2 header
  if (s->dst + sizeof(lzfse_compressed_block_header_v2) > s->dst_end) {
    ok = 0;
    goto END;
  }
  header2 = (lzfse_compressed_block_header_v2 *)(s->dst);

  // Setup header V1
  header1.magic = LZFSE_COMPRESSEDV1_BLOCK_MAGIC;
  header1.n_raw_bytes = m_sum + l_sum;
  header1.n_matches = s->n_matches;
  header1.n_literals = s->n_literals;

  // Normalize occurrence tables to freq tables
  fse_normalize_freq(LZFSE_ENCODE_L_STATES, LZFSE_ENCODE_L_SYMBOLS, l_occ,
                     header1.l_freq);
  fse_normalize_freq(LZFSE_ENCODE_M_STATES, LZFSE_ENCODE_M_SYMBOLS, m_occ,
                     header1.m_freq);
  fse_normalize_freq(LZFSE_ENCODE_D_STATES, LZFSE_ENCODE_D_SYMBOLS, d_occ,
                     header1.d_freq);
  fse_normalize_freq(LZFSE_ENCODE_LITERAL_STATES, LZFSE_ENCODE_LITERAL_SYMBOLS,
                     literal_occ, header1.literal_freq);

  // Compress freq tables to V2 header, and get actual size of V2 header
  s->dst += lzfse_encode_v1_freq_table(header2, &header1);

  // Initialize encoder tables from freq tables
  fse_init_encoder_table(LZFSE_ENCODE_L_STATES, LZFSE_ENCODE_L_SYMBOLS,
                         header1.l_freq, l_encoder);
  fse_init_encoder_table(LZFSE_ENCODE_M_STATES, LZFSE_ENCODE_M_SYMBOLS,
                         header1.m_freq, m_encoder);
  fse_init_encoder_table(LZFSE_ENCODE_D_STATES, LZFSE_ENCODE_D_SYMBOLS,
                         header1.d_freq, d_encoder);
  fse_init_encoder_table(LZFSE_ENCODE_LITERAL_STATES,
                         LZFSE_ENCODE_LITERAL_SYMBOLS, header1.literal_freq,
                         literal_encoder);

  // Encode literals
  {
    fse_out_stream out;
    fse_out_init(&out);
    fse_state state0, state1, state2, state3;
    state0 = state1 = state2 = state3 = 0;

    uint8_t *buf = s->dst;
    uint32_t i = s->n_literals; // I multiple of 4
    // We encode starting from the last literal so we can decode starting from
    // the first
    while (i > 0) {
      if (buf + 16 > s->dst_end) {
        ok = 0;
        goto END;
      } // out full
      i -= 4;
      fse_encode(&state3, literal_encoder, &out, s->literals[i + 3]); // 10b
      fse_encode(&state2, literal_encoder, &out, s->literals[i + 2]); // 10b
#if !FSE_IOSTREAM_64
      fse_out_flush(&out, &buf);
#endif
      fse_encode(&state1, literal_encoder, &out, s->literals[i + 1]); // 10b
      fse_encode(&state0, literal_encoder, &out, s->literals[i + 0]); // 10b
      fse_out_flush(&out, &buf);
    }
    fse_out_finish(&out, &buf);

    // Update header with final encoder state
    header1.literal_bits = out.accum_nbits; // [-7, 0]
    header1.n_literal_payload_bytes = (uint32_t)(buf - s->dst);
    header1.literal_state[0] = state0;
    header1.literal_state[1] = state1;
    header1.literal_state[2] = state2;
    header1.literal_state[3] = state3;

    // Update state
    s->dst = buf;

  } // literals

  // Encode L,M,D
  {
    fse_out_stream out;
    fse_out_init(&out);
    fse_state l_state, m_state, d_state;
    l_state = m_state = d_state = 0;

    uint8_t *buf = s->dst;
    uint32_t i = s->n_matches;

    // Add 8 padding bytes to the L,M,D payload
    if (buf + 8 > s->dst_end) {
      ok = 0;
      goto END;
    } // out full
    store8(buf, 0);
    buf += 8;

    // We encode starting from the last match so we can decode starting from the
    // first
    while (i > 0) {
      if (buf + 16 > s->dst_end) {
        ok = 0;
        goto END;
      } // out full
      i -= 1;

      // D requires 23b max
      int32_t d_value = s->d_values[i];
      uint8_t d_symbol = d_base_from_value(d_value);
      int32_t d_nbits = d_extra_bits[d_symbol];
      int32_t d_bits = d_value - d_base_value[d_symbol];
      fse_out_push(&out, d_nbits, d_bits);
      fse_encode(&d_state, d_encoder, &out, d_symbol);
#if !FSE_IOSTREAM_64
      fse_out_flush(&out, &buf);
#endif

      // M requires 17b max
      int32_t m_value = s->m_values[i];
      uint8_t m_symbol = m_base_from_value(m_value);
      int32_t m_nbits = m_extra_bits[m_symbol];
      int32_t m_bits = m_value - m_base_value[m_symbol];
      fse_out_push(&out, m_nbits, m_bits);
      fse_encode(&m_state, m_encoder, &out, m_symbol);
#if !FSE_IOSTREAM_64
      fse_out_flush(&out, &buf);
#endif

      // L requires 14b max
      int32_t l_value = s->l_values[i];
      uint8_t l_symbol = l_base_from_value(l_value);
      int32_t l_nbits = l_extra_bits[l_symbol];
      int32_t l_bits = l_value - l_base_value[l_symbol];
      fse_out_push(&out, l_nbits, l_bits);
      fse_encode(&l_state, l_encoder, &out, l_symbol);
      fse_out_flush(&out, &buf);
    }
    fse_out_finish(&out, &buf);

    // Update header with final encoder state
    header1.n_lmd_payload_bytes = (uint32_t)(buf - s->dst);
    header1.lmd_bits = out.accum_nbits; // [-7, 0]
    header1.l_state = l_state;
    header1.m_state = m_state;
    header1.d_state = d_state;

    // Update state
    s->dst = buf;

  } // L,M,D

  // Final state update, here we had enough space in DST, and are not going to
  // revert state
  s->n_literals = 0;
  s->n_matches = 0;

  // Final payload size
  header1.n_payload_bytes =
      header1.n_literal_payload_bytes + header1.n_lmd_payload_bytes;

  // Encode state info in V2 header (we previously encoded the tables, now we
  // set the other fields)
  lzfse_encode_v1_state(header2, &header1);

END:
  if (!ok) {
    // Revert state, DST was full

    // Revert the d_prev encoding
    uint32_t d_prev = 0;
    for (uint32_t i = 0; i < s->n_matches; i++) {
      uint32_t d = s->d_values[i];
      if (d == 0)
        s->d_values[i] = d_prev;
      else
        d_prev = d;
    }

    // Revert literal count
    s->n_literals = n_literals0;

    // Revert DST
    s->dst = dst0;

    return LZFSE_STATUS_DST_FULL; // DST full
  }

  return LZFSE_STATUS_OK;
}

/*! @abstract Push a L,M,D match into the STATE.
 * @return LZFSE_STATUS_OK if OK.
 * @return LZFSE_STATUS_DST_FULL if the match can't be pushed, meaning one of
 * the buffers is full. In that case the state is not modified. */
static inline int lzfse_push_lmd(lzfse_encoder_state *s, uint32_t L,
                                 uint32_t M, uint32_t D) {
  // Check if we have enough space to push the match (we add some margin to copy
  // literals faster here, and round final count later)
  if (s->n_matches + 1 + 8 > LZFSE_MATCHES_PER_BLOCK)
    return LZFSE_STATUS_DST_FULL; // state full
  if (s->n_literals + L + 16 > LZFSE_LITERALS_PER_BLOCK)
    return LZFSE_STATUS_DST_FULL; // state full

  // Store match
  uint32_t n = s->n_matches++;
  s->l_values[n] = L;
  s->m_values[n] = M;
  s->d_values[n] = D;

  // Store literals
  uint8_t *dst = s->literals + s->n_literals;
  const uint8_t *src = s->src + s->src_literal;
  uint8_t *dst_end = dst + L;
  if (s->src_literal + L + 16 > s->src_end) {
    // Careful at the end of SRC, we can't read 16 bytes
    if (L > 0)
      memcpy(dst, src, L);
  } else {
    copy16(dst, src);
    dst += 16;
    src += 16;
    while (dst < dst_end) {
      copy16(dst, src);
      dst += 16;
      src += 16;
    }
  }
  s->n_literals += L;

  // Update state
  s->src_literal += L + M;

  return LZFSE_STATUS_OK;
}

/*! @abstract Split MATCH into one or more L,M,D parts, and push to STATE.
 * @return LZFSE_STATUS_OK if OK.
 * @return LZFSE_STATUS_DST_FULL if the match can't be pushed, meaning one of the
 * buffers is full. In that case the state is not modified. */
static int lzfse_push_match(lzfse_encoder_state *s, const lzfse_match *match) {
  // Save the initial n_matches, n_literals, src_literal
  uint32_t n_matches0 = s->n_matches;
  uint32_t n_literals0 = s->n_literals;
  lzfse_offset src_literals0 = s->src_literal;

  // L,M,D
  uint32_t L = (uint32_t)(match->pos - s->src_literal); // literal count
  uint32_t M = match->length;                           // match length
  uint32_t D = (uint32_t)(match->pos - match->ref);     // match distance
  int ok = 1;

  // Split L if too large
  while (L > LZFSE_ENCODE_MAX_L_VALUE) {
    if (lzfse_push_lmd(s, LZFSE_ENCODE_MAX_L_VALUE, 0, 1) != 0) {
      ok = 0;
      goto END;
    } // take D=1 because most frequent, but not actually used
    L -= LZFSE_ENCODE_MAX_L_VALUE;
  }

  // Split if M too large
  while (M > LZFSE_ENCODE_MAX_M_VALUE) {
    if (lzfse_push_lmd(s, L, LZFSE_ENCODE_MAX_M_VALUE, D) != 0) {
      ok = 0;
      goto END;
    }
    L = 0;
    M -= LZFSE_ENCODE_MAX_M_VALUE;
  }

  // L,M in range
  if (L > 0 || M > 0) {
    if (lzfse_push_lmd(s, L, M, D) != 0) {
      ok = 0;
      goto END;
    }
    L = M = 0;
    (void)L;
    (void)M; // dead stores
  }

END:
  if (!ok) {
    // Revert state
    s->n_matches = n_matches0;
    s->n_literals = n_literals0;
    s->src_literal = src_literals0;

    return LZFSE_STATUS_DST_FULL; // state tables full
  }

  return LZFSE_STATUS_OK; // OK
}

/*! @abstract Backend: add MATCH to state S. Encode block if necessary, when
 * state is full.
 * @return LZFSE_STATUS_OK if OK.
 * @return LZFSE_STATUS_DST_FULL if the match can't be added, meaning one of the
 * buffers is full. In that case the state is not modified. */
static int lzfse_backend_match(lzfse_encoder_state *s,
                               const lzfse_match *match) {
  // Try to push the match in state
  if (lzfse_push_match(s, match) == LZFSE_STATUS_OK)
    return LZFSE_STATUS_OK; // OK, match added to state

  // Here state tables are full, try to emit block
  if (lzfse_encode_matches(s) != LZFSE_STATUS_OK)
    return LZFSE_STATUS_DST_FULL; // DST full, match not added

  // Here block has been emitted, re-try to push the match in state
  return lzfse_push_match(s, match);
}

/*! @abstract Backend: add L literals to state S. Encode block if necessary,
 * when state is full.
 * @return LZFSE_STATUS_OK if OK.
 * @return LZFSE_STATUS_DST_FULL if the literals can't be added, meaning one of
 * the buffers is full. In that case the state is not modified. */
static int lzfse_backend_literals(lzfse_encoder_state *s, lzfse_offset L) {
  // Create a fake match with M=0, D=1
  lzfse_match match;
  lzfse_offset pos = s->src_literal + L;
  match.pos = pos;
  match.ref = match.pos - 1;
  match.length = 0;
  return lzfse_backend_match(s, &match);
}

/*! @abstract Backend: flush final block, and emit end of stream
 * @return LZFSE_STATUS_OK if OK.
 * @return LZFSE_STATUS_DST_FULL if either the final block, or the end-of-stream
 * can't be added, meaning one of the buffers is full. If the block was emitted,
 * the state is updated to reflect this. Otherwise, it is left unchanged. */
static int lzfse_backend_end_of_stream(lzfse_encoder_state *s) {
  // Final match triggers write, otherwise emit blocks when we have enough
  // matches stored
  if (lzfse_encode_matches(s) != LZFSE_STATUS_OK)
    return LZFSE_STATUS_DST_FULL; // DST full

  // Emit end-of-stream block
  if (s->dst + 4 > s->dst_end)
    return LZFSE_STATUS_DST_FULL; // DST full
  store4(s->dst, LZFSE_ENDOFSTREAM_BLOCK_MAGIC);
  s->dst += 4;

  return LZFSE_STATUS_OK; // OK
}

// ===============================================================
// Encoder state management

/*! @abstract Initialize state:
 * @code
 * - hash table with all invalid pos, and value 0.
 * - pending match to NO_MATCH.
 * - src_literal to 0.
 * - d_prev to 0.
 @endcode
 * @return LZFSE_STATUS_OK */
int lzfse_encode_init(lzfse_encoder_state *s) {
  const lzfse_match NO_MATCH = {0};
  lzfse_history_set line;
  for (int i = 0; i < LZFSE_ENCODE_HASH_WIDTH; i++) {
    line.pos[i] = -4 * LZFSE_ENCODE_MAX_D_VALUE; // invalid pos
    line.value[i] = 0;
  }
  // Fill table
  for (int i = 0; i < LZFSE_ENCODE_HASH_VALUES; i++)
    s->history_table[i] = line;
  s->pending = NO_MATCH;
  s->src_literal = 0;

  return LZFSE_STATUS_OK; // OK
}

/*! @abstract Translate state \p src forward by \p delta > 0.
 * Offsets in \p src are updated backwards to point to the same positions.
 * @return  LZFSE_STATUS_OK */
int lzfse_encode_translate(lzfse_encoder_state *s, lzfse_offset delta) {
  assert(delta >= 0);
  if (delta == 0)
    return LZFSE_STATUS_OK; // OK

  // SRC
  s->src += delta;

  // Offsets in SRC
  s->src_end -= delta;
  s->src_encode_i -= delta;
  s->src_encode_end -= delta;
  s->src_literal -= delta;

  // Pending match
  s->pending.pos -= delta;
  s->pending.ref -= delta;

  // history_table positions, translated, and clamped to invalid pos
  int32_t invalidPos = -4 * LZFSE_ENCODE_MAX_D_VALUE;
  for (int i = 0; i < LZFSE_ENCODE_HASH_VALUES; i++) {
    int32_t *p = &(s->history_table[i].pos[0]);
    for (int j = 0; j < LZFSE_ENCODE_HASH_WIDTH; j++) {
      lzfse_offset newPos = p[j] - delta; // translate
      p[j] = (int32_t)((newPos < invalidPos) ? invalidPos : newPos); // clamp
    }
  }

  return LZFSE_STATUS_OK; // OK
}

// ===============================================================
// Encoder front end

int lzfse_encode_base(lzfse_encoder_state *s) {
  lzfse_history_set *history_table = s->history_table;
  lzfse_history_set *hashLine = 0;
  lzfse_history_set newH;
  const lzfse_match NO_MATCH = {0};
  int ok = 1;

  memset(&newH, 0x00, sizeof(newH));

  // 8 byte padding at end of buffer
  s->src_encode_end = s->src_end - 8;
  for (; s->src_encode_i < s->src_encode_end; s->src_encode_i++) {
    lzfse_offset pos = s->src_encode_i; // pos >= 0

    // Load 4 byte value and get hash line
    uint32_t x = load4(s->src + pos);
    hashLine = history_table + hashX(x);
    lzfse_history_set h = *hashLine;

    // Prepare next hash line (component 0 is the most recent) to prepare new
    // entries (stored later)
    {
      newH.pos[0] = (int32_t)pos;
      for (int k = 0; k < LZFSE_ENCODE_HASH_WIDTH - 1; k++)
        newH.pos[k + 1] = h.pos[k];
      newH.value[0] = x;
      for (int k = 0; k < LZFSE_ENCODE_HASH_WIDTH - 1; k++)
        newH.value[k + 1] = h.value[k];
    }

    // Do not look for a match if we are still covered by a previous match
    if (pos < s->src_literal)
      goto END_POS;

    // Search best incoming match
    lzfse_match incoming = {.pos = pos, .ref = 0, .length = 0};

    // Check for matches.  We consider matches of length >= 4 only.
    for (int k = 0; k < LZFSE_ENCODE_HASH_WIDTH; k++) {
      uint32_t d = h.value[k] ^ x;
      if (d)
        continue; // no 4 byte match
      int32_t ref = h.pos[k];
      if (ref + LZFSE_ENCODE_MAX_D_VALUE < pos)
        continue; // too far

      const uint8_t *src_ref = s->src + ref;
      const uint8_t *src_pos = s->src + pos;
      uint32_t length = 4;
      uint32_t maxLength =
        (uint32_t)(s->src_end - pos - 8); // ensure we don't hit the end of SRC
      while (length < maxLength) {
        uint64_t d = load8(src_ref + length) ^ load8(src_pos + length);
        if (d == 0) {
          length += 8;
          continue;
        }

        length +=
            (__builtin_ctzll(d) >> 3); // ctzll must be called only with D != 0
        break;
      }
      if (length > incoming.length) {
        incoming.length = length;
        incoming.ref = ref;
      } // keep if longer
    }

    // No incoming match?
    if (incoming.length == 0) {
      // We may still want to emit some literals here, to not lag too far behind
      // the current search point, and avoid
      // ending up with a literal block not fitting in the state.
      lzfse_offset n_literals = pos - s->src_literal;
      // The threshold here should be larger than a couple of MAX_L_VALUE, and
      // much smaller than LITERALS_PER_BLOCK
      if (n_literals > 8 * LZFSE_ENCODE_MAX_L_VALUE) {
        // Here, we need to consume some literals. Emit pending match if there
        // is one
        if (s->pending.length > 0) {
          if (lzfse_backend_match(s, &s->pending) != LZFSE_STATUS_OK) {
            ok = 0;
            goto END;
          }
          s->pending = NO_MATCH;
        } else {
          // No pending match, emit a full LZFSE_ENCODE_MAX_L_VALUE block of
          // literals
          if (lzfse_backend_literals(s, LZFSE_ENCODE_MAX_L_VALUE) !=
              LZFSE_STATUS_OK) {
            ok = 0;
            goto END;
          }
        }
      }
      goto END_POS; // no incoming match
    }

    // Limit match length (it may still be expanded backwards, but this is
    // bounded by the limit on literals we tested before)
    if (incoming.length > LZFSE_ENCODE_MAX_MATCH_LENGTH) {
      incoming.length = LZFSE_ENCODE_MAX_MATCH_LENGTH;
    }

    // Expand backwards (since this is expensive, we do this for the best match
    // only)
    while (incoming.pos > s->src_literal && incoming.ref > 0 &&
           s->src[incoming.ref - 1] == s->src[incoming.pos - 1]) {
      incoming.pos--;
      incoming.ref--;
    }
    incoming.length += pos - incoming.pos; // update length after expansion

    // Match filtering heuristic (from LZVN). INCOMING is always defined here.

    // Incoming is 'good', emit incoming
    if (incoming.length >= LZFSE_ENCODE_GOOD_MATCH) {
      if (lzfse_backend_match(s, &incoming) != LZFSE_STATUS_OK) {
        ok = 0;
        goto END;
      }
      s->pending = NO_MATCH;
      goto END_POS;
    }

    // No pending, keep incoming
    if (s->pending.length == 0) {
      s->pending = incoming;
      goto END_POS;
    }

    // No overlap, emit pending, keep incoming
    if (s->pending.pos + s->pending.length <= incoming.pos) {
      if (lzfse_backend_match(s, &s->pending) != LZFSE_STATUS_OK) {
        ok = 0;
        goto END;
      }
      s->pending = incoming;
      goto END_POS;
    }

    // Overlap: emit longest
    if (incoming.length > s->pending.length) {
      if (lzfse_backend_match(s, &incoming) != LZFSE_STATUS_OK) {
        ok = 0;
        goto END;
      }
    } else {
      if (lzfse_backend_match(s, &s->pending) != LZFSE_STATUS_OK) {
        ok = 0;
        goto END;
      }
    }
    s->pending = NO_MATCH;

  END_POS:
    // We are done with this src_encode_i.
    // Update state now (s->pending has already been updated).
    *hashLine = newH;
  }

END:
  return ok ? LZFSE_STATUS_OK : LZFSE_STATUS_DST_FULL;
}

int lzfse_encode_finish(lzfse_encoder_state *s) {
  const lzfse_match NO_MATCH = {0};

  // Emit pending match
  if (s->pending.length > 0) {
    if (lzfse_backend_match(s, &s->pending) != LZFSE_STATUS_OK)
      return LZFSE_STATUS_DST_FULL;
    s->pending = NO_MATCH;
  }

  // Emit final literals if any
  lzfse_offset L = s->src_end - s->src_literal;
  if (L > 0) {
    if (lzfse_backend_literals(s, L) != LZFSE_STATUS_OK)
      return LZFSE_STATUS_DST_FULL;
  }

  // Emit all matches, and end-of-stream block
  if (lzfse_backend_end_of_stream(s) != LZFSE_STATUS_OK)
    return LZFSE_STATUS_DST_FULL;

  return LZFSE_STATUS_OK;
}
